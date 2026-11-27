"""Pack verified 48K next-gen tokenizers into immutable Seed-17 DGX document-proxy caches."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe.dgx_handoff import file_digest
from experiments.mom_bpe.diagnose import format_json, immutable, sha
from experiments.mom_bpe.frontend_analysis import token_payloads
from experiments.mom_bpe.proxy_blocks import BOUNDARY, MODES, POLICY, EncodedSplit, member, publish, receipt
from experiments.mom_bpe.scale_data import code_digest, document_bytes, verified_plan
from experiments.mom_bpe.scale_fit import VOCAB
from experiments.mom_bpe.scale_proxy_plan import parameter_counts
from experiments.mom_bpe.verify_frozen import load_rows
from mltok import registry
from mltok.lib.atomic import atomic_writer
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import canonical
from mltok.lib.mingram_core import ARTIFACT as MINGRAM_ARTIFACT, MinGramBackend, UniformPrefixMinGramBackend
from mltok.lib.picky_backend import ARTIFACT as PICKY_V1_ARTIFACT, EventBackend
from mltok.lib.uniform_prefix import ARTIFACT as PICKY_V2_ARTIFACT, ESC, SP, UniformPrefixEventBackend
from mltok.snapshots import _lock, file_identity

HERE = Path(__file__).resolve().parent
NEXTGEN_ARMS = ("T8-LS-9999", "M48-Guard-9999", "ML-MinGram-WP-G1000")
DEFAULT_BASELINE_INPUTS = HERE / "t5_t8_inputs_v1.json"
DEFAULT_MODELS_DIR = Path(r"C:\DUK\MTech Project\nextgen-runs\scale48k-v2\cov9999")
DEFAULT_CORPUS_DIR = Path(r"C:\DUK\MTech Project\Tokenizer-pipeline-git\experiments\mom_bpe\runs\scale48k-v1\a-100mb")
BYTE_RE = re.compile(r"<0x([0-9A-F]{2})>")
MODEL_CANDIDATES = (
    ("picky_v2", PICKY_V2_ARTIFACT),
    ("ml_mingram", MINGRAM_ARTIFACT),
    ("picky_v1", PICKY_V1_ARTIFACT),
)
EXTRA_MODEL_FILES = ("eval_summary.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


@contextlib.contextmanager
def with_nextgen_arms(driver_module, proxy_module=None, extra_arms=NEXTGEN_ARMS):
    """Temporarily allow next-gen arm names in unmodified proxy_blocks / document_proxy modules."""
    old_driver = driver_module.ARMS
    old_proxy = getattr(proxy_module, "ARMS", None) if proxy_module is not None else None
    merged = tuple(dict.fromkeys((*old_driver, *extra_arms)))
    driver_module.ARMS = merged
    if proxy_module is not None:
        proxy_module.ARMS = merged
    try:
        yield
    finally:
        driver_module.ARMS = old_driver
        if proxy_module is not None and old_proxy is not None:
            proxy_module.ARMS = old_proxy


def detect_model_artifact(arm_dir: Path) -> tuple[str, Path]:
    """Locate the single authoritative tokenizer artifact in `arm_dir`."""
    found = [(kind, arm_dir / filename) for kind, filename in MODEL_CANDIDATES if (arm_dir / filename).is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one model artifact in {arm_dir}, found {[p.name for _, p in found]}")
    return found[0]


def load_nextgen_backend(arm_dir: Path, arm_name: str, normalizer, *, require_full_vocab: bool = True):
    """Load and verify a next-gen backend and its canonical artifact digest."""
    kind, model_path = detect_model_artifact(arm_dir)
    if kind == "picky_v2":
        backend = UniformPrefixEventBackend.load(arm_dir)
        uniform_prefix = True
    elif kind == "ml_mingram":
        backend = MinGramBackend.load(arm_dir)
        uniform_prefix = isinstance(backend, UniformPrefixMinGramBackend)
    else:
        backend = EventBackend.load(arm_dir)
        uniform_prefix = False

    model_sha = file_identity(model_path)[1]
    specials = dict(zip(DEFAULT_SPECIALS, range(len(DEFAULT_SPECIALS))))
    if backend.special_ids != specials:
        raise ValueError(f"{arm_name} control IDs differ from DEFAULT_SPECIALS")
    if backend.fallback_ids != frozenset(range(len(DEFAULT_SPECIALS), len(DEFAULT_SPECIALS) + 256)):
        raise ValueError(f"{arm_name} byte-fallback IDs differ from 256-byte range")
    if backend.vocab_size != VOCAB:
        raise ValueError(f"{arm_name} vocab_size ({backend.vocab_size}) differs from {VOCAB}")
    if require_full_vocab and len(backend._pieces) != VOCAB:
        raise ValueError(f"{arm_name} occupied {len(backend._pieces)} of {VOCAB} slots")
    for probe in ("", "കേരളം India", "▁മലയാളം ▂", "ൻ്റ ൎക്കു് ക്\u200dഷ"):
        if backend.expected(probe) != normalizer.normalize(probe):
            raise ValueError(f"{arm_name} normalizer mismatch on probe {probe!r}")
    return backend, model_path.name, model_sha, uniform_prefix


def _transport_byte_map(canonical_text: str, data: bytes) -> tuple[bytes, list[int]]:
    """Map every UTF-8 byte of `encode_transport(canonical_text)` to canonical UTF-8 byte offsets."""
    t_parts: list[bytes] = [SP.encode("utf-8")]
    tbyte_to_cbyte: list[int] = [0, 0, 0]
    c_pos = 0
    for ch in canonical_text:
        ch_bytes = ch.encode("utf-8")
        if ch == ESC:
            t_ch = (ESC + ESC).encode("utf-8")
            t_parts.append(t_ch)
            tbyte_to_cbyte.extend([c_pos] * (len(t_ch) - 1) + [c_pos + len(ch_bytes)])
        elif ch == SP:
            t_ch = (ESC + SP).encode("utf-8")
            t_parts.append(t_ch)
            tbyte_to_cbyte.extend([c_pos] * (len(t_ch) - 1) + [c_pos + len(ch_bytes)])
        elif ch == " ":
            t_ch = SP.encode("utf-8")
            t_parts.append(t_ch)
            tbyte_to_cbyte.extend([c_pos] * (len(t_ch) - 1) + [c_pos + len(ch_bytes)])
        else:
            t_parts.append(ch_bytes)
            tbyte_to_cbyte.extend(range(c_pos + 1, c_pos + len(ch_bytes) + 1))
        c_pos += len(ch_bytes)
    if c_pos != len(data):
        raise ValueError("Canonical text UTF-8 byte length mismatch")
    return b"".join(t_parts), tbyte_to_cbyte


def token_byte_ends(encoding, backend, data: bytes, uniform_prefix: bool) -> list[int]:
    """Compute `<u4` canonical byte-end markers with `BOUNDARY` flag for `encoding`."""
    size = len(data)
    if not uniform_prefix:
        ends = []
        for _, end, _, _, _, payload in token_payloads(encoding, backend, native=False):
            if payload != data[end - len(payload):end]:
                raise ValueError("Token payload differs from canonical UTF-8")
            boundary = end == size or (data[end] & 0xC0) != 0x80
            ends.append(end | (BOUNDARY if boundary else 0))
        if size >= BOUNDARY or not size or not ends or ends[-1] != size | BOUNDARY:
            raise ValueError("Token payloads did not cover an eligible document")
        return ends

    expected_transport, tbyte_to_cbyte = _transport_byte_map(encoding.text, data)
    ends = []
    t_offset = 0
    reconstructed = bytearray()
    for identifier, piece in zip(encoding.ids, encoding.tokens, strict=True):
        if identifier in backend.fallback_ids:
            match = BYTE_RE.fullmatch(piece)
            if not match:
                raise ValueError(f"Invalid byte fallback token spelling: {piece!r}")
            t_payload = bytes([int(match[1], 16)])
        else:
            t_payload = piece.encode("utf-8")
        if not t_payload:
            raise ValueError("Empty uniform-prefix token payload")
        reconstructed.extend(t_payload)
        t_offset += len(t_payload)
        if t_offset > len(expected_transport):
            raise ValueError("Uniform-prefix token payloads overflowed expected transport length")
        end = tbyte_to_cbyte[t_offset - 1]
        boundary = end == size or (data[end] & 0xC0) != 0x80
        ends.append(end | (BOUNDARY if boundary else 0))
    if bytes(reconstructed) != expected_transport or t_offset != len(expected_transport):
        raise ValueError("Uniform-prefix transport bytes differ from expected document encoding")
    if size >= BOUNDARY or not size or not ends or ends[-1] != size | BOUNDARY:
        raise ValueError("Uniform-prefix token payloads did not cover an eligible document")
    return ends


def write_nextgen_split(
    cache_dir: Path,
    split: str,
    rows,
    backend,
    normalize,
    uniform_prefix: bool,
    tag: str,
    byte_limit: int | None = None,
) -> str:
    """Encode a split into deterministic `<u2` `ids.bin` + `<u4` `ends.bin` and receipt `{split}.json`."""
    folder = cache_dir / f"{split}-{tag[:12]}"
    folder.mkdir(parents=True, exist_ok=True)
    id_path, end_path = folder / "ids.bin", folder / "ends.bin"
    docs, offset, spent = [], 0, 0
    controls = set(backend.special_ids.values())
    with atomic_writer(id_path, overwrite=False) as ids_file, atomic_writer(end_path, overwrite=False) as ends_file:
        for row in rows:
            size = row["canonical_bytes"]
            if byte_limit is not None and spent + size > byte_limit:
                break
            raw, data = document_bytes(row, normalize)
            enc = backend.encode_offsets(row["text"])
            if (
                enc.text.encode("utf-8") != data
                or backend.decode(enc.ids) != enc.text
                or any(type(i) is not int or not 0 <= i < backend.vocab_size or i in controls for i in enc.ids)
            ):
                raise ValueError(f"{split} doc {row['id']!r} failed canonical encoding or control isolation")
            ends = token_byte_ends(enc, backend, data, uniform_prefix)
            ids_file.write(np.asarray(enc.ids, dtype="<u2").tobytes())
            ends_file.write(np.asarray(ends, dtype="<u4").tobytes())
            docs.append({
                "id": row["id"],
                "canonical_sha256": row["canonical_sha256"],
                "canonical_bytes": size,
                "raw_bytes": len(raw),
                "start": offset,
                "length": len(enc.ids),
            })
            offset += len(enc.ids)
            spent += size
    if not docs:
        raise ValueError(f"Cache selection includes no {split} documents")
    split_payload = {
        "documents": docs,
        "canonical_bytes": spent,
        "ids": member(cache_dir, id_path),
        "ends": member(cache_dir, end_path),
    }
    return publish(cache_dir / f"{split}.json", split_payload)


def _copy_if_absent_verified(src: Path, dst: Path) -> tuple[int, str]:
    expected = file_identity(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        data = src.read_bytes()
        with atomic_writer(dst, overwrite=False) as out:
            out.write(data)
    observed = file_identity(dst)
    if observed != expected:
        raise ValueError(f"Copied file differs from source: {dst}")
    return observed


def pack_nextgen_inputs(
    *,
    models_dir: Path,
    corpus_manifest: Path,
    out_dir: Path,
    arms: tuple[str, ...] = NEXTGEN_ARMS,
    baseline_inputs: Path | None = DEFAULT_BASELINE_INPUTS,
    require_full_vocab: bool = True,
    train_prefix_bytes: int | None = None,
    budget_overrides: dict[str, int] | None = None,
    update_targets: int | None = None,
    checkpoint_updates: int | None = None,
) -> dict:
    """Pack all next-gen fits into binary caches, verify parity & lossless decode, and write manifest."""
    from experiments.mom_bpe import document_proxy as driver
    from experiments.mom_bpe import proxy_blocks

    registry.load_plugins()
    if registry.load_errors():
        raise ValueError(f"Plugin import errors: {registry.load_errors()}")

    normalizer_spec = {"type": "ml_conservative", "terminal_chillu": True}
    normalizer = registry.make("normalizer", normalizer_spec)
    normalizer_fp = registry.fingerprint("normalizer", normalizer_spec)
    files, corpus_manifest_sha = verified_plan(corpus_manifest, normalizer.normalize, normalizer_fp)
    corpus_plan = json.loads(corpus_manifest.read_bytes())
    streams = {split: corpus_plan["splits"][split]["canonical_stream_sha256"] for split in ("train", "validation")}

    cfg_args = SimpleNamespace(seed=None, update_targets=update_targets, microbatch_blocks=None, checkpoint_updates=checkpoint_updates)
    proxy_cfg, config_sha = driver.read_config(cfg_args)
    optimizer_sha = sha(canonical(proxy_cfg["optimizer"]))
    context = proxy_cfg["context"]
    param_total = parameter_counts(proxy_cfg["model"]["n_layer"], proxy_cfg["model"]["d_model"], context)["total"]
    budgets = {mode: (budget_overrides or {}).get(mode, proxy_cfg["budgets"][mode]) for mode in MODES}

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    specials = dict(zip(DEFAULT_SPECIALS, range(len(DEFAULT_SPECIALS))))

    baseline_ref = json.loads(baseline_inputs.read_bytes()) if baseline_inputs and baseline_inputs.is_file() else None
    if baseline_ref is not None and train_prefix_bytes is None:
        if (
            streams["validation"] != baseline_ref["validation"]["canonical_stream_sha256"]
            or corpus_manifest_sha != baseline_ref["corpus_manifest_sha256"]
            or corpus_plan["splits"]["validation"]["canonical_bytes"] != 999950
            or corpus_plan["splits"]["validation"]["rows"] != 186
        ):
            raise ValueError("Scale-A corpus streams or canonical validation bytes do not match frozen T5/T8 baseline")

    arms_manifest = {}
    recorded_paths: list[Path] = []
    shared_val_docs = None

    for arm in arms:
        src_fit_dir = models_dir / arm
        backend, model_file, model_sha, uniform_prefix = load_nextgen_backend(
            src_fit_dir, arm, normalizer, require_full_vocab=require_full_vocab
        )

        # Copy immutable model artifacts into packaged input tree.
        dst_fit_rel = f"scale48k-nextgen-v1/fits/{arm}"
        dst_fit_dir = out_dir / dst_fit_rel
        _copy_if_absent_verified(src_fit_dir / model_file, dst_fit_dir / model_file)
        recorded_paths.append(dst_fit_dir / model_file)
        copied_extras = []
        for extra in EXTRA_MODEL_FILES:
            src_extra = src_fit_dir / extra
            if src_extra.is_file():
                _copy_if_absent_verified(src_extra, dst_fit_dir / extra)
                recorded_paths.append(dst_fit_dir / extra)
                copied_extras.append(extra)

        cache_identity = {
            "schema": 1,
            "arm": arm,
            "manifest_sha256": corpus_manifest_sha,
            "streams": streams,
            "tokenizer_sha256": model_sha,
            "model_file": model_file,
            "normalizer": normalizer_fp,
            "uniform_prefix": uniform_prefix,
            "encoder": backend.artifact["encoder"],
            "context": context,
            "policy": POLICY,
            "specials": specials,
            "vocab": VOCAB,
            "train_prefix_bytes": train_prefix_bytes,
            "cache_code": code_digest(proxy_blocks.__file__),
            "packager_code": code_digest(__file__),
        }
        cache_rel = f"dp/cache/{arm}-{sha(canonical(cache_identity))}"
        cache_dir = out_dir / cache_rel
        cache_dir.mkdir(parents=True, exist_ok=True)
        with _lock(cache_dir / ".prepare.lock"):
            cache_receipt_path = cache_dir / "cache.json"
            if cache_receipt_path.exists():
                prev_cache, cache_sha = receipt(cache_receipt_path)
                if prev_cache["identity"] != cache_identity:
                    raise ValueError(f"Existing cache identity differs for {arm}")
                split_hashes = prev_cache["splits"]
            else:
                split_hashes = {}
                for split in ("train", "validation"):
                    split_receipt = cache_dir / f"{split}.json"
                    if split_receipt.exists():
                        _, split_sha = receipt(split_receipt)
                    else:
                        limit = train_prefix_bytes if split == "train" else None
                        split_sha = write_nextgen_split(
                            cache_dir,
                            split,
                            load_rows(files[split][0]),
                            backend,
                            normalizer.normalize,
                            uniform_prefix,
                            tag=sha(canonical([arm, split, model_sha])),
                            byte_limit=limit,
                        )
                    split_obj = EncodedSplit(cache_dir.resolve(), split, split_sha)
                    if split == "validation" or train_prefix_bytes is None:
                        exp = corpus_plan["splits"][split]
                        if (split_obj.canonical_bytes, len(split_obj.docs)) != (exp["canonical_bytes"], exp["rows"]):
                            raise ValueError(f"{arm} {split} cache byte/doc count differs from corpus manifest")
                    split_hashes[split] = split_sha
                cache_sha = publish(cache_receipt_path, {"identity": cache_identity, "splits": split_hashes})

        # Verify 100% lossless roundtrip from cached validation token IDs.
        val_rows = list(load_rows(files["validation"][0]))
        val_split = EncodedSplit(cache_dir.resolve(), "validation", split_hashes["validation"])
        if len(val_split.docs) != len(val_rows):
            raise ValueError(f"{arm} validation cached doc count mismatch")
        for row, cached_doc in zip(val_rows, val_split.docs, strict=True):
            canon = normalizer.normalize(row["text"])
            tok_ids = val_split.ids[cached_doc["start"]:cached_doc["start"] + cached_doc["length"]].tolist()
            if backend.decode(tok_ids) != canon:
                raise ValueError(f"{arm} validation roundtrip decode failed on {row['id']}")

        val_keys = [(d["id"], d["canonical_sha256"], d["canonical_bytes"]) for d in val_split.docs]
        if shared_val_docs is None:
            shared_val_docs = val_keys
        elif val_keys != shared_val_docs:
            raise ValueError(f"{arm} validation document stream ordering differs")

        recorded_paths.append(cache_receipt_path)
        for split in ("train", "validation"):
            split_json_path = cache_dir / f"{split}.json"
            split_data, _ = receipt(split_json_path)
            recorded_paths.extend([
                split_json_path,
                cache_dir / split_data["ids"]["path"],
                cache_dir / split_data["ends"]["path"],
            ])

        # Plan & publish immutable exposure ledgers for fixed_source_bytes and fixed_training_tokens.
        ledgers_info = {}
        with with_nextgen_arms(driver, proxy_blocks):
            for mode in MODES:
                ledger_rel = f"dp/ledgers/{arm}-{mode}.json"
                ledger_path = out_dir / ledger_rel
                plan_obj = driver.fit_cache(SimpleNamespace(
                    command="plan",
                    cache=cache_dir,
                    mode=mode,
                    budget=budgets[mode],
                    out=None,
                    seed=None,
                    update_targets=update_targets,
                    microbatch_blocks=None,
                    checkpoint_updates=checkpoint_updates,
                ))
                if ledger_path.exists():
                    saved_plan, plan_sha = receipt(ledger_path)
                    if saved_plan != plan_obj:
                        raise ValueError(f"Existing ledger conflict for {arm}/{mode}")
                else:
                    ledger_path.parent.mkdir(parents=True, exist_ok=True)
                    plan_sha = publish(ledger_path, plan_obj)
                recorded_paths.append(ledger_path)
                ledgers_info[mode] = {
                    "ledger": ledger_rel,
                    "plan_sha256": plan_sha,
                    "budget": budgets[mode],
                    "forecast": plan_obj["forecast"],
                    "exposure": plan_obj["exposure"],
                }

        if baseline_ref is not None and train_prefix_bytes is None and budgets[MODES[0]] == 50_000_000:
            fs_stream = ledgers_info[MODES[0]]["exposure"]["ordered_complete_documents_sha256"]
            if fs_stream != baseline_ref["fixed_source_stream_sha256"]:
                raise ValueError(f"{arm} fixed_source_bytes selection stream differs from frozen T8")

        arms_manifest[arm] = {
            "arm": arm,
            "cache": cache_rel,
            "cache_receipt_sha256": cache_sha,
            "fit": dst_fit_rel,
            "model_file": model_file,
            "extra_files": copied_extras,
            "tokenizer_sha256": model_sha,
            "uniform_prefix": uniform_prefix,
            "modes": ledgers_info,
            "validation": {
                "canonical_bytes": val_split.canonical_bytes,
                "canonical_stream_sha256": streams["validation"],
                "documents": len(val_split.docs),
                "split_sha256": split_hashes["validation"],
            },
        }

    files_index = {}
    for path in sorted({p.resolve() for p in recorded_paths}):
        rel = path.relative_to(out_dir).as_posix()
        size, digest = file_digest(path)
        files_index[rel] = {"size": size, "sha256": digest}

    first_mode_stream = arms_manifest[arms[0]]["modes"][MODES[0]]["exposure"]["ordered_complete_documents_sha256"]
    manifest = {
        "schema": "mom-nextgen-inputs-v1",
        "arms": arms_manifest,
        "arms_order": list(arms),
        "config_sha256": config_sha,
        "optimizer_sha256": optimizer_sha,
        "context": context,
        "parameters": param_total,
        "corpus_manifest_sha256": corpus_manifest_sha,
        "train_stream_sha256": streams["train"],
        "fixed_source_stream_sha256": first_mode_stream,
        "validation": {
            "canonical_bytes": corpus_plan["splits"]["validation"]["canonical_bytes"],
            "canonical_stream_sha256": streams["validation"],
            "documents": corpus_plan["splits"]["validation"]["rows"],
        },
        "t8_baseline_parity_verified": (
            baseline_ref is not None
            and streams["validation"] == baseline_ref["validation"]["canonical_stream_sha256"]
            and first_mode_stream == baseline_ref.get("fixed_source_stream_sha256")
            and streams["train"] == corpus_plan["splits"]["train"]["canonical_stream_sha256"]
        ),
        "files": files_index,
    }
    manifest_bytes = format_json(manifest)
    manifest_path = out_dir / "nextgen-inputs-manifest.json"
    immutable(manifest_path, manifest_bytes)
    manifest_sha = sha(manifest_path.read_bytes())
    immutable(out_dir / "nextgen-inputs-manifest.sha256", (manifest_sha + "\n").encode("utf-8"))
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "file_count": len(files_index),
        "total_bytes": sum(item["size"] for item in files_index.values()),
        "t8_baseline_parity_verified": manifest["t8_baseline_parity_verified"],
        "arms": {
            arm: {mode: info["forecast"] for mode, info in data["modes"].items()}
            for arm, data in arms_manifest.items()
        },
    }


def upload_packed_inputs_hub(out_dir: Path, repo: str, *, commit_message: str = "Upload verified next-gen DGX inputs") -> dict:
    """Upload all files listed in `nextgen-inputs-manifest.json` + manifest to a private HF repo."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from huggingface_hub import CommitOperationAdd, HfApi, get_token

    token = get_token()
    if not token:
        raise ValueError("Authenticate with a Hugging Face write token before uploading inputs")
    if not repo or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Specify a valid private Hugging Face repository (<owner>/<name>)")
    out_dir = out_dir.resolve()
    manifest_path = out_dir / "nextgen-inputs-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("schema") != "mom-nextgen-inputs-v1":
        raise ValueError("Invalid nextgen-inputs-manifest.json schema")
    for rel, entry in manifest["files"].items():
        if file_digest(out_dir / rel) != (entry["size"], entry["sha256"]):
            raise ValueError(f"Input file verification failed before upload: {rel}")

    api = HfApi(token=token)
    info = api.repo_info(repo, repo_type="model", timeout=20)
    if info.private is not True:
        raise ValueError("Target Hugging Face repository must be private")
    revision = info.sha
    existing = set(api.list_repo_files(repo, repo_type="model", revision=revision))
    to_upload = [*sorted(manifest["files"]), "nextgen-inputs-manifest.json", "nextgen-inputs-manifest.sha256"]
    operations = [
        CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(out_dir / rel))
        for rel in to_upload
        if rel not in existing
    ]
    if operations:
        revision = api.create_commit(
            repo,
            repo_type="model",
            parent_commit=revision,
            operations=operations,
            commit_message=commit_message,
        ).oid
    return {
        "repository": repo,
        "revision": revision,
        "uploaded_files": len(operations),
        "total_manifest_files": len(to_upload),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack")
    pack.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    pack.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_DIR / "manifest.json")
    pack.add_argument("--out-dir", type=Path, required=True)
    pack.add_argument("--baseline-inputs", type=Path, default=DEFAULT_BASELINE_INPUTS)
    pack.add_argument("--no-baseline-check", action="store_true")
    pack.add_argument("--allow-underfilled", action="store_true")
    pack.add_argument("--arms", nargs="+", default=list(NEXTGEN_ARMS))

    upload = sub.add_parser("upload-hub")
    upload.add_argument("--out-dir", type=Path, required=True)
    upload.add_argument("--hub-repo", required=True)
    upload.add_argument("--enable", action="store_true")

    args = parser.parse_args()
    if args.command == "pack":
        report = pack_nextgen_inputs(
            models_dir=args.models_dir,
            corpus_manifest=args.corpus_manifest,
            out_dir=args.out_dir,
            arms=tuple(args.arms),
            baseline_inputs=None if args.no_baseline_check else args.baseline_inputs,
            require_full_vocab=not args.allow_underfilled,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "upload-hub":
        if not args.enable:
            raise ValueError("Hugging Face upload requires --enable")
        print(json.dumps(upload_packed_inputs_hub(args.out_dir, args.hub_repo), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
