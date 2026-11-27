"""Multi-seed DGX replication, aggregation, and zero-retrain held-out evaluator."""

import argparse
import contextlib
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import signal
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import uuid4

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import numpy as np
import torch

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe import dgx_resources as gpu
from experiments.mom_bpe import nextgen_dgx as ng
from experiments.mom_bpe import nextgen_pack as pack
from experiments.mom_bpe.dgx_handoff import file_digest
from experiments.mom_bpe.scale_mlbpe import MatchedMLBPE
from mltok import registry
from mltok.lib.artifact_paths import unlinked_path
from mltok.lib.atomic import atomic_writer
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import canonical
from mltok.lib.canonical_bpb import summarize_bpb
from mltok.lib.mingram_core import ARTIFACT as MINGRAM_ARTIFACT, MinGramBackend
from mltok.lib.picky_backend import ARTIFACT as PICKY_V1_ARTIFACT, EventBackend
from mltok.lib.uniform_prefix import ARTIFACT as PICKY_V2_ARTIFACT, UniformPrefixEventBackend

OPERATIONAL = ng.OPERATIONAL
DEFAULT_RESULTS_REPO = ng.DEFAULT_RESULTS_REPO
NEXTGEN_ARMS = ng.NEXTGEN_ARMS
BASELINE_ARMS = ng.BASELINE_ARMS
ALL_SEVEN_ARMS = ng.ALL_SEVEN_ARMS
MODES = ng.MODES
BOOTSTRAP_SEED = ng.BOOTSTRAP_SEED
BOOTSTRAP_DRAWS = ng.BOOTSTRAP_DRAWS
MULTISEED_DIR = "runs/dp/nextgen-multiseed-v1"
HELDOUT_DIR = "runs/dp/nextgen-heldout-v1"
ALLOWED_HUB_PREFIXES = ("seed_replication/", "heldout/")
require = ng.require


class NativeHeldoutBackend(MatchedMLBPE):
    """Adapter exposing unified `MOMBackend` contract over legacy native `tokenizer.json`."""

    def __init__(self, folder: Path, normalizer):
        super().__init__(folder)
        self.expected = normalizer.normalize
        self.vocab_size = self.occupied

    def encode_offsets(self, text):
        return super().encode_offsets(self.expected(text))


def load_arm_backend(fit_dir: Path, arm: str, normalizer):
    """Load the verified tokenizer backend for any of the 7 benchmark arms."""
    fit_dir = unlinked_path(fit_dir)
    if (fit_dir / PICKY_V2_ARTIFACT).is_file():
        return UniformPrefixEventBackend.load(fit_dir), True
    if (fit_dir / MINGRAM_ARTIFACT).is_file():
        backend = MinGramBackend.load(fit_dir)
        from mltok.lib.mingram_core import UniformPrefixMinGramBackend

        return backend, isinstance(backend, UniformPrefixMinGramBackend)
    if (fit_dir / PICKY_V1_ARTIFACT).is_file():
        return EventBackend.load(fit_dir), False
    if arm == "M48" and (fit_dir / "tokenizer.json").is_file():
        return NativeHeldoutBackend(fit_dir, normalizer), True
    raise ValueError(f"Could not detect tokenizer artifact for {arm} in {fit_dir}")


def resolve_arm_cache(
    root: Path,
    arm: str,
    *,
    inputs_manifest: Path | None = None,
    baseline_inputs: Path | None = None,
) -> Path:
    """Resolve an existing pre-packed binary cache for `arm` without re-packing."""
    require(arm in ALL_SEVEN_ARMS, f"Unsupported arm: {arm}")
    if arm in NEXTGEN_ARMS:
        manifest_path = unlinked_path(inputs_manifest) if inputs_manifest else (unlinked_path(root) / ng.INPUTS_DIR / "nextgen-inputs-manifest.json")
        if manifest_path.is_file():
            manifest, _ = ng.load_inputs_manifest(manifest_path)
            cache_dir = unlinked_path(manifest_path.parent / manifest["arms"][arm]["cache"])
            require((cache_dir / "cache.json").is_file(), f"Missing cache.json at {cache_dir}")
            return cache_dir
        candidates = sorted((unlinked_path(root) / ng.INPUTS_DIR / "dp/cache").glob(f"{arm}-*"))
    else:
        base_manifest = unlinked_path(baseline_inputs) if baseline_inputs else (Path(__file__).with_name("t5_t8_inputs_v1.json"))
        frozen_runs = unlinked_path(root) / "t5-t8-frozen-runs-v1"
        if base_manifest.is_file() and arm in json.loads(base_manifest.read_bytes()).get("arms", {}):
            ref = json.loads(base_manifest.read_bytes())["arms"][arm]["cache"]
            cache_dir = frozen_runs / ref
            if (cache_dir / "cache.json").is_file():
                return cache_dir
        candidates = sorted((frozen_runs / "dp/cache").glob(f"{arm}-*"))
        if not candidates:
            candidates = sorted((unlinked_path(root) / "runs/dp/cache").glob(f"{arm}-*"))
    valid = [c for c in candidates if (c / "cache.json").is_file()]
    require(len(valid) == 1, f"Expected exactly 1 existing cache for {arm}; found {len(valid)}")
    return valid[0]


def verify_seed_list(seeds: list[int]) -> list[int]:
    if not seeds:
        raise ValueError("Provide at least one integer random seed")
    for s in seeds:
        if type(s) is not int or not 0 <= s < 2**32:
            raise ValueError(f"Seed out of 32-bit unsigned range: {s!r}")
    return list(dict.fromkeys(seeds))


def validate_multiseed_result(out: Path, arm: str, mode: str, seed: int, plan: dict, expected_docs: int = 186, *, check_runtime: bool = True) -> dict | None:
    """Validate an already-completed multiseed run without retraining."""
    from experiments.mom_bpe import document_proxy as driver

    result_path = out / "result.json"
    identity_path = out / "identity.json"
    if not result_path.is_file():
        return None
    result = ng.read_receipt(result_path)
    identity = json.loads(identity_path.read_bytes())
    require(
        result["identity"] == identity
        and identity["arm"] == arm
        and identity["mode"] == mode
        and identity["config"]["seed"] == seed
        and identity["parameters"] == 38_799_936
        and identity["forecast"] == plan["forecast"]
        and identity["validation"] == plan["validation"]
        and identity["implementation"] == {
            "driver": gpu.PINNED_DRIVER,
            "blocks": driver.code_digest(driver.blocks.__code__.co_filename),
            "scorer": driver.code_digest(driver.summarize_bpb.__code__.co_filename),
        },
        f"Multiseed result identity conflict for {arm}/{mode}/seed{seed}",
    )
    if check_runtime:
        dev = identity["runtime"].get("gpu")
        require(dev and dev["physical_index"] != 7 and "A100" in dev["type"], f"Invalid GPU runtime record for {arm}/seed{seed}")
    ref = result["checkpoint"]
    ckpt_path = out / ref["path"]
    require(file_digest(ckpt_path) == (ref["size"], ref["sha256"]), f"Corrupt checkpoint at {ckpt_path}")
    require(len(result["evaluation"]["per_document"]) == expected_docs, f"Expected {expected_docs} evaluation docs")
    return result


def run_multiseed(
    *,
    root: Path,
    source: Path,
    arms: list[str],
    seeds: list[int],
    budgets: list[str],
    uuid: str,
    allocated: bool,
    enabled: bool,
    shared: bool = False,
    inputs_manifest: Path | None = None,
    baseline_inputs: Path | None = None,
    expected_docs: int = 186,
    check_runtime: bool = True,
) -> dict:
    """Execute training & Seed-matched scoring for selected `(arm, mode, seed)` tuples."""
    from experiments.mom_bpe import document_proxy as driver
    from experiments.mom_bpe import proxy_blocks
    from mltok.snapshots import _lock

    require(enabled and allocated, "GPU training requires --enable and --allocated")
    arms = [a for a in dict.fromkeys(arms)]
    seeds = verify_seed_list(seeds)
    budgets = [b for b in dict.fromkeys(budgets)]
    require(all(a in ALL_SEVEN_ARMS for a in arms), f"Invalid arm in {arms}")
    require(all(b in MODES for b in budgets), f"Invalid budget mode in {budgets}")

    policy = gpu.GpuAdmission(5120, 10, True) if shared else gpu.GpuAdmission()
    observed = policy.admit(uuid)
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    hardware = driver.gpu_preflight(policy.min_free_mib, max_gpu_util_pct=policy.max_gpu_util_pct, allow_shared_gpu=policy.allow_shared_gpu)
    require("A100" in hardware["type"] and hardware["physical_index"] == observed["physical_index"], "Expected allocated A100")

    root = unlinked_path(root)
    base_dir = unlinked_path(root / MULTISEED_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    completed = {}

    with _lock(base_dir / ".multiseed.lock"):
        for mode in budgets:
            for seed in seeds:
                for arm in arms:
                    cache_dir = resolve_arm_cache(root, arm, inputs_manifest=inputs_manifest, baseline_inputs=baseline_inputs)
                    out = base_dir / f"seed{seed}" / arm / mode
                    out.mkdir(parents=True, exist_ok=True)
                    with pack.with_nextgen_arms(driver, proxy_blocks):
                        plan_obj = driver.fit_cache(SimpleNamespace(
                            command="plan",
                            cache=cache_dir,
                            mode=mode,
                            budget=None,
                            out=None,
                            seed=seed,
                            update_targets=None,
                            microbatch_blocks=None,
                            checkpoint_updates=None,
                        ))
                        existing = validate_multiseed_result(out, arm, mode, seed, plan_obj, expected_docs=expected_docs, check_runtime=check_runtime)
                        if existing is not None:
                            completed[f"{arm}/{mode}/seed{seed}"] = {"status": "reused", "bpb": existing["evaluation"]["bpb"]}
                            continue
                        run_args = SimpleNamespace(
                            command="run",
                            cache=cache_dir,
                            out=out,
                            mode=mode,
                            device="cuda",
                            threads=1,
                            budget=None,
                            seed=seed,
                            update_targets=None,
                            microbatch_blocks=None,
                            checkpoint_updates=None,
                            min_free_mib=policy.min_free_mib,
                            max_gpu_util_pct=policy.max_gpu_util_pct,
                            allow_shared_gpu=policy.allow_shared_gpu,
                        )
                        result = driver.fit_cache(run_args)
                        driver.torch.cuda.empty_cache()
                    validate_multiseed_result(out, arm, mode, seed, plan_obj, expected_docs=expected_docs, check_runtime=check_runtime)
                    completed[f"{arm}/{mode}/seed{seed}"] = {"status": "trained", "bpb": result["evaluation"]["bpb"]}
    return {"schema": "mom-nextgen-multiseed-run-v1", "completed": completed}


def _find_seed_result(root: Path, arm: str, mode: str, seed: int, reference_dir: Path | None = None) -> Path | None:
    candidates = [
        unlinked_path(root / MULTISEED_DIR / f"seed{seed}" / arm / mode),
        unlinked_path(root / f"runs/dp/seed{seed}" / arm / mode),
    ]
    if reference_dir is not None:
        candidates.append(unlinked_path(reference_dir) / f"seed{seed}" / arm / mode)
    if seed == 17:
        if arm in NEXTGEN_ARMS:
            candidates.insert(0, unlinked_path(root / ng.BATCH_DIR / "jobs" / f"{arm}-{mode}"))
        candidates.extend(ng._baseline_candidate_dirs(root, arm, mode, reference_dir))
    for folder in candidates:
        if (folder / "result.json").is_file() and (folder / "identity.json").is_file():
            return folder
    return None


def _seed_pooled_bootstrap(a_by_seed: dict[int, list[dict]], b_by_seed: dict[int, list[dict]], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    common_seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if not common_seeds:
        raise ValueError("No common seeds available for paired multi-seed bootstrap")
    ref_rows = a_by_seed[common_seeds[0]]
    keys = ng.frozen.doc_keys(ref_rows)
    sizes = np.array([d["canonical_bytes"] for d in ref_rows], dtype=np.float64)
    index = np.random.default_rng(seed).integers(len(ref_rows), size=(draws, len(ref_rows)))
    denom = sizes[index].sum(axis=1) * math.log(2)

    per_seed = {}
    mean_delta_nats = np.zeros(len(ref_rows), dtype=np.float64)
    for s in common_seeds:
        rows_a = a_by_seed[s]
        rows_b = b_by_seed[s]
        require(ng.frozen.doc_keys(rows_a) == keys and ng.frozen.doc_keys(rows_b) == keys, f"Doc alignment differs at seed {s}")
        delta_s = np.array([rb["nll_nats"] - ra["nll_nats"] for ra, rb in zip(rows_a, rows_b, strict=True)], dtype=np.float64)
        mean_delta_nats += delta_s
        per_seed[str(s)] = {
            "delta_bpb": float(delta_s.sum() / (sizes.sum() * math.log(2))),
            "document_bootstrap_95pct": np.quantile(delta_s[index].sum(axis=1) / denom, [0.025, 0.975]).tolist(),
        }
    mean_delta_nats /= len(common_seeds)
    pooled_delta = float(mean_delta_nats.sum() / (sizes.sum() * math.log(2)))
    pooled_ci = np.quantile(mean_delta_nats[index].sum(axis=1) / denom, [0.025, 0.975]).tolist()
    return {
        "common_seeds": common_seeds,
        "delta_bpb": pooled_delta,
        "document_bootstrap_95pct": pooled_ci,
        "per_seed": per_seed,
    }


def aggregate_multiseed(
    *,
    root: Path,
    arms: list[str],
    seeds: list[int],
    budgets: list[str],
    out_path: Path | None = None,
    reference_dir: Path | None = None,
    expected_docs: int = 186,
) -> dict:
    """Aggregate per-seed BPB statistics and multi-seed paired bootstrap CIs vs M48 and T8."""
    root = unlinked_path(root)
    arms = [a for a in dict.fromkeys(arms)]
    seeds = verify_seed_list(seeds)
    budgets = [b for b in dict.fromkeys(budgets)]
    protocols = {}

    for mode in budgets:
        loaded: dict[str, dict[int, dict]] = {}
        for arm in dict.fromkeys([*arms, "M48", "T8"]):
            arm_seeds = {}
            for s in seeds:
                folder = _find_seed_result(root, arm, mode, s, reference_dir=reference_dir)
                if folder is None:
                    continue
                res = ng.read_receipt(folder / "result.json")
                ident = json.loads((folder / "identity.json").read_bytes())
                require(
                    res["identity"] == ident
                    and ident["arm"] == arm
                    and ident["mode"] == mode
                    and ident["config"]["seed"] == s
                    and len(res["evaluation"]["per_document"]) == expected_docs,
                    f"Result metadata conflict in {folder}",
                )
                arm_seeds[s] = res
            if arm_seeds:
                loaded[arm] = arm_seeds

        summary_arms = {}
        for arm in arms:
            arm_data = loaded.get(arm, {})
            require(arm_data, f"No verified results found for {arm}/{mode} across seeds {seeds}")
            vals = np.array([arm_data[s]["evaluation"]["bpb"] for s in sorted(arm_data)], dtype=np.float64)
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary_arms[arm] = {
                "seeds": sorted(arm_data),
                "n_seeds": len(vals),
                "per_seed_bpb": {str(s): float(arm_data[s]["evaluation"]["bpb"]) for s in sorted(arm_data)},
                "mean_bpb": float(np.mean(vals)),
                "std_bpb": std_val,
                "min_bpb": float(np.min(vals)),
                "max_bpb": float(np.max(vals)),
            }

        paired_vs_m48 = {}
        paired_vs_t8 = {}
        for arm in arms:
            for ref_name, target in (("M48", paired_vs_m48), ("T8", paired_vs_t8)):
                if ref_name in loaded and arm != ref_name:
                    a_docs = {s: res["evaluation"]["per_document"] for s, res in loaded[ref_name].items()}
                    b_docs = {s: res["evaluation"]["per_document"] for s, res in loaded[arm].items()}
                    if set(a_docs) & set(b_docs):
                        target[f"{arm} minus {ref_name}"] = _seed_pooled_bootstrap(a_docs, b_docs)

        protocols[mode] = {
            "arms": summary_arms,
            "paired_vs_m48": paired_vs_m48,
            "paired_vs_t8": paired_vs_t8,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        }

    report = {
        "schema": "mom-nextgen-multiseed-aggregate-v1",
        "arms": arms,
        "seeds_requested": seeds,
        "budgets": budgets,
        "protocols": protocols,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }
    if out_path is not None:
        out_path = unlinked_path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        digest = ng.metadata_digest(report)
        payload = (json.dumps({**report, "sha256": digest}, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with atomic_writer(out_path, overwrite=True) as stream:
            stream.write(payload)
    return report


class _HeldoutSplit:
    """In-memory `EncodedSplit`-compatible container for frozen `document_proxy.score`."""

    def __init__(self, docs: list[dict], ids_array: np.ndarray, ends_array: np.ndarray, canonical_bytes: int):
        self.docs = docs
        self.ids = ids_array
        self.ends = ends_array
        self.canonical_bytes = canonical_bytes

    def endpoint(self, index: int, consumed: int) -> tuple[int, bool]:
        doc = self.docs[index]
        if consumed == 0:
            return 0, True
        if consumed >= doc["length"]:
            return doc["canonical_bytes"], True
        value = int(self.ends[doc["start"] + consumed - 1])
        boundary_flag = 1 << 31
        return value & (boundary_flag - 1), bool(value & boundary_flag)


def load_heldout_documents(path: Path) -> list[dict]:
    """Load plain `.jsonl`, `.jsonl.gz`, or line-delimited `.txt` held-out Malayalam texts."""
    path = unlinked_path(path)
    require(path.is_file(), f"Missing held-out corpus file: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    is_jsonl = path.name.endswith((".jsonl", ".jsonl.gz"))
    docs = []
    with opener(path, "rb") as stream:
        for idx, raw_line in enumerate(stream):
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line.strip():
                continue
            if is_jsonl:
                obj = json.loads(line)
                text = obj["text"]
                doc_id = str(obj.get("id", f"heldout-{idx}"))
            else:
                text = line
                doc_id = f"heldout-{idx}"
            if text:
                docs.append({"id": doc_id, "text": text})
    require(docs, f"Held-out file contains no nonempty documents: {path}")
    return docs


def _resolve_fit_dir_for_arm(root: Path, arm: str, inputs_manifest: Path | None = None) -> Path:
    if arm in NEXTGEN_ARMS:
        manifest_path = unlinked_path(inputs_manifest) if inputs_manifest else (unlinked_path(root) / ng.INPUTS_DIR / "nextgen-inputs-manifest.json")
        if manifest_path.is_file():
            m, _ = ng.load_inputs_manifest(manifest_path)
            p = unlinked_path(manifest_path.parent / m["arms"][arm]["fit"])
            if p.is_dir():
                return p
        p = unlinked_path(root / ng.INPUTS_DIR / f"scale48k-nextgen-v1/fits/{arm}")
        if p.is_dir():
            return p
    candidates = [
        unlinked_path(root / "t5-t8-frozen-runs-v1" / ("frontend-optimization-v1" if arm == "F-space" else "scale48k-next-v1") / "full100/fits" / arm),
        unlinked_path(root / f"fits/{arm}"),
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise ValueError(f"Could not locate fit directory for {arm} under {root}")


def score_heldout_checkpoints(
    *,
    heldout_path: Path,
    arms: list[str],
    root: Path | None = None,
    mode: str = MODES[0],
    seed: int = 17,
    fit_dirs: dict[str, Path] | None = None,
    checkpoint_paths: dict[str, Path] | None = None,
    inputs_manifest: Path | None = None,
    device_str: str = "cpu",
    microbatch_blocks: int = 1,
    uuid: str = "",
    shared: bool = False,
    out_path: Path | None = None,
    reference_dir: Path | None = None,
) -> dict:
    """Encode a held-out corpus and score saved `.pt` checkpoints with zero retraining."""
    from experiments.mom_bpe import document_proxy as driver

    registry.load_plugins()
    require(not registry.load_errors(), "Plugin import errors")
    normalizer = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    mom_surface = registry.make("metric", "mom_surface")

    if device_str == "cuda":
        policy = gpu.GpuAdmission(5120, 10, True) if shared else gpu.GpuAdmission()
        policy.admit(uuid)
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    raw_docs = load_heldout_documents(heldout_path)
    canon_docs = []
    for row in raw_docs:
        canon = normalizer.normalize(row["text"])
        raw_b = row["text"].encode("utf-8")
        can_b = canon.encode("utf-8")
        require(len(can_b) > 0, f"Held-out document became empty after normalization: {row['id']}")
        canon_docs.append({
            "id": row["id"],
            "text": row["text"],
            "canonical_text": canon,
            "canonical_bytes": len(can_b),
            "canonical_sha256": hashlib.sha256(can_b).hexdigest(),
            "raw_bytes": len(raw_b),
        })

    arm_reports = {}
    raw_results = {}
    for arm in dict.fromkeys(arms):
        require(arm in ALL_SEVEN_ARMS, f"Unsupported arm: {arm}")
        fit_dir = unlinked_path(fit_dirs[arm]) if fit_dirs and arm in fit_dirs else _resolve_fit_dir_for_arm(root, arm, inputs_manifest=inputs_manifest)
        ckpt_path = None
        if checkpoint_paths and arm in checkpoint_paths:
            ckpt_path = unlinked_path(checkpoint_paths[arm])
        else:
            require(root is not None, "Provide --root or --checkpoint-map")
            folder = _find_seed_result(root, arm, mode, seed, reference_dir=reference_dir)
            require(folder is not None, f"Missing saved run folder for {arm}/{mode}/seed{seed}")
            res = ng.read_receipt(folder / "result.json")
            ckpt_path = folder / res["checkpoint"]["path"]
        require(ckpt_path.is_file(), f"Missing checkpoint file: {ckpt_path}")

        backend, uniform_prefix = load_arm_backend(fit_dir, arm, normalizer)
        encoded_docs = []
        id_chunks = []
        end_chunks = []
        offset = 0
        total_bytes = 0
        for row in canon_docs:
            enc = backend.encode_offsets(row["text"])
            data = row["canonical_text"].encode("utf-8")
            if arm != "M48":
                require(backend.decode(enc.ids) == enc.text and enc.text.encode("utf-8") == data, f"Lossless check failed for {arm} on {row['id']}")
            if isinstance(backend, NativeHeldoutBackend):
                # Compute native ends safely via character offset monotonic projection.
                ends = []
                b_map = [0]
                for ch in enc.text:
                    b_map.append(b_map[-1] + len(ch.encode("utf-8")))
                for _, c_end in enc.offsets:
                    end = b_map[min(c_end, len(b_map) - 1)]
                    boundary = end == len(data) or (data[end] & 0xC0) != 0x80
                    ends.append(end | ((1 << 31) if boundary else 0))
            else:
                ends = pack.token_byte_ends(enc, backend, data, uniform_prefix)
            encoded_docs.append({
                "id": row["id"],
                "canonical_sha256": row["canonical_sha256"],
                "canonical_bytes": row["canonical_bytes"],
                "raw_bytes": row["raw_bytes"],
                "start": offset,
                "length": len(enc.ids),
            })
            id_chunks.append(np.asarray(enc.ids, dtype="<u2"))
            end_chunks.append(np.asarray(ends, dtype="<u4"))
            offset += len(enc.ids)
            total_bytes += row["canonical_bytes"]

        split = _HeldoutSplit(encoded_docs, np.concatenate(id_chunks), np.concatenate(end_chunks), total_bytes)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ident = state["identity"]
        spec = ident["config"]["model"]
        context = ident["config"]["context"]
        vocab = state["model"]["wte.weight"].shape[0]
        model = registry.make("proxy_model", spec).build(vocab, context).to(device)
        model.load_state_dict(state["model"], strict=True)

        specials = dict(zip(DEFAULT_SPECIALS, range(len(DEFAULT_SPECIALS))))
        evaluation = driver.score(model, split, context, specials, microbatch_blocks, device, bf16=(device.type == "cuda"))
        surface = mom_surface.compute(backend, [(r["text"], "heldout") for r in canon_docs])

        raw_results[arm] = {"evaluation": evaluation}
        arm_reports[arm] = {
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": file_digest(ckpt_path)[1],
            "seed": ident["config"]["seed"],
            "mode": ident["mode"],
            "bpb": evaluation["bpb"],
            "predicted_tokens": evaluation["predicted_tokens"],
            "canonical_bytes": evaluation["canonical_bytes"],
            "bytes_per_prediction": evaluation["canonical_bytes"] / evaluation["predicted_tokens"],
            "mom_surface": surface,
            "per_document": evaluation["per_document"],
        }

    paired = ng.paired_bootstrap_all(raw_results, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS) if len(raw_results) >= 2 else {}
    report = {
        "schema": "mom-nextgen-heldout-v1",
        "heldout_file": str(heldout_path),
        "heldout_sha256": file_digest(heldout_path)[1],
        "documents": len(canon_docs),
        "canonical_bytes": sum(d["canonical_bytes"] for d in canon_docs),
        "mode": mode,
        "seed": seed,
        "arms": arm_reports,
        "paired": paired,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }
    if out_path is not None:
        out_path = unlinked_path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        digest = ng.metadata_digest(report)
        payload = (json.dumps({**report, "sha256": digest}, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with atomic_writer(out_path, overwrite=True) as stream:
            stream.write(payload)
    return report


def backup_multiseed_and_heldout_hub(
    *,
    root: Path,
    repo: str,
    enabled: bool,
    hub_checkpoints: bool = False,
    api=None,
) -> dict:
    """Upload verified multi-seed and held-out artifacts strictly under `seed_replication/` or `heldout/`."""
    ng._force_no_xet()
    require(enabled, "Private Hugging Face backup requires explicit --enable")
    require(repo and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo), "Invalid private Hugging Face repository name")
    from huggingface_hub import CommitOperationAdd

    root = unlinked_path(root)
    files: dict[str, Path] = {}
    multiseed_root = unlinked_path(root / MULTISEED_DIR)
    if multiseed_root.is_dir():
        for res_path in sorted(multiseed_root.glob("seed*/*/*/result.json")):
            mode_dir = res_path.parent
            arm_dir = mode_dir.parent
            seed_dir = arm_dir.parent
            ident_path = mode_dir / "identity.json"
            require(ident_path.is_file(), f"Missing identity.json alongside {res_path}")
            res = ng.read_receipt(res_path)
            files[f"seed_replication/{seed_dir.name}/{arm_dir.name}/{mode_dir.name}/result.json"] = res_path
            files[f"seed_replication/{seed_dir.name}/{arm_dir.name}/{mode_dir.name}/identity.json"] = ident_path
            if hub_checkpoints:
                ckpt_path = mode_dir / res["checkpoint"]["path"]
                require(ckpt_path.is_file(), f"Missing checkpoint {ckpt_path}")
                files[f"seed_replication/{seed_dir.name}/{arm_dir.name}/{mode_dir.name}/{ckpt_path.name}"] = ckpt_path
        for json_file in sorted(multiseed_root.glob("aggregate*.json")):
            files[f"seed_replication/{json_file.name}"] = json_file

    heldout_root = unlinked_path(root / HELDOUT_DIR)
    if heldout_root.is_dir():
        for h_file in sorted(heldout_root.glob("*.json")):
            files[f"heldout/{h_file.name}"] = h_file

    require(files, "No verified multi-seed or held-out outputs found to back up")
    for remote_name in files:
        require(
            remote_name.startswith(ALLOWED_HUB_PREFIXES) and not remote_name.startswith("seed17/"),
            f"Refusing to write outside seed_replication/ or heldout/: {remote_name}",
        )

    expected = {name: file_digest(path) for name, path in files.items()}
    api = api or ng.hub_client()
    info = api.repo_info(repo, repo_type="model", timeout=20)
    require(info.private is True, "Backup repository must be private")
    revision = ng.verify_hub_revision(api, repo, info.sha)
    existing = set(api.list_repo_files(repo, repo_type="model", revision=revision))

    staging_parent = multiseed_root if multiseed_root.is_dir() else heldout_root

    def compare(name: str, pin: str) -> None:
        with TemporaryDirectory(prefix="ms-hub-verify-", dir=staging_parent) as tmp:
            ng.download_member(api, {"repo": repo, "revision": pin, "filename": name}, Path(tmp), expected[name], force=True)

    for name in set(files) & existing:
        compare(name, revision)
    additions = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(files[name]))
        for name in sorted(set(files) - existing)
    ]
    if additions:
        revision = api.create_commit(
            repo,
            repo_type="model",
            parent_commit=revision,
            operations=additions,
            commit_message="Back up verified multi-seed and held-out nextgen outputs",
        ).oid
    ng.verify_hub_revision(api, repo, revision)
    for name in files:
        compare(name, revision)

    return {
        "repository": repo,
        "revision": revision,
        "uploaded_files": len(additions),
        "verified_files": len(files),
        "prefixes": sorted({name.split("/", 1)[0] + "/" for name in files}),
        "files": {name: {"size": s, "sha256": d} for name, (s, d) in expected.items()},
    }


def _parse_kv_paths(items: list[str] | None) -> dict[str, Path]:
    out = {}
    for item in items or ():
        require("=" in item, f"Expected ARM=PATH argument, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = Path(v)
    return out


def main() -> None:
    ng._force_no_xet()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--root", type=Path, required=True)
    run_p.add_argument("--source", type=Path, required=True)
    run_p.add_argument("--arms", nargs="+", required=True)
    run_p.add_argument("--seeds", nargs="+", type=int, required=True)
    run_p.add_argument("--budgets", nargs="+", default=list(MODES))
    run_p.add_argument("--uuid", default="")
    run_p.add_argument("--enable", action="store_true")
    run_p.add_argument("--allocated", action="store_true")
    run_p.add_argument("--shared", action="store_true")
    run_p.add_argument("--inputs-manifest", type=Path)
    run_p.add_argument("--baseline-inputs", type=Path)

    agg_p = sub.add_parser("aggregate")
    agg_p.add_argument("--root", type=Path, required=True)
    agg_p.add_argument("--arms", nargs="+", default=list(ALL_SEVEN_ARMS))
    agg_p.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 101])
    agg_p.add_argument("--budgets", nargs="+", default=list(MODES))
    agg_p.add_argument("--reference-dir", type=Path)
    agg_p.add_argument("--out", type=Path)

    held_p = sub.add_parser("score-heldout")
    held_p.add_argument("--heldout", type=Path, required=True)
    held_p.add_argument("--arms", nargs="+", required=True)
    held_p.add_argument("--root", type=Path)
    held_p.add_argument("--mode", choices=MODES, default=MODES[0])
    held_p.add_argument("--seed", type=int, default=17)
    held_p.add_argument("--fit-map", nargs="*", help="Optional ARM=FIT_DIR overrides")
    held_p.add_argument("--checkpoint-map", nargs="*", help="Optional ARM=CKPT_PT overrides")
    held_p.add_argument("--inputs-manifest", type=Path)
    held_p.add_argument("--reference-dir", type=Path)
    held_p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    held_p.add_argument("--microbatch-blocks", type=int, default=1)
    held_p.add_argument("--uuid", default="")
    held_p.add_argument("--shared", action="store_true")
    held_p.add_argument("--out", type=Path)

    bak_p = sub.add_parser("backup-hub")
    bak_p.add_argument("--root", type=Path, required=True)
    bak_p.add_argument("--hub-repo", default=DEFAULT_RESULTS_REPO)
    bak_p.add_argument("--enable", action="store_true")
    bak_p.add_argument("--hub-checkpoints", action="store_true")

    args = parser.parse_args()
    if args.command == "run":
        deploy_root = Path(__file__).resolve().parents[2]
        if str(deploy_root) not in sys.path:
            sys.path.insert(0, str(deploy_root))
        source = args.source.resolve()
        ng.verify_source(source)
        ng.verify_operational_parity(deploy_root, source)
        res = run_multiseed(
            root=args.root,
            source=source,
            arms=args.arms,
            seeds=args.seeds,
            budgets=args.budgets,
            uuid=args.uuid,
            allocated=args.allocated,
            enabled=args.enable,
            shared=args.shared,
            inputs_manifest=args.inputs_manifest,
            baseline_inputs=args.baseline_inputs,
        )
    elif args.command == "aggregate":
        out = args.out or (unlinked_path(args.root) / MULTISEED_DIR / "aggregate.json")
        res = aggregate_multiseed(
            root=args.root,
            arms=args.arms,
            seeds=args.seeds,
            budgets=args.budgets,
            out_path=out,
            reference_dir=args.reference_dir,
        )
    elif args.command == "score-heldout":
        res = score_heldout_checkpoints(
            heldout_path=args.heldout,
            arms=args.arms,
            root=args.root,
            mode=args.mode,
            seed=args.seed,
            fit_dirs=_parse_kv_paths(args.fit_map),
            checkpoint_paths=_parse_kv_paths(args.checkpoint_map),
            inputs_manifest=args.inputs_manifest,
            device_str=args.device,
            microbatch_blocks=args.microbatch_blocks,
            uuid=args.uuid,
            shared=args.shared,
            out_path=args.out,
            reference_dir=args.reference_dir,
        )
    else:
        res = backup_multiseed_and_heldout_hub(
            root=args.root,
            repo=args.hub_repo,
            enabled=args.enable,
            hub_checkpoints=args.hub_checkpoints,
        )
    print(json.dumps(res, indent=2, default=str))


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    main()
