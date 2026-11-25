"""Verify or extract a DGX bundle, preflight its runtime, or verify a smoke result."""

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import stat
import sys
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile


MANIFEST = "bundle-manifest.json"
SUMS = "SHA256SUMS"
SCHEMA = "mom-dgx-handoff-v1"
ARMS = ("M48", "T5", "T8", "F-space")
MODES = ("fixed_source_bytes", "fixed_training_tokens")
RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", re.I)
HEX = re.compile(r"[0-9a-f]{64}")


def digest_stream(stream, target=None):
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(2**20):
        digest.update(chunk)
        size += len(chunk)
        if target is not None:
            target.write(chunk)
    return size, digest.hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return digest_stream(stream)


def safe_name(name):
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or path.as_posix() != name
            or any(part in (".", "..") or part.endswith((".", " ")) or RESERVED.fullmatch(part) for part in path.parts)
            or any(char in '\\:<>"|?*' or ord(char) < 32 for char in name)):
        raise ValueError(f"Unsafe bundle path {name!r}")
    return path


def safe_names(names):
    folded = set()
    spellings = {}
    for name in names:
        path = safe_name(name)
        for part in (path, *path.parents):
            spelling = str(part)
            if spellings.setdefault(spelling.casefold(), spelling) != spelling:
                raise ValueError(f"Case-colliding bundle path {name!r}")
        if name.casefold() in folded:
            raise ValueError(f"Duplicate or case-colliding bundle path {name!r}")
        folded.add(name.casefold())
    if any(str(parent) in folded for name in folded for parent in PurePosixPath(name).parents):
        raise ValueError("A bundle file shadows another file's parent directory")


def is_link(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def parse_manifest(raw):
    manifest = json.loads(raw)
    if manifest["schema"] != SCHEMA or manifest["profile"] not in ("smoke", "full"):
        raise ValueError("Unsupported DGX bundle manifest")
    files = manifest["files"]
    names = [item["path"] for item in files]
    safe_names([*names, MANIFEST, SUMS])
    for item in files:
        if (type(item["bytes"]) is not int or item["bytes"] < 0 or not HEX.fullmatch(item["sha256"])
                or not item["identity"] or item.get("arm") not in (None, *ARMS)):
            raise ValueError(f"Invalid file entry {item['path']!r}")
    return manifest, {item["path"]: (item["bytes"], item["sha256"]) for item in files}


def sums_bytes(raw, files):
    all_sums = {name: identity[1] for name, identity in files.items()}
    all_sums[MANIFEST] = hashlib.sha256(raw).hexdigest()
    return "".join(f"{digest}  {name}\n" for name, digest in sorted(all_sums.items())).encode()


def verify_bundle(root):
    if is_link(root) or not root.is_dir():
        raise ValueError("Expected a real bundle root directory")
    found = []
    for folder, dirs, files in os.walk(root, followlinks=False):
        base = Path(folder)
        if any(is_link(base / name) for name in [*dirs, *files]):
            raise ValueError("Symlink found in extracted bundle")
        if any(not (base / name).is_file() for name in files):
            raise ValueError("Non-regular file found in extracted bundle")
        found.extend((base / name).relative_to(root).as_posix() for name in files)
    safe_names(found)
    raw = (root / MANIFEST).read_bytes()
    manifest, expected = parse_manifest(raw)
    if set(found) != {*expected, MANIFEST, SUMS} or (root / SUMS).read_bytes() != sums_bytes(raw, expected):
        raise ValueError("Extracted membership or checksum list differs from the manifest")
    for name, identity in expected.items():
        if file_digest(root / name) != identity:
            raise ValueError(f"Size or SHA-256 differs for {name}")
    return manifest


def extract_bundle(archive, destination, sha256):
    # Use the same open file for outer checksum and ZIP reads.
    with archive.open("rb") as stream:
        if not HEX.fullmatch(sha256) or digest_stream(stream)[1] != sha256:
            raise ValueError("Archive SHA-256 differs from the trusted transfer receipt")
        stream.seek(0)
        with ZipFile(stream) as bundle:
            members = bundle.infolist()
            safe_names([item.filename for item in members])
            for item in members:
                mode = stat.S_IFMT(item.external_attr >> 16)
                if item.is_dir() or item.external_attr & 0x10 or mode not in (0, stat.S_IFREG):
                    raise ValueError(f"Non-regular ZIP member {item.filename!r}")
            if bundle.getinfo(MANIFEST).file_size > 4 * 2**20:
                raise ValueError("Bundle manifest exceeds its size limit")
            raw = bundle.read(MANIFEST)
            _, expected = parse_manifest(raw)
            sums = sums_bytes(raw, expected)
            if ({item.filename for item in members} != {*expected, MANIFEST, SUMS}
                    or bundle.getinfo(SUMS).file_size != len(sums) or bundle.read(SUMS) != sums
                    or any(bundle.getinfo(name).file_size != size for name, (size, _) in expected.items())):
                raise ValueError("ZIP membership, file sizes or checksum list differs from the manifest")
            if any(is_link(parent) for parent in (destination, *destination.parents)):
                raise ValueError("Extraction destination contains a symbolic link")
            destination.mkdir(exist_ok=False)
            for item in members:
                path = destination.joinpath(*safe_name(item.filename).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as member, path.open("xb") as target:
                    actual = digest_stream(member, target)
                if item.filename in expected and actual != expected[item.filename]:
                    raise ValueError(f"ZIP member differs from recorded hash for {item.filename}")
    return verify_bundle(destination)


def imports(source):
    if sys.version_info < (3, 10):
        raise ValueError("The document proxy requires Python 3.10 or newer")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from experiments.mom_bpe import document_proxy, proxy_blocks

    if not Path(document_proxy.__file__).resolve().is_relative_to(source):
        raise ValueError("Project imports resolved outside the verified source snapshot")
    return document_proxy, proxy_blocks


def check_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label} differs from its verified reference")


def plan(driver, cache, mode, budget=None, update_targets=None, checkpoint_updates=None):
    return driver.fit_cache(SimpleNamespace(command="plan", cache=cache, mode=mode, budget=budget, out=None,
                           seed=None, update_targets=update_targets, microbatch_blocks=None,
                           checkpoint_updates=checkpoint_updates))


def check_inputs(source, arms, *, ledgers, encoders=False):
    """Authenticate local files and recalculate the existing ledgers without a model update."""
    driver, blocks = imports(source)
    from experiments.mom_bpe.scale_mlbpe import FILES
    from experiments.mom_bpe.scale_proxy_plan import verified_backend
    from experiments.mom_bpe.verify_frozen import checked_member
    from mltok import registry
    from mltok.lib.bpe_core import DEFAULT_SPECIALS
    from mltok.lib.bpe_progress import canonical
    from mltok.lib.picky_backend import ARTIFACT

    runs = source / "experiments/mom_bpe/runs"
    manifest_path = runs / "scale48k-v1/a-100mb/manifest.json"
    evidence, _ = blocks.receipt(source / "experiments/mom_bpe/document-proxy-evidence-v1.json")
    manifest = json.loads(manifest_path.read_bytes())
    check_equal(file_digest(manifest_path)[1], evidence["manifest_sha256"], "Training manifest SHA-256")
    streams = {name: manifest["splits"][name]["canonical_stream_sha256"] for name in ("train", "validation")}
    specials = dict(zip(DEFAULT_SPECIALS, range(len(DEFAULT_SPECIALS))))
    if encoders:
        registry.load_plugins()
        normalizer = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    validated = {}
    held_documents = None
    for arm in arms:
        choices = sorted((runs / "dp/cache").glob(f"{arm}-*"))
        if len(choices) != 1:
            raise ValueError(f"Expected exactly one cache at {runs / 'dp/cache' / (arm + '-*')}")
        cache = choices[0]
        spec, cache_sha = blocks.receipt(cache / "cache.json")
        identity, published = spec["identity"], evidence["arms"][arm]
        check_equal(cache_sha, published["cache_sha256"], f"{arm} cache receipt")
        check_equal(cache.name, arm + "-" + hashlib.sha256(canonical(identity)).hexdigest(), f"{arm} cache directory")
        check_equal((identity["arm"], identity["tokenizer_sha256"], identity["manifest_sha256"], identity["streams"],
                     identity["vocab"], identity["specials"]),
                    (arm, published["tokenizer_sha256"], evidence["manifest_sha256"], streams, 48000, specials), arm)
        fit_dir = runs / ("frontend-optimization-v1" if arm == "F-space" else "scale48k-next-v1") / "full100/fits" / arm
        fit_path = fit_dir / "fit.json"
        fit = json.loads(fit_path.read_bytes())
        check_equal(file_digest(fit_path)[1], identity["fit_sha256"], f"{arm} fit SHA-256")
        check_equal(fit["identity"], json.loads((fit_dir / "identity.json").read_bytes()), f"{arm} fit identity file")
        check_equal(fit["identity"], identity["preprocessing_fit_identity"], f"{arm} preprocessing recipe")
        check_equal((fit["occupied_vocab"], fit["identity"]["requested_vocab"]), (48000, 48000), f"{arm} vocabulary")
        check_equal((fit["identity"]["arm"], fit["identity"]["manifest_sha256"],
                     fit["identity"]["train_stream_sha256"], fit["identity"]["validation_stream_sha256"]),
                    (arm, evidence["manifest_sha256"], streams["train"], streams["validation"]), f"{arm} source")
        validation = json.loads((fit_dir / "validation.json").read_bytes())
        val_digest = validation.pop("result_sha256")
        check_equal(hashlib.sha256(canonical(validation)).hexdigest(), val_digest, f"{arm} tokenizer evaluation receipt")
        check_equal(validation["fit"], fit, f"{arm} evaluated fit")
        model_sha = fit["files_sha256"]["tokenizer.json"] if arm == "M48" else fit["model_sha256"]
        check_equal(model_sha, identity["tokenizer_sha256"], f"{arm} encoder SHA-256")
        encoder_members = FILES if arm == "M48" else (ARTIFACT,)
        if encoders:
            backend, actual, occupied = verified_backend(fit_dir, fit, normalizer)
            check_equal((actual, occupied, backend.special_ids, len(backend.fallback_ids)),
                        (model_sha, 48000, specials, 256), f"{arm} encoder vocabulary and controls")
        if arm == "M48" and (fit_dir / "tokenizer.json").exists():
            for name in FILES:
                check_equal(file_digest(fit_dir / name)[1], fit["files_sha256"][name], f"M48 {name}")
        elif (fit_dir / ARTIFACT).exists():
            check_equal(file_digest(fit_dir / ARTIFACT)[1], model_sha, f"{arm} encoder file")
        elif ledgers:
            raise FileNotFoundError(fit_dir / encoder_members[0])
        cache_paths = [cache / "cache.json"]
        for split in ("train", "validation"):
            split_path = cache / f"{split}.json"
            split_info, split_sha = blocks.receipt(split_path)
            check_equal(split_sha, spec["splits"][split], f"{arm} {split} receipt")
            cache_paths.extend([split_path, *(checked_member(cache, split_info[key]) for key in ("ids", "ends"))])
            if split == "train":
                check_equal(split_info["canonical_bytes"], published["train_cache_bytes"], f"{arm} cached train bytes")
                continue
            docs = [(doc["id"], doc["canonical_sha256"], doc["canonical_bytes"]) for doc in split_info["documents"]]
            if held_documents is None:
                held_documents = docs
            check_equal(docs, held_documents, f"{arm} ordered validation documents")
            check_equal((split_sha, len(docs), split_info["canonical_bytes"], streams["validation"]),
                        (published["validation_split_sha256"], 186, 999950,
                         evidence["validation"]["canonical_stream_sha256"]), f"{arm} held-out cache")
            check_equal(sum(doc["length"] + 1 for doc in split_info["documents"]),
                        published["validation_predicted_targets"], f"{arm} held-out targets")
        reports = []
        if ledgers:
            for mode in MODES:
                path = runs / f"dp/ledgers/{arm}-{mode}.json"
                saved, saved_sha = blocks.receipt(path)
                check_equal(saved_sha, published[mode]["plan_sha256"], f"{arm} {mode} receipt SHA-256")
                check_equal(plan(driver, cache, mode), saved, f"{arm} {mode} recomputed plan")
                forecast = saved["forecast"]
                for key, value in forecast.items():
                    check_equal(value, published[mode][key], f"{arm} {mode} {key}")
                check_equal(saved["optimizer_sha256"], evidence["optimizer_sha256"], f"{arm} {mode} optimizer recipe")
                if mode == MODES[0]:
                    check_equal((forecast["canonical_bytes"], saved["exposure"]["ordered_complete_documents_sha256"]),
                                (49998152, evidence["fixed_source_stream_sha256"]), f"{arm} fixed-byte selection")
                else:
                    check_equal(forecast["predicted_tokens"], 5000000, f"{arm} fixed-target predictions")
                reports.append(path)
        smoke = plan(driver, cache, MODES[1], 1024, 512, 1) if arm == "F-space" else None
        if smoke:
            check_equal((smoke["forecast"]["predicted_tokens"], smoke["forecast"]["optimizer_updates"]),
                        (1024, 2), "F-space smoke forecast")
        validated[arm] = {"cache": cache, "cache_sha256": cache_sha, "cache_paths": cache_paths, "fit_dir": fit_dir,
                          "tokenizer_sha256": model_sha, "encoder_members": encoder_members, "ledgers": reports,
                          "smoke": smoke}
    return validated


def preflight(root, scratch, manifest):
    driver, _ = imports(root / "source")
    import numpy
    import tokenizers
    import torch
    import yaml

    if os.environ.get("MLTOK_PLUGINS"):
        raise ValueError("Clear MLTOK_PLUGINS before checking the pinned project plugins")
    driver.registry.load_plugins()
    errors = driver.registry.load_errors()
    if errors:
        raise ValueError("Project plugin import failures: " + ", ".join(Path(path).name for path, _ in errors))
    if ("weights_only" not in inspect.signature(torch.load).parameters
            or "including_emulation" not in inspect.signature(torch.cuda.is_bf16_supported).parameters):
        raise ValueError("PyTorch lacks the required checkpoint or BF16 APIs")
    if any(is_link(path) for path in (scratch, *scratch.parents)):
        raise ValueError("Scratch path contains a symbolic link")
    if scratch.resolve().is_relative_to(root.resolve()):
        raise ValueError("Select a scratch directory separate from the extracted bundle")
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="preflight-", dir=scratch) as probe:
        staged = Path(probe) / "staged"
        with staged.open("xb") as stream:
            stream.write(b"writable\n")
            stream.flush()
            os.fsync(stream.fileno())
        linked = staged.with_name("linked")
        os.link(staged, linked)
        check_equal(file_digest(staged), file_digest(linked), "Scratch hard-link verification")
    free = shutil.disk_usage(scratch).free
    needed = manifest["storage"]["additional_free_scratch_bytes"]
    if free < needed:
        raise ValueError(f"Scratch has {free} free bytes; bundle requires {needed}")
    if torch.version.cuda is None:
        raise ValueError("PyTorch is a CPU-only build")
    gpu = driver.gpu_preflight(8192)
    if not torch.cuda.is_available() or "A100" not in gpu["type"]:
        raise ValueError("An eligible CUDA A100 was not detected")
    return {"python": sys.version.split()[0], "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "numpy": numpy.__version__, "tokenizers": tokenizers.__version__, "yaml": yaml.__version__,
            "plugins": len(driver.registry.describe()), "gpu": gpu, "scratch_free_bytes": free,
            "scratch_required_bytes": needed}


def verify_smoke(root, output):
    driver, blocks = imports(root / "source")
    from experiments.mom_bpe.verify_frozen import checked_member

    result, digest = blocks.receipt(output / "result.json")
    identity, evaluation = result["identity"], result["evaluation"]
    check_equal(identity, json.loads((output / "identity.json").read_bytes()), "Published smoke identity")
    checked = check_inputs(root / "source", ("F-space",), ledgers=False)["F-space"]["smoke"]
    for key, value in checked.items():
        check_equal(identity[key], value, f"Smoke {key}")
    gpu = identity["runtime"]["gpu"]
    if identity["parameters"] != 38799936 or not gpu or gpu["physical_index"] == 7 or "A100" not in gpu["type"]:
        raise ValueError("Result does not describe an eligible 38.8M A100 smoke run")
    docs = evaluation["per_document"]
    cache, = (root / "source/experiments/mom_bpe/runs/dp/cache").glob("F-space-*")
    held = blocks.EncodedSplit(cache, "validation", identity["validation"]["split_sha256"])
    check_equal([(d["id"], d["canonical_sha256"], d["canonical_bytes"], d["predicted_tokens"]) for d in docs],
                [(d["id"], d["canonical_sha256"], d["canonical_bytes"], d["length"]+1) for d in held.docs],
                "Smoke validation documents and targets")
    summary = driver.summarize_bpb((d["nll_nats"] for d in docs), 999950, sum(d["predicted_tokens"] for d in docs))
    check_equal(summary, {k: v for k, v in evaluation.items() if k != "per_document"}, "Smoke BPB totals")
    final = checked_member(output, result["checkpoint"])
    generations = []
    for name in sorted(output.glob("step-*.pt")):
        match = driver.GENERATION.fullmatch(name.name)
        size, checksum = file_digest(name)
        if not match or name.is_symlink() or match[2] != checksum:
            raise ValueError(f"Invalid smoke checkpoint {name.name}")
        with name.open("rb") as stream:
            driver._verify_checkpoint_archive(stream)
        generations.append({"step": int(match[1]), "name": name.name, "bytes": size, "sha256": checksum})
    check_equal([item["step"] for item in generations], [1, 2], "Smoke checkpoint generation count")
    check_equal(generations[-1]["name"], final.name, "Smoke final checkpoint")
    return {"result_sha256": digest, "checkpoints": generations, "optimizer_updates": 2,
            "predicted_tokens": 1024, "evaluation": summary, "gpu_uuid": gpu["uuid"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("archive", type=Path)
    extract.add_argument("destination", type=Path)
    extract.add_argument("--sha256", required=True)
    for command in ("verify", "caches", "preflight", "smoke-result"):
        item = sub.add_parser(command)
        item.add_argument("bundle", type=Path)
        if command == "preflight":
            item.add_argument("--scratch", type=Path, required=True)
        elif command == "smoke-result":
            item.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        manifest = extract_bundle(args.archive, args.destination, args.sha256)
        result = {"profile": manifest["profile"], "files_verified": len(manifest["files"])}
    else:
        manifest = verify_bundle(args.bundle)
        root = args.bundle.resolve()
        if args.command == "preflight":
            result = preflight(root, args.scratch.absolute(), manifest)
        elif args.command == "smoke-result":
            result = verify_smoke(root, args.out.resolve())
        elif args.command == "caches":
            full = manifest["profile"] == "full"
            checked = check_inputs(root / "source", ARMS if full else ("F-space",), ledgers=full)
            result = {"checked_arms": list(checked), "verified_ledgers": 8 if full else 0,
                      "smoke_updates": checked["F-space"]["smoke"]["forecast"]["optimizer_updates"]}
        else:
            result = {"profile": manifest["profile"], "files_verified": len(manifest["files"])}
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
