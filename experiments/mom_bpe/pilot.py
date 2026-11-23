"""Prepare a bounded freeze-derived CPU pilot and run one restartable arm."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe.diagnose import HERE, format_json, immutable, sha
from experiments.mom_bpe.verify_frozen import checked_member, load_rows, verify
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import canonical
from mltok.lib.io import data_fingerprint
from mltok.lib.mom_backend import ARTIFACT
from mltok.snapshots import _lock, file_identity

ARMS = {"A": "T0-matched", "B": "T1", "C": "T2"}
TRAIN_CAP = 8 * 1024**2
EVAL_CAP = 16 * 1024**2
CHECKPOINT_EVERY = 2048


def configuration():
    registry.load_plugins()
    data = (HERE / "arms.yaml").read_bytes()
    cfg = yaml.safe_load(data)
    if cfg["vocab"] != 48000:
        raise ValueError("The primary pilot configuration requires a 48K vocabulary target")
    specs = {
        arm: {**cfg["common_algorithm"], **cfg["arms"][name], "require_full_vocabulary": False}
        for arm, name in ARMS.items()
    }
    if any(s["morphology_penalty_milli"] or s.get("alignment_file") or s.get("use_verified_alignment")
           for s in specs.values()):
        raise ValueError("The first pilot excludes morphology")
    normalizer, pretokenizer = (cfg[k] for k in ("normalizer", "pretokenizer"))
    return {
        "config_sha256": sha(data), "pilot_code_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "vocab_target": cfg["vocab"], "specials": DEFAULT_SPECIALS,
        "normalizer": normalizer, "pretokenizer": pretokenizer,
        "normalizer_fingerprint": registry.fingerprint("normalizer", normalizer),
        "pretokenizer_fingerprint": registry.fingerprint("pretokenizer", pretokenizer),
        "metric_fingerprint": registry.fingerprint("metric", "mom_surface"),
        "checkpoint_every": CHECKPOINT_EVERY,
        "arms": {arm: {"spec": spec, "fingerprint": registry.fingerprint("algorithm", spec)}
                 for arm, spec in specs.items()},
    }


def materialize(root, entries, cap, *, select_prefix):
    rows, nbytes = [], 0
    for entry in entries:
        for row in load_rows(checked_member(root, entry)):
            size = row["canonical_bytes"]
            if nbytes + size > cap:
                if select_prefix and rows:
                    return rows, nbytes
                raise ValueError("An evaluation set or first training document exceeds the pilot byte limit")
            rows.append(row)
            nbytes += size
    if nbytes == 0:
        raise ValueError("The pilot slice contains no canonical bytes")
    return rows, nbytes


def prepare(manifest, out, train_bytes):
    if type(train_bytes) is not int or not 1 <= train_bytes <= TRAIN_CAP:
        raise ValueError(f"Pilot training byte cap must be within 1 and {TRAIN_CAP}")
    manifest, out = Path(manifest).resolve(), Path(out).resolve()
    freeze = verify(manifest, require_morphology=False)
    raw = manifest.read_bytes()
    if sha(raw) != freeze["manifest_sha256"]:
        raise ValueError("Frozen manifest changed after verification")
    source = json.loads(raw)
    train, ntrain = materialize(manifest.parent, source["splits"]["train"], train_bytes, select_prefix=True)
    held, nheld = materialize(manifest.parent, source["splits"]["eval"], EVAL_CAP, select_prefix=False)
    orthography = [r for r in load_rows(checked_member(manifest.parent, source["orthography_panel"]))
                   if r["status"] == "adjudicated"]
    if sum(len(r["text"].encode()) for r in orthography) > EVAL_CAP:
        raise ValueError("Orthography panel exceeds the pilot byte limit")
    config = configuration()
    if config["normalizer_fingerprint"] != source["normalizer_fingerprint"]:
        raise ValueError("Normalizer changed after freeze verification")
    out.mkdir(parents=True, exist_ok=True)
    with _lock(out / ".prepare.lock"):
        files = {}
        for scope, rows in (("train", train), ("eval", held), ("orthography", orthography)):
            filename = f"{scope}.jsonl"
            content = b"".join(canonical(row) + b"\n" for row in rows)
            immutable(out / filename, content)
            files[scope] = {"path": filename, "size": len(content), "sha256": sha(content)}
        plan = {
            "schema": "mom-cpu-pilot-v1", "freeze": freeze, "config": config,
            "source_protocol_sha256": source["protocol_sha256"], "sources": source["sources"],
            "training_byte_cap": train_bytes, "training_canonical_bytes": ntrain, "training_docs": len(train),
            "evaluation_canonical_bytes": nheld, "evaluation_docs": len(held), "files": files,
            "training_fingerprint": data_fingerprint([str(out / files["train"]["path"])]),
        }
        immutable(out / "plan.json", format_json(plan))
    return plan


def peak_rss_bytes():
    if sys.platform != "win32":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    import ctypes
    from ctypes import wintypes

    class Memory(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (field, ctypes.c_size_t) for field in (
                "peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get_process = kernel.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD)
    get_memory.restype = wintypes.BOOL
    sample = Memory()
    sample.size = ctypes.sizeof(sample)
    if not get_memory(get_process(), ctypes.byref(sample), sample.size):
        raise ctypes.WinError(ctypes.get_last_error())
    return sample.peak_rss


def measure(backend, rows, source_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[source_key]].append((row["text"], row[source_key]))
    metric = registry.make("metric", "mom_surface")

    def summarize(docs):
        scores = metric.compute(backend, docs)
        counts = scores["mom_ids"]
        canonical_text = [backend.expected(t) for t, _ in docs]
        words = sum(len(t.split()) for t in canonical_text)
        nbytes = sum(len(t.encode()) for t in canonical_text)
        return {**scores, "canonical_bytes": nbytes, "canonical_words": words,
                "fertility": counts / words if words else None,
                "bytes_per_token": nbytes / counts if counts else None,
                "fallback_rate": scores["mom_byte_fallback_ids"] / counts if counts else None}

    return {"all": summarize([doc for docs in grouped.values() for doc in docs]),
            "by_stratum": {label: summarize(docs) for label, docs in sorted(grouped.items())}}


def run(plan_path, arm):
    if arm not in ARMS:
        raise ValueError("Expected pilot arm A, B or C")
    path = Path(plan_path).resolve()
    raw = path.read_bytes()
    plan = json.loads(raw)
    resolved = configuration()
    if plan["schema"] != "mom-cpu-pilot-v1" or plan["config"] != resolved:
        raise ValueError("Pilot plan configuration, code or runtime changed. Prepare a new pilot identity.")
    base = path.parent
    train, held, orth = (checked_member(base, plan["files"][k]) for k in ("train", "eval", "orthography"))
    if data_fingerprint([str(train)]) != plan["training_fingerprint"]:
        raise ValueError("Training slice differs from the frozen pilot")
    out = base / arm
    identity = {"plan_sha256": sha(raw), "arm": arm, "training_fingerprint": plan["training_fingerprint"],
                "algorithm_fingerprint": resolved["arms"][arm]["fingerprint"]}
    with _lock(out / ".fit.lock"):
        immutable(out / "identity.json", format_json(identity))
        algorithm = registry.make("algorithm", resolved["arms"][arm]["spec"])
        normalizer = registry.make("normalizer", resolved["normalizer"])
        model_path, fit_path, result_path = (out / name for name in (ARTIFACT, "fit.json", "metrics.json"))
        if fit_path.exists():
            fit = json.loads(fit_path.read_bytes())
            if fit["identity"] != identity or fit["model_sha256"] != file_identity(model_path)[1]:
                raise ValueError("Published fit metadata does not match the arm model")
            backend = algorithm.load(out, normalizer)
        else:
            start = time.perf_counter()
            backend, info = algorithm.train(TrainContext(
                vocab=resolved["vocab_target"], specials=resolved["specials"], normalizer=normalizer,
                pretokenizer=registry.make("pretokenizer", resolved["pretokenizer"]), word_counts=None,
                shard_paths=[str(train)], out_dir=out, resources={"merge_checkpoint_every": resolved["checkpoint_every"]},
            ))
            if info["canonical_bytes"] != plan["training_canonical_bytes"] or info["documents"] != plan["training_docs"]:
                raise ValueError("Arm exposure differs from the frozen pilot slice")
            fit = {"identity": identity, "model_sha256": file_identity(model_path)[1], "training": info,
                   "fit_attempt_wall_seconds": time.perf_counter() - start,
                   "fit_attempt_process_peak_rss_bytes": peak_rss_bytes(),
                   "primary_48k_eligible": info["occupied_vocab"] == resolved["vocab_target"]}
            immutable(fit_path, format_json(fit))
        if not fit["primary_48k_eligible"]:
            occupied = fit["training"]["occupied_vocab"]
            raise ValueError(f"Scientific arm occupied {occupied} of 48000 vocabulary slots; evaluation is blocked")
        if result_path.exists():
            result = json.loads(result_path.read_bytes())
            fingerprint = result.pop("result_sha256")
            if (result["fit_sha256"] != file_identity(fit_path)[1]
                    or fingerprint != sha(canonical(result))):
                raise ValueError("Existing metrics do not match the completed fit")
            result["result_sha256"] = fingerprint
            return result
        result = {
            "fit_sha256": file_identity(fit_path)[1], "model_sha256": fit["model_sha256"],
            "identity": identity, "training_config": resolved, "fit": fit,
            "evaluation_sha256": plan["files"]["eval"]["sha256"],
            "natural_text": measure(backend, load_rows(held), "domain"),
            "orthography": measure(backend, load_rows(orth), "category"),
        }
        result["result_sha256"] = sha(canonical(result))
        immutable(result_path, format_json(result))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("prepare")
    pre.add_argument("--manifest", required=True, type=Path)
    pre.add_argument("--out", required=True, type=Path)
    pre.add_argument("--train-bytes", type=int, default=4 * 1024**2)
    fit = commands.add_parser("run")
    fit.add_argument("--plan", required=True, type=Path)
    fit.add_argument("--arm", required=True, choices=tuple(ARMS))
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.manifest, args.out, args.train_bytes)
            print(args.out / "plan.json")
        else:
            run(args.plan, args.arm)
            print(args.plan.parent / args.arm / "metrics.json")
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Pilot stopped. {type(exc).__name__}: {exc}\n")
