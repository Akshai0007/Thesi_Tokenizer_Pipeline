"""Run one independently resumable, matched 48K tokenizer fit or measurement."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import yaml

from experiments.mom_bpe.diagnose import AUDIT, HERE, format_json, immutable, sha
from experiments.mom_bpe.orthographic_errors import audit
from experiments.mom_bpe.performance import PublishedBaseline, measure
from experiments.mom_bpe.pilot import peak_rss_bytes
from experiments.mom_bpe.scale_data import BUDGETS, code_digest, document_bytes, verified_plan
from experiments.mom_bpe.verify_frozen import load_rows
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import canonical
from mltok.lib.mom_backend import preprocessing, pretokenizer_split
from mltok.lib.orthographic_units import merge_domains
from mltok.lib.picky_backend import ARTIFACT
from mltok.snapshots import _lock, file_identity

VOCAB = 48000
INTERVAL = 1024
DEFAULT_ARMS = ("T1", "T3", "T4", "T5", "T8", "T8-guard-only", "T8-guard-off", "T0")


def checked_run(path, allow_large):
    registry.load_plugins()
    config_bytes = (HERE / "optimization-arms.yaml").read_bytes()
    cfg = yaml.safe_load(config_bytes)
    if not allow_large and json.loads(path.read_bytes())["scale"] != "A":
        raise ValueError("Use a larger-capacity environment and --allow-large for plans B/C")
    spec = cfg["normalizer"]
    normalizer = registry.make("normalizer", spec)
    files, digest = verified_plan(path, normalizer.normalize, registry.fingerprint("normalizer", spec))
    plan = json.loads(path.read_bytes())
    if sha(path.read_bytes()) != digest or (not allow_large and plan["scale"] != "A"):
        raise ValueError("Data plan changed during verification")
    return plan, digest, files, normalizer, cfg, sha(config_bytes)


def growth(manifest, out, *, allow_large=False):
    plan, digest, files, normalizer, cfg, config_digest = checked_run(manifest, allow_large)
    if plan["splits"]["train"]["canonical_bytes"] > BUDGETS["A"]:
        raise ValueError("Word-type profiling retains all merge domains in memory. Give it a bounded A plan.")
    pre = preprocessing(normalizer, registry.make("pretokenizer", cfg["pretokenizer"])).pre_tokenizer
    counts, points, spent = Counter(), [], 0
    budgets = iter((1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000))
    milestone = next(budgets)
    start = perf_counter()
    for row in load_rows(files["train"][0]):
        _, normalized = document_bytes(row, normalizer.normalize)
        counts.update(merge_domains(normalized.decode(), lambda text: pretokenizer_split(pre, text)))
        spent += len(normalized)
        while milestone is not None and spent >= milestone:
            points.append({"threshold": milestone, "canonical_bytes": spent, "merge_domain_types": len(counts)})
            milestone = next(budgets, None)
    result = {"manifest_sha256": digest, "config_sha256": config_digest, "code_sha256": code_digest(__file__),
              "points": points, "canonical_bytes": spent, "merge_domain_types": len(counts),
              "repeated_merge_domain_types": sum(n >= 2 for n in counts.values()),
              "distinct_type_scalar_positions": sum(map(len, counts)), "seconds": perf_counter() - start,
              "peak_rss_bytes": peak_rss_bytes()}
    immutable(out, format_json(result))
    return result


def arm_spec(cfg, arm):
    if arm == "T8-guard-only":
        return {**cfg["arms"]["T8"], "picky_threshold_milli": None}
    if arm == "T8-guard-off":
        return {**cfg["arms"]["T8"], "guard_milli": 0}
    return cfg["arms"][arm]


def measurements(model, validation_path, normalizer, occupied, baseline):
    rows = list(load_rows(validation_path))
    strata = {}
    for key in ("src", "domain"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        strata[key] = {k: measure(model, group, normalizer, occupied) for k, group in sorted(groups.items())}
    probes = json.loads((AUDIT / "fidelity-cases.json").read_bytes())["cases"]
    return {"validation": measure(model, rows, normalizer, occupied), "strata": strata,
            "boundary_attribution": None if baseline else audit(model, rows, example_limit=0),
            "fidelity": measure(model, probes, normalizer, occupied, require_hashes=False)}


def run(manifest, out, arm, *, fit_only=False, allow_large=False):
    plan, digest, files, normalizer, cfg, config_digest = checked_run(manifest, allow_large)
    baseline = arm == "T0"
    spec = None if baseline else arm_spec(cfg, arm)
    model = PublishedBaseline() if baseline else None
    identity = {"arm": arm, "requested_vocab": VOCAB, "manifest_sha256": digest, "config_sha256": config_digest,
                "train_stream_sha256": plan["splits"]["train"]["canonical_stream_sha256"],
                "validation_stream_sha256": plan["splits"]["validation"]["canonical_stream_sha256"],
                "specials": list(DEFAULT_SPECIALS), "fallback_count": 256, "checkpoint_interval": INTERVAL,
                "normalizer": registry.fingerprint("normalizer", cfg["normalizer"]),
                "pretokenizer": registry.fingerprint("pretokenizer", cfg["pretokenizer"]),
                "algorithm": model.identity if baseline else registry.fingerprint("algorithm", spec),
                "code": {name: code_digest(HERE / name) for name in
                         ("scale_fit.py", "scale_data.py", "performance.py", "orthographic_errors.py")},
                "fidelity_panel_sha256": file_identity(AUDIT / "fidelity-cases.json")[1]}
    model_path, fit_path, metric_path = (out / name for name in (ARTIFACT, "fit.json", "validation.json"))
    with _lock(out / ".arm.lock"):
        immutable(out / "identity.json", format_json(identity))
        if baseline:
            fit = {"identity": identity, "occupied_vocab": model.occupied, "historical_reference_only": True}
        else:
            algo = registry.make("algorithm", spec)
            if fit_path.exists():
                fit = json.loads(fit_path.read_bytes())
                if (fit["identity"] != identity or fit["model_sha256"] != file_identity(model_path)[1]):
                    raise ValueError("Existing trained artifact differs from the verified fit")
                model = algo.load(out, normalizer)
            else:
                start = perf_counter()
                model, info = algo.train(TrainContext(
                    VOCAB, DEFAULT_SPECIALS, normalizer, registry.make("pretokenizer", cfg["pretokenizer"]), out, None,
                    list(map(str, files["train"])), {"merge_checkpoint_every": INTERVAL},
                ))
                if (info["canonical_bytes"], info["raw_bytes"], info["documents"]) != tuple(
                        plan["splits"]["train"][key] for key in ("canonical_bytes", "raw_bytes", "rows")):
                    raise ValueError("Fitted exposure differs from the training manifest")
                paths = sorted((out / "event-checkpoints").glob("merge-*.json.gz"))
                checkpoint_files = [{"file": path.name, "bytes": size, "sha256": value}
                                    for path in paths for size, value in [file_identity(path)]]
                fit = {"identity": identity, "occupied_vocab": info["occupied_vocab"], "training": info,
                       "status": "OCCUPIED_48K" if info["occupied_vocab"] == VOCAB else "UNDERFILLED",
                       "seconds": perf_counter() - start, "peak_rss_bytes": peak_rss_bytes(),
                       "model_sha256": file_identity(model_path)[1], "checkpoints": checkpoint_files}
                immutable(fit_path, format_json(fit))
        if fit_only:
            return fit
        if fit["occupied_vocab"] != VOCAB:
            raise ValueError("Tokenizer underfilled 48K. See fit.json for the measured capacity and stop reason.")
        if metric_path.exists():
            result = json.loads(metric_path.read_bytes())
            expected = result.pop("result_sha256")
            if result["fit"] != fit or sha(canonical(result)) != expected:
                raise ValueError("Existing validation receipt differs")
            return {**result, "result_sha256": expected}
        if not baseline:
            if model.special_ids != dict(zip(DEFAULT_SPECIALS, range(len(DEFAULT_SPECIALS)))):
                raise ValueError("Tokenizer control-token IDs differ")
            if model.fallback_ids != frozenset(range(len(DEFAULT_SPECIALS), len(DEFAULT_SPECIALS) + 256)):
                raise ValueError("Tokenizer byte fallback IDs differ")
        result = {**measurements(model, files["validation"][0], normalizer, fit["occupied_vocab"], baseline),
                  "fit": fit, "qualification": plan["qualification"], "scientific_holdout": False,
                  "special_ids": model.special_ids}
        result["result_sha256"] = sha(canonical(result))
        immutable(metric_path, format_json(result))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--arm", choices=DEFAULT_ARMS)
    target.add_argument("--growth", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args()
    if args.growth:
        result = growth(args.manifest, args.out, allow_large=args.allow_large)
    else:
        result = run(args.manifest, args.out, args.arm, fit_only=args.fit_only, allow_large=args.allow_large)
        fit = result.get("fit", result)
        result = {"arm": args.arm, "occupied": fit["occupied_vocab"], "seconds": fit.get("seconds"),
                  "status": fit.get("status"), "validation_tokens": result.get("validation", {}).get("ids")}
    print(json.dumps(result, sort_keys=True))
