"""Materialize a verified ablation or compute its exposure ledger without launching it."""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe.diagnose import HERE, format_json, immutable, sha
from experiments.mom_bpe.verify_frozen import checked_member, load_rows, verify

from mltok import registry
from mltok.lib.exposure import account_exposure
from mltok.lib.mom_backend import ARTIFACT, MOMBackend, preprocessing
from mltok.snapshots import file_identity


def prepare(path, model_dir=None, token_budget=None):
    frozen = verify(path)
    manifest_data = path.read_bytes()
    if sha(manifest_data) != frozen["manifest_sha256"]:
        raise ValueError("Frozen manifest changed after verification")
    manifest = json.loads(manifest_data)
    cfg_bytes = (HERE / "arms.yaml").read_bytes()
    cfg = yaml.safe_load(cfg_bytes)
    alignment = manifest["training_alignments"]
    arms = {}
    for arm, params in cfg["arms"].items():
        if arm == "T0":
            arms[arm] = params
            continue
        params = dict(params)
        spec = dict(cfg["common_algorithm"])
        if params.pop("use_verified_alignment", False):
            spec.update(alignment_file=alignment["path"], alignment_sha256=alignment["sha256"])
        spec.update(params)
        arms[arm] = {"algorithm": spec, "algorithm_fingerprint": registry.fingerprint("algorithm", spec)}
    plan = {
        "freeze": frozen, "arm_config_sha256": sha(cfg_bytes), "vocab": cfg["vocab"],
        "normalizer": cfg["normalizer"], "pretokenizer": cfg["pretokenizer"],
        "normalizer_fingerprint": registry.fingerprint("normalizer", cfg["normalizer"]),
        "pretokenizer_fingerprint": registry.fingerprint("pretokenizer", cfg["pretokenizer"]),
        "metrics": cfg["metrics"], "metric_fingerprints": {m: registry.fingerprint("metric", m) for m in cfg["metrics"]},
        "ordered_shards": manifest["splits"], "arms": arms, "downstream": cfg["downstream"],
        "training_started": False,
    }
    if model_dir is not None:
        model = MOMBackend.load(model_dir)
        norm = registry.make("normalizer", cfg["normalizer"])
        pre = registry.make("pretokenizer", cfg["pretokenizer"])
        if (model.vocab_size != cfg["vocab"] or
                model.artifact["preprocessing"] != json.loads(preprocessing(norm, pre).to_str())):
            raise ValueError("Model vocabulary or preprocessing differs from the matched ablations")
        def rows():
            for entry in manifest["splits"]["train"]:
                yield from load_rows(checked_member(path.parent, entry))

        plan["exposure"] = {
            "model_sha256": file_identity(model_dir / ARTIFACT)[1],
            "fixed_source_bytes": account_exposure(model, rows()),
        }
        if token_budget is not None:
            plan["exposure"]["fixed_training_tokens"] = account_exposure(model, rows(), token_budget)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--predicted-tokens", type=int)
    args = parser.parse_args()
    if args.predicted_tokens is not None and args.model_dir is None:
        parser.error("--predicted-tokens needs --model-dir")
    try:
        plan = prepare(args.manifest.resolve(), args.model_dir, args.predicted_tokens)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        immutable(args.out, format_json(plan))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Plan blocked. {type(exc).__name__}: {exc}\n")
    print(json.dumps({"manifest": plan["freeze"]["manifest_sha256"], "training_started": False}))
