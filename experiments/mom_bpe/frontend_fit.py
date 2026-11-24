"""Run a 48K frontend comparison with the existing guarded event learner."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import yaml

from experiments.mom_bpe.diagnose import HERE, format_json, immutable, sha
from experiments.mom_bpe.pilot import peak_rss_bytes
from experiments.mom_bpe.scale_data import code_digest
from experiments.mom_bpe.scale_fit import INTERVAL, VOCAB, checked_run, measurements
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import canonical
from mltok.lib.picky_backend import ARTIFACT
from mltok.snapshots import _lock, file_identity


def pretokenizer_spec(arm):
    data = (HERE / "frontend-arms.yaml").read_bytes()
    choice = yaml.safe_load(data)["arms"][arm]
    base = choice["base"]
    if not choice["attach_space"]:
        return base, choice, sha(data)
    native = json.loads(registry.make("pretokenizer", base).hf().__getstate__())
    if native["type"] != "Split" or native["behavior"] != "Isolated" or native["invert"]:
        raise ValueError("Leading-space frontend requires an isolated native partition")
    return {"type": "leading_space", "pattern": native["pattern"]["Regex"]}, choice, sha(data)


def run(manifest, out, arm):
    plan, digest, files, normalizer, cfg, config_digest = checked_run(manifest, allow_large=False)
    pre_spec, choice, arm_digest = pretokenizer_spec(arm)
    algo_spec = cfg["arms"]["T8"]
    algo = registry.make("algorithm", algo_spec)
    identity = {"arm": arm, "manifest_sha256": digest, "frontend_config_sha256": arm_digest,
                "frontend": choice, "config_sha256": config_digest, "requested_vocab": VOCAB,
                "validation_stream_sha256": plan["splits"]["validation"]["canonical_stream_sha256"],
                "train_stream_sha256": plan["splits"]["train"]["canonical_stream_sha256"],
                "specials": list(DEFAULT_SPECIALS), "fallback_count": 256, "checkpoint_interval": INTERVAL,
                "alg_spec": algo_spec, "algorithm": registry.fingerprint("algorithm", algo_spec),
                "normalizer": registry.fingerprint("normalizer", cfg["normalizer"]),
                "base_pretokenizer": registry.fingerprint("pretokenizer", choice["base"]),
                "effective_pretokenizer": registry.fingerprint("pretokenizer", pre_spec),
                "code_sha256": code_digest(__file__)}
    model_path, fit_path, validation_path = (out / name for name in (ARTIFACT, "fit.json", "validation.json"))
    with _lock(out / ".arm.lock"):
        immutable(out / "identity.json", format_json(identity))
        if fit_path.exists():
            fit = json.loads(fit_path.read_bytes())
            if fit["identity"] != identity or fit["model_sha256"] != file_identity(model_path)[1]:
                raise ValueError("Completed frontend fit does not match its identity")
            model = algo.load(out, normalizer)
        else:
            start = perf_counter()
            model, info = algo.train(TrainContext(
                VOCAB, DEFAULT_SPECIALS, normalizer, registry.make("pretokenizer", pre_spec), out, None,
                list(map(str, files["train"])), {"merge_checkpoint_every": INTERVAL},
            ))
            actual = tuple(info[k] for k in ("canonical_bytes", "raw_bytes", "documents"))
            expected = tuple(plan["splits"]["train"][key] for key in ("canonical_bytes", "raw_bytes", "rows"))
            if actual != expected:
                raise ValueError("Frontend received a different training stream")
            fit = {"identity": identity, "training": info, "occupied_vocab": info["occupied_vocab"],
                   "model_sha256": file_identity(model_path)[1],
                   "seconds": perf_counter() - start, "peak_rss_bytes": peak_rss_bytes()}
            immutable(fit_path, format_json(fit))
        if fit["occupied_vocab"] != VOCAB or model.special_ids != dict(zip(DEFAULT_SPECIALS, range(len(DEFAULT_SPECIALS)))):
            raise ValueError("Frontend fit must occupy 48K with the shared control IDs")
        if validation_path.exists():
            report = json.loads(validation_path.read_bytes())
            expected = report.pop("result_sha256")
            if report["fit"] != fit or sha(canonical(report)) != expected:
                raise ValueError("Existing frontend evaluation failed verification")
            return {**report, "result_sha256": expected}
        report = {**measurements(model, files["validation"][0], normalizer, VOCAB, baseline=False),
                  "fit": fit, "qualification": plan["qualification"], "scientific_holdout": False}
        if report["validation"]["canonical_roundtrip"] != report["validation"]["docs"]:
            raise ValueError("Frontend model failed canonical round-trip")
        report["result_sha256"] = sha(canonical(report))
        immutable(validation_path, format_json(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("F-word", "F-space", "F-word-space"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.manifest, args.out, args.arm)
    print(json.dumps({"arm": args.arm, "tokens": report["validation"]["ids"],
                      "seconds": report["fit"]["seconds"], "peak_rss_bytes": report["fit"]["peak_rss_bytes"]}))
