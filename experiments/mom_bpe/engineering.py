"""Fit and resume one explicitly diagnostic CPU arm on repository-authored fixtures."""

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe.diagnose import AUDIT, TRAIN, format_json, immutable, sha
from experiments.mom_bpe.pilot import ARMS, configuration, measure, peak_rss_bytes
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_progress import BPEProgress, canonical
from mltok.lib.io import data_fingerprint
from mltok.lib.mom_backend import ARTIFACT
from mltok.lib.orthographic_units import UNITIZERS, merge_domains
from mltok.snapshots import _lock, file_identity

LABEL = "ENGINEERING_DIAGNOSTIC_ONLY"
VOCAB = 288


class PlannedInterruption(Exception):
    pass


def fallback_attribution(backend, train, held, pre, segmenter):
    unitize = UNITIZERS[segmenter]

    def units(text):
        for span in merge_domains(backend.expected(text), pre.split):
            yield from unitize(span)

    known = set(backend.artifact["initial"])
    seen = {unit for text in train for unit in units(text)}
    result = {}
    for row in held:
        domain = row["domain"]
        stats = result.setdefault(domain, {"absent_unit_bytes": 0, "pruned_unit_bytes": 0, "observed_byte_ids": 0})
        predicted = 0
        for unit in units(row["text"]):
            if unit in known:
                continue
            size = len(unit.encode())
            predicted += size
            key = "pruned_unit_bytes" if unit in seen else "absent_unit_bytes"
            stats[key] += size
        observed = sum(i in backend.fallback_ids for i in backend.encode_offsets(row["text"]).ids)
        if predicted != observed:
            raise ValueError("Observed fallback did not match the independent unit attribution")
        stats["observed_byte_ids"] += observed
    return result


def run(out, arm):
    cfg = configuration()
    arm_spec = cfg["arms"][arm]
    algo = registry.make("algorithm", arm_spec["spec"])
    norm = registry.make("normalizer", cfg["normalizer"])
    pre = registry.make("pretokenizer", cfg["pretokenizer"])
    out.mkdir(parents=True, exist_ok=True)
    authored = b"".join(canonical({"text": t, "src": "authored_fixture"}) + b"\n" for t in TRAIN)
    if sum(len(norm.normalize(t).encode()) for t in TRAIN) > 4 * 1024**2:
        raise ValueError("Engineering input exceeds the fixture budget")
    source = out / "authored-train.jsonl"
    immutable(source, authored)
    fixture_bytes = (AUDIT / "fixtures.json").read_bytes()
    fidelity_bytes = (AUDIT / "fidelity-cases.json").read_bytes()
    fixture = json.loads(fixture_bytes)
    held = [{"text": text, "domain": name} for name, texts in fixture["domains"].items() for text in texts]
    orth = [{"text": text, "category": name} for name, text in fixture["unicode_cases"]]
    fidelity = [{"text": row["text"], "category": row["id"]} for row in json.loads(fidelity_bytes)["cases"]]
    identity = {
        "scope": LABEL, "arm": arm, "configuration": arm_spec, "target_vocab": VOCAB,
        "normalizer": cfg["normalizer_fingerprint"], "pretokenizer": cfg["pretokenizer_fingerprint"],
        "metric": cfg["metric_fingerprint"], "driver_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "training_fingerprint": data_fingerprint([str(source)]),
        "fixtures_sha256": sha(fixture_bytes), "fidelity_sha256": sha(fidelity_bytes),
    }
    root = out / arm
    with _lock(root / ".engineering.lock"):
        immutable(root / "identity.json", format_json(identity))
        result_path = root / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_bytes())
            if (result["identity"] != identity or any(
                file_identity(root / name / ARTIFACT)[1] != result[name]["model_sha256"]
                for name in ("reference", "resumed")
            )):
                raise ValueError("Published engineering results no longer match their inputs")
            return result

        def fit(name):
            directory = root / name
            start = time.perf_counter()
            backend, info = algo.train(TrainContext(
                vocab=VOCAB, specials=cfg["specials"], normalizer=norm, pretokenizer=pre, word_counts=None,
                shard_paths=[str(source)], out_dir=directory, resources={"merge_checkpoint_every": 1}, log=lambda *_: None,
            ))
            return backend, {
                "scope": LABEL, "model_sha256": file_identity(directory / ARTIFACT)[1], "training": info,
                "fit_wall_seconds": time.perf_counter() - start,
                "process_peak_rss_bytes": peak_rss_bytes(),
                "checkpoint_generations": len(list((directory / "merge-checkpoints").glob("merge-*.json.gz"))),
            }

        reference, reference_info = fit("reference")
        save = BPEProgress.save
        pause = root / "interruption.json"
        if not pause.exists():
            def stop_after_save(self, merges, seqs, flags):
                save(self, merges, seqs, flags)
                if len(merges) == 1:
                    immutable(pause, format_json({"scope": LABEL, "verified_merge_count": 1}))
                    raise PlannedInterruption

            with patch.object(BPEProgress, "save", stop_after_save):
                try:
                    fit("resumed")
                except PlannedInterruption:
                    pass
            if not pause.exists():
                raise ValueError("Engineering run failed to exercise an interrupted merge")
        resumed, resumed_info = fit("resumed")
        loaded = algo.load(root / "resumed", norm)
        texts = [row["text"] for row in held + orth + fidelity]
        if (reference_info["model_sha256"] != resumed_info["model_sha256"]
                or not resumed_info["training"]["resumed_merges"]
                or reference.encode_batch(texts) != resumed.encode_batch(texts)
                or resumed.encode_batch(texts) != loaded.encode_batch(texts)):
            raise ValueError("Interrupted, uninterrupted, and reloaded encoders differ")
        result = {
            "scope": LABEL, "identity": identity, "resume_and_reload_exact": True,
            "reference": reference_info, "resumed": resumed_info,
            "natural_probes": measure(loaded, held, "domain"), "orthography": measure(loaded, orth, "category"),
            "fidelity": measure(loaded, fidelity, "category"),
            "fallback_attribution": fallback_attribution(loaded, TRAIN, held, pre, arm_spec["spec"]["segmenter"]),
        }
        immutable(result_path, format_json(result))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    args = parser.parse_args()
    result = run(args.out, args.arm)
    print(json.dumps({"scope": result["scope"], "arm": args.arm,
                      "occupied": result["resumed"]["training"]["occupied_vocab"],
                      "resume_exact": result["resume_and_reload_exact"]}))
