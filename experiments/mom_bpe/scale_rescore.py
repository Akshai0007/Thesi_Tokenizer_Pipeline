"""Rescore saved 8 MB event tokenizers on a common, disjoint development split."""

import argparse
import json
from pathlib import Path

from experiments.mom_bpe.corpus_overlap import DuplicateIndex
from experiments.mom_bpe.diagnose import format_json, immutable, sha
from experiments.mom_bpe.scale_data import code_digest
from experiments.mom_bpe.scale_fit import VOCAB, checked_run, measurements
from experiments.mom_bpe.verify_frozen import load_rows
from mltok.lib.bpe_progress import canonical
from mltok.lib.picky_backend import ARTIFACT, EventBackend
from mltok.snapshots import _lock, file_identity


def disjoint(train_file, validation_file):
    index = DuplicateIndex(":memory:", feature_order=str)
    try:
        for split, path in (("pilot_train", train_file), ("common_validation", validation_file)):
            with index.db:
                for row in load_rows(path):
                    index.insert({**row, "src": split, "id": f"{split}:{row['id']}"}, row["text"])
        if any(row["source_a"] != row["source_b"] for row in index.counts()):
            raise ValueError("The pilot training and common validation splits contain overlapping groups")
    finally:
        index.db.close()


def run(pilot_manifest, target_manifest, fitted, out):
    train, pilot_sha, pilot_paths, pilot_normalizer, _, _ = checked_run(pilot_manifest, allow_large=False)
    target, target_sha, paths, normalizer, _, _ = checked_run(target_manifest, allow_large=False)
    if (train["splits"]["train"]["canonical_bytes"] > 8_000_000
            or train["scope"] != target["scope"]
            or type(pilot_normalizer) is not type(normalizer)):
        raise ValueError("Expected the bounded pilot and an equivalent normalization policy")
    disjoint(pilot_paths["train"][0], paths["validation"][0])
    with _lock(out / ".rescore.lock"):
        for arm in ("T1", "T3", "T5", "T8"):
            model_path = fitted / arm / ARTIFACT
            fit = json.loads((fitted / arm / "fit.json").read_bytes())
            if (fit["occupied_vocab"] != VOCAB or fit["identity"]["arm"] != arm
                    or fit["identity"]["manifest_sha256"] != pilot_sha
                    or file_identity(model_path)[1] != fit["model_sha256"]):
                raise ValueError("Pilot tokenizer fingerprint or training manifest differs")
            identity = {"arm": arm, "train_manifest_sha256": pilot_sha, "evaluation_manifest_sha256": target_sha,
                        "validation_sha256": target["splits"]["validation"]["sha256"],
                        "model_sha256": fit["model_sha256"], "driver_sha256": code_digest(__file__),
                        "pilot_train_development_overlap": "source group, exact hashes and exact five-gram Jaccard checked"}
            path = out / f"{arm}.json"
            if path.exists():
                result = json.loads(path.read_bytes())
                expected = result.pop("result_sha256")
                if result["identity"] != identity or expected != sha(canonical(result)):
                    raise ValueError("Previous pilot rescore differs from its identity")
                continue
            result = {"identity": identity, **measurements(EventBackend.load(fitted / arm), paths["validation"][0],
                                                            normalizer, VOCAB, baseline=False)}
            result["result_sha256"] = sha(canonical(result))
            immutable(path, format_json(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "target", "fitted", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.pilot, args.target, args.fitted, args.out)
