"""Create a whole-document scale-A subset and verify paired development counts."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

from experiments.mom_bpe.diagnose import format_json, immutable, sha
from experiments.mom_bpe.performance import PublishedBaseline, checked_canonical
from experiments.mom_bpe.scale_data import CAPACITY, code_digest, publish
from experiments.mom_bpe.scale_fit import VOCAB, checked_run
from experiments.mom_bpe.verify_frozen import load_rows
from mltok.lib.bpe_progress import canonical
from mltok.lib.picky_backend import ARTIFACT, EventBackend
from mltok.snapshots import _lock, file_identity


def prefix(rows, budget):
    for row in rows:
        budget -= row["canonical_bytes"]
        if budget < 0:
            break
        yield row


def subset(manifest, out, budget):
    parent, digest, files, normalizer, _, _ = checked_run(manifest, allow_large=False)
    if parent["qualification"] != CAPACITY or not 0 < budget < parent["splits"]["train"]["canonical_bytes"]:
        raise ValueError("Expected a strict subset of a verified scale-A capacity plan")
    recipe = {"qualification": CAPACITY, "scale": "A", "canonical_target": budget,
              "scope": parent["scope"], "parent_manifest_sha256": digest,
              "selection": "whole-document ordered prefix", "subset_code_sha256": code_digest(__file__)}
    with _lock(out / ".prepare.lock"):
        immutable(out / "recipe.json", format_json(recipe))

        def rows(split):
            stream = load_rows(files[split][0])
            return prefix(stream, budget) if split == "train" else stream

        result = publish(out, recipe, rows, normalizer.normalize)
        if result["splits"]["validation"] != parent["splits"]["validation"]:
            raise ValueError("Subset changed the validation split")
    return result


def paired(manifest, reference, challenger, out):
    plan, digest, files, normalizer, _, _ = checked_run(manifest, allow_large=False)
    models, receipts = [], []
    for directory in (reference, challenger):
        saved = json.loads((directory / "validation.json").read_bytes())
        receipt = saved.pop("result_sha256")
        fit = saved["fit"]
        identity = fit["identity"]
        if (sha(canonical(saved)) != receipt or identity["manifest_sha256"] != digest
                or identity["validation_stream_sha256"] != plan["splits"]["validation"]["canonical_stream_sha256"]
                or fit["occupied_vocab"] != VOCAB):
            raise ValueError("Expected a verified 48K result on the common validation manifest")
        if identity["arm"] == "T0":
            model = PublishedBaseline()
            if model.identity != identity["algorithm"]:
                raise ValueError("Historical baseline fingerprint differs")
        else:
            if file_identity(directory / ARTIFACT)[1] != fit["model_sha256"]:
                raise ValueError("Tokenizer artifact SHA-256 differs")
            model = EventBackend.load(directory)
        models.append(model)
        receipts.append((identity["arm"], receipt, saved["validation"]["ids"]))
    differences, local = [], []
    for row in load_rows(files["validation"][0]):
        source = checked_canonical(row, normalizer)
        a, b = [len(model.encode_offsets(source).ids) for model in models]
        differences.append(b - a)
        local.append({"document_sha256": row["sha256"], "canonical_bytes": row["canonical_bytes"],
                      "reference_tokens": a, "challenger_tokens": b, "difference": b - a})
    expected = [r[2] for r in receipts]
    if not local or [sum(r[k] for r in local) for k in ("reference_tokens", "challenger_tokens")] != expected:
        raise ValueError("Recomputed document counts differ from validation receipts")
    report = {"manifest_sha256": digest, "validation_sha256": plan["splits"]["validation"]["sha256"],
              "results": [{"arm": a, "result_sha256": h, "tokens": n} for a, h, n in receipts],
              "documents": len(local), "difference_sum": sum(differences), "difference_median": median(differences),
              "difference_min": min(differences), "difference_max": max(differences),
              "fewer": sum(d < 0 for d in differences), "more": sum(d > 0 for d in differences),
              "tied": differences.count(0), "histogram": dict(sorted(Counter(map(str, differences)).items()))}
    with _lock(out / ".paired.lock"):
        immutable(out / "per-document.json", format_json(local))
        immutable(out / "summary.json", format_json(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make_subset = commands.add_parser("subset")
    make_subset.add_argument("--bytes", required=True, type=int)
    compare = commands.add_parser("paired")
    compare.add_argument("--reference", required=True, type=Path)
    compare.add_argument("--challenger", required=True, type=Path)
    for command in (make_subset, compare):
        command.add_argument("--manifest", required=True, type=Path)
        command.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "subset":
        output = subset(args.manifest, args.out, args.bytes)
    else:
        output = paired(args.manifest, args.reference, args.challenger, args.out)
    print(json.dumps(output, sort_keys=True))
