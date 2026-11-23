"""Prepare blind reviewer sheets from the existing queue and desk evidence."""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe.diagnose import AUDIT, HERE, format_json, immutable, sha


def tsv(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build(out):
    queue = (AUDIT / "morphology-annotation-v1.tsv").read_bytes()
    desk = (HERE / "morphology-desk-review.json").read_bytes()
    measured = (HERE / "review-results-v1.json").read_bytes()
    probe = json.loads(measured)["morphology_desk_review"]
    if probe["input_sha256"] != sha(desk):
        raise ValueError("The observed readings belong to a different desk-review input")
    initial = list(csv.DictReader(io.StringIO(queue.decode()), delimiter="\t"))
    desk_rows = json.loads(desk)["forms"]
    observed = {r["text"]: r for r in probe["results"]}
    blind = [{"id": r["id"], "surface": r["raw_surface"], "category": r["category"], "group": r["group"]}
             for r in initial]
    # Context-bearing desk rows are distinct from the earlier isolated prompts.
    blind += [{"id": f"desk-{i:03d}", "surface": r["text"], "context": r["context"]}
              for i, r in enumerate(desk_rows, 1)]
    out.mkdir(parents=True, exist_ok=True)
    sheet = tsv(("id", "surface", "context", "category", "group", "review_status",
                 "raw_offsets_json", "canonical_offsets_json", "notes"), blind)
    adjudication = tsv(("id", "reviewer_1", "reviewer_2", "decision", "agreed_raw_offsets_json",
                        "agreed_canonical_offsets_json", "family", "reason"), [{"id": r["id"]} for r in blind])
    evidence = format_json({
        "queue_notes": {r["id"]: {"status": r["queue_status"], "prior_offsets": r["prior_raw_offsets"],
                                 "analyses": json.loads(r["mlmorph_nfc_candidates"])}
                        for r in initial},
        "desk_notes": {f"desk-{i:03d}": {"desk": r, "observed": observed[r["text"]]}
                       for i, r in enumerate(desk_rows, 1)},
    })
    files = {"reviewer-1.tsv": sheet, "reviewer-2.tsv": sheet,
             "adjudication.tsv": adjudication, "evidence-after-blind-review.json": evidence}
    for name, data in files.items():
        immutable(out / name, data)
    manifest = {
        "schema": "mom-morphology-review-packet-v2",
        "status": "awaiting two independent human reviews", "authorized_training_labels": 0,
        "forms_to_review": len(blind), "isolated_prompts": len(initial), "contextual_prompts": len(desk_rows),
        "dependencies": probe["dependencies"],
        "transducer_sha256": probe["compiled_transducer_sha256"],
        "transducer_size": probe["compiled_transducer_size"],
        "dependency_lock_sha256": sha((AUDIT / "morphology-dependencies.json").read_bytes()),
        "input_sha256": {"queue": sha(queue), "desk_review": sha(desk), "analyzer_observations": sha(measured)},
        "file_sha256": {name: sha(data) for name, data in files.items()},
        "annotation": "Review independently before opening evidence. Use accepted, ambiguous, or abstain. Enter zero-based codepoint offsets separately for raw and canonical forms. Keep uncertain readings and unresolved cases with a reason. Save filled copies under a new identity.",
    }
    immutable(out / "manifest.json", format_json(manifest))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.out)
    print(args.out / "manifest.json")
