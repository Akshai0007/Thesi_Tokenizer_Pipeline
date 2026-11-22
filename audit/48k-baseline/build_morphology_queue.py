"""Build the unadjudicated annotation queue with pinned analyzer suggestions."""
import csv
import hashlib
import importlib.metadata
import io
import json
import sys
import unicodedata
from collections import Counter
from itertools import accumulate
from pathlib import Path

from mlmorph import Analyser

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from mltok.lib.atomic import atomic_writer

HUMAN_FIELDS = (
    "reviewer_1_context", "reviewer_1_raw_offsets", "reviewer_1_nfc_offsets",
    "reviewer_2_context", "reviewer_2_raw_offsets", "reviewer_2_nfc_offsets",
    "adjudicated_status", "adjudicated_raw_offsets", "adjudicated_nfc_offsets", "notes",
)


def compact(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def digest(content):
    return hashlib.sha256(content).hexdigest()


def rows(old, seeds):
    for index, item in enumerate(old["morphology"], 1):
        parts = item["parts"]
        surface = item.get("word", "".join(parts))
        alignment = item.get("alignment", "manual_surface_fixture")
        offsets = None
        if "".join(parts) == surface and alignment != "unscored_sandhi":
            offsets = list(accumulate(map(len, parts)))[:-1]
        yield {
            "id": f"prior-{index:03d}", "category": item["type"], "group": item["group"],
            "raw_surface": surface, "prior_alignment": alignment, "prior_parts": compact(parts),
            "prior_raw_offsets": "" if offsets is None else compact(offsets),
        }
    index = 0
    for category, groups in seeds["groups"].items():
        for group, surfaces in groups.items():
            for surface in surfaces:
                index += 1
                yield {"id": f"seed-{index:03d}", "category": category, "group": group,
                       "raw_surface": surface}


def save(filename, content):
    target = HERE / filename
    if target.exists():
        if target.read_bytes() != content:
            raise SystemExit(f"{filename} differs. Use a new version for changed inputs or annotations.")
        return
    with atomic_writer(target) as stream:
        stream.write(content)


def main():
    old = (HERE / "fixtures.json").read_bytes()
    seeds = (HERE / "morphology-candidates.json").read_bytes()
    lock = json.loads((HERE / "morphology-dependencies.json").read_bytes())
    versions = {a["project"]: importlib.metadata.version(a["project"]) for a in lock["artifacts"]}
    if any(versions[a["project"]] != a["version"] for a in lock["artifacts"]):
        raise SystemExit("Morphology analyzer dependencies differ from the pinned versions")
    analyzer = Analyser()
    counts, categories, surfaces = Counter(), Counter(), set()
    with io.StringIO(newline="") as output:
        writer = csv.DictWriter(output, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_ALL, fieldnames=(
            "id", "category", "group", "raw_surface", "nfc_surface", "prior_alignment", "prior_parts",
            "prior_raw_offsets", "mlmorph_nfc_candidates", "mlmorph_nfc_candidate_count",
            "queue_status", *HUMAN_FIELDS,
        ))
        writer.writeheader()
        for row in rows(json.loads(old), json.loads(seeds)):
            raw = row["raw_surface"]
            if raw in surfaces:
                raise SystemExit(f"Duplicate review seed at {row['id']}")
            surfaces.add(raw)
            nfc = unicodedata.normalize("NFC", raw)
            suggestions = sorted(analyzer.analyse(nfc) or [], key=lambda pair: (pair[1], pair[0]))
            status = ("alignment_unresolved" if row.get("prior_alignment") == "unscored_sandhi"
                      else "multiple_analyzer_candidates" if len(suggestions) > 1
                      else "no_analyzer_candidate" if not suggestions else "review_required")
            row.update(nfc_surface=nfc, mlmorph_nfc_candidates=compact(suggestions[:5]),
                       mlmorph_nfc_candidate_count=len(suggestions), queue_status=status)
            writer.writerow(row)
            counts["prior_forms" if row["id"].startswith("prior-") else "new_forms"] += 1
            counts["forms_with_analyzer_candidates"] += bool(suggestions)
            counts["forms_with_multiple_analyzer_candidates"] += len(suggestions) > 1
            counts[status] += 1
            if row.get("prior_alignment") == "manual_surface_fixture":
                counts["prior_manual_offsets_carried_forward"] += len(json.loads(row["prior_raw_offsets"]))
            if row["id"].startswith("seed-"):
                categories[row["category"]] += 1
        queue = output.getvalue().encode("utf-8")
    if counts["prior_manual_offsets_carried_forward"] != 16:
        raise SystemExit("Historical tentative boundary count changed")
    summary = {
        "status": "awaiting independent human annotation",
        "human_verified_forms": 0, "human_verified_boundaries": 0,
        "counts": dict(counts), "new_forms_by_category": dict(categories),
        "analyzer_versions": versions, "unicode_version": unicodedata.unidata_version,
        "inputs": {"fixtures_sha256": digest(old), "candidates_sha256": digest(seeds)},
        "queue_sha256": digest(queue),
        "analyzer_fields": "First five ranked NFC analyses. Weights are ranks, not probabilities or surface offsets.",
    }
    save("morphology-annotation-v1.tsv", queue)
    save("morphology-annotation-v1.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
