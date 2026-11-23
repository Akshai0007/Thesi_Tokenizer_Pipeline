"""Review the archived fallback counts and probe the pinned analyzer without training."""

import argparse
import importlib.metadata
import json
import sys
from collections import Counter
from importlib import resources
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe.diagnose import AUDIT, HERE, TRAIN, format_json, immutable, sha
from mltok import registry
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.morph_alignment import project, reviewed_mlmorph_candidates, surface_boundaries
from mltok.lib.orthographic_units import UNITIZERS, merge_domains


def fallback_probe():
    registry.load_plugins()
    normalizer = registry.make("normalizer", "ml_conservative").hf()
    pre = registry.make("pretokenizer", "script_runs")
    fixtures = json.loads((AUDIT / "fixtures.json").read_bytes())["domains"]
    held = [text for rows in fixtures.values() for text in rows]
    original = json.loads((HERE / "diagnostic-results-v1.json").read_bytes())
    by_mode = {}
    for name, arm in (("codepoint", "T0-matched"), ("grapheme", "T1"), ("akshara", "T2")):
        split = UNITIZERS[name]

        def unit_counts(texts):
            return Counter(u for text in texts for span in merge_domains(normalizer.normalize_str(text), pre.split)
                           for u in split(span))

        train = unit_counts(TRAIN)
        eval_ = unit_counts(held)
        ranked = sorted(train, key=lambda word: (-train[word], word))
        original_ids = original["arms"][arm]["ordinary"]["mom_byte_fallback_ids"]

        def fallback_bytes(units):
            return sum(len(u.encode()) * n for u, n in eval_.items() if u not in units)

        available = 384 - len(DEFAULT_SPECIALS) - 256
        cutoff, mass = [], 0
        total = sum(train.values())
        for unit in ranked:
            if len(cutoff) >= available // 2 or mass >= 0.99999 * total:
                break
            cutoff.append(unit)
            mass += train[unit]
        predicted = fallback_bytes(set(cutoff))
        unseen = fallback_bytes(train)
        if predicted != original_ids:
            raise ValueError("Fallback attribution failed to match the archived diagnostics")
        by_mode[arm] = {"train_units": len(train), "evaluation_units": len(eval_), "selected_units": len(cutoff),
                        "fallback_ids": predicted, "unseen_unit_utf8_bytes": unseen,
                        "seen_but_excluded_unit_utf8_bytes": predicted - unseen,
                        "fallback_if_all_seen_units_were_selected": unseen}
    return by_mode


def morphology_probe():
    from mlmorph import Analyser, Generator

    dependency_lock = json.loads((AUDIT / "morphology-dependencies.json").read_bytes())
    versions = {row["project"]: importlib.metadata.version(row["project"]) for row in dependency_lock["artifacts"]}
    if any(row["version"] != versions[row["project"]] for row in dependency_lock["artifacts"]):
        raise ValueError("Use the pinned Mlmorph and SFST wheels")
    fst = resources.files("mlmorph").joinpath("data", "malayalam.a").read_bytes()
    if sha(fst) != "f6eb3d3fb73fdc7e1f422d5a0b14e20e33fc736d4a830e4395213577611dfa72":
        raise ValueError("Mlmorph transducer SHA-256 changed")
    review = (HERE / "morphology-desk-review.json").read_bytes()
    analyzer, generator = Analyser(), Generator()
    units = UNITIZERS["akshara"]
    results = []
    for entry in json.loads(review)["forms"]:
        word = entry["text"]
        rows = analyzer.analyse(word, foreign_word_check=False)
        candidates = [{"reading": reading, "rank": rank, "structured": analyzer.parse_analysis(reading)["morphemes"]}
                      for reading, rank in rows]
        expected_reading = entry.get("reading")
        expected_offsets = entry["offsets"]
        reproduced = None
        offsets = None
        if expected_reading:
            if expected_reading not in {x["reading"] for x in candidates}:
                raise ValueError("Reviewed analysis was absent from the pinned transducer")
            reproduced = word in generator.generate(expected_reading, weighted=False)
            if not reproduced:
                raise ValueError("Reviewed reading did not regenerate the exact surface")
            parts = entry["parts"]
            projected = surface_boundaries(word, parts, entry.get("surface_parts", parts), entry["operation"], units)
            offsets = list(projected) if projected is not None else None
        production = project(word, reviewed_mlmorph_candidates(word, analyzer, [entry] if expected_reading else []), units)
        if production:
            raise ValueError("Desk review improperly authorized a training boundary")
        results.append({"text": word, "candidates": candidates, "generated_exact_surface": reproduced,
                        "expected_offsets": expected_offsets, "observed_offsets": offsets,
                        "training_boundary_offsets": list(production), "agreed": offsets == expected_offsets})
    return {"dependencies": versions, "compiled_transducer_size": len(fst), "compiled_transducer_sha256": sha(fst),
            "input_sha256": sha(review), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"prototype_fallback_attribution": fallback_probe(), "morphology_desk_review": morphology_probe()}
    if not all(row["agreed"] for row in result["morphology_desk_review"]["results"]):
        raise SystemExit("Unexpected surface projection in the desk review")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    immutable(args.out, format_json(result))
    print(args.out)
