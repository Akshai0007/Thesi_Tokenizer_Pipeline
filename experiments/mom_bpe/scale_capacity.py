"""Profile initial merge opportunities before reserving resources for a 48K fit."""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from experiments.mom_bpe.diagnose import format_json, immutable
from experiments.mom_bpe.pilot import peak_rss_bytes
from experiments.mom_bpe.scale_data import BUDGETS, code_digest, document_bytes
from experiments.mom_bpe.scale_fit import VOCAB, checked_run
from experiments.mom_bpe.verify_frozen import load_rows
from mltok import registry
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.hybrid_units import initial_units, select_alphabet
from mltok.lib.mom_backend import preprocessing, pretokenizer_split
from mltok.lib.orthographic_units import merge_domains
from mltok.snapshots import _lock


def pair_profile(counts, unitize, alphabet):
    """Count raw initial pairs that satisfy the trainer's frequency floor."""
    initial = set(alphabet)
    pairs = Counter()
    weighted_unknown_bytes = pair_type_postings = known_unit_positions = 0
    for piece, frequency in counts.items():
        units = unitize(piece)
        candidates = Counter((a, b) for a, b in zip(units, units[1:]) if a in initial and b in initial)
        for pair, multiplicity in candidates.items():
            pairs[pair] += frequency * multiplicity
        pair_type_postings += len(candidates)
        known_unit_positions += sum(u in initial for u in units)
        weighted_unknown_bytes += frequency * sum(len(u.encode()) for u in units if u not in initial)
    floor = [pair for pair, n in pairs.items() if n >= 2]
    return {"initial_distinct_pairs": len(pairs), "initial_pairs_frequency_ge_2": len(floor),
            "distinct_novel_concatenations_frequency_ge_2": len({a + b for a, b in floor if a + b not in initial}),
            "pair_type_postings": pair_type_postings, "known_unit_positions_in_types": known_unit_positions,
            "fallback_bytes_in_merge_domains": weighted_unknown_bytes}


def run(manifest, out):
    plan, digest, files, normalizer, cfg, config_digest = checked_run(manifest, allow_large=False)
    if plan["splits"]["train"]["canonical_bytes"] > BUDGETS["A"]:
        raise ValueError("The initial pair profile is limited to scale A")
    identity = {"manifest_sha256": digest, "config_sha256": config_digest, "code_sha256": code_digest(__file__),
                "training_stream_sha256": plan["splits"]["train"]["canonical_stream_sha256"],
                "algorithms": {arm: registry.fingerprint("algorithm", cfg["arms"][arm])
                               for arm in ("T1", "T3", "T5", "T8")}}
    with _lock(out.parent / ".capacity-profile.lock"):
        if out.exists():
            report = json.loads(out.read_bytes())
            if report["identity"] != identity:
                raise ValueError("Existing profile uses a different data or code identity")
            return report
        start = perf_counter()
        pre = preprocessing(normalizer, registry.make("pretokenizer", cfg["pretokenizer"])).pre_tokenizer
        counts, points, spent = Counter(), [], 0
        marks = iter((1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000))
        mark = next(marks)
        for row in load_rows(files["train"][0]):
            _, normalized = document_bytes(row, normalizer.normalize)
            counts.update(merge_domains(normalized.decode(), lambda text: pretokenizer_split(pre, text)))
            spent += len(normalized)
            while mark is not None and spent >= mark:
                points.append({"threshold": mark, "canonical_bytes": spent, "merge_domain_types": len(counts)})
                mark = next(marks, None)
        lexical = VOCAB - len(DEFAULT_SPECIALS) - 256
        by_unitization = {}
        for arm in ("T1", "T3", "T5"):
            spec = cfg["arms"][arm]
            segmenter = spec["segmenter"]
            alphabet, atoms, coverage = select_alphabet(counts, segmenter, lexical // 2, spec["alphabet_coverage"],
                                                         spec["hybrid_min_count"], spec["hybrid_max_atoms"])
            by_unitization[segmenter] = {**pair_profile(counts, initial_units(segmenter, atoms), alphabet),
                                         "alphabet": len(alphabet), "protected_atoms": len(atoms),
                                         "initial_unit_coverage": coverage, "free_lexical_slots": lexical - len(alphabet)}
        report = {"identity": identity, "canonical_bytes": spent, "target_shortfall": plan["canonical_shortfall"],
                  "word_type_growth": points, "merge_domain_types": len(counts),
                  "repeated_merge_domain_types": sum(n >= 2 for n in counts.values()), "by_unitization": by_unitization,
                  "arms": {"T1": "codepoint", "T3": "akshara", "T5": "hybrid", "T8": "hybrid"},
                  "interpretation": "Initial pair counts precede the first event and do not prove final occupancy.",
                  "seconds": perf_counter() - start, "peak_rss_bytes": peak_rss_bytes()}
        immutable(out, format_json(report))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.out), sort_keys=True))
