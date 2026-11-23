"""Scoring helpers for canonical surface offsets and reviewed morphology."""

from itertools import combinations


def exact_boundaries(encoding, fallback_ids):
    for i, ((start, pos), (following, end)) in enumerate(zip(encoding.offsets, encoding.offsets[1:])):
        if 0 <= start < pos == following < end <= len(encoding.text):
            if encoding.ids[i] not in fallback_ids and encoding.ids[i + 1] not in fallback_ids:
                yield pos


def score_morphology(encode, fallbacks, reference_forms):
    """Use reviewed forms only; measure consistency across repeated surface stems."""
    total = gold = predicted = hits = unmeasurable = 0
    stems = {}
    for row in reference_forms:
        if row["status"] != "adjudicated":
            continue
        total += 1
        surface = row["text"]
        refs = set(row["offsets"])
        gold += len(refs)
        enc = encode(surface)
        prefix = None
        end = row.get("stem_end")
        if enc.text != surface or fallbacks.intersection(enc.ids):
            unmeasurable += 1
        else:
            cuts = set(exact_boundaries(enc, fallbacks))
            predicted += len(cuts)
            hits += len(refs & cuts)
            if end is not None and (end in cuts or end == len(surface)):
                prefix = tuple(i for i, (_, right) in zip(enc.ids, enc.offsets) if right <= end)
        if end:
            stems.setdefault((row["group"], surface[:end]), []).append(prefix)
    pairs = list(pair for reps in stems.values() for pair in combinations(reps, 2))
    both = [(a, b) for a, b in pairs if a is not None and b is not None]
    return {
        "adjudicated_forms": total, "unmeasurable_forms": unmeasurable, "gold_boundaries": gold,
        "predicted_boundaries": predicted, "matched_boundaries": hits,
        "micro_precision": hits / predicted if predicted else None,
        "micro_recall": hits / gold if gold else None,
        "micro_f1": 2 * hits / (gold + predicted) if gold + predicted else None,
        "repeated_stem_pairs": len(pairs), "isolatable_stem_pairs": len(both),
        "consistent_stem_pairs": sum(a == b for a, b in both),
    }
