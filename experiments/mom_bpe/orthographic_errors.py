"""Attribute canonical token boundaries, including verified byte-backed cuts."""

import unicodedata
from bisect import bisect_right
from collections import Counter
from itertools import accumulate, groupby

from mltok.lib.mom_backend import pretokenizer_split
from mltok.lib.orthographic_units import UNITIZERS, merge_domains
from mltok.lib.unicode_ml import CHILLU, ML_SET, VIRAMA, VOWEL_SIGNS, ZWJ, ZWNJ

JOINERS = {ZWJ, ZWNJ}
CHILLUS = set(CHILLU) | {"\u0d54", "\u0d55", "\u0d56"}
VOWELS = set(VOWEL_SIGNS)


def exact_segments(backend, encoding):
    """Recover scalar spans for contiguous byte IDs, validating UTF-8 exactly."""
    first_byte = min(backend.fallback_ids)
    spans, continuation = [], 0
    for (fallback, bounds), pairs in groupby(
        zip(encoding.ids, encoding.offsets), key=lambda pair: (pair[0] in backend.fallback_ids, pair[1]),
    ):
        left, right = bounds
        ids = [value for value, _ in pairs]
        surface = encoding.text[left:right]
        if not (0 <= left < right <= len(encoding.text) and left == (spans[-1][1] if spans else 0)):
            raise ValueError("Encoder offsets are not contiguous")
        if fallback:
            payload = bytes(token - first_byte for token in ids)
            if payload != surface.encode():
                raise ValueError("Byte offsets do not reproduce the canonical surface span")
            spans.extend((i, i + 1, True) for i in range(left, right))
            continuation += len(ids) - (right - left)
        else:
            if len(ids) != 1 or backend.decode(ids) != surface:
                raise ValueError("Lexical token disagrees with its canonical span")
            spans.append((left, right, False))
    if len(encoding.text) != (spans[-1][1] if spans else 0):
        raise ValueError("Encoder offsets missed canonical text")
    return spans, continuation


def primary_pattern(left, right):
    if left in JOINERS or right in JOINERS:
        return "joiner"
    if VIRAMA in (left, right):
        return "virama_or_conjunct"
    if right in VOWELS:
        return "dependent_vowel"
    if left in CHILLUS or right in CHILLUS:
        return "chillu"
    if unicodedata.category(right).startswith("M"):
        return "combining_mark"
    return "other"


def nearby_malayalam(text, site):
    def neighbor(index, step):
        while 0 <= index < len(text) and text[index] in JOINERS:
            index += step
        return 0 <= index < len(text) and text[index] in ML_SET

    return neighbor(site - 1, -1) and neighbor(site, 1)


def audit(backend, rows, example_limit=3):
    totals, primary, mechanisms = Counter(), Counter(), Counter()
    examples = {}
    protected = set(backend.artifact["atoms"])
    initial = set(backend.artifact["initial"])

    def pre(surface):
        return pretokenizer_split(backend._transport.pre_tokenizer, surface)

    for row in rows:
        enc = backend.encode_offsets(row["text"])
        surface = enc.text
        spans, continuation = exact_segments(backend, enc)
        units = UNITIZERS["akshara"](surface)
        ends = list(accumulate(map(len, units)))
        egc = set(accumulate(map(len, UNITIZERS["grapheme"](surface))))
        unit_ends = set(ends)
        pre_cuts = set(accumulate(map(len, pre(surface))))
        merge_cuts = set(accumulate(map(len, merge_domains(surface, pre))))
        byte_positions = list(accumulate((len(char.encode()) for char in surface), initial=0))
        totals["documents"] += 1
        totals["byte_ids"] += sum(i in backend.fallback_ids for i in enc.ids)
        totals["unicode_scalar_continuation_transitions"] += continuation
        totals["legacy_excluded_transitions"] += max(0, len(enc.ids) - 1) - sum(
            not (a[2] or b[2]) for a, b in zip(spans, spans[1:])
        )
        totals["malayalam_reference_interior_sites"] += sum(
            i not in unit_ends and nearby_malayalam(surface, i) for i in range(1, len(surface))
        )
        for a, b in zip(spans, spans[1:]):
            site = a[1]
            is_byte = a[2] or b[2]
            totals["exact_scalar_transitions"] += 1
            if is_byte:
                totals["byte_adjacent_exact_transitions"] += 1
                totals["byte_adjacent_grapheme_breaks"] += site not in egc
            if not nearby_malayalam(surface, site):
                continue
            legacy_malayalam = surface[site - 1] in ML_SET and surface[site] in ML_SET
            if not is_byte and legacy_malayalam:
                totals["legacy_covered_malayalam_transitions"] += 1
            if site in unit_ends:
                continue
            if is_byte:
                totals["byte_adjacent_akshara_breaks"] += 1
            elif legacy_malayalam:
                totals["legacy_akshara_breaks"] += 1
            else:
                totals["additional_joiner_context_lexical_breaks"] += 1
            label = primary_pattern(surface[site - 1], surface[site])
            scope = "byte_adjacent" if is_byte else "legacy_lexical" if legacy_malayalam else "joiner_context_lexical"
            primary[f"{scope}/{label}"] += 1
            unit = units[bisect_right(ends, site)]
            flags = []
            if len(unit) > 1 and unit not in protected and backend.artifact["segmenter"] == "hybrid":
                flags.append("unprotected_akshara_scalar_backoff")
                if any(char not in initial for char in unit):
                    flags.append("missing_initial_scalar")
            if site in merge_cuts:
                flags.append("actual_pretokenizer_domain_cut")
            elif site in pre_cuts:
                flags.append("native_cut_coalesced_before_bpe")
            for flag in flags:
                mechanisms[f"{scope}/{flag}"] += 1
            key = f"{scope}/{label}"
            if len(examples.setdefault(key, [])) < example_limit:
                examples[key].append({
                    "document": row["id"], "canonical_codepoint_offset": site,
                    "canonical_utf8_offset": byte_positions[site], "left_span": a[:2], "right_span": b[:2],
                    "adjacent_codepoints": [f"U+{ord(ch):04X}" for ch in surface[site-1:site+1]],
                    "mechanisms": flags,
                })
    return {"totals": dict(totals), "primary_patterns": dict(primary), "mechanisms": dict(mechanisms),
            "examples": examples, "selected_model_segmenter": backend.artifact["segmenter"]}
