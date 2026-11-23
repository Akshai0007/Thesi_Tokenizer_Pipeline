"""Malayalam boundary audit v2. Use original text and never bridge removed pieces."""

import re
from collections import Counter

import regex

from mltok.contracts import Metric
from mltok.lib.unicode_ml import ML_SET, VIRAMA, ZWJ, ZWNJ, akshara_starts
from mltok.registry import register

_BYTE = re.compile(r"<0x[0-9A-F]{2}>\Z")
_GRAPHEME = regex.compile(r"\X")
_JOINERS = frozenset((ZWJ, ZWNJ))


def _coordinates(text, unit):
    if unit == "codepoint":
        return None
    ends = {0: 0}
    byte = 0
    for index, char in enumerate(text, 1):
        byte += len(char.encode("utf-8"))
        ends[byte] = index
    return ends


def _ml_context(text, pos):
    left, right = pos - 1, pos
    while left >= 0 and text[left] in _JOINERS:
        left -= 1
    while right < len(text) and text[right] in _JOINERS:
        right += 1
    return left >= 0 and right < len(text) and text[left] in ML_SET and text[right] in ML_SET


@register("metric", "boundary_offsets_v2")
class BoundaryOffsetsV2(Metric):
    """Use exact contiguous offsets. The HF Python default is Unicode codepoints."""

    def __init__(self, offset_unit="codepoint", example_limit=24):
        if offset_unit not in ("codepoint", "utf8_byte"):
            raise ValueError("offset_unit must be codepoint or utf8_byte")
        self.offset_unit = offset_unit
        self.example_limit = example_limit

    def compute(self, backend, docs):
        tokenizer = getattr(backend, "tok", None)
        c, examples = Counter(), []
        if tokenizer is None:
            return {"offset_v2_available": False}
        added = tokenizer.get_added_tokens_decoder()
        specials = {idx for idx, obj in added.items() if obj.special}
        byte_fallback = bool(getattr(getattr(tokenizer, "model", None), "byte_fallback", False))
        for text, source in docs:
            c["docs"] += 1
            enc = tokenizer.encode(text, add_special_tokens=False)
            native = enc.offsets
            coords = _coordinates(text, self.offset_unit)
            offsets = native if coords is None else [
                (coords.get(a, -1), coords.get(b, -1)) for a, b in native
            ]
            akshara = akshara_starts(text)
            graphemes = {part.end() for part in _GRAPHEME.finditer(text)}
            for i, ((a, p), (q, b)) in enumerate(zip(offsets, offsets[1:])):
                c["transitions"] += 1
                pair = enc.tokens[i:i + 2]
                if any(idx in specials for idx in enc.ids[i:i + 2]):
                    c["special_excluded"] += 1
                    continue
                if byte_fallback and any(_BYTE.fullmatch(token) for token in pair):
                    c["byte_fallback_excluded"] += 1
                    continue
                if not 0 <= a < p == q < b <= len(text):
                    c["ambiguous_offsets_excluded"] += 1
                    continue
                if not _ml_context(text, p):
                    c["outside_malayalam"] += 1
                    continue
                c["eligible"] += 1
                ak_break, egc_break = p not in akshara, p not in graphemes
                c["akshara_breaks"] += ak_break
                c["egc_breaks"] += egc_break
                c["references_disagree"] += ak_break != egc_break
                c["akshara_after_virama"] += ak_break and text[p - 1] == VIRAMA
                if (ak_break or egc_break) and len(examples) < self.example_limit:
                    examples.append({"source": source, "text": text, "character_offset": p,
                                     "utf8_offset": len(text[:p].encode("utf-8")),
                                     "native_offsets": native[i:i + 2], "tokens": pair,
                                     "akshara_break": ak_break, "egc_break": egc_break})
        return {
            "offset_v2_available": True, "offset_v2_native_unit": self.offset_unit,
            **{f"offset_v2_{key}": c[key] for key in (
                "docs", "transitions", "eligible", "special_excluded", "byte_fallback_excluded",
                "ambiguous_offsets_excluded", "outside_malayalam", "akshara_breaks",
                "egc_breaks", "references_disagree", "akshara_after_virama")},
            "offset_v2_akshara_rate": c["akshara_breaks"] / c["eligible"] if c["eligible"] else None,
            "offset_v2_egc_rate": c["egc_breaks"] / c["eligible"] if c["eligible"] else None,
            "offset_v2_examples": examples,
        }
