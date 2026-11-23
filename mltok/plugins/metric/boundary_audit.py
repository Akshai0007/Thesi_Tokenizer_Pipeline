"""Audit Malayalam token boundaries using offsets in the original string."""
import re
import unicodedata
from collections import Counter

import regex

from mltok.contracts import Metric
from mltok.lib.unicode_ml import SP, VIRAMA, ZWJ, ZWNJ, akshara_starts, is_combining, is_ml
from mltok.registry import register

_BYTE = re.compile(r"<0x[0-9A-F]{2}>\Z")
_EGC = regex.compile(r"\X")


def _kind(text, pos):
    left, right = text[pos - 1], text[pos]
    if left in (ZWJ, ZWNJ) or right in (ZWJ, ZWNJ):
        return "joiner_adjacent"
    if left == VIRAMA:
        return "after_virama"
    if right == VIRAMA:
        return "before_virama"
    if right in "\u0d00\u0d01\u0d02\u0d03":
        return "before_modifier"
    if unicodedata.category(right).startswith("M"):
        return "before_mark"
    return "other"


@register("metric", "boundary_audit")
class BoundaryAudit(Metric):
    """Exact contiguous Unicode offsets, with separate akshara and grapheme rules."""

    def __init__(self, example_limit=48):
        self.example_limit = example_limit

    def compute(self, backend, docs):
        tok = getattr(backend, "tok", None)
        if tok is None:
            return {"boundary_audit_available": False}
        c, ak_kinds, egc_kinds = Counter(), Counter(), Counter()
        examples = []
        for text, source in docs:
            c["docs"] += 1
            enc = tok.encode(text, add_special_tokens=False)
            ak = akshara_starts(text)
            egc = {part.end() for part in _EGC.finditer(text)}
            lexical = [not _BYTE.fullmatch(token) and not (token.startswith("<") and token.endswith(">"))
                       for token in enc.tokens]
            for token, keep in zip(enc.tokens, lexical):
                head = token.lstrip(SP)[:1]
                if keep and head and is_ml(head):
                    c["malayalam_initial_tokens"] += 1
                    c["combining_mark_starts"] += is_combining(head)
            for i in range(len(enc.tokens) - 1):
                c["token_transitions"] += 1
                if not lexical[i] or not lexical[i + 1]:
                    c["byte_or_special_transitions_excluded"] += 1
                    continue
                start, pos = enc.offsets[i]
                next_start, stop = enc.offsets[i + 1]
                if not (0 <= start < pos == next_start < stop <= len(text)):
                    c["ambiguous_offset_transitions_excluded"] += 1
                    continue
                if not (is_ml(text[pos - 1]) and is_ml(text[pos])):
                    continue
                c["malayalam_boundaries"] += 1
                kind = _kind(text, pos)
                c["akshara_violations"] += pos not in ak
                c["egc_violations"] += pos not in egc
                c["rules_disagree"] += (pos in ak) != (pos in egc)
                if pos not in ak:
                    ak_kinds[kind] += 1
                if pos not in egc:
                    egc_kinds[kind] += 1
                if (pos not in ak or pos not in egc) and len(examples) < self.example_limit:
                    lo, hi = max(0, min(start, pos - 12)), min(len(text), max(stop, pos + 12))
                    lo, hi = max(lo, pos - 32), min(hi, pos + 32)
                    examples.append({"source": source, "text": text[lo:hi], "offset": pos - lo,
                                     "offset_unit": "codepoint", "token_pair": enc.tokens[i:i + 2],
                                     "akshara_allowed_offsets": sorted(p - lo for p in ak if lo < p < hi),
                                     "egc_allowed_offsets": sorted(p - lo for p in egc if lo < p < hi),
                                     "akshara_violation": pos not in ak, "egc_violation": pos not in egc,
                                     "kind": kind})
        return {
            "boundary_audit_available": True,
            **{f"boundary_{key}": c[key] for key in (
                "docs", "token_transitions", "byte_or_special_transitions_excluded",
                "ambiguous_offset_transitions_excluded", "malayalam_boundaries",
                "akshara_violations", "egc_violations", "rules_disagree",
                "malayalam_initial_tokens", "combining_mark_starts")},
            "boundary_akshara_rate": c["akshara_violations"] / max(c["malayalam_boundaries"], 1),
            "boundary_egc_rate": c["egc_violations"] / max(c["malayalam_boundaries"], 1),
            "boundary_mark_start_rate": c["combining_mark_starts"] / max(c["malayalam_initial_tokens"], 1),
            "boundary_akshara_by_kind": dict(ak_kinds),
            "boundary_egc_by_kind": dict(egc_kinds),
            "boundary_examples": examples,
        }
