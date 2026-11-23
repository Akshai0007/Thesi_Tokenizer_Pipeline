"""Canonical-offset and fidelity audit for backends exposing surface encodings."""

from collections import Counter

from mltok.contracts import Metric
from mltok.lib.mom_backend import implementation_identity, runtime_identity
from mltok.lib.orthographic_units import UNITIZERS
from mltok.lib.surface_scoring import exact_boundaries
from mltok.lib.unicode_ml import ML_SET, ZWJ, ZWNJ, akshara_starts
from mltok.registry import register


def _malayalam(text, i, step):
    while 0 <= i < len(text) and text[i] in (ZWJ, ZWNJ):
        i += step
    return 0 <= i < len(text) and text[i] in ML_SET


@register("metric", "mom_surface")
class MOMSurface(Metric):
    @staticmethod
    def fingerprint_dependencies():
        return {**runtime_identity(), **implementation_identity(
            "orthographic_units.py", "surface_scoring.py", "unicode_ml.py")}

    def compute(self, backend, docs):
        if not callable(getattr(backend, "encode_offsets", None)):
            return {"mom_surface_available": False}
        counts = Counter()
        for text, _ in docs:
            encoding = backend.encode_offsets(text)
            canonical = encoding.text
            decoded = backend.decode(encoding.ids)
            counts["docs"] += 1
            counts["raw_equal"] += decoded == text
            counts["canonical_equal"] += decoded == canonical
            counts["canonicalized_docs"] += text != canonical
            counts["byte_fallback_ids"] += sum(i in backend.fallback_ids for i in encoding.ids)
            counts["ids"] += len(encoding.ids)
            akshara = akshara_starts(canonical)
            egc, pos = set(), 0
            for unit in UNITIZERS["grapheme"](canonical):
                pos += len(unit)
                egc.add(pos)
            mom, pos = set(), 0
            for unit in UNITIZERS["akshara"](canonical):
                pos += len(unit)
                mom.add(pos)
            cuts = list(exact_boundaries(encoding, backend.fallback_ids))
            counts["offset_exclusions"] += max(0, len(encoding.ids) - 1) - len(cuts)
            counts["akshara_reference_sites"] += sum(
                _malayalam(canonical, pos - 1, -1) and _malayalam(canonical, pos, 1) for pos in akshara
            )
            counts["egc_reference_sites"] += sum(
                _malayalam(canonical, pos - 1, -1) and _malayalam(canonical, pos, 1) for pos in egc
            )
            for pos in cuts:
                if _malayalam(canonical, pos - 1, -1) and _malayalam(canonical, pos, 1):
                    counts["eligible"] += 1
                    counts["akshara_breaks"] += pos not in akshara
                    counts["egc_breaks"] += pos not in egc
                    counts["grammar_breaks"] += pos not in mom
        return {
            "mom_surface_available": True,
            **{f"mom_{key}": counts[key] for key in (
                "docs", "raw_equal", "canonical_equal", "canonicalized_docs", "byte_fallback_ids",
                "ids", "offset_exclusions", "eligible", "akshara_reference_sites", "egc_reference_sites",
                "akshara_breaks", "egc_breaks", "grammar_breaks")},
            **{f"mom_{reference}_breaks_per_1000_reference_sites": (
                1000 * counts[f"{reference}_breaks"] / counts[f"{reference}_reference_sites"]
                if counts[f"{reference}_reference_sites"] else None
            ) for reference in ("akshara", "egc")},
        }
