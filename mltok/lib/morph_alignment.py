"""Conservative surface projection of externally supported morphological analyses."""

import hashlib
import json
from itertools import accumulate
from pathlib import Path

from .bpe_core import CONFIDENCE_SCALE

_OPERATIONS = {"concatenation", "affix", "agglutination", "sandhi", "gemination", "stem_mutation"}


def independent_reviewers(values):
    return (isinstance(values, list) and all(type(r) is str and r and r == r.strip() for r in values)
            and len({r.casefold() for r in values}) >= 2)


def surface_boundaries(surface, parts, realized, operation, units):
    """Check one explicit surface analysis, independent of its authorization."""
    if (not isinstance(parts, list) or not isinstance(realized, list)
            or type(operation) is not str or operation not in _OPERATIONS
            or len(parts) != len(realized) or not parts
            or any(type(part) is not str or not part for part in (*parts, *realized))
            or (parts != realized and operation == "concatenation") or "".join(realized) != surface):
        return None
    sites = tuple(accumulate(map(len, realized)))[:-1]
    return sites if set(sites) <= set(accumulate(map(len, units(surface)))) else None


def project(surface, candidates, units):
    """Project only complete, compatible surface decompositions with evidence.

    Confidence is fixed point in thousandths. Conflicting offsets, incomplete
    rewrites or offsets inside an initial unit result in abstention for the form.
    """
    offsets, confidence = None, CONFIDENCE_SCALE
    for row in candidates:
        if not isinstance(row, dict):
            return {}
        weight = row.get("confidence_milli")
        if (not independent_reviewers(row.get("reviewers"))
                or type(weight) is not int or not 1 <= weight <= CONFIDENCE_SCALE):
            return {}
        parts = row.get("parts")
        sites = surface_boundaries(surface, parts, row.get("surface_parts", parts), row.get("operation"), units)
        if sites is None or (offsets is not None and offsets != sites):
            return {}
        offsets = sites
        confidence = min(confidence, weight)
    return {site: confidence for site in offsets or ()}


def load_reviewed_forms(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Morphology alignment file SHA-256 mismatch")
    payload = json.loads(data)
    if payload.get("schema") != "mom-surface-alignments-v1" or not isinstance(payload.get("forms"), dict):
        raise ValueError("Expected a versioned morphology alignment artifact")
    if any(not word or not isinstance(candidates, list) or not candidates for word, candidates in payload["forms"].items()):
        raise ValueError("Expected nonempty surface forms and candidate lists")
    return payload["forms"]


def reviewed_mlmorph_candidates(surface, analyzer, reviewed):
    """Join exact analyzer readings to reviewed rules. Unknown readings abstain."""
    matched = [row for row in reviewed if row.get("analysis")]
    readings = {reading for reading, _rank in analyzer.analyse(surface) or []}
    if not readings or not readings <= {row["analysis"] for row in matched}:
        return []
    return [row for row in matched if row["analysis"] in readings]
