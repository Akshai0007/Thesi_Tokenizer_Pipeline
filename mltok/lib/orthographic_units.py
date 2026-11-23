"""Initial BPE units and common pretokenizer safety for the matched arms."""

import regex

from .unicode_ml import CHILLU, CONSONANTS, INDEP_VOWELS, MODIFIERS, VIRAMA, VOWEL_SIGNS, ZWJ, ZWNJ

_EGC = regex.compile(r"\X")
_BASES = "".join(CONSONANTS + CHILLU) + "\u0d54\u0d55\u0d56"
_VOWELS = "".join(INDEP_VOWELS)
_SIGNS = "".join(VOWEL_SIGNS)
_MARKS = "".join(MODIFIERS) + "\u0d00\u0d3b\u0d3c"
_REPH = "\u0d4e"
# Chillus can carry dependent signs and can themselves start a written
# conjunct, for example chillu n + virama + rra. ZWNJ ends the conjunct.
_MALAYALAM = regex.compile(
    rf"(?:{_REPH}?(?:[{_BASES}]{VIRAMA}{ZWJ}?)*[{_BASES}]|[{_VOWELS}])"
    rf"[{_SIGNS}]*(?:{VIRAMA}[{ZWJ}{ZWNJ}]?)?[{_MARKS}]*|\X"
)
UNITIZERS = {"codepoint": list, "grapheme": _EGC.findall, "akshara": _MALAYALAM.findall}


def _ends(units):
    offset = 0
    for unit in units:
        offset += len(unit)
        yield offset


def merge_domains(text, split):
    """Coalesce cuts inside either reference unit identically across the ablations."""
    pieces = split(text)
    if any(not piece for piece in pieces) or "".join(pieces) != text:
        raise ValueError("Pretokenizer must partition its input exactly")
    safe = set(_ends(_EGC.findall(text))) & set(_ends(_MALAYALAM.findall(text)))
    start = 0
    for end in _ends(pieces):
        if end in safe:
            yield text[start:end]
            start = end
