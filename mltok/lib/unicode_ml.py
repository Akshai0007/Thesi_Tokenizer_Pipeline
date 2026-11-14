"""Malayalam Unicode character classes and akshara segmentation.

An *akshara* is the orthographic syllable: (consonant + virama)* + base
consonant / independent vowel / chillu, followed by vowel signs, an optional
trailing virama (samvruthokaram) and modifiers (anusvara, visarga, candrabindu).
A token that starts with a combining mark (vowel sign, virama, ...) or that
cuts through an akshara is what Malayalam tokenizers most often get wrong.
"""
from __future__ import annotations

import unicodedata

import regex

SP = "\u2581"  # "▁" word-start marker (same symbol SentencePiece uses)
VIRAMA = "\u0d4d"
ZWNJ = "\u200c"
ZWJ = "\u200d"


def _assigned(cps) -> list[str]:
    return [chr(c) for c in cps if unicodedata.category(chr(c)) != "Cn"]


CONSONANTS = _assigned(range(0x0D15, 0x0D3B))
INDEP_VOWELS = _assigned(list(range(0x0D05, 0x0D15)) + [0x0D60, 0x0D61])
CHILLU = _assigned(range(0x0D7A, 0x0D80))
VOWEL_SIGNS = [
    c
    for c in _assigned(list(range(0x0D3E, 0x0D4D)) + [0x0D57, 0x0D62, 0x0D63])
    if unicodedata.category(c).startswith("M")
]
MODIFIERS = ["\u0d01", "\u0d02", "\u0d03"]  # candrabindu, anusvara, visarga
ML_BLOCK = _assigned(range(0x0D00, 0x0D80))
ML_SET = frozenset(ML_BLOCK)

_C = "".join(CONSONANTS)
_V = "".join(INDEP_VOWELS)
_L = "".join(CHILLU)
_S = "".join(VOWEL_SIGNS)
_M = "".join(MODIFIERS)

# A conjunct is C + virama (+ optional explicit ZWJ) repeated; ZWNJ deliberately
# breaks the conjunct, so it may only appear after a *trailing* virama.
AKSHARA_RE = regex.compile(
    rf"(?:[{_C}]{VIRAMA}{ZWJ}?)*[{_C}{_V}{_L}][{_S}]*(?:{VIRAMA}[{ZWJ}{ZWNJ}]?)?[{_M}]*|\X"
)


def akshara_split(s: str) -> list[str]:
    return AKSHARA_RE.findall(s)


def akshara_starts(s: str) -> set[int]:
    """Character offsets at which an akshara begins (plus len(s) as sentinel)."""
    pos, out = 0, set()
    for a in AKSHARA_RE.findall(s):
        out.add(pos)
        pos += len(a)
    out.add(pos)
    return out


def is_combining(ch: str) -> bool:
    """True for characters that must never start a token (marks, ZWJ, ZWNJ)."""
    return unicodedata.category(ch).startswith("M") or ch in (ZWJ, ZWNJ)


def is_ml(ch: str) -> bool:
    return ch in ML_SET or ch in (ZWJ, ZWNJ)


def ml_ratio(text: str) -> float:
    """Share of alphabetic characters that are Malayalam letters/marks."""
    letters = ml = 0
    for ch in text:
        if ch.isalpha() or (ch in ML_SET and unicodedata.category(ch).startswith("M")):
            letters += 1
            if ch in ML_SET:
                ml += 1
    return ml / letters if letters else 0.0
