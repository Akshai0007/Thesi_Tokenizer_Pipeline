"""The word-start marker is owned by the ENGINE so every normalizer / pre-tokenizer plugin
can stay unaware of it. Plain text -> marked text: prepend "▁" and turn spaces into "▁"."""
from __future__ import annotations

from tokenizers import normalizers

SP = "\u2581"


def mark_words(s: str) -> str:
    # empty input stays empty: HF's Prepend is a no-op on "" and both paths must agree
    return SP + s.replace(" ", SP) if s else ""


def full_normalize(normalizer, text: str) -> str:
    """Python path: plugin normalization, then the marker."""
    return mark_words(normalizer.normalize(text))


def hf_full_normalizer(normalizer) -> normalizers.Normalizer:
    """HF path: plugin normalizer, then the marker (what ships inside tokenizer.json)."""
    return normalizers.Sequence(
        [normalizer.hf(), normalizers.Prepend(SP), normalizers.Replace(" ", SP)]
    )
