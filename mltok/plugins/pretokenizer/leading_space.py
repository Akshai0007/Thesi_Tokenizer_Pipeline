"""Permit a merge across one literal ASCII space before an existing piece."""

from tokenizers import Regex, pre_tokenizers

from mltok.contracts import PreTokenizer
from mltok.registry import register


@register("pretokenizer", "leading_space")
class LeadingSpace(PreTokenizer):
    def __init__(self, pattern=r"\p{L}[\p{L}\p{M}]*|\s+|\X"):
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("Expected a native isolated split pattern")
        # The original whitespace alternative still consumes multiple spaces,
        # tabs and line breaks intact. No marker or virtual leading space is used.
        self._split = pre_tokenizers.Split(
            Regex(r" ?(?=\S)(?:" + pattern + r")|(?:" + pattern + ")"), behavior="isolated")

    def split(self, text):
        return [piece for piece, _ in self._split.pre_tokenize_str(text)]

    def hf(self):
        return self._split
