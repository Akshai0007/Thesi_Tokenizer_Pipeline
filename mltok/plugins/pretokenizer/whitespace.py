"""Whitespace-delimited chunks; punctuation stays attached (SentencePiece-like)."""
import regex
from tokenizers import Regex, pre_tokenizers

from mltok.contracts import PreTokenizer
from mltok.registry import register

SP = "\u2581"
PATTERN = rf"{SP}?[^\s{SP}]+|\s+|{SP}(?={SP})|{SP}"
_RE = regex.compile(PATTERN)


@register("pretokenizer", "ws")
class Whitespace(PreTokenizer):
    def split(self, marked_text):
        return _RE.findall(marked_text)

    def hf(self):
        return pre_tokenizers.Split(Regex(PATTERN), behavior="isolated", invert=False)
