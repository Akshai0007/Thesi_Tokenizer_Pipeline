"""Letters+combining marks (+ZWJ/ZWNJ) form one piece; every digit alone; punctuation runs separate.
Keeps Malayalam vowel signs/virama inside the word piece, and keeps the word-start marker attached."""
import regex
from tokenizers import Regex, pre_tokenizers

from mltok.contracts import PreTokenizer
from mltok.registry import register

SP, ZWNJ, ZWJ = "\u2581", "\u200c", "\u200d"
_W = rf"\p{{L}}\p{{M}}{ZWNJ}{ZWJ}"
PATTERN = (
    rf"{SP}?[{_W}]+|{SP}?\p{{N}}|{SP}?[^\s{SP}{_W}\p{{N}}]+|\s+|{SP}(?={SP})|{SP}"
)
_RE = regex.compile(PATTERN)


@register("pretokenizer", "word")
class Word(PreTokenizer):
    def split(self, marked_text):
        return _RE.findall(marked_text)

    def hf(self):
        return pre_tokenizers.Split(Regex(PATTERN), behavior="isolated", invert=False)
