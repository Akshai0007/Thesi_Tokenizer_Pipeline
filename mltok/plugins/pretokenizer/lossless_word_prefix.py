"""Lossless uniform word-prefix pretokenizer compatible with escaped transport markers."""

import regex
import tokenizers
from tokenizers import Regex, pre_tokenizers

from mltok.contracts import PreTokenizer
from mltok.lib.uniform_prefix import ESC, SP
from mltok.lib.unicode_ml import ZWJ, ZWNJ
from mltok.registry import register

_W = rf"\p{{L}}\p{{M}}{ZWNJ}{ZWJ}"
PATTERN = "|".join([
    rf"{ESC}[{ESC}{SP}]",
    rf"{SP}?[{_W}]+",
    rf"{SP}?\p{{N}}(?![\p{{M}}{ZWNJ}{ZWJ}])",
    rf"{SP}?[^\s{SP}{ESC}{_W}\p{{N}}]+",
    r"\s+",
    rf"{SP}(?={SP})",
    SP,
    r"\X",
])


@register("pretokenizer", "lossless_word_prefix")
class LosslessWordPrefix(PreTokenizer):
    """Word-prefix pretokenizer keeping escaped literal markers isolated and lossless."""

    def __init__(self):
        self._pre = pre_tokenizers.Split(Regex(PATTERN), behavior="isolated", invert=False)

    @staticmethod
    def fingerprint_dependencies():
        return [regex.__version__, tokenizers.__version__]

    def split(self, marked_text):
        return [piece for piece, _ in self._pre.pre_tokenize_str(marked_text)]

    def hf(self):
        return self._pre
