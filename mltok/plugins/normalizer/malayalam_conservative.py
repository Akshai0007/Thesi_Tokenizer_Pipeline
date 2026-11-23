"""NFC and context-limited word-final legacy chillu normalization."""

import unicodedata

import regex
import tokenizers
from tokenizers import Regex, normalizers

from mltok.contracts import Normalizer
from mltok.lib.unicode_ml import VIRAMA, ZWJ, ZWNJ
from mltok.registry import register

_LEGACY_BASES = "\u0d23\u0d28\u0d30\u0d32\u0d33"
_PATTERNS = [
    (rf"(?<!{VIRAMA})(?<!{VIRAMA}[{ZWJ}{ZWNJ}]){base}{VIRAMA}{ZWJ}(?![\p{{L}}\p{{M}}{ZWJ}{ZWNJ}])",
     chr(0x0D7A + i))
    for i, base in enumerate(_LEGACY_BASES)
]
_PYTHON = [(regex.compile(pattern), chillu) for pattern, chillu in _PATTERNS]


@register("normalizer", "ml_conservative")
class MalayalamConservative(Normalizer):
    def __init__(self, terminal_chillu=True):
        if type(terminal_chillu) is not bool:
            raise ValueError("terminal_chillu must be a boolean")
        self.terminal_chillu = terminal_chillu

    @staticmethod
    def fingerprint_dependencies():
        return [unicodedata.unidata_version, regex.__version__, tokenizers.__version__]

    def normalize(self, text):
        text = unicodedata.normalize("NFC", text)
        if self.terminal_chillu:
            for pattern, replacement in _PYTHON:
                text = pattern.sub(replacement, text)
        return text

    def hf(self):
        steps = [normalizers.NFC()]
        if self.terminal_chillu:
            steps.extend(normalizers.Replace(Regex(p), r) for p, r in _PATTERNS)
        return normalizers.Sequence(steps)
