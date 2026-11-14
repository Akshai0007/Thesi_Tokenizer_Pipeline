"""Unicode NFC only (composes the Malayalam O/OO/AU vowel signs; does NOT unify old chillu)."""
import unicodedata

from tokenizers import normalizers

from mltok.contracts import Normalizer
from mltok.registry import register


@register("normalizer", "nfc")
class NFC(Normalizer):
    def normalize(self, text):
        return unicodedata.normalize("NFC", text)

    def hf(self):
        return normalizers.NFC()
