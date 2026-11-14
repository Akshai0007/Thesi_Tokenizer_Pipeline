"""No normalization (raw text)."""
from tokenizers import normalizers

from mltok.contracts import Normalizer
from mltok.registry import register


@register("normalizer", "none")
class Identity(Normalizer):
    def normalize(self, text):
        return text

    def hf(self):
        return normalizers.Sequence([])
