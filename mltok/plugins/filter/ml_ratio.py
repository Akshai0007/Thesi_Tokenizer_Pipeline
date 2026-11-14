"""Keep documents whose share of Malayalam letters (among all letters) is >= min_ratio.
Replace this file (or pass a different `min_ratio`) to target another language."""
from mltok.contracts import DocFilter
from mltok.lib.unicode_ml import ml_ratio
from mltok.registry import register


@register("filter", "ml_ratio")
class MalayalamRatioFilter(DocFilter):
    def __init__(self, min_ratio=0.5):
        self.min_ratio = min_ratio

    def keep(self, text):
        return ml_ratio(text) >= self.min_ratio
