"""Keep documents whose length (characters) is within [min_chars, max_chars]."""
from mltok.contracts import DocFilter
from mltok.registry import register


@register("filter", "length")
class LengthFilter(DocFilter):
    def __init__(self, min_chars=80, max_chars=200_000):
        self.min_chars, self.max_chars = min_chars, max_chars

    def keep(self, text):
        n = len(text)
        return n >= self.min_chars and (not self.max_chars or n <= self.max_chars)
