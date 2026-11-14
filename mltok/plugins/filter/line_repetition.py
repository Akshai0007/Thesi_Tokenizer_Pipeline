"""Drop boilerplate/spam: documents with >= min_lines lines of which fewer than min_unique are distinct."""
from mltok.contracts import DocFilter
from mltok.registry import register


@register("filter", "line_repetition")
class LineRepetitionFilter(DocFilter):
    def __init__(self, min_lines=5, min_unique=0.3):
        self.min_lines, self.min_unique = min_lines, min_unique

    def keep(self, text):
        lines = [ln for ln in text.split("\n") if ln.strip()]
        return len(lines) < self.min_lines or len(set(lines)) / len(lines) >= self.min_unique
