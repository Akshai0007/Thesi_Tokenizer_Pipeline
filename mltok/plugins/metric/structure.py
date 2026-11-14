"""Malayalam script-integrity diagnostics (need backend.tokens()):
mark_start_rate     share of Malayalam-initial tokens that START with a combining mark
akshara_break_rate  share of in-word Malayalam token boundaries that cut through an akshara
Diagnostics, not objectives - the proxy LM decides whether they matter."""
import re

from mltok.contracts import Metric
from mltok.lib.unicode_ml import akshara_starts, is_combining, is_ml
from mltok.registry import register

SP = "\u2581"
_BYTE = re.compile(r"^<0x[0-9A-F]{2}>$")


def _special(t):
    return len(t) > 2 and t.startswith("<") and t.endswith(">") and t not in ("<>",)


@register("metric", "structure")
class Structure(Metric):
    def __init__(self, docs=3000):
        self.docs = docs

    def compute(self, backend, docs):
        mark_num = mark_den = br = bd = 0
        for text, _ in docs[: self.docs]:
            toks = backend.tokens(text)
            if toks is None:
                return {}
            toks = [t for t in toks if not _BYTE.match(t) and not _special(t)]
            surface = "".join(toks)
            for t in toks:
                s = t.lstrip(SP)
                if s and is_ml(s[0]):
                    mark_den += 1
                    mark_num += int(is_combining(s[0]))
            starts, pos = akshara_starts(surface), 0
            for t in toks[:-1]:
                pos += len(t)
                if pos < len(surface) and is_ml(surface[pos - 1]) and is_ml(surface[pos]):
                    bd += 1
                    br += int(pos not in starts)
        return {
            "mark_start_rate": mark_num / mark_den if mark_den else 0.0,
            "akshara_break_rate": br / bd if bd else 0.0,
        }
