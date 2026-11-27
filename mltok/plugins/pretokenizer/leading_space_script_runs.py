"""Native serialized pretokenization attaching one leading ASCII space to script runs."""

from functools import lru_cache

import regex
import tokenizers
from tokenizers import Regex, pre_tokenizers

from mltok.contracts import PreTokenizer
from mltok.registry import register


@lru_cache(maxsize=1)
def _scalars():
    return "".join(map(chr, range(0x110000)))


def _letter_ranges(property_pattern):
    pattern = regex.compile(rf"[\p{{L}}&&{property_pattern}]+", regex.V1)
    ranges = []
    for match in pattern.finditer(_scalars()):
        run = match.group()
        ranges.append(run if len(run) == 1 else f"{run[0]}-{run[-1]}")
    return "".join(ranges)


@lru_cache(maxsize=32)
def _script_pattern(script):
    primary = _letter_ranges(rf"\p{{sc={script}}}")
    if not primary:
        raise ValueError(f"Script {script} has no primary letters")
    shared = _letter_ranges(rf"[\p{{scx={script}}}&&[\p{{sc=Common}}\p{{sc=Inherited}}]]")
    attached = r"\p{M}" + "\u200c\u200d"
    base = rf"[{primary}][{attached}]*"
    if not shared:
        return rf"(?:{base})+"
    prefix = rf"(?:[{shared}][{attached}]*)*"
    rest = rf"(?:[{primary}{shared}][{attached}]*)*"
    return prefix + base + rest


@register("pretokenizer", "leading_space_script_runs")
class LeadingSpaceScriptRuns(PreTokenizer):
    """Attach a single leading ASCII space to Malayalam/Latin runs while isolating other tokens."""

    def __init__(self, scripts=("Malayalam", "Latin")):
        if (
            not isinstance(scripts, (tuple, list))
            or not scripts
            or any(type(s) is not str or not regex.fullmatch("[A-Za-z_]+", s) for s in scripts)
            or len(set(scripts)) != len(scripts)
        ):
            raise ValueError("scripts must be a nonempty list of distinct Unicode script names")
        patterns = [_script_pattern(s) for s in scripts]
        word_union = "|".join(patterns)
        # Keep non-space whitespace (newlines, tabs) separate from ASCII space runs so
        # "\n കേരളം" still attaches the single space to "കേരളം" while "  കേരളം" keeps "  " intact.
        full_pattern = "|".join([
            rf" (?:{word_union})",
            word_union,
            r"\p{N}+(?![\p{M}\u200c\u200d])",
            r" +",
            r"[^\S ]+",
            r"\X",
        ])
        self._pre = pre_tokenizers.Split(Regex(full_pattern), behavior="isolated")

    @staticmethod
    def fingerprint_dependencies():
        return [regex.__version__, tokenizers.__version__]

    def split(self, marked_text):
        return [piece for piece, _ in self._pre.pre_tokenize_str(marked_text)]

    def hf(self):
        return self._pre
