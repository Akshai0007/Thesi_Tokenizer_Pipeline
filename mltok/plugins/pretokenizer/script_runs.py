"""Native serialized pretokenization with parameterized Unicode script runs."""

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
    # A shared letter can extend the current run. At the start it waits for a
    # primary letter, so configured script order cannot claim a neutral prefix.
    prefix = rf"(?:[{shared}][{attached}]*)*"
    rest = rf"(?:[{primary}{shared}][{attached}]*)*"
    return prefix + base + rest


@register("pretokenizer", "script_runs")
class ScriptRuns(PreTokenizer):
    def __init__(self, scripts=("Malayalam", "Latin")):
        if (not isinstance(scripts, (tuple, list)) or not scripts
                or any(type(s) is not str or not regex.fullmatch("[A-Za-z_]+", s) for s in scripts)
                or len(set(scripts)) != len(scripts)):
            raise ValueError("scripts must be a nonempty list of distinct Unicode script names")
        # Resolve Script_Extensions locally. The native HF regex engine has Script
        # but does not accept Script_Extensions property syntax.
        patterns = [_script_pattern(s) for s in scripts]
        self._pre = pre_tokenizers.Split(
            Regex("|".join([*patterns, r"\s+", r"\X"])), behavior="isolated",
        )

    @staticmethod
    def fingerprint_dependencies():
        return [regex.__version__, tokenizers.__version__]

    def split(self, marked_text):
        return [piece for piece, _ in self._pre.pre_tokenize_str(marked_text)]

    def hf(self):
        return self._pre
