"""Training-time preference for repairing scalar splits in Malayalam aksharas."""

from itertools import groupby

from .bpe_core import CONFIDENCE_SCALE
from .orthographic_units import UNITIZERS
from .unicode_ml import ML_SET


def guarded_examples(selected, alphabet, atoms):
    """Build the same hybrid sequences with optional internal merge bonuses."""
    known_symbols, protected = set(alphabet), set(atoms)
    examples, fallback_bytes, weighted_sites = [], 0, 0
    for word, count in selected:
        units, repair = [], set()
        position = 0
        for orth in UNITIZERS["akshara"](word):
            if orth in protected:
                units.append(orth)
            else:
                units.extend(orth)
                if any(c in ML_SET for c in orth):
                    repair.update(range(position + 1, position + len(orth)))
            position += len(orth)
        start = 0
        for known, group in groupby(units, known_symbols.__contains__):
            sequence = list(group)
            if known:
                offsets = {}
                end = start
                for symbol in sequence[:-1]:
                    end += len(symbol)
                    if end in repair:
                        # The generic event trainer subtracts boundary weights.
                        # A negative weight rewards closing this interior cut.
                        offsets[end - start] = -CONFIDENCE_SCALE
                weighted_sites += count * len(offsets)
                examples.append((sequence, count, offsets))
            else:
                fallback_bytes += count * sum(len(s.encode()) for s in sequence)
            start += sum(map(len, sequence))
    return examples, fallback_bytes, weighted_sites
