"""Choose Malayalam akshara atoms and preserve a reversible scalar backoff."""

from collections import Counter

import regex

from .orthographic_units import UNITIZERS

_ML_LETTER = regex.compile(r"[\p{L}&&\p{Script=Malayalam}]", regex.V1)


def preferred_atoms(counts, minimum, limit):
    frequency = Counter()
    for word, count in counts.items():
        for unit in UNITIZERS["akshara"](word):
            if len(unit) > 1 and _ML_LETTER.search(unit):
                frequency[unit] += count
    return [unit for unit, count in sorted(frequency.items(), key=lambda row: (-row[1], row[0]))
            if count >= minimum][:limit]


def initial_units(segmenter, atoms):
    if segmenter != "hybrid":
        return UNITIZERS[segmenter]
    protected = frozenset(atoms)

    def hybrid(word):
        return [piece for unit in UNITIZERS["akshara"](word)
                for piece in ((unit,) if unit in protected else unit)]

    return hybrid


def select_alphabet(counts, segmenter, capacity, coverage, minimum, atom_cap):
    atom_limit = min(atom_cap, capacity // 3)
    atoms = preferred_atoms(counts, minimum, atom_limit) if segmenter == "hybrid" and atom_limit else []
    unitize = initial_units(segmenter, atoms)
    units = Counter()
    scalar_counts = Counter()
    for word, count in counts.items():
        for unit in unitize(word):
            units[unit] += count
        if atoms:
            for char in word:
                scalar_counts[char] += count
    if atoms:
        # Reserve scalar representations even for letters seen exclusively
        # within a protected atom. They can still represent a new combination.
        alphabet = atoms + sorted(scalar_counts, key=lambda c: (-scalar_counts[c], c))[:capacity - len(atoms)]
    else:
        mass = 0
        total = sum(units.values())
        alphabet = []
        for unit, count in sorted(units.items(), key=lambda row: (-row[1], row[0])):
            if len(alphabet) >= capacity or mass >= total * coverage:
                break
            alphabet.append(unit)
            mass += count
    return alphabet, atoms, sum(units[unit] for unit in alphabet) / sum(units.values())
