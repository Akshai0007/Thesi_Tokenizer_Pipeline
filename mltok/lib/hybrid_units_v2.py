"""Coverage-respecting Malayalam hybrid seed unit builder.

Unlike v1 (`hybrid_units.py`), `select_alphabet_v2` enforces `alphabet_coverage`
on non-core scalar units even when `atoms` is non-empty, while always preserving:
- core Malayalam Unicode block + ZWJ (U+200D) + ZWNJ (U+200C),
- ASCII printable characters + standard whitespace,
- configured orthographic `atoms` (and any caller-supplied transport markers).
The 256 UTF-8 byte fallback tokens occupy their separate reserved backend slots,
so rare non-Malayalam/noise scalars below the coverage cutoff fall back to UTF-8
bytes instead of consuming lexical vocabulary slots.
"""

from collections import Counter

from .hybrid_units import initial_units, preferred_atoms
from .unicode_ml import ML_BLOCK, ZWJ, ZWNJ

CORE_SCALARS = tuple(
    dict.fromkeys([
        *ML_BLOCK,
        ZWJ,
        ZWNJ,
        *(chr(cp) for cp in range(0x20, 0x7F)),
        "\t",
        "\n",
        "\r",
    ])
)


def select_alphabet_v2(
    counts,
    segmenter,
    capacity,
    coverage,
    minimum,
    atom_cap,
    *,
    extra_forced=(),
):
    """Select protected atoms + core scalars + coverage-bounded non-core scalars."""
    atom_limit = min(atom_cap, capacity // 3)
    atoms = preferred_atoms(counts, minimum, atom_limit) if segmenter == "hybrid" and atom_limit else []
    unitize = initial_units(segmenter, atoms)
    units = Counter()
    scalar_counts = Counter()
    for word, count in counts.items():
        for unit in unitize(word):
            units[unit] += count
        for char in word:
            scalar_counts[char] += count

    core_all = list(dict.fromkeys([*CORE_SCALARS, *extra_forced]))
    if segmenter in ("hybrid", "codepoint"):
        scalar_cap = max(0, capacity - len(atoms))
        seen_core = sorted(
            (c for c in core_all if scalar_counts[c] > 0),
            key=lambda c: (-scalar_counts[c], c),
        )
        unseen_core = [c for c in core_all if scalar_counts[c] == 0]
        ordered_core = seen_core + unseen_core

        if scalar_cap >= len(ordered_core):
            core_kept = ordered_core
        else:
            # Tiny test budgets: keep seen core scalars first, then fill unseen up to cap.
            core_kept = ordered_core[:scalar_cap]
        core_set = set(core_kept)

        total_mass = sum(scalar_counts.values()) or 1
        covered_mass = sum(scalar_counts[c] for c in core_kept)
        extra = []
        for ch, count in sorted(scalar_counts.items(), key=lambda row: (-row[1], row[0])):
            if ch in core_set:
                continue
            if len(core_kept) + len(extra) >= scalar_cap or covered_mass >= total_mass * coverage:
                break
            extra.append(ch)
            covered_mass += count
        alphabet = atoms + core_kept + extra
    else:
        total = sum(units.values()) or 1
        mass = 0
        alphabet = []
        for unit, count in sorted(units.items(), key=lambda row: (-row[1], row[0])):
            if len(alphabet) >= capacity or mass >= total * coverage:
                break
            alphabet.append(unit)
            mass += count

    unit_total = sum(units.values()) or 1
    return alphabet, atoms, sum(units[unit] for unit in alphabet) / unit_total
