"""Synthetic Malayalam-like text for smoke / speed tests ONLY (no linguistic validity)."""
from __future__ import annotations

import itertools
import random
from collections import Counter
from pathlib import Path
from typing import Iterator

from .unicode_ml import akshara_split

VIRAMA, ZWJ = "\u0d4d", "\u200d"
SUFFIXES = [
    "", "", "", "ഉടെ", "ിന്റെ", "ിൽ", "ിലേക്ക്", "ോട്", "യും", "ക്കു", "കൾ", "കളുടെ",
    "യുടെ", "ാണ്", "ായിരുന്നു", "ുന്നു", "ിച്ചു", "ിക്കുന്നു", "ത്തിൽ", "ത്തെ", "ാൻ",
]
_LEGACY = {  # atomic chillu -> legacy consonant+virama+ZWJ encoding
    "\u0d7a": "\u0d23" + VIRAMA + ZWJ, "\u0d7b": "\u0d28" + VIRAMA + ZWJ,
    "\u0d7c": "\u0d30" + VIRAMA + ZWJ, "\u0d7d": "\u0d32" + VIRAMA + ZWJ,
    "\u0d7e": "\u0d33" + VIRAMA + ZWJ, "\u0d7f": "\u0d15" + VIRAMA + ZWJ,
}


def _legacy(rng, text: str, p: float = 0.2) -> str:
    if rng.random() < p:
        for new, old in _LEGACY.items():
            text = text.replace(new, old)
    return text


def synthetic_docs(seed_file, n_docs=5000, n_word_types=0, seed=0) -> Iterator[str]:
    """n_word_types=0: recombine the seed words; >0: random Zipf-sampled aksharas (larger vocabulary)."""
    rng = random.Random(seed)
    text = Path(seed_file).read_text(encoding="utf-8")
    real = [w for w in text.split() if "\u0d00" <= w[0] <= "\u0d7f"]
    if n_word_types:
        inv: Counter = Counter()
        for w in real:
            inv.update(akshara_split(w))
        akshs = list(inv)
        aw = [inv[a] ** 0.8 + 0.5 for a in akshs]
        vocab = set(real)
        while len(vocab) < n_word_types:
            n = rng.choices([1, 2, 3, 4, 5, 6], [1, 4, 6, 5, 3, 1])[0]
            w = "".join(rng.choices(akshs, aw, k=n))
            if rng.random() < 0.3:
                w += rng.choice(SUFFIXES)
            vocab.add(w)
        vocab = sorted(vocab)
        rng.shuffle(vocab)
        cum = list(itertools.accumulate(1 / (r + 1) ** 1.05 for r in range(len(vocab))))
        for _ in range(n_docs):
            sents = [
                " ".join(rng.choices(vocab, cum_weights=cum, k=rng.randint(4, 16))) + rng.choice([".", ".", "?", "!"])
                for _ in range(rng.randint(2, 10))
            ]
            t = "\n".join(sents) if rng.random() < 0.3 else " ".join(sents)
            yield _legacy(rng, t)
    else:
        words = list(dict.fromkeys(w for ln in text.splitlines() for w in ln.split()))
        weights = [1 / r for r in range(1, len(words) + 1)]
        rng.shuffle(words)
        for _ in range(n_docs):
            sents = []
            for _ in range(rng.randint(2, 8)):
                toks = []
                for w in rng.choices(words, weights, k=rng.randint(4, 14)):
                    if rng.random() < 0.35 and w and "\u0d00" <= w[0] <= "\u0d7f":
                        w += rng.choice(SUFFIXES)
                    toks.append(w)
                sents.append(" ".join(toks) + rng.choice([".", ".", "?", "!"]))
            t = "\n".join(sents) if rng.random() < 0.3 else " ".join(sents)
            yield _legacy(rng, t)
