"""Synthetic Malayalam-like documents (smoke / speed tests only - never for design decisions)."""
from pathlib import Path

from mltok.contracts import Source
from mltok.lib.synth import synthetic_docs
from mltok.registry import register

_DEFAULT_SEED = Path(__file__).resolve().parents[3] / "data" / "seed_ml.txt"


@register("source", "synthetic")
class SyntheticSource(Source):
    """Params: seed_file, n_docs, n_word_types (0 = recombine seed words), seed."""

    example_params = {"n_docs": 5}

    def __init__(self, id="synthetic", seed_file=None, n_docs=5000, n_word_types=0, seed=0, **params):
        super().__init__(id=id, **params)
        self.seed_file = str(seed_file or _DEFAULT_SEED)
        self.n_docs, self.n_word_types, self.seed = n_docs, n_word_types, seed

    def iter_docs(self):
        for t in synthetic_docs(self.seed_file, self.n_docs, self.n_word_types, self.seed):
            yield {"text": t, "src": self.id}
