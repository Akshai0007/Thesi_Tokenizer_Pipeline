"""Malayalam-aware constrained BPE (our trainer).

Three registered variants live in this one file:
  mlbpe        constraint (no token may START with a combining mark) + akshara seed merges
  mlbpe_plain  same trainer, no constraint, no seeds  (control experiment)
  mlbpe_morph  mlbpe + training-time penalty for merges crossing morpheme boundaries
               (needs `boundary_file`: lines "word<TAB>offset,offset", offsets without the marker)

ALL Malayalam-specific knowledge (what is an invalid token, which merges to seed) is defined here,
so replacing this file is how you adapt the trainer to another language or rule set.
"""
from pathlib import Path

from mltok.contracts import Algorithm
from mltok.lib.backends import HFBackend
from mltok.lib.bpe_core import build_tokenizer, train_bpe
from mltok.lib.io import select_alphabet
from mltok.lib.marker import SP, hf_full_normalizer
from mltok.lib.unicode_ml import (
    CHILLU, CONSONANTS, INDEP_VOWELS, ML_BLOCK, MODIFIERS, VIRAMA, VOWEL_SIGNS, ZWJ, ZWNJ, is_combining,
)
from mltok.registry import register


def token_ok(tok: str) -> bool:
    """A token must not start with a combining mark (vowel sign, virama, ZWJ/ZWNJ ...)."""
    s = tok.lstrip(SP)
    return True if not s else not is_combining(s[0])


def akshara_seed_pairs() -> list[tuple[str, str]]:
    pairs = [(c, VIRAMA) for c in CONSONANTS]
    for c in CONSONANTS:
        pairs += [(c, s) for s in VOWEL_SIGNS] + [(c, m) for m in MODIFIERS]
    for v in INDEP_VOWELS + CHILLU:
        pairs += [(v, m) for m in MODIFIERS]
    return pairs


def load_boundaries(path):
    out = {}
    if path and Path(path).exists():
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            if "\t" in ln:
                w, offs = ln.split("\t", 1)
                out[w] = tuple(int(x) for x in offs.split(",") if x.strip())
    return out


class _Base(Algorithm):
    needs_word_counts = True
    uses_pretokenizer = True
    constraint, seeds, morph = True, True, False

    def __init__(self, max_token_len=0, min_pair_count=2, alphabet_coverage=0.99999,
                 max_word_types=0, seed_mode="all", boundary_file=None, boundary_lambda=0.5):
        self.max_token_len, self.min_pair_count = max_token_len, min_pair_count
        self.alphabet_coverage, self.max_word_types = alphabet_coverage, max_word_types
        self.seed_mode, self.boundary_file, self.boundary_lambda = seed_mode, boundary_file, boundary_lambda

    def train(self, ctx):
        alphabet = select_alphabet(ctx.word_counts, self.alphabet_coverage, forced=ML_BLOCK + [ZWNJ, ZWJ])
        res = train_bpe(
            ctx.word_counts,
            vocab_target=ctx.vocab,
            n_specials=len(ctx.specials),
            alphabet=alphabet,
            token_ok=token_ok if self.constraint else None,
            seed_pairs=akshara_seed_pairs() if self.seeds else None,
            seed_mode=self.seed_mode,
            max_token_len=self.max_token_len,
            min_pair_count=self.min_pair_count,
            max_word_types=self.max_word_types or ctx.resources.get("max_word_types", 0),
            boundaries=load_boundaries(self.boundary_file) if self.morph else None,
            boundary_lambda=self.boundary_lambda if self.morph else 0.0,
            log=ctx.log,
        )
        tok = build_tokenizer(
            res, specials=ctx.specials,
            hf_normalizer=hf_full_normalizer(ctx.normalizer), hf_pretokenizer=ctx.pretokenizer.hf(),
        )
        be = HFBackend(tok, ctx.normalizer, name=self.name, specials=ctx.specials)
        be.save(ctx.out_dir)
        return be, res.info

    def load(self, out_dir, normalizer):
        return HFBackend.load(out_dir, normalizer, name=self.name)


@register("algorithm", "mlbpe")
class MLBPE(_Base):
    pass


@register("algorithm", "mlbpe_plain")
class MLBPEPlain(_Base):
    constraint, seeds = False, False


@register("algorithm", "mlbpe_morph")
class MLBPEMorph(_Base):
    morph = True
