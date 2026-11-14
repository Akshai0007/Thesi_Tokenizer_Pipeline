"""Stock Hugging Face BpeTrainer (Rust) - the standard-practice baseline. No byte fallback."""
from tokenizers import Tokenizer, decoders, models, trainers

from mltok.contracts import Algorithm
from mltok.lib.backends import HFBackend
from mltok.lib.io import iter_texts, select_alphabet
from mltok.lib.marker import SP, hf_full_normalizer
from mltok.lib.unicode_ml import ML_BLOCK
from mltok.registry import register


@register("algorithm", "hf_bpe")
class HFBPE(Algorithm):
    needs_word_counts = True  # only used to choose the alphabet
    uses_pretokenizer = True

    def __init__(self, min_frequency=2, alphabet_coverage=0.99999):
        self.min_frequency, self.alphabet_coverage = min_frequency, alphabet_coverage

    def train(self, ctx):
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.normalizer = hf_full_normalizer(ctx.normalizer)
        tok.pre_tokenizer = ctx.pretokenizer.hf()
        alphabet = select_alphabet(ctx.word_counts, self.alphabet_coverage, forced=ML_BLOCK)
        trainer = trainers.BpeTrainer(
            vocab_size=ctx.vocab, special_tokens=ctx.specials, initial_alphabet=alphabet,
            limit_alphabet=len(alphabet) + 16, min_frequency=self.min_frequency, show_progress=False,
        )
        tok.train_from_iterator(iter_texts(ctx.shard_paths), trainer)
        tok.decoder = decoders.Sequence(
            [decoders.Replace(SP, " "), decoders.Fuse(), decoders.Strip(content=" ", left=1, right=0)]
        )
        be = HFBackend(tok, ctx.normalizer, name=self.name, specials=ctx.specials)
        be.save(ctx.out_dir)
        return be, {"note": "HF BpeTrainer baseline (no byte fallback)"}

    def load(self, out_dir, normalizer):
        return HFBackend.load(out_dir, normalizer, name=self.name)
