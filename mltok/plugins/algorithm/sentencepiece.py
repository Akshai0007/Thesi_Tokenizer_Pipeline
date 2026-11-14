"""SentencePiece Unigram / BPE (what most Indic tokenizers use). Does its own whitespace
pre-tokenization, so the pre-tokenizer setting is ignored (uses_pretokenizer = False)."""
from pathlib import Path

from mltok.contracts import Algorithm
from mltok.lib.backends import SPMBackend
from mltok.lib.io import iter_texts
from mltok.registry import register


class _SPM(Algorithm):
    needs_word_counts = False
    uses_pretokenizer = False
    model_type = "unigram"
    contract_vocab = 600

    def __init__(self, character_coverage=0.99995, max_mb=300, input_sentence_size=5_000_000):
        self.character_coverage, self.max_mb, self.input_sentence_size = character_coverage, max_mb, input_sentence_size

    def train(self, ctx):
        import sentencepiece as spm

        out = Path(ctx.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        corpus = out / "spm_train.txt"
        budget, written = int(ctx.resources.get("spm_max_mb", self.max_mb) * 1_000_000), 0
        with open(corpus, "w", encoding="utf-8") as f:
            for text in iter_texts(ctx.shard_paths):
                for ln in ctx.normalizer.normalize(text).split("\n"):
                    if ln.strip():
                        f.write(ln + "\n")
                        written += len(ln.encode("utf-8"))
                if written >= budget:
                    break
        prefix = str(out / "spm")
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=prefix, model_type=self.model_type, vocab_size=ctx.vocab,
            character_coverage=self.character_coverage, byte_fallback=True, split_digits=True,
            split_by_whitespace=True, split_by_unicode_script=False, normalization_rule_name="identity",
            remove_extra_whitespaces=False, add_dummy_prefix=True, pad_id=0, bos_id=1, eos_id=2, unk_id=3,
            user_defined_symbols=[s for s in ctx.specials if s.startswith("<|")],
            num_threads=int(ctx.resources.get("threads", 8)),
            input_sentence_size=self.input_sentence_size, shuffle_input_sentence=True, minloglevel=2,
        )
        corpus.unlink(missing_ok=True)
        return SPMBackend(prefix + ".model", ctx.normalizer, self.name), {"note": "SentencePiece"}

    def load(self, out_dir, normalizer):
        return SPMBackend(Path(out_dir) / "spm.model", normalizer, self.name)


@register("algorithm", "spm_unigram")
class SPMUnigram(_SPM):
    model_type = "unigram"


@register("algorithm", "spm_bpe")
class SPMBPE(_SPM):
    model_type = "bpe"
