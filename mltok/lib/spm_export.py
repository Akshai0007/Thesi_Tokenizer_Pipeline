"""SentencePiece Unigram  ->  Hugging Face `tokenizer.json` (so the winner loads with AutoTokenizer / vLLM / TGI).

The converter rebuilds the exact piece table (same ids and scores), byte fallback, digit splitting and the
word-start marker pipeline. Fidelity is never assumed: `parity()` re-tokenizes text with BOTH tokenizers and
reports the share of identical id sequences, and the export refuses to be called "verified" below the threshold.

Known, documented difference: text that contains special tokens (e.g. `<|im_start|>`) is treated the way the rest
of this repo treats it (the marker is added per text segment), not the way SentencePiece itself treats it.
Plain-text pretraining is unaffected.
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

from .marker import SP, hf_full_normalizer


def _proto(model_path):
    from sentencepiece import sentencepiece_model_pb2 as pb

    m = pb.ModelProto()
    m.ParseFromString(Path(model_path).read_bytes())
    return m


def spm_to_hf(model_path, normalizer) -> Tokenizer:
    """`normalizer` must be the SAME Normalizer plugin instance the SentencePiece model was trained with
    (SentencePiece itself runs with an identity normalizer; our normalization is applied before it)."""
    m = _proto(model_path)
    if m.trainer_spec.model_type != 1:
        raise ValueError("only SentencePiece UNIGRAM models can be converted (model_type=1)")
    if len(m.normalizer_spec.precompiled_charsmap):
        raise ValueError("SentencePiece model uses a normalization charsmap; train with normalization_rule_name=identity")
    pieces = list(m.pieces)
    unk_id = next(i for i, p in enumerate(pieces) if p.type == 2)  # UNKNOWN
    vocab = [(p.piece, float(p.score)) for p in pieces]
    tok = Tokenizer(models.Unigram(vocab, unk_id=unk_id, byte_fallback=bool(m.trainer_spec.byte_fallback)))
    tok.normalizer = hf_full_normalizer(normalizer)
    steps = [pre_tokenizers.Metaspace(replacement=SP, prepend_scheme="never", split=True)]
    if m.trainer_spec.split_digits:
        steps.append(pre_tokenizers.Digits(individual_digits=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(steps)
    tok.decoder = decoders.Sequence(
        [decoders.Replace(SP, " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(content=" ", left=1, right=0)]
    )
    # control (pad/bos/eos) and user-defined symbols are atomic special tokens
    specials = [p.piece for p in pieces if p.type in (3, 4)]  # CONTROL, USER_DEFINED
    tok.add_special_tokens(specials)
    return tok


EDGE_TEXTS = [
    "a  b   c", " leading", "trailing ", "tabs\tand\nnewlines\n\n", "1234567890 \u0d66\u0d67\u0d68 3.14, 1,00,000",
    "emoji \U0001F600 and \u0939\u093f\u0928\u094d\u0926\u0940 \u65e5\u672c\u8a9e", "zwnj: \u0d38\u0d26\u0d4d\u200c\u0d35\u0d3e\u0d30\u0d02 old: \u0d28\u0d4d\u200d \u0d30\u0d4d\u200d",
    "WhatsApp \u0d7d ok? yes!", "((( ))) ... --- ??? !!!", "", " ", "x", "a" * 300, "https://example.com/path?q=1&r=2",
]


def parity(spm_path, tok: Tokenizer, normalizer, texts: list[str]) -> dict:
    """Compare SentencePiece and the converted tokenizer on `texts`.

    identical        same id sequence
    ties             different ids but EQUAL total Unigram score (and equal decoding): the two implementations
                     break an exact Viterbi tie differently - equally probable, both lossless, not a conversion error
    real_mismatches  different ids AND different score  -> a conversion bug (export is rejected)
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    identical = ties = 0
    bad: list[dict] = []
    hf_ids = [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]
    for t, h in zip(texts, hf_ids):
        s = sp.encode(normalizer.normalize(t))
        if s == h:
            identical += 1
            continue
        same_score = abs(sum(sp.get_score(i) for i in s) - sum(sp.get_score(i) for i in h)) < 1e-4
        if same_score and sp.decode(s) == sp.decode(h):
            ties += 1
        elif len(bad) < 5:
            bad.append({"text": t[:60], "spm": sp.id_to_piece(s)[:8], "hf": [tok.id_to_token(i) for i in h[:8]]})
    n = max(len(texts), 1)
    real = len(texts) - identical - ties
    return {"texts": len(texts), "identical": identical, "ties": ties, "real_mismatches": real,
            "identical_rate": identical / n, "equivalent_rate": (identical + ties) / n, "examples": bad}


def export_hf(spm_path, out_dir, normalizer, check_texts: list[str] | None = None) -> dict:
    """Write tokenizer.json (+ config files) next to the SentencePiece model; returns the parity report.
    `ok` is True only if there are NO real mismatches on the check texts (+ built-in edge cases)."""
    from .backends import save_hf_tokenizer
    from .bpe_core import DEFAULT_SPECIALS

    tok = spm_to_hf(spm_path, normalizer)
    specials = [t for t in DEFAULT_SPECIALS if tok.token_to_id(t) is not None]
    save_hf_tokenizer(tok, out_dir, specials)
    rep = parity(spm_path, tok, normalizer, list(check_texts or []) + EDGE_TEXTS)
    rep["ok"] = rep["real_mismatches"] == 0
    return rep
