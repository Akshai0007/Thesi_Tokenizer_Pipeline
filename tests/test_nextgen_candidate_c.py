"""Offline CPU tests for Candidate C: ML-MinGram (BPE-seeded Hard-EM Unigram)."""

import hashlib
import json
import math
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.canonical_bpb import canonical_bpb
from mltok.lib.mingram_core import (
    ARTIFACT,
    MinGramBackend,
    UniformPrefixMinGramBackend,
    is_malayalam_dangling,
    train_mingram_unigram,
    viterbi_segment,
)
from mltok.lib.synth import synthetic_docs

ROOT = Path(__file__).resolve().parents[1]

EDGE_TEXTS = [
    "",
    " ",
    "  ",
    "   \n\t  ",
    "\r\n",
    "leading and trailing ",
    "ൻ്റ ൎക്കു് ക് ങ്ങൾ ക്\u200dഷ ക്\u200cഷ സദ്\u200cവാരം അവന്\u200d",
    "▁",
    "▁മലയാളം ▁English  ▁ ",
    "▂ ▂▁ ▁▂ ▂▂▁ ▂▂▂▂▁",
    "<|endoftext|> <bos> <eos> <unk> <pad> <s> </s> <|im_start|> <|im_end|>",
    "മലയാളം India 12345! 3.14, ൧൨൩? A  2️⃣👩\u200d💻 x\u0301മലയാളം",
    "漢字 𐍈 ☃ emoji 😀🎉",
]


def _corpus_texts():
    ml_docs = list(synthetic_docs(ROOT / "data" / "seed_ml.txt", n_docs=500, n_word_types=2500, seed=29))
    extra = [
        "കേരളം India മലയാളം 12345! ദുഃഖം എൻ്റ ൎക്കു് ക്\u200dഷ ക്\u200cഷ.",
        "പുതിയ ടോക്കണൈസർ പരീക്ഷണം വിജയകരമായി പൂർത്തിയായി. English words mixed 2026.",
        "▁മലയാളം ▂ <pad> <s> </s> <unk> <|im_start|> <|im_end|> <|endoftext|>",
    ] * 10
    return ml_docs + extra


def test_viterbi_and_dangling_filter_and_hard_em_pruning():
    log_probs = {
        "a": math.log(0.1),
        "b": math.log(0.1),
        "ab": math.log(0.5),
        "c": math.log(0.3),
    }
    toks, cost = viterbi_segment("abc!", log_probs, fallback_cost=20.0)
    assert toks == ["ab", "c", "!"]
    assert math.isfinite(cost)

    # Dangling Malayalam mark check
    assert is_malayalam_dangling("ാ")
    assert is_malayalam_dangling("▁്ക")
    assert not is_malayalam_dangling("ക")
    assert not is_malayalam_dangling("▁കേ")

    # Hard-EM prunes excess tokens down to target_vocab while preserving protected units
    word_counts = {"ab": 50, "cd": 40, "abcd": 500, "ax": 1}
    protected = {"a", "b", "c", "d", "x", "്"}
    seed_tokens = ["ab", "cd", "abcd", "ax", "bc", "da", "്ക"]
    scores, stats = train_mingram_unigram(
        word_counts,
        seed_tokens,
        target_vocab=8,
        protected=protected,
        shrink_factor=0.75,
        em_iters=2,
        final_em_iters=2,
        filter_malayalam_dangling=True,
    )
    assert len(scores) == 8
    assert protected <= set(scores)
    assert "്ക" not in scores
    assert stats["dangling_malayalam_filtered"] == 1
    assert "abcd" in scores


@pytest.mark.parametrize("pre_spec", ["lossless_word_prefix", "leading_space_script_runs"])
def test_ml_mingram_end_to_end_both_pretokenizers(tmp_path, pre_spec):
    texts = _corpus_texts()
    shard = tmp_path / "train.jsonl"
    shard.write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts) + "\n",
        encoding="utf-8",
    )

    normalizer = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    pretokenizer = registry.make("pretokenizer", pre_spec)
    algo = registry.make(
        "algorithm",
        {
            "type": "ml_mingram",
            "seed_multiplier": 1.15,
            "guard_milli": 1000,
            "picky_threshold_milli": 900,
            "alphabet_coverage": 0.99,
            "shrink_factor": 0.75,
            "em_iters": 2,
            "final_em_iters": 2,
            "filter_malayalam_dangling": True,
            "uniform_prefix": "auto",
            "export_hf_json": True,
            "hybrid_min_count": 4,
            "require_full_vocabulary": True,
        },
    )
    vocab = 1600
    out_dir = tmp_path / f"mingram-{pre_spec}"
    ctx = TrainContext(
        vocab=vocab,
        specials=DEFAULT_SPECIALS,
        normalizer=normalizer,
        pretokenizer=pretokenizer,
        out_dir=out_dir,
        word_counts=None,
        shard_paths=[str(shard)],
        log=lambda *_: None,
    )
    backend, info = algo.train(ctx)
    reloaded = algo.load(out_dir, normalizer)

    # Exact target vocab reached after Hard-EM Viterbi pruning from 1.15x seed
    assert info["occupied_vocab"] == vocab
    assert backend.vocab_size == vocab
    assert info["seed_budget"] > vocab - len(DEFAULT_SPECIALS) - 256
    assert info["pruning_rounds"] >= 1
    assert (out_dir / ARTIFACT).is_file()
    assert (out_dir / "tokenizer.json").is_file()
    assert info["hf_export"]["hf_export_lossless_verified"] is True

    if pre_spec == "lossless_word_prefix":
        assert info["uniform_prefix"] is True
        assert isinstance(backend, UniformPrefixMinGramBackend)
        assert isinstance(reloaded, UniformPrefixMinGramBackend)
    else:
        assert info["uniform_prefix"] is False
        assert isinstance(backend, MinGramBackend)
        assert not isinstance(backend, UniformPrefixMinGramBackend)

    # Lossless round-trip & non-control ID isolation on backend + reloaded backend + HF tokenizer.json
    control_ids = set(backend.special_ids.values())
    hf_tok = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    assert hf_tok.get_vocab_size() == vocab

    for text in EDGE_TEXTS:
        enc = backend.encode_offsets(text)
        assert enc.ids == reloaded.encode_offsets(text).ids
        assert control_ids.isdisjoint(enc.ids)
        assert backend.decode(enc.ids) == backend.expected(text)
        assert reloaded.decode(enc.ids) == reloaded.expected(text)
        if text:
            assert enc.offsets[-1][1] == len(enc.text)
            assert all(0 <= a <= b <= len(enc.text) for a, b in enc.offsets)

        hf_enc = hf_tok.encode(text, add_special_tokens=False)
        assert control_ids.isdisjoint(hf_enc.ids)
        assert hf_tok.decode(hf_enc.ids, skip_special_tokens=False) == backend.expected(text)

    # Rare foreign scalars excluded below coverage=0.99 fall back to UTF-8 bytes
    for rare in ("漢", "字", "𐍈", "☃"):
        assert rare not in set(backend.artifact["initial"])
        enc = backend.encode_offsets(rare)
        non_fallback = [tok for idx, tok in zip(enc.ids, enc.tokens) if idx not in backend.fallback_ids]
        assert non_fallback == (["\u2581"] if info["uniform_prefix"] else [])
        assert any(idx in backend.fallback_ids for idx in enc.ids)
        assert backend.decode(enc.ids) == rare

    # Control IDs render verbatim when explicitly supplied to decode
    probe_ids = backend.encode_offsets("കേരളം ▁").ids
    assert backend.decode([backend.special_ids["<s>"], *probe_ids, backend.eos_id]) == "<s>കേരളം ▁</s>"

    # Full compatibility with validation metrics & canonical_bpb
    docs = [(t, "fixture") for t in EDGE_TEXTS if t]
    for metric_name in ("basic", "roundtrip", "structure", "boundary_offsets_v2", "mom_surface"):
        res = registry.make("metric", metric_name).compute(backend, docs)
        assert isinstance(res, dict)
        json.dumps(res)
        if metric_name == "roundtrip":
            assert res["roundtrip_ok"] == 1.0
        elif metric_name == "mom_surface":
            assert res["mom_canonical_equal"] == len(docs)

    bpb_rows = []
    for idx, text in enumerate(["കേരളം ▁", "<pad> <unk> <|endoftext|>", "ൻ്റ ൎക്കു് 漢字"]):
        raw = backend.expected(text).encode("utf-8")
        bpb_rows.append({
            "id": f"c-{idx}",
            "text": text,
            "canonical_bytes": len(raw),
            "canonical_sha256": hashlib.sha256(raw).hexdigest(),
        })
    bpb = canonical_bpb(backend, bpb_rows, lambda _x, y: [math.log(2)] * len(y))
    assert bpb["bpb"] == pytest.approx(bpb["predicted_tokens"] / bpb["canonical_bytes"])
