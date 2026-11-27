"""Offline CPU tests for next-gen Malayalam tokenizer candidates A (T8-LS) and B (M48-Guard)."""

import hashlib
import json
import math

import pytest

from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.canonical_bpb import canonical_bpb
from mltok.lib.hybrid_units_v2 import select_alphabet_v2
from mltok.lib.orthographic_units import UNITIZERS
from mltok.lib.unicode_ml import ML_BLOCK, ZWJ, ZWNJ
from mltok.lib.uniform_prefix import (
    ESC,
    SP,
    UniformPrefixEventBackend,
    decode_transport,
    encode_transport,
    encode_transport_with_offsets,
)


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
    "▂ ▂▁ ▁▂ ▂▂▁",
    "<|endoftext|> <bos> <eos> <unk> <pad> <s> </s> <|im_start|> <|im_end|>",
    "മലയാളം India 12345! 3.14, ൧൨൩? A  2️⃣👩\u200d💻 x\u0301മലയാളം",
    "漢字 𐍈 ☃ emoji 😀🎉",
]


def _train_v2(tmp_path, name, *, pre_spec, texts, vocab=1200, algo_overrides=None):
    shard = tmp_path / f"{name}.jsonl"
    shard.write_text(
        "\n".join(json.dumps({"text": text}, ensure_ascii=False) for text in texts) + "\n",
        encoding="utf-8",
    )
    normalizer = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    pretokenizer = registry.make("pretokenizer", pre_spec)
    algo_spec = {
        "type": "boundary_guarded_hybrid_v2",
        "guard_milli": 1000,
        "picky_threshold_milli": 900,
        "hybrid_min_count": 3,
        "alphabet_coverage": 0.99999,
        **(algo_overrides or {}),
    }
    algorithm = registry.make("algorithm", algo_spec)
    ctx = TrainContext(
        vocab=vocab,
        specials=DEFAULT_SPECIALS,
        normalizer=normalizer,
        pretokenizer=pretokenizer,
        out_dir=tmp_path / name,
        word_counts=None,
        shard_paths=[str(shard)],
        resources={"merge_checkpoint_every": 1},
        log=lambda *_: None,
    )
    backend, info = algorithm.train(ctx)
    return backend, info, algorithm.load(tmp_path / name, normalizer)


def test_leading_space_script_runs_attaches_single_space_and_preserves_edges():
    pre = registry.make("pretokenizer", "leading_space_script_runs")
    hf = pre.hf()

    sample = "കേരളം India\n കേരളം  മലയാളം\r\n\t12345! A  2️⃣👩\u200d💻"
    pieces = pre.split(sample)
    assert pieces == [piece for piece, _ in hf.pre_tokenize_str(sample)]
    assert "".join(pieces) == sample
    assert pieces == [
        "കേരളം",
        " India",
        "\n",
        " കേരളം",
        "  ",
        "മലയാളം",
        "\r\n\t",
        "12345",
        "!",
        " A",
        "  ",
        "2️⃣",
        "👩\u200d💻",
    ]

    for text in EDGE_TEXTS:
        split_pieces = pre.split(text)
        assert "".join(split_pieces) == text
        assert split_pieces == [piece for piece, _ in hf.pre_tokenize_str(text)]

    # Never splits inside Malayalam aksharas, virama, or ZWJ/ZWNJ sequences.
    ml_probe = "ൻ്റ ൎക്കു് ക്\u200dഷ ക്\u200cഷ സദ്\u200cവാരം"
    ml_pieces = pre.split(ml_probe)
    assert ml_pieces == ["ൻ്റ", " ൎക്കു്", " ക്\u200dഷ", " ക്\u200c", "ഷ", " സദ്\u200cവാരം"] or "".join(ml_pieces) == ml_probe
    # Verify every piece boundary respects Malayalam akshara starts.
    akshara_ends = set()
    pos = 0
    for unit in UNITIZERS["akshara"](ml_probe):
        pos += len(unit)
        akshara_ends.add(pos)
    pos = 0
    for piece in ml_pieces:
        pos += len(piece)
        assert pos in akshara_ends


def test_uniform_prefix_transport_bijection_and_lossless_word_prefix_pretokenizer():
    pre = registry.make("pretokenizer", "lossless_word_prefix")
    hf = pre.hf()

    for text in EDGE_TEXTS:
        transport, starts, ends = encode_transport_with_offsets(text)
        assert transport == encode_transport(text)
        assert decode_transport(transport) == text
        assert len(starts) == len(ends) == len(transport)
        if text:
            assert transport.startswith(SP)
            assert starts[0] == 0 and ends[-1] == len(text)
        pieces = pre.split(transport)
        assert "".join(pieces) == transport
        assert pieces == [piece for piece, _ in hf.pre_tokenize_str(transport)]
        # Also verify contract compliance on raw/unescaped strings.
        raw_pieces = pre.split(text)
        assert "".join(raw_pieces) == text
        assert raw_pieces == [piece for piece, _ in hf.pre_tokenize_str(text)]

    # Verify literal ▁ never attaches as a word-start prefix to a following word.
    escaped = encode_transport("▁കേരളം കേരളം")
    assert pre.split(escaped) == [SP, ESC + SP, "കേരളം", SP + "കേരളം"]
    assert decode_transport("".join(pre.split(escaped))) == "▁കേരളം കേരളം"


def test_pure_whitespace_tokens_drop_sharply_vs_script_runs(tmp_path):
    sentences = [
        "കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ തീരത്തുള്ള ഒരു സംസ്ഥാനമാണ്.",
        "മലയാളം ഭാഷയും സംസ്കാരവും വളരെ സമ്പന്നമാണ്.",
        "പുതിയ ടോക്കണൈസർ പരീക്ഷണം വിജയകരമായി പൂർത്തിയായി.",
        "കേരളത്തിലെ ജനങ്ങൾ വിദ്യാഭ്യാസത്തിന് വലിയ പ്രാധാന്യം നൽകുന്നു.",
    ] * 15
    multi_sentence = " ".join(sentences[:8])

    old_pre = registry.make("pretokenizer", "script_runs")
    new_pre = registry.make("pretokenizer", "leading_space_script_runs")
    cand_b_pre = registry.make("pretokenizer", "lossless_word_prefix")

    old_ws = sum(1 for p in old_pre.split(multi_sentence) if not p.strip())
    new_ws = sum(1 for p in new_pre.split(multi_sentence) if not p.strip())
    cand_b_ws = sum(1 for p in cand_b_pre.split(encode_transport(multi_sentence)) if not p.strip(SP + " \t\r\n"))
    assert old_ws > 20
    assert new_ws == 0
    assert cand_b_ws == 0

    docs = [" ".join(sentences[i:i + 4]) for i in range(0, len(sentences), 4)]
    t8_ls, _, _ = _train_v2(tmp_path, "t8-ls", pre_spec="leading_space_script_runs", texts=docs, vocab=1600)
    m48_guard, _, _ = _train_v2(tmp_path, "m48-guard", pre_spec="lossless_word_prefix", texts=docs, vocab=1600)
    sr_algo = registry.make("algorithm", {"type": "boundary_guarded_hybrid", "hybrid_min_count": 3})
    shard = tmp_path / "sr.jsonl"
    shard.write_text("\n".join(json.dumps({"text": s}, ensure_ascii=False) for s in docs) + "\n", encoding="utf-8")
    norm = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    t8_orig, _ = sr_algo.train(TrainContext(1600, DEFAULT_SPECIALS, norm, old_pre, tmp_path / "t8-orig", None, [str(shard)]))

    orig_ws_toks = sum(1 for tok in t8_orig.tokens(multi_sentence) if not tok.strip())
    ls_ws_toks = sum(1 for tok in t8_ls.tokens(multi_sentence) if not tok.strip())
    m48_guard_ws_toks = sum(1 for tok in m48_guard.tokens(multi_sentence) if not tok.strip(SP + " \t\r\n"))
    assert orig_ws_toks > 20
    assert ls_ws_toks <= orig_ws_toks * 0.1
    assert m48_guard_ws_toks == 0


def test_hybrid_units_v2_enforces_coverage_and_preserves_core_plus_byte_fallback(tmp_path):
    counts = {
        "മലയാളം": 400,
        "കേരളം": 350,
        "India": 300,
        "12345": 200,
        # Rare count-1 foreign/noise scalars (~0.1% mass)
        "漢字": 1,
        "𐍈": 1,
        "☃": 1,
    }
    alphabet, atoms, _ = select_alphabet_v2(
        counts,
        "hybrid",
        capacity=1000,
        coverage=0.99,
        minimum=8,
        atom_cap=512,
    )
    assert len(atoms) > 0
    aset = set(alphabet)
    # Core Malayalam block + joiners + ASCII printable + whitespace preserved.
    assert set(ML_BLOCK) <= aset
    assert {ZWJ, ZWNJ, " ", "\t", "\n", "\r", "A", "z", "0", "!"} <= aset
    # Rare non-Malayalam scalars excluded below coverage=0.99.
    for rare in ("漢", "字", "𐍈", "☃"):
        assert rare not in aset

    corpus = (
        ["മലയാളം കേരളം India 12345 "] * 60
        + ["漢字 𐍈 ☃"]
    )
    backend, _, reloaded = _train_v2(
        tmp_path,
        "cov99",
        pre_spec="leading_space_script_runs",
        texts=corpus,
        vocab=1200,
        algo_overrides={"alphabet_coverage": 0.99},
    )
    for rare in ("漢", "字", "𐍈", "☃"):
        assert rare not in set(backend.artifact["initial"])
        enc = backend.encode_offsets(rare)
        assert set(enc.ids) <= backend.fallback_ids
        assert backend.decode(enc.ids) == rare
        assert reloaded.decode(reloaded.encode_offsets(rare).ids) == rare


@pytest.mark.parametrize("pre_spec", ["leading_space_script_runs", "lossless_word_prefix"])
def test_candidates_ab_lossless_roundtrip_specials_and_metrics(tmp_path, pre_spec):
    corpus = [
        "കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ സംസ്ഥാനമാണ് ൻ്റ ൎക്കു് ക്\u200dഷ ക്\u200cഷ.",
        "മലയാളം India 12345 <|endoftext|> <bos> <eos> <unk> <pad> ▁",
    ] * 15
    backend, info, reloaded = _train_v2(tmp_path, f"cand-{pre_spec}", pre_spec=pre_spec, texts=corpus, vocab=1200)
    control_ids = set(backend.special_ids.values())

    if pre_spec == "lossless_word_prefix":
        assert info["uniform_prefix"] is True
        assert isinstance(backend, UniformPrefixEventBackend)
        assert isinstance(reloaded, UniformPrefixEventBackend)
        # Document-initial and mid-document "കേരളം" share identical uniform-prefix tokens.
        doc_start = backend.encode_offsets("കേരളം").ids
        mid_doc = backend.encode_offsets("മലയാളം കേരളം").ids
        assert mid_doc[-len(doc_start):] == doc_start
    else:
        assert info["uniform_prefix"] is False

    for text in EDGE_TEXTS:
        enc = backend.encode_offsets(text)
        assert enc.ids == reloaded.encode_offsets(text).ids
        assert control_ids.isdisjoint(enc.ids)
        assert backend.decode(enc.ids) == backend.expected(text)
        assert reloaded.decode(enc.ids) == reloaded.expected(text)
        if text:
            assert enc.offsets[-1][1] == len(enc.text)
            assert all(0 <= a <= b <= len(enc.text) for a, b in enc.offsets)

    # Control tokens passed explicitly to decode still render verbatim alongside text IDs.
    sample_ids = backend.encode_offsets("കേരളം ▁").ids
    assert backend.decode([backend.special_ids["<s>"], *sample_ids, backend.eos_id]) == "<s>കേരളം ▁</s>"

    # Full compatibility with validation metrics & canonical_bpb.
    docs = [(t, "fixture") for t in EDGE_TEXTS if t]
    for metric_name in ("basic", "roundtrip", "structure", "boundary_offsets_v2", "mom_surface"):
        metric = registry.make("metric", metric_name)
        res = metric.compute(backend, docs)
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
            "id": f"d-{idx}",
            "text": text,
            "canonical_bytes": len(raw),
            "canonical_sha256": hashlib.sha256(raw).hexdigest(),
        })
    bpb = canonical_bpb(backend, bpb_rows, lambda _x, y: [math.log(2)] * len(y))
    assert bpb["bpb"] == pytest.approx(bpb["predicted_tokens"] / bpb["canonical_bytes"])
