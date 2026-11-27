"""Offline CPU unit tests for Malayalam Unicode edge cases, EventBackend isolation, and canonical BPB."""

import hashlib
import json
import math

import pytest

from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.canonical_bpb import canonical_bpb
from mltok.lib.orthographic_units import UNITIZERS
from mltok.lib.picky_backend import EventBackend
from mltok.lib.unicode_ml import akshara_split


@pytest.mark.parametrize("text", [
    "ദുഃഖം അതഃ",
    "ഓം\u0d01 ക\u0d01",
    "എൻ്റ ൻ്റ",
    "ൎക്കു് ൎ",
    "ക്\u200d ക്\u200d. ക്\u200dഷ",
])
def test_malayalam_conservative_preserves_unicode_edge_cases(text):
    normalizer = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    assert normalizer.normalize(text) == normalizer.hf().normalize_str(text) == text
    assert "\u0d7f" not in normalizer.normalize(text)


def test_malayalam_conservative_converts_terminal_n_chillu_but_not_k_chillu():
    normalizer = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    assert normalizer.normalize("അവന്\u200d ക്\u200d") == "അവൻ ക്\u200d"
    assert normalizer.hf().normalize_str("അവന്\u200d ക്\u200d") == "അവൻ ക്\u200d"


@pytest.mark.parametrize(("surface", "expected"), [
    ("ദുഃഖം", ["ദുഃ", "ഖം"]),
    ("ക\u0d01ക", ["ക\u0d01", "ക"]),
    ("ൻ്റ", ["ൻ്റ"]),
    ("ൎക്കു്", ["ൎക്കു്"]),
])
def test_orthographic_akshara_unitizer_handles_visarga_candrabindu_reph_and_chillu_conjunct(surface, expected):
    assert UNITIZERS["akshara"](surface) == expected
    assert "".join(expected) == surface


@pytest.mark.xfail(
    reason="Known post-T5/T8 cleanup: unicode_ml.AKSHARA_RE omits chillu conjuncts and U+0D00 unlike orthographic_units._MALAYALAM",
    strict=True,
)
@pytest.mark.parametrize("surface", ["ൻ്റ", "ക\u0d00ക"])
def test_unicode_ml_akshara_split_matches_orthographic_unitizer(surface):
    assert akshara_split(surface) == UNITIZERS["akshara"](surface)


def _train_event_backend(tmp_path, name, *, algo_spec, texts, vocab=360):
    shard = tmp_path / f"{name}.jsonl"
    shard.write_text(
        "\n".join(json.dumps({"text": text}, ensure_ascii=False) for text in texts) + "\n",
        encoding="utf-8",
    )
    normalizer = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    pretokenizer = registry.make("pretokenizer", {"type": "script_runs", "scripts": ["Malayalam", "Latin"]})
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
    return algorithm.train(ctx)


def test_event_backend_lossless_roundtrip_and_control_isolation_for_all_specials(tmp_path):
    corpus = [
        "കിളികൾ കൂട്ടത്തിൽ കടലിൽ ദുഃഖം എൻ്റ ൎക്കു് English 123 ",
        "മലയാളം Mixed text <pad> <s> </s> <unk> <|im_start|> <|im_end|> <|endoftext|> ▁",
    ] * 12
    backend, _ = _train_event_backend(
        tmp_path,
        "event-specials",
        algo_spec={"type": "boundary_guarded_hybrid", "guard_milli": 1000, "picky_threshold_milli": 900, "hybrid_min_count": 3},
        texts=corpus,
    )
    control_ids = set(backend.special_ids.values())
    cases = [
        *DEFAULT_SPECIALS,
        " ".join(DEFAULT_SPECIALS),
        "▁",
        "▁മലയാളം ▁English",
        "മലയാളം ▁ <pad> <s> </s> <unk> <|im_start|> <|im_end|> <|endoftext|> English 123 ദുഃഖം എൻ്റ ൎക്കു് ക്\u200d ക\u0d01",
        "അവന്\u200d. ഓം\u0d01 😀",
    ]
    for text in cases:
        enc = backend.encode_offsets(text)
        assert control_ids.isdisjoint(enc.ids)
        assert backend.decode(enc.ids) == backend.expected(text)
        assert enc.offsets[0][0] == 0
        assert enc.offsets[-1][1] == len(enc.text)


def test_canonical_bpb_on_event_backend_and_rejects_forged_controls(tmp_path):
    corpus = ["കിളികൾ കൂട്ടത്തിൽ കടലിൽ പൂക്കൾ മലയാളം English text "] * 15
    backend, _ = _train_event_backend(
        tmp_path,
        "event-bpb",
        algo_spec={"type": "picky_bpe", "picky_threshold_milli": 900, "hybrid_min_count": 3},
        texts=corpus,
    )
    texts = [
        "കിളികൾ English 123",
        "<pad> <s> </s> <unk> <|im_start|> <|im_end|> <|endoftext|>",
        "▁അവന്\u200d ദുഃഖം എൻ്റ ൎക്കു് 🌻",
    ]
    rows = []
    expected_bytes = 0
    expected_targets = 0
    for idx, text in enumerate(texts):
        canonical_text = backend.expected(text)
        raw = canonical_text.encode("utf-8")
        expected_bytes += len(raw)
        expected_targets += len(backend.encode_offsets(text).ids) + 1
        rows.append({
            "id": f"doc-{idx}",
            "text": text,
            "canonical_bytes": len(raw),
            "canonical_sha256": hashlib.sha256(raw).hexdigest(),
        })

    result = canonical_bpb(backend, rows, lambda _inputs, targets: [math.log(2)] * len(targets))
    assert result["canonical_bytes"] == expected_bytes
    assert result["predicted_tokens"] == expected_targets
    assert result["bpb"] == pytest.approx(expected_targets / expected_bytes)

    with pytest.raises(ValueError, match="canonical bytes"):
        canonical_bpb(backend, [{**rows[0], "canonical_bytes": rows[0]["canonical_bytes"] + 1}], lambda *_: [])


def test_boundary_guarded_hybrid_triggers_picky_remove_under_negative_guard_offsets(tmp_path):
    # Force scalar fallback (hybrid_min_count=100) so negative guard_milli=-1000 offsets
    # apply to "കി" and "ളി"; "കി"+"ളി" consumes >90% of "കി", triggering Picky remove.
    corpus = ["കിളി " * 40 + "കിതു "]
    vocab = len(DEFAULT_SPECIALS) + 256 + 16
    backend, info = _train_event_backend(
        tmp_path,
        "guarded-remove",
        algo_spec={
            "type": "boundary_guarded_hybrid",
            "guard_milli": 1000,
            "picky_threshold_milli": 900,
            "hybrid_min_count": 100,
        },
        texts=corpus,
        vocab=vocab,
    )
    assert info["guarded_internal_sites"] > 0
    assert info["n_removed"] >= 1
    assert any(event[0] == "remove" for event in backend.artifact["events"])

    reloaded = EventBackend.load(tmp_path / "guarded-remove")
    probes = ["കിളി", "കിതു", "കിളികിതു", "ദുഃഖം എൻ്റ ൎക്കു് <pad> ▁"]
    assert reloaded.encode_batch(probes) == backend.encode_batch(probes)
    for probe in probes:
        assert reloaded.decode(reloaded.encode_offsets(probe).ids) == reloaded.expected(probe)
