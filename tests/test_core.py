"""Correctness of the building blocks (lib + builtin plugins)."""
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from mltok import registry
from mltok.lib.bpe_core import DEFAULT_SPECIALS, build_tokenizer, reference_encode_word, train_bpe
from mltok.lib.io import select_alphabet
from mltok.lib.marker import SP, full_normalize, hf_full_normalizer
from mltok.lib.synth import synthetic_docs
from mltok.lib.unicode_ml import ML_BLOCK, akshara_split, is_combining
from mltok.plugins.algorithm.mlbpe import akshara_seed_pairs, token_ok  # noqa: F401  (import path check only)

ROOT = Path(__file__).resolve().parents[1]
SEED = (ROOT / "data" / "seed_ml.txt").read_text(encoding="utf-8").splitlines()
OLD_N = "\u0d28\u0d4d\u200d"
EXTRA = [
    "old chillu: " + OLD_N + " \u0d30\u0d4d\u200d \u0d15\u0d4d\u200d \u0d23\u0d4d\u200d \u0d32\u0d4d\u200d \u0d33\u0d4d\u200d",
    "zwnj: സദ്\u200cവാരം zwj: ക്\u200dഷ", "a  b   c\n\nd\te", "junk\u200bchars\ufeff and\u00adsoft and \r\n crlf",
    "emoji 😀 mixed हिन्दी 123 4.5% ok?!", "ൊ ോ ൌ decomposed: \u0d46\u0d3e \u0d47\u0d3e \u0d46\u0d57", "",
]
TEXTS = SEED + EXTRA


def norm(spec): return registry.make("normalizer", spec)
def pre(spec): return registry.make("pretokenizer", spec)


def test_nfc_does_not_unify_chillu_but_ml_does():
    assert unicodedata.normalize("NFC", OLD_N) != "\u0d7b"
    assert norm("ml").normalize(OLD_N) == "\u0d7b"
    assert norm("nfc").normalize(OLD_N) == OLD_N


@pytest.mark.parametrize("word,expected", [
    ("മലയാളം", ["മ", "ല", "യാ", "ളം"]), ("കേരളം", ["കേ", "ര", "ളം"]),
    ("വിദ്യാർത്ഥി", ["വി", "ദ്യാ", "ർ", "ത്ഥി"]), ("ആണ്", ["ആ", "ണ്"]),
    ("സദ്\u200cവാരം", ["സ", "ദ്\u200c", "വാ", "രം"]),
])
def test_akshara_split(word, expected):
    assert akshara_split(word) == expected


@pytest.mark.parametrize("spec", ["none", "nfc", "ml", {"type": "ml", "zw": "drop"}])
def test_python_and_hf_normalizers_agree(spec):
    n = norm(spec)
    hf = hf_full_normalizer(n)
    for t in TEXTS:
        assert hf.normalize_str(t) == full_normalize(n, t), (spec, t)


@pytest.mark.parametrize("mode", ["ws", "word"])
def test_python_and_hf_pretokenizers_agree_and_cover(mode):
    p, n = pre(mode), norm("ml")
    for t in TEXTS:
        s = full_normalize(n, t)
        assert "".join(p.split(s)) == s
        assert [x for x, _ in p.hf().pre_tokenize_str(s)] == p.split(s), (mode, t)


def test_word_mode_keeps_marks_inside_words():
    w = pre("word").split(full_normalize(norm("ml"), "മലയാളം 123"))
    assert w[0] == SP + "മലയാളം" and w[1:] == [SP + "1", "2", "3"]


@pytest.fixture(scope="module")
def wc():
    n, p = norm("ml"), pre("word")
    c = Counter()
    for d in synthetic_docs(ROOT / "data" / "seed_ml.txt", n_docs=600):
        c.update(p.split(full_normalize(n, d)))
    return c


def _train(wc, **kw):
    return train_bpe(wc, vocab_target=1500, n_specials=len(DEFAULT_SPECIALS),
                     alphabet=select_alphabet(wc, forced=ML_BLOCK), log=lambda *_: None, **kw)


def test_constraint_forbids_mark_start_tokens(wc):
    res = _train(wc, token_ok=token_ok, seed_pairs=akshara_seed_pairs())
    merged = res.tokens[len(res.alphabet):]
    assert merged and all(token_ok(t) for t in merged) and res.info["n_seeded"] > 300


def test_plain_bpe_does_produce_mark_start_tokens(wc):
    merged = _train(wc).tokens
    assert any(not token_ok(t) for t in merged), "unconstrained BPE should show the failure mode"


def test_hf_export_matches_reference_encoder(wc):
    res = _train(wc, token_ok=token_ok, seed_pairs=akshara_seed_pairs())
    n, p = norm("ml"), pre("word")
    tok = build_tokenizer(res, specials=DEFAULT_SPECIALS, hf_normalizer=hf_full_normalizer(n), hf_pretokenizer=p.hf())
    rank = {m: i for i, m in enumerate(res.merges)}
    for t in SEED[:30]:
        ref = [x for w in p.split(full_normalize(n, t)) for x in reference_encode_word(w, rank)]
        assert tok.encode(t, add_special_tokens=False).tokens == ref, t


def test_roundtrip_including_byte_fallback(wc):
    res = _train(wc)
    n = norm("ml")
    tok = build_tokenizer(res, specials=DEFAULT_SPECIALS, hf_normalizer=hf_full_normalizer(n), hf_pretokenizer=pre("word").hf())
    assert tok.get_vocab_size() % 64 == 0
    for t in TEXTS + [" ", "  x", "x  ", "\n", "😀😀"]:
        assert tok.decode(tok.encode(t, add_special_tokens=False).ids) == n.normalize(t), repr(t)
    assert tok.token_to_id("<unk>") not in tok.encode("😀", add_special_tokens=False).ids


def test_boundary_penalty_discourages_crossing_merges():
    wcs = Counter({SP + "abcd": 100, SP + "abxy": 100})
    alpha = [SP, "a", "b", "c", "d", "x", "y"]
    kw = dict(vocab_target=len(DEFAULT_SPECIALS) + 256 + len(alpha) + 2, n_specials=len(DEFAULT_SPECIALS),
              alphabet=alpha, min_pair_count=1, log=lambda *_: None)
    pen = train_bpe(wcs, boundaries={"abcd": (2,), "abxy": (2,)}, boundary_lambda=1.0, **kw)
    assert ("b", "c") not in pen.merges[:2]


def test_is_combining():
    assert is_combining("\u0d3f") and is_combining("\u0d4d") and is_combining("\u200d") and not is_combining("ക")
