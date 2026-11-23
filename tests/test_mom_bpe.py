import hashlib
import json
from collections import Counter
from itertools import accumulate

import pytest

from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS, merge_units, train_bpe
from mltok.lib.exposure import account_exposure
from mltok.lib.mom_backend import MOMBackend, SurfaceEncoding
from mltok.lib.morph_alignment import project, reviewed_mlmorph_candidates
from mltok.lib.orthographic_units import UNITIZERS, merge_domains
from mltok.lib.surface_scoring import score_morphology


def plugin(kind, name):
    registry.load_plugins()
    return registry.make(kind, name)


@pytest.mark.parametrize(("source", "normalized"), [
    ("കൊ", "കൊ"), ("അവന്\u200d.", "അവൻ."), ("ക്\u200dഷ", "ക്\u200dഷ"),
    ("ന്\u200dമ", "ന്\u200dമ"), ("ക്ന്\u200d", "ക്ന്\u200d"),
    ("ക്\u200dന്\u200d", "ക്\u200dന്\u200d"), ("ന്\u200d\u0301", "ന്\u200d\u0301"),
    ("ക്\u200cഷ", "ക്\u200cഷ"), ("കണ്ടു്", "കണ്ടു്"), ("മൎ ദൔ ദൕ ദൖ", "മൎ ദൔ ദൕ ദൖ"),
    ("ക്\u200d.", "ക്\u200d."), ("ന്\u200dറ", "ന്\u200dറ"),
    ("ന്\u200c ം\u200b \ufeff\u00ad\u2060\r\n", "ന്\u200c ം\u200b \ufeff\u00ad\u2060\r\n"),
])
def test_conservative_normalization_and_hf_parity(source, normalized):
    n = plugin("normalizer", "ml_conservative")
    assert n.normalize(source) == n.hf().normalize_str(source) == normalized
    assert n.normalize(normalized) == n.hf().normalize_str(normalized) == normalized


def test_nfc_only_disables_legacy_conversion():
    n = plugin("normalizer", {"type": "ml_conservative", "terminal_chillu": False})
    assert n.normalize("ന്\u200d കൊ") == n.hf().normalize_str("ന്\u200d കൊ") == "ന്\u200d കൊ"


@pytest.mark.parametrize(("text", "pieces"), [
    ("abമലയാളംz", ["ab", "മലയാളം", "z"]),
    ("x\u0301മലയാളം", ["x\u0301", "മലയാളം"]),
    ("ക2൧,!", ["ക", "2", "൧", ",", "!"]),
    ("A  2️⃣👩\u200d💻", ["A", "  ", "2️⃣", "👩\u200d💻"]),
    ("▁ക ▁c", ["▁", "ക", " ", "▁", "c"]),
])
def test_script_runs_preserve_text_and_match_hf(text, pieces):
    p = plugin("pretokenizer", "script_runs")
    assert p.split(text) == [s for s, _ in p.hf().pre_tokenize_str(text)] == pieces
    assert "".join(pieces) == text


def test_script_extensions_and_multilingual_configuration():
    p = plugin("pretokenizer", {"type": "script_runs", "scripts": ["Hiragana", "Katakana", "Devanagari", "Latin"]})
    assert p.split("あーアーaक्षय") == ["あー", "アー", "a", "क्षय"]


@pytest.mark.parametrize("scripts", [("Hiragana", "Katakana"), ("Katakana", "Hiragana")])
def test_shared_letter_prefix_uses_the_following_primary_script(scripts):
    p = plugin("pretokenizer", {"type": "script_runs", "scripts": scripts})
    assert p.split("ーあ ーア あーア") == ["ーあ", " ", "ーア", " ", "あー", "ア"]


def test_script_fallback_does_not_cut_protected_clusters():
    p = plugin("pretokenizer", "script_runs")
    assert list(merge_domains("abक्षx", p.split)) == ["ab", "क्ष", "x"]


@pytest.mark.parametrize(("surface", "expected"), [
    ("ക്ഷി", ["ക്ഷി"]), ("ക്\u200dഷ", ["ക്\u200dഷ"]),
    ("ക്\u200cഷ", ["ക്\u200c", "ഷ"]), ("കണ്ടു്", ["ക", "ണ്ടു്"]),
    ("ൻക", ["ൻ", "ക"]), ("മലയാളം", ["മ", "ല", "യാ", "ളം"]),
    ("ൻ്റ", ["ൻ്റ"]), ("ൎക്കു്", ["ൎക്കു്"]), ("ൕ്ക", ["ൕ്ക"]),
    ("ക\u0d00ക", ["ക\u0d00", "ക"]),
])
def test_malayalam_initial_units(surface, expected):
    assert UNITIZERS["akshara"](surface) == expected
    assert "".join(expected) == surface


def test_grapheme_and_akshara_references_differ():
    surface = "ക\u200cക"
    assert UNITIZERS["grapheme"](surface) != UNITIZERS["akshara"](surface)


def evidence(parts, *, realized=None, operation="concatenation", confidence=900, two_reviewers=True):
    row = {"parts": parts, "confidence_milli": confidence, "operation": operation,
           "reviewers": ["synthetic-A", "synthetic-B"] if two_reviewers else ["synthetic-A"]}
    if realized is not None:
        row["surface_parts"] = realized
    return row


@pytest.mark.parametrize(("surface", "parts", "realized", "operation"), [
    ("പൂക്കൾ", ["പൂ", "ക്കൾ"], None, "affix"),
    ("കുട്ടികൾക്കും", ["കുട്ടി", "കൾ", "ക്കും"], None, "agglutination"),
    ("കടൽക്കര", ["കടൽ", "കര"], ["കടൽ", "ക്കര"], "gemination"),
    ("വീട്ടിൽ", ["വീട്", "ഇൽ"], ["വീ", "ട്ടിൽ"], "stem_mutation"),
])
def test_reviewed_surface_exact_and_allomorph_projection(surface, parts, realized, operation):
    row = evidence(parts, realized=realized, operation=operation)
    assert project(surface, [row], UNITIZERS["akshara"]) == {
        pos: 900 for pos in list(accumulate(map(len, realized or parts)))[:-1]
    }


def test_projection_abstains_on_fused_sandhi_and_conflicting_or_unreviewed_candidates():
    units = UNITIZERS["akshara"]
    assert project("ക്ഷേത്രോത്സവം", [evidence(["ക്ഷേത്രം", "ഉത്സവം"], operation="sandhi")], units) == {}
    assert project("ക്ഷി", [evidence(["ക്", "ഷി"])], units) == {}
    assert project("മലയാളം", [evidence(["മല", "യാളം"]), evidence(["മലയാ", "ളം"])], units) == {}
    assert project("മലയാളം", [evidence(["മല", "യാളം"], two_reviewers=False)], units) == {}
    assert project("മലയാളം", [evidence(["മല", "യാളം"]), evidence(["മല", "യാളം"], confidence=500)],
                   units) == {2: 500}


@pytest.mark.parametrize("reviewers", [["Person", " person"], ["Person", "person"], ["", "other"]])
def test_same_reviewer_alias_or_empty_identifier_does_not_authorize_projection(reviewers):
    row = {**evidence(["പൂ", "ക്കൾ"]), "reviewers": reviewers}
    assert project("പൂക്കൾ", [row], UNITIZERS["akshara"]) == {}


def test_mlmorph_joins_only_exact_reviewed_readings_and_ignores_analyzer_rank():
    from types import SimpleNamespace

    row = {**evidence(["പൂ", "ക്കൾ"]), "analysis": "പൂ<n><pl>"}
    analyzer = SimpleNamespace(analyse=lambda _: [(row["analysis"], 192)])
    assert reviewed_mlmorph_candidates("പൂക്കൾ", analyzer, [row]) == [row]
    analyzer.analyse = lambda _: [(row["analysis"], 1), ("unmatched", 200)]
    assert reviewed_mlmorph_candidates("പൂക്കൾ", analyzer, [row]) == []


def test_exact_penalty_changes_only_rank_priority_and_zero_control_is_exact():
    words = {"ab": 10, "cd": 8}
    params = {"alphabet": list("abcd"), "vocab_target": 262, "n_specials": 1, "initial_units": list}
    plain = train_bpe(words, **params)
    zero = train_bpe(words, boundary_weights={"ab": {1: 1000}}, penalty_milli=0, **params)
    soft = train_bpe(words, boundary_weights={"ab": {1: 1000}}, penalty_milli=500, **params)
    assert plain.merges == zero.merges == [("a", "b")]
    assert soft.merges == [("c", "d")]


def test_morphology_does_not_change_the_raw_minimum_pair_count():
    res = train_bpe({"ab": 2, "cd": 1}, vocab_target=262, n_specials=1,
                    alphabet=list("abcd"), initial_units=list, penalty_milli=500,
                    boundary_weights={"ab": {1: 1000}})
    assert res.merges == [("a", "b")]


def test_atomic_output_identity_and_rank_replay_with_self_overlap():
    units = lambda w: ["aa", "a"] if w == "aaa" else list(w)
    training = train_bpe({"aaa": 10, "aab": 9, "bbbbb": 8}, alphabet=["a", "aa", "b"],
                         n_specials=1, vocab_target=270, initial_units=units)
    assert ("a", "a") not in training.merges
    rank = {pair: i for i, pair in enumerate(training.merges)}
    assert "".join(merge_units(units("bbbbb"), rank)) == "bbbbb"
    assert len(training.merges) == len(set(a + b for a, b in training.merges))


def test_heap_trainer_matches_independent_weighted_recount_and_replay():
    words = {"aaabbc": 13, "aacabb": 9, "abacbc": 7, "ccccb": 4}
    weights = {"aaabbc": {2: 700, 3: 1000}, "abacbc": {4: 500}}
    alphabet = list("abc")
    result = train_bpe(words, vocab_target=274, n_specials=1, alphabet=alphabet,
                       initial_units=list, penalty_milli=400, boundary_weights=weights)
    table = dict(zip(alphabet, range(len(alphabet))))
    split = {word: list(word) for word in words}
    for pair in result.merges:
        scores = Counter()
        for word, parts in split.items():
            for pos, a, b in zip(accumulate(map(len, parts)), parts, parts[1:]):
                if a + b not in table:
                    scores[a, b] += words[word] * (1_000_000 - 400 * weights.get(word, {}).get(pos, 0))
        expected = min(scores, key=lambda x: (-scores[x], tuple(map(table.get, x))))
        assert pair == expected
        table["".join(pair)] = len(table)
        for word, parts in split.items():
            updated = []
            i = 0
            while i < len(parts):
                if tuple(parts[i:i + 2]) == pair:
                    updated.append("".join(pair))
                    i += 2
                else:
                    updated.append(parts[i])
                    i += 1
            split[word] = updated
    assert all(merge_units(list(word), {p: i for i, p in enumerate(result.merges)}) == parts
               for word, parts in split.items())


def fixture_context(tmp_path, name="run"):
    shard = tmp_path / "sample.jsonl"
    texts = ["പൂക്കൾ മലയാളം കടൽക്കര ക്ഷേത്രത്തിൽ", "abമലയാളം മലയാളം പൂക്കൾ▁ <pad> <|im_start|>"] * 5
    shard.write_text("\n".join(json.dumps({"text": t}) for t in texts), encoding="utf-8")
    return TrainContext(
        vocab=320, specials=DEFAULT_SPECIALS, normalizer=plugin("normalizer", "ml_conservative"),
        pretokenizer=plugin("pretokenizer", "script_runs"), word_counts=None, shard_paths=[str(shard)],
        out_dir=tmp_path / name, log=lambda *_: None,
    )


@pytest.mark.parametrize("segmenter", ["codepoint", "grapheme", "akshara"])
def test_artifact_reload_rank_parity_and_raw_versus_canonical_roundtrip(tmp_path, segmenter):
    algo = plugin("algorithm", {"type": "mom_bpe", "segmenter": segmenter})
    ctx = fixture_context(tmp_path)
    backend, info = algo.train(ctx)
    assert info["documents"] == 10
    texts = ["", " ", "\r\n", " ക▁ക <pad> <|im_start|>  ", "👩\u200d💻 汉字", "കൊ", "അവന്\u200d"]
    ids = backend.encode_batch(texts)
    assert algo.load(ctx.out_dir, ctx.normalizer).encode_batch(texts) == ids
    assert [backend.decode(row) for row in ids] == [*texts[:5], "കൊ", "അവൻ"]
    assert not set(backend.special_ids.values()).intersection(i for row in ids for i in row)
    assert backend.vocab_size == 320 and info["occupied_vocab"] <= 320
    algo.train(ctx)  # Same inputs produce an identical durable artifact.
    assert MOMBackend.load(ctx.out_dir).encode_batch(texts) == ids


def test_weighted_arm_verifies_artifact_and_counts_supported_forms(tmp_path):
    ctx = fixture_context(tmp_path)
    path = tmp_path / "reviewed.json"
    data = json.dumps({"schema": "mom-surface-alignments-v1",
                       "forms": {"പൂക്കൾ": [evidence(["പൂ", "ക്കൾ"])]}}).encode()
    path.write_bytes(data)
    config = {"type": "mom_bpe", "alignment_file": str(path), "alignment_sha256": hashlib.sha256(data).hexdigest()}
    weighted = plugin("algorithm", {**config, "morphology_penalty_milli": 400})
    disabled = plugin("algorithm", config)
    baseline = plugin("algorithm", "mom_bpe")
    model, result = weighted.train(ctx)
    assert result["supported_morphology_forms"] == 1
    control, _ = disabled.train(fixture_context(tmp_path, "zero"))
    vanilla, _ = baseline.train(fixture_context(tmp_path, "orthographic"))
    assert control.artifact["merges"] == vanilla.artifact["merges"]
    assert model.artifact["provenance"]["alignment_sha256"] == config["alignment_sha256"]
    tiny_alphabet = plugin("algorithm", {**config, "morphology_penalty_milli": 400, "alphabet_coverage": 0.01})
    with pytest.raises(ValueError, match="reachable boundaries"):
        tiny_alphabet.train(fixture_context(tmp_path, "pruned"))
    path.write_bytes(data + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        weighted.train(ctx)


def test_mom_metrics_use_canonical_offsets_and_exclude_fallback(tmp_path):
    model, _ = plugin("algorithm", "mom_bpe").train(fixture_context(tmp_path))
    metric = plugin("metric", "mom_surface")
    out = metric.compute(model, [("ക്ഷേത്രത്തിൽ മലയാളം", "fixture"), ("കൊ🌻", "fixture"), ("ക്\u200cഷ", "fixture")])
    assert out["mom_raw_equal"] == 2 and out["mom_canonical_equal"] == 3
    assert out["mom_akshara_breaks"] == 0
    assert out["mom_grammar_breaks"] == 0
    assert out["mom_byte_fallback_ids"] > 0 and out["mom_offset_exclusions"] > 0


def test_exposure_counts_canonical_bytes_and_refuses_partial_byte_estimates(tmp_path):
    model, _ = plugin("algorithm", "mom_bpe").train(fixture_context(tmp_path))
    rows = [{"id": "one", "text": "🌻"}, {"id": "two", "text": "കൊ"}]
    all_docs = account_exposure(model, rows)
    assert all_docs["predicted_eos"] == 2
    assert all_docs["full_raw_bytes"] > all_docs["full_canonical_bytes"]
    assert account_exposure(model, rows, 1)["partial"]["canonical_bytes"] is None
    assert account_exposure(model, rows, 4)["partial"]["canonical_bytes"] == 4
    with pytest.raises(ValueError, match="cannot fill"):
        account_exposure(model, rows, 100)


def test_existing_model_cannot_be_overwritten(tmp_path):
    algo = plugin("algorithm", "mom_bpe")
    ctx = fixture_context(tmp_path)
    model, _ = algo.train(ctx)
    path = ctx.out_dir / "mom-bpe.json"
    original = path.read_bytes()
    other = MOMBackend({**model.artifact, "vocab_size": model.vocab_size + 1})
    with pytest.raises(FileExistsError):
        other.save(ctx.out_dir)
    assert path.read_bytes() == original


def test_artifact_refuses_a_different_encoder_identity(tmp_path):
    model, _ = plugin("algorithm", "mom_bpe").train(fixture_context(tmp_path))
    with pytest.raises(ValueError, match="encoder"):
        MOMBackend({**model.artifact, "encoder": {"unknown": "sha256"}})


def test_morphology_scoring_separates_coverage_and_consistency():
    rows = [{"status": "adjudicated", "text": t, "group": "pu", "offsets": [2], "stem_end": 2}
            for t in ("പൂക്കൾ", "പൂക്കളും", "പൂക്കളിൽ")]
    rows.append({"status": "unresolved", "text": "പൂ"})

    def encode(text):
        ids = [300, 301] if text != "പൂക്കളിൽ" else [300, 10]
        return SurfaceEncoding(text, ids, [], [(0, 2), (2, len(text))])

    result = score_morphology(encode, {10}, rows)
    assert result["adjudicated_forms"] == 3 and result["unmeasurable_forms"] == 1
    assert result["matched_boundaries"] == 2 and result["micro_recall"] == 2 / 3
    assert (result["repeated_stem_pairs"], result["isolatable_stem_pairs"], result["consistent_stem_pairs"]) == (3, 1, 1)
