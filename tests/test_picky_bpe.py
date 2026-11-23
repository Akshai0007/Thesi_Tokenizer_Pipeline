import json
from collections import Counter
from hashlib import sha256

import pytest

from experiments.mom_bpe.development_corpus import extract
from experiments.mom_bpe.performance import input_plan
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import BPEProgress
from mltok.lib.hybrid_units import initial_units, select_alphabet
from mltok.lib.orthographic_units import UNITIZERS
from mltok.lib.picky_backend import ARTIFACT, EventBackend
from mltok.lib.picky_events import EventProgram
from mltok.lib.picky_training import CorpusCounts, fit
from mltok.plugins.algorithm.picky_bpe import training_examples


def examples(words):
    return [(list(word), count, {}) for word, count in words.items()]


def test_reference_prunes_and_splits_during_training():
    # No tie in the first two choices. "ab" is used 200 of 217 times in
    # "abc". Its 17 residual occurrences must be split before the next merge.
    corpus = examples({"abc": 200, "abx": 10, "aby": 7, "bq": 20})
    program, stats = fit(list("abcqxy"), corpus, 8, threshold=900)
    assert program.events == [
        ("merge", "a", "b"), ("merge", "ab", "c"), ("remove", "ab", ["a", "b"]),
        ("merge", "b", "q"),
    ]
    assert program.replay(list("abcabx")) == ["abc", "a", "b", "x"]
    assert stats["n_removed"] == 1 and "ab" not in program.vocabulary()
    assert stats["standalone_train_tokens"] == 200 + 30 + 21 + 20


def test_exact_ios_threshold_and_self_merge_count():
    # The public reference uses strict >. 9 out of 10 is kept at .900.
    corpus = examples({"abc": 9, "abx": 1})
    keep, _ = fit(list("abcx"), corpus, 6, threshold=900)
    prune, _ = fit(list("abcx"), corpus, 6, threshold=899)
    assert not keep.removals
    assert prune.removals["ab"] == [2]
    # Four initial a's merge into two aa's; one non-overlapping aa-aa merge is
    # sufficient for reference IoS 1 when neither aa remains.
    same, _ = fit(["a"], examples({"aaaa": 10}), 3, threshold=900)
    assert same.events[2] == ("remove", "aa", ["a", "a"])
    assert same.replay(list("aaa")) == ["a", "a", "a"]


def test_event_chronology_reactivation_and_nested_split_back():
    events = [
        ("merge", "a", "b"), ("merge", "ab", "c"), ("remove", "ab", ["a", "b"]),
        ("merge", "a", "b"), ("merge", "abc", "d"), ("remove", "abc", ["ab", "c"]),
        ("merge", "ab", "e"), ("remove", "ab", ["a", "b"]),
    ]
    active = EventProgram(list("abcde"), events)
    assert active.replay(list("abcde")) == ["abcd", "e"]
    assert active.replay(list("abcabe")) == ["a", "b", "c", "abe"]
    assert active.replay(list("abx")) == ["a", "b", "x"]
    assert not {"ab", "abc"} & set(active.vocabulary())
    broken = events[:-1] + [("remove", "ab", ["abc"])]
    with pytest.raises(ValueError, match="split-back"):
        EventProgram(list("abcde"), broken)
    with pytest.raises(ValueError, match="inactive"):
        EventProgram(list("abcde"), events + [("merge", "ab", "a")])


def test_pair_index_matches_independent_recount_with_split_and_offsets():
    program = EventProgram(list("abcqx"))
    corpus = [(["a", "b", "c"], 71, {1: 800, 2: 1000}), (["a", "b", "x"], 3, {1: 900}),
              (["c", "c", "c"], 2, {1: 500, 2: 1000})]
    index = CorpusCounts(corpus, program, 400)
    assert index.crossings["c", "c"] == 1000  # Only the first cc is merged.
    events = [("merge", "a", "b"), ("merge", "ab", "c"), ("remove", "ab", ["a", "b"])]
    from mltok.lib.picky_events import merge, split

    for kind, token, value in events:
        program.append(kind, token, value)
        if kind == "merge":
            index.replace(index.by_pair[token, value], lambda line: merge(line, token, value))
        else:
            index.replace(index.by_token[token], lambda line: split(line, token, value))
        actual = Counter()
        for symbols, frequency, _ in corpus:
            final = program.replay(symbols)
            for pair in zip(final, final[1:]):
                actual[pair] += frequency
        assert index.pairs == actual
    assert index.crossings["a", "b"] == 3 * 900


def test_hybrid_atom_and_unicode_scalar_backoff():
    train = {"കുട്ടികൾ": 10, "കടലിൽ": 2}
    alphabet, atoms, mass = select_alphabet(train, "hybrid", 150, 1.0, 5, 40)
    units = initial_units("hybrid", atoms)
    assert "ട്ടി" in atoms and "ട്ടി" in units("കുട്ടികൾ")
    assert set("ട്ട") <= set(alphabet)
    for word in ("കൂട്ടിൽ", "ൻ്റ", "ന്‍", "ൎക്കം", "ക്‌ഷ", "👩🏽‍🔬", "\u0d00"):
        sequence = units(word)
        assert "".join(sequence) == word
        assert set(atoms) & set(sequence) <= set(UNITIZERS["akshara"](word))
    assert mass == 1.0


def context(tmp_path, corpus, output, *, vocab=320):
    registry.load_plugins()
    return TrainContext(
        vocab=vocab, specials=DEFAULT_SPECIALS,
        normalizer=registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True}),
        pretokenizer=registry.make("pretokenizer", {"type": "script_runs", "scripts": ["Malayalam", "Latin"]}),
        word_counts=None, shard_paths=[str(corpus)], out_dir=tmp_path / output,
        resources={"merge_checkpoint_every": 1},
    )


def test_encoder_resume_and_exact_reload_after_damaged_newest_generation(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "കിളികൾ കുട്ടികൾ കടലിൽ " * 20}, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = {"type": "picky_bpe", "picky_threshold_milli": 900, "hybrid_min_count": 5}
    full_ctx = context(tmp_path, data, "full")
    algo = registry.make("algorithm", cfg)
    full, _ = algo.train(full_ctx)
    interrupted = context(tmp_path, data, "resumed")
    original = BPEProgress.save
    generations = 0

    def crash(self, events, seqs, flags):
        nonlocal generations
        original(self, events, seqs, flags)
        generations += 1
        if generations == 2:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(BPEProgress, "save", crash)
    with pytest.raises(RuntimeError, match="interruption"):
        algo.train(interrupted)
    files = sorted((interrupted.out_dir / "event-checkpoints").glob("*.gz"))
    files[-1].write_bytes(b"incomplete")
    monkeypatch.setattr(BPEProgress, "save", original)
    resumed, info = algo.train(interrupted)
    assert info["resumed_events"] > 0
    assert (full_ctx.out_dir / ARTIFACT).read_bytes() == (interrupted.out_dir / ARTIFACT).read_bytes()
    loaded = EventBackend.load(interrupted.out_dir)
    texts = ["", "കിളികൾ", "പൂക്കൾ\r\n🙂 <pad> ▁", "ക്‌ഷ ೞ", "പിന്‍കോഡ്"]
    assert full.encode_batch(texts) == resumed.encode_batch(texts) == loaded.encode_batch(texts)
    assert [loaded.decode(ids) for ids in loaded.encode_batch(texts)] == list(map(loaded.expected, texts))
    assert files[-1].read_bytes() == b"incomplete"


def test_hybrid_morphology_projects_only_consensus_reference_sites(tmp_path):
    word = "കിളികൾ"
    row = {"parts": ["കിളി", "കൾ"], "operation": "affix", "reviewers": ["test one", "test two"],
           "confidence_milli": 800}
    corpus = [(word, 10)]
    # Backoff scalars expose every character offset, but the independent
    # reference still rejects a split inside ളി.
    unitize = initial_units("hybrid", ())
    cases = [row, {**row, "parts": [word[:3], word[3:]]}]
    data, _, seen, usable = training_examples(corpus, set(word), unitize, {word: [row]})
    assert data == [(list(word), 10, {len("കിളി"): 800})] and seen == usable == 1
    assert training_examples(corpus, set(word), unitize, {word: cases})[-1] == 0
    candidates = [(["a", "b"], 100, {1: 1000}), (["a", "c"], 80, {})]
    assert fit(list("abc"), candidates, 4)[0].events[0] == ("merge", "a", "b")
    assert fit(list("abc"), candidates, 4, penalty=400)[0].events[0] == ("merge", "a", "c")


def test_weighted_and_combined_plugins_accept_only_pinned_reviewed_fixtures(tmp_path):
    text = "കിളികൾ കടലിൽ കുട്ടികൾ കിളികൾ പൂക്കൾ " * 12
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"text": text}, ensure_ascii=False) + "\n", encoding="utf-8")
    configuration = {"type": "picky_bpe", "hybrid_min_count": 3}
    fixture = tmp_path / "alignment.json"
    payload = {
        "schema": "mom-surface-alignments-v1", "qualification": "adjudicated_training",
        "forms": {"കിളികൾ": [{"parts": ["കിളി", "കൾ"], "operation": "affix",
                              "reviewers": ["synthetic test one", "synthetic test two"],
                              "confidence_milli": 800}]},
    }
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    pinned = {"alignment_file": str(fixture), "alignment_sha256": sha256(fixture.read_bytes()).hexdigest()}
    control = context(tmp_path, train, "control")
    original, _ = registry.make("algorithm", configuration).train(control)
    zero, _ = registry.make("algorithm", {**configuration, **pinned, "picky_threshold_milli": 1000}).train(
        context(tmp_path, train, "explicit-off"),
    )
    assert zero.artifact["events"] == original.artifact["events"]
    assert zero.encode_batch([text]) == original.encode_batch([text])
    for name, refinement in (("morphology", None), ("combined", 900)):
        candidate = {**configuration, **pinned, "morphology_penalty_milli": 400, "picky_threshold_milli": refinement}
        model, report = registry.make("algorithm", candidate).train(context(tmp_path, train, name))
        assert report["supported_morphology_forms"] == 1
        assert model.decode(model.encode_batch([text])[0]) == model.expected(text)
    fixture.write_text(json.dumps({**payload, "qualification": "automated"}), encoding="utf-8")
    with pytest.raises(ValueError, match="adjudicated"):
        registry.make("algorithm", {**configuration, **pinned, "morphology_penalty_milli": 400}).train(
            context(tmp_path, train, "blocked"),
        )


def test_extraction_omits_quotes_media_and_citations():
    lines = "മലയാള ഭാഷയുടെ ചരിത്രവും സംസ്കാരവും രേഖപ്പെടുത്തുന്നതിന് പ്രാദേശിക ഗ്രന്ഥങ്ങൾക്ക് പ്രാധാന്യമുണ്ട്. " * 2
    page = f"<p>{lines}<sup>citation</sup></p><blockquote><p>{lines}</p></blockquote>"
    page += f'<p>{lines}<q>quotation</q></p><table><p>{lines}</p></table>'
    assert extract(page) == lines.strip()


def test_development_manifest_cannot_authorize_weighted_or_frozen_fit(tmp_path):
    registry.load_plugins()
    norm = {"type": "ml_conservative", "terminal_chillu": True}
    empty = tmp_path / "manifest.json"
    empty.write_text(json.dumps({"schema": "mom-tokenizer-development-v1", "verified_freeze": False,
                                 "qualification": "EXPLORATORY_WIKIMEDIA_DEVELOPMENT",
                                 "recipe": {"normalizer_fingerprint": registry.fingerprint("normalizer", norm)}}))
    with pytest.raises(ValueError, match="adjudicated"):
        input_plan(empty, "development", {"morphology_penalty_milli": 400}, norm, 48000)
    with pytest.raises(ValueError, match="independently reviewed"):
        input_plan(empty, "frozen", {}, norm, 48000)
