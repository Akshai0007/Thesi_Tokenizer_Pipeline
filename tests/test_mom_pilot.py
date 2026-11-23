import hashlib
import math
from contextlib import contextmanager

import pytest

from experiments.mom_bpe import morphology_packet
from mltok import registry
from mltok.lib.bpe_core import DEFAULT_SPECIALS, train_bpe
from mltok.lib import bpe_progress
from mltok.lib.bpe_progress import BPEProgress
from mltok.lib.canonical_bpb import canonical_bpb
from mltok.lib.mom_backend import MOMBackend, SCHEMA, encoder_identity, preprocessing, runtime_identity


def test_durable_weighted_resume_reconstructs_the_exact_merge_order(tmp_path, monkeypatch):
    path = tmp_path / "progress"
    progress = BPEProgress(path, "test", 1)
    options = dict(alphabet=list("abcd"), n_specials=1, vocab_target=268, initial_units=list,
                   penalty_milli=400, boundary_weights={"abba": {2: 700}})
    counts = {"abba": 8, "bccd": 6, "acda": 5, "aaaab": 3}
    clean = train_bpe(counts, **options)
    save = progress.save

    def interrupted(merges, *state):
        save(merges, *state)
        if len(merges) == 2:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(progress, "save", interrupted)
        with pytest.raises(KeyboardInterrupt):
            train_bpe(counts, progress=progress, **options)
    assert len(progress.latest()["merges"]) == 2
    clean_resume = train_bpe(counts, progress=progress, **options)
    assert clean_resume.info["resumed_merges"] == 2
    assert clean_resume.merges == clean.merges and clean_resume.tokens == clean.tokens


def test_checkpoint_corruption_falls_back_without_retiring_other_files(tmp_path):
    progress = BPEProgress(tmp_path, "test", 1)
    progress.save([("a", "b")], [[2]], [None])
    progress.save([("a", "b"), ("ab", "c")], [[3]], [None])
    prior, new = sorted(tmp_path.glob("merge-*.json.gz"))
    original = prior.read_bytes()
    new.write_bytes(b"truncated")
    assert len(progress.latest()["merges"]) == 1
    with pytest.raises(ValueError, match="No usable"):
        BPEProgress(tmp_path, "different input", 1).latest()
    assert prior.read_bytes() == original and new.read_bytes() == b"truncated"


def test_failed_checkpoint_write_preserves_last_generation(tmp_path, monkeypatch):
    progress = BPEProgress(tmp_path, "test", 1)
    progress.save([("a", "b")], [[2]], [None])
    previous, = tmp_path.glob("merge-*.json.gz")
    data = previous.read_bytes()
    atomic = bpe_progress.atomic_writer

    @contextmanager
    def interrupted(path, **kwargs):
        with atomic(path, **kwargs) as stream:
            yield stream
            raise OSError("interrupted before publication")

    monkeypatch.setattr(bpe_progress, "atomic_writer", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        progress.save([("a", "b"), ("ab", "c")], [[3]], [None])
    assert list(tmp_path.iterdir()) == [previous]
    assert previous.read_bytes() == data


def test_morphology_packet_skips_blind_exposure_and_keeps_abstentions(tmp_path):
    result = morphology_packet.build(tmp_path)
    assert result["authorized_training_labels"] == 0 and result["forms_to_review"] == 108
    blind = (tmp_path / "reviewer-1.tsv").read_bytes()
    assert blind == (tmp_path / "reviewer-2.tsv").read_bytes()
    assert b"<locative>" not in blind and b"multiple_analyzer_candidates" not in blind
    assert b"multiple_analyzer_candidates" in (tmp_path / "evidence-after-blind-review.json").read_bytes()
    assert morphology_packet.build(tmp_path) == result


def test_canonical_bpb_counts_each_text_id_and_eos_and_weights_by_canonical_bytes():
    import json

    registry.load_plugins()
    transport = preprocessing(registry.make("normalizer", "ml_conservative"),
                              registry.make("pretokenizer", "script_runs"))
    backend = MOMBackend({
        "schema": SCHEMA, "runtime": runtime_identity(), "encoder": encoder_identity(),
        "segmenter": "grapheme", "specials": DEFAULT_SPECIALS, "initial": ["കൊ"], "merges": [],
        "preprocessing": json.loads(transport.to_str()), "vocab_size": 264,
    })
    texts = ["കൊ", "🌻<pad>"]
    canonical = ["കൊ", "🌻<pad>"]
    rows = [{"id": str(i), "text": text, "canonical_bytes": len(c.encode()),
             "canonical_sha256": hashlib.sha256(c.encode()).hexdigest()}
            for i, (text, c) in enumerate(zip(texts, canonical))]
    calls = []

    def nll(inputs, targets):
        calls.append((inputs, targets))
        assert inputs[0] == backend.special_ids["<s>"] and targets[-1] == backend.eos_id
        assert inputs[1:] == targets[:-1]
        assert set(targets[:-1]).isdisjoint(backend.special_ids.values())
        # Different per-document NLLs distinguish byte weighting from averaging.
        return [len(calls) * math.log(2)] * len(targets)

    out = canonical_bpb(backend, rows, nll)
    expected_bits = 2 + 2 * 10  # Lexical കൊ plus EOS, then nine fallback bytes plus EOS.
    assert out["canonical_bytes"] == 15 and out["predicted_tokens"] == 12
    assert out["bpb"] == pytest.approx(expected_bits / 15)
    with pytest.raises(ValueError, match="canonical bytes"):
        canonical_bpb(backend, [{**rows[0], "canonical_bytes": 9}], nll)
    with pytest.raises(ValueError, match="one finite"):
        canonical_bpb(backend, rows, lambda *_: [])
    with pytest.raises(ValueError, match="one finite"):
        canonical_bpb(backend, rows, lambda _, targets: [math.nan] * len(targets))
