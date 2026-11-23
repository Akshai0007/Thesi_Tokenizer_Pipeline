import hashlib
import json
import math
from contextlib import contextmanager

import pytest

from experiments.mom_bpe import morphology_packet, replacement
from mltok import registry
from mltok.lib.bpe_core import DEFAULT_SPECIALS, train_bpe
from mltok.lib import bpe_progress
from mltok.lib.bpe_progress import BPEProgress
from mltok.lib.canonical_bpb import TorchWindowNLL, canonical_bpb
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


def scoring_backend():
    registry.load_plugins()
    transport = preprocessing(registry.make("normalizer", "ml_conservative"),
                              registry.make("pretokenizer", "script_runs"))
    return MOMBackend({
        "schema": SCHEMA, "runtime": runtime_identity(), "encoder": encoder_identity(),
        "segmenter": "grapheme", "specials": DEFAULT_SPECIALS, "initial": ["കൊ"], "merges": [],
        "preprocessing": json.loads(transport.to_str()), "vocab_size": 264,
    })


def test_canonical_bpb_counts_each_text_id_and_eos_and_weights_by_canonical_bytes():
    backend = scoring_backend()
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


def test_decoder_rejects_unallocated_and_invalid_ids_and_spells_reserved_controls():
    original = scoring_backend()
    backend = MOMBackend({**original.artifact, "vocab_size": original.vocab_size + 2})
    assert backend.decode([backend.special_ids["<pad>"], backend.eos_id]) == "<pad></s>"
    assert backend.decode([7 + 255]) == "\ufffd"
    assert backend.decode(backend.encode_batch(["🌻<pad>"])[0]) == "🌻<pad>"
    for bad in (-1, backend.vocab_size):
        with pytest.raises(ValueError, match="out of range"):
            backend.decode([bad])
    with pytest.raises(ValueError, match="unallocated"):
        backend.decode([original.vocab_size])
    for bad in (True, 3.0, "3"):
        with pytest.raises(TypeError, match="integers"):
            backend.decode([bad])


def test_reviewed_encoder_identity_remains_loadable():
    original = scoring_backend()
    identity = {
        **encoder_identity(),
        "mom_backend.py": "4b75e5886aea6ae1291e5170f1a3654c66d99fbc7c4711aee503d121079d022a",
        "bpe_core.py": "48d2546e2e92cf4ed5b584861654a8bfeed6643a3fd12dde9ffad0c58146876c",
    }
    restored = MOMBackend({**original.artifact, "encoder": identity})
    texts = ["കൊ", "🌻<pad>"]
    assert restored.encode_batch(texts) == original.encode_batch(texts)


def test_model_backed_bpb_matches_serial_unpadded_sliding_windows():
    import torch
    from torch.nn import functional as F

    backend = scoring_backend()
    pads_observed = []

    class TinyCausal(torch.nn.Module):
        def forward(self, *, input_ids, attention_mask):
            pads_observed.append(bool((attention_mask == 0).any()))
            positions = torch.arange(input_ids.shape[1], device=input_ids.device) + 1
            prefix = (input_ids * attention_mask * positions).cumsum(dim=1)
            ids = torch.arange(backend.vocab_size, device=input_ids.device)
            return torch.cos((prefix[..., None] + ids) / 71) * 3

    model = TinyCausal().eval()
    rows = []
    for idx, text in enumerate(("കൊ", "🌻<pad>")):
        canonical = backend.expected(text).encode()
        rows.append({"id": str(idx), "text": text, "canonical_bytes": len(canonical),
                     "canonical_sha256": hashlib.sha256(canonical).hexdigest()})
    ctx = 3
    measured = canonical_bpb(backend, rows, TorchWindowNLL(model, context=ctx, pad_id=0, batch_size=4))
    assert any(pads_observed)
    losses = []
    for row in rows:
        text_ids = backend.encode_batch([row["text"]])[0]
        history = [backend.special_ids["<s>"]]
        for target in [*text_ids, backend.eos_id]:
            inputs = torch.tensor([history[-ctx:]])
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs))[0, -1]
            losses.append(F.cross_entropy(logits[None, :], torch.tensor([target])).item())
            history.append(target)
    assert measured["canonical_bytes"] == 15 and measured["predicted_tokens"] == len(losses) == 12
    assert measured["bpb"] == pytest.approx(math.fsum(losses) / math.log(2) / 15, abs=1e-7)


def test_replacement_draft_preserves_page_identity_and_disjoint_splits(tmp_path):
    pages = []
    for pageid, text in enumerate(("മലയാളം", "തിരുവോണം", "കിളികൾ പറന്നു", "വീടുകൾ പണിയുന്നു"), 1):
        raw = text.encode()
        pages.append({"ns": 0, "pageid": pageid, "title": text, "revisions": [{
            "revid": 10 + pageid, "timestamp": "2025-01-01T00:00:00Z", "size": len(raw),
            "sha1": hashlib.sha1(raw).hexdigest(),
            "slots": {"main": {"contentmodel": "wikitext", "content": text}},
        }]})
    export = tmp_path / "source.json"
    export.write_bytes(json.dumps({"query": {"pages": pages}}).encode())
    out = tmp_path / "draft"
    doc = replacement.build(export, out)
    assert doc == replacement.build(export, out)
    assert not doc["verified_freeze"]
    train, held = (doc["panels"][k]["documents"] for k in ("train", "eval"))
    assert {r["group"] for r in train}.isdisjoint(r["group"] for r in held)
    assert len(train) == 3 and len(held) == 1
    assert all("text" not in r and "source_revision" in r for r in train + held)
    assert (out / "source-export.json").read_bytes() == export.read_bytes()
