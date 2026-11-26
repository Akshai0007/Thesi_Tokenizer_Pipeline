import json
from types import SimpleNamespace

import pytest

from experiments.mom_bpe import frontend_analysis
from experiments.mom_bpe.diagnose import sha
from experiments.mom_bpe.frontend_analysis import load_fit, site_reference, token_payloads
from experiments.mom_bpe.frontend_fit import pretokenizer_spec
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import BPEProgress, canonical
from mltok.lib.mom_backend import SurfaceEncoding
from mltok.lib.picky_backend import ARTIFACT


@pytest.mark.parametrize("arm", ["F-space", "F-word-space"])
def test_leading_space_only_attaches_the_existing_piece(arm):
    spec, _, _ = pretokenizer_spec(arm)
    frontend = registry.make("pretokenizer", spec)
    text = "മലയാളം  മലയാളം മലയാളം\t\nമലയാളം"
    assert frontend.split(text) == ["മലയാളം", "  ", "മലയാളം", " മലയാളം", "\t\n", "മലയാളം"]
    assert frontend.split(" a") == [" a"]
    assert frontend.split(" a\u0301") == [" a\u0301"]
    assert frontend.split(" ") == [" "]


@pytest.mark.parametrize("arm", ["F-space", "F-word", "F-word-space"])
def test_frontend_reloads_and_reconstructs_exact_bytes(tmp_path, arm):
    spec, _, _ = pretokenizer_spec(arm)
    pre = registry.make("pretokenizer", spec)
    norm = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    algorithm = registry.make("algorithm", "boundary_guarded_hybrid")
    data = tmp_path / "source.jsonl"
    text = "മലയാളം മലയാളംABC! മലയാളം മലയാളം  മലയാളം"
    data.write_bytes((json.dumps({"text": text * 25}, ensure_ascii=False) + "\n").encode())
    model, _ = algorithm.train(TrainContext(300, DEFAULT_SPECIALS, norm, pre, tmp_path, None, [str(data)]))
    restored = algorithm.load(tmp_path, norm)
    for case in (text, "മലയാളം  ▁\t👩‍🔬 <s> എ\u0301 "):
        encoded = model.encode_offsets(case)
        assert encoded == restored.encode_offsets(case)
        assert model.decode(encoded.ids) == norm.normalize(case)
        assert not set(encoded.ids) & set(model.special_ids.values())
        assert b"".join(row[-1] for row in token_payloads(encoded, model, False)) == encoded.text.encode()


def test_native_payload_cuts_distinguish_bytes_from_reference_sites():
    encoded = SurfaceEncoding("a കാ", [1, 2, 3, 4, 5],
                              ["▁a", "▁", "<0xE0>", "<0xB4>", "<0x95>ാ"], [])
    invalid = SimpleNamespace(fallback_ids={3, 4, 5})
    with pytest.raises(ValueError, match="byte spelling"):
        list(token_payloads(encoded, invalid, True))
    encoded = SurfaceEncoding("a കാ", [1, 2, 3, 4, 5, 6],
                              ["▁a", "▁", "<0xE0>", "<0xB4>", "<0x95>", "ാ"], [])
    pieces = list(token_payloads(encoded, invalid, True))
    refs, scalars = site_reference(encoded.text)
    assert b"".join(row[-1] for row in pieces) == encoded.text.encode()
    assert refs["akshara"] == {5}
    assert pieces[2][1] not in scalars
    assert pieces[4][1] == 5


def test_frontend_fit_rejects_a_relabelled_evaluation_receipt(tmp_path):
    identity = {"arm": "T8", "manifest_sha256": "train", "validation_stream_sha256": "eval"}
    report = {"fit": {"identity": identity, "occupied_vocab": 48000}}
    report["result_sha256"] = sha(canonical(report))
    (tmp_path / "validation.json").write_bytes(canonical(report))
    with pytest.raises(ValueError, match="verified 48K fit"):
        load_fit("F-space", tmp_path, "train", "eval")


def test_saved_analysis_reopens_with_identical_histogram_checksums(tmp_path, monkeypatch):
    pieces = ["a", "bc", "defghijklm"]
    text = "".join(pieces)
    ids = [1, 2, 3]
    model = SimpleNamespace(fallback_ids=set(), special_ids={}, decode=lambda _: text,
                            encode_offsets=lambda _: SurfaceEncoding(text, ids, pieces, []))
    validation = {"sha256": "file", "canonical_stream_sha256": "stream", "canonical_bytes": len(text)}
    plan = {"splits": {"validation": validation}}
    row = {"canonical_sha256": "row", "text": text}
    report = {"validation": {"ids": 3}, "fit": {"occupied_vocab": 48000}}
    monkeypatch.setattr(frontend_analysis, "checked_run", lambda *a, **kw: (
        plan, "manifest", {"validation": [tmp_path / "validation.jsonl"]}, None, None, None))
    monkeypatch.setattr(frontend_analysis, "load_fit", lambda *a: (model, False, "fit", report, {}))
    monkeypatch.setattr(frontend_analysis, "load_rows", lambda _: [row])
    monkeypatch.setattr(frontend_analysis, "checked_canonical", lambda record, _: record["text"])

    arms = {"F-space": tmp_path / "fit"}
    out = tmp_path / "comparison"
    first = frontend_analysis.analyze(tmp_path / "manifest.json", arms, out)
    assert first["arms"]["F-space"]["payload_bytes_distribution"] == {"1": 1, "2": 1, "10": 1}
    assert json.loads((out / "summary.json").read_bytes()) == first
    assert frontend_analysis.analyze(tmp_path / "manifest.json", arms, out) == first


def test_leading_space_checkpoint_resume_keeps_identical_event_artifact(tmp_path, monkeypatch):
    spec, _, _ = pretokenizer_spec("F-space")
    norm = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    algorithm = registry.make("algorithm", "boundary_guarded_hybrid")
    data = tmp_path / "train.jsonl"
    data.write_bytes(canonical({"text": "മലയാളം മലയാളത്തിൽ മലയാളങ്ങൾ മലയാളിയുടെ ഭാഷയിൽ. " * 25}) + b"\n")

    def context(out):
        return TrainContext(330, DEFAULT_SPECIALS, norm, registry.make("pretokenizer", spec),
                            out, None, [str(data)], {"merge_checkpoint_every": 1})

    completed, recovered = tmp_path / "complete", tmp_path / "resume"
    original = BPEProgress.save
    algorithm.train(context(completed))

    def interrupted(self, events, seqs, flags):
        original(self, events, seqs, flags)
        raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(BPEProgress, "save", interrupted)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            algorithm.train(context(recovered))
    _, info = algorithm.train(context(recovered))
    assert info["resumed_events"] > 0
    assert (completed / ARTIFACT).read_bytes() == (recovered / ARTIFACT).read_bytes()
