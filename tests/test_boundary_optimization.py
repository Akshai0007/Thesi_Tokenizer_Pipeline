import json

import pytest
import torch

from experiments.mom_bpe import optimization_proxy
from experiments.mom_bpe.orthographic_errors import audit, exact_segments, primary_pattern
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import BPEProgress
from mltok.lib.mom_backend import SurfaceEncoding
from mltok.lib.orthographic_guard import guarded_examples
from mltok.lib.picky_backend import ARTIFACT, EventBackend
from mltok.lib.picky_training import fit


def test_repair_rewards_real_internal_positions_and_never_crosses_unknown_scalars():
    word = "ക്കാകാൽ"
    alphabet = set(word) | {"കാ"}
    rows, lost, count = guarded_examples([(word, 5)], alphabet, ["കാ"])
    assert rows == [(["ക", "്", "ക", "ാ", "കാ", "ൽ"], 5, {1: -1000, 2: -1000, 3: -1000})]
    assert lost == 0 and count == 15
    split, lost, weight = guarded_examples([(word, 5)], alphabet - {"്"}, ["കാ"])
    assert split[0] == (["ക"], 5, {})
    assert lost == 15 and weight == 5
    samples = [(["a", "b"], 8, {1: -1000}), (["a", "c"], 10, {})]
    assert fit(list("abc"), samples, 4, penalty=1000)[0].events == [("merge", "a", "b")]
    assert fit(list("abc"), samples, 4, penalty=0)[0].events == [("merge", "a", "c")]


def setup(tmp_path):
    registry.load_plugins()
    src = tmp_path / "train.jsonl"
    src.write_text(json.dumps({"text": "കിളികൾ കൂട്ടത്തിൽ കടലിൽ പൂക്കൾ കളിക്കുന്നു. " * 20},
                              ensure_ascii=False) + "\n", encoding="utf-8")
    return src


def context(src, output):
    norm = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    return TrainContext(340, DEFAULT_SPECIALS, norm, registry.make("pretokenizer", "script_runs"),
                        output, None, [str(src)], {"merge_checkpoint_every": 1})


def test_guard_off_replays_same_events_and_ids_as_t5(tmp_path):
    src = setup(tmp_path)
    t5 = registry.make("algorithm", {"type": "picky_bpe", "picky_threshold_milli": 900}).train(
        context(src, tmp_path / "t5"))[0]
    off = registry.make("algorithm", {"type": "boundary_guarded_hybrid", "guard_milli": 0}).train(
        context(src, tmp_path / "off"))[0]
    assert off.artifact["initial"] == t5.artifact["initial"]
    assert off.artifact["events"] == t5.artifact["events"]
    texts = ["", "കൂട്ടത്തിൽ", "ചില്ലുകൾ", "ക്‌ഷ", "അന്യമായ Malayalam USB‑C 🙂", "<pad>"]
    assert off.encode_batch(texts) == t5.encode_batch(texts)


def test_guarded_checkpoint_generation_is_exact_after_restart(tmp_path, monkeypatch):
    src = setup(tmp_path)
    algo = registry.make("algorithm", "boundary_guarded_hybrid")
    complete = tmp_path / "complete"
    backend, _ = algo.train(context(src, complete))
    recovered = tmp_path / "recovered"
    original = BPEProgress.save

    def crash(self, events, seqs, flags):
        original(self, events, seqs, flags)
        if len(events) >= 2:
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(BPEProgress, "save", crash)
    with pytest.raises(RuntimeError, match="simulated"):
        algo.train(context(src, recovered))
    monkeypatch.setattr(BPEProgress, "save", original)
    _, info = algo.train(context(src, recovered))
    assert info["resumed_events"] >= 2
    assert (complete / ARTIFACT).read_bytes() == (recovered / ARTIFACT).read_bytes()
    reread = EventBackend.load(recovered)
    stress = ["കീ", "ൻ്റ", "ക്കൂ", "ൎക്കാ", "ക്‌ഷ", "ക്‍ഷ", "ന്‍", "ൔ", "👩🏽‍🔬", "\u0d00"]
    assert reread.encode_batch(stress) == backend.encode_batch(stress)
    assert [reread.decode(x) for x in reread.encode_batch(stress)] == list(map(reread.expected, stress))


def test_byte_backed_akshara_cut_is_measured_and_invalid_offsets_fail_closed(tmp_path):
    src = setup(tmp_path)
    model = registry.make("algorithm", {"type": "picky_bpe", "segmenter": "akshara"}).train(
        context(src, tmp_path / "akshara"))[0]
    text = "ഗൈ"
    enc = model.encode_offsets(text)
    assert len(enc.ids) == 6 and model.fallback_ids.issuperset(enc.ids)
    result = audit(model, [{"id": "test", "text": text}])
    assert result["totals"].get("legacy_akshara_breaks", 0) == 0
    assert result["totals"]["byte_adjacent_akshara_breaks"] == 1
    assert result["totals"]["unicode_scalar_continuation_transitions"] == 4
    example = result["examples"]["byte_adjacent/dependent_vowel"][0]
    assert (example["canonical_codepoint_offset"], example["canonical_utf8_offset"]) == (1, 3)
    with pytest.raises(ValueError, match="reproduce"):
        exact_segments(model, SurfaceEncoding(text, enc.ids[::-1], enc.tokens, enc.offsets))


@pytest.mark.parametrize("left,right,label", [("ക", "്", "virama_or_conjunct"), ("്", "ഷ", "virama_or_conjunct"),
                                           ("ക", "ാ", "dependent_vowel"), ("ൻ", "ക", "chillu"),
                                           ("്", "\u200d", "joiner"), ("ൽ", "\u200c", "joiner"),
                                           ("ക", "\u0d02", "combining_mark")])
def test_classify_orthography(left, right, label):
    assert primary_pattern(left, right) == label


def new_proxy():
    registry.load_plugins()
    return optimization_proxy.RightPaddedGPT(registry.make("proxy_model", optimization_proxy.MODEL).build(20, 16), 0)


def test_document_windows_and_right_padding_mask():
    class Fixture:
        special_ids = {"<pad>": 0, "<s>": 1, "</s>": 2}

        def encode_offsets(self, text):
            return SurfaceEncoding(text, list(map(int, text)), [], [])

        def decode(self, ids):
            return "".join(map(str, ids))

    assert list(optimization_proxy.windows(Fixture(), [{"text": "345"}, {"text": "6"}], 2)) == [
        ([1], 3), ([1, 3], 4), ([3, 4], 5), ([4, 5], 2), ([1], 6), ([1, 6], 2),
    ]
    model = new_proxy().eval()
    shorter = model(torch.tensor([[1, 4]]), torch.tensor([[1, 1]]))
    padded = model(torch.tensor([[1, 4, 0, 0]]), torch.tensor([[1, 1, 0, 0]]))
    torch.testing.assert_close(shorter, padded[:, :2], rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="right-padded"):
        model(torch.tensor([[1, 0, 4]]), torch.tensor([[1, 0, 1]]))
    assert optimization_proxy.capped_text("ക👩🏽‍🔬ക", 8) == "ക"


def test_resumed_cpu_proxy_restores_optimizer_and_rng(tmp_path, monkeypatch):
    schedule = [[([1], 3), ([1, 3], 2)]] * 9
    full = tmp_path / "full"
    alternate = tmp_path / "alternate"
    full.mkdir()
    alternate.mkdir()
    identity = {"model": "fixture"}
    pristine, _, expected_compute = optimization_proxy.fit(full, identity, 9, new_proxy, schedule)
    original = optimization_proxy.save_step

    def crash(*args):
        original(*args)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(optimization_proxy, "save_step", crash)
    with pytest.raises(RuntimeError, match="interrupted"):
        optimization_proxy.fit(alternate, identity, 9, new_proxy, schedule)
    monkeypatch.setattr(optimization_proxy, "save_step", original)
    clone, resumed, actual_compute = optimization_proxy.fit(alternate, identity, 9, new_proxy, schedule)
    assert resumed == 8 and expected_compute == actual_compute
    assert all(torch.equal(v, clone.state_dict()[k]) for k, v in pristine.state_dict().items())
