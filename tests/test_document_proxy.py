import hashlib
import json
import math
import random

import numpy as np
import pytest
import torch

from experiments.mom_bpe import document_proxy as driver
from experiments.mom_bpe import dgx_resources as resources
from experiments.mom_bpe.optimization_proxy import RightPaddedGPT
from experiments.mom_bpe.proxy_blocks import EncodedSplit, blocks, selection, sequences, write_split
from mltok import registry
from mltok.lib.canonical_bpb import TorchWindowNLL, canonical_bpb
from mltok.lib.mom_backend import SurfaceEncoding


SPECIALS = {"<pad>": 0, "<s>": 1, "</s>": 2}
CPU = torch.device("cpu")


class Bytes:
    special_ids = SPECIALS
    fallback_ids = set(range(3, 259))
    vocab_size = 259

    def __init__(self):
        self.calls = 0

    def encode_offsets(self, text):
        self.calls += 1
        data = text.encode()
        return SurfaceEncoding(text, [byte + 3 for byte in data], [f"<0x{byte:02X}>" for byte in data], [])

    def decode(self, ids):
        return bytes(i - 3 for i in ids).decode()


def record(i, text):
    data = text.encode()
    digest = hashlib.sha256(data).hexdigest()
    return {"id": str(i), "text": text, "sha256": digest, "canonical_sha256": digest,
            "raw_bytes": len(data), "canonical_bytes": len(data)}


@pytest.fixture
def encoded(tmp_path):
    rows = [record(i, text) for i, text in enumerate(("aകz", "mnop", "w"))]
    backend = Bytes()
    digest = write_split(tmp_path, "train", iter(rows), backend, str, False, None)
    assert backend.calls == len(rows)
    return EncodedSplit(tmp_path, "train", digest), rows, backend


def tiny_model():
    return registry.make("proxy_model", {"type": "gpt", "n_layer": 1, "n_head": 2, "d_model": 16}).build(259, 4)


def test_target_ledgers_and_utf8_byte_prefix(encoded):
    data, rows, _ = encoded
    ids, source = selection(data, "fixed_source_bytes", 7)
    assert ids == [(0, 6)]       # stop at the first non-fitting document
    assert source["canonical_bytes"] == 5 and source["predicted_eos"] == 1
    chosen, partial = selection(data, "fixed_training_tokens", 3)
    assert chosen == [(0, 3)] and partial["predicted_eos"] == 0
    assert partial["partial"] == {"document_id": "0", "canonical_sha256": rows[0]["canonical_sha256"],
                                  "predicted_text_tokens": 3, "canonical_bytes": 3, "ends_on_utf8_scalar": False}
    _, before_eos = selection(data, "fixed_training_tokens", 5)
    assert before_eos["partial"]["canonical_bytes"] == 5 and before_eos["full_documents"] == 0
    _, after_eos = selection(data, "fixed_training_tokens", 6)
    assert after_eos["partial"] is None and after_eos["full_documents"] == 1
    _, second = selection(data, "fixed_training_tokens", 7)
    assert second["full_canonical_bytes"] == 5 and second["partial"]["canonical_bytes"] == 1
    with pytest.raises(ValueError, match="cannot meet"):
        selection(data, "fixed_training_tokens", 100)
    with pytest.raises(ValueError, match="cannot meet"):
        selection(data, "fixed_source_bytes", 100)


def test_blocks_no_duplicate_targets_or_cross_document_context(encoded):
    data, rows, backend = encoded
    chosen, exposure = selection(data, "fixed_source_bytes", data.canonical_bytes)
    all_blocks = list(blocks(data, chosen, 2))
    covered, emitted = 0, 0
    for index, row in enumerate(rows):
        runs = [item for item in all_blocks if item.doc == index]
        pairs = [sequences(data, item, 1, 2) for item in runs]
        original = backend.encode_offsets(row["text"]).ids
        assert [t for _, ys in pairs for t in ys] == [*original, 2]
        assert [t for xs, _ in pairs for t in xs] == [1, *original]
        assert all(0 < len(xs) == len(ys) <= 2 for xs, ys in pairs)
        covered += sum(item.source_bytes for item in runs)
        emitted += sum(item.targets for item in runs)
    assert covered == exposure["canonical_bytes"] and emitted == exposure["predicted_tokens"]
    _, p = driver.schedule(data, chosen, 4, 5, 2)
    assert p[-1]["predicted_tokens"] == emitted and p[-1]["right_padding_positions"] > 0
    assert p[-1]["predicted_eos"] == len(rows)


def test_bpb_matches_reference_on_equivalent_contexts_and_ignores_padding(encoded):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    data, rows, backend = encoded
    model = tiny_model().eval()
    fast = driver.score(model, data, 4, SPECIALS, 2, CPU)
    unpadded = driver.score(model, data, 4, SPECIALS, 1, CPU)
    ref = TorchWindowNLL(RightPaddedGPT(model, 0).eval(), context=4, pad_id=0, batch_size=4)

    def equivalent(inputs, targets):
        return [v for i in range(0, len(inputs), 4) for v in ref(inputs[i:i + 4], targets[i:i + 4])]

    slow = canonical_bpb(backend, rows, equivalent)
    assert fast["canonical_bytes"] == slow["canonical_bytes"] == 10
    assert fast["predicted_tokens"] == slow["predicted_tokens"] == 13
    assert fast["bpb"] == pytest.approx(slow["bpb"], abs=2e-6)
    assert fast["bpb"] == pytest.approx(unpadded["bpb"], abs=2e-6)
    assert fast["nll_nats"] / (math.log(2) * 10) == fast["bpb"]
    assert [r["id"] for r in fast["per_document"]] == [r["id"] for r in rows]


def test_checkpoint_replay_exact_and_corrupt_generation_retained(tmp_path, encoded, monkeypatch):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data, _, _ = encoded
    chosen, _ = selection(data, "fixed_source_bytes", 10)
    planned, progress = driver.schedule(data, chosen, 2, 3, 2)
    config = {"checkpoint_updates": 1, "optimizer": {"peak_lr": .003, "end_lr": .0003,
              "warmup_fraction": .05, "clip": 1.0}}

    def state():
        model = torch.nn.Sequential(tiny_model(), torch.nn.Dropout(.15))
        return model, torch.optim.AdamW(model.parameters(), lr=.003)

    def run(folder, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return driver.train(folder, {"experiment": "fixture"}, data, planned, progress, state, config,
                            "fixed_source_bytes", SPECIALS, CPU)

    done, resumed = tmp_path / "t", tmp_path / "r"
    done.mkdir()
    resumed.mkdir()
    full, nll, reference = run(done, 17)
    save = driver.save_generation

    def interrupt(out, checkpoint):
        entry = save(out, checkpoint)
        if checkpoint["step"] == 2:
            raise RuntimeError("interrupted")
        return entry

    with monkeypatch.context() as patch:
        patch.setattr(driver, "save_generation", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            run(resumed, 17)
    newest = next(resumed.glob("step-000002-*.pt"))
    newest.write_bytes(b"damaged")
    first = next(resumed.glob("step-000001-*.pt"))
    with first.open("rb"):
        recovered, recovered_nll, final = run(resumed, 99)
    assert newest.read_bytes() == b"damaged"
    assert nll == recovered_nll
    assert all(torch.equal(a, b) for a, b in zip(full.parameters(), recovered.parameters(), strict=True))

    def load(folder, entry):
        return torch.load(folder / entry["path"], weights_only=True)

    expected, actual = load(done, reference), load(resumed, final)
    assert expected["exposure"] == actual["exposure"] == progress[-1]
    assert expected["scheduler"] == actual["scheduler"]
    assert expected["rng"]["numpy"] == actual["rng"]["numpy"]
    assert expected["rng"]["python"] == actual["rng"]["python"]
    assert torch.equal(expected["rng"]["torch"], actual["rng"]["torch"])
    assert expected["optimizer"]["param_groups"] == actual["optimizer"]["param_groups"]
    for i, tensors in expected["optimizer"]["state"].items():
        assert all(torch.equal(a, actual["optimizer"]["state"][i][k]) for k, a in tensors.items())
    assert driver.score(full, data, 2, SPECIALS, 2, CPU) == driver.score(recovered, data, 2, SPECIALS, 2, CPU)
    with monkeypatch.context() as patch:
        patch.setattr(driver, "save_generation", lambda *_: pytest.fail("completed training restarted"))
        assert run(resumed, 123)[2] == final


def test_gradient_accumulation_weights_by_targets_with_uneven_microbatches(encoded):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    data, _, _ = encoded
    chosen, _ = selection(data, "fixed_source_bytes", 10)
    batch = list(blocks(data, chosen, 4))
    model = tiny_model()
    inputs, targets = driver.batch_tensors(data, batch, SPECIALS, CPU)
    total = (targets != driver.IGNORE).sum().item()
    driver.token_losses(model, inputs, targets, bf16=False, reduction="mean").backward()
    reference = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    for group in driver.grouped(batch, 2):
        x, y = driver.batch_tensors(data, group, SPECIALS, CPU)
        (driver.token_losses(model, x, y, bf16=False, reduction="sum") / total).backward()
    for expected, parameter in zip(reference, model.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, expected, atol=2e-7, rtol=2e-5)


def test_gpu_seven_rejected_before_torch_gpu_access(monkeypatch):
    uuid = "GPU-deadbeef-1234-1234-1234-123456789abc"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    monkeypatch.setattr(resources.subprocess, "check_output", lambda *a, **k: f"7, {uuid}, 51200, 0\n")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("touched CUDA"))
    with pytest.raises(ValueError, match="prohibited"):
        driver.gpu_preflight(8192)


def test_resource_telemetry_survives_interrupted_cpu_replay(tmp_path, encoded, monkeypatch):
    torch.set_num_threads(1)
    data, _, _ = encoded
    chosen, _ = selection(data, "fixed_source_bytes", 10)
    planned, progress = driver.schedule(data, chosen, 2, 3, 2)
    config = {"checkpoint_updates": 1, "optimizer": {"peak_lr": .003, "end_lr": .0003,
              "warmup_fraction": .05, "clip": 1.0}}
    identity = {"experiment": "fixture", "implementation": {"driver": resources.PINNED_DRIVER}}
    observed = {"uuid": "GPU-deadbeef-1234-1234-1234-123456789abc", "physical_index": 1,
                "free_mib": 9600, "utilization_pct": 4, "other_compute_processes": 0}
    monkeypatch.setattr(resources, "observe_gpu", lambda _: {**observed, "free_mib": 8600, "utilization_pct": 12})

    class MockCuda:
        def reset_peak_memory_stats(self, device):
            assert device == CPU

        def max_memory_allocated(self, device):
            return 4096

        def max_memory_reserved(self, device):
            return 8192

    def state():
        model = tiny_model()
        return model, torch.optim.AdamW(model.parameters(), lr=.003)

    def attempt():
        telemetry = resources.ResourceTelemetry(tmp_path, resources.GpuAdmission(), observed, MockCuda(), CPU, {})
        with telemetry:
            return driver.train(tmp_path, identity, data, planned, progress, state, config,
                                "fixed_source_bytes", SPECIALS, CPU, telemetry=telemetry)

    persist = resources.ResourceTelemetry.checkpoint

    def interrupt(telemetry, step, member):
        persist(telemetry, step, member)
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(resources.ResourceTelemetry, "checkpoint", interrupt)
        with pytest.raises(KeyboardInterrupt):
            attempt()

    folder, = (tmp_path / "resource-telemetry").iterdir()
    prior = {p: p.read_bytes() for p in folder.iterdir()}
    first, = tmp_path.glob("step-000001-*.pt")
    original_checkpoint = first.read_bytes()
    _, _, final = attempt()
    later, = (p for p in folder.parent.iterdir() if p != folder)
    assert all(p.read_bytes() == data for p, data in prior.items())
    assert first.read_bytes() == original_checkpoint
    assert torch.load(first, weights_only=True)["identity"] == identity

    def read(path):
        return json.loads(path.read_bytes())

    assert read(folder / "start.json")["initial"] == observed
    assert read(folder / "end.json")["status"] == "interrupted"
    assert read(later / "resume.json")["starting_update"] == 1
    assert read(later / "resume.json")["checkpoint"]["path"] == first.name
    assert not (later / "update-000001.json").exists()
    end = read(later / "end.json")
    assert end["status"] == "training_finished"
    assert end["gpu"]["free_mib"] == 8600 and end["gpu"]["utilization_pct"] == 12
    assert end["peak_allocated_bytes"] == 4096 and end["peak_reserved_bytes"] == 8192
    assert end["training_seconds"] >= read(later / f"update-{len(planned):06d}.json")["training_seconds"]
    assert read(later / f"update-{len(planned):06d}.json")["checkpoint"] == final
