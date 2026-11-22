"""Fault injection and exact CPU continuation for proxy checkpoints."""
import json
import random
import struct
import zipfile
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from torch import nn

from conftest import base_cfg
from mltok import proxy_train
from mltok.runner import Run


class Backend:
    name, vocab_size, eos_id = "fixture", 8, 0

    def encode_batch(self, texts):
        return [[int(ch) for ch in text] for text in texts]


class StochasticModel:
    name = "fixture"

    def __init__(self):
        self.batches = []

    def build(self, vocab_size, seq_len):
        batches = self.batches

        class Net(nn.Sequential):
            def forward(self, ids):
                hidden = super().forward(ids)
                if self.training:
                    batches.append(ids.tolist())
                    hidden = hidden * (1 + random.random() / 10 + np.random.random() / 10)
                return hidden

        return Net(nn.Embedding(vocab_size, 8), nn.Dropout(0.25), nn.Linear(8, vocab_size))

    def n_embedding_params(self, model):
        return model[0].weight.numel()


TRAIN = ["1234", "2345", "3456", "4567", "7654", "6543", "5432", "4321"]
EVAL = TRAIN[:3]
OPTIONS = dict(seq_len=4, batch=2, epochs=1, seed=42, device="cpu", ckpt_key="test", log=lambda *_: None)


def fit(path, model=None, **overrides):
    return proxy_train.train_proxy(Backend(), TRAIN, EVAL, model or StochasticModel(),
                                   ckpt_path=path, **{**OPTIONS, "ckpt_every": 2, **overrides})


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_exact(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_exact(left, right)
    else:
        assert a == b


def test_exact_cpu_resume_after_successful_checkpoint(tmp_path):
    uninterrupted, interrupted, resumed = (StochasticModel() for _ in range(3))
    clean, split = tmp_path / "clean.ckpt", tmp_path / "split.ckpt"
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        expected = fit(clean, uninterrupted)

        def interrupt(path):
            assert load(path)["step"] == 2
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            fit(split, interrupted, on_ckpt=interrupt)
        actual = fit(split, resumed)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    assert uninterrupted.batches == interrupted.batches + resumed.batches
    assert_exact(load(clean), load(split))
    for field in ("steps", "bpb", "eval_nll_per_token", "train_tokens", "train_bytes"):
        assert actual[field] == expected[field]


def test_interrupt_before_first_checkpoint_restarts_cleanly(tmp_path):
    path = tmp_path / "proxy.ckpt"

    def interrupt(message):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fit(path, log=interrupt)
    assert not list(tmp_path.iterdir())
    assert fit(path)["steps"] == load(path)["step"]


def test_failed_checkpoint_write_preserves_previous_valid_progress(tmp_path, monkeypatch):
    path = tmp_path / "proxy.ckpt"
    save = torch.save
    committed = []

    def fail_second(state, stream):
        if state["step"] == 4:
            stream.write(b"partial archive")
            raise OSError("write interrupted")
        save(state, stream)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail_second)
        with pytest.raises(OSError, match="write interrupted"):
            fit(path, on_ckpt=lambda p: committed.append(p.read_bytes()))
    assert path.read_bytes() == committed[0]
    assert not list(tmp_path.glob("*.tmp"))
    assert fit(path)["steps"] == load(path)["step"]


@pytest.mark.parametrize("damage", ["truncated", "bit_flip", "invalid_rng"])
def test_falls_back_when_newest_checkpoint_cannot_be_restored(tmp_path, damage):
    path = tmp_path / "proxy.ckpt"
    expected = fit(path)
    backup = path.with_suffix(".previous.ckpt")
    original = backup.read_bytes()
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:20])
    elif damage == "bit_flip":
        with zipfile.ZipFile(path) as archive:
            member = next(item for item in archive.infolist() if item.filename.endswith("/data/0"))
        data = bytearray(path.read_bytes())
        lengths = struct.unpack_from("<HH", data, member.header_offset + 26)
        data[member.header_offset + 30 + sum(lengths)] ^= 1
        path.write_bytes(data)
    else:
        state = load(path)
        state["torch_rng"] = torch.empty(0, dtype=torch.uint8)
        torch.save(state, path)
    logs = []
    actual = fit(path, log=logs.append)
    assert any("resuming from checkpoint at step 4/5" in line for line in logs)
    assert backup.read_bytes() == original
    assert actual["bpb"] == expected["bpb"]


def test_interruption_during_restore_preserves_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "proxy.ckpt"
    expected = fit(path)
    original = path.read_bytes()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", interrupt)
        with pytest.raises(KeyboardInterrupt):
            fit(path)
    model = StochasticModel()
    assert fit(path, model)["bpb"] == expected["bpb"]
    assert path.read_bytes() == original and not model.batches


def test_interrupted_rotation_preserves_both_generations(tmp_path, monkeypatch):
    path = tmp_path / "proxy.ckpt"
    backup = path.with_suffix(".previous.ckpt")

    def stop_at_step_four(p):
        if load(p)["step"] == 4:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fit(path, on_ckpt=stop_at_step_four)
    before = path.read_bytes(), backup.read_bytes()

    def interrupt(source, target):
        target.write(source.read(8))
        raise OSError("rotation interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(proxy_train.shutil, "copyfileobj", interrupt)
        with pytest.raises(OSError, match="rotation interrupted"):
            fit(path)
    assert (path.read_bytes(), backup.read_bytes()) == before
    assert fit(path)["steps"] == load(path)["step"]


def test_newer_backup_can_resume_without_training(tmp_path):
    path = tmp_path / "proxy.ckpt"
    backup = path.with_suffix(".previous.ckpt")
    expected = fit(path)
    older, newer = backup.read_bytes(), path.read_bytes()
    path.write_bytes(older)
    backup.write_bytes(newer)
    model = StochasticModel()
    assert fit(path, model)["bpb"] == expected["bpb"]
    assert not model.batches


def test_legacy_checkpoint_can_resume(tmp_path):
    path = tmp_path / "proxy.ckpt"

    def interrupt(p):
        state = load(p)
        legacy = {key: state[key] for key in ("key", "steps", "step", "model", "opt", "scaler", "rng")}
        torch.save(legacy, p)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fit(path, on_ckpt=interrupt)
    logs = []
    assert fit(path, log=logs.append)["steps"] == 5
    assert any("legacy checkpoint lacks full RNG" in line for line in logs)
    assert any("resuming from checkpoint at step 2/5" in line for line in logs)


def test_incompatible_run_preserves_both_generations(tmp_path):
    path = tmp_path / "proxy.ckpt"
    fit(path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    model = StochasticModel()
    with pytest.raises(RuntimeError, match="No usable proxy checkpoint"):
        fit(path, model, micro_batch=1)
    assert not model.batches
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.fixture
def proxy_config(tmp_path):
    cfg = base_cfg(tmp_path)
    for split, texts in (("train", TRAIN), ("eval", EVAL)):
        shard = tmp_path / "clean" / split / "sample.jsonl"
        shard.parent.mkdir(parents=True)
        shard.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    cfg["proxy"].update(seq_len=4, batch=2, checkpoint_every_steps=0)
    return cfg


def test_result_publication_resumes_and_is_idempotent(proxy_config, monkeypatch):
    from mltok import runner

    run = Run(proxy_config)
    result = run.proxy_path(run.base, 0)
    result.parent.mkdir(parents=True)
    model = StochasticModel()
    monkeypatch.setattr(run, "ensure_tokenizer", lambda v: (Backend(), {}))
    make = runner.make
    monkeypatch.setattr(runner, "make", lambda kind, spec: model if kind == "proxy_model" else make(kind, spec))
    train = proxy_train.train_proxy
    monkeypatch.setattr(proxy_train, "train_proxy", lambda *args, **kwargs: train(*args, device="cpu", **kwargs))

    @contextmanager
    def interrupt_publication(path):
        with proxy_train.atomic_writer(path) as stream:
            yield stream
            raise OSError("publication interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "atomic_writer", interrupt_publication)
        with pytest.raises(OSError, match="publication interrupted"):
            run.ensure_proxy(run.base, 0)
    checkpoint = result.with_suffix(".ckpt")
    original = checkpoint.read_bytes()
    assert not result.exists() and load(checkpoint)["step"] == load(checkpoint)["steps"]
    model.batches.clear()
    result.write_bytes(b'{"partial":')  # Simulate an incomplete result from the old writer.

    def interrupt_sync(path):
        assert run._proxy_done(run.base, 0)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(run, "_queue_proxy_result", interrupt_sync)
        with pytest.raises(KeyboardInterrupt):
            run.ensure_proxy(run.base, 0)
    published = result.read_bytes()
    assert run.ensure_proxy(run.base, 0) == json.loads(published)
    assert checkpoint.read_bytes() == original and result.read_bytes() == published
    assert not model.batches


@pytest.mark.parametrize("change", ["micro_batch", "eval_docs", "same_size_corpus"])
def test_incompatible_results_have_distinct_paths(proxy_config, change):
    original = Run(proxy_config)
    path = original.proxy_path(original.base, 0)
    path.parent.mkdir(parents=True)
    result = dict(_key=original.proxy_key(original.base, 0), seed=0, bpb=1, steps=5,
                  n_params=10, n_nonembed=5, train_tokens=40, seconds=1)
    path.write_text(json.dumps(result), encoding="utf-8")
    assert original._proxy_done(original.base, 0)
    if change == "micro_batch":
        proxy_config["proxy"]["micro_batch"] = 1
    elif change == "eval_docs":
        proxy_config["search"]["eval_docs"] = 1
    else:
        shard = original.eval_shards[0]
        with open(shard, "r+b") as stream:
            content = stream.read()
            stream.seek(0)
            stream.write(content.replace(b"1234", b"4321", 1))
    changed = Run(proxy_config)
    assert changed.proxy_path(changed.base, 0) != path
    assert not changed._proxy_done(changed.base, 0)
    assert json.loads(path.read_text(encoding="utf-8")) == result


def test_interrupt_during_evaluation_resumes_from_final_step(tmp_path):
    class InterruptedEvaluation(StochasticModel):
        def build(self, *args):
            model = super().build(*args)

            def interrupt(module, inputs):
                if not module.training:
                    raise KeyboardInterrupt

            model.register_forward_pre_hook(interrupt)
            return model

    path = tmp_path / "proxy.ckpt"
    with pytest.raises(KeyboardInterrupt):
        fit(path, InterruptedEvaluation())
    assert load(path)["step"] == load(path)["steps"]
    model = StochasticModel()
    assert fit(path, model)["bpb"] == fit(tmp_path / "reference.ckpt")["bpb"]
    assert not model.batches
