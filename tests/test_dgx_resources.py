import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.mom_bpe import dgx_handoff, dgx_resources as gpu, document_proxy as trainer


UUID = "GPU-deadbeef-1234-1234-1234-123456789abc"
OTHER = "GPU-feedbeef-1234-1234-1234-123456789abc"


@pytest.fixture
def machine(monkeypatch):
    state = SimpleNamespace(index=1, free=8192, util=5, processes="", queries=[])

    def query(command, **kwargs):
        assert command[0] == "nvidia-smi" and kwargs["timeout"] == 10
        field = command[1]
        state.queries.append(field)
        if field == "--query-gpu=index,uuid,memory.free,utilization.gpu":
            return f"{state.index}, {UUID}, {state.free}, {state.util}\n"
        assert field == "--query-compute-apps=gpu_uuid,pid"
        if isinstance(state.processes, Exception):
            raise state.processes
        return state.processes

    monkeypatch.setattr(gpu.subprocess, "check_output", query)
    return state


def test_default_admits_only_through_five_percent(machine):
    policy = gpu.GpuAdmission()
    assert policy.admit(UUID)["utilization_pct"] == 5
    machine.util = 6
    with pytest.raises(ValueError, match="exceeds 5%"):
        policy.admit(UUID)


def test_shared_requires_permission_and_caps_utilization_at_fifteen(machine):
    with pytest.raises(ValueError, match="requires --allow-shared-gpu"):
        gpu.GpuAdmission(max_gpu_util_pct=10)
    with pytest.raises(ValueError, match="between 0 and 15%"):
        gpu.GpuAdmission(max_gpu_util_pct=16, allow_shared_gpu=True)
    machine.free, machine.util = 5120, 15
    policy = gpu.GpuAdmission(5120, 15, True)
    assert policy.admit(UUID)["utilization_pct"] == 15
    machine.util = 16
    with pytest.raises(ValueError, match="exceeds 15%"):
        policy.admit(UUID)


def test_memory_limit_still_applies_on_shared_gpu(machine):
    machine.free = 5119
    with pytest.raises(ValueError, match="insufficient free memory"):
        gpu.GpuAdmission(5120, 10, True).admit(UUID)


@pytest.mark.parametrize("invalid", ["", "1", "GPU-dead-beef", "MIG-" + UUID, UUID + "," + OTHER])
def test_only_full_physical_uuid_is_accepted(machine, invalid):
    with pytest.raises(ValueError, match="full physical GPU UUID"):
        gpu.GpuAdmission().admit(invalid)
    assert not machine.queries


def test_process_detection_distinguishes_self_other_device_and_other_job(machine):
    other_pid = os.getpid() + 1
    machine.processes = f"{UUID}, {os.getpid()}\n{OTHER}, {other_pid}\n"
    assert gpu.GpuAdmission().admit(UUID)["other_compute_processes"] == 0
    machine.processes += f"{UUID}, {other_pid}\n{UUID}, {other_pid}\n"
    with pytest.raises(ValueError, match="Other GPU compute processes"):
        gpu.GpuAdmission().admit(UUID)
    assert gpu.GpuAdmission(8192, 5, True).admit(UUID)["other_compute_processes"] == 1


@pytest.mark.parametrize("problem", ["[Not Supported]\n", f"{UUID}, [Insufficient Permissions]\n", '"invalid csv\n',
                                      subprocess.TimeoutExpired("nvidia-smi", 10),
                                      subprocess.CalledProcessError(1, "nvidia-smi")])
def test_shared_gpu_requires_verified_process_information(machine, problem):
    machine.processes = problem
    with pytest.raises(ValueError, match="Cannot verify GPU compute processes"):
        gpu.GpuAdmission(5120, 10, True).admit(UUID)


def test_handoff_preflight_uses_same_policy(machine, tmp_path, monkeypatch):
    monkeypatch.setattr(dgx_handoff, "imports", lambda _: (trainer, None))
    monkeypatch.delenv("MLTOK_PLUGINS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(trainer.torch.version, "cuda", "12.1")
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(trainer.torch.cuda, "is_bf16_supported", lambda including_emulation: not including_emulation)
    monkeypatch.setattr(trainer.torch.cuda, "get_device_name", lambda _: "NVIDIA A100")
    monkeypatch.setattr(trainer.torch.cuda, "get_device_capability", lambda _: (8, 0))
    monkeypatch.setattr(trainer.torch.backends.cudnn, "version", lambda: 90100)
    machine.free, machine.util = 5120, 10
    machine.processes = f"{UUID}, {os.getpid() + 1}\n"
    manifest = {"storage": {"additional_free_scratch_bytes": 10}}
    scratch = tmp_path / "scratch"
    with pytest.raises(ValueError, match="insufficient free memory"):
        dgx_handoff.preflight(tmp_path / "bundle", scratch, manifest)
    options = dict(min_free_mib=5120, max_gpu_util_pct=10, allow_shared_gpu=True)
    result = dgx_handoff.preflight(tmp_path / "bundle", scratch, manifest, **options)
    assert result["gpu_admission"] == options
    assert result["gpu"] == trainer.gpu_preflight(5120, max_gpu_util_pct=10, allow_shared_gpu=True)
    assert set(result["gpu"]) == {"physical_index", "uuid", "type", "capability", "cuda", "cudnn"}


def test_trainer_rechecks_admission_after_preflight(machine, tmp_path, monkeypatch):
    args = SimpleNamespace(command="run", cache=tmp_path, mode="fixed_source_bytes", budget=None,
                           out=tmp_path / "run", seed=None, update_targets=None, microbatch_blocks=None,
                           checkpoint_updates=None, threads=1, device="cuda", min_free_mib=5120,
                           max_gpu_util_pct=10, allow_shared_gpu=True)
    cache = {"identity": {"schema": 1, "context": 512, "policy": trainer.POLICY, "arm": "M48", "vocab": trainer.VOCAB,
                          "specials": dict(zip(trainer.DEFAULT_SPECIALS, range(len(trainer.DEFAULT_SPECIALS)))),
                          "manifest_sha256": "manifest", "tokenizer_sha256": "tokenizer",
                          "streams": {"validation": "held"}},
             "splits": {"train": "train", "validation": "validation"}}
    forecast = {"predicted_tokens": 20, "predicted_eos": 1, "canonical_bytes": 80}
    monkeypatch.setattr(trainer, "receipt", lambda _: (cache, "cache"))
    monkeypatch.setattr(trainer, "EncodedSplit", lambda *_: SimpleNamespace(docs=[{}], canonical_bytes=80))
    monkeypatch.setattr(trainer, "selection", lambda *_: ([], forecast))
    monkeypatch.setattr(trainer, "schedule", lambda *_: ([object()], [{}, forecast]))
    monkeypatch.setattr(trainer.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**35))
    monkeypatch.setattr(trainer.torch, "manual_seed", lambda _: None)
    monkeypatch.setattr(trainer, "train", lambda *_a, **_kw: pytest.fail("started training after admission changed"))
    machine.free, machine.util = 5120, 10
    machine.processes = f"{UUID}, {os.getpid() + 1}\n"

    def initial(minimum, *, max_gpu_util_pct, allow_shared_gpu):
        assert (minimum, max_gpu_util_pct, allow_shared_gpu) == (5120, 10, True)
        assert gpu.GpuAdmission(minimum, max_gpu_util_pct, allow_shared_gpu).admit(UUID)["utilization_pct"] == 10
        machine.util = 11
        return {"physical_index": 1, "uuid": UUID}

    monkeypatch.setattr(trainer, "gpu_preflight", initial)
    with pytest.raises(ValueError, match="exceeds 10%"):
        trainer.fit_cache(args)
    assert machine.queries[-1] == "--query-gpu=index,uuid,memory.free,utilization.gpu"
    assert not (args.out / "identity.json").exists()
    assert not (args.out / "resource-telemetry").exists()


@pytest.mark.parametrize(("module", "base"), [(trainer, ["run", "--cache", ".", "--mode", "fixed_source_bytes", "--out", "."]),
                                               (dgx_handoff, ["preflight", "bundle", "--scratch", "scratch"])])
def test_cli_propagates_shared_policy(monkeypatch, module, base):
    monkeypatch.setattr(sys, "argv", ["cli", *base, "--min-free-mib", "5120", "--max-gpu-util-pct", "10", "--allow-shared-gpu"])
    if module is trainer:
        def run(args):
            assert (args.min_free_mib, args.max_gpu_util_pct, args.allow_shared_gpu) == (5120, 10, True)
            return {"evaluation": {}}
        monkeypatch.setattr(module, "fit_cache", run)
    else:
        def run(_root, _scratch, _manifest, **opts):
            assert opts == dict(min_free_mib=5120, max_gpu_util_pct=10, allow_shared_gpu=True)
            return {}
        monkeypatch.setattr(module, "verify_bundle", lambda _: {})
        monkeypatch.setattr(module, "preflight", run)
    module.main()


def test_only_exact_operational_revision_preserves_legacy_driver_identity():
    actual = trainer.code_digest(trainer.__file__)
    assert actual == gpu.RESOURCE_DRIVER and actual != gpu.PINNED_DRIVER
    assert gpu.scientific_driver_digest(actual) == gpu.PINNED_DRIVER
    other = "0" * 64
    assert gpu.scientific_driver_digest(other) == other
