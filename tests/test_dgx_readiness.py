import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from experiments.mom_bpe import dgx_readiness as checks
from experiments.mom_bpe import document_proxy as driver
from experiments.mom_bpe.proxy_blocks import publish


NB = Path(__file__).resolve().parents[1] / "notebooks/mom_bpe_dgx_readiness.ipynb"


def test_notebook_cells_are_independent_and_gated(tmp_path, monkeypatch):
    notebook = json.loads(NB.read_bytes())
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    actions = ["environment", "source", "identity", "inventory", "hashes", "devices", "admission",
               "telemetry", "report", "storage"]
    assert [c["id"] for c in cells] == actions
    pin = "a" * 40
    source = tmp_path / ("resource-source-" + pin)
    helper = source / "experiments/mom_bpe/dgx_readiness.py"
    helper.parent.mkdir(parents=True)
    helper.touch()
    monkeypatch.setenv("MOM_BPE_RESOURCE_REVISION", pin)
    monkeypatch.setenv("MOM_BPE_READINESS_SESSION", "mock")
    import subprocess
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **_: pin if "rev-parse" in cmd else "")
    invoke = Mock()
    monkeypatch.setattr(subprocess, "run", invoke)
    for cell in cells:
        assert cell["execution_count"] is None and cell["outputs"] == []
        body = "".join(cell["source"])
        compile(body, cell["id"], "exec")
        # Substitute the host-specific private directory and ownership check on Windows.
        body = body.replace('Path("/jupyter/socse.user63/mom-bpe-private-recovery")', repr_path(tmp_path))
        body = body.replace("root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077", "False")
        exec(body, {})
        cmd = invoke.call_args.args[0]
        assert invoke.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        if cell["id"] not in ("source", "environment"):
            assert cmd[-6] == cell["id"]
            assert cmd[-3:] == ["", "False", "False"]
            compile(cmd[4], "worker", "exec")


def repr_path(path):
    return f"Path({str(path)!r})"


def test_inventory_and_independent_final_hash(tmp_path, monkeypatch):
    folder = tmp_path / "runs/dp/seed17/M48/fixed_source_bytes"
    folder.mkdir(parents=True)
    blob = b"test-generation"
    from hashlib import sha256
    digest = sha256(blob).hexdigest()
    path = folder / f"step-000002-{digest}.pt"
    path.write_bytes(blob)
    identity = {"forecast": {"optimizer_updates": 2}, "config": {"checkpoint_updates": 1}}
    (folder / "identity.json").write_text(json.dumps(identity))
    publish(folder / "result.json", {"identity": identity, "evaluation": {"bpb": 1.0},
                                     "checkpoint": {"path": path.name, "size": len(blob), "sha256": digest}})
    pins = tuple(checks.file_digest(folder / name)[1] for name in ("identity.json", "result.json"))
    monkeypatch.setattr(checks, "PINS", {("M48", "fixed_source_bytes"): pins})
    assert checks.seed17(tmp_path, "hashes")["M48/fixed_source_bytes"]["missing_steps"] == [1]
    path.write_bytes(b"x" * len(blob))
    with pytest.raises(ValueError, match="checksum"):
        checks.seed17(tmp_path, "hashes")
    (folder / "identity.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        checks.seed17(tmp_path, "identity")


@pytest.mark.parametrize("shared,minimum,maximum", [(False, 8192, 5), (True, 5120, 10)])
def test_diagnostic_preflight_and_telemetry(tmp_path, monkeypatch, shared, minimum, maximum):
    uuid = "GPU-00000000-1234-1234-1234-123456789abc"
    state = {"uuid": uuid, "physical_index": 1, "free_mib": minimum, "utilization_pct": maximum}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(checks.gpu, "observe_gpu", lambda _: state)
    monkeypatch.setattr(checks.gpu, "_other_compute_processes", lambda _: {991} if shared else set())
    def driver_check(mib, *, max_gpu_util_pct, allow_shared_gpu):
        assert (mib, max_gpu_util_pct, allow_shared_gpu) == (minimum, maximum, shared)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == uuid
        return {**state, "type": "NVIDIA A100"}
    monkeypatch.setattr(driver, "gpu_preflight", driver_check)
    monkeypatch.setattr(driver.torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(driver.torch.cuda, "max_memory_allocated", lambda _: 0)
    monkeypatch.setattr(driver.torch.cuda, "max_memory_reserved", lambda _: 0)
    session = tmp_path / "session"
    monkeypatch.setattr(checks, "seed17", lambda *_: {"fixture": True})
    monkeypatch.setattr(checks, "inspect_devices", lambda: {"fixture": True})
    for action in ("identity", "inventory", "hashes", "devices"):
        checks.run(action, tmp_path, session)
    checks.run("admission", tmp_path, session, uuid, shared, True)
    checks.run("telemetry", tmp_path, session)
    checks.run("report", tmp_path, session)
    assert checks.latest(session, "report")[1]["admission"]["policy"]["allow_shared_gpu"] is shared
    checks.run("inventory", tmp_path, session)
    with pytest.raises(ValueError, match="Rerun hashes"):
        checks.run("report", tmp_path, session)
    with pytest.raises(ValueError, match="allocation"):
        checks.run("admission", tmp_path, session, uuid, shared, False)
    with pytest.raises(ValueError, match="incomplete"):
        checks.telemetry(session)


def test_report_requires_all_current_session_checks(tmp_path):
    with pytest.raises(ValueError, match="identity cell"):
        checks.run("report", tmp_path, tmp_path / "session")
