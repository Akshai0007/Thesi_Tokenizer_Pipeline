import hashlib
import io
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mom_bpe import dgx_batch as b
from experiments.mom_bpe import dgx_resources as gpu
from mltok.lib.canonical_bpb import summarize_bpb


REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "experiments/mom_bpe"


def test_pinned_queue():
    manifest = b.load_pinned(HERE / "t5_t8_batch_v1.json", b.BATCH_SHA)
    assert [(j["arm"], j["seed"], j["mode"], j["budget"]) for j in manifest["queue"]] == [
        ("T5", 17, b.MODES[0], 50_000_000), ("T8", 17, b.MODES[0], 50_000_000),
        ("T5", 17, b.MODES[1], 5_000_000), ("T8", 17, b.MODES[1], 5_000_000),
    ]
    base = b.load_pinned(HERE / "t5_t8_inputs_v1.json", manifest["input_manifest_sha256"])
    for j in manifest["queue"][:2]:
        assert j["plan_sha256"] == base["arms"][j["arm"]]["plan_sha256"]


@pytest.fixture
def batch(tmp_path, monkeypatch):
    jobs, plans = [], {}
    for mode in b.MODES:
        for arm in ("T5", "T8"):
            job = {"id": f"{arm}-{mode}", "arm": arm, "mode": mode, "seed": 17, "budget": 3,
                   "forecast": {"optimizer_updates": 2}, "validation": {},
                   "output": f"runs/{arm}-{b.MODES.index(mode)}"}
            plan = {k: job[k] for k in ("arm", "mode", "budget", "forecast", "validation")}
            plan["config"] = {"seed": 17, "checkpoint_updates": 1}
            job["plan_sha256"] = b.metadata_digest(plan)
            jobs.append(job)
            plans[job["id"]] = plan
    batch = b.Batch(tmp_path / "root", {"queue": jobs}, {"parameters": 3}, manifest_sha="fixture")
    batch.stage("batch-manifest", {"manifest": batch.manifest})
    batch.stage("verified-inputs", {"plans": {j["arm"]: plans[j["id"]] for j in jobs[:2]}})
    batch.stage("verified-plans", {"plans": plans})
    batch.stage("cpu-ready", {"ready": True})
    for first, second in (("PENDING", "INPUT_VERIFIED"), ("INPUT_VERIFIED", "PLAN_VERIFIED"), ("PLAN_VERIFIED", "READY")):
        batch.advance_preparation(first, second)
    rows = [{"id": str(i), "canonical_sha256": f"{i:064x}", "canonical_bytes": i + 1, "length": 1}
            for i in range(186)]
    monkeypatch.setattr(batch, "held", lambda _: rows)
    monkeypatch.setattr(b.frozen, "verify_identity", lambda *args: None)
    monkeypatch.setattr(b.frozen, "audit", lambda *args, **kw: (None, {}))
    monkeypatch.setattr(batch, "storage", lambda *args: {"free_bytes": 100, "additional_bytes_reserved": 1})
    return batch


def finish(batch, job, *, checkpoint_identity=None):
    docs = []
    for d in batch.held(job["arm"]):
        docs.append({**d, **summarize_bpb([1.0], d["canonical_bytes"], 2)})
    metrics = summarize_bpb([1.] * len(docs), sum(d["canonical_bytes"] for d in docs), 2 * len(docs))
    identity = {"job": job["id"]}
    buffer = io.BytesIO()
    torch.save({"identity": checkpoint_identity or identity, "step": 2, "exposure": job["forecast"],
                "train_nll_nats": 1.0, "model": {"weight": torch.ones(1)},
                "optimizer": {"state": {0: {"step": torch.tensor(2)}}, "param_groups": [{"lr": 0.001}]},
                "scheduler": {"lr": 0.001}, "rng": dict.fromkeys(("python", "numpy", "torch", "cuda"))}, buffer)
    checkpoint = buffer.getvalue()
    digest = hashlib.sha256(checkpoint).hexdigest()
    name = f"step-000002-{digest}.pt"
    path = batch.root / job["output"]
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_bytes(checkpoint)
    (path / "identity.json").write_text(json.dumps(identity))
    b.write_receipt(path / "result.json", {"identity": identity, "training_nll_nats": 1.0,
                                           "evaluation": {"per_document": docs, **metrics},
                                           "checkpoint": {"path": name, "size": len(checkpoint), "sha256": digest}})
    return path / name


def approval(batch, scope="all"):
    return {"manifest_sha256": batch.sha, "scope": scope, "uuid": "fake", "shared": False}


def test_interrupted_execution_reconciles_missing_worker_and_skips_completed(batch, monkeypatch):
    first, second = batch.selection("first")
    calls = []

    def fit(job, *args):
        calls.append(job["id"])
        finish(batch, job)
        if len(calls) == 1:
            raise KeyboardInterrupt()

    monkeypatch.setattr(batch, "fit", fit)
    with pytest.raises(KeyboardInterrupt):
        batch.execute("first", approval(batch))
    assert batch.state(first) == "INTERRUPTED"
    assert batch.state(second) == "READY"
    batch.record(second, "RUNNING", attempt="killed")
    batch.execute("first", approval(batch))
    batch.execute("first", approval(batch))
    assert calls == [first["id"], second["id"]]
    assert [batch.event(second, p)["state"] for p in batch.events(second)][-5:] == [
        "RUNNING", "INTERRUPTED", "RUNNING", "COMPLETED", "RESULT_VERIFIED",
    ]
    assert len(list((batch.dir / "logs" / first["id"]).glob("*.log"))) == 1


def test_failed_admission_does_not_advance_queue(batch, monkeypatch):
    def reject(*args):
        raise ValueError("GPU resources changed")

    monkeypatch.setattr(batch, "fit", reject)
    with pytest.raises(ValueError, match="resources changed"):
        batch.execute("first", approval(batch))
    first, second = batch.selection("first")
    assert batch.state(first) == "FAILED"
    assert batch.state(second) == "READY"


def test_gpu_seven_is_rejected_before_cuda(batch, monkeypatch):
    uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    monkeypatch.setattr(gpu, "_query", lambda *args: [("7", uuid, "20000", "0")])
    with pytest.raises(ValueError, match="GPU 7"):
        batch.admit({**approval(batch), "uuid": uuid})


def test_second_pair_requires_matching_scope_and_verified_first(batch):
    with pytest.raises(ValueError, match="does not cover"):
        batch.execute("second", approval(batch, "first"))
    with pytest.raises(ValueError, match="both first-pair"):
        batch.execute("second", approval(batch))


def test_second_pair_requires_persisted_first_pair_check(batch, monkeypatch):
    for job in batch.selection("first"):
        finish(batch, job)
    with pytest.raises(ValueError, match="Missing first-pair-verified"):
        batch.execute("second", approval(batch))
    check = batch.result_check("first")
    assert len(check["checkpoints"]) == 2
    with pytest.raises(ValueError, match="All planned results"):
        batch.result_check("all")
    monkeypatch.setattr(batch, "fit", lambda job, *_: finish(batch, job))
    assert set(batch.execute("second", approval(batch)).values()) == {"RESULT_VERIFIED"}
    assert len(batch.result_check("all")["checkpoints"]) == 4


def test_full_unstarted_queue_reserves_15_gib(batch):
    assert b.Batch.storage(batch, batch.jobs)["additional_bytes_reserved"] == 15 * b.GIB


def test_session_report_counts_only_verified_completed_jobs(batch, monkeypatch):
    first, second = batch.selection("first")
    finish(batch, first)
    checkpoint = finish(batch, second)
    checkpoint.write_bytes(b"damaged")
    monkeypatch.setattr(batch, "active_user_gpu_processes", lambda: {"status": "fixture"})
    report = batch.report()
    assert (report["planned"], report["verified_completed"], report["pending_or_failed"]) == (4, 1, 3)
    assert list(report["results"]) == [first["id"]]
    assert second["id"] in report["verification_issues"]
    assert b.read_receipt(next((batch.dir / "session-reports").glob("*.json"))) == report


def test_failed_smoke_revokes_reuse_of_older_session_receipt(batch, monkeypatch):
    from experiments.mom_bpe import dgx_readiness

    monkeypatch.setattr(dgx_readiness, "preflight", lambda *args: {"fixture": True})
    batch.smoke("test", "all", "fixture", False, True, True)
    assert batch.approval("test")["scope"] == "all"

    def reject(*args):
        raise ValueError("Recheck failed")

    monkeypatch.setattr(dgx_readiness, "preflight", reject)
    with pytest.raises(ValueError, match="Recheck failed"):
        batch.smoke("test", "all", "fixture", False, True, True)
    with pytest.raises(ValueError, match="not approved"):
        batch.approval("test")


def test_exit_without_verified_artifacts_is_failed(batch, monkeypatch):
    monkeypatch.setattr(batch, "fit", lambda *args: None)
    with pytest.raises(ValueError, match="without a verifiable"):
        batch.execute("first", approval(batch))
    assert batch.state(batch.jobs[0]) == "FAILED"


@pytest.mark.parametrize("missing", [False, True])
def test_failed_final_checkpoint_cannot_become_completed(batch, missing):
    checkpoint = finish(batch, batch.jobs[0])
    if missing:
        checkpoint.unlink()
        failure = FileNotFoundError
    else:
        checkpoint.write_bytes(b"corrupt")
        failure = ValueError
    with pytest.raises(failure):
        batch.execute("first", approval(batch))
    assert batch.state(batch.jobs[0]) == "FAILED"
    assert batch.state(batch.jobs[1]) == "READY"


def test_loss_of_previously_verified_result_blocks_retraining(batch, monkeypatch):
    first = batch.jobs[0]
    checkpoint = finish(batch, first)
    batch.verify_finished("first")
    (checkpoint.parent / "result.json").unlink()
    monkeypatch.setattr(batch, "fit", lambda *args: pytest.fail("would retrain a previously verified result"))
    for _ in range(2):
        with pytest.raises(ValueError, match="Previously completed"):
            batch.execute("first", approval(batch))


def test_matching_checksum_does_not_hide_conflicting_checkpoint_payload(batch):
    job = batch.jobs[0]
    finish(batch, job, checkpoint_identity={"job": "different"})
    with pytest.raises(ValueError, match="checkpoint payload differs"):
        batch.verify_finished("first")
    assert batch.state(job) == "READY"


def test_manifest_conflict_and_immutable_receipt(tmp_path):
    target = tmp_path / "pin.json"
    target.write_bytes(b"{}")
    with pytest.raises(ValueError, match="Manifest conflict"):
        b.load_pinned(target, "f" * 64)
    receipt = tmp_path / "receipt.json"
    b.write_receipt(receipt, {"ready": True})
    before = receipt.read_bytes()
    b.write_receipt(receipt, {"ready": True})
    with pytest.raises(ValueError, match="Immutable receipt conflict"):
        b.write_receipt(receipt, {"ready": False})
    assert receipt.read_bytes() == before


def test_backup_verifies_existing_destination_and_preserves_source(batch, tmp_path):
    original = finish(batch, batch.jobs[0])
    expected = original.read_bytes()
    dest = tmp_path / "approved"
    dest.mkdir()
    copy = dest / ("mom-batch-" + batch.sha) / original.relative_to(batch.root)
    copy.parent.mkdir(parents=True)
    copy.write_bytes(b"existing wrong copy")
    with pytest.raises(ValueError, match="Backup destination differs"):
        batch.backup(dest, "first", True)
    assert batch.state(batch.jobs[0]) == "RESULT_VERIFIED"
    assert original.read_bytes() == expected and copy.read_bytes() == b"existing wrong copy"
    assert not list((batch.dir / "backup-receipts").glob("*.json"))
    other = tmp_path / "other-approved"
    other.mkdir()
    report = batch.backup(other, "first", True)
    assert report["jobs"] == [batch.jobs[0]["id"]]
    assert batch.state(batch.jobs[0]) == "BACKED_UP"
    assert original.read_bytes() == expected


def test_single_artifact_category_has_its_own_verified_revision(batch, tmp_path):
    finish(batch, batch.jobs[0])
    destination = tmp_path / "approved"
    destination.mkdir()
    report = batch.backup(destination, "first", True, "results")
    assert len(report["verified_members"]) == 1
    assert {ref["kind"] for ref in report["verified_members"].values()} == {"results"}
    assert report["backup_revision_sha256"] == next((batch.dir / "backup-receipts").glob("*.json")).stem
    assert batch.state(batch.jobs[0]) == "RESULT_VERIFIED"


def test_restore_can_recover_missing_member_without_overwriting_receipt(batch, monkeypatch):
    batch.data["files"] = {"frozen.json": {}}
    file = batch.runs / "frozen.json"
    calls = []

    def restore(runs, archive, data):
        calls.append(archive)
        file.parent.mkdir(exist_ok=True, parents=True)
        file.write_bytes(b"frozen")
        return {"restored": 1}

    monkeypatch.setattr(b.frozen, "restore", restore)
    batch.restore(Path("package.zip"), True)
    original = batch.stage_path("restored-inputs").read_bytes()
    batch.restore(None, True)
    file.unlink()
    batch.restore(Path("package.zip"), True)
    assert len(calls) == 2
    assert batch.stage_path("restored-inputs").read_bytes() == original
    assert list((batch.dir / "restorations").glob("*.json"))

def test_paired_statistics_use_weighted_canonical_byte_denominator():
    rows = [{"id": "a", "canonical_sha256": "a", "canonical_bytes": 1, "nll_nats": 1.},
            {"id": "b", "canonical_sha256": "b", "canonical_bytes": 99, "nll_nats": 2.}]
    result = lambda docs: {"evaluation": {"per_document": docs}}
    other = [{**d, "nll_nats": 2 * d["nll_nats"]} for d in rows]
    report = b.frozen.pairs({"T5": result(rows), "T8": result(other)})
    stats = report["T8 minus T5"]
    assert stats["delta_bpb"] == pytest.approx(3 / (100 * math.log(2)))
    assert stats["documents_favoring"] == {"T5": 2, "T8": 0}
    with pytest.raises(ValueError, match="alignment"):
        b.frozen.pairs({"T5": result(rows), "T8": result(other[::-1])})


def test_notebook_cells_independent_and_opt_in(tmp_path, monkeypatch):
    book = json.loads((REPO / "notebooks/mom_bpe_dgx_batch.ipynb").read_bytes())
    code = {c["id"]: c for c in book["cells"] if c["cell_type"] == "code"}
    assert list(code) == ["environment", "retrieve-source", "source", "authenticate", "restore", "verify", "plan", "ready", "inspect", "smoke",
                          "first", "first-results", "second", "results", "analyze", "backup", "report"]
    for name, cell in code.items():
        assert not cell["outputs"] and cell["execution_count"] is None
        src = "".join(cell["source"])
        compile(src, name, "exec")
        assert all(word not in src.lower() for word in (".zip", ".tar", "unzip", "--archive"))
    assert 'run("restore-individual")' in "".join(code["restore"]["source"])
    monkeypatch.setenv("MOM_BPE_DEPLOY_REVISION", "a" * 40)
    invoked = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: invoked.append(cmd))
    for _ in range(2):
        for name in ("smoke", "first", "second", "backup"):
            src = "".join(code[name]["source"]).replace('Path("/jupyter/socse.user63/mom-bpe-private-recovery")',
                                                       f"Path({str(tmp_path)!r})")
            src = src.replace("root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077", "False")
            exec(compile(src, name, "exec"), {})
    assert invoked == []


def test_deployment_verifies_revision_and_every_file(tmp_path, monkeypatch):
    revision = "a" * 40
    for name in b.DEPLOYMENT_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())

    def git(command):
        if command[-2] == "rev-parse":
            return revision.encode()
        return command[-1].split(":", 1)[1].encode()

    monkeypatch.setattr(subprocess, "check_output", git)
    assert len(b.verify_deployment(tmp_path, revision)["file_sha256"]) == 7
    with pytest.raises(ValueError, match="commit mismatch"):
        b.verify_deployment(tmp_path, "b" * 40)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="file differs"):
        b.verify_deployment(tmp_path, revision)
    path.unlink()
    with pytest.raises(ValueError, match="Missing GitHub"):
        b.verify_deployment(tmp_path, revision)


class HubFixture:
    def __init__(self, payload, revision="a" * 40, events=()):
        self.payload = payload
        self.revision = revision
        self.events = list(events)
        self.calls = []
        self.checked = []

    def repo_info(self, repo, *, repo_type, revision, timeout):
        self.checked.append((repo, revision))
        return SimpleNamespace(private=True, sha=self.revision)

    def hf_hub_download(self, repo, filename, *, repo_type, revision, local_dir, force_download):
        self.calls.append((revision, force_download, local_dir))
        path = local_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        event = self.events.pop(0) if self.events else None
        if event == "interrupt":
            path.with_suffix(".incomplete").write_bytes(b"partial")
            raise KeyboardInterrupt
        if isinstance(event, BaseException):
            raise event
        path.write_bytes(event if event is not None else self.payload)
        return str(path)

    def create_commit(self, *args, **kwargs):
        pytest.fail("The restoration path attempted an upload")


def hub_inventory():
    return {"refs/result.json": {"size": 4, "sha256": hashlib.sha256(b"good").hexdigest(),
                                "source": {"repo": "owner/repo", "revision": "a" * 40, "filename": "result.json"}}}


def test_hub_wrong_revision_and_missing_remote_member(tmp_path):
    inventory = hub_inventory()
    wrong = HubFixture(b"good", revision="b" * 40)
    row = b.restore_members(tmp_path, tmp_path / "stage", inventory, wrong)["refs/result.json"]
    assert row["status"] == "unverified_revision" and not wrong.calls
    error = type("RemoteEntryNotFoundError", (Exception,), {})()
    missing = HubFixture(b"good", events=[error])
    row = b.restore_members(tmp_path, tmp_path / "stage", inventory, missing)["refs/result.json"]
    assert row["status"] == "download_failed" and len(missing.calls) == 1
    assert not (tmp_path / "refs/result.json").exists()


def test_hub_bad_sha_is_retried_and_never_published(tmp_path):
    api = HubFixture(b"bad!")
    row = b.restore_members(tmp_path, tmp_path / "stage", hub_inventory(), api)["refs/result.json"]
    assert row["status"] == "download_failed"
    assert [force for _, force, _ in api.calls] == [False, True, True]
    assert not (tmp_path / "refs/result.json").exists()


def test_hub_interrupted_transfer_resumes_in_same_stage_and_reuses_verified_file(tmp_path):
    api = HubFixture(b"good", events=["interrupt"])
    with pytest.raises(KeyboardInterrupt):
        b.restore_members(tmp_path, tmp_path / "stage", hub_inventory(), api)
    assert not (tmp_path / "refs/result.json").exists()
    row = b.restore_members(tmp_path, tmp_path / "stage", hub_inventory(), api)["refs/result.json"]
    assert row["status"] == "restored"
    assert api.calls[0][2] == api.calls[1][2]
    offline = HubFixture(b"bad!")
    row = b.restore_members(tmp_path, tmp_path / "stage", hub_inventory(), offline)["refs/result.json"]
    assert row["status"] == "reused" and not offline.checked and not offline.calls


def test_conflict_preserves_original_and_saves_verified_replacement(tmp_path):
    path = tmp_path / "refs/result.json"
    path.parent.mkdir()
    path.write_bytes(b"bad!")
    row = b.restore_members(tmp_path, tmp_path / "stage", hub_inventory(), HubFixture(b"good"))["refs/result.json"]
    assert row["status"] == "conflict_preserved" and path.read_bytes() == b"bad!"
    assert Path(row["verified_replacement"]).read_bytes() == b"good"


def test_unmapped_inputs_and_unauthorized_backup_do_not_contact_hub(tmp_path, monkeypatch):
    api = HubFixture(b"good")
    entry = {"size": 4, "sha256": hashlib.sha256(b"good").hexdigest(), "source": None}
    row = b.restore_members(tmp_path, tmp_path / "stage", {"frozen/cache.bin": entry}, api)["frozen/cache.bin"]
    assert row["status"] == "missing_local_only" and not api.checked and not api.calls
    monkeypatch.setattr(b, "hub_client", lambda: pytest.fail("Unapproved backup accessed Hub credentials"))
    batch = b.Batch(tmp_path, {"queue": []}, {})
    with pytest.raises(ValueError, match="explicit authorization"):
        batch.backup_hub("owner/private", "all", False, False)
