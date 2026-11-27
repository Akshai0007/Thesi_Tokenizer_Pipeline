"""CPU-only offline tests for DGX batch checkpoint recovery, gating, and lock protection."""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mom_bpe import dgx_batch as b
from experiments.mom_bpe import dgx_resources as gpu
from experiments.mom_bpe import document_proxy as driver
from experiments.mom_bpe.proxy_blocks import publish, receipt, write_split
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.canonical_bpb import summarize_bpb
from mltok.lib.mom_backend import SurfaceEncoding
from mltok.snapshots import SnapshotError, _lock

REPO = Path(__file__).resolve().parents[1]


def _runtime_identity():
    return {
        "system": "Linux",
        "python": "3.12.0",
        "torch": "2.5.1+cu121",
        "numpy": "2.5.2",
        "threads": 1,
        "precision": "bf16_autocast",
        "cublas_workspace": ":4096:8",
        "gpu": {
            "physical_index": 1,
            "uuid": "GPU-deadbeef-1234-1234-1234-123456789abc",
            "type": "NVIDIA A100-SXM4-80GB",
            "cuda": "12.1",
        },
    }


def _save_checkpoint(out, identity, step, exposure, *, lr=0.001, train_nll=1.0):
    buf = io.BytesIO()
    torch.save(
        {
            "identity": identity,
            "step": step,
            "exposure": exposure,
            "scheduler": {"lr": lr},
            "train_nll_nats": train_nll,
            "model": {"weight": torch.ones(1)},
            "optimizer": {"state": {0: {"step": torch.tensor(step)}}, "param_groups": [{"lr": lr}]},
            "rng": dict.fromkeys(("python", "numpy", "torch", "cuda")),
        },
        buf,
    )
    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    path = out / f"step-{step:06d}-{digest}.pt"
    out.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path, digest, len(payload)


def _job_identity(batch, job, plan):
    return {
        **plan,
        "manifest_sha256": batch.data["corpus_manifest_sha256"],
        "tokenizer_sha256": batch.data["arms"][job["arm"]]["tokenizer_sha256"],
        "parameters": 38799936,
        "implementation": {
            "driver": gpu.PINNED_DRIVER,
            "blocks": driver.code_digest(driver.blocks.__code__.co_filename),
            "scorer": driver.code_digest(driver.summarize_bpb.__code__.co_filename),
        },
        "model_plugin": driver.registry.fingerprint("proxy_model", plan["config"]["model"]),
        "runtime": _runtime_identity(),
    }


def _complete_job(batch, job, plan, held_rows):
    out = batch.root / job["output"]
    identity = _job_identity(batch, job, plan)
    ckpt_path, digest, size = _save_checkpoint(
        out,
        identity,
        plan["forecast"]["optimizer_updates"],
        plan["forecast"],
    )
    docs = [{**row, **summarize_bpb([1.0], row["canonical_bytes"], row["length"] + 1)} for row in held_rows]
    metrics = summarize_bpb([1.0] * len(docs), sum(r["canonical_bytes"] for r in docs), sum(r["predicted_tokens"] for r in docs))
    (out / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    b.write_receipt(
        out / "result.json",
        {
            "identity": identity,
            "training_nll_nats": 1.0,
            "evaluation": {"per_document": docs, **metrics},
            "checkpoint": {"path": ckpt_path.name, "size": size, "sha256": digest},
        },
    )
    return out


@pytest.fixture
def real_audit_batch(tmp_path, monkeypatch):
    driver.registry.load_plugins()
    model_cfg = {"type": "gpt", "n_layer": 7, "n_head": 7, "d_model": 448}
    jobs, plans = [], {}
    for mode in b.MODES:
        for arm in ("T5", "T8"):
            job = {
                "id": f"{arm}-{mode}",
                "arm": arm,
                "mode": mode,
                "seed": 17,
                "budget": 3,
                "forecast": {"optimizer_updates": 3},
                "validation": {},
                "output": f"runs/{arm}-{b.MODES.index(mode)}",
            }
            plan = {k: job[k] for k in ("arm", "mode", "budget", "forecast", "validation")}
            plan["config"] = {"seed": 17, "checkpoint_updates": 1, "model": model_cfg}
            job["plan_sha256"] = b.metadata_digest(plan)
            jobs.append(job)
            plans[job["id"]] = plan

    input_manifest = {
        "parameters": 38799936,
        "corpus_manifest_sha256": "c" * 64,
        "arms": {
            "T5": {"tokenizer_sha256": "a" * 64},
            "T8": {"tokenizer_sha256": "b" * 64},
        },
    }
    batch = b.Batch(tmp_path / "root", {"queue": jobs}, input_manifest, manifest_sha="fixture")
    batch.stage("batch-manifest", {"manifest": batch.manifest})
    batch.stage("verified-inputs", {"plans": {j["arm"]: plans[j["id"]] for j in jobs[:2]}})
    batch.stage("verified-plans", {"plans": plans})
    batch.stage("cpu-ready", {"ready": True})
    for old, new in (("PENDING", "INPUT_VERIFIED"), ("INPUT_VERIFIED", "PLAN_VERIFIED"), ("PLAN_VERIFIED", "READY")):
        batch.advance_preparation(old, new)

    held_rows = [{"id": str(i), "canonical_sha256": f"{i:064x}", "canonical_bytes": i + 1, "length": 1} for i in range(186)]
    monkeypatch.setattr(batch, "held", lambda _: held_rows)
    monkeypatch.setattr(batch, "storage", lambda *args: {"free_bytes": 100 * b.GIB, "additional_bytes_reserved": 1})
    return batch, plans, held_rows


def test_execute_resumes_from_valid_earlier_checkpoint_when_newest_is_truncated(real_audit_batch, monkeypatch):
    batch, plans, held_rows = real_audit_batch
    first, second = batch.selection("first")
    out1 = batch.root / first["output"]
    id1 = _job_identity(batch, first, plans[first["id"]])
    (out1).mkdir(parents=True, exist_ok=True)
    (out1 / "identity.json").write_text(json.dumps(id1), encoding="utf-8")
    p1, _, _ = _save_checkpoint(out1, id1, 1, {"optimizer_updates": 1})
    p2, _, _ = _save_checkpoint(out1, id1, 2, {"optimizer_updates": 2})
    p2.write_bytes(b"truncated-corrupt-checkpoint")

    resumed_from = []

    def fake_fit(job, plan, approval, log):
        out = batch.root / job["output"]
        if job["id"] == first["id"]:
            # Emulate restore_generation selecting newest checksum-valid generation.
            for candidate in sorted(out.glob("step-*.pt"), reverse=True):
                match = driver.GENERATION.fullmatch(candidate.name)
                if hashlib.sha256(candidate.read_bytes()).hexdigest() == match[2]:
                    state = torch.load(candidate, map_location="cpu", weights_only=True)
                    resumed_from.append(state["step"])
                    break
        _complete_job(batch, job, plan, held_rows)

    monkeypatch.setattr(batch, "fit", fake_fit)
    states = batch.execute("first", {"manifest_sha256": batch.sha, "scope": "all", "uuid": "fake", "shared": False})
    assert states == {first["id"]: "RESULT_VERIFIED", second["id"]: "RESULT_VERIFIED"}
    assert resumed_from == [1]
    # Damaged checkpoint must remain untouched for forensic audit.
    assert p2.read_bytes() == b"truncated-corrupt-checkpoint"
    assert p1.is_file()


def test_execute_fails_safely_when_all_intermediate_checkpoints_are_corrupt(real_audit_batch, monkeypatch):
    batch, plans, _ = real_audit_batch
    first, second = batch.selection("first")
    out1 = batch.root / first["output"]
    id1 = _job_identity(batch, first, plans[first["id"]])
    out1.mkdir(parents=True, exist_ok=True)
    (out1 / "identity.json").write_text(json.dumps(id1), encoding="utf-8")
    p1, _, _ = _save_checkpoint(out1, id1, 1, {"optimizer_updates": 1})
    p2, _, _ = _save_checkpoint(out1, id1, 2, {"optimizer_updates": 2})
    p1.write_bytes(b"corrupt-1")
    p2.write_bytes(b"corrupt-2")

    monkeypatch.setattr(batch, "fit", lambda *args: pytest.fail("Started training despite corrupt checkpoints"))
    with pytest.raises(ValueError, match="No checksum-valid generation"):
        batch.execute("first", {"manifest_sha256": batch.sha, "scope": "all", "uuid": "fake", "shared": False})
    assert batch.state(first) == "FAILED"
    assert batch.state(second) == "READY"
    assert p1.read_bytes() == b"corrupt-1" and p2.read_bytes() == b"corrupt-2"


def test_fit_cache_resumes_from_final_checkpoint_without_result_json(tmp_path, monkeypatch):
    class ByteBackend:
        special_ids = dict(zip(DEFAULT_SPECIALS, range(len(DEFAULT_SPECIALS))))
        fallback_ids = set(range(len(DEFAULT_SPECIALS), len(DEFAULT_SPECIALS) + 256))
        vocab_size = 48000

        def encode_offsets(self, text):
            data = text.encode("utf-8")
            base = len(DEFAULT_SPECIALS)
            return SurfaceEncoding(text, [base + b for b in data], [f"<0x{b:02X}>" for b in data], [])

        def decode(self, ids):
            base = len(DEFAULT_SPECIALS)
            return bytes(i - base for i in ids).decode("utf-8")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    backend = ByteBackend()

    def make_row(doc_id, text):
        raw = text.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        return {"id": doc_id, "text": text, "sha256": digest, "canonical_sha256": digest,
                "raw_bytes": len(raw), "canonical_bytes": len(raw)}

    train_split = write_split(cache_dir, "train", iter([make_row("t0", "ab")]), backend, str, False, None)
    val_split = write_split(cache_dir, "validation", iter([make_row("v0", "cd")]), backend, str, False, None)
    cache_id = {
        "schema": 1,
        "context": 512,
        "policy": driver.POLICY,
        "arm": "T8",
        "vocab": driver.VOCAB,
        "specials": backend.special_ids,
        "tokenizer_sha256": "t" * 64,
        "manifest_sha256": "m" * 64,
        "streams": {"validation": "v" * 64},
    }
    publish(cache_dir / "cache.json", {"identity": cache_id, "splits": {"train": train_split, "validation": val_split}})

    monkeypatch.setattr(driver.shutil, "disk_usage", lambda _: SimpleNamespace(free=64 * b.GIB))
    out_dir = tmp_path / "run"
    args = SimpleNamespace(
        command="run",
        cache=cache_dir,
        mode="fixed_training_tokens",
        budget=3,
        out=out_dir,
        seed=17,
        update_targets=3,
        microbatch_blocks=1,
        checkpoint_updates=1,
        threads=1,
        device="cpu",
    )

    real_score = driver.score

    def fail_during_score(*_a, **_kw):
        raise KeyboardInterrupt("simulated interrupt before result.json")

    monkeypatch.setattr(driver, "score", fail_during_score)
    with pytest.raises(KeyboardInterrupt, match="simulated interrupt"):
        driver.fit_cache(args)

    checkpoints = list(out_dir.glob("step-000001-*.pt"))
    assert len(checkpoints) == 1
    assert not (out_dir / "result.json").exists()

    monkeypatch.setattr(driver, "score", real_score)
    real_train = driver.train
    calls = []

    def verify_no_train_updates(out, identity, split, planned, progress, new_state, config, mode, specials, device, **kw):
        # Ensure train returns without stepping optimizer when final checkpoint is restored.
        model, nll, ckpt = real_train(out, identity, split, planned, progress, new_state, config, mode, specials, device, **kw)
        calls.append((ckpt["path"], len(planned)))
        return model, nll, ckpt

    monkeypatch.setattr(driver, "train", verify_no_train_updates)
    monkeypatch.setattr(driver.torch.optim.AdamW, "step", lambda *_: pytest.fail("Optimizer stepped on final-step resume"))

    result = driver.fit_cache(args)
    assert calls == [(checkpoints[0].name, 1)]
    assert (out_dir / "result.json").is_file()
    saved, _ = receipt(out_dir / "result.json")
    assert saved["checkpoint"]["path"] == checkpoints[0].name
    assert result["evaluation"]["canonical_bytes"] == 2
    assert result["evaluation"]["predicted_tokens"] == 3


def test_corrupt_pair1_result_or_identity_blocks_second_pair_and_analyze(real_audit_batch):
    batch, plans, held_rows = real_audit_batch
    for job in batch.selection("first"):
        _complete_job(batch, job, plans[job["id"]], held_rows)
    batch.result_check("first")

    first = batch.selection("first")[0]
    out1 = batch.root / first["output"]
    result_path = out1 / "result.json"
    identity_path = out1 / "identity.json"
    orig_result = result_path.read_bytes()
    orig_identity = identity_path.read_bytes()

    approval = {"manifest_sha256": batch.sha, "scope": "all", "uuid": "fake", "shared": False}

    # Tamper result.json
    tampered = json.loads(orig_result)
    tampered["training_nll_nats"] = 999.0
    result_path.write_bytes(json.dumps(tampered).encode("utf-8"))
    with pytest.raises(ValueError, match="Receipt SHA-256 differs"):
        batch.execute("second", approval)
    with pytest.raises(ValueError, match="Receipt SHA-256 differs"):
        batch.analyze()

    # Restore result.json and tamper identity.json
    result_path.write_bytes(orig_result)
    bad_id = json.loads(orig_identity)
    bad_id["parameters"] = 123
    identity_path.write_bytes(json.dumps(bad_id).encode("utf-8"))
    with pytest.raises(ValueError, match="Run identity conflict"):
        batch.execute("second", approval)
    with pytest.raises(ValueError, match="Run identity conflict"):
        batch.analyze()


def test_verify_inputs_rejects_tampered_historical_reference_json(tmp_path):
    from experiments.mom_bpe.dgx_readiness import PINS

    root = tmp_path / "root"
    root.mkdir()
    inputs = json.loads((REPO / "experiments/mom_bpe/t5_t8_inputs_v1.json").read_text(encoding="utf-8"))
    batch_manifest = json.loads((REPO / "experiments/mom_bpe/t5_t8_batch_v1.json").read_text(encoding="utf-8"))
    batch = b.Batch(root, batch_manifest, inputs)

    # Populate dummy frozen files matching expected size/sha256 only up to verified-inputs fast path,
    # then create verified-inputs receipt and tamper a historical baseline file.
    for rel, ref in inputs["files"].items():
        path = batch.runs / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    # Patch file expectations to empty bytes on this synthetic copy so check_historical_hashes is reached.
    batch.data = {**inputs, "files": {rel: {"size": 0, "sha256": hashlib.sha256(b"").hexdigest()} for rel in inputs["files"]}}
    batch.stage("verified-inputs", {"plans": {}, "raw_corpus": {}, "baselines": {}})

    for (arm, mode), (id_sha, res_sha) in PINS.items():
        folder = batch.baseline / "seed17" / arm / mode
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "identity.json").write_text(json.dumps({"pin": id_sha}), encoding="utf-8")
        (folder / "result.json").write_text(json.dumps({"pin": res_sha}), encoding="utf-8")

    # Even with verified-inputs receipt present, check_historical_hashes fails immediately on PINS mismatch.
    with pytest.raises(ValueError, match="Historical reference differs"):
        batch.verify_inputs()


def test_controller_and_ablation_lock_contention_raises_snapshot_error(tmp_path):
    lock_path = tmp_path / ".controller.lock"
    with _lock(lock_path):
        with pytest.raises(SnapshotError, match="Another process is synchronizing"):
            with _lock(lock_path):
                pass


def test_batch_storage_enforces_first_pair_8gib_floor_and_partial_checkpoint_math(real_audit_batch):
    batch, plans, _ = real_audit_batch
    # 2 unstarted first-pair jobs -> 8 GiB floor (call unbound method to bypass fixture stub)
    first_pair = b.Batch.storage(batch, batch.selection("first"))
    assert first_pair["additional_bytes_reserved"] == 8 * b.GIB

    # 4 unstarted jobs -> 15 GiB floor
    all_jobs = b.Batch.storage(batch, batch.jobs)
    assert all_jobs["additional_bytes_reserved"] == 15 * b.GIB

    # Partial recovery: 1 valid + 1 corrupt checkpoint present out of 3 on job 0 (2 + 3 = 5 missing across first pair)
    first_job = batch.jobs[0]
    partial = b.Batch.storage(
        batch,
        batch.selection("first"),
        recovery={
            first_job["id"]: {
                f"step-000001-{'a' * 64}.pt": {"valid_bytes": True},
                f"step-000002-{'b' * 64}.pt": {"valid_bytes": False},
            }
        },
    )
    expected_bytes = (5 + 2) * (12 * batch.data["parameters"] + 2**20)
    assert partial["additional_bytes_reserved"] == expected_bytes
    assert partial["additional_bytes_reserved"] < 8 * b.GIB
