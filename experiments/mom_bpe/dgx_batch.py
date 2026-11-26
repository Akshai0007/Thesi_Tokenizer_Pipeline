"""Restartable DGX orchestration around the pinned document-proxy trainer."""

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from time import time_ns
from uuid import uuid4

if __package__:
    from . import t5_t8_dgx as frozen
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import t5_t8_dgx as frozen

OPERATIONAL = "9b872ffb5a66bdf4d23df58acebbba7e788915d8"
BATCH_SHA = "d0bfb6b3994cff4ca2c0159718e0a29da91e1f8aa029f5fa3d7169541905a5b7"
BATCH_DIR = "runs/dp/t5-t8-seed17-batch-v1"
FIRST_DIR = "runs/dp/t5-t8-seed17-fixed-source-v1"
MODES = ("fixed_source_bytes", "fixed_training_tokens")
GIB = 2**30
BACKUP_KINDS = ("checkpoints", "results", "identities", "configuration", "telemetry", "manifests",
                "receipts", "analysis", "logs")
require = frozen.require
TRANSITIONS = {
    "PENDING": {"INPUT_VERIFIED"},
    "INPUT_VERIFIED": {"PLAN_VERIFIED"},
    "PLAN_VERIFIED": {"READY"},
    "READY": {"RUNNING", "FAILED", "COMPLETED"},
    "RUNNING": {"INTERRUPTED", "FAILED", "COMPLETED"},
    "INTERRUPTED": {"RUNNING", "FAILED", "COMPLETED"},
    "FAILED": {"RUNNING", "FAILED", "COMPLETED"},
    "COMPLETED": {"RESULT_VERIFIED", "FAILED"},
    "RESULT_VERIFIED": {"BACKED_UP", "FAILED"},
    "BACKED_UP": {"FAILED"},
}


def load_pinned(path, digest):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, f"Manifest conflict: {path.name}")
    return json.loads(data)


def metadata_digest(data):
    from mltok.lib.bpe_progress import canonical

    return hashlib.sha256(canonical(data)).hexdigest()


def write_receipt(path, data):
    from mltok.lib.atomic import atomic_writer

    digest = metadata_digest(data)
    payload = (json.dumps({**data, "sha256": digest}, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with atomic_writer(path, overwrite=False) as stream:
            stream.write(payload)
    except FileExistsError:
        require(path.read_bytes() == payload, f"Immutable receipt conflict: {path}")
    return digest


def read_receipt(path):
    data = frozen.read(path)
    expected = data.pop("sha256")
    require(metadata_digest(data) == expected, f"Receipt SHA-256 differs: {path}")
    return data


def verify_source(source):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    require(git("rev-parse", "HEAD") == OPERATIONAL
            and not git("status", "--porcelain", "--ignored", "--untracked-files=all"),
            "Expected the clean, dedicated operational checkout")


def verify_environment():
    import importlib.metadata as md
    import unicodedata
    import torch

    expected = {"regex": "2026.9.29", "tokenizers": "0.23.3", "PyYAML": "6.0.3", "fasttext-wheel": "0.9.2",
                "pyarrow": "26.0.0", "numpy": "2.5.2", "torch": "2.5.1+cu121"}
    observed = {name: md.version(name) for name in expected}
    runtime = (platform.system(), platform.python_version(), unicodedata.unidata_version, torch.version.cuda)
    require(runtime == ("Linux", "3.12.0", "15.0.0", "12.1") and observed == expected,
            f"DGX environment differs: {runtime}, {observed}")
    return {"python": runtime[1], "unicode": runtime[2], "cuda": runtime[3], "dependencies": observed}


def copy_verified(src, dst, expected):
    """Publish a complete copy only after checking both ends."""
    from experiments.mom_bpe.dgx_handoff import digest_stream, file_digest
    from mltok.lib.atomic import atomic_writer

    require(file_digest(src) == expected, f"Backup source differs: {src}")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as stream, atomic_writer(dst, overwrite=False) as out:
            require(digest_stream(stream, out) == expected, f"Backup source changed: {src}")
    require(file_digest(dst) == expected, f"Backup destination differs: {dst}")


class Batch:
    def __init__(self, root, queue_manifest, input_manifest, *, manifest_sha=BATCH_SHA, directory=BATCH_DIR):
        self.root = root
        self.dir = frozen.safe(root, directory)
        self.runs = frozen.safe(root, "t5-t8-frozen-runs-v1")
        self.baseline = frozen.safe(root, "t5-t8-seed17-reference-v1")
        self.manifest = queue_manifest
        self.data = input_manifest
        self.sha = manifest_sha
        self.jobs = tuple(queue_manifest["queue"])
        for field in ("id", "output"):
            require(len({job[field] for job in self.jobs}) == len(self.jobs), f"Duplicate queue {field}")

    def selection(self, group):
        if group == "all":
            return self.jobs
        return tuple(job for job in self.jobs if job["mode"] == MODES[("first", "second").index(group)])

    def stage_path(self, name):
        return frozen.safe(self.dir, name + ".json")

    def stage(self, name, payload=None):
        path = self.stage_path(name)
        if payload is not None:
            write_receipt(path, {"manifest_sha256": self.sha, **payload})
        require(path.is_file(), f"Missing {name} receipt")
        data = read_receipt(path)
        require(data["manifest_sha256"] == self.sha, f"Receipt manifest conflict: {name}")
        return data

    def events(self, job):
        directory = frozen.safe(self.dir, "state/" + job["id"])
        return sorted(directory.glob("[0-9]*.json"))

    def state(self, job):
        events = self.events(job)
        if not events:
            return "PENDING"
        return self.event(job, events[-1])["state"]

    def event(self, job, path):
        record = read_receipt(frozen.safe(path.parent, path.name))
        require(record["manifest_sha256"] == self.sha and record["job"] == job["id"], "Job state conflict")
        return record

    def record(self, job, state, **detail):
        prior = self.state(job)
        require(state in TRANSITIONS[prior], f"Invalid transition from {prior} to {state}")
        path = frozen.safe(self.dir, f"state/{job['id']}/{len(self.events(job)) + 1:08d}.json")
        write_receipt(path, {"manifest_sha256": self.sha, "job": job["id"], "state": state, **detail})

    def advance_preparation(self, old, new):
        for job in self.jobs:
            if self.state(job) == old:
                self.record(job, new)

    def restore(self, archive, enabled):
        require(enabled, "Archive restoration requires an explicit opt-in")
        previous = self.stage_path("restored-inputs").exists()
        if not previous or any(not frozen.safe(self.runs, name).is_file() for name in self.data["files"]):
            require(archive is not None, "Supply the exact Package B path")
            restored = frozen.restore(self.runs, archive, self.data)
            if not previous:
                self.stage("restored-inputs", restored)
            else:
                self.save_named("restorations", {"manifest_sha256": self.sha, **restored})
        return self.stage("restored-inputs")

    def verify_inputs(self):
        if self.stage_path("verified-inputs").exists():
            for name, entry in self.data["files"].items():
                frozen.verified_file(frozen.safe(self.runs, name), entry)
            for mode in MODES:
                self.check_historical_hashes(mode)
        else:
            plans, held, raw = frozen.verify_inputs(self.runs, self.data)
            historical = {mode: self.historical(mode, held["T5"]) for mode in MODES}
            self.stage("verified-inputs", {"plans": plans, "raw_corpus": raw,
                        "baselines": {m: {a: r["evaluation"]["bpb"] for a, r in arms.items()}
                                      for m, arms in historical.items()}})
        result = self.stage("verified-inputs")
        self.advance_preparation("PENDING", "INPUT_VERIFIED")
        return result

    def verify_plan(self, job, plan):
        require(metadata_digest(plan) == job["plan_sha256"] and plan["mode"] == job["mode"]
                and plan["arm"] == job["arm"] and plan["config"]["seed"] == job["seed"]
                and plan["budget"] == job["budget"] and plan["forecast"] == job["forecast"]
                and plan["validation"] == job["validation"], f"Exposure plan conflict: {job['id']}")

    def plan(self):
        from experiments.mom_bpe import document_proxy as driver
        from experiments.mom_bpe.dgx_handoff import plan as pinned_plan

        verified = self.stage("verified-inputs")["plans"]
        self.advance_preparation("PENDING", "INPUT_VERIFIED")
        if not self.stage_path("verified-plans").exists():
            plans = {}
            for job in self.jobs:
                cache = self.cache(job["arm"])
                value = verified[job["arm"]] if job["mode"] == MODES[0] else pinned_plan(driver, cache, job["mode"])
                self.verify_plan(job, value)
                plans[job["id"]] = value
            self.stage("verified-plans", {"plans": plans})
        result = self.stage("verified-plans")
        self.advance_preparation("INPUT_VERIFIED", "PLAN_VERIFIED")
        return result

    def cache(self, arm):
        return frozen.safe(self.runs, self.data["arms"][arm]["cache"])

    def held(self, arm):
        from experiments.mom_bpe.proxy_blocks import EncodedSplit, receipt

        cache = self.cache(arm)
        self.verify_cache_pin(arm)
        data, _ = receipt(cache / "cache.json")
        return EncodedSplit(cache, "validation", data["splits"]["validation"]).docs

    def verify_cache_pin(self, arm):
        name = self.data["arms"][arm]["cache"] + "/cache.json"
        frozen.verified_file(frozen.safe(self.runs, name), self.data["files"][name])

    def storage(self, jobs, recovery=None):
        plans = self.stage("verified-plans")["plans"]
        missing = 0
        for job in jobs:
            if self.state(job) in ("COMPLETED", "RESULT_VERIFIED", "BACKED_UP"):
                continue
            plan = plans[job["id"]]
            interval, final = plan["config"]["checkpoint_updates"], plan["forecast"]["optimizer_updates"]
            expected = set(range(interval, final, interval)) | {final}
            checked = (recovery or {}).get(job["id"], {})
            present = set()
            if checked:
                from experiments.mom_bpe.document_proxy import GENERATION

                present = {int(GENERATION.fullmatch(name)[1]) for name, ref in checked.items() if ref["valid_bytes"]}
            missing += len(expected - present)
        # Three FP32 state arrays per parameter plus serialization overhead.
        required = (missing + 2) * (12 * self.data["parameters"] + 2**20)
        total_generations = sum(math.ceil(plans[job["id"]]["forecast"]["optimizer_updates"] /
                                          plans[job["id"]]["config"]["checkpoint_updates"]) for job in jobs)
        if missing == total_generations and jobs:
            required = max(required, (15 if len(jobs) == 4 else 8) * GIB)
        disk = shutil.disk_usage(self.dir)
        return {"free_bytes": disk.free, "filesystem_used_bytes": disk.used,
                "filesystem_total_bytes": disk.total, "additional_bytes_reserved": required}

    def ready(self, group):
        resource = self.storage(self.selection(group))
        require(resource["free_bytes"] >= resource["additional_bytes_reserved"], "Insufficient batch checkpoint storage")
        if not self.stage_path("cpu-ready").exists():
            self.stage("cpu-ready", {"ready": True})
        self.stage("cpu-ready")
        self.advance_preparation("PLAN_VERIFIED", "READY")
        return resource

    def approval(self, session):
        path = frozen.safe(self.dir, f"sessions/{session}")
        attempts = sorted(path.glob("smoke-*"))
        require(attempts, "Run the authorized smoke for this session")
        receipt_path = frozen.safe(attempts[-1], "approved.json")
        require(receipt_path.is_file(), "Latest smoke attempt was not approved")
        return read_receipt(receipt_path)

    def smoke(self, session, scope, uuid, shared, allocated, enabled):
        from experiments.mom_bpe.dgx_readiness import preflight

        require(enabled and allocated, "GPU smoke requires explicit authorization and institutional allocation")
        self.stage("cpu-ready")
        out = frozen.safe(self.dir, f"sessions/{session}/smoke-{time_ns()}-{uuid4().hex}")
        out.mkdir(parents=True)
        resource = self.storage(self.selection(scope))
        require(resource["free_bytes"] >= resource["additional_bytes_reserved"], "Insufficient checkpoint storage for approval scope")
        result = preflight(out, uuid, shared, allocated)
        write_receipt(out / "approved.json", {"manifest_sha256": self.sha, "scope": scope,
                      "uuid": uuid, "shared": shared, "smoke": result})
        return result

    def check_historical_hashes(self, mode):
        from experiments.mom_bpe.dgx_handoff import file_digest
        from experiments.mom_bpe.dgx_readiness import PINS

        for arm in frozen.BASELINES:
            path = frozen.safe(self.baseline, f"seed17/{arm}/{mode}")
            for leaf, digest in zip(("identity.json", "result.json"), PINS[arm, mode], strict=True):
                require(file_digest(frozen.safe(path, leaf))[1] == digest, f"Historical reference differs: {arm}/{mode}/{leaf}")

    def historical(self, mode, held):
        if mode == MODES[0]:
            return frozen.load_baselines(self.baseline, self.data, held)
        self.check_historical_hashes(mode)
        refs = {}
        for arm in frozen.BASELINES:
            path = frozen.safe(self.baseline, f"seed17/{arm}/{mode}")
            result = read_receipt(path / "result.json")
            identity = frozen.read(path / "identity.json")
            require(result["identity"] == identity and identity["arm"] == arm and identity["mode"] == mode
                    and identity["config"]["seed"] == 17 and identity["budget"] == 5_000_000
                    and identity["forecast"]["predicted_tokens"] == 5_000_000
                    and identity["parameters"] == self.data["parameters"]
                    and identity["manifest_sha256"] == self.data["corpus_manifest_sha256"]
                    and identity["optimizer_sha256"] == self.data["optimizer_sha256"]
                    and identity["config_sha256"] == self.data["config_sha256"], "Historical matched-token identity differs")
            frozen.verify_result(result, held, check_targets=False)
            refs[arm] = result
        return refs

    def validate_completed(self, job, plan, held):
        from experiments.mom_bpe.document_proxy import GENERATION
        from experiments.mom_bpe.dgx_handoff import file_digest

        out = frozen.safe(self.root, job["output"])
        file = frozen.safe(out, "result.json")
        earlier = [self.event(job, p) for p in self.events(job)]
        verified = [event for event in earlier if event["state"] == "RESULT_VERIFIED"]
        if not file.exists():
            require(not any(event["state"] in ("COMPLETED", "RESULT_VERIFIED", "BACKED_UP") for event in earlier),
                    f"Previously completed result is missing: {job['id']}")
            return None
        result = read_receipt(file)
        from experiments.mom_bpe import document_proxy as driver
        driver.registry.load_plugins()
        identity = frozen.read(frozen.safe(out, "identity.json"))
        frozen.verify_identity(identity, plan, job["arm"], self.data)
        require(result["identity"] == identity and all(p["result_identity"] == metadata_digest(result) for p in verified),
                "Final result identity differs")
        ref = result["checkpoint"]
        name = ref["path"]
        match = GENERATION.fullmatch(name)
        require(match and int(match[1]) == plan["forecast"]["optimizer_updates"] and match[2] == ref["sha256"],
                "Final checkpoint reference differs")
        checkpoint = frozen.safe(out, name)
        require(file_digest(checkpoint) == (ref["size"], ref["sha256"]), "Final checkpoint is corrupt")
        if not verified:
            state = driver.torch.load(checkpoint, map_location="cpu", weights_only=True)
            require(state["identity"] == identity and state["step"] == int(match[1])
                    and state["exposure"] == plan["forecast"]
                    and math.isfinite(state["train_nll_nats"]) and state["train_nll_nats"] >= 0
                    and state["train_nll_nats"] == result["training_nll_nats"]
                    and state["model"] and all(driver.torch.is_tensor(tensor) for tensor in state["model"].values())
                    and state["optimizer"]["state"] and state["optimizer"]["param_groups"]
                    and all(group["lr"] == state["scheduler"]["lr"] for group in state["optimizer"]["param_groups"])
                    and {"python", "numpy", "torch", "cuda"} <= state["rng"].keys(),
                    "Final checkpoint payload differs")
        frozen.verify_result(result, held, check_targets=True)
        return result

    def complete(self, job, result):
        if self.state(job) not in ("COMPLETED", "RESULT_VERIFIED", "BACKED_UP"):
            self.record(job, "COMPLETED", checkpoint=result["checkpoint"])
        if self.state(job) == "COMPLETED":
            self.record(job, "RESULT_VERIFIED", result_identity=metadata_digest(result))

    def admit(self, approval):
        from experiments.mom_bpe import dgx_resources as gpu
        from experiments.mom_bpe import document_proxy as driver

        policy = gpu.GpuAdmission(5120, 10, True) if approval["shared"] else gpu.GpuAdmission()
        observed = policy.admit(approval["uuid"])
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == approval["uuid"], "GPU mask differs from the allocation")
        device = driver.gpu_preflight(policy.min_free_mib, max_gpu_util_pct=policy.max_gpu_util_pct,
                                      allow_shared_gpu=policy.allow_shared_gpu)
        require("A100" in device["type"] and device["physical_index"] == observed["physical_index"], "Expected allocated A100")
        return policy, device

    def fit(self, job, plan, approval, log):
        from experiments.mom_bpe import document_proxy as driver
        from types import SimpleNamespace

        out = frozen.safe(self.root, job["output"])
        self.verify_cache_pin(job["arm"])
        policy, hardware = self.admit(approval)
        identity = frozen.safe(out, "identity.json")
        if identity.exists():
            require(frozen.read(identity)["runtime"]["gpu"] == hardware, "Resume on the original allocated GPU")
        args = SimpleNamespace(command="run", cache=self.cache(job["arm"]), out=out, mode=job["mode"],
                               device="cuda", threads=1, budget=None, seed=None, update_targets=None,
                               microbatch_blocks=None, checkpoint_updates=None, min_free_mib=policy.min_free_mib,
                               max_gpu_util_pct=policy.max_gpu_util_pct, allow_shared_gpu=policy.allow_shared_gpu)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            driver.fit_cache(args)
            # Release this process's idle cache before checking resources for the next job.
            driver.torch.cuda.empty_cache()

    def execute(self, group, approval):
        require(approval["manifest_sha256"] == self.sha and (approval["scope"] == "all" or group == "first"),
                "Session approval does not cover this pair")
        self.stage("cpu-ready")
        self.advance_preparation("PLAN_VERIFIED", "READY")
        plans = self.stage("verified-plans")["plans"]
        require(self.selection(group), "This queue has no jobs in the selected pair")
        require(all(self.state(job) not in ("PENDING", "INPUT_VERIFIED", "PLAN_VERIFIED")
                    for job in self.jobs), "CPU readiness is incomplete")
        if group == "second":
            checks = {}
            for job in self.selection("first"):
                result = self.validate_completed(job, plans[job["id"]], self.held(job["arm"]))
                require(result is not None, "Verify both first-pair results before starting the second pair")
                self.complete(job, result)
                checks[job["id"]] = metadata_digest(result)
            if checks:
                require(self.stage("first-pair-verified")["result_identities"] == checks,
                        "First-pair result verification differs")
        for job in self.selection(group):
            plan = plans[job["id"]]
            self.verify_plan(job, plan)
            if self.state(job) == "RUNNING":
                self.record(job, "INTERRUPTED", reason="Prior worker no longer holds the batch lock")
            attempt = uuid4().hex
            try:
                held = self.held(job["arm"])
                result = self.validate_completed(job, plan, held)
                if result is not None:
                    self.complete(job, result)
                    print(f"{job['id']} already verified", flush=True)
                    continue
                out = frozen.safe(self.root, job["output"])
                _, generations = frozen.audit(out, job["arm"], plan, self.data, inspect_resume=True)
                capacity = self.storage(self.selection(approval["scope"]), {job["id"]: generations})
                require(capacity["free_bytes"] >= capacity["additional_bytes_reserved"], "Insufficient checkpoint storage")
                log_path = frozen.safe(self.dir, f"logs/{job['id']}/{attempt}.log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self.record(job, "RUNNING", attempt=attempt, log=log_path.relative_to(self.dir).as_posix())
                print(f"Starting {job['id']}, log {log_path}", flush=True)
                with log_path.open("x", encoding="utf-8", buffering=1) as log:
                    print(f"Attempt {attempt} for {job['id']}", file=log, flush=True)
                    try:
                        self.fit(job, plan, approval, log)
                    except (Exception, KeyboardInterrupt):
                        traceback.print_exc(file=log)
                        raise
                result = self.validate_completed(job, plan, held)
                require(result is not None, "Trainer exited without a verifiable final result")
                self.complete(job, result)
            except KeyboardInterrupt:
                if self.state(job) == "RUNNING":
                    self.record(job, "INTERRUPTED", attempt=attempt)
                raise
            except Exception as exc:
                self.record(job, "FAILED", attempt=attempt, reason=str(exc))
                raise
        return {job["id"]: self.state(job) for job in self.selection(group)}

    def verify_finished(self, group):
        self.stage("cpu-ready")
        self.advance_preparation("PLAN_VERIFIED", "READY")
        plans = self.stage("verified-plans")["plans"]
        results = {}
        for job in self.selection(group):
            if self.state(job) == "RUNNING":
                self.record(job, "INTERRUPTED", reason="Prior worker no longer holds the batch lock")
            plan = plans[job["id"]]
            self.verify_plan(job, plan)
            result = self.validate_completed(job, plan, self.held(job["arm"]))
            if result is not None:
                self.complete(job, result)
                results[job["id"]] = result
        return results

    def result_check(self, group):
        results = self.verify_finished(group)
        if group == "first":
            require(len(results) == len(self.selection(group)), "Both first-pair results must verify")
        elif group == "all":
            require(len(results) == len(self.jobs), "All planned results must verify")
        data = {"manifest_sha256": self.sha, "scope": group,
                "result_identities": {job: metadata_digest(result) for job, result in results.items()},
                "checkpoints": {job: result["checkpoint"] for job, result in results.items()}}
        if group == "first":
            self.stage("first-pair-verified", {"result_identities": data["result_identities"]})
        self.save_named("result-checks", data)
        return data

    def comparisons(self, results):
        reports = {}
        for mode in MODES:
            jobs = [j for j in self.jobs if j["mode"] == mode]
            if {job["arm"] for job in jobs} != set(frozen.ARMS) or any(j["id"] not in results for j in jobs):
                continue
            refs = self.historical(mode, self.held(jobs[0]["arm"]))
            members = {**refs, **{j["arm"]: results[j["id"]] for j in jobs}}
            metrics = {}
            for arm, result in members.items():
                job = next((j for j in jobs if j["arm"] == arm), None)
                evaluation = result["evaluation"]
                metrics[arm] = {"bpb": evaluation["bpb"], "development_targets": evaluation["predicted_tokens"],
                                "development_documents": len(evaluation["per_document"]),
                                "canonical_development_bytes": evaluation["canonical_bytes"],
                                "canonical_bytes_per_prediction": evaluation["canonical_bytes"] / evaluation["predicted_tokens"],
                                "exposure": result["identity"]["forecast"], "checkpoint": result["checkpoint"],
                                "experiment_identity": metadata_digest(result["identity"]),
                                "implementation": result["identity"]["implementation"],
                                "checkpoint_verified_this_session": job is not None,
                                "telemetry": frozen.telemetry(frozen.safe(self.root, job["output"])) if job else None,
                                "runtime_identity": result["identity"]["runtime"]}
            documents = []
            for rows in zip(*(r["evaluation"]["per_document"] for r in members.values()), strict=True):
                scores = {arm: row["bpb"] for arm, row in zip(members, rows, strict=True)}
                documents.append({"id": rows[0]["id"], "canonical_sha256": rows[0]["canonical_sha256"],
                                  "canonical_bytes": rows[0]["canonical_bytes"], "bpb": scores,
                                  "T8_minus_T5_bpb": scores["T8"] - scores["T5"]})
            reports[mode] = {"metrics": metrics, "paired": frozen.pairs(members), "documents": documents,
                             "bootstrap_draws": 10000, "bootstrap_seed": frozen.BOOTSTRAP_SEED}
        return reports

    def analyze(self):
        reports = self.comparisons(self.verify_finished("all"))
        require(reports, "No complete matched pair is available")
        orthography = {}
        for arm in frozen.ARMS:
            name = self.data["arms"][arm]["fit"] + "/validation.json"
            path = frozen.safe(self.runs, name)
            frozen.verified_file(path, self.data["files"][name])
            metrics = frozen.read(path)["validation"]
            orthography[arm] = {key: metrics[key] for key in (
                "docs", "eligible_transitions", "akshara_breaks", "akshara_reference_sites",
                "grapheme_breaks", "grapheme_reference_sites", "bytes_per_token", "fertility")}
        report = {"manifest_sha256": self.sha, "protocols": reports, "tokenizer_validation": orthography,
                  "operational_commit": OPERATIONAL, "corpus_manifest_sha256": self.data["corpus_manifest_sha256"],
                  "development_stream_sha256": self.data["validation"]["canonical_stream_sha256"],
                  "scope": "Seed 17 and paired development documents. Independent held-out evaluation is pending."}
        self.save_named("development", report)
        return report

    def save_named(self, stem, report):
        return write_receipt(frozen.safe(self.dir, f"{stem}/{metadata_digest(report)}.json"), report)

    def backup(self, destination, group, enabled, kind="all"):
        from experiments.mom_bpe.document_proxy import GENERATION
        from experiments.mom_bpe.dgx_handoff import file_digest
        from mltok.lib.artifact_paths import unlinked_path

        require(enabled, "Backup requires explicit authorization for this destination")
        destination = unlinked_path(destination)
        require(destination.is_dir() and not destination.is_relative_to(self.root)
                and not self.root.is_relative_to(destination), "Use a separate approved backup mount")
        destination = frozen.safe(destination, "mom-batch-" + self.sha)
        results = self.verify_finished(group)
        require(results, "No verified completed jobs to back up")
        # Preserve all batch evidence and per-generation files, including investigation artifacts.
        plans = self.stage("verified-plans")["plans"]
        self.stage("training-configuration", {"operational_commit": OPERATIONAL,
                    "jobs": {job: {"config": plan["config"], "config_sha256": plan.get("config_sha256")}
                             for job, plan in plans.items()}})
        files = {self.stage_path(name): category for name, category in
                 (("batch-manifest", "manifests"), ("verified-inputs", "receipts"), ("verified-plans", "receipts"),
                  ("cpu-ready", "receipts"), ("training-configuration", "configuration"))}
        for name in ("verified-source", "verified-environment", "restored-inputs", "first-pair-verified"):
            if self.stage_path(name).is_file():
                files[self.stage_path(name)] = "receipts"
        for leaf in ("state", "logs", "development", "session-reports", "inspections", "result-checks", "restorations"):
            category = {"logs": "logs", "development": "analysis"}.get(leaf, "receipts")
            files.update({p: category for p in frozen.safe(self.dir, leaf).rglob("*") if p.is_file()})
        for job in self.selection(group):
            if job["id"] not in results:
                continue
            out = frozen.safe(self.root, job["output"])
            files[frozen.safe(out, "identity.json")] = "identities"
            files[frozen.safe(out, "result.json")] = "results"
            files.update({p: "checkpoints" for p in out.iterdir() if p.is_file() and GENERATION.fullmatch(p.name)})
            files.update({p: "telemetry" for p in frozen.safe(out, "resource-telemetry").glob("*/*.json") if p.is_file()})
        saved = {}
        for path, category in sorted(files.items()):
            if kind not in ("all", category):
                continue
            name = path.relative_to(self.root).as_posix()
            size, digest = file_digest(frozen.safe(self.root, name))
            saved[name] = {"size": size, "sha256": digest, "kind": category}
        require(saved, f"No verified artifacts of kind {kind}")
        pending_bytes = sum(ref["size"] for name, ref in saved.items() if not frozen.safe(destination, name).exists())
        require(shutil.disk_usage(destination.parent).free >= pending_bytes, "Insufficient backup destination capacity")
        for name, ref in saved.items():
            src = frozen.safe(self.root, name)
            expected = (ref["size"], ref["sha256"])
            copy_verified(src, frozen.safe(destination, name), expected)
        # Independently reopen every destination member before publishing the revision receipt.
        for name, ref in saved.items():
            frozen.verified_file(frozen.safe(destination, name), ref)
        report = {"manifest_sha256": self.sha, "jobs": sorted(results), "kind": kind,
                  "result_identities": {job: metadata_digest(result) for job, result in results.items()},
                  "destination": str(destination), "verified_members": saved}
        name = metadata_digest(report) + ".json"
        remote = frozen.safe(destination, "backup-receipts/" + name)
        write_receipt(remote, report)
        require(read_receipt(remote) == report, "Destination backup receipt differs")
        digest = self.save_named("backup-receipts", report)
        local = frozen.safe(self.dir, "backup-receipts/" + name)
        require(file_digest(local) == file_digest(remote), "Backup receipts differ")
        for job in self.jobs:
            if kind == "all" and job["id"] in results and self.state(job) != "BACKED_UP":
                self.record(job, "BACKED_UP", receipt=digest)
        return {**report, "backup_revision_sha256": digest}

    def report(self):
        from experiments.mom_bpe.dgx_handoff import file_digest

        verified, issues = {}, {}
        plans = self.stage("verified-plans")["plans"] if self.stage_path("verified-plans").is_file() else {}
        for job in self.jobs:
            if self.state(job) == "RUNNING":
                self.record(job, "INTERRUPTED", reason="Prior worker no longer holds the batch lock")
            key = job["id"]
            if key not in plans:
                continue
            try:
                self.verify_plan(job, plans[key])
                result = self.validate_completed(job, plans[key], self.held(job["arm"]))
                if result is not None:
                    self.complete(job, result)
                    verified[key] = result
            except (ValueError, OSError, RuntimeError, KeyError) as exc:
                issues[key] = str(exc)
        outputs = {}
        for job in self.jobs:
            key = job["id"]
            if key in verified:
                result = verified[key]
                outputs[key] = {"bpb": result["evaluation"]["bpb"], "checkpoint": result["checkpoint"],
                                "result_identity": metadata_digest(result),
                                "telemetry": frozen.telemetry(frozen.safe(self.root, job["output"]))}
        backups = {}
        for path in sorted((self.dir / "backup-receipts").glob("*.json")):
            receipt = read_receipt(path)
            try:
                remote = frozen.safe(Path(receipt["destination"]), "backup-receipts/" + path.name)
                require(file_digest(path) == file_digest(remote), "Destination backup receipt differs")
            except (OSError, ValueError) as exc:
                backups[path.stem] = {"status": "receipt_unavailable", "reason": str(exc)}
                continue
            backups[path.stem] = {"status": "verified_at_backup", "jobs": receipt["jobs"],
                                  "kind": receipt.get("kind", "all"), "destination": receipt["destination"]}
        backed_up = {job for backup in backups.values()
                     if backup["status"] == "verified_at_backup" and backup["kind"] == "all" for job in backup["jobs"]}
        disk = shutil.disk_usage(self.dir)
        comparisons = {}
        if verified:
            try:
                comparisons = self.comparisons(verified)
            except (ValueError, OSError, KeyError) as exc:
                issues["comparisons"] = str(exc)
        for protocol in comparisons.values():
            protocol.pop("documents")  # Full rows live in the scientific report.
        data = {"manifest_sha256": self.sha, "operational_commit": OPERATIONAL,
                "planned": len(self.jobs), "verified_completed": len(verified),
                "pending_or_failed": len(self.jobs) - len(verified), "verification_issues": issues,
                "jobs": {j["id"]: self.state(j) for j in self.jobs}, "results": outputs,
                "comparisons": comparisons, "scratch": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
                "gpu_processes": self.active_user_gpu_processes(),
                "analysis_receipts": sorted(p.name for p in (self.dir / "development").glob("*.json")),
                "backups": backups,
                "cleanup": {"authorized": False, "missing_full_backup_receipt": sorted(set(outputs) - backed_up),
                            "review": "Reverify backup coverage and confirm process ownership before seeking removal approval."},
                "scientific_scope": "Seed-17 development evaluation. Independent held-out and multi-seed T5/T8 evidence remain open."}
        self.save_named("session-reports", data)
        return data

    @staticmethod
    def active_user_gpu_processes():
        if platform.system() != "Linux":
            return {"status": "unavailable_outside_dgx"}
        from experiments.mom_bpe.dgx_resources import GPU_UUID, _query

        try:
            rows = _query("--query-compute-apps=gpu_uuid,pid", "GPU compute processes")
            processes = []
            unknown = []
            for row in rows:
                if row == ("No running processes found",):
                    continue
                require(len(row) == 2 and GPU_UUID.fullmatch(row[0]) and row[1].isdigit(), "GPU process listing is malformed")
                uuid, pid = row
                try:
                    owner = Path("/proc", pid).stat().st_uid
                except OSError:
                    unknown.append(int(pid))
                    continue
                if owner == os.getuid():
                    processes.append({"pid": int(pid), "uuid": uuid})
            return {"status": "observed", "owned": processes, "unresolved_owner_pids": unknown}
        except ValueError as exc:
            return {"status": "unavailable", "reason": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("source", "environment", "restore", "verify", "plan", "ready", "inspect", "smoke",
                                               "first", "second", "results", "analyze", "backup", "report"))
    for path in ("root", "source", "archive", "destination", "manifest"):
        parser.add_argument("--" + path, type=Path, required=path in ("root", "source"))
    parser.add_argument("--manifest-sha")
    parser.add_argument("--session", default="session-1")
    parser.add_argument("--scope", choices=("first", "second", "all"), default="first")
    parser.add_argument("--backup-kind", choices=("all", *BACKUP_KINDS), default="all")
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--allocated", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--uuid", default="")
    args = parser.parse_args()
    require(re.fullmatch("[A-Za-z0-9_-]{1,64}", args.session), "Invalid DGX session label")
    require(bool(args.manifest) == bool(args.manifest_sha), "Specify both a reviewed queue manifest and its SHA-256")
    require(args.manifest_sha is None or re.fullmatch("[0-9a-f]{64}", args.manifest_sha), "Invalid queue SHA-256")
    source = args.source.resolve()
    verify_source(source)
    sys.path.insert(0, str(source))
    from mltok.lib.artifact_paths import unlinked_path
    from mltok.snapshots import _lock
    from experiments.mom_bpe.dgx_readiness import inspect_devices

    here = Path(__file__).resolve().parent
    manifest_sha = args.manifest_sha or BATCH_SHA
    directory = BATCH_DIR if manifest_sha == BATCH_SHA else "runs/dp/mom-batch-" + manifest_sha
    default = here / "t5_t8_batch_v1.json"
    manifest = load_pinned(args.manifest or default, manifest_sha)
    require(manifest["schema"] == "mom-dgx-batch-v1" and manifest["operational_commit"] == OPERATIONAL
            and manifest["input_manifest_sha256"] == frozen.MANIFEST_SHA
            and manifest["queue"], "Queue is incompatible with the frozen T5/T8 adapter")
    if manifest_sha != BATCH_SHA:
        reviewed = load_pinned(default, BATCH_SHA)["queue"]
        for job in manifest["queue"]:
            require(job in reviewed, "Custom queues must reuse reviewed jobs and their original output identities")
    require(hashlib.sha256((here / "t5_t8_dgx.py").read_bytes()).hexdigest() == manifest["bootstrap_helper_sha256"],
            "Frozen input helper differs")
    inputs = load_pinned(here / "t5_t8_inputs_v1.json", manifest["input_manifest_sha256"])
    root = unlinked_path(args.root)
    require(root.is_dir() and not source.is_relative_to(root / directory), "Use a dedicated source and private root")
    if os.name == "posix":
        require(root.stat().st_uid == os.getuid() and not root.stat().st_mode & 0o077, "Expected user-owned private root")
    batch = Batch(root, manifest, inputs, manifest_sha=manifest_sha, directory=directory)
    with _lock(batch.dir / ".controller.lock"):
        batch.stage("batch-manifest", {"manifest": manifest})
        if args.action in ("first", "second"):
            require(args.enable, "GPU execution is disabled")
            approval = batch.approval(args.session)
            require(batch.stage("verified-environment")["runtime"] == verify_environment(), "DGX environment changed")
            from experiments.mom_bpe import dgx_resources as gpu
            # Resolve and reject prohibited physical indices before applying a process-local mask.
            gpu.observe_gpu(approval["uuid"])
            os.environ["CUDA_VISIBLE_DEVICES"] = approval["uuid"]
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            # Also coordinate with the original fixed-source notebook on both preserved output paths.
            with _lock(root / FIRST_DIR / ".ablation.lock"):
                result = batch.execute(args.action, approval)
        elif args.action == "smoke":
            require(args.scope in ("first", "all"), "Approve the first pair or the full queue")
            require(args.enable and args.allocated, "GPU smoke requires explicit authorization and institutional allocation")
            require(batch.stage("verified-environment")["runtime"] == verify_environment(), "DGX environment changed")
            result = batch.smoke(args.session, args.scope, args.uuid, args.shared, args.allocated, args.enable)
        elif args.action == "source":
            result = batch.stage("verified-source", {"commit": OPERATIONAL})
        elif args.action == "environment":
            result = batch.stage("verified-environment", {"runtime": verify_environment()})
        elif args.action == "restore":
            result = batch.restore(args.archive, args.enable)
        elif args.action == "ready":
            result = batch.ready(args.scope)
        elif args.action == "inspect":
            result = {"devices": inspect_devices(), "storage": batch.storage(batch.jobs)}
            batch.save_named("inspections", {"manifest_sha256": batch.sha, **result})
        elif args.action == "results":
            result = batch.result_check(args.scope)
        elif args.action == "backup":
            require(args.destination is not None, "Supply the approved backup directory")
            result = batch.backup(args.destination, args.scope, args.enable, args.backup_kind)
        else:
            actions = {"verify": batch.verify_inputs, "plan": batch.plan,
                       "analyze": batch.analyze, "report": batch.report}
            result = actions[args.action]()
    print(json.dumps(result, indent=2), flush=True)


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    main()
