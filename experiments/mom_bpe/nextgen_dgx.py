"""Restartable DGX batch controller and 7-arm analyzer for Seed-17 next-gen evaluation."""

import argparse
import contextlib
import hashlib
import itertools
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
from tempfile import TemporaryDirectory
from time import time_ns
from types import SimpleNamespace
from uuid import uuid4

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe import dgx_resources as gpu
from experiments.mom_bpe import nextgen_pack as pack
from experiments.mom_bpe import t5_t8_dgx as frozen
from experiments.mom_bpe.dgx_handoff import digest_stream, file_digest
from mltok.lib.artifact_paths import unlinked_path
from mltok.lib.atomic import atomic_writer
from mltok.lib.bpe_progress import canonical
from mltok.lib.canonical_bpb import summarize_bpb

OPERATIONAL = "9b872ffb5a66bdf4d23df58acebbba7e788915d8"
REPOSITORY = "https://github.com/Akshai0007/Thesi_Tokenizer_Pipeline.git"
DEFAULT_INPUTS_REPO = "Akzhai/mom-bpe-nextgen-inputs-private"
DEFAULT_RESULTS_REPO = "Akzhai/mom-bpe-dgx-results-private"
REFERENCE_REVISION = "60e3d255c7afd86ec568e5a728f98a8c6e433219"
NEXTGEN_ARMS = pack.NEXTGEN_ARMS
BASELINE_ARMS = ("M48", "F-space", "T5", "T8")
ALL_SEVEN_ARMS = (*BASELINE_ARMS, *NEXTGEN_ARMS)
MODES = ("fixed_source_bytes", "fixed_training_tokens")
BOOTSTRAP_SEED = 20261125
BOOTSTRAP_DRAWS = 10_000
GIB = 2**30
BATCH_DIR = "runs/dp/nextgen-seed17-batch-v1"
INPUTS_DIR = "nextgen-inputs-v1"
REFERENCE_DIR = "t5-t8-seed17-reference-v1"
NEXTGEN_HUB_PREFIX = "seed17/nextgen/"
DEPLOYMENT_FILES = (
    "experiments/mom_bpe/nextgen_dgx.py",
    "experiments/mom_bpe/nextgen_pack.py",
    "notebooks/mom_bpe_nextgen_dgx_batch.ipynb",
    "tests/test_nextgen_dgx_batch.py",
)
BACKUP_KINDS = (
    "checkpoints",
    "results",
    "identities",
    "configuration",
    "telemetry",
    "manifests",
    "receipts",
    "analysis",
    "logs",
)
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
require = frozen.require


def metadata_digest(data) -> str:
    return hashlib.sha256(canonical(data)).hexdigest()


def write_receipt(path: Path, data: dict) -> str:
    digest = metadata_digest(data)
    payload = (json.dumps({**data, "sha256": digest}, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with atomic_writer(path, overwrite=False) as stream:
            stream.write(payload)
    except FileExistsError:
        require(path.read_bytes() == payload, f"Immutable receipt conflict: {path}")
    return digest


def read_receipt(path: Path) -> dict:
    data = frozen.read(path)
    expected = data.pop("sha256")
    require(metadata_digest(data) == expected, f"Receipt SHA-256 differs: {path}")
    return data


def _force_no_xet() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def hub_client(*, optional: bool = False):
    _force_no_xet()
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if optional and not token:
        return None
    require(token, "Authenticate with an approved Hugging Face credential")
    return HfApi(token=token)


def verify_hub_revision(api, repo: str, revision: str) -> str:
    require(repo and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo), "Invalid Hugging Face repository name")
    if revision == "main":
        info = api.repo_info(repo, repo_type="model", timeout=20)
        require(info.private is True and re.fullmatch(r"[0-9a-f]{40}", info.sha), "Private repository required")
        return info.sha
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "An immutable 40-hex Hugging Face revision is required")
    info = api.repo_info(repo, repo_type="model", revision=revision, timeout=20)
    require(info.private is True and info.sha == revision, "Hugging Face privacy or revision differs")
    return info.sha


def download_member(api, remote: dict, scratch: Path, expected: tuple[int, str] | None, *, attempts: int = 3, force: bool = False) -> Path:
    _force_no_xet()
    scratch.mkdir(parents=True, exist_ok=True)
    error = "UnknownDownloadError"
    for _ in range(attempts):
        try:
            path = Path(
                api.hf_hub_download(
                    remote["repo"],
                    remote["filename"],
                    repo_type="model",
                    revision=remote["revision"],
                    local_dir=scratch,
                    force_download=force,
                )
            )
            target = frozen.safe(scratch, remote["filename"])
            require(path.resolve() == target and target.is_file(), "Unexpected downloaded file path")
            if expected is None or file_digest(target) == expected:
                return target
            force = True
            error = "SHA256OrSizeMismatch"
        except Exception as exc:
            error = type(exc).__name__
            if error in {
                "RevisionNotFoundError",
                "EntryNotFoundError",
                "RemoteEntryNotFoundError",
                "RepositoryNotFoundError",
                "GatedRepoError",
            }:
                break
    raise ValueError(f"Individual artifact download failed ({error})")


def copy_verified(src: Path, dst: Path, expected: tuple[int, str]) -> None:
    require(file_digest(src) == expected, f"Backup source differs: {src}")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as stream, atomic_writer(dst, overwrite=False) as out:
            require(digest_stream(stream, out) == expected, f"Backup source changed during copy: {src}")
    require(file_digest(dst) == expected, f"Backup destination differs: {dst}")


def verify_source(source: Path) -> None:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    require(
        git("rev-parse", "HEAD") == OPERATIONAL
        and not git("status", "--porcelain", "--ignored", "--untracked-files=all"),
        "Expected the clean, dedicated operational checkout at pinned SHA",
    )


def verify_operational_parity(deploy_root: Path, source: Path) -> None:
    """Ensure shared operational proxy trainer/blocks/scorer/YAML in deploy checkout match `source`."""
    from experiments.mom_bpe.scale_data import code_digest

    shared_files = (
        "experiments/mom_bpe/document_proxy.py",
        "experiments/mom_bpe/proxy_blocks.py",
        "experiments/mom_bpe/dgx_resources.py",
        "experiments/mom_bpe/document-proxy.yaml",
        "mltok/lib/canonical_bpb.py",
        "mltok/plugins/proxy_model/gpt.py",
    )
    for rel in shared_files:
        dep_file = deploy_root / rel
        src_file = source / rel
        require(dep_file.is_file() and src_file.is_file(), f"Missing operational file: {rel}")
        require(code_digest(dep_file) == code_digest(src_file), f"Operational parity mismatch: {rel}")


def _write_standalone_receipt(path: Path, data: dict) -> dict:
    digest = metadata_digest(data)
    payload = (json.dumps({**data, "sha256": digest}, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with atomic_writer(path, overwrite=False) as stream:
            stream.write(payload)
    except FileExistsError:
        require(path.read_bytes() == payload, f"Immutable receipt conflict: {path}")
    return {**data, "sha256": digest}


def verify_deployment(source: Path, revision: str) -> dict:
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "Supply the exact published GitHub commit")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])

    require(git("rev-parse", "HEAD").decode().strip() == revision, "GitHub deployment commit mismatch")
    hashes = {}
    for name in DEPLOYMENT_FILES:
        path = frozen.safe(source, name)
        require(path.is_file(), f"Missing GitHub deployment file: {name}")
        blob = git("show", f"{revision}:{name}")
        digest = hashlib.sha256(blob).hexdigest()
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"GitHub deployment file differs: {name}")
        hashes[name] = digest
    return {"repository": REPOSITORY, "commit": revision, "file_sha256": hashes}


def verify_environment() -> dict:
    import importlib.metadata as md
    import unicodedata
    import torch

    expected = {
        "regex": "2026.9.29",
        "tokenizers": "0.23.3",
        "PyYAML": "6.0.3",
        "fasttext-wheel": "0.9.2",
        "pyarrow": "26.0.0",
        "numpy": "2.5.2",
        "torch": "2.5.1+cu121",
    }
    observed = {name: md.version(name) for name in expected}
    runtime = (platform.system(), platform.python_version(), unicodedata.unidata_version, torch.version.cuda)
    require(
        runtime == ("Linux", "3.12.0", "15.0.0", "12.1") and observed == expected,
        f"DGX environment differs: {runtime}, {observed}",
    )
    return {"python": runtime[1], "unicode": runtime[2], "cuda": runtime[3], "dependencies": observed}


def build_nextgen_queue(inputs_manifest: dict, inputs_sha: str) -> dict:
    require(inputs_manifest.get("schema") == "mom-nextgen-inputs-v1", "Unsupported next-gen inputs manifest schema")
    arms_order = tuple(inputs_manifest.get("arms_order", NEXTGEN_ARMS))
    require(set(arms_order) == set(NEXTGEN_ARMS) and len(arms_order) == len(NEXTGEN_ARMS), "Expected all 3 next-gen arms")
    jobs = []
    for mode in MODES:
        for arm in arms_order:
            arm_entry = inputs_manifest["arms"][arm]
            mode_entry = arm_entry["modes"][mode]
            jobs.append({
                "id": f"{arm}-{mode}",
                "arm": arm,
                "mode": mode,
                "seed": 17,
                "budget": mode_entry["budget"],
                "forecast": mode_entry["forecast"],
                "validation": arm_entry["validation"],
                "output": f"{BATCH_DIR}/jobs/{arm}-{mode}",
                "plan_sha256": mode_entry["plan_sha256"],
            })
    return {
        "schema": "mom-nextgen-dgx-batch-v1",
        "operational_commit": OPERATIONAL,
        "input_manifest_sha256": inputs_sha,
        "queue": jobs,
    }


def load_inputs_manifest(path: Path, expected_sha: str | None = None) -> tuple[dict, str]:
    require(path.is_file(), f"Missing next-gen inputs manifest: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None:
        require(digest == expected_sha, f"Inputs manifest SHA-256 differs: {digest} != {expected_sha}")
    manifest = json.loads(raw)
    require(manifest.get("schema") == "mom-nextgen-inputs-v1", "Invalid nextgen-inputs-manifest schema")
    return manifest, digest


def verify_nextgen_identity(identity: dict, plan: dict, arm: str, input_manifest: dict, driver_module, *, check_runtime: bool = True) -> None:
    require(
        all(identity.get(k) == v for k, v in plan.items())
        and identity["manifest_sha256"] == input_manifest["corpus_manifest_sha256"]
        and identity["tokenizer_sha256"] == input_manifest["arms"][arm]["tokenizer_sha256"]
        and identity["parameters"] == 38_799_936
        and identity["implementation"] == {
            "driver": gpu.PINNED_DRIVER,
            "blocks": driver_module.code_digest(driver_module.blocks.__code__.co_filename),
            "scorer": driver_module.code_digest(driver_module.summarize_bpb.__code__.co_filename),
        }
        and identity["model_plugin"] == driver_module.registry.fingerprint("proxy_model", plan["config"]["model"]),
        f"Next-gen run identity conflict: {arm}",
    )
    if check_runtime:
        runtime = identity["runtime"]
        device = runtime.get("gpu")
        require(
            device
            and device["physical_index"] != 7
            and "A100" in device["type"]
            and gpu.GPU_UUID.fullmatch(device["uuid"])
            and (
                runtime["system"],
                runtime["python"],
                runtime["torch"],
                runtime["numpy"],
                device["cuda"],
                runtime["threads"],
                runtime["precision"],
                runtime["cublas_workspace"],
            ) == ("Linux", "3.12.0", "2.5.1+cu121", "2.5.2", "12.1", 1, "bf16_autocast", ":4096:8"),
            f"Unrecognized training runtime: {arm}",
        )


def verify_nextgen_result(result: dict, held: list[dict], *, check_targets: bool, expected_docs: int = 186) -> list[dict]:
    evaluation = result["evaluation"]
    rows = evaluation["per_document"]
    require(
        frozen.doc_keys(rows) == frozen.doc_keys(held)
        and len({r["id"] for r in rows}) == expected_docs,
        "Development document order, identity or count differs",
    )
    for doc, ref_doc in zip(rows, held, strict=True):
        targets = doc["predicted_tokens"]
        require(not check_targets or targets == ref_doc["length"] + 1, "Validation target count differs")
        expected = summarize_bpb([doc["nll_nats"]], doc["canonical_bytes"], targets)
        require(all(doc[k] == v for k, v in expected.items()), "Per-document BPB differs")
    totals = summarize_bpb(
        (d["nll_nats"] for d in rows),
        sum(d["canonical_bytes"] for d in held),
        sum(d["predicted_tokens"] for d in rows),
    )
    require(all(evaluation[k] == v for k, v in totals.items()), "Aggregate BPB differs")
    return rows


def paired_bootstrap_all(results: dict[str, dict], *, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Compute exact paired 10,000-draw document bootstrap across all arms present in `results`."""
    first = next(iter(results.values()))["evaluation"]["per_document"]
    keys = frozen.doc_keys(first)
    sizes = np.array([d["canonical_bytes"] for d in first], dtype=np.float64)
    index = np.random.default_rng(seed).integers(len(first), size=(draws, len(first)))
    denom = sizes[index].sum(axis=1) * math.log(2)
    losses = {}
    for arm, result in results.items():
        rows = result["evaluation"]["per_document"]
        require(frozen.doc_keys(rows) == keys, f"Paired development-document alignment differs for {arm}")
        losses[arm] = np.array([d["nll_nats"] for d in rows], dtype=np.float64)
    values = {}
    for a, b in itertools.combinations(losses, 2):
        delta = losses[b] - losses[a]
        values[f"{b} minus {a}"] = {
            "delta_bpb": float(delta.sum() / (sizes.sum() * math.log(2))),
            "document_bootstrap_95pct": np.quantile(delta[index].sum(axis=1) / denom, [0.025, 0.975]).tolist(),
            "documents_favoring": {a: int((delta > 0).sum()), b: int((delta < 0).sum())},
            "tied_documents": int((delta == 0).sum()),
        }
    return values


def telemetry(out: Path) -> dict:
    attempts = []
    for attempt in sorted(unlinked_path(out / "resource-telemetry").glob("*")):
        start = frozen.safe(attempt, "start.json")
        end = frozen.safe(attempt, "end.json")
        updates = sorted(attempt.glob("update-*.json"))
        tail = frozen.read(end) if end.exists() else frozen.read(frozen.safe(attempt, updates[-1].name)) if updates else {}
        attempts.append({
            "attempt": attempt.name,
            "closed": end.exists(),
            "initial": frozen.read(start) if start.exists() else None,
            "last_measurement": tail,
        })
    samples = [a["last_measurement"] for a in attempts]
    return {
        "attempts": attempts,
        "measured_training_seconds": sum(s.get("training_seconds", 0) for s in samples) if attempts else None,
        **{
            field: max((s[field] for s in samples if s.get(field) is not None), default=None)
            for field in ("peak_allocated_bytes", "peak_reserved_bytes")
        },
    }


def _baseline_candidate_dirs(root: Path, arm: str, mode: str, reference_dir: Path | None = None) -> list[Path]:
    dirs = []
    if reference_dir is not None:
        dirs.append(unlinked_path(reference_dir) / f"seed17/{arm}/{mode}")
        dirs.append(unlinked_path(reference_dir) / f"{arm}/{mode}")
    dirs.extend([
        unlinked_path(root / REFERENCE_DIR / f"seed17/{arm}/{mode}"),
        unlinked_path(root / f"runs/dp/seed17/{arm}/{mode}"),
    ])
    if arm in ("T5", "T8"):
        if mode == MODES[0]:
            dirs.append(unlinked_path(root / f"runs/dp/t5-t8-seed17-fixed-source-v1/{arm}"))
        else:
            dirs.append(unlinked_path(root / f"runs/dp/t5-t8-seed17-batch-v1/jobs/{arm}-{mode}"))
    return dirs


class NextGenBatch:
    def __init__(
        self,
        root: Path,
        queue_manifest: dict,
        input_manifest: dict,
        *,
        manifest_sha: str,
        directory: str = BATCH_DIR,
        inputs_dir: str = INPUTS_DIR,
        reference_dir: Path | None = None,
        expected_docs: int = 186,
        require_full_vocab: bool = True,
        check_runtime: bool = True,
        enforce_reference_pins: bool = True,
    ):
        self.root = unlinked_path(root)
        self.dir = frozen.safe(self.root, directory)
        self.inputs_dir = frozen.safe(self.root, inputs_dir)
        self.reference_dir = unlinked_path(reference_dir) if reference_dir is not None else frozen.safe(self.root, REFERENCE_DIR)
        self.manifest = queue_manifest
        self.data = input_manifest
        self.sha = manifest_sha
        self.jobs = tuple(queue_manifest["queue"])
        self.expected_docs = expected_docs
        self.require_full_vocab = require_full_vocab
        self.check_runtime = check_runtime
        self.enforce_reference_pins = enforce_reference_pins
        for field in ("id", "output"):
            require(len({job[field] for job in self.jobs}) == len(self.jobs), f"Duplicate queue {field}")

    def selection(self, group: str) -> tuple[dict, ...]:
        if group == "all":
            return self.jobs
        return tuple(job for job in self.jobs if job["mode"] == MODES[("first", "second").index(group)])

    def stage_path(self, name: str) -> Path:
        return frozen.safe(self.dir, name + ".json")

    def stage(self, name: str, payload: dict | None = None) -> dict:
        path = self.stage_path(name)
        if payload is not None:
            write_receipt(path, {"manifest_sha256": self.sha, **payload})
        require(path.is_file(), f"Missing {name} receipt")
        data = read_receipt(path)
        require(data["manifest_sha256"] == self.sha, f"Receipt manifest conflict: {name}")
        return data

    def save_named(self, stem: str, report: dict) -> str:
        return write_receipt(frozen.safe(self.dir, f"{stem}/{metadata_digest(report)}.json"), report)

    def events(self, job: dict) -> list[Path]:
        directory = frozen.safe(self.dir, "state/" + job["id"])
        return sorted(directory.glob("[0-9]*.json"))

    def state(self, job: dict) -> str:
        events = self.events(job)
        if not events:
            return "PENDING"
        return self.event(job, events[-1])["state"]

    def event(self, job: dict, path: Path) -> dict:
        record = read_receipt(frozen.safe(path.parent, path.name))
        require(record["manifest_sha256"] == self.sha and record["job"] == job["id"], "Job state conflict")
        return record

    def record(self, job: dict, state: str, **detail) -> None:
        prior = self.state(job)
        require(state in TRANSITIONS[prior], f"Invalid transition from {prior} to {state}")
        path = frozen.safe(self.dir, f"state/{job['id']}/{len(self.events(job)) + 1:08d}.json")
        write_receipt(path, {"manifest_sha256": self.sha, "job": job["id"], "state": state, **detail})

    def advance_preparation(self, old: str, new: str) -> None:
        for job in self.jobs:
            if self.state(job) == old:
                self.record(job, new)

    def cache(self, arm: str) -> Path:
        return frozen.safe(self.inputs_dir, self.data["arms"][arm]["cache"])

    def held(self, arm: str) -> list[dict]:
        from experiments.mom_bpe.proxy_blocks import EncodedSplit, receipt

        cache = self.cache(arm)
        self.verify_cache_pin(arm)
        data, _ = receipt(cache / "cache.json")
        return EncodedSplit(cache, "validation", data["splits"]["validation"]).docs

    def verify_cache_pin(self, arm: str) -> None:
        name = self.data["arms"][arm]["cache"] + "/cache.json"
        frozen.verified_file(frozen.safe(self.inputs_dir, name), self.data["files"][name])

    def verify_inputs(self) -> dict:
        from experiments.mom_bpe import document_proxy as driver
        from experiments.mom_bpe import proxy_blocks
        from mltok import registry

        for name, entry in self.data["files"].items():
            frozen.verified_file(frozen.safe(self.inputs_dir, name), entry)

        registry.load_plugins()
        require(not registry.load_errors(), "Plugin import errors")
        normalizer = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
        plans, docs = {}, {}

        for arm in NEXTGEN_ARMS:
            arm_entry = self.data["arms"][arm]
            fit_dir = frozen.safe(self.inputs_dir, arm_entry["fit"])
            _, model_file, model_sha, uniform_prefix = pack.load_nextgen_backend(
                fit_dir, arm, normalizer, require_full_vocab=self.require_full_vocab
            )
            require(
                model_file == arm_entry["model_file"]
                and model_sha == arm_entry["tokenizer_sha256"]
                and uniform_prefix == arm_entry["uniform_prefix"],
                f"Model identity differs for {arm}",
            )
            folder = self.cache(arm)
            cache, cache_digest = proxy_blocks.receipt(folder / "cache.json")
            identity = cache["identity"]
            require(
                cache_digest == arm_entry["cache_receipt_sha256"]
                and identity["tokenizer_sha256"] == model_sha
                and identity["arm"] == arm
                and identity["manifest_sha256"] == self.data["corpus_manifest_sha256"],
                f"Cache or corpus identity differs for {arm}",
            )
            with pack.with_nextgen_arms(driver, proxy_blocks):
                for mode in MODES:
                    mode_entry = arm_entry["modes"][mode]
                    saved, saved_sha = proxy_blocks.receipt(frozen.safe(self.inputs_dir, mode_entry["ledger"]))
                    observed = driver.fit_cache(SimpleNamespace(
                        command="plan",
                        cache=folder,
                        mode=mode,
                        budget=mode_entry["budget"],
                        out=None,
                        seed=None,
                        update_targets=None,
                        microbatch_blocks=None,
                        checkpoint_updates=None,
                    ))
                    require(
                        observed == saved
                        and saved_sha == mode_entry["plan_sha256"]
                        and observed["config"]["seed"] == 17
                        and observed["config_sha256"] == self.data["config_sha256"]
                        and observed["optimizer_sha256"] == self.data["optimizer_sha256"]
                        and observed["forecast"] == mode_entry["forecast"]
                        and observed["validation"] == arm_entry["validation"],
                        f"Recomputed exposure plan differs for {arm}/{mode}",
                    )
                    plans[f"{arm}-{mode}"] = observed
            docs[arm] = proxy_blocks.EncodedSplit(folder, "validation", cache["splits"]["validation"]).docs

        first_keys = frozen.doc_keys(docs[NEXTGEN_ARMS[0]])
        for arm in NEXTGEN_ARMS[1:]:
            require(frozen.doc_keys(docs[arm]) == first_keys, f"Development document ordering differs for {arm}")

        result = self.stage("verified-inputs", {"plans": plans})
        self.advance_preparation("PENDING", "INPUT_VERIFIED")
        return result

    def verify_plan(self, job: dict, plan: dict) -> None:
        require(
            metadata_digest(plan) == job["plan_sha256"]
            and plan["mode"] == job["mode"]
            and plan["arm"] == job["arm"]
            and plan["config"]["seed"] == job["seed"]
            and plan["budget"] == job["budget"]
            and plan["forecast"] == job["forecast"]
            and plan["validation"] == job["validation"],
            f"Exposure plan conflict: {job['id']}",
        )

    def plan(self) -> dict:
        verified = self.stage("verified-inputs")["plans"]
        self.advance_preparation("PENDING", "INPUT_VERIFIED")
        if not self.stage_path("verified-plans").exists():
            plans = {}
            for job in self.jobs:
                value = verified[job["id"]]
                self.verify_plan(job, value)
                plans[job["id"]] = value
            self.stage("verified-plans", {"plans": plans})
        result = self.stage("verified-plans")
        self.advance_preparation("INPUT_VERIFIED", "PLAN_VERIFIED")
        return result

    def storage(self, jobs, recovery=None) -> dict:
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
        required = (missing + 2) * (12 * self.data["parameters"] + 2**20)
        total_generations = sum(
            math.ceil(plans[job["id"]]["forecast"]["optimizer_updates"] / plans[job["id"]]["config"]["checkpoint_updates"])
            for job in jobs
        )
        if missing == total_generations and jobs:
            required = max(required, (15 if len(jobs) >= 6 else 8) * GIB)
        disk = shutil.disk_usage(self.dir)
        return {
            "free_bytes": disk.free,
            "filesystem_used_bytes": disk.used,
            "filesystem_total_bytes": disk.total,
            "additional_bytes_reserved": required,
        }

    def ready(self, group: str) -> dict:
        resource = self.storage(self.selection(group))
        require(resource["free_bytes"] >= resource["additional_bytes_reserved"], "Insufficient batch checkpoint storage")
        if not self.stage_path("cpu-ready").exists():
            self.stage("cpu-ready", {"ready": True})
        self.stage("cpu-ready")
        self.advance_preparation("PLAN_VERIFIED", "READY")
        return resource

    def approval(self, session: str) -> dict:
        path = frozen.safe(self.dir, f"sessions/{session}")
        attempts = sorted(path.glob("smoke-*"))
        require(attempts, "Run the authorized smoke for this session")
        receipt_path = frozen.safe(attempts[-1], "approved.json")
        require(receipt_path.is_file(), "Latest smoke attempt was not approved")
        return read_receipt(receipt_path)

    def smoke(self, session: str, scope: str, uuid: str, shared: bool, allocated: bool, enabled: bool) -> dict:
        from experiments.mom_bpe.dgx_readiness import preflight

        require(enabled and allocated, "GPU smoke requires explicit authorization and institutional allocation")
        self.stage("cpu-ready")
        out = frozen.safe(self.dir, f"sessions/{session}/smoke-{time_ns()}-{uuid4().hex}")
        out.mkdir(parents=True)
        resource = self.storage(self.selection(scope))
        require(resource["free_bytes"] >= resource["additional_bytes_reserved"], "Insufficient checkpoint storage for approval scope")
        result = preflight(out, uuid, shared, allocated)
        write_receipt(
            out / "approved.json",
            {"manifest_sha256": self.sha, "scope": scope, "uuid": uuid, "shared": shared, "smoke": result},
        )
        return result

    def audit_job(self, out: Path, arm: str, plan: dict, *, inspect_resume: bool = False) -> tuple[dict | None, dict]:
        from experiments.mom_bpe import document_proxy as driver

        state_path = frozen.safe(out, "identity.json")
        complete = frozen.safe(out, "result.json")
        identity = frozen.read(state_path) if state_path.is_file() else None
        if identity is not None:
            verify_nextgen_identity(identity, plan, arm, self.data, driver, check_runtime=self.check_runtime)
        checkpoints = {}
        if out.exists():
            for path in sorted(out.glob("step-*.pt"), reverse=True):
                match = driver.GENERATION.fullmatch(path.name)
                require(match and identity is not None, f"Unidentified checkpoint {path}")
                size, digest = file_digest(frozen.safe(out, path.name))
                checkpoints[path.name] = {"size": size, "sha256": digest, "valid_bytes": digest == match[2]}
        if complete.exists():
            result = read_receipt(complete)
            ref = result["checkpoint"]
            match = driver.GENERATION.fullmatch(ref["path"])
            actual = checkpoints.get(ref["path"], {})
            require(
                result["identity"] == identity
                and match
                and int(match[1]) == plan["forecast"]["optimizer_updates"]
                and actual == {"size": ref["size"], "sha256": ref["sha256"], "valid_bytes": True},
                f"Completed result or final checkpoint differs: {arm}",
            )
            return result, checkpoints
        if inspect_resume and checkpoints:
            for name, entry in checkpoints.items():
                if entry["valid_bytes"]:
                    state = driver.torch.load(frozen.safe(out, name), map_location="cpu", weights_only=True)
                    step = int(driver.GENERATION.fullmatch(name)[1])
                    require(
                        state["identity"] == identity
                        and state["step"] == step
                        and 0 < step <= plan["forecast"]["optimizer_updates"]
                        and state["exposure"]["optimizer_updates"] == step,
                        f"Checkpoint payload identity conflict: {arm}/{name}",
                    )
                    break
            else:
                raise ValueError(f"No checksum-valid generation for {arm}; preserve and recover from verified backup")
        return None, checkpoints

    def validate_completed(self, job: dict, plan: dict, held: list[dict]) -> dict | None:
        from experiments.mom_bpe import document_proxy as driver
        from experiments.mom_bpe.document_proxy import GENERATION

        out = frozen.safe(self.root, job["output"])
        file = frozen.safe(out, "result.json")
        earlier = [self.event(job, p) for p in self.events(job)]
        verified = [event for event in earlier if event["state"] == "RESULT_VERIFIED"]
        if not file.exists():
            require(
                not any(event["state"] in ("COMPLETED", "RESULT_VERIFIED", "BACKED_UP") for event in earlier),
                f"Previously completed result is missing: {job['id']}",
            )
            return None
        result = read_receipt(file)
        driver.registry.load_plugins()
        identity = frozen.read(frozen.safe(out, "identity.json"))
        verify_nextgen_identity(identity, plan, job["arm"], self.data, driver, check_runtime=self.check_runtime)
        require(
            result["identity"] == identity
            and all(p["result_identity"] == metadata_digest(result) for p in verified),
            "Final result identity differs",
        )
        ref = result["checkpoint"]
        name = ref["path"]
        match = GENERATION.fullmatch(name)
        require(match and int(match[1]) == plan["forecast"]["optimizer_updates"] and match[2] == ref["sha256"],
                "Final checkpoint reference differs")
        checkpoint = frozen.safe(out, name)
        require(file_digest(checkpoint) == (ref["size"], ref["sha256"]), "Final checkpoint is corrupt")
        if not verified:
            state = driver.torch.load(checkpoint, map_location="cpu", weights_only=True)
            require(
                state["identity"] == identity
                and state["step"] == int(match[1])
                and state["exposure"] == plan["forecast"]
                and math.isfinite(state["train_nll_nats"])
                and state["train_nll_nats"] >= 0
                and state["train_nll_nats"] == result["training_nll_nats"]
                and state["model"]
                and all(driver.torch.is_tensor(tensor) for tensor in state["model"].values())
                and state["optimizer"]["state"]
                and state["optimizer"]["param_groups"]
                and all(group["lr"] == state["scheduler"]["lr"] for group in state["optimizer"]["param_groups"])
                and {"python", "numpy", "torch", "cuda"} <= state["rng"].keys(),
                "Final checkpoint payload differs",
            )
        verify_nextgen_result(result, held, check_targets=True, expected_docs=self.expected_docs)
        return result

    def complete(self, job: dict, result: dict) -> None:
        if self.state(job) not in ("COMPLETED", "RESULT_VERIFIED", "BACKED_UP"):
            self.record(job, "COMPLETED", checkpoint=result["checkpoint"])
        if self.state(job) == "COMPLETED":
            self.record(job, "RESULT_VERIFIED", result_identity=metadata_digest(result))

    def admit(self, approval: dict):
        from experiments.mom_bpe import document_proxy as driver

        policy = gpu.GpuAdmission(5120, 10, True) if approval["shared"] else gpu.GpuAdmission()
        observed = policy.admit(approval["uuid"])
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == approval["uuid"], "GPU mask differs from the allocation")
        device = driver.gpu_preflight(
            policy.min_free_mib,
            max_gpu_util_pct=policy.max_gpu_util_pct,
            allow_shared_gpu=policy.allow_shared_gpu,
        )
        require("A100" in device["type"] and device["physical_index"] == observed["physical_index"], "Expected allocated A100")
        return policy, device

    def fit(self, job: dict, plan: dict, approval: dict, log) -> None:
        from experiments.mom_bpe import document_proxy as driver
        from experiments.mom_bpe import proxy_blocks

        out = frozen.safe(self.root, job["output"])
        self.verify_cache_pin(job["arm"])
        policy, hardware = self.admit(approval)
        identity = frozen.safe(out, "identity.json")
        if identity.exists():
            require(frozen.read(identity)["runtime"]["gpu"] == hardware, "Resume on the original allocated GPU")
        args = SimpleNamespace(
            command="run",
            cache=self.cache(job["arm"]),
            out=out,
            mode=job["mode"],
            device="cuda",
            threads=1,
            budget=None,
            seed=None,
            update_targets=None,
            microbatch_blocks=None,
            checkpoint_updates=None,
            min_free_mib=policy.min_free_mib,
            max_gpu_util_pct=policy.max_gpu_util_pct,
            allow_shared_gpu=policy.allow_shared_gpu,
        )
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), pack.with_nextgen_arms(driver, proxy_blocks):
            driver.fit_cache(args)
            driver.torch.cuda.empty_cache()

    def execute(self, group: str, approval: dict) -> dict:
        require(
            approval["manifest_sha256"] == self.sha and (approval["scope"] == "all" or group == "first"),
            "Session approval does not cover this pair",
        )
        self.stage("cpu-ready")
        self.advance_preparation("PLAN_VERIFIED", "READY")
        plans = self.stage("verified-plans")["plans"]
        require(self.selection(group), "This queue has no jobs in the selected group")
        require(
            all(self.state(job) not in ("PENDING", "INPUT_VERIFIED", "PLAN_VERIFIED") for job in self.jobs),
            "CPU readiness is incomplete",
        )
        if group == "second":
            checks = {}
            for job in self.selection("first"):
                result = self.validate_completed(job, plans[job["id"]], self.held(job["arm"]))
                require(result is not None, "Verify all first-group results before starting fixed_training_tokens")
                self.complete(job, result)
                checks[job["id"]] = metadata_digest(result)
            require(self.stage("first-pair-verified")["result_identities"] == checks, "First-group result verification differs")

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
                _, generations = self.audit_job(out, job["arm"], plan, inspect_resume=True)
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

    def verify_finished(self, group: str) -> dict:
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

    def result_check(self, group: str) -> dict:
        results = self.verify_finished(group)
        if group == "first":
            require(len(results) == len(self.selection(group)), "All fixed_source_bytes results must verify")
        elif group == "all":
            require(len(results) == len(self.jobs), "All 6 planned results must verify")
        data = {
            "manifest_sha256": self.sha,
            "scope": group,
            "result_identities": {job: metadata_digest(result) for job, result in results.items()},
            "checkpoints": {job: result["checkpoint"] for job, result in results.items()},
        }
        if group == "first":
            self.stage("first-pair-verified", {"result_identities": data["result_identities"]})
        self.save_named("result-checks", data)
        return data

    def load_baseline_results(self, mode: str, held: list[dict], *, api=None, results_repo: str = DEFAULT_RESULTS_REPO) -> dict[str, dict]:
        from experiments.mom_bpe import document_proxy as driver
        from experiments.mom_bpe.dgx_readiness import PINS

        refs = {}
        for arm in BASELINE_ARMS:
            folder = next(
                (d for d in _baseline_candidate_dirs(self.root, arm, mode, self.reference_dir) if (d / "result.json").is_file() and (d / "identity.json").is_file()),
                None,
            )
            if folder is None and api is not None:
                revision = REFERENCE_REVISION if arm in ("M48", "F-space") else verify_hub_revision(api, results_repo, "main")
                dest = frozen.safe(self.reference_dir, f"seed17/{arm}/{mode}")
                for leaf in ("identity.json", "result.json"):
                    filename = f"seed17/{arm}/{mode}/{leaf}"
                    with TemporaryDirectory(prefix="ref-dl-", dir=self.dir) as tmp:
                        dl = download_member(api, {"repo": results_repo, "revision": revision, "filename": filename}, Path(tmp), None)
                        dest.mkdir(parents=True, exist_ok=True)
                        if not (dest / leaf).exists():
                            (dest / leaf).write_bytes(dl.read_bytes())
                folder = dest
            require(folder is not None, f"Missing baseline Seed-17 results for {arm}/{mode}")

            if self.enforce_reference_pins and arm in ("M48", "F-space"):
                for leaf, digest in zip(("identity.json", "result.json"), PINS[arm, mode], strict=True):
                    require(file_digest(frozen.safe(folder, leaf))[1] == digest, f"Historical reference differs: {arm}/{mode}/{leaf}")

            result = read_receipt(frozen.safe(folder, "result.json"))
            identity = frozen.read(frozen.safe(folder, "identity.json"))
            require(
                result["identity"] == identity
                and identity["arm"] == arm
                and identity["mode"] == mode
                and identity["config"]["seed"] == 17
                and identity["parameters"] == 38_799_936
                and identity["manifest_sha256"] == self.data["corpus_manifest_sha256"]
                and identity["optimizer_sha256"] == self.data["optimizer_sha256"]
                and identity["config_sha256"] == self.data["config_sha256"]
                and identity["implementation"] == {
                    "driver": gpu.PINNED_DRIVER,
                    "blocks": driver.code_digest(driver.blocks.__code__.co_filename),
                    "scorer": driver.code_digest(driver.summarize_bpb.__code__.co_filename),
                },
                f"Baseline identity or implementation conflict: {arm}/{mode}",
            )
            verify_nextgen_result(result, held, check_targets=False, expected_docs=self.expected_docs)
            refs[arm] = result
        return refs

    def comparisons(self, results: dict[str, dict], *, api=None, results_repo: str = DEFAULT_RESULTS_REPO) -> dict:
        reports = {}
        for mode in MODES:
            jobs = [j for j in self.jobs if j["mode"] == mode]
            if {j["arm"] for j in jobs} != set(NEXTGEN_ARMS) or any(j["id"] not in results for j in jobs):
                continue
            held = self.held(jobs[0]["arm"])
            refs = self.load_baseline_results(mode, held, api=api, results_repo=results_repo)
            members = {**refs, **{j["arm"]: results[j["id"]] for j in jobs}}
            metrics = {}
            for arm in ALL_SEVEN_ARMS:
                result = members[arm]
                job = next((j for j in jobs if j["arm"] == arm), None)
                evaluation = result["evaluation"]
                metrics[arm] = {
                    "bpb": evaluation["bpb"],
                    "development_targets": evaluation["predicted_tokens"],
                    "development_documents": len(evaluation["per_document"]),
                    "canonical_development_bytes": evaluation["canonical_bytes"],
                    "canonical_bytes_per_prediction": evaluation["canonical_bytes"] / evaluation["predicted_tokens"],
                    "exposure": result["identity"]["forecast"],
                    "checkpoint": result["checkpoint"],
                    "experiment_identity": metadata_digest(result["identity"]),
                    "implementation": result["identity"]["implementation"],
                    "checkpoint_verified_this_session": job is not None,
                    "telemetry": telemetry(frozen.safe(self.root, job["output"])) if job else None,
                    "runtime_identity": result["identity"]["runtime"],
                }
            paired = paired_bootstrap_all(members, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS)
            documents = []
            for rows in zip(*(members[arm]["evaluation"]["per_document"] for arm in ALL_SEVEN_ARMS), strict=True):
                scores = {arm: row["bpb"] for arm, row in zip(ALL_SEVEN_ARMS, rows, strict=True)}
                documents.append({
                    "id": rows[0]["id"],
                    "canonical_sha256": rows[0]["canonical_sha256"],
                    "canonical_bytes": rows[0]["canonical_bytes"],
                    "bpb": scores,
                })
            reports[mode] = {
                "arms": list(ALL_SEVEN_ARMS),
                "metrics": metrics,
                "paired": paired,
                "vs_m48": {arm: paired[f"{arm} minus M48"] for arm in ALL_SEVEN_ARMS[1:]},
                "vs_t8": {
                    arm: paired[f"{arm} minus T8"]
                    for arm in NEXTGEN_ARMS
                },
                "documents": documents,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        return reports

    def analyze(self, *, fetch_baselines: bool = False, results_repo: str = DEFAULT_RESULTS_REPO) -> dict:
        api = hub_client(optional=not fetch_baselines) if fetch_baselines else None
        results = self.verify_finished("all")
        require(len(results) == len(self.jobs), "All 6 next-gen jobs must verify before 7-arm analysis")
        reports = self.comparisons(results, api=api, results_repo=results_repo)
        require(set(reports) == set(MODES), "Both exposure protocols are required for 7-arm analysis")

        orthography = {}
        for arm in NEXTGEN_ARMS:
            summary_rel = self.data["arms"][arm]["fit"] + "/eval_summary.json"
            summary_path = frozen.safe(self.inputs_dir, summary_rel)
            if summary_path.is_file():
                summary = frozen.read(summary_path)
                orthography[arm] = {
                    "dev_targets_with_eos": summary.get("dev_targets_with_eos"),
                    "bytes_per_prediction": summary.get("bytes_per_prediction"),
                    "pure_whitespace_tokens": summary.get("pure_whitespace_tokens"),
                    "vocab_other_script": summary.get("vocab_other_script"),
                    "mom_surface": summary.get("mom_surface"),
                }

        report = {
            "schema": "mom-nextgen-7arm-analysis-v1",
            "manifest_sha256": self.sha,
            "operational_commit": OPERATIONAL,
            "corpus_manifest_sha256": self.data["corpus_manifest_sha256"],
            "development_stream_sha256": self.data["validation"]["canonical_stream_sha256"],
            "arms": list(ALL_SEVEN_ARMS),
            "protocols": reports,
            "nextgen_tokenizer_validation": orthography,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "scope": "Seed-17 7-arm development comparison across 186 paired Malayalam Wikipedia documents.",
        }
        digest = self.save_named("development", report)
        analysis_path = frozen.safe(self.dir, "analysis.json")
        payload = (json.dumps({**report, "sha256": digest}, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if analysis_path.exists():
            require(analysis_path.read_bytes() == payload, "Existing analysis.json differs")
        else:
            with atomic_writer(analysis_path, overwrite=False) as out:
                out.write(payload)
        return {**report, "analysis_path": str(analysis_path), "sha256": digest}

    def restore_individual_hub(
        self,
        *,
        inputs_repo: str = DEFAULT_INPUTS_REPO,
        inputs_revision: str = "main",
        expected_manifest_sha: str | None = None,
        restore_baselines: bool = False,
        results_repo: str = DEFAULT_RESULTS_REPO,
        api=None,
    ) -> dict:
        api = api or hub_client(optional=True)
        resolved_inputs_rev = verify_hub_revision(api, inputs_repo, inputs_revision) if api else inputs_revision
        manifest_path = self.inputs_dir / "nextgen-inputs-manifest.json"
        if not manifest_path.is_file() and api is not None:
            with TemporaryDirectory(prefix="inputs-manifest-", dir=self.dir) as tmp:
                dl = download_member(
                    api,
                    {"repo": inputs_repo, "revision": resolved_inputs_rev, "filename": "nextgen-inputs-manifest.json"},
                    Path(tmp),
                    None,
                )
                self.inputs_dir.mkdir(parents=True, exist_ok=True)
                data = dl.read_bytes()
                if expected_manifest_sha is not None:
                    require(hashlib.sha256(data).hexdigest() == expected_manifest_sha, "Downloaded inputs manifest SHA differs")
                with atomic_writer(manifest_path, overwrite=False) as out:
                    out.write(data)
        inputs_manifest, inputs_sha = load_inputs_manifest(manifest_path, expected_manifest_sha)

        restored_counts = {"reused": 0, "restored": 0, "failed": 0}
        members_status = {}
        for rel, ref in sorted(inputs_manifest["files"].items()):
            dest = frozen.safe(self.inputs_dir, rel)
            expected = (ref["size"], ref["sha256"])
            if dest.exists() and dest.is_file() and file_digest(dest) == expected:
                restored_counts["reused"] += 1
                members_status[rel] = "reused"
                continue
            if api is None:
                restored_counts["failed"] += 1
                members_status[rel] = "missing_auth"
                continue
            staging = frozen.safe(self.dir, "downloads/" + metadata_digest({"repo": inputs_repo, "rev": resolved_inputs_rev, "rel": rel}))
            dl = download_member(
                api,
                {"repo": inputs_repo, "revision": resolved_inputs_rev, "filename": rel},
                staging,
                expected,
            )
            copy_verified(dl, dest, expected)
            restored_counts["restored"] += 1
            members_status[rel] = "restored"

        baseline_status = {}
        if restore_baselines:
            require(api is not None, "Authentication required to restore Seed-17 baselines")
            results_head = verify_hub_revision(api, results_repo, "main")
            for arm in BASELINE_ARMS:
                rev = REFERENCE_REVISION if arm in ("M48", "F-space") else results_head
                for mode in MODES:
                    for leaf in ("identity.json", "result.json"):
                        rel = f"seed17/{arm}/{mode}/{leaf}"
                        dest = frozen.safe(self.reference_dir, rel)
                        if dest.is_file():
                            baseline_status[rel] = "reused"
                            continue
                        staging = frozen.safe(self.dir, "downloads/" + metadata_digest({"repo": results_repo, "rev": rev, "rel": rel}))
                        dl = download_member(api, {"repo": results_repo, "revision": rev, "filename": rel}, staging, None)
                        expected = file_digest(dl)
                        copy_verified(dl, dest, expected)
                        baseline_status[rel] = "restored"

        report = {
            "manifest_sha256": inputs_sha,
            "inputs_repo": inputs_repo,
            "inputs_revision": resolved_inputs_rev,
            "complete": restored_counts["failed"] == 0,
            "counts": restored_counts,
            "baseline_files": baseline_status,
        }
        self.save_named("individual-restorations", report)
        return report

    def backup_hub(self, repo: str, group: str, enabled: bool, checkpoints: bool, *, api=None) -> dict:
        _force_no_xet()
        require(enabled, "Private Hugging Face backup requires explicit authorization")
        require(
            repo and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo),
            "Supply the separately approved private model repository",
        )
        from huggingface_hub import CommitOperationAdd

        results = self.verify_finished(group)
        require(results, "No verified completed jobs to back up")
        files = {}
        for job in self.selection(group):
            result = results.get(job["id"])
            if result is None:
                continue
            out = frozen.safe(self.root, job["output"])
            names = ["result.json", "identity.json"]
            if checkpoints:
                names.append(result["checkpoint"]["path"])
            for name in names:
                files[f"{NEXTGEN_HUB_PREFIX}{job['arm']}/{job['mode']}/{name}"] = frozen.safe(out, name)
        analysis_path = frozen.safe(self.dir, "analysis.json")
        if analysis_path.is_file():
            files[f"{NEXTGEN_HUB_PREFIX}analysis.json"] = analysis_path

        for remote_name in files:
            require(
                remote_name.startswith(NEXTGEN_HUB_PREFIX)
                and not any(remote_name.startswith(f"seed17/{b}/") for b in BASELINE_ARMS),
                f"Refusing to write outside {NEXTGEN_HUB_PREFIX}: {remote_name}",
            )

        expected = {name: file_digest(path) for name, path in files.items()}
        require(
            shutil.disk_usage(self.dir).free > max(size for size, _ in expected.values()) + 32 * 2**20,
            "Insufficient space for independent backup verification",
        )
        api = api or hub_client()
        info = api.repo_info(repo, repo_type="model", timeout=20)
        require(info.private is True, "Backup repository must be private")
        revision = info.sha
        verify_hub_revision(api, repo, revision)
        existing = set(api.list_repo_files(repo, repo_type="model", revision=revision))

        def compare(name, pin):
            with TemporaryDirectory(prefix="hub-verify-", dir=self.dir) as tmp:
                remote = {"repo": repo, "revision": pin, "filename": name}
                download_member(api, remote, Path(tmp), expected[name], force=True)

        for name in set(files) & existing:
            compare(name, revision)
        additions = [
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(files[name]))
            for name in sorted(set(files) - existing)
        ]
        if additions:
            revision = api.create_commit(
                repo,
                repo_type="model",
                parent_commit=revision,
                operations=additions,
                commit_message="Back up verified Seed-17 next-gen evaluation outputs",
            ).oid
        verify_hub_revision(api, repo, revision)
        for name in files:
            compare(name, revision)
        report = {
            "manifest_sha256": self.sha,
            "repository": repo,
            "revision": revision,
            "hub_prefix": NEXTGEN_HUB_PREFIX,
            "jobs": sorted(results),
            "files": {name: {"size": size, "sha256": digest} for name, (size, digest) in expected.items()},
        }
        self.save_named("hub-backup-receipts", report)
        return report

    def report(self) -> dict:
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
        outputs = {
            key: {
                "bpb": verified[key]["evaluation"]["bpb"],
                "checkpoint": verified[key]["checkpoint"],
                "result_identity": metadata_digest(verified[key]),
                "telemetry": telemetry(frozen.safe(self.root, job["output"])),
            }
            for job in self.jobs
            if (key := job["id"]) in verified
        }
        disk = shutil.disk_usage(self.dir)
        data = {
            "manifest_sha256": self.sha,
            "operational_commit": OPERATIONAL,
            "planned": len(self.jobs),
            "verified_completed": len(verified),
            "pending_or_failed": len(self.jobs) - len(verified),
            "verification_issues": issues,
            "jobs": {j["id"]: self.state(j) for j in self.jobs},
            "results": outputs,
            "scratch": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
            "analysis_present": (self.dir / "analysis.json").is_file(),
            "hub_backup_receipts": sorted(p.name for p in (self.dir / "hub-backup-receipts").glob("*.json")),
        }
        self.save_named("session-reports", data)
        return data


def _resolve_inputs_manifest_path(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return unlinked_path(explicit)
    return frozen.safe(root, f"{INPUTS_DIR}/nextgen-inputs-manifest.json")


def main() -> None:
    _force_no_xet()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "source",
            "environment",
            "authenticate",
            "restore-individual",
            "verify",
            "plan",
            "ready",
            "inspect",
            "smoke",
            "first",
            "second",
            "results",
            "analyze",
            "backup-hub",
            "report",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path)
    parser.add_argument("--inputs-manifest-sha")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--deployment-revision")
    parser.add_argument("--inputs-repo", default=DEFAULT_INPUTS_REPO)
    parser.add_argument("--inputs-revision", default="main")
    parser.add_argument("--results-repo", default=DEFAULT_RESULTS_REPO)
    parser.add_argument("--hub-repo", default=DEFAULT_RESULTS_REPO)
    parser.add_argument("--hub-checkpoints", action="store_true")
    parser.add_argument("--restore-baselines", action="store_true")
    parser.add_argument("--fetch-baselines", action="store_true")
    parser.add_argument("--session", default="session-1")
    parser.add_argument("--scope", choices=("first", "second", "all"), default="first")
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--allocated", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--uuid", default="")
    args = parser.parse_args()

    require(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.session), "Invalid DGX session label")
    require(args.inputs_manifest_sha is None or re.fullmatch(r"[0-9a-f]{64}", args.inputs_manifest_sha), "Invalid manifest SHA-256")
    deploy_root = Path(__file__).resolve().parents[2]
    if str(deploy_root) not in sys.path:
        sys.path.insert(0, str(deploy_root))
    source = args.source.resolve()
    verify_source(source)
    verify_operational_parity(deploy_root, source)
    from experiments.mom_bpe.dgx_readiness import inspect_devices
    from mltok.snapshots import _lock

    root = unlinked_path(args.root)
    require(root.is_dir() and not source.is_relative_to(root / BATCH_DIR), "Use a dedicated source and private root")
    if os.name == "posix":
        require(root.stat().st_uid == os.getuid() and not root.stat().st_mode & 0o077, "Expected user-owned private root")

    batch_dir = frozen.safe(root, BATCH_DIR)
    deployment = verify_deployment(deploy_root, args.deployment_revision) if args.deployment_revision else None

    if args.action == "source":
        batch_dir.mkdir(parents=True, exist_ok=True)
        with _lock(batch_dir / ".controller.lock"):
            src_receipt = _write_standalone_receipt(batch_dir / "verified-source.json", {"schema": "mom-nextgen-env-v1", "commit": OPERATIONAL})
            result = {**src_receipt}
            if deployment:
                result = {**result, "deployment": deployment}
                write_receipt(batch_dir / f"verified-deployments/{metadata_digest(deployment)}.json", deployment)
        print(json.dumps(result, indent=2), flush=True)
        return

    if args.action == "environment":
        batch_dir.mkdir(parents=True, exist_ok=True)
        with _lock(batch_dir / ".controller.lock"):
            result = _write_standalone_receipt(
                batch_dir / "verified-environment.json",
                {"schema": "mom-nextgen-env-v1", "operational_commit": OPERATIONAL, "runtime": verify_environment()},
            )
        print(json.dumps(result, indent=2), flush=True)
        return

    if args.action == "authenticate":
        api = hub_client()
        in_sha = verify_hub_revision(api, args.inputs_repo, args.inputs_revision)
        res_sha = verify_hub_revision(api, args.results_repo, REFERENCE_REVISION)
        print(json.dumps({
            "inputs_repo": args.inputs_repo,
            "inputs_revision": in_sha,
            "results_repo": args.results_repo,
            "reference_revision": res_sha,
            "xet_disabled": os.environ.get("HF_HUB_DISABLE_XET") == "1",
        }, indent=2))
        return

    if args.action == "restore-individual":
        inputs_dir = frozen.safe(root, INPUTS_DIR)
        batch_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _resolve_inputs_manifest_path(root, args.inputs_manifest)
        if not manifest_path.is_file():
            api = hub_client()
            rev = verify_hub_revision(api, args.inputs_repo, args.inputs_revision)
            with TemporaryDirectory(prefix="inputs-manifest-", dir=batch_dir) as tmp:
                dl = download_member(api, {"repo": args.inputs_repo, "revision": rev, "filename": "nextgen-inputs-manifest.json"}, Path(tmp), None)
                inputs_dir.mkdir(parents=True, exist_ok=True)
                with atomic_writer(manifest_path, overwrite=False) as out:
                    out.write(dl.read_bytes())
        inputs_manifest, inputs_sha = load_inputs_manifest(manifest_path, args.inputs_manifest_sha)
        queue_manifest = build_nextgen_queue(inputs_manifest, inputs_sha)
        batch = NextGenBatch(root, queue_manifest, inputs_manifest, manifest_sha=metadata_digest(queue_manifest), reference_dir=args.reference_dir)
        with _lock(batch.dir / ".controller.lock"):
            batch.stage("batch-manifest", {"manifest": queue_manifest})
            result = batch.restore_individual_hub(
                inputs_repo=args.inputs_repo,
                inputs_revision=args.inputs_revision,
                expected_manifest_sha=inputs_sha,
                restore_baselines=args.restore_baselines,
                results_repo=args.results_repo,
                api=hub_client(optional=True),
            )
        if not result["complete"]:
            print(json.dumps(result, indent=2), flush=True)
            raise ValueError("Next-gen input restoration incomplete")
        print(json.dumps(result, indent=2), flush=True)
        return

    manifest_path = _resolve_inputs_manifest_path(root, args.inputs_manifest)
    inputs_manifest, inputs_sha = load_inputs_manifest(manifest_path, args.inputs_manifest_sha)
    queue_manifest = build_nextgen_queue(inputs_manifest, inputs_sha)
    batch_sha = metadata_digest(queue_manifest)
    batch = NextGenBatch(root, queue_manifest, inputs_manifest, manifest_sha=batch_sha, reference_dir=args.reference_dir)

    with _lock(batch.dir / ".controller.lock"):
        batch.stage("batch-manifest", {"manifest": queue_manifest})
        if args.action in ("first", "second"):
            require(args.enable, "GPU execution is disabled")
            approval = batch.approval(args.session)
            batch.verify_inputs()
            env_receipt = read_receipt(batch.dir / "verified-environment.json")
            require(env_receipt["runtime"] == verify_environment(), "DGX environment changed")
            gpu.observe_gpu(approval["uuid"])
            os.environ["CUDA_VISIBLE_DEVICES"] = approval["uuid"]
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            result = batch.execute(args.action, approval)
        elif args.action == "smoke":
            require(args.scope in ("first", "all"), "Approve the first group or the full 6-job queue")
            require(args.enable and args.allocated, "GPU smoke requires explicit authorization and institutional allocation")
            batch.verify_inputs()
            env_receipt = read_receipt(batch.dir / "verified-environment.json")
            require(env_receipt["runtime"] == verify_environment(), "DGX environment changed")
            result = batch.smoke(args.session, args.scope, args.uuid, args.shared, args.allocated, args.enable)
        elif args.action == "ready":
            result = batch.ready(args.scope)
        elif args.action == "inspect":
            result = {"devices": inspect_devices(), "storage": batch.storage(batch.jobs)}
            batch.save_named("inspections", {"manifest_sha256": batch.sha, **result})
        elif args.action == "results":
            result = batch.result_check(args.scope)
        elif args.action == "analyze":
            result = batch.analyze(fetch_baselines=args.fetch_baselines, results_repo=args.results_repo)
        elif args.action == "backup-hub":
            result = batch.backup_hub(args.hub_repo, args.scope, args.enable, args.hub_checkpoints)
        else:
            actions = {"verify": batch.verify_inputs, "plan": batch.plan, "report": batch.report}
            result = actions[args.action]()
    print(json.dumps(result, indent=2), flush=True)


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    main()
