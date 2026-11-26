"""GPU admission and operational telemetry for the document-local experiment."""

import csv
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from mltok.lib.artifact_paths import unlinked_path
from mltok.lib.atomic import atomic_writer


GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
LOG = logging.getLogger(__name__)
PINNED_DRIVER = "440fdc95961049237392a9cd576b41486e90b5d48711bdd5f66a88a84e527625"
# Only this reviewed operational revision may reuse the original scientific identity.
RESOURCE_DRIVER = "9a0015323164d106b2d310b95d008fee9d6cdc2eb45db021b477a9a4a1fe9f6c"


def scientific_driver_digest(actual):
    return PINNED_DRIVER if actual == RESOURCE_DRIVER else actual


def _query(fields, description):
    try:
        output = subprocess.check_output(
            ["nvidia-smi", fields, "--format=csv,noheader,nounits"],
            text=True, timeout=10, stderr=subprocess.DEVNULL,
        )
        return [tuple(cell.strip() for cell in row) for row in csv.reader(output.splitlines(), strict=True) if row]
    except (OSError, UnicodeError, csv.Error, subprocess.SubprocessError) as exc:
        raise ValueError(f"Cannot verify {description} with nvidia-smi") from exc


def observe_gpu(uuid):
    if not GPU_UUID.fullmatch(uuid):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one full physical GPU UUID")
    rows = _query("--query-gpu=index,uuid,memory.free,utilization.gpu", "physical GPU resources")
    matches = [row for row in rows if len(row) == 4 and row[1] == uuid]
    if len(matches) != 1:
        raise ValueError("Physical GPU UUID is missing or ambiguous in nvidia-smi")
    index, _, free, utilization = matches[0]
    if not all(value.isascii() and value.isdecimal() for value in (index, free, utilization)):
        raise ValueError("Cannot verify physical GPU index, free memory or utilization")
    if int(index) == 7:
        raise ValueError("Physical GPU 7 is prohibited")
    if int(utilization) > 100:
        raise ValueError("Cannot verify physical GPU utilization")
    return {"physical_index": int(index), "uuid": uuid, "free_mib": int(free), "utilization_pct": int(utilization)}


def _other_compute_processes(uuid):
    rows = _query("--query-compute-apps=gpu_uuid,pid", "GPU compute processes")
    if rows == [("No running processes found",)]:
        return set()
    if any(len(row) != 2 or not GPU_UUID.fullmatch(row[0]) or not row[1].isascii()
           or not row[1].isdecimal() or int(row[1]) < 1 for row in rows):
        raise ValueError("Cannot verify GPU compute processes from nvidia-smi")
    return {int(pid) for device, pid in rows if device == uuid and int(pid) != os.getpid()}


@dataclass(frozen=True)
class GpuAdmission:
    min_free_mib: int = 8192
    max_gpu_util_pct: int = 5
    allow_shared_gpu: bool = False

    def __post_init__(self):
        if type(self.min_free_mib) is not int or self.min_free_mib < 1:
            raise ValueError("Expected a positive minimum free GPU memory in MiB")
        if type(self.max_gpu_util_pct) is not int or not 0 <= self.max_gpu_util_pct <= 15:
            raise ValueError("Maximum GPU utilization must be between 0 and 15%")
        if type(self.allow_shared_gpu) is not bool:
            raise ValueError("Shared GPU permission must be explicit")
        if self.max_gpu_util_pct > 5 and not self.allow_shared_gpu:
            raise ValueError("A threshold above 5% requires --allow-shared-gpu")

    def admit(self, uuid):
        observation = observe_gpu(uuid)
        if observation["free_mib"] < self.min_free_mib:
            raise ValueError(f"GPU has insufficient free memory ({observation['free_mib']} MiB)")
        if observation["utilization_pct"] > self.max_gpu_util_pct:
            raise ValueError(f"GPU utilization exceeds {self.max_gpu_util_pct}%")
        others = _other_compute_processes(uuid)
        if others and not self.allow_shared_gpu:
            raise ValueError("Other GPU compute processes are active. Shared use requires --allow-shared-gpu")
        return {**observation, "other_compute_processes": len(others)}


class ResourceTelemetry:
    """Store immutable per-attempt records outside results and checkpoints."""

    def __init__(self, out, policy, initial, cuda, device, provenance):
        self.path = Path(out) / "resource-telemetry" / uuid4().hex
        self.policy = policy
        self.initial = initial
        self.cuda = cuda
        self.device = device
        self.provenance = provenance
        self.enabled = False
        self.peaks_available = False
        self.started = None

    def _write(self, name, record):
        if self.enabled:
            try:
                with atomic_writer(self.path / name, overwrite=False) as stream:
                    stream.write((json.dumps(record, sort_keys=True, indent=2) + "\n").encode())
            except OSError as exc:
                LOG.warning("Could not persist GPU telemetry %s (%s)", name, type(exc).__name__)

    def __enter__(self):
        try:
            unlinked_path(self.path).mkdir(parents=True)
            self.enabled = True
        except (OSError, ValueError) as exc:
            LOG.warning("Could not start GPU telemetry (%s)", type(exc).__name__)
        if self.enabled:
            try:
                self.cuda.reset_peak_memory_stats(self.device)
                self.peaks_available = True
            except RuntimeError:
                LOG.warning("Could not reset GPU allocator counters")
        self._write("start.json", {"schema": "mom-dgx-resource-telemetry-v1", "admission": asdict(self.policy),
                                   "initial": self.initial, "operational_code_sha256": self.provenance})
        self.started = time.monotonic()
        return self

    def restored(self, step, checkpoint):
        self._write("resume.json", {"starting_update": step, "checkpoint": checkpoint})

    def _sample(self):
        result = {"training_seconds": round(time.monotonic() - self.started, 6), "gpu": None,
                  "peak_allocated_bytes": None, "peak_reserved_bytes": None}
        if self.peaks_available:
            try:
                result["peak_allocated_bytes"] = self.cuda.max_memory_allocated(self.device)
                result["peak_reserved_bytes"] = self.cuda.max_memory_reserved(self.device)
            except RuntimeError:
                LOG.warning("Could not read GPU allocator counters")
        try:
            result["gpu"] = observe_gpu(self.initial["uuid"])
        except ValueError:
            LOG.warning("Could not sample physical GPU free memory and utilization")
        return result

    def checkpoint(self, step, member):
        if self.enabled:
            self._write(f"update-{step:06d}.json", {"update": step, "checkpoint": member, **self._sample()})

    def __exit__(self, exc_type, _exc, _traceback):
        if self.enabled:
            status = "training_finished" if exc_type is None else "interrupted" if exc_type is KeyboardInterrupt else "failed"
            self._write("end.json", {"status": status, **self._sample()})
