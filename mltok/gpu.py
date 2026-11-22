"""Shared-GPU etiquette for the DGX (80 GB A100, we may use at most ~60 GB).

* `gpu_status()`        reads nvidia-smi (no-op on machines without it)
* `wait_for_gpu()`      blocks until >= min_free_gb is free and utilization is low
* `cap_memory()`        hard-caps this process's CUDA allocator at max_gb
"""
from __future__ import annotations

import shutil
import subprocess
import time


def gpu_status(index: int = 0) -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.total,memory.used,utilization.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        ).strip()
        total, used, util, name = [x.strip() for x in out.split(",", 3)]
        return {
            "name": name,
            "total_gb": float(total) / 1024,
            "used_gb": float(used) / 1024,
            "free_gb": (float(total) - float(used)) / 1024,
            "util_pct": float(util),
        }
    except Exception:  # noqa: BLE001
        return None


def wait_for_gpu(
    min_free_gb: float = 60.0,
    max_util_pct: float = 30.0,
    poll_s: int = 60,
    timeout_s: int = 6 * 3600,
    index: int = 0,
    log=print,
) -> bool:
    """Return true only after confirming that the requested GPU is available."""
    if str(index).strip().isdigit() and int(index) == 7:
        log("[gpu] physical GPU 7 is reserved on the university DGX")
        return False
    t0 = time.time()
    while True:
        s = gpu_status(index)
        if s is None:
            log("[gpu] could not verify GPU availability")
            return False
        if s["free_gb"] >= min_free_gb and s["util_pct"] <= max_util_pct:
            log(f"[gpu] ok: {s['free_gb']:.1f} GB free, util {s['util_pct']:.0f}%")
            return True
        if time.time() - t0 > timeout_s:
            log("[gpu] timed out waiting for a free GPU")
            return False
        log(f"[gpu] busy: {s['free_gb']:.1f} GB free, util {s['util_pct']:.0f}% - waiting {poll_s}s")
        time.sleep(poll_s)


def cap_memory(max_gb: float = 60.0, index: int = 0, log=print) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        total = torch.cuda.get_device_properties(index).total_memory / 1024**3
        frac = min(1.0, max_gb / total)
        torch.cuda.set_per_process_memory_fraction(frac, index)
        log(f"[gpu] capped at {max_gb:.0f} GB of {total:.0f} GB (fraction {frac:.2f})")
    except Exception as e:  # noqa: BLE001
        log(f"[gpu] could not cap memory: {e}")


def gpu_count() -> int:
    """Number of GPUs visible to nvidia-smi (0 if none). Does not initialise CUDA."""
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True, timeout=15
        )
        return len([ln for ln in out.splitlines() if ln.strip()])
    except Exception:  # noqa: BLE001
        return 0
