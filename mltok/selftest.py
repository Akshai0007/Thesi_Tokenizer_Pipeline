"""`python -m mltok selftest` - run this first on Kaggle (and on the DGX).

Checks, in order:
 1. PyTorch / CUDA visibility, GPU names, compute capability, chosen AMP dtype
 2. a short proxy-LM training on EVERY visible GPU (finite loss, no NaN, tokens/s, peak memory)
 3. if >= 2 GPUs: both GPUs run at the same time in separate processes pinned with
    CUDA_VISIBLE_DEVICES (exactly how the pipeline fans proxy runs out)
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tiny_backend_and_text():
    from .contract_checks import tiny_backend

    be, docs = tiny_backend(vocab=2048)
    cut = int(len(docs) * 0.9)
    return be, docs[:cut], docs[cut:]


def _check_device(device: str, log=print) -> bool:
    import torch

    from . import registry
    from .proxy_train import train_proxy

    be, tr, ev = _tiny_backend_and_text()
    size = "tiny" if device.startswith("cuda") else "nano"
    t0 = time.time()
    registry.load_plugins()
    r = train_proxy(
        be, tr, ev, registry.make("proxy_model", {"type": "gpt", "size": size}), seq_len=256, batch=16, micro_batch=8, train_mb=3, eval_mb=0.4,
        lr=1e-3, device=device, log=lambda *_: None,
    )
    ok = math.isfinite(r["bpb"]) and r["bpb"] > 0
    log(
        f"  [{'PASS' if ok else 'FAIL'}] {device}: bpb={r['bpb']:.3f} dtype={r['amp_dtype']} "
        f"tok/s={r['tokens_per_sec']:,} peak_mem={r['peak_mem_gb']} GB steps={r['steps']} ({time.time() - t0:.0f}s)"
    )
    return ok


def run_selftest(single: bool = False, log=print) -> bool:
    import torch

    ok = True
    log(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    n = torch.cuda.device_count()
    for i in range(n):
        pr = torch.cuda.get_device_properties(i)
        log(f"  GPU {i}: {pr.name} | {pr.total_memory / 1024**3:.1f} GB | sm{pr.major}{pr.minor} "
            f"| amp={'bf16' if pr.major >= 8 else 'fp16 (+GradScaler)'}")
    if n == 0:
        log("  no CUDA device: testing the CPU path only (the proxy stage will be very slow on CPU)")
        return _check_device("cpu", log)

    for i in range(1 if single else n):
        ok &= _check_device(f"cuda:{i}", log)

    if not single and n >= 2:
        log(f"  concurrent run on {n} GPUs (separate processes, CUDA_VISIBLE_DEVICES pinned)...")
        procs = []
        for i in range(n):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "mltok", "selftest", "--single"], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT),
            ))
        for i, p in enumerate(procs):
            out, _ = p.communicate()
            good = p.returncode == 0
            ok &= good
            last = [ln for ln in out.splitlines() if "bpb=" in ln]
            log(f"  [{'PASS' if good else 'FAIL'}] worker {i}: {last[-1].strip() if last else out[-300:]}")
    log("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return ok
