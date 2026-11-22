"""Proxy-LM training/scoring ENGINE (not a plugin). The architecture comes from a `proxy_model`
plugin; this file owns data encoding, optimiser, schedule, mixed precision and bits-per-byte.

Every tokenizer trains the same model on the same text (equal bytes) and is scored on the same
held-out text with bits-per-byte:  BPB = sum(token NLL in nats) / ln 2 / (UTF-8 bytes scored).
BPB does not depend on vocabulary or segmentation, so it compares tokenizers fairly.
Known confounds (reported with results): a fixed context window holds more text for a
low-fertility tokenizer, and bigger vocabularies add embedding parameters (n_nonembed is logged).
"""
from __future__ import annotations

import hashlib
import math
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lib.atomic import atomic_writer


def _encode_stream(backend, texts, eos_id, budget_bytes, chunk=2000):
    ids, nbytes, buf = [], 0, []
    for t in texts:
        buf.append(t)
        if len(buf) >= chunk:
            nbytes += _flush(backend, buf, eos_id, ids)
            buf = []
            if nbytes >= budget_bytes:
                break
    if buf and nbytes < budget_bytes:
        nbytes += _flush(backend, buf, eos_id, ids)
    return np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64), nbytes


def _flush(backend, buf, eos_id, out):
    enc = backend.encode_batch(buf)
    for e in enc:
        out.append(np.asarray(e + [eos_id], dtype=np.int64))
    return sum(len(t.encode("utf-8")) for t in buf)


def _verify_checkpoint_archive(stream):
    with zipfile.ZipFile(stream) as archive:
        if archive.testzip() is not None:
            raise ValueError("checkpoint archive has a corrupt member")
    stream.seek(0)


def train_proxy(
    backend,
    train_texts,
    eval_texts,
    model_plugin,
    *,
    seq_len=512,
    batch=64,
    train_mb=100,
    eval_mb=10,
    epochs=1,
    lr=1e-3,
    seed=0,
    device=None,
    micro_batch=0,
    ckpt_path=None,
    ckpt_every=0,
    ckpt_key="",
    on_ckpt=None,
    log=print,
) -> dict:
    """Save at the configured interval and at the final step when a path is supplied.

    The final checkpoint is saved before evaluation and retained by the caller.
    Exact continuation requires deterministic operations and the same execution environment.
    Legacy checkpoints lack global RNG and input fingerprints.
    """
    t0 = time.time()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    eos = backend.eos_id if backend.eos_id is not None and backend.eos_id >= 0 else 0
    tr, tr_bytes = _encode_stream(backend, train_texts, eos, int(train_mb * 1e6))
    ev, ev_bytes = _encode_stream(backend, eval_texts, eos, int(eval_mb * 1e6))
    if len(tr) < seq_len * 4 or len(ev) < seq_len + 1:
        raise ValueError("not enough data for proxy run")
    V = backend.vocab_size
    dt_np = np.uint16 if V < 65536 else np.int32
    tr, ev = tr.astype(dt_np), ev.astype(dt_np)

    steps = max(1, math.ceil(len(tr) * epochs / (batch * seq_len)))
    warm = max(1, int(0.03 * steps))

    def lr_at(i):
        if i < warm:
            return lr * (i + 1) / warm
        p = (i - warm) / max(1, steps - warm)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    use_cuda = device.startswith("cuda")
    if use_cuda:
        # decide from compute capability: newer PyTorch reports bf16 as "supported" on
        # T4/P100 through slow emulation; native bf16 needs sm80+ (A100/H100).
        major = torch.cuda.get_device_capability(torch.device(device))[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        amp_dtype = torch.float32
    mb = micro_batch if micro_batch and micro_batch < batch else batch
    n_micro = math.ceil(batch / mb)
    hi = len(tr) - seq_len - 1
    start = 0

    def new_state():
        model = model_plugin.build(V, seq_len).to(device)
        return (
            model,
            torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1),
            torch.amp.GradScaler("cuda", enabled=use_cuda and amp_dtype == torch.float16),
            np.random.default_rng(seed),
        )

    def rng_state():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                torch.cuda.get_rng_state(device) if use_cuda else None)

    def restore_rng(state):
        python_rng, numpy_rng, torch_rng, cuda_rng = state
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if use_cuda:
            torch.cuda.set_rng_state(cuda_rng, device)

    checkpoint = Path(ckpt_path) if ckpt_path else None
    primary_valid = False
    restored = None
    if checkpoint is not None:
        previous = checkpoint.with_suffix(".previous.ckpt")
        metadata = {
            "seed": seed, "vocab": V, "seq_len": seq_len, "batch": batch, "micro_batch": mb,
            "train_mb": train_mb, "eval_mb": eval_mb, "epochs": epochs, "lr": lr,
            "schedule": "warmup_cosine_v1", "warmup_steps": warm,
            "train_bytes": tr_bytes, "eval_bytes": ev_bytes,
            "train_ids": hashlib.sha256(tr).hexdigest(), "eval_ids": hashlib.sha256(ev).hexdigest(),
            "amp_dtype": str(amp_dtype), "device_type": torch.device(device).type,
            "torch_version": str(torch.__version__),
        }
        available = [p for p in (checkpoint, previous) if p.exists()]
        initial_rng = rng_state() if available else None
        for path in available:
            try:
                # Only restore checkpoints from trusted writers.
                with path.open("rb") as stream:
                    _verify_checkpoint_archive(stream)
                    ck = torch.load(stream, map_location="cpu", weights_only=False)
                step = ck["step"]
                version = ck.get("version", 0)
                if (version not in (0, 1) or ck["key"] != ckpt_key or ck["steps"] != steps
                        or type(step) is not int or not 0 < step <= steps
                        or (version == 1 and ck["metadata"] != metadata)):
                    raise ValueError("checkpoint does not match this training run")
                if step <= start:
                    continue
                model, opt, scaler, rng = new_state()
                build_rng = rng_state()
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                scaler.load_state_dict(ck["scaler"])
                rng.bit_generator.state = ck["rng"]
                if version == 1:
                    build_rng = (ck["python_rng"], ck["numpy_rng"], ck["torch_rng"], ck["cuda_rng"])
                restore_rng(build_rng)
                restored = (model, opt, scaler, rng, build_rng, version)
                start, primary_valid = step, path == checkpoint
                if start == steps:
                    break
            except Exception as e:
                log(f"    [{backend.name}] could not load checkpoint {path.name} ({type(e).__name__})")
            finally:
                restore_rng(initial_rng)
        if available and restored is None:
            raise RuntimeError("No usable proxy checkpoint. Existing checkpoints were preserved.")
        if restored is not None:
            model, opt, scaler, rng, saved_rng, version = restored
            del ck, restored
            if version == 0:
                restore_rng(saved_rng)
                log(f"    [{backend.name}] legacy checkpoint lacks full RNG and input metadata")
            log(f"    [{backend.name}] resuming from checkpoint at step {start}/{steps}")

    if start == 0:
        model, opt, scaler, rng = new_state()
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model_plugin.n_embedding_params(model)
    if start and version == 1:
        restore_rng(saved_rng)

    def save_checkpoint(step):
        nonlocal primary_valid
        if primary_valid:
            with checkpoint.open("rb") as source, atomic_writer(previous) as target:
                shutil.copyfileobj(source, target)
                target.flush()
                _verify_checkpoint_archive(target)
        python_rng, numpy_rng, torch_rng, cuda_rng = rng_state()
        state = {
            "version": 1, "key": ckpt_key, "steps": steps, "step": step, "metadata": metadata,
            "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
            "rng": rng.bit_generator.state, "python_rng": python_rng, "numpy_rng": numpy_rng,
            "torch_rng": torch_rng, "cuda_rng": cuda_rng,
        }
        with atomic_writer(checkpoint) as stream:
            torch.save(state, stream)
            stream.flush()
            _verify_checkpoint_archive(stream)
        primary_valid = True
        if on_ckpt:
            on_ckpt(ckpt_path)
    model.train()
    last = float("nan")
    for i in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(i)
        st = rng.integers(0, hi, size=batch)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for m in range(n_micro):
            sl = st[m * mb : (m + 1) * mb]
            xb = torch.from_numpy(np.stack([tr[x : x + seq_len + 1] for x in sl]).astype(np.int64)).to(device)
            with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=amp_dtype, enabled=use_cuda):
                logits = model(xb[:, :-1])
                loss = F.cross_entropy(logits.float().reshape(-1, V), xb[:, 1:].reshape(-1))
            scaler.scale(loss * (len(sl) / batch)).backward()
            loss_sum += loss.item() * (len(sl) / batch)
        loss = torch.tensor(loss_sum)
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if checkpoint is not None and (i + 1 == steps or (ckpt_every and (i + 1) % ckpt_every == 0)):
            save_checkpoint(i + 1)
        last = loss.item() if (i % 50 == 0 or i == steps - 1) else last
        if i % max(1, steps // 10) == 0:
            log(f"    [{backend.name}] step {i}/{steps} loss {last:.3f}")

    model.eval()
    n_win = (len(ev) - 1) // seq_len
    nll, scored = 0.0, 0
    with torch.no_grad():
        for s in range(0, n_win, batch):
            idxs = range(s, min(s + batch, n_win))
            for k in range(0, len(idxs), mb):
                sub = idxs[k : k + mb]
                xb = np.stack([ev[j * seq_len : j * seq_len + seq_len + 1] for j in sub]).astype(np.int64)
                xb = torch.from_numpy(xb).to(device)
                with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=amp_dtype, enabled=use_cuda):
                    logits = model(xb[:, :-1])
                nll += F.cross_entropy(logits.float().reshape(-1, V), xb[:, 1:].reshape(-1), reduction="sum").item()
                scored += xb[:, 1:].numel()
    bytes_scored = ev_bytes * scored / len(ev)
    bpb = nll / math.log(2) / bytes_scored
    return {
        "name": backend.name,
        "bpb": bpb,
        "eval_nll_per_token": nll / scored,
        "n_params": n_params,
        "n_nonembed": n_params - n_embed,
        "train_tokens": int(len(tr)),
        "train_bytes": int(tr_bytes),
        "bytes_per_token": tr_bytes / len(tr),
        "steps": steps,
        "seconds": round(time.time() - t0, 1),
        "device": device,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(torch.device(device)) / 1024**3, 2) if use_cuda else 0.0,
        "tokens_per_sec": round(len(tr) * epochs / max(time.time() - t0, 1e-9)),
        "model": getattr(model_plugin, "name", "?"),
    }
