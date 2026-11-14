"""Proxy-LM training/scoring ENGINE (not a plugin). The architecture comes from a `proxy_model`
plugin; this file owns data encoding, optimiser, schedule, mixed precision and bits-per-byte.

Every tokenizer trains the same model on the same text (equal bytes) and is scored on the same
held-out text with bits-per-byte:  BPB = sum(token NLL in nats) / ln 2 / (UTF-8 bytes scored).
BPB does not depend on vocabulary or segmentation, so it compares tokenizers fairly.
Known confounds (reported with results): a fixed context window holds more text for a
low-fertility tokenizer, and bigger vocabularies add embedding parameters (n_nonembed is logged).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    log=print,
) -> dict:
    t0 = time.time()
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

    model = model_plugin.build(V, seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model_plugin.n_embedding_params(model)
    steps = max(1, math.ceil(len(tr) * epochs / (batch * seq_len)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and amp_dtype == torch.float16)
    mb = micro_batch if micro_batch and micro_batch < batch else batch
    n_micro = math.ceil(batch / mb)
    rng = np.random.default_rng(seed)
    hi = len(tr) - seq_len - 1
    model.train()
    last = float("nan")
    for i in range(steps):
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
