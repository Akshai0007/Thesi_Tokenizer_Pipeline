"""Train and score the existing 38.8M GPT on immutable document-local token caches."""

import argparse
import json
import math
import os
import pickle
import platform
import random
import re
import shutil
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from experiments.mom_bpe import dgx_resources
from experiments.mom_bpe.diagnose import format_json, immutable, sha
from experiments.mom_bpe.proxy_blocks import (ARMS, MODES, POLICY, EncodedSplit, blocks, member, prepare_cache,
                                               publish, receipt, selection, sequences)
from experiments.mom_bpe.scale_data import code_digest
from experiments.mom_bpe.scale_fit import VOCAB
from experiments.mom_bpe.scale_proxy_plan import parameter_counts
from experiments.mom_bpe.verify_frozen import checked_member
from mltok import registry
from mltok.lib.atomic import atomic_writer
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import canonical
from mltok.lib.canonical_bpb import summarize_bpb
from mltok.proxy_train import _verify_checkpoint_archive
from mltok.snapshots import _lock, file_identity


CONFIG = Path(__file__).with_name("document-proxy.yaml")
GENERATION = re.compile(r"step-(\d{6})-([a-f0-9]{64})(?:-[a-f0-9]{12})?\.pt")
IGNORE = -100


def grouped(iterable, size):
    iterator = iter(iterable)
    while part := list(islice(iterator, size)):
        yield part


def schedule(split, chosen, context, update_targets, microbatch):
    planned, pending = [], []
    targets, index = 0, 0
    progress = [{"optimizer_updates": 0, "next_block": 0, "predicted_tokens": 0, "canonical_bytes": 0,
                 "context_positions": 0, "right_padding_positions": 0, "predicted_eos": 0, "microbatches": 0}]

    def append_update():
        batch_groups = list(grouped(pending, microbatch))
        padding = sum(max(block.targets for block in batch) * len(batch) for batch in batch_groups) - targets
        eos = sum(b.stop > split.docs[b.doc]["length"] for b in pending)
        prior = progress[-1]
        progress.append({"optimizer_updates": len(progress), "next_block": index,
                         "predicted_tokens": prior["predicted_tokens"] + targets,
                         "canonical_bytes": prior["canonical_bytes"] + sum(b.source_bytes for b in pending),
                         "context_positions": prior["context_positions"] + targets,
                         "right_padding_positions": prior["right_padding_positions"] + padding,
                         "predicted_eos": prior["predicted_eos"] + eos,
                         "microbatches": prior["microbatches"] + len(batch_groups)})
        planned.append(batch_groups)

    for block in blocks(split, chosen, context):
        pending.append(block)
        index += 1
        targets += block.targets
        if targets >= update_targets:
            append_update()
            pending, targets = [], 0
    if pending:
        append_update()
    return planned, progress


def batch_tensors(split, batch, specials, device):
    pairs = [sequences(split, block, specials["<s>"], specials["</s>"]) for block in batch]
    width = max(len(x) for x, _ in pairs)
    inputs = [x + [specials["<pad>"]] * (width - len(x)) for x, _ in pairs]
    targets = [y + [IGNORE] * (width - len(y)) for _, y in pairs]
    return (torch.tensor(inputs, device=device, dtype=torch.long), torch.tensor(targets, device=device, dtype=torch.long))


def token_losses(model, inputs, targets, *, bf16, reduction):
    # A real position sees only its own left context under causal GPT attention.
    with torch.autocast(inputs.device.type, dtype=torch.bfloat16, enabled=bf16):
        logits = model(inputs)
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                             ignore_index=IGNORE, reduction=reduction)
    return losses.reshape(targets.shape) if reduction == "none" else losses


@torch.inference_mode()
def score(model, split, context, specials, microbatch, device, bf16=False):
    model.eval()
    selected = [(i, doc["length"] + 1) for i, doc in enumerate(split.docs)]
    nats = [[] for _ in selected]
    for batch in grouped(blocks(split, selected, context), microbatch):
        inputs, targets = batch_tensors(split, batch, specials, device)
        values = token_losses(model, inputs, targets, bf16=bf16, reduction="none").sum(1, dtype=torch.float64).tolist()
        for block, value in zip(batch, values, strict=True):
            nats[block.doc].append(value)
    per_doc = [{"id": doc["id"], "canonical_sha256": doc["canonical_sha256"],
                **summarize_bpb([math.fsum(ns)], doc["canonical_bytes"], doc["length"] + 1)}
               for doc, ns in zip(split.docs, nats, strict=True)]
    summary = summarize_bpb((doc["nll_nats"] for doc in per_doc), split.canonical_bytes,
                            sum(doc["predicted_tokens"] for doc in per_doc))
    return {**summary, "per_document": per_doc}


def clock_key(mode):
    return "canonical_bytes" if mode == MODES[0] else "predicted_tokens"


def learning_rate(policy, position, total):
    fraction = position / total
    warm = policy["warmup_fraction"]
    peak, end = policy["peak_lr"], policy["end_lr"]
    if fraction < warm:
        return peak * fraction / warm
    portion = (fraction - warm) / (1 - warm)
    return end + (peak - end) * (1 + math.cos(math.pi * portion)) / 2


def rng_state(device):
    kind, keys, pos, gaussian, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": (kind, keys.tolist(), pos, gaussian, cached),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    kind, keys, pos, gaussian, cached = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_generation(out, state):
    """Verify a closed generation, then link it at a fresh name on the same volume."""
    with TemporaryDirectory(prefix="write-", dir=out) as folder:
        temp = Path(folder) / "state.pt"
        with atomic_writer(temp, overwrite=False) as stream:
            torch.save(state, stream)
        with temp.open("rb") as stream:
            _verify_checkpoint_archive(stream)
        _, digest = file_identity(temp)
        final = out / f"step-{state['step']:06d}-{digest}.pt"
        try:
            os.link(temp, final)
        except FileExistsError:
            if file_identity(final)[1] != digest:
                final = final.with_name(f"{final.stem}-{uuid4().hex[:12]}.pt")
                os.link(temp, final)
        if os.name == "posix":
            fd = os.open(out, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    return member(out, final)


def restore_generation(out, identity, progress, new_state, device, rate):
    candidates = sorted(out.glob("step-*.pt"), reverse=True)
    initial_rng = rng_state(device)
    for path in candidates:
        try:
            match = GENERATION.fullmatch(path.name)
            if not match or path.is_symlink() or file_identity(path)[1] != match[2]:
                continue
            step = int(match[1])
            if not 0 < step < len(progress):
                continue
            with path.open("rb") as stream:
                state = torch.load(stream, map_location="cpu", weights_only=True)
            if (state["identity"] != identity or state["step"] != step or state["exposure"] != progress[step]
                    or state["scheduler"] != rate(step) or not math.isfinite(state["train_nll_nats"])):
                continue
            model, optimizer = new_state()
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if any(group["lr"] != state["scheduler"]["lr"] for group in optimizer.param_groups):
                raise ValueError("Checkpoint optimizer and scheduler disagree")
            restore_rng(state["rng"], device)
            return model, optimizer, step, state["train_nll_nats"], member(out, path)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError, BadZipFile):
            restore_rng(initial_rng, device)
    if candidates:
        raise ValueError("No usable checkpoint matches this run. Existing generations are preserved.")
    return (*new_state(), 0, 0.0, None)


def train(out, identity, split, planned, progress, new_state, config, mode, specials, device, *, telemetry=None):
    key = clock_key(mode)
    optimizer_policy = config["optimizer"]

    def rate(step):
        clock = progress[step][key]
        total = progress[-1][key]
        return {"clock": key, "position": clock, "total": total,
                "lr": learning_rate(optimizer_policy, clock, total)}

    model, optimizer, start, total_nll, checkpoint = restore_generation(out, identity, progress, new_state, device, rate)
    if telemetry is not None:
        telemetry.restored(start, checkpoint)
    model.train()
    local_nll = torch.zeros((), device=device, dtype=torch.float64)
    for step, microbatches in enumerate(planned[start:], start + 1):
        count = progress[step]["predicted_tokens"] - progress[step - 1]["predicted_tokens"]
        scheduler = rate(step)
        for group in optimizer.param_groups:
            group["lr"] = scheduler["lr"]
        optimizer.zero_grad(set_to_none=True)
        for batch in microbatches:
            inputs, targets = batch_tensors(split, batch, specials, device)
            summed = token_losses(model, inputs, targets, bf16=device.type == "cuda", reduction="sum")
            (summed / count).backward()
            local_nll += summed.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_policy["clip"], error_if_nonfinite=True)
        optimizer.step()
        if step == len(planned) or step % config["checkpoint_updates"] == 0:
            total_nll += local_nll.item()
            local_nll.zero_()
            checkpoint = save_generation(out, {"identity": identity, "step": step, "exposure": progress[step],
                "scheduler": scheduler, "rng": rng_state(device), "train_nll_nats": total_nll,
                "model": model.state_dict(), "optimizer": optimizer.state_dict()})
            if telemetry is not None:
                telemetry.checkpoint(step, checkpoint)
            print(f"Checkpointed update {step}/{len(planned)}", flush=True)
    return model, total_nll, checkpoint


def gpu_preflight(minimum_mib=8192, *, max_gpu_util_pct=5, allow_shared_gpu=False):
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    observed = dgx_resources.GpuAdmission(minimum_mib, max_gpu_util_pct, allow_shared_gpu).admit(uuid)
    if (os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8") or torch.cuda.device_count() != 1
            or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise ValueError("Expected one BF16 GPU and deterministic CUBLAS_WORKSPACE_CONFIG")
    return {"physical_index": observed["physical_index"], "uuid": uuid, "type": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}


def read_config(args):
    raw = CONFIG.read_bytes()
    config = yaml.safe_load(raw)
    for key in ("seed", "update_targets", "microbatch_blocks", "checkpoint_updates"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if not 0 <= config["seed"] < 2**32 or any(config[k] < 1 for k in
                                            ("update_targets", "microbatch_blocks", "checkpoint_updates")):
        raise ValueError("Seed and training batch/checkpoint settings are out of range")
    policy = config["optimizer"]
    if policy["method"] != "adamw" or policy["schedule"] != "exposure_warmup_cosine_v1":
        raise ValueError("Expected AdamW and the pinned exposure schedule")
    return config, sha(raw.replace(b"\r\n", b"\n"))


def fit_cache(args):
    config, config_sha = read_config(args)
    root = args.cache.resolve()
    cache, cache_sha = receipt(root / "cache.json")
    enc = cache["identity"]
    context, specials, vocab = config["context"], enc["specials"], enc["vocab"]
    if (enc["schema"] != 1 or enc["context"] != context or enc["policy"] != POLICY or enc["arm"] not in ARMS
            or vocab != VOCAB or specials != dict(zip(DEFAULT_SPECIALS, range(len(DEFAULT_SPECIALS))))):
        raise ValueError("Cache schema, block policy, vocabulary or arm differs")
    train_split = EncodedSplit(root, "train", cache["splits"]["train"])
    val_split = EncodedSplit(root, "validation", cache["splits"]["validation"])
    limit = config["budgets"][args.mode] if args.budget is None else args.budget
    chosen, ledger = selection(train_split, args.mode, limit)
    planned, progress = schedule(train_split, chosen, context, config["update_targets"], config["microbatch_blocks"])
    if any(progress[-1][k] != ledger[k] for k in ("predicted_tokens", "predicted_eos", "canonical_bytes")):
        raise ValueError("Planned blocks differ from the exposure ledger")
    result_plan = {"mode": args.mode, "budget": limit, "config": config, "config_sha256": config_sha,
                   "optimizer_sha256": sha(canonical(config["optimizer"])), "cache_sha256": cache_sha,
                   "arm": enc["arm"], "forecast": progress[-1], "exposure": ledger,
                   "validation": {"split_sha256": cache["splits"]["validation"], "canonical_bytes": val_split.canonical_bytes,
                                  "documents": len(val_split.docs), "canonical_stream_sha256": enc["streams"]["validation"]}}
    if args.command == "plan":
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            publish(args.out, result_plan)
        return result_plan

    registry.load_plugins()
    device = torch.device(args.device)
    admission = dgx_resources.GpuAdmission(getattr(args, "min_free_mib", 8192), getattr(args, "max_gpu_util_pct", 5),
                                          getattr(args, "allow_shared_gpu", False))
    gpu = gpu_preflight(admission.min_free_mib, max_gpu_util_pct=admission.max_gpu_util_pct,
                        allow_shared_gpu=admission.allow_shared_gpu) if device.type == "cuda" else None
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    spec = config["model"]
    count = parameter_counts(spec["n_layer"], spec["d_model"], context)["total"]
    if count != 38_799_936 or spec != {"type": "gpt", "n_layer": 7, "n_head": 7, "d_model": 448} or context != 512:
        raise ValueError("Expected the fixed 38.8M GPT architecture")
    policy = config["optimizer"]
    driver_digest = code_digest(__file__)
    implementation = {"driver": dgx_resources.scientific_driver_digest(driver_digest),
                      "blocks": code_digest(blocks.__code__.co_filename),
                      "scorer": code_digest(summarize_bpb.__code__.co_filename)}
    identity = {**result_plan, "tokenizer_sha256": enc["tokenizer_sha256"], "manifest_sha256": enc["manifest_sha256"],
                "parameters": count, "model_plugin": registry.fingerprint("proxy_model", spec), "implementation": implementation,
                "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "machine": platform.machine(),
                            "system": platform.system(), "python": platform.python_version(), "threads": args.threads,
                            "gpu": gpu, "precision": "bf16_autocast" if gpu else "fp32",
                            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if gpu else None}}

    def new_state():
        model = registry.make("proxy_model", spec).build(vocab, context).to(device)
        if sum(parameter.numel() for parameter in model.parameters()) != count:
            raise ValueError("GPT parameter count differs from the fixed architecture")
        optimizer = torch.optim.AdamW(model.parameters(), lr=policy["peak_lr"], betas=tuple(policy["betas"]),
                                      weight_decay=policy["weight_decay"], foreach=False, fused=False)
        return model, optimizer

    out = args.out.resolve()
    with _lock(out / ".fit.lock"):
        final = out / "result.json"
        if final.exists():
            result, _ = receipt(final)
            if result["identity"] != identity:
                raise ValueError("Completed result is incompatible")
            checked_member(out, result["checkpoint"])
            immutable(out / "identity.json", format_json(identity))
            return result
        # Approximate three FP32 tensors per parameter (weights and two Adam moments).
        interval = config["checkpoint_updates"]
        generations = math.ceil(len(planned) / interval)
        existing = sum(1 for path in out.glob("step-*.pt") if GENERATION.fullmatch(path.name))
        if shutil.disk_usage(out).free < (max(generations - existing, 0) + 2) * (12 * count + 2**20):
            raise ValueError("Insufficient disk for immutable optimizer checkpoints and an in-flight generation")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        telemetry = nullcontext()
        if gpu:
            observation = admission.admit(gpu["uuid"])
            if observation["physical_index"] != gpu["physical_index"]:
                raise ValueError("Physical GPU assignment changed after preflight")
            telemetry = dgx_resources.ResourceTelemetry(
                out, admission, observation, torch.cuda, device,
                {"driver": driver_digest, "resources": code_digest(dgx_resources.__file__)},
            )
        immutable(out / "identity.json", format_json(identity))
        with telemetry as resources:
            model, nats, checkpoint = train(out, identity, train_split, planned, progress, new_state, config, args.mode,
                                            specials, device, telemetry=resources)
        checked_member(out, checkpoint)
        evaluation = score(model, val_split, context, specials, config["microbatch_blocks"], device, bool(gpu))
        result = {"identity": identity, "checkpoint": checkpoint, "training_nll_nats": nats, "evaluation": evaluation}
        publish(final, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--tokenizer", type=Path, required=True)
    prepare.add_argument("--arm", choices=ARMS, required=True)
    prepare.add_argument("--cache-root", type=Path, required=True)
    prepare.add_argument("--train-prefix-bytes", type=int)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--mode", choices=MODES, required=True)
        command.add_argument("--budget", type=int)
        command.add_argument("--out", type=Path, required=name == "run")
        for key in ("seed", "update-targets", "microbatch-blocks", "checkpoint-updates"):
            command.add_argument("--" + key, type=int)
        command.add_argument("--threads", type=int, choices=range(1, 65), default=1)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--min-free-mib", type=int, default=8192)
        command.add_argument("--max-gpu-util-pct", type=int, default=5)
        command.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare_cache(args.manifest, args.tokenizer, args.cache_root, args.arm,
                            train_prefix_bytes=args.train_prefix_bytes))
    else:
        try:
            dgx_resources.GpuAdmission(args.min_free_mib, args.max_gpu_util_pct, args.allow_shared_gpu)
        except ValueError as exc:
            parser.error(str(exc))
        output = fit_cache(args)
        if args.command == "run":
            output = {k: v for k, v in output["evaluation"].items() if k != "per_document"}
        print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
