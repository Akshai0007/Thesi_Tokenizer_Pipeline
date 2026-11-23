"""Run a bounded CPU-only document-local BPB comparison."""

import argparse
import io
import json
import math
from collections import Counter
from itertools import islice
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn

from experiments.mom_bpe.development_corpus import NORMALIZER
from experiments.mom_bpe.diagnose import format_json, immutable, sha
from experiments.mom_bpe.optimization import verified_input
from experiments.mom_bpe.performance import checked_canonical, from_files
from experiments.mom_bpe.pilot import peak_rss_bytes
from experiments.mom_bpe.verify_frozen import load_rows
from mltok import registry
from mltok.lib.bpe_progress import canonical
from mltok.lib.canonical_bpb import TorchWindowNLL, canonical_bpb
from mltok.lib.exposure import account_exposure
from mltok.lib.orthographic_units import UNITIZERS
from mltok.lib.picky_backend import ARTIFACT, EventBackend
from mltok.snapshots import _lock, file_identity

BUDGETS = ("fixed_source_bytes", "fixed_training_tokens")
MODEL = {"type": "gpt", "size": "nano", "n_layer": 1, "n_head": 2, "d_model": 16}
CONTEXT = 16
BATCH = 32
SEED = 17
TOKEN_BUDGET = 512


def capped_text(text, budget):
    units, used = [], 0
    for unit in UNITIZERS["grapheme"](text):
        size = len(unit.encode())
        if used + size > budget:
            break
        units.append(unit)
        used += size
    return "".join(units)


def prepare(path, out):
    registry.load_plugins()
    files, data_sha = verified_input(path, NORMALIZER)
    norm = registry.make("normalizer", NORMALIZER)
    plan = {"development_sha256": data_sha, "schema": "mom-cpu-document-bpb-v1",
            "model": MODEL, "context": CONTEXT, "microbatch": BATCH, "seed": SEED, "fixed_token_budget": TOKEN_BUDGET,
            "lr": .003, "device": "cpu", "epochs": 1, "torch": torch.__version__, "policy": {
                "training_order": "manifest order; first two documents per source",
                "validation_order": "manifest order; first document per source",
                "document_canonical_caps": {"train": 1800, "validation": 1200},
                "context": "per-target sliding prefix inside the same document",
                "mask": "causal attention and right padding; last non-padding position scored",
                "BOS": "context only", "EOS": "one target per complete bounded excerpt",
            }, "driver_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))}
    with _lock(out / ".prepare.lock"):
        for split in ("train", "validation"):
            rows, seen = [], Counter()
            limit = 2 if split == "train" else 1
            cap = plan["policy"]["document_canonical_caps"][split]
            for row in from_files(files[split]):
                if seen[row["src"]] >= limit:
                    continue
                text = norm.normalize(capped_text(checked_canonical(row, norm), cap))
                if not text:
                    continue
                if len(text.encode()) > cap:
                    raise ValueError("Canonical prefix exceeded the bounded proxy cap")
                seen[row["src"]] += 1
                identity = sha(text.encode())
                rows.append({"id": f"{row['id']}:cpu-prefix-{identity[:12]}", "parent_id": row["id"],
                             "src": row["src"], "text": text, "canonical_sha256": identity,
                             "canonical_bytes": len(text.encode())})
            if not rows:
                raise ValueError("No bounded proxy examples")
            member = out / f"{split}.jsonl"
            immutable(member, b"".join(canonical(r) + b"\n" for r in rows))
            plan[split] = {"sha256": file_identity(member)[1], "rows": len(rows),
                           "canonical_bytes": sum(r["canonical_bytes"] for r in rows),
                           "parents": [r["parent_id"] for r in rows]}
        immutable(out / "plan.json", format_json(plan))
    return plan


class RightPaddedGPT(nn.Module):
    """A causal model never attends to right padding from a scored position."""

    def __init__(self, model, pad_id):
        super().__init__()
        self.model, self.pad_id = model, pad_id

    def forward(self, input_ids, attention_mask):
        if (input_ids.shape != attention_mask.shape or not torch.all((attention_mask == 0) | (attention_mask == 1))
                or torch.any(attention_mask[:, 1:] > attention_mask[:, :-1])
                or not torch.all(attention_mask[:, 0] == 1)
                or not torch.all(input_ids[attention_mask == 0] == self.pad_id)):
            raise ValueError("Expected a nonempty right-padded binary attention mask")
        return self.model(input_ids)


def windows(backend, rows, context):
    bos, eos = (backend.special_ids[t] for t in ("<s>", "</s>"))
    controls = set(backend.special_ids.values())
    for row in rows:
        encoding = backend.encode_offsets(row["text"])
        if encoding.text != row["text"] or backend.decode(encoding.ids) != row["text"] or controls.intersection(encoding.ids):
            raise ValueError("Proxy input is not exact canonical text or includes literal control IDs")
        source = [bos, *encoding.ids]
        for i, target in enumerate([*encoding.ids, eos]):
            yield source[max(0, i + 1 - context):i + 1], target


def forward_batch(model, examples):
    inputs, targets = zip(*examples)
    lengths = torch.tensor(list(map(len, inputs)))
    width = int(lengths.max())
    batch = torch.tensor([v + [model.pad_id] * (width - len(v)) for v in inputs])
    mask = torch.arange(width) < lengths[:, None]
    logits = model(batch, mask.long())
    chosen = logits[torch.arange(len(examples)), lengths - 1]
    loss = F.cross_entropy(chosen, torch.tensor(targets))
    return loss, batch.numel(), int(lengths.sum())


def save_step(out, identity, step, model, optimizer, compute):
    buffer = io.BytesIO()
    torch.save({"identity": identity, "step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": torch.random.get_rng_state(), "compute": compute}, buffer)
    data = buffer.getvalue()
    immutable(out / f"step-{step:05d}-{sha(data)}.pt", data)


def fit(out, identity, steps, new_model, schedule):
    saved = sorted(out.glob("step-*.pt"), reverse=True)
    for path in [*saved, None]:
        if path is None and saved:
            raise ValueError("No usable proxy checkpoint. All previous generations were retained.")
        torch.manual_seed(SEED)
        model = new_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        if path is None:
            start, compute = 0, [0, 0]
            break
        try:
            data = path.read_bytes()
            _, step_number, digest = path.stem.split("-")
            if digest != sha(data):
                continue
            state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            if state["identity"] != identity or not 0 < state["step"] == int(step_number) <= steps:
                continue
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            torch.random.set_rng_state(state["rng"])
            start, compute = state["step"], list(state["compute"])
            break
        except (RuntimeError, ValueError, KeyError, TypeError, OSError):
            continue
    model.train()
    for i, examples in enumerate(schedule[start:], start + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, allocated, valid = forward_batch(model, examples)
        loss.backward()
        optimizer.step()
        compute[0] += allocated
        compute[1] += valid
        if i == steps or i % 8 == 0:
            save_step(out, identity, i, model, optimizer, compute)
    return model.eval(), start, compute


def run(plan_path, tokenizer, out, budget):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    registry.load_plugins()
    plan_data = plan_path.read_bytes()
    plan = json.loads(plan_data)
    if plan["schema"] != "mom-cpu-document-bpb-v1" or plan["device"] != "cpu" or plan["model"] != MODEL or (
        plan["driver_sha256"] != sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))
        or plan["torch"] != torch.__version__ or plan["context"] != CONTEXT or plan["microbatch"] != BATCH
        or plan["lr"] != .003 or plan["seed"] != SEED or plan["fixed_token_budget"] != TOKEN_BUDGET
    ):
        raise ValueError("Proxy plan or runtime differs")
    rows = {}
    for split in ("train", "validation"):
        path = plan_path.parent / f"{split}.jsonl"
        if file_identity(path)[1] != plan[split]["sha256"]:
            raise ValueError("Proxy data SHA changed")
        rows[split] = list(load_rows(path))
    metadata = json.loads((tokenizer / "validation.json").read_bytes())
    model_sha = file_identity(tokenizer / ARTIFACT)[1]
    if (model_sha != metadata["fit"]["model_sha256"]
            or metadata["fit"]["identity"]["data_sha256"] != plan["development_sha256"]
            or any(metadata[key]["canonical_roundtrip"] != metadata[key]["docs"]
                   for key in ("validation", "fidelity_probes"))):
        raise ValueError("Use a fidelity-verified tokenizer from the matched development experiment")
    encoder = EventBackend.load(tokenizer)
    limit = plan["fixed_token_budget"] if budget == "fixed_training_tokens" else None
    exposure = account_exposure(encoder, rows["train"], limit)
    examples = list(islice(windows(encoder, rows["train"], CONTEXT), limit))
    if len(examples) != exposure["predicted_tokens"]:
        raise ValueError("Proxy training stream did not meet its exposure ledger")
    schedule = [examples[i:i+BATCH] for i in range(0, len(examples), BATCH)]
    identity = {"plan": sha(plan_data), "model_sha256": model_sha, "budget": budget,
                "proxy_plugin": registry.fingerprint("proxy_model", MODEL)}
    with _lock(out / ".fit.lock"):
        immutable(out / "identity.json", format_json(identity))
        result = out / "result.json"
        if result.exists():
            previous = json.loads(result.read_bytes())
            if previous["identity"] != identity:
                raise ValueError("Proxy result describes a different run")
            return previous

        def new_model():
            return RightPaddedGPT(registry.make("proxy_model", MODEL).build(encoder.vocab_size, CONTEXT),
                                  encoder.special_ids["<pad>"])

        start = perf_counter()
        model, resumed, compute = fit(out, identity, len(schedule), new_model, schedule)
        fit_s = perf_counter() - start
        scorer = TorchWindowNLL(model, context=CONTEXT, pad_id=model.pad_id, batch_size=BATCH)
        before_score = perf_counter()
        documents = {row["id"]: canonical_bpb(encoder, [row], scorer) for row in rows["validation"]}
        nats = math.fsum(r["nll_nats"] for r in documents.values())
        nbytes = sum(r["canonical_bytes"] for r in documents.values())
        scored = {"bpb": nats / math.log(2) / nbytes, "canonical_bytes": nbytes, "nll_nats": nats,
                  "predicted_tokens": sum(r["predicted_tokens"] for r in documents.values())}
        total_params = sum(p.numel() for p in model.parameters())
        embedding = model.model.wte.weight.numel()
        output = {"qualification": "BOUNDED_EXPLORATORY_CPU_PROXY", "identity": identity, "exposure": exposure,
                  "optimizer_steps": len(schedule), "resumed_steps": resumed, "train_wall_seconds": fit_s,
                  "score_wall_seconds": perf_counter() - before_score, "peak_rss_bytes": peak_rss_bytes(),
                  "compute": {"padded_training_input_positions": compute[0], "valid_training_context_positions": compute[1],
                              "embedding_parameters": embedding, "other_parameters": total_params - embedding},
                  "evaluation": scored, "per_document": documents}
        immutable(result, format_json(output))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    specs = parser.add_subparsers(dest="command", required=True)
    pre = specs.add_parser("prepare")
    pre.add_argument("--manifest", type=Path, required=True)
    pre.add_argument("--out", type=Path, required=True)
    fit_cmd = specs.add_parser("run")
    fit_cmd.add_argument("--plan", type=Path, required=True)
    fit_cmd.add_argument("--tokenizer", type=Path, required=True)
    fit_cmd.add_argument("--out", type=Path, required=True)
    fit_cmd.add_argument("--budget", choices=BUDGETS, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.manifest, args.out)))
    else:
        print(json.dumps(run(args.plan, args.tokenizer, args.out, args.budget)["evaluation"]))
