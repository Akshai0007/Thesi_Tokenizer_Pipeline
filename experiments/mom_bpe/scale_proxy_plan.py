"""Verify a candidate and produce a development-only BPB exposure ledger."""

import argparse
import json
import math
from itertools import islice
from pathlib import Path

from experiments.mom_bpe.diagnose import format_json, immutable
from experiments.mom_bpe.scale_data import document_bytes, within_bytes
from experiments.mom_bpe.scale_fit import VOCAB, checked_run
from experiments.mom_bpe.scale_mlbpe import FILES as NATIVE_FILES, MatchedMLBPE
from experiments.mom_bpe.verify_frozen import load_rows
from mltok import registry
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.canonical_bpb import canonical_bpb
from mltok.lib.exposure import account_exposure
from mltok.lib.picky_backend import ARTIFACT, EventBackend
from mltok.lib.picky_events import EventProgram
from mltok.snapshots import file_identity


def parameter_counts(layers, width, context):
    hidden = int(8 * width / 3 / 64 + 1) * 64
    tied = VOCAB * width
    positions = context * width
    transformer = layers * (2 * width + 4 * width**2 + 3 * hidden * width) + width
    return {"tied_embeddings": tied, "position_embeddings": positions, "transformer": transformer,
            "non_embedding": transformer, "total": tied + positions + transformer}


class NativeProxyBackend(MatchedMLBPE):
    def __init__(self, folder, normalizer):
        super().__init__(folder)
        self.expected = normalizer.normalize
        self.vocab_size = self.occupied

    def encode_offsets(self, text):
        return super().encode_offsets(self.expected(text))


def verified_backend(tokenizer, fit, normalizer):
    if fit["identity"]["arm"] == "M48":
        members = {name: file_identity(tokenizer / name)[1] for name in NATIVE_FILES}
        if members != fit["files_sha256"]:
            raise ValueError("Matched MLBPE artifact hashes differ")
        backend = NativeProxyBackend(tokenizer, normalizer)
        return backend, members["tokenizer.json"], backend.occupied
    model_sha = file_identity(tokenizer / ARTIFACT)[1]
    if model_sha != fit["model_sha256"]:
        raise ValueError("Event tokenizer artifact hash differs")
    backend = EventBackend.load(tokenizer)
    program = EventProgram(backend.artifact["initial"], backend.artifact["events"])
    occupied = len(program.active) + len(backend.special_ids) + len(backend.fallback_ids)
    return backend, model_sha, occupied


def document_local_checks(backend, rows, plan):
    """Exercise the actual target windows and score accounting without a training step."""
    import torch

    from experiments.mom_bpe.optimization_proxy import RightPaddedGPT, windows

    integration = plan["integration_model"]
    context, batch_size = integration["context"], 4
    spec = {"type": "gpt", "n_layer": integration["layers"], "n_head": integration["heads"],
            "d_model": integration["width"]}
    pad, bos, eos = (backend.special_ids[token] for token in ("<pad>", "<s>", "</s>"))
    with torch.device("meta"):
        model = RightPaddedGPT(registry.make("proxy_model", spec).build(backend.vocab_size, context), pad)
    if sum(parameter.numel() for parameter in model.parameters()) != integration["parameters"]["total"]:
        raise ValueError("The actual GPT architecture differs from its planned parameter count")

    def canonical_rows(documents):
        for row in documents:
            yield {**row, "text": backend.expected(row["text"])}

    budgets = {
        "fixed_source_bytes": (within_bytes(rows("train"), plan["source_byte_budget"]), None),
        "fixed_training_tokens": (rows("train"), plan["predicted_token_budget"]),
    }
    ledger = {}
    for key, (documents, limit) in budgets.items():
        stream = islice(windows(backend, canonical_rows(documents), context), limit)
        targets = eos_targets = positions = padded = batches = 0
        while batch := list(islice(stream, batch_size)):
            targets += len(batch)
            eos_targets += sum(target == eos for _, target in batch)
            positions += sum(len(inputs) for inputs, _ in batch)
            padded += max(len(inputs) for inputs, _ in batch) * len(batch)
            batches += 1
        if (targets, eos_targets) != (plan[key]["predicted_tokens"], plan[key]["predicted_eos"]):
            raise ValueError("Document-local target windows differ from the exposure ledger")
        ledger[key] = {"targets": targets, "eos_targets": eos_targets, "context_positions": positions,
                       "right_padding_positions": padded - positions, "microbatches": batches}

    def one_bit_per_target(inputs, targets):
        if inputs[0] != bos or targets[-1] != eos or inputs[1:] != targets[:-1]:
            raise ValueError("BPB target and context sequences differ from the document policy")
        return [math.log(2)] * len(targets)

    metrics = canonical_bpb(backend, rows("validation"), one_bit_per_target)
    nbytes, ntokens = metrics["canonical_bytes"], metrics["predicted_tokens"]
    expected = plan["validation"]
    if ((nbytes, ntokens) != (expected["full_canonical_bytes"], expected["predicted_tokens"])
            or not math.isclose(metrics["bpb"], ntokens / nbytes, rel_tol=1e-14)):
        raise ValueError("Document-local BPB denominator or target count differs")
    return {"training_started": False, "context": context, "microbatch": batch_size, "train": ledger,
            "validation": {"documents": metrics["documents"], "canonical_bytes": nbytes, "targets": ntokens},
            "synthetic_denominator_check": "one bit per target; no model-backed score"}


def prepare(manifest, tokenizer, out, *, source_bytes, predicted_tokens, verify_windows=False):
    if not 0 < source_bytes <= 1_000_000 or not 0 < predicted_tokens <= 250_000:
        raise ValueError("Expected bounded integration budgets of at most 1 MB and 250,000 targets")
    plan, digest, files, normalize, _, config = checked_run(manifest, allow_large=False)
    fit_path = tokenizer / "fit.json"
    fit = json.loads(fit_path.read_bytes())
    identity = fit["identity"]
    if (fit["occupied_vocab"] != VOCAB or identity["manifest_sha256"] != digest
            or identity["validation_stream_sha256"] != plan["splits"]["validation"]["canonical_stream_sha256"]
            or identity["config_sha256"] != config):
        raise ValueError("Proxy integration requires a complete 48K tokenizer trained on the matching plan")
    backend, model_sha, occupied = verified_backend(tokenizer, fit, normalize)
    if (backend.vocab_size != VOCAB or backend.special_ids != dict(zip(DEFAULT_SPECIALS, range(len(DEFAULT_SPECIALS))))
            or occupied != VOCAB or len(backend.fallback_ids) != 256):
        raise ValueError("Unexpected vocabulary or control-token policy")

    def rows(split):
        return load_rows(files[split][0])

    def checked(documents):
        for row in documents:
            _, canonical = document_bytes(row, normalize.normalize)
            encoded = backend.encode_offsets(row["text"])
            if (encoded.text != canonical.decode() or backend.decode(encoded.ids) != encoded.text
                    or set(encoded.ids).intersection(backend.special_ids.values())):
                raise ValueError("Proxy source is not losslessly representable with non-control text IDs")
            yield row

    fixed_source = account_exposure(backend, checked(within_bytes(rows("train"), source_bytes)))
    if not fixed_source["full_documents"]:
        raise ValueError("Fixed source budget selected no documents")
    fixed_tokens = account_exposure(backend, checked(rows("train")), predicted_tokens)
    partial = fixed_tokens["partial"]
    if isinstance(backend, NativeProxyBackend) and partial and partial["canonical_bytes"] is not None:
        row = next(row for row in rows("train") if row["id"] == partial["document_id"])
        encoding = backend.encode_offsets(row["text"])
        prefix = backend.decode(encoding.ids[:partial["predicted_text_tokens"]]).encode()
        if prefix != encoding.text.encode()[:partial["canonical_bytes"]]:
            partial["canonical_bytes"] = None
    eval_exposure = account_exposure(backend, checked(rows("validation")))
    if eval_exposure["full_canonical_bytes"] != plan["splits"]["validation"]["canonical_bytes"]:
        raise ValueError("Proxy evaluation exposure differs from the manifest")
    result = {"schema": "mom-48k-proxy-exposure-v1", "qualification": "DEVELOPMENT_ONLY",
              "fit_sha256": file_identity(fit_path)[1], "tokenizer_sha256": model_sha, "arm": identity["arm"],
              "manifest_sha256": digest, "validation_stream_sha256": identity["validation_stream_sha256"],
              "source_byte_budget": source_bytes, "predicted_token_budget": predicted_tokens,
              "fixed_source_bytes": fixed_source, "fixed_training_tokens": fixed_tokens, "validation": eval_exposure,
              "integration_model": {"layers": 7, "heads": 7, "width": 448, "context": 512, "seeds": [17],
                                    "parameters": parameter_counts(7, 448, 512)},
              "confirmation_model": {"layers": 12, "heads": 12, "width": 768, "context": 512, "seeds": [17, 23, 41],
                                     "parameters": parameter_counts(12, 768, 512)}}
    if verify_windows:
        result["document_local_checks"] = document_local_checks(backend, rows, result)
    immutable(out, format_json(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-bytes", type=int, default=100_000)
    parser.add_argument("--predicted-tokens", type=int, default=8192)
    parser.add_argument("--verify-windows", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.tokenizer, args.out, source_bytes=args.source_bytes,
                             predicted_tokens=args.predicted_tokens, verify_windows=args.verify_windows), sort_keys=True))
