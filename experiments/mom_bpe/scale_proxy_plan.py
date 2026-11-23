"""Verify a candidate and produce a development-only BPB exposure ledger."""

import argparse
import json
from pathlib import Path

from experiments.mom_bpe.diagnose import format_json, immutable
from experiments.mom_bpe.scale_data import document_bytes, within_bytes
from experiments.mom_bpe.scale_fit import VOCAB, checked_run
from experiments.mom_bpe.verify_frozen import load_rows
from mltok.lib.bpe_core import DEFAULT_SPECIALS
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


def prepare(manifest, tokenizer, out, *, source_bytes, predicted_tokens):
    if not 0 < source_bytes <= 1_000_000 or not 0 < predicted_tokens <= 250_000:
        raise ValueError("Expected bounded integration budgets of at most 1 MB and 250,000 targets")
    plan, digest, files, normalize, _, config = checked_run(manifest, allow_large=False)
    fit_path, model_path = tokenizer / "fit.json", tokenizer / ARTIFACT
    fit = json.loads(fit_path.read_bytes())
    identity = fit["identity"]
    if (fit["occupied_vocab"] != VOCAB or identity["manifest_sha256"] != digest
            or identity["validation_stream_sha256"] != plan["splits"]["validation"]["canonical_stream_sha256"]
            or identity["config_sha256"] != config or file_identity(model_path)[1] != fit["model_sha256"]):
        raise ValueError("Proxy integration requires a complete 48K tokenizer trained on the matching plan")
    backend = EventBackend.load(tokenizer)
    program = EventProgram(backend.artifact["initial"], backend.artifact["events"])
    if (backend.vocab_size != VOCAB or backend.special_ids != dict(zip(DEFAULT_SPECIALS, range(len(DEFAULT_SPECIALS))))
            or len(program.active) + len(backend.special_ids) + len(backend.fallback_ids) != VOCAB):
        raise ValueError("Unexpected vocabulary or control-token policy")

    def rows(split):
        for row in load_rows(files[split][0]):
            _, canonical = document_bytes(row, normalize.normalize)
            encoded = backend.encode_offsets(row["text"])
            if (encoded.text != canonical.decode() or backend.decode(encoded.ids) != encoded.text
                    or set(encoded.ids).intersection(backend.special_ids.values())):
                raise ValueError("Proxy source is not losslessly representable with non-control text IDs")
            yield row

    fixed_source = account_exposure(backend, within_bytes(rows("train"), source_bytes))
    if not fixed_source["full_documents"]:
        raise ValueError("Fixed source budget selected no documents")
    fixed_tokens = account_exposure(backend, rows("train"), predicted_tokens)
    eval_exposure = account_exposure(backend, rows("validation"))
    if eval_exposure["full_canonical_bytes"] != plan["splits"]["validation"]["canonical_bytes"]:
        raise ValueError("Proxy evaluation exposure differs from the manifest")
    result = {"schema": "mom-48k-proxy-exposure-v1", "qualification": "DEVELOPMENT_ONLY",
              "fit_sha256": file_identity(fit_path)[1], "tokenizer_sha256": fit["model_sha256"], "arm": identity["arm"],
              "manifest_sha256": digest, "validation_stream_sha256": identity["validation_stream_sha256"],
              "source_byte_budget": source_bytes, "predicted_token_budget": predicted_tokens,
              "fixed_source_bytes": fixed_source, "fixed_training_tokens": fixed_tokens, "validation": eval_exposure,
              "integration_model": {"layers": 7, "heads": 7, "width": 448, "context": 512, "seeds": [17],
                                    "parameters": parameter_counts(7, 448, 512)},
              "confirmation_model": {"layers": 12, "heads": 12, "width": 768, "context": 512, "seeds": [17, 23, 41],
                                     "parameters": parameter_counts(12, 768, 512)}}
    immutable(out, format_json(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-bytes", type=int, default=100_000)
    parser.add_argument("--predicted-tokens", type=int, default=8192)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.tokenizer, args.out, source_bytes=args.source_bytes,
                             predicted_tokens=args.predicted_tokens), sort_keys=True))
