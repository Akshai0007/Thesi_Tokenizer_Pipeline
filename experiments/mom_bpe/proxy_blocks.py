"""Verified token caches and ordered document-local exposure for the 48K proxy."""

import hashlib
import json
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import tokenizers

from experiments.mom_bpe.diagnose import format_json, immutable, sha
from experiments.mom_bpe.frontend_analysis import token_payloads
from experiments.mom_bpe.scale_data import code_digest, document_bytes
from experiments.mom_bpe.scale_fit import VOCAB, checked_run
from experiments.mom_bpe.scale_proxy_plan import verified_backend
from experiments.mom_bpe.verify_frozen import checked_member, load_rows
from mltok import registry
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import canonical
from mltok.lib.atomic import atomic_writer
from mltok.snapshots import _lock, file_identity


ARMS = ("M48", "T5", "T8", "F-space")
MODES = ("fixed_source_bytes", "fixed_training_tokens")
BOUNDARY = 1 << 31
POLICY = "document blocks; reset positions, carry one preceding token; BOS first input, EOS complete-only; right padding"


def publish(path, data):
    digest = sha(canonical(data))
    immutable(path, format_json({**data, "sha256": digest}))
    return digest


def receipt(path):
    value = json.loads(path.read_bytes())
    digest = value.pop("sha256")
    if sha(canonical(value)) != digest:
        raise ValueError(f"Receipt checksum differs for {path.name}")
    return value, digest


def member(base, file):
    size, digest = file_identity(file)
    return {"path": file.relative_to(base).as_posix(), "size": size, "sha256": digest}


def write_split(out, split, rows, backend, normalize, native, byte_limit):
    folder = out / f"{split}-{uuid4().hex[:12]}"
    folder.mkdir()
    id_path, end_path = folder / "ids.bin", folder / "ends.bin"
    docs, offset, spent = [], 0, 0
    controls = set(backend.special_ids.values())
    with atomic_writer(id_path, overwrite=False) as ids_file, atomic_writer(end_path, overwrite=False) as ends_file:
        for row in rows:
            size = row["canonical_bytes"]
            if byte_limit is not None and spent + size > byte_limit:
                break
            raw, data = document_bytes(row, normalize)
            enc = backend.encode_offsets(row["text"])
            if (enc.text.encode() != data or backend.decode(enc.ids) != enc.text
                    or any(type(i) is not int or not 0 <= i < backend.vocab_size or i in controls for i in enc.ids)):
                raise ValueError("A document failed canonical encoding or token ID validation")
            ends = []
            for start, end, _, _, _, payload in token_payloads(enc, backend, native):
                if payload != data[start:end]:
                    raise ValueError("Token payload differs from canonical UTF-8")
                boundary = end == size or data[end] & 0xc0 != 0x80
                ends.append(end | (BOUNDARY if boundary else 0))
            if size >= BOUNDARY or not size or not ends or ends[-1] != size | BOUNDARY:
                raise ValueError("Token payloads did not cover an eligible document")
            ids_file.write(np.asarray(enc.ids, dtype="<u2").tobytes())
            ends_file.write(np.asarray(ends, dtype="<u4").tobytes())
            docs.append({"id": row["id"], "canonical_sha256": row["canonical_sha256"], "canonical_bytes": size,
                         "raw_bytes": len(raw), "start": offset, "length": len(enc.ids)})
            offset += len(enc.ids)
            spent += size
    if not docs:
        raise ValueError(f"Cache selection includes no {split} documents")
    data = {"documents": docs, "canonical_bytes": spent, "ids": member(out, id_path), "ends": member(out, end_path)}
    return publish(out / f"{split}.json", data)


class EncodedSplit:
    def __init__(self, out, split, digest):
        data, actual = receipt(out / f"{split}.json")
        if actual != digest:
            raise ValueError("Encoded split differs from its cache identity")
        self.docs, self.canonical_bytes = data["documents"], data["canonical_bytes"]
        starts = 0
        for doc in self.docs:
            if doc["start"] != starts or min(doc["canonical_bytes"], doc["length"]) < 1:
                raise ValueError("Invalid cached document coordinates")
            starts += doc["length"]
        if (len({doc["id"] for doc in self.docs}) != len(self.docs) or not starts
                or sum(doc["canonical_bytes"] for doc in self.docs) != self.canonical_bytes
                or data["ids"]["size"] != starts * 2 or data["ends"]["size"] != starts * 4):
            raise ValueError("Cached split size or document index differs")
        self.ids = np.memmap(checked_member(out, data["ids"]), mode="r", dtype="<u2")
        self.ends = np.memmap(checked_member(out, data["ends"]), mode="r", dtype="<u4")

    def endpoint(self, index, consumed):
        doc = self.docs[index]
        if consumed == 0:
            return 0, True
        if consumed >= doc["length"]:
            return doc["canonical_bytes"], True
        value = int(self.ends[doc["start"] + consumed - 1])
        return value & (BOUNDARY - 1), bool(value & BOUNDARY)


def prepare_cache(manifest, tokenizer, root, arm, *, context=512, train_prefix_bytes=None):
    if arm not in ARMS or not 1 <= context <= 512 or (train_prefix_bytes is not None and train_prefix_bytes < 1):
        raise ValueError("Expected a supported arm, context and optional positive cache cap")
    plan, digest, files, normalizer, _, config = checked_run(manifest, allow_large=False)
    fit_path = tokenizer / "fit.json"
    fit = json.loads(fit_path.read_bytes())
    fit_id = fit["identity"]
    streams = {name: plan["splits"][name]["canonical_stream_sha256"] for name in ("train", "validation")}
    if (fit_id["arm"] != arm or fit_id["manifest_sha256"] != digest or fit_id["config_sha256"] != config
            or any(fit_id[f"{split}_stream_sha256"] != stream for split, stream in streams.items())):
        raise ValueError("Tokenizer fit does not match the verified arm and dataset")
    backend, model_sha, occupied = verified_backend(tokenizer, fit, normalizer)
    specials = dict(zip(DEFAULT_SPECIALS, range(len(DEFAULT_SPECIALS))))
    if (backend.vocab_size != VOCAB or fit["occupied_vocab"] != VOCAB or occupied != VOCAB
            or backend.special_ids != specials or len(backend.fallback_ids) != 256):
        raise ValueError("Expected the shared 48K vocabulary and control policy")
    identity = {"schema": 1, "arm": arm, "manifest_sha256": digest, "streams": streams,
                "tokenizer_sha256": model_sha, "fit_sha256": file_identity(fit_path)[1],
                "normalizer": registry.fingerprint("normalizer", {"type": normalizer.name, **normalizer.params}),
                "preprocessing_fit_identity": fit_id, "tokenizers_version": tokenizers.__version__,
                "context": context, "policy": POLICY,
                "specials": specials, "vocab": VOCAB, "train_prefix_bytes": train_prefix_bytes,
                "cache_code": code_digest(__file__), "payload_code": code_digest(token_payloads.__code__.co_filename)}
    out = root / (arm + "-" + sha(canonical(identity)))
    with _lock(out / ".prepare.lock"):
        result = out / "cache.json"
        if result.exists():
            previous, _ = receipt(result)
            if previous["identity"] != identity:
                raise ValueError("Existing cache has a different identity")
            for split, checksum in previous["splits"].items():
                EncodedSplit(out.resolve(), split, checksum)
            return out
        splits = {}
        for split in streams:
            path = out / f"{split}.json"
            if path.exists():
                _, checksum = receipt(path)
            else:
                limit = train_prefix_bytes if split == "train" else None
                checksum = write_split(out, split, load_rows(files[split][0]), backend, normalizer.normalize,
                                       arm == "M48", limit)
            cached = EncodedSplit(out.resolve(), split, checksum)
            if split == "validation" or train_prefix_bytes is None:
                expected = plan["splits"][split]
                if (cached.canonical_bytes, len(cached.docs)) != (expected["canonical_bytes"], expected["rows"]):
                    raise ValueError("Cached split does not match the full source")
            splits[split] = checksum
        publish(result, {"identity": identity, "splits": splits})
    return out


@dataclass(frozen=True)
class Block:
    doc: int
    start: int
    stop: int
    source_bytes: int

    @property
    def targets(self):
        return self.stop - self.start


def selection(split, mode, budget):
    """Select a whole-document byte prefix or an exact target prefix."""
    if mode not in MODES or type(budget) is not int or budget < 1:
        raise ValueError("Expected a supported exposure mode and a positive budget")
    available = (split.canonical_bytes if mode == MODES[0] else
                 sum(doc["length"] + 1 for doc in split.docs))
    if available < budget:
        raise ValueError(f"Cache cannot meet {mode} budget {budget}; only {available} available")
    chosen = []
    totals = {"full_documents": 0, "full_canonical_bytes": 0, "full_raw_bytes": 0,
              "predicted_tokens": 0, "predicted_eos": 0, "partial": None}
    ordered = hashlib.sha256()
    for i, doc in enumerate(split.docs):
        tokens, size = doc["length"] + 1, doc["canonical_bytes"]
        if mode == MODES[0] and totals["full_canonical_bytes"] + size > budget:
            break
        take = tokens if mode == MODES[0] else min(tokens, budget - totals["predicted_tokens"])
        if not take:
            break
        chosen.append((i, take))
        totals["predicted_tokens"] += take
        if take == tokens:
            totals["full_documents"] += 1
            totals["full_canonical_bytes"] += size
            totals["full_raw_bytes"] += doc["raw_bytes"]
            totals["predicted_eos"] += 1
            ordered.update(canonical([doc["id"], doc["canonical_sha256"], size]) + b"\n")
        else:
            end, scalar = split.endpoint(i, take)
            totals["partial"] = {"document_id": doc["id"], "canonical_sha256": doc["canonical_sha256"],
                                 "predicted_text_tokens": take, "canonical_bytes": end, "ends_on_utf8_scalar": scalar}
    if not chosen:
        raise ValueError("Budget selects no complete document or targets")
    totals["ordered_complete_documents_sha256"] = ordered.hexdigest()
    totals["canonical_bytes"] = totals["full_canonical_bytes"] + (totals["partial"] or {}).get("canonical_bytes", 0)
    return chosen, totals


def blocks(split, chosen, context):
    for index, count in chosen:
        for start in range(0, count, context):
            stop = min(start + context, count)
            first = split.endpoint(index, start)[0]
            yield Block(index, start, stop, split.endpoint(index, stop)[0] - first)


def sequences(split, block, bos, eos):
    doc = split.docs[block.doc]
    base, length = doc["start"], doc["length"]
    targets = split.ids[base + block.start:base + min(block.stop, length)].tolist()
    if block.stop > length:
        targets.append(eos)
    previous = int(split.ids[base + block.start - 1]) if block.start else bos
    return [previous, *targets[:-1]], targets
