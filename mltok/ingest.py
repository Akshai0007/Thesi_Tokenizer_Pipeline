"""ENGINE data stage (not a plugin). Knows nothing about any particular dataset or filter:

  sources (plugins)  --iter_docs-->  raw/<id>.jsonl.gz
  raw  --filters (plugins), control-char strip, exact dedup, hash split-->  clean/{train,eval}/*.jsonl.gz

Config:
  sources: [{type: <source plugin>, id: <unique>, max_docs: N, ...plugin params}]
  filters: [{type: <filter plugin>, ...params}]      (applied in order; counted per filter)
  split:   {eval_permille: 5, shard_mb: 64, max_train_mb: 0}
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from . import registry
from .lib.io import iter_docs, list_shards

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def ingest_sources(sources: list, raw_dir, log=print) -> list[str]:
    """Run every source plugin once; finished raw files are reused (delete one to refetch)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for spec in sources:
        s = registry.spec_of(spec)
        sid, max_docs = s.get("id", s["type"]), int(s.get("max_docs", 0) or 0)
        out = raw_dir / f"{sid}.jsonl.gz"
        out_paths.append(str(out))
        if out.exists():
            log(f"[ingest] {sid}: exists, skipping")
            continue
        try:
            src = registry.make("source", {**s, "id": sid})
        except Exception as e:  # noqa: BLE001
            log(f"[ingest] {sid}: cannot build source plugin ({type(e).__name__}: {e})")
            out_paths.pop()
            continue
        n, tmp = 0, str(out) + ".tmp"
        try:
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                for d in src.iter_docs():
                    t = d.get("text")
                    if not t:
                        continue
                    f.write(json.dumps({"text": t, "src": d.get("src", sid)}, ensure_ascii=False) + "\n")
                    n += 1
                    if max_docs and n >= max_docs:
                        break
                    if n % 100_000 == 0:
                        log(f"[ingest] {sid}: {n} docs")
            os.replace(tmp, out)
            log(f"[ingest] {sid}: wrote {n} docs -> {out}")
        except Exception as e:  # noqa: BLE001 - a failing dataset must not abort the whole run
            log(f"[ingest] {sid}: FAILED after {n} docs ({type(e).__name__}: {e}); check the source params")
            Path(tmp).unlink(missing_ok=True)
            out_paths.pop()
    return out_paths


def clean_corpus(in_paths, out_dir, filters: list, *, eval_permille=5, shard_mb=64, max_train_mb=0, log=print) -> dict:
    """Filters (plugins) + exact dedup + deterministic hash split + sharding.
    Text is NOT normalized here, so normalization remains an experimental factor."""
    flt = [registry.make("filter", f) for f in filters]
    out_dir = Path(out_dir)
    for sub in ("train", "eval"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
        for old in (out_dir / sub).glob("*.jsonl.gz"):
            old.unlink()
    seen: set[bytes] = set()
    st: Counter = Counter()
    writers: dict[str, dict] = {}

    def writer(split):
        w = writers.get(split)
        if w is None or w["bytes"] >= shard_mb * 1_000_000:
            if w is not None:
                w["f"].close()
            idx = (w["idx"] + 1) if w else 0
            w = {"f": gzip.open(out_dir / split / f"{split}-{idx:04d}.jsonl.gz", "wt", encoding="utf-8"), "bytes": 0, "idx": idx}
            writers[split] = w
        return w

    train_bytes = 0
    for p in in_paths:
        for text, src in iter_docs(p):
            st["seen"] += 1
            text = _CTRL.sub("", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
            rejected = next((f.name for f in flt if not f.keep(text)), None)
            if rejected:
                st["filtered"] += 1
                st[f"filtered_by_{rejected}"] += 1
                continue
            h = hashlib.blake2b(text.encode("utf-8"), digest_size=12).digest()
            if h in seen:
                st["duplicate"] += 1
                continue
            seen.add(h)
            split = "eval" if int.from_bytes(h[:4], "big") % 1000 < eval_permille else "train"
            if split == "train" and max_train_mb and train_bytes >= max_train_mb * 1_000_000:
                continue
            w = writer(split)
            line = json.dumps({"text": text, "src": src}, ensure_ascii=False) + "\n"
            w["f"].write(line)
            nb = len(text.encode("utf-8"))
            w["bytes"] += len(line.encode("utf-8"))
            st[f"{split}_docs"] += 1
            st[f"{split}_bytes"] += nb
            st[f"{split}_docs_{src}"] += 1
            if split == "train":
                train_bytes += nb
    for w in writers.values():
        w["f"].close()
    stats = dict(st)
    (out_dir / "clean_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"[clean] {stats}")
    return stats
