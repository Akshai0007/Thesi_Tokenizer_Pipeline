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
from .lib.atomic import atomic_writer
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


def _source_key(path) -> str:
    return Path(path).name.split(".")[0]  # raw files are named <source id>.jsonl.gz


def clean_corpus(in_paths, out_dir, filters: list, *, eval_permille=5, shard_mb=64, max_train_mb=0,
                 balance=True, weights=None, log=print) -> dict:
    """Filters (plugins) + exact dedup + deterministic hash split + sharding.
    Text is NOT normalized here, so normalization remains an experimental factor.

    Source mixing (matters only when `max_train_mb` caps the training set):
      balance=True   sources are INTERLEAVED, always taking the next document from the source that is furthest
                     below its target share (accepted train bytes / weight). `weights` maps source id -> weight
                     (default: equal). A source that runs out simply stops contributing; the others fill the cap.
      balance=False  files are read one after another, so a cap is filled by the FIRST source alone (this is
                     what silently produced a 100%-web training set before; kept only for reproducing old runs).
    """
    flt = [registry.make("filter", f) for f in filters]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = out_dir / ".mltok-cleaning"
    with atomic_writer(pending) as stream:
        stream.write(b"cleaning in progress\n")
    for sub in ("train", "eval"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
        for old in (out_dir / sub).glob("*.jsonl.gz"):
            old.unlink()
    seen: set[bytes] = set()
    st: Counter = Counter()
    writers: dict[str, dict] = {}
    cap = max_train_mb * 1_000_000

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

    def process(text: str, src: str) -> int:
        """Handle one raw document; returns the train bytes it added (0 if filtered/duplicate/eval/capped)."""
        nonlocal train_bytes
        st["seen"] += 1
        text = _CTRL.sub("", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
        rejected = next((f.name for f in flt if not f.keep(text)), None)
        if rejected:
            st["filtered"] += 1
            st[f"filtered_by_{rejected}"] += 1
            return 0
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=12).digest()
        if h in seen:
            st["duplicate"] += 1
            return 0
        seen.add(h)
        split = "eval" if int.from_bytes(h[:4], "big") % 1000 < eval_permille else "train"
        if split == "train" and cap and train_bytes >= cap:
            return 0
        w = writer(split)
        line = json.dumps({"text": text, "src": src}, ensure_ascii=False) + "\n"
        w["f"].write(line)
        nb = len(text.encode("utf-8"))
        w["bytes"] += len(line.encode("utf-8"))
        st[f"{split}_docs"] += 1
        st[f"{split}_bytes"] += nb
        st[f"{split}_docs_{src}"] += 1
        if split == "train":
            st[f"train_bytes_{src}"] += nb
            train_bytes += nb
            return nb
        return 0

    paths = list(in_paths)
    if not balance or len(paths) == 1:
        if cap and len(paths) > 1:
            log("[clean] WARNING: balance=False with a training cap - the first source will fill the cap alone")
        for p in paths:
            for text, src in iter_docs(p):
                process(text, src)
    else:
        keys = [_source_key(p) for p in paths]
        w = {k: float((weights or {}).get(k, 1.0)) for k in keys}
        if any(x <= 0 for x in w.values()):
            raise ValueError("source weights must be > 0 (remove a source from `sources` to exclude it)")
        its = {k: iter_docs(p) for k, p in zip(keys, paths)}
        acc = {k: 0.0 for k in keys}       # accepted train bytes per source
        consumed = {k: 0 for k in keys}    # raw docs consumed per source (keeps eval proportional after the cap)
        active = list(keys)
        while active:
            if cap and train_bytes >= cap:  # training set full: keep reading only to fill the eval split evenly
                k = min(active, key=lambda x: (consumed[x] / w[x], keys.index(x)))
            else:
                k = min(active, key=lambda x: (acc[x] / w[x], keys.index(x)))
            try:
                text, src = next(its[k])
            except StopIteration:
                active.remove(k)
                continue
            consumed[k] += 1
            acc[k] += process(text, src)
    for wr in writers.values():
        wr["f"].close()
    stats = dict(st)
    tot = sum(v for k, v in stats.items() if k.startswith("train_bytes_")) or 1
    stats["train_share"] = {k[len("train_bytes_"):]: round(v / tot, 3) for k, v in stats.items() if k.startswith("train_bytes_")}
    stats["shards"] = sorted(p.relative_to(out_dir).as_posix()
                             for split in ("train", "eval") for p in (out_dir / split).glob("*.jsonl.gz"))
    with atomic_writer(out_dir / "clean_stats.json") as stream:
        stream.write(json.dumps(stats, indent=2).encode("utf-8"))
    pending.unlink()
    log(f"[clean] {stats}")
    return stats
