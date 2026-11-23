"""Corpus I/O shared by the engine and plugins: readers, shard listing, word counting."""
from __future__ import annotations

import gzip
import hashlib
import multiprocessing as mp
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import json

from .marker import SP, full_normalize


def open_text(path, mode: str = "rt"):
    p = str(path)
    return gzip.open(p, mode, encoding="utf-8") if p.endswith(".gz") else open(p, mode, encoding="utf-8")


def iter_docs(path) -> Iterator[tuple[str, str]]:
    """Yield (text, source). `.jsonl[.gz]` -> {"text","src"}; `.txt` -> one doc per line."""
    p = str(path)
    jsonl = Path(p).name.endswith((".jsonl", ".jsonl.gz"))
    with open_text(p) as f:
        for line in f:
            if jsonl:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield d.get("text", ""), d.get("src", "unk")
            else:
                line = line.rstrip("\n")
                if line:
                    yield line, "txt"


def is_shard(path) -> bool:
    return Path(path).name.endswith((".jsonl", ".jsonl.gz", ".txt"))


def list_shards(d) -> list[str]:
    d = Path(d)
    if d.is_file():
        return [str(d)]
    return [str(p) for p in sorted(d.rglob("*"))
            if p.is_file() and is_shard(p) and not any(part.startswith(".") for part in p.relative_to(d).parts)]


def iter_texts(paths: Iterable[str], limit_bytes: int = 0) -> Iterator[str]:
    seen = 0
    for p in paths:
        for text, _ in iter_docs(p):
            yield text
            if limit_bytes:
                seen += len(text.encode("utf-8"))
                if seen >= limit_bytes:
                    return


def data_fingerprint(paths: list[str]) -> str:
    """Fingerprint the ordered shard contents, independent of gzip metadata."""
    digests = []
    for path in paths:
        digest = hashlib.sha256()
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        format_name = "jsonl" if Path(path).name.endswith((".jsonl", ".jsonl.gz")) else "text"
        digests.append((format_name, digest.hexdigest()))
    return hashlib.sha256(json.dumps(digests).encode("ascii")).hexdigest()


# ----------------------------------------------------------------------- word statistics
def _count_shard(args) -> Counter:
    path, norm_spec, pre_spec, max_docs, max_local = args
    from .. import registry

    norm = registry.make("normalizer", norm_spec)
    pre = registry.make("pretokenizer", pre_spec)
    c: Counter = Counter()
    for i, (text, _) in enumerate(iter_docs(path)):
        if max_docs and i >= max_docs:
            break
        c.update(pre.split(full_normalize(norm, text)))
        if max_local and len(c) > max_local:
            c = Counter({w: n for w, n in c.items() if n > 1})
    return c


def count_words(paths, norm_spec, pre_spec, *, workers=1, max_docs_per_shard=0, max_local_types=30_000_000, log=print) -> Counter:
    tasks = [(p, norm_spec, pre_spec, max_docs_per_shard, max_local_types) for p in paths]
    total: Counter = Counter()
    if workers <= 1 or len(tasks) == 1:
        for t in tasks:
            total.update(_count_shard(t))
    else:
        with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            for i, c in enumerate(pool.imap_unordered(_count_shard, tasks), 1):
                total.update(c)
                log(f"[count] shard {i}/{len(tasks)} -> {len(total):,} word types")
    return total


def select_alphabet(word_counts: dict[str, int], coverage: float = 0.99999, forced: list[str] | None = None) -> list[str]:
    """`forced` characters are always in the alphabet; further characters are added by frequency
    until `coverage` of character mass is reached; the rest is handled by byte fallback."""
    cc: Counter = Counter()
    for w, n in word_counts.items():
        for ch in w:
            cc[ch] += n
    base = [SP] + [chr(c) for c in range(0x21, 0x7F)] + ["\n", "\t"]
    forced = list(dict.fromkeys(base + list(forced or [])))
    fs = set(forced)
    total = sum(cc.values()) or 1
    covered = sum(cc[c] for c in forced if c in cc)
    extra: list[str] = []
    for ch, n in cc.most_common():
        if covered / total >= coverage:
            break
        if ch in fs or unicodedata.category(ch).startswith("C"):
            continue
        extra.append(ch)
        covered += n
    return forced + extra
