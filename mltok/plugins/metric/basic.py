"""Compression metrics: fertility (tokens/word), bytes & chars per token, Renyi efficiency,
vocabulary use, <unk> rate, per-source fertility, encode throughput. Denominators are the RAW text."""
import time
from collections import defaultdict

import numpy as np

from mltok.contracts import Metric
from mltok.registry import register


def renyi_efficiency(counts, vocab_size, alpha=2.5):
    c = np.asarray(counts)[np.asarray(counts) > 0].astype(np.float64)
    p = c / c.sum()
    return float((np.log2((p**alpha).sum()) / (1 - alpha)) / np.log2(vocab_size))


@register("metric", "basic")
class Basic(Metric):
    def compute(self, backend, docs):
        texts = [t for t, _ in docs]
        t0 = time.time()
        ids = backend.encode_batch(texts)
        dt = max(time.time() - t0, 1e-9)
        lens = np.array([len(x) for x in ids], dtype=np.int64)
        words = np.array([len(t.split()) for t in texts], dtype=np.int64)
        chars = np.array([len(t) for t in texts], dtype=np.int64)
        nbytes = np.array([len(t.encode("utf-8")) for t in texts], dtype=np.int64)
        n_tok = int(lens.sum())
        flat = np.fromiter((i for x in ids for i in x), dtype=np.int64, count=n_tok)
        counts = np.bincount(flat, minlength=backend.vocab_size) if n_tok else np.zeros(backend.vocab_size)
        unk = int((flat == backend.unk_id).sum()) if backend.unk_id is not None and n_tok else 0
        by_src = defaultdict(lambda: [0, 0])
        for (_, s), n, w in zip(docs, lens, words):
            by_src[s][0] += int(n)
            by_src[s][1] += int(w)
        return {
            "vocab_size": backend.vocab_size, "docs": len(texts), "words": int(words.sum()),
            "bytes": int(nbytes.sum()), "tokens": n_tok,
            "fertility": n_tok / max(int(words.sum()), 1),
            "bytes_per_token": int(nbytes.sum()) / max(n_tok, 1),
            "chars_per_token": int(chars.sum()) / max(n_tok, 1),
            "renyi_eff": renyi_efficiency(counts, backend.vocab_size) if n_tok else 0.0,
            "vocab_used": float((counts > 0).mean()),
            "unk_rate": unk / max(n_tok, 1),
            "tok_per_sec": n_tok / dt,
            "fertility_by_source": {s: v[0] / max(v[1], 1) for s, v in by_src.items()},
        }
