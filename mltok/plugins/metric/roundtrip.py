"""Lossless round-trip: share of sampled docs where decode(encode(x)) == backend.expected(x)."""
from mltok.contracts import Metric
from mltok.registry import register


@register("metric", "roundtrip")
class RoundTrip(Metric):
    def __init__(self, docs=2000):
        self.docs = docs

    def compute(self, backend, docs):
        texts = [t for t, _ in docs[: self.docs]]
        if not texts or backend.tokens(texts[0]) is None:
            return {}
        enc = backend.encode_batch(texts)
        ok = sum(backend.decode(e) == backend.expected(t) for e, t in zip(enc, texts))
        return {"roundtrip_ok": ok / len(texts)}
