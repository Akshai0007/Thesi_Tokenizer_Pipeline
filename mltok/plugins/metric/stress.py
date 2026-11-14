"""Robustness: lossless round-trip on a fixed stress set that eval corpora rarely contain
(emoji, other scripts, ZWNJ/ZWJ, old-style chillu, control characters, odd whitespace, very long words).
Key: stress_roundtrip_ok (share of stress strings that round-trip exactly)."""
from mltok.contracts import Metric
from mltok.registry import register

STRESS = [
    "emoji 😀🎉 and symbols ©®™ € ₹",
    "other scripts: हिन्दी தமிழ் తెలుగు ಕನ್ನಡ 日本語 한국어 العربية русский",
    "zwnj: സദ്\u200cവാരം zwj: ക്\u200dഷ chillu-old: \u0d28\u0d4d\u200d \u0d30\u0d4d\u200d",
    "tabs\tand\nnewlines\n\n\n and   many    spaces ",
    "ൊ ോ ൌ \u0d46\u0d3e \u0d47\u0d3e \u0d46\u0d57",
    "digits 0123456789 ൦൧൨൩൪൫ and 3.14159, 1,00,000",
    "ഇത്" * 200,
    "a" * 500,
    "\u200b\ufeff\u00ad invisible junk \r\n crlf",
    "",
    " ",
]


@register("metric", "stress")
class Stress(Metric):
    def compute(self, backend, docs):
        if backend.tokens("x") is None:  # count-only Hub baselines
            return {}
        enc = backend.encode_batch(STRESS)
        ok = sum(backend.decode(e) == backend.expected(t) for e, t in zip(enc, STRESS))
        return {"stress_roundtrip_ok": ok / len(STRESS)}
