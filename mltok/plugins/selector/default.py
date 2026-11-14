"""Default decision rules. Eligible = lossless on eval docs AND on the stress set, (almost) no <unk>;
optional mark-start cap.
Intrinsic ranking = fertility (lower is better); final ranking = proxy-LM bits-per-byte."""
from mltok.contracts import Selector
from mltok.registry import register


@register("selector", "default")
class DefaultSelector(Selector):
    def __init__(self, min_roundtrip=0.999, min_stress=1.0, max_unk_rate=1e-3, max_mark_start=None, rank_by="fertility"):
        self.min_roundtrip, self.min_stress, self.max_unk_rate = min_roundtrip, min_stress, max_unk_rate
        self.max_mark_start, self.rank_by = max_mark_start, rank_by

    def eligible(self, m):
        if m.get("roundtrip_ok", 1.0) < self.min_roundtrip or m.get("unk_rate", 0.0) > self.max_unk_rate:
            return False
        if m.get("stress_roundtrip_ok", 1.0) < self.min_stress:
            return False
        return self.max_mark_start is None or m.get("mark_start_rate", 0.0) <= self.max_mark_start

    def score(self, m):
        return float(m[self.rank_by])

    def proxy_score(self, p):
        return float(p["bpb"])
