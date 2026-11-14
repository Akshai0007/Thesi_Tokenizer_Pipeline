"""Malayalam normalizer: NFC + old-style chillu (C+virama+ZWJ) -> atomic chillu U+0D7A..0D7F
+ removal of invisible junk (ZWSP, BOM, soft hyphen, word joiner) and CR.
Param zw: 'keep' (default) keeps remaining ZWJ/ZWNJ (ZWNJ can be meaningful); 'drop' removes them."""
import unicodedata

import regex
from tokenizers import Regex, normalizers

from mltok.contracts import Normalizer
from mltok.registry import register

V, ZWJ, ZWNJ = "\u0d4d", "\u200d", "\u200c"
CHILLU_MAP = {
    "\u0d23" + V + ZWJ: "\u0d7a", "\u0d28" + V + ZWJ: "\u0d7b", "\u0d30" + V + ZWJ: "\u0d7c",
    "\u0d32" + V + ZWJ: "\u0d7d", "\u0d33" + V + ZWJ: "\u0d7e", "\u0d15" + V + ZWJ: "\u0d7f",
}
JUNK = "\u200b\ufeff\u00ad\u2060\u180e"
_JUNK_RE = regex.compile(f"[{JUNK}\r]")


@register("normalizer", "ml")
class MalayalamNormalizer(Normalizer):
    def __init__(self, zw="keep"):
        if zw not in ("keep", "drop"):
            raise ValueError("zw must be 'keep' or 'drop'")
        self.zw = zw

    def normalize(self, text):
        s = unicodedata.normalize("NFC", text)
        for old, new in CHILLU_MAP.items():
            s = s.replace(old, new)
        s = _JUNK_RE.sub("", s)
        if self.zw == "drop":
            s = s.replace(ZWNJ, "").replace(ZWJ, "")
        return s

    def hf(self):
        steps = [normalizers.NFC()]
        steps += [normalizers.Replace(o, n) for o, n in CHILLU_MAP.items()]
        steps.append(normalizers.Replace(Regex(f"[{JUNK}\r]"), ""))
        if self.zw == "drop":
            steps.append(normalizers.Replace(Regex(f"[{ZWNJ}{ZWJ}]"), ""))
        return normalizers.Sequence(steps)
