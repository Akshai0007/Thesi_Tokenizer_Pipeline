"""Starter files for `python -m mltok new-plugin <kind> <name>`.
Every template is a WORKING plugin that passes `mltok check` (tested), so you edit a valid file."""
from string import Template

T = {}

T["source"] = '''"""$name: describe where the documents come from."""
from mltok.contracts import Source
from mltok.registry import register


@register("source", "$name")
class $Class(Source):
    """Params: (list them). Config: {type: $name, id: my_data, max_docs: 0, ...}"""

    example_params = {}  # params that make the contract check read a few documents

    def __init__(self, id="$name", **params):
        super().__init__(id=id, **params)

    def iter_docs(self):
        # yield {"text": <document>, "src": <label used for per-source metrics>} for every document
        for i in range(3):
            yield {"text": f"example document {i}", "src": self.id}
'''

T["filter"] = '''"""$name: decide which documents are kept."""
from mltok.contracts import DocFilter
from mltok.registry import register


@register("filter", "$name")
class $Class(DocFilter):
    def __init__(self, min_chars=1):
        self.min_chars = min_chars

    def keep(self, text):
        return len(text) >= self.min_chars
'''

T["normalizer"] = '''"""$name: text normalization. normalize() and hf() MUST give identical results."""
import unicodedata

from tokenizers import normalizers

from mltok.contracts import Normalizer
from mltok.registry import register


@register("normalizer", "$name")
class $Class(Normalizer):
    def normalize(self, text):
        return unicodedata.normalize("NFC", text)

    def hf(self):
        return normalizers.NFC()
'''

T["pretokenizer"] = '''"""$name: pre-tokenization. Input text already has spaces replaced by the marker."""
import regex
from tokenizers import Regex, pre_tokenizers

from mltok.contracts import PreTokenizer
from mltok.registry import register

SP = "\\u2581"
PATTERN = rf"{SP}?[^\\s{SP}]+|\\s+|{SP}(?={SP})|{SP}"
_RE = regex.compile(PATTERN)


@register("pretokenizer", "$name")
class $Class(PreTokenizer):
    def split(self, marked_text):
        return _RE.findall(marked_text)

    def hf(self):
        return pre_tokenizers.Split(Regex(PATTERN), behavior="isolated", invert=False)
'''

T["algorithm"] = '''"""$name: tokenizer training algorithm (this starter = plain BPE via the shared trainer)."""
from mltok.contracts import Algorithm
from mltok.lib.backends import HFBackend
from mltok.lib.bpe_core import build_tokenizer, train_bpe
from mltok.lib.io import select_alphabet
from mltok.lib.marker import hf_full_normalizer
from mltok.lib.unicode_ml import ML_BLOCK
from mltok.registry import register


@register("algorithm", "$name")
class $Class(Algorithm):
    needs_word_counts = True      # the engine passes ctx.word_counts
    uses_pretokenizer = True      # False -> the pretokenizer setting is ignored for this algorithm
    contract_vocab = 1200

    def __init__(self, min_pair_count=2):
        self.min_pair_count = min_pair_count

    def train(self, ctx):
        res = train_bpe(
            ctx.word_counts, vocab_target=ctx.vocab, n_specials=len(ctx.specials),
            alphabet=select_alphabet(ctx.word_counts, forced=ML_BLOCK),
            min_pair_count=self.min_pair_count, log=ctx.log,
        )
        tok = build_tokenizer(res, specials=ctx.specials, hf_normalizer=hf_full_normalizer(ctx.normalizer),
                              hf_pretokenizer=ctx.pretokenizer.hf())
        be = HFBackend(tok, ctx.normalizer, name=self.name, specials=ctx.specials)
        be.save(ctx.out_dir)
        return be, res.info

    def load(self, out_dir, normalizer):
        return HFBackend.load(out_dir, normalizer, name=self.name)
'''

T["metric"] = '''"""$name: an extra intrinsic metric. Return a flat, JSON-serialisable dict."""
from mltok.contracts import Metric
from mltok.registry import register


@register("metric", "$name")
class $Class(Metric):
    def compute(self, backend, docs):
        ids = backend.encode_batch([t for t, _ in docs])
        longest = max((len(x) for x in ids), default=0)
        return {"${name}_longest_doc_tokens": longest}
'''

T["proxy_model"] = '''"""$name: architecture of the proxy language model: forward(idx[B,T]) -> logits[B,T,V]."""
import torch.nn as nn

from mltok.contracts import ProxyModel
from mltok.registry import register


class Net(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.mix = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.out = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        h = self.emb(idx)
        return self.out(h + self.mix(h))


@register("proxy_model", "$name")
class $Class(ProxyModel):
    def __init__(self, d_model=128):
        self.d = d_model

    def build(self, vocab_size, seq_len):
        return Net(vocab_size, self.d)
'''

T["selector"] = '''"""$name: eligibility and ranking rules (lower score = better)."""
from mltok.contracts import Selector
from mltok.registry import register


@register("selector", "$name")
class $Class(Selector):
    def eligible(self, m):
        return m.get("roundtrip_ok", 1.0) >= 0.999

    def score(self, m):
        return float(m["fertility"])

    def proxy_score(self, p):
        return float(p["bpb"])
'''


T["store"] = '''"""$name: where checkpoints are pushed to / restored from (this starter copies to a directory)."""
import shutil
from pathlib import Path

from mltok.contracts import Store
from mltok.registry import register


@register("store", "$name")
class $Class(Store):
    example_params = {"path": "__TMP__"}   # lets `mltok check` run a push/pull round trip

    def __init__(self, path):
        self.root = Path(path)

    def push(self, files, message=""):
        for local, remote in files:               # one batch = one commit/upload in a real store
            dest = self.root / remote
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dest)

    def pull(self, remote_prefix, local_dir, skip_existing=True):
        base, n = self.root / remote_prefix, 0
        if not base.exists():
            return 0
        for f in (p for p in base.rglob("*") if p.is_file()):
            dest = Path(local_dir) / f.relative_to(base)
            if skip_existing and dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            n += 1
        return n

    def delete(self, remote_paths):
        for r in remote_paths:
            (self.root / r).unlink(missing_ok=True)
'''


def render(kind: str, name: str) -> str:
    cls = "".join(w.capitalize() for w in name.replace("-", "_").split("_")) or "Plugin"
    return Template(T[kind]).substitute(name=name, Class=cls)
