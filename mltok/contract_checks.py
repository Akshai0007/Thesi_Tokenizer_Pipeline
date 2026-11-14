"""`python -m mltok check` - does this plugin file fit the pipeline?

Checks use only `lib` fixtures (never other plugins), so a broken plugin cannot make an
unrelated plugin's check fail. Statuses: PASS, FAIL (blocks use), WARN (works, but look),
SKIP (needs params / optional dependency missing).

Rules checked for EVERY plugin:
  * independence: imports only `mltok.contracts`, `mltok.registry`, `mltok.lib.*` (+ third-party)
  * the contract of its kind (see mltok/contracts.py)
"""
from __future__ import annotations

import ast
import json
import tempfile
from collections import Counter
from pathlib import Path

from . import registry
from .contracts import Backend, Source, TrainContext
from .lib.io import iter_docs
from .lib.marker import full_normalize, hf_full_normalizer, mark_words

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PREFIXES = ("mltok.contracts", "mltok.registry", "mltok.lib")

SAMPLE_TEXTS = [
    "കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ തീരത്തുള്ള ഒരു സംസ്ഥാനമാണ്.",
    "ചോദ്യം: നിങ്ങൾ എവിടെയാണ് താമസിക്കുന്നത്?\nഉത്തരം: ഞാൻ കൊച്ചിയിലാണ്.",
    "Hello world, English text 12345! mixed മിക്സഡ് 3.5%",
    "  leading and  double  spaces ",
    "old chillu: \u0d28\u0d4d\u200d \u0d30\u0d4d\u200d zwnj: സദ്\u200cവാരം",
    "junk\u200bchars\ufeff and\u00adsoft \r\n crlf",
    "emoji 😀 and हिन्दी \t tab",
    "",
]


def R(status: str, msg: str):
    return (status, msg)


# --------------------------------------------------------------------------- fixtures (lib only)
class _FxNorm:
    name = "fx"

    def normalize(self, text):
        import unicodedata

        return unicodedata.normalize("NFC", text)

    def hf(self):
        from tokenizers import normalizers

        return normalizers.NFC()


class _FxPre:
    name = "fx"
    P = r"\u2581?[\p{L}\p{M}\u200c\u200d]+|\u2581?\p{N}|\u2581?[^\s\u2581\p{L}\p{M}\p{N}\u200c\u200d]+|\s+|\u2581(?=\u2581)|\u2581"

    def split(self, s):
        import regex

        return regex.findall(self.P, s)

    def hf(self):
        from tokenizers import Regex, pre_tokenizers

        return pre_tokenizers.Split(Regex(self.P), behavior="isolated", invert=False)


def fixture_docs(n=1500):
    from .lib.synth import synthetic_docs

    return list(synthetic_docs(ROOT / "data" / "seed_ml.txt", n_docs=n, n_word_types=4000, seed=1))


def tiny_backend(docs=None, vocab=1500):
    """A small lib-only HF backend (for metric / selftest fixtures). Returns (backend, docs)."""
    from .lib.backends import HFBackend
    from .lib.bpe_core import DEFAULT_SPECIALS, build_tokenizer, train_bpe
    from .lib.io import select_alphabet
    from .lib.unicode_ml import ML_BLOCK

    docs = docs or fixture_docs()
    norm, pre = _FxNorm(), _FxPre()
    wc: Counter = Counter()
    for t in docs:
        wc.update(pre.split(full_normalize(norm, t)))
    res = train_bpe(wc, vocab_target=vocab, n_specials=len(DEFAULT_SPECIALS),
                    alphabet=select_alphabet(wc, forced=ML_BLOCK), log=lambda *_: None)
    tok = build_tokenizer(res, specials=DEFAULT_SPECIALS, hf_normalizer=hf_full_normalizer(norm), hf_pretokenizer=pre.hf())
    return HFBackend(tok, norm, name="fixture"), docs


# --------------------------------------------------------------------------- independence
def check_independence(path: str) -> list:
    bad = []
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                bad.append(f"relative import (line {node.lineno}); use absolute `mltok.lib...` imports")
                continue
            mods = [node.module or ""]
        for m in mods:
            if m.startswith("mltok") and not m.startswith(ALLOWED_PREFIXES):
                bad.append(f"imports {m!r} (line {node.lineno}); plugins may only import mltok.contracts / registry / lib")
    return [R("FAIL", b) for b in bad] or [R("PASS", "independent: imports only contracts/registry/lib/third-party")]


# --------------------------------------------------------------------------- per-kind checks
def _instantiate(kind, name, spec=None, extra=None):
    s = {**(spec or {"type": name}), **(extra or {})}
    return registry.make(kind, s)


def _check_source(name, cls):
    out = []
    if cls.iter_docs is Source.iter_docs:
        return [R("FAIL", "iter_docs() is not implemented")]
    ex = getattr(cls, "example_params", None)
    if ex is None:
        try:
            _instantiate("source", name, extra={"id": "contract"})
            ex = {}
        except Exception:  # noqa: BLE001
            return out + [R("SKIP", "needs parameters; set `example_params = {...}` on the class to enable the data check")]
    src = _instantiate("source", name, extra={"id": "contract", **ex})
    docs = []
    for d in src.iter_docs():
        docs.append(d)
        if len(docs) >= 5:
            break
    ok = docs and all(isinstance(d, dict) and isinstance(d.get("text"), str) and d["text"] for d in docs)
    return out + [R("PASS" if ok else "FAIL", f"iter_docs() yields dicts with non-empty 'text' ({len(docs)} sampled)")]


def _check_filter(name, cls):
    f = _instantiate("filter", name)
    try:
        res = [f.keep(t) for t in SAMPLE_TEXTS + ["x" * 5000]]
    except Exception as e:  # noqa: BLE001
        return [R("FAIL", f"keep() raised {type(e).__name__}: {e}")]
    return [R("PASS" if all(isinstance(r, bool) for r in res) else "FAIL", "keep(text) returns bool on all samples")]


def _check_normalizer(name, cls):
    n = _instantiate("normalizer", name)
    out, bad = [], []
    hf = hf_full_normalizer(n)
    for t in SAMPLE_TEXTS:
        py = full_normalize(n, t)
        if hf.normalize_str(t) != py:
            bad.append((t[:25], py[:25], hf.normalize_str(t)[:25]))
    out.append(R("FAIL" if bad else "PASS", f"python normalize() == hf() on {len(SAMPLE_TEXTS)} samples" + (f"; first mismatch {bad[0]}" if bad else "")))
    idem = all(n.normalize(n.normalize(t)) == n.normalize(t) for t in SAMPLE_TEXTS)
    out.append(R("PASS" if idem else "WARN", "idempotent: normalize(normalize(x)) == normalize(x)"))
    return out


def _check_pretokenizer(name, cls):
    p = _instantiate("pretokenizer", name)
    out, bad = [], []
    hf = p.hf()
    for t in SAMPLE_TEXTS:
        s = mark_words(t)
        py = p.split(s)
        if "".join(py) != s:
            bad.append(("coverage", t[:25]))
        elif [x for x, _ in hf.pre_tokenize_str(s)] != py:
            bad.append(("python!=hf", t[:25]))
    return out + [R("FAIL" if bad else "PASS", "split() covers the input exactly and equals hf().pre_tokenize" + (f"; first problem {bad[0]}" if bad else ""))]


def _check_algorithm(name, cls):
    algo = _instantiate("algorithm", name)
    out = []
    for attr, typ in (("needs_word_counts", bool), ("uses_pretokenizer", bool), ("contract_vocab", int)):
        if not isinstance(getattr(algo, attr, None), typ):
            out.append(R("FAIL", f"class attribute {attr!r} missing or not {typ.__name__}"))
    if out:
        return out
    norm, pre = _FxNorm(), _FxPre()
    docs = fixture_docs(1500)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        shard = d / "train.jsonl"
        shard.write_text("\n".join(json.dumps({"text": t, "src": "fx"}, ensure_ascii=False) for t in docs), encoding="utf-8")
        wc = None
        if algo.needs_word_counts:
            wc = Counter()
            for t in docs:
                wc.update(pre.split(full_normalize(norm, t)))
        ctx = TrainContext(vocab=algo.contract_vocab, specials=["<pad>", "<s>", "</s>", "<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"],
                           normalizer=norm, pretokenizer=pre if algo.uses_pretokenizer else None, out_dir=d / "out",
                           word_counts=dict(wc) if wc is not None else None, shard_paths=[str(shard)],
                           resources={"threads": 1, "spm_max_mb": 5}, log=lambda *_: None)
        be, info = algo.train(ctx)
        out.append(R("PASS" if isinstance(be, Backend) and isinstance(info, dict) else "FAIL", "train(ctx) returns (Backend, dict)"))
        sample = docs[:40]
        ids = be.encode_batch(sample)
        in_range = all(isinstance(i, int) and 0 <= i < be.vocab_size for x in ids for i in x)
        out.append(R("PASS" if in_range and be.vocab_size > 0 else "FAIL", f"encode_batch ids are ints within vocab_size={be.vocab_size}"))
        rt = sum(be.decode(e) == be.expected(t) for e, t in zip(ids, sample))
        out.append(R("PASS" if rt == len(sample) else "FAIL", f"lossless round-trip on fixture docs ({rt}/{len(sample)})"))
        be2 = algo.load(d / "out", norm)
        out.append(R("PASS" if be2.encode_batch(sample[:10]) == ids[:10] else "FAIL", "load(out_dir) reproduces train() encodings"))
        out.append(R("PASS" if be.vocab_size <= algo.contract_vocab + 64 else "WARN", "vocab_size respects the requested size"))
    return out


def _check_metric(name, cls):
    m = _instantiate("metric", name)
    be, docs = tiny_backend()
    res = m.compute(be, [(t, "fx") for t in docs[:200]])
    ok = isinstance(res, dict) and all(isinstance(k, str) for k in res)
    try:
        json.dumps(res)
    except TypeError:
        ok = False
    return [R("PASS" if ok else "FAIL", f"compute() returns a JSON-serialisable dict ({len(res)} keys)")]


def _check_proxy_model(name, cls):
    try:
        import torch
    except ImportError:
        return [R("SKIP", "torch not installed")]
    pm = _instantiate("proxy_model", name)
    model = pm.build(300, 32)
    x = torch.randint(0, 300, (2, 16))
    y = model(x)
    ok = tuple(y.shape) == (2, 16, 300) and bool(torch.isfinite(y).all())
    n_emb = pm.n_embedding_params(model)
    return [R("PASS" if ok else "FAIL", f"forward(idx[2,16]) -> logits[2,16,300], finite ({tuple(y.shape)})"),
            R("PASS" if isinstance(n_emb, int) and n_emb >= 0 else "FAIL", f"n_embedding_params() -> int ({n_emb})")]


def _check_selector(name, cls):
    s = _instantiate("selector", name)
    m = {"fertility": 2.0, "bytes_per_token": 9.0, "roundtrip_ok": 1.0, "unk_rate": 0.0, "mark_start_rate": 0.01, "akshara_break_rate": 0.05}
    ok = isinstance(s.eligible(m), bool) and isinstance(s.score(m), (int, float)) and isinstance(s.proxy_score({"bpb": 1.2}), (int, float))
    return [R("PASS" if ok else "FAIL", "eligible()->bool, score()->float, proxy_score()->float")]


_CHECKS = {"source": _check_source, "filter": _check_filter, "normalizer": _check_normalizer,
           "pretokenizer": _check_pretokenizer, "algorithm": _check_algorithm, "metric": _check_metric,
           "proxy_model": _check_proxy_model, "selector": _check_selector}


def check_plugin(kind: str, name: str) -> list:
    cls = registry.plugin_class(kind, name)
    res = check_independence(cls._plugin_file)
    try:
        res += _CHECKS[kind](name, cls)
    except Exception as e:  # noqa: BLE001
        import traceback

        res.append(R("FAIL", f"check crashed: {type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"))
    return res


def run_checks(kind=None, name=None, file=None, extra_dirs=(), log=print) -> bool:
    registry.load_plugins(extra_dirs)
    if file:
        registry._import_file(Path(file))
    for p, tb in registry.load_errors():
        log(f"[LOAD ERROR] {p}\n{tb}")
    targets = []
    for k in registry.KINDS:
        for n in registry.available(k):
            cls = registry.plugin_class(k, n)
            if file and Path(cls._plugin_file).resolve() != Path(file).resolve():
                continue
            if (kind and k != kind) or (name and n != name):
                continue
            targets.append((k, n))
    ok = not registry.load_errors()
    for k, n in targets:
        log(f"{k}/{n}   ({Path(registry.plugin_class(k, n)._plugin_file).name})")
        for status, msg in check_plugin(k, n):
            log(f"   [{status}] {msg}")
            ok &= status != "FAIL"
    for o in registry.overrides():
        log(f"[note] plugin override: {o}")
    log("ALL CONTRACTS SATISFIED" if ok else "CONTRACT CHECK FAILED")
    return ok and bool(targets)
