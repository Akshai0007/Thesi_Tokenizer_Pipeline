"""`python -m mltok <command>`"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .runner import _log, load_config


def _kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def _plugins(cfg=None, extra=()):
    from . import registry

    registry.load_plugins(list((cfg or {}).get("plugins_dir", [])) + list(extra or []))


def cmd_sysinfo(a):
    import multiprocessing

    print("python", platform.python_version(), "| cpus", multiprocessing.cpu_count())
    for mod in ("tokenizers", "sentencepiece", "regex", "numpy", "torch", "transformers", "datasets"):
        try:
            m = __import__(mod)
            print(f"{mod:14s}", getattr(m, "__version__", "ok"))
        except Exception:  # noqa: BLE001
            print(f"{mod:14s}", "MISSING")
    from .gpu import gpu_status

    print("gpu:", gpu_status())


def cmd_ingest(a):
    from .ingest import ingest_sources

    cfg = load_config(a.config, _kv(a.set))
    _plugins(cfg)
    ingest_sources(cfg["sources"], cfg["paths"]["raw"], log=_log)


def cmd_clean(a):
    from .ingest import clean_corpus
    from .lib.io import list_shards

    cfg = load_config(a.config, _kv(a.set))
    _plugins(cfg)
    paths = list_shards(cfg["paths"]["raw"])
    if not paths:
        raise SystemExit(f"no raw files under {cfg['paths']['raw']} - run `mltok ingest` first")
    clean_corpus(paths, cfg["paths"]["clean"], cfg.get("filters", []), log=_log, **cfg.get("split", {}))
    sy = _sync(cfg)
    if sy.enabled and sy.include_data:
        sy.push_tree(cfg["paths"]["clean"], message="cleaned data")


def _sync(cfg):
    from .sync import Sync

    return Sync(cfg, Path(cfg["paths"]["runs"]) / cfg["name"], Path(cfg["paths"]["clean"]), log=_log)


def cmd_restore(a):
    cfg = load_config(a.config, _kv(a.set))
    _plugins(cfg)
    sy = _sync(cfg)
    if not sy.enabled:
        print("no store configured (store: none) - nothing to restore")
        return
    sy.restore(a.what)


def cmd_push(a):
    cfg = load_config(a.config, _kv(a.set))
    _plugins(cfg)
    sy = _sync(cfg)
    if not sy.enabled:
        print("no store configured (store: none) - nothing to push")
        return
    sy.push_tree(sy.root, message=f"{sy.name}: manual push")
    if a.data and sy.clean_dir:
        sy.include_data = True            # explicit request on the command line
        sy.push_tree(sy.clean_dir, message="cleaned data")


def cmd_store_check(a):
    """Push a tiny file, read it back, delete it: proves the store (and its token) works BEFORE a long run."""
    import tempfile
    import time

    cfg = load_config(a.config, _kv(a.set))
    _plugins(cfg)
    sy = _sync(cfg)
    if not sy.enabled:
        print("[FAIL] no store configured (store: none)")
        sys.exit(1)
    try:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "ping.txt"
            payload = f"mltok store check {time.time()}"
            f.write_text(payload)
            remote = f"healthcheck/{sy.name}/ping.txt"
            sy.store.push([(f, remote)], "mltok store check")
            got = Path(d) / "got"
            n = sy.store.pull(f"healthcheck/{sy.name}", got)
            ok = n == 1 and (got / "ping.txt").read_text() == payload
            sy.store.delete([remote])
    except Exception as e:  # noqa: BLE001 - print the type/message only, never credentials
        print(f"[FAIL] store check failed: {type(e).__name__}: {str(e)[:200]}")
        sys.exit(1)
    print("[PASS] store push -> pull round trip works" if ok else "[FAIL] store round trip returned different data")
    sys.exit(0 if ok else 1)


def cmd_run(a):
    from .runner import run

    cfg = load_config(a.config, _kv(a.set))
    _log(f"winner: {run(cfg, a.stage)}")


def cmd_pin(a):
    """Make a deliberate choice (not necessarily the proxy-BPB winner) the run's best/, with the reason recorded."""
    from .runner import Run

    cfg = load_config(a.config, _kv(a.set))
    r = Run(cfg)
    r.pin(a.variant, a.reason)


def cmd_synth(a):
    import gzip

    from .lib.synth import synthetic_docs

    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        for t in synthetic_docs(a.seed_file, a.n_docs, a.n_word_types if a.mode == "akshara" else 0):
            f.write(json.dumps({"text": t, "src": "synthetic"}, ensure_ascii=False) + "\n")
    _log(f"synthetic corpus -> {a.out} (SMOKE TESTING ONLY)")


def cmd_eval(a):
    from . import registry
    from .lib.backends import HFBackend
    from .lib.io import iter_docs, list_shards

    _plugins(None, a.plugins_dir)
    norm = registry.make("normalizer", json.loads(a.normalizer) if a.normalizer.startswith("{") else a.normalizer)
    be = HFBackend.load(a.tokenizer, norm, name=a.tokenizer)
    docs = []
    for p in list_shards(a.data):
        for d in iter_docs(p):
            docs.append(d)
            if len(docs) >= a.max_docs:
                break
        if len(docs) >= a.max_docs:
            break
    out = {}
    for m in a.metrics.split(","):
        out.update(registry.make("metric", m).compute(be, docs))
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_export_hf(a):
    """Convert an existing SentencePiece Unigram model (e.g. a previous run's best/spm.model) to tokenizer.json."""
    from . import registry
    from .lib.io import iter_docs, list_shards
    from .lib.spm_export import export_hf

    _plugins(None, a.plugins_dir)
    norm = registry.make("normalizer", json.loads(a.normalizer) if a.normalizer.startswith("{") else a.normalizer)
    texts = []
    if a.data:
        for p in list_shards(a.data):
            for t, _ in iter_docs(p):
                texts.append(t)
                if len(texts) >= a.max_docs:
                    break
            if len(texts) >= a.max_docs:
                break
    out = a.out or str(Path(a.spm).parent)
    rep = export_hf(a.spm, out, norm, texts)
    print(json.dumps({k: v for k, v in rep.items() if k != "examples"}, indent=2))
    for ex in rep["examples"]:
        print("REAL MISMATCH:", ex)
    print(f"wrote tokenizer.json etc. to {out}")
    sys.exit(0 if rep["ok"] else 1)


def cmd_verify(a):
    from .verify import verify_dir

    _plugins(None, a.plugins_dir)
    sys.exit(0 if verify_dir(a.tokenizer, normalizer=a.normalizer) else 1)


def cmd_proxy_worker(a):
    from .gpu import cap_memory
    from .runner import Run

    cfg = json.loads(Path(a.cfg).read_text())
    r = Run(cfg)
    g = cfg.get("gpu", {})
    if g.get("guard"):
        cap_memory(g.get("max_gb", 60), log=_log)
    for job in a.variants.split(","):
        dname, seed = job.rsplit(":", 1)
        v = json.loads((r.vroot / dname / "variant.json").read_text())
        r.variants[r.vname(v)] = v
        r.ensure_proxy(v, int(seed))
    r.sync.flush(force=True, message="proxy worker: final flush")


def cmd_selftest(a):
    from .selftest import run_selftest

    sys.exit(0 if run_selftest(single=a.single) else 1)


def cmd_plugins(a):
    from . import registry

    _plugins(None, a.plugins_dir)
    print(f"{'kind':13s} {'name':16s} file / description")
    for r in registry.describe():
        rel = r["file"].split("mltok/")[-1] if "mltok/" in r["file"] else r["file"]
        print(f"{r['kind']:13s} {r['name']:16s} {rel}\n{'':30s}{r['doc'][:100]}")
    for p, tb in registry.load_errors():
        print(f"[LOAD ERROR] {p}\n{tb}")
    for o in registry.overrides():
        print(f"[override] {o}")


def cmd_check(a):
    from .contract_checks import run_checks

    ok = run_checks(kind=a.kind, name=a.name, file=a.file, extra_dirs=a.plugins_dir)
    sys.exit(0 if ok else 1)


def cmd_new_plugin(a):
    from .templates import render

    out = Path(a.dir) / f"{a.name}.py"
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(a.kind, a.name))
    print(f"wrote {out}\nnext: python -m mltok check --file {out}\nthen reference it in the config: "
          f"{{type: {a.name}}}  and add `plugins_dir: [{a.dir}]`")


def main(argv=None):
    from . import registry

    p = argparse.ArgumentParser(prog="mltok")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, config=True, plugdir=False):
        sp = sub.add_parser(name)
        if config:
            sp.add_argument("-c", "--config", required=True)
            sp.add_argument("--set", nargs="*", help="override config keys, e.g. proxy.train_mb=50")
        if plugdir:
            sp.add_argument("--plugins-dir", nargs="*", default=[], help="extra plugin directories")
        sp.set_defaults(fn=fn)
        return sp

    add("sysinfo", cmd_sysinfo, config=False)
    add("ingest", cmd_ingest)
    add("download", cmd_ingest)  # alias
    add("clean", cmd_clean)
    s = add("run", cmd_run)
    s.add_argument("--stage", default="all", choices=["intrinsic", "proxy", "baselines", "all"])
    s = add("restore", cmd_restore)
    s.add_argument("--what", default="run", choices=["run", "data", "all"])
    add("store-check", cmd_store_check)
    s = add("push", cmd_push)
    s.add_argument("--data", action="store_true", help="also push the cleaned data shards")
    s = add("pin", cmd_pin)
    s.add_argument("--variant", required=True, help="variant name as printed in the report, e.g. spm_unigram-ml-v48000")
    s.add_argument("--reason", required=True, help="why this choice (recorded in best/SELECTION.json)")
    s = add("synth", cmd_synth, config=False)
    s.add_argument("--seed-file", default="data/seed_ml.txt")
    s.add_argument("--out", required=True)
    s.add_argument("--n-docs", type=int, default=5000)
    s.add_argument("--mode", default="words", choices=["words", "akshara"])
    s.add_argument("--n-word-types", type=int, default=100000)
    s = add("eval", cmd_eval, config=False, plugdir=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--max-docs", type=int, default=20000)
    s.add_argument("--normalizer", default="ml", help="plugin name or JSON spec")
    s.add_argument("--metrics", default="basic,structure,roundtrip")
    s = add("export-hf", cmd_export_hf, config=False, plugdir=True)
    s.add_argument("--spm", required=True, help="path to spm.model (SentencePiece UNIGRAM)")
    s.add_argument("--out", help="output directory (default: next to the model)")
    s.add_argument("--normalizer", default="ml", help="the normalizer the model was trained with (plugin name or JSON spec)")
    s.add_argument("--data", help="optional cleaned shards to use as extra parity-check text")
    s.add_argument("--max-docs", type=int, default=2000)
    s = add("verify", cmd_verify, config=False, plugdir=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--normalizer", default="ml")
    s = add("proxy-worker", cmd_proxy_worker, config=False)
    s.add_argument("--cfg", required=True)
    s.add_argument("--variants", required=True)
    s = add("selftest", cmd_selftest, config=False)
    s.add_argument("--single", action="store_true", help="test only the first visible device")
    add("plugins", cmd_plugins, config=False, plugdir=True)
    s = add("check", cmd_check, config=False, plugdir=True)
    s.add_argument("--kind", choices=registry.KINDS)
    s.add_argument("--name")
    s.add_argument("--file", help="check only the plugin(s) defined in this file")
    s = add("new-plugin", cmd_new_plugin, config=False)
    s.add_argument("kind", choices=registry.KINDS)
    s.add_argument("name")
    s.add_argument("--dir", default="my_plugins")

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
