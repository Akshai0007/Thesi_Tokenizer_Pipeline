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


def cmd_run(a):
    from .runner import run

    cfg = load_config(a.config, _kv(a.set))
    _log(f"winner: {run(cfg, a.stage)}")


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
