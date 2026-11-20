"""ENGINE: plugin-driven experiment runner. Contains no dataset, language or algorithm specifics.

Stages (each result is cached under a key made from the *source code + params* of every plugin
it depends on, so swapping a plugin file recomputes exactly what depended on it):

  tokenizer  key = data + normalizer + pretokenizer(if used) + algorithm + vocab + resources
  metrics    key = tokenizer + eval data + metric plugins           (tokenizer is reused)
  proxy LM   key = tokenizer + train/eval data + proxy_model plugin + proxy settings
  report     always rebuilt (cheap)
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import registry
from .contracts import TrainContext
from .gpu import cap_memory, gpu_count, wait_for_gpu
from .lib.backends import HubBackend
from .lib.bpe_core import DEFAULT_SPECIALS
from .lib.io import count_words, data_fingerprint, iter_docs, iter_texts, list_shards
from .registry import fingerprint, keyhash, make, slug, spec_of
from .sync import Sync

DEFAULT_METRICS = ["basic", "structure", "roundtrip", "stress"]
DEFAULT_COLUMNS = ["fertility", "bytes_per_token", "renyi_eff", "mark_start_rate", "akshara_break_rate", "unk_rate", "roundtrip_ok", "stress_roundtrip_ok"]


def se_diff(a: dict, b: dict) -> float:
    """Standard error of (mean_a - mean_b) from per-variant seed statistics (bpb_std is the population std)."""
    def var_of_mean(x):
        n = x.get("n_seeds", 1)
        sample_var = (x.get("bpb_std", 0.0) ** 2) * n / (n - 1) if n > 1 else 0.0
        return sample_var / max(n, 1)
    return (var_of_mean(a) + var_of_mean(b)) ** 0.5


def load_config(path: str, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for k, v in (overrides or {}).items():
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return cfg


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Run:
    def __init__(self, cfg: dict):
        registry.load_plugins(cfg.get("plugins_dir", []))
        self.cfg = cfg
        self.root = Path(cfg["paths"]["runs"]) / cfg["name"]
        self.vroot = self.root / "variants"
        self.cache = self.root / "cache"
        for d in (self.vroot, self.cache):
            d.mkdir(parents=True, exist_ok=True)
        self.train_shards = list_shards(Path(cfg["paths"]["clean"]) / "train")
        self.eval_shards = list_shards(Path(cfg["paths"]["clean"]) / "eval")
        if not self.train_shards or not self.eval_shards:
            raise SystemExit(f"No cleaned shards under {cfg['paths']['clean']} - run `mltok clean` first.")
        self.res = cfg.get("resources", {})
        self.specials = cfg.get("specials", DEFAULT_SPECIALS)
        comp = cfg["components"]
        self.base = self.canon(
            {"normalizer": spec_of(comp["normalizer"]), "pretokenizer": spec_of(comp["pretokenizer"]),
             "algorithm": spec_of(comp["algorithm"]), "vocab": int(comp["vocab"])}
        )
        self.selector = make("selector", cfg.get("selector", "default"))
        self.metric_specs = [spec_of(m) for m in cfg.get("metrics", DEFAULT_METRICS)]
        self.data_fp = data_fingerprint(self.train_shards)
        self.eval_fp = data_fingerprint(self.eval_shards)
        self._eval_docs = None
        self._wc: dict = {}
        self._inst: dict = {}
        self.variants: dict[str, dict] = {}
        self.timings: dict[str, float] = {}
        self.sync = Sync(cfg, self.root, Path(cfg["paths"]["clean"]), log=_log)

    # ---------------------------------------------------------------- components
    def algo_cls(self, v):
        return registry.plugin_class("algorithm", v["algorithm"]["type"])

    def canon(self, v: dict) -> dict:
        v = {**v}
        if not self.algo_cls(v).uses_pretokenizer:
            v["pretokenizer"] = {"type": "-"}  # irrelevant for this algorithm -> no duplicate runs
        return v

    def inst(self, v):
        key = json.dumps(v, sort_keys=True)
        if key not in self._inst:
            pre = None if v["pretokenizer"]["type"] == "-" else make("pretokenizer", v["pretokenizer"])
            self._inst[key] = (make("normalizer", v["normalizer"]), pre, make("algorithm", v["algorithm"]))
        return self._inst[key]

    def vname(self, v) -> str:
        parts = [slug(v["algorithm"]), slug(v["normalizer"])]
        if v["pretokenizer"]["type"] != "-":
            parts.append(slug(v["pretokenizer"]))
        return "-".join(parts) + f"-v{v['vocab']}"

    def _pre_fp(self, v):
        return None if v["pretokenizer"]["type"] == "-" else fingerprint("pretokenizer", v["pretokenizer"])

    def _word_key(self, v) -> str:
        return keyhash(self.data_fp, fingerprint("normalizer", v["normalizer"]), self._pre_fp(v),
                       self.res.get("max_docs_per_shard", 0))

    def tok_key(self, v) -> str:
        return keyhash(
            self.data_fp, fingerprint("normalizer", v["normalizer"]), self._pre_fp(v),
            fingerprint("algorithm", v["algorithm"]), v["vocab"], self.specials,
            {k: self.res.get(k) for k in ("max_word_types", "spm_max_mb", "max_docs_per_shard")},
        )

    def vdir(self, v) -> Path:
        return self.vroot / f"{self.vname(v)}-{self.tok_key(v)[:8]}"

    # ---------------------------------------------------------------- data
    def eval_docs(self):
        if self._eval_docs is None:
            n = self.cfg.get("search", {}).get("eval_docs", 20000)
            docs = []
            for p in self.eval_shards:
                for d in iter_docs(p):
                    docs.append(d)
                    if len(docs) >= n:
                        break
                if len(docs) >= n:
                    break
            self._eval_docs = docs
            _log(f"eval set: {len(docs)} docs")
        return self._eval_docs

    def word_counts(self, v):
        key = self._word_key(v)
        if key in self._wc:
            return self._wc[key]
        f = self.cache / f"words-{key}.pkl"
        if f.exists():
            wc = pickle.loads(f.read_bytes())
        else:
            _log(f"counting words ({slug(v['normalizer'])}, {slug(v['pretokenizer'])}) ...")
            wc = dict(count_words(self.train_shards, v["normalizer"], v["pretokenizer"],
                                  workers=self.res.get("workers", 1),
                                  max_docs_per_shard=self.res.get("max_docs_per_shard", 0), log=_log))
            f.write_bytes(pickle.dumps(wc, protocol=4))
            self.sync.mark(f)
            self.sync.flush(message=f"{self.root.name}: word table")
        self._wc[key] = wc
        _log(f"word table: {len(wc):,} types")
        return wc

    # ---------------------------------------------------------------- stage: tokenizer
    def ensure_tokenizer(self, v):
        """Returns (backend, info). Trains only if no finished artifact with this key exists."""
        d = self.vdir(v)
        norm, pre, algo = self.inst(v)
        done = d / "tokenizer.ok"
        if done.exists():
            return algo.load(d, norm), json.loads(done.read_text())
        _log(f"== training {self.vname(v)}")
        d.mkdir(parents=True, exist_ok=True)
        wc = self.word_counts(v) if algo.needs_word_counts else None
        ctx = TrainContext(vocab=v["vocab"], specials=self.specials, normalizer=norm, pretokenizer=pre,
                           out_dir=d, word_counts=wc, shard_paths=self.train_shards, resources=self.res, log=_log)
        t0 = time.time()
        be, info = algo.train(ctx)
        info = {**info, "train_seconds_total": round(time.time() - t0, 1)}
        mass = info.get("word_mass_covered")
        if mass is not None and mass < 0.97:
            _log(f"   WARNING: {self.vname(v)} trained on word types covering only {mass:.1%} of the text "
                 f"(resources.max_word_types caps the word table) - raise it, or this variant is handicapped against "
                 f"algorithms that see all the data")
        (d / "variant.json").write_text(json.dumps(v))
        done.write_text(json.dumps(info))
        self.sync.mark(d)
        self.sync.flush(message=f"{self.root.name}: tokenizer {self.vname(v)}")
        return be, info

    # ---------------------------------------------------------------- stage: metrics
    def metrics_key(self, v) -> str:
        return keyhash(self.tok_key(v), self.eval_fp, self.cfg.get("search", {}).get("eval_docs", 20000),
                       [fingerprint("metric", m) for m in self.metric_specs])

    def ensure_metrics(self, v) -> dict:
        mp = self.vdir(v) / "metrics.json"
        key = self.metrics_key(v)
        if mp.exists():
            m = json.loads(mp.read_text())
            if m.get("_key") == key:
                return m
            _log(f"   metrics for {self.vname(v)} are stale (plugin or data changed) -> recomputing")
        be, info = self.ensure_tokenizer(v)
        docs = self.eval_docs()
        m: dict = {}
        for ms in self.metric_specs:
            m.update(make("metric", ms).compute(be, docs))
        m.update(_key=key, name=self.vname(v), variant=v, train_info=info)
        mp.write_text(json.dumps(m, indent=2))
        self.sync.mark(mp)
        self.sync.flush(message=f"{self.root.name}: metrics {self.vname(v)}")
        _log("   " + self.vname(v) + ": " + " ".join(
            f"{k}={m[k]:.4g}" for k in ("fertility", "bytes_per_token", "mark_start_rate", "akshara_break_rate", "roundtrip_ok") if k in m))
        return m

    # ---------------------------------------------------------------- stage: proxy LM
    def _pc(self) -> dict:
        return self.cfg.get("proxy", {})

    def seeds(self) -> list[int]:
        pc = self._pc()
        return [int(x) for x in (pc.get("seeds") or [pc.get("seed", 0)])]

    def proxy_key(self, v, seed: int) -> str:
        pc = self._pc()
        return keyhash(self.tok_key(v), self.eval_fp, self.data_fp,
                       fingerprint("proxy_model", pc.get("model", {"type": "gpt", "size": "tiny"})),
                       {k: pc.get(k) for k in ("seq_len", "batch", "train_mb", "eval_mb", "epochs", "lr")}, int(seed))

    def proxy_path(self, v, seed: int) -> Path:
        return self.vdir(v) / f"proxy_seed{seed}.json"

    def _proxy_done(self, v, seed: int) -> bool:
        p = self.proxy_path(v, seed)
        return p.exists() and json.loads(p.read_text()).get("_key") == self.proxy_key(v, seed)

    def ensure_proxy(self, v, seed: int) -> dict:
        from .proxy_train import train_proxy

        pp = self.proxy_path(v, seed)
        if self._proxy_done(v, seed):
            return json.loads(pp.read_text())
        if pp.exists():
            _log(f"   proxy result for {self.vname(v)} seed {seed} is stale -> retraining")
        pc = self._pc()
        be, _ = self.ensure_tokenizer(v)
        _log(f"== proxy LM for {self.vname(v)} (seed {seed})")
        r = train_proxy(
            be, iter_texts(self.train_shards), (t for t, _ in self.eval_docs()),
            make("proxy_model", pc.get("model", {"type": "gpt", "size": "tiny"})),
            seq_len=pc.get("seq_len", 512), batch=pc.get("batch", 64), train_mb=pc.get("train_mb", 100),
            eval_mb=pc.get("eval_mb", 10), epochs=pc.get("epochs", 1), lr=pc.get("lr", 1e-3),
            seed=seed, micro_batch=pc.get("micro_batch", 0),
            ckpt_path=str(self.vdir(v) / f"proxy_seed{seed}.ckpt"), ckpt_every=int(pc.get("checkpoint_every_steps", 0)),
            ckpt_key=self.proxy_key(v, seed),
            on_ckpt=lambda path: (self.sync.mark(path), self.sync.flush()),
            log=_log,
        )
        r["_key"] = self.proxy_key(v, seed)
        r["seed"] = seed
        pp.write_text(json.dumps(r, indent=2))
        self.sync.mark(pp)
        self.sync.mark_delete(self.vdir(v) / f"proxy_seed{seed}.ckpt")
        self.sync.flush(message=f"{self.root.name}: proxy {self.vname(v)} seed {seed}")
        _log(f"   {self.vname(v)} seed {seed}: BPB={r['bpb']:.4f} ({r['seconds']}s)")
        return r

    def proxy_aggregate(self, v) -> dict:
        """Mean over seeds (what the selector sees) + spread, so noise is visible in the report."""
        import statistics

        rs = [json.loads(self.proxy_path(v, sd).read_text()) for sd in self.seeds()]
        bpbs = [r["bpb"] for r in rs]
        agg = {k: x for k, x in rs[0].items() if not k.startswith("_")}
        agg.update(bpb=statistics.fmean(bpbs), bpb_std=(statistics.pstdev(bpbs) if len(bpbs) > 1 else 0.0),
                   bpb_seeds=bpbs, n_seeds=len(bpbs))
        return agg

    def resolve_gpus(self) -> list:
        g = self._pc().get("gpus", "auto")
        if g == "auto":
            return list(range(gpu_count()))
        return list(g) if isinstance(g, (list, tuple)) else [g]

    def run_proxies(self, names: list[str]) -> dict[str, dict]:
        """One job per (variant, seed). With >= 2 GPUs: one worker process per GPU
        (CUDA_VISIBLE_DEVICES pinned); a "cpu" entry runs a CPU worker (used by tests)."""
        todo = [(n, sd) for n in names for sd in self.seeds() if not self._proxy_done(self.variants[n], sd)]
        gpus = self.resolve_gpus()
        if todo:
            if len(gpus) < 2 or len(todo) < 2:
                g = self.cfg.get("gpu", {})
                if g.get("guard"):
                    cap_memory(g.get("max_gb", 60), log=_log)
                for n, sd in todo:
                    self.ensure_proxy(self.variants[n], sd)
            else:
                self._run_workers(todo, gpus)
        return {n: self.proxy_aggregate(self.variants[n]) for n in names}

    def _run_workers(self, todo: list[tuple[str, int]], gpus: list) -> None:
        cfg_path = self.root / "cfg.json"
        cfg_path.write_text(json.dumps(self.cfg))
        k = min(len(gpus), len(todo))
        order = sorted(todo, key=lambda t: -self.variants[t[0]]["vocab"])
        procs = []
        for gid, bucket in zip(gpus[:k], [order[i::k] for i in range(k)]):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gid == "cpu" else str(gid), TOKENIZERS_PARALLELISM="true")
            logf = self.root / f"proxy_worker_gpu{gid}.log"
            fh = open(logf, "w")
            jobs = ",".join(f"{self.vdir(self.variants[n]).name}:{sd}" for n, sd in bucket)
            cmd = [sys.executable, "-m", "mltok", "proxy-worker", "--cfg", str(cfg_path), "--variants", jobs]
            _log(f"worker on GPU {gid}: {len(bucket)} jobs -> log {logf.name}")
            procs.append((gid, subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh, logf))
        while any(p.poll() is None for _, p, _, _ in procs):
            time.sleep(20)
            for gid, p, _, lf in procs:
                if p.poll() is None:
                    tail = lf.read_text(errors="ignore").strip().splitlines()[-1:] or [""]
                    _log(f"  [gpu {gid}] {tail[0][:110]}")
        for gid, p, fh, lf in procs:
            fh.close()
            if p.returncode != 0:
                raise RuntimeError(f"proxy worker on GPU {gid} failed (rc={p.returncode}); see {lf}")

    # ---------------------------------------------------------------- search
    def stage_intrinsic(self) -> dict[str, dict]:
        t0 = time.time()
        factors = self.cfg.get("search", {}).get("factors", {})
        for f in factors:
            if f not in ("normalizer", "pretokenizer", "algorithm"):
                raise ValueError(f"search.factors key {f!r} must be normalizer / pretokenizer / algorithm")
        self.variants = {self.vname(self.base): self.base}
        for f, vals in factors.items():
            for val in vals:
                v = self.canon({**self.base, f: spec_of(val)})
                self.variants.setdefault(self.vname(v), v)
        results = {n: self.ensure_metrics(v) for n, v in list(self.variants.items())}

        if self.cfg.get("search", {}).get("combine", True) and factors:
            best = {}
            for f, vals in factors.items():
                cands = []
                for val in list(vals) + [self.base[f]]:
                    n = self.vname(self.canon({**self.base, f: spec_of(val)}))
                    if n in results and self.selector.eligible(results[n]):
                        cands.append((self.selector.score(results[n]), spec_of(val)))
                best[f] = min(cands, key=lambda x: x[0])[1] if cands else self.base[f]
            comb = self.canon({**self.base, **best})
            n = self.vname(comb)
            if n not in results:
                _log(f"combined candidate: { {k: slug(x) for k, x in best.items()} }")
                self.variants[n] = comb
                results[n] = self.ensure_metrics(comb)
        self.timings["intrinsic_s"] = round(time.time() - t0, 1)
        self.sync.flush(force=True, message=f"{self.root.name}: intrinsic stage done")
        return results

    def stage_proxy(self, results: dict[str, dict]) -> dict[str, dict]:
        t0 = time.time()
        pc = self._pc()
        if not pc.get("enabled", True):
            return {}
        g = self.cfg.get("gpu", {})
        if g.get("guard"):
            wait_for_gpu(g.get("min_free_gb", 60), g.get("max_util_pct", 30), index=g.get("index", 0), log=_log)
        _log(f"proxy GPUs: {self.resolve_gpus() or 'none (CPU)'}")
        ranked = sorted((n for n, m in results.items() if self.selector.eligible(m)),
                        key=lambda n: self.selector.score(results[n]))
        chosen = ranked[: pc.get("top_k", 3)]
        for n, v in self.variants.items():  # reference algorithms always get a proxy run
            pre_ok = v["pretokenizer"]["type"] == "-" or v["pretokenizer"] == self.base["pretokenizer"]  # SentencePiece ignores it
            if (v["algorithm"]["type"] in pc.get("always_include", ["hf_bpe"]) and v["normalizer"] == self.base["normalizer"]
                    and pre_ok and v["vocab"] == self.base["vocab"] and n not in chosen):
                chosen.append(n)
        out = self.run_proxies(chosen)

        sweep = pc.get("vocab_sweep", [])
        if sweep and out:
            pool = [n for n in out if self.selector.eligible(results[n])] or list(out)
            win = min(pool, key=lambda n: self.selector.proxy_score(out[n]))
            new = []
            for V in sweep:
                v2 = {**self.variants[win], "vocab": int(V)}
                n2 = self.vname(v2)
                if n2 == win or n2 in self.variants:
                    continue
                self.variants[n2] = v2
                results[n2] = self.ensure_metrics(v2)
                new.append(n2)
            out.update(self.run_proxies(new))
        self.timings["proxy_s"] = round(time.time() - t0, 1)
        self.sync.flush(force=True, message=f"{self.root.name}: proxy stage done")
        return out

    def stage_baselines(self) -> dict[str, dict]:
        out = {}
        for name in self.cfg.get("baselines", []):
            f = self.cache / f"baseline_{name.replace('/', '__')}_{self.eval_fp}.json"
            if f.exists():
                out[name] = json.loads(f.read_text())
                continue
            try:
                be = HubBackend(name)
                m = {}
                for ms in self.metric_specs:
                    m.update(make("metric", ms).compute(be, self.eval_docs()))
                out[name] = m
                f.write_text(json.dumps(m))
                _log(f"baseline {name}: fertility={m['fertility']:.3f}")
            except Exception as e:  # noqa: BLE001
                _log(f"baseline {name}: skipped ({type(e).__name__}: {str(e)[:100]})")
        return out

    # ---------------------------------------------------------------- provenance / pin
    def provenance_of(self, v) -> dict:
        return {
            "variant": v,
            "plugin_fingerprints": {
                "normalizer": fingerprint("normalizer", v["normalizer"]),
                "pretokenizer": self._pre_fp(v),
                "algorithm": fingerprint("algorithm", v["algorithm"]),
                "metrics": [fingerprint("metric", m) for m in self.metric_specs],
            },
            "plugin_files": {k: registry.plugin_class(k, v[k]["type"])._plugin_file for k in ("normalizer", "algorithm")},
            "data_fingerprint": {"train": self.data_fp, "eval": self.eval_fp},
        }

    def pin(self, variant_name: str, reason: str) -> dict:
        """Make a DELIBERATE choice the run's `best/` (it need not be the proxy-BPB winner), with the trade-off recorded.
        Only a variant that was trained AND evaluated in this run directory, and is eligible, can be pinned."""
        cands = [d for d in self.vroot.iterdir() if d.is_dir() and d.name.rsplit("-", 1)[0] == variant_name and (d / "tokenizer.ok").exists()]
        if not cands:
            have = sorted({d.name.rsplit("-", 1)[0] for d in self.vroot.iterdir() if d.is_dir()})
            raise SystemExit(f"no finished variant named {variant_name!r} in {self.vroot}. Available: {have}")
        d = max(cands, key=lambda p: (p / "metrics.json").stat().st_mtime if (p / "metrics.json").exists() else 0)
        if not (d / "metrics.json").exists():
            raise SystemExit(f"{variant_name} has no metrics.json - run the intrinsic stage first")
        m = json.loads((d / "metrics.json").read_text())
        if not self.selector.eligible(m):
            raise SystemExit(f"{variant_name} is not eligible (lossless / stress / unk rules) - it cannot be pinned")
        v = json.loads((d / "variant.json").read_text())
        summ = json.loads((self.root / "proxy_summary.json").read_text()) if (self.root / "proxy_summary.json").exists() else {}
        mine = summ.get(variant_name)
        def _eligible(name):  # compare only against variants that could actually be chosen (lossless etc.)
            ms = [dd / "metrics.json" for dd in self.vroot.iterdir() if dd.is_dir() and dd.name.rsplit("-", 1)[0] == name and (dd / "metrics.json").exists()]
            return bool(ms) and self.selector.eligible(json.loads(max(ms, key=lambda f: f.stat().st_mtime).read_text()))
        pool = {n: x for n, x in summ.items() if _eligible(n)}
        best_name = min(pool, key=lambda n: pool[n]["bpb"]) if pool else None
        info = {"reason": reason, "variant": variant_name}
        if mine and best_name:
            info.update(proxy_winner=best_name, proxy_winner_bpb=pool[best_name]["bpb"], bpb=mine["bpb"],
                        bpb_gap_pct=(mine["bpb"] / pool[best_name]["bpb"] - 1) * 100,
                        z_vs_proxy_winner=((mine["bpb"] - pool[best_name]["bpb"]) / se_diff(mine, pool[best_name])
                                           if se_diff(mine, pool[best_name]) > 0 else None))
        best = self.root / "best"
        if best.exists():
            shutil.rmtree(best)
        shutil.copytree(d, best)
        sel = {"winner": variant_name, "basis": f"PINNED by decision: {reason}", "pinned": info, "metrics": m,
               "proxy": mine, "provenance": self.provenance_of(v)}
        (best / "SELECTION.json").write_text(json.dumps(sel, indent=2, default=str))
        self.sync.mark(best)
        self.sync.flush(force=True, message=f"{self.root.name}: pinned {variant_name}")
        _log(f"pinned {variant_name} -> {best}" + (f" (BPB {info['bpb_gap_pct']:+.2f}% vs proxy winner {best_name})" if "bpb_gap_pct" in info else ""))
        return sel

    # ---------------------------------------------------------------- report
    def finish(self, results, proxy, baselines):
        sel = self.selector
        cols = self.cfg.get("report", {}).get("columns", DEFAULT_COLUMNS)
        rows = []
        for n, m in results.items():
            rows.append({"name": n, "vocab": m["vocab_size"], **{c: m.get(c) for c in cols},
                         "proxy_bpb": proxy.get(n, {}).get("bpb"), "eligible": sel.eligible(m)})
        rows.sort(key=lambda r: (r["proxy_bpb"] is None, r["proxy_bpb"] or 0, r.get("fertility") or 0))
        with open(self.root / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        if proxy:
            pool = [n for n in proxy if sel.eligible(results[n])]
            win = min(pool or list(proxy), key=lambda n: sel.proxy_score(proxy[n]))
            basis = "lowest proxy-LM score (bits-per-byte by default) among ELIGIBLE variants" + ("" if pool else " (WARNING: none eligible)")
        else:
            elig = [n for n, m in results.items() if sel.eligible(m)] or list(results)
            win = min(elig, key=lambda n: sel.score(results[n]))
            basis = "best intrinsic score among eligible variants (proxy stage not run)"
        v = self.variants[win]
        best = self.root / "best"
        if best.exists():
            shutil.rmtree(best)
        shutil.copytree(self.vdir(v), best)
        provenance = self.provenance_of(v)
        (best / "SELECTION.json").write_text(json.dumps(
            {"winner": win, "basis": basis, "metrics": results[win], "proxy": proxy.get(win), "provenance": provenance},
            indent=2, default=str))

        def fmt(x, nd=3):
            return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

        L = [f"# Run report: {self.cfg['name']}", "", f"**Selected:** `{win}` ({basis})", "",
             "## Results", "", "| variant | vocab | " + " | ".join(cols) + " | proxy BPB | eligible |",
             "|---|---|" + "---|" * (len(cols) + 2)]
        for r in rows:
            L.append(f"| {r['name']} | {r['vocab']} | " + " | ".join(fmt(r[c], 4) for c in cols)
                     + f" | {fmt(r['proxy_bpb'], 4)} | {r['eligible']} |")
        if baselines:
            L += ["", "## Published tokenizers (count metrics only, same eval text)", "",
                  "| tokenizer | vocab | fertility | B/token |", "|---|---|---|---|"]
            for n, m in sorted(baselines.items(), key=lambda kv: kv[1].get("fertility", 9e9)):
                L.append(f"| {n} | {m.get('vocab_size')} | {fmt(m.get('fertility'))} | {fmt(m.get('bytes_per_token'), 2)} |")
        if proxy:
            L += ["", "## Proxy LM details (BPB = mean over seeds)", "",
                  "| variant | BPB | ± std | seeds | dtype | params | non-embed | train tokens | steps | tok/s | peak GB | sec |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            ranked = sorted(proxy.items(), key=lambda kv: sel.proxy_score(kv[1]))
            for n, p in ranked:
                L.append(f"| {n} | {p['bpb']:.4f} | {p.get('bpb_std', 0):.4f} | {p.get('n_seeds', 1)} | {p.get('amp_dtype', '-')} | "
                         f"{p['n_params']:,} | {p['n_nonembed']:,} | {p['train_tokens']:,} | {p['steps']} | "
                         f"{p.get('tokens_per_sec', '-')} | {p.get('peak_mem_gb', '-')} | {p['seconds']} |")
            elig = [(n, p) for n, p in ranked if sel.eligible(results[n])]
            if len(elig) >= 2:
                (a, pa), (b, pb) = elig[0], elig[1]
                gap, se = pb["bpb"] - pa["bpb"], se_diff(pa, pb)
                if pa.get("n_seeds", 1) < 2 or pb.get("n_seeds", 1) < 2:
                    L += ["", f"_Top-2 gap: {gap:.4f} BPB ({a} vs {b}). Fewer than 2 seeds, so noise is unknown - rerun with `proxy.seeds: [0,1,2]`._"]
                else:
                    z = gap / se if se > 0 else float("inf")
                    verdict = ("clearly larger than seed noise" if z >= 2 else "NOT clearly larger than seed noise - treat as a TIE")
                    L += ["", f"_Top-2 gap: {gap:.4f} BPB = {z:.1f} x the standard error of the difference ({a} vs {b}) -> {verdict}._"]
        L += ["", "## Provenance", "", "```json", json.dumps(provenance, indent=1, default=str), "```",
              "", f"Stage times (s): {self.timings}", "",
              "_Caveats: intrinsic metrics are proxies; BPB comes from a small model at fixed sequence length "
              "(low-fertility tokenizers see more text per window). Confirm the winner on the larger DGX proxy._"]
        (self.root / "report.md").write_text("\n".join(L))
        (self.root / "timings.json").write_text(json.dumps(self.timings))
        (self.root / "proxy_summary.json").write_text(json.dumps(
            {n: {k: p.get(k) for k in ("bpb", "bpb_std", "n_seeds", "bpb_seeds")} for n, p in proxy.items()}, indent=1))
        for f in ("report.md", "results.csv", "timings.json", "proxy_summary.json"):
            self.sync.mark(self.root / f)
        self.sync.mark(best)
        self.sync.flush(force=True, message=f"{self.root.name}: report + best ({win})")
        _log(f"report -> {self.root / 'report.md'}; best -> {best}")
        return win


def run(cfg: dict, stage: str = "all") -> str:
    registry.load_plugins(cfg.get("plugins_dir", []))
    root = Path(cfg["paths"]["runs"]) / cfg["name"]
    pre = Sync(cfg, root, Path(cfg["paths"]["clean"]), log=_log)
    if pre.enabled and pre.restore_on_start:  # fresh session: pull finished work first
        pre.restore("all" if pre.include_data else "run")
    r = Run(cfg)
    try:
        results = r.stage_intrinsic()
        proxy = r.stage_proxy(results) if stage in ("all", "proxy") else {}
        baselines = r.stage_baselines() if stage in ("all", "baselines", "proxy") else {}
        return r.finish(results, proxy, baselines)
    finally:  # crash, Ctrl-C or normal exit: never leave finished work un-pushed
        r.sync.flush(force=True, message=f"{r.root.name}: final flush")
