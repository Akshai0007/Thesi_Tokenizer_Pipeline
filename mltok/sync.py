"""ENGINE: checkpoint sync. Talks to whatever `store` plugin the config names; knows nothing about Hugging Face.

Policy
------
* Every finished unit of work marks its files "dirty": word table, trained tokenizer, metrics,
  each proxy-LM seed, report / best / freeze record. Dirty files are uploaded in ONE batch.
* Pushes are throttled (`sync.min_interval_s`, default 120 s) so Hub commit rate limits are not hit;
  stage ends, errors and exits flush immediately.
* Mid-training proxy-LM checkpoints are big, so they have their own slower interval
  (`sync.ckpt_push_interval_s`, default 900 s) and only the newest one is kept remotely.
* A failing store NEVER stops the run: the error is logged, files stay queued, the next flush retries.
* `restore()` pulls finished work (not overwriting local files) so a fresh session resumes where the
  last one stopped; the content-addressed cache then decides what is still valid.

Config:
  store: {type: hf_hub, repo: user/name, ...}      # or none (default)
  sync:  {enabled: true, min_interval_s: 120, ckpt_push_interval_s: 900, restore: true,
          include_cache: true, include_checkpoints: true, include_data: false}
"""
from __future__ import annotations

import time
from pathlib import Path

from . import registry

_SKIP_SUFFIX = (".tmp", ".part", ".pyc", ".lock")


class Sync:
    def __init__(self, cfg: dict, run_root: Path, clean_dir: Path | None = None, log=print):
        self.log = log
        self.root = Path(run_root)
        self.clean_dir = Path(clean_dir) if clean_dir else None
        self.name = self.root.name
        sc = cfg.get("sync", {}) or {}
        spec = cfg.get("store") or "none"
        self.store = registry.make("store", spec)
        self.enabled = bool(sc.get("enabled", True)) and registry.spec_of(spec)["type"] != "none"
        self.min_interval = float(sc.get("min_interval_s", 120))
        self.ckpt_interval = float(sc.get("ckpt_push_interval_s", 900))
        self.restore_on_start = bool(sc.get("restore", True))
        self.include_cache = bool(sc.get("include_cache", True))
        self.include_ckpt = bool(sc.get("include_checkpoints", True))
        self.include_data = bool(sc.get("include_data", False))
        self._pushed: dict[Path, tuple] = {}   # file -> (mtime_ns, size) of the last successful push
        self._dirty: dict[Path, str] = {}
        self._ckpt: dict[Path, str] = {}
        self._to_delete: list[str] = []
        self._last = 0.0
        self._last_ckpt = 0.0
        self.pushes = 0
        self.failures = 0

    # ------------------------------------------------------------------ mapping
    def remote_for(self, local: Path) -> str | None:
        local = Path(local).resolve()
        for base, prefix in ((self.root, f"runs/{self.name}"), (self.clean_dir, "data/clean")):
            if base is None:
                continue
            try:
                return f"{prefix}/{local.relative_to(base.resolve()).as_posix()}"
            except ValueError:
                continue
        return None

    def _is_ckpt(self, p: Path) -> bool:
        return p.name.endswith(".ckpt")

    # ------------------------------------------------------------------ marking
    def mark(self, path) -> None:
        """Queue a file or a whole directory for the next push."""
        if not self.enabled:
            return
        p = Path(path)
        files = [f.resolve() for f in (p.rglob("*") if p.is_dir() else [p]) if f.is_file()]
        for f in files:
            if f.name.endswith(_SKIP_SUFFIX):
                continue
            rel = self.remote_for(f)
            if rel is None:
                continue
            st = f.stat()
            if self._pushed.get(f) == (st.st_mtime_ns, st.st_size):
                continue  # unchanged since the last successful push
            if rel.startswith("data/") and not self.include_data:
                continue  # cleaned data can be gigabytes: opt-in only (sync.include_data)
            if self._is_ckpt(f):
                if self.include_ckpt:
                    self._ckpt[f] = rel
            elif "/cache/" in f"/{rel}" and not self.include_cache:
                continue
            else:
                self._dirty[f] = rel

    def mark_delete(self, path) -> None:
        """A local file was removed (e.g. a finished checkpoint): remove its remote copy too."""
        if not self.enabled:
            return
        rel = self.remote_for(Path(path))
        self._ckpt.pop(Path(path).resolve(), None)
        if rel:
            self._to_delete.append(rel)

    # ------------------------------------------------------------------ pushing
    def flush(self, force: bool = False, message: str = "") -> bool:
        """Upload queued files. Returns True if nothing is left queued."""
        if not self.enabled:
            return True
        now = time.time()
        batch = {}
        if self._dirty and (force or now - self._last >= self.min_interval):
            batch.update(self._dirty)
        if self._ckpt and (force or now - self._last_ckpt >= self.ckpt_interval):
            batch.update(self._ckpt)
        if not batch and not (force and self._to_delete):
            return not (self._dirty or self._ckpt)
        files = [(f, r) for f, r in batch.items() if f.exists()]
        ok = True
        try:
            for i in range(0, len(files), 100):  # chunk: keeps each commit small
                chunk = files[i : i + 100]
                self.store.push(chunk, message or f"{self.name}: {len(chunk)} file(s)")
                for f, _ in chunk:
                    self._dirty.pop(f, None)
                    self._ckpt.pop(f, None)
                    if f.exists():
                        st = f.stat()
                        self._pushed[f] = (st.st_mtime_ns, st.st_size)
            if files:
                self.pushes += 1
                self._last = time.time()
                if any(self._is_ckpt(f) for f, _ in files):
                    self._last_ckpt = self._last
                self.log(f"[sync] pushed {len(files)} file(s)")
            if self._to_delete:
                self.store.delete(self._to_delete)
                self._to_delete = []
        except Exception as e:  # noqa: BLE001 - never kill the run because of the store
            ok = False
            self.failures += 1
            self.log(f"[sync] push failed ({type(e).__name__}: {str(e)[:160]}); files stay queued, will retry")
        return ok and not (self._dirty and force)

    # ------------------------------------------------------------------ restoring
    def restore(self, what: str = "run") -> int:
        """Fetch finished work from the store. what: run | data | all. Never overwrites local files."""
        if not self.enabled:
            return 0
        n = 0
        try:
            if what in ("run", "all"):
                got = self.store.pull(f"runs/{self.name}", self.root, skip_existing=True)
                self.log(f"[sync] restored {got} run file(s) from the store")
                n += got
            if what in ("data", "all") and self.clean_dir is not None:
                got = self.store.pull("data/clean", self.clean_dir, skip_existing=True)
                self.log(f"[sync] restored {got} cleaned-data file(s) from the store")
                n += got
        except Exception as e:  # noqa: BLE001
            self.failures += 1
            self.log(f"[sync] restore failed ({type(e).__name__}: {str(e)[:160]}); continuing from local state")
        return n

    def push_tree(self, path, message: str = "") -> None:
        """Mark everything under `path` and push right away (used for reports, data, freeze records)."""
        self.mark(path)
        self.flush(force=True, message=message)
