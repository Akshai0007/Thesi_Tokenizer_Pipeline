"""ENGINE: checkpoint sync. Talks to whatever `store` plugin the config names; knows nothing about Hugging Face.

Policy
------
* Finished work marks files dirty. Revision-aware stores publish verified snapshots of
  the selected scopes. Other stores upload batches of changed files through the original API.
* Pushes are throttled (`sync.min_interval_s`, default 120 s) so Hub commit rate limits are not hit;
  stage ends, errors and exits flush immediately.
* Proxy-LM checkpoints have their own slower interval (`sync.ckpt_push_interval_s`,
  default 900 s). The current and previous generations are retained after publication.
* Pushes run in a BACKGROUND thread: training never waits for the network (a Hub outage cannot stall it).
  Stage ends, errors and exits (`force=True`) wait for the push to finish.
* A failing store NEVER stops the run: the error is logged, files stay queued, the next flush retries.
* Verified restore stages one compatible snapshot before installation. Legacy stores
  retain their original per-file restore behavior.

Config:
  store: {type: hf_hub, repo: user/name, ...}      # or none (default)
  sync:  {enabled: true, min_interval_s: 120, ckpt_push_interval_s: 900, restore: true,
          include_cache: true, include_checkpoints: true, include_data: false}
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import registry
from .contracts import StoreAccessError
from .snapshots import NoSnapshot, SnapshotError, Snapshots

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
        self.snapshot = Snapshots(cfg, self.store, self.root, self.clean_dir, log) if self.store.verified_snapshots else None
        self.last_error = None
        self._pushed: dict[Path, tuple] = {}   # file -> (mtime_ns, size) of the last successful push
        self._dirty: dict[Path, str] = {}
        self._ckpt: dict[Path, str] = {}
        self._to_delete: list[str] = []
        self._last = 0.0
        self._last_ckpt = time.time()   # a checkpoint is pushed only after a FULL interval: short jobs never upload one
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mltok-sync")
        self._inflight = None
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
        with self._lock:
            self._mark_files(files)

    def _mark_files(self, files) -> None:
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
        if not self.enabled or self.snapshot is not None:
            return
        rel = self.remote_for(Path(path))
        with self._lock:
            self._ckpt.pop(Path(path).resolve(), None)
            if rel:
                self._to_delete.append(rel)

    # ------------------------------------------------------------------ pushing
    def wait(self) -> None:
        """Block until any background push has finished."""
        fut = self._inflight
        if fut is not None:
            fut.result()

    def flush(self, force: bool = False, message: str = "", block: bool | None = None) -> bool:
        """Upload queued files. `force` ignores the throttle and WAITS (stage end / error / exit); a normal flush
        is throttled and runs in the background. Returns True if nothing is left queued."""
        if not self.enabled:
            return True
        if self.snapshot is not None:
            scopes = {"run", "data"} if self.include_data else {"run"}
            return self._flush_snapshot(scopes, force, force if block is None else block)
        block = force if block is None else block
        if block:
            self.wait()
        with self._lock:
            if self._inflight is not None and not self._inflight.done():
                return False  # a push is running; files stay queued and go with the next flush
            now = time.time()
            batch = {}
            if self._dirty and (force or now - self._last >= self.min_interval):
                batch.update(self._dirty)
            if self._ckpt and (force or now - self._last_ckpt >= self.ckpt_interval):
                batch.update(self._ckpt)
            deletes = list(self._to_delete) if (force or batch) else []
            if not batch and not deletes:
                return not (self._dirty or self._ckpt)
            items = []
            for f, r in batch.items():
                if f.exists():
                    st = f.stat()
                    items.append((f, r, (st.st_mtime_ns, st.st_size)))
            self._to_delete = [d for d in self._to_delete if d not in deletes]
        if block:
            return self._push_batch(items, message, deletes) and not self._dirty
        self._inflight = self._pool.submit(self._push_batch, items, message, deletes)
        return False

    def _push_batch(self, items, message, deletes) -> bool:
        ok = True
        try:
            for i in range(0, len(items), 100):  # chunk: keeps each commit small
                chunk = items[i : i + 100]
                self.store.push([(f, r) for f, r, _ in chunk], message or f"{self.name}: {len(chunk)} file(s)")
                with self._lock:
                    for f, _, sig in chunk:
                        if f.exists() and (f.stat().st_mtime_ns, f.stat().st_size) == sig:  # not modified while uploading
                            self._dirty.pop(f, None)
                            self._ckpt.pop(f, None)
                            self._pushed[f] = sig
            if items:
                with self._lock:
                    self.pushes += 1
                    self._last = time.time()
                    if any(self._is_ckpt(f) for f, _, _ in items):
                        self._last_ckpt = self._last
                self.log(f"[sync] pushed {len(items)} file(s)")
            if deletes:
                self.store.delete(deletes)
        except Exception as e:  # noqa: BLE001 - never kill the run because of the store
            ok = False
            with self._lock:
                self.failures += 1
                self._to_delete.extend(deletes)
            self.log(f"[sync] push failed ({type(e).__name__}). Files stay queued for retry.")
        return ok

    def _flush_snapshot(self, scopes, force, block):
        if block:
            self.wait()
        with self._lock:
            if self._inflight is not None and not self._inflight.done():
                return False
            now = time.time()
            due = ((self._dirty and now - self._last >= self.min_interval)
                   or (self._ckpt and now - self._last_ckpt >= self.ckpt_interval))
            if not force and not due:
                return not (self._dirty or self._ckpt)
            queued = {}
            for path, remote in {**self._dirty, **self._ckpt}.items():
                scope = "data" if remote.startswith("data/") else "run"
                if scope in scopes and path.exists():
                    stat = path.stat()
                    queued[path] = (stat.st_mtime_ns, stat.st_size)
            self._last = now
            self._last_ckpt = now
            self._inflight = self._pool.submit(self._publish_snapshot, scopes, queued)
        return self._inflight.result() if block else False

    def _publish_snapshot(self, scopes, queued):
        try:
            snapshot = self.snapshot.publish(scopes, self.include_cache, self.include_ckpt)
        except Exception as exc:
            self.failures += 1
            self.last_error = str(exc) if isinstance(exc, (SnapshotError, StoreAccessError)) else type(exc).__name__
            self.log(f"[sync] remote snapshot failed: {self.last_error}. Local artifacts retained.")
            return False
        with self._lock:
            for path, sig in queued.items():
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                if (stat.st_mtime_ns, stat.st_size) == sig:
                    self._dirty.pop(path, None)
                    self._ckpt.pop(path, None)
                    self._pushed[path] = sig
        self.last_error = None
        if snapshot:
            self.pushes += 1
            self.log(f"[sync] verified remote snapshot {snapshot}")
        else:
            self.log("[sync] no artifacts available for a remote snapshot")
        return bool(snapshot)

    def close(self, message=""):
        try:
            return self.flush(force=True, message=message)
        finally:
            self._pool.shutdown(wait=True)

    # ------------------------------------------------------------------ restoring
    def restore(self, what: str = "run", snapshot_id=None) -> int:
        """Fetch finished work from the store. what: run | data | all. Never overwrites local files."""
        if not self.enabled:
            return 0
        self.last_error = None
        if self.snapshot is not None:
            scopes = {"run", "data"} if what == "all" else {what}
            try:
                count, selected = self.snapshot.restore(scopes, snapshot_id)
                self.log(f"[sync] verified restore {selected}: installed {count} files")
                return count
            except NoSnapshot as exc:
                self.log(f"[sync] {exc}")
                return 0
            except Exception as exc:
                self.failures += 1
                self.last_error = str(exc) if isinstance(exc, (SnapshotError, StoreAccessError)) else type(exc).__name__
                self.log(f"[sync] restore failed: {self.last_error}")
                return 0
        if snapshot_id is not None:
            self.last_error = "The configured store does not support verified snapshots"
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
            self.log(f"[sync] restore failed ({type(e).__name__}). Continuing from local state.")
        return n

    def push_tree(self, path, message: str = "") -> bool:
        """Mark everything under `path` and push right away (used for reports, data, freeze records)."""
        self.mark(path)
        if self.snapshot is not None:
            scopes = {"data"} if self.clean_dir is not None and Path(path) == self.clean_dir else {"run"}
            if scopes == {"run"} and self.include_data:
                scopes.add("data")
            return self._flush_snapshot(scopes, True, True)
        return self.flush(force=True, message=message)
