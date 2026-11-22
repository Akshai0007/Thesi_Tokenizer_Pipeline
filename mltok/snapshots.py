"""Verified, content-addressed snapshots for revision-aware stores."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .lib.artifact_paths import relative_path, unlinked_path
from .lib.atomic import atomic_writer

_DIGEST = re.compile(r"[0-9a-f]{64}")


class SnapshotError(RuntimeError):
    """Snapshot integrity, compatibility, or local destination failure."""


class NoSnapshot(SnapshotError):
    """No snapshot was published for this experiment."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_identity(path, opener=open):
    digest = hashlib.sha256()
    size = 0
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _write(path, value):
    with atomic_writer(path) as stream:
        stream.write(canonical(value))


def _relative(value):
    try:
        p = relative_path(value)
        if any(part.startswith(".") for part in p.parts):
            raise ValueError
        return p
    except (TypeError, ValueError):
        raise SnapshotError("Invalid snapshot file path") from None


def _safe(path):
    try:
        return unlinked_path(path)
    except ValueError:
        raise DestinationConflict("Snapshot path contains a symlink or reparse point") from None


def pending_restore_path(run):
    return run.parent / ".mltok-snapshots" / f"{identity(run.name)}.restore.json"


def restored_corpus_path(run):
    return run.parent / ".mltok-snapshots" / f"{identity(run.name)}.restored.json"


def _download_path(store, remote, revision, directory):
    path = _safe(store.fetch_at(remote, revision, directory))
    if not path.is_relative_to(_safe(directory)) or not path.is_file():
        raise SnapshotError("Downloaded artifact is outside staging or is not a file")
    return path


def _download_directory():
    return tempfile.TemporaryDirectory(prefix="ml-", dir=Path(tempfile.gettempdir()).resolve())


@contextmanager
def _lock(path):
    """One local writer across processes. The OS releases the lock after a crash."""
    path = _safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SnapshotError("Another process is synchronizing this experiment") from None
        yield


class Snapshots:
    def __init__(self, cfg, store, run, clean, log):
        self.store, self.log = store, log
        self.roots = {"run": run}
        if clean is not None:
            self.roots["data"] = clean
        self.experiment = run.name
        relevant = {k: cfg[k] for k in (
            "sources", "filters", "split", "components", "search", "resources", "specials",
            "metrics", "selector", "baselines", "proxy", "plugins_dir",
        ) if k in cfg}
        self.config = identity(relevant)
        self.prefix = f"snapshots/v1/experiments/{identity(self.experiment)}"
        local = run.parent / ".mltok-snapshots"
        self.lock = local / f"{identity(self.experiment)}.lock"
        self.pending = pending_restore_path(run)
        self.receipt = restored_corpus_path(run)
        self._digests = {}

    def _blob(self, digest):
        return f"snapshots/v1/blobs/{digest}"

    def _inventory(self, scope, include_cache=True, include_ckpt=True):
        root = self.roots.get(scope)
        if root is None:
            return {}
        root = _safe(root)
        if scope == "data" and (root / ".mltok-cleaning").exists():
            raise SnapshotError("Cleaned corpus has an unfinished write")
        if not root.exists():
            return {}
        files = {}
        for path in sorted(root.rglob("*")):
            _safe(path)
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if path.suffix in (".tmp", ".part", ".pyc", ".lock"):
                continue
            if scope == "run" and (rel.name == "cfg.json" or rel.name.startswith("proxy_worker_gpu")):
                continue
            if scope == "run" and rel.parts[0] == "best" and not (root / "best" / "SELECTION.json").is_file():
                continue
            if scope == "run" and rel.parts[0] == "variants" and len(rel.parts) > 2:
                if not (root / "variants" / rel.parts[1] / "tokenizer.ok").is_file():
                    continue
            if not include_cache and rel.parts[0] == "cache":
                continue
            if not include_ckpt and path.suffix == ".ckpt":
                continue
            if scope == "data" and rel.as_posix() != "clean_stats.json":
                if rel.parts[0] not in ("train", "eval") or ".jsonl" not in path.name:
                    continue
            st = path.stat()
            files[rel.as_posix()] = (path, (st.st_ino, st.st_size, st.st_mtime_ns))
        return files

    def _dataset(self):
        files = self._inventory("data")
        shards = {rel: record for rel, record in files.items() if rel.startswith(("train/", "eval/"))}
        if not shards:
            return None
        records = []
        for rel, (path, signature) in shards.items():
            cached = self._digests.get(path)
            if cached is None or cached[0] != signature:
                opener = gzip.open if path.suffix == ".gz" else open
                cached = self._digests[path] = (signature, file_identity(path, opener))
            records.append((rel, *cached[1]))
        if self._inventory("data") != files:
            raise SnapshotError("Cleaned corpus changed during snapshot capture")
        return identity(records)

    def check_restored_corpus(self):
        if not self.receipt.exists():
            return
        saved = json.loads(_safe(self.receipt).read_bytes())
        if (saved["root"] == str(self.roots["run"].resolve()) and saved["dataset"] is not None
                and saved["dataset"] != self._dataset()):
            raise SnapshotError("Local cleaned corpus does not match the restored run. Restore the matching corpus or use a fresh run.")

    def _read_manifest(self, remote, revision):
        name = PurePosixPath(remote).name
        if not _DIGEST.fullmatch(name.removesuffix(".json")) or not name.endswith(".json"):
            raise SnapshotError("Invalid snapshot identity")
        try:
            with _download_directory() as download:
                path = _download_path(self.store, remote, revision, download)
                manifest = json.loads(path.read_bytes())
            body = manifest["snapshot"]
            if manifest["id"] != name[:-5] or identity(body) != manifest["id"]:
                raise SnapshotError("Snapshot manifest digest does not match")
            if body["version"] != 1 or body["experiment"] != self.experiment:
                raise SnapshotError("Snapshot experiment does not match")
            if not _DIGEST.fullmatch(body["config"]) or (
                    body["dataset"] is not None and not _DIGEST.fullmatch(body["dataset"])):
                raise SnapshotError("Invalid configuration or dataset fingerprint")
            if not isinstance(manifest["revision"], str) or not re.fullmatch(r"[a-f0-9]{40,64}", manifest["revision"]):
                raise SnapshotError("Invalid snapshot revision")
            if datetime.fromisoformat(manifest["completed_at"]).utcoffset() is None:
                raise SnapshotError("Snapshot completion time has no timezone")
            seen = set()
            for item in body["files"]:
                scope, p = item["scope"], _relative(item["path"])
                key = scope, p.as_posix().casefold()
                if (scope not in ("run", "data") or key in seen or type(item["size"]) is not int
                        or item["size"] < 0 or not _DIGEST.fullmatch(item["sha256"])):
                    raise SnapshotError("Invalid snapshot inventory")
                seen.add(key)
            if not seen:
                raise SnapshotError("Empty snapshot inventory")
            for scope, rel in seen:
                if any((scope, parent.as_posix()) in seen for parent in PurePosixPath(rel).parents):
                    raise SnapshotError("Snapshot contains conflicting file paths")
        except (KeyError, TypeError, ValueError, AttributeError):
            raise SnapshotError("Malformed snapshot manifest") from None
        return manifest

    def _verify(self, item, revision, temp):
        path = _download_path(self.store, self._blob(item["sha256"]), revision, temp)
        if file_identity(path) != (item["size"], item["sha256"]):
            raise SnapshotError("Remote artifact size or SHA-256 does not match")
        return path

    def publish(self, scopes, include_cache, include_ckpt):
        with _lock(self.lock), tempfile.TemporaryDirectory(prefix="mltok-snapshot-", dir=self.lock.parent) as directory:
            if _safe(self.pending).exists():
                raise SnapshotError("Finish the interrupted restore before publishing")
            temp = Path(directory)
            captured = {s: self._inventory(s, include_cache, include_ckpt) for s in sorted(scopes)}
            if "data" in captured:
                names = captured["data"]
                if "clean_stats.json" not in names or not all(
                        any(rel.startswith(split + "/") for rel in names) for split in ("train", "eval")):
                    raise SnapshotError("Cleaned corpus is incomplete")
            files, blobs = [], {}
            for scope, inventory in captured.items():
                if len({rel.casefold() for rel in inventory}) != len(inventory):
                    raise SnapshotError("Snapshot paths collide on a case-insensitive filesystem")
                for rel, (source, _) in inventory.items():
                    _relative(rel)
                    staged = temp / str(len(files))
                    if source.suffix == ".ckpt":
                        try:
                            os.link(source, staged)  # checkpoint generations use atomic replacement
                        except OSError:
                            shutil.copyfile(source, staged)
                    else:
                        shutil.copyfile(source, staged)
                    if source.name in ("tokenizer.ok", "variant.json", "SELECTION.json", "clean_stats.json"):
                        try:
                            metadata = json.loads(staged.read_bytes())
                            if not isinstance(metadata, dict):
                                raise ValueError
                        except (ValueError, UnicodeError):
                            raise SnapshotError("Local completion metadata is incomplete") from None
                        self._required_files(scope, rel, metadata, inventory)
                    size, digest = file_identity(staged)
                    files.append({"scope": scope, "path": rel, "size": size, "sha256": digest})
                    blobs[digest] = staged
            if not files:
                return None
            dataset = self._dataset()
            if any(self._inventory(s, include_cache, include_ckpt) != inv for s, inv in captured.items()):
                raise SnapshotError("Local artifacts changed during snapshot capture")
            body = {"version": 1, "experiment": self.experiment, "config": self.config,
                    "dataset": dataset, "files": files}
            snapshot = identity(body)
            remote = f"{self.prefix}/{snapshot}.json"
            revision = self.store.revision(create=True)
            existing = self.store.list_at(revision)
            if remote in existing:
                manifest = self._read_manifest(remote, revision)
                self._verify_all(files, manifest["revision"])
                return snapshot
            pending = [(path, self._blob(digest)) for digest, path in blobs.items()
                       if self._blob(digest) not in existing]
            for start in range(0, len(pending), 100):
                revision = self.store.push_at(pending[start:start + 100], "mltok snapshot artifacts", revision)
            self._verify_all(files, revision)
            marker = temp / "manifest.json"
            _write(marker, {"snapshot": body, "id": snapshot, "revision": revision,
                            "completed_at": datetime.now(timezone.utc).isoformat()})
            published = self.store.push_at([(marker, remote)], "mltok verified snapshot", revision)
            recorded = self._read_manifest(remote, published)
            if recorded["revision"] != revision:
                raise SnapshotError("Published manifest does not record the verified artifact revision")
            return snapshot

    def _required_files(self, scope, rel, metadata, inventory):
        if scope == "data" and rel == "clean_stats.json" and "shards" in metadata:
            expected = {p for p in inventory if p.startswith(("train/", "eval/"))}
            if not isinstance(metadata["shards"], list) or metadata["shards"] != sorted(expected):
                raise SnapshotError("Cleaned corpus shards do not match its completion record")
        marker = PurePosixPath(rel)
        if scope != "run" or marker.name != "tokenizer.ok" or marker.parts[0] != "variants":
            return
        variant = (marker.parent / "variant.json").as_posix()
        if variant not in inventory:
            raise SnapshotError("Tokenizer variant metadata is missing")
        if "_artifact_paths" in metadata:
            required = metadata["_artifact_paths"]
            if not isinstance(required, list) or not required:
                raise SnapshotError("Tokenizer completion record has no artifacts")
        else:
            # Historical artifacts predate explicit inventories. Check known backend files.
            spec = json.loads(inventory[variant][0].read_bytes()).get("algorithm", {})
            algorithm = spec.get("type") if isinstance(spec, dict) else spec
            required = {"spm_bpe": ["spm.model", "spm.vocab"], "spm_unigram": ["spm.model", "spm.vocab"],
                        "hf_bpe": ["tokenizer.json"], "mlbpe": ["tokenizer.json"],
                        "mlbpe_plain": ["tokenizer.json"], "mlbpe_morph": ["tokenizer.json"]}.get(algorithm, [])
        if any((marker.parent / _relative(path)).as_posix() not in inventory for path in required):
            raise SnapshotError("A required tokenizer artifact is missing")

    def _verify_all(self, files, revision):
        seen = set()
        for item in files:
            if item["sha256"] in seen:
                continue
            with _download_directory() as directory:
                self._verify(item, revision, directory)
            seen.add(item["sha256"])

    def _manifests(self, revision):
        names = sorted(p for p in self.store.list_at(revision) if p.startswith(self.prefix + "/"))
        manifests = []
        for name in names:
            try:
                manifests.append(self._read_manifest(name, revision))
            except (FileNotFoundError, SnapshotError) as exc:
                self.log(f"[snapshot] invalid manifest ignored: {exc}")
        if names and not manifests:
            raise SnapshotError("Snapshot manifests exist but none passed verification")
        return sorted(manifests, key=lambda m: (datetime.fromisoformat(m["completed_at"]), m["id"]), reverse=True)

    def cleanup_plan(self, retain=2):
        """List older manifests for review. Blobs and manifests remain untouched."""
        if retain < 2:
            raise SnapshotError("At least two recovery generations must be retained")
        manifests = self._manifests(self.store.revision())
        return {"delete": [], "older_manifests_for_review": [
            f"{self.prefix}/{m['id']}.json" for m in manifests[retain:]]}

    def restore(self, scopes, snapshot=None):
        with _lock(self.lock):
            if not scopes <= self.roots.keys():
                raise SnapshotError("Restore destination is not configured")
            roots = {scope: str(_safe(self.roots[scope])) for scope in scopes}
            if len(roots) == 2:
                run, data = Path(roots["run"]), Path(roots["data"])
                if run == data or run in data.parents or data in run.parents:
                    raise DestinationConflict("Run and data restore destinations must be separate")
            pending_path = _safe(self.pending)
            pending = json.loads(pending_path.read_bytes()) if pending_path.exists() else None
            if pending and (roots != pending["roots"] or (snapshot and snapshot != pending["id"])):
                raise SnapshotError("Finish the interrupted restore with the same scope and snapshot")
            selected = pending["id"] if pending else snapshot
            existing_dataset = self._dataset() if "data" not in scopes else None
            manifests = self._manifests(self.store.revision())
            if not manifests and not selected:
                raise NoSnapshot("No verified snapshot for this experiment. Legacy remote files are not restored automatically.")
            for manifest in manifests:
                body = manifest["snapshot"]
                if selected and selected != manifest["id"]:
                    continue
                if pending and pending.get("revision", manifest["revision"]) != manifest["revision"]:
                    continue
                if (body["config"] != self.config or not scopes <= {f["scope"] for f in body["files"]}
                        or (existing_dataset is not None and body["dataset"] != existing_dataset)):
                    continue
                try:
                    if scopes == {"data"} and self._inventory("run"):
                        raise DestinationConflict("Restore data into a fresh run or restore both scopes together")
                    return self._restore_one(manifest, scopes, roots), manifest["id"]
                except (FileNotFoundError, SnapshotError) as exc:
                    # A local conflict or an interrupted installation must not trigger a different restore.
                    if self.pending.exists() or isinstance(exc, DestinationConflict):
                        raise
                    self.log(f"[snapshot] {manifest['id']} failed verification: {exc}")
            raise SnapshotError("No usable compatible verified snapshot. Legacy files require explicit export and verification.")

    def _restore_one(self, manifest, scopes, roots):
        items = [i for i in manifest["snapshot"]["files"] if i["scope"] in scopes]
        with ExitStack() as stack:
            staging = {}
            for scope in sorted(scopes):
                root = _safe(self.roots[scope])
                root.parent.mkdir(parents=True, exist_ok=True)
                staging[scope] = Path(stack.enter_context(tempfile.TemporaryDirectory(
                    prefix=".mltok-stage-", dir=root.parent))) / "snapshot"
                staging[scope].mkdir()
            for item in items:
                dest = staging[item["scope"]] / _relative(item["path"])
                dest.parent.mkdir(parents=True, exist_ok=True)
                with _download_directory() as download:
                    source = self._verify(item, manifest["revision"], download)
                    with source.open("rb") as inp, atomic_writer(dest) as out:
                        shutil.copyfileobj(inp, out)
                if file_identity(dest) != (item["size"], item["sha256"]):
                    raise SnapshotError("Staged artifact size or SHA-256 does not match")
            install = []
            for scope in sorted(scopes):
                root = _safe(self.roots[scope])
                if root.exists() and not root.is_dir():
                    raise DestinationConflict("Restore destination is not a directory")
                if root.exists() and any(root.iterdir()):
                    expected = {i["path"]: (i["size"], i["sha256"]) for i in items if i["scope"] == scope}
                    actual = {}
                    for p in root.rglob("*"):
                        _safe(p)
                        if p.is_file():
                            actual[p.relative_to(root).as_posix()] = file_identity(p)
                    if actual != expected:
                        raise DestinationConflict("Restore destination differs from the verified snapshot. Choose a fresh directory.")
                else:
                    install.append(scope)
            if install:
                _write(self.pending, {"id": manifest["id"], "revision": manifest["revision"], "roots": roots})
                for scope in install:
                    root = _safe(self.roots[scope])
                    if root.exists():
                        root.rmdir()  # only an empty directory
                    os.replace(staging[scope], root)
                    if os.name == "posix":
                        fd = os.open(root.parent, os.O_RDONLY)
                        try:
                            os.fsync(fd)
                        finally:
                            os.close(fd)
            if "run" in scopes:
                _write(_safe(self.receipt), {"root": roots["run"], "dataset": manifest["snapshot"]["dataset"],
                                      "id": manifest["id"]})
            self.pending.unlink(missing_ok=True)
            return sum(i["scope"] in install for i in items)


class DestinationConflict(SnapshotError):
    """An existing destination must be preserved."""
