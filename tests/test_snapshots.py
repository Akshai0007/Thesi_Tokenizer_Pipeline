"""Offline Hub revision and snapshot fault injection."""
import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from mltok.contracts import StoreAccessError
from mltok.plugins.store.hf_hub import HFHubStore
from mltok.snapshots import identity
from mltok.sync import Sync


class HttpError(RuntimeError):
    def __init__(self, status):
        super().__init__("credential fixture must not appear")
        self.response = SimpleNamespace(status_code=status)


class RemoteEntryNotFoundError(HttpError):
    pass


class RevisionNotFoundError(HttpError):
    pass


class Hub:
    def __init__(self):
        self.head = "0" * 40
        self.revisions = {self.head: {}}
        self.commits = []
        self.downloads = []
        self.private = True
        self.fail_commit = None
        self.fail_download = None
        self.access = self.auth = None

    def create_repo(self, repo, **kwargs):
        assert kwargs["private"]

    def repo_info(self, repo, **kwargs):
        if self.access:
            raise HttpError(self.access)
        return SimpleNamespace(sha=self.head, private=self.private)

    def whoami(self):
        if self.auth:
            raise HttpError(self.auth)
        return {"name": "fixture"}

    def list_repo_files(self, repo, *, revision, **kwargs):
        return list(self.revisions[revision])

    def create_commit(self, repo_id, operations, *, parent_commit, **kwargs):
        assert parent_commit == self.head
        changes = {op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in operations}
        if self.fail_commit:
            self.fail_commit(changes)
        head = f"{len(self.revisions):040x}"
        self.revisions[head] = {**self.revisions[self.head], **changes}
        self.head = head
        self.commits.append(changes)
        return SimpleNamespace(oid=head)

    def hf_hub_download(self, repo_id, filename, *, revision, local_dir, force_download, **kwargs):
        assert force_download
        if revision not in self.revisions:
            raise RevisionNotFoundError(404)
        self.downloads.append((filename, revision))
        if filename not in self.revisions[revision]:
            raise RemoteEntryNotFoundError(404)
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.fail_download:
            self.fail_download(filename, dest)
        dest.write_bytes(self.revisions[revision][filename])
        return str(dest)

    def manifests(self):
        return [json.loads(v) for p, v in self.revisions[self.head].items() if "/experiments/" in p]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    hub = Hub()
    monkeypatch.setattr(HFHubStore, "api_factory", lambda self: hub)
    cfg = {"store": {"type": "hf_hub", "repo": "fixture/private", "retries": 1},
           "components": {"vocab": 1600}, "sync": {"min_interval_s": 0, "ckpt_push_interval_s": 0}}
    root = tmp_path / "local" / "experiment"
    root.mkdir(parents=True)
    (root / "progress.ckpt").write_bytes(b"checkpoint A")
    (root / "metrics.json").write_bytes(b"metrics")
    logs = []

    def make(where=root, config=cfg, clean=None):
        return Sync(config, where, clean, log=logs.append)

    return hub, cfg, root, make, logs


def test_verified_upload_restore_and_idempotent_retry(experiment, tmp_path):
    hub, _, root, make, logs = experiment
    assert make().push_tree(root)
    manifest, = hub.manifests()
    assert manifest["id"] == identity(manifest["snapshot"])
    assert all(rev == manifest["revision"] for name, rev in hub.downloads if "/blobs/" in name)
    assert len(hub.commits) == 2 and "verified remote snapshot" in logs[-1]
    assert make().push_tree(root) and len(hub.commits) == 2
    target = tmp_path / "restore" / root.name
    assert make(target).restore(snapshot_id=manifest["id"]) == 2
    assert (target / "progress.ckpt").read_bytes() == b"checkpoint A"


@pytest.mark.parametrize("phase", ["blob_commit", "verification", "manifest_commit"])
def test_failed_publication_preserves_prior_snapshot_and_retries(experiment, phase):
    hub, _, root, make, _ = experiment
    assert make().push_tree(root)
    old, = hub.manifests()
    (root / "progress.ckpt").write_bytes(b"checkpoint B")

    def fail_commit(changes):
        if any(("/experiments/" in path) == (phase == "manifest_commit") for path in changes):
            raise ConnectionError("interrupted commit")

    def fail_download(name, dest):
        if "/blobs/" in name:
            dest.write_bytes(b"partial")
            raise ConnectionError("interrupted download")

    if phase == "verification":
        hub.fail_download = fail_download
    else:
        hub.fail_commit = fail_commit
    failed = make()
    assert not failed.push_tree(root) and failed.last_error
    assert hub.manifests() == [old]
    hub.fail_commit = hub.fail_download = None
    assert make().push_tree(root)
    assert len(hub.manifests()) == 2
    blobs = [path for commit in hub.commits for path in commit if "/blobs/" in path]
    assert len(blobs) == len(set(blobs)) == 3


@pytest.mark.parametrize("damage", ["missing", "size", "sha256", "manifest", "revision"])
def test_corrupt_newest_snapshot_falls_back(experiment, tmp_path, damage):
    hub, _, root, make, _ = experiment
    assert make().push_tree(root)
    (root / "progress.ckpt").write_bytes(b"checkpoint B")
    assert make().push_tree(root)
    newest = hub.manifests()[-1]
    item = next(i for i in newest["snapshot"]["files"] if i["path"] == "progress.ckpt")
    remote = "mltok/snapshots/v1/blobs/" + item["sha256"]
    revision = hub.revisions[newest["revision"]]
    if damage == "missing":
        del revision[remote]
    elif damage == "revision":
        del hub.revisions[newest["revision"]]
    elif damage == "manifest":
        head = hub.revisions[hub.head]
        name = next(p for p in head if p.endswith(newest["id"] + ".json"))
        newest["snapshot"]["files"][0]["sha256"] = "0" * 64
        head[name] = json.dumps(newest).encode()
    else:
        revision[remote] = b"bad" if damage == "size" else b"checkpoint C"
    target = tmp_path / "restore" / root.name
    assert make(target).restore() == 2
    assert (target / "progress.ckpt").read_bytes() == b"checkpoint A"


def test_incompatible_snapshot_and_corrupt_local_destination(experiment, tmp_path):
    hub, cfg, root, make, _ = experiment
    assert make().push_tree(root)
    target = tmp_path / "other" / root.name
    incompatible = make(target, {**cfg, "components": {"vocab": 2000}})
    assert incompatible.restore(snapshot_id=hub.manifests()[0]["id"]) == 0
    assert "compatible" in incompatible.last_error and not target.exists()
    target.mkdir(parents=True)
    (target / "progress.ckpt").write_bytes(b"corrupt local")
    conflicted = make(target)
    assert conflicted.restore() == 0 and "fresh directory" in conflicted.last_error
    assert list(target.iterdir()) == [target / "progress.ckpt"]
    assert (target / "progress.ckpt").read_bytes() == b"corrupt local"


@pytest.mark.parametrize(("access", "auth", "private", "expected"), [
    (401, 401, True, "expired"), (401, None, True, "authenticated account"),
    (403, None, True, "permission denied"), (404, None, True, "unavailable"),
    (None, None, False, "not confirmed private"),
])
def test_hub_access_failures_are_explicit_and_redacted(experiment, access, auth, private, expected):
    hub, _, root, make, logs = experiment
    hub.access, hub.auth, hub.private = access, auth, private
    sync = make()
    assert sync.restore() == 0 and expected in sync.last_error
    assert not sync.push_tree(root) and expected in sync.last_error
    assert not hub.commits and all("credential fixture" not in line for line in logs)


def test_interrupted_restore_leaves_destination_untouched(experiment, tmp_path):
    hub, _, root, make, _ = experiment
    assert make().push_tree(root)
    target = tmp_path / "restore" / root.name

    def interrupt(name, dest):
        if "/blobs/" in name:
            dest.write_bytes(b"partial")
            raise ConnectionError("download interrupted")

    hub.fail_download = interrupt
    assert make(target).restore() == 0 and not target.exists()
    hub.fail_download = None
    assert make(target).restore() == 2


def test_background_failure_and_shutdown_retry(experiment):
    hub, _, root, make, logs = experiment
    started, release = threading.Event(), threading.Event()

    def interrupt_once(changes):
        started.set()
        assert release.wait(5)
        hub.fail_commit = None
        raise ConnectionError("first upload failed")

    hub.fail_commit = interrupt_once
    sync = make()
    sync.mark(root)
    assert sync.flush() is False
    assert started.wait(5)
    release.set()
    assert sync.close()
    assert sync.failures == 1 and len(hub.manifests()) == 1
    assert any("failed" in line for line in logs) and "verified" in logs[-1]


def test_cleanup_is_dry_run_and_retains_latest_two(experiment):
    hub, _, root, make, _ = experiment
    for i in range(3):
        (root / "metrics.json").write_text(str(i))
        assert make().push_tree(root)
    commits = list(hub.commits)
    plan = make().snapshot.cleanup_plan()
    assert not plan["delete"]
    assert len(plan["older_manifests_for_review"]) == 1 and hub.commits == commits
    assert plan["older_manifests_for_review"][0].endswith(hub.manifests()[0]["id"] + ".json")


def test_all_restore_retries_same_snapshot_after_first_install(experiment, tmp_path, monkeypatch):
    hub, cfg, root, make, _ = experiment
    clean = tmp_path / "clean"
    for split in ("train", "eval"):
        shard = clean / split / "shard.jsonl"
        shard.parent.mkdir(parents=True)
        shard.write_text('{"text":"sample"}\n')
    sync = make(root, {**cfg, "sync": {"include_data": True}}, clean)
    assert not sync.push_tree(root) and not hub.manifests()
    (clean / "clean_stats.json").write_text("{}")
    assert sync.push_tree(root)
    target = tmp_path / "elsewhere" / root.name
    target_data = tmp_path / "restored-clean"
    restore = make(target, cfg, target_data)
    replace = os.replace

    def interrupt(source, destination):
        if destination == target:
            raise OSError("simulated directory installation failure")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt)
        assert restore.restore("all") == 0
    assert restore.snapshot.pending.is_file()
    assert target_data.exists() and not target.exists()
    assert restore.restore("run") == 0 and restore.snapshot.pending.is_file()
    assert restore.restore("all") == 2 and not restore.snapshot.pending.exists()
    restore.snapshot.check_restored_corpus()
    (target_data / "train" / "shard.jsonl").write_text('{"text":"change"}\n')
    from mltok.snapshots import SnapshotError

    with pytest.raises(SnapshotError, match="does not match"):
        restore.snapshot.check_restored_corpus()


def test_second_writer_does_not_publish_during_pending_upload(experiment):
    hub, _, root, make, _ = experiment
    started, release = threading.Event(), threading.Event()

    def pause(changes):
        started.set()
        assert release.wait(5)

    hub.fail_commit = pause
    primary = make()
    primary.mark(root)
    primary.flush()
    assert started.wait(5)
    try:
        competing = make()
        assert not competing.push_tree(root)
        assert "Another process" in competing.last_error
    finally:
        release.set()
    assert primary.close() and len(hub.manifests()) == 1


def test_missing_required_local_artifacts_prevent_publication(experiment, tmp_path):
    hub, cfg, root, make, _ = experiment
    variant = root / "variants" / "fixture"
    variant.mkdir(parents=True)
    (variant / "variant.json").write_text("{}")
    (variant / "tokenizer.ok").write_text(json.dumps({"_artifact_paths": ["tokenizer.json"]}))
    sync = make()
    assert not sync.push_tree(root) and "required tokenizer artifact" in sync.last_error
    (variant / "tokenizer.json").write_text("{}")
    clean = tmp_path / "clean"
    for split in ("train", "eval"):
        shard = clean / split / "shard.jsonl"
        shard.parent.mkdir(parents=True)
        shard.write_text('{"text":"sample"}\n')
    (clean / "clean_stats.json").write_text(json.dumps({
        "shards": ["eval/shard.jsonl", "train/missing.jsonl", "train/shard.jsonl"],
    }))
    sync = make(root, {**cfg, "sync": {"include_data": True}}, clean)
    assert not sync.push_tree(root) and "shards do not match" in sync.last_error
    assert not hub.commits


@pytest.mark.parametrize("path", ["../escaped", "windows\\escape", "C:/escape", ".git/config", "NUL.txt"])
def test_restore_rejects_unsafe_manifest_paths(experiment, tmp_path, path):
    hub, _, root, make, _ = experiment
    assert make().push_tree(root)
    manifest, = hub.manifests()
    head = hub.revisions[hub.head]
    name = next(p for p in head if p.endswith(manifest["id"] + ".json"))
    del head[name]
    manifest["snapshot"]["files"][0]["path"] = path
    manifest["id"] = identity(manifest["snapshot"])
    head[f"{name.rsplit('/', 1)[0]}/{manifest['id']}.json"] = json.dumps(manifest).encode()
    target = tmp_path / "restore" / root.name
    sync = make(target)
    assert sync.restore() == 0 and "none passed verification" in sync.last_error
    assert not target.exists() and not (target.parent / "escaped").exists()


def test_restore_rejects_redirected_parent(experiment, tmp_path):
    _, _, root, make, _ = experiment
    assert make().push_tree(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "redirect"
    if os.name == "nt":
        subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(linked), str(outside)],
                       check=True, capture_output=True)
    else:
        linked.symlink_to(outside, target_is_directory=True)
    sync = make(root, clean=linked / "corpus")
    assert sync.restore("data") == 0 and "symlink or reparse point" in sync.last_error
    assert not list(outside.iterdir())


def test_download_outside_staging_is_rejected(experiment, tmp_path, monkeypatch):
    _, _, root, make, _ = experiment
    assert make().push_tree(root)
    target = tmp_path / "elsewhere" / root.name
    sync = make(target)
    original = sync.store.fetch_at
    external = root / "progress.ckpt"

    def download(remote, revision, directory):
        if "/blobs/" in remote:
            return external
        return original(remote, revision, directory)

    monkeypatch.setattr(sync.store, "fetch_at", download)
    assert sync.restore() == 0 and not target.exists()
    assert external.read_bytes() == b"checkpoint A"


def test_snapshot_namespace_cannot_be_overwritten_or_deleted(experiment):
    hub, _, root, make, _ = experiment
    sync = make()
    assert sync.push_tree(root)
    name = next(p.removeprefix("mltok/") for p in hub.revisions[hub.head] if "/blobs/" in p)
    commits = list(hub.commits)
    with pytest.raises(StoreAccessError, match="cannot be overwritten"):
        sync.store.push_at([(root / "metrics.json", name)], "test", hub.head)
    with pytest.raises(StoreAccessError, match="verified publication"):
        sync.store.push([(root / "metrics.json", name)])
    with pytest.raises(StoreAccessError, match="deletion is disabled"):
        sync.store.delete([name])
    assert hub.commits == commits


def test_unfinished_restore_and_corpus_block_run_even_with_store_disabled(experiment, tmp_path):
    from mltok.runner import Run

    _, cfg, root, make, _ = experiment
    clean = tmp_path / "clean"
    config = {**cfg, "store": "none", "name": root.name,
              "paths": {"runs": str(root.parent), "clean": str(clean)}}
    journal = make().snapshot.pending
    journal.parent.mkdir()
    journal.write_text("{}")
    with pytest.raises(RuntimeError, match="interrupted snapshot restore"):
        Run(config)
    assert not (root / "variants").exists()
    journal.unlink()
    clean.mkdir()
    (clean / ".mltok-cleaning").touch()
    with pytest.raises(RuntimeError, match="unfinished write"):
        Run(config)
    assert not make(root, cfg, clean).push_tree(root)
    (clean / ".mltok-cleaning").unlink()
    receipt = make().snapshot.receipt
    receipt.write_text(json.dumps({"root": str(root), "dataset": "0" * 64}))
    from mltok.snapshots import SnapshotError

    with pytest.raises(SnapshotError, match="does not match the restored run"):
        Run(config)
    assert not (root / "variants").exists()


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_hub_privacy_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match="privacy must be explicitly"):
        HFHubStore("fixture/private", private=value)
