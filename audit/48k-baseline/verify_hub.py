"""Verify the project storage stack on the authorized private test repository."""
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).with_name("hub-verification.json")
REPO = "Akzhai/mltok-storage-test"
sys.path.insert(0, str(ROOT))


def main():
    if sys.platform == "win32":
        import truststore

        truststore.inject_into_ssl()
    from huggingface_hub import HfApi, get_token

    from mltok import registry
    from mltok.contracts import StoreAccessError
    from mltok.lib.atomic import atomic_writer
    from mltok.snapshots import file_identity
    from mltok.sync import Sync

    trace = {"repository": REPO, "checks": {}, "status": "in_progress"}

    def save():
        with atomic_writer(REPORT) as out:
            out.write(json.dumps(trace, indent=2, ensure_ascii=False).encode())

    def check(name, result):
        trace["checks"][name] = bool(result)
        save()
        if not result:
            raise RuntimeError("Verification check failed")

    def status(exc):
        return getattr(getattr(exc, "response", None), "status_code", None)

    try:
        token = get_token()
        check("credential_available", token is not None)
        api = HfApi(token=token)
        check("authenticated_as_Akzhai", api.whoami().get("name", "").casefold() == "akzhai")
        try:
            info = api.repo_info(REPO, repo_type="model", timeout=20)
        except Exception as exc:
            if status(exc) != 404:
                raise
            api.create_repo(REPO, private=True, repo_type="model", exist_ok=False)
            info = api.repo_info(REPO, repo_type="model", timeout=20)
            trace["repository_created"] = True
        else:
            trace["repository_created"] = False
        trace["visibility"] = "private" if info.private is True else "unverified_or_public"
        trace["initial_file_count"] = len(info.siblings or [])
        check("private_repository", info.private is True)

        registry.load_plugins()
        tmp = Path(tempfile.mkdtemp(prefix="mltok-live-"))
        name = "synthetic-" + uuid.uuid4().hex[:12]
        clean = tmp / "clean"
        original = tmp / "generation1" / name
        original.mkdir(parents=True)
        cfg = {"name": name, "components": {"vocab": 16},
               "store": {"type": "hf_hub", "repo": REPO, "private": True, "retries": 1},
               "sync": {"include_data": True}}
        (original / "experiment.json").write_text(json.dumps({"name": name, "synthetic": True, "vocab": 16}))
        (original / "dummy.ckpt").write_bytes(b"synthetic checkpoint A\n")
        (original / "metrics.json").write_text('{"synthetic":true,"score":1}')
        for split in ("train", "eval"):
            shard = clean / split / "fixture.jsonl"
            shard.parent.mkdir(parents=True)
            shard.write_text(json.dumps({"text": f"synthetic {split} sample"}) + "\n", encoding="utf-8")
        (clean / "clean_stats.json").write_text(json.dumps({"shards": ["eval/fixture.jsonl", "train/fixture.jsonl"]}))
        trace.update({"experiment": name, "local_test_directory": str(tmp)})
        save()

        def sync(path, corpus, options=None):
            instance = Sync(options or cfg, path, corpus, log=lambda _message: None)
            # Reuse the already verified identity in memory for all writes.
            instance.store.api_factory = lambda: api
            return instance

        def manifests(instance):
            return instance.snapshot._manifests(instance.store.revision())

        def files(tree):
            return {p.relative_to(tree).as_posix(): file_identity(p) for p in tree.rglob("*") if p.is_file()}

        def record(manifest):
            return {"id": manifest["id"], "pinned_revision": manifest["revision"],
                    "artifact_count": len(manifest["snapshot"]["files"]),
                    "dataset_fingerprint": manifest["snapshot"]["dataset"],
                    "configuration_fingerprint": manifest["snapshot"]["config"],
                    "artifacts": manifest["snapshot"]["files"]}

        first = sync(original, clean)
        check("upload_and_remote_sha256_readback", first.push_tree(original))
        old, = manifests(first)
        trace["snapshots"] = [record(old)]
        save()
        dest = tmp / "restored" / name
        restored_clean = tmp / "restored-data"
        restored = sync(dest, restored_clean)
        count = trace["snapshots"][0]["artifact_count"]
        check("pinned_independent_restore", restored.restore("all", old["id"]) == count)
        before = files(dest), files(restored_clean)
        check("restored_file_sizes_and_sha256", before == (files(original), files(clean)))
        check("idempotent_restore", restored.restore("all", old["id"]) == 0 and restored.last_error is None)

        newer = tmp / "generation2" / name
        shutil.copytree(original, newer)
        (newer / "dummy.ckpt").write_bytes(b"synthetic checkpoint B\n")
        (newer / "metrics.json").write_text('{"synthetic":true,"score":2}')
        publication = sync(newer, clean)
        real_push = publication.store.push_at

        def stop_before_manifest(batch, message, parent):
            if any("/experiments/" in path for _, path in batch):
                raise StoreAccessError("Synthetic interruption before manifest publication")
            return real_push(batch, message, parent)

        publication.store.push_at = stop_before_manifest
        check("interrupted_upload_keeps_only_previous_manifest",
              not publication.push_tree(newer) and [m["id"] for m in manifests(first)] == [old["id"]])
        publication.store.push_at = real_push
        check("retry_publishes_verified_snapshot", publication.push_tree(newer))
        new, prev = manifests(publication)
        check("both_completed_snapshots_retained", prev["id"] == old["id"] and new["id"] != old["id"])
        trace["snapshots"].append(record(new))
        save()

        real_fetch = restored.store.fetch_at

        def interrupted_download(remote, revision, directory):
            if "/blobs/" in remote:
                raise StoreAccessError("Synthetic download interruption")
            return real_fetch(remote, revision, directory)

        restored.store.fetch_at = interrupted_download
        check("interrupted_download_preserves_local_data",
              restored.restore("all", old["id"]) == 0 and bool(restored.last_error)
              and (files(dest), files(restored_clean)) == before)
        restored.store.fetch_at = real_fetch
        check("retry_after_interruption", restored.restore("all", old["id"]) == 0 and restored.last_error is None)

        mismatch_root = tmp / "incompatible" / name
        mismatch = sync(mismatch_root, tmp / "incompatible-data", {**cfg, "components": {"vocab": 17}})
        check("incompatible_config_rejected", mismatch.restore("all", old["id"]) == 0
              and bool(mismatch.last_error) and not mismatch_root.exists())
        check("conflicting_restore_preserves_existing_valid_data",
              restored.restore("all", new["id"]) == 0 and bool(restored.last_error)
              and (files(dest), files(restored_clean)) == before)

        fall_root = tmp / "fallback" / name
        fall_data = tmp / "fallback-data"
        fallback = sync(fall_root, fall_data)
        real_fallback_fetch = fallback.store.fetch_at

        def latest_blob_unavailable(remote, revision, directory):
            if "/blobs/" in remote and revision == new["revision"]:
                raise FileNotFoundError("Synthetic unavailability at the latest pinned revision")
            return real_fallback_fetch(remote, revision, directory)

        fallback.store.fetch_at = latest_blob_unavailable
        check("older_verified_snapshot_fallback", fallback.restore("all") == count
              and (files(fall_root), files(fall_data)) == before)
        trace["status"] = "passed"
    except Exception as exc:
        trace["status"] = "failed"
        trace["error_type"] = type(exc).__name__
        trace["http_status"] = status(exc)
    finally:
        save()
    print(json.dumps({"status": trace["status"], "checks": trace["checks"], "report": str(REPORT)}))
    return 0 if trace["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
