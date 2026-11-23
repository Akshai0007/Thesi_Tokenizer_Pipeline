"""Back up only small audit documents to the already verified private repository."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
REPO = "Akzhai/mltok-storage-test"


def main():
    if sys.platform == "win32":
        import truststore

        truststore.inject_into_ssl()
    from huggingface_hub import HfApi, get_token

    from mltok import registry
    from mltok.lib.atomic import atomic_writer
    from mltok.snapshots import file_identity, identity
    from mltok.sync import Sync

    verified = json.loads((HERE / "hub-verification.json").read_bytes())
    if verified.get("status") != "passed" or verified.get("visibility") != "private":
        raise SystemExit("Live verification is required before audit publication")
    token = get_token()
    if not token:
        raise SystemExit("Configured Hugging Face credential unavailable")
    api = HfApi(token=token)
    result = {"repository": REPO, "status": "in_progress"}
    try:
        if api.whoami().get("name", "").casefold() != "akzhai":
            raise RuntimeError("Unexpected authenticated account")
        if api.repo_info(REPO, repo_type="model", timeout=20).private is not True:
            raise RuntimeError("Repository privacy could not be confirmed")
        sample = HERE / "measurement-v3"
        complete = json.loads((sample / "COMPLETE.json").read_bytes())
        for entry in complete["files"]:
            if file_identity(sample / entry["path"]) != (entry["size"], entry["sha256"]):
                raise RuntimeError("Audit completion hashes do not match")
        members = [HERE / name for name in (
            "REPORT.md", "baseline-artifacts.json", "retrieval.json", "fixtures.json",
        )] + [sample / name for name in (
            "configuration.json", "measurements.json", "vocabulary-analysis.json", "unicode-analysis.json",
            "boundary-analysis.json", "boundary-root-causes.json", "morphology-analysis.json",
            "domain-analysis.json", "hf-compatibility.json",
        )]
        hashes = {p.name: file_identity(p) for p in members}
        bytes_total = sum(size for size, _ in hashes.values())
        if len(members) != len(hashes) or bytes_total > 256 * 1024:
            raise RuntimeError("Backup allowlist has duplicate names or exceeds the audit budget")
        name = "audit-48k-" + identity(hashes)[:16]
        cfg = {"name": name, "components": {"vocab": 48000},
               "search": {"baseline_revision": complete["artifact_revision"],
                          "document_bundle": identity(hashes)},
               "store": {"type": "hf_hub", "repo": REPO, "private": True, "retries": 1}}
        registry.load_plugins()
        with tempfile.TemporaryDirectory(prefix="ml-audit-") as temp:
            root = Path(temp) / "publish" / name
            root.mkdir(parents=True)
            for p in members:
                shutil.copyfile(p, root / p.name)
            def sync(path):
                s = Sync(cfg, path, log=lambda _message: None)
                s.store.api_factory = lambda: api
                return s

            publish = sync(root)
            if not publish.push_tree(root):
                result["operation"] = "publish"
                raise RuntimeError("No verified snapshot was published")
            manifest, = publish.snapshot._manifests(publish.store.revision())
            restored = Path(temp) / "restore" / name
            restore = sync(restored)
            if restore.restore("run", manifest["id"]) != len(hashes):
                raise RuntimeError("Audit snapshot restoration failed")
            if {p.name: file_identity(p) for p in restored.iterdir()} != hashes:
                raise RuntimeError("Audit backup restore hash mismatch")
        result.update(status="passed", private=True, experiment=name, snapshot_id=manifest["id"],
                      pinned_revision=manifest["revision"], files=len(hashes), bytes=bytes_total,
                      all_independent_restore_hashes_match=True)
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__,
                      http_status=getattr(getattr(exc, "response", None), "status_code", None))
    with atomic_writer(HERE / "audit-backup.json") as out:
        out.write(json.dumps(result, indent=2).encode())
    print(json.dumps(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
