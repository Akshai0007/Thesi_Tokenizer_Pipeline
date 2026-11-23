"""Pin production artifacts and discover the historical evaluation data without downloading training files."""
import hashlib
import itertools
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
TOKENIZER = "Akzhai/malayalam-mlbpe-tokenizer"
ARCHIVE = "Akzhai/mltok-malayalam-runs"
EXPECTED = "c1321250cd78"
MIB = 1024 * 1024


def main():
    if sys.platform == "win32":
        import truststore

        truststore.inject_into_ssl()
    from huggingface_hub import HfApi, get_token

    from mltok.lib.atomic import atomic_writer
    from mltok.snapshots import file_identity

    def persist(name, data):
        with atomic_writer(HERE / name) as out:
            out.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))

    def historical_fingerprint(entries):
        # The historical runner keyed byte lengths and base names, so this is not a SHA-256 content claim.
        h = hashlib.sha1()
        for name, size in sorted(entries):
            h.update(f"{PurePosixPath(name).name}:{size};".encode())
        return h.hexdigest()[:12]

    token = get_token()
    if token is None:
        print("No configured Hugging Face credential")
        return 2
    api = HfApi(token=token)
    outcome = {"status": "in_progress", "tokenizer_repository": TOKENIZER, "archive_repository": ARCHIVE,
               "expected_historical_eval_fingerprint": EXPECTED, "expected_eval_documents": 22120}
    persist("retrieval.json", outcome)
    try:
        if api.whoami().get("name", "").casefold() != "akzhai":
            raise RuntimeError("Unexpected Hugging Face account")

        def fetch(repo, filename, revision, scratch):
            return Path(api.hf_hub_download(repo, filename, repo_type="model", revision=revision, local_dir=scratch))

        info = api.repo_info(TOKENIZER, repo_type="model", files_metadata=True, timeout=25)
        revision = info.sha
        allowed = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                   "chat_template.jinja", "tokenizer.model", "spm.model", "spm.vocab", "vocab.json", "merges.txt",
                   "README.md", "LICENSE", "config.json"}
        members = {s.rfilename: s.size for s in info.siblings or [] if s.rfilename in allowed}
        if ("tokenizer.json" not in members or not revision or any(type(v) is not int for v in members.values())
                or sum(members.values()) > 40 * MIB):
            raise RuntimeError("Tokenizer artifact set is missing or outside the byte budget")
        root = HERE / "original" / revision
        if root.exists():
            saved = json.loads((HERE / "baseline-artifacts.json").read_text(encoding="utf-8"))
            if saved["revision"] != revision or any(file_identity(root / row["path"]) != (row["size"], row["sha256"])
                                                    for row in saved["artifacts"]):
                raise RuntimeError("Existing immutable baseline does not match its identity")
        else:
            root.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=root.parent, prefix=".download-") as scratch:
                stage = Path(scratch) / "artifacts"
                stage.mkdir()
                metadata = []
                for name, expected_size in sorted(members.items()):
                    local = fetch(TOKENIZER, name, revision, Path(scratch) / "hub")
                    size, digest = file_identity(local)
                    if expected_size != size:
                        raise RuntimeError("Tokenizer artifact length mismatch")
                    shutil.copyfile(local, stage / name)
                    if file_identity(stage / name) != (size, digest):
                        raise RuntimeError("Tokenizer artifact copy failed integrity check")
                    metadata.append({"path": name, "size": size, "sha256": digest})
                saved = {"repository": TOKENIZER, "revision": revision, "directory": str(root.relative_to(HERE)),
                         "artifacts": metadata}
                persist("baseline-artifacts.json", saved)
                os.replace(stage, root)
        outcome["baseline"] = {"revision": revision, "directory": str(root.relative_to(HERE)),
                               "files": len(members), "bytes": sum(members.values())}
        persist("retrieval.json", outcome)

        try:
            archive = api.repo_info(ARCHIVE, repo_type="model", timeout=25)
            archive_rev = archive.sha
            if not archive_rev:
                raise RuntimeError("Archive has no readable revision")
            rows = list(itertools.islice(api.list_repo_tree(
                ARCHIVE, repo_type="model", revision=archive_rev, recursive=True), 30001))
        except Exception as exc:
            outcome["eval"] = {"status": "archive_inventory_failed", "error_type": type(exc).__name__,
                               "http_status": getattr(getattr(exc, "response", None), "status_code", None)}
            outcome["status"] = "baseline_verified_eval_unavailable"
            persist("retrieval.json", outcome)
            print(json.dumps(outcome))
            return 0
        inventory = {r.path: r.size for r in rows[:30000] if type(getattr(r, "size", None)) is int}
        groups = defaultdict(list)
        for path, size in inventory.items():
            key = PurePosixPath(path)
            if (key.name.endswith((".jsonl", ".jsonl.gz", ".txt"))
                    and ("eval" in key.parts[:-1] or key.name.startswith("eval-"))):
                groups[str(key.parent)].append((path, size))
        candidates = [{"prefix": name, "files": len(entries), "bytes": sum(s for _, s in entries),
                       "legacy_fingerprint": historical_fingerprint(entries)}
                      for name, entries in sorted(groups.items())]
        evidence = {"archive_revision": archive_rev, "inventory_truncated": len(rows) > 30000,
                    "file_count": len(inventory), "eval_groups": candidates, "metadata_matches": []}
        markers = sorted((p for p, size in inventory.items() if size <= 512 * 1024
                          and (PurePosixPath(p).name in ("freeze.json", "SELECTION.json", "clean_stats.json", "report.md")
                               or "/experiments/" in p and p.endswith(".json"))),
                         key=lambda p: (PurePosixPath(p).name not in ("freeze.json", "clean_stats.json"), p))
        expected_entries = [(p, s, None, p, archive_rev) for g in candidates if g["legacy_fingerprint"] == EXPECTED
                            for p, s in groups[g["prefix"]]] if sum(
                                g["legacy_fingerprint"] == EXPECTED for g in candidates) == 1 else []
        with tempfile.TemporaryDirectory(prefix="ml-archive-") as scratch:
            for marker in markers[:60]:
                try:
                    local = fetch(ARCHIVE, marker, archive_rev, scratch)
                    data = local.read_bytes()
                except Exception as exc:
                    if getattr(getattr(exc, "response", None), "status_code", None) in (401, 403):
                        raise
                    evidence.setdefault("metadata_read_errors", []).append({"path": marker, "type": type(exc).__name__})
                    continue
                if EXPECTED.encode() in data or b"22120" in data:
                    evidence["metadata_matches"].append({"path": marker, "sha256": hashlib.sha256(data).hexdigest(),
                                                         "contains_expected_fingerprint": EXPECTED.encode() in data,
                                                         "contains_reported_count": b"22120" in data})
                if "/experiments/" in marker and not expected_entries:
                    try:
                        manifest = json.loads(data)
                        snap = manifest["snapshot"]
                        canonical = json.dumps(snap, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                               allow_nan=False).encode("ascii")
                        if manifest["id"] != PurePosixPath(marker).stem or hashlib.sha256(canonical).hexdigest() != manifest["id"]:
                            continue
                        items = [f for f in snap["files"] if f["scope"] == "data"
                                 and f["path"].startswith("eval/")]
                        if historical_fingerprint([(f["path"], f["size"]) for f in items]) == EXPECTED:
                            prefix = marker.split("/experiments/", 1)[0]
                            expected_entries = [(f["path"], f["size"], f["sha256"],
                                                 f"{prefix.rsplit('/v1', 1)[0]}/v1/blobs/{f['sha256']}",
                                                 manifest["revision"]) for f in items]
                    except (ValueError, KeyError, TypeError):
                        continue
            evidence["metadata_files_checked"] = min(60, len(markers))
            evidence["metadata_scan_truncated"] = len(markers) > 60
            if expected_entries and sum(row[1] for row in expected_entries) <= 192 * MIB:
                eval_root = HERE / "historical-eval" / archive_rev
                names = [PurePosixPath(p).name for p, *_ in expected_entries]
                if len(set(names)) != len(names):
                    raise RuntimeError("Conflicting evaluation basenames")
                downloaded = []
                eval_root.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=eval_root.parent, prefix=".eval-") as staging:
                    for (path, expected_size, expected_digest, remote, pinned), name in zip(expected_entries, names):
                        source = eval_root / name if eval_root.exists() else fetch(ARCHIVE, remote, pinned, scratch)
                        size, digest = file_identity(source)
                        if size != expected_size or (expected_digest and digest != expected_digest):
                            raise RuntimeError("Historical evaluation artifact is incompatible")
                        if not eval_root.exists():
                            shutil.copyfile(source, Path(staging) / name)
                        downloaded.append({"archive_path": path, "file": name, "revision": pinned,
                                           "size": size, "sha256": digest})
                    if not eval_root.exists():
                        install = Path(staging) / "complete"
                        install.mkdir()
                        for name in names:
                            os.replace(Path(staging) / name, install / name)
                        os.replace(install, eval_root)
                evidence.update(status="legacy_fingerprint_matched", artifacts=downloaded,
                                directory=str(eval_root.relative_to(HERE)))
            else:
                evidence["status"] = "original_eval_not_located" if not expected_entries else "download_budget_exceeded"
        outcome["eval"] = evidence
        outcome["status"] = "retrieval_complete"
    except Exception as exc:
        outcome["status"] = "failed"
        outcome["error_type"] = type(exc).__name__
        outcome["http_status"] = getattr(getattr(exc, "response", None), "status_code", None)
    persist("retrieval.json", outcome)
    print(json.dumps({k: outcome[k] for k in ("status", "baseline", "error_type", "http_status") if k in outcome}))
    print(json.dumps({"eval": {k: outcome.get("eval", {}).get(k) for k in
                              ("status", "archive_revision", "file_count", "eval_groups", "metadata_matches")}}))
    return 1 if outcome["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
