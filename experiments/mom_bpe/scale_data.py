"""Prepare verifiable, train/development-only data plans for matched 48K fits."""

import argparse
import hashlib
import importlib.metadata
import json
import sqlite3
import unicodedata
from collections import Counter
from contextlib import closing
from itertools import islice
from pathlib import Path

from experiments.mom_bpe import corpus_quality as quality
from experiments.mom_bpe.corpus_build import capture_rows, provisional_filter, write_rows_once
from experiments.mom_bpe.corpus_materialize import reviewed_proposal
from experiments.mom_bpe.corpus_overlap import DuplicateIndex, signatures
from experiments.mom_bpe.diagnose import AUDIT, format_json, immutable, sha
from experiments.mom_bpe.scale_wikipedia import SOURCE
from experiments.mom_bpe.verify_frozen import checked_member
from mltok.snapshots import _lock, file_identity

SCHEMA = "mom-48k-data-plan-v1"
BUDGETS = {"A": 100_000_000, "B": 1_500_000_000, "C": 8_000_000_000}
VALIDATION_BYTES = 1_000_000
SPLITS = ("train", "validation")
CAPACITY = "WIKIMEDIA_CAPACITY_DIAGNOSTIC"
REVIEWED = "REVIEWED_DEVELOPMENT"
LID_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"


def code_digest(path):
    return sha(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def scope(normalizer_fingerprint):
    return {"protocol_sha256": file_identity(AUDIT / "evaluation-protocol-v1.json")[1],
            "normalizer_fingerprint": normalizer_fingerprint, "unicode": unicodedata.unidata_version,
            "regex": importlib.metadata.version("regex")}


def document_bytes(row, normalize):
    raw = row["text"].encode()
    normalized = normalize(row["text"])
    canonical = normalized.encode()
    if (len(raw) != row["raw_bytes"] or sha(raw) != row["sha256"]
            or len(canonical) != row["canonical_bytes"] or sha(canonical) != row["canonical_sha256"]):
        raise ValueError("Data plan document hash or UTF-8 byte count differs")
    return raw, canonical


def split_identity(rows, normalize, seen=None):
    digest = hashlib.sha256()
    counts, domains, sources = Counter(), Counter(), Counter()
    for row in rows:
        raw, canonical = document_bytes(row, normalize)
        if seen is not None:
            try:
                seen.execute("INSERT INTO records VALUES(?,?,?,?)",
                             (row["id"], row["group"], row["sha256"], row["canonical_sha256"]))
            except sqlite3.IntegrityError:
                raise ValueError("Duplicate document, SHA-256 or connected group in data plan") from None
        # Length framing preserves boundaries without counting separators as source bytes.
        digest.update(len(canonical).to_bytes(8, "big"))
        digest.update(canonical)
        counts.update(rows=1, raw_bytes=len(raw), canonical_bytes=len(canonical))
        domains[row["domain"]] += len(canonical)
        sources[row["src"]] += len(canonical)
    return {**counts, "canonical_stream_sha256": digest.hexdigest(),
            "canonical_bytes_by_domain": dict(domains), "canonical_bytes_by_source": dict(sources)}


def publish(out, recipe, rows, normalize):
    stats = {}
    for split in SPLITS:
        path = out / f"{split}.jsonl"
        saved = write_rows_once(path, rows(split))
        stats[split] = {**saved, **split_identity(capture_rows(path), normalize)}
        if not stats[split].get("canonical_bytes"):
            raise ValueError("Data plan requires nonempty train and development validation splits")
    target = recipe["canonical_target"]
    actual = stats["train"]["canonical_bytes"]
    if actual > target:
        raise ValueError("Training data exceeds its declared canonical byte target")
    manifest = {"schema": SCHEMA, **recipe, "verified_freeze": False,
                "canonical_shortfall": target - actual, "splits": stats}
    immutable(out / "manifest.json", format_json(manifest))
    return manifest


def verified_plan(path, normalize, normalizer_fingerprint):
    path = path.resolve()
    data = path.read_bytes()
    value = json.loads(data)
    kind = value.get("qualification")
    if (value.get("schema") != SCHEMA or kind not in (CAPACITY, REVIEWED)
            or value.get("verified_freeze") is not False or set(value["splits"]) != set(SPLITS)
            or value.get("scope") != scope(normalizer_fingerprint)
            or value["scale"] not in BUDGETS or (kind == CAPACITY and value["scale"] != "A")):
        raise ValueError("Expected a current train/development-only data plan")
    expected_target = BUDGETS[value["scale"]]
    target = value["canonical_target"]
    if type(target) is not int or not 0 < target <= expected_target or (kind == REVIEWED and target != expected_target):
        raise ValueError("Invalid data plan budget")
    files = {}
    with closing(sqlite3.connect("")) as seen:
        seen.execute("CREATE TABLE records(id TEXT PRIMARY KEY NOT NULL, family TEXT UNIQUE NOT NULL, "
                     "raw TEXT UNIQUE NOT NULL, canonical TEXT UNIQUE NOT NULL)")
        for split in SPLITS:
            entry = value["splits"][split]
            member = checked_member(path.parent, {**entry, "path": entry["file"]})
            if split == "validation" and entry["canonical_bytes"] > VALIDATION_BYTES:
                raise ValueError("Development validation exceeds the bounded evaluator budget")
            observed = split_identity(capture_rows(member), normalize, seen)
            if any(entry.get(key) != val for key, val in observed.items()) or not observed.get("canonical_bytes"):
                raise ValueError("Data plan split accounting differs")
            files[split] = [member]
    actual = value["splits"]["train"]["canonical_bytes"]
    if not 0 < actual <= target or value["canonical_shortfall"] != target - actual:
        raise ValueError("Data plan training exposure differs")
    return files, sha(data)


def within_bytes(rows, limit):
    spent = 0
    for row in rows:
        size = row["canonical_bytes"]
        if size and spent + size <= limit:
            spent += size
            yield row


def screened_rows(payload, normalize, language):
    for item in capture_rows(payload):
        raw = item["text"]
        if item["src"] != SOURCE["source"]:
            raise ValueError("Unexpected source in the Wikimedia capture")
        attrs = item["attributes"]
        identifier = "mlwiki:" + attrs["id"]
        canonical = normalize(raw)
        metrics = quality.profile(raw, canonical, language)
        reasons = provisional_filter(metrics)
        if metrics["control_chars"]:
            reasons.append("control_chars")
        if metrics["repeated_line_hits"] > 5 and metrics["repeated_line_hits"] * 4 > len(raw.splitlines()):
            reasons.append("repeated_line_boilerplate")
        row = {"id": identifier, "group": identifier, "src": item["src"], "domain": "encyclopedic",
               "source_revision": SOURCE["revision"], "source_coordinate": item["row_index"],
               "source_url": attrs["url"], "history_url": attrs["history"], "sha256": sha(raw.encode()),
               "canonical_sha256": sha(canonical.encode()), "raw_bytes": metrics["raw_bytes"],
               "canonical_bytes": metrics["canonical_bytes"]}
        yield row, raw, reasons


def duplicate_order(receipts):
    frequency = Counter()
    for payload, _ in receipts[:4]:
        for row in islice(capture_rows(payload), 128):
            frequency.update(signatures(quality.duplicate_view(row["text"])))
    return frequency


def capacity(capture, output, lid_path, normalize, fingerprint, target, group_limit=None):
    if not 0 < target <= BUDGETS["A"]:
        raise ValueError("Capacity training is limited to 100,000,000 canonical bytes")
    capture = capture.resolve()
    complete_path = capture / "capture-complete.json"
    complete = json.loads(complete_path.read_bytes())
    if (complete["schema"] != "mom-wikipedia-capacity-capture-v1" or not complete["groups"]
            or complete["recipe_sha256"] != file_identity(capture / "recipe.json")[1]):
        raise ValueError("A completed pinned capacity capture is required")
    receipts = []
    if group_limit is not None and not 1 <= group_limit <= len(complete["groups"]):
        raise ValueError("Group limit must select a nonempty prefix of verified captures")
    for name in complete["groups"][:group_limit]:
        if not name.startswith("wiki-") or Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Invalid capture receipt path")
        receipt = json.loads((capture / "captures" / name).read_bytes())
        if any(receipt.get(k) != val for k, val in SOURCE.items()) or receipt["file"] != Path(name).stem + ".jsonl":
            raise ValueError("Capture receipt uses different source pins")
        payload = checked_member(capture / "captures", {"path": receipt["file"], "size": receipt["bytes"],
                                                       "sha256": receipt["sha256"]})
        receipts.append((payload, receipt))
    if len({r["file"] for _, r in receipts}) != len(receipts):
        raise ValueError("Capture receipt repeated a row group")
    language = quality.FastTextLanguage(lid_path, LID_SHA256)
    order = duplicate_order(receipts)
    recipe = {"qualification": CAPACITY, "scale": "A", "canonical_target": target, "scope": scope(fingerprint),
              "capture_sha256": file_identity(complete_path)[1], "lid_sha256": LID_SHA256,
              "capture_payloads": {r["file"]: r["sha256"] for _, r in receipts},
              "duplicate_feature_order_sha256": sha(format_json(order)),
              "lid_distribution": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                                   if d.metadata["Name"].lower() in ("fasttext", "fasttext-wheel")},
              "source": SOURCE, "code_sha256": {p.name: code_digest(p) for p in
                (Path(__file__), Path(__file__).with_name("corpus_overlap.py"),
                 Path(__file__).with_name("corpus_quality.py"), Path(__file__).with_name("corpus_build.py"))},
              "partition": "connected document groups; stable SHA-256 10% validation pool; ordered whole rows"}
    with _lock(output / ".prepare.lock"):
        immutable(output / "recipe.json", format_json(recipe))
        index = DuplicateIndex(output / "dedup.sqlite3", feature_order=lambda term: (order[term], term))
        try:
            db = index.db
            for payload, receipt in receipts:
                prior = db.execute("SELECT value FROM settings WHERE key=?", (payload.name,)).fetchone()
                if prior:
                    stored = json.loads(prior[0])
                    if stored["sha256"] != receipt["sha256"]:
                        raise ValueError("Wikimedia shard differs from completed index generation")
                    continue
                with db:
                    stats, rejected = Counter(), Counter()
                    for row, raw, reasons in screened_rows(payload, normalize, language):
                        stats.update(rows=1, raw_bytes=row["raw_bytes"])
                        if reasons:
                            rejected.update(reasons)
                        else:
                            index.insert(row, raw)
                            stats.update(screened_rows=1, screened_canonical_bytes=row["canonical_bytes"])
                    if stats["rows"] != receipt["rows"] or file_identity(payload) != (receipt["bytes"], receipt["sha256"]):
                        raise ValueError("Wikimedia payload changed while indexing")
                    db.execute("INSERT INTO settings VALUES(?,?)", (payload.name, json.dumps(
                        {"sha256": receipt["sha256"], "stats": dict(stats), "rejected": dict(rejected)})))
            stats, rejected = Counter(), Counter()
            for value, in db.execute("SELECT value FROM settings ORDER BY key"):
                item = json.loads(value)
                stats.update(item["stats"])
                rejected.update(item["rejected"])
            with db:
                groups = db.execute("SELECT count(*) FROM documents WHERE parent IS NULL").fetchone()[0]
                report = {"capture": dict(stats), "rejections_by_reason": dict(rejected), "unique_groups": groups,
                          "duplicate_links": index.counts(), "scope": "single-source, local engineering only"}
                immutable(output / "quality.json", format_json(report))

            def rows(split):
                for identity, raw, record in db.execute(
                        "SELECT identity,text,body FROM documents WHERE parent IS NULL ORDER BY identity"):
                    is_validation = int(sha(f"{SCHEMA}:{identity}".encode())[:12], 16) % 10 == 0
                    if is_validation == (split == "validation"):
                        yield {**json.loads(record), "text": raw}

            return publish(output, {**recipe, "quality_sha256": file_identity(output / "quality.json")[1]},
                           lambda split: within_bytes(rows(split), target if split == "train" else VALIDATION_BYTES),
                           normalize)
        finally:
            index.db.close()


def reviewed(study, review, output, normalize, fingerprint, scale, validation_per_domain):
    target = BUDGETS[scale]
    with _lock(output / ".prepare.lock"):
        # The proposal independently verifies review evidence and reserves its holdout groups first.
        # Export only training and development validation to the experiment-facing directory.
        proposal = output / "private-proposal"
        result = reviewed_proposal(study, review, proposal, raw_budget=target, validation_per_domain=validation_per_domain,
                                   mode="development" if scale == "A" else "dgx")
        recipe = {"qualification": REVIEWED, "scale": scale, "canonical_target": target,
                  "scope": scope(fingerprint), "proposal_sha256": file_identity(proposal / "manifest.json")[1],
                  "code_sha256": code_digest(__file__),
                  "raw_source_budget": target, "raw_shortfall": result["unspent_training_raw_bytes"],
                  "development_domain_shortfalls": result["validation_shortfalls"]}
        files = {split: checked_member(proposal.resolve(), {**result["splits"][split],
                 "path": result["splits"][split]["file"]}) for split in SPLITS}
        return publish(output, recipe, lambda split: within_bytes(capture_rows(files[split]),
                       target if split == "train" else VALIDATION_BYTES), normalize)


if __name__ == "__main__":
    from experiments.mom_bpe.corpus_materialize import normalizer

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    local = modes.add_parser("capacity")
    local.add_argument("--capture", type=Path, required=True)
    local.add_argument("--lid", type=Path, required=True)
    local.add_argument("--target", type=int, default=BUDGETS["A"])
    local.add_argument("--group-limit", type=int)
    eligible = modes.add_parser("reviewed")
    eligible.add_argument("--study", type=Path, required=True)
    eligible.add_argument("--review", type=Path, required=True)
    eligible.add_argument("--scale", choices=BUDGETS, required=True)
    eligible.add_argument("--validation-per-domain", type=int, default=100)
    for command in (local, eligible):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    normalize, fingerprint = normalizer()
    if args.mode == "capacity":
        report = capacity(args.capture, args.out, args.lid, normalize, fingerprint, args.target, args.group_limit)
    else:
        report = reviewed(args.study, args.review, args.out, normalize, fingerprint, args.scale,
                          args.validation_per_domain)
    print(json.dumps({"target": report["canonical_target"], "shortfall": report["canonical_shortfall"],
                      "splits": report["splits"]}, sort_keys=True))
