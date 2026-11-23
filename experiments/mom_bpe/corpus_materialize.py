"""Materialize an authored diagnostic or a reviewed, three-way corpus proposal."""

import argparse
import inspect
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe.corpus_build import capture_rows, write_rows_once
from experiments.mom_bpe.corpus_overlap import DuplicateIndex
from experiments.mom_bpe.diagnose import AUDIT, HERE, TRAIN, format_json, immutable, sha
from experiments.mom_bpe.verify_frozen import checked_member
from mltok import registry
from mltok.lib.morph_alignment import independent_reviewers
from mltok.snapshots import _lock, file_identity

SPLITS = ("heldout_proposal", "validation", "train")


def normalizer():
    registry.load_plugins()
    spec = yaml.safe_load((HERE / "arms.yaml").read_bytes())["normalizer"]
    return registry.make("normalizer", spec).hf().normalize_str, registry.fingerprint("normalizer", spec)


def publish(out, recipe, rows, status, diagnostics):
    files, counts = {}, {}
    for split in SPLITS:
        tally = Counter()

        def observed():
            for row in rows(split):
                tally["raw_utf8_bytes"] += row["raw_bytes"]
                tally["canonical_utf8_bytes"] += row["canonical_bytes"]
                tally["rows"] += 1
                tally[f"domain:{row['domain']}"] += 1
                tally[f"source:{row['src']}"] += 1
                yield row

        files[split] = write_rows_once(out / f"{split}.jsonl", observed())
        counts[split] = dict(tally)
    result = {
        "status": status, "verified_freeze": False, "recipe_sha256": file_identity(out / "recipe.json")[1],
        "dataset_fingerprint": sha(format_json(files)), "splits": files, "measurements": counts, **diagnostics(counts),
    }
    immutable(out / "manifest.json", format_json(result))
    return result


def diagnostic(out, byte_budget):
    if byte_budget <= 0:
        raise ValueError("Expected a positive raw byte budget")
    canonical, fingerprint = normalizer()
    train = list(dict.fromkeys(TRAIN))
    fixtures = json.loads((AUDIT / "fixtures.json").read_bytes())["domains"]
    test = [values[0] for _, values in sorted(fixtures.items()) if values]
    if len(test) < 2:
        raise ValueError("The authored validation and held-out fixtures are missing")
    inputs = {"train": train, "validation": test[::2], "heldout_proposal": test[1::2]}
    recipe = {"mode": "diagnostic", "normalizer": fingerprint, "raw_byte_budget": byte_budget,
              "code_sha256": {name: sha((HERE / name).read_bytes().replace(b"\r\n", b"\n")) for name in
                              ("corpus_materialize.py", "corpus_overlap.py")},
              "fixture_sha256": sha(format_json(inputs)), "source": "existing project-authored engineering fixtures"}
    out.mkdir(parents=True, exist_ok=True)
    with _lock(out / ".materialize.lock"):
        immutable(out / "recipe.json", format_json(recipe))
        db = DuplicateIndex(":memory:")
        assignment = {}
        try:
            with db.db:
                for split in SPLITS:
                    spent = 0
                    for text in inputs[split]:
                        raw, normalized = text.encode(), canonical(text).encode()
                        if split == "train" and spent + len(raw) > byte_budget:
                            continue
                        digest = sha(raw)
                        record = {"id": f"{split}:{len(assignment)}", "group": digest, "sha256": digest,
                                  "canonical_sha256": sha(normalized), "raw_bytes": len(raw),
                                  "canonical_bytes": len(normalized), "src": "authored_fixture", "domain": "engineering"}
                        db.insert(record, text)
                        assignment[record["id"]] = split
                        spent += len(raw)
                rows = {s: [] for s in SPLITS}
                groups = set()
                for group, text, row in db.members():
                    if group not in groups:
                        rows[assignment[row["id"]]].append({**row, "group": group, "text": text})
                        groups.add(group)
        finally:
            db.db.close()
        if not all(rows.values()):
            raise ValueError("Diagnostic fixtures do not cover all three disjoint splits within the budget")
        return publish(out, recipe, rows.__getitem__, "ENGINEERING_DIAGNOSTIC_ONLY", lambda _: {})


def prepare_choices(index, db, protocol_id, rights, ledger, domains):
    with db:
        for decision in capture_rows(ledger):
            row = index.db.execute("SELECT n,body FROM documents WHERE identity=?", (decision["id"],)).fetchone()
            if row is None:
                raise ValueError("Review ledger references an unindexed document")
            n, body = row
            document = json.loads(body)
            source = document["src"]
            if (source not in rights or rights[source]["revision"] != document["source_revision"]
                    or decision["sha256"] != document["sha256"] or decision["domain"] not in domains
                    or not isinstance(decision.get("permission_scope"), str) or not decision["permission_scope"].strip()):
                raise ValueError("Review decision lacks matching document, source revision, or domain evidence")
            eligible = decision["eligible_splits"]
            if (not isinstance(eligible, list) or not eligible or set(eligible).difference(SPLITS)
                    or not document["raw_bytes"] or document["quarantine_reasons"]):
                raise ValueError("Invalid eligibility or document failed the automatic screen")
            if set(eligible).difference({"train"}) and (
                not document["complete_document"] or not document["source_url"]
            ):
                raise ValueError("Validation and held-out proposals require complete, traceable documents")
            # Separate connected roots enforce one document per group across all splits.
            group = index.root(n)
            group_identity, = index.db.execute("SELECT identity FROM documents WHERE n=?", (group,)).fetchone()
            rank = sha(f"{protocol_id}\n{group_identity}".encode())
            for split in sorted(set(eligible)):
                priority = rights[source]["priority"] if split == "train" else 0
                db.execute("INSERT INTO options VALUES(?,?,?,?,?,?,?)",
                           (split, decision["domain"], n, group, priority, rank, document["raw_bytes"]))
        db.execute("INSERT INTO progress VALUES('reviewed')")


def balanced_byte_caps(domains, raw_budget):
    cap, remainder = divmod(raw_budget, len(domains))
    return {domain: cap + (i < remainder) for i, domain in enumerate(sorted(domains))}


def assign(db, domains, byte_caps, validation_per_domain):
    with db:
        for split in SPLITS:
            for domain, goal in sorted(domains.items()):
                docs = goal if split == "heldout_proposal" else validation_per_domain if split == "validation" else None
                budget = byte_caps[domain] if split == "train" else None
                for n, group, size in db.execute(
                    "SELECT n,cluster,raw_bytes FROM options WHERE split=? AND domain=? ORDER BY priority,rank,n",
                    (split, domain),
                ):
                    if docs == 0 or budget == 0:
                        break
                    if budget is not None and size > budget:
                        continue
                    inserted = db.execute("INSERT OR IGNORE INTO chosen VALUES(?,?,?,?)",
                                          (group, n, split, domain)).rowcount
                    if inserted:
                        if docs is not None:
                            docs -= 1
                        if budget is not None:
                            budget -= size
        db.execute("INSERT INTO progress VALUES('assigned')")


def reviewed_proposal(study, review_path, out, *, raw_budget, validation_per_domain, mode,
                      domain_balancing=balanced_byte_caps):
    if mode not in ("development", "dgx") or raw_budget <= 0 or validation_per_domain <= 0:
        raise ValueError("Expected a proposal mode, positive raw byte budget, and positive validation quota")
    review = json.loads(review_path.read_bytes())
    source_manifest = json.loads((study / "recipe.json").read_bytes())
    complete = json.loads((study / "index-complete.json").read_bytes())
    study_identity = file_identity(study / "recipe.json")[1]
    if (complete.get("recipe_sha256") != study_identity or review.get("study_recipe_sha256") != study_identity
            or not source_manifest.get("index_only") or review.get("schema") != "mom-corpus-permissions-v1"
            or not review.get("version") or not independent_reviewers(review.get("reviewers"))):
        raise ValueError("Expected independent reviewers and a completed, matching corpus study index")
    rights = review["sources"]
    for source, row in rights.items():
        if (not source or not row.get("license") or not row.get("attribution") or not row.get("revision")
                or row.get("local_research_use") is not True or type(row.get("priority")) is not int):
            raise ValueError("Sources need recorded local research use, license, attribution, revision and priority")
        checked_member(review_path.resolve().parent, row["permission_evidence"])
    ledger = checked_member(review_path.resolve().parent, review["document_ledger"])
    protocol_bytes = (AUDIT / "evaluation-protocol-v1.json").read_bytes()
    if source_manifest["baseline_protocol_sha256"] != sha(protocol_bytes):
        raise ValueError("Indexed evaluation protocol differs from this proposal")
    protocol = json.loads(protocol_bytes)
    domains = protocol["panels"]["natural_text"]["target_per_domain"]
    byte_caps = domain_balancing(domains, raw_budget)
    if (set(byte_caps) != set(domains) or any(type(v) is not int or v < 0 for v in byte_caps.values())
            or sum(byte_caps.values()) > raw_budget):
        raise ValueError("Domain balancing must stay within the reviewed domains and training byte budget")
    canonical, fingerprint = normalizer()
    if fingerprint != source_manifest["normalizer_fingerprint"]:
        raise ValueError("Indexed normalizer fingerprint differs from this runtime")
    recipe = {"mode": mode, "study_recipe_sha256": study_identity, "protocol_sha256": sha(protocol_bytes),
        "review_sha256": file_identity(review_path)[1],
        "raw_byte_budget": raw_budget, "validation_per_domain": validation_per_domain, "heldout_quotas": domains,
        "train_raw_byte_caps": byte_caps,
        "domain_balancing": {
            "callable": f"{domain_balancing.__module__}.{domain_balancing.__qualname__}",
            "code_sha256": sha(Path(inspect.getsourcefile(domain_balancing)).read_bytes().replace(b"\r\n", b"\n")),
        },
        "materializer_code_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))}
    out.mkdir(parents=True, exist_ok=True)
    with _lock(study / ".study.lock"), _lock(out / ".materialize.lock"):
        immutable(out / "recipe.json", format_json(recipe))
        if not (study / "study.sqlite3").is_file():
            raise ValueError("The reviewed corpus index is missing")
        index = DuplicateIndex(study / "study.sqlite3")
        db = sqlite3.connect(out / "selection.sqlite3")
        try:
            expected = {(json.dumps([r["source"], r.get("shard", "")]), r["sha256"]) for r in source_manifest["captures"]}
            if set(index.db.execute("SELECT key,value FROM settings")) != expected:
                raise ValueError("The study database does not match the completed capture ledger")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA temp_store=FILE")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS progress(stage TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS options(split TEXT,domain TEXT,n INTEGER,cluster INTEGER,
                    priority INTEGER,rank TEXT,raw_bytes INTEGER,PRIMARY KEY(split,n));
                CREATE INDEX IF NOT EXISTS choice_order ON options(split,domain,priority,rank,n);
                CREATE TABLE IF NOT EXISTS chosen(cluster INTEGER PRIMARY KEY,n INTEGER,split TEXT,domain TEXT);
            """)
            stages = {stage for stage, in db.execute("SELECT stage FROM progress")}
            if "reviewed" not in stages:
                with index.db:
                    prepare_choices(index, db, protocol["identity"], rights, ledger, domains)
            if "assigned" not in stages:
                assign(db, domains, byte_caps, validation_per_domain)

            def selected(split):
                for n, group, domain in db.execute("SELECT n,cluster,domain FROM chosen WHERE split=? ORDER BY n", (split,)):
                    text, body = index.db.execute("SELECT text,body FROM documents WHERE n=?", (n,)).fetchone()
                    name, = index.db.execute("SELECT identity FROM documents WHERE n=?", (group,)).fetchone()
                    row = json.loads(body)
                    raw, normalized = text.encode(), canonical(text).encode()
                    if (sha(raw), len(raw), sha(normalized), len(normalized)) != (
                        row["sha256"], row["raw_bytes"], row["canonical_sha256"], row["canonical_bytes"]
                    ):
                        raise ValueError("Indexed source or canonical document hash mismatch")
                    yield {**row, "group": name, "source_group": row["group"], "domain": domain, "text": text,
                           "license_review": "review ledger matched exact document hash",
                           "source_license": rights[row["src"]]["license"]}

            def shortfalls(counts):
                return {"heldout_shortfalls": {d: target - counts["heldout_proposal"].get(f"domain:{d}", 0)
                                               for d, target in domains.items()},
                        "validation_shortfalls": {d: validation_per_domain - counts["validation"].get(f"domain:{d}", 0)
                                                  for d in domains},
                        "unspent_training_raw_bytes": raw_budget - counts["train"].get("raw_utf8_bytes", 0)}

            return publish(out, recipe, selected, "REVIEWED_PROPOSAL_ONLY", shortfalls)
        finally:
            db.close()
            index.db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("diagnostic", "development", "dgx"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-byte-budget", type=int, required=True)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--validation-per-domain", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "diagnostic":
        result = diagnostic(args.out, args.raw_byte_budget)
    else:
        if not args.study or not args.review:
            parser.error("Study and independent review evidence are required for development or DGX")
        result = reviewed_proposal(args.study, args.review, args.out, mode=args.mode, raw_budget=args.raw_byte_budget,
                                   validation_per_domain=args.validation_per_domain)
    print(json.dumps(result, sort_keys=True))
