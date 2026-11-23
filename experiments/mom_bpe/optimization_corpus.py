"""Extend Wikipedia development with bounded, attributed Wikisource prose."""

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.mom_bpe.corpus_acquire import public_session
from experiments.mom_bpe.corpus_build import write_rows_once
from experiments.mom_bpe.corpus_overlap import DuplicateIndex
from experiments.mom_bpe.corpus_quality import FastTextLanguage, profile
from experiments.mom_bpe.development_corpus import LICENSE, NORMALIZER, TERMS, extract
from experiments.mom_bpe.diagnose import AUDIT, HERE, format_json, immutable, sha
from experiments.mom_bpe.verify_frozen import checked_member, load_rows
from mltok import registry
from mltok.snapshots import _lock, file_identity

SITE = "https://ml.wikisource.org"
IDENTITY = "wikimedia-prose-optimization-v1"
WORKS = {
    "indulekha": ("ഇന്ദുലേഖ", "O. Chandu Menon", "literature", "train"),
    "kundalatha": ("കുന്ദലത", "Appu Nedungadi", "literature", "train"),
    "marthandavarma": ("മാർത്താണ്ഡവർമ്മ", "C. V. Raman Pillai", "literature", "validation"),
    "kerala-panineeyam": ("കേരളപാണിനീയം", "A. R. Raja Raja Varma", "grammar", "train"),
}
CAP = 12 * 1024**2
PER_PAGE = 384 * 1024
PER_WORK = 8


def fetch(client, params, ledger, maximum):
    with client.get(SITE + "/w/api.php", params={"format": "json", "formatversion": 2, **params},
                    stream=True, timeout=25) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_content(8192):
            ledger[0] += len(chunk)
            if ledger[0] > CAP or len(body) + len(chunk) > maximum:
                raise ValueError("Wikisource response cap reached")
            body.extend(chunk)
    result = json.loads(body)
    if result.get("error"):
        raise ValueError("Wikisource API refused a selected revision")
    return result


def acquire(directory, pins=None):
    directory.mkdir(parents=True, exist_ok=True)
    capture = directory / "capture.json"
    expected = json.loads(pins.read_bytes())["wikisource_capture"] if pins else None
    ledger = [0]
    with _lock(directory / ".capture.lock"), public_session(
        "MalayalamTokenizerResearch/1.0 (public Wikimedia study; https://github.com/Akshai0007/Thesi_Tokenizer_Pipeline)"
    ) as client:
        if capture.is_file():
            prior = json.loads(capture.read_bytes())
            if expected is not None and expected != prior:
                raise ValueError("Existing capture differs from replay pins")
            return prior
        inventory = directory / "inventory.json"
        if inventory.is_file():
            plan = json.loads(inventory.read_bytes())
        elif expected is not None:
            plan = {"rights": expected["site_rightsinfo"], "pages": [
                {k: row[k] for k in ("work", "title", "pageid", "revid", "sha1", "timestamp")} for row in expected["documents"]
            ]}
            immutable(inventory, format_json(plan))
        else:
            rights = fetch(client, {"action": "query", "meta": "siteinfo", "siprop": "rightsinfo"}, ledger, 32768)
            rights = rights["query"]["rightsinfo"]
            if rights["url"].removesuffix("deed.ml").rstrip("/") != LICENSE.rstrip("/"):
                raise ValueError("Unexpected Malayalam Wikisource license")
            pages = []
            for work, (title, _, _, _) in WORKS.items():
                found = fetch(client, {"action": "query", "list": "allpages", "apnamespace": 0, "aplimit": 100,
                                       "apprefix": title + "/", "apfilterredir": "nonredirects"}, ledger, 98304)
                names = [item["title"] for item in found["query"]["allpages"] if item["title"].count("/") == 1]
                chosen = sorted(names, key=lambda name: (sha(f"{IDENTITY}\n{name}".encode()), name))[:PER_WORK]
                if not chosen:
                    continue
                res = fetch(client, {"action": "query", "prop": "revisions", "rvprop": "ids|sha1|timestamp",
                                     "titles": "|".join(chosen)}, ledger, 65536)
                for page in res["query"]["pages"]:
                    if page.get("revisions"):
                        row = page["revisions"][0]
                        pages.append({"work": work, "title": page["title"], "pageid": page["pageid"],
                                      **{k: row[k] for k in ("revid", "sha1", "timestamp")}})
            plan = {"rights": rights, "pages": sorted(pages, key=lambda row: (row["work"], row["pageid"]))}
            immutable(inventory, format_json(plan))
        records, unavailable = [], []
        for row in plan["pages"]:
            path = directory / f"{row['revid']}.html"
            if not path.exists():
                try:
                    received = fetch(client, {"action": "parse", "oldid": row["revid"], "prop": "text|revid",
                                              "disableeditsection": 1}, ledger, PER_PAGE)["parse"]
                    if (received["revid"], received["pageid"]) != (row["revid"], row["pageid"]):
                        raise ValueError("Wikisource revision does not match the request")
                    immutable(path, received["text"].encode())
                except (OSError, ValueError) as exc:
                    if ledger[0] > CAP:
                        raise
                    unavailable.append({"revision": row["revid"], "error_type": type(exc).__name__})
                    continue
            size, digest = file_identity(path)
            records.append({**row, "rendered_bytes": size, "rendered_sha256": digest,
                            "attribution_url": f"{SITE}/w/index.php?oldid={row['revid']}",
                            "history_url": f"{SITE}/w/index.php?curid={row['pageid']}&action=history"})
        receipt = {"documents": records, "site_rightsinfo": plan["rights"], "source": SITE, "terms": TERMS,
                   "unavailable": expected["unavailable"] if expected else unavailable}
        if expected is not None and receipt != expected:
            raise ValueError("Wikisource render differs from the pinned capture")
        immutable(capture, format_json(receipt))
    return receipt


def prepare(previous, capture, lid, out):
    registry.load_plugins()
    norm = registry.make("normalizer", NORMALIZER)
    language_pin = json.loads(lid.read_bytes())
    classify = FastTextLanguage(lid.parent / "lid.176.ftz", language_pin["model_sha256"])
    wiki_previous = json.loads(previous.read_bytes())
    if wiki_previous["qualification"] != "EXPLORATORY_WIKIMEDIA_DEVELOPMENT" or wiki_previous["verified_freeze"]:
        raise ValueError("Expected the previous pinned exploratory Wikipedia development input")
    wikibooks = json.loads((capture / "capture.json").read_bytes())
    recipe = {"identity": IDENTITY, "previous_wikipedia_sha256": file_identity(previous)[1],
              "wikisource_capture_sha256": file_identity(capture / "capture.json")[1],
              "reference_protocol_sha256": file_identity(AUDIT / "evaluation-protocol-v1.json")[1],
              "extractor_sha256": sha((HERE / "development_corpus.py").read_bytes().replace(b"\r\n", b"\n")),
              "preparation_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
              "normalizer": NORMALIZER, "normalizer_fingerprint": registry.fingerprint("normalizer", NORMALIZER),
              "lid_model_sha256": language_pin["model_sha256"], "license": LICENSE,
              "dedup": "source ancestry and raw/canonical SHA-256 and NFC-white char 5-gram Jaccard >=.85",
              "book_family_policy": WORKS, "canonical_budget_per_split": 2 * 1024**2}
    out.mkdir(parents=True, exist_ok=True)
    with _lock(out / ".prepare.lock"):
        immutable(out / "recipe.json", format_json(recipe))
        index = DuplicateIndex(":memory:")
        rejected = Counter()
        try:
            with index.db:
                for split in ("train", "validation"):
                    entry = wiki_previous["splits"][split]
                    path = checked_member(previous.resolve().parent, {**entry, "path": entry["file"]})
                    for row in load_rows(path):
                        raw = row.pop("text")
                        if sha(raw.encode()) != row["sha256"] or sha(norm.normalize(raw).encode()) != row["canonical_sha256"]:
                            raise ValueError("Previous Wikipedia document changed")
                        index.insert({**row, "assigned": split, "family": row["group"]}, raw)
                for row in wikibooks["documents"]:
                    rendered = capture / f"{row['revid']}.html"
                    if file_identity(rendered) != (row["rendered_bytes"], row["rendered_sha256"]):
                        raise ValueError("Wikisource artifact failed verification")
                    raw = extract(rendered.read_text(encoding="utf-8"))
                    normalized = norm.normalize(raw)
                    scores = profile(raw, normalized, classify)
                    tag = scores["language"]
                    if (len(raw.encode()) < 600 or not tag["windows"] or tag["ml_score"] < .8
                            or scores["malayalam_letters"] < .70 * scores["letters"]
                            or scores["replacement_chars"] or scores["control_chars"]):
                        rejected["wikisource_quality_screen"] += 1
                        continue
                    work = row["work"]
                    _, author, domain, split = WORKS[work]
                    group = f"mlwikisource:{row['pageid']}"
                    index.insert({
                        **row, "id": f"{group}:{row['revid']}", "group": group, "family": f"book:{work}",
                        "src": "mlwikisource", "domain": domain, "assigned": split, "original_author": author,
                        "sha256": sha(raw.encode()), "canonical_sha256": sha(normalized.encode()),
                        "canonical_bytes": len(normalized.encode()), "raw_bytes": len(raw.encode()),
                        "source_license": LICENSE, "quality": scores,
                    }, raw)
                clusters = {}
                for group, raw, row in index.members():
                    clusters.setdefault(group, []).append({**row, "text": raw})
                evidence = index.counts()
        finally:
            index.db.close()
        selected = {"train": [], "validation": []}
        for group, members in sorted(clusters.items(), key=lambda item: (sha(f"{IDENTITY}\n{item[0]}".encode()), item[0])):
            if len({r["assigned"] for r in members}) != 1:
                rejected["cross_split_duplicate_rows"] += len(members)
                continue
            rejected["within_split_duplicates"] += len(members) - 1
            row = members[0]
            split = row.pop("assigned")
            selected[split].append({**row, "dedup_group": group})
        files = {}
        for split, rows in selected.items():
            nbytes = sum(row["canonical_bytes"] for row in rows)
            if not 0 < nbytes <= recipe["canonical_budget_per_split"]:
                raise ValueError("Optimization corpus differs from the local source-byte budget")
            files[split] = {**write_rows_once(out / f"{split}.jsonl", rows), "canonical_bytes": nbytes,
                            "records": [{k: v for k, v in row.items() if k != "text"} for row in rows],
                            "source_counts": dict(Counter(row["src"] for row in rows)),
                            "domain_counts": dict(Counter(row["domain"] for row in rows))}
        manifest = {"schema": "mom-boundary-development-v1", "qualification": "EXPLORATORY_WIKIMEDIA_PROSE",
                    "recipe": recipe, "splits": files, "duplicate_witnesses": evidence, "excluded": dict(rejected),
                    "verified_freeze": False, "independent_reviewers": []}
        immutable(out / "manifest.json", format_json(manifest))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--lid", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--pins", type=Path)
    args = parser.parse_args()
    if args.fetch:
        acquire(args.capture, args.pins)
    result = prepare(args.previous, args.capture, args.lid, args.out)
    print(json.dumps({s: {k: d[k] for k in ("rows", "canonical_bytes", "source_counts", "domain_counts")}
                      for s, d in result["splits"].items()}))
