"""Stage an independently versioned draft from pinned Malayalam Wikipedia revisions."""

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe.diagnose import HERE, format_json, immutable, sha
from mltok import registry
from mltok.lib.bpe_progress import canonical

DRAFT = "ml-wikipedia-replacement-source-draft-v1"
REPO = "https://ml.wikipedia.org"
LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"
TERMS = "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"


def duplicate_view(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def shingles(text):
    return {text[i:i + 5] for i in range(max(1, len(text) - 4))}


def build(export, out):
    raw_export = export.read_bytes()
    if len(raw_export) > 700_000:
        raise ValueError("Public source export exceeds the draft budget")
    registry.load_plugins()
    spec = yaml.safe_load((HERE / "arms.yaml").read_bytes())["normalizer"]
    norm = registry.make("normalizer", spec).hf()
    rows, exact, signatures = [], set(), []
    total = 0
    pages = json.loads(raw_export)["query"]["pages"]
    for page in sorted(pages, key=lambda p: p["pageid"]):
        if page["ns"] != 0 or len(page["revisions"]) != 1:
            raise ValueError("Each selected page needs one pinned main-namespace revision")
        revision = page["revisions"][0]
        slot = revision["slots"]["main"]
        if slot["contentmodel"] != "wikitext":
            raise ValueError("Unexpected public revision content model")
        text = slot["content"]
        original = text.encode()
        total += len(original)
        if not text or len(original) != revision["size"] or total > 350_000:
            raise ValueError("Public revision size mismatch or source budget exceeded")
        digest = revision["sha1"]
        actual = int(hashlib.sha1(original).hexdigest(), 16)
        if actual != int(digest, 16 if len(digest) == 40 else 36):
            raise ValueError("Public revision content does not match its published digest")
        cleaned = norm.normalize_str(text).encode()
        view = duplicate_view(text)
        sig = shingles(view)
        group = f"mlwiki:{page['pageid']}"
        identity = f"{group}:{revision['revid']}"
        if any(row["group"] == group for row in rows):
            raise ValueError("Repeated document ancestry in the source export")
        if view in exact or any(100 * len(sig & other) >= 85 * len(sig | other) for other in signatures):
            raise ValueError("Exact or near-duplicate candidate requires source review before splitting")
        exact.add(view)
        signatures.append(sig)
        rows.append({
            "id": identity, "group": group, "text": text, "domain": "encyclopedic", "src": "ml-wikipedia",
            "title": page["title"], "source_repository": REPO, "source_revision": revision["revid"],
            "revision_timestamp": revision["timestamp"], "source_sha1": digest,
            "attribution_url": f"{REPO}/w/index.php?oldid={revision['revid']}",
            "source_license": LICENSE, "usage_evidence": TERMS, "rights_review": "pending page-specific review",
            "sha256": sha(original), "raw_bytes": len(original),
            "canonical_sha256": sha(cleaned), "canonical_bytes": len(cleaned),
            "duplicate_review": "draft exhaustive NFC/whitespace five-gram screen passed",
        })
    if len(rows) < 4:
        raise ValueError("Insufficient independent draft source documents")
    ranked = sorted(rows, key=lambda r: (sha(f"{DRAFT}\n{r['group']}".encode()), r["group"]))
    heldout = {r["group"] for r in ranked[:max(1, len(rows) // 3)]}
    out.mkdir(parents=True, exist_ok=True)
    immutable(out / "source-export.json", raw_export)
    files = {}
    for split in ("train", "eval"):
        items = [r for r in ranked if (r["group"] in heldout) == (split == "eval")]
        filename = f"draft-{split}.jsonl"
        payload = b"".join(canonical({**r, "split": split}) + b"\n" for r in items)
        immutable(out / filename, payload)
        files[split] = {"file": filename, "bytes": len(payload), "sha256": sha(payload),
                        "raw_bytes": sum(r["raw_bytes"] for r in items),
                        "canonical_bytes": sum(r["canonical_bytes"] for r in items),
                        "documents": [{k: v for k, v in r.items() if k != "text"} for r in items]}
    draft = {
        "identity": DRAFT, "status": "DRAFT_SOURCE_STAGING_ONLY", "verified_freeze": False,
        "source_export_sha256": sha(raw_export), "corpus_documents": len(rows), "source_wikitext_bytes": total,
        "normalizer_fingerprint": registry.fingerprint("normalizer", spec), "wikitext_retained": True,
        "split_method": "Stable hash rank of draft ID and page ancestry. First floor(N/3) groups held out.",
        "panels": files, "human_approvals": [
            "Confirm page-specific rights, third-party quotations and attribution before reuse.",
            "Review full-document prose extraction and repeat near-duplicate checks on extracted text.",
            "Collect the missing domains and complete the protocol quotas, then obtain two freeze reviews.",
            "Independently adjudicate orthography and keep held-out data sealed from hyperparameter selection.",
        ],
    }
    immutable(out / "draft-manifest.json", format_json(draft))
    return draft


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = build(args.export, args.out)
    print(json.dumps({"status": data["status"], "documents": data["corpus_documents"],
                      "wikitext_bytes": data["source_wikitext_bytes"]}))
