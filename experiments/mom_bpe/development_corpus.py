"""Prepare a bounded, attributed Wikimedia development split for exploratory fits."""

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from experiments.mom_bpe.corpus_acquire import public_session
from experiments.mom_bpe.corpus_build import write_rows_once
from experiments.mom_bpe.corpus_overlap import DuplicateIndex
from experiments.mom_bpe.corpus_quality import script_metrics
from experiments.mom_bpe.diagnose import AUDIT, format_json, immutable, sha
from mltok import registry
from mltok.snapshots import _lock, file_identity

API = "https://ml.wikipedia.org/w/api.php"
TERMS = "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content"
LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"
IDENTITY = "mlwiki-development-v1"
NORMALIZER = {"type": "ml_conservative", "terminal_chillu": True}
TOTAL_CAP = 24 * 1024**2
ARTICLE_CAP = 768 * 1024
PROSE_CAP = 18000
TITLES = (
    "കേരളം", "മലയാളം", "മലയാളലിപി", "തിരുവിതാംകൂർ", "കാസർഗോഡ്", "വയനാട്", "എറണാകുളം",
    "ആലപ്പുഴ", "പാലക്കാട്", "ഇടുക്കി", "പത്തനംതിട്ട", "കുമരകം", "കുട്ടനാട്", "മുന്നാർ",
    "സഹ്യപർവതം", "പെരിയാർ", "വേമ്പനാട്ടുകായൽ", "അഷ്ടമുടിക്കായൽ", "നീലഗിരി",
    "മഴ", "കടൽ", "നക്ഷത്രം", "ചന്ദ്രൻ", "സൗരയൂഥം", "ഭൂമി", "അന്തരീക്ഷം", "ജലം",
    "നെല്ല്", "തെങ്ങ്", "മാവ്", "ആന", "പുലി", "മയിൽ", "കടുവ", "തേനീച്ച",
    "ഓണം", "വിഷു", "കഥകളി", "മോഹിനിയാട്ടം", "തെയ്യം", "വള്ളംകളി", "കളരിപ്പയറ്റ്",
    "ഗ്രന്ഥശാല", "കമ്പ്യൂട്ടർ", "ഇന്റർനെറ്റ്", "വിക്കിപീഡിയ", "സ്വതന്ത്ര സോഫ്റ്റ്‌വെയർ",
    "ജനാധിപത്യം", "വിദ്യാഭ്യാസം", "ആരോഗ്യം", "ഭരണഘടന", "ഇന്ത്യ", "കൃഷി", "കാലാവസ്ഥ",
    "പത്മനാഭസ്വാമി ക്ഷേത്രം", "ശബരിമല", "മട്ടാഞ്ചേരി", "കോഴിക്കോട്", "കൊച്ചി",
    "വൈക്കം മുഹമ്മദ് ബഷീർ", "ശ്രീനാരായണ ഗുരു", "കുമാരനാശാൻ", "മലയാളസാഹിത്യം",
)
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_EXCLUDED = {"table", "figure", "blockquote", "q", "pre", "code", "aside", "sup", "style", "script", "math", "audio"}
_CLASSES = {"reference", "reflist", "hatnote", "infobox", "navbox", "noprint", "mw-editsection", "quotebox"}
_QUOTE = re.compile(r'["“”‘’«»]|https?://|copyright|all rights reserved', re.I)


class Prose(HTMLParser):
    """Collect ordinary article paragraphs while excluding attributed excerpts."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.paragraphs = [], []
        self.current = None
        self.rejected = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        blocked = (tag in _EXCLUDED or bool(classes & _CLASSES)
                   or "mw:Extension/ref" in (attrs.get("typeof") or ""))
        inherited = self.stack and self.stack[-1][1]
        if tag == "p" and not (blocked or inherited):
            self.current, self.rejected = [], False
        if self.current is not None and (tag in {"q", "blockquote"} or "quotebox" in classes):
            self.rejected = True
        if tag not in _VOID:
            self.stack.append((tag, bool(blocked or inherited)))
        elif tag == "br" and self.current is not None and not (blocked or inherited):
            self.current.append(" ")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "p" and self.current is not None:
            text = " ".join("".join(self.current).split())
            script = script_metrics(text)
            if (not self.rejected and not _QUOTE.search(text) and len(text.encode()) >= 240
                    and script["malayalam_letters"] >= 0.70 * script["letters"] > 0):
                self.paragraphs.append(text)
            self.current = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, content):
        if self.current is not None and not (self.stack and self.stack[-1][1]):
            self.current.append(content)


def extract(html):
    parser = Prose()
    parser.feed(html)
    parser.close()
    paragraphs = []
    size = 0
    for text in parser.paragraphs:
        length = len(text.encode()) + (2 if paragraphs else 0)
        if size + length <= PROSE_CAP:
            paragraphs.append(text)
            size += length
    return "\n\n".join(paragraphs)


def fetch(client, params, ledger, cap):
    with client.get(API, params={"format": "json", "formatversion": 2, **params}, timeout=25, stream=True) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(8192):
            ledger[0] += len(chunk)
            if ledger[0] > TOTAL_CAP or len(data) + len(chunk) > cap:
                raise ValueError("Wikimedia bounded response exceeded the sample budget")
            data.extend(chunk)
    result = json.loads(data)
    if "error" in result:
        raise ValueError("Wikimedia API refused the selected article revision")
    return result


def acquire(directory, pins=None):
    directory.mkdir(parents=True, exist_ok=True)
    host = "https://ml.wikipedia.org"
    ledger = [0]
    expected = json.loads(pins.read_bytes())["source_receipt"] if pins is not None else None
    request = {"titles": list(TITLES), "per_response_cap": ARTICLE_CAP, "total_cap": TOTAL_CAP,
               "pinned_rendering_sha256": sha(format_json(expected)) if expected else None}
    with _lock(directory / ".acquire.lock"), public_session(
        "MalayalamTokenizerResearch/1.0 (public Wikimedia study; https://github.com/Akshai0007/Thesi_Tokenizer_Pipeline)"
    ) as client:
        immutable(directory / "request.json", format_json(request))
        if (directory / "capture.json").is_file():
            return json.loads((directory / "capture.json").read_bytes())
        inv = directory / "revisions.json"
        if inv.is_file():
            inventory = json.loads(inv.read_bytes())
        elif expected is not None:
            inventory = {"license": expected["site_rightsinfo"], "pages": [
                {key: row[key] for key in ("pageid", "title", "revid", "parentid", "timestamp", "sha1")}
                for row in expected["documents"]
            ]}
            immutable(inv, format_json(inventory))
        else:
            rights = fetch(client, {"action": "query", "meta": "siteinfo", "siprop": "rightsinfo"}, ledger, 65536)
            license_url = rights["query"]["rightsinfo"]["url"].removesuffix("deed.ml")
            if license_url.rstrip("/") != LICENSE.rstrip("/"):
                raise ValueError("Unexpected Wikimedia site license")
            pages = {}
            for start in range(0, len(TITLES), 12):
                query = fetch(client, {
                    "action": "query", "titles": "|".join(TITLES[start:start + 12]), "redirects": 1,
                    "prop": "revisions", "rvprop": "ids|timestamp|sha1",
                }, ledger, 65536)
                for page in query["query"]["pages"]:
                    if page["ns"] == 0 and page.get("revisions"):
                        pages[page["pageid"]] = {"pageid": page["pageid"], "title": page["title"], **page["revisions"][0]}
            inventory = {"license": rights["query"]["rightsinfo"], "pages": [pages[k] for k in sorted(pages)]}
            immutable(inv, format_json(inventory))
        recovered = []
        unavailable = []
        for page in inventory["pages"]:
            rid = page["revid"]
            path = directory / f"revision-{rid}.html"
            if not path.is_file():
                try:
                    obj = fetch(client, {"action": "parse", "oldid": rid, "prop": "text|revid", "disableeditsection": 1},
                                ledger, ARTICLE_CAP)["parse"]
                    if obj["revid"] != rid or obj["pageid"] != page["pageid"]:
                        raise ValueError("Wikimedia delivered a different source revision")
                except (ValueError, OSError) as exc:
                    if ledger[0] > TOTAL_CAP:
                        raise
                    unavailable.append({"revision": rid, "error_type": type(exc).__name__})
                    continue
                immutable(path, obj["text"].encode())
            recovered.append({**page, "render_sha256": file_identity(path)[1], "render_bytes": path.stat().st_size,
                              "attribution_url": f"{host}/w/index.php?oldid={rid}",
                              "history_url": f"{host}/w/index.php?curid={page['pageid']}&action=history"})
        receipt = {"source": API, "license": LICENSE, "terms": TERMS, "site_rightsinfo": inventory["license"],
                   "documents": recovered, "unavailable": expected["unavailable"] if expected else unavailable}
        if expected is not None and expected != receipt:
            raise ValueError("Pinned Wikimedia rendering differs. Preserve the prior capture.")
        immutable(directory / "capture.json", format_json(receipt))
    return receipt


def prepare(capture, out):
    registry.load_plugins()
    normalizer = registry.make("normalizer", NORMALIZER)
    source_hash = file_identity(capture / "capture.json")[1]
    source = json.loads((capture / "capture.json").read_bytes())
    previous = json.loads((Path(__file__).with_name("replacement-draft-v1.json")).read_bytes())
    reserved_ids = {row["group"] for scope in previous["panels"].values() for row in scope["documents"]}
    recipe = {
        "identity": IDENTITY, "capture_sha256": source_hash, "license": LICENSE, "terms": TERMS,
        "normalizer": NORMALIZER, "normalizer_fingerprint": registry.fingerprint("normalizer", NORMALIZER),
        "extractor_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "frozen_protocol_sha256": file_identity(AUDIT / "evaluation-protocol-v1.json")[1],
        "prior_replacement_draft_sha256": file_identity(Path(__file__).with_name("replacement-draft-v1.json"))[1],
        "use": "exploratory development only", "verified_freeze": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    with _lock(out / ".prepare.lock"):
        immutable(out / "recipe.json", format_json(recipe))
        index = DuplicateIndex(":memory:")
        try:
            with index.db:
                for row in source["documents"]:
                    path = capture / f"revision-{row['revid']}.html"
                    if file_identity(path) != (row["render_bytes"], row["render_sha256"]):
                        raise ValueError("Pinned rendered page is damaged")
                    group = f"mlwiki:{row['pageid']}"
                    if group in reserved_ids:
                        continue
                    text = extract(path.read_text(encoding="utf-8"))
                    if len(text.encode()) < 600:
                        continue
                    canonical = normalizer.normalize(text)
                    item = {
                        **row, "id": f"{group}:{row['revid']}", "group": group, "src": "mlwiki",
                        "domain": "encyclopedic", "sha256": sha(text.encode()),
                        "canonical_sha256": sha(canonical.encode()), "raw_bytes": len(text.encode()),
                        "canonical_bytes": len(canonical.encode()), "source_license": LICENSE,
                    }
                    index.insert(item, text)
                candidates = {}
                for group, text, row in index.members():
                    candidates.setdefault(group, {**row, "text": text})
                overlap = index.counts()
        finally:
            index.db.close()
        ranked = sorted(candidates, key=lambda group: (sha(f"{IDENTITY}\n{group}".encode()), group))
        if len(ranked) < 12:
            raise ValueError("Development candidates did not provide enough distinct source groups")
        selected = {"validation": [], "train": []}
        for i, group in enumerate(ranked):
            selected["validation" if i % 5 == 0 else "train"].append(candidates[group])
        manifests = {}
        for split, rows in selected.items():
            manifests[split] = {
                **write_rows_once(out / f"{split}.jsonl", rows),
                "canonical_bytes": sum(row["canonical_bytes"] for row in rows),
                "raw_bytes": sum(row["raw_bytes"] for row in rows),
                "documents": [{k: v for k, v in row.items() if k != "text"} for row in rows],
            }
        result = {
            "schema": "mom-tokenizer-development-v1", "qualification": "EXPLORATORY_WIKIMEDIA_DEVELOPMENT",
            "recipe": recipe, "splits": manifests, "overlap_witnesses": overlap,
            "deduplication": "raw SHA-256, canonical SHA-256, article ancestry, NFC whitespace 5-gram Jaccard >= 0.85",
            "license_basis": "Wikimedia page text terms; excluded visible quotations, templates, media and marked references",
            "independent_reviewers": [], "verified_freeze": False,
        }
        immutable(out / "manifest.json", format_json(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--pins", type=Path)
    args = parser.parse_args()
    if args.fetch:
        acquire(args.capture, args.pins)
    result = prepare(args.capture, args.out)
    print(json.dumps({key: result["splits"][key]["rows"] for key in ("train", "validation")}))
