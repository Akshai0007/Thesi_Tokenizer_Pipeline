"""Build a checkpointed disk-backed corpus proposal and bounded source-quality survey."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe import corpus_quality as quality
from experiments.mom_bpe.corpus_overlap import DuplicateIndex, THRESHOLD
from experiments.mom_bpe.diagnose import AUDIT, HERE, format_json, immutable, sha
from mltok import registry
from mltok.lib.atomic import atomic_writer
from mltok.lib.bpe_progress import canonical
from mltok.snapshots import _lock, file_identity

IDENTITY = "malayalam-source-study-v1"
ROW_LIMIT = 2 * 1024**2
TOKEN_PANEL = 2000


def capture_rows(path):
    with path.open("rb") as stream:
        while line := stream.readline(ROW_LIMIT + 1):
            if len(line) > ROW_LIMIT:
                raise ValueError("Captured record exceeds the bounded reader limit")
            yield json.loads(line)


def provisional_filter(metrics):
    """Transparent screening, calibrated on neither a final panel nor a gold LID set."""
    lid = metrics["language"]
    reasons = []
    if metrics["letters"] < 20:
        reasons.append("short_language_evidence")
    if metrics["malayalam_letters"] < 0.70 * metrics["letters"]:
        reasons.append("low_malayalam_script_share")
    if lid is None or lid["ml_score"] is None or lid["ml_score"] < 0.80:
        reasons.append("weak_independent_language_evidence")
    for marker in ("replacement_chars", "unattached_ml_marks", "repeated_virama",
                   "html_hits", "synthetic_proxy_hits"):
        if metrics[marker]:
            reasons.append(marker)
    return reasons


def sample_partition(cluster):
    position = int(sha(f"{IDENTITY}\n{cluster}".encode())[:12], 16) % 10
    return "validation" if position == 0 else "heldout_proposal" if position == 1 else "train"


@dataclass(frozen=True)
class Components:
    source: object = capture_rows
    quality: object = quality.profile
    filter: object = provisional_filter
    overlap: object = DuplicateIndex
    sampling: object = sample_partition


def provenance(source, data, capture):
    attrs = data.get("attributes", {})
    origin, host = quality.page_group({**attrs, "url": attrs.get("source_document_url") or attrs.get("url") or attrs.get("u")})
    coordinate = str(data["row_index"])
    doc_id = f"{source}:{coordinate}"
    native = attrs.get("doc_id") or attrs.get("id")
    parent = attrs.get("source_document_id") or native
    parent_key = f"{source}:native:{parent}" if isinstance(parent, (str, int)) and str(parent) else None
    register = attrs.get("web-register")
    register_hint = None
    if isinstance(register, dict):
        codes = {k: v for k, v in register.items() if isinstance(k, str) and k.isupper() and isinstance(v, (int, float))}
        if codes:
            register_hint = max(codes, key=lambda k: (codes[k], k))
    declared = attrs.get("domain") or attrs.get("source")
    return {
        "id": doc_id, "upstream_document_id": native, "src": source,
        "group": origin or parent_key or doc_id, "native_group": parent_key,
        "source_repository": capture["repo"], "source_revision": capture["revision"],
        "storage_revision": capture.get("storage_revision"), "source_coordinate": coordinate,
        "source_url": origin, "host": host,
        "complete_document": source not in ("smc", "indiccorp") and data.get("complete_document", True) is True,
        "domain": "encyclopedic" if source == "smc" else "unreviewed",
        "upstream_register_hint": register_hint,
        "upstream_subsource": declared if isinstance(declared, str) else None,
    }


def write_rows_once(path, rows):
    digest = hashlib.sha256()
    size = 0
    count = 0
    try:
        with atomic_writer(path, overwrite=False) as stream:
            for row in rows:
                line = canonical(row) + b"\n"
                stream.write(line)
                digest.update(line)
                size += len(line)
                count += 1
    except FileExistsError:
        if file_identity(path) != (size, digest.hexdigest()):
            raise ValueError("An existing corpus artifact differs from the current proposal") from None
    return {"file": path.name, "size": size, "sha256": digest.hexdigest(), "rows": count}


def read_receipts(captures, index_only):
    receipts, revisions = {}, {}
    for entry in captures:
        for manifest in ([entry] if entry.is_file() else sorted(entry.glob("*.json"))):
            capture = json.loads(manifest.read_bytes())
            source = capture["source"]
            shard = capture.get("shard", "")
            if not isinstance(source, str) or not source or not isinstance(shard, str) or (shard and not index_only):
                raise ValueError("An explicit capture shard is only supported in indexed preparation")
            filename = capture.get("file", f"{source}.jsonl")
            if (Path(filename).name != filename or any(c in filename for c in "\\:/")
                    or filename in ("", ".", "..")):
                raise ValueError("Expected a capture filename within its manifest directory")
            path = manifest.parent / filename
            identity = (source, shard)
            revision = tuple(capture.get(k) for k in ("repo", "revision", "storage_revision"))
            if identity in receipts or revisions.setdefault(source, revision) != revision:
                raise ValueError("Duplicate capture shard or mixed revisions for a single source")
            if file_identity(path) != (capture["bytes"], capture["sha256"]):
                raise ValueError("Captured source hash or byte size differs from its manifest")
            receipts[identity] = (path, capture)
    if not receipts:
        raise ValueError("At least one captured source is required")
    return receipts


def source_summary(db, source, normalizer, analyzer, panel_size):
    lengths, char_lengths, scores, total = [], [], [], Counter()
    words, first, languages, subsource, registers = Counter(), Counter(), Counter(), Counter(), Counter()
    hosts, flagged = set(), Counter()
    panel_tokens = 0
    for body, text in db.execute("SELECT body,text FROM documents WHERE source=? ORDER BY n", (source,)):
        row = json.loads(body)
        q = row["quality"]
        lengths.append(q["raw_bytes"])
        char_lengths.append(q["chars"])
        total.update({k: v for k, v in q.items() if isinstance(v, int)})
        flagged.update(row["quarantine_reasons"])
        if not row["quarantine_reasons"]:
            total["screening_pass_docs"] += 1
        if q["language"] and q["language"]["windows"]:
            scores.append(q["language"]["ml_score"])
            languages.update(q["language"]["top_languages"])
        if row["host"]:
            hosts.add(row["host"])
        if row["upstream_subsource"]:
            subsource[row["upstream_subsource"]] += 1
        if row["upstream_register_hint"]:
            registers[row["upstream_register_hint"]] += 1
        for word in quality.words(normalizer(text)):
            words[word] += 1
            if panel_tokens < panel_size:
                first[word] += 1
                panel_tokens += 1

    def quantiles(values):
        ordered = sorted(values)
        return {name: ordered[int(frac * (len(ordered) - 1))] for name, frac in
                (("minimum", 0), ("p10", .1), ("median", .5), ("p90", .9), ("maximum", 1))} if ordered else {}

    return {
        "documents": len(lengths), "characters": quantiles(char_lengths), "raw_byte_distribution": quantiles(lengths),
        "totals": dict(total), "screening_flagged_documents": dict(flagged),
        "malayalam_letter_share": total["malayalam_letters"] / total["letters"] if total["letters"] else None,
        "latin_letter_share": total["latin_letters"] / total["letters"] if total["letters"] else None,
        "language_assessed_documents": len(scores), "ml_confidence": quantiles(scores),
        "ml_confidence_mean": sum(scores) / len(scores) if scores else None,
        "language_windows_by_top_label": dict(languages), "document_url_host_count": len(hosts),
        "upstream_register_argmax": dict(registers), "upstream_subsource_count": len(subsource),
        "malayalam_word_tokens": sum(words.values()), "malayalam_types": len(words),
        "single_occurrence_types": sum(n == 1 for n in words.values()),
        "shared_token_budget": panel_tokens, "types_at_shared_budget": len(first),
        "morphological_sample": quality.morphological_summary(words, analyzer) if analyzer else None,
    }, words, first


def build(captures, out, lid_manifest, *, morphology=False, index_only=False, components=Components()):
    registry.load_plugins()
    norm_spec = yaml.safe_load((HERE / "arms.yaml").read_bytes())["normalizer"]
    normalization = registry.make("normalizer", norm_spec).hf().normalize_str
    lid = json.loads(lid_manifest.read_bytes())
    language = quality.FastTextLanguage(lid_manifest.parent / "lid.176.ftz", lid["model_sha256"])
    analyzer = None
    morph_identity = None
    if morphology and index_only:
        raise ValueError("Morphological reporting requires bounded survey mode")
    if morphology:
        from importlib import resources
        from mlmorph import Analyser

        binary = resources.files("mlmorph").joinpath("data", "malayalam.a").read_bytes()
        expected = json.loads((HERE / "review-results-v1.json").read_bytes())["morphology_desk_review"]
        versions = {name: importlib.metadata.version(name) for name in expected["dependencies"]}
        if sha(binary) != expected["compiled_transducer_sha256"] or versions != expected["dependencies"]:
            raise ValueError("The morphology transducer differs from the reviewed configuration")
        analyzer = Analyser()
        morph_identity = {"versions": versions, "sha256": sha(binary)}
    receipts = read_receipts(captures, index_only)
    recipe = {
        "schema": IDENTITY, "approval": "SOURCE_DIAGNOSTIC_ONLY", "normalizer": norm_spec, "index_only": index_only,
        "components": {
            name: {"callable": f"{item.__module__}.{item.__qualname__}",
                   "code_sha256": sha(Path(inspect.getsourcefile(item)).read_bytes().replace(b"\r\n", b"\n"))}
            for name, item in vars(components).items()
        },
        "normalizer_fingerprint": registry.fingerprint("normalizer", norm_spec),
        "baseline_protocol_sha256": sha((AUDIT / "evaluation-protocol-v1.json").read_bytes()),
        "code_sha256": {p: sha((HERE / p).read_bytes().replace(b"\r\n", b"\n")) for p in
                        ("corpus_build.py", "corpus_quality.py", "corpus_overlap.py")},
        "lid_identity": lid, "morphology_identity": morph_identity,
        "unicode_version": unicodedata.unidata_version, "regex_version": importlib.metadata.version("regex"),
        "captures": [c for _, (_, c) in sorted(receipts.items())],
        "near_duplicate": {"view": "NFC + whitespace collapse", "gram_size": 5,
                           "jaccard_percent": THRESHOLD, "candidate_generation": "exact prefix bound, verified sets"},
        "screening": {"min_letters": 20, "min_ml_score": 0.80, "min_malayalam_letter_share": 0.70},
    }
    out.mkdir(parents=True, exist_ok=True)
    with _lock(out / ".study.lock"):
        immutable(out / "recipe.json", format_json(recipe))
        index = components.overlap(out / "study.sqlite3")
        db = index.db
        try:
            for (source, shard), (path, capture) in sorted(receipts.items()):
                key = json.dumps([source, shard])
                existing = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
                if existing:
                    if existing != (capture["sha256"],):
                        raise ValueError("Indexed shard checkpoint fingerprint differs")
                    continue
                # The completed input shard and its duplicate edges commit together.
                with db:
                    count = 0
                    for data in components.source(path):
                        if data["src"] != source:
                            raise ValueError("Capture row has the wrong source")
                        raw = data["text"]
                        text = normalization(raw)
                        metrics = components.quality(raw, text, language)
                        index.insert({
                            **provenance(source, data, capture), "sha256": sha(raw.encode()),
                            "canonical_sha256": sha(text.encode()), "raw_bytes": metrics["raw_bytes"],
                            "canonical_bytes": metrics["canonical_bytes"], "quality": metrics,
                            "quarantine_reasons": components.filter(metrics),
                            "license_review": "pending; source policy and underlying content rights require review",
                        }, raw)
                        count += 1
                    if count != capture["rows"] or file_identity(path) != (capture["bytes"], capture["sha256"]):
                        raise ValueError("Captured source changed during indexing or has the wrong row count")
                    db.execute("INSERT INTO settings VALUES(?,?)", (key, capture["sha256"]))
            if index_only:
                report = {"identity": IDENTITY, "status": "INDEX_COMPLETE_REVIEW_PENDING",
                          "recipe_sha256": file_identity(out / "recipe.json")[1], "verified_freeze": False}
                immutable(out / "index-complete.json", format_json(report))
                return report
            prior_words = {}
            by_source = {}
            # The survey profiles bounded captures, not production training volumes.
            sources = sorted(source for source, _ in receipts)
            available_words = {s: sum(1 for raw, in db.execute("SELECT text FROM documents WHERE source=?", (s,))
                                      for _ in quality.words(normalization(raw))) for s in sources}
            panel_size = min([TOKEN_PANEL] + [n for n in available_words.values() if n])
            for source in sources:
                by_source[source], prior_words[source], _ = source_summary(db, source, normalization, analyzer, panel_size)
            for source, summary in by_source.items():
                others = set().union(*(set(v) for k, v in prior_words.items() if k != source))
                summary["types_not_observed_in_other_captures"] = len(prior_words[source].keys() - others)
                summary["unique_type_panel_caveat"] = "Measured within unequal bounded captures; not a corpus population claim."
            with db:
                allocation = write_rows_once(out / "proposed-layout.jsonl", (
                    {**row, "connected_group": group, "proposed_split": components.sampling(group)}
                    for group, _, row in index.members()
                ))
            report = {"identity": IDENTITY, "status": "SOURCE_DIAGNOSTIC_ONLY",
                      "by_source": by_source, "overlap": index.counts(), "proposal": allocation,
                      "recipe_sha256": file_identity(out / "recipe.json")[1], "verified_freeze": False}
            immutable(out / "quality-report.json", format_json(report))
        finally:
            db.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", action="append", required=True, type=Path)
    parser.add_argument("--lid", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--morphology", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    report = build(args.captures, args.out, args.lid, morphology=args.morphology, index_only=args.index_only)
    print(json.dumps({k: v["documents"] for k, v in report["by_source"].items()}
                     if "by_source" in report else report))
