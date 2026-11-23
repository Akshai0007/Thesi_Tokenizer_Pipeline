"""Read-only preflight for the future licensed evaluation freeze."""

import argparse
import gzip
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = ROOT / "audit/48k-baseline/evaluation-protocol-v1.json"
sys.path.insert(0, str(ROOT))

from mltok.snapshots import file_identity
from mltok import registry
from mltok.lib.morph_alignment import independent_reviewers, load_reviewed_forms, project
from mltok.lib.orthographic_units import UNITIZERS


def checked_member(root, entry):
    name = entry["path"]
    rel = PurePosixPath(name)
    if rel.is_absolute() or rel.as_posix() != name or ".." in rel.parts or any(c in name for c in "\\:"):
        raise ValueError("Frozen manifest has an unsafe local path")
    path = root
    for part in rel.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Frozen artifact path contains a symbolic link")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError("Frozen artifact is missing or outside the input directory")
    if file_identity(path) != (entry["size"], entry["sha256"]):
        raise ValueError("Frozen artifact size or SHA-256 mismatch")
    return path


def load_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for line in stream:
            row = json.loads(line)
            canonical = (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            if line != canonical:
                raise ValueError("Expected canonical UTF-8 JSONL")
            yield row


def verify(manifest_path, *, require_morphology=True):
    path = Path(manifest_path).resolve()
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    manifest_bytes = path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema") != "mom-frozen-inputs-v1"
            or manifest.get("protocol_sha256") != hashlib.sha256(protocol_bytes).hexdigest()
            or not independent_reviewers(manifest.get("reviewers"))):
        raise ValueError("Expected a protocol-pinned freeze independently reviewed by two maintainers")
    sources = manifest["sources"]
    if not sources or any(not all(details.get(k) for k in ("revision", "license", "permission_evidence"))
                          for details in sources.values()):
        raise ValueError("Frozen sources require revisions, licenses, and permission evidence")
    for key in ("near_duplicate_review", "train_eval_separation_review"):
        checked_member(path.parent, manifest[key])

    arms = yaml.safe_load((HERE / "arms.yaml").read_bytes())
    registry.load_plugins()
    if manifest["normalizer_fingerprint"] != registry.fingerprint("normalizer", arms["normalizer"]):
        raise ValueError("Frozen normalizer implementation or runtime changed")
    normalizer = registry.make("normalizer", arms["normalizer"])
    native = normalizer.hf()
    split_ledger, ids = {}, set()
    eval_groups, eval_digests, eval_canonical, domains = set(), set(), set(), Counter()
    for split in ("eval", "train"):
        tally = Counter()
        for entry in manifest["splits"][split]:
            verified = checked_member(path.parent, entry)
            for row in load_rows(verified):
                raw, identifier, group = row["text"], row["id"], row["group"]
                if not isinstance(raw, str) or not raw or not identifier or not group or identifier in ids:
                    raise ValueError("Frozen records need nonempty text and unique stable identifiers")
                if row["src"] not in sources or hashlib.sha256(raw.encode()).hexdigest() != row["sha256"]:
                    raise ValueError("Frozen document source or SHA-256 is invalid")
                ids.add(identifier)
                canonical = " ".join(unicodedata.normalize("NFC", raw).split()).encode()
                digest = hashlib.sha256(canonical).digest()
                lexical = native.normalize_str(raw)
                lexical_sha = hashlib.sha256(lexical.encode()).hexdigest()
                if (lexical != normalizer.normalize(raw)
                        or lexical_sha != row["canonical_sha256"]
                        or len(lexical.encode()) != row["canonical_bytes"]):
                    raise ValueError("Frozen lexical normalization or canonical fingerprint changed")
                if split == "eval":
                    if digest in eval_digests or lexical_sha in eval_canonical:
                        raise ValueError("Duplicate evaluation documents after canonicalization")
                    eval_groups.add(group)
                    eval_digests.add(digest)
                    eval_canonical.add(lexical_sha)
                    domains[row["domain"]] += 1
                elif group in eval_groups or digest in eval_digests or lexical_sha in eval_canonical:
                    raise ValueError("Train/eval overlap by group or canonical text")
                tally[row["src"]] += 1
        if dict(tally) != manifest["source_counts"][split] or not tally:
            raise ValueError("Frozen source counts disagree with the manifest")
        split_ledger[split] = sum(tally.values())

    panel = protocol["panels"]
    if dict(domains) != panel["natural_text"]["target_per_domain"]:
        raise ValueError("Frozen natural evaluation does not meet the protocol quotas")
    orthography = list(load_rows(checked_member(path.parent, manifest["orthography_panel"])))
    approved = [row for row in orthography if row["status"] == "adjudicated"]
    if (len(approved) < panel["orthography"]["target_reviewed_cases"]
            or len({row["id"] for row in approved}) != len(approved)
            or any(not row["text"] or not row["category"] or not independent_reviewers(row["reviewers"])
                   for row in approved)):
        raise ValueError("Orthography panel is not independently adjudicated")
    reviewed = _verify_morphology(path.parent, manifest, native, panel) if require_morphology else None
    if file_identity(path)[1] != manifest_sha:
        raise ValueError("Frozen manifest changed during verification")
    return {"manifest_sha256": manifest_sha, "documents": split_ledger,
            "reviewed_orthography_cases": len(approved), "reviewed_morphology_cases": reviewed,
            "calibration": "Review receipts checked for integrity; reviewer and license claims need human sign-off"}


def _verify_morphology(root, manifest, native, panel):
    morphology = [row for row in load_rows(checked_member(root, manifest["morphology_panel"]))
                  if row["status"] == "adjudicated"]
    for row in morphology:
        text, offsets = row["text"], row["offsets"]
        if (not text or native.normalize_str(text) != text or not row["group"] or not independent_reviewers(row["reviewers"])
                or offsets != sorted(set(offsets)) or any(type(i) is not int or not 0 < i < len(text) for i in offsets)):
            raise ValueError("Invalid adjudicated morphology surface offsets")
        if "stem_end" in row and (type(row["stem_end"]) is not int or row["stem_end"] not in (*offsets, len(text))):
            raise ValueError("A morphology stem end must be an annotated boundary")
    if (len({row["id"] for row in morphology}) != len(morphology)
            or len(morphology) < panel["morphology"]["target_reviewed_forms"]
            or len({row["group"] for row in morphology}) < panel["morphology"]["target_minimum_groups"]):
        raise ValueError("Morphology panel does not meet the protocol size and family targets")
    alignment = checked_member(root, manifest["training_alignments"])
    rules = json.loads(alignment.read_bytes())
    if rules.get("schema") != "mom-surface-alignments-v1" or rules.get("qualification") != "adjudicated_training":
        raise ValueError("Training alignment artifact is not marked independently adjudicated")
    reviewed = load_reviewed_forms(alignment, manifest["training_alignments"]["sha256"])
    heldout_groups = {row["group"] for row in morphology}
    heldout_forms = {row["text"] for row in morphology}
    for word, analyses in reviewed.items():
        if word in heldout_forms or native.normalize_str(word) != word or any(
            not row.get("group") or row["group"] in heldout_groups or not independent_reviewers(row["reviewers"])
            for row in analyses
        ):
            raise ValueError("Training alignment lacks adjudication or overlaps held-out morphology")
    if not any(project(word, analyses, UNITIZERS["akshara"]) for word, analyses in reviewed.items()):
        raise ValueError("Training morphology has no defensible surface alignment")
    return len(morphology)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.exit(2, "Preflight blocked. A verified frozen-input manifest has not been supplied.\n")
    try:
        print(json.dumps(verify(args.manifest), sort_keys=True))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"Preflight blocked. {type(exc).__name__}: {exc}\n")
