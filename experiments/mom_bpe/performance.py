"""Run one bounded tokenizer fit or prepare its downstream exposure ledger."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

import yaml
from tokenizers import Tokenizer

from experiments.mom_bpe.diagnose import AUDIT, HERE, format_json, immutable, sha
from experiments.mom_bpe.pilot import peak_rss_bytes
from experiments.mom_bpe.verify_frozen import checked_member, load_rows, verify
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import canonical
from mltok.lib.exposure import account_exposure
from mltok.lib.mom_backend import SurfaceEncoding
from mltok.lib.orthographic_units import UNITIZERS
from mltok.lib.picky_backend import ARTIFACT
from mltok.lib.surface_scoring import score_morphology
from mltok.lib.unicode_ml import ML_SET, akshara_starts
from mltok.snapshots import _lock, file_identity

ARMS = tuple(f"T{i}" for i in range(8))
DEV_MAX_BYTES = 2 * 1024**2
BYTE_SPELLING = re.compile(r"<0x[0-9A-F]{2}>")


class PublishedBaseline:
    def __init__(self):
        spec = json.loads((AUDIT / "baseline-artifacts.json").read_bytes())
        folder = AUDIT / "original" / spec["revision"]
        for item in spec["artifacts"]:
            checked_member(folder.resolve(), item)
        self.identity = spec["artifacts"][1]["sha256"]
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        vocabulary = self.tokenizer.get_vocab()
        self.special_ids = {piece: vocabulary[piece] for piece in DEFAULT_SPECIALS if piece in vocabulary}
        self.fallback_ids = frozenset(i for piece, i in vocabulary.items() if BYTE_SPELLING.fullmatch(piece))
        self.occupied = self.tokenizer.get_vocab_size()

    def encode_offsets(self, canonical_text):
        result = self.tokenizer.encode(canonical_text, add_special_tokens=False)
        return SurfaceEncoding(canonical_text, result.ids, result.tokens, result.offsets)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def input_plan(path, mode, spec, norm, vocab):
    path = path.resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    morphology = bool(spec.get("morphology_penalty_milli"))
    if mode == "frozen":
        if vocab != 48000:
            raise ValueError("The primary frozen comparison requires 48000 occupied slots")
        panel = morphology or bool(manifest.get("morphology_panel"))
        verified = verify(path, require_morphology=panel)
        if verified["manifest_sha256"] != sha(raw):
            raise ValueError("Frozen evaluation manifest changed during verification")
        if morphology or (panel and spec.get("segmenter") == "hybrid"):
            alignment = manifest["training_alignments"]
            spec.update(alignment_file=str(checked_member(path.parent, alignment)), alignment_sha256=alignment["sha256"])
        data = {name: [checked_member(path.parent, item) for item in manifest["splits"][split]]
                for name, split in (("train", "train"), ("evaluation", "eval"))}
        data["orthography"] = [checked_member(path.parent, manifest["orthography_panel"])]
        data["morphology"] = [checked_member(path.parent, manifest["morphology_panel"])] if panel else []
        return data, manifest, sha(raw)
    if (manifest["schema"] != "mom-tokenizer-development-v1" or manifest["verified_freeze"]
            or manifest["qualification"] != "EXPLORATORY_WIKIMEDIA_DEVELOPMENT"
            or manifest["recipe"]["normalizer_fingerprint"] != registry.fingerprint("normalizer", norm)):
        raise ValueError("Expected a matching exploratory Wikimedia development manifest")
    if morphology:
        raise ValueError("Morphology training needs an independently adjudicated training and held-out manifest")
    data = {}
    groups, digests = set(), set()
    for name, split in (("train", "train"), ("evaluation", "validation")):
        item = manifest["splits"][split]
        entry = {"path": item["file"], "size": item["size"], "sha256": item["sha256"]}
        if item["size"] > DEV_MAX_BYTES:
            raise ValueError("Development input exceeds the laptop corpus cap")
        data[name] = [checked_member(path.parent, entry)]
        rows = list(load_rows(data[name][0]))
        if (len(rows) != item["rows"] or not rows
                or sum(r["canonical_bytes"] for r in rows) != item["canonical_bytes"]):
            raise ValueError("Development exposure differs from its manifest")
        for row in rows:
            if row["group"] in groups or row["canonical_sha256"] in digests:
                raise ValueError("Development data contains an overlapping article family or canonical text")
            groups.add(row["group"])
            digests.add(row["canonical_sha256"])
    return data, manifest, sha(raw)


def from_files(paths):
    for path in paths:
        yield from load_rows(path)


def checked_canonical(row, normalizer):
    raw = row["text"]
    canonical_text = normalizer.normalize(raw)
    if row.get("sha256") != sha(raw.encode()) or row.get("canonical_sha256") != sha(canonical_text.encode()):
        raise ValueError("Document raw or canonical SHA-256 differs")
    if len(canonical_text.encode()) != row["canonical_bytes"]:
        raise ValueError("Document canonical byte length differs")
    return canonical_text


def measure(backend, rows, normalizer, occupied, *, require_hashes=True):
    # Fallible native byte offsets (e.g. the historical backend) receive the
    # same lexical-only coverage rule, without imputing excluded transitions.
    counts = Counter()
    ids_used = Counter()
    lexical = set()
    for row in rows:
        raw = row["text"]
        target = checked_canonical(row, normalizer) if require_hashes else normalizer.normalize(raw)
        start = perf_counter()
        result = backend.encode_offsets(target)
        counts["encoding_seconds"] += perf_counter() - start
        if result.text != target:
            raise ValueError("Encoder offsets do not refer to the shared canonical text")
        decoded = backend.decode(result.ids)
        counts["docs"] += 1
        counts["raw_roundtrip"] += decoded == raw
        counts["canonical_roundtrip"] += decoded == target
        counts["canonical_bytes"] += len(target.encode())
        counts["canonical_words"] += len(target.split())
        counts["ids"] += len(result.ids)
        ids_used.update(result.ids)
        counts["transitions_all"] += max(0, len(result.ids) - 1)
        counts["fallback_ids"] += sum(i in backend.fallback_ids for i in result.ids)
        alphabet_specials = backend.special_ids.values()
        lexical.update(i for i in result.ids if i not in backend.fallback_ids and i not in alphabet_specials)
        valid = []
        for token, (left, right) in zip(result.ids, result.offsets):
            usable = token not in backend.fallback_ids and token not in alphabet_specials and 0 <= left < right <= len(target)
            if usable and isinstance(backend, PublishedBaseline):
                usable = backend.decode([token]) == target[left:right]
            valid.append(usable)
        egc, akshara = set(), set()
        for name, refs in (("grapheme", egc), ("akshara", akshara)):
            location = 0
            for unit in UNITIZERS[name](target):
                location += len(unit)
                refs.add(location)
        alternate = akshara_starts(target)
        counts["grapheme_reference_sites"] += max(0, len(egc) - 1)
        counts["akshara_reference_sites"] += sum(
            target[pos - 1] in ML_SET and target[pos] in ML_SET for pos in akshara if 0 < pos < len(target))
        for j, ((left, pos), (following, end)) in enumerate(zip(result.offsets, result.offsets[1:])):
            if not (valid[j] and valid[j + 1] and left < pos == following < end):
                continue
            counts["eligible_transitions"] += 1
            counts["grapheme_breaks"] += pos not in egc
            if target[pos - 1] in ML_SET and target[pos] in ML_SET:
                counts["malayalam_eligible_transitions"] += 1
                counts["akshara_breaks"] += pos not in akshara
                counts["alternate_akshara_breaks"] += pos not in alternate
    output = dict(counts)
    for name in ("grapheme_breaks", "akshara_breaks", "alternate_akshara_breaks", "eligible_transitions",
                 "malayalam_eligible_transitions", "grapheme_reference_sites", "akshara_reference_sites"):
        output.setdefault(name, 0)
    output["offset_exclusions"] = counts["transitions_all"] - counts["eligible_transitions"]
    output["lexical_slots"] = occupied - len(backend.fallback_ids) - len(backend.special_ids)
    output["standalone_lexical_ids"] = len(lexical)
    output["one_occurrence_lexical_ids"] = sum(ids_used[i] == 1 for i in lexical)
    output["fertility"] = counts["ids"] / counts["canonical_words"] if counts["canonical_words"] else None
    output["bytes_per_token"] = counts["canonical_bytes"] / counts["ids"] if counts["ids"] else None
    output["fallback_rate"] = counts["fallback_ids"] / counts["ids"] if counts["ids"] else None
    output["encoding_bytes_per_second"] = counts["canonical_bytes"] / counts["encoding_seconds"] if counts["encoding_seconds"] else None
    return output


def run(path, out, arm, vocab, *, mode="development", fit_only=False, predicted_tokens=None):
    registry.load_plugins()
    config_bytes = (HERE / "performance-arms.yaml").read_bytes()
    config = yaml.safe_load(config_bytes)
    spec = {**config["algorithm"], **config["arms"][arm]} if arm != "T0" else {}
    data, manifest, manifest_sha = input_plan(path, mode, spec, config["normalizer"], vocab)
    norm = registry.make("normalizer", config["normalizer"])
    total_train = (manifest["splits"]["train"]["canonical_bytes"] if mode == "development" else None)
    # Verify one shared canonical definition before training or planning.
    for file in (*data["train"], *data["evaluation"]):
        for row in load_rows(file):
            checked_canonical(row, norm)
    identity = {
        "manifest_sha256": manifest_sha, "mode": mode, "arm": arm, "vocab_target": 48000 if arm == "T0" else vocab,
        "config_sha256": sha(config_bytes),
        "driver_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "normalizer": registry.fingerprint("normalizer", config["normalizer"]),
        "pretokenizer": registry.fingerprint("pretokenizer", config["pretokenizer"]),
        "algorithm": registry.fingerprint("algorithm", spec) if arm != "T0" else "published_48k",
    }
    fit_path = out / "fit.json"
    with _lock(out / ".fit.lock"):
        immutable(out / "identity.json", format_json(identity))
        if arm == "T0":
            backend = PublishedBaseline()
            model_sha = backend.identity
            fit = {"model_sha256": model_sha, "training": {"occupied_vocab": backend.occupied},
                   "seconds": None, "peak_rss_bytes": None}
        else:
            algorithm = registry.make("algorithm", spec)
            model_path = out / ARTIFACT
            if fit_path.is_file():
                fit = json.loads(fit_path.read_bytes())
                model_sha = file_identity(model_path)[1]
                if fit["model_sha256"] != model_sha:
                    raise ValueError("Fit receipt differs from the completed tokenizer")
                backend = algorithm.load(out, norm)
            else:
                start = perf_counter()
                backend, info = algorithm.train(TrainContext(
                    vocab=vocab, specials=DEFAULT_SPECIALS, normalizer=norm,
                    pretokenizer=registry.make("pretokenizer", config["pretokenizer"]), out_dir=out,
                    word_counts=None, shard_paths=list(map(str, data["train"])),
                    resources={"merge_checkpoint_every": 512},
                ))
                if total_train is not None and info["canonical_bytes"] != total_train:
                    raise ValueError("Training did not consume the shared canonical bytes")
                model_sha = file_identity(model_path)[1]
                fit = {"model_sha256": model_sha, "training": info,
                       "seconds": perf_counter() - start, "peak_rss_bytes": peak_rss_bytes()}
        immutable(fit_path, format_json(fit))
        if fit_only:
            return fit
        occupied = fit["training"]["occupied_vocab"]
        if mode == "frozen" and occupied != 48000:
            raise ValueError("Underfilled primary arm. The scientific evaluation remains sealed.")
        result_path = out / ("validation.json" if mode == "development" else "evaluation.json")
        if result_path.exists():
            result = json.loads(result_path.read_bytes())
            digest = result.pop("result_sha256")
            if digest != sha(canonical(result)) or result["model_sha256"] != model_sha:
                raise ValueError("Existing metrics differ from their verified receipt")
            result["result_sha256"] = digest
        else:
            validation = measure(backend, from_files(data["evaluation"]), norm, occupied)
            if data.get("orthography"):
                orthography = [row for row in from_files(data["orthography"]) if row["status"] == "adjudicated"]
            else:
                orthography = [{"text": row["text"]} for row in
                               json.loads((AUDIT / "fidelity-cases.json").read_bytes())["cases"]]
            result = {"qualification": manifest.get("qualification", "FROZEN_EVALUATION"),
                      "identity": identity, "model_sha256": model_sha, "fit": fit, "validation": validation,
                      "special_ids": backend.special_ids, "fidelity_probes": measure(
                          backend, orthography, norm, occupied, require_hashes=False)}
            if data.get("morphology"):
                result["adjudicated_morphology"] = score_morphology(
                    backend.encode_offsets, backend.fallback_ids, from_files(data["morphology"]),
                )
            result["result_sha256"] = sha(canonical(result))
            immutable(result_path, format_json(result))
        if predicted_tokens is not None:
            if any(result[key]["canonical_roundtrip"] != result[key]["docs"]
                   for key in ("validation", "fidelity_probes")):
                raise ValueError("Downstream exposure planning requires complete canonical round-trip fidelity")
            exposure = {
                "qualification": "TOKENIZER_EXPOSURE_PLAN_ONLY", "model_sha256": model_sha,
                "manifest_sha256": manifest_sha, "protocol": config["downstream"],
                "fixed_source_bytes": account_exposure(backend, from_files(data["train"])),
                "fixed_training_tokens": account_exposure(backend, from_files(data["train"]), predicted_tokens),
            }
            immutable(out / f"exposure-{predicted_tokens}.json", format_json(exposure))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--vocab", type=int, default=48000)
    parser.add_argument("--mode", choices=("development", "frozen"), default="development")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--predicted-tokens", type=int)
    args = parser.parse_args()
    result = run(args.input, args.out, args.arm, args.vocab, mode=args.mode, fit_only=args.fit_only,
                 predicted_tokens=args.predicted_tokens)
    print(json.dumps({"arm": args.arm, "occupied": result.get("fit", result)["training"]["occupied_vocab"]}))
