"""Build small CPU-only MOM-BPE fixtures and publish an immutable diagnostic report."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import yaml
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT = ROOT / "audit/48k-baseline"
sys.path.insert(0, str(ROOT))

from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.atomic import atomic_writer
from mltok.lib.backends import HFBackend
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.exposure import account_exposure
from mltok.lib.mom_backend import runtime_identity
from mltok.lib.morph_alignment import load_reviewed_forms, project, reviewed_mlmorph_candidates
from mltok.lib.orthographic_units import UNITIZERS
from mltok.lib.surface_scoring import score_morphology

TRAIN = (
    "പൂക്കൾ കടൽക്കര വീട്ടിൽ കുട്ടികൾക്കും ക്ഷേത്രോത്സവം",
    "പൂക്കൾ പൂക്കളും മഴപ്പാട്ടും നിലാവും",
    "വീട്ടിൽ കടൽക്കര കുട്ടികൾക്കും പൂക്കൾ",
    "abc123 മലയാളം തമിഴ് <pad> ▁ <|im_start|> ക്ഷേത്രം",
) * 8


def sha(data):
    return hashlib.sha256(data).hexdigest()


def immutable(path, data):
    try:
        with atomic_writer(path, overwrite=False) as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError("Different evidence already exists. Use a new output directory.")


def metrics(backend, docs, metric_names):
    out = {}
    for name in metric_names:
        out.update(registry.make("metric", name).compute(backend, docs))
    out.pop("tok_per_sec", None)
    out.pop("boundary_examples", None)
    out.pop("offset_v2_examples", None)
    return out


def format_json(payload):
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def analyzer_probe(formulas):
    from mlmorph import Analyser

    lock = json.loads((AUDIT / "morphology-dependencies.json").read_bytes())
    versions = {item["project"]: importlib.metadata.version(item["project"]) for item in lock["artifacts"]}
    if any(versions[item["project"]] != item["version"] for item in lock["artifacts"]):
        raise ValueError("Mlmorph and SFST versions must match the audit lock")
    analyzer = Analyser()
    evidence = {}
    for word, rows in formulas.items():
        candidates = analyzer.analyse(word) or []
        supported = reviewed_mlmorph_candidates(word, analyzer, rows)
        evidence[word] = {"analyses": len(candidates),
                          "projected": project(word, supported, UNITIZERS["akshara"])}
    return {"versions": versions, "forms": evidence}


def run(out, *, mlmorph=False):
    spec_path = HERE / "arms.yaml"
    cfg = yaml.safe_load(spec_path.read_bytes())
    registry.load_plugins()
    norm = registry.make("normalizer", cfg["normalizer"])
    pre = registry.make("pretokenizer", cfg["pretokenizer"])
    align_path = HERE / "diagnostic_alignments.json"
    align_hash = sha(align_path.read_bytes())
    formulas = load_reviewed_forms(align_path, align_hash)
    projected = {word: project(word, choices, UNITIZERS["akshara"]) for word, choices in formulas.items()}
    fixture_data = "".join(json.dumps({"text": t, "src": "authored_diagnostic"}) + "\n" for t in TRAIN).encode()
    frozen = json.loads((AUDIT / "fixtures.json").read_bytes())
    docs = [(text, domain) for domain, texts in frozen["domains"].items() for text in texts]
    orthography = [(text, category) for category, text in frozen["unicode_cases"]]
    fidelity = [row["text"] for row in json.loads((AUDIT / "fidelity-cases.json").read_bytes())["cases"]]
    freeze_hashes = {n: sha((AUDIT / n).read_bytes())
                     for n in ("fixtures.json", "fidelity-cases.json", "evaluation-protocol-v1.json")}
    out.mkdir(parents=True, exist_ok=True)
    immutable(out / "train-fixtures.jsonl", fixture_data)
    outputs = {}
    fingerprints = {}
    backends = {}
    for arm, spec in cfg["arms"].items():
        if arm == "T0":
            p = AUDIT / "original" / spec["published_revision"] / "tokenizer.json"
            if not p.is_file():
                outputs[arm] = {"available": False}
                continue
            if sha(p.read_bytes()) != spec["tokenizer_sha256"]:
                raise ValueError("The immutable published tokenizer failed SHA-256 verification")
            model = HFBackend(Tokenizer.from_file(str(p)), registry.make("normalizer", "ml"))
            outputs[arm] = {
                "available": True,
                "ordinary": metrics(model, docs, ("basic", "roundtrip", "boundary_offsets_v2")),
                "fidelity": metrics(model, [(text, "fidelity") for text in fidelity], ("roundtrip",)),
            }
            continue
        spec = dict(spec)
        algorithm_spec = {**cfg["common_algorithm"], "require_full_vocabulary": False}
        if spec.pop("use_verified_alignment", False):
            algorithm_spec.update(alignment_file=str(align_path), alignment_sha256=align_hash)
        algorithm_spec.update(spec)
        algo = registry.make("algorithm", algorithm_spec)
        source = {**algorithm_spec}
        if "alignment_file" in source:
            source["alignment_file"] = align_path.name  # Machine-independent config ID.
        fingerprints[arm] = registry.fingerprint("algorithm", source)
        backend, info = algo.train(TrainContext(
            vocab=384, specials=DEFAULT_SPECIALS, normalizer=norm, pretokenizer=pre,
            out_dir=out / arm, word_counts=None, shard_paths=[str(out / "train-fixtures.jsonl")], log=lambda *_: None,
        ))
        check = algo.load(out / arm, norm)
        if check.encode_batch(fidelity) != backend.encode_batch(fidelity):
            raise AssertionError("Reloaded artifact changed the token IDs")
        backends[arm] = check
        references = [
            {"status": "adjudicated", "text": text, "group": text, "offsets": list(sites)}
            for text, sites in projected.items() if sites
        ]
        exposure_rows = [{"text": t, "id": f"diagnostic-{i}"} for i, t in enumerate(TRAIN)]
        outputs[arm] = {
            "model_sha256": sha((out / arm / "mom-bpe.json").read_bytes()),
            "train_diagnostic": {k: v for k, v in info.items() if k != "train_seconds"},
            "ordinary": metrics(check, docs, cfg["metrics"]),
            "orthography": metrics(check, orthography, ("mom_surface",)),
            "fidelity": metrics(check, [(text, "fidelity") for text in fidelity], ("mom_surface",)),
            "synthetic_morphology_projection_check": score_morphology(
                check.encode_offsets, check.fallback_ids, references),
            "fixed_source_bytes": account_exposure(check, exposure_rows),
            "fixed_training_tokens": account_exposure(check, exposure_rows, 120),
        }
    left, right = backends["T2"], backends["T3-off"]
    zero_control = (
        left.artifact["merges"] == right.artifact["merges"]
        and left.artifact["initial"] == right.artifact["initial"]
        and left.encode_batch([t for t, _ in docs] + fidelity) == right.encode_batch([t for t, _ in docs] + fidelity)
    )
    if not zero_control:
        raise AssertionError("The zero-weight arm changed the tokenizer")
    result = {
        "scope": "authored CPU diagnostics only; no frozen-corpus or downstream measurements",
        "arm_config_sha256": sha(spec_path.read_bytes()), "runtime": runtime_identity(),
        "diagnostic_implementation_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "training_fixture_sha256": sha(fixture_data), "diagnostic_alignment_sha256": align_hash,
        "foundation_inputs_sha256": freeze_hashes, "algorithm_fingerprints": fingerprints,
        "normalizer_fingerprint": registry.fingerprint("normalizer", cfg["normalizer"]),
        "pretokenizer_fingerprint": registry.fingerprint("pretokenizer", cfg["pretokenizer"]),
        "metric_fingerprints": {name: registry.fingerprint("metric", name) for name in
                                sorted(set(cfg["metrics"]) | {"boundary_offsets_v2"})},
        "synthetic_projected_boundaries": projected, "zero_weight_control_exact": zero_control, "arms": outputs,
    }
    if mlmorph:
        result["optional_mlmorph_probe"] = analyzer_probe(formulas)
    immutable(out / "COMPLETE.json", format_json(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mlmorph", action="store_true", help="Use the previously verified local analyzer wheels")
    args = parser.parse_args()
    data = run(args.out, mlmorph=args.mlmorph)
    print(json.dumps({"arms": list(data["arms"]), "zero_weight_control_exact": data["zero_weight_control_exact"]}))
