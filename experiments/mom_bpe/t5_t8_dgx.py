"""Prepare, execute, or audit exactly two fixed-source seed-17 proxy jobs."""
import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

OPERATIONAL = "9b872ffb5a66bdf4d23df58acebbba7e788915d8"
MANIFEST_SHA = "4266825c1132cb44555313b0b68cd489bbfe55566a508be2a8ed606a0bad7750"
ARMS = ("T5", "T8")
BASELINES = ("M48", "F-space")
MODE = "fixed_source_bytes"
BOOTSTRAP_SEED = 20261125


def require(condition, text):
    if not condition:
        raise ValueError(text)


def safe(root, name):
    from mltok.lib.artifact_paths import relative_path, unlinked_path

    return unlinked_path(root / relative_path(name))


def read(path):
    return json.loads(path.read_bytes())


def verified_file(path, entry):
    from experiments.mom_bpe.dgx_handoff import file_digest

    require(path.is_file() and file_digest(path) == (entry["size"], entry["sha256"]),
            f"Missing or inconsistent input: {path}")


def restore(runs, archive_path, manifest):
    """Copy only the selected original members into a separate input directory."""
    from mltok.lib.artifact_paths import unlinked_path
    from mltok.lib.atomic import atomic_writer

    archive_path = unlinked_path(archive_path)
    verified_file(archive_path, manifest["archive_b"])
    pending = []
    for name, entry in manifest["files"].items():
        path = safe(runs, name)
        if path.exists():
            verified_file(path, entry)
        else:
            pending.append((path, entry))
    require(shutil.disk_usage(runs.parent).free >= sum(e["size"] for _, e in pending),
            "Insufficient space for verified input restoration")
    with ZipFile(archive_path) as archive:
        for path, entry in pending:
            path.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with archive.open(entry["archive_member"]) as member, atomic_writer(path, overwrite=False) as out:
                while chunk := member.read(2**20):
                    size += len(chunk)
                    digest.update(chunk)
                    out.write(chunk)
                require((size, digest.hexdigest()) == (entry["size"], entry["sha256"]),
                        f"Restored archive member differs: {entry['archive_member']}")
    return {"restored": len(pending), "verified_existing": len(manifest["files"]) - len(pending)}


def verify_inputs(runs, manifest):
    from experiments.mom_bpe import document_proxy as driver
    from experiments.mom_bpe.dgx_handoff import plan
    from experiments.mom_bpe.proxy_blocks import EncodedSplit, receipt
    from experiments.mom_bpe.scale_proxy_plan import verified_backend

    for name, entry in manifest["files"].items():
        verified_file(safe(runs, name), entry)
    raw = {}
    for name, entry in manifest["raw_corpus_optional"].items():
        path = safe(runs, name)
        if path.exists():
            verified_file(path, entry)
        raw[name] = "verified" if path.exists() else "absent; frozen encoded inputs are available"

    driver.registry.load_plugins()
    require(not driver.registry.load_errors(), "Plugin import errors")
    spec = {"type": "ml_conservative", "terminal_chillu": True}
    normalizer = driver.registry.make("normalizer", spec)
    normalizer_hash = driver.registry.fingerprint("normalizer", spec)
    specials = dict(zip(driver.DEFAULT_SPECIALS, range(len(driver.DEFAULT_SPECIALS))))
    plans, docs = {}, {}
    for arm in ARMS:
        expected = manifest["arms"][arm]
        folder, fit_dir = (safe(runs, expected[key]) for key in ("cache", "fit"))
        cache, cache_digest = receipt(folder / "cache.json")
        fit = read(fit_dir / "fit.json")
        backend, model_sha, occupied = verified_backend(fit_dir, fit, normalizer)
        identity = cache["identity"]
        require((cache_digest, model_sha) == (expected["cache_receipt_sha256"], expected["tokenizer_sha256"])
                and identity["tokenizer_sha256"] == model_sha and identity["arm"] == arm
                and identity["manifest_sha256"] == manifest["corpus_manifest_sha256"],
                f"Cache, corpus or tokenizer identity differs for {arm}")
        require(backend.special_ids == specials == identity["specials"] and len(backend.fallback_ids) == 256
                and occupied == backend.vocab_size == identity["vocab"] == 48000
                and normalizer_hash == identity["normalizer"] == fit["identity"]["normalizer"]
                and identity["streams"]["validation"] == manifest["validation"]["canonical_stream_sha256"],
                f"Vocabulary, normalization, or validation identity differs for {arm}")
        saved, digest = receipt(safe(runs, expected["ledger"]))
        observed = plan(driver, folder, MODE)
        require(observed == saved and digest == expected["plan_sha256"]
                and observed["config_sha256"] == manifest["config_sha256"]
                and observed["optimizer_sha256"] == manifest["optimizer_sha256"]
                and observed["config"]["seed"] == 17 and observed["budget"] == 50_000_000
                and observed["exposure"]["ordered_complete_documents_sha256"] == manifest["fixed_source_stream_sha256"]
                and observed["forecast"] == expected["forecast"] and observed["validation"] == expected["validation"],
                f"Recomputed exposure differs for {arm}")
        docs[arm] = EncodedSplit(folder, "validation", cache["splits"]["validation"]).docs
        plans[arm] = observed
    require(doc_keys(docs["T5"]) == doc_keys(docs["T8"]), "T5/T8 development documents differ")
    return plans, docs, raw


def doc_keys(docs):
    return [(d["id"], d["canonical_sha256"], d["canonical_bytes"]) for d in docs]


def verify_result(result, held, *, check_targets):
    from mltok.lib.canonical_bpb import summarize_bpb

    evaluation = result["evaluation"]
    rows = evaluation["per_document"]
    require(doc_keys(rows) == doc_keys(held) and len({r["id"] for r in rows}) == 186,
            "Development document order, identity or count differs")
    for doc, frozen in zip(rows, held, strict=True):
        targets = doc["predicted_tokens"]
        require(not check_targets or targets == frozen["length"] + 1, "Validation target count differs")
        expected = summarize_bpb([doc["nll_nats"]], doc["canonical_bytes"], targets)
        require(all(doc[k] == v for k, v in expected.items()), "Per-document BPB differs")
    totals = summarize_bpb((d["nll_nats"] for d in rows), sum(d["canonical_bytes"] for d in held),
                           sum(d["predicted_tokens"] for d in rows))
    require(all(evaluation[k] == v for k, v in totals.items()), "Aggregate BPB differs")
    return rows


def load_baselines(directory, manifest, held):
    from experiments.mom_bpe.dgx_handoff import file_digest
    from experiments.mom_bpe.dgx_readiness import PINS
    from experiments.mom_bpe.proxy_blocks import receipt

    results = {}
    for arm in BASELINES:
        folder = safe(directory, manifest["baseline"][arm]["path"])
        for name, digest in zip(("identity.json", "result.json"), PINS[arm, MODE], strict=True):
            require(file_digest(safe(folder, name))[1] == digest, f"Historical baseline differs: {arm}/{name}")
        result, _ = receipt(folder / "result.json")
        identity = read(folder / "identity.json")
        require(identity == result["identity"] and identity["config"]["seed"] == 17
                and identity["parameters"] == 38799936 and identity["mode"] == MODE
                and identity["manifest_sha256"] == manifest["corpus_manifest_sha256"]
                and identity["optimizer_sha256"] == manifest["optimizer_sha256"]
                and identity["config_sha256"] == manifest["config_sha256"]
                and identity["exposure"]["ordered_complete_documents_sha256"] == manifest["fixed_source_stream_sha256"]
                and identity["forecast"]["canonical_bytes"] == 49998152,
                f"Historical baseline exposure or implementation differs: {arm}")
        verify_result(result, held, check_targets=False)
        results[arm] = result
    return results


def verify_identity(identity, plan, arm, manifest):
    from experiments.mom_bpe import dgx_resources as gpu
    from experiments.mom_bpe import document_proxy as driver

    require(all(identity.get(k) == v for k, v in plan.items())
            and identity["manifest_sha256"] == manifest["corpus_manifest_sha256"]
            and identity["tokenizer_sha256"] == manifest["arms"][arm]["tokenizer_sha256"]
            and identity["parameters"] == 38799936
            and identity["implementation"] == {"driver": gpu.PINNED_DRIVER,
                                               "blocks": driver.code_digest(driver.blocks.__code__.co_filename),
                                               "scorer": driver.code_digest(driver.summarize_bpb.__code__.co_filename)}
            and identity["model_plugin"] == driver.registry.fingerprint("proxy_model", plan["config"]["model"]),
            f"Run identity conflict: {arm}")
    runtime = identity["runtime"]
    device = runtime.get("gpu")
    require(device and device["physical_index"] != 7 and "A100" in device["type"]
            and gpu.GPU_UUID.fullmatch(device["uuid"])
            and (runtime["system"], runtime["python"], runtime["torch"], runtime["numpy"], device["cuda"],
                 runtime["threads"], runtime["precision"], runtime["cublas_workspace"])
            == ("Linux", "3.12.0", "2.5.1+cu121", "2.5.2", "12.1", 1, "bf16_autocast", ":4096:8"),
            f"Unrecognized training runtime: {arm}")


def audit(out, arm, plan, manifest, *, inspect_resume=False):
    from experiments.mom_bpe import document_proxy as driver
    from experiments.mom_bpe.dgx_handoff import file_digest
    from experiments.mom_bpe.proxy_blocks import receipt

    state_path = safe(out, "identity.json")
    complete = safe(out, "result.json")
    identity = read(state_path) if state_path.is_file() else None
    if identity is not None:
        verify_identity(identity, plan, arm, manifest)
    checkpoints = {}
    for path in sorted(out.glob("step-*.pt"), reverse=True):
        match = driver.GENERATION.fullmatch(path.name)
        require(match and identity is not None, f"Unidentified checkpoint {path}")
        size, digest = file_digest(safe(out, path.name))
        checkpoints[path.name] = {"size": size, "sha256": digest, "valid_bytes": digest == match[2]}
    if complete.exists():
        result, _ = receipt(complete)
        ref = result["checkpoint"]
        match = driver.GENERATION.fullmatch(ref["path"])
        actual = checkpoints.get(ref["path"], {})
        require(result["identity"] == identity and match
                and int(match[1]) == plan["forecast"]["optimizer_updates"]
                and actual == {"size": ref["size"], "sha256": ref["sha256"], "valid_bytes": True},
                f"Completed result or final checkpoint differs: {arm}")
        return result, checkpoints
    if inspect_resume and checkpoints:
        for name, entry in checkpoints.items():
            if entry["valid_bytes"]:
                state = driver.torch.load(safe(out, name), map_location="cpu", weights_only=True)
                step = int(driver.GENERATION.fullmatch(name)[1])
                require(state["identity"] == identity and state["step"] == step
                        and 0 < step <= plan["forecast"]["optimizer_updates"]
                        and state["exposure"]["optimizer_updates"] == step,
                        f"Checkpoint payload identity conflict: {arm}/{name}")
                break
        else:
            raise ValueError(f"No checksum-valid generation for {arm}; preserve and recover from verified backup")
    return None, checkpoints


def pairs(results):
    import numpy as np

    first = next(iter(results.values()))["evaluation"]["per_document"]
    keys = doc_keys(first)
    sizes = np.array([d["canonical_bytes"] for d in first])
    index = np.random.default_rng(BOOTSTRAP_SEED).integers(len(first), size=(10_000, len(first)))
    denom = sizes[index].sum(axis=1) * math.log(2)
    losses = {}
    for name, result in results.items():
        rows = result["evaluation"]["per_document"]
        require(doc_keys(rows) == keys, "Paired development-document alignment differs")
        losses[name] = np.array([d["nll_nats"] for d in rows])
    values = {}
    for a, b in itertools.combinations(losses, 2):
        delta = losses[b] - losses[a]
        values[f"{b} minus {a}"] = {
            "delta_bpb": float(delta.sum() / (sizes.sum() * math.log(2))),
            "document_bootstrap_95pct": np.quantile(delta[index].sum(axis=1) / denom, [0.025, 0.975]).tolist(),
            "documents_favoring": {a: int((delta > 0).sum()), b: int((delta < 0).sum())},
            "tied_documents": int((delta == 0).sum()),
        }
    return values


def telemetry(out):
    from mltok.lib.artifact_paths import unlinked_path

    attempts = []
    for attempt in sorted(unlinked_path(out / "resource-telemetry").glob("*")):
        start = safe(attempt, "start.json")
        end = safe(attempt, "end.json")
        updates = sorted(attempt.glob("update-*.json"))
        tail = read(end) if end.exists() else read(safe(attempt, updates[-1].name)) if updates else {}
        attempts.append({"attempt": attempt.name, "closed": end.exists(), "initial": read(start) if start.exists() else None,
                         "last_measurement": tail})
    samples = [a["last_measurement"] for a in attempts]
    return {"attempts": attempts, "measured_training_seconds": sum(s.get("training_seconds", 0) for s in samples) if attempts else None,
            **{field: max((s[field] for s in samples if s.get(field) is not None), default=None)
               for field in ("peak_allocated_bytes", "peak_reserved_bytes")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("restore", "verify", "inspect", "T5", "T8", "report"))
    for name in ("source", "runs", "baseline", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--allocated", action="store_true")
    parser.add_argument("--uuid", default="")
    args = parser.parse_args()
    source = args.source.resolve()
    def git(*command):
        return subprocess.check_output(["git", "-C", str(source), *command], text=True).strip()
    require(git("rev-parse", "HEAD") == OPERATIONAL and not git("status", "--porcelain", "--ignored", "--untracked-files=all"),
            "Expected a clean, dedicated checkout at the immutable operational SHA")
    sys.path.insert(0, str(source))
    from experiments.mom_bpe import dgx_resources as gpu
    from mltok.lib.artifact_paths import unlinked_path
    from mltok.snapshots import _lock
    from experiments.mom_bpe.dgx_readiness import inspect_devices

    manifest_bytes = Path(__file__).with_name("t5_t8_inputs_v1.json").read_bytes()
    require(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA, "Input manifest differs")
    manifest = json.loads(manifest_bytes)
    output = unlinked_path(args.output)
    require(output.name == "t5-t8-seed17-fixed-source-v1" and not output.is_relative_to(args.runs.resolve())
            and not args.runs.resolve().is_relative_to(output)
            and not output.is_relative_to(args.baseline.resolve()) and not output.is_relative_to(source),
            "Experiment output must be isolated from source, input and historical trees")
    if args.action == "restore":
        require(args.archive is not None and args.enable, "Explicitly select the local verified Package B archive")
        require(args.runs.name == "t5-t8-frozen-runs-v1", "Restore into the dedicated input directory")
        print(json.dumps(restore(args.runs, args.archive, manifest)))
        return
    if args.action == "inspect":
        print(json.dumps(inspect_devices(), indent=2))
        return

    training = args.action in ARMS
    if training:
        require(args.enable and args.allocated, "Training is disabled until explicitly enabled and allocated")
        admission = gpu.GpuAdmission()
        admission.admit(args.uuid)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.uuid
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    from experiments.mom_bpe import document_proxy as driver
    from experiments.mom_bpe.proxy_blocks import publish

    plans, held, raw = verify_inputs(args.runs, manifest)
    baselines = load_baselines(args.baseline, manifest, held["T5"])
    if args.action == "verify":
        output.mkdir(parents=True, exist_ok=True)
        digest = publish(output / "verified-inputs.json", {"manifest_sha256": MANIFEST_SHA, "operational_commit": OPERATIONAL,
                         "source": manifest["corpus_manifest_sha256"], "plans": plans,
                         "baseline_bpb": {a: r["evaluation"]["bpb"] for a, r in baselines.items()}})
        print(json.dumps({"receipt": digest, "raw_corpus": raw, "forecasts": {a: p["forecast"] for a, p in plans.items()}}, indent=2))
        return
    if training:
        arm = args.action
        out = safe(output, arm)
        with _lock(output / ".ablation.lock"):
            result, generations = audit(out, arm, plans[arm], manifest, inspect_resume=True)
            if result is not None:
                verify_result(result, held[arm], check_targets=True)
                print(f"{arm} complete; final checkpoint verified")
                return
            hardware = driver.gpu_preflight()
            require("A100" in hardware["type"], "Expected the institutionally allocated A100")
            if (out / "identity.json").exists():
                require(hardware == read(out / "identity.json")["runtime"]["gpu"], "Resume on the original GPU identity")
            generation_bytes = 12 * manifest["parameters"] + 2**20
            valid_steps = {driver.GENERATION.fullmatch(name)[1] for name, member in generations.items()
                           if member["valid_bytes"]}
            require(shutil.disk_usage(output).free >= max(2, 9 - len(valid_steps)) * generation_bytes,
                    "Insufficient scratch for immutable generations")
            args.command, args.mode, args.device, args.threads = "run", MODE, "cuda", 1
            args.cache, args.out = safe(args.runs, manifest["arms"][arm]["cache"]), out
            args.budget = args.seed = args.update_targets = args.microbatch_blocks = args.checkpoint_updates = None
            result = driver.fit_cache(args)
            audit(out, arm, plans[arm], manifest)
            verify_result(result, held[arm], check_targets=True)
            print(f"{arm} complete; result and checkpoint verified")
        return

    results = dict(baselines)
    metrics = {}
    for arm in (*BASELINES, *ARMS):
        if arm in ARMS:
            out = safe(output, arm)
            result, generations = audit(out, arm, plans[arm], manifest)
            require(result is not None, f"Awaiting {arm}")
            verify_result(result, held[arm], check_targets=True)
            results[arm] = result
            gpu_metrics = telemetry(out)
            integrity = {"new_final_checkpoint_hash_verified": True, "generations": generations}
        else:
            result = results[arm]
            gpu_metrics = None
            integrity = {"historical_checkpoint_reference": True, "weight_bytes_read_in_comparison": False}
        identity, evaluation = result["identity"], result["evaluation"]
        metrics[arm] = {"bpb": evaluation["bpb"], "validation_predicted_targets": evaluation["predicted_tokens"],
                        "training_exposure": identity["forecast"], "checkpoint": result["checkpoint"],
                        "integrity": integrity, "gpu_telemetry": gpu_metrics, "runtime": identity["runtime"]}
    report = {"scope": "paired development documents, fixed source bytes, seed 17", "manifest_sha256": MANIFEST_SHA,
              "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": 10_000, "metrics": metrics, "paired": pairs(results)}
    from mltok.lib.bpe_progress import canonical
    identity = hashlib.sha256(canonical(report)).hexdigest()
    publish(output / f"development-comparison-{identity}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
