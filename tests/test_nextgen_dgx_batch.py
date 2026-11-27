"""Offline CPU unit tests for nextgen_pack, nextgen_dgx, 7-arm bootstrap, and notebook."""

import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mom_bpe import dgx_resources as gpu
from experiments.mom_bpe import nextgen_dgx as ng
from experiments.mom_bpe import nextgen_pack as pack
from experiments.mom_bpe import proxy_blocks, scale_data
from experiments.mom_bpe import document_proxy as driver
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.canonical_bpb import summarize_bpb

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks/mom_bpe_nextgen_dgx_batch.ipynb"


def _make_synthetic_corpus_and_models(root: Path):
    """Build a valid mini corpus (186 val docs + train docs) and train all 3 next-gen models."""
    registry.load_plugins()
    norm = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    norm_fp = registry.fingerprint("normalizer", {"type": "ml_conservative", "terminal_chillu": True})
    normalizer = norm.normalize

    base_texts = [
        "കേരളം ഇന്ത്യയിലെ മനോഹരമായ ഒരു സംസ്ഥാനമാണ്. പുതിയ വിദ്യാർത്ഥികൾ പഠിക്കുന്നു.",
        "കിളികൾ കേരളം എൻ്റ ൎക്കു് ക്\u200dഷ അക്ഷരം മലയാളം ഭാഷ.",
        "▁മലയാളം literal ▂ transport edges and mixed English 2026 numbers!",
        "പുതിയ ടോക്കണൈസർ പരീക്ഷണം വിജയകരമായി പൂർത്തിയായി. 漢字 fallback.",
    ]
    train_texts = [f"{t} രേഖ-{idx}." for idx, t in enumerate(base_texts * 8)]

    corpus_dir = root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    # Build synthetic train & validation splits matching scale_data.publish format.
    val_texts = [f"കേരളം രേഖ {i} എൻ്റ അക്ഷരം ▁" for i in range(186)]

    def make_rows(prefix, texts):
        rows = []
        for idx, text in enumerate(texts):
            canon = normalizer(text)
            raw_b = text.encode("utf-8")
            can_b = canon.encode("utf-8")
            rows.append({
                "id": f"mlwiki:{prefix}-{idx}",
                "group": f"mlwiki:{prefix}-{idx}",
                "src": "mlwiki_snapshot",
                "domain": "encyclopedic",
                "sha256": hashlib.sha256(raw_b).hexdigest(),
                "canonical_sha256": hashlib.sha256(can_b).hexdigest(),
                "raw_bytes": len(raw_b),
                "canonical_bytes": len(can_b),
                "text": text,
            })
        return rows

    train_rows = make_rows("train", train_texts)
    val_rows = make_rows("val", val_texts)
    train_bytes = sum(r["canonical_bytes"] for r in train_rows)
    recipe = {
        "scale": "A",
        "qualification": "WIKIMEDIA_CAPACITY_DIAGNOSTIC",
        "canonical_target": train_bytes,
        "scope": scale_data.scope(norm_fp),
    }
    scale_data.publish(
        corpus_dir,
        recipe,
        lambda split: train_rows if split == "train" else val_rows,
        normalizer,
    )

    models_dir = root / "models"
    specs = {
        "T8-LS-9999": (
            "leading_space_script_runs",
            {"type": "boundary_guarded_hybrid_v2", "guard_milli": 1000, "alphabet_coverage": 0.99, "require_full_vocabulary": False},
        ),
        "M48-Guard-9999": (
            "lossless_word_prefix",
            {"type": "boundary_guarded_hybrid_v2", "guard_milli": 1000, "alphabet_coverage": 0.99, "require_full_vocabulary": False},
        ),
        "ML-MinGram-WP-G1000": (
            "lossless_word_prefix",
            {
                "type": "ml_mingram",
                "guard_milli": 1000,
                "akshara_guard_milli": 1000,
                "alphabet_coverage": 0.99,
                "hybrid_min_count": 2,
                "require_full_vocabulary": False,
            },
        ),
    }
    train_jsonl = str(corpus_dir / "train.jsonl")
    for arm, (pre_name, algo_spec) in specs.items():
        arm_out = models_dir / arm
        arm_out.mkdir(parents=True, exist_ok=True)
        pre = registry.make("pretokenizer", pre_name)
        algo = registry.make("algorithm", algo_spec)
        algo.train(TrainContext(48000, DEFAULT_SPECIALS, norm, pre, arm_out, None, [train_jsonl], log=lambda *_: None))
    return corpus_dir / "manifest.json", models_dir, train_bytes


def test_nextgen_pack_creates_verified_binary_caches_and_manifest(tmp_path):
    corpus_manifest, models_dir, train_bytes = _make_synthetic_corpus_and_models(tmp_path)
    out_dir = tmp_path / "packed"

    # Full-vocab check rejects underfilled fixture fits cleanly.
    with pytest.raises(ValueError, match="occupied"):
        pack.pack_nextgen_inputs(
            models_dir=models_dir,
            corpus_manifest=corpus_manifest,
            out_dir=out_dir,
            baseline_inputs=None,
            require_full_vocab=True,
        )

    out_dir = tmp_path / "packed_ok"
    summary = pack.pack_nextgen_inputs(
        models_dir=models_dir,
        corpus_manifest=corpus_manifest,
        out_dir=out_dir,
        baseline_inputs=None,
        require_full_vocab=False,
        budget_overrides={"fixed_source_bytes": min(2048, train_bytes), "fixed_training_tokens": 500},
        update_targets=256,
        checkpoint_updates=1,
    )
    manifest_path = Path(summary["manifest_path"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["schema"] == "mom-nextgen-inputs-v1"
    assert manifest["parameters"] == 38_799_936
    assert set(manifest["arms"]) == set(ng.NEXTGEN_ARMS)
    assert tuple(manifest["arms_order"]) == ng.NEXTGEN_ARMS
    assert manifest["validation"]["documents"] == 186

    # Verify every binary cache & receipt hash matches and EncodedSplit decodes cleanly.
    for rel, ref in manifest["files"].items():
        p = out_dir / rel
        assert p.is_file()
        assert p.stat().st_size == ref["size"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == ref["sha256"]

    for arm in ng.NEXTGEN_ARMS:
        cache_dir = out_dir / manifest["arms"][arm]["cache"]
        cache_data, cache_sha = proxy_blocks.receipt(cache_dir / "cache.json")
        assert cache_sha == manifest["arms"][arm]["cache_receipt_sha256"]
        val_split = proxy_blocks.EncodedSplit(cache_dir, "validation", cache_data["splits"]["validation"])
        assert len(val_split.docs) == 186
        assert val_split.canonical_bytes == manifest["validation"]["canonical_bytes"]
        assert val_split.endpoint(0, val_split.docs[0]["length"]) == (val_split.docs[0]["canonical_bytes"], True)

        # Unmodified driver plan matches manifest plan_sha256 when next-gen arm names are admitted.
        with pack.with_nextgen_arms(driver, proxy_blocks):
            for mode in ng.MODES:
                mode_ref = manifest["arms"][arm]["modes"][mode]
                replan = driver.fit_cache(SimpleNamespace(
                    command="plan",
                    cache=cache_dir,
                    mode=mode,
                    budget=mode_ref["budget"],
                    out=None,
                    seed=None,
                    update_targets=256,
                    microbatch_blocks=None,
                    checkpoint_updates=1,
                ))
                assert replan["forecast"] == mode_ref["forecast"]


def test_nextgen_dgx_gpu7_blocked_and_xet_disabled_and_manifest_check(tmp_path, monkeypatch):
    # Xet disabled and hf_transfer disabled enforced at import and by helper
    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"
    assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "0"
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    ng._force_no_xet()
    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"
    assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "0"

    # Physical GPU 7 must always be rejected
    uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    monkeypatch.setattr(gpu, "_query", lambda *args: [("7", uuid, "40000", "0")])
    with pytest.raises(ValueError, match="GPU 7"):
        gpu.GpuAdmission().admit(uuid)

    # Corrupt inputs manifest SHA must fail
    bad_manifest = tmp_path / "inputs.json"
    bad_manifest.write_text(json.dumps({"schema": "wrong-schema"}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        ng.load_inputs_manifest(bad_manifest, expected_sha="0" * 64)
    with pytest.raises(ValueError, match="Invalid nextgen-inputs-manifest schema"):
        ng.load_inputs_manifest(bad_manifest)


def _write_mock_result(folder: Path, *, arm: str, mode: str, seed: int, impl: dict, held_docs: list[dict], doc_nlls: list[float], forecast: dict, extra_checkpoint: bool = True):
    folder.mkdir(parents=True, exist_ok=True)
    per_doc = [
        {**doc, **summarize_bpb([nll], doc["canonical_bytes"], doc["length"] + 1)}
        for doc, nll in zip(held_docs, doc_nlls, strict=True)
    ]
    eval_summary = summarize_bpb(
        (d["nll_nats"] for d in per_doc),
        sum(d["canonical_bytes"] for d in held_docs),
        sum(d["predicted_tokens"] for d in per_doc),
    )
    identity = {
        "arm": arm,
        "mode": mode,
        "budget": 50_000_000 if mode == ng.MODES[0] else 5_000_000,
        "config": {"seed": seed, "checkpoint_updates": 1, "model": {"type": "gpt", "n_layer": 7, "n_head": 7, "d_model": 448}},
        "forecast": forecast,
        "validation": {},
        "manifest_sha256": "m" * 64,
        "tokenizer_sha256": "t" * 64,
        "optimizer_sha256": "o" * 64,
        "config_sha256": "c" * 64,
        "parameters": 38_799_936,
        "implementation": impl,
        "model_plugin": "p" * 64,
        "runtime": {"system": "Linux"},
    }
    updates = forecast["optimizer_updates"]
    buf = io.BytesIO()
    torch.save({
        "identity": identity,
        "step": updates,
        "exposure": forecast,
        "train_nll_nats": 10.0,
        "model": {"w": torch.ones(1)},
        "optimizer": {"state": {0: {"step": torch.tensor(updates)}}, "param_groups": [{"lr": 0.0003}]},
        "scheduler": {"lr": 0.0003},
        "rng": dict.fromkeys(("python", "numpy", "torch", "cuda")),
    }, buf)
    ckpt_bytes = buf.getvalue()
    ckpt_sha = hashlib.sha256(ckpt_bytes).hexdigest()
    ckpt_name = f"step-{updates:06d}-{ckpt_sha}.pt"
    if extra_checkpoint:
        (folder / ckpt_name).write_bytes(ckpt_bytes)
    (folder / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    ng.write_receipt(folder / "result.json", {
        "identity": identity,
        "training_nll_nats": 10.0,
        "evaluation": {"per_document": per_doc, **eval_summary},
        "checkpoint": {"path": ckpt_name, "size": len(ckpt_bytes), "sha256": ckpt_sha},
    })
    return identity


def test_nextgen_batch_gates_resume_backup_and_7arm_bootstrap(tmp_path, monkeypatch):
    held = [{"id": str(i), "canonical_sha256": f"{i:064x}", "canonical_bytes": 100 + (i % 7), "length": 10} for i in range(186)]
    impl = {
        "driver": gpu.PINNED_DRIVER,
        "blocks": driver.code_digest(driver.blocks.__code__.co_filename),
        "scorer": driver.code_digest(summarize_bpb.__code__.co_filename),
    }

    arms_manifest = {}
    for arm in ng.NEXTGEN_ARMS:
        modes_dict = {}
        for mode in ng.MODES:
            forecast = {"optimizer_updates": 2, "predicted_tokens": 200, "canonical_bytes": 1000}
            plan_obj = {
                "arm": arm,
                "mode": mode,
                "budget": 50_000_000 if mode == ng.MODES[0] else 5_000_000,
                "forecast": forecast,
                "validation": {},
                "config": {"seed": 17, "checkpoint_updates": 1, "model": {"type": "gpt", "n_layer": 7, "n_head": 7, "d_model": 448}},
            }
            modes_dict[mode] = {
                "budget": plan_obj["budget"],
                "forecast": forecast,
                "plan_sha256": ng.metadata_digest(plan_obj),
                "plan_obj": plan_obj,
            }
        arms_manifest[arm] = {
            "arm": arm,
            "cache": f"dp/cache/{arm}",
            "tokenizer_sha256": "t" * 64,
            "fit": f"scale48k-nextgen-v1/fits/{arm}",
            "validation": {},
            "modes": modes_dict,
        }

    inputs_manifest = {
        "schema": "mom-nextgen-inputs-v1",
        "arms_order": list(ng.NEXTGEN_ARMS),
        "arms": arms_manifest,
        "corpus_manifest_sha256": "m" * 64,
        "optimizer_sha256": "o" * 64,
        "config_sha256": "c" * 64,
        "parameters": 38_799_936,
        "validation": {"canonical_stream_sha256": "v" * 64},
        "files": {},
    }
    root = tmp_path / "root"
    root.mkdir()
    queue = ng.build_nextgen_queue(inputs_manifest, "i" * 64)
    assert [j["id"] for j in queue["queue"]] == [
        "T8-LS-9999-fixed_source_bytes",
        "M48-Guard-9999-fixed_source_bytes",
        "ML-MinGram-WP-G1000-fixed_source_bytes",
        "T8-LS-9999-fixed_training_tokens",
        "M48-Guard-9999-fixed_training_tokens",
        "ML-MinGram-WP-G1000-fixed_training_tokens",
    ]

    batch = ng.NextGenBatch(
        root,
        queue,
        inputs_manifest,
        manifest_sha="fixture-sha",
        require_full_vocab=False,
        check_runtime=False,
        enforce_reference_pins=False,
    )
    batch.stage("batch-manifest", {"manifest": batch.manifest})
    plans = {
        f"{arm}-{mode}": arms_manifest[arm]["modes"][mode]["plan_obj"]
        for arm in ng.NEXTGEN_ARMS
        for mode in ng.MODES
    }
    batch.stage("verified-inputs", {"plans": plans})
    batch.stage("verified-plans", {"plans": plans})
    batch.stage("cpu-ready", {"ready": True})
    for old, new in (("PENDING", "INPUT_VERIFIED"), ("INPUT_VERIFIED", "PLAN_VERIFIED"), ("PLAN_VERIFIED", "READY")):
        batch.advance_preparation(old, new)

    monkeypatch.setattr(batch, "held", lambda _arm: held)
    monkeypatch.setattr(batch, "storage", lambda *args: {"free_bytes": 100 * ng.GIB, "additional_bytes_reserved": ng.GIB})
    monkeypatch.setattr(ng, "verify_nextgen_identity", lambda *args, **kwargs: None)

    # Synthetic per-doc NLLs per arm so 7-arm paired bootstrap deltas are exact & distinct.
    nll_map = {
        "M48": [10.0] * 186,
        "F-space": [10.5] * 186,
        "T5": [10.2] * 186,
        "T8": [10.1] * 186,
        "T8-LS-9999": [9.8] * 186,
        "M48-Guard-9999": [9.6] * 186,
        "ML-MinGram-WP-G1000": [9.4] * 186,
    }
    for base_arm in ng.BASELINE_ARMS:
        for mode in ng.MODES:
            _write_mock_result(
                batch.reference_dir / f"seed17/{base_arm}/{mode}",
                arm=base_arm,
                mode=mode,
                seed=17,
                impl=impl,
                held_docs=held,
                doc_nlls=nll_map[base_arm],
                forecast={"optimizer_updates": 2, "predicted_tokens": 200, "canonical_bytes": 1000},
                extra_checkpoint=False,
            )

    def mock_fit(job, plan, approval, log):
        _write_mock_result(
            root / job["output"],
            arm=job["arm"],
            mode=job["mode"],
            seed=17,
            impl=impl,
            held_docs=held,
            doc_nlls=nll_map[job["arm"]],
            forecast=job["forecast"],
        )

    approval_all = {"manifest_sha256": batch.sha, "scope": "all", "uuid": "fake", "shared": False}
    approval_first = {"manifest_sha256": batch.sha, "scope": "first", "uuid": "fake", "shared": False}

    # Pair 2 blocked before Pair 1 finishes and verifies
    with pytest.raises(ValueError, match="does not cover"):
        batch.execute("second", approval_first)
    with pytest.raises(ValueError, match="Verify all first-group"):
        batch.execute("second", approval_all)

    calls = []
    monkeypatch.setattr(batch, "fit", lambda job, plan, appr, log: (calls.append(job["id"]), mock_fit(job, plan, appr, log)))
    batch.execute("first", approval_all)
    assert calls == [
        "T8-LS-9999-fixed_source_bytes",
        "M48-Guard-9999-fixed_source_bytes",
        "ML-MinGram-WP-G1000-fixed_source_bytes",
    ]
    # Calling execute("first") again skips completed jobs safely
    batch.execute("first", approval_all)
    assert len(calls) == 3

    # Pair 2 still requires persisted first-pair verification receipt
    with pytest.raises(ValueError, match="Missing first-pair-verified"):
        batch.execute("second", approval_all)
    batch.result_check("first")

    batch.execute("second", approval_all)
    assert len(calls) == 6
    batch.result_check("all")

    # Test checkpoint fallback: earlier valid step-*.pt rescues truncated newest step-*.pt
    resume_dir = tmp_path / "resume_job"
    resume_plan = plans["T8-LS-9999-fixed_source_bytes"]
    valid_id = _write_mock_result(
        resume_dir,
        arm="T8-LS-9999",
        mode=ng.MODES[0],
        seed=17,
        impl=impl,
        held_docs=held,
        doc_nlls=nll_map["T8-LS-9999"],
        forecast={"optimizer_updates": 1, "predicted_tokens": 100, "canonical_bytes": 500},
    )
    (resume_dir / "result.json").unlink()
    # Rename step-000002? No, above wrote step-000001; write corrupt step-000002 with fake hash
    (resume_dir / f"step-000002-{'f' * 64}.pt").write_bytes(b"corrupted-checkpoint")
    res_none, gens = batch.audit_job(resume_dir, "T8-LS-9999", resume_plan, inspect_resume=True)
    assert res_none is None
    assert gens[f"step-000002-{'f' * 64}.pt"]["valid_bytes"] is False
    # If all generations are corrupt, fail loudly
    for valid_pt in resume_dir.glob("step-000001-*.pt"):
        valid_pt.write_bytes(b"also-corrupt")
    with pytest.raises(ValueError, match="No checksum-valid generation"):
        batch.audit_job(resume_dir, "T8-LS-9999", resume_plan, inspect_resume=True)

    # Full 7-arm analyze + bootstrap accuracy verification
    analysis = batch.analyze()
    assert (batch.dir / "analysis.json").is_file()
    assert analysis["arms"] == list(ng.ALL_SEVEN_ARMS)
    proto = analysis["protocols"][ng.MODES[0]]
    assert set(proto["metrics"]) == set(ng.ALL_SEVEN_ARMS)
    assert len(proto["paired"]) == 21  # C(7, 2)
    sizes = np.array([d["canonical_bytes"] for d in held], dtype=np.float64)
    expected_delta = ((9.4 - 10.0) * 186) / (sizes.sum() * math.log(2))
    pair_entry = proto["paired"]["ML-MinGram-WP-G1000 minus M48"]
    assert pair_entry["delta_bpb"] == pytest.approx(expected_delta, rel=1e-12)
    assert pair_entry["documents_favoring"] == {"M48": 0, "ML-MinGram-WP-G1000": 186}
    assert pair_entry["tied_documents"] == 0

    # Verify backup-hub rejects public repos and stays strictly under seed17/nextgen/
    class FakeHubApi:
        def __init__(self, private: bool):
            self.private = private
            self.sha = "a" * 40
            self.uploaded_paths = []
            self.stored = {}

        def repo_info(self, repo, repo_type="model", revision=None, timeout=20):
            return SimpleNamespace(private=self.private, sha=self.sha)

        def list_repo_files(self, repo, repo_type="model", revision=None):
            return list(self.stored)

        def create_commit(self, repo, repo_type="model", parent_commit=None, operations=None, commit_message=None):
            for op in operations:
                self.uploaded_paths.append(op.path_in_repo)
                self.stored[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
            self.sha = "b" * 40
            return SimpleNamespace(oid=self.sha)

        def hf_hub_download(self, repo, filename, repo_type="model", revision=None, local_dir=None, force_download=False):
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.stored[filename])
            return str(target)

    with pytest.raises(ValueError, match="private"):
        batch.backup_hub("owner/public-repo", "all", True, False, api=FakeHubApi(private=False))

    fake_api = FakeHubApi(private=True)
    hub_report = batch.backup_hub("Akzhai/mom-bpe-dgx-results-private", "all", True, False, api=fake_api)
    assert hub_report["hub_prefix"] == "seed17/nextgen/"
    assert all(p.startswith("seed17/nextgen/") for p in fake_api.uploaded_paths)
    assert "seed17/nextgen/analysis.json" in fake_api.uploaded_paths
    assert len(fake_api.uploaded_paths) == 13  # 6 jobs * 2 files + analysis.json


def test_nextgen_notebook_structure():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_sources = ["\n".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                    for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_sources) >= 10
    combined = "\n".join(code_sources)
    for src in code_sources:
        ast.parse(src)
    assert "nextgen_dgx.py" in combined
    assert "dgx_batch.py" not in combined
    for flag in (
        "smoke_approved = False",
        "allocated = False",
        "shared_approved = False",
        "enable_training = False",
        "backup_approved = False",
        "checkpoint_backup_approved = False",
    ):
        assert flag in combined
    assert 'HF_HUB_DISABLE_XET="1"' in combined
    assert 'HF_HUB_ENABLE_HF_TRANSFER="0"' in combined
