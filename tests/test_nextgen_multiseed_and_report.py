"""Offline CPU tests for nextgen_multiseed.py and nextgen_report.py."""

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
from experiments.mom_bpe import nextgen_multiseed as ms
from experiments.mom_bpe import nextgen_report as rep
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.canonical_bpb import summarize_bpb


def _write_seed_result(
    root: Path,
    *,
    arm: str,
    mode: str,
    seed: int,
    nlls: list[float],
    docs_meta: list[dict],
    is_nextgen: bool = True,
):
    folder = root / (ms.MULTISEED_DIR if seed != 17 or not is_nextgen else f"{ng.BATCH_DIR}/jobs")
    if seed == 17 and is_nextgen:
        out = root / ng.BATCH_DIR / "jobs" / f"{arm}-{mode}"
    else:
        out = root / ms.MULTISEED_DIR / f"seed{seed}" / arm / mode
    out.mkdir(parents=True, exist_ok=True)
    per_doc = [
        {**d, **summarize_bpb([val], d["canonical_bytes"], d["length"] + 1)}
        for d, val in zip(docs_meta, nlls, strict=True)
    ]
    summary = summarize_bpb(
        (d["nll_nats"] for d in per_doc),
        sum(d["canonical_bytes"] for d in docs_meta),
        sum(d["predicted_tokens"] for d in per_doc),
    )
    identity = {
        "arm": arm,
        "mode": mode,
        "config": {"seed": seed, "model": {"type": "gpt", "n_layer": 7, "n_head": 7, "d_model": 448}},
        "forecast": {"optimizer_updates": 2, "predicted_tokens": 200},
        "validation": {},
        "parameters": 38_799_936,
    }
    buf = io.BytesIO()
    torch.save({"identity": identity, "step": 2, "model": {"w": torch.ones(1)}}, buf)
    ckpt_bytes = buf.getvalue()
    ckpt_sha = hashlib.sha256(ckpt_bytes).hexdigest()
    ckpt_name = f"step-000002-{ckpt_sha}.pt"
    (out / ckpt_name).write_bytes(ckpt_bytes)
    (out / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    ng.write_receipt(out / "result.json", {
        "identity": identity,
        "training_nll_nats": 5.0,
        "evaluation": {"per_document": per_doc, **summary},
        "checkpoint": {"path": ckpt_name, "size": len(ckpt_bytes), "sha256": ckpt_sha},
    })
    return summary["bpb"]


def test_multiseed_gpu7_block_xet_off_and_cache_reuse(tmp_path, monkeypatch):
    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"
    assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "0"

    # Physical GPU 7 blocked cleanly
    uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    monkeypatch.setattr(gpu, "_query", lambda *args: [("7", uuid, "40000", "0")])
    with pytest.raises(ValueError, match="--enable and --allocated"):
        ms.run_multiseed(
            root=tmp_path,
            source=tmp_path,
            arms=["ML-MinGram-WP-G1000"],
            seeds=[42],
            budgets=["fixed_source_bytes"],
            uuid=uuid,
            allocated=False,
            enabled=False,
        )
    with pytest.raises(ValueError, match="GPU 7"):
        ms.run_multiseed(
            root=tmp_path,
            source=tmp_path,
            arms=["ML-MinGram-WP-G1000"],
            seeds=[42],
            budgets=["fixed_source_bytes"],
            uuid=uuid,
            allocated=True,
            enabled=True,
        )

    # Seed validation
    assert ms.verify_seed_list([42, 101, 42]) == [42, 101]
    with pytest.raises(ValueError, match="Seed out of 32-bit"):
        ms.verify_seed_list([-1])

    # Pre-packed cache resolution (reuses existing cache.json without re-packing)
    root = tmp_path / "recovery"
    ng_cache = root / ng.INPUTS_DIR / "dp/cache/ML-MinGram-WP-G1000-abc"
    ng_cache.mkdir(parents=True)
    (ng_cache / "cache.json").write_text("{}", encoding="utf-8")
    t8_cache = root / "t5-t8-frozen-runs-v1/dp/cache/T8-xyz"
    t8_cache.mkdir(parents=True)
    (t8_cache / "cache.json").write_text("{}", encoding="utf-8")
    assert ms.resolve_arm_cache(root, "ML-MinGram-WP-G1000") == ng_cache
    assert ms.resolve_arm_cache(root, "T8", baseline_inputs=tmp_path / "absent.json") == t8_cache


def test_multiseed_aggregation_math_and_paired_bootstrap(tmp_path):
    docs_meta = [{"id": f"d-{i}", "canonical_sha256": f"{i:064x}", "canonical_bytes": 100 + (i % 5), "length": 8} for i in range(186)]
    root = tmp_path / "root"
    bpbs_mg = []
    for seed, nll_m48, nll_t8, nll_mg in (
        (17, 10.0, 10.2, 9.6),
        (42, 10.2, 10.3, 9.8),
        (101, 9.8, 10.1, 9.4),
    ):
        _write_seed_result(root, arm="M48", mode="fixed_source_bytes", seed=seed, nlls=[nll_m48] * 186, docs_meta=docs_meta, is_nextgen=False)
        _write_seed_result(root, arm="T8", mode="fixed_source_bytes", seed=seed, nlls=[nll_t8] * 186, docs_meta=docs_meta, is_nextgen=False)
        bpbs_mg.append(
            _write_seed_result(root, arm="ML-MinGram-WP-G1000", mode="fixed_source_bytes", seed=seed, nlls=[nll_mg] * 186, docs_meta=docs_meta, is_nextgen=True)
        )

    out_json = root / ms.MULTISEED_DIR / "aggregate.json"
    agg = ms.aggregate_multiseed(
        root=root,
        arms=["M48", "T8", "ML-MinGram-WP-G1000"],
        seeds=[17, 42, 101],
        budgets=["fixed_source_bytes"],
        out_path=out_json,
    )
    assert out_json.is_file()
    stats = agg["protocols"]["fixed_source_bytes"]["arms"]["ML-MinGram-WP-G1000"]
    assert stats["n_seeds"] == 3
    assert stats["seeds"] == [17, 42, 101]
    assert stats["mean_bpb"] == pytest.approx(float(np.mean(bpbs_mg)), rel=1e-12)
    assert stats["std_bpb"] == pytest.approx(float(np.std(bpbs_mg, ddof=1)), rel=1e-12)
    assert stats["min_bpb"] == pytest.approx(min(bpbs_mg), rel=1e-12)
    assert stats["max_bpb"] == pytest.approx(max(bpbs_mg), rel=1e-12)

    sizes = np.array([d["canonical_bytes"] for d in docs_meta], dtype=np.float64)
    expected_delta_vs_m48 = (-0.4 * 186) / (sizes.sum() * math.log(2))
    vs_m48 = agg["protocols"]["fixed_source_bytes"]["paired_vs_m48"]["ML-MinGram-WP-G1000 minus M48"]
    assert vs_m48["common_seeds"] == [17, 42, 101]
    assert vs_m48["delta_bpb"] == pytest.approx(expected_delta_vs_m48, rel=1e-12)
    lo, hi = vs_m48["document_bootstrap_95pct"]
    assert lo < vs_m48["delta_bpb"] < hi < 0.0

    vs_t8 = agg["protocols"]["fixed_source_bytes"]["paired_vs_t8"]["ML-MinGram-WP-G1000 minus T8"]
    assert vs_t8["delta_bpb"] < expected_delta_vs_m48


def test_score_heldout_zero_retrain_and_hub_backup_guards(tmp_path):
    registry.load_plugins()
    norm = registry.make("normalizer", {"type": "ml_conservative", "terminal_chillu": True})

    train_path = tmp_path / "mini_train.jsonl"
    train_docs = [
        "കേരളം ഇന്ത്യയിലെ സുന്ദരമായ സംസ്ഥാനം ആണ്. വിദ്യാർത്ഥികൾ പുതിയ അക്ഷരം പഠിക്കുന്നു.",
        "കിളികൾ കേരളം എൻ്റ ൎക്കു് ക്\u200dഷ അക്ഷരം ▁ മലയാളം 2026.",
    ] * 6
    train_path.write_text("\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in train_docs) + "\n", encoding="utf-8")

    fits = {}
    ckpts = {}
    model_spec = {"type": "gpt", "n_layer": 2, "n_head": 2, "d_model": 64}
    for arm, pre_name, algo_spec in (
        ("T8-LS-9999", "leading_space_script_runs", {"type": "boundary_guarded_hybrid_v2", "guard_milli": 1000, "alphabet_coverage": 0.99, "hybrid_min_count": 2, "require_full_vocabulary": False}),
        ("ML-MinGram-WP-G1000", "lossless_word_prefix", {"type": "ml_mingram", "guard_milli": 1000, "akshara_guard_milli": 1000, "alphabet_coverage": 0.99, "hybrid_min_count": 2, "require_full_vocabulary": False}),
    ):
        fit_dir = tmp_path / "fits" / arm
        fit_dir.mkdir(parents=True)
        pre = registry.make("pretokenizer", pre_name)
        algo = registry.make("algorithm", algo_spec)
        backend, _ = algo.train(TrainContext(48000, DEFAULT_SPECIALS, norm, pre, fit_dir, None, [str(train_path)], log=lambda *_: None))
        fits[arm] = fit_dir

        torch.manual_seed(17)
        model = registry.make("proxy_model", model_spec).build(48000, 128)
        ckpt_path = tmp_path / f"{arm}.pt"
        torch.save({
            "identity": {"arm": arm, "mode": "fixed_source_bytes", "config": {"seed": 17, "context": 128, "model": model_spec}},
            "model": model.state_dict(),
        }, ckpt_path)
        ckpts[arm] = ckpt_path

    # Score plain UTF-8 .txt held-out corpus with zero retraining
    heldout_txt = tmp_path / "flores_sample.txt"
    heldout_txt.write_text(
        "വിദ്യാർത്ഥികൾ പുതിയ അക്ഷരം പഠിക്കുന്നു.\nകിളികൾ കേരളം എൻ്റ ▁ ൎക്കു് മലയാളം.\n",
        encoding="utf-8",
    )
    out_report = tmp_path / "heldout/scores.json"
    res = ms.score_heldout_checkpoints(
        heldout_path=heldout_txt,
        arms=["T8-LS-9999", "ML-MinGram-WP-G1000"],
        fit_dirs=fits,
        checkpoint_paths=ckpts,
        device_str="cpu",
        out_path=out_report,
    )
    assert out_report.is_file()
    assert res["documents"] == 2
    assert set(res["arms"]) == {"T8-LS-9999", "ML-MinGram-WP-G1000"}
    assert "ML-MinGram-WP-G1000 minus T8-LS-9999" in res["paired"]
    for arm_rec in res["arms"].values():
        assert arm_rec["bpb"] > 0
        assert arm_rec["mom_surface"]["mom_canonical_equal"] == 2
        assert arm_rec["mom_surface"]["mom_akshara_breaks"] == 0

    # Test backup_multiseed_and_heldout_hub enforces seed_replication/ & heldout/ only + private repo
    root = tmp_path / "hub_root"
    docs_meta = [{"id": f"h-{i}", "canonical_sha256": f"{i:064x}", "canonical_bytes": 50, "length": 4} for i in range(4)]
    _write_seed_result(root, arm="ML-MinGram-WP-G1000", mode="fixed_source_bytes", seed=42, nlls=[5.0] * 4, docs_meta=docs_meta)
    (root / ms.MULTISEED_DIR / "aggregate.json").write_text("{}", encoding="utf-8")
    (root / ms.HELDOUT_DIR).mkdir(parents=True)
    (root / ms.HELDOUT_DIR / "flores.json").write_bytes(out_report.read_bytes())

    class FakeHubApi:
        def __init__(self, private: bool):
            self.private = private
            self.sha = "1" * 40
            self.uploaded = []
            self.store = {}

        def repo_info(self, repo, repo_type="model", revision=None, timeout=20):
            return SimpleNamespace(private=self.private, sha=self.sha)

        def list_repo_files(self, repo, repo_type="model", revision=None):
            return list(self.store)

        def create_commit(self, repo, repo_type="model", parent_commit=None, operations=None, commit_message=None):
            for op in operations:
                self.uploaded.append(op.path_in_repo)
                self.store[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
            self.sha = "2" * 40
            return SimpleNamespace(oid=self.sha)

        def hf_hub_download(self, repo, filename, repo_type="model", revision=None, local_dir=None, force_download=False):
            t = Path(local_dir) / filename
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(self.store[filename])
            return str(t)

    with pytest.raises(ValueError, match="private"):
        ms.backup_multiseed_and_heldout_hub(root=root, repo="owner/public", enabled=True, api=FakeHubApi(private=False))

    fake = FakeHubApi(private=True)
    bak = ms.backup_multiseed_and_heldout_hub(root=root, repo="Akzhai/mom-bpe-dgx-results-private", enabled=True, api=fake)
    assert bak["prefixes"] == ["heldout/", "seed_replication/"]
    assert all(p.startswith(("seed_replication/", "heldout/")) for p in fake.uploaded)
    assert not any(p.startswith("seed17/") for p in fake.uploaded)
    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"


def test_nextgen_report_generates_latex_tables_and_vector_pdf_png_figures(tmp_path):
    analysis = {
        "schema": "mom-nextgen-7arm-analysis-v1",
        "protocols": {
            "fixed_source_bytes": {
                "arms": list(rep.ALL_SEVEN_ARMS),
                "metrics": {
                    arm: {"bpb": 0.90 + 0.002 * idx, "canonical_bytes_per_prediction": 12.8 - 0.3 * idx}
                    for idx, arm in enumerate(rep.ALL_SEVEN_ARMS)
                },
                "paired": {
                    f"{arm} minus M48": {"delta_bpb": -0.005 + 0.001 * i, "document_bootstrap_95pct": [-0.009, -0.001]}
                    for i, arm in enumerate(rep.ALL_SEVEN_ARMS[1:])
                } | {
                    f"{arm} minus T8": {"delta_bpb": -0.008, "document_bootstrap_95pct": [-0.012, -0.004]}
                    for arm in ("T8-LS-9999", "M48-Guard-9999", "ML-MinGram-WP-G1000")
                },
            }
        },
        "nextgen_tokenizer_validation": {
            "T8-LS-9999": {"dev_targets_with_eos": 77119, "bytes_per_prediction": 12.9663, "pure_whitespace_tokens": 5508, "vocab_other_script": 316, "mom_surface": {"mom_akshara_breaks": 456, "mom_akshara_breaks_per_1000_reference_sites": 4.049, "mom_byte_fallback_ids": 141, "mom_canonical_equal": 186, "mom_docs": 186}},
            "M48-Guard-9999": {"dev_targets_with_eos": 77341, "bytes_per_prediction": 12.9291, "pure_whitespace_tokens": 3535, "vocab_other_script": 376, "mom_surface": {"mom_akshara_breaks": 456, "mom_akshara_breaks_per_1000_reference_sites": 4.049, "mom_byte_fallback_ids": 141, "mom_canonical_equal": 186, "mom_docs": 186}},
            "ML-MinGram-WP-G1000": {"dev_targets_with_eos": 77765, "bytes_per_prediction": 12.8586, "pure_whitespace_tokens": 4302, "vocab_other_script": 364, "mom_surface": {"mom_akshara_breaks": 269, "mom_akshara_breaks_per_1000_reference_sites": 2.389, "mom_byte_fallback_ids": 141, "mom_canonical_equal": 186, "mom_docs": 186}},
        },
    }
    multiseed = {
        "schema": "mom-nextgen-multiseed-aggregate-v1",
        "protocols": {
            "fixed_source_bytes": {
                "arms": {
                    "M48": {"n_seeds": 3, "mean_bpb": 0.9060, "std_bpb": 0.0012, "min_bpb": 0.9048, "max_bpb": 0.9072},
                    "ML-MinGram-WP-G1000": {"n_seeds": 3, "mean_bpb": 0.8990, "std_bpb": 0.0010, "min_bpb": 0.8980, "max_bpb": 0.9000},
                },
                "paired_vs_m48": {
                    "ML-MinGram-WP-G1000 minus M48": {"delta_bpb": -0.0070, "document_bootstrap_95pct": [-0.0110, -0.0030]}
                },
            }
        },
    }
    heldout = {
        "schema": "mom-nextgen-heldout-v1",
        "arms": {
            "ML-MinGram-WP-G1000": {"bpb": 0.9123, "bytes_per_prediction": 12.75, "mom_surface": {"mom_akshara_breaks": 12, "mom_akshara_breaks_per_1000_reference_sites": 1.25}}
        },
    }

    a_path = tmp_path / "analysis.json"
    m_path = tmp_path / "multiseed.json"
    h_path = tmp_path / "heldout.json"
    a_path.write_text(json.dumps(analysis), encoding="utf-8")
    m_path.write_text(json.dumps(multiseed), encoding="utf-8")
    h_path.write_text(json.dumps(heldout), encoding="utf-8")

    out_dir = tmp_path / "pub"
    res = rep.generate_report(out_dir=out_dir, analysis_path=a_path, multiseed_path=m_path, heldout_path=h_path)

    for tkey in ("intrinsic", "seed17_7arm", "multiseed", "heldout"):
        tfile = Path(res["tables"][tkey])
        assert tfile.is_file()
        tex = tfile.read_text(encoding="utf-8")
        assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
        assert "ML-MinGram-WP-G1000" in tex

    for fkey in ("pareto", "vocab", "forest"):
        pdf_p, png_p = [Path(p) for p in res["figures"][fkey]]
        assert pdf_p.is_file() and pdf_p.suffix == ".pdf"
        assert png_p.is_file() and png_p.suffix == ".png"
        pdf_bytes = pdf_p.read_bytes()
        png_bytes = png_p.read_bytes()
        assert pdf_bytes.startswith(b"%PDF-") and b"%%EOF" in pdf_bytes[-32:]
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n") and len(png_bytes) > 100
