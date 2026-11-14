"""End-to-end mechanics on a tiny synthetic corpus (CPU)."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import base_cfg, prepare_data
from mltok.plugins.metric.basic import renyi_efficiency
from mltok.runner import Run, run


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    d = tmp_path_factory.mktemp("ws")
    cfg = base_cfg(d)
    prepare_data(cfg)
    return cfg


def test_end_to_end_outputs_and_provenance(workspace):
    win = run(workspace)
    root = Path(workspace["paths"]["runs"]) / "t"
    for f in ("report.md", "results.csv", "timings.json"):
        assert (root / f).exists()
    sel = json.loads((root / "best" / "SELECTION.json").read_text())
    assert sel["winner"] == win and sel["provenance"]["plugin_fingerprints"]["algorithm"]
    assert "Provenance" in (root / "report.md").read_text()
    for mp in (root / "variants").glob("*/metrics.json"):
        m = json.loads(mp.read_text())
        assert m["roundtrip_ok"] == 1.0 and m["unk_rate"] == 0.0, mp.parent.name
    for pp in (root / "variants").glob("*/proxy_seed*.json"):
        assert 0 < json.loads(pp.read_text())["bpb"] < 8


def test_stress_set_separates_lossless_from_lossy_tokenizers(workspace):
    """hf_bpe has no byte fallback -> must fail the stress set and be ineligible; our BPE passes."""
    root = Path(workspace["paths"]["runs"]) / "t"
    ms = {p.parent.name.split("-")[0]: json.loads(p.read_text()) for p in (root / "variants").glob("*/metrics.json")}
    assert ms["mlbpe"]["stress_roundtrip_ok"] == 1.0
    assert ms["hf_bpe"]["stress_roundtrip_ok"] < 1.0
    sel = json.loads((root / "best" / "SELECTION.json").read_text())
    assert sel["winner"].split("-")[0] != "hf_bpe" and "ELIGIBLE" in sel["basis"]


def test_resume_skips_finished_work(workspace):
    root = Path(workspace["paths"]["runs"]) / "t"
    stamp = {p: p.stat().st_mtime_ns for p in (root / "variants").glob("*/*.json")}
    run(workspace)
    assert stamp == {p: p.stat().st_mtime_ns for p in (root / "variants").glob("*/*.json")}


def test_parallel_workers_fan_out(workspace):
    """Two CPU 'GPUs' exercise the exact worker fan-out used for Kaggle's 2 GPUs."""
    cfg = copy.deepcopy(workspace)
    cfg["name"] = "t_par"
    cfg["proxy"].update(gpus=["cpu", "cpu"], micro_batch=4)
    assert run(cfg)
    root = Path(cfg["paths"]["runs"]) / "t_par"
    assert (root / "proxy_worker_gpucpu.log").exists()
    assert len(list((root / "variants").glob("*/proxy_seed*.json"))) >= 2


def test_micro_batching_keeps_loss_scale(workspace):
    outs = []
    for mb in (0, 2):
        cfg = copy.deepcopy(workspace)
        cfg["name"] = f"t_mb{mb}"
        cfg["proxy"].update(gpus=[], micro_batch=mb, top_k=1)
        cfg["search"]["factors"] = {}
        run(cfg)
        root = Path(cfg["paths"]["runs"]) / cfg["name"]
        outs.append(json.loads(next((root / "variants").glob("*/proxy_seed*.json")).read_text())["bpb"])
    assert abs(outs[0] - outs[1]) < 0.35 * outs[0]


def test_irrelevant_factors_do_not_duplicate_runs(workspace):
    cfg = copy.deepcopy(workspace)
    cfg["search"]["factors"] = {"algorithm": ["spm_unigram"], "pretokenizer": ["ws", "word"]}
    r = Run(cfg)
    r.stage_intrinsic()
    names = [n for n in r.variants if n.startswith("spm_unigram")]
    assert len(names) == 1  # SentencePiece ignores the pre-tokenizer -> one run, not two


def test_renyi_bounds():
    assert renyi_efficiency(np.ones(100), 100) == pytest.approx(1.0)
    assert renyi_efficiency(np.array([1000, 1, 1, 1]), 4) < 0.5


def test_multi_seed_proxy_reports_mean_and_spread(workspace):
    cfg = copy.deepcopy(workspace)
    cfg["name"] = "t_seeds"
    cfg["proxy"].update(seeds=[0, 1], top_k=1, gpus=[])
    cfg["search"]["factors"] = {}
    run(cfg)
    root = Path(cfg["paths"]["runs"]) / "t_seeds"
    files = sorted((root / "variants").glob("*/proxy_seed*.json"))
    assert {f.name for f in files} == {"proxy_seed0.json", "proxy_seed1.json"}
    bpbs = [json.loads(f.read_text())["bpb"] for f in files]
    p = json.loads((root / "best" / "SELECTION.json").read_text())["proxy"]
    assert p["n_seeds"] == 2 and abs(p["bpb"] - sum(bpbs) / 2) < 1e-9 and p["bpb_std"] > 0
    assert "seed" in (root / "report.md").read_text().lower()
    # rerunning with a third seed reuses the two finished ones
    stamp = {f: f.stat().st_mtime_ns for f in files}
    cfg["proxy"]["seeds"] = [0, 1, 2]
    run(cfg)
    assert all(f.stat().st_mtime_ns == stamp[f] for f in files)
    assert len(list((root / "variants").glob("*/proxy_seed*.json"))) == 3
