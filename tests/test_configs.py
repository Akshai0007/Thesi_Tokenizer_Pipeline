"""Every shipped config must reference only plugins/params that exist - a typo here would waste hours of GPU time."""
from pathlib import Path

import pytest
import yaml

from mltok import registry

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))
SPLIT_KEYS = {"eval_permille", "shard_mb", "max_train_mb", "balance", "weights"}
RES_KEYS = {"workers", "threads", "max_word_types", "spm_max_mb", "max_docs_per_shard"}
PROXY_KEYS = {"enabled", "top_k", "always_include", "model", "seq_len", "batch", "micro_batch", "gpus", "train_mb", "eval_mb", "epochs",
              "checkpoint_every_steps", "seeds", "seed", "lr", "vocab_sweep"}


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_config_references_only_existing_components(path):
    c = yaml.safe_load(path.read_text())
    spec = registry.spec_of
    for s in c["sources"]:
        registry.plugin_class("source", spec(s)["type"])
    for f in c["filters"]:
        registry.plugin_class("filter", spec(f)["type"])
    comp = c["components"]
    for kind, key in (("normalizer", "normalizer"), ("pretokenizer", "pretokenizer"), ("algorithm", "algorithm")):
        registry.make(kind, comp[key])                                     # constructs it: bad params fail here
    assert isinstance(comp["vocab"], int) and comp["vocab"] % 64 == 0, "vocab must be a multiple of 64"
    for f, vals in c["search"].get("factors", {}).items():
        assert f in ("normalizer", "pretokenizer", "algorithm")
        for v in vals:
            registry.make(f, v)
    for m in c.get("metrics", ["basic"]):
        registry.make("metric", m)
    registry.make("selector", c.get("selector", "default"))
    registry.make("store", c.get("store") or "none") if spec(c.get("store") or "none")["type"] != "hf_hub" else registry.plugin_class("store", "hf_hub")
    pm = c.get("proxy", {})
    if pm.get("model"):
        registry.make("proxy_model", pm["model"])


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_config_keys_are_known_and_consistent(path):
    c = yaml.safe_load(path.read_text())
    assert set(c.get("split", {})) <= SPLIT_KEYS, set(c.get("split", {})) - SPLIT_KEYS
    assert set(c.get("resources", {})) <= RES_KEYS, set(c.get("resources", {})) - RES_KEYS
    assert set(c.get("proxy", {})) <= PROXY_KEYS, set(c.get("proxy", {})) - PROXY_KEYS
    ids = {registry.spec_of(s).get("id", s["type"]) for s in c["sources"]}
    assert set(c.get("split", {}).get("weights", {})) <= ids, "split.weights names a source that does not exist"
    for v in c.get("proxy", {}).get("vocab_sweep", []):
        assert v % 64 == 0
    if c["paths"]["runs"].startswith("/data"):                             # DGX configs
        assert c.get("proxy", {}).get("gpus") != "auto", "NEVER auto on the DGX"
        if c.get("proxy", {}).get("enabled"):
            assert c["gpu"]["max_gb"] <= 60 and c["gpu"].get("guard")


def test_dgx_final_trains_the_decided_design_on_all_data():
    c = yaml.safe_load((ROOT / "configs" / "dgx_final.yaml").read_text())
    assert registry.spec_of(c["components"]["algorithm"])["type"] == "spm_unigram" and c["components"]["vocab"] == 48000
    assert c["split"]["max_train_mb"] == 0 and c["proxy"]["enabled"] is False and not c["search"]["factors"]


def test_dgx_confirm_brackets_the_decision_with_seeds():
    c = yaml.safe_load((ROOT / "configs" / "dgx_confirm.yaml").read_text())
    assert c["components"]["vocab"] == 48000 and 64000 in c["proxy"]["vocab_sweep"]
    assert len(c["proxy"]["seeds"]) >= 3 and "mlbpe" in c["search"]["factors"]["algorithm"]


def test_both_dgx_steps_share_one_bounded_raw_download():
    cf = yaml.safe_load((ROOT / "configs" / "dgx_confirm.yaml").read_text())
    fi = yaml.safe_load((ROOT / "configs" / "dgx_final.yaml").read_text())
    assert cf["paths"]["raw"] == fi["paths"]["raw"] and cf["sources"] == fi["sources"]     # one ingest serves both
    web = next(s for s in cf["sources"] if s["id"] == "fineweb2")
    assert 0 < web["max_docs"] <= 1_500_000          # the full 3.3M docs would be a ~3.5 h download nobody reads
    assert cf["paths"]["clean"] != fi["paths"]["clean"] and cf["name"] != fi["name"]       # safe to run side by side after ingest


def test_dgx_availability_check_rejects_reserved_and_unverifiable_gpu(monkeypatch):
    from mltok import gpu

    probed = []
    monkeypatch.setattr(gpu, "gpu_status", lambda index: probed.append(index))
    assert not gpu.wait_for_gpu(index=7)
    assert not probed
    assert not gpu.wait_for_gpu(index=0)
    assert probed == [0]
