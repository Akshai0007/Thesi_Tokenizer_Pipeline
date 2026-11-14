import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mltok import registry  # noqa: E402

registry.load_plugins()


@pytest.fixture
def isolated_registry():
    """Plugins loaded/overridden inside a test must not leak into other tests."""
    snap = {k: dict(v) for k, v in registry._REG.items()}
    loaded, errs, ovr = set(registry._LOADED), list(registry._ERRORS), list(registry._OVERRIDES)
    mods = set(sys.modules)
    yield
    for k in registry._REG:
        registry._REG[k].clear()
        registry._REG[k].update(snap[k])
    registry._LOADED.clear(); registry._LOADED.update(loaded)
    registry._ERRORS[:] = errs; registry._OVERRIDES[:] = ovr
    for m in set(sys.modules) - mods:
        if m.startswith("mltok_plugin_"):
            sys.modules.pop(m, None)


def base_cfg(tmp: Path, name="t", **over) -> dict:
    cfg = {
        "name": name,
        "paths": {"raw": str(tmp / "raw"), "clean": str(tmp / "clean"), "runs": str(tmp / "runs")},
        "sources": [{"type": "synthetic", "id": "syn", "n_docs": 3000, "n_word_types": 8000}],
        "filters": [{"type": "length", "min_chars": 20}, {"type": "ml_ratio", "min_ratio": 0.3}],
        "split": {"eval_permille": 80, "shard_mb": 1},
        "components": {"normalizer": "ml", "pretokenizer": "word", "algorithm": "mlbpe", "vocab": 1600},
        "resources": {"workers": 1, "spm_max_mb": 3},
        "search": {"eval_docs": 200, "combine": True, "factors": {"algorithm": ["mlbpe_plain", "hf_bpe"]}},
        "proxy": {"enabled": True, "top_k": 2, "model": {"type": "gpt", "size": "nano"}, "seq_len": 64, "batch": 8,
                  "train_mb": 0.15, "eval_mb": 0.04, "lr": 0.003, "gpus": []},
        "baselines": [],
    }
    cfg.update(over)
    return cfg


def prepare_data(cfg: dict) -> None:
    from mltok.ingest import clean_corpus, ingest_sources
    from mltok.lib.io import list_shards

    registry.load_plugins(cfg.get("plugins_dir", []))
    ingest_sources(cfg["sources"], cfg["paths"]["raw"], log=lambda *_: None)
    clean_corpus(list_shards(cfg["paths"]["raw"]), cfg["paths"]["clean"], cfg["filters"], log=lambda *_: None, **cfg["split"])
