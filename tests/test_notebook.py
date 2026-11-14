"""The Kaggle notebook is code too: it must stay valid and in sync with the CLI and configs."""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / "notebooks" / "kaggle_runner.ipynb").read_text())
CODE = [("".join(c["source"])) for c in NB["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_and_every_code_cell_compiles():
    assert NB["nbformat"] == 4 and CODE
    for i, src in enumerate(CODE):
        compile(src, f"cell{i}", "exec")
        assert not re.search(r"^\s*[!%]", src, re.M), "IPython magics make cells untestable outside Kaggle"


def test_notebook_only_uses_existing_cli_commands_and_configs():
    from mltok.cli import main  # noqa: F401  (import check)

    text = "\n".join(CODE)
    cli = (ROOT / "mltok" / "cli.py").read_text()
    for cmd in set(re.findall(r'mt\(f?"(\w[\w-]*)', text)):
        assert f'add("{cmd}"' in cli or f'"{cmd}"' in cli, cmd
    for cfg in re.findall(r'"(configs/[\w.]+\.yaml)"', text):
        assert (ROOT / cfg).exists(), cfg


def test_notebook_variations_reference_real_config_keys():
    cfg = yaml.safe_load((ROOT / "configs" / "kaggle_small.yaml").read_text())
    text = "\n".join(CODE)
    assert "proxy.seeds" in text and "seeds" in cfg["proxy"]
    for key in ("proxy.vocab_sweep", "proxy.train_mb", "proxy.top_k", "search.factors"):
        top, sub = key.split(".")
        assert sub in cfg[top], key


def test_rounds_are_gated_and_default_is_safe():
    first = CODE[0]
    assert re.search(r"^ROUND = 1\b", first, re.M)          # Run All does the cheap round only
    assert "RUN_VARIATIONS = []" in first                   # no accidental long variations
    assert "MIN_GPUS = 2" in first
