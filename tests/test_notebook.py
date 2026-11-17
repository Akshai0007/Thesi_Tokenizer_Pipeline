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


def test_notebook_checkpoint_settings_and_token_safety():
    first = CODE[0]
    for setting in ("SYNC = True", "HF_REPO =", "HF_SECRET_NAMES", "SYNC_DATA = False", "CKPT_PUSH_S"):
        assert setting in first, setting
    text = "\n".join(CODE)
    cli = (ROOT / "mltok" / "cli.py").read_text()
    for cmd in ("store-check", "restore", "push"):
        assert f'add("{cmd}"' in cli and cmd in text, cmd
    token_cell = next(c for c in CODE if "UserSecretsClient" in c and "HF_FOUND" in c)
    assert "_SECRETS.append(val)" in token_cell                       # masked in every log line
    for line in token_cell.splitlines():
        if "print(" in line:
            assert "val" not in line.replace("HF_FOUND", "") and 'environ["HF_TOKEN"]' not in line, line   # never printed
    assert "Akzhai" not in "".join(CODE[1:]), "no user-specific names outside the settings cell"


def test_freeze_gates_are_honest():
    text = "\n".join(CODE)
    assert "export-hf" in text                                     # tells the user how to convert an old spm-only run
    assert "has_json and mt(" in text                              # verify can no longer be silently skipped
    assert "runner-up" in text and "train_share" in text and "eligible variant" in text and "standard error" in text           # winner-vs-runner-up gap + data-mix guard
    assert "custom BPE did NOT win" in text                        # inconvenient results are printed, not hidden


def test_one_shot_mode_is_available_but_off_by_default():
    first = CODE[0]
    assert "ONE_SHOT = False" in first and "EXTRAS_START_BEFORE_H" in first and "T0 = time.time()" in first
    assert "if ONE_SHOT:\n    ROUND = 5" in first                 # one Run All = rounds 1..5
    text = "\n".join(CODE)
    assert "session time used" in text                              # summary cell
    assert "ONE_SHOT and used_h > EXTRAS_START_BEFORE_H" in text    # extras are skipped when time is short


def test_dry_run_gpu_gate_only_judges_jobs_that_actually_ran():
    text = "\n".join(CODE)
    assert "os.path.getmtime(f) >= T0" in text and "restored from the store" in text
