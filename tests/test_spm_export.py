"""SentencePiece winners must be usable by training stacks: exported to Hugging Face format with checked parity."""
import json
from pathlib import Path

import pytest

from conftest import base_cfg, prepare_data
from mltok import registry
from mltok.cli import main
from mltok.lib.spm_export import EDGE_TEXTS, export_hf, parity, spm_to_hf
from mltok.runner import Run
from mltok.verify import verify_dir

quiet = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def spm_variant(tmp_path_factory):
    d = tmp_path_factory.mktemp("spm")
    cfg = base_cfg(d)
    cfg["components"] = {"normalizer": "ml", "pretokenizer": "word", "algorithm": "spm_unigram", "vocab": 1500}
    cfg["search"]["factors"] = {}
    cfg["proxy"]["enabled"] = False
    prepare_data(cfg)
    r = Run(cfg)
    be, info = r.ensure_tokenizer(r.base)
    return r, r.vdir(r.base), info


def test_spm_unigram_training_also_exports_tokenizer_json(spm_variant):
    r, vdir, info = spm_variant
    assert (vdir / "spm.model").exists() and (vdir / "tokenizer.json").exists()
    assert info["hf_export"]["ok"] and info["hf_export"]["real_mismatches"] == 0


def test_exported_tokenizer_passes_the_real_verify(spm_variant):
    _, vdir, _ = spm_variant
    assert verify_dir(str(vdir), normalizer="ml", log=quiet)        # lossless, byte fallback, AutoTokenizer, chat template


def test_hf_and_sentencepiece_agree_on_ids_or_exact_score_ties(spm_variant):
    from mltok.lib.synth import synthetic_docs

    r, vdir, _ = spm_variant
    norm = registry.make("normalizer", "ml")
    tok = spm_to_hf(vdir / "spm.model", norm)
    texts = list(synthetic_docs(Path(__file__).resolve().parents[1] / "data" / "seed_ml.txt", n_docs=300, n_word_types=3000, seed=5)) + EDGE_TEXTS
    rep = parity(vdir / "spm.model", tok, norm, texts)
    assert rep["real_mismatches"] == 0 and rep["equivalent_rate"] == 1.0, rep["examples"]
    assert rep["identical_rate"] > 0.99          # ties are rare; they are equally probable segmentations


def test_export_hf_cli_converts_an_existing_spm_only_folder(spm_variant, tmp_path, capsys):
    _, vdir, _ = spm_variant
    old = tmp_path / "old_best"
    old.mkdir()
    (old / "spm.model").write_bytes((vdir / "spm.model").read_bytes())
    assert not verify_dir(str(old), log=quiet)                       # spm-only folder is not usable yet ...
    with pytest.raises(SystemExit) as e:
        main(["export-hf", "--spm", str(old / "spm.model"), "--normalizer", "ml"])
    assert e.value.code == 0 and (old / "tokenizer.json").exists()
    assert verify_dir(str(old), normalizer="ml", log=quiet)          # ... and is after the conversion


def test_verify_explains_what_to_do_for_spm_only_folders(spm_variant, tmp_path):
    _, vdir, _ = spm_variant
    old = tmp_path / "x"
    old.mkdir()
    (old / "spm.model").write_bytes((vdir / "spm.model").read_bytes())
    msgs = []
    assert not verify_dir(str(old), log=msgs.append)
    assert any("export-hf" in m for m in msgs)


def test_converter_rejects_non_unigram_models(tmp_path):
    import sentencepiece as spm

    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(["ഇത് ഒരു പരീക്ഷണം ആണ്"] * 300 + ["കേരളം ഇന്ത്യയുടെ സംസ്ഥാനം"] * 300), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "b"), model_type="bpe", vocab_size=80,
                                   normalization_rule_name="identity", minloglevel=2)
    with pytest.raises(ValueError, match="UNIGRAM"):
        spm_to_hf(tmp_path / "b.model", registry.make("normalizer", "ml"))
