"""The plug-in promise: change ONE file (or one config entry) and the rest of the pipeline is untouched."""
import ast
import json
import time
from pathlib import Path

import pytest

from conftest import base_cfg, prepare_data
from mltok import registry
from mltok.contract_checks import run_checks
from mltok.runner import Run, run

ROOT = Path(__file__).resolve().parents[1]
quiet = lambda *_: None  # noqa: E731

MY_SOURCE = '''
from mltok.contracts import Source
from mltok.lib.synth import synthetic_docs
from mltok.registry import register

@register("source", "my_corpus")
class MyCorpus(Source):
    """A brand-new dataset: one file, nothing else in the repo changes."""
    example_params = {}
    def __init__(self, id="mine", n=2500, **p):
        super().__init__(id=id, **p); self.n = n
    def iter_docs(self):
        for t in synthetic_docs(__import__("pathlib").Path(%r), n_docs=self.n, n_word_types=6000, seed=7):
            yield {"text": t, "src": self.id}
''' % str(ROOT / "data" / "seed_ml.txt")

NORM = '''
import unicodedata
from tokenizers import normalizers
from mltok.contracts import Normalizer
from mltok.registry import register

@register("normalizer", "mynorm")
class MyNorm(Normalizer):
    """%s"""
    def normalize(self, text):
        return unicodedata.normalize("NFC", text)%s
    def hf(self):
        return normalizers.Sequence([normalizers.NFC()%s])
'''


def norm_src(variant):  # variant 2 additionally lower-cases -> different behaviour AND different file hash
    if variant == 1:
        return NORM % ("v1", "", "")
    return NORM % ("v2", ".lower()", ", normalizers.Lowercase()")


def test_new_dataset_is_just_a_new_file(tmp_path, isolated_registry):
    plug = tmp_path / "plugins"
    plug.mkdir()
    (plug / "my_corpus.py").write_text(MY_SOURCE)
    assert run_checks(file=str(plug / "my_corpus.py"), log=quiet)  # fits the contract before any run

    cfg = base_cfg(tmp_path, plugins_dir=[str(plug)], sources=[{"type": "my_corpus", "id": "mine", "n": 2500}])
    cfg["search"]["factors"] = {}
    cfg["proxy"]["enabled"] = False
    prepare_data(cfg)
    assert run(cfg)
    stats = json.loads((tmp_path / "clean" / "clean_stats.json").read_text())
    assert stats["train_docs_mine"] > 1000  # the new dataset flowed through untouched engine code
    assert (tmp_path / "runs" / "t" / "best" / "tokenizer.json").exists()


def test_replace_builtin_by_name_changes_fingerprint_and_retrains(tmp_path, isolated_registry):
    cfg = base_cfg(tmp_path)
    cfg["search"]["factors"] = {}
    cfg["proxy"]["enabled"] = False
    prepare_data(cfg)
    fp_builtin = registry.fingerprint("normalizer", "ml")
    run(cfg)
    n_dirs_1 = len(list((tmp_path / "runs" / "t" / "variants").iterdir()))

    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "ml.py").write_text(NORM.replace("mynorm", "ml") % ("replacement", "", ""))
    cfg["plugins_dir"] = [str(plug)]
    run(cfg)
    assert registry.fingerprint("normalizer", "ml") != fp_builtin
    assert any("normalizer/ml" in o for o in registry.overrides())
    assert len(list((tmp_path / "runs" / "t" / "variants").iterdir())) == n_dirs_1 + 1  # retrained, old kept


def test_cache_invalidation_is_surgical(tmp_path, isolated_registry):
    """Edit one normalizer file: only variants using it recompute; the others are reused."""
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    for d, n in ((v1, 1), (v2, 2)):
        d.mkdir()
        (d / "mynorm.py").write_text(norm_src(n))
    cfg = base_cfg(tmp_path, plugins_dir=[str(v1)])
    cfg["search"] = {"eval_docs": 200, "combine": False, "factors": {"normalizer": ["none", "mynorm"]}}
    cfg["proxy"]["enabled"] = False
    prepare_data(cfg)
    run(cfg)
    vroot = tmp_path / "runs" / "t" / "variants"
    ok1 = {p.parent.name: p.stat().st_mtime_ns for p in vroot.glob("*/tokenizer.ok")}
    time.sleep(0.05)

    cfg["plugins_dir"] = [str(v2)]  # "replace the file"
    run(cfg)
    ok2 = {p.parent.name: p.stat().st_mtime_ns for p in vroot.glob("*/tokenizer.ok")}
    untouched = [k for k in ok1 if ok1[k] == ok2.get(k)]
    new = [k for k in ok2 if k not in ok1]
    assert any(k.startswith("mlbpe-ml-word") for k in untouched)      # base (ml) variant reused
    assert any(k.startswith("mlbpe-none-word") for k in untouched)    # 'none' variant reused
    assert len(new) == 1 and new[0].startswith("mlbpe-mynorm-word")   # only the edited plugin's variant retrained


def test_metric_swap_recomputes_metrics_but_reuses_tokenizers(tmp_path, isolated_registry):
    cfg = base_cfg(tmp_path)
    cfg["search"]["factors"] = {}
    cfg["proxy"]["enabled"] = False
    prepare_data(cfg)
    run(cfg)
    vroot = tmp_path / "runs" / "t" / "variants"
    tok_t = {p.parent.name: p.stat().st_mtime_ns for p in vroot.glob("*/tokenizer.ok")}

    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "longest.py").write_text('''
from mltok.contracts import Metric
from mltok.registry import register

@register("metric", "longest")
class Longest(Metric):
    def compute(self, backend, docs):
        return {"longest_doc_tokens": max(len(x) for x in backend.encode_batch([t for t, _ in docs]))}
''')
    cfg["plugins_dir"] = [str(plug)]
    cfg["metrics"] = ["basic", "structure", "roundtrip", "longest"]
    run(cfg)
    assert tok_t == {p.parent.name: p.stat().st_mtime_ns for p in vroot.glob("*/tokenizer.ok")}  # not retrained
    m = json.loads(next(vroot.glob("*/metrics.json")).read_text())
    assert "longest_doc_tokens" in m


def test_replacing_dataset_invalidates_everything_downstream(tmp_path, isolated_registry):
    cfg = base_cfg(tmp_path)
    cfg["search"]["factors"] = {}
    cfg["proxy"]["enabled"] = False
    prepare_data(cfg)
    r1 = Run(cfg)
    k1 = r1.tok_key(r1.base)
    cfg2 = base_cfg(tmp_path / "other", sources=[{"type": "synthetic", "id": "syn", "n_docs": 2000, "n_word_types": 5000, "seed": 5}])
    prepare_data(cfg2)
    r2 = Run(cfg2)
    assert r2.tok_key(r2.base) != k1 and r2.data_fp != r1.data_fp


def test_engine_files_never_import_plugins():
    for f in ("runner.py", "ingest.py", "proxy_train.py", "cli.py", "contracts.py", "registry.py", "gpu.py", "verify.py"):
        tree = ast.parse((ROOT / "mltok" / f).read_text())
        for node in ast.walk(tree):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                   [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(m.startswith("mltok.plugins") for m in mods), (f, mods)


def test_lib_never_imports_plugins():
    for p in (ROOT / "mltok" / "lib").glob("*.py"):
        assert "mltok.plugins" not in p.read_text() and "from ..plugins" not in p.read_text(), p.name


def test_broken_plugin_is_reported_not_fatal(tmp_path, isolated_registry):
    plug = tmp_path / "p"
    plug.mkdir()
    (plug / "broken.py").write_text("this is not python !!!")
    registry.load_plugins([plug])
    assert any("broken.py" in p for p, _ in registry.load_errors())
    assert "ml" in registry.available("normalizer")  # everything else still works


def test_check_rejects_nonconforming_plugins(tmp_path, isolated_registry):
    bad_equiv = tmp_path / "bad_equiv.py"  # python and hf versions disagree
    bad_equiv.write_text('''
from tokenizers import normalizers
from mltok.contracts import Normalizer
from mltok.registry import register
@register("normalizer", "bad_equiv")
class Bad(Normalizer):
    def normalize(self, text): return text.lower()
    def hf(self): return normalizers.Sequence([])
''')
    assert not run_checks(file=str(bad_equiv), log=quiet)

    coupled = tmp_path / "coupled.py"  # imports another plugin -> breaks independence
    coupled.write_text('''
from mltok.contracts import DocFilter
from mltok.plugins.filter.length import LengthFilter
from mltok.registry import register
@register("filter", "coupled")
class Coupled(DocFilter):
    def keep(self, text): return LengthFilter().keep(text)
''')
    assert not run_checks(file=str(coupled), log=quiet)

    unimpl = tmp_path / "unimpl.py"
    unimpl.write_text('''
from mltok.contracts import Source
from mltok.registry import register
@register("source", "unimpl")
class U(Source): pass
''')
    assert not run_checks(file=str(unimpl), log=quiet)


def test_unknown_plugin_name_gives_helpful_error():
    with pytest.raises(KeyError) as e:
        registry.make("normalizer", "does_not_exist")
    assert "Available" in str(e.value) and "ml" in str(e.value)
