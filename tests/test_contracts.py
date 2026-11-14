"""Every registered plugin must satisfy its contract; every starter template must be a valid plugin."""
import pytest

from mltok import registry
from mltok.contract_checks import check_plugin, run_checks
from mltok.templates import T, render

ALL = [(k, n) for k in registry.KINDS for n in registry.available(k)]


@pytest.mark.parametrize("kind,name", ALL, ids=[f"{k}/{n}" for k, n in ALL])
def test_builtin_plugin_satisfies_contract(kind, name):
    res = check_plugin(kind, name)
    fails = [m for s, m in res if s == "FAIL"]
    assert not fails, fails


def test_all_kinds_have_at_least_one_plugin():
    assert all(registry.available(k) for k in registry.KINDS)


@pytest.mark.parametrize("kind", sorted(T))
def test_starter_template_passes_check(kind, tmp_path, isolated_registry):
    f = tmp_path / f"my_{kind}.py"
    f.write_text(render(kind, f"my_{kind}"))
    assert run_checks(file=str(f), log=lambda *_: None)
