"""GPU tests: skipped automatically when CUDA is not available (e.g. in CI / sandbox)."""
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def test_selftest_on_all_gpus():
    from mltok.selftest import run_selftest

    assert run_selftest()


def test_amp_dtype_follows_compute_capability():
    from mltok.selftest import _check_device

    assert _check_device("cuda:0")
