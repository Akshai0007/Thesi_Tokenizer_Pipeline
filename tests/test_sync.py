"""Checkpointing: finished work is pushed, a fresh session restores it, crashes resume, store failures never kill a run."""
import copy
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import base_cfg, prepare_data
from mltok import registry
from mltok.contract_checks import fixture_docs, tiny_backend
from mltok.contracts import StoreAccessError
from mltok.plugins.store.hf_hub import HFHubStore
from mltok.runner import run
from mltok.sync import Sync

quiet = lambda *_: None  # noqa: E731


def store_cfg(tmp, name="t", **sync):
    cfg = base_cfg(tmp, name=name)
    cfg["search"]["factors"] = {}
    cfg["proxy"].update(top_k=1, checkpoint_every_steps=3)
    cfg["store"] = {"type": "local_dir", "path": str(tmp / "backup")}
    cfg["sync"] = {"min_interval_s": 0, "ckpt_push_interval_s": 0, **sync}
    return cfg


def test_finished_work_is_pushed_then_restored_without_retraining(tmp_path, capsys):
    cfg = store_cfg(tmp_path)
    prepare_data(cfg)
    win = run(cfg)
    bk = tmp_path / "backup" / "runs" / "t"
    assert list(bk.glob("variants/*/tokenizer.json")) and list(bk.glob("variants/*/metrics.json"))
    assert list(bk.glob("variants/*/proxy_seed0-*.json")) and (bk / "report.md").exists()
    assert (bk / "best" / "tokenizer.json").exists() and list(bk.glob("cache/words-*.pkl"))
    assert list(bk.glob("variants/*/proxy_seed0-*.ckpt")), "finished checkpoints are retained for verified recovery"

    shutil.rmtree(tmp_path / "runs")            # simulate a brand-new Kaggle machine
    capsys.readouterr()
    assert run(cfg) == win
    out = capsys.readouterr().out
    assert "restored" in out and "== training" not in out and "== proxy LM" not in out and "counting words" not in out


def test_cleaned_data_is_opt_in_and_restorable(tmp_path):
    cfg = store_cfg(tmp_path, include_data=False)
    prepare_data(cfg)
    sy = Sync(cfg, tmp_path / "runs" / "t", tmp_path / "clean", log=quiet)
    sy.push_tree(tmp_path / "clean")                       # include_data False -> nothing queued
    assert not (tmp_path / "backup" / "data").exists()
    cfg["sync"]["include_data"] = True
    sy = Sync(cfg, tmp_path / "runs" / "t", tmp_path / "clean", log=quiet)
    sy.push_tree(tmp_path / "clean")
    assert list((tmp_path / "backup" / "data" / "clean" / "train").glob("*.jsonl.gz"))
    shutil.rmtree(tmp_path / "clean")
    assert sy.restore("data") > 0 and list((tmp_path / "clean" / "train").glob("*.jsonl.gz"))


def test_a_failing_store_never_kills_the_run(tmp_path, isolated_registry, capsys):
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "broken_store.py").write_text('''
from mltok.contracts import Store
from mltok.registry import register
@register("store", "broken_store")
class Broken(Store):
    def push(self, files, message=""): raise ConnectionError("hub unreachable")
    def pull(self, remote_prefix, local_dir, skip_existing=True): raise ConnectionError("hub unreachable")
''')
    cfg = store_cfg(tmp_path)
    cfg["plugins_dir"] = [str(plug)]
    cfg["store"] = {"type": "broken_store"}
    prepare_data(cfg)
    assert run(cfg)                                          # completes despite every push failing
    out = capsys.readouterr().out
    assert "push failed" in out and "restore failed" in out
    assert (tmp_path / "runs" / "t" / "best" / "tokenizer.json").exists()


def test_proxy_resumes_from_checkpoint_after_a_crash(tmp_path, capsys):
    from mltok.proxy_train import train_proxy

    be, docs = tiny_backend(vocab=1200)
    tr, ev = docs[:1200], docs[1200:1300]
    model = registry.make("proxy_model", {"type": "gpt", "size": "nano"})
    kw = dict(seq_len=64, batch=8, train_mb=0.25, eval_mb=0.05, lr=3e-3, seed=0, log=quiet)
    clean = train_proxy(be, tr, ev, model, **kw)

    ck = tmp_path / "p.ckpt"
    saves = []

    def crash_on_second(path):
        saves.append(path)
        if len(saves) == 2:
            raise RuntimeError("simulated crash (session killed)")

    with pytest.raises(RuntimeError):
        train_proxy(be, tr, ev, model, ckpt_path=str(ck), ckpt_every=4, ckpt_key="k1", on_ckpt=crash_on_second, **kw)
    assert ck.exists()

    logs = []
    resumed = train_proxy(be, tr, ev, model, ckpt_path=str(ck), ckpt_every=4, ckpt_key="k1", **{**kw, "log": logs.append})
    assert any("resuming from checkpoint at step 8" in x for x in logs)
    assert abs(resumed["bpb"] - clean["bpb"]) < 0.02 * clean["bpb"]    # same result as an uninterrupted run
    assert ck.exists(), "training leaves checkpoint retention to the caller"


def test_corrupt_checkpoint_without_fallback_is_preserved(tmp_path):
    from mltok.proxy_train import train_proxy

    be, docs = tiny_backend(vocab=1200)
    model = registry.make("proxy_model", {"type": "gpt", "size": "nano"})
    ck = tmp_path / "p.ckpt"
    ck.write_bytes(b"not a checkpoint")
    logs = []
    with pytest.raises(RuntimeError, match="No usable proxy checkpoint"):
        train_proxy(be, docs[:1200], docs[1200:1300], model, seq_len=64, batch=8, train_mb=0.2, eval_mb=0.05,
                    ckpt_path=str(ck), ckpt_every=3, ckpt_key="k", log=logs.append)
    assert any("could not load checkpoint" in x for x in logs) and ck.read_bytes() == b"not a checkpoint"


def test_pushes_are_batched_and_throttled(tmp_path):
    class Counting(registry.plugin_class("store", "local_dir")):
        pushes = 0

        def push(self, files, message=""):
            Counting.pushes += 1
            super().push(files, message)

    cfg = {"store": {"type": "local_dir", "path": str(tmp_path / "bk")}, "sync": {"min_interval_s": 3600, "ckpt_push_interval_s": 3600}}
    sy = Sync(cfg, tmp_path / "runs" / "t", log=quiet)
    sy.store = Counting(path=str(tmp_path / "bk"))
    (tmp_path / "runs" / "t").mkdir(parents=True)
    for i in range(5):
        (tmp_path / "runs" / "t" / f"f{i}.json").write_text("x")
    sy.mark(tmp_path / "runs" / "t")
    sy.flush()                       # first push: allowed (interval counted from "never")
    sy.wait()
    assert Counting.pushes == 1 and len(list((tmp_path / "bk" / "runs" / "t").glob("*.json"))) == 5
    (tmp_path / "runs" / "t" / "late.json").write_text("y")
    sy.mark(tmp_path / "runs" / "t" / "late.json")
    sy.flush()                       # inside min_interval -> throttled
    sy.wait()
    assert Counting.pushes == 1
    sy.flush(force=True)             # stage end / error / exit -> immediate
    assert Counting.pushes == 2 and (tmp_path / "bk" / "runs" / "t" / "late.json").exists()


class FakeApi:
    """Records what the Hub store would do, using the real huggingface_hub method names."""

    def __init__(self, files=None, fail_first=0):
        self.files, self.calls, self.fail_first = dict(files or {}), [], fail_first
        self.revision = "0" * 40

    def repo_info(self, repo_id, **kw):
        return SimpleNamespace(private=True, sha=self.revision)

    def create_repo(self, repo_id, *, private=None, repo_type=None, exist_ok=False, **kw):
        self.calls.append(("create_repo", repo_id, private, repo_type, exist_ok))

    def create_commit(self, repo_id, operations, *, commit_message, repo_type=None, **kw):
        if self.fail_first:
            self.fail_first -= 1
            raise ConnectionError("transient")
        for op in operations:
            if type(op).__name__ == "CommitOperationAdd":
                self.files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
            else:
                self.files.pop(op.path_in_repo, None)
        self.calls.append(("create_commit", commit_message))
        self.revision = f"{int(self.revision, 16) + 1:040x}"
        return SimpleNamespace(oid=self.revision)

    def list_repo_files(self, repo_id, *, repo_type=None, **kw):
        return list(self.files)

    def hf_hub_download(self, repo_id, filename, *, repo_type=None, local_dir=None, **kw):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.files[filename])
        return str(p)

    def super_squash_history(self, repo_id, **kw):
        self.calls.append(("squash", repo_id))


def test_hf_store_is_private_prefixed_retrying_and_never_touches_other_files(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    api = FakeApi(files={"README.md": b"my model card", "weights.safetensors": b"precious"}, fail_first=1)
    monkeypatch.setattr(HFHubStore, "api_factory", lambda self=None: api, raising=False)
    st = HFHubStore(repo="Akzhai/mltok-malayalam-runs", prefix="mltok", retries=3)
    f = tmp_path / "a.json"
    f.write_text("{}")
    st.push([(f, "runs/t/variants/x/a.json")], "msg")                      # first create_commit fails once, then succeeds
    assert api.calls[0] == ("create_repo", "Akzhai/mltok-malayalam-runs", True, "model", True)   # private, exist_ok
    assert api.files["mltok/runs/t/variants/x/a.json"] == b"{}"
    assert api.files["README.md"] == b"my model card" and api.files["weights.safetensors"] == b"precious"  # untouched
    new_paths = set(api.files) - {"README.md", "weights.safetensors"}
    assert all(p.startswith("mltok/") for p in new_paths)
    st.push([(f, "runs/t/b.json")], "again")
    assert not any(call[0] == "squash" for call in api.calls)
    with pytest.raises(ValueError, match="squashing is disabled"):
        HFHubStore(repo="fixture/test", squash_every=2)

    out = tmp_path / "restored"
    assert st.pull("runs/t", out) == 2 and (out / "variants" / "x" / "a.json").read_text() == "{}"
    assert st.pull("runs/t", out) == 0                                       # nothing overwritten
    with pytest.raises(StoreAccessError, match="deletion is disabled"):
        st.delete(["runs/t/b.json"])
    assert "mltok/runs/t/b.json" in api.files and "README.md" in api.files


def test_hf_store_token_comes_from_env_only():
    import inspect

    src = inspect.getsource(HFHubStore)
    assert "os.environ" in src and "token=" in src
    assert "token" not in inspect.signature(HFHubStore.__init__).parameters  # cannot be put in a config file


def test_unchanged_files_are_not_pushed_twice(tmp_path):
    class Counting(registry.plugin_class("store", "local_dir")):
        sent = []

        def push(self, files, message=""):
            Counting.sent += [r for _, r in files]
            super().push(files, message)

    root = tmp_path / "runs" / "t"
    root.mkdir(parents=True)
    (root / "a.json").write_text("1")
    sy = Sync({"store": {"type": "local_dir", "path": str(tmp_path / "bk")}, "sync": {"min_interval_s": 0}}, root, log=quiet)
    sy.store = Counting(path=str(tmp_path / "bk"))
    sy.push_tree(root)
    sy.push_tree(root)                                   # nothing changed -> nothing sent
    assert Counting.sent == ["runs/t/a.json"]
    time.sleep(0.01)
    (root / "a.json").write_text("22")                   # changed -> sent again
    sy.push_tree(root)
    assert Counting.sent == ["runs/t/a.json", "runs/t/a.json"]


def test_pushes_run_in_the_background_and_never_block_the_caller(tmp_path):
    """Regression for the real run: a slow/failing Hub stalled training for ~100 s because pushes were synchronous."""
    import threading

    gate_open = threading.Event()

    class Slow(registry.plugin_class("store", "local_dir")):
        def push(self, files, message=""):
            gate_open.wait(10)                              # simulates a stuck upload / Hub 503 retries
            super().push(files, message)

    root = tmp_path / "runs" / "t"
    root.mkdir(parents=True)
    (root / "a.json").write_text("1")
    sy = Sync({"store": {"type": "local_dir", "path": str(tmp_path / "bk")}, "sync": {"min_interval_s": 0}}, root, log=quiet)
    sy.store = Slow(path=str(tmp_path / "bk"))
    sy.mark(root)
    t0 = time.time()
    sy.flush()                                              # returns immediately although the push is stuck
    assert time.time() - t0 < 1.0
    assert not (tmp_path / "bk" / "runs" / "t" / "a.json").exists()
    gate_open.set()
    sy.flush(force=True)                                    # force waits for the background push to finish
    assert (tmp_path / "bk" / "runs" / "t" / "a.json").exists()


def test_first_checkpoint_is_not_pushed_until_a_full_interval_has_passed(tmp_path):
    """Regression for the real run: every ~4-minute job uploaded a 283 MB checkpoint it never needed."""
    class Counting(registry.plugin_class("store", "local_dir")):
        sent = []

        def push(self, files, message=""):
            Counting.sent += [r for _, r in files]
            super().push(files, message)

    root = tmp_path / "runs" / "t"
    root.mkdir(parents=True)
    ck = root / "proxy_seed0.ckpt"
    ck.write_bytes(b"x" * 100)
    sy = Sync({"store": {"type": "local_dir", "path": str(tmp_path / "bk")},
               "sync": {"min_interval_s": 0, "ckpt_push_interval_s": 3600}}, root, log=quiet)
    sy.store = Counting(path=str(tmp_path / "bk"))
    sy.mark(ck)
    sy.flush()
    sy.wait()
    assert Counting.sent == []                              # short job: checkpoint never leaves the machine
    sy.flush(force=True)                                    # exit/crash: now it is worth saving
    assert Counting.sent == ["runs/t/proxy_seed0.ckpt"]


def test_file_modified_during_upload_stays_queued(tmp_path):
    import threading

    started, release = threading.Event(), threading.Event()

    class Hold(registry.plugin_class("store", "local_dir")):
        def push(self, files, message=""):
            started.set()
            release.wait(10)
            super().push(files, message)

    root = tmp_path / "runs" / "t"
    root.mkdir(parents=True)
    f = root / "a.json"
    f.write_text("v1")
    sy = Sync({"store": {"type": "local_dir", "path": str(tmp_path / "bk")}, "sync": {"min_interval_s": 0}}, root, log=quiet)
    sy.store = Hold(path=str(tmp_path / "bk"))
    sy.mark(f)
    sy.flush()
    assert started.wait(5)
    time.sleep(0.01)
    f.write_text("version-two")                             # changed while v1 is uploading
    sy.mark(f)
    release.set()
    sy.wait()
    sy.flush(force=True)
    assert (tmp_path / "bk" / "runs" / "t" / "a.json").read_text() == "version-two"
