"""Hugging Face Hub repo as the checkpoint store (private by default).
Params: repo ("user/name"), private=True, repo_type="model"|"dataset", prefix="mltok" (everything is written
under this folder of the repo - other files in the repo are never touched), token_env="HF_TOKEN",
retries=3, squash_every=0 (>0: squash repo history every N pushes to reclaim LFS storage; irreversible).
The token is read from the environment variable only (never from config, never printed)."""
import os
import shutil
import tempfile
import time
from pathlib import Path

from mltok.contracts import Store
from mltok.registry import register


@register("store", "hf_hub")
class HFHubStore(Store):
    api_factory = None  # tests inject a fake API here

    def __init__(self, repo, private=True, repo_type="model", prefix="mltok", token_env="HF_TOKEN",
                 retries=3, squash_every=0):
        self.repo, self.private, self.repo_type = repo, private, repo_type
        self.prefix, self.token_env, self.retries, self.squash_every = prefix.strip("/"), token_env, retries, squash_every
        self._ready, self._pushes = False, 0

    # ------------------------------------------------------------------ helpers
    def _api(self):
        if self.api_factory is not None:
            return self.api_factory()
        from huggingface_hub import HfApi

        return HfApi(token=os.environ.get(self.token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)

    def _path(self, remote):
        return f"{self.prefix}/{remote}".strip("/") if self.prefix else remote

    def _retry(self, fn):
        err = None
        for i in range(max(1, self.retries)):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                err = e
                if i + 1 < self.retries:
                    time.sleep(2 ** (i + 1))
        raise err

    def _ensure_repo(self, api):
        if not self._ready:
            api.create_repo(self.repo, repo_type=self.repo_type, private=self.private, exist_ok=True)
            self._ready = True

    # ------------------------------------------------------------------ contract
    def push(self, files, message=""):
        from huggingface_hub import CommitOperationAdd

        api = self._api()
        self._retry(lambda: self._ensure_repo(api))
        ops = [CommitOperationAdd(path_in_repo=self._path(r), path_or_fileobj=str(l)) for l, r in files]
        self._retry(lambda: api.create_commit(repo_id=self.repo, repo_type=self.repo_type, operations=ops,
                                              commit_message=message or "mltok checkpoint"))
        self._pushes += 1
        if self.squash_every and self._pushes % self.squash_every == 0:
            api.super_squash_history(repo_id=self.repo, repo_type=self.repo_type)

    def pull(self, remote_prefix, local_dir, skip_existing=True):
        api = self._api()
        try:
            names = api.list_repo_files(self.repo, repo_type=self.repo_type)
        except Exception:  # noqa: BLE001 - repo does not exist yet: nothing to restore
            return 0
        base = self._path(remote_prefix).rstrip("/") + "/"
        n = 0
        for name in names:
            if not name.startswith(base):
                continue
            dest = Path(local_dir) / name[len(base):]
            if skip_existing and dest.exists():
                continue
            with tempfile.TemporaryDirectory() as tmp:
                got = self._retry(lambda: api.hf_hub_download(repo_id=self.repo, filename=name,
                                                              repo_type=self.repo_type, local_dir=tmp))
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(got), str(dest))
            n += 1
        return n

    def delete(self, remote_paths):
        from huggingface_hub import CommitOperationDelete

        api = self._api()
        names = set(api.list_repo_files(self.repo, repo_type=self.repo_type))
        ops = [CommitOperationDelete(path_in_repo=self._path(r)) for r in remote_paths if self._path(r) in names]
        if ops:
            api.create_commit(repo_id=self.repo, repo_type=self.repo_type, operations=ops, commit_message="mltok: remove obsolete checkpoint")
