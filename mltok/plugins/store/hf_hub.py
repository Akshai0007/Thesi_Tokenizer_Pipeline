"""Hugging Face Hub repo as the checkpoint store (private by default).
Credentials come from the environment or the Hugging Face CLI credential store."""
import os
import shutil
import tempfile
import time
from pathlib import Path

from mltok.contracts import Store, StoreAccessError
from mltok.lib.artifact_paths import relative_path, unlinked_path
from mltok.registry import register


@register("store", "hf_hub")
class HFHubStore(Store):
    api_factory = None  # tests inject a fake API here
    verified_snapshots = True

    def __init__(self, repo, private=True, repo_type="model", prefix="mltok", token_env="HF_TOKEN",
                 retries=3, squash_every=0):
        if type(private) is not bool:
            raise ValueError("Hub privacy must be explicitly set to true or false")
        self.repo, self.private, self.repo_type = repo, private, repo_type
        self.prefix, self.token_env, self.retries = prefix.strip("/"), token_env, retries
        if self.prefix:
            relative_path(self.prefix)
        if squash_every:
            raise ValueError("Automatic Hub history squashing is disabled")

    # ------------------------------------------------------------------ helpers
    def _api(self):
        if self.api_factory is not None:
            return self.api_factory()
        from huggingface_hub import HfApi

        return HfApi(token=os.environ.get(self.token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)

    def _path(self, remote):
        try:
            relative_path(remote)
        except (TypeError, ValueError):
            raise StoreAccessError("Invalid Hub artifact path") from None
        return f"{self.prefix}/{remote}" if self.prefix else remote

    def _retry(self, fn, *, entry=False):
        for i in range(max(1, self.retries)):
            try:
                return fn()
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 401:
                    try:
                        self._api().whoami()
                    except Exception as auth:
                        auth_status = getattr(getattr(auth, "response", None), "status_code", None)
                        if auth_status in (401, 403) or type(auth).__name__ == "LocalTokenNotFoundError":
                            raise StoreAccessError("Hub credentials are missing, invalid, or expired") from None
                        raise StoreAccessError("Hub authentication could not be verified") from None
                    raise StoreAccessError("Hub repository unavailable to the authenticated account") from None
                if status == 403:
                    raise StoreAccessError("Hub permission denied") from None
                if status == 404:
                    if entry and any(kind in type(e).__name__ for kind in ("EntryNotFound", "RevisionNotFound")):
                        raise FileNotFoundError("Hub snapshot artifact or pinned revision missing") from None
                    raise StoreAccessError("Hub repository or revision unavailable") from None
                if status in (409, 412):
                    raise StoreAccessError("Hub revision changed during upload. Retry synchronization.") from None
                if i + 1 >= self.retries:
                    raise StoreAccessError("Hub operation failed") from None
                time.sleep(2 ** (i + 1))

    def _ensure_repo(self, api):
        self._retry(lambda: api.create_repo(self.repo, repo_type=self.repo_type, private=self.private, exist_ok=True))

    def revision(self, create=False):
        api = self._api()
        if create:
            self._ensure_repo(api)
        info = self._retry(lambda: api.repo_info(self.repo, repo_type=self.repo_type))
        if self.private and getattr(info, "private", None) is not True:
            raise StoreAccessError("Hub repository is not confirmed private")
        if not info.sha:
            raise StoreAccessError("Hub repository has no readable revision")
        return info.sha

    def list_at(self, revision):
        api = self._api()
        prefix = self.prefix + "/" if self.prefix else ""
        return {p[len(prefix):] for p in self._retry(lambda: api.list_repo_files(
            self.repo, repo_type=self.repo_type, revision=revision)) if p.startswith(prefix)}

    def fetch_at(self, remote, revision, directory):
        api = self._api()
        return Path(self._retry(lambda: api.hf_hub_download(
            repo_id=self.repo, filename=self._path(remote), repo_type=self.repo_type,
            revision=revision, local_dir=directory, force_download=True), entry=True))

    def push_at(self, files, message, parent):
        from huggingface_hub import CommitOperationAdd

        # Recheck visibility before each write, including manifest publication.
        if self.revision() != parent:
            raise StoreAccessError("Hub revision changed during upload. Retry synchronization.")
        api = self._api()
        files = list(files)
        protected = {r for _, r in files if r.startswith("snapshots/")}
        if protected and protected & self.list_at(parent):
            raise StoreAccessError("An existing snapshot manifest or blob cannot be overwritten")
        if len({r for _, r in files}) != len(files):
            raise StoreAccessError("Duplicate Hub artifact paths in one commit")
        ops = [CommitOperationAdd(path_in_repo=self._path(r), path_or_fileobj=str(l)) for l, r in files]
        result = self._retry(lambda: api.create_commit(
            repo_id=self.repo, repo_type=self.repo_type, operations=ops,
            commit_message=message, parent_commit=parent))
        return result.oid

    # ------------------------------------------------------------------ contract
    def push(self, files, message=""):
        files = list(files)
        if any(r.startswith("snapshots/") for _, r in files):
            raise StoreAccessError("Use verified publication for the Hub snapshot namespace")
        self.push_at(files, message or "mltok upload", self.revision(create=True))

    def pull(self, remote_prefix, local_dir, skip_existing=True):
        revision = self.revision()
        base = remote_prefix.rstrip("/") + "/"
        n = 0
        for name in sorted(self.list_at(revision)):
            if not name.startswith(base):
                continue
            try:
                dest = unlinked_path(Path(local_dir) / relative_path(name[len(base):]))
            except ValueError:
                raise StoreAccessError("Unsafe Hub restore destination") from None
            if skip_existing and dest.exists():
                continue
            with tempfile.TemporaryDirectory() as tmp:
                got = self.fetch_at(name, revision, tmp)
                if not unlinked_path(got).is_relative_to(unlinked_path(tmp)):
                    raise StoreAccessError("Hub download is outside staging")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(got), str(dest))
            n += 1
        return n

    def delete(self, remote_paths):
        raise StoreAccessError("Hub deletion is disabled. Use an explicitly approved cleanup workflow.")
