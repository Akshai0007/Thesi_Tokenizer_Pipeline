"""Copy checkpoints to another directory (a mounted drive, a Kaggle dataset staging folder, NFS ...).
Params: path."""
import shutil
from pathlib import Path

from mltok.contracts import Store
from mltok.registry import register


@register("store", "local_dir")
class LocalDirStore(Store):
    example_params = {"path": "__TMP__"}

    def __init__(self, path):
        self.root = Path(path)

    def push(self, files, message=""):
        for local, remote in files:
            dest = self.root / remote
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            shutil.copy2(local, tmp)
            tmp.replace(dest)

    def pull(self, remote_prefix, local_dir, skip_existing=True):
        base, n = self.root / remote_prefix, 0
        if not base.exists():
            return 0
        for f in sorted(p for p in base.rglob("*") if p.is_file() and not p.name.endswith(".part")):
            dest = Path(local_dir) / f.relative_to(base)
            if skip_existing and dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            n += 1
        return n

    def delete(self, remote_paths):
        for r in remote_paths:
            (self.root / r).unlink(missing_ok=True)
