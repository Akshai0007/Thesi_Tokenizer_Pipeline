"""No remote store: nothing is pushed, nothing is restored (local-only runs)."""
from mltok.contracts import Store
from mltok.registry import register


@register("store", "none")
class NoStore(Store):
    enabled = False
    example_params = {}

    def push(self, files, message=""):
        return None

    def pull(self, remote_prefix, local_dir, skip_existing=True):
        return 0
