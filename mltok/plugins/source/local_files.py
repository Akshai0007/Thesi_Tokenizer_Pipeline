"""Your own files: .jsonl / .jsonl.gz ({"text":...}), .txt (one doc per line), or .parquet.
Params: path (file, directory or glob), text (field name for jsonl/parquet)."""
import glob
import gzip
import json
import os

from mltok.contracts import Source
from mltok.registry import register


def _expand(path):
    if os.path.isdir(path):
        return sorted(
            os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs
            if f.endswith((".jsonl", ".jsonl.gz", ".txt", ".txt.gz", ".parquet"))
        )
    return sorted(glob.glob(path)) or [path]


@register("source", "local_files")
class LocalFilesSource(Source):
    def __init__(self, id, path, text="text", **params):
        super().__init__(id=id, **params)
        self.path, self.field = path, text

    def iter_docs(self):
        for p in _expand(self.path):
            gz = p.endswith(".gz")
            if p.endswith(".parquet"):
                import pyarrow.parquet as pq

                for batch in pq.ParquetFile(p).iter_batches(columns=[self.field]):
                    for t in batch.column(0).to_pylist():
                        if t:
                            yield {"text": t, "src": self.id}
                continue
            opener = gzip.open if gz else open
            with opener(p, "rt", encoding="utf-8") as f:
                for line in f:
                    if ".jsonl" in p:
                        try:
                            t = json.loads(line).get(self.field)
                        except json.JSONDecodeError:
                            continue
                    else:
                        t = line.rstrip("\n")
                    if t:
                        yield {"text": t, "src": self.id}
