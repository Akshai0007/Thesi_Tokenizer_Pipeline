"""Immutable merge generations for bounded, trusted local BPE runs."""

import gzip
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from zlib import error as ZlibError

from .atomic import atomic_writer


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


class BPEProgress:
    def __init__(self, directory, identity, interval):
        if type(interval) is not int or interval < 1:
            raise ValueError("Merge checkpoint interval must be a positive integer")
        self.directory = Path(directory)
        self.identity = identity
        self.interval = interval

    def latest(self):
        names = sorted(self.directory.glob("merge-*.json.gz"), reverse=True)
        for path in names:
            try:
                with gzip.open(path, "rb") as stream:
                    digest = stream.readline().strip()
                    raw = stream.read()
                if len(digest) != 64 or hashlib.sha256(raw).hexdigest().encode() != digest:
                    continue
                state = json.loads(raw)
                if state["schema"] == 1 and state["identity"] == self.identity:
                    if int(path.name.split("-")[1]) == len(state["merges"]):
                        return state
            except (EOFError, OSError, ValueError, KeyError, TypeError, ZlibError, IndexError):
                continue
        if names:
            raise ValueError("No usable merge checkpoint matches this run. Existing generations were retained.")
        return None

    def save(self, merges, seqs, flags):
        raw = canonical({"schema": 1, "identity": self.identity, "merges": merges, "seqs": seqs, "flags": flags})
        digest = hashlib.sha256(raw).hexdigest()
        compressed = gzip.compress(digest.encode() + b"\n" + raw, mtime=0)
        self.directory.mkdir(parents=True, exist_ok=True)
        key = f"merge-{len(merges):06d}-{digest}"
        path = self.directory / f"{key}.json.gz"
        while True:
            try:
                with atomic_writer(path, overwrite=False) as stream:
                    stream.write(compressed)
                return
            except FileExistsError:
                if path.read_bytes() == compressed:
                    return
                # A damaged generation must remain available for inspection.
                path = self.directory / f"{key}-{uuid4().hex}.json.gz"
