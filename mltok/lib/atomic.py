"""Publish files through a durable, same-directory temporary write."""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_writer(path):
    path = Path(path)
    stream = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)
