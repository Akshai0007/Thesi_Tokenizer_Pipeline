"""Common path checks for local and Hub artifacts."""
import os
import re
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath


def relative_path(value):
    p = PurePosixPath(value)
    if (p.is_absolute() or p.as_posix() != value or not p.parts
            or any(part in (".", "..") or re.search(r'[\\:<>|"\x00-\x1f?*]', part)
                   or part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved()
                   for part in p.parts)):
        raise ValueError("Invalid relative artifact path")
    return p


def unlinked_path(path):
    """Reject symlinks and Windows reparse points in any existing path component."""
    absolute = Path(os.path.abspath(path))
    for part in (*reversed(absolute.parents), absolute):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
                getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("Artifact path contains a symlink or reparse point")
    return absolute
