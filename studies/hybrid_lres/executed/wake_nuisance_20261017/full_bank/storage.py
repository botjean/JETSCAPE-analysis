"""Enforce the shared new-study storage limit without racing other tasks."""
import fcntl
import os
from pathlib import Path


def write_checked(path, data, root, cap):
    path, root = Path(path), Path(root)
    if path.exists():
        raise ValueError('Refuse to replace an existing immutable output')
    with (root/'STORAGE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        used = 0
        for directory, _, files in os.walk(root, followlinks=False):
            for name in files:
                candidate = Path(directory)/name
                if not candidate.is_symlink():
                    used += candidate.stat().st_size
        if used+len(data) > cap:
            raise ValueError('Approved new-study storage cap would be exceeded')
        with path.open('xb') as stream:
            stream.write(data)
