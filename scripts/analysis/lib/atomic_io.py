"""Durable same-filesystem replacement helpers for analysis producers.

These helpers provide single-file replacement and advisory locking primitives.
They do not claim a multi-file filesystem transaction.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# 2026-10-07 (实验笔记 §536.4): the module imported fcntl unconditionally, so every producer that
# writes through it (sr_per_mode among them) could not start on Windows. The platform branch is
# chosen explicitly — no exception is swallowed to get there.
if os.name == "nt":
    import msvcrt
else:
    import fcntl


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes made by replace/unlink operations.

    POSIX only: Windows cannot open a directory for fsync; NTFS journals the rename that
    os.replace performs, which is the guarantee this function adds on POSIX.
    """
    if os.name == "nt":
        return
    fd = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold an inter-process exclusive ``flock`` on a persistent lock file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)  # blocking (retries ~10 s, then raises)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write through a sibling temporary, replace, then fsync the parent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
