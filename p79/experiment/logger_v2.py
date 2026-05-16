from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _fsync_dir(directory: Path) -> None:
    """B-198 (/stress A1.4b-ii Claude D6): fsync the directory entry after
    `os.replace` so the rename hits stable storage on the next journal
    cycle. Pre-fix: `os.replace` is atomic at the inode level, but the
    directory entry update sits in ext4 journal up to ~30s before flush.
    DGX crash between rename + flush → reboot sees pre-rename state, paper
    runs may silently lose just-written summaries. Best-effort: on platforms
    where dir fsync raises (NFS, FAT), swallow the error — atomicity is
    still better than nothing."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # platform doesn't support dir fsync; not a hard failure


class LoggerV2:
    def __init__(self, condition_dir: Path):
        self.condition_dir = condition_dir
        self.episodes_dir = self.condition_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        # B-289 fix (2026-05-16, A1.8): fsync the parent of `episodes_dir` so
        # the directory entry hits stable storage at construction time. Pre-fix
        # the `mkdir(parents=True)` left the entry in ext4 journal up to ~30s;
        # a Spark crash between mkdir and the first write_step would leave the
        # parent dir entry unflushed → reboot sees no episodes/ → subsequent
        # writes go to a recreated dir but run_meta references the missing
        # original path. Same B-198 lineage as write_episode_summary.
        _fsync_dir(self.condition_dir)
        _fsync_dir(self.episodes_dir)

    def write_condition_meta(self, meta: Dict[str, Any]) -> None:
        self.condition_dir.mkdir(parents=True, exist_ok=True)
        path = self.condition_dir / "condition_meta.json"
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic on same filesystem
        _fsync_dir(path.parent)  # B-198 flush dir entry to disk

    def step_log_path(self, site: str, task_id: int) -> Path:
        return self.episodes_dir / f"{site}_task_{task_id}_steps_v2.jsonl"

    def summary_path(self, site: str, task_id: int) -> Path:
        return self.episodes_dir / f"{site}_task_{task_id}_summary_v2.json"

    def write_step(self, site: str, task_id: int, record: Dict[str, Any]) -> None:
        path = self.step_log_path(site, task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_episode_summary(self, site: str, task_id: int, summary: Dict[str, Any]) -> None:
        path = self.summary_path(site, task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic on same filesystem
        _fsync_dir(path.parent)  # B-198 flush dir entry to disk

    def condition_summary_path(self) -> Path:
        return self.condition_dir / "condition_summary_v2.json"

    def write_condition_summary(self, payload: Dict[str, Any]) -> None:
        self.condition_dir.mkdir(parents=True, exist_ok=True)
        path = self.condition_summary_path()
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic on same filesystem
        _fsync_dir(path.parent)  # B-198 flush dir entry to disk


def write_run_summary_atomic(run_summary_path: Path, payload: Dict[str, Any]) -> None:
    """B-331 (/stress A1.9 Mode B F6 OOB, 2026-05-16): atomic + fsync write
    for run_summary_v2.json. Pre-fix runner/main.py:710 used plain
    `json.dump` → crash mid-write → run_summary truncated. condition_summary
    already uses the LoggerV2 atomic+fsync chain (line 86-91); this helper
    extends the same durability contract to the run-level summary.

    Standalone function (not a LoggerV2 method) because run_summary lives at
    output_root, not at any single condition_dir.
    """
    run_summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = run_summary_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, run_summary_path)
    _fsync_dir(run_summary_path.parent)
