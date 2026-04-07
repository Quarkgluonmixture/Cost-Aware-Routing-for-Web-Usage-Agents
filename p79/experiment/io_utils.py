"""Shared JSONL I/O utilities with restart dedup and corrupt-line handling."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def dedup_restart_lines(file_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the last run when a JSONL has restart artifacts.

    The runner appends to JSONL (mode='a'), so if the watchdog/queue
    restarts a task, earlier runs' steps remain in the file.  We detect
    restarts by step_idx resetting to 0 and keep lines from the last run
    (matching the authoritative summary_v2.json which is overwritten).
    """
    if not file_lines:
        return file_lines
    last_run_start = 0
    for i, rec in enumerate(file_lines):
        if i > 0 and rec.get("step_idx", -1) == 0:
            last_run_start = i
    if last_run_start > 0:
        logger.debug(
            "Dedup: discarded %d lines from earlier run(s)", last_run_start
        )
    return file_lines[last_run_start:]


def read_jsonl_dedup(path: Path) -> List[Dict[str, Any]]:
    """Read a single JSONL file, deduplicating restart artifacts."""
    file_lines: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                file_lines.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Dropped corrupt JSONL line in %s: %.100s", path, line)
                continue
    return dedup_restart_lines(file_lines)
