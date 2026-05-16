"""Shared JSONL I/O utilities with restart dedup and corrupt-line handling."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# B-196 (/stress A1.4b-ii codex B-ii-4, P1): module-level corruption counter
# so `analysis.py::analyze_run` can emit a canonical `jsonl_integrity_report.csv`
# alongside `parse_failures.csv` (B-174). Reset at the start of each
# `analyze_run` and read at the end. Each entry records:
#   {"path": str, "lines_read": int, "corrupt_lines": int,
#    "dedup_discarded": int, "summary_identity_mismatch": bool}
_JSONL_INTEGRITY_LOG: List[Dict[str, Any]] = []


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


def read_jsonl_dedup(
    path: Path,
    summary_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Read a single JSONL file, deduplicating restart artifacts.

    B-180 (/stress A1.4b-i codex B7): when ``summary_path`` is provided, the
    last JSONL segment is validated against the summary's authoritative
    fields ((schema_version, run_id, condition_id, seed, benchmark_site,
    task_id, steps)). If the segment doesn't match, log a warning + still
    return the last-segment lines (caller decides whether to consume) so
    audit can see the divergence without crashing analysis. Pre-fix: a
    restart that wrote ``step_idx=0`` and then crashed before summary
    overwrite would have the old complete summary co-exist with the new
    partial segment; the dedup unconditionally kept the partial, breaking
    step-level cost/latency diagnostics while episode-level summaries
    pointed at the old run.
    """
    file_lines: List[Dict[str, Any]] = []
    corrupt_count = 0
    total_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                file_lines.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Dropped corrupt JSONL line %d in %s: %.100s", line_num, path, line)
                corrupt_count += 1
                continue
    last_segment = dedup_restart_lines(file_lines)
    dedup_discarded = max(0, len(file_lines) - len(last_segment))

    identity_mismatch = False
    if summary_path is not None and last_segment:
        identity_mismatch = _validate_against_summary(path, last_segment, summary_path)

    # B-196: record per-file integrity stats; consumed by analysis.py
    _JSONL_INTEGRITY_LOG.append({
        "path": str(path),
        "lines_read": total_lines,
        "corrupt_lines": corrupt_count,
        "dedup_discarded": dedup_discarded,
        "summary_identity_mismatch": identity_mismatch,
    })

    return last_segment


def _validate_against_summary(
    jsonl_path: Path,
    last_segment: List[Dict[str, Any]],
    summary_path: Path,
) -> bool:
    """B-180 helper: emit warning if last JSONL segment doesn't match summary.

    Identity tuple checked: (schema_version, run_id, condition_id, seed,
    benchmark_site, task_id). Cardinality: len(last_segment) vs summary
    `steps` field. All mismatches are logged + caller may inspect; not raised
    because in-progress data is legitimately mid-flight.

    Returns True if a mismatch was detected (per B-196 integrity report).
    """
    if not summary_path.exists():
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "B-180 read_jsonl_dedup: cannot read summary %s for identity check: %s",
            summary_path, exc,
        )
        return False

    any_mismatch = False
    first = last_segment[0]
    identity_keys = ("schema_version", "run_id", "condition_id", "seed",
                     "benchmark_site", "task_id")
    mismatches = []
    for k in identity_keys:
        s_val = summary.get(k)
        l_val = first.get(k)
        if s_val is None or l_val is None:
            continue  # field not stamped in either side; skip
        if s_val != l_val:
            mismatches.append(f"{k}: summary={s_val!r} vs jsonl={l_val!r}")
    if mismatches:
        any_mismatch = True
        logger.warning(
            "B-180 identity mismatch %s ↔ %s: %s",
            jsonl_path, summary_path, "; ".join(mismatches),
        )

    summary_steps = summary.get("steps")
    if isinstance(summary_steps, int) and summary_steps != len(last_segment):
        any_mismatch = True
        logger.warning(
            "B-180 step count mismatch %s: summary.steps=%d vs jsonl_segment=%d "
            "(may indicate restart-crash; summary points to older run)",
            jsonl_path, summary_steps, len(last_segment),
        )
    return any_mismatch
