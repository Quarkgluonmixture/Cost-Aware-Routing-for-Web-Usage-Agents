"""Identity-checked episode-summary row loading for paper-grade producers.

The filename task ID and the payload task ID are two encodings of the same
logical identity.  Canonical analysis must never choose between them: they
must agree, and each logical task may occur only once in an episodes directory.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

from p79.experiment.io_utils import load_episode_summary_strict


_SUMMARY_NAME_RE = re.compile(r"(?:^|_)task_(\d+)_summary_v2\.json$")


def filename_task_id(path: Path) -> int:
    """Return the task ID encoded in a canonical summary filename."""
    match = _SUMMARY_NAME_RE.search(path.name)
    if match is None:
        raise ValueError(
            f"Cannot parse canonical task ID from episode summary filename: {path}"
        )
    return int(match.group(1))


def load_task_rows(
    episodes_dir: Path,
    *,
    strict_mode: str | None = None,
    reject_needs_reevaluation: bool = True,
) -> dict[int, dict[str, Any]]:
    """Load one identity-validated ``task_id -> payload`` map.

    Corrupt/type-invalid rows follow the repository's ``P79_STRICT`` policy.
    Identity mismatch and duplicate logical IDs are always hard errors,
    including in lenient diagnostic mode: neither condition has a safe row to
    prefer, and silently choosing one can split H1 from H2/H3 task universes.
    """
    episodes_dir = Path(episodes_dir)
    if not episodes_dir.exists():
        return {}
    if strict_mode is None:
        strict_env = os.environ.get("P79_STRICT", "1").lower()
        strict_mode = "lenient" if strict_env in ("0", "false", "no") else "strict"
    if strict_mode not in {"strict", "lenient"}:
        raise ValueError(f"strict_mode must be 'strict' or 'lenient', got {strict_mode!r}")

    rows: dict[int, dict[str, Any]] = {}
    source_by_task: dict[int, Path] = {}
    for path in sorted(episodes_dir.glob("*_summary_v2.json")):
        filename_id = filename_task_id(path)
        payload = load_episode_summary_strict(
            path,
            mode=strict_mode,
            reject_needs_reevaluation=reject_needs_reevaluation,
        )
        if payload is None:
            continue
        payload_id = int(payload["task_id"])
        if payload_id != filename_id:
            raise ValueError(
                "Episode summary task identity mismatch: "
                f"filename task_id={filename_id}, payload task_id={payload_id}, path={path}"
            )
        if payload_id in rows:
            raise ValueError(
                "Duplicate logical episode summary task_id="
                f"{payload_id}: {source_by_task[payload_id]} and {path}"
            )
        rows[payload_id] = payload
        source_by_task[payload_id] = path
    return rows


def load_cell_task_rows(
    cell: dict[str, Any],
    *,
    modes: Iterable[str],
    strict_mode: str | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Load identity-checked rows for every requested mode in one cell."""
    cell_modes = cell.get("modes", {})
    return {
        mode: (
            load_task_rows(cell_modes[mode], strict_mode=strict_mode)
            if cell_modes.get(mode) is not None
            else {}
        )
        for mode in modes
    }
