#!/usr/bin/env python3
"""[Outcome 0a] Outcome dimension — aggregate SR per mode.

Outputs:
- docs/analysis/cross_sites/sr_per_mode.json
- docs/analysis/cross_sites/sr_per_mode.md

§139.8 + /stress A1.6 (2026-05-16): FP post-hoc layer retired entirely.
`success` is canonical; no na_fp / eval_fp / visual_fp / adjusted_success
emission. Filename retained as `sr_fp_per_mode` for callers; new schema only
emits canonical `n_success` / `sr_pct` plus completeness ratio against the
N/A-excluded `scored_task_count`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells

from p79.experiment.analysis import scored_task_count

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT_JSON = ROOT / "docs/analysis/cross_sites/sr_per_mode.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/sr_per_mode.md"


def _summary_dirs_from_registry() -> dict[str, dict[str, dict[str, Path]]]:
    out: dict[str, dict[str, dict[str, Path]]] = {}
    for baseline in BASELINE_ORDER:
        out[baseline] = {}
        for site in SITE_ORDER:
            out[baseline][site] = {
                cell.mode: cell.episodes_dir
                for cell in get_cells(baseline=baseline, site=site)
            }
    return out


BASELINE_ORDER = ["B0", "B1", "B2"]
SITE_ORDER = ["reddit", "classifieds"]
SUMMARY_DIRS = _summary_dirs_from_registry()
MODE_ORDER = PAPER_MODES


def task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)_summary", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def aggregate_cell(baseline: str, site: str, mode: str, ep_dir: Path) -> dict[str, Any]:
    # B-283 fix (2026-05-16, A1.8): use strict loader to guard against the
    # `bool("false")` truthy attack (codex Mode B F3). Pre-fix path was
    # `bool(row.get("success", False))` — JSON string "false" is Python truthy
    # → SR inflated silently. Strict loader raises on type mismatch at boundary.
    from p79.experiment.io_utils import load_episode_summary_strict

    rows: dict[int, dict[str, Any]] = {}
    for path in sorted(ep_dir.glob("*_summary_v2.json")):
        tid = task_id(path)
        if tid in rows:
            print(f"[warn] duplicate task summary ignored for {baseline}/{site}/{mode}: {path}", file=sys.stderr)
            continue
        # Lenient mode in aggregator: log + skip (don't crash whole pipeline);
        # strict-mode escalation lives in validate_run.py for paper-grade gate.
        loaded = load_episode_summary_strict(path, mode="lenient")
        if loaded is None:
            continue  # corrupt or type-mismatch — already logged by loader
        rows[tid] = loaded

    n_total = len(rows)
    # Post-strict: every row has `success: bool`. The defensive `== True` keeps
    # the intent crystal clear (paper §1 hero number rides on this line).
    n_success = sum(1 for row in rows.values() if row.get("success") is True)
    expected_n = scored_task_count(site, "visualwebarena")
    complete = expected_n > 0 and n_total >= expected_n

    return {
        "baseline": baseline,
        "site": site,
        "mode": mode,
        "n_total": n_total,
        "expected_n": expected_n,
        "complete": complete,
        "completeness_ratio": round(n_total / expected_n, 6) if expected_n else 0.0,
        "n_success": n_success,
        "sr_pct": round(pct(n_success, n_total), 6),
        "source_dir": str(ep_dir.relative_to(ROOT)),
    }


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def write_markdown(summary_table: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# SR per Mode")
    lines.append("")
    lines.append("Standalone Outcome aggregation from paper-grade per-task `summary_v2.json` files.")
    lines.append("")
    lines.append("## Main Table")
    lines.append("")
    lines.append("| baseline | site | mode | n | expected | complete | SR |")
    lines.append("|---|---|---|---:|---:|:---:|---:|")
    for row in summary_table:
        complete_marker = "✓" if row["complete"] else f"{row['completeness_ratio']:.0%}"
        lines.append(
            f"| {row['baseline']} | {row['site']} | {row['mode']} | {row['n_total']} | "
            f"{row['expected_n']} | {complete_marker} | "
            f"{fmt_pct(row['sr_pct'])} |"
        )

    lines.append("")
    lines.append("## SR ranking per (baseline, site)")
    lines.append("")
    for baseline in BASELINE_ORDER:
        for site in SITE_ORDER:
            rows = [row for row in summary_table if row["baseline"] == baseline and row["site"] == site]
            if not rows:
                continue
            rows.sort(key=lambda row: (-row["sr_pct"], row["mode"]))
            ranking = " > ".join(f"{row['mode']} {fmt_pct(row['sr_pct'])}" for row in rows)
            lines.append(f"- {baseline} {site}: {ranking}")

    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "§139.8 + /stress A1.6 (2026-05-16) hard-delete: post-hoc `adjusted_success` / "
        "`na_fp` / `eval_fp` / `visual_fp` layer fully retired. `success` is canonical "
        "(N/A excluded at task-load, B-91 LLM-judge empty-pred guard). `expected_n` "
        "comes from `scored_task_count(site)`; cells with `complete == false` are not "
        "headline data and downstream meta-analysis should drop them."
    )
    OUT_MD.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    summary_table: list[dict[str, Any]] = []
    cells: dict[str, dict[str, Any]] = {}
    for baseline in BASELINE_ORDER:
        baseline_dirs = SUMMARY_DIRS.get(baseline, {})
        for site in SITE_ORDER:
            site_dirs = baseline_dirs.get(site, {})
            for mode in MODE_ORDER:
                ep_dir = site_dirs.get(mode)
                if ep_dir is None:
                    continue
                if not ep_dir.exists():
                    continue
                cell = aggregate_cell(baseline, site, mode, ep_dir)
                if cell["n_total"] == 0:
                    continue
                cells[f"{baseline}/{site}/{mode}"] = cell
                summary_table.append(cell)

    out = {
        "method": "canonical SR (success) aggregation from per-task summary_v2.json; "
                  "expected_n from scored_task_count; no post-hoc FP adjustment",
        "schema_version": "v2-2026-05-16-fp-retire",
        "cells": cells,
        "summary_table": summary_table,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    write_markdown(summary_table)
    print(f"[json] {OUT_JSON}")
    print(f"[md] {OUT_MD}")


if __name__ == "__main__":
    main()
