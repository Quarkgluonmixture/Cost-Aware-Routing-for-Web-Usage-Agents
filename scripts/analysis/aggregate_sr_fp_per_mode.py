#!/usr/bin/env python3
"""[Layer 0a + 0b] Outcome — aggregate SR + FP per mode.

Outputs:
- docs/analysis/cross_sites/sr_fp_per_mode.json
- docs/analysis/cross_sites/sr_fp_per_mode.md
"""

from __future__ import annotations

from collections import Counter
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT_JSON = ROOT / "docs/analysis/cross_sites/sr_fp_per_mode.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/sr_fp_per_mode.md"

SUMMARY_DIRS = {
    "reddit": {
        "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
        "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
        "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
        "Phantom-SoM": RESULTS / "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0/episodes",
        "P-text": RESULTS / "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0/episodes",
    },
    "classifieds": {
        "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
        "SoM": RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
        "Vision": RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        "Phantom-SoM": RESULTS / "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0/episodes",
        "P-text": RESULTS / "B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
    },
}

MODE_ORDER = ["DOM", "P-text", "Phantom-SoM", "SoM", "Vision"]


def task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)_summary", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def aggregate_cell(site: str, mode: str, ep_dir: Path) -> dict[str, Any]:
    rows: dict[int, dict[str, Any]] = {}
    for path in sorted(ep_dir.glob("*_summary_v2.json")):
        tid = task_id(path)
        if tid in rows:
            print(f"[warn] duplicate task summary ignored for {site}/{mode}: {path}", file=sys.stderr)
            continue
        rows[tid] = read_json(path)

    n_total = len(rows)
    n_raw_success = 0
    n_adjusted_success = 0
    fp_count = 0
    fp_breakdown: Counter[str] = Counter()

    for row in rows.values():
        raw = bool(row.get("success", False))
        adjusted = bool(row.get("adjusted_success", row.get("success", False)))
        n_raw_success += int(raw)
        n_adjusted_success += int(adjusted)
        if raw and not adjusted:
            fp_count += 1
            reason = str(row.get("fp_reason") or "").strip() if "fp_reason" in row else ""
            fp_breakdown[reason or "unspecified"] += 1

    return {
        "site": site,
        "mode": mode,
        "n_total": n_total,
        "n_raw_success": n_raw_success,
        "n_adjusted_success": n_adjusted_success,
        "raw_sr_pct": round(pct(n_raw_success, n_total), 6),
        "adjusted_sr_pct": round(pct(n_adjusted_success, n_total), 6),
        "fp_count": fp_count,
        "fp_rate_pct": round(pct(fp_count, n_total), 6),
        "fp_breakdown": dict(sorted(fp_breakdown.items())),
        "source_dir": str(ep_dir.relative_to(ROOT)),
    }


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def write_markdown(summary_table: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# SR + FP per Mode")
    lines.append("")
    lines.append("Standalone Layer 0a/0b aggregation from paper-grade B0 per-task `summary_v2.json` files.")
    lines.append("")
    lines.append("## Main Table")
    lines.append("")
    lines.append("| site | mode | n | raw SR | adjusted SR | FP count | FP rate | FP breakdown |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in summary_table:
        breakdown = ", ".join(f"{reason}={count}" for reason, count in row["fp_breakdown"].items()) or "none"
        lines.append(
            f"| {row['site']} | {row['mode']} | {row['n_total']} | "
            f"{fmt_pct(row['raw_sr_pct'])} | {fmt_pct(row['adjusted_sr_pct'])} | "
            f"{row['fp_count']} | {fmt_pct(row['fp_rate_pct'])} | {breakdown} |"
        )

    lines.append("")
    lines.append("## FP rate ranking per site")
    lines.append("")
    for site in ["reddit", "classifieds"]:
        rows = [row for row in summary_table if row["site"] == site]
        rows.sort(key=lambda row: (row["fp_rate_pct"], row["mode"]))
        ranking = " < ".join(f"{row['mode']} {fmt_pct(row['fp_rate_pct'])}" for row in rows)
        lines.append(f"- {site}: {ranking}")

    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Raw SR counts `success == true`; adjusted SR counts `adjusted_success == true` "
        "with fallback to `success` when the adjusted field is absent. FP count is raw success minus adjusted success."
    )
    OUT_MD.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    summary_table: list[dict[str, Any]] = []
    cells: dict[str, dict[str, Any]] = {}
    for site in ["reddit", "classifieds"]:
        for mode in MODE_ORDER:
            cell = aggregate_cell(site, mode, SUMMARY_DIRS[site][mode])
            cells[f"{site}/{mode}"] = cell
            summary_table.append(cell)

    out = {
        "method": "aggregate raw/adjusted SR + FP from per-task summary_v2.json",
        "data_source": "paper-grade B0 5-mode runs (FRESH 04-29)",
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
