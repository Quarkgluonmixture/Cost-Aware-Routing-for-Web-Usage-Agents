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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells

from p79.experiment.analysis import scored_task_count
from scripts.analysis.lib.atomic_io import atomic_write_text
from scripts.analysis.lib.canonical_task_universe import (
    expected_scored_ids,
    task_id_set_sha256,
)

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


def aggregate_cell(
    baseline: str,
    site: str,
    mode: str,
    ep_dir: Path,
    *,
    expected_ids: frozenset[int] | set[int] | None = None,
) -> dict[str, Any]:
    # B-283 fix (2026-05-16, A1.8): use strict loader to guard against the
    # `bool("false")` truthy attack (codex Mode B F3). Pre-fix path was
    # `bool(row.get("success", False))` — JSON string "false" is Python truthy
    # → SR inflated silently. Strict loader raises on type mismatch at boundary.
    from p79.experiment.io_utils import load_episode_summary_strict

    rows: dict[int, dict[str, Any]] = {}
    task_id_mismatch_files: list[str] = []
    for path in sorted(ep_dir.glob("*_summary_v2.json")):
        filename_tid = task_id(path)
        # Lenient mode in aggregator: log + skip (don't crash whole pipeline);
        # strict-mode escalation lives in validate_run.py for paper-grade gate.
        # B-549 (/stress A1.5 P0-2-AB* Claude+codex OOB sibling propagation,
        # 2026-05-17): add `reject_needs_reevaluation=True` to close B-486
        # quarantine leak into paper §1 SR canonical producer. A1.5b Phase 2
        # B-542 added the strict-loader kwarg + propagated to
        # `aggregate_phase1_full_prereg_decision._load_cell_per_task` +
        # `aggregate_phantom_lift.load()`, but missed THIS paper §1 SR
        # canonical producer (codex OOB Bug Table P0-1). Current archive 0
        # quarantined episodes verified (5100 summaries), but ANY Phase 1a
        # `_run_and_record_episode` exception path writes
        # `needs_reevaluation=True` → quarantined episodes would silently
        # enter denominator as `success=False` → paper §1 SR table polluted.
        # Lenient mode + reject_quarantine → loader returns None for
        # quarantined rows → skipped here. Strict-mode escalation lives in
        # `validate_run.py`.
        loaded = load_episode_summary_strict(
            path, mode="lenient", reject_needs_reevaluation=True,
        )
        if loaded is None:
            continue  # corrupt or type-mismatch or B-486 quarantine — already logged
        loaded_tid = int(loaded["task_id"])
        if loaded_tid != filename_tid:
            task_id_mismatch_files.append(path.name)
        if loaded_tid in rows:
            print(f"[warn] duplicate task summary ignored for {baseline}/{site}/{mode}: {path}", file=sys.stderr)
            continue
        rows[loaded_tid] = loaded

    n_total = len(rows)
    if expected_ids is None:
        expected_set, task_set_sha = expected_scored_ids(site)
    else:
        expected_set = frozenset(int(t) for t in expected_ids)
        task_set_sha = task_id_set_sha256(expected_set)
    observed_ids = frozenset(rows)
    missing_ids = sorted(expected_set - observed_ids)
    extra_ids = sorted(observed_ids - expected_set)
    expected_n = len(expected_set)
    exact_set = observed_ids == expected_set and not task_id_mismatch_files
    canonical_rows = {tid: rows[tid] for tid in expected_set if tid in rows}
    # Post-strict: every row has `success: bool`. The defensive `== True` keeps
    # the intent crystal clear (paper §1 hero number rides on this line).
    n_success = sum(1 for row in canonical_rows.values() if row.get("success") is True)
    # B-598 (/stress A1.6a P0-3-AB Claude + codex overlap, 2026-05-17):
    # paper §1 SR canonical producer MUST pass `strict=True`. Pre-fix
    # `scored_task_count(site, "visualwebarena")` defaulted strict=False
    # → missing config silently returned 0 → `complete = expected_n > 0 ...`
    # short-circuited to False even when n_total had full N data → cell
    # silently demoted to incomplete, paper §1 hero "N cells" miscounted.
    # All other paper-grade callers (active_processes / axis1 / fig3 /
    # run_registry / mechanism_per_task) already use strict=True; this
    # was the lone holdout post-§139.8.
    if expected_ids is None:
        expected_n_from_count = scored_task_count(site, "visualwebarena", strict=True)
        if expected_n != expected_n_from_count:
            raise ValueError(
                f"canonical task helper/count mismatch for {site}: "
                f"{expected_n} != {expected_n_from_count}"
            )
    complete = expected_n > 0 and exact_set

    # B-403 (/stress A1.1 v8 Mode B P1-9, 2026-05-16): image_encode_error
    # symmetric-exclude transparency column. Agent comments at
    # `qwen3vl_agent.py:355-363` + `gemma3vl_agent.py:330-336` mandated
    # exclusion of steps with `image_encode_error > 0` for paper-grade
    # cross-baseline SR comparability. Pre-fix: no aggregator implemented
    # this — infra failures (PIL decode / base64 OOM on B0 proxy) were
    # silently scored as model/task failures. EpisodeSummaryV2 now stamps
    # per-episode count (B-403 in runner). Aggregator emits 3 columns:
    #   n_image_encode_error_episodes: episodes with ≥1 bad-image step
    #   image_encode_error_episode_rate: ratio for disclosure
    #   sr_pct_clean: SR computed excluding bad-image episodes
    # Reviewer can compare `sr_pct` vs `sr_pct_clean`: gap >> 0 indicates
    # infra-failure contamination biased the headline.
    n_bad_image = sum(
        1 for row in canonical_rows.values()
        if int(row.get("image_encode_error_step_count", 0) or 0) > 0
    )
    clean_rows = [
        row for row in canonical_rows.values()
        if int(row.get("image_encode_error_step_count", 0) or 0) == 0
    ]
    n_clean = len(clean_rows)
    n_success_clean = sum(1 for row in clean_rows if row.get("success") is True)

    # B-600 (/stress A1.6a P1-2-AC Claude+codex overlap, 2026-05-17):
    # infra-noise transparency appendix. `benchmark_noise=True` flags
    # api_rate_limit / playwright_crash / docker_service_error / etc —
    # DIFFERENT semantic from N/A FP (§139.8 upstream-fixed) AND from
    # image_encode_error (B-403 cross-baseline parity). Paper §1 hero
    # = raw `sr_pct` (Q2 user decision 2026-05-17 + §139.8 alignment);
    # `sr_pct_infra_clean` exposes residual infra-noise sensitivity for
    # paper §3 transparency appendix. Gap (`sr_pct` − `sr_pct_infra_clean`)
    # should be small when watchdog auto-clean protocol is healthy; large
    # gap = forensic flag for infra instability that re-run should
    # clean up.
    n_infra_noise = sum(
        1 for row in canonical_rows.values()
        if bool(row.get("benchmark_noise", False))
    )
    infra_clean_rows = [
        row for row in canonical_rows.values()
        if not bool(row.get("benchmark_noise", False))
    ]
    n_infra_clean = len(infra_clean_rows)
    n_success_infra_clean = sum(
        1 for row in infra_clean_rows if row.get("success") is True
    )

    try:
        source_dir = str(ep_dir.relative_to(ROOT))
    except ValueError:
        source_dir = str(ep_dir)

    return {
        "baseline": baseline,
        "site": site,
        "mode": mode,
        "n_total": n_total,
        "observed_n": n_total,
        "expected_n": expected_n,
        "complete": complete,
        "complete_exact": complete,
        "task_set_sha256": task_set_sha,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "task_id_mismatch_files": task_id_mismatch_files,
        "completeness_ratio": round(n_total / expected_n, 6) if expected_n else 0.0,
        "n_success": n_success,
        "sr_denominator_n": expected_n,
        "sr_pct": round(pct(n_success, expected_n), 6),
        # B-403 (P1-9): image_encode_error symmetric-exclude transparency
        "n_image_encode_error_episodes": n_bad_image,
        "image_encode_error_episode_rate": round(pct(n_bad_image, n_total), 6),
        "n_clean": n_clean,
        "n_success_clean": n_success_clean,
        "sr_pct_clean": round(pct(n_success_clean, n_clean), 6),
        # B-600 (A1.6a P1-2-AC): benchmark_noise infra-clean transparency
        "n_infra_noise_episodes": n_infra_noise,
        "infra_noise_episode_rate": round(pct(n_infra_noise, n_total), 6),
        "n_infra_clean": n_infra_clean,
        "n_success_infra_clean": n_success_infra_clean,
        "sr_pct_infra_clean": round(pct(n_success_infra_clean, n_infra_clean), 6),
        "source_dir": source_dir,
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
    lines.append("| baseline | site | mode | observed | expected | exact | missing IDs | extra IDs | task-set SHA | SR |")
    lines.append("|---|---|---|---:|---:|:---:|---|---|---|---:|")
    for row in summary_table:
        complete_marker = "✓" if row["complete"] else f"{row['completeness_ratio']:.0%}"
        missing = ",".join(str(t) for t in row["missing_ids"]) or "—"
        extra = ",".join(str(t) for t in row["extra_ids"]) or "—"
        lines.append(
            f"| {row['baseline']} | {row['site']} | {row['mode']} | {row['n_total']} | "
            f"{row['expected_n']} | {complete_marker} | {missing} | {extra} | "
            f"`{row['task_set_sha256'][:12]}…` | "
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
        "comes from the exact scored IDs selected by `p79.experiment.tasks.load_tasks` "
        "and is cross-checked against `scored_task_count(site)`. `complete` requires "
        "observed IDs == expected IDs; missing/extra IDs fail closed. Canonical `sr_pct` "
        "always uses `expected_n` as its denominator, and extra IDs never enter the numerator."
    )
    atomic_write_text(OUT_MD, "\n".join(lines).rstrip() + "\n")


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
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "method": "canonical SR (success) aggregation from per-task summary_v2.json; "
                  "exact scored task-ID set from p79.experiment.tasks.load_tasks; "
                  "fixed expected-set denominator; no post-hoc FP adjustment",
        "schema_version": "v3-2026-07-14-exact-task-set",
        "cells": cells,
        "summary_table": summary_table,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(OUT_JSON, json.dumps(out, indent=2) + "\n")
    write_markdown(summary_table)
    print(f"[json] {OUT_JSON}")
    print(f"[md] {OUT_MD}")


if __name__ == "__main__":
    main()
