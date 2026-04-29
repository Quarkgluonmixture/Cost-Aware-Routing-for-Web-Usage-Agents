#!/usr/bin/env python3
"""[Outcome 0g] Outcome dimension — routing signal quality across conditions.

Outputs:
- results/phantom_paper/auroc_cross_condition.csv
- results/phantom_paper/auroc_cross_condition.md
- results/phantom_paper/auroc_cross_condition_summary.md

Outcome 0g: per-mode routing AUROC evidence for router-usable signals.

See docs/checkpoints/paper_planning.md §3 Outcome dimension framework.

Aggregate per-mode routing signal AUROC + 95% bootstrap CI across runs.

Reads existing per-run `analysis/signals/combined/tables/cross_mode_auroc.csv`
(produced by analyze_confidence_calibration.py) and merges them into a single
paper-ready table with run/baseline/site metadata.

Usage:
    python3 scripts/analysis/aggregate_routing_auroc.py \\
        --runs results/visualwebarena/phase1/B0_3mode_classifieds_20260413 \\
               results/visualwebarena/phase1/B0_3mode_reddit_20260422 \\
               results/visualwebarena/phase1/B0_phantom_som_classifieds_20260426 \\
               results/visualwebarena/phase1/B0_phantom_som_reddit_20260428 \\
               results/visualwebarena/phase1/B0_phantom_text_classifieds_20260427 \\
               results/visualwebarena/phase1/B0_phantom_text_reddit_20260427 \\
               results/visualwebarena/phase1/B1_3mode_classifieds_20260413 \\
               results/visualwebarena/phase1/B1_3mode_reddit_20260413 \\
        --output results/phantom_paper/auroc_cross_condition.csv

Output columns: baseline, site, mode, signal, signal_type, AUROC,
                AUROC_ci_lower, AUROC_ci_upper, n, run_id

A second markdown summary lands at <output>.md with a paper-ready table
showing top-3 signals per (baseline, site, mode).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = [
    REPO / "results/visualwebarena/phase1/B0_3mode_classifieds_20260413",
    REPO / "results/visualwebarena/phase1/B0_3mode_reddit_20260422",
    REPO / "results/visualwebarena/phase1/B0_phantom_som_classifieds_20260426",
    REPO / "results/visualwebarena/phase1/B0_phantom_som_reddit_20260428",
    REPO / "results/visualwebarena/phase1/B0_phantom_text_classifieds_20260427",
    REPO / "results/visualwebarena/phase1/B0_phantom_text_reddit_20260427",
    REPO / "results/visualwebarena/phase1/B1_3mode_classifieds_20260413",
    REPO / "results/visualwebarena/phase1/B1_3mode_reddit_20260413",
]


def _phantom_prompt_runs() -> list[Path]:
    """Return all available B0/B1 phantom_prompt run dirs (cls + red)."""
    out: list[Path] = []
    for baseline in ("B0", "B1"):
        for site in ("reddit", "classifieds"):
            for path in sorted((REPO / "results/visualwebarena/phase1").glob(
                f"{baseline}_phantom_prompt_{site}_*"
            )):
                if path.is_dir():
                    out.append(path)
    return out


# Auto-extend with any P-prompt runs that exist on disk
DEFAULT_RUNS = DEFAULT_RUNS + _phantom_prompt_runs()


def parse_run_id(run_dir: Path) -> tuple[str, str]:
    """Extract (baseline, site) from a run_id like B0_phantom_text_classifieds_20260427."""
    name = run_dir.name
    baseline = "B0" if name.startswith("B0") else ("B1" if name.startswith("B1") else "?")
    for site in ("classifieds", "reddit", "shopping_admin", "shopping"):
        if f"_{site}_" in name or name.endswith(f"_{site}"):
            return baseline, site
    m = re.search(r"_(classifieds|reddit|shopping_admin|shopping)", name)
    return baseline, m.group(1) if m else "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[str(p) for p in DEFAULT_RUNS],
                    help="run dirs to aggregate")
    ap.add_argument("--output", default=str(REPO / "results/phantom_paper/auroc_cross_condition.csv"))
    args = ap.parse_args()

    rows: list[pd.DataFrame] = []
    for run_str in args.runs:
        run_dir = Path(run_str)
        cm_path = run_dir / "analysis/signals/combined/tables/cross_mode_auroc.csv"
        single_path = run_dir / "analysis/signals/combined/tables/auroc_all_metrics.csv"
        baseline, site = parse_run_id(run_dir)
        if cm_path.exists():
            df = pd.read_csv(cm_path)
            if df.empty:
                continue
        elif single_path.exists():
            # Single-condition (e.g. phantom) runs — derive mode from condition dir name
            cond_dirs = [d for d in run_dir.glob("phase1_*") if d.is_dir()]
            if not cond_dirs:
                print(f"  [skip] {run_dir.name}: no condition dir")
                continue
            mode = cond_dirs[0].name.replace("phase1_", "").replace("_router_0", "")
            df = pd.read_csv(single_path).rename(columns={"metric": "signal"})
            df = df.assign(mode=mode)
        else:
            print(f"  [skip] {run_dir.name}: no AUROC tables")
            continue
        df = df.assign(baseline=baseline, site=site, run_id=run_dir.name)
        rows.append(df)

    if not rows:
        print("No AUROC data found in any run.")
        return 1

    full = pd.concat(rows, ignore_index=True)
    full = full[[
        "baseline", "site", "mode", "signal", "signal_type",
        "AUROC", "AUROC_ci_lower", "AUROC_ci_upper", "n", "run_id",
    ]]
    full = full.sort_values(["baseline", "site", "mode", "AUROC"], ascending=[True, True, True, False])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False)
    print(f"wrote {out} ({len(full)} rows)")

    # Markdown top-3 per (baseline, site, mode)
    md = out.with_suffix(".md")
    lines = [
        "# Cross-condition routing signal AUROC (top-3 per cell)",
        "",
        "AUROC ≥ 0.5 means signal correlates with success; CI from 1000-resample bootstrap.",
        "",
        "| Baseline | Site | Mode | Signal | AUROC | 95% CI | n |",
        "|---|---|---|---|---:|---|---:|",
    ]
    grouped = full.groupby(["baseline", "site", "mode"], dropna=False)
    for (b, s, m), grp in grouped:
        top = grp.nlargest(3, "AUROC")
        for _, r in top.iterrows():
            ci = ""
            if pd.notna(r["AUROC_ci_lower"]) and pd.notna(r["AUROC_ci_upper"]):
                ci = f"[{r['AUROC_ci_lower']:.3f}, {r['AUROC_ci_upper']:.3f}]"
            lines.append(
                f"| {b} | {s} | {m} | {r['signal']} | "
                f"{r['AUROC']:.3f} | {ci} | {int(r['n'])} |"
            )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md}")

    # Cross-mode max-AUROC summary (paper-ready Section 6 claim support)
    summary_md = out.parent / "auroc_cross_condition_summary.md"
    summary_lines = [
        "# Routing signal AUROC summary — max per (baseline, site, mode)",
        "",
        "Section 6 claim: AUROC ≥ baseline (DOM/SoM/Vision) for Phantom modes.",
        "",
        "| Baseline | Site | Mode | Max-AUROC signal | AUROC | 95% CI | n |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for (b, s, m), grp in grouped:
        top = grp.nlargest(1, "AUROC")
        if top.empty:
            continue
        r = top.iloc[0]
        ci = ""
        if pd.notna(r["AUROC_ci_lower"]) and pd.notna(r["AUROC_ci_upper"]):
            ci = f"[{r['AUROC_ci_lower']:.3f}, {r['AUROC_ci_upper']:.3f}]"
        summary_lines.append(
            f"| {b} | {s} | {m} | {r['signal']} | "
            f"{r['AUROC']:.3f} | {ci} | {int(r['n'])} |"
        )
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"wrote {summary_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
