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
        --output results/phantom_paper/auroc_cross_condition.csv

Output columns: baseline, site, mode, signal, signal_type, AUROC,
                AUROC_ci_lower, AUROC_ci_upper, n, run_id

A second markdown summary lands at <output>.md with a paper-ready table
showing top-3 signals per (baseline, site, mode).
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

try:
    from scripts.analysis.lib.run_registry import canonical_mode, get_run_dirs_paper_vwa
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.analysis.lib.run_registry import canonical_mode, get_run_dirs_paper_vwa


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = get_run_dirs_paper_vwa()


def parse_run_id(run_dir: Path) -> tuple[str, str]:
    """Extract (baseline, site) from a paper run id.

    v6 fix (P1-9, codex pre-fire #8): B2 (Gemma3-VL, added 2026-05-14) baseline parsing
    added. Previously B2-prefixed runs silently parsed as "?" leading to dropped or
    misattributed AUROC rows in the cross-condition table — paper §6 cross-baseline
    cost-aware evidence was structurally incomplete on B2 cells.
    """
    name = run_dir.name
    if name.startswith("B0"):
        baseline = "B0"
    elif name.startswith("B1"):
        baseline = "B1"
    elif name.startswith("B2"):
        baseline = "B2"
    else:
        baseline = "?"
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
            # Single-condition (e.g. phantom) runs — derive mode from condition dir name.
            # /stress A1.19 P1-7-A (2026-05-17, Claude): pre-fix used `cond_dirs[0].name`
            # first-match, silently mislabeling multi-condition runs (e.g., 3-mode runs
            # with DOM + SoM + Vision conditions) with only the first condition's mode.
            # Now: if exactly 1 condition, label correctly; if 0 or >1, fail-loud-skip
            # because single_path AUROC is run-level and can't be attributed to a single
            # mode without ambiguity.
            cond_dirs = [d for d in run_dir.glob("phase1_*") if d.is_dir()]
            if not cond_dirs:
                print(f"  [skip] {run_dir.name}: no condition dir")
                continue
            if len(cond_dirs) > 1:
                names = [d.name for d in cond_dirs]
                print(
                    f"  [skip] {run_dir.name}: single-path AUROC ambiguous — run "
                    f"contains {len(cond_dirs)} conditions {names}; cannot attribute "
                    f"`auroc_all_metrics.csv` to a unique mode without per-condition "
                    f"breakdown. Re-run analyze_confidence_calibration.py to emit "
                    f"per-condition `cross_mode_auroc.csv` instead."
                )
                continue
            mode = cond_dirs[0].name.replace("phase1_", "").replace("_router_0", "")
            df = pd.read_csv(single_path).rename(columns={"metric": "signal"})
            df = df.assign(mode=canonical_mode(mode))
        else:
            print(f"  [skip] {run_dir.name}: no AUROC tables")
            continue
        if "mode" in df.columns:
            df["mode"] = df["mode"].map(lambda value: canonical_mode(str(value)))
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

    # B-1056 (/stress A2.3c Mode A F5 + Mode B B4, 2026-05-18): per-(mode, signal)
    # AUROC family-corrected multiplicity emission. Pre-fix file emitted only
    # AUROC + CI; prereg §3 line 423 + §4 line 424 promise "Holm-corrected
    # within mode" + "BH q-value reported for transparency". Producer never
    # emitted p-values, making Holm/BH mathematically impossible to apply
    # downstream → prereg promise vapor.
    #
    # Approach (Hanley-McNeil 1982 standard for AUROC null testing):
    #   1. SE_AUROC ≈ (CI_upper - CI_lower) / (2 × 1.96) [normal-approx from CI]
    #   2. Z = (AUROC - 0.5) / SE_AUROC [H0: AUROC ≤ 0.5 = random]
    #   3. p_one_sided = 1 - Φ(Z) [upper-tail]
    #   4. Holm within each (baseline, site, mode) group (per prereg §4 line 423)
    #   5. BH-FDR across full exploratory family (per prereg §4 line 424)
    #
    # Mode B P0-5-B catch: AUROC family had NO p-values to correct at all;
    # this addition enables the prereg-promised multiplicity transparency.

    def _phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

    def _derive_p_one_sided(row):
        auroc = row.get("AUROC")
        ci_lo = row.get("AUROC_ci_lower")
        ci_hi = row.get("AUROC_ci_upper")
        if pd.isna(auroc) or pd.isna(ci_lo) or pd.isna(ci_hi):
            return None
        se = (ci_hi - ci_lo) / (2 * 1.96)
        if se <= 0:
            return 1.0 if auroc <= 0.5 else 0.0  # degenerate
        z = (auroc - 0.5) / se
        return float(1.0 - _phi(z))

    full["p_one_sided"] = full.apply(_derive_p_one_sided, axis=1)

    def _holm_within(group):
        # Holm-Bonferroni step-down per prereg §4 line 423 (within mode)
        ps = group["p_one_sided"].tolist()
        n = len(ps)
        if n == 0:
            return group.assign(p_holm=None, holm_m=0)
        indexed = sorted(((i, p) for i, p in enumerate(ps) if p is not None), key=lambda x: x[1])
        p_holm = [None] * n
        prev_adj = 0.0
        for rank, (orig_idx, p) in enumerate(indexed):
            adj = min(1.0, max(prev_adj, p * (n - rank)))
            p_holm[orig_idx] = adj
            prev_adj = adj
        return group.assign(p_holm=p_holm, holm_m=n)

    # pandas 2.2 groupby.apply EXCLUDES grouping columns from the result (deprecated
    # behavior → FutureWarning "grouping columns will be excluded"). Back up keys +
    # restore index-aligned (_holm_within keeps row count + index labels, so label-
    # aligned assignment is safe). Exposed by first paper-grade manifest promote
    # 2026-05-28; reproduces at full k_cells=6 (pandas-version bug, not partial-specific).
    _keys_bak = full[["baseline", "site", "mode"]].copy()
    full = full.groupby(["baseline", "site", "mode"], group_keys=False).apply(_holm_within)
    for _k in ("baseline", "site", "mode"):
        if _k not in full.columns:
            full[_k] = _keys_bak[_k]

    # BH-FDR across full exploratory family (per prereg §4 line 424)
    # Benjamini-Hochberg 1995: for ranks i=1..N sorted by p ascending,
    # q_i = min over k>=i of (p_k * N / k); enforce monotone non-decreasing
    valid_ps = [(i, p) for i, p in enumerate(full["p_one_sided"].tolist()) if p is not None]
    valid_ps.sort(key=lambda x: x[1])
    n_bh = len(valid_ps)
    q_bh_list = [None] * len(full)
    if n_bh > 0:
        # Compute raw BH q from highest rank down (enforce monotone)
        running_min = 1.0
        for rev_rank in range(n_bh - 1, -1, -1):
            orig_idx, p = valid_ps[rev_rank]
            q_raw = p * n_bh / (rev_rank + 1)
            running_min = min(running_min, q_raw)
            q_bh_list[orig_idx] = min(1.0, running_min)
    full["q_bh"] = q_bh_list
    full["bh_family_N"] = n_bh

    # pandas 2.x groupby(...).apply (line ~183) can push group keys (baseline/site/mode)
    # to the index; restore as columns before sort references them. (Exposed by first
    # paper-grade manifest promote 2026-05-28; reproduces at full k_cells=6 too.)
    if any(k not in full.columns for k in ("baseline", "site", "mode")):
        full = full.reset_index()
    full = full.sort_values(["baseline", "site", "mode", "AUROC"], ascending=[True, True, True, False])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # B-1056 ensures CSV includes p_one_sided, p_holm, holm_m, q_bh, bh_family_N
    # columns (added above). Downstream readers can now diff prereg §4 line
    # 423-424 promise against artifact CSV + find Holm/BH q-values per-row.
    full.to_csv(out, index=False)
    print(f"wrote {out} ({len(full)} rows; B-1056 p_one_sided + p_holm + q_bh emitted)")

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
