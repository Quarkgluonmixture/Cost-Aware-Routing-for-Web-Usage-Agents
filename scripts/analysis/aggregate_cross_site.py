#!/usr/bin/env python3
"""[Efficiency 3a-3c] Efficiency dimension — cross-site cost, latency, and SR aggregation.

Outputs:
- results/phantom_paper/cross_site/cross_site_aggregation.csv
- results/phantom_paper/cross_site/cross_site_summary.json
- results/phantom_paper/cross_site/cross_site_{cost,sr}_comparison.png

Efficiency 3a token/cost, 3b image embedding support, and 3c latency inputs.

See docs/checkpoints/paper_planning.md §3 Efficiency dimension framework.

Cross-site aggregation for Phase 1 experiments.

Reads condition_summary_v2.json from multiple run directories (one per site),
computes cross-site SR / cost / visual-task statistics, and writes comparison
tables and plots.

Usage:
    python3 scripts/analysis/aggregate_cross_site.py \\
        --run-dirs results/visualwebarena/phase1/<cls_run> \\
                   results/visualwebarena/phase1/<reddit_run> \\
                   results/visualwebarena/phase1/<shopping_run> \\
        [--output-dir results/visualwebarena/phase1/cross_site_analysis/] \\
        [--b1-label "B1"] \\
        [--no-adjusted]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

try:
    from scripts.analysis.lib.run_registry import PAPER_MODES, canonical_mode, get_run_dirs_paper_vwa
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.analysis.lib.run_registry import PAPER_MODES, canonical_mode, get_run_dirs_paper_vwa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _detect_site(run_dir: Path) -> str:
    """Infer site name from run_id or condition_meta files."""
    # Try condition_meta
    for p in run_dir.glob("*/condition_meta.json"):
        try:
            d = _read_json(p)
            # condition_meta may have benchmark_site or site embedded in run_id
            run_id = d.get("run_id", "")
            for site in ("classifieds", "reddit", "shopping"):
                if site in run_id.lower() or site in run_dir.name.lower():
                    return site
        except Exception:
            pass
    # Fallback: check run_dir name
    for site in ("classifieds", "reddit", "shopping"):
        if site in run_dir.name.lower():
            return site
    # Last resort: look at episode files
    for p in run_dir.glob("*/episodes/*_summary_v2.json"):
        try:
            d = _read_json(p)
            s = d.get("benchmark_site", "")
            if s:
                return s
        except Exception:
            pass
    return run_dir.name


def _extract_stub_adjusted_sr(stub_note: str) -> Optional[float]:
    """Parse 'Adjusted SR=19/234' pattern from stub note."""
    m = re.search(r"[Aa]djusted SR[=\s]*([\d.]+)[/\s]*([\d]+)", stub_note)
    if m:
        return float(m.group(1)) / float(m.group(2))
    m2 = re.search(r"[Aa]djusted SR[=\s]*([0-9.]+)%", stub_note)
    if m2:
        return float(m2.group(1)) / 100.0
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_condition_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all condition_summary_v2.json files from run_dir."""
    summaries = []
    for p in run_dir.glob("*/condition_summary_v2.json"):
        try:
            d = _read_json(p)
            d["_source_file"] = str(p)
            summaries.append(d)
        except Exception as e:
            print(f"  [WARN] Cannot read {p}: {e}")
    return summaries


def load_fp_stats(run_dir: Path) -> Dict[str, Any]:
    """Try to load cross_representation_summary.json for FP stats."""
    for pattern in [
        "analysis/results/cross_representation/cross_representation_summary.json",
        "analysis/results/cross_representation/*/cross_representation_summary.json",
    ]:
        for p in run_dir.glob(pattern):
            try:
                return _read_json(p)
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

MODES = PAPER_MODES

# Adjusted SR regex patterns for stub notes
_STUB_ADJ_RE = re.compile(r"[Aa]djusted SR[=:\s]*([\d]+)[/\s]*([\d]+)")


def _get_adjusted_sr(
    cond: Dict[str, Any],
    cross_rep_per_site: Optional[Dict[str, Dict[str, float]]] = None,
    site: Optional[str] = None,
) -> Optional[float]:
    """Legacy archive-only adjusted SR extractor.

    B-405 (/stress A1.1 v8 Mode B P2-4, 2026-05-16): the post-hoc
    `adjusted_success` / `compute_adjusted_success` layer was retired in
    §139.8 — `success` is now the canonical paper-grade outcome (N/A
    exclusion happens at task load via `task.exclude_na_tasks=true`
    default + upstream B-91 evaluator empty-pred guard fix). This helper
    is preserved for **archive-only** reading of pre-§139.8
    `cross_representation_summary.json` artifacts and stub notes — any
    post-§139.8 run will see `cross_rep_per_site=None` and stub-less
    summaries, so this function returns None for all current data.
    Output column `adjusted_sr` is therefore label-as-legacy in
    `aggregate_run_dir`. Do NOT re-introduce post-hoc FP adjustment as
    a paper claim.

    Non-stub conditions don't carry adjusted SR in condition_summary_v2.json
    (it's computed by analyze_cross_representation.py separately). Look it up
    in the cross_rep_per_site map if provided.
    """
    if cond.get("_stub"):
        note = cond.get("_stub_note", "")
        return _extract_stub_adjusted_sr(note)
    # Cross-rep override: per_site[site].per_mode_sr_adjusted[mode]
    if cross_rep_per_site and site:
        site_block = cross_rep_per_site.get(site) or {}
        per_mode_adj = site_block.get("per_mode_sr_adjusted") or {}
        mode = cond.get("observation_mode", "")
        if mode in per_mode_adj:
            return float(per_mode_adj[mode])
    return None


def aggregate_run_dir(run_dir: Path, site: str, label: str) -> List[Dict[str, Any]]:
    """Extract per-mode rows from a single run_dir."""
    summaries = load_condition_summaries(run_dir)
    if not summaries:
        print(f"  [WARN] No condition summaries in {run_dir}")
        return []

    # Load cross_representation_summary.json for adjusted_sr — it's where the
    # §95 FP-filtered numbers live (condition_summary_v2.json only carries raw).
    fp_stats = load_fp_stats(run_dir)
    cross_rep_per_site = fp_stats.get("per_site") if isinstance(fp_stats, dict) else None

    rows = []
    for cond in summaries:
        mode = canonical_mode(str(cond.get("observation_mode", "")))
        if not mode:
            continue
        is_stub = bool(cond.get("_stub"))
        raw_sr = float(cond.get("success_rate", 0.0))
        adj_sr = _get_adjusted_sr(cond, cross_rep_per_site=cross_rep_per_site, site=site)
        if is_stub:
            print(f"  [STUB] site={site} mode={mode} raw_sr={raw_sr:.3f} adj_sr={adj_sr}")

        # B-405 (/stress A1.1 v8 Mode B P2-4, 2026-05-16): post-§139.8 the
        # `adjusted_success` post-hoc FP-filter layer is retired (paper-grade
        # canonical = `success`; N/A excluded at task-load; B-91 evaluator
        # guard at source). `adjusted_sr` here is an archive-only field for
        # pre-§139.8 run dirs; surface a one-line warning when populated so
        # paper-grade callers cannot accidentally cite it without notice.
        if adj_sr is not None:
            print(
                f"  [B-405 legacy-archive] site={site} mode={mode} "
                f"adjusted_sr={adj_sr:.4f} sourced from pre-§139.8 archive "
                f"(do NOT cite in paper §1/§3 — `raw_sr` is canonical)",
                file=sys.stderr,
            )
        rows.append({
            "label": label,
            "site": site,
            "mode": mode,
            "raw_sr": round(raw_sr, 4),
            # B-405: `adjusted_sr` retained for archive comparison only.
            # Paper-grade post-§139.8 cite `raw_sr`. Future v3 schema bump
            # can drop the column entirely once cross_representation_summary
            # legacy artifacts no longer need to round-trip.
            "adjusted_sr": round(adj_sr, 4) if adj_sr is not None else None,
            "avg_cost_usd": round(float(cond.get("avg_total_cost_usd", 0.0)), 6),
            "avg_steps": round(float(cond.get("avg_steps", 0.0)), 2),
            "avg_total_energy_kwh": cond.get("avg_total_energy_kwh"),
            "episodes": int(cond.get("episodes", 0)),
            "is_stub": is_stub,
        })
    return rows


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not HAS_PD:
        import csv
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_grouped_bar(
    data: List[Dict[str, Any]],
    x_key: str,
    group_key: str,
    value_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    label_prefix: str = "",
) -> None:
    """Generic grouped bar chart (x=sites, groups=modes)."""
    if not HAS_MPL or not data:
        return
    import numpy as np

    xs = sorted(set(d[x_key] for d in data))
    groups = sorted(set(d[group_key] for d in data))
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)
    x = np.arange(len(xs))

    fig, ax = plt.subplots(figsize=(max(6, len(xs) * 1.8), 5))
    for i, grp in enumerate(groups):
        vals = []
        for site in xs:
            match = [d for d in data if d[x_key] == site and d[group_key] == grp]
            vals.append(float(match[0][value_key]) if match and match[0].get(value_key) is not None else 0.0)
        ax.bar(x + i * width, vals, width, label=f"{label_prefix}{grp}")

    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(xs, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if "sr" in value_key.lower() or "rate" in value_key.lower():
        ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-site aggregation for Phase 1 experiments"
    )
    parser.add_argument(
        "--run-dirs", nargs="+", default=None,
        help="Run directories (default: paper VWA runs from run_manifest.yaml)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: results/visualwebarena/phase1/cross_site_analysis/)",
    )
    parser.add_argument("--b1-label", default="B1", help="Label for plot titles")
    parser.add_argument(
        "--no-adjusted", action="store_true",
        help="Do not attempt to use adjusted SR (stubs only have raw data anyway)",
    )
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs] if args.run_dirs else get_run_dirs_paper_vwa()
    for rd in run_dirs:
        if not rd.is_dir():
            print(f"[ERROR] Not a directory: {rd}")
            sys.exit(1)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results/visualwebarena/phase1/cross_site_analysis")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    use_adjusted = not args.no_adjusted

    # --- Collect rows ---
    print("[1/4] Loading condition summaries...")
    all_rows: List[Dict[str, Any]] = []
    site_meta: List[Dict[str, Any]] = []
    for rd in run_dirs:
        site = _detect_site(rd)
        print(f"  run_dir={rd.name}  site={site}")
        rows = aggregate_run_dir(rd, site, label=args.b1_label)
        all_rows.extend(rows)
        episodes = rows[0]["episodes"] if rows else 0
        site_meta.append({"site": site, "run_dir": str(rd), "n_modes": len(rows), "episodes": episodes})

    if not all_rows:
        print("[ERROR] No data collected. Check --run-dirs.")
        sys.exit(1)

    # --- cross_site_aggregation.csv ---
    print("[2/4] Writing cross_site_aggregation.csv...")
    sr_col = "adjusted_sr" if use_adjusted else "raw_sr"
    aggregation_rows = []
    for r in all_rows:
        sr_val = r.get("adjusted_sr") if use_adjusted else r.get("raw_sr")
        if sr_val is None:
            sr_val = r.get("raw_sr")  # fallback
        aggregation_rows.append({
            "site": r["site"],
            "mode": r["mode"],
            "raw_sr": r["raw_sr"],
            "adjusted_sr": r.get("adjusted_sr"),
            "sr_used": round(float(sr_val), 4) if sr_val is not None else None,
            "avg_cost_usd": r["avg_cost_usd"],
            "avg_steps": r["avg_steps"],
            "avg_total_energy_kwh": r.get("avg_total_energy_kwh"),
            "episodes": r["episodes"],
            "is_stub": r["is_stub"],
        })
    _save_csv(aggregation_rows, out_dir / "cross_site_aggregation.csv")

    # --- cross_site_sr_comparison.png ---
    print("[3/4] Generating plots...")
    _plot_grouped_bar(
        data=aggregation_rows,
        x_key="site",
        group_key="mode",
        value_key="sr_used",
        title=f"{args.b1_label} Phase 1 — Success Rate by Site × Mode"
              + (" (adjusted)" if use_adjusted else " (raw)"),
        ylabel="Success Rate",
        out_path=out_dir / "cross_site_sr_comparison.png",
    )

    _plot_grouped_bar(
        data=aggregation_rows,
        x_key="site",
        group_key="mode",
        value_key="avg_cost_usd",
        title=f"{args.b1_label} Phase 1 — Avg Cost (USD) by Site × Mode",
        ylabel="Avg Cost per Episode (USD)",
        out_path=out_dir / "cross_site_cost_comparison.png",
    )

    # --- cross_site_summary.json ---
    print("[4/4] Writing summary JSON...")
    sites = sorted(set(r["site"] for r in all_rows))
    per_site: Dict[str, Any] = {}
    for site in sites:
        site_rows = [r for r in all_rows if r["site"] == site]
        per_site[site] = {
            m: {
                "raw_sr": next((r["raw_sr"] for r in site_rows if r["mode"] == m), None),
                "adjusted_sr": next((r.get("adjusted_sr") for r in site_rows if r["mode"] == m), None),
                "avg_cost_usd": next((r["avg_cost_usd"] for r in site_rows if r["mode"] == m), None),
                "avg_steps": next((r["avg_steps"] for r in site_rows if r["mode"] == m), None),
            }
            for m in MODES
            if any(r["mode"] == m for r in site_rows)
        }

    # Weighted-average SR across sites (equal weight per site)
    weighted_sr: Dict[str, List[float]] = {}
    for r in all_rows:
        m = r["mode"]
        sr = r.get("adjusted_sr") if use_adjusted else r.get("raw_sr")
        if sr is not None:
            weighted_sr.setdefault(m, []).append(float(sr))
    global_avg_sr = {
        m: round(sum(vals) / len(vals), 4) for m, vals in weighted_sr.items()
    }

    summary = {
        "label": args.b1_label,
        "sites": sites,
        "use_adjusted_sr": use_adjusted,
        "per_site": per_site,
        "global_avg_sr_per_mode": global_avg_sr,
        "outputs": [f.name for f in sorted(out_dir.iterdir()) if f.is_file()],
    }
    _write_json(summary, out_dir / "cross_site_summary.json")

    print(f"\nDone! Outputs in: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
