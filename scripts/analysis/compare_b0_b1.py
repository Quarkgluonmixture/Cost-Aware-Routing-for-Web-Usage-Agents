#!/usr/bin/env python3
"""Per-run diagnostic; not part of the 4-dimension evidence framework.

Cross-baseline Phase 1 comparison script (B0 vs B1 vs B2).

3-model deep-update 2026-05-18: B2 (Gemma3-VL, google/gemma-3-4b-it, cross-family
4B local) added to scope alongside B0 (Qwen3-VL-235B API) + B1 (Qwen3-VL-4B local).
B2 input optional — when absent, script falls back to legacy B0 vs B1 mode.

Reads condition_summary_v2.json from each provided run directory and computes
SR / cost / Mirage Effect comparisons.

Usage:
    python3 scripts/analysis/compare_b0_b1.py \\
        --b0-run-dir results/visualwebarena/phase1/<b0_classifieds_run> \\
        --b1-run-dir results/visualwebarena/phase1/<b1_classifieds_run> \\
        [--b2-run-dir results/visualwebarena/phase1/<b2_classifieds_run>] \\
        [--site classifieds] \\
        [--output-dir results/visualwebarena/phase1/b0_vs_b1_vs_b2/] \\
        [--use-legacy-adjusted]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _extract_stub_adjusted_sr(stub_note: str) -> Optional[float]:
    """Parse 'Adjusted SR=19/234' pattern from stub note."""
    m = re.search(r"[Aa]djusted SR[=:\s]*([\d]+)[/\s]*([\d]+)", stub_note)
    if m:
        return float(m.group(1)) / float(m.group(2))
    m2 = re.search(r"[Aa]djusted SR[=\s]*([0-9.]+)%", stub_note)
    if m2:
        return float(m2.group(1)) / 100.0
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

MODES = ["dom", "som", "vision"]


def load_conditions(run_dir: Path, model_label: str) -> Dict[str, Dict[str, Any]]:
    """Load condition summaries keyed by observation_mode.

    Returns {mode: cond_dict} for all modes found in run_dir. Also attempts
    to enrich each cond with adjusted_sr from cross_representation_summary.json
    so non-stub conditions get a real adjusted SR (was: always None).
    """
    result: Dict[str, Dict[str, Any]] = {}
    for p in run_dir.glob("*/condition_summary_v2.json"):
        try:
            d = _read_json(p)
            mode = d.get("observation_mode", "")
            if mode:
                d["_model_label"] = model_label
                result[mode] = d
        except Exception as e:
            print(f"  [WARN] Cannot read {p}: {e}")

    # Enrich with cross_representation adjusted_sr (per-mode).
    cross_rep_per_site: Optional[Dict[str, Any]] = None
    for pattern in [
        "analysis/results/cross_representation/cross_representation_summary.json",
        "analysis/results/cross_representation/*/cross_representation_summary.json",
    ]:
        for cp in run_dir.glob(pattern):
            try:
                cross_rep_per_site = _read_json(cp).get("per_site")
                break
            except Exception:
                pass
        if cross_rep_per_site:
            break
    if cross_rep_per_site:
        for site_block in cross_rep_per_site.values():
            if not isinstance(site_block, dict):
                continue
            per_mode_adj = site_block.get("per_mode_sr_adjusted") or {}
            for mode, adj in per_mode_adj.items():
                if mode in result:
                    result[mode]["_adjusted_sr_xrep"] = float(adj)
            # Single-site run: stop after first site block with data.
            if per_mode_adj:
                break
    return result


def _get_adjusted_sr(cond: Dict[str, Any]) -> Optional[float]:
    if cond.get("_stub"):
        return _extract_stub_adjusted_sr(cond.get("_stub_note", ""))
    # Non-stub: pulled from cross_representation_summary.json by load_conditions.
    if "_adjusted_sr_xrep" in cond:
        return float(cond["_adjusted_sr_xrep"])
    return None


def _effective_sr(cond: Dict[str, Any], use_adjusted: bool) -> float:
    """Return the best available SR (adjusted if requested and available, else raw)."""
    raw = float(cond.get("success_rate", 0.0))
    if not use_adjusted:
        return raw
    adj = _get_adjusted_sr(cond)
    return adj if adj is not None else raw


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------

def build_comparison_rows(
    baseline_conds: Dict[str, Dict[str, Dict]],
    use_adjusted: bool,
) -> List[Dict[str, Any]]:
    """Build cross-baseline comparison CSV rows.

    3-model deep-update 2026-05-18: was 2-baseline hardcoded; now iterates
    arbitrary `baseline_conds` mapping {label → {mode → cond}}.
    """
    rows = []
    all_modes = sorted({mode for conds in baseline_conds.values() for mode in conds.keys()})
    for model_label, cond_map in baseline_conds.items():
        for mode in all_modes:
            cond = cond_map.get(mode)
            if cond is None:
                rows.append({
                    "model": model_label, "mode": mode,
                    "raw_sr": None, "adjusted_sr": None, "sr_used": None,
                    "avg_cost_usd": None, "avg_steps": None, "episodes": None,
                    "is_stub": None,
                })
                continue
            raw_sr = float(cond.get("success_rate", 0.0))
            adj_sr = _get_adjusted_sr(cond)
            sr_used = _effective_sr(cond, use_adjusted)
            rows.append({
                "model": model_label,
                "mode": mode,
                "raw_sr": round(raw_sr, 4),
                "adjusted_sr": round(adj_sr, 4) if adj_sr is not None else None,
                "sr_used": round(sr_used, 4),
                "avg_cost_usd": round(float(cond.get("avg_total_cost_usd", 0.0)), 6),
                "avg_steps": round(float(cond.get("avg_steps", 0.0)), 2),
                "episodes": int(cond.get("episodes", 0)),
                "is_stub": bool(cond.get("_stub", False)),
            })
    return rows


def build_mirage_rows(
    baseline_conds: Dict[str, Dict[str, Dict]],
    use_adjusted: bool,
) -> List[Dict[str, Any]]:
    """Build per-baseline SoM−DOM gap rows (Mirage Effect diagnostic)."""
    rows = []
    for model_label, cond_map in baseline_conds.items():
        som_cond = cond_map.get("som")
        dom_cond = cond_map.get("dom")
        if som_cond is None or dom_cond is None:
            rows.append({
                "model": model_label,
                "som_sr": None, "dom_sr": None, "mirage_gap": None,
            })
            continue
        som_sr = _effective_sr(som_cond, use_adjusted)
        dom_sr = _effective_sr(dom_cond, use_adjusted)
        rows.append({
            "model": model_label,
            "som_sr": round(som_sr, 4),
            "dom_sr": round(dom_sr, 4),
            "mirage_gap": round(som_sr - dom_sr, 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_comparison(
    comparison_rows: List[Dict[str, Any]],
    models: List[str],
    site: str,
    label: str,
    use_adjusted: bool,
    out_path: Path,
) -> None:
    """Grouped bar chart: models × modes for SR and cost (dual axis).

    3-model deep-update 2026-05-18: `models` is now an arg (was hardcoded
    ["B0","B1"]); offset math reworked for N-baseline even spacing.
    """
    if not HAS_MPL or not HAS_NP:
        return

    modes = MODES
    n_modes = len(modes)
    n_models = len(models)
    width = min(0.35, 0.85 / max(n_models, 1))
    x = np.arange(n_modes)

    sr_by_model: Dict[str, List[float]] = {}
    cost_by_model: Dict[str, List[float]] = {}
    for mdl in models:
        sr_by_model[mdl] = []
        cost_by_model[mdl] = []
        for mode in modes:
            row = next((r for r in comparison_rows if r["model"] == mdl and r["mode"] == mode), None)
            sr_by_model[mdl].append(float(row["sr_used"] or 0) if row else 0.0)
            cost_by_model[mdl].append(float(row["avg_cost_usd"] or 0) if row else 0.0)

    sr_label = "Adjusted SR" if use_adjusted else "Raw SR"
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    for i, mdl in enumerate(models):
        offset = (i - (n_models - 1) / 2) * width
        ax1.bar(x + offset, sr_by_model[mdl], width, label=f"{mdl} {sr_label}", alpha=0.8)
        ax2.plot(
            x + offset, cost_by_model[mdl], "D--",
            label=f"{mdl} Avg Cost", markersize=6,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, fontsize=11)
    ax1.set_ylabel(sr_label, color="tab:blue")
    ax2.set_ylabel("Avg Cost (USD)", color="tab:orange")
    ax1.set_title(f"{label} {' vs '.join(models)} — {site} Phase 1")
    ax1.set_ylim(0, max(0.4, ax1.get_ylim()[1]))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_mirage_effect(
    mirage_rows: List[Dict[str, Any]],
    site: str,
    label: str,
    out_path: Path,
) -> None:
    """Horizontal bar chart of SoM-DOM gap per model."""
    if not HAS_MPL or not HAS_NP:
        return

    valid = [r for r in mirage_rows if r.get("mirage_gap") is not None]
    if not valid:
        return

    models = [r["model"] for r in valid]
    gaps = [float(r["mirage_gap"]) for r in valid]

    fig, ax = plt.subplots(figsize=(7, max(3, len(valid) * 0.8)))
    y = np.arange(len(models))
    ax.barh(y, gaps, color=["tab:blue" if g >= 0 else "tab:red" for g in gaps])
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SoM SR − DOM SR (Mirage Gap)")
    ax.set_title(f"{label} Mirage Effect (SoM−DOM) — {site}")
    for i, (g, m) in enumerate(zip(gaps, models)):
        ax.text(g + 0.002, i, f"{g:+.1%}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Phase 1 results across baselines (B0/B1/B2)"
    )
    parser.add_argument("--b0-run-dir", required=True, help="B0 (Qwen3-VL-235B API) run directory")
    parser.add_argument("--b1-run-dir", required=True, help="B1 (Qwen3-VL-4B local) run directory")
    parser.add_argument(
        "--b2-run-dir", default=None,
        help="B2 (Gemma3-VL, optional) run directory. When omitted, script "
             "operates in legacy B0 vs B1 mode for back-compat.",
    )
    parser.add_argument("--site", default="classifieds", help="Site name (default: classifieds)")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: results/visualwebarena/phase1/<combo>/ "
             "where <combo> = b0_vs_b1 or b0_vs_b1_vs_b2 depending on inputs)",
    )
    parser.add_argument(
        "--use-legacy-adjusted", action="store_true",
        help="OPT-IN for legacy adjusted_sr (post-§139.8 RETIRED, archive Appendix D "
             "sensitivity only). Default = raw `success` is canonical (paper-grade). "
             "/stress A1.19 P0-2 (2026-05-17, codex Mode B P1-6-B sibling propagation): "
             "flipped semantics from --no-adjusted to enforce §139.8 retirement spec.",
    )
    parser.add_argument(
        "--no-adjusted", action="store_true",
        help="DEPRECATED — no-op (raw is now default; use --use-legacy-adjusted to opt into archive).",
    )
    args = parser.parse_args()

    # 3-model deep-update 2026-05-18: build label→run_dir dict; B2 optional.
    baseline_runs: Dict[str, Path] = {"B0": Path(args.b0_run_dir), "B1": Path(args.b1_run_dir)}
    if args.b2_run_dir:
        baseline_runs["B2"] = Path(args.b2_run_dir)
    for label, rd in baseline_runs.items():
        if not rd.is_dir():
            print(f"[ERROR] {label} not a directory: {rd}")
            sys.exit(1)

    # Output dir naming: legacy b0_vs_b1 when 2-baseline; b0_vs_b1_vs_b2 when 3.
    default_combo = "_vs_".join(b.lower() for b in baseline_runs.keys())
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"results/visualwebarena/phase1/{default_combo}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # /stress A1.19 P0-2 (2026-05-17): default = raw `success` canonical (§139.8 retire);
    # legacy adjusted only via explicit opt-in. `--no-adjusted` retained as no-op.
    use_adjusted = args.use_legacy_adjusted
    site = args.site

    # --- Load data ---
    print("[1/4] Loading condition summaries...")
    baseline_conds: Dict[str, Dict[str, Dict[str, Any]]] = {
        label: load_conditions(rd, label) for label, rd in baseline_runs.items()
    }

    for label, conds in baseline_conds.items():
        print(f"  {label} modes: {sorted(conds.keys())}")
        for mode, cond in conds.items():
            if cond.get("_stub"):
                sr = cond.get("success_rate")
                adj = _get_adjusted_sr(cond)
                print(f"  [STUB] {label} {mode}: raw={sr:.3f} adj={adj}")

    # --- Build tables ---
    print("[2/4] Building comparison tables...")
    comparison_rows = build_comparison_rows(baseline_conds, use_adjusted)
    mirage_rows = build_mirage_rows(baseline_conds, use_adjusted)

    combo_prefix = "_vs_".join(baseline_runs.keys())
    _save_csv(comparison_rows, out_dir / f"{combo_prefix}_comparison.csv")
    _save_csv(mirage_rows, out_dir / f"{combo_prefix}_mirage_effect.csv")

    # --- Plots ---
    print("[3/4] Generating plots...")
    _plot_comparison(
        comparison_rows, models=list(baseline_runs.keys()),
        site=site, label=f"Phase1 {site}",
        use_adjusted=use_adjusted,
        out_path=out_dir / f"{combo_prefix}_comparison.png",
    )
    _plot_mirage_effect(
        mirage_rows, site=site, label=f"Phase1 {site}",
        out_path=out_dir / f"{combo_prefix}_mirage_effect.png",
    )

    # --- Summary JSON ---
    print("[4/4] Writing summary JSON...")
    summary: Dict[str, Any] = {
        "site": site,
        "use_adjusted_sr": use_adjusted,
        "baseline_runs": {label: str(rd) for label, rd in baseline_runs.items()},
        "modes_per_baseline": {label: sorted(c.keys()) for label, c in baseline_conds.items()},
        "comparison": {
            label: {
                cr["mode"]: {
                    "sr_used": cr.get("sr_used"),
                    "raw_sr": cr.get("raw_sr"),
                    "adjusted_sr": cr.get("adjusted_sr"),
                    "avg_cost_usd": cr.get("avg_cost_usd"),
                    "avg_steps": cr.get("avg_steps"),
                }
                for cr in comparison_rows if cr["model"] == label
            }
            for label in baseline_runs.keys()
        },
        "mirage_effect": {
            mr["model"]: {
                "som_sr": mr.get("som_sr"),
                "dom_sr": mr.get("dom_sr"),
                "mirage_gap": mr.get("mirage_gap"),
            }
            for mr in mirage_rows
        },
        "outputs": [f.name for f in sorted(out_dir.iterdir()) if f.is_file()],
    }
    summary_name = f"{combo_prefix.lower()}_summary.json"
    _write_json(summary, out_dir / summary_name)

    print(f"\nDone! Outputs in: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
