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


def _detect_baseline_from_run_dir(run_dir: Path) -> str:
    """A1.21 P1-2 fix (B-531): infer baseline from run_dir name prefix (B0/B1/B2).

    Naming convention: `B0_3mode_classifieds_<date>` / `B1_phantom_som_reddit_<date>` etc.
    Returns 'unknown' if no canonical prefix detected.
    """
    name = run_dir.name
    for b in ("B0", "B1", "B2"):
        if name.startswith(f"{b}_"):
            return b
    return "unknown"


def aggregate_run_dir(run_dir: Path, site: str, label: str) -> List[Dict[str, Any]]:
    """Extract per-mode rows from a single run_dir."""
    summaries = load_condition_summaries(run_dir)
    if not summaries:
        print(f"  [WARN] No condition summaries in {run_dir}")
        return []
    baseline = _detect_baseline_from_run_dir(run_dir)  # A1.21 P1-2

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
        # B-1409 (/stress A2.7 P1-8-B* codex Mode B OOB, 2026-05-18): propagate
        # `cost_unit_basis` from condition_summary so downstream aggregation +
        # plotting can stratify before pooling. Pre-fix this aggregator wrote
        # `avg_cost_usd` from `avg_total_cost_usd` with no basis tag — B0 (api_usd)
        # and B1/B2 (electricity_usd_derived) rows were silently put into the
        # same column. Per paper §3.5.1 footnote `cost-basis-cross-baseline`
        # the cross-baseline cost is always reported as per-baseline ratios
        # under a single basis, never as cross-baseline pooled absolute USD.
        # `cost_unit_basis` is computed at runner step_record stamp (B-563
        # /stress A1.22 P0-4-A* + B-564 /stress A1.22 P0-5-A*); condition-level
        # rollup is the modal basis across the condition's steps (set at
        # `aggregate_condition_metrics`).
        _basis = cond.get("cost_unit_basis")
        rows.append({
            "label": label,
            "baseline": baseline,  # A1.21 P1-2 (B-531): baseline propagation
            "site": site,
            "mode": mode,
            "raw_sr": round(raw_sr, 4),
            # B-405: `adjusted_sr` retained for archive comparison only.
            # Paper-grade post-§139.8 cite `raw_sr`. Future v3 schema bump
            # can drop the column entirely once cross_representation_summary
            # legacy artifacts no longer need to round-trip.
            "adjusted_sr": round(adj_sr, 4) if adj_sr is not None else None,
            "avg_cost_usd": round(float(cond.get("avg_total_cost_usd", 0.0)), 6),
            # B-1409: carry basis tag so downstream cross-baseline pooling
            # is stratify-able. Aggregators / figure scripts that pool across
            # baselines MUST partition rows by `cost_unit_basis` before
            # computing any pooled mean / table cell.
            "cost_unit_basis": _basis if isinstance(_basis, str) else "unknown",
            "avg_steps": round(float(cond.get("avg_steps", 0.0)), 2),
            "avg_total_energy_kwh": cond.get("avg_total_energy_kwh"),
            # B-1410 (/stress A2.7 P1-5-AB* 2-AI overlap A+B, 2026-05-18):
            # canonical cross-baseline latency = retry-adjusted per §3.5.1
            # B-1402 framework. Both `avg_total_latency_ms` (raw, sensitivity)
            # and `avg_total_latency_minus_retry_ms` (canonical) carried so
            # downstream plot/table scripts can pick either as primary +
            # the other as sensitivity column. `None` until the runner
            # episode-summary rollup write path lands post-parallel-merge.
            "avg_total_latency_ms": cond.get("avg_total_latency_ms"),
            "avg_total_latency_minus_retry_ms": cond.get("avg_total_latency_minus_retry_ms"),
            # B-1669 (/stress A2.11 P0-4-C 2026-05-18, user Q6=A): canonical
            # latency now ALSO subtracts VWA env busy_wait (page settle / cold
            # cache stalls, e.g., 99s busy:1 wait per p79/experiment/runner/
            # main.py:2122). Pre-fix `avg_total_latency_minus_retry_ms` only
            # subtracted B0 proxy retry scaffold (B-1402 A2.7) → 2026-05-18
            # red 99s busy-wait × 8 stalls inflated cross-cell latency. Paper
            # §1 latency table uses `avg_total_latency_canonical_ms` as
            # primary; raw + minus_retry retained as sensitivity columns.
            # canonical = raw - retry - busy_wait = minus_retry - busy_wait
            "avg_busy_wait_total_ms": cond.get("avg_busy_wait_total_ms"),
            # /stress 2026-05-20 P0-C3-E + P1-14-E (Track E F3): explicit
            # None-check both operands. Pre-fix `(x or 0)` short-circuit
            # treated None and 0.0 identically → if avg_busy_wait_total_ms
            # missing (legacy vintage), canonical = minus_retry - 0 = no
            # busy_wait subtraction; if minus_retry silent-zero-injected
            # (Fix #5 closes upstream), canonical = 0 - busy_wait = negative.
            # Both paths produced wrong paper §1 canonical-latency values.
            # Post-fix: None propagates → downstream consumers must handle
            # explicit None instead of treating missing as zero.
            # B-1780 (/stress GRL audit 2026-05-20, user Q3=A, B-1773 follow-up):
            # canonical now subtracts a THIRD term — the recovered dom artifact
            # screenshot-timeout (C1b). canonical = minus_retry − busy_wait −
            # recovered. The recovered term uses `or 0.0` (NOT None-propagate like
            # the other two) because a missing/legacy value genuinely means "no
            # C1b recovery occurred" (0 ms), whereas missing minus_retry/busy_wait
            # means "wrong vintage, cannot compute" (None). Raw + minus_retry +
            # recovered_total + recovered_rate carried as sensitivity columns.
            "avg_total_latency_canonical_ms": (
                None
                if cond.get("avg_total_latency_minus_retry_ms") is None
                or cond.get("avg_busy_wait_total_ms") is None
                else float(cond["avg_total_latency_minus_retry_ms"])
                - float(cond["avg_busy_wait_total_ms"])
                - float(cond.get("avg_screenshot_timeout_recovered_total_ms") or 0.0)
            ),
            "avg_screenshot_timeout_recovered_total_ms": cond.get("avg_screenshot_timeout_recovered_total_ms"),
            "screenshot_timeout_recovered_episode_rate": cond.get("screenshot_timeout_recovered_episode_rate"),
            # P1-2 (/stress accounting audit 2026-05-21, codex Mode B OOB): carry
            # the canonical-latency operand (busy_wait) into the row so the CSV can
            # publish all three subtraction terms. canonical = minus_retry −
            # busy_wait − recovered (computed above); without busy_wait in the CSV
            # the canonical estimand is not reproducible from the artifact.
            "avg_busy_wait_total_ms": cond.get("avg_busy_wait_total_ms"),
            # Protocol Reset #6/#7/#8 (§244 canonical, 2026-05-20): two-budget
            # accounting + three-column cost per cell. Cost columns are already
            # None-guarded upstream (metrics._avg_or_none → None on legacy vintage
            # = "cannot compute"); carried directly. Cross-baseline pooling MUST
            # stratify by `cost_unit_basis` (B0 API-USD vs B1/B2 local).
            # /stress accounting audit 2026-05-21 (Q1=A): paper §1 PRIMARY cost =
            # `total_billed` (honest "what you pay", hardest to attack);
            # `canonical` + `protocol_wasted` are §4 efficiency decomposition.
            # `parse_error_rate` + `model_call_attempt` exposed to defuse the
            # gemini Mode C "canonical-only flatters B0 / free-look" attack.
            "avg_agent_action_step_count": cond.get("avg_agent_action_step_count"),
            "avg_valid_action_step_count": cond.get("avg_valid_action_step_count"),
            "avg_model_call_attempt_count": cond.get("avg_model_call_attempt_count"),
            "avg_runner_iteration_count": cond.get("avg_runner_iteration_count"),
            "avg_parse_error_injected_wait_count": cond.get("avg_parse_error_injected_wait_count"),
            "avg_total_billed_cost_usd": cond.get("avg_total_billed_cost_usd"),
            "avg_canonical_action_cost_usd": cond.get("avg_canonical_action_cost_usd"),
            "avg_protocol_wasted_cost_usd": cond.get("avg_protocol_wasted_cost_usd"),
            # P1-3 coverage transparency + P1-7/§4 parse-error-rate (audit 2026-05-21)
            "cost_column_coverage_count": cond.get("cost_column_coverage_count"),
            "cost_column_coverage_rate": cond.get("cost_column_coverage_rate"),
            "cost_coverage_partial": cond.get("cost_coverage_partial"),
            "parse_error_rate": cond.get("parse_error_rate"),
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
        "--use-legacy-adjusted", action="store_true",
        help="OPT-IN for legacy adjusted_sr (post-§139.8 RETIRED, archive Appendix D "
             "sensitivity only). Default = raw `success` is canonical (paper-grade). "
             "/stress A1.19 P0-2 (2026-05-17, 3-AI overlap Claude+Codex+Gemini): "
             "flipped semantics from --no-adjusted to enforce §139.8 retirement spec.",
    )
    parser.add_argument(
        "--no-adjusted", action="store_true",
        help="DEPRECATED — no-op (raw is now default; use --use-legacy-adjusted to opt into archive).",
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

    # /stress A1.19 P0-2 (2026-05-17): default = raw `success` canonical (§139.8 retire);
    # legacy adjusted only via explicit opt-in. `--no-adjusted` retained as no-op for
    # backward-compat with existing Makefile invocations.
    use_adjusted = args.use_legacy_adjusted

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
    # A1.21 P1-2 fix (B-531, codex unique OOB): add `baseline` field to all rows so
    # B0/B1/B2 don't collide on (site, mode) key. Pre-fix: cross-site tables had no
    # baseline column → reviewer/aggregator couldn't tell whether `reddit/DOM` was
    # B0, B1, or B2; downstream `per_site` lookup via `next(r["mode"]==m)` picked
    # the first matching row by mode only → silent baseline misattribution.
    aggregation_rows = []
    for r in all_rows:
        sr_val = r.get("adjusted_sr") if use_adjusted else r.get("raw_sr")
        if sr_val is None:
            sr_val = r.get("raw_sr")  # fallback
        aggregation_rows.append({
            "baseline": r.get("baseline", "unknown"),  # A1.21 P1-2: was missing
            "site": r["site"],
            "mode": r["mode"],
            "raw_sr": r["raw_sr"],
            "adjusted_sr": r.get("adjusted_sr"),
            "sr_used": round(float(sr_val), 4) if sr_val is not None else None,
            "avg_cost_usd": r["avg_cost_usd"],
            # B-1409 (/stress A2.7 P1-8-B*): carry cost_unit_basis into the
            # cross-site CSV so reviewers see per-row basis tag. Stratify
            # before pooling across baselines (B0=api_usd ≠ B1/B2=
            # electricity_usd_derived; mixing produces ~1000× unit-collision).
            "cost_unit_basis": r.get("cost_unit_basis", "unknown"),
            "avg_steps": r["avg_steps"],
            "avg_total_energy_kwh": r.get("avg_total_energy_kwh"),
            # B-1410 (/stress A2.7 P1-5-AB*): canonical = retry-adjusted; raw
            # latency carried as sensitivity column per §3.5.1 B-1402 estimand.
            "avg_total_latency_ms": r.get("avg_total_latency_ms"),
            "avg_total_latency_minus_retry_ms": r.get("avg_total_latency_minus_retry_ms"),
            # P1-2 (/stress accounting audit 2026-05-21, codex Mode B OOB): publish
            # the canonical-latency estimand + its 3 subtraction operands so the
            # value is reproducible from the CSV. canonical = minus_retry −
            # busy_wait − screenshot_recovered. Pre-fix only raw + minus_retry were
            # emitted → the headline canonical latency was computed (aggregate_run_dir
            # L309) then DROPPED from the published artifact.
            "avg_total_latency_canonical_ms": r.get("avg_total_latency_canonical_ms"),
            "avg_busy_wait_total_ms": r.get("avg_busy_wait_total_ms"),
            "avg_screenshot_timeout_recovered_total_ms": r.get("avg_screenshot_timeout_recovered_total_ms"),
            # P1-2 (/stress accounting audit 2026-05-21, codex Mode B): the Protocol
            # Reset two-budget counters + three-column cost were carried per-cell in
            # `all_rows` but DROPPED here → never reached the published CSV, so the
            # paper §1 cost estimand was unauditable from the artifact. §1 PRIMARY
            # cost = `avg_total_billed_cost_usd` (Q1=A); canonical + wasted = §4.
            "avg_total_billed_cost_usd": r.get("avg_total_billed_cost_usd"),
            "avg_canonical_action_cost_usd": r.get("avg_canonical_action_cost_usd"),
            "avg_protocol_wasted_cost_usd": r.get("avg_protocol_wasted_cost_usd"),
            "avg_agent_action_step_count": r.get("avg_agent_action_step_count"),
            "avg_valid_action_step_count": r.get("avg_valid_action_step_count"),
            "avg_model_call_attempt_count": r.get("avg_model_call_attempt_count"),
            "avg_runner_iteration_count": r.get("avg_runner_iteration_count"),
            "avg_parse_error_injected_wait_count": r.get("avg_parse_error_injected_wait_count"),
            "parse_error_rate": r.get("parse_error_rate"),
            "cost_column_coverage_count": r.get("cost_column_coverage_count"),
            "cost_column_coverage_rate": r.get("cost_column_coverage_rate"),
            "cost_coverage_partial": r.get("cost_coverage_partial"),
            "episodes": r["episodes"],
            "is_stub": r["is_stub"],
        })
    # P1-1 fail-loud (/stress accounting audit 2026-05-21, user Q1=A): a non-stub
    # row with billed cost but unknown/None cost_unit_basis means the
    # step→episode→condition basis chain broke (B-1798) — emitting it as "unknown"
    # silently makes the §1 cross-baseline cost stratification unverifiable. Fail
    # loud instead. Legacy/archive re-aggregation (pre-B-1798 vintage) can bypass
    # via CROSS_SITE_ALLOW_UNKNOWN_BASIS=1.
    import os as _os
    if not _os.environ.get("CROSS_SITE_ALLOW_UNKNOWN_BASIS"):
        _bad_basis = [
            r for r in aggregation_rows
            if not r.get("is_stub")
            and r.get("avg_total_billed_cost_usd") is not None
            and r.get("cost_unit_basis") in (None, "unknown", "")
        ]
        if _bad_basis:
            raise ValueError(
                f"cross_site aggregation: {len(_bad_basis)} paper-grade row(s) carry "
                f"billed cost but cost_unit_basis is unknown/None — the "
                f"step→episode→condition basis chain broke (B-1798); §1 cross-baseline "
                f"cost stratification would be unverifiable. Offending "
                f"(baseline,site,mode): "
                f"{[(r.get('baseline'), r.get('site'), r.get('mode')) for r in _bad_basis][:8]}. "
                f"Fix the episode-summary rollup, or set "
                f"CROSS_SITE_ALLOW_UNKNOWN_BASIS=1 for legacy archive re-aggregation."
            )
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

    # B-1409 (/stress A2.7 P1-8-B*): the cross-site cost plot pools rows across
    # baselines (B0+B1+B2) at single y-axis. When cost_unit_basis mixes api_usd
    # (B0) with electricity_usd_derived (B1/B2), the plot is a unit-collision
    # artifact, NOT a scientific cost number. Detect the mix here and downgrade
    # the plot to per-basis subplots (one per basis) OR label the y-axis with
    # the mixed-basis warning if all rows happen to share a basis.
    _bases = sorted({r.get("cost_unit_basis", "unknown") for r in aggregation_rows})
    _bases = [b for b in _bases if b not in ("unknown", "")]
    _mixed_basis = len(_bases) > 1
    _basis_label = _bases[0] if (_bases and not _mixed_basis) else "unknown"
    # P1-6 (/stress accounting audit 2026-05-21, codex Mode B): a "MIXED BASIS"
    # *label* is not stratification — pooling B0 API-USD with B1/B2 electricity-
    # derived USD on one y-axis is a unit-collision artifact (~1000×) regardless
    # of the warning text. SUPPRESS the absolute-cost plot when bases mix (emit a
    # message + the per-basis breakdown is recoverable from the CSV which carries
    # cost_unit_basis per row). Single-basis runs plot normally.
    # P1-4 (/stress accounting audit 2026-05-21, codex Mode B): suppress when the
    # basis is MIXED *or* entirely UNKNOWN. Pre-fix the all-unknown case
    # (`_bases == []` after filtering) gave `_mixed_basis = False` → fell to the
    # ELSE plot branch and wrote an absolute-cost figure labelled "unknown" — the
    # exact evidence-loss case treated as safe-to-plot. A figure with no known
    # cost basis is not a scientific cost number.
    _has_cost = any(r.get("avg_total_billed_cost_usd") is not None for r in aggregation_rows)
    if _mixed_basis or (not _bases and _has_cost):
        print(
            "[P1-4/P1-6 cost-basis stratification] SUPPRESSED cross_site_cost_comparison.png "
            f"— cost_unit_basis is {'mixed: ' + str(_bases) if _mixed_basis else 'entirely unknown'}. "
            "An absolute-cost plot here is a unit-collision artifact (B0 api_usd vs "
            "B1/B2 electricity_usd_derived) or basis-less. Stratify by basis from "
            "cross_site_aggregation.csv (per-row cost_unit_basis) before any cost "
            "figure. See paper §3.5.1 + B-1409 + /stress audit 2026-05-21 P1-4/P1-6."
        )
    else:
        _plot_grouped_bar(
            data=aggregation_rows,
            x_key="site",
            group_key="mode",
            # Q1=A: §1 primary cost = total_billed; fall back to legacy avg_cost_usd
            # only if total_billed absent (legacy vintage).
            value_key="avg_total_billed_cost_usd" if any(
                r.get("avg_total_billed_cost_usd") is not None for r in aggregation_rows
            ) else "avg_cost_usd",
            title=(
                f"{args.b1_label} Phase 1 — Avg Billed Cost (USD) by Site × Mode  "
                f"[cost_unit_basis = {_basis_label}]"
            ),
            ylabel=f"Avg Billed Cost per Episode (USD, basis = {_basis_label})",
            out_path=out_dir / "cross_site_cost_comparison.png",
        )

    # --- cross_site_summary.json ---
    print("[4/4] Writing summary JSON...")
    sites = sorted(set(r["site"] for r in all_rows))
    # A1.21 P1-2 fix (B-531): per_site now groups by (baseline, site, mode) to prevent
    # B0/B1/B2 row collision. Output is per_site[site][baseline][mode] = {...}.
    per_site: Dict[str, Any] = {}
    baselines = sorted({r.get("baseline", "unknown") for r in all_rows})
    for site in sites:
        site_rows = [r for r in all_rows if r["site"] == site]
        per_site[site] = {}
        for baseline in baselines:
            baseline_rows = [r for r in site_rows if r.get("baseline") == baseline]
            if not baseline_rows:
                continue
            per_site[site][baseline] = {
                m: {
                    "raw_sr": next((r["raw_sr"] for r in baseline_rows if r["mode"] == m), None),
                    "adjusted_sr": next((r.get("adjusted_sr") for r in baseline_rows if r["mode"] == m), None),
                    "avg_cost_usd": next((r["avg_cost_usd"] for r in baseline_rows if r["mode"] == m), None),
                    "avg_steps": next((r["avg_steps"] for r in baseline_rows if r["mode"] == m), None),
                    # P1-2 (/stress accounting audit 2026-05-21): Protocol Reset
                    # estimand in the published JSON. §1 primary = total_billed (Q1=A).
                    "avg_total_billed_cost_usd": next((r.get("avg_total_billed_cost_usd") for r in baseline_rows if r["mode"] == m), None),
                    "avg_canonical_action_cost_usd": next((r.get("avg_canonical_action_cost_usd") for r in baseline_rows if r["mode"] == m), None),
                    "avg_protocol_wasted_cost_usd": next((r.get("avg_protocol_wasted_cost_usd") for r in baseline_rows if r["mode"] == m), None),
                    "avg_valid_action_step_count": next((r.get("avg_valid_action_step_count") for r in baseline_rows if r["mode"] == m), None),
                    "avg_model_call_attempt_count": next((r.get("avg_model_call_attempt_count") for r in baseline_rows if r["mode"] == m), None),
                    "parse_error_rate": next((r.get("parse_error_rate") for r in baseline_rows if r["mode"] == m), None),
                    "cost_unit_basis": next((r.get("cost_unit_basis") for r in baseline_rows if r["mode"] == m), None),
                    "cost_coverage_partial": next((r.get("cost_coverage_partial") for r in baseline_rows if r["mode"] == m), None),
                }
                for m in MODES
                if any(r["mode"] == m for r in baseline_rows)
            }

    # Weighted-average SR across sites (equal weight per site).
    # P1-3 (/stress accounting audit 2026-05-21, codex Mode B unique OOB): key by
    # (baseline, mode), NOT mode alone. Pre-fix the mode-only key silently pooled
    # B0+B1+B2 into one per-mode SR (real artifact: global_avg_sr_per_mode=
    # {"DOM": 0.0} computed from 3 distinct baseline rows) → destroyed the
    # 3-baseline design on any nonzero run. Now nested baseline→mode.
    weighted_sr: Dict[str, Dict[str, List[float]]] = {}
    for r in all_rows:
        b = r.get("baseline", "unknown")
        m = r["mode"]
        sr = r.get("adjusted_sr") if use_adjusted else r.get("raw_sr")
        if sr is not None:
            weighted_sr.setdefault(b, {}).setdefault(m, []).append(float(sr))
    global_avg_sr_per_baseline_mode = {
        b: {m: round(sum(vals) / len(vals), 4) for m, vals in modes.items()}
        for b, modes in weighted_sr.items()
    }

    summary = {
        "label": args.b1_label,
        "sites": sites,
        "use_adjusted_sr": use_adjusted,
        "per_site": per_site,
        # P1-3 (/stress 2026-05-21): per-(baseline,mode) SR — the prior pooled
        # `global_avg_sr_per_mode` field is REMOVED (it averaged across baselines,
        # destroying the 3-baseline design). Consumers must read per-baseline.
        "global_avg_sr_per_baseline_mode": global_avg_sr_per_baseline_mode,
        "outputs": [f.name for f in sorted(out_dir.iterdir()) if f.is_file()],
    }
    _write_json(summary, out_dir / "cross_site_summary.json")

    print(f"\nDone! Outputs in: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
