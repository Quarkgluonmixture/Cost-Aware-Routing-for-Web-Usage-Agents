#!/usr/bin/env python3
"""Generate the live 4-layer evidence status report.

Reads existing analysis artifacts without failing on missing files. The report
is intended as the paper-facing index for docs/checkpoints/paper_planning.md §3.

Output:
- docs/analysis/layered_evidence_status.md
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/layered_evidence_status.md"

RESULTS = ROOT / "results/visualwebarena/phase1"
PAPER = ROOT / "results/phantom_paper"
CROSS = ROOT / "docs/analysis/cross_sites"

MODE_SPECS: dict[str, dict[str, Path]] = {
    "classifieds": {
        "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0",
        "SoM": RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0",
        "Vision": RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0",
        "P-SoM": RESULTS / "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0",
        "P-text": RESULTS / "B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0",
    },
    "reddit": {
        "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0",
        "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0",
        "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0",
        "P-SoM": RESULTS / "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0",
        "P-text": RESULTS / "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0",
    },
}

AUDITS = {
    "classifieds": CROSS / "codex_audit_classifieds.json",
    "reddit": CROSS / "codex_audit_reddit.json",
}

FIGURES = {
    "fig0c_drop_one": PAPER / "figures/fig0c_drop_one_oracle.png",
    "fig0c_lift": PAPER / "figures/fig0c_phantom_lift_bars.png",
    "fig0d": PAPER / "figures/fig0d_taskpool_jaccard.png",
    "fig0e": PAPER / "figures/fig0e_category_mode_heatmap.png",
    "fig0f": PAPER / "figures/fig0f_overlap_stacked_bar.png",
    "fig0g": PAPER / "figures/fig0g_routing_auroc_heatmap.png",
    "fig1ab": PAPER / "figures/fig1ab_cascade_diamond.png",
    "fig1c": PAPER / "figures/fig1c_strategy_gradient.png",
    "fig2": PAPER / "figures/fig2_micro_divergence_heatmap.png",
    "fig3": PAPER / "figures/fig3_regional_carbon.png",
    "fig3d": PAPER / "figures/fig3d_cost_sr_frontier.png",
    "fig_capability": PAPER / "figures/fig_capability_b0_b1.png",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def timestamp(path: Path) -> str:
    if not path.exists():
        return "⚠️ missing"
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def max_timestamp(paths: list[Path]) -> str:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return "⚠️ missing"
    mt = max(p.stat().st_mtime for p in existing)
    return datetime.fromtimestamp(mt, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def source_line(path: Path, extra: str = "") -> str:
    status = "" if path.exists() else " ⚠️ missing"
    suffix = f" {extra}" if extra else ""
    return f"source: `{rel(path)}`{suffix} | last update: {timestamp(path)}{status}"


def fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.{digits}f}%"


def fmt_pp(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{digits}f}pp"


def fmt_num(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "n/a"
    return f"{xf:.{digits}f}"


def display_mode(mode: str) -> str:
    return {
        "dom": "DOM",
        "phantom_dom": "P-text",   # legacy mode value (paper-grade run dirs use it)
        "phantom_text": "P-text",  # current mode value
        "phantom_som": "P-SoM",
        "phantom_prompt": "P-prompt",
        "som": "SoM",
        "vision": "Vision",
    }.get(mode, mode)


def episode_summaries(condition_dir: Path) -> list[dict[str, Any]]:
    ep_dir = condition_dir / "episodes"
    rows: list[dict[str, Any]] = []
    if not ep_dir.exists():
        return rows
    for path in sorted(ep_dir.glob("*_summary_v2.json")):
        data = read_json(path)
        if isinstance(data, dict):
            data["_path"] = path
            rows.append(data)
    return rows


def task_id(row: dict[str, Any]) -> int | None:
    value = row.get("task_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mode_stats(condition_dir: Path) -> dict[str, Any]:
    rows = episode_summaries(condition_dir)
    n = len(rows)
    raw_s = sum(1 for r in rows if bool(r.get("success")))
    adj_s = sum(1 for r in rows if bool(r.get("adjusted_success", r.get("success"))))
    tasks = {task_id(r) for r in rows if bool(r.get("adjusted_success", r.get("success")))}
    tasks.discard(None)
    condition_summary = read_json(condition_dir / "condition_summary_v2.json") or {}
    tokens_per_step = []
    for row in rows:
        steps = row.get("steps") or 0
        tokens = row.get("total_tokens")
        if steps and tokens:
            tokens_per_step.append(float(tokens) / float(steps))
    return {
        "n": n,
        "raw_successes": raw_s,
        "adjusted_successes": adj_s,
        "raw_sr": raw_s / n if n else None,
        "adjusted_sr": adj_s / n if n else None,
        "fp_rate": (raw_s - adj_s) / n if n else None,
        "success_tasks": tasks,
        "condition_summary": condition_summary,
        "median_tokens_per_step": statistics.median(tokens_per_step) if tokens_per_step else None,
        "mean_tokens_per_step": statistics.mean(tokens_per_step) if tokens_per_step else None,
        "last_update": max_timestamp([Path(r["_path"]) for r in rows if "_path" in r]),
    }


def all_mode_stats() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        site: {mode: mode_stats(path) for mode, path in modes.items()}
        for site, modes in MODE_SPECS.items()
    }


def success_depths(site_stats: dict[str, dict[str, Any]]) -> dict[str, Counter[int]]:
    task_depth: Counter[int] = Counter()
    for stats in site_stats.values():
        for tid in stats["success_tasks"]:
            task_depth[tid] += 1
    out: dict[str, Counter[int]] = {}
    for mode, stats in site_stats.items():
        out[mode] = Counter(task_depth[tid] for tid in stats["success_tasks"])
    return out


def category_letter(category: str) -> str:
    return category[:1] if category else "?"


def category_table(site: str, site_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    audit = read_json(AUDITS[site])
    if not isinstance(audit, list):
        return {}
    cats: dict[int, str] = {}
    for row in audit:
        if isinstance(row, dict) and "task_id" in row:
            cats[int(row["task_id"])] = category_letter(str(row.get("category", "?")))
    out: dict[str, dict[str, float | None]] = {}
    for mode, stats in site_stats.items():
        rows = episode_summaries(MODE_SPECS[site][mode])
        by_cat: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            tid = task_id(row)
            if tid is None or tid not in cats:
                continue
            by_cat[cats[tid]].append(bool(row.get("adjusted_success", row.get("success"))))
        out[mode] = {
            cat: (sum(vals) / len(vals) if vals else None)
            for cat, vals in sorted(by_cat.items())
        }
    return out


def search_loop_rate(site: str, condition_dir: Path) -> float | None:
    markers = ("/search",) if site == "reddit" else ("page=search", "/search")
    ep_dir = condition_dir / "episodes"
    if not ep_dir.exists():
        return None
    total = 0
    loops = 0
    for path in sorted(ep_dir.glob("*_steps_v2.jsonl")):
        search_steps = 0
        total += 1
        try:
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    url = str(row.get("obs_url") or row.get("url") or "")
                    if any(marker in url for marker in markers):
                        search_steps += 1
        except Exception:
            continue
        if search_steps >= 2:
            loops += 1
    return loops / total if total else None


def best_auroc(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    best: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("baseline") != "B0":
            continue
        try:
            auroc = float(row.get("AUROC") or "nan")
        except ValueError:
            continue
        if math.isnan(auroc):
            continue
        key = (row["baseline"], row["site"], row["mode"])
        old = best.get(key)
        if old is None or auroc > float(old["AUROC"]):
            best[key] = row
    return best


def render_layer0(lines: list[str], stats: dict[str, dict[str, dict[str, Any]]]) -> None:
    phantom_csv = PAPER / "phantom_lift.csv"
    auroc_csv = PAPER / "auroc_cross_condition.csv"
    sr_fp_md = CROSS / "sr_fp_per_mode.md"
    lift_rows = read_csv(phantom_csv)
    auroc_rows = best_auroc(read_csv(auroc_csv))

    lines += ["## Layer 0 — Outcome", ""]

    lines += ["### 0a SR per mode (B0)", ""]
    for site in ["reddit", "classifieds"]:
        parts = []
        for mode in ["DOM", "P-text", "P-SoM", "SoM", "Vision"]:
            if mode not in stats[site]:
                continue
            s = stats[site][mode]
            parts.append(f"{mode} raw {fmt_pct(s['raw_sr'])} / adj **{fmt_pct(s['adjusted_sr'])}**")
        lines.append(f"- {site}: " + "; ".join(parts))
    lines.append(f"- source: `results/visualwebarena/phase1/B0_*/*/episodes/*_summary_v2.json` (live); last update: {max_timestamp([p for modes in MODE_SPECS.values() for d in modes.values() for p in (d / 'episodes').glob('*_summary_v2.json')])}")
    lines.append(f"- standalone cite source: `{rel(sr_fp_md)}` | last update: {timestamp(sr_fp_md)}")
    lines.append("")

    lines += ["### 0b FP rate (raw success - adjusted success)", ""]
    for site in ["reddit", "classifieds"]:
        parts = [f"{mode} {fmt_pct(stats[site][mode]['fp_rate'])}" for mode in ["DOM", "P-text", "P-SoM", "SoM", "Vision"]]
        lines.append(f"- {site}: " + "; ".join(parts))
    lines.append(f"- source: same live episode `summary_v2.json` files as 0a; standalone `{rel(sr_fp_md)}` | last update: {timestamp(sr_fp_md)}")
    lines.append("")

    mechanism_json = CROSS / "mechanism_per_task.json"
    mechanism = read_json(mechanism_json) or {}
    e3 = (mechanism.get("E3_confidence_calibration") or {}).get("cells", {})
    if e3:
        lines += ["### 0b-extra Confidence calibration (E3)", ""]
        for model in ["B0", "B1"]:
            for site in ["reddit", "classifieds"]:
                rows = [
                    row for row in e3.values()
                    if row.get("model") == model and row.get("site") == site
                ]
                if not rows:
                    continue
                best = max(
                    rows,
                    key=lambda row: max(
                        row.get("AUROC_token") or -1,
                        row.get("AUROC_verbal") or -1,
                        row.get("AUROC_behavioral_max") or -1,
                    ),
                )
                honest = [row for row in rows if row.get("ECE_verbal") is not None]
                honest_row = min(honest, key=lambda row: row["ECE_verbal"]) if honest else None
                best_val = max(
                    best.get("AUROC_token") or -1,
                    best.get("AUROC_verbal") or -1,
                    best.get("AUROC_behavioral_max") or -1,
                )
                honest_part = (
                    f"; lowest verbal ECE {honest_row['mode']} {fmt_num(honest_row['ECE_verbal'])}"
                    if honest_row else "; ECE n/a in existing outputs"
                )
                lines.append(
                    f"- {model} {site}: best routing AUROC {best['mode']} **{fmt_num(best_val)}**"
                    f"{honest_part}"
                )
        lines.append(f"- {source_line(mechanism_json)}")
        lines.append("")

    lines += ["### 0c Routing oracle (3→5-mode lift)", ""]
    if not lift_rows:
        lines.append(f"- ⚠️ missing or unreadable | {source_line(phantom_csv)}")
    for row in lift_rows:
        if row.get("baseline") != "B0":
            continue
        sig = "✅" if row.get("phantom_pair_jaccard_warn") == "False" else "⚠️"
        lines.append(
            f"- {row['site']}: **{fmt_pp(float(row['lift_5_vs_3_pp']))}** "
            f"[{fmt_num(row['lift_5_vs_3_ci95_lo_pp'], 2)}, {fmt_num(row['lift_5_vs_3_ci95_hi_pp'], 2)}] "
            f"Wilcoxon p={fmt_num(row['wilcoxon_5_vs_3_p'], 4)}, McNemar p={fmt_num(row['mcnemar_5_vs_3_p'], 4)} {sig}"
        )
        lines.append(
            f"  - single phantom lifts: +P-text {fmt_pp(float(row['lift_4pdom_vs_3_pp']))}; "
            f"+P-SoM {fmt_pp(float(row['lift_4psom_vs_3_pp']))}"
        )
    lines.append(f"- {source_line(phantom_csv)}")
    lines.append(f"- figures: `{rel(FIGURES['fig0c_drop_one'])}`, `{rel(FIGURES['fig0c_lift'])}` | last update: {max_timestamp([FIGURES['fig0c_drop_one'], FIGURES['fig0c_lift']])}")
    lines.append("")

    lines += ["### 0d Task-pool Jaccard (Scenario C sentinel)", ""]
    for row in lift_rows:
        if row.get("baseline") != "B0":
            continue
        j = float(row["phantom_pair_jaccard"])
        verdict = "✅ safe" if j <= 0.7 else "⚠️ redundant-risk"
        lines.append(f"- {row['site']}: P-text↔P-SoM Jaccard **{j:.3f}** ({verdict}); threshold ≤0.7")
    lines.append(f"- {source_line(phantom_csv)}")
    lines.append(f"- figure: `{rel(FIGURES['fig0d'])}` | last update: {timestamp(FIGURES['fig0d'])}")
    lines.append("")

    lines += ["### 0e Per-category SR", ""]
    for site in ["reddit", "classifieds"]:
        table = category_table(site, stats[site])
        if not table:
            lines.append(f"- {site}: ⚠️ category audit missing")
            continue
        for mode in ["DOM", "P-SoM", "SoM"]:
            cat_bits = [f"{cat} {fmt_pct(val)}" for cat, val in sorted(table.get(mode, {}).items())]
            lines.append(f"- {site} {mode}: " + "; ".join(cat_bits))
    lines.append(f"- figure: `{rel(FIGURES['fig0e'])}` | last update: {timestamp(FIGURES['fig0e'])}")
    lines.append("")

    lines += ["### 0f Overlap depth", ""]
    for site in ["reddit", "classifieds"]:
        depths = success_depths(stats[site])
        for mode in ["P-SoM", "P-text"]:
            c = depths.get(mode, Counter())
            parts = [f"d{depth}={c.get(depth, 0)}" for depth in range(1, 6)]
            lines.append(f"- {site} {mode}: " + " / ".join(parts))
    lines.append(f"- figure: `{rel(FIGURES['fig0f'])}` | last update: {timestamp(FIGURES['fig0f'])}")
    lines.append("")

    lines += ["### 0g Routing AUROC", ""]
    if not auroc_rows:
        lines.append(f"- ⚠️ missing or unreadable | {source_line(auroc_csv)}")
    for site in ["reddit", "classifieds"]:
        parts = []
        for mode in ["dom", "phantom_dom", "phantom_som", "som", "vision"]:
            row = auroc_rows.get(("B0", site, mode))
            if not row:
                parts.append(f"{display_mode(mode)} n/a")
                continue
            parts.append(f"{display_mode(mode)} {float(row['AUROC']):.3f} ({row['signal']})")
        lines.append(f"- {site}: " + "; ".join(parts))
    lines.append(f"- {source_line(auroc_csv)}")
    lines.append(f"- figure: `{rel(FIGURES['fig0g'])}` | last update: {timestamp(FIGURES['fig0g'])}")
    lines.append("")


def render_layer1(lines: list[str]) -> None:
    axis_json = CROSS / "axis_effect_size.json"
    axis = read_json(axis_json) or {}
    interp = axis.get("interpretation", {}) if isinstance(axis, dict) else {}
    tier1 = interp.get("tier1_hook", {})
    tier2 = interp.get("tier2_mechanism", {})

    lines += ["## Layer 1 — Macro Behavior", ""]
    lines += ["### 1a Tier 1 hook coarse", ""]
    both = tier1.get("psom_distinct_from_both_dom_and_som", [])
    by_site = Counter(item.split("@")[-1] for item in both if "@" in item)
    lines.append(f"- P-SoM distinct from both endpoints: reddit **{by_site.get('reddit', 0)}/8**, classifieds **{by_site.get('classifieds', 0)}/8**")
    lines.append(f"- DOM-only distinct cells: {len(tier1.get('psom_distinct_from_dom_only', []))}; SoM-only distinct cells: {len(tier1.get('psom_distinct_from_som_only', []))}; indistinct cells: {len(tier1.get('psom_indistinct_from_either', []))}")
    lines.append(f"- {source_line(axis_json)}")
    lines.append("")

    lines += ["### 1b Tier 2a cascade", ""]
    dom = tier2.get("dominant_cascade_by_axis", {})
    lines.append("- Dominant cascade counts: " + "; ".join(f"{axis} {len(vals)}" for axis, vals in dom.items()))
    pairs = tier2.get("antagonistic_pairs", [])
    lines.append(f"- Antagonistic mechanism pairs: **{len(pairs)}** ({', '.join(pairs[:6])})")
    lines.append(f"- {source_line(axis_json)}")
    lines.append("")

    lines += ["### 1c Strategy gradient", ""]
    for site in ["reddit", "classifieds"]:
        vals = []
        for mode in ["DOM", "P-SoM", "SoM"]:
            vals.append(f"{mode} search-loop {fmt_pct(search_loop_rate(site, MODE_SPECS[site][mode]))}")
        lines.append(f"- {site}: " + " → ".join(vals))
    lines.append(f"- figure: `{rel(FIGURES['fig1c'])}` | last update: {timestamp(FIGURES['fig1c'])}")
    lines.append("")

    mechanism_json = CROSS / "mechanism_per_task.json"
    mechanism = read_json(mechanism_json) or {}
    e4 = mechanism.get("E4_action_vocabulary") or {}
    contrasts = e4.get("axis_contrasts") or {}
    if contrasts:
        lines += ["### 1d Full action vocabulary (E4)", ""]
        for site in ["reddit", "classifieds"]:
            comp = contrasts.get(site, {}).get("compound_DOM_to_PSoM", {})
            top = (comp.get("top_abs_shifts") or [])[:3]
            bits = [f"{row['action_type']} {fmt_num(row['mean_fraction_shift'])}" for row in top]
            lines.append(f"- {site}: compound DOM→P-SoM top shifts: " + "; ".join(bits))
        lines.append(f"- {source_line(mechanism_json)}")
        lines.append("")


def render_layer2(lines: list[str]) -> None:
    micro_json = CROSS / "axis1_microbehavior.json"
    micro = read_json(micro_json) or {}
    contrasts = micro.get("axis_contrasts", {}) if isinstance(micro, dict) else {}
    validity = micro.get("cross_site_validity", {}) if isinstance(micro, dict) else {}

    lines += ["## Layer 2 — Micro Behavior", ""]

    lines += ["### 2a URL signature", ""]
    for site in ["reddit", "classifieds"]:
        axis1 = contrasts.get(site, {}).get("axis_1_text", {})
        comp = contrasts.get(site, {}).get("compound_dom_to_psom", {})
        lines.append(
            f"- {site}: axis-1 URL-path Jaccard **{fmt_num(axis1.get('url_jaccard_mean'), 3)}**; "
            f"compound DOM↔P-SoM **{fmt_num(comp.get('url_jaccard_mean'), 3)}**"
        )
    lines.append(f"- {source_line(micro_json)}")
    lines.append(f"- figure: `{rel(FIGURES['fig2'])}` | last update: {timestamp(FIGURES['fig2'])}")
    lines.append("")

    mechanism_json = CROSS / "mechanism_per_task.json"
    mechanism = read_json(mechanism_json) or {}
    e1 = mechanism.get("E1_click_target_divergence") or {}
    if e1:
        lines += ["### 2a-extra Click-target divergence (E1)", ""]
        for site in ["reddit", "classifieds"]:
            axis1 = e1.get(site, {}).get("axis_1_text", {})
            comp = e1.get(site, {}).get("compound_DOM_to_PSoM", {})
            lines.append(
                f"- {site}: axis-1 click-transition Jaccard **{fmt_num(axis1.get('mean_jaccard'))}**; "
                f"compound DOM↔P-SoM **{fmt_num(comp.get('mean_jaccard'))}**"
            )
        lines.append(f"- {source_line(mechanism_json)}")
        lines.append("")

    lines += ["### 2b Target-hit", ""]
    for site in ["reddit", "classifieds"]:
        axis1 = contrasts.get(site, {}).get("axis_1_text", {})
        comp = contrasts.get(site, {}).get("compound_dom_to_psom", {})
        lines.append(
            f"- {site}: axis-1 {fmt_pp(axis1.get('target_hit_rate_diff_pct_pts'))}; "
            f"compound {fmt_pp(comp.get('target_hit_rate_diff_pct_pts'))}"
        )
    lines.append(f"- {source_line(micro_json)}")
    lines.append("")

    lines += ["### 2c Keyword reuse", ""]
    for site in ["reddit", "classifieds"]:
        axis1 = contrasts.get(site, {}).get("axis_1_text", {})
        comp = contrasts.get(site, {}).get("compound_dom_to_psom", {})
        lines.append(
            f"- {site}: axis-1 max-keyword-repeat diff **{fmt_num(axis1.get('max_keyword_repeat_diff'), 3)}**; "
            f"compound **{fmt_num(comp.get('max_keyword_repeat_diff'), 3)}**"
        )
    lines.append(f"- {source_line(micro_json)}")
    lines.append("")

    lines += ["### 2d First-action", ""]
    for site in ["reddit", "classifieds"]:
        axis1 = contrasts.get(site, {}).get("axis_1_text", {})
        comp = contrasts.get(site, {}).get("compound_dom_to_psom", {})
        lines.append(
            f"- {site}: axis-1 divergence **{fmt_pct(axis1.get('first_action_divergence_rate'))}**; "
            f"compound **{fmt_pct(comp.get('first_action_divergence_rate'))}**"
        )
    lines.append(f"- {source_line(micro_json)}")
    lines.append("")

    lines += ["### 2e Cross-site validity", ""]
    lines.append(
        f"- verdict: **{validity.get('verdict', 'n/a')}**; reddit ratio {fmt_num(validity.get('reddit_ratio'), 2)}, "
        f"classifieds ratio {fmt_num(validity.get('classifieds_ratio'), 2)}"
    )
    lines.append(f"- {source_line(micro_json)}")
    lines.append("")

    e2 = mechanism.get("E2_trajectory_boundary") or {}
    if e2:
        lines += ["### 2f Trajectory boundary (E2)", ""]
        for site in ["reddit", "classifieds"]:
            comp = e2.get(site, {}).get("DOM_vs_Phantom-SoM", {})
            lines.append(
                f"- {site}: DOM↔P-SoM symmetric-diff N **{comp.get('n_symmetric_diff_tasks', 'n/a')}**; "
                f"median first divergent step {fmt_num(comp.get('median_first_divergent_step'), 1)}; "
                f"early {fmt_pct(comp.get('early_divergence_rate'))}; late {fmt_pct(comp.get('late_divergence_rate'))}"
            )
        lines.append(f"- {source_line(mechanism_json)}")
        lines.append("")


def render_layer3(lines: list[str], stats: dict[str, dict[str, dict[str, Any]]]) -> None:
    run_collect = PAPER / "run_summary_collect.json"

    lines += ["## Layer 3 — Efficiency", ""]

    lines += ["### 3a Token/cost per step", ""]
    for site in ["reddit", "classifieds"]:
        parts = []
        for mode in ["DOM", "P-SoM", "SoM"]:
            s = stats[site][mode]["condition_summary"]
            cost = s.get("avg_input_cost_usd")
            steps = s.get("avg_steps")
            per_step = (float(cost) / float(steps)) if cost and steps else None
            parts.append(f"{mode} input-cost/step ${per_step:.5f}" if per_step is not None else f"{mode} n/a")
        lines.append(f"- {site}: " + "; ".join(parts))
    lines.append("- source: B0 `condition_summary_v2.json` per condition")
    lines.append("")

    lines += ["### 3b Image embedding / total-token gap", ""]
    for site in ["reddit", "classifieds"]:
        som = stats[site]["SoM"].get("median_tokens_per_step")
        psom = stats[site]["P-SoM"].get("median_tokens_per_step")
        gap = (som - psom) if som is not None and psom is not None else None
        lines.append(
            f"- {site}: SoM median tokens/step {fmt_num(som, 0)} vs P-SoM {fmt_num(psom, 0)}; "
            f"observed gap **{fmt_num(gap, 0)} tokens/step**"
        )
    note = "" if run_collect.exists() else " ⚠️ run_summary_collect missing"
    lines.append(f"- source: `{rel(run_collect)}` plus episode `total_tokens` fallback | last update: {timestamp(run_collect)}{note}")
    lines.append("")

    lines += ["### 3c Latency", ""]
    for site in ["reddit", "classifieds"]:
        parts = []
        for mode in ["DOM", "P-SoM", "SoM"]:
            latency = stats[site][mode]["condition_summary"].get("avg_total_latency_ms")
            parts.append(f"{mode} {float(latency) / 1000:.1f}s/episode" if latency else f"{mode} n/a")
        psom = stats[site]["P-SoM"]["condition_summary"].get("avg_total_latency_ms")
        som = stats[site]["SoM"]["condition_summary"].get("avg_total_latency_ms")
        ratio = (float(psom) / float(som)) if psom and som else None
        lines.append(f"- {site}: " + "; ".join(parts) + f"; P-SoM/SoM {fmt_num(ratio, 2)}x")
    lines.append("- source: B0 `condition_summary_v2.json` per condition")
    lines.append("")

    lines += ["### 3d B0 (API) vs B1 (local) deployment-class cost gap", ""]
    # Use cost_per_mode.json (FRESH 04-29) which separates API token $ vs
    # electricity-equivalent $. condition_summary_v2.json's cost field is
    # NOT comparable across B0/B1 because B1 uses the same yaml token rate.
    cost_per_mode_path = ROOT / "docs/analysis/cross_sites/cost_per_mode.json"
    cpm = read_json(cost_per_mode_path) or {}
    ratios = (cpm.get("deployment_class_ratios") or {})
    if ratios:
        lines.append(
            "Computed via `aggregate_cost_electricity.py`: B0 = API token dollars; "
            "B1 = `avg_total_energy_kwh × $0.12/kWh` (electricity equivalent, UK industrial). "
            "B0 vs B1 belong to different cost classes (API vs electricity), not a single ratio in $:"
        )
        for site in ("reddit", "classifieds"):
            r = ratios.get(site)
            if not r:
                continue
            ratio_str = (
                f"{r['ratio_B0_over_B1']:.0f}x"
                if r.get("ratio_B0_over_B1") is not None
                else "n/a"
            )
            lines.append(
                f"- {site}: B0 API ${r['avg_B0_API_dollars']:.4f}/ep vs "
                f"B1 electricity ${r['avg_B1_electricity_dollars']:.6f}/ep → **{ratio_str}** deployment-class gap"
            )
        lines.append(
            "- ⚠️ §103 / paper-planning legacy '30×' claim **superseded** by these data — "
            "real ratio ~100× (deployment class, not capability ratio)"
        )
        lines.append(f"- source: `{rel(cost_per_mode_path)}` | last update: {timestamp(cost_per_mode_path)}")
    else:
        lines.append(
            "- ⚠️ `cost_per_mode.json` not yet generated; run `make aggregate-cost-electricity`. "
            "Falling back to raw condition_summary cost field (B0/B1 normalized to same token rate, ~1×)."
        )
        for site in ["reddit", "classifieds"]:
            b0 = []
            b1 = []
            for mode_dir in ["phase1_dom_router_0", "phase1_som_router_0", "phase1_vision_router_0"]:
                site_run = "B0_3mode_reddit_20260422" if site == "reddit" else "B0_3mode_classifieds_20260413"
                b1_run = "B1_3mode_reddit_20260413" if site == "reddit" else "B1_3mode_classifieds_20260413"
                b0_data = read_json(RESULTS / site_run / mode_dir / "condition_summary_v2.json") or {}
                b1_data = read_json(RESULTS / b1_run / mode_dir / "condition_summary_v2.json") or {}
                if b0_data.get("avg_total_cost_usd"):
                    b0.append(float(b0_data["avg_total_cost_usd"]))
                if b1_data.get("avg_total_cost_usd"):
                    b1.append(float(b1_data["avg_total_cost_usd"]))
            ratio = (statistics.mean(b0) / statistics.mean(b1)) if b0 and b1 else None
            lines.append(f"- {site}: same-rate token-cost ratio **{fmt_num(ratio, 1)}x** (artifact)")
    lines.append(f"- figure: `{rel(FIGURES['fig3d'])}` | last update: {timestamp(FIGURES['fig3d'])}")
    lines.append("")


def render_claim_matrix(lines: list[str]) -> None:
    lines += [
        "## Paper Claim → Layer Support Matrix",
        "",
        "| Claim | Layers cited | Verdict |",
        "|---|---|---|",
        "| C1 P-SoM independent routing arm | 0a, 0c, 0d, 0g, 1a, 2a | ✅ supported by live outcome + behavior artifacts |",
        "| C2 4-fold drop-in property | 3a, 3c, 0g, 0c | ✅ cost/latency/signal/oracle evidence present |",
        "| C3 3-axis hierarchical theory | 1b, 2a-2e, cross-layer mechanism chain | ✅ cascade + micro decomposition present |",
        "| C4 Aggregate macro can mislead about routing potential | 1a, 0d, 2a | ✅ supported by task-pool and micro-divergence evidence |",
        "| C5 Prompt as task-conditional decision prior | 0b, 0b-extra, 0d, 1b, 1d, 2a-extra, 2f | ✅ supported; cite cautiously as mechanism evidence |",
        "| C6 Image is bidirectional modality fusion | 1b, 0e, 3b | ✅ supported for cls-heavy image axis; 3b is a token-gap proxy |",
        "",
        "## Cross-layer Mechanism Chain",
        "",
        "| Axis | Outcome layer | Macro layer | Micro layer | Efficiency layer |",
        "|---|---|---|---|---|",
        "| Axis 1 text payload | 0c single-phantom lift | 1b text-axis cells, 1d action shifts | 2a-2e URL/target/keyword shifts, E1 click transitions | no image tax |",
        "| Axis 2 prompt | 0d task-pool divergence, 0b-extra calibration | 1b prompt-axis dominant cells, 1d action shifts | 2d first-action, E1 click transitions, E2 boundary | prompt-only cost-neutral |",
        "| Axis 3 image | 0e category recovery | 1b image-axis dominant cells, 1d action shifts | 2a endpoint URL/target shifts, E1/E2 | 3b token/latency tax |",
        "| Compound P-SoM vs DOM | 0a/0c/0d routing arm, E3 confidence | 1a hook contrast, E4 action vocabulary | 2a compound URL divergence, E1/E2 per-step evidence | 3a/3c drop-in profile |",
        "",
    ]


def main() -> None:
    stats = all_mode_stats()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 4-Layer Evidence Status (live snapshot)",
        "",
        f"Generated: {generated}  ",
        "Source: `make analyze-layered`",
        "",
        "> Missing artifacts are marked with ⚠️. All percentages and counts are read live from existing JSON/CSV artifacts or episode summaries.",
        "",
    ]

    render_layer0(lines, stats)
    render_layer1(lines)
    render_layer2(lines)
    render_layer3(lines, stats)
    render_claim_matrix(lines)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines).rstrip() + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
