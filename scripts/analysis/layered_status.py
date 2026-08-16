#!/usr/bin/env python3
"""Generate the live 4-dimension evidence status report.

Reads existing analysis artifacts without failing on missing files. The report
is intended as the paper-facing index for docs/checkpoints/paper_planning.md §3.

Output:
- docs/analysis/layered_evidence_status.md (filename retained as CLI alias)
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.analysis.lib.run_registry import get_cells
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.analysis.lib.run_registry import get_cells


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/layered_evidence_status.md"

RESULTS = ROOT / "results/visualwebarena/phase1"
PAPER = ROOT / "results/phantom_paper"
CROSS = ROOT / "docs/analysis/cross_sites"
SR_JSON = CROSS / "sr_per_mode.json"
SR_MD = CROSS / "sr_per_mode.md"


def _mode_specs_from_registry(baseline: str = "B0") -> dict[str, dict[str, Path]]:
    """Resolve live condition directories from the canonical run registry."""
    out: dict[str, dict[str, Path]] = {"classifieds": {}, "reddit": {}}
    for site in out:
        for cell in get_cells(baseline=baseline, site=site):
            out[site][cell.mode] = cell.run_dir / cell.condition_subdir
    return out


MODE_SPECS = _mode_specs_from_registry("B0")

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
    # §139.8 + /stress A1.6 (2026-05-16) hard-delete: post-hoc FP layer
    # retired; `success` is canonical. `raw_*` / `adjusted_*` / `fp_rate`
    # legacy output keys removed.
    n_success = sum(1 for r in rows if bool(r.get("success")))
    tasks = {task_id(r) for r in rows if bool(r.get("success"))}
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
        "n_success": n_success,
        "sr": n_success / n if n else None,
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
            by_cat[cats[tid]].append(bool(row.get("success")))
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
    lift_rows = read_csv(phantom_csv)
    auroc_rows = best_auroc(read_csv(auroc_csv))
    if not SR_JSON.is_file() or not SR_MD.is_file():
        raise RuntimeError(
            "Layer-0 canonical SR sources must exist before status generation: "
            f"json={SR_JSON.is_file()} md={SR_MD.is_file()}"
        )
    sr_payload = read_json(SR_JSON) or {}
    sr_rows = sr_payload.get("summary_table", []) if isinstance(sr_payload, dict) else []
    if not isinstance(sr_rows, list) or not sr_rows:
        raise RuntimeError(f"Layer-0 canonical SR summary_table is empty: {SR_JSON}")
    for row in sr_rows:
        if not isinstance(row, dict) or row.get("complete_exact") is not True:
            continue
        value = row.get("sr_pct")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(
                "Landed exact canonical SR row has no renderable sr_pct: "
                f"{row.get('baseline')}/{row.get('site')}/{row.get('mode')}={value!r}"
            )
    sr_by_key = {
        (row.get("baseline"), row.get("site"), row.get("mode")): row
        for row in sr_rows if isinstance(row, dict)
    }

    lines += ["## Outcome — task 成功 / 路由 arm 证据", ""]

    lines += ["### 0a SR per mode (canonical)", ""]
    if not sr_by_key:
        lines.append(f"- ⚠️ missing or unreadable | {source_line(SR_JSON)}")
    for baseline in ("B0", "B1", "B2"):
        if not any(key[0] == baseline for key in sr_by_key):
            continue
        for site in ("reddit", "classifieds"):
            parts = []
            for mode in ("DOM", "P-text", "P-prompt", "P-SoM", "SoM", "Vision"):
                row = sr_by_key.get((baseline, site, mode))
                value = row.get("sr_pct") if row else None
                rendered = (
                    f"{float(value):.2f}%"
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    else "n/a"
                )
                parts.append(f"{mode} **{rendered}**")
            lines.append(f"- {baseline} {site}: " + "; ".join(parts))
    lines.append(f"- canonical source: `{rel(SR_JSON)}` | last update: {timestamp(SR_JSON)}")
    lines.append(f"- standalone cite source: `{rel(SR_MD)}` | last update: {timestamp(SR_MD)}")
    lines.append("")

    # §139.8 + /stress A1.6 (2026-05-16): the "0b FP rate" block is retired —
    # post-hoc adjusted/raw FP rate is structurally 0 after the upstream
    # B-91 guard + N/A task-load exclusion.

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
        for mode in ["P-SoM", "P-text", "P-prompt"]:
            if mode not in stats[site]:
                continue
            c = depths.get(mode, Counter())
            parts = [f"d{depth}={c.get(depth, 0)}" for depth in range(1, 7)]
            lines.append(f"- {site} {mode}: " + " / ".join(parts))
    lines.append(f"- figure: `{rel(FIGURES['fig0f'])}` | last update: {timestamp(FIGURES['fig0f'])}")
    lines.append("")

    lines += ["### 0g Routing AUROC", ""]
    if not auroc_rows:
        lines.append(f"- ⚠️ missing or unreadable | {source_line(auroc_csv)}")
    for site in ["reddit", "classifieds"]:
        parts = []
        for mode in ["dom", "phantom_dom", "phantom_prompt", "phantom_som", "som", "vision"]:
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

    lines += ["## Macro — agent 平均怎么 act", ""]
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

    lines += ["## Micro — per-step 决策", ""]

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

    lines += ["## Efficiency — cost / latency / carbon", ""]

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
    """Claim -> evidence INDEX. Deliberately carries no numbers.

    Rewritten 2026-08-13. The previous table listed C1-C6 from the pre-REALM framing and
    marked all six supported; four of those claims no longer stand (C1 "P-SoM independent
    routing arm" is the one the rerun floor demoted, §398.8/§406; C2 "4-fold drop-in" is
    the retired hook; C3/C6 belong to the 3-axis mechanism framing shelved 2026-05-14).
    An index that points at retired claims is worse than no index, because it reads as
    live support.

    Per §450.8 this section states no cell count, interval or per-cell figure: those live
    in the producing artifacts, and an index that restates them becomes a second, drifting
    source. Every row points at the artifact instead.
    """
    lines += [
        "## Paper Claim → Evidence Index",
        "",
        "> Rewritten **2026-08-13** against the REALM submission's actual contributions."
        " The retired C1-C6 table is preserved at the bottom of this file, marked"
        " superseded, because it is cited from older notes.",
        "> **No numbers here by design** (§450.8) — this is an index; each row names the"
        " artifact that owns the figure.",
        "",
        "| Claim (as submitted) | Dimensions | Owning artifact | Verdict |",
        "|---|---|---|---|",
        "| (i) The six representations are genuinely complementary: each solves tasks the"
        " others do not, their failures differ structurally, and which one wins is a"
        " property of the deployment | 0a, 0c, 0d, 0e, 0f, 1a-1d, 2a-2f |"
        " `sr_per_mode.md`, `cross_mode_failure_signatures.md`, `phantom_lift.csv` |"
        " ✅ live artifacts present |",
        "| (ii) Most apparent oracle headroom is rerun variance; the bound that survives"
        " the rerun control is the cost ceiling at unchanged success | 0c, 3a |"
        " `noise_floor_inventory.md` |"
        " ⚠️ **three gaps, all load-bearing** — (1) read §1b there: the observed band is a"
        " draw, not a bound, and the threshold it derives is scoped to SAME-arm reruns;"
        " (2) the measured floors cover DOM/SoM/Vision in a single cell, so **no phantom arm"
        " has a clean same-condition floor**, and the phantom arms are the ones the drop-one"
        " hero runs on; (3) the instrument itself is **only well-powered on B0** — discordance"
        " scales with success rate, so the B1 and B2 cells cannot measure their own floor."
        " B0 phantom-arm replicates in flight (`_b1_floor_watcher.sh`) |",
        "| (iii) The benchmarks cannot produce routing supervision: labels arrive at the"
        " success rate, so the tested routing constructions land at or below trivial fixed"
        " policies | 0b, 0b-extra, 0g |"
        " `router_pooled_tier_learnability`, `confidence_cascade.md` |"
        " ✅ five constructions + two controls |",
        "| (iv) **NEW** Which routing question is answerable is decided by label supply AND"
        " signal, and those fail separately | 0b-extra, 0g |"
        " `abstention_learnability.md`, `abstention_site_transfer.md`,"
        " `early_abort_B0_classifieds.md`,"
        " `retry_vs_switch_label_supply.md` | ✅ four questions, one works — see next table."
        " The one that works has now been tested across sites, and it **splits**: ranking"
        " transfers, the operating point does not |",
        "| (v) **NEW** A representation carries deployment properties orthogonal to success"
        " rate | 2g*, 3e*, 3f*, 3g* |"
        " `representation_deployment_profile.md`, `latency_decomposition.md`,"
        " `energy_carbon_audit.md`, `fusion_premium.md` |"
        " ✅ see deployment table below |",
        "",
        "\\* Sub-codes marked with an asterisk are artifact-owned rather than rendered in"
        " the four dimensions above: they are static products, not live snapshots.",
        "",
        "## Routing question → label supply → signal → outcome",
        "",
        "> The finding this table encodes: the circularity named in the draft's §7 has **two"
        " distinguishable failure modes** — a label that does not exist, and a label that"
        " exists with no signal behind it. Only the pre-flight question has both. Figures"
        " live in the artifacts named in the last column.",
        "",
        "| Routing question | Label supply | Signal | Outcome | Artifact |",
        "|---|---|---|---|---|",
        "| Which mode per task | **starved** — most cells admit no classifier under the"
        " min-class rule | — | fails | `router_pooled_tier_learnability` |",
        "| Retry the same arm or switch | adequate, but most of the decision set is"
        " preference-free (both actions fail together) | — | no gain over fixed |"
        " `retry_vs_switch_label_supply.md` |",
        "| Abort at step k | **every episode has one** | **absent** — prefix-only AUROC sits"
        " at its own shuffle null | fails, and loses to truncate-at-k |"
        " `early_abort_B0_classifieds.md` |",
        "| **Abstain before running** | **every task has one** | **present** |"
        " **works within a cell — the only held-out cost saving in the paper.** Across"
        " sites it **splits**: the ranking survives (5 of 6 matched transfers clear a"
        " 200-permutation null; the 6th is indeterminate, not negative), the operating"
        " point does not |"
        " `abstention_learnability.md`, `abstention_site_transfer.md` |",
        "",
        "⚠️ The last row is the only place in this file where a *generalisation* claim is"
        " licensed at all, and it is licensed in one half only. Two sites is two points:"
        " what `abstention_site_transfer` shows is that ranking survived the one site change"
        " available and calibration did not — not that either behaviour would recur on a"
        " third site.",
        "",
        "⚠️ The 0b-extra row above reports whole-episode confidence AUROC, which for any"
        " prefix decision is looking at the future. The prefix-only recomputation is in"
        " `early_abort_B0_classifieds.md`, and it agrees with the April 2026 literature"
        " review's own context line (token-level confidence non-discriminative on this"
        " setup) rather than with 0b-extra.",
        "",
        "## Deployment properties a single-mode deployment cannot measure about itself",
        "",
        "| # | Finding | Dimension | Artifact |",
        "|---|---|---|---|",
        "| D1 | Fusion does not earn its premium against a rerun of one arm | 0a, 0c |"
        " `fusion_premium.md`, `leakage_sensitivity.md` |",
        "| D2 | Unstable element ids move which element is chosen; position-keyed payloads"
        " are unaffected | 2a, 2b | draft §4 (`latency`/`churn` audit) |",
        "| D3 | The feature a practitioner reaches for first (does the task supply a"
        " reference image) does not help | 0g | `router_covariate_baseline` |",
        "| D4 | Which representation wins does not transfer across task sets | 0a, 0d |"
        " `sr_per_mode.md`, registered Jaccard sentinel (0d) |",
        "| D5 | Latency is not where a representation change acts: the model call is a"
        " minority of a step, and the share moves with the serving stack, not with model"
        " size | 3c | `latency_decomposition.md` |",
        "| D6 | Abstention buys a held-out saving; its oracle is far larger and not"
        " reachable | 3a | `abstention_learnability.md` |",
        "| D7 | Failure diagnosability differs sharply by representation — vision's failures"
        " are the least attributable | 2g* | `representation_deployment_profile.md` §1 |",
        "| D8 | The per-step token tail differs by representation far more than the median"
        " does; screenshot-bearing payloads are the flat ones | 3e* |"
        " `representation_deployment_profile.md` §2 |",
        "| D9 | Carbon is not an independent axis on this instrument — it tracks elapsed"
        " time | 3f* | `energy_carbon_audit.md` |",
        "| D10 | A pre-flight abstention policy's **ranking** survives a site change on the"
        " cells that carry enough events to test it; its **operating point** does not, and"
        " the nominal budget axis is **quantised** — several percentage budgets resolve to"
        " the same integer task allowance and are therefore the same policy | 0g, 3a |"
        " `abstention_site_transfer.md` §4 |",
        "| D11 | The one interval that showed fusion significantly beaten anywhere depended on"
        " accumulated site state — but its discordant count is single-digit, so the honest"
        " reading is **underpowered, not corrected**. All **audited** cells now include zero;"
        " 2 of the 8 (`wa_red_*`) have never been audited for this defect | 0a, 0c |"
        " `leakage_sensitivity.md` §1 (the `d` column) + §3 |",
        "",
        "⚠️ **Provenance boundary on every B0 figure in this file (B-1970, 2026-08-16).**"
        " The AWS proxy changed its response shape between the last archived B0 run and"
        " 2026-08-16. The drift was representation-only and lost no information, but it"
        " establishes that the provider mutates without notice: **archived B0 data and any"
        " future B0 data are not on the same provider snapshot**, and any analysis that"
        " subtracts one from the other has to say so. → `master_bug_catalog` B-1970.",
        "",
        "## ⚠️ SUPERSEDED — pre-REALM claim matrix and 3-axis mechanism chain",
        "",
        "> Kept because older notes cite C1-C6 and the axis chain by name. **Do not read"
        " these as live support.** C1 is the claim the rerun floor demoted (§398.8/§406);"
        " C2 is the retired 4-fold drop-in hook; C3/C6 belong to the 3-axis mechanism"
        " framing shelved 2026-05-14 with §5. The dimension pointers in them remain"
        " accurate as pointers.",
        "",
        "| Claim | Dimensions cited | Verdict (as of the retired framing) |",
        "|---|---|---|",
        "| C1 P-SoM independent routing arm | 0a, 0c, 0d, 0g, 1a, 2a | ⚠️ superseded |",
        "| C2 4-fold drop-in property | 3a, 3c, 0g, 0c | ⚠️ superseded |",
        "| C3 3-axis hierarchical theory | 1b, 2a-2e, axis chain | ⚠️ shelved with §5 |",
        "| C4 Aggregate macro can mislead about routing potential | 1a, 0d, 2a |"
        " ↺ survives in contribution (i) |",
        "| C5 Prompt as task-conditional decision prior | 0b, 0b-extra, 0d, 1b, 1d,"
        " 2a-extra, 2f | ⚠️ mechanism reading shelved; behavioural rows stand |",
        "| C6 Image is bidirectional modality fusion | 1b, 0e, 3b | ⚠️ shelved with §5 |",
        "",
        "| Axis (retired framing) | Outcome | Macro | Micro | Efficiency |",
        "|---|---|---|---|---|",
        "| Axis 1 text payload | 0c single-phantom lift | 1b, 1d | 2a-2e, E1 | no image tax |",
        "| Axis 2 prompt | 0d, 0b-extra | 1b, 1d | 2d, E1, E2 | prompt-only cost-neutral |",
        "| Axis 3 image | 0e category recovery | 1b, 1d | 2a, E1/E2 | 3b token/latency tax |",
        "| Compound P-SoM vs DOM | 0a/0c/0d, E3 | 1a, E4 | 2a, E1/E2 | 3a/3c |",
        "",
        "⚠️ Naming trap: `mechanism_per_task.json` holds E1-E4, which are **behavioural**"
        " metrics (click-target divergence, trajectory boundary, confidence calibration,"
        " action vocabulary), not mechanistic evidence. The mechanistic line (activation"
        " patching, layer probes) is shelved, and its linear-probe results were themselves"
        " ruled the wrong tool for this contrastive setup (§111.2).",
        "",
    ]


def main() -> None:
    stats = all_mode_stats()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 4-dimension Evidence Status (live snapshot)",
        "",
        f"Generated: {generated}",
        "Source: `make analyze-layered` (CLI alias preserved)",
        "",
        "> Four orthogonal dimensions: Outcome / Macro / Micro / Efficiency. Sub-codes (0a / 1c / 2a / 3d) remain as figure-internal anchors.",
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
