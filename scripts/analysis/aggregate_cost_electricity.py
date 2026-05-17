#!/usr/bin/env python3
"""[Efficiency 3a + 3d] Efficiency dimension — principled cost aggregation across B0/B1/B2.

v6 update 2026-05-16: B2 (Gemma3-VL local 4B, added 2026-05-14) added to electricity-
equivalent cost class alongside B1. See cell-level branch + paper §6 disclosure.

B0 cost = API token cost (avg_total_cost_usd from Qwen3-VL-235B-A22B per-token rates)
B1 cost = electricity-equivalent (avg_total_energy_kwh × electricity price)

Rationale: B0 (API call) and B1 (local 4B inference) belong to DIFFERENT cost
classes. The current condition_summary_v2.json applies the same per-token rate
to both, which makes B1 cost APPEAR similar to B0 (~$0.05/ep). That's a
methodological artifact — B1 has no actual API dollars. The principled
comparison is API cost vs electricity cost, with explicit annotation that
the two are not directly ratio-comparable in dollar terms (different deployment
classes).

Output:
- docs/analysis/cross_sites/cost_per_mode.json (machine-readable)
- docs/analysis/cross_sites/cost_per_mode.md   (paper-ready table)

See paper_planning.md §3 Efficiency dimension (sub-code 3d) framework.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.analysis.lib.run_registry import PAPER_MODES, get_cells

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT_JSON = ROOT / "docs/analysis/cross_sites/cost_per_mode.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/cost_per_mode.md"

# Industrial electricity price (UK avg ~£0.10/kWh ≈ $0.12 — pessimistic for
# DGX Spark in user's region; matches `region: uk` in B1 runs' energy config).
ELECTRICITY_USD_PER_KWH = 0.12

# Efficiency 3a token cost field (computed at run-time using metrics.cost_api rates).
# For B0 this = API token cost. For B1 this = same rate x token count, but B1
# pays $0 in actual API dollars, so we mark it "non-comparable" and report
# electricity-equivalent instead.
def _condition_subpath(cell) -> str:
    return str((cell.run_dir / cell.condition_subdir).relative_to(RESULTS))


def _runs_from_registry() -> dict[str, dict[str, dict[str, str]]]:
    runs: dict[str, dict[str, dict[str, str]]] = {}
    for baseline in ("B0", "B1", "B2"):
        runs[baseline] = {}
        for site in ("reddit", "classifieds"):
            specs = get_cells(baseline=baseline, site=site)
            runs[baseline][site] = {
                cell.mode: _condition_subpath(cell)
                for cell in specs
                if cell.mode in PAPER_MODES
            }
    return runs


RUNS = _runs_from_registry()


def safe_load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect_cell(baseline: str, site: str, mode: str, sub: str) -> dict:
    p = RESULTS / sub / "condition_summary_v2.json"
    d = safe_load(p)
    if d is None:
        return {"available": False, "reason": "missing condition_summary_v2.json"}
    avg_token_cost = d.get("avg_total_cost_usd")
    avg_energy_kwh = d.get("avg_total_energy_kwh")
    avg_co2e_kg = d.get("avg_total_co2e_kg")
    avg_steps = d.get("avg_steps")
    cell = {
        "available": True,
        "run_subpath": sub,
        "avg_steps": avg_steps,
        # Efficiency 3a token cost (B0 = real API $; B1 = artifact, see notes)
        "avg_token_cost_usd_yaml_rate": avg_token_cost,
        # Efficiency 3d energy + electricity cost (only B1 has reliable energy data)
        "avg_energy_kwh": avg_energy_kwh,
        "avg_electricity_usd": (
            None
            if avg_energy_kwh is None
            else avg_energy_kwh * ELECTRICITY_USD_PER_KWH
        ),
        "avg_co2e_kg": avg_co2e_kg,
    }
    if baseline == "B0":
        cell["paper_cost_usd"] = avg_token_cost  # paid API dollars
        cell["paper_cost_class"] = "API_token_dollars"
    elif baseline in ("B1", "B2"):
        # v6 fix (P1-13, codex pre-fire #11): B2 (Gemma3-VL local 4B inference, added
        # 2026-05-14) belongs to same "electricity-equivalent" cost class as B1 (Qwen3-VL
        # local 4B). Previously fell into else branch (treated as B1) but docstring +
        # markdown reporting only declared B0/B1 — paper §6 B2 cost story was structurally
        # absent. Now explicit branch + class label.
        cell["paper_cost_usd"] = (
            None
            if avg_energy_kwh is None
            else avg_energy_kwh * ELECTRICITY_USD_PER_KWH
        )
        cell["paper_cost_class"] = "electricity_equivalent"
    else:
        # Unknown baseline (e.g., legacy "?") — skip cost assignment, paper §6 must exclude
        cell["paper_cost_usd"] = None
        cell["paper_cost_class"] = "unknown_baseline"
    return cell


def main() -> None:
    cells: dict[str, dict[str, dict[str, dict]]] = {}
    for baseline, sites in RUNS.items():
        cells[baseline] = {}
        for site, modes in sites.items():
            cells[baseline][site] = {}
            for mode, sub in modes.items():
                if sub is None:
                    cells[baseline][site][mode] = {"available": False, "reason": "no run dir"}
                    continue
                cells[baseline][site][mode] = collect_cell(baseline, site, mode, sub)

    # Cross-class summary stats
    summary = {
        "method": (
            "B0 reports avg_total_cost_usd from per-token API rates (Qwen3-VL-235B-A22B "
            "$0.001/1k input, $0.005/1k output). B1 reports avg_total_energy_kwh × "
            f"${ELECTRICITY_USD_PER_KWH:.2f}/kWh as electricity-equivalent cost — local "
            "inference pays no API dollars; the per-token cost field in B1 "
            "condition_summary_v2.json is artifact (uses B0 rates) and is NOT comparable. "
            "B0 vs B1 dollar costs belong to different classes (API call cost vs "
            "electricity), so the paper presents both side-by-side, not a single ratio."
        ),
        "electricity_usd_per_kwh": ELECTRICITY_USD_PER_KWH,
        # A1.21 P1-4 fix (B-532, codex F7): B2 cost class added — was structurally
        # absent from reviewer-facing metadata despite being collected (loop on L56-59).
        "cost_classes": {
            "B0": "API_token_dollars (Qwen3-VL-235B-A22B per DashScope pricing)",
            "B1": "electricity_equivalent (DGX Spark, UK industrial rate)",
            "B2": "electricity_equivalent (DGX Spark, UK industrial rate; "
                  "Gemma3-VL google/gemma-3-4b-it local 4B inference, same "
                  "deployment class as B1 — added 2026-05-14 advisor lock)",
        },
        "paper_caveat": (
            "The qualitative cost gap between API and local inference is large "
            "(2–3 orders of magnitude per these data) but is fundamentally a "
            "deployment-mode comparison, not a model-size ratio. Reporting a single "
            "multiplier (e.g. '30x') without specifying the cost class is misleading."
        ),
    }

    # Build B0 vs B1 deployment-class ratio per site (informative, not a paper claim)
    ratios: dict[str, dict] = {}
    for site in ("reddit", "classifieds"):
        b0_cells = cells.get("B0", {}).get(site, {})
        b1_cells = cells.get("B1", {}).get(site, {})
        b0_costs = [
            cell["paper_cost_usd"]
            for cell in b0_cells.values()
            if cell.get("paper_cost_usd") is not None
        ]
        b1_costs = [
            cell["paper_cost_usd"]
            for cell in b1_cells.values()
            if cell.get("paper_cost_usd") is not None
        ]
        if b0_costs and b1_costs:
            avg_b0 = sum(b0_costs) / len(b0_costs)
            avg_b1 = sum(b1_costs) / len(b1_costs)
            ratios[site] = {
                "avg_B0_API_dollars": round(avg_b0, 6),
                "avg_B1_electricity_dollars": round(avg_b1, 8),
                "ratio_B0_over_B1": round(avg_b0 / avg_b1, 2) if avg_b1 > 0 else None,
                "note": "ratio across deployment classes, not capability tiers",
            }

    out = {
        "summary": summary,
        "cells": cells,
        "deployment_class_ratios": ratios,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[json] {OUT_JSON}")

    # Markdown report
    lines: list[str] = []
    lines.append("# Efficiency — Cost Per Mode (deployment-class aware)\n")
    lines.append(summary["method"] + "\n")
    lines.append("## B0 — API token dollars (paid)\n")
    lines.append("| site | mode | avg_steps | avg_total_cost_usd ($/ep) |")
    lines.append("|---|---|---:|---:|")
    for site in ("reddit", "classifieds"):
        for mode, cell in cells["B0"][site].items():
            if not cell.get("available"):
                lines.append(f"| {site} | {mode} | n/a | n/a (pending) |")
                continue
            cost_str = (
                f"${cell['paper_cost_usd']:.4f}"
                if cell.get("paper_cost_usd") is not None
                else "n/a"
            )
            steps = cell.get("avg_steps")
            steps_str = f"{steps:.1f}" if isinstance(steps, (int, float)) else "n/a"
            lines.append(f"| {site} | {mode} | {steps_str} | {cost_str} |")
    lines.append("")

    lines.append("## B1 — electricity equivalent ($/ep)\n")
    lines.append(
        f"Computed as `avg_total_energy_kwh × ${ELECTRICITY_USD_PER_KWH:.2f}/kWh` "
        "(DGX Spark, UK industrial rate per `metrics.energy.region: uk` in B1 yaml).\n"
    )
    lines.append("| site | mode | avg_steps | avg_energy_kwh | avg_co2e_kg | avg_electricity_usd ($/ep) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for site in ("reddit", "classifieds"):
        for mode, cell in cells["B1"][site].items():
            if not cell.get("available"):
                lines.append(f"| {site} | {mode} | n/a | n/a | n/a | n/a (pending) |")
                continue
            steps = cell.get("avg_steps")
            steps_str = f"{steps:.1f}" if isinstance(steps, (int, float)) else "n/a"
            kwh = cell.get("avg_energy_kwh")
            kwh_str = f"{kwh:.5f}" if isinstance(kwh, (int, float)) else "n/a"
            co2 = cell.get("avg_co2e_kg")
            co2_str = f"{co2:.5f}" if isinstance(co2, (int, float)) else "n/a"
            usd = cell.get("paper_cost_usd")
            usd_str = (
                f"${usd:.6f}"
                if isinstance(usd, (int, float))
                else "n/a (run not yet complete)"
            )
            lines.append(f"| {site} | {mode} | {steps_str} | {kwh_str} | {co2_str} | {usd_str} |")
    lines.append("")

    # A1.21 P1-4 fix (B-532, codex F7): B2 cost markdown section added — was
    # structurally absent from reviewer-facing report despite data collection.
    lines.append("## B2 — electricity equivalent ($/ep, Gemma3-VL local 4B)\n")
    lines.append(
        f"Computed as `avg_total_energy_kwh × ${ELECTRICITY_USD_PER_KWH:.2f}/kWh` "
        "(DGX Spark, UK industrial rate per `metrics.energy.region: uk` in B2 yaml). "
        "Same deployment class as B1 (per advisor §138 B2 ≈ B1 matched-capability lock).\n"
    )
    lines.append("| site | mode | avg_steps | avg_energy_kwh | avg_co2e_kg | avg_electricity_usd ($/ep) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for site in ("reddit", "classifieds"):
        for mode, cell in cells.get("B2", {}).get(site, {}).items():
            if not cell.get("available"):
                lines.append(f"| {site} | {mode} | n/a | n/a | n/a | n/a (pending) |")
                continue
            steps = cell.get("avg_steps")
            steps_str = f"{steps:.1f}" if isinstance(steps, (int, float)) else "n/a"
            kwh = cell.get("avg_energy_kwh")
            kwh_str = f"{kwh:.5f}" if isinstance(kwh, (int, float)) else "n/a"
            co2 = cell.get("avg_co2e_kg")
            co2_str = f"{co2:.5f}" if isinstance(co2, (int, float)) else "n/a"
            usd = cell.get("paper_cost_usd")
            usd_str = (
                f"${usd:.6f}"
                if isinstance(usd, (int, float))
                else "n/a (run not yet complete)"
            )
            lines.append(f"| {site} | {mode} | {steps_str} | {kwh_str} | {co2_str} | {usd_str} |")
    lines.append("")

    lines.append("## Deployment-class ratio (informative, not a paper claim)\n")
    lines.append("| site | avg B0 API ($/ep) | avg B1 electricity ($/ep) | ratio (B0/B1) |")
    lines.append("|---|---:|---:|---:|")
    for site, r in ratios.items():
        ratio_str = (
            f"{r['ratio_B0_over_B1']:.0f}×"
            if r.get("ratio_B0_over_B1") is not None
            else "n/a"
        )
        lines.append(
            f"| {site} | ${r['avg_B0_API_dollars']:.4f} | ${r['avg_B1_electricity_dollars']:.6f} | {ratio_str} |"
        )
    lines.append("")
    lines.append(summary["paper_caveat"])
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"[md]   {OUT_MD}")


if __name__ == "__main__":
    main()
