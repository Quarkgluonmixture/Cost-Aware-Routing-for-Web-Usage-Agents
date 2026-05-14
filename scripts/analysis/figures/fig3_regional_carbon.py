#!/usr/bin/env python3
"""[Efficiency supporting] Efficiency dimension — regional carbon sensitivity.

Output:
- results/phantom_paper/figures/fig3_regional_carbon.png

Regional carbon sensitivity from B1 measured episode energy.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from p79.experiment.energy_tracker import REGION_INTENSITY_G_PER_KWH


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT = ROOT / "results/phantom_paper/figures/fig3_regional_carbon.png"

MODES = ["DOM", "SoM", "Vision"]
COLORS = {
    "DOM": "#4c78a8",
    "SoM": "#f58518",
    "Vision": "#54a24b",
    "Phantom-SoM": "#b279a2",
    "P-text": "#e45756",
    "Phantom-prompt": "#9467bd",
}
# F40 audit fix 2026-05-09: RUNS resolved via run_registry instead of
# hardcoded archived run paths. Lazy at runtime so import doesn't fail
# when paper-grade cells absent.
import os as _os
import sys as _sys
_sys.path.insert(0, str(ROOT / "scripts/analysis"))
from lib.run_registry import get_cells as _get_cells  # noqa: E402

# §139.8: scored-set sizes (total − N/A excluded at load) from the single
# source of truth, not pre-exclusion 234/210. Labels derive from the count.
from p79.experiment.analysis import scored_task_count as _scored_task_count
_SITE_N = {_s: _scored_task_count(_s, "visualwebarena") for _s in ("classifieds", "reddit")}
_SITE_LABELS = {
    "classifieds": f"Classifieds (N={_SITE_N['classifieds']})",
    "reddit": f"Reddit (N={_SITE_N['reddit']})",
}


def _resolve_runs(grade: list | None = None) -> dict:
    """Map (site → {expected, label, DOM, SoM, Vision episodes}) using
    run_registry. Raises RuntimeError if any required B1 cell missing.
    """
    out: dict[str, dict] = {}
    missing: list[str] = []
    if grade is None:
        env_grade = _os.environ.get("P79_AGGREGATOR_GRADE", "")
        grade = [g.strip() for g in env_grade.split(",") if g.strip()] or None
    for site in ("classifieds", "reddit"):
        site_entry = {"expected": _SITE_N[site], "label": _SITE_LABELS[site]}
        for mode in ("DOM", "SoM", "Vision"):
            cells = _get_cells(baseline="B1", site=site, mode=mode, grade=grade)
            if not cells:
                missing.append(f"B1 {site} {mode}")
                continue
            site_entry[mode] = cells[0].episodes_dir
        out[site] = site_entry
    if missing:
        raise RuntimeError(
            "fig3_regional_carbon: missing paper-grade cells "
            f"{missing}. Update run_manifest.yaml or set "
            "P79_AGGREGATOR_GRADE=archived for legacy sensitivity."
        )
    return out


# Lazy module-level placeholder; populated by main() at runtime.
RUNS: dict[str, dict] = {}
DISPLAY_NAME = {
    "norway": "Norway",
    "france": "France",
    "usa": "USA",
    "china": "China",
    "india": "India",
    "poland": "Poland",
    "south_africa": "South Africa",
}
REPRESENTATIVE_REGIONS = ["norway", "france", "usa", "china", "india", "poland", "south_africa"]


def task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)_summary", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def median_energy_kwh(ep_dir: Path, expected: int) -> tuple[float, int]:
    values: list[float] = []
    seen: set[int] = set()
    for path in sorted(ep_dir.glob("*_summary_v2.json")):
        tid = task_id(path)
        if tid in seen:
            continue
        seen.add(tid)
        with path.open() as f:
            record = json.load(f)
        value = record.get("total_energy_kwh")
        if value is not None:
            values.append(float(value))
    if len(seen) != expected:
        print(f"[warn] {ep_dir}: summaries n={len(seen)}/{expected}")
    if len(values) != len(seen):
        print(f"[warn] {ep_dir}: energy n={len(values)}/{len(seen)}")
    if not values:
        raise RuntimeError(f"No total_energy_kwh values under {ep_dir}")
    return statistics.median(values), len(values)


def region_items() -> list[tuple[str, float]]:
    # The table has 45 entries including the world average. The deployment
    # sensitivity axis excludes "world" and uses it as a reference intensity.
    return sorted(
        [(region, intensity) for region, intensity in REGION_INTENSITY_G_PER_KWH.items() if region != "world"],
        key=lambda item: (item[1], item[0]),
    )


def region_positions(items: list[tuple[str, float]]) -> dict[str, int]:
    return {region: index for index, (region, _) in enumerate(items)}


def draw_world_reference(ax: plt.Axes, items: list[tuple[str, float]], energies: dict[str, tuple[float, int]]) -> None:
    intensities = [value for _, value in items]
    world = REGION_INTENSITY_G_PER_KWH["world"]
    # Interpolate the position where 475 gCO2/kWh falls on the sorted region axis.
    for index in range(len(intensities) - 1):
        lo, hi = intensities[index], intensities[index + 1]
        if lo <= world <= hi:
            frac = 0.0 if hi == lo else (world - lo) / (hi - lo)
            xpos = index + frac
            break
    else:
        xpos = len(intensities) - 1
    ax.axvline(xpos, color="#555555", linestyle="--", linewidth=1.0, alpha=0.65)
    som_world_y = energies["SoM"][0] * world
    ax.axhline(som_world_y, color=COLORS["SoM"], linestyle="--", linewidth=1.0, alpha=0.38)
    ax.text(
        xpos + 0.25,
        0.96,
        "world avg\n475 g/kWh",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
    )
    ax.text(
        len(items) - 1.0,
        som_world_y,
        "SoM @ world avg",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#9a4f08",
    )


def draw_site(ax: plt.Axes, site: str, items: list[tuple[str, float]], energies: dict[str, tuple[float, int]]) -> None:
    x = list(range(len(items)))
    intensities = [intensity for _, intensity in items]
    for mode in MODES:
        median_kwh, _ = energies[mode]
        y = [median_kwh * intensity for intensity in intensities]
        ax.plot(x, y, color=COLORS[mode], linewidth=2.0, marker="o", markersize=2.6, label=mode)

    draw_world_reference(ax, items, energies)
    pos = region_positions(items)
    tick_regions = [region for region in REPRESENTATIVE_REGIONS if region in pos]
    tick_positions = [pos[region] for region in tick_regions]
    ax.set_xticks(tick_positions, [DISPLAY_NAME[region] for region in tick_regions], rotation=30, ha="right")
    ax.set_xlim(-0.75, len(items) - 0.25)
    ax.set_title(RUNS[site]["label"], fontsize=12, fontweight="bold")
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Median per-task CO2 (g)")
    ax.set_xlabel("Deployment region, sorted by carbon intensity")

    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(tick_positions)
    top.set_xticklabels([f"{REGION_INTENSITY_G_PER_KWH[region]:.0f}" for region in tick_regions], fontsize=8)
    top.set_xlabel("Carbon intensity (gCO2/kWh)", fontsize=9)

    for region in ["france", "usa", "china", "india", "poland"]:
        if region not in pos:
            continue
        xpos = pos[region]
        ax.axvline(xpos, color="#999999", linestyle=":", linewidth=0.7, alpha=0.45)
        som_y = energies["SoM"][0] * REGION_INTENSITY_G_PER_KWH[region]
        ax.annotate(
            DISPLAY_NAME[region],
            xy=(xpos, som_y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#444444",
            arrowprops={"arrowstyle": "-", "color": "#aaaaaa", "lw": 0.6},
        )


def main() -> None:
    # F40 audit 2026-05-09: resolve RUNS at runtime via run_registry.
    global RUNS
    RUNS = _resolve_runs()

    items = region_items()
    print(
        f"Loaded {len(REGION_INTENSITY_G_PER_KWH)} carbon-intensity entries "
        f"({len(items)} deployment regions + world reference)"
    )
    print(f"Region range: {items[0][0]}={items[0][1]:.0f} -> {items[-1][0]}={items[-1][1]:.0f} gCO2/kWh")

    all_energies: dict[str, dict[str, tuple[float, int]]] = {}
    for site, spec in RUNS.items():
        all_energies[site] = {}
        for mode in MODES:
            median_kwh, n = median_energy_kwh(spec[mode], spec["expected"])
            all_energies[site][mode] = (median_kwh, n)
            print(f"{site} {mode}: median_energy={median_kwh:.8f} kWh n={n}")
        for region in ["norway", "france", "usa", "china", "india", "poland", "south_africa"]:
            vals = ", ".join(
                f"{mode}={all_energies[site][mode][0] * REGION_INTENSITY_G_PER_KWH[region]:.3f}g"
                for mode in MODES
            )
            print(f"{site} {DISPLAY_NAME[region]} ({REGION_INTENSITY_G_PER_KWH[region]:.0f}): {vals}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.7), sharey=False)
    for ax, site in zip(axes, ["classifieds", "reddit"]):
        draw_site(ax, site, items, all_energies[site])

    pending_handles = [
        Line2D([0], [0], color=COLORS["Phantom-SoM"], linestyle=":", lw=2, label="Phantom-SoM (B1) pending"),
        Line2D([0], [0], color=COLORS["P-text"], linestyle=":", lw=2, label="P-text (B1) pending"),
        Line2D([0], [0], color=COLORS["Phantom-prompt"], linestyle=":", lw=2, label="P-prompt (B1) pending"),
    ]
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles + pending_handles, labels + [h.get_label() for h in pending_handles], loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Regional Carbon Sensitivity (B1 Qwen3-VL-4B local; B0 235B proxy API not measurable)",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Energy from B1 (Qwen3-VL-4B) NVML measurement on cls (N=234) / red (N=210). "
        "Per-region intensity from IEA 2023 / ElectricityMaps. Phantom-SoM/P-text/P-prompt pending B1 re-run; expected to be comparable or lower than DOM. "
        "B0 (Qwen3-VL-235B via proxy API) energy is not directly observable on local hardware; B1 measurement serves as a lower-bound reference for representation-driven carbon sensitivity.",
        ha="center",
        fontsize=7.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
