#!/usr/bin/env python3
"""Skeleton for the cost-vs-adjusted-SR Pareto frontier figure.

Future implementation should read each condition's ``condition_summary_v2.json``
for cost fields such as ``avg_total_cost_usd`` or
``avg_total_model_cost_usd``, join those with episode-level adjusted success
rates, then plot one point per (model, mode, site) tuple with the non-dominated
Pareto frontier connected.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phantom_paper/figures/fig7_cost_sr_frontier.png"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_title("Cost vs Adjusted Success Pareto Frontier", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cost per task (USD)")
    ax.set_ylabel("Adjusted success rate (%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.text(
        0.5,
        0.57,
        "Pareto frontier coming soon",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#333333",
    )
    ax.text(
        0.5,
        0.45,
        "TBD: cost data extraction pending",
        ha="center",
        va="center",
        fontsize=12,
        color="#666666",
    )
    ax.text(
        0.5,
        0.32,
        "Planned inputs: condition_summary_v2.json cost fields + episode-level adjusted_SR",
        ha="center",
        va="center",
        fontsize=9,
        color="#777777",
    )
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
