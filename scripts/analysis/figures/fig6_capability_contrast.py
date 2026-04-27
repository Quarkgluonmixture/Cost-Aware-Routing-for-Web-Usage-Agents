#!/usr/bin/env python3
"""B0-vs-B1 capability contrast from disagreement cluster analysis."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "results/phantom_paper/analyses/disagreement_clusters.md"
OUT = ROOT / "results/phantom_paper/figures/fig6_capability_contrast.png"

LABELS = {
    "DOM visual-missing": "DOM\nvisual\nmissing",
    "DOM search-loop": "DOM\nsearch\nloop",
    "SoM early-finish/wrong-commit": "SoM\nearly\nfinish",
    "SoM visual-hijack/click-loop": "SoM\nvisual\nhijack",
    "Vision text/grounding loops": "Vision\ntext\nloops",
    "Vision element-misground": "Vision\nelement\nmisground",
}


def parse_contrast_table() -> list[dict[str, float | str]]:
    text = ANALYSIS.read_text()
    start = text.index("### B0-vs-B1 contrast table")
    rows: list[dict[str, float | str]] = []
    for line in text[start:].splitlines():
        if not line.startswith("| "):
            if rows:
                break
            continue
        if "pattern" in line.lower() or "share" in line.lower() or "---" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 6:
            continue
        pattern, b0_count, b0_share, b1_count, b1_share, shift = parts[:6]
        match = re.search(r"([+-]?\d+(?:\.\d+)?)", shift)
        rows.append(
            {
                "pattern": pattern,
                "b0_count": b0_count,
                "b0_share": float(b0_share.rstrip("%")),
                "b1_count": b1_count,
                "b1_share": float(b1_share.rstrip("%")),
                "shift": float(match.group(1)) if match else float("nan"),
            }
        )
    if not rows:
        raise RuntimeError(f"Could not parse contrast table from {ANALYSIS}")
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = parse_contrast_table()
    patterns = [str(row["pattern"]) for row in rows]
    xlabels = [LABELS.get(pattern, pattern) for pattern in patterns]
    b0 = np.array([float(row["b0_share"]) for row in rows])
    b1 = np.array([float(row["b1_share"]) for row in rows])
    shift = np.array([float(row["shift"]) for row in rows])
    x = np.arange(len(rows))
    width = 0.36

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.55, 1.0]})

    ax = axes[0]
    bars0 = ax.bar(x - width / 2, b0, width, label="B0", color="#4c78a8")
    bars1 = ax.bar(x + width / 2, b1, width, label="B1", color="#f58518")
    highlight = patterns.index("SoM visual-hijack/click-loop")
    for bars in (bars0, bars1):
        bars[highlight].set_edgecolor("#b91c1c")
        bars[highlight].set_linewidth(2.2)
    ax.set_title("Failure-pattern shares (cls + red aggregate)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Share within mode-specific failures (%)")
    ax.set_xticks(x, xlabels)
    ax.set_ylim(0, max(b0.max(), b1.max()) + 14)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for bars in (bars0, bars1):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.annotate(
        "+43.7 pp",
        xy=(highlight + width / 2, b1[highlight]),
        xytext=(highlight + 0.75, b1[highlight] + 9),
        arrowprops={"arrowstyle": "->", "color": "#b91c1c", "lw": 1.6},
        color="#b91c1c",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[1]
    colors = ["#b91c1c" if val > 0 else "#2563eb" for val in shift]
    bars = ax.barh(np.arange(len(rows)), shift, color=colors, alpha=0.86)
    bars[highlight].set_edgecolor("#7f1d1d")
    bars[highlight].set_linewidth(2.2)
    ax.axvline(0, color="#444444", linewidth=1.0)
    ax.set_title("B1 - B0 shift", fontsize=12, fontweight="bold")
    ax.set_yticks(np.arange(len(rows)), xlabels)
    ax.set_xlabel("Shift in percentage points")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, shift):
        xpos = val + (1.0 if val >= 0 else -1.0)
        ax.text(
            xpos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8.5,
        )

    fig.suptitle("Capability x Representation Failure Shift", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Source: disagreement_clusters.md B0-vs-B1 contrast table. The table is aggregate across classifieds and reddit; site-specific contrast was not present in the source.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
