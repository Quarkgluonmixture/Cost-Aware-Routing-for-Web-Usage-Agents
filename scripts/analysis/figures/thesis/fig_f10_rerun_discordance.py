#!/usr/bin/env python3
"""Thesis F10 — how much a single rerun moves the per-task outcome (half-column).

Deliberately small. F10b already carries the argument (a representation arm is
worth no more than a rerun arm); this figure only answers the question a reader
asks immediately afterwards — "how was that floor measured?" — by showing the
task-level disagreement between two runs of ONE (cell, mode).

Scope is one cell, B0 x VWA-classifieds, with six independently replicated
arms. Until 2026-08-03 only dom and vision existed and the fused arm's floor was
BORROWED from them; the som pair then landed inside the borrowed band, which is
why C3 no longer rests on an extrapolation. WebArena also has a replicate, but it
pools five modes over ten shared tasks and is not a floor for its own baseline
arm, so it is named in the caption rather than plotted next to these.

Cohen's kappa exists only for the two pairs `phase0b_noise_floor.md` computed; it
is shown where available and simply absent otherwise, never carried across arms.

Numbers are parsed from the two source documents, never hardcoded.

Output: final_dissertation/figures/fig_f10_rerun_discordance.{png,pdf}
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import _style as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SRC_INV = ROOT / "docs/analysis/cross_sites/noise_floor_inventory.md"
SRC_K = ROOT / "docs/analysis/cross_sites/phase0b_noise_floor.md"
OUT = ROOT / "final_dissertation/figures/fig_f10_rerun_discordance"

C_BAR = "#D55E00"
C_K = "#5A5A5A"

# TERMS.md locks the paper-facing spelling; capitalize() would give "Som".
MODE_LABEL = {"dom": "DOM", "vision": "Vision", "som": "SoM",
              "ptext": "P-text", "pprompt": "P-prompt", "psom": "P-SoM"}

# §1 floors: | `B0.cls.dom` | scope | 224 | **7.14pp** | **4.91pp** | 12.05% |
RE_FLOOR = re.compile(
    r"\|\s*`(B0\.cls\.(\w+))`[^|]*\|\s*[^|]*[A-Za-z][^|]*\|\s*(\d+)\s*\|\s*"
    r"\*\*([\d.]+)pp\*\*\s*\|\s*\*\*([\d.]+)pp\*\*\s*\|\s*([\d.]+)%\s*\|")
# phase0b §2: | **dom** R... | ... | ... | ... | ... | 12.1pp | 0.559 | ... |
RE_KAPPA = re.compile(
    r"\|\s*\*\*(\w+)\*\*\s*R\d+\s*↔\s*R\d+\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|"
    r"\s*([\d.]+)pp\s*\|\s*([\d.]+)\s*\|")


def parse():
    inv = SRC_INV.read_text(encoding="utf-8")
    rows = []
    for _pair, mode, n, _a, _b, disc in RE_FLOOR.findall(inv):
        rows.append({"mode": mode, "n": int(n), "disc": float(disc), "k": None})
    if len(rows) < 2:
        raise SystemExit(f"{SRC_INV}: parsed {len(rows)} B0.cls pairs — "
                         "refusing to plot a partial figure")
    kap = {m: float(k) for m, _d, k in RE_KAPPA.findall(
        SRC_K.read_text(encoding="utf-8"))}
    for r in rows:
        r["k"] = kap.get(r["mode"])
    rows.sort(key=lambda r: r["disc"], reverse=True)
    return rows


def build(ax, rows):
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        ax.barh(yi, r["disc"], height=0.46, color=C_BAR, zorder=3)
        ax.text(r["disc"] + 0.35, yi, f"{r['disc']:.2f}%", va="center",
                fontsize=S.FS_LABEL, color=C_BAR, fontweight="bold")
        if r["k"] is not None:
            ax.text(r["disc"] + 3.4, yi, f"κ = {r['k']:.3f}", va="center",
                    fontsize=S.FS_VALUE, color=C_K)

    ax.set_yticks(y, [MODE_LABEL.get(r["mode"], r["mode"])
                      for r in rows], fontsize=S.FS_PANEL)
    ax.set_xlim(0, max(r["disc"] for r in rows) + 7.0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("tasks whose outcome flips between two runs of one mode  [%]")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows = parse()
    n = rows[0]["n"]
    no_k = [r["mode"] for r in rows if r["k"] is None]

    S.apply()
    fig, ax = plt.subplots(figsize=(0.72 * S.PRINT_W_IN, 2.1))
    build(ax, rows)
    # Cell, sample size, which arms lack a kappa, and why the WebArena
    # replicate is excluded are all caption material now.

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   ({len(rows)} arms, "
          f"kappa for {len(rows) - len(no_k)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
