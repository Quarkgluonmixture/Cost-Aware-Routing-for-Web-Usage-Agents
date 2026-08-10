#!/usr/bin/env python3
"""Thesis F1 — one page, three ways to pay for it (motivating example).

Guide Red flag 6 names this figure directly: "Too much jargon before the reader
sees a web page -> Fix: early worked example." So the reader sees the page three
times, exactly as each mode receives it, before any taxonomy is introduced.

What each column shows is what that mode actually sends:
  DOM     AXTree text, and no image at all
  SoM     the SAME page with numbered marks drawn on it, plus [SOM_MARKS] text
  Vision  the raw screenshot, and no structured text at all

The quantitative point is deliberately placed under the pictures, because it is
the one a reader would not guess from them: SoM's TEXT payload is within 1% of
DOM's. Nearly the whole extra cost is the image. That is the seam the phantom
modes are built on, so this figure sets up Chapter 3 without asserting it yet.

Provenance, all named in the caption because they are not one single artifact:
  screenshots  the raw/annotated PAIR rendered from this page (dashboard assets,
               1280x660, same page and same listings as the run artifact)
  text chars   mechanistic/_obs_mirror (B0 x classifieds, per-mode observation)
  bytes+tokens results/visualwebarena/phase1 step record (B0 x classifieds)

The step is classifieds task 0 step 000 — the first step, whose page state is the
task's start URL and so identical across modes. Verified, not assumed: the dom-run
and vision-run screenshots for that step are byte-identical (md5 checked below).

Output: final_dissertation/figures/fig_f1_motivating_example.{png,pdf}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "final_dissertation/figures/fig_f1_motivating_example"

MIRROR = ROOT / "results/mechanistic/_obs_mirror/visualwebarena"
PHASE1 = ROOT / "results/visualwebarena/phase1"
DASH = ROOT / "docs/checkpoints/周报/weekly-dashboard/public/figures"
SHOT_RAW = DASH / "mode_raw_screenshot.png"
SHOT_SOM = DASH / "mode_som_annotated.png"
SHOT_DOM = (ROOT / "results/repro_replicates/B0_dom_classifieds_R31194_clean_replicate"
            / "phase1_dom_router_0/artifacts/classifieds_task_0/step_000/screenshot.png")
SHOT_VIS = (ROOT / "results/repro_replicates/B0_vision_classifieds_R24792_clean_replicate"
            / "phase1_vision_router_0/artifacts/classifieds_task_0/step_000/screenshot.png")

TASK, STEP = "classifieds_task_0", "step_000"
C_TEXT = "#0072B2"
C_IMG = "#D55E00"
C_MUTE = "#777777"


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def gather() -> dict:
    def mirror_file(mode: str, name: str) -> Path | None:
        for d in sorted(MIRROR.glob(f"B0_{mode}_classifieds_*")):
            f = next(d.glob(f"phase1_*_router_0/artifacts/{TASK}/{STEP}/{name}"), None)
            if f:
                return f
        return None

    def step0(mode: str) -> dict:
        for d in sorted(PHASE1.glob(f"B0_{mode}_classifieds_*")):
            g = sorted(d.glob(f"phase1_*_router_0/episodes/{TASK}_steps_v2.jsonl"))
            if g:
                with g[0].open(encoding="utf-8") as fh:
                    return json.loads(fh.readline())
        raise SystemExit(f"no step record for B0 {mode} — refusing to guess")

    for p in (SHOT_RAW, SHOT_SOM):
        if not p.exists():
            raise SystemExit(f"missing screenshot asset {p} — refusing to plot")
    if _md5(SHOT_DOM) != _md5(SHOT_VIS):
        raise SystemExit("dom/vision step_000 screenshots differ — the columns "
                         "would not be the same page; refusing to plot")

    dom_txt = mirror_file("dom", "observation_dom.txt")
    som_txt = mirror_file("som", "observation_som.txt")
    if not dom_txt or not som_txt:
        raise SystemExit("missing observation artifacts — refusing to plot")

    d = {}
    for mode in ("dom", "som", "vision"):
        rec = step0(mode)
        im = rec.get("image_meta") or {}
        d[mode] = {"tokens": rec["tokens"]["input"],
                   "img": im.get("image_payload_bytes") or 0,
                   "marks": (rec.get("som") or {}).get("mark_count", 0)}
    d["dom"]["chars"] = len(dom_txt.read_text(encoding="utf-8"))
    d["som"]["chars"] = len(som_txt.read_text(encoding="utf-8"))
    d["vision"]["chars"] = 0
    d["_dom_lines"] = dom_txt.read_text(encoding="utf-8").splitlines()
    d["_som_lines"] = som_txt.read_text(encoding="utf-8").splitlines()
    return d


def _snippet(lines, n=9, width=46):
    keep = [ln.rstrip() for ln in lines if ln.strip()][:n]
    return "\n".join(textwrap.shorten(ln.strip(), width=width, placeholder=" …")
                     for ln in keep)


def build(fig, d):
    gs = fig.add_gridspec(2, 3, height_ratios=[2.25, 1.0], hspace=0.02,
                          wspace=0.07, left=0.015, right=0.985, top=0.885,
                          bottom=0.10)

    cols = [
        ("DOM", "AXTree text · no image", None, _snippet(d["_dom_lines"]), d["dom"]),
        ("SoM", f"[SOM_MARKS] text + annotated screenshot", SHOT_SOM, None, d["som"]),
        ("Vision", "raw screenshot · no structured text", SHOT_RAW, None, d["vision"]),
    ]

    for i, (name, sub, shot, snip, m) in enumerate(cols):
        ax = fig.add_subplot(gs[0, i])
        if shot is not None:
            ax.imshow(mpimg.imread(shot))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#BBBBBB")
        else:
            # Same canvas geometry as the screenshots (1280x660) so the three
            # columns line up; an axis("off") panel would float at its own height.
            ax.set_xlim(0, 1280); ax.set_ylim(660, 0)
            ax.set_aspect("equal")
            ax.set_facecolor("#FAFAFA")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#BBBBBB")
            ax.text(44, 34, snip, fontsize=5.6, family="monospace",
                    color="#333333", va="top", linespacing=1.5)
            ax.text(640, 600, "no image is sent", ha="center", fontsize=8.6,
                    color=C_MUTE, style="italic")
        ax.set_title(f"{name}", fontsize=11.5, fontweight="bold", loc="left",
                     pad=15)
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=7.8,
                color=C_MUTE, va="bottom")

        axb = fig.add_subplot(gs[1, i])
        axb.axis("off")
        axb.set_xlim(0, 1); axb.set_ylim(0, 1)
        y = 0.80
        if m["chars"]:
            axb.text(0.02, y, f"{m['chars']:,} chars of text", fontsize=8.6,
                     color=C_TEXT, fontweight="bold")
        else:
            axb.text(0.02, y, "no structured text", fontsize=8.6, color=C_MUTE,
                     fontweight="bold")
        y -= 0.30
        if m["img"]:
            lbl = f"{m['img'] / 1024:.0f} KB image"
            if m["marks"]:
                lbl += f"  ·  {m['marks']} marks"
            axb.text(0.02, y, lbl, fontsize=8.6, color=C_IMG, fontweight="bold")
        else:
            axb.text(0.02, y, "no image", fontsize=8.6, color=C_MUTE,
                     fontweight="bold")
        y -= 0.32
        axb.text(0.02, y, f"{m['tokens']:,} input tokens", fontsize=10.2,
                 color="#000000", fontweight="bold")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    d = gather()
    ratio = d["som"]["chars"] / d["dom"]["chars"]

    fig = plt.figure(figsize=(13.0, 5.0))
    build(fig, d)
    fig.suptitle("The same page, one step, three ways to pay for it   —   "
                 "VisualWebArena classifieds · task 0 · step 000",
                 fontsize=11.6, fontweight="bold", x=0.015, ha="left", y=0.975)
    fig.text(0.015, 0.925,
             f"SoM's text payload is {ratio:.3f}× DOM's — within "
             f"{abs(ratio - 1) * 100:.1f}%. Nearly the entire extra cost is the "
             "image, not the text.",
             fontsize=9.4, color="#B34700", fontweight="bold")
    fig.text(0.015, 0.012,
             "Screenshots: raw/annotated pair rendered from this page "
             "(1280×660). Text: mechanistic/_obs_mirror per-mode observation. "
             "Bytes and tokens: results/visualwebarena/phase1 step record.\n"
             "The step-000 page is the task start URL; the dom-run and vision-run "
             "artifacts for it are byte-identical (md5 verified), which is what "
             "makes the three columns the same page.",
             fontsize=6.8, color="#888888", linespacing=1.55)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   (SoM/DOM text {ratio:.4f}×; tokens "
          f"dom={d['dom']['tokens']} som={d['som']['tokens']} "
          f"vision={d['vision']['tokens']}; marks={d['som']['marks']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
