#!/usr/bin/env python3
"""Render the poster's figures and parse the demo strip's numbers.

``_style.py`` opens with the rule this script obeys: *author at the printed
width*. A figure drawn 5.12in wide for the A4 text block and then dropped into a
poster panel is barely scaled, so its 8.5pt labels land on the A1 sheet at about
10pt — legible at arm's length, not from across a room.

Outputs (all under ``figures/``):

* ``poster_dominance_plane.png`` — thesis F13, re-authored at the inner width of
  the poster's figure box. Nothing in ``final_dissertation/`` is touched: this
  imports the thesis script's own ``build()`` and overrides only the shared
  style constants.

* ``demo_strip.json`` + ``thumb_<task>.png`` — the three tasks the laptop demo
  replays, with each mode's outcome, step count and billed cost **parsed from
  the episode summaries**, never typed in. A task qualifies only if the three
  modes' outcomes differ AND every mode's outcome is identical on the
  independent replicate run; the script asserts the second condition, so a task
  that flips on rerun cannot reach the sheet. The thumbnail is the final page of
  the winning mode's replicate run (the only runs whose artifacts survive).

Usage::

    .venv/bin/python3 deliverables/showcase/poster_figures.py
    .venv/bin/python3 deliverables/showcase/build_poster.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
THESIS_FIGS = REPO / "scripts" / "analysis" / "figures" / "thesis"
OUTDIR = Path(__file__).resolve().parent / "figures"
PHASE1 = REPO / "results" / "visualwebarena" / "phase1"
REPL = REPO / "results" / "repro_replicates"
CFG = REPO / "external" / "visualwebarena" / "config_files" / "vwa" / "test_classifieds"

sys.path.insert(0, str(THESIS_FIGS))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import _style as S  # noqa: E402

# Inner width of the template figure box in the poster's right-hand block.
COL_W_IN = 362 / 25.4

S.FS_TICK = 18.0
S.FS_LABEL = 22.0
S.FS_VALUE = 20.0
S.FS_PANEL = 24.0

# B0 x classifieds. "A" is the canonical phase-1 run, "B" the independent
# replicate. Only the two clean replicates kept their per-step artifacts.
RUNS = {
    ("look", "A"): PHASE1 / "B0_vision_classifieds_20260526_141916_610351680_689390_R32024",
    ("look", "B"): REPL / "B0_vision_classifieds_R24792_clean_replicate",
    ("read", "A"): PHASE1 / "B0_dom_classifieds_20260525_194618_553890342_530647_R21557",
    ("read", "B"): REPL / "B0_dom_classifieds_R31194_clean_replicate",
    ("both", "A"): PHASE1 / "B0_som_classifieds_20260526_041601_863239369_602235_R5313",
    ("both", "B"): PHASE1 / "B0_som_classifieds_20260803_084743_413015398_3677519_R30696",
}
# (task, mode whose replicate run supplies the thumbnail, step index).
# 17 is solved only by BOTH, whose artifacts were cleaned; its thumbnail is the
# last page of the READ replicate: a $900 bike whose handlebars are not red —
# the text-only agent matched the price and could not check the colour.
DEMO = [(130, "look", -1), (76, "read", -1), (17, "read", -1)]


def save(fig, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}.png"
    fig.savefig(path, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


def dominance_plane() -> None:
    import fig_f13_dominance_plane as f

    rows, _proto = f.load(f.SRC)
    S.apply()
    fig, ax = plt.subplots(figsize=(COL_W_IN, 6.8))
    f.build(ax, rows)

    x0, _ = ax.get_xlim()
    for artist in list(ax.texts):
        text = artist.get_text()
        if "Pareto" in text:
            artist.set_text("CHEAPER  AND  NO WORSE")
            artist.set_position((x0 / 2, 4.1))
            artist.set_ha("center")
            artist.set_va("center")
            artist.set_rotation(90)
            artist.set_fontsize(S.FS_LABEL)
        elif "always-cheapest" in text:
            artist.remove()
        elif text.startswith("WA-red·B1"):
            artist.set_position((9, -26))

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles,
              ["learned choice", "learned, scored on its own training tasks",
               "perfect hindsight"],
              loc="lower right", frameon=False, handletextpad=0.4,
              borderpad=0.2, fontsize=19)
    ax.set_xlabel("cost relative to always-cheapest   "
                  "(log$_2$ ratio: 0 = same, 1 = double)", fontsize=S.FS_LABEL)
    ax.set_ylabel("more tasks solved than\nalways-cheapest, per 100",
                  fontsize=S.FS_LABEL)
    save(fig, "poster_dominance_plane")


def _cond(run: Path) -> Path:
    hits = glob.glob(str(run / "phase1_*_router_0"))
    assert len(hits) == 1, f"expected one condition dir under {run}, got {hits}"
    return Path(hits[0])


def _summary(run: Path, task: int) -> dict:
    hits = glob.glob(str(_cond(run) / "episodes" / f"classifieds_task_{task}_summary*.json"))
    assert len(hits) == 1, f"expected one summary for task {task} under {run}, got {hits}"
    return json.loads(Path(hits[0]).read_text())


def demo_strip() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for task, thumb_mode, step in DEMO:
        intent = json.loads((CFG / f"{task}.json").read_text())["intent"]
        modes = {}
        for mode in ("look", "read", "both"):
            a, b = _summary(RUNS[(mode, "A")], task), _summary(RUNS[(mode, "B")], task)
            assert bool(a["success"]) == bool(b["success"]), (
                f"task {task} {mode}: outcome flips between runs — not demo material")
            cost = a.get("total_billed_cost_usd")
            assert cost is not None, f"task {task} {mode}: no total_billed_cost_usd"
            modes[mode] = {"success": bool(a["success"]),
                           "steps": int(a["agent_action_step_count"]),
                           "cost_usd": float(cost),
                           "steps_rerun": int(b["agent_action_step_count"])}
        outcomes = {m["success"] for m in modes.values()}
        assert len(outcomes) == 2, f"task {task}: all modes agree — nothing to show"
        steps = sorted(glob.glob(str(_cond(RUNS[(thumb_mode, "B")]) / "artifacts"
                                     / f"classifieds_task_{task}" / "step_*")))
        assert steps, f"task {task}: no artifacts in the {thumb_mode} replicate"
        src = Path(steps[step]) / "screenshot.png"
        with Image.open(src) as im:
            assert im.size == (1280, 720), f"unexpected screenshot size {im.size}"
            im.crop((0, 0, 1280, 600)).save(OUTDIR / f"thumb_{task}.png")
        out[str(task)] = {"intent": intent, "modes": modes,
                          "thumb": f"thumb_{task}.png",
                          "thumb_from": f"{thumb_mode} replicate, {Path(steps[step]).name}"}
        print(f"  task {task:3d}: " + "  ".join(
            f"{m}={'✓' if v['success'] else '✗'} {v['steps']}st ${v['cost_usd']:.3f}"
            for m, v in modes.items()))
    (OUTDIR / "demo_strip.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  wrote {(OUTDIR / 'demo_strip.json').relative_to(REPO)}")


def eyes() -> None:
    """What each way of seeing actually sends, on one real page — the same assets
    thesis F1 uses (same page, same listings, dom/vision step-000 screenshots
    md5-identical). LOOK = raw screenshot, BOTH = the marked screenshot, READ =
    the first lines of the element list the agent is given."""
    import fig_f1_motivating_example as f1
    for src, name in ((f1.SHOT_RAW, "eye_look"), (f1.SHOT_SOM, "eye_both")):
        assert src.exists(), f"missing F1 asset {src}"
        with Image.open(src) as im:
            w, h = im.size
            im.crop((0, 0, w, int(w * 0.56))).save(OUTDIR / f"{name}.png")
    dom = None
    for d in sorted(f1.MIRROR.glob("B0_dom_classifieds_*")):
        hit = next(d.glob("phase1_*_router_0/artifacts/classifieds_task_0/step_000/observation_dom.txt"), None)
        if hit:
            dom = hit
            break
    assert dom is not None, "no mirrored DOM observation for classifieds task 0"
    lines = [ln.rstrip() for ln in dom.read_text(encoding="utf-8").splitlines() if ln.strip()]
    keep = [ln for ln in lines if ln.lstrip().startswith("[")][:7]
    (OUTDIR / "eye_read.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"  wrote figures/eye_look.png, eye_both.png, eye_read.txt ({len(keep)} lines from {dom.name})")


LABEL_SUPPLY_MD = REPO / "docs" / "analysis" / "cross_sites" / "router_label_supply_diagnosis.md"
LEARN_JSON = REPO / "docs" / "analysis" / "cross_sites" / "router_triage_learnability_with_wa.json"
LEFT_W_IN = 173 / 25.4   # inner width of the left column's figure box


def label_supply() -> None:
    """Fig 3: usable "which view" training examples against the best single
    view's success rate, one point per VisualWebArena setting. Both numbers are
    parsed from the analysis artefacts (the first table of the label-supply
    diagnosis, and baseline_policy.sr_pct of the 8-cell learnability run)."""
    import re
    rows = {}
    trainable = {}
    text = LABEL_SUPPLY_MD.read_text(encoding="utf-8")
    for m in re.finditer(r"^\| (B\d)_(classifieds|reddit) \| (\d+) \| \*\*(\d+)\*\* \| ([\d.]+)% \| (\d)/6 \|", text, re.M):
        rows[(m.group(2), m.group(1))] = int(m.group(4))
    for m in re.finditer(r"^\| (B\d)_(classifieds|reddit) \| (\d+) \| \d \| [^|]+ \| (\*\*no\*\*|yes) \|", text, re.M):
        trainable[(m.group(2), m.group(1))] = (m.group(4) == "yes")
    assert len(rows) == 6 and len(trainable) == 6, (rows, trainable)
    cells = json.loads(LEARN_JSON.read_text())["cells"]
    sr = {(c["site"], c["baseline_model"]): c["baseline_policy"]["sr_pct"] for c in cells}
    S.apply()
    fig, ax = plt.subplots(figsize=(LEFT_W_IN, 3.7))
    for key, n in rows.items():
        x = sr[key]
        filled = trainable[key]
        ax.scatter([x], [n], s=260, marker="o",
                   facecolors=S.C_INK if filled else "white", edgecolors=S.C_INK,
                   linewidths=2.2, zorder=3)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 110)
    ax.set_xlabel("tasks the best single view solves  (%)", fontsize=16)
    ax.set_ylabel("usable training examples\nfor “which view”", fontsize=16)
    ax.tick_params(labelsize=14)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker="o", ls="", ms=13, mfc=S.C_INK, mec=S.C_INK),
               Line2D([], [], marker="o", ls="", ms=13, mfc="white", mec=S.C_INK, mew=2)],
              ["enough to train a classifier", "not enough"], loc="upper left",
              frameon=False, fontsize=15, handletextpad=0.3)
    save(fig, "poster_label_supply")
    print("    " + "  ".join(f"{k[0][:3]}·{k[1]}: sr={sr[k]:.1f}% n={n} {'✓' if trainable[k] else '✗'}"
                             for k, n in rows.items()))


if __name__ == "__main__":
    print("rendering poster-scale figures")
    dominance_plane()
    demo_strip()
    eyes()
    label_supply()
