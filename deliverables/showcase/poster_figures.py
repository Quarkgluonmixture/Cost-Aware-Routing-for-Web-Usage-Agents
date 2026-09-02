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
    fig, ax = plt.subplots(figsize=(COL_W_IN, 7.5))
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
              ["learned router", "learned (in-sample)", "hindsight oracle"],
              loc="lower right", frameon=False, handletextpad=0.4,
              borderpad=0.2, fontsize=S.FS_LABEL)
    ax.set_xlabel("cost, relative to always using the cheapest mode   "
                  "[$\\log_2$ ratio]", fontsize=S.FS_LABEL)
    ax.set_ylabel("success over always-cheapest  [pp]", fontsize=S.FS_LABEL)
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


if __name__ == "__main__":
    print("rendering poster-scale figures")
    dominance_plane()
    demo_strip()
