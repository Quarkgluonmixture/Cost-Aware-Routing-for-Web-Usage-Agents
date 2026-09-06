"""Build the three-lane replay demo for the 16 Sep showcase board.

WHAT IT MAKES
-------------
`data/task_<N>.json` + `frames/<N>/<LANE>/*.png` for a static page that replays the
SAME task in three synchronised lanes — LOOK (vision), READ (dom), BOTH (SoM) —
stepping through the agent's own screenshots with its click drawn on top, its
one-line thought under each frame, and a running dollar counter per lane.

WHY THESE THREE TASKS
---------------------
They are the three shapes of the finding, one each, and their outcome is the same on
the canonical run and on its replicate (SHOWCASE_PREP §8's rule — a task whose
three-way outcome moves between runs would be showing noise, not a finding):

    task 130   LOOK ✓ 2 steps   READ ✗ 9    BOTH ✓ 3    seeing it is enough
    task  76   LOOK ✗ 26        READ ✓ 12   BOTH ✗ 7    reading it is enough
    task  17   LOOK ✗ 9         READ ✗ 8    BOTH ✓ 6    neither alone is enough

EVERYTHING HERE IS A RECORDING. No site is contacted. Each lane is ONE recorded run
and is labelled as such on the page; frames are the agent's own screenshots, and the
overlay is drawn from the same step record that produced the action. Re-recording a
run to obtain artifacts would be fine and would be labelled a run; hand-assembling a
trajectory would not, and nothing here does that.

THE COORDINATE TRAP (B-1860)
----------------------------
The three lanes do NOT share a coordinate space, and drawing them as if they did is
the single easiest way to make this demo lie:

  * LOOK  emits `action.coordinate` in Qwen's **0-1000 normalised** space
          (`coordinate_type == "qwen_0_1000"`) -> must be scaled by W/1000, H/1000
  * READ  emits `element_bbox` in **viewport pixels** already -> drawn as is
  * BOTH  emits `element_bbox` too, plus the `element_id` that the SoM overlay
          numbered -> we show the numbered SoM frame, so the mark is already in
          the image and the box is only a highlight

`coordinate_type` is asserted rather than assumed: a future run that changes the
convention should fail this build, not silently draw the click in the wrong place.

Usage:  .venv/bin/python3 deliverables/showcase/demo/build_demo_data.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from p79.experiment.io_utils import read_jsonl_dedup  # noqa: E402

HERE = Path(__file__).resolve().parent
REPL = REPO / "results" / "repro_replicates"
PH1 = REPO / "results" / "visualwebarena" / "phase1"

# Lane -> condition dir. LOOK and READ come from the registered clean replicates (the
# same runs the poster's figures are built from); BOTH comes from the SoM arm's
# replicate, chosen so all three lanes sit on the same side of their replicate pair.
LANES = {
    "LOOK": REPL / "B0_vision_classifieds_R24792_clean_replicate" / "phase1_vision_router_0",
    "READ": REPL / "B0_dom_classifieds_R31194_clean_replicate" / "phase1_dom_router_0",
    "BOTH": PH1 / "B0_som_classifieds_20260803_084743_413015398_3677519_R30696" / "phase1_som_router_0",
}
TASKS = [130, 76, 17]

# One line per task, saying WHY that task falls the way it does. Each is written
# against the task's own `intent` (printed by the build) and its measured outcome —
# not against the story we would like to tell. If a rerun moves an outcome, the
# caption is wrong and must be rewritten, which is why the intent is shipped to the
# page next to it: a visitor can check the claim against the task in front of them.
CAPTION = {
    130: "The answer is in the picture. The text tree never says “sunset”.",
    76: "The answer is in the form. LOOK wanders 26 steps and never edits it.",
    17: "Red handlebars must be seen; $900–950 must be filtered. Neither alone is enough.",
}


class BuildError(RuntimeError):
    """Fail loud. A demo that silently drops a lane still looks like a demo."""


def _frame(cond: Path, task: int, step: int, lane: str) -> Path | None:
    """The image a visitor should see for this step.

    BOTH shows the SoM-annotated frame — the numbered marks ARE the representation
    under test, so showing the clean screenshot would misrepresent what the agent saw.
    """
    d = cond / "artifacts" / f"classifieds_task_{task}"
    if lane == "BOTH":
        som = d / "som" / f"step_{step:03d}_som.png"
        if som.exists():
            return som
    shot = d / f"step_{step:03d}" / "screenshot.png"
    return shot if shot.exists() else None


def _mark(rec: dict, lane: str, w: int, h: int) -> dict | None:
    """Overlay geometry in VIEWPORT PIXELS, or None when the step has no target."""
    act = rec.get("action") or {}
    if lane == "LOOK":
        c = act.get("coordinate")
        if not c:
            return None
        ctype = act.get("coordinate_type")
        if ctype != "qwen_0_1000":
            raise BuildError(
                f"LOOK step has coordinate_type={ctype!r}, expected 'qwen_0_1000'. "
                f"The 0-1000 -> pixel scaling below would put the click in the wrong "
                f"place (B-1860). Fix the scaling deliberately rather than relaxing "
                f"this check.")
        return {"kind": "point", "x": c[0] * w / 1000.0, "y": c[1] * h / 1000.0}
    bbox = rec.get("element_bbox")
    if not bbox:
        return None
    x, y, bw, bh = bbox
    return {"kind": "box", "x": x, "y": y, "w": bw, "h": bh,
            "id": act.get("element_id")}


def _intent(task: int) -> str:
    """The task as the agent was given it — shipped to the page so a visitor can
    check the caption against the actual instruction rather than taking it on faith."""
    for cond in LANES.values():
        p = cond.parent / "task_configs" / f"classifieds_task_{task}.json"
        if p.exists():
            return json.loads(p.read_text()).get("intent", "")
    raise BuildError(f"task {task}: no task_config found under any lane")


def build_task(task: int) -> dict:
    out = {"task": task, "caption": CAPTION[task],
           "intent": _intent(task), "lanes": {}}
    for lane, cond in LANES.items():
        summ_p = cond / "episodes" / f"classifieds_task_{task}_summary_v2.json"
        steps_p = cond / "episodes" / f"classifieds_task_{task}_steps_v2.jsonl"
        if not summ_p.exists() or not steps_p.exists():
            raise BuildError(f"{lane} task {task}: missing {summ_p if not summ_p.exists() else steps_p}")
        summ = json.loads(summ_p.read_text())
        recs = read_jsonl_dedup(str(steps_p))

        dest = HERE / "frames" / str(task) / lane
        dest.mkdir(parents=True, exist_ok=True)
        frames, cum = [], 0.0
        for i, rec in enumerate(recs):
            src = _frame(cond, task, i, lane)
            if src is None:
                continue
            shutil.copyfile(src, dest / f"{i:03d}.png")
            cost = (rec.get("cost_usd") or {}).get("model") or 0.0
            cum += cost
            act = rec.get("action") or {}
            frames.append({
                "step": i,
                "img": f"frames/{task}/{lane}/{i:03d}.png",
                "thought": (act.get("thought") or "").strip(),
                "action": rec.get("action_type"),
                "ok": bool(rec.get("action_success")),
                "mark": _mark(rec, lane, 1280, 720),
                "cost_cum": round(cum, 4),
            })
        if not frames:
            raise BuildError(f"{lane} task {task}: no frames — artifacts missing?")
        out["lanes"][lane] = {
            "success": bool(summ["success"]),
            "steps": summ["steps"],
            "cost": round(summ.get("total_model_cost_usd") or cum, 4),
            "run": cond.parent.name,
            "frames": frames,
        }
    return out


def main() -> int:
    (HERE / "data").mkdir(exist_ok=True)
    index = []
    for t in TASKS:
        d = build_task(t)
        (HERE / "data" / f"task_{t}.json").write_text(json.dumps(d, indent=1))
        row = {"task": t, "caption": d["caption"],
               "outcome": {k: v["success"] for k, v in d["lanes"].items()}}
        index.append(row)
        marks = {k: sum(1 for f in v["frames"] if f["mark"]) for k, v in d["lanes"].items()}
        print(f"  intent: {d['intent']}")
        print(f"task {t:>3}: " + "  ".join(
            f"{k} {'PASS' if v['success'] else 'fail'} {len(v['frames'])}f/{marks[k]}mk ${v['cost']:.3f}"
            for k, v in d["lanes"].items()))
    (HERE / "data" / "index.json").write_text(json.dumps(index, indent=1))

    # data.js — the same payload as a plain global, because the page must open by
    # double-click at a showcase board. `fetch()` on a file:// URL is blocked by CORS,
    # so a JSON-reading page would be silently empty exactly when no one can debug it;
    # a <script src> is not subject to that rule. Images are unaffected either way.
    payload = {str(t): json.loads((HERE / "data" / f"task_{t}.json").read_text())
               for t in TASKS}
    (HERE / "data.js").write_text(
        "// generated by build_demo_data.py — do not edit\n"
        "window.DEMO_TASKS = " + json.dumps([str(t) for t in TASKS]) + ";\n"
        "window.DEMO = " + json.dumps(payload, indent=1) + ";\n")

    total_png = sum(1 for _ in (HERE / "frames").rglob("*.png"))
    mb = sum(f.stat().st_size for f in (HERE / "frames").rglob("*.png")) / 1e6
    print(f"\n✓ {len(TASKS)} tasks -> {HERE/'data'}")
    print(f"✓ data.js ({(HERE/'data.js').stat().st_size/1e3:.0f} kB) + "
          f"{total_png} frames ({mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
