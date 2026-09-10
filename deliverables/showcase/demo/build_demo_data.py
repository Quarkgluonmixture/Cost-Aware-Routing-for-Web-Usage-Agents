"""Build the three-lane replay demo for the 16 Sep showcase board.

WHAT IT MAKES
-------------
`data/task_<N>.json` + `frames/<N>/<LANE>/*.png` for a static page that replays the
SAME task in three synchronised lanes — LOOK (vision), READ (dom), BOTH (SoM) —
stepping through the agent's own screenshots with its click drawn on top, its
one-line thought under each frame, and three running meters per lane: dollars,
recorded wall-clock time, and an estimated CO2e range.

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

THE LEARNED CHOICE (the arrow on the page)
------------------------------------------
Each task also carries the view that the project's learned router picked for it,
read from the fold-held-out replay (`router_offline_replay.json`): the router that
made the pick was trained WITHOUT this task. The build refuses an in-sample
prediction, because the page presents the pick as something decided in advance.
Nothing is chosen to flatter the router — on these three tasks it is wrong once,
right once, and once picks a fourth view that is not one of the lanes; the page
says so, including that the fourth view's own two recorded runs disagree.

THE CO2e ROW IS AN ESTIMATE, NOT A MEASUREMENT
----------------------------------------------
The model is served remotely (AWS eu-west-2), so there is no local energy draw to
measure and every run records `energy.source = disabled`. The row is therefore
computed, not read: input and output tokens × a published energy-per-token RANGE ×
datacentre PUE range × UK grid intensity. It is shown as a range with "≈" on the page
and its constants and sources live in `CARBON` below. Token counts are a workload
indicator, not an energy measurement (Fernandez et al., ACL 2025) — which is exactly
why the page never shows a single number here.

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

# ---- the learned choice ---------------------------------------------------------
ROUTER_REPLAY = REPO / "results/phantom_paper/l1_router_offline_20260715/router_offline_replay.json"
ROUTER_CELL = "B0_classifieds"
MODE_LANE = {"vision": "LOOK", "dom": "READ", "som": "BOTH"}
# A pick outside the three lanes is reported by its own recorded runs — the canonical
# run the router was scored against and its registered replicate — never by one run
# alone, because one run of a view is exactly what rerun noise can flip.
OTHER_VIEW = {
    "phantom_prompt": {
        "name": "text tree, read with the marked-screenshot instructions",
        "runs": [
            PH1 / "B0_phantom_prompt_classifieds_20260528_040546_107246795_987141_R14655" / "phase1_phantom_prompt_router_0",
            PH1 / "B0_phantom_prompt_classifieds_20260817_184335_813828144_2037698_R12207" / "phase1_phantom_prompt_router_0",
        ],
    },
}

# ---- CO2e estimate ----------------------------------------------------------------
# A range at every stage; the page shows [lo, hi] and never a point value. Sources
# were gathered 2026-09-10 (arXiv ids confirmed through the arXiv API); the full
# write-up with the "what makes this >3x wrong" list is in README "The CO2e row".
CARBON: dict | None = {
    # J per token, GPU only.
    # OUTPUT: measured on THIS model — ML.ENERGY Leaderboard v3 (companion to Chung et
    #   al. 2025, arXiv 2505.06371): Qwen3-VL-235B-A22B BF16 on 8xH100, 7.26 J at batch 8
    #   down to ~1.1 J at batch 192-256. The range is the batching we cannot see.
    # INPUT: no prefill measurement exists for this model. Scaled from measured dense
    #   prefill (Chung et al. 2025, App. B.3: Llama-3.1-8B 0.017 J, 70B 0.18 J per input
    #   token) to ~22B active; the high end doubles 70B for MoE overhead and the vision
    #   encoder — a judgement, not a measurement.
    "j_per_input_token": [0.05, 0.4],
    "j_per_output_token": [1.0, 8.0],
    # GPU -> facility: host + idle factor 1.0-1.6 (Google 2025, arXiv 2508.15734, Table 1:
    # 0.24 Wh comprehensive vs 0.14 Wh accelerator-only) x AWS eu-west-2 PUE 1.23 (AWS
    # Region PUE table, 2025).
    "facility_factor": [1.23, 1.97],
    # The proxy is served from AWS eu-west-2 (London): UK DESNZ 2026 conversion factor,
    # 131 g/kWh at generation, 144 g/kWh with transmission and distribution losses.
    "grid_g_per_kwh": [131.0, 144.0],
    "region": "AWS eu-west-2 (London)",
    "boundary": "operational only (no embodied emissions); location-based grid factor",
}


class BuildError(RuntimeError):
    """Fail loud. A demo that silently drops a lane still looks like a demo."""


def _co2_g(tok_in: float, tok_out: float) -> tuple[float, float]:
    """Estimated operational CO2e in grams for this many tokens, as (low, high)."""
    if CARBON is None:
        raise BuildError("CARBON constants are not set — the CO2e row would be unsourced")
    (ei_lo, ei_hi), (eo_lo, eo_hi) = CARBON["j_per_input_token"], CARBON["j_per_output_token"]
    p_lo, p_hi = CARBON["facility_factor"]
    g_lo, g_hi = CARBON["grid_g_per_kwh"]
    j_lo = tok_in * ei_lo + tok_out * eo_lo
    j_hi = tok_in * ei_hi + tok_out * eo_hi
    return (j_lo * p_lo / 3.6e6 * g_lo, j_hi * p_hi / 3.6e6 * g_hi)


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


def _previous_build() -> dict:
    """The payload of the data.js this build is about to replace, or {}."""
    p = HERE / "data.js"
    if not p.exists():
        return {}
    import re
    m = re.search(r"window\.DEMO = (\{.*\});", p.read_text(), re.S)
    return json.loads(m.group(1)) if m else {}


_PREV = _previous_build()


def _committed_frame_ok(task: int, lane: str, run: str, dst: Path) -> bool:
    """Reuse a committed frame when its source artifact is no longer on this machine.

    `frames/` is committed precisely because its inputs live under gitignored
    `results/` and do not travel (the SoM run's artifacts have also vanished from
    DGX twice, §501.6). A committed frame is only trusted if the build it came from
    names the SAME run for this lane and listed this exact frame — otherwise a frame
    from another run could be relabelled silently.
    """
    if not dst.exists():
        return False
    prev = (_PREV.get(str(task)) or {}).get("lanes", {}).get(lane)
    rel = dst.relative_to(HERE).as_posix()
    if not prev or prev.get("run") != run or rel not in {f["img"] for f in prev["frames"]}:
        raise BuildError(
            f"{lane} task {task}: source artifact missing and the committed frame {rel} "
            f"cannot be tied to run {run} by the previous build — pull the artifacts "
            f"(README 'Rebuilding') instead of reusing it")
    return True


def _intent(task: int) -> str:
    """The task as the agent was given it — shipped to the page so a visitor can
    check the caption against the actual instruction rather than taking it on faith."""
    for cond in LANES.values():
        p = cond.parent / "task_configs" / f"classifieds_task_{task}.json"
        if p.exists():
            return json.loads(p.read_text()).get("intent", "")
    raise BuildError(f"task {task}: no task_config found under any lane")


def _pick(task: int, lanes: dict) -> dict:
    """The view the learned router picked for this task, held out from its training."""
    rep = json.loads(ROUTER_REPLAY.read_text())
    recs = [r for r in rep["cells"][ROUTER_CELL]["task_records"] if r["task_id"] == task]
    if len(recs) != 1:
        raise BuildError(f"task {task}: {len(recs)} router records in {ROUTER_REPLAY.name}")
    r = recs[0]
    if r["prediction_status"] != "oof":
        raise BuildError(
            f"task {task}: router prediction is {r['prediction_status']!r}, not 'oof'. "
            f"The page presents the pick as made without seeing this task; an "
            f"in-sample pick would be a result the router had already been shown.")
    if r["signal_strength_fallback_fired"]:
        raise BuildError(f"task {task}: the router fell back to its default view — "
                         f"the page has no wording for that yet; add it before shipping")
    mode = r["selected_mode"]
    out = {"mode": mode, "source": ROUTER_REPLAY.relative_to(REPO).as_posix()}
    if mode in MODE_LANE:
        lane = MODE_LANE[mode]
        out.update(lane=lane, success=lanes[lane]["success"])
        return out
    if mode not in OTHER_VIEW:
        raise BuildError(f"task {task}: router picked {mode!r}, which is neither a lane "
                         f"nor described in OTHER_VIEW")
    ov = OTHER_VIEW[mode]
    runs = []
    for d in ov["runs"]:
        s = json.loads((d / "episodes" / f"classifieds_task_{task}_summary_v2.json").read_text())
        runs.append({"run": d.parent.name, "success": bool(s["success"]), "steps": s["steps"]})
    out.update(lane=None, name=ov["name"], runs=runs)
    return out


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
        frames, cum, sec, t_in, t_out = [], 0.0, 0.0, 0, 0
        for i, rec in enumerate(recs):
            # Meters accumulate over EVERY step record, framed or not — a step whose
            # screenshot is missing still cost money and time.
            cum += (rec.get("cost_usd") or {}).get("model") or 0.0
            sec += ((rec.get("latency_ms") or {}).get("total") or 0.0) / 1000.0
            tok = rec.get("tokens") or {}
            t_in += tok.get("input") or 0
            t_out += tok.get("output") or 0
            src = _frame(cond, task, i, lane)
            dst = dest / f"{i:03d}.png"
            if src is not None:
                shutil.copyfile(src, dst)
            elif not _committed_frame_ok(task, lane, cond.parent.name, dst):
                continue
            act = rec.get("action") or {}
            lo, hi = _co2_g(t_in, t_out)
            frames.append({
                "step": i,
                "img": f"frames/{task}/{lane}/{i:03d}.png",
                "thought": (act.get("thought") or "").strip(),
                "action": rec.get("action_type"),
                "ok": bool(rec.get("action_success")),
                "mark": _mark(rec, lane, 1280, 720),
                "cost_cum": round(cum, 4),
                "sec_cum": round(sec, 1),
                "tok_cum": t_in + t_out,
                "co2_cum": [round(lo, 4), round(hi, 4)],
            })
        if not frames:
            raise BuildError(f"{lane} task {task}: no frames — artifacts missing?")
        # The running meters must land on the episode's own totals, or the page is
        # showing a different run than the one its header names.
        checks = [("cost", cum, summ.get("total_model_cost_usd"), 1e-4),
                  ("seconds", sec, (summ.get("total_latency_ms") or 0) / 1000.0, 0.5),
                  ("tokens", t_in + t_out, summ.get("total_tokens"), 0)]
        for name, got, want, tol in checks:
            if want is None or abs(got - want) > tol:
                raise BuildError(f"{lane} task {task}: summed {name} {got} != summary {want}")
        lo, hi = _co2_g(t_in, t_out)
        out["lanes"][lane] = {
            "success": bool(summ["success"]),
            "steps": summ["steps"],
            "cost": round(cum, 4),
            "seconds": round(sec, 1),
            "tokens": t_in + t_out,
            "co2": [round(lo, 4), round(hi, 4)],
            "run": cond.parent.name,
            "frames": frames,
        }
    out["pick"] = _pick(task, out["lanes"])
    return out


def main() -> int:
    (HERE / "data").mkdir(exist_ok=True)
    index = []
    for t in TASKS:
        d = build_task(t)
        (HERE / "data" / f"task_{t}.json").write_text(json.dumps(d, indent=1))
        row = {"task": t, "caption": d["caption"],
               "outcome": {k: v["success"] for k, v in d["lanes"].items()},
               "pick": d["pick"]["mode"]}
        index.append(row)
        marks = {k: sum(1 for f in v["frames"] if f["mark"]) for k, v in d["lanes"].items()}
        print(f"  intent: {d['intent']}")
        print(f"task {t:>3}: " + "  ".join(
            f"{k} {'PASS' if v['success'] else 'fail'} {len(v['frames'])}f/{marks[k]}mk "
            f"${v['cost']:.3f} {v['seconds']:.0f}s {v['co2'][0]:.3g}-{v['co2'][1]:.3g}g"
            for k, v in d["lanes"].items()))
        p = d["pick"]
        print(f"          learned choice -> {p['mode']}"
              + (f" = {p['lane']} ({'PASS' if p['success'] else 'fail'})" if p["lane"]
                 else " (not a lane): " + ", ".join(
                     f"{r['run'][-6:]} {'PASS' if r['success'] else 'fail'}" for r in p["runs"])))
    (HERE / "data" / "index.json").write_text(json.dumps(index, indent=1))

    # data.js — the same payload as a plain global, because the page must open by
    # double-click at a showcase board. `fetch()` on a file:// URL is blocked by CORS,
    # so a JSON-reading page would be silently empty exactly when no one can debug it;
    # a <script src> is not subject to that rule. Images are unaffected either way.
    payload = {str(t): json.loads((HERE / "data" / f"task_{t}.json").read_text())
               for t in TASKS}
    carbon_public = {k: v for k, v in CARBON.items()}
    (HERE / "data.js").write_text(
        "// generated by build_demo_data.py — do not edit\n"
        "window.DEMO_TASKS = " + json.dumps([str(t) for t in TASKS]) + ";\n"
        "window.CARBON = " + json.dumps(carbon_public, indent=1) + ";\n"
        "window.DEMO = " + json.dumps(payload, indent=1) + ";\n")

    total_png = sum(1 for _ in (HERE / "frames").rglob("*.png"))
    mb = sum(f.stat().st_size for f in (HERE / "frames").rglob("*.png")) / 1e6
    print(f"\n✓ {len(TASKS)} tasks -> {HERE/'data'}")
    print(f"✓ data.js ({(HERE/'data.js').stat().st_size/1e3:.0f} kB) + "
          f"{total_png} frames ({mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
