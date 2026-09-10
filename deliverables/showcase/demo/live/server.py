"""Live lanes for the showcase demo's "Try your own" tab.

WHAT IT DOES
------------
A visitor types a task; this server writes a one-task config and starts the ordinary
p79 runner three times — LOOK (vision), READ (dom), BOTH (SoM) — against the
classifieds site on quark. The page then follows each lane over Server-Sent Events:
the agent's own screenshot, the click drawn from its step record, its stated reason,
and the same three meters as the replay (cost, time, CO2e range).

It reads the runner's own outputs as they land (the step JSONL is fsync'd per step,
the screenshot is written right after the step's timing window), so nothing here
re-implements the agent. The overlay geometry and the CO2e estimate come from
`build_demo_data.py`, so a live lane and a recorded lane are drawn by the same code.

WHAT IT IS NOT
--------------
* Not scored. There is no evaluator for a free-text task; the task config carries a
  sentinel answer no agent will produce, and the page never shows a ✓/✗ for a live
  lane — only what the agent said when it finished, for the visitor to judge.
* Not paper data. Output goes to demo/live/runs/ (gitignored), P79_PAPER_GRADE=0.
* Not concurrent. One session at a time: the three lanes already share one site and
  one login, which is the collision the launch rules exist to prevent in paper-grade
  runs; a second visitor's three more lanes on top would make it worse.

Run on DGX:   .venv/bin/python3 deliverables/showcase/demo/live/server.py
              (LIVE_PORT=8799, LIVE_MAX_STEPS=12, VWA_REMOTE_HOST=100.95.81.103)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
import time
import uuid
from pathlib import Path

from aiohttp import web

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
REPO = DEMO.parents[2]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(REPO))
import build_demo_data as B  # noqa: E402  (overlay geometry + CO2e estimate, shared with the replay)
from p79.experiment.io_utils import read_jsonl_dedup  # noqa: E402

RUNS = HERE / "runs"
LANE_MODE = {"LOOK": "vision", "READ": "dom", "BOTH": "som"}
CAP = int(os.environ.get("LIVE_MAX_STEPS", "12"))
PORT = int(os.environ.get("LIVE_PORT", "8799"))
SESSION_WALL_S = int(os.environ.get("LIVE_SESSION_WALL_S", "480"))
TASK_ID = 90000          # outside every real VWA task id, so it can't be mistaken for one
UNSCORED = "__p79_live_demo_unscored__"

# The site's PHP sessions expire after ~24 min (VWA quirk); a login younger than this
# is reused, anything older is redone before the next session starts.
LOGIN_TTL_S = 15 * 60
AUTH_STATE = RUNS / "auth" / "classifieds_state.json"

SESSION: dict | None = None   # the one running session
START_LOCK = asyncio.Lock()   # two visitors pressing Run at once must not both launch


async def _ensure_login() -> bool:
    if AUTH_STATE.exists() and time.time() - AUTH_STATE.stat().st_mtime < LOGIN_TTL_S:
        return True
    with open(RUNS / "login.log", "ab") as log:
        p = await asyncio.create_subprocess_exec(
            "bash", str(HERE / "run_lane.sh"), "login", str(AUTH_STATE.parent),
            stdout=log, stderr=asyncio.subprocess.STDOUT)
        try:
            rc = await asyncio.wait_for(p.wait(), 90)
        except asyncio.TimeoutError:
            p.kill()
            rc = -1
    return rc == 0 and AUTH_STATE.exists()


def _task_json(intent: str, storage_state: Path) -> dict:
    return {
        "sites": ["classifieds"], "task_id": TASK_ID,
        # The runner does not log in again on a lane's first episode; it opens the
        # browser with whatever state file the task names. So the server logs in to
        # quark's site itself before each session and points every lane here.
        "require_login": True, "storage_state": str(storage_state),
        "start_url": "__CLASSIFIEDS__", "geolocation": None,
        "viewport_size": {"width": 1280},
        "intent_template": "{{intent}}", "intent": intent, "image": None,
        "instantiation_dict": {"intent": intent},
        # One site, three lanes at once: a per-task reset in one lane would wipe the
        # page out from under the other two mid-step.
        "require_reset": False,
        # Local string check against a sentinel — no LLM judge is called and BLIP-2 is
        # never loaded (that only happens for page_image_query). The score is ignored.
        "eval": {"eval_types": ["string_match"],
                 "reference_answers": {"must_include": [UNSCORED]},
                 "reference_url": "", "program_html": [], "string_note": "",
                 "reference_answer_raw_annotation": ""},
        "reasoning_difficulty": "", "visual_difficulty": "", "overall_difficulty": "",
        "comments": "live showcase demo — unscored", "intent_template_id": -1,
    }


def _cfg_yaml(mode: str, run_id: str, task_path: Path, out_root: Path) -> str:
    # JSON is valid YAML, so the generated config needs no YAML emitter.
    return json.dumps({
        "defaults": [f"configs/exp_v2_B0_{mode}_classifieds.yaml"],
        "experiment": {"name": f"LIVE_B0_{mode}_classifieds", "run_id": run_id,
                       "output_root": str(out_root)},
        "task": {"include_sites": ["classifieds"],
                 "site_configs": {"classifieds": str(task_path)},
                 "task_ids": {"classifieds": [TASK_ID]}},
        "runtime": {"resume": False},
    }, indent=1)


def _cond_dir(s: dict, lane: str) -> Path:
    mode = LANE_MODE[lane]
    return s["out"] / "visualwebarena" / "phase1" / s["run_ids"][lane] / f"phase1_{mode}_router_0"


def _img_path(s: dict, lane: str, i: int) -> Path:
    art = _cond_dir(s, lane) / "artifacts" / f"classifieds_task_{TASK_ID}"
    if lane == "BOTH":
        som = art / "som" / f"step_{i:03d}_som.png"
        if som.exists():
            return som
    return art / f"step_{i:03d}" / "screenshot.png"


def _clean_intent(raw: str) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", " ", raw or "").strip()
    return re.sub(r"\s+", " ", s)[:300]


async def _kill(s: dict) -> None:
    for p in s["procs"].values():
        if p.returncode is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    await asyncio.sleep(2)
    for p in s["procs"].values():
        if p.returncode is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


async def start(request: web.Request) -> web.Response:
    global SESSION
    body = await request.json()
    intent = _clean_intent(body.get("intent", ""))
    if len(intent) < 8:
        return web.json_response({"error": "task too short"}, status=400)
    async with START_LOCK:
        if SESSION and any(p.returncode is None for p in SESSION["procs"].values()):
            return web.json_response({"error": "busy", "id": SESSION["id"]}, status=409)
        if not await _ensure_login():
            return web.json_response(
                {"error": "could not log in to the classifieds site — is the site up? (see live/runs/login.log)"},
                status=502)
        return await _launch(intent)


async def _launch(intent: str) -> web.Response:
    global SESSION
    sid = time.strftime("%H%M%S") + "_" + uuid.uuid4().hex[:4]
    sdir = RUNS / sid
    sdir.mkdir(parents=True)
    task_path = sdir / "task.json"
    task_path.write_text(json.dumps([_task_json(intent, AUTH_STATE)], indent=1))
    s = {"id": sid, "intent": intent, "dir": sdir, "out": sdir / "out", "cap": CAP,
         "t0": time.time(), "run_ids": {}, "procs": {}}
    for lane, mode in LANE_MODE.items():
        run_id = f"LIVE_{sid}_{mode}"
        cfg = sdir / f"{lane}.yaml"
        cfg.write_text(_cfg_yaml(mode, run_id, task_path, s["out"]))
        log = open(sdir / f"{lane}.log", "wb")
        s["run_ids"][lane] = run_id
        s["procs"][lane] = await asyncio.create_subprocess_exec(
            "bash", str(HERE / "run_lane.sh"), str(cfg), run_id, str(CAP),
            stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    SESSION = s
    asyncio.get_running_loop().call_later(
        SESSION_WALL_S, lambda: asyncio.ensure_future(_kill(s)))
    return web.json_response({"id": sid, "cap": CAP, "intent": intent})


def _event(s: dict, lane: str, recs: list, i: int, acc: dict) -> dict:
    rec = recs[i]
    acc["usd"] += (rec.get("cost_usd") or {}).get("model") or 0.0
    acc["sec"] += ((rec.get("latency_ms") or {}).get("total") or 0.0) / 1000.0
    tok = rec.get("tokens") or {}
    acc["tin"] += tok.get("input") or 0
    acc["tout"] += tok.get("output") or 0
    try:
        mark = B._mark(rec, lane, 1280, 720)
    except B.BuildError:
        mark = None           # an unexpected coordinate convention: draw no click rather than a wrong one
    act = rec.get("action") or {}
    lo, hi = B._co2_g(acc["tin"], acc["tout"])
    return {"type": "step", "lane": lane, "step": i,
            "img": f"/frame/{s['id']}/{lane}/{i}?v={int(time.time()*1000)}",
            "thought": (act.get("thought") or "").strip(),
            "action": rec.get("action_type"), "mark": mark,
            "answer": act.get("answer") if rec.get("action_type") in ("finish", "stop") else None,
            "cost_cum": round(acc["usd"], 4), "sec_cum": round(acc["sec"], 1),
            "tok_cum": acc["tin"] + acc["tout"], "co2_cum": [round(lo, 4), round(hi, 4)]}


async def events(request: web.Request) -> web.StreamResponse:
    sid = request.match_info["sid"]
    s = SESSION
    if not s or s["id"] != sid:
        raise web.HTTPNotFound()
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                       "Cache-Control": "no-cache",
                                       "Access-Control-Allow-Origin": "*"})
    await resp.prepare(request)

    async def send(obj: dict) -> None:
        await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

    await send({"type": "hello", "id": sid, "intent": s["intent"], "cap": s["cap"]})
    sent = {k: 0 for k in LANE_MODE}
    acc = {k: {"usd": 0.0, "sec": 0.0, "tin": 0, "tout": 0} for k in LANE_MODE}
    finished = set()
    last_ping = time.time()
    while len(finished) < len(LANE_MODE):
        for lane in LANE_MODE:
            if lane in finished:
                continue
            steps = _cond_dir(s, lane) / "episodes" / f"classifieds_task_{TASK_ID}_steps_v2.jsonl"
            recs = read_jsonl_dedup(str(steps)) if steps.exists() else []
            while sent[lane] < len(recs):
                i = sent[lane]
                # the screenshot is written just after the step's timing window;
                # give it a moment rather than send a frame that 404s
                for _ in range(20):
                    if _img_path(s, lane, i).exists():
                        break
                    await asyncio.sleep(0.25)
                await send(_event(s, lane, recs, i, acc[lane]))
                sent[lane] += 1
            p = s["procs"][lane]
            if p.returncode is not None and sent[lane] >= len(recs):
                last = recs[-1] if recs else {}
                if last.get("action_type") in ("finish", "stop"):
                    why = "finished"
                elif len(recs) >= s["cap"]:
                    why = "budget"
                else:
                    why = "error"
                await send({"type": "done", "lane": lane, "why": why, "steps": len(recs),
                            "answer": (last.get("action") or {}).get("answer"),
                            "exit": p.returncode})
                finished.add(lane)
        if time.time() - last_ping > 10:
            await resp.write(b": ping\n\n")
            last_ping = time.time()
        await asyncio.sleep(0.5)
    await send({"type": "end", "id": sid})
    return resp


async def frame(request: web.Request) -> web.FileResponse:
    s = SESSION
    sid, lane, i = request.match_info["sid"], request.match_info["lane"], request.match_info["i"]
    if not s or s["id"] != sid or lane not in LANE_MODE or not i.isdigit():
        raise web.HTTPNotFound()
    p = _img_path(s, lane, int(i))
    if not p.exists():
        raise web.HTTPNotFound()
    return web.FileResponse(p, headers={"Access-Control-Allow-Origin": "*",
                                        "Cache-Control": "no-store"})


async def stop(request: web.Request) -> web.Response:
    if SESSION:
        await _kill(SESSION)
    return web.json_response({"stopped": True})


async def health(request: web.Request) -> web.Response:
    busy = bool(SESSION and any(p.returncode is None for p in SESSION["procs"].values()))
    return web.json_response({"ok": True, "busy": busy, "cap": CAP,
                              "carbon": B.CARBON is not None})


@web.middleware
async def cors(request: web.Request, handler):
    # The page is opened from file:// (origin "null"), so every response needs CORS.
    if request.method == "OPTIONS":
        resp = web.Response()
    else:
        resp = await handler(request)
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


def main() -> None:
    RUNS.mkdir(exist_ok=True)
    app = web.Application(middlewares=[cors])
    app.add_routes([web.get("/health", health), web.post("/run", start),
                    web.options("/run", health), web.get("/events/{sid}", events),
                    web.get("/frame/{sid}/{lane}/{i}", frame), web.post("/stop", stop),
                    web.options("/stop", health)])
    print(f"live demo server on :{PORT}  cap={CAP} steps  runs -> {RUNS}")
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)


if __name__ == "__main__":
    main()
