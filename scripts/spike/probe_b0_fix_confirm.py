#!/usr/bin/env python3
"""Confirm the eid-conditional-required fix on the real proxy across diverse
pages (robustness + safety). Action generation only.

  search/type pages  -> fix should make element_id PRESENT (was omitted)
  listing/scroll pages -> fix must NOT break non-element actions: the if/then
     only requires element_id for click/type/hover/select_option, so scroll/
     back/goto must still validate without it.

Run on A100:  .venv/bin/python3 scripts/spike/probe_b0_fix_confirm.py
"""
from __future__ import annotations
import copy
import json
import os
import sys
from pathlib import Path

import requests

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from p79.agents._shared_vl_utils import build_mode_prompt_dispatch_table, format_history
from p79.agents.proxy_api_agent import _WEB_ACTION_TOOL

_OLD = "Output ONLY valid JSON. No markdown blocks, no explanations."
_NEW = "Use the web_action tool for every action. Put reasoning in the thought parameter."


def _key():
    k = os.environ.get("PROXY_API_KEY", "")
    if k:
        return k
    p = Path(_REPO) / ".auth" / "qwen_api"
    if p.exists():
        for ln in p.read_text().splitlines():
            if ln.strip().startswith("rp_"):
                return ln.strip()
    return ""


API_KEY = _key()
BASE_URL = "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke"
MODEL = "qwen.qwen3-vl-235b-a22b"
HEADERS = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
RUN = "results/visualwebarena/phase1/B0_dom_classifieds_20260520_133223_291174117_410560_R2987"
ART = RUN + "/phase1_dom_router_0/artifacts"

# (task, step, category)
PAGES = [
    (19, 0, "search"), (34, 0, "search"), (13, 0, "search"), (51, 1, "search"),
    (19, 6, "listing"), (19, 2, "listing"),
]
INTENTS = {}


def load_intent(tid):
    if not INTENTS:
        raw = os.path.join(_REPO, "external/visualwebarena/config_files/vwa/test_classifieds.raw.json")
        try:
            for t in json.load(open(raw)):
                INTENTS[int(t.get("task_id", -1))] = t.get("intent", "")
        except OSError:
            pass
    return INTENTS.get(tid, "Complete the task on this page.")


def fix_tool():
    t = copy.deepcopy(_WEB_ACTION_TOOL)
    t["function"]["parameters"]["allOf"] = [{
        "if": {"properties": {"action_type": {"enum": ["click", "type", "hover", "select_option"]}}},
        "then": {"required": ["element_id"]}}]
    return t


FIX = fix_tool()


def msg(tid, step):
    obs = open(f"{_REPO}/{ART}/classifieds_task_{tid}/step_{step:03d}/observation_dom.txt",
               encoding="utf-8", errors="replace").read()
    sysp = build_mode_prompt_dispatch_table()["dom"].replace(_OLD, _NEW)
    return [{"role": "user", "content":
             f"Task: {load_intent(tid)}\nSystem: {sysp}\n{format_history([])}Accessibility Tree:\n{obs}"}]


def call(messages, tool):
    p = {"model": MODEL, "messages": messages, "max_tokens": 512, "temperature": 0.0,
         "tools": [tool], "tool_choice": "required"}
    try:
        r = requests.post(BASE_URL, json=p, headers=HEADERS, timeout=90)
        b = r.json()
    except Exception as e:
        return None, f"err:{e}"
    tc = b.get("tool_calls") if isinstance(b, dict) else None
    if not tc:
        return None, "no_tool_call"
    try:
        a = json.loads(tc[0].get("function", {}).get("arguments", ""))
    except Exception:
        return None, "unparseable"
    return a, None


def needs_eid(at):
    return at in ("click", "type", "hover", "select_option")


def main():
    if not API_KEY:
        print("no key", file=sys.stderr); return 1
    print(f"[confirm] key={API_KEY[:6]}... (baseline vs eid-cond-required fix)\n")
    rows = []
    for tid, step, cat in PAGES:
        m = msg(tid, step)
        a_base, e_base = call(m, _WEB_ACTION_TOOL)
        a_fix, e_fix = call(m, FIX)
        def fmt(a, e):
            if a is None:
                return f"({e})"
            at, eid = a.get("action_type"), a.get("element_id")
            ok = (eid is not None) if needs_eid(at) else True
            return f"{at} eid={eid} {'VALID' if ok else 'INVALID'}"
        base_s, fix_s = fmt(a_base, e_base), fmt(a_fix, e_fix)
        rows.append((tid, step, cat, base_s, fix_s))
        print(f"[task {tid:>2} step {step} {cat:8s}] baseline: {base_s:34s} | FIX: {fix_s}")
    out = Path(_REPO) / "scripts/spike/probe_b0_fix_confirm_out.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n[saved] {out}")
    # tally
    search = [r for r in rows if r[2] == "search"]
    fix_valid = sum(1 for r in search if "VALID" in r[4] and "INVALID" not in r[4])
    print(f"\n=== search pages: fix VALID {fix_valid}/{len(search)} "
          f"(baseline was omit) ===")
    listing = [r for r in rows if r[2] == "listing"]
    lv = sum(1 for r in listing if "INVALID" not in r[4])
    print(f"=== listing pages: fix produced VALID action {lv}/{len(listing)} "
          f"(conditional must NOT break scroll/click) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
