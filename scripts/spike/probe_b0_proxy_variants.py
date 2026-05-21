#!/usr/bin/env python3
"""B0 proxy variant probe — isolate the type/search element_id omission cause
AND test fixes, on the REAL prompt + REAL kayak search obs (action generation
only; nothing is executed).

Earlier non-API evidence localized the cause to the proxy 235B (local 4B + the
SAME optional schema emits element_id; 235B emits it on CLICK but omits on
TYPE/search + hedges url). This probe runs the proxy directly with controlled
variants:

  baseline      current schema (eid optional) + tool_choice=required   [reproduce omit]
  tc_auto       same, tool_choice="auto"                                [forced-vs-not isolation]
  eid_req_cond  element_id conditionally required (if/then)             [fix: structural]
  eid_req_uncond element_id unconditionally required                    [fix: structural strong]
  no_url        url field removed from schema                          [fix: remove the hedge]
  prompt_nudge  + explicit 'search = type into element [N], never url'  [fix: prompt]

Reads the key via the same .auth loader the queue uses (script-side; not printed).
Run on A100:  .venv/bin/python3 scripts/spike/probe_b0_proxy_variants.py
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


def _load_key():
    k = os.environ.get("PROXY_API_KEY", "")
    if k:
        return k
    p = Path(_REPO) / ".auth" / "qwen_api"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith("rp_"):
                return line
    return ""


API_KEY = _load_key()
BASE_URL = os.environ.get("PROXY_API_BASE",
                          "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke")
MODEL = os.environ.get("PROXY_MODEL_NAME", "qwen.qwen3-vl-235b-a22b")
HEADERS = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}

# Real kayak search page (the canonical TYPE/search failure) + intent.
OBS_REL = ("results/visualwebarena/phase1/"
           "B0_dom_classifieds_smoke_20260521_011514_927816307_499724_R14256/"
           "phase1_dom_router_0/artifacts/classifieds_task_0/step_000/observation_dom.txt")
INTENT = "Find me the cheapest blue kayak on this site."


def build_user_msg(extra_nudge=""):
    obs = open(os.path.join(_REPO, OBS_REL), encoding="utf-8", errors="replace").read()
    sysp = build_mode_prompt_dispatch_table()["dom"].replace(_OLD, _NEW) + extra_nudge
    return f"Task: {INTENT}\nSystem: {sysp}\n{format_history([])}Accessibility Tree:\n{obs}"


def tool_eid_required(conditional):
    t = copy.deepcopy(_WEB_ACTION_TOOL)
    if conditional:
        t["function"]["parameters"]["allOf"] = [{
            "if": {"properties": {"action_type": {"enum": ["click", "type", "hover", "select_option"]}}},
            "then": {"required": ["element_id"]}}]
    else:
        t["function"]["parameters"]["required"] = ["action_type", "thought", "element_id"]
    return t


def tool_no_url():
    t = copy.deepcopy(_WEB_ACTION_TOOL)
    t["function"]["parameters"]["properties"].pop("url", None)
    return t


def post(payload, label):
    try:
        r = requests.post(BASE_URL, json=payload, headers=HEADERS, timeout=90)
    except requests.RequestException as e:
        return {"label": label, "ok": False, "err": f"net:{e}"}
    out = {"label": label, "status": r.status_code, "ok": r.status_code == 200,
           "elapsed_ms": int(r.elapsed.total_seconds() * 1000)}
    try:
        out["body"] = r.json()
    except json.JSONDecodeError:
        out["body_text"] = r.text[:500]
    return out


def verdict(r):
    if not r.get("ok"):
        return f"HTTP {r.get('status')} {str(r.get('body') or r.get('body_text') or r.get('err'))[:120]}"
    body = r["body"]
    tc = body.get("tool_calls") if isinstance(body, dict) else None
    if not tc:
        c = body.get("content") if isinstance(body, dict) else None
        return f"NO tool_call (content head: {str(c)[:90]})"
    args_raw = tc[0].get("function", {}).get("arguments", "")
    try:
        a = json.loads(args_raw)
    except Exception:
        return f"tool_call UNPARSEABLE: {args_raw[:90]}"
    eid = a.get("element_id")
    return (f"action={a.get('action_type')} element_id={eid} "
            f"url={'YES' if a.get('url') else 'no'} keys={sorted(a.keys())} "
            f"=> {'✅eid PRESENT' if eid is not None else '❌eid OMITTED'}")


def main():
    if not API_KEY:
        print("ERROR: no PROXY_API_KEY / .auth key", file=sys.stderr)
        return 1
    print(f"[probe] endpoint={BASE_URL} model={MODEL} key={API_KEY[:6]}...\n")
    base_msg = [{"role": "user", "content": build_user_msg()}]
    nudge = ("\n\nCRITICAL FOR SEARCH/FILTER: to search or filter, you MUST use "
             "action_type='type' with the element_id [N] of the search/keyword "
             "textbox shown in the Accessibility Tree. NEVER use 'url'/'goto' to "
             "perform a search. element_id is MANDATORY for type.")
    nudge_msg = [{"role": "user", "content": build_user_msg(nudge)}]
    common = {"model": MODEL, "max_tokens": 512, "temperature": 0.0}

    variants = [
        ("baseline_req",   {**common, "messages": base_msg, "tools": [_WEB_ACTION_TOOL], "tool_choice": "required"}),
        ("tc_auto",        {**common, "messages": base_msg, "tools": [_WEB_ACTION_TOOL], "tool_choice": "auto"}),
        ("eid_req_cond",   {**common, "messages": base_msg, "tools": [tool_eid_required(True)], "tool_choice": "required"}),
        ("eid_req_uncond", {**common, "messages": base_msg, "tools": [tool_eid_required(False)], "tool_choice": "required"}),
        ("no_url",         {**common, "messages": base_msg, "tools": [tool_no_url()], "tool_choice": "required"}),
        ("prompt_nudge",   {**common, "messages": nudge_msg, "tools": [_WEB_ACTION_TOOL], "tool_choice": "required"}),
    ]
    results = []
    for label, payload in variants:
        r = post(payload, label)
        v = verdict(r)
        results.append((label, v, r))
        print(f"[{label:15s}] {v}  ({r.get('elapsed_ms','?')}ms)")

    out = Path(_REPO) / "scripts/spike/probe_b0_proxy_variants_out.json"
    out.write_text(json.dumps([{"label": l, "verdict": v,
                                "body": r.get("body"), "status": r.get("status")}
                               for l, v, r in results], indent=2, default=str))
    print(f"\n[saved] {out}")
    print("\n=== READ ===")
    print("baseline omit + tc_auto present  => forcing(required) is the trigger")
    print("baseline omit + tc_auto omit     => model prior (not forcing)")
    print("eid_req_* present                => structural-required fix WORKS on proxy")
    print("no_url present                   => removing hedge field fixes it")
    print("prompt_nudge present             => prompt fix works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
