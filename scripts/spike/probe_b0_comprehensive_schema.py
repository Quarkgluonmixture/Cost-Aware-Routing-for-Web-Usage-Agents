#!/usr/bin/env python3
"""Test the COMPREHENSIVE mode-aware conditional schema on the real proxy.

The element_id fix only covered one field. Root cause (tool_choice=required ->
minimal call omits optional fields) is GENERAL, so every action-type's
semantically-required field needs a conditional required. Also: requiring
element_id strictly breaks VISION mode (no AXTree -> coordinate only), so
click/type/hover must require element_id OR coordinate (anyOf).

This probe checks the proxy ENFORCES:
  (1) anyOf in then  (element_id OR coordinate) — dom should still yield element_id
  (2) multiple if/then conditionals (text, scroll_direction, option_*, ...)
across diverse dom pages. Action generation only.

Run on A100.
"""
from __future__ import annotations
import copy, json, os, sys
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
    return next((l.strip() for l in p.read_text().splitlines()
                 if l.strip().startswith("rp_")), "") if p.exists() else ""


API_KEY = _key()
BASE_URL = "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke"
MODEL = "qwen.qwen3-vl-235b-a22b"
HEADERS = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
ART = ("results/visualwebarena/phase1/"
       "B0_dom_classifieds_20260520_133223_291174117_410560_R2987/phase1_dom_router_0/artifacts")

# (task, step, what we hope to elicit)
PAGES = [(19, 0, "type-search"), (34, 0, "type-search"),
         (19, 6, "listing-scroll/type"), (19, 8, "listing"),
         (13, 0, "dropdown/select")]
INTENTS = {}


def intent(tid):
    if not INTENTS:
        raw = f"{_REPO}/external/visualwebarena/config_files/vwa/test_classifieds.raw.json"
        try:
            for t in json.load(open(raw)):
                INTENTS[int(t.get("task_id", -1))] = t.get("intent", "")
        except OSError:
            pass
    return INTENTS.get(tid, "Complete the task.")


def comprehensive(tool):
    t = copy.deepcopy(tool)
    t["function"]["parameters"]["allOf"] = [
        {"if": {"properties": {"action_type": {"enum": ["click", "type", "hover"]}}},
         "then": {"anyOf": [{"required": ["element_id"]}, {"required": ["coordinate"]}]}},
        {"if": {"properties": {"action_type": {"const": "type"}}},
         "then": {"required": ["text"]}},
        {"if": {"properties": {"action_type": {"const": "select_option"}}},
         "then": {"required": ["element_id"],
                  "anyOf": [{"required": ["option_label"]}, {"required": ["option_value"]},
                            {"required": ["option_index"]}]}},
        {"if": {"properties": {"action_type": {"const": "scroll"}}},
         "then": {"required": ["scroll_direction"]}},
        {"if": {"properties": {"action_type": {"const": "tab_focus"}}},
         "then": {"required": ["page_number"]}},
        {"if": {"properties": {"action_type": {"const": "press"}}},
         "then": {"required": ["key"]}},
        {"if": {"properties": {"action_type": {"const": "goto"}}},
         "then": {"required": ["url"]}},
    ]
    return t


COMP = comprehensive(_WEB_ACTION_TOOL)

# what each action_type semantically needs (for completeness check)
NEED = {"click": ["element_id|coordinate"], "type": ["element_id|coordinate", "text"],
        "hover": ["element_id|coordinate"], "select_option": ["element_id", "option_label|option_value|option_index"],
        "scroll": ["scroll_direction"], "tab_focus": ["page_number"], "press": ["key"], "goto": ["url"]}


def has(a, spec):
    return any(a.get(k) is not None for k in spec.split("|"))


def call(tid, step, tool):
    obs = open(f"{_REPO}/{ART}/classifieds_task_{tid}/step_{step:03d}/observation_dom.txt",
               encoding="utf-8", errors="replace").read()
    sysp = build_mode_prompt_dispatch_table()["dom"].replace(_OLD, _NEW)
    msg = [{"role": "user", "content":
            f"Task: {intent(tid)}\nSystem: {sysp}\n{format_history([])}Accessibility Tree:\n{obs}"}]
    p = {"model": MODEL, "messages": msg, "max_tokens": 512, "temperature": 0.0,
         "tools": [tool], "tool_choice": "required"}
    try:
        b = requests.post(BASE_URL, json=p, headers=HEADERS, timeout=90).json()
        a = json.loads(b["tool_calls"][0]["function"]["arguments"])
        return a
    except Exception as e:
        return {"_err": str(e)[:80]}


def complete(a):
    at = a.get("action_type")
    if at not in NEED:
        return True, "no-req-fields"
    miss = [s for s in NEED[at] if not has(a, s)]
    return (len(miss) == 0), ("OK" if not miss else f"MISSING {miss}")


def main():
    if not API_KEY:
        print("no key", file=sys.stderr); return 1
    print(f"[comp-probe] key={API_KEY[:6]}... current(eid-only) vs comprehensive\n")
    for tid, step, hint in PAGES:
        a_cur = call(tid, step, _WEB_ACTION_TOOL)   # current shipped schema (eid-only fix not yet on this _WEB_ACTION_TOOL copy? see note)
        a_comp = call(tid, step, COMP)
        ok_cur, m_cur = complete(a_cur) if "_err" not in a_cur else (False, a_cur["_err"])
        ok_comp, m_comp = complete(a_comp) if "_err" not in a_comp else (False, a_comp["_err"])
        print(f"[t{tid} s{step} {hint:20s}]")
        print(f"    current : {a_cur.get('action_type')} -> {'✅' if ok_cur else '❌'} {m_cur}")
        print(f"    COMPREHENSIVE: {a_comp.get('action_type')} -> {'✅' if ok_comp else '❌'} {m_comp}  keys={sorted(k for k in a_comp if not k.startswith('_'))}")
    print("\n[read] comprehensive ✅ on all action types it produces => multi-if/then + anyOf enforce on proxy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
