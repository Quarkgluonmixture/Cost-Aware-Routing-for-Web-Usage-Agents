#!/usr/bin/env python3
"""Local (NON-API) confirmation of the B0 element_id root cause + fix.

Runs Qwen3-VL-4B (B1's local model) on the SAME observation in 3 conditions to
test whether the tool schema's `required` array — not perception/capability —
drives element_id omission:

  A  prose-JSON  (B1's actual mode, no tools)            -> expect element_id PRESENT
  B  tool-calling, element_id OPTIONAL (B0 current)      -> expect element_id OMITTED
  C  tool-calling, element_id REQUIRED (proposed fix)    -> expect element_id PRESENT

A=present, B=omitted, C=present  ⇒  root cause = structural required-array
(reproduces across model size: 4B local ↔ 235B proxy), and the fix is sound —
all without one proxy/API call.

Run on the dedicated A100 (no GPU contention):
  .venv/bin/python3 scripts/spike/local_toolcall_eid_experiment.py
"""
from __future__ import annotations
import copy
import json
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from p79.agents._shared_vl_utils import build_mode_prompt_dispatch_table, format_history
from p79.agents.proxy_api_agent import _WEB_ACTION_TOOL

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

_OLD = "Output ONLY valid JSON. No markdown blocks, no explanations."
_NEW = "Use the web_action tool for every action. Put reasoning in the thought parameter."

# kayak labeled-search page (the canonical B0-failing case) + a couple others.
PAGES = [
    ("classifieds", 0, "results/visualwebarena/phase1/"
     "B0_dom_classifieds_smoke_20260521_011514_927816307_499724_R14256/"
     "phase1_dom_router_0/artifacts/classifieds_task_0/step_000/observation_dom.txt"),
]


def load_intent(site, tid):
    raw = os.path.join(_REPO, f"external/visualwebarena/config_files/vwa/test_{site}.raw.json")
    try:
        for t in json.load(open(raw)):
            if int(t.get("task_id", -1)) == tid:
                return t.get("intent", "")
    except OSError:
        pass
    return "Find the cheapest blue kayak."


def required_variant(required_list):
    tool = copy.deepcopy(_WEB_ACTION_TOOL)
    tool["function"]["parameters"]["required"] = required_list
    return tool


def parse_toolcall(text):
    """Extract arguments dict from Qwen <tool_call>...</tool_call> output."""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:  # some templates emit bare JSON
        m2 = re.search(r"\{.*\}", text, re.DOTALL)
        blob = m2.group(0) if m2 else None
    if not blob:
        return None
    try:
        d = json.loads(blob)
    except Exception:
        return None
    return d.get("arguments", d)


def parse_json_action(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"[load] {MODEL} @ {REV[:8]} ...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, revision=REV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, revision=REV)

    table = build_mode_prompt_dispatch_table()
    b1_prompt = table["dom"]
    b0_prompt = b1_prompt.replace(_OLD, _NEW)

    tool_optional = _WEB_ACTION_TOOL                                   # B current
    tool_req_uncond = required_variant(["action_type", "thought", "element_id"])
    tool_req_cond = copy.deepcopy(_WEB_ACTION_TOOL)                    # proposed fix (if/then)
    tool_req_cond["function"]["parameters"]["allOf"] = [{
        "if": {"properties": {"action_type": {"enum": ["click", "type", "hover", "select_option"]}}},
        "then": {"required": ["element_id"]},
    }]

    def gen(messages, tools=None):
        text = proc.apply_chat_template(messages, tools=tools, tokenize=False,
                                        add_generation_prompt=True)
        inputs = proc(text=[text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return proc.decode(gen_ids, skip_special_tokens=True)

    for site, tid, rel in PAGES:
        obs = open(os.path.join(_REPO, rel), encoding="utf-8", errors="replace").read()
        intent = load_intent(site, tid)
        print(f"\n{'#'*72}\n# {site} task {tid} | intent: {intent[:80]}\n{'#'*72}")

        # A — prose-JSON (B1 mode): single user message, no tools
        userA = f"Task: {intent}\nSystem: {b1_prompt}\n{format_history([])}Accessibility Tree:\n{obs}"
        outA = gen([{"role": "user", "content": userA}], tools=None)
        actA = parse_json_action(outA)
        eidA = (actA or {}).get("element_id")
        print(f"\n[A prose-JSON]    action_type={ (actA or {}).get('action_type')} "
              f"element_id={eidA}  {'✅PRESENT' if eidA is not None else '❌OMITTED'}")
        print(f"   raw: {outA[:200].strip()}")

        # B — tool-calling, element_id OPTIONAL (B0 current)
        userB = f"Task: {intent}\nSystem: {b0_prompt}\n{format_history([])}Accessibility Tree:\n{obs}"
        outB = gen([{"role": "user", "content": userB}], tools=[tool_optional])
        argsB = parse_toolcall(outB)
        eidB = (argsB or {}).get("element_id")
        print(f"\n[B tool eid-OPT]  action_type={(argsB or {}).get('action_type')} "
              f"element_id={eidB} keys={list((argsB or {}).keys())} "
              f"{'✅PRESENT' if eidB is not None else '❌OMITTED'}")
        print(f"   raw: {outB[:200].strip()}")

        # C1 — tool-calling, element_id UNCONDITIONALLY required
        outC1 = gen([{"role": "user", "content": userB}], tools=[tool_req_uncond])
        argsC1 = parse_toolcall(outC1)
        eidC1 = (argsC1 or {}).get("element_id")
        print(f"\n[C1 tool eid-REQ uncond] action_type={(argsC1 or {}).get('action_type')} "
              f"element_id={eidC1} {'✅PRESENT' if eidC1 is not None else '❌OMITTED'}")
        print(f"   raw: {outC1[:200].strip()}")

        # C2 — tool-calling, element_id conditionally required (proposed if/then fix)
        outC2 = gen([{"role": "user", "content": userB}], tools=[tool_req_cond])
        argsC2 = parse_toolcall(outC2)
        eidC2 = (argsC2 or {}).get("element_id")
        print(f"\n[C2 tool eid-REQ if/then] action_type={(argsC2 or {}).get('action_type')} "
              f"element_id={eidC2} {'✅PRESENT' if eidC2 is not None else '❌OMITTED'}")
        print(f"   raw: {outC2[:200].strip()}")

    print(f"\n{'='*72}\n[verdict] A=present & B=omitted ⇒ structural required-array is the cause; "
          f"C1/C2=present ⇒ fix works (model-level, non-API).\n{'='*72}")


if __name__ == "__main__":
    main()
