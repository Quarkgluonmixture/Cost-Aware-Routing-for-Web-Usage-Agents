#!/usr/bin/env python3
"""Pure payload construction + B0-vs-B1 diff (no API, no key, no GPU).

Root-cause locator: B0 (tool-calling) omits element_id where B1 (prose-JSON)
includes it, on the SAME observation. B0 and B1 share build_mode_prompt_
dispatch_table() + the same user-message format, so the cause must live in the
two B0-only deltas: (1) the "Output ONLY valid JSON" -> "Use the web_action
tool" one-line replacement, (2) the attached web_action tool schema. This script
constructs both payloads and surfaces exactly what differs + whether element_id
is structurally required.
"""
import json
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from p79.agents._shared_vl_utils import build_mode_prompt_dispatch_table
from p79.agents.proxy_api_agent import _WEB_ACTION_TOOL

_OLD = "Output ONLY valid JSON. No markdown blocks, no explanations."
_NEW = "Use the web_action tool for every action. Put reasoning in the thought parameter."


def main():
    table = build_mode_prompt_dispatch_table()
    b1_prompt = table["dom"]                       # prose-JSON (B1/B2)
    b0_prompt = b1_prompt.replace(_OLD, _NEW)      # tool-calling (B0)

    print("=" * 70)
    print("1) PROMPT DIFF (B1 prose-JSON  vs  B0 tool-calling)")
    print("=" * 70)
    if _OLD in b1_prompt:
        print(f"  B1 has line: {_OLD!r}")
    print(f"  B0 replaced with: {_NEW!r}")
    print(f"  prompts identical apart from that line? {b0_prompt == b1_prompt.replace(_OLD, _NEW)}")
    other = b1_prompt.replace(_OLD, "").strip() == b0_prompt.replace(_NEW, "").strip()
    print(f"  no OTHER text differs?                 {other}")

    print("\n" + "=" * 70)
    print("2) Does the SHARED prompt still describe element_id as required (prose)?")
    print("=" * 70)
    for ln in b0_prompt.splitlines():
        if re.search(r"element_id", ln) and re.search(r"ALWAYS|Type:|Click", ln):
            print("   PROMPT:", ln.strip())

    print("\n" + "=" * 70)
    print("3) web_action TOOL SCHEMA — is element_id STRUCTURALLY required?")
    print("=" * 70)
    params = _WEB_ACTION_TOOL["function"]["parameters"]
    req = params.get("required", [])
    print(f"   parameters.required = {req}")
    print(f"   element_id in required? {'element_id' in req}   <-- THE STRUCTURAL CONSTRAINT")
    eid = params["properties"]["element_id"]
    print(f"   element_id property type = {eid.get('type')}")
    print(f"   element_id description    = {eid.get('description')!r}")
    # fields the model can fill INSTEAD of element_id
    fillable = [k for k in params["properties"] if k not in ("thought", "confidence", "action_type")]
    print(f"   other optional fields the model may fill = {fillable}")

    print("\n" + "=" * 70)
    print("4) ROOT-CAUSE STATEMENT")
    print("=" * 70)
    print("""   B0 receives, simultaneously:
     - PROSE: prompt's Action Schema examples + 'ALWAYS element_id' (governs B1)
     - STRUCTURE: web_action tool schema with required=[action_type, thought]
       (element_id OPTIONAL) + 'Use the web_action tool for every action'
   In tool-calling mode the model follows the STRUCTURAL required-array over the
   prose. element_id is not in required -> model fills action_type+thought+text
   (+ optional url) and omits element_id. B1 has no tool schema, so its only
   action spec is the prose -> it emits element_id. The conflict is B0-only.""")

    # dump full constructed B0 prompt for the record
    out = os.path.join(_REPO, "scripts/spike/b0_dom_prompt_constructed.txt")
    with open(out, "w") as f:
        f.write(b0_prompt)
    print(f"\n[dumped B0 dom system prompt -> {out}]")


if __name__ == "__main__":
    main()
