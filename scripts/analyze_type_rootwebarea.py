#!/usr/bin/env python3
"""
Analyze steps where action_type == "type" targets a RootWebArea element.

For each condition, reads JSONL step files with dedup logic (keep last run
if step_idx resets), then cross-references the element_id against
observation_dom.txt to check if it's a RootWebArea.
"""

import json
import os
import re
from pathlib import Path


RUN_DIR = Path(
    "/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/"
    "results/visualwebarena/phase1/B1_3mode_classifieds_20260404_141103"
)

CONDITIONS = ["phase1_dom_router_0", "phase1_som_router_0"]


from p79.experiment.io_utils import read_jsonl_dedup


def find_element_in_dom(dom_path, element_id):
    """
    Look up [element_id] in the DOM text and return the full line if it
    contains 'RootWebArea'. Returns None if not found or not RootWebArea.
    """
    if not os.path.isfile(dom_path):
        return None

    pattern = re.compile(rf"^\[{element_id}\]\s+RootWebArea\b")
    with open(dom_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if pattern.match(line.strip()):
                return line.strip()
    return None


def analyze_condition(condition_id):
    """Return list of hits for this condition."""
    episodes_dir = RUN_DIR / condition_id / "episodes"
    artifacts_dir = RUN_DIR / condition_id / "artifacts"
    hits = []

    jsonl_files = sorted(episodes_dir.glob("*_steps_v2.jsonl"))

    for jf in jsonl_files:
        m = re.search(r"task_(\d+)_steps_v2\.jsonl$", jf.name)
        if not m:
            continue
        task_id = int(m.group(1))

        steps = read_jsonl_dedup(jf)

        for step in steps:
            if step.get("action_type") != "type":
                continue

            action = step.get("action", {})
            element_id = action.get("element_id")
            if element_id is None:
                continue

            step_idx = step["step_idx"]
            dom_path = (
                artifacts_dir
                / f"classifieds_task_{task_id}"
                / f"step_{step_idx:03d}"
                / "observation_dom.txt"
            )

            dom_line = find_element_in_dom(str(dom_path), element_id)
            if dom_line is not None:
                hits.append(
                    {
                        "condition": condition_id,
                        "task_id": task_id,
                        "step_idx": step_idx,
                        "element_id": element_id,
                        "dom_element": dom_line,
                        "typed_text": action.get("text", ""),
                        "thought": action.get("thought", ""),
                    }
                )

    return hits


def main():
    all_hits = []
    for cond in CONDITIONS:
        print(f"\n{'='*80}")
        print(f"Condition: {cond}")
        print(f"{'='*80}")
        hits = analyze_condition(cond)
        all_hits.extend(hits)

        if not hits:
            print("  No type->RootWebArea steps found.")
            continue

        unique_tasks = set(h["task_id"] for h in hits)
        print(f"  Found {len(hits)} type->RootWebArea steps across {len(unique_tasks)} unique tasks\n")

        for h in sorted(hits, key=lambda x: (x["task_id"], x["step_idx"])):
            print(f"  task_id={h['task_id']:>4d}  step_idx={h['step_idx']:>2d}  "
                  f"element_id={h['element_id']:>4d}")
            print(f"    DOM line : {h['dom_element']}")
            print(f"    Typed    : {h['typed_text']!r}")
            print(f"    Thought  : {h['thought'][:120]}")
            print()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for cond in CONDITIONS:
        cond_hits = [h for h in all_hits if h["condition"] == cond]
        unique_tasks = set(h["task_id"] for h in cond_hits)
        print(f"  {cond}: {len(cond_hits)} steps, {len(unique_tasks)} unique tasks affected")

    total_tasks = set(h["task_id"] for h in all_hits)
    print(f"\n  Total across conditions: {len(all_hits)} steps, {len(total_tasks)} unique task IDs")


if __name__ == "__main__":
    main()
