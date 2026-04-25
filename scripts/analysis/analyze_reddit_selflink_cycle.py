#!/usr/bin/env python3
"""
Analyze Reddit "comment self-link cycle" escape patterns in B0 and B1 DOM mode.

Detects tasks where the agent repeatedly clicks on a post page without the URL
changing (self-link cycle), then classifies how the agent escapes the cycle.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = Path("/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results/visualwebarena/phase1")

RUNS = {
    "B0": BASE / "B0_3mode_reddit_20260422" / "phase1_dom_router_0",
    "B1": BASE / "B1_3mode_reddit_20260413" / "phase1_dom_router_0",
}

# Reddit post page pattern: /f/<forum>/<id> with optional slug
POST_PAGE_RE = re.compile(r"/f/\w+/\d+")

MIN_CYCLE_CLICKS = 2  # minimum consecutive same-URL clicks to count as a cycle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_steps(jsonl_path: Path) -> list[dict]:
    """Read a steps JSONL file, skipping corrupt lines."""
    steps = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    steps.sort(key=lambda s: s.get("step_idx", 0))
    return steps


def read_summary(summary_path: Path) -> dict:
    """Read a summary JSON file (pretty-printed, not JSONL)."""
    with open(summary_path) as f:
        return json.load(f)


def is_post_page(url: str) -> bool:
    """Check if URL matches a Reddit post page."""
    if not url:
        return False
    return bool(POST_PAGE_RE.search(url))


def normalize_post_url(url: str) -> str:
    """Normalize post URL by stripping slug suffix for comparison.

    /f/memes/127531 and /f/memes/127531/too-much-of-anything
    should be considered the same post.
    """
    m = re.search(r"(/f/\w+/\d+)", url)
    return m.group(1) if m else url


def detect_cycles(steps: list[dict]) -> list[dict]:
    """Detect self-link cycles in a step sequence.

    A cycle is >=MIN_CYCLE_CLICKS consecutive click actions on a post page
    where the normalized post URL doesn't change.

    Returns list of cycle info dicts.
    """
    cycles = []
    i = 0
    while i < len(steps):
        step = steps[i]
        obs_url = step.get("obs_url", "")
        action_type = step.get("action_type", "")

        # Must be on a post page and clicking
        if not is_post_page(obs_url) or action_type != "click":
            i += 1
            continue

        # Start tracking a potential cycle
        post_id = normalize_post_url(obs_url)
        cycle_start = i
        j = i + 1

        while j < len(steps):
            next_step = steps[j]
            next_url = next_step.get("obs_url", "")
            next_action = next_step.get("action_type", "")
            next_post_id = normalize_post_url(next_url) if is_post_page(next_url) else None

            # Still on same post page and clicking
            if next_action == "click" and next_post_id == post_id:
                j += 1
                continue
            else:
                break

        cycle_length = j - cycle_start  # number of consecutive clicks on same post
        if cycle_length >= MIN_CYCLE_CLICKS:
            cycles.append({
                "cycle_start": cycle_start,
                "cycle_end": j - 1,  # last click step index
                "cycle_length": cycle_length,
                "post_url": post_id,
                "first_escape_idx": j if j < len(steps) else None,
            })

        i = j  # skip past this cycle

    return cycles


def classify_escape(steps: list[dict], cycle_info: dict) -> dict:
    """Classify how the agent escapes a cycle.

    Returns a dict with escape classification details.
    """
    escape_idx = cycle_info["first_escape_idx"]
    total_steps = len(steps)
    last_step = steps[-1]
    is_done = last_step.get("done", False)

    if escape_idx is None or escape_idx >= total_steps:
        return {
            "escape_type": "truncated",
            "escape_step": None,
            "escape_action": None,
            "post_escape_actions": [],
        }

    escape_step = steps[escape_idx]
    escape_action = escape_step.get("action_type", "")
    escape_url = escape_step.get("obs_url", "")
    post_url = cycle_info["post_url"]

    # Collect post-escape action sequence (up to 5 steps after escape)
    post_escape = []
    for k in range(escape_idx, min(escape_idx + 5, total_steps)):
        s = steps[k]
        post_escape.append({
            "step": s.get("step_idx"),
            "action": s.get("action_type"),
            "url": s.get("obs_url", ""),
            "thought_snippet": s.get("action", {}).get("thought", "")[:100],
        })

    # Classify
    if escape_action == "scroll":
        escape_type = "scroll"
    elif escape_action == "go_back":
        escape_type = "back"
    elif escape_action == "type":
        escape_type = "type"
    elif escape_action == "finish" or escape_action == "stop":
        escape_type = "finish"
    elif escape_action == "click":
        # Click that changed URL = navigated away
        if not is_post_page(escape_url) or normalize_post_url(escape_url) != post_url:
            escape_type = "navigate_away"
        else:
            escape_type = "other_click"
    elif escape_action == "goto":
        escape_type = "goto"
    else:
        escape_type = f"other:{escape_action}"

    return {
        "escape_type": escape_type,
        "escape_step": escape_idx,
        "escape_action": escape_action,
        "post_escape_actions": post_escape,
    }


def check_scroll_outcome(steps: list[dict], escape_idx: int) -> dict:
    """For scroll escapes, analyze what happened after scrolling."""
    result = {
        "saw_comment_area": False,
        "next_action_after_scroll": None,
        "next_actions_sequence": [],
    }

    # Check DOM after scroll for comment-related elements
    if escape_idx + 1 < len(steps):
        next_step = steps[escape_idx + 1]
        result["next_action_after_scroll"] = next_step.get("action_type", "")

        # Check thought for comment-related keywords
        thought = next_step.get("action", {}).get("thought", "")
        if any(kw in thought.lower() for kw in ["comment", "textarea", "input", "reply", "type"]):
            result["saw_comment_area"] = True

    # Collect next 5 actions after scroll
    for k in range(escape_idx + 1, min(escape_idx + 6, len(steps))):
        s = steps[k]
        result["next_actions_sequence"].append(s.get("action_type", ""))

    return result


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_all():
    all_results = {}  # model -> list of per-task results

    for model_label, run_dir in RUNS.items():
        ep_dir = run_dir / "episodes"
        results = []

        # Find all step files
        step_files = sorted(ep_dir.glob("reddit_task_*_steps_v2.jsonl"))

        for sf in step_files:
            # Extract task ID
            m = re.search(r"reddit_task_(\d+)_steps", sf.name)
            if not m:
                continue
            task_id = int(m.group(1))

            steps = read_steps(sf)
            if not steps:
                continue

            # Try to read summary
            summary_path = sf.parent / sf.name.replace("_steps_v2.jsonl", "_summary_v2.json")
            summary = {}
            if summary_path.exists():
                try:
                    summary = read_summary(summary_path)
                except Exception:
                    pass

            final_score = summary.get("score", steps[-1].get("reward", 0.0))
            agent_finished = summary.get("agent_finished", False)
            total_steps = len(steps)

            # Detect cycles
            cycles = detect_cycles(steps)
            if not cycles:
                continue

            for ci, cycle in enumerate(cycles):
                escape = classify_escape(steps, cycle)

                scroll_detail = None
                if escape["escape_type"] == "scroll":
                    scroll_detail = check_scroll_outcome(steps, escape["escape_step"])

                results.append({
                    "task_id": task_id,
                    "model": model_label,
                    "cycle_idx": ci,
                    "cycle_start": cycle["cycle_start"],
                    "cycle_end": cycle["cycle_end"],
                    "cycle_length": cycle["cycle_length"],
                    "post_url": cycle["post_url"],
                    "escape_type": escape["escape_type"],
                    "escape_step": escape["escape_step"],
                    "escape_action": escape["escape_action"],
                    "post_escape_actions": escape["post_escape_actions"],
                    "scroll_detail": scroll_detail,
                    "final_score": final_score,
                    "agent_finished": agent_finished,
                    "total_steps": total_steps,
                })

        all_results[model_label] = results

    return all_results


def print_results(all_results: dict):
    """Print formatted analysis tables."""

    # -----------------------------------------------------------------------
    # Table 1: Escape type summary
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("表1：循环逃出方式统计")
    print("=" * 80)
    print()

    # Count unique tasks per escape type (take first/primary cycle per task)
    escape_counts = {}
    for model in ["B1", "B0"]:
        counts = defaultdict(int)
        # Group by task_id, take first cycle
        seen_tasks = set()
        for r in all_results.get(model, []):
            if r["task_id"] in seen_tasks:
                continue
            seen_tasks.add(r["task_id"])
            counts[r["escape_type"]] += 1
        escape_counts[model] = counts

    # Collect all escape types
    all_types = sorted(set(
        list(escape_counts.get("B0", {}).keys()) +
        list(escape_counts.get("B1", {}).keys())
    ))

    # Print table
    print(f"{'逃出方式':<20} {'B1 (4B) 数量':>14} {'B0 (235B) 数量':>15} {'B1 task_ids':<30} {'B0 task_ids':<30}")
    print("-" * 110)

    # Build task_id lists per escape type
    task_ids_by_type = {model: defaultdict(list) for model in ["B0", "B1"]}
    for model in ["B0", "B1"]:
        seen = set()
        for r in all_results.get(model, []):
            if r["task_id"] in seen:
                continue
            seen.add(r["task_id"])
            task_ids_by_type[model][r["escape_type"]].append(r["task_id"])

    for etype in all_types:
        b1_count = escape_counts.get("B1", {}).get(etype, 0)
        b0_count = escape_counts.get("B0", {}).get(etype, 0)
        b1_ids = sorted(task_ids_by_type["B1"].get(etype, []))
        b0_ids = sorted(task_ids_by_type["B0"].get(etype, []))
        b1_ids_str = ",".join(str(x) for x in b1_ids[:15])
        b0_ids_str = ",".join(str(x) for x in b0_ids[:15])
        if len(b1_ids) > 15:
            b1_ids_str += f"...+{len(b1_ids)-15}"
        if len(b0_ids) > 15:
            b0_ids_str += f"...+{len(b0_ids)-15}"
        print(f"{etype:<20} {b1_count:>14} {b0_count:>15}   {b1_ids_str:<28} {b0_ids_str:<28}")

    # Totals
    b1_total = sum(escape_counts.get("B1", {}).values())
    b0_total = sum(escape_counts.get("B0", {}).values())
    print("-" * 110)
    print(f"{'总计':<20} {b1_total:>14} {b0_total:>15}")
    print()

    # -----------------------------------------------------------------------
    # Table 2: Scroll escape details
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("表2：scroll 逃出的 task 详情")
    print("=" * 80)
    print()

    scroll_tasks = []
    for model in ["B1", "B0"]:
        seen = set()
        for r in all_results.get(model, []):
            if r["task_id"] in seen:
                continue
            seen.add(r["task_id"])
            if r["escape_type"] == "scroll":
                scroll_tasks.append(r)

    if scroll_tasks:
        print(f"{'task_id':>8} {'模型':<6} {'循环步数':>8} {'scroll步':>9} {'scroll后行为':<20} {'见评论区?':<10} {'后续动作序列':<40} {'成功?':<6} {'agent_fin':<10}")
        print("-" * 130)
        for r in sorted(scroll_tasks, key=lambda x: (x["model"], x["task_id"])):
            sd = r["scroll_detail"] or {}
            next_act = sd.get("next_action_after_scroll", "?")
            saw_comment = "是" if sd.get("saw_comment_area") else "否"
            seq = "->".join(sd.get("next_actions_sequence", []))
            success = "是" if r["final_score"] == 1.0 else "否"
            af = "是" if r["agent_finished"] else "否"
            print(f"{r['task_id']:>8} {r['model']:<6} {r['cycle_length']:>8} {r['escape_step']:>9} {next_act:<20} {saw_comment:<10} {seq:<40} {success:<6} {af:<10}")
        print()
    else:
        print("(无 scroll 逃出的 task)")
        print()

    # -----------------------------------------------------------------------
    # Table 3: Comparison summary
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("表3：对比总结")
    print("=" * 80)
    print()

    for model in ["B1", "B0"]:
        results = all_results.get(model, [])
        if not results:
            continue

        # Unique tasks
        seen = set()
        unique = []
        for r in results:
            if r["task_id"] not in seen:
                seen.add(r["task_id"])
                unique.append(r)

        total_cycle_tasks = len(unique)
        type_counts = defaultdict(int)
        type_success = defaultdict(int)
        for r in unique:
            type_counts[r["escape_type"]] += 1
            if r["final_score"] == 1.0:
                type_success[r["escape_type"]] += 1

        # Also count agent_finished success (true task completion, not url_match)
        type_agent_finished_success = defaultdict(int)
        for r in unique:
            if r["final_score"] == 1.0 and r["agent_finished"]:
                type_agent_finished_success[r["escape_type"]] += 1

        print(f"--- {model} ---")
        print(f"  存在自链接循环的 task 总数: {total_cycle_tasks}")
        for etype in sorted(type_counts.keys()):
            cnt = type_counts[etype]
            suc = type_success[etype]
            af_suc = type_agent_finished_success[etype]
            pct = suc / cnt * 100 if cnt > 0 else 0
            print(f"  {etype}: {cnt} tasks, 成功 {suc} ({pct:.1f}%), "
                  f"其中 agent_finished+成功 {af_suc}")
        overall_success = sum(1 for r in unique if r["final_score"] == 1.0)
        overall_af = sum(1 for r in unique if r["final_score"] == 1.0 and r["agent_finished"])
        print(f"  循环 task 总成功: {overall_success}/{total_cycle_tasks} "
              f"({overall_success/total_cycle_tasks*100:.1f}%), "
              f"agent_finished+成功: {overall_af}")
        print(f"  注: 未逃出(truncated)+成功 通常是 url_match 评测自动得分，agent 并未真正完成")
        print()

    # -----------------------------------------------------------------------
    # Detailed per-task breakdown (all tasks with cycles)
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("附录：所有存在循环的 task 详细信息")
    print("=" * 80)
    print()

    for model in ["B1", "B0"]:
        results = all_results.get(model, [])
        if not results:
            continue

        print(f"--- {model} ---")
        # Group by task_id
        by_task = defaultdict(list)
        for r in results:
            by_task[r["task_id"]].append(r)

        for tid in sorted(by_task.keys()):
            task_cycles = by_task[tid]
            r0 = task_cycles[0]
            print(f"  task {tid}: {len(task_cycles)} cycle(s), "
                  f"total_steps={r0['total_steps']}, score={r0['final_score']}, "
                  f"agent_finished={r0['agent_finished']}")
            for c in task_cycles:
                post_acts = [p["action"] for p in c["post_escape_actions"]]
                print(f"    cycle@step[{c['cycle_start']}-{c['cycle_end']}] "
                      f"len={c['cycle_length']} -> escape={c['escape_type']} "
                      f"at step {c['escape_step']}, "
                      f"post_url={c['post_url']}")
                if c["post_escape_actions"]:
                    thoughts = [p.get("thought_snippet", "") for p in c["post_escape_actions"]]
                    print(f"      逃出后动作: {post_acts}")
                    if c["escape_type"] == "scroll" and c.get("scroll_detail"):
                        sd = c["scroll_detail"]
                        print(f"      scroll后: saw_comment={sd['saw_comment_area']}, "
                              f"next={sd['next_action_after_scroll']}, "
                              f"seq={sd['next_actions_sequence']}")
                    # Print first escape thought
                    if thoughts and thoughts[0]:
                        print(f"      逃出时thought: {thoughts[0]}")
        print()


if __name__ == "__main__":
    all_results = analyze_all()
    print_results(all_results)
