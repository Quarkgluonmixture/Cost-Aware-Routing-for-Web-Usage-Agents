#!/usr/bin/env python3
"""Per-run diagnostic; not part of the 4-dimension evidence framework.

Supplementary analysis: How do tasks with self-link loops still succeed?
And compute B1-B0 correlation in loop behavior."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from p79.experiment.io_utils import read_jsonl_dedup

B1_RUN = ROOT / "results/visualwebarena/phase1/B1_3mode_reddit_20260413"
B0_RUN = ROOT / "results/visualwebarena/phase1/B0_3mode_reddit_20260422"
TASK_CONFIGS_DIR = B1_RUN / "task_configs"
B1_DOM = B1_RUN / "phase1_dom_router_0/episodes"
B0_DOM = B0_RUN / "phase1_dom_router_0/episodes"

POST_URL_RE = re.compile(r"/f/\w+/\d+")
COMMENT_KW = re.compile(r"\bcomment(?:s|ed|ing)?\b|\brepl(?:y|ies|ied)\b", re.IGNORECASE)


def load_configs():
    configs = {}
    for p in sorted(TASK_CONFIGS_DIR.glob("reddit_task_*.json")):
        with open(p) as f:
            cfg = json.load(f)
        configs[cfg["task_id"]] = cfg
    return configs


def get_summary(ep_dir, tid):
    p = ep_dir / f"reddit_task_{tid}_summary_v2.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def analyze_loop_escape(ep_dir, tid):
    """For tasks with loops, how did they break out?"""
    steps_path = ep_dir / f"reddit_task_{tid}_steps_v2.jsonl"
    if not steps_path.exists():
        return None
    steps = read_jsonl_dedup(steps_path)

    # Find sequences of same-URL clicks on post pages
    on_post = False
    prev_url = None
    consec = 0
    escape_actions = []
    in_loop = False

    for i, step in enumerate(steps):
        obs_url = step.get("obs_url", "").rstrip("/").split("#")[0].split("?")[0]
        action_type = step.get("action_type", "")

        if POST_URL_RE.search(obs_url):
            if action_type == "click" and prev_url and obs_url == prev_url:
                consec += 1
                in_loop = consec >= 2
            else:
                if in_loop:
                    # Escaped from loop!
                    escape_actions.append({
                        "step": i,
                        "action_type": action_type,
                        "loop_len": consec,
                        "obs_url": obs_url,
                    })
                    in_loop = False
                consec = 1 if action_type == "click" else 0
        else:
            if in_loop:
                escape_actions.append({
                    "step": i,
                    "action_type": action_type,
                    "loop_len": consec,
                    "obs_url": obs_url,
                })
                in_loop = False
            consec = 0

        prev_url = obs_url

    return escape_actions


def main():
    configs = load_configs()
    all_tids = sorted(configs.keys())

    # ── Part A: How do looping tasks succeed? ────────────────────────
    print("=" * 100)
    print("Part A: Tasks with Self-Link Loops that STILL SUCCEEDED")
    print("=" * 100)

    for label, ep_dir in [("B1 DOM", B1_DOM), ("B0 DOM", B0_DOM)]:
        print(f"\n  [{label}]")
        for tid in all_tids:
            summary = get_summary(ep_dir, tid)
            if not summary or not summary.get("success"):
                continue

            # Check if this task has loops
            steps_path = ep_dir / f"reddit_task_{tid}_steps_v2.jsonl"
            if not steps_path.exists():
                continue
            steps = read_jsonl_dedup(steps_path)

            prev_url = None
            max_consec = 0
            consec = 0
            for step in steps:
                obs_url = step.get("obs_url", "").rstrip("/").split("#")[0].split("?")[0]
                action_type = step.get("action_type", "")
                if POST_URL_RE.search(obs_url) and action_type == "click":
                    if prev_url and obs_url == prev_url:
                        consec += 1
                    else:
                        consec = 1
                    max_consec = max(max_consec, consec)
                else:
                    consec = 0
                prev_url = obs_url

            if max_consec < 2:
                continue

            cfg = configs[tid]
            eval_types = cfg.get("eval", {}).get("eval_types", [])
            intent = cfg.get("intent", "")[:70]

            escapes = analyze_loop_escape(ep_dir, tid)
            escape_desc = "; ".join(
                f"step {e['step']}: {e['action_type']} (after {e['loop_len']} clicks)"
                for e in (escapes or [])
            ) or "never escaped (url_match?)"

            print(f"    task {tid:>3d} [eval={eval_types}] loop={max_consec} | {intent}")
            print(f"             escape: {escape_desc}")

    # ── Part B: B1-B0 correlation ────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("Part B: B1 vs B0 Loop Correlation")
    print("=" * 100)

    both_loop = 0
    b1_only = 0
    b0_only = 0
    neither = 0

    for tid in all_tids:
        b1_has = False
        b0_has = False

        for ep_dir, flag_name in [(B1_DOM, "b1"), (B0_DOM, "b0")]:
            steps_path = ep_dir / f"reddit_task_{tid}_steps_v2.jsonl"
            if not steps_path.exists():
                continue
            steps = read_jsonl_dedup(steps_path)
            prev_url = None
            consec = 0
            max_c = 0
            for step in steps:
                obs_url = step.get("obs_url", "").rstrip("/").split("#")[0].split("?")[0]
                action_type = step.get("action_type", "")
                if POST_URL_RE.search(obs_url) and action_type == "click":
                    if prev_url and obs_url == prev_url:
                        consec += 1
                    else:
                        consec = 1
                    max_c = max(max_c, consec)
                else:
                    consec = 0
                prev_url = obs_url

            if flag_name == "b1":
                b1_has = max_c >= 2
            else:
                b0_has = max_c >= 2

        if b1_has and b0_has:
            both_loop += 1
        elif b1_has:
            b1_only += 1
        elif b0_has:
            b0_only += 1
        else:
            neither += 1

    print(f"\n  Contingency table (>= 2 consecutive same-URL clicks on post page):")
    print(f"                       B0 has loop    B0 no loop")
    print(f"    B1 has loop         {both_loop:>5d}          {b1_only:>5d}")
    print(f"    B1 no loop          {b0_only:>5d}          {neither:>5d}")
    print(f"\n  Interpretation:")
    print(f"    - {both_loop} tasks trap BOTH models → inherently loop-prone tasks")
    print(f"    - {b1_only} tasks trap only B1 → B1-specific weakness")
    print(f"    - {b0_only} tasks trap only B0 → B0-specific weakness")
    print(f"    - {neither} tasks trap neither")

    # ── Part C: Step budget impact ───────────────────────────────────
    print(f"\n{'=' * 100}")
    print("Part C: Step Budget Impact of Self-Link Loops")
    print("=" * 100)

    for label, ep_dir in [("B1 DOM", B1_DOM), ("B0 DOM", B0_DOM)]:
        total_steps_all = 0
        total_wasted_all = 0
        task_count = 0

        for tid in all_tids:
            steps_path = ep_dir / f"reddit_task_{tid}_steps_v2.jsonl"
            if not steps_path.exists():
                continue
            steps = read_jsonl_dedup(steps_path)
            total_steps_all += len(steps)
            task_count += 1

            prev_url = None
            consec = 0
            wasted = 0
            for step in steps:
                obs_url = step.get("obs_url", "").rstrip("/").split("#")[0].split("?")[0]
                action_type = step.get("action_type", "")
                if POST_URL_RE.search(obs_url) and action_type == "click":
                    if prev_url and obs_url == prev_url:
                        consec += 1
                        if consec >= 2:  # 2nd+ consecutive = wasted
                            wasted += 1
                    else:
                        consec = 1
                else:
                    consec = 0
                prev_url = obs_url
            total_wasted_all += wasted

        print(f"\n  [{label}]")
        print(f"  Tasks analyzed: {task_count}")
        print(f"  Total steps across all tasks: {total_steps_all}")
        print(f"  Total wasted steps (self-link): {total_wasted_all}")
        print(f"  Wasted as % of total steps: {total_wasted_all/total_steps_all*100:.1f}%")
        print(f"  Avg steps per task: {total_steps_all/task_count:.1f}")
        print(f"  If wasted steps were recovered: {(total_steps_all-total_wasted_all)/task_count:.1f} avg steps/task")

    # ── Part D: URL-match free pass analysis ─────────────────────────
    print(f"\n{'=' * 100}")
    print("Part D: Self-Link Loop Success Mechanism")
    print("=" * 100)
    print("\n  For url_match tasks, arriving at the post page = success regardless of loop.")
    print("  The 'N comments' self-link doesn't change the URL, so the agent stays on")
    print("  the correct page and gets credit for url_match.\n")

    for label, ep_dir in [("B1 DOM", B1_DOM), ("B0 DOM", B0_DOM)]:
        loop_success_url = 0
        loop_success_other = 0
        loop_fail = 0

        for tid in all_tids:
            steps_path = ep_dir / f"reddit_task_{tid}_steps_v2.jsonl"
            if not steps_path.exists():
                continue
            steps = read_jsonl_dedup(steps_path)
            prev_url = None
            consec = 0
            max_c = 0
            for step in steps:
                obs_url = step.get("obs_url", "").rstrip("/").split("#")[0].split("?")[0]
                action_type = step.get("action_type", "")
                if POST_URL_RE.search(obs_url) and action_type == "click":
                    if prev_url and obs_url == prev_url:
                        consec += 1
                    else:
                        consec = 1
                    max_c = max(max_c, consec)
                else:
                    consec = 0
                prev_url = obs_url

            if max_c < 2:
                continue

            summary = get_summary(ep_dir, tid)
            cfg = configs[tid]
            eval_types = cfg.get("eval", {}).get("eval_types", [])
            if summary and summary.get("success"):
                if "url_match" in eval_types:
                    loop_success_url += 1
                else:
                    loop_success_other += 1
            else:
                loop_fail += 1

        print(f"  [{label}]")
        print(f"    Loop + success via url_match:  {loop_success_url}")
        print(f"    Loop + success via other eval: {loop_success_other}")
        print(f"    Loop + fail:                   {loop_fail}")
        total_loop = loop_success_url + loop_success_other + loop_fail
        if total_loop:
            print(f"    url_match accounts for {loop_success_url}/{loop_success_url+loop_success_other} "
                  f"({loop_success_url/(loop_success_url+loop_success_other)*100:.0f}%) of loop successes")
        print()


if __name__ == "__main__":
    main()
