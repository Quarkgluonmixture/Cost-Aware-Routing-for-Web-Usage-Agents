#!/usr/bin/env python3
"""Per-run diagnostic; not part of the 4-layer evidence framework.

Analyze "comment self-link loop" pattern in VWA Reddit tasks.

Postmill's post pages have "N comments" links that point back to the same page.
Agents frequently click these links repeatedly, trying to "navigate to comments",
when they should just scroll down.

This script:
1. Identifies comment-related tasks from task configs
2. Detects self-link click loops in B1 and B0 DOM episodes
3. Outputs summary tables
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from p79.experiment.io_utils import read_jsonl_dedup

B1_RUN = ROOT / "results/visualwebarena/phase1/B1_3mode_reddit_20260413"
B0_RUN = ROOT / "results/visualwebarena/phase1/B0_3mode_reddit_20260422"

TASK_CONFIGS_DIR = B1_RUN / "task_configs"
B1_DOM_EPISODES = B1_RUN / "phase1_dom_router_0/episodes"
B0_DOM_EPISODES = B0_RUN / "phase1_dom_router_0/episodes"

# Also check SOM and Vision for cross-ref
B1_SOM_EPISODES = B1_RUN / "phase1_som_router_0/episodes"
B1_VIS_EPISODES = B1_RUN / "phase1_vision_router_0/episodes"
B0_SOM_EPISODES = B0_RUN / "phase1_som_router_0/episodes"
B0_VIS_EPISODES = B0_RUN / "phase1_vision_router_0/episodes"

# ── Postmill URL pattern ───────────────────────────────────────────────
# /f/<forum>/<post_id>[/<slug>]
POST_URL_RE = re.compile(r"/f/\w+/\d+")

# ── comment-related keyword matching ──────────────────────────────────
COMMENT_KEYWORDS = re.compile(
    r"\bcomment(?:s|ed|ing)?\b|\brepl(?:y|ies|ied)\b",
    re.IGNORECASE,
)


def is_comment_related_task(config: dict) -> bool:
    """Check if a task's intent is comment-related."""
    intent = config.get("intent", "")
    template = config.get("intent_template", "")
    combined = f"{intent} {template}"
    return bool(COMMENT_KEYWORDS.search(combined))


def load_task_configs() -> dict:
    """Load all reddit task configs, return {task_id: config}."""
    configs = {}
    for p in sorted(TASK_CONFIGS_DIR.glob("reddit_task_*.json")):
        with open(p) as f:
            cfg = json.load(f)
        configs[cfg["task_id"]] = cfg
    return configs


def get_summary(episodes_dir: Path, task_id: int) -> dict | None:
    """Load summary for a task."""
    p = episodes_dir / f"reddit_task_{task_id}_summary_v2.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def analyze_episode(episodes_dir: Path, task_id: int) -> dict:
    """Analyze a single episode for self-link loop patterns.

    Returns dict with:
        - reached_post: bool
        - first_post_step: int or None
        - post_url: str or None
        - selflink_clicks: int  (total clicks where URL didn't change on a post page)
        - max_consecutive_selflink: int
        - comment_keyword_clicks: int (selflink clicks where thought mentions comment)
        - scrolled_on_post: bool
        - scroll_steps_on_post: int
        - total_steps: int
        - wasted_steps: int (selflink clicks that are clearly wasted)
        - success: bool or None
    """
    result = {
        "reached_post": False,
        "first_post_step": None,
        "post_url": None,
        "selflink_clicks": 0,
        "max_consecutive_selflink": 0,
        "comment_keyword_clicks": 0,
        "scrolled_on_post": False,
        "scroll_steps_on_post": 0,
        "total_steps": 0,
        "wasted_steps": 0,
        "success": None,
        "selflink_sequences": [],  # list of (start_step, length)
    }

    steps_path = episodes_dir / f"reddit_task_{task_id}_steps_v2.jsonl"
    if not steps_path.exists():
        return result

    steps = read_jsonl_dedup(steps_path)
    result["total_steps"] = len(steps)

    # Get success from summary
    summary = get_summary(episodes_dir, task_id)
    if summary:
        result["success"] = summary.get("success", None)

    # Track state
    on_post_page = False
    current_post_url = None
    consecutive_selflink = 0

    for i, step in enumerate(steps):
        obs_url = step.get("obs_url", "")
        action_type = step.get("action_type", "")
        action = step.get("action", {})
        thought = ""
        if isinstance(action, dict):
            thought = action.get("thought", "")
        elif isinstance(action, str):
            thought = action

        # Check if we're on a post page
        if POST_URL_RE.search(obs_url):
            if not on_post_page:
                on_post_page = True
                if not result["reached_post"]:
                    result["reached_post"] = True
                    result["first_post_step"] = i
                    result["post_url"] = obs_url

            # Normalize URL for comparison (strip trailing slash, fragment)
            norm_url = obs_url.rstrip("/").split("#")[0].split("?")[0]
            if current_post_url is None:
                current_post_url = norm_url

            # Check for self-link click: click action where we stay on same post URL
            if action_type == "click":
                # Check if URL is essentially the same post page
                # Postmill URLs: /f/forum/id or /f/forum/id/slug
                # The key part is /f/forum/id
                prev_base = re.search(r"/f/\w+/\d+", current_post_url)
                curr_base = re.search(r"/f/\w+/\d+", norm_url)

                if prev_base and curr_base and prev_base.group() == curr_base.group():
                    # Same post page - this is a self-link click
                    # But only count if thought mentions comment/reply
                    if COMMENT_KEYWORDS.search(thought):
                        result["selflink_clicks"] += 1
                        result["comment_keyword_clicks"] += 1
                        consecutive_selflink += 1
                        result["wasted_steps"] += 1
                    else:
                        # Still a self-link but without comment keyword
                        # Could be other navigation attempts
                        result["selflink_clicks"] += 1
                        consecutive_selflink += 1
                        result["wasted_steps"] += 1
                else:
                    # Navigated to a different page
                    if consecutive_selflink >= 2:
                        result["selflink_sequences"].append(
                            (i - consecutive_selflink, consecutive_selflink)
                        )
                    consecutive_selflink = 0
            elif action_type == "scroll":
                result["scrolled_on_post"] = True
                result["scroll_steps_on_post"] += 1
                if consecutive_selflink >= 2:
                    result["selflink_sequences"].append(
                        (i - consecutive_selflink, consecutive_selflink)
                    )
                consecutive_selflink = 0
            else:
                if consecutive_selflink >= 2:
                    result["selflink_sequences"].append(
                        (i - consecutive_selflink, consecutive_selflink)
                    )
                consecutive_selflink = 0

            current_post_url = norm_url
        else:
            if on_post_page and consecutive_selflink >= 2:
                result["selflink_sequences"].append(
                    (i - consecutive_selflink, consecutive_selflink)
                )
            on_post_page = False
            current_post_url = None
            consecutive_selflink = 0

    # Final sequence
    if consecutive_selflink >= 2:
        result["selflink_sequences"].append(
            (len(steps) - consecutive_selflink, consecutive_selflink)
        )

    result["max_consecutive_selflink"] = max(
        (seq[1] for seq in result["selflink_sequences"]), default=0
    )

    return result


def analyze_all_steps_for_selflink(episodes_dir: Path, task_id: int) -> dict:
    """More aggressive detection: any sequence of clicks on the same post URL
    where the URL doesn't change, regardless of thought content."""
    steps_path = episodes_dir / f"reddit_task_{task_id}_steps_v2.jsonl"
    if not steps_path.exists():
        return {"url_unchanged_click_sequences": [], "total_url_unchanged_clicks": 0}

    steps = read_jsonl_dedup(steps_path)

    sequences = []
    current_seq_len = 0
    current_seq_start = 0
    prev_url = None

    for i, step in enumerate(steps):
        obs_url = step.get("obs_url", "").rstrip("/").split("#")[0].split("?")[0]
        action_type = step.get("action_type", "")

        if action_type == "click" and POST_URL_RE.search(obs_url):
            if prev_url and obs_url == prev_url:
                if current_seq_len == 0:
                    current_seq_start = i - 1
                    current_seq_len = 2
                else:
                    current_seq_len += 1
            else:
                if current_seq_len >= 2:
                    sequences.append((current_seq_start, current_seq_len))
                current_seq_len = 0
        else:
            if current_seq_len >= 2:
                sequences.append((current_seq_start, current_seq_len))
            current_seq_len = 0

        if action_type in ("click", "scroll", "type", "goto"):
            prev_url = obs_url
        # For other action types, keep prev_url

    if current_seq_len >= 2:
        sequences.append((current_seq_start, current_seq_len))

    total = sum(s[1] - 1 for s in sequences)  # wasted = seq_len - 1 (first click was intentional)
    return {"url_unchanged_click_sequences": sequences, "total_url_unchanged_clicks": total}


def main():
    # ── Step 1: Load task configs and find comment-related tasks ──────
    all_configs = load_task_configs()
    comment_tasks = {}
    all_task_ids = sorted(all_configs.keys())

    for tid, cfg in sorted(all_configs.items()):
        if is_comment_related_task(cfg):
            comment_tasks[tid] = cfg

    print(f"=" * 100)
    print(f"VWA Reddit: Comment Self-Link Loop Analysis")
    print(f"=" * 100)
    print(f"\nTotal Reddit tasks: {len(all_configs)}")
    print(f"Comment-related tasks: {len(comment_tasks)}")
    print()

    # Show which tasks are comment-related
    print("Comment-related task IDs and intents:")
    for tid, cfg in sorted(comment_tasks.items()):
        intent = cfg.get("intent", "")
        # Truncate for display
        if len(intent) > 80:
            intent = intent[:77] + "..."
        eval_type = cfg["eval"]["eval_types"][0] if cfg.get("eval", {}).get("eval_types") else "?"
        print(f"  task {tid:>3d} [{eval_type:>12s}]: {intent}")
    print()

    # ── Step 2: Analyze B1 and B0 DOM episodes ───────────────────────
    print(f"{'=' * 100}")
    print(f"Table 1: Comment-Related Tasks - Self-Link Loop Detection (DOM mode)")
    print(f"{'=' * 100}")
    header = (
        f"{'task':>6s} | {'intent (truncated)':<50s} | {'post?':>5s} | "
        f"{'SL-clk':>6s} | {'max-con':>7s} | {'scroll':>6s} | "
        f"{'B1-ok':>5s} | {'B0-ok':>5s} | {'waste':>5s}"
    )
    print(header)
    print("-" * len(header))

    b1_results = {}
    b0_results = {}
    b1_all_selflink = {}
    b0_all_selflink = {}

    for tid in sorted(comment_tasks.keys()):
        cfg = comment_tasks[tid]
        intent = cfg.get("intent", "")[:50]

        b1_res = analyze_episode(B1_DOM_EPISODES, tid)
        b0_res = analyze_episode(B0_DOM_EPISODES, tid)
        b1_results[tid] = b1_res
        b0_results[tid] = b0_res

        # Also do aggressive detection
        b1_all_selflink[tid] = analyze_all_steps_for_selflink(B1_DOM_EPISODES, tid)
        b0_all_selflink[tid] = analyze_all_steps_for_selflink(B0_DOM_EPISODES, tid)

        b1_ok = "Y" if b1_res["success"] else ("N" if b1_res["success"] is not None else "?")
        b0_ok = "Y" if b0_res["success"] else ("N" if b0_res["success"] is not None else "?")
        post = "Y" if b1_res["reached_post"] else "N"
        scroll = "Y" if b1_res["scrolled_on_post"] else "N"

        print(
            f"{tid:>6d} | {intent:<50s} | {post:>5s} | "
            f"{b1_res['selflink_clicks']:>6d} | {b1_res['max_consecutive_selflink']:>7d} | {scroll:>6s} | "
            f"{b1_ok:>5s} | {b0_ok:>5s} | {b1_res['wasted_steps']:>5d}"
        )

    # ── Step 3: Summary statistics ───────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"Table 2: Summary Statistics")
    print(f"{'=' * 100}")

    def compute_stats(results, label):
        total = len(results)
        reached_post = sum(1 for r in results.values() if r["reached_post"])
        has_loop = sum(
            1 for r in results.values()
            if r["max_consecutive_selflink"] >= 2
        )
        has_loop_3plus = sum(
            1 for r in results.values()
            if r["max_consecutive_selflink"] >= 3
        )
        total_wasted = sum(r["wasted_steps"] for r in results.values())
        avg_wasted = total_wasted / has_loop if has_loop else 0
        total_selflink = sum(r["selflink_clicks"] for r in results.values())
        scrolled = sum(1 for r in results.values() if r["scrolled_on_post"])
        success = sum(1 for r in results.values() if r["success"])
        loop_and_fail = sum(
            1 for r in results.values()
            if r["max_consecutive_selflink"] >= 2 and not r["success"]
        )
        loop_and_success = sum(
            1 for r in results.values()
            if r["max_consecutive_selflink"] >= 2 and r["success"]
        )

        print(f"\n  [{label}]")
        print(f"  Comment-related tasks:               {total}")
        print(f"  Reached post page:                   {reached_post}")
        print(f"  Has self-link loop (>=2 consec.):     {has_loop} ({has_loop/total*100:.1f}%)")
        print(f"  Has self-link loop (>=3 consec.):     {has_loop_3plus} ({has_loop_3plus/total*100:.1f}%)")
        print(f"  Total self-link clicks:               {total_selflink}")
        print(f"  Total wasted steps:                   {total_wasted}")
        print(f"  Avg wasted steps (among looping):     {avg_wasted:.1f}")
        print(f"  Scrolled on post page:                {scrolled}")
        print(f"  Successful:                           {success} ({success/total*100:.1f}%)")
        print(f"  Loop + fail:                          {loop_and_fail}")
        print(f"  Loop + success:                       {loop_and_success}")

        return {
            "total": total,
            "has_loop": has_loop,
            "total_wasted": total_wasted,
            "success": success,
            "loop_and_fail": loop_and_fail,
        }

    b1_stats = compute_stats(b1_results, "B1 DOM (Qwen3-VL-4B)")
    b0_stats = compute_stats(b0_results, "B0 DOM (Qwen3-235B-A22B)")

    # ── Step 4: Extend to ALL tasks (not just comment-related) ───────
    print(f"\n{'=' * 100}")
    print(f"Table 3: Self-Link Loop Detection across ALL Reddit Tasks (not just comment-related)")
    print(f"{'=' * 100}")

    b1_all_results = {}
    b0_all_results = {}

    for tid in all_task_ids:
        b1_all_results[tid] = analyze_episode(B1_DOM_EPISODES, tid)
        b0_all_results[tid] = analyze_episode(B0_DOM_EPISODES, tid)

    all_b1_loop = {
        tid: r for tid, r in b1_all_results.items()
        if r["max_consecutive_selflink"] >= 2
    }
    all_b0_loop = {
        tid: r for tid, r in b0_all_results.items()
        if r["max_consecutive_selflink"] >= 2
    }

    print(f"\n  B1 DOM: {len(all_b1_loop)} / {len(all_task_ids)} tasks have self-link loop (>= 2 consecutive)")
    print(f"  B0 DOM: {len(all_b0_loop)} / {len(all_task_ids)} tasks have self-link loop (>= 2 consecutive)")

    # Show non-comment tasks with loops
    non_comment_b1_loop = {
        tid: r for tid, r in all_b1_loop.items() if tid not in comment_tasks
    }
    if non_comment_b1_loop:
        print(f"\n  Non-comment tasks with self-link loops in B1 DOM:")
        for tid, r in sorted(non_comment_b1_loop.items()):
            cfg = all_configs[tid]
            intent = cfg.get("intent", "")[:60]
            ok = "Y" if r["success"] else "N"
            print(f"    task {tid:>3d} (loop={r['max_consecutive_selflink']}, ok={ok}): {intent}")

    non_comment_b0_loop = {
        tid: r for tid, r in all_b0_loop.items() if tid not in comment_tasks
    }
    if non_comment_b0_loop:
        print(f"\n  Non-comment tasks with self-link loops in B0 DOM:")
        for tid, r in sorted(non_comment_b0_loop.items()):
            cfg = all_configs[tid]
            intent = cfg.get("intent", "")[:60]
            ok = "Y" if r["success"] else "N"
            print(f"    task {tid:>3d} (loop={r['max_consecutive_selflink']}, ok={ok}): {intent}")

    # ── Step 5: Detailed look at worst offenders ─────────────────────
    print(f"\n{'=' * 100}")
    print(f"Table 4: Worst Self-Link Offenders (B1 DOM, top 10 by consecutive clicks)")
    print(f"{'=' * 100}")

    worst = sorted(
        b1_all_results.items(),
        key=lambda x: x[1]["max_consecutive_selflink"],
        reverse=True,
    )[:15]

    for tid, r in worst:
        if r["max_consecutive_selflink"] < 2:
            break
        cfg = all_configs[tid]
        intent = cfg.get("intent", "")[:55]
        ok = "Y" if r["success"] else "N"
        comment_tag = "[CMT]" if tid in comment_tasks else "     "
        print(
            f"  task {tid:>3d} {comment_tag} max_consec={r['max_consecutive_selflink']:>2d} "
            f"total_SL={r['selflink_clicks']:>2d} scroll={r['scrolled_on_post']!s:>5s} "
            f"ok={ok} | {intent}"
        )

    # ── Step 6: Look at thought patterns in self-link loops ──────────
    print(f"\n{'=' * 100}")
    print(f"Table 5: Thought Patterns in Self-Link Loops (B1 DOM, sample)")
    print(f"{'=' * 100}")

    # Pick tasks with worst loops and show their thought patterns
    sample_tasks = [tid for tid, r in worst if r["max_consecutive_selflink"] >= 3][:5]

    for tid in sample_tasks:
        cfg = all_configs[tid]
        steps_path = B1_DOM_EPISODES / f"reddit_task_{tid}_steps_v2.jsonl"
        if not steps_path.exists():
            continue
        steps = read_jsonl_dedup(steps_path)
        print(f"\n  task {tid}: {cfg.get('intent', '')[:70]}")
        print(f"  {'─' * 90}")

        for i, step in enumerate(steps):
            obs_url = step.get("obs_url", "")
            action_type = step.get("action_type", "")
            action = step.get("action", {})
            thought = ""
            if isinstance(action, dict):
                thought = action.get("thought", "")
            if POST_URL_RE.search(obs_url) and action_type == "click":
                # Truncate thought
                thought_short = thought[:100] + ("..." if len(thought) > 100 else "")
                print(f"    step {i}: [{action_type}] url={obs_url[-40:]}")
                print(f"            thought: {thought_short}")

    # ── Step 7: Cross-reference with other failure patterns ──────────
    print(f"\n{'=' * 100}")
    print(f"Table 6: Cross-Reference with Other Failure Patterns")
    print(f"{'=' * 100}")

    # Check for search-over-browse pattern (agent does search instead of browsing)
    def has_search_action(episodes_dir, tid):
        """Check if agent used search/goto with search URL."""
        steps_path = episodes_dir / f"reddit_task_{tid}_steps_v2.jsonl"
        if not steps_path.exists():
            return False
        steps = read_jsonl_dedup(steps_path)
        for step in steps:
            action_type = step.get("action_type", "")
            obs_url = step.get("obs_url", "")
            action = step.get("action", {})
            if action_type == "goto":
                target = action.get("url", "") if isinstance(action, dict) else ""
                if "search" in target.lower():
                    return True
            if "search" in obs_url.lower() and "?" in obs_url:
                return True
        return False

    # Check if task is a visual task (has image reference)
    def is_visual_task(cfg):
        return bool(cfg.get("image"))

    loop_tasks_b1 = {
        tid for tid, r in b1_all_results.items()
        if r["max_consecutive_selflink"] >= 2
    }

    search_and_loop = set()
    visual_and_loop = set()
    for tid in loop_tasks_b1:
        cfg = all_configs[tid]
        if has_search_action(B1_DOM_EPISODES, tid):
            search_and_loop.add(tid)
        if is_visual_task(cfg):
            visual_and_loop.add(tid)

    print(f"\n  Tasks with self-link loop (B1 DOM):        {len(loop_tasks_b1)}")
    print(f"  Loop + search-over-browse:                  {len(search_and_loop)}")
    if search_and_loop:
        print(f"    IDs: {sorted(search_and_loop)}")
    print(f"  Loop + visual task:                         {len(visual_and_loop)}")
    if visual_and_loop:
        print(f"    IDs: {sorted(visual_and_loop)}")

    # ── Step 8: B1 vs B0 comparison for loop tasks ───────────────────
    print(f"\n{'=' * 100}")
    print(f"Table 7: B1 vs B0 Comparison for Tasks with Self-Link Loops")
    print(f"{'=' * 100}")

    union_loop = loop_tasks_b1 | {
        tid for tid, r in b0_all_results.items()
        if r["max_consecutive_selflink"] >= 2
    }

    print(f"\n  {'task':>6s} | {'B1 loop':>7s} | {'B1 waste':>8s} | {'B1 ok':>5s} | "
          f"{'B0 loop':>7s} | {'B0 waste':>8s} | {'B0 ok':>5s} | {'comment?':>8s}")
    print(f"  {'-'*70}")

    for tid in sorted(union_loop):
        b1r = b1_all_results.get(tid, {})
        b0r = b0_all_results.get(tid, {})
        b1_loop = b1r.get("max_consecutive_selflink", 0)
        b0_loop = b0r.get("max_consecutive_selflink", 0)
        b1_waste = b1r.get("wasted_steps", 0)
        b0_waste = b0r.get("wasted_steps", 0)
        b1_ok = "Y" if b1r.get("success") else "N"
        b0_ok = "Y" if b0r.get("success") else "N"
        cmt = "Y" if tid in comment_tasks else ""

        print(f"  {tid:>6d} | {b1_loop:>7d} | {b1_waste:>8d} | {b1_ok:>5s} | "
              f"{b0_loop:>7d} | {b0_waste:>8d} | {b0_ok:>5s} | {cmt:>8s}")

    # ── Step 9: url_match free-pass analysis ─────────────────────────
    print(f"\n{'=' * 100}")
    print(f"Table 8: Self-Link Loop vs Eval Type (url_match gets free pass)")
    print(f"{'=' * 100}")

    for label, results, configs in [
        ("B1 DOM", b1_all_results, all_configs),
        ("B0 DOM", b0_all_results, all_configs),
    ]:
        loop_url_match = 0
        loop_url_match_success = 0
        loop_other_eval = 0
        loop_other_eval_success = 0

        for tid, r in results.items():
            if r["max_consecutive_selflink"] < 2:
                continue
            cfg = configs[tid]
            eval_types = cfg.get("eval", {}).get("eval_types", [])
            if "url_match" in eval_types:
                loop_url_match += 1
                if r["success"]:
                    loop_url_match_success += 1
            else:
                loop_other_eval += 1
                if r["success"]:
                    loop_other_eval_success += 1

        print(f"\n  [{label}]")
        print(f"  Loop + url_match eval:    {loop_url_match} (success: {loop_url_match_success})")
        print(f"  Loop + other eval:        {loop_other_eval} (success: {loop_other_eval_success})")

    # ── Step 10: Final summary ───────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 100}")

    total_b1_loops = len(loop_tasks_b1)
    total_b1_wasted = sum(
        b1_all_results[tid]["wasted_steps"] for tid in loop_tasks_b1
    )
    avg_wasted_per_loop = total_b1_wasted / total_b1_loops if total_b1_loops else 0

    print(f"""
  1. Comment-related tasks: {len(comment_tasks)} / {len(all_configs)} Reddit tasks ({len(comment_tasks)/len(all_configs)*100:.1f}%)

  2. Self-link loop prevalence (B1 DOM, >= 2 consecutive same-URL clicks on post page):
     - {total_b1_loops} / {len(all_task_ids)} tasks ({total_b1_loops/len(all_task_ids)*100:.1f}%)
     - {total_b1_wasted} total wasted steps
     - {avg_wasted_per_loop:.1f} avg wasted steps per affected task

  3. B0 comparison:
     - {len(all_b0_loop)} / {len(all_task_ids)} tasks ({len(all_b0_loop)/len(all_task_ids)*100:.1f}%)
     - {sum(b0_all_results[tid]['wasted_steps'] for tid in all_b0_loop)} total wasted steps

  4. Comment self-link loops are a SUBSET of the broader "stuck on post page" pattern.
     Among comment-related tasks:
     - B1: {sum(1 for r in b1_results.values() if r['max_consecutive_selflink'] >= 2)} / {len(comment_tasks)} have loops
     - B0: {sum(1 for r in b0_results.values() if r['max_consecutive_selflink'] >= 2)} / {len(comment_tasks)} have loops
""")


if __name__ == "__main__":
    main()
