#!/usr/bin/env python3
"""Per-run diagnostic; not part of the 4-layer evidence framework.

Analyze "search-over-browse" bias in VWA Reddit tasks.

search-over-browse = agent uses search bar when it could have browsed
existing category/listing navigation already visible on the page.

Detection strategy (V2 -- URL-based):
  A step is "search-related" when:
    1. The obs_url (page the agent sees at that step) is on /search, OR
    2. A type action on a non-/search page leads to /search in the next step
       (i.e., the agent typed into the search box).
  This avoids false-positives from typing comments/answers into text fields.

Criteria for "search-over-browse":
  Agent performed search actions AND
    (a) start_url is /f/xxx (subreddit) -- already in the target category, OR
    (b) start_url is /forums -- could browse categories instead, OR
    (c) Agent visited a /f/xxx page BEFORE the first search action.
"""

import json
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path("/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents")

# Paths
B1_RUN = PROJECT / "results/visualwebarena/phase1/B1_3mode_reddit_20260413"
B0_RUN = PROJECT / "results/visualwebarena/phase1/B0_3mode_reddit_20260422"

B1_EPISODES = B1_RUN / "phase1_dom_router_0/episodes"
B0_EPISODES = B0_RUN / "phase1_dom_router_0/episodes"

B1_CONFIGS = B1_RUN / "task_configs"
B0_CONFIGS = B0_RUN / "task_configs"


def load_task_configs(config_dir):
    """Load all task configs, return dict keyed by task_id."""
    configs = {}
    for f in sorted(glob.glob(str(config_dir / "reddit_task_*.json"))):
        with open(f) as fh:
            cfg = json.load(fh)
        configs[cfg["task_id"]] = cfg
    return configs


def read_jsonl_safe(path):
    """Read JSONL with corrupt-line tolerance."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def read_summary(path):
    """Read summary JSON safely."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def extract_url_path(url):
    """Extract path after :9999 or after host."""
    if ":9999" in url:
        return url.split(":9999", 1)[-1]
    parts = url.split("/", 3)
    return "/" + parts[3] if len(parts) > 3 else "/"


def is_subreddit_page(url):
    path = extract_url_path(url)
    return path.startswith("/f/")


def is_forums_page(url):
    path = extract_url_path(url)
    return path.startswith("/forums")


def is_search_page(url):
    path = extract_url_path(url)
    return path.startswith("/search")


def is_homepage(url):
    path = extract_url_path(url)
    return path in ("", "/", "/all")


def classify_start_url(url):
    if "|AND|" in url:
        return "multi_site"
    path = extract_url_path(url)
    if path.startswith("/f/"):
        return "subreddit"
    if path.startswith("/search"):
        return "search"
    if path.startswith("/forums"):
        return "forums"
    if path.startswith("/comments"):
        return "comments"
    if path.startswith("/user/") or path.startswith("/u/"):
        return "user"
    if path in ("", "/", "/all"):
        return "homepage"
    return "other"


def analyze_episode(steps, start_url):
    """
    Analyze a single episode for search-over-browse behavior (V2).

    Search detection is purely URL-based:
    - Step i is "search step" if obs_url[i] is /search
    - OR action_type == 'type' at step i and obs_url[i+1] is /search
      (agent typed into search box, causing navigation to search page)
    """
    result = {
        "search_steps": [],            # step indices with search activity
        "total_steps": len(steps),
        "urls_visited": [],
        "was_on_category_before_search": False,
        "first_search_step": None,
        "search_queries": [],          # typed text that led to search
        "search_url_steps": [],        # steps where obs_url is /search
        "type_to_search_steps": [],    # type actions leading to /search
    }

    if not steps:
        return result

    # Build obs_url list
    urls = [s.get("obs_url", "") for s in steps]
    result["urls_visited"] = urls

    # Detect search steps
    search_step_set = set()

    for i, step in enumerate(steps):
        obs_url = urls[i]
        action_type = step.get("action_type", "")
        action = step.get("action", {})

        # (A) Agent is on a /search page -- this step is search-related
        if is_search_page(obs_url):
            search_step_set.add(i)
            result["search_url_steps"].append(i)

        # (B) Agent types something and the NEXT step's URL is /search
        #     (the type action submitted a search query)
        if action_type == "type" and i + 1 < len(steps):
            next_url = urls[i + 1]
            if is_search_page(next_url) and not is_search_page(obs_url):
                search_step_set.add(i)
                result["type_to_search_steps"].append(i)
                # Extract the query text
                if isinstance(action, dict):
                    text = action.get("text", "")
                    if text:
                        result["search_queries"].append(
                            text.strip().replace("\n", "").replace("\\n", "")
                        )

    result["search_steps"] = sorted(search_step_set)

    if result["search_steps"]:
        first = result["search_steps"][0]
        result["first_search_step"] = first

        # Check if agent was on a category page before first search
        visited_category_before = is_subreddit_page(start_url) or is_forums_page(start_url)
        for j in range(first):
            if is_subreddit_page(urls[j]) or is_forums_page(urls[j]):
                visited_category_before = True
                break
        result["was_on_category_before_search"] = visited_category_before

    return result


def judge_search_over_browse(task_config, b1_analysis, b0_analysis):
    """
    Determine if a task exhibits search-over-browse bias.

    Criteria:
    - Agent used search (has search_steps) AND
      - start_url is subreddit (/f/xxx), OR
      - start_url is forums page, OR
      - Agent visited a category page before searching
    """
    start_url = task_config.get("start_url", "")
    start_cat = classify_start_url(start_url)

    reasons = []

    for label, analysis in [("B1", b1_analysis), ("B0", b0_analysis)]:
        if not analysis or not analysis["search_steps"]:
            continue

        if start_cat == "subreddit":
            reasons.append(f"{label}: searched from subreddit start_url")
        elif start_cat == "forums":
            reasons.append(f"{label}: searched from forums start_url")
        elif analysis["was_on_category_before_search"]:
            reasons.append(f"{label}: visited category then searched")

    return (len(reasons) > 0, "; ".join(reasons) if reasons else "")


def main():
    print("=" * 130)
    print("VWA Reddit: Search-over-Browse Bias Analysis (V2 -- URL-based detection)")
    print("=" * 130)
    print()

    b1_configs = load_task_configs(B1_CONFIGS)
    b0_configs = load_task_configs(B0_CONFIGS)

    all_task_ids = sorted(set(list(b1_configs.keys()) + list(b0_configs.keys())))
    print(f"Total Reddit tasks found in configs: {len(all_task_ids)}")

    # Analyze each task
    results = []
    for tid in all_task_ids:
        cfg = b1_configs.get(tid) or b0_configs.get(tid)
        if not cfg:
            continue

        start_url = cfg.get("start_url", "")
        intent = cfg.get("intent", "")
        start_cat = classify_start_url(start_url)

        if start_cat == "multi_site":
            continue

        # B1
        b1_steps = read_jsonl_safe(B1_EPISODES / f"reddit_task_{tid}_steps_v2.jsonl")
        b1_summary = read_summary(B1_EPISODES / f"reddit_task_{tid}_summary_v2.json")
        b1_analysis = analyze_episode(b1_steps, start_url) if b1_steps else None
        b1_success = b1_summary.get("success", False) if b1_summary else None

        # B0
        b0_steps = read_jsonl_safe(B0_EPISODES / f"reddit_task_{tid}_steps_v2.jsonl")
        b0_summary = read_summary(B0_EPISODES / f"reddit_task_{tid}_summary_v2.json")
        b0_analysis = analyze_episode(b0_steps, start_url) if b0_steps else None
        b0_success = b0_summary.get("success", False) if b0_summary else None

        is_sob, reason = judge_search_over_browse(cfg, b1_analysis, b0_analysis)

        b1_visited_search = any(is_search_page(u) for u in (b1_analysis["urls_visited"] if b1_analysis else []))
        b0_visited_search = any(is_search_page(u) for u in (b0_analysis["urls_visited"] if b0_analysis else []))

        results.append({
            "task_id": tid,
            "intent": intent,
            "start_cat": start_cat,
            "start_url_has_f": "/f/" in start_url,
            # B1
            "b1_search_steps": len(b1_analysis["search_steps"]) if b1_analysis else 0,
            "b1_total_steps": b1_analysis["total_steps"] if b1_analysis else 0,
            "b1_first_search": b1_analysis["first_search_step"] if b1_analysis else None,
            "b1_success": b1_success,
            "b1_visited_search": b1_visited_search,
            "b1_queries": b1_analysis["search_queries"] if b1_analysis else [],
            # B0
            "b0_search_steps": len(b0_analysis["search_steps"]) if b0_analysis else 0,
            "b0_total_steps": b0_analysis["total_steps"] if b0_analysis else 0,
            "b0_first_search": b0_analysis["first_search_step"] if b0_analysis else None,
            "b0_success": b0_success,
            "b0_visited_search": b0_visited_search,
            "b0_queries": b0_analysis["search_queries"] if b0_analysis else [],
            # Judgment
            "search_over_browse": is_sob,
            "sob_reason": reason,
        })

    for r in results:
        r["any_search_b1"] = r["b1_visited_search"]
        r["any_search_b0"] = r["b0_visited_search"]
        r["any_search"] = r["b1_visited_search"] or r["b0_visited_search"]

    # =========================================================================
    # TABLE 1: Per-task analysis
    # =========================================================================
    print("=" * 130)
    print("TABLE 1: Per-task Search-over-Browse Analysis (only tasks with search activity shown)")
    print("=" * 130)
    print()

    hdr = (
        f"{'TID':>4} | {'Intent (truncated)':45} | {'Start':9} | "
        f"{'B1 Srch':>7} | {'B1 Tot':>6} | {'B1 %':>5} | {'B1 OK':>5} | "
        f"{'B0 Srch':>7} | {'B0 Tot':>6} | {'B0 %':>5} | {'B0 OK':>5} | "
        f"{'SoB':>3}"
    )
    print(hdr)
    print("-" * len(hdr))

    sob_tasks = []
    for r in results:
        # Only show tasks with any search activity or SoB
        if not r["any_search"] and not r["search_over_browse"]:
            continue

        b1_pct = (
            f"{r['b1_search_steps']/r['b1_total_steps']*100:.0f}%"
            if r["b1_total_steps"] > 0 else "N/A"
        )
        b0_pct = (
            f"{r['b0_search_steps']/r['b0_total_steps']*100:.0f}%"
            if r["b0_total_steps"] > 0 else "N/A"
        )
        b1_ok = "Y" if r["b1_success"] else ("N" if r["b1_success"] is not None else "-")
        b0_ok = "Y" if r["b0_success"] else ("N" if r["b0_success"] is not None else "-")
        sob = "Yes" if r["search_over_browse"] else ""

        intent_short = r["intent"][:45]
        row = (
            f"{r['task_id']:>4} | {intent_short:45} | {r['start_cat']:9} | "
            f"{r['b1_search_steps']:>7} | {r['b1_total_steps']:>6} | {b1_pct:>5} | {b1_ok:>5} | "
            f"{r['b0_search_steps']:>7} | {r['b0_total_steps']:>6} | {b0_pct:>5} | {b0_ok:>5} | "
            f"{sob:>3}"
        )
        print(row)

        if r["search_over_browse"]:
            sob_tasks.append(r)

    print()
    print(f"Total tasks: {len(results)}, Shown (with search): {sum(1 for r in results if r['any_search'] or r['search_over_browse'])}")
    print()

    # =========================================================================
    # TABLE 1b: SoB tasks detailed
    # =========================================================================
    print("=" * 130)
    print("TABLE 1b: Tasks with Search-over-Browse bias (detailed)")
    print("=" * 130)
    print()

    for r in sob_tasks:
        print(f"Task {r['task_id']}:")
        print(f"  Intent: {r['intent'][:120]}")
        print(f"  Start category: {r['start_cat']}")
        if r["b1_queries"]:
            print(f"  B1 search queries: {r['b1_queries'][:5]}")
        if r["b0_queries"]:
            print(f"  B0 search queries: {r['b0_queries'][:5]}")
        print(f"  B1: {r['b1_search_steps']} search / {r['b1_total_steps']} total steps, success={r['b1_success']}")
        print(f"  B0: {r['b0_search_steps']} search / {r['b0_total_steps']} total steps, success={r['b0_success']}")
        print(f"  Reason: {r['sob_reason']}")
        print()

    # =========================================================================
    # TABLE 2: Aggregate statistics
    # =========================================================================
    print("=" * 130)
    print("TABLE 2: Aggregate Statistics")
    print("=" * 130)
    print()

    total = len(results)
    sob_count = sum(1 for r in results if r["search_over_browse"])
    any_search_b1 = sum(1 for r in results if r["any_search_b1"])
    any_search_b0 = sum(1 for r in results if r["any_search_b0"])
    any_search_either = sum(1 for r in results if r["any_search"])

    print(f"--- 1. Basic Counts ---")
    print(f"Total tasks analyzed (excl. multi-site): {total}")
    print(f"Tasks where B1 visited /search URL: {any_search_b1} ({any_search_b1/total*100:.1f}%)")
    print(f"Tasks where B0 visited /search URL: {any_search_b0} ({any_search_b0/total*100:.1f}%)")
    print(f"Tasks where either visited /search: {any_search_either} ({any_search_either/total*100:.1f}%)")
    print(f"Tasks with search-over-browse bias: {sob_count} ({sob_count/total*100:.1f}%)")
    print()

    # Success rates
    def success_rate(task_list, key):
        valid = [r for r in task_list if r[key] is not None]
        if not valid:
            return 0, 0
        succ = sum(1 for r in valid if r[key])
        return succ, len(valid)

    print(f"--- 2. Success Rates: SoB vs Non-SoB ---")
    sob_list = [r for r in results if r["search_over_browse"]]
    non_sob_list = [r for r in results if not r["search_over_browse"]]

    for key, label in [("b1_success", "B1 (4B)"), ("b0_success", "B0 (235B)")]:
        s1, n1 = success_rate(sob_list, key)
        s2, n2 = success_rate(non_sob_list, key)
        r1 = f"{s1}/{n1} ({s1/n1*100:.1f}%)" if n1 > 0 else "N/A"
        r2 = f"{s2}/{n2} ({s2/n2*100:.1f}%)" if n2 > 0 else "N/A"
        print(f"  {label} SoB tasks:     {r1}")
        print(f"  {label} Non-SoB tasks: {r2}")
    print()

    print(f"--- 3. Success Rates: Any /search visit vs No /search visit ---")
    for bx, bkey in [("B1 (4B)", "any_search_b1"), ("B0 (235B)", "any_search_b0")]:
        skey = "b1_success" if "B1" in bx else "b0_success"
        search_list = [r for r in results if r[bkey]]
        no_search_list = [r for r in results if not r[bkey]]
        s1, n1 = success_rate(search_list, skey)
        s2, n2 = success_rate(no_search_list, skey)
        r1 = f"{s1}/{n1} ({s1/n1*100:.1f}%)" if n1 > 0 else "N/A"
        r2 = f"{s2}/{n2} ({s2/n2*100:.1f}%)" if n2 > 0 else "N/A"
        print(f"  {bx} tasks WITH search: {r1}")
        print(f"  {bx} tasks w/o  search: {r2}")
    print()

    print(f"--- 4. Search-over-Browse by Start URL Category ---")
    cats = sorted(set(r["start_cat"] for r in results))
    print(f"  {'Category':12} | {'Total':>5} | {'SoB':>5} | {'SoB%':>5} | {'Any-Search':>10} | {'Search%':>7}")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*7}")
    for cat in cats:
        ct = [r for r in results if r["start_cat"] == cat]
        cs = [r for r in ct if r["search_over_browse"]]
        ca = [r for r in ct if r["any_search"]]
        print(
            f"  {cat:12} | {len(ct):5} | {len(cs):5} | "
            f"{len(cs)/len(ct)*100:4.0f}% | {len(ca):10} | "
            f"{len(ca)/len(ct)*100:5.0f}%"
        )
    print()

    print(f"--- 5. B0 vs B1 Search Behavior Comparison ---")
    b1_only = sum(1 for r in results if r["any_search_b1"] and not r["any_search_b0"])
    b0_only = sum(1 for r in results if r["any_search_b0"] and not r["any_search_b1"])
    both = sum(1 for r in results if r["any_search_b1"] and r["any_search_b0"])
    neither = sum(1 for r in results if not r["any_search_b1"] and not r["any_search_b0"])
    print(f"  Both B0+B1 search:  {both:3}")
    print(f"  Only B1 searches:   {b1_only:3}")
    print(f"  Only B0 searches:   {b0_only:3}")
    print(f"  Neither searches:   {neither:3}")
    print()

    print(f"--- 6. Average Search Steps (among tasks that search) ---")
    b1s = [r["b1_search_steps"] for r in results if r["b1_search_steps"] > 0]
    b0s = [r["b0_search_steps"] for r in results if r["b0_search_steps"] > 0]
    if b1s:
        print(f"  B1: mean={sum(b1s)/len(b1s):.1f}, median={sorted(b1s)[len(b1s)//2]}, max={max(b1s)}, N={len(b1s)}")
    if b0s:
        print(f"  B0: mean={sum(b0s)/len(b0s):.1f}, median={sorted(b0s)[len(b0s)//2]}, max={max(b0s)}, N={len(b0s)}")
    print()

    print(f"--- 7. Subreddit-start Tasks: Search vs No-search Success ---")
    sub_tasks = [r for r in results if r["start_cat"] == "subreddit"]
    for bx, bkey, skey in [
        ("B1", "any_search_b1", "b1_success"),
        ("B0", "any_search_b0", "b0_success"),
    ]:
        ss = [r for r in sub_tasks if r[bkey]]
        ns = [r for r in sub_tasks if not r[bkey]]
        s1, n1 = success_rate(ss, skey)
        s2, n2 = success_rate(ns, skey)
        r1 = f"{s1}/{n1} ({s1/n1*100:.1f}%)" if n1 > 0 else "N/A"
        r2 = f"{s2}/{n2} ({s2/n2*100:.1f}%)" if n2 > 0 else "N/A"
        print(f"  {bx} subreddit + search:    {r1}")
        print(f"  {bx} subreddit + no-search: {r2}")
    print()

    print(f"--- 8. First Search Step Distribution ---")
    b1f = [r["b1_first_search"] for r in results if r["b1_first_search"] is not None]
    b0f = [r["b0_first_search"] for r in results if r["b0_first_search"] is not None]
    if b1f:
        print(f"  B1: mean={sum(b1f)/len(b1f):.1f}, median={sorted(b1f)[len(b1f)//2]}, "
              f"at step 0: {sum(1 for x in b1f if x == 0)}, at step 0-1: {sum(1 for x in b1f if x <= 1)}, N={len(b1f)}")
    if b0f:
        print(f"  B0: mean={sum(b0f)/len(b0f):.1f}, median={sorted(b0f)[len(b0f)//2]}, "
              f"at step 0: {sum(1 for x in b0f if x == 0)}, at step 0-1: {sum(1 for x in b0f if x <= 1)}, N={len(b0f)}")
    print()

    # =========================================================================
    # TABLE 3: SoB by intent template pattern
    # =========================================================================
    print("=" * 130)
    print("TABLE 3: SoB Frequency by Intent Pattern")
    print("=" * 130)
    print()

    intent_patterns = defaultdict(lambda: {"total": 0, "sob": 0, "b1_succ": 0, "b0_succ": 0, "tasks": []})
    for r in results:
        # Extract first ~30 chars as pattern key
        intent = r["intent"]
        # Normalize: take first sentence or first 50 chars
        pattern = intent.split(".")[0][:50].strip()
        ip = intent_patterns[pattern]
        ip["total"] += 1
        if r["search_over_browse"]:
            ip["sob"] += 1
        if r["b1_success"]:
            ip["b1_succ"] += 1
        if r["b0_success"]:
            ip["b0_succ"] += 1
        ip["tasks"].append(r["task_id"])

    # Sort by SoB count descending
    sorted_patterns = sorted(intent_patterns.items(), key=lambda x: -x[1]["sob"])
    print(f"  {'Intent Pattern':55} | {'Total':>5} | {'SoB':>5} | {'SoB%':>5} | {'B1 SR':>6} | {'B0 SR':>6}")
    print(f"  {'-'*55}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}")
    for pattern, data in sorted_patterns:
        if data["sob"] == 0:
            continue
        b1sr = f"{data['b1_succ']/data['total']*100:.0f}%" if data['total'] > 0 else "N/A"
        b0sr = f"{data['b0_succ']/data['total']*100:.0f}%" if data['total'] > 0 else "N/A"
        print(
            f"  {pattern:55} | {data['total']:5} | {data['sob']:5} | "
            f"{data['sob']/data['total']*100:4.0f}% | {b1sr:>6} | {b0sr:>6}"
        )
    print()


if __name__ == "__main__":
    main()
