#!/usr/bin/env python3
"""B0 grounding-failure audit (pre-Fire-6 investigation, 2026-05-21).

Two questions the smoke (N=1 kayak task) could not answer:
  A. PAGE LANDSCAPE  — how prevalent are search/filter pages with anonymous
     (a11y-name='') input elements across cls/red tasks?
  B-coarse CORRELATION — does B0's invalid/parse-error rate track those page
     features (mode-2 obs-grounding), or is it uniform across page types
     (mode-1 structural tool-schema)? Mined from a required-era B0 run's
     step records joined with the step's observation_dom.txt.

Pure static + retrospective: no model calls. Run on DGX (data is local).
"""
import json
import glob
import os
import re
import sys
from collections import defaultdict

# AXTree element line: [N] role 'name' ...   (name may be '...' or "...")
_ELEM_RE = re.compile(r"\[(\d+)\]\s+(\w+)\s+(['\"])(.*?)\3")
_INPUT_ROLES = {"textbox", "combobox", "searchbox", "spinbutton"}
_SEARCH_RE = re.compile(
    r"search|filter|refine|keyword|sort\s*by|price|categor|\bcity\b|location|"
    r"\bmin\b|\bmax\b|\bzip\b|postal|radius|sort",
    re.IGNORECASE,
)


def parse_obs(path):
    """Return per-observation features from one observation_dom.txt."""
    n_input = n_anon = n_labeled = n_anon_with_label = 0
    has_search_ctx = False
    prev_role = prev_name = None
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return None
    for ln in lines:
        m = _ELEM_RE.search(ln)
        if not m:
            continue
        role, name = m.group(2), m.group(4)
        if _SEARCH_RE.search(name):
            has_search_ctx = True
        if role in _INPUT_ROLES:
            n_input += 1
            if name.strip() == "":
                n_anon += 1
                # anonymous input whose visible label is the preceding StaticText
                if prev_role == "StaticText" and prev_name and prev_name.strip():
                    n_anon_with_label += 1
            else:
                n_labeled += 1
        prev_role, prev_name = role, name
    if n_input == 0:
        risk = "none"
    elif has_search_ctx and n_anon >= 2:
        risk = "high"
    elif (has_search_ctx and n_anon == 1) or n_anon >= 3:
        risk = "medium"
    else:
        risk = "low"
    return dict(
        n_input=n_input, n_anon=n_anon, n_labeled=n_labeled,
        n_anon_with_label=n_anon_with_label, has_search_ctx=has_search_ctx,
        risk=risk,
    )


def site_of(path):
    if "reddit" in path:
        return "reddit"
    if "classifieds" in path:
        return "classifieds"
    return "other"


def task_of(path):
    m = re.search(r"(classifieds|reddit)_task_(\d+)", path)
    return f"{m.group(1)}_{m.group(2)}" if m else None


# ---------------------------------------------------------------- audit A
def audit_a(run_globs):
    obs_files = []
    for g in run_globs:
        obs_files += glob.glob(g + "/**/observation_dom.txt", recursive=True)
    print(f"[A] scanning {len(obs_files)} observation files for page landscape")
    per_task_risk = defaultdict(lambda: "none")
    risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    obs_risk_count = defaultdict(int)
    site_task = defaultdict(set)
    for f in obs_files:
        feat = parse_obs(f)
        if feat is None:
            continue
        obs_risk_count[feat["risk"]] += 1
        t = task_of(f)
        s = site_of(f)
        if t:
            site_task[s].add(t)
            if risk_order[feat["risk"]] > risk_order[per_task_risk[(s, t)]]:
                per_task_risk[(s, t)] = feat["risk"]
    print(f"[A] per-OBSERVATION risk distribution (n={sum(obs_risk_count.values())}):")
    for r in ("high", "medium", "low", "none"):
        c = obs_risk_count[r]
        tot = sum(obs_risk_count.values())
        print(f"      {r:7s}: {c:5d}  ({100*c/max(tot,1):4.1f}%)")
    for s in ("classifieds", "reddit"):
        tasks = sorted(site_task[s])
        if not tasks:
            continue
        rc = defaultdict(int)
        for t in tasks:
            rc[per_task_risk[(s, t)]] += 1
        risky = rc["high"] + rc["medium"]
        print(f"[A] {s}: {len(tasks)} unique tasks observed | "
              f"risky(high+med)={risky} ({100*risky/len(tasks):.0f}%) "
              f"| high={rc['high']} med={rc['medium']} low={rc['low']} none={rc['none']}")
    return per_task_risk


# ---------------------------------------------------------------- audit B-coarse
def audit_b_coarse(run_dir):
    cond = os.path.join(run_dir, "phase1_dom_router_0")
    eps = glob.glob(os.path.join(cond, "episodes", "*_steps_v2.jsonl"))
    art = os.path.join(cond, "artifacts")
    print(f"\n[B-coarse] required-era B0 run: {os.path.basename(run_dir)}")
    print(f"[B-coarse] {len(eps)} task episodes")
    # buckets: (page-type) -> [n_invalid, n_total]
    by_search = {True: [0, 0], False: [0, 0]}
    by_anon = {True: [0, 0], False: [0, 0]}
    by_risk = defaultdict(lambda: [0, 0])
    n_steps = n_invalid = n_obs_missing = 0
    for ep in eps:
        # artifact subdir matches the episode basename sans "_steps_v2.jsonl"
        # (e.g. "classifieds_task_5"), NOT the reformatted task_of() slug.
        task_dir = os.path.basename(ep).replace("_steps_v2.jsonl", "")
        for line in open(ep):
            d = json.loads(line)
            si = d.get("step_idx")
            pv = d.get("parse_valid")
            ec = d.get("error_category")
            invalid = (pv is False) or (ec in ("invalid_element_id", "invalid_action_type",
                                                "invalid_coord", "parse_error", "parse_failed"))
            n_steps += 1
            if invalid:
                n_invalid += 1
            obs = os.path.join(art, task_dir, f"step_{si:03d}", "observation_dom.txt")
            feat = parse_obs(obs) if os.path.exists(obs) else None
            if feat is None:
                n_obs_missing += 1
                continue
            for bucket, key in ((by_search, feat["has_search_ctx"]),
                                (by_anon, feat["n_anon"] > 0)):
                bucket[key][1] += 1
                if invalid:
                    bucket[key][0] += 1
            by_risk[feat["risk"]][1] += 1
            if invalid:
                by_risk[feat["risk"]][0] += 1
    print(f"[B-coarse] {n_steps} steps | {n_invalid} invalid "
          f"({100*n_invalid/max(n_steps,1):.1f}%) | obs missing for {n_obs_missing} steps")

    def rate(b):
        return f"{b[0]}/{b[1]} ({100*b[0]/max(b[1],1):.1f}%)"
    print("[B-coarse] invalid rate by page feature:")
    print(f"      has_search_context=True : {rate(by_search[True])}")
    print(f"      has_search_context=False: {rate(by_search[False])}")
    print(f"      has_anon_input=True     : {rate(by_anon[True])}")
    print(f"      has_anon_input=False    : {rate(by_anon[False])}")
    print("[B-coarse] invalid rate by page risk:")
    for r in ("high", "medium", "low", "none"):
        if by_risk[r][1]:
            print(f"      {r:7s}: {rate(by_risk[r])}")


if __name__ == "__main__":
    base = "results/visualwebarena/phase1"
    # Audit A: broad page landscape across the biggest cls + red runs (protocol-independent)
    a_globs = [
        f"{base}/B0_dom_classifieds_20260519_174455_061534393_281104_R12090",
        f"{base}/B0_dom_classifieds_20260518_212838_723280241_199081_R19740",
        f"{base}/B0_dom_classifieds_20260520_133223_291174117_410560_R2987",
        f"{base}/B1_dom_classifieds_20260519_131413_542244188_257478_R30754",
        f"{base}/B0_dom_reddit_20260519_005439_733960924_218912_R1821",
    ]
    audit_a(a_globs)
    # Audit B-coarse: required-era B0 run with both step records + observations
    audit_b_coarse(f"{base}/B0_dom_classifieds_20260520_133223_291174117_410560_R2987")
