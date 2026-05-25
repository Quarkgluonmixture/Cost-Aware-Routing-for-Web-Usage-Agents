#!/usr/bin/env python3
"""Cross-mode routing failure taxonomy — task-centric, O(mode) not O(mode^2).

Avoids pairwise mode comparison (C(6,2)=15) and exclusive-subset explosion (2^6).
Single unified failure taxonomy per (task, mode); all views are aggregations.

Failure types (deterministic, from step log — no sub-agent):
  BUDGET : trajectory_incomplete (never reached a valid finish)
  NAV    : finished but never visited any "correct" listing (navigation/search miss)
  IMG    : finished AND reached a correct listing but produced wrong answer (perception)
  EARLY  : (modifier) steps < EARLY_STEPS — gave up fast
"correct listing" = union of {finish item-id of any mode that SOLVED this task}
                     + reference_url id (url_match tasks).

Usage:
  python scripts/analysis/cross_mode_failure_taxonomy.py \
      --site classifieds --model B0 \
      --run dom=<run_dir> --run som=<run_dir> --run vision=<run_dir>
"""
import argparse, json, re, glob, sys
from collections import Counter, defaultdict

EARLY_STEPS = 5
ID_RE = re.compile(r"[?&]id=(\d+)")
SCAT_RE = re.compile(r"sCategory=(\d+)")
SPAT_RE = re.compile(r"sPattern=([^&]+)")


def listsig(url):
    """List-page signature: category and/or search-pattern (prefixed to avoid clash)."""
    s = set()
    cm = SCAT_RE.search(url or "")
    if cm:
        s.add("c:" + cm.group(1))
    pm = SPAT_RE.search(url or "")
    if pm:
        s.add("p:" + pm.group(1))
    return s


def trace(run_dir, site, task):
    """Return per-episode trace dict, or None if missing."""
    hits = glob.glob(f"{run_dir}/*/episodes/{site}_task_{task}_steps_v2.jsonl")
    if not hits:
        hits = glob.glob(f"{run_dir}/episodes/{site}_task_{task}_steps_v2.jsonl")
    if not hits:
        return None
    visited, fin_id, fin_ans, n, fin_seen, urls = [], None, None, 0, False, []
    for line in open(hits[0]):
        if not line.strip():
            continue
        d = json.loads(line); n += 1
        u = d.get("obs_url", "") or ""
        urls.append(u)
        m = ID_RE.search(u)
        if m:
            visited.append(m.group(1))
        if d.get("action_type") == "finish":
            fin_seen = True
            a = d.get("action", {}) or {}
            fin_ans = str(a.get("answer", "") or a.get("text", ""))
            fm = ID_RE.search(d.get("obs_url", "") or "")
            fin_id = fm.group(1) if fm else None
    # summary for success + trajectory_incomplete
    sg = glob.glob(f"{run_dir}/*/episodes/{site}_task_{task}_summary_v2.json") or \
         glob.glob(f"{run_dir}/episodes/{site}_task_{task}_summary_v2.json")
    succ = incomplete = False
    if sg:
        s = json.load(open(sg[0]))
        succ = bool(s.get("success"))
        incomplete = bool(s.get("trajectory_incomplete"))
    return {"visited": visited, "fin_id": fin_id, "fin_ans": fin_ans, "urls": urls,
            "steps": n, "success": succ, "incomplete": incomplete, "finished": fin_seen}


def ref_id(run_dir, site, task):
    cg = glob.glob(f"{run_dir}/task_configs/{site}_task_{task}.json")
    if not cg:
        return None
    ev = (json.load(open(cg[0])).get("eval") or {})
    m = ID_RE.search(ev.get("reference_url") or "")
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--model", default="B0")
    ap.add_argument("--run", action="append", required=True,
                    help="mode=run_dir, repeatable")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        mode, _, rd = spec.partition("=")
        runs[mode] = rd.rstrip("/")
    modes = list(runs)

    # discover common task ids
    per_mode_tasks = {}
    for m, rd in runs.items():
        ids = set()
        for f in glob.glob(f"{rd}/*/episodes/{args.site}_task_*_summary_v2.json") or \
                 glob.glob(f"{rd}/episodes/{args.site}_task_*_summary_v2.json"):
            mm = re.search(rf"{args.site}_task_(\d+)_summary", f)
            if mm:
                ids.add(int(mm.group(1)))
        per_mode_tasks[m] = ids
    tasks = sorted(set.intersection(*per_mode_tasks.values()))

    tr = {}  # (task, mode) -> trace
    for t in tasks:
        for m in modes:
            tr[(t, m)] = trace(runs[m], args.site, t)

    # success matrix + task classes
    succ = {(t, m): (tr[(t, m)] or {}).get("success", False) for t in tasks for m in modes}
    cls = {}  # task -> 'universal-solve'|'universal-fail'|'routable'
    for t in tasks:
        sv = [succ[(t, m)] for m in modes]
        cls[t] = ("universal-solve" if all(sv) else
                  "universal-fail" if not any(sv) else "routable")

    # correct listing set per task
    correct = {}
    for t in tasks:
        cset = set()
        for m in modes:
            if succ[(t, m)] and tr[(t, m)] and tr[(t, m)]["fin_id"]:
                cset.add(tr[(t, m)]["fin_id"])
        r = ref_id(runs[modes[0]], args.site, t)
        if r:
            cset.add(r)
        correct[t] = cset

    # correct list-page signatures per task (NAV split). = start_url sig + list-page
    # sigs the SOLVING modes passed through before reaching the correct item. A failed
    # mode that visited one of these saw the correct listing's thumbnail (THUMBNAIL);
    # one that didn't never got there (SEARCH-NAV). If the task has NO sig at all
    # (search-only without sCategory/sPattern, ~65% of cls) the split is undecidable
    # → UNCLEAR-NAV (honest: do not pretend to know nav-vs-thumbnail).
    correct_sigs = {}
    for t in tasks:
        sig = set()
        cg = glob.glob(f"{runs[modes[0]]}/task_configs/{args.site}_task_{t}.json")
        if cg:
            sig |= listsig(json.load(open(cg[0])).get("start_url", "") or "")
        for m in modes:
            if not succ[(t, m)] or not tr[(t, m)]:
                continue
            for u in tr[(t, m)]["urls"]:
                im = ID_RE.search(u)
                if im and im.group(1) in correct[t]:
                    break
                sig |= listsig(u)
        correct_sigs[t] = sig

    # failure taxonomy per (task, mode) failed
    def ftype(t, m):
        d = tr[(t, m)]
        if d is None:
            return None
        if d["success"]:
            return None
        if d["incomplete"] or not d["finished"]:
            base = "BUDGET"
        else:
            cset = correct[t]
            if cset and (set(d["visited"]) & cset):
                base = "IMG"
            elif not correct_sigs[t]:
                base = "UNCLEAR-NAV"
            else:
                vsig = set()
                for u in d["urls"]:
                    vsig |= listsig(u)
                base = "THUMBNAIL" if (correct_sigs[t] & vsig) else "SEARCH-NAV"
        return base + ("+EARLY" if d["steps"] < EARLY_STEPS else "")

    # --- aggregations ---
    out = []
    def w(s=""):
        out.append(s)

    w(f"# Cross-mode Routing Failure Taxonomy — {args.model} {args.site}")
    w()
    w(f"> Modes: {', '.join(modes)} | common tasks N={len(tasks)} | "
      f"deterministic taxonomy (no sub-agent) | EARLY_STEPS={EARLY_STEPS}")
    w(f"> ⚠️ PROVISIONAL — {len(modes)}/6 mode, single (model,site), presence-light "
      f"(NAV/IMG from step log). NOT paper-grade.")
    w()
    w("## 0. 方法 — 为什么 task-centric 而非 pairwise")
    w()
    w("6 mode 两两比较 = C(6,2)=15 对; exclusive 子集 = 2^6 爆炸。本框架给每个 (task,mode)")
    w("失败打**统一 taxonomy 标签**, 所有视图是标签聚合 → 核心是「失败类型 × mode」矩阵")
    w("(行=失败类型固定, 列=mode), **O(mode) 不是 O(mode²)**。6 mode 时只多 3 列, 不重写")
    w("叙事。脚本: `scripts/analysis/cross_mode_failure_taxonomy.py` (确定性, 无 sub-agent)。")
    w()
    # 1. task classes
    cc = Counter(cls.values())
    w("## 1. Task classes (success matrix)")
    w()
    w("| class | N | meaning |")
    w("|---|---:|---|")
    w(f"| universal-solve | {cc['universal-solve']} | all modes solve (easy) |")
    w(f"| universal-fail | {cc['universal-fail']} | no mode solves (hard / benchmark-FP) |")
    w(f"| **routable** | {cc['routable']} | partial — **routing value lives here** |")
    w()
    # 2. failure-type × mode matrix
    ft = defaultdict(lambda: Counter())  # ftype_base -> {mode: n}
    for t in tasks:
        for m in modes:
            f = ftype(t, m)
            if f:
                ft[f.split("+")[0]][m] += 1
                if "EARLY" in f:
                    ft["(of which EARLY)"][m] += 1
    w("## 2. Failure-type × mode matrix  (← O(mode), 6-mode just adds columns)")
    w()
    w("| failure type | " + " | ".join(modes) + " | meaning |")
    w("|---|" + "---:|" * len(modes) + "---|")
    meanings = {"SEARCH-NAV": "had category/pattern sig but never reached it (nav miss, behavioral)",
                "THUMBNAIL": "reached correct list page, wrong listing picked (thumbnail recog = IMG upstream)",
                "UNCLEAR-NAV": "task has NO sig (~65% cls search/on-this-page) — nav-vs-thumbnail UNDECIDABLE",
                "IMG": "reached correct listing detail but wrong answer (perception/reasoning)",
                "BUDGET": "trajectory_incomplete (no valid finish)",
                "(of which EARLY)": f"...gave up < {EARLY_STEPS} steps"}
    for f in ["SEARCH-NAV", "THUMBNAIL", "UNCLEAR-NAV", "IMG", "BUDGET", "(of which EARLY)"]:
        row = " | ".join(str(ft[f].get(m, 0)) for m in modes)
        w(f"| {f} | {row} | {meanings.get(f,'')} |")
    w()
    w("> **NAV 三分** (列表页 sig=sCategory+sPattern 判 '是否到达正确列表页'): **SEARCH-NAV**=有 sig 但")
    w("> 没到 (导航失败, 行为层) · **THUMBNAIL**=到了正确列表页没点对缩略图 (= **图像识别上游**) ·")
    w("> **UNCLEAR-NAV**=task 无任何 sig (~65% cls 是 search/on-this-page 无 sCategory) → 当前判据**判不了**")
    w("> nav-vs-thumbnail (诚实标注不强分)。**可靠的只有 THUMBNAIL+IMG (图像识别全谱, dom>som>vision 梯度)")
    w("> + BUDGET**; SEARCH-NAV 真实但小 (~6-11); UNCLEAR-NAV 大 = **判据天花板**, 拆它需 listing-level")
    w("> observation (correct item link 出现在 agent 哪个列表页) — 要 sync DOM/som obs 文本 (next refinement)。")
    w()
    # 3. routing value: exclusive solves + how the OTHER modes failed
    w("## 3. Routing value per mode (exclusive solves + how others failed)")
    w()
    w("| mode | SR | exclusive-solve | (others' failure on those tasks) |")
    w("|---|---:|---:|---|")
    for m in modes:
        n_succ = sum(succ[(t, m)] for t in tasks)
        excl = [t for t in tasks if succ[(t, m)] and not any(succ[(t, o)] for o in modes if o != m)]
        # how others failed on this mode's exclusive tasks
        oc = Counter()
        for t in excl:
            for o in modes:
                if o != m:
                    f = ftype(t, o)
                    if f:
                        oc[f.split("+")[0]] += 1
        ocs = ", ".join(f"{k}:{v}" for k, v in oc.most_common())
        w(f"| {m} | {n_succ/len(tasks)*100:.1f}% | {len(excl)} | {ocs} |")
    w()
    w(f"full {len(modes)}-mode oracle SR = "
      f"{sum(1 for t in tasks if any(succ[(t,m)] for m in modes))/len(tasks)*100:.1f}%")
    w()
    # 4. exclusive task ids (for drill-down)
    w("## 4. Exclusive task ids (drill-down)")
    w()
    for m in modes:
        excl = [t for t in tasks if succ[(t, m)] and not any(succ[(t, o)] for o in modes if o != m)]
        w(f"- **{m}**: {excl}")

    text = "\n".join(out)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")
        print(f"\n[written] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
