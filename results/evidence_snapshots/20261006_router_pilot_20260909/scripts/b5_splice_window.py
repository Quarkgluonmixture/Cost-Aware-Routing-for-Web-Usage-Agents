"""B5 (GPT-5.6 terra) x classifieds: how long is the per-step splice window?

first_divergent_step = first step index where two modes' obs_url differs.
Past that point the two runs are on different pages, so any per-step splice
of one onto the other is fiction. This measures that window directly.
Mirrors scripts/analysis/mechanism_per_task.py:449.
"""
import json, glob, os, statistics as st
from itertools import combinations
from p79.experiment.io_utils import read_jsonl_dedup

RUNS = {
    "dom":      "B5_dom_classifieds_20260820_202158_076182888_2491046_R29736",
    "som":      "B5_som_classifieds_20260826_084506_787426966_3444289_R31483",
    "vision":   "B5_vision_classifieds_20260827_180917_669753800_3680965_R24364",
    "ptext":    "B5_phantom_text_classifieds_20260828_131337_301148671_3809444_R4968",
    "pprompt":  "B5_phantom_prompt_classifieds_20260829_015238_489893385_3902862_R10294",
    "psom":     "B5_phantom_som_classifieds_20260829_201900_676822651_4026859_R18439",
}
BASE = "results/visualwebarena/phase1"

def load(run):
    """task_id -> (url_trajectory, success)"""
    out = {}
    conds = glob.glob(os.path.join(BASE, run, "*"))
    for c in conds:
        for sp in glob.glob(os.path.join(c, "episodes", "*_steps_v2.jsonl")):
            tid = int(os.path.basename(sp).split("_task_")[1].split("_steps")[0])
            urls = [r.get("obs_url") for r in read_jsonl_dedup(sp) if r.get("obs_url")]
            summ = sp.replace("_steps_v2.jsonl", "_summary_v2.json")
            succ = None
            if os.path.exists(summ):
                try: succ = bool(json.load(open(summ)).get("success"))
                except Exception: pass
            if urls: out[tid] = (urls, succ)
    return out

def fds(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return min(len(a), len(b)) if len(a) != len(b) else None

data = {m: load(r) for m, r in RUNS.items()}
for m, d in data.items():
    print(f"  {m:<9} {len(d):>4} episodes, median len {st.median([len(v[0]) for v in d.values()]):.0f}")

print(f"\n{'pair':<20}{'n':>5}{'med window':>12}{'<=3 step':>10}{'never div':>11}   | disagree-only")
print("-"*82)
allw, alld = [], []
for m1, m2 in combinations(RUNS, 2):
    d1, d2 = data[m1], data[m2]
    common = sorted(set(d1) & set(d2))
    if not common: continue
    ws, dis_ws = [], []
    never = 0
    for t in common:
        w = fds(d1[t][0], d2[t][0])
        if w is None:
            never += 1; w = min(len(d1[t][0]), len(d2[t][0]))
        ws.append(w)
        if d1[t][1] is not None and d2[t][1] is not None and d1[t][1] != d2[t][1]:
            dis_ws.append(w)
    le3 = sum(1 for w in ws if w <= 3) / len(ws) * 100
    dtxt = (f"n={len(dis_ws):>3} med={st.median(dis_ws):>4.1f} "
            f"<=3:{sum(1 for w in dis_ws if w<=3)/len(dis_ws)*100:>5.1f}%") if dis_ws else "n=0"
    print(f"{m1}+{m2:<12}{len(common):>5}{st.median(ws):>12.1f}{le3:>9.1f}%{never/len(ws)*100:>10.1f}%   | {dtxt}")
    allw += ws; alld += dis_ws

print("-"*82)
print(f"ALL PAIRS   n={len(allw)}  median window={st.median(allw):.1f}  <=3 step={sum(1 for w in allw if w<=3)/len(allw)*100:.1f}%")
if alld:
    print(f"DISAGREE    n={len(alld)}  median window={st.median(alld):.1f}  <=3 step={sum(1 for w in alld if w<=3)/len(alld)*100:.1f}%")
