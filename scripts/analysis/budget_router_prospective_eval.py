#!/usr/bin/env python3
"""Prospective (pre-registered) test of the pre-flight budget router on shop_B1 arms that had not
landed when the predictions were frozen (2026-09-09, 笔记 §505.21–24, §505.27 item #0).

FREEZE (run once, before the held-out arms are read):
    python scripts/analysis/budget_router_prospective_eval.py freeze
  -> docs/checkpoints/pre_run/budget_router_prospective_shop_B1_20260909.json
     per-task difficulty score + tier assignment for the two pre-declared policies, trained on
     shop_B1 {dom, som, vision, psom} + shop_B0 {dom, som, vision}; ptext/pprompt outcomes NEVER read.

EVALUATE (later, when B1 ptext / pprompt shopping have landed):
    python scripts/analysis/budget_router_prospective_eval.py eval --run <run_dir> [--run <run_dir>]
  Truncation is exact (per-step cost/latency from the step JSONL). Reports, for the pre-declared
  policies, SR loss / cost saved / latency saved vs (a) fixed cap at matched cost, (b) random tier
  assignment (expectation over 200 draws). PRIMARY prospective set = task ids not in
  `seen_ptext_ids` (the 171 ptext episodes already on disk at freeze time); the rest is reported
  separately and labelled non-prospective.

Pre-declared success criterion (direction only; shop_B1 has no rerun band):
  learned SR loss < fixed-cap SR loss at matched cost AND learned SR loss < random SR loss,
  for the PRIMARY policy (two-tier) on the PRIMARY set.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, math, os, re, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from p79.experiment.io_utils import read_jsonl_dedup  # noqa: E402

FREEZE = REPO / "docs/checkpoints/pre_run/budget_router_prospective_shop_B1_20260909.json"
TRAIN = {  # condition -> run dir (largest landed run at freeze time)
    "shop_B1|dom": "results/visualwebarena/phase1/B1_dom_shopping_20260809",
    "shop_B1|som": "results/visualwebarena/phase1/B1_som_shopping_20260812",
    "shop_B1|vision": "results/visualwebarena/phase1/B1_vision_shopping_20260906",
    "shop_B1|psom": "results/visualwebarena/phase1/B1_phantom_som_shopping_20260814",
    "shop_B0|dom": "results/visualwebarena/phase1/B0_dom_shopping_20260804_003607_264370398_3845634_R3561",
    "shop_B0|som": "results/visualwebarena/phase1/B0_som_shopping_20260806_113115_297007393_109097_R12449",
    "shop_B0|vision": "results/visualwebarena/phase1/B0_vision_shopping_20260807_191852_632106648_362979_R23934",
}
FEATURE_RUN = TRAIN["shop_B1|dom"]   # step-0 page stats come from the dom arm's first observation (same page for every mode)
TASK_CFG = "results/visualwebarena/phase1/B0_dom_shopping_20260804_003607_264370398_3845634_R3561/task_configs"
KW = ("cheapest", "most expensive", "most recent", "how many", "color", "image", "picture", "subscribe", "comment", "post", "price", "review")
POLICIES = {  # pre-declared
    "two_tier": {"tiers": [(0.5, 5)], "full": 30},                  # bottom 50% by score -> cap 5
    "three_tier": {"tiers": [(0.2, 0), (0.3, 8)], "full": 30},      # bottom 20% -> 0 (abstain), next 30% -> cap 8
}

def _episodes(run_dir: str):
    out = {}
    for sf in glob.glob(os.path.join(run_dir, "*", "episodes", "*_summary_v2.json")):
        try: sj = json.load(open(sf))
        except Exception: continue
        tid = sj.get("task_id")
        if tid is None: continue
        out[int(tid)] = {"success": bool(sj.get("success")), "steps": sj.get("steps"), "summary": sf}
    return out

def _step0(run_dir: str):
    feats = {}
    for sp in glob.glob(os.path.join(run_dir, "*", "episodes", "*_steps_v2.jsonl")):
        tid = int(os.path.basename(sp).split("_task_")[1].split("_steps")[0])
        try: recs = read_jsonl_dedup(sp)
        except Exception: continue
        s0 = [r for r in recs if r.get("step_idx") == 0]
        if not s0: continue
        sd = s0[-1].get("state_digest") or {}
        feats[tid] = {"dom_complexity": sd.get("dom_complexity") or 0, "text_length": sd.get("text_length") or 0}
    return feats

def _tasks():
    meta = {}
    for f in glob.glob(os.path.join(TASK_CFG, "*.json")):
        j = json.load(open(f)); t = j.get("task_id")
        if t is None: continue
        img = j.get("image"); meta[int(t)] = {"intent": j.get("intent", "") or "", "has_image": bool(img) and str(img) != "None"}
    return meta

def features(tid, meta, s0):
    m = meta[tid]; it = m["intent"].lower(); d = s0.get(tid, {"dom_complexity": 0, "text_length": 0})
    return [d["dom_complexity"], math.log1p(d["text_length"]), float(m["has_image"]), math.log1p(len(it))] + [float(w in it) for w in KW]

def freeze():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    meta = _tasks(); s0 = _step0(FEATURE_RUN)
    tids = sorted(t for t in meta if t in s0)
    conds = {c: _episodes(REPO / r) for c, r in TRAIN.items()}
    X, y = [], []
    cond_names = list(TRAIN)
    for c, eps in conds.items():
        oh = [float(c == k) for k in cond_names]
        for t in tids:
            if t in eps: X.append(features(t, meta, s0) + oh); y.append(float(eps[t]["success"]))
    X = np.array(X); y = np.array(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X, y)
    # frozen difficulty score = mean predicted P(success) over the four landed shop_B1 conditions
    b1 = [c for c in cond_names if c.startswith("shop_B1")]
    score = {}
    for t in tids:
        ps = [clf.predict_proba(np.array([features(t, meta, s0) + [float(c == k) for k in cond_names]]))[0, 1] for c in b1]
        score[t] = float(np.mean(ps))
    order = sorted(tids, key=lambda t: score[t])
    tiers = {}
    for name, pol in POLICIES.items():
        caps = {t: pol["full"] for t in tids}; start = 0
        for frac, cap in pol["tiers"]:
            k = int(round(frac * len(tids)))
            for t in order[start:start + k]: caps[t] = cap
            start += k
        tiers[name] = caps
    seen_ptext = sorted(_episodes(REPO / "results/visualwebarena/phase1/B1_phantom_text_shopping_20260908"))
    h = hashlib.sha256()
    for c in cond_names:
        for t in sorted(conds[c]): h.update(f"{c}|{t}|{int(conds[c][t]['success'])}".encode())
    out = {"frozen_at": "2026-09-09", "protocol": __doc__, "train_conditions": TRAIN, "n_train_rows": int(len(y)),
           "train_label_sha256": h.hexdigest(), "features": ["dom_complexity", "log1p_text_length", "has_image", "log1p_intent_len"] + [f"kw:{w}" for w in KW] + [f"cond:{c}" for c in cond_names],
           "policies": POLICIES, "score": {str(t): score[t] for t in tids}, "tiers": {n: {str(t): c for t, c in caps.items()} for n, caps in tiers.items()},
           "seen_ptext_ids": seen_ptext, "n_seen_ptext": len(seen_ptext), "held_out_arms": ["shop_B1|ptext (tasks not in seen_ptext_ids)", "shop_B1|pprompt (all tasks, not started at freeze)"],
           "success_criterion": "two_tier on PRIMARY set: SR loss < fixed-cap SR loss at matched cost AND < random SR loss (direction only; no band on shop_B1)"}
    FREEZE.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"frozen {len(tids)} tasks; train rows {len(y)}; seen ptext {len(seen_ptext)}; sha256(labels)={h.hexdigest()[:16]}…")
    for n, caps in tiers.items(): print(f"  {n}: " + ", ".join(f"cap {c}: {sum(1 for v in caps.values() if v == c)}" for c in sorted(set(caps.values()))))

def _trunc(eps, caps):
    n = len(eps); sr = cost = lat = 0.0
    for e, c in zip(eps, caps):
        k = min(int(c), len(e["cost"])); cost += sum(e["cost"][:k]); lat += sum(e["lat"][:k]); sr += e["success"] and len(e["cost"]) <= c
    return sr / n * 100, cost / n, lat / n

def evaluate(run_dirs):
    fz = json.loads(FREEZE.read_text()); seen = set(fz["seen_ptext_ids"])
    for run in run_dirs:
        eps = []
        for sp in glob.glob(os.path.join(run, "*", "episodes", "*_steps_v2.jsonl")):
            tid = int(os.path.basename(sp).split("_task_")[1].split("_steps")[0]); summ = sp.replace("_steps_v2.jsonl", "_summary_v2.json")
            if not os.path.exists(summ): continue
            try: recs = read_jsonl_dedup(sp); sj = json.load(open(summ))
            except Exception: continue
            eps.append({"task": tid, "success": bool(sj.get("success")), "cost": [(r.get("cost_usd") or {}).get("total") or 0 for r in recs], "lat": [(r.get("latency_ms") or {}).get("total") or 0 for r in recs]})
        is_ptext = "phantom_text" in run
        sets = {"PRIMARY (prospective)": [e for e in eps if not (is_ptext and e["task"] in seen)]}
        if is_ptext: sets["non-prospective (seen at freeze)"] = [e for e in eps if e["task"] in seen]
        print(f"\n=== {os.path.basename(run)}: {len(eps)} episodes")
        for sname, S_ in sets.items():
            if len(S_) < 20: print(f"  {sname}: n={len(S_)} too few"); continue
            F = _trunc(S_, [10**6] * len(S_)); print(f"  {sname}: n={len(S_)}  full SR {F[0]:.1f}%  cost ${F[1]:.4f}  latency {F[2]/1000:.0f}s")
            rng = np.random.RandomState(0)
            for pname in fz["policies"]:
                caps = [fz["tiers"][pname].get(str(e["task"]), 30) for e in S_]
                sr, c, l = _trunc(S_, caps)
                fixed = min(((abs(_trunc(S_, [k] * len(S_))[1] - c), k) for k in range(3, 31)))[1]; fs, fc, fl = _trunc(S_, [fixed] * len(S_))
                rl = []
                for _ in range(200):
                    perm = rng.permutation(len(S_)); rc = [caps[i] for i in perm]; rl.append(F[0] - _trunc(S_, rc)[0])
                verdict = "PASS" if (F[0] - sr < F[0] - fs and F[0] - sr < np.mean(rl)) else "FAIL"
                print(f"    {pname:<11} learned: SR loss {F[0]-sr:+.2f}pp cost −{(1-c/F[1])*100:.0f}% latency −{(1-l/F[2])*100:.0f}% | fixed cap {fixed} @ same cost: loss {F[0]-fs:+.2f}pp | random tiers: loss {np.mean(rl):+.2f}pp  -> {verdict}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["freeze", "eval"]); ap.add_argument("--run", action="append", default=[])
    a = ap.parse_args()
    if a.cmd == "freeze":
        if FREEZE.exists(): sys.exit(f"refusing to overwrite frozen predictions: {FREEZE}")
        freeze()
    else: evaluate(a.run)
