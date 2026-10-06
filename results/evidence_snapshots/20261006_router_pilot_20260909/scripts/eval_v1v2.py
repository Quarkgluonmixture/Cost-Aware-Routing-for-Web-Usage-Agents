"""v1 (intent-only zero-shot) vs v2 (all labels + failure profiles + few-shot), one cell."""
import json, os, collections, random, statistics as st
SP=os.path.dirname(os.path.abspath(__file__))
MODES=["dom","som","vision","ptext","pprompt","psom"]
full=json.load(open(os.path.join(SP,"full_table.json")))
CELL="cls_B0"
succ={m:{int(t):v[0] for t,v in full[f"{CELL}|{m}"].items()} for m in MODES}
tasks=sorted(set.intersection(*[set(v) for v in succ.values()]))
n0=len(tasks)
sr={m: sum(succ[m][t] for t in tasks)/n0*100 for m in MODES}
bm=max(sr,key=sr.get)
orac=sum(1 for t in tasks if any(succ[m][t] for m in MODES))/n0*100
rng=random.Random(0)

VARIANTS=[("v1 zero-shot, intent only","router_classifieds_luna.json"),
          ("v1 zero-shot, intent only (terra)","router_classifieds_terra.json"),
          ("v2 all labels + few-shot","v2_clsB0_full.json"),
          ("v2 all labels, no base rate","v2_clsB0_nobr.json")]
print(f"cell={CELL}  n={n0}  best-single={bm} {sr[bm]:.2f}%  oracle={orac:.2f}%  "
      f"rerun band 4.46-7.59pp")
print(f"\n{'router':<36}{'n':>5}{'SR':>8}{'Δbest':>8}{'Δtop':>8}{'Δrand':>8}{'top mode':>20}")
print("-"*93)
for name,fn in VARIANTS:
    p=os.path.join(SP,fn)
    if not os.path.exists(p): print(f"{name:<36}  (pending)"); continue
    ch={int(k):v for k,v in json.load(open(p))["choices"].items()}
    use=sorted(t for t in tasks if ch.get(t) in MODES)
    if not use: print(f"{name:<36}  (no parsable)"); continue
    n=len(use)
    llm=sum(succ[ch[t]][t] for t in use)/n*100
    s2={m: sum(succ[m][t] for t in use)/n*100 for m in MODES}
    top=collections.Counter(ch[t] for t in use).most_common(1)[0]
    rnd=st.mean([sum(succ[rng.choice(MODES)][t] for t in use)/n*100 for _ in range(200)])
    print(f"{name:<36}{n:>5}{llm:>7.2f}%{llm-s2[bm]:>+8.2f}{llm-s2[top[0]]:>+8.2f}"
          f"{llm-rnd:>+8.2f}   {top[0]} {top[1]/n*100:.0f}%")
print("-"*93)
print("Δbest = vs the single best fixed mode | Δtop = vs always-<router's own most-chosen mode>")
print("Δtop > 0 is the only thing that shows per-task discrimination bought anything.")
