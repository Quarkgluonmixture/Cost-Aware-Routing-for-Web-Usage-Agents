"""Final: v1 (intent-only) vs v3 (all labels, cost-aware) vs 3-class READ/LOOK/BOTH.

Three axes, router overhead charged to the router, effect sizes read against the
measured rerun band -- a pp difference smaller than the band is not a result.
"""
import json, os, collections
SP=os.path.dirname(os.path.abspath(__file__))
MODES=["dom","som","vision","ptext","pprompt","psom"]
full=json.load(open(os.path.join(SP,"full_table2.json")))
cells=collections.defaultdict(dict)
for k,v in full.items():
    c,m=k.split("|"); cells[c][m]={int(t):v[t] for t in v}
nf=json.load(open("docs/analysis/cross_sites/noise_floor_inventory.json"))
band=collections.defaultdict(list)
_map={"B0.cls":"cls_B0","B0.red":"red_B0","B1.cls":"cls_B1","B1.red":"red_B1","B5.cls":"cls_B5"}
for p in nf["clean_pairs"]:
    k=".".join(p["label"].split(".")[:2])
    if k in _map: band[_map[k]]+=[p["self_drop_a_to_b_pp"],p["self_drop_b_to_a_pp"]]
V1={"cls":"router_classifieds_luna.json","red":"router_reddit_luna.json",
    "shop":"router_shopping3_luna.json","wa":"router_wa_reddit_luna.json"}
OVH=(0.0539/224, 3.0)
def dom_(p,q): return q[0]>=p[0] and q[1]<=p[1] and q[2]<=p[2] and (q[0]>p[0] or q[1]<p[1] or q[2]<p[2])

print(f"{'cell':<12}{'router':<16}{'n':>5}{'SR%':>8}{'Δbest':>8}{'Δtop':>8}"
      f"{'cost$':>9}{'lat s':>8}{'undom':>8}{'band':>12}")
print("-"*103)
for cid in sorted(cells):
    ms=[m for m in MODES if m in cells[cid]]
    if len(ms)<3: continue
    tasks=sorted(set.intersection(*[set(cells[cid][m]) for m in ms]))
    if len(tasks)<50: continue
    n0=len(tasks)
    sr={m:sum(cells[cid][m][t][0] for t in tasks)/n0*100 for m in ms}
    bm=max(sr,key=sr.get)
    b=band.get(cid); btxt=f"{min(b):.1f}-{max(b):.1f}" if b else "—"
    fixed=[(f"always-{m}", sr[m],
            sum((cells[cid][m][t][1] or 0) for t in tasks)/n0,
            sum((cells[cid][m][t][2] or 0) for t in tasks)/n0/1000) for m in ms]
    rows=[]
    # v1
    site=cid.split("_")[0]; key="wa" if cid.endswith("_WA") else site
    f1=os.path.join(SP,V1.get(key,""))
    if os.path.exists(f1):
        ch={int(k):v for k,v in json.load(open(f1))["choices"].items()}
        use=[t for t in tasks if ch.get(t) in ms]
        if len(use)>=50: rows.append(("v1 intent-only",use,{t:ch[t] for t in use}))
    # v3
    f3=os.path.join(SP,f"v3all_{cid}.json")
    if os.path.exists(f3):
        ch={int(k):v for k,v in json.load(open(f3))["choices"].items()}
        use=[t for t in tasks if ch.get(t) in ms]
        if len(use)>=50: rows.append(("v3 all+costaware",use,{t:ch[t] for t in use}))
    # 3-class
    fc=os.path.join(SP,f"c3_{cid}.json")
    if os.path.exists(fc):
        j=json.load(open(fc)); reps={int(k):v for k,v in j["reps"].items()}
        fold_of={t:fi for fi,fl in enumerate(j["folds"]) for t in fl}
        ch={}
        for k,v in j["choices"].items():
            t=int(k); fi=fold_of.get(t)
            if fi is None or v not in ("READ","LOOK","BOTH"): continue
            ch[t]={"READ":reps[fi],"LOOK":"vision","BOTH":"som"}[v]
        use=[t for t in tasks if ch.get(t) in ms]
        if len(use)>=50: rows.append(("3class R/L/B",use,{t:ch[t] for t in use}))
    printed=False
    for name,use,ch in rows:
        n=len(use)
        s=sum(cells[cid][ch[t]][t][0] for t in use)/n*100
        c=sum((cells[cid][ch[t]][t][1] or 0) for t in use)/n + OVH[0]
        l=sum((cells[cid][ch[t]][t][2] or 0) for t in use)/n/1000 + OVH[1]
        s2={m:sum(cells[cid][m][t][0] for t in use)/n*100 for m in ms}
        top=collections.Counter(ch.values()).most_common(1)[0][0]
        und=not any(dom_((s,c,l),q[1:]) for q in fixed)
        print(f"{cid if not printed else '':<12}{name:<16}{n:>5}{s:>7.2f}%"
              f"{s-s2[bm]:>+8.2f}{s-s2[top]:>+8.2f}{c:>9.4f}{l:>8.1f}"
              f"{'YES' if und else 'no':>8}{btxt if not printed else '':>12}")
        printed=True
