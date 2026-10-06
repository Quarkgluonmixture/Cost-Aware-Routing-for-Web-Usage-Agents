"""Pareto evaluation: is the LLM router undominated on (success, cost, latency)?

The paper's own router criterion (Table 8) is undominated-on-three-axes, not
"higher SR". Judging a router on SR alone hides exactly the case it exists for:
lower SR bought back in cost.

Router overhead is charged to the router. The paper's rule-based comparator is
a 0-token regex; an LLM router is not free, and omitting its own bill would be
the comparison the paper spends §5 warning about.
"""
import json, os, collections
SP=os.path.dirname(os.path.abspath(__file__))
MODES=["dom","som","vision","ptext","pprompt","psom"]
full=json.load(open(os.path.join(SP,"full_table2.json")))
cells=collections.defaultdict(dict)
for k,v in full.items():
    c,m=k.split("|"); cells[c][m]={int(t):v[t] for t in v}

# measured router overhead per task (from the runs we just did)
OVH={"luna":(0.0539/224, 3.0), "terra":(0.2962/224, 4.0),
     "v2full":(0.0,4.0), "v2nobr":(0.0,4.0)}   # v2 cost filled below if available

def dom_(p,q):
    return q[0]>=p[0] and q[1]<=p[1] and q[2]<=p[2] and (q[0]>p[0] or q[1]<p[1] or q[2]<p[2])

FILES={"cls_B0":[("v1-luna","router_classifieds_luna.json","luna"),
                 ("v1-terra","router_classifieds_terra.json","terra"),
                 ("v2-full","v2_clsB0_full.json","v2full"),
                 ("v2-nobaserate","v2_clsB0_nobr.json","v2nobr"),
                 ("v3-cost-aware","v3_clsB0.json","v2full")],
       "red_B0":[("v1-luna","router_reddit_luna.json","luna"),
                 ("v1-terra","router_reddit_terra.json","terra"),
                 ("v3-cost-aware","v3_redB0.json","v2full")],
       "cls_B5":[("v1-luna","router_classifieds_luna.json","luna"),
                 ("v1-terra","router_classifieds_terra.json","terra"),
                 ("v3-cost-aware","v3_clsB5.json","v2full")],
       "shop_B0":[("v1-luna","router_shopping3_luna.json","luna"),
                  ("v1-terra","router_shopping3_terra.json","terra")],
       "red_B0_WA":[("v1-luna","router_wa_reddit_luna.json","luna"),
                    ("v1-terra","router_wa_reddit_terra.json","terra")]}

for cid,variants in FILES.items():
    if cid not in cells: continue
    ms=[m for m in MODES if m in cells[cid]]
    tasks=set.intersection(*[set(cells[cid][m]) for m in ms])
    if len(tasks)<50: continue
    print(f"\n=== {cid} (n={len(tasks)}) ===")
    rows=[]
    for m in ms:
        r=[cells[cid][m][t] for t in tasks]; n=len(r)
        rows.append((f"always-{m}", sum(1 for x in r if x[0])/n*100,
                     sum((x[1] or 0) for x in r)/n, sum((x[2] or 0) for x in r)/n/1000))
    for name,fn,ov in variants:
        p=os.path.join(SP,fn)
        if not os.path.exists(p): continue
        ch={int(k):v for k,v in json.load(open(p))["choices"].items()}
        use=sorted(t for t in tasks if ch.get(t) in ms)
        if len(use)<50: continue
        n=len(use); oc,ol=OVH.get(ov,(0,0))
        sr=sum(1 for t in use if cells[cid][ch[t]][t][0])/n*100
        co=sum((cells[cid][ch[t]][t][1] or 0) for t in use)/n + oc
        la=sum((cells[cid][ch[t]][t][2] or 0) for t in use)/n/1000 + ol
        rows.append((name,sr,co,la))
    print(f"  {'policy':<18}{'SR%':>7}{'cost$/ep':>11}{'lat s/ep':>10}   undominated?")
    for r in sorted(rows,key=lambda x:-x[1]):
        und = not any(dom_(r[1:],q[1:]) for q in rows if q[0]!=r[0])
        star = "**LLM**" if not r[0].startswith("always-") else ""
        print(f"  {r[0]:<18}{r[1]:>6.2f}{r[2]:>11.4f}{r[3]:>10.1f}   "
              f"{'YES' if und else 'dominated'}  {star}")
