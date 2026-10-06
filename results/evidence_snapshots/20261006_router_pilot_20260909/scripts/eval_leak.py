"""random 5-fold vs GroupKFold-by-template: how much of v3 was template matching?

86.6% of test tasks have a same-template sibling in the random-fold few-shot pool.
Same-template VWA tasks are near-duplicates ("find the oldest listed item whose
image ..." with the category swapped), so a router can answer by matching the
template rather than by reading the task. Grouping removes that route.
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
OVH=(0.0539/224,3.0)
def dom_(p,q): return q[0]>=p[0] and q[1]<=p[1] and q[2]<=p[2] and (q[0]>p[0] or q[1]<p[1] or q[2]<p[2])
print(f"{'cell':<10}{'split':<16}{'n':>5}{'SR%':>8}{'Δbest':>8}{'cost$':>9}{'lat s':>8}"
      f"{'strictly dominates':>20}{'band':>11}")
print("-"*97)
for cid in ["cls_B0","cls_B1","red_B0","red_B2"]:
    ms=[m for m in MODES if m in cells[cid]]
    tasks=sorted(set.intersection(*[set(cells[cid][m]) for m in ms]))
    b=band.get(cid); btxt=f"{min(b):.1f}-{max(b):.1f}" if b else "—"
    first=True
    for lab,fn in (("random 5-fold",f"v3all_{cid}.json"),("GroupKFold tmpl",f"v3g_{cid}.json")):
        p=os.path.join(SP,fn)
        if not os.path.exists(p): print(f"{cid if first else '':<10}{lab:<16}  (pending)"); first=False; continue
        ch={int(k):v for k,v in json.load(open(p))["choices"].items()}
        use=[t for t in tasks if ch.get(t) in ms]
        if len(use)<50: continue
        n=len(use)
        R=(sum(cells[cid][ch[t]][t][0] for t in use)/n*100,
           sum((cells[cid][ch[t]][t][1] or 0) for t in use)/n+OVH[0],
           sum((cells[cid][ch[t]][t][2] or 0) for t in use)/n/1000+OVH[1])
        fx={m:(sum(cells[cid][m][t][0] for t in use)/n*100,
               sum((cells[cid][m][t][1] or 0) for t in use)/n,
               sum((cells[cid][m][t][2] or 0) for t in use)/n/1000) for m in ms}
        bm=max(fx,key=lambda m:fx[m][0])
        beats=[m for m in ms if dom_(fx[m],R)]
        print(f"{cid if first else '':<10}{lab:<16}{n:>5}{R[0]:>7.2f}%{R[0]-fx[bm][0]:>+8.2f}"
              f"{R[1]:>9.4f}{R[2]:>8.1f}{len(beats):>15} arms{btxt if first else '':>11}")
        first=False
