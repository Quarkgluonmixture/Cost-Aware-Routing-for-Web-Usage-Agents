"""Does the router STRICTLY dominate any fixed policy? (undominated is a low bar;
the paper notes always-pprompt at 19.64% SR is 'equally undominated'.)"""
import json, os, collections
SP=os.path.dirname(os.path.abspath(__file__))
MODES=["dom","som","vision","ptext","pprompt","psom"]
full=json.load(open(os.path.join(SP,"full_table2.json")))
cells=collections.defaultdict(dict)
for k,v in full.items():
    c,m=k.split("|"); cells[c][m]={int(t):v[t] for t in v}
OVH=(0.0539/224,3.0)
def dom_(p,q): return q[0]>=p[0] and q[1]<=p[1] and q[2]<=p[2] and (q[0]>p[0] or q[1]<p[1] or q[2]<p[2])
print(f"{'cell':<12}{'SR%':>7}{'cost$':>9}{'lat s':>8}   {'strictly dominates':<34}{'dominated by'}")
print("-"*96)
for cid in sorted(cells):
    f=os.path.join(SP,f"v3all_{cid}.json")
    if not os.path.exists(f): continue
    ms=[m for m in MODES if m in cells[cid]]
    tasks=sorted(set.intersection(*[set(cells[cid][m]) for m in ms]))
    if len(tasks)<50: continue
    ch={int(k):v for k,v in json.load(open(f))["choices"].items()}
    use=[t for t in tasks if ch.get(t) in ms]
    if len(use)<50: continue
    n=len(use)
    R=(sum(cells[cid][ch[t]][t][0] for t in use)/n*100,
       sum((cells[cid][ch[t]][t][1] or 0) for t in use)/n+OVH[0],
       sum((cells[cid][ch[t]][t][2] or 0) for t in use)/n/1000+OVH[1])
    fx={m:(sum(cells[cid][m][t][0] for t in use)/n*100,
           sum((cells[cid][m][t][1] or 0) for t in use)/n,
           sum((cells[cid][m][t][2] or 0) for t in use)/n/1000) for m in ms}
    beats=[m for m in ms if dom_(fx[m],R)]
    lost =[m for m in ms if dom_(R,fx[m])]
    print(f"{cid:<12}{R[0]:>6.2f}{R[1]:>9.4f}{R[2]:>8.1f}   "
          f"{(', '.join(beats) if beats else '— none —'):<34}{', '.join(lost) if lost else '—'}")
