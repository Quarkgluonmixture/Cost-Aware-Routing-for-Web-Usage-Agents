import json, os, collections
SP=os.path.dirname(os.path.abspath(__file__))
MODES=["dom","som","vision","ptext","pprompt","psom"]
full=json.load(open(os.path.join(SP,"full_table2.json")))
cells=collections.defaultdict(dict)
for k,v in full.items():
    c,m=k.split("|"); cells[c][m]={int(t):v[t] for t in v}

def dominated(p,q):  # q dominates p ? (SR higher-better, cost/lat lower-better)
    return q[0]>=p[0] and q[1]<=p[1] and q[2]<=p[2] and (q[0]>p[0] or q[1]<p[1] or q[2]<p[2])

for cid in sorted(cells):
    if cid.startswith("shop_B0_WA"): continue
    ms=[m for m in MODES if m in cells[cid]]
    if len(ms)<3: continue
    tasks=set.intersection(*[set(cells[cid][m]) for m in ms])
    if len(tasks)<50: continue
    n=len(tasks); prof={}
    for m in ms:
        r=[cells[cid][m][t] for t in tasks]
        sr=sum(1 for x in r if x[0])/n*100
        cost=sum((x[1] or 0) for x in r)/n
        lat=sum((x[2] or 0) for x in r)/n/1000
        prof[m]=(sr,cost,lat)
    front=[m for m in ms if not any(dominated(prof[m],prof[q]) for q in ms if q!=m)]
    print(f"\n{cid}  (n={n})   Pareto-undominated modes: {', '.join(front)}")
    print(f"  {'mode':<9}{'SR%':>7}{'cost$/ep':>11}{'lat s/ep':>10}   {'on frontier'}")
    for m in sorted(ms,key=lambda x:-prof[x][0]):
        s,c,l=prof[m]
        print(f"  {m:<9}{s:>6.2f}{c:>11.4f}{l:>10.1f}   {'YES' if m in front else ''}")
