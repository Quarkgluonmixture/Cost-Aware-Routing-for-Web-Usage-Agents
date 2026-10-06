"""Contextual bandit replay on full-information logs (every arm's outcome is logged per task).
Context = pre-flight, model-free step-0 page/task stats (what a router sees before any call).
Reward = success - lam * cost/median_cost(cell). LinUCB + Thompson (Bayesian ridge).
Replay over 200 random task orders. Report mean per-task reward, SR, cost vs fixed arms.
"""
import json, sys, math, collections, numpy as np
S=sys.argv[1]
rows=[json.loads(l) for l in open(f"{S}/step0.jsonl")]
meta=json.load(open(f"{S}/task_meta.json"))
EP=collections.defaultdict(dict)
for r in rows:
    if r["arm"]=="canon": EP[(r["cell"],r["mode"])][r["task_id"]]=r
def site_key(cell):
    s=cell.split("_")[0]; return s+"_WA" if cell.endswith("_WA") else s
def ctx(r,m):
    it=(m.get("intent","") or "").lower()
    return np.array([1.0,(r.get("s0_dom_complexity") or 0)/100,math.log1p(r.get("s0_text_length") or 0)/10,float(m.get("has_image",False)),
        math.log1p(len(it))/6]+[float(w in it) for w in ("cheapest","most","how many","color","image","subscribe","comment","post","price","review")])
cells=sorted({c for (c,m) in EP if c!="shop_B0_WA"})
LAM=float(sys.argv[2]) if len(sys.argv)>2 else 0.5
print(f"lambda={LAM} (reward = success - lambda*cost/median_cost)")
print(f"{'cell':<11}{'n':>4}{'arms':>5}{'fixed-best':>22}{'fixed-cheapest':>22}{'oracle-reward':>14}{'LinUCB':>22}{'Thompson':>22}{'lastQ arm dist (LinUCB)'}")
for cell in cells:
    modes=sorted({m for (c,m) in EP if c==cell}); sk=site_key(cell)
    tids=sorted(set.intersection(*[set(EP[(cell,m)]) for m in modes])); tids=[t for t in tids if "s0_action_type" in EP[(cell,modes[0])][t]]
    n=len(tids); K=len(modes)
    succ=np.array([[EP[(cell,m)][t]["success"] for m in modes] for t in tids],float)
    cost=np.array([[EP[(cell,m)][t].get("cost") or 0 for m in modes] for t in tids])
    medc=np.median(cost[cost>0]) if (cost>0).any() else 1.0
    R=succ-LAM*cost/medc
    X=np.array([ctx(EP[(cell,modes[0])][t],meta.get(f"{sk}|{t}",{})) for t in tids]); d=X.shape[1]
    fixed=[(R[:,k].mean(),succ[:,k].mean()*100,cost[:,k].mean(),modes[k]) for k in range(K)]
    fb=max(fixed); fc=min(fixed,key=lambda z:z[2]); orc=R.max(1).mean()
    rng=np.random.RandomState(0); res={"LinUCB":[], "Thompson":[]}; lastq=collections.Counter()
    for rep in range(200):
        order=rng.permutation(n)
        for algo in res:
            A=[np.eye(d) for _ in range(K)]; b=[np.zeros(d) for _ in range(K)]
            rs=[]; ss=[]; cs=[]
            for i,idx in enumerate(order):
                x=X[idx]; sc=[]
                for k in range(K):
                    Ainv=np.linalg.inv(A[k]); th=Ainv@b[k]
                    if algo=="LinUCB": sc.append(th@x+1.0*math.sqrt(x@Ainv@x))
                    else: sc.append(rng.multivariate_normal(th,0.25*Ainv)@x)
                k=int(np.argmax(sc)); r=R[idx,k]
                A[k]+=np.outer(x,x); b[k]+=r*x
                rs.append(r); ss.append(succ[idx,k]); cs.append(cost[idx,k])
                if algo=="LinUCB" and i>=0.75*n: lastq[modes[k]]+=1
            res[algo].append((np.mean(rs),np.mean(ss)*100,np.mean(cs)))
    lu=np.mean(res["LinUCB"],0); ts=np.mean(res["Thompson"],0)
    tot=sum(lastq.values()); dist=" ".join(f"{m}:{lastq[m]/tot*100:.0f}%" for m in modes if lastq[m]/tot>0.02)
    print(f"{cell:<11}{n:>4}{K:>5}  {fb[3]:<7}{fb[0]:+.3f}/{fb[1]:4.1f}%  {fc[3]:<7}{fc[0]:+.3f}/{fc[1]:4.1f}%  {orc:>+12.3f}   {lu[0]:+.3f}/{lu[1]:4.1f}%/${lu[2]:.4f}   {ts[0]:+.3f}/{ts[1]:4.1f}%/${ts[2]:.4f}   {dist}")
