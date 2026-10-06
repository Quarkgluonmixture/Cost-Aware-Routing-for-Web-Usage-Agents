import json, collections, os, random, statistics as st
SP=os.path.dirname(os.path.abspath(__file__))
MODES=["dom","som","vision","ptext","pprompt","psom"]
full=json.load(open(os.path.join(SP,"full_table.json")))
cells=collections.defaultdict(dict)
for k,v in full.items():
    c,m=k.split("|"); cells[c][m]={int(t):x[0] for t,x in v.items()}

nf=json.load(open("docs/analysis/cross_sites/noise_floor_inventory.json"))
band=collections.defaultdict(list)
_map={"B0.cls":"cls_B0","B0.red":"red_B0","B1.cls":"cls_B1","B1.red":"red_B1","B5.cls":"cls_B5"}
for p in nf["clean_pairs"]:
    k=".".join(p["label"].split(".")[:2])
    if k in _map: band[_map[k]]+=[p["self_drop_a_to_b_pp"],p["self_drop_b_to_a_pp"]]

SITEFILE={"cls":"classifieds","red":"reddit","shop":"shopping"}
rng=random.Random(0)
print(f"{'cell':<11}{'rt':<6}{'LLM':>7}{'best-single':>14}{'always-top':>13}"
      f"{'Δbest':>8}{'Δtop':>8}{'oracle':>8}{'band':>12}  verdict")
print("-"*100)
res=[]
for cid in sorted(cells):
    if cid.startswith("shop_B0_WA"): continue
    site=cid.split("_")[0]
    modes=[m for m in MODES if m in cells[cid]]
    tasks=set.intersection(*[set(cells[cid][m]) for m in modes])
    if len(tasks)<50: continue
    n=len(tasks)
    sr={m: sum(cells[cid][m][t] for t in tasks)/n*100 for m in modes}
    bm=max(sr,key=sr.get)
    orac=sum(1 for t in tasks if any(cells[cid][m][t] for m in modes))/n*100
    b=band.get(cid); btxt=f"{min(b):.1f}-{max(b):.1f}" if b else "—"
    for rt in ("luna","terra"):
        f=os.path.join(SP,f"router_{SITEFILE[site]}_{rt}.json")
        if not os.path.exists(f): continue
        ch={int(k):v for k,v in json.load(open(f))["choices"].items()}
        use=[t for t in tasks if ch.get(t) in modes]
        if len(use)<50: continue
        llm=sum(cells[cid][ch[t]][t] for t in use)/len(use)*100
        top=collections.Counter(ch[t] for t in use).most_common(1)[0][0]
        atop=sum(cells[cid][top][t] for t in use)/len(use)*100
        bsr=sum(cells[cid][bm][t] for t in use)/len(use)*100
        db,dt=llm-bsr, llm-atop
        v=("**inside band**" if b and abs(db)<=max(b) else ("beats band" if b and db>max(b)
            else ("loses beyond band" if b and db<-max(b) else "no band")))
        print(f"{cid:<11}{rt:<6}{llm:>6.2f}%{bsr:>9.2f}%({bm:<7}){atop:>8.2f}%({top:<7})"
              f"{db:>+7.2f}{dt:>+8.2f}{orac:>7.2f}%{btxt:>12}  {v}")
        res.append((cid,rt,db,dt))
print("-"*100)
for rt in ("luna","terra"):
    ds=[d for c,r,d,_ in res if r==rt]; dts=[t for c,r,_,t in res if r==rt]
    if ds: print(f"{rt}: mean Δ vs best-single {st.mean(ds):+.2f}pp (n={len(ds)} cells), "
                 f"mean Δ vs always-top {st.mean(dts):+.2f}pp; beats best-single in "
                 f"{sum(1 for d in ds if d>0)}/{len(ds)}")
