import json, collections, os, random, statistics as st
SP=os.path.dirname(os.path.abspath(__file__))
MODES=["dom","som","vision","ptext","pprompt","psom"]
full=json.load(open(os.path.join(SP,"full_table.json")))
cells=collections.defaultdict(dict)
costs=collections.defaultdict(dict)
for k,v in full.items():
    c,m=k.split("|"); cells[c][m]={int(t):x[0] for t,x in v.items()}
    costs[c][m]={int(t):(x[1] or 0.0) for t,x in v.items()}

nf=json.load(open("docs/analysis/cross_sites/noise_floor_inventory.json"))
band=collections.defaultdict(list)
_map={"B0.cls":"cls_B0","B0.red":"red_B0","B1.cls":"cls_B1","B1.red":"red_B1","B5.cls":"cls_B5"}
for p in nf["clean_pairs"]:
    k=".".join(p["label"].split(".")[:2])
    if k in _map: band[_map[k]]+=[p["self_drop_a_to_b_pp"],p["self_drop_b_to_a_pp"]]

def routerfile(cid, rt):
    if cid.endswith("_WA"): return f"router_wa_reddit_{rt}.json"
    s=cid.split("_")[0]
    if s=="cls": return f"router_classifieds_{rt}.json"
    if s=="red": return f"router_reddit_{rt}.json"
    if s=="shop": return f"router_shopping{'3' if cid=='shop_B0' else '5'}_{rt}.json"
    return None

rng=random.Random(0)
print(f"{'cell':<12}{'rt':<6}{'n':>5}{'LLM':>8}{'best-single':>15}{'always-top':>14}"
      f"{'random':>8}{'oracle':>8}{'Δbest':>8}{'Δtop':>8}{'Δrand':>8}{'band':>11}")
print("-"*126)
res=collections.defaultdict(list)
for cid in sorted(cells):
    if cid.startswith("shop_B0_WA"): continue
    modes=[m for m in MODES if m in cells[cid]]
    if len(modes)<3: continue
    tasks=set.intersection(*[set(cells[cid][m]) for m in modes])
    if len(tasks)<50: continue
    b=band.get(cid); btxt=f"{min(b):.1f}-{max(b):.1f}" if b else "—"
    for rt in ("luna","terra"):
        f=os.path.join(SP, routerfile(cid,rt) or "")
        if not f or not os.path.exists(f): continue
        ch={int(k):v for k,v in json.load(open(f))["choices"].items()}
        use=sorted(t for t in tasks if ch.get(t) in modes)
        if len(use)<50: continue
        n=len(use)
        llm=sum(cells[cid][ch[t]][t] for t in use)/n*100
        sr={m: sum(cells[cid][m][t] for t in use)/n*100 for m in modes}
        bm=max(sr,key=sr.get); bsr=sr[bm]
        top=collections.Counter(ch[t] for t in use).most_common(1)[0][0]
        atop=sr[top]
        rnd=st.mean([sum(cells[cid][rng.choice(modes)][t] for t in use)/n*100 for _ in range(200)])
        orac=sum(1 for t in use if any(cells[cid][m][t] for m in modes))/n*100
        c_llm=sum(costs[cid][ch[t]][t] for t in use)/n
        c_top=sum(costs[cid][top][t] for t in use)/n
        cpct=(c_llm-c_top)/c_top*100 if c_top else 0.0
        db,dt,dr=llm-bsr, llm-atop, llm-rnd
        print(f"{cid:<12}{rt:<6}{n:>5}{llm:>7.2f}%{bsr:>10.2f}%({bm:<6}){atop:>9.2f}%({top:<6})"
              f"{rnd:>7.2f}%{orac:>7.2f}%{db:>+7.2f}{dt:>+8.2f}{dr:>+8.2f}{btxt:>11}{cpct:>+8.1f}%")
        res[rt].append((cid,db,dt,dr))
print("-"*126)
for rt,rows in res.items():
    dbs=[r[1] for r in rows]; dts=[r[2] for r in rows]; drs=[r[3] for r in rows]
    print(f"{rt:<6} n={len(rows):>2} cells | mean Δbest {st.mean(dbs):+.2f}pp "
          f"(beats best-single {sum(1 for d in dbs if d>0)}/{len(dbs)}) | "
          f"mean Δtop {st.mean(dts):+.2f}pp (beats {sum(1 for d in dts if d>0)}/{len(dts)}) | "
          f"mean Δrand {st.mean(drs):+.2f}pp (beats {sum(1 for d in drs if d>0)}/{len(drs)})")
