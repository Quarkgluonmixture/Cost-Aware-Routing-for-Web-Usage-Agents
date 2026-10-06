"""Evaluate the zero-shot LLM router against the paper's own comparators.

Read against the rerun band, per project discipline: a pp difference means
nothing until it is compared to what re-running one arm would have bought.
"""
import csv, json, collections, statistics as st, random, os, sys
MODES=["dom","som","vision","ptext","pprompt","psom"]
SP=os.path.dirname(os.path.abspath(__file__))

rows=list(csv.DictReader(open("results/phantom_paper/per_task_sr.csv")))
cells=collections.defaultdict(dict)
for r in rows:
    cells[r["cell_id"]][int(r["task_id"])]={m: float(r["sr_"+m])>0 for m in MODES}

nf=json.load(open("docs/analysis/cross_sites/noise_floor_inventory.json"))
band=collections.defaultdict(list)
_map={"B0.cls":"cls_B0","B0.red":"red_B0","B1.cls":"cls_B1","B1.red":"red_B1","B5.cls":"cls_B5"}
for p in nf["clean_pairs"]:
    k=".".join(p["label"].split(".")[:2])
    if k in _map: band[_map[k]] += [p["self_drop_a_to_b_pp"],p["self_drop_b_to_a_pp"]]

SITE_OF={"cls":"classifieds","red":"reddit"}
rng=random.Random(0)

print(f"{'cell':<9}{'router':<7}{'LLM SR':>8}{'best-single':>13}{'Δ vs best':>11}"
      f"{'Δ vs always-top':>16}{'oracle':>8}{'rerun band':>13}  verdict")
print("-"*104)
for cid in sorted(cells):
    site=SITE_OF[cid.split("_")[0]]
    tasks=cells[cid]; n=len(tasks)
    sr={m: sum(t[m] for t in tasks.values())/n*100 for m in MODES}
    bm=max(sr,key=sr.get)
    orac=sum(1 for t in tasks.values() if any(t[m] for m in MODES))/n*100
    b=band.get(cid); btxt=f"{min(b):.2f}-{max(b):.2f}" if b else "—"
    for rm in ["luna","terra"]:
        f=os.path.join(SP,f"router_{site}_{rm}.json")
        if not os.path.exists(f): continue
        ch={int(k):v for k,v in json.load(open(f))["choices"].items()}
        common=[t for t in tasks if t in ch and ch[t] in MODES]
        if not common: continue
        llm=sum(tasks[t][ch[t]] for t in common)/len(common)*100
        top=collections.Counter(ch[t] for t in common).most_common(1)[0][0]
        always_top=sum(tasks[t][top] for t in common)/len(common)*100
        d_best=llm-sr[bm]; d_top=llm-always_top
        v="**inside band**" if b and abs(d_best)<=max(b) else (f"exceeds" if b else "no band")
        print(f"{cid:<9}{rm:<7}{llm:>7.2f}%{sr[bm]:>8.2f}%({bm:<7}){d_best:>+10.2f}pp"
              f"{d_top:>+15.2f}pp{orac:>7.2f}%{btxt:>13}  {v}")
print("-"*104)
for rm in ["luna","terra"]:
    for site in ["classifieds","reddit"]:
        f=os.path.join(SP,f"router_{site}_{rm}.json")
        if not os.path.exists(f): continue
        j=json.load(open(f)); c=collections.Counter(j["choices"].values())
        tot=sum(c.values())
        top,topn=c.most_common(1)[0]
        print(f"{rm:<6} {site:<12} n={tot:<4} cost=${j['total_cost_usd']:.3f}  "
              f"top={top} {topn/tot*100:.1f}%  dist={dict(c.most_common())}")
