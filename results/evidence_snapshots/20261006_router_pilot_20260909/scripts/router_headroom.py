"""LLM router 的可用空间: best-single vs oracle, per cell, 对着 rerun band 读."""
import csv, collections, statistics as st
MODES = ["dom","som","vision","ptext","pprompt","psom"]
rows = list(csv.DictReader(open("results/phantom_paper/per_task_sr.csv")))
cells = collections.defaultdict(list)
for r in rows: cells[r["cell_id"]].append(r)

# rerun band (self_drop) per cell, from noise_floor_inventory
import json
nf = json.load(open("docs/analysis/cross_sites/noise_floor_inventory.json"))
band = collections.defaultdict(list)
for p in nf["clean_pairs"]:
    c = p["label"].split(".")[0]+"_"+p["label"].split(".")[1]
    c = {"B0_cls":"cls_B0","B0_red":"red_B0","B1_cls":"cls_B1","B1_red":"red_B1","B5_cls":"cls_B5"}.get(c,c)
    band[c] += [p["self_drop_a_to_b_pp"], p["self_drop_b_to_a_pp"]]

print(f"{'cell':<10}{'n':>5}{'best mode':>10}{'best SR':>9}{'oracle':>8}{'headroom':>10}{'rerun band':>14}{'verdict'}")
print("-"*88)
for cid in sorted(cells):
    rs = cells[cid]; n = len(rs)
    sr = {m: sum(float(r["sr_"+m]) for r in rs)/n*100 for m in MODES}
    bm = max(sr, key=sr.get)
    orac = sum(1 for r in rs if any(float(r["sr_"+m])>0 for m in MODES))/n*100
    head = orac - sr[bm]
    b = band.get(cid)
    btxt = f"{min(b):.2f}-{max(b):.2f}pp" if b else "—"
    if b:
        v = "**inside band**" if head <= max(b) else f"exceeds by {head-max(b):+.1f}pp"
    else:
        v = "no band"
    print(f"{cid:<10}{n:>5}{bm:>10}{sr[bm]:>8.2f}%{orac:>7.2f}%{head:>9.2f}pp{btxt:>14}   {v}")
