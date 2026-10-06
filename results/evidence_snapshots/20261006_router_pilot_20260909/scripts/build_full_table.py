"""Build per-task x per-mode success across ALL landed cells (not just the paper's 6).

Self-check: on the 6 cells that per_task_sr.csv covers, this table must agree
task-for-task. If it doesn't, the run-selection rule picked a non-canonical run.
"""
import json, glob, os, re, csv, collections, sys

MODE_OF = [("phantom_text","ptext"),("phantom_prompt","pprompt"),("phantom_som","psom"),
           ("phantom_dom","ptext"),("_som_","som"),("_dom_","dom"),("_vision_","vision")]
def mode_of(run):
    for pat,m in MODE_OF:
        if pat in run: return m
    return None
def site_of(run):
    for pat,s in (("classifieds","cls"),("reddit","red"),("shopping","shop")):
        if pat in run: return s
    return None

table=collections.defaultdict(dict)   # (bench,bl,site,mode) -> {tid: (success,cost,run,ts)}
for bench in ("visualwebarena","webarena"):
    for d in sorted(glob.glob(f"results/{bench}/phase1/*/")):
        run=os.path.basename(d.rstrip("/"))
        if "smoke" in run or run.startswith("_archive") or run.startswith("latest"): continue
        bl=run.split("_")[0]
        if not re.fullmatch(r"B\d", bl): continue
        m,s=mode_of(run),site_of(run)
        if not m or not s: continue
        ts=(re.search(r"_(\d{8})_",run) or re.search(r"_(\d{8})$",run))
        ts=ts.group(1) if ts else "00000000"
        recs={}
        for f in glob.glob(os.path.join(d,"*","episodes","*_summary_v2.json")):
            try: j=json.load(open(f))
            except Exception: continue
            tid=j.get("task_id")
            if tid is None: continue
            recs[int(tid)]=(bool(j.get("success")), j.get("total_cost_usd"),
                            j.get("total_latency_ms"), j.get("total_tokens"), j.get("steps"))
        if not recs: continue
        key=(bench,bl,s,m)
        # keep the LARGEST run; tie -> EARLIEST (canonical tends to be the first full run)
        prev=table.get(key)
        if not prev or (len(recs)>len(prev)) or (len(recs)==len(prev) and ts<prev.get("__ts__","z")):
            table[key]={t:v for t,v in recs.items()}; table[key]["__ts__"]=ts; table[key]["__run__"]=run

cells=collections.defaultdict(dict)
for (bench,bl,s,m),recs in table.items():
    cid=f"{s}_{bl}" + ("" if bench=="visualwebarena" else "_WA")
    cells[cid][m]={t:v for t,v in recs.items() if isinstance(t,int)}

print(f"{'cell':<12}{'modes':<38}{'n(min-max)':>14}")
for cid in sorted(cells):
    ms=cells[cid]; ns=[len(v) for v in ms.values()]
    print(f"{cid:<12}{','.join(sorted(ms)):<38}{min(ns)}-{max(ns):>4}")

# ---- self-check against canonical per_task_sr.csv (6 cells) ----
print("\n=== self-check vs per_task_sr.csv ===")
canon=collections.defaultdict(dict)
for r in csv.DictReader(open("results/phantom_paper/per_task_sr.csv")):
    for m in ("dom","som","vision","ptext","pprompt","psom"):
        canon[r["cell_id"]].setdefault(m,{})[int(r["task_id"])]=float(r["sr_"+m])>0
ok=bad=missing=0
for cid,modes in canon.items():
    if cid not in cells: print(f"  {cid}: MISSING from built table"); missing+=1; continue
    for m,tv in modes.items():
        mine=cells[cid].get(m,{})
        for t,v in tv.items():
            if t not in mine: missing+=1
            elif mine[t][0]==v: ok+=1
            else: bad+=1
tot=ok+bad
print(f"  agree {ok}/{tot} ({ok/tot*100:.2f}%)  disagree {bad}  missing {missing}")
json.dump({f"{c}|{m}":{str(t):list(v) for t,v in tv.items()}
           for c,ms in cells.items() for m,tv in ms.items()},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"full_table2.json"),"w"))
print("  -> full_table.json written")
