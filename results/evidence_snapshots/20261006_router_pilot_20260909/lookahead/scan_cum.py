import json, glob, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.getcwd())
from p79.experiment.io_utils import read_jsonl_dedup
S=sys.argv[1]
rows=[json.loads(l) for l in open(f"{S}/step0.jsonl")]
jobs=sorted({(r["cell"],r["mode"],r["cond_dir"]) for r in rows if r["arm"]=="canon" and r["cell"]!="shop_B0_WA"})
def one(k):
    cell,mode,cdir=k; out=[]
    for sf in glob.glob(os.path.join(cdir,"episodes","*_steps_v2.jsonl")):
        tid=int(os.path.basename(sf).split("_task_")[1].split("_steps")[0])
        try: recs=read_jsonl_dedup(sf)
        except Exception: continue
        c=[(r.get("cost_usd") or {}).get("total") or 0 for r in recs]; l=[(r.get("latency_ms") or {}).get("total") or 0 for r in recs]
        summ=sf.replace("_steps_v2.jsonl","_summary_v2.json")
        try: sj=json.load(open(summ)); succ=bool(sj.get("success")); steps=sj.get("steps")
        except Exception: continue
        out.append({"cell":cell,"mode":mode,"task":tid,"success":succ,"steps":steps,"n_rec":len(recs),"cost_steps":c,"lat_steps":l})
    return out
with Pool(16) as p: res=[x for xs in p.imap_unordered(one,jobs) for x in xs]
json.dump(res,open(f"{S}/cum_scan.json","w")); print("DONE",len(res))
