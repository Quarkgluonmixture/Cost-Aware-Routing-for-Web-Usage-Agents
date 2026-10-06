"""Step-0 (and step-1) feature table for every episode in every landed cell + replicate arms.

Run selection for canonical arms mirrors results/router_llm_pilot_20260909/scripts/build_full_table.py
(largest run per (bench, baseline, site, mode); tie -> earliest). Replicate arms come from
docs/analysis/cross_sites/noise_floor_inventory.json clean_pairs (arm_a / arm_b).
Output: one JSONL row per episode.
"""
import json, glob, os, re, sys, collections
from multiprocessing import Pool
sys.path.insert(0, os.getcwd())
from p79.experiment.io_utils import read_jsonl_dedup

OUT = sys.argv[1]
MODE_OF = [("phantom_text","ptext"),("phantom_prompt","pprompt"),("phantom_som","psom"),
           ("phantom_dom","ptext"),("_som_","som"),("_dom_","dom"),("_vision_","vision")]
def mode_of(run):
    for pat,m in MODE_OF:
        if pat in run: return m
def site_of(run):
    for pat,s in (("classifieds","cls"),("reddit","red"),("shopping","shop")):
        if pat in run: return s

# ---- canonical run selection (copied rule) ----
table={}
for bench in ("visualwebarena","webarena"):
    for d in sorted(glob.glob(f"results/{bench}/phase1/*/")):
        run=os.path.basename(d.rstrip("/"))
        if "smoke" in run or run.startswith("_archive") or run.startswith("latest"): continue
        bl=run.split("_")[0]
        if not re.fullmatch(r"B\d", bl): continue
        m,s=mode_of(run),site_of(run)
        if not m or not s: continue
        ts=(re.search(r"_(\d{8})_",run) or re.search(r"_(\d{8})$",run)); ts=ts.group(1) if ts else "00000000"
        conds=[c for c in glob.glob(os.path.join(d,"*","episodes")) ]
        n=sum(len(glob.glob(os.path.join(c,"*_summary_v2.json"))) for c in conds)
        if n==0: continue
        key=(bench,bl,s,m)
        prev=table.get(key)
        if not prev or n>prev[0] or (n==prev[0] and ts<prev[1]):
            table[key]=(n,ts,d.rstrip("/"))
jobs=[]  # (label, cell, mode, arm_tag, condition_dir)
for (bench,bl,s,m),(n,ts,d) in table.items():
    cell=f"{s}_{bl}"+("" if bench=="visualwebarena" else "_WA")
    for c in glob.glob(os.path.join(d,"*")):
        if os.path.isdir(os.path.join(c,"episodes")):
            jobs.append((cell,m,"canon",c))
nf=json.load(open("docs/analysis/cross_sites/noise_floor_inventory.json"))
for p in nf["clean_pairs"]:
    bl,site,m=p["label"].split(".")
    cell=f"{site}_{bl}"
    for tag,key in (("repA","arm_a"),("repB","arm_b")):
        jobs.append((cell,m,tag,p[key]))
print(f"{len(jobs)} condition dirs", file=sys.stderr)

CONF=["mean_logprob","min_logprob","mean_margin","min_margin","mean_entropy","max_entropy","verbalized"]
def g(d,*ks):
    for k in ks:
        if not isinstance(d,dict): return None
        d=d.get(k)
    return d
def step_feats(r,prefix):
    o={}
    for c in CONF: o[f"{prefix}_{c}"]=g(r,"confidence",c)
    th=g(r,"action","thought") or g(r,"raw_action","thought") or ""
    o[f"{prefix}_thought"]=th; o[f"{prefix}_thought_len"]=len(th)
    o[f"{prefix}_action_type"]=r.get("action_type") or g(r,"action","action_type")
    o[f"{prefix}_element_id"]=g(r,"action","element_id")
    o[f"{prefix}_action_text"]=g(r,"action","text")
    o[f"{prefix}_action_success"]=r.get("action_success")
    o[f"{prefix}_page_changed"]=r.get("page_changed")
    o[f"{prefix}_parse_valid"]=r.get("parse_valid")
    o[f"{prefix}_tool_call_valid"]=r.get("tool_call_valid")
    o[f"{prefix}_url_before"]=g(r,"state_digest","url_before")
    o[f"{prefix}_url_after"]=g(r,"state_digest","url_after")
    o[f"{prefix}_dom_complexity"]=g(r,"state_digest","dom_complexity")
    o[f"{prefix}_text_length"]=g(r,"state_digest","text_length")
    o[f"{prefix}_mark_count"]=g(r,"som","mark_count")
    o[f"{prefix}_tokens_in"]=g(r,"tokens","input"); o[f"{prefix}_tokens_out"]=g(r,"tokens","output")
    o[f"{prefix}_cost"]=g(r,"cost_usd","total")
    o[f"{prefix}_lat_infer"]=g(r,"latency_ms","backend_infer")
    o[f"{prefix}_lat_total"]=g(r,"latency_ms","total")
    o[f"{prefix}_done"]=r.get("done"); o[f"{prefix}_reward"]=r.get("reward")
    o[f"{prefix}_locator_ok"]=g(r,"locator_route_meta","success")
    o[f"{prefix}_error"]=r.get("error_category")
    return o

def one(job):
    cell,mode,tag,cdir=job
    rows=[]
    for sf in glob.glob(os.path.join(cdir,"episodes","*_summary_v2.json")):
        try: summ=json.load(open(sf))
        except Exception: continue
        tid=summ.get("task_id")
        if tid is None: continue
        sp=sf.replace("_summary_v2.json","_steps_v2.jsonl")
        row={"cell":cell,"mode":mode,"arm":tag,"cond_dir":cdir,"task_id":int(tid),
             "success":bool(summ.get("success")),"steps":summ.get("steps"),
             "cost":summ.get("total_cost_usd"),"latency":summ.get("total_latency_ms"),
             "tokens":summ.get("total_tokens"),"n_step_records":0}
        if os.path.exists(sp):
            try: recs=list(read_jsonl_dedup(sp))
            except Exception as e:
                row["read_error"]=str(e)[:100]; rows.append(row); continue
            row["n_step_records"]=len(recs)
            s0=[r for r in recs if r.get("step_idx")==0]
            s1=[r for r in recs if r.get("step_idx")==1]
            if s0: row.update(step_feats(s0[-1],"s0"))
            if s1: row.update(step_feats(s1[-1],"s1"))
            # step-level confidence trajectory (mean_logprob / verbalized) for later
            row["conf_traj_mlp"]=[g(r,"confidence","mean_logprob") for r in recs]
            row["conf_traj_verb"]=[g(r,"confidence","verbalized") for r in recs]
            row["url_after_traj"]=[g(r,"state_digest","url_after") for r in recs]
        rows.append(row)
    return rows

if __name__=="__main__":
    with Pool(16) as pool, open(OUT,"w") as f:
        n=0
        for rows in pool.imap_unordered(one,jobs):
            for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n"); n+=1
            print(f"{n} episodes", file=sys.stderr, end="\r")
    print(f"\nDONE {n}", file=sys.stderr)
