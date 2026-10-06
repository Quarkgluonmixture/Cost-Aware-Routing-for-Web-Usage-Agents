"""One-step-lookahead router: incremental AUROC + offline policy evaluation with five controls.

Feature tiers (all computed from the CHEAP mode's own step-0 record):
  OBS0   : model-free step-0 page + task stats  (what §457 abstention used)
  +MODEL0: + cheap model's step-0 output (confidence fields / thought stats / action type)
  +POST0 : + result of executing step-0 action (action_success / page_changed / url changed)
           -- NOT splice-legal for upgrade-to-rich (rich would start from a touched site);
              legal only for cheap-restart. Reported as an information-increment bound.
Labels (all defined on every task):
  self  : cheap mode succeeds          (dense)
  any   : any of the cell's modes succeeds (abstention label, dense)
  rich  : rich mode succeeds           (dense, cross-mode)
  upg   : cheap fails AND rich succeeds (the routing target; sparse positives)
CV: task-level StratifiedKFold(5, seed 42) and GroupKFold by intent_template_id. L2 LR.
Shuffle null: permute TRAIN labels only (as §6/§457), 30 reps, median.
Policy eval: escalate top-f tasks by out-of-fold score; cost = cheap step-0 inference cost
  + rich full episode when escalated, cheap full episode otherwise.
"""
import json, sys, re, collections, random, math, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
S=sys.argv[1]; OUT=sys.argv[2]
rows=[json.loads(l) for l in open(f"{S}/step0.jsonl")]
meta=json.load(open(f"{S}/task_meta.json"))
nf=json.load(open("docs/analysis/cross_sites/noise_floor_inventory.json"))
band=collections.defaultdict(list)
for p in nf["clean_pairs"]:
    bl,site,m=p["label"].split("."); band[f"{site}_{bl}"]+=[p["self_drop_a_to_b_pp"],p["self_drop_b_to_a_pp"]]

def site_key(cell):
    s=cell.split("_")[0]; return s+"_WA" if cell.endswith("_WA") else s
EP=collections.defaultdict(dict)   # (cell,mode,arm) -> tid -> row
for r in rows: EP[(r["cell"],r["mode"],r["arm"])][r["task_id"]]=r

UNC=re.compile(r"\b(not (?:visible|sure|clear|certain|shown|present)|cannot|can't|unclear|unable|no (?:\w+ )?(?:visible|match|listing|result)|does not (?:appear|show)|doesn't (?:appear|show)|hard to)\b",re.I)
IMG=re.compile(r"\b(image|picture|photo|screenshot|thumbnail|appears? to (?:be|show)|looks like|shown in)\b",re.I)
PLAN=re.compile(r"\b(search|type|enter)\b",re.I)
NAV=re.compile(r"\b(scroll|category|categories|dropdown|select|filter|navigate|forum)\b",re.I)
FIN=re.compile(r"\b(finish|answer|complete|done|stop)\b",re.I)
ACTS=["type","click","select_option","scroll","finish","goto","hover","key_press","wait"]

def feats(r, tier, bl, mkey):
    m=meta.get(mkey,{})
    f={}
    # OBS0
    f["dom_complexity"]=r.get("s0_dom_complexity") or 0
    f["text_length"]=math.log1p(r.get("s0_text_length") or 0)
    f["mark_count"]=r.get("s0_mark_count") or 0
    f["has_image"]=float(m.get("has_image",False))
    f["intent_len"]=math.log1p(len(m.get("intent","")))
    it=m.get("intent","").lower()
    for w in ("cheapest","most expensive","most recent","newest","oldest","how many","count","color","colour","image","picture","subscribe","comment","post","reply","price","review","rating"):
        f["int_"+w.replace(" ","_")]=float(w in it)
    if tier=="OBS0": return f
    # MODEL0
    for c in ("mean_logprob","min_logprob","mean_margin","min_margin"):
        if bl!="B5": f[c]=r.get("s0_"+c) if r.get("s0_"+c) is not None else 0.0
    if bl in ("B1","B2"):
        for c in ("mean_entropy","max_entropy"): f[c]=r.get("s0_"+c) if r.get("s0_"+c) is not None else 0.0
    v=r.get("s0_verbalized"); f["verbalized"]=v if v is not None else 0.85; f["verb_missing"]=float(v is None)
    f["thought_len"]=math.log1p(r.get("s0_thought_len") or 0)
    f["tokens_out"]=math.log1p(r.get("s0_tokens_out") or 0)
    f["lat_infer"]=math.log1p(r.get("s0_lat_infer") or 0)
    th=r.get("s0_thought") or ""
    f["th_unc"]=float(bool(UNC.search(th))); f["th_img"]=float(bool(IMG.search(th))); f["th_plan"]=float(bool(PLAN.search(th)))
    f["th_nav"]=float(bool(NAV.search(th))); f["th_fin"]=float(bool(FIN.search(th)))
    f["th_n_unc"]=len(UNC.findall(th))
    at=r.get("s0_action_type") or "none"
    for a in ACTS: f["act_"+a]=float(at==a)
    f["has_eid"]=float(r.get("s0_element_id") is not None)
    f["parse_valid"]=float(bool(r.get("s0_parse_valid"))); f["tc_valid"]=float(bool(r.get("s0_tool_call_valid")))
    if tier=="MODEL0": return f
    # POST0
    f["act_success"]=float(bool(r.get("s0_action_success"))); f["page_changed"]=float(bool(r.get("s0_page_changed")))
    f["url_changed"]=float((r.get("s0_url_before") or "")!=(r.get("s0_url_after") or ""))
    f["done0"]=float(bool(r.get("s0_done")))
    return f

def oof_scores(X,y,groups,cv,seed=42,shuffle=False):
    n=len(y); s=np.zeros(n); rng=np.random.RandomState(seed)
    if cv=="strat": splits=StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y)
    else: splits=GroupKFold(5).split(X,y,groups)
    for tr,te in splits:
        ytr=y[tr].copy()
        if shuffle: rng.shuffle(ytr)
        if ytr.sum()==0 or ytr.sum()==len(ytr): s[te]=ytr.mean(); continue
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000))
        clf.fit(X[tr],ytr); s[te]=clf.predict_proba(X[te])[:,1]
    return s
def auc(y,s):
    return roc_auc_score(y,s) if 0<y.sum()<len(y) else float("nan")

results={"auroc":[], "policy":[]}
cells=sorted({c for (c,m,a) in EP if a=="canon" and c!="shop_B0_WA"})
for cell in cells:
    bl=cell.split("_")[1]; sk=site_key(cell)
    modes=sorted({m for (c,m,a) in EP if c==cell and a=="canon"})
    tids=sorted(set.intersection(*[set(EP[(cell,m,"canon")]) for m in modes]))
    tids=[t for t in tids if "s0_action_type" in EP[(cell,modes[0],"canon")][t]]
    if len(tids)<100: print("skip",cell,len(tids)); continue
    succ={m:np.array([EP[(cell,m,"canon")][t]["success"] for t in tids],float) for m in modes}
    cost={m:np.array([EP[(cell,m,"canon")][t].get("cost") or 0 for t in tids]) for m in modes}
    cost0={m:np.array([EP[(cell,m,"canon")][t].get("s0_cost") or 0 for t in tids]) for m in modes}
    groups=np.array([meta.get(f"{sk}|{t}",{}).get("template","?") for t in tids])
    y_any=np.max(np.stack([succ[m] for m in modes]),0)
    sr={m:succ[m].mean()*100 for m in modes}; mc={m:cost[m].mean() for m in modes}
    best=max(sr,key=sr.get); cheapest=min(mc,key=mc.get)
    print(f"\n=== {cell} n={len(tids)} modes={modes} best={best}({sr[best]:.1f}%) cheapest={cheapest}(${mc[cheapest]:.4f}) any={y_any.mean()*100:.1f}%",flush=True)
    pairs=set()
    for c_ in (cheapest,"dom","vision"):
        if c_ in modes:
            for r_ in (best,"som"):
                if r_ in modes and r_!=c_: pairs.add((c_,r_))
    for cheap in modes:
        Xt={tier:np.array([[v for v in feats(EP[(cell,cheap,"canon")][t],tier,bl,f"{sk}|{t}").values()] for t in tids]) for tier in ("OBS0","MODEL0","POST0")}
        labels={"self":succ[cheap],"any":y_any}
        for (c_,r_) in pairs:
            if c_==cheap: labels[f"rich:{r_}"]=succ[r_]; labels[f"upg:{r_}"]=((1-succ[cheap])*succ[r_])
        for lab,y in labels.items():
            if y.sum()<5: continue
            for tier in ("OBS0","MODEL0","POST0"):
                for cv in ("strat","group"):
                    s=oof_scores(Xt[tier],y,groups,cv); a=auc(y,s)
                    nulls=[auc(y,oof_scores(Xt[tier],y,groups,cv,seed=100+k,shuffle=True)) for k in range(30)]
                    results["auroc"].append({"cell":cell,"cheap":cheap,"label":lab,"tier":tier,"cv":cv,"auroc":a,"null_med":float(np.nanmedian(nulls)),"null_p95":float(np.nanpercentile(nulls,95)),"n_pos":int(y.sum()),"n":len(y)})
                    if cv=="strat" and tier=="MODEL0": print(f"  {cheap:<8}{lab:<12}{tier:<7} AUROC={a:.3f} null={np.nanmedian(nulls):.3f}/{np.nanpercentile(nulls,95):.3f} pos={int(y.sum())}",flush=True)
        # ---- policy evaluation for pairs with this cheap ----
        for (c_,r_) in pairs:
            if c_!=cheap: continue
            for tier in ("OBS0","MODEL0","POST0"):
                # score = oof P(upg) ; also try P(rich)-P(self)
                y_upg=(1-succ[cheap])*succ[r_]
                if y_upg.sum()<3: continue
                s_upg=oof_scores(Xt[tier],y_upg,groups,"strat")
                s_diff=oof_scores(Xt[tier],succ[r_],groups,"strat")-oof_scores(Xt[tier],succ[cheap],groups,"strat")
                n=len(tids)
                base_c=(succ[cheap].mean()*100,cost[cheap].mean()); base_r=(succ[r_].mean()*100,cost[r_].mean())
                orc_mask=y_upg>0
                orc=(np.where(orc_mask,succ[r_],succ[cheap]).mean()*100, np.where(orc_mask,cost0[cheap]+cost[r_],cost[cheap]).mean())
                # cheap-restart counterfactual if replicate exists
                rep=None
                if (cell,cheap,"repB") in EP and (cell,cheap,"repA") in EP:
                    A=EP[(cell,cheap,"repA")]; B=EP[(cell,cheap,"repB")]
                    if all(t in A and t in B for t in tids):
                        # which arm is canonical? use the one != canon as restart outcome
                        canon_dir=EP[(cell,cheap,"canon")][tids[0]]["cond_dir"]
                        other=B if A[tids[0]]["cond_dir"]==canon_dir else A
                        rep={"succ":np.array([other[t]["success"] for t in tids],float),"cost":np.array([other[t].get("cost") or 0 for t in tids])}
                for sname,score in (("p_upg",s_upg),("p_rich-p_self",s_diff)):
                    order=np.argsort(-score)
                    curve=[]
                    for frac in (0.05,0.1,0.15,0.2,0.3,0.4,0.5):
                        k=int(round(frac*n)); esc=np.zeros(n,bool); esc[order[:k]]=True
                        srp=np.where(esc,succ[r_],succ[cheap]).mean()*100
                        cp=np.where(esc,cost0[cheap]+cost[r_],cost[cheap]).mean()
                        rnd_sr=(1-frac)*base_c[0]+frac*base_r[0]; rnd_c=(1-frac)*base_c[1]+frac*(cost0[cheap].mean()+base_r[1])
                        pt={"frac":frac,"sr":srp,"cost":cp,"rand_sr":rnd_sr,"rand_cost":rnd_c,"beats_rich":bool(srp>=base_r[0] and cp<=base_r[1] and (srp>base_r[0] or cp<base_r[1]))}
                        if rep is not None:
                            pt["restart_sr"]=np.where(esc,rep["succ"],succ[cheap]).mean()*100
                            pt["restart_cost"]=np.where(esc,cost0[cheap]+rep["cost"],cost[cheap]).mean()
                            # restart chosen by same score vs random restart
                            pt["restart_rand_sr"]=(1-frac)*base_c[0]+frac*rep["succ"].mean()*100
                        curve.append(pt)
                    results["policy"].append({"cell":cell,"cheap":cheap,"rich":r_,"tier":tier,"score":sname,"n":n,
                        "always_cheap":base_c,"always_rich":base_r,"oracle":orc,"oracle_k":int(orc_mask.sum()),
                        "band":band.get(cell),"curve":curve,"has_restart":rep is not None})
                    if tier=="MODEL0" and sname=="p_upg":
                        bt=f"{min(band[cell]):.1f}-{max(band[cell]):.1f}" if band.get(cell) else "—"
                        print(f"  POLICY {cheap}->{r_} [{tier}/{sname}] cheap {base_c[0]:.1f}%/${base_c[1]:.4f} rich {base_r[0]:.1f}%/${base_r[1]:.4f} oracle {orc[0]:.1f}%/${orc[1]:.4f} (k={int(orc_mask.sum())}) band={bt}",flush=True)
                        for pt in curve:
                            extra=f" restart {pt['restart_sr']:.1f}%/${pt['restart_cost']:.4f}" if "restart_sr" in pt else ""
                            print(f"     f={pt['frac']:.2f} SR={pt['sr']:.1f}% (rand {pt['rand_sr']:.1f}%) cost=${pt['cost']:.4f} (rand ${pt['rand_cost']:.4f}) beats_rich={pt['beats_rich']}{extra}",flush=True)
json.dump(results,open(OUT,"w"),indent=1)
print("\nLOOKAHEAD_DONE",flush=True)
