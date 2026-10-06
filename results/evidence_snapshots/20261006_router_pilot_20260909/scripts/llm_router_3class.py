"""3-class router: READ / LOOK / BOTH, arm-matched, fold-internal representative.

READ = the no-image class {dom, ptext, pprompt, psom}. Represented by ONE arm so
the classes are arm-matched -- taking the union of its four arms is the unfair
comparison the paper flags (four arms against one each).

The representative arm is chosen on the TRAINING folds only. Choosing it on the
full cell would leak the test fold into the option set.
"""
import os, sys, json, glob, random, collections, argparse
import requests
from concurrent.futures import ThreadPoolExecutor
URL="https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke"
READ=["dom","ptext","pprompt","psom"]
CLASS_DESC={
 "READ":"the agent gets the page as text only - an accessibility tree, or a numbered list of the interactive elements. It never sees a picture of the page.",
 "LOOK":"the agent gets a screenshot only. It sees the page the way a person would, but gets no text listing of the elements.",
 "BOTH":"the agent gets a screenshot with numbered boxes drawn on the interactive elements, plus the matching numbered list as text.",
}
def load_tasks(site):
    bench="webarena" if site.startswith("wa_") else "visualwebarena"
    d=glob.glob(f"results/{bench}/phase1/B0_dom_{site}_2026*/task_configs")[0]
    out={}
    for f in glob.glob(os.path.join(d,"*.json")):
        j=json.load(open(f)); t=j.get("task_id")
        if t is None: continue
        out[int(t)]={"intent":j.get("intent",""),
                     "rd":j.get("reasoning_difficulty"),"vd":j.get("visual_difficulty"),
                     "od":j.get("overall_difficulty"),
                     "img":j.get("image") not in (None,"None",""),
                     "ev":",".join((j.get("eval") or {}).get("eval_types") or [])}
    return out
def fmt(t,m):
    b=[f'intent: "{m["intent"]}"']
    for k,l in (("rd","reasoning_difficulty"),("vd","visual_difficulty"),("od","overall_difficulty")):
        if m.get(k): b.append(f"{l}={m[k]}")
    b.append(f"has_reference_image={m['img']}")
    if m.get("ev"): b.append(f"eval={m['ev']}")
    return "; ".join(b)
def ask(model,key,p):
    body={"model":model,"max_tokens":640,"temperature":0,"messages":[{"role":"user","content":p}]}
    for _ in range(3):
        try:
            r=requests.post(URL,headers={"X-Api-Key":key},json=body,timeout=120)
            if r.status_code!=200: continue
            t=(r.json().get("text") or "").strip().upper().strip(".:,`*")
            for c in ("READ","LOOK","BOTH"):
                if t==c or t.startswith(c): return c
            return "__unparsed__:"+t[:12]
        except Exception: continue
    return "__error__"
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cell",required=True); ap.add_argument("--site",required=True)
    ap.add_argument("--model",default="global.openai.gpt-5.6-luna")
    ap.add_argument("--shots",type=int,default=30); ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    key=os.getenv("PROXY_API_KEY") or sys.exit("no key")
    full=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"full_table2.json")))
    avail=[m for m in ["dom","som","vision","ptext","pprompt","psom"] if f"{a.cell}|{m}" in full]
    read=[m for m in READ if m in avail]
    if not read or "som" not in avail: sys.exit("cell lacks READ or BOTH")
    S={m:{int(t):v[0] for t,v in full[f"{a.cell}|{m}"].items()} for m in avail}
    C={m:{int(t):(v[1] or 0) for t,v in full[f"{a.cell}|{m}"].items()} for m in avail}
    L={m:{int(t):(v[2] or 0)/1000 for t,v in full[f"{a.cell}|{m}"].items()} for m in avail}
    meta=load_tasks(a.site)
    tasks=sorted(set.intersection(*[set(S[m]) for m in avail]) & set(meta))
    rng=random.Random(0); order=tasks[:]; rng.shuffle(order)
    folds=[order[i::a.folds] for i in range(a.folds)]
    jobs=[]; reps={}
    for fi,test in enumerate(folds):
        tr=[t for t in tasks if t not in set(test)]
        rep=max(read,key=lambda m:sum(S[m][t] for t in tr))          # fold-internal
        reps[fi]=rep
        cls={"READ":rep,"LOOK":"vision" if "vision" in avail else None,"BOTH":"som"}
        cls={k:v for k,v in cls.items() if v}
        prof="\n".join(
            f"- {k}: {CLASS_DESC[k]}\n    measured here: success "
            f"{sum(S[v][t] for t in tr)/len(tr)*100:.1f}%  cost ${sum(C[v][t] for t in tr)/len(tr):.4f}/episode"
            f"  latency {sum(L[v][t] for t in tr)/len(tr):.0f}s/episode" for k,v in cls.items())
        lab={t:[k for k,v in cls.items() if S[v][t]] for t in tr}
        pool=[t for t in tr if lab[t]]
        for t in test:
            shots=rng.sample(pool,min(a.shots,len(pool))) if pool else []
            ex="\n".join(f"  {fmt(s,meta[s])}\n    -> solved by: {', '.join(lab[s])}" for s in shots)
            p=(f"You are choosing how a web-browsing agent should see the page for one task.\n\n"
               f"The objective is the best trade-off across success, dollar cost and latency - "
               f"not maximum success alone. Only pay for a richer view when this task needs it.\n\n"
               f"{prof}\n\nExamples from other tasks here, with which view actually solved them:\n{ex}\n\n"
               f"Task to route:\n  {fmt(t,meta[t])}\n\nAnswer with exactly one of READ, LOOK, BOTH.")
            jobs.append((t,p))
    print(f"cell={a.cell} n={len(tasks)} reps={reps}",flush=True)
    with ThreadPoolExecutor(max_workers=10) as ex:
        res=dict(zip([j[0] for j in jobs], ex.map(lambda j: ask(a.model,key,j[1]), jobs)))
    json.dump({"cell":a.cell,"reps":reps,"folds":[list(f) for f in folds],"choices":res},
              open(a.out,"w"),indent=1)
    print("  ",dict(collections.Counter(res.values()).most_common()),flush=True)
