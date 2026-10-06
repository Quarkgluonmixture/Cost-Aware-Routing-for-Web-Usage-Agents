"""LLM router v2 — every label and feature the project has, not just the intent.

v1 gave the model one line of intent text and nothing else; that is the weakest
possible router and its failure proves little. v2 gives:
  per-task   : intent, template id, reasoning/visual/overall difficulty, has-image,
               eval type, require_login
  per-mode   : all 40 measured failure-rule frequencies + base success rate
  supervision: K held-out few-shot examples (task -> mode that solved it)

The few-shot pool is drawn ONLY from tasks where some mode succeeded, because
that is the only place a which-mode label exists. That constraint is the
supply-value coupling itself, so v2 is subject to it exactly as a trained
router would be -- unlike v1.

5-fold: for each fold, few-shot examples come from the other four folds.
"""
import os, sys, json, glob, csv, random, collections, argparse
import requests
from concurrent.futures import ThreadPoolExecutor

URL="https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke"
CANON={"dom":"dom","som":"som","vision":"vision",
       "phantom_text":"ptext","phantom_prompt":"pprompt","phantom_som":"psom"}
MODES=["dom","som","vision","ptext","pprompt","psom"]
DESC={
 "dom":"accessibility tree as text. No image.",
 "som":"annotated screenshot with numbered boxes on interactive elements, plus the numbered element list as text.",
 "vision":"raw screenshot only, no text description of the page.",
 "ptext":"numbered element list as text, instructions phrased for a text-only agent. No image.",
 "pprompt":"accessibility tree as text, instructions phrased for an agent expecting an annotated screenshot. No image.",
 "psom":"numbered element list as text, instructions phrased for an agent expecting an annotated screenshot. No image.",
}

def load_tasks(site="classifieds"):
    d=glob.glob(f"results/visualwebarena/phase1/B0_dom_{site}_2026*/task_configs")[0]
    out={}
    for f in glob.glob(os.path.join(d,"*.json")):
        j=json.load(open(f)); tid=j.get("task_id")
        if tid is None: continue
        out[int(tid)]={
            "intent": j.get("intent",""),
            "template": j.get("intent_template_id"),
            "reason_d": j.get("reasoning_difficulty"),
            "visual_d": j.get("visual_difficulty"),
            "overall_d": j.get("overall_difficulty"),
            "has_image": j.get("image") not in (None,"None",""),
            "eval": ",".join((j.get("eval") or {}).get("eval_types") or []),
            "login": bool(j.get("require_login")),
        }
    return out

def build_mode_block(cell, baserate=True):
    d=json.load(open("docs/analysis/cross_sites/cross_mode_failure_signatures.json"))
    rules=[r for r in d["part_a_signature_frequency"]["rules"] if r.get("per_mode_pct")]
    rules.sort(key=lambda r:-r.get("spread_all_modes_pp",0))
    full=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"full_table.json")))
    sr={}
    for m in MODES:
        k=f"{cell}|{m}"
        if k in full:
            v=full[k]; sr[m]=sum(1 for x in v.values() if x[0])/len(v)*100
    lines=["Observation modes available, with measured behaviour from prior runs on this benchmark:"]
    for m in MODES:
        lines.append(f"- {m}: {DESC[m]}")
        if baserate:
            lines.append(f"    overall success rate on this backbone+site: {sr.get(m,float('nan')):.1f}%")
    lines.append("")
    lines.append(f"Measured failure-mode frequencies per mode (all {len(rules)} rules the "
                 "project's failure taxonomy defines; % of failed episodes hitting each rule):")
    for r in rules:
        pm=r["per_mode_pct"]
        cells=" ".join(f"{CANON.get(k,k)}={pm.get(k,0):.0f}%" for k in
                       ["dom","som","vision","phantom_text","phantom_prompt","phantom_som"])
        lines.append(f"  {r['rule_id']} {r['rule_name']}: {cells}")
    return "\n".join(lines)

def fmt_task(t, meta):
    f=meta
    bits=[f'intent: "{f["intent"]}"']
    if f.get("template") is not None: bits.append(f'template_id={f["template"]}')
    for k,lab in (("reason_d","reasoning_difficulty"),("visual_d","visual_difficulty"),
                  ("overall_d","overall_difficulty")):
        if f.get(k): bits.append(f'{lab}={f[k]}')
    bits.append(f'has_reference_image={f["has_image"]}')
    if f.get("eval"): bits.append(f'eval={f["eval"]}')
    bits.append(f'requires_login={f["login"]}')
    return "; ".join(bits)

def ask(model,key,prompt):
    body={"model":model,"max_tokens":768,"temperature":0,
          "messages":[{"role":"user","content":prompt}]}
    for _ in range(3):
        try:
            r=requests.post(URL,headers={"X-Api-Key":key},json=body,timeout=120)
            if r.status_code!=200: continue
            t=(r.json().get("text") or "").strip().lower().strip(".:,`*")
            for m in MODES:
                if t==m or t.startswith(m): return m
            return "__unparsed__:"+t[:15]
        except Exception: continue
    return "__error__"

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cell",default="cls_B0"); ap.add_argument("--site",default="classifieds")
    ap.add_argument("--model",default="global.openai.gpt-5.6-luna")
    ap.add_argument("--shots",type=int,default=30); ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--no-baserate",action="store_true")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    key=os.getenv("PROXY_API_KEY") or sys.exit("no key")
    meta=load_tasks(a.site)
    full=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"full_table.json")))
    succ={m:{int(t):v[0] for t,v in full[f"{a.cell}|{m}"].items()} for m in MODES
          if f"{a.cell}|{m}" in full}
    tasks=sorted(set.intersection(*[set(v) for v in succ.values()]) & set(meta))
    # label exists only where some mode succeeded
    label={t:[m for m in MODES if succ[m][t]] for t in tasks}
    labelled=[t for t in tasks if label[t]]
    print(f"cell={a.cell} n={len(tasks)} labelled={len(labelled)} "
          f"({len(labelled)/len(tasks)*100:.1f}%)  shots={a.shots} folds={a.folds}",flush=True)
    modeblock=build_mode_block(a.cell, baserate=not a.no_baserate)
    rng=random.Random(0); order=tasks[:]; rng.shuffle(order)
    folds=[order[i::a.folds] for i in range(a.folds)]
    jobs=[]
    for fi,test in enumerate(folds):
        pool=[t for t in labelled if t not in set(test)]
        for t in test:
            shots=rng.sample(pool, min(a.shots,len(pool)))
            ex="\n".join(f"  {fmt_task(s,meta[s])}\n    -> solved by: {', '.join(label[s])}"
                         for s in shots)
            p=(f"You are choosing which observation format a web-browsing agent should receive "
               f"for a task. Pick the one most likely to solve it.\n\n{modeblock}\n\n"
               f"Examples from other tasks on this same site and backbone, showing which "
               f"mode(s) actually solved them:\n{ex}\n\n"
               f"Now the task to route:\n  {fmt_task(t,meta[t])}\n\n"
               f"Answer with exactly one mode name from the list and nothing else.")
            jobs.append((t,p))
    print(f"  prompt chars ~{len(jobs[0][1])}",flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        res=dict(zip([j[0] for j in jobs], ex.map(lambda j: ask(a.model,key,j[1]), jobs)))
    json.dump({"cell":a.cell,"model":a.model,"shots":a.shots,"choices":res},open(a.out,"w"),indent=1)
    print("  ",dict(collections.Counter(res.values()).most_common()),flush=True)
