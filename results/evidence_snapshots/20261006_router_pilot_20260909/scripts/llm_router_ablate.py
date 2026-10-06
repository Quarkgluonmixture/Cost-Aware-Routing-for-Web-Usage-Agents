"""Position/描述 ablation for the zero-shot LLM router.

If the router's 85-89% preference for `som` survives shuffling the option
order, it is a judgement about the task. If it moves, it was an artefact of
how the options were laid out and the headline table is void.

variant `rev`   : option list reversed (som moves from 2nd to 5th)
variant `alpha` : options relabelled A-F, descriptions kept, order shuffled
"""
import os, sys, json, glob, collections, argparse, random
import requests
from concurrent.futures import ThreadPoolExecutor
URL="https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke"
DESC={
 "dom":"the page's accessibility tree, as text. No image.",
 "som":"an annotated screenshot with numbered boxes drawn on interactive elements, plus the numbered element list as text.",
 "vision":"a raw screenshot only, with no text description of the page.",
 "ptext":"the numbered element list as text, with instructions phrased for a text-only agent. No image.",
 "pprompt":"the accessibility tree as text, with instructions phrased for an agent that expects an annotated screenshot. No image.",
 "psom":"the numbered element list as text, with instructions phrased for an agent that expects an annotated screenshot. No image.",
}
MODES=list(DESC)

def build(order, alpha):
    if alpha:
        letters="ABCDEF"
        opts="\n".join(f"- {letters[i]}: {DESC[m]}" for i,m in enumerate(order))
        tail=("Answer with exactly one letter from the list above and nothing else.")
        return opts, tail, {letters[i]:m for i,m in enumerate(order)}
    opts="\n".join(f"- {m}: {DESC[m]}" for m in order)
    return opts, "Answer with exactly one option name from the list above and nothing else.", {m:m for m in order}

def ask(model,intent,key,opts,tail,mapping):
    p=(f"You are routing a web-browsing agent. For the task below, choose which observation "
       f"format the agent should receive on each step.\n\n{opts}\n\nTask: {intent}\n\n{tail}")
    body={"model":model,"max_tokens":512,"temperature":0,"messages":[{"role":"user","content":p}]}
    for _ in range(3):
        try:
            r=requests.post(URL,headers={"X-Api-Key":key},json=body,timeout=90)
            if r.status_code!=200: continue
            t=(r.json().get("text") or "").strip().strip(".:,`*")
            for k,v in mapping.items():
                if t==k or t.lower()==k.lower() or t.lower().startswith(k.lower()): return v
            return "__unparsed__:"+t[:15]
        except Exception: continue
    return "__error__"

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--variant",choices=["rev","alpha"],required=True)
    ap.add_argument("--model",default="global.openai.gpt-5.6-luna")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    key=os.getenv("PROXY_API_KEY") or sys.exit("no key")
    d=glob.glob("results/visualwebarena/phase1/B0_dom_classifieds_20260525_194618_553890342_530647_R21557/task_configs")[0]
    intents={int(os.path.basename(f).rsplit("_",1)[1].split(".")[0]): json.load(open(f)).get("intent","")
             for f in glob.glob(os.path.join(d,"*.json"))}
    order=list(reversed(MODES)) if a.variant=="rev" else random.Random(7).sample(MODES,len(MODES))
    opts,tail,mapping=build(order, a.variant=="alpha")
    print(f"variant={a.variant} order={order}",flush=True)
    tids=sorted(intents)
    with ThreadPoolExecutor(max_workers=6) as ex:
        res=dict(zip(tids, ex.map(lambda t: ask(a.model,intents[t],key,opts,tail,mapping), tids)))
    json.dump({"variant":a.variant,"order":order,"choices":res},open(a.out,"w"),indent=1)
    print("  ",dict(collections.Counter(res.values()).most_common()),flush=True)
