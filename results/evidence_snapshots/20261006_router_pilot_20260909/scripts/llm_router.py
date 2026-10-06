"""Zero-shot LLM router: task intent -> observation mode.

Why this is not one of the paper's five policies: those all need a which-mode
label to fit, and the paper's central obstruction is that such labels are
produced only at the agent's success rate. A zero-shot LLM router needs none.
So it tests whether the supply-value coupling is the binding constraint, or
whether the signal simply isn't in the intent text.

Router model is GPT-5.6 (terra/luna) = NOT any of the routed backbones
(B0 Qwen3-VL-235B / B1 Qwen3-VL-4B / B2 Gemma-3-4B), so no self-routing.
"""
import os, sys, json, csv, glob, collections, random, argparse
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke"
MODES = ["dom","som","vision","ptext","pprompt","psom"]

# Neutral descriptions, no hint about which is better. Sourced from
# p79/experiment/som.py mode branches + CLAUDE.md 关键变量.
PROMPT = """You are routing a web-browsing agent. For the task below, choose which observation format the agent should receive on each step.

Options:
- dom: the page's accessibility tree, as text. No image.
- som: an annotated screenshot with numbered boxes drawn on interactive elements, plus the numbered element list as text.
- vision: a raw screenshot only, with no text description of the page.
- ptext: the numbered element list as text, with instructions phrased for a text-only agent. No image.
- pprompt: the accessibility tree as text, with instructions phrased for an agent that expects an annotated screenshot. No image.
- psom: the numbered element list as text, with instructions phrased for an agent that expects an annotated screenshot. No image.

Task: {intent}

Answer with exactly one option name from the list above and nothing else."""

def load_intents(site):
    pat = {"classifieds":"B0_dom_classifieds_20260525_194618_553890342_530647_R21557",
           "reddit":"B0_dom_reddit_*", "shopping":"B0_dom_shopping_*", "wa_reddit":"B0_dom_wa_reddit_2*"}[site]
    bench = "webarena" if site.startswith("wa_") else "visualwebarena"
    ds = glob.glob(f"results/{bench}/phase1/{pat}/task_configs")
    if not ds: sys.exit(f"no task_configs for {site}")
    out = {}
    for f in glob.glob(os.path.join(ds[0], "*.json")):
        tid = int(os.path.basename(f).rsplit("_",1)[1].split(".")[0])
        out[tid] = json.load(open(f)).get("intent","")
    return out

def ask(model, intent, key, retries=3):
    body = {"model": model, "max_tokens": 512, "temperature": 0,
            "messages":[{"role":"user","content":PROMPT.format(intent=intent)}]}
    for _ in range(retries):
        try:
            r = requests.post(URL, headers={"X-Api-Key":key,"Content-Type":"application/json"},
                              json=body, timeout=90)
            if r.status_code != 200: continue
            j = r.json()
            txt = (j.get("text") or "").strip().lower().strip(".:,`*")
            cost = float(j.get("usage",{}).get("cost",0) or 0)
            for m in MODES:
                if txt == m or txt.startswith(m): return m, cost
            return ("__unparsed__:"+txt[:20]), cost
        except Exception:
            continue
    return "__error__", 0.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="classifieds")
    ap.add_argument("--model", default="global.openai.gpt-5.6-luna")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--modes", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = os.getenv("PROXY_API_KEY","")
    if not key: sys.exit("PROXY_API_KEY not set")
    if a.modes:
        keep=a.modes.split(",")
        lines=[l for l in PROMPT.split("\n") if not l.startswith("- ") or l.split(":")[0][2:] in keep]
        PROMPT="\n".join(lines); MODES=keep
    intents = load_intents(a.site)
    tids = sorted(intents)[:a.limit] if a.limit else sorted(intents)
    print(f"{a.site}: {len(tids)} tasks, model={a.model}", flush=True)
    res, costs = {}, []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for tid, (m, c) in zip(tids, ex.map(lambda t: ask(a.model, intents[t], key), tids)):
            res[tid] = m; costs.append(c)
    json.dump({"site":a.site,"model":a.model,"choices":res,
               "total_cost_usd":sum(costs)}, open(a.out,"w"), indent=1)
    d = collections.Counter(res.values())
    print(f"  cost ${sum(costs):.4f}  choices: {dict(d.most_common())}", flush=True)
