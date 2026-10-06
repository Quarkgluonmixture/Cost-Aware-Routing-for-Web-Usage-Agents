"""Same decision, three deciders: does THIS task need the image?

image arm  = best of {som, vision};  no-image arm = cheapest of {dom,ptext,pprompt,psom}
oracle : knows which side solves it (upper bound)
regex  : the paper's 0-token visual-intent predicate  (free)
llm    : implied by v3's 6-way pick (som/vision => image)   (paid)
"""
import json,os,re,glob,collections
SP=os.path.dirname(os.path.abspath(__file__))
M=["dom","som","vision","ptext","pprompt","psom"]; NOIMG=["dom","ptext","pprompt","psom"]
RE=re.compile(r"\b(image|picture|photo|screenshot)\b|\bcolou?r of\b|"
              r"\bhow many\b[^.]{0,40}\bin (?:the|this)\b", re.IGNORECASE)
full=json.load(open(os.path.join(SP,"full_table2.json")))
def intents(site):
    b="webarena" if site.startswith("wa_") else "visualwebarena"
    d=glob.glob(f"results/{b}/phase1/B0_dom_{site}_2026*/task_configs")
    return {int(os.path.basename(f).rsplit("_",1)[1].split(".")[0]):
            json.load(open(f)).get("intent","") for f in glob.glob(os.path.join(d[0],"*.json"))} if d else {}
SITE={"cls":"classifieds","red":"reddit","shop":"shopping"}
print(f"{'cell':<11}{'decider':<9}{'要图%':>7}{'SR%':>7}{'cost$':>9}{'Δcost':>8}"
      f"{'vs always-img SR':>17}{'agree w/ regex':>15}")
print("-"*88)
for cid in sorted(set(k.split("|")[0] for k in full)):
    ms=[m for m in M if f"{cid}|{m}" in full]
    if len(ms)<5: continue
    site="wa_reddit" if cid.endswith("_WA") else SITE.get(cid.split("_")[0])
    I=intents(site)
    if not I: continue
    S={m:{int(t):full[f"{cid}|{m}"][t][0] for t in full[f"{cid}|{m}"]} for m in ms}
    C={m:{int(t):(full[f"{cid}|{m}"][t][1] or 0) for t in full[f"{cid}|{m}"]} for m in ms}
    T=sorted(set.intersection(*[set(S[m]) for m in ms]) & set(I))
    if len(T)<50: continue
    n=len(T)
    sr={m:sum(S[m][t] for t in T)/n*100 for m in ms}
    IMG=max([m for m in ("som","vision") if m in ms],key=lambda m:sr[m])
    TXT=min([m for m in NOIMG if m in ms],key=lambda m:sum(C[m][t] for t in T)/n)
    base_sr=sr[IMG]; base_c=sum(C[IMG][t] for t in T)/n
    v3=os.path.join(SP,f"v3all_{cid}.json")
    llm={}
    if os.path.exists(v3):
        ch={int(k):v for k,v in json.load(open(v3))["choices"].items()}
        llm={t:(ch[t] in ("som","vision")) for t in T if ch.get(t) in ms}
    dec={"oracle":{t:(S[IMG][t] or not S[TXT][t]) for t in T},
         "regex":{t:bool(RE.search(I[t])) for t in T},
         "llm":llm}
    rg=dec["regex"]
    for name in ("oracle","regex","llm"):
        d=dec[name]
        use=[t for t in T if t in d]
        if len(use)<50: continue
        pick={t:(IMG if d[t] else TXT) for t in use}
        s=sum(S[pick[t]][t] for t in use)/len(use)*100
        c=sum(C[pick[t]][t] for t in use)/len(use)
        b_s=sum(S[IMG][t] for t in use)/len(use)*100
        b_c=sum(C[IMG][t] for t in use)/len(use)
        ag=sum(1 for t in use if d[t]==rg[t])/len(use)*100
        print(f"{cid if name=='oracle' else '':<11}{name:<9}"
              f"{sum(d[t] for t in use)/len(use)*100:>6.0f}%{s:>6.2f}%{c:>9.4f}"
              f"{(c-b_c)/b_c*100:>+7.1f}%{s-b_s:>+16.2f}pp"
              f"{(f'{ag:.0f}%' if name!='regex' else '—'):>15}")
    print(f"{'':<11}{'(img='+IMG+', txt='+TXT+')':<9}")
