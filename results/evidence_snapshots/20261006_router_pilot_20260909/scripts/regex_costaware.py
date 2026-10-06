"""The paper's 0-token visual-intent regex, judged on cost instead of on SR.

Table 19 killed this policy because it "still loses to always-Vision -- the
screenshot does not hurt on the unflagged tasks either". True on the success
axis. But on the unflagged tasks the agent is paying for an image it does not
need, and that is a cost result the SR-only comparison cannot see.

flagged  -> the strongest image-bearing arm
unflagged-> the cheapest arm (two variants: cheapest overall / cheapest no-image)
No labels, no training, no per-episode signal.
"""
import json,os,re,glob,collections
SP=os.path.dirname(os.path.abspath(__file__))
M=["dom","som","vision","ptext","pprompt","psom"]
NOIMG=["dom","ptext","pprompt","psom"]
RE=re.compile(r"\b(image|picture|photo|screenshot)\b|\bcolou?r of\b|"
              r"\bhow many\b[^.]{0,40}\bin (?:the|this)\b", re.IGNORECASE)
full=json.load(open(os.path.join(SP,"full_table2.json")))
def intents(site):
    bench="webarena" if site.startswith("wa_") else "visualwebarena"
    d=glob.glob(f"results/{bench}/phase1/B0_dom_{site}_2026*/task_configs")
    if not d: return {}
    return {int(os.path.basename(f).rsplit("_",1)[1].split(".")[0]):
            json.load(open(f)).get("intent","") for f in glob.glob(os.path.join(d[0],"*.json"))}
SITE={"cls":"classifieds","red":"reddit","shop":"shopping"}
def dom_(p,q): return q[0]>=p[0] and q[1]<=p[1] and q[2]<=p[2] and (q[0]>p[0] or q[1]<p[1] or q[2]<p[2])
print(f"{'cell':<11}{'policy':<22}{'SR%':>7}{'Δbest':>8}{'cost$':>9}{'Δcost':>8}"
      f"{'lat s':>8}{'strictly dominates':>22}")
print("-"*95)
for cid in sorted(set(k.split("|")[0] for k in full)):
    ms=[m for m in M if f"{cid}|{m}" in full]
    if len(ms)<5: continue
    site=SITE.get(cid.split("_")[0]); 
    if cid.endswith("_WA"): site="wa_reddit"
    I=intents(site)
    if not I: continue
    S={m:{int(t):full[f"{cid}|{m}"][t][0] for t in full[f"{cid}|{m}"]} for m in ms}
    C={m:{int(t):(full[f"{cid}|{m}"][t][1] or 0) for t in full[f"{cid}|{m}"]} for m in ms}
    L={m:{int(t):(full[f"{cid}|{m}"][t][2] or 0)/1000 for t in full[f"{cid}|{m}"]} for m in ms}
    T=sorted(set.intersection(*[set(S[m]) for m in ms]) & set(I))
    if len(T)<50: continue
    n=len(T)
    sr={m:sum(S[m][t] for t in T)/n*100 for m in ms}
    best=max(sr,key=sr.get)
    fx={m:(sr[m],sum(C[m][t] for t in T)/n,sum(L[m][t] for t in T)/n) for m in ms}
    base=fx[best]
    flag=[t for t in T if RE.search(I[t])]
    rich=max([m for m in ms if m in ("som","vision")],key=lambda m:sr[m])
    for tag,pool in (("regex: cheap=any",ms),("regex: cheap=no-image",[m for m in NOIMG if m in ms])):
        if not pool: continue
        cheap=min(pool,key=lambda m:fx[m][1])
        pick={t:(rich if t in set(flag) else cheap) for t in T}
        R=(sum(S[pick[t]][t] for t in T)/n*100,
           sum(C[pick[t]][t] for t in T)/n, sum(L[pick[t]][t] for t in T)/n)
        beats=[m for m in ms if dom_(fx[m],R)]
        print(f"{cid:<11}{tag+f' [{rich}/{cheap}]':<22}{R[0]:>6.2f}%{R[0]-base[0]:>+8.2f}"
              f"{R[1]:>9.4f}{(R[1]-base[1])/base[1]*100:>+7.1f}%{R[2]:>8.1f}"
              f"{(', '.join(beats) if beats else '—'):>22}")
    print(f"{'':<11}{'(always-'+best+')':<22}{base[0]:>6.2f}%{'':>8}{base[1]:>9.4f}{'':>8}{base[2]:>8.1f}"
          f"{'  flagged '+str(len(flag))+'/'+str(n):>22}")
