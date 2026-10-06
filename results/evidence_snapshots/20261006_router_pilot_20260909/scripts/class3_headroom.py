"""3-class (READ / LOOK / BOTH) routing headroom, both matched and unmatched.

READ = {dom, ptext, pprompt, psom}  LOOK = {vision}  BOTH = {som}
UNMATCHED gives READ four arms against one each -- the exact comparison the
paper flags as unfair. ARM-MATCHED picks one representative arm for READ.
"""
import json, os, collections, itertools
SP=os.path.dirname(os.path.abspath(__file__))
MODES=["dom","som","vision","ptext","pprompt","psom"]
READ=["dom","ptext","pprompt","psom"]
full=json.load(open(os.path.join(SP,"full_table2.json")))
cells=collections.defaultdict(dict)
for k,v in full.items():
    c,m=k.split("|"); cells[c][m]={int(t):v[t] for t in v}

print(f"{'cell':<12}{'n':>5} | {'6-mode':>18} | {'3-class UNMATCHED':>22} | {'3-class ARM-MATCHED':>26}")
print(f"{'':<12}{'':>5} | {'best':>8}{'oracle':>10} | {'best':>10}{'oracle':>12} | {'READ rep':>10}{'best':>8}{'oracle':>8}")
print("-"*104)
for cid in sorted(cells):
    ms=[m for m in MODES if m in cells[cid]]
    if len(ms)<5: continue
    tasks=sorted(set.intersection(*[set(cells[cid][m]) for m in ms]))
    if len(tasks)<50: continue
    n=len(tasks)
    S={m:{t:cells[cid][m][t][0] for t in tasks} for m in ms}
    sr={m:sum(S[m].values())/n*100 for m in ms}
    orac6=sum(1 for t in tasks if any(S[m][t] for m in ms))/n*100
    read=[m for m in READ if m in ms]
    # unmatched: READ solves if ANY of its arms solves
    U={"READ":{t:any(S[m][t] for m in read) for t in tasks},
       "LOOK":{t:S["vision"][t] for t in tasks} if "vision" in S else None,
       "BOTH":{t:S["som"][t] for t in tasks} if "som" in S else None}
    U={k:v for k,v in U.items() if v}
    usr={k:sum(v.values())/n*100 for k,v in U.items()}
    oracU=sum(1 for t in tasks if any(U[k][t] for k in U))/n*100
    # arm-matched: READ represented by its single best arm
    rep=max(read,key=lambda m:sr[m])
    M={"READ":S[rep]}
    if "vision" in S: M["LOOK"]=S["vision"]
    if "som" in S: M["BOTH"]=S["som"]
    msr={k:sum(v.values())/n*100 for k,v in M.items()}
    oracM=sum(1 for t in tasks if any(M[k][t] for k in M))/n*100
    print(f"{cid:<12}{n:>5} | {max(sr.values()):>7.2f}%{orac6:>9.2f}% | "
          f"{max(usr.values()):>9.2f}%{oracU:>11.2f}% | {rep:>10}{max(msr.values()):>7.2f}%{oracM:>7.2f}%")
