"""Noise envelope for cross-mode oracle metrics (unique solves / drop-one).

WHY THIS EXISTS
---------------
`replicate_metric_noise.json` checks 26 metrics against the measured rerun band but
EXCLUDES exactly one:

    {"metric": "n_unique_solves", "excluded": "cross-mode by construction"}

That exemption was correct when only some arms had a replicate: you cannot perturb a
cross-mode metric by replicating one arm. It stops being correct once EVERY arm in a
cell has a same-condition replicate — then the 2^N assignment envelope (each arm
independently drawn from run A or run B) IS the band.

cls-B0 reached that state on 2026-08-18 when the floor chain landed replicates for the
three phantom arms. This script computes the envelope. See 实验笔记 §470.

CAVEAT ON DRIFT
---------------
Only DOM and Vision have replicates adjacent in time to their canonical run. The other
four B-runs are 2026-08 reruns that cross ~24 runtime commits. Sanity check in the
output: the drift-free arms do NOT show lower flip rates than the drift-crossing ones
(Vision, the cleanest at 1.5 days apart, is the HIGHEST at 14.3%), so run-to-run noise
dominates code drift. The CLEAN SUBSET section reports the drift-free lower bound.

Register new replicates in aggregate_noise_floor_inventory.py CLEAN_PAIRS as well —
that is the canonical registry; this script keeps its own map because it needs all six
arms of one cell, which CLEAN_PAIRS does not yet carry.

Usage:  .venv/bin/python3 scripts/analysis/unique_solve_noise_envelope.py
"""
import json, glob, os, itertools, statistics as st
from collections import defaultdict

ROOT = "results/visualwebarena/phase1"
REP  = "results/repro_replicates"

RUNS = {  # mode -> (runA_dir, runB_dir)
 "DOM":      (f"{ROOT}/B0_dom_classifieds_20260525_194618_553890342_530647_R21557",
              f"{REP}/B0_dom_classifieds_R31194_clean_replicate"),
 "SoM":      (f"{ROOT}/B0_som_classifieds_20260526_041601_863239369_602235_R5313",
              f"{ROOT}/B0_som_classifieds_20260803_084743_413015398_3677519_R30696"),
 "Vision":   (f"{ROOT}/B0_vision_classifieds_20260526_141916_610351680_689390_R32024",
              f"{REP}/B0_vision_classifieds_R24792_clean_replicate"),
 "P-text":   (f"{ROOT}/B0_phantom_text_classifieds_20260526_233303_901232655_764510_R31183",
              f"{ROOT}/B0_phantom_text_classifieds_20260817_092244_763693821_1962797_R20043"),
 "P-SoM":    (f"{ROOT}/B0_phantom_som_classifieds_20260527_191300_844420226_914570_R32031",
              f"{ROOT}/B0_phantom_som_classifieds_20260818_040525_430521618_2113605_R13257"),
 "P-prompt": (f"{ROOT}/B0_phantom_prompt_classifieds_20260528_040546_107246795_987141_R14655",
              f"{ROOT}/B0_phantom_prompt_classifieds_20260817_184335_813828144_2037698_R12207"),
}
MODES = list(RUNS)

def load(run_dir):
    """task_id -> bool success, excluding sr_excluded."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, "*", "episodes", "*_summary_v2.json")):
        try: d = json.load(open(f))
        except Exception: continue
        if d.get("sr_excluded"): continue
        tid = d.get("task_id")
        if tid is None: continue
        out[int(tid)] = bool(d.get("success"))
    return out

S = {m: {"A": load(a), "B": load(b)} for m, (a, b) in RUNS.items()}

# common scored task set across all 12 runs
common = None
for m in MODES:
    for v in ("A", "B"):
        ks = set(S[m][v])
        common = ks if common is None else (common & ks)
common = sorted(common)
print(f"common scored tasks across all 12 runs: n={len(common)}\n")

print("=== per-arm SR + replicate drift ===")
print(f"{'mode':9} {'A solved':>9} {'B solved':>9} {'Δnet':>6} {'flips':>6} {'ΔSR pp':>7}")
for m in MODES:
    a = [t for t in common if S[m]["A"][t]]; b = [t for t in common if S[m]["B"][t]]
    fl = sum(1 for t in common if S[m]["A"][t] != S[m]["B"][t])
    print(f"{m:9} {len(a):9} {len(b):9} {len(b)-len(a):+6} {fl:6} {100*(len(b)-len(a))/len(common):+7.2f}")

def uniques(assign):
    """assign: mode->'A'/'B'. returns mode->unique count on `common`."""
    u = {}
    for m in MODES:
        c = 0
        for t in common:
            if S[m][assign[m]][t] and not any(S[o][assign[o]][t] for o in MODES if o != m):
                c += 1
        u[m] = c
    return u

allA = uniques({m: "A" for m in MODES})
allB = uniques({m: "B" for m in MODES})
print("\n=== unique solves: all-A vs all-B (both are legitimate single-run readings) ===")
print(f"{'mode':9} {'all-A':>6} {'all-B':>6} {'Δ':>5}")
for m in MODES:
    print(f"{m:9} {allA[m]:6} {allB[m]:6} {allB[m]-allA[m]:+5}")
print(f"{'TOTAL':9} {sum(allA.values()):6} {sum(allB.values()):6} {sum(allB.values())-sum(allA.values()):+5}")

print("\n=== swap ONE arm to its replicate, keep other five (6 perturbations) ===")
print(f"{'swapped':9} | " + " ".join(f"{m:>8}" for m in MODES))
print("-"*(12+9*len(MODES)))
print(f"{'(none)':9} | " + " ".join(f"{allA[m]:8}" for m in MODES))
swings = defaultdict(list)
for sw in MODES:
    asg = {m: "A" for m in MODES}; asg[sw] = "B"
    u = uniques(asg)
    print(f"{sw:9} | " + " ".join(f"{u[m]:8}" for m in MODES))
    for m in MODES: swings[m].append(u[m] - allA[m])

print("\n=== per-mode unique swing from a SINGLE-arm replicate swap ===")
print(f"{'mode':9} {'baseline':>8} {'min':>5} {'max':>5} {'range':>6}")
for m in MODES:
    lo, hi = min(swings[m]), max(swings[m])
    print(f"{m:9} {allA[m]:8} {allA[m]+lo:5} {allA[m]+hi:5} {hi-lo:6}")

# full 2^6 envelope
print("\n=== full 2^6 = 64 assignment envelope (every arm independently A or B) ===")
env = defaultdict(list)
for combo in itertools.product("AB", repeat=len(MODES)):
    u = uniques(dict(zip(MODES, combo)))
    for m in MODES: env[m].append(u[m])
print(f"{'mode':9} {'obs(all-A)':>10} {'min':>5} {'max':>5} {'range':>6} {'mean':>6} {'sd':>5}")
for m in MODES:
    v = env[m]
    print(f"{m:9} {allA[m]:10} {min(v):5} {max(v):5} {max(v)-min(v):6} {st.mean(v):6.1f} {st.stdev(v):5.2f}")

# ---- CLEAN SUBSET: only DOM + Vision have a code-drift-free replicate ----
# Their B runs are the archive replicates (05-23 / 05-25), adjacent to the A runs.
# The other four B runs are 2026-08 reruns that cross 24 runtime commits incl.
# B0 proxy response-shape fixes (B-1970/1979/1980) and retry budget (B-1880/1881).
print("\n\n" + "="*70)
print("CLEAN SUBSET — perturb only the two drift-free arms (DOM, Vision); 2^2 = 4")
print("="*70)
CLEAN = ["DOM", "Vision"]
envc = defaultdict(list)
for combo in itertools.product("AB", repeat=len(CLEAN)):
    asg = {m: "A" for m in MODES}
    asg.update(dict(zip(CLEAN, combo)))
    u = uniques(asg)
    for m in MODES: envc[m].append(u[m])
print(f"{'mode':9} {'obs(all-A)':>10} {'min':>5} {'max':>5} {'range':>6}  drift-free?")
for m in MODES:
    v = envc[m]
    tag = "yes (perturbed)" if m in CLEAN else "held at A"
    print(f"{m:9} {allA[m]:10} {min(v):5} {max(v):5} {max(v)-min(v):6}  {tag}")

print("\n=== flip rate: drift-free arms vs 2026-08 rerun arms ===")
print(f"{'mode':9} {'flips':>6} {'%':>7}  replicate provenance")
prov = {"DOM":"archive 05-23 (drift-free)","Vision":"archive 05-25 (drift-free)",
        "SoM":"rerun 08-03 (crosses 24 commits)","P-text":"rerun 08-17 (crosses 24)",
        "P-SoM":"rerun 08-18 (crosses 24)","P-prompt":"rerun 08-17 (crosses 24)"}
for m in MODES:
    fl = sum(1 for t in common if S[m]["A"][t] != S[m]["B"][t])
    print(f"{m:9} {fl:6} {100*fl/len(common):6.1f}%  {prov[m]}")
