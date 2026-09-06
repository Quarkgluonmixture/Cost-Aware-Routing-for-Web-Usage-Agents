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
import json, glob, os, itertools, statistics as st, argparse
from collections import defaultdict

ROOT = "results/visualwebarena/phase1"
REP  = "results/repro_replicates"

# --- cell registry -------------------------------------------------------------
# Each entry needs ALL SIX arms replicated; the 2^6 envelope is not defined on five.
# `clean` = the subset whose replicate is adjacent in time to its canonical run, i.e.
# crosses no material code drift. It is None where no such subset exists — see red_b0.
CELLS = {
 "cls_b0": {
  "site": "classifieds",
  "runs": {
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
  },
  "clean": ["DOM", "Vision"],
  "prov": {"DOM":"archive 05-23 (drift-free)","Vision":"archive 05-25 (drift-free)",
           "SoM":"rerun 08-03 (crosses 24 commits)","P-text":"rerun 08-17 (crosses 24)",
           "P-SoM":"rerun 08-18 (crosses 24)","P-prompt":"rerun 08-17 (crosses 24)"},
 },
 # Second cell, 2026-09-06. The three phantom arms were registered in CLEAN_PAIRS on
 # 2026-08-26; SoM/DOM/Vision landed 2026-09-02/04/06 as the b0_reddit_replicate chain
 # (intent: pre_run/b0_reddit_replicate_chain_launch_intent_20260902.md, Reading 1).
 #
 # ⚠️ NO CLEAN SUBSET EXISTS HERE, and that is a real limitation rather than an
 # oversight: every reddit canonical run is 2026-06/07 and every replicate 2026-08/09,
 # so all six arms cross ~2 months of runtime commits. cls_b0 could isolate two
 # drift-free arms and show that run-to-run noise dominated code drift; reddit cannot
 # make that check on its own evidence. What it DOES have is the same-condition
 # comparison from compare_cross_run_same_condition.py: start_url_mismatch = 0 on all
 # six pairs and every flip classified model_nondeterm.
 "red_b0": {
  "site": "reddit",
  "runs": {
   "DOM":      (f"{ROOT}/B0_dom_reddit_20260625_154833_928747130_2827521_R11344",
                f"{ROOT}/B0_dom_reddit_20260904_010441_409707415_681024_R9525"),
   "SoM":      (f"{ROOT}/B0_som_reddit_20260627_035453_162107997_3024022_R20936",
                f"{ROOT}/B0_som_reddit_20260902_194848_784669986_474818_R11761"),
   "Vision":   (f"{ROOT}/B0_vision_reddit_20260628_094255_184327569_3222015_R17559",
                f"{ROOT}/B0_vision_reddit_20260905_081817_462107405_893625_R17511"),
   "P-text":   (f"{ROOT}/B0_phantom_text_reddit_20260629_140253_060787566_3384189_R32139",
                f"{ROOT}/B0_phantom_text_reddit_20260821_165404_673791669_2663733_R2359"),
   "P-SoM":    (f"{ROOT}/B0_phantom_som_reddit_20260701_223127_661875492_3649813_R28173",
                f"{ROOT}/B0_phantom_som_reddit_20260824_152956_638287802_3173740_R26550"),
   "P-prompt": (f"{ROOT}/B0_phantom_prompt_reddit_20260709",
                f"{ROOT}/B0_phantom_prompt_reddit_20260823_075453_423269667_2958720_R11669"),
  },
  "clean": None,
  "prov": {"DOM":"rerun 09-04 (canonical 06-25, ~2.3 mo)",
           "Vision":"rerun 09-05 (canonical 06-28, ~2.3 mo)",
           "SoM":"rerun 09-02 (canonical 06-27, ~2.2 mo)",
           "P-text":"rerun 08-21 (canonical 06-29, ~1.8 mo)",
           "P-SoM":"rerun 08-24 (canonical 07-01, ~1.8 mo)",
           "P-prompt":"rerun 08-23 (canonical 07-09, ~1.5 mo)"},
 },
}

def _load_ext(run_dir):
    """task_id -> bool success, excluding sr_excluded. Same as load(); defined early
    because --compare runs before the module-level cls pipeline below."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, "*", "episodes", "*_summary_v2.json")):
        try: d = json.load(open(f))
        except Exception: continue
        if d.get("sr_excluded"): continue
        tid = d.get("task_id")
        if tid is None: continue
        out[int(tid)] = bool(d.get("success"))
    return out


_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--cell", choices=sorted(CELLS), default="cls_b0",
                 help="which (site, backbone) cell to compute the envelope on "
                      "(default: cls_b0, the 2026-08-18 original)")
_ap.add_argument("--compare", action="store_true",
                 help="compute EVERY registered cell and emit the cross-side comparison "
                      "to docs/analysis/cross_sites/unique_solve_envelope_cross_cell.md")
_args = _ap.parse_args()

# --- comparison mode: the cross-SIDE reading, across every registered cell ---------
# Split out because the per-cell dump above answers "how much can one arm's unique
# count move", while the hero claim is about whether the VISUAL side's arms stay
# above the TEXT side's once every arm is free to be drawn from either run. That is a
# statement about two GROUPS of lower bounds, and it is only visible with the cells
# side by side.
VISUAL_SIDE = ["SoM", "Vision"]
TEXT_SIDE   = ["P-text", "P-SoM", "P-prompt"]

if _args.compare:
    import subprocess, sys as _sys
    rows = {}
    for _name, _cfg in CELLS.items():
        _S = {m: {"A": None, "B": None} for m in _cfg["runs"]}
        for m, (a, b) in _cfg["runs"].items():
            _S[m]["A"], _S[m]["B"] = _load_ext(a), _load_ext(b)
        _modes = list(_cfg["runs"])
        _common = None
        for m in _modes:
            for v in ("A", "B"):
                ks = set(_S[m][v])
                _common = ks if _common is None else (_common & ks)
        _common = sorted(_common)
        env = defaultdict(list)
        for combo in itertools.product("AB", repeat=len(_modes)):
            asg = dict(zip(_modes, combo))
            for m in _modes:
                c = sum(1 for t in _common
                        if _S[m][asg[m]][t] and not any(_S[o][asg[o]][t]
                                                        for o in _modes if o != m))
                env[m].append(c)
        rows[_name] = {"site": _cfg["site"], "n": len(_common),
                       "min": {m: min(env[m]) for m in _modes},
                       "max": {m: max(env[m]) for m in _modes}}

    out = ["---", "type: analysis", "status: complete",
           f"created: 2026-09-06",
           "purpose: does the cross-SIDE unique-coverage difference survive the "
           "replicate-assignment envelope on more than one cell?",
           "producer: scripts/analysis/unique_solve_noise_envelope.py --compare",
           "---", "",
           "# Cross-side unique coverage, under the 2^6 assignment envelope", "",
           "Regenerate: `.venv/bin/python3 scripts/analysis/unique_solve_noise_envelope.py --compare`", "",
           "Each arm may be drawn from either of its two same-condition runs, so every",
           "cell below is 2^6 = 64 assignments. The number that matters is each arm's",
           "**minimum** unique-solve count over those 64: it is what the arm contributes",
           "that no other arm does, in the least favourable assignment. A lower bound that",
           "can be driven to 0 means the arm has no assignment-robust unique contribution.", "",
           "## Per-arm lower bound (min over 64 assignments)", ""]
    cells = list(rows)
    hdr = "| arm | side | " + " | ".join(f"{c} (n={rows[c]['n']})" for c in cells) + " |"
    out += [hdr, "|---|---|" + "---|" * len(cells)]
    for m in VISUAL_SIDE + TEXT_SIDE + ["DOM"]:
        side = ("visual" if m in VISUAL_SIDE else
                "text" if m in TEXT_SIDE else "text (AXTree, not in either side group)")
        cellvals = " | ".join(f"**{rows[c]['min'][m]}**–{rows[c]['max'][m]}" for c in cells)
        out += [f"| `{m}` | {side} | {cellvals} |"]
    out += ["", "## The comparison the hero rests on", ""]
    out += ["| cell | visual side, lowest bound | text side, highest bound | separation |",
            "|---|---|---|---|"]
    verdicts = {}
    for c in cells:
        vlo = min(rows[c]["min"][m] for m in VISUAL_SIDE)
        thi = max(rows[c]["min"][m] for m in TEXT_SIDE)
        gap = vlo - thi
        verdicts[c] = (vlo, thi, gap)
        varm = min(VISUAL_SIDE, key=lambda m: rows[c]["min"][m])
        tarm = max(TEXT_SIDE, key=lambda m: rows[c]["min"][m])
        out += [f"| {c} ({rows[c]['site']}) | {vlo} (`{varm}`) | {thi} (`{tarm}`) | "
                f"**{gap:+d}** |"]
    out += ["", "## Reading", ""]
    for c in cells:
        vlo, thi, gap = verdicts[c]
        if gap > 0:
            out += [f"- **{c}**: the two sides are separated by {gap} — every visual arm "
                    f"keeps a unique contribution that no assignment of the text arms "
                    f"reaches."]
        elif gap == 0:
            out += [f"- **{c}**: the sides **touch** at {vlo}. The visual side's weakest "
                    f"arm and the text side's strongest arm have the same lower bound, so "
                    f"on this cell 'the visual side contributes more uniquely' is **not** "
                    f"supported arm-by-arm — it holds only for the stronger visual arm."]
        else:
            out += [f"- **{c}**: **inverted** ({gap}). A text arm has a higher "
                    f"assignment-robust unique contribution than the weakest visual arm."]
    out += ["", "⚠️ **Scope.** Both cells are B0. A cell needs all six arms replicated to "
            "appear here, and only B0 has that on two sites. Nothing here licenses a "
            "statement about B1 or B2, whose floors are a different size entirely "
            "(see `serving_mode_floor.md`).", ""]
    dest = "docs/analysis/cross_sites/unique_solve_envelope_cross_cell.md"
    io_open = open(dest, "w")
    io_open.write("\n".join(out))
    io_open.close()
    print("\n".join(out))
    print(f"\n✓ wrote {dest}")
    raise SystemExit(0)

CELL = CELLS[_args.cell]
RUNS = CELL["runs"]
MODES = list(RUNS)
print(f"### cell = {_args.cell}  (site={CELL['site']}, backbone=B0)\n")

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
CLEAN = CELL["clean"]
if CLEAN is None:
    print("CLEAN SUBSET — NOT AVAILABLE for this cell")
    print("="*70)
    print("Every arm's replicate crosses ~1.5-2.3 months of runtime commits, so there is")
    print("no drift-free subset to isolate. The full 2^6 envelope above therefore carries")
    print("run-to-run noise AND code drift together and cannot separate them. Do not")
    print("quote it as a pure noise band; the cls_b0 cell is the only one that showed")
    print("(via its drift-free DOM/Vision arms) that noise dominates drift, and that")
    print("finding is NOT transferable to another site without its own check.")
else:
    print(f"CLEAN SUBSET — perturb only the drift-free arms ({', '.join(CLEAN)}); "
          f"2^{len(CLEAN)} = {2**len(CLEAN)}")
    print("="*70)
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
prov = CELL["prov"]
for m in MODES:
    fl = sum(1 for t in common if S[m]["A"][t] != S[m]["B"][t])
    print(f"{m:9} {fl:6} {100*fl/len(common):6.1f}%  {prov[m]}")
