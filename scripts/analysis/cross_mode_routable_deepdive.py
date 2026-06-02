#!/usr/bin/env python3
"""Cross-mode routable deep-dive — extends cross_mode_failure_taxonomy.py (§291/§306).

§306 first-look gave the headline (routable 88/224 = 39% / 6-mode oracle 43.3% / +16pp
over best-single som). This script DEEPENS that first-look into router-actionable signal:

  A. Routable decomposition  — split the 88 routable into exclusive (k=1 mode solves,
     "must-route") vs shared (k>=2, "route-forgiving") + the k=1..5 solve-count histogram.
  B. Oracle marginal         — greedy-by-SR cumulative oracle; each mode's NET-NEW solves =
     its irreplaceable routing-portfolio contribution (where +16pp actually comes from).
  C. Router features         — task-intrinsic features available at routing time
     (visual_difficulty / eval_type / has_image) cross-tabbed against mode-CLASS capability
     (image-modes {som,vision} that see pixels vs text-modes {dom,phantom_*} that don't).
     "img_only" = image-class solves AND text-class all fail = the cleanest routing signal.
  D. Noise sensitivity       — exclusive-solves (k=1) are the quantity MOST inflated by the
     ~13-14% B0 serving floor (§302/§308); a single fail->pass flip fabricates a spurious
     exclusive. Magnitudes are upper bounds; the feature->class DIRECTION is more robust.

PROVISIONAL — single (B0, classifieds) run; oracle MAX is one-sided upward-biased by the
serving floor (§308 B0=13.3%, §302 vision=14.3%), which is the SAME order as the +16pp
lift. Needs replicate-calibration before any magnitude is paper-grade. NOT a gate.

Usage (same --run interface as the taxonomy script):
  python scripts/analysis/cross_mode_routable_deepdive.py --site classifieds --model B0 \
      --run dom=<dir> --run som=<dir> --run vision=<dir> \
      --run phantom_text=<dir> --run phantom_som=<dir> --run phantom_prompt=<dir> \
      --out docs/analysis/vwa_classifieds/B0_classifieds_6mode_routable_deepdive.md
"""
import argparse, json, glob, os, re, sys
from collections import Counter

# reuse the deterministic trace/listsig helpers from the sibling taxonomy script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_mode_failure_taxonomy import trace, listsig  # noqa: E402

# what each mode's agent actually SEES (paper §3 phantom = "skip annotated image"):
#   image-modes see pixels (som = annotated image + marks; vision = raw screenshot)
#   text-modes are pixel-blind (dom = AXTree; phantom_* = AXTree/regex text, NO image)
IMAGE_MODES = {"som", "vision"}
TEXT_MODES = {"dom", "phantom_text", "phantom_som", "phantom_prompt"}


def features(run_dir, site, task):
    """Task-intrinsic features available to a router BEFORE the episode runs."""
    cg = glob.glob(f"{run_dir}/task_configs/{site}_task_{task}.json")
    if not cg:
        return None
    d = json.load(open(cg[0]))
    ev = (d.get("eval") or {})
    et = set(ev.get("eval_types") or [])
    # primary eval-type bucket (the eval mechanism = a proxy for answer shape)
    evb = ("program_html" if "program_html" in et else
           "url_match" if "url_match" in et else
           "string_match" if "string_match" in et else "other")
    return {
        "visual": (d.get("visual_difficulty") or "?").replace("mediun", "medium"),
        "reasoning": d.get("reasoning_difficulty") or "?",
        "overall": d.get("overall_difficulty") or "?",
        "eval": evb,
        "has_image": bool(d.get("image")),
        "start_sig": bool(listsig(d.get("start_url", "") or "")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--model", default="B0")
    ap.add_argument("--run", action="append", required=True, help="mode=run_dir, repeatable")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        mode, _, rd = spec.partition("=")
        runs[mode] = rd.rstrip("/")
    modes = list(runs)
    img_modes = [m for m in modes if m in IMAGE_MODES]
    txt_modes = [m for m in modes if m in TEXT_MODES]

    # common task ids (intersection across modes)
    per_mode = {}
    for m, rd in runs.items():
        ids = set()
        for f in (glob.glob(f"{rd}/*/episodes/{args.site}_task_*_summary_v2.json") or
                  glob.glob(f"{rd}/episodes/{args.site}_task_*_summary_v2.json")):
            mm = re.search(rf"{args.site}_task_(\d+)_summary", f)
            if mm:
                ids.add(int(mm.group(1)))
        per_mode[m] = ids
    tasks = sorted(set.intersection(*per_mode.values()))

    # success matrix
    succ = {}
    for t in tasks:
        for m in modes:
            tr = trace(runs[m], args.site, t)
            succ[(t, m)] = bool(tr and tr.get("success"))

    SR = {m: sum(succ[(t, m)] for t in tasks) / len(tasks) for m in modes}
    solve_count = {t: sum(succ[(t, m)] for m in modes) for t in tasks}
    n = len(modes)
    routable = [t for t in tasks if 0 < solve_count[t] < n]
    excl_tasks = [t for t in tasks if solve_count[t] == 1]
    shared_tasks = [t for t in tasks if 1 < solve_count[t] < n]
    img_solve = {t: any(succ[(t, m)] for m in img_modes) for t in tasks}
    txt_solve = {t: any(succ[(t, m)] for m in txt_modes) for t in tasks}

    feat = {t: (features(runs[modes[0]], args.site, t) or {}) for t in tasks}

    out = []
    w = out.append

    w(f"# Cross-mode Routable Deep-dive — {args.model} {args.site}")
    w("")
    w(f"> Modes: {', '.join(modes)} | common N={len(tasks)} | image-modes={img_modes} "
      f"text-modes={txt_modes} | deterministic (no sub-agent)")
    w("> ⚠️ **PROVISIONAL** — single (model,site) run. Oracle MAX is one-sided "
      "upward-biased by the B0 serving floor (§308 13.3% / §302 vision 14.3%), which is the")
    w("> **same order as the +16pp lift**. Magnitudes = UPPER BOUNDS, need replicate-"
      "calibration (§293/§306/§309). Feature→class DIRECTION is more noise-robust than "
      "oracle MAGNITUDE. NOT a gate.")
    w("")

    # --- A. routable decomposition ---
    oracle = sum(1 for t in tasks if solve_count[t] > 0)
    best_single = max(SR, key=SR.get)
    w("## A. Routable decomposition — must-route (k=1) vs route-forgiving (k≥2)")
    w("")
    w("| solve-count k | N tasks | meaning |")
    w("|---:|---:|---|")
    hist = Counter(solve_count[t] for t in tasks)
    labels = {0: "universal-fail (routing can't help)", n: "universal-solve (routing free)"}
    for k in range(0, n + 1):
        lab = labels.get(k, ("**exclusive — must pick THE mode**" if k == 1
                             else f"shared by {k} modes — route-forgiving"))
        w(f"| {k} | {hist.get(k,0)} | {lab} |")
    w("")
    w(f"- **routable = {len(routable)}** ({len(routable)/len(tasks)*100:.0f}%) = "
      f"exclusive **{len(excl_tasks)}** (k=1, routing MUST be correct) + "
      f"shared **{len(shared_tasks)}** (k≥2, any of several modes works).")
    w(f"- 6-mode oracle = {oracle}/{len(tasks)} = {oracle/len(tasks)*100:.1f}%; "
      f"best-single = {best_single} {SR[best_single]*100:.1f}% → "
      f"**oracle lift = +{(oracle/len(tasks)-SR[best_single])*100:.1f}pp**.")
    w(f"- The lift is carried by the **{len(excl_tasks)} exclusive + the shared tasks "
      f"best-single fails**. Exclusive tasks are the noise-fragile core (§D).")
    w("")

    # --- B. oracle marginal (greedy by SR desc) ---
    w("## B. Oracle marginal — where the +16pp portfolio value lives")
    w("")
    w("Greedy add modes by SR (desc); marginal = NET-NEW tasks no higher-SR mode solved.")
    w("A mode's marginal = its **irreplaceable** routing contribution (drop it → oracle falls).")
    w("")
    w("| order | mode | SR | marginal NEW | cumulative oracle |")
    w("|---|---|---:|---:|---:|")
    covered = set()
    for i, m in enumerate(sorted(modes, key=lambda x: SR[x], reverse=True), 1):
        solved_m = {t for t in tasks if succ[(t, m)]}
        new = solved_m - covered
        covered |= solved_m
        w(f"| {i} | {m} | {SR[m]*100:.1f}% | +{len(new)} | "
          f"{len(covered)} ({len(covered)/len(tasks)*100:.1f}%) |")
    w("")
    # leave-one-out oracle drop = each mode's irreplaceable solves regardless of order
    w("Leave-one-out (order-independent): tasks ONLY this mode solves (= exclusive) — "
      "drop it and oracle loses exactly these:")
    w("")
    w("| mode | exclusive (LOO oracle loss) |")
    w("|---|---:|")
    for m in modes:
        e = [t for t in tasks if succ[(t, m)] and solve_count[t] == 1]
        w(f"| {m} | {len(e)} |")
    w("")

    # --- C. router features ---
    w("## C. Router feature candidates — task-intrinsic feature × mode-CLASS capability")
    w("")
    w("For each feature value, over ALL tasks with that value: image-class SR "
      "(any of {som,vision}) vs text-class SR (any of {dom,phantom_*}); "
      "**img_only** = image-class solves AND text-class ALL fail (= must route to pixels); "
      "**txt_only** = text-class solves AND image-class all fail (= pixels not needed / hurt).")
    w("")

    def feat_table(key, order):
        w(f"### feature: `{key}`")
        w("")
        w("| value | N | img-class SR | txt-class SR | img_only | txt_only | both | neither |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|")
        vals = order or sorted({feat[t].get(key) for t in tasks if feat[t]},
                               key=lambda x: str(x))
        for v in vals:
            sub = [t for t in tasks if feat[t].get(key) == v]
            if not sub:
                continue
            isr = sum(img_solve[t] for t in sub) / len(sub)
            tsr = sum(txt_solve[t] for t in sub) / len(sub)
            io = sum(img_solve[t] and not txt_solve[t] for t in sub)
            to = sum(txt_solve[t] and not img_solve[t] for t in sub)
            bo = sum(img_solve[t] and txt_solve[t] for t in sub)
            ne = sum(not img_solve[t] and not txt_solve[t] for t in sub)
            w(f"| {v} | {len(sub)} | {isr*100:.0f}% | {tsr*100:.0f}% | "
              f"{io} | {to} | {bo} | {ne} |")
        w("")

    feat_table("visual", ["easy", "medium", "hard"])
    feat_table("eval", ["url_match", "string_match", "program_html", "other"])
    feat_table("has_image", [False, True])
    feat_table("overall", ["easy", "medium", "hard"])

    # img_only / txt_only id drill-down (section6 examples + manual spot-check)
    io_ids = [t for t in tasks if img_solve[t] and not txt_solve[t]]
    to_ids = [t for t in tasks if txt_solve[t] and not img_solve[t]]
    w(f"- **img_only tasks** (N={len(io_ids)}, pixels REQUIRED): {io_ids}")
    w(f"- **txt_only tasks** (N={len(to_ids)}, pixels NOT needed): {to_ids}")
    w("")

    # --- D. noise sensitivity ---
    w("## D. Noise sensitivity — which numbers survive the ~13-14% serving floor")
    w("")
    w("| quantity | value | noise exposure |")
    w("|---|---:|---|")
    w(f"| best-single SR ({best_single}) | {SR[best_single]*100:.1f}% | "
      "LOW — single mode, noise ~symmetric (some true-solves flip out, some true-fails "
      "flip in) → ≈unbiased |")
    w(f"| 6-mode oracle SR | {oracle/len(tasks)*100:.1f}% | "
      "HIGH — MAX over 6 modes is **one-sided**; picks up every positive flip, ignores "
      "negatives → UPWARD biased |")
    w(f"| oracle lift | +{(oracle/len(tasks)-SR[best_single])*100:.1f}pp | "
      "HIGH — = oracle − best-single, inherits the one-sided oracle bias |")
    w(f"| exclusive-solves (k=1) | {len(excl_tasks)} | "
      "HIGHEST — one fail→pass flip on a universal-fail task fabricates a spurious "
      "exclusive; §302 decomposed 14/224 ≈ 6% fail→pass per replicate, near-boundary |")
    w(f"| shared-solves (k≥2) | {len(shared_tasks)} | "
      "MEDIUM — needs ≥2 modes to agree, harder to fake by independent flips |")
    w(f"| img_only / txt_only counts | {len(io_ids)} / {len(to_ids)} | "
      "MEDIUM — class-level (any-of-2 / any-of-4) absorbs single-mode flips better than "
      "per-mode exclusive |")
    w("")
    w("> **Defensible separation**: the feature→class *direction* (which feature predicts "
      "image-class vs text-class advantage) reflects systematic representation differences "
      "and survives symmetric noise scatter; the oracle *magnitude* (+16pp) is a one-sided "
      "UPPER BOUND. Router-feature claims travel; the headline lift needs a replicate "
      "(§293 replicate-calibrated MC) before it is paper-grade.")
    w("")

    text = "\n".join(out)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")
        print(f"\n[written] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
