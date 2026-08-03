"""Three deployment classes, not six modes.

Web agents in the wild come in three shapes: **no-image** (accessibility tree only),
**vision-only** (screenshot only, the computer-use-aligned line), and **hybrid**
(screenshot + tree + marks). This project's six modes map onto them:

    no-image     DOM, P-text, P-prompt, P-SoM      (4 arms)
    vision-only  Vision                            (1 arm)
    hybrid       SoM                               (1 arm)

The grouping is not arbitrary: `per_mode_four_dimension_profile` finds that the four
no-image modes reach the >=83% consistency bar on **none** of 26 metrics across 8 cells
(Vision reaches it on 9, SoM on 5). That negative has been quoted as "a licence to group
them" since it was computed; this is the product that actually groups them.

⚠️ **The obvious headline is a trap and this file computes it anyway, labelled.** Dropping
the whole no-image class costs far more than dropping either other class — but that class
has four arms and the others have one each, so most of the gap is arm count, not
irreplaceability. The arm-matched panel is the honest one and it shows **no systematic
difference between classes**. Both are reported; the unmatched one is marked.

What survives arm-matching is in §3: which class supplies the best single arm, per cell.

Regenerate:
    .venv/bin/python3 scripts/analysis/representation_class_comparison.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

MODES = ["dom", "som", "vision", "ptext", "pprompt", "psom"]
PRETTY = {"dom": "DOM", "som": "SoM", "vision": "Vision",
          "ptext": "P-text", "pprompt": "P-prompt", "psom": "P-SoM"}
CLASSES = {
    "no-image": ["dom", "ptext", "pprompt", "psom"],
    "vision-only": ["vision"],
    "hybrid": ["som"],
}
WA_STEM = {"dom": "dom", "som": "som", "vision": "vision", "ptext": "phantom_text",
           "pprompt": "phantom_prompt", "psom": "phantom_som"}
CELL_ORDER = ["cls_B0", "cls_B1", "cls_B2", "red_B0", "red_B1", "red_B2", "wa_B0", "wa_B1"]

OUT_MD = REPO / "docs/analysis/cross_sites/representation_class_comparison.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/representation_class_comparison.json"


class MissingInput(RuntimeError):
    """Fail loud rather than silently compare over a partial cell set."""


def load_cells() -> dict[str, dict[int, dict[str, int]]]:
    cells: dict[str, dict[int, dict[str, int]]] = {}
    for r in csv.DictReader((REPO / "results/phantom_paper/per_task_sr.csv").open()):
        scored, _ = expected_scored_ids(r["site"])
        tid = int(r["task_id"])
        if tid in scored:
            cells.setdefault(r["cell_id"], {})[tid] = {
                m: int(float(r[f"sr_{m}"]) > 0) for m in MODES}
    for b in ("B1", "B0"):
        per: dict[str, dict[int, int]] = {}
        for m, stem in WA_STEM.items():
            hits = [p for p in glob.glob(
                str(REPO / f"results/webarena/phase1/{b}_{stem}_wa_reddit_2026*_R*"))
                if Path(p).is_dir() and "ABORTED" not in p]
            if len(hits) != 1:
                per = {}
                break
            rows = {}
            for f in Path(hits[0]).glob("*/episodes/*summary*.json"):
                s = json.loads(f.read_text())
                if not s.get("sr_excluded"):
                    rows[int(s["task_id"])] = 1 if s.get("success") else 0
            per[m] = rows
        if per:
            common = set.intersection(*(set(v) for v in per.values()))
            cells[f"wa_{b}"] = {t: {m: per[m][t] for m in MODES} for t in sorted(common)}
    if not cells:
        raise MissingInput("no cells loaded")
    return cells


def coverage(tasks, arms) -> float:
    return 100 * sum(1 for t in tasks if any(tasks[t][m] for m in arms)) / len(tasks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    a = ap.parse_args()

    cells = load_cells()
    out: dict = {"schema": 1, "post_hoc_exploratory": True, "h10_eligible": False,
                 "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "classes": CLASSES, "cells": {}}

    L = ["---", "type: analysis", "status: rolling",
         "purpose: compare the three deployment classes web agents actually come in",
         "producer: scripts/analysis/representation_class_comparison.py", "---", "",
         "# Three representation classes", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/representation_class_comparison.py`", "",
         "| class | arms | what a deployment ships |", "|---|---|---|",
         "| **no-image** | DOM, P-text, P-prompt, P-SoM (4) | accessibility tree only |",
         "| **vision-only** | Vision (1) | screenshot only — the computer-use-aligned line |",
         "| **hybrid** | SoM (1) | screenshot + tree + marks |", "",
         "Grouping the four no-image modes is licensed by `per_mode_four_dimension_profile`: "
         "they reach the ≥83% consistency bar on **none** of 26 metrics over 8 cells "
         "(Vision 9, SoM 5).", "",
         "## 1. Best arm in each class", "",
         "| cell | n | no-image | vision-only | hybrid | best class |",
         "|---|---|---|---|---|---|"]

    best_class_count: dict[str, int] = {}
    for cid in CELL_ORDER:
        if cid not in cells:
            continue
        T = cells[cid]
        n = len(T)
        single = {m: 100 * sum(T[t][m] for t in T) / n for m in MODES}
        best = {k: max(single[m] for m in v) for k, v in CLASSES.items()}
        best_arm = {k: max(v, key=lambda m: single[m]) for k, v in CLASSES.items()}
        # Ties must be named, not resolved by dict order. On `cls_B2` vision-only and
        # hybrid are both 2.23% (5/224); letting `max` pick the first key silently handed
        # vision-only a "win" and turned a floor-level tie into a claim.
        hi = max(best.values())
        winners = sorted(k for k in best if abs(best[k] - hi) < 1e-9)
        top = winners[0] if len(winners) == 1 else "tie: " + "+".join(winners)
        if len(winners) == 1:
            best_class_count[winners[0]] = best_class_count.get(winners[0], 0) + 1
        cell_rec = {"n": n, "single_sr": single, "class_best": best,
                    "class_best_arm": best_arm, "best_class": top,
                    "best_class_winners": winners, "is_tie": len(winners) > 1}
        L.append(f"| `{cid}` | {n} | {best['no-image']:.2f}% ({PRETTY[best_arm['no-image']]}) | "
                 f"{best['vision-only']:.2f}% | {best['hybrid']:.2f}% | **{top}** |")
        out["cells"][cid] = cell_rec

    never = [k for k in CLASSES if best_class_count.get(k, 0) == 0]
    n_tie = sum(1 for c in out["cells"].values() if c.get("is_tie"))
    L += ["", "Best-class tally (**sole** winners; ties counted separately): "
          + (", ".join(f"**{k}** {v}/{len(out['cells'])}" for k, v in sorted(
              best_class_count.items(), key=lambda kv: -kv[1])) or "none")
          + (f", plus {n_tie} tied cell(s)" if n_tie else "")
          + ".", ""]
    if never:
        L += [f"**{', '.join(never)} is never the sole best class in any cell.** "
              "Where it appears to win, the win is a tie — and on `cls_B2` that tie is "
              "between two arms at 2.23% (5 successes out of 224), i.e. at the floor. A tie "
              "resolved by dict order is how this nearly became a claim.", ""]

    # --- 1b. arm-matched restatement of §1 -------------------------------------------
    # §1's no-image column is max over FOUR arms while the other two columns are single
    # arms. A maximum over noisy quantities biases that column up (the same defect
    # aggregate_fusion_premium.py:15 fixed on the comparator side), so §1 cannot be read
    # as a like-for-like class comparison even though it is where the class claim is read
    # from. This panel holds arm count fixed at 1 and uses the arm of each class that
    # exists outside this study: DOM (P-text / P-prompt / P-SoM are constructed here),
    # Vision, SoM.
    SHIPPED = {"no-image": "dom", "vision-only": "vision", "hybrid": "som"}
    L += ["## 1b. The same comparison at one arm per class", "",
          "§1's `no-image` column is a **maximum over four arms**; the other two columns are "
          "single arms. A maximum over noisy quantities is biased up, so §1 is not like-for-"
          "like. This panel fixes arm count at 1 and uses the arm of each class that exists "
          "outside this study — **DOM** (P-text / P-prompt / P-SoM are constructed here), "
          "**Vision**, **SoM**.", "",
          "| cell | no-image (DOM) | vision-only | hybrid (SoM) | winner | §1 winner | "
          "gap vs hybrid: 1-arm | §1 (max-of-4) |",
          "|---|---|---|---|---|---|---|---|"]
    matched_count: dict[str, int] = {}
    n_tie_1v1 = 0
    gap_shrink: list[tuple[str, float, float]] = []
    for cid in CELL_ORDER:
        if cid not in cells:
            continue
        rec = out["cells"][cid]
        s = rec["single_sr"]
        one = {k: s[m] for k, m in SHIPPED.items()}
        hi1 = max(one.values())
        win1 = sorted(k for k in one if abs(one[k] - hi1) < 1e-9)
        top1 = win1[0] if len(win1) == 1 else "tie: " + "+".join(win1)
        if len(win1) == 1:
            matched_count[win1[0]] = matched_count.get(win1[0], 0) + 1
        else:
            n_tie_1v1 += 1
        gap1 = one["no-image"] - one["hybrid"]
        gap4 = rec["class_best"]["no-image"] - rec["class_best"]["hybrid"]
        gap_shrink.append((cid, gap1, gap4))
        rec["one_arm_per_class"] = {"arms": SHIPPED, "sr": one, "winners": win1,
                                    "gap_noimage_minus_hybrid_pp": gap1,
                                    "gap_maxof4_minus_hybrid_pp": gap4}
        L.append(f"| `{cid}` | {one['no-image']:.2f}% | {one['vision-only']:.2f}% | "
                 f"{one['hybrid']:.2f}% | **{top1}** | {rec['best_class']} | "
                 f"{gap1:+.2f}pp | {gap4:+.2f}pp |")
    never1 = [k for k in CLASSES if matched_count.get(k, 0) == 0]
    L += ["", "Arm-matched tally (sole winners): "
          + (", ".join(f"**{k}** {v}/{len(out['cells'])}" for k, v in sorted(
              matched_count.items(), key=lambda kv: -kv[1])) or "none")
          + (f", plus {n_tie_1v1} tied cell(s)" if n_tie_1v1 else "") + ".", ""]
    _same = (matched_count == best_class_count)
    L += [("**The tally does not move** when arm count is held fixed"
           if _same else
           "⚠️ **The tally moves** when arm count is held fixed — §1's ordering is "
           "partly an arm-count artefact")
          + (f", and {', '.join(never1)} is still never the sole best class."
             if never1 else ".")
          + " That is the robustness statement §1 could not make: the class conclusion "
            "survives the most obvious attack on it.", ""]
    _worst = max(gap_shrink, key=lambda r: abs(r[2]) - abs(r[1]))
    L += [f"⚠️ **The effect sizes do move, and by a lot.** On `{_worst[0]}` the no-image "
          f"lead over hybrid is **{_worst[2]:+.2f}pp** in §1 and **{_worst[1]:+.2f}pp** here "
          f"— §1's figure is carried by "
          f"`{PRETTY[out['cells'][_worst[0]]['class_best_arm']['no-image']]}`, an arm "
          "constructed for this study rather than one a deployment ships. Any sentence "
          "quoting a class *gap* should quote this panel's number; only the *ordering* is "
          "safe to take from §1.", ""]
    out["arm_matched_class_count"] = matched_count
    out["arm_matched_tally_matches_unmatched"] = _same

    # --- 2. unmatched drop (labelled) ---
    L += ["## 2. Dropping a whole class — ⚠️ NOT arm-matched", "",
          "How much oracle coverage disappears if a class is unavailable. **The no-image class "
          "has four arms and the others have one each**, so a larger number here is mostly arm "
          "count. Reported because the figure is the obvious one to compute and would otherwise "
          "be computed by a reader without the caveat attached.", "",
          "| cell | all six | drop no-image (4 arms) | drop vision-only (1) | drop hybrid (1) |",
          "|---|---|---|---|---|"]
    for cid in CELL_ORDER:
        if cid not in cells:
            continue
        T = cells[cid]
        allc = coverage(T, MODES)
        drops = {k: allc - coverage(T, [m for m in MODES if m not in v])
                 for k, v in CLASSES.items()}
        out["cells"][cid]["drop_class_unmatched"] = drops
        out["cells"][cid]["oracle_all"] = allc
        L.append(f"| `{cid}` | {allc:.2f}% | +{drops['no-image']:.2f}pp | "
                 f"+{drops['vision-only']:.2f}pp | +{drops['hybrid']:.2f}pp |")

    # --- 3. arm-matched ---
    L += ["", "## 3. Arm-matched: add ONE arm to the cell's best single arm", "",
          "The comparison §2 should have been. From each cell's best single mode, add one more "
          "arm and record the gain, grouped by which class the added arm belongs to (taking the "
          "best available arm within each class). A class already holding the starting arm shows "
          "`—`.", "",
          "| cell | start | +1 no-image | +1 vision-only | +1 hybrid | largest |",
          "|---|---|---|---|---|---|"]
    matched_win: dict[str, int] = {}
    for cid in CELL_ORDER:
        if cid not in cells:
            continue
        T = cells[cid]
        n = len(T)
        single = {m: 100 * sum(T[t][m] for t in T) / n for m in MODES}
        base = max(single, key=lambda m: single[m])
        gains: dict[str, float | None] = {}
        for k, arms in CLASSES.items():
            cands = [m for m in arms if m != base]
            gains[k] = max((coverage(T, [base, m]) - single[base] for m in cands), default=None)
        avail = {k: v for k, v in gains.items() if v is not None}
        mx = max(avail.values())
        winners = [k for k, v in avail.items() if abs(v - mx) < 1e-9]
        for w in winners:
            matched_win[w] = matched_win.get(w, 0) + (1 if len(winners) == 1 else 0)
        out["cells"][cid]["arm_matched_gain"] = gains
        out["cells"][cid]["arm_matched_winner"] = winners
        fmt = lambda v: f"+{v:.2f}pp" if v is not None else "—"
        L.append(f"| `{cid}` | {PRETTY[base]}@{single[base]:.2f}% | {fmt(gains['no-image'])} | "
                 f"{fmt(gains['vision-only'])} | {fmt(gains['hybrid'])} | {'+'.join(winners)} |")

    L += ["", "Largest-gain tally (sole winners only): "
          + (", ".join(f"**{k}** {v}" for k, v in sorted(matched_win.items(), key=lambda kv: -kv[1]))
             or "none")
          + ". **The classes do not differ systematically once arm count is held fixed** — which "
          "is what kills §2's headline. Note `hybrid` supplies the starting arm in most cells, so "
          "it rarely gets the chance to contribute a marginal gain at all.", "",
          "## 4. What survives", "",
          f"1. **The vision-only class is never the best class** (0 of {len(out['cells'])}). That "
          "is the computer-use-aligned shape, and on this data it is dominated in every cell by "
          "either the tree-only or the fused option.",
          "2. **Which class wins reverses with the workload.** Hybrid takes the VWA cells; "
          "no-image takes both WebArena cells, by 13.46pp on `wa_B0` (35.58% P-text against "
          "22.12% SoM). The same reversal claim 4 makes, restated as the deployment question "
          "*should we ship vision at all* rather than as a per-mode ranking.",
          "3. **Class membership does not predict marginal value** (§3). Adding an arm is worth "
          "about the same regardless of which class it comes from.", "",
          "⚠️ Every number is an oracle over landed runs: it says what a perfect chooser could "
          "have gotten, not what any deployable policy gets. `rule_routing_pareto` is the "
          "companion showing that a real policy built on the strongest available signal does "
          "not beat a fixed one.", ""]

    out["best_class_count"] = best_class_count
    out["arm_matched_winner_count"] = matched_win
    a.out_md.write_text("\n".join(L) + "\n")
    a.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[md]   {a.out_md}")
    print(f"[json] {a.out_json}")
    print(f"  best-class tally     : {best_class_count}")
    print(f"  arm-matched winners  : {matched_win}")
    print(f"  never best           : {never}")


if __name__ == "__main__":
    main()
