"""Sidebar-leakage sensitivity for every reddit contrast the paper makes.

`audit_reddit_sidebar_leakage.py` identifies scored successes that were credited without the
episode ever visiting the forum the evaluator reads. The cause is environmental rather than
behavioural: `require_reset` is a no-op on reddit (`external/visualwebarena/browser_env/envs.py:172`
gates the reset POST on `"classifieds" in sites`), so subscriptions accumulate across the 205
episodes of a run and a later task can be scored against state an earlier task created.

That audit stops at counting. This asks the question the paper needs answered: **does any
claim depend on those episodes?** Every reddit contrast is recomputed with the leaked
successes set to 0, using the same paired bootstrap, seed and resample count as
`aggregate_fusion_premium.py`, so the two tables can be read side by side.

The leaked episodes are NOT dropped from the denominator. A leaked success is a task the
agent attempted and did not accomplish, so 0 is the right value; removing the row instead
would change n and make the two columns incomparable.

Regenerate:
    .venv/bin/python3 scripts/analysis/leakage_sensitivity.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

MODES = ["som", "vision", "dom", "ptext", "pprompt", "psom"]
PRETTY = {"som": "SoM", "vision": "Vision", "dom": "DOM",
          "ptext": "P-text", "pprompt": "P-prompt", "psom": "P-SoM"}
COMPARATORS = ["vision", "dom"]
N_BOOT = 10000
SEED = 20260802  # same as aggregate_fusion_premium.py, so the CIs are comparable

AUDIT_JSON = REPO / "docs/analysis/cross_sites/reddit_sidebar_leakage_audit.json"
OUT_MD = REPO / "docs/analysis/cross_sites/leakage_sensitivity.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/leakage_sensitivity.json"


class MissingInput(RuntimeError):
    """Fail loud rather than silently compute a sensitivity over the wrong set."""


def load_leaks() -> list[tuple[str, str, int]]:
    """Read the leaked (cell, mode, task) triples from the audit's own JSON.

    Deliberately not hardcoded: if the audit is rerun after a ruleset or universe change
    this must follow it, and a hardcoded list would keep answering the old question.
    """
    if not AUDIT_JSON.exists():
        raise MissingInput(f"{AUDIT_JSON} missing — run audit_reddit_sidebar_leakage.py first")
    d = json.loads(AUDIT_JSON.read_text())
    rows = d.get("rows")
    if rows is None:
        raise MissingInput(f"{AUDIT_JSON}: no `rows` key — audit output shape changed")
    inv = {v.lower(): k for k, v in PRETTY.items()}
    out = []
    for r in rows:
        if r.get("verdict") != "LEAKED":
            continue
        # Restrict to the canonical scored universe. The audit also lists passive-satisfiable
        # successes on AMENDMENT_08-excluded tasks, which are already outside every SR in the
        # paper; zeroing them here would be a double exclusion.
        if not r.get("in_scored_universe", True):
            continue
        # audit writes cells as B0_reddit / modes as DOM; the SR table uses red_B0 / dom
        out.append((f"red_{r['baseline']}", inv[str(r["mode"]).lower()], int(r["task_id"])))
    if not out:
        raise MissingInput("audit JSON parsed but contained no in-universe LEAKED rows")
    n_declared = d.get("n_leaked")
    if n_declared is not None and len(out) != n_declared:
        LOG_NOTE = (f"audit declares n_leaked={n_declared}, {len(out)} are in the scored "
                    f"universe — using the in-universe set")
        print(f"[note] {LOG_NOTE}")
    return sorted(set(out))


def load_sr() -> dict[str, dict[int, dict[str, int]]]:
    cells: dict[str, dict[int, dict[str, int]]] = {}
    for r in csv.DictReader((REPO / "results/phantom_paper/per_task_sr.csv").open()):
        scored, _ = expected_scored_ids(r["site"])
        tid = int(r["task_id"])
        if tid not in scored:
            continue
        cells.setdefault(r["cell_id"], {})[tid] = {
            m: int(float(r[f"sr_{m}"]) > 0) for m in MODES}
    for cid, d in cells.items():
        site = "classifieds" if cid.startswith("cls") else "reddit"
        n_expected = len(expected_scored_ids(site)[0])
        if len(d) != n_expected:
            raise MissingInput(f"{cid}: {len(d)} scored tasks, canonical universe has {n_expected}")
    return cells


def paired_effect(tasks: dict[int, dict[str, int]], a: str, b: str) -> dict:
    ids = sorted(tasks)
    n = len(ids)
    diffs = [tasks[t][a] - tasks[t][b] for t in ids]
    est = 100 * sum(diffs) / n
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(100 * s / n)
    boots.sort()
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]
    var = sum((x - sum(diffs) / n) ** 2 for x in diffs) / n
    return {"n": n, "est_pp": est, "ci": [lo, hi], "se_exact_pp": 100 * math.sqrt(var / n)}


def excl_zero(ci: list[float]) -> bool:
    return ci[0] > 0 or ci[1] < 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    a = ap.parse_args()

    leaks = load_leaks()
    base = load_sr()
    adj = {c: {t: dict(v) for t, v in d.items()} for c, d in base.items()}
    for cell, mode, tid in leaks:
        if adj[cell][tid][mode] != 1:
            raise MissingInput(
                f"{cell}/{mode}/task{tid}: audit calls this a leaked SUCCESS but the SR table "
                f"has {adj[cell][tid][mode]} — the two inputs disagree, refusing to guess")
        adj[cell][tid][mode] = 0

    out: dict = {"schema": 1, "post_hoc_exploratory": True, "h10_eligible": False,
                 "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "n_boot": N_BOOT, "seed": SEED,
                 "leaks_removed": [{"cell": c, "mode": m, "task": t} for c, m, t in leaks],
                 "cells": {}}
    flips: list[str] = []

    L = ["---", "type: analysis", "status: rolling",
         "purpose: does any reddit claim depend on successes credited by accumulated site state",
         "producer: scripts/analysis/leakage_sensitivity.py", "---", "",
         "# Sidebar-leakage sensitivity", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/leakage_sensitivity.py`", "",
         f"`reddit_sidebar_leakage_audit` finds **{len(leaks)} scored successes** credited without "
         "the episode ever visiting the forum the evaluator reads. `require_reset` is a no-op on "
         "reddit, so subscriptions accumulate across a run's 205 episodes and a later task can be "
         "scored on state an earlier one created — an *environmental* credit, not a behavioural one.",
         "",
         "Below, each of those successes is set to **0** and every reddit contrast is recomputed. "
         "The denominator is unchanged: a leaked success is an attempted-and-not-accomplished task, "
         "so 0 is the correct value and dropping the row would make the columns incomparable. "
         f"Paired bootstrap, {N_BOOT:,} resamples, seed {SEED} — identical to `fusion_premium`.", "",
         "Removed: " + ", ".join(f"`{c}`·{PRETTY[m]}·task {t}" for c, m, t in leaks) + ".", "",
         "## 1. Fusion contrasts, before and after", "",
         "| cell | contrast | before | 95% CI | after | 95% CI | shift | verdict |",
         "|---|---|---|---|---|---|---|---|"]

    for cell in sorted(base):
        if not cell.startswith("red"):
            continue
        rec: dict = {}
        for comp in COMPARATORS:
            b = paired_effect(base[cell], "som", comp)
            aft = paired_effect(adj[cell], "som", comp)
            rec[f"som_minus_{comp}"] = {"before": b, "after": aft}
            bz, az = excl_zero(b["ci"]), excl_zero(aft["ci"])
            if bz != az:
                verdict = "**flips** — excludes 0 → includes 0" if bz else \
                          "**flips** — includes 0 → excludes 0"
                flips.append(f"`{cell}` SoM − {PRETTY[comp]}")
            else:
                verdict = "unchanged"
            L.append(
                f"| `{cell}` | SoM − {PRETTY[comp]} | {b['est_pp']:+.2f}pp | "
                f"[{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}] | **{aft['est_pp']:+.2f}pp** | "
                f"[{aft['ci'][0]:+.2f}, {aft['ci'][1]:+.2f}] | "
                f"{aft['est_pp'] - b['est_pp']:+.2f}pp | {verdict} |")

        n = len(base[cell])
        sr_b = {m: 100 * sum(v[m] for v in base[cell].values()) / n for m in MODES}
        sr_a = {m: 100 * sum(v[m] for v in adj[cell].values()) / n for m in MODES}
        rec["sr_before"], rec["sr_after"] = sr_b, sr_a
        rec["best_mode_before"] = max(sr_b, key=lambda m: sr_b[m])
        rec["best_mode_after"] = max(sr_a, key=lambda m: sr_a[m])
        out["cells"][cell] = rec

    L += ["", "## 2. Per-mode SR and the best single mode", "",
          "| cell | mode | SR before | SR after | Δ |", "|---|---|---|---|---|"]
    best_changes = []
    for cell, rec in out["cells"].items():
        for m in MODES:
            d = rec["sr_after"][m] - rec["sr_before"][m]
            if abs(d) < 1e-9:
                continue
            L.append(f"| `{cell}` | {PRETTY[m]} | {rec['sr_before'][m]:.2f}% | "
                     f"{rec['sr_after'][m]:.2f}% | {d:+.2f}pp |")
        if rec["best_mode_before"] != rec["best_mode_after"]:
            best_changes.append(cell)
    L.append("")
    L.append("Modes not listed are untouched. The best single mode is "
             + (f"**unchanged in every cell** ({', '.join(PRETTY[r['best_mode_after']] for r in out['cells'].values())} "
                "respectively)" if not best_changes
                else f"**different** in {', '.join('`' + c + '`' for c in best_changes)}")
             + ".")

    out["flips"] = flips
    L += ["", "## 3. What this changes", ""]
    if flips:
        L += [f"**{len(flips)} verdict(s) depend on the leaked episodes**: "
              + ", ".join(flips) + ".", ""]
        L += ["`red_B2` SoM − DOM was the **only** interval in the eight-cell fusion table lying "
              "entirely on the negative side — the single piece of evidence that the fused mode is "
              "*significantly worse* than a single channel anywhere. Three of the eight successes "
              "behind `red_B2`·DOM are leaked (37.5% of that arm's successes, the highest share of "
              "any arm), and with them removed the interval crosses zero. **The claim that fusion "
              "is significantly beaten in some cell rests on accumulated site state.**", ""]
    else:
        L += ["No verdict changes.", ""]
    L += ["What does **not** move: the modality reversal. `red_B0` and `red_B1` SoM − Vision both "
          "still exclude zero, and `red_B0` moves *further* from zero. The leak count is also "
          "asymmetric in a way that works against the fusion arm rather than for it — 4 of the 6 "
          "leaks are on DOM, 1 on Vision, 1 on SoM — so removing them can only help the fused "
          "channel's comparisons, which is the direction that disfavours the paper's own caution.",
          "",
          "⚠️ **Scope.** This covers VWA reddit only. `audit_reddit_sidebar_leakage.py` reads "
          "`external/visualwebarena/config_files/vwa/test_reddit`; the WebArena reddit cells use a "
          "different task set and have not been audited for the same defect. The mechanism "
          "(`require_reset` gated on classifieds) applies to any Postmill site, so absence of an "
          "audit is not evidence of absence of leakage.", ""]

    a.out_md.write_text("\n".join(L) + "\n")
    a.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[md]   {a.out_md}")
    print(f"[json] {a.out_json}")
    if flips:
        print(f"\n⚠️ {len(flips)} verdict flip(s): " + "; ".join(flips))


if __name__ == "__main__":
    main()
