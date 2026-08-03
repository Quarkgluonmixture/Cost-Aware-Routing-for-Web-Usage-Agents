"""Does the 0-token visual-intent rule buy anything a fixed policy does not?

`visual_intent_routing` established that the rule identifies, before any episode runs, a
task subset where the screenshot is worth +22.54pp against +0.65pp elsewhere (cls_B0).
That is a statement about *where the signal is*. It is not yet a statement about whether
routing on it is worth doing, because a router only earns its keep if the policy it
produces is not dominated by some fixed policy that needs no signal at all.

So: put the rule-based policy on the same (success, cost, latency) frontier as the six
fixed single-mode policies and ask whether anything dominates it. Domination here is the
ordinary Pareto one — no worse on all three axes, strictly better on at least one.

Two things this deliberately does NOT do. It does not learn anything: the partition is a
regex over the task intent, fixed in advance, so there is no train/test split to get
wrong and no in-sample optimism to correct for. And it does not compare against the
6-arm oracle, which is unattainable by construction; the comparators are the policies a
deployment could actually run.

Cost and latency are per-attempt cell means from `per_mode_four_dimension_profile`, and
are comparable **within a cell only** (B0 bills a proxy API, B1/B2 are electricity-derived).

Regenerate:
    .venv/bin/python3 scripts/analysis/rule_routing_pareto.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

VISUAL_INTENT_RE = re.compile(
    r"\b(image|picture|photo|screenshot)\b|\bcolou?r of\b|"
    r"\bhow many\b[^.]{0,40}\bin (?:the|this)\b",
    re.IGNORECASE,
)
MODES = ["dom", "som", "vision", "ptext", "pprompt", "psom"]
PRETTY = {"dom": "DOM", "som": "SoM", "vision": "Vision",
          "ptext": "P-text", "pprompt": "P-prompt", "psom": "P-SoM"}
CFG_DIR = {
    "classifieds": REPO / "external/visualwebarena/config_files/vwa/test_classifieds",
    "reddit": REPO / "external/visualwebarena/config_files/vwa/test_reddit",
}
PROFILE = REPO / "docs/analysis/cross_sites/per_mode_four_dimension_profile.json"
OUT_MD = REPO / "docs/analysis/cross_sites/rule_routing_pareto.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/rule_routing_pareto.json"

# Rule-based policies worth putting on the frontier. Each maps flagged -> arm A,
# unflagged -> arm B. Chosen a priori: the flagged set is "needs the screenshot", so the
# flagged arm is an image arm and the unflagged arm is the cheap text arm.
RULE_POLICIES = [("vision", "dom"), ("som", "dom"), ("som", "ptext")]


class MissingInput(RuntimeError):
    """Fail loud rather than silently drop a comparator and shrink the frontier."""


def flagged_set(site: str) -> set[int]:
    scored, _ = expected_scored_ids(site)
    d = CFG_DIR[site]
    if not d.is_dir():
        raise MissingInput(f"{d} missing")
    out, missing = set(), []
    for tid in sorted(scored):
        p = d / f"{tid}.json"
        if not p.exists():
            missing.append(tid)
            continue
        cfg = json.loads(p.read_text())
        if not cfg.get("image") and VISUAL_INTENT_RE.search(str(cfg.get("intent") or "")):
            out.add(tid)
    if missing:
        raise MissingInput(f"{site}: {len(missing)} task configs absent")
    return out


def load_sr() -> dict[str, dict[int, dict[str, int]]]:
    cells: dict[str, dict[int, dict[str, int]]] = {}
    for r in csv.DictReader((REPO / "results/phantom_paper/per_task_sr.csv").open()):
        scored, _ = expected_scored_ids(r["site"])
        tid = int(r["task_id"])
        if tid in scored:
            cells.setdefault(r["cell_id"], {})[tid] = {
                m: int(float(r[f"sr_{m}"]) > 0) for m in MODES}
    return cells


def load_unit_costs() -> dict[str, dict[str, dict[str, float]]]:
    """{cell_id: {mode: {cost, latency, latency_canonical}}} — per-attempt cell means."""
    if not PROFILE.exists():
        raise MissingInput(f"{PROFILE} missing — run per_mode_four_dimension_profile.py")
    prof = json.loads(PROFILE.read_text())
    out: dict[str, dict[str, dict[str, float]]] = {}
    for cell in prof["cells"]:
        cid = ("cls" if cell["site"] == "classifieds" else "red") + "_" + cell["baseline"]
        per = {}
        for pretty, blk in cell["per_mode"].items():
            key = next(k for k, v in PRETTY.items() if v == pretty)
            per[key] = {"cost": blk["mean_cost_usd"],
                        "latency": blk["mean_latency_s"],
                        "latency_canonical": blk.get("mean_latency_canonical_s",
                                                     blk["mean_latency_s"])}
        out[cid] = per
    return out


def evaluate(tasks, unit, chooser) -> dict:
    n = len(tasks)
    sr = 100 * sum(tasks[t][chooser(t)] for t in tasks) / n
    cost = sum(unit[chooser(t)]["cost"] for t in tasks) / n
    lat = sum(unit[chooser(t)]["latency"] for t in tasks) / n
    latc = sum(unit[chooser(t)]["latency_canonical"] for t in tasks) / n
    return {"sr_pct": sr, "cost": cost, "latency_s": lat, "latency_canonical_s": latc}


def dominates(a: dict, b: dict, lat_key: str) -> bool:
    """a dominates b: no worse on all three, strictly better on at least one."""
    ge = (a["sr_pct"] >= b["sr_pct"] - 1e-9 and a["cost"] <= b["cost"] + 1e-12
          and a[lat_key] <= b[lat_key] + 1e-9)
    gt = (a["sr_pct"] > b["sr_pct"] + 1e-9 or a["cost"] < b["cost"] - 1e-12
          or a[lat_key] < b[lat_key] - 1e-9)
    return ge and gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    a = ap.parse_args()

    sr = load_sr()
    units = load_unit_costs()
    out: dict = {"schema": 1, "post_hoc_exploratory": True, "h10_eligible": False,
                 "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "predicate": VISUAL_INTENT_RE.pattern, "cells": {}}

    L = ["---", "type: analysis", "status: rolling",
         "purpose: is a policy built on the 0-token visual-intent rule dominated by a fixed one",
         "producer: scripts/analysis/rule_routing_pareto.py", "---", "",
         "# Rule routing on the (success, cost, latency) frontier", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/rule_routing_pareto.py`", "",
         "`visual_intent_routing` showed **where** the screenshot pays. This asks whether "
         "**routing on that** beats not routing. A router earns its keep only if no signal-free "
         "fixed policy dominates it — no worse on all three axes, strictly better on one.", "",
         "The partition is a regex over the task intent: nothing is learned, so there is no "
         "train/test split and no in-sample optimism. Cost/latency are per-attempt cell means "
         "and are **within-cell comparable only**.", ""]

    verdict_rows: list[tuple[str, str, list[str]]] = []
    for site, pre in (("classifieds", "cls"), ("reddit", "red")):
        F = flagged_set(site)
        for b in ("B0", "B1", "B2"):
            cid = f"{pre}_{b}"
            if cid not in sr or cid not in units:
                continue
            tasks, unit = sr[cid], units[cid]
            policies: dict[str, dict] = {}
            for m in MODES:
                policies[f"always-{PRETTY[m]}"] = evaluate(tasks, unit, lambda t, m=m: m)
            for hi, lo in RULE_POLICIES:
                name = f"rule: flag→{PRETTY[hi]} else {PRETTY[lo]}"
                policies[name] = evaluate(
                    tasks, unit, lambda t, hi=hi, lo=lo: hi if t in F else lo)

            lat_key = "latency_canonical_s"
            dom_by = {n: [o for o in policies if o != n and dominates(policies[o], policies[n], lat_key)]
                      for n in policies}
            frontier = sorted(n for n in policies if not dom_by[n])

            L += [f"## `{cid}` — flagged {len([t for t in F if t in tasks])}/{len(tasks)}", "",
                  "| policy | SR | cost | latency (canonical) | on frontier? | dominated by |",
                  "|---|---|---|---|---|---|"]
            for name in sorted(policies, key=lambda k: (-policies[k]["sr_pct"], policies[k]["cost"])):
                p = policies[name]
                mark = "**yes**" if not dom_by[name] else "no"
                by = "—" if not dom_by[name] else ", ".join(f"`{x}`" for x in dom_by[name])
                star = " ⭐" if name.startswith("rule") else ""
                L.append(f"| {name}{star} | {p['sr_pct']:.2f}% | {p['cost']:.5f} | "
                         f"{p[lat_key]:.1f}s | {mark} | {by} |")
            L.append("")
            out["cells"][cid] = {"n_flagged": len([t for t in F if t in tasks]),
                                 "policies": policies, "frontier": frontier,
                                 "dominated_by": dom_by}
            for name in policies:
                if name.startswith("rule"):
                    verdict_rows.append((cid, name, dom_by[name]))

    survivors = [(c, n) for c, n, d in verdict_rows if not d]
    L += ["## Verdict", ""]
    if survivors:
        by_cell: dict[str, list[str]] = {}
        for c, n in survivors:
            by_cell.setdefault(c, []).append(n)
        L += [f"**A rule policy survives on the frontier in {len(by_cell)} of "
              f"{len(out['cells'])} cells**: "
              + "; ".join(f"`{c}` ({len(v)} of {len(RULE_POLICIES)})" for c, v in sorted(by_cell.items()))
              + ".", ""]
    else:
        L += ["**No rule policy survives anywhere** — a fixed single mode dominates it in "
              "every cell.", ""]
    L += ["Surviving the frontier is a low bar: it means *nothing dominates*, not that the "
          "policy is preferable. Read it as \"routing is not ruled out here\" rather than "
          "\"routing wins here\". The cells where it is dominated are the informative ones — "
          "there, the signal is real (see `visual_intent_routing`) and routing on it still buys "
          "nothing, because the arm the rule sends work *to* is already the right arm to send "
          "everything to.", ""]

    a.out_md.write_text("\n".join(L) + "\n")
    a.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[md]   {a.out_md}")
    print(f"[json] {a.out_json}")
    for c, n, d in verdict_rows:
        print(f"  {c:8} {n:34} {'ON FRONTIER' if not d else 'dominated by ' + ', '.join(d)}")


if __name__ == "__main__":
    main()
