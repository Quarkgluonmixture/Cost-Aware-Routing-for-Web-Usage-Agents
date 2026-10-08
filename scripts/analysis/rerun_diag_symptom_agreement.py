#!/usr/bin/env python3
"""Do reruns of the same condition produce the same /diag SYMPTOMS on the same task?

Why (2026-10-08). `rerun_flip_failure_anatomy` found the coarse reason bucket of a both-fail
task agrees across reruns (pooled kappa 0.62, B0 median 0.46), and that flips part at step 0.
A coarse bucket can agree because the task's template fixes it. This asks the finer question on
Tier-1 rule hits, with the template as the control, using the scans written by
`rerun_pairs_diag_scan.py` (both arms, one ruleset).

Questions and reading rules, fixed before any number was computed (2026-10-08):

  Unit: (pair, task) over the site's scored universe. Symptom set = the set of Tier-1 rule ids
  hit on the episode, minus STATIC rules (decided by code inspection, not by data: P15 reads
  only config + mode; P43 is a neutral task×mode label emitted on any failure). Agreement of
  two sets = Jaccard (1 when both are empty).
  ⚠️ Correction after the first run: P6 and P16 also read only config + mode (steps serve only
  to detect the site), so their hits are fixed by (task, mode). The first run showed both at
  kappa 1.00, which is how the miss was caught; they join STATIC by the same code criterion.
  POST HOC sensitivity (named as such, added after the first run): also drop every rule whose
  primary-scope both-fail kappa is >= 0.9 — intent-gated rules whose behavioural part rarely
  varies could carry agreement the template baseline does not fully remove.

  Q1  Both-fail tasks. Observed agreement J(a_i, b_i) against two exact permutation baselines
      (expected J if arm b's sets were shuffled among tasks):
        cond-level  shuffle among all both-fail tasks of the pair  (what the condition predicts)
        tmpl-level  shuffle among both-fail tasks of the same intent template in the pair
      Excess = observed − baseline, per task; mean over tasks; 95% CI by cluster bootstrap over
      (pair, template), 2000 draws. Template-level excess is computed on tasks whose template
      has >= 2 both-fail tasks in the pair (singletons carry no information and are counted).
      Reading: tmpl-level excess CI excludes 0 AND point >= 0.05 → symptoms are specific to the
      task beyond its template. CI includes 0 → the template explains the agreement.
      PRIMARY scope = pairs with >= 5 flips (non-deterministic serving; same stratum as the
      anatomy product). Pairs with < 5 flips include byte-level reproductions (B1·cls) whose
      agreement is trivially ~1; reported, not pooled into the primary line.
  Q2  Per rule (non-static, >= 15 hits over both arms of both-fail tasks in the primary scope):
      Cohen's kappa of hit_a vs hit_b, and P(hit_b | hit_a) against P(hit_b). Descriptive.
  Q3  Selection check. Mean observed J and cond-level excess for both-success and flip tasks
      too, so the both-fail number is not read as if conditioning on failure were free.
  Q4  Flips. Rules hit on the FAILING run of a flipped task that are also hit on its succeeding
      run: share. High = those rules mark the task, not the failure. Descriptive.

Inputs   results/diag_scans/v11_rerun_pairs/<label>/{a,b}.json ; CLEAN_PAIRS ;
         <arm a run>/task_configs/<site>_task_<t>.json (intent_template_id)
Outputs  docs/analysis/cross_sites/rerun_diag_symptom_agreement.{md,json}
Regenerate:  python scripts/analysis/rerun_pairs_diag_scan.py
             python scripts/analysis/rerun_diag_symptom_agreement.py
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402
from scripts.analysis.lib.replicate_pairs import SITE_OF, clean_pairs  # noqa: E402

SCAN_ROOT = REPO / "results/diag_scans/v11_rerun_pairs"
OUT_MD = REPO / "docs/analysis/cross_sites/rerun_diag_symptom_agreement.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/rerun_diag_symptom_agreement.json"
STATIC = frozenset({"P6", "P15", "P16", "P43"})
POSTHOC_KAPPA = 0.9
MIN_FLIPS = 5
MIN_RULE_HITS = 15
N_BOOT = 2000
SEED = 20261008


class MissingInput(RuntimeError):
    pass


def jac(x: frozenset, y: frozenset) -> float:
    if not x and not y:
        return 1.0
    return len(x & y) / len(x | y)


def load_scan(label: str, side: str) -> tuple[dict[int, dict], str]:
    p = SCAN_ROOT / label / f"{side}.json"
    if not p.exists():
        raise MissingInput(f"{p} missing — run rerun_pairs_diag_scan.py first")
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = {}
    for e in d["results"]:
        rows[int(e["task_id"])] = {
            "success": bool(e["success"]),
            "rules": frozenset(h["rule_id"] for h in e["hits"]) - STATIC,
        }
    return rows, d["ruleset_version"]


def template_of(a_dir: Path, site: str, tid: int) -> int:
    f = a_dir.parent / "task_configs" / f"{site}_task_{tid}.json"
    if not f.exists():
        raise MissingInput(f"{f} missing")
    t = json.loads(f.read_text(encoding="utf-8")).get("intent_template_id")
    if t is None:
        raise MissingInput(f"{f}: no intent_template_id")
    return int(t)


def baseline(units: list[dict], key: str | None) -> None:
    """Exact permutation expectation: mean J(a_i, b_j) over j in i's group (j = i included)."""
    groups = collections.defaultdict(list)
    for u in units:
        groups[u[key] if key else 0].append(u)
    for g in groups.values():
        for u in g:
            u[f"base_{key or 'cond'}"] = statistics.fmean(jac(u["A"], v["B"]) for v in g)
            u[f"gsize_{key or 'cond'}"] = len(g)


def boot_ci(units: list[dict], field: str, rng: np.random.Generator) -> list[float] | None:
    if not units:
        return None
    clusters = collections.defaultdict(list)
    for u in units:
        clusters[(u["pair"], u["tmpl"])].append(u[field])
    keys = list(clusters)
    sums = np.array([sum(clusters[k]) for k in keys])
    cnts = np.array([len(clusters[k]) for k in keys])
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarise(units: list[dict], rng: np.random.Generator) -> dict:
    for u in units:
        u["ex_cond"] = u["obs"] - u["base_cond"]
        u["ex_tmpl"] = u["obs"] - u["base_tmpl"]
    tm = [u for u in units if u["gsize_tmpl"] >= 2]
    f = lambda us, k: float(statistics.fmean(u[k] for u in us)) if us else None  # noqa: E731
    return {
        "n": len(units),
        "obs": f(units, "obs"), "base_cond": f(units, "base_cond"),
        "excess_cond": f(units, "ex_cond"), "excess_cond_ci": boot_ci(units, "ex_cond", rng),
        "n_tmpl_informative": len(tm),
        "obs_tmpl_subset": f(tm, "obs"), "base_tmpl": f(tm, "base_tmpl"),
        "excess_tmpl": f(tm, "ex_tmpl"), "excess_tmpl_ci": boot_ci(tm, "ex_tmpl", rng),
        "exact_set_match": f([{**u, "m": float(u["A"] == u["B"])} for u in units], "m"),
    }


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def main() -> int:
    rng = np.random.default_rng(SEED)
    pairs_meta, units_by_group = [], collections.defaultdict(list)
    versions = set()
    for label, a_dir, _b_dir in clean_pairs():
        site, bench, _ = SITE_OF[label.split(".")[1]]
        ra, va = load_scan(label, "a")
        rb, vb = load_scan(label, "b")
        versions |= {va, vb}
        scored = expected_scored_ids(site, bench)[0]
        tids = sorted(scored & ra.keys() & rb.keys())
        missing = len(scored) - len(tids)
        flips = sum(ra[t]["success"] != rb[t]["success"] for t in tids)
        pairs_meta.append({"pair": label, "n": len(tids), "missing_from_scan": missing, "flips": flips,
                           "primary": flips >= MIN_FLIPS})
        for t in tids:
            sa, sb = ra[t]["success"], rb[t]["success"]
            grp = "both_fail" if not (sa or sb) else "both_success" if (sa and sb) else "flip"
            units_by_group[grp].append({
                "pair": label, "task": t, "tmpl": template_of(a_dir, site, t),
                "A": ra[t]["rules"], "B": rb[t]["rules"], "obs": jac(ra[t]["rules"], rb[t]["rules"]),
                "primary": flips >= MIN_FLIPS, "fail_side": None if grp != "flip" else ("a" if not sa else "b"),
            })
    if len(versions) != 1:
        raise MissingInput(f"ruleset mismatch across scans: {versions}")

    # baselines are computed within pair (and within template) — never across pairs
    for grp, units in units_by_group.items():
        by_pair = collections.defaultdict(list)
        for u in units:
            by_pair[u["pair"]].append(u)
        for us in by_pair.values():
            baseline(us, None)
            baseline(us, "tmpl")

    out: dict = {"ruleset_version": versions.pop(), "static_rules_excluded": sorted(STATIC),
                 "min_flips_primary": MIN_FLIPS, "pairs": pairs_meta, "q1": {}, "q3": {}}
    bf = units_by_group["both_fail"]
    for scope, us in (("primary", [u for u in bf if u["primary"]]),
                      ("low_flip", [u for u in bf if not u["primary"]]), ("all", bf)):
        out["q1"][scope] = summarise(us, rng)
    out["q1"]["per_pair"] = {p["pair"]: summarise([u for u in bf if u["pair"] == p["pair"]], rng)
                             for p in pairs_meta}
    for grp in ("both_success", "flip"):
        out["q3"][grp] = summarise([u for u in units_by_group[grp] if u["primary"]], rng)

    # Q2 per rule, primary both-fail
    prim = [u for u in bf if u["primary"]]
    rules = collections.Counter(r for u in prim for r in (u["A"] | u["B"]))
    q2 = []
    for r, _ in rules.most_common():
        hits = [(r in u["A"], r in u["B"]) for u in prim]
        n_hit = sum(a for a, _ in hits) + sum(b for _, b in hits)
        if n_hit < MIN_RULE_HITS:
            continue
        a_hit = [b for a, b in hits if a]
        q2.append({"rule": r, "hits_a_plus_b": n_hit, "kappa": kappa(hits),
                   "p_b_given_a": sum(a_hit) / len(a_hit) if a_hit else None,
                   "p_b": sum(b for _, b in hits) / len(hits)})
    out["q2"] = q2

    # POST HOC sensitivity: drop rules with primary both-fail kappa >= POSTHOC_KAPPA, recompute
    drop = frozenset(r["rule"] for r in q2 if r["kappa"] is not None and r["kappa"] >= POSTHOC_KAPPA)
    sens = [{**u, "A": u["A"] - drop, "B": u["B"] - drop} for u in prim]
    for u in sens:
        u["obs"] = jac(u["A"], u["B"])
    by_pair = collections.defaultdict(list)
    for u in sens:
        by_pair[u["pair"]].append(u)
    for us in by_pair.values():
        baseline(us, None)
        baseline(us, "tmpl")
    out["q1"]["posthoc_drop_high_kappa"] = {"dropped": sorted(drop), **summarise(sens, rng)}

    # Q4 flips: failing-run rules also present on the succeeding run
    fl = [u for u in units_by_group["flip"] if u["primary"]]
    fail_r = sum(len(u["A"] if u["fail_side"] == "a" else u["B"]) for u in fl)
    shared = sum(len(u["A"] & u["B"]) for u in fl)
    out["q4"] = {"n_flips": len(fl), "rules_on_failing_run": fail_r,
                 "also_on_succeeding_run": shared, "share": shared / fail_r if fail_r else None}

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(out), encoding="utf-8")
    print(render(out))
    return 0


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def ci(c):
    return "—" if c is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def render(o: dict) -> str:
    L = ["# Rerun symptom agreement — Tier-1 /diag on both arms of every rerun pair", "",
         f"*Generated by `scripts/analysis/rerun_diag_symptom_agreement.py` · ruleset `{o['ruleset_version']}` · "
         f"static rules excluded: {', '.join(o['static_rules_excluded'])} · questions and reading rules are in the "
         f"script docstring, fixed before the numbers.*", "",
         "Agreement = Jaccard of the two runs' rule-hit sets on the same task. Baselines are exact permutation "
         "expectations within the pair: **cond** = shuffle among the pair's tasks of the same group; "
         "**tmpl** = shuffle among tasks of the same intent template. Excess CIs: cluster bootstrap over "
         "(pair, template).", "",
         f"Primary scope = pairs with ≥ {o['min_flips_primary']} flips.", "",
         "## Q1. Both-fail tasks", "",
         "| scope | n | observed | cond base | excess over cond | tmpl-informative n | observed (subset) | tmpl base | excess over tmpl | exact same set |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in ("primary", "low_flip", "all"):
        r = o["q1"][s]
        L.append(f"| {s} | {r['n']} | {fmt(r['obs'])} | {fmt(r['base_cond'])} | {fmt(r['excess_cond'])} {ci(r['excess_cond_ci'])} "
                 f"| {r['n_tmpl_informative']} | {fmt(r['obs_tmpl_subset'])} | {fmt(r['base_tmpl'])} | "
                 f"{fmt(r['excess_tmpl'])} {ci(r['excess_tmpl_ci'])} | {fmt(r['exact_set_match'])} |")
    s = o["q1"]["posthoc_drop_high_kappa"]
    L += ["", f"**Post hoc sensitivity** (primary scope, also dropping rules with kappa ≥ 0.9: "
          f"{', '.join(s['dropped']) or 'none'}): observed {fmt(s['obs'])}, excess over cond "
          f"{fmt(s['excess_cond'])} {ci(s['excess_cond_ci'])}, excess over tmpl {fmt(s['excess_tmpl'])} "
          f"{ci(s['excess_tmpl_ci'])} (n = {s['n_tmpl_informative']}), exact same set {fmt(s['exact_set_match'])}."]
    L += ["", "Per pair:", "", "| pair | n | flips | both-fail | observed | excess over cond | excess over tmpl (n) |",
          "|---|---|---|---|---|---|---|"]
    for p in o["pairs"]:
        r = o["q1"]["per_pair"][p["pair"]]
        miss = f" ({p['missing_from_scan']} missing)" if p["missing_from_scan"] else ""
        L.append(f"| `{p['pair']}` | {p['n']}{miss} | {p['flips']} | {r['n']} | {fmt(r['obs'])} | "
                 f"{fmt(r['excess_cond'])} | {fmt(r['excess_tmpl'])} ({r['n_tmpl_informative']}) |")
    L += ["", "## Q2. Per rule (primary both-fail)", "",
          "| rule | hits (a+b) | kappa | P(b hit given a hit) | P(b hit) |", "|---|---|---|---|---|"]
    for r in o["q2"]:
        L.append(f"| {r['rule']} | {r['hits_a_plus_b']} | {fmt(r['kappa'], 2)} | {fmt(r['p_b_given_a'], 2)} | {fmt(r['p_b'], 2)} |")
    L += ["", "## Q3. Selection check (primary scope, other task groups)", "",
          "| group | n | observed | excess over cond | excess over tmpl |", "|---|---|---|---|---|"]
    for g in ("both_success", "flip"):
        r = o["q3"][g]
        L.append(f"| {g} | {r['n']} | {fmt(r['obs'])} | {fmt(r['excess_cond'])} {ci(r['excess_cond_ci'])} | "
                 f"{fmt(r['excess_tmpl'])} {ci(r['excess_tmpl_ci'])} |")
    q4 = o["q4"]
    L += ["", "## Q4. Flips", "",
          f"On {q4['n_flips']} primary-scope flips, {q4['also_on_succeeding_run']} of the "
          f"{q4['rules_on_failing_run']} rule hits on the failing run also fire on the succeeding run "
          f"(share {fmt(q4['share'], 2)}).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
