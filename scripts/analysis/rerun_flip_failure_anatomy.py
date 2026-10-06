#!/usr/bin/env python3
"""Anatomy of rerun flips: when the same task succeeds in one run and fails in the other.

Why (实验笔记 §531.10). Both workshop reviews (REALM, VLM4RWD) accepted the rerun control as
the strongest contribution and said the label-supply explanation for why routing cannot be
learned is "not yet established". The explanation says the rows a router must learn are the
rows reruns flip. This product asks what a flip looks like from the failing side, over all
registered same-condition pairs (`CLEAN_PAIRS`), with zero new compute.

Questions and reading rules, fixed before the numbers were computed (2026-10-06):

  Q1  Is the failure TYPE stable across reruns?  Tasks failing in both runs of a pair; share
      with the same paper-taxonomy bucket (aggregate_failure_modes.PAPER_TAXONOMY), and
      Cohen's kappa against the agreement the two runs' bucket marginals give by chance.
      kappa >= 0.6: the failure type is a property of (task, condition).
      kappa <  0.4: the failure type is itself rerun noise.  In between: say so, no verdict.
  Q2  How do flips fail?  Bucket of the failing run on flipped tasks, against the bucket
      distribution of tasks failing in both runs of the same pairs.  Descriptive only.
  Q3  When do the two runs part?  First step at which the action signatures differ
      (action type, url after the action without host, typed text), by task group
      (flip / both-success / both-fail).  Share already apart at step 0 and median step.
  Q4  How late is a flip decided?  For flips, the shared prefix as a fraction of the
      successful run's length.  Near 0 = the runs never walked together; near 1 = they
      walked together almost to the end and parted at the last decisions.

Inputs
  CLEAN_PAIRS (aggregate_noise_floor_inventory.py, read with ast — no import)
  results/diag_scans/reason_rows_20261006/<label>/{a,b}/episode_reason_rows.csv
      both arms re-derived with the CURRENT analyze_reason_diagnostics.py, so a bucket
      difference cannot be a script-version difference
  <arm>/episodes/*_steps_v2.jsonl  (exact name; `.stale_*` quarantine files never match)
Scored universe per site from canonical_task_universe.  Episodes whose step file length
disagrees with the summary's step count are excluded and counted (§400.1 identity class).

Outputs: docs/analysis/cross_sites/rerun_flip_failure_anatomy.{md,json}
Regenerate: python scripts/analysis/rerun_flip_failure_anatomy.py
"""
from __future__ import annotations

import ast
import collections
import csv
import json
import statistics
import sys
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.aggregate_failure_modes import fine_to_paper  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

REASON_ROOT = REPO / "results/diag_scans/reason_rows_20261006"
OUT_MD = REPO / "docs/analysis/cross_sites/rerun_flip_failure_anatomy.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/rerun_flip_failure_anatomy.json"
SITE_OF = {"cls": ("classifieds", "visualwebarena"), "red": ("reddit", "visualwebarena"),
           "wared": ("reddit", "webarena")}


class MissingInput(RuntimeError):
    pass


def clean_pairs() -> list[tuple[str, str, str]]:
    src = (REPO / "scripts/analysis/aggregate_noise_floor_inventory.py").read_text(encoding="utf-8")
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "CLEAN_PAIRS" for t in n.targets):
            return ast.literal_eval(n.value)
    raise MissingInput("CLEAN_PAIRS not found")


def reason_rows(label: str, side: str) -> dict[int, dict]:
    p = REASON_ROOT / label / side / "episode_reason_rows.csv"
    if not p.exists():
        raise MissingInput(f"{p} missing — run analyze_reason_diagnostics for this arm first")
    with p.open(encoding="utf-8", newline="") as f:
        return {int(r["task_id"]): r for r in csv.DictReader(f)}


def _sig(rec: dict) -> tuple:
    a = rec.get("action") or {}
    at = rec.get("action_type") or (a.get("action_type") if isinstance(a, dict) else None)
    u = urlsplit((rec.get("state_digest") or {}).get("url_after") or "")
    text = (a.get("text") or "").strip().lower() if isinstance(a, dict) and at == "type" else ""
    return (at, u.path + ("?" + u.query if u.query else ""), text)


def steps(arm: Path, site: str, tid: int) -> list[tuple] | None:
    stem = f"{site}_task_{tid}"
    sp, sm = arm / "episodes" / f"{stem}_steps_v2.jsonl", arm / "episodes" / f"{stem}_summary_v2.json"
    if not sp.exists() or not sm.exists():
        return None
    recs = [json.loads(x) for x in sp.read_text(encoding="utf-8").splitlines() if x.strip()]
    # dedup on step_idx, last write wins (resume can append a repeated step)
    by = {}
    for r in recs:
        by[r.get("step_idx")] = r
    n_sum = json.loads(sm.read_text(encoding="utf-8")).get("steps")
    if n_sum is not None and int(n_sum) != len(by):
        return None
    return [_sig(by[k]) for k in sorted(by, key=lambda x: (x is None, x))]


def first_divergence(x: list[tuple], y: list[tuple]) -> int:
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return min(len(x), len(y))


def kappa(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    agree = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return {"n": n, "agreement": agree, "chance": pe,
            "kappa": (agree - pe) / (1 - pe) if pe < 1 else None}


def build() -> dict:
    out = {"schema": "2026-10-06-rerun-flip-anatomy-v1", "pairs": {}}
    pooled_both_fail, pooled_flip_bucket, pooled_both_bucket = [], collections.Counter(), collections.Counter()
    div = collections.defaultdict(list)
    late = []
    excluded_identity = 0
    for label, a_path, b_path in clean_pairs():
        site, bench = SITE_OF[label.split(".")[1]]
        scored = expected_scored_ids(site, bench)[0]
        ra, rb = reason_rows(label, "a"), reason_rows(label, "b")
        common = sorted(scored & set(ra) & set(rb))
        if len(common) != len(scored):
            raise MissingInput(f"{label}: {len(common)} of {len(scored)} scored tasks in both arms")
        both_fail, groups = [], collections.Counter()
        for t in common:
            sa, sb = ra[t]["success"] == "True", rb[t]["success"] == "True"
            ba, bb = fine_to_paper(ra[t]["reason_bucket"]), fine_to_paper(rb[t]["reason_bucket"])
            g = "both_success" if sa and sb else "both_fail" if not (sa or sb) else "flip"
            groups[g] += 1
            if g == "both_fail":
                both_fail.append((ba, bb))
                pooled_both_bucket[ba] += 1
                pooled_both_bucket[bb] += 1
            elif g == "flip":
                pooled_flip_bucket[bb if sa else ba] += 1
            xa = steps(REPO / a_path, site, t)
            xb = steps(REPO / b_path, site, t)
            if xa is None or xb is None:
                excluded_identity += 1
                continue
            d = first_divergence(xa, xb)
            div[g].append(d)
            if g == "flip":
                succ_len = len(xa) if sa else len(xb)
                late.append(d / succ_len if succ_len else 0.0)
        pooled_both_fail += both_fail
        out["pairs"][label] = {"n": len(common), "groups": dict(groups), "q1": kappa(both_fail)}
    out["q1_pooled"] = kappa(pooled_both_fail)
    # POST HOC (added after the pooled number was seen): the pooled kappa mixes pairs with no
    # flips at all (near-deterministic local runs, kappa ~1) with stochastic API pairs. Report
    # the strata so the pooled verdict cannot be read as describing the API backbones.
    strata = collections.defaultdict(list)
    for label, a_path, b_path in clean_pairs():
        p = out["pairs"][label]
        strata[label.split(".")[0]].append(label)
        if p["groups"].get("flip", 0) >= 5:
            strata["pairs with >= 5 flips"].append(label)
    out["q1_strata_post_hoc"] = {
        k: {"pairs": len(v), "kappa_median": statistics.median(
            [out["pairs"][l]["q1"]["kappa"] for l in v if out["pairs"][l]["q1"].get("kappa") is not None])}
        for k, v in strata.items()}
    out["q2"] = {"flip_failing_side": dict(pooled_flip_bucket),
                 "both_fail_per_run": dict(pooled_both_bucket)}
    out["q3"] = {g: {"n": len(v), "share_apart_at_step0": sum(1 for x in v if x == 0) / len(v),
                     "median_first_divergence": statistics.median(v)}
                 for g, v in div.items() if v}
    out["q4"] = {"n": len(late), "median": statistics.median(late) if late else None,
                 "share_below_0.25": sum(1 for x in late if x < 0.25) / len(late) if late else None,
                 "share_at_or_above_0.75": sum(1 for x in late if x >= 0.75) / len(late) if late else None}
    out["excluded_identity_or_missing"] = excluded_identity
    return out


def _verdict(k: float | None) -> str:
    if k is None:
        return "undefined"
    return ("failure type is a property of (task, condition)" if k >= 0.6 else
            "failure type is itself rerun noise" if k < 0.4 else
            "between the pre-declared thresholds — no verdict")


def render(d: dict) -> str:
    q1 = d["q1_pooled"]
    L = ["---", "type: analysis", "status: complete", "post_hoc_exploratory: true",
         "purpose: what a rerun flip looks like from the failing side, over every registered pair",
         "producer: scripts/analysis/rerun_flip_failure_anatomy.py", "---", "",
         "# Rerun flips: how the failing run fails, and when the two runs part", "",
         "Regenerate: `python scripts/analysis/rerun_flip_failure_anatomy.py`. Questions and "
         "reading thresholds were fixed in the producer docstring before computing.", "",
         f"{len(d['pairs'])} registered same-condition pairs. Episodes excluded for a step-file / "
         f"summary length mismatch or a missing file: {d['excluded_identity_or_missing']} "
         "(Q3/Q4 only).", "",
         "## Q1. Is the failure type stable across reruns?", "",
         f"Tasks failing in both runs: **{q1['n']}**. Same paper bucket in both runs: "
         f"**{100*q1['agreement']:.1f}%**; chance from the marginals {100*q1['chance']:.1f}%; "
         f"**kappa = {q1['kappa']:.2f}** → {_verdict(q1['kappa'])}.", "",
         "| pair | n | flip | both fail | same bucket | kappa |", "|---|---|---|---|---|---|"]
    for lab, p in d["pairs"].items():
        k = p["q1"]
        L.append(f"| `{lab}` | {p['n']} | {p['groups'].get('flip', 0)} | {p['groups'].get('both_fail', 0)} | "
                 + (f"{100*k['agreement']:.0f}% | {k['kappa']:.2f} |" if k.get("n") and k.get("kappa") is not None
                    else "— | — |"))
    L += ["", "**Post hoc strata** (added after the pooled value was seen; median of per-pair kappa):", "",
          "| stratum | pairs | median kappa |", "|---|---|---|"]
    for k, v in d["q1_strata_post_hoc"].items():
        L.append(f"| {k} | {v['pairs']} | {v['kappa_median']:.2f} |")
    L += ["", "⚠️ The pooled kappa clears the 0.6 line partly because two B1·classifieds pairs have "
          "no flips and agree almost perfectly. Read the strata before quoting the pooled verdict."]
    fl, bf = d["q2"]["flip_failing_side"], d["q2"]["both_fail_per_run"]
    nf, nb = sum(fl.values()), sum(bf.values())
    L += ["", "## Q2. How do flips fail?", "",
          "Bucket of the failing run on flipped tasks, against both-fail tasks (each run counted).",
          "", "| paper bucket | flips (failing side) | both-fail |", "|---|---|---|"]
    for b in sorted(set(fl) | set(bf), key=lambda b: -(fl.get(b, 0) / max(nf, 1))):
        L.append(f"| {b} | {fl.get(b, 0)} ({100*fl.get(b, 0)/max(nf, 1):.0f}%) | "
                 f"{bf.get(b, 0)} ({100*bf.get(b, 0)/max(nb, 1):.0f}%) |")
    L += ["", "## Q3. When do the two runs part?", "",
          "First step at which (action type, url after the action, typed text) differ.", "",
          "| group | tasks | already apart at step 0 | median first divergence |", "|---|---|---|---|"]
    for g in ("flip", "both_success", "both_fail"):
        if g in d["q3"]:
            q = d["q3"][g]
            L.append(f"| {g} | {q['n']} | {100*q['share_apart_at_step0']:.0f}% | {q['median_first_divergence']} |")
    q4 = d["q4"]
    if q4["n"]:
        L += ["", "## Q4. How late is a flip decided?", "",
              f"Shared prefix as a fraction of the successful run's length, over {q4['n']} flips: "
              f"median **{q4['median']:.2f}**; below 0.25: {100*q4['share_below_0.25']:.0f}%; "
              f"0.75 or more: {100*q4['share_at_or_above_0.75']:.0f}%."]
    L += ["", "⚠️ Paper buckets come from `analyze_reason_diagnostics.py` (a rule-based reason "
          "classifier), not from /diag rules or human reading. Action signatures ignore element "
          "ids (SoM ids are re-keyed per page), so two runs clicking different elements that lead "
          "to the same URL count as not yet diverged."]
    return "\n".join(L) + "\n"


def main() -> int:
    d = build()
    OUT_JSON.write_text(json.dumps(d, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(d), encoding="utf-8")
    print(f"✓ {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
