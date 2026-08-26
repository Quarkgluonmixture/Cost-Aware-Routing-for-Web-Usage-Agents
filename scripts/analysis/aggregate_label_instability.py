#!/usr/bin/env python3
"""Where does run-to-run instability sit relative to the tasks a router learns from?

`noise_floor_inventory.py` reports how much a rerun moves an aggregate. This asks a different
question about the same flips: are they spread evenly over the benchmark, or concentrated on the
tasks where the routing decision is actually contested?

The distinction decides how the result reads. "Half the routing labels are unstable" invites the
reply that benchmarks are noisy in general. "The instability is enriched N-fold exactly on the
decision boundary" is a statement about structure, and it bounds any per-task router regardless
of sample size, because no estimator repairs a target that is not reproducible.

Credit: the enrichment framing came from a zero-preset external review (2026-08-02) that was
given the numbers with no access to our draft. We had computed the 51% and never computed its
complement.

Scope. Exactly one cell carries two same-condition replicates (B0 x classifieds), and only two
of its six arms were replicated, once each. Every figure here is therefore a LOWER bound on the
flip rate: replicating the other four arms can only add flips, and adding flips to the
disagreement set is what the enrichment measures.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

LOG = logging.getLogger("label_instability")
INV = REPO / "docs/analysis/cross_sites/noise_floor_inventory.json"
SR = REPO / "results/phantom_paper/per_task_sr.csv"
OUT_MD = REPO / "docs/analysis/cross_sites/label_instability.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/label_instability.json"

CELL = "cls_B0"
SITE = "classifieds"
MODES = ["dom", "som", "vision", "ptext", "pprompt", "psom"]
TEXT = {"dom", "ptext", "pprompt", "psom"}
IMAGE = {"som", "vision"}


class MissingInput(RuntimeError):
    """Fail loud: a missing flip list would silently understate the enrichment."""


def load() -> tuple[set[int], dict[int, set[str]], list[str]]:
    if not INV.exists():
        raise MissingInput(f"{INV} missing; run aggregate_noise_floor_inventory.py first")
    inv = json.loads(INV.read_text())
    flips: set[int] = set()
    arms = []
    # B-1994 (2026-08-26): this module is scoped to ONE cell (`CELL`), but it used
    # to union the flip ids of EVERY registered pair. That was safe only while
    # CLEAN_PAIRS held nothing but B0.cls rows. It now also holds B1 and B5 rows
    # (same site, wrong baseline) and reddit rows — and reddit task ids occupy the
    # same integer range as classifieds ones while meaning different tasks, so a
    # reddit flip on task 145 would have been counted as a classifieds flip on
    # task 145. Unlike the sibling defect in retry_vs_switch_label_supply.py this
    # one fails SILENTLY: no shape check would catch it. The shipped artefact
    # predates the first foreign row, so nothing published is affected.
    _prefix = CELL.split("_")[1] + "." + CELL.split("_")[0] + "."   # cls_B0 -> B0.cls.
    for cp in inv["clean_pairs"]:
        if not cp["label"].startswith(_prefix):
            continue
        flips |= set(cp["flip_tasks_a_to_b"]) | set(cp["flip_tasks_b_to_a"])
        arms.append(cp["label"])
    if not arms:
        raise MissingInput(
            f"no clean pairs match cell {CELL} (prefix {_prefix!r}) in {INV}")
    if not flips:
        raise MissingInput("no flip task ids in the inventory")
    scored, _ = expected_scored_ids(SITE)
    solve = {}
    for r in csv.DictReader(SR.open()):
        if r["cell_id"] != CELL:
            continue
        t = int(r["task_id"])
        if t in scored:
            solve[t] = {m for m in MODES if float(r[f"sr_{m}"]) > 0}
    if len(solve) != len(scored):
        raise MissingInput(f"{CELL}: {len(solve)} tasks, canonical universe has {len(scored)}")
    return flips, solve, arms


def strata(solve: dict[int, set[str]]) -> dict[str, list[int]]:
    """Each stratum is the row set one routing formulation would actually train on."""
    all_tasks = list(solve)
    def three_way(s: set[str]) -> int:
        return len({bool(s & TEXT), "vision" in s, "som" in s})
    return {
        "which-mode label rows (any mode solved)":
            [t for t, s in solve.items() if s],
        "…of those, the arms DISAGREE (the choice matters)":
            [t for t, s in solve.items() if s and len(s) < len(MODES)],
        "three-way channel decision is contested":
            [t for t, s in solve.items() if three_way(s) > 1],
        "exactly one mode solved it (label unambiguous)":
            [t for t, s in solve.items() if len(s) == 1],
        "COMPLEMENT: no mode solved, or all did":
            [t for t, s in solve.items() if not s or len(s) == len(MODES)],
        "whole cell":
            all_tasks,
    }


def build() -> dict:
    flips, solve, arms = load()
    n = len(solve)
    out = {"schema": "2026-08-02-label-instability-v1", "post_hoc_exploratory": True,
           "cell": CELL, "n": n, "n_flipped": len(flips), "replicated_arms": arms,
           "strata": {}}
    base_key = "COMPLEMENT: no mode solved, or all did"
    st = strata(solve)
    base = st[base_key]
    base_rate = len(set(base) & flips) / len(base) if base else 0.0
    for name, ids in st.items():
        k = len(set(ids) & flips)
        rate = k / len(ids) if ids else 0.0
        out["strata"][name] = {
            "n_tasks": len(ids), "share_of_cell": 100 * len(ids) / n,
            "n_flipped": k, "flip_rate_pct": 100 * rate,
            "share_of_all_flips": 100 * k / len(flips) if flips else 0.0,
            "enrichment_vs_complement": (rate / base_rate) if base_rate else None,
        }
        LOG.info("%-52s n=%3d flip=%3d (%.1f%%)", name[:52], len(ids), k, 100 * rate)
    out["difficulty_null"] = difficulty_null(flips, solve)
    # Same computation with the replicated arms removed from the difficulty proxy — see the
    # docstring of difficulty_null for why the shipped version is circular.
    repl = {a.rsplit(".", 1)[-1] for a in arms}
    _proxy = [m for m in MODES if m not in repl]
    if _proxy:
        out["difficulty_null_leave_replicated_out"] = difficulty_null(
            flips, solve, arms_for_proxy=_proxy)
        out["leave_replicated_out_available"] = True
    else:
        # B-1995 (2026-08-26). The anti-circularity control has been DESTROYED BY
        # SUCCESS, not by a bug. It works by rebuilding the difficulty proxy from
        # the arms that were NOT replicated, so that an arm's solve status cannot
        # decide both "is this task contested" and "did it flip". When it was
        # written (2026-08-02) only dom and vision carried replicates and four
        # arms were free. The floor chain gave ptext/pprompt/psom replicates on
        # 2026-08-17/18 and som on 08-03, so all six are now replicated and the
        # control has no arms left to stand on. Previously this divided by zero.
        #
        # Reporting None is the honest outcome, NOT a degradation to the six-arm
        # figure: difficulty_null's own docstring says the two figures differ by
        # ~4x and that "neither may be quoted alone". With one of them structurally
        # unavailable, the surviving one may not be promoted to the headline — it
        # has lost the control that licensed it.
        #
        # The fix is not code. It is a different control: leave-ONE-out (rebuild
        # the proxy from the other five arms, per arm) does not require any arm to
        # be un-replicated. That changes the estimand, so it is a decision, not a
        # patch, and is deliberately not made here.
        out["difficulty_null_leave_replicated_out"] = None
        out["leave_replicated_out_available"] = False
        out["leave_replicated_out_unavailable"] = {
            "reason": "every mode in MODES now carries a replicate; no un-replicated "
                      "arm remains to build an independent difficulty proxy from",
            "replicated_modes": sorted(repl),
            "modes": list(MODES),
            "consequence": "the six-arm enrichment figure has lost its anti-circularity "
                           "control and must not be quoted as a headline on its own",
            "candidate_fix": "leave-one-out proxy (five other arms per arm) — changes the "
                             "estimand, needs an explicit decision",
        }
    out["replicated_arms_short"] = sorted(repl)
    return out


def difficulty_null(flips: set[int], solve: dict[int, set[str]],
                    arms_for_proxy: list[str] | None = None) -> dict:
    """How much of the enrichment is arithmetic rather than structure?

    "Contested" means at least one arm solved the task and at least one did not, which is by
    definition a mid-difficulty band. A task whose true per-run success rate is p flips between
    two runs with probability 2p(1-p) — maximal near p=0.5 and zero at either end. So a reviewer
    can ask whether the enrichment is just that the complement is full of tasks nobody solves.

    Taking k/6 (how many of the six modes solved it) as a difficulty proxy for p, this reports
    the flip rate the arithmetic alone would produce. Two things fall out, and they point in
    opposite directions — both belong in the paper:

      * the complement's predicted rate is exactly 0 (k=0 and k=6 both give 2p(1-p)=0), so the
        arithmetic enrichment is INFINITE. The observed 17x is therefore not inflated by this
        mechanism; it is deflated by it. Nine all-solved tasks flip twice, which the model
        forbids outright.
      * inside the contested band the observed rate exceeds the floor by only ~1.4x. So most of
        the 51% is the band being mid-difficulty, and the excess above that is what is left for
        "structure" to explain.

    ⚠️ TWO crudenesses, and the second is the serious one.
    (a) The six modes are different representations, not six draws from one model, so k/6
        estimates difficulty rather than p.
    (b) CIRCULARITY. The flips are defined by rerunning `dom` and `vision`. If those two arms
        also enter the difficulty proxy, then a task's solve status on them decides BOTH whether
        it counts as contested AND whether it counts as flipped. Pass `arms_for_proxy` to build
        the proxy from the OTHER four arms and break the loop; the caller reports both. The
        difference is large — 17.4x collapses to 3.95x — so neither figure may be quoted alone.
        (§H stress P0-3, 2026-08-02.)

    Note the six-arm version is still the correct operationalisation of the CLAIM (a router
    chooses among all six arms, so "contested" must be defined over all six). It is the
    difficulty control that must not reuse the replicated arms.
    """
    arms = list(arms_for_proxy) if arms_for_proxy is not None else list(MODES)
    K = len(arms)
    aset = set(arms)
    k = {t: len(s & aset) for t, s in solve.items()}
    contested = {t for t, s in solve.items() if 0 < len(s & aset) < K}
    comp = set(solve) - contested

    def obs(S): return len(S & flips) / len(S) if S else None

    def pred(S): return (sum(2 * (k[t] / K) * (1 - k[t] / K) for t in S) / len(S)) if S else None

    oc, ok_ = obs(contested), obs(comp)
    pc, pk = pred(contested), pred(comp)
    return {
        "proxy": f"k/{K} where k = how many of {arms} solved the task",
        "arms_in_proxy": arms,
        "circular": aset & {"dom", "vision"} == {"dom", "vision"},
        "contested": {"n": len(contested), "observed_flip_rate": oc, "binomial_floor": pc,
                      "observed_over_floor": (oc / pc) if pc else None},
        "complement": {"n": len(comp), "observed_flip_rate": ok_, "binomial_floor": pk},
        "arithmetic_enrichment": "infinite (complement floor is exactly 0)",
        "per_k": [{"k": kk,
                   "n": sum(1 for t in solve if k[t] == kk),
                   "n_flipped": sum(1 for t in solve if k[t] == kk and t in flips),
                   "observed": obs({t for t in solve if k[t] == kk}),
                   "binomial_floor": 2 * (kk / K) * (1 - kk / K)}
                  for kk in range(K + 1)
                  if any(k[t] == kk for t in solve)],
    }


def render(d: dict) -> str:
    L = ["---", "type: analysis", "status: complete", "created: 2026-08-02",
         "purpose: is run-to-run instability spread over the benchmark, or concentrated on the "
         "tasks a router learns from",
         "post_hoc_exploratory: true",
         # Arm count read from the data, never hardcoded: this line said "two of six" for a
         # day after the som replicate landed (2026-08-03), while the numbers below had
         # already moved. A stale scope note on fresh numbers is worse than no note.
         f"scope_warning: one cell ({d['cell']}), {len(d['replicated_arms'])} of "
         f"{len(MODES)} arms replicated once each. Every "
         "figure is a LOWER bound on the flip rate; replicating more arms can only add flips.",
         "producer: scripts/analysis/aggregate_label_instability.py", "---", "",
         "# Where the instability sits", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_label_instability.py`", "",
         f"Cell `{d['cell']}`, n = {d['n']}. Replicated arms: "
         + ", ".join(f"`{a}`" for a in d["replicated_arms"]) + f". "
         f"**{d['n_flipped']} of {d['n']} tasks change outcome between the two runs of at least "
         "one replicated arm.**", "",
         "| stratum | tasks | share of cell | flipped | flip rate | share of all flips | vs complement |",
         "|---|---|---|---|---|---|---|"]
    for name, s in d["strata"].items():
        en = s["enrichment_vs_complement"]
        ens = "—" if en is None else (f"**{en:.1f}x**" if en >= 1.5 else f"{en:.2f}x")
        bold = "**" if "DISAGREE" in name else ""
        L.append(f"| {bold}{name}{bold} | {s['n_tasks']} | {s['share_of_cell']:.1f}% | "
                 f"{s['n_flipped']} | {bold}{s['flip_rate_pct']:.1f}%{bold} | "
                 f"{s['share_of_all_flips']:.1f}% | {ens} |")
    dis = d["strata"]["…of those, the arms DISAGREE (the choice matters)"]
    comp = d["strata"]["COMPLEMENT: no mode solved, or all did"]
    L += ["", "## Reading", "",
          f"The tasks on which the arms disagree are the only rows a which-mode router can learn "
          f"from, and they are **{dis['share_of_cell']:.1f}%** of the cell. They carry "
          f"**{dis['share_of_all_flips']:.1f}%** of all observed flips. Their flip rate is "
          f"**{dis['flip_rate_pct']:.1f}%** against **{comp['flip_rate_pct']:.1f}%** on the "
          f"complement, an enrichment of **{dis['enrichment_vs_complement']:.1f}x**.", "",
          "This is not the same statement as 'the benchmark is noisy'. Aggregate success rate "
          "between these same two runs moves by under 2.3 points, which any reader would call "
          "reproducible. The per-task counterfactual labels that routing needs are not, and the "
          "gap between those two facts is the point: **instability concentrates precisely where "
          "the decision is contested**, so a router is fitted on the least stable subset of the "
          "benchmark by construction.", "",
          "It also bounds the problem independently of sample size. More data does not repair a "
          "target that a rerun rewrites, so this obstruction is of a different kind from the "
          "supply and predictability results, which a larger or easier benchmark could move."]

    # The obvious attack on the enrichment, answered with a number rather than an argument.
    dn = d["difficulty_null"]
    co, cm = dn["contested"], dn["complement"]
    L += ["", "## Is the enrichment just arithmetic?", "",
          "\"Contested\" means at least one arm solved the task and at least one did not, which "
          "is by definition a **mid-difficulty band**. A task with true per-run success rate *p* "
          "flips between two runs with probability *2p(1−p)*: maximal near 0.5, zero at either "
          "end. So the enrichment could be nothing but the complement being full of tasks nobody "
          "solves. Taking *k/6* — how many of the six modes solved it — as a difficulty proxy:",
          "",
          "| set | n | observed flip rate | binomial floor *2p(1−p)* |",
          "|---|---|---|---|",
          f"| contested | {co['n']} | {100 * co['observed_flip_rate']:.2f}% | "
          f"{100 * co['binomial_floor']:.2f}% |",
          f"| complement | {cm['n']} | {100 * cm['observed_flip_rate']:.2f}% | "
          f"{100 * cm['binomial_floor']:.2f}% |", "",
          "**The attack fails, and it fails in the unexpected direction.** The complement's "
          "predicted rate is exactly zero — *k*=0 and *k*=6 both give *2p(1−p)*=0 — so the "
          "arithmetic enrichment is **infinite**. The observed figure is therefore *deflated* by "
          "this mechanism, not inflated: the complement flips more than the model permits at "
          "all, including two of the nine tasks that every mode solved.", "",
          "**But the same table limits the claim.** Inside the contested band the observed rate "
          f"exceeds the floor by only **{co['observed_over_floor']:.2f}×** "
          f"({100 * co['observed_flip_rate']:.1f}% against {100 * co['binomial_floor']:.1f}%). "
          "Most of the 51% is the band being mid-difficulty; the excess above that floor is what "
          "is left for structure to carry. The honest sentence is that instability concentrates "
          "on contested tasks **and** that being contested is itself most of the reason.", "",
          "| *k* solved | n | flipped | observed | floor |",
          "|---|---|---|---|---|"]
    for r in dn["per_k"]:
        L.append(f"| {r['k']} | {r['n']} | {r['n_flipped']} | {100 * r['observed']:.2f}% | "
                 f"{100 * r['binomial_floor']:.2f}% |")
    L += ["", "⚠️ The proxy is crude: the six modes are different representations, not six draws "
          "from one model, so *k/6* estimates difficulty rather than *p*. The per-*k* rates are "
          "not monotone in the floor, which is itself evidence the proxy is imperfect.", ""]

    # The circularity, and what the number becomes without it.
    lo = d["difficulty_null_leave_replicated_out"]
    if lo is None:
        u = d["leave_replicated_out_unavailable"]
        L += ["### …and is the proxy circular?", "",
              "**Yes, and the control that used to answer this is no longer available.** "
              f"{u['reason']}. Replicated: `{'`, `'.join(u['replicated_modes'])}`.", "",
              f"⚠️ {u['consequence']}. Candidate fix: {u['candidate_fix']}.", "",
              "This is a consequence of the replicate inventory becoming *complete*, not of "
              "a defect — the control was only ever possible while some arm lacked a "
              "replicate. It is recorded here rather than silently dropped.", ""]
        return "\n".join(L)
    lc, lm = lo["contested"], lo["complement"]
    enr6 = co["observed_flip_rate"] / cm["observed_flip_rate"]
    enr4 = lc["observed_flip_rate"] / lm["observed_flip_rate"]
    L += ["### …and is the proxy circular?", "",
          f"Yes, partly. The flips are defined by rerunning **{'** and **'.join(d['replicated_arms_short'])}**, "
          "and those same two arms enter the six-mode difficulty proxy. A task's solve status on "
          "them therefore decides *both* whether it counts as contested *and* whether it counts "
          "as flipped. Rebuilding the proxy from the other four arms breaks the loop:", "",
          "| proxy | contested | complement | enrichment |", "|---|---|---|---|",
          f"| all six modes (as claimed) | {100 * co['observed_flip_rate']:.2f}% (n={co['n']}) | "
          f"{100 * cm['observed_flip_rate']:.2f}% (n={cm['n']}) | **{enr6:.2f}×** |",
          f"| the four not replicated | {100 * lc['observed_flip_rate']:.2f}% (n={lc['n']}) | "
          f"{100 * lm['observed_flip_rate']:.2f}% (n={lm['n']}) | **{enr4:.2f}×** |", "",
          f"**Neither figure may be quoted alone.** The six-mode version is the correct "
          "operationalisation of the *claim* — a router chooses among all six arms, so "
          "\"contested\" has to be defined over all six — but as a *difficulty control* it "
          "reuses the arms that define the outcome. The four-arm version is not circular and "
          f"still shows **{enr4:.2f}×** enrichment, with the complement rate rising from "
          f"{100 * cm['observed_flip_rate']:.2f}% to {100 * lm['observed_flip_rate']:.2f}% "
          "because genuinely unstable tasks move out of the contested set. The honest sentence "
          f"is that instability is enriched on contested tasks by somewhere between "
          f"{min(enr4, enr6):.1f}× and {max(enr4, enr6):.1f}× depending on whether the "
          "definition of contested is allowed to see the replicated arms."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    d = build()
    OUT_JSON.write_text(json.dumps(d, indent=2))
    OUT_MD.write_text(render(d))
    print(f"✓ {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
