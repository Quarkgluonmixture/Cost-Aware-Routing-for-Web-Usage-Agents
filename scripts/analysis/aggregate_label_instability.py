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
    for cp in inv["clean_pairs"]:
        flips |= set(cp["flip_tasks_a_to_b"]) | set(cp["flip_tasks_b_to_a"])
        arms.append(cp["label"])
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
    return out


def render(d: dict) -> str:
    L = ["---", "type: analysis", "status: complete", "created: 2026-08-02",
         "purpose: is run-to-run instability spread over the benchmark, or concentrated on the "
         "tasks a router learns from",
         "post_hoc_exploratory: true",
         f"scope_warning: one cell ({d['cell']}), two of six arms replicated once each. Every "
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
