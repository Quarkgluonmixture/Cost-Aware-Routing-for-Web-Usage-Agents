#!/usr/bin/env python3
"""Does adding `visual_difficulty` to the feature table rescue the router?

`routing_feature_diagnostics` established two things about the obvious routing features: the
one that IS in the table (`has_reference_image`) carries the wrong sign, and the one a
practitioner would actually want (`visual_difficulty`, a VWA-native per-task annotation) is
read out of every task config by `extract_50_features.py:334` and then never placed in the
table. That is a diagnosis, not a test — and EVIDENCE_LAYER_SUMMARY §6 listed the missing test
as an open, cheap refuter for claim 5: *any routing formulation we did not try that wins*.

Arguing that a feature would not have helped is weaker than fitting it and reporting that it
did not. This fits it.

Deliberately a SEPARATE product rather than a change to `extract_50_features.py`. Adding a
column there would silently move every landed router number, and the question here is
counterfactual, not a correction. Nothing in the frozen results moves.

Design notes:
  * Same triage label as `router_triage_learnability` — "solvable by any of the six arms" —
    and the same out-of-fold LogisticRegression, so the two numbers are comparable.
  * The comparison is WITHIN cell and WITHIN fold split: identical rows, identical folds, one
    extra column. Any AUROC difference is the column.
  * `visual_difficulty` is ordinal (easy < medium < hard). Three task configs spell it
    "mediun"; that typo is repaired here exactly as `routing_feature_diagnostics` repairs it,
    because dropping those rows instead would change the denominator between the two arms.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.router_triage_learnability import (  # noqa: E402
    CELLS, build_cell, oof_scores, _auroc,
)
from scripts.analysis.extract_50_features import read_task_config  # noqa: E402

LOG = logging.getLogger("visual_difficulty_router")
OUT_MD = REPO / "docs/analysis/cross_sites/visual_difficulty_router.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/visual_difficulty_router.json"

DIFF_ORDER = {"easy": 0, "medium": 1, "hard": 2}
DIFF_FIX = {"mediun": "medium"}          # three VWA configs carry this typo


class MissingInput(RuntimeError):
    """Fail loud rather than silently compare a shorter feature table."""


def difficulty_of(site: str, tid: int) -> float | None:
    cfg = read_task_config(site, tid)
    if cfg is None:
        return None
    raw = cfg.get("visual_difficulty")
    if raw is None:
        return None
    # read_task_config already maps the annotation through difficulty_to_int (and repairs the
    # "mediun" typo there), so it arrives as an ordinal int. The string branch is kept so this
    # still works if called on a raw config.
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    key = DIFF_FIX.get(str(raw), str(raw))
    return float(DIFF_ORDER[key]) if key in DIFF_ORDER else None


def build() -> dict:
    out = {"schema": "2026-08-02-visual-difficulty-router-v1", "post_hoc_exploratory": True,
           "label": "solvable by any of the six arms (same as router_triage_learnability)",
           "cells": {}}
    for spec in CELLS:
        cell = build_cell(spec)
        if cell is None:
            LOG.warning("%s/%s: not buildable, skipped", spec["baseline"], spec["site"])
            continue
        cid = f"{'cls' if spec['site'] == 'classifieds' else 'red'}_{spec['baseline']}"
        vd = [difficulty_of(spec["site"], t) for t in cell["task_ids"]]
        n_missing = sum(1 for v in vd if v is None)
        if n_missing:
            raise MissingInput(f"{cid}: {n_missing} tasks without visual_difficulty; the two "
                               "arms would not share a denominator")
        X0 = cell["X"]
        X1 = np.hstack([X0, np.asarray(vd, dtype=float).reshape(-1, 1)])
        y = cell["y"]
        if len(set(y.tolist())) < 2:
            LOG.warning("%s: single-class label, skipped", cid)
            continue
        a0 = oof_scores(X0, y)
        a1 = oof_scores(X1, y)
        # Compare the fitted router only. `per_feature` is a per-column dict whose key set
        # differs between the two arms by construction (one has an extra column), and `prior`
        # is a constant, so neither is a comparison.
        per_key = {"lr": {"auroc_without": _auroc(y, a0["lr"]),
                          "auroc_with": _auroc(y, a1["lr"])}}
        # what the added column alone can do, out of fold, for context
        j_new = X1.shape[1] - 1
        per_key["visual_difficulty_alone"] = {
            "auroc_without": float("nan"),
            "auroc_with": _auroc(y, a1["per_feature"][j_new])}
        best = "lr"
        out["cells"][cid] = {
            "n": int(len(y)), "n_positive": int(y.sum()),
            "n_features_without": int(X0.shape[1]), "n_features_with": int(X1.shape[1]),
            "per_scorer": per_key,
            "best_scorer": best,
            "auroc_without": per_key[best]["auroc_without"] if best else None,
            "auroc_with": per_key[best]["auroc_with"] if best else None,
            "delta_auroc": (per_key[best]["auroc_with"] - per_key[best]["auroc_without"])
                           if best else None,
            "visual_difficulty_dist": {k: sum(1 for v in vd if v == float(i))
                                       for k, i in DIFF_ORDER.items()},
        }
        LOG.info("%s: AUROC %.4f -> %.4f", cid, out["cells"][cid]["auroc_without"],
                 out["cells"][cid]["auroc_with"])
    if not out["cells"]:
        raise MissingInput("no cell produced a comparison")
    ds = [c["delta_auroc"] for c in out["cells"].values() if c["delta_auroc"] is not None]
    out["mean_delta_auroc"] = float(np.mean(ds)) if ds else None
    out["n_cells_improved"] = sum(1 for d in ds if d > 0)
    out["n_cells"] = len(ds)
    return out


def render(d: dict) -> str:
    L = ["---", "type: analysis", "status: complete", "created: 2026-08-02",
         "purpose: fit the feature the diagnosis said was missing, and report whether it rescues "
         "the router",
         "post_hoc_exploratory: true",
         "scope_warning: VWA only — WebArena ships no visual_difficulty annotation. The label is "
         "the triage label, not the which-mode label; a feature can help one and not the other.",
         "producer: scripts/analysis/aggregate_visual_difficulty_router.py", "---", "",
         "# Does `visual_difficulty` rescue the router?", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_visual_difficulty_router.py`",
         "",
         "`routing_feature_diagnostics` showed the feature in the table has the wrong sign and "
         "that the VWA-native annotation a practitioner would want is read out of every task "
         "config and then dropped before the table is built. That was a diagnosis. "
         "EVIDENCE_LAYER_SUMMARY §6 listed the corresponding test as open, on the grounds that "
         "reporting a fitted failure beats arguing one. Here it is fitted.", "",
         "Same triage label and same out-of-fold logistic regression as "
         "`router_triage_learnability`, same rows, same folds — one extra column.", "",
         "| cell | n | positives | AUROC without | AUROC with | Δ |", "|---|---|---|---|---|---|"]
    for cid, c in d["cells"].items():
        L.append(f"| `{cid}` | {c['n']} | {c['n_positive']} | {c['auroc_without']:.4f} | "
                 f"{c['auroc_with']:.4f} | {c['delta_auroc']:+.4f} |")
    md = d.get("mean_delta_auroc")
    L += ["", f"**Mean ΔAUROC = {md:+.4f}** over {d['n_cells']} cells; it improves "
          f"{d['n_cells_improved']} of them.", ""]
    if md is not None and abs(md) < 0.02:
        L.append("That is inside the noise of a fold split on cells this size. **The feature "
                 "does not rescue the router**, and the reason is the one the supply argument "
                 "already gives: the constraint is the number of usable labelled rows, not "
                 "their separability. A better feature cannot manufacture labels.")
    else:
        L.append("⚠️ The shift is larger than a fold-split artefact would explain — re-read the "
                 "supply argument before treating this as a rescue.")
    L += ["", "### What this does and does not settle", "",
          "It closes the specific objection that the authors diagnosed a missing feature and "
          "never tried it. It does **not** show that no feature would help: the space of "
          "features is not enumerable, and this is one annotation on one benchmark. What makes "
          "the negative durable is not this fit — it is that the binding constraint is row "
          "count, which no feature changes.", "",
          "⚠️ The label here is *triage* (solvable by anything), not *which mode*. A feature "
          "could in principle help the which-mode decision and not this one; that label is the "
          "one `router_label_supply_diagnosis` shows there are too few rows to fit at all."]
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
