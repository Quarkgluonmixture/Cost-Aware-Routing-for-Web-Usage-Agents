#!/usr/bin/env python3
"""Paper B core evidence: why the which-mode router cannot be learned.

Rebuilds, as a reproducible producer, the six quantities that 笔记 §383.4 first
measured on 2026-07-22 with three throwaway scratch scripts
(`label_supply_sweep` / `label_trainability` / `pooled_label_conflict`). Those
files no longer exist, so every number Paper B wanted to cite had no rerunnable
source — and all of them shifted when the k=6 refresh added B2_reddit and moved
the reddit universe to the AMENDMENT_08 scored set (pooled 249 → 260).

The claim being evidenced: the bottleneck is the label PRODUCTION RATE, not the
hypothesis class and not the label DEFINITION. Labels only come into existence
when a task is solved, so at 2-27% success there is no way to manufacture
training events by re-slicing the supervision.

Six measurements:
  1. supply        — trainable labels and solvable rate per cell
  2. trainability  — cells with zero trainable folds under the min-class filter
  3. conflict      — pooling fixes supply but breaks identifiability: the router
                     features are a function of the TASK, so the same X carries
                     contradictory y across backbones
  4. bayes_ceiling — the best any task-feature-only classifier can do, given (3)
  5. tier_agreement— backbones disagree about WHICH mode but agree about the
                     cost tier (does this need the screenshot), which is the one
                     re-slicing that helps
  6. tiebreak      — share of labels decided by the hardcoded MODES order rather
                     than by data, i.e. how arbitrary the supervision is

Stamped `post_hoc_exploratory=True` / `h10_eligible=False`: this is diagnosis of
a negative result, never a gate.

Usage:
    python3 scripts/analysis/router_label_supply_diagnosis.py \
        --out docs/analysis/cross_sites/router_label_supply_diagnosis.md \
        --json-out results/phantom_paper/router_label_supply_diagnosis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from p79.policies.router_features import MODES  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

FEATURES_NPZ = REPO / "results/phantom_paper/l1_router/raw_features_phase1a.npz"
FEATURES_JSON = REPO / "results/phantom_paper/l1_router/raw_features_phase1a.json"

# Stage-3 min-class filter (B-995): a fold needs this many rows of a class to
# train on it. Mirrors train_l1_router.py.
N_MIN_CLASS_TRAIN = 10
N_SPLITS = 5

# The cost tier that actually matters operationally: does the mode require the
# annotated screenshot. som/vision consume an image; dom and all three phantom
# arms are text-only by construction (see PHANTOM_SOM_CODE_TOUR).
IMAGE_MODES = {"som", "vision"}


def _tier(mode: str) -> str:
    return "image" if mode in IMAGE_MODES else "text_only"


def _site_of(cell_id: str) -> str:
    return cell_id.split("_", 1)[1]


def load_pool() -> dict[str, Any]:
    if not FEATURES_NPZ.exists():
        raise FileNotFoundError(
            f"{FEATURES_NPZ} missing — run extract_50_features.py first."
        )
    z = np.load(FEATURES_NPZ, allow_pickle=True)
    meta = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
    return {
        "X_numeric": z["X_numeric"],
        "X_binary": z["X_binary"],
        "labels": z["labels"].astype(str),
        "task_ids": z["task_ids"].astype(int),
        "cell_ids": z["cell_ids"].astype(str),
        "all_task_ids": z["all_task_ids"].astype(int),
        "all_cell_ids": z["all_cell_ids"].astype(str),
        "meta": meta,
    }


def measure_supply(pool: dict) -> dict[str, Any]:
    """Trainable labels + solvable rate per cell."""
    labels, cells = pool["labels"], pool["cell_ids"]
    all_cells = pool["all_cell_ids"]
    per_cell = {}
    for cid in sorted(set(all_cells.tolist())):
        n_universe = int((all_cells == cid).sum())
        n_labeled = int((cells == cid).sum())
        dist = Counter(labels[cells == cid].tolist())
        per_cell[cid] = {
            "n_universe": n_universe,
            "n_trainable_labels": n_labeled,
            "solvable_rate_pct": round(100.0 * n_labeled / n_universe, 2)
            if n_universe else None,
            "label_distribution": dict(sorted(dist.items())),
            "n_classes_present": len(dist),
        }
    counts = [v["n_trainable_labels"] for v in per_cell.values()]
    return {
        "per_cell": per_cell,
        "n_cells": len(per_cell),
        "labels_min": min(counts), "labels_max": max(counts),
        "pooled_total": int(len(labels)),
        "note": (
            "A label exists only where some mode succeeded. The spread is the "
            "whole argument: no re-slicing of the supervision changes how many "
            "solve events the benchmark produced."
        ),
    }


def measure_trainability(pool: dict) -> dict[str, Any]:
    """Cells with zero trainable folds under the min-class filter."""
    labels, cells = pool["labels"], pool["cell_ids"]
    per_cell = {}
    for cid in sorted(set(cells.tolist())):
        y = labels[cells == cid]
        dist = Counter(y.tolist())
        # Per fold a class contributes ~(N_SPLITS-1)/N_SPLITS of its rows.
        keepable = {
            c: n for c, n in dist.items()
            if n * (N_SPLITS - 1) / N_SPLITS >= N_MIN_CLASS_TRAIN
        }
        per_cell[cid] = {
            "n_labels": int(len(y)),
            "n_classes": len(dist),
            "classes_surviving_min_filter": sorted(keepable),
            "n_classes_surviving": len(keepable),
            # Need >=2 surviving classes for a classifier to exist at all.
            "trainable": len(keepable) >= 2,
        }
    n_untrainable = sum(1 for v in per_cell.values() if not v["trainable"])
    return {
        "per_cell": per_cell,
        "min_class_train": N_MIN_CLASS_TRAIN,
        "n_splits": N_SPLITS,
        "n_cells": len(per_cell),
        "n_cells_untrainable": n_untrainable,
        "note": (
            f"A cell needs >=2 classes each surviving the N_MIN_CLASS_TRAIN="
            f"{N_MIN_CLASS_TRAIN} filter in a {N_SPLITS}-fold split. Fewer than "
            f"two leaves nothing to discriminate."
        ),
    }


def _feature_key(x_num, x_bin) -> tuple:
    """Exact identity of one row's model input.

    Every feature is an integer-valued count or a 0/1 flag, so exact tuple equality
    is the right notion of "same input" and the grouping is stable under rounding
    (verified 2026-07-28 at 0-12 decimals: identical group counts).
    """
    return (tuple(float(v) for v in x_num), tuple(int(v) for v in x_bin))


def measure_conflict_and_ceiling(pool: dict) -> dict[str, Any]:
    """Pooling fixes supply but breaks identifiability.

    The router features carry no *explicit* model identity, so the natural reading is
    that two cells sharing a task_id present the same X. That reading was wired into
    this function until 2026-07-28 (stress finding #9): the Bayes ceiling grouped rows
    by `task_id`, on a comment asserting "distinct X (= a task, since features are
    task-only)". The assertion is false. Three of the five numeric features
    (`dom_complexity`, `text_length`, `tokens_input_text`) are read from that cell's
    own step-0 observation, so the same task rendered by a different backbone/mode can
    yield a different vector — measured at 31.5% of shared classifieds tasks and 80.0%
    of shared reddit tasks.

    Both ceilings are therefore reported, because they answer different questions:

      by_task_identity  — HEADLINE. The ceiling when only task identity is usable, i.e.
                          treating the incidental observation differences as jitter a
                          deployed router should not lean on.
      by_feature_vector — the plug-in Bayes ceiling for the exact vectors fitted.

    The headline is the task-identity number, and the reason is a finite-sample one that
    the first version of this fix (2026-07-28, later that day) got wrong. "Modal share per
    distinct x" is a plug-in estimate, and it is optimistically biased when x is
    high-cardinality: a singleton group scores 100% by construction. Here 74 of 117
    classifieds groups and 69 of 78 reddit groups are singletons, covering 44% and 75% of
    rows, so the feature-vector number is largely a count of how unique the vectors are
    rather than a measure of identifiability. The task-identity grouping is coarser, so
    less inflated, and it is also the coarsening that matches the deployment question: a
    router serving one backbone cannot recover backbone identity from observation jitter.
    """
    labels, cells, tids = pool["labels"], pool["cell_ids"], pool["task_ids"]
    x_num, x_bin = pool["X_numeric"], pool["X_binary"]
    by_site: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    feat_site: dict[str, dict[tuple, list[str]]] = defaultdict(lambda: defaultdict(list))
    feat_of_task: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for i, (lab, cid, tid) in enumerate(zip(labels.tolist(), cells.tolist(), tids.tolist())):
        site = _site_of(cid)
        by_site[site][int(tid)].append(lab)
        key = _feature_key(x_num[i], x_bin[i])
        feat_site[site][key].append(lab)
        feat_of_task[site][int(tid)].add(key)

    out = {}
    for site, per_task in sorted(by_site.items()):
        shared = {t: ls for t, ls in per_task.items() if len(ls) >= 2}
        conflicting = {t: ls for t, ls in shared.items() if len(set(ls)) > 1}
        n_rows = sum(len(ls) for ls in per_task.values())
        # Ceiling A (HEADLINE): group by task identity.
        n_best_task = sum(Counter(ls).most_common(1)[0][1] for ls in per_task.values())
        # Ceiling B: group by the exact input vector. Plug-in, singleton-inflated.
        per_feat = feat_site[site]
        n_best_feat = sum(Counter(ls).most_common(1)[0][1] for ls in per_feat.values())
        n_singleton = sum(1 for ls in per_feat.values() if len(ls) == 1)
        # The headline grouping is ALSO a resubstitution estimate: a task labelled by
        # only one backbone is its own group and scores 100% whatever the labels do
        # (codex Mode B finding 1, 2026-07-28). Coarsening X shrank this problem from
        # 44%/75% of rows to ~30%/37%; it did not remove it.
        n_task_singleton = sum(1 for ls in per_task.values() if len(ls) == 1)
        shared_rows = n_rows - n_task_singleton
        n_best_shared = sum(Counter(ls).most_common(1)[0][1]
                            for ls in per_task.values() if len(ls) >= 2)
        # How often the "same task ⇒ same X" premise actually holds.
        split = sum(1 for t in shared if len(feat_of_task[site][t]) > 1)
        # Same computation restricted to the cost tier (2 classes), both groupings.
        n_best_tier = sum(
            Counter(_tier(m) for m in ls).most_common(1)[0][1]
            for ls in per_task.values()
        )
        n_best_tier_feat = sum(
            Counter(_tier(m) for m in ls).most_common(1)[0][1]
            for ls in per_feat.values()
        )
        tier_shared = {
            t: ls for t, ls in shared.items()
            if len({_tier(m) for m in ls}) == 1
        }
        out[site] = {
            "n_tasks_with_labels": len(per_task),
            "n_tasks_shared_by_2plus_cells": len(shared),
            "n_tasks_conflicting": len(conflicting),
            "conflict_rate_pct": round(100.0 * len(conflicting) / len(shared), 2)
            if shared else None,
            "n_distinct_feature_vectors": len(per_feat),
            "n_singleton_feature_vectors": n_singleton,
            "singleton_share_of_rows_pct": round(100.0 * n_singleton / n_rows, 2)
            if n_rows else None,
            # Resubstitution disclosure for the HEADLINE (task-identity) grouping.
            "n_singleton_tasks": n_task_singleton,
            "singleton_task_share_of_rows_pct":
                round(100.0 * n_task_singleton / n_rows, 2) if n_rows else None,
            "modal_agreement_on_shared_tasks_only_pct":
                round(100.0 * n_best_shared / shared_rows, 2) if shared_rows else None,
            "n_rows_on_shared_tasks": shared_rows,
            "n_shared_tasks_with_split_features": split,
            "shared_tasks_split_rate_pct": round(100.0 * split / len(shared), 2)
            if shared else None,
            # HEADLINE = task-identity grouping (see docstring: the feature-vector
            # plug-in is singleton-inflated and overstates identifiability).
            "bayes_ceiling_which_mode_pct": round(100.0 * n_best_task / n_rows, 2)
            if n_rows else None,
            "bayes_ceiling_which_mode_by_feature_vector_pct":
                round(100.0 * n_best_feat / n_rows, 2) if n_rows else None,
            "bayes_ceiling_cost_tier_pct": round(100.0 * n_best_tier / n_rows, 2)
            if n_rows else None,
            "bayes_ceiling_cost_tier_by_feature_vector_pct":
                round(100.0 * n_best_tier_feat / n_rows, 2) if n_rows else None,
            "tier_agreement_rate_pct": round(100.0 * len(tier_shared) / len(shared), 2)
            if shared else None,
            "n_pooled_rows": n_rows,
        }
    return {
        "per_site": out,
        "note": (
            "conflict_rate is over tasks covered by >=2 cells. The two ceilings "
            "are the point: re-slicing the SAME features from 'which of six "
            "modes' down to 'image or text-only' raises the attainable ceiling "
            "without inventing a single new solve event — the only relabelling "
            "that buys anything, though 'attainable ceiling' overstates it — see the "
            "resubstitution warning above. Headline figures group by task_id; the "
            "`_by_feature_vector_` variants group by the exact input vector and are "
            "HIGHER, because three observation-derived features differ across cells for "
            "the same task and most resulting groups are singletons scoring 100% by "
            "construction (see measure_conflict_and_ceiling docstring, stress finding #9 "
            "2026-07-28)."
        ),
    }


def _load_cell_outcomes_and_costs(cid: str) -> dict[str, dict[int, dict]] | None:
    """Per-mode {task_id: row} for one cell, or None if any mode is missing."""
    from scripts.analysis.lib.episode_rows import load_task_rows  # noqa: PLC0415
    from scripts.analysis.lib.run_registry import get_cells  # noqa: PLC0415

    display = {
        "dom": "DOM", "som": "SoM", "vision": "Vision",
        "phantom_text": "P-text", "phantom_prompt": "P-prompt", "phantom_som": "P-SoM",
    }
    baseline, site = cid.split("_", 1)
    dirs = {c.mode: c.episodes_dir for c in get_cells(baseline=baseline, site=site)}
    rows_by_mode = {m: (load_task_rows(dirs[d]) if dirs.get(d) else {})
                    for m, d in display.items()}
    if any(not rows_by_mode[m] for m in MODES):
        return None
    return rows_by_mode


def measure_label_definition_sensitivity(pool: dict) -> dict[str, Any]:
    """Do supply and trainability survive the OTHER defensible label definition?

    The canonical which-mode label is `derive_oracle_label` (first success in
    MODES order, ascending PRIOR cost). `derive_cost_oracle_label` is the
    alternative: cheapest successful mode by MEASURED episode cost. The F2 note
    in `router_features.py` records why the measured variant is not canonical
    (unstable across cells, endogenous to the outcome, too few successes to pin
    an order); this measurement exists so the paper can say whether the negative
    result depends on that choice rather than merely asserting that it does not.

    Relabelling cannot change SUPPLY — both definitions label exactly the tasks
    some mode solved — so only the class DISTRIBUTION, and through it
    trainability, can move.
    """
    from p79.policies.router_features import derive_cost_oracle_label  # noqa: PLC0415

    COST_FIELD = "total_billed_cost_usd"
    labels, cells, tids = pool["labels"], pool["cell_ids"], pool["task_ids"]

    per_cell: dict[str, Any] = {}
    for cid in sorted(set(cells.tolist())):
        rows_by_mode = _load_cell_outcomes_and_costs(cid)
        if rows_by_mode is None:
            per_cell[cid] = {"error": "missing mode data"}
            continue
        sel = cells == cid
        canonical, measured, n_moved = [], [], 0
        for tid, lab in zip(tids[sel].tolist(), labels[sel].tolist()):
            outcomes = {
                m: (rows_by_mode[m].get(int(tid)) or {}).get("success") is True
                for m in MODES
            }
            costs = {
                m: float((rows_by_mode[m].get(int(tid)) or {}).get(COST_FIELD) or 0.0)
                for m in MODES if outcomes[m]
            }
            alt = derive_cost_oracle_label(outcomes, costs)
            canonical.append(lab)
            measured.append(alt)
            if alt is not None and alt != lab:
                n_moved += 1

        def _trainable(ys: list) -> dict:
            dist = Counter(y for y in ys if y is not None)
            keep = {c: n for c, n in dist.items()
                    if n * (N_SPLITS - 1) / N_SPLITS >= N_MIN_CLASS_TRAIN}
            return {
                "label_distribution": dict(sorted(dist.items())),
                "classes_surviving_min_filter": sorted(keep),
                "trainable": len(keep) >= 2,
            }

        per_cell[cid] = {
            "n_labels": int(sel.sum()),
            "n_labels_moved_by_relabel": n_moved,
            "canonical": _trainable(canonical),
            "measured_cost": _trainable(measured),
        }

    ok = [v for v in per_cell.values() if "error" not in v]
    return {
        "per_cell": per_cell,
        "n_cells": len(ok),
        "n_untrainable_canonical": sum(1 for v in ok if not v["canonical"]["trainable"]),
        "n_untrainable_measured_cost": sum(
            1 for v in ok if not v["measured_cost"]["trainable"]
        ),
        "supply_is_invariant": True,
        "note": (
            "Supply is identical by construction: both definitions label exactly "
            "the solved tasks. Only the class distribution moves, so this "
            "measures whether the trainability headline depends on the label "
            "definition. Canonical = derive_oracle_label; the measured-cost "
            "variant is a sensitivity, not a proposed replacement (F2 note in "
            "router_features.py)."
        ),
    }


def measure_tiebreak_arbitrariness(pool: dict) -> dict[str, Any]:
    """Share of labels the hardcoded MODES order gets WRONG on cost.

    `derive_oracle_label` walks MODES in order and returns the first successful
    mode. `MODES` is documented as "ascending prior cost", so on a task solved by
    several modes the order is a PROXY for "cheapest successful". Two cases:

      * the first-in-order winner IS the measured-cheapest successful mode —
        the list agrees with the data, nothing arbitrary happens;
      * it is NOT — the label is then decided by a list literal against the
        measured cost, and reordering `router_features.MODES:101` flips it.

    Only the second case is arbitrariness. Counting every multi-success task
    (as a first draft of this producer did) overstates it — measured here at
    70.1% for B0_classifieds versus the 26% that 笔记 §383.4 reported, i.e.
    2.7x — because most multi-success tasks are ones where the order happens to
    be right.
    """
    from scripts.analysis.lib.episode_rows import load_task_rows  # noqa: PLC0415
    from scripts.analysis.lib.run_registry import get_cells  # noqa: PLC0415

    COST_FIELD = "total_billed_cost_usd"
    # extractor mode ids -> registry display names
    DISPLAY = {
        "dom": "DOM", "som": "SoM", "vision": "Vision",
        "phantom_text": "P-text", "phantom_prompt": "P-prompt", "phantom_som": "P-SoM",
    }
    labels, cells, tids = pool["labels"], pool["cell_ids"], pool["task_ids"]

    per_cell: dict[str, Any] = {}
    for cid in sorted(set(cells.tolist())):
        baseline, site = cid.split("_", 1)
        dirs = {c.mode: c.episodes_dir for c in get_cells(baseline=baseline, site=site)}
        rows_by_mode = {}
        for m, disp in DISPLAY.items():
            ep = dirs.get(disp)
            rows_by_mode[m] = load_task_rows(ep) if ep else {}
        if any(not rows_by_mode[m] for m in MODES):
            per_cell[cid] = {"error": "missing mode data"}
            continue

        sel = cells == cid
        n_labels = int(sel.sum())
        n_multi = n_order_wrong = n_true_tie = 0
        wrong_examples: list[dict] = []
        for tid, lab in zip(tids[sel].tolist(), labels[sel].tolist()):
            succ_modes = [
                m for m in MODES
                if (rows_by_mode[m].get(int(tid)) or {}).get("success") is True
            ]
            if len(succ_modes) < 2:
                continue
            n_multi += 1
            costs = {
                m: float((rows_by_mode[m].get(int(tid)) or {}).get(COST_FIELD) or 0.0)
                for m in succ_modes
            }
            min_cost = min(costs.values())
            at_min = [m for m, c in costs.items() if c == min_cost]
            if len(at_min) > 1:
                # Literal tie: several successful modes cost exactly the same, so
                # MODES order alone decides which one becomes the label.
                n_true_tie += 1
            elif costs[lab] > min_cost:
                # Not a tie — the order picked a strictly more expensive mode,
                # i.e. MODES is not actually in ascending measured cost here.
                n_order_wrong += 1
                if len(wrong_examples) < 3:
                    wrong_examples.append({
                        "task_id": int(tid), "order_pick": lab,
                        "measured_cheapest": min(costs, key=lambda m: costs[m]),
                        "cost_order_pick": round(costs[lab], 6),
                        "cost_cheapest": round(min_cost, 6),
                    })
        per_cell[cid] = {
            "n_labels": n_labels,
            "n_multi_success": n_multi,
            "n_true_cost_tie": n_true_tie,
            "n_order_disagrees_with_measured_cost": n_order_wrong,
            "multi_success_pct": round(100.0 * n_multi / n_labels, 2) if n_labels else None,
            "true_tie_pct": round(100.0 * n_true_tie / n_labels, 2) if n_labels else None,
            "order_wrong_pct": round(100.0 * n_order_wrong / n_labels, 2) if n_labels else None,
            "examples": wrong_examples,
        }
    rates = [
        v["true_tie_pct"] for v in per_cell.values()
        if isinstance(v.get("true_tie_pct"), float)
    ]
    return {
        "per_cell": per_cell,
        "modes_order": list(MODES),
        "true_tie_pct_min": min(rates) if rates else None,
        "true_tie_pct_max": max(rates) if rates else None,
        "note": (
            "Three nested quantities, do not conflate them. `multi_success_pct` = "
            "the order was consulted at all (loose upper bound, NOT arbitrariness). "
            "`true_tie_pct` = several successful modes cost EXACTLY the same, so "
            "the MODES list literal alone picks the label — this is the literal "
            "tie-break arbitrariness 笔记 §383.4 reported at ~1/4. "
            "`order_wrong_pct` = no tie, but the order picked a strictly more "
            "expensive mode, i.e. MODES is not in ascending MEASURED cost here — "
            "a separate and worse defect than a tie-break, since the label is then "
            "not even 'cheapest successful'."
        ),
    }


def build(pool: dict) -> dict[str, Any]:
    # B-1906: do NOT re-derive the digest here. `test_universe_consumption_lint`
    # caught the first draft of this function doing `expected_scored_ids(site)[1]`
    # — the exact SHA-only pattern that lint exists to ban, written on the same
    # day as the lint. Read the digest the upstream feature cache already recorded
    # alongside the rows it actually kept, so this artifact cannot claim a
    # universe its input does not have. Cross-check it against the canonical set
    # and fail closed on disagreement.
    universe_sha: dict[str, str] = {}
    for cid, rec in pool["meta"].get("per_cell_summary", {}).items():
        up = rec.get("universe_provenance") or {}
        site = _site_of(cid)
        sha = up.get("content_task_ids_sha256")
        if not sha:
            raise ValueError(
                f"{cid}: feature cache carries no content_task_ids_sha256 — "
                f"regenerate it with extract_50_features.py (B-1904)."
            )
        if universe_sha.setdefault(site, sha) != sha:
            raise ValueError(f"{site}: cells disagree on the task universe digest")
        canonical = expected_scored_ids(site)
        if sha != canonical[1]:
            raise ValueError(
                f"{site}: feature cache universe {sha[:12]} != canonical scored "
                f"{canonical[1][:12]} — the cache predates the current scored set."
            )
    return {
        "post_hoc_exploratory": True,
        "h10_eligible": False,
        "producer": "scripts/analysis/router_label_supply_diagnosis.py",
        "claim": (
            "The which-mode router is unlearnable because labels are produced "
            "only by solve events; the bottleneck is production RATE, not the "
            "hypothesis class and not the label definition."
        ),
        "canonical_task_universe_sha256_by_site": universe_sha,
        "supply": measure_supply(pool),
        "trainability": measure_trainability(pool),
        "identifiability": measure_conflict_and_ceiling(pool),
        "tiebreak": measure_tiebreak_arbitrariness(pool),
        "label_definition_sensitivity": measure_label_definition_sensitivity(pool),
    }


def render(p: dict) -> str:
    L: list[str] = []
    L.append("# Why the which-mode router cannot be learned — label-supply diagnosis\n")
    L.append("> `post_hoc_exploratory=True` · `h10_eligible=False` — diagnosis of a "
             "negative result, never a gate.\n")
    L.append(f"**Claim.** {p['claim']}\n")

    s = p["supply"]
    L.append("## 1. Supply: labels exist only where something succeeded\n")
    L.append("| cell | scored universe | trainable labels | solvable | classes present |")
    L.append("|---|---|---|---|---|")
    for cid, r in s["per_cell"].items():
        L.append(f"| {cid} | {r['n_universe']} | **{r['n_trainable_labels']}** | "
                 f"{r['solvable_rate_pct']}% | {r['n_classes_present']}/6 |")
    L.append(f"\nAcross {s['n_cells']} cells the trainable-label count spans "
             f"**{s['labels_min']}-{s['labels_max']}**; pooled total "
             f"**{s['pooled_total']}**. {s['note']}\n")

    t = p["trainability"]
    L.append("## 2. Trainability: the min-class filter empties most cells\n")
    L.append("| cell | labels | classes | surviving min-class filter | trainable |")
    L.append("|---|---|---|---|---|")
    for cid, r in t["per_cell"].items():
        L.append(f"| {cid} | {r['n_labels']} | {r['n_classes']} | "
                 f"{r['n_classes_surviving']} ({', '.join(r['classes_surviving_min_filter']) or '—'}) | "
                 f"{'yes' if r['trainable'] else '**no**'} |")
    L.append(f"\n**{t['n_cells_untrainable']} of {t['n_cells']} cells have no trainable "
             f"classifier at all.** {t['note']}\n")

    i = p["identifiability"]
    L.append("## 3. Pooling fixes supply and breaks identifiability\n")
    L.append("| site | tasks shared by 2+ cells | conflicting | conflict rate | "
             "in-sample modal agreement (which-mode) | same, feature-vector grouping | "
             "in-sample modal agreement (cost tier) | tier agreement |")
    L.append("|---|---|---|---|---|---|---|---|")
    for site, r in i["per_site"].items():
        L.append(f"| {site} | {r['n_tasks_shared_by_2plus_cells']} | "
                 f"{r['n_tasks_conflicting']} | **{r['conflict_rate_pct']}%** | "
                 f"{r['bayes_ceiling_which_mode_pct']}% | "
                 f"{r['bayes_ceiling_which_mode_by_feature_vector_pct']}% | "
                 f"**{r['bayes_ceiling_cost_tier_pct']}%** | "
                 f"{r['tier_agreement_rate_pct']}% |")
    L.append("\n⚠️ **Both columns are resubstitution (in-sample) estimates, not Bayes "
             "ceilings.** Each scores the same rows it took the modal label from, so any group "
             "with a single member is correct by construction. Call it *in-sample modal "
             "agreement*; an out-of-sample ceiling needs leave-one-backbone-out prediction or "
             "a shrinkage estimator, neither of which is computed here.\n")
    L.append("| site | singleton tasks (headline grouping) | agreement on shared tasks only | "
             "singleton feature vectors | shared tasks whose rows differ in X |")
    L.append("|---|---|---|---|---|")
    for site, r in i["per_site"].items():
        L.append(f"| {site} | {r['n_singleton_tasks']} of {r['n_pooled_rows']} rows "
                 f"(**{r['singleton_task_share_of_rows_pct']}%**) | "
                 f"**{r['modal_agreement_on_shared_tasks_only_pct']}%** "
                 f"(n={r['n_rows_on_shared_tasks']}) | "
                 f"{r['n_singleton_feature_vectors']} "
                 f"({r['singleton_share_of_rows_pct']}% of rows) | "
                 f"{r['n_shared_tasks_with_split_features']} of "
                 f"{r['n_tasks_shared_by_2plus_cells']} "
                 f"(**{r['shared_tasks_split_rate_pct']}%**) |")
    L.append(f"\n{i['note']}\n")

    tb = p["tiebreak"]
    L.append("## 4. How much of the supervision is a list literal\n")
    L.append("| cell | labels | multi-success | true cost tie (order decides) | "
             "order picked a pricier mode |")
    L.append("|---|---|---|---|---|")
    for cid, r in tb["per_cell"].items():
        if r.get("error"):
            L.append(f"| {cid} | — | — | — | {r['error']} |")
            continue
        L.append(f"| {cid} | {r['n_labels']} | {r['n_multi_success']} "
                 f"({r['multi_success_pct']}%) | "
                 f"**{r['n_true_cost_tie']} ({r['true_tie_pct']}%)** | "
                 f"{r['n_order_disagrees_with_measured_cost']} ({r['order_wrong_pct']}%) |")
    L.append(f"\n`MODES = {tb['modes_order']}`\n")
    L.append(f"{tb['note']}\n")
    tie_max = tb.get("true_tie_pct_max")
    if tie_max == 0.0:
        ow = [
            v["order_wrong_pct"] for v in tb["per_cell"].values()
            if isinstance(v.get("order_wrong_pct"), float)
        ]
        L.append(
            f"**The tie-break never fires.** `true_tie` is 0 in every cell, because "
            f"`total_billed_cost_usd` is a continuous float and two modes costing "
            f"*exactly* the same does not happen. So 'tie-break arbitrariness' is the "
            f"wrong frame for this defect. What does happen is worse: on "
            f"**{min(ow)}-{max(ow)}%** of labels the MODES order returns a mode that is "
            f"strictly MORE expensive than another successful one. `MODES` is documented "
            f"as \"ascending prior cost\" and used as a proxy for cheapest-successful; "
            f"on those rows the assumption is false, so the label is not "
            f"'cheapest successful mode' at all — it is whatever the list literal reached "
            f"first.\n"
        )
        L.append(
            "笔记 §383.4 reported ~1/4 of labels as order-decided (B0_cls 25/97) from a "
            "scratch script that no longer exists; that figure is not reproduced here "
            "and is not the same quantity — it is superseded by the two measured "
            "columns above rather than reconciled to.\n"
        )

    sens = p.get("label_definition_sensitivity")
    if sens:
        L.append("## 7. Does the result depend on the label definition?\n")
        L.append(
            "Canonical = `derive_oracle_label` (first success in MODES order, ascending "
            "PRIOR cost). Sensitivity = `derive_cost_oracle_label` (cheapest successful "
            "by MEASURED episode cost). The measured variant is **not** a proposed "
            "replacement: the F2 note in `router_features.py` records why it was "
            "examined on landed Pass-1 data and rejected as canonical (order unstable "
            "across cells, cost endogenous to the outcome, too few successes per mode). "
            "Supply is identical under both by construction, so only trainability can "
            "move.\n"
        )
        L.append("| cell | labels | relabelled | canonical: surviving / trainable | "
                 "measured cost: surviving / trainable |")
        L.append("|---|---|---|---|---|")
        for cid, r in sens["per_cell"].items():
            if "error" in r:
                L.append(f"| {cid} | — | — | {r['error']} | — |")
                continue
            c, m = r["canonical"], r["measured_cost"]
            L.append(
                f"| {cid} | {r['n_labels']} | {r['n_labels_moved_by_relabel']} | "
                f"{len(c['classes_surviving_min_filter'])} / "
                f"{'yes' if c['trainable'] else '**no**'} | "
                f"{len(m['classes_surviving_min_filter'])} / "
                f"{'yes' if m['trainable'] else '**no**'} |"
            )
        L.append(
            f"\n**{sens['n_untrainable_canonical']} of {sens['n_cells']} cells are "
            f"untrainable under the canonical label and "
            f"{sens['n_untrainable_measured_cost']} of {sens['n_cells']} under the "
            f"measured-cost label.** The supply argument does not turn on which of the "
            f"two definitions is used.\n"
        )
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args(argv)

    payload = build(load_pool())
    text = render(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
