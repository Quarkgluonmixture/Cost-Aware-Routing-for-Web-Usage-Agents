#!/usr/bin/env python3
"""Intelligent-baseline ladder for the learned router (paper §6.5, B-1006 R5 defense).

WHAT THIS IS
------------
Four "intelligent" (non-random) baselines that bound the learned router's lift
from above and below on the phantom routing space. They are paper §6 DISCLOSURE
rows — NOT a gating criterion. The H10 operational deployment gate
(`aggregate_h10_pareto.py`) is unchanged; this script only produces a side table
the reviewer can read to see the router sits in the sensible region of the
(Cost, SR) plane.

THE LADDER (paper_drafts/section6_router.md §6.5)
-------------------------------------------------
  (a) always-cheapest-mode  (= always-DOM in the 5-arm set)
        Cost-axis LOWER BOUND. No single-mode policy can be cheaper than always
        routing to the cheapest mode, so its mean cost is the global cost floor.
        The router's "cost-aware" claim is meaningful only relative to this point.

  (b) decision-stump single-feature  (depth-1 tree, e.g. "DOM_tokens > T -> A else B")
        Bounds the value of the FULL feature set over a one-feature heuristic. A
        depth-1 tree splits on the single most informative feature and emits at
        most two modes (exactly the §6.5 `>10K -> P-text else DOM` shape). If the
        18/53-feature router barely beats this, the extra features buy little.

  (c) per-task-lookup-table  (infinite-capacity reductio)
        SR UPPER BOUND. A model with one free parameter per task can memorise each
        task's oracle-best mode; its SR is the absolute ceiling (= oracle SR). This
        is implemented as a direct {task_id -> cheapest-successful-mode} table,
        which is the zero-regularisation limit of "LR + task_id one-hot evaluated
        in-sample" (we use the table because an LR with n≈p one-hot columns hits
        perfect separation / non-convergence without changing the result). The
        router's generalisation HEADROOM = per_task_lookup_SR - router_SR.

  (d) LR-DOM-features-only  (text-feature ablation)
        Bounds the contribution of the rich text features. In the canonical Phase
        1a pipeline this drops the 30 TF-IDF columns and keeps intent-regex +
        browser state. The pre-fire ARCHIVE has no TF-IDF bank (it predates the
        53-dim E'' extractor), so the archive proxy instead drops the 4 intent-
        regex semantic features and keeps browser/structural state only — the same
        KIND of ablation (how much does text signal add?), reported as such.

Plus `learned_router_proxy` = the 8-dim balanced LR (mirrors
`l1_archive_simulation.py` Variant B) — the thing being bounded on archive. Paper
§6 main number is the Phase 1a Pass-2 fire learned router; this proxy stands in
for development-sanity bounding only.

CAVEATS (read first)
--------------------
- SANITY-CHECK ONLY, same Option-C caveat as `l1_archive_simulation.py`: archive
  outcomes are pre-fire / directional. Numbers here cannot enter the paper as SR
  claims; paper §6 main data is the Phase 1a Pass-2 fire.
- B0-only / classifieds-only on the current host until reddit + B1/B2 land. The
  driver auto-discovers whatever cells have >= MIN_MODES_FOR_LADDER modes of data.
- DISCLOSURE, NOT GATING. Nothing here feeds the H10 pass/fail verdict.

OUTPUTS
-------
- docs/checkpoints/router/intelligent_baseline_ladder_archive.{md,json}
      human-readable dev-sanity report (version-controlled)
- results/phantom_paper/l1_router/intelligent_baseline_ladder_disclosure.json
      machine artifact read by aggregate_h10_pareto.py (gitignored; regenerate
      post-fire before running the H10 aggregator for matching numbers)

USAGE
-----
    python -m scripts.analysis.intelligent_baseline_ladder            # all discoverable cells
    python -m scripts.analysis.intelligent_baseline_ladder --baseline B0 --site classifieds
    # equivalently (sibling-import safe):
    python scripts/analysis/intelligent_baseline_ladder.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Archive episodes are pre-fire vintage; tolerate legacy cost fields. Set BEFORE
# importing aggregate (it reads the flag at module import). Fire episodes carry
# total_billed_cost_usd so this is a no-op on paper-grade data.
os.environ.setdefault("P79_ALLOW_LEGACY_COST", "1")

# Sibling imports (scripts/analysis on path whether run as -m or as a file).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from aggregate_h10_pareto import (  # noqa: E402  (sibling import after path setup)
    find_pass1_run_dirs,
    collect_per_task_outcomes_with_metrics,
)

# Shared single-source feature + oracle definitions (router /stress B-1805~B-1807):
# train ≡ serve ≡ archive ≡ ladder by construction.
from p79.policies.router_features import (  # noqa: E402
    COLOR_RE,
    COMPARE_RE,
    NAV_RE,
    SEARCH_RE,
    MODES,
    derive_oracle_label,
)

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ROUTER_ARTIFACT_DIR = REPO / "results/phantom_paper/l1_router"
DISCLOSURE_PATH = ROUTER_ARTIFACT_DIR / "intelligent_baseline_ladder_disclosure.json"
REPORT_MD = REPO / "docs/checkpoints/router/intelligent_baseline_ladder_archive.md"
REPORT_JSON = REPO / "docs/checkpoints/router/intelligent_baseline_ladder_archive.json"

SCHEMA_VERSION = "ladder-2026-06-02-v1"
ALL_CELLS = [(b, s) for b in ("B0", "B1", "B2") for s in ("classifieds", "reddit")]
MIN_MODES_FOR_LADDER = 2  # need >=2 modes for routing to be non-trivial


# ── Feature spec (8-dim archive proxy, mirrors l1_archive_simulation) ──────────
# (name, kind, is_numeric_scaled). kind="text_semantic" features are the
# droppable text signal for the arm-(d) ablation; "structural" stay.
FEATURE_SPEC: list[tuple[str, str, bool]] = [
    ("site_cls", "structural", False),         # cell-constant within a per-cell run
    ("has_image", "structural", False),
    ("intent_color", "text_semantic", False),
    ("intent_search", "text_semantic", False),
    ("intent_compare", "text_semantic", False),
    ("intent_nav", "text_semantic", False),
    ("intent_tok_count", "structural", True),  # numeric -> per-fold StandardScaler
    ("axtree_elements", "structural", True),   # numeric -> per-fold StandardScaler
]
FEATURE_NAMES = [f[0] for f in FEATURE_SPEC]
TEXT_SEMANTIC = {f[0] for f in FEATURE_SPEC if f[1] == "text_semantic"}


@dataclass
class LadderConfig:
    seed: int = 42
    n_splits: int = 5
    bootstrap_n: int = 1000


@dataclass
class ArmResult:
    arm: str
    role: str
    sr_mean: float
    sr_ci: tuple[float, float]
    cost_mean: float
    cost_ci: tuple[float, float]
    n: int
    routed_mode_dist: dict[str, int] = field(default_factory=dict)
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "role": self.role,
            "sr_mean_pct": round(100 * self.sr_mean, 3),
            "sr_ci_95_pct": [round(100 * self.sr_ci[0], 3), round(100 * self.sr_ci[1], 3)],
            "cost_mean_usd": round(self.cost_mean, 6),
            "cost_ci_95_usd": [round(self.cost_ci[0], 6), round(self.cost_ci[1], 6)],
            "n": self.n,
            "routed_mode_dist": self.routed_mode_dist,
            "note": self.note,
        }


# ── Feature loading (run-local task_configs + DOM step-0) ──────────────────────
def _find_dom_run(run_dirs: list[Path]) -> Optional[Path]:
    for d in run_dirs:
        if "_dom_" in d.name:
            return d
    return run_dirs[0] if run_dirs else None


def load_cell_features(
    site: str, run_dirs: list[Path], task_ids: list[int]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Per-task 8-dim feature dict from run-local task_configs + DOM step-0.

    obs_1 features are mode-agnostic (entry-page AXTree is the same regardless of
    which mode the agent will use); we read axtree size from the DOM run's step-0,
    matching `p1_archive_simulation.load_task_features`. Returns (features, prov).
    """
    prov: dict[str, Any] = {"axtree_defaulted": 0, "intent_missing": 0}
    # task config (intent, image) — present identically in every run's task_configs/
    cfg_dir: Optional[Path] = None
    for d in run_dirs:
        cand = d / "task_configs"
        if cand.is_dir():
            cfg_dir = cand
            break
    # axtree element count from DOM run step-0 records
    dom_run = _find_dom_run(run_dirs)
    ep_dir: Optional[Path] = None
    if dom_run is not None:
        eps = list(dom_run.glob("phase1_*/episodes"))
        ep_dir = eps[0] if eps else None

    feats: dict[int, dict[str, Any]] = {}
    for tid in task_ids:
        intent = ""
        has_image = False
        if cfg_dir is not None:
            cfg_file = cfg_dir / f"{site}_task_{tid}.json"
            if cfg_file.exists():
                try:
                    cfg = json.loads(cfg_file.read_text())
                    intent = cfg.get("intent", "") or ""
                    has_image = cfg.get("image") not in (None, "None", "")
                except (OSError, json.JSONDecodeError):
                    pass
        if not intent:
            prov["intent_missing"] += 1
        axtree = 0
        if ep_dir is not None:
            steps_file = ep_dir / f"{site}_task_{tid}_steps_v2.jsonl"
            if steps_file.exists():
                try:
                    with steps_file.open() as f:
                        step0 = json.loads(f.readline())
                    axtree = int(step0.get("state_digest", {}).get("dom_complexity", 0) or 0)
                except (OSError, json.JSONDecodeError, ValueError):
                    axtree = 0
        if axtree == 0:
            prov["axtree_defaulted"] += 1
        feats[tid] = {
            "site_cls": 1.0 if site == "classifieds" else 0.0,
            "has_image": 1.0 if has_image else 0.0,
            "intent_color": 1.0 if COLOR_RE.search(intent) else 0.0,
            "intent_search": 1.0 if SEARCH_RE.search(intent) else 0.0,
            "intent_compare": 1.0 if COMPARE_RE.search(intent) else 0.0,
            "intent_nav": 1.0 if NAV_RE.search(intent) else 0.0,
            "intent_tok_count": float(len(intent.split())),
            "axtree_elements": float(axtree),
        }
    return feats, prov


# ── Matrix / oracle helpers ────────────────────────────────────────────────────
def _success_dict(modes: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {m: bool(v.get("success", 0)) for m, v in modes.items()}


def _design_matrix(
    feats: dict[int, dict[str, Any]], tids: list[int], feature_names: list[str]
) -> np.ndarray:
    return np.array(
        [[feats[t][name] for name in feature_names] for t in tids], dtype=float
    )


def _numeric_idx(feature_names: list[str]) -> list[int]:
    scaled = {f[0] for f in FEATURE_SPEC if f[2]}
    return [i for i, name in enumerate(feature_names) if name in scaled]


# ── Estimator factories ────────────────────────────────────────────────────────
def _make_lr(numeric_idx: list[int]):
    """Balanced multinomial LR with per-fold StandardScaler on numeric columns.

    Mirrors l1_archive_simulation Variant B (P1-11 per-fold scaling — no train/test
    leak). class_weight='balanced' prevents collapse to the majority oracle mode.
    """
    if numeric_idx:
        pre = ColumnTransformer(
            [("scale", StandardScaler(), numeric_idx)], remainder="passthrough"
        )
        return Pipeline(
            [
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
            ]
        )
    return LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")


def _make_stump(seed: int):
    """Depth-1 decision tree = single-feature, two-outcome heuristic (§6.5 (b)).

    Scale-invariant (no scaler needed). balanced class_weight makes it the
    STRONGEST honest one-feature rule (tighter bound on feature-set value) rather
    than a strawman that always predicts the majority mode.
    """
    return DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=seed)


# ── CV routing ─────────────────────────────────────────────────────────────────
def _cv_predict_modes(
    X: np.ndarray,
    oracle_labels: list[Optional[str]],
    estimator_factory: Callable[[], Any],
    cfg: LadderConfig,
) -> list[str]:
    """Stratified 5-fold CV. Train on train-fold TRAINABLE tasks (non-None oracle),
    predict a mode for EVERY test-fold task. Returns predicted mode per row of X.

    Held-out evaluation (a task is never predicted by a model trained on itself),
    matching the router's E'' task-held-out protocol philosophy. Tasks with no
    successful mode (oracle None) are excluded from TRAINING but still routed at
    test time (they fail whatever mode → 0 SR contribution, like the real router).
    """
    n = len(oracle_labels)
    preds: list[Optional[str]] = [None] * n
    # Stratify on oracle label; None -> its own stratum to keep fold balance.
    strat = np.array([lab if lab is not None else "_nosucc" for lab in oracle_labels])
    # StratifiedKFold needs >=n_splits members per class; merge rare strata into a
    # pooled "_rare" bucket only for the SPLIT (training still uses true labels).
    counts = {c: int((strat == c).sum()) for c in set(strat)}
    strat_for_split = np.array(
        [c if counts[c] >= cfg.n_splits else "_rare" for c in strat]
    )
    if len(set(strat_for_split)) < 2 or min(
        np.bincount(np.unique(strat_for_split, return_inverse=True)[1])
    ) < cfg.n_splits:
        # Not enough structure to stratify cleanly → single in-sample-free fallback
        # via plain KFold-like contiguous split.
        skf_splits = _contiguous_folds(n, cfg.n_splits)
    else:
        skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        skf_splits = list(skf.split(X, strat_for_split))

    for tr_idx, te_idx in skf_splits:
        tr_trainable = [i for i in tr_idx if oracle_labels[i] is not None]
        y_tr = [oracle_labels[i] for i in tr_trainable]
        if len(set(y_tr)) < 2:
            # Degenerate train fold (one or zero classes): predict the single class
            # (or cheapest mode if none) for all test tasks.
            only = y_tr[0] if y_tr else MODES[0]
            for i in te_idx:
                preds[i] = only
            continue
        est = estimator_factory()
        est.fit(X[tr_trainable], y_tr)
        te_pred = est.predict(X[te_idx])
        for i, p in zip(te_idx, te_pred):
            preds[i] = str(p)
    # Any unfilled (shouldn't happen) → cheapest mode.
    return [p if p is not None else MODES[0] for p in preds]


def _contiguous_folds(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic contiguous k-fold index split (stratification fallback)."""
    idx = np.arange(n)
    folds = np.array_split(idx, k)
    out = []
    for j in range(k):
        te = folds[j]
        tr = np.concatenate([folds[m] for m in range(k) if m != j]) if k > 1 else te
        out.append((tr, te))
    return out


# ── Per-arm outcome vectors ─────────────────────────────────────────────────────
def _outcome_vectors(
    outcomes: dict[int, dict[str, dict[str, Any]]],
    tids: list[int],
    routed_modes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Given a routed mode per task, return (success_arr, cost_arr) over tids.

    A routed mode missing from a task's outcome row (defensive; full-coverage
    archive has all modes) counts as failure at the cheapest mode's cost.
    """
    succ, cost = [], []
    for t, m in zip(tids, routed_modes):
        row = outcomes[t]
        if m in row:
            succ.append(int(row[m]["success"]))
            cost.append(float(row[m]["cost_usd"]))
        else:
            fallback = next((mm for mm in MODES if mm in row), None)
            succ.append(0)
            cost.append(float(row[fallback]["cost_usd"]) if fallback else 0.0)
    return np.array(succ, dtype=float), np.array(cost, dtype=float)


def _paired_bootstrap(
    success: np.ndarray, cost: np.ndarray, cfg: LadderConfig
) -> dict[str, Any]:
    """Per-task paired bootstrap (matches aggregate_h10_pareto.paired_bootstrap)."""
    rng = np.random.default_rng(cfg.seed)
    n = len(success)
    if n == 0:
        nan = float("nan")
        return {"sr_mean": nan, "sr_ci": (nan, nan), "cost_mean": nan, "cost_ci": (nan, nan), "n": 0}
    sr_reps, cost_reps = [], []
    for _ in range(cfg.bootstrap_n):
        idx = rng.integers(0, n, n)
        sr_reps.append(float(success[idx].mean()))
        cost_reps.append(float(cost[idx].mean()))
    return {
        "sr_mean": float(success.mean()),
        "sr_ci": (float(np.percentile(sr_reps, 2.5)), float(np.percentile(sr_reps, 97.5))),
        "cost_mean": float(cost.mean()),
        "cost_ci": (float(np.percentile(cost_reps, 2.5)), float(np.percentile(cost_reps, 97.5))),
        "n": n,
    }


def _mode_dist(routed_modes: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in routed_modes:
        out[m] = out.get(m, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ── Orchestrator ────────────────────────────────────────────────────────────────
def evaluate_ladder_for_cell(
    baseline: str,
    site: str,
    outcomes: dict[int, dict[str, dict[str, Any]]],
    feats: dict[int, dict[str, Any]],
    cfg: LadderConfig,
) -> dict[str, Any]:
    """Compute the 5-arm ladder (4 baselines + learned-router proxy) for one cell."""
    tids = sorted(outcomes.keys())
    n = len(tids)
    oracle_labels = [derive_oracle_label(_success_dict(outcomes[t])) for t in tids]

    # Mean cost per mode (total-billed; for empirical cheapest-mode disclosure).
    modes_present = sorted({m for t in tids for m in outcomes[t]})
    mean_cost = {
        m: float(np.mean([outcomes[t][m]["cost_usd"] for t in tids if m in outcomes[t]]))
        for m in modes_present
    }
    empirical_cheapest = min(mean_cost, key=mean_cost.get) if mean_cost else MODES[0]
    cost_vals = list(mean_cost.values())
    cost_spread_pct = (
        100 * (max(cost_vals) - min(cost_vals)) / min(cost_vals)
        if cost_vals and min(cost_vals) > 0 else 0.0
    )
    # §6.5 names the cost-only baseline "always-DOM"; route to dom by name (cheapest
    # text-only mode BY DESIGN). The cheapest-by-total-billed mode is disclosed
    # separately — on archive the single modes are near-tied (phantom drop-in cost
    # equivalence, paper §3) so dom is not always strictly min-total-cost.
    dom_mode = "dom" if "dom" in modes_present else empirical_cheapest

    arms: dict[str, ArmResult] = {}

    def _arm(name: str, role: str, routed: list[str], note: str = "") -> ArmResult:
        s, c = _outcome_vectors(outcomes, tids, routed)
        bs = _paired_bootstrap(s, c, cfg)
        return ArmResult(
            arm=name, role=role, sr_mean=bs["sr_mean"], sr_ci=bs["sr_ci"],
            cost_mean=bs["cost_mean"], cost_ci=bs["cost_ci"], n=bs["n"],
            routed_mode_dist=_mode_dist(routed), note=note,
        )

    # (a) always-cheapest-mode = always-DOM (§6.5 cost-only baseline)
    routed_a = [dom_mode] * n
    note_a = (
        f"= always-DOM (§6.5 cost-only baseline). empirical cheapest single mode by "
        f"total-billed cost = '{empirical_cheapest}'"
        + (
            ""
            if empirical_cheapest == dom_mode
            else f"; DOM not strictly cheapest (5 single modes near-tied, spread {cost_spread_pct:.1f}%)"
        )
    )
    arms["always_cheapest_dom"] = _arm("always_cheapest_dom", "cost_only_baseline", routed_a, note_a)

    # (c) per-task lookup table (oracle) — SR upper bound; no-success tasks → dom (cheap, fails anyway)
    routed_c = [lab if lab is not None else dom_mode for lab in oracle_labels]
    n_nosucc = sum(1 for lab in oracle_labels if lab is None)
    arms["per_task_lookup"] = _arm(
        "per_task_lookup", "sr_upper_bound", routed_c,
        note=f"oracle / infinite-capacity ceiling; {n_nosucc} task(s) have no successful mode",
    )

    # learned_router_proxy (full 8-dim balanced LR, held-out CV) — the thing bounded
    X_full = _design_matrix(feats, tids, FEATURE_NAMES)
    routed_lr = _cv_predict_modes(
        X_full, oracle_labels, lambda: _make_lr(_numeric_idx(FEATURE_NAMES)), cfg
    )
    arms["learned_router_proxy"] = _arm(
        "learned_router_proxy", "bounded_subject", routed_lr,
        note="8-dim balanced LR (mirrors l1_archive_simulation Variant B); archive proxy for the Phase 1a fire router",
    )

    # (b) decision-stump single feature
    routed_b = _cv_predict_modes(X_full, oracle_labels, lambda: _make_stump(cfg.seed), cfg)
    arms["decision_stump"] = _arm(
        "decision_stump", "feature_floor", routed_b,
        note="depth-1 tree (single feature, <=2 modes); bounds value of the full feature set",
    )

    # (d) LR-DOM-features-only (drop text-semantic features)
    dom_feature_names = [f[0] for f in FEATURE_SPEC if f[1] != "text_semantic"]
    X_dom = _design_matrix(feats, tids, dom_feature_names)
    routed_d = _cv_predict_modes(
        X_dom, oracle_labels, lambda: _make_lr(_numeric_idx(dom_feature_names)), cfg
    )
    arms["lr_dom_features_only"] = _arm(
        "lr_dom_features_only", "text_ablation", routed_d,
        note=(
            "LR on browser/structural features only (archive proxy: drops the 4 intent-regex "
            "semantic features; canonical Phase 1a arm drops the 30 TF-IDF columns). Bounds text-signal value."
        ),
    )

    bounds = _bounding_checks(arms)
    return {
        "cell_id": f"{baseline}_{site}",
        "baseline": baseline,
        "site": site,
        "n_tasks": n,
        "n_no_success": n_nosucc,
        "modes_present": modes_present,
        "empirical_cheapest_single_mode": empirical_cheapest,
        "dom_is_empirically_cheapest": empirical_cheapest == dom_mode,
        "cost_spread_pct": round(cost_spread_pct, 2),
        "mean_cost_per_mode_usd": {m: round(v, 6) for m, v in mean_cost.items()},
        "oracle_label_distribution": _mode_dist([l for l in oracle_labels if l is not None]),
        "arms": {name: a.as_row() for name, a in arms.items()},
        "bounding_checks": bounds,
    }


def _bounding_checks(arms: dict[str, ArmResult]) -> dict[str, Any]:
    """Verify the ladder brackets the learned-router proxy as expected.

    Two kinds of result:
    - STRUCTURAL invariant (`sr_ceiling_holds`): the per-task-lookup oracle SR is
      >= every other arm's SR. This is mathematically guaranteed — the oracle
      succeeds on every task on which ANY mode succeeds, so any routing policy's
      successes are a subset of the oracle's. If it is False, there is a code bug.
    - INFORMATIVE signals (the R5 story): does the router proxy beat the
      single-feature stump and the no-text ablation, and how much generalisation
      headroom remains to the memorisation ceiling. These are findings, not
      guarantees — on thin / skewed archive cells the floor arms can tie the proxy.

    Note on cost: `always_cheapest_dom` is the §6.5 NAMED cost-only baseline, not a
    universal cost lower bound — a per-task policy (e.g. the oracle) can be cheaper
    than the cheapest single mode by exploiting per-task cost variation. So we do
    NOT assert "always-DOM is cheaper than every arm"; we disclose the per-mode cost
    spread at the cell level instead (see cost_spread_pct / mean_cost_per_mode_usd).
    """
    proxy = arms["learned_router_proxy"]
    floor = arms["always_cheapest_dom"]
    ceil = arms["per_task_lookup"]
    stump = arms["decision_stump"]
    dom_only = arms["lr_dom_features_only"]

    sr_ceiling_holds = ceil.sr_mean >= max(
        a.sr_mean for a in arms.values() if a.arm != "per_task_lookup"
    ) - 1e-12

    checks = {
        # STRUCTURAL (must hold)
        "sr_ceiling_holds": bool(sr_ceiling_holds),
        # INFORMATIVE (R5 story)
        "proxy_beats_stump": bool(proxy.sr_mean >= stump.sr_mean - 1e-12),
        "proxy_beats_dom_only_or_equal": bool(proxy.sr_mean >= dom_only.sr_mean - 1e-12),
        "router_headroom_pp": round(100 * (ceil.sr_mean - proxy.sr_mean), 3),
        "text_feature_value_pp": round(100 * (proxy.sr_mean - dom_only.sr_mean), 3),
        "feature_set_value_over_stump_pp": round(100 * (proxy.sr_mean - stump.sr_mean), 3),
        "router_lift_over_dom_pp": round(100 * (proxy.sr_mean - floor.sr_mean), 3),
        # cost reference points (USD, total-billed)
        "cost_always_dom_usd": round(floor.cost_mean, 6),
        "cost_proxy_usd": round(proxy.cost_mean, 6),
        "cost_oracle_usd": round(ceil.cost_mean, 6),
    }
    checks["structural_invariant_holds"] = bool(sr_ceiling_holds)
    return checks


# ── Disclosure artifact (read by aggregate_h10_pareto.py) ───────────────────────
def build_disclosure(cells_evaluated: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "disclosure_only": True,
        "not_a_gate": (
            "Paper §6.5 intelligent-baseline ladder (B-1006 R5 reviewer defense). These rows "
            "BOUND the learned router from above (per_task_lookup oracle ceiling) and below "
            "(always_cheapest cost floor); they are DISCLOSURE and do NOT enter the H10 "
            "operational deployment gate. Archive/dev-sanity numbers are directional only — "
            "paper §6 main data is the Phase 1a Pass-2 fire."
        ),
        "ladder_arms": {
            "always_cheapest": "cost-axis lower bound (= always-DOM)",
            "decision_stump": "single-feature heuristic floor (depth-1 tree)",
            "per_task_lookup": "infinite-capacity SR ceiling (oracle / task_id one-hot reductio)",
            "lr_dom_features_only": "text-feature ablation (drops TF-IDF on fire / intent-regex on archive)",
            "learned_router_proxy": "8-dim LR proxy for the router (archive sanity only)",
        },
        "cells": cells_evaluated,
    }


def write_disclosure(payload: dict[str, Any]) -> None:
    ROUTER_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DISCLOSURE_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote disclosure artifact: {DISCLOSURE_PATH}")


# ── Markdown dev-sanity report ──────────────────────────────────────────────────
def render_report_md(payload: dict[str, Any]) -> str:
    md = [
        "# Intelligent-baseline ladder — archive dev-sanity (paper §6.5, B-1006 R5)",
        "",
        "> ⚠️ **SANITY-CHECK ONLY / DISCLOSURE, NOT GATING.** Same Option-C caveat as "
        "`l1_archive_simulation.py`: archive outcomes are pre-fire / directional and cannot "
        "enter the paper as SR claims. Paper §6 main data = Phase 1a Pass-2 fire. These four "
        "baselines bound the learned router; they do NOT feed the H10 operational deployment gate.",
        "",
        f"Generated: `{payload['generated_utc']}` · schema `{payload['schema_version']}`",
        "",
        "## Ladder definition",
        "",
        "| Arm | Role | Bounds the router... |",
        "|---|---|---|",
        "| `always_cheapest_dom` (= always-DOM) | cost-only baseline | the cost-axis reference point |",
        "| `decision_stump` (depth-1 tree) | feature floor | from below on feature-set value |",
        "| `per_task_lookup` (oracle) | SR upper bound | from above (infinite-capacity ceiling) |",
        "| `lr_dom_features_only` | text ablation | from the text-feature side |",
        "| `learned_router_proxy` (8-dim LR) | bounded subject | (the archive stand-in being bracketed) |",
        "",
    ]
    for cell_id, c in payload["cells"].items():
        if c.get("status") and c["status"] != "ok":
            md.append(f"## {cell_id}\n\n_({c['status']})_\n")
            continue
        md.append(f"## {cell_id}")
        md.append("")
        cheap_note = (
            "DOM is empirically cheapest"
            if c["dom_is_empirically_cheapest"]
            else f"empirical cheapest = `{c['empirical_cheapest_single_mode']}` (DOM not strictly min)"
        )
        md.append(
            f"- n tasks: **{c['n_tasks']}** ({c['n_no_success']} with no successful mode) · "
            f"modes: `{c['modes_present']}`"
        )
        md.append(
            f"- total-billed cost near-tie: single-mode cost spread **{c['cost_spread_pct']:.1f}%** "
            f"({cheap_note}) — `{c['mean_cost_per_mode_usd']}`"
        )
        md.append(f"- oracle label distribution: `{c['oracle_label_distribution']}`")
        md.append("")
        md.append("| Arm | Role | SR % [95% CI] | Cost USD [95% CI] | Routed modes |")
        md.append("|---|---|---|---|---|")
        # Order: cost baseline, stump, dom_only, proxy, ceiling (ascending capacity).
        order = [
            "always_cheapest_dom", "decision_stump", "lr_dom_features_only",
            "learned_router_proxy", "per_task_lookup",
        ]
        for name in order:
            a = c["arms"][name]
            dist = ", ".join(f"{m}={n}" for m, n in list(a["routed_mode_dist"].items())[:4])
            md.append(
                f"| `{a['arm']}` | {a['role']} | "
                f"{a['sr_mean_pct']:.2f} [{a['sr_ci_95_pct'][0]:.2f}, {a['sr_ci_95_pct'][1]:.2f}] | "
                f"{a['cost_mean_usd']:.5f} [{a['cost_ci_95_usd'][0]:.5f}, {a['cost_ci_95_usd'][1]:.5f}] | "
                f"{dist} |"
            )
        md.append("")
        b = c["bounding_checks"]
        md.append("**Ladder bounding** (R5 defense):")
        md.append("")
        md.append(f"- SR ceiling holds (oracle ≥ every arm) — STRUCTURAL invariant: **{b['sr_ceiling_holds']}**")
        md.append(f"- router proxy beats single-feature stump: **{b['proxy_beats_stump']}** ({b['feature_set_value_over_stump_pp']:+.2f} pp)")
        md.append(f"- router proxy beats no-text ablation: **{b['proxy_beats_dom_only_or_equal']}** ({b['text_feature_value_pp']:+.2f} pp)")
        md.append(f"- generalisation headroom to memorisation ceiling (oracle − proxy): **{b['router_headroom_pp']:+.2f} pp**")
        md.append(f"- router lift over always-DOM (proxy − always-DOM): **{b['router_lift_over_dom_pp']:+.2f} pp**")
        md.append(
            f"- cost reference points (total-billed USD): always-DOM={b['cost_always_dom_usd']:.5f} · "
            f"proxy={b['cost_proxy_usd']:.5f} · oracle={b['cost_oracle_usd']:.5f}"
        )
        md.append("")
        md.append(
            "_Interpretation_: a learned router worth deploying must sit STRICTLY above the "
            "stump / no-text floor (positive feature-set + text-feature value) while leaving the "
            "ceiling gap as its generalisation headroom. On the small skewed archive the floor "
            "arms can tie the proxy — that itself is the honest pre-fire read; the Phase 1a fire "
            "router is the real test."
        )
        md.append("")
    return "\n".join(md) + "\n"


def write_report(payload: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    REPORT_MD.write_text(render_report_md(payload))
    print(f"Wrote report: {REPORT_MD}")
    print(f"Wrote report: {REPORT_JSON}")


# ── Driver ───────────────────────────────────────────────────────────────────────
def run_cell(baseline: str, site: str, cfg: LadderConfig) -> dict[str, Any]:
    run_dirs = find_pass1_run_dirs(baseline, site)
    if not run_dirs:
        return {"cell_id": f"{baseline}_{site}", "status": "no_pass1_runs"}
    outcomes = collect_per_task_outcomes_with_metrics(run_dirs, site)
    if not outcomes:
        return {"cell_id": f"{baseline}_{site}", "status": "no_outcomes"}
    modes_present = {m for t in outcomes for m in outcomes[t]}
    if len(modes_present) < MIN_MODES_FOR_LADDER:
        return {
            "cell_id": f"{baseline}_{site}",
            "status": f"insufficient_modes ({sorted(modes_present)})",
            "n_tasks": len(outcomes),
        }
    feats, fprov = load_cell_features(site, run_dirs, sorted(outcomes.keys()))
    rec = evaluate_ladder_for_cell(baseline, site, outcomes, feats, cfg)
    rec["status"] = "ok"
    rec["feature_provenance"] = fprov
    rec["pass1_run_dirs"] = [d.name for d in run_dirs]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", help="B0 | B1 | B2 (subset; default: all discoverable)")
    ap.add_argument("--site", help="classifieds | reddit (subset)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    args = ap.parse_args()

    cfg = LadderConfig(seed=args.seed, bootstrap_n=args.bootstrap_n)
    if args.baseline and args.site:
        cells = [(args.baseline, args.site)]
    else:
        cells = ALL_CELLS

    evaluated: dict[str, dict[str, Any]] = {}
    for baseline, site in cells:
        print(f"\n=== ladder: {baseline}_{site} ===")
        rec = run_cell(baseline, site, cfg)
        evaluated[rec["cell_id"]] = rec
        if rec.get("status") == "ok":
            b = rec["bounding_checks"]
            print(
                f"  n={rec['n_tasks']} empirical_cheapest={rec['empirical_cheapest_single_mode']} "
                f"cost_spread={rec['cost_spread_pct']:.1f}% | "
                f"sr_ceiling_holds={b['sr_ceiling_holds']} "
                f"headroom={b['router_headroom_pp']:+.2f}pp "
                f"feat_value_vs_stump={b['feature_set_value_over_stump_pp']:+.2f}pp "
                f"text_value={b['text_feature_value_pp']:+.2f}pp"
            )
        else:
            print(f"  {rec.get('status')}")

    payload = build_disclosure(evaluated)
    write_disclosure(payload)
    write_report(payload)

    ok = [c for c in evaluated.values() if c.get("status") == "ok"]
    print(f"\n=== ladder done: {len(ok)}/{len(evaluated)} cells evaluated ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
