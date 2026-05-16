#!/usr/bin/env python3
"""L1 (learned task-prior LR) archive simulation — class-imbalance audit.

SANITY-CHECK ONLY. Same caveat as p1_archive_simulation.py: archive uses same
task IDs as Phase 1a, outcomes pre-§107 + pre-§139.8. Numbers are directional.
Paper-grade L1 claim requires Phase 1a fresh-data 5-fold CV.

Tests 3 class-imbalance handling variants per v5 §2.2:
    A: uniform class weight (control — expect collapse to majority)
    B: balanced class weight (sklearn class_weight='balanced')
    C: binary target {dom_friendly, escalation_needed} + downstream hand rule

Method:
    1. Combine B0 cls (234) + B0 red (210) = 444 tasks
    2. Feature engineering: site one-hot, intent regex (color, search),
       has_ref_image, intent_token_count, axtree_element_count
    3. 5-fold site-stratified CV (preserves cls/red proportion per fold)
    4. Per fold: train LR, predict on test fold, look up archive outcome
    5. Aggregate: mean SR + bootstrap 95% CI; compare vs always_phantom_som

Outputs:
    docs/checkpoints/router/l1_archive_simulation_2026-05-16.{md,json}
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_archive_simulation import load_cell_outcomes, load_task_features, ARCHIVE_RUNS

REPO = Path(__file__).resolve().parents[2]

# Intent regex banks (mechanism-anchored, not archive-calibrated)
COLOR_RE   = re.compile(r"\b(color|red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey)\b", re.I)
SEARCH_RE  = re.compile(r"\b(find|search|locate|how many|how much)\b", re.I)
COMPARE_RE = re.compile(r"\b(cheapest|most expensive|highest|lowest|best|worst|biggest|smallest)\b", re.I)
NAV_RE     = re.compile(r"\b(go to|navigate|open|visit)\b", re.I)


def build_design_matrix(cells_data: dict) -> tuple[np.ndarray, np.ndarray, list[int], list[str], dict]:
    """Combine cls + red into single (X, oracle_label, task_ids, sites) design matrix."""
    rows = []
    oracle_labels = []
    task_ids = []
    sites = []
    matrices = {}  # (site, tid) -> outcome dict
    feature_names = ["site_cls", "has_image", "color_intent", "search_intent",
                     "compare_intent", "nav_intent", "intent_tok_count",
                     "axtree_elements"]
    for site, (matrix, features) in cells_data.items():
        for tid, feat in features.items():
            if tid not in matrix:
                continue
            outcomes = matrix[tid]
            # oracle best mode = first mode with success=True, tie-break by MODES order
            best = None
            for m in ["dom", "som", "vision", "phantom_text", "phantom_prompt", "phantom_som"]:
                if outcomes.get(m, False):
                    best = m; break
            if best is None:
                best = "dom"  # if no mode succeeds, default label = dom
            intent = feat["intent"] or ""
            row = [
                1.0 if site == "classifieds" else 0.0,
                1.0 if feat.get("has_image") else 0.0,
                1.0 if COLOR_RE.search(intent) else 0.0,
                1.0 if SEARCH_RE.search(intent) else 0.0,
                1.0 if COMPARE_RE.search(intent) else 0.0,
                1.0 if NAV_RE.search(intent) else 0.0,
                float(len(intent.split())),
                float(feat["dom_complexity"]),
            ]
            rows.append(row)
            oracle_labels.append(best)
            task_ids.append(tid)
            sites.append(site)
            matrices[(site, tid)] = outcomes
    X = np.array(rows, dtype=float)
    # v6 fix (P1-11, codex pre-fire #4): z-score is now applied PER-FOLD on train data only
    # inside fold_cv_evaluate (sklearn StandardScaler in Pipeline), NOT on full X before split.
    # Previously this line did `(X[:, j] - X.mean()) / X.std()` over full design matrix =
    # train+test info leak into preprocessing. Now z-score columns marked here for downstream
    # Pipeline use; raw values returned.
    return X, np.array(oracle_labels), task_ids, sites, matrices, feature_names


def fold_cv_evaluate(X, y_oracle, task_ids, sites, matrices, class_weight=None,
                     binary_mode=False, n_splits=5, seed=42, n_repeats=1) -> dict:
    """5-fold site-stratified CV. Predict mode → look up archive outcome → SR.

    v7 fix (Q4 user decision 2026-05-16): n_repeats supports repeated stratified k-fold
    (10 repeats × 5-fold = 50 train-test pairs at default) for more robust archive SR
    estimate. Each repeat uses different random_state derived from base seed.
    """
    if binary_mode:
        # y_binary = 1 if oracle_best != dom (escalation needed)
        y = (y_oracle != "dom").astype(int)
    else:
        y = y_oracle
    # Stratify on site to preserve cls/red proportion in folds
    site_int = np.array([0 if s == "classifieds" else 1 for s in sites])
    fold_outcomes = []
    fold_predictions = []
    confusion = {}
    for repeat_idx in range(n_repeats):
        # v7 fix (Q4 user decision 2026-05-16): repeated stratified k-fold for robust
        # archive SR estimate. n_repeats=10 → 50 train-test pairs at default k=5.
        repeat_seed = seed + repeat_idx * 1000
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
        for fold_idx, (tr, te) in enumerate(skf.split(X, site_int)):
            # v6 fix (P1-11): z-score numeric features inside train fold only.
            preprocessor = ColumnTransformer(
                transformers=[
                    ("scale_numeric", StandardScaler(), [6, 7]),  # intent_tok_count, axtree_elements
                ],
                remainder="passthrough",
            )
            lr = Pipeline([
                ("preprocess", preprocessor),
                ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight, solver="lbfgs")),
            ])
            lr.fit(X[tr], y[tr])
            if binary_mode:
                preds = lr.predict(X[te])
                modes = []
                for i, p in zip(te, preds):
                    if p == 0:
                        modes.append("dom")
                    elif X[i, 1] == 1.0:  # has_image
                        modes.append("som")
                    else:
                        modes.append("phantom_som")
            else:
                modes = list(lr.predict(X[te]))
            for i, m in zip(te, modes):
                site = sites[i]; tid = task_ids[i]
                outcomes = matrices[(site, tid)]
                if m in outcomes:
                    ok = int(outcomes[m])
                    fold_outcomes.append((repeat_idx * 10 + fold_idx, site, tid, m, ok))
                    fold_predictions.append(m)
                    confusion[m] = confusion.get(m, 0) + 1
                else:
                    # v6 fix (P1-2): missing-mode prediction = invalid (excluded from SR).
                    confusion[f"invalid_{m}"] = confusion.get(f"invalid_{m}", 0) + 1
                    fold_predictions.append(f"invalid_{m}")
    return {
        "fold_outcomes": fold_outcomes,
        "prediction_dist": confusion,
        "n_total": len(fold_outcomes),
        "n_repeats": n_repeats,
        "n_splits": n_splits,
    }


def aggregate_sr(fold_outcomes, bootstrap_n=1000, seed=42) -> dict:
    """Overall SR + per-site SR + 95% bootstrap CI."""
    rng = np.random.default_rng(seed)
    all_ok = np.array([r[4] for r in fold_outcomes])
    all_site = np.array([r[1] for r in fold_outcomes])
    n = len(all_ok)
    overall_sr = 100 * all_ok.mean() if n else 0
    boot_overall = []
    for _ in range(bootstrap_n):
        idx = rng.integers(0, n, n)
        boot_overall.append(100 * all_ok[idx].mean())
    overall_ci = (float(np.percentile(boot_overall, 2.5)), float(np.percentile(boot_overall, 97.5)))
    per_site = {}
    for site in ["classifieds", "reddit"]:
        mask = all_site == site
        if mask.sum() == 0: continue
        sr = 100 * all_ok[mask].mean()
        boot = []
        for _ in range(bootstrap_n):
            idx = rng.integers(0, mask.sum(), mask.sum())
            boot.append(100 * all_ok[mask][idx].mean())
        per_site[site] = {
            "sr": sr, "n": int(mask.sum()),
            "ci95": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        }
    return {"overall_sr": overall_sr, "overall_ci": overall_ci, "per_site": per_site, "n": n}


def baseline_sr(matrices, mode: str) -> dict:
    """SR of always-route-to-mode baseline, per site."""
    out = {}
    for site in ["classifieds", "reddit"]:
        outcomes = [v[mode] for (s, _), v in matrices.items() if s == site and mode in v]
        if outcomes:
            out[site] = {"sr": 100 * np.mean(outcomes), "n": len(outcomes)}
    return out


def main():
    report = {"run_date": datetime.utcnow().isoformat() + "Z"}

    # Load cells
    cells_data = {}
    for site in ["classifieds", "reddit"]:
        matrix, modes, skipped = load_cell_outcomes("B0", site)
        features = load_task_features("B0", site, list(matrix.keys()))
        cells_data[site] = (matrix, features)
        print(f"Loaded {site}: {len(matrix)} tasks, modes={modes}, skipped={skipped}")

    X, y, task_ids, sites, matrices, feature_names = build_design_matrix(cells_data)
    print(f"Design matrix: X.shape={X.shape}, y label dist={dict(zip(*np.unique(y, return_counts=True)))}")
    report["n_total"] = len(task_ids)
    report["label_distribution"] = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    report["features"] = feature_names

    # v7 (Q4 user decision 2026-05-16): use n_repeats=10 for repeated stratified 5-fold
    # × 10 repeats = 50 train-test pairs per variant. More robust archive SR estimate for
    # development sanity check; paper-grade number source remains Phase 1a LOCO.
    N_REPEATS = 10

    # Variant A: uniform
    print("\n=== Variant A: uniform class weight (repeated 5-fold × 10) ===")
    A = fold_cv_evaluate(X, y, task_ids, sites, matrices, class_weight=None, binary_mode=False, n_repeats=N_REPEATS)
    A_agg = aggregate_sr(A["fold_outcomes"])
    print(f"  Predictions: {A['prediction_dist']}")
    print(f"  Overall SR: {A_agg['overall_sr']:.2f}% [{A_agg['overall_ci'][0]:.2f}, {A_agg['overall_ci'][1]:.2f}]")
    for s, info in A_agg["per_site"].items():
        print(f"  {s}: {info['sr']:.2f}% [{info['ci95'][0]:.2f}, {info['ci95'][1]:.2f}] (n={info['n']})")

    # Variant B: balanced
    print("\n=== Variant B: balanced class weight (repeated 5-fold × 10) ===")
    B = fold_cv_evaluate(X, y, task_ids, sites, matrices, class_weight="balanced", binary_mode=False, n_repeats=N_REPEATS)
    B_agg = aggregate_sr(B["fold_outcomes"])
    print(f"  Predictions: {B['prediction_dist']}")
    print(f"  Overall SR: {B_agg['overall_sr']:.2f}% [{B_agg['overall_ci'][0]:.2f}, {B_agg['overall_ci'][1]:.2f}]")
    for s, info in B_agg["per_site"].items():
        print(f"  {s}: {info['sr']:.2f}% [{info['ci95'][0]:.2f}, {info['ci95'][1]:.2f}] (n={info['n']})")

    # Variant C: binary + hand rule
    print("\n=== Variant C: binary target + hand rule (repeated 5-fold × 10) ===")
    C = fold_cv_evaluate(X, y, task_ids, sites, matrices, class_weight="balanced", binary_mode=True, n_repeats=N_REPEATS)
    C_agg = aggregate_sr(C["fold_outcomes"])
    print(f"  Predictions: {C['prediction_dist']}")
    print(f"  Overall SR: {C_agg['overall_sr']:.2f}% [{C_agg['overall_ci'][0]:.2f}, {C_agg['overall_ci'][1]:.2f}]")
    for s, info in C_agg["per_site"].items():
        print(f"  {s}: {info['sr']:.2f}% [{info['ci95'][0]:.2f}, {info['ci95'][1]:.2f}] (n={info['n']})")

    # Baselines
    print("\n=== Baselines ===")
    always_dom = baseline_sr(matrices, "dom")
    always_psom = baseline_sr(matrices, "phantom_som")
    always_som = baseline_sr(matrices, "som")
    for name, b in [("always_dom", always_dom), ("always_som", always_som), ("always_phantom_som", always_psom)]:
        for s, info in b.items():
            print(f"  {name}/{s}: {info['sr']:.2f}% (n={info['n']})")

    report["variants"] = {
        "A_uniform": {"agg": A_agg, "predictions": A["prediction_dist"]},
        "B_balanced": {"agg": B_agg, "predictions": B["prediction_dist"]},
        "C_binary": {"agg": C_agg, "predictions": C["prediction_dist"]},
    }
    report["baselines"] = {
        "always_dom": always_dom,
        "always_som": always_som,
        "always_phantom_som": always_psom,
    }

    # Verdict
    print("\n=== Verdict ===")
    psom_cls = always_psom["classifieds"]["sr"]
    psom_red = always_psom["reddit"]["sr"]
    best_variant = None; best_delta = -999
    for name, agg in [("A", A_agg), ("B", B_agg), ("C", C_agg)]:
        delta_cls = agg["per_site"]["classifieds"]["sr"] - psom_cls
        delta_red = agg["per_site"]["reddit"]["sr"] - psom_red
        print(f"  Variant {name}: Δcls={delta_cls:+.2f}pp / Δred={delta_red:+.2f}pp vs always_phantom_som")
        if min(delta_cls, delta_red) > best_delta:
            best_delta = min(delta_cls, delta_red); best_variant = name
    report["verdict"] = {"best_variant": best_variant, "best_min_delta_pp": best_delta}
    print(f"  Best variant: {best_variant} (min Δ across cells: {best_delta:+.2f}pp)")

    # Write outputs
    out_json = REPO / "docs/checkpoints/router/l1_archive_simulation_2026-05-16.json"
    out_md = REPO / "docs/checkpoints/router/l1_archive_simulation_2026-05-16.md"
    out_json.write_text(json.dumps(report, indent=2, default=str))

    md = [
        "# L1 (learned task-prior LR) archive simulation — SANITY-CHECK ONLY",
        "",
        "> ⚠️ NOT preregistration lock substrate. Same Option C caveats as `p1_archive_simulation_findings_2026-05-16.md` + `archive_diagnostic_2026-05-16.md`.",
        "",
        f"Run date: `{report['run_date']}`",
        f"Total tasks: {report['n_total']} (cls + red, B0 only)",
        "",
        "## Oracle label distribution",
        "",
        f"```\n{report['label_distribution']}\n```",
        "",
        "## Variant comparison (5-fold site-stratified CV)",
        "",
        "| Variant | Method | Overall SR | cls SR [95% CI] | red SR [95% CI] | vs always_phantom_som |",
        "|---|---|---:|---|---|---|",
    ]
    for name, label in [("A_uniform", "A: uniform LR (control)"),
                         ("B_balanced", "B: balanced LR"),
                         ("C_binary", "C: binary + hand rule")]:
        v = report["variants"][name]["agg"]
        cls = v["per_site"]["classifieds"]
        red = v["per_site"]["reddit"]
        d_cls = cls["sr"] - psom_cls
        d_red = red["sr"] - psom_red
        md.append(f"| {label} | LR | {v['overall_sr']:.2f}% [{v['overall_ci'][0]:.2f}, {v['overall_ci'][1]:.2f}] | "
                  f"{cls['sr']:.2f}% [{cls['ci95'][0]:.2f}, {cls['ci95'][1]:.2f}] | "
                  f"{red['sr']:.2f}% [{red['ci95'][0]:.2f}, {red['ci95'][1]:.2f}] | "
                  f"cls {d_cls:+.2f} / red {d_red:+.2f} |")

    md.append("")
    md.append("## Baselines (no router)")
    md.append("")
    md.append("| Baseline | cls SR | red SR |")
    md.append("|---|---:|---:|")
    for name, b in [("always_dom", always_dom), ("always_som", always_som), ("always_phantom_som", always_psom)]:
        cls_sr = b.get("classifieds", {}).get("sr", float("nan"))
        red_sr = b.get("reddit", {}).get("sr", float("nan"))
        md.append(f"| {name} | {cls_sr:.2f}% | {red_sr:.2f}% |")

    md.append("")
    md.append("## Prediction distribution (mode distribution from CV predictions)")
    md.append("")
    md.append("| Variant | Distribution |")
    md.append("|---|---|")
    for name in ["A_uniform", "B_balanced", "C_binary"]:
        preds = report["variants"][name]["predictions"]
        total = sum(preds.values())
        dist = ", ".join(f"{m}={c}({100*c/total:.1f}%)" for m, c in sorted(preds.items(), key=lambda x: -x[1]))
        md.append(f"| {name} | {dist} |")

    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(f"**Best variant**: {best_variant} (min Δ across cells: {best_delta:+.2f}pp vs always_phantom_som)")
    md.append("")
    if best_delta >= 1.0:
        md.append(f"✅ **L1 viable** — Variant {best_variant} beats `always_phantom_som` by ≥ 1pp on the harder cell. proposals_v5 §2.4 path 1 (2-layer compose paper-grade).")
    elif best_delta >= -0.5:
        md.append(f"🟡 **L1 partial** — Variant {best_variant} roughly matches `always_phantom_som`. proposals_v5 §2.4 path 2 (L1 site-conditional, L2 universal).")
    else:
        md.append(f"❌ **L1 not viable on archive** — all variants underperform `always_phantom_som` by > 0.5pp. proposals_v5 §2.4 path 3 (drop L1; paper §6 = L2 verbose reactive on phantom_som default).")
    md.append("")
    md.append("Note: archive ≠ Phase 1a. L1 may behave differently on fresh post-fix data + with B1/B2 capability tier feature variance.")

    out_md.write_text("\n".join(md))
    print(f"\nWrote: {out_md}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
