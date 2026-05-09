#!/usr/bin/env python3
"""Stage 2B/2C layer-resolved significance test.

Question: Is the apparent mid-layer (L11-L17) disruption in
patching_continuation_results.json statistically significant, or could
it be sampling noise given task heterogeneity?

Procedure:
1. Load per-task per-layer results from forward (2B) and reverse (2C) runs.
2. For each layer L_n in [0, 5, 11, 17, 23, 29], paired t-test (and Wilcoxon
   signed-rank as non-parametric backup) of:
     H0: overlap_to_target(L_n) >= overlap_to_target(L_35)  (no disruption)
     H1: overlap_to_target(L_n) <  overlap_to_target(L_35)  (one-sided, disruption)
   per-task paired difference (L_n - L_35).
3. Holm-Bonferroni correction across the 6 tested layers (per direction).
4. Cross-direction comparison: Welch's t-test on (L17 - L35) per-task difference
   between forward and reverse, to test if disruption magnitude differs.

Output: markdown summary written to results/mechanistic/layer_significance_<date>.md
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats


# F12 audit fix 2026-05-09: L0-L35 are **transformer block outputs**, NOT
# embedding output. The patching hooks (`activation_patching.py`
# `register_forward_hook` on each `Qwen3VLTextDecoderLayer`) capture the
# output of block i, so:
#   L0  = output of block 0 (first decoder layer post-norm). Often
#         near-target because only one block has transformed the
#         embedding; NOT the embedding itself.
#   L35 = output of block 35 (final block). Output ≈ target by
#         construction since only final_norm + lm_head remain to produce
#         target tokens.
# If you need true embedding output (n_layers+1 indexing), add a hook to
# `model.model.language_model.embed_tokens` instead.
TEST_LAYERS = [0, 5, 11, 17, 23, 29]
BASELINE_LAYER = 35


def load_per_task_results(json_path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data["config"], data["per_task"]


def extract_metric_grid(per_task: list[dict], metric: str) -> np.ndarray:
    """Return shape (n_tasks, n_layers) ndarray for a given metric.
    metric ∈ {token_overlap_to_source, token_overlap_to_target, ld_to_source, ld_to_target}.
    """
    rows = []
    for t in per_task:
        rows.append([pl[metric] for pl in t["per_layer"]])
    return np.asarray(rows, dtype=float)


def holm_correct(pvals: list[float]) -> list[tuple[float, bool]]:
    """Holm-Bonferroni step-down. Input p-values, output (adj_p, reject@0.05)."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    reject = [False] * n
    prev_adj = 0.0
    for rank, i in enumerate(order):
        adj[i] = min(1.0, max(prev_adj, pvals[i] * (n - rank)))
        prev_adj = adj[i]
        reject[i] = adj[i] < 0.05
    # Once a rejection is rejected, all weaker p-values cannot reject either
    rejected_so_far = True
    for i in order:
        if not reject[i]:
            rejected_so_far = False
        if not rejected_so_far:
            reject[i] = False
    return [(adj[i], reject[i]) for i in range(n)]


def per_direction_layer_test(
    label: str, per_task: list[dict], metric: str = "token_overlap_to_target"
) -> dict:
    """Paired t-test + Wilcoxon for each test-layer vs L35 baseline.

    Direction convention: H1 = layer disrupts vs baseline. For overlap metrics
    higher means closer to target → disruption is layer < baseline (alternative='less').
    For Levenshtein-distance metrics, lower means closer to target → disruption is
    layer > baseline (alternative='greater'). We auto-detect from metric name.

    Returns dict with per-layer rows + Holm-adjusted p-values.
    """
    grid = extract_metric_grid(per_task, metric)  # (N, 36)
    n_tasks = grid.shape[0]
    baseline = grid[:, BASELINE_LAYER]

    # Direction: 'less' for overlap-style, 'greater' for ld-style metrics
    is_distance_metric = metric.startswith("ld_")
    alt_direction = "greater" if is_distance_metric else "less"

    rng = np.random.default_rng(seed=42)
    n_boot = 10000

    rows = []
    raw_pvals = []
    for L in TEST_LAYERS:
        layer_vals = grid[:, L]
        diff = layer_vals - baseline

        # C9 fix: handle constant-column edge case (e.g. cell D L0 has all 1.0
        # → 0 variance → NaN t-stat). Skip test, mark as null.
        if np.std(diff, ddof=1) < 1e-12 if n_tasks > 1 else True:
            t_stat, t_p_one = float("nan"), 1.0
            w_stat, w_p_one = float("nan"), 1.0
            ci_lo, ci_hi = float(diff.mean()), float(diff.mean())
        else:
            # Paired t-test (use scipy's built-in alternative= for clarity)
            t_stat, t_p_one = stats.ttest_rel(layer_vals, baseline, alternative=alt_direction)

            # Wilcoxon signed-rank (non-parametric backup)
            try:
                w_stat, w_p_one = stats.wilcoxon(layer_vals, baseline,
                                                 zero_method="wilcox", alternative=alt_direction)
            except ValueError:
                w_stat, w_p_one = float("nan"), 1.0

            # C3 fix: bootstrap percentile 95% CI on mean diff (resample tasks)
            boot_means = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n_tasks, size=n_tasks)
                boot_means[b] = diff[idx].mean()
            ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

        rows.append({
            "layer": L,
            "mean_layer": float(layer_vals.mean()),
            "mean_baseline": float(baseline.mean()),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)) if n_tasks > 1 else 0.0,
            "ci_lo_95": ci_lo,
            "ci_hi_95": ci_hi,
            "t_stat": float(t_stat),
            "t_p_one_sided": float(t_p_one),
            "wilcoxon_p_one_sided": float(w_p_one),
        })
        raw_pvals.append(t_p_one)

    holm = holm_correct(raw_pvals)
    for row, (adj, rej) in zip(rows, holm):
        row["t_p_holm_adj"] = adj
        row["holm_reject_h0"] = rej

    return {
        "label": label,
        "metric": metric,
        "n_tasks": n_tasks,
        "baseline_layer": BASELINE_LAYER,
        "rows": rows,
    }


def cross_direction_test(
    fwd_per_task: list[dict],
    rev_per_task: list[dict],
    layer: int = 17,
    metric: str = "token_overlap_to_target",
) -> dict:
    """Welch's t-test: are forward and reverse disruption magnitudes equal at layer L?

    Test H0: mean(fwd L_n - L_35) == mean(rev L_n - L_35).
    """
    fwd_grid = extract_metric_grid(fwd_per_task, metric)
    rev_grid = extract_metric_grid(rev_per_task, metric)
    fwd_diff = fwd_grid[:, layer] - fwd_grid[:, BASELINE_LAYER]
    rev_diff = rev_grid[:, layer] - rev_grid[:, BASELINE_LAYER]

    t_stat, t_p_two = stats.ttest_ind(fwd_diff, rev_diff, equal_var=False)

    return {
        "layer": layer,
        "metric": metric,
        "fwd_n": int(fwd_grid.shape[0]),
        "rev_n": int(rev_grid.shape[0]),
        "fwd_mean_diff": float(fwd_diff.mean()),
        "fwd_std_diff": float(fwd_diff.std(ddof=1)),
        "rev_mean_diff": float(rev_diff.mean()),
        "rev_std_diff": float(rev_diff.std(ddof=1)),
        "welch_t": float(t_stat),
        "welch_p_two_sided": float(t_p_two),
    }


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001 ***"
    if p < 0.01:
        return f"{p:.3f} **"
    if p < 0.05:
        return f"{p:.3f} *"
    return f"{p:.3f}"


def render_markdown(
    fwd_overlap: dict, fwd_ld: dict,
    rev_overlap: dict, rev_ld: dict,
    cross_overlap_l17: dict, cross_ld_l17: dict,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    out = []
    out.append(f"# Stage 2 Layer-Resolved Significance Test ({today})")
    out.append("")
    out.append("Tests whether mid-layer disruption (L11-L17) in continuation patching")
    out.append("is statistically significant or sampling noise. Procedure: per-task")
    out.append("paired t-test of layer L_n vs L35 baseline, one-sided H1: disruption.")
    out.append("Holm-Bonferroni correction across 6 tested layers per direction.")
    out.append("")
    out.append("Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 (Holm-adjusted).")
    out.append("")

    for direction_results in [fwd_overlap, rev_overlap, fwd_ld, rev_ld]:
        out.append(f"## {direction_results['label']} — metric: `{direction_results['metric']}`")
        out.append(f"N tasks: {direction_results['n_tasks']}, baseline layer: L{direction_results['baseline_layer']}")
        out.append("")
        out.append("| Layer | mean(L) | mean(L35) | Δ mean | Δ std | 95% CI (boot) | t-stat | p (raw) | p (Holm) | reject H0 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in direction_results["rows"]:
            ci_str = f"[{r.get('ci_lo_95', 0.0):+.3f}, {r.get('ci_hi_95', 0.0):+.3f}]"
            out.append(
                f"| L{r['layer']:>2} | {r['mean_layer']:.3f} | {r['mean_baseline']:.3f} | "
                f"{r['mean_diff']:+.3f} | {r['std_diff']:.3f} | {ci_str} | {r['t_stat']:+.2f} | "
                f"{fmt_p(r['t_p_one_sided'])} | {fmt_p(r['t_p_holm_adj'])} | "
                f"{'✓ Yes' if r['holm_reject_h0'] else '✗ No'} |"
            )
        out.append("")

    out.append("## Cross-direction comparison @ L17 (Welch's t-test)")
    out.append("")
    out.append("Are forward and reverse disruption magnitudes statistically distinguishable?")
    out.append("")
    for cross in [cross_overlap_l17, cross_ld_l17]:
        out.append(f"**Metric**: `{cross['metric']}` @ L{cross['layer']}")
        out.append(f"- Forward (N={cross['fwd_n']}): Δ = {cross['fwd_mean_diff']:+.3f} ± {cross['fwd_std_diff']:.3f}")
        out.append(f"- Reverse (N={cross['rev_n']}): Δ = {cross['rev_mean_diff']:+.3f} ± {cross['rev_std_diff']:.3f}")
        out.append(f"- Welch's t: {cross['welch_t']:+.2f}, p (two-sided) = {fmt_p(cross['welch_p_two_sided'])}")
        out.append("")

    out.append("## Interpretation")
    out.append("")
    out.append("- **Holm-rejected layers** in forward direction = layers where mid-layer disruption survives multiple-comparison correction.")
    out.append("- **Cross-direction p**: small p indicates forward and reverse magnitudes differ; large p means they're indistinguishable (consistent with bidirectional mechanism).")
    out.append("")
    out.append("**Caveats**:")
    out.append("- Forward (24 task) and reverse (15 task) ran on DIFFERENT task subsets")
    out.append("  (curated by directional composite score). Selection-bias artifact not")
    out.append("  separable from genuine bidirectional mechanism without 2x2 control")
    out.append("  (see `qsub_2x2_*_myriad.sh` cells C/D).")
    out.append("- Pattern-based mirage curation (curate_mirage_tasks.py) may correlate")
    out.append("  with patching effect size — strong-tier may be effectively the high-")
    out.append("  effect-size subset of full task population.")
    out.append("")

    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fwd-results", default="results/mechanistic/stage2b_curated_b1_cls_myriad/patching_continuation_results.json")
    p.add_argument("--rev-results", default="results/mechanistic/stage2c_reverse_curated_b1_cls_myriad/patching_continuation_results.json")
    p.add_argument("--cellc-results", default="results/mechanistic/stage2b_2x2_fwd_revtasks_myriad/patching_continuation_results.json",
                   help="Cell C: forward direction × reverse-tier 15 tasks (2x2 control)")
    p.add_argument("--celld-results", default="results/mechanistic/stage2c_2x2_rev_strongtasks_myriad/patching_continuation_results.json",
                   help="Cell D: reverse direction × strong-tier 24 tasks (2x2 control)")
    p.add_argument("--output", default=None,
                   help="Output markdown path. Default: results/mechanistic/layer_significance_<date>.md")
    args = p.parse_args()

    fwd_cfg, fwd_per_task = load_per_task_results(Path(args.fwd_results))
    rev_cfg, rev_per_task = load_per_task_results(Path(args.rev_results))
    cellc_per_task = []
    celld_per_task = []
    if Path(args.cellc_results).exists():
        _, cellc_per_task = load_per_task_results(Path(args.cellc_results))
    if Path(args.celld_results).exists():
        _, celld_per_task = load_per_task_results(Path(args.celld_results))

    print(f"Cell A (fwd × strong):    N={len(fwd_per_task)} tasks")
    print(f"Cell B (rev × reverse):   N={len(rev_per_task)} tasks")
    print(f"Cell C (fwd × reverse):   N={len(cellc_per_task)} tasks")
    print(f"Cell D (rev × strong):    N={len(celld_per_task)} tasks")

    cells = [
        ("Cell A: forward × strong-tier (24)", fwd_per_task),
        ("Cell B: reverse × reverse-tier (15)", rev_per_task),
        ("Cell C: forward × reverse-tier (15)", cellc_per_task),
        ("Cell D: reverse × strong-tier (24)", celld_per_task),
    ]
    cell_results_overlap = []
    cell_results_ld = []
    for label, ptasks in cells:
        if not ptasks:
            continue
        cell_results_overlap.append(per_direction_layer_test(
            f"{label} — overlap_to_target", ptasks, metric="token_overlap_to_target",
        ))
        cell_results_ld.append(per_direction_layer_test(
            f"{label} — LD_to_target", ptasks, metric="ld_to_target",
        ))
    # Keep legacy names for renderer
    fwd_overlap = cell_results_overlap[0] if cell_results_overlap else None
    rev_overlap = cell_results_overlap[1] if len(cell_results_overlap) > 1 else None
    fwd_ld = cell_results_ld[0] if cell_results_ld else None
    rev_ld = cell_results_ld[1] if len(cell_results_ld) > 1 else None
    # Note for LD: higher = more disruption (output further from target). The
    # directionality of "less than baseline" inverts. We flip sign of diff
    # internally via the metric name handling — but the test is paired so the
    # interpretation needs care. To keep consistent with overlap interpretation,
    # we manually flip the test direction for LD by negating the metric.

    cross_overlap_l17 = cross_direction_test(
        fwd_per_task, rev_per_task, layer=17,
        metric="token_overlap_to_target",
    )
    cross_ld_l17 = cross_direction_test(
        fwd_per_task, rev_per_task, layer=17,
        metric="ld_to_target",
    )

    md = render_markdown(fwd_overlap, fwd_ld, rev_overlap, rev_ld,
                         cross_overlap_l17, cross_ld_l17)
    # Append all 4-cell tables after the legacy 2-cell layout (back-compat)
    extra = ["\n## All cells (2x2 expanded)\n"]
    for r in cell_results_overlap + cell_results_ld:
        extra.append(f"### {r['label']}")
        extra.append(f"N={r['n_tasks']}, baseline L{r['baseline_layer']}")
        extra.append("")
        extra.append("| Layer | mean(L) | mean(L35) | Δ mean | Δ std | 95% CI (boot) | t-stat | p (raw) | p (Holm) | reject H0 |")
        extra.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in r["rows"]:
            ci_str = f"[{row.get('ci_lo_95', 0.0):+.3f}, {row.get('ci_hi_95', 0.0):+.3f}]"
            extra.append(
                f"| L{row['layer']:>2} | {row['mean_layer']:.3f} | {row['mean_baseline']:.3f} | "
                f"{row['mean_diff']:+.3f} | {row['std_diff']:.3f} | {ci_str} | {row['t_stat']:+.2f} | "
                f"{fmt_p(row['t_p_one_sided'])} | {fmt_p(row['t_p_holm_adj'])} | "
                f"{'✓ Yes' if row['holm_reject_h0'] else '✗ No'} |"
            )
        extra.append("")
    md = md + "\n".join(extra)

    if args.output:
        out_path = Path(args.output)
    else:
        today = datetime.now().strftime("%Y%m%d")
        out_path = Path(f"results/mechanistic/layer_significance_{today}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nWrote: {out_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
