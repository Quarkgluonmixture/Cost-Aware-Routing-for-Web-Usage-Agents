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


# Tested layers (vs L35 baseline). L0 is embedding output (often near-target);
# L35 is final post-block (output should == target by construction).
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

    rows = []
    raw_pvals = []
    for L in TEST_LAYERS:
        layer_vals = grid[:, L]
        diff = layer_vals - baseline
        # Paired t-test (use scipy's built-in alternative= for clarity)
        t_stat, t_p_one = stats.ttest_rel(layer_vals, baseline, alternative=alt_direction)

        # Wilcoxon signed-rank (non-parametric backup)
        try:
            w_stat, w_p_one = stats.wilcoxon(layer_vals, baseline,
                                             zero_method="wilcox", alternative=alt_direction)
        except ValueError:
            w_stat, w_p_one = float("nan"), 1.0

        rows.append({
            "layer": L,
            "mean_layer": float(layer_vals.mean()),
            "mean_baseline": float(baseline.mean()),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)) if n_tasks > 1 else 0.0,
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
        out.append("| Layer | mean(L) | mean(L35) | Δ mean | Δ std | t-stat | p (raw) | p (Holm) | reject H0 |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for r in direction_results["rows"]:
            out.append(
                f"| L{r['layer']:>2} | {r['mean_layer']:.3f} | {r['mean_baseline']:.3f} | "
                f"{r['mean_diff']:+.3f} | {r['std_diff']:.3f} | {r['t_stat']:+.2f} | "
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
    p.add_argument("--output", default=None,
                   help="Output markdown path. Default: results/mechanistic/layer_significance_<date>.md")
    args = p.parse_args()

    fwd_cfg, fwd_per_task = load_per_task_results(Path(args.fwd_results))
    rev_cfg, rev_per_task = load_per_task_results(Path(args.rev_results))

    print(f"Forward: N={len(fwd_per_task)} tasks (config tier deduced from --reverse=False, default 'strong')")
    print(f"Reverse: N={len(rev_per_task)} tasks (config tier deduced from --reverse=True, default 'reverse')")

    fwd_overlap = per_direction_layer_test(
        "Forward (som→phantom_som) — overlap_to_target",
        fwd_per_task, metric="token_overlap_to_target",
    )
    fwd_ld = per_direction_layer_test(
        "Forward (som→phantom_som) — Levenshtein dist to target",
        fwd_per_task, metric="ld_to_target",
    )
    rev_overlap = per_direction_layer_test(
        "Reverse (phantom_som→som) — overlap_to_target",
        rev_per_task, metric="token_overlap_to_target",
    )
    rev_ld = per_direction_layer_test(
        "Reverse (phantom_som→som) — Levenshtein dist to target",
        rev_per_task, metric="ld_to_target",
    )
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
