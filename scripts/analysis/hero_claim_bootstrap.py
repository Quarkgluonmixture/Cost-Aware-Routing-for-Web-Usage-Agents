#!/usr/bin/env python3
"""W1 hero-claim bootstrap CI (response to /stress W1 attack).

/stress reviewer 2026-05-12 attacked the paper §1 hero claim:
  "Phantom-SoM matches or modestly exceeds full SoM on reddit
   (13.81% vs 10.48%, N=210); the gap is within 2σ under run-to-run
   variability we observe in same-condition repeats"

The author's own hedging suggests +3.33pp is statistically marginal.
The reviewer demands per-seed bootstrap 95% CI on the pairwise comparison
and on the drop-one oracle, with strict-positive lower bound or downgrade
the prose.

This script loads B0 reddit per-task adjusted_success for all 6 completed
modes (DOM, SoM, Vision, P-SoM, P-text, P-prompt), bootstraps 10000 task
resamples (N=210 with replacement), and reports for each comparison:
  - Point estimate
  - Bootstrap 95% percentile CI
  - P(diff > 0) — strict-positive bootstrap probability
  - P(diff > 1pp) — practical-significance bootstrap probability

Also applies to classifieds for cross-site sanity (expect SoM > P-SoM
on cls = sanity check passes if cls reddit story is calibrated).

Outputs:
  docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_MD = ROOT / "docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md"

# B0 condition directory mapping per site.
SITES = {
    "reddit": {
        "dom":            "B0_3mode_reddit_20260422/phase1_dom_router_0",
        "som":            "B0_3mode_reddit_20260422/phase1_som_router_0",
        "vision":         "B0_3mode_reddit_20260422/phase1_vision_router_0",
        "phantom_som":    "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0",
        "phantom_text":   "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0",
        "phantom_prompt": "B0_phantom_prompt_reddit_20260429/phase1_phantom_prompt_router_0",
    },
    "classifieds": {
        "dom":            "B0_3mode_classifieds_20260413/phase1_dom_router_0",
        "som":            "B0_3mode_classifieds_20260413/phase1_som_router_0",
        "vision":         "B0_3mode_classifieds_20260413/phase1_vision_router_0",
        "phantom_som":    "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0",
        "phantom_text":   "B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0",
    },
}


def load_adjusted_success(episodes_dir: Path) -> dict[int, bool]:
    """Load per-task adjusted_success bool from episodes/*_summary_v2.json files."""
    out = {}
    if not episodes_dir.exists():
        return out
    for p in sorted(episodes_dir.glob("*_summary_v2.json")):
        # files: <site>_task_<int>_summary_v2.json
        m = re.search(r"task_(\d+)", p.name)
        if not m:
            continue
        tid = int(m.group(1))
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        # adjusted_success preferred; fall back to success
        v = rec.get("adjusted_success", rec.get("success", False))
        out[tid] = bool(v)
    return out


def build_success_matrix(site: str) -> tuple[np.ndarray, list[int], list[str]]:
    """Build (N_tasks x N_modes) binary success matrix on the same-task subset."""
    mode_dirs = SITES[site]
    per_mode = {}
    for mode, rel in mode_dirs.items():
        epi_dir = ROOT / "results/visualwebarena/phase1" / rel / "episodes"
        per_mode[mode] = load_adjusted_success(epi_dir)
    # same-task subset: tasks present in ALL modes
    task_sets = [set(d.keys()) for d in per_mode.values()]
    common_tasks = sorted(set.intersection(*task_sets)) if task_sets else []
    modes = list(per_mode.keys())
    n = len(common_tasks)
    M = np.zeros((n, len(modes)), dtype=int)
    for i, t in enumerate(common_tasks):
        for j, m in enumerate(modes):
            M[i, j] = int(per_mode[m].get(t, False))
    return M, common_tasks, modes


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, B: int = 10000, seed: int = 42):
    """Bootstrap 95% percentile CI on (mean(a) - mean(b)) per-task paired."""
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for k in range(B):
        idx = rng.integers(0, n, size=n)
        diffs[k] = 100 * (a[idx].mean() - b[idx].mean())
    return {
        "point": float(100 * (a.mean() - b.mean())),
        "ci_lo": float(np.quantile(diffs, 0.025)),
        "ci_hi": float(np.quantile(diffs, 0.975)),
        "p_gt_0": float((diffs > 0).mean()),
        "p_gt_1pp": float((diffs > 1.0).mean()),
        "median": float(np.median(diffs)),
    }


def bootstrap_drop_one_ci(M: np.ndarray, drop_mode_idx: int, modes: list[str],
                          B: int = 10000, seed: int = 42):
    """Bootstrap drop-one oracle: oracle SR with all modes vs oracle SR without mode i."""
    n = M.shape[0]
    rng = np.random.default_rng(seed)
    drops = np.empty(B)
    other_idx = [j for j in range(M.shape[1]) if j != drop_mode_idx]
    for k in range(B):
        idx = rng.integers(0, n, size=n)
        Msub = M[idx]
        oracle_all = (Msub.sum(axis=1) > 0).mean()
        oracle_without = (Msub[:, other_idx].sum(axis=1) > 0).mean()
        drops[k] = 100 * (oracle_all - oracle_without)
    return {
        "point": float(100 * ((M.sum(axis=1) > 0).mean() -
                              (M[:, other_idx].sum(axis=1) > 0).mean())),
        "ci_lo": float(np.quantile(drops, 0.025)),
        "ci_hi": float(np.quantile(drops, 0.975)),
        "p_gt_0": float((drops > 0).mean()),
        "p_gt_1pp": float((drops > 1.0).mean()),
        "median": float(np.median(drops)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstraps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    args = p.parse_args()

    np.random.seed(args.seed)
    lines = [
        "# Hero-claim bootstrap CI (W1 defuse)",
        "",
        f"Per-seed bootstrap 95% percentile CI on paired adjusted-SR diffs and drop-one oracle. "
        f"B={args.bootstraps}, seed={args.seed}. Tasks resampled with replacement at task level.",
        "",
        "**Defuse target**: /stress W1 attack — paper §1 hero claim 'P-SoM 13.81% > SoM 10.48% reddit' "
        "is statistically marginal under author's own 2σ hedge.",
        "",
    ]

    for site in ["reddit", "classifieds"]:
        try:
            M, tasks, modes = build_success_matrix(site)
        except Exception as e:
            lines.append(f"## {site}: FAILED to load — {e}")
            continue
        n = M.shape[0]
        lines += [
            f"## {site} (N={n} same-task)",
            "",
            "**Per-mode adjusted SR (%)**:",
            "",
        ]
        for j, m in enumerate(modes):
            sr = 100 * M[:, j].mean()
            lines.append(f"- {m}: {sr:.2f}%")
        lines.append("")

        # Key pairwise comparisons
        mi = {m: j for j, m in enumerate(modes)}

        # Define comparisons to run
        comps = []
        if "phantom_som" in mi and "som" in mi:
            comps.append(("P-SoM vs SoM", "phantom_som", "som"))
        if "phantom_som" in mi and "dom" in mi:
            comps.append(("P-SoM vs DOM", "phantom_som", "dom"))
        if "phantom_text" in mi and "dom" in mi:
            comps.append(("P-text vs DOM", "phantom_text", "dom"))
        if "phantom_som" in mi and "phantom_text" in mi:
            comps.append(("P-SoM vs P-text", "phantom_som", "phantom_text"))

        lines += [
            "**Pairwise SR difference, bootstrap 95% CI:**",
            "",
            "| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for label, a, b in comps:
            r = bootstrap_diff_ci(M[:, mi[a]], M[:, mi[b]], B=args.bootstraps, seed=args.seed)
            sign = "✓ strict-pos" if r["ci_lo"] > 0 else ("✗ crosses 0" if r["ci_hi"] > 0 else "✗ strict-neg")
            lines.append(
                f"| {label} | {r['point']:+.2f} | {r['median']:+.2f} | "
                f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_gt_0']:.3f} | {r['p_gt_1pp']:.3f} | "
            )
            lines.append(f"|  | | | {sign} | | |")
        lines.append("")

        # Drop-one oracle on 4-mode set (DOM, SoM, Vision, P-SoM)
        core_modes = ["dom", "som", "vision", "phantom_som"]
        if all(m in mi for m in core_modes):
            sub_idx = [mi[m] for m in core_modes]
            Msub = M[:, sub_idx]
            lines += [
                f"**Drop-one oracle on {len(core_modes)}-mode set ({', '.join(core_modes)}), bootstrap 95% CI:**",
                "",
                "| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for j, m in enumerate(core_modes):
                r = bootstrap_drop_one_ci(Msub, j, core_modes,
                                           B=args.bootstraps, seed=args.seed + j)
                sign = "✓ strict-pos" if r["ci_lo"] > 0 else "✗ crosses 0"
                lines.append(
                    f"| {m} | {r['point']:+.2f} | {r['median']:+.2f} | "
                    f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_gt_0']:.3f} | {r['p_gt_1pp']:.3f} | "
                )
                lines.append(f"|  | | | {sign} | | |")
            lines.append("")

    # Verdict
    lines += [
        "## Verdict on /stress W1",
        "",
        "Read the **reddit P-SoM vs SoM** row + **reddit drop-one P-SoM** row:",
        "",
        "- If both CIs are strict-positive (ci_lo > 0) AND P(diff > 0) > 0.95 → **W1 attack defused**, "
        "  §1 hero claim is bootstrap-supported. Remove the '2σ hedge' from line 5, lead with the magnitude.",
        "- If CIs cross zero but P(diff > 0) > 0.80 → **W1 partially defused**, the claim is directional",
        "  but not strictly statistically significant. §1 hero must downgrade to 'competitive within 2σ' as",
        "  the author already wrote, but the complementarity (Jaccard / drop-one positive on N=7 tasks) carries",
        "  the structural weight.",
        "- If P(diff > 0) < 0.80 → **W1 sustained**, §1 hero claim must rewrite to 'parity / complementarity",
        "  rather than dominance'. The single-mode comparison is unsupported.",
    ]

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"summary → {args.output_md}")


if __name__ == "__main__":
    main()
