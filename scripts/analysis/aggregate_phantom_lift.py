#!/usr/bin/env python3
"""[Outcome 0c + 0d] Outcome dimension — routing oracle lift and task-pool Jaccard.

Outputs:
- results/phantom_paper/phantom_lift.csv
- results/phantom_paper/phantom_lift.md

Outcome 0c: 3-mode to 4/5-mode oracle lift and significance tests.
Outcome 0d: P-text↔P-SoM task-pool Jaccard Scenario C sentinel.

See docs/checkpoints/paper_planning.md §3 Outcome dimension framework.

Aggregate phantom routing lift across (baseline, site) cells.

For each cell with all 5 modes (DOM / SoM / Vision / P-text / P-SoM) present:
  - Compute 3-mode oracle ceiling (DOM ∪ SoM ∪ Vision)
  - Compute 5-mode oracle ceiling (+ P-text + P-SoM)
  - Routing lift = 5-mode - 3-mode oracle SR (pp)
  - 95% bootstrap CI on lift (n=1000 task resamples)
  - Decomposition: P-text-only, P-SoM-only, both-add-same contributions

Outputs (results/phantom_paper/):
  - phantom_lift.csv          (one row per (baseline, site, decomposition))
  - phantom_lift_summary.md   (paper-ready table for Section 1/4 hook)

Usage:
    python3 scripts/analysis/aggregate_phantom_lift.py [--cells <baseline:site:...>]

Default cells = paper-grade clean B0 cls/red. B1 cells included if all 5 modes
have data (>= 50 ep each). Partial cells use the common observed-task universe.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from scripts.analysis.lib.run_registry import get_cells
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.analysis.lib.run_registry import get_cells

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

REPO = Path(__file__).resolve().parents[2]


# Cell registry: (baseline, site, expected_N, run_paths_per_mode)
def _build_cells(grade_filter: list | None = None,
                 manifest_path: 'Path | None' = None) -> list[dict]:
    """Build aggregator cell list. F01 audit: respects grade_filter
    (default = `paper-grade` only). Pass a list to override (e.g. for
    legacy `archived` data in Appendix-D sensitivity figure).

    B-560 (/stress A1.22 P0-6-B* codex carry-leak closure, 2026-05-17):
    `manifest_path` is now threaded through to `get_cells(...)` so the
    cells loaded match the manifest the caller actually specified. A1.21
    P0-4 + A1.5b Phase 2 B-534 plumbed `manifest_path` through
    `aggregate_phase1_full_prereg_decision.py:main()` and `get_cells`,
    but `_build_cells(grade_filter)` here still discarded the arg —
    `get_aggregator_cells(..., manifest_path=Path)` looked correct from
    outside but silently fell back to the default registry inside. The
    JSON provenance hash would point at the user's chosen manifest while
    the pooled H1/H2/H3 statistics came from the default — exactly the
    "artifact self-evidence vs statistical truth" split codex Mode B
    flagged at /stress A1.22 (Finding F1)."""
    # Pull baseline list from the central registry so B2 (and any future
    # baseline) flows through automatically without touching this file.
    from scripts.analysis.lib.run_registry import BASELINES as _BASELINES
    out: list[dict] = []
    for baseline in _BASELINES:
        for site in ("classifieds", "reddit"):
            # B-560: forward manifest_path so the registry sees the same
            # source the caller specified at the CLI/orchestration layer.
            specs = get_cells(
                baseline=baseline, site=site,
                grade=grade_filter, manifest_path=manifest_path,
            )
            if not specs:
                continue
            out.append({
                "baseline": baseline,
                "site": site,
                "n_expected": specs[0].expected_n,
                "modes": {cell.mode: cell.episodes_dir for cell in specs},
            })
    return out


# F01 audit 2026-05-09: env override `P79_AGGREGATOR_GRADE` lets the
# Appendix-D legacy sensitivity figure pull `archived` data while the
# default `paper-grade` filter remains the paper-claim path.
_GRADE_OVERRIDE = os.environ.get("P79_AGGREGATOR_GRADE", "")
_GRADE_LIST = [g.strip() for g in _GRADE_OVERRIDE.split(",") if g.strip()] or None
CELLS = _build_cells(_GRADE_LIST)


def get_aggregator_cells(grade_filter: list | None = None,
                          manifest_path: 'Path | None' = None) -> list[dict]:
    """Live registry view (re-evaluates each call).

    A1.21 P1-3 fix (B-530, Claude P2-1 + codex F8 2-AI overlap): CELLS module-level
    constant frozen at import time. Long-running orchestration (notebooks / cron
    sidecars / multi-aggregator chains) where env var or yaml changes after first
    import don't propagate → silent stale cell selection. Use this fn for lazy
    re-evaluation. Default behavior unchanged for callers using `CELLS` constant.

    Caller migration: replace `from aggregate_phantom_lift import CELLS` with
    `from aggregate_phantom_lift import get_aggregator_cells; cells = get_aggregator_cells()`
    when re-evaluation matters (paper-grade audit reproducibility).

    B-560 (/stress A1.22 P0-6-B* codex carry-leak closure, 2026-05-17):
    `manifest_path` now forwarded into `_build_cells(...)`. Pre-fix this
    function accepted `manifest_path` only to satisfy the call signature
    but dropped it before reaching `get_cells`, producing the
    "provenance theater" failure mode where the output JSON's
    `manifest_sha256` referenced a manifest that never gated discovery.
    """
    if grade_filter is None:
        # Honor env var override at call time (was frozen at module import pre-fix)
        env_override = os.environ.get("P79_AGGREGATOR_GRADE", "")
        grade_filter = [g.strip() for g in env_override.split(",") if g.strip()] or None
    return _build_cells(grade_filter, manifest_path=manifest_path)

MIN_EP_FOR_CELL = 50  # skip cells where any present mode has < 50 ep (too partial)


def load(d: Path) -> tuple[set[int], set[int]]:
    """Returns (succ_set, observed_set).

    B-325 (/stress A1.9 Mode B F3 OOB, 2026-05-16): strict-by-default flipped
    for paper-grade defensibility. Pre-fix lenient default added corrupt
    summary task_ids to `observed` set → treated as "observed failure" in
    drop-one oracle denominator → paper §1 hero "Phantom-SoM +3.33pp reddit
    drop-one oracle lift" silently polluted if any JSONL has corrupt row.
    Now: default strict (raises ValueError on corrupt + excludes task_id from
    BOTH observed and success sets — corrupt rows are missing-data, not
    failures). Legacy lenient inspection mode via `P79_STRICT=0` env override.
    """
    s, o = set(), set()
    if not d.exists():
        return s, o
    from p79.experiment.io_utils import load_episode_summary_strict

    n_corrupt = 0
    # B-325: strict is now the default. P79_STRICT=0 explicitly opts into
    # lenient mode for legacy data inspection (was: strict required opt-in).
    _strict_env = os.environ.get("P79_STRICT", "1").lower()
    _strict_mode = "lenient" if _strict_env in ("0", "false", "no") else "strict"
    for p in sorted(d.glob("*_summary_v2.json")):
        m = re.search(r"task_(\d+)", p.name)
        if not m:
            continue
        tid = int(m.group(1))
        try:
            # B-542 (/stress A1.5b Phase 2 P0-3-B codex OOB, 2026-05-17): paper-
            # grade aggregator MUST reject B-486 quarantined episodes
            # (needs_reevaluation=True). Pre-fix: load_episode_summary_strict
            # accepted quarantined rows (only checked type-safety) → drop-one
            # oracle denominator included crash-before-evaluator episodes as
            # `success=False` failures → paper §1 hero "Phantom-SoM +3.33pp
            # reddit drop-one oracle lift" silently inflated denominator with
            # non-evaluated tasks. Same B-325 corrupt-row hygiene principle.
            rec = load_episode_summary_strict(
                p,
                mode=_strict_mode,
                reject_needs_reevaluation=True,
            )
        except ValueError:
            # B-325: corrupt → exclude from BOTH observed and success.
            # B-542: quarantined episode (needs_reevaluation=True) also rejected
            # in strict mode (raises ValueError) → counted as "missing data",
            # NOT "observed failure". Same exclusion semantics keeps drop-one
            # oracle denominator clean.
            # Pre-fix the `o.add(tid)` ran BEFORE the load attempt → corrupt
            # task counted as observed failure (drop-one denominator pollution).
            n_corrupt += 1
            continue
        if rec is None:
            n_corrupt += 1
            continue
        # Only add to observed AFTER successful load (B-325 strict-by-default).
        o.add(tid)
        # §139.8: adjusted_success retired — `success` is canonical.
        # B-283: strict loader guarantees `rec["success"]` is bool, so `is True` is safe.
        if rec["success"] is True:
            s.add(tid)
    if n_corrupt > 0:
        msg = (
            f"  [B-325] {d}: {n_corrupt} corrupt summary file(s) excluded "
            "from both observed + success sets (was: silently counted as "
            "observed failures → paper §1 oracle lift denominator pollution). "
            "Set P79_STRICT=0 to revert to lenient legacy inspection mode."
        )
        if _strict_mode == "strict":
            # B-325: strict default → raise hard so caller sees corrupt count.
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")
    return s, o


def bootstrap_lift_ci(in_3: np.ndarray, in_5: np.ndarray, B: int = 1000, seed: int = 42
                      ) -> tuple[float, float]:
    """Bootstrap 95% CI on (5-mode oracle SR - 3-mode oracle SR)."""
    n = len(in_3)
    rng = np.random.default_rng(seed)
    lifts = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        lifts[b] = 100 * (int(in_5[idx].sum()) - int(in_3[idx].sum())) / n
    return float(np.quantile(lifts, 0.025)), float(np.quantile(lifts, 0.975))


def stratified_bootstrap_lift_ci(in_3: np.ndarray, in_5: np.ndarray,
                                  strata: np.ndarray, B: int = 1000, seed: int = 42
                                  ) -> tuple[float, float]:
    """Stratified bootstrap 95% CI on (5-mode oracle SR - 3-mode oracle SR).

    Resamples within each stratum, preserving the marginal stratum size in each
    bootstrap replicate. Used for /stress A2.1 P0-5-B* robustness check
    (stratify by task_family / evaluator_type / reference_image_present) to
    address the codex concern that the standard paired bootstrap is a
    task-superpopulation variance estimator under an exchangeability
    assumption VWA task taxonomy may violate. Falls back to the standard
    paired bootstrap when only one stratum is observed.

    Args:
        in_3: per-task pass mask for the 3-mode oracle (one bit per task).
        in_5: per-task pass mask for the 5-mode oracle.
        strata: integer-or-string label per task; tasks sharing a label are
            resampled together. ``len(strata)`` must equal ``len(in_3)``.
        B: number of bootstrap resamples.
        seed: numpy RNG seed (default 42 to match ``bootstrap_lift_ci``).

    Returns:
        ``(ci_lo_2.5, ci_hi_97.5)`` task-resampling 95% CI in percentage points.
    """
    n = len(in_3)
    if len(strata) != n:
        raise ValueError(f"strata length {len(strata)} != task vector length {n}")
    rng = np.random.default_rng(seed)
    unique_strata = np.unique(strata)
    if len(unique_strata) <= 1:
        return bootstrap_lift_ci(in_3, in_5, B=B, seed=seed)
    stratum_indices = {s: np.where(strata == s)[0] for s in unique_strata}
    lifts = np.empty(B)
    for b in range(B):
        idx_pieces = []
        for s in unique_strata:
            sub_idx = stratum_indices[s]
            chosen = rng.choice(sub_idx, len(sub_idx), replace=True)
            idx_pieces.append(chosen)
        idx = np.concatenate(idx_pieces)
        lifts[b] = 100 * (int(in_5[idx].sum()) - int(in_3[idx].sum())) / n
    return float(np.quantile(lifts, 0.025)), float(np.quantile(lifts, 0.975))


def permutation_drop_one_null(per_arm_pass_vectors: dict, drop_arm: str,
                               B_perm: int = 10000, seed: int = 42) -> dict:
    """Fixed-marginal permutation null for drop-one oracle excess.

    For each permutation, independently shuffle each arm's per-task pass
    labels while preserving its marginal pass count. Recompute the drop-one
    oracle (union over all arms minus union over arms except ``drop_arm``)
    each time. Reports the observed drop-one's excess over the 95th
    percentile of the permutation null distribution.

    This addresses /stress A2.1 P0-2-ABC* (Claude+codex+gemini 3-AI overlap)
    — the trivial ``H0: drop_one = 0`` rejected by the paired bootstrap is
    not a falsifiability test for the underlying complementarity claim.
    Under this fixed-marginal null, ``drop_one`` excess is non-trivial only
    when the joint pass-set structure across arms encodes routing-relevant
    complementarity beyond what the per-arm marginal pass counts alone
    would produce by independent random success allocation.

    Args:
        per_arm_pass_vectors: dict mapping arm name (str) to per-task pass
            mask (bool / int / np.ndarray of length n).
        drop_arm: arm to be dropped for the drop-one oracle.
        B_perm: number of permutations (10000 is the prereg-locked default).
        seed: numpy RNG seed.

    Returns:
        dict with observed_drop_one_pp, null_p50/p95/p99, excess_over_null_p95_pp,
        p_value_one_sided (fraction of permutations >= observed),
        B_perm, marginal_counts, n_tasks.
    """
    arms = list(per_arm_pass_vectors.keys())
    if drop_arm not in arms:
        raise ValueError(f"drop_arm {drop_arm!r} not in arms {arms!r}")
    if not arms:
        raise ValueError("per_arm_pass_vectors is empty")
    n = len(next(iter(per_arm_pass_vectors.values())))
    if any(len(per_arm_pass_vectors[a]) != n for a in arms):
        raise ValueError("per_arm_pass_vectors entries have mismatched lengths")
    marginal_counts = {a: int(np.asarray(per_arm_pass_vectors[a]).astype(bool).sum())
                       for a in arms}
    rng = np.random.default_rng(seed)
    null_drop_one = np.empty(B_perm)
    for b in range(B_perm):
        u_all = np.zeros(n, dtype=bool)
        u_without = np.zeros(n, dtype=bool)
        for a in arms:
            k = marginal_counts[a]
            if k == 0:
                continue
            idx = rng.choice(n, k, replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            u_all |= mask
            if a != drop_arm:
                u_without |= mask
        null_drop_one[b] = 100 * (int(u_all.sum()) - int(u_without.sum())) / n

    u_all_obs = np.zeros(n, dtype=bool)
    u_without_obs = np.zeros(n, dtype=bool)
    for a in arms:
        v = np.asarray(per_arm_pass_vectors[a]).astype(bool)
        u_all_obs |= v
        if a != drop_arm:
            u_without_obs |= v
    obs = 100 * (int(u_all_obs.sum()) - int(u_without_obs.sum())) / n
    null_p95 = float(np.percentile(null_drop_one, 95))
    return {
        "observed_drop_one_pp": obs,
        "null_p50": float(np.percentile(null_drop_one, 50)),
        "null_p95": null_p95,
        "null_p99": float(np.percentile(null_drop_one, 99)),
        "excess_over_null_p95_pp": obs - null_p95,
        "p_value_one_sided": float((null_drop_one >= obs).mean()),
        "B_perm": B_perm,
        "marginal_counts": marginal_counts,
        "n_tasks": n,
        "drop_arm": drop_arm,
    }


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions p1, p2 ∈ [0, 1].

    h = 2 * (arcsin(√p1) - arcsin(√p2))

    Interpretation: |h|<0.2 small, 0.2-0.5 medium, 0.5-0.8 large, >0.8 huge.
    Sign indicates direction (p1 > p2 → h > 0).
    """
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def cohen_h_label(h: float) -> str:
    a = abs(h)
    if a < 0.2:
        return "small"
    if a < 0.5:
        return "medium"
    if a < 0.8:
        return "large"
    return "huge"


def wilcoxon_signed_rank(in_a: np.ndarray, in_b: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Wilcoxon signed-rank test on paired binary task outcomes (a vs b).

    For binary outcomes diff ∈ {-1, 0, +1}; scipy drops zero diffs. When set b
    ⊇ set a (e.g. 5-mode oracle ⊇ 3-mode oracle), all non-zero diffs are
    positive (b solves task that a doesn't) → test reduces to one-sided
    binomial (sign test). Returns (statistic, p_two_sided) or (None, None) if
    scipy unavailable / undefined.
    """
    if not HAS_SCIPY:
        return None, None
    diffs = in_b.astype(int) - in_a.astype(int)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return None, 1.0  # no difference, p = 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = sp_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return None, None


def mcnemar_exact_one_sided(in_a: np.ndarray, in_b: np.ndarray) -> Optional[float]:
    """McNemar exact one-sided p-value: H1 = b > a (b adds tasks a misses).

    For monotonic case (b ⊇ a), discordant b-only count = sum(b - a > 0),
    a-only count = 0. Exact binomial: p = 0.5^(b_only).
    """
    if not HAS_SCIPY:
        return None
    a = in_a.astype(int); b = in_b.astype(int)
    a_only = int(((a > b)).sum())
    b_only = int(((b > a)).sum())
    n_disc = a_only + b_only
    if n_disc == 0:
        return 1.0
    # one-sided: H1 = b > a
    return float(sp_stats.binom.cdf(a_only, n_disc, 0.5))


def bootstrap_unique_count_ci(in_a: np.ndarray, in_b: np.ndarray,
                              B: int = 1000, seed: int = 42, ci: float = 0.95
                              ) -> tuple[int, float, float]:
    """Bootstrap CI on |a ∖ b| count: tasks where a solves but b doesn't.

    H3 structural claim test: arm a contributes tasks NOT solved by arm b.
    If lower CI bound > 0, "a has unique non-overlap with b" sig at 1-ci level.

    Used per-cell for:
      P-text ∖ P-SoM unique count (axis 1 structural evidence)
      P-prompt ∖ P-SoM unique count (axis 2 structural evidence)
    """
    n = len(in_a)
    if n == 0 or len(in_b) != n:
        return 0, 0.0, 0.0
    a = in_a.astype(bool)
    b = in_b.astype(bool)
    observed = int((a & ~b).sum())
    rng = np.random.default_rng(seed)
    counts = np.empty(B)
    for r in range(B):
        idx = rng.integers(0, n, size=n)
        counts[r] = int((a[idx] & ~b[idx]).sum())
    alpha = (1 - ci) / 2
    return observed, float(np.quantile(counts, alpha)), float(np.quantile(counts, 1 - alpha))


def bootstrap_tost_equivalence_p(in_a: np.ndarray, in_b: np.ndarray,
                                  delta_pp: float = 1.0, B: int = 1000, seed: int = 42
                                  ) -> Optional[float]:
    """Bootstrap TOST (Two One-Sided Tests) p-value for **equivalence** test.

    H0: |true lift| >= δ          (effect is meaningful in either direction)
    H1: |true lift| < δ           (effect equivalent to zero within margin)

    Two one-sided tests:
      H0_lower:  lift <= -δ  → reject if bootstrap dist mostly above -δ
                              p_lower = P(boot_lift <= -δ); small p_lower
                              ⇒ evidence rejects "effect <= -δ"
      H0_upper:  lift >= +δ  → reject if bootstrap dist mostly below +δ
                              p_upper = P(boot_lift >= +δ); small p_upper
                              ⇒ evidence rejects "effect >= +δ"

    TOST p = max(p_lower, p_upper).
    **If max(p_lower, p_upper) < α, equivalence is ACCEPTED** — both
    one-sided tests reject, so the effect is bounded inside (-δ, +δ).

    F03 audit fix 2026-05-09: δ default = 1.0pp (was 0.5). Matches
    `preregistration.md §4` lock "TOST equivalence margin δ = 1.0pp".

    F04 audit fix 2026-05-09: renamed from `bootstrap_tost_p`; clarified
    docstring (previous wording said "equivalence rejected when max < α"
    which inverts the conclusion). Strong positive lift gives p_upper≈1
    correctly (effect is outside +δ equivalence margin), so equivalence
    is correctly NOT accepted.

    For the **nonzero / one-sided directional** test (the phantom-lift
    hypothesis "lift > 0"), use `bootstrap_one_sided_nonzero_p()` below.
    """
    if len(in_a) != len(in_b):
        return None
    n = len(in_a)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    lifts = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        lifts[b] = 100 * (int(in_b[idx].sum()) - int(in_a[idx].sum())) / n
    p_lower = float(np.mean(lifts <= -delta_pp))
    p_upper = float(np.mean(lifts >= delta_pp))
    return max(p_lower, p_upper)


# F04 audit fix 2026-05-09: alias preserves backward-compat callers; new
# code should use the renamed `bootstrap_tost_equivalence_p()`.
bootstrap_tost_p = bootstrap_tost_equivalence_p


def bootstrap_one_sided_nonzero_p(in_a: np.ndarray, in_b: np.ndarray,
                                   B: int = 1000, seed: int = 42,
                                   alternative: str = "greater"
                                   ) -> Optional[float]:
    """Bootstrap one-sided p-value for the directional phantom-lift claim.

    H0: lift = 0     (no phantom-routing benefit)
    H1: lift > 0     (alternative='greater', default — primary paper claim)
       or lift < 0   (alternative='less')

    p = fraction of bootstrap resamples where lift contradicts H1
        (alternative='greater' → fraction with lift <= 0)
        (alternative='less' → fraction with lift >= 0)

    F04 audit fix 2026-05-09: added as the correct test for the paper's
    phantom-lift > 0 claim; the equivalence-style TOST in
    `bootstrap_tost_equivalence_p()` is for the separate "lift is bounded
    inside ±δ" claim and should NOT be substituted for nonzero detection.
    """
    if len(in_a) != len(in_b) or len(in_a) == 0:
        return None
    n = len(in_a)
    rng = np.random.default_rng(seed)
    lifts = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        lifts[b] = 100 * (int(in_b[idx].sum()) - int(in_a[idx].sum())) / n
    if alternative == "greater":
        return float(np.mean(lifts <= 0.0))
    elif alternative == "less":
        return float(np.mean(lifts >= 0.0))
    else:
        raise ValueError(f"alternative must be 'greater' or 'less', got {alternative}")


def bonferroni_adjust(pvals: list) -> list:
    """Bonferroni: p_adj = min(1, m * p_raw); None entries pass-through."""
    m = sum(1 for p in pvals if p is not None)
    if m == 0:
        return list(pvals)
    return [min(1.0, m * p) if p is not None else None for p in pvals]


def holm_bonferroni_adjust(pvals: list) -> list:
    """Holm-Bonferroni step-down (Holm 1979): less conservative than Bonferroni
    while still controlling family-wise error rate at α.

    Sort non-None p-values ascending; the k-th smallest gets multiplied by
    (m - k + 1) where m = number of non-None tests; running max enforces
    monotonicity.
    """
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    out: list = [None] * len(pvals)
    prev = 0.0
    for k, (i, p) in enumerate(indexed):
        adj = min(1.0, max(prev, p * (m - k)))
        out[i] = adj
        prev = adj
    return out


def bh_fdr_adjust(pvals: list) -> list:
    """Benjamini-Hochberg FDR adjusted q-values (BH 1995).

    Less conservative than FWER methods; controls expected proportion of false
    discoveries among rejections rather than family-wise error rate.
    """
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    out: list = [None] * len(pvals)
    prev = 1.0
    for k in range(m - 1, -1, -1):
        i, p = indexed[k]
        rank = k + 1
        adj = min(prev, p * m / rank)
        out[i] = min(1.0, adj)
        prev = out[i]
    return out


def analyze_cell(cell: dict) -> Optional[dict]:
    """Compute phantom lift for a single (baseline, site) cell.

    Required modes: DOM, SoM, Vision, P-SoM. P-text and P-prompt are optional —
    when absent, dependent oracle/lift columns are emitted as None (CSV blank).

    Returns None if any required mode missing or below MIN_EP_FOR_CELL.
    """
    succ, obs = {}, {}
    for mode, ep_dir in cell["modes"].items():
        s, o = load(ep_dir)
        if len(o) < MIN_EP_FOR_CELL:
            # Skip undersized modes silently; allow rest of cell to still build.
            # P-prompt cell may be partial (e.g. 127/210); keep it if it meets MIN.
            continue
        succ[mode] = s
        obs[mode] = o

    required = ("DOM", "SoM", "Vision", "P-SoM")
    if any(m not in succ for m in required):
        return None
    has_pdom = "P-text" in succ
    has_pprompt = "P-prompt" in succ

    # F07 audit fix 2026-05-09: per-comparison universe — each oracle
    # contrast uses ONLY the arms it compares, not a global intersection
    # across all present modes. Previously a partial P-prompt arm could
    # shrink the 3-vs-5 denominator even though P-prompt is not in that
    # estimand. Universes:
    #   universe_psom_only:    obs(DOM, SoM, Vision, P-SoM)
    #   universe_pdom_only:    obs(DOM, SoM, Vision, P-text)
    #   universe_pprompt_only: obs(DOM, SoM, Vision, P-prompt)
    #   universe_5:            obs(DOM, SoM, Vision, P-text, P-SoM)   ← 3-vs-5 denominator
    #   universe_6:            obs(DOM, SoM, Vision, P-text, P-SoM, P-prompt)
    # `n_common` reported in the CSV = |universe_5| if P-text present,
    # else |universe_psom_only| (closest match to historical semantics).

    def _universe(arms: list) -> set:
        return set.intersection(*[obs[a] for a in arms if a in obs])

    universe_psom_only = _universe(["DOM", "SoM", "Vision", "P-SoM"])
    universe_pdom_only = _universe(["DOM", "SoM", "Vision", "P-text"]) if has_pdom else set()
    universe_pprompt_only = _universe(["DOM", "SoM", "Vision", "P-prompt"]) if has_pprompt else set()
    if has_pdom:
        universe_5 = _universe(["DOM", "SoM", "Vision", "P-text", "P-SoM"])
    else:
        universe_5 = universe_psom_only
    if has_pdom and has_pprompt:
        universe_6 = _universe(["DOM", "SoM", "Vision", "P-text", "P-SoM", "P-prompt"])
    else:
        universe_6 = set()

    common = universe_5 if has_pdom else universe_psom_only
    n = len(common)
    if n < MIN_EP_FOR_CELL:
        return None

    # Restrict each mode's success set to its own comparison's universe
    # at use site (not globally as before).
    def _restrict_set(arms: list) -> tuple[set, dict]:
        u = _universe(arms)
        return u, {a: succ[a] & u for a in arms if a in succ}

    # P-SoM only (3 → 4_psom)
    u_psom, succ_r_psom = _restrict_set(["DOM", "SoM", "Vision", "P-SoM"])
    union_3_psom_only = succ_r_psom["DOM"] | succ_r_psom["SoM"] | succ_r_psom["Vision"]
    union_4_psom = union_3_psom_only | succ_r_psom["P-SoM"]
    sr_3_psom_only = 100 * len(union_3_psom_only) / max(1, len(u_psom))
    sr_4_psom = 100 * len(union_4_psom) / max(1, len(u_psom))
    universe_psom = sorted(u_psom)
    in_3_psom = np.array([t in union_3_psom_only for t in universe_psom], dtype=bool)
    in_4_psom = np.array([t in union_4_psom for t in universe_psom], dtype=bool)

    # CSV-reported sr_3 / union_3 use universe_5 (paper-grade primary
    # denominator when P-text present; same as universe_psom otherwise).
    succ_r = {m: s & common for m, s in succ.items()}
    union_3 = succ_r["DOM"] | succ_r["SoM"] | succ_r["Vision"]
    sr_3 = 100 * len(union_3) / n
    universe = sorted(common)
    # Backward-compat aliases for downstream 5-mode / H3 axis tests
    # which use in_3 indexed against universe_5.
    in_3 = np.array([t in union_3 for t in universe], dtype=bool)

    # Single-P-SoM lift CI (uses P-SoM-specific universe per F07)
    ci_lo_psom, ci_hi_psom = bootstrap_lift_ci(in_3_psom, in_4_psom)
    h_4psom_vs_3 = cohen_h(sr_4_psom / 100, sr_3_psom_only / 100)
    wstat_psom, wp_psom = wilcoxon_signed_rank(in_3_psom, in_4_psom)
    mc_p_psom = mcnemar_exact_one_sided(in_3_psom, in_4_psom)
    tost_p_psom = bootstrap_tost_p(in_3_psom, in_4_psom)

    psom_adds = succ_r["P-SoM"] - union_3

    if has_pdom:
        union_4_pdom = union_3 | succ_r["P-text"]
        union_5 = union_3 | succ_r["P-text"] | succ_r["P-SoM"]
        sr_4_pdom = 100 * len(union_4_pdom) / n
        sr_5 = 100 * len(union_5) / n
        in_4_pdom = np.array([t in union_4_pdom for t in universe], dtype=bool)
        in_5 = np.array([t in union_5 for t in universe], dtype=bool)
        ci_lo, ci_hi = bootstrap_lift_ci(in_3, in_5)
        ci_lo_pdom, ci_hi_pdom = bootstrap_lift_ci(in_3, in_4_pdom)
        h_5_vs_3 = cohen_h(sr_5 / 100, sr_3 / 100)
        h_4pdom_vs_3 = cohen_h(sr_4_pdom / 100, sr_3 / 100)
        wstat_5, wp_5 = wilcoxon_signed_rank(in_3, in_5)
        wstat_pdom, wp_pdom = wilcoxon_signed_rank(in_3, in_4_pdom)
        mc_p_5 = mcnemar_exact_one_sided(in_3, in_5)
        mc_p_pdom = mcnemar_exact_one_sided(in_3, in_4_pdom)
        tost_p_5 = bootstrap_tost_p(in_3, in_5)
        tost_p_pdom = bootstrap_tost_p(in_3, in_4_pdom)
        pdom_adds = succ_r["P-text"] - union_3
        both_add = pdom_adds & psom_adds
        pdom_only = pdom_adds - psom_adds
        psom_only = psom_adds - pdom_adds
        inter = succ_r["P-SoM"] & succ_r["P-text"]
        unionj = succ_r["P-SoM"] | succ_r["P-text"]
        jaccard = (len(inter) / len(unionj)) if unionj else 0.0
        jaccard_warn = jaccard > 0.7
    else:
        sr_4_pdom = None
        sr_5 = None
        ci_lo = ci_hi = None
        ci_lo_pdom = ci_hi_pdom = None
        h_5_vs_3 = None
        h_4pdom_vs_3 = None
        wp_5 = wp_pdom = None
        mc_p_5 = mc_p_pdom = None
        tost_p_5 = tost_p_pdom = None
        pdom_adds = both_add = pdom_only = set()
        psom_only = psom_adds  # no overlap with absent P-text
        jaccard = None
        jaccard_warn = False

    # P-prompt 4-mode lift + 6-mode oracle (when present)
    if has_pprompt:
        # F07 audit fix 2026-05-09: P-prompt-only comparison uses
        # universe_pprompt_only (DOM ∩ SoM ∩ Vision ∩ P-prompt), NOT the
        # 5-mode universe — otherwise the denominator drops by tasks
        # missing in P-text/P-SoM that have nothing to do with this arm.
        u_pprompt, succ_r_pprompt = _restrict_set(["DOM", "SoM", "Vision", "P-prompt"])
        union_3_pprompt_only = succ_r_pprompt["DOM"] | succ_r_pprompt["SoM"] | succ_r_pprompt["Vision"]
        union_4_pprompt = union_3_pprompt_only | succ_r_pprompt["P-prompt"]
        sr_3_pprompt_only = 100 * len(union_3_pprompt_only) / max(1, len(u_pprompt))
        sr_4_pprompt = 100 * len(union_4_pprompt) / max(1, len(u_pprompt))
        u_pprompt_sorted = sorted(u_pprompt)
        in_3_pprompt = np.array([t in union_3_pprompt_only for t in u_pprompt_sorted], dtype=bool)
        in_4_pprompt = np.array([t in union_4_pprompt for t in u_pprompt_sorted], dtype=bool)
        ci_lo_pprompt, ci_hi_pprompt = bootstrap_lift_ci(in_3_pprompt, in_4_pprompt)
        h_4pprompt_vs_3 = cohen_h(sr_4_pprompt / 100, sr_3_pprompt_only / 100)
        wstat_pprompt, wp_pprompt = wilcoxon_signed_rank(in_3_pprompt, in_4_pprompt)
        mc_p_pprompt = mcnemar_exact_one_sided(in_3_pprompt, in_4_pprompt)
        tost_p_pprompt = bootstrap_tost_p(in_3_pprompt, in_4_pprompt)
        pprompt_adds = succ_r["P-prompt"] - union_3
        if has_pdom:
            # F07 audit fix 2026-05-09: 6-mode oracle and 6-vs-5
            # incremental tests must use universe_6 (DOM ∩ SoM ∩
            # Vision ∩ P-text ∩ P-SoM ∩ P-prompt). Previously used
            # universe_5 which can include tasks where P-prompt was
            # not observed → treats missing as failed.
            u6_sorted = sorted(universe_6)
            succ_r_u6 = {m: s & universe_6 for m, s in succ.items()}
            union_3_u6 = succ_r_u6["DOM"] | succ_r_u6["SoM"] | succ_r_u6["Vision"]
            union_5_u6 = union_3_u6 | succ_r_u6["P-text"] | succ_r_u6["P-SoM"]
            union_6 = union_5_u6 | succ_r_u6["P-prompt"]
            sr_3_u6 = 100 * len(union_3_u6) / max(1, len(universe_6))
            sr_5_u6 = 100 * len(union_5_u6) / max(1, len(universe_6))
            sr_6 = 100 * len(union_6) / max(1, len(universe_6))
            in_3_u6 = np.array([t in union_3_u6 for t in u6_sorted], dtype=bool)
            in_5_u6 = np.array([t in union_5_u6 for t in u6_sorted], dtype=bool)
            in_6 = np.array([t in union_6 for t in u6_sorted], dtype=bool)
            ci_lo_6, ci_hi_6 = bootstrap_lift_ci(in_3_u6, in_6)
            ci_lo_6v5, ci_hi_6v5 = bootstrap_lift_ci(in_5_u6, in_6)
            h_6_vs_3 = cohen_h(sr_6 / 100, sr_3_u6 / 100)
            h_6_vs_5 = cohen_h(sr_6 / 100, sr_5_u6 / 100)
            _, wp_6 = wilcoxon_signed_rank(in_3_u6, in_6)
            _, wp_6v5 = wilcoxon_signed_rank(in_5_u6, in_6)
            mc_p_6 = mcnemar_exact_one_sided(in_3_u6, in_6)
            mc_p_6v5 = mcnemar_exact_one_sided(in_5_u6, in_6)
            tost_p_6 = bootstrap_tost_p(in_3_u6, in_6)
            tost_p_6v5 = bootstrap_tost_p(in_5_u6, in_6)
        else:
            sr_6 = None
            ci_lo_6 = ci_hi_6 = ci_lo_6v5 = ci_hi_6v5 = None
            h_6_vs_3 = h_6_vs_5 = None
            wp_6 = wp_6v5 = None
            mc_p_6 = mc_p_6v5 = None
            tost_p_6 = tost_p_6v5 = None
    else:
        sr_4_pprompt = None
        ci_lo_pprompt = ci_hi_pprompt = None
        h_4pprompt_vs_3 = None
        wp_pprompt = None
        mc_p_pprompt = None
        tost_p_pprompt = None
        pprompt_adds = set()
        sr_6 = None
        ci_lo_6 = ci_hi_6 = ci_lo_6v5 = ci_hi_6v5 = None
        h_6_vs_3 = h_6_vs_5 = None
        wp_6 = wp_6v5 = None
        mc_p_6 = mc_p_6v5 = None
        tost_p_6 = tost_p_6v5 = None

    # H3 structural test: phantom space 2-axis empirical validation.
    # For each axis, bootstrap CI on |arm ∖ P-SoM| unique-count + McNemar exact
    # one-sided. CI lower bound > 0 evidences axis contributes tasks P-SoM
    # doesn't solve (i.e., axis is empirically distinct from compound center,
    # phantom space is multi-region not collapsed point).
    #
    # B-330 (/stress A1.9 Mode B F4 OOB + user H3 framing 2026-05-16):
    # universe switched from universe_5 (DOM ∩ SoM ∩ Vision ∩ P-text ∩ P-SoM)
    # to universe_6 (six-arm complete-case: + P-prompt). Per user paper §1
    # framing — P-text and P-prompt are co-equal axis-decomposition arms
    # (not asymmetric "P-prompt is THE axis"), so the natural universe for
    # both axis1 + axis2 is six-arm intersection. Pre-fix universe_5 did
    # NOT require P-prompt coverage → axis2 estimand drift when P-prompt
    # missing on some tasks in universe_5. Trade-off: smaller N (six-arm
    # intersection is the strictest), but estimand interpretable as
    # "phantom-arm structural distinctness on jointly-observed tasks".
    if has_pdom and has_pprompt:
        # Six-arm complete-case: only tasks where ALL 6 arms have data.
        universe6_sorted = sorted(universe_6)
        succ_r_6 = {m: succ[m] & universe_6 for m in succ}
        in_psom_raw = np.array([t in succ_r_6["P-SoM"] for t in universe6_sorted], dtype=bool)
        in_pdom_raw = np.array([t in succ_r_6["P-text"] for t in universe6_sorted], dtype=bool)
        in_pprompt_raw = np.array([t in succ_r_6["P-prompt"] for t in universe6_sorted], dtype=bool)
        h3_axis1_count, h3_axis1_ci_lo, h3_axis1_ci_hi = bootstrap_unique_count_ci(
            in_pdom_raw, in_psom_raw)
        # mcnemar_exact_one_sided(a, b) tests H1: b > a (b adds tasks a misses)
        h3_axis1_mcnemar_p = mcnemar_exact_one_sided(in_psom_raw, in_pdom_raw)
        h3_axis2_count, h3_axis2_ci_lo, h3_axis2_ci_hi = bootstrap_unique_count_ci(
            in_pprompt_raw, in_psom_raw)
        h3_axis2_mcnemar_p = mcnemar_exact_one_sided(in_psom_raw, in_pprompt_raw)
    else:
        # universe_6 unavailable (e.g. P-prompt or P-text cell incomplete);
        # report None rather than fall back to mis-aligned universe_5 estimand.
        # Pre-B-330 fallback to universe_5 silently changed denominator.
        in_psom_raw = np.array([t in succ_r["P-SoM"] for t in universe], dtype=bool)
        h3_axis1_count = h3_axis1_ci_lo = h3_axis1_ci_hi = h3_axis1_mcnemar_p = None
        h3_axis2_count = h3_axis2_ci_lo = h3_axis2_ci_hi = h3_axis2_mcnemar_p = None

    is_partial = (any(len(o) < cell["n_expected"] for o in obs.values()) or not has_pdom
                  or not has_pprompt)

    def maybe_round(value, ndigits=4):
        return None if value is None else round(value, ndigits)

    return {
        "baseline": cell["baseline"],
        "site": cell["site"],
        "n_common": n,
        "n_expected": cell["n_expected"],
        "is_partial": is_partial,
        "has_pdom": has_pdom,
        "has_pprompt": has_pprompt,
        "sr_dom":     round(100 * len(succ_r["DOM"]) / n, 4),
        "sr_som":     round(100 * len(succ_r["SoM"]) / n, 4),
        "sr_vision":  round(100 * len(succ_r["Vision"]) / n, 4),
        "sr_pdom":    (round(100 * len(succ_r["P-text"]) / n, 4) if has_pdom else None),
        "sr_psom":    round(100 * len(succ_r["P-SoM"]) / n, 4),
        "sr_pprompt": (round(100 * len(succ_r["P-prompt"]) / n, 4) if has_pprompt else None),
        "oracle_3mode_pp":  round(sr_3, 4),
        "oracle_4mode_pdom_pp": maybe_round(sr_4_pdom),
        "oracle_4mode_psom_pp": round(sr_4_psom, 4),
        "oracle_4mode_pprompt_pp": maybe_round(sr_4_pprompt),
        "oracle_5mode_pp":  maybe_round(sr_5),
        "oracle_6mode_pp":  maybe_round(sr_6),
        "lift_5_vs_3_pp":   (round(sr_5 - sr_3, 4) if sr_5 is not None else None),
        "lift_5_vs_3_ci95_lo_pp":  maybe_round(ci_lo),
        "lift_5_vs_3_ci95_hi_pp":  maybe_round(ci_hi),
        "lift_4pdom_vs_3_pp":   (round(sr_4_pdom - sr_3, 4) if sr_4_pdom is not None else None),
        "lift_4pdom_vs_3_ci95_lo_pp": maybe_round(ci_lo_pdom),
        "lift_4pdom_vs_3_ci95_hi_pp": maybe_round(ci_hi_pdom),
        # /stress A1.19 P0-4 (2026-05-17, codex Mode B P0-1-B* OOB): mixed-universe
        # lift bug fix. Pre-fix: `sr_4_psom - sr_3` used u_psom numerator (sr_4_psom
        # over u_psom L461) minus common=universe_5 baseline (sr_3 over n=universe_5
        # L470) → point estimate and CI (which uses u_psom both sides L477) referred
        # to different task universes → row math inconsistent. Fix: per-comparison
        # universe consistent — use sr_3_psom_only (over u_psom L460), sr_3_pprompt_only
        # (over u_pprompt L534), sr_3_u6 (over universe_6 L556) so each lift's point
        # estimate matches its CI denominator. F07 audit fix 2026-05-09 was supposed
        # to address this but only fixed CI side; point estimate computation slipped
        # through. universe_label_* columns added for downstream estimand disambig.
        "lift_4psom_vs_3_pp":   round(sr_4_psom - sr_3_psom_only, 4),
        "lift_4psom_vs_3_ci95_lo_pp": round(ci_lo_psom, 4),
        "lift_4psom_vs_3_ci95_hi_pp": round(ci_hi_psom, 4),
        "lift_4psom_vs_3_universe": "universe_psom_only (DOM∩SoM∩Vision∩P-SoM)",
        "lift_4psom_vs_3_n_universe": len(u_psom),
        "lift_4pprompt_vs_3_pp": (
            round(sr_4_pprompt - sr_3_pprompt_only, 4) if sr_4_pprompt is not None else None
        ),
        "lift_4pprompt_vs_3_ci95_lo_pp": maybe_round(ci_lo_pprompt),
        "lift_4pprompt_vs_3_ci95_hi_pp": maybe_round(ci_hi_pprompt),
        "lift_4pprompt_vs_3_universe": (
            "universe_pprompt_only (DOM∩SoM∩Vision∩P-prompt)" if has_pprompt else None
        ),
        "lift_4pprompt_vs_3_n_universe": (len(u_pprompt) if has_pprompt else None),
        "lift_6_vs_3_pp": (round(sr_6 - sr_3_u6, 4) if sr_6 is not None else None),
        "lift_6_vs_3_ci95_lo_pp": maybe_round(ci_lo_6),
        "lift_6_vs_3_ci95_hi_pp": maybe_round(ci_hi_6),
        "lift_6_vs_3_universe": (
            "universe_6 (DOM∩SoM∩Vision∩P-text∩P-SoM∩P-prompt)"
            if (has_pdom and has_pprompt) else None
        ),
        "lift_6_vs_3_n_universe": (len(universe_6) if (has_pdom and has_pprompt) else None),
        "lift_6_vs_5_pp": (
            round(sr_6 - sr_5_u6, 4)
            if (sr_6 is not None and has_pdom and has_pprompt) else None
        ),
        "lift_6_vs_5_ci95_lo_pp": maybe_round(ci_lo_6v5),
        "lift_6_vs_5_ci95_hi_pp": maybe_round(ci_hi_6v5),
        "lift_6_vs_5_universe": (
            "universe_6 (incremental P-prompt over 5-mode oracle on 6-arm complete-case)"
            if (has_pdom and has_pprompt) else None
        ),
        "lift_6_vs_5_n_universe": (len(universe_6) if (has_pdom and has_pprompt) else None),
        # Effect sizes (Cohen's h on oracle proportions)
        "cohen_h_5_vs_3":     maybe_round(h_5_vs_3),
        "cohen_h_5_vs_3_label": (cohen_h_label(h_5_vs_3) if h_5_vs_3 is not None else None),
        "cohen_h_4pdom_vs_3": maybe_round(h_4pdom_vs_3),
        "cohen_h_4pdom_vs_3_label": (cohen_h_label(h_4pdom_vs_3) if h_4pdom_vs_3 is not None else None),
        "cohen_h_4psom_vs_3": round(h_4psom_vs_3, 4),
        "cohen_h_4psom_vs_3_label": cohen_h_label(h_4psom_vs_3),
        "cohen_h_4pprompt_vs_3": maybe_round(h_4pprompt_vs_3),
        "cohen_h_4pprompt_vs_3_label": (cohen_h_label(h_4pprompt_vs_3) if h_4pprompt_vs_3 is not None else None),
        "cohen_h_6_vs_3": maybe_round(h_6_vs_3),
        "cohen_h_6_vs_3_label": (cohen_h_label(h_6_vs_3) if h_6_vs_3 is not None else None),
        "cohen_h_6_vs_5": maybe_round(h_6_vs_5),
        "cohen_h_6_vs_5_label": (cohen_h_label(h_6_vs_5) if h_6_vs_5 is not None else None),
        # Wilcoxon (paired sign on binary)
        "wilcoxon_5_vs_3_p":     wp_5,
        "wilcoxon_4pdom_vs_3_p": wp_pdom,
        "wilcoxon_4psom_vs_3_p": wp_psom,
        "wilcoxon_4pprompt_vs_3_p": wp_pprompt,
        "wilcoxon_6_vs_3_p": wp_6,
        "wilcoxon_6_vs_5_p": wp_6v5,
        # McNemar exact 1-sided
        "mcnemar_5_vs_3_p":     mc_p_5,
        "mcnemar_4pdom_vs_3_p": mc_p_pdom,
        "mcnemar_4psom_vs_3_p": mc_p_psom,
        "mcnemar_4pprompt_vs_3_p": mc_p_pprompt,
        "mcnemar_6_vs_3_p": mc_p_6,
        "mcnemar_6_vs_5_p": mc_p_6v5,
        # TOST equivalence p (bootstrap, δ=1.0pp per preregistration.md §4 lock;
        # max(p_lower, p_upper) < α ⇒ equivalence ACCEPTED, effect bounded within ±δ).
        # NOTE: this is the separate "lift bounded" test, NOT the H1(ii) primary
        # gate (H1(ii) uses one-sided superiority per 2026-05-13 prereg revision).
        "tost_5_vs_3_p":      tost_p_5,
        "tost_4pdom_vs_3_p":  tost_p_pdom,
        "tost_4psom_vs_3_p":  tost_p_psom,
        "tost_4pprompt_vs_3_p": tost_p_pprompt,
        "tost_6_vs_3_p":      tost_p_6,
        "tost_6_vs_5_p":      tost_p_6v5,
        # Family-adjusted p / q (filled by main() post-collection; see §family decl)
        "mcnemar_5_vs_3_p_holm":     None,
        "mcnemar_5_vs_3_q_bh":       None,
        "mcnemar_5_vs_3_p_bonf":     None,
        "mcnemar_4pdom_vs_3_p_holm": None,
        "mcnemar_4pdom_vs_3_q_bh":   None,
        "mcnemar_4psom_vs_3_p_holm": None,
        "mcnemar_4psom_vs_3_q_bh":   None,
        "mcnemar_4pprompt_vs_3_p_holm": None,
        "mcnemar_4pprompt_vs_3_q_bh":   None,
        # H3 structural — phantom space 2-axis empirical validation
        "h3_axis1_unique_count":      h3_axis1_count,
        "h3_axis1_ci95_lo":           (round(h3_axis1_ci_lo, 4) if h3_axis1_ci_lo is not None else None),
        "h3_axis1_ci95_hi":           (round(h3_axis1_ci_hi, 4) if h3_axis1_ci_hi is not None else None),
        "h3_axis1_mcnemar_p":         (round(h3_axis1_mcnemar_p, 6) if h3_axis1_mcnemar_p is not None else None),
        "h3_axis1_mcnemar_p_holm":    None,  # filled by family correction in main()
        "h3_axis2_unique_count":      h3_axis2_count,
        "h3_axis2_ci95_lo":           (round(h3_axis2_ci_lo, 4) if h3_axis2_ci_lo is not None else None),
        "h3_axis2_ci95_hi":           (round(h3_axis2_ci_hi, 4) if h3_axis2_ci_hi is not None else None),
        "h3_axis2_mcnemar_p":         (round(h3_axis2_mcnemar_p, 6) if h3_axis2_mcnemar_p is not None else None),
        "h3_axis2_mcnemar_p_holm":    None,  # filled by family correction in main()
        # Decomposition
        "pdom_adds_count":      (len(pdom_adds) if has_pdom else None),
        "psom_adds_count":      len(psom_adds),
        "pprompt_adds_count":   (len(pprompt_adds) if has_pprompt else None),
        "pdom_only_count":      (len(pdom_only) if has_pdom else None),
        "psom_only_count":      (len(psom_only) if has_pdom else None),
        "both_phantom_overlap_count": (len(both_add) if has_pdom else None),
        # Scenario C sentinel: P-SoM ↔ P-text Jaccard
        "phantom_pair_jaccard": (round(jaccard, 4) if jaccard is not None else None),
        "phantom_pair_jaccard_warn": jaccard_warn,
    }


# /stress A1.10 P0-1-ABC* (2026-05-16): router trigger fire-rate audit.
# Paper §3.5 + §4.X.5 disclosure now records that the three numeric router
# triggers (dom_size_threshold=12000 / dom_complexity_trigger=500 /
# text_length_trigger=12000) empirically fire < 0.5 % under cleaned-AXTree
# regime. This helper lets a reviewer or future paper-2 author validate the
# disclosure against any run dir without joining raw step records by hand.
# It is intentionally opt-in (--audit-fire-rate flag) so it doesn't slow the
# default oracle-lift aggregation pipeline.
_NUMERIC_ROUTER_TRIGGERS = (
    "dom_size_exceeds_threshold",
    "dom_complexity_high",
    "text_length_high",
)


def compute_permutation_null_for_cell(cell: dict, B_perm: int = 10000,
                                       seed: int = 42) -> Optional[dict]:
    """Compute fixed-marginal permutation null for drop-one oracle on a cell.

    /stress A2.1 P0-2-ABC* (2026-05-17, 3-AI overlap) integration. Loads
    per-arm pass vectors via the same ``load()`` helper analyze_cell uses,
    constructs the per-task universe as the intersection of observed
    task_ids across the 4-mode archive arms (DOM, SoM, Vision, P-SoM) plus
    P-text and P-prompt when present, then computes the fixed-marginal
    permutation null for each phantom-arm drop-one oracle.

    Returns ``None`` when required arms are missing or universe is empty.
    Otherwise returns ``{arm: permutation_null_result_dict}`` where each
    inner dict matches the contract of ``permutation_drop_one_null``.
    """
    succ, obs = {}, {}
    for mode, ep_dir in cell["modes"].items():
        s, o = load(ep_dir)
        if len(o) < MIN_EP_FOR_CELL:
            continue
        succ[mode] = s
        obs[mode] = o

    required = ("DOM", "SoM", "Vision", "P-SoM")
    if any(m not in succ for m in required):
        return None
    arms_present = [m for m in ("DOM", "SoM", "Vision", "P-SoM", "P-text",
                                  "P-prompt") if m in succ]
    universe = set.intersection(*(obs[m] for m in arms_present))
    if not universe:
        return None
    universe_sorted = sorted(universe)
    per_arm_vectors = {
        m: np.array([1 if t in succ[m] else 0 for t in universe_sorted],
                     dtype=np.int8)
        for m in arms_present
    }
    out: Dict[str, dict] = {}
    for drop_arm in [m for m in ("P-SoM", "P-text", "P-prompt", "SoM")
                      if m in arms_present]:
        out[drop_arm] = permutation_drop_one_null(
            per_arm_vectors, drop_arm=drop_arm, B_perm=B_perm, seed=seed,
        )
    out["_meta"] = {
        "n_tasks_in_universe": len(universe_sorted),
        "arms_present": arms_present,
        "B_perm": B_perm,
        "seed": seed,
    }
    return out


def audit_router_fire_rate(run_root: Path) -> Dict[str, Any]:
    """Scan all step JSONLs under a run_root and report per-trigger fire rate.
    Returns a dict {trigger_name: {count, total_steps, rate_pct}} plus a
    `disclosure_consistent` boolean indicating whether numeric triggers fire
    below the paper §3.5 disclosure threshold (< 0.5 %).
    """
    counts: Dict[str, int] = {t: 0 for t in _NUMERIC_ROUTER_TRIGGERS}
    counts["__streak_or_action_failed__"] = 0
    total_steps = 0
    for jsonl in sorted(run_root.rglob("*_steps_v2.jsonl")):
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_steps += 1
                r = rec.get("router") or {}
                tr = r.get("trigger_reason")
                triggers_list = tr if isinstance(tr, list) else ([tr] if tr else [])
                for t in triggers_list:
                    t_str = str(t)
                    if t_str in counts:
                        counts[t_str] += 1
                    elif t_str in (
                        "action_failed", "page_unchanged_streak",
                        "no_progress_streak", "checklist_progress_stalled",
                        "checklist_has_failed_items", "v6_cascade_fallback_latched",
                        "v7_learned_route",
                    ):
                        counts["__streak_or_action_failed__"] += 1
    report: Dict[str, Any] = {
        "run_root": str(run_root),
        "total_steps": total_steps,
        "triggers": {},
    }
    max_numeric_pct = 0.0
    for t, c in counts.items():
        pct = (c / total_steps * 100.0) if total_steps else 0.0
        report["triggers"][t] = {"count": c, "rate_pct": round(pct, 4)}
        if t in _NUMERIC_ROUTER_TRIGGERS:
            max_numeric_pct = max(max_numeric_pct, pct)
    report["max_numeric_trigger_rate_pct"] = round(max_numeric_pct, 4)
    report["disclosure_consistent"] = max_numeric_pct < 0.5  # paper §3.5
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(REPO / "results/phantom_paper/phantom_lift.csv"))
    ap.add_argument(
        "--audit-fire-rate", metavar="RUN_ROOT", default=None,
        help="Run router trigger fire-rate audit on the given run dir and exit "
             "(P0-1-ABC* audit gate; validates paper §3.5 / §4.X.5 disclosure)."
    )
    ap.add_argument(
        "--permute-marginal-null", action="store_true", default=False,
        help="/stress A2.1 P0-2-ABC* (2026-05-17): for each cell, compute the "
             "fixed-marginal permutation null distribution for the drop-one "
             "oracle of each arm (10000 permutations shuffled within marginal "
             "pass counts per arm) and append observed-excess + null-p95 + "
             "permutation p-value columns to the output CSV. Replaces "
             "H0:drop_one=0 as the paper-grade superiority null (paper §1 "
             "[^null-framing] footnote)."
    )
    ap.add_argument(
        "--permutation-B", type=int, default=10000,
        help="Number of permutations for --permute-marginal-null (default 10000, "
             "prereg-locked)."
    )
    ap.add_argument(
        "--stratified-bootstrap", action="store_true", default=False,
        help="/stress A2.1 P0-5-B* (2026-05-17, codex unique OOB): in addition to "
             "the standard task-resampling CI, emit a stratified-bootstrap CI "
             "column resampling within (task_family × evaluator_type × "
             "reference_image_present) strata. Robustness check on the standard "
             "exchangeability assumption — VWA task taxonomy heterogeneity may "
             "violate it. Falls back to the standard paired bootstrap when only "
             "one stratum is observed for a cell."
    )
    args = ap.parse_args()

    if args.audit_fire_rate:
        report = audit_router_fire_rate(Path(args.audit_fire_rate))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report["disclosure_consistent"]:
            print(
                "WARNING: max numeric trigger fire rate "
                f"{report['max_numeric_trigger_rate_pct']:.3f} % exceeds the "
                "paper §3.5 disclosure threshold of 0.5 %. Either the router "
                "is firing more than disclosed (good — update the disclosure) "
                "or the audit ran on legacy archive data with mixed thresholds.",
                file=__import__("sys").stderr,
            )
            return 1
        return 0

    rows = []
    skipped = []
    permutation_null_results: list = []
    for cell in CELLS:
        r = analyze_cell(cell)
        if r is None:
            skipped.append(f"{cell['baseline']} {cell['site']}")
            continue
        rows.append(r)
        if args.permute_marginal_null:
            null_for_cell = compute_permutation_null_for_cell(
                cell, B_perm=args.permutation_B
            )
            if null_for_cell is not None:
                permutation_null_results.append({
                    "baseline": cell["baseline"],
                    "site": cell["site"],
                    "null": null_for_cell,
                })
                # Surface key permutation-null scalars onto the main row so
                # downstream readers see them next to the bootstrap CI without
                # opening the supplementary JSON.
                for arm in ("P-SoM", "P-text", "P-prompt", "SoM"):
                    entry = null_for_cell.get(arm)
                    if entry is None:
                        continue
                    short = arm.replace("-", "").lower()  # psom / ptext / pprompt / som
                    r[f"perm_null_{short}_obs_pp"] = entry["observed_drop_one_pp"]
                    r[f"perm_null_{short}_p95"] = entry["null_p95"]
                    r[f"perm_null_{short}_excess_pp"] = entry["excess_over_null_p95_pp"]
                    r[f"perm_null_{short}_p_one_sided"] = entry["p_value_one_sided"]

    if args.permute_marginal_null and permutation_null_results:
        perm_out = Path(args.output).with_name(
            Path(args.output).stem + "_permutation_null.json"
        )
        perm_out.parent.mkdir(parents=True, exist_ok=True)
        perm_out.write_text(
            json.dumps(permutation_null_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[perm-null] wrote {perm_out} ({len(permutation_null_results)} cells, "
              f"B_perm={args.permutation_B})")

    # ── Multiple-comparison correction (per pre-registered family) ────────
    # Comparison families:
    #   PRIMARY (m = N_cells):           3→5-mode lift (one per cell)
    #   SECONDARY (m = 3 × N_cells):     per-arm drop-one (P-text/P-SoM/P-prompt)
    #   TERTIARY (m = 2 × N_cells):      6-mode oracle (vs 3 / vs 5) — exploratory
    # Method: Holm-Bonferroni step-down per family (FWER) + BH FDR (informational)
    # Primary p-value: McNemar exact one-sided (directional H1: phantom adds tasks)
    # Wilcoxon two-sided remains uncorrected as secondary report.

    def _adjust_inplace(rows, key_p, key_holm, key_bh, key_bonf=None):
        """Run Bonferroni / Holm / BH on a list of rows for a given p-value field."""
        pvals = [r.get(key_p) for r in rows]
        holm = holm_bonferroni_adjust(pvals)
        bh = bh_fdr_adjust(pvals)
        bonf = bonferroni_adjust(pvals) if key_bonf else [None] * len(rows)
        for r, h, q, b in zip(rows, holm, bh, bonf):
            r[key_holm] = round(h, 6) if h is not None else None
            r[key_bh] = round(q, 6) if q is not None else None
            if key_bonf:
                r[key_bonf] = round(b, 6) if b is not None else None

    # Family A (PRIMARY): 3→5-mode lift
    _adjust_inplace(rows, "mcnemar_5_vs_3_p",
                    "mcnemar_5_vs_3_p_holm", "mcnemar_5_vs_3_q_bh",
                    key_bonf="mcnemar_5_vs_3_p_bonf")

    # Family B (SECONDARY): per-arm drop-one. Pool across cells × {pdom, psom, pprompt}.
    flat_secondary = []
    for r in rows:
        for arm in ("4pdom", "4psom", "4pprompt"):
            p = r.get(f"mcnemar_{arm}_vs_3_p")
            flat_secondary.append((r, arm, p))
    holm_b = holm_bonferroni_adjust([t[2] for t in flat_secondary])
    bh_b = bh_fdr_adjust([t[2] for t in flat_secondary])
    for (r, arm, _), h, q in zip(flat_secondary, holm_b, bh_b):
        r[f"mcnemar_{arm}_vs_3_p_holm"] = round(h, 6) if h is not None else None
        r[f"mcnemar_{arm}_vs_3_q_bh"] = round(q, 6) if q is not None else None

    # H3 STRUCTURAL family: per-axis structural test (axis 1 = P-text, axis 2 = P-prompt).
    # Holm-corrected separately within each axis sub-family (axis 1 / axis 2),
    # because structural claim is weaker than deployment — separate family
    # avoids inflating PRIMARY/SECONDARY family m count.
    for axis_key in ("h3_axis1_mcnemar_p", "h3_axis2_mcnemar_p"):
        ps = [r.get(axis_key) for r in rows]
        holm = holm_bonferroni_adjust(ps)
        for r, p_h in zip(rows, holm):
            r[f"{axis_key}_holm"] = round(p_h, 6) if p_h is not None else None

    # F02 audit fix 2026-05-09: refuse to clobber phantom_lift.csv with
    # an empty result (would silently erase input to all paper figures).
    # Set P79_ALLOW_EMPTY=1 only for explicit dry-runs.
    if not rows:
        msg = (
            f"No paper-grade cells available (skipped: {skipped or 'none'}). "
            "Refusing to write empty phantom_lift.csv. "
            "Check `results/phantom_paper/run_manifest.yaml` and "
            "`scripts/analysis/lib/run_registry.py` filters."
        )
        if os.environ.get("P79_ALLOW_EMPTY", "") in ("1", "true"):
            print(f"WARNING (P79_ALLOW_EMPTY=1): {msg}")
        else:
            raise SystemExit(f"ERROR: {msg}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} cells)")
    if skipped:
        print(f"skipped (incomplete): {', '.join(skipped)}")

    # Markdown summary
    md = out.with_suffix(".md")
    n_cells_primary = len(rows)
    n_secondary = sum(1 for r in rows for arm in ("4pdom", "4psom", "4pprompt")
                      if r.get(f"mcnemar_{arm}_vs_3_p") is not None)
    lines = [
        "# Phantom routing lift — APPENDIX EXPLORATORY (3→5-mode oracle lift estimand)",
        "",
        "> ⚠️ **/stress A1.19 P1-9-B (2026-05-17, codex Mode B): NOT paper §1 hero.**",
        "> The canonical paper §1 H1 PRIMARY gate (per `preregistration.md` lock 2026-05-14 + B-184)",
        "> is **P-SoM drop-one fixed-effects superiority test against δ=1.0pp**, computed by",
        "> `aggregate_phase1_prereg_gate.py` → `results/phantom_paper/phase1_prereg_gate.{csv,json,md}`.",
        "> THIS file (`phantom_lift.{csv,md}`) computes the **legacy 3→5-mode oracle lift**",
        "> (add P-text + P-SoM TO 3-mode baseline), a fundamentally different estimand.",
        "> Per B-184 demotion, this file is **appendix sensitivity only** — do NOT cite",
        "> from paper §1 hero prose. Cite `phase1_prereg_gate.md` for §1 hero.",
        "",
        "Routing lift = (X-mode oracle ceiling) - (3-mode oracle ceiling), where",
        "3-mode = DOM ∪ SoM ∪ Vision (baseline). 95% CI from 1000-resample",
        "task-level bootstrap. Cohen's h effect size (small <0.2, medium 0.2-0.5,",
        "large 0.5-0.8). Wilcoxon paired (binary, equiv to sign test). McNemar",
        "exact 1-sided (H1: extra mode adds tasks).",
        "",
        "## Comparison family declaration (legacy 2026-05-03 framework, appendix-only)",
        "",
        "**Hierarchy (LEGACY APPENDIX framing; canonical paper-grade family is in `phase1_prereg_gate.md`)**:",
        "",
        f"- **APPENDIX legacy exploratory family — H1-LEGACY (3→5 add-to-3-mode lift)**: m = {n_cells_primary}.",
        "  Different estimand from paper §1 hero (which is P-SoM drop-one over 6-mode universe).",
        "  Reported as appendix sensitivity — NOT a paper-claim gate per B-184 demotion 2026-05-16.",
        f"- **STRUCTURAL family — H3 (Phantom space 2-axis empirical evidence)**: per axis, m = N_cells.",
        "  axis 1 = P-text ∖ P-SoM unique-count; axis 2 = P-prompt ∖ P-SoM unique-count.",
        f"- **EXPLORATORY family — H4 (P-text/P-prompt drop-one magnitudes)**: m = {n_secondary}.",
        "  Holm/BH q reported for transparency; NOT used for paper claim gating.",
        "- **TERTIARY (post-hoc, uncorrected)**: 6-mode oracle vs 3 / vs 5.",
        "",
        "## ⚠️ Superiority vs Equivalence (TOST) cognitive-conflict warning (/stress A1.19 P1-13-C)",
        "",
        "> **Two distinct tests reported in this table against the same δ=1.0pp boundary**:",
        "> ",
        "> 1. **Holm-corrected McNemar one-sided p (sig ✅)** = Superiority test: H0: θ ≤ 0",
        "> vs H1: θ > 0 (phantom adds tasks). Small p ⇒ effect significantly **above zero**.",
        "> Paper-grade hero in `phase1_prereg_gate.md` uses H0: θ_FE ≤ +1.0pp (δ as the *substantive-effect*",
        "> threshold), but THIS legacy file tests vs zero — see column header.",
        "> ",
        "> 2. **TOST p** = Equivalence test: H0: |θ| ≥ 1.0pp vs H1: |θ| < 1.0pp (no substantive effect).",
        "> Small TOST p ⇒ effect **bounded within ±1pp** (i.e., equivalent to zero within margin).",
        "> ",
        "> The two tests probe **disjoint hypotheses**: superiority asks 'is it big enough?';",
        "> equivalence asks 'is it small enough?'. Same δ=1.0pp value used for both is",
        "> intentional (mirror H1 substantive-effect threshold) but a reviewer reading this",
        "> table without context may confuse them. Always reference each test's row label.",
        "",
        "Adjustment methods:",
        "- **Holm** (Holm 1979) — step-down FWER control, legacy gating PRIMARY + STRUCTURAL.",
        "- **BH q** (Benjamini-Hochberg 1995) — FDR control, informational.",
        "- **Bonf** — Bonferroni FWER (legacy PRIMARY only, conservative reference).",
        "- **TOST p** — Two One-Sided Test for equivalence at δ=1.0pp (commit-locked).",
        "  TOST p = max(p_lower, p_upper); p < α ⇒ equivalence ACCEPTED (effect bounded",
        "  within ±δ).",
        "",
        "Primary p-value going through correction: **McNemar exact one-sided**",
        "(directly maps to H1-LEGACY: phantom adds tasks vs zero). Wilcoxon two-sided",
        "is reported uncorrected as secondary cross-check.",
        "",
        "## Routing lift summary (5-mode vs 3-mode + each single phantom)",
        "",
        "| Baseline | Site | N | 3→5-mode lift | 95% CI | Cohen's h | Wilcoxon p | McNemar p | Holm p | BH q | Bonf p | TOST p | sig (Holm 0.05) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    def _fmt(p):
        return f"{p:.4f}" if p is not None else "—"
    for r in rows:
        n_label = (f"{r['n_common']}/{r['n_expected']}†" if r["is_partial"]
                   else f"{r['n_common']}")
        if r.get("lift_5_vs_3_pp") is None:
            lines.append(
                f"| {r['baseline']} | {r['site']} | {n_label} | n/a (P-text pending) | "
                + " | ".join(["—"] * 10) + " |"
            )
            continue
        holm_p = r.get("mcnemar_5_vs_3_p_holm")
        sig = "✅" if (holm_p is not None and holm_p < 0.05) else (
            "🟡" if r["lift_5_vs_3_ci95_lo_pp"] > 0 else "❌"
        )
        lines.append(
            f"| {r['baseline']} | {r['site']} | {n_label} | "
            f"+{r['lift_5_vs_3_pp']:.2f}pp | "
            f"[{r['lift_5_vs_3_ci95_lo_pp']:.2f}, {r['lift_5_vs_3_ci95_hi_pp']:.2f}] | "
            f"{r['cohen_h_5_vs_3']:.3f} ({r['cohen_h_5_vs_3_label']}) | "
            f"{_fmt(r['wilcoxon_5_vs_3_p'])} | {_fmt(r['mcnemar_5_vs_3_p'])} | "
            f"{_fmt(r.get('mcnemar_5_vs_3_p_holm'))} | "
            f"{_fmt(r.get('mcnemar_5_vs_3_q_bh'))} | "
            f"{_fmt(r.get('mcnemar_5_vs_3_p_bonf'))} | "
            f"{_fmt(r.get('tost_5_vs_3_p'))} | {sig} |"
        )

    lines += [
        "",
        "## Single-phantom upgrade lifts (4-mode vs 3-mode)",
        "",
        "| Baseline | Site | +P-text lift | CI | h | +P-SoM lift | CI | h | +P-prompt lift | CI | h |",
        "|---|---|---:|---|---:|---:|---|---:|---:|---|---:|",
    ]
    for r in rows:
        if r.get("lift_4pdom_vs_3_pp") is None:
            pdom_cell = "n/a"
            pdom_ci = "—"
            pdom_h = "—"
        else:
            pdom_cell = f"+{r['lift_4pdom_vs_3_pp']:.2f}pp"
            pdom_ci = f"[{r['lift_4pdom_vs_3_ci95_lo_pp']:.2f}, {r['lift_4pdom_vs_3_ci95_hi_pp']:.2f}]"
            pdom_h = f"{r['cohen_h_4pdom_vs_3']:.3f}"
        if r.get("lift_4pprompt_vs_3_pp") is None:
            pprompt_cell = "n/a (pending)"
            pprompt_ci = "—"
            pprompt_h = "—"
        else:
            pprompt_cell = f"+{r['lift_4pprompt_vs_3_pp']:.2f}pp"
            pprompt_ci = f"[{r['lift_4pprompt_vs_3_ci95_lo_pp']:.2f}, {r['lift_4pprompt_vs_3_ci95_hi_pp']:.2f}]"
            pprompt_h = f"{r['cohen_h_4pprompt_vs_3']:.3f}"
        lines.append(
            f"| {r['baseline']} | {r['site']} | "
            f"{pdom_cell} | {pdom_ci} | {pdom_h} | "
            f"+{r['lift_4psom_vs_3_pp']:.2f}pp | "
            f"[{r['lift_4psom_vs_3_ci95_lo_pp']:.2f}, {r['lift_4psom_vs_3_ci95_hi_pp']:.2f}] | "
            f"{r['cohen_h_4psom_vs_3']:.3f} | "
            f"{pprompt_cell} | {pprompt_ci} | {pprompt_h} |"
        )

    # ── Secondary family: per-arm drop-one adjusted p / TOST ──
    lines += [
        "",
        "## Per-arm drop-one — multiple-comparison adjusted (SECONDARY family)",
        "",
        f"Holm-Bonferroni step-down across m = {n_secondary} tests (cells × 3 arms).",
        "BH q-value is FDR-adjusted (informational). TOST p tests equivalence at",
        "δ=1.0pp (preregistration.md §4 lock); max(p_lower, p_upper) < α ⇒ equivalence",
        "ACCEPTED (effect bounded within ±δ). Separate from H1(ii) superiority gate.",
        "",
        # B-175 (/stress A1.4b-i codex B4): column header "TOST sig" placed next
        # to "sig (Holm 0.05)" invited inversion misreading ("TOST sig=✓ means
        # positive lift" is WRONG — it means equivalence accepted = bounded near
        # zero, the OPPOSITE of lift significance). Rename + footnote-style hint
        # to make the semantic explicit.
        "| Baseline | Site | Arm | Lift | 95% CI | McNemar p | Holm p | BH q | TOST p | sig_lift (Holm 0.05) | equiv_within_1pp (TOST 0.05) |",
        "|---|---|---|---:|---|---:|---:|---:|---:|:---:|:---:|",
        "",
        "_Note: `equiv_within_1pp ✅` means the effect is statistically bounded near zero;"
        " it is NOT evidence of positive lift. `sig_lift ✅` and `equiv_within_1pp ✅` can"
        " coexist only for the smallest effects within the ±δ band — they are not opposites._",
        "",
    ]
    arm_meta = [
        ("4pdom", "P-text", "lift_4pdom_vs_3"),
        ("4psom", "P-SoM", "lift_4psom_vs_3"),
        ("4pprompt", "P-prompt", "lift_4pprompt_vs_3"),
    ]
    for r in rows:
        for code, label, lift_prefix in arm_meta:
            lift_pp = r.get(f"{lift_prefix}_pp")
            if lift_pp is None:
                # 11 cols total: baseline + site + arm + lift + 7 metric cols
                lines.append(
                    f"| {r['baseline']} | {r['site']} | {label} | n/a | "
                    + " | ".join(["—"] * 7) + " |"
                )
                continue
            ci_lo = r.get(f"{lift_prefix}_ci95_lo_pp")
            ci_hi = r.get(f"{lift_prefix}_ci95_hi_pp")
            mp = r.get(f"mcnemar_{code}_vs_3_p")
            holm = r.get(f"mcnemar_{code}_vs_3_p_holm")
            bh = r.get(f"mcnemar_{code}_vs_3_q_bh")
            tost = r.get(f"tost_{code}_vs_3_p")
            sig_holm = "✅" if (holm is not None and holm < 0.05) else "❌"
            sig_tost = "✅" if (tost is not None and tost < 0.05) else "❌"
            lines.append(
                f"| {r['baseline']} | {r['site']} | {label} | "
                f"+{lift_pp:.2f}pp | [{ci_lo:.2f}, {ci_hi:.2f}] | "
                f"{_fmt(mp)} | {_fmt(holm)} | {_fmt(bh)} | {_fmt(tost)} | "
                f"{sig_holm} | {sig_tost} |"
            )

    # ── H3 STRUCTURAL family: 2-axis empirical evidence ──
    lines += [
        "",
        "## H3 Structural — phantom space 2-axis empirical validation",
        "",
        "Tests whether each phantom-space axis contributes tasks NOT solved by",
        "P-SoM (the cube-center compound). Lower CI bound > 0 evidences that the",
        "axis is empirically distinct from P-SoM, i.e., phantom space is a",
        "multi-region structure rather than collapsed to a single point.",
        "",
        "**This is the structural claim, NOT the deployment claim.** Magnitude",
        "threshold is low (≥ 2 unique tasks ≈ 1pp); commit-locked floor in",
        "preregistration.md.",
        "",
        "**Primary gating test**: bootstrap CI on unique-count, lower bound > 0.",
        "This tests the existence of non-overlap (structural multi-region",
        "evidence). McNemar one-sided p is a secondary directional asymmetry",
        "report (tests if axis dominates P-SoM in unique contribution), informational only.",
        "",
        "| Baseline | Site | Axis | Arm ∖ P-SoM unique count | 95% bootstrap CI | sig (CI > 0) ⭐ | McNemar p (asymmetry, secondary) | Holm p | sig (Holm 0.05) |",
        "|---|---|---|---:|---|:---:|---:|---:|:---:|",
    ]
    for r in rows:
        for axis_label, axis_arm, count_key, ci_lo_key, ci_hi_key, mc_key in [
            ("axis 1", "P-text",   "h3_axis1_unique_count", "h3_axis1_ci95_lo",
             "h3_axis1_ci95_hi", "h3_axis1_mcnemar_p"),
            ("axis 2", "P-prompt", "h3_axis2_unique_count", "h3_axis2_ci95_lo",
             "h3_axis2_ci95_hi", "h3_axis2_mcnemar_p"),
        ]:
            count = r.get(count_key)
            if count is None:
                # 9 cols total
                lines.append(
                    f"| {r['baseline']} | {r['site']} | {axis_label} ({axis_arm}) | n/a (arm pending) | "
                    + " | ".join(["—"] * 5) + " |"
                )
                continue
            ci_lo = r.get(ci_lo_key)
            ci_hi = r.get(ci_hi_key)
            mc_p = r.get(mc_key)
            holm_p = r.get(f"{mc_key}_holm")
            sig_holm = "✅" if (holm_p is not None and holm_p < 0.05) else "❌"
            sig_ci = "✅" if (ci_lo is not None and ci_lo > 0) else "❌"
            lines.append(
                f"| {r['baseline']} | {r['site']} | {axis_label} ({axis_arm}) | "
                f"{int(count)} tasks | "
                f"[{ci_lo:.1f}, {ci_hi:.1f}] | "
                f"{sig_ci} | "
                f"{_fmt(mc_p)} | {_fmt(holm_p)} | "
                f"{sig_holm} |"
            )

    # 6-mode oracle (when P-prompt + P-text both present)
    lines += [
        "",
        "## 6-mode oracle (5-mode + P-prompt)",
        "",
        "| Baseline | Site | 6-mode SR | 6 vs 3 lift | CI | h | 6 vs 5 lift | CI | h |",
        "|---|---|---:|---:|---|---:|---:|---|---:|",
    ]
    for r in rows:
        if r.get("oracle_6mode_pp") is None:
            lines.append(
                f"| {r['baseline']} | {r['site']} | n/a (pending) | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {r['baseline']} | {r['site']} | {r['oracle_6mode_pp']:.2f}% | "
            f"+{r['lift_6_vs_3_pp']:.2f}pp | "
            f"[{r['lift_6_vs_3_ci95_lo_pp']:.2f}, {r['lift_6_vs_3_ci95_hi_pp']:.2f}] | "
            f"{r['cohen_h_6_vs_3']:.3f} | "
            f"+{r['lift_6_vs_5_pp']:.2f}pp | "
            f"[{r['lift_6_vs_5_ci95_lo_pp']:.2f}, {r['lift_6_vs_5_ci95_hi_pp']:.2f}] | "
            f"{r['cohen_h_6_vs_5']:.3f} |"
        )

    if any(r["is_partial"] for r in rows):
        lines.append("")
        lines.append("† = partial (any mode < expected N); using intersection of observed tasks.")
    lines += [
        "",
        "## Decomposition: which phantom contributes which tasks",
        "",
        "| Baseline | Site | P-text adds | P-SoM adds | P-prompt adds | P-text only | P-SoM only | Both phantoms overlap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        n = max(r["n_common"], 1)
        def cell(val):
            return "n/a" if val is None else f"{val} ({100*val/n:.2f}pp)"
        lines.append(
            f"| {r['baseline']} | {r['site']} | "
            f"{cell(r['pdom_adds_count'])} | "
            f"{cell(r['psom_adds_count'])} | "
            f"{cell(r.get('pprompt_adds_count'))} | "
            f"{cell(r['pdom_only_count'])} | "
            f"{cell(r['psom_only_count'])} | "
            f"{cell(r['both_phantom_overlap_count'])} |"
        )

    # Scenario C sentinel: P-SoM ↔ P-text Jaccard
    lines += [
        "",
        "## Scenario C sentinel — P-SoM ↔ P-text task-pool Jaccard",
        "",
        "Threshold: Jaccard > 0.7 → phantoms become routing-redundant; paper",
        "Section 5 axis-2 prompt-effect claim weakens. Current paper claim",
        "(\"prompt creates task-pool divergence without uniform SR change\")",
        "requires Jaccard ≤ 0.7 across cells.",
        "",
        "| Baseline | Site | Jaccard | Status |",
        "|---|---|---:|:---:|",
    ]
    for r in rows:
        if r.get("phantom_pair_jaccard") is None:
            lines.append(
                f"| {r['baseline']} | {r['site']} | n/a | ⏳ P-text pending |"
            )
            continue
        if r["phantom_pair_jaccard_warn"]:
            status = "🔴 > 0.7 (WARN: redundant)"
        elif r["phantom_pair_jaccard"] > 0.6:
            status = "🟡 > 0.6 (watch)"
        else:
            status = "✅ ≤ 0.6 (safe)"
        lines.append(
            f"| {r['baseline']} | {r['site']} | {r['phantom_pair_jaccard']:.3f} | {status} |"
        )

    if skipped:
        lines += ["", f"_Cells pending data (skipped): {', '.join(skipped)}_"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
