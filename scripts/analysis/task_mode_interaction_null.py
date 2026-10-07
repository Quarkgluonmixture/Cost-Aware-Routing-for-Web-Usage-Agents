#!/usr/bin/env python3
"""Is there reproducible task×mode structure beyond task difficulty? A no-interaction null that
conditions on the margins — 2026-10-07, post_hoc_exploratory (笔记 §543)

Why this exists. The routing story's load-bearing explanation is "label supply": the per-task
which-mode signal is smaller than run-to-run noise, so single-run labels cannot train a router.
Reviewers of both workshop versions said that explanation is not yet established (VLM4RWD 1MR9
#1, REALM). Its most direct evidence, the variance decomposition of 笔记 §505.7, had no tracked
producer, covered two cells and carried no uncertainty; the difficulty floor of
`label_instability` compared a six-arm union with a single-arm rate (§541). This product
replaces both readings with one model-based test on the three cells where all six arms have a
same-condition rerun (cls_B0, red_B0, wared_B1).

Data per cell: Y[task, mode, run] in {0, 1}, six modes × two runs (run a = canonical, b = the
registered replicate, `lib/replicate_pairs`).

Null. logit P(Y_imr = 1) = a_i + b_mr: every task has its own difficulty, every (mode, run)
column its own easiness, and there is NO persistent task×mode interaction. Under this (Rasch)
model the row and column sums are sufficient, so conditional on them every 0/1 matrix with
those margins is equally likely. The null is sampled exactly that way: curveball trades on the
task × (mode, run) matrix, which keep every task's total and every column's success count
fixed. Nothing is fitted, and task difficulty is conditioned on rather than proxied (cf. the
k/6 proxy of `label_instability`). Mixing is checked by re-running the primary statistic's null
with 4× the thinning and comparing.

Statistics, each recomputed on every null matrix:
  PRIMARY  sigma2_int — the noise-corrected task×mode interaction component of the two-way
           random-effects ANOVA with r = 2 replicates, (MS_int − MS_err) / 2. It is the
           run-a/run-b covariance of the interaction residuals, i.e. how much of a task's mode
           profile reproduces on a rerun. Holm across the three cells.
  secondary  the interaction/noise ratio; the contested-task flip rate and its enrichment over
           the complement (label_instability's statistic, contested defined on run a);
           the number of tasks with a stable strict preference (some mode 2/2, another 0/2).
Descriptive (no null): the full decomposition (task, mode, interaction, noise) with task-bootstrap
95% intervals, and the D-study — reliability of a task×mode label averaged over k runs,
sigma2_int / (sigma2_int + sigma2_err / k), and the k at which it passes 0.5 (interaction ≥
noise/k), with intervals.

What the null does NOT exclude: task difficulty itself (the a_i), which is exactly what the
budget line routes on. A pass means "there is task×mode structure the margins do not explain";
a fail means "the observed mode profiles reproduce no better than a model without any".

Usage:
  python scripts/analysis/task_mode_interaction_null.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.lib.replicate_pairs import (  # noqa: E402
    MODE_KEYS, cell_id, full_paired_cells, outcome_matrix,
)

SEED = 42
N_NULL = 2000
N_CHAINS = 4
N_BOOT = 2000
THIN_PER_ROW = 5          # curveball trades between kept samples, per task row
BURN_PER_ROW = 50
OUT_MD = REPO / "docs/analysis/cross_sites/task_mode_interaction_null.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/task_mode_interaction_null.json"
M = len(MODE_KEYS)


# ---------------------------------------------------------------------------- data

def load(baseline: str, site_key: str) -> tuple[list[int], np.ndarray]:
    a = outcome_matrix(baseline, site_key, "a")
    b = outcome_matrix(baseline, site_key, "b")
    tasks = sorted(a)
    Y = np.array([[[a[t][m], b[t][m]] for m in MODE_KEYS] for t in tasks], dtype=float)
    return tasks, Y            # (n, 6, 2)


# ---------------------------------------------------------------------------- statistics

def anova(Y: np.ndarray) -> dict:
    """Two-way random-effects ANOVA (task × mode) with r replicates per cell."""
    n, m, r = Y.shape
    g = Y.mean()
    cm = Y.mean(2)                         # task × mode cell means
    ti = cm.mean(1)                        # task means
    mj = cm.mean(0)                        # mode means
    ss_task = m * r * ((ti - g) ** 2).sum()
    ss_mode = n * r * ((mj - g) ** 2).sum()
    ss_int = r * ((cm - ti[:, None] - mj[None, :] + g) ** 2).sum()
    ss_err = ((Y - cm[:, :, None]) ** 2).sum()
    ms_task, ms_mode = ss_task / (n - 1), ss_mode / (m - 1)
    ms_int, ms_err = ss_int / ((n - 1) * (m - 1)), ss_err / (n * m * (r - 1))
    s_err = ms_err
    s_int = (ms_int - ms_err) / r
    s_task = (ms_task - ms_int) / (m * r)
    s_mode = (ms_mode - ms_int) / (n * r)
    return {"task": s_task, "mode": s_mode, "interaction": s_int, "noise": s_err,
            "interaction_over_noise": s_int / s_err if s_err > 0 else None}


def label_stats(Y: np.ndarray) -> dict:
    a, b = Y[:, :, 0], Y[:, :, 1]
    solved_a = a.sum(1)
    contested = (solved_a > 0) & (solved_a < Y.shape[1])
    flipped = (a != b).any(1)
    fc = flipped[contested].mean() if contested.any() else float("nan")
    fo = flipped[~contested].mean() if (~contested).any() else float("nan")
    both = Y.sum(2)                        # 0, 1 or 2 successes per (task, mode)
    stable = ((both == 2).any(1) & (both == 0).any(1))
    return {"contested_flip_rate": float(fc), "complement_flip_rate": float(fo),
            "enrichment": float(fc / fo) if fo > 0 else None,
            "n_stable_strict_preference": int(stable.sum())}


def statistics(Y: np.ndarray) -> dict:
    d = anova(Y)
    return {"sigma2_int": d["interaction"], "int_over_noise": d["interaction_over_noise"],
            **label_stats(Y)}


# ---------------------------------------------------------------------------- curveball null

def _to_rows(Y: np.ndarray) -> list[int]:
    flat = Y.reshape(len(Y), -1).astype(int)          # columns = (mode, run), 12 bits
    w = 1 << np.arange(flat.shape[1])
    return [int(v) for v in flat @ w]


def _to_matrix(rows: list[int], n_cols: int) -> np.ndarray:
    arr = np.array(rows, dtype=np.int64)[:, None]
    bits = (arr >> np.arange(n_cols)) & 1
    return bits.reshape(len(rows), n_cols // 2, 2).astype(float)   # any arm count (§550)


def _bits(x: int) -> list[int]:
    out = []
    while x:
        low = x & -x
        out.append(low)
        x ^= low
    return out


def curveball(rows: list[int], n_trades: int, rng: np.random.Generator) -> None:
    """In-place curveball trades; each keeps both rows' totals and every column total."""
    n = len(rows)
    pairs = rng.integers(0, n, size=(n_trades, 2))
    for i, j in pairs:
        if i == j:
            continue
        a, b = rows[i], rows[j]
        da, db = a & ~b, b & ~a
        if not da or not db:
            continue
        pool = _bits(da | db)
        k = len(_bits(da))
        pick = rng.permutation(len(pool))[:k]
        new_da = 0
        for p in pick:
            new_da |= pool[p]
        shared = a & b
        rows[i] = shared | new_da
        rows[j] = shared | ((da | db) ^ new_da)


def null_draws(Y: np.ndarray, n_draws: int, thin_per_row: int, seed: int) -> list[dict]:
    n = len(Y)
    n_cols = Y.shape[1] * Y.shape[2]
    per_chain = math.ceil(n_draws / N_CHAINS)
    out = []
    for c in range(N_CHAINS):
        rng = np.random.default_rng(seed + c)
        rows = _to_rows(Y)
        curveball(rows, BURN_PER_ROW * n, rng)
        for _ in range(per_chain):
            curveball(rows, thin_per_row * n, rng)
            out.append(statistics(_to_matrix(rows, n_cols)))
    return out[:n_draws]


# ---------------------------------------------------------------------------- bootstrap / D-study

def bootstrap(Y: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(Y)
    comps = {k: [] for k in ("task", "mode", "interaction", "noise", "interaction_over_noise")}
    kstar = []
    for _ in range(N_BOOT):
        d = anova(Y[rng.integers(0, n, n)])
        for k in comps:
            comps[k].append(d[k])
        kstar.append(k_needed(d["interaction"], d["noise"]))
    ci = {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in comps.items()}
    ks = np.array([np.inf if v is None else v for v in kstar])
    ci["k_needed"] = [float(np.quantile(ks, 0.025)), float(np.quantile(ks, 0.975))]
    return ci


def k_needed(s_int: float, s_err: float) -> float | None:
    """Smallest integer k with s_int ≥ s_err / k (label reliability ≥ 0.5); None if s_int ≤ 0."""
    if s_int <= 0:
        return None
    return float(math.ceil(s_err / s_int))


def reliability(s_int: float, s_err: float, k: int) -> float | None:
    if s_int <= 0:
        return 0.0
    return s_int / (s_int + s_err / k)


# ---------------------------------------------------------------------------- per cell

def p_upper(obs: float, null: list[float]) -> float:
    return (sum(v >= obs - 1e-12 for v in null) + 1) / (len(null) + 1)


def run_cell(cell: tuple[str, str], n_null: int = N_NULL) -> dict:
    t0 = time.time()
    baseline, site = cell
    tasks, Y = load(baseline, site)
    obs = statistics(Y)
    dec = anova(Y)
    nd = null_draws(Y, n_null, THIN_PER_ROW, SEED)
    nd_check = null_draws(Y, n_null // 4, 4 * THIN_PER_ROW, SEED + 100)

    def col(draws, key):
        return [d[key] for d in draws if d[key] is not None and not math.isnan(d[key])]

    null = {}
    for key in ("sigma2_int", "int_over_noise", "contested_flip_rate", "enrichment",
                "n_stable_strict_preference"):
        v = col(nd, key)
        null[key] = {"observed": obs[key], "null_mean": float(np.mean(v)),
                     "null_q05": float(np.quantile(v, 0.05)), "null_q95": float(np.quantile(v, 0.95)),
                     "p_upper": p_upper(obs[key], v) if obs[key] is not None else None}
    chk = col(nd_check, "sigma2_int")
    mixing = {"thin_per_row": [THIN_PER_ROW, 4 * THIN_PER_ROW],
              "sigma2_int_null_mean": [null["sigma2_int"]["null_mean"], float(np.mean(chk))],
              "sigma2_int_null_q95": [null["sigma2_int"]["null_q95"], float(np.quantile(chk, 0.95))],
              "p_upper": [null["sigma2_int"]["p_upper"], p_upper(obs["sigma2_int"], chk)]}
    res = {
        "cell_id": cell_id(baseline, site), "n_tasks": len(tasks),
        "sr_by_mode_run": {m: [float(Y[:, j, 0].mean()), float(Y[:, j, 1].mean())]
                           for j, m in enumerate(MODE_KEYS)},
        "decomposition": dec, "decomposition_ci95": bootstrap(Y, SEED),
        "d_study": {"k_needed": k_needed(dec["interaction"], dec["noise"]),
                    "reliability_by_k": {str(k): reliability(dec["interaction"], dec["noise"], k)
                                         for k in (1, 2, 3, 5, 9, 20)}},
        "null": null, "mixing_check": mixing, "n_null": len(nd),
    }
    print(f"{res['cell_id']}: sigma2_int {obs['sigma2_int']:.4f} null mean {null['sigma2_int']['null_mean']:.4f} "
          f"p={null['sigma2_int']['p_upper']:.4f} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return res


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    items = sorted((p, k) for k, p in pvals.items())
    out, alive = {}, True
    for i, (p, k) in enumerate(items):
        alive = alive and p <= alpha / (len(items) - i)
        out[k] = alive
    return out


# ---------------------------------------------------------------------------- render

def _f(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def render(payload: dict) -> str:
    cells = payload["cells"]
    L = [
        "# Task×mode structure beyond difficulty — a no-interaction null on the fully replicated cells",
        "",
        "Generated by `scripts/analysis/task_mode_interaction_null.py`. `post_hoc_exploratory=True`. "
        "Chronicle: 实验笔记 §543. Supersedes the untracked §505.7 decomposition and the k/6 difficulty "
        "floor of `label_instability` as the test of the label-supply explanation.",
        "",
        "Null: logit P(success) = task difficulty + (mode, run) easiness, **no persistent task×mode "
        "interaction**. Its sufficient statistics are the row and column sums, so the null is sampled "
        f"exactly by curveball trades that keep every task's total and every (mode, run) column's "
        f"success count fixed ({N_CHAINS} chains, burn-in {BURN_PER_ROW}·n trades, {THIN_PER_ROW}·n "
        f"between kept draws, B={payload['n_null']}). Nothing is fitted; difficulty is conditioned on, "
        "not proxied. p = plus-one upper-tail estimator.",
        "",
        "## 1. Primary: does the task×mode interaction reproduce on a rerun?",
        "",
        "sigma²_int = (MS_int − MS_err)/2 from the task × mode ANOVA with two runs: the run-a/run-b "
        "covariance of the interaction residuals. Holm across the three cells.",
        "",
        "| cell | n | sigma²_int observed | null mean [q05, q95] | p | Holm | interaction / noise | null mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        s, r = c["null"]["sigma2_int"], c["null"]["int_over_noise"]
        L.append(f"| {c['cell_id']} | {c['n_tasks']} | {s['observed']:.4f} | {s['null_mean']:.4f} "
                 f"[{s['null_q05']:.4f}, {s['null_q95']:.4f}] | {s['p_upper']:.4f} "
                 f"| {'pass' if c['holm'] else '—'} | {_f(r['observed'], 2)} | {_f(r['null_mean'], 2)} |")
    L += ["", "Mixing check (same statistic, 4× the thinning, B/4 draws): null mean / q95 / p",
          ""]
    for c in cells:
        mc = c["mixing_check"]
        L.append(f"- {c['cell_id']}: thin {mc['thin_per_row'][0]}·n → {mc['sigma2_int_null_mean'][0]:.4f} / "
                 f"{mc['sigma2_int_null_q95'][0]:.4f} / p {mc['p_upper'][0]:.4f}; thin "
                 f"{mc['thin_per_row'][1]}·n → {mc['sigma2_int_null_mean'][1]:.4f} / "
                 f"{mc['sigma2_int_null_q95'][1]:.4f} / p {mc['p_upper'][1]:.4f}")
    L += ["", "## 2. Secondary: the label-level readings under the same null", "",
          "Contested = some but not all six modes solved the task on run a; flipped = any mode differs "
          "between the runs (the `label_instability` definitions). Stable strict preference = some mode "
          "2/2 and another 0/2 on the same task. Not multiplicity-corrected; read as descriptions of how "
          "far the data sit from a world without task×mode structure.",
          "",
          "| cell | statistic | observed | null mean [q05, q95] | p (upper) |", "|---|---|---|---|---|"]
    for c in cells:
        for key, lab, nd in (("contested_flip_rate", "contested flip rate", 3),
                             ("enrichment", "flip enrichment vs complement", 2),
                             ("n_stable_strict_preference", "tasks with a stable strict preference", 1)):
            s = c["null"][key]
            L.append(f"| {c['cell_id']} | {lab} | {_f(s['observed'], nd)} | {_f(s['null_mean'], nd)} "
                     f"[{_f(s['null_q05'], nd)}, {_f(s['null_q95'], nd)}] | {_f(s['p_upper'], 4)} |")
    L += ["", "## 3. Decomposition and D-study (descriptive)", "",
          f"Variance components of the 0/1 outcome, task-bootstrap 95% intervals (B={N_BOOT}). "
          "D-study: reliability of a task×mode label averaged over k runs = σ²_int / (σ²_int + σ²_noise/k); "
          "k needed = smallest k with reliability ≥ 0.5.",
          "",
          "| cell | task | mode | interaction | noise | interaction / noise | k needed |",
          "|---|---|---|---|---|---|---|"]
    for c in cells:
        d, ci = c["decomposition"], c["decomposition_ci95"]
        cell = [f"{d[k]:.4f} [{ci[k][0]:.4f}, {ci[k][1]:.4f}]" for k in ("task", "mode", "interaction", "noise")]
        r = d["interaction_over_noise"]
        kk = c["d_study"]["k_needed"]
        L.append(f"| {c['cell_id']} | " + " | ".join(cell)
                 + f" | {_f(r, 2)} [{ci['interaction_over_noise'][0]:.2f}, {ci['interaction_over_noise'][1]:.2f}]"
                 + f" | {'—' if kk is None else int(kk)} [{ci['k_needed'][0]:.0f}, "
                 + ("∞" if not np.isfinite(ci['k_needed'][1]) else f"{ci['k_needed'][1]:.0f}") + "] |")
    L += ["", "Reliability of a k-run-averaged task×mode label:", "",
          "| cell | " + " | ".join(f"k={k}" for k in cells[0]["d_study"]["reliability_by_k"]) + " |",
          "|---|" + "---|" * len(cells[0]["d_study"]["reliability_by_k"])]
    for c in cells:
        L.append(f"| {c['cell_id']} | " + " | ".join(f"{v:.2f}" for v in c["d_study"]["reliability_by_k"].values()) + " |")
    L += ["", "The null excludes persistent task×mode structure, not task difficulty: the budget line, which "
          "routes on difficulty, is outside what this tests.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--n-null", type=int, default=N_NULL, help="smoke runs only; outputs go to --out")
    ap.add_argument("--out", type=Path, default=OUT_MD)
    ap.add_argument("--json-out", type=Path, default=OUT_JSON)
    args = ap.parse_args()
    if args.n_null != N_NULL and (args.out == OUT_MD or args.json_out == OUT_JSON):
        raise SystemExit("a non-default --n-null must not overwrite the tracked product")
    cells = full_paired_cells()
    if not cells:
        raise RuntimeError("no fully replicated cell found in CLEAN_PAIRS")
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        results = list(ex.map(run_cell, cells, [args.n_null] * len(cells)))
    verdict = holm({r["cell_id"]: r["null"]["sigma2_int"]["p_upper"] for r in results})
    for r in results:
        r["holm"] = bool(verdict[r["cell_id"]])
    payload = {"post_hoc_exploratory": True,
               "producer": "scripts/analysis/task_mode_interaction_null.py",
               "protocol": {"null": "Rasch: logit p = task + (mode, run); sampled conditional on margins "
                                    "by curveball", "chains": N_CHAINS, "burn_per_row": BURN_PER_ROW,
                            "thin_per_row": THIN_PER_ROW, "seed": SEED, "n_boot": N_BOOT,
                            "modes": list(MODE_KEYS)},
               "n_null": args.n_null, "cells": results}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    args.out.write_text(render(payload), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
