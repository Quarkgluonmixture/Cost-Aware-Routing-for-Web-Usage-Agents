---
type: provenance
status: superseded
gate: H10
k: 5
supersedes: results/phantom_paper/l1_router_offline_20260715/h10_entropy_gate.json
superseded_by: docs/checkpoints/pre_run/h10_artifact_regen_provenance_2026-07-31.md
created: 2026-07-22
updated: 2026-07-31
---

# H10 canonical router-artifact regeneration — provenance witness (2026-07-22)

`results/phantom_paper/*` is gitignored (`.gitignore:35`), so the artifacts themselves
cannot carry a git witness. This note is the tracked witness: it records what was
regenerated, why, the resulting SHA256s, and the standing obligation to re-run at k=6.

## 1. Why — the canonical gate directory was never refreshed after Pass-1

`aggregate_h10_pareto.py:66` reads the entropy-DEFER artifact from
`ROUTER_ARTIFACT_DIR = results/phantom_paper/l1_router/`. That directory still held the
**2026-05-18 placeholder**:

```json
{"status": "no_data_yet", "n_total_tasks": 0,
 "message": "No pooled tasks available — Stage 2 cannot run without Pass-1 outcomes."}
```

Every real Stage-1/2/3 run since had been written to dated sibling directories
(`l1_router_offline_20260715/`, `l1_router_rehearsal_20260702/`), so
`l1_router/h10_entropy_gate.json` never existed and the verdict fell through its
fail-closed branch with `h10_status = "entropy_unavailable"`.

Consequence: H10 was being suppressed for a **plumbing** reason, while the paper prose
attributed the suppression to a **substantive** one ("two cells lack trainable labels").
The conclusion coincided; the stated mechanism did not.

## 2. What was run (k=5, five complete paper-grade cells)

```
extract_50_features.py     --output <dir>/raw_features_phase1a.npz \
                           --cells B0_classifieds B0_reddit B1_classifieds B1_reddit B2_classifieds
train_l1_router_with_mi.py --raw-features <dir>/raw_features_phase1a.npz --out-dir <dir>
train_l1_router.py         --all --out-dir <dir>
aggregate_h10_pareto.py    --all --allow-partial-dev
```

Producers were **run, not modified** — no gating producer source changed in this operation.
Pipeline first executed into a scratch directory, inspected, then promoted.

`--cells` is explicit and load-bearing: see §5.

## 3. Result — three cells have zero trainable folds, not two

| cell | train-fold label entropy | trainable folds |
|---|---|---|
| B0_classifieds | 2.161 | 5/5 |
| B1_classifieds | 2.097 | 4/5 (fold 4 `insufficient_train_data`) |
| B0_reddit | NaN | **0/5** |
| B1_reddit | NaN | **0/5** |
| B2_classifieds | NaN | **0/5** |

Stage 3 summary: **1/5 cells fully trained**, 4 incomplete, 0 failed.

The superseded 2026-07-15 artifact showed only **two** NaN cells because `B0_reddit` was
absent from its pool entirely — that cell's P-prompt condition landed 2026-07-16 01:35Z,
after the artifact was built. Adding it makes a third.

Independently reproduced: a separate derivation using the fail-closed loader
(`router_offline_replay.collect_cell_outcomes`, which enforces exactly six paper-grade
modes plus canonical task-universe SHA) returns identical per-cell label counts
(97 / 55 / 55 / 26 / 16) and the same three untrainable cells.

### 3.1 The gate fails for a different reason than the prose says

- `h10_entropy_gate_passed = true`, `h10_status = "ok"` — the **entropy DEFER does not
  fire**. Labels that exist are not concentrated (2.1–2.2 bits vs the 1.0-bit threshold).
- The operative blocker is `insufficient_train_data`: `N_MIN_CLASS_TRAIN = 10`
  (`train_l1_router.py:61`) drops any class with <10 training samples in a fold, then the
  `len(set(y)) < 2` guard rejects the fold. Against absolute label counts of 16–55 this
  is binding.
- Additionally, `Pass-2 runs: 0` on every cell — the learned-router conditions have never
  been fired, so `n_cells_with_data = 0` and `k_of_n_string = "0/0"` independently.

Paper prose describing the suppression should name **insufficient per-fold training data**
(and the absent Pass-2 runs), not label concentration.

## 4. Verdict diff

| field | before | after |
|---|---|---|
| `h10_status` | `entropy_unavailable` | `ok` |
| `entropy_defer_reason` | "entropy gate artifact … absent (fail-closed)" | `None` |
| `operational_gate_passed_pre_entropy` | `False` | `None` |
| `operational_gate_passed` | `False` | `False` (unchanged) |
| `k_of_n_string` | `0/0` | `0/0` (unchanged) |

The deployability conclusion is unchanged and, if anything, stronger: at the 5-of-6
grid criterion only one cell yields a complete model.

## 5. Hazard found while doing this — mode-completeness is not guarded

`extract_50_features.collect_per_task_outcomes` globs whatever episode summaries exist
under the discovered run dirs. It does **not** assert that all six modes are present, nor
that each mode covers the full scored task universe. `derive_oracle_label` then reads
`outcomes.get(m, False)` — so a **mode with no data is silently scored as a failure**.

Run without `--cells`, Stage 1 pulled `B2_reddit` into the pool and emitted 16 oracle
labels for it, derived from 4 complete modes + one 36%-complete mode
(`B2_phantom_som_reddit_20260722`, 74/205) + one mode that has not started
(`phantom_prompt`). No warning was printed.

The same file guards this exact semantics one level down
(`extract_50_features.py:66-78`, P1-9: "a MISSING success field must NOT be coerced to
False — that fabricates a failure outcome and corrupts the oracle label"). The guard
exists at the **episode** level and is missing at the **mode** level. Compare
`router_offline_replay.load_paper_grade_entries`, which raises on any missing or extra
mode.

**Required before the k=6 regeneration**: add a mode-completeness + task-universe
assertion to the Stage-1 path, or the k=6 re-run risks silently ingesting a partial cell.

## 6. Standing obligation — re-run at k=6

`B2_reddit` is mid-collection (`phantom_som` 74/205 as of 2026-07-22; `phantom_prompt`
not started). When it lands and is manifest-bound, this entire pipeline must be re-run
with all six cells and this note superseded. The k=5 artifacts below are **not** the
submission-final state.

## 7. SHA256 (first 16 hex) at promote time

Repo `HEAD` = `81d8eb8`. Promoted 2026-07-22; verdict `captured_at` 2026-07-22T17:31:02Z.

| artifact | sha256[:16] |
|---|---|
| `l1_router/h10_entropy_gate.json` | `471dc316a1a4ce60` |
| `l1_router/stage2_summary.json` | `bb6f1732b1d10905` |
| `l1_router/stage3_summary.json` | `a712180c2be2e073` |
| `l1_router/raw_features_phase1a.npz` | `5cba0306f5428700` |
| `h10_pareto_verdict.json` | `e4d0bff270f9ad11` |

Pre-promote state preserved at `results/phantom_paper/_l1_router_backup_pre_regen_20260722/`
(12 files) and `results/phantom_paper/_backup_pre_regen_20260722_h10_pareto_verdict.{json,md,csv}`.
Deprecated single-pickle heads (`*_lr.pkl`, 2026-05-16, superseded by the fold-aware
bundle per B-1640) were removed from the canonical directory and exist only in that backup.
