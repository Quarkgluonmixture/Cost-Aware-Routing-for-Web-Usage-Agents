---
type: provenance
status: active
gate: H10
k: 6
supersedes: docs/checkpoints/pre_run/h10_artifact_regen_provenance_2026-07-22.md
created: 2026-07-31
updated: 2026-07-31
---

# H10 canonical router-artifact regeneration — provenance witness (2026-07-31, k=6)

Discharges the standing obligation in
[[h10_artifact_regen_provenance_2026-07-22]] §6 ("when B2_reddit lands and is
manifest-bound, this entire pipeline must be re-run with all six cells and this
note superseded. The k=5 artifacts are **not** the submission-final state").

`results/phantom_paper/*` is gitignored (`.gitignore:35`), so the artifacts cannot
carry a git witness. This note is the tracked witness.

## 1. Why now

`B2_reddit` landed and bound: 36/36 Pass-1 conditions complete (2026-07-29,
`audit_steps_summary_identity`). The k=5 verdict was declared void on landing by
PROTOCOL_NOTE_06 ("unconditional k=6 regeneration if that cell lands and binds
before submission").

**Precondition satisfied first.** §5 of the superseded note required a
mode-completeness assertion on the Stage-1 path *before* any k=6 re-run, because
`collect_per_task_outcomes` globs whatever episode summaries exist and
`derive_oracle_label` scores an absent mode as a failure. Verified present before
running: commit `554cc7c` ("fix(analysis): B-1887 guard mode completeness before
oracle-label derivation", +156 lines in `extract_50_features.py`), which adds
`assess_mode_completeness()` at :153 — it reports absent modes, partial modes, and
whether the cell covers the canonical scored set. **No completeness warning fired
during this run**, i.e. all six cells are mode-complete.

## 2. What was run

Producers were **run, not modified** — no gating producer source changed. Executed
into a scratch directory (`results/phantom_paper/l1_router_k6_20260731/`),
inspected, then promoted.

```
extract_50_features.py     --output <scratch>/raw_features_phase1a.npz \
                           --cells B0_classifieds B0_reddit B1_classifieds \
                                   B1_reddit B2_classifieds B2_reddit
train_l1_router_with_mi.py --raw-features <scratch>/raw_features_phase1a.npz \
                           --out-dir <scratch>
train_l1_router.py         --all --out-dir <scratch>
# ── promote ──
aggregate_h10_pareto.py    --all --allow-partial-dev
```

`--cells` remains explicit and load-bearing for the reason given in the superseded
note §5. The only change from the k=5 invocation is the added `B2_reddit`.

**Cross-check on Stage 1**: pooled total = **260 tasks**, byte-identical to the
independently-derived total in `router_label_supply_diagnosis.md` (97 + 53 + 55 +
24 + 16 + 15 = 260). Two different code paths, same number.

## 3. Result — a sixth untrainable cell, conclusion unchanged

| cell | train-fold label entropy | trainable folds |
|---|---|---|
| B0_classifieds | 2.1608 | 5/5 |
| B1_classifieds | 2.0971 | 4/5 (fold 4 `insufficient_train_data`) |
| B0_reddit | NaN | **0/5** |
| B1_reddit | NaN | **0/5** |
| B2_classifieds | NaN | **0/5** |
| **B2_reddit** | **NaN** | **0/5** ← new |

Stage 3 summary: **1/6 cells fully trained**, 5 incomplete, 0 failed
(k=5 was 1/5 fully trained, 4 incomplete, 0 failed).

`B2_reddit` behaves exactly as `router_label_supply_diagnosis` predicted from its
15 trainable labels: no class survives `N_MIN_CLASS_TRAIN = 10` in a 5-fold split,
so the `len(set(y)) < 2` guard rejects every fold.

### 3.1 The gate still fails for a different reason than the prose says

Unchanged from k=5, now confirmed at full scope:

- `h10_entropy_gate_passed = true`, `h10_status = "ok"`, `global_entropy_min_bits
  = 2.0971` (threshold 1.0) — **the entropy DEFER does not fire**. Labels that
  exist are not concentrated.
- The operative blocker is `insufficient_train_data`, and independently
  `Pass-2 runs: 0` on every one of the six cells → `n_cells_with_data = 0`.

Paper prose describing the suppression must name **insufficient per-fold training
data** (and the absent Pass-2 runs), not label concentration.

## 4. Verdict diff — every gate field is unchanged

| field | k=5 (2026-07-27 capture) | k=6 |
|---|---|---|
| `h10_status` | `ok` | `ok` |
| `k_of_n_string` | `0/0` | `0/0` |
| `k_cells_passing_cell_level` | 0 | 0 |
| `n_cells_with_data` | 0 | 0 |
| `operational_gate_passed` | `False` | `False` |
| `h10_entropy_gate` (nested) | 5-cell | **6-cell** ← only change |
| `captured_at` | 2026-07-27T12:48:32Z | 2026-07-31T17:12:57Z |

This confirms 笔记 §383.1's prediction that B2_reddit "只多一行, 论点一字不改".

**What this regeneration actually fixed was a provenance inconsistency, not a
number.** The 2026-07-27 verdict already enumerated six cells under `per_cell`,
while the `h10_entropy_gate` it embedded was still the 2026-07-22 five-cell
artifact — the verdict described a k=6 pool through a k=5 gate. The two now agree.

## 5. Found while promoting — a stale partial-cell artifact was still canonical

`results/phantom_paper/l1_router/` contained `B2_reddit_fold_assignment.json` and
`B2_reddit_lr_meta.json` **before** this run, even though the k=5 promote used an
explicit 5-cell `--cells` list that cannot emit them. They are residue of the
unguarded run described in the superseded note §5 — the one that "pulled
`B2_reddit` into the pool and emitted 16 oracle labels for it, derived from 4
complete modes + one 36%-complete mode + one mode that has not started. No warning
was printed."

Because the k=5 promote never wrote those two filenames, it never overwrote them
either: the canonical directory has been a **mixed-vintage** state since
2026-07-22 — five clean cells plus one partial-collection cell. Nothing consumed
them (the gate JSON is what `aggregate_h10_pareto.py:66` reads, and that was
5-cell), so no published number is affected. This k=6 promote overwrites both with
data derived from the complete cell.

Pre-overwrite `B2_reddit_lr_meta.json` sha256[:16] = `0d71e217521e4582`, preserved
in the backup below.

**Generalizable**: promoting with a *narrower* `--cells` list than a previous run
leaves the extra cells' files behind. Promote is a file-level copy, not a
directory sync — it cannot delete what it does not produce.

## 6. SHA256 (first 16 hex) at promote time

Repo `HEAD` = `be40557`. Promoted 2026-07-31; verdict `captured_at`
2026-07-31T17:12:57Z.

| artifact | sha256[:16] |
|---|---|
| `l1_router/h10_entropy_gate.json` | `ed74784f620a1a0a` |
| `l1_router/stage2_summary.json` | `a1c05f6ec8649dc7` |
| `l1_router/stage3_summary.json` | `46412f3445ae2765` |
| `l1_router/raw_features_phase1a.npz` | `63ea1cfddee58642` |
| `h10_pareto_verdict.json` | `d2f690b6d4ef510f` |

Pre-promote state preserved at
`results/phantom_paper/_l1_router_backup_pre_regen_20260731/` (38 files) and
`results/phantom_paper/_backup_pre_regen_20260731_h10_pareto_verdict.json`.
Scratch retained at `results/phantom_paper/l1_router_k6_20260731/` (36 files).

## 7. Remaining obligation

Pass-2 (learned-router) conditions have still never been fired — `Pass-2 runs: 0`
on all six cells, which is why `k_of_n_string` is `0/0` independently of
trainability. H10 cannot move off `0/0` until Pass-2 data exists; that is a data
gap, not an artifact-staleness gap, and this note does not discharge it.
