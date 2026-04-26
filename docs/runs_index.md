# Run Directory Index

Index of `results/{benchmark}/{phase}/` directories — what each run is, current
status, and notable caveats. The `results/` tree is git-ignored, so this
markdown is the canonical record.

> **Update protocol**: when a new run dir is created or status changes, add a
> row here. When a run is rederived (§97 audit fix), note the rederive batch.

---

## VWA Phase 1

### B0 (Qwen3-VL-235B via proxy)

| Run dir | Site | Status | Episodes | Notes |
|---|---|---|---|---|
| `B0_3mode_classifieds_20260413` | classifieds | ✅ canonical | 234 × 3 modes = 702 | §97 rederived (PUR -12pp); has `adjusted_success` field |
| `B0_3mode_reddit_20260422` | reddit | ✅ canonical | 210 × 3 modes = 630 | §97 rederived (PUR -9pp) |
| `B0_3mode_shopping_20260421` | shopping | ⚠ partial | 466 × 1 mode (DOM only) | §97 rederived; SoM/Vision pending |
| `B0_3mode` | — | gallery agg | — | Symlinks aggregated for combined gallery |

### B1 (Qwen3-VL-4B local)

| Run dir | Site | Status | Episodes | Notes |
|---|---|---|---|---|
| `B1_3mode_classifieds_20260413` | classifieds | ✅ done | 234 × 3 modes | Pre-§97 schema; needs rederive |
| `B1_3mode_reddit_20260413` | reddit | 🟢 in-progress | 210 DOM/210 Vision/126→210 SoM | SoM rerun with max_marks=200 (§94); watch script auto-rederives on completion |
| `B1_3mode_shopping_20260413` | shopping | 🟢 queued | 466 DOM done; SoM 0/466 (clean); Vision 0/466 | Will start auto after reddit SoM finishes (watch script step 4) |
| `B1_3mode` | — | gallery agg | — | Symlinked for combined gallery |

---

## VWA Phase 2 (not started)

Pending: Phase 2 router experiments. See 实验笔记 §15/§24 for design.

---

## VWA Phase 3 (not started)

Pending: Module ablation (M1-M4). See `p79/experiment/conditions.py:164-187`.

---

## WebArena Phase 1

| Run dir | Site | Status | Episodes | Notes |
|---|---|---|---|---|
| `B0_wa_3mode_shopping_20260417` | shopping | ⏸ no episodes | 0 | Initial setup, no successful runs yet |
| `B0_wa_3mode_shopping_20260423` | shopping | ⏸ no episodes | 0 | Same |
| `B0_wa_3mode` | — | gallery placeholder | — | — |

---

## Cross-site / Cross-model aggregations

| Output dir | Source | Notes |
|---|---|---|
| `results/visualwebarena/phase1/cross_site_analysis/` | `aggregate_cross_site.py` | After all 3 sites done |
| `results/visualwebarena/phase1/b0_vs_b1/` | `compare_b0_b1.py` | Per-site comparison |

---

## Audit / Rederive history

| Date | Audit | Affected runs | Effect |
|---|---|---|---|
| 2026-04-26 | §97 batch 1-3 (~80 bugs) | B0 classifieds/reddit/shopping(DOM) rederived | PUR -10pp; adjusted_success field added |
| 2026-04-26 | §97 Step-2 (single source of truth) | All B0 rederived again | episode_summary now carries `adjusted_success` natively |

---

## Background processes (current)

Use `make schedule-list` to see live PIDs.

| Purpose | PID (latest) | Script |
|---|---|---|
| B1 reddit SoM runner | 1916623 | `run_experiment.py` |
| B1 reddit watchdog | 1906908 | `experiment_watchdog.py` |
| Auto rederive + shopping launch | 2220458 | `wait_for_reddit_then_rederive.sh` |
