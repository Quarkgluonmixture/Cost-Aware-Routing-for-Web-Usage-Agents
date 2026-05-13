# Exp 3 v3 — Per-task logit lens KL vs KL-of-means (defuse 2026-05-13)

**Audit target** (`/stress v5 OOB attack 1, 2026-05-13`): `stage4_logit_lens_axis2.py:114`
computes `KL(decode(mean_h_A), decode(mean_h_B))` — KL of decoded-averaged-means.
Plan.md §1.2 hero claim 'amplification 8-44×' depends on this matching the paper's
implied per-task amplification `E_task[KL(decode(h_A_t), decode(h_B_t))]`.
By Jensen's inequality + non-linearity of softmax, these can differ.

This script computes BOTH on the same v2 NPZ and reports the ratio.

**Interpretation**:
- ratio ≈ 1 → 'amplification' framing terminology-fix-able (KL-of-means is a defensible proxy)
- ratio ≫ 1 (>2×) → per-task signal is MUCH stronger; paper UNDERSTATES mechanism
- ratio ≪ 1 (<0.5×) → KL-of-means inflates; 'amplification' hero claim collapses

## Classifieds (cls)

### Axis-2 (prompt-family)

| Pair | Peak L per-task | Peak KL per-task ± std | Peak L of-means | Peak KL of-means | Ratio @ per-task peak |
|---|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L25** | 0.0980 ± 0.0367 | L25 | 0.0879 | **1.12×** |
| DOM vs P-prompt  (axis-2 hierarchical) | **L22** | 0.0553 ± 0.0070 | L21 | 0.0459 | **1.34×** |

### Axis-1 (text-format)

| Pair | Peak L per-task | Peak KL per-task ± std | Peak L of-means | Peak KL of-means | Ratio @ per-task peak |
|---|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L3** | 0.0547 ± 0.0138 | L3 | 0.0425 | **1.29×** |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L3** | 0.0509 ± 0.0123 | L3 | 0.0393 | **1.29×** |

## Reddit (red)

### Axis-2 (prompt-family)

| Pair | Peak L per-task | Peak KL per-task ± std | Peak L of-means | Peak KL of-means | Ratio @ per-task peak |
|---|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L22** | 0.0734 ± 0.0093 | L25 | 0.0574 | **1.88×** |
| DOM vs P-prompt  (axis-2 hierarchical) | **L23** | 0.0600 ± 0.0164 | L25 | 0.0488 | **1.45×** |

### Axis-1 (text-format)

| Pair | Peak L per-task | Peak KL per-task ± std | Peak L of-means | Peak KL of-means | Ratio @ per-task peak |
|---|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.0919 ± 0.1421 | L2 | 0.0330 | **2.85×** |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.0828 ± 0.1164 | L3 | 0.0391 | **3.87×** |

## Verdict logic

- All 8 ratios (2 sites × 4 pairs) within [0.5, 2.0] → terminology-only fix
- Any ratio > 2 → mechanism stronger than reported, paper UNDERSTATES
- Any ratio < 0.5 → 'amplification' hero claim REJECTED, §1.2 rewrite required
