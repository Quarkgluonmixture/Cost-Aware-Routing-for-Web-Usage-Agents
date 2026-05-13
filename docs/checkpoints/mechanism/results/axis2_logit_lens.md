# Exp 3 — Logit lens at late layers (axis-2 vs axis-1) — per-task paired (v2 NPZ)

P0-1 fix (2026-05-13): now computes BOTH per-task paired KL (paper-grade) AND
KL-of-decoded-mode-means (legacy proxy) with ratio per pair. Per-task KL is the
paper-grade quantity; of-means reported for monotonicity check only.

P0-8 fix (2026-05-13): lm_head + norm now loaded in fp32 (not bf16). bf16 mantissa
= 7 bits quantizes 4th-decimal of KL between similar distributions to noise. fp32
preserves sub-permille precision needed for the cosine-causal disjoint claim.

**Interpretation of ratio per-task / of-means**:
- ratio ≈ 1 → 'amplification' framing terminology-fix-able (KL-of-means is defensible proxy)
- ratio ≫ 1 (>2×) → per-task signal MUCH stronger; paper UNDERSTATES mechanism
- ratio ≪ 1 (<0.5×) → KL-of-means inflates; 'amplification' hero claim collapses

## Classifieds site

### Axis-2 (prompt-family) pairs

| Pair | Peak L per-task | Peak KL per-task ± std | Peak L of-means | Peak KL of-means | Ratio @ per-task peak | KL @ L17 (per-task) | KL @ L23 (per-task) | n_paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L25** | 0.0975 ± 0.0361 | L25 | 0.0883 | **1.10×** | 0.0334 | 0.0695 | 24 |
| DOM vs P-prompt  (axis-2 hierarchical) | **L22** | 0.0549 ± 0.0044 | L25 | 0.0434 | **1.51×** | 0.0274 | 0.0495 | 24 |

### Axis-1 (text-format) pairs

| Pair | Peak L per-task | Peak KL per-task ± std | Peak L of-means | Peak KL of-means | Ratio @ per-task peak | KL @ L17 (per-task) | KL @ L23 (per-task) | n_paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L3** | 0.0546 ± 0.0097 | L3 | 0.0465 | **1.18×** | 0.0352 | 0.0401 | 24 |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L3** | 0.0512 ± 0.0092 | L3 | 0.0440 | **1.16×** | 0.0287 | 0.0439 | 24 |

## Reddit site

### Axis-2 (prompt-family) pairs

| Pair | Peak L per-task | Peak KL per-task ± std | Peak L of-means | Peak KL of-means | Ratio @ per-task peak | KL @ L17 (per-task) | KL @ L23 (per-task) | n_paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L22** | 0.0718 ± 0.0098 | L25 | 0.0583 | **1.59×** | 0.0353 | 0.0695 | 24 |
| DOM vs P-prompt  (axis-2 hierarchical) | **L23** | 0.0587 ± 0.0161 | L25 | 0.0453 | **1.82×** | 0.0263 | 0.0587 | 24 |

### Axis-1 (text-format) pairs

| Pair | Peak L per-task | Peak KL per-task ± std | Peak L of-means | Peak KL of-means | Ratio @ per-task peak | KL @ L17 (per-task) | KL @ L23 (per-task) | n_paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.0919 ± 0.1409 | L3 | 0.0381 | **3.95×** | 0.0320 | 0.0919 | 24 |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.0825 ± 0.1170 | L3 | 0.0356 | **3.25×** | 0.0298 | 0.0825 | 24 |

## Interpretation

Three hypotheses tested:

- **H_A (axis-2 absent from output)**: axis-2 per-task KL flat <0.1 at all layers → prompt-family
  effect bypasses logit lens, only visible via attention heads or runtime decoding.
- **H_B (axis-2 amplified at output)**: axis-2 per-task KL peak at L21-L25 ≫ cosine 0.005-0.009 magnitude →
  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling
  'knows but says differently' mirror).
- **H_C (axis-2 tracks residual stream)**: axis-2 per-task KL peak at L36 same as cosine peak →
  prompt prior signal proportional to mid-layer geometry, no amplification.

Cross-site replication should hold for any of the three.
