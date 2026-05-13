# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)

Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.
For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement
across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets
amplified into output distribution divergence by late-layer decoding.

## Classifieds site

### Axis-2 (prompt-family) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L25** | 0.0879 | 0.0134 | 0.0520 | 0.0000 |
| DOM vs P-prompt  (axis-2 hierarchical) | **L21** | 0.0459 | 0.0026 | 0.0240 | 0.0000 |

### Axis-1 (text-format) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L3** | 0.0425 | 0.0164 | 0.0096 | 0.0000 |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L3** | 0.0393 | 0.0167 | 0.0242 | 0.0000 |

## Reddit site

### Axis-2 (prompt-family) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L25** | 0.0574 | 0.0192 | 0.0391 | 0.0000 |
| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0488 | 0.0106 | 0.0415 | 0.0000 |

### Axis-1 (text-format) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L2** | 0.0330 | 0.0126 | 0.0322 | 0.0000 |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L3** | 0.0391 | 0.0136 | 0.0214 | 0.0000 |

## Interpretation

Three hypotheses tested:

- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family
  effect bypasses logit lens, only visible via attention heads or runtime decoding.
- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →
  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling
  'knows but says differently' mirror).
- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →
  prompt prior signal proportional to mid-layer geometry, no amplification.

Cross-site replication should hold for any of the three. Compare axis-2 KL magnitudes to
axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.
