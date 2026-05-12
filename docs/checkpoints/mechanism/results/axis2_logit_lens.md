# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)

Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.
For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement
across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets
amplified into output distribution divergence by late-layer decoding.

## Classifieds site

### Axis-2 (prompt-family) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L23** | 0.1621 | 0.0215 | 0.1621 | 0.0003 |
| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0444 | 0.0184 | 0.0234 | 0.0000 |

### Axis-1 (text-format) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5508 | 0.1299 | 0.5508 | 0.0001 |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6953 | 0.1069 | 0.6953 | 0.0003 |

## Reddit site

### Axis-2 (prompt-family) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L24** | 0.1260 | 0.0371 | 0.1230 | 0.0002 |
| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0508 | 0.0228 | 0.0325 | 0.0000 |

### Axis-1 (text-format) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5273 | 0.0898 | 0.5273 | 0.0000 |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6172 | 0.0806 | 0.6172 | 0.0002 |

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
