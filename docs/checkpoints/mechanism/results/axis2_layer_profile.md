# Exp 1 — Axis-2 (prompt-family) layer profile — per-task paired (v2 NPZ)

**P0-7 + P1-6 fix (2026-05-13)**: per-task paired cosine gap per layer with
task-level bootstrap 95% CI (1000 resamples). Previous version computed cosine
of pooled mode-means — mixed task-content variance into 'layer profile' claim.
Now uses (task_id, step) inner-join via `_paired_npz_helpers.paired_rows` then
averages per-task cosine gap. CI is from resampling tasks (NOT (task,step) rows)
with replacement, preserving within-task step paired structure.

**Question**: Method 4.2 at L17 shows prompt-family makes ~0 geometric contribution to residual stream.
But forest plot drop-one places P-SoM as unique hero, implying axis-2 (prompt) contributes
behaviorally. **Where in the model does axis-2 act?**

**Method**: For each prompt-only pair (text format fixed, prompt swap), compute paired
per-task cosine gap across 37 layers. Overlay axis-1-only (text swap, prompt fixed) +
image-axis P-SoM↔SoM reference curves to calibrate scale.

## Results — classifieds site (stage4_multimode_b1_cls, 24 paired rows across 24 unique tasks)

| Pair | Group | L0 | L4 | L17 [CI] | L36 | Peak L | Peak gap [95% CI] | n_paired |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0005 | 0.0028 [0.0026, 0.0031] | 0.0099 | **L36** | 0.0099 [0.0091, 0.0106] | 24 |
| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0005 | 0.0035 [0.0032, 0.0039] | 0.0121 | **L36** | 0.0121 [0.0113, 0.0130] | 24 |
| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0043 | 0.0037 [0.0034, 0.0040] | 0.0082 | **L36** | 0.0082 [0.0075, 0.0088] | 24 |
| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0041 | 0.0031 [0.0029, 0.0032] | 0.0079 | **L36** | 0.0079 [0.0074, 0.0084] | 24 |
| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0383 | 0.0406 [0.0397, 0.0417] | 0.0492 | **L36** | 0.0492 [0.0462, 0.0520] | 24 |

## Results — reddit site (stage4_multimode_b1_reddit, 24 paired rows across 24 unique tasks)

| Pair | Group | L0 | L4 | L17 [CI] | L36 | Peak L | Peak gap [95% CI] | n_paired |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0006 | 0.0028 [0.0027, 0.0030] | 0.0098 | **L36** | 0.0098 [0.0091, 0.0105] | 24 |
| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0037 [0.0035, 0.0039] | 0.0109 | **L36** | 0.0109 [0.0100, 0.0117] | 24 |
| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0040 | 0.0035 [0.0032, 0.0038] | 0.0082 | **L36** | 0.0082 [0.0075, 0.0088] | 24 |
| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0036 | 0.0032 [0.0030, 0.0035] | 0.0080 | **L36** | 0.0080 [0.0073, 0.0087] | 24 |
| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0402 | 0.0404 [0.0394, 0.0413] | 0.0408 | **L36** | 0.0408 [0.0368, 0.0454] | 24 |

## Interpretation

Three hypotheses about axis-2 mechanism layer:

1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.
2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.
3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.

Compare peak layers above against axis-1 (text-format) pairs and image-axis reference (~0.04 magnitude).
If axis-2 pair peak CI overlaps 0, hypothesis 1 holds; if CI lower-bound > 0.005, hypothesis 2 or 3.
