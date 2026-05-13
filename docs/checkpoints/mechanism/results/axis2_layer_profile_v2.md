# Exp 1 — Axis-2 (prompt-family) layer profile

**Question**: Method 4.2 at L17 shows prompt-family makes ~0 geometric contribution to residual stream
(P-SoM↔P-text 0.0028, DOM↔P-prompt 0.0013). But forest plot drop-one places P-SoM as unique hero,
implying axis-2 (prompt) contributes behaviorally. **Where in the model does axis-2 act?**

**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.
Overlay axis-1-only (text swap, prompt fixed) + image-axis P-SoM↔SoM reference curves to calibrate scale.

## Results — classifieds site (stage4_multimode_b1_cls, 288 ex)

| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
|---|---|---:|---:|---:|---:|---:|---:|
| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0013 | 0.0068 | **L36** | 0.0068 |
| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0002 | 0.0019 | 0.0088 | **L36** | 0.0088 |
| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0035 | 0.0021 | 0.0047 | **L36** | 0.0047 |
| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0034 | 0.0017 | 0.0048 | **L36** | 0.0048 |
| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0375 | 0.0386 | 0.0416 | **L36** | 0.0416 |

## Results — reddit site (stage4_multimode_b1_reddit, 288 ex)

| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
|---|---|---:|---:|---:|---:|---:|---:|
| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0015 | 0.0063 | **L36** | 0.0063 |
| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0002 | 0.0020 | 0.0069 | **L36** | 0.0069 |
| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0030 | 0.0019 | 0.0037 | **L36** | 0.0037 |
| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0027 | 0.0016 | 0.0042 | **L36** | 0.0042 |
| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0386 | 0.0367 | 0.0316 | **L4** | 0.0386 |

## Interpretation

Three hypotheses about axis-2 mechanism layer:

1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.
2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.
3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.

Compare peak layers above against axis-1 (text-format) pairs (the established mechanism with L17 peak) and image-axis reference (~0.04 magnitude). If axis-2 pair peak < 0.01 at all layers, hypothesis 1 holds.
