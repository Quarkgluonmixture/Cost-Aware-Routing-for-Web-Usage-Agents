# Exp 1 — Axis-2 (prompt-family) layer profile

**Question**: Method 4.2 at L17 shows prompt-family makes ~0 geometric contribution to residual stream
(P-SoM↔P-text 0.0028, DOM↔P-prompt 0.0013). But forest plot drop-one places P-SoM as unique hero,
implying axis-2 (prompt) contributes behaviorally. **Where in the model does axis-2 act?**

**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.
Overlay axis-1-only (text swap, prompt fixed) + image-axis P-SoM↔SoM reference curves to calibrate scale.

## Results — classifieds site (stage4_multimode_b1_cls, 288 ex)

| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
|---|---|---:|---:|---:|---:|---:|---:|
| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0013 | 0.0067 | **L36** | 0.0067 |
| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0028 | 0.0089 | **L23** | 0.0114 |
| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0134 | 0.0120 | 0.0201 | **L23** | 0.0254 |
| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0127 | 0.0113 | 0.0201 | **L23** | 0.0292 |
| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0394 | 0.0412 | 0.0411 | **L17** | 0.0412 |

## Results — reddit site (stage4_multimode_b1_reddit, 288 ex)

| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
|---|---|---:|---:|---:|---:|---:|---:|
| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0012 | 0.0059 | **L36** | 0.0059 |
| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0027 | 0.0080 | **L23** | 0.0098 |
| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0125 | 0.0092 | 0.0183 | **L23** | 0.0217 |
| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0115 | 0.0086 | 0.0176 | **L23** | 0.0240 |
| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0434 | 0.0423 | 0.0434 | **L4** | 0.0434 |

## Interpretation

Three hypotheses about axis-2 mechanism layer:

1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.
2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.
3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.

Compare peak layers above against axis-1 (text-format) pairs (the established mechanism with L17 peak) and image-axis reference (~0.04 magnitude). If axis-2 pair peak < 0.01 at all layers, hypothesis 1 holds.
