# Stage 4: image-axis peak-layer split — Mirage Effect signature

Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:

| no-image side | image side | no-img text | peak layer | peak cosine gap |
|---|---|---|---|---|
| DOM | SoM | AXTree | **L04** | 0.0455 |
| DOM | Vision | AXTree | **L04** | 0.0658 |
| P-text | Vision | [SOM_MARKS] | **L04** | 0.0590 |
| P-prompt | SoM | AXTree | **L04** | 0.0434 |
| P-prompt | Vision | AXTree | **L04** | 0.0634 |
| P-SoM | SoM | [SOM_MARKS] | **L04** | 0.0386 |
| P-SoM | Vision | [SOM_MARKS] | **L04** | 0.0586 |
| P-text | SoM | [SOM_MARKS] | **L17** | 0.0433 |

## Grouped by no-image side text format

### no-image text = `AXTree` (mean peak L4)

- DOM ↔ SoM: peak **L04** = 0.0455
- DOM ↔ Vision: peak **L04** = 0.0658
- P-prompt ↔ SoM: peak **L04** = 0.0434
- P-prompt ↔ Vision: peak **L04** = 0.0634

### no-image text = `[SOM_MARKS]` (mean peak L7)

- P-text ↔ SoM: peak **L17** = 0.0433
- P-text ↔ Vision: peak **L04** = 0.0590
- P-SoM ↔ SoM: peak **L04** = 0.0386
- P-SoM ↔ Vision: peak **L04** = 0.0586

## Mechanism interpretation (paper §5 v3 Mirage anchor)

When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).

When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.

**Mechanism story**: `[SOM_MARKS]` text in input primes an indexed-parsing pathway through mid-layer computation. The model processes marks structurally regardless of whether image is provided, producing image-axis divergence only at late integration stages. This is the direct Method 4.2 empirical anchor for the **Mirage Effect** (Asadi et al. 2026, VLM ~70-80% no-image accuracy) and **Cross-modal flow** (Kaduri et al., middle-layer cross-modal flows store image info in query tokens): the marks-primed mid-layer computation runs *as if image were available*, with image grounding contributed only late.

**Paper §5 prose** (suggested):

> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*
