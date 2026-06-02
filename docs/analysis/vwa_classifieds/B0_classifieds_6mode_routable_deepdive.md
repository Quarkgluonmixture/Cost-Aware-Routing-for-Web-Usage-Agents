# Cross-mode Routable Deep-dive — B0 classifieds

> Modes: dom, som, vision, phantom_text, phantom_som, phantom_prompt | common N=224 | image-modes=['som', 'vision'] text-modes=['dom', 'phantom_text', 'phantom_som', 'phantom_prompt'] | deterministic (no sub-agent)
> ⚠️ **PROVISIONAL** — single (model,site) run. Oracle MAX is one-sided upward-biased by the B0 serving floor (§308 13.3% / §302 vision 14.3%), which is the
> **same order as the +16pp lift**. Magnitudes = UPPER BOUNDS, need replicate-calibration (§293/§306/§309). Feature→class DIRECTION is more noise-robust than oracle MAGNITUDE. NOT a gate.

## A. Routable decomposition — must-route (k=1) vs route-forgiving (k≥2)

| solve-count k | N tasks | meaning |
|---:|---:|---|
| 0 | 127 | universal-fail (routing can't help) |
| 1 | 29 | **exclusive — must pick THE mode** |
| 2 | 25 | shared by 2 modes — route-forgiving |
| 3 | 12 | shared by 3 modes — route-forgiving |
| 4 | 9 | shared by 4 modes — route-forgiving |
| 5 | 13 | shared by 5 modes — route-forgiving |
| 6 | 9 | universal-solve (routing free) |

- **routable = 88** (39%) = exclusive **29** (k=1, routing MUST be correct) + shared **59** (k≥2, any of several modes works).
- 6-mode oracle = 97/224 = 43.3%; best-single = som 27.2% → **oracle lift = +16.1pp**.
- The lift is carried by the **29 exclusive + the shared tasks best-single fails**. Exclusive tasks are the noise-fragile core (§D).

## B. Oracle marginal — where the +16pp portfolio value lives

Greedy add modes by SR (desc); marginal = NET-NEW tasks no higher-SR mode solved.
A mode's marginal = its **irreplaceable** routing contribution (drop it → oracle falls).

| order | mode | SR | marginal NEW | cumulative oracle |
|---|---|---:|---:|---:|
| 1 | som | 27.2% | +61 | 61 (27.2%) |
| 2 | vision | 25.0% | +15 | 76 (33.9%) |
| 3 | phantom_prompt | 19.6% | +13 | 89 (39.7%) |
| 4 | dom | 17.4% | +4 | 93 (41.5%) |
| 5 | phantom_text | 15.6% | +2 | 95 (42.4%) |
| 6 | phantom_som | 15.6% | +2 | 97 (43.3%) |

Leave-one-out (order-independent): tasks ONLY this mode solves (= exclusive) — drop it and oracle loses exactly these:

| mode | exclusive (LOO oracle loss) |
|---|---:|
| dom | 4 |
| som | 6 |
| vision | 9 |
| phantom_text | 2 |
| phantom_som | 2 |
| phantom_prompt | 6 |

## C. Router feature candidates — task-intrinsic feature × mode-CLASS capability

For each feature value, over ALL tasks with that value: image-class SR (any of {som,vision}) vs text-class SR (any of {dom,phantom_*}); **img_only** = image-class solves AND text-class ALL fail (= must route to pixels); **txt_only** = text-class solves AND image-class all fail (= pixels not needed / hurt).

### feature: `visual`

| value | N | img-class SR | txt-class SR | img_only | txt_only | both | neither |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 66 | 36% | 35% | 10 | 9 | 14 | 33 |
| medium | 77 | 32% | 27% | 10 | 6 | 15 | 46 |
| hard | 81 | 33% | 25% | 13 | 6 | 14 | 48 |

### feature: `eval`

| value | N | img-class SR | txt-class SR | img_only | txt_only | both | neither |
|---|---:|---:|---:|---:|---:|---:|---:|
| url_match | 131 | 34% | 32% | 16 | 13 | 29 | 73 |
| string_match | 62 | 42% | 26% | 15 | 5 | 11 | 31 |
| program_html | 31 | 16% | 19% | 2 | 3 | 3 | 23 |

### feature: `has_image`

| value | N | img-class SR | txt-class SR | img_only | txt_only | both | neither |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 159 | 29% | 19% | 29 | 14 | 17 | 99 |
| True | 65 | 46% | 51% | 4 | 7 | 26 | 28 |

### feature: `overall`

| value | N | img-class SR | txt-class SR | img_only | txt_only | both | neither |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 39 | 59% | 38% | 12 | 4 | 11 | 12 |
| medium | 84 | 36% | 33% | 11 | 9 | 19 | 45 |
| hard | 101 | 23% | 21% | 10 | 8 | 13 | 70 |

- **img_only tasks** (N=33, pixels REQUIRED): [49, 83, 90, 97, 100, 106, 110, 111, 112, 118, 123, 124, 125, 127, 130, 131, 148, 160, 163, 165, 166, 171, 173, 179, 186, 187, 188, 192, 193, 203, 209, 221, 233]
- **txt_only tasks** (N=21, pixels NOT needed): [1, 2, 4, 12, 23, 40, 56, 64, 68, 76, 93, 105, 116, 117, 132, 137, 142, 146, 183, 201, 217]

## D. Noise sensitivity — which numbers survive the ~13-14% serving floor

| quantity | value | noise exposure |
|---|---:|---|
| best-single SR (som) | 27.2% | LOW — single mode, noise ~symmetric (some true-solves flip out, some true-fails flip in) → ≈unbiased |
| 6-mode oracle SR | 43.3% | HIGH — MAX over 6 modes is **one-sided**; picks up every positive flip, ignores negatives → UPWARD biased |
| oracle lift | +16.1pp | HIGH — = oracle − best-single, inherits the one-sided oracle bias |
| exclusive-solves (k=1) | 29 | HIGHEST — one fail→pass flip on a universal-fail task fabricates a spurious exclusive; §302 decomposed 14/224 ≈ 6% fail→pass per replicate, near-boundary |
| shared-solves (k≥2) | 59 | MEDIUM — needs ≥2 modes to agree, harder to fake by independent flips |
| img_only / txt_only counts | 33 / 21 | MEDIUM — class-level (any-of-2 / any-of-4) absorbs single-mode flips better than per-mode exclusive |

> **Defensible separation**: the feature→class *direction* (which feature predicts image-class vs text-class advantage) reflects systematic representation differences and survives symmetric noise scatter; the oracle *magnitude* (+16pp) is a one-sided UPPER BOUND. Router-feature claims travel; the headline lift needs a replicate (§293 replicate-calibrated MC) before it is paper-grade.

