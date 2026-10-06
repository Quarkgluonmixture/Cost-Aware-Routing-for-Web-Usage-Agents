---
type: analysis
status: complete
created: 2026-09-06
purpose: does the cross-SIDE unique-coverage difference survive the replicate-assignment envelope on more than one cell?
producer: scripts/analysis/unique_solve_noise_envelope.py --compare
---

# Cross-side unique coverage, under the 2^6 assignment envelope

Regenerate: `.venv/bin/python3 scripts/analysis/unique_solve_noise_envelope.py --compare`

Each arm may be drawn from either of its two same-condition runs, so every
cell below is 2^6 = 64 assignments. The number that matters is each arm's
**minimum** unique-solve count over those 64: it is what the arm contributes
that no other arm does, in the least favourable assignment. A lower bound that
can be driven to 0 means the arm has no assignment-robust unique contribution.

## Per-arm lower bound (min over 64 assignments)

| arm | side | cls_b0 (n=224) | red_b0 (n=203) | wared_b1 (n=104) |
|---|---|---|---|---|
| `SoM` | visual | **6**–12 | **4**–8 | **1**–2 |
| `Vision` | visual | **6**–11 | **2**–5 | **0**–2 |
| `P-text` | text | **0**–3 | **0**–6 | **1**–4 |
| `P-SoM` | text | **0**–4 | **0**–8 | **0**–2 |
| `P-prompt` | text | **1**–6 | **2**–4 | **2**–5 |
| `DOM` | text (AXTree, not in either side group) | **0**–6 | **0**–5 | **1**–4 |

## The comparison the hero rests on

| cell | visual side, lowest bound | text side, highest bound | separation |
|---|---|---|---|
| cls_b0 (classifieds) | 6 (`SoM`) | 1 (`P-prompt`) | **+5** |
| red_b0 (reddit) | 2 (`Vision`) | 2 (`P-prompt`) | **+0** |
| wared_b1 (WA-reddit) | 0 (`Vision`) | 2 (`P-prompt`) | **-2** |

## Reading

- **cls_b0**: the two sides are separated by 5 — every visual arm keeps a unique contribution that no assignment of the text arms reaches.
- **red_b0**: the sides **touch** at 2. The visual side's weakest arm and the text side's strongest arm have the same lower bound, so on this cell 'the visual side contributes more uniquely' is **not** supported arm-by-arm — it holds only for the stronger visual arm.
- **wared_b1**: **inverted** (-2). A text arm has a higher assignment-robust unique contribution than the weakest visual arm.

⚠️ **Scope.** 3 cells (cls_b0, red_b0, wared_b1; backbone B0, B1). A cell needs all six arms replicated to appear here. The B1 cell is a different backbone on a different benchmark (WebArena) and is locally served; read it beside the B0 cells, not pooled with them. Nothing here licenses a statement about B2.
