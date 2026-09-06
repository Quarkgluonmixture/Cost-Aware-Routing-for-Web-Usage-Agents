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

| arm | side | cls_b0 (n=224) | red_b0 (n=205) |
|---|---|---|---|
| `SoM` | visual | **6**–12 | **4**–8 |
| `Vision` | visual | **6**–11 | **2**–6 |
| `P-text` | text | **0**–3 | **0**–6 |
| `P-SoM` | text | **0**–4 | **0**–8 |
| `P-prompt` | text | **1**–6 | **2**–5 |
| `DOM` | text (AXTree, not in either side group) | **0**–6 | **0**–5 |

## The comparison the hero rests on

| cell | visual side, lowest bound | text side, highest bound | separation |
|---|---|---|---|
| cls_b0 (classifieds) | 6 (`SoM`) | 1 (`P-prompt`) | **+5** |
| red_b0 (reddit) | 2 (`Vision`) | 2 (`P-prompt`) | **+0** |

## Reading

- **cls_b0**: the two sides are separated by 5 — every visual arm keeps a unique contribution that no assignment of the text arms reaches.
- **red_b0**: the sides **touch** at 2. The visual side's weakest arm and the text side's strongest arm have the same lower bound, so on this cell 'the visual side contributes more uniquely' is **not** supported arm-by-arm — it holds only for the stronger visual arm.

⚠️ **Scope.** Both cells are B0. A cell needs all six arms replicated to appear here, and only B0 has that on two sites. Nothing here licenses a statement about B1 or B2, whose floors are a different size entirely (see `serving_mode_floor.md`).
