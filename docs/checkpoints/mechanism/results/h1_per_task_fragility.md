# H1 per-task fragility check

**Sample**: 45 (task, step) pairs from format_variation_b1_cls

## Aggregate verdict per individual (task, step) pair

- **AXTree-DOM peak ≤ L10** (early image-axis peak): 9/45 = **20%**
- **≥4/7 marks-like variants peak ≥ L20** (late image-axis peak): 39/45 = **87%**
- **BOTH conditions** (strict dichotomy per task): 5/45 = **11%**

## Per-task peak-layer distribution

AXTree-DOM peak layer: mean = **27.9**, std = 13.1, range L04-L36
Marks-like (avg across 7) peak layer: mean = **31.9**, std = 8.0
**Separation** = marks - dom = **+4.0 layers**

## Verdict

→ **H1 WEAK per-task**: dichotomy is averaged effect, not per-task universal. Paper §5 framing must acknowledge per-task variability.

## Top 5 dichotomy-confirming (task, step) pairs (largest separation)

| Task ID | Step | AXTree peak | Marks avg peak | Separation |
|---|---|---|---|---|
| 214 | 5 | L04 | L36.0 | **+32.0** |
| 228 | 2 | L04 | L36.0 | **+32.0** |
| 32 | 5 | L04 | L31.4 | **+27.4** |
| 228 | 5 | L04 | L29.4 | **+25.4** |
| 9 | 2 | L04 | L24.6 | **+20.6** |

## Bottom 5 (task, step) pairs (smallest / inverse separation)

| Task ID | Step | AXTree peak | Marks avg peak | Separation |
|---|---|---|---|---|
| 61 | 5 | L17 | L16.4 | -0.6 |
| 20 | 2 | L36 | L33.3 | -2.7 |
| 122 | 2 | L36 | L33.3 | -2.7 |
| 60 | 5 | L17 | L11.6 | -5.4 |
| 37 | 2 | L36 | L28.0 | -8.0 |
