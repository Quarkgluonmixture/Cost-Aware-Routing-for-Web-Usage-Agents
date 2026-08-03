# Mechanism evidence (frozen §5)

Frozen §5 evidence, read from canonical results/mechanistic JSON rather than from the prose write-ups in docs/checkpoints/mechanism/results/. Mechanism work was shelved 2026-05-14; this product exists so the shelved results are citable and distinguishable from results that were never obtained.

## Linear readability vs geometric magnitude (Method 4.2 v2)

| site | modes | examples | pairs | pairs at AUROC 1.000 | worst pair | image gap | text-format gap | prompt-family gap |
|---|---|---|---|---|---|---|---|---|
| classifieds | 6 | 144 | 15 | 15/15 | 1.000 | 0.0416 (L36) | 0.0047 (L36) | 0.0088 (L36) |
| reddit | 6 | 144 | 15 | 15/15 | 1.000 | 0.0386 (L04) | 0.0037 (L36) | 0.0069 (L36) |

## Causal patching with both controls (Stage 3, prompt-family axis)

`displacement` = 1 − overlap with the unpatched target (how far the output moved). 
`convergence` = overlap with the source continuation (whether it moved *toward the source*). Displacement alone cannot separate steering from destruction.

| site | arm | n | peak displacement | disp layer | peak convergence | conv layer | displacement at L23 |
|---|---|---|---|---|---|---|---|
| classifieds | real | 24 | 0.230 | L13 | 0.188 | L14 | 0.040 |
| classifieds | random_injection | 24 | 0.992 | L27 | 0.085 | L02 | 0.982 |
| classifieds | task_shuffled | 24 | 0.296 | L17 | 0.157 | L00 | 0.058 |
| reddit | real | 24 | 0.338 | L16 | 0.188 | L09 | 0.107 |
| reddit | random_injection | 24 | 0.992 | L20 | 0.100 | L03 | 0.915 |
| reddit | task_shuffled | 24 | 0.312 | L14 | 0.159 | L20 | 0.158 |
