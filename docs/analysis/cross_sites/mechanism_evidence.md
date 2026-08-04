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

## The 2026-08 sweep (24 cells, DGX) — grouped by axis

Finished 2026-08-03 and read by nothing until 2026-08-04. Random injection is the destruction control: it maximises displacement while converging on nothing, which is why displacement alone proves no steering.

| axis | cell | src to tgt | n | peak disp | L | peak conv | L |
|---|---|---|---|---|---|---|---|
| control:random_injection | `p5_psom_ptext_rand_cls` | phantom_som to phantom_text | 24 | 0.995 | L33 | 0.067 | L02 |
| control:random_injection | `p5_psom_ptext_rand_red` | phantom_som to phantom_text | 24 | 0.998 | L31 | 0.083 | L02 |
| control:random_injection | `p5_rand_cls` | som to phantom_som | 24 | 0.992 | L32 | 0.068 | L05 |
| control:random_injection | `p5_rand_red` | som to phantom_som | 24 | 0.993 | L17 | 0.088 | L03 |
| image | `p1_fwd_strong_cls` | som to phantom_som | 24 | 0.274 | L12 | 0.175 | L02 |
| image | `p1_fwd_strong_red` | som to phantom_som | 24 | 0.247 | L16 | 0.225 | L02 |
| image | `p1_rev_reverse_cls` | phantom_som to som (rev) | 15 | 0.312 | L16 | 0.255 | L14 |
| image | `p1_rev_reverse_red` | phantom_som to som (rev) | 15 | 0.475 | L16 | 0.275 | L17 |
| image | `p3_fwd_revtier_cls` | som to phantom_som | 15 | 0.251 | L16 | 0.303 | L16 |
| image | `p3_fwd_revtier_red` | som to phantom_som | 15 | 0.303 | L14 | 0.188 | L17 |
| image | `p3_rev_strongtier_cls` | phantom_som to som (rev) | 24 | 0.208 | L16 | 0.196 | L18 |
| image | `p3_rev_strongtier_red` | phantom_som to som (rev) | 24 | 0.311 | L16 | 0.198 | L26 |
| image+one-text-axis | `p4_som_pprompt_cls` | som to phantom_prompt | 24 | 0.333 | L14 | 0.223 | L13 |
| image+one-text-axis | `p4_som_pprompt_red` | som to phantom_prompt | 24 | 0.293 | L11 | 0.247 | L12 |
| image+one-text-axis | `p4_som_ptext_cls` | som to phantom_text | 24 | 0.271 | L14 | 0.172 | L30 |
| image+one-text-axis | `p4_som_ptext_red` | som to phantom_text | 24 | 0.300 | L15 | 0.217 | L13 |
| image+text-format+prompt | `p4_som_dom_cls` | som to dom | 24 | 0.475 | L15 | 0.233 | L11 |
| image+text-format+prompt | `p4_som_dom_red` | som to dom | 24 | 0.390 | L16 | 0.225 | L23 |
| prompt-style | `p2_psom_ptext_cls` | phantom_som to phantom_text | 24 | 0.271 | L14 | 0.172 | L30 |
| prompt-style | `p2_psom_ptext_red` | phantom_som to phantom_text | 24 | 0.300 | L15 | 0.217 | L13 |
| prompt-style | `p2_taskshuf_cls` | phantom_som to phantom_text | 24 | 0.275 | L15 | 0.156 | L35 |
| prompt-style | `p2_taskshuf_red` | phantom_som to phantom_text | 24 | 0.292 | L14 | 0.163 | L11 |
| prompt-style | `p5_psom_ptext_rev_cls` | phantom_text to phantom_som (rev) | 24 | 0.342 | L13 | 0.228 | L33 |
| prompt-style | `p5_psom_ptext_rev_red` | phantom_text to phantom_som (rev) | 24 | 0.514 | L15 | 0.223 | L17 |

## Same configuration, run twice (2026-05 Myriad vs 2026-08 DGX)

Six arms re-ran under a field-identical config. **The convergence peak layer moved in 5 of 6.** This matters more than the value movement: the content-specific reading of the patching result is argued from the peak LAYER (real mid-stack, shuffled at the boundary), so a peak layer that is not reproducible cannot carry that argument by itself. The same rerun discipline this paper applies to success rates applies here.

| site | arm | peak conv 05 to 08 | conv layer 05 to 08 | moved? | tied layers 05/08 | curve range 05/08 |
|---|---|---|---|---|---|---|
| classifieds | real | 0.188 to 0.172 | L14 to L30 | **yes** | 1/6 | 0.023/0.014 |
| reddit | real | 0.188 to 0.217 | L09 to L13 | **yes** | 1/1 | 0.012/0.038 |
| classifieds | task_shuffled | 0.157 to 0.156 | L00 to L35 | **yes** | 1/7 | 0.007/0.007 |
| reddit | task_shuffled | 0.159 to 0.163 | L20 to L11 | **yes** | 2/1 | 0.008/0.006 |
| classifieds | random_injection | 0.085 to 0.067 | L02 to L02 | no | 1/1 | 0.078/0.059 |
| reddit | random_injection | 0.100 to 0.083 | L03 to L02 | **yes** | 1/1 | 0.093/0.076 |
