### One-line verdict

**Needs replacement-of-framing.** δ=1.0pp is defensible, but only as an H1 superiority threshold; K_h1/K_h3 should not be framed as 75%/67% pass gates under N=4.

### K_h1

**Archive evidence**  
P-SoM drop-one archive cells are all positive: B0 cls +2.56pp, SE 0.981; B0 red +3.33pp, SE 1.096; B1 cls +1.71pp, SE 0.766. Pooled DL RE = **+2.34pp [1.30, 3.37]**, SE 0.529, **I²=0.0%**, τ²=0, Holm pass. LOO also stays significant: +2.33, +2.03, +2.91 depending on dropped cell.

Power file says per-cell MDE is **5.5-7.8pp** for cls/red, while observed phantom effects are mostly **1-5pp**; proxy per-cell power is 0.13 at 2pp, 0.23 at 3pp, 0.53 at 5pp. So a 3/4 K rule is underpowered for the actual phenomenon scale.

**Recommended value**  
**No inferential threshold. Report count only.** Suggested rows: `positive cells / 4`, `bootstrap CI > 0 cells / 4`, `Holm-significant cells / 4`.

If the advisor wants a descriptive benchmark, use **3/4 = “strong per-cell consistency”**, not K_h1=0.75 as a decision rule.

**Framing fix**  
Replace “75% hero claim pass ratio” with: “H1 is gated by pooled random-effects meta + one-sided δ-superiority; per-cell count is transparency only.” At N=4, 75% just means 3/4, so the percentage adds fake precision.

### K_h3

**Archive evidence**  
The structural-arm proxy evidence is weaker and more heterogeneous than P-SoM:

P-text drop-in: pooled **+2.44pp [0.32, 4.56]**, SE 1.081, **I²=71.0%**, τ²=2.455. Per-cell: +3.42, +3.81, +0.85pp. LOO is fragile: dropping B0 cls or B0 red flips Holm to non-significant.

P-prompt drop-in: pooled **+1.88pp [0.38, 3.37]**, SE 0.763, **I²=28.3%**, k=2 only. Per-cell: +2.86 and +1.28pp. LOO is fragile when B0 red is dropped.

Axis report supports non-collapse mechanistically: P-SoM differs from both DOM and SoM on 10 metric/site combinations, and the cascade reveals 6 antagonistic axis pairs. But this is not the same as the preregistered H3 unique-task count test. The archive files provided do **not** contain enough per-cell `P-text \ P-SoM` and `P-prompt \ P-SoM` unique-count data to calibrate a K_h3 threshold directly.

**Recommended value**  
**No K_h3 threshold.** Keep the **≥2 unique tasks per cell** noise-floor rule, but report per-axis counts descriptively: `axis positive / 4`, `axis CI > 0 / 4`, `unique-count floor met / 4`.

If forced to keep a display benchmark, use **3/4 descriptive only**, but do not justify it as “67% weaker than hero.” At N=4, 0.67 and 0.75 both become 3/4.

**Framing fix**  
Replace “K_h3=67% structural pass ratio” with: “H3 is gated by pooled per-axis meta subject to heterogeneity rules; K is a transparency replication count only.” Also flag P-text I²=71% as near the preregistered high-heterogeneity danger zone.

### δ

**Archive evidence**  
For P-SoM, pooled θ_RE = **+2.34pp**, SE 0.529, CI **[1.30, 3.37]**. Against δ=1.0pp superiority: z = (2.34−1.0)/0.529 = **2.53**, one-sided p ≈ **0.0057**. It clears δ=1.0pp with real but not huge headroom: the two-sided CI lower bound is only **0.30pp above** δ.

LOO against δ=1.0pp is borderline but still pass-like under one-sided α=0.05: dropping B0 cls gives p≈0.046; dropping B0 red p≈0.044; dropping B1 cls p≈0.0045.

**Recommended value**  
Keep **δ = 1.0pp**.

δ=0.5pp would be easier to clear but weakens the substantive claim; under superiority it is not “too hard,” it is too permissive. δ=2.0pp would fail the archive: z≈0.64, p≈0.26, and LOO would be worse. So 1.0pp is the best archive-grounded compromise.

**Framing fix**  
δ is no longer a “cost equivalence margin” for H1. It is the **substantive superiority threshold** for drop-one lift: H0 θ ≤ +1.0pp. Cost equivalence is H2(a), currently ±10% median cost, not SR pp.

### P0-Level Concerns

The advisor question should not mainly be “0.75 vs 0.67?” That distinction collapsed when N became 4.

The real methodology decision is: **is classic DerSimonian-Laird + z-test acceptable at k=4, or should the prereg lock require REML/Hartung-Knapp or at least a sensitivity decision rule?** The archive itself warns τ² is unstable with k<5, and the untested fourth cell is exactly **B1 reddit**, so current calibration is based on only 3 archive cells.

Second, H3 is much shakier than H1: P-text has **I²=71%** and LOO fragility. If the structural claim depends on pooled H3, the heterogeneity branch needs to be explicit before lock.

Also clean stale wording: followup.md still says 16-cell, gate, and TOST/cost-equivalence; prereg still has some router-family TOST/K_h1 language if H7/H8 stays in paper-1 scope.