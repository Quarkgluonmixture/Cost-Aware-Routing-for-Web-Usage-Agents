# Hero-claim bootstrap CI (W1 defuse)

Per-seed bootstrap 95% percentile CI on paired adjusted-SR diffs and drop-one oracle. B=10000, seed=42. Tasks resampled with replacement at task level.

**Defuse target**: /stress W1 attack — paper §1 hero claim 'P-SoM 13.81% > SoM 10.48% reddit' is statistically marginal under author's own 2σ hedge.

## reddit (N=210 same-task)

**Per-mode adjusted SR (%)**:

- dom: 9.52%
- som: 10.48%
- vision: 6.67%
- phantom_som: 13.81%
- phantom_text: 12.38%
- phantom_prompt: 9.52%

**Pairwise SR difference, bootstrap 95% CI:**

| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |
|---|---:|---:|---:|---:|---:|
| P-SoM vs SoM | +3.33 | +3.33 | [-0.95, +7.62] | 0.914 | 0.828 | 
|  | | | ✗ crosses 0 | | |
| P-SoM vs DOM | +4.29 | +4.29 | [+0.00, +8.57] | 0.963 | 0.914 | 
|  | | | ✗ crosses 0 | | |
| P-text vs DOM | +2.86 | +2.86 | [-0.95, +6.67] | 0.918 | 0.810 | 
|  | | | ✗ crosses 0 | | |
| P-SoM vs P-text | +1.43 | +1.43 | [-1.90, +5.24] | 0.739 | 0.548 | 
|  | | | ✗ crosses 0 | | |

**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**

| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |
|---|---:|---:|---:|---:|---:|
| dom | +1.90 | +1.90 | [+0.48, +3.81] | 0.981 | 0.767 | 
|  | | | ✓ strict-pos | | |
| som | +1.90 | +1.90 | [+0.48, +3.81] | 0.980 | 0.762 | 
|  | | | ✓ strict-pos | | |
| vision | +1.43 | +1.43 | [+0.00, +3.33] | 0.949 | 0.574 | 
|  | | | ✗ crosses 0 | | |
| phantom_som | +3.33 | +3.33 | [+0.95, +6.19] | 0.998 | 0.969 | 
|  | | | ✓ strict-pos | | |

## classifieds (N=234 same-task)

**Per-mode adjusted SR (%)**:

- dom: 14.10%
- som: 21.37%
- vision: 13.68%
- phantom_som: 14.53%
- phantom_text: 14.53%

**Pairwise SR difference, bootstrap 95% CI:**

| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |
|---|---:|---:|---:|---:|---:|
| P-SoM vs SoM | -6.84 | -6.84 | [-12.39, -1.28] | 0.005 | 0.001 | 
|  | | | ✗ strict-neg | | |
| P-SoM vs DOM | +0.43 | +0.43 | [-3.42, +4.70] | 0.538 | 0.374 | 
|  | | | ✗ crosses 0 | | |
| P-text vs DOM | +0.43 | +0.43 | [-3.42, +4.27] | 0.546 | 0.376 | 
|  | | | ✗ crosses 0 | | |
| P-SoM vs P-text | +0.00 | +0.00 | [-4.27, +4.27] | 0.464 | 0.317 | 
|  | | | ✗ crosses 0 | | |

**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**

| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |
|---|---:|---:|---:|---:|---:|
| dom | +2.14 | +2.14 | [+0.43, +4.27] | 0.993 | 0.877 | 
|  | | | ✓ strict-pos | | |
| som | +8.55 | +8.55 | [+5.13, +12.39] | 1.000 | 1.000 | 
|  | | | ✓ strict-pos | | |
| vision | +3.42 | +3.42 | [+1.28, +5.98] | 1.000 | 0.988 | 
|  | | | ✓ strict-pos | | |
| phantom_som | +2.56 | +2.56 | [+0.85, +4.70] | 0.999 | 0.943 | 
|  | | | ✓ strict-pos | | |

## Verdict on /stress W1

Read the **reddit P-SoM vs SoM** row + **reddit drop-one P-SoM** row:

- If both CIs are strict-positive (ci_lo > 0) AND P(diff > 0) > 0.95 → **W1 attack defused**,   §1 hero claim is bootstrap-supported. Remove the '2σ hedge' from line 5, lead with the magnitude.
- If CIs cross zero but P(diff > 0) > 0.80 → **W1 partially defused**, the claim is directional
  but not strictly statistically significant. §1 hero must downgrade to 'competitive within 2σ' as
  the author already wrote, but the complementarity (Jaccard / drop-one positive on N=7 tasks) carries
  the structural weight.
- If P(diff > 0) < 0.80 → **W1 sustained**, §1 hero claim must rewrite to 'parity / complementarity
  rather than dominance'. The single-mode comparison is unsupported.
