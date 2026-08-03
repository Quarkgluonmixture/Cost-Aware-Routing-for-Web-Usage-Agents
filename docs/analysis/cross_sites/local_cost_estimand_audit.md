---
type: analysis
status: complete
purpose: what the locally-served dollar figure measures, and whether its per-mode ordering survives a change of basis
post_hoc_exploratory: true
producer: scripts/analysis/local_cost_estimand_audit.py
---

# The locally-served dollar figure

Regenerate: `.venv/bin/python3 scripts/analysis/local_cost_estimand_audit.py`

`EVIDENCE_LAYER_SUMMARY` §4a said the local per-token constant "was derived for a different accelerator" and that "within-cell ratios are unaffected because it is a single multiplier". The first half is right and now has a citation; **the second half is wrong**.

## 1. Where the constant came from, and where the runs ran

`configs/exp_v2_base.yaml:66-81`, verbatim: *"DGX Spark GB10: $4,699 / 3yr + 140W × $0.12/kWh ≈ $0.20/hr"* and *"DGX Spark GB10 ~$0.20/hr, ~60 tok/s → ~$0.00093/1k tokens"*. Every paper-grade condition was served on **`a100-jiaming-test`** (36/36 by `env_snapshot.json`).

The same file's **energy** block *was* migrated — `hardware_profile: "a100_pcie_40gb"`, commented *"B-118 (2026-05-15): canonical paper-grade rerun host = A100 Condenser VM; dgx_spark profile retired per A100 migration"*. The **cost** block sits four lines above it and still says DGX Spark. One half of the hardware assumption was migrated and the other was not, in the same file.

A second inconsistency inside the same pipeline: the cost derivation assumes **140 W**; the energy telemetry records **66.3 W** (`energy_carbon_audit`). Two power figures, one pipeline.

## 2. The assumed throughput is off by 4–9×

The derivation assumes **60 tok/s**. Measured throughput on the actual host is **248–551 tok/s** (4.1–9.2× the assumption), **and it varies by mode** — which is exactly why a fixed price-per-token cannot preserve an ordering over modes whose steps differ in token density.

| cell | mode | tokens/step | model ms/step | measured tok/s | $ by token | $ by GPU-time |
|---|---|---|---|---|---|---|
| `B1·classifieds` | DOM | 2,886 | 7,882 | 366 | 0.002773 | 0.000438 |
| `B1·classifieds` | SoM | 3,699 | 8,994 | 411 | 0.003542 | 0.000500 |
| `B1·classifieds` | Vision | 2,254 | 7,658 | 294 | 0.002180 ⬅ | 0.000425 ⬅ |
| `B1·classifieds` | P-text | 2,759 | 8,040 | 343 | 0.002656 | 0.000447 |
| `B1·classifieds` | P-prompt | 3,182 | 8,275 | 385 | 0.003051 | 0.000460 |
| `B1·classifieds` | P-SoM | 3,076 | 9,186 | 335 | 0.002967 | 0.000510 |
| `B1·reddit` | DOM | 3,628 | 7,867 | 461 | 0.003467 | 0.000437 |
| `B1·reddit` | SoM | 4,088 | 8,809 | 464 | 0.003907 | 0.000489 |
| `B1·reddit` | Vision | 2,742 | 7,713 | 356 | 0.002639 ⬅ | 0.000428 |
| `B1·reddit` | P-text | 3,238 | 7,381 | 439 | 0.003101 | 0.000410 ⬅ |
| `B1·reddit` | P-prompt | 3,742 | 8,555 | 437 | 0.003589 | 0.000475 |
| `B1·reddit` | P-SoM | 3,472 | 9,076 | 383 | 0.003349 | 0.000504 |
| `B2·classifieds` | DOM | 2,853 | 10,443 | 273 | 0.002754 | 0.000580 |
| `B2·classifieds` | SoM | 3,808 | 10,402 | 366 | 0.003639 | 0.000578 |
| `B2·classifieds` | Vision | 2,584 | 10,418 | 248 | 0.002500 ⬅ | 0.000579 |
| `B2·classifieds` | P-text | 2,744 | 10,169 | 270 | 0.002651 | 0.000565 |
| `B2·classifieds` | P-prompt | 3,219 | 10,035 | 321 | 0.003099 | 0.000557 |
| `B2·classifieds` | P-SoM | 3,028 | 9,351 | 324 | 0.002913 | 0.000519 ⬅ |
| `B2·reddit` | DOM | 3,967 | 9,233 | 430 | 0.003783 | 0.000513 |
| `B2·reddit` | SoM | 4,880 | 8,849 | 551 | 0.004624 | 0.000492 |
| `B2·reddit` | Vision | 2,972 | 8,539 | 348 | 0.002854 ⬅ | 0.000474 ⬅ |
| `B2·reddit` | P-text | 3,882 | 9,225 | 421 | 0.003703 | 0.000513 |
| `B2·reddit` | P-prompt | 3,857 | 8,794 | 439 | 0.003679 | 0.000489 |
| `B2·reddit` | P-SoM | 3,841 | 8,663 | 443 | 0.003662 | 0.000481 |

⬅ marks the cheapest mode under each basis. `$ by GPU-time` prices the same episodes at the same $0.20/hr the token constant was derived from, applied to measured model time instead of to a token count.

## 3. The ordering does not survive the change of basis

**The cheapest mode changes in 2 of 4 local cells** (`B1·reddit`, `B2·classifieds`) between the token basis and the GPU-time basis. That is a *within-cell* reordering, which is what §4a said could not happen.

Which basis is right depends on the claim. A local deployment rents or owns the accelerator by the second, so GPU-time is the deployment-facing quantity; the token basis is a proxy for it that the config's own comment derives *from* it via an assumed throughput. Neither is reported here as correct — the point is that the per-mode cost ordering on locally-served backbones is **estimand-dependent**, the same conclusion `latency_decomposition` reached for latency and `outcome_efficiency` reached for the denominator.

## 4. One worry that turns out to be harmless

The constant is really two (input 0.00093, output 0.00185), so a reader will ask whether the output:input price ratio drives the ordering. It does not: output tokens are **1.9–4.1%** of the total, and sweeping the ratio from 2× to 10× reorders **0 of 4** cells. §4a's "effectively a single multiplier" reasoning is therefore right on this point — just not sufficient for its conclusion, because the token basis itself is the problem.

## 5. Scope

**B0 is unaffected.** It pays a real API bill at published per-token rates, so its dollars are dollars. This page is about B1 and B2 only, where a "cost" is a modelling choice rather than an invoice — and where, per `cost_per_mode`, those figures were already flagged as belonging to a different class from B0's and never combined into one ratio.
