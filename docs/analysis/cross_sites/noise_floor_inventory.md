---
type: analysis
status: complete
created: 2026-08-01
purpose: every measured run-to-run noise floor, next to the arm-count-matched oracle-ceiling gain it must be judged against
scope_warning: every number carries its own scope; do NOT do arithmetic across rows (§302 category-error retraction, §300.2 cross-GPU drift). The only comparisons drawn are within one (model, site) cell at equal arm count.
producer: scripts/analysis/aggregate_noise_floor_inventory.py
---

# Noise-floor inventory

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_noise_floor_inventory.py`

## 1. Measured same-condition run-to-run floors

`self_drop(a→b) = |{a solves} ∖ {b solves}| / n`. Two runs of ONE `(model, site, mode)`; the pair is a rerun, so this is the ceiling gain buyable by adding a **rerun** as one extra arm.

| pair | scope | n | self_drop a→b | self_drop b→a | discordance |
|---|---|---|---|---|---|
| `B0.cls.dom` | B0 x classifieds, canonical n=224 | 224 | **7.14pp** | **4.91pp** | 12.05% |
| `B0.cls.vision` | B0 x classifieds, canonical n=224 | 224 | **7.59pp** | **6.70pp** | 14.29% |
| `B0.cls.som` | B0 x classifieds, canonical n=224 | 224 | **5.36pp** | **7.59pp** | 12.95% |
| `B0.cls.ptext` | B0 x classifieds, canonical n=224 | 224 | **5.80pp** | **4.46pp** | 10.27% |
| `B0.cls.pprompt` | B0 x classifieds, canonical n=224 | 224 | **7.59pp** | **4.91pp** | 12.50% |
| `B0.cls.psom` | B0 x classifieds, canonical n=224 | 224 | **6.70pp** | **5.36pp** | 12.05% |
| `B1.cls.vision` | B1 x classifieds, canonical n=224 | 224 | **0.00pp** | **0.00pp** | 0.00% |
| `B1.cls.som` | B1 x classifieds, canonical n=224 | 224 | **0.00pp** | **0.00pp** | 0.00% |
| `B1.cls.dom` | B1 x classifieds, canonical n=224 | 224 | **1.34pp** | **1.79pp** | 3.12% |
| `B1.wa-red` (**new**) | B1 x WA-reddit, registered 10-task pilot draw x 5 modes | 50 | **2.00pp** | **4.00pp** | 6.00% |

### 1b. The mean-difference floor is two draws, not a bound

The set-difference functional above is the one claim 1 needs. Claims 3 and 4 compare **mean** success rates between two modes, and the matched floor for that is `|SR(a) − SR(b)|` on the same replicate pairs — which is where the band `0.89–2.23pp` comes from. Those two numbers are **one observation each of a random quantity**, and the quantity's own spread is computable from the discordant counts already in the table above. Under the exchangeability null (same condition, so each discordant task flips either way with probability ½) `D = (2X − d)/n` with `X ~ Binom(d, ½)`, so `SD(D) = √d / n`:

| pair | n | discordant d | observed \|ΔSR\| | **SD(ΔSR) under the null** | one-sided 95% | two-sided 95% |
|---|---|---|---|---|---|---|
| `B0.cls.dom` | 224 | 27 | 2.23pp | **2.32pp** | 3.82pp | ±4.55pp |
| `B0.cls.vision` | 224 | 32 | 0.89pp | **2.53pp** | 4.15pp | ±4.95pp |
| `B0.cls.som` | 224 | 29 | 2.23pp | **2.40pp** | 3.95pp | ±4.71pp |
| `B0.cls.ptext` | 224 | 23 | 1.34pp | **2.14pp** | 3.52pp | ±4.20pp |
| `B0.cls.pprompt` | 224 | 28 | 2.68pp | **2.36pp** | 3.89pp | ±4.63pp |
| `B0.cls.psom` | 224 | 27 | 1.34pp | **2.32pp** | 3.82pp | ±4.55pp |
| `B1.cls.vision` | 224 | 0 | 0.00pp | **0.00pp** | 0.00pp | ±0.00pp |
| `B1.cls.som` | 224 | 0 | 0.00pp | **0.00pp** | 0.00pp | ±0.00pp |
| `B1.cls.dom` | 224 | 7 | 0.45pp | **1.18pp** | 1.94pp | ±2.32pp |

⚠️ **The band's upper edge (2.68pp) is of the same order as one standard deviation (0.00–2.53pp).** So "clears the band" is not "clears the noise": an effect has to reach roughly **0.00–4.15pp** before a single rerun would be unlikely to produce it by itself. Both readings are reported because they answer different questions — *what did repetition actually deliver* (the two draws) versus *what could repetition deliver* (the null spread). Reading a 2.2pp effect against a 2.23pp "measured floor" is comparing a draw to a draw.

🚫 **Scope of that threshold — it is NOT a general significance bar** (/stress gemini G1, 2026-08-16). `SD(ΔSR) = √d / n` is derived from **this pair's own discordance** `d`, i.e. from re-running ONE arm. A cross-mode contrast (say SoM − DOM) has its own, larger `d`, hence its own wider null; judging it against a rerun-derived bar borrows `Var(A − A′)` to adjudicate `A − B` and is a category error. The number above answers exactly one question — *could a single rerun of the same arm have manufactured this?* — which is the arm-count-matched comparison §2 makes. For any other contrast, compute that contrast's own off-diagonal counts (McNemar / its own permutation test).

⚠️ This null assumes only exchangeability of the two runs; it does **not** model environment drift, which is one-directional and is what the P-SoM restart pair below shows. Where drift is present the true spread is larger than `√d / n`, so these thresholds are themselves a lower bound.

### The B1 floor was not missing — it was unrecognised

`phase0b_noise_floor.md` §7.1 lists *a locally-served (B1) same-mode replicate* as the top thing still needed, queued behind the WA run. It already existed: the WA 10-task pilot and the WA full-104 run are the **same condition** — `exp_v2_wa_full_reddit_base.yaml` only deletes `task.task_ids.reddit` — so their task overlap is a same-condition rerun. Per-mode:

| mode | paired n | \|pilot ∖ full\| | \|full ∖ pilot\| |
|---|---|---|---|
| dom | 10 | 0  | 0  |
| som | 10 | 0  | 0  |
| vision | 10 | 1 [597] | 1 [652] |
| ptext | 10 | 0  | 0  |
| pprompt | 10 | 0  | 1 [607] |

P-SoM is excluded from the pooled figure and reported alone: its `20260727` directory is a **restarted partial of the full run**, not the registered pilot draw (task ids 27..584, 2/10 overlap). On its 26 shared tasks it shows 3 / 0 one-directional flips (11.54% discordance) — one-directional, which reads more like state drift than symmetric noise.

**This refutes a live `CLAIM_UNVERIFIED`** — *"B1 是完全确定性的 (do_sample=False 贪婪解码 → 重跑 bit-identical)"*. Step-level greedy determinism (§298.2 133/133, §397.10 within-group 1.000) is real and is **not the same property**: an episode also carries site state, wall-clock, and session lifetime. The 2026-07-29 decision *"B1/B2 重跑地板不用测"* rested on the step-level evidence and does not survive.

⚠️ **This floor includes environment drift, not only stochasticity.** The two runs are days apart, and `require_reset` is a no-op on reddit (§402), so subscriptions accumulate across episodes. For the comparison in §2 that is correct — the paper's own conditions were also run at different times and carry the same drift — but the quantity must be named *run-to-run including environment drift*, never *decoding stochasticity*.

## 2. The arm-count-matched comparison

A floor is only interpretable against a gain of the **same functional at the same arm count**. Adding one arm to a single-mode baseline raises the oracle ceiling by `|{added} ∖ {baseline}| / n` — identical in form to `self_drop`, whether the added arm is a different representation or a rerun.

| cell | best single mode | +1 best **distinct representation** | +1 **rerun** (measured floor) | verdict |
|---|---|---|---|---|
| B0 · VWA-cls (n=224) | som @ 27.23% | **7.14pp** (dom) | 4.46 – 7.59pp | **indistinguishable — inside the rerun band** |
| B1 · WA-red (n=104; floor = 5 modes × 10 shared tasks) | dom @ 16.35% | **4.81pp** (ptext) | 0.00 – 10.00pp *(pooled would read 2.00–4.00)* | **indistinguishable — inside the rerun band** |
| B0 · WA-red (n=104; no pilot → no floor) | ptext @ 35.58% | **5.77pp** (dom) | — | no floor measured on this cell |
| B1 · VWA-cls (n=224) | som @ 14.29% | **4.91pp** (vision) | — | no floor measured on this cell |
| B2 · VWA-cls (n=224) | som @ 2.23% | **2.23pp** (vision) | — | no floor measured on this cell |
| B0 · VWA-red (n=203) | som @ 14.78% | **4.93pp** (dom) | — | no floor measured on this cell |
| B1 · VWA-red (n=203) | som @ 7.39% | **1.97pp** (pprompt) | — | no floor measured on this cell |
| B2 · VWA-red (n=203) | dom @ 3.94% | **1.97pp** (vision) | — | no floor measured on this cell |

Two cells carry a floor, and they differ in model family, benchmark and serving path. On `B0 · VWA-cls` the extra representation lands **inside** the rerun band. On `B1 · WA-red` it lands **inside** the honest band. ⚠️ Corrected 2026-08-04: it read *just outside by 0.81pp* against a band pooled over 5 modes on one shared 10-task draw — 50 observations carrying 10 independent tasks. Against the unpooled per-mode floors the gain is comfortably inside. the same order, and on a floor estimated from only n=50. Neither cell shows a representation arm worth appreciably more than a rerun arm; one shows it worth no more at all.

⚠️ **`B0 · VWA-cls` now carries 6 replicated arms, not one** (dom, vision, som, ptext, pprompt, psom) — the band above is the min/max over all of them. The `som` pair landed 2026-08-03 and it is the one that matters most: **claim 3 is about the fused arm, and until that day its floor was borrowed from DOM and Vision.** The borrowed band turned out to be right — SoM's own set-difference floor 5.36–7.59pp sits inside it and its mean-difference draw is 2.23pp, matching DOM's, so no number downstream moves. That is a robustness result rather than a correction, and it is worth more than the numbers: the claim no longer rests on an extrapolation.

### What this licenses, and what it does not

**Licensed on `B0 x VWA-cls`.** At the one-arm margin a distinct representation is worth no more than a rerun of the same representation: same cell, same `n`, same functional, same arm count.

⚠️ **The WA row does not have that property and must not be read as if it did.** Its rerun figure pools five mode-specific pilot-vs-full comparisons over the **10 registered pilot tasks**, giving `pooled_n=50` panels over ten distinct tasks — while the distinct-arm column beside it is computed on **104** tasks starting from `dom`. Worse for the comparison, `dom`'s own pilot-vs-full is **0 flips in either direction**; the 2-4pp band is generated entirely by `vision` and `pprompt`. So it is not the rerun of the arm the row's baseline names, the two columns do not share `n`, and the five panels are correlated through the same ten tasks. Read the WA line as *some* arms of this cell move by 2-4pp under repetition, not as a floor for `dom`. (codex Mode B, §H stress P1-6, 2026-08-02.)

**Not licensed.** *"The whole 6-mode ceiling gain is noise."* We hold one rerun arm, not five, and reruns have their own diminishing returns. The five-arm gain is reported below **with its arm count attached** and is never set against a one-rerun floor.

| cell | best single | 6-mode oracle | gain, **5 arms added** |
|---|---|---|---|
| B0 · VWA-cls (n=224) | 27.23% | 43.30% | 16.07pp |
| B1 · WA-red (n=104; floor = 5 modes × 10 shared tasks) | 16.35% | 30.77% | 14.42pp |
| B0 · WA-red (n=104; no pilot → no floor) | 35.58% | 51.92% | 16.35pp |
| B1 · VWA-cls (n=224) | 14.29% | 24.55% | 10.27pp |
| B2 · VWA-cls (n=224) | 2.23% | 7.14% | 4.91pp |
| B0 · VWA-red (n=203) | 14.78% | 26.11% | 11.33pp |
| B1 · VWA-red (n=203) | 7.39% | 11.82% | 4.43pp |
| B2 · VWA-red (n=203) | 3.94% | 7.39% | 3.45pp |

## 3. Consequence for the paper's four-step spine

| step | effect size | floor it meets | outcome |
|---|---|---|---|
| ① a real ceiling exists | 5-arm gain 4.39–16.07pp | one rerun buys 2.0–7.6pp | **survives, but the headline needs the rerun baseline printed next to it** |
| ② H3 axes are structural | 1.35 / 2.09pp pooled | lowest floor measured 2.0pp | **does not survive as a positive claim** — below even the most permissive floor |
| ③ structure < rerun floor | — | — | **this is the floor finding; now on two deployment forms, not one backbone** |
| ④ not learnable | 0/6 Pareto | — | **survives; noise only strengthens a negative result** |

Noise destroys positive claims. Of this paper's load-bearing steps, ③ and ④ are negative, ② was already demoted to weak evidence on 2026-07-28, and ① is the only positive one left — which is why §2's caveat is the whole cost.
