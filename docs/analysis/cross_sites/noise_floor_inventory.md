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
| `B1.wa-red` (**new**) | B1 x WA-reddit, registered 10-task pilot draw x 5 modes | 50 | **2.00pp** | **4.00pp** | 6.00% |

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
| B0 · VWA-cls (n=224) | som @ 27.23% | **7.14pp** (dom) | 4.91 – 7.59pp | **indistinguishable — inside the rerun band** |
| B1 · WA-red (n=104; floor n=50) | dom @ 16.35% | **4.81pp** (ptext) | 2.00 – 4.00pp | above the band by 0.81pp |
| B1 · VWA-cls (n=224) | som @ 14.29% | **4.91pp** (vision) | — | no floor measured on this cell |
| B2 · VWA-cls (n=224) | som @ 2.23% | **2.23pp** (vision) | — | no floor measured on this cell |
| B0 · VWA-red (n=203) | som @ 14.78% | **4.93pp** (dom) | — | no floor measured on this cell |
| B1 · VWA-red (n=203) | som @ 7.39% | **1.97pp** (pprompt) | — | no floor measured on this cell |
| B2 · VWA-red (n=203) | dom @ 3.94% | **1.97pp** (vision) | — | no floor measured on this cell |

Two cells carry a floor, and they differ in model family, benchmark and serving path. On `B0 · VWA-cls` the extra representation lands **inside** the rerun band. On `B1 · WA-red` it lands **just outside**, by 0.81pp — above the floor, but of the same order, and on a floor estimated from only n=50. Neither cell shows a representation arm worth appreciably more than a rerun arm; one shows it worth no more at all.

### What this licenses, and what it does not

**Licensed.** At the one-arm margin, a distinct representation is worth no more than a rerun of the same representation. Same cell, same `n`, same functional, same arm count.

**Not licensed.** *"The whole 6-mode ceiling gain is noise."* We hold one rerun arm, not five, and reruns have their own diminishing returns. The five-arm gain is reported below **with its arm count attached** and is never set against a one-rerun floor.

| cell | best single | 6-mode oracle | gain, **5 arms added** |
|---|---|---|---|
| B0 · VWA-cls (n=224) | 27.23% | 43.30% | 16.07pp |
| B1 · WA-red (n=104; floor n=50) | 16.35% | 30.77% | 14.42pp |
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
