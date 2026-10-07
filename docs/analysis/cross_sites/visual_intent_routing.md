---
type: analysis
status: rolling
purpose: does a 0-token text rule identify, in advance, the tasks where the screenshot pays
producer: scripts/analysis/visual_intent_routing.py
---

# Visual-intent routing — an ex-ante partition

Regenerate: `.venv/bin/python3 scripts/analysis/visual_intent_routing.py`

The predicate is a regex over the task intent plus a check that the task carries no reference image:

```
\b(image|picture|photo|screenshot)\b|\bcolou?r of\b|\bhow many\b[^.]{0,40}\bin (?:the|this)\b
```

Both terms read the **task config**. No model call, no episode, no tokens — this is decidable before anything runs.

⚠️ **This is not `P43` as shipped.** The production rule adds `if summary.get("success"): return []`, which makes its hit set outcome-dependent — tasks the text-only arms solved are excluded by construction, so an arm comparison inside that set measures the selection. The outcome filter is dropped here; everything else is P43's predicate verbatim.

⚠️ **Provenance.** The rule was written for *reddit* ("64 reddit tasks, previously invisible to every Tier-1 rule"). Its classifieds hits were incidental and never examined, so the classifieds rows are **out-of-sample** — the regex was not tuned on them.

## classifieds — flagged n=71

| cell | arm | flagged Δ vs DOM | 95% CI | rest Δ vs DOM | 95% CI | concentration |
|---|---|---|---|---|---|---|
| `cls_B0` | vision | **+22.54pp** (23/71 vs 7/71) | [+9.86, +33.80] | +0.65pp | [-5.88, +7.84] | **+21.88pp** |
| `cls_B0` | som | **+19.72pp** (21/71 vs 7/71) | [+7.04, +32.39] | +5.23pp | [-1.31, +12.42] | **+14.49pp** |
| `cls_B1` | vision | **+16.90pp** (13/71 vs 1/71) | [+8.45, +25.35] | +1.31pp | [-3.92, +6.54] | **+15.59pp** |
| `cls_B1` | som | **+12.68pp** (10/71 vs 1/71) | [+5.63, +21.13] | +5.88pp | [+0.00, +11.76] | **+6.79pp** |
| `cls_B2` | vision | **+1.41pp** (3/71 vs 2/71) | [-4.23, +7.04] | +0.65pp | [-1.31, +3.27] | **+0.75pp** |
| `cls_B2` | som | **+0.00pp** (2/71 vs 2/71) | [-5.63, +5.63] | +1.31pp | [-1.31, +3.92] | **-1.31pp** |

## reddit — flagged n=63

| cell | arm | flagged Δ vs DOM | 95% CI | rest Δ vs DOM | 95% CI | concentration |
|---|---|---|---|---|---|---|
| `red_B0` | vision | **-3.17pp** (6/63 vs 8/63) | [-14.29, +7.94] | -8.57pp | [-14.29, -2.86] | **+5.40pp** |
| `red_B0` | som | **+0.00pp** (8/63 vs 8/63) | [-9.52, +9.52] | +0.71pp | [-4.29, +5.71] | **-0.71pp** |
| `red_B1` | vision | **-3.17pp** (1/63 vs 3/63) | [-9.52, +3.17] | -3.57pp | [-7.86, +0.71] | **+0.40pp** |
| `red_B1` | som | **+1.59pp** (4/63 vs 3/63) | [-3.17, +7.94] | +1.43pp | [-2.14, +5.00] | **+0.16pp** |
| `red_B2` | vision | **+4.76pp** (4/63 vs 1/63) | [-1.59, +12.70] | -5.00pp | [-8.57, -1.43] | **+9.76pp** |
| `red_B2` | som | **-1.59pp** (0/63 vs 1/63) | [-4.76, +0.00] | -3.57pp | [-7.14, +0.00] | **+1.98pp** |

## WebArena reddit

Same predicate. The universe is the six-mode task intersection (WA has no AMENDMENT_08 list) and the configs come from each run's mirrored `task_configs/`.

⚠️ **The predicate barely fires here.** It flags 71 of 224 classifieds tasks and 63 of 203 VWA-reddit tasks, but only **5 of 104** on WA — WebArena's intents are worded differently and the regex, which was written against VWA phrasing, mostly misses. Whatever these rows show, they are not a test of the classifieds result: **the WA cells are a coverage note, not a replication.**

| cell | n flagged | arm | flagged Δ vs DOM | 95% CI | rest Δ vs DOM | 95% CI |
|---|---|---|---|---|---|---|
| `wa_B1` | 5/104 | vision | **degenerate** — no mode solves any flagged task, so this is *no information*, not a measured null | — | -7.07pp | [-15.15, +0.00] |
| `wa_B1` | 5/104 | som | **degenerate** — no mode solves any flagged task, so this is *no information*, not a measured null | — | -3.03pp | [-9.09, +4.04] |
| `wa_B0` | 5/104 | vision | **degenerate** — no mode solves any flagged task, so this is *no information*, not a measured null | — | -8.08pp | [-17.17, +1.01] |
| `wa_B0` | 5/104 | som | **degenerate** — no mode solves any flagged task, so this is *no information*, not a measured null | — | -5.05pp | [-13.13, +4.04] |

## What this says

On classifieds the partition separates: `cls_B0` pays **+22.54pp** for the screenshot on the flagged tasks against **+0.65pp** on the rest. The flagged set is 71 of 224 tasks and the predicate costs nothing to evaluate.

Two things this is **not**. It is not a claim that the flagged tasks are unsolvable without the screenshot — several are solved by DOM. And it is not a router: the partition is fixed and known in advance, which is what makes it cheap, but nothing here learns it or adapts it.

Three caveats that belong beside the number, in order of how much they cost:

1. **It needs capability to cash in.** `cls_B2` gets +1.41pp on the same flagged set — the weakest backbone cannot use the screenshot even when the rule correctly says it is needed. Two of three backbones, not three of three.
2. **It is site-specific, and on reddit the sign flips.** `red_B0` gets -3.17pp on its flagged tasks — the screenshot *hurts* there. Same predicate, opposite verdict, which is the modality reversal showing up in a third functional.
3. **The counts are small.** The flagged sets are ~70 tasks and the successes behind the largest gap are in the low twenties against single digits; the intervals above are paired bootstrap over tasks and should be read, not the point estimates alone.

## Extension (2026-10-08, 笔记 §548): the image-only contrast and the extension cells

**SoM − P-SoM** differ only in the annotated screenshot (same `[SOM_MARKS]` payload, same prompt), so it is the cleanest reading of what the image alone buys; `vision − dom` above also swaps the text payload away. Extension cells: **cls_B5** (GPT-5.6; its Vision run is the broken coordinate contract, B-1997, so only SoM-based contrasts), **shopping** B0 / B1 in a `clean` version (whole task rows touched by B-2002 on any arm, the 42 grid-order and 4 B-2003 tasks dropped) and an `all` version for sensitivity. WebArena stays a coverage note: the predicate flags 5 of 104 tasks.

| cell | contrast | n flagged | flagged Δ [95% CI] | rest Δ [95% CI] | concentration | |
|---|---|---|---|---|---|---|
| `cls_B0` | som − psom | 71 | **+25.35pp** (21/71 vs 3/71) [+14.08, +36.62] | +5.23pp [-1.31, +11.76] | **+20.12pp** | |
| `cls_B1` | som − psom | 71 | **+11.27pp** (10/71 vs 2/71) [+2.82, +21.13] | +5.88pp [+0.00, +11.76] | **+5.39pp** | |
| `cls_B2` | som − psom | 71 | **+2.82pp** (2/71 vs 0/71) [+0.00, +7.04] | +0.65pp [-1.96, +3.27] | **+2.16pp** | |
| `red_B0` | som − psom | 63 | **+3.17pp** (8/63 vs 6/63) [-6.35, +12.70] | +4.29pp [-0.71, +9.29] | **-1.11pp** | |
| `red_B1` | som − psom | 63 | **+1.59pp** (4/63 vs 3/63) [+0.00, +4.76] | +1.43pp [-1.43, +4.29] | **+0.16pp** | |
| `red_B2` | som − psom | 63 | **degenerate** — no information | — | +0.71pp | — |
| `cls_B5` | som − psom | 71 | **+29.58pp** (27/71 vs 6/71) [+16.90, +42.25] | +7.19pp [+0.65, +13.73] | **+22.39pp** | |
| `cls_B5` | som − dom | 71 | **+28.17pp** (27/71 vs 7/71) [+16.90, +39.44] | +6.54pp [-0.65, +13.73] | **+21.63pp** | |
| `shop_B0_clean` | som − dom | 48 | **-2.08pp** (2/48 vs 3/48) [-6.25, +0.00] | +5.00pp [+0.94, +9.06] | **-7.08pp** | |
| `shop_B0_clean` | vision − dom | 48 | **+4.17pp** (5/48 vs 3/48) [-6.25, +14.58] | +3.75pp [-0.62, +8.12] | **+0.42pp** | |
| `shop_B0_all` | som − dom | 56 | **-1.79pp** (2/56 vs 3/56) [-5.36, +0.00] | +4.26pp [+0.53, +7.98] | **-6.04pp** | |
| `shop_B0_all` | vision − dom | 56 | **+3.57pp** (5/56 vs 3/56) [-5.36, +12.50] | +2.93pp [-1.06, +6.91] | **+0.65pp** | |
| `shop_B1_clean` | som − psom | 35 | **+5.71pp** (2/35 vs 0/35) [+0.00, +14.29] | +3.82pp [-0.64, +8.28] | **+1.89pp** | |
| `shop_B1_clean` | som − dom | 35 | **+2.86pp** (2/35 vs 1/35) [-5.71, +11.43] | +3.82pp [-0.64, +8.28] | **-0.96pp** | |
| `shop_B1_clean` | vision − dom | 35 | **+5.71pp** (3/35 vs 1/35) [-5.71, +17.14] | +0.00pp [-5.10, +5.10] | **+5.71pp** | |
| `shop_B1_all` | som − psom | 56 | **+3.57pp** (2/56 vs 0/56) [+0.00, +8.93] | +2.93pp [+0.27, +5.85] | **+0.65pp** | |
| `shop_B1_all` | som − dom | 56 | **+1.79pp** (2/56 vs 1/56) [-3.57, +7.14] | +2.93pp [+0.27, +5.59] | **-1.14pp** | |
| `shop_B1_all` | vision − dom | 56 | **+3.57pp** (3/56 vs 1/56) [-3.57, +10.71] | +0.27pp [-2.66, +3.19] | **+3.31pp** | |

Shopping: the predicate flags 56 scored tasks before any drop. Cells read: `cls_B5` 224/224, `shop_B0_clean` 368/432, `shop_B0_all` 432/432, `shop_B1_clean` 192/432, `shop_B1_all` 432/432.

