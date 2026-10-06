---
type: framing-input
status: living
created: 2026-08-26
updated: 2026-10-06
purpose: the evidence layer read against the seven reviewer attack surfaces (§0, live, for
         COLING 2027 / ARR 2026-10-12); §1–§5 are the 08-26 record of what the August
         replicates unlocked, kept as written apart from the retraction banners
---

# Evidence delta — what the August data lets us say that we could not say before

## 0. Status against the seven attack surfaces — as of 2026-10-06

COLING 2027 (via ARR 2026-10-12) replaced NAACL as the target on 10-06; the filename keeps
the old name because notes cite it. **This section supersedes §3 and §4.** It is still not a
frame (user 10-06: complete the evidence layer first, choose the frame after).

**Pointers, not numbers.** Each row names the product or 实验笔记 § that owns the number;
read it there. The three coverage facts below are derived, so each carries its recompute.

**Coverage the rows rely on** (derived 2026-10-06):

- **11 cells**: VWA·classifieds × B0/B1/B2/B5 · VWA·reddit × B0/B1/B2 · VWA·shopping × B0/B1 ·
  WA·reddit × B0/B1. Recompute: `docs/analysis/run_inventory/run_matrix.md` (regenerate per its README).
- **Rerun band per cell.** All six arms: cls_B0, red_B0, WA_B1. Partial: cls_B1 (3/6), red_B1 (2/6),
  cls_B5 (DOM only). None: cls_B2, red_B2, shop_B0, shop_B1, WA_B0 ⇒ **5 of 11 cells cannot be
  read against a band** (6 of 11 on 09-09; WA_B1 closed by the 09-15 chain, §529.4).
  Recompute: the labels of `CLEAN_PAIRS` in `scripts/analysis/aggregate_noise_floor_inventory.py`
  (`<baseline>.<site>.<mode>`; `wared` = WA·reddit).
- **Cross-site product suite** (`docs/analysis/cross_sites/*.json`) is wired for 8 units:
  shop_B0/shop_B1 appear in none, cls_B5 as a cell in one (§531.2). The only analysis that
  covers all 11 cells is the §505 router pilot (`results/evidence_snapshots/20261006_router_pilot_20260909/`).
  Recompute: grep the cell keys across those JSONs.

| # | surface | what the layer holds (owner) | moved since 08-26 | still open — and closable before 10-12 on existing data? |
|---|---|---|---|---|
| 1 | routing generalisation | §505.19: task difficulty pools across backbone and site, task×mode fit does not · budget router frozen before its test data was read (`pre_run/budget_router_prospective_shop_B1_20260909.*`), evaluated §510.3 / §515.3 | **yes** — the first prospective test exists; the pre-declared primary policy passes the direction criterion on both held-out shop_B1 arms | single run, B1 only, no band on shop_B1; both held-out arms are the text arms that B-2002 hits hardest (§531.9) ⇒ the result must carry that caveat. Wording only. |
| 2 | cross-site / cross-benchmark | §505 pilot over 11 cells, 2 benchmarks, 4 sites · WA_B1 band (§529.4) · shopping: 9 conditions registered opt-in (§531.5), Tier-1 scan + Tier-2 sample (`docs/analysis/vwa_shopping/`, §531.9) | **yes** — shopping went from "no landed run" (§4 below) to 9 full conditions with a first failure attribution | (a) suite not opted in to shopping / cls_B5 — **0-compute, per product**; (b) shopping mode comparisons carry B-2002 (query concatenation, mode-uneven) and B-2003 (wishlist not reset); (c) category-grid order across runs unverified; (d) VWA and WA are one benchmark lineage — not closable. |
| 3 | is the baseline strong enough | B5 (GPT-5.6) on cls in five modes (manifest `extension:`) · §505.10: the stronger backbone moves the mode main effect, not the interaction | **yes** — B5 now spans modes, not only DOM | B5·vision is the broken coordinate-contract run (B-1997, §508) — not a capability reading; B5 is one site. Not closable (paid). |
| 4 | beats a simple heuristic | §387.16.4 (always-cheapest + label-shuffle controls) · §505.4 · §505.18 (hand rules forward, mined rules checked on reruns) · `rule_routing_pareto`, `router_objective_ordering` | **yes** — heuristics tested in both directions | which learned-vs-always-cheapest count the paper uses is unpinned: three counts under different policies and baselines (§530.4 #3). **User decision.** |
| 5 | is the cost-accuracy trade-off stable | budget frontiers §505.21/22, per cell §505.24 · `outcome_efficiency` · `multimetric_pareto(_with_wa)` | **mixed** — the budget line has an SR–cost frontier; but the band's upper edge moved when WA entered and C1 is retracted (§529.4), so "smaller than the floor" has to be re-read per cell | (a) noise-band definition and scope per use (§530.4 #2) — **user decision**; (b) for representation routing the products give dominance verdicts, not the frontier itself, which is what the VLM4RWD AC asked for — 0-compute re-cut, not done; (c) early-stop / budget precedent literature unchecked (§505.27) — 0-compute, must precede any novelty wording. |
| 6 | do DOM / SoM / Vision observations generalise | `representation_class_comparison`, `representation_deployment_profile`, `per_mode_four_dimension_profile(_with_wa)`, `cross_mode_failure_signatures` | **yes** — a fourth backbone (B5) and a third VWA site (shopping) exist | neither is in those products (row 2a); shopping's cross-mode reading needs the B-2002 caveat. |
| 7 | near-perfect AUROC = artifact? | §111.2 (linear probe is the wrong tool), §127.1 (in-sample), §394 retraction, §460.3 (`0b-extra`'s high value is whole-episode aggregation), §505.3 (the model's first output adds ≈ 0 AUROC over pre-flight features) | **yes** — §505.3 is a direct test | the 08-26 label-noise argument survives C1's retraction (labels flip between identical reruns on both serving paths), but cite `noise_floor_inventory`, not `serving_mode_floor`. |

**What the reviewers already named** (REALM ×3 `_status/issues/issue_realm_reviews_2026-09-09.md`,
VLM4RWD ×2 `_status/issues/issue_vlm4rwd_reviews_2026-09-29.md`), mapped onto the layer:

- *Rerun control is the strongest contribution* — `noise_floor_inventory` (24 pairs). Its stated
  weakness is coverage, and that is the 5-of-11 above. No new pair can land before 10-12
  (A100 unreachable; anything from the 09-15 chain after 09-24 is only there).
- *The label-supply explanation is not established* — `router_label_supply_diagnosis`,
  `retry_vs_switch_label_supply`, `supply_value_coupling`, and now `rerun_flip_failure_anatomy`
  (§531.10): flipping reruns part from step 0 and their failure type is only partly stable. That
  is consistent with labels set by stochastic execution, but it is not a direct test of supply;
  absent one, the explanation reads as a hypothesis consistent with the data.
- *SR–cost trade-off not explored beyond dominance* — row 5 (b).

---

Written against `realm/section1_intro.md` (the 8-item inventory) and
`_status/tasks/task_coling2027_main.md` (the seven attack surfaces). **This is not a
frame.** The REALM verdict lands 09-07 and the claim should be chosen against
coverage, not before it — three frames died in 08-01/08-03 for exactly that reason.
What follows is the raw material: what became sayable, what became false, what is
still out of reach.

---

## 1. Three things became sayable

### C1. The reproducibility floor groups by **serving path**, not by model

> ⛔ **RETRACTED 2026-10-06** (实验笔记 §529.4). The 09-15 local replicate chain put all three
> powered local WA·B1 arms above the API group's lower edge; `serving_mode_floor` now reports
> the groups as overlapping, and the pre-declared intent file
> (`pre_run/local_replicate_chain_launch_intent_20260915.md`) fixed that outcome as "C1 is dead
> as stated". The text below is the 08-26 record; do not cite it.

Product: `docs/analysis/cross_sites/serving_mode_floor.{json,md}`

|  | arms | families | sites | floor |
|---|---|---|---|---|
| **API-served** | 10 | Qwen, OpenAI | 2 | **7.39–14.29%** |
| **locally served** | 3 | Qwen | 1 | **0.00–3.12%** |

The ranges do not overlap; the gap is 4.26pp. Exact one-sided rank test on a perfect
split: **p = 0.0035**. Restricted to arms that carry an interval (`d ≥ 10`), all ten
API arms qualify and two of three local ones do, the split survives, and the gap
*widens* to **7.39pp** (p = 0.0152) — the only local arm above zero was the
underpowered one.

**Why this could not be said before 2026-08-21.** The project held one API backbone.
"B0 has a 12% floor" and "API serving has a 12% floor" were the same sentence. B0 is
MoE *and* proxy-served; B1 is dense *and* local — perfectly confounded. A second API
model from an unrelated family (GPT-5.6-terra, closed weights, 12.95%) removes the
*family* reading.

**Independent corroboration on the local side.** 实验笔记 §298.2: a controlled
step-level probe on B1 returned determinism **133/133 OK**. The local near-zero is
not only an inference from replicate pairs.

**What it does not remove: scale.** The API group is 235B/undisclosed, the local
group is 4B. Serving path still covaries with size. The experiment that settles it is
the same checkpoint served both ways; neither direction fits this project's compute
envelope, so naming it is the honest substitute, not performing it.

**No mechanism, by standing adjudication.** §302.5 fixed the claim at an *observable
provider-dependent floor* and named three escape hatches unusable without a
server-side audit artifact: "MoE is the cause", "switch provider", "provider bug".
The repo holds no expert-route log, no batch id, no instance id, no model SHA. C1 is
a statement about where the floor is, never about why.

**Why it is worth more than a caveat.** Agent benchmarks report success as a point
estimate, and nearly all of them run through an API, once. If an identical condition
disagrees with itself on 7–14% of tasks under API serving and on 0–3% locally, then
the reporting convention is wrong in a way that is a property of the *serving path* —
and that is a claim about the field's method, not about this project's routing
result. It stands whether or not the routing line survives review.

### C2. "A new representation is worth about what a rerun is worth" now has a second site

| cell | +1 distinct arm | measured rerun floor | verdict |
|---|---|---|---|
| `cls·B0` | +7.14pp (DOM) | 4.46–7.59pp (all six arms) | inside |
| `red·B0` | +4.93pp (DOM) | **1.97–6.90pp (text side)** | **inside** |
| `WA·B1` | +4.81pp | 0.00–10.00pp (pooled, 10 shared tasks) | inside |

`red·B0` moved from *"no floor on this cell"* to a measured band this session. The
added arm there is DOM — text side — and reddit's replicated arms are the three
phantom modes, also text side, so the comparison is like-for-like. The table now
**gates on side**: reading a text-side floor against an image-bearing arm is the
cross-site form of the per-arm-threshold error §477.2 banned, so the verdict column
says so instead of silently comparing.

### C3. The "is the baseline strong enough" surface is answered

B5 = GPT-5.6-terra, a closed frontier model, on `cls·dom`: **23.66% / 25.00%** across
its two runs, against B0's 17.41%, B1's 6.25%, B2's 1.34%. The prediction was recorded
before the fire (user, 08-19, "b5 我估计会很强") and the intent file fixed what each
outcome would mean, so this is not a post-hoc reading.

Second-order and more interesting than the SR: **the stronger model does not have a
smaller floor** (12.95%, squarely inside B0's 10.27–14.29%). Capability and
reproducibility are not the same axis.

---

## 2. Three existing statements are now false

| where | says | actually |
|---|---|---|
| inventory **#7** | "Only **2 of 8** cells carry a measured floor, and **neither measures it on the arm being added**" | **3 of 8** carry one. On `cls·B0` all six arms are replicated, so whichever arm the comparison adds carries its own floor. Both clauses dead. |
| inventory **#5** | rerun band "**0.89–2.23pp**" | six-arm range is **0.89–2.68pp**; `cls_B0`'s +2.23pp no longer "equals the band's upper edge exactly" |
| Table 27 caption | "three arms of that cell … **no VWA-reddit cell** … carries one at all" | `B0.cls x6, B0.red x3, B1.cls x3, B5.cls x1` — fixed 2026-08-26, now read from the registry |

⚠️ **REALM #192 is a submitted snapshot and must not be back-edited.** These are
corrections for the *next* draft. The submitted text was true when submitted.

---

## 3. Against the seven attack surfaces (as of 2026-08-26 — superseded by §0)

| # | surface | status after August |
|---|---|---|
| 1 | routing generalisation | **unmoved** |
| 2 | cross-site / cross-benchmark | **partly** — the floor now has two sites; `B5 × reddit` is armed and would give the second API model a second site |
| 3 | is the baseline strong enough | **answered** (C3) — and `sol` is unnecessary by the pre-declared criterion |
| 4 | beats a simple heuristic | **unmoved**; the hard negative in §387.16.4 still has to be met head-on |
| 5 | is the cost-accuracy trade-off stable | **strengthened** — C1 makes "smaller than the floor" a measurable disqualifier rather than a hedge, and the floor is now measured on both sites |
| 6 | do DOM/SoM/Vision generalise | **in flight** — Phase C is running B5 across five more modes on `cls` |
| 7 | near-perfect AUROC = artifact? | **indirectly strengthened** — if the label itself flips on 7–14% of tasks under API serving, a ceiling on attainable AUROC follows from the label noise, independent of the two artifacts already diagnosed (§111.2 wrong tool, §127.1 in-sample) |

---

## 4. What is still unbuyable, and why (as of 2026-08-26 — superseded by §0)

> Two rows no longer hold: shopping now has 9 full conditions (§0 row 2), and the
> `B1 × reddit` buy was meant to support C1, which is retracted.

| gap | why it matters | buyable? |
|---|---|---|
| same checkpoint served both ways | the only thing that separates *serving path* from *scale* in C1 | **no** — needs either a 235B self-host or an API endpoint for a 4B; neither is in the envelope |
| local side, second family | would make C1 symmetric (2 families each side) | **no** — B2's SR is 0.45–2.23%, so `d ≈ 1.8`. This is a power limit, not a scheduling one: running it would produce a number that cannot be read |
| local side, second site | C1's local group is one site | **yes**, cheap — `B1 × reddit` replicate, local GPU only, no API spend |
| `B5 × reddit` | C2 and C3 both get a second site for the second API model | **already armed** (`_b5_reddit_chain.sh`, waiting on Phase C) |
| a third workload | inventory says two workloads cannot identify a moderator | **no** — `shopping` has zero landed directories |

**The cheapest remaining buy is `B1 × reddit`**: local weights, no proxy spend, and it
is the one gap that makes C1's control group cross-site rather than single-site. It is
also *direction-independent* — it strengthens C1 no matter which frame the 09-07
verdict points at, which is exactly the property
`task_coling2027_main.md` asks of work done in this window.

---

## 5. How to read this after 09-07 (as of 2026-08-26)

> Its premise — C1 standing independent of the routing line — fell with C1 (§529.4).

C1 is the only item here that stands **independent of the routing line**. If the
verdict attacks routing, C1 survives as a methods contribution; if it attacks the
measurement, C1 is the measurement. That asymmetry is worth knowing before choosing
which of the two the next paper leads with — but the choice itself waits for the
verdict, and this document deliberately does not make it.
