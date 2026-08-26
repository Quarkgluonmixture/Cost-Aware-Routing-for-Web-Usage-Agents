---
type: framing-input
status: living
created: 2026-08-26
purpose: what the August replicates unlocked, which existing statements they falsify,
         and what is still unbuyable — input for the reframe after the 09-07 verdict
---

# Evidence delta — what the August data lets us say that we could not say before

Written against `realm/section1_intro.md` (the 8-item inventory) and
`_status/tasks/task_naacl2027_main.md` (the seven attack surfaces). **This is not a
frame.** The REALM verdict lands 09-07 and the claim should be chosen against
coverage, not before it — three frames died in 08-01/08-03 for exactly that reason.
What follows is the raw material: what became sayable, what became false, what is
still out of reach.

---

## 1. Three things became sayable

### C1. The reproducibility floor groups by **serving path**, not by model

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

## 3. Against the seven attack surfaces

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

## 4. What is still unbuyable, and why

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
`task_naacl2027_main.md` asks of work done in this window.

---

## 5. How to read this after 09-07

C1 is the only item here that stands **independent of the routing line**. If the
verdict attacks routing, C1 survives as a methods contribution; if it attacks the
measurement, C1 is the measurement. That asymmetry is worth knowing before choosing
which of the two the next paper leads with — but the choice itself waits for the
verdict, and this document deliberately does not make it.
