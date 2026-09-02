# Poster content — Holistic AI × UCL CDI Showcase, 16 Sep 2026

**Author-facing source of truth.** The strings are inlined in `build_poster.py`;
any edit must be mirrored in both. Every number carries its scope, because the
poster is printed on silk and cannot be corrected.

> **v4 (2026-09-02) — the template's own skeleton.** A reviewer at Holistic AI
> read the v2 exhibition design against the posters the organisers circulated
> and asked for four things: *one main system diagram; compact; font sizes and
> formats consistent; stay on the template.* v4 is that brief. Every number,
> scope line and baseline name is carried over from v2 (see the audit at the
> end for the two that changed because the ledger had moved on). What went:
> the 200pt hero zero, the dark takeaway band, the drawn kayak listing, the
> calibration card, the serif pull-quotes — and with them fourteen type sizes
> across three families. What the sheet uses now is the template's own scale,
> read from its placeholders: title 41 Georgia · standfirst 28 Georgia ·
> section header 14 Consolas bold · body 17.65 Arial · caption 12.7 Arial grey
> · metric 32.5 / 10.6 Consolas. Emphasis is bold only.

> **The two baselines.** Half the numbers below are measured against the
> **best-success fixed mode** and half against **always-cheapest**. They are not
> interchangeable, and the metric strip is exactly where that distinction gets
> flattened. Every number therefore prints its own baseline.

---

## Title

> **Can a web agent learn when a screenshot is worth the cost?**

Must set on ONE line in the 431mm the logos leave — `build_poster.py` asserts
this. The template anchors the title box to its *bottom* edge, so an over-long
title grows upward off the top of the sheet rather than down into the byline.

## Standfirst (template band, 28pt, two lines)

> Choosing the page representation per task could solve up to 16 more tasks in
> 100 than the best fixed mode — yet none of 8 learned routers beat always using
> the cheapest mode on both success and cost.

At the template's 28pt the band holds **two** lines (~190 characters), and the
build asserts it. Both baselines are named inside the sentence: `than the best
fixed mode` for the ceiling, `always using the cheapest mode` for the 0/8. The
v2 trap — a pronoun pointing the 0/8 back at the ceiling — cannot recur because
there is no pronoun. `up to 16` is the largest of the 8 cells (`wa_reddit·B0`,
+16.35pp); the metric strip says so in its label.

## Fig 1 — the system diagram (full width)

Thesis Fig 1.1 (`final_dissertation/figures/fig_overview.pdf`), all three
panels, rasterised at 300 dpi and placed at the template's full 557mm width,
where its type prints at 1.12× the figure's native size. Section header:
`HOW A WEB AGENT SEES A PAGE, AND WHAT WE COMPARED`.

> **Fig 1.** The agent (①) is held fixed — same task, actions, step limit and
> cost accounting — and only the page encoding it is handed (②) varies: six
> observation modes, DOM, three text-only variants (P-text, P-prompt, P-SoM),
> SoM and Vision. Each runs on 8 website–model settings from VisualWebArena and
> WebArena (8,934 episodes). Three policies (③) are compared: a fixed mode, a
> hindsight oracle that picks the best mode per task after the fact, and a
> learned router that must choose before the task runs.

Episodes = 6 × (224×3 + 203×3 + 104×2); per-cell `n` from
`router_triage_learnability_with_wa.json`. The three-way mode grouping is
`TERMS.md §1.1`, verified from step records (`image_payload_bytes == 0` for all
four text-side modes). The diagram's own labels (`Learned triage`, `Confidence
cascade`) are the thesis's; the poster's prose says `learned router`.

## Column 1 · THE PROBLEM

> A web agent looks at a page, decides what to click or type, acts, and repeats.
> Before each step it must be handed some encoding of the page.
>
> That encoding is usually chosen once and paid for at every step: cheap text,
> an expensive annotated screenshot, or both.

Then one real page, sent three ways (a small ruled table, not a figure):

| | tokens | payload |
|---|---|---|
| **DOM** | 3,314 tokens | text only, no image |
| **SoM** | 4,335 tokens | text + 143 KB marked image |
| **Vision** | 3,123 tokens | 110 KB screenshot, no text |

> SoM's text is within 1% of DOM's; nearly all of its extra cost is the image.

Scope line printed above the table: `B0 · classifieds task 0 · first step`.
Values are **not copied from the thesis figure** — they were re-read from the
F1 pipeline on 2026-09-02 (`fig_f1_motivating_example.gather()`: dom 3,314 /
som 4,335 / vision 3,123 input tokens; image 146,220 B → 143 KB, 30 marks;
112,680 B → 110 KB; SoM/DOM chars 1.0077). The dom-run and vision-run step-000
screenshots are md5-identical, so the three rows are the same page.

> **Is the screenshot needed at every step — and can the steps that need it be
> identified cheaply enough to be worth identifying?**

The dissertation's research question, in the abstract's own words.

## Column 1 · WHY IT CANNOT BE LEARNED

> A routing label exists only when the agent solves a task. Here the best single
> mode solves just **2–36%** of tasks, which leaves typically **15–97** usable
> labels per setting.
>
> **The agents that would gain most from routing produce the least supervision
> to learn it.**
>
> Deliberately shrinking the training data confirms scarcity is the mechanism,
> and prices it: the failing settings would need at least **2.1–4.2×** more
> tasks than the benchmarks contain — a specification, not an impossibility.

`2–36%` is `baseline_policy.sr_pct` — the best single mode's own SR — on the
8-cell matched set (§450.8); the 6-cell VWA figure is 2–27 and must not be
quoted next to "8 settings". The poster now says *the best single mode solves*
rather than *these agents solve*, which is what the quantity is.
`15–97` is C5 (`router_label_supply_diagnosis.md`), 4 of 6 VWA cells; the
denominator is dropped on the poster (a six-cell diagnosis beside eight-cell
numbers costs more than it buys) and is in `SHOWCASE_PREP.md §5` for the spoken
answer. `2.1–4.2×` is §453.2's *replaced_by* wording — a specification, not an
impossibility — and carries `at least` because the arithmetic assumes class
proportions constant in `n` (it is a lower bound).

## Column 1 · HOW MUCH OF THIS IS NOISE?

> Rerunning **one unchanged mode** on the same tasks flips **10–14%** of
> outcomes and by itself buys **2.0–7.6 pp** of success (B0 · classifieds, six
> replicated modes, n=224).
>
> Every gain on this sheet is read against that band, not against zero.

`10–14%` is the six-arm figure (§477.2: 10.27–14.29% across all six modes on
cls·B0). **v2 printed `12–14%` with "three replicated modes"** — correct for its
date, superseded once the six-arm replicates landed. `2.0–7.6 pp` is
`noise_floor_inventory.md §3 ①` (§450.10), the mandatory companion of any
ceiling number.

## Columns 2–3 · RESULTS

Metric strip (the template's own component; each label names its baseline):

| number | label |
|---|---|
| `+16.35 pp` | CEILING, LARGEST OF 8 · VS BEST FIXED MODE |
| `0 of 8` | LEARNED ROUTERS BEAT ALWAYS-CHEAPEST |
| `1 of 8` | HINDSIGHT ORACLES BEAT ALWAYS-CHEAPEST |

The build asserts each label fits its tile; the first draft's label overprinted
the second tile.

Fig 2 — thesis F13, re-authored at the 362mm inner width of the figure box
(`poster_figures.py`, type 18–24pt):

> **Fig 2.** Every policy in every setting against one fixed baseline, **always
> use the cheapest mode** (★). A win lands in the shaded region: cheaper *and* no
> worse. Always-cheapest is cheapest on average, not per episode, which is why a
> few points sit left of it. Nested cross-validation; 10,000 bundle permutations.

Three verdicts, body size, bold lead:

> **The ceiling is real.** In hindsight, choosing the mode per task solves
> **+3.45 to +16.35 pp** more than the best single fixed mode, at 1.6–35.3%
> lower cost, in 8 of 8 settings.
>
> **Nothing we trained wins.** **0 of 8** learned routers beat always-cheapest on
> both success and cost — and even the hindsight oracle does so in only
> **1 of 8**.
>
> **What survives is a bound, not a router.** Sending the tasks nobody solves to
> the cheapest mode saves 9.5–30.6% at identical success in 8 of 8 — against the
> best-success fixed mode, and plain always-cheapest usually saves more.

`1.6–35.3%` is the **cost-aware tie-break** figure (§452.2 *replaced_by*). The
older `13.7–35.3%` is **RETRACTED**: on two cells the "best single mode" was an
SR tie and the list-order tie-break picked the *dearer* one, inflating the
saving (cls·B2: −23.4% → −1.6%). v2's build script still carried 13.7 in a
docstring; nothing printed it, and v4 prints the corrected number.
The 1-of-8 line is mandatory (§450.12): without it `0 of 8` reads as a claim
about learners. `deployable` is the load-bearing word in the spoken version of
the second verdict — one hindsight oracle *does* reach the win region.
The third verdict keeps §387.16.3's caveat (*plain always-cheapest usually saves
more*); without it, "what survives" plus a percentage reads as a deployable win.

## Columns 2–3 · TAKEAWAY

> **Routing is not only a model-selection problem: its learnability depends on
> the competence of the agent producing the labels.** So, in this order: improve
> the agent, then generate reliable supervision, then learn selective perception.
>
> Measured inside the 2–36% success regime we observed. This conclusion need not
> hold for stronger agents.

The scope line is load-bearing. `THESIS_ONE_SENTENCE.md` explicitly does **not**
claim routing is unlearnable in general; the finding is bounded to the observed
success regime.

---

## Deliberately NOT on the poster

Per `THESIS_ONE_SENTENCE.md` "这篇论文不主张什么":

- ❌ that P-SoM substitutes for SoM (the value is complementarity)
- ❌ that "dropping the image barely costs anything"
- ❌ that routing is unlearnable for other models or benchmarks
- ❌ token/dollar cost silently re-labelled as energy or carbon
- ❌ the retracted AUROC 0.65–0.72 narrative (§394), the retracted "ceiling is
  entirely in cost" (§396.2), the 6-cell "1.7–3.3pp drop-one" hero, the
  retracted "cost floor" sentence (§476.4), the retracted `13.7–35.3%` (§452.2)
- ❌ the `7.14 pp` vs `4.46–7.59 pp` calibration pair — it was v2's card; v4
  states the band (`10–14%`, `2.0–7.6 pp`) and leaves the pair to the spoken
  answer in `SHOWCASE_PREP.md §4`
- ❌ a standalone "what this poster does not claim" box — every limit travels
  with the number it limits

## Cut in v4, and why

- **The hero.** A 200pt zero was the three-metre hook; the reviewer read it as
  a dashboard. The three numbers now sit in the template's metric strip at the
  template's size, each with its baseline in its label.
- **The kayak listing.** A drawn, illustrative search result. Fig 1's real
  screenshot does the same job with a benchmark page, and the "same page, three
  ways" table gives the price the picture cannot.
- **Serif pull-quotes and the dark band.** Two extra type sizes and a second
  colour field; both were "format" in the reviewer's sense.
- **The two-column asymmetric grid.** Replaced by the template's own three
  columns (2+3 merged for the figure), so the sheet reads as one of the
  organisers' set.

## Audit trail — what changed between v2 and v4 because the ledger moved

| v2 printed | v4 prints | why |
|---|---|---|
| `12–14%`, three replicated modes | `10–14%`, six replicated modes | six-arm replicates landed (§477.2) |
| (13.7–35.3% in a docstring only) | `1.6–35.3%` | §452.2 retracted the tie-break-inflated figure |
| `2–36% of tasks these agents solve` | `the best single mode solves just 2–36%` | that is what `baseline_policy.sr_pct` measures (§450.8) |
| `2.1–4.2×` spoken only | on the poster, with `at least` | §453.2 wording; lower bound |

## The v2 lesson, kept

§495 recorded it: **checking the numbers is not checking the claims.** The
sentence that reached a printed v2 draft — *"always-cheapest is a cost floor, so
anything that protects success costs more"* — carried no number and so was never
queried, and had been retracted twice. v4 re-ran `known.py` on every number
*and* on every causal connective in the prose before building; that is how
`13.7` was caught in a docstring and `12–14` on the sheet.

## Build

```
.venv/bin/python3 deliverables/showcase/poster_figures.py   # Fig 1 raster + Fig 2 at column width
.venv/bin/python3 deliverables/showcase/build_poster.py     # asserts: not resized, nothing overruns, title 1 line, standfirst ≤ 2 lines, no stray '*', metric labels fit
soffice --headless --convert-to pdf --outdir deliverables/showcase deliverables/showcase/poster_jiaming_wei.pptx
```

Text is measured with Arimo/Noto Serif, but the PDF is exported by
LibreOffice, whose line advance for Liberation Sans is **1.20×** the point size
per unit of line spacing, not the 1.118 the face file gives. The v4 draft was
built with 1.118 and every section header sat on the paragraph above it; the
constant is now calibrated to the renderer (`LINE_HEIGHT` in `build_poster.py`).
