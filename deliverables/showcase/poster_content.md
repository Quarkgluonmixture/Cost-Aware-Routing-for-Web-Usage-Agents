# Poster content — Holistic AI × UCL CDI Showcase, 16 Sep 2026

**Author-facing source of truth.** The strings are inlined in `build_poster.py`;
any edit must be mirrored in both. Every number carries its scope, because the
poster is printed on silk and cannot be corrected. Deadline for the print PDF:
**4 Sep 2026** (organiser confirmed a replacement is accepted until then).

> **v8 (2026-09-03) — v6's look, v7's words.** v7 moved the loop diagram to the
> foot and put the numbers and prose first; the author's eye: *the previous one
> looked better*. It did — the top half had become text, and the one picture-
> rich element (the loop, with real screenshots) had been demoted, which also
> cut against the reviewer brief's *one main system diagram*. v8 keeps every
> v7 wording fix and puts the sheet back in v6's order: loop on top at v6
> size → number strip + the three comparison definitions → THE CATCH / why /
> Fig 3 / takeaway (left) · Fig 2 / verdicts / laptop bridge (right). Fig 2 is
> 6.8in tall and Fig 3 3.7in to make room. Lesson: a zero-preset reader
> optimises what it was asked to optimise (reading order, claims); the visual
> hierarchy is the author's call, and the eye was right.

> **v7 (2026-09-03) — result first, then why, then how.** A zero-preset GPT
> review of v6 (prompted only with the event, the template constraint and the
> laptop demo; no design rationale, no intended claim) found three things worth
> acting on and two not. **Accepted:** (1) the standfirst joined two different
> comparisons in one sentence — a researcher's first question; it is now two
> sentences, each naming its comparison, and the hindsight number is called
> *perfect hindsight* so nobody reads it as a deployed result; (2) *"WHY IT
> CANNOT BE LEARNED"* claimed too much — now *"WHY LEARNING THE CHOICE FAILS
> HERE"*, and *"confirms scarcity is the mechanism"* is now *"points to
> scarcity as the main bottleneck"* (alternatives — features, capacity, label
> noise — are not excluded on this sheet; §453.2 is the same direction);
> (3) the reading order was still a dissertation's (pipeline first): the three
> numbers and their comparisons now come first, the catch and the why second,
> the evidence beside them, and the loop diagram at the foot as *how we
> measured it*. Also from the review: a one-line key for the three comparisons
> under the number strip, with the rerun band attached to the hindsight line;
> the demo section cut to a bridge (task · ✓/✗ · steps · $, no frames) with
> the cherry-pick defused in words (*chosen so each view wins once; not how
> often each wins*); the catch pulled out as a callout; **Fig 3** (usable
> "which view" examples against the best single view's success rate, six
> VisualWebArena settings, parsed from `router_label_supply_diagnosis.md` +
> the 8-cell learnability JSON) — the figure the review asked for, and the
> figure the thesis's C5 was always missing on the sheet. **Rejected:** a
> bigger *0 of 8* (the reviewer brief said one type scale; v2's 200pt zero was
> what earned "dashboard"); dropping the title's breadth (kept, with *Today's*
> added and the scope line retained at the foot of TAKEAWAY).

> **v6 (2026-09-02, late) — no jargon, a loop that loops, a scoreboard.** Three
> author notes on v5: *jargon like "oracle" off* · *if the demo has it, does the
> poster need it too?* · *the agent loop is not quite there*. So: (1) every
> technical term is replaced by plain words — see the vocabulary table below;
> (2) the three big demo frames become a **scoreboard** (small frame · task ·
> ✓/✗ steps $ per way of seeing): the poster carries the demo's *results*, the
> laptop carries its *process*; (3) Fig 1 is now a real loop — the action
> arrow returns to the page, each way of seeing shows **what it really sends**
> (raw screenshot / element list / marked screenshot, the thesis F1 assets),
> and the agent card says what is held fixed.
>
> | jargon | on the poster |
> |---|---|
> | observation mode / representation | way of seeing (LOOK / READ / BOTH) |
> | hindsight oracle | best choice in hindsight |
> | learned router / triage | a learned choice |
> | fixed mode | one fixed choice |
> | always-cheapest | always using the cheapest way (kept — it is plain) |
> | pp | more tasks in 100 |
> | accessibility tree / AXTree / DOM | the page as text: its elements and labels |
> | SoM / marked screenshot | the screenshot with numbered boxes, plus the text |
> | episode | task attempt |
> | routing label / supervision | training example |
> | selective perception | learning when to look |
> | nested cross-validation / permutations | "scored only on tasks they never saw" (the rest is in SHOWCASE_PREP §4) |
> | in-sample | scored on its own training tasks |
> | log₂ ratio | log₂ ratio: 0 = same, 1 = double |

> **v5 (2026-09-02, evening) — "Look, read, or both?"** The poster now stands
> beside a laptop that replays the same task through three ways of seeing the
> page. That split decides what goes on silk: **the demo shows the phenomenon**
> (one task, three eyes, three behaviours, three bills, step by step); **the
> poster shows the system and the measurement** (where the decision sits in
> the agent loop, what it was worth in hindsight, whether it could be learned,
> why not). The reviewer brief from earlier the same day still holds — one main
> system diagram, compact, one type scale, the template's own skeleton — and
> the system diagram is now *the routing decision in the loop*, drawn in
> native shapes, not the thesis's "what each mode looks like" figure (the demo
> has that job).
>
> Vocabulary: **LOOK** = Vision (screenshot only, blue) · **READ** = DOM
> (accessibility-tree text only, orange; the three text-only variants P-text /
> P-prompt / P-SoM are READ variants) · **BOTH** = SoM (marked screenshot +
> text, green). Same three colours on the laptop.

> **The two baselines.** Half the numbers below are measured against the
> **best-success fixed mode** and half against **always-cheapest**. They are not
> interchangeable. Every number prints its own baseline.

---

## Title (two lines at the template's 41pt)

> **Look, read, or both?**
> **Today's web agents can't yet learn how to see a page**

Rejected: *The screenshot tax* — catchy and **false against F1's own numbers**
(Vision 3,123 tokens < DOM 3,314 < SoM 4,335; in most cells Vision is the
cheapest mode, F7). The expensive thing is *both*, not the screenshot.

## Standfirst (template band, 28pt, two lines)

> Perfect hindsight would solve up to 16 more tasks in 100 than the best single
> view. None of 8 learned choices beat always using the cheapest view on both
> success and cost.

Two sentences, two comparisons, each named. *Perfect hindsight* is the plain
word for the oracle. The v6 form ("Seen the right way, a page would let the
agent solve…") read as an attainable treatment effect; the review caught it.
`up to 16` = largest of 8 cells (`wa_reddit·B0`, +16.35pp); the strip says so.

## Fig 1 — the agent loop (full width, native shapes)

Header: `THE AGENT LOOP, AND WHERE THE DECISION SITS` — the first thing under the band (v8; v7 briefly had it at the foot). Five cards left to right,
block arrows between them, and a return arrow along the bottom from the agent
back to the page:

| card | prints |
|---|---|
| THE TASK + THE LIVE PAGE | “Show me the cheapest bike with red handlebars between $900–950.” · *Part of the intent is in the pictures, part in the text.* |
| WHO DECIDES HOW TO SEE IT? (dashed, accent) | **One fixed view** — the same view for every task · **Perfect hindsight** — knowing afterwards which view solved it · **A learned choice** — made before the task runs, from the page · `this box is what we measure` |
| LOOK / READ / BOTH | the screenshot only · 3,123 tokens on this page · *raw screenshot* / the page as text: its elements and labels · 3,314 tokens · +3 text-only variants · *the first six lines of the element list* / the screenshot with numbered boxes, plus the text · 4,335 tokens · *the marked screenshot* |
| THE AGENT, ONE STEP AT A TIME | **think → act** · click · type · scroll · go back · finish · *Only the page view changes; model, prompt, step budget and cost accounting stay fixed.* |
| WHEN IT STOPS | **✓ solved / ✗ not** · **$ for the attempt** · 8 website × model combinations · two public benchmarks · 6 views · 8,934 attempts |
| return arrow | the action changes the page → next step · up to 30 steps per task · the bill grows with every step |

> **Fig 1.** Everything is held fixed except how the page is shown. The dashed
> box is what this work measures — one fixed view, perfect hindsight, or a
> choice a model had to learn — each judged on **both** success and cost.

The three "what it sends" thumbnails and the element-list lines are the thesis
F1 assets (`fig_f1_motivating_example.py`: same page, dom/vision step-000
screenshots md5-identical), copied by `poster_figures.py`; the element lines
are shortened for display only (url tails and indentation dropped). Token
counts are F1's, re-read 2026-09-02. The example task is classifieds #17, the
scoreboard's BOTH-wins task. Episodes = 6 × (224×3 + 203×3 + 104×2).

## Under the loop · RESULTS ACROSS 8,934 TASK ATTEMPTS (full width)

Metric strip (template component; each label names its comparison):

| number | label |
|---|---|
| `+16.35 in 100` | PERFECT HINDSIGHT, BEST OF 8 · VS ONE FIXED VIEW |
| `0 of 8` | LEARNED CHOICES THAT BEAT ALWAYS-CHEAPEST |
| `1 of 8` | HINDSIGHT CHOICES THAT BEAT ALWAYS-CHEAPEST |

Under the strip, the three comparisons in one line each (three columns):

> **ONE FIXED VIEW** — the single view that solves most tasks in a setting, used
> for every task.
> **ALWAYS-CHEAPEST** — the single view that costs least on average in a
> setting, used for every task.
> **PERFECT HINDSIGHT** — for each task, the view that solved it, picked after
> the fact. An optimistic bound: rerunning one unchanged view flips 10–14% of
> outcomes and by itself gains 2.0–7.6 tasks in 100.

The rerun band moved here from the WHY prose so it sits beside the number it
qualifies (§450.10: the ceiling must print its rerun baseline next to it).

## Left column · THE CATCH → WHY LEARNING THE CHOICE FAILS HERE → Fig 3 → TAKEAWAY

Callout (template tinted box, accent bar, label `THE CATCH`):

> **The agents that would gain most from choosing a view solve the fewest tasks
> — so they produce the fewest examples to learn the choice from.**

> A training example for “which view” exists only when the agent solves a task.
> Here the best single view solves just **2–36%** of tasks, leaving typically
> **15–97** usable examples per setting — enough to train a classifier in only
> 2 of the 6 VisualWebArena settings.
>
> **Fig 3.** Usable “which view” examples against the best single view's success
> rate, one point per VisualWebArena setting. Examples exist only where tasks
> get solved.
>
> Shrinking the training data on purpose points to scarcity as the main
> bottleneck, and prices it: the failing settings would need at least
> **2.1–4.2×** more tasks than the benchmarks contain.

Fig 3 data (parsed, not typed): trainable "which view" labels from the first
table of `router_label_supply_diagnosis.md` — cls·B0 97 · red·B0 53 · cls·B1 55
· red·B1 24 · cls·B2 16 · red·B2 15 — against `baseline_policy.sr_pct` from
`router_triage_learnability_with_wa.json` (27.2 / 14.8 / 14.3 / 7.4 / 2.2 /
3.9 %). Filled = survives the min-class filter (cls·B0, cls·B1); hollow = not
(C5's "4 of 6"). Six points because C5 is a VisualWebArena diagnosis; the
caption says so and no 8-cell range sits next to it. *points to … the main
bottleneck* is weaker than the abstract's *confirms … the mechanism* on
purpose: the sheet cannot show that the alternatives were excluded.
`2–36%` = `baseline_policy.sr_pct`, 8-cell matched set (§450.8). `15–97` = C5.
`2.1–4.2×` = §453.2 wording, `at least` because it is a lower bound.

> **Learning how to see is not only a modelling problem: whether it can be
> learned depends on how good the agent producing the examples already is.** So,
> in this order: improve the agent, then collect reliable examples, then learn
> when to look.
>
> Measured in the 2–36% success regime we observed. This need not hold for
> stronger agents.

## Right column · Fig 2 → three verdicts → WATCH THE LAPTOP BESIDE THIS POSTER

Fig 2 — thesis F13 at the 362mm inner width of the figure box, plain-English
axes and legend (*learned choice · learned, scored on its own training tasks ·
perfect hindsight*):

> **Fig 2.** Every way of choosing a view, in every setting, compared with one
> fixed rule: **always use the cheapest view** (★). A win lands in the shaded
> region — cheaper *and* no worse. Always-cheapest is cheapest on average, not on
> every task, which is why a few points sit left of it. Learned choices are
> scored only on tasks they never saw.

> **Perfect hindsight would pay.** Picking the winning view for each task after
> the fact would solve **3.45 to 16.35 more tasks in every 100** than the best
> single view, and spend 1.6–35.3% less — in all 8 settings.
>
> **Nothing we trained could do it.** In **0 of 8** settings did a learned choice
> beat always using the cheapest view on both success and cost — and even
> perfect hindsight manages that in only **1 of 8**.
>
> **What survives is a bound, not a method.** Sending the tasks nobody solves to
> the cheapest view saves 9.5–30.6% at the same success in 8 of 8 — but that too
> needs hindsight, and plain always-cheapest usually saves more.

`1.6–35.3%` is the cost-aware tie-break figure (§452.2; `13.7–35.3%` is
RETRACTED). The 1-of-8 line is mandatory (§450.12). The third verdict keeps
§387.16.3's caveat.

The laptop bridge — results only, no frames (the screen has those):

> Same task, three views, different behaviour and different bills — step by
> step. Three illustrative tasks, chosen so each view wins once; **not** how
> often each wins.

| task | LOOK | READ | BOTH |
|---|---|---|---|
| Navigate to the item on this page whose image is taken during a sunset. | ✓ 2 steps · $0.007 | ✗ 21 steps · $0.096 | ✓ 3 steps · $0.014 |
| Navigate to my listing of the blue bike and change the price to $85.50 (including in the description). | ✗ 30 steps · $0.116 | ✓ 9 steps · $0.079 | ✗ 6 steps · $0.029 |
| Show me the cheapest bike with red handlebars between $900-950. | ✗ 19 steps · $0.075 | ✗ 6 steps · $0.025 | ✓ 7 steps · $0.037 |

> One recorded attempt per view, B0 · classifieds. Every ✓ / ✗ came out the same
> on an independent rerun; steps and bill differ run to run.

Parsed by `poster_figures.py` (`figures/demo_strip.json`), which **asserts**
each ✓/✗ is identical on the replicate run. Selection: 24 of 224 classifieds
tasks have three-way-different, rerun-stable outcomes; one per winner was
chosen. The kayak (task 0) fails in all three views in both runs — spoken
material only.

## Deliberately NOT on the poster

Per `THESIS_ONE_SENTENCE.md` "这篇论文不主张什么":

- ❌ that P-SoM substitutes for SoM (the value is complementarity)
- ❌ that "dropping the image barely costs anything"
- ❌ that routing is unlearnable for other models or benchmarks
- ❌ token/dollar cost silently re-labelled as energy or carbon
- ❌ the retracted AUROC 0.65–0.72 narrative (§394), the retracted "ceiling is
  entirely in cost" (§396.2), the 6-cell "1.7–3.3pp drop-one" hero, the
  retracted "cost floor" sentence (§476.4), the retracted `13.7–35.3%` (§452.2)
- ❌ that a demo task is "visual by nature" — the strip says *this run*, and the
  footnote says reruns flip 10–14% of outcomes
- ❌ that "red handlebars" can *only* be seen — the diagram says *part of the
  intent is in the pictures*, which is all the data licenses
- ❌ the `7.14 pp` vs `4.46–7.59 pp` calibration pair — spoken answer only

## Superseded designs

- **v2** (§495): exhibition object — 200pt hero zero, drawn kayak listing,
  dark takeaway band. Reviewer: "a dashboard".
- **v4** (§498): the template's skeleton with the thesis overview figure as the
  system diagram. Author: "still not a poster" — because it printed the
  system, the evidence *and* the phenomenon, and a sheet that carries all
  three reads as a compressed paper. v5 hands the phenomenon to the laptop.

## The v2 lesson, kept

**Checking the numbers is not checking the claims** (§495) — and **checking the
poster is not checking the documents beside it** (§498.3). v5 re-ran `known.py`
on every number and connective, and the strip's numbers are parsed, not typed.

## Build

```
.venv/bin/python3 deliverables/showcase/poster_figures.py   # Fig 2 at box width + demo strip (parsed, asserted) + thumbnails
.venv/bin/python3 deliverables/showcase/build_poster.py     # asserts: not resized · title 2×1 line · standfirst ≤ 2 lines · no stray '*' · metric labels fit · panels in their boxes
soffice --headless --convert-to pdf --outdir deliverables/showcase deliverables/showcase/poster_jiaming_wei.pptx
```

Text is measured with Arimo/Noto Serif at the renderer's line advance
(LibreOffice: **1.20×** per unit of line spacing, not the face file's 1.118 —
§498.2).
