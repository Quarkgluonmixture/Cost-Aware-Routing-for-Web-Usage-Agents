# Poster content — Holistic AI × UCL CDI Showcase, 16 Sep 2026

**Author-facing source of truth.** The strings are inlined in `build_poster.py`;
any edit must be mirrored in both. Every number carries its scope, because the
poster is printed on silk and cannot be corrected. Deadline for the print PDF:
**4 Sep 2026** (organiser confirmed a replacement is accepted until then).

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
> **Web agents can't yet learn how to see a page**

Rejected: *The screenshot tax* — catchy and **false against F1's own numbers**
(Vision 3,123 tokens < DOM 3,314 < SoM 4,335; in most cells Vision is the
cheapest mode, F7). The expensive thing is *both*, not the screenshot.

## Standfirst (template band, 28pt, two lines)

> Seen the right way, a page would let the agent solve up to 16 more tasks in
> 100 than the best fixed choice — yet nothing we trained beat always using the
> cheapest way on both success and cost.

Both baselines named; no pronoun. `up to 16` = largest of 8 cells
(`wa_reddit·B0`, +16.35pp); the metric strip says so.

## Fig 1 — the agent loop (full width, native shapes)

Header: `THE AGENT LOOP, AND WHERE THE DECISION SITS`. Five cards left to right,
block arrows between them, and a return arrow along the bottom from the agent
back to the page:

| card | prints |
|---|---|
| THE TASK + THE LIVE PAGE | “Show me the cheapest bike with red handlebars between $900–950.” · *Part of the intent is in the pictures, part in the text.* · the start page (raw screenshot) |
| WHO DECIDES HOW TO SEE IT? (dashed, accent) | **One fixed choice** — the same way of seeing for every task · **Best choice in hindsight** — knowing afterwards which way solved it · **A learned choice** — made before the task runs, from what the page looks like · `this box is what we measure` |
| LOOK / READ / BOTH | the screenshot only · 3,123 tokens on this page · *raw screenshot* / the page as text: its elements and labels · 3,314 tokens · three text-only variants share this · *the first six lines of the element list* / the screenshot with numbered boxes, plus the text · 4,335 tokens · *the marked screenshot* |
| THE AGENT, ONE STEP AT A TIME | **think → act** · click · type · scroll · go back · finish · *Same model, same prompt, same step budget and same cost accounting whichever way it sees — only what it is shown changes.* |
| WHEN IT STOPS | **✓ solved / ✗ not** · **$ for the whole attempt** · 8 website × model combinations · two public benchmarks · 6 ways of seeing · 8,934 task attempts |
| return arrow | the action changes the page → next step · up to 30 steps per task · the bill grows with every step |

> **Fig 1.** Everything is held fixed except how the page is shown to the agent.
> The dashed box is what this work measures — one fixed way of seeing, the best
> way in hindsight, or a way a model had to learn to choose — each judged on
> **both** success and cost. The laptop beside this poster replays the loop on
> the three tasks below.

The three "what it sends" thumbnails and the element-list lines are the thesis
F1 assets (`fig_f1_motivating_example.py`: same page, dom/vision step-000
screenshots md5-identical), copied by `poster_figures.py`; the element lines
are shortened for display only (url tails and indentation dropped). Token
counts are F1's, re-read 2026-09-02. The example task is classifieds #17, the
scoreboard's BOTH-wins task. Episodes = 6 × (224×3 + 203×3 + 104×2).

## Column 1 · WHAT THE SCREEN BESIDE YOU SHOWS (the scoreboard)

Three tasks, a small frame each, and every way of seeing's real outcome as
`✓/✗ · steps · $` under LOOK / READ / BOTH column heads. **Nothing here is typed
by hand**: `poster_figures.py` parses `figures/demo_strip.json`
from the episode summaries and **asserts** that each mode's ✓/✗ is identical on
the independent replicate run — a task that flips on rerun cannot reach the
sheet. Numbers shown are the canonical run's (`A`); costs are
`total_billed_cost_usd`.

| task | intent | LOOK | READ | BOTH | frame printed |
|---|---|---|---|---|---|
| 130 | Navigate to the item on this page whose image is taken during a sunset. | ✓ 2 steps $0.007 | ✗ 21 steps $0.096 | ✓ 3 steps $0.014 | LOOK replicate, last page: the sunset boat |
| 76 | Navigate to my listing of the blue bike and change the price to $85.50 (including in the description). | ✗ 30 steps $0.116 | ✓ 9 steps $0.079 | ✗ 6 steps $0.029 | READ replicate, last page: “Great! We've just updated your listing · 85.50 $” |
| 17 | Show me the cheapest bike with red handlebars between $900-950. | ✗ 19 steps $0.075 | ✗ 6 steps $0.025 | ✓ 7 steps $0.037 | READ replicate, last page: a $900 bike whose handlebars are not red — price matched, colour unchecked |

Selection: of 224 classifieds tasks, **24** have three-way-different outcomes
that agree between canonical and replicate runs (patterns, LOOK READ BOTH:
✓✗✓ 13 · ✗✗✓ 4 · ✓✗✗ 3 · ✗✓✓ 2 · ✗✓✗ 2). One task per "who wins" pattern was
chosen. The kayak (task 0) fails in all three modes in both runs — it is spoken
material, not demo material. BOTH's artifacts were cleaned in both SoM runs, so
task 17's frame comes from the READ run; its caption says so implicitly (the
bike shown is the *wrong* one).

> One recorded attempt per way of seeing, B0 · classifieds. Every ✓ / ✗ came out
> the same on an independent rerun; steps and bill differ run to run — across
> all tasks a rerun flips 10–14% of outcomes.

## Columns 2–3 · RESULTS ACROSS 8,934 TASK ATTEMPTS

Metric strip (template component; each label names its baseline):

| number | label |
|---|---|
| `+16.35 in 100` | IN HINDSIGHT, BEST OF 8 · VS ONE FIXED CHOICE |
| `0 of 8` | LEARNED CHOICES THAT BEAT ALWAYS-CHEAPEST |
| `1 of 8` | HINDSIGHT CHOICES THAT BEAT ALWAYS-CHEAPEST |

Fig 2 — thesis F13 at the 362mm inner width of the figure box:

> **Fig 2.** Every way of choosing, in every setting, compared with one fixed
> rule: **always use the cheapest way of seeing** (★). A win lands in the shaded
> region — cheaper *and* no worse. Always-cheapest is cheapest on average, not on
> every task, which is why a few points sit left of it. Learned choices are
> scored only on tasks they never saw.

Axes: *more tasks solved than always-cheapest, per 100* / *cost relative to
always-cheapest (log₂ ratio: 0 = same, 1 = double)*. Legend: learned choice ·
learned, scored on its own training tasks · best choice in hindsight.

Three verdicts (body size, bold lead):

> **Choosing well would pay.** Picking the right way of seeing for each task
> after the fact would solve **3.45 to 16.35 more tasks in every 100** than the
> best single fixed choice, and spend 1.6–35.3% less — in all 8 settings.
>
> **Nothing we trained could do it.** In **0 of 8** settings did a learned choice
> beat simply always using the cheapest way on both success and cost — and even
> choosing with hindsight manages that in only **1 of 8**.
>
> **What survives is a bound, not a method.** Sending the tasks nobody solves to
> the cheapest way saves 9.5–30.6% at the same success in 8 of 8 — but that too
> needs hindsight, and plain always-cheapest usually saves more.

`1.6–35.3%` is the cost-aware tie-break figure (§452.2 *replaced_by*; the older
`13.7–35.3%` is RETRACTED). The 1-of-8 line is mandatory (§450.12). The third
verdict keeps §387.16.3's caveat.

## Column 1 · WHY IT CANNOT BE LEARNED

> A training example for “which way of seeing” exists only when the agent
> solves a task. Here the best single way solves just **2–36%** of tasks,
> leaving typically **15–97** usable examples per setting — **the agents that
> would gain most from choosing produce the fewest examples to learn from.**
>
> Shrinking the training data on purpose confirms scarcity is the mechanism,
> and prices it: the failing settings would need at least **2.1–4.2×** more
> tasks than the benchmarks contain.
>
> Rerunning **one unchanged way of seeing** flips **10–14%** of outcomes and by
> itself buys **2.0–7.6** more solved tasks in 100 (B0 · classifieds, six
> repeated ways, n=224); every gain on this sheet is read against that band, not
> against zero.

`2–36%` = `baseline_policy.sr_pct`, 8-cell matched set (§450.8; the 6-cell
figure 2–27 must not sit next to "8 settings"). `15–97` = C5, 4 of 6 VWA cells
(denominator dropped on the sheet; in `SHOWCASE_PREP.md §5`). `2.1–4.2×` =
§453.2 wording, `at least` because it is a lower bound. `10–14%` = six-arm
(§477.2); `2.0–7.6 pp` = §450.10.

## Column 1 · TAKEAWAY

> **Learning how to see is not only a modelling problem: whether it can be
> learned depends on how good the agent producing the examples already is.** So,
> in this order: improve the agent, then collect reliable examples, then learn
> when to look.
>
> Measured in the 2–36% success regime we observed. This need not hold for
> stronger agents.

The thesis sentence is *"Routing is not only a model-selection problem: its
learnability depends on the competence of the agent producing the labels."*
Same claim, same scope; only the words changed.

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
