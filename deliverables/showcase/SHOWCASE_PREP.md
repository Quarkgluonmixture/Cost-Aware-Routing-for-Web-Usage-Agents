# Showcase prep — everything to send and say

**Holistic AI × UCL CDI MSc Research Showcase · Wed 16 Sep 2026 · UCL Centre for AI**

Poster: `poster_jiaming_wei.pdf` — **v9.10, 2026-09-04** (`poster_v9_jiaming_wei.pdf`; v8.3 kept as `poster_jiaming_wei.pdf`) (A1 594×841mm, 1 page, fonts
embedded, figures 300–350dpi). Numbers, their scope and the version history:
`poster_content.md` (read its v9/v8.3 headers first; the v9 rebuild and its asset library are in `POSTER_ASSET_LIBRARY.md`). `EXHIBITION_PASS.md` is the
superseded v2 rationale, kept for the record.

Vocabulary at the board = the poster's: **view** (LOOK / READ / BOTH), **one
fixed view**, **always-cheapest**, **perfect hindsight**, **a learned choice**,
**training examples**, **more tasks in 100**. The thesis words (mode, oracle,
router, labels, pp) are for researchers who use them first.

---

# 0. Checklist → see `ROADMAP.md` (2026-09-11)

The day-by-day plan, the decisions (D1–D10) and every phase's done-condition now
live in **`ROADMAP.md`**. Phase gates, for a glance:

| Phase | When | Closes when |
|---|---|---|
| 0 | Fri 11 Sep | four logistics questions sent to Zekun (`ROADMAP.md §6`); D1 / D3 confirmed by the author |
| 1 · demo | 11–12 Sep | §506.10 fixes (1A 2A 3B) + `?task=130&autoplay=0` talk mode + portable rebuilt + `talk_130.mp4`; committed |
| 2 · script | 12–13 Sep | `talk_content.md` (7 slides, 1,150–1,300 words) + §2 / §3 / §5 below rewritten to the **v9.10** panel order; /stress spot-check clean |
| 3 · slides | template day, latest Mon 14 Sep | pptx + PDF open on quark without substitution / overflow; sent to Zekun by Tue noon if slides are collected |
| 4 · rehearsal | 14–15 Sep | two consecutive timed runs ≤ 9:30 with the browser switch |
| 5 · packing | Tue 15 Sep evening | folder + USB + adapters + one end-to-end live run through the tunnel |
| 6 · the day | Wed 16 Sep | timeline and fallback ladder in `ROADMAP.md §3` |

⚠️ **§2, §3 and §5 below were written against poster v8 and are stale for the
printed v9.10** (six panels; no number strip, no "THE CATCH", no Fig 2 / Fig 3).
They are rewritten in Phase 2. Until then, use them only for the *numbers and
their scopes* (§5) and the *answers* (§4, §6), not for the walking order.

Both deliverables go **to Zekun directly** — the organiser stated they cannot
receive files.

---

# 1. Message to Zekun (Slack DM or email) — copy as is

**To:** `zekun.wu@holisticai.com`
**Attach:** `poster_jiaming_wei.pdf`
**Subject:** Showcase poster + oral slot — Jiaming Wei

> Hi Zekun,
>
> Attached is my poster for the 16 September showcase, A1 portrait, print-ready
> PDF (fonts embedded, figures at 300dpi). Built from the template without
> resizing the slide.
>
> **Look, read, or both? Today's web agents can't yet learn how to see a page**
>
> I'd also like to put my name in for one of the oral slots. In 12 minutes I'd
> cover three things in order:
>
> A web agent can *look* at a page (screenshot), *read* it (the page as text)
> or do both — and they cost different amounts. I ran the same tasks through six
> such views, four of them screenshot-free, across 8 website × model settings
> and ~8,900 task attempts.
>
> (1) Perfect hindsight finds a real opportunity: up to 16 more tasks solved in
> 100 than the best single view. (2) But nothing we trained could take it: in 0
> of 8 settings did a learned choice beat always using the cheapest view on both
> success and cost — and even perfect hindsight manages that in only 1 of 8, so
> this isn't just a weak classifier. (3) The reason is the catch: the best single
> view solves only 2–36% of tasks, and a training example for "which view" only
> exists when a task gets solved — the agents that would gain most from choosing
> produce the fewest examples to learn from.
>
> Beside the poster I'd like to run a small laptop demo: three real tasks
> replayed the three ways (one only *look* solves, one only *read*, one only
> *both*). Would a table and a power socket by the board be possible?
>
> Happy to take a shorter slot if that helps the schedule.
>
> Best,
> Jiaming

---

# 2. Board walk — 90 seconds

> ⚠️ **Stale for v9.10 (2026-09-11)** — written for the v8 sheet. Rewrite in `ROADMAP.md` Phase 2 to the six-panel order; the printed poster is described in `ROADMAP.md §0`.

Sheet order: loop diagram → number strip → catch / why / Fig 3 (left) · Fig 2 /
laptop (right). The walk skips the loop unless they ask for the method.

**① Title + standfirst**
> A web agent can look at a page, read it, or both. Choosing right would help a
> lot — in perfect hindsight, up to 16 more tasks in a hundred over the best
> single view. But nothing we trained beat simply always using the cheapest
> view on both success and cost.

**② The number strip and the three definitions under it**
> Three comparisons, and they are different: the best single view, the cheapest
> single view, and perfect hindsight. Hindsight is a bound, not a method — and
> an optimistic one: rerunning the same view flips ten to fourteen percent of
> outcomes by itself.

**③ THE CATCH (left)**
> Here's why learning fails. A training example only exists when a task gets
> solved, and these agents solve two to thirty-six percent of tasks. The agents
> that would gain most from choosing produce the fewest examples to learn from.

**④ Fig 3**
> Six settings: the more the agent solves, the more examples there are, and only
> two settings had enough to train a classifier at all.

**⑤ Fig 2 (right)**
> Every way of choosing, against always-cheapest. The shaded region is a win —
> cheaper and no worse. Zero learned choices land there; even perfect hindsight
> lands there in one of eight.

**⑥ The laptop**
> Watch it: same task, three views, different behaviour and bills. Three
> illustrative tasks, one per winner — not how often each wins.

**⑦ The loop at the top, if they want the method**
> Only the page view changes; the dashed box is what we measure.

**If they only have 20 seconds**, use ① ③ ⑤ and stop.

---

# 3. Oral slot — 12 minutes (only if confirmed)

> ⚠️ **Stale for v9.10 (2026-09-11)** — written for the v8 sheet. Rewrite in `ROADMAP.md` Phase 2 to the six-panel order; the printed poster is described in `ROADMAP.md §0`.

Same spine as the board walk, four blocks of roughly three minutes. Do not add
new results; add *why each step was necessary*.

### Block 1 — The problem (3 min)
Open with the kayak task on a slide (spoken material only — it fails in all
three views in both runs, so it is not on the poster or in the demo). Cheapest
is red, answer is blue.
- Six views; four send no image at all, one sends text + screenshot, one sends
  the screenshot only.
- 8 website × model settings across VisualWebArena and WebArena, ~8,900 task
  attempts.
- The question: is the expensive context necessary at every step, and can you
  tell in advance?

### Block 2 — The ceiling, and the ruler (3 min)
- Perfect hindsight: +3.45 to +16.35pp over the best single fixed view, at
  1.6–35.3% lower cost, same direction in all 8. (Say 1.6, not the old 13.7:
  §452.2 retracted it — on two cells the "best single mode" was an SR tie and the
  list-order tie-break picked the dearer one.)
- **Then immediately the correction** — this is the part that makes it honest:
  rerunning one view on the same tasks flips 10–14% of outcomes (six
  replicated views on cls·B0). Adding a
  different representation buys 7.14pp; rerunning the one you have buys
  4.46–7.59pp. Not distinguishable.
- Takeaway of the block: the ceiling is real but smaller than it first looks, and
  we report the band so anyone can apply the same correction.

### Block 3 — The failure, and why it isn't the learner (3 min)
- Five routing policies, fully nested cross-validation, 10,000 bundle
  permutations. 0 of 8 beat always-cheapest on both success and cost.
- Two controls, both needed: always-cheapest as a fixed-policy baseline, and
  label-shuffle as a null. One without the other misses a different cell.
- **The calibration that matters**: perfect hindsight (the oracle) only reaches
  the win region in 1 of 8 either. So the barrier isn't primarily the estimator.

### Block 4 — The mechanism, and what it implies (3 min)
- Label supply: a "which view" example exists only when a task is solved; base
  success is 2–36%; 15–97 usable examples per VWA cell, enough to train in only
  2 of 6 — this is Fig 3 on the poster.
- Three independent escapes all blocked: continuous labels (VWA score is binary),
  pooling (contradiction rates 56–57%), relabelling (cost tiers don't create new
  solve events).
- Undersampling control: more data pushes the learner toward the oracle, and the
  oracle is already outside the win region in 7 of 8. Priced it: the four failing
  cells need 2.1–4.2× more tasks.
- Close on the design implication, not on the negative result.

---

# 4. Questions you will get

**"Could 0 of 8 just be too little training data?"** ← most likely question
> We tested exactly that. Learning curves are still rising and unregularised
> models beat a permutation floor, so there *is* signal. But more data pushes the
> learner toward the oracle, and the oracle itself only reaches the win region in
> 1 of 8. More data can't cross a line the oracle doesn't cross. We priced it
> anyway: the four failing settings need 2.1 to 4.2 times more tasks. That's a
> specification, not an impossibility.

**"What does always-cheapest actually mean — per task or fixed?"**
> Fixed: the single view that costs least on average in that setting, used for
> every task. Not a per-task pick. The key under the number strip says so.

**"Is +16.35 something a real system reached?"**
> No — it's perfect hindsight, an upper bound. The poster calls it that, and it
> prints the rerun band next to it. What a real learned choice reached is the
> plot: none in the win region.

**"Why does the view matter so much for a given task?"**
> We measured it, we didn't explain it — the mechanism work is out of scope
> here. What the demo shows is the pattern: text-only views fail on intents
> that live in the picture, image-only views wander on intents that live in the
> text, and BOTH pays for both.

**"So what should a production agent do today?"**
> On these benchmarks, always using the cheapest view is hard to beat on both
> counts, and BOTH is the dearest. If you must pick one view, pick it for your
> task mix. A learned per-task choice isn't there yet — and the poster says why:
> improve the agent first, collect reliable examples, then learn when to look.

**"Isn't always-cheapest too weak a baseline?"**
> The opposite — it's hard to beat, because it's the cheapest fixed policy on
> average. Note *on average*, not a per-episode floor: on an individual task,
> switching mode can finish sooner and cost less than exhausting the step budget
> cheaply. That's exactly where the points left of zero on the plot come from.

**"With 10–14% rerun variance, are your results just noise?"**
> That's precisely why we measured it first and then used it as a ruler. Adding a
> new representation buys 7.14pp; a plain rerun of the same one buys 4.46–7.59pp.
> They overlap. So we don't use that ceiling as a positive claim — what survives
> the correction is the cost result and the label-supply mechanism.

**"What about stronger models?"**
> That's what the conclusion points at, and it's falsifiable. We don't claim
> routing is unlearnable in general — only that inside the 2–36% success regime
> we observed, the supervision isn't there. Label supply and routing opportunity
> rise together, so a stronger agent should be the test.

**"Why six modes? Why those?"**
> They're a 2×2 of text format × prompt style, plus the two that involve the
> screenshot. The four text-side ones are verified image-free from the step
> records — `image_payload_bytes == 0`. That lets us separate "what's in the
> text" from "is there a picture", which a DOM-vs-screenshot comparison can't.

**"Is this deployable?"**
> Not as it stands, and the poster says so. Both ceilings are retrospective. The
> nearest thing to deployable is the triage result — send the tasks nobody solves
> to the cheapest mode — but that's also a hindsight bound, and in most settings
> plain always-cheapest saves more anyway. What makes triage interesting isn't
> the saving, it's that its label is far easier to supply.

---

# 5. Number crib — every figure with its baseline

> ⚠️ **Stale for v9.10 (2026-09-11)** — written for the v8 sheet. Rewrite in `ROADMAP.md` Phase 2 to the six-panel order; the printed poster is described in `ROADMAP.md §0`.

**Never quote a number without the phrase in the "must say" column.** Two
different baselines are in play and they are not interchangeable.

Everything here is on the poster except where marked — so if you say it, the
visitor can find it. The one exception is flagged in the table.

| Number | What it is | Must say |
|---|---|---|
| `+3.45 to +16.35 pp` | hindsight picker's success gain (poster: "3.45 to 16.35 more tasks in every 100") | **vs the best single fixed choice**; percentage *points*; retrospective |
| `1.6–35.3%` | that picker's cost saving | same baseline as above; **cost-aware tie-break** — `13.7–35.3%` is RETRACTED (§452.2), never say it |
| `0 of 8` | learned routers that win | **vs always-cheapest**, and **on both** success and cost |
| `1 of 8` | perfect-hindsight choices that win (the poster's word for the oracle) | same bar — this line is mandatory, never quote 0/8 alone. The strip labels say **won on both success and cost**; +16.35 is **success only** |
| `2–36%` | base success rate | 8-cell matched set. **The 6-cell VWA figure is 2–27 — never say that next to "8 settings"** |
| `10–14%` | rerun discordance | B0 · classifieds, n=224, **six** replicated modes (§477.2; v2 said 12–14 / three) |
| `2.0–7.6 pp` | what one rerun buys by itself | same cell; printed beside 10–14% — the companion of every ceiling number |
| `7.14 pp` vs `4.46–7.59 pp` | new view vs a rerun | same cell; the point lands *inside* the band. **Not on the poster** — spoken answer only |
| `9.5–30.6%` | triage cost saving, 8 of 8 | **vs the best-success fixed mode**; hindsight bound; in most pairs always-cheapest saves more |
| `15–97` | usable "which view" examples | per VWA cell; the poster says *enough to train a classifier in only 2 of the 6 VisualWebArena settings* and shows the six points in Fig 3 |
| `2.1–4.2×` | corpus the failing cells would need | on the poster with **at least** (lower bound; undersampling control, F16). reddit needs the most, 846 tasks |
| strip: 130 · 76 · 17 | the three demo tasks' ✓/✗, steps, $ | **one recorded run** (B0 · classifieds, canonical); ✓/✗ identical on the replicate, steps and $ are not. Never say a task "is visual" — say "in this run" |
| `24 of 224` | tasks whose three-way outcome differs and is rerun-stable | selection pool for the demo; **not on the poster** |
| Fig 3: 97 · 53 · 55 · 24 · 16 · 15 | usable "which view" examples per VisualWebArena setting, vs best-single-view SR 27.2 / 14.8 / 14.3 / 7.4 / 2.2 / 3.9 % | six settings, **not eight**; filled = enough to train (cls·B0, cls·B1) |
| `8,934` | task attempts (the poster's word for episodes) | 6 views × (224×3 + 203×3 + 104×2) |
| demo meters: cost · time · CO₂e ≈ | under each lane: billed $, the run's recorded wall-clock, an estimated CO₂e **range** | **one recorded run** each. CO₂e is **estimated from tokens**, never measured (the model runs in an AWS London datacentre; no local draw exists). Say "roughly", give the range, never one number |
| demo arrow: 130 → READ ✗ · 76 → READ ✓ · 17 → a fourth view | the view a learned choice picked, **trained without that task** | three tasks are an illustration, not a score — **the score is the poster's 0 of 8**. On 17 the picked view's own two runs split ✓ once, ✗ once |
| live tab | a visitor's own task, run now in all three views, each capped at 12 steps | **not scored**, and not part of any number on the poster |

---

# 6. Things not to say

These are all overclaims that were removed from the poster on purpose. Under
pressure they are easy to say by accident.

| ❌ Don't say | ✅ Say instead |
|---|---|
| "always-cheapest is a cost floor, so anything better costs more" | "cheapest fixed policy **on average**, not a per-episode floor" |
| "the problem isn't the learner" | "learning failure alone can't explain it — the oracle fails the same test in 7 of 8" |
| "a stronger agent will overturn this" | "this conclusion need not hold for stronger agents" |
| "routing doesn't work" / "routing is unlearnable" | "not learnable **in the 2–36% success regime we observed**" |
| "we save 30% of cost" | "up to 30.6%, in hindsight, against the best-success fixed mode" |
| "dropping the screenshot is basically free" | "the image-free modes are **complementary** to the screenshot ones, not substitutes" |
| "P-SoM can replace SoM" | never — the claim is complementarity |
| "the router gets 0.65–0.72 AUROC" | retracted (§394); one cell is 0.483, below chance |
| "+16 % more tasks" | "16 more tasks **in 100**" (the poster's form) or "+16 percentage points" — on a 2% base, "%" inflates it wildly |
| "it cannot be learned" / "scarcity is **the** mechanism" | "learning the choice **fails here**" / "scarcity is the **main bottleneck**" — the sheet cannot show the alternatives were excluded |
| "the oracle", "the router", "modes", "pp" (to a non-researcher) | "perfect hindsight", "a learned choice", "views", "more tasks in 100" — the poster's words |

| "the router picks the right view" | "a learned choice picked READ on two of these; it was right once — and across 8 settings it never beat always-cheapest" |
| "this live run shows it works" | "live runs are an unscored demo; every number on the poster comes from recorded runs" |
| "LOOK emitted 0.3 grams on task 130" | "roughly 0.02 to 0.3 grams — estimated from tokens, not measured" |

Also avoid: quoting token or dollar cost as *measured* energy or carbon — the demo's
CO₂e row is an estimate built from tokens, and says so with "≈"; describing `dom`
mode as raw HTML (it is an accessibility tree).

---

# 7. On the day

| Time | |
|---|---|
| 09:00 | Poster set-up |
| 09:30 | Registration & refreshments |
| 10:00 | **Exhibition opens — sticker voting starts** |
| 10:30–12:35 | Opening remarks, alumni talks, PhD route session |
| 12:35 | Lunch + posters (long open stretch) |
| **13:15–14:35** | **Authors at their boards** ← the votes are won here |
| 14:35 | Break — **voting closes** |
| 14:45 | Student presentations (your slot, if confirmed) |
| 15:30 | Keynote |
| 16:30 | Awards (£300 / £200 / £100) |
| 17:15 | Take-down |

**Voting closes at 14:35, before the oral slots.** The 13:15–14:35 stretch at the
board is what decides the prize, not the talk. Prioritise being at the board and
running the 90-second walk repeatedly.

Prize note: 1st place includes mentorship to write the work up as a workshop
paper, and the cover of the showcase proceedings.

---

# 8. The laptop demo — plan (for 16 Sep, not for the print deadline)

**What it shows.** "Look, read, or both?": the same task replayed in three
synchronised columns (LOOK / READ / BOTH) — the poster's bridge table (right
column) is the scoreboard for exactly these three tasks — stepping through the agent's
screenshots with the click drawn (LOOK: click coordinates; READ: the element's
bbox; BOTH: the numbered mark), the agent's one-line *thought* under each
frame, and a running **$** counter per column. Three tasks, one per winner:
130 (look), 76 (read), 17 (both); optionally the kayak (task 0) as "all three
fail" — honest, and it walks the visitor to the poster's WHY.

**Update 2026-09-10:** a fourth tab, *try your own*, now runs a visitor's task live
on top of the replay (user decision). Agent and site both run on DGX (an arm64
rebuild of the classifieds site, its own stack on 127.0.0.1:9981 — never the A100);
it is unscored and one session at a time. The replay remains
the default and still needs no network — see `demo/README.md` → *Live*.

**Rules.** Replay, never live (site on the A100, venue network unknown). Only
tasks whose three-way outcome is identical on the replicate run (24 of 224
qualify; `poster_figures.py` asserts it). Label every column "one recorded run".
Re-recording a run to obtain artifacts is fine and is labelled as a run;
hand-made trajectories are not.

**Data.** LOOK and READ: per-step `screenshot.png` + `observation_dom.txt` and
step records (`element_bbox`, `coordinate`, `thought`) exist for all 224 tasks
in `results/repro_replicates/B0_{vision,dom}_classifieds_*_clean_replicate`.
BOTH: both SoM runs had their artifacts cleaned → **re-record tasks 130/76/17
in SoM once, artifacts kept**, on a site nobody else is using. The B0·reddit
replicate chain owns the A100 until ~7 Sep (one site chain per host); quark's
docker was down on 2 Sep. Window: 7–15 Sep.

**Build.** A static HTML page (no server): `demo/index.html` reading a JSON
per task with frames + actions; keyboard → next step, auto-play loop for when
nobody is at the board. Reuse the poster's three colours. Extra sites
(reddit / shopping) only if a stable three-way task exists there with
artifacts — do not re-record for the sake of variety.
