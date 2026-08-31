# Showcase prep — everything to send and say

**Holistic AI × UCL CDI MSc Research Showcase · Wed 16 Sep 2026 · UCL Centre for AI**

Poster: `poster_jiaming_wei.pdf` (A1 594×841mm, 1 page, fonts embedded, 300dpi).
Numbers and their scope: `poster_content.md`. Design rationale: `EXHIBITION_PASS.md`.

---

# 0. Checklist

| When | What | Status |
|---|---|---|
| **Now** | Email Zekun: poster PDF + oral slot pitch (§1) | ☐ |
| Wed 2 Sep | Hard deadline for both | — |
| Before 16 Sep | Rehearse the 90-second board walk (§2) until it needs no notes | ☐ |
| Before 16 Sep | Only if a slot is confirmed: expand §3 to 12 minutes | ☐ |
| Before 16 Sep | Read §5 and §6 twice — the numbers and the things not to say | ☐ |
| 16 Sep 09:00 | Poster set-up | ☐ |

Both deliverables go **to Zekun directly** — the organiser stated they cannot
receive files.

---

# 1. Email to Zekun — copy as is

**To:** `zekun.wu@holisticai.com`
**Attach:** `poster_jiaming_wei.pdf`
**Subject:** Showcase poster + oral slot — Jiaming Wei

> Hi Zekun,
>
> Attached is my poster for the 16 September showcase, A1 portrait, print-ready
> PDF (fonts embedded, figures at 300dpi). Built from the template without
> resizing the slide.
>
> **Can a web agent learn when a screenshot is worth the cost?**
>
> I'd also like to put my name in for one of the oral slots. In 12 minutes I'd
> cover three things in order:
>
> Screenshots help web agents but cost more, and nobody checks whether each step
> needs one. Using a real VisualWebArena task — *"find me the cheapest blue
> kayak"*, where "blue" can be visual and "cheapest" is textual — I ran six
> observation modes, four of them screenshot-free, across 8 benchmark–model
> settings and ~8,900 episodes.
>
> (1) In hindsight there is real opportunity: up to +16.35pp over the best single
> fixed mode. (2) But learned, zero of eight routers were both more successful
> and cheaper than always picking the cheapest mode — and only one of eight
> *hindsight* oracles clears that bar either, so this isn't just a bad
> classifier. (3) The deeper problem is that these agents solve only 2–36% of
> tasks, so the agent itself produces almost no supervision for learning when to
> route. Selective perception becomes learnable only once the underlying agent is
> competent enough to generate reliable labels.
>
> Happy to take a shorter slot if that helps the schedule.
>
> Best,
> Jiaming

---

# 2. Board walk — 90 seconds

The poster's layout **is** the script. Your hand moves top → left → right →
lower-left → bottom, never backwards.

**① Point at the title**
> Screenshots help web agents, but they cost money at every single step. So I
> asked whether the agent can learn when looking is actually worth paying for.

**② Point at the kayak listing (left)**
> Here's the problem in one task. Find the cheapest *blue* kayak. The cheapest
> one is red. So you need to look, and you need to read — and which half matters
> changes task by task.

**③ Point at `+16.35 pp` (top left)**
> If you knew afterwards which of six ways of seeing the page would solve each
> task, you'd solve up to 16 more tasks in a hundred than the best single fixed
> choice. So there is genuinely something worth routing to.

**④ Point at the big orange `0` (top right)**
> But learned — nothing. Zero of eight settings produced a router that was both
> more successful *and* cheaper than just always using the cheapest mode.

**⑤ Point at the plot (right)**
> And it isn't just a bad classifier. Only one of eight *hindsight oracles*
> clears the same bar. The shaded region is where a win would sit. It's empty.

**⑥ Point at the paradox (lower left)**
> Here's why. These agents solve between 2 and 36 percent of tasks, and a
> routing label only comes into existence when a task gets solved. The agents
> that would gain most from routing are exactly the ones producing the fewest
> labels to learn it.

**⑦ Point at the bottom band**
> So the order matters: improve the agent, get reliable supervision, *then* learn
> selective perception. Routing isn't only a model-selection problem — its
> learnability depends on how good the agent underneath already is.

**If they only have 20 seconds**, use ①②④⑥ and stop.

---

# 3. Oral slot — 12 minutes (only if confirmed)

Same spine as the board walk, four blocks of roughly three minutes. Do not add
new results; add *why each step was necessary*.

### Block 1 — The problem (3 min)
Open with the kayak task on screen. Cheapest is red, answer is blue.
- Six observation modes; four send no image at all, one sends text + screenshot,
  one sends the screenshot only.
- 8 benchmark–model settings across VisualWebArena and WebArena, ~8,900 episodes.
- The question: is the expensive context necessary at every step, and can you
  tell in advance?

### Block 2 — The ceiling, and the ruler (3 min)
- Hindsight picker: +3.45 to +16.35pp over the best single fixed mode, at
  13.7–35.3% lower cost, same direction in all 8.
- **Then immediately the correction** — this is the part that makes it honest:
  rerunning one mode on the same tasks flips 12–14% of outcomes. Adding a
  different representation buys 7.14pp; rerunning the one you have buys
  4.46–7.59pp. Not distinguishable.
- Takeaway of the block: the ceiling is real but smaller than it first looks, and
  we report the band so anyone can apply the same correction.

### Block 3 — The failure, and why it isn't the learner (3 min)
- Five routing policies, fully nested cross-validation, 10,000 bundle
  permutations. 0 of 8 beat always-cheapest on both success and cost.
- Two controls, both needed: always-cheapest as a fixed-policy baseline, and
  label-shuffle as a null. One without the other misses a different cell.
- **The calibration that matters**: the hindsight oracle only reaches the win
  region in 1 of 8 either. So the barrier isn't primarily the estimator.

### Block 4 — The mechanism, and what it implies (3 min)
- Label supply: a routing label exists only when a task is solved; base success
  is 2–36%; four of six cells left 15–97 trainable labels.
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

**"Isn't always-cheapest too weak a baseline?"**
> The opposite — it's hard to beat, because it's the cheapest fixed policy on
> average. Note *on average*, not a per-episode floor: on an individual task,
> switching mode can finish sooner and cost less than exhausting the step budget
> cheaply. That's exactly where the points left of zero on the plot come from.

**"With 12–14% rerun variance, are your results just noise?"**
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

**Never quote a number without the phrase in the "must say" column.** Two
different baselines are in play and they are not interchangeable.

Everything here is on the poster except where marked — so if you say it, the
visitor can find it. The one exception is flagged in the table.

| Number | What it is | Must say |
|---|---|---|
| `+3.45 to +16.35 pp` | hindsight picker's success gain | **vs the best single fixed mode**; percentage *points*; retrospective |
| `13.7–35.3%` | that picker's cost saving | same baseline as above |
| `0 of 8` | learned routers that win | **vs always-cheapest**, and **on both** success and cost |
| `1 of 8` | hindsight oracles that win | same bar — this line is mandatory, never quote 0/8 alone |
| `2–36%` | base success rate | 8-cell matched set. **The 6-cell VWA figure is 2–27 — never say that next to "8 settings"** |
| `12–14%` | rerun discordance | B0 · classifieds, n=224, three replicated modes |
| `7.14 pp` vs `4.46–7.59 pp` | new mode vs a rerun | same cell; the point lands *inside* the band |
| `9.5–30.6%` | triage cost saving, 8 of 8 | **vs the best-success fixed mode**; hindsight bound; in most pairs always-cheapest saves more |
| `15–97` | trainable labels | 4 of 6 VWA cells (the poster drops the denominator; you can give it if asked) |
| `2.1–4.2×` | corpus the failing cells would need | **not on the poster** — spoken answer only (undersampling control, F16). reddit needs the most, 846 tasks |
| `8,934` | episodes | 6 modes × (224×3 + 203×3 + 104×2) |

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
| "+16 % more tasks" | "+16 percentage **points**" — on a 2% base, "%" inflates it wildly |

Also avoid: quoting token or dollar cost as energy or carbon; describing `dom`
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
