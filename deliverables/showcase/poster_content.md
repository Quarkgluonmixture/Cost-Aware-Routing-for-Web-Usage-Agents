# Poster content — Holistic AI × UCL CDI Showcase, 16 Sep 2026

**Author-facing source of truth.** The strings are inlined in `build_poster.py`;
any edit must be mirrored in both. Every number carries its scope, because the
poster is printed on silk and cannot be corrected.

> **The two baselines.** Half the numbers below are measured against the
> **best-success fixed mode** and half against **always-cheapest**. They are not
> interchangeable, and a poster hero row is exactly where that distinction gets
> flattened. Every hero number therefore prints its own baseline.

---

## Title

> **Can a web agent learn when a screenshot is worth the cost?**

Must set on ONE line in the 431mm the logos leave — `build_poster.py` asserts
this. The template anchors the title box to its *bottom* edge, so an over-long
title grows upward off the top of the sheet rather than down into the byline.

## Standfirst

> Hindsight reveals a real routing opportunity. Against always-cheapest, none of
> eight learned routers — and only one of eight hindsight oracles — win on both
> success and cost. These agents solve just 2–36% of tasks, starving the router
> of the supervision it would need.

> **Two traps this sentence had to avoid.** (a) "…and none could take **it**" —
> the pronoun points the 0/8 result back at the +16.35pp ceiling, which is a
> *different baseline*. The fix is to name the baseline (`Against
> always-cheapest`) rather than to refer back. (b) "the obstruction is not the
> learner" — one oracle of eight *does* clear the bar, so that overstates;
> stating both counts side by side lets the reader draw the weaker, correct
> conclusion themselves.

## Hero — two quantities in tension

`2–36%` was moved out of the hero to the paradox panel: it is the *cause* of the
paradox, not a headline, and as a third KPI it diluted the contradiction.

| Side | Number | Label | Baseline printed under it | Reads as |
|---|---|---|---|---|
| IN HINDSIGHT | `+16.35 pp` (76pt) | MORE TASKS SOLVED PER 100 | vs the best single fixed mode | So there is something genuinely worth routing to. |
| WHEN ACTUALLY LEARNED | `0` (200pt, orange) `of 8` | LEARNED ROUTERS | beat always-cheapest on both success and cost | Not one beat always-cheapest on both counts. |

The two are separated by a rule and joined only by narrative connectives
(`in hindsight` / `when actually learned`) — never `vs`, an arrow or an equals,
because the baselines differ. Orange is the dominance plane's own `learned
router` colour, so the encoding repeats when a visitor reaches the figure.

Provenance: C1 `router_objective_ordering.md` (8 cells, `oracle_sr_cost`) ·
C4 `router_triage_learnability_with_wa.md:124` · §450.8 (8-cell 18-feature
matched set; the 6-cell VWA figure is 2–27 and must not be quoted next to
"8 pairs").

`pp` not `%`: the gain is absolute percentage points. On a cell whose base SR is
2%, writing "%" inflates +1pp into +50% — §450.9 fixes pp for this reason.

---

## 01 · WHAT WE ACTUALLY DID  *(main row, left)*

> **“Find me the cheapest blue kayak on this site.”**
> A REAL TASK — VISUALWEBARENA, CLASSIFIEDS #0

Verbatim from `external/visualwebarena/config_files/vwa/test_classifieds/0.json`.
Not an illustration: it is task 0 of the 234 the poster reports on. It was chosen
because it carries the motivation in itself — one half of the intent is visual,
the other is textual.

> **“blue”** can be visual. **“cheapest”** is textual. Which half of the task is
> in front of you decides whether the screenshot is worth paying for.

Not "wants a picture": it is unverified that the colour is available *only*
visually — a listing title may well contain the word. The weaker claim carries
the same intuition and closes the attack.

Then the workflow, drawn rather than prose:

```
RUN THE SAME TASK SIX WAYS
    NO IMAGE AT ALL      DOM · P-text · P-prompt · P-SoM
    TEXT + SCREENSHOT    SoM
    SCREENSHOT ONLY      Vision
MEASURE SUCCESS × COST
THEN COMPARE THREE POLICIES
    FIXED MODE        the same choice every task
    HINDSIGHT ORACLE  the best choice, known afterwards
    LEARNED ROUTER    a choice made before the task runs
```

> 6 modes · 8 benchmark–model settings · 8,934 episodes

The three-way grouping is `TERMS.md §1.1`, verified from step records
(`image_payload_bytes == 0` for all four text-side modes). Episodes =
6 × (224×3 + 203×3 + 104×2), per-cell `n` from
`router_triage_learnability_with_wa.json`.

## 02 · NOTHING WE TRAINED WON  *(main row, right — the dominant figure)*

> Every policy, in every pair, measured against the one simple fixed baseline:
> **always use the cheapest mode**. A win means landing in the shaded region —
> cheaper *and* no worse.

> **0 of 8** learned routers land in the win region.
> **1 of 8** hindsight oracles clear the same bar — so learning failure alone
> cannot explain this.

> Always-cheapest is the cheapest fixed policy **on average**, not a per-episode
> floor — which is why a few points sit left of it. Nested cross-validation;
> 10,000 bundle permutations.

> **Scope note.** The 1-of-8 line is mandatory, not decoration. §450.12: without
> it, `0 of 8` reads as a claim about learners, which the data do not support.
> The defensible statement is that **no profitable *deployable* operating point**
> exists under this cost accounting — `deployable` is load-bearing, since one
> hindsight oracle does reach the win region and an oracle is not deployable.
> Dropping that word makes the sentence false against this poster's own figure.

## 03 · THE ROUTING PARADOX  *(second row, left)*

```
THESE AGENTS SOLVE 2–36% OF TASKS
SO FEW TASKS ARE EVER SOLVED
SO FEW ROUTING LABELS EXIST          often only 15–97 usable labels
SO THERE IS TOO LITTLE TO LEARN FROM
```

> **The agents that would gain most from routing produce the least supervision
> to learn it.**

Provenance: C5 `router_label_supply_diagnosis.md` — 15–97 trainable labels in
4 of 6 cells. The `4 of 6` denominator is **dropped on the poster**: C5 is a
six-cell VWA diagnosis while every other number is stated over 8 pairs, and an
unexplained denominator switch costs a reader more than the precision buys.

## 04 · CALIBRATION · WHAT SURVIVES  *(second row, right)*

> Rerunning **one mode on the same tasks** flips **12–14%** of outcomes. So we
> measured the ceiling against that ruler:

Drawn as one shared 0–9pp axis: the rerun floor is an **interval** (4.46–7.59)
and the new-mode gain is a **point** (7.14), so the band is a band and the gain
is a rule through it. Drawing both as bars asserted they were the same kind of
quantity, which they are not.

> The gain from adding a representation lands **inside** the range a plain rerun
> already covers — not distinguishable here. B0 · classifieds, n=224, three
> replicated modes.

> **9.5–30.6% cheaper at identical success, in 8 of 8 pairs** — send the tasks
> *nobody* solves to the cheapest mode.
> Against the best-success fixed mode, and still a hindsight bound; in most pairs
> plain always-cheapest saves more. What makes it worth naming is the label:
> *solvable or not* is far easier to supply than *which mode*.

> **Scope note.** Different baseline from §02's always-cheapest; §450.12 requires
> the baseline name to travel with this number. Zero success loss is by
> construction, not a finding (C1b).

## Takeaway band

> **Routing is not only a model-selection problem. Its learnability depends on
> the competence of the agent producing the labels.**
>
> Measured inside the 2–36% success regime we observed. This conclusion need not
> hold for stronger agents.

> SO, IN THIS ORDER
> 1. Improve the agent
> 2. Generate reliable supervision
> 3. Then learn selective perception

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
  entirely in cost" (§396.2), or the 6-cell "1.7–3.3pp drop-one" hero
- ❌ a standalone "what this poster does not claim" box — every limit now travels
  with the number it limits, which is where a reader actually meets it

## Cut in the redesign, and why

- **Nine numbered sections** → four panels. The sections were a document
  structure; the poster needed a hierarchy.
- **The eight-row rerun forest plot** → a calibration card. It was competing
  with the main result for visual weight while making a secondary point.
- **The caveat box** → per-number scope lines (above).
- **`WHAT THIS POSTER DOES NOT CLAIM`, `IS IT JUST TOO LITTLE DATA?`** — the
  undersampling control (`router_undersampling_control.md`, F16) is a strong
  answer to an obvious attack, but it is a *reviewer's* question, not a
  visitor's. It moves to the spoken answer; see the open question below.

## P0 caught in final polish — a retracted sentence re-imported

The first build of this panel carried:

> Always-cheapest is a cost floor, so anything that protects success costs more.

It was lifted from §450.12 without checking whether it still stood. It does not:
`known.py "cost floor"` returns **§476.4 RETRACTED**, which killed this exact
sentence in the thesis's own Figure 6.1 caption, and `known.py "always-cheapest"`
returns two more retractions of the same claim (§474.3, §474.8).

The refutation is visible **on the poster's own figure**: if a chooser
*necessarily* spent more, no point could sit left of x=0 — and several oracle
points do. always-cheapest is the cheapest fixed policy *on average*; on an
individual task, switching mode can finish sooner and cost less than exhausting
the step budget cheaply. That is where the 1-of-8 oracle win comes from.

Two lessons, both already in `paper_process_pitfalls.md`'s territory:

1. **Checking the numbers is not checking the claims.** Every figure on the
   first build was traced to the ledger. This sentence was a *causal assertion*
   about a baseline, carried no number, and was never queried.
2. **A retracted claim survives in the documents that quoted it.** §476.4 fixed
   the thesis caption; the sentence stayed alive in §450.12's prose and was
   re-imported from there months later.
