> **Superseded 2026-09-02.** This pass produced v2 (the exhibition design: hero
> zero, kayak scene, dark takeaway band). A Holistic AI reviewer asked for the
> opposite — one system diagram, compact, one type scale, the template's own
> skeleton — and v4 replaced it. Kept as the record of what was tried and why;
> the live rationale is `poster_content.md` (v4 header + audit trail).

# Exhibition pass — research poster → exhibition object

Scientific content frozen. Nothing here changed a number, a baseline, a scope or
a claim; the edits are hierarchy, distance layering, visual tension and the
physical path a visitor and a presenter take across the sheet.

Verified by rendering the real PDF at three scales — 14 dpi (≈3m), 55 dpi (≈1m),
100 dpi (close read) — and inspecting each, twice.

---

## Phase A — what the audit found

At 3m the previous version was **three equal-weight numbers, one academic
figure, and a dark band top and bottom**. Nothing broke the grid, so nothing
asked to be walked over to. `0 of 8` was typographically indistinguishable from
`+16.35 pp` and `2–36%`, which turned the result into one KPI among three.

The other finding was structural: **`2–36%` was doing nothing in the hero.** It
is the *first step of the routing paradox*, and stating it up top both diluted
the contradiction and pre-empted the panel that explains it.

---

## 1. What was enlarged

| Element | Before | After |
|---|---|---|
| `0` (learned routers that win on both) | 72pt, one of three | **200pt**, the only oversized element on the sheet |
| Title | 34pt | **42pt**, still one line |
| `+16.35 pp` | 72pt | 76pt, deliberately *smaller* than the zero |
| The paradox statement | 13.5pt inside boxes | **27pt serif**, unboxed |
| `2–36%` / `15–97` | body text | 38pt, as the paradox's evidence |
| Figure legend | pipeline vocabulary | relabelled `learned router` / `hindsight oracle` |

## 2. What was unboxed

- **The hero band.** A tinted card containing three tinted cells became
  whitespace with a single hairline rule. Card rhythm was what made it read as a
  dashboard.
- **The routing paradox.** Four rounded boxes joined by arrows became one
  editorial statement. Boxes imply a *process*; this is a *claim*, and it is the
  one idea a visitor should carry away.
- **The task quote.** A quote card became a scene (below).

Retained as cards, deliberately: the calibration panel (it genuinely is an
aside) and the survives strip (a coloured rule, not a box).

## 3. What was demoted or removed

- `2–36%` left the hero for the paradox panel, where it is the cause rather than
  a headline.
- The workflow's two arrows merged into one.
- The QR caption `Scan for code & data` (a file listing) became
  **`Explore all 8 settings →`** (a reason to scan). Still one QR.

## 4. The 3-metre hook

**A very large orange zero.**

The orange is not decorative: it is the colour the dominance plane already uses
for `learned router`. A visitor who is pulled in by the zero and walks up to the
figure meets the same encoding a second time, so the colour is carrying the
claim rather than labelling a category.

The hook is arithmetic curiosity — *why would it be zero?* — and the answer is
three panels away, which is the point.

## 5. Physical presentation path

Designed as a clockwise Z, so nothing requires walking back:

```
      ①  title (top)                    "Can it learn when to look?"
      ②  hero left    +16.35 pp         "In hindsight there is real opportunity"
      ③  hero right   0 of 8            "But nothing we trained took it"
      ④  01 left      kayak scene       "Here is the problem, concretely"
      ⑤  02 right     dominance plane   "And it isn't just a bad learner"
      ⑥  03 lower-l   the paradox       "Weak agents starve their own router"
      ⑦  takeaway     bottom band       "So: agent first, supervision, then routing"
```

No left→right→left ping-pong: ②③ sit side by side across the top, ④⑤ side by
side in the middle, ⑥⑦ close it out downward.

## 6. The kayak scene — the largest single gain

The task stayed verbatim (`VisualWebArena, classifieds #0`). What changed is that
it is now **performed rather than described**: a crude three-row listing, drawn
in native shapes, with **two blue items and one red**.

A visitor solves the agent's task in about two seconds — filter by sight, then
compare by reading — and discovers that the cheapest item is not the answer. The
routing problem is experienced before any of the poster's vocabulary appears.
Labelled `ILLUSTRATIVE — NOT A BENCHMARK SCREENSHOT`, because inventing a
benchmark page would misrepresent the data.

Then one line of intuition and a product-shaped hook:

> **“blue”** can be visual. **“cheapest”** is textual.
> **Should the agent pay to look?**

## 7. Scientific wording deliberately NOT touched for punchiness

Every one of these would read better shortened, and every one stays:

| Kept | Why |
|---|---|
| `vs the best single fixed mode` / `vs always using the cheapest mode` under each hero number | Two different baselines. A hero row is exactly where that gets flattened; the visual tension must not imply one measurement against another, so the connective is narrative (`in hindsight` / `when actually learned`), never `vs` or an arrow. |
| `ROUTERS THAT WERE BOTH MORE SUCCESSFUL AND CHEAPER` | §474.9: the learned router raises success in six of seven cells and loses on cost. Splitting the joint predicate into parallel clauses changes the logical strength — the most dangerous de-jargon failure mode in this project. |
| `1 of 8 hindsight oracles clear the same bar` | §450.12. Without it `0 of 8` reads as a claim about learners, which the data do not support. |
| `cheapest fixed policy on average, not a per-episode floor` | §476.4 RETRACTED the stronger form. It also explains the points left of x=0 on the poster's own figure. |
| `still a hindsight bound; in most pairs plain always-cheapest saves more` | §387.16.3 caveat. Without it, `WHAT SURVIVES` plus a large percentage reads as a deployable win. |
| `This conclusion need not hold for stronger agents` | Not "a stronger agent should overturn it" — `THESIS_ONE_SENTENCE.md` declines to extrapolate, it does not predict a flip. |
| `B0 · classifieds, n=224, three replicated modes` | The rerun band's cell identity; lost once already when the forest plot became a card. |

## 8. Why this is now an exhibition object

The previous version answered *what did you find* well and *why should I care*
not at all — it had no element that broke rank, so at exhibition distance it was
indistinguishable from every other well-made academic poster in the room.

It now has one oversized element carrying a contradiction, one scene a visitor
can solve themselves, one memorable idea set as a statement rather than a
module, and a reading path that does not double back. The evidence sits exactly
where it did, at the size a researcher needs it — but it is no longer competing
with the hook for the first five seconds.

## Final micro-pass (before freeze)

Six items, five presentational and one factual:

1. **Standfirst re-joined the baselines.** "…and none of eight learned routers
   could take **it**" pointed the 0/8 result back at the +16.35pp ceiling — the
   same pronoun failure already fixed in the hero, left standing one level up.
   Now names the baseline instead of referring back: *"Against always-cheapest,
   none of eight learned routers — and only one of eight hindsight oracles —
   win on both success and cost."*
2. **`free` / `trivial` removed everywhere.** always-cheapest is not free; it
   was only ever free *to implement*. Now `one simple fixed baseline`.
3. **The zero got a second-tier label.** At 3m the hook landed but its subject
   did not — the only gloss was 15pt monospace. Hierarchy is now
   `0 of 8` → **LEARNED ROUTERS** (28pt) → exact criterion (15pt).
4. **Lower-left rebalanced** by letting the paradox occupy its space (statement
   27→31pt, evidence pair 38→44pt) rather than by adding content. The gap to
   the right-hand column closed from 43mm to 26mm.
5. **Section numbers dropped.** `01 ·` … `04 ·` was the last of the document DNA;
   position already navigates.
6. **`no profitable operating point exists` → `no profitable *deployable*
   operating point`.** One hindsight oracle *does* reach the win region, so
   without `deployable` the source-of-truth contradicted the poster's own
   figure. Traced to a root cause worth recording: §450.12's own `caveats`
   field still carried the retracted cost-floor claim, because the retraction
   happened in §474.3/§476.4 and the retracted wording stayed where it was. A
   ledger entry now points at it.

## Still open

- **F16 undersampling control** stays off the poster, per the standing decision.
  Prepared as a 20-second spoken answer to *"could 0/8 just be too little data?"*
- **The paradox panel closes 44mm above its box.** Left as breathing room rather
  than filled — the whitespace under an editorial statement is doing work.
