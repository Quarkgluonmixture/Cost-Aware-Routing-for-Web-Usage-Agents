# Launch intent — B0 × reddit × {SoM, DOM, Vision} replicate (declared 2026-09-02, BEFORE fire)

Same discipline as `floor_chain_launch_intent_20260817.md`,
`reframe_chain_launch_intent_20260819.md`, `b5_reddit_chain_launch_intent_20260826.md`
and `b1_reddit_chain_launch_intent_20260826.md` (§469.7): the cells and what each
outcome would mean are fixed here, before any number exists.

**Every cell below gets reported, whatever its number says.** A cell that lands
and is omitted requires a written reason in this file, in the same commit.

## What this chain is for — three declared readings, not precision

B0 × VWA-reddit currently has replicates on **only its three phantom arms**
(P-text 7.39% / P-prompt 11.33% / P-SoM 10.34%, registered 2026-08-26). Its
SoM, DOM and Vision arms have never been rerun. That single fact is load-bearing
for three separate claims, and this chain is declared against all three.

### Reading 1 — the 2^6 unique-solve envelope, on a second site

§470.3 computed the 2^6 assignment envelope for unique-solve on **cls·B0 only**:
the three phantom arms have a unique-contribution lower bound of **0-1** (can be
driven to zero), while SoM/Vision bound at **6**. That asymmetry is the current
quantitative backing for the surviving hero framing ("the cross-*side* coverage
difference is robust", CLAUDE.md). It rests on one cell.

The envelope needs all six arms replicated. Reddit has three. This chain supplies
the other three.

| B0·reddit envelope lands at | what it means |
|---|---|
| phantom arms bound ≈0, SoM/Vision bound clearly >0 | the cross-side difference holds on **two** sites. The hero framing stops resting on a single cell. |
| SoM/Vision **also** bound ≈0 | **the cross-side coverage difference does not replicate on reddit.** The hero must be restricted to classifieds in prose, not hedged. |
| phantom arms bound clearly >0 | stronger than cls, but the two sites now disagree about the phantom arms and that disagreement must be reported as such, not averaged away. |

### Reading 2 — is C1's API-side lower bound real, or site drift?

`serving_mode_floor.md` (C1) reads the API group as **7.39-14.29%**. The 7.39%
is `B0·red·ptext`, and 台账 §478.4 carries an open `CLAIM_UNVERIFIED` against
exactly it: its canonical run is from June, its replicate from August, and its
flip asymmetry (**A-only 12 / B-only 4**, ≈3:1) does not look like the roughly
symmetric noise the other arms show. Shape-wise that is closer to one-way drift.

The three new cells rerun June-era canonical arms against September replicates —
the same two-month spread, the same site, the same backbone.

| new arms' flip asymmetry | what it means for C1 |
|---|---|
| roughly symmetric (< 2:1) | the 3:1 on P-text is that arm's own property. **7.39% stands** as C1's API lower bound. |
| also ≥ 2:1 and same-signed | it is a **June↔August site-state drift**, not model nondeterminism. 7.39% is then unusable as a floor, C1's API range becomes 10.34-14.29%, and the gap against the local group must be recomputed. |

### Reading 3 — C1's separation itself

The local group's upper bound moved to **3.41%** two days ago (`B1·red·dom`,
2026-09-02). Three more API-side floors are three more chances to falsify the
grouping.

| new floors land at | what it means for C1 |
|---|---|
| **≥ 7%** | consistent with the existing API group. C1 gains three arms and a second site on the API side is further corroborated. |
| **3.41-7%** | the API group's lower bound descends toward the local group. The groups still do not overlap, but the 3.98pp gap shrinks and the claim weakens to a narrower one. |
| **< 3.41%** | **the groups overlap. C1 is dead as stated** and `serving_mode_floor.md` must be retracted rather than hedged. |

## The cells, in fire order

| # | Cell | n (scored) | Est. wall-clock | Est. cost |
|---|---|---|---|---|
| **A1** | B0 × red × `som` | 203 | ~35 h | ~$23 |
| **A2** | B0 × red × `dom` | 203 | ~35 h | ~$23 |
| **A3** | B0 × red × `vision` | 203 | ~35 h | ~$21 |

Budget: **~$67 against a measured $106.43 remaining** (`proxy_budget_watch.py
--once`, 2026-09-02). Headroom ~$39. The $300 requested 2026-08-27 has still not
landed and is **not** assumed here.

**Ordering is load-bearing.** `som` and `dom` carry d≈17.5 and can report an
interval; `vision` carries d≈9.3 and cannot. If the chain is cut — by deadline,
by quota, or by a proxy outage — it is cut at `vision`, and the surviving cells
are the two powered ones. The fallback is a weaker version of the same test, not
a different test.

## Power, declared up front

`d ≈ n × SR × 0.59` (§468 / B-1972); `d < 10` ⇒ inventory, not an interval.
Projected from the archived B0 reddit SRs at n=203:

| Cell | archive SR | projected d | verdict |
|---|---|---|---|
| `som` | 14.6% | **17.5** | interval |
| `dom` | 14.6% | **17.5** | interval |
| `vision` | 7.8% | **9.3** | **inventory only** (0.7 short of the bar) |

`vision` is declared underpowered before it runs, and is run anyway: Readings 1
and 2 need all six arms (the envelope is not defined on five), and an
inventory-grade point still moves a point across a boundary even when it cannot
tighten an interval. It is **not** licensed to carry a CI in any downstream table.

## Registration

All three pairs are to be registered in `aggregate_noise_floor_inventory.py`
`CLEAN_PAIRS` once they land, **as declared** — per §469.7, registering after
seeing the numbers would make the list a post-hoc selection. Registration itself
moves the noise-floor canonical and is therefore **left to the user**, as with
the B1×reddit pair from 2026-08-30.

## Halt conditions

- any cell finishes with episodes ≠ **205** (the COLLECTION denominator; the
  scored one is 203 — B-1992 was exactly this confusion)
- `proxy_budget_watch.py --once` drops below **$30** remaining → halt before the
  next cell rather than start one that cannot finish
- proxy returns sustained non-2xx (the B0 outage mode) → halt, do not silently
  burn retries

## Known, accepted, and NOT to be re-diagnosed mid-flight

- `.locks/manifest_bind_halt.marker` is present (2026-09-02 16:57, from the
  B1×reddit chain). It gates **only** `experiment_watchdog.py:1171`
  RESUME_MISSING relaunch and `queue_phase1_paper_grade.sh:146`; neither
  `queue_chain.sh` nor `queue_baseline.sh` reads it (verified 2026-09-02, not
  assumed from §469.6). **Consequence accepted:** watchdog will not auto-refill
  missing episodes while it stands, so the done-monitor's 205-episode check is
  the backstop.
- Each cell will raise a COMPLETE-ghost urgent on landing, because
  `validate_fire_manifest.py:227` only exempts replicates already in
  `CLEAN_PAIRS`, and a run cannot be registered before it exists. This is the
  same expected sequence as 08-21/23/24 (§487.7) and 08-30 (§492.3). Its
  diagnostic text will be eaten by the 400-char truncation at
  `experiment_watchdog.py:1185`. **Do not spend time reading it.**
