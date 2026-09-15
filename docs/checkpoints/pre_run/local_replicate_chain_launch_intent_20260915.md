# Launch intent — local replicate chain: WA·B1·reddit ×6 → B1·shop·som → B2 ×3 (declared 2026-09-15, BEFORE fire)

Same discipline as `floor_chain_launch_intent_20260817.md`,
`reframe_chain_launch_intent_20260819.md`, `b1_reddit_chain_launch_intent_20260826.md`,
`b0_reddit_replicate_chain_launch_intent_20260902.md` and
`b1_shopping_chain_launch_intent_20260906.md` (§469.7): the cells and what each outcome
would mean are fixed here, before any episode exists.

**Every cell below gets reported, whatever its number says.** A cell that lands and is
omitted requires a written reason in this file, in the same commit.

## Why this chain and not the one §505.27 scheduled

§505.27 (2026-09-09) ordered five replicate/fire items after B1·shop, four of them paid
(B0 shop ×3 ~$135 · WA red B0 ×6 ~$50 · B5 cls som ~$50 · B5 red ~$80), and ruled
"不跑 B2 任何 replicate". It cited "预算 $546 实测".

**That figure was not a balance.** `$546` is the *cost-estimate total* in the budget table
at 实验笔记 L27405 (shop + WA two sites + B4). The live balance, measured
2026-09-15 08:59 UTC with `proxy_budget_watch.py --once`, is **$30.18** — at the $30 halt
line the 09-02 intent set. B5 calls the same proxy (`i5xpracyci…/model-api/invoke`), so
**every paid item is blocked** until the pool is topped up.

What remains runnable is local and free. The user chose this set and order on 2026-09-15
(option A), **including three B2 cells — a deliberate reversal of §505.27's "no B2
replicate"**, recorded here as the user's decision. Paid items preempt at the next cell
boundary once the pool is topped up; by the ordering below, that cuts the B2 tail first.

## The cells, in fire order

| # | Cell | collection n | orig SR | d = n·SR·0.59 | grade | est. wall | orig run |
|---|---|---|---|---|---|---|---|
| L1 | B1 × WA-reddit × `dom` | 104 | 16.35 | 10.0 | interval (at the bar) | ~15 h | `…_R13217` (07-27) |
| L2 | B1 × WA-reddit × `phantom_prompt` | 104 | 16.35 | 10.0 | interval (at the bar) | ~16 h | `…_R21734` (07-30) |
| L3 | B1 × WA-reddit × `phantom_text` | 104 | 16.35 | 10.0 | interval (at the bar) | ~16 h | `…_R10542` (07-29) |
| L4 | B1 × WA-reddit × `som` | 104 | 13.46 | 8.3 | inventory | ~15 h | `…_R301` (07-28) |
| L5 | B1 × WA-reddit × `phantom_som` | 104 | 11.54 | 7.1 | inventory | ~16 h | `…_R11421` (07-30) |
| L6 | B1 × WA-reddit × `vision` | 104 | 9.62 | 5.9 | inventory | ~15 h | `…_R20074` (07-29) |
| L7 | B1 × VWA-shop × `som` | 435 | 7.59 | **19.5** | interval | ~48 h | `B1_som_shopping_20260812` |
| L8 | B2 × VWA-reddit × `dom` | 205 | 3.90 | 4.7 | inventory | ~52 h | `B2_dom_reddit_20260715` |
| L9 | B2 × VWA-cls × `som` | 224 | 2.23 | 2.9 | inventory | ~25 h | `B2_som_classifieds_20260611_…_R3380` |
| L10 | B2 × VWA-cls × `vision` | 224 | 2.23 | 2.9 | inventory | ~28 h | `B2_vision_classifieds_20260612_…_R928` |

All $0 (local bf16 on the A100). Wall-clock from the original cells' first→last episode
mtime; total ≈ 245 h ⇒ **lands ~2026-09-25**, against a hard halt of 2026-10-05 (below).

**Ordering is load-bearing.** Within WA the three d=10 arms go first, so a cut leaves the
powered arms. B2 goes last because it is the lowest-power block and the one §505.27 had
excluded; if paid work preempts, B2 is what gets cut.

⚠️ **Denominators to resolve on landing, not assume**: the collection n above is what each
cell must finish with. Scored n comes from `canonical_task_universe` (VWA-reddit 203,
VWA-shop protocol exclusions per AMENDMENT_10, WA-reddit per its own universe). B-1992 was
exactly this confusion; §510 found a protocol-excluded task inflating an envelope bound.

## Reading 1 — WA·B1 six arms: does C1 survive a second benchmark?

C1 as of `serving_mode_floor.md` (regenerated 2026-09-02): **API 4.93–14.29%** (13 arms,
2 families, 2 sites) vs **local 0.00–3.45%** (5 arms, Qwen only, 2 VWA sites; only
classifieds carries d≥10). Every local arm so far is VWA.

| WA·B1 floors land at | what it means for C1 |
|---|---|
| **all ≤ 3.45%** | the local group holds on a second benchmark. With L1–L3 at d=10 this is the first local reading outside VWA that carries an interval. |
| **any powered arm (L1–L3) ≥ 4.93%** | **C1 is dead as stated** — a local arm inside the API range means the floor tracks workload/benchmark, not serving path. `serving_mode_floor.md` is retracted, not hedged. |
| **between (3.45–4.93%)** | the gap closes; C1 survives only as "separates on VWA". **Written first because it is the easiest to spin.** |

Inventory arms (L4–L6) landing ≥ 4.93% are reported in the same table; they cannot kill C1
alone but they are not dropped for being underpowered.

**Second reading, same data — the first six-arm envelope on a local backbone.** B1·cls
has replicates on 3 arms only, so no local 2⁶ unique-solve envelope exists anywhere. WA·B1
will have one. Declared in advance: if the six floors are ≈ 0, the envelope collapses onto
the single-run assignment, and what it certifies is only *that B1's WA unique-solve counts
reproduce* — it says nothing about the API-side noise the hero is read against.

## Reading 2 — B1·shop·som: the named coverage gap

`serving_mode_floor.md §3` names it: "the local group spans 2 sites only at INVENTORY
grade; restricted to d≥10 it covers 1 — VWA-classifieds". L7 (d=19.5) is the one cell in
reach that can close that.

| B1·shop·som floor | meaning |
|---|---|
| **≤ 3.45%** | the local range holds at interval grade on a second site; the §3 caveat is removed |
| **≥ 4.93%** | C1 dead as stated (same rule as Reading 1) |
| **between** | C1 narrows to classifieds at interval grade |

⚠️ **What L7 does NOT buy**: a band for shop·B0. A local floor cannot bound an effect
measured on B0 (§470.5 / §471). Every "within band" judgment on shop·B0 remains
unreadable until the paid B0 shop replicate runs.

## Reading 3 — B2 ×3: declared inventory, with a structural limit stated up front

`serving_mode_floor.md §3` already says measuring B2 **cannot give the local group a second
family** — a power limit, not a scheduling one. That stands; these cells do not change it.

A sharper limit, derived before the fire: per-task discordance between two runs cannot
exceed the sum of their success rates. At SR 2.23% on both runs, L9/L10 can reach at most
**≈ 4.46%**, which is **below the API floor's lower edge (4.93%)**. So:

- **L9 / L10 cannot falsify C1** unless the replicate's own SR rises above ≈ 2.7%. A 0.00% on
  either is expected under *both* "deterministic local serving" and "almost never
  succeeds", and distinguishes neither.
- **L8 (red·dom, SR 3.90%, ceiling ≈ 7.8%) is the only B2 cell that can cross 4.93%.** If it
  does, it is reported as a local Gemma arm inside the API range — at inventory grade, but
  not dismissed as "underpowered", because a boundary crossing is exactly what an
  inventory point can show.

## Code drift between original and replicate (disclosed, not assumed away)

Fire-path code on the A100 is functionally DGX HEAD: a per-file sha256 over
`p79/ scripts/queues/ scripts/run_experiment.py configs/ experiment_watchdog.py` (269 files)
differs in one file, `_lib_paper_grade_gates.sh`, and only in a bug-number string in a
comment and a log line (B-1991 vs B-1993) plus a trailing newline. No fire-path commit since
`a444e05`.

Between each original run and now:

| pair | commits in between that touch the decision path of this backbone |
|---|---|
| WA·B1 (07-27…30 → 09-15) | B-1996 `9bcbbed` (strip out-of-table chars before `type`; changes behaviour only when such a char is typed, and records `type_text_sanitized`). B-1966 `3c7d348` is additive in `som.py`. B-1985/B-1986 touch proxy paths only (B0/B5). B-1957/61/62 are launch gates. |
| B1·shop·som (08-12 → 09-15) | B-1996 only. The same code already produced B1·shop P-text/P-prompt (09-08/09-11). |
| B2·red·dom (07-15 → 09-15) | as WA·B1. |
| B2·cls·som/vision (06-11/12 → 09-15) | as above **plus** B-1880/B-1881/B-1883 (06-20…22: proxy retry, pre-flight transient episode-retry, auth episode-retry budget) and `b0dade8` (additive B3 files + 7 lines in `factory.py`). **~3 months of drift**, the same situation the floor chain disclosed for `B1.cls.dom` (06-03 → 08-19). |

Energy/cost: both arms of every pair record `source: "psutil_profile"` (pynvml is not
installed in the A100 venv), so the 2026-09-12 driver incident below does not touch cost
comparability.

## Halt conditions

- any cell finishes with episodes ≠ its **collection n** (104 / 435 / 205 / 224)
- another site chain found running → refuse (host-global lease, hard rule #3)
- **2026-10-05 00:00 UTC** → halt the remaining cells regardless of progress (one week to ARR)
- proxy pool topped up → stop at the next cell boundary and re-plan with the paid items
  first (user, 2026-09-15). Not a failure; the unrun tail is recorded as preempted.
- `condition_aborted: true` on any cell → halt the chain, diagnose before continuing

## Known and accepted — do NOT re-diagnose mid-flight

- `.locks/manifest_bind_halt.marker` stands (2026-09-13 18:10 UTC, written as the B1·shop
  chain landed). It gates only `experiment_watchdog.py` RESUME_MISSING and
  `queue_phase1_paper_grade.sh`; `queue_chain.sh` / `queue_baseline.sh` do not read it
  (verified 2026-09-02). Consequence accepted: no automatic refill of missing episodes, so
  the collection-n check is the backstop.
- Each cell raises a COMPLETE-ghost urgent on landing, because a replicate cannot be in
  `CLEAN_PAIRS` before it exists (§487.7 / §492.3). Expected; its text is truncated at
  400 chars. Registration moves the noise-floor canonical and is **left to the user**.
- **NVIDIA driver, 2026-09-15**: unattended-upgrades on 09-12 06:09 UTC moved the user-space
  libraries 580.173.02 → 580.178.04 while the loaded kernel module stayed 580.173.02
  (`nvidia-smi`: "Driver/library version mismatch"; CUDA compute still worked). Healed in
  place with rmmod/modprobe per §387.1 (no reboot, KubeVirt detach hazard avoided); module
  now 580.178.04. The 16 `*nvidia*580*` packages are on `apt-mark hold` for the duration.
  Undo: `sudo apt-mark unhold $(apt-mark showhold | grep nvidia)`. Kernel 5.15.0-191 is
  installed but not booted; **do not reboot** mid-chain.
- Launched with `FORCE_NEW=1` so every cell mints a fresh timestamped run_id; without it
  `mint_run_id` glob-resumes the finished original and the "replicate" runs zero episodes.

## Prediction, recorded before the fact

From the local pattern so far (cls 0.00/0.00/3.12, reddit 1.97/3.45) and §298.2's step-level
determinism probe (133/133): WA·B1 floors **0–3.5%**, with `dom` the most likely to be
non-zero; B1·shop·som **0–2%**; all three B2 cells **0%**.

That prediction supports C1. The row that would cost most is any of L1–L3 at ≥ 4.93%:
C1 would not survive its first benchmark outside VWA. Writing the comfortable prediction
down first is what makes that row reportable if it arrives.
