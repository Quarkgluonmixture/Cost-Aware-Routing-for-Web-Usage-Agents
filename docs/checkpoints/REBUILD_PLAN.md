---
type: plan
status: active
created: 2026-07-28
purpose: zero-preset rebuild of the two REALM papers — anti-redo ledger first, then claims, then rewrite
audience: the next session (start here, not at next_steps §0)
---

# Zero-preset rebuild — start here

> **User decision 2026-07-28**: do ALL phases, no scaling down. Previous session's
> handoff (`next_steps.md §0`) is still valid for *state*, but this doc supersedes it for
> *what to do next*.

## Why this exists

The 2026-07-27/28 session redid work that had already been done, and got it wrong, three
times in one sitting. Each time the user had to catch it. The pattern was NOT insufficient
depth — it was **reasoning forward from just-read code without first asking whether the thing
had already been measured or already been adjudicated**. Concretely:

| I claimed | Reality | Already existed |
|---|---|---|
| "no same-mode replicate exists" | 15 manifest pairs + a dedicated clean replicate dir | `compare_cross_run_same_condition.py` prints the exact statistic; §302.1 has the number (6.7/7.6pp) |
| derived a "detector sensitivity" story for the id channel | the id channel was measured properly, with an id-agnostic metric | `b0_paired_idperturb_replay.py` + `docs/checkpoints/probes/*idperturb*.json` (B1 20%, B0 12.5%) |
| labelled Vision as a native-id arm | Vision is coordinate-based, zero element ids | any step record: `coordinate_type: qwen_0_1000` |

`§396.7` had recorded this exact lesson ("check whether it was already adjudicated before
acting") one round earlier. It recurred anyway. **So the fix is not a resolution, it is an
artifact: an index from question → prior finding.** The chronicle is ~20k lines / ~400
sections, append-only; grep only works when you already guess the right keyword.

## 🚨 Read this before trusting anything

1. **Treat every conclusion the previous session wrote as UNVERIFIED.** Especially
   §397.4–§397.9. `§397.10` is the correction entry — **read it with §397.4 and §397.9 or
   you will inherit retracted framings as fact.**
2. **Do not do arithmetic across the noise numbers.** §302 retracted the linear
   decomposition (`12.1% ≈ 10.5% + 1-2pp`) as a category error across 4 incomparable
   dimensions. §300.2 records ±3-5pp cross-GPU drift on the flip rate. Quote each with its
   scope; never subtract.
3. **The papers are frozen for prose edits.** Do not keep patching sentences — Phase 4
   rewrites them. The only edits allowed before Phase 3 are removals of statements that are
   factually false.

## Phases

### Phase −1 — correct the polluted record ✅ DONE 2026-07-28
- `paperA/limitations.md`: the sentence "No same-mode replicate exists in our data" was
  **false** and had been committed and synced to Overleaf. Replaced with the honest version
  (one clean pair exists; 6.7/7.6pp; upper bound; B0-MoE-specific; local cells unmeasured).
- 笔记 `§397.10` appended as the retraction entry for §397.4 + §397.9.
- `next_steps.md §0` forbidden-numbers table corrected.

### Phase 0 — the anti-redo ledger ⬅️ START HERE
Deliverable is a **queryable** artifact, not a summary. Suggested location
`docs/reference/KNOWN.md` (+ a machine-readable sibling), with MEMORY.md holding only a
pointer — it is large and will grow, so it does not belong in memory files.

Five record types:

```
MEASURED    quantity | value | scope (model/site/mode/n) | caveats | source (§ + artifact path) | superseded_by
ADJUDICATED decision | reasoning | date | where recorded (code comment / prereg / §) | do-not-relitigate
RETRACTED   former claim | why dead | § | what replaced it
DATA        what exists | path | grade (clean / pre-fix / archived) | what it can support
COMPUTE     what is running or planned | host | status | ETA | what it unblocks | what it BLOCKS
```

`COMPUTE` exists because the previous session twice mis-stated run status — it told the user a
finished DGX job was still running, and it did not know that a wanted experiment was queued
behind an unrelated one. Both are the same defect as the `DATA` misses: **state that lives only
in a process table or in someone's head**. Every entry needs the *blocks* field, not just ETA,
because the binding constraint here is mutual exclusion, not duration.

Seed entries already known (verify each, do not trust this table):
- MEASURED: id-perturbation flip B1 **20.0%** / B0 **12.5%**, within-group consistency B1
  **1.000** / B0 0.867 — `docs/checkpoints/probes/b0_paired_idperturb_20260529_*.json`
- MEASURED: same-condition self_drop **6.7pp / 7.6pp**, discordance 14.3pp, κ 0.614 —
  §302.1, B0·vision·classifieds, n=224
- MEASURED: AMENDMENT_07 sequential-id switch SR 30.4% → 27.2%, Δ **−3.2pp** — §299.4
- ADJUDICATED: measured-cost tie-break for oracle labels **rejected** —
  `router_features.py:78-101`, B-1806, 2026-06-09
- ADJUDICATED: K-of-N is transparency-only, not a gate — preregistration §2.5
- RETRACTED: "MoE residual 1-2pp" linear decomposition — §302
- RETRACTED: drop-one hero 1.7-3.3pp — superseded by 6-mode k=6 (0.0-1.3pp, H1 FAIL)
- DATA: `results/repro_replicates/B0_vision_classifieds_R24792_clean_replicate/` — clean
- DATA: 15 (model,mode,site) pairs in `run_manifest.yaml` whose second run is
  `grade=archived` = pre-fix ⇒ confounded. Archived runs use **merged naming**
  (`B0_3mode_reddit_20260422` holds DOM+SoM+Vision) — a directory-prefix search misses them.

COMPUTE — verified 2026-07-28 against `ps` / `_status/tasks/*.md` frontmatter, not from memory:

| what | host | status | ETA | unblocks / BLOCKS |
|---|---|---|---|---|
| WA reddit full, 6 modes × 104 | A100, chain pid 2658570 | **running**, step 1/6 (dom, `B1_dom_wa_reddit_..._R13217`) | ~3 days from 07-27 18:00Z | benchmark-generalisation annex. **BLOCKS every VWA reddit run** — shared postmill container + shared `.auth/reddit_state.json` (B-647). This is what a fresh reddit replicate queues behind |
| **mechanistic canonical sweep, 24 cells** | DGX **sweep driver pid 38603** (`.sweep.pid`) + supervisor (poll 300s, ≤40 restarts) | ⚠️ **RUNNING**, 2/24 done, cell 3/24 `p1_rev_reverse_cls` in flight as worker pid 1638252, **21.7 GB VRAM** | cells take ~800-845 min each; 2 cells burned ~27 h ⇒ the **08-01 sweep deadline truncates it around cell 7-8**, it will never reach 24 | §5 mechanism, **shelved** by advisor 2026-05-14 ⇒ archive only, feeds no current paper. **BLOCKS `task_b3_mimo`, whose frontmatter says "适配 2026-07 下旬 (DGX)" — a shelved workstream is holding the GPU a roadmap workstream needs.** Stopping it is a user decision |
| ~~cell 2 `p1_fwd_strong_red`, worker pid 38617~~ | DGX | that ONE CELL finished cleanly 07-28 03:34 (24 tasks, `run_manifest.json` emitted, log ends `pilot DONE`) | — | ⚠️ **2026-07-28: the previous session read this single worker's exit as the whole sweep completing.** Checking a worker pid instead of `.sweep.pid` is the same defect as the DATA misses above — verify the driver, not a child |
| `task_pass1_baseline` | A100 | active, 36/36 landed | — | the k=6 substrate both papers use |
| **`task_pass2_router` — live router pass** | — | **SUPERSEDED, NEVER FIRED** | — | ⚠️ paperA §4.4 discloses this. `k_of_n = 0/0`. Replaced by the offline suite; live router is paper-2 |
| `task_wa_pilot` | A100, B1 local (proxy-immune) | pending | 2026-08, slots into the B3 adaptation gap, ~days | WA generalisation |
| `task_b3_mimo` (B3 = MiMo-VL) | DGX adapt → A100 fire | pending | adapt late 07; fire early 08, **12 conditions ≈ 2-2.5 weeks** | cross-family third model |
| `task_b0_replicate_annex` | A100 + proxy quota | pending | 2026-08 **opportunistic** (grab a proxy window, precedent §369) | ⚠️ a pre-submission B0 rerun was **rejected as outcome-dependent sampling** — do not re-propose it |
| `task_shop_expansion` (Phase 1b: shop × 3 models × 6 modes) | A100 | pending | **2026-09+**, journal/major-revision horizon | R3 → R1 framing decision |
| WA shopping / shopping_admin | — | **deliberately not opened** | — | no reset implementation ⇒ cannot be paper-grade clean |
| `task_grl_subpaper` | — | queued, NOT critical path | 2026-06 slot missed | — |
| `task_doi2_bundle` | — | **blocked** | — | — |
| possible fresh **locally-served** same-mode replicate | A100 | **not scheduled** — Phase 0b decides whether it is needed | queues behind WA reddit (~3 days) | the H3 noise floor for B1/B2 cells |

**Execution note**: chunk the chronicle by § ranges and fan out subagents writing into one
schema. Do not read it linearly — that spends most of a context window and yields a summary
rather than a lookup.

### Phase 0b — self-oracle floor sweep (run in PARALLEL with Phase 0)
Offline, reads cached artifacts, does not touch the sites, does not queue on the A100.
`compare_cross_run_same_condition.py` over every available pair, reporting **separately**:
- clean vs pre-fix pairs (never pool them)
- API-served B0 vs locally-served B1/B2 (B0 carries a ~13% serving-inconsistency floor;
  B1 is deterministic on repeated identical input, so its model-side floor may be ~0)

**This gates whether H3 is a positive result at all**, so it must land before Phase 3.
Open question it answers: is the 6.7/7.6pp floor a B0-vision artifact, or general? If
general, H3's 1.35/2.09pp axes sit inside the noise.

Unverified hypothesis worth testing here: axis-1 (`|P-text \ P-SoM|`) is **within** one
id-regime (both 1..K) while axis-2 (`|P-prompt \ P-SoM|`) **crosses** the AMENDMENT_07
boundary. If the crossing is what makes axis-2 the larger axis, the asymmetry §3.2 calls
"informative" has a mundane cause.

### Phase 1 — claim inventory of both papers, zero preset
Every claim currently in `paperA/` + `paperB/`: claim | number | source artifact | status
(verified / unverified / **contradicted by ledger**). Cross-join against Phase 0. This is
where the previous session's errors surface mechanically instead of by user catch.

### Phase 2 — innovations and counter-arguments
Only after the ledger exists, otherwise both are speculation. **Filter candidate
counter-arguments through the ledger before dispatching any cross-AI round** — in the
2026-07-28 round, 2 of codex's and 2 of Gemini's findings were either re-litigating settled
decisions or misreadings, and the ledger would have caught them for free.

### Phase 3 — decide paper structure
Given what survived. The live fork: Paper B's claim ("the ceiling is real but unlearnable")
depends on neither H3 nor the hallucination metric, so it is the more robust of the two;
Paper A's positive result is exactly the most uncertain thing.

### Phase 4 — rewrite

### Phase 5 — memory + CLAUDE.md
Add a **before-you-claim checklist** derived from the actual failure modes, not generic advice:
1. Has this quantity already been measured? (search the ledger, not the chronicle)
2. Has this decision already been adjudicated? (code comments carry adjudications)
3. Is the metric's basis constant across the arms being compared?
4. Is this in-sample or out-of-sample?

## Scheduling reality
Everything on the critical path is **effort-bound and offline**, including the floor sweep.
Only one thing is wall-clock-bound: *if* the sweep shows a fresh locally-served replicate is
needed, that needs the A100, which is occupied ~3 days by the WA full run — and WA reddit
cannot run beside VWA reddit (shared postmill container, `.auth/reddit_state.json`).

## State at handoff (verify, do not trust)
- git `origin/fix/b1878-reddit-reference-image`, working tree clean, 0 unpushed
- Overleaf synced at repo `9cf49b8`; ⚠️ **the Limitations fix from Phase −1 is NOT yet
  synced** — re-run `bash scripts/maintenance/overleaf_sync.sh` when convenient
- Gates as of Phase −1: `make deslop-ratchet` PASS · both papers content ≤ p8 ·
  `pytest -q` 1626 passed
- Background: A100 chain pid 2658570 alive, WA full, step 1/6 (dom), ETA ~3 days.
  DGX mechanistic 38617 **finished** 07-28 03:34 (§5 mechanism, shelved, archive only)
