# Launch intent — finish the 09-15 local replicate chain: B2 × cls (som, vision) → B1·shop·som resume → B2·red·dom (declared 2026-10-08, BEFORE fire)

Same discipline as `local_replicate_chain_launch_intent_20260915.md`, which this resumes. That chain
aborted on 2026-09-24 21:37 when B1·shop·som (R22515) hit a 30 h per-condition wallclock cap at
283 / 435 episodes; L8–L10 (the three B2 cells) never started (实验笔记 §551). The A100 has been
idle since. user 2026-10-08: start.

## The cells, in fire order

| # | cell | collection n | why | est. wall | original run (arm a) |
|---|---|---|---|---|---|
| 1 | B2 × VWA-cls × `som` | 224 | cls_B2 has no band; first arm | ~25 h | `B2_som_classifieds_20260611_…_R3380` |
| 2 | B2 × VWA-cls × `vision` | 224 | cls_B2 second arm | ~28 h | `B2_vision_classifieds_20260612_…_R9288` |
| 3 | B1 × VWA-shop × `som` — **resume R22515** | 435 (152 to go) | first band on shopping | ~16 h | `B1_som_shopping_20260812` |
| 4 | B2 × VWA-red × `dom` | 205 | red_B2 first arm | ~52 h | `B2_dom_reddit_20260715` |

All $0 (local bf16 on the A100). Estimates: B2 from the 09-15 intent file; shopping from R22515's
own rate (283 episodes in 30 h ≈ 6.4 min/episode).

**Ordering is load-bearing, and reversed from 09-15.** ARR is 2026-10-12. Cells 1–3 (~69 h) can land
before it; cell 4 cannot and is for the COLING commitment (2026-12-23). The two cls cells go first
because together they give cls_B2 a two-arm band; shopping third because its value is one arm.

## Mechanics

`queue_chain.sh` applies `FORCE_NEW` to the whole chain, and the shopping cell must NOT mint a new
run (it must glob-resume R22515, which is the newest `B1_som_shopping_*` by mtime; `mint_run_id`
falls back to a fresh run if R22515's env_snapshot commit / submodule SHA no longer match — A100
HEAD is still `fdb45634`, the same as the 09-21 chain). So three chains run back-to-back from one
wrapper, each only if the previous exited 0:

1. `FORCE_NEW=1` — `queue_baseline.sh B2 som classifieds`, `queue_baseline.sh B2 vision classifieds`
2. (no FORCE_NEW) — `queue_baseline.sh B1 som shopping`
3. `FORCE_NEW=1` — `queue_baseline.sh B2 dom reddit`

`RESET_BEFORE=1` (chain default), `MAX_CONDITION_HOURS=60` (the cap is opt-in for B1/B2; 09-21 set
30, which is what killed R22515; 60 keeps a hang guard above every estimate).

## Halt conditions

- any cell finishes with episodes ≠ collection n (224 / 224 / 435 / 205)
- the shopping cell starts a fresh run instead of resuming R22515 → stop the wrapper, diagnose
- `condition_aborted: true` or a chain exiting non-zero → the wrapper stops; diagnose before continuing
- another site chain found running → refuse (host-global lease)
- the condenser cert expires 2026-10-15T01:15 — the fire does not need it, only the pull does

## Prediction, recorded before the fact

09-15 predicted all three B2 cells at a 0% floor and B1·shop·som at 0–2%. Since then: the WA·B1
replicates landed at 5.8–9.6% (C1 retracted, §529.4), and the partial shopping pair already reads
19 / 283 = 6.71% discordance (§551). Revised: **B2 cls som / vision 0–3%** (B2 solves 2% of cls, so
discordance is capped by the few solves); **B1·shop·som 5–9%** on the full set.

## Registration

Each landed cell is a replicate that can enter `CLEAN_PAIRS` only when complete. Registration moves
the noise-floor canonical and is **left to the user**, as in every previous chain.
