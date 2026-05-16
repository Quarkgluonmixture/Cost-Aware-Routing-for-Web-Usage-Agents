# Router archive diagnostic — SANITY-CHECK ONLY (NOT preregistration lock substrate)

> ⚠️ **METHODOLOGY REFRAME 2026-05-16 evening (Option C)**: this archive diagnostic is **NOT a preregistration lock substrate**. Archive uses the same task IDs (cls 0-233, red 0-209) that Phase 1a will use, and `success` values have drifted between pre-bug archive and Phase 1a (post-§107 + post-§139.8). The originally-intended "DEFER triggered on archive" / "δ → 2×SD" / "anchor fallback triggered on archive" verdicts are **retracted as pre-registration substrate** — they cannot serve as independent calibration because archive is a correlated-population sanity check, not an independent test set.
>
> What this diagnostic IS: **pre-fire directional confidence check on correlated-population pre-bug data**. Verdicts inform whether methodology direction is sane (e.g., is label entropy in a plausible range? does anchor identity stabilize across resamples?) BEFORE firing fresh Phase 1a. Verdicts do NOT lock Phase 1a outcomes.
>
> What this diagnostic IS NOT: independent test, preregistration lock substrate, or trigger for the H10 DEFER / anchor fallback conditions. Those DEFER/fallback rules are pre-data-locked in `preregistration.md §2 H9/H10 + §4`, but their **triggers evaluate on Phase 1a fresh-data train folds post-launch**, not on this archive.
>
> See `docs/checkpoints/router/proposals_v4.md` §A for the methodology critique that motivated this reframe, and `preregistration.md Appendix A 2026-05-16 evening` entry for the chronicle.

Run date: 2026-05-16T08:29:14.649775Z
Archive: B0 cls + reddit x 6 modes (paper-grade pre-bug archive — correlated to Phase 1a task population)

## Gate summary

| Cell | n tasks | G-1 entropy | P2 viable? | G-2 Kendall tau | Anchor stable? | G-4 noise SD (pp) | Delta raise? |
|---|---|---|---|---|---|---|---|
| B0_classifieds (partial: 5/6 modes) | 234 | ... | ... | ... | ... | ... | ... |
| B0_classifieds | 234 | 0.734 | YES | 0.696 | NO | 2.23 | YES |
| B0_reddit | 210 | 0.606 | NO | 0.841 | YES | 2.17 | YES |

## Per-cell detail

### B0_classifieds

**G-1 label entropy**: H = 0.734 (threshold log(2) = 0.693); P2 VIABLE.
Label histogram: `{'dom': 185, 'phantom_text': 7, 'vision': 11, 'som': 29, 'phantom_som': 2}`. Majority mode `dom` baseline SR = 14.96%.

**G-2 anchor Kendall tau**: mean tau = 0.696 (FLICKER — preregistration C2 fallback triggers). Full-cell best-single-mode = `som`; majority-winner-across-resamples = `som` (100.0%).
Anchor winner distribution: `{'som': 500}`.

**G-4 router-vs-anchor noise SD**: SD = 2.23pp, mean oracle lift = 12.79pp [8.97, 16.67]. delta_h9/delta_h10 calibrated to 4.47pp (RAISE from 1.0pp default).

### B0_reddit

**G-1 label entropy**: H = 0.606 (threshold log(2) = 0.693); P2 NOT VIABLE — H10 DEFER condition triggers.
Label histogram: `{'dom': 181, 'som': 10, 'phantom_text': 8, 'phantom_prompt': 4, 'phantom_som': 2, 'vision': 5}`. Majority mode `dom` baseline SR = 11.43%.

**G-2 anchor Kendall tau**: mean tau = 0.841 (STABLE). Full-cell best-single-mode = `phantom_som`; majority-winner-across-resamples = `phantom_som` (58.4%).
Anchor winner distribution: `{'phantom_som': 292, 'phantom_text': 199, 'som': 9}`.

**G-4 router-vs-anchor noise SD**: SD = 2.17pp, mean oracle lift = 10.82pp [7.14, 14.29]. delta_h9/delta_h10 calibrated to 4.35pp (RAISE from 1.0pp default).


## Directional confidence (NOT preregistration lock)

> 🔄 **Reframed 2026-05-16 evening per Option C methodology fix**. These per-cell signals indicate **direction**, not **decision**. The actual H10 DEFER / anchor fallback / δ choices trigger on Phase 1a fresh-data train folds post-launch (see `preregistration.md` revisions).

1. **Label entropy direction (G-1)** — reddit B0 archive shows entropy 0.606 < log(2)=0.693, suggesting (correlated-population signal only) that learned classifier P2 may face collapsed-label challenge on reddit. **NOT a pre-data DEFER trigger** — Phase 1a fresh-data per-cell entropy on B0/B1/B2 reddit train folds will be the actual trigger. Could shift post-bug-fix (`success` drift) and after B1/B2 capability tier introduced.

2. **Anchor stability direction (G-2)** — cls Kendall τ=0.696 just below 0.7 (would have triggered fallback under original archive-trigger framing); reddit τ=0.841 stable. **NOT a pre-data fallback trigger** — Phase 1a fresh-data per-cell Kendall τ on B0/B1/B2 cls/red train-fold resamples will be the actual trigger.

3. **δ calibration (G-4) — RETRACTED entirely** — the originally-proposed "raise δ to max(1.0pp, 2×SD) if SD > 0.5pp" rule was statistically incoherent (SD is the sampling SE of the lift estimator, which enters the test statistic Z = (θ−δ)/SE, **not** δ itself which is the effect-size floor) AND archive-dependent. δ stays at 1.0pp (mirror H1 effect-size floor). Per-cell power at this δ + expected 1-3pp effects is modest (~12-20%); paper §6 prose discloses this and relies on across-cell FE pooling.

## Why these verdicts cannot pre-register Phase 1a outcomes

Same-task-ID overlap: archive uses cls 0-233 / red 0-209 exactly as Phase 1a will. Per-task patterns (which modes solve which tasks) carry over with ~95% correlation post-bug-fix drift. "Pre-data lock on archive" framing collapses on this fact — it's pattern leak, not honest pre-registration. The clean alternative is Phase 1a fresh-data triggers + treat archive as informal pre-fire confidence (this doc's reframed role).

## TODO second pass (browser-state gates G-3 / G-5 / G-6)

- G-3 P1 threshold validation (bucket SR gap at dom_size=12000 + dom_complexity=500 on archive) — needs step-1 JSONL browser-state read
- G-5 runtime intent regex coverage vs codex audit Cat A/B/C/D (target >= 70%) — needs task.intent + codex_audit_*.json cross-check
- G-6 hijack threshold validation ((B1+B2) + density > 90 markers cell SR ranking) — needs B1 + B2 archive (currently B0 only)
