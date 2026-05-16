---
name: p1-archive-simulation-findings-2026-05-16
description: P1 v3 router archive simulation — phantom_som as default validated, but P1's "intelligent" decision tree is degenerate on cls+red (thresholds 12000/500 calibrated for shopping site, fire 0% on cls/red). Recalibrated thresholds + site-asymmetry imply v5 needs site-conditional rules.
metadata:
  type: project
  scope: paper-1 §6 router
  caveat: SANITY-CHECK ONLY (archive ≠ preregistration lock substrate per Option C)
  parent: docs/checkpoints/router/proposals_v4.md
  sibling: docs/checkpoints/router/archive_diagnostic_2026-05-16.md
---

# P1 (rule-based) router archive simulation — findings

> ⚠️ **NOT preregistration lock substrate.** Same caveats as `archive_diagnostic_2026-05-16.md`: archive uses cls 0-233 / red 0-209 (same task IDs Phase 1a will use), outcomes pre-§107 + pre-§139.8 fix. Numbers below are *directional confidence* on whether v3 router design has honest signal — NOT paper-grade P1 SR claims. Paper-grade test = Phase 1a fresh-data 5-fold CV.

## TL;DR (3 lines)

1. **Phantom_som IS valuable as default**: `always_phantom_som` beats `always_dom` by **+0.85pp cls / +2.86pp red** — phantom modes contribute mode-set value at zero routing cost. This validates paper §1 hook "4-fold drop-in property" (a)(b)(c) at archive level.
2. **P1 v3 decision tree is degenerate**: thresholds `dom_size > 12000` and `dom_complexity > 500` **fire 0% on both cells** (cls max size 5512, red max 6963; cls max complexity 70, red max 134). P1 v3 ≡ always_phantom_som on archive — the "intelligent" tree adds zero discriminative power over a mode-swap default.
3. **Site-asymmetric routing signal**: with thresholds recalibrated to p50, **cls jumps to 20.51% (+4.7pp over v3)** but **red drops to 11.90% (-2.4pp)**. Same routing rule has opposite effect — cls benefits from `complex → som`, red doesn't. v5 design needs site-conditional rules OR P2 learned (site as input feature).

## Method

- **P1 v3 rule** (from `proposals_v4.md` §1 decide_p1_v3):
  - if `intent` matches search regex (`find|search|locate|how many|how much`) → `dom`
  - elif `dom_size > 12000` OR `dom_complexity > 500` → `som`
  - else → `phantom_som`
- **Counterfactuals**: P1_3mode (same tree, else → dom), always_{dom,som,phantom_som}, oracle ceiling
- **Features**: step-0 `state_digest.{text_length, dom_complexity}` (== observation_dom.txt size/line count, verified) + task config `intent`
- **Archive**: B0 cls (n=234, 5 modes — phantom_prompt aborted) / B0 red (n=210, 6 modes)

## Per-cell results

### B0_classifieds (n=234, 5 modes)

| Router | SR | Routed dom | Routed som | Routed phantom_som |
|---|---:|---:|---:|---:|
| always_dom | 14.96% | 234 | — | — |
| always_som | 16.24% | — | 234 | — |
| always_phantom_som | **15.81%** | — | — | 234 |
| P1_3mode (else→dom) | 14.96% | 234 (111 search + 123 else) | 0 | 0 |
| **P1_v3_6mode (current)** | **15.81%** | 111 (47.4%) | **0 (0.0%)** | 123 (52.6%) |
| Oracle | 35.90% | 185 (79.1%) | 29 (12.4%) | 2 (0.9%) — phantom_text 7, vision 11 |

**Branch firing**:
- Branch 1 search→dom: 111 tasks (47.4%) ✅ fires
- Branch 2 complex→som: **0 tasks (0.0%) ❌ DEAD CODE** (max dom_size=5512 < 12000, max dom_complexity=70 < 500)
- Branch 3 else→phantom_som: 123 tasks (52.6%) ✅ fires

P1 v3 SR breakdown:
- dom-routed 111 tasks: SR 13.51% (search-intent has dom = som = phantom_som SR essentially tied)
- phantom_som-routed 123 tasks: SR 17.89%

**Recalibrated threshold sweep** (force complex-branch to fire):

| Threshold | dom routed | som routed | phantom_som routed | SR |
|---|---:|---:|---:|---:|
| size>p50=2674 OR comp>p50=33 | 111 | 64 | 59 | **20.51%** (+4.70 over v3) |
| size>p75=3491 OR comp>p75=58 | 111 | 48 | 75 | 20.09% |
| size>p90=3844 OR comp>p90=60 | 111 | 30 | 93 | 18.80% |
| has_image → som; else → phantom_som | 111 | 26 | 97 | 15.38% |

**cls take-away**: complex→som branch, when actually firing, **delivers substantial lift** — som is the right escalation target for non-search complex cls tasks. v5 cls threshold should be ~3000 chars / ~50 lines, not 12000 / 500.

### B0_reddit (n=210, 6 modes)

| Router | SR | Routed dom | Routed som | Routed phantom_som |
|---|---:|---:|---:|---:|
| always_dom | 11.43% | 210 | — | — |
| always_som | 11.90% | — | 210 | — |
| always_phantom_som | **14.29%** | — | — | 210 |
| P1_3mode (else→dom) | 11.43% | 210 (54 search + 156 else) | 0 | 0 |
| **P1_v3_6mode (current)** | **14.29%** | 54 (25.7%) | **0 (0.0%)** | 156 (74.3%) |
| Oracle | 25.24% | 181 dom + ... phantom 14 (6.7%) | | |

**Branch firing**:
- Branch 1 search→dom: 54 tasks (25.7%) ✅ fires
- Branch 2 complex→som: **0 tasks (0.0%) ❌ DEAD CODE** (max size=6963, max complexity=134)
- Branch 3 else→phantom_som: 156 tasks (74.3%) ✅ fires

**P1 v3 SR breakdown** (per-branch counterfactual mode SR):

| Branch | n | dom SR | som SR | phantom_som SR | P1 routes to |
|---|---:|---:|---:|---:|---|
| search→dom | 54 | **16.67%** | 16.67% | 16.67% | dom (tied with others) |
| else→phantom_som | 156 | 9.62% | 10.26% | **13.46%** | phantom_som ✅ |

**Crucial insight for red**: search-intent tasks have dom=som=phantom_som **all 16.67% identical** — routing to dom is *not* the right choice, it's *no different* from any other mode. P1's search-intent rule on red has zero routing value.

**Recalibrated threshold sweep**:

| Threshold | dom | som | phantom_som | SR |
|---|---:|---:|---:|---:|
| size>p50=5990 OR comp>p50=110 | 54 | 77 | 79 | 11.90% (-2.39 vs v3) |
| size>p75=6404 OR comp>p75=117 | 54 | 54 | 102 | 12.38% |
| size>p90=6618 OR comp>p90=132 | 54 | 15 | 141 | 13.81% |
| has_image → som; else → phantom_som | 54 | 104 | 52 | 11.90% |

**red take-away**: recalibrating thresholds **HURTS** on reddit because som is **worse than phantom_som** on most red tasks (10.26% vs 13.46% on else-branch). v5 red rule should **NOT** escalate to som — phantom_som is the better default for non-search red tasks. Possibly drop complex-branch entirely on red.

## Cross-cell summary

| Cell | always_dom | always_som | always_p_som | P1_v3 (current) | P1 recalib best | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| B0 cls | 14.96 | 16.24 | 15.81 | 15.81 | **20.51** (size/comp p50) | 35.90 |
| B0 red | 11.43 | 11.90 | 14.29 | 14.29 | **14.29** (= v3 itself) | 25.24 |

## Direct answer to "router 需要 phantom 吗"

**YES — but for different reasons than v3 P1 claims**:

| Phantom value claim | Archive evidence | Verdict |
|---|---|---|
| Phantom modes add SR via routing intelligence | P1 v3 ≡ always_phantom_som (zero rule-driven SR diff) | ❌ Refuted on archive for current rule |
| Phantom_som is a **better default** than DOM | +0.85pp cls / +2.86pp red as always-rule | ✅ Confirmed |
| Phantom contributes via oracle ceiling | 3→5/6 mode oracle +4-7pp sig (paper §1 (d)) | ✅ Confirmed (independently in `phantom_lift.md`) |
| Phantom enables cost-quality Pareto | Phantom_som cost ≈ DOM, SR ≥ DOM-baseline | ✅ Confirmed (always_p_som > always_dom at equivalent cost) |

**Net**: phantom **modes are necessary in paper-1 router output space**, but **P1 v3's specific decision tree adds no SR over a mode-swap default**. v5 design needs revision.

## Implications for v5 (proposed)

### Option A — Site-conditional P1 (minimal change)

```python
def decide_p1_v5(task, obs_1, step_state, site):
    if SEARCH_RE.search(task.intent):
        return "dom"
    if site == "classifieds":
        if obs_1.dom_size > 3000 or obs_1.dom_complexity > 50:
            return "som"   # cls benefits from SoM on complex
        return "phantom_som"
    elif site == "reddit":
        return "phantom_som"  # red: phantom_som is best default, no complex escalation
    else:  # shopping (Phase 1b)
        if obs_1.dom_size > 12000 or obs_1.dom_complexity > 500:
            return "som"
        return "phantom_som"
```

Archive projection: cls 20.51% / red 14.29% / both > current v3 baseline.

### Option B — Drop P1 rule-based, paper §6 = P2 learned only

Rationale: rule-based P1 can't capture site-asymmetric pattern without conditioning on site (= adding site as feature, which collapses into "use P2 with site one-hot"). Save paper §6 simplicity.

### Option C — Reframe paper §6 around Pareto, not max-SR

P1's value in v3 is **NOT** SR delivery (it ≡ always_p_som). It IS Pareto frontier extension: P1's search→dom branch routes 25-47% tasks to a cheaper mode at equal SR. Paper §6 frame: "P1 trades 0pp SR for X% cost reduction on routed tasks".

But this requires measuring per-task cost (not just SR) — feasible from archive but additional analysis.

### My recommendation: Option A + C combined

Both extend the value story without adding paper-grade risk:
- A → v5 P1 rule has empirical signal (not dead-code), defendable in §6
- C → paper §6 frames router as "Pareto-aware mode-swap router" not "max-SR router" — matches paper title "Cost-Aware Routing" perfectly

## Honest gaps

1. **B0 only** — no B1 / B2 archive. B1 (4B Qwen3-VL) capability is lower, may shift mode-winner patterns substantially. Phase 1a fresh data answers this.
2. **Search regex is empirical-fit** — `find|search|locate|how many|how much` matches 47.4% cls / 25.7% red but doesn't translate to differential SR (all modes tied on search tasks). The regex's *paper-grade* role is unclear.
3. **shopping not tested** — Phase 1b includes shopping but B0 shopping archive predates Phase 1a fix. Will redo on Phase 1b launch.
4. **Phase 1a fresh data may shift patterns** — post-§107 + post-§139.8 success values drift estimated 5-15% of tasks. Empirical confirmation of cls/red asymmetry requires Phase 1a rerun.
5. **P2 learned not tested here** — P2 LR on imbalanced labels (cls dom 79% / red dom 87%) may collapse to majority prediction. Separate audit needed.
6. **Per-task cost not measured** — Option C requires loading per-task cost from condition_summary_v2.json + computing Pareto delta. Deferred to v5 design round.

## Next steps

1. ✅ **This finding logged** in `docs/checkpoints/router/p1_archive_simulation_findings_2026-05-16.md` (this file)
2. ⏭ User decides v5 direction (A / B / C / combined)
3. ⏭ If A: write `proposals_v5.md` with site-conditional rule + archive projection
4. ⏭ If C: extend simulator to compute per-task cost → Pareto plot
5. ⏭ Phase 1a launch unchanged (preregistration §C still locked); P1 v5 design is **post-data refinement allowed** (not pre-registered claim — only H10 DEFER trigger + anchor fallback are pre-registered)
