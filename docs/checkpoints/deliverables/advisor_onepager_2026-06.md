---
type: deliverable
status: draft
audience: advisor
deliverable: 1 of 2
created: 2026-06-10
updated: 2026-06-10
---

# Cost-Aware Routing for Web-Usage Agents

**Project summary, research questions, experiments & timeline** · Jiaming Wei · 10 June 2026 · *advisor written-feedback deliverable 1 of 2*

## 1. Goal

Web agents read pages through different *observation representations* — accessibility-tree text (**DOM**), Set-of-Marks annotated screenshots (**SoM**), or raw screenshots (**Vision**) — whose token cost and latency differ by large factors, while success rates differ *per task* rather than uniformly. This project (a) characterises the success–cost trade-off of **six** representation modes on VisualWebArena, including a newly identified **phantom routing space**: three hybrid modes (**P-text, P-prompt, P-SoM**) that keep SoM-style text or prompt structure while dropping the annotated image; and (b) tests whether **cost-aware, per-task routing** over these modes improves the success–cost Pareto frontier beyond any fixed mode. Per advisor steers (14 & 29 May): scope = phenomenon + router (mechanism analysis parked); initial venue = workshop; the thesis is the full write-up.

## 2. Research questions

1. **RQ1 — Drop-in phenomenon** (H1/H2, primary): Is P-SoM a "drop-in" routing arm — cost ≈ DOM, ~50 % lower latency than SoM, success not significantly below the best baseline — such that *removing* it from the routing menu measurably costs success? Gate: one-sided fixed-effects pooled drop-one test, θ_FE > +1.0 pp, α = 0.05, k = 6 (site × model) cells.
2. **RQ2 — Structure** (H3): Is the phantom space organised by two independent axes (text format × prompt style), i.e. a structural region rather than one lucky configuration?
3. **RQ3 — Learned routing** (H10): Can a learned router, using pre-execution task features, choose per-task modes that are **Pareto non-dominated** (success vs cost) against all six fixed-mode baselines?
4. **RQ4 — Secondary**: Which failure classes are *routing-rescuable* (cross-mode failure taxonomy)? Do findings hold across model scale and family? All claims are reported against a measured run-to-run noise floor (disclosed limitation, per 29 May).

## 3. Experiments (pre-registered: OSF DOI 10.17605/OSF.IO/9QCWU)

- **Pass-1 — 36 baseline conditions**: {classifieds, reddit} × {Qwen3-VL-235B (API), Qwen3-VL-4B (local), Gemma3-4B (local)} × {DOM, SoM, Vision, P-text, P-prompt, P-SoM}; 224 / 205 scored tasks per site; paper-grade protocol (per-condition site reset, one-baseline-per-site, watchdog + run manifest, witnessed amendments).
- **Pass-2 — 6 learned-router conditions**: one per site × model cell, trained on Pass-1 outcomes with site-stratified cross-validation.
- **Analysis**: FE-pooled drop-one gate (RQ1) · axis non-overlap bootstrap (RQ2) · Pareto non-dominance + cost/latency accounting (RQ3) · 3-tier failure attribution feeding the cross-mode taxonomy (RQ4).
- **Status (10 Jun)**: 12/36 conditions complete (235B + 4B Qwen on classifieds, 6/6 modes each); Gemma3 classifieds running (mode 1/6 at ~58 %); reddit auto-chained to follow.

## 4. Self-imposed deadlines (key milestones; full 11-milestone table tracked in the repo and checked every session)

Advisor-set hard deadlines: **this one-pager ≤ 22 Jun (ASAP)** · **literature-review chapter ≤ 20 Jul**. Self-targets are deliberately earlier, leaving review slack before each hard deadline.

| Date | Milestone |
|------|-----------|
| **Fri 12 Jun** *(hard: 22 Jun)* | **This one-pager → advisor (deliverable 1)** |
| Fri 26 Jun | Pass-1 complete: all 36 conditions landed (compute-bound; serialized on one A100) |
| Wed 08 Jul | Full statistical analysis + Pass-2 router trained, run, and judged |
| **Mon 13 Jul** *(hard: 20 Jul)* | **Literature-review chapter draft → advisor (deliverable 2)** |
| Mon 10 Aug | Full thesis draft v1 → advisor; revisions incorporated by 24 Aug |
| early Sep (TBC) | Official submission (exact date to be confirmed) |

Writing deadlines run in parallel with compute and are not blocked by it. About one week of slack before the August writing block absorbs the main risks: cluster/VM interruptions (resume protocol in place; completed runs immutable), slow small-model runtimes, and API serving nondeterminism (disclosed limitation, per 29 May).
