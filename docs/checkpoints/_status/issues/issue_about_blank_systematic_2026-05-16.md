---
type: issue
category: paper-grade-investigation
status: open
priority: medium
action: 4-phase systematic study of about:blank trigger + recovery + paper-grade attribution decision. NOT a direct patch — user explicitly redirected from patch path 2026-05-16.
created: 2026-05-16
updated: 2026-05-16
paper_section: "§3.4 (SR canonical) + §3.5 (transparency metrics)"
audit_source: /stress A1.4a Claude Mode A F1
file_paths:
  - p79/experiment/runner/main.py
  - scripts/analysis/about_blank_frequency.py
  - docs/checkpoints/实验笔记.md
b_number: B-? (deferred; depends on Phase 4 outcome)
---

# F1 about:blank silent attribution — systematic study

## Background (Claude Mode A F1, /stress A1.4a 2026-05-16)

`p79/experiment/runner/main.py:1052-1090` recovers when post-step URL is `about:blank` by navigating to `task.raw_task.start_url`. Recovery succeeds → `state_after = build_page_state(retry_obs)` reflects the start_url. Then `detect_page_state_change(state_before, state_after, action_type)` is called with **the original (failed) action_type**, comparing pre-action page state with post-recovery start-url state. Result: huge state delta → most page_change_reasons trigger → `action_success=True` attributed to the failed action.

Paper-grade concern: Cluster 1 ON_TARGET rate (paper §3 evidence layer for B-01/02/33) may be inflated because failed-action-then-about:blank-then-recovery counts as success. Cross-baseline impact depends on per-site about:blank frequency.

## User Q&A redirect (2026-05-16)

> "about:blank 的问题我觉得应该系统研究下"

User redirected from direct-patch path (force `action_success=False, page_change_reasons=["about_blank_recovery"]` only) to systematic study path. Rationale: about:blank handling is consequential enough that **either way** (current attribution vs. force-fail) needs empirical justification, not heuristic patch.

## 4-phase investigation plan

### Phase 1 — Measure frequency

Cross site × mode × baseline, count steps where:
- `page_change_reasons` contains `"about_blank_recovery"`, OR
- `state_digest.url_after.startswith("about:blank")`

Output: table of about:blank rate per (cls/red/shop) × (dom/som/vision/phantom_*) × (B0/B1/B2).

Tool: `scripts/analysis/about_blank_frequency.py` (Phase 1 product).

### Phase 2 — Trigger pattern

For each about:blank step, classify the **preceding action**:
- `click` on element with `target="_blank"` attribute
- `click` on `window.open()` JS hook
- `tab_focus` to non-existent tab
- `hover` triggering popup
- Other / unknown

Output: action_type → about:blank rate distribution. If clear pattern (e.g., 80% click on target=_blank), framework-level fix at locator_dispatch (use Playwright's `expect_popup` / `expect_page`) becomes high-priority.

### Phase 3 — Recovery 后果

Compare downstream SR / step count for:
- Trajectories with ≥1 about:blank recovery
- Trajectories without any about:blank

If recovered trajectories have systematically lower SR (e.g., agent confused by suddenly being on start_url), recovery is doing harm even after the framework "saves" the episode. If recovered trajectories have higher SR (rare but possible — sometimes agent benefits from a clean restart), current attribution is paper-grade defensible.

### Phase 4 — Handling decision

Based on Phase 1-3 data:

- **Path A (current, status quo)**: Recovery + silent attribution. Defensible if (a) about:blank rate <1%, OR (b) recovered trajectories have neutral / positive SR delta. Paper §3.5 add disclosure paragraph.
- **Path B (force fail)**: Mark `action_success=False, page_change_reasons=["about_blank_recovery"]` only. Defensible if recovered trajectories systematically harm SR. Likely also need to NOT advance `step_idx` (treat as no-op).
- **Path C (abort)**: Raise `RuntimeError("about_blank_unrecoverable")` from recovery branch → episode terminates as env_error. Defensible only if about:blank rate is rare (<0.5%) AND consistently harmful.

## Status

- [x] User Q&A redirect logged (2026-05-16)
- [x] Issue file created (this file)
- [ ] Phase 1: write `scripts/analysis/about_blank_frequency.py` and run on Phase 1a data
- [ ] Phase 2: extend script with action-type classification
- [ ] Phase 3: trajectory comparison
- [ ] Phase 4: handling decision + chronicle update + paper §3.5 prose update

## Cross-link

- Chronicle: 实验笔记 §148.2 (user Q&A) + §148.6 (G4 mini-investigation track)
- Master bug catalog: §149 audit status table (under "🔬 Mini-investigation")
- Code: `p79/experiment/runner/main.py:1052-1090`
- Analysis script: `scripts/analysis/about_blank_frequency.py` (Phase 1 product, this commit)
