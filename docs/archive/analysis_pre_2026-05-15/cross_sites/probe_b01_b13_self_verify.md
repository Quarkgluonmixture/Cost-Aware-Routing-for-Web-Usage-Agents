# Self-Verify Probe — B-01 TYPE 100% + B-13 NOT_A_BUG

Audit date: 2026-04-30

## Purpose

Self-verify B-01 TYPE 100% scaffold (codex claim) + B-13 NOT_A_BUG (codex 0/5 with 3 REPLAY_FAIL).

## B-01 TYPE Silent Failure

- Codex probe_audit_verification claim: scaffold fraction **1.0** (15/15)
- Self-verify probed: 12 cases, replay ok: 11
- Breakdown: {'SCAFFOLD_TYPE_BUG': 11, 'REPLAY_FAIL': 1}
- Strict scaffold fraction (only SCAFFOLD_TYPE_BUG): **1.0**
- Lenient (incl. NEAR_INPUT_BUT_OFFSET): **1.0**
- EDITABLE_AT_CENTER fraction: **0.0** (these are NOT bugs — agent's center actually hits an input)

## B-13 action_fail_but_page_changed

- Tier 4 I2 violations: 25
- Codex probe_audit_verification claim: 0/5 scaffold (3 REPLAY_FAIL + 2 REPLAY_DID_NOT_CHANGE)
- Self-verify probed: 8 cases via state_digest log analysis (no Playwright replay — independent of codex's REPLAY_FAIL artifacts)
- Breakdown: {'REPLAY_FAIL': 2, 'PAGE_CHANGED_FALSE_TRIGGER': 6}
- Runner false negative count: **0**
- Verdict: **NOT_A_BUG_CONFIRMED**

## Per-case detail

### B-01 TYPE
- classifieds task 102 step 8 (DOM) → **SCAFFOLD_TYPE_BUG** | center hits HEADER (no nearby input within 6 ancestors) — Meta+A would select page text
- classifieds task 29 step 16 (DOM) → **SCAFFOLD_TYPE_BUG** | center hits HEADER (no nearby input within 6 ancestors) — Meta+A would select page text
- classifieds task 159 step 9 (SoM) → **SCAFFOLD_TYPE_BUG** | center hits HEADER (no nearby input within 6 ancestors) — Meta+A would select page text
- classifieds task 29 step 10 (DOM) → **SCAFFOLD_TYPE_BUG** | center hits HEADER (no nearby input within 6 ancestors) — Meta+A would select page text
- classifieds task 29 step 9 (DOM) → **SCAFFOLD_TYPE_BUG** | center hits HEADER (no nearby input within 6 ancestors) — Meta+A would select page text
- classifieds task 8 step 23 (SoM) → **SCAFFOLD_TYPE_BUG** | center hits SELECT (no nearby input within 6 ancestors) — Meta+A would select page text
- classifieds task 29 step 17 (DOM) → **SCAFFOLD_TYPE_BUG** | center hits HEADER (no nearby input within 6 ancestors) — Meta+A would select page text
- classifieds task 198 step 6 (SoM) → **SCAFFOLD_TYPE_BUG** | center hits HEADER (no nearby input within 6 ancestors) — Meta+A would select page text
- classifieds task 64 step 6 (DOM) → **SCAFFOLD_TYPE_BUG** | center hits HEADER (no nearby input within 6 ancestors) — Meta+A would select page text
- classifieds task 11 step 7 (DOM) → **REPLAY_FAIL** | auth/obs_url/bbox missing
- classifieds task 29 step 11 (DOM) → **SCAFFOLD_TYPE_BUG** | center hits HEADER (no nearby input within 6 ancestors) — Meta+A would select page text
- classifieds task 5 step 15 (DOM) → **SCAFFOLD_TYPE_BUG** | center hits HEADER (no nearby input within 6 ancestors) — Meta+A would select page text

### B-13 I2
- classifieds task 226 step 7 (P-SoM) → **REPLAY_FAIL** | step_idx out of range
- classifieds task 226 step 8 (P-SoM) → **REPLAY_FAIL** | step_idx out of range
- classifieds task 231 step 1 (Vision) → **PAGE_CHANGED_FALSE_TRIGGER** | logged page_changed=True but no url/title/scroll/text_similarity evidence — false trigger
- classifieds task 231 step 2 (Vision) → **PAGE_CHANGED_FALSE_TRIGGER** | logged page_changed=True but no url/title/scroll/text_similarity evidence — false trigger
- classifieds task 232 step 1 (Vision) → **PAGE_CHANGED_FALSE_TRIGGER** | logged page_changed=True but no url/title/scroll/text_similarity evidence — false trigger
- classifieds task 232 step 2 (Vision) → **PAGE_CHANGED_FALSE_TRIGGER** | logged page_changed=True but no url/title/scroll/text_similarity evidence — false trigger
- reddit task 183 step 1 (Vision) → **PAGE_CHANGED_FALSE_TRIGGER** | logged page_changed=True but no url/title/scroll/text_similarity evidence — false trigger
- reddit task 183 step 2 (Vision) → **PAGE_CHANGED_FALSE_TRIGGER** | logged page_changed=True but no url/title/scroll/text_similarity evidence — false trigger