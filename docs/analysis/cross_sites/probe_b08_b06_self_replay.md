# Self-Replay Probe — B-08 SCROLL + B-06 SELECT_OPTION

Audit date: 2026-04-30

## Methodology

Self-replay probe uses logged state_digest.scroll_y_before/after instead of fresh-state replay (avoids the codex 'didn't replay prior steps' issue). For SCROLL: classifies via logged delta + obs_url page geometry (scrollHeight, viewport, starting scroll_y). For SELECT: navigates to obs_url, identifies element at logged bbox, separates native <select> from custom div dropdown, then verifies framework's no-args .select_option() dispatch path.

## SCROLL silent failure (B-08)

- Tier 2 claim: 667 ep / 14.85%
- Probed: 20 (replay ok: 19)
- Scaffold fraction of replayed: **0.053**
- Legit fraction of replayed: 0.947
- False-positive-of-Tier2 (scroll actually moved ≥5px): 0
- Extrapolated true blast radius: **35 ep**
- Breakdown: {'LEGIT_AT_BOTTOM': 18, 'REPLAY_FAIL': 1, 'SCAFFOLD_SCROLL_BUG': 1}

## SELECT_OPTION arg-drop (B-06)

- Tier 2 claim: 149 ep / 3.32%
- Probed: 20 (replay ok: 20)
- Scaffold (native <select> + arg-drop) fraction of replayed: **0.1**
- Non-native (custom div dropdown) fraction: 0.9
- Extrapolated true arg-drop blast radius: **15 ep**
- Breakdown: {'OTHER_CUSTOM_DROPDOWN': 18, 'SCAFFOLD_SELECT_ARG_DROP': 2}

## Comparison to Codex `probe_audit_verification`

- Codex SCROLL: 10 cases, scaffold fraction 0.3
- Self-replay SCROLL: 20 cases, scaffold fraction 0.053

- Codex SELECT: 8 cases, scaffold fraction 0.286
- Self-replay SELECT: 20 cases, scaffold fraction 0.1

## Per-case detail

### SCROLL cases

- classifieds task 11 step 10 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (2027+720≥2747) | logged delta=0px
- classifieds task 29 step 15 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (1855+720≥2575) | logged delta=0px
- classifieds task 32 step 6 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (1955+720≥2675) | logged delta=0px
- classifieds task 16 step 10 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (1887+720≥2607) | logged delta=0px
- classifieds task 29 step 6 (DOM) → **REPLAY_FAIL** | playwright error: TimeoutError: Page.goto: Timeout 15000ms exceeded.
Call log:
  - navigating to "http://100.95.81.103:9 | logged delta=0px
- classifieds task 11 step 11 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (2027+720≥2747) | logged delta=0px
- classifieds task 32 step 5 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (1955+720≥2675) | logged delta=0px
- classifieds task 24 step 11 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (1002+720≥1722) | logged delta=0px
- classifieds task 24 step 6 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (1002+720≥1722) | logged delta=0px
- classifieds task 6 step 11 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (2067+720≥2787) | logged delta=0px
- classifieds task 5 step 14 (DOM) → **SCAFFOLD_SCROLL_BUG** | room=568px expected≈576px but scroll_y stayed at 1818 | logged delta=0px
- classifieds task 24 step 12 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (1002+720≥1722) | logged delta=0px
- classifieds task 16 step 9 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (1887+720≥2607) | logged delta=0px
- classifieds task 11 step 4 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (2027+720≥2747) | logged delta=0px
- classifieds task 12 step 5 (DOM) → **LEGIT_AT_BOTTOM** | already at bottom (1228+720≥1948) | logged delta=0px
- classifieds task 63 step 4 (Vision) → **LEGIT_AT_BOTTOM** | already at bottom (1915+720≥2635) | logged delta=0px
- classifieds task 135 step 9 (Vision) → **LEGIT_AT_BOTTOM** | already at bottom (1875+720≥2595) | logged delta=0px
- classifieds task 2 step 7 (Vision) → **LEGIT_AT_BOTTOM** | already at bottom (1755+720≥2475) | logged delta=0px
- classifieds task 134 step 2 (Vision) → **LEGIT_AT_BOTTOM** | already at bottom (197+720≥917) | logged delta=0px
- classifieds task 159 step 5 (Vision) → **LEGIT_AT_BOTTOM** | already at bottom (2386+720≥3106) | logged delta=0px

### SELECT cases

- classifieds task 122 step 8 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is LABEL not <select> — different code path, not arg-drop bug
- classifieds task 154 step 6 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is LABEL not <select> — different code path, not arg-drop bug
- classifieds task 133 step 24 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is LABEL not <select> — different code path, not arg-drop bug
- classifieds task 133 step 22 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is LABEL not <select> — different code path, not arg-drop bug
- classifieds task 133 step 1 (DOM) → **SCAFFOLD_SELECT_ARG_DROP** | native <select> with 24 options; framework calls .select_option() with NO args (line 1395 actions.py) → would clear or n
- classifieds task 133 step 17 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is LABEL not <select> — different code path, not arg-drop bug
- classifieds task 154 step 7 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is LABEL not <select> — different code path, not arg-drop bug
- classifieds task 114 step 16 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is LABEL not <select> — different code path, not arg-drop bug
- classifieds task 145 step 1 (DOM) → **SCAFFOLD_SELECT_ARG_DROP** | native <select> with 24 options; framework calls .select_option() with NO args (line 1395 actions.py) → would clear or n
- classifieds task 154 step 5 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is LABEL not <select> — different code path, not arg-drop bug
- classifieds task 133 step 23 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is LABEL not <select> — different code path, not arg-drop bug
- reddit task 169 step 1 (SoM) → **OTHER_CUSTOM_DROPDOWN** | target is SPAN not <select> — different code path, not arg-drop bug
- reddit task 113 step 3 (SoM) → **OTHER_CUSTOM_DROPDOWN** | target is SPAN not <select> — different code path, not arg-drop bug
- reddit task 28 step 3 (SoM) → **OTHER_CUSTOM_DROPDOWN** | target is SPAN not <select> — different code path, not arg-drop bug
- reddit task 183 step 3 (SoM) → **OTHER_CUSTOM_DROPDOWN** | target is SPAN not <select> — different code path, not arg-drop bug
- reddit task 187 step 1 (SoM) → **OTHER_CUSTOM_DROPDOWN** | target is SPAN not <select> — different code path, not arg-drop bug
- reddit task 186 step 1 (SoM) → **OTHER_CUSTOM_DROPDOWN** | target is SPAN not <select> — different code path, not arg-drop bug
- shopping task 252 step 26 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is INPUT not <select> — different code path, not arg-drop bug
- shopping task 252 step 28 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is INPUT not <select> — different code path, not arg-drop bug
- shopping task 251 step 6 (DOM) → **OTHER_CUSTOM_DROPDOWN** | target is INPUT not <select> — different code path, not arg-drop bug