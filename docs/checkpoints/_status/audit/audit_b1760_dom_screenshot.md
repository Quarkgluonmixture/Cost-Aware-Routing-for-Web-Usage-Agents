---
type: audit
ref: B-1760
title: DOM mode screenshot.png regression
status: deferred
priority: P1
effort: 2h
phase: post-fire
blocker: post cls B0 SoM land
---

# B-1760 · DOM mode screenshot.png regression

`obs.image=None` for accessibility_tree across 91/91 step records on Fire-3 cls B0 DOM。
Archive 2026-05-15 had it; logic byte-identical archive↔HEAD; runtime instrument needed。

Trigger: post cls B0 SoM cell land。Acceptance: re-fire smoke6 / 10-task pilot, verify
`screenshot.png` per step + `annotate_screenshots.py` produces `screenshot_annotated.png`。

Paper §3 evidence layer NOT blocked (DOM trajectory + schema-v2 fields present);
screenshot is audit-layer only。
