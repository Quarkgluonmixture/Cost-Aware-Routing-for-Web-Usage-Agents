---
type: audit
ref: B-1842~1847
title: parse-error rescue accounting symmetry
status: deferred
priority: P1
effort: 2-3h
phase: post-fire
blocker: fire 跑中 p79/ immutable
---

# B-1842~1847 · parse-error rescue accounting symmetry

3-AI /stress 2026-05-23, disclosure 已 land `section4_limitations_disclosure.md §4.X.19`。
forward code remediations:
- canonical-latency 加 `−parse_error_injected_wait_ms` 对称扣除 (B-1842)
- rename `parse_error_rate`→`injected_wait_rate` + `parse_valid_before_rescue` flag 区分 rescue-wait vs model-wait (B-1843)
- `no_progress_rate` per-cell covariate (B-1844)
- `termination_reason` episode 字段 (B-1846)
- B0 `tool_call_emit_rate≥0.95` condition gate in `paper_grade_check.py` (B-1847)

实测 sink rate B0≈0 / B1 0 / B2 0.7% → 量级 negligible, disclosure 已充分。
deferred post-fire (fire 跑中 `p79/` code immutable — 续链 spawn import)。
