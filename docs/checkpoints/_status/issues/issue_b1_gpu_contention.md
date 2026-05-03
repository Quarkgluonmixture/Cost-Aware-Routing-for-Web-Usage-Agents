---
type: issue
category: blocker
status: active
priority: high
action: 联系 seonglae 协调 GPU sharing or 接受 slow progression / RunPod 4090 dedicated 解决 (issue_14cell_phantom_rerun)
updated: 2026-05-03
---

# B1 phantom runner GPU contention (DGX shared)

Current target: **B1 phantom_prompt classifieds** (PID 3826576, 110/234, ~5 ep/h, ETA ~25h). seonglae 并行任务持续抢占 GPU。

## History
- B1 phantom_som cls: ✅ done 2026-05-02 16:41 (despite contention, completed)
- B1 phantom_text cls: ✅ done 2026-05-02 10:46
- B1 phantom_prompt cls: ▶️ running 2026-05-01 launch, 1.0-5.0 ep/h variable rate, ETA ~25h

## Note (2026-05-03)
B1 phantom_prompt cls is **pre-Phase-A** data — will likely be废 after 14-cell rerun on RunPod (`issue_14cell_phantom_rerun.md`). Whether to stop chain step 2 or let it complete depends on RunPod onboarding timeline. Current decision: let it complete (~25h is tolerable).

## Long-term resolution
Issue `issue_14cell_phantom_rerun.md` (RunPod $200 4090 dedicated) eliminates GPU contention entirely for paper-grade rerun.
