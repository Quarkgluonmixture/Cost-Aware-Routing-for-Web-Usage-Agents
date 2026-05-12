---
name: myriad-watcher-silent-miss
type: issue
status: open
severity: medium
discovered: 2026-05-12
file_paths:
  - scripts/maintenance/glm/myriad_watcher.py
  - logs/cron/myriad_state.json
---

# Myriad watcher state-diff blind to SSH-down windows

## Symptom (2026-05-12)
- P5b 353890 + P5a 354382 跑 r 状态 06:25 → GONE 07:31/08:09. cron `*/5min` 应抓 NEW + GONE 事件并 fire `_dispatch_gone_hook` → `auto_pull_myriad_cell.sh`.
- 实际: `myriad_state.json` 全程只有 single `{"353763": qw}`. P5a/P5b 从未进入 watcher state machine. auto_pull 没 fire. 用户手动 base64 SSH chain 拉数据.

## Root cause
`scripts/maintenance/glm/myriad_watcher.py:207-225` — 当 `ssh_chain` returns `None` (SSH timeout/error), main loop **silently exits with state.json untouched**, increments `SSH_FAIL_FILE` counter, sends ntfy only after 3 consecutive fails. SSH-down 段 (推测 quark Tailscale fluctuation overnight 06:25-08:09) 让 state machine 跳过整个 r 阶段; 当 SSH 恢复, new_state 与旧 state.json 一致 (P5a/P5b 已 GONE 看不到), `diff_states` 0 events, GONE 永久丢失.

## Why not just "fix the silent exit"
State-diff approach 本质假设 cron 总能 sample r 阶段. 即使 add verbose log on SSH fail, 也无法 recover 已错过的 GONE event.

## Fix proposal (post-hoc safety net)
新建 cron entry `myriad_sentinel_scan.py @ */30min`:
1. SSH list `/home/ucab352/Scratch/p79/results/mechanistic/*/pilot_summary.md` mtime 远端
2. 对比本地 `results/mechanistic/*/pilot_summary.md` 缺失或 mtime 旧的 dir → trigger auto_pull
3. Idempotent — 已 pull 的跳过

GONE_HOOKS 的 prefix-match 完全不依赖, 直接基于 result-dir manifest 反向同步. 不替代现有 watcher, 是 safety net.

## Workaround (until fix)
P5a + P5b 已手动 base64 chain pull. P4 (353763 qw 16h+) 在 watcher 视线内, 完成时 GONE_HOOK 会正常 fire (前提: SSH chain 不再 down).

## References
- commit `00a5ea8` plan.md "P5a + P5b done"
- `scripts/maintenance/glm/myriad_watcher.py:96-105` GONE_HOOKS entries for stage4fv_red / stage4mm_red — 已在表里, 只是 watcher 没看到
- `scripts/maintenance/auto_pull_myriad_cell.sh` — base64 chain transfer reference
