---
name: myriad-watcher-silent-miss
type: issue
status: patched
severity: medium
discovered: 2026-05-12
patched: 2026-05-12-late
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

---

## Update 2026-05-12 late — 第二次复发 + 真根因发现 + patch

**2nd recurrence**: Exp 5 cellhprompt cls (359511) + red (359512) Myriad jobs submitted 同 day, 完成 21:42 + 21:54, state.json 显示 `{}`, GONE_HOOK 未触发. 同 yesterday 的 P5a/P5b 完全 same symptom.

**Deeper diagnosis**:
- `ssh_fail_count` file **不存在** → SSH chain 没 None-fail (yesterday's hypothesis 的 "ssh timeout" 路径没被走过)
- `myriad_watcher.log` 0 字节 → watcher 一直 print 不出 error
- State.json mtime fresh → watcher 在每 5 min 写新 state.json
- 但 state.json 是 `{}` → **qstat 返回空但 ssh_chain returned 非 None**

**真根因**: `ssh_chain()` line 156-172 — outer ssh `subprocess.run(... )` returncode=0 但 stdout=空, 因为 **inner ssh (quark→myriad) silent-fail (Cisco VPN drop / 内层 hang)** 时, PowerShell on Windows quark **不 propagate inner exit code 到 outer ssh exit code**. Python parse `parse_qstat("")` → `{}`. State.json overwrite with `{}`. Watcher 视野中 jobs 从未存在.

(yesterday 的 "SSH down → state.json 未触" 假设是错的; 真情况是 outer ssh always returned 0 with empty stdout → state.json **被** 触, 写成 `{}`)

## Fix applied (commit 即将 push)
`scripts/maintenance/glm/myriad_watcher.py`:
1. **Sentinel guard** — `_qstat_with_sentinel()` wrapper: 加 `&& echo __QSTAT_OK__` 到 qstat 命令; stdout 不含 sentinel → 视为 None (chain failure), preserve old_state, 进 SSH_FAIL_FILE 计数路径
2. **Double-probe guard** — `old_state` 非空 + `new_state` 空 → 立即 2nd ssh_chain probe; 第 2 次 None → ntfy high + preserve old_state; 第 2 次 empty → accept

Syntax validated. 不影响 normal qw/r/GONE 流, 只在 silent-fail edge case fire guard.

**Manual recovery 2nd time**:
- `bash scripts/maintenance/auto_pull_myriad_cell.sh 359511 cellhprm_cls stage3_cellhprompt_cls_fwd_ptext_myriad`
- `bash scripts/maintenance/auto_pull_myriad_cell.sh 359512 cellhprm_red stage3_cellhprompt_red_fwd_ptext_myriad`
- Both SENTINEL_OK_PILOT, 5 files / cell, ~1.5 MB

**Status → patched**, monitor 接下来几个 Myriad cycle 看 sentinel guard 是否 work. 如果 SSH chain 再 silent-fail, `ssh_fail_count` 应该 increment 而不是 state.json `{}`.

(post-hoc cron sentinel_scan idea from earlier 还可以作 belt-and-suspenders, 但 sentinel + double-probe patch 应该 cover 主要 silent-fail case)

## References
- commit `00a5ea8` plan.md "P5a + P5b done"
- `scripts/maintenance/glm/myriad_watcher.py:96-105` GONE_HOOKS entries for stage4fv_red / stage4mm_red — 已在表里, 只是 watcher 没看到
- `scripts/maintenance/auto_pull_myriad_cell.sh` — base64 chain transfer reference
