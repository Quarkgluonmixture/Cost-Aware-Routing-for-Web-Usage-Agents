---
type: issue
category: bug
status: resolved
priority: medium
action: ✅ Fixed 2026-05-04 — `experiment_watchdog.py` + 4 queue scripts patched
created: 2026-05-04
resolved: 2026-05-04
---

# experiment_watchdog.py 没 self-exit, runner 完成后变 init-orphan

## Symptom

Runner 完成 condition (写 `condition_summary_v2.json`) 退出后, watchdog 不会自动退出, 而是继续每 30s poll → annotate (`0 screenshots`) → regenerate gallery 永远循环。Parent shell (queue_chain) 退出后, watchdog 被 init reaped 成 PPID=1 的孤儿, 仍跑数天 (实测 PID 3826604 跑 3d 8h 才被手 kill)。

## Root cause

`scripts/maintenance/experiment_watchdog.py` line 1302 `while True:` 主循环只通过 `--once` flag 或 SIGTERM 退出 (line 1689 `if args.once: break`)。`condition_completed = (cond_dir / "condition_summary_v2.json").exists()` (line 1340) 这个 boolean 只在 retry 决策内用 ("condition done 后不再 retry error episodes"), **从不触发 break/exit**。

watchdog 也不接收 `--runner-pid`, 无法 detect "我守的 runner 死了 → 该退出了"。

## Concrete impact

- 单 watchdog idle loop ~几 KB log/h × 几天 = MB 级 disk waste + 占 1 个 file descriptor
- PLAYBOOK §1 反复显示 ZOMBIE-WATCHDOG 干扰 critical-path narrative
- cron `glm-refresh-playbook` 拿 stale watchdog 当 active run, 误导 decision

## Proposed fix (1 file)

`scripts/maintenance/experiment_watchdog.py`:

1. 加 `--runner-pid` argparse + 主循环 check `_pid_alive(runner_pid)`, dead → break (with grace period 60s 等 final summary 写入)
2. 或者 (更简) 主循环检查 `condition_completed AND no_new_episode_for(N min)` → break
3. 同步 update `queue_phantom_*.sh` 把 `--runner-pid $RUNNER_PID` 传给 watchdog spawn

## Verification

`B1_phantom_prompt_classifieds_20260501_v2.log` 含数千行 `[GALLERY] regenerated` (~每 30s 一行) 跨 3+ 天, 没有 `[POST-ANALYSIS]` 之外的有效活动。Run 早在 5/4 早上就 234/234 done。

## Workaround until fix

`make active` 现 detect 这种状态显示 `ZOMBIE-WATCHDOG (no runner)`; 手动 `kill -TERM <PID>` 干净退出 (不需 SIGKILL)。

## Resolution (2026-05-04)

**`scripts/maintenance/experiment_watchdog.py`**:
- Added `--runner-pid` argparse (optional, int).
- Main loop end (before `if args.once: break`) now checks single-condition self-exit:
  - **Path A** (`--runner-pid` passed): `os.kill(pid, 0)` liveness probe; `condition_summary_v2.json` exists + runner dead → log + `_persist_state()` + break.
  - **Path B** (legacy launchers no `--runner-pid`): `condition_summary_v2.json` exists + idle ≥ `idle_alert_secs` → break.
- Multi-condition mode unchanged (no easy bound).

**4 queue scripts** (`queue_phantom_som.sh` / `queue_phantom_text.sh` / `queue_phantom_prompt.sh` / `queue_baseline.sh`):
- Added `RUNNER_PID=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)` capture before watchdog spawn.
- `${RUNNER_PID:+--runner-pid "${RUNNER_PID}"}` conditional flag in spawn args (graceful when pgrep returns empty).

**Functional verification**: Spawned watchdog on the previously-orphaned `B1_phantom_prompt_classifieds_20260501` run with `--runner-pid 999999` (bogus) — log shows `[watchdog] condition ... complete + runner pid=999999 dead → exiting` followed by `[watchdog][FINAL]`, exited within ~30s instead of looping for days.
