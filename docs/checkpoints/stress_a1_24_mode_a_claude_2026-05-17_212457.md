# A1.24 Mode A — Claude /stress findings

Scope: **pre-fire** (8 artifacts, target ≥7 findings + ≥3 OOB)
Artifacts: scripts/maintenance/clear_tasks.py / scripts/maintenance/experiment_watchdog.py / p79/experiment/logger_v2.py / p79/experiment/runner/main.py / p79/experiment/io_utils.py / .claude/CLAUDE.md / Makefile / scripts/maintenance/sync_a100_results.sh

## Verdict (one-sentence)

CLAUDE.md L206 hard rule "clear_tasks 统一入口" misrepresents reality — clear_tasks is **single-script footgun** missing watchdog's 6-layer defense (`.in_progress` marker / `.stale_` archive guard / per-condition flock); `--force` 实证 已 destroy paper-grade data 一次 (pilot wave-1, 2026-04-30). 10 findings (6 P0/P0-OOB, 4 P1/P2).

## Strong claims (survive attack)

- L244-268 B-226 atomic digest write (A1.5 Item 10): temp + fsync + os.replace + dir-fsync — full POSIX atomic. ✓
- L181-185 in-progress protection (steps OR artifacts without summary skip with explicit "use --force") — correct invariant when not bypassed. ✓
- L37 `_EXCLUDED_DIRS` covers (analysis / task_configs / _vwa) ; L73 `not p.suffix` filters json files at run-root. ✓

## Weak claims — principled methodology errors

### Finding 1 [P0 OOB] — `--force` 无 PID-lock / 无 cross-session check
**Claim** — clear_tasks.py:134-135 `--force` help="Also delete tasks that may be in-progress"; L181-185 bypass.
**代码现实** — `--force` 跳过 in-progress 保护, 但 ZERO 其他 check: (a) 不 read `.in_progress` per-episode marker (runner writes, watchdog reads at L1427+L1444); (b) 没 `pgrep -f run_experiment` PID-liveness; (c) 没 ntfy alert; (d) 没 mandatory `--dry-run` pre-step; (e) 没 cross-session lockfile.
**攻击** — 2026-04-30 `docs/archive/analysis_pre_2026-05-15/cross_sites/pilot_t0_decision_final.md:64` 实证: pilot wave-1 (launched 11:19 BST) destroyed at 12:01-12:03 BST by **another Claude Code session** (Myriad config work) 误判 "busy:1 free wait" 为 stuck → ran `clear_tasks.py --force`. 真实是 paper-grade in-flight 数据. 当前 multi-Claude-session 已是常态 (本 session 并发 GRL audit + A1.22), 这 footgun 直接威胁 Phase 1a fire integrity.
**Defuse** — (1) `--force` 检查 `.in_progress` marker presence; (2) `pgrep -f "run_experiment.*${SITE}"` PID check; (3) require `--i-know-this-is-destructive` 双 flag; (4) ntfy 自动 alert.
**Effort** — 30 min
**Confidence** — high

### Finding 2 [P0 OOB] — `_clean_orphan_artifacts` 缺 `.in_progress` marker check
**Claim** — clear_tasks.py:91-93, 109-111 仅 mtime ≥ 10min 保护.
**代码现实** — Watchdog experiment_watchdog.py:1427-1428 + :1444-1445 dual guard: `if (_art / ".in_progress").exists(): continue` + 同 logic for steps file. clear_tasks 完全不 read marker, **仅 mtime**.
**攻击** — B0 baseline 单 task wallclock 5-15min (large-model + LLM eval). 边缘: t=0 episode start (marker), t=1min steps dump, t=2-12min agent reasoning (无 artifacts I/O), t=12min operator 跑 `--clean-orphan-artifacts` → mtime>10min cutoff → wipe near-complete episode. Watchdog 不会犯, clear_tasks 会. 6-layer defense **asymmetric**.
**Defuse** — Copy watchdog L1427-1428 + L1444-1445 marker check (5 lines).
**Effort** — 20 min
**Confidence** — high

### Finding 3 [P0 OOB] — `_clean_orphan_artifacts` 不 skip `.stale_<ts>` archives
**Claim** — clear_tasks.py:84-118 没 `.stale_` filename guard.
**代码现实** — Watchdog L1418 + L1435 显式 skip B-488 archives: `if ".stale_" in _art.name: continue`. B-488 是 runner crash-recovery forensic preservation pattern (no-summary by design).
**攻击** — Runner crash → restart archives prior-attempt 到 `<task>.stale_<ts>` (no summary). operator 跑 `--clean-orphan-artifacts` → wipe forensic → post-hoc 调查不可能. 违反 paper §3.5 evaluator-authority + crash-recovery 链.
**Defuse** — Copy watchdog L1418 + L1435 skip (2 lines).
**Effort** — 10 min
**Confidence** — high

### Finding 4 [P0 OOB] — Concurrent watchdog + clear_tasks orphan race (无 lock)
**Claim** — clear_tasks.py L98 + L116 + L197 + L204 全部 `shutil.rmtree()` / `unlink()` no try/except. Watchdog L1429 + L1446 同.
**代码现实** — 两进程并发同一 condition_dir: process A 先到 `shutil.rmtree(artifact)` 成功, process B 同检测 → 同样 enter rm → `FileNotFoundError` → script exit. 不是 corruption 但 ops fail loud.
**攻击** — DGX cron `glm-update-cells` 10min watchdog snapshot. Operator 手动 `clear_tasks.py --clean-orphan-artifacts` 同时 → race → operator 看 stack trace → "cleanup 失败" → 二次跑 `--force` "解决" → F1 风险叠加.
**Defuse** — Per-condition flock `/tmp/clear_tasks_${run}_${cond}.lock`; EAFP `try/except FileNotFoundError: pass` wrap 4 处.
**Effort** — 30 min
**Confidence** — high

### Finding 5 [P1 OOB] — `--clean-orphan-artifacts` 不清 stale digest 记录
**Claim** — clear_tasks.py:152-154 early return:
```python
if not args.tasks:
    return 0
```
digest cleanup L213-269 仅 task-level fire.
**代码现实** — `--clean-orphan-artifacts` deletes artifact dirs + orphan steps JSONL but NOT digest records. Digest layer **仍 active** (verified `find -name "digest_*.jsonl"` shows `analysis/digest/digest_{dom,som,vision}.jsonl` 在 B1_3mode runs).
**攻击** — Watchdog auto-orphan-clean (L1429) wipe orphan artifact, 而 digest 仍指向被删 task → aggregator pulls stale → silently include phantom episode → paper §1 SR denominator 错. Manual `--clean-orphan-artifacts` 同 bug. task-mode + orphan-mode 不对称 → end-state silently divergent.
**Defuse** — Extract L213-269 digest-clean 块成函数, orphan branch (L152) 之前调用, 用 deleted orphan filename derive task_id_set.
**Effort** — 1h (含 orphan-mode digest invariant test)
**Confidence** — medium (digest 可能 partial retire per B-743, 需 verify is digest still consumer-active)

### Finding 6 [P1 OOB] — Cross-host fire-order race: sync_a100 + clear_tasks
**Claim** — sync_a100_results.sh:57-64 `--delete-after` 注释 mentions B-841 propagation, NO fire-order lock with clear_tasks.
**代码现实** — Comment intent: A100 deletion propagates DGX. Reality: NO ssh-level flock / NO ntfy "deletion start" → DGX rsync (10min cron) 可能在 A100 clear_tasks loop 中跑 → partial mixed state.
**攻击** — t=0 A100 clear_tasks --tasks 85-131 start; t=2s DGX cron rsync start; t=3s rsync 拉 partial (some unlink'd, some still); t=4s clear_tasks done → condition_summary 删; t=6s rsync `--delete-after` 跑 propagation. DGX final = arbitrary mixed snapshot. Paper-grade `make analysis` DGX → SR ≠ A100.
**Defuse** — `flock` on A100 deletion start, sync_a100 wait for lock release; 或: contract doc "rsync wait clear_tasks +60s".
**Effort** — 1h
**Confidence** — medium (race window 短, but paper-grade)

### Finding 7 [P1] — `_parse_task_ids` 无 sanity check
**Claim** — clear_tasks.py:40-50 `_parse_task_ids` accepts arbitrary int range.
**代码现实** — Edge cases: `--tasks 100-99` → `range(100, 100)` = empty silent; `--tasks 5-10000` → 9996 attempts silent; `--tasks 0-` → `int("")` raises ungraceful; `--tasks "  85  - 131  "` lo/hi 没 strip → `int("  85  ")` 实际 OK (Python int 容忍 whitespace).
**攻击** — Operator typo `--tasks 100-99` → "skipped 0 (not found)" → 以为 cleanup 完成 → re-fire runner → resume gate 看到 stale summaries → silent ingest pollution.
**Defuse** — `if lo > hi: raise ValueError`; `if any(t < 0 or t > scored_task_count): raise ValueError`; cap by `len(task_configs)`.
**Effort** — 15 min
**Confidence** — high

### Finding 8 [P1] — `--site` 无 whitelist
**Claim** — clear_tasks.py:129 `--site` no validation.
**代码现实** — Typo `--site shoping` → prefix `shoping_task_0` → no match → "skipped 0" silent.
**攻击** — F7 + F8 复合 typo 路径 → cleanup 静默失败 → re-fire 拉 stale → paper pollution. CLAUDE.md "VWA 只有 shopping/reddit/classifieds 三站" hard rule 没 enforce 到 entry script.
**Defuse** — `_VALID_SITES = {"classifieds", "reddit", "shopping"}` + reject unknown explicit msg.
**Effort** — 5 min
**Confidence** — high

### Finding 9 [P1] — `condition_summary_v2.json` stale-detection fragile
**Claim** — clear_tasks.py:280-282 用 `tc_dir.glob("*.json")` 数 task_configs.
**代码现实** — Current verified `task_configs/` 仅 `reddit_task_*.json` 3 files. Glob 当前 OK. Future siblings (`_manifest.json` / `.skipped.json` / etc.) → `total` +1 spuriously → `remaining < total` 永远 trigger → `condition_summary_v2.json` 永久被 delete → watchdog/runner loop infinite re-aggregation.
**攻击** — 当前 OK 但 fragile against any future task_configs refactor.
**Defuse** — `glob("*_task_*.json")` 精确匹配 P79 naming convention; 或 import `p79.experiment.tasks._enumerate_task_configs` 单一源.
**Effort** — 10 min
**Confidence** — medium

### Finding 10 [P2] — `--clean-orphan-artifacts` 无 default dry-run
**Claim** — clear_tasks.py:131-133 `--clean-orphan-artifacts` fires deletion 默认.
**代码现实** — Best practice: destructive ops 默认 dry-run + `--apply` 显式 confirm. 当前 single-line invocation immediate delete.
**攻击** — F1 multi-session 场景, second session 复制粘贴 zero confirm. Demoted P2 因 F1 已 strongly cover.
**Defuse** — Flip default: `--apply` required; bare = dry-run + summary.
**Effort** — 10 min (但 breaking change, 需 update runbook)
**Confidence** — low (UX vs safety trade-off, defer user 决定)

---

## Phase 0 self-audit

- **Scope**: pre-fire ✓ (8 artifacts, ≥7 findings, ≥3 OOB target)
- **Artifacts**: 8 declared, 8 cited ✓
- **Findings**: 10 (target 7) ✓
- **OOB**: 6 — F1/F2/F3/F4/F5/F6 ✓ (target 3)
- **Specificity**: all findings quote file:line + actual line numbers + 实证 incident pointers ✓
- **Bilingual**: ✓ (中文 attack + English code refs)

## Bug Table (Mode A only — will merge with B+C)

### P0
| # | Bug | Blast Radius | Launch 卡? |
|---|---|---|---|
| P0-1-A* | `clear_tasks.py:134-135,181-185` `--force` 无 PID-lock / 无 marker check / 无 ntfy | `--force` 是 in-progress 保护的 escape hatch. 实证 2026-04-30 pilot wave-1 destroyed by another Claude session 误用 `--force`. 当前 multi-session 常态 → Phase 1a fire 期间任何 session 一行命令可 wipe live data. Paper §1 hero numbers depend on uninterrupted episode JSONL — `--force` footgun 直接 obsoletes 任何 Phase 1a run that survives this window. | 不卡 launch, 卡 paper-grade data integrity throughout fire |
| P0-2-A* | `clear_tasks.py:91-93,109-111` orphan cleanup mtime-only, 缺 `.in_progress` marker | Long-running episode (B0 5-15min wallclock + LLM eval) artifacts mtime>10min cutoff while episode still active. operator 跑 `--clean-orphan-artifacts` → wipe live data. Watchdog L1427+L1444 has marker guard, clear_tasks 不对称. CLAUDE.md "统一入口" claim broken at 6-layer defense level. | 不卡 launch, 卡 long-running B0 episodes mid-flight |
| P0-3-A* | `clear_tasks.py:84-118` 不 skip `.stale_<ts>` archives | B-488 runner crash-recovery 设计 `<task>.stale_<ts>` archives = no-summary by design. clear_tasks orphan-mode 把它们当 orphan wipe → 失去 forensic. 违反 paper §3.5 evaluator-authority + crash-recovery 链. Watchdog L1418/L1435 已 skip — asymmetric. | 不卡 launch, 卡 crash post-mortem |
| P0-4-A* | clear_tasks + watchdog 并发 race 无 lock | DGX cron `glm-update-cells` 10min watchdog scan. Manual clear_tasks 同时 → `shutil.rmtree` race → 2nd process FileNotFoundError → operator 看 stack trace → 误以为 cleanup 失败 → 二次跑 `--force` → F1 footgun 叠加. | 不卡 launch, 卡 ops debug + F1 cascade |

### P1
| # | Bug | Blast Radius | Launch 卡? |
|---|---|---|---|
| P1-1-A* | `clear_tasks.py:152-154` `--clean-orphan-artifacts` early return 不清 digest | Digest layer 仍 active (find verified `analysis/digest/digest_*.jsonl`). Orphan cleanup deletes artifact dirs but digest 仍指向 wiped tasks → aggregator pulls stale → paper §1 SR denominator silently 错. task-mode + orphan-mode 不对称. | 不卡 launch, 卡 paper §1 hero numerator |
| P1-2-A* | sync_a100_results.sh + clear_tasks 跨主机 fire-order 无 lock | A100 clear_tasks loop 与 DGX rsync `--delete-after` 重叠 → DGX final state = arbitrary mixed snapshot. DGX-side `make analysis` SR ≠ A100 真值. B-841 propagation comment mentions intent 但 lock 未 enforce. | 不卡 launch, 卡 cross-host paper-grade integrity |
| P1-3-A | `_parse_task_ids` 无 sanity (lo>hi, out-of-range silent) | typo `--tasks 100-99` silently → 0 deleted reported as "skipped 0" → operator 以为 cleanup ok → re-fire → resume gate sees stale → silent paper pollution. | 不卡 launch, 卡 ops typo recovery |
| P1-4-A | `--site` 无 whitelist enforce | typo `--site shoping` → silent no-op + "skipped 0". CLAUDE.md 三站 hard rule 没 enforce 到 entry script. F7 + F8 complex typo path corrupts cleanup → re-fire → paper pollution. | 不卡 launch, 卡 ops typo recovery |
| P1-5-A | `condition_summary_v2.json` stale logic glob `*.json` fragile | Future task_configs sibling files (e.g. `_manifest.json`) → `total` 误差 → condition_summary 永久被 delete → watchdog/runner loop. 当前 OK 但 fragile. | 不卡 |

### P2
| # | Bug | Blast Radius | Launch 卡? |
|---|---|---|---|
| P2-1-A | `--clean-orphan-artifacts` 无 default dry-run | UX vs safety trade-off; F1 已 cover footgun lesson stronger. | 不卡 |
