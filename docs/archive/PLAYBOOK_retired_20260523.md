---
type: playbook
status: rolling
last_review: 2026-05-13
audience: self-only
sync_transport: tailscale-rsync
---

# PLAYBOOK

> ⚠️ **GLM 自动刷新 2026-05-13 后停摆** — §1/§2 的 live status 自 2026-05-13 起未再被 GLM cron 更新（`last_review: 2026-05-13`）。下方 critical path / cron 健康度**非实时、已严重过期**（仍停在 A100-fire 前世界）。**实时状态请用 `make active` CLI + `cells.base`**。本文件 git-untracked（Tailscale rsync 同步）；体系定位待决策。

---

## §1 当前 critical path snapshot

▶️ GPU 空闲，无任何 runner/watchdog 进程。两个 active cell 停在 2%：B0 classifieds P-prompt（critical, post-fix）等 B1 phantom_prompt cls 解封（~25h），B1 reddit P-SoM 被 queue_chain Tier 1 step 2 阻塞——两条链串行等待，实际零推进。今早 git 连推 2 个 audit commit（301b28e pipeline 27 findings + fc208a5 stress-v5 redesign），昨天 phase1 audit 6 条修复已落地。make active 确认 GPU 完全没人在用。

⏳ 4 个 pending cell 全卡队列链：B1 reddit P-text 等 Tier 1 step 4；B1 reddit P-prompt 已被 issue_14cell 的 A100 16-cell 重跑取代（pre-Phase-A 作废）；shopping 站 P-SoM 和 P-text 都等 B0 dom shopping 先完。瓶颈就一个——A100 SSH 验证解锁后 shopping 5-cell 和 16-cell 重跑同时动。

🔴 三个 high-priority issue 串依赖链：phase1_audit（今日，6 条 paper-grade 发现，~3h 修）→ paper_grade_rerun_5cells（5 个 shopping cell）→ 都等 A100 SSH。advisor_sync 走独立关键路径，preregistration DOI 全卡导师回复 11 个问题。14cell_phantom_rerun 也 blocked 等 advisor + A100。b1_shopping_dom（medium）同样等 A100 SSH。

👉 建议: 先验证 A100 SSH 连通性（一石三鸟解锁 3 issue），然后按 Recommended fix order 推 phase1_audit，争取今天下午触发 5-cell paper-grade rerun。
---

## §2 自动化运行状态

### 2.1 Cron job 健康度 (last 24h)

| Job | 上次 run | 状态 | 备注 |
| :--- | :--- | :--- | :--- |
| glm-update-cells | 2026-05-13T10:00+00:00 | ✅ | 更新 0/26 cells，无变动 |
| error-scan | 2026-05-13T09:05+00:00 | ✅ | 132 文件，0 错误，disk 14% free |
| myriad-watcher | 2026-05-09T07:45+00:00 | ⚠️ | log 空（0 bytes），cron 可能停了 |
| check-links | 2026-05-10 | ✅ | 无 BROKEN/missing/WARN |

### 2.2 Cell 状态变更近况 (changelog tail)

- 05-04 13:10 cell_b1_red_psom: pid_dead cleared, progress→2%
- 05-04 13:06 cell_b0_cls_pprompt: pid_dead cleared, progress→2%
- 05-04 12:56 cell_b0_cls_pprompt: status→active, pid→175127
- 05-04 12:33 cell_b1_red_psom: status→active, pid→135755
- 05-04 12:05 6 cells batch done (B1 cls som/vision/dom + B1 red som/dom/vision)
- 05-04 12:05 cell_b0_red_som/dom: rerun_detected, status→done

（自 05-04 后无新变更，距今 9 天）

### 2.3 Dead link warnings

✅ 无 broken link（最新检查 2026-05-10）

### 2.4 Ntfy fail alerts 历史

- 2026-05-13T01:45: ⚠️ error-scan fail
- 2026-05-13T01:40: ⚠️ error-scan fail
- 2026-05-13T01:40: ⚠️ glm-update-cells fail
- 2026-05-13T01:30: ⚠️ glm-update-cells fail
- 2026-05-13T01:30: ⚠️ error-scan fail
- 2026-05-13T01:25: ⚠️ error-scan fail
- 2026-05-13T01:20: ⚠️ glm-update-cells fail
- 2026-05-13T01:20: ⚠️ error-scan fail

（01:20-01:45 集中爆发 8 次，之后恢复，当前正常）

### 2.5 🔴 Active errors / warnings (runner / watchdog log scan, last 24h)

✅ 近 24h 无 runner / watchdog 错误 (扫了 132 个 log 文件)
---

## §3 Session bootstrap

> 一句话给 Claude: 读 [[实验笔记]] 最后几个 § + [[paper_planning]] + [[next_steps]] + [[issue_advisor_sync_2026-05-14]] + [[PLAYBOOK]]

---

## §4 Pipeline (minimum-viable workflow)

### 启动新实验 — 1 命令

```bash
make launch BASELINE=B0 SITE=reddit MODE=phantom_text       # auto cell + precheck + nohup
make launch BASELINE=B1 SITE=classifieds MODE=som DRY=1     # 干跑预览
make launch BASELINE=B0 SITE=shopping MODE=phantom_prompt RESET=0   # rerun without reset
```

`MODE`: `dom` / `som` / `vision` / `phantom_text` / `phantom_som` / `phantom_prompt`. Wrapper 自动 create cell note + GLM precheck + nohup launch + cron 接管。

### 新数据 / 重跑分析 — `make analysis`

cell 跑完或 batch 新数据落盘后, 跑全 pipeline 重生 figures + cross-condition CSV + status:

```bash
make analysis                # 全 pipeline (~5-10 min) — validate + per-run + cross-condition + figures
make analysis FAST=1         # 跳过 per-run (~30s) — 只 aggregator + figures regen
make analysis RUN=results/visualwebarena/phase1/<RUN_ID>   # 单 run + downstream
make figures                 # 仅 figures regen (~10s)
```

输出: `results/phantom_paper/figures/*.png` + `results/phantom_paper/auroc_cross_condition.{csv,md}` + `phantom_lift.md`. `make analysis` 完后**手动**进 brain decision 层: 笔记 chronicle / paper_planning §3 findings / paper_drafts prose.

### 新 issue / 进展 / finding — 单文件改动

| 触发 | 改文件数 | 改哪 |
|---|---|---|
| 新 issue | **1** | `_status/issues/issue_*.md` frontmatter |
| Cell 完成 / 进度 / re-run | **0** | cron 全自动 (frontmatter + `history` + `last_run_id` + status flips) |
| 新 finding | **1** | 笔记 append §X chronicle (含 #finding tag) |
| 跨 X pattern | +1 | paper_planning §3 findings |
| Framework decision | +1 | paper_planning §19 decision log + `_status/issues/issue_advisor_sync_2026-05-14.md` (ADVISOR_SYNC.md retired 2026-05-15) |

### 查 live state

| 时间粒度 | 看哪 |
|---|---|
| Right-now (秒级) | `make active` CLI |
| ~5min snapshot (errors) | PLAYBOOK §2.5 (🤖 cron `*/5min` log scan) |
| ~10min snapshot (cells) | `cells.base` (Obsidian Bases view, cron `*/10min` 同步 frontmatter) |
| Today's narrative + 瓶颈 | PLAYBOOK §1 (🤖 GLM `0 */2 *`, narrative briefing) |
| Cron health + cell changelog | PLAYBOOK §2 (🤖 GLM `15,45 * * * *` fast §2 + `0 */2` full) |
| Next 3 actions | next_steps §0 |

### 4-zoom 写 prose 时 (paper §2/§5/§7)

问 2 个问题:
- **Evidence 哪格?** 4×4 grid (Outcome / Macro / Micro / Efficiency × cross-task / mode / site / model)
- **Explanation 哪 zoom?** Zoom 1 architectural / Zoom 2 M1/M2 activation / Zoom 3 named phenomena / Zoom 4 model-internal

写 prose 时 explicit link evidence ↔ explanation. ⚠️ 不要 evidence-as-explanation。

---

## §5 我手动维护清单

### Bases 数据层 (frontmatter, 单源化)

| File | 何时改 | Auto? | Sync transport |
|---|---|---|---|
| `_status/section*.md` | section status / words / blocker 变 | manual | git |
| `_status/cells/cell_*.md` | semantic 字段 (blocker/eta/target_section/priority/sr_adj/drop_one) **edit on DGX via VSCode Remote-SSH** | **🤖 cron `*/10min`**: status / progress / sr_raw / last_run_id / pid / history / finalized_at — re-run 自动 detect + **liveness check** (PID dead → clear stale + flip done) | **Tailscale scp** (gitignored, Windows pull-only viewer) |
| `_status/codex/codex_*.md` | codex lifecycle (ready→running→done, done 后删 file) | manual | git |
| `_status/issues/issue_*.md` | issue status (active→backlog/resolved) | manual | git |

### docs 自维护

| Doc | 何时改 |
|---|---|
| `next_steps.md §0` | hook 变 / next 3 actions 改 |
| `paper_planning.md §3 findings` | 新 cross-X pattern |
| `paper_planning.md §19 decision log` | framing 落地 |
| `_status/issues/issue_advisor_sync_<date>.md` frontmatter + paper_planning §19 | advisor 反馈后 (open → discussed → decided); ADVISOR_SYNC.md retired 2026-05-15 |
| `paper_section2_framework.canvas` | framework 改 |
| `实验笔记.md` | **append-only**, 不改过去 § |
| `PLAYBOOK.md §1 + §2` | 🤖 GLM 2h full + 30min §2-only 重写 (自己改 §3-§10 OK, §1+§2 下次 refresh 覆盖) |

### 跨 session 同步 (DGX → Windows)

**Live state (Tailscale scp, ~1min latency, 不走 git)**:
- `PLAYBOOK.md` (gitignored)
- `_status/cells/*.md` (gitignored, 20 cells × 6 sites × 3 models scope)
- `results/phantom_paper/{auroc_cross_condition,phantom_lift}.md` (本来就 gitignored)
- transport: Windows Task Scheduler `Pull PLAYBOOK from DGX` `*/1min` 跑 `pull_playbook.ps1` → scp via Tailscale (S4U logon, 无 console flicker)

**Source of truth (git, ~10min latency)**:
- paper drafts / 实验笔记 / paper_planning / next_steps / `_status/issues/issue_advisor_sync_*.md` (ADVISOR_SYNC.md retired 2026-05-15)
- code (p79/, scripts/, configs/)
- `_status/issues/_status/codex/_status/section/*.md` (manual edit, low-freq)
- transport: DGX commit+push → Windows Obsidian Git plugin auto-pull `*/10min`

**立即 sync** (跨任一 transport):
- Windows: `Ctrl+P` → "Obsidian Git: Pull" (forces git pull only — rsync 等 next 1min)
- 强制 rsync: PowerShell `Start-ScheduledTask -TaskName "Pull PLAYBOOK from DGX"`

---

## §6 自动 / 不需我维护

### 真实时 / 自动 trigger

| 数据 | 来源 |
|---|---|
| Active processes | `make active` (实时扫 ps + episode mtime) |
| 4 Bases views | `_status/*.md` frontmatter (Obsidian 自动重算) |
| Figures + cross-condition CSV | `make analysis` (新数据后手动 + post-hook 自动 fire PLAYBOOK refresh) |
| Watchdog auto-clean | watchdog daemon + runner-side resume (6-layer cross-component pipeline, B-766 post-A1.15 cold-start; detect/alert/refresh/cleanup/verify in `experiment_watchdog.py` + resume in `runner/main.py:762`. Edge cases: §4.X.15) |
| Active errors / warnings | `error_scan.py` cron `*/5min` → `logs/cron/error_scan.json` → PLAYBOOK §2.5 |
| Windows live view | Task Scheduler `*/1min` scp pull (PLAYBOOK + cells + paper aggregates) |

### Cron Sidecar (✅ ACTIVE 2026-05-02, 5 jobs)

| Job | Cadence | 用途 |
|---|---|---|
| **error-scan** | `*/5 min` | 扫 logs/{B*,queue*,watchdog*}.log + logs/cron/*.log 抓 Traceback / OOM / NOT_LOGGED_IN / Timeout / HTTP 5xx → JSON 给 §2.5 |
| **glm-update-cells** | `*/10 min` | cell frontmatter sync + re-run detection (last_run_id 变 → archive history) + **liveness check** (dead PID 自动 clear) + `cell_changelog.jsonl` 追加 |
| **glm-refresh-playbook-s2** | `15,45 * * * *` (30min) | fast §2 refresh — cron health / changelog / dead-links / ntfy / **§2.5 active errors**. SECTION=2 跳过 `make active` subprocess |
| **glm-refresh-playbook** (full) | `0 */2 * * *` (2h) | full §1 + §2 dual-section GLM call — narrative critical path 早报 + automation board |
| **check-links** | `0 0 * * 0` (weekly Sun) | 扫 docs/ broken wikilinks + path refs |
| **glm-myriad-sync** (📍 quark Task Scheduler) | `*/30 min` | quark-side GLM 5.1 SSH 到 Myriad qstat + 自动 scp 完成 cell + git commit-push 回 DGX. 替代 DGX→quark→Myriad fragile SSH chain. **未部署** — 模板见 `docs/reference/glm_quark_myriad_sync.md`, 用户手动启用 |

**GLM 统领角色**: `glm-refresh-playbook` 聚合 §1 (active processes via `make active` + cells/issues frontmatter) + §2 (cron health + changelog + ntfy + error_scan.json) → 单 GLM call 写 morning briefing 风格 §1 (3 段 + "👉 建议下一步") + 5-subsection §2。`glm-refresh-playbook-s2` 是同 script `--section 2` 模式，跳过 §1 inputs，只 refresh §2 板块。

**Post-hook auto-trigger** (新加 2026-05-02): `make launch` / `make analysis` 完成后自动 fire `glm-refresh-playbook` 在后台 — 新 run / 新数据立刻反映到 PLAYBOOK，不等 next 2h cron tick。

**Logs**: `logs/cron/glm_*.log` + `logs/cron/error_scan.json` + `logs/cron/dead_links_<date>.log` + `logs/cron/cell_changelog.jsonl`
**Ntfy topic**: `p79-exp-dgx-spark` (cron 失败自动 push priority=high)
**Manage**: `crontab -l` / `crontab -r` / `crontab scripts/maintenance/crontab.txt`

### 手动 trigger

```bash
make launch BASELINE= SITE= MODE= [RESET=1] [DRY=1]    # 一键启动新实验 (含 post-hook 自动 PLAYBOOK refresh)
make glm-update-cells [APPLY=1] [FORCE=1]              # cell sync 立即跑 (FORCE 跳过 active+pid 安全网)
make glm-refresh-playbook [APPLY=1] [SECTION=1|2|both] # PLAYBOOK §1+§2 立即 refresh, SECTION=2 fast mode
make glm-pre-launch-check QUEUE= BASELINE= SITE= [RESET=1]  # 单独 precheck (make launch 已 wrap)
make error-scan [HOURS=24]                             # 扫 errors → logs/cron/error_scan.json
make check-links                                       # dead link scan
```

---

## §7 常见命令 cheatsheet

### Git
```bash
git status -s                     # terse status
git log --oneline -10             # recent 10
git diff --stat                   # diff summary
```

### Make
```bash
make help                         # 列所有 targets
make active                       # 实时 process scan
make analysis [FAST=1]            # 全 pipeline (~5-10 min) / FAST 跳 per-run
make figures                      # 仅 fig regen
make rederive RUN=<dir>           # 修补 episode summary
make analyze RUN=<dir>            # 单 run pipeline
make compare B0=<run> B1=<run> SITE=<site>
make clean-tasks RUN=<run> COND=<cond> SITE=<site> TASKS=0-465
```

### Experiment launch
```bash
make launch BASELINE=B0 SITE=reddit MODE=phantom_text    # 1 命令 (推荐)
nohup bash scripts/queues/queue_chain.sh \
  "queue_X.sh ..." "queue_Y.sh ..." \
  > logs/chain.log 2>&1 &                                # multi-cell sequential
```

### Obsidian (Windows 端)
- `Ctrl+P` → "Obsidian Git: Pull" — 立即 git sync (PLAYBOOK + cells 走 Tailscale rsync 不在此列)
- `Ctrl+O` — fuzzy 跳 heading
- 右侧 tag pane filter `#finding` `#literature` `#bug` `#infra` `#design`
- 双击 .canvas / .base — visual / table view
- Obsidian Git plugin 配置: **Auto Pull only**, Push disabled (Windows 是 read-only consumer)
- 强制 rsync (PLAYBOOK + cells + paper aggregates): PowerShell `Start-ScheduledTask -TaskName "Pull PLAYBOOK from DGX"`

### DGX env
```bash
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
.venv/bin/python3 ...                            # NOT python
setsid nohup ... > log 2>&1 < /dev/null &        # 后台长任务
```

---

## §8 TODO 自己 (Claude 不能代做)

- [ ] 联系学长定 sync schedule
- [ ] RunPod 经费走流程 (advisor align 后)
- [ ] 联系 seonglae 协调 GPU sharing
- [x] Windows Obsidian Git plugin 装好 + 验证 auto-pull (2026-05-02 done)
- [x] Windows Tailscale rsync 配 Task Scheduler 1min pull (2026-05-02 done, S4U logon)
- [ ] paper 投稿前 freeze: regenerate paper supplement from `reference/master_bug_catalog.md`
- [ ] (advisor sync 后) 决定 14-cell rerun cell list final
- [ ] (advisor sync 后) 决定 early-stop A/B/C
- [ ] (advisor sync 后) 决定 SteerMoE scope (i)/(ii)/(iii)

---

## §9 catch-all

> 想到啥写啥, 周期性整理回 §3-§8。

-

---

## §10 Meta — PLAYBOOK 自维护

- **更新频率**: §3-§10 想到就改 (rolling); §1+§2 由 GLM cron `0 */2 * * *` (full) + `15,45 * * * *` (§2-only fast) 重写
- **Sync transport**: PLAYBOOK 是 **gitignored**, 走 Tailscale rsync DGX→Windows (Task Scheduler `*/1min`)。改 §3-§10 在 DGX 上, ~1min 内 Windows 自动看到
- **Review cadence**: 每周看一次, 整理 §9 catch-all 进 §3-§8 结构化区
- **Frontmatter `last_review`**: 每次 review 后 update
- **不放这里**: paper content (5-doc 已 cover) / live data (cells.base / make active 已 cover) / chronicle (实验笔记) / advisor (`_status/issues/issue_advisor_sync_*.md` + paper_planning §19; ADVISOR_SYNC.md retired 2026-05-15)
- **专放这里**: 自己反复需要回忆的 operating procedure / TODO 自己 / catch-all / pipeline cheatsheet
