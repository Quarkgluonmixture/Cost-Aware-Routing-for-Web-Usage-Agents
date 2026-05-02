---
type: playbook
status: rolling
last_review: 2026-05-02
audience: self-only
---

# PLAYBOOK

---

## §1 当前 critical path snapshot

▶️ B1 phantom_prompt classifieds 依然是当前的主战场，目前已经跑了近 1.5 天，但进度仅 27% (63/234)。吞吐量掉到了 1 ep/h，预计还要约 7 天才能跑完，主要受制于 GPU 资源争夺。底层的 queue chain 已经运行近 3 天，目前正卡在第二阶段的 `queue_phantom_prompt.sh B1 classifieds`，单这一步就耗了 33 小时。

⏳ 目前有 7 个 cell 在排队阻塞。B0 classifieds P-prompt 因为 B1 同站点互斥的硬性规则动弹不得。其余 6 个 cell（涉及 shopping 和 reddit 的 P-SoM/P-text）全部被 queue chain 的 Tier 1/2 死死卡住，必须要等前序任务彻底跑完才能释放推进。

🔴 当前的核心卡点就是 GPU 算力。Issue 列表明确指出我们需要去跟 seonglae 协调 GPU sharing，否则整体进度只能一直龟速爬行。此外，B1 shopping dom 的重跑任务还在等 RunPod 资源，而高优的 paper_grade 任务需要一次性拉起 5 个 cell，这与当前本就不富裕的算力直接冲突。

👉 建议: 立刻联系 seonglae 协调 GPU sharing 方案，或评估能否暂停部分低优任务，优先为 paper_grade 重跑腾出算力。

---

## §2 自动化运行状态

### 2.1 Cron job 健康度 (last 24h)

| Job | 上次 run | 状态 | 备注 |
|---|---|---|---|
| glm-update-cells | 2026-05-02 16:50 | ✅ | 正常 |
| glm-refresh-playbook-s2 | 2026-05-02 16:45 | ✅ | 正常 |
| glm-refresh-playbook | 2026-05-02 07:00 | ✅ | 正常 |
| check-links | (从未运行) | — | 首次运行待相应 cron 触发 |

### 2.2 Cell 状态变更近况 (changelog tail)

- 16:41 cell_b1_cls_phantom_som: pid_dead_cleared, status→done, progress→100
- 16:41 cell_b0_red_pprompt: pid_dead_cleared, status→done, progress→100
- 10:46 cell_b1_cls_phantom_text: status→done, progress→100
- 10:46 cell_b0_red_ptext: last_run_id 更新
- 10:46 cell_b0_cls_ptext: last_run_id 更新
- 09:44 cell_b0_red_vision: last_run_id 更新
- 09:44 cell_b0_red_som: last_run_id 更新
- 09:44 cell_b0_cls_vision: last_run_id 更新

### 2.3 Dead link warnings
⚠️ Dead link 扫描尚未运行过，暂无数据。

### 2.4 Ntfy fail alerts 历史
✅ 近 24h 无失败

### 2.5 🔴 Active errors / warnings (runner / watchdog log scan, last 24h)
✅ 近 24h 无 runner / watchdog 错误 (扫了 76 个 log 文件)

---

## §3 Session bootstrap

> 一句话给 Claude: 读 [[实验笔记]] 最后几个 § + [[paper_planning]] + [[next_steps]] + [[ADVISOR_SYNC]] + [[PLAYBOOK]]

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
| Framework decision | +1 | paper_planning §19 decision log + ADVISOR_SYNC §2 |

### 查 live state

| 时间粒度 | 看哪 |
|---|---|
| Right-now (秒级) | `make active` CLI |
| ~10min snapshot | `cells.base` (Obsidian Bases view) |
| Today's narrative + 瓶颈 | PLAYBOOK §1 (🤖 GLM @daily) |
| Cron health + cell changelog | PLAYBOOK §2 (🤖 GLM @daily) |
| Next 3 actions | next_steps §0 |

### 4-zoom 写 prose 时 (paper §2/§5/§7)

问 2 个问题:
- **Evidence 哪格?** 4×4 grid (Outcome / Macro / Micro / Efficiency × cross-task / mode / site / model)
- **Explanation 哪 zoom?** Zoom 1 architectural / Zoom 2 M1/M2 activation / Zoom 3 named phenomena / Zoom 4 model-internal

写 prose 时 explicit link evidence ↔ explanation. ⚠️ 不要 evidence-as-explanation。

---

## §5 我手动维护清单

### Bases 数据层 (frontmatter, 单源化)

| File | 何时改 | Auto? |
|---|---|---|
| `_status/section*.md` | section status / words / blocker 变 | manual |
| `_status/cells/cell_*.md` | (semantic 字段) blocker / eta / target_section / priority / sr_adj / drop_one | **🤖 cron @10min**: status / progress / sr_raw / n / last_run_id / pid / history / finalized_at — re-run 自动 detect (last_run_id 变 → archive 旧 sr_raw 到 history → flip done→active) |
| `_status/codex/codex_*.md` | codex lifecycle (ready→running→done, done 后删 file) | manual |
| `_status/issues/issue_*.md` | issue status (active→backlog/resolved) | manual |

### docs 自维护

| Doc | 何时改 |
|---|---|
| `next_steps.md §0` | hook 变 / next 3 actions 改 |
| `paper_planning.md §3 findings` | 新 cross-X pattern |
| `paper_planning.md §19 decision log` | framing 落地 |
| `ADVISOR_SYNC.md §2` | advisor 反馈后 (open → discussed → decided) |
| `paper_section2_framework.canvas` | framework 改 |
| `实验笔记.md` | **append-only**, 不改过去 § |
| `PLAYBOOK.md §1 + §2` | 🤖 GLM @daily 重写 (自己改也 OK, 下次 refresh 覆盖) |

### 跨 session 同步 (DGX vs Windows Obsidian)

- DGX 改 → commit + push
- Windows Obsidian Git plugin auto-pull (10 min interval)
- 立即 sync → Windows `Ctrl+P` → "Obsidian Git: Pull"

---

## §6 自动 / 不需我维护

### 真实时 / 自动 trigger

| 数据 | 来源 |
|---|---|
| Active processes | `make active` (实时扫 ps + episode mtime) |
| 4 Bases views | `_status/*.md` frontmatter (Obsidian 自动重算) |
| Figures + cross-condition CSV | `make analysis` (新数据后手动) |
| Watchdog auto-clean | runner-side daemon (6-layer protocol) |

### GLM Sidecar Cron (✅ ACTIVE 2026-05-02)

| Job | Cadence | 用途 |
|---|---|---|
| **glm-update-cells** | `*/10 min` | cell frontmatter sync + re-run detection + `cell_changelog.jsonl` |
| **glm-refresh-playbook** | `@daily 08:00 BST` | 重写 PLAYBOOK §1 (critical path) + §2 (automation status) |
| **check-links** | `@weekly Sun 00:00` | 扫 docs/ broken wikilinks + path refs |

**GLM 统领角色**: `glm-refresh-playbook` 每日聚合 (cell changelog + dead links + ntfy fails + `make active` + `_status/`) → 一次 GLM call 写 §1 critical path + §2 automation board。其他 cron 喂数据。

**Logs**: `logs/cron/glm_*.log` + `logs/cron/dead_links_<date>.log` + `logs/cron/cell_changelog.jsonl`
**Ntfy topic**: `p79-exp-dgx-spark` (failures auto-push priority high)
**Manage**: `crontab -l` / `crontab -r` / `crontab scripts/maintenance/crontab.txt`

### 手动 trigger

```bash
make launch BASELINE= SITE= MODE= [RESET=1] [DRY=1]    # 一键启动新实验
make glm-update-cells [APPLY=1] [FORCE=1]              # cell sync 立即跑
make glm-refresh-playbook [APPLY=1]                    # PLAYBOOK §1+§2 立即 refresh
make glm-pre-launch-check QUEUE= BASELINE= SITE= [RESET=1]  # 单独 precheck (make launch 已 wrap)
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
- `Ctrl+P` → "Obsidian Git: Pull" — 立即 sync
- `Ctrl+O` — fuzzy 跳 heading
- 右侧 tag pane filter `#finding` `#literature` `#bug` `#infra` `#design`
- 双击 .canvas / .base — visual / table view

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
- [ ] Windows Obsidian Git plugin 装好 + 验证 auto-pull
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

- **更新频率**: 想到就改 (rolling); §1+§2 由 GLM cron @daily 重写
- **Review cadence**: 每周看一次, 整理 §9 catch-all 进 §3-§8 结构化区
- **Frontmatter `last_review`**: 每次 review 后 update
- **不放这里**: paper content (5-doc 已 cover) / live data (cells.base / make active 已 cover) / chronicle (实验笔记) / advisor (ADVISOR_SYNC)
- **专放这里**: 自己反复需要回忆的 operating procedure / TODO 自己 / catch-all / pipeline cheatsheet
