---
type: playbook
status: rolling
last_review: 2026-05-02
audience: self-only
---

# PLAYBOOK — 自己用的 working manual

> **给自己看的备忘录**, 不是给 Claude 不是给学长。Self-talk 语气 OK. 想到啥更新啥。
> 跟 paper-content 5-doc 不同 — 这是 **operating procedures** (how-to-do-things), 不是 strategy / live state / chronicle / prose。

---

## §1 当前 critical path snapshot (🤖 GLM auto-refresh @daily 08:00 BST + manual scratchpad)

> 自己 scratchpad. 用 ✅/⏳/🚫/🔴 标. 改这里, 不改 next_steps.
> *Last GLM refresh: 2026-05-02 10:51*

✅ B0 reddit P-prompt 运行中，progress 8%，eta ~6h
⏳ B1 classifieds P-SoM progress 79%，GPU contention（seonglae 占 95%，peak 仅 4 ep/h），eta ~10-15d
🚫 B0 classifieds P-prompt [pending] — blocked by B1 phantom_som cls done（same-site XOR 硬规则）
🚫 B1 classifieds P-text [queued] — blocked by B1 phantom_som cls done（queue chain Tier 1）
🔴 issue_b1_gpu_contention (high) + issue_paper_grade_rerun_5cells (high) 待处理

今日瓶颈: B1 classifieds P-SoM 受 seonglae GPU 95% 占用制约，throughput 骤降，ETA 拉长至 10-15d；需尽快协调 GPU sharing 或接受慢速推进，同时 5-cell paper grade rerun 排队等待一次性 launch。

---

## §2 自动化运行状态 (🤖 GLM-orchestrated daily summary)

> **GLM 统领**: daily 08:00 BST 这一节连同 §1 一起被 glm_playbook_refresh 重写。
> 数据源: `logs/cron/cell_changelog.jsonl` + `logs/cron/dead_links_*.log` + ntfy fail history + 各 cron job 最后 exit status。
> 自己想立即 refresh: `make glm-refresh-playbook APPLY=1`
> *Last GLM refresh: 2026-05-02 10:51*

### 2.1 Cron job 健康度 (last 24h)

| Job | 上次 run | 状态 | 备注 |
|---|---|---|---|
| glm-update-cells | 05-02 09:50 UTC | ✅ ok | |
| glm-refresh-playbook | 05-02 07:00 UTC | ✅ ok | |
| check-links | — | — | 从未执行 |

### 2.2 Cell 状态变更近况 (changelog tail)

- `09:44` cell_b0_red_vision: last_run_id→B0_3mode_reddit_20260422
- `09:44` cell_b0_red_som: last_run_id→B0_phantom_som_reddit_20
- `09:44` cell_b0_red_psom: last_run_id→B0_phantom_som_reddit_20
- `09:44` cell_b0_red_dom: last_run_id→B0_phantom_text_reddit_2
- `09:44` cell_b0_cls_vision: last_run_id→B0_3mode_classifieds_202
- `09:44` cell_b0_cls_som: last_run_id→B0_3mode_classifieds_202
- `09:44` cell_b0_cls_psom: last_run_id→B0_phantom_som_classifie
- `09:44` cell_b0_cls_dom: last_run_id→B0_3mode_classifieds_202

### 2.3 Dead link warnings

⚠️ `check-links` 尚未执行过，暂无扫描结果。建议尽快触发首次 dead link 检测。

### 2.4 Ntfy fail alerts 历史

- 2026-05-02 01:17 UTC — ⚠️ P79 cron fail: test-fail
- 2026-05-02 01:42 UTC — ⚠️ P79 cron fail: fail-test-claude
- 2026-05-02 01:42 UTC — P79 fail-test from claude

---

## §3 Session bootstrap (开新对话怎么快速给 Claude 上下文)

### Quick (30s) — 简单问题
让 Claude 读: `docs/checkpoints/next_steps.md` (lean 215 行, current state 一览)

### Standard (1-2 min) — 日常工作
让 Claude 顺序读:
1. `docs/checkpoints/next_steps.md` (current state + blockers + next 3 actions)
2. `docs/checkpoints/实验笔记.md` 最后 N 个 § (最新 §108.X 系列, 看 5-7 个就够)
3. `docs/checkpoints/paper_planning.md` §1 (hook) + §2 (theory framework Zoom 1-4) + §20 (doc workflow)

### Full reload (5 min) — paper writing / advisor sync 前
加上:
4. `paper_planning.md` §3 (findings) + §6 (risks) + §19 (decision log)
5. `ADVISOR_SYNC.md` §0-§2 (current advisor state + 5 open framing decisions)

### Obsidian-side 自己 quick scan (新 session 也建议)
- `cells.base` 切 "Active 跑中" view — 看现在跑啥
- `issues.base` 切 "Active blockers" view — 看 blocker
- `paper_section2_framework.canvas` — 复习 §2 framework dual layer

### 一句话给 Claude prompt 模板
> "读 next_steps.md, 实验笔记最后 §108.10+ 全部, paper_planning §1+§2+§20, 然后告诉我现在 critical path 是什么"

---

## §4 新数据到了之后的标准流程

### Step 1: Validate + Freeze + 看图
```bash
make validate RUN=<run_dir>    # 验证 single run 完整性 (episode count / summary / artifacts)
make analysis                  # 全 pipeline ~5-10 min (validate + per-run + cross-condition + figures)
make analysis FAST=1           # 跳过 per-run, 只 aggregator + figures (~30s)
```
**先 validate 再 analysis** — 防 partial / 污染数据进入 cross-condition aggregation.
看 `results/phantom_paper/figures/` 重生 PNGs.
看 `results/phantom_paper/auroc_cross_condition.{csv,md}` 等 aggregations.

**(基本不需要手动)** Cron 每 10 min 自动 GLM auto-update cell frontmatter + re-run detection. 如想立即 sync:
```bash
make glm-update-cells              # dry-run 看 diff
make glm-update-cells APPLY=1      # 实际写
```

### Step 2: 整合到 docs (按依赖顺序)
1. **实验笔记** append §X chronicle (date + finding + evidence + tag `#finding/#bug/#infra/etc.`)
2. **`_status/cells/cell_*.md`** frontmatter — `status` / `progress` / `sr_raw` / `last_run_id` / `pid` / `history` 由 cron 自动维护; `sr_adj` / `drop_one` / `blocker` / `eta` 仍人工
3. **paper_planning §3 findings** 加新 finding (如 cross-site / cross-capability pattern 出现)
4. **next_steps §0 current state** 如 paper hook 变 / next 3 actions 改 / blocker 变
5. **paper_planning §19 decision log** 如有重大 framing decision
6. **ADVISOR_SYNC §1 snapshot** 如改 advisor view (sync prep 阶段)

### Step 3: 4-zoom 双层 integrate (paper §2/§5 prose 准备)
问自己 2 个问题:
- **Evidence 哪格?** 4×4 grid (Outcome / Macro / Micro / Efficiency × cross-task / mode / site / model = 16 cells)
- **Explanation 哪 zoom?**
  - Zoom 1 architectural (deductive)
  - Zoom 2 M1/M2 behavioral activation 2×2
  - Zoom 2.5 别扭 (provisional, lean §7 narrative use 不进 §2 main)
  - Zoom 3 named phenomena (lit-anchored, M1/M2/M3 三 axis)
  - Zoom 4 model-internal (3-anchor 三角 future work)

写 paper §2/§5 prose 时 **explicit link** evidence ↔ explanation:
> "the [Macro × mode 35.7% search-loop 数据] suggests [Zoom 2 M2 flat-list activation], consistent with [Zoom 3 Sclar prompt-format sensitivity]"

⚠️ **不要 evidence-as-explanation** (reviewer 最忌)

### Step 4: 触发 codex prose update (按需)
- Section 4 fresh-data: codex #11 (~30K, ready)
- Section 5 mechanism: codex #13 (~50K, 待 #11 一起)
- Section 6 routing: codex #16 (~50K, after Tier 1+2 prototype)

`_status/codex/codex_*.md` frontmatter `status: ready → running → done` 顺手 update

---

## §5 我手动维护清单

### Bases 数据层 (frontmatter, 单源化)
| File | 何时改 | 改啥 | Auto? |
|---|---|---|---|
| `_status/section*.md` | section status / words / blocker 变 | `status` `progress` `blocker` `words` | manual |
| `_status/cells/cell_*.md` | cell 完成 / 状态变 / re-run 发起 | `status` `progress` `sr_raw` `n` `last_run_id` `pid` `history` `finalized_at` (auto via cron); `sr_adj` `drop_one` `blocker` `eta` `target_section` `priority` (manual) | **🤖 cron @10min** (skips active+pid; auto-detects re-run via `last_run_id` change → flips status done→active + archives prior sr_raw to `history`) |
| `_status/codex/codex_*.md` | codex task lifecycle | `status` (ready→running→done, done 后**删 file**) | manual |
| `_status/issues/issue_*.md` | issue 状态变 | `status` (active→backlog/resolved). resolved 后删 file 或留作 chronicle | manual |

### docs 自维护
| Doc | 何时改 |
|---|---|
| `next_steps.md §0` | paper hook 变 / next 3 actions 改 / blocker 变 |
| `paper_planning.md §3 findings` | 新 cross-site / cross-capability pattern 出现 |
| `paper_planning.md §19 decision log` | 重大 framing 落地 (timestamp + decision + rationale + status) |
| `ADVISOR_SYNC.md §2 framing decisions` | advisor 反馈后 status (open → discussed → decided) |
| `ADVISOR_SYNC.md §4 sync history` | meeting 完后填 actual notes (placeholder section 4.2 等) |
| `paper_section2_framework.canvas` | framework 改 (重大 retract / 新 zoom / 新 anchor) |
| `实验笔记.md` | **append-only**, 不改过去 § |
| `PLAYBOOK.md §1 + §2` | 🤖 GLM @daily 重写 — 自己改 §1 scratchpad 也 OK (下次 GLM run 会再覆写) |

### 跨 session 同步 (DGX vs Windows Obsidian)
- DGX 改 → commit + push
- Windows Obsidian Git plugin auto-pull (10 min interval)
- 想立即 sync → Windows `Ctrl+P` → "Obsidian Git: Pull"

---

## §6 自动 / 不需我维护

### 真实时 / 自动 trigger
| 数据 | 来源 | 触发 |
|---|---|---|
| Active processes | `make active` | 实时扫 ps + episode mtime |
| 4 Bases views | `_status/*.md` frontmatter | Obsidian Bases 自动重算 |
| Figures + cross-condition CSV | `make analysis` | 手动 (新数据后) |
| Watchdog auto-clean | 6-layer protocol | runner-side daemon |
| obs_prepare cost / energy | episode_summary_v2.json | runner 写入 (§69) |
| paper.bib BibTeX | codex #10 (Gemini DR 6/6 done) | 手动 trigger codex |

### 🤖 GLM Sidecar Cron (✅ ACTIVE 2026-05-02)

Installed via `crontab scripts/maintenance/crontab.txt`. **失败自动 ntfy 通知** (priority high, last 500 chars output)。

| Job | Cadence | 用途 | Failure handling |
|---|---|---|---|
| **glm-update-cells** | `*/10 min` | cell frontmatter sync + re-run detection + changelog jsonl | ⚠️ ntfy on exit≠0 — skips active+pid cells (safety) |
| **glm-refresh-playbook** | `@daily 08:00 BST` | 重写 PLAYBOOK §1 (critical path) + §2 (automation status) | ⚠️ ntfy on exit≠0 |
| **check-links** | `@weekly Sun 00:00` | 扫 docs/ 全部 broken wikilinks + path refs | log only (always exit 1 if found) |

**GLM 统领角色**: `glm-refresh-playbook` 每日聚合三个 cron 的产出 (cell changelog / dead links / fail alerts) + `make active` + `_status/` 全部 frontmatter → 一次 GLM 5.1 call 同时写 PLAYBOOK §1 critical path + §2 automation status。其他两个 cron 都是 deterministic infrastructure 给 GLM 喂数据。

**Ntfy topic**: `p79-exp-dgx-spark` (override via `NTFY_TOPIC` env)
**Logs**: `logs/cron/glm_*.log` + `logs/cron/dead_links_<date>.log` + `logs/cron/cell_changelog.jsonl`
**Manage**: `crontab -l` 查 / `crontab -r` 卸 / `crontab scripts/maintenance/crontab.txt` 重装

### 手动 trigger (cron 之外按需)
```bash
make glm-update-cells [APPLY=1] [FORCE=1]              # cells frontmatter sync
make glm-refresh-playbook [APPLY=1]                    # PLAYBOOK §1+§2 refresh
make glm-pre-launch-check QUEUE= BASELINE= SITE= [RESET=1]  # pre-launch sanity
make check-links                                       # dead link scan
```

---

## §7 不要做的事 (self-reminder)

### Doc separation
- ❌ next_steps 复制 paper_planning 内容 (用 wikilink `[[paper_planning#§5]]`, 不 copy)
- ❌ next_steps 写过去 chronicle (归笔记)
- ❌ next_steps 写 advisor 内容 (归 ADVISOR_SYNC)
- ❌ paper_drafts 加 Obsidian wikilinks/callouts/frontmatter (保 plain markdown 兼容 codex/pandoc/LaTeX 转换)
- ❌ 硬编码 active processes 到 markdown (走 `make active` 实时)

### Git
- ❌ git push 不告诉 Claude (Claude 规则: commit autonomous OK, **push 必须 explicit ask**)
- ❌ amend / force-push 主分支
- ❌ commit `.env` `.auth/` `vwa_env_remote.sh` (gitignored, 但要警惕)

### 实验
- ❌ 同 site 同时跑 B0 + B1 (cross-contam — account / cart / session race, 实证 04-26 reddit 已发生)
- ❌ 不用 `RESET_BEFORE=1` 跑 paper-grade (state pollute, 实证 04-28 dirty_no_reset)
- ❌ 用 `python` 命令 (DGX 用 `.venv/bin/python3`)
- ❌ 裸跑 `python scripts/run_experiment.py` (必须走 queue script — race-safe + reset + watchdog)

### Memory / Claude
- ❌ 让 Claude 从 memory 取 SR / FP 数字 (memory 仅静态事实, 数字从 docs/analysis 实时拉)
- ❌ 让 Claude 在 paper drafts 加 Obsidian syntax (上面 doc separation 重申)

### 学长沟通
- ❌ 主线说 "别扭 framework" (provisional, cognitive overload risk)
- ❌ 先讲 VWA bugs (会 anchor anxiety, 后讲 Phantom finding 主线再披露)

---

## §8 常见命令 cheatsheet

### Git
```bash
git status -s              # terse status
git log --oneline -10      # recent 10
git diff --stat            # diff summary
git diff --cached --stat   # staged diff
```

### Make
```bash
make help                  # 列所有 targets
make active                # 实时 process scan (取代硬编码)
make analysis              # 全 pipeline (~5-10 min)
make analysis FAST=1       # 跳过 per-run, 只 aggregator + figures (~30s)
make figures               # 仅 fig regen (~10s)
make rederive RUN=<dir>    # 修补 episode summary
make analyze RUN=<dir>     # 单 run per-run pipeline
make compare B0=<run> B1=<run> SITE=<site>  # B0 vs B1
make clean-tasks RUN=<run> COND=<cond> SITE=<site> TASKS=0-465
```

### Experiment launch (一定走 queue script)
```bash
RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh B0 dom shopping
RESET_BEFORE=1 bash scripts/queues/queue_phantom_som.sh B0 reddit
RESET_BEFORE=1 bash scripts/queues/queue_phantom_text.sh B0 reddit
RESET_BEFORE=1 bash scripts/queues/queue_phantom_prompt.sh B0 reddit
nohup bash scripts/queues/queue_chain.sh "queue_X.sh ..." "queue_Y.sh ..." > logs/chain.log 2>&1 &
```

### Obsidian (Windows 端)
- `Ctrl+P` → "Obsidian Git: Pull" — 立即 sync
- `Ctrl+O` — fuzzy 跳 heading (取代 grep)
- 右侧 tag pane filter `#finding` `#literature` `#bug` `#infra` `#design`
- 双击 .canvas → visual canvas
- 双击 .base → table/cards view selector

### DGX env
```bash
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
.venv/bin/python3 ...                        # NOT python
setsid nohup ... > log 2>&1 < /dev/null &    # 后台长任务
```

---

## §9 TODO 自己 (Claude 不能代做)

- [ ] 联系学长定 sync schedule (人际事)
- [ ] RunPod 经费走流程 (advisor align 后)
- [ ] 联系 seonglae 协调 GPU sharing (or 接受 slow progression)
- [ ] Windows Obsidian Git plugin 装好 + 验证 auto pull works
- [ ] paper 投稿前 freeze: regenerate paper supplement from `reference/master_bug_catalog.md`
- [ ] (advisor sync 后) 决定 14-cell rerun cell list final
- [ ] (advisor sync 后) 决定 early-stop A/B/C
- [ ] (advisor sync 后) 决定 SteerMoE scope (i)/(ii)/(iii)

---

## §10 还想不起来的事 (catch-all, 想到啥写啥)

> 这块留空, 日常想到 doc separation / 流程 / decision 不在前面 §1-§9 cover 的, 写这里. 周期性整理回 §1-§9.

-

---

## §11 Meta — PLAYBOOK 自维护

- **更新频率**: 想到就改 (rolling); §1+§2 由 GLM cron @daily 重写
- **Review cadence**: 每周看一次, 整理 §10 catch-all 进 §3-§9 结构化区
- **Frontmatter `last_review`**: 每次 review 后 update
- **不放这里**: paper content (那 5-doc 已 cover) / live data (那 _status/ + Bases 已 cover) / chronicle (那 实验笔记 已 cover) / advisor (那 ADVISOR_SYNC 已 cover)
- **专放这里**: 自己反复需要回忆的 operating procedure / not-to-do reminder / TODO 自己 / catch-all 思路
- **GLM-managed sections** (§1 + §2): 自己也可以手改 (next refresh 会覆写). §1 是 today's bottleneck scratch, §2 是自动化 health board.
