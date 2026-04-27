# P79 Next Steps — 行动汇总

> 中央 plan 文档。汇总所有 pending experiments / analyses / paper-writing tasks。
> 实验笔记 (`docs/checkpoints/实验笔记.md`) 顶部"行动规划"区是 quick reference，
> 这个文档是详细 owner / blocker / ETA 跟踪。
> 最后更新：2026-04-27 evening

---

## 当前 status snapshot

```
Active processes (7):
  ├─ runner B0_phantom_dom_classifieds (running, ~30+ ep)
  ├─ runner B0_phantom_dom_reddit (running, ~few ep)
  ├─ watchdog B0_phantom_dom_classifieds
  ├─ watchdog B0_phantom_dom_reddit
  ├─ chain orchestrator B0 cls (dom,som)
  ├─ chain orchestrator B0 red (dom,som)
  └─ queue_b1_after_b0 sequencer

Active data (paper-grade clean):
  ├─ B0 cls + red 3-mode baseline (rederived 04-27, 数字 stable per §103)
  ├─ B1 cls + red 3-mode baseline (rederived 04-27)
  └─ B0 phantom_dom cls + red (跑中)

Cleared / pending re-run:
  ├─ B0 phantom_classifieds (待 chain re-run after dom)
  ├─ B0 phantom_reddit (同上)
  ├─ B1 phantom_reddit (待 queue_b1_after_b0 trigger)
  ├─ B1 phantom_classifieds (同上)
  └─ All shopping data (cleared, 等 Myriad)
```

ETA paper-grade clean cls + red full 4-mode matrix: **~36h from now**

---

## 1. 实验队列（优先级排序）

### 1.1 立即跑中（不要动）

| Cell | Status | ETA | Owner |
|---|---|---|---|
| B0 phantom_dom cls (chain step 1) | 跑中 | ~3-6h | runner 自动 |
| B0 phantom_dom red (chain step 1) | 跑中 | ~3-6h | runner 自动 |
| Chain B0 cls (dom→som) | sleeping | ~6-12h total | chain script 自动 |
| Chain B0 red (dom→som) | sleeping | ~6-12h total | chain script 自动 |

### 1.2 自动 follow-up（chain done 后立即触发）

| Cell | Trigger | Owner |
|---|---|---|
| B0 phantom_som cls re-run (fresh state) | chain B0 cls dom 完成后 | chain 自动 |
| B0 phantom_som red re-run | chain B0 red dom 完成后 | chain 自动 |
| B1 phantom_som cls + B1 phantom_dom cls (sequential GPU) | queue_b1_after_b0: B0 全 chain 退出 | sequencer 自动 |
| B1 phantom_som red + B1 phantom_dom red | queue_b1_after_b0: B1 cls done | sequencer 自动 |

### 1.3 等资源（手动决定时机）

| Cell | Blocker | 启动命令 | Cost |
|---|---|---|---|
| WA × 3 sites × B0+B1 × som+dom (12 cells) | 学长 align + 当前 chain done | `make phantom-wa-all && make phantom-dom-wa-all` | ~$6 + 24h GPU |
| Cross-model Claude Opus 4.7 cls+red 4-mode | 学长 align + agent 适配 | (需先 add Anthropic agent) | ~$70 |
| B0_3mode_shopping (DOM/SoM/Vision baseline) | Myriad GPU 上线 | 改写 queue_b0_with_reset 的 site filter | ~$7 + 18h API |
| B0/B1 phantom shopping (4 cells) | Myriad GPU + B0 baseline first | sequential after baseline | ~$10 + 24h |
| B1 phantom shopping (2 cells) | Myriad GPU 独占 | `make phantom B=B1 M=som S=shopping` | ~12h GPU 独占 |

---

## 2. Codex 任务队列

### 2.1 已完成

| Task | Output | Commit |
|---|---|---|
| Section 4 draft (1725 words) + 4 figure scripts | docs/analysis/paper_drafts/section4_empirical_findings.md + scripts/analysis/figures/ | bfe0154 |
| Disagreement task cluster analysis (B0 cls+red) | results/phantom_paper/analyses/disagreement_clusters.md | ded0ef6 |

### 2.2 现在能发包（数据 ready, paper-value 高）

| Task | Estimated tokens | 输出位置 | Paper value |
|---|---|---|---|
| **B1 disagreement cluster analysis** (mirror Tier 1A on B1) | ~500K | append to disagreement_clusters.md | ⭐⭐⭐ capability × representation 关键 evidence |
| **Cross-mode trajectory diff (Tier 1B)** | ~700K | results/phantom_paper/analyses/trajectory_diff.md | ⭐⭐ paper figure visualization power |
| **Codex audit category × failure pattern correlation** | ~200K | append to disagreement_clusters.md or new file | ⭐⭐ Section 5 quantitative anchor |
| **Section 1 Intro draft** | ~15K | docs/analysis/paper_drafts/section1_intro.md | ⭐ paper hook |
| **Section 2 Background + bibtex** | ~35K | docs/analysis/paper_drafts/section2_background.md + paper.bib | ⭐⭐ |
| **Section 3 Phantom-SoM Definition** | ~18K | docs/analysis/paper_drafts/section3_definition.md | ⭐ |

### 2.3 等数据后发（codex prompt 已 prep，等 trigger）

| Task | Blocker | 触发条件 |
|---|---|---|
| Phantom-SoM step traces 补 (disagreement analysis) | B0 phantom_som cls + red chain done | watchdog ntfy COMPLETE 后手动 trigger codex |
| WA disagreement analysis | WA × 3 sites × B0+B1 done | 同上 |
| Cross-model Claude disagreement | Claude Opus 4.7 run done | 同上 |
| Section 5 prose update with phantom traces | Phantom traces analyzed | 同上 |
| Section 6 Cross-Site/Model Generalization draft | WA + Claude data done | 同上 |
| Section 7 Discussion + Future Work | 全部 data + Section 1-6 done | 最后 |

---

## 3. 决策待 align（学长讨论）

| 决策 | Options | 推荐 | 影响 |
|---|---|---|---|
| Paper split | (a) Phantom-only paper + Routing follow-up, (b) Combined paper | **(a)** | 决定 Section 6 内容 |
| Cross-model 范围 | (a) 0 model, (b) 1 model (Opus 4.7), (c) 2 model (Opus + Nova) | **(b) Opus 4.7 同价 + Most Capable** | 决定 Section 6 / 是否 Cross-model 章节 |
| 跨 benchmark | (a) VWA+WA 够, (b) 加 Mind2Web | **(a) 够** | reviewer 接受度 |
| Routing experiment 时机 | (a) 这一篇就做, (b) follow-up paper | **(b) defer** | Paper structure |
| Shopping Myriad 时机 | 等 Myriad 上线（不可控） | wait | 时间线 |

---

## 4. Paper 1 (Phantom-only) Section 状态

| Section | Status | Source / Owner |
|---|---|---|
| 1. Intro | 未写 | codex prep prompt ready |
| 2. Background + Related Work | 未写 (deep research doc ready) | codex |
| 3. Phantom-SoM Definition + Ablation Setup | 未写 | codex |
| **4. Empirical Findings** | **✅ Codex draft 1725w (cls+red B0)** | codex (bfe0154) |
| 5. Mechanism (Two-Knob) | 部分 evidence ready (disagreement cluster) | codex 数据 ready, prose 待写 |
| 6. Cross-Site/Model Generalization | 等数据 (WA + Claude) | TBD |
| 7. Discussion | 未写 | codex (paper end-stage) |

Section 4-5 数据 + figures 是 paper 主体核心，~~已 ready~~ 待 chain done 后 phantom traces 补全后 100% 完整。

---

## 5. Future paper 2 (Routing) — defer

记录 routing-relevant 已积累的 infra (供未来 paper 2 用):

- Routing signals: 4 baselines × confidence_summary.json (`overall_usable: True`)
  - Behavioral signals: action_diversity / max_repeat_streak / url_revisit (AUROC 0.62-0.77)
  - Verbalized signals: ep_mean_verbalized (0.69-0.77)
  - Cross-mode AUROC computed
- Router infrastructure: `p79/experiment/router.py` RuleBasedRouter scaffold ready
- Pending: routing rule design, threshold tuning, Phase 2 condition variants

---

## 6. Daily routine（routine ops 我可以 monitor）

- ✅ ntfy notifications: condition complete + idle alerts (working)
- ✅ Auto rederive after condition complete (watchdog)
- ✅ Auto figures regen after condition complete (watchdog hook)
- ✅ Auto unified gallery refresh after condition (watchdog)
- 🔔 Persistent error / session lost → high priority ntfy

需要人工 attention 的 trigger:
- ntfy "PERSISTENT ERROR" → 调试 + 决定如何 recover
- ntfy "session lost streak" → 检查 site auth
- ntfy "P79 IDLE" 30+ min → 检查 runner alive

---

## 7. Open issues / 跟踪中的 bug

| Issue | 状态 | Blocker |
|---|---|---|
| B0 phantom_classifieds 04-26 跑过 234 ep on 04-24 reset state, cleared for paper-grade | 待 chain re-run | none |
| Phantom-SoM step traces unavailable for cleared runs | 待 chain re-run done | ~30h |
| WA tasks 5 reddit 含"image" intent keyword (5/106) | 不 codex audit (verified 100% non-visual benchmark) | n/a |
| Magento auth bug (cookie domain split) | ✅ 已 fix (commit 7150db8) | quark side base_url 改 IP |

---

## 优先级总结（如果只能做一件事）

**Now (cls+red chain 跑中, 不打扰)**:
- 发包给 codex: B1 disagreement analysis (Tier 1A-B1)
- 不动 active runs
- 学长讨论 cross-model + paper split + WA 启动时机

**+6h**:
- B0 phantom_dom cls + red 完成 → chain auto-trigger som re-run
- 给 codex: Phantom traces 补 (after som re-run done)

**+12h**:
- B0 phantom_som re-run done → chain done → queue_b1_after_b0 trigger
- 给 codex: Section 1-3 paper drafts

**+24h**:
- B1 cls chain done
- 给 codex: cross-baseline disagreement compare

**+36h**:
- B1 red chain done → cls+red full matrix paper-grade clean
- 启 WA × 3 sites × B0+B1 chains
- 启 Cross-model Claude Opus 4.7 (parallel)

**+3 days**:
- WA done → Section 6 generalization draft
- Cross-model done → Section 6 model dimension

**+5 days (Myriad 上线)**:
- B0 + B1 shopping baseline + phantom 全部跑
- Final 24-cell matrix complete

**Paper writing**: ~1 week after data complete → submit
