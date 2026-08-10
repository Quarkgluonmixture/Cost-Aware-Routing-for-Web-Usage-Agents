---
type: task
status: active
priority: P0
horizon: now
order: 1
blocker: ""
eta: "**2026-09-01 硬截止** — 全稿交付。剩 ~22 天。**图已不是瓶颈: 主文 13 张 2026-08-10 落地 (六张不可割全齐), prose 是唯一 critical path**。REALM 稿 (Submission #192, 8 页正文, 审查中) 可作为结果章骨架复用; 数据与分析已全部落地, 这是纯写作 critical path"
detail: final_dissertation/
created: 2026-06-10
updated: 2026-08-08
---

# Thesis full draft → 2026-09-01 硬截止

**落点 = `final_dissertation/`** (2026-08-08 user 指定; `final_dissertation/prior/` 放学长给的
往届优秀作品作参照, 不必逐字读)。REALM 稿是结果章骨架来源, 但**毕设不是 REALM 稿的扩写** ——
毕设要 problem-first + concept-first + 文献图谱 + benchmark EDA, 这些 8 页会议稿里没有。

**当前**: `final_dissertation/` 除 `prior/` 外为空, 全稿待写。数据与分析已全部落地
(cls+red Phase 1a 42 conditions / WA pilot / mechanism archive), **不再需要任何新实验**。

原链 (已被 09-01 硬截止取代): results+discussion 07-24 (D8) → 全稿 v1 08-10 (D9)
→ 修订完 08-24 (D10) → submission early-Sep (D11)。

## 学长毕设写作 rubric 13 条 (2026-07-24, 详 issue_advisor_sync_2026-07-24)

写/审每章时逐条对照。⭐ = 结构级, 🔲 = 待做离线工作项, ✅ = 基本满足待声明。

> ⭐ **2026-08-10: 五个起手文件已齐**（Guide §26）。`FIGURE_PLAN.md` 落地 = **17 主文图 + 9 appendix
> + 6 条图注雷区 + 四检查登记表**，Stage A（图↔章问题↔claim↔数据源绑定）已定，Stage B（页数预算）
> 待 handbook（**user 已走 AskUCL**；`GPT_SEARCH_PROMPTS.md` P1 并行加速）。
> 决策已锁：**disc+concl 合并为 Ch7** · shop/WA 只作 external validation 落地才进 · **A2 pass@K 现在做**。
> ⚠️ 顺带核出三处过时口径（C4 `0/6→0/8` · `B=200→B=10000` · WA learnability 缺口其实 08-03 已解决）
> 与一个未修 bug（`router_triage_learnability.py` 散文段硬编码 6 格，m 未随 8 格更新）→ 笔记 §450。

- [ ] 1. ⭐ **problem-first** — 开局讲 problem 不讲背景铺垫 (intro 重排)
- [x] 2. ✅ **system structure 图** = **F3 已画** (2026-08-10) — 端到端架构图。**规格已定 = FIGURE_PLAN F3**：不是装饰性架构图，
       画 *causal comparison boundary*（`CHAPTER_CHAIN.md:237` 原文链条）。零数据依赖，可立即开工
- [ ] 3. ✅ **public-access 声明** — VWA/Qwen/Gemma/MiMo 均 public + OSF; ⚠️ B0=proxy 需诚实标"非公开可复现"
- [ ] 4. ⭐ **concept-first intro** — 从 concept 建立, 不堆 statement, 不太具体
- [ ] 5. ⭐ **disc+concl 合并** (limitation 内嵌) — **今日 D8 章结构直接受此影响**
- [ ] 6. **appendix 非 orphan** — 每个 appendix 正文有 "见 Appendix X" 指针 (S1-S6 待补入口)
- [ ] 7. 🔲 **Zekun Wu 原文** — 拉共同作者论文 → bib + 文献图谱 (arXiv API 核验, 非 WebSearch)
- [x] 8. ✅ **文献图谱** = **F2 已画** (2026-08-10, 27 篇全核) — 可视 literature map。**规格已定 = FIGURE_PLAN F2**（四簇: 观测表征 /
       模型-模态 routing 与 cascade / confidence deferral / cost-aware inference）。**素材依赖 GPT P2**，
       每篇必须带"它优化的是什么"这一维，否则图谱显不出缺口
- [x] 9. ✅ **visualization** — **2026-08-10 规划完成**: `FIGURE_PLAN.md` 17 主文 + 9 appendix，
       dashboard style，每张过 Guide §7.2 四检查。⚠️ 规划≠画完，实现分派见 FIGURE_PLAN §6
       （优先级 F0 → F3 → F14 → F13 → F10b → F1）
- [x] 10. ✅ **benchmark EDA** — **2026-08-09 done**: `scripts/analysis/benchmark_eda.py` →
       `docs/analysis/benchmark_eda/corpus_eda.{md,json}` (六语料 × 分布/难度/eval/长度/模板)。
       三个副产品进了笔记 §445: **run set vs scored set 双口径**(解释 205↔203 / 435↔432 并存)、
       benchmark 自带 2 个错别字、**WA 零 ref image = OOD 的结构性理由**。
       ⚠️ 图还没配，但**规格已定 = FIGURE_PLAN F4**（数据已在 `corpus_eda.json`，需画的三个副产品已列）；正文引用待写
- [ ] 11. 🔲 **每章有开头+结尾** — 导语 + 小结/过渡 (逐章补, 机械必做)
- [ ] 12. 🔲 **公式** — obs 构造/drop-one/estimand/router 规则形式化
- [ ] 13. **每个 setting justify** — 4B/三站/step_002/24 task/50 tokens 逐个给理由 (素材在笔记)
- [ ] 14. (学长止于 13, 留白)

**并行策略**: 2/8/10/12 纯离线不占 GPU → 可与 mechanistic sweep 同时推进。
**通用性**: 1/4/6/12/13 对 REALM Paper A/B 也适用。
