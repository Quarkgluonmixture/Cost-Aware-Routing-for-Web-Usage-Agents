---
type: task
status: active
priority: P0
horizon: now
order: 1
blocker: ""
eta: "**2026-09-01 硬截止 (user 2026-08-08)** — 全稿交付。剩 ~24 天。REALM 稿 (Submission #192, 8 页正文, 审查中) 可作为结果章骨架复用; 数据与分析已全部落地, 这是纯写作 critical path"
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

- [ ] 1. ⭐ **problem-first** — 开局讲 problem 不讲背景铺垫 (intro 重排)
- [ ] 2. 🔲 **system structure 图** — 端到端架构图 (obs→backend→runner→router→eval)
- [ ] 3. ✅ **public-access 声明** — VWA/Qwen/Gemma/MiMo 均 public + OSF; ⚠️ B0=proxy 需诚实标"非公开可复现"
- [ ] 4. ⭐ **concept-first intro** — 从 concept 建立, 不堆 statement, 不太具体
- [ ] 5. ⭐ **disc+concl 合并** (limitation 内嵌) — **今日 D8 章结构直接受此影响**
- [ ] 6. **appendix 非 orphan** — 每个 appendix 正文有 "见 Appendix X" 指针 (S1-S6 待补入口)
- [ ] 7. 🔲 **Zekun Wu 原文** — 拉共同作者论文 → bib + 文献图谱 (arXiv API 核验, 非 WebSearch)
- [ ] 8. 🔲 **文献图谱** — 可视 literature map (簇: routing/观测/confidence/mechanism)
- [ ] 9. **visualization** — 系统化配图 (dashboard style: 少黑话少字多图)
- [ ] 10. 🔲 **benchmark EDA** — task 分布/难度/站点差异/长度分布 (纯离线可现做)
- [ ] 11. 🔲 **每章有开头+结尾** — 导语 + 小结/过渡 (逐章补, 机械必做)
- [ ] 12. 🔲 **公式** — obs 构造/drop-one/estimand/router 规则形式化
- [ ] 13. **每个 setting justify** — 4B/三站/step_002/24 task/50 tokens 逐个给理由 (素材在笔记)
- [ ] 14. (学长止于 13, 留白)

**并行策略**: 2/8/10/12 纯离线不占 GPU → 可与 mechanistic sweep 同时推进。
**通用性**: 1/4/6/12/13 对 REALM Paper A/B 也适用。
