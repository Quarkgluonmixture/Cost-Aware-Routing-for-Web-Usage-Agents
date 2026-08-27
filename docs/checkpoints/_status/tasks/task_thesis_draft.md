---
type: task
status: active
priority: P0
horizon: now
order: 1
blocker: ""
eta: "**2026-09-05 硬截止**（user 2026-08-19 从 09-01 延长）— 剩 ~9 天。**2026-08-27 大改已落地**: 稿件迁至 **UCL PhD Thesis Template**，Overleaf 评审层换新项目 **6a8f68ace4443ea9751d6201**（旧 6a7a7331d2e6523a360245d4 **已停用，别再往那推**，两边版式不同）。现 **114 页 / 15 图 / 20 表 / 36 bib**，编译 0 error / 0 undefined / 9 overfull（最大 10pt 在参考文献，其余 <2pt）。图全部重做（图内散文 105→0，由 `make thesis-figures` 的 `check_no_prose.py` 门禁强制）；正文 em dash 88→0。页数预算 ≤100 页那条 **已失效需重问** —— 模板 12pt + 40mm 装订边把 89 页推到 114 页，这是版式造成的不是内容增加。**剩余 = ① supervisor 反馈（Overleaf 已是最新 08-27）② rubric #7/#13 ③ 页数上限重新确认**。不需要任何新实验"
detail: final_dissertation/
created: 2026-06-10
updated: 2026-08-27
---

# Thesis full draft → 2026-09-05 硬截止

**落点 = `final_dissertation/`** (2026-08-08 user 指定; `final_dissertation/prior/` 放学长给的
往届优秀作品作参照, 不必逐字读)。REALM 稿是结果章骨架来源, 但**毕设不是 REALM 稿的扩写** ——
毕设要 problem-first + concept-first + 文献图谱 + benchmark EDA, 这些 8 页会议稿里没有。

**当前 (2026-08-11)**: **全稿 v1 已落地** —— `final_dissertation/tex/` 79 页, 7 章 + 4 附录 +
17 图 + 33 条核验 bib, 编译 0 undefined ref / 0 overfull。Overleaf 单向同步已接
(`scripts/maintenance/overleaf_thesis_sync.sh`, 先本地编译再推)。三家 AI submission-ready
审计已过 (笔记 §452), 18 条 finding 全修完 —— 含两条改 headline 的:
C1 成本口径 **13.7-35.3% → 1.6-35.3%** (SR 平局按列表序破平选中更贵的对照);
六次重跑 vs 六个 mode 残差仅 **3.53pp** 低于本文自己的门槛 ⇒ C1 改判为「天花板的**价格**」。

**剩余** = ① supervisor 反馈 (⚠️ 见下: Overleaf 评审层落后 10 天) ② ~~页数预算~~ **已解 2026-08-22 (≤100, 现 89)** ③ rubric #7 (Zekun Wu 原文) + #13 (每个 setting justify)
④ rubric #13 逐个 setting 补 justify。**不需要任何新实验**。

**2026-08-27 更新 — 排版层大改 (笔记 §481/§482/§483, commits `fc5f764` `a0a632b`)**:

- **模板**: 迁 UCL PhD Thesis Template。引用切 biblatex+biber (`natbib=true` 保住全部
  `\citep/\citet`)。修了模板自带两个缺陷 —— 字族未声明先用导致粗体小型大写被静默丢弃;
  `\fancypagestyle` 的 `\fancyhead[R]` 漏进 live style 使全书右上角永久挂章标题。
- **Overleaf 换项目**: **6a8f68ace4443ea9751d6201**。`overleaf_thesis_sync.sh` 已改指向,
  本地 clone 在 `~/overleaf-thesis-ucl`。⚠️ 旧项目停用, 推错导师会看到旧排版。
- **图 18 → 15 张**: 图内散文 105 → 0 (论断全部下沉 caption)。REALM 稿三张直接复用
  (overview / diamond / ceilings)。规则由 `check_no_prose.py` 读**渲染后的 PDF** 强制,
  串在 `make thesis-figures` 里。⚠️ **单独跑图脚本不安全** (只写 `figures/`,
  `tex/figures/` 仍旧图), 必须走 make。
- **页数 89 → 114**: 模板 12pt + 40mm 装订边所致, 非内容增加。⚠️ **≤100 页那条要重新问**。
- 横排页 1 处 (`fig_overview`, 2.67:1)。⚠️ 「UCL 明文允许横排」**无依据** ——
  COMP0191 handbook 仍未拿到, 现有依据只有模板自带 `pdflscape` + 通行做法。

原链 (已被 09-05 硬截止取代): results+discussion 07-24 (D8) → 全稿 v1 08-10 (D9)
→ 修订完 08-24 (D10) → submission early-Sep (D11)。

## 学长毕设写作 rubric 13 条 (2026-07-24, 详 issue_advisor_sync_2026-07-24)

写/审每章时逐条对照。⭐ = 结构级, 🔲 = 待做离线工作项, ✅ = 基本满足待声明。

> ⭐ **2026-08-10: 五个起手文件已齐**（Guide §26）。`FIGURE_PLAN.md` 落地 = **17 主文图 + 9 appendix
> + 6 条图注雷区 + 四检查登记表**，Stage A（图↔章问题↔claim↔数据源绑定）已定，Stage B（页数预算）
> 待 handbook（**user 已走 AskUCL**；`GPT_SEARCH_PROMPTS.md` P1 并行加速）。
> 决策已锁：**disc+concl 合并为 Ch7** · shop/WA 只作 external validation 落地才进 · **A2 pass@K 现在做**。
> ⚠️ 顺带核出三处过时口径（C4 `0/6→0/8` · `B=200→B=10000` · WA learnability 缺口其实 08-03 已解决）
> 与一个未修 bug（`router_triage_learnability.py` 散文段硬编码 6 格，m 未随 8 格更新）→ 笔记 §450。

- [x] 1. ✅ **problem-first** (Ch1 §1.1 开篇即 problem) — — 开局讲 problem 不讲背景铺垫 (intro 重排)
- [x] 2. ✅ **system structure 图** = **F3 已画** (2026-08-10) — 端到端架构图。**规格已定 = FIGURE_PLAN F3**：不是装饰性架构图，
       画 *causal comparison boundary*（`CHAPTER_CHAIN.md:237` 原文链条）。零数据依赖，可立即开工
- [x] 3. ✅ **public-access 声明** (frontmatter 独立一章, B0=proxy 已标『replicable from logs, not independently reproducible』) — — VWA/Qwen/Gemma/MiMo 均 public + OSF; ⚠️ B0=proxy 需诚实标"非公开可复现"
- [x] 4. ✅ **concept-first intro** (Ch1 §1.3 Concepts before machinery) — — 从 concept 建立, 不堆 statement, 不太具体
- [x] 5. ✅ **disc+concl 合并** (Ch7) — (limitation 内嵌) — **今日 D8 章结构直接受此影响**
- [x] 6. ✅ **appendix 非 orphan** (A/B/C/D 各有正文 `见 Appendix X` 指针) — — 每个 appendix 正文有 "见 Appendix X" 指针 (S1-S6 待补入口)
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
- [x] 11. ✅ **每章有开头+结尾** (`\chapterhandoff` 宏, 12 处 = Ch1 尾 + Ch2-6 首尾 + Ch7 首; grep 可查) — — 导语 + 小结/过渡 (逐章补, 机械必做)
- [x] 12. ✅ **公式** (8 条: router/agent policy · oracle · drop-one · net saving · floor SD · rerun union · value decomposition) — — obs 构造/drop-one/estimand/router 规则形式化
- [~] 13. 部分 **每个 setting justify** (max_steps=30 / 4B backbone / 两站 / scored-set 已给理由; 其余待补) — — 4B/三站/step_002/24 task/50 tokens 逐个给理由 (素材在笔记)
- [ ] 14. (学长止于 13, 留白)

**并行策略**: 2/8/10/12 纯离线不占 GPU → 可与 mechanistic sweep 同时推进。
**通用性**: 1/4/6/12/13 对 REALM Paper A/B 也适用。
