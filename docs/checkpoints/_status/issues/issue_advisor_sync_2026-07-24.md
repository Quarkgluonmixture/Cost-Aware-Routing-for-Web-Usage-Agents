---
type: issue
category: decision
status: active
priority: high
action: 学长教毕设写作法 13 条 → 逐条落成 task_thesis_draft 可执行清单; D8 (results+discussion 章, 今日) 结构直接受第 5 条影响 (disc+concl 合并)
updated: 2026-07-24
created: 2026-07-24
---

# Advisor sync 2026-07-24 — 毕设 (thesis) 写作方法 13 条

学长口授的毕设论文写作 rubric。**注意 scope**: 这是**毕设 (thesis, D7-D11, official
submission early-Sep)** 的写作法, 与 REALM workshop 双稿 (Paper A/B, 08-05, 8 页) 不同尺度 —
其中若干条 (system-structure visual / EDA benchmark / 文献图谱 / chapter framing) 是
thesis-scale; 但 problem-first / justify-settings / non-orphan-appendix / 公式 对两者都适用。

下方每条 = 学长原话 → 我的可执行解读 → **当前工作满足度**。原话逐字保存防解读漂移。

## 13 条 (逐条)

1. **problem first** — 开篇 problem-driven: 先讲要解决的 problem, 不要先堆背景/方法铺垫。
   → 当前 paper §1 是 phantom-space 现象 hook 起, 偏 finding-first; thesis 需重排成
   "web agent 的 SR-cost 权衡是个 problem → 表征选择是杠杆" 的 problem 开局。**GAP**。

2. **system structure (visual)** — 要有系统结构图 (visual), 让读者一眼看懂 pipeline。
   → 已有 `docs/checkpoints/canvas/experiment_matrix` + PHANTOM_SOM_CODE_TOUR; 但缺一张
   **端到端 system 架构图** (obs mode → backend → runner → router → eval)。**部分/GAP**。

3. **确认 public access to resources** — 用到的资源 (benchmark/模型/代码/数据) 必须 public
   可访问, 并在文中声明。VWA (public) / Qwen3-VL-4B (HF public) / Gemma-3-4b-it (HF) /
   MiMo-VL (HF) 都 public; 自产物 → OSF DOI 公开。**基本满足**, 需在 thesis 显式成段声明
   (repro statement 已有雏形)。⚠️ B0 = 235B via **AWS proxy** 非 public → 需诚实标注
   "B0 通过受限 proxy, 非公开可复现" (这条恰是 reviewer/考官会追的点)。

4. **concept 开始, 不要太具体, 不要写成 statement, introduction** — Introduction 从 concept
   层建立, 不要一上来太细节, 不要写成一串断言 (statement) 堆砌。
   → intro 应先立 concept (representation routing / 观测模态的信息-成本权衡), 再收敛到本文
   具体 phantom 构造。**GAP** (当前 draft intro 偏 claim-dense)。

5. **discussion 和 conclusion 合二为一 (limitation)** — Discussion 与 Conclusion 合并成一章,
   limitation 纳入其中。
   → ⭐ **今日 D8 (results+discussion 章) 直接受影响**: 章结构按合并版组织, 不要分裂成
   独立 Conclusion 章。当前 paper_drafts 有独立 §7 Discussion + §8 Limitations → thesis 版
   合并。**今日执行项**。

6. **appendix 要非 orphan** — 每个 appendix 必须被正文交叉引用 (non-orphan), 不能挂在那没人指。
   → supplement S1-S6 骨架已在; 需逐个在正文补 "见 Appendix X" 指针。REALM Paper A 的
   appendix 无限也适用此条。**待检查每个 appendix 的正文入口**。

7. **找论文 Zekun 原文** — 找 co-author **Zekun Wu** 的原始论文 (读 + 引 + 对齐)。
   → Zekun Wu = 共同作者 + 互审提名人 (§375)。动作: arXiv/scholar 拉 Zekun Wu 一作/通讯
   论文, 判定与本文 routing/agent/评估的接点, 入 bib + 文献图谱。**待做** (可 codex/zero-preset
   cross-AI 核验, 遵 [[feedback_arxiv_api_for_verification]] 用 arXiv API 不用 WebSearch)。

8. **文献图谱** — 做文献图谱 (literature map/graph), 结构化呈现相关工作的谱系。
   → 已有 `docs/literature/` + raw_digest_triage; 需产出一张**可视文献图谱** (簇: routing /
   观测模态 / confidence 信号 / mechanism), 节点=论文, 边=关系。thesis §2 用。**GAP** (有素材无图)。

9. **visualization** — 多用可视化。
   → 已有大量 figure 脚本; thesis 需系统化配图, 尤其配合第 10 条 EDA。遵
   [[feedback_dashboard_style]] (少黑话/少字/多可视化) + dataviz skill。**持续项**。

10. **EDA benchmark** — 对 benchmark 做 EDA (探索性数据分析), 展示数据特征 (task 分布 /
    难度 / 站点差异 / 长度分布)。
    → VWA cls234/red210/shop466 的 task 分布、intent 长度、action 类型分布、site 难度差异
    尚无系统 EDA 章。**GAP** (新工作项, 但纯离线可现在做, 不占 GPU)。

11. **每一个 chapter 都有开头和结尾** — 每章要有开头 (导语: 本章要干什么) + 结尾 (小结 +
    过渡到下章)。chapter framing / transitions。
    → paper_drafts 各 section 目前无统一 opening/closing 段。thesis 逐章补。**GAP** (机械但必做)。

12. **公式** — 要有公式 (formalization): 形式化定义 + 方程。
    → 6 个 obs mode 的构造、drop-one oracle、cost/latency estimand、router 决策规则均可
    形式化。当前多为 prose 描述。**GAP** (把已有定义写成公式)。

13. **每一个 setting 都 justify** — 每个实验 setting / 设计选择都要给出 justification (为什么
    这么设: 为什么 4B / 为什么这三站 / 为什么 step_002 / 为什么 24 task / 为什么 50 tokens)。
    → 散见于笔记 WHY, 但 thesis 需成段落 justify 每个 knob。**GAP** (素材在笔记, 需收敛成文)。

14. (学长列表止于 13, 第 14 项空 — 留白, 可能后续补)

## 落地映射

- **今日 (D8)**: 第 5 条 — results+discussion 章按 disc+concl 合并结构写 (limitation 内嵌)。
- **thesis 结构级 GAP (1/4/11)**: problem-first 重排 + concept-first intro + 逐章 framing。
- **新离线工作项 (2/8/10/12)**: system 架构图 / 文献图谱 / benchmark EDA / 公式化 — **纯离线,
  不占 GPU, 可与 mechanistic sweep 并行推进**。
- **核查项 (3/6/13)**: public-access 声明 (含 B0 proxy 诚实标注) / appendix 非 orphan / 每
  setting justify。
- **调研项 (7)**: Zekun Wu 原文 → bib + 文献图谱 (zero-preset cross-AI 核验)。

## Cross-link
- [[task_thesis_draft]] (可执行清单已 append) · [[实验笔记]] §385 · paper_planning §19 decision log
- 时间链: D8 (results+disc 章, **今日 07-24**) → D9 全稿 v1 (08-10) → D10 修订 (08-24) → D11 官方提交 (early-Sep)
- 区别 REALM Paper A/B (08-05, 8 页) — 部分条 (1/4/6/12/13) 对两者通用
