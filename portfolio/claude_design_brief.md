# Claude Design 弹药包 — 求职三件套

> 用法：打开 `claude.ai/design`（Pro/Max 账号，和 Claude Code 同账号不同入口）。
> **Part A 在 onboarding 里贴一次**（建立设计系统，之后每个项目自动继承）。
> **Part B/C/D 各自开一个新项目**，把对应 brief 整段贴进去。
> 导出 HTML 后回到 Claude Code（我），走 **Part E** 由我落地 + 核对数字。
>
> 现状盘点（2026-06-02）：
> - `index.html` = 手写 portfolio one-pager，已成熟（5 段 + 手绘 SVG + scroll-reveal）。
> - `Phantom Space.html` = **已经是 Claude Design 导出的 animated explainer**（产物 ③ 已有 v1）。
> - `results/phantom_paper/figures/*.png` = 20+ 张 paper-grade 图，现成可喂。
> - **缺口 = slides（产物 ②，完全没有）→ 优先级最高。**

---

## Part A — 设计系统（onboarding 贴这段，建立一次，三件套共用）

```
请把以下设计系统作为本团队的固定视觉语言，之后所有项目都自动套用。
这是一个学术研究求职作品的视觉系统：克制、可信、像高质量论文 + 现代 web 的结合。

— 调色板 —
背景纸色 paper      #f7f7f5   （暖白，正文底）
卡片 card           #ffffff
正文墨色 ink         #17191e   （近黑）
次要文字 muted       #5b626d
分隔线 line          #e6e7e2
主强调 HERO          #b21e45   （crimson，唯一的品牌色，钉死用这个；
                                 不要用旧 explainer 的 #7a1e1e —— 统一成 #b21e45）
HERO 浅填充          #f6e3e9
正向/最佳 good        #1f7a5c   （绿，标"best"数字）
中性 slate           #3f5168
诚实声明 warn         #9a6b00   （琥珀，仅用于 provenance / honest-status 框的左边框）

— 字体 —
正文：衬线 "Iowan Old Style" → Palatino → Georgia → "Noto Serif"，约 15.5px / 行高 1.62
UI / 标签 / 数字：无衬线 -apple-system / "Segoe UI" / Roboto / "Noto Sans"
代码：等宽 ui-monospace / Menlo
大标题 h1：38px，weight 600，字距 -0.02em
分节标签（kicker / h2）：12–12.5px，全大写，字距 0.13–0.22em，HERO 色，weight 700
导语 lead：23px，weight 500
所有数字用 tabular-nums（等宽数字，表格对齐）

— 组件 —
卡片：白底，1px line 边框，圆角 14px，极淡阴影
chip：胶囊，白底 1px 边框，圆角 999px（放技术栈标签）
provenance 框：琥珀左边框 + 暖黄底（#fffdf6），放"诚实声明 / honest status"
takeaway 框：HERO 左边框 + 白底，放每节结论金句
水平条形图：浅灰轨道 + 实心填充，进场时 scaleX 0→1 动画
表格：thead 全大写小字，数字右对齐，最佳值用 good 绿加粗

— 动效 —
scroll-reveal：元素进视口时 opacity 0→1 + 上移 14px→0，0.7s 缓动
SVG 线条 draw-in：stroke-dashoffset 动画
必须 respect prefers-reduced-motion（无障碍）
版心 max-width 1000px

— 气质 —
克制、学术、可信。不要花哨渐变 / 霓虹 / emoji 堆砌。
诚实优先：永远保留 honest-status / provenance 声明，不夸大。
```

---

## Part B — 产物 ②：答辩 / workshop slides（**优先做这个，唯一的真缺口**）

```
把下面这份研究做成一个演示文稿（presenter-facing slide deck），10–12 页，
严格套用我们已建立的设计系统（crimson #b21e45 / 纸色 / 衬线正文 / 等距 SVG 立方体风格）。
一页一个核心 idea，给演讲者讲，不是给人读的密集文档。

研究：Cost-Aware Routing for Web-Usage Agents（本科毕设，独立完成）
一句话：web agent 可以把页面读成 screenshot / accessibility tree / Set-of-Marks；
       哪种表征值得付费？能不能在它们之间 route 打败任何单一表征？

叙事弧（按这个顺序出 slide）：
 1. 标题页 — 题目 + 核心问句 + "本科毕设·独立·~120K LOC Python·4 个月 1141 个测试"
 2. The phantom routing space — 六个观测模式是一个 2×2×2 设计立方体的角
    （text channel × prompt style × 是否渲染截图）；整个"no-image 面"= phantom space
 3. 实验设置 — 3 个模型族（Qwen3-VL 235B / 4B · Gemma-3-4B）× 6 模式 × 3 个 VWA 站点；
    preregistered + power-analysed；3-tier compute fleet
 4. 实验表 — 六模式正面交锋：没有单一 phantom 模式赢在原始成功率上
    （DOM 17.4 / SoM 27.2 / Vision 25.0 / P-text 15.6 / P-SoM 15.6 / P-prompt 19.6 %）
 5. Cost 是地板，routing 是杠杆 — 六模式都挤在 $0.065–0.072 窄带，
    但 6-mode routed oracle 达 43.3%（比最佳单模式 +16.1pp）
 6. 加一个 phantom arm 有用吗？— pooled meta (k=3)：P-SoM +2.34pp [1.30, 3.37]，I²=0%，
    过 preregistered gate (H₀: θ≤+1.0pp) p=0.006；power 81%→k=6 时projected 97%
    金句："一个正面交锋会输的表征，统计上反而是最值得 route 过去的"
 7. Failure analysis — 224 个任务里只有 88 个"routable"，那就是全部机会所在；
    确定性 taxonomy 显示 P-SoM 专门救 image-perception 失败
 8. 工程 — race-safe 分布式编排跨 3 个算力层 + 6 层 contamination watchdog +
    schema-versioned JSONL + 1141 个测试守护分析管线
 9. Honest status — classifieds 全六模式跑完(最大模型)，其他 cell 还在 fire；
    k=6 干净重跑进行中；failure taxonomy 是 provisional。每个数字都真实且标注 scope
10. 谢页 — 联系方式 + GitHub + "Built on VisualWebArena"

约束：
- 所有数字必须和我给的完全一致，不要自己编或四舍五入改动
- 每页保留学术克制气质；hero 立方体那页用等距 2×2×2 风格（参考我们的 explainer）
- 给我可导出 PPTX 的版本
```

> 做完导出 **PPTX**（演示直接用）。这一件不需要我介入实现。

---

## Part C — 产物 ①：portfolio one-pager 升级（喂现有 HTML，**只做定点增强**）

```
我有一个已经成型的求职 one-pager（HTML 我会上传：portfolio/index.html）。
它已经很好了，请不要推倒重做、不要改任何数字、不要动 honest-status 声明。
只做下面这几个"交互性升级"，严格保持现有视觉语言：

1. 让第 01 节的等距设计立方体可交互：鼠标拖拽轻微旋转 / hover 时高亮"no-image 面"
   并浮出每个角的标签（DOM / P-text / P-prompt / P-SoM / SoM / Vision）
2. 让第 03 节的 cost 散点图在滚动进入时动画：六个点从轴飞到各自位置，
   routing-headroom 阴影带渐显
3. 嵌入演示视频（我会上传 portfolio 里的 mp4），带一个静帧封面，点击播放
4. 加一个右侧 sticky 迷你导航 / section 进度指示（01–05）
5. 加一个 dark mode 切换（沿用同一套语义色，反转纸色/墨色）
6. 移动端打磨：表格在窄屏可横向滚动而不挤压

绝对约束：
- 不改任何实验数字、CI、p 值、power
- 保留 provenance 琥珀框原文（"Read this honestly..."）
- 保持 prefers-reduced-motion 支持
给我导出 standalone HTML。
```

> 做完导出 **HTML** → 走 Part E 交给我，我合并进 repo + 核对每个数字仍指向真实源文件。

---

## Part D — 产物 ③：图解释器（你已有 explainer v1，二选一）

你已经有 `Phantom Space.html`（Claude Design 导出的 animated explainer）。两条路：

**(i) 扩充现有 explainer** — 在同一风格里再加几张图的动画讲解（forest plot、failure matrix）。
**(ii) 新建"图库走查"** — 把 `results/phantom_paper/figures/` 里的关键 PNG 导入，逐张配旁白。

推荐 (ii) 的 brief（新建项目，上传下面这些 PNG）：

```
我会上传一组研究图（PNG）。请做一个 animated 图库走查（scrollytelling），
严格套用我们的设计系统。每张图配一句"这张图告诉你什么"的旁白，滚动逐张揭示。

建议入选图（按叙事排序）：
- fig_phantom_structure_venn.png   — 模式间任务重叠/互补的结构
- fig0c_drop_one_oracle.png        — drop-one oracle lift（routing 价值的核心证据）
- fig_meta_forest.png / fig_forest_drop_one.png — pooled meta forest（per-cell + 合并）
- fig0a_sr_per_mode_heatmap.png    — 各模式成功率热图
- fig0g_routing_auroc_heatmap.png  — routing 信号 AUROC
- fig3d_cost_sr_frontier.png       — cost–SR 前沿
- fig_failure_modes_per_cell.png   — 失败类型分布

气质：像一篇会动的 paper figure walkthrough，克制、可信，不花哨。
给我导出 standalone HTML。
```

> 注意：这些 PNG 是真实分析产出，旁白只描述图、**不要让它编新数字**。

---

## Part E — Handoff 回 Claude Code（我接手的部分）

任意产物从 Claude Design 导出 HTML 后，回到这个终端，告诉我文件路径（或贴进来），我做：

1. **数字核对（最关键）** — Claude Design 会生成看似合理的内容；研究求职作品里**每个数字都必须能溯源到真实文件**。我会逐个比对：
   - SR / cost → `docs/analysis/cross_sites/cost_per_mode.json`
   - pooled lift → `results/phantom_paper/meta_phantom_lift.csv`
   - power → `docs/analysis/cross_sites/power_analysis.md`
   - failure taxonomy → `docs/analysis/vwa_classifieds/B0_classifieds_6mode_failure_taxonomy.md`
   任何对不上的数字就地标红改回真值。这一步对应你 paper-grade 的核实纪律。
2. **合并进 repo** — 放进 `portfolio/`，统一资源路径（mp4、图）。
3. **工程化清理** — 响应式 / 性能（2MB bundle 瘦身）/ 无障碍（alt 文本、reduced-motion）。
4. **commit**（push 前会问你确认）。

---

## 推荐执行顺序

1. **先贴 Part A** 建设计系统（一次，10 分钟）。
2. **做 Part B（slides）** — 唯一真缺口，求职/答辩两用，杠杆最高。
3. **做 Part C（portfolio 增强）** — 你的门面，导出后我帮你核数字 + 合并。
4. **Part D（图解释器）** — 你已有 v1，时间够再扩。

设计系统一旦钉死，B/C/D 三者风格自动统一 —— 这正是"建一次、共用"的回报。

---

## Part F — slides 渲染修复 prompt（PPTX 导出崩坏）

> 贴进 deck 项目。核心：bug 只在**导出的 PPTX**出现，live 预览是好的 → 必须强制它检查导出 + 缩短文字到不溢出。

```
The Phantom Routing Defense Deck's exported PPTX has layout/rendering defects.
The DATA, NARRATIVE, 10-slide STRUCTURE, and the crimson/paper/serif design system are all
CORRECT — do NOT change any of them. The ONLY problem is LAYOUT IN THE EXPORTED PPTX:
text boxes overflow, collide with neighbours, and — most visibly — words at line-wrap points
get DUPLICATED.

CRITICAL: this bug appears ONLY in the exported PPTX/PDF, not the live preview. After fixing,
you MUST export to PPTX and inspect the rendered slides — do not judge by the web view.

(1) DECK-WIDE — kill text overflow and the duplicated-word artifact.
Examples of the duplication (these words are doubled at wrap points and must read once):
  "schema schema version", "the analysis analysis pipeline", "silently silently reads",
  "produces produces these figures is itself itself covered", "the clean the clean run",
  "The next two slides two slides", "over best single mode mode", "labelling loop. loop.".
Fix by: sizing every text box large enough for its text at the export font size with generous
padding; SHORTENING copy so each box fits without unexpected wrapping; turning OFF autofit-shrink;
using manual line breaks where a specific wrap is intended so the exporter never re-flows.
Prefer fewer words / shorter lines over tightly packed paragraphs.

(2) SLIDE-SPECIFIC rebuilds:
- Slide 2 (design cube): the bottom-left "phantom space" callout card is fully collapsed — its
  text is stacked on top of itself. Rebuild it as a clean short card. Mode labels must each stay
  on ONE line — "DOM" is rendering as "DO / M". Keep DOM, P-text, P-prompt, P-SoM, SoM, Vision intact.
- Slide 3 (setup): "Gemma-3-4B" overflows its chip ("4B" wraps below) — widen it or shorten the
  label. The "Preregistered" and "Power-analysed" card text is garbled ("no clean run — no clean run",
  "targ et", "com mitting com pute") — re-set as normal text reading
  "Hypotheses, gate threshold and stopping rule fixed before the clean run — no post-hoc
  target-shopping." and "Sample sizes chosen to detect a +1.0pp effect at target power before
  committing compute."
- Slide 4 (raw SR): the kicker "SIX MODES, RAW SUCCESS RATE" overlaps the title. Separate them.
  Takeaway should read "The next two slides show why that reading is wrong." Bar labels split
  ("SoM/M", "DO/M") — keep each on one line.
- Slide 5 (cost): the big "43.3%" stat breaks as "43.3" + "%" on two lines and overlaps its caption
  "6-mode routed oracle". Render "43.3%" as one unit with the caption cleanly below. Fix
  "over best single mode" (drop the doubled word).
- Slide 7 (failure): the "136" stat renders as "13" with "6" dropped below, colliding with
  "fixed ceiling / fixed floor". Render 136 as one number. Fix "labelling loop." (single).
- Slide 8 (engineering): all four card bodies duplicate words. Rewrite short and clean:
   • Race-safe orchestration — one distributed scheduler across three compute tiers; no lost or
     double-counted runs under contention.
   • Contamination watchdog — six independent layers catch train/test leakage and cross-run bleed
     before any cell is admitted.
   • Schema-versioned logs — every result row carries a schema version, so analysis never reads
     stale shapes.
   • Tests on the pipeline — the analysis path that produces these figures is itself covered by
     1,141 tests.

(3) NUMBER corrections (everywhere they appear — slides 1, 8, 10):
  "~117K LOC Python" -> "~120K LOC Python"
  "1,121 tests" / "1,121 test functions" -> "1,141 tests"

(4) AUTHOR / contact (defense deck):
  Slide 1 (title): add a byline "Jiaming Wei · Undergraduate thesis".
  Slide 10 (closing): keep the GitHub repo link; add "ucab352@ucl.ac.uk ·
  linkedin.com/in/jiaming-wei-810ab938b".

Keep everything else identical. Re-export PPTX and confirm every slide's text fits with NO overflow,
NO collision, and NO duplicated words.
```

