---
type: showcase-planning
status: draft
purpose: 海报 v9 重做的素材库 — 先筛素材，再定版式
created: 2026-09-03
---

# 海报素材库（v9 重做）

**为什么先做这个**：v5–v8 是"先定结构，再找图填"，结果纸上七成是文字。
学长给的参考（`117982.png`，NeurIPS 横版三栏）是反过来的——**图先占位，文字只做短 bullet**。
所以 v9 从"我们手上到底有哪些图能放"开始。

**Vision（user 2026-09-03）**：路过的人一眼被吸引，**看两眼就知道在做什么**。
⇒ 判据不是"这张图严谨吗"，而是**"不懂这个领域的人 3 秒内能不能看出它在说什么"**。

**Title**（用论文题目）：
> **When Is Expensive Perception Worth Paying For?**
> *Measuring the Ceiling, the Predictability and the Economics of Representation Routing in Web Agents*

---

## 0. 筛选标准

| 评级 | 含义 |
|---|---|
| ⭐⭐⭐ | **路人 3 秒懂**。有真实截图、颜色分组明显、或形状本身就是结论 |
| ⭐⭐ | 看两眼懂。需要一句图注，但不需要背景知识 |
| ⭐ | 同行懂。要先解释轴或术语 |
| ✖ | 不放海报。审稿人才需要的方法学控制 |

---

## 1. 真实截图（user 特别要求"多放截屏"）

| ID | 素材 | 路径 | 内容 | 评级 |
|---|---|---|---|---|
| **S1** | raw ↔ SoM 标注配对 | `docs/checkpoints/周报/weekly-dashboard/public/figures/mode_{raw_screenshot,som_annotated}.png` (1280×660) | 同一页面：左原始截图 / 右带蓝框编号标注。**这是"三种看法"最直观的一张** | ⭐⭐⭐ |
| **S2** | agent 真实轨迹截图 | `results/repro_replicates/B0_{dom,vision}_classifieds_*/phase1_*/artifacts/<task>/step_*/screenshot.png` | **7,082 张**真实运行截图：商品列表、详情页、下拉菜单展开、搜索结果、购物车 | ⭐⭐⭐ |
| **S3** | AXTree 文本样本 | `deliverables/showcase/figures/eye_read.txt` | READ 模式真正送进模型的东西（`[2] RootWebArea 'Classifieds'…`）。**与截图并排才有冲击力** | ⭐⭐⭐ |
| **S4** | 三站点标识 | `external/visualwebarena/environment_docker/webarena-homepage/static/figures/{reddit,onestopshop}.png` | 站点缩略图，可做"跑了哪些网站"的图标条 | ⭐⭐ |
| **S5** | v8 已裁好的三模式缩略图 | `deliverables/showcase/figures/{eye_look,eye_both,thumb_130,thumb_76,thumb_17}.png` | 现成、尺寸已调、与 laptop demo 同源 | ⭐⭐ |

> ⚠️ **SoM 标注截图只有 S1 一张**（som run 的 artifacts 已清理）。要更多标注图得重新跑，
> 或用 S2 的原始截图自己叠框。

---

## 1b. ⭐ 头号素材：矢量系统图（第一轮漏掉）

| ID | 素材 | 路径 | 内容 | 评级 |
|---|---|---|---|---|
| **A0** | Representation Routing 三段系统图 | `deliverables/representation_routing_merged_three_sections.svg`（**矢量**，另有 `image.png` / `image copy.png` 两个 PNG 版） | ① Agent ↔ Web Page（机器人插画 + one agent step 流程）② Grounding representations（DOM 橙 / SoM 绿 / Vision 蓝，**每格嵌真实截图**）③ Routing and outcomes（三种策略 + 两个结果） | ⭐⭐⭐ |

> **这是目前手上最抓人的一张**：彩色、有插画、嵌真实截图、三段式把整个研究讲完，
> 而且是 **SVG 矢量**，横跨海报全宽也不糊。毕设里的 `fig_overview.pdf` 是它的简化版。
> **建议：直接做海报顶部横幅**，路人扫一眼就知道"agent 看网页 → 三种看法 → 路由与结果"。

---

## 2. 毕设图 — 概念 / 系统

| ID | 图 | 文件 | 它说什么 | 评级 | 建议 |
|---|---|---|---|---|---|
| **F1** | Motivating example | `fig_f1_motivating_example.png` | 三行 = DOM / SoM / Vision 真正送进模型的东西，**每行带真实截图 + 字符数 + token 数** | ⭐⭐⭐ | **主图候选**。整张海报最"一眼懂"的一张 |
| **F0** | Thesis overview | `fig_overview.pdf` | 左任务 → 中六种表征 → 右 agent loop，底部结论带 | ⭐⭐ | 可做左栏顶部的 system diagram |
| **F1b** | Phantom diamond | `fig_f1_diamond_schematic.pdf` | 四个 phantom mode 的 2×2 构造（格式 × prompt 风格） | ⭐ | 太概念，**建议不放** |
| **F5** | Design matrix | `fig_f5_design_matrix.png` | 6 mode × 8 cell 热力表，橙/绿/蓝三色 = 无图/图文/纯图 | ⭐⭐⭐ | **表格候选**。既是表又是图 |

## 3. 毕设图 — 结果

| ID | 图 | 文件 | 它说什么 | 评级 | 建议 |
|---|---|---|---|---|---|
| **F6** | SR by class | `fig_f6_sr_by_class.png` | 8 行点阵，圈出每格最优——**没有一种看法在所有格子里最优** | ⭐⭐⭐ | **强推**。形状即结论 |
| **F13** | Dominance plane | `fig_f13_dominance_plane.png` | 绿色胜区内**一个点都没有** | ⭐⭐⭐ | **强推**。空白本身就是结论 |
| **F8** | Ceilings | `fig_ceilings.pdf` | 左：单模 vs 六模并集的 SR 差；右：省了多少钱 | ⭐⭐ | 可放，需一句图注 |
| **F7** | Cost–SR frontier | `fig_f7_cost_sr_frontier.png` | 8 个小 panel，成本与成功率不共线 | ⭐⭐ | panel 太多，**建议只取 2 格** |
| **F10** | Rerun discordance | `fig_f10_rerun_discordance.png` | 六条横条：同一设置跑两次，**10–14% 的任务结果会翻面** | ⭐⭐⭐ | **强推**。极简、反直觉、路人也懂 |

## 4. 毕设图 — 语料 / 方法学

| ID | 图 | 文件 | 评级 | 建议 |
|---|---|---|---|---|
| **F4** | Corpus EDA（四 panel） | `fig_f4_corpus_eda.png` | ⭐⭐ | 可取 panel B（WA 三站 ref-image 全 0）单独用 |
| **F11** | 特征集对照 AUROC | `fig_f11_feature_set_ablation.png` | ⭐ | ✖ 需要解释 AUROC |
| **F9** | drop-one vs 噪声带 | `fig_f9_drop_one_vs_floor.png` | ⭐ | ✖ 审稿人图 |
| **F10b** | one-arm margin | `fig_f10b_one_arm_margin.png` | ⭐ | ✖ 审稿人图 |
| **F12** | 置换零分布 | `fig_f12_permutation_control.png` | ⭐ | ✖ 审稿人图 |
| **F16** | 欠采样控制 | `fig_f16_undersampling_control.png` | ✖ | ✖ 审稿人图 |

## 5. 表格（`docs/checkpoints/paper_drafts/ablation_tables.md`，共 44 张，由脚本生成）

| ID | 表 | 内容 | 评级 | 建议 |
|---|---|---|---|---|
| **T1** | Table 1 — SR per mode | 8 cell × 6 mode 成功率，**加粗每行最优** | ⭐⭐⭐ | **强推**。最好读的一张表 |
| **T42** | Table 42 — 完美选择能买到什么 | 两个天花板，只有一个过了自己的控制 | ⭐⭐ | 可放，需精简列 |
| **T2** | Table 2 — 三类部署方式的最优 | no-image / vision-only / hybrid | ⭐⭐ | 与 F6 重复，二选一 |
| **T44** | Table 44 — 标签供给 = 路由价值 | 能学的行数 = 能赢的行数 | ⭐⭐ | 讲"为什么学不会"时用 |
| 其余 40 张 | — | 方法学 / 口径 / 控制 | ✖ | ✖ |

## 6. `results/phantom_paper/figures/`（53 张，第一轮漏筛 — 有好几张比毕设图更适合海报）

| ID | 图 | 文件 | 它说什么 | 评级 | 建议 |
|---|---|---|---|---|---|
| **P1** | SR 热力图 | `fig0a_sr_per_mode_heatmap.png` | 6 看法 × 6 设置，深浅即高低，每格带 N | ⭐⭐⭐ | 与 F5 二选一（P1 更像图，F5 更像表） |
| **P2** | 解题池堆叠条 | `fig0f_overlap_stacked_bar.png` | 每种看法解了多少题、其中多少是**只有它解开的** | ⭐⭐⭐ | **强推**。直接回答"为什么要路由" |
| **P3** | 韦恩图 ×6 | `fig_phantom_structure_venn.png` | 三种文字看法的解题集**重叠但不重合** | ⭐⭐⭐ | **强推**。韦恩图是路人最熟的图形 |
| **P4** | 失败原因分布 | `fig_failure_modes_per_cell.png` | 24 行彩色堆叠 = 每个设置的失败长什么样 | ⭐⭐⭐ | 视觉冲击强；**行太多需裁到 8 行** |
| **P5** | 区域碳排放 | `fig3_regional_carbon.png` | 同一实验换部署地区，碳排放差好几倍 | ⭐⭐⭐ | ⭐ **和你的专业（AI for Sustainable Development）直接对口**，其它素材都没这条 |
| **P6** | API vs 本地电费 | `fig3d_cost_sr_frontier.png` | 同一实验两种计费口径差 **107× / 82×**，红框标注 | ⭐⭐ | 讲"成本"时很有说服力 |
| **P7** | SR by class（workshop 版） | `deliverables/vlm4rwd/figures/fig_sr_by_class.pdf` | 与 F6 同数据，**更干净**（三色 + 圈最优） | ⭐⭐⭐ | **建议用它替掉 F6** |
| **P8** | 能力 × 表征失败迁移 | `周报/…/fig_capability_b0_b1.png` | 小模型的失败往哪偏，红色 **+43.7pp** | ⭐⭐ | 备选 |
| **P9** | 任务池 Jaccard ×6 | `fig0d_taskpool_jaccard.png` | 红蓝矩阵，看法之间解题集相似度 | ⭐⭐ | 与 P3 重复，二选一 |
| **P10** | 路由 Pareto 平面 | `fig_router_pareto_plane.png` | 各策略在成本-成功率平面的位置 + Oracle 星标 | ⭐⭐ | 与 F13 重复，二选一 |
| **P11** | 融合溢价森林图 | `vlm4rwd/figures/fig_fusion_forest.pdf` | 效应量对着**灰色重跑噪声带**读 | ⭐⭐ | 想讲"我们做了严格对照"时用一张就够 |

**同库其余约 40 张**：mechanistic（`fig_stage4_*` / `fig_axis2_*` / `fig_mech_*`）+ micro-divergence
+ per-task fragility。⚠️ mechanistic 属 advisor 2026-05-14 判"先不要管"的 §5，**不进海报**。

## 6b. 其它图库

- `deliverables/vlm4rwd/figures/` — 8 张 workshop 稿图（`fig_sr_by_class` / `fig_fusion_forest` /
  `fig_partition_forest` 是毕设没有的独立版本，且**排版更干净**）
- `docs/checkpoints/周报/weekly-dashboard/public/figures/` — 24 张，仅 3 张为独有
  （`fig_capability_b0_b1` + S1 的截图配对）
- `results/mechanistic/` — 66 张 patching 输出，**属暂搁的 §5，不放**
- 各 run 的 `analysis/*/plots/` — cost / step / latency 分布直方图，**质量一般，不建议**

---

## 7. 初步取舍建议（待 user 拍板）

**头号**：**A0 矢量系统图**（做顶部横幅）
**一定放**：S1 截图配对 · S2 真实轨迹 · F1 三看法并排 · P2 或 P3（谁独解了什么）· F13 空胜区 · P7（替 F6）· T1 成功率表
**很值得放**：P5 碳排放（专业对口，别的素材没有）· F10 翻面率 · P1 或 F5 热力图
**可以放**：P6 成本口径 107× · F8 天花板 · P4 失败分布（裁到 8 行）· S4 站点条
**不放**：F9 · F10b · F11 · F12 · F16 · F1b · mechanistic 全部 · 其余 40 张表

**大数字**（`+16.35 in 100` / `0 of 8` / `1 of 8`）—— user 判"很low"，v9 取消。
同一信息改由 **F13 的空白胜区**和 **T1 的加粗列**承担。

---

## 8. 待定的版式问题

1. **纸张**：organiser 的 A1 竖版模板（594×841，v5–v8 一直遵守，"do not resize")
   vs 参考图的横版三栏。**换横版要先确认 organiser 允不允许。**
2. **栏数**：竖版 A1 放 8–10 张图，3 栏还是 2 栏。
3. **demo 分工**：laptop 放三任务逐步回放；海报放静态截图。user 已确认**不冲突**。

---

## 9. 截图怎么放 — 三个做出来的候选

产物在 `deliverables/showcase/candidates/`（**图都在 showcase 下，不在 temp**）。
三个候选都用同一批真实运行截图，区别是**讲哪一层故事**。

### 候选 A — 同一任务，两种看法走岔 `strip_A_two_lanes.png`

`classifieds task 76`：*"Navigate to my listing of the blue bike and change the price to $85.50
(including in the description)."*

| | | |
|---|---|---|
| **READ**（只读页面文字） | 12 步 · $0.090 | ✅ 解开 |
| **LOOK**（只看截图） | 26 步 · $0.103 | ❌ 没解开 |

⭐ **这条最有说服力，因为证据在图里而不在图注里**：LOOK 的 step 8 和 step 17 两张截图
**几乎一模一样**（都是空白的 "Publish a listing" 页）——不需要任何解释，一眼就看出它在原地打转。
贵 15%、慢一倍、还没做成。

⚠️ **一个要处理的坑**：LOOK 的最后一帧（step 25）看起来像成功了（页面显示 85.50 $ 和
"we've just updated your listing"）。它确实改了价格，但把 description **整段替换**了，
而任务要求价格也出现在描述里，所以判失败。**要么换掉这一帧，要么图注写明**，否则读者
会觉得记分板在骗人。

### 候选 B — 三个任务，三个不同赢家 `strip_B_three_tasks.png`

v8 右下角那张桥接表的升级版：换成真截图。截图内容正好呼应任务（日落的船 / 蓝色自行车 /
$900 的山地车），这一点很讨喜。

⚠️ **与 laptop demo 重复度最高** —— demo 就在旁边逐步回放同样这三个任务。
**建议不放海报**，或只留一行。
⚠️ task 17 的截图是 READ 的结果页，而 READ 在这题上是 ✗（BOTH 的 artifacts 已清）。
图里那台车 $900 但把手不是红的——**这恰好说明纯文字 agent 匹配了价格却验证不了颜色**，
是个好细节，但必须写进图注，否则就是自相矛盾。

### 候选 C — agent 到底看见什么 `strip_C_page_variety.png`

六张不同类型的真实页面：首页 / 分类列表 / 我的清单 / 商品页 / 空白发布表单 /
**agent 走到了另一个站**（`localhost:7770`，shopping）。

回答的是"你们到底跑了什么"，适合放方法栏、小尺寸。最后那张是白捡的：一个 classifieds
任务的 agent 溜达到了 shopping 站去搜 "gingerbread man pillow"。

> ⚠️ 标签必须对着 **截图内容 + 该 step 的 `obs_url`** 双重核对。
> 第一版把搜索结果页标成 "category dropdown open"、把 shopping 页当成 classifieds，
> 都是只看文件名想当然的结果。

### 我的建议

- **A 放海报**（核心证据带，横跨一栏或全宽）
- **C 放海报**（方法栏，小尺寸，6 格压成 3 格也行）
- **B 让给 laptop demo**，海报上不重复

---

## 10. v9 落地了什么（2026-09-03）

产物：`poster_v9_jiaming_wei.{pptx,pdf}` · 脚本 `build_poster_v9.py` + `poster_figures_v9.py`。
**v8 整套保留不动**（`build_poster.py` / `poster_jiaming_wei.pdf`），v9 是另起的一套。

**版式**（A1 竖版，organiser 模板，未改尺寸 — user 2026-09-03 确认「竖版的必须」）：

| 位置 | 内容 | 出处 |
|---|---|---|
| 标题 / 副标题条 | 论文题目 + 副标题（不再是结论句） | `main.tex` |
| 顶部通栏 | A0 三段系统图，**全宽** | `representation_routing_merged_three_sections.svg` |
| 中部通栏 | 两看法截图带：同一任务，READ 解开 / LOOK 放弃 | replicate artifacts |
| 左栏 | 韦恩图（解不同的题）+ 行为极值表 | P3 · REALM Table 5 |
| 中栏 | 6 看法 × 8 设置成功率矩阵 + 失败不对称表 | F5 · REALM Table 41 |
| 右栏 | 空胜区 + 重跑噪声一句 + SO WHAT | F13 |

**去掉的**：大数字条（user 判「很low」）· 翻面率图（改一句话）· 碳排放（user 2026-09-03 判不用）。

### 三个当场踩到的坑

**① 截图裁得太狠 = 把唯一的信息维度切掉了。**
第一版统一裁「页面上半 50%」，六帧全变成同一条 OsClass 导航栏。截图带要传达的**不是页面上的文字**
（那个尺寸没人读得清，也不需要），而是**页面形状变没变**。改成保留 75%，形状立刻回来。
⇒ 规则：**缩略图的裁切要按"这一维是不是还在"判断，不是按"能不能读清字"判断。**

**② `obs_url` 与 `screenshot.png` 不同时序。**
URL 记的是动作**前**的页面，截图是动作**后**的状态。按 URL 挑出的「三个空表单」里有一个已经填好了。
改成**直接对截图做像素分组**才对上。⇒ artifact 的时序约定要实测，不能按字段名推断。

**③ 像素分组挖出比原图注更强的事实。**
把每步截图缩到 160×90 灰度、按平均绝对差 <2.0 分组：

| | 步数 | 见过几个不同页面 | 最大重复组 |
|---|---|---|---|
| READ | 12 | 8 | 4 帧 |
| LOOK | 26 | **9** | **7 帧像素级相同** |

海报上 LOOK 的后三帧取自那个 7 帧组，跨度 17 步。两个数字在 build 时现算，没有手写。

### 措辞纪律（REALM 自曝的坑，海报上没越线）

REALM `Table 5` 的 caption 自曝：「≥7/8 设置」这个门槛是 cell 数从 6 涨到 8 **之后**才定的
（83% → 87.5%），若用字面 ≥5/8 则 DOM+stext 会过两个指标，"四个纯文字 mode 全 0" 这个负面结论会崩。
⇒ 海报只写**「极值集中在能看见页面的两个 view」**，
**不写** "the text-only views are behaviourally inseparable" —— 后者需要成对等价区间检验，REALM 明说没做。

### 还没解决的

- F13 左上角 "Pareto-dominates…" 与绿区数据点轻微重叠（毕设原图带来的，缩到栏宽后更明显）
- 两张表 15–19pt，比正文小 —— 它们是「走近两米看」的内容；抓眼靠顶部系统图和截图带

### ④ 截图的时序语义 + 动作标签 = 第三种读法（user 2026-09-03 抓到）

`step 9 · scroll` 配一张页面顶部的截图，被读成**「滚了但页面没动」**。查数据：这一步的
scroll `action_success=True`、`page_changed=True`，而且 **LOOK 比 READ 滚得更多**
（10/26 步 vs 3/12 步）——「Vision 不会 scroll」不成立。

误读的来源是**两个不同时刻被并排放在一起**：截图是该步**开始时**的页面，动作标签是它
**接下来**要做的事。两者都对，拼在一起就多出一种字面读法。

修法不是解释，是**换掉要讲的那件事**：动作类型在这里根本不是重点，重点是
**它三次回到同一个已填好的编辑页**。所以重复帧的第二行改成红色的
`back on the step 3 page`，图注补一句 *"Each frame is the page that step **starts** on."*

⇒ 规则：**一张图旁边并排两个不同时刻的量，必然产生第三种读法**；要么统一时刻，
要么把标注换成不依赖时刻的事实。

---

## 11. v9.6 — 按修改单执行的一轮（2026-09-03）

user 给了一份逐条修改单（P0/P1/P2 + 施工顺序 + 验收标准）。全部执行，无自由发挥。

### P0 · 结构

**六图从列优先改行优先**。原来三栏各两个，读者眼里会自动renumber 成 1-3-5 / 2-4-6；
改成两行各三个后编号与阅读顺序一致：

| 行 | | | |
|---|---|---|---|
| 1 | ① 六视图八设置 | ② 解不同的题 | ③ 行为不同 |
| 2 | ④ 失败方式不同 | ⑤ 那按任务选？没这么简单 | ⑥ 为什么 |

**⑤ 的标题从 `IT DOES NOT WORK` 改为 `NOT SO FAST.`** —— 前者会被读成「按任务选这件事本身无效」，
而 oracle 说的不是这个。这是这轮唯一的**科学语义修正**，其余都是版面。

### P1 · caption 大砍

每个 panel 的 caption 压到一行，方法学信息（dot 是什么、baseline 是什么、颜色是什么意思）
一律移出：要么在图内，要么留给看板时的对话。

- ① 全删（`SIX VIEWS, EIGHT SETTINGS` + 图内 legend 已经说完）
- ② `The sets overlap — but do not coincide.`
- ③ `Vision scrolls ~4× more.`
- ④ `One side dies of something you can name; the other never arrives.`（第二句删）
- ⑤ `A win lands in the shaded region. None does.`
- ⑥ `More routing upside, less usable training signal.`

省出的空间**定向给 ⑤ 和 ⑥**（各放大约 20% 和 39%），①的热力图反过来压 12%。

### P1 · READ / LOOK 区

面积不动，只做三件事：`solved` / `gave up` 独占一行、字号 15.5→20pt；红字的
`step 1's page again` / `step 3's page again` 保留；图注压成一句。

⚠️ 改这里踩了一次：`8 different pages` 在 40mm 标签栏里换行，把下面的 verdict 顶重叠了。
标签栏的每一行都必须**在 40mm 内单行**，改成 `8 pages seen` 并加了 line_count 断言式验证。

### P2 · subtitle

`Measuring the ceiling, the predictability and the economics of representation routing in web agents`
→ `Testing when richer web-agent representations help — and whether their value can be predicted cheaply`

三个学术词各自换成路人能读的说法（ceiling→help · predictability→can be predicted · economics→cheaply），
科学含义不变。

### 验收（user 指定的三条）

- **1m**：六个 panel 的图形语义都成立——⑤ 的绿区空着、⑥ 的实心/空心点、④ 的橙蓝条长短、
  ③ 的 Vision 蓝点分离、① 的颜色分布、② 的圆交叠。caption 勉强可读。
- **3m**：结构与配色可辨，文字糊——海报的正常表现。
- **空白**：① 无 caption 但图更高，底边与 ②③ 的 caption 基本齐平，没有留下缺口。

> user 的收尾判断记在这里：**这轮之后停止「设计」**，接下来只做真实 A1 尺寸的可读性校验，
> 否则会从优化滑进过度打磨。

### 追加：截图带定为三帧（user 2026-09-03）

| | 每帧宽 | 每帧高 | 带子总高 |
|---|---|---|---|
| 4 帧 | 125.6mm | 43.3mm | 122.7mm |
| **3 帧** | **168.4mm (+34%)** | 54.5mm | 152.5mm (+30mm) |

**换的理由不是精简，是配对方式变了**：四帧时 LOOK 是两组配对（1=18、3=20），读者要比对两次；
三帧后首帧与末帧是同一页，绕圈一眼可见、不需要比对。

**最终的三帧是一个受控对照**（user 2026-09-03 提的结构，数据正好支持）：

| 帧 | READ | LOOK | 实测 |
|---|---|---|---|
| 1 | step 2 | step 2 | **两条 lane 像素级同一张图**（diff 0.00）——两个 run 到 step 2 为止完全一致，step 3 才分岔 |
| 2 | step 5 填表 | step 13 滚到传图区 | diff 12.96，明显不同 |
| 3 | step 11 成功（绿条 + 85.50 $）| step 19 | 与 step 2 **diff 0.00** —— 回到了第一帧 |

> ⚠️ 第二帧是**按实测差异**选的，不是按步号：LOOK 的 step 3 与 READ 的 step 5 差异只有
> **0.16**（同一个表单），放上去会读成重复帧。**并排的两格必须先量过差异再放。**

30mm 的代价分摊：截图裁切 0.62→0.58 · Venn 缩到 0.86（**行 1 的高度由 Venn 决定，
不是由热力图决定——先压热力图是白压**）· panel 5 缩到 0.78 · 帧标注区 18→16mm。
热力图反而从 0.50 放回 0.56。
