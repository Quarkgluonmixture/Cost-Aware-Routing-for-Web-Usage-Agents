---
type: conclusions
batch: E2
status: done
created: 2026-10-06
source: E2.jsonl（DATA 67 条，§450.1 → §527.7）
---

# 数据资产清单（E2 批：67 条 DATA，§450.1–§527.7）

**读法**：这一批的 67 条 DATA 覆盖 2026-08-10 → 09-15，项目从「跑实验」转向「交付」：
毕设（08-10 起图、08-11 全稿 v1、09-08 提交）、VLM4RWD workshop 投稿（08-21）、REALM 评审（09-09）、
09-16 Holistic AI × UCL CDI showcase（海报、demo、演讲）。实验侧只有几块新数据：B5（GPT-5.6 terra）cls 六 mode、
reddit 五格 replicate、step-0 特征表、一组 post-hoc 可学性分析，以及三份发车前声明（B5 reddit、预算路由 prospective、本地 replicate chain）。
**先看第一节**：盘上状态是 2026-10-06 按 `docs/analysis/run_inventory/run_inventory.json` 与直接 `test -e` 实测的；
台账里的 `artifact_exists` 是抽取时的值，和现在不一定一致。

盘上状态记号：**在** = 台账路径在本 checkout 实测存在（`results/` 下的 junction 指向 `E:\p79-runs\`）；
**在（仅 E:）** = 本 checkout 没有，但 `E:\p79-runs\` 或 DGX 最终镜像 `E:\dgx-jiaming-backup\workspace\Cost-Aware-Routing-for-Web-Usage-Agents\`（2026-09-18 快照）里有；
**不在** = 都找不到；**无法判断** = 在外部系统（Overleaf / OpenReview / 日历 / 已停用主机），本次没查。

---

## 一、盘上状态与台账不一致的（引用前必读）

| 什么 | 台账路径 | § | 2026-10-06 实测 |
|---|---|---|---|
| 毕设 F0 / F2 / F3 / F8 四张图 | `final_dissertation/figures/fig_f0_thesis_overview.*`、`fig_f2_literature_map.*`、`fig_f3_comparison_boundary.*`、`fig_f8_oracle_ceiling.*` | §450.11 §450.13 §450.16 | **不在**。生成脚本已挪进 `scripts/analysis/figures/thesis/_retired/`；figures 目录里现在是 `fig_overview.pdf` 与 `fig_ceilings.pdf`（由 `scripts/analysis/figures/fig_ceilings.py` 生成，被 `ch1_introduction.tex` / `ch4_representation.tex` 引用）。台账没有记录退役发生在哪一节 |
| 毕设 Reader's Guide | `final_dissertation/tex/frontmatter.tex` | §474.5 | **原路径不在**。§481 迁 UCL 模板后，`tex/readers_guide.tex` 文件头注释写着 guide 从旧 front matter 拆成独立文件。内容是否逐字一致：未比对 |
| 毕设编译产物 | `final_dissertation/tex/main.pdf` | §481 | **在（仅 E:）**：本 checkout 只有 `main.bcf` / `main.run.xml`，PDF 只在 DGX 镜像里。是不是 09-08 提交版：无法判断 |
| codex / gemini 审计产物 | `docs/checkpoints/codex_outputs/`、`codex_prompts/`、`gemini_outputs/` | §462 §474.5 §474.8 §506.10 | **在（仅 E:）**。这几个目录在 `.gitignore` 里，本 checkout 没有，DGX 镜像里各文件实测都在 |
| B5 cls cell 笔记 | `docs/checkpoints/_status/cells/cell_b5_cls_*.md` | §509.1 | **在（仅 E:）**（gitignored，仅 DGX 镜像） |
| 唯一的 SoM 编号标注图 | `docs/checkpoints/周报/weekly-dashboard/public/figures/mode_som_annotated.png` | §450.15 | **在（仅 E:）**（`周报/` 整目录 gitignored） |
| 观测 mirror | `results/mechanistic/_obs_mirror` | §450.15 | **在（仅 E:）**：`results/mechanistic` 是实目录不是 junction，`_obs_mirror` 只在 `E:\p79-runs\mechanistic\_obs_mirror`（README 警告：路径超 Windows MAX_PATH，要用 `\\?\`） |
| two-arm 可学性 JSON | `results/phantom_paper/two_arm_action_learnability.json` | §490.5 | **在（仅 E:）**：`E:\p79-runs\phantom_paper\` 下有，仓库 `results/phantom_paper/` 只放行 3 个白名单文件 |
| lookahead 三个 producer | `scripts/{extract_step0.py,lookahead_eval.py,bandit_replay.py}` | §505 | **在（仅 E:）**：位置是 `E:\p79-runs\router_llm_pilot_20260909\scripts\`，不在仓库 `scripts/` 下 |
| 演讲版录屏 | `deliverables/showcase/demo/talk_130.webm` | §507.4 | **原路径不在**；现位于 `deliverables/showcase/talk/talk_130.webm` |
| 演讲离线包 | `tmp/showcase_talk_bundle.zip` | §511.6 §512.1 §513.4 | **在（仅 E:）**，但 DGX 镜像是 09-18 快照，原文说「改片子或台本后须重打」⇒ 是哪一版无法判断 |
| corpus EDA 数据 | `corpus_eda.json`（台账未给目录） | §450.16 | 在：`docs/analysis/benchmark_eda/corpus_eda.json` |
| 匿名 repo 导出物 | scratchpad | §475.7 | **不在**（只剩导出脚本，导出物本来就没入库） |
| 碳排原始调研稿 | session scratchpad | §506.2 | **不在**（原文即「未入库」；入库的是 README 与 `CARBON` 常量） |
| REALM 评审 | OpenReview forum `EAplLx6gCD` | §504 | 台账 `artifact_exists=false`，但归档副本 `docs/checkpoints/_status/issues/issue_realm_reviews_2026-09-09.md` **在**；OpenReview 本身无法判断 |
| 海报 v1 / v4 / v5 / v8.2 | 都写到同一个 `poster_jiaming_wei.pdf` | §495–§499.8 | 文件**在**，但同名覆盖：盘上只有最后一次写入；之前各版只能去 git 历史找（未查） |

---

## 二、毕设：图、起手文件与全稿

### 2.1 起手文件与 13 张主文图（§450.x，2026-08-10）

| 数据 | 路径 | 能支撑 | 不能支撑 | 盘上 |
|---|---|---|---|---|
| FIGURE_PLAN.md + GPT_SEARCH_PROMPTS.md | `final_dissertation/` | 图↔章问题↔claim↔数据源的绑定；割序已预定 (F5 退表 → F15 并入 F14 → F7 并入 F6)，六张不可割 F0/F3/F8/F10/F13/F14 | grade = planning artifact（Stage B 待 handbook） | 在 |
| 五个 GPT 搜索结果 + 核验层 | `final_dissertation/search_results/*.md` (167KB) + `VERIFICATION.md` | 引文已过 arXiv API + crossref 核验 (33 arXiv + 27 DOI)；F2 文献图谱 (27 ID 全 clean)、F10b/F12 方法学引文 | 台账未给 | 在 |
| F0 + F3 示意图 | `figures/fig_f0_*`、`fig_f3_*` | paper-grade schematic，零数据依赖 | — | **不在**（见第一节） |
| F8 + F10b | `figures/fig_f8_oracle_ceiling.*` + `fig_f10b_one_arm_margin.*` | C1 headline + 必带 rerun 基线；数字从 `noise_floor_inventory.md` 正则解析 | — | F10b 在；F8 **不在** |
| F1 motivating example + 观测素材全盘点 | `figures/fig_f1_motivating_example.*`；素材见第一节 | 真实 artifact 非示意图，同页面经 md5 验证；Ch1 + Ch3 phantom 构造的铺垫 | — | F1 在；素材部分仅 E: |
| F2 文献图谱 + F4 六语料 EDA | `figures/fig_f2_*` + `fig_f4_corpus_eda.*` | F2 素材 = P2 搜索 27 篇 (arXiv API 全核, 0 mismatch)；F4 数据 = `corpus_eda.json` (08-09) | — | F4 在；F2 **不在** |
| F11 哑铃图 + F12 置换零分布 | `figures/fig_f11_*` + `fig_f12_*` | F11 反击攻击 A1『只是欠采样』；F12 = C4 的置换控制，Holm 按实际 cell 数独立重算 | — | 在 |
| 08-10 session 总产出 | 起手文件 + 13 张图 + 13 个脚本 | 图脚本一律从产物解析/重算，解析不全拒绝出图 | — | 部分在（4 张已退役） |

- **F8/F10b 的视觉结论**（§450.13 原文）：*"三个 cell 的 **5-arm** 增益 (4.91/4.43/3.45pp) 落在 **1-arm** 重跑带 (2.00–7.59pp) 内外缘"*。
  scope：5-arm 增益 vs 1-arm 重跑带，两者是不同对象，并列不相减。
- **caveat**：F0/F2/F3/F8 后来被退役（脚本进 `_retired/`），台账没写哪一节决定的。引 §450.11 / §450.13 的「图已落地」时，记住这四张图不在终稿里。

### 2.2 全稿编译产物的版本链

| § | 日期 | 产物 | 页数与状态（原文） |
|---|---|---|---|
| §452 | 08-11 | 全稿 v1：7 章 + 4 附录 + 17 主文图 + 33 条核验 bib | **79 页**，0 undefined ref，0 overfull >20pt；DRAFT_v1，未过 supervisor / viva |
| §474.5 | 08-21 | 新增 Reader's Guide 2 页 + 两份 codex 冷读 | **84→87 页**，0 overfull / 0 undefined |
| §474.8 | 08-21 | 第三轮冷读：四篇附录 (116KB, 201 条 typed) | cold-read audit |
| §475.8-adjacent | 08-22 | Overleaf 同步到 `e596b94` | **89 页**，0 undefined；12 文件 +898/-509 |
| §476.5 | 08-22 | 四项修正 (11 处) + `rerun_union_extrapolation.{md,json}` | **90 页(+1)**，0 undefined，0 overfull |
| §477.5 | 08-22 | 四项修正 + 三家审计 20 条 + 阈值对口重构 | **90 页** / 0 undefined / 0 overfull |
| §481 | 08-27 | 迁 UCL PhD Thesis Template 后的 `main.pdf` | 0 error / 0 undefined / 8 overfull (最大 10.2pt) / **111 页** / 16 图 / 20 表 / 36 条参考文献 |
| §502.1 | 09-08 | Research Paper Declaration Form | **submitted (2026-09-08 16:00 随毕设提交)** |

- **当前值**：终稿是 §481 模板版，09-08 随 §502.1 一起提交。§481 *"可支持「排版层面 submission-ready」，**不支持**任何内容层面的结论"*。
- **能支撑**：§452 是 09-01 硬截止的交付物；§502.1 支撑 *"毕设披露了与 REALM #192 / VLM4RWD 的成果重合 + 三张复用图 + 四作者贡献"*（措辞逐字取自官方表）。
  §476.5 / §477.5：PDF 里每个生成数字都能追溯到 JSON（§477.5：*"六臂 × 8 字段 + 4 条 band + headroom 三元组**逐个可在 PDF 中追溯**"*）。
- **不能支撑**：§452 *"**不能**当 camera-ready: 页数预算仍等 COMP0191 handbook"*。
- **已作废**：§475.8-adjacent 原文 *"同日 MEASURED『落后 10 天』条已作废, 勿再据其行动"*（被同一节的 sync 推翻）。
  §476.5 改掉的三处错误表述（'upper bound on repetition' / 'no profitable' / 'necessarily also spends'）；§477.5 指出 MD renderer 与 JSON `model` 字段 *"此前仍在逐字发已撤回的结论(『Reported as an upper bound』), §476 只改了 docstring 没改渲染器"*，§477.5 从生成脚本根上改掉。
- **caveats**（原文）：
  - §476.5：*"⚠️ **这 11 处改写本身没有过审计**"*；页数 90/100，余量收窄到 10 页。§477.5 随后把三家审计的 20 条落地，且 *"**20 条里 9 条是当天改写引入的**, 再次坐实 §474.9"*。
  - §477.5：*"**`final_dissertation/tex/figures/` 是 `figures/` 的手工副本, 无同步脚本**"*，18 张图冻在 08-11 09:38。
  - §475.8-adjacent：*"**同步是手动触发的, 没有任何机制提醒它落后**"*；*"**『同步了』≠『导师知道该看了』**"*。
  - §474.8：附录 C 的 step 4 缺目标函数（留作者）；D 因表格被 detex 省略而无法完整判断。
- **盘上**：`tex/` 源在；`main.pdf` 仅 E:；`frontmatter.tex` 已拆成 `readers_guide.tex`；Overleaf `6a7a7331d2e6523a360245d4` 无法判断；`rerun_union_extrapolation.{py,json}` 在；`research_paper_declaration.tex` 在。

---

## 三、Router 负结果的对照与可学性分析（post-hoc，均非 gate）

全部在 `docs/analysis/cross_sites/`，盘上实测**在**（§490.5 的 JSON 除外，见第一节）。共同点：复用既有 CV / estimator / label-shuffle null 基础设施，**不是 gating producer**。

| § | 数据 | 产物 | 能支撑 | 不能支撑（原文） |
|---|---|---|---|---|
| §453 | 欠采样对照（学习曲线 + in-sample 可分性 + oracle 上界 + supply 定价）+ F16 | `router_undersampling_control.{md,json}`；`post_hoc_exploratory=True, h10_eligible=False` | 关闭已知攻击 A1；毕设 Ch5 §5.7 | *"**不能**支持『更大 benchmark 其他都不变』—— 只证了『更好的分类器不改变结论』"* |
| §455 | retry-vs-switch label supply 全套 + 1034 行 per-task CSV | `retry_vs_switch_label_supply.{md,json}` + `retry_vs_switch_per_task.csv` | §455.1–.3 全部数字；per-task CSV 可直接当「retry 还是 switch」分类器的标签基质 | phantom 臂 (无重跑) / 其他 cell / 立即 retry 的收益 (这些重跑含环境漂移是上界) |
| §457 | abstention 可学性（label 供给 / held-out AUROC + null / 5 档损失容忍度帕累托） | `abstention_learnability.{md,json}` | §457.1-.3 全部数字；「benchmark 供得上哪类 routing 标签」 | WA cell / phantom 臂单独的结论 / rerun-calibrated 的 universal-fail 定义 / B2 两格的省钱数字 (无实质 AUROC 信号) |
| §458 | 表征部署画像（35 格可诊断性 + B0 两站六 mode per-step token 尾部） | `representation_deployment_profile.{md,json}` | §458.1/.2 全部数字；「表征的部署属性与 SR 正交」 | B1/B2 的 token 尾部 / 跨站外推尾部 / 把可诊断性当自然律 (是本 ruleset 的下界) |
| §459 | early-abort 可学性（6 mode × 3 个 k） | `early_abort_B0_classifieds.{md,json}` | §459 全部数字；「label 充足但信号缺失」这一失效模式 | 其他 cell/backbone (只跑 B0×cls) / 全体 episode 的判别力 (是 survive-to-k 子集) / 把 matched-loss 那栏当部署建议 |
| §490.5 | two-arm action label 可学性 | 脚本 + `two_arm_action_learnability.md` + JSON | *"oracle_triage 不是天花板的构造性证明(8/8, 稳)"*；路由两半失败的机制解释（正类稀缺，不是 target 选错） | *"任何'z 比 y 更好学'或'学出来的策略 Pareto-beat always-cheapest'的说法 —— 两者数据都不支持"* |

- **caveats**：§458 的 token 尾部 *"无 tokens.input 的 mode 跳过而非报 0"*；§459 特征严格只取前 k 步；§455 缺 task 直接 fail-loud。
- 这些 producer 的数字本身在台账别的类型里（MEASURED），本文件只登记产物。

---

## 四、新 backbone 的数据：B4、B5

### 4.1 B4 = Claude Sonnet 5（§461，08-13）

- **是什么**：6 个 canonical cls config + 2 个 1-task smoke config + 4 个 queue 脚本的 B4 支持 + 自动开火 launcher/watcher。grade = deployed-pending-smoke。
- **能支撑**：B4 cls 6-condition fire 的发车；WA-shop B0 dom gate。
- **不能支撑**（原文）：*"在 smoke 通过前引用任何 B4 数字; B4 × reddit (config 未写)"*。
- **盘上**：config 与脚本**在**。`run_inventory.json` 里 B4 只有两个 smoke run（`B4_dom_classifieds_smoke_20260816_…_R31716`、`B4_som_classifieds_smoke_20260816_…_R30439`，各 1 ep），**没有任何 B4 canonical cls run**。

### 4.2 B5 = GPT-5.6 terra × classifieds（§492.1 / §509.1 / §509.2）

- **当前值**（§492.1，n=224 = classifieds 完整 scored universe）：
  SoM 37.05% / DOM 25.00%(A2) 23.66%(A1) / P-text 24.11% / P-SoM 22.77% / P-prompt 21.88% / Vision 12.05%。
- **登记**（§509.1）：五个非 vision condition 进 `run_manifest.yaml` extension + `failure_modes_per_cell.json` extension_cells + cell 笔记。grade = *"paper-grade 数据, 预注册集合之外 (extension)"*。
- **diag**（§509.2）：五份 per-condition diag digest + 跨 mode 汇总，在 `docs/analysis/vwa_classifieds/B5_*`。
- **能支撑**：NAACL 攻击面 #3「baseline 够不够强」的跨 backbone 表征比较；B5 与 B0/B1/B2 的失败构成并表（同 taxonomy）。
  A1/A2 是同格 replicate 对 (Δ=1.34pp, 落在 rerun band 0.89-2.23pp 内)，08-21 commit 24a573a 注册进 CLEAN_PAIRS。
- **不能支撑**：*"任何 pooled / K-of-N 统计"*（§509.1）。
- **caveats**（原文）：
  - §492.1：*"⚠️ 这 8 个 run 全部在 validate_fire_manifest 的扫描集之外 … 不受 ghost/双跑保护; vision 另有一个 08-26 的 135/224 半截 run, 非 replicate, 勿混入"*
  - §509.1：*"vision 不在内 (B-1997); dom canonical = R29736, cell 笔记 last_run_id 显示 cron 选的 replicate R15476"*
  - §509.2：*"P31 行是下界 (B-1999); P10 在 B5 上以误报为主 (B-2000 + 语义混比); P33 在 som 上非死因"*
  - scope 提醒：§492.1 的 rerun band 0.89-2.23pp 是 B5 cls 同格 A1/A2 的口径；和 §450.13 的 1-arm 重跑带 2.00–7.59pp 不是一个对象，**不能互换、不能相减**。
- **盘上**：`results/visualwebarena/phase1/B5_*_classifieds_*` 全**在**。inventory 记录：dom R29736（`extension:paper-grade`，`B5.cls.dom:canonical`）+ R15476（`B5.cls.dom:replicate`）、som R31483、ptext R4968、pprompt R10294、psom R18439（均 `extension:paper-grade`）、vision R24364（224 ep，**未登记**）、半截 vision R16160（inventory 记 n=136，completeness 0.6071；台账写的是 135/224，两边差 1，未核）、另有 4 个 dom smoke（各 1 ep）。diag digest 在；cell 笔记仅 E:。

### 4.3 B5 × reddit 发车前声明（§478.7，08-26）

- **是什么**：B5 × reddit × {dom, som, vision} chain 的 launch intent，power 预先声明 dom d≈24 / som d≈15 / vision d≈8.7 (inventory-only 但仍然跑)，含价格漂移披露与 live-quota gate 设计。
- **能支撑**：B5 跨站结论的 pre-registration；*"三个 cell 无论数字如何都须报告"*。
- **盘上**：intent、queue 脚本、config **在**；`run_inventory.json` 里**没有任何 B5 reddit run**。§515.4 的 launch intent 文件（非台账）写着代理余额实测 $30.18、付费项全部被挡，可作旁证。

---

## 五、Replicate 与噪声地板

### 5.1 五格 reddit replicate 结算（§500，09-06）

- **产物**：`unique_solve_envelope_cross_cell.md`（新）+ `noise_floor_inventory.{md,json}` + `serving_mode_floor.{md,json}` + `unique_solve_noise_envelope.py` 的 `--cell/--compare`；CLEAN_PAIRS **13→18 行**。
- **grade**：*"paper-grade: 五格均 205 ep 完整, 均按各自 launch intent as-declared 注册 (§469.7); 回归验证 cls_b0 默认输出逐字不变; pytest HEAD 25 failed → 24, 新引入 0"*。
- **能支撑**：C1 的跨站版本（含其 inventory-only 限定）；hero 措辞从『视觉侧 vs 文本侧』收缩到『SoM 单臂』；Reading 2 关闭 §478.4。
- **盘上**：全部**在**。`run_inventory/README.md` 也记 18 个 `CLEAN_PAIRS` replicate（2026-10-06），与 §500 的 18 行一致。

### 5.2 本地 replicate chain 发车前声明（§515.4，09-15）

- **是什么**：10 格落点表、C1 证伪判据、B2 结构上限、逐对代码漂移、停机条件、事前预测；commit b805f79 早于 09:29:04 UTC 发车。
- **能支撑**：落地时按事前声明读数与登记 CLEAN_PAIRS（登记留给 user）；证明 B2 cls 两格的「证伪不了 C1」是事前判断。
- **盘上**：intent 文件**在**；chain 日志在 A100（无法判断）。按 `run_inventory.json` 对照 intent 文件里的 L1–L10：
  - L1–L6（B1 × WA-reddit × 6 mode）都已落地，104 ep 完整：dom R3792（09-15 09:29:04，和 intent 的发车时刻一致）、phantom_prompt R28726、phantom_text R31888、som R20276（09-17 那次 R19939 卡死后只剩 26 ep，另存为 `_archive_hung_…`）、phantom_som R122、vision R18252。registry 一栏全是 None，**还没登记**。
  - L7（B1 × VWA-shop × som）：`B1_som_shopping_20260923_…_R22515` 只有 93 ep（completeness 0.2138），拉取时还在跑。
  - L8–L10（B2 三格）：inventory 里**没有对应的 09 月 run**。

---

## 六、Step-0 特征表与预算路由 prospective

| § | 数据 | 路径 | 能支撑 | 不能支撑 | 盘上 |
|---|---|---|---|---|---|
| §505 | step-0 特征表 (21,291 episode = 11 cell canonical + 18 对 replicate 臂) + lookahead/bandit/3-arm 原始输出 + 三个 producer | `results/router_llm_pilot_20260909/lookahead/…`；tracked digest `docs/analysis/cross_sites/one_step_lookahead_2026-09-09.md` | §505 全部数字的复算；后续以 step-0 记录为输入的 router 分析不必再扫 18k 文件 | grade = exploratory：*"非 paper-grade producer, 未进 make analysis"* | 数据与 digest **在**；三个脚本只在 `E:\p79-runs\router_llm_pilot_20260909\scripts\` |
| §505.28 | 预算路由对 shop_B1 held-out 臂的冻结预测（434 题难度分 + two_tier / three_tier，seen_ptext_ids (216)，预先声明判据）+ 评估脚本 | `docs/checkpoints/pre_run/budget_router_prospective_shop_B1_20260909.{json,md}` + `budget_router_prospective_eval.py`；git tag `budget-router-prospective-20260909` | 预算路由的 prospective 方向检验（two_tier 在 PRIMARY 集上 SR 损失 < 同成本固定 cap 且 < 随机分档） | *"幅度结论 (shop_B1 无 band) 或对 216 条已落地 P-text 的 prospective 主张"* | **在** |

- §505 的一致性检查（原文）：*"canonical 臂选择规则与 pilot 一致, 13,523/13,523 与 full_table2 逐 task 一致"*。
- §505.28 的 held-out 臂按日期推断对应 `B1_phantom_text_shopping_20260908`、`B1_phantom_prompt_shopping_20260911`（对应关系是本文件按 run 名推断的，台账没给 run id）；两者在 inventory 里都是 435 ep 完整，所以 prospective 检验需要的数据在盘上。检验有没有跑、结果如何，不在本批台账里。

---

## 七、探针与估算（probe-grade / 展示用）

| § | 数据 | 路径 | 能支撑 | 不能支撑（原文） | 盘上 |
|---|---|---|---|---|---|
| §456 | 图像通道 + 实测计费探针（九候选 × 文本/图像双调用）+ 08-12 registry 快照 | `docs/checkpoints/probes/proxy_vision_channel_20260812_221840.json` + `proxy_model_registry_20260812_221116.json`；producer `probe_proxy_vision_channel.py`（`--vision` 才计费） | 哪些 proxy model 能吃图；各家 image token 计量差异；真实单价 (usage.cost 反推) | 任何 VWA 任务表现推断 / SoM 密集标注图的表现 / 1280×720 真截图的 token 量（grade：单张合成图） | 在 |
| §506.2 | Qwen3-VL-235B-A22B 每 token 能耗区间 + 设施系数 + 英国电网强度出处（含「什么会让它错 3 倍以上」）；arXiv 2505.06371 / 2508.15734 / 2310.03003 / 2504.17674 已用 API 核实 | `deliverables/showcase/demo/README.md` 'The CO2e row' + `build_demo_data.py` `CARBON` | demo 上的「≈ 区间」与口头回答 | *"论文里任何碳排数字或『X% token 节省 = X% 碳排节省』(毕设可持续性一节立场不变)"* | README 与脚本在；原始调研稿不在 |

- §506.2 grade 原文：*"J_out 为本模型实测 (ML.ENERGY v3), J_in 上限为判断, 电网为 DESNZ 2026 官方系数"*。

---

## 八、跨 AI 审计与 framing 输入

| § | 数据 | 路径 | 能支撑 | 不能支撑 | 盘上 |
|---|---|---|---|---|---|
| §462 | codex xhigh 零预设 framing 提案全文（335,690 tokens, gpt-5.6-sol xhigh） | `docs/checkpoints/codex_outputs/frame_zero_preset_2026-08-13.md` + prompt | NAACL framing 候选；载重/降级的第三方判断；*"两处已证实的方法学缺陷 (§462.1 leak 政策 / §462.2 阈值选择偏倚)"* | *"其中的 arXiv ID (未核) / 它对未读产物的推断"* | 仅 E: |
| §474.5 / §474.8 | 毕设三轮 codex 零预设冷读（190KB + 218KB；附录轮 116KB, 201 条 typed） | `codex_outputs/thesis_outsider_readability{,_B,_C}_2026-08-21.md` | 术语就地解码；附录级可读性缺口清单。§474.5：*"术语表含 `arm` —— 全稿 20 次却从未定义, 而 arm-count-matched 是 ch4 的核心论证"* | 台账未给 | 仅 E: |
| §480 | NAACL reframe 的 framing 输入：三条新解锁主张 (C1 serving-path 地板 / C2 第二站点 / C3 baseline 强度) + 三条已为假的现有陈述 + 七攻击面对照 + 四个缺口的可买性分类 | `docs/checkpoints/paper_drafts/naacl_evidence_delta.md` | 09-07 意见到手后选 claim 的原材料 | grade 原文 *"framing input (not a frame)"*；*"明确不替代该选择"* | 在 |
| §506.10 | showcase demo v2 /stress spot-check：Mode A (Claude) 4 条 + Mode C (agy) 3 条；Mode B (codex) 因 OpenAI 用量上限 6 s 退出、无输出 | `docs/checkpoints/gemini_outputs/showcase_demo_v2_2026-09-10_182221.md` + codex prompts | demo 对外说法的修改清单 | *"**不可**当作 live 系统层已被独立审过的证据"*（codex 负责的 live server / run_lane / site-compose 系统层未经第二家审） | 仅 E: |

- §462 grade：零预设，preflight 14/14 路径存在，post-flight 三项通过。

---

## 九、对外投稿、评审与开源准备

| § | 数据 | 路径 | 状态 / 能支撑 | 不能支撑 | 盘上 |
|---|---|---|---|---|---|
| §473 → §473.7 | VLM4RWD @ NeurIPS 2026 workshop 投稿件 | `deliverables/vlm4rwd/`（`main.pdf` = 提交件） | §473 submission-ready → §473.7 **submitted 2026-08-21**（non-archival）；正文 8 页 / 全文 39 页 / 0 error / 0 undefined / 61 caption 全完整 / 匿名 0 泄露；notif **2026-09-29**，camera-ready 2026-10 | 不能当 REALM 在审稿的替代或更新：二者已分叉（本副本修了 tab02/paper-B 残留/Multiplicity 三处原稿 bug，REALM 那边未修，且 intro 多一段 workshop 对位）。源真相仍在 Overleaf `6a59017b04233a73ed5ec570` | 在（Overleaf 无法判断） |
| §504 | REALM #192 三份 official review 全文 + camera-ready 表单现状（Archival=Non-archival / Cross Submission=ACL ARR 2026 August） | OpenReview `EAplLx6gCD`；归档副本 `docs/checkpoints/_status/issues/issue_realm_reviews_2026-09-09.md` | camera-ready 逐条对位（三条交集意见 §504.2）+ ARR reframe 攻击面清单；grade = external-authoritative | *"任何关于 reviewer 身份或后续 venue 结果的推断"* | 归档副本在 |
| §475.7 | 匿名 submission repo 的 dry-run 导出物 + 导出器 | `scripts/maintenance/export_anonymous_repo.py` | 1201 文件 / 119M（含 248 张 benchmark 截图），单个 `Anonymous <anon@example.com>` commit，4692 处 scrub，独立身份探针 + 248 个二进制路径探针**双清** | grade = dry-run | 脚本在；导出物不在 |

- **§473.7 caveat**：*"⚠️ **非归档** ⇒ 不影响 NAACL 2027 ARR (10-12) 投稿权"*。
- **§475.7 caveat**（原文）：*"**尚未 push, 也不应现在 push** —— 内容取决于最终稿实际引用哪些脚本/表 … 兑现点是 ARR 10-12"*；排除项（`docs/analysis/` 179 文件、`scripts/maintenance/` 92 个 host-specific 脚本）按需回补，*"回补后须重跑双探针"*。

---

## 十、领域综述（外部二手）

- **§505.13 → §505.16**：§505.13 是 user 在对话里提供的 CU 领域综述（15 个系统的 perception/vision/routing 表 + 9 篇 arXiv，编号经 API 核实全部存在且标题吻合），当时 *"未落文件"*。
  §505.16 把它与第二份综述存成文件：`docs/literature/raw/2026-09-09-cu-landscape-survey-1-semantic-first.md` + `…-survey-2-router-placement.md`（盘上**在**）。⇒ **§505.13 的「未落文件」状态已被 §505.16 取代**，引用请指 §505.16 的文件。
- **能支撑**：NAACL related-work 候选清单与 framing（*"task 级 interface 路由是生产共识; 无显式 per-step cost-aware router"*）。
- **不能支撑**：§505.13 *"**不可**直接引用其转述的数字 (如 ComponentBench 34.2pp / OSWorld-Human 75–94%) 而不读原文"*；§505.16 *"不可支持『生产系统有 learned router』(综述自己否定)"*。
- **caveat**：grade = external, 二手（AI 综述）；arXiv 存在性已核，*"数字/GitHub issue/平台文档主张全部未核"*。

---

## 十一、Showcase 海报（09-16 Holistic AI × UCL CDI，A1）

### 11.1 版本链

| § | 版本 | 产物 | 要点（原文） |
|---|---|---|---|
| §495 | v1 | `deliverables/showcase/`：`poster_jiaming_wei.pdf` (A1 594x841mm, 1 页, 6 字体全嵌入, 主视觉 300dpi) + pptx + 管线 + 文案单一来源 | **不含** F16 欠采样对照（留作口头 defense），不含 mechanism 线 |
| §498.1 | v4 | 同名 PDF + `figures/poster_dominance_plane.png` + `poster_overview.png`[gitignored] | 7 字体全嵌，Fig 1 300dpi / Fig 2 350dpi；不支持任何超出 8 格 / 2–36% SR 体制的外推 |
| §499.3 | v5「Look, read, or both?」 | 同名 PDF + `figures/demo_strip.json` + `thumb_{130,76,17}.png` | 窄带数字由脚本从 episode summary 解析并 assert A/B 一致；不支持『某题天生视觉/文本』 |
| §499.5 | Fig 3 | `figures/poster_label_supply.png` | 解析自 `router_label_supply_diagnosis.md` 首表 (labels 97/53/55/24/16/15, trainable yes/no/yes/no/no/no) + `router_triage_learnability_with_wa.json`；无手抄数字 |
| §499.7 | v8.2 | 同名 PDF | 20.5pt body / 15pt captions / 36pt headline numbers；只改可读性，不扩 claim |
| §499.8 | v8.3 | 同名 PDF | 正文 17.65→23pt、图注 12.71→17pt、小标签 10.59→15.5pt、数字 32.48→42pt；版式与科学主张不变 |
| §499.10 | v9.6 | **新文件** `poster_v9_jiaming_wei.{pdf,pptx}` + `build_poster_v9.py` + `poster_figures_v9.py` | 六图行优先重排；措辞按 REALM Table 5 caption 的自曝限制与 panel 5 的语义修正校准 |
| §499.12 | **v9.10 定稿** | `poster_v9_jiaming_wei.{pdf,pptx}` + `print_test_tiles.py` + `figures/v9/` + `POSTER_ASSET_LIBRARY.md` | 所有 claim 与 REALM/毕设证据的对应关系记在 `POSTER_ASSET_LIBRARY.md` §11-§15 |

- **当前值**：定稿是 §499.12 的 `poster_v9_jiaming_wei.pdf`（v9.10）。`poster_jiaming_wei.pdf` 是 v1→v8.3 一路同名覆盖的旧线，盘上只剩最后一次写入。
- **已作废**：v1 / v4 / v5 / v8.2 / v8.3 作为「当前海报」都已被 v9 线取代（§499.10 起改用新文件名）。
- **能支撑**：现场展示与板前讲解；*"每个数字的 scope 与基线以 poster_content.md 为准"*。
- **不能支撑**：任何超出 `poster_content.md` 已登记 scope 的科学主张。§499.5 Fig 3：*"只覆盖 6 个 VWA 格, 不得与 8 格数字并列成同一口径; 不支持因果 (mechanism) 主张, 只支持 bottleneck"*。
- **盘上**：两个 PDF/pptx、`poster_content.md`、`figures/poster_label_supply.png`、`demo_strip.json`、`print_test_tiles.py` 均**在**。

### 11.2 素材库（§499.9）

- `deliverables/showcase/POSTER_ASSET_LIBRARY.md` + `candidates/` 六张候选图（盘上**在**）。六个目录 160+ 张图分级，含真实截图 7,082 张（两个 clean replicate）、44 张 ablation 表、三个截图带候选。
- grade 原文：*"planning-grade: 分级是主观可读性判断不是测量"*；截图带标签经截图内容 + 该 step 的 obs_url 双重核过（第一版把搜索结果页标成 dropdown、把 shopping 页当成 classifieds）。
- **不支持任何科学主张**。

---

## 十二、Showcase demo 与十分钟演讲

### 12.1 demo（§501 → §507.4）

| § | 产物 | 状态 / 能支撑 | 不能支撑 |
|---|---|---|---|
| §501 | 三栏同步 replay demo：`deliverables/showcase/demo/{index.html,build_demo_data.py,build_portable.py,README.md,data.js,data/,frames/}` + `demo_portable.html` | board-ready：三题成败经 canonical×replicate 双向核对一致；三栏对齐 0px 错位 | *"现象层展示, **不承担证据主张**"* |
| §507.4 | demo 第三版：learned-choice 红绿框改为对照其他栏上色、演讲模式 `?task=<id>&autoplay=0`、碳排/措辞四处修正、39 s 录屏 | headless Chromium 断言通过；`demo_portable.html` 重建 (11.7 MB)；`talk_130.webm` (1600×900, VP8) 作浏览器故障兜底 | 台账未给 |

- §507.4 落地的就是 §506.10 审查清单（P1 红绿框上色、碳排悬停去掉 'published'；P2 四条措辞）。
- **盘上**：demo 目录与 portable 在；webm 已从 `demo/` 挪到 `talk/`。

### 12.2 演讲 deck 与台本的版本链

| § | 版本 | 产物 | 状态（原文） |
|---|---|---|---|
| §507.5 | v0 | `deliverables/showcase/talk/`：8 张 deck、`rehearsal-script.md` (735 词)、RUNBOOK、fallback.html、talk.pdf (8 页)、`check_talk.py` | check_talk.py PASS；尚未 user 出声重写台本、模板换皮、彩排 |
| §511.2 | 套主办方模板 | `talk/index.html` + `talk.pdf` + `fig/tpl_*` + `Showcase-Speaker-Deck.pptx` | 五种分辨率截图人工看过；PDF 里第 2、8 张只有标题 |
| §511.6 | 离线打包 | `tmp/showcase_talk_bundle.zip` | 临时产物，不入库；*"改片子或台本后须重打"* |
| §512.1 | v1 | deck 10 张 + 备用、台本 v1 (824 词)、`fig/behaviour.png`、`fig/failure.png` | *"台本仍是模型初稿"* |
| §513.4 | v2 | `talk_figures.py` + `fig/talk_{behaviour,failure,hindsight,routing,label_supply}.png`、deck 10 页 + 参考页、台本 v2 (788 词) | *"台本仍是模型初稿"* |

- **当前值**：台账内最晚是 §513.4 的 v2；§516.2 的当天手册也写「台本 v2」。
- **已作废**：v0（8 张）、v1（10 张，824 词）的 deck/台本被 v2 取代；§512.1 的 `fig/behaviour.png` / `failure.png` 被 §513.4 的 `talk_*` 版取代（旧文件仍在 `fig/` 下）。

### 12.3 演讲用的单张图与真实抓取素材

| § | 数据 | 路径 | 能支撑 | 不能支撑（原文） |
|---|---|---|---|---|
| §517.2 | 选对看法的成功率 / 成本 / 用时 / CO₂e 逐设置表 | `talk/hindsight_efficiency.py` → `hindsight_efficiency.json` | hindsight 页数字来源；带两道自检（成本 = `router_objective_ordering` 的 oracle_sr_cost；逐步 token 加总 = total_tokens） | grade 原文 *"post-hoc 描述"* |
| §520.1 | 固定标签产出率双对数示意图 | `talk/fig/talk_scaling.png` | 低成功率下达到同一标签数量需要更多任务 | *"幂律拟合或具体外推倍数"*；*"数学关系示意，非实测或拟合"* |
| §521.2 | why 页：6 个 VWA 设置的能做对题率 vs 第二常见正确看法的例子数，训练分界线 12.5 | `talk/fig/talk_capability.png`（点来自 `router_undersampling_control.json` whichmode_scale） | *"6 个 VWA 设置里现在 2 个够训练、4 个不够"* | *"越线所需的成功率、越线后能赢过永远用最便宜、WebArena 两个设置"*；箭头为方向示意 |
| §524.2 | 开场 agents 页的真实抓取（Wikipedia CreateAccount 快照 + 截图 + 节选） | `talk/real_capture.py` + `talk/fig/real_wiki_createaccount_*` + `real_wiki_snapshot_excerpt.png` | Playwright MCP 给模型的是文本、截图型 agent 看的是一张图 | *"Claude 或 Astra 在该页的实际表现、两种输入的 token 成本比较、Astra 的具体接口"* |
| §527.2 | BOTH 块：同一会话截图 + 视口内 25 个可交互元素坐标 + 用本项目 `som.py` 画的编号框；紧凑快照 20 / 161 行 | `talk/fig/real_wiki_createaccount_boxes.json` · `…_som.png` · `real_wiki_snapshot_excerpt_compact.png` | 演讲里「BOTH = 截图 + 编号框」的直观印象 | *"不能说这是某个产品的界面，也不是 agent 做任务的记录"*；元素选择是我们的 selector，不是 VWA 无障碍树节点集 |
| §527.5 | browser-use 0.13.10 对同一页的真实高亮截图（48 个交互元素）+ 文本表示 + 裁剪版 + 同裁剪的 Playwright MCP 截图 | `talk/fig/browser_use_wiki_*` · `real_wiki_form_screenshot.png`；脚本 `talk/browser_use_capture.py` | *"browser-use 默认就把可点元素框起来编号"* | *"不能说它和 demo 里我们的青色框是同一个工具，只能说同一类做法"* |

- **caveats**：§524.2 / §527.2 / §527.5 都是单次抓取，*"页面会变"*。§527.2：19:00 BST 重抓时快照仍 161 行（ref 编号变了）、截图与 16:38 那次逐字节相同。
- **盘上**：以上全部**在**。

---

## 十三、Showcase 当天的运维状态（时点快照，不是证据）

| § | 数据 | 路径 | 能支撑 | 不能支撑 | 盘上 |
|---|---|---|---|---|---|
| §516.2 | 当天手册（只读汇总页） | `deliverables/showcase/day-of.html`（已发布为 artifact） | 09-15 晚准备与 09-16 当天查阅 | *"派生文件，事实变动先改源文件"*（源 = ROADMAP / RUNBOOK / demo README / 台本 v2 / SHOWCASE_PREP） | 在 |
| §526.1 | 09-16 Google 日历事件 16 个（标题前缀 `Showcase · `） | user 的 Google 主日历 | 当天提醒与时间线 | *"不是安排或台词的权威来源"*；描述冻在 09-15 的片子版本 | 无法判断（外部系统，未查） |
| §527.7 | live 后端前夜状态：classifieds 容器 23:43 重置，live server 在 tmux 会话 `showcase`，health ok，代理余额 $30.18 | DGX tmux `showcase`；容器 `p79live_cls_web` / `p79live_cls_db` | 「今晚准备 B」已完成 | grade 原文 *"运行状态快照，明早会变；余额多人共用"* | 不在（运行态；DGX 2026-09-18 起停用） |

---

## 对旧结论层的 supersede

本批 67 条 DATA **没有一条点名推翻 §397 及更早的结论**。批内的取代关系（供后续批次查）：

- §475.8-adjacent → 作废同日（08-22）MEASURED「Overleaf 落后 10 天」。
- §505.16 → 取代 §505.13 的「未落文件」状态（综述已存为 `docs/literature/raw/` 下两份文件）。
- §499.10 / §499.12 → 海报 v9 线取代 v1–v8.3 的 `poster_jiaming_wei.pdf`。
- §513.4 → 演讲 v2 取代 v0（§507.5）/ v1（§512.1）。
- §500 → hero 措辞从『视觉侧 vs 文本侧』收缩到『SoM 单臂』；Reading 2 关闭 §478.4。
- §481 → 模板版 `main.pdf`（111 页）取代 §452–§477.5 的旧版编译产物。

---

## 覆盖性闭合

本批 **67 条**，用到 **67 条**，未归入任何主题的 **0 条**。

| 主题 | § |
|---|---|
| 二、毕设 | §450.1 §450.9 §450.11 §450.13 §450.15 §450.16 §450.17 §450.19 · §452 §474.5 §474.8 §475.8-adjacent §476.5 §477.5 §481 §502.1 |
| 三、可学性分析 | §453 §455 §457 §458 §459 §490.5 |
| 四、B4 / B5 | §461 §478.7 §492.1 §509.1 §509.2 |
| 五、replicate | §500 §515.4 |
| 六、step-0 与 prospective | §505 §505.28 |
| 七、探针与估算 | §456 §506.2 |
| 八、跨 AI 审计 | §462 §480 §506.10（§474.5 / §474.8 的冷读产物也在此表引用） |
| 九、投稿与开源 | §473 §473.7 §504 §475.7 |
| 十、领域综述 | §505.13 §505.16 |
| 十一、海报 | §495 §498.1 §499.3 §499.5 §499.7 §499.8 §499.9 §499.10 §499.12 |
| 十二、demo 与演讲 | §501 §507.4 §507.5 §511.2 §511.6 §512.1 §513.4 §517.2 §520.1 §521.2 §524.2 §527.2 §527.5 |
| 十三、当天运维 | §516.2 §526.1 §527.7 |
