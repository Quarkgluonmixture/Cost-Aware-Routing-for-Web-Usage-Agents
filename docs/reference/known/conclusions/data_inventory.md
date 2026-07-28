# 数据资产清单（E 批：49 条 DATA，§3–§397.10）

Claude 主 session 逐条通读，2026-07-28。**先看第一节** —— 台账里有多条数据的路径
已经不在磁盘上，引用它们会得到空目录而不是报错。

---

## ⚠️ 一、已经不在磁盘上（引用前必读）

台账抽取时逐条 `test -e` 实测过。这些条目**在笔记里被当作证据引用过**，但数据已不存在：

| 什么 | 原路径 | § | 状态 |
|---|---|---|---|
| 旧 reddit run `B1_3mode_reddit_20260412`（13/387 ep 放弃）+ 替代 run `B1_3mode_reddit_20260413` + `B1_3mode_shopping_20260413` | `results/visualwebarena/phase1/` | §35 | **三个目录均已不存在**（2026-07-28 实测）。仅 `B1_3mode_classifieds_20260413` 还在 |
| Fire-day 污染 run（cls R3903 含 110 个 `error=asyncio` + steps=0 的 summary；red R12405） | `.../B0_dom_classifieds_20260518_..._R3903` | §228 | **已清理**。原文当时要求 re-fire 前 `rm -rf` 避免 stale-resume |
| B-991 之后、tool_choice fix 之前的全部 B0 数据（Fire-3/4/5/6 + R2987） | `results/visualwebarena/phase1/` | §243 §244 | **已从磁盘清除**。§251 另指出这批 B0 在 search/type 上被系统性压低 = **残废 baseline，对比不公平** |
| `label_supply_sweep` / `label_trainability` / `pooled_label_conflict` 三套 exploratory 产物 | scratch（未入 gating） | §383.4 | §395.1 记 **"五天后已全丢"**，已由 `router_label_supply_diagnosis.py` 重建为正式 producer |
| 笔记 §3–§56 指向的 `docs/analysis/classifieds/*.md` | — | §3-§56 | **指针全部失效**（commit 6101088 迁到 `vwa_classifieds/` 后归档）。正文现在 `docs/archive/analysis_pre_2026-05-15/` |

**另一类"存在但支撑不了任何东西"**：

- **§26 Pass-2 learned router 的 LR substrate** `results/phantom_paper/l1_router/` ——
  A2.10 时点磁盘上 **0/52 fold-aware artifact、0/6 B2 artifact**。
  *"Pass-1 fire 物理不受阻，但 Pass-2 额外需要 Pass-1 完成后跑 LR 训练管线
  （~12-24h GPU-time per cell × 6 cells）才能生成这 52 个 artifact"*
- **§14 标签错误文件** `layer_significance_20260509.md` —— 内容实际是 **Cell F+G** 数据
  却被脚本按 arg slot 标为 **Cell A/B**。canonical 是两个显式标注版本
  （`_cellab_cls_` / `_cellfg_reddit_`）。**什么都不要从 deprecated 文件引用**
- **§8 broken shopping 数据**（Magento base_url 配成 `metis.lti.cs.cmu.edu` 导致 guest 状态）——
  全部 cleared，*"只作历史 provenance trace"*

---

## 二、clean 且可用

### 噪声地板与 id 通道（Phase 0b 的直接输入）

| 数据 | 路径 | 能支撑 |
|---|---|---|
| **两个** clean replicate | `results/repro_replicates/B0_dom_classifieds_R31194_clean_replicate` + `B0_vision_classifieds_R24792_clean_replicate` | dom floor（id-churn + MoE **混合**）与 vision floor（≈ 纯 B0 serving/MoE，因 vision 输入是纯截图无 element_id） |
| B0/B1 配对 id-perturbation probe（5 份 json，含 flip-biased 早期版与无偏版） | `docs/checkpoints/probes/b0_paired_idperturb_20260529_*.json` | serving-floor bar（B0 13.3% / B1 0）+ **flip-overlap Venn（交集只 {60}）= cross-model subtraction 是 category error 的集合论级证据** |
| N=5 same-payload replay 双 provider（AWS Bedrock + DashScope 各 100 req） | `docs/checkpoints/probes/replay_step0_n5_*.json` | provider-dependent **双层** noise framing + cross-provider audit-artifact gap（`system_fingerprint=None` 100/100 双 provider） |
| codex cold-start 的 vision run-to-run 独立审计（1.9MB / 14872 行）+ 其 zero-preset prompt | `docs/checkpoints/codex_outputs/vision_moe_anomaly_2026-05-27.md` | 9 候选排序 + 全 224 step-0 recompute + **retract 1-2pp MoE residual 的原文措辞** |
| run-to-run noise session checkpoint | `docs/checkpoints/session_checkpoint_2026-05-25_runtorun_noise.md` | §292/§293/§294 的完整上下文；**§292 措辞错误的 canonical 纠正来源** |

⚠️ 两条 id-perturbation 相关条目都挂着 `named by RETRACTED §397.10`（§302 的线性减法作废）。
**数据本身有效，被作废的是拿它们做减法的那个论证。**

### Phase 1a canonical fire

- **§42** classifieds 18 个 paper-grade condition run（B0/B1/B2 × 6 mode）+ reddit 已 land 部分。
  准入以 `fire_manifest` bound + `run_manifest.yaml` promote 为准
- **§48** B2 reddit 六 mode **全部 205/205**（k=6 数据完备）⇒ 解锁三件事：
  "5 of 6" 免责声明整段删除 · B-1284 cross-family modifier 具备解除条件 ·
  Protocol Note 06 两轨制披露块可删
- **§34** 21 个 per-condition diag digest（cls 18 + reddit 3）。命名约定
  `<model>_<mode>_<site>_diag_digest.md`（**per-condition 非 per-cell**）

### Router 离线评估

**§43**：三套隔离产物 `l1_router_offline_20260715/` · `l1_router_sweep_20260715/` ·
`l1_router_rehearsal_20260702/` —— **全部大字标 OFFLINE / NON-GATE，不替代 live Pass-2/H10**。
**§44** B0 reddit 复现全套：*"cls 配方冻结跨格复现失败的负结果；router 论点的 transfer 边界"*

### 外部见证

**§30 OSF DOI 1**：`10.17605/OSF.IO/9QCWU`，GUID `9qcwu`，submitted **2026-05-18T23:10:06Z**
（OSF API `date_registered`；UI 显示的 May 19 12:10 AM 是 BST）。archive 即时完成无 48h 等待。
支撑 prereg 的 **pre-canonical-outcome-creation 时间锚**。

---

## 三、archived —— 只作 provenance，不能进新分析

- **§21** 2026-05-15 之前的全部 `docs/analysis` 输出（含 `sr_fp_per_mode.{json,md}` 旧 schema）
  → `docs/archive/analysis_pre_2026-05-15`。*"不能用于新 aggregator 消费（schema 已 break）"*
- **§1 §3** §3–§56 与 §77/§80 的 finding 正文 → 同一归档目录。
  **can_support 原文**：*"任何要引用 §3-§56 finding 数字的场合必须从这里读，**不要重新分析**"*
- **§23** B0 proxy capability probe v1 → 只证明 migration 前 Anthropic-style payload 被 400 拒，
  **不能支撑任何 SR/cost 结论**

---

## 四、pre-fix —— 存在但不是 paper-grade

这些的共同点：**早于某个修复**，因此不可与 canonical 并列。

| 数据 | 早于什么 | § |
|---|---|---|
| B0/B1 reddit SoM 重跑（max_marks=200，各 210/210） | §105 swatch fix 与 §107 Phase A 4-cluster fix | §5 |
| §97 rederive 回填（B0 cls 702 ep + red 630 ep + shop DOM 464 ep） | 同上 | §6 |
| B1 VWA shopping（DOM 466 完整，SoM 4/466 中断） | §97 修复；`validate_run` 报 **44.2% 覆盖率 FAIL** | §4 |
| B2 DOM pilot（cls + red 各 3 ep） | grade=in-flight ⇒ **不进 paper-grade 聚合池**，只证明管线可跑 | §28 |
| Mirage curation 全量评分（209 task jsonl gitignored 138KB） | — | §12 |
| Stage 1-4 mechanism 16 个 artifact | 原文注：**不受 pre-§116 early-stop bug 影响**（mechanistic inference 是单步非 trajectory 派生），但**底层 observation artifacts 来自 pre-Phase-A run** | §16 |
| Cell E provenance gap（manifest `random_inject` 字段为 None） | 靠三点旁证确认确为 random-injection variant：目录名 + L35 baseline ≠ 1.000 + 与 Cell A 相同 24 task IDs | §15 |

**§47 mechanistic canonical artifacts**（1.75 GB，gitignored）是 clean 的 A100 数据源，
但 *"§5 属 advisor 2026-05-14 已搁置范围，不进当前 paper"*。

---

## 五、文献与外部参考

- **§9** Deep Research 文献综述：5 个 dimension **全部 confirm literature gap**
  （A SoM lit 全 bundle text+image / B representation routing 缺 text-format-level /
  C AXTree vs flat list head-to-head 缺失 / D prompt format sensitivity 有 theory anchor
  但无人应用到 web agent routing / E cost-aware web agent 只 prune 不 reformat）
- **§10** Gemini DR Zoom 3 六份报告（~1351 行 / ~80K text）。Q3 verdict =
  **no study isolates SoM-style flat text as standalone observation**（first-work claim lit-verified）。
  ⚠️ 同时提供**反向 counter-evidence catalog**：text-only 在 perception-conditioned /
  dynamic GUI / behavioral tracking task 上 collapse
- **§11** verified 中国 industry arXiv/GitHub cheat sheet（2026-05-04 逐条 WebFetch+WebSearch 核过）
  —— **替换 V2 全部 fabricated arXiv ID**

---

## 六、被显式 defer 的技术债

**§25**（`master_bug_catalog.md ## A2.3d`）：9 条 P1/P2 被显式延后而非解决 ——
IV-weight 不确定性 disclaim · per-site 2-stage meta · heterogeneity-cap 75% 阈值论证 ·
k=6 power projection B2 SE 行 · "4-5 of 6" 描述性 prose · unmodified-HKSJ shrink 声明 ·
`aggregate_phantom_meta:254` CI-to-SE 换算 · `sensitivity_loo_meta` "DL primary" claim ·
`preregistration_decision_test` stale docstring。

> can_support 原文：*"记录哪些统计方法学问题被**显式 defer** 而不是解决 —— **重做前应先查这里**"*

**§32** accounting reset 的 disclosure debt（#74）：paper §3.5 需披露 two-budget + safety caps
是 P79-GRL reliability layer（upstream 无 parse-error caps）/ 三列 cost 估计量 /
off-site goto 的 VWA-domain 约束。⚠️ **pre-reset 数据这些字段全 None/0** ⇒
disclosure 必须说明字段的 vintage 边界。

**§22 §27** 两批 pre-existing red tests 已定性归因（B-650 fix 从未进 production；
`analysis.py:1306` 的 `NameError cell_key`），**证明不是当轮 audit 引入**。
