---
type: analysis
status: audit
created: 2026-09-22
purpose: 从零判定「CU-Pareto / Computer-Use Operating Envelope」这条高层 framing 能不能成为一篇论文；先实测轴的可识别性、配置空间覆盖度、前沿的抗噪性，再裁决
scope_warning: 本文件不是阶段权威。stage / 进度 / 下一步以 docs/checkpoints/phase1_plan.md + next_steps.md 为准。本文件只给判断与证据指针。
inherits: docs/analysis/trajectory_adaptive_compute_audit_2026-09-22.md 的**事实、数据审计、负结果、证据指针**；其 framing 结论（v1 §8 / v2.8 / v3.7）在本文件中**不作预设**
---

# CU-Pareto / Operating Envelope — grounded framing audit

> **一句话裁决**：按你写的那个样子（"map and decompose the operating frontier"），**判死** ——
> 因为这个项目**已经做过**跨模型/跨表征的 Pareto 分解（台账 §377，本次独立复算数字完全一致），
> 而且 2026 年有三篇论文已经把"oracle headroom ≠ 可实现收益"这一格占满了。
> 但同一批证据里有一篇**更强**的论文，它不是"画前沿"，而是**"这些轴在有状态真实浏览器里根本没被识别出来"**：
> 五条资源轴里四条在本数据上塌成同一条或不可识别，200 条 nominal Pareto 支配只有 **24%** 扛得住重采样、**30%** 扛得住实测 rerun band，
> 唯一能精确评估的 step 轴在弱 cell 上**也不可识别**（成功里 88% 从未显式 finish），
> 而真正正交的第六条轴（不可逆状态破坏）**不随能力单调**、且没有任何已发表的 CU 成本模型给它定过价。
> ⇒ **framing 从"测量前沿"改成"这个领域还没有能承载前沿的测量"**，这条成立，且比 trajectory-routing 更强。理由见 §12。

---

## 0. 读之前必须知道的四件事

### 0.1 本文件继承什么、不继承什么

继承：`trajectory_adaptive_compute_audit_2026-09-22.md` 的全部**实测事实**（数据位置、schema、死字段、
缺失模式、内部负结果、E: 台账指针），以及 v2.6 的物理限制（`artifacts/` 本机为 0，逐步 observation 不可得）。
**不继承**：它的 v1 §8 / v2.8 / v3.7 三版路线裁决。本文件从零判。

### 0.2 A100 回拉的状态（影响哪些分析可做）

回拉在跑，独立进程，harness 看不到。已落地部分：`E:\a100-condenser-backup\home-ubuntu\...`（workspace 已 DONE，
8.09 GB）；84 GiB 的 `artifacts/` 主体仍在传。**本文件的全部分析都不依赖 artifacts**，
按 v2.6 的降级口径做（只用 `state_digest` / `text_similarity` / `page_change_reasons` / 事件流）。
A100 比 E: 多约 1,100 个 episode 的逐步日志（9 月那批），本文件未包含它们。

### 0.3 本次实测用的是白名单口径，不是裸 glob

上一份 audit 的探针是裸 glob（它自己标注了"可行性探针，不是可引用的测量"）。
**本文件不同**：全部经 `_incoming_evidence_20260920/BUNDLE_MANIFEST.json` 的
`grade == "paper-grade"` 白名单取 run（该 manifest 的 `source_manifest_sha256` =
`d9eea73f31a16fbf216a8a4298d886b97a5f7aa65b5179c5e36eb5ae6fb7d4c7`，指向
`results/phantom_paper/run_manifest.yaml`），因此本文件的数字是**可引用口径**。
例外两处会显式标注：B5 相关数字（B5 不在 paper-grade 白名单里）与 E: 台账引文。

**复算脚本**：`C:\workspace\_p79_audit_scratch\{cu_inv,cu_pertask,cu_frontier,cu_frontier2,cu_econ,cu_trunc_audit,cu_trunc_band,cu_18arm,risk_scan,b5_scan,extract_steps}.py`
（`extract_steps.py` 从 E: 镜像抽了全部 36 个 paper-grade condition 的 7,722 个 `*_steps_v2.jsonl` 的逐步
cost/tokens/latency/energy/action/page_changed，落到 `cu_steps.jsonl`，0 缺失）。

### 0.4 一条必须先说的更正（影响你 prompt 里的前提）

你的 prompt 写「v3 已确认 B1 是 `electricity_usd_derived`、B0/B5 是 `api_usd`，尺度差约 1000×，不能直接同轴」。
**标签对，尺度错，而且错的方向让问题更糟**。实测见 §2.2：`total_billed_cost_usd` 这个字段在 B1/B2 上
**存的不是电费**，是按本地 token 常数（`exp_v2_base.yaml:80-81` 的 `$0.00093/1k` in、`$0.00185/1k` out）
算出来的 token 价。逐 episode 精确对账：B1 cls·DOM task 0 的 `total_input_cost_usd` = 0.07299291
= 78,487 tok × 0.00093/1k，`total_output_cost_usd` = 0.00416805 = 2,253 tok × 0.00185/1k，两者相加
= `total_tokens` = 80,740，**一位不差**。

⇒ 三个 backbone 的 stored cost 换算成单价分别是 **B0 $0.00107–0.00111 / B1 $0.00096–0.00097 /
B2 $0.00095–0.00097 每 1k token**（各 12 个 condition 的范围，实测）。
**它们只差 1.11–1.17×，不是 1000×。**

这比"差 1000×"危险得多：差 1000× 的两个数没人会去并排；差 15% 的两个数**看起来可以直接同轴**，
而它们一个是**发票**、一个是**基于错了 4–9 倍的吞吐假设算出来的硬件摊销**（`local_cost_estimand_audit.md`：
假设 60 tok/s，实测 248–551 tok/s）。真正的电费口径（`energy_kwh × $0.12`）是 **$0.000677/episode**（B1 cls·DOM），
与 stored 的 $0.05951 差 **88×**。同一个"本地成本"在本仓库里有**三个相差最多 88 倍的值**，
而字段名 `cost_unit_basis` 声明的是其中**不被该字段存储的那一个**。

> 这条更正本身就是 §12 里那篇论文的第一张图。

---

## 1. Canonical operating set（实测，白名单口径）

36 个 paper-grade condition = **2 site × 3 model × 6 mode**，7,722 个 scored episode（`sr_excluded` 已剔）。
⚠️ **shopping / WebArena / B5 / B3 / B4 都不在 paper-grade 白名单里**——这是本次审计第一个结构性发现，
见 §3。

| site | model | mode | n | SR% | steps | latency(s) | tokens | stored $ | unit label | energy kWh | waste% |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| cls | B0 | DOM | 224 | 17.4 | 15.6 | 114 | 63,045 | 0.06962 | api_usd | — | 87 |
| cls | B0 | SoM | 224 | **27.2** | 13.7 | 106 | 67,000 | 0.07236 | api_usd | — | 87 |
| cls | B0 | Vision | 224 | 25.0 | 15.9 | 125 | 58,302 | 0.06481 | api_usd | — | 87 |
| cls | B0 | P-text | 224 | 15.6 | 15.8 | 120 | 62,154 | 0.06919 | api_usd | — | 90 |
| cls | B0 | P-prompt | 224 | 19.6 | 15.0 | 108 | 62,555 | 0.06853 | api_usd | — | 87 |
| cls | B0 | P-SoM | 224 | 15.6 | 16.2 | 118 | 65,334 | 0.07206 | api_usd | — | 91 |
| cls | B1 | DOM | 224 | 6.2 | 21.4 | 309 | 61,904 | 0.05951 | electricity* | 0.005644 | 98 |
| cls | B1 | SoM | 224 | 14.3 | 18.0 | 262 | 62,953 | 0.06028 | electricity* | 0.004818 | 88 |
| cls | B1 | Vision | 224 | 12.5 | 20.2 | 270 | 44,524 | 0.04316 | electricity* | 0.004966 | 94 |
| cls | B1 | P-text | 224 | 7.6 | 22.5 | 314 | 61,074 | 0.05879 | electricity* | 0.005745 | 97 |
| cls | B1 | P-prompt | 224 | 6.7 | 21.4 | 301 | 65,628 | 0.06304 | electricity* | 0.005523 | 97 |
| cls | B1 | P-SoM | 224 | 6.7 | 21.3 | 312 | 61,985 | 0.05970 | electricity* | 0.005707 | 95 |
| cls | B2 | DOM | 224 | 1.3 | 27.4 | 402 | 79,653 | 0.07676 | electricity* | 0.007420 | 99 |
| cls | B2 | SoM | 224 | 2.2 | 24.4 | 374 | 95,081 | 0.09075 | electricity* | 0.006952 | 99 |
| cls | B2 | Vision | 224 | 2.2 | 28.2 | 418 | 73,126 | 0.07065 | electricity* | 0.007767 | 98 |
| cls | B2 | P-text | 224 | 0.4 | 26.8 | 399 | 75,946 | 0.07320 | electricity* | 0.007363 | 99 |
| cls | B2 | P-prompt | 224 | 1.8 | 27.8 | 396 | 87,948 | 0.08453 | electricity* | 0.007300 | 99 |
| cls | B2 | P-SoM | 224 | 0.9 | 28.4 | 411 | 87,931 | 0.08456 | electricity* | 0.007576 | 99 |
| red | B0 | DOM | 205 | **14.6** | 20.2 | 553 | 93,112 | 0.10128 | api_usd | — | 90 |
| red | B0 | SoM | 205 | 14.6 | 20.0 | 449 | 102,568 | 0.10996 | api_usd | — | 90 |
| red | B0 | Vision | 205 | 7.8 | 23.2 | 415 | 88,325 | 0.09747 | api_usd | — | 97 |
| red | B0 | P-text | 205 | 13.7 | 23.0 | 558 | 95,467 | 0.10495 | api_usd | — | 89 |
| red | B0 | P-prompt | 205 | 12.7 | 19.8 | 444 | 93,297 | 0.10107 | api_usd | — | 90 |
| red | B0 | P-SoM | 205 | 11.2 | 22.7 | 533 | 98,687 | 0.10779 | api_usd | — | 92 |
| red | B1 | DOM | 205 | 6.8 | 23.4 | 603 | 76,172 | 0.07302 | electricity* | 0.010976 | 94 |
| red | B1 | SoM | 205 | 8.3 | 22.2 | 605 | 82,845 | 0.07936 | electricity* | 0.011035 | 91 |
| red | B1 | Vision | 205 | 2.9 | 23.1 | 453 | 53,611 | 0.05204 | electricity* | 0.008303 | 98 |
| red | B1 | P-text | 205 | 6.8 | 25.5 | 595 | 71,945 | 0.06921 | electricity* | 0.010783 | 95 |
| red | B1 | P-prompt | 205 | 6.3 | 23.7 | 610 | 79,558 | 0.07624 | electricity* | 0.011091 | 95 |
| red | B1 | P-SoM | 205 | 6.8 | 25.2 | 614 | 77,531 | 0.07461 | electricity* | 0.011164 | 93 |
| red | B2 | DOM | 205 | 3.9 | 28.4 | 669 | 98,849 | 0.09464 | electricity* | 0.012236 | 96 |
| red | B2 | SoM | 205 | 1.5 | 26.4 | 622 | 117,311 | 0.11155 | electricity* | 0.011416 | 98 |
| red | B2 | Vision | 205 | 2.4 | 26.9 | 554 | 70,820 | 0.06833 | electricity* | 0.010180 | 98 |
| red | B2 | P-text | 205 | 2.4 | 27.3 | 678 | 92,375 | 0.08855 | electricity* | 0.012364 | 97 |
| red | B2 | P-prompt | 205 | 0.5 | 27.7 | 601 | 103,980 | 0.09937 | electricity* | 0.011018 | 100 |
| red | B2 | P-SoM | 205 | 1.5 | 27.9 | 641 | 98,575 | 0.09440 | electricity* | 0.011416 | 99 |

`electricity*` = 字段这么**声明**，但存的是 token 价（§0.4）。`waste%` = `wasted_cost_usd / total_billed_cost_usd`。

---

## 2. Q1 — 哪些 frontier 是 apples-to-apples？（轴可识别性审计）

**结论先行：五条资源轴里，能独立承载一条 frontier 的只有 `steps`，而且只在强 cell 上。**
逐条如下。

### 2.1 success–energy：**不是轴，且跨 backbone 不存在**

`energy_carbon_audit.md`（24 condition 实测）：per-step energy 与 per-step latency 相关
**r = 0.9659–0.9998（均值 0.9935）**，记录功率恒定在 **66.3 W（跨 condition SD 0.40 W）**。
近似常功率 × 时间 **就是**时间。且两条硬伤：`use_pynvml: true` 被配置了但每一步记的是 `psutil_profile`
（CPU 包估算，不是 GPU 计数器）；**B0 的 energy 全部是 `disabled`**（走 API，本地无功耗，这是对的设置）。

⇒ **success–energy 前沿：删掉。** 它在 B1/B2 内部是 success–latency 的换算，在跨 backbone 上根本不存在。
（本次实测复核：36 个 paper-grade condition 中 B0 的 12 个 `total_energy_kwh` 非空率 0%，B1/B2 的 24 个是 100%。）

### 2.2 success–provider cost：**只有 B0（和白名单外的 B5）是发票；B1/B2 的"成本"是 token 数换了个单位**

见 §0.4。补一条本次新测的：**stored cost 与 tokens 的 episode 级 Pearson r 中位数 = 1.000（min 0.998，
36 个 condition）**。也就是说 success–cost 前沿与 success–tokens 前沿**在图上是同一张图**，
差别只有一个 backbone 常数。

再叠上 `local_cost_estimand_audit.md` 的实测：本地成本换成 GPU-time basis（同样 $0.20/hr，作用在实测 model time 上），
**4 个本地 cell 里有 2 个最便宜的 mode 变了**。⇒ 本地 $ 是 **estimand-dependent**，
单一 basis 的 success–cost 前沿在本地 backbone 上**不是唯一确定的对象**。

⇒ 可写的：**B0（+ B5）内部的 success–cost 前沿是真发票，可写。**
不可写的：任何把 B0 与 B1/B2 的美元放同一条轴的图；任何不声明 basis 的本地成本排序。

### 2.3 success–latency：**在 mode 之间不可分辨**

两条独立证据指向同一处：

- `replicate_metric_noise.md`：`mean_latency_s` 的跨 mode 极差 19.562s 对 rerun band 22.488s，
  **ratio 0.87×**；`mean_latency_canonical_s` 19.204 / 22.846 = **0.84×**。
  26 个行为指标里只有 3 个低于 1.0×，**两个 latency 指标都在里面**。
- `latency_decomposition.md`：model call 只占 step 的 **22–67%**；去掉容器后
  **8 个 cell 里 4 个最快 mode 变了**，且 4/5 个 reddit 系 cell 翻、0/3 个 classifieds cell 翻——
  这是容器效应的形状，不是噪声的形状。

⇒ **任何"X mode 最快"的句子都不成立。** 能活的只有 `multimetric_pareto` 的弱版本：
**cost 排序与 latency 排序不是同一个排序**（8 cell 平均 ρ = −0.014），这是关于两个 ranking 的陈述，
与 estimand 无关。

### 2.4 success–tokens：**可写，但它就是 success–cost**

见 2.2。它是本数据上**唯一跨三个 backbone 单位一致**的资源轴（token 是数，不是价）。
`multimetric_pareto` 自己写着：加上 tokens 只在 **1 个 cell** 里扩大了前沿。

### 2.5 success–steps：**唯一一条有独立信息的轴——但只在强 cell 上被识别**

好消息，实测：**在 episode 级，steps 与 tokens 的 r 中位 0.926**（所以"多跑一步 ≈ 多花一份钱"）；
但**在 mode 之间**（也就是 Pareto 图真正用的那个层面），`r(steps, tokens)` 在 6 个 cell 里是
**−0.53 / +0.15 / −0.54 / −0.29 / +0.00 / −0.06** —— **四个负、两个≈0**。
SoM 步数最少但每步 token 最多。⇒ **steps 与 tokens 在 mode 轴上是真正分开的两条轴**，
这是整份数据里资源轴唯一一次真正分叉。

坏消息见 §7.2：steps 轴的截断前沿在弱 cell 上**不可识别**。

### 2.6 Q1 的汇总裁决表

| frontier | 同 backbone 内 | 跨 backbone | 判定 |
|---|---|---|---|
| success–steps | ✅ 可写，且是唯一有独立信息的 | ✅ 单位一致（步就是步） | **主轴**，但弱 cell 上需报识别区间 |
| success–tokens | ✅ | ✅ 单位一致 | **可写**，但等价于 cost |
| success–cost | ✅ 仅 B0/B5（真发票） | ⛔ 永不 | B1/B2 的"$"必须改称 token-priced |
| success–latency | ⚠️ 只能说"排序与 cost 不同" | ⛔（容器/硬件不同） | 不可命名最快 mode |
| success–energy | ⛔（= 时间） | ⛔（B0 根本没有） | **删轴** |
| success–risk | 见 §9 | 见 §9 | **唯一真正正交的新轴** |

---

## 3. Q2 — configuration space X = (model, perception, scaffold, step budget, routing/control)

**实测结论：五维里只有两维被真正点过火。** 逐维盘点（证据 = 108 份 `condition_meta.json` 全量 +
120 个 `configs/*.yaml`）：

| 维度 | 取值域（设计上） | 实际有 paper-grade 数据的 | 空的部分 |
|---|---|---|---|
| **model** | B0/B1/B2/B3/B4/B5 | **B0, B1, B2**（3 个） | B3 = 0 episode（只有 config）；B4 = proxy 协议层拒绝 Anthropic 族，2 次 smoke 全 400；**B5 有 1,708 个 episode 但不在 paper-grade 白名单** |
| **perception** | dom/som/vision/phantom_{text,som,prompt} | **6 个全有，2 site × 3 model 全满** | wa/shopping 上不全 |
| **scaffold** | `modules: [none, m1, m2, m3, m4]` | **只有 none** | 108/108 条件的 `module_flags` 四个全 False。**m4（two-stage planner/grounding）实现完整、从未开过** |
| **step budget** | `runtime.max_steps` | **只有 30** | 120 个 config 里 119 个是 30；唯一的 15 在 `exp_v2_B3_som_classifieds_pilot.yaml`，而 B3 有 0 个 episode ⇒ **step budget 从未在 fire 里变过**，全部是 post-hoc 截断模拟 |
| **routing/control** | `router_on` / `router_kind` | **只有 off** | 108/108 是 `router_on: False, router_variant: "baseline"`。6 个 learned-router config 存在，但走的是"开跑前选一次 mode"的路径 |
| **reasoning compute** | （你 prompt 里的第 6 维） | **完全没有** | 三个 backbone 都没有 reasoning-effort 旋钮；`tokens.thinking` 全项目恒 0 |
| **verification/recovery** | | **完全没有** | `RuleBasedRouter` 代码完整、从未启用；`checkpoint_*` 三个字段声明后全仓库再无出现 |

**这一节是对 framing 最硬的约束**：你的 operating-family 定义有六个决定维度，
我们在**两个半**上有数据（model 3 档、perception 6 档、step budget 只有 post-hoc）。
一篇叫 "mapping the operating envelope" 的论文，**图上会有四个维度是空的**——
而审稿人一定会数这个。

---

## 4. Q3 — 五种 frontier 的构造与比较

全部逐 cell 内做（不跨 backbone 池化 $），n=224（cls）/ 205（red）。

### 4.1 Nominal empirical frontier（单次均值，6 mode）

| cell | SR×steps | SR×latency | SR×tokens | SR×cost |
|---|---|---|---|---|
| cls_B0 | 1/6 {SoM} | 1/6 {SoM} | 2/6 {SoM, Vision} | 2/6 {SoM, Vision} |
| cls_B1 | 1/6 {SoM} | 1/6 {SoM} | 2/6 {SoM, Vision} | 2/6 {SoM, Vision} |
| cls_B2 | 1/6 {SoM} | 1/6 {SoM} | 1/6 {Vision} | 1/6 {Vision} |
| red_B0 | 2/6 | 3/6 | 2/6 | 3/6 |
| red_B1 | 1/6 {SoM} | 3/6 | 3/6 | 3/6 |
| red_B2 | 3/6 | 2/6 | 2/6 | 2/6 |

合计 **200 条 strict domination**（4 轴 × 6 cell）。

### 4.2 Robust frontier（两种口径，都实测）

**(a) 逐 task 配对 bootstrap（B=2000，同 task 集重采样，SR 与资源同抽）**：
200 条 nominal domination 里 **48 条（24.0%）**在 95% 水平上成立。

| cell | steps | latency | tokens | cost |
|---|---|---|---|---|
| cls_B0 | 5/12 | 2/11 | 1/8 | 1/9 |
| cls_B1 | 5/11 | 8/11 | 4/9 | 4/9 |
| cls_B2 | 1/8 | 1/9 | 0/6 | 0/7 |
| red_B0 | 4/11 | 0/5 | 0/6 | 0/5 |
| red_B1 | 1/9 | 0/7 | 0/6 | 0/6 |
| red_B2 | 1/7 | 0/6 | 5/11 | 5/11 |

**(b) 实测 rerun band 口径**（band 来自 `replicate_metric_noise.md` 的三个 B0·cls 复制臂：
SR 2.232pp / n_steps 0.469 / latency 22.846s / tokens 2186.054 / cost 0.002；⚠️ 只在 B0·cls 上实测，
其余 cell 是**借用**）：**60/200（30.0%）**的支配的两个分量都超过各自 band。
按此口径 `cls_B2` 与 `red_B1` 的**全部四条轴上前沿都是 6/6**——即**这两个 cell 上根本没有可分辨的操作点**。

⚠️ 两个口径答的是不同问题：(a) 是"同一次 run 的 task 抽样噪声"，(b) 是"重跑一次会不会翻"。
`noise_floor_inventory.md §1b` 已经证明后者更严：SR 差的 null SD 是 **2.32–2.53pp**，
单侧 95% 要到 **3.82–4.15pp** 才算清得过一次重跑。⇒ **(b) 仍是乐观的下界。**

### 4.3 Oracle frontier

来自 `routing_ceiling.md`（leak policy = primary，已锁），本次独立复算一致：

| cell | best single | 6-mode oracle | headroom | +1 arm | rerun once |
|---|---|---|---|---|---|
| cls_B0 | SoM 27.23% | 43.30% | +16.07pp | +7.14pp | 4.91–7.59pp |
| red_B0 | SoM 14.78% | 26.11% | +11.33pp | +4.93pp | — |
| cls_B1 | SoM 14.29% | 24.55% | +10.27pp | +4.91pp | — |
| cls_B2 | SoM 2.23% | 7.14% | +4.91pp | +2.23pp | — |

**关键读法**（`noise_floor_inventory` 已裁定且不得改写）：在 cls_B0 上，
**多加一个不同表征臂（+7.14pp）与多跑一次同一臂（4.91–7.59pp）无法区分**。
六臂 oracle 的 +16.07pp 必须**带着"这是 5 个额外臂"一起引**。

### 4.4 Attainable / learned frontier

- learned router：真嵌套 CV 下 **0/6 cell** 能 Pareto 胜过 always-cheapest；
  在最有利的角落（同 family 池化 × cost-tier 标签）扩展为 **0/26**。
- rule router（0-token 正则）：在 cls_B0 上**落在**三轴前沿里但**不严格支配任何固定策略**；
  在 cls_B1 上**被 always-Vision 支配**。
- triage 可学性：AUROC 0.615–0.864，但 **5/6 cell 的最强单特征是 `reasoning_difficulty`**
  ——VisualWebArena 任务配置里自带的**人工难度标注**，任何部署都没有。
- **唯一的正结果**：step-budget router 的预注册前瞻检验（冻结于 2026-09-09，git tag
  `budget-router-prospective-20260909`），在 shop_B1 两条 held-out 臂上 **2/2 PASS**
  （§510.3：P-text n=216，two_tier +1.39pp vs 等成本固定 cap +1.85 vs 随机 +2.11；
  §515.3：P-prompt n=432，two_tier +0.69pp 且成本 −44% vs cap-15 +2.55 vs 随机 +2.06）。
  ⚠️ 台账自带限定：单 run、只有 B1、SR≈5%、无 rerun band ⇒ **方向判据不是显著性**。
  ⚠️⚠️ **本次新发现的一条额外限定见 §7.2**——它打在这个结果的识别性上，比上面那条更重。

### 4.5 Risk-adjusted frontier

见 §9。**能构造，且是本数据上唯一一条与现有轴正交、且统计上显著的轴。**

### 4.6 五种 frontier 的关系（本节的可发表命题）

> **Oracle frontier 宽（+4.9 ~ +16.1pp），nominal frontier 窄（1–3/6 臂），
> robust frontier 几乎退化（24–30% 的支配存活；两个 cell 完全退化），
> attainable frontier 在表征/模型轴上等于最好的固定臂（0/6、0/26），
> 只有在 step 轴上存在一条冻结在前、2/2 通过的正结果。**

---

## 5. Q4 — Frontier attribution：哪个 lever 真的扩大 attainable frontier

### 5.1 Oracle 层（18 臂同平面，本次独立复算，与台账 §377 完全一致）

| site | best single | +perception diversity（同模型 6 mode oracle） | +model diversity（同 mode 3 backbone oracle） | joint（18 臂） |
|---|---|---|---|---|
| classifieds | B0/SoM 27.23% | **43.30%（+16.07pp，6 臂）** | 31.25%（+4.02pp，3 臂） | 46.43%（+19.20pp，18 臂） |
| reddit | B0/DOM 14.63% | **26.83%（+12.20pp，6 臂）** | 17.56%（+2.93pp，3 臂） | 29.76%（+15.12pp，18 臂） |

⇒ **表征轴携带的 oracle headroom 约为模型轴的 4×**（§377 已记为 4–5×）。

**per-task 独解（本次新算）**：classifieds 上 B0 任意 mode 解 97 题（43.3%），其中 **44 题只有 B0 能解**；
B1 独解 **4 题（1.8pp）**，B2 独解 **2 题（0.9pp）**。reddit：B0 独解 28（13.7pp）、B1 独解 4（2.0pp）、B2 独解 2（1.0pp）。
⇒ **两个 4B 模型加在一起，对组合能力的独立贡献不到 3pp**，而它们各自要占一整份预算。
⚠️ menu-specific：B1/B2 都是 4B 级；强-强模型的 menu 可能不同（§377 自带这条 caveat，B5 可检验但不在白名单）。

### 5.2 Matched-budget 层（这是你要的"等资源预算"）

用 **steps** 作预算单位（三个 backbone 单位一致）。

| lever | classifieds 的可达 SR | 预算（mean steps/task） | 相对 best-single 的预算倍数 |
|---|---|---|---|
| 固定最优单配置（无任何 diversity） | **27.23%** | 13.7 | 1.0× |
| **perception diversity，portfolio 记账（6 臂全跑）** | 43.30% | 92.2 | **6.7×** |
| **model diversity，portfolio 记账（3 臂全跑）** | 31.25% | 56.0 | **4.1×** |
| perception diversity，oracle 记账（只付被选中那臂） | 43.30% | ≈13.7 | 1.0×（**不可达**） |
| step-budget control（cap 15，同一臂） | 17.3–18.0%（见 §7.2 的识别区间） | ≈9.0 | **0.66×** |
| learned which-mode routing | ≤ best single（0/6 Pareto） | 1.0× | — |

**同预算约束下的直接回答**：在 ≤13.7 步的预算内，**18 个配置里最好的仍然是 B0/SoM 本身**
（实测：`best config at <= 13.7 steps` = ('B0','SoM') 27.23%）。
reddit 同理（≤20.2 步内最优仍是 B0/DOM 14.63%）。

⇒ **Q4 的答案（实测，非推测）**：
1. **model diversity 在本 menu 上几乎不扩大任何前沿**——oracle 只 +2.9~4.0pp，独解贡献 <3pp，portfolio 记账要 3.6–4.1× 预算。
2. **perception diversity 扩大 oracle frontier 最多（4× 于 model），但扩大 attainable frontier 的量是 0**
   （0/6、0/26），且 portfolio 记账要 6.4–6.7× 预算换 +12~16pp。
3. **step-budget control 是唯一一个把前沿往"便宜"方向真实推动、且有前瞻证据的 lever**
   （cap 15 = 0.66× 预算换 −2.8pp，learned 分档在 11/11 cell 优于等成本固定 cap）。
4. **learned routing 不扩大任何东西。**

---

## 6. Q5 — 负结果能否上升为"complementarity ≠ deployable gain"，以及它还新不新

### 6.1 能，而且本项目的证据链比多数论文完整

本项目对同一命题有四条**互相独立**的证据：
(i) oracle headroom 存在且可观（+4.9~16.1pp）；
(ii) learned router 在真嵌套 CV 下 0/6、池化后 0/26；
(iii) 失败的**机制**被定位到**标签产生率**（标签只在任务被解开时诞生，而 base SR 只有 0.4–27%）——
这一条是多数 routing 论文没有的；
(iv) 加一个臂的收益与**重跑一次**同一臂的收益无法区分（arm-count-matched 对照）——
这一条据我所知在 web agent 文献里没有第二家做过。

### 6.2 但这个结论在 2026 年已经被发表了三次（逐条 arXiv API 核实，含控制查询）

控制查询 `id_list=2307.13854` → 200 + WebArena 条目通过；下列 id 我**亲自 curl 复核**了其中
2606.26836 / 2608.08265 / 2604.27151 / 2609.02309 / 2506.16042 五条的标题与摘要首句。

| 论文 | arXiv id | 日期 | 它占掉了哪一格 |
|---|---|---|---|
| **The Capability Frontier: Benchmarks Miss 82% of Model Performance** | 2606.26836v1 | 2026-06-25 | **直接定义** "Capability Frontier = 一组模型在 oracle 选择下逐 cost 水平的 Pareto 前沿"；21 LLM × 16 benchmark；同时纠正"单模型低估"与"对噪声取 max 高估"；SOTA 精度可在 **85% 成本下降**处达到 |
| **Opportunity Is Not Realizability: Selection-Valid Diagnostics for Multi-LLM Routing** | 2608.08265v1 | 2026-08-08 | 把 oracle 分成三个 estimand（outcome-oracle opportunity / 声明信号下的 Bayes 最优 / 学到的 router 的 held-out 收益），给 selection-valid CI；**certified oracle gap 9.7–30.7 点，最强可部署 router 只回收 7.5–14.4%** |
| **How Much of the Routing Gap Is Real?** | 2607.03436v2 | 2026-07-03 | 把 router-to-oracle gap 分解成"可复现的专长优势"与"single-draw 标签噪声"，**后者占 12–36%**；证明该地板**没有任何 single-commit router 能关上**，但 best-of-K 能 |
| **The Routing Plateau** | 2606.07587v1 | 2026-05-27 | 21 种 routing 方法 × 5 benchmark 收敛到远低于 oracle 的窄带；归因于 **predictability bottleneck** |
| **LLMRouterBench** | 2601.07206v1 | 2026-01-12 | >400K 实例 / 21 数据集 / 33 模型；统一评测下多数 routing 方法**打不过简单基线**；与 oracle 的差距主因是 model-recall |
| **Budget-Dependent Rankings in LLM Evaluation** | 2608.12150v1 | 2026-08-12 | 7 档 token 预算 × 4 模型；**排名随预算反转**；oracle complementarity 最高 +27.8pp，budget-aware router 只回收 **14.1%** |
| **COVER: Identifiable Evaluation of Coalition Routing** | 2608.28475v1 | 2026-08-28 | end-to-end 精度差**不能识别** routing effect；提出先冻结信息边界与 team family 的评测契约 |

**逐条影响（必须写进任何 related work）**：

1. **"oracle headroom ≠ deployable gain" 不能再当 contribution。** 2608.08265 给了 certified CI，
   2607.03436 给了噪声分解，2606.07587 给了机制。我们的 0/6 与 0/26 变成**在新领域的第四次确认**。
2. **"portfolio complementarity"这个词也被占了**（2606.26836 定义的 Capability Frontier 就是它）。
3. **我们的 (iv)"加一个臂 ≈ 重跑一次"仍然是独有的**，因为上面所有论文的"重跑"是
   **同一个静态 prompt 的重采样**，而我们的是**重跑一整条有状态浏览器 episode，含环境漂移**
   （`noise_floor_inventory` 明写这条 band"包含环境漂移，不是解码随机性"）。
   2607.03436 量的正是 single-draw 噪声，但它在静态基准上量；**在有状态环境里量这个东西没人做过**。
4. **我们的 (iii) 标签产生率机制**与 2606.07587 的 predictability bottleneck 是**不同的解释**
   （它说"router 学到的是全局平均趋势"，我们说"低 base SR 下正例根本不够"）。这是可主张的差异，
   但必须正面对比，不能装作没看见。

### 6.3 ⚠️ 半衰期警告

2026 年 1–9 月这一格进来了至少 7 篇。**投稿前必须重做一次系统 sweep**，
且路线图要按"这个空格可能在我们点火前被关掉"排优先级。

---

## 7. Q6 — Trajectory economics（全部实测自 7,722 份逐步 JSONL）

### 7.1 支出在哪里

- **失败 episode 吃掉的支出份额：36 个 paper-grade condition 的中位数 95.2%，范围 86.9–99.6%。**
  最低三个是 B0·cls 的 SoM/P-prompt/Vision（86.9–87.2%），最高是 B2 的 P-text/P-SoM/P-prompt（99.5–99.6%）。
  （对照：台账 §505.21 在 60 个 condition 上报 92%，范围 76–99%——口径不同，它含 WA 与 shopping。
  **两个数都真，引用时必须带口径。**）
- **每步成本几乎是平的**：cls 上 step 1 → step 30 的平均步成本比值
  **B0 1.06× / B1 1.12× / B2 1.04×**。⇒ 累计支出对步数**近似线性**，
  "cost ∝ steps"这个近似在本数据上成立到 4–12% 以内。
- **死算力尾巴**：最后一次页面变化之后仍然发生的支出占 episode 总支出
  **B0 14.1% / B1 27.3% / B2 30.4%**（cls，逐 episode 均值）。
  这是"模型还在动但世界已经不动了"的直接计量。

### 7.2 ⛔ 边际步数价值：一条打在项目**唯一正结果**上的新发现

截断前沿的前提是"成功发生在第几步是已知的"。**本次实测发现这个前提只在强 cell 上成立。**

| cell | 成功数 | 其中 steps≥30 | 其中 `agent_finished=False` |
|---|---:|---:|---:|
| cls_B0 | 270 | 10（3.7%） | 9（**3.3%**） |
| cls_B1 | 121 | 24（19.8%） | 24（**19.8%**） |
| cls_B2 | 20 | 8（40.0%） | 8（**40.0%**） |
| red_B0 | 153 | 35（22.9%） | 34（**22.2%**） |
| red_B1 | 78 | 38（48.7%） | 39（**50.0%**） |
| red_B2 | 25 | 22（88.0%） | 22（**88.0%**） |

这些 episode **从未发出 finish**；是 evaluator 对跑满 30 步后的**最终状态**判了成功。
对它们，"如果在第 15 步截断会不会成功"**在本数据上无法回答**——目标状态可能早就达成了。

⇒ 截断前沿必须写成**识别区间**（下界 = 这些成功在任何 cap 下都算丢；上界 = 都算还在）：

| cell | 全 SR | cap 5 | cap 10 | cap 15 | cap 20 | cap 25 |
|---|---:|---|---|---|---|---|
| cls_B0 | 20.09% | 9.7–10.4 @30% | 15.3–15.9 @51% | 17.3–18.0 @66% | 18.4–19.0 @79% | 19.0–19.6 @90% |
| cls_B1 | 9.00% | 4.8–6.5 @21% | 5.7–7.4 @39% | 6.2–8.0 @55% | 6.6–8.4 @71% | 7.1–8.9 @86% |
| cls_B2 | 1.49% | 0.5–1.1 @18% | 0.7–1.3 @35% | 0.8–1.4 @52% | 0.8–1.4 @68% | 0.8–1.4 @83% |
| red_B0 | 12.44% | 3.5–6.3 @22% | 6.3–9.1 @41% | 7.9–10.7 @58% | 8.8–11.5 @73% | 8.9–11.6 @87% |
| red_B1 | 6.34% | 1.5–4.7 @19% | 2.2–5.4 @37% | 2.5–5.7 @53% | 2.8–5.9 @69% | 3.0–6.2 @85% |
| red_B2 | 2.03% | **0.1–1.9** @18% | 0.1–1.9 @35% | 0.1–1.9 @51% | 0.2–2.0 @67% | 0.2–2.0 @83% |

（`@x%` = 该 cap 下的实际累计支出占未截断支出的百分比，逐 episode 精确求和，非线性外推。）

**读法**：`cls_B0` 的区间宽 0.6–0.7pp，可用；`red_B2` 的区间宽 1.8pp 而全 SR 只有 2.03pp ——
**整条曲线不可识别**。

**这直接打在 §4.4 的前瞻正结果上**：那个 2/2 PASS 是在 **shop_B1、SR≈5%** 上做的，
正是识别区间最宽的那一类 cell。我**没有**重跑 `budget_router_prospective_eval.py`，
所以不能说它的结论错；但**必须先核这一条**：
> 在 shop_B1 的两条 held-out 臂上，成功 episode 里 `agent_finished=False` 的比例是多少？
> 若与 red_B1 的 50% 同量级，那么"learned two_tier 的 SR 损失 +0.69pp vs 固定 cap +2.55pp"
> 这个 1.9pp 的差距，可能整个落在识别区间内部。

⇒ **这是本次审计最重要的一条新增待办**，排在任何新 fire 之前，代价 0 compute。

### 7.3 边际步数价值（只有强 cell 可读）

| cell | early（cap5→8）pp / 单位预算 | late（cap15→30）pp / 单位预算 | 比值 |
|---|---:|---:|---:|
| cls_B0 | 25.92 | 8.12 | **3.2×** |
| red_B0 | 16.06 | 10.75 | 1.5× |
| cls_B1 | 5.50 | 6.14 | 0.9× |
| red_B1 | 4.67 | 8.20 | 0.6× |

⚠️ **只有 cls_B0 这一行可读**（未 finish 成功仅 3.3%）。其余行的 "late" 被 §7.2 的识别问题
**系统性抬高**（未 finish 成功只在 cap=30 那一档才被计入，制造了一个假的末端跃升）。
⇒ 可主张的：**在标签最干净的 cell 上，前 8 步的边际产出是后 15 步的 3.2 倍。**
不可主张的：任何弱 cell 上的"边际价值递增/递减"。

---

## 8. Q7 — Robust Pareto 的口径设计

§4.2 给了两个已实测的口径。**建议的正式口径（三层，逐层更严）**：

1. **Level-1 · within-run task bootstrap**（已跑，B=2000）：
   答"同一次 run 的任务抽样会不会翻"。本数据上 **24.0%** 存活。
2. **Level-2 · rerun-band dominance**（已跑）：要求 SR 差与资源差**各自**超过实测 rerun band。
   本数据上 **30.0%** 存活。⚠️ band 只在 B0·cls 三个臂上实测，其余是借用——
   `noise_floor_inventory` 已证明借用这次是对的（SoM 臂落地后验证），但那是一次运气好的外推。
3. **Level-3 · exchangeability null**（推荐作为论文的主口径，`noise_floor_inventory §1b` 已给出公式）：
   `D = (2X − d)/n`，`X ~ Binom(d, ½)` ⇒ `SD(D) = √d/n`。实测 cls 三臂 SD = **2.32 / 2.53 / 2.40 pp**，
   单侧 95% 阈值 **3.82–4.15pp**。⇒ 任何小于 ~4pp 的 SR 优势**不该进 Pareto 判定**。

**不建议**做 full Bayesian：本数据每个 (cell, mode) 只有 1 次 run（3 个臂有 2 次），
层次模型的 run 级方差会几乎完全由先验决定，而先验正是争议所在。
Bootstrap + exchangeability null 两者都是**非参数且可复算的**，更适合一篇以"测量可信度"为卖点的论文。

**一条必须写进论文的推论**：
> 把 rerun 不确定性纳入后，`cls_B2` 与 `red_B1` 在四条轴上的 Pareto 前沿都是 **6/6 臂**——
> 即**在这两个 cell 上，六个操作点彼此不可分辨**。
> 一张 nominal Pareto 图会在这里画出 1–3 个"前沿点"，而那是噪声的形状。

---

## 9. Q8 — Risk-adjusted frontier：能构造，而且结论与直觉相反

### 9.1 口径（严格遵守现有 data-quality 规则）

⛔ 七个 mutation / risk 计数器在 20,284 份 episode summary 里**没有一份 > 0**，
`footprint_risk_score` 没有一份非 None ⇒ **一律不用**。
本节的事件全部来自 `infra_covariates`（受害者标记）与 `trajectory_events.jsonl`（事件流），
这是 B-1868 / `PROTOCOL_NOTE_01` 规定的 preserve + covariate 协议的产物。

### 9.2 实测（paper-grade 白名单内）

| cell | episodes | session-loss 受害 episode | 事件波数 | 每 episode 率 |
|---|---:|---:|---:|---:|
| B0 · cls | 1,344 | **0** | 0 | 0% |
| B1 · cls | 1,344 | **0** | 0 | 0% |
| B2 · cls | 1,344 | **16** | 9 | 1.19% |
| B0 · red | 1,230 | 0 | 0 | 0% |
| B1 · red | 1,230 | 0 | 0 | 0% |
| B2 · red | 1,230 | 0 | 0 | 0% |

B2·cls 的 16 个受害者的 task id（按 run 分组，可见波结构）：
DOM `{5,6,7} {37,38} {77,78}`；P-text `{73,74,75} {228,229}`；P-SoM `{197,198}`；P-prompt `{6} {76}`。

**白名单外的对照（必须一起报，因为它推翻了原先的假说）**：
- **B5（gpt-5.6-terra，全项目最强模型）cls：15 个受害 episode / 1,708，6 个波，全部在 vision mode**
  （两个 run：`{26} {99,100,101}` 与 `{5,6,7} {26,27,28} {76,77,78,79} {187}`）。
- B0 cls（含非白名单 run）：2 / 2,241，1 个波。

单侧 Fisher exact（episode 级，同 site）：
B2 vs B1 **p = 1.5e-05**；B5 vs B1 **p = 1.6e-04**；B2 vs B0 **p = 9.3e-06**；
**B2 vs B5 p = 0.25（不可区分）**。
按**波**计数（更保守，因为污染是簇发的）：B2 vs B1 p = 1.9e-03；B5 vs B1 p = 3.1e-02。

### 9.3 结论（与 trajectory audit v1 的 P3 假说相反）

v1 §3.8 写「破坏状态的是**便宜模型**」。**本次实测否掉这句**：
**B2（最弱）与 B5（最强）的 session 破坏率在统计上不可区分（p=0.25），两者都显著高于 B0 与 B1。**
⇒ **风险不随能力单调**，它是一条**与 capability 轴正交**的轴。

这反而让 risk-adjusted frontier 更有意思，不是更没意思：
一条按 SR/cost 画的前沿会把 B5·vision 与 B2 一起放在"低 SR 区"，
而一条 risk-adjusted 前沿会显示它们**因为完全不同的原因**在那里。

### 9.4 能构造到什么程度

**能**：per-(model, site, mode) 的不可逆事件率 + blast radius（每波的受害 episode 数 × 其成本）+
计入后的 cost-per-success。9 个波 / 16 个受害者，Fisher 显著，可作为**一节**。
**不能**：肇事集不打标（click 只存数字 `element_id`，grep "logout" 假阴），
要做"哪一步是高风险动作"的监督**必须 join observation**，而 artifacts 本机为 0（等 A100 回拉）。
**也不能**：给出一个通用的"每次 session loss 值多少钱"——那要一个本项目明确拒绝假设的汇率
（台账 §476.4：论文拒绝定义"一次额外 success 值多少钱"）。
⇒ **risk 轴只能以"事件率 + 受影响预算份额"的形式进前沿，不能折成单一标量。**

---

## 10. Q9 — Prior work 核实结果与 GPT-6 Astra

### 10.1 arXiv 逐条核实（控制查询通过；5 条我亲自 curl 复核）

**你列的 10 条全部存在**，包括看起来最像编造的那两条。

| 你给的名字 | arXiv id | 日期 | 它实际做了什么（摘要口径） |
|---|---|---|---|
| OSWorld-Human | **2506.16042v2** | 2025-06-19 / v2 2026-05-18 | CUA 的**时间效率**基准：大模型的规划/反思/评判调用主导延迟，轨迹后段的步骤可比前段慢 **3×**；人工标注最短轨迹后，16 个 agent 里最好的也要 **2.7–4.3× 于必要步数** |
| Efficient GUI Agents（survey） | **2609.02309v1** | 2026-09-02 | 系统性 survey：observation / context-memory / action / planner-runtime 四类效率；点名开放问题含 **verifier 成本的诚实核算**与**跨基准可比性** |
| Step-level Optimization for Efficient Computer-use Agents | **2604.27151v1** | 2026-04-29 | 事件驱动的逐步 cascade：小策略默认，**Stuck Monitor**（进度退化）与 **Milestone Monitor**（语义检查点）触发升级到强模型；免重训 |
| Screenshots or Tools? | **2608.03327v2** | 2026-08-04 | 同 harness 下 screenshot vs MCP 工具；丢弃工具调用后的冗余截图 + 图像历史减半 ⇒ input token 降约 ⅓；重训压缩版 **37.8% vs 33.0%** 且只花 **53%** input 成本 |
| EvoRoute | **2601.02695v1** | 2026-01-06 | 提出 **Agent System Trilemma**（性能 / 成本 / 延迟）；逐步选 Pareto 最优 backbone；GAIA + BrowseComp+ 上称成本降 **80%**、延迟降 **>70%** |
| TwinRouterBench | **2605.18859v2** | 2026-05-14 | agentic routing 的静态 + 动态双轨基准；静态轨 970 个 **router-visible prefix**；动态轨以真实 API 花费计成本 |
| Capability Frontier | **2606.26836v1** | 2026-06-25 | 见 §6.2 |
| LLMRouterBench | **2601.07206v1** | 2026-01-12 | 见 §6.2 |
| How Much of the Routing Gap Is Real | **2607.03436v2** | 2026-07-03 | 见 §6.2 |
| Opportunity Is Not Realizability | **2608.08265v1** | 2026-08-08 | 见 §6.2 |

**额外找到、与本 framing 正面相撞的三篇**：

| 论文 | id | 日期 | 为什么撞 |
|---|---|---|---|
| **AgentCARD**（Specialize Roles, Mix Deployments: Pushing the Cost-Accuracy Frontier of LLM Agent Teams） | 2606.20629v1 | 2026-05-28 | 角色分解 harness + **统一 API/自托管成本模型** + **Pareto 前沿分析** + Shapley 瓶颈诊断；异质团队以 **12× 更低的每任务成本**追平最强同质团队。**这是"配置扫描 + 成本前沿"这一格最接近的占位者** |
| **Holistic Agent Leaderboard (HAL)** | 2510.11977v1 | 2025-10-13 | 21,730 次 rollout，**models × scaffolds × benchmarks 三维**显式分析。"诚实地扫配置空间并报告"这件事的基础设施论文 |
| **3D Optimization for AI Inference Scaling** | 2510.18905v3 | 2025-10-21 | 把推理缩放形式化为 accuracy/cost/latency 三目标 MOO；但是**仿真**（9 个模拟 LLM），不是真 agent run |

⇒ **"把 agent 当 operating family 而不是一个点"这个 framing 本身已有人占**（EvoRoute 的 trilemma、
AgentCARD 的 cost-accuracy frontier、HAL 的三维扫描、2609.02309 survey 的整个论点）。
**2609.02309 的第一句话几乎就是你的 motivation**：*"the field still reports progress primarily
through task success. We argue that practical deployment depends equally on efficiency."*

### 10.2 GPT-6 Astra（公开材料，非 arXiv；逐条标注可信度）

| 事实 | 核实结果 | 可信度 |
|---|---|---|
| OSWorld 2.0 **72.6% @ ~40 min** vs GPT-5.6 Sol **65.7% @ ~75 min** | **数字逐字确认**，但基准行标的是 **"OSWorld 2.0 (v2026.08.08, **offline set, partial score**)"**，且全表脚注 **"Evaluation scores are the maximum at any effort"** | OFFICIAL-BLOG |
| per-token 更贵但部分任务 estimated API cost/task 更低 | **确认**，且有逐基准数字（Terminal-Bench Science 低 ~27%、Terminal-Bench 4.0 低 ~9%、BenchCAD 低 ~43%、GPQA 低 ~37%）⚠️ **没有一条是 computer-use 基准**；OSWorld 没有 $/task | OFFICIAL-BLOG |
| Fast mode = 最多 2× speed / 2× price | 价格 2× **确认**；速度 OpenAI 自己两处不一致（blog "up to 2x"，docs "up to 2.5×"）；**Bedrock 上不提供 Fast** | OFFICIAL-BLOG + OFFICIAL-DOCS |
| reasoning effort 可在对话中动态改且保持 cache | **确认**：通过 `configuration_update` **input item**（不是请求参数），请求级 `reasoning.effort` 保持不变以保留 prefix cache；不得相邻两条、不得与 auto-compaction 同用；Astra 档位 `low/medium/high/xhigh/max`，**不支持 `none`（400）** | OFFICIAL-DOCS |
| CU safety / AutoReview 自身是 cost–latency–risk 权衡 | **部分确认**：AutoReview 是 **Codex 的协议**，不是通用 API 特性；内部 CU safety benchmark Astra 2.4% → 加 AutoReview **1.8%**（Sol 22.0% → 4.3%）；⚠️ **成本与延迟代价从未被量化**；⚠️ *"In the API, the task will stop"* —— 对长程 web agent 测量是真实混淆项 | OFFICIAL-DOCS |

**价格（OFFICIAL-DOCS，USD / 1M token，short context）**：
`gpt-6-astra` **$10 in / $1 cached in / $50 out**；`gpt-5.6-sol` $4 / $0.40 / $20；
`gpt-5.6-terra`（本项目的 B5）**$2 / $0.20 / $12**。Bedrock global CRIS 与 OpenAI 直连同价；
in-region 加 10%。Fast mode = 2×；Batch/Flex = 50%。**未找到任何 computer-use 专门附加费。**

**与本项目相关的三条集成事实**：
1. **Astra 的 tool calling 需要 Responses API**（"GPT-6 Astra supports Chat Completions, but its
   tool calling requires Responses"）。
2. **logprobs 在 Astra 上任何配置都拿不到**（migration guide 要求移除 `logprobs`/`top_logprobs`，
   且 Astra 不能设 `reasoning.effort: none`）。这与 B5 的现状一致，**不是新问题**。
3. Bedrock GA 2026-09-08，id `openai.gpt-6-astra` / `us.` / `global.` 前缀；
   `bedrock-runtime` 上**无 server-side tool use**。

### 10.3 "一个 model 不是一个 operating point，而是一个 configurable operating surface" —— 值不值得当 motivation

**值得，但它不是我们的 novelty，而是我们的 motivation 的外部背书。** 理由三条：

- ✅ **它现在是厂商自己在说的话**：同一个 `gpt-6-astra` 有 5 档 reasoning effort、可中途切换且保 cache、
  有 Fast mode（2× 价换 ≤2–2.5× 速）、有可开关的 safety 层。这是"model = surface"最硬的现实证据，
  而且**是 2026-09 才出现的**——早于此的论文没法这么论证。
- ⚠️ **但 OpenAI 自己把这个 surface 藏起来了**：benchmark 表标注 "maximum at any effort"，
  **不发布 effort → accuracy/cost 曲线**。唯一的定量曲线来自 Artificial Analysis（第三方），
  且其 Intelligence Index 分数与 OpenAI 自己引用的版本号对不上（AA 说 53，OpenAI 表里 61.2 / v4.1.1）。
  ⇒ **"厂商把模型做成 surface，却只发布 surface 上的一个点"本身就是一条可写的观察**，
  而且它正好支撑"需要独立的 operating-envelope 测量"这个立论。
- ⛔ **不能**用 Astra 的 coding/science 上的 cost-per-task 下降去论证 computer-use 上的下降——
  OpenAI 一条 CU 的 $/task 都没发。

---

## 11. Q10 — GPT-6 Astra anchor：可行性与成本估算（只估不点火）

### 11.1 最小改动可行性

B5 的 config（`configs/exp_v2_B5_dom_classifieds.yaml`，只在 E: 镜像上）已经**因为同一族限制**
走了正确的路：
- `use_tool_calling: true` 但 `structured_output: "response_format"` —— 因为该 proxy 上
  `tools` + `reasoning_effort` 直接 400，官方的解法（`/v1/responses`）**该 proxy 不支持**
  （它白名单顶层字段并静默丢弃未知字段，已用 `reasoning_effort:"ZZZ_INVALID"` 与
  `totally_bogus_param_xyz:123` 两个探针证明，两者都回 200）。
- `logprobs_unavailable: true` —— 已声明，不是静默空列。

⇒ **Astra 的两个已知集成障碍（Responses-only tool calling、无 logprobs）在 B5 这条路径上已经被绕过了。**
最小改动 = **改 `api_name` 为 `global.openai.gpt-6-astra` + 改 `cost_api` 两个费率 + 新建 6 个 mode 的 config**。

**唯一的 gating 未知**：该 AWS proxy 的模型注册表**有没有** `gpt-6-astra`。
这条无法从本地判定，必须探。代价 = **1 个 episode 的 smoke，约 $0.66**（估算见下）。
⚠️ 台账 §471.2 的教训：**价格要从 registry 读，不要从文档读**——Astra 在这个 proxy 上的实际计价
可能与 OpenAI 公开价不同，smoke 时一并读回。

### 11.2 成本估算（基于 B5 的实测 token profile，非推测）

B5 cls 的逐 mode 实测（从 `total_input_cost_usd / 0.002 × 1000` 与 `total_output_cost_usd / 0.012 × 1000`
反解，和已落盘的 `total_tokens` 逐位对上）：

| mode | n | SR% | steps | input tok/ep | output tok/ep | terra 实付 $/ep |
|---|---:|---:|---:|---:|---:|---:|
| DOM | 452 | 24.12 | 16.2 | 52,105 | 2,799 | 0.1378 |
| SoM | 224 | **37.05** | 15.3 | 67,875 | 2,614 | 0.1671 |
| Vision | 360 | 10.28 | 27.6 | 85,045 | 4,616 | 0.2260 |
| phantom(3 合) | 672 | 22.92 | 17.3 | 55,855 | 2,909 | 0.1466 |

**224 题 classifieds，单 condition 的 Astra 账（$10/1M in、$50/1M out，无缓存）**：

| mode | input 成本 | output 成本 | **合计** | 相对 terra 实付 |
|---|---:|---:|---:|---:|
| DOM | $116.7 | $31.3 | **≈ $148** | 4.8× |
| SoM | $152.0 | $29.3 | **≈ $181** | 4.8× |
| Vision | $190.5 | $51.7 | **≈ $242** | 4.8× |

**带 prompt cache 的乐观档**（假设 50% input 命中 $1/1M 的 cached 档——web agent 每步的 system prompt +
intent 是稳定前缀，但 AXTree/截图每步都变，50% 是乐观假设）：DOM ≈ **$96**、SoM ≈ **$114**。

**⚠️ 一条会让上表全部偏低的风险**：Astra 是 reasoning model，**reasoning token 按 output 计费（$50/1M）**。
本项目全程 `tokens.thinking ≡ 0`（proxy 不回 reasoning token），
所以上面的 output 估算**很可能只覆盖了可见输出**。若 reasoning token 是可见 output 的 2–3 倍，
DOM 的合计会从 $148 涨到 **$210–270**。**这个不确定性必须在 smoke 时实测（对账 proxy 回的 usage 字段），
不能靠估。**

### 11.3 它能买到什么，以及值不值

| 选项 | 代价（估） | 买到什么 | 裁决 |
|---|---:|---|---|
| **A. 1 episode smoke** | **$0.66** + 半天工 | 回答三个 gating 问题：registry 有没有 astra / 实际费率 / usage 里有没有 reasoning token | ⭐⭐ **必做，且应在任何其他决定之前做** |
| **B. 224 题 × 1 mode（SoM）单点 anchor** | **$114–181**（+reasoning 风险到 ~$270） | **一个点**。能说"新一代模型把 cls 的最好操作点从 37.1% 推到 X%"，但那是 model-generation shift 的**单点**，不是 surface | ⚠️ **性价比低**。它证明不了 framing，只增加一个 baseline |
| **C. reasoning-effort 曲线：预注册子集 × 5 档** | 60 题 × 5 档 ≈ **$150–300**（DOM）/ $190–380（SoM） | **本项目第一条 within-model operating curve**——第六个维度从"完全没有"变成"有一条曲线" | ⭐⭐⭐ **这才是唯一值得花的那笔钱** |
| D. 224 题 × 5 档 × 多 mode | **$740–2,700+** | 完整 surface | ⛔ 超出合理预算，且 §3 的其他空维度仍然空着 |

**关于 C 的三条设计要求（写在花钱之前）**：
1. 60 题子集必须**预注册**（冻结 task id + 抽样脚本 + 判据，照 `budget-router-prospective-20260909` 的 freeze+tag 模式），
   否则它就是又一次 in-sample 选点。
2. **必须同时记录每档的实际 usage**（含 reasoning token），因为"effort 曲线"的横轴就是花了多少算力，
   而这个项目从来没有过这个量。
3. **SR 与 Phase-1 的任何数字不可并排**：不同 model、不同 API 形状、且子集不是 scored universe。
   写进预注册。

⚠️ **预算是你的决定**：memory 里的 $546 是**需求估算不是余额**。C 档的下限 $150 与上限 $380 相差 2.5×，
差别全在 reasoning token 上——所以 **A 必须先做**。

---

## 12. 裁决

### 12.1 你写的那个 framing：判死（给理由，不是给结论）

> "Can we reorganize our experiments into *mapping and decomposing the operating frontier of stateful
> web/computer-use agents*?"

**不能，按这个样子不能。** 三条独立的理由，每条单独都够：

1. **我们自己已经做过了。** 台账 **§377「Cross-object Pareto 分解 — 表征轴 vs 模型轴的 oracle headroom
   （18 臂同平面）」** 与 `scripts/analysis/cross_object_pareto.py` 已经产出
   27.23% → 43.30%（+16.07pp）vs 31.25%（+4.02pp）→ joint 46.43%，并已记下
   "表征轴携带的 oracle headroom 约为模型轴的 4–5×"。**本次我在不看它的情况下独立复算，数字一位不差。**
   再加上 `multimetric_pareto` / `per_mode_four_dimension_profile` / `outcome_efficiency` /
   `routing_ceiling` / `rule_routing_pareto`，"画并分解前沿"这件事**在本仓库里是已完成状态**。
   ⇒ 重新组织成一篇"map the frontier"的论文 = **把旧结果换名字**。
2. **外部已占满。** 2609.02309（survey，2026-09-02）的第一段论点就是你的 motivation；
   2606.26836 定义了 Capability Frontier；2606.20629（AgentCARD）已经做了"统一成本模型 + Pareto 前沿 +
   瓶颈归因"；2510.11977（HAL）已经做了 models × scaffolds × benchmarks 三维扫描；
   EvoRoute 已经命名了 performance–cost–latency trilemma。
3. **我们的 envelope 是空的。** §3 实测：五个设计维度里只有 **model（3 档）× perception（6 档）**
   被真正点过火；scaffold 全 False、router 全 off、max_steps 全 30、reasoning compute 根本没有旋钮。
   一篇 "operating envelope" 论文交出去，图上会有四维是空的。

**若只做到"换名字"，这条线到此为止。**

### 12.2 但同一批证据里有一篇更强的论文，而且它成立的原因是具体的

**不是"我们测出了前沿"，而是"这些轴在有状态真实浏览器里还没被识别出来，我们有证据，而且我们量出了它有多严重"。**

它成立的真正原因，是四条**没有任何外部论文能提供**的实测：

1. **四条资源轴在真实部署里塌成一条或不可识别**（§2）：
   energy 与 latency r = 0.9935 且 B0 根本没有；stored cost 与 tokens r = 1.000；
   本地 $ 换个 basis 就有 2/4 cell 的最便宜 mode 变了；latency 的跨 mode 极差只有 rerun band 的 **0.84–0.87×**，
   去掉容器后 4/8 cell 的最快 mode 变了。
   **上面每一篇 prior work 的 cost 轴都是一张静态 prompt 的发票。我们这里 cost 是一个建模选择，而且建错了。**
2. **前沿抗不住噪声，而噪声是在有状态环境里实测的**（§4.2、§8）：
   200 条 nominal Pareto 支配只有 **24%** 扛住 task bootstrap、**30%** 扛住实测 rerun band；
   两个 cell 的前沿完全退化成 6/6。2607.03436 量的是静态基准上的 single-draw 噪声；
   **在一个每次重跑都会带环境漂移的有状态浏览器里量这件事，没有第二家做过。**
3. **唯一能精确评估的那条轴，在弱 cell 上也不可识别**（§7.2）：
   成功 episode 里从未显式 finish 的比例从 **3.3%（cls_B0）到 88%（red_B2）**，
   ⇒ 截断前沿只能写成区间，而区间在弱 cell 上宽过整条曲线。
   **这条同时打在我们自己唯一的正结果上，正因如此它才可信。**
4. **一条真正正交、且不随能力单调的第六轴**（§9）：
   B2（最弱，1.19%/episode）与 B5（最强，0.88%/episode）的 session 破坏率不可区分（p=0.25），
   两者都显著高于 B0/B1（p ≤ 1.6e-04）。**没有任何已发表的 CU 成本模型给不可逆状态破坏定过价。**

**这篇论文的贡献类型是 measurement validity，不是 method。** 而这正好匹配这个项目的实际资产：
它最强的东西一直是**口径纪律**（noise floor inventory、estimand 审计、retraction 记录），不是新方法。

### 12.3 最强 framing

> ### **"Your Frontier Is Not Identified: What It Takes to Measure the Cost–Success Trade-off of Stateful Computer-Use Agents"**
>
> 在 7,722 个 paper-grade episode（2 site × 3 backbone × 6 perception mode，全部逐步 telemetry）上，
> 我们把 CU agent 的 cost–success 前沿当作一个**测量问题**而不是一个优化问题来处理，
> 并证明：在有状态的真实浏览器部署里，文献默认可用的五条资源轴中有四条**不被识别**，
> 而常规做法（single-run mean 上画 Pareto）会系统性地报告**不存在的操作点**。
> 我们给出三层抗噪支配口径、一条被忽略的正交风险轴，以及一套任何 CU 效率论文都该过的**最小可识别性清单**。

### 12.4 三个可替代 framing

| # | framing | 真正的 novelty | 被谁占掉 | 强度 |
|---|---|---|---|---|
| **Alt-1** | **"How Much of a Computer-Use Frontier Survives a Rerun?"** —— 把 §4.2 + §8 单独做成一篇噪声论文 | 在**有状态环境**里实测 rerun band（含环境漂移），并用它重判 Pareto；arm-count-matched 对照（加一臂 ≈ 重跑一臂） | 2607.03436 占了"routing gap 里有多少是噪声"，但它是静态基准 + 重采样；2608.08265 占了 selection-valid 诊断 | ⭐⭐⭐ 最干净、最难反驳，但**单薄**（可能只够 workshop / short paper） |
| **Alt-2** | **"The Unpriced Tail: Irreversible State Damage as a Missing Axis in Computer-Use Cost Models"** —— §9 扩写 | 风险**不随能力单调**（B2≈B5，p=0.25）；blast-radius 记账；七个死字段的 retraction 史作为方法论警示 | InferAct（2407.11843）占了"执行前拦截不可逆动作"；AVR（2603.12823）占了"高风险 action 升级到最强模型" | ⭐⭐ n=16 受害 / 9 波、单站点、肇事集不打标 ⇒ 现在只够**一节**，要成篇必须专门 fire |
| **Alt-3** | **"Buy Steps, Not Representations: A Budget-Axis Account of Computer-Use Agent Economics"** —— §5 + §7 | 失败吃掉 95.2% 支出（中位，36 condition）；每步成本近似平（1.04–1.12×）；表征 oracle headroom 是模型轴的 4× 但 attainable 是 0；预注册 budget router 2/2 PASS | 2604.27151（Step-level Optimization）占了 step-level cascade；2511.17006（Budget-Aware Tool Use）占了"抬预算本身无效"；2608.12150 占了"排名随预算反转" | ⭐⭐ **前提有风险**：§7.2 的识别问题正好打在 2/2 PASS 那个结果上。**必须先做 §13 的 A0，否则不要立这个 framing** |

### 12.5 用已有数据能支持的 claim（每条都已实测，可直接进论文）

1. 在同 backbone 内，**stored cost 与 token 数的 episode 级 r = 1.000（36 condition）** ⇒ cost 轴不是独立轴。
2. **energy 与 latency r = 0.9935**，功率恒 66.3 W，B0 无 energy ⇒ energy 轴不存在。
3. **latency 的跨 mode 极差 / rerun band = 0.84–0.87×** ⇒ latency 不能分辨 mode；去掉容器后 4/8 cell 最快 mode 变了。
4. 本地美元的 **basis 变更让 2/4 cell 的最便宜 mode 翻转**；而 `cost_unit_basis` 字段声明的 basis
   **不是该字段存储的量**（差 88×）。
5. **200 条 nominal Pareto 支配只有 24%（bootstrap）/ 30%（rerun band）存活**；两个 cell 前沿完全退化。
6. **表征 oracle headroom（+16.07 / +12.20pp）是模型 oracle headroom（+4.02 / +2.93pp）的约 4 倍**；
   两个 4B 模型的独解合计 < 3pp；portfolio 记账要 6.4–6.7× 预算。
7. **matched step budget 下，18 个配置里最优的仍然是最优的单配置本身**（两个 site 都是）。
8. **失败 episode 吃掉 95.2% 的支出**（中位，36 condition，范围 86.9–99.6%）；
   **最后一次页面变化之后的支出占 14.1% / 27.3% / 30.4%**（B0/B1/B2，cls）。
9. **每步成本近似恒定**（step30/step1 = 1.04–1.12×）⇒ 步数即预算。
10. **截断前沿只在强 cell 上被识别**：未显式 finish 的成功占比 3.3% → 88%。
11. **不可逆 session 破坏不随能力单调**：B2 1.19% vs B5 0.88%（p=0.25），两者 ≫ B0/B1（p ≤ 1.6e-04）。
12. learned which-mode routing **0/6、0/26**；triage 的最强单特征是基准自带的人工难度标注。

### 12.6 必须新增的最小分析（全部 0 新 compute）

见 §13。

### 12.7 可选新实验

| 实验 | 代价 | 换到什么 |
|---|---|---|
| **Astra 1-episode smoke** | $0.66 | 三个 gating 答案（registry / 实际费率 / reasoning token 计不计） |
| **Astra reasoning-effort 曲线，预注册 60 题 × 5 档** | $150–380 | 第六维度从"空"变成"一条曲线"；也是"model = surface"的唯一自测证据 |
| **B0 / B1 各加一个 replicate 臂（reddit）** | 2 个 condition-run | 把 rerun band 从"只有 B0·cls 实测、其余借用"变成两个 site 都有。**这是 Alt-1 成篇的前提** |
| **classifieds × B2 的重复 run（风险轴）** | 2–3 个 condition-run | 把 9 个波变成可做统计的风险率 |
| ⛔ **不做**：任何新的 which-mode learned router | — | 0/6 + 0/26 已决定性 |

### 12.8 五张主图

1. **Fig 1 · 轴塌缩图**：4 个小面板 —— (energy, latency) r=0.9935；(cost, tokens) r=1.000；
   latency 跨 mode 极差 vs rerun band（0.84×）；本地 $ 在两种 basis 下的 mode 排序交叉图。
   **一张图说清"你以为的五条轴其实是两条"**。
2. **Fig 2 · 抗噪前沿**：6 个 cell × (SR, steps) 散点，每个 mode 带 bootstrap 椭圆 + rerun band 方框；
   nominal 前沿用虚线、robust 前沿用实线。`cls_B2` 与 `red_B1` 的实线**退化成整个点云的包络**。
3. **Fig 3 · Lever attribution**：横轴 = 预算倍数（步），纵轴 = SR。四条：best single（一个点）、
   perception portfolio（6.7×, +16.07pp）、model portfolio（4.1×, +4.02pp）、
   step-budget 截断曲线（0.18×→1.0×，**带识别区间阴影**）。**oracle 记账的点用空心画，标注"不可达"**。
4. **Fig 4 · Trajectory economics**：上 = 累计支出 vs step（成功/失败分组，带 95% 区间）；
   下 = 失败支出份额的 36-condition 分布（中位 95.2%）+ 最后一次页面变化后的支出份额。
5. **Fig 5 · 风险轴**：per-(model, site) 的 session-loss 事件率 forest plot（Fisher CI），
   横轴按 SR 排序 —— **视觉上直接看到风险不随能力单调**（B2 与 B5 并排在顶部，中间隔着 B0/B1 的 0）。

### 12.9 Abstract-level contribution statement

> We show that the cost–success operating frontier of a stateful computer-use agent is, in current practice,
> **not an identified object**. On 7,722 paper-grade episodes spanning two live web environments, three
> backbones and six perception modes — with full per-step telemetry — we audit the five resource axes the
> literature plots against success and find that four of them collapse or fail to identify: energy is
> wall-clock in other units (r=0.99, and absent entirely on API-served models), dollar cost is token count
> rescaled (r=1.00) with a locally-served basis whose per-mode ordering flips under a defensible change of
> estimand, latency resolves modes below its own measured rerun band (0.84×) and reverses under container
> accounting, and the one exactly-priced axis — step budget — is identified only where the agent explicitly
> terminates, which ranges from 97% to 12% of successes across cells. Applying three progressively stricter
> dominance criteria, only **24–30% of 200 nominal Pareto dominations survive**, and two of six cells have
> no separable operating points at all. Decomposing the frontier by design lever at matched resource budget,
> perception diversity carries **4×** the oracle headroom of model diversity (+16.1pp vs +4.0pp) yet neither
> yields any attainable gain (0/6 and 0/26 learned policies Pareto-dominate the trivial fixed policy), while
> a portfolio that actually realises the oracle costs 6.7× the budget. Finally we identify a sixth,
> genuinely orthogonal axis that no published computer-use cost model prices: irreversible environment
> damage, whose rate is **not monotone in capability** — the weakest and the strongest backbone are
> statistically indistinguishable (p=0.25) and both far exceed the mid-tier ones (p≤1.6e-04). We release a
> minimal identifiability checklist that any computer-use efficiency claim should clear before a frontier
> is plotted.

### 12.10 最终裁决：CU-Pareto vs trajectory-routing

| 判据 | CU-Pareto（**按 §12.3 重述后**） | trajectory-routing（前一份 audit 的 v3.7） |
|---|---|---|
| **更强？** | ✅ 贡献类型是 measurement validity，**不需要新 fire 就完整**；负结果在这里是**产物**而不是"我们没做出来" | ⚠️ 需要 Gate 1/2 的 RCT（~4 个 condition-run + 每 run 1–1.5h 重置开销），且三条不受 terminal-label 污染的现存证据**方向为负** |
| **更能利用已有工作量？** | ✅ 直接吃掉 noise_floor / estimand 审计 / 36 condition 全量 / 7,722 份逐步日志 / 全部 retraction 史 —— 而这些在 routing framing 下都只是**脚注** | ⚠️ 已有的 12 条负结果里只有 2–3 条进得了正文 |
| **更抗最新 prior work？** | ✅ 上面 10 篇 + 3 篇近邻里，**没有一篇**在有状态真实浏览器上做轴可识别性；2609.02309 survey 甚至**点名**"verifier 成本的诚实核算与跨基准可比性"是开放问题 | ⛔ 2604.27151（step-level cascade）、AVR、COTA、2609.02057 四篇把 actuator 选择这一格压得很紧，半衰期 6 个月 |
| **风险** | 是"negative/methods"论文，**venue 选择更挑**（更适合 benchmark/eval track 或 reproducibility track，而不是 main track 的 method slot） | 风险在**钱和时间**：要点火，而点火结果很可能是阴性 |

> ### **裁决：按你原样写 = 换名字，判死。按 §12.3 重述 = 成立，且比 trajectory-routing 更强。**
>
> 它成立的真正原因**不是**"把 agent 看成 operating family"——那句话 2609.02309 的 survey 已经替你说了，
> 而且说得更早。它成立是因为：**你手上恰好有唯一一份能证明"这些轴还不能被测量"的数据**——
> 三个 backbone 的三种成本口径、两个 site 的容器差异、三个实测 replicate 臂、
> 7,722 条逐步支出曲线、以及一份把自己七个字段判死的 retraction 记录。
> 换句话说：**这个项目最有价值的资产从来不是它的前沿，是它对自己前沿的不信任。**

---

## 13. 最小分析计划（全部 0 新 compute，按"最便宜地杀死或点亮一个假设"排序）

| # | 动作 | 它能杀死/点亮什么 | 预估 |
|---|---|---|---|
| **A0** | ⭐⭐ **在 shop_B1 的两条 held-out 臂上统计 `agent_finished=False` 的成功占比**，并把 §4.4 的 2/2 PASS 重算成识别区间 | 若比例与 red_B1 的 50% 同量级，**1.9pp 的 learned-vs-fixed 差距可能整个落在区间内** ⇒ Alt-3 当场死、主 framing 的 §7 要改写 | 1–2h |
| A1 | 把 `cost_unit_basis` 的 mislabel 提一个 B-number 进 master bug catalog，并扫全部消费 `total_billed_cost_usd` 的 aggregator | 决定有多少已落盘的"成本"数字要带更正声明 | 2h |
| A2 | 把 §4.2 的三层抗噪口径做成 `scripts/analysis/robust_pareto.py`（白名单取 run，逐 cell，shuffle-null 逐档） | 把本文件的 24%/30% 从探针升级成 canonical 产物 | 4h |
| A3 | 把 §9 的风险率做成 canonical 产物（含 B5 对照与波级聚类），并逐 (model, site, mode) 报 blast radius | 把"不随能力单调"从本文件的一条实测升级成可引用的图 | 3h |
| A4 | 逐 cell 重算 §7.1 的失败支出份额与死尾巴，纳入 WA / shopping（在 A100 回拉完成后） | 把 95.2% 的分母从 36 扩到全量，并与 §505.21 的 92% 对账 | 3h |
| A5 | Astra 1-episode smoke（$0.66），只回答 registry / 费率 / reasoning-token 三问 | 决定 §11.3 的 C 档到底是 $150 还是 $380 | 半天 + $0.66 |
| ⛔ | **不做**：任何 which-mode learned router 重试；任何消费七个死字段的分析；任何跨 backbone 的美元同轴图 | — | — |

---

## 14. 本文件的证据出处

- **本次实测（白名单口径，脚本在 `C:\workspace\_p79_audit_scratch\`）**：
  `cu_inv.py`（36 condition 全量聚合）· `cu_pertask.py`（7,722 行 per-task 表）·
  `cu_frontier.py`（nominal + bootstrap robust）· `cu_frontier2.py`（rerun-band robust + lever attribution）·
  `cu_18arm.py`（18 臂跨模型前沿）· `extract_steps.py` + `cu_econ.py`（逐步支出经济学）·
  `cu_trunc_audit.py` + `cu_trunc_band.py`（截断识别区间）· `risk_scan.py` / `b5_scan.py` · `meta_scan.py`（配置空间覆盖度）
- **项目既有产物（本文件只引不复述）**：`docs/analysis/cross_sites/{noise_floor_inventory,replicate_metric_noise,
  multimetric_pareto,latency_decomposition,energy_carbon_audit,local_cost_estimand_audit,outcome_efficiency,
  routing_ceiling,rule_routing_pareto,router_triage_learnability,cost_per_mode}.md` ·
  `docs/analysis/_data_quality_audit.md` · `final_dissertation/THESIS_ONE_SENTENCE.md`
- **代码事实**：`p79/experiment/types.py:230-256`（`cost_unit_basis` 的 enum 与注释）·
  `p79/experiment/metrics.py:860-877` · `configs/exp_v2_base.yaml:66-95`（本地成本常数 + energy profile）·
  E: 上的 `configs/exp_v2_B5_dom_classifieds.yaml`（B5 的三条 proxy 限制与探针记录）
- **E: 台账（§449+ 只在 E: 上）**：§377（cross-object Pareto，本文件独立复算一致）· §476.4（operating point 的措辞裁定）·
  §505.21（92% 失败支出 + 固定 cap 前沿）· §505.28 / §510.3 / §515.3（budget router 预注册与 2/2 PASS）· §463.2 · §473.3
- **外部文献**：§10.1 的 13 条 arXiv id，控制查询 `2307.13854` 通过；其中
  2606.26836 / 2608.08265 / 2604.27151 / 2609.02309 / 2506.16042 由本会话主循环亲自 curl 复核标题与摘要。
  GPT-6 Astra 事实与价格来自 openai.com 官方 blog / developers.openai.com docs /
  deploymentsafety.openai.com system card / docs.aws.amazon.com Bedrock model card，逐条标注可信度见 §10.2。
