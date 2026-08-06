# 机制层 v2 计划 —— 对准转向后的论文（2026-08-06）

> 取代 `plan.md` 的实验清单部分。`plan.md` 写于 phantom-routing-space framing 下，
> 那个 framing 已被 REALM 提交版取代（`sections/` = complementarity → noise →
> upperbound → lowerbound → **gap** → threats → discussion）。
>
> 本文件回答一个问题：**在新 framing 下，机制层做什么才不是装饰。**

---

## §0 先说结论

论文自己在 `6_gap.tex` 列了「什么能改变答案」四条 —— graded evaluators / replicates
as norm / 第三个 workload / online cascade infra。**四条全是测量层与基础设施层，没有
一条是机制层的。** 这既是机会也是警告：机制层若做不出攻击那堵墙的东西，它在这篇论文
里就没有位置。

那堵墙（`6_gap.tex` 原文）：

> routing supervision is produced at the success rate, so routing is least learnable
> exactly where it would be most valuable; and the rows a router must learn from are
> the contested ones, which are the rows that flip between identical reruns.

一堵墙，两条边：**标签供给**（supply）与**行的不稳定**（noise）。机制层能碰的是后者，
以及绕过前者的一种可能路径。下面三个实验按 leverage 排序，M1 最可行，M2 upside 最大，
M3 工作量最小且最接得上论文既有文本。

---

## §1 先止损：两条已经站不住的旧结论

在做新东西之前，这两条必须先处理，否则新工作会建在坏地基上。

| | 状态 |
|---|---|
| **峰层论证** — t39 caption 的「真实臂收敛峰在中层 L14，打乱臂塌到边界层 L00」 | ❌ **不成立**。`p2_psom_ptext_cls` 峰值 0.171667 有 **6 个层精确并列**，9 层落在 0.005 内，整条曲线极差仅 **0.0142**；对照组（随机注入）极差 0.059–0.093。只有破坏性注入产生了有形状的曲线，真实臂与打乱臂基本是平的，argmax 落在平坦曲线上等于随机取点 |
| **同配置重跑的峰层稳定性** | ❌ **6 个里 5 个移动**（Myriad 2026-05 vs DGX 2026-08，逐字段核对 config 相同）。cls·real: L14→L30，并列层数 1→6 |

> 这篇论文用「同配置重跑」打掉了行为层的效应量，现在同一个方法打掉了自己的机制结论。
> 这不是弱点，是自洽 —— 但**必须写出来**，t38/t39 的 caption 不能继续用旧口径。
>
> **方法论后果（约束下面所有实验）**：机制层从此**不报 argmax 类的量**（峰层、最敏感层、
> 定位到第几层）。要报的是对 argmax 不敏感的量：**曲线极差、并列层数、效应的符号与量级、
> 以及跨重跑的一致性**。任何新实验都必须预注册「曲线极差低于多少判为无定位」。

### ⛔ 第三条：那个「未报告的正面结果」也不能用了（B-1966，2026-08-06 当天发现）

首版这里写了一条「从未报告的正面结果」：*图像轴位移最大，`som → dom` 0.475 (cls) /
0.390 (red)，超过 prompt-style 轴与 text-format 轴* —— 并说它对 argmax 不敏感所以在新
口径下存活。**撤回。**

核对 canonical 数字时先撞见一个症状：24 个 cell 只有 **22 个不同的 per_task payload**，
`p4_som_ptext_*`（声明 `source=som`）与 `p2_psom_ptext_*`（声明 `source=phantom_som`）
**逐位相同**。顺藤查到根因，三项同时成立：

1. `run_stage2b_continuation_pilot.py:403` —— `source_screenshot_path = str(screenshot_annotated)`，
   **不看 `args.source_mode`，无条件传标注图**；
2. 同文件 `source_text_payload_for` 把 `som` 与 `phantom_som` 同映射到 `source_som_marks`；
3. `build_mode_prompt_dispatch_table()` 里两者的 system prompt **逐字节相同**
   （md5 `1dcacec32c53`）—— 这是**设计如此**，因为 `phantom_som` ≡ SoM prompt **+ 跳过标注图**。

文本相同 + prompt 相同 + 图相同 ⇒ **`som` 与 `phantom_som` 在 source 侧完全不可区分**，
即 **`phantom_som` 被实现成了 `som`**。

**后果**：三条轴的 source **全都带图**，所以那个差异不是「图像轴 vs 文本轴」，而是
「同一个带图的 SoM 表征 → 三个不同 target」。**「图像轴」这个命名和解读不成立。**
更广地：任何以「有图 vs 无图」为分界的机制结论都无效，而 t39 的 "real arm"
声称 source 是 `phantom_som`（无图）实则带图 —— **论文表格的 mode 标注与实际计算不符**。

> **`p4_som_ptext` 不是浪费的一次跑，它是这个 bug 的探针。** 正因为它与 p2 只差
> `source_mode` 一个参数，逐位相同才把缺陷暴露出来；其余 p4 cell 因 target 不同而结果
> 不同，**掩盖了同一个缺陷**。这也是为什么修它的时候必须补一条「两个 mode 的 source
> 输入必须不同」的断言测试 —— 否则同类缺陷只能靠 24 选 2 的偶然碰撞发现。

⇒ **机制层现存的 patching 结论，正面的和负面的，现在都不可用。** 先修 B-1966 重跑，
才谈得上读数。这不改变 M1/M2/M3 的设计（三者都不依赖历史 patching 产物），但取消了
「有一条现成正面结果可以接」这个前提。

---

## §2 数据版图（决定什么能做、什么不能）

> **⚠️ 本节 2026-08-06 首版写错并已重写。** 首版称「WA observation artifacts = 0，
> 机制层无法覆盖第二基准」。那个 0 是在 **DGX 本地** `find` 出来的，而
> `sync_a100_results.sh` 自 2026-07-21 起对 `phase1/` 施加 `--exclude artifacts/`
> ＋ `--delete-excluded`，脚本注释原文：*"A100 is the source of truth and retains
> them"*。**0 是同步策略的产物，不是世界的属性。** 更正见下表。
>
> 教训与 AMENDMENT_10 §2 那次（裸 curl 不带 `storage_state` 去探 `require_login` 页面，
> 把 24 个 task 误判成「资源不存在」）**是同一个**，间隔不到一天。写下来的教训没有生效，
> 因为缺的不是知识而是动作：**测到 0，必须先问一次「为什么是 0」再往下写。**

实测于 2026-08-06（**A100 侧**，`condense-a100`）：

| 资产 | 规模 | 能支持什么 |
|---|---|---|
| **VWA `phase1/` artifacts（A100）** | **437,681 文件**；20+ condition 覆盖 **B0/B1/B2 × cls/red/shop × 6 modes**，每 condition 4,800–7,900 个 `observation_dom.txt` | M1/M2/M3 全部，且 cell 数足以做跨 cell 统计 |
| **WA `phase1/` artifacts（A100）** | **65,405 文件 / 28,159 个 `observation_dom.txt`**；**12 condition** = B0 + B1 × wa_reddit × 6 modes | **机制层可以覆盖第二基准** |
| 文本观测总量（VWA+WA） | **251,549 文件 / 0.84 GB**（均 3.5 KB） | 全量拉到 DGX 完全可行 |
| 截图等全量 artifacts | VWA 44 G + WA 3.9 G | 前期三个实验都不需要图，**不拉** |
| `results/mechanistic/_canonical_artifacts/`（DGX 已有） | 2 cells（B1·cls 225 + B1·red 206），8,586 step，**dom+som 成对**（0 缺失）+ SoM 标注图 symlink | M3 的现成配对基底；M1 pilot 的最快起点 |
| `results/repro_replicates/`（DGX 已有） | 2 个 clean replicate（B0·dom·cls + B0·vision·cls），7,082 dom 观测 | flip vs stable 的配对分析 |
| `scripts/analysis/b0_paired_idperturb_replay.py` | 成熟、已用于论文。读 cached artifacts，**不连 live 站点**；B0(proxy)+B1(local GPU) 双层；已实现 `permute_ids` / id-agnostic 的 `resolve`+`dsig` / mode_flip / consistency | M1 直接复用 |
| `p79/mechanistic/` | `activation_patching.py` / `extract_hidden_states.py` / `linear_probe.py` | M1 / M2 |

### 拉取策略：**必须落在 `phase1/` 之外**

文本观测已镜像到 **`results/mechanistic/_obs_mirror/{visualwebarena,webarena}/`**。

⚠️ **不能拉进 `results/*/phase1/`**：那棵树受 `--exclude artifacts/ + --delete-excluded`
管理，下一次 cron sync 会把拉下来的东西清干净。这正是 `repro_replicates/README.md`
记录过的坑（它把自己放在 `phase1/` 外面就是为了这个）。`_obs_mirror` 在 `mechanistic/`
下，cron 的 delete 策略够不到。

### ⚠️ 剩下的真实约束

**(1) A100 磁盘余量 60 G（已用 88%）**，而 fire 还要跑约 10 天。VWA `phase1/` 已占 44 G。
这不影响机制层（我们只读、且只拉 0.84 G 文本），但**是 fire 侧的独立风险**，需要有人盯。

**(2) 配对观测来自 som trajectory。** `_canonical_artifacts` 的两个 cell 都是 `B1_som_*` 的 run —— 同一次运行里
同时记录了 dom 与 som 两种观测。所以 dom 观测是「当时若用 dom 会看到什么」的**反事实
观测**，agent 实际走的是 som 轨迹。

- 对 **M3 是好事**：页面状态被天然控制住了，两种表征对的是同一个页面。
- 对 **M2 是限制**：`step_001` 之后两个 mode 的轨迹会分叉，dom 观测不再是 dom 真会看到的。
  **因此 M2 只能用 `step_000`**（轨迹尚未分叉，两 mode 起点相同）。

  这个限制恰好与我们想要的一致 —— routing 决策必须在**花钱之前**做出，所以本来就该用
  step_000。但理由要写对：是数据结构强制的，不是我们选的。

---

## §3 M1 — id-churn 的机制解释：把 noise floor 从现象降为可修的缺陷

**为什么这是最高 leverage**：noise floor（分辨阈 3.8–4.2pp）是这篇论文里一切效应量的
下限。论文测到了它，但把它当作仪器属性接受了。如果它的一个主要来源是**可定位、可修**的，
那么阈值会降，可分辨的效应量变小，routing 实验重新变得可做 —— 这是论文那四条之外的
**第五条「什么能改变答案」，且是唯一一条不需要改 benchmark、不需要加 workload 的**。

### 论文已给的地基（`3_noise.tex`）

> a replay that shuffles ids while holding the page fixed changes which element is
> chosen on **20.0% of B1 steps** and **12.5% of B0 steps**. Because churn is a
> property of the text payload and not of the prompt, it reaches exactly the two
> modes carrying an accessibility tree, DOM and DOM+sprompt. The modes carrying
> `[SOM_MARKS]` are keyed 1…K by position and are unaffected.

行为层已测；**为什么模型对无信息 token 敏感**是空的。

### 断言（可证伪）

> element-id token 的表征在「选哪个元素」的决策中承载了不该有的因果权重。把 id-shuffled
> 运行中某一层／某些位置的激活换成 id-fixed 运行的对应激活，能把决策拉回原选择，且恢复率
> 显著高于随机位置对照。

### 做法

- **基底选 B1（本地 Qwen3-VL-4B，temp=0）**，理由是论文自己给的：B1/B2 的 step 级
  **bit-identical（133/133）**，没有 serving 噪声污染因果测量。B0 的 hosted MoE 做不到这点，
  只能作为外部效度的第二层，不能做因果基底。
- 配对：同一 `step_000` obs，id-fixed vs `permute_ids` 打乱（保 role/name/bbox/行序）。
- activation patching：逐层把 id-shuffled 的 residual stream 换成 id-fixed 的，测决策恢复率。
- **报告口径（§1 的约束）**：不报峰层。报恢复率曲线的**极差**、**并列层数**、以及
  **同配置重跑的一致性**。预注册：若真实臂的曲线极差 < 随机位置对照的极差，判为**无定位**，
  按阴性结果写。

### 这条不主张什么

它**不增加标签供给**。它降低噪声，从而降低所需效应量。这两件事不能混为一谈 ——
`3_noise.tex` 已经演示过混淆估计量的后果（"A mean-difference threshold applied to a
set-difference gain would be arithmetic across estimands"）。

### 若成立，可操作的推论

论文自己已经指出 `[SOM_MARKS]` 按位置 1…K 编号因而免疫。那么 AXTree 侧改用位置编号
（而非 native node id）是一个**直接的、可测的降噪改动** —— 这是机制结论能落到工程上的
出口，也是这条实验真正的价值所在。

---

## §4 M2 — 跨 cell 迁移的 routability probe：直接攻击 supply obstruction

**攻击点**：论文的 bind 是「标签只在 agent 成功的地方产生，弱 cell 标签少」。
**如果 routability 信号能从标签富裕的 cell 迁移到标签贫瘠的 cell，这个 bind 就被绕开了。**

### 断言（可证伪）

> 从 `step_000` hidden state 训练的 probe，预测「本 mode 能否解此 task」，在**跨 cell**
> 迁移时 AUROC 显著 > 0.5，且 > 论文那个 zero-cost text rule 的 AUROC。

### 三条必须有的对照

| 对照 | 作用 |
|---|---|
| 随机 | 地板 |
| **论文的 zero-cost rule（读 task 文本）** | 真正的竞争者 —— probe 若打不过一个零成本文本规则，机制信号无增量价值 |
| **同 cell 内训练的 probe** | 天花板 —— 迁移损失多少 |

收益必须换算到论文的**同一 cost 轴**（9.5–30.6% 那条），否则无法与那五个 policy 比较。

### ⚠️ 这条最容易过度宣称，两处必须写死

1. **训练 probe 仍然需要标签。** 它攻击的是「标签必须来自**同一个** cell」，不是「不需要
   标签」。断言里不能出现 "label-free"。这条是永久的，不随数据量改变。
2. **cell 数取决于 normalize 多少，不是硬上限 2。** 首版这里写「只有 2 个 cell → 一组
   方向对 → 只能算 pilot」，那是基于错的数据版图（见 §2 更正）。A100 上有 20+ 个 VWA
   condition 与 12 个 WA condition 的观测，够覆盖论文那 8 个 cell。
   **现存的硬限制换成了两条更小的**：(a) `_canonical_artifacts` 目前只 normalize 了
   2 个 cell 的 **dom+som 成对**布局，扩到更多 cell 要跑
   `normalize_canonical_artifacts.py`；(b) 每个 cell 的标签数仍然受该 cell 的 SR 限制 ——
   这正是论文那堵墙，probe 绕不开它，只能试图跨 cell 搬运。

### 与论文 ρ=0.952 那条的关系

论文说 label supply 与 routable share 以 ρ=0.952 同向、mean gap 1.65pp，并指出非结构的
部分是 **routable share 随 per-mode rate 近似线性增长，而互相独立的 mode 应近二次**。
M2 若成立，等于说这条耦合可以被**跨 cell 的表征信号**部分绕开 —— 直接对话论文最锋利的
那一段。M3 则从另一侧解释这条耦合本身。

---

## §5 M3 — mode 不独立的表征证据：解释论文那个「线性而非二次」

**这是论文自己留下的、未解释的经验观察**（`6_gap.tex`）：

> What is not structural is that the routable share grows roughly *linearly* in the
> per-mode rate, where mutually independent modes would make it grow near-quadratically
> at these success levels.

论文观察到了 mode 不独立，但没有给出机制。

### 断言

> 不同 mode 的表征高度共享，且共享程度可预测它们成功集合的重叠度。

**关于「图像轴相似度最低」这个更强的版本**：首版把它写进了断言，理由是与位移结果
（0.475 / 0.390 > 文本轴）同向 —— **该理由已随 B-1966 作废**（§1 第三条：那三条轴的
source 全都带图，位移差异不是图像轴造成的）。

它仍然是一个**合理的先验**（行为层「能不能看见图这一刀最深」是独立证据），但现在
**M3 要独立检验它，不能引位移结果当支撑**。这反而让 M3 更有价值：它成了第一个
不受 B-1966 污染的图像轴证据来源，因为它读的是 observation 与表征本身，不经过
patching 那条坏掉的路径。

### 做法

- 直接吃 `_canonical_artifacts` 的 **8586 组 dom/som 配对观测** —— 同 task 同 step，
  页面状态天然受控，这是现成的配对表征对比，不需要新跑任何 episode。
- 逐层算跨 mode 表征相似度（CKA / 子空间主角）。
- 关联表征相似度与成功集合的 Jaccard 重叠。

### 工作量最小

⚠️ 若只用现成的 2 个 normalize 好的 cell，3 条轴上算不出统计显著性，那样它**只能作为
机制叙事支撑，不能作为独立 claim**。要成为独立 claim，得先把更多 cell 过一遍
`normalize_canonical_artifacts.py`（§2 更正后这是可行的，不再是数据不存在）。

它的价值在于把论文的一个 anomaly 从「我们观察到」变成「因为表征不独立」，并给出一个
可操作的推论：**要让 routing 有价值，得在表征上最不相似的轴（图像轴）拉开 mode，
而不是在 prompt-style 轴上** —— 这直接指导下一代 mode 设计。

---

## §6 算力分配

| 平台 | 状态（2026-08-06 实测） | 派什么活 |
|---|---|---|
| **DGX Spark** (`spark-9ea3`) | 环境齐、数据在本地、GPU 共享有争抢 | M1 pilot（先跑通）、M2 probe 训练（轻）、全部分析与聚合 |
| **Holistic AI Sparks** (`spark-9017` idle + `spark-97a6` mix) | `ssh sparks` 通；`main` 分区 wallclock **3 天**；⚠️ **`/clusterhome/jiaming` 是空的**（只有 Desktop/snap）—— 无 venv、无代码、无数据 | M1 全量 patching sweep（GPU 密集、按 task 天然可切片）、M3 hidden state 批量提取 |
| A100 | 正在跑 shop_b0_tail（~10 天）；**是 artifacts 的 source of truth**；磁盘只剩 60 G | **不跑计算**，只做只读 rsync（0.84 G 文本，已完成）。补 WA artifacts **不需要重跑、不需要排队** —— 数据一直都在 |

### 建议的顺序（不要先铺 Sparks）

Sparks 的启动成本是实打实的半天：装 venv + 传 8586 个观测 + 验证 GB10 上的 CUDA 与
sm_121 fallback。**先在 DGX 上把 M1 pilot 跑通、确认曲线极差不是平的**，再决定是否值得
铺到 Sparks 全量。否则可能花半天搭环境，只为跑出一条和 §1 一样平的曲线。

所有三个实验都是**离线**的（读 cached artifacts + 本地 GPU forward），不需要 live site，
因此 Sparks 的「VWA 连通性未验证」不构成障碍 —— 这一点是这批实验能上 Sparks 的前提。

---

## §7 优先级与判据

| | 实验 | 先决条件 | 判为失败的标准 |
|---|---|---|---|
| 1 | **M1** id-churn 机制 | 无（工具已就绪） | 真实臂曲线极差 < 随机位置对照 → 无定位，按阴性写 |
| 2 | **M3** 跨 mode 表征相似度 | 无（数据已就绪） | 表征相似度与成功集合重叠度无关联 → 「不独立」的机制解释不成立。（图像轴那条改为独立检验，**不得**引 B-1966 污染的位移结果当支撑） |
| 3 | **M2** 跨 cell probe | M3 提供的表征口径 | 跨 cell AUROC ≈ 0.5 或 ≤ zero-cost text rule → 无迁移价值 |
| 0 | **止损 a**：t38/t39 caption 改口径 | 无 | — （camera-ready 前必须做） |
| 0 | **止损 b**：修 B-1966 后重跑 patching | 无 | — （在此之前**任何** patching 数字都不可引用，正负皆然） |

**三条都预设了可以写成阴性。** 这是刻意的：这篇论文的说服力来自它敢报 negative result，
机制层如果只准备了「成立」的叙事，会和论文的气质相冲。
