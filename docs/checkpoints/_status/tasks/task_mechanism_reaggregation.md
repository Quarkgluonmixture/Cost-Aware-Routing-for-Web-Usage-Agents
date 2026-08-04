---
type: task
status: todo
horizon: post-submission
owner: jiaming
eta: "REALM 08-05 之后。数据与工具已就绪, 卡的是 framing 决定 (机制层 advisor 05-14 搁置, 08-03 解冻)"
---

# 机制层重新聚合 — 数据已跑完, 结论已变

## 一句话

DGX 上 24 个 cell 的 patching sweep **2026-08-03 19:13 全部跑完**
(`logs/mechanistic_canonical/`, marker `24/24, 18 ran, 6 skipped`),
而 `mechanism_evidence.json` 在 **01:25 (之后) 重生成时仍只读 5 月的 Myriad 目录**。
证据单的 t38/t39 因此报的是三个月前的数据。聚合脚本已于 2026-08-04 扩展并重跑,
产物已含新数据; **未进论文, 因为用不用是 framing 决定**。

## 三个发现, 按严重度

### 1. 🔴 峰层论证不成立 —— 曲线是平的, argmax 任意

t39 的 caption 论证内容特异性靠**层剖面**: "真实臂收敛峰在中层 L14, 打乱臂塌到边界层 L00"。

实测 `p2_psom_ptext_cls` 的收敛曲线: **峰值 0.171667 有 6 个层精确并列**
(L00/L24/L26/L27/L29/L30), 9 层落在 0.005 内, **整条曲线极差仅 0.0142**。

对照组的极差是 **0.059–0.093** (随机注入)。也就是说 —— **只有破坏性的随机注入产生了
一条有形状的曲线; 真实臂与打乱臂的曲线基本是平的**, 而 argmax 落在平坦曲线上等于随机取点。

聚合脚本现已并列输出 `peak_convergence_n_tied` 与 `convergence_spread`,
任何关于峰层的陈述必须对着这两列读。

### 2. 🔴 同配置重跑, 峰层 6 个里 5 个移动

逐字段核对过 config 相同 (模型/36层/24任务/step2/max_new_tokens/source-target/tier),
即**同一实验跑两次** (Myriad 2026-05 vs DGX 2026-08):

| site | arm | peak conv | conv layer | 并列层数 | 曲线极差 |
|---|---|---|---|---|---|
| cls | real | 0.188 → 0.172 | L14 → L30 | 1 → **6** | 0.023 / 0.014 |
| red | real | 0.188 → 0.217 | L09 → L13 | 1 → 1 | 0.012 / 0.038 |
| cls | task_shuffled | 0.157 → 0.156 | L00 → L35 | 1 → **7** | 0.007 / 0.007 |
| red | task_shuffled | 0.159 → 0.163 | L20 → L11 | 2 → 1 | 0.008 / 0.006 |
| cls | random_injection | 0.085 → 0.067 | L02 → L02 | 1 → 1 | 0.078 / 0.059 |
| red | random_injection | 0.100 → 0.083 | L03 → L02 | 1 → 1 | 0.093 / 0.076 |

> 这篇论文用「同配置重跑」打掉了效应量, 现在同一个方法打掉了自己的机制结论。
> 这不是弱点, 是自洽 —— 但必须写出来, 不能让 t39 继续用旧口径。

### 3. 🟢 一个从未报告的正面结果: 图像轴的位移最大

非随机臂里最大位移出现在**图像轴**: `som → dom` 位移 **0.475 (cls) / 0.390 (red)**,
远超 prompt-style 轴 (0.271 / 0.300) 与 text-format 轴 (0.333 / 0.293)。
**把 SoM 的隐状态贴进 DOM 的运行, 位移比任何纯文本轴都大** —— 与行为层
「能不能看见图这一刀最深」同向, 但这条从未被任何产物读过。

## 已就绪 / 待做

- ✅ `aggregate_mechanism_evidence.py` 已扩展: 读 `results/mechanistic/canonical/` 24 cell、
  按轴分组、同配置重跑对照、并列输出并列层数与曲线极差
- ✅ 产物已重跑 (`mechanism_evidence.{json,md}`)
- ✅ 工具口径已验证: 同一脚本在老数据上**逐位复现** t39 的六个数字
- ⬜ **t38/t39 的 caption 需改** —— 现在的层剖面论证站不住
- ⬜ 是否进论文 = framing 决定 (advisor 05-14 搁置 / 08-03 解冻, 学长可能不知道已解冻)
- ⬜ WA 无机制数据 (24 cell 全是 cls/red), 机制层不覆盖第二基准
