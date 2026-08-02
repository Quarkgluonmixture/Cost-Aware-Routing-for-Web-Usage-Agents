---
type: conclusions
batch: D3
status: done
created: 2026-07-28
source: scratchpad/batches/D3.jsonl (219 条 MEASURED，§207.4–§311，2026-05-17 ~ 2026-06-02)
---

# 测量结论 D3 (§207.4–§311)

> **读法**：本文件是**聚合层**，不是逐条转写。每个主题给「当前值 / 演变 / 已作废 / caveats / 证据 / 原文片段」。
> **禁止在本文件内做算术**——所有数字按原 scope 并列。台账 §302 的线性分解已被后续 CORRECTION 判为
> category error（跨 model / modality / serving / perturbation 四个不可比维度），本文件不重犯。
> 标 `[聚合者推论]` 的才是我加的判断，其余全部来自台账原文。

---

## 1. B0 classifieds 六模式的 full-cohort SR

**当前值**（全部 B0 = Qwen3-VL-235B-A22B via AWS proxy / classifieds / n=224 / Gate-3 fresh 衬底，逐 run 并列，**不取平均、不排名以外的运算**）：

| mode | SR | run | § |
|---|---|---|---|
| dom | 15.18% | R31194 | §277 |
| dom | 17.4% (39/224) | R21557 | §297.1 §298.1 |
| som | 27.2% (SR=27.2%, 224/224 ep) | R11315（Gate 2 canary） | §273 |
| som | 30.4% (68/224) | R9725 | §283 |
| vision | 24.1% (54/224) | R24792（post-B-1860） | §290 §291 |
| vision | 25.0% (56/224) | R32024 | §302.1 |
| phantom_som | 15.6% (35/224) | R32031 | §304 |
| phantom_prompt | 19.6% (224 ep) | R14655 | §305 |
| phantom_text | 25.4% (17/67，PARTIAL 交集) | R2647 vs archive R19776 | §292 |
| 6-mode taxonomy 引用的 best-single | som 27.2% (61/224) | §306 未点名 run_id | §306 |

**演变**:
- som：§273 canary R11315 报 27.2%（**故意不 restart docker**，保 6 天退化衬底）→ §283 R9725 报 30.4%（其 diag digest 后被 §299 supersede）→ §306 taxonomy 用 27.2% (61/224) 作 best-single。
- vision：§285 R3671 pre-B-1860 SR 13.84%（被判 non-canonical 并 archive，coordinate-scaffold 主导）→ §290 R24792 post-fix 24.1% → §302.1 R32024 25.0%。
- phantom_text：§286 撤回过一次 "ptext≈som" 的说法（混 substrate + 混 N）；task 0-69 subset 同 Gate3 衬底重比得 dom 18.8 < ptext 27.5 < som 37.7。

**已作废**:
- §285 的 vision SR 13.84%（R3671）——该 run 被标 non-canonical 并 archive，只作 RCA 用。
- §283 R9725 的 diag 结论被 §299 supersede（SR 数字本身台账未撤）。
- §273 canary R11315 被 Gate 3（2026-05-23）archive 为 non-paper-grade，退为 methods/appendix 的 Gate 2 证据（§275）。
- §258 R9755 的 B0 dom cls 14.7% 被 §275 archive 为 non-paper-grade probe。

**caveats**（原文）:
- §273: "canary = probe-only, 后于 Gate 3 (2026-05-23) 被 archive 为 non-paper-grade, 退为 methods/appendix 的 Gate 2 证据; retry 安全网在位但未被动用 (退化 ~7s 从未破 30s) → retry 吸收能力生产未实测"。
- §283: "P14 实测 presence-only (8/8 success-hit FP, success 22.1% 也 fire) → failed P14=65 虚高需重度打折; coverage 71.8% 低是 ruleset dom-biased 的预期不是 bug"；并且 "≥3 个 benchmark-FP → 真实 SR ≥ 31.7%, 30.4% 是下界"。
- §286: "partial SR 对比必须 substrate + task-subset 双匹配"；task 0-69 是 easy subset，"都高于各自全 224"。
- §292: PARTIAL，"全量待 R2647 完成复算"。
- §304 数据源经实证核对："A100 condition_summary success_rate=0.15625 与本地 diag 一致 + observation_mode=phantom_som + 中断版 R14849 已 archive 未混"。

**证据**: §273 §277 §283 §285 §286 §290 §292 §297.1 §302.1 §304 §305 §306；`docs/analysis/vwa_classifieds/B0_*_classifieds_diag_digest.md`、`docs/analysis/vwa_classifieds/B0_classifieds_6mode_failure_taxonomy.md`。

**原文片段**: 「SR 30.4% (68/224) vs dom 15.18% (R31194) = 翻倍; P14 URL 自环占失败 41.7%; deterministic coverage 71.8% (对比 dom 87.9%); ≥3 个 benchmark-FP → 真实 SR ≥ 31.7%, 30.4% 是下界」(§283)

---

## 2. B0 cls dom SR 的跨 run 序列 + partial-run 陷阱

**当前值**: 同 (B0, cls, dom) 在不同 run 上的 full-cohort SR 依次为 R2987 14.1%（2026-05-20, pre-B-1794）· R9755 14.7% (224 ep) · R31194 15.18% · R21557 17.4% (39/224)。历史 archive 参考值 raw_sr = 0.1496（`compare_b0_b1.py` 历史 paper-grade 参考，§236.1）。

**演变**:
- §234 Fire-3 前 25 episodes 报 5/25 = 20.0%（partial）→ §236.1 与 archive 0.1496 比较得 "+5.04pp 差 = 0.71σ"（N=25 二项 SE ≈ 7.1pp）→ §253.2 R9755 63 ep 报 20.6% → §258 R9755 全量 224 ep = 14.7%。
- cls task 有强难度排序（§253.2）："task 0-44 都约 7%, 45-59 = 47%, 60-74 = 75%; 累积 SR 走势 6.7%→8.9%→18.3%→20.6%"。
- §297.1 archive↔current 的 SR 差随 n 收窄："Δ=+7.4pp@81task → +8.0pp@88task → +5.8pp@102 → 全量 +2.2pp@224"。

**已作废**:
- §234 的 20.0% (5/25)、§253.2 的 20.6% (63 ep) 作为 condition 级 SR 全部失效——同一 run R9755 全量收在 14.7%。
- §110 的 partial 加权 11.7%（R2987 前 60 task 构成加权）同理只是 partial 诊断值。
- §82 该 run 后被 §244 整体判为 non-canonical（post-B-991 pre-tool_choice-fix）。

**caveats**（原文，一字不改）:
- §253.2: "教训: 绝不在 run 没跑完时判 SR — 我自己也报过 8.9% 的假警; full-cohort 才是真数"。
- §110: "VWA task 类型在 0-233 上非均匀分布, 按顺序跑时前缀构成系统性偏离全集 → partial SR 不仅 n 小连难度构成都偏; 小 n binomial 噪声 SE≈4.7pp @ n=73, ≈20pp @ n=4"。
- §297.1: "partial 阶段 SR 差是 partial-on-easy 假象 (前段 18.6% / 尾段 8.3%) + within-noise 上偏"。
- §236.1: "N=25 partial; 趋势偏高不偏低所以无污染信号; cls task 升序且 task 0 较简单 → 早期 SR 高估是已知现象, 预期 N=224 时回落到 0.15 附近"。

**证据**: §110 §234 §236.1 §253.2 §258 §277 §297.1；`scripts/analysis/compare_b0_b1.py`。

**原文片段**: 「早期低是 partial-run 窗口噪声 — cls task 有强难度排序: task 0-44 都约 7%, 45-59 = 47%, 60-74 = 75%」(§253.2)

---

## 3. 同 condition 重跑的噪声地板（discordance / κ / self-oracle drop）

**当前值**（**各自 scope 并列，禁止相加/相减/取平均**）:

| 量 | 值 | scope | § |
|---|---|---|---|
| per-task 对称翻转率 | 12% (9/73) | B0 cls dom, R19740 ∩ R12090 公共 73 task 前缀 | §242 |
| discordance | 27/224 = 12.1%（split 16/11, net +5, McNemar 精确双侧 p≈0.442 不显著） | B0 cls dom, archive R31194 ↔ current R21557 | §297.1 §298.1 |
| discordance | 32/224 = 14.3%（16 PASS→fail / 14 fail→PASS / 2 边缘） | B0 cls vision, R24792 ↔ R32024 | §302.1 |
| Cohen kappa | 0.614 | B0 cls vision, 同上一对 | §302.1 |
| self-oracle drop-one | A→B 6.7pp / B→A 7.6pp（不对称 1pp） | B0 cls vision, 同上一对 | §302.1 |
| net SR Δ | +0.9pp (24.1% → 25.0%) | B0 cls vision 同上 | §302.1 |
| net SR Δ | +2.2pp (15.2% → 17.4%) | B0 cls dom 同上 | §297.1 |
| net SR Δ | −3.0pp (28.4% → 25.4%)，6 flip | B0 cls phantom_text, PARTIAL 67 task 交集 | §292 |
| 实测 per-task noise floor | ~9% (6/67) | B0 cls phantom_text PARTIAL | §292 |
| single-realization net SR shift 噪声 std | ≈ ±2.5pp（√32/224） | B0 cls 224-task 单次 realization | §303.7 |

**演变**: §242 先在 dom 上给 "12% per-task flip = B0 cls noise floor" → §292 在 P-text 给 ~9% → §297.1 全量 dom 12.1% + McNemar 不显著 → §302.1 vision 给出最完整一组（discordance / κ / self-oracle 双向）。§302.1 自陈初报 "30 model_nondeterm" 是 `--max-flip-detail` 默认 truncate 100 的 reporting artifact，真值 32/32 全 model_nondeterm。

**已作废**:
- §292 本节把 6 flip 全归 model 非确定、并称 element_id flip decision-harmless 的判断，**已被 §294 纠正为错**（台账原文）。
- §182（R9725 step-0 观测差异结构）被 §294 supersede。
- 跨批次：D3 记录 207–210 带 flag「named by **RETRACTED** §397.10: §302 的线性减法 12.1% ≈ 10.5% + 1-2pp」——即把 dom discordance 12.1% 拆成 id-effect 10.5% + 余项的线性分解**已作废**。
- 跨批次：记录 109/110/111 带 flag「named by RETRACTED §243: §242 结论『Fire-6 继续』」与「named by RETRACTED §298.4: element_id 是 red herring, 真机制是 MoE」。

**caveats**（一字不改）:
- 工具自带（**凡引用 self_drop 必须随附**）: `instability proxy, NOT H1 drop-one bias correction; 小样本/可能混代码版本 = upper-bound risk trigger`（`compare_cross_run_same_condition.py`；本条按编排者指令随附）。
- §302.1 原文: "14.3% 是 real noise floor 而 +0.9pp 只是 net 不是真噪声地板; 三层分解初报 30 model_nondeterm 是 --max-flip-detail 默认 truncate 100 的 reporting artifact, 真值 32/32 全 model_nondeterm; self-oracle drop 只能作 instability diagnostic 不是 bias estimate (§293); 1pp 不对称的解释在 codex 解构后已变"。
- §303.7: "→ DOM vs P-text gap 完全在 noise std 内, 不能 reliably 区分; 这是 §302 discordance 32 反推的 binomial 估计非独立测量"。
- §297.1: "flip 全 model_nondeterm; 0 reset/start_url 污染"；"dom 代码 AMENDMENT_07 未动 (保 native nodeId) 故差落 run-to-run 噪声内"。
- §242: "temperature 0.0 + seed 42 已在 3 个 baseline 的 config 里确认"。

**证据**: §242 §292 §297.1 §298.1 §302.1 §303.7；`scripts/analysis/compare_cross_run_same_condition.py`。

**原文片段**: 「discordance 32/224 = 14.3% (16 PASS→fail / 14 fail→PASS / 2 边缘); Cohen kappa = 0.614; self-oracle drop-one A→B 6.7pp / B→A 7.6pp (不对称 1pp); reset-goto scan 0 mismatched (224/224 step-0 url_before 一致)」(§302.1)

---

## 4. id-perturbation：element_id 通道的因果强度

**当前值**（三组独立设计，**各自 scope 并列**）:

| 设计 | 值 | scope | § |
|---|---|---|---|
| Tier A 受控 id 扰动（自然 archive↔current 观测对） | id-flip **10.5%** (14/133)；determinism 133/133 OK；0 NONDET, 0 confound；flip tasks [10,11,12,16,17,59,64,92,93,94,107,108,118,125] | B1 (Qwen3-VL-4B dense, temp=0, DGX GB10) / cls / dom / step-0 / n=133 | §298.2 |
| paired id-shuffle（within-obs 全 bijection） | mode_flip **B0 0.125 (5/40)** / **B1 0.200 (8/40)**；id 边际 drop B−A：B0 **+0.023 (→null)** / B1 **0.000** | B0 (AWS-235B, N=12 重放) 与 B1 (local-4B, N=3, temp=0) / cls / step-0 / 同一批 sample:40 task (seed=42) | §308 |
| serving floor (=1−consistency_A) | **B0 13.3% (0.867)** / **B1 0% (1.000 逐 task 无例外)**；B0 有 15/40 task consistency_A<1.0（最低 task 60/198 = 0.333） | 同上 | §308 |
| flip id 集合交集 | B0 {41,60,63,182,198} vs B1 {1,22,23,60,108,153,170,231}，交集只 {60} | 同上 | §308 |
| flip 步的观测比对 | 14/14 modulo-id 字节一致，0 第三方内容差异；但 0/14 byte-identical | B0 / cls / dom / 14 个 flip task | §297.2 |
| Tier B activation patching | 6 task × 36 层，模式 task-dependent（见下） | B1 dense / cls / dom / Myriad A100-PCIE-40GB job 433134 | §300.3 §300.4 |

Tier B per-layer 原始计数（arch-match / curr / other）：task 93 = 30/6/0（全网广）· task 125 = 26 (L0-10 居多)/9 (L21-35 居多)/1（早层 id 晚层 curr 分裂）· task 92 = 15/9/12 · task 17 = 2 (L0+L10)/31/3 · task 118 = 7 (L3/5/8/11/13/23/25 散点)/29/0 · task 64 = 0/0/36（chaos，每层 patch 产生第三动作）。

**演变**: §297.2 先确认 flip 的分歧只能落在 {element_id-rep ∪ MoE}，且两者在 cross-version 数据里拆不开 → §298.2 用 B1 dense（零 MoE）拿到纯 id 因果 10.5% → §308 换成 within-baseline paired 无偏抽样，同时给出 serving floor 与 id 边际效应，并**自我修正**早先 16-task run 报的 25.8%。

**已作废**:
- §308 原文自陈: "无偏 sample:40 的 floor 是 13.3% 而非早先 16-task run 报的 **25.8%** — 那 16 个是 flip-biased (专挑 flip) 的 selection bias"。**25.8% 已死。**
- 跨批次 flag：记录 200 / 197 / 198 / 199 带「named by RETRACTED §298.4: element_id 是 red herring, 真机制是 MoE (session 中途我自己说的)」；记录 205 带「named by RETRACTED §384.2: paper §5 (mechanism) 结论建立在 pre-fix archive 上」；记录 218 带「named by RETRACTED §324 / §309」。
- §292 里「element_id flip decision-harmless」的判断已被 §294 纠正为错。

**caveats**（一字不改）:
- §298.2: "B1 < B0 capability → 10.5% 是 id-effect 上界, B0 真实 id-sensitivity 应 ≤ 此; step-0 only; imgs=0/1 都有 (id-effect 不依赖多模态); 踩 GB10 device_map=auto CPU offload 坑后强制 GPU0"。
- §308: "20% ≠ §298 的 10.5% 且不矛盾 (within-obs 全 bijection = id channel sensitivity 上界; §298 是 natural archive churn = 贴近 deployment 下界, 两数并列框出上下界); step-0 only (多步累积/衰减未测); flip = 决策不稳定 ≠ success delta; n=40 CI 宽 (0.125 vs 0.20 是 descriptive 非 inferential); B1 N=3 够仅因逐 task consistency=1.0 无例外; 未 commit"；另 "29 次 AWS 503 全被 3-retry 吸收 (0 ERR)"。
- §300.3/§300.4: "emergence_layer metric 有 inherent bias (patch 越早层下游 re-compute 越多 → 影响越大), 故 5/6 task 的 emergence=0 部分是 protocol artifact 不代表 id 一定在 L0; 结论只到 task-dependent distributed signal 不是单层定位"。
- §297.2: "divergence 全落 {element_id-rep ∪ MoE}, 但 MoE 与 id-rep 在 cross-version 数据里拆不开 (只有 SoM-family post-sequential 或 vision 能拿 byte-identical 输入)"。

**证据**: §297.2 §298.2 §300.2 §300.3 §300.4 §308；`scripts/analysis/tier_a_id_perturbation.py`、`scripts/analysis/tier_b_id_patching.py`、`docs/checkpoints/probes/b0_paired_idperturb_20260529_152959_b0_unbiased.json`、`results/mechanistic/tier_b_id_patching_20260528_233656_myriad455328`。

**原文片段**: 「serving floor (=1−consistency_A): B0 13.3% (0.867) / B1 0% (1.000 逐 task 无例外); mode_flip: B0 0.125 (5/40) / B1 0.200 (8/40); id 边际 drop B−A: B0 +0.023 (→null) / B1 0.000」(§308)

---

## 5. SOM_MARKS 的 id 来源与 sequential 重编号（AMENDMENT_07 的上游证据）

**当前值**:
- P79 的 `[SOM_MARKS]` id = **CDP getFullAXTree nodeId**，乱序不连续（真实 obs 铁证：`[5]Logout` 夹在 `[115][118]` 之间），**非**标准 SoM (Yang 2023) / VWA visid 的 sequential（§294）。
- 155-task step-0 四 run 观测的跨 run 字节一致率（§295）：**raw nodeId 4/155 = 3%** · **id-stripped（内容+顺序）155/155** · **sequential 重编号后 155/155**。
- dom 侧：§297.1 caveat 明写 "dom 代码 **AMENDMENT_07 未动** (保 native nodeId)"。

**演变**: §294 定位 id 来源 → §295 证明 sequential 重编号能 100% 消 churn（这是 AMENDMENT_07 的动机测量）。

**已作废**: 本批内无。

**caveats**（一字不改）:
- §294: "axis 1 (DOM↔P-text) 两边都是 nodeId 故不 confound (好消息)"。
- §295: "机制 = per-document AXID 计数器按 DOM 子树整体偏移 (task_0 页头 block −1 / 列表 block +0) + session 元素独立 jitter (Logout 5/6/8/5), 顺序恒定 → 不是均匀 +k 但顺序不变故 sequential 消 churn 100%; 残留 = B0 MoE (字节相同输入仍翻), sequential 不治"。

**证据**: §294 §295 §297.1；`external/visualwebarena/browser_env/processors.py`。

**原文片段**: 「raw nodeId 4-run 字节全同仅 4/155 (3%); id-stripped (内容+顺序) 155/155 全同; sequential 重编号后 155/155 全同」(§295)

> ⚠️ **AMENDMENT_07 前后 SR（Δ−3.2pp）不在本批**：D3 (§207.4–§311) 只含上述**上游动机测量**与「dom 未动」这一条 caveat，**没有**任何 AMENDMENT_07 前后 SR 对比记录。相关矛盾见文末矛盾清单。

---

## 6. 跨 GPU / 跨 provider 的 greedy 非确定性

**当前值**:
- **跨 GPU type**（§300.2）：同一 Tier B harness、同一 14-task flip 子集，job **432990 (V100) 报 9 个 task baseline flip**；job **433134 (A100-PCIE-40GB) 报 6 个**；差 3 个 task（10/11/12 在 V100 翻、在 A100 不翻）。
- **跨 provider**（§302.8，20 task × 5 次 same-payload replay）：
  - AWS Bedrock（wall 255s）：unique5/5 = **16/20 (80%)**，unique1（full deterministic）= **0/20**，mixed 4/20（全 ≥4 unique），margin 均值 **4-5 logit**，action class jump (click↔type↔select_option) 频繁，1 个 503 timeout。
  - DashScope intl（wall 379s）：unique5/5 = **1/20 (5%)**，unique1 = **4/20 (20%, task 7/8/54/60)**，mixed 15/20 (75%)，margin 均值 **16-17 logit**，action class jump 极少（coord 微漂 ±5 pixel）。
  - `system_fingerprint` 两 provider 均 **0/100 calls** 返回。
- **vision 双 run 输入字节一致性**（§302.2）：step-0 screenshot md5 **224/224 一致**，image metadata 224/224，url_before 224/224；而 **step-0 exact actions 仅 2/224 一致（222 不同）**；32 个 success flip **32/32** 在 step-0 已 action diverge。
- codex cold-start 候选排序（§302.3）：#1 Remote B0 serving nondeterminism under nominal greedy（Bedrock kernels / dynamic batching / tie-breaking），解释全部 6 个观测信号；#2 provider alias/deployment drift（未 pin snapshot）；#3 tool-calling constrained-decoder instability；#4 MoE-specific routing nondeterminism（plausible 但 under-evidenced）；#5-#9 全排除。

**演变**: §300.2 先在本地 B1 上发现跨 GPU 漂移 → §302.2/§302.3 在 B0 vision 上确认 byte-identical 输入仍 99% 分叉 → §302.8 用双 provider 对照把「cross-provider universal」框架推翻。

**已作废**: §302.8 原文自陈——第一次 1-task sanity（2/2 unique）导出的 "cross-provider universal framing" **已被 DashScope 4/20 bit-exact 反驳**。

**caveats**（一字不改）:
- §300.2: "根因 = fp 算术顺序差异导致 argmax 偶翻 (cuda kernel reduction 顺序跨 GPU 不同); 单 GPU type 内确定, 跨 GPU 不确定; reproducibility implication: DGX GB10 上测的 10.5% 在另一 GPU type 可能 ±3-5pp; 需 paper §3.5 GPU-type metadata disclosure"。
- §302.8: "cross-provider control 必须 batch-level (≥20 task)"；scope 含 "DGX 客户端 0 GPU / 总成本约 $1"。
- §302.2: "vision agent 看不到 DOM (proxy_api_agent.py:575 obs_section 为空, screenshot only) 故 observation_dom 漂移无关; 99% step-0 action diverge under byte-identical screenshot = server-side nondeterminism 是 dominant, 不是后续轨迹漂出来的"。
- §302.3: "cold-start prompt 不列我的候选以避免锚定; 排序是 codex 判断非实验证据; audit artifact gap 是硬 blocker (env_snapshot.json:83 provider_immutable_sha_available=false, proxy_api_agent.py:811 POST 未存 request_id/instance_id/SHA/headers)"。

**证据**: §300.2 §302.2 §302.3 §302.8；`docs/checkpoints/codex_outputs/vision_moe_anomaly_2026-05-27.md`、`docs/checkpoints/probes/replay_step0_n5_20260527_013506_batch1_within_minute.json`。

**原文片段**: 「432990 (V100) 报 9 个 task baseline flip; 433134 (A100-PCIE-40GB) 报 6 个; 差 3 个 task (10/11/12 在 V100 翻在 A100 不翻)」(§300.2)

---

## 7. cross-mode oracle 与 best-single（routable 空间）

**当前值**（§306，B0 / cls / 6 mode / n=224 / 6 个 canonical Gate-3 fresh run）:
- task class: **universal-solve 9 / universal-fail 127 / routable 88 (39%)**
- **6-mode oracle SR 43.3% (97/224)** vs **best-single som 27.2% (61/224)** = **+16pp oracle gain**
- exclusive-solve: vision 9 / som 6 / pprompt 6 / dom 4 / ptext 2 / psom 2
- 失败类型×mode: THUMBNAIL 24-37 跨 mode 主导 · BUDGET 56-69 (psom 69 最高) · IMG 6-16 · SEARCH-NAV 3-15 · UNCLEAR-NAV 58-70

**演变**: §291 首版 3-mode（dom+som+vision）报 universal-solve 15 / universal-fail 138 (62%) / routable 71；**full 3-mode oracle SR 38.4%**；som 独家解 18、vision 独家解 10；图像识别全谱 (THUMBNAIL+IMG) 梯度 dom 50 > som 30 > vision 27；SEARCH-NAV 真实仅 8/7/11；UNCLEAR-NAV 59-73 → §306 扩到 6-mode 后全部改写（见上）。

**已作废**: §291 的整套 3-mode 数字（oracle 38.4% / routable 71 / universal-fail 138 / som 独家 18 / vision 独家 10）被 §306 supersede。

**caveats**（一字不改）:
- §306 脚本自标 **PROVISIONAL NOT paper-grade**，四条：
  "(1) §302 的约 14% per-task serving flip 使 88 routable 部分是 noise flip 误判, +16pp 与 14% noise floor 同量级必须 replicate 扣 noise; (2) single cell (只 B0 cls); (3) UNCLEAR-NAV 是判据天花板 (约 65% cls task 无 sCategory sig); (4) 本 taxonomy 独立于 P-rule ruleset 故不受 ruleset 版本不一致影响"。
- §291: "PROVISIONAL — 3/6 mode, 单 (B0,cls), 禁 cross-mode 定量 (discover-then-freeze); UNCLEAR-NAV 是判据天花板 (65% cls task 无 sCategory sig); 早期只看 vision-exclusive 会高估 vision (选择偏差)"。
- 跨批次 flag：记录 216 带「named by RETRACTED §309」。

**证据**: §291 §306；`docs/analysis/cross_sites/cross_mode_failure_taxonomy_B0_classifieds.md`、`docs/analysis/vwa_classifieds/B0_classifieds_6mode_failure_taxonomy.md`。

**原文片段**: 「6-mode oracle SR 43.3% (97/224) vs best-single som 27.2% (61/224) = +16pp oracle gain」(§306)

---

## 8. H1 gate 估计量与 archive meta 统计（drop-one / FE / power）

**当前值**（**全部是 archive vintage，pre-Gate3，不可与 canonical 混用**）:
- **6-mode strict drop-one**（canonical producer 锁定的 H1 gate 估计量）：**cls 1.28 / red 0.95 / b1cls 1.28, pooled ~1.1pp**（§274）。
- **4-mode ADD (`4psom_vs_3`)**：θ_FE = **2.336 / 2.34pp**, se_fe = **0.529**, k_cells = **3**，I² = **0.0%**, τ² = 0.0, p_Q = 0.46（§8 §16 §209）——台账标 **SECONDARY / appendix-only**。
- **power**（§209）：per-cell paired-bootstrap SE = **0.916pp**（理论 1-sample 上界的 1/2.2）；SE_FE = **0.529pp** (k=3)；empirical power = **81% (k=3 archive)** / **97% (projected k=6 Phase 1a)**；隐含 empirical πD ≈ **0.019** = 最坏情况 2p(1-p)=0.18 的约 1/10。
- 其他 archive 异质性：P-text drop-in **I²=71%**；oracle lift **I²=70.5%**（§211.2）。
- **HKSJ**：se_re=0.529, se_hksj_unmodified=0.466 → **12% shrinkage**；modified non-shrink guard `SE_HKSJ_mod = max(SE_HKSJ, SE_RE)` 恢复到 0.529（§215）。
- **joint family-wise Type I**：H1 (m=1) + H10 (m=1) 无跨家族 Holm → **1 − (1 − 0.05)² ≈ 0.0975 (9.75%)**（§215）。
- prereg P1-5 的 **±2pp 锚**：2× median archive SE → **1.92pp ≈ 2.0pp**（§212.3）。

**演变**: §209 用 archive 重算 FE-pool gate 的实证 power，把 A2.3a 的一批 finding 降级（P0-1-AC* RETRACTED / P0-2-A* → P1 方向反转 / P0-3-A* → P1 conclusion-invariant / P1-4-A* → P2 empirical I²=0%）→ §274 发现 lock day-1 的内部矛盾：**gate 用 6-mode strict drop-one，而 power 表 (line 361) 用 +2.336pp 的 4-mode ADD**，"strict ≤ ADD by construction → 用 ADD 算 power 会 OVERSTATE; **97% power 是 4-mode 虚高数**"。

**已作废**:
- **97% projected power 作为 6-mode strict gate 的 power 已作废**（§274 判为 4-mode 虚高数）。
- A2.3a 的 P0-1-AC*（RETRACTED）。
- §209 原文自陈：81%/97% "**NOT 48%**"——A2.3a Mode A F2 的理论计算 48% 已被实证推翻（亦见 CLAUDE.md 已 land 的同一裁定）。

**caveats**（一字不改）:
- §209: "k=6 的 97% 是 projection 不是测量; archive 是 3-cell pilot 不是 Phase 1a 数据; SE 小是因为 mode 间真实相关高 (共享 task-difficulty variance)"。
- §8: "archive placeholder, 非 Phase 1a outcome (§230 明确要求在 OSF DOI 1 README 里声明这一点)"。
- §274: "reddit 0.95 已 <δ=1.0pp → H1-strict 实为擦边 gate; 与之对照的 +2.336pp 是 4-mode ADD (4psom_vs_3) 属 SECONDARY/appendix-only, 两者不可混; archive 数字不可与 canonical 混用"。
- §211.2: "P-text / oracle 的高 I² 属 secondary/tertiary 不是 hero arm; R5 的 75% heterogeneity cap 攻击在 hero arm 上实证未激活"。
- §215 (HKSJ): "non-shrink guard 依据 Röver-Knapp-Friede 2015 §4; 只用于 hero arm 的 Appendix-D-bis sensitivity 行"。
- §215 (9.75%): "主动 disclose 在 prereg §3 + paper §8; 辩护是 structural 而非 statistical … 引 Bender & Lange 2001 + Cox-style sequential-decision"。

**证据**: §209 §211.2 §212.3 §215 §274；`results/phantom_paper/meta_phantom_lift.csv`、`results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv`、`scripts/analysis/aggregate_phase1_prereg_gate.py`、`docs/analysis/cross_sites/power_analysis.md`。

**原文片段**: 「gate = 6-mode drop-one (aggregate_phase1_prereg_gate.py:_cell_drop_one_theta_se, SIX_MODES, oracle_6 − oracle_5_no_psom, line 107-136) vs power 表 (:361) 用 +2.336pp = 4-mode ADD」(§274)

---

## 9. B0 serialization：tool_choice 与 element_id（B-991 → B-1794）

**当前值**（协议 reset 后，`proxy_api_agent.py:727` = `tool_choice='required'`，§278）:
- R9755 (Fire-6 re-fire, cls B0 dom, 666 steps): **parse_error_rate = 0%** / parse_valid 666:0 / action_source 全 tool_call / **element_id 覆盖 type 89/89 + click 225/225 = 100%**（§253.2）。
- serialization health table 全绿（§244）：**B0 required emit 100% / valid ~95%** · **B1 text JSON parse_error 0%（149 steps，95% 上界 <2%）** · **B2 (Gemma3) text-JSON parse_error 0.7%（306 steps / 11 episodes）**。
- paper-grade verdict 报的 B0 dom **parse_error_rate = 0.00058 (1/1710)**（§278）。

**演变**（完整链）:
1. §210 首个 N=30 probe（简化 schema）：emit_rate **100%** / schema_valid_rate **13.3% (4/30)**；60% 发 `select_option` 且 `text='Lowest Price'`（意图正确）被 `validate_action` 拒。
2. §210 换 production `_WEB_ACTION_TOOL` 的 N=30 full-stack probe：emit 100% / schema_valid **100%** / logprobs_present 100%；cost mean $0.00179 ± $0.0001；elapsed 1216±258ms；logprob token count 57.5±19.8 —— **被 §214 证明是假 PASS**（model 30/30 发 `element_id:[37]` list，production validate_action 会拒）。
3. §214 本地 production-path probe（post-fix）：agent_valid=1.0 / dispatchable=1.0 / confidence_present=1.0。
4. §243 Root cause 1：`tool_choice='auto'` + system prompt "Output ONLY valid JSON" 双协议冲突 → B0 emit tool_calls **仅 22.5%** → 77% 退 text-parse → **31% parse_error** → 注入 wait 死循环吃 step budget。probe 铁证：**126-char prompt（无 JSON 指令）emit 100% vs 真实 3881-char prompt（含指令）emit 0%**；`tool_choice='required'`（OpenAI 字符串）emit **0%→100%** / valid ~95% / logprobs 100%。
5. §250.3 Smoke A：B0 的 invalid 全是 `invalid_element_id` —— emit type 却缺 element_id（raw: `{type, text:'blue kayak\n'}`，部分混 url）；同 observation 下 **B1 的 type 17/17、B2 的 4/4 都带 element_id**。
6. §251 B-1794 定根因：6 变体真 proxy probe —— "tc=required baseline 漏 eid + 捏造 url ❌; tc=auto 发 eid=140 ✓; 把 eid 放进 required 后发 ✓; 删掉 url 仍漏 (改用 option_label 凑数); prompt nudge 仍漏"。
7. §251.3 修复后：proxy variant probe 6/6 + comprehensive probe 5/5；schema≡validator 程序化证明 8/8 + 10 个 invariant 测试；2×30 步 dom smoke **0 invalid**；1207 pytest。
8. §253.2 B-1794 对 SR 的净效果：**R2987 (pre) 14.1% → R9755 (post) 20.6%**。

**已作废**:
- §210 的 "full-stack probe schema_valid_rate=1.0 = Q1=A gate PASS" —— 被 §214 判为假 PASS；记录 11/13/14 带 flag「named by **RETRACTED** §214: Q1=A pilot gate PASS」。
- §243 的 tool_choice 分析被 §251 supersede（§243 认定 required 是解，§251 证明 required 本身诱发 degenerate minimal call）。
- §115 的 serialization health table 被 §244 supersede（S4 缺口由 B2 0.7% 补齐）。
- CLAUDE.md 曾写 `tool_choice=auto` —— §278 判为 stale（"repo 里 9 处写 auto, 其中 7 处是 OSF deposit / fire-lock / provenance / chronicle / catalog = 故意不可改的审计链, 只有 2 处是 active 该改"）。
- GLM rescue：历史 rescue rate **1.49%**，B-396 已在所有 active config 关掉 `use_glm_fallback`（45 files verified, 2026-05-16，§208.4）；GLM rescue 曾掩盖的真实 parse_error 率 = **25-30%**（§243）。

**caveats**（一字不改）:
- §210 (简化 schema): "根因 = probe schema drift (只有 6 字段, 缺 option_label/option_value/option_index/coordinate/answer/page_number), 不是 migration bug"。
- §210 (tool_choice 开销 **4.79× input_tokens**, $0.000146 → $0.000700): "user 裁定这是 B0 内部跨 mode 比较里的 constant offset, 跨 mode 相减抵消 → 不需 disclosure; paper §1 cost 比较是 within-B0 cross-mode 不是 cross-baseline"。
- §251.2: "结论: tool_choice='required' 的强制本身让 Bedrock 产出只满足 required 数组 (action_type+thought) 的 minimal/degenerate call, 漏掉所有 optional 字段; element_id 最易漏是因为 235B 有「搜索 = 输入查询 + 跳 URL」的竞争 prior"。
- §251.1: "(b) 同时排除了 capability/grounding 假说 — 4B 能完美 ground 匿名 textbox"；§250.3: "匿名 textbox 不是根因 (两个 4B 模型都 ground 成功了)"。
- §253.2: "解释 = search 类动作恢复 (pre-fix 时 search/type 漏 element_id 被压低); 与 pre-reset 的 ~20% 同量级但协议更干净且 upstream-aligned"。**注意此 20.6% 是 63-ep partial，见主题 2。**
- §278: "3-AI /stress 指出 parse_error_rate 名不副实 (合法 wait 被记成 parse error); §3 prose 另报 B0≈0 / B1 0 / B2 0.7% → 经验 inert"。
- §208.4: "1.49% 是历史存档值; strict-SR sensitivity 在 Phase 1a clean substrate 上 vacuous, 只留 Appendix-D"。

**证据**: §208.4 §210 §214 §243 §244 §250.2 §250.3 §251 §253.2 §278；`p79/agents/proxy_api_agent.py`、`p79/backends/action_utils.py`、`tests/test_b0_schema_validator_consistency.py`、`docs/checkpoints/probes/proxy_full_stack_225749.json`。

**原文片段**: 「probe 铁证: 126-char prompt (无 JSON 指令) emit 100% vs 真实 3881-char prompt (含指令) emit 0%; tool_choice='required' (OpenAI 字符串) emit 0%→100% / valid ~95% / logprobs 100%」(§243)

---

## 10. vision 模式的坐标契约（B-1860 前后）

**当前值**（§290，B0 / cls / vision / n=224 / R24792 = B-1860 APPLIED 后首个 vision run）:
- parse_error **0.0273% (1/3662)**，较 pre-fix 降约 **500×**；tool_call_invalid 同 1/3662。
- 59 深挖 ep 的 `b1860_coord_residual` = **0/59**（7 agent 独立确认）；`glm_fallback_attempted=0`。
- 唯一残留 invalid_coord (task9 step29) 是 `action_executed=None` 的 fail-loud no-op = 设计行为；telemetry `coordinate_normalization` 显示 `x_regime=qwen_0_1000` / `recovered=true`。

**演变**: §285 R3671 pre-fix 实测 —— parse_error_rate **13.6%**（dom/som 仅 0.06%，高 **200×**）；**484 emitted coords y_max=972 > viewport 720 但从不超 1000**；**507 parse error = 484 合法 JSON + 23 真 malformed**；**104 misclick**；**48% ep (108/224) ≥3 parse error，其中 99% (107/108) fail**；**39 ep 命中 total-cap=5**；**model ~75% 遵守 prompt [0,1] / ~25% 退回 native 0-1000** → §287 codex 复审抓到 fix 自身 4 个 bug → §290 落地验证。

**已作废**: §285 的 vision SR **13.84%** 与该 run 全部指标——"该 run 后被标 non-canonical 并 archive (RCA 用), vision SR 13.84% 被 coordinate-scaffold 主导, 非 clean model 能力, 不可跨 mode 比"。

**caveats**（一字不改）:
- §287 fix 自身 4 bug: "V-F1[P1] true_oob fail-closed no-op 缺失 (4 wrapper coord site: 原 fix 把 true_oob 标记后 eps-clamp 到 viewport 边缘仍执行 create_mouse_click_action = 右上角真实点击污染页面) / V-F2[P1] 下游 negative coord 归 malformed 被 caller skip → 既不计 P1 也不计 leak (under-count) / V-F3[P2] dead_zone tag 无 read path / V-F4a[P2] annotate_screenshots.py 未 import normalizer"；设计选择 "validator 仍接受 true_oob (parse_valid=True) 而 wrapper no-op — 因 true_oob 是格式合法但 grounding 越界 = grounding miss ≠ parse_error"；"V-F4b (glm_batch_digest.py) 被撤回因该工具半退役不在 fire 路径 → **代码有 bug ≠ bug 会发生**"。
- §297.3（证明 dom 不受影响）: dom **0 坐标动作 (0/3420 archive + 0/1340 current)**；"prompt 明文禁坐标; B-1860 只改 make_som_prompt + make_vision_prompt 不碰 make_dom_prompt"。记录 199 带 flag「named by RETRACTED §297.5: dom archive 被 B-1860 prompt 改动污染」。
- §292: "B-1860 对 P-text 无副作用 (P-text 无图无 coord 不在坐标栈)"。
- §290 (SR): "benchmark-FP 后经截图 forensic 翻案 (task 40 改判 agent-limit) → 净 benchmark-FP≈0; dom-only 视觉规则 P6/P15/P16/P21 全 0 命中 = mode-gate 正确"。

**证据**: §285 §287 §290 §297.3；`p79/envs/vwa_wrapper.py`、`docs/analysis/vwa_classifieds/B0_vision_classifieds_diag_digest.md`。

**原文片段**: 「parse_error 13.6% (R3671 pre-fix) → 0.0273% (1/3662, R24792), 降约 500×」(§290)

---

## 11. /diag P-rule 的覆盖率与误报率

**当前值**:
- deterministic failed-coverage：**R31194 (fresh, dom) 87.9%** · som R9725 **71.8% → 74.4%**（ruleset 3-domsom 后）· dom **87.9% → 85.8%**（同次重扫，P14 假覆盖诚实去除）（§277 §283 §284）。
- Tier-1 扫描成本：**224 ep = 0.23s / 19MB / 0 token**，逐 episode 内存 O(1)（§261.2）。
- 误报率实测：**P6 在 R31194 的 30/34 success 上 fire = 误报 88%**（§277）· P14 success-fire **15→0**、failed **65→19**（§284）· P33 success-fire **1/17 = 5.9%**（§304）· 新规则在 held-out run（R2987 同 site 不同 run + R1821 reddit 真 held-out）**FP 全 0%**（§261）。
- failed-hit causal verify 的 presence-coincidence：R21557 dom **3/3 not causal** + R5313 som **4/5 not causal**，唯一 causal = T9 P31 budget，**合计约 80% presence-coincidence**（§299.6）。

**演变**: §259 Tier-1 首次在 R9755 上 156/156 failed 全覆盖 / 0 token → §261 self-evolving 一轮后 **no-hit failed 35→13 (-63%)，coverage 82%→93% (+11pp)**，新规则 FP：P15 0% / P17 5% / P18 6%（对比既有 P6 22% / P14 12%）→ §277 在 fresh run R31194 上掉到 **87.9%**，"5pp 掉幅 = 实证确认 93% 确有 in-sample 虚高"。ruleset 版本 2-dom → 3-domsom → 4 → 5。

**已作废**:
- **93% coverage 作为泛化值已作废**（§277 实证 87.9%）；§261 自标 "93% 是 in-sample (P15-P18 从 R9755 拟合又在 R9755 评估), 3-AI 审计明确定位为 post-hoc fit"。
- §259 的 "156/156 failed 全覆盖" 分母被 §261 修正：B-1829 分母 bug 后 failed 计数应为 **191**（178≠191 曾是 `--failed-only` 分母 bug）。
- §299.5 的 P10 三类误报来源被 §347 supersede（跨批次）。
- §304 的 P33 被 §325 supersede（跨批次）。

**caveats**（一字不改）:
- §261: "digest 定位为 internal 诊断而非 paper failure-analysis"；§261（held-out）: "只证伪了 overfit-to-误报, coverage 93% 仍是 in-sample (held-out run 都 partial 测不出真 coverage)"。
- §277: "R9755 的 13 个 no-hit 有 10 个在 R31194 仍 no-hit (盲区可复现), fresh run 另暴露 13 个新盲区"；P6 误报根因 "B0 多模态把 reference image OCR 当文字搜索绕过视觉需求; 12 个规则命中的 causal verify 里有 5 个规则名标错 → presence≠causation"。
- §284: "P10 首版 strip 全部日期致 +4 新 FP, 改双变量后 0 新增 FP — 没跑验证根本发现不了; 2 个边界 success-fire (task233/151) 是 url_match 读 live URL 不看 finish 的机制特性, 记录不调优避免 over-fit"。
- §299.5 P10 三类误报来源: "(a) URL 端口数字渗漏 … (b) SoM element_id 数字渗漏 … (c) DATE_CONTEXT_RE 月/日残留"；"carveout 提议待 freeze 落码"。
- §299.6: "样本极小 (3+5); implications = per-rule failed-only 计数只能读作 presence detector 不能读作 causation, **paper 不能引 P14 命中 N 次因 URL 自环失败 这类话**"。
- §261.2: "Tier-2 真瓶颈是 no-hit 子集大小非总 task 数; 故设 max 50 ep deep-dive/单次 cap"。
- §259（为什么需要 Tier-1）: step log ~80KB/ep → R9755 191 failed ≈ **4.6M tokens**；全 Fire-6 36 condition ≈ **165M tokens**（"估算非实测 token 计数"）。

**证据**: §259 §261 §261.2 §277 §283 §284 §299.5 §299.6 §304；`scripts/analysis/diag_pattern_match.py`、各 `*_diag_digest.md`。

**原文片段**: 「no-hit failed 35→13 (-63%); deterministic coverage 82%→93% (+11pp); success 误报 +0; 新规则 FP 率 P15 0% / P17 5% / P18 6% (对比既有 P6 22% / P14 12%)」(§261)

---

## 12. 失败三分类归因（agent-limit / scaffold-bug / benchmark-FP）

**当前值**（逐 run 并列）:
- R9755 (B0 cls dom, 191 failed): **100% agent-limit, 0 scaffold-bug, 0 benchmark-FP**；13 个纯视觉真盲区（`deterministic_candidate=false`）（§261）。
- R31194 (B0 cls dom): 35 深挖子集 **100% agent-limit / 0 scaffold / 0 benchmark-FP**（§277）。
- R24792 (B0 cls vision): 59 深挖 —— no-hit 50 = **47 agent-limit + 1 benchmark-FP + 2 unclear**；**0 scaffold-bug**；success-hit **9/9 纯误报**（hit_causal 全 false）（§290）。
- R32031 (B0 cls phantom_som): Tier-2 59 ep（54 no-hit 全覆盖 + 5 success-hit），**54 no-hit 全 agent-limit, 0 scaffold, 0 benchmark-FP**（§304）。
- R9725/R5313 (som): **scaffold-bug=0**（52 深挖 + P8 全 run 0）（§283）。
- 例外候选（§299.2 §299.3）：**T180** —— cls CSS 星级评分 widget 的 radio input 完全不暴露在 AXTree → "cls 任意 give N stars task 在 dom/som/vision 三 mode 都必 fail"；**T36** —— cross-site cls+shopping task，shopping localhost:7770 全程不可达 → "cross-site task 在 single-site fire 下必 fail"；**T216** —— 跨 R21557 dom + R31194 dom + R24792 vision + R5313 som **四 run/mode 全部一致** finish `item&id=82390`（语义正确），而 reference `id=66046` 在所有 4 run 所有 DOM artifact 中 grep 无命中 = 强 benchmark-FP 信号。

**演变**: §261 首次给 100% agent-limit → 被 3-AI 审计软化 → §290/§304 在 vision / psom 上重复同一结论 → §299.2/§299.3 找出前 3 个真例外候选。

**已作废**: §261 的 "191 个 failed 100% agent-limit" 的**表述**被 3-AI 审计软化（数值未撤）。

**caveats**（一字不改）:
- §261: "由 3-AI 审计软化为 '35 深挖证因 + 156 规则命中' — 156 是 presence 命中非 causation; 存在 35→191 外推; **P8=0 不证 scaffold=0**; 单 condition 无 cross-mode 对照"。
- §290: "benchmark-FP 后经截图 forensic 翻案 (task 40 改判 agent-limit) → 净 benchmark-FP≈0"。
- §299.2/§299.3: "T180/T36 是 B-number candidate 当时未正式登记; T216 加入 FP-candidate 排除列表待 6-mode 完整数据 finalize"。

**证据**: §261 §277 §283 §290 §299.2 §299.3 §304。

**原文片段**: 「T216 (Weiman Fruitwood Oval Cart $420) 跨 R21557 dom + R31194 dom + R24792 vision + R5313 som 四 run/mode 全部一致 finish item&id=82390 (语义正确), 而 reference id=66046 在所有 4 run 所有 DOM artifact 中均未出现 (grep 无命中) = 强 benchmark-FP 信号」(§299.3)

---

## 13. SoM-specific 失败机制 + P31/BUDGET 维度

**当前值**:
- **SoM failed_NO_HIT = 59 vs DOM = 22 (2.7×)**，其中约 50% 来自 P6/P15/P16 mode-gate dom-biased（规则库覆盖原因）+ 约 50% 来自真实 som-specific 四类机制（§299.1）：
  - (A) viewport-bound SoM marks 让 form submit button 看不见（T4/T75/T76 item_edit + T8 publish，每步 `[SOM_MARKS]` 与标注图仅含 viewport 内 **17-21 元素**，Save/Update/Publish 从未出现在任何 mark；同 task DOM mode success 因 `in_viewport_ratio` bug 保留全 form）
  - (B) scroll-after-rerender 丢 price/title context（T31/T32）
  - (C) SoM 标注框 + JPEG 压缩颜色感知偏差（T56 DOM 文本明确 Red/Black 而看图判 orange）
  - (D) gallery row 二维位置歧义（T14/T41）
- **P31（budget 耗尽未完成）失败集计数**：**vision 73 / dom 58 / som 49**（§290）；phantom_som 的 P31 = **69 (30.8%)**，是该 mode 头号维度（§304）。
- §306 taxonomy 的 **BUDGET 类 = 56-69（psom 69 最高）**（与 P31 定义不同，见 caveats）。
- **phantom-family 视觉盲**（§305）：pprompt no-hit 53 与 psom no-hit 51 重叠 shared 33 / unique 20；shared verify **6/6 = same_as_psom_visual_blind**；success-fire **16/16 全 FP**；P33 success-fire 0/5。

**演变**: §290 首次给三 mode P31 计数 → §304 在 psom 上确认 P31 主导 → §305 把 P34 "从 psom-specific 升级为 no-image-family-wide 定性"（纯 discover 不落码，ruleset 仍 v5）。

**已作废**: §290 的 P31 三 mode 计数被 §318 supersede（跨批次）。

**caveats**（一字不改）:
- §299.1: "约 50/50 的拆分是估计不是精确计数; 与 paper SoM 救视觉 单方向论点反直觉 — long form / commenting / gallery 任务 SoM 输 DOM"。
- §290 (P31): "P31 是三 mode 头号失败维度, 之前无规则捕获; 后续多节证明 **P31 跨 mode 含义不同 (finish-less artifact vs 真卡死) 不可裸比**"。
- §306: "本 taxonomy 独立于 P-rule ruleset 故不受 ruleset 版本不一致影响" → [聚合者推论] BUDGET 与 P31 是两套定义，D3 未给出二者的换算关系，**不要合并**。
- §305: "视觉盲与 prompt 格式 ([SOM_MARKS] vs AXTree) 无关 = no annotated image 的直接结构后果"。

**证据**: §290 §299.1 §304 §305 §306；`docs/analysis/vwa_classifieds/B0_som_classifieds_diag_digest.md`。

**原文片段**: 「SoM failed_NO_HIT = 59 vs DOM = 22 (2.7×), 其中约 50% 来自 P6/P15/P16 mode-gate dom-biased (规则库覆盖原因) + 约 50% 来自真实 som-specific 机制四类」(§299.1)

---

## 14. 计量层：latency / cost / carbon 的 cross-baseline 不对称与 B-1828 污染

**当前值**:
- **obs_prepare**：som（含画框）**143ms** vs dom（不画框）**0.077ms**；画框 **~140ms/step ≈ 1.3% step latency**（§260）。
- 画框经 `overhead_cost_per_ms` 折进 `cost.total` 的量级：**som 实测 8e-6/step，占 0.18%**（§260）。
- **B0 som vs dom 的 proxy input_tokens 中位数差**：som 中位 **4300** vs dom **3221**，差 **1079 ≈ 1280x720 图 token 1228 (= w×h/750)**（§260）。
- **cross-baseline cost basis 相差约 1000×**：B0 商业 API USD vs B1+B2 electricity-derived USD（§220.3 §222）；`api_usd ($0.005/1K tok)` vs `electricity_usd_derived ($0.0000005/1K tok)`（§222）。
- **carbon**：B0 `energy.source=disabled`（远程 Bedrock 本地 GPU idle = 本质 N/A）vs B1/B2 `psutil_profile/pynvml`（实测 B1 co2e_kg=5.42e-5）（§260 Part E）。
  > ⚠️ **更正 2026-08-02 (§407.20)** —— B1/B2 的 source **不是** `psutil_profile/pynvml` 二选一，
  > 而是 **一律 `psutil_profile`**：config 写着 `use_pynvml: true` + `hardware_profile:
  > a100_pcie_40gb`，但 **NVML 静默回落到 psutil，而 psutil 读 CPU 不读 GPU**。实测
  > `power_watts` 均值 **66.3 / 66.7 W**、CV = 0.03，而 A100 PCIe 40GB 满载 200–250W。
  > ⇒ 碳列是**壁钟时间换了个单位**（co2e~latency r = **0.9999** 四格全部），
  > **不报碳的理由是仪器没在测那个东西，不是它与 cost 冗余**。r=0.9999 是症状不是结论。
- **LLM evaluator (judge) API 成本估算**：VWA gpt-4o-mini judge 累计约 **150K-250K 次调用，约 $10-$80 USD 区间**（§220.3，"是估算区间不是账单实测"）。
- accounting reset 落地（§248）：StepRecordV2 +3 flag / EpisodeSummaryV2 +5 counter +3 cost；Smoke A 三 baseline "canonical+wasted==billed 全为 True"（§250.2）。

**演变**: §222 先发现 **B-1600**（retry-adjusted latency 的写入层缺失）与 **B-1601**（`aggregate_h10_pareto.py` 673 LOC，0 处引用 `cost_unit_basis`）→ §248/§249 accounting reset 落地 + codex 抓 4 个实现层盲区 → §252.1 re-Smoke A 验证字段落盘 → §260 系统审查计量层，定位 **B-1828（画框计入 obs_prepare）是唯一漏网真 bug** → §281 发现 AMENDMENT_03 的 sibling 遗漏（漏掉 canonical gate 的 `_load_cell_per_task`，H2a cost source 第 3 个）。

**已作废**: §250.2 Smoke A 结果被 §252.1 supersede（"这是 B-1794 修复之前的 vintage"）。

**caveats**（一字不改）:
- §260: "这是 B-1828 instrumentation 污染的量级估计; 方向保守 (修复后 P-SoM latency 优势更强); 受影响 = som/phantom_text 两个 phantom arm, phantom_prompt 走单独分支不受影响"。
- §260: "paper §1 cost canonical (metrics.py:452 = model billed) 不含 obs_prepare → B-1828 净影响是 latency 为主, cost canonical 不污染"。
- §260: "B0 不拆 image/text token (image_token_count_method 仅 B1/B2 exact_id_match) → cross-baseline 可审计性不对称, 但不影响 cost 正确性; 历史 B0_dom (router+archive) 约 32% step 带图 (router escalation), paper-grade R9755 纯 dom 0%"。
- §260 Part E: "cross-baseline 不对称已系统处理 (cost_unit_basis 分层 B-563 + cost_total_mixed_unit_warn B-565 + paper §1 脚注只报 within-baseline ratio 绝不跨 baseline 平均绝对 USD)"。
- §208.8（cross-family 结构差异）: tokenizer Qwen3 BPE **151,936** vs Gemma3 SentencePiece **262,144**；vision encoder B1 Qwen ViT dynamic-resolution vs B2 SigLIP 896×896 256-token-budget → "因此 [SOM_MARKS] flat text 的 Δtokens(mode)/tokens(DOM) 必须 per-baseline 比 ratio 不能跨 baseline 绝对相减; cls 视觉 grounding SR delta 不能单归因 LLM reasoning"。
- §222 (B-1600): "Pass-1 会结构性成功但 rollup 全空 → memory project_cost_latency_canonical_estimand 的三轴框架 post-fire 结构性失效"。
- §249（codex 抓的 4 个盲区，(3)(4) 是实测复现非静态推断）: 异常路径 EpisodeSummaryV2 漏 5 counter + 3 cost / published CSV/JSON 掉 8 个字段 / `compute_three_column_cost` 遇 `model:None` 直接 TypeError / 新 cost 字段不在 `_HERO_NUMERIC_FIELDS` strict guard 里 → `'1e309'→inf` poisoning 重新打开。

**证据**: §208.8 §220.3 §222 §248 §249 §250.2 §252.1 §260 §281；`p79/experiment/metrics.py`、`scripts/analysis/aggregate_h10_pareto.py`。

**原文片段**: 「som obs_prepare=143ms vs dom 0.077ms; 画框 ~140ms/step ≈ 1.3% step latency」(§260)

---

## 15. 单 condition wallclock / 吞吐 / 推理引擎 spike

**当前值**:
- **B0 cls 单 condition 实测**：dom **8.41h** + som **8.18h**（~8.3h/mode）；docker restart 开销 **75s/cond × 36 ≈ 45min**（§275）。
- Gate 2 canary R11315 吞吐：全程 **~7.8h (28.8 ep/h)**，latency **~110s/ep**（retry-adj 108.7s），**~14 steps/ep**（§273）。
- **B2 (Gemma/local)**：约 **13.6 min/episode**；HF generate clean 约 **17-18 tok/s**，runner 里约 **11 tok/s**；`output_scores` 只占 **7%**（非元凶）（§244）。
- **HF eager 4B**：A100 实测 **77ms/token**（qwen HF eager warmup 后 **26.35 tok/s**）；内存带宽下限约 5ms/token (~190 tok/s) → "差 >10× 全是 kernel-launch overhead"（§246）。
- **vLLM Tier-0 spike（model-only）**：qwen **26.35 → 112.38 tok/s (4.26×)**；gemma **20.76 → 102.78 tok/s (4.95×)**（§246）。
- **action parity**：vLLM vs HF eager qwen **90%** / gemma **70%**；torch.compile (default inductor) vs HF eager qwen **90%** / gemma **72.5%**（§246）。
- Phase 1b blocker：shop B0 **17.16h > 16h budget**（§239.5）。

**演变**: §275 用实测 wallclock 推翻编排器原写的 24h/6modes（4h/mode）——"偏低 2×; 36-cond 重估 **12-21 天**"（该重估被 §314 supersede，跨批次）。

**已作废**: 编排器原 ETA "24h/6modes (4h/mode)"；§275 的 36-cond ETA 12-21 天被 §314 supersede。

**caveats**（一字不改）:
- §275: "B1/B2 latency 无数据 = 最大未知"。
- §246 (vLLM): "只是 model-only; generate 占 step 约 55-65% → 单流 4-5× 顶多把总 wallclock 拉到 ~2× (100h→~50h), env 成新瓶颈"。
- §246 (parity): "spot-check 确认是真分叉非 artifact — 分叉发生在 thought 生成阶段 (greedy 下 kernel 数值微扰翻转 near-tie token → 级联改推理分支); compile 与 vLLM 的分叉集有交集 (qwen idx11/12 共同; gemma 7 个共同) → 根因是 4B 决策脆弱 (near-tie 多) 与引擎无关; step-parity 是最坏上界, episode-level SR delta 未验证"。scope: "40 个真实 dom step, 按长度谱 1096-4028 tok 分层抽样 … 一致性判据 = action_type + element_id"。
- §246 (env): "vllm 0.21.0 装在隔离 .venv-vllm (自带 tf 5.9.0/cu130) 与主 .venv (tf 5.8.1/cu128) 零污染共存; flashinfer sampler JIT 需 nvcc 而 A100 无 CUDA toolkit → 用 VLLM_USE_FLASHINFER_SAMPLER=0 + VLLM_ATTENTION_BACKEND=FLASH_ATTN 绕过; SamplingParams(logprobs=2) → 4/6 confidence (entropy 需全词表, vLLM 不给 → None), 正好与 B0 的 4/6 对齐; LLM(revision=...) 的 SHA pin 可用"；"装 CUDA toolkit 用 flashinfer 可能更快 (未测)"。
- §246 (HF eager): "generate_ms 只测 generate (prefill+decode); short-output(~100tok)+medium-input(~2.3k) 让 prefill 占比放大 effective tok/s"。
- §244 (B2): "benchmark 推翻了 output_scores 是慢速主因的假说"。

**证据**: §239.5 §244 §246 §273 §275；`scripts/spike`。

**原文片段**: 「qwen 26.35 → 112.38 tok/s (4.26×); gemma 20.76 → 102.78 tok/s (4.95×)」(§246)

---

## 16. VWA substrate 事实（DB seed / scored 计数 / evaluator 类型分布 / 难度排序）

**当前值**:
- **scored task count**：`scored_task_count('classifieds')=224` / `('reddit')=205`（bash 实跑 helper 核实，零 AI 推断，§224）。
- **cls DB 真实 seed 范围**（live DB, 2026-05-18）：`SELECT MIN, MAX, COUNT FROM oc_t_item WHERE b_active=1` = **1 | 84154 | 84149**；其余约 84137 条来自 sibling `osclass_craigslist.sql` 且 `fk_i_user_id=0`（系统 seed）（§227；§229 gemini 独立复核同值 1, 84154, 84149）。
- **evaluator 类型分布**（§247，B-1783 修正）：cls 只有 **31 个 program_html task（29 个 isolate）**，大头是 **url_match 131 / string_match 78**；red 是 **71**（不是约 136）。修正后 baseline cls 29/29 + red 71/71 PASS。
- **cls task 难度排序**：task 0-44 约 7% / 45-59 = 47% / 60-74 = 75%（§253.2）。
- **multi-site task**：reddit **40 个 (19%，其中 29 个提 wikipedia)**，classifieds 只有 **2 个**；6 个近期 cls run 里模型 emit 缺失 action 的次数 = **0**（0 invalid_action_type）（§245）。
- **partial 前缀构成偏差**（R2987 前 60 task，§242）：no_browser_required (string_match, B0 弱项) **43% (26/60) @ 7.7% SR**；isolated_program_html_context **15% (9) @ 0%**；agent_page **42% (25) @ 20%**；加权 = **11.7%**。

**演变**: §226 首版 B-1571 认定 cls sentinel 冲突 —— "oc_t_item (user-posted) = 12 (expected 0)"，并用文件级 grep 得到 12-ID canonical set（user_id=1, username='1', IDs 84143-84154, dt 2023-10/11）→ **§227 被 live-DB 证伪**。§220.3 的 paper §1 N=234/210 → §224 改为 **224/205**。

**已作废**:
- **12-ID canonical seed set 已死**（§227）："文件级 grep (只覆盖 fk_i_user_id=1 的 12 行) 因此严重低估 seed 范围; 用 12-ID canonical set 的 sentinel 实测报 'outside canonical seed set = 84137 (expected 0)' = **false-FAIL**"。
- paper §1 prose 的 **N=234/210** 已作废（§224）。
- B-1768 docstring 的 "232/234 cls program_html isolate" —— §247 实测为 **8× 高估**。

**caveats**（一字不改）:
- §247: "功能上无 bug (task 4+75 确实 isolate; url_match 不调 page.goto 故不受 eval-goto-timeout 影响) — 错的只是 COVERAGE CLAIM, 但那是 reviewer 一查就破的假 SBOM/§4 figure。新静态验证器 verify_eval_context_classification.py 首跑即抓到"。
- §245: "结论: cls/reddit 用 9-action 功能上完全够, restore 是 defensibility 保险不是能力需求; action set 在所有 condition 上 held-constant 所以不偏倚 #4 的 within-protocol paired estimand"。
- §229 (gemini live DB): "这条查询 Claude session 被 auto-classifier 挡住不能跑 → cross-AI 的价值是 trust-boundary 多样性而不只是 finding 多样性"。

**证据**: §224 §226 §227 §229 §242 §245 §247 §253.2；`scripts/maintenance/reset_vwa_sites.sh`、`scripts/maintenance/verify_eval_context_classification.py`。

**原文片段**: 「docker exec classifieds_db mysql 'SELECT MIN, MAX, COUNT FROM oc_t_item WHERE b_active=1' = 1 | 84154 | 84149; 其余约 84137 条来自 sibling osclass_craigslist.sql 且 fk_i_user_id=0 (系统 seed)」(§227)

---

## 17. Fire-3 ~ Fire-6 的事故计量与基础设施可靠性

**当前值 / 事故账**（时间序，逐事件原值）:

| 事件 | 计量 | § |
|---|---|---|
| Fire-day catastrophe (B-1581) | cls B0 dom 烧掉 **110 tasks (task=29..110)**，**324 次** `RuntimeError: VWAWrapper.reset() detected an active asyncio loop`；watchdog 每 task 重试 3× 共约 **330 次** retry-cycle | §228 |
| 同上，A100 runner.log 实测 | `VWAWrapper.reset() detected an active asyncio loop` = **369 occurrences**；`success=False steps=0` = **123 zero-step failures**；原文估计若不停，Pass-1 会浪费 **~1-2 周 wallclock 且 55%+ task 污染** | §229 |
| Fire-3 死亡时间线 | 21:28 启动 → 22:30-01:37 cls 完成 75 tasks → 00:53:52 task 75 evaluator `page.goto(...id=84148)` 30s timeout ×3 → EvaluatorUnavailableError（B-486 quarantine）→ runner abort → **00:54:39 master 轮询见 PID 死判 done 启 red** → red 4h 内只跑 **14 episodes（~17 min/ep vs 正常 5 min/ep）** → 04:56:56 触 B-1665 **14400s wallclock SIGTERM** | §237.1 |
| P0-2-B* smoking gun | `queue_phase1_paper_grade.sh:728-740` master 用 **`kill -0` 轮询**——只能判活/死，判不了 exit code → cls chain 的 quarantine **exit 1 被静默解读为 done** | §237.2 §237.3 |
| 跨 fire 同类 evaluator 故障 | Fire-3 cls task 75 + Fire-4 cls task 75 + Fire-5 cls task 4 —— **3 次 fire 3 次 quarantine**，同一 error class（Page.goto Timeout 30000ms）但 task ID 不同 | §239.4 |
| Fire-6 cls task 4 abort | id=84144，`EvaluatorUnavailableError` Page.goto 30s×3 → chain fail-closed abort rc=1（red 没启，无脏数据）；**这是 Fire-3/4/5/6 第 4 次同一处**。反证：DB 里 item 84144 `b_active=1`（没被删）；curl id=84144 = **0.17s** 健康 | §253 |
| 根因（Stage C） | agent 进入 stateful EDIT form（**154KB DOM, 2582 个 inline city `<option>`**）→ evaluator 的 program_html `Page.goto(item_url)` **复用 agent 累积态的 runner page** → 30s timeout；Path Z（Playwright MCP, 已认证, FRESH page）同一 goto 仅 **639ms** | §241 |
| A100 真实负载复现 | cls task 4 + 75 重跑；task 75 success=True, eval isolated, goto **10.4s** 无 timeout | §241 |
| B-1803 re-fire 确认 | R9755 task 0→1→3→4（task 4 跑约 14 min agent steps）→ maxtask=5, evalerr=0, **零 EvaluatorUnavailableError** | §253.1 |
| Fire-6 历史 `.done` rc 分布 | **8× rc=1 + 1× rc=143，全 cls**，时间跨度 5/18-5/22 | §276 |
| B-1839 docker restart 效果 | R31194 reset log 5-table sentinel: **items_contam=0**，实时 oc_t_item=84149 | §276 |
| 图像落盘 saga | B-1832 (.tmp 后缀 → PIL encoder 推断失败): **5344 次 save 全失败**，som png=0 / screenshot png=0（run 到 101 ep）· B-1835（函数内局部 `import os` 遮蔽）: **UnboundLocalError ×353**，EP=8 时 PNG=0 · 生产验证 closed: R10016 **img=798, errflood=0** | §265 §268 §269 |
| B-1836 eval retry 关键词不匹配 | 3× forensic 全部 **attempt=0**（Fire-5 task4 + Fire-6 task32 + R10016），0 retry 痕迹；误导 message 写 "All 3 retries exhausted" 实际只 1 次 | §269 |
| B-1861 watchdog ntfy read-timeout | Gate3 cls B0 ptext R19776 被 SIGTERM kill（**180/224，丢 44 ep**） | §288 |
| B-1868 pre-fix 物理删除 | R14849 (B0 P-SoM cls) task 142→143 边界 session lost → watchdog streak=3 → **silent delete 3 ep (task 143/144/145)** + B-863 reaper 5min 内 purge = **forensic-irrecoverable** | §303.1 |
| B-1848 Playwright driver-wedge | MainThread 阻塞在 Playwright 同步 API 事件循环 select/epoll；cls docker 健康（curl 0.22s, Up 7h）；AWS proxy 443 **CLOSE-WAIT 25B 未读 = 症状非根因** | §280 |
| VWA docker 冷/热启动 | classifieds 冷 **9.96s** vs 热 **0.085s (~117× speedup)**；shopping 冷 **14.68s** vs 热 **0.13s**；另引 0.103s 热值（curl -w 实测非估算） | §222 |
| 其他固定开销 | GCP metadata 探测 **9s × 36 conditions ≈ 5 min**（§232）· reddit GLM PLAYBOOK cold-start 最坏 **255s** → launch.sh:200 sleep 30→300（§208.8）· B0 som cls **p95_step_latency = 34.4s**（task 75 实测，逼近 30s timeout）（§262） |

**演变**: §228 首次把 fire 崩溃归因到 VWA submodule `async_envs.py` 的 `asyncio.run()` → **§237 明确 RETRACT 该归因**："P79 从不实例化 AsyncScriptBrowserEnv, async_envs.py 在 P79 路径里是死代码; 那 146/369 次 stale-loop 命中是 B-1581 patch 检测到自己上一集安装的 loop"。evaluator 挂死从 "agent-modified substrate" 归因（§253 推翻）→ Stage C 的 "复用 agent 累积态 page"（§241）。

**已作废**:
- §228 的 `async_envs.py` 归因（被 §237 RETRACT）。
- §233 Fire-3 attempt #4/#5 的 orchestrator 缺陷描述被 §237 supersede。
- 旧 next_steps 里 "agent-modified substrate" 的 evaluator 归因（§253 反证）。

**caveats**（一字不改）:
- §241: "Path Z 是隔离环境 … **10.4s (vs Path Z 639ms) = A100 真实累积负载 → 这类复现必须在 A100 跑, 隔离 MCP 会把 margin 低估**"。
- §253.1: "B-1803 只保证 eval 能执行不挂, 不改 scoring; task 4 的 success/fail 仍是 agent 真实表现"。
- §265: "estimand 不受影响 (save 在计时窗外且被 catch), 真正受害是 gallery 视觉 artifact 全丢; 外层 try/except 把 fatal 降级成 per-step warning 掩盖了它"。
- §268: "唯一抓到它的是生产环境 + P2-1b artifact-count gate (smoke MockEnv 无图跳过, loop-replica 独立脚本无函数遮蔽, py_compile 不查名字解析)"。
- §288: "best-effort 通知 (ntfy) 杀关键基础设施 (watchdog) 的反模式; chain fail-safe 设计正确但放大了 1 行漏洞"。
- §303.1: "B-1777 (paper_grade=True ⇒ never delete+retry) 只 cover `_can_auto_retry` error-retry path, session-cleanup path 漏了 sibling guard 整年 = 完美的 scope gap 实证; 该 3 ep 已物理 gone 故 catalog 需显式 disclose forensic gap"。
- §280: "网络层证据 (CLOSE-WAIT) 极具误导性; 唯一可信是执行栈 (py-spy) 不是周边症状"。
- §276: "机制是 fresh 容器让 reset SQL/HTTP 可靠, docker restart 本身不清表 (DB named volume 持久)"；"仅第一 condition 已验"。
- §236.3 (B-1760, DOM 模式 screenshot 缺失): run_dir `find -name '*.png'` 返回 **0 个 PNG**；前 25 tasks 的 **91/91 step record 的 `artifact_paths.screenshot = null`**；对照 archive `.../step_000/screenshot.png = 273 KB` 存在。"回归窗口 2026-05-15 有 → 2026-05-18 无; 表层逻辑 archive↔HEAD 字节相同 … **根因不明需运行时插桩**"。

**证据**: §208.8 §222 §228 §229 §232 §233 §236.3 §237 §239.4 §241 §253 §262 §265 §268 §269 §276 §280 §288 §303.1；`docs/checkpoints/quarantine_registry.jsonl`。

**原文片段**: 「cls B0 dom 烧掉 110 tasks (task=29..110), 324 次 RuntimeError: VWAWrapper.reset() detected an active asyncio loop; task 0 之后每个 task 都在 0 steps 立刻失败」(§228)

---

## 18. P79 vs upstream VWA 的 prompt / action-set divergence

**当前值**:
- **12+1 处 prompt divergence**（§243）：#1 zero-shot（P79 0 examples vs upstream 5-shot "In summary"）· #4 `wait` action 是 P79 发明且是 parse-fail sink（`action_utils.py:392`，upstream ActionTypes 无 WAIT）· #9 "NEVER give up / EXHAUSTIVE search"（`_shared_vl_utils.py:86-87`，upstream 无）· #5 丢掉 hover/press/new_tab/goto · #13 max_steps 值 30=30 但语义偏离（P79 `step_idx += 1` 无条件含 injected wait，例 task 53 = **1 真决策 + 29 wait**；upstream 30 = 30 agent decisions）· 另有 JSON serialization / confidence field / tab 改名 (go_back→back) / scroll superset / select_option 提升。
- **action 层四层一致性**（§245）：prompt / validator / wrapper / B0 schema **四层内部完全一致无 silent drift 无 active bug**；9-action 共同集 = click/type/scroll/wait/back/forward/finish/select_option/tab_focus；相对 upstream 缺 hover/press/new_tab/close_tab/goto，自创 wait，select_option 是一等增强。
- **restore 后新 action 的实际使用率**（§253.2）：R9755 **666 步里 goto×1，hover/press/new_tab 各 0**；prompt 文件 R12090 之后只被 commit 4141e0b 动过 = **纯加法**，core rules 1-8 + click/type/select_option/scroll schema + search 引导**一字未改**。

**演变**: §243 列出 divergence → §245 做四层一致性核查 + 需求分析（multi-site task 靠 tab_focus 即可）+ restore patch（18 用例，本 scope 68 passed / 1 skipped，后续全套 1171 passed）→ §253.2 用生产数据证明 restore 不改行为。

**已作废**: 本批内无。

**caveats**（一字不改）:
- §243: "P79 是完全替换 upstream CoT prompt 不是增量修改; #1/#9/#5 标 validity-risk, #4 标 P79 design 非 bug, 多数其余标 disclose 级"。
- §245: "action set 在所有 condition 上 held-constant 所以不偏倚 #4 的 within-protocol paired estimand"。
- §245 (cross-AI 复审): codex PASS 高质量 350KB / 5 findings 全核实属实；gemini PASS 但 **1 个 P0 幻觉已 refute**（称 press 序列化返回 `hover [key]`，实际 `vwa_wrapper:1486 hover / :1493 press` 均正确 + `test_serialize_press` 铁证，"gemini 把相邻分支看串"）。
- §245 测试: "当时另有 2 个 test_som_and_schema tool_call_* contract 失败, 属并行 GRL session commit dbb1bda 的 pre-existing, 非本 patch"。

**证据**: §243 §245 §253.2；`p79/agents/_shared_vl_utils.py`、`p79/backends/action_utils.py`、`tests/test_action_set_restore.py`。

**原文片段**: 「#13 max_steps 值 30=30 但语义偏离 (P79 step_idx += 1 无条件含 injected wait, 例 task 53 = 1 真决策 + 29 wait; upstream 30 = 30 agent decisions)」(§243)

---

## 19. learned router (L1) 与 rule-based router 信号

**当前值**:
- **archive L1 router 模拟（Variant B balanced LR）per-site lift**：**cls +2pp / red −3.95pp**（§218.3）；§221.1 复述为 **cls +2.02pp / red −3.95pp**；到 6-cell 的 FE 投影 **−0.97pp**。
- **rule-based router 的 DOM-size 信号**（B-1401，§219）：`section3_definition.md:167` 已 disclose **`dom_size_threshold=12000` 只触发 0.14% 的 step**（cleaned-AXTree 体制下有效 escalation **100% 来自 streak counter**），而 `section1_intro.md:29` 仍用 raw DOM-size signal 卖 rule-based router。
- **实现规模**（§218.4 §218.6）：`extract_50_features.py` 521 LOC（5 numeric + 15 binary）· `train_l1_router_with_mi.py` 497 LOC · `aggregate_h10_pareto.py` 629 LOC；Chunk A/B/C = 4 files +1541 / 2 files +822 / 5 files +1479，invariants 28-28 / 18-18 / 26-26 PASS；自有 72/72 PASS。
- **两个实现缺陷**：
  - B-1804（§254）：`train_l1_router_with_mi.py:202` 的 `mutual_info_classif(X, y, random_state=seed)` 中 X 是 dense hstack (L186) → sklearn `discrete_features='auto'` 对 dense 输入默认全当 continuous → **15 个 binary 特征被用 k-NN (Kraskov) 熵估计**，{0,1} 轴上大量 distance-0 ties 靠 seeded 噪声打破 → binary MI 有偏且下偏；另 `N_SELECTED=18` (L52) 写死无 K-sensitivity。
  - §255：cls task config 中 `reasoning_difficulty` 字段为字符串（会让 `extract_50_features.py:221` 的 `int()` 崩）的比例 = **234/234**（bash 实测）。
- **S3 /stress P0-1**（§311，三谱系独立命中）：`aggregate_h10_pareto.py:721` `operational_gate_passed` 用 **non-dominance 弱判据**，而 `router_strictly_better` (:371 θ-CI 下界>0) 算了没接进 verdict → **零学习的退化 router 落 phantom_som 点即非支配通过**。

**演变**: §218.3 首次给 archive 模拟 → §221.1 明确裁定 "archive 不是 prereg substrate"；§254/§255 核出实现缺陷；§311 抓 gate 判据缺陷。

**已作废**: 无（archive 模拟数从未升为可判 H10 的值）。

**caveats**（一字不改）:
- §218.3: "用于 Q4 estimand 选择的 informative anchor; archive 不是 prereg substrate (§221 明确裁定)"。
- §221.1: "原文明确: **这些数不能用来判 H10**; 到 6-cell 的 FE 投影 (-0.97pp) 是投影不是测量"。
- §254: "本 session 对照真实代码逐条核过 (铁律: 不信任何 critique 的 file:line); P2-1 score_func 用 lambda 不可 pickle 属 fragile"。
- §255: "runtime main.py:2242-2246 的 except:pass 会静默归零掩盖崩溃"。
- §219 (B-1401): "reviewer R5 的 scientifically dishonest 攻击面; gemini Mode C unique P0 OOB"。
- §311 scope: "learned router pipeline + workshop framing, **audit-only 不改 code/prose**"。

**证据**: §218.3 §218.4 §219 §221.1 §254 §255 §311；`scripts/analysis/train_l1_router_with_mi.py`、`scripts/analysis/extract_50_features.py`、`docs/checkpoints/router/l1_archive_simulation_2026-05-16.md`。

**原文片段**: 「cls +2.02pp / red -3.95pp … 原文明确: 这些数不能用来判 H10; 到 6-cell 的 FE 投影 (-0.97pp) 是投影不是测量」(§221.1)

---

## 20. 统计 / 分析管线的实现缺口（prose 承诺 ↔ 代码真值）

**当前值**（逐条为独立缺口，多数已修，此处记**当时实测**）:
- **B-1301**（§215）：prereg §2 H1 L85（B-1009 amendment）承诺 primary = **1000 次 paired-bootstrap pool replicate 的 percentile p**，但 canonical producer 仍算 `p_one_sided = 1 − Φ(z)`（normal-Z Wald），canonical line 857 读 `h1_pass = fe['gate_passed']`；commit 1fb9d7a 只动了 prose 没加 bootstrap 代码路径。smoke 两种 p 值：**percentile p = 0.0000 / normal-Z p = 5e-06**（§215）。
- **TOST 退役后的 zombie producer**（§213.2）：**7 个 live 站点** —— `aggregate_phantom_lift.py:445 def + :494 alias + 4 call sites (L668/687/688/730/757/758) + 6 CSV 列 + MD warning + secondary table col`；`preregistration_decision_test.py:328-517 def + caller + CLI --TOST-delta-pp`；`p79/experiment/analysis.py:1170 _bootstrap_tost_paired_success + results['tost_equivalence'] + success_tost_eq_delta1pp`。
- **AUROC multiplicity vapor**（§213.2）：`aggregate_routing_auroc.py:127-136` 只 emit AUROC + CI，**没有 p 值**；prereg §4 line 423-424 却承诺 Holm + BH q-value → **数学上不可能**。
- **meta CSV 写序 bug**（§213.2）：Holm 在 L426-444 计算，CSV 在 L361-366 写 → **machine CSV consumer 拿到未校正的 `p_re_one_sided`，Markdown reader 看到 corrected 值**。
- **Holm 实现分散度**（§213.2）：**5 处独立实现**，边界 `<` vs `<=` 漂移（`analyze_confidence_calibration` 用 `<=`，其余用 `<`）。
- **cell_key NameError**（§240）：`analysis._compute_statistical_tests` 在 3 处 `flat_rows.append` 引用 `cell_key`，commit b72a3e7 只加了 `_cell_key()` helper + 列，漏了赋值行 → 自 b72a3e7 起**任何 ≥2-condition run 的 runner 后置分析都静默 exit 1**，McNemar / paired-bootstrap-lift / Holm 全无产出（单 condition run 不受影响）。
- **AMENDMENT_03 sibling 遗漏**（§281）：只列了 `aggregate_cost_electricity` + `aggregate_h10_pareto` 两个 producer，**漏掉 canonical gate 的 `_load_cell_per_task`（H2a cost source 第 3 个）**。
- **VWA lock split-brain**（B-1400，§219）：`preflight_v2.sh:413` pin 在 **2f9b0b4**，`snapshot_env.py:184` 的 `VWA_LOCKED_HEAD_SHA` 仍 pin **1c3a615** → orchestrator 通过但 runner 在 provenance capture 处死（`head_match=False` under `P79_PAPER_GRADE=1`）。
- **seed 决定性 sibling gap**（B-1602，§222）：`_seed_global_rng` (runner main.py:125) 只读 env；只在 yaml 里设 paper_grade 的用户拿到 LAX 的 `torch.use_deterministic_algorithms(warn_only=True)`，而 evaluator + diagnostic_controls + backend 传播都走 STRICT。

**演变**: 这批缺口集中在 A2.3c/A2.3d/A2.7/§240/§281 被逐个抓出；共同模式 = **sibling-propagation gap**（"localized 修复必查 sibling set 非仅 touched file"，B-1853，§281）。

**已作废**: §213.2 的 "A2.3a B-957 的 retirement 只改了 power_analysis.py + prereg" 这一认知（sibling 全景由 §213.2 补全）。

**caveats**（一字不改）:
- §215 (B-1301): "3-AI overlap (Claude F1 + codex F1 独立抓到, gemini F5/F7 间接); 若 fire 会 emit prose-promise ↔ artifact-truth 不符的 phase1_full_prereg_decision.json"。
- §215 (smoke p 值): "smoke 用途; 原文指出 reviewer 无法复现这个引用值正是 B-1301 的问题之一"。
- §213.2 (Holm 分散): "P2 按 Q16 defer, 待批量 centralization 重构"。
- §213.2 (CSV 写序): "reproducibility gap: 同一 artifact 两个 reader 看到不同数"。
- §240: "test_runner_integration 的失败日志里能看到 `[runner] Post-condition analysis exited 1 ... NameError: cell_key`"。
- §219 (B-1400): "sibling-propagation gap: A1.18-re Chunk 2 扫了 Makefile + preflight + paper §4.X.11 + prereg 但漏 snapshot_env.py + locked_versions.md header 行"。
- §222 (B-1602): "B-868 自己的注释块 (L688-690) 就描述了这个并行不对称, 但修复只碰了 fingerprint 一半"。

**证据**: §213.2 §215 §219 §222 §240 §281；`scripts/analysis/aggregate_phase1_full_prereg_decision.py`、`scripts/analysis/aggregate_phantom_meta.py`、`p79/experiment/analysis.py`。

**原文片段**: 「Holm 在 L426-444 计算, CSV 在 L361-366 写 → machine CSV consumer 拿到未校正的 p_re_one_sided, Markdown reader 看到的却是 corrected 值」(§213.2)

---

## 21. phantom 三臂的 2×2 构造（设计层核查）

**当前值**（§260 Part C，全 baseline / 全 site / 三个 phantom mode）:
- 干净 2×2 = prompt[DOM/SoM] × text[AXTree/SOM_MARKS]，**全 phantom 无图**：
  - **P-text** = DOM prompt + `[SOM_MARKS]` text（只改文本）
  - **P-prompt** = SoM prompt + AXTree text（只改 prompt）
  - **P-SoM** = SoM prompt + `[SOM_MARKS]` text（= som 去图）
  - `marked_image=None`
- som input token 随 mark_count 增长 **4281→4763→4946** = `[SOM_MARKS]` 文本量。
- dispatch 位置：prompt 轴在 `_shared_vl_utils.py:260-265`（3 agent 共享），text 轴在 `som.py:345-365`。

**演变**: §260 Part C 一次性核查通过。

**已作废**: 无。

**caveats**（一字不改）: "设计层正确但计量层当时有 B-1828 (画框进 obs_prepare)"。

**证据**: §260 Part C；`p79/agents/_shared_vl_utils.py`、`p79/experiment/som.py`。

**原文片段**: 「P-text = DOM prompt + [SOM_MARKS] text (只改文本); P-prompt = SoM prompt + AXTree text (只改 prompt); P-SoM = SoM prompt + [SOM_MARKS] text (= som 去图); marked_image=None」(§260 Part C)

---

## 22. cross-AI audit（Mode A/B/C）的产出、runtime 与幻觉率

**当前值**（每轮 audit 的原始计数，**不做跨轮平均**）:

| 轮次 | Mode A | Mode B (codex) | Mode C (gemini) | 合并 | § |
|---|---|---|---|---|---|
| A2.2 | 7 | 7 / 188s | 7 / 235s | 21 findings, 13 OOB, 2 overlap; Phase 4 4/4 | §208.1 |
| A2.3a | — | 7 / 3 OOB / 230s / 3-3 | 5 / 2 OOB / 50s / 2-2 | — | §209 |
| A2.4a | 7 | 8 | 7 | 22 findings, 16 unique OOB, 2 个 2-AI overlap; B 193s + C 254s | §211.1 |
| A2.4b | 9 | SKIPPED（wrapper `set -u` 下 `ZSH_VERSION` unbound） | 7 / 255s / 9.6KB | Phase 4 幻觉 **2/7 = 29%** | §212.3 §212.4 |
| A2.3c | 7 / 5 OOB | 7 / 4 P0-OOB / 255s / 8.8K / 3-3 | 6 / 3 OOB / 86s / 9.0K / **5/6（1 幻觉 = 17%）** | — | §213.1-3 |
| A2.3b | — | — | — | 22 unified, 9 OOB = 41%; 3-AI overlap 1 / 2-AI 1 / codex unique 3 / gemini unique 2 / Claude unique 4 | §214 |
| A2.3d | 12 / 3 OOB | 9 / 5 OOB / 690s / 41248B / 4-4 | 7 / 3 OOB / ~420s / 12682B / 3-3 | 22 unified → ~17 unique | §215 |
| A2.6a | 8 / 3 OOB | 8 / 4 OOB / 222s / 4-4 | 5 / 2 OOB / 219s | 17 unique = 8 P0 + 8 P1 + 1 P2; **82% unique（§A2 史上最高）** | §216 |
| A2.6b | 9 (4 OOB) | 9 (4 OOB) | 7 (3 OOB) | 22 distinct, 16 unique-lineage + 3 个 2-AI overlap; 17 fixes | §217 |
| A2.5 | 7 / 3 OOB | 6 / 4 OOB / 142s / 2-2 | 5 / 2 OOB / 231s | 1 个 3-AI + 3 个 2-AI + 9 个 1-AI unique | §218.1 |
| A2.7 | 8 / 5 OOB | 8 / 5 OOB / 418s | 7 / 4 OOB / 249s | 17 dedup vs per-lineage 平均 8 = **2.1×**；两个 P0 fire-blocker 都是 lineage-unique | §219 |
| A2.9 | 15 / 6 OOB | 10 / 3 OOB / 199s / **2.07MB 输出** | 10 / 6 OOB / 417s / 12.7KB | 16 unique cluster（3-AI 6 / 2-AI 7 / 1-AI 3），**OOB 11/16 = 68.75%** | §220.1 §220.2 |
| A2.8 | 10 / 3 OOB | 11 / 4 OOB / 242s / 175 行 / **Phase 4 6/6 REAL** | 8 / 4 OOB / 62s / **8/8 REAL 但行号偏 4-100 行** | 25 dedup（29 raw − 4 overlap），11 OOB，**86% lineage-unique** | §221 §221.4 |
| A2.6c | 10 / 6 OOB | 8 / 5 OOB / 240s / 3-3 | **FAIL**（Google 服务端 429 capacity on gemini-3.1-pro-preview，内部重试 3 次后放弃）→ A+B 为 canonical | 12 fixes | §223 |
| A1.25 (post-fire) | 10 / 5 OOB（冷读 11 文件 ~95 min） | 9 / 6.7 min / 7.4KB | 7 / 3 OOB / 8.9KB | — | §229 |
| A2.11 | 8 / 4 OOB | 5 / 3 OOB / 193s / 3-3 | 5 / 1 OOB / 291s（F1 行号幻觉） | 12 findings (5 P0 + 5 P1 + 2 P2) | §232 |
| witness bug | 5 / 3 OOB | 3 / 2 OOB / 5362B | 3 / 2 OOB / 4407B + 2 条新 reviewer-3 OOB | — | §231 |
| Fire-3 recovery | — | 9 / 4 OOB / 4.8 min / 3-3 | 7 / 5 OOB / 6.7 min / 3-3 | 18 findings (8 P0 + 9 P1 + 1 P2); 3-AI overlap 1 / 2-AI 4 / codex unique 9 / gemini unique 2 | §237.2 |
| Phase 0 R2 | 5（非 push-blocker） | 6 / 4 OOB / 8970B / 3-3 | 5 / 3 OOB / 8015B / 3-3 | 9 push-blocker (4 P0 + 5 P1)；两家均 ~5.3 min | §238.1 |
| accounting | 8 | PASS 264s / 3-3 + 1 复现 TypeError | PASS 46s（<60s 被 flag 但 6741 tokens + 2/2 反驳 failure premise → accept） | — | §249 |
| GRL 边界 | — | PASS / 3-3 / 308s | — | 2 P0 + 4 P1 越界 | §247 |
| S3 router | 6 / 3 OOB | 9 / 6 OOB / 296s / 5-of-5 | 5 / 2 OOB / 379s | 16 dedup（P0×4 / P1×8 / P2×3） | §311 |
| 6-lineage split-session | 50 findings / 25 OOB / 18 P0（1 hub + 5 track 并行，wallclock **11 min vs 串行 2-3h**） | — | — | Phase 4 11/11 verify real + 2 个自捉 file:line 幻觉 | §239.1 §239.2 |

- **Mode A 自审的相对贡献**（§238.3）：Round 1 **7/18 = 38%**；Round 2 **5/14 unique = 36% 但其中 push-blocker 占比 0%**，而 cross-AI findings 里 **100% 是 push-blocker**。
- **gemini 行号系统性偏移**（§207.4 §216.6 §221.4 §232）：A2.1 "F1 cited paper_planning.md:88 实际 89; F5 cited section2_background.md:37 实际 17"；A2.6a "cited paper_planning.md:60 / preregistration.md:299 / :438-439 / :447 / paper_planning.md:121-131 → 实际 L45 / L314 / L458 / L460 / L134"；A2.8 "行号偏 4-100 行"；A2.11 "F1 行号幻觉 (规则存在但引错 L215, 实际 L169)"。
- **gemini 计数幻觉实例**（§220.4）：称 §4-limitations 有 **44 处** cross-AI 引用；实际更宽 regex grep = **27（虚高 37%）**；Claude 自数 §1=14 + §4-lim=11 + §8=10 (35+)。

**演变**: 从 A2.2 的 21 findings / 2 overlap，到 A2.6a 82% unique、A2.8 86% lineage-unique；A2.7 引入结构化 scope-split 得 **2.1× 产出倍数**；§239.1 证明**同 lineage 拆 6 条 cold session 也能产出 50 findings / 11 min**。

**已作废**: 无数值作废；但 §229 Mode C 首次 dispatch "传了 prompt 内容而非路径 → wrapper 拒 (wallclock=0s rc=1)"，retry1 才成功。

**caveats**（一字不改）:
- §207.4: gemini 行号偏移 "verbatim quote 匹配 → spot-check 通过; 属 A1.9 已记录的 lineage idiosyncrasy 不是 hallucination"；§216.6: "quoted text 全部 REAL, 只是行号错; **修之前必须 grep-verify**"。
- §212.4: "29% 在 v7.8 lineage idiosyncrasy 阈值 (<5/7) 内 → per-finding trust modulation, 不整轮 retry; 两个被标幻觉的 attack 本身 valid"。
- §220.7: "方向性结论 (引用很多) 仍成立; Phase 4 1/7 幻觉 < 5/7 阈值 → 只打 per-finding ⚠️ 不整轮 retry"。
- §220.2: "**2MB 不是幻觉/失败, 是 codex CLI v0.130 把 reasoning trace 全流出来了**"。
- §221.4: "Mode C 62s 低于 1/3 阈值但内容 substantive"；§209: "Mode C 按 PASS-WITH-CAVEAT 接受, 与 A1.24 的 73s 同类张力"。
- §238.3: "结论: Mode A 对自己的 fix-attempt 做自审, **无法逃出同一 lineage 的 framing/code 盲区**"。
- §239.2: "2 个自捉幻觉: Hub-H6 引 metrics.py:405 实际 :480 (Track B 抓); Track E F1 引 analysis.py:1428-1432 实际是 Holm-Bonferroni 代码 (Track E 自报) → 同 lineage 拆 cold session 仍有 cross-read 价值"。
- §229: "Mode A 被 auto-classifier 挡住不能跑 mysql probe = live-DB 第 3 层验证空洞"；§217 codex F2: "纯 prose audit 结构性看不见 (只在 code docstring 里)"。
- §252: gemini cold-context "独立重新推出 4 条已 lock 的决策 … 唯一真正新的设计 ask = GRL-off ablation"。
- §271 教训: "**Mode A 干净 ≠ fix 干净, 我自己的 fix 也会引入 P0**"。

**证据**: §207.4 §208.1 §209 §211.1 §212.3-4 §213.1-3 §214 §215 §216 §217 §218.1 §219 §220.1-7 §221 §223 §229 §231 §232 §237.2 §238.1 §238.3 §239.1-2 §247 §249 §252 §311。

**原文片段**: 「Round 1: Mode A 7/18 = 38% of unified findings; Round 2: Mode A 5/14 unique = 36% 但其中 push-blocker 占比 0%, 而 cross-AI findings 里 100% 是 push-blocker」(§238.3)

---

## 23. repo 测试数与 fix wave 规模

**当前值 / 演变链**（repo-wide pytest passed 数，按时间）:
**852**（pre-A2.1）→ **866** passed / 10 skipped / 0 failed（+24 A2.1 新测试；净 +14，§207.4）→ **848**（B0 migration 后，排除 pre-existing `glm_client` import error，§210）→ **957** pass / 10 skip / 1 inherited（A2.5，§218.6）→ **1074 → 1088**（31 red tests 收口，commit ac925a1，§240）→ **1118**（Fire-6 RCA Stage C，§241）→ **1162**（GRL audit，§247）→ **1171**（action-set restore 后全套，§245）→ **1192** / 10 skipped（+21，accounting，§248）→ **1197**（B-1786 后，§249）→ **1207**（B-1794 后，§251.3）→ **1221**（pre-Fire-6 /stress 落地，含 12 个反向不变式 + P1-6 crash regression，§252）。

其他规模计量：
- A1.25 fix wave：**13 fixes B-1582~B-1595（B-1592 空号），11 files，263 insertions / 55 deletions**，全部 py_compile + bash -n PASS（§229）。
- A2.8 落地：**16 files，+258/-73 LOC，12 active fixes（B-1550~B-1561），4 commits + git tag `prereg-h10-locked`**（§221.3）。
- Phase 0 收口：**2 轮 /stress，27 unique findings（18 + 9），20 fixes 落地（11 + 9），2 个 P2 延后，5 commits 净 +1198/-61 LOC，post-R2 14/14 tests pass**（§238.4）。
- Phase 1 硬化 + Phase 2 telemetry：Phase 1 **11 fixes，313 +/- 23 LOC 跨 8 files**；Phase 2 **24 个新 schema 字段（4 step + 20 episode）**，`tests/test_schema_4place_sync.py` 138 LOC 7 invariants 全 PASS（§237.3 §237.4）。
- Mode A solo 深入审：**8 findings，4 OOB = 50%；6-chunk commit chain；22 个新回归测试（Chunk1 6 / Chunk2 7 / Chunk3 9），0/22 fail**（§222）。
- deferred subset 逐条核验（§239, 2026-05-21）：10 paper/stats + 1 Phase 1b → **7 真 live（B-1787~B-1793）+ 2 已有 prereg defense 的争议点 + 2 已失效**。

**已作废**: 无（均为时点快照）。

**caveats**（一字不改）:
- §207.4: "净 +14 因为 parallel session 的 10 个测试尚未进 tests/ 子树"。
- §210: "那个 import error 是 sys.path 问题, 不在 migration scope"。
- §237.4: "20 个 episode 字段里 10 个 attempt-lineage + 8 个 footprint 目前全 None/0 (等 checkpoint-restore 基建 / aggregator 侧计算延到 paper-2)"。
- §229: P2-16~P2-21 六项延后（含 "reset_vwa_sites.sh:155 user_count > 20 启发式阈值未实证"）。
- §239 (2026-05-21): "核验结果印证 user 的「P0 标签膨胀」判断; raw track 内容已抢救进 master_bug_catalog"。

**证据**: §207.4 §210 §218.6 §221.3 §222 §229 §237.3-4 §238.4 §239 §240 §241 §245 §247 §248 §249 §251.3 §252。

**原文片段**: 「2 轮 /stress; 27 unique findings (18 Round 1 + 9 Round 2); 20 fixes 落地 (11 + 9); 2 个 P2 延后; 5 commits 净 +1198/-61 LOC; post-R2 14/14 tests pass」(§238.4)

---

## 24. Provenance / witness / SBOM 的计数与陷阱

**当前值**:
- **DOI 1 canonical witness**：interim 扫描 `artifact_existence_check_doi1_interim_20260518T144258Z.txt`，**3295 bytes，SHA-256 7563f0d55b651b604746ef0498fba3439ad7d7e130af97f0adda55e2bc7f1bf8，all-zero counts**（§231）。
- **MANIFEST 校验**：`osf_deposit_DOI1_20260518T211628Z/` 的 `MANIFEST_SHA256.txt` **29/29 条目字节级干净**（§235）。正确命令是 `awk -F'\t' '{print $2}' MANIFEST_SHA256.txt | sha256sum -c`（原 guide 的 `sha256sum -c MANIFEST_SHA256.txt` 不 work，MANIFEST 有行号前缀）。
- **witness SHA 双值消歧**（§235）：`011fa4c0…` = content-only SHA（第 1-43 行，不含自述 epilogue）；`6056b905…` = 全文件 SHA（44 行含 epilogue）。
- **B2 模型缓存**（§222）：HF SHA `093f9f388b31de276ce2de164bdc2081324b9767` 缺在 3 个 `pre_run/` 文件共 **9 处**；A100 侧 `du -sh blobs/` = **8.1G**，14 个 file snapshot，**0 个 `.incomplete`**。
- **VWA SHA**：`f0c835b` → `2f9b0b47175a1bffa01e13100e3075e212161a89`（扫 2 份文档，§220.3，被 §241 supersede）；SBOM diff-SHA 实测 **894e5afa ≠ 文档写的 20921a57**（§247）。
- **OSF manifest 缺口**：`grep -Ec 'B-440|B-448|...' osf_lock_manifest.md = 0`（pre-fix，§223）；paper.bib 计数漂移 —— manifest L23+L43 硬编码 **57**，实际 **67**（P2 defer 未修，§223）。

**演变**: §230 首版 witness 报 A100 三层零态（UTC 2026-05-18T13:57:22Z: **0 episode summaries / 0 condition summaries / 0 step JSONLs**；witness 文件 1390 bytes，SHA-256 e0e591f5…）→ **§231 RETRACT**（"grep pattern 漏了 `_v2` 后缀, 零计数是方法学假阴性 (当时 fire-2 在跑, task 0-3 已完成)"）→ §233 attempt #3 witness **tier 退化**（21:06:50Z 抓，约在 runner 启动后 2 min → tier 2 pre-outcome-inspection：`episode_summary_v2_count=0 + condition_summary_v2_count=0` **BUT `episode_steps_v2_count=1`**）→ 改 pre-launch witness 策略。

**已作废**:
- **§230 的首版 witness 全部作废**（文件已移入 `pre_run/retracted/…_VOID_RETRACTION_ONLY.txt`）。
- SBOM **diff-SHA 退役**（§247 结论："tree-chain 是唯一 enforced witness, diff-SHA 退役"）。
- §220.3 的 VWA SHA 修正被 §241 supersede。

**caveats**（一字不改）:
- §233: "witness 文件首行 `STATUS: post-outcome-creation` 会永久留在 OSF immutable bundle 里成为 reviewer 攻击面"。
- §231 (interim): "是 audit-trail bridge 不是 canonical DOI 1 witness"。
- §235: "witness 文件第 44 行自引用自身 SHA 在数学上不可能 (preimage 问题); 修法 = README frontmatter 用全文件 SHA … witness 文件本身保持字节不变以维持 witness-chain 完整性"。
- §222 (B2 SHA): "是 doc-trail 传播缺口不是数据缺口 (模型 2026-05-14 起就 cached 且当天复验)"。
- §221.7: git tag "是 §7 witness chain 的第 1/3 层 (advisor email + OSF DOI 仍 pending)"。

**证据**: §220.3 §221.3 §222 §223 §230 §231 §233 §235 §247；`docs/checkpoints/pre_run/`。

**原文片段**: 「⚠️ 该 witness 在 §231 被 RETRACT — grep pattern 漏了 _v2 后缀, 零计数是方法学假阴性 (当时 fire-2 在跑, task 0-3 已完成)」(§230)

---

## 25. 文档 / 工具层的数据质量计量

**当前值**:
- **master_bug_catalog B-number 逆序**：**142** 条 canonical entry（全文 7524 行，截至 2026-05-21）；"这 142 条 grandfathered 不 enforce; B-1822+ 才强制单调追加"（§257）。
- **gallery**：聚合页 **8255 个 `img_path` 全 null**（§260）；两个 bug —— B-1752 `_RUN_FAMILY_RE` 匹配不了新 run-dir 命名 → R19740 不可解析；B-1753 `line 1407` 的 `baseline_aliases` 集合缺 `B2_3mode` → **0 个 paper-grade B2 run 能聚合**（§234）。
- **status_query.py 首跑**：**83-vs-82 缺口** = `issue_a2_5_b_id_reservation.md` 缺 `type` 字段 → `issues.base`（filter type==issue）两端隐身；archive 后渲染器 0 隐身；45 tests pass（§289）。
- **cron 同步覆盖缺口**：`sync_a100_results.sh`（每 15min）只同步 `results/visualwebarena/phase1/`；顶层 `results/phase1_paper_grade/` + `results/B0_3mode/` + `results/B0_unified/` 是 **A100-only**（§234）。
- **analysis 管线 3-model 化的修改面**：**8 个 figure 脚本**（初始扫出 7 + 中途 sibling-pattern 检查发现 `fig3c_latency_per_step.py`）+ `compare_b0_b1.py`（4 处硬编码）+ Makefile + run_manifest.yaml + 12 个 B2 cell frontmatter（§225；"原文一度写 7，B-1682 在 §232 修正为 8"）。
- **cron lit-digest 核验**（2026-05-29，10 篇，§307）：**10/10 论文真实存在（无 hallucinate）**，但 **6 处 digest 失真**（threat→support 反向翻转 1 / 缩写错 1 / 外推 overclaim 1 / 术语作用域错 1 / venue 元数据 1 / 方法误标 1）；数字层反而准（Same-Task 18%/+4.6pp/32%、D2Snap 67%/65% 全核实属实）。
- **A100 pre-fire 状态**（2026-05-19 ~18:35 BST）：disk **485 GB / 92%（42 GB avail）**；git HEAD e34511f（需 pull 5 commits）；0 个 P79 进程；3 个 stale lock（§238.4）。
- **B-1683 manifest 门禁 smoke**：`queue_phase1_paper_grade.sh status` → **0/36 + 0/6 = 0/42 INCOMPLETE**（§232）。

**演变**: 索引算法 v1 曾因 inline cross-ref 污染 running-max **误报 1046 逆序**，修为 strong-def-only 才得 142（§257）。

**已作废**: "1046 逆序" 已作废（算法 bug）；"7 个 figure 脚本" 已被修正为 8（B-1682）。

**caveats**（一字不改）:
- §260 (gallery): "根因是底图 step_XXX/screenshot.png 未落盘 + 生产者写 `artifact_paths.som_image` 而消费者读 `artifact_paths.screenshot` 的契约错配"。
- §234 (cron): "pre-existing gap, 非本轮引入; follow-up B-### 待定 (脚本存在, 同步范围缺口是配置层)"。
- §289: "设计选择 = 渲染器解析 .base 而非硬编码视图 → CLI 与 Obsidian single source 不分叉; 动态视图副作用 = 第一次跑就找出手写表永不暴露的哑节点"。
- §307: "教训 = GLM lit-digest 能答有哪些 paper + 元数据, **不能答对我是友是敌**(threat/support 方向 + 缩写展开 + 术语作用域必须读原文)"。
- §232 (B-1683): "launch-pass2 目前只是 stub, 委托给 queue_phase1_router_paper_grade.sh 并受 LR 训练管线门禁"。

**证据**: §225 §232 §234 §238.4 §257 §260 §289 §307；`scripts/maintenance/index_bug_catalog.py`、`scripts/maintenance/status_query.py`、`scripts/maintenance/generate_gallery.py`。

**原文片段**: 「算法 v1 曾因 inline cross-ref 污染 running-max 误报 1046 逆序, 修为 strong-def-only 才得 142」(§257)

---

## ⚠️ 矛盾清单

> 规则：**两侧并列，不选边**。以下是同一个量在本批（或跨批）有不同值、且台账原文没说清哪个对的。

### C1. B-1794 的 SR 净效果用的是 partial，而同 run 全量结果相反

- **A 侧**（§253.2）：「R2987 (2026-05-20, pre-B-1794) **14.1%** → R9755 (post-B-1794) **20.6%**」，解释为 "search 类动作恢复"。
- **B 侧**（§258）：同一个 R9755 全量 224 ep 的 SR = **14.7%**；而 §253.2 自己的 20.6% 标注 scope 是 **63 ep**。
- 台账未在任何一节把 A 侧的 "+6.5pp 净效果" 用全量数重算，也未撤回该结论。**并列记录，不调和。**

### C2. B0 som cls 的 condition SR：27.2% vs 30.4%

- **A 侧**：§273 canary R11315 = **27.2%（224/224 ep）**，故意不 restart docker 保 6 天退化衬底；§306 taxonomy 的 best-single som = **27.2% (61/224)**（未点名 run_id）。
- **B 侧**：§283 R9725 = **30.4% (68/224)**，且原文称 "≥3 个 benchmark-FP → 真实 SR ≥ 31.7%, 30.4% 是下界"。
- 另有第三个 som run **R5313**（§299.1 §299.5 引用），台账未给其 condition SR。台账没说 §306 的 61/224 来自哪个 run。**并列。**

### C3. AMENDMENT_07：SoM-family 是否已重编号 1..K

- **A 侧（本批内）**：§295 实证 "sequential 重编号后 155/155 全同"（= 重编号能消 churn，AMENDMENT_07 的动机）；§297.1 caveat 明写 "dom 代码 AMENDMENT_07 未动 (保 native nodeId)"，反向暗示 **SoM-family 侧动了**。
- **B 侧（跨批，不在 D3 范围）**：编排者告知 §321/§322/§346 的 observation forensic **读到原始 nodeId**，与 "已重编号 1..K" 冲突。
- **本批同时记录**：§294 实证 "P79 SOM_MARKS = CDP getFullAXTree nodeId, 乱序不连续（`[5]Logout` 夹在 `[115][118]` 之间）"——这条是 **pre-AMENDMENT_07 的状态描述**，台账未标注它在 AMENDMENT_07 后是否仍成立。
- **D3 内没有 AMENDMENT_07 前后 SR（Δ−3.2pp）的任何记录。** 两侧都记，不替任一方消解。

### C4. R9755 的 failed episode 计数：156 vs 191

- **A 侧**（§259）：「diag_pattern_match.py Tier-1 在 R9755 上 **156/156 failed 全覆盖**, 0 token」。
- **B 侧**（§261 §153 caveat）：「后续 §261 修 B-1829 分母 bug 后 failed 计数应为 **191**（178≠191 曾是 `--failed-only` 分母 bug）」；§261/§162 全部按 **191 failed** 叙述。
- 台账给了分母 bug 的存在，但没有回头声明 156 这个数字作废还是只是不同分母下的正确值。**并列。**

### C5. P31 计数 vs BUDGET 类计数（vision）

- **A 侧**（§290）：P31（budget 耗尽未完成）**vision 73 / dom 58 / som 49**。
- **B 侧**（§306）：6-mode taxonomy 的 **BUDGET 56-69（psom 69 最高）**——区间不含 73。
- §306 自称 "本 taxonomy 独立于 P-rule ruleset"，即两者定义不同；但台账**没有**给出二者的换算或说明哪个是 vision 的 budget-failure 真值。**并列，禁止相减。** [聚合者推论] 二者很可能是不同定义下的两套计数，但台账未证。

### C6. deterministic coverage：93% vs 87.9%（是否算矛盾看你怎么读）

- **A 侧**（§261）：coverage **82%→93%**，同时自标 "93% 是 in-sample … post-hoc fit"。
- **B 侧**（§277）：fresh run R31194 上 **87.9%**，原文称 "5pp 掉幅 = 实证确认 93% 确有 in-sample 虚高"。
- 这一条台账**说清了**（B 侧覆盖 A 侧），故不算真矛盾；列在此仅提醒**引用 93% 必须带 in-sample 标签**。

---

## 未归主题的孤条

1. **B-1019 Macro archive 数字迁移**（§211.4）：「**22.7% / 10.8%** 从 §1 移到 §5」，scope = archive 数据、16 sub-cell coverage gap closure；caveat "archive 数字, 非 Phase 1a outcome"。台账未交代这两个数各自是什么量。
2. **A2.3a finding 降级表**（§209）：P0-1-AC* RETRACTED / P0-2-A* → P1（方向反转）/ P0-3-A* → P1（conclusion-invariant）/ P1-4-A* → P2（empirical I²=0%）；caveat "降级依据是 archive 实测, 非新数据"。
3. **A2.10 Phase 4 bash 实证**（§224）：B-1596 via `scored_task_count` 实跑；B-1598 via `pytest test_b660_pareto_preserves_true_ties` FAIL(pre-fix)→PASS(post-fix)；B-1641 via grep 跨脚本命名不符（gate 用 `{cell_id}_vectorizer_fold{k}.pkl` ≠ loader+trainer 用 `vectorizer_fold{k}.pkl`）；B-1640+1645 via py_compile + runner smoke 7/7 PASS。caveat "零 AI 推断的 numeric claim, 全部落到实际磁盘状态"。
4. **A100 VWA 凭据 env 缺口**（B-1574，§226）：`wc -c scripts/vwa_env_remote.sh = 514`；`grep -c 'VWA_.*_USER' … = 0`（`.sh` 与 `.sh.example` 修前均 0）；`auth_refresh.py:46-51` 的 `_ACCOUNT_ENV_KEYS` 需要 4 个 `VWA_<SITE>_USER/PASS`。
5. **reddit auth false-negative**（B-1577，§227）：`outcome=cred_wrong … LOGIN_FAILED (no_logout_marker) -> http://localhost:9999/`，最终 URL 是 postmill 首页而非 `/login` → **登录其实成功**；根因 = postmill 把 logout 链接渲染在 JS 驱动的用户下拉里，初始 DOM 里没有。
6. **NLTK 3.9+ punkt_tab 路径解析 bug**（B-1751，§233）：`nltk.data.find('tokenizers/punkt_tab')` 去查 `tokenizers/punkt/PY3_tab`；即使 download 成功 + 目录在磁盘 + `sent_tokenize/word_tokenize` 端到端 PASS 也报失败。修法 = preflight 改用 filesystem `os.path.isdir()` 跨 5 个标准 NLTK 路径检查。
7. **Fire-3 20-task 的 schema 健康度**（§234）：所有必需 gating 字段齐（含 `schema_version 2.0` / seed 42 / `evaluator_authority_mode post_B545_vwa_score_only`）；0 needs_reevaluation；0 auth_refresh 与 0 auth fail；**73 次 Action cycle detected + 73 次 diagnostic only**；0 EvaluatorUnavailableError；energy 字段 None + `energy_partial=True`。caveat "observation_mode 不显式存, 由 condition_id + run_id 隐含 … 非 paper-grade blocker"。
8. **S4 run task 8 瞬时故障**（§244）：transient `agent_navigation Page.goto timeout`（非 eval，与 C1 无关）；事后 cls curl 200 / 0.23s；"暴露 GRL reliability gap = agent-nav 无 retry"。
9. **B0 vision 的 `about:blank` / cross-site task**（§299.2）：T36 —— agent step_5 后全程 about_blank，shopping localhost:7770 全程不可达 = "同 host 只跑一条 site chain 的 hard rule 后果"。
10. **Gate 1.5 两个 P0**（§271）：codex P0-1 —— "我的 fix 让 timeout `is_nav_error=True` → agent-page program_html 的 timeout 落入 B-329 `return score=0.0` (silent) 而非 fail-closed raise → **16 tasks (2 cls + 14 red) silent deflate SR 且 B0-biased**"；gemini P0-2 (→B-1837) —— "eval 5-retry vs agent-step 0-retry asymmetry + B0 慢 → differential baseline rescue … **inflate B0 paired superiority**"（走 measure-then-decide，不改 code）。
