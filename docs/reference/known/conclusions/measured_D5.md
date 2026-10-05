---
type: conclusions
batch: D5
status: done
created: 2026-10-06
source: scratchpad/batches/D5.jsonl (172 条 MEASURED，§398.1–§474.9，2026-07-28→08-21)
---

# 测量结论 D5 (§398.1–§474.9)

> **读法**：每个主题的「当前值」是截至 §474.9 现在算数的数字；「已作废」的数字**禁引**。
> 本批时期（2026-07-28 → 08-21）发生了四件事：① Phase 0b 把**同条件重跑地板**从一对 dom pair 补到 B0·cls 六臂全齐、
> 并发现 **B1 本地格基本确定性**（0.00% / 0.00% / 3.12%）—— 本批最承重的一条线是「表征臂 vs 重跑臂不可区分」；
> ② router 负结果被加严（同族 pooled × cost-tier 0/26），triage producer 的死特征与硬编码散文被修，旧 AUROC 一批作废，
> 「abstention」作为新的二元标签被量化（可部署省钱 14.4–24.6%）；③ 新增 WA-reddit 第七/八格，形成 VWA-vs-WA「任务模态轴」；
> ④ 后半段（§442 起）大量是运维与稿件工程：B-1969 cls 死锁污染、AWS proxy 预算耗尽与候选模型探测、B5 (GPT-5.6) 接入、A100 git 溯源、毕设 / REALM / VLM4RWD 稿件。
> **noise 类数字（discordance / self_drop / floor / band / 门槛）一律各自带 scope 并列，禁止加减**（沿用 §397.10 禁令）。
> ⚠️ §407 起的条目里 「§4 / §5 / §6 / §8 / §3.3」 常指**稿件章节**而非笔记 §，本文照抄不改，凡能判断的标「(稿件章节)」。
> **跨批**：另一位聚合者在并行做 D6（§475.2 → §527.2）；本文件只写到 §474.9 为止的状态，凡在 §466 之后仍活跃的主题末尾注「§475 之后见 measured_D6.md」。

---

# A. 噪声地板与 replicate（本批主线）

## A1. B0·VWA-classifieds 同条件重跑地板 —— 从一对到六臂全齐

**当前值（B0 = qwen3-vl-235b via AWS proxy / classifieds / 六个 canonical mode / n=224 公共计分集 / 12 个 run，§470.3）**:
flip: DOM **27=12.1%** / SoM **29=12.9%** / **Vision 32=14.3%** / P-text **23=10.3%** / P-SoM **27=12.1%** / P-prompt **28=12.5%**；
净 ΔSR: **−2.23 / +2.23 / −0.89 / −1.34 / −1.34 / −2.68 pp**（同序）。
§470.7 的全景表述：CLEAN_PAIRS 中 **B0 六臂 10.27-14.29%**。

**并列口径（各自带 scope，不相减）**:
- §398.2 dom pair（B0 × classifieds × dom, n=224，首次跑到 canonical N）: **self_drop 4.9 / 7.1pp；discordance 12.1pp；0 reset 污染**；与 vision pair **6.7/7.6pp** 一起证明地板不是 vision-specific。
- §450.14（三个独立复现 arm，discordance 出自 noise_floor_inventory.md §1，κ 出自 phase0b_noise_floor.md §2）: **Vision 14.29% (κ 0.614) · SoM 12.95% (κ 未算) · DOM 12.05% (κ 0.559)**。

**演变**:
- §398.2: README:52 此前只有 partial@88 → dom pair 首次到 n=224。
- §450.14: 三臂（dom / vision / som），κ 只两对。
- §470.3: 六臂第一次全齐（含三个 phantom 臂）。
- §470.5 / §470.7: 与 B1 对照（见 A2）→「那 12% **不是 benchmark 噪声/agent 随机性/VWA 不稳定, 是 B0 serving 栈特有的**」。

**已作废**: 无数字被推翻；但 §398.2 caveat 里「全部 B0/MoE, 不可外推到本地确定性 backbone」的**缺口**已被 §406 / §470.5 / §470.7 / §472.3 部分补上（见 A2）。

**caveats**:
- §470.3（一字不改）: 「DOM 与 Vision 的 replicate 与其 canonical run 相邻(2.4/1.5 天, 无 code drift), 其余四臂的 B run 跨 ~24 个 runtime commit(含 B0 proxy 响应形状 B-1970/1979/1980 + retry budget B-1880/1881) ⇒ 那四个数是**含 drift 的上界**。但 sanity check 反向: **最干净的 Vision flip 率最高(14.3%)**, 跨 commit 四臂均 ≤12.9% ⇒ run-to-run 噪声主导、drift 不主导」；「这 6 对里只有 4 对进了 `CLEAN_PAIRS`, 三个 phantom 臂尚未注册」。
- §450.14: 「**κ 只有 vision/dom 两对算过** … **缺的就空着, 不跨 arm 借用**」；「该量是 **instability proxy, NOT H1 drop-one bias correction** (工具自带 caveat)」；WA 的复现不可并排读（pooled 5 modes × 10 shared tasks）。
- §470.5: 「B1 那格买地板是『免费但没用』… **跨 model 借地板是错的**」。

**证据**: §398.2 / §450.14 / §470.3；`docs/analysis/cross_sites/phase0b_noise_floor.md`、`docs/analysis/cross_sites/noise_floor_inventory.md`、`scripts/analysis/unique_solve_noise_envelope.py`

**原文片段**: 「self_drop 4.9 / 7.1pp; discordance 12.1pp; 0 reset 污染」(§398.2)；「flip (n=224): DOM 27=12.1% / SoM 29=12.9% / **Vision 32=14.3%** / P-text 23=10.3% / P-SoM 27=12.1% / P-prompt 28=12.5%」(§470.3)

> §475 之后见 measured_D6.md

---

## A2. B1 (Qwen3-VL-4B, local) 的重跑地板 —— 基本确定性，但不是一刀切

**当前值（各自 scope，并列）**:
- **B1 × classifieds × som**（R31705(0604) ↔ R28065(0816), n=224）: **0.00%**，success 224/224 全同，步数 220/224 全同，4 个 task 异轨同归（§470.5）。
- **B1 × classifieds × vision**（n=224, temp=0）: **0.00% discordance 且步数差异 0**，SR **12.50%→12.50%**，逐 episode 字节级重现（§470.7）。
- **B1 × classifieds × dom**（arm A = R17188 2026-06-03, arm B = R14980 2026-08-19, n=224）: **3.12% (7/224)**，SR **6.25% → 6.70%**，self_drop **1.34 / 1.79 pp**（§472.3）。
- **B1 × WA-reddit**（注册 10-task pilot draw × 5 个 clean mode, n=50 配对）: **self_drop 2.00 / 4.00pp；discordance 6.00%**；翻转 **3/50** 全部集中在 vision (2) 与 P-prompt (1)，dom/som/P-text 共 30 对零翻转（§406）。

**演变**:
- §398.2: 「B1/B2 本地格 replicate 0 个」。
- §406: 发现 WA pilot × full-104 的 task 重叠就是同 condition 重跑 → 首个 B1 地板（WA，含环境漂移）。
- §470.5 → §470.7: B1·cls som / vision 均 0.00% → 「CLEAN_PAIRS 8 对的全景是一刀切二分: **B0 六臂 10.27-14.29% vs B1 两臂 0.00%**」。
- §472.3: B1·cls·dom **3.12%** → 打破「一刀切」。

**已作废**:
- §398.2「B1/B2 本地格 replicate 0 个」→ 被 **§406 部分推翻**（台账 superseded_by=§406）。
- §470.7「B1 两臂 0.00% ⇒ 一刀切二分」作为 B1 的全称表述 → 被 **§472.3** 的 dom 3.12% 打破（两条 0.00% 本身仍有效）。
- §470.9 AMENDMENT 的 Reason 1 → 被 §472.3 推翻；「Reason 2/3 不依赖它 ⇒ 取消格 6-8 的决定本身未被推翻 (user 2026-08-20 维持取消)」。

**caveats**:
- §406（一字不改）: 「含环境漂移不纯是随机性 (两 run 相隔数天, reddit require_reset 是 no-op 故订阅跨 episode 累积 §402) ⇒ 只能命名为 run-to-run including environment drift, 不可称 decoding stochasticity。⚠️ n=50, 3/50 的精确二项 CI 约 [1.3%, 16.5%]。⚠️ 是 WA 不是 VWA」；P-SoM 的 20260727 目录是 full run 重启残段，**26 对 3/0 单向翻转 (11.54%) 单列不并入**。
- §470.7: 格 4 是「first B1 floor with any power」(d≈16.6) —— 「**不是 power 不足导致的 0**」；「该 0 **不能用来约束 B0 上的任何效应量**」；仍是单 site (cls)。
- §472.3: 「**不是同代码复现**: 两臂相隔 2.5 个月代码漂移。som/vision 跨同一段漂移却给 0.00%, 所以漂移解释不了 dom 的 7 个翻转 —— **但也没被排除**」；「d≈8.3 < 10 ⇒ 按本项目自己的门槛是**描述性 inventory, 不报 CI**」。

**证据**: §406 / §470.5 / §470.7 / §472.3；`docs/analysis/cross_sites/noise_floor_inventory.md`、`docs/analysis/cross_sites/noise_floor_inventory.json`、`scripts/analysis/aggregate_noise_floor_inventory.py`

**原文片段**: 「**B1.cls.som = 0.00%** (n=224 …) vs **B0 六臂 10.3-14.3%**」(§470.5)；「**3.12% (7/224)**, SR 6.25% → 6.70%, self_drop 1.34 / 1.79 pp」(§472.3)

> §475 之后见 measured_D6.md

---

## A3. replicate 清单、可测性与注册表

**当前值**:
- manifest（`run_manifest.yaml` 65 run 条目 / 40 个 distinct (model, site, mode)）: **19 组 ≥2-run (非 15)**；多数第二个 run 目录不在磁盘（§398.2）。
- `noise_floor_inventory §1`（2026-08-16 快照）: 只有 **4 个 pair** —— `B0.cls.{dom,vision,som}` (各 n=224) + `B1.wa-red` (n=**50**) ⇒ 「完整地板只在**一个格**…**没有任何 phantom 臂有干净的同条件地板**, 而 drop-one hero 恰恰跑在 phantom 臂上」（§467.1）。此后 §470.3 补齐 B0·cls 三个 phantom 臂（6 对中 4 对进 CLEAN_PAIRS），§470.7 记 CLEAN_PAIRS **8 对**。
- 可测性经验关系（§468.5）: `d ≈ n × SR × 0.59`（三个已测 B0 pair 反推: **0.46/0.58/0.74**）；代入 **B0 cls phantom 臂 d≈20.7-26.1** · B1 cls phantom 臂 **d≈8.3-10.1** · B1 red 全部六臂 **d≈3.0-8.9** · B0 red phantom 臂 d≈13.0-15.9；实测单格成本 B0 cls dom **$15.59** / psom **$16.14**；B0 red dom **$20.76** / psom **$22.10**（scope: VWA cls n=224 / red n=203；SR 取自 sr_per_mode canonical）。
- 注册表跨机解析（§469.5）: DGX 3.12.3 / A100 **3.10.12**；`aggregate_noise_floor_inventory.py:607-608` PEP 701 f-string 在 3.10 是 SyntaxError，`validate_fire_manifest.registered_replicate_run_ids()` **fail-closed 返回空注册表** ⇒ 所有 registered replicate 一起被判 ghost；修复后 A100 侧 CLEAN_PAIRS 解出 4 pairs，validator EXIT=0。

**已作废**: §397.10(3)「manifest 里 15 组」→ **§398.2 改为 19 组**（见文末 supersede）。

**caveats**:
- §398.2: 「`B1_3mode_classifieds_20260413` 是陷阱: 目录在、manifest 写 expected_n=234、三个 subdir 各只剩 1 个 episode ⇒ `test -d` 过而数据不在」。
- §467.1: 唯一的 P-SoM pair 是 restart partial（26 shared task, 3/0 单向 flip = 11.54% discordance），「**不要当作 phantom 臂有地板**」。
- §468.5: 「那个 0.59 的比值在三个锚点上跨度 **0.46-0.74 (1.6×)**, 所以 d≈8-9 那几格实际可能落在 5-13 —— 方向可信, 精确值不可引」；成本是 `per_task_sr.csv` 的 leak-kept 口径；「**reddit 比 classifieds 贵**」。
- §469.5: 「`ast.parse(..., feature_version=(3,10))` **检测不到**此类问题(实测照样接受)」⇒ 守护测试只能是源码层正则 heuristic。

**证据**: §398.2 / §467.1 / §468.5 / §469.5；`docs/analysis/cross_sites/phase0b_noise_floor.md`、`docs/analysis/cross_sites/noise_floor_inventory.md`、`scripts/queues/_b1_floor_watcher.sh`、`tests/test_noise_floor_registry_parses_on_fire_host.py`

**原文片段**: 「19 组 ≥2-run (非 15); 多数第二个 run 目录不在磁盘; B1/B2 本地格 replicate 0 个」(§398.2)；「完整地板只在**一个格**; 且三个 pair 全是 dom/som/vision」(§467.1)

> §475 之后见 measured_D6.md

---

## A4. 表征臂 vs 重跑臂 —— 同臂数下不可区分

**当前值**:
- **one-arm margin（§450.10，仅两个有实测地板的 cell）**: 🔴 **两者无法区分**。`B0·VWA-cls` (n=224): +1 最佳异表征 **7.14pp** (dom) vs +1 重跑 **4.91–7.59pp** ⇒ inside the rerun band；`B1·WA-red` (n=104): +1 异表征 **4.81pp** (ptext) vs 重跑 **0.00–10.00pp** ⇒ indistinguishable。原文: *『Neither cell shows a representation arm worth appreciably more than a rerun arm; one shows it worth no more at all.』*
- **六次重跑并集 vs 六-mode oracle（§452.1，仅 cls·B0, n=224）**: U(6) = DOM **29.09%** / Vision **38.84%** / **SoM 39.77%**；六-mode oracle **43.30%** ⇒ 残差仅 **3.53pp**，低于本文门槛（正态 **3.82–4.15pp** / 精确 **4.02–4.46pp**）⇒ 「同臂数下六个表征与六次重跑**不可区分**」。
- **固定六臂预算两种花法（§455.3，B0 × VWA-cls, n=224）**: 6 表征×1 代 union **43.30%** / 3 表征×2 代 union **44.64%**；独占 6-rep 3 题 [68,117,142] / 3×2 6 题 [22,50,78,147,161,206] ⇒ 「只能报『两种花法不可区分』」。
- **逐 base 读（§455.2，B0 × VWA-cls, n=224，六个 base = 三臂×两代）**: 重跑 **4.91–7.59pp 随 base 无趋势**；换表征 **3.12–17.41pp 强依赖 base**；switch/retry 比值 dom.b(15.18%) 2.12–2.44× · dom.a(17.41%) 3.09–3.45× · vision.b(24.11%) 0.82–1.47× · vision.a(25.00%) 1.07–1.60× · som.a(27.23%) 0.65–0.94× · som.b(29.46%) 0.58–1.17×。
- **在最强截图模式之上加 1 条臂（§407.3，7 格，臂数对齐）**: cls_B0/B1/B2: +最好文本臂 **7.14/3.57/1.79** 对 +第二截图臂 **6.70/4.91/2.23** · red_B0/B1/B2: **4.93/1.97/3.94** 对 **3.45/0.99/0.99** · wa_red_B1: **8.65** 对 **2.88**。B1 × WA-reddit 逐臂: +ptext/+pprompt **4.81pp** 出地板，+som/+vision **3.85pp** 与 +psom **1.92pp** 在 **2.00-4.00pp** 地板内。

**演变**:
- §406: B0·cls +7.14 vs +4.91~+7.59 ⇒ 带内；B1·WA-red +4.81 vs +2.00~+4.00 ⇒ 「高出带子 0.81pp」。
- §407.3: 加第二截图视角 vs 加文本通道在 VWA cls 与 WA 上方向相反 → 模态轴（见 D2）。
- §450.10: WA 行的重跑带改读为 0.00–10.00pp ⇒ WA 格也 indistinguishable。
- §452.1 / §455.2 / §455.3: 扩成并集外推、逐 base、预算花法。

**已作废**:
- §406「B1·WA-red 高出带子 0.81pp」的读法 → 被 **§450.10** 改为 indistinguishable（WA rerun 数字是 5 mode 池化到 10 个 pilot task，「不是 dom 的地板」）。

**caveats**:
- §406: 「不许据此推出『整个 6-mode 天花板增益都是噪声』—— 手上只有 1 个重跑臂不是 5 个…5 臂增益 (16.07pp) 必须带臂数标注单独报, 绝不与 1-重跑地板对撞」；「不可跨格做算术」。
- §450.10: 「**7.14pp 已是 best 异表征**, phantom arm 贡献更小 ⇒ 攻击只会更强」；「**口径不可跨**: 这是 1-arm baseline +1 arm; C2 的 drop-one 是 6-arm oracle −1 arm」；WA 行 pooled_n=50 但只 10 个独立 task，且 dom 自己 pilot-vs-full 是 **0 flips**。
- §452.1: 「**是上界不是估计**」；「U(1) 是构造出来的不算证据; **信用全在 U(2) 的样本外检验** (误差 +1.12/+0.44/−1.11pp, som 那格是**低估**重复)」；「两者 serving cost 不同: oracle 每任务 1 个 episode, 重跑 6 个 ⇒ 存活的主张是**天花板的价格**不是高度」；一格结论，不外推。
- §455.2: 「union gain 与 base SR 天然负相关, 故 switch 随 base 降**部分自动**; 真发现是**只有 switch 缩而 retry 不缩**」；三臂一 cell。
- §455.3: 「**1.34pp = 3 题, 而本 cell 同 condition discordance 12–14%**」；两预算可能共享至多 3 臂 ⇒ 非独立对比；3×2 的臂相隔 2–69 天收了环境漂移 ⇒ 对它偏有利。
- §407.3: 「不是『主张在 VWA 不成立』而是一条对称规律的两个观测点」。

**证据**: §406 / §407.3 / §450.10 / §452.1 / §455.2 / §455.3；`docs/analysis/cross_sites/noise_floor_inventory.md`、`docs/analysis/cross_sites/rerun_union_extrapolation.md`、`docs/analysis/cross_sites/retry_vs_switch_label_supply.md`、`results/webarena/phase1/`

**原文片段**: 「给最好单模式 (som 27.23%) 再加 1 个最佳不同表征 (dom) = +7.14pp; 加 1 个同模式重跑 = +4.91 ~ +7.59pp ⇒ 落在重跑带子内部, 不可区分」(§406)；「六-mode oracle **43.30%** ⇒ 残差仅 **3.53pp**」(§452.1)

---

## A5. 「过带 ≠ 过噪声」：泛函、门槛与 drop-one 三侧分解

**当前值**:
- **门槛（§450.10，B0·VWA-cls 三个已重跑 arm: dom d=27 / vision d=32 / som d=29, n=224）**: measured floor 上边缘 **2.23pp** 与其自身标准差 **2.32–2.53pp** 同量级 ⇒ ***『clears the band is not clears the noise』***；效应需达 **3.82–4.15pp** (one-sided 95%) 才不太可能由单次重跑单独产生；交换性零假设 `D=(2X−d)/n`, `X~Binom(d,½)`, `SD(D)=√d/n`。§452.1 另给精确版 **4.02–4.46pp**。
- 🔴 「**drop-one oracle 的 1.7–3.3pp 落在 3.82–4.15pp 门槛以下** ⇒ C2 不能作正面结构主张」（§450.10）。
- **两个泛函的噪声量级不同（§407.8，cls_B0）**: 集合差 (|{A解}∖{B解}|/n) **4.91-7.59pp**；均值差 (SR(A)−SR(B)) **0.89-2.23pp**。模式间 SR 差 SoM 27.23 vs DOM 17.41 = **9.82pp** 远在噪声外；但 SoM 27.23 vs Vision 25.00 = **2.23pp** 正好等于均值位移 ⇒ 前二名排名不安全，进 limitations。
- **drop-one 独有贡献三侧分解（§450.18，`results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv`，VWA 6 cells, complete-case）**: 文字侧 DOM **0.45–2.96** (6/6>0) · P-text **0.00–0.99** (5/6) · P-prompt **0.00–2.68** (4/6) · P-SoM **0.00–1.34** (5/6)；结合侧 SoM **0.49–4.46** (6/6)；视觉侧 Vision **0.99–4.02** (6/6)。**36 个值里只有 3 个**达到 3.82pp 门槛，**全部是 SoM/Vision, 无一 phantom**。

**已作废**: 无。（注：§407.8 的「② 被打掉是因为它恰好用集合差」是解释，不是数字修正。）

**caveats**:
- §450.10: 「该零假设**只假设两次运行可交换**, **不建模环境漂移** … 有漂移时真实 spread 更大, 这些阈值本身是**下界**」；「『观测到的 |ΔSR| 0.89–2.23pp』是**每格一次抽样**, 不是界」。
- §450.18: 「源行自带 **grade=NON_PAPER_GRADE**」；「门槛 3.82–4.15pp 只从 **B0·VWA-cls 一格**的三个复现 arm 导出」；「该量是 oracle 组合下的边际贡献, **不是**单模 SR 差」。
- §407.8: 「同一批噪声, 两方向翻转在集合差上相加、在均值差上相消 (B0.cls.dom 12.05% 不一致率 → 净 SR 位移只有 2.23pp)」。
- 相关: drop-one 1.7–3.3pp 的来源 `meta_phantom_lift.csv` mtime 05-17，「定稿前必核」（见 L1 / §450.6）。

**证据**: §407.8 / §450.10 / §450.18 / §452.1；`noise_floor_inventory.md`、`results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv`、FIGURE_PLAN F9/F10b

**原文片段**: 「measured floor 上边缘 **2.23pp** 与其**自身标准差 2.32–2.53pp 同量级** ⇒ ***『clears the band is not clears the noise』***」(§450.10)；「**36 个值里只有 3 个**达到 3.82pp 过噪声门槛, **全部是 SoM/Vision, 无一 phantom**」(§450.18)

---

## A6. unique-solve 的 replicate 噪声包络

**当前值（B0 / classifieds / 六 mode / n=224；2^6 = 64 种赋值，每臂独立取 run A 或 B，§470.3）**:
观测(all-A) → [min, max]: SoM **6→[6,12]** · Vision **9→[6,11]** · DOM **4→[0,6]** · P-prompt **6→[1,6]** · P-text **2→[0,3]** · P-SoM **2→[0,4]**。保守版（只扰动 DOM+Vision 两个无 drift 臂）: P-SoM 仍能 **2→0**。

**caveats**: 「判决是**跨 side 稳健 (SoM/Vision 下界 6) 而 side 内不稳 (三个 phantom 臂下界 0-1)**, 不是『phantom 臂无区别』—— 区别在 task identity 上(§470.2), 只是计数量级落在噪声内」；单 cell；「all-A 结果与台账 §306 exclusive-solve 逐格一致(交叉验证通过), 且 §306 caveat(1) 正是要求做此事」。

**证据**: §470.3；`scripts/analysis/unique_solve_noise_envelope.py`

**原文片段**: 「SoM 6→[6,12] · Vision 9→[6,11] · DOM 4→[0,6] · P-prompt 6→[1,6] · P-text 2→[0,3] · P-SoM 2→[0,4]」(§470.3)

> §475 之后见 measured_D6.md

---

## A7. 标签不稳定性在决策边界上的富集 + retry/switch 的标签供给

**当前值（三臂版，B0×cls n=224，dom/vision/som 各复制一次，§464.2）**:
**67/224 (29.9%)** 的 task 在至少一个复制臂上翻转。分层: which-mode label rows 97 个 flip **62.9%** · **arms DISAGREE 88 个 (39.3% of cell), flip 67.0%, 承载 88.1% 的全部 flip** · 恰好一个 mode 解出 29 个 flip **79.3%** · **补集 136 个 flip 5.9%** ⇒ enrichment **11.4×**。

**retry 还是 switch 的标签供给（§455.1，B0 × VWA-cls / dom+som+vision / n=224）**: decision set (base 失败) **158–190 = 全 cell 70–85%**，但 neither（两 action 齐失败）= **124 恒定** (55% of n=224)；真正带偏好的 contested 仅 **11.6–25.0%**，对 which-mode 同臂数 contested **24.11% = 1.04×**；actionable 上限 **29.46%** ⇒ **否掉**了「label ∝ (1−SR) ⇒ 低 SR 时充足」的假说。

**演变**:
- §407.4（两臂 DOM+Vision，实测翻转 49/224）: 六分类标签行 97 行中 47 翻 = **48.5%** · 臂间分歧 88 中 45 = **51.1%** · 三分决策不一致 74 中 36 = **48.6%** · 唯一解行 29 中 15 = **51.7%**。
- §407.22: 补集 136 题 4 翻 = **2.9%** ⇒ 富集 **17.4x**；全部翻转的 91.8% 落在只占 39.3% 题目的集合里；四种行集富集 **16.5-17.6x**。
- §409.1: 难度代理循环 —— 全六模式代理 51.14% (n=88) / 2.94% (n=136) ⇒ **17.39x**；留出被重跑的 dom+vision: 46.88% (n=64) / 11.88% (n=160) ⇒ **3.95x**；「诚实区间 = **3.9-17.4x**」。
- §464.2: 三臂重算 → **11.4×**。

**已作废**: 两臂版的 **49/224** 被 §462.1 明标 **stale**（「只有 DOM+Vision 重复时算的」）；当前以 §464.2 三臂版为准。两臂版的 17.4x / 3.9-17.4x 只作历史，不再作为当前富集值引用。

**caveats**:
- §407.4: 「这是下界 —— 六条臂只测了两条每条只重跑一次」；「给出任何 per-task router 的性能上界, 与样本量无关」。
- §464.2: 「仍是**下界**」；「『contested』按定义是中等难度带, 而 flip 概率 2p(1−p) 在 p=0.5 处最大 ⇒ 富集有一部分是算术; 产物自带 binomial-floor 自查 (两臂时 observed 51.14% vs floor 37.25%), 三臂版的该对照需一并读」；「用 `per_task_sr.csv` 定义 contested ⇒ **leak-kept 口径**」（但 §465.1 证实 classifieds 零影响，见 E4）。
- §409.1: 「两个数都不能单独引」。
- §455.1: 全量 oracle-conditioned；「重跑对相隔 2–69 天含环境漂移 ⇒ 立即 retry 的收益应更小」；三臂一 cell，phantom 无重跑故不涉及。

**证据**: §407.4 / §407.22 / §409.1 / §455.1 / §464.2；`docs/analysis/cross_sites/label_instability.md`、`docs/analysis/cross_sites/retry_vs_switch_label_supply.md`

**原文片段**: 「**arms DISAGREE (choice matters) 88 个 (39.3% of cell), flip 67.0%, 承载 88.1% 的全部 flip** … ⇒ enrichment **11.4×**」(§464.2)；「decision set 大 ≠ label 多, 失败集 67% 是两 action 齐失败无偏好可学」(§455.1)

---

# B. Phase 0b 的 null / id-regime

## B1. fixed-marginal permutation null（首次执行）与观测 Jaccard

**当前值**:
- permutation null（6 cell × 4 arm, B=10000, seed 42，prereg 锁定 B）: **0 / 24 arm×cell 为正 excess**；唯一非显著是 **B2·cls SoM (excess 0.000, p=0.7065)**（§398.2）。
- 观测 Jaccard vs 同一 fixed-marginal 独立 shuffle（6 cell, B=2000, seed 42）: **6/6 cell 观测值在 null p95 之上**；中位数比值 **3.75× (B0·cls) – 23.04× (B2·red)**（§398.2）。

**caveats**: 「该 null 此前从未被执行过 (无 *_permutation_null.json / CSV 无 perm_null_* 列 / paper 无 §4.Y); claim 只活在旧 omnibus draft 与 paper_drafts_locked/, paperA/ grep 全空 ⇒ 非待投论文的活缺陷」；Jaccard「不可与 §1:21 的 archive 4-mode Jaccard 0.29–0.49 混引 (不同 universe)」。

**证据**: §398.2；`docs/analysis/cross_sites/phase0b_noise_floor.md`

**原文片段**: 「0 / 24 arm×cell 为正 excess; 唯一非显著是 B2·cls SoM (excess 0.000, p=0.7065)」(§398.2)

---

## B2. id-regime 实证 —— AMENDMENT_07 生效

**当前值（B1, min/median/max 的 element_id 量级，§398.2）**: B1·p-som **1/12/68** · p-text **1/13/72** (mark_count 33/35) vs p-prompt **139/4074/26235** · dom **2/3606/61833**。

**caveats**: 「证实 §397.10(1) 的 id-namespace 归属: text payload 决定 regime, [SOM_MARKS]→1..K, AXTree→native nodeId」。

**证据**: §398.2；`docs/analysis/cross_sites/phase0b_noise_floor.md`

**原文片段**: 「B1·p-som 1/12/68 · p-text 1/13/72 (mark_count 33/35) vs p-prompt 139/4074/26235 · dom 2/3606/61833」(§398.2)

---

# C. Router / triage / abstention

## C1. 同族 pooled × cost-tier（H-pool）—— 最有利角落也不翻；per-task cost headroom

**当前值**:
- **H-pool 检验（2 site × {same_family(B0+B1), all_three, per_cell} × {cost_tier, which_mode}, task-held-out 5-fold 真嵌套, paired bootstrap B=1000，§399.1）**: 严格支配 always-cheapest **0/26** arm×cell（最有利角落 same-family×cost-tier **0/4**）；相对六固定 mode 菜单非支配 **0/26**；锁定判据（95% 非支配 vs always-cheapest）**5/26**。
- **唯一通过锁定判据的 cell（§399.2，reddit, B0, 7 arms）**: reddit·B0 在 7 臂里过 5 臂；真实原因是**对照弱** —— 该格 always-cheapest=Vision 仅 **7.39%** SR vs best-single 参考 **11.33%**；router 到 **13.3-15.3%**（超过 best single）但每 task 贵 **2.7-10.2%**，六模式菜单在 **35-71%** 配对重采样里仍支配它。
- **per-task cost headroom（6 cells, 全 scored universe, total_billed_cost_usd, cell 内可比，§401.1）**: always-cheapest(Vision) 在 **47.3-70.9%** 的 task 上不是该 task 最便宜的；per-task cost oracle 比它便宜 **22.2-46.2%**（B0cls −46.2 / B0red −42.1 / B1cls −41.0 / B1red −28.8 / B2cls −32.8 / B2red −22.2）。
- **cost-tier 标签性质（6 cells，§399.5/§399.6）**: `MODE_COST_TIER[derive_oracle_label(x)] ≡ 『是否存在 text-only mode 成功』`，6 cell 全部 **0 mismatch**（MODES tier 序列 [0,0,0,0,1,1] 单调）；reddit 同族池 tier 分布 text_only **63** / image **14**，B1·reddit 单格 **20/4**。

**已作废**:
- §399.2 原句「同族/粗粒度/池化三者都不是原因」的**因果表述** —— 据 D4 B11（§401 cross-AI 复审，非本批 MEASURED 条目）已收回，现只报共现与点估计。本批 §399.2 条目里的数字本身不受影响。
- 「always-cheapest 是成本下界所以支配不可能」的反驳 → 被 **§401.1 证伪**（但 §450.12 又重新使用了该表述，见矛盾清单 #2）。

**caveats**:
- §399.1: 「H-pool 不成立 ⇒ 走规格预案『不支配则结论加限定后更强』。§392.2 的 0/6 在最有利配置下依旧成立。post_hoc_exploratory, 非 H10 gate」。
- §399.2: 「per_cell 臂是规格外补的第 5 臂, 没有它这些 pass 会被误读成池化的功劳 (同型风险 §316)」。
- §399.5/§399.6: 「同族 reddit 的『5.0% 冲突 / 97.5% 天花板』有一部分来自标签几乎不变, 不是两 backbone 在难题上一致 (同型陷阱 §394 天花板≠可学性)」。
- §401.1: 「Vision 只是 per-mode mean 最低不是 per-task 恒最低 ⇒ Pareto 支配不被定义排除, 观察到的失败是实证的。该表本身即 cost-routing headroom 上界, 可独立进论文」。

**证据**: §399.1 / §399.2 / §399.5 / §399.6 / §401.1；`docs/analysis/cross_sites/router_pooled_tier_learnability.md`

**原文片段**: 「严格支配 always-cheapest 0/26 arm×cell (最有利角落 same-family×cost-tier 0/4); 相对六固定 mode 菜单非支配 0/26」(§399.1)；「always-cheapest(Vision) 在 47.3-70.9% 的 task 上不是该 task 最便宜的」(§401.1)

---

## C2. triage learnability —— 死特征、硬编码散文、修正后的数字

**当前值（triage 二元标签 = 六臂任一可解；双口径并列）**:
- **逐格 AUROC（§450.2）**: 20 特征 VWA-only → 18 特征跨 benchmark 匹配集: cls·B0 **0.726→0.683** · red·B0 **0.780→0.700** · cls·B1 **0.732→0.705** · red·B1 **0.864→0.723 (−0.141)** · cls·B2 **0.642→0.646 (+0.004)** · red·B2 **0.615→0.526** ⇒ **6 格里 5 格下降**；20 特征版里 **5/6 格的最强单特征是 `reasoning_difficulty`**（VWA 人工难度标注，部署拿不到）。
- **修正后的散文数字（§450.8，6 格 VWA-20feat / 8 格匹配-18feat）**: ① 存活 cell (reddit·B2) AUROC/best-single **0.615/0.800** · **0.526/0.790**，两个口径都高于随机 ② AUROC 区间 **0.615–0.864** / **0.526–0.758** ③ 'no SR-lossless saving' 格数 **0** / **1** ④ base SR **2–27%** / **2–36%** ⑤ reddit·B2 learned vs always-cheapest **+1.97pp/+2.2%** / **+2.46pp/+1.7%**。
- **置换控制（B=10000 bundle permutation，§450.17）**: **1/8 存活 Holm** (red·B2 p=**0.0004** < 0.0063)；**2/8 的 saving 被零假设大体复现**，其中 `cls·B1` 的 **null median 10.1% > observed 2.6%**；去掉语料自带两列后 **5/6 VWA 格 AUROC 下降**，最大 `red·B1` **−0.141**（其最强单特征恰是被 drop 的 `has_reference_image`）。
- **Pareto 胜区（8 cells, 18-feature 匹配集, x=log2(cost/cost_always_cheapest), y=SR−SR_always_cheapest，§450.12）**: learned triage (fully nested) **0/8**（与产物 :124『0 of 8』独立吻合）；oracle_triage (事后) **仅 1/8**。
- **学习曲线（六个 VWA 格, 5-fold × 20 repeats, seed 42，§453.1）**: AUROC 25%→100% 涨 **+0.032~+0.083**（18 特征部署集）；12 条曲线里 **9 条**减速；近乎无正则化 (C=1e6) in-sample 超出置换记忆地板 **+0.075~+0.116**，6 格里 5 格 p≤0.01。

**演变**:
- §399.4: 发现 `router_triage_learnability.py:98` `_feature_row` 从 step-0 读 `intent_token_count` / `reasoning_difficulty` ⇒ **20 个特征里 2 个恒为 0**，影响 §388.4/§392.2 系列数字。
- §412: visual_difficulty 加进 triage 特征表，OOF AUROC 变化 cls_B0 −0.0064 · cls_B1 −0.0139 · cls_B2 −0.0081 · red_B0 +0.0014 · red_B1 +0.0277 · red_B2 +0.0461，mean **+0.0078**，改善 3/6 格（见 C5）。
- §450.2: 20 vs 18 特征并排。
- §450.4: 发现 **4 处硬编码 `6`**（`:760`『4/6』· `:767`『m=6』Holm · `:787`『0 of 6』· `:791`『one of six cells』）；8 格族 Holm 最紧阈值应为 0.05/8=**6.25e-3** 而非 8.33e-3。
- §450.8: 散文参数化重渲染 → 上述 5 个连带数字更正；§450.17 的 F12 按 m=8 独立重算 Holm 与产物一致，「反过来验证了 §450.8 修的散文参数化是对的」。

**已作废**（被 §450.8 推翻，**禁引**）:
- 存活 cell 「AUROC/best-single **0.483/0.711**, 'below chance'」→ 0.615/0.800（6 格）/ 0.526/0.790（8 格）。
- AUROC 区间「**0.651–0.717**」→ 0.615–0.864 / 0.526–0.758。
- 「'no SR-lossless saving' **2** 格」→ 0 / 1；「Two cells yield no saving」与自身 §3 表格矛盾。
- 「reddit·B2 +1.97pp/**~2.4%**」→ +1.97pp/+2.2%。
- 「one of six cells」与自身「2 of 6 reject」矛盾；with_wa 文件里「Holm's tightest threshold over six cells is 0.05/6」+「m=6 … 1 of 6 reject」为 6 格残留。
- （这几条推翻的是 D4 B10/B11 引过的数字，见文末 supersede。）

**caveats**:
- §399.4: 「方向性: 特征更少只让 learned 臂更弱, 而 §392.2 是负结果, 故其结论偏保守不翻; 但要引用该系列 AUROC 绝对值需先重跑。router_pooled_tier_learnability.py 走 canonical 路径未继承此缺陷」。
- §450.2: 「两个特征集**不可比但必须并排**: 18 特征集不是 20 特征表的子集, 每格都重拟合过」；「20 特征的高 AUROC 部分来自部署拿不到的一列, 所以 18 特征才是 deployment-faithful 口径」；不外推到别的 benchmark。
- §450.8: 「**`base SR` = `baseline_policy.sr_pct` (最强单模自身 SR), 不是 `solvable_rate_pct` (任一 mode 可解比例, 7–43%/7–52%)**」；「表格数字**全部未变** (逐字 diff 验证), 变的只有散文」。
- §450.4: 「**未修** (改它要连带重跑产物 + witness + 台账)」；「这是**该脚本的复发型缺陷**, 修时应把整段 cell-count 参数化而非逐处替换」。（§450.8 随后完成参数化。）
- §450.12（一字不改）: 「🔴 **oracle 也只有 1/8 进胜区 ⇒ C4 的负结果部分是对照选择的产物** … ⇒ 正确表述是『**在该成本口径下不存在划算的可部署点**』, **不是**『学习器不行』。0/8 仍成立, 校准的是性质不是结论」；「C1b 的 triage_only 省 9.5–30.6% 是**相对 best-SR fixed mode**, 基线不同 —— 引用时必须带基线名」。
- §450.17: 「`cls·B1` 零假设高于实测 = 该格**无信号**的直接证据」；「过零假设**必要不充分**」；WA 两格无 20 特征对应。
- §453.1: 「**两条都对本文不利**: 曲线仍在涨 + 特征确有信号 ⇒ **不能**用『没东西可学』打发欠采样攻击」；减速判据用 0.55 切分（中位索引报 12/12，取保守 9/12）；which-mode 半边行数不够做曲线。

**证据**: §399.4 / §412 / §450.2 / §450.4 / §450.8 / §450.12 / §450.17 / §453.1；`scripts/analysis/router_triage_learnability.py`、`docs/analysis/cross_sites/router_triage_learnability{,_with_wa}.{md,json}`、`docs/analysis/cross_sites/router_undersampling_control.md`、FIGURE_PLAN F11/F12/F13、CLAIM_EVIDENCE_MATRIX C4

**原文片段**: 「存活 cell (reddit·B2) AUROC/best-single: **0.615/0.800** (6 格) · **0.526/0.790** (8 格) — 旧写 0.483/0.711 且称 'below chance', **两个口径都高于随机**」(§450.8)；「**learned triage (fully nested): 0/8** … **oracle_triage (事后): 仅 1/8**」(§450.12)

---

## C3. abstention（「六 mode 是否全失败」）—— 标签充足、排序可学、操作点不可迁移

**当前值**:
- **标签供给（6 个 VWA cell, 六 mode N=1 each，§457.1）**: 每 cell 全部 n 都可训（cls 224 / red 203）vs which-mode 的 **97/53/55/24/16/15** ⇒ **2.3× / 3.8× / 4.1× / 8.5× / 14.0× / 13.5×**；`min_class_n=10` 下 which-mode **4/6 cell 拟合不了**，abstention **6/6 全部可拟合**（最小类 = B2 的 solvable 侧 15-16）；universal-fail 占比 **56.7% (B0·cls) ~ 92.9% (B2·cls)**。
- **held-out AUROC 与 label-shuffle null（task-level 5-fold SEED=42 / fold-local 标准化 + L2 LR (C=1.0)，§457.2）**: B1·red **0.864** (null 0.477, gap +0.387) · B0·red **0.780** (0.476, +0.304) · B1·cls **0.732** (0.478, +0.254) · B0·cls **0.726** (0.347, +0.379) · B2·cls **0.642** (0.604, **+0.038**) · B2·red **0.615** (0.462, +0.153)。[聚合者注: 这六个 AUROC 与 C2 §450.2 的 20 特征 triage AUROC 逐格相同 —— abstention 标签就是 triage 标签的补。]
- **可部署省钱前沿（嵌套阈值: 内层 5-fold 选阈值，外折未见地评估，§465）**: ≤5% 损失预算下 B0·cls **15.1%** (实损 3 solvable) · B0·red **22.8%** (1) · B1·cls **14.4%** (2) · B1·red **24.6%** (0) · B2·cls 6.0% (1) · B2·red 14.5% (**4, 占该 cell 15 个 solvable 的 27%**)；0% 预算下 2.3%(0) / 6.2%(1) / 9.6%(1) / 19.8%(0) / 6.0%(1) / 14.5%(4)。⇒ **可引用的区间是四个 B0/B1 cell 的 14.4–24.6%**。
- **oracle 上界（§457.3，分母 = per-task min(cost_dom, cost_psom) 之和）**: **63.8% / 78.5% / 79.3% / 90.3% / 93.6% / 92.3%**。
- **leave-one-site-out 迁移（6 个 VWA cell，只换 split，§467.2）**: **排序迁移得动** —— pooled 6 个里 5 个过自己的 label-shuffle 零分布；相对格内天花板落在 **−0.141 ~ +0.130**（`B2_reddit` 训外站反而更好）。**操作点迁移不动** —— 24 个迁移阈值只有 **15 个**在测试站守住买来的预算；最坏 `B1_reddit→B1_classifieds` 在 ≤5% 预算下放弃 **70.5%**、毁掉 **24/55 (43.6%)** 可解题，而它的 AUROC **0.738 是 6 个 matched 里最高**且高于自己格内天花板 **0.732**。
- **200 次置换后的迁移判决（§468.1）**: `B2_cls→B2_red` **p=0.050 clears**；`B2_red→B2_cls` **p=0.313 判不了**；经验 null p95 = **0.637–0.731**，高于解析 Hanley-McNeil SD 给的 **0.628**。
- **pooled 训练集特征相同但标签冲突（§468.3，20 维特征 × B0/B1/B2）**: classifieds 224 task: 特征三模型逐字节相同 **131 (58.5%)** · 标签冲突 **94 (42.0%)** · 两者同时 **56 (25.0%)**；reddit 同口径 **12 (5.9%) / 51 / 3 (1.5%)** ⇒ 撤回 pooled 协议。

**演变**: §457.1–§457.3（标签 / AUROC / oracle+held-out 省钱）→ §465（嵌套阈值，可部署）→ §467.2（换站）→ §468.1（单次置换判反，改 200 次）→ §468.3（pooled 撤回）。

**已作废**:
- §457.3 的 held-out 省钱各列（0 损失 = 4.2% / 4.4% / 8.5% / **24.7%** / 4.7% / 0.8%；≤2% = 6.2 / **15.2** / 14.7 / 24.7 / 4.7 / 0.8%；≤5% = **11.2 / 16.7 / 19.1 / 47.2** / 4.7 / 0.8%；≤10% = 22.1 / 38.8 / 34.9 / 47.9 / 23.8 / 5.4%）作为「可部署」数字 → 被 **§465** 的嵌套阈值版替代（「『零损失』不再是承诺」）。oracle 那一行仍作上界有效。
- §467.2 v1 用**单次 permutation** 当零分布的两条判决 → 被 **§468.1** 两个方向全部判反（`B2_cls→B2_red` v1『什么都没学到』→ p=0.050 clears；`B2_red→B2_cls` v1 默认过 → p=0.313 判不了）。

**caveats**:
- §457.1: 「label 是 N=1/mode —— 一次抽样里无 mode 解出 ≠ 不可解; 本 cell 同 condition rerun discordance 12-14% ⇒ 一部分 universal-fail 二次抽样会翻」；WA cell 不在 extractor 的 CELLS 里。
- §457.2: 「**B2 两格弱 (+0.038 / +0.153) 应视为无实质信号**」；「特征含 step-0 观测统计 ⇒ 决策需第一页加载完 (但无 model call)」。⚠️ §468.1 记「同缺陷仍在 `abstention_learnability` 的 `auroc_label_shuffle` 列 (本轮未改)」⇒ **§457.2 的 null 与 gap 列是单次置换，未修**。
- §457.3: 「**oracle 与 held-out 差一个数量级, 两条都必须报**」；与 §5（稿件章节）9.5-30.6% 「**口径不同** … 不可直接相减」；「**pre-flight 但不免费**」；B2 两格省钱数字不可用。
- §465: 「nested 有时**高于** oracle (B0·red 6.2% vs 4.4%), 因为 oracle 被『外层必须零损失』硬卡住而 nested 不受该约束 —— 多省的钱是用真实损失买的」；「**B2·red 的 14.5% 不可用** (丢 27% 的可解任务)」。
- §467.2: 「**两个站是两个点**」；「机制是 base-rate 漂移 (universal-fail 56.7%–92.9%), AUROC 是 rank-based **结构上看不见**」；saved 列成本来自 `per_task_sr.csv` (leak-kept, reddit n=205)，标签来自 canonical n=203。
- §468.1: 「解析 SD 是下界, 不能用它代替 permutation」；「未过的那条应写 **indeterminate, not negative**」（零分布自身 SD **0.075**）。
- §468.3: 「**两站极不对称** (cls 58.5% vs red 5.9%) —— 因为 `text_length`/`tokens_input_text` 在 reddit 上逐模型不同而 cls 上相同」；只否掉 pooled 协议，matched-model 协议不受影响。

**证据**: §457.1 / §457.2 / §457.3 / §465 / §467.2 / §468.1 / §468.3；`docs/analysis/cross_sites/abstention_learnability.md`、`docs/analysis/cross_sites/abstention_site_transfer.md`、`scripts/analysis/abstention_site_transfer.py`

**原文片段**: 「⇒ 可引用的区间是四个 B0/B1 cell 的 **14.4–24.6%**」(§465)；「**看排序最好的方向作为策略失败最惨**」(§467.2)

> §475 之后见 measured_D6.md

---

## C4. early-abort 与 confidence 信号的聚合窗口

**当前值**:
- **early-abort（B0 × VWA-classifieds × 6 mode × k∈{3,5,10}，同 §6 CV/estimator/null，§459.2）**: AUROC **0.336–0.667** vs label-shuffle null **0.368–0.622**，**多行低于自己的 null**（P-text k=10 0.336<0.368 · P-SoM k=3 0.355<0.522 · SoM k=10 0.516<0.622）；最好 DOM k=10 **0.667** (null 0.423)。matched-loss 下 **18/18 全输给 truncate-at-k**（DOM k=3: fixed 81.3% 步 vs learned 70.5%, 同 −34 succ）。零损失前沿 learned 省 **0–7.5%** 步，fixed 省 0%。
- **三个 confidence 数字的对照（§460.3）**: `routing_signals.md` (2026-04) context 行 **AUROC = 0.497 (non-discriminative)**；证据层 **0b-extra** B1·cls 最佳 routing AUROC **0.870**；§459 前 k 步重算 **0.336–0.667** ⇒ 「**§459 与 2026-04 那条一致, 与 0b-extra 不一致**」。

**已作废**: 以 0b-extra **0.870** 作为「early-abort 有信号」的依据 —— 被 **§460.3** 否定（「它曾被这样用」；已在证据层该表下加 ⚠️ 注）。

**caveats**:
- §459.2: 「**每行 conditional on surviving to step k** … 是残余困难子集上的判别力不是全体 (n 随 k 从 217 降到 130)」；「**matched-loss 那个操作点不可部署** (truncate k=3 丢掉约 34/39 个 success)」；单 cell 单 backbone，label N=1。
- §460.3: 「三个数字口径不同不可直接相减 —— 但**聚合窗口**是解释差异的那个变量: 0b-extra 的高值来自整 episode 聚合, 对任何 prefix 决策等于看未来」。

**证据**: §459.2 / §460.3；`docs/analysis/cross_sites/early_abort_B0_classifieds.md`、`docs/literature/routing_signals.md`、`docs/analysis/layered_evidence_status.md`

**原文片段**: 「matched-loss 下 **18/18 全输给 truncate-at-k**」(§459.2)；「0b-extra **不可**作为 early-abort 有信号的依据 (它曾被这样用)」(§460.3)

---

## C5. 路由特征诊断：has_reference_image / visual_difficulty / 平局 / 0-token 正则分区

**当前值**:
- **has_reference_image 符号是反的（6 VWA 格，§407.5）**: cls_B0 带参考图 (65 题): dom 33.8 som 40.0 vision 30.8 ptext 35.4 pprompt 36.9 psom 32.3；不带参考图 (159 题): dom 10.7 som 22.0 vision 22.6 ptext 7.5 pprompt 12.6 psom 8.8；臂数对齐的边际收益同向，「直觉只在 1/6 格成立」。
- **visual_difficulty 分层（6 VWA 格，§407.5）**: easy **+1.41** → medium **+2.15** → hard **+2.42pp**（均值单调），但只在 classifieds 单调（cls_B0 1.5→6.5→14.8），reddit 反号。
- **visual_difficulty 进 triage 特征表（§412，与 router_triage_learnability 同一 OOF LR）**: mean **+0.0078**，改善 3/6 格；符号按站点分裂（cls 三格全负 / red 三格全正）。
- **平局率（6 VWA 格 x 2 stratum，§408.1）**: 12 个 (cell, stratum) 里 **5 个**有平局；但 intuition_holds 在全部 tie-break 组合下 **6/6 一致**；产物现报并列范围（red_B1 那格 **-0.81…+0.81**, 跨零）。
- **0-token 事前正则分区（VISUAL_INTENT_RE AND 无任务级参考图；配对 bootstrap 10000, seed 20260803，§419.3）**: classifieds 标记集 **71/224**: cls_B0 vision **+22.54pp [+9.86,+33.80]** (23/71 vs 7/71)、cls_B1 **+16.90pp [+8.45,+25.35]** 排除零，cls_B2 **+1.41pp** 跨零；补集 153 个 **+0.65 / +1.31 / +0.65pp** 全部跨零；vision > som (+22.54 vs +19.72) ⇒ 标记的是「需要图」不是「需要 SoM 标注」。

**已作废**: §408.1 之前 routing_feature_diagnostics 表里「最强文本模式」平局格的**具体符号数字**（如 red_B1 +0.81 / −0.81）—— 是 `max(set)` + str hash 的「抽签结果」，现以并列范围替代；结论「直觉只在 1/6 格成立」不受影响。

**caveats**:
- §407.5: 「main.py:2890 把 reference_images 塞进 BackendStepContext 无任何按 mode 过滤 ⇒ 六个模式全收得到题目自带的参考图, phantom 缺的是页面截图不是参考图。has_reference_image 测的是『题目里有没有图』, 决定路由的是『页面要不要用眼睛读』」；「extract_50_features.py:334 读出来了然后没进特征表 … 更好的特征不是救星, 不改 §4 的算术 —— 四个格子仍凑不出两个类」。
- §412: 「在这个规模的 fold split 噪声内; 最大改善那格 (red_B2) AUROC 仅 0.48→0.53 ≈ 随机」；label 是 triage 不是 which-mode。⚠️ 其 red_B2 基线 0.48 是 §450.8 修正前的旧值（见矛盾清单 #1）。
- §419.3（一字不改五条）: 「① **要能力才能兑现** … ② **reddit 上反号**: red_B0 −3.17pp, 截图有害。③ **WA 不能检验**: 同一正则在 WA 只标 5/104 且 0/5 vs 0/5 全员失败 ⇒ 退化格无信息, 非 measured null。④ **不是 P43 as shipped** —— 剥掉了 `if summary.get('success'): return []`。⑤ **不是 router**: 分区固定不学习」。

**证据**: §407.5 / §408.1 / §412 / §419.3；`docs/analysis/cross_sites/routing_feature_diagnostics.md`、`docs/analysis/cross_sites/visual_difficulty_router.md`、`docs/analysis/cross_sites/visual_intent_routing.md`

**原文片段**: 「has_reference_image 测的是「题目里有没有图」, 决定路由的是「页面要不要用眼睛读」」(§407.5)；「cls_B0 vision +22.54pp [+9.86,+33.80] (23/71 vs 7/71, 排除零)」(§419.3)

---

# D. 融合 / 模态轴 / 载荷

## D1. fusion premium —— 融合是「保险」不是「能力」

**当前值（7 格, 配对 bootstrap 10000 次 seeded LCG，§407.21）**:
- SoM − Vision: k=7, 池化 **+1.43pp, 95% CI [+0.12, +2.75]**，越过 0 但不越过重跑带 (0.89-2.23)；SoM − DOM: k=7, **+0.89pp, [−0.40, +2.18]**，连 0 都没越过。
- 逐格对「与负载匹配的单通道」: cls_B0/B1/B2 对 vision **+2.23/+1.79/+0.00**，red_B0/B1 对 dom **+0.49/+1.48**（全含 0），wa_red_B1 **−2.88** 含 0。
- red_B2 对 dom: §407.21 报 **−2.96 [−5.91, −0.49]** 显著为负 → **§418.2 泄漏敏感性后 −1.48pp [−3.45,+0.49] 跨零**（6 个环境送的成功置 0，分母不变，同 seed 20260802 同 10000 次）⇒ 「八格里唯一那条『融合显著输给单通道』翻转」；red_B0/red_B1 的 SoM−Vision 仍排除零，red_B0 反而离零更远。
- 「7/7 格融合从来没有显著赢过与负载匹配的那个单通道, 只赢过不匹配的那个 (cls 上对 DOM +8.04/+9.82, red 上对 Vision +4.93/+7.39)」；保费 **+2.5% 到 +17.7%**。

**演变**:
- §407.2: 「融合 (SoM) 相对 max(最强文本, 纯视觉)」7 格: cls_B0 +2.23pp/+5.6% · cls_B1 +1.79/+2.5% · red_B1 +1.48/+9.1% · red_B0 +0.49/+8.8% · cls_B2 +0.00/+7.4% · red_B2 −2.96/+17.7% · wa_red_B1 −2.88；臂数对齐独占覆盖 1 胜 3 平 3 负。
- §407.21: 判 §407.2 估计量偏向自己（两个含噪量取 max），改用先验固定对照 + FE 池化。
- §418.2: 泄漏置零后 red_B2 翻转。
- §463.3 (Kimi K3 读出): fusion_premium 的 **I² = 53% / 75%** ⇒ 「异质性大到不存在可 pool 的共同效应, pooled 估计不该作主证据 (该用 per-cell: 0/8 clear, 唯一显著区间为负)」。

**已作废**:
- §407.2 的 max(最强文本, 纯视觉) 估计量 → 被 **§407.21** 替代（台账 superseded_by=§407.21）。
- red_B2 「融合显著输给单通道 −2.96 [−5.91, −0.49]」→ 被 **§418.2** 翻为跨零。

**caveats**:
- §407.2: 「没有任何一格里融合的优势明确超过重跑地板 (2.23pp 上沿)… 只能说『形状不匹配』不能说 refute」。
- §407.21: 「post-hoc / exploratory, 不是 H1, 不 gate」；「『更强的单通道』是逐格读出来的 ⇒ post-hoc。仍缺融合模式自己的重跑地板 (地板测在 DOM/Vision 上)」（§462.1 记该句「SoM replicate 缺失」**已 stale** —— §450.14 起已有 SoM 重跑）。
- §418.2: 「6 个泄漏里 4 个在 DOM ⇒ 去掉它们是**帮**融合臂的, 即对本文自身保守立场不利的方向。**WA 两格未 audit**」。
- §463.3: 「三条都是 Kimi 从产物原文读出并重新解释的, **我未独立复核数字**。引用前应回原产物核对」。⚠️ Kimi 的「唯一显著区间为负」与 §418.2 的跨零结论需对齐口径（见矛盾清单 #5）。

**证据**: §407.2 / §407.21 / §418.2 / §463.3；`docs/analysis/cross_sites/fusion_premium.md`、`docs/analysis/cross_sites/leakage_sensitivity.md`、`docs/checkpoints/codex_outputs/frame_zero_preset_kimi-k3_2026-08-13.md`

**原文片段**: 「⇒ 融合的价值是『保险』不是『能力』, 保的是你选错通道, 保费 +2.5% 到 +17.7%」(§407.21)；「red_B2 SoM−DOM 从 −2.96pp [−5.91,−0.49] 变 −1.48pp [−3.45,+0.49] **跨零**」(§418.2)

---

## D2. 任务模态轴：VWA vs WA（WA-reddit 第七/八格）

**当前值**:
- **结构差异（§406，config 直接计数）**: VWA reddit **84/210 = 40.0%** 的 task 带 goal-side reference image；WA reddit **0/106 = 0.0%**（schema 里没有 image 字段）。
- **计分宇宙内带参考图占比（§407.17）**: classifieds **65/224 = 29.0%** · reddit **79/203 = 38.9%** · 合计 **432/1281 = 33.7%**。
- **B1 × WA-reddit 六 mode 全量 SR（n=104 = 106 − 2 个 N/A per §139.8，全 scored pool，§406）**: dom 17/104 = **16.35%** · P-text 17 = **16.35%** · P-prompt 17 = **16.35%** · som 14 = **13.46%** · P-SoM 12 = **11.54%** · vision 10 = **9.62%**；6-mode oracle **30.77%**。
- **axis 管线第七格（B1 x WA-reddit n=104，§412）**: VWA 6 格 dominant cascade axis: text 12 · prompt 9 · image 19；WA 1 格: text 1 · prompt 4 · image 2；7 格合计 Tier-1: effect-only 18 · BH 7 · off-segment 6 · Holm 1（BH 幸存全来自 VWA）。
- **axis1 decision-over-macro ratio（8 格 --with-wa，§420.5）**: B0/wa_reddit = **0.97, 首次跌破 1**；B1/wa_reddit = **2.98**；VWA 六格 **1.34–4.07** 不变；target-hit diff 在 WA 是 **9.47/12.63pp** 而 VWA 六格**全 0.00pp**。
- **WA step 记录存在（B1 x WA-reddit，§407.25）**: 六个 WA mode 每个 104 个 step 文件全部在 paper-grade host 上，**132.6 MB**；v8 ruleset 原样跑 WA（**76-84/104** episode 有命中）。
- 模态轴的边际证据（加第二截图臂 vs 加文本臂在 VWA cls 与 WA 方向相反）见 A4 §407.3。

**演变**: §406 首次结构计数（40.0%）→ §407.17 改为计分宇宙口径（29.0 / 38.9 / 33.7%）→ §445.3 / §450.16 语料级全表（见 H1）。

**已作废**: 「VWA 三站带参考图 **40.0%**」作为计分宇宙数字 → 被 **§407.17** 修正（「旧数 40.0% 是 §4:91 就有的, 且被加了个 .0 给它一个不存在的精度」）。§406 的 84/210 = 40.0% 作为 **reddit 语料**计数本身仍有效。

**caveats**:
- §406: 「这是 by-construction 的结构事实, 与台账 C3 争议中的两套『视觉任务占比』定义 (自动 84.3% / codex 手动 99.5%) 是不同的量, 不可互换引用。⚠️ 由此: 『VWA 全是视觉任务』的说法结构上偏强 —— 只有 40% 带目标图」；WA SR「exploratory, 未 promote 进 run_manifest.yaml; 最高单模式仅 17 个标签」⇒ 标签供给瓶颈的跨 benchmark 确认。
- §407.17: 「逐站差异 29.0 vs 38.9 不伤模态轴论证 —— 轴是 VWA-vs-WA 不是 cls-vs-red, WA 是 0%」。
- §412: 「WA 的 3 个 effect-only 组合多重比较后一个不剩 (n=104 对 203/224), 且逐 mode 计数是描述不是检验」。
- §420.5: 「verdict 仍为 generalizes, 但**靠 B1 的 2.98 通过而非 B0** —— `_site_ok` 是『任一 baseline ratio>1 即该 site 通过』的宽松判据。修 verdict 前它**根本没看 wa_reddit**」。
- §407.25: 「WA reddit 与 VWA reddit 是同一个 Postmill 应用。sync_a100_results.sh 的注释自己写着 keep step JSONLs ⇒ 是同步缺口不是策略」。

**证据**: §406 / §407.17 / §407.25 / §412 / §420.5；`external/visualwebarena/config_files/`、`results/webarena/phase1/`、`docs/analysis/cross_sites/routing_feature_diagnostics.md`、`docs/analysis/cross_sites/axis_effect_size_report_with_wa.md`、`docs/analysis/cross_sites/axis1_microbehavior_report_with_wa.md`

**原文片段**: 「VWA reddit 84/210 = 40.0% 的 task 带 goal-side reference image; WA reddit 0/106 = 0.0%」(§406)；「B0/wa_reddit = **0.97, 首次跌破 1**」(§420.5)

---

## D3. SoM vs DOM 的文本载荷 —— 多花的钱在图上

**当前值**:
- **step_000 字符比（B0 dom-run vs som-run 按 (task, step_000) 配对，§452.4）**: cls median **1.0077** (n=224 任务, 97.3% 落在 ±5%) / red median **1.0041** (n=205, 71.2%)。
- **单页三 mode 负载（B0 × VWA-cls × task 0 × step 000，§450.15）**: DOM 2,602 chars AXTree, 无图, **3,314** input tokens · SoM **2,622 chars** `[SOM_MARKS]` (= DOM 的 **1.008×**), 143 KB 图, 30 marks, **4,335** tokens · Vision 无结构文本, 110 KB 图, **3,123** tokens。
- 成本由步数驱动而非 token（§449.1，见 J）。

**已作废**: 稿件里引「1.008× … on a matched sample」—— 来自 fig_f1 脚本的**单个 step**（n=1 当样本），被 **§452.4** 替换为 n=224 / n=205 的 step_000 中位数。

**caveats**:
- §452.4: 「**只有 step_000 有意义** —— 之后两个 run 已选了不同动作、在**不同页面**上; 全步 pooling (2506 对) 得 median 0.980 / p05 0.38 / p95 1.86 / max 29.1, 是假量」。
- §450.15: 「**`observation_dom.txt` 所有 mode 都是同样的 2602 chars** —— 它是页面 AXTree 快照(记录用), **不是**该 mode 送出的东西」；单步单任务，不是全条件均值。

**证据**: §450.15 / §452.4；`results/mechanistic/_obs_mirror`

**原文片段**: 「**SoM 的文本只比 DOM 多 0.8%, 多花的钱几乎全在那张图上**」(§450.15)

---

# E. 行为画像 / 失败归因 / 数据卫生

## E1. per-mode 四维画像 —— 经验发现 0 个，四个无截图模式始终 0

**当前值（6 VWA 格 x 26 指标，§412）**: ≥5/6 计数 **Vision 9 · SoM 8 · DOM/P-text/P-prompt/P-SoM 各 0**；第 26 个指标 scroll_inert_rate: Vision 5/6 最高 · SoM 5/6 最低，比值 **1.23-3.50x**。
**18 指标的 7 个 6/6 一致极值（§401.2 三分类）**: **0 个经验发现** / 3 个架构下游（scroll_frac · action_fail_rate · no_change_rate）/ 4 个构造必然（locator_fallback · tokens · cost · cost_rel_dom）。

**演变**:
- §400.2: 18 指标中 7 个有 6/6 一致极值 mode，全是 Vision；当时判 **3 个经验发现**（scroll_frac 最高 1.25-6.77× / action-execution failure rate 最高 1.06-1.60× / page-unchanged no-op rate 最高 1.07-1.58×）+ 4 个 by construction。
- §401.2: tie 修复后 unique-solves 极值 mode 由 SoM 变 Vision；三分类后经验发现 **3 → 0**。
- §401.5: step 级比率两个 estimand —— B1·cls Vision action-failure task-macro **0.4540** vs pooled-step **0.6386**；B0·cls Vision **0.1499 vs 0.2238**；B0·red Vision **0.3780 vs 0.4565**；均已并列报。
- §407.6: 一致度门槛随机排序期望假阳（18 指标 x 2 端 = 36 检验）: 6/6 → **0.005** · 5/6 → **0.144** · 4/6 → **1.880**；放宽到 5/6 后 SoM 有 5 个指标够格而三个 phantom 仍 0 个。
- §407.22: 24 指标 —— Vision ≥5/6 **8** 个（7 带标记, 1 未裁定 = URL 重访）；SoM **7** 个（其中三条是同一签名数了三次）；DOM / P-SoM / P-text / P-prompt **全部 0**。
- §408.2: P-SoM 同时区别于 DOM 和 SoM 的 (metric, cell) 组合 **15 个**，覆盖全部 6 格（finish_rate 5/6 格独立, n_steps 3 格, scroll_frac 2 格）；另 7 个只区别于 DOM、14 个只区别于 SoM、12 个与两端点都不可分（|effect|>0.1, 192 个 contrast）。
- §409.5: binary contrast 有效样本 —— finish_rate: B1/reddit n=203 里 **55** 个 discordant；B2/classifieds n=224 里仅 **26** 个；Wilcoxon 与 McNemar exact 一致 **0.0104 vs 0.0145**。
- §412: 26 指标（见当前值）。

**已作废**: §400.2「**3 个经验发现**」→ 被 **§401.2** 作废（台账 superseded_by=§401.2）。

**caveats**:
- §400.2/§401.2: 「⚙️ by-construction 四条不得当行为发现引用 —— Vision 发坐标零 element id 故几乎不进 locator 路径 (残留 0.002-0.011 …)」；「◆ 三条同属一条机械链: 坐标寻址 → 点不准 → 页面不变 → 被迫滚动重定位。幅度真实但方向可从设计预测; 升格为行为发现需先建立『坐标寻址系统应得多少』的基线再证明 Vision 超出」；「Evidence ≠ Explanation」；post_hoc_exploratory。
- §401.5: 「未声明 estimand 的比率不可复现 (codex cross-AI Mode B 2026-07-29)」。
- §407.22: 「加了 6 个专门去找差异的指标之后四个无截图模式仍然是 0 —— 这条否定比 18 指标时更强」；click/type 分失败率 Vision 仅 3/6 最高与 4/6 最低。
- §408.2: 「多重比较未校正 (192 个 contrast), post-hoc」；与「四个无截图模式行为不可分」不矛盾（配对 contrast vs mode-vs-mode 极值，两个不同的量）。
- §409.5: 「问题是披露 —— 报的 n 是配对数, 读者据以判断结论建立在多少数据上的是 discordant 数」。
- §412: 「该指标来自未读字段普查而非为找差异而设计, 落点仍只在有截图一侧。18→24→25→26 指标, 四个无截图模式始终 0」。

**证据**: §400.2 / §401.2 / §401.5 / §407.6 / §407.22 / §408.2 / §409.5 / §412；`docs/analysis/cross_sites/per_mode_four_dimension_profile.md`、`docs/analysis/cross_sites/axis_effect_size_report.md`

**原文片段**: 「7 个 6/6 一致极值中 **0 个经验发现** / 3 个架构下游 / 4 个构造必然」(§401.2)；「26 指标下 ≥5/6 计数: Vision 9 · SoM 8 · DOM/P-text/P-prompt/P-SoM 各 **0**」(§412)

---

## E2. steps↔summary 身份审计与 universe lint 盲区

**当前值**:
- **全库身份不一致（6 cell × 6 mode, strict_identity=True，§400.1）**: 36 组合 / **7686** scored episode 中仅 **2 个 = 0.03%**，全在 B0·red·P-SoM（task 87, 149）；另 36 个 step 文件是 AMENDMENT_08 protocol-excluded (reddit 58/160) 已排除未审。
- **根因（B0_phantom_som_reddit R28173，§400.1）**: quarantine→resume rerun 写了新 summary 但没换 steps JSONL；task 149 旧 summary (steps=18 tokens=68741) 与当前 JSONL 逐位吻合，现 summary 的 **49657** 不等于 JSONL 任何前缀和 (step12=49101/step13=53024) ⇒ 两次不同执行；mtime 差 5 天 / 4.7 小时。
- **lint 盲区（§400.3）**: `test_universe_consumption_lint` 检「文件是否引用 canonical universe」而非「每条读取路径是否都过 gate」—— `per_mode_four_dimension_profile.py` 的 steps_layer 仍是裸 glob ⇒ Macro/Micro 均值被 protocol-excluded task 污染；加过滤重跑后 7 个 unanimous 结论不变（仅 type_frac 5/6→4/6 等小数点级变化）。

**已作废**: §400.1「**个案非流程缺陷**」措辞 → **§401.3(5) 收回**：「两 episode 走同一条 quarantine→resume 转换失败, 2/7686 界定 blast radius 而非机制非系统性」。

**caveats**: Outcome/Efficiency 读 summary 不受影响；Macro/Micro 须排除并披露；「strict_identity 检测正确不是误报」；§400.3「影响不实质但分母正确性必须」。

**证据**: §400.1 / §400.3；`docs/analysis/cross_sites/steps_summary_identity_audit.json`、`tests/test_universe_consumption_lint.py`

**原文片段**: 「36 组合 / 7686 scored episode 中仅 2 个 = 0.03%, 全在 B0·red·P-SoM (task 87, 149)」(§400.1)

---

## E3. diag 失败归因：覆盖度、B2·reddit Tier-2、规则失明、方向不对称、新规则

**当前值**:
- **digest 覆盖（41 个 per-condition digest，§401.6）**: 41 = 36 paper-grade condition（6 cell × 6 mode 全覆盖）+ 5 run-specific；三分类可读 **31** · digest 自称不完整 **9** · 指针文件 **1** · 无法解析 **0**；含 non-agent-limit 信号 30 个，其中结构化计数非零 17 个。
- **layout（§401.6）**: 4 种 —— A 每类一行的表 (3) · B/C 三分类压单行 (14) · D 三列表但计数列是散文 (17) · 指针文件 (1)；解析器初版只认 layout A ⇒ **29 个被误报『无法解析』并被当成零**，打印出「pipeline 全干净」的假结论 (B-1913)。
- **B2·reddit Tier-2 补齐（B2 × reddit × 6 mode，§402.1）**: **14 个 no-hit failed 全部 agent-limit** (high confidence ×14)；**scaffold-bug 0 · benchmark-FP 0**；task 179 `invalid_select_option` = parse guard 正常；64/vision `policy_blocked_offsite` = 护栏按设计工作；B1↔B2 matched-capability: task 129 B1 5/6 mode 解出而 B2 全败，task 171 同样 5/6 vs 0。
- **两条结构性失明（§402.3）**: (a) P19 三处硬编码 classifieds 的 'page=search' ⇒ 18 个 reddit condition 上从未 fire；补上后全语料 fire **144** 次 success-FP **0**；(b) `tokens.input_image` 对 B0 字段缺失（B0 som/vision 0 个 step 带该字段, B1 max=1984 / B2 max=2304）⇒ P43 及 keyed on image token 的规则对 B0 静默失明；另 **17383/87693=19.8%** 的失败步报 page_changed=True 但 URL 未变。
- **两方向失败机制不对称（cls + red 六格配对, v8 ruleset，§407.23）**: 图像通道赢 / 文本通道失败 (91 题 364 episode): P27 **2.98x** · P17 **2.20x** · P16 **2.18x** · P43 **1.66x** (196 hits) · P31 **0.47x** · P25 **0.26x**；文本通道赢 / 图像通道失败 (71 题 142 episode): 除一条 10-hit 的 P17 1.56x 外没有任何签名越过 1.5x。
- **绕开规则词汇表的六个候选机制（§412，n=142 分歧 episode 对 2304 基线失败）**: 从不搜索 0.83x · 跑满预算 0.81x · 半数步动作失败 0.74x · 半数步页面没变 0.77x · 五步内结束 1.15x · 有解析失败 0.84x ⇒ 图像通道在文本通道独解的题上失败得**更不病态**。
- **P47 / P48 三站触发率（36 VWA + 6 WA condition, ruleset 9-wa-p47p48，§411）**: P47: VWA-cls **0.00%**（站点门）· VWA-red **0.00%**（真测量, 12/205 到过, 0 个 finish）· WA-red **3.70%** (22/594 failed, 0/106 success)；P48: VWA-cls 0.00% · VWA-red **0.29%** (10/3434) · WA-red 0.00% (0/594)；计分宇宙内 success 命中 0。
- **P49 SUBMIT_PAGE_ANCHOR_MISCLICK（ruleset v11, 8 格，§420.2）**: **3.61×** —— text-wins 侧首条清过 1.5× 的规则；已因果验证（WA som task 610/614）；**8 个 hit 全在 WA 两格** (B0:5 / B1:3)，VWA 六格零贡献。
- **失败可诊断性（35 个 (baseline, site, mode) 格, 5-bucket taxonomy，§458.1）**: 全表 **28.4%** (B2/red DOM) ~ **92.8%** (B0/cls P-prompt)；**Vision 在每个 cell 都最差或接近最差**: B0/cls 76.8% vs P-prompt 92.8% (−16.0) · B0/red 46.6% vs 81.0% (−34.4) · B1/cls **44.4% vs DOM 89.0% (−44.6)** · B1/red 37.7% vs 77.5% (−39.8) · B2/cls 47.0% · B2/red 32.5%。

**已作废**:
- 「pipeline 全干净」（解析器只认 layout A 时的输出）→ 被 **§401.6** 作废 (B-1913)。
- 「B2·reddit 整格 Tier-2 从没做过 ⇒ scaffold-bug/benchmark-FP 未知」→ 被 **§402.1** 补齐（0 / 0）。

**caveats**:
- §401.6: 「9 个自称不完整的分布成片 … corpus 级结论标 not admissible」；「教训: 解析失败必须与测量为零区分, 且覆盖度要先于内容报告」。
- §402.1: 「B2·reddit ~1-4% SR 是真能力地板, 与 §338 六源收敛一致」。
- §402.3: 「P14 自己的 R9725 FP-narrowing 因此压制了它在死锁上的命中」。
- §407.23: 「⚠️ TEXT 4 臂 vs IMAGE 2 臂 ⇒ 两侧任务数互相不可比, 只读组内富集; 富集是比值不是检验, 无区间; post-hoc」；「任何写成对称规律的版本都必须带上这个不对称」。
- §412: 「六个候选是我们选的, 证明不了不存在机制; 它关掉的是『残差是 VWA 形状规则词汇表的产物』这个具体反驳」。
- §411: P48 的 1 个 success 命中 = reddit_task_160，在 AMENDMENT_08 之外且为 passive-satisfiable 假成功。
- §420.2: 「8 正好是 MIN_HITS 门槛 ⇒ 正确表述是『WA 上残差有名字, VWA 六格上仍未解释』, **不是残差消失**」；「P49 是死因不是 risk-marker」。
- §458.1: 「**可诊断性是本 P-rule ruleset 的性质不是自然律** … 这是**下界**」；「B2 全线低 (28-61%) … 属 sanity check 而非新发现」；「这是失败**归因**能力不是失败**率**」。

**证据**: §401.6 / §402.1 / §402.3 / §407.23 / §411 / §412 / §420.2 / §458.1；`docs/analysis/cross_sites/diag_digest_index.md`、`scripts/analysis/index_diag_digests.py`、`scripts/analysis/diag_pattern_match.py`、`docs/analysis/cross_sites/conditional_failure_attribution.md`、`results/diag_scans/v9_vwa/` + `results/diag_scans/v9_wa/`、`docs/analysis/cross_sites/representation_deployment_profile.md`

**原文片段**: 「图像通道的优势有可命名的机制, 文本通道的优势没有」(§407.23)；「解析器初版只认 layout A ⇒ 29 个被误报『无法解析』并被当成零, 进而打印出『pipeline 全干净』的假结论 (B-1913)」(§401.6)

---

## E4. reddit sidebar 跨 episode 泄漏与 leak 政策一致性

**当前值**:
- **逐 episode 实查（9 个读 `#sidebar>section>ul` 的 reddit task × 18 model×mode，§402.6）**: scored universe 内 **LEAKED 6 · earned 31 · failed 107**；六个泄漏 = B0·DOM t171 / B0·Vision t189 / B1·SoM t189 / B2·DOM t178+t188+t189；占比 B2·DOM **37.5% (8中3)** · B0·Vision **6.7%** · B1·SoM **6.7%** · B0·DOM **3.4%** · 其余 14 格 **0%**；实质受影响仅 **B2·DOM 一格 (8→5)**。
- **leak 政策影响面（per_task_sr.csv 的 6 个 VWA cell，§465.1）**: 6 个 LEAKED **全部在 reddit**；置零后 universal-fail 计数 red_B0 150→150 · red_B1 179→179 · **red_B2 188→191 (+3)** · cls 三个全 +0 ⇒ 「**影响面 = 6 个 cell 里的 1 个, 3/205 = 1.5%**」；「**需重跑的产物为零, 需写进 limitation 的是一句话**」。
- **产物间政策不一致（§462.1）**: `routing_ceiling.md` 声明 **primary = leaked successes 置 0, 分母不变 (user decision 2026-08-04)**；`per_task_sr.csv` 的 red_B2 `sr_dom` = **8/205 = 3.90%** ⇒ leak-kept 且 n=205，而 canonical 是 leak-corrected 且 n=203（codex 报 corrected best-fixed = **2.46%**）。
- 泄漏对 fusion 的影响见 D1（§418.2）。

**演变**:
- §402.5: 7 个 task 共读同一 sidebar 选择器；占各 cell 成功数 B2·SoM **50.0%** · B2·DOM **37.5%** · B1·P-text/P-SoM **33.3%** · B1·SoM **20.0%** · B0·DOM **17.2%**；18 cell 成功分布 task160 13/18 · 188 11/18 · 189 9/18 · 178 8/18 · 171 7/18 · 170 1/18 · 190 1/18。
- §402.6: 纠正为逐 episode 判据 → LEAKED 6。
- §462.1 → §465.1: 确认 leak-kept 输入的影响面。

**已作废**:
- §402.5「reddit SR 里 **10-37%** 受污染」→ 被 **§402.6** 判为「过度警报」（那是 sidebar task 占成功数的比例，不是泄漏比例）。
- 「读 sidebar 的是 7 个 task」→ §402.6 改为 **9 个**（漏了 158 aww / 159 newark|nyc）。

**caveats**:
- §402.5: 「实证 B2·dom 的 178/188/189 三个判成功的 task 从未访问过所需 forum … 判定由执行顺序决定非 agent 能力」；require_reset 在 reddit 为 no-op（`envs.py:172 TODO(jykoh)`）。
- §402.6: 「31/37 的 sidebar 成功是真挣来的 … 不需动 scored universe」。
- §462.1（一字不改）: 「**本 session 所有用 `per_task_sr.csv` 的数字都在 leak-kept + n=205 口径上** (§457 的成本表、§455.3 的六臂预算对比)…camera-ready 前 0/36 与 0/8 分析必须在 primary 政策下重算 (无需新 episode)。同类一致性问题另有三条: label_instability 的 49/224 已 stale … / failure taxonomy 某些地方用 205 而 scored universe 是 203 / fusion_premium 里『SoM replicate 缺失』那句已 stale」。
- §465.1: 「**classifieds 零影响** ⇒ §464.2 的三臂 label_instability (B0×cls) 与 §455 的 retry-vs-switch (B0×cls) 数字干净」；唯一受影响的 red_B2 恰是已标注「AUROC 无实质信号、省钱数字不可用」的 cell。

**证据**: §402.5 / §402.6 / §462.1 / §465.1；`docs/analysis/cross_sites/reddit_sidebar_leakage_audit.md`、`docs/analysis/cross_sites/reddit_sidebar_leakage_audit_with_wa.json`、`docs/analysis/cross_sites/routing_ceiling.md`、`results/phantom_paper/per_task_sr.csv`、`external/visualwebarena/browser_env/envs.py`

**原文片段**: 「⚠️ 纠正 §402.5 的『reddit SR 里 10-37% 受污染』—— 那是 sidebar task 占成功数的比例不是泄漏比例, 属过度警报」(§402.6)；「影响面 = 6 个 cell 里的 1 个, 3/205 = 1.5%」(§465.1)

---

# F. 多指标 Pareto（latency / cost / carbon）

## F1. latency 是独立轴；碳轴否决；per-success 与 token 尾部

**当前值**:
- **三维 Pareto（6 VWA 格，§407.19）**: latency 跨度 **1.12-1.40x** vs cost **1.12-1.63x**；**3/6 格最便宜 ≠ 最快**（恰好是三个 classifieds 格, Vision 最便宜 / SoM 最快；三个 reddit 格两者同一）；加进去 3/6 格前沿变宽，B2·cls 从 1 个模式变 5 个；tokens 作第四轴每一格都不改变任何东西。
- **两种 latency estimand 下的稳健性（§408.4）**: raw 与 canonical 下都是 **3/6 格拓宽**，frontier 成员与每格最快模式完全相同；只有跨度变（B0/reddit **1.404x → 1.343x**, B0/cls **1.183 → 1.181**）。
- **canonical/raw 比值（6 VWA 格 x 6 mode，§408.4）**: B1/B2 全部 **1.000**（36 个比值）；B0/reddit P-text **0.890** · P-prompt **0.898** · Vision **0.930** · P-SoM **0.947** · DOM **0.966** · SoM **0.979**；B0/classifieds **0.972-0.993**；足以换位: raw 下 SoM 461.4 < P-prompt 498.4，canonical 下 P-prompt 447.7 < SoM 451.6。
- **cost~latency 秩相关（6 VWA 格 x 6 mode，§409.2）**: 逐格 ρ B0/B1/B2·cls = **-0.600 / -0.257 / -0.600**；B0/B1/B2·red = **+0.143 / +0.771 / -0.029**；mean = **-0.095**（canonical 下 -0.067）；逐格 exact permutation p 全不显著 (0.40-0.91)。
- **proxy 排队污染不成立（每 mode 前 60 episode，§407.15）**: B0 dom/som/vision 每步 **6.92/7.59/6.85s** CV **0.15/0.22/0.18** (tok/step 3804/4722/3470)；B1 本地 **13.67/13.63/12.69s** CV **0.19/0.16/0.11** (tok/step 2673/3238/2009)；B1 每步延迟几乎不随 token 变 ⇒ 本地推理被固定开销主导；vision 最便宜但 episode 最慢 = 步数最多 (13.8 vs som 11.8)。
- **碳轴否决依据（B1/B2 四格，§407.20）**: co2e~cost r = **0.45-0.95** (site-dependent)；co2e~latency r = **0.9999** 四格全部；power_watts 均值 **66.3/66.7W**, CV = **0.03**, 范围 **57.8-81.0W**；B1 321/321 步、B2 542/542 步全部采满。
- **站外导航与容器基线（8 格全覆盖，§418.3）**: 站外步 classifieds **0.00/0.00/0.16%**，reddit(VWA+WA) **1.05–2.13%**；**站外步更快**（6 格中 4 格 ratio **0.52–0.90×**）；站内 env_step 中位 classifieds **4.5–5.8s** vs reddit **6.6–11.3s = 1.69×**。
- **per-success vs per-attempt（6 VWA 格，配对 bootstrap 10000 次，§413）**: 有内容的 cell 4 个；分母换成 per-success 后最便宜的模式变了 **2/4**，最快的变了 **2/4**；Vision per-attempt 6/6 最便宜，per-success 只在 classifieds 三格最便宜；SoM per-success 延迟点估计四格全最优；**所有对比的 CI 全部重叠**（cls_B0: SoM 0.266[0.205,0.355] 对 Vision 0.259[0.200,0.348]）。
- **per-step 输入 token 尾部（B0 × {classifieds, reddit} × 6 mode，§458.2）**: B0×cls Vision p50 **3495** / p99 **4265** / max **4335** / p99÷p50 = **1.22**；带 AXTree 的五个 mode max 全在 **17752–18914**（p50 3663–4741, p99 5472–6794, p99÷p50 1.33–1.75）⇒ 文本表征 max 是 Vision max 的 **4.1–4.4×**、是自身 p50 的约 **4.7×**；B0×red 的 max 只 **5548–10679**。
- **Kimi K3 读出（§463.3，未独立复核）**: `latency_decomposition` 去掉容器时间后最快表征的排名在 **4/8 cell 翻转**（reddit 家族 4/5 翻, classifieds 0/3 翻）⇒ latency 排名是容器混淆的。

**已作废**: §407.20 前「碳/功率没采」的说法 → 被 §407.20 推翻（「此前说『没采』是只看了 B0 的 step record」）。

**caveats**:
- §407.19: 「双刃: 三轴上支配更难 ⇒ §5.3 负结论 a fortiori 成立, 但非支配也更容易满足 ⇒ 凡把非支配当信息的地方都要对着更宽的前沿读。tokens 是 check 不是 axis」。
- §408.4: 「types.py:446 明写设计意图是与 raw 并列报告, 而九个产物没一个读它」；§3.3（稿件章节）的跨 benchmark 跨度对比用的是 raw 值。
- §409.2: 「描述性结构不是检验, 承重的是跨格规律性 (分裂跟着 site 走)。canonical 版本在 4/6 格与 raw 恒等, 只有 B0 两格真的测了」。
- §407.20: 「功率近乎恒定 ⇒ co2e = 常数 x 时间 = latency 换单位。真正的问题是 NVML 静默回落 psutil」；B0 source=disabled。
- §418.3: 「真正大的是容器基线差 1.69×, 它**不威胁 claim 9**(格内比模式), 但『跟着站点走』这句话同时携带基础设施, 跨站延迟数字不可裸引」。
- §413: 「只能承载方法学主张『分母必须被声明』, 不能承载『X 比 Y 高效』。比值继承成功率噪声两次」；B2 两格最好模式仅 5 和 8 个成功被排除。
- §458.2: 「`tokens.input` 是**总输入**; `input_text`/`input_image` 拆分在 B0 上是 null」；「step 数不带 `runs` 列不可跨行比较」；「尾部风险**站点相关**」；只覆盖 B0。
- §463.3: 「**我未独立复核数字**。引用前应回原产物核对」。⚠️ 与 §408.4「每一格的最快模式完全相同」口径不同（容器时间分解 vs raw/canonical），见矛盾清单 #6。

**证据**: §407.15 / §407.19 / §407.20 / §408.4 / §409.2 / §413 / §418.3 / §458.2 / §463.3；`docs/analysis/cross_sites/multimetric_pareto.md`、`docs/analysis/cross_sites/per_mode_four_dimension_profile.md`、`docs/analysis/cross_sites/offsite_navigation_audit.md`、`docs/analysis/cross_sites/outcome_efficiency.md`、`docs/analysis/cross_sites/representation_deployment_profile.md`

**原文片段**: 「co2e~latency r = 0.9999 四格全部; power_watts 均值 66.3/66.7W, CV = 0.03」(§407.20)；「raw 与 canonical 下都是 3/6 格拓宽, frontier 成员完全相同」(§408.4)

---

## F2. 未消费字段 sweep

**当前值**:
- 全仓 `scripts/analysis/**/*.py` 字符串匹配: step **100/212** 未读 · summary **86/124** 未读（§408.4）；最重的一个是 `total_latency_canonical_ms`。
- 实际可用性（B1 六 mode 各 40 episode 采样，§412）: 0% nonzero: retry_count · screenshot_timeout_recovered · destructive_action_count · partial_recovery_step_count · unknown_failure_reasons；从不填充: tokens.thinking · text_fallback_used · tool_call_valid · network_retry_count · checklist_completion_rate · benchmark_noise_category；可用: state_digest.scroll_y_* (跨 mode spread **1.44x**) · runner_intervention_count (**3.5%** nonzero)。

**caveats**: 「『186 个未读字段』的绝大多数是 schema 里有、从不写的死字段 ⇒ claim 6 的『指标池只有我们选的那么宽』这个弱点, 真实宽度比看上去小」(§412)。

**证据**: §408.4 / §412；§G1 未消费字段 sweep（台账未给产物路径）

**原文片段**: 「step 100/212 未读 · summary 86/124 未读」(§408.4)

---

# G. B-1969 cls cron 死锁对 Phase 1a 的污染

**当前值（v2, canonical 白名单，§442.8；VWA classifieds Phase 1a canonical 18 cell，经 pass1_run_manifest.json 白名单）**:
canonical 18 run × 224 = **4032** episode 中 **77 (1.91%)** 记录 reset-time Page.goto timeout；timeout_events=**77** / retry_events=**77** / latency 条目=**154**；首次尝试 median **30.99s**，成功重试 median **19.11s**；recovered 全数。分层检验 **O=4 vs E=6.31**，20000 次 plus-one **p=0.2397**。drop-one 敏感性（cell-specific 反事实, 10000 次）最大 **p95** 偏移 B0 **0.45** / B1 **1.34** (vision) / B2 **0.45** pp。

**修复验证（HTTP 探测层，非 agent episode 指标，§442）**: 修前串行 6 次 3 次 12s 超时(000)；修后串行 10 次 0 失败(全 0.21s)，并发 8 请求总耗时 **0.48s**。

**演变**: §442 修复即时验证 → §442.7 v1 污染面与 hero 敏感性 → **§442.8 三家 /stress 推翻 v1** → v2。

**已作废（被 §442.8 推翻，禁引）**:
- §442.7 v1: 「18 run / **4257** episode / 88203 step」「78 个 episode (**1.83%**)」「156 次超时尝试」「SR=**5.13%** vs 同 task 未触发 **12.65%** (n=1138)」「单侧二项 **p=0.024**」「step 长尾率在每一个 step_idx 位置都是对照组的 2-3×」「分布 B0 0.45% / B1 2.23% / B2 3.05%」。推翻理由（台账 superseded_by 原文）: 「denominator 应为 canonical 4032 非 glob 4257; 『156 次超时』实为 77 超时+77 重试(154 条延迟记录); 12.65% 基线跨 cell 混合 = Simpson's paradox, 分层后 O=4 vs E=6.31 plus-one p=0.24 未识别出效应」。
- §442.7 v1 hero 敏感性「期望偏移中位 B0 0.00pp / B1 0.45pp / B2 0.45pp, p95 上界约 1.3pp」「极端上界 B2 2.68pp」—— 同被 §442.8 推翻（v2 数字见上）。

**caveats**（§442.8，一字不改）: 「⚠️ 1.91% 是**探测下界非发生率** —— reset_goto_timeout_count 只在 env.reset 记账, episode 中段撞窗口不可见; 644/4032 (15.97%) 含 >=1 个 env_step>=12s 但长延迟不特异于本缺陷, 故 644 是上界, 真值**不可分离**。⚠️ 77 个 flagged 里只有 **19** 个有 post-reset 长 step, 58 个没有 ⇒ 不能说『整个 episode 都在降级时段』。⚠️ 时间对齐永久不可得」；§442: 「即时验证只证明『当下不死锁』; 原故障是**周期性**的 … 需 3h soak」（soak 结果另记，本批无条目）。

**证据**: §442 / §442.7 / §442.8；`docs/reference/master_bug_catalog.md` B-1969、`scripts/analysis/scan_b1969_contamination.py (v2, seed 20260808)`

**原文片段**: 「canonical 18 run × 224 = **4032** episode 中 **77 (1.91%)** … 分层检验 O=4 vs E=6.31, 20000 次 plus-one p=0.2397」(§442.8)

---

# H. benchmark 语料事实

## H1. run/scored set、参考图、模板多样性、错别字、截断偏差、子集 gate

**当前值**:
- **六语料 run set / scored set（2026-08-09，§445.1；箭头 = 语料 → run set → scored set）**: VWA cls 234→224→**224** · VWA red 210→205→**203** (protocol 160,58) · VWA shop 466→435→**432** (protocol 463,465,345) · WA red 106→104→**104** · WA shop 192→173→**173** · WA shop_admin 182→176→**176**。§450.16 的 scored/total 同值: VWA-cls **224/234** · VWA-red **203/210** · VWA-shop **432/466** · WA-red **104/106** · WA-shop **173/192** · WA-shop-admin **176/182**。
- **参考图与难度标注（§445.3 / §450.16）**: **WA 三站 ref image 全部为 0** 且 `reasoning_difficulty` 为 None；VWA 三站 **29.1% / 40.0% / 36.3%**（§445.3）= ref-image **68/234 · 84/210 · 169/466**（§450.16）；指令长度中位数 **12–21** 词。
- **模板多样性（§445.3）**: distinct intent_template — VWA cls 234task/**75** · red 210/**87** · shop 466/**152**；WA red 106/**21** (5.0 task/模板) · shop 192/**49** · shop_admin 182/**41**。
- **语料错别字（§445.2）**: VWA shopping task 179 `reasoning_difficulty='hrad'` + VWA classifieds task 161 `visual_difficulty='mediun'`。
- **「取前 200」的系统性偏差（VWA shopping 435 run set，§447.2）**: 前 200 (task_id 序) 模板覆盖 **69/151 = 46%**，难度 easy **34%**/hard **26%**/medium 39%；全集 435: 151/151, 23%/32%/45%；分层 200: **151/151 = 100%**, 21%/33%/46%。
- **子集 run 与 paper-grade 分母冲突（§448.2）**: `queue_chain.sh:476` 写死 `SITE_EXPECTED_N[shopping]=435`；`:544` `ep != expected_n` 直接 FATAL；`paper_scored_task_count('shopping')` 恒返回 **432**；200 子集真实 scored = **198**。

**caveats**:
- §445.1: 「WA reddit 106→104 是 2 个 N/A (task **723 / 726**, 'Like/DisLike all submissions', fuzzy_match='N/A') **不是丢数据**」；「WA 三站 protocol 排除为 0 是**当前状态**, 不是永久属性」。
- §445.3: 「WA 零 ref image 是 **WA 作为 OOD 检验而非『第三个 VWA 站』的结构性理由**」；「模板数是『我们到底问了多少个不同问题』的诚实分母 —— 报 n=234 时真实多样性是 75」。
- §450.16: 「**ref image 缺失 = 任务规格的差异, 不自动等于『不需要视觉 grounding』** (攻击 A3 已迫使降级)」；错别字「**保持不修**并排除出难度统计」；「只有 scored set 用于任何比率」。
- §445.2: 「benchmark_eda.py **报告而不自动修正**」。
- §447.2: 「`max_tasks_per_site` **按文件顺序截断** … **会因为与模型无关的理由抬高 SR**」。
- §448.2: 「**代码缺陷仍在** —— 435 全集方案绕过了它, 但未来任何子集 run 都会撞。降级 backlog 而非已修」。
- 与 D2 的 §407.17 计分宇宙口径（cls 29.0% / red 38.9%）是不同分母，**不可互换**。

**证据**: §445.1 / §445.2 / §445.3 / §447.2 / §448.2 / §450.16；`docs/analysis/benchmark_eda/corpus_eda.{md,json}`、`configs/exp_v2_B1_shop_strat200_base.yaml`、`queue_chain.sh`、FIGURE_PLAN F4

**原文片段**: 「WA 三站 ref image 全部为 0, VWA 三站 29.1% / 40.0% / 36.3%」(§445.3)；「前 200 (task_id 序): 模板覆盖 69/151 = 46%」(§447.2)

---

# I. AWS proxy 与候选模型

## I1. 共享预算池与耗尽

**当前值**:
- **余额与燃烧率（AWS proxy i5xpracyci, DGX 与 A100 共用同一池，§444.1）**: budget_limit **$1000**，2026-08-09 00:00 已用 **997.93 (99.79%)**，剩 **$2.07**；实测燃烧 **$0.888/h** = **$0.06/episode**；反推 B0 全部项目历史正好耗尽这 $1000。
- **耗尽时响应 body（§448.3）**: `403 {"error":"Budget exceeded","usedUsd":999.9999863,"budgetUsd":1000}`。
- **B0_vision_shopping 撞墙时间线（§446.1）**: 01:17:56 完成 task 404 (**374/466** ep) → 01:19:45 task 405 撞 403 → 01:20:20 chain 写 done marker rc=1（后续 3 个 phantom condition 未启动）→ **02:03 max_tokens=1 探针仍报 ok, remaining $0.222** → 02:33 探针首次 quota:403 → ~08:40 人类才知道。

**caveats**:
- §444.1: 「燃烧率随 mode/site 变 — 0.888/h 是 shop vision (B0, 30 步/ep) 的实测值」；「B1/B2/B3 走本地 GPU **不占此池**」。
- §448.3: 「告警分类应 **body 关键词优先于状态码** … 403 body 能读出精确余额, 而 200-only 的 metadata 路径恰好在池子空时失明」。
- §446.1: 「**停止本身是设计如此** … 坏的不是 run 而是告知通道」。

**证据**: §444.1 / §446.1 / §448.3；`docs/checkpoints/probes/proxy_model_registry_20260809_000040.json`、`scripts/maintenance/proxy_budget_watch.py::probe + parse_exhausted_balance`

**原文片段**: 「2026-08-09 00:00 已用 997.93 (99.79%), 剩 $2.07」(§444.1)

---

## I2. registry 与 VL 通道可用性（「列出 ≠ 能调用」）

**当前值（最新快照，§471.3, 2026-08-19 registry 58 models；五关 = HTTP200 / tool_call / schema-valid / logprobs / 真读到图）**:
**`moonshotai.kimi-k2.5` ✓✓✓✓✓** (0.002/0.008 = 2×/1.6× B0) · **`nvidia.nemotron-nano-12b-v2` ✓✓✓✓✓** (0.00015/0.00025 = 0.15×/0.05×) · B0 控制 ✓✓✓✓✓；Claude 全族仍撞协议墙 `allOf`。

**演变**:
- §444.4 (2026-08-09): `GET /model-api/models` **57 模型 / 13 provider**；逐个打 19 个 → **13 可达**；**5 个 Anthropic 条目是死的**（sonnet-5 / opus-5 / opus-4-7 / opus-4-8 / fable-5 全部 400，均标 0.001/0.005 占位价）；haiku-4-5 与 sonnet-4-6 同价却真能调通。
- §456.3 (2026-08-12): **56 模型 / 12 provider**；三天内 **3 增 4 减 10 重定价**。
- §456.1 (2026-08-12, 224×224 双色块 PNG): 图像通道 SAW_IT **6/9**: qwen3-vl-235b(B0, +95 img tok, 1×) · sonnet-5 (+103, 3×) · sonnet-4-6 (+97, 3×) · opus-5 (+103, 15×) · haiku-4-5 (+97, 1× 严格同价) · gemma-3-27b-it (+288, 0.3×)；IMAGE_BILLED_BUT_MISREAD 2 个（gemma-4-31b / gemma-4-26b-a4b, +578 tok, 答 'White'）；HTTP 400 1 个 (zai.glm-5)；探针总计费 **$0.005533**。
- §456.3: 两个意外候选 —— `eu.anthropic.claude-haiku-4-5-20251001-v1:0` = **0.001/0.005 与 B0 严格同价**且图像通道通；`google.gemma-3-27b-it` = **0.3×** B0 价格、B2 同族 **6.75 倍参数**版。
- §468 (2026-08-16, 59 models / 12 providers, 生产形状 payload): 四关全过 `qwen.qwen3-vl-235b-a22b` (B0) · `moonshotai.kimi-k2.5` (2× B0 价) · `nvidia.nemotron-nano-12b-v2` (0.15×)；**协议墙**: Anthropic 全族 (`allOf` 被拒) + Amazon Nova 全族；**权限墙**: `global.openai.gpt-5.6-{sol,terra,luna}`（与 B0 严格同价 0.001/0.005）→ 403 未授权；**能力不够**: gpt-oss-120b/20b · nemotron-3-super-120b · voxtral-small · qwen3-235b-2507；`gemma-3-27b-it`/`gemma-3-12b-it` HTTP 200 但**看不到图**；**Kimi K3 在 AWS registry 上不存在**。

**已作废**:
- §456.1 / §456.3「haiku-4-5 = 与 B0 严格同价的可用对照」→ 被 **§468** 否定（「**同属 Anthropic, 撞同一堵墙**」）。
- §456.1 / §456.3「gemma-3-27b-it 图像通道通」→ 在 **§468** 生产形状 payload 下「HTTP 200 但**看不到图**」（两次探针 payload 不同：224×224 双色块 vs 320×160 带 `[7]` 标注图；并列）。

**caveats**:
- §444.4: 「**价格推不出可用性, 加新模型前必须逐个打一发**」；「B0 是 `qwen.qwen3-vl-235b-a22b` (0.001/0.005) **不是** `qwen.qwen3-235b-a22b-2507` (0.0008/0.004)」；可达 != VL。
- §456.1: 「判据必须**颜色答案 + input-token 增量双条件**」；「同一张图各家 tokenizer 差 **6 倍** ⇒ image token 非常数」；「单张合成图不等于 VWA 真截图」。
- §456.3: 「registry 是**时变**的 … 任何『可用模型清单』或『预算表』的引用都必须带快照日期」；gemma-3-27b-it 是 proxy-served 而 B2 是本地 bf16 ⇒ 不可直接相减（参见 §302）。
- §468: 「403 与 400 含义不同: **GPT-5.6 是订阅没开, 可以去要**」；「registry 显示名会漂 … **以 modelId 为准**」；Kimi K2.5 未验多步 agent 表现。
- §471.3: 「只验**单次生产形状调用**」；「Nemotron 是 12B nano, SR 可能低到 d 不足 … 便宜不等于有用」；「**B4 smoke 不必再跑**」。

**证据**: §444.4 / §456.1 / §456.3 / §468 / §471.3；`docs/checkpoints/probes/proxy_model_registry_20260809_000040.json`、`docs/checkpoints/probes/proxy_vision_channel_20260812_221840.json`、`docs/checkpoints/probes/proxy_model_registry_20260812_221116.json`、`scripts/maintenance/probe_proxy_vision_channel.py`、`scripts/maintenance/probe_model_five_gates.py`

**原文片段**: 「**5 个 Anthropic 条目是死的** … 而它们**都标 0.001/0.005**(未接通条目的占位价)」(§444.4)；「此前记的 `haiku-4-5`『与 B0 严格同价』候选**同属 Anthropic, 撞同一堵墙**」(§468)

> §475 之后见 measured_D6.md

---

## I3. GPT-5.6 / B5 接入路径

**当前值**:
- **tools 路不通（`global.openai.gpt-5.6-luna`，§471.1）**: 纯文本 **200** · `reasoning_effort:"none"` 单独 **200** · `+logprobs` **400** `unsupported_parameter` · `+tools` **400** `Function tools with reasoning_effort are not supported` · `+tools+reasoning_effort:"none"` **仍 400**。
- **response_format 路（§471.5 luna / §472.8 terra）**: `json_schema` 非 strict → **200** · schema_valid=True · saw_image=True · confidence=**0.99**；`strict:True` → **200 但返回空串**；`json_object` → **400**；terra 五种变体全部 **200**；proxy 丢弃未知顶层字段（`reasoning_effort:"ZZZ_INVALID"` 与 `totally_bogus_param_xyz:123` 均返 200）。
- **每次返回的顶层 JSON 对象数（terra，§472.9）**: **6 / 3 / 5**（三次）；加「Emit exactly ONE JSON object ... do not plan ahead」后 **1 / 1 / 1** (3/3)。
- **backend→agent 白名单丢键（§472.9）**: **3 个** `structured_output` · `logprobs_unavailable` · `image_format` —— yaml 写了等于没写，无报错 (B-1985)。
- **parse_valid 率（B5 × classifieds × dom，§472.11）**: 探针 **3/3 = 100%** · smoke 单 episode **7/9 = 78%** · **cell A1 前 6 episode(75 步) 74/75 = 98.7%**；修复前 **0/3**；残留失败全部是 `multiple_actions`。
- **成本外推（cell A1 前 6 episode，§472.11）**: 6 episode **$0.7606** ⇒ 224 ep 外推 **$28.39**（intent 预算 ~$32）。

**caveats**:
- §471.1: 「**两堵墙性质不同**: tools 是 **proxy 转发层** … **可以去要**; logprobs 是**模型侧**不支持, 大概率无解」；「GPT 只能当**无 confidence 信号的降级 baseline**」。
- §471.5: 「logprobs 在两条路上都拿不到 ⇒ P79 confidence schema 6 字段只剩 verbalized, 须在 paper §3.5 disclose」；单次调用验证。
- §472.8: 「**排除了** tier 差异作为 B5 smoke 400 的原因, 但**没有定位**真因 (B-1984)」；「**luna 把 JSON 吐了两遍拼接**, terra 只吐一遍」；`probe_model_five_gates.py` 同日控制组失败（qwen `✓✓✗✓✓`）⇒ 该次输出不可当证据。
- §472.9: 「**`response_format` 约束的是每个对象的形状, 不是对象的个数**」；「解析器的 `multiple_actions` 作废策略 (B-409/P1-3-B*) **是对的**, 不要为此放宽它」；B-1985「**写来抓这件事的守卫被同一动作解除武装** … **形状是复发**: B-340 …」，修法是不变量测试 `tests/test_b1985_model_cfg_forwarding.py`。
- §472.11: 「**探针能证明机制可行, 证明不了比率**」；98.7% 是运行中前 6 个 episode，全量 224 落地后须复算；「决定**不加 take-first 兜底**, 1.3% 按 B5 的 format 失败率如实报」；成本「**外推不是实测**」、「价格随时会漂 (§471.2 记错过 6 倍)」。

**证据**: §471.1 / §471.5 / §472.8 / §472.9 / §472.11；`scripts/maintenance/probe_model_five_gates.py`、`p79/agents/proxy_api_agent.py`、`tests/test_b1985_model_cfg_forwarding.py`、`results/visualwebarena/phase1/B5_dom_classifieds_20260820_202158_076182888_2491046_R29736`、B-1984 / B-1985 / B-1986

**原文片段**: 「**cell A1 前 6 episode(75 步) 74/75 = 98.7%**。修复前 0/3」(§472.11)

> §475 之后见 measured_D6.md

---

## I4. proxy 响应形状漂移与 parallel_tool_calls

**当前值**:
- **响应形状漂移（qwen.qwen3-vl-235b-a22b, 4 种 payload 变体，§466.2）**: `body["content"]` 由 `str` 变为 Anthropic block list；**新增**顶层 `body["text"]` 与 block 文本逐字节相同；`body["tool_calls"]` / `body["logprobs"]` 不变 ⇒ 纯表征变化，零信息损失 (B-1970)。
- **`parallel_tool_calls` 字段（三臂对照, T=0, 2026-08-17，§469.2）**: false / true / 不带 三臂全部 HTTP 200 且都返回 **2 个 web_action** ⇒ proxy 收下该 key 然后丢弃。

**caveats**:
- §466.2: 「漂移窗口只能定到 **08-10 (B0 上次真跑) ~ 08-16**」；「归档 B0 数据与此后新 B0 数据**不在同一 provider 快照**上」；「不要把『这次无损』外推成『漂移总是无损』」；只测了 B0 那个 modelId。
- §469.2: 「这个 prompt 是**故意写来诱发**并行发射的, 因此**不能**当作真实 VWA 任务下的发生率」；「旧代码静默取 [0], 不留痕迹, 该问题在归档数据上永久不可观测」。

**证据**: §466.2 / §469.2；`docs/reference/master_bug_catalog.md#B-1970`、`docs/checkpoints/probes/parallel_tool_calls_20260817.md`

**原文片段**: 「false / true / 不带 三臂全部 HTTP 200 且都返回 **2 个 web_action** ⇒ proxy 收下该 key 然后丢弃」(§469.2)

---

# J. 吞吐 / 单价 / ETA

**当前值**:
- **episode 吞吐（从 canonical run 的 wallclock 极差，§468.10）**: **B0 × classifieds 25.9 ep/h (8.6h/格)** · **B1 × classifieds 11.3 ep/h (19.8h/格)** · **B0 × reddit 仅 4.6 ep/h (44h/格)** ⇒ B0 reddit 比 classifieds **慢 5.6 倍**；B0 reddit 历史极端值 `phantom_prompt` **154.5h**、`phantom_som` **184.4h**，干净的几个 **5.0-7.2 ep/h**。
- **其他吞吐点（A100，§449.2）**: B0 VWA shop **7.5-14.0 ep/h** · WA reddit **9.8-12.2 ep/h**；B1 VWA shop **8.0 ep/h** · cls **10.6-11.0** · red vision **6.5**。§451: B1 shopping dom 实测 **6.8 ep/h**（08-10 12:03Z 162 个 → 08-11 01:05Z 251 个），低于早先 8.6 ep/h ⇒ B1 shop 三格 ETA **~6.3 天 (约 08-17)**。
- **B1 per-episode 与 Magento reset（§446 / §447.1）**: Magento reset **~33 分钟**（08:15:45 → 08:48:50）+ 模型加载 ~2 分钟；B1 **9.3-10.8 min/episode**（reddit 四个 run 205 ep 各 31.6/35.5/36.4/36.7h）；shop 435 ep ⇒ **76h/condition ≈ 3.2 天**，六格 ≈ **19 天**。
- **B0 VWA shopping per-mode 单价（condition_summary 实账，§449.1）**: dom **$0.1198** / som **$0.0979** / vision **$0.0722**（各 435、435、374 个 episode）。
- **成本主导因子（§449.1）**: 站点间远大于 mode 间 —— cls 六个 mode 全在 **$.065-.072**（极差 **<11%**），red 全在 **$.098-.110**（**<12%**），cls→shop 差 **73%** ⇒ 成本由**步数**驱动。
- **补尾实际花费（§449.7）**: B0 vision shopping 补尾 61 ep 实际 **$5.59** (avg **$0.0916**/ep) vs 事前估 **$4.4** (avg $0.0722) —— **偏高 27%**；condition 全程合计 **$32.51** / 434 个计费 episode；**B0_vision_shopping = 435/435 闭合**，B1_dom_shopping resume 91→92。

**已作废**:
- 「B0 ≈ B1 ≈ 11 ep/h」⇒ B0 cls 三格 2.6 天 → 被 **§468.10** 推翻（实际 1.1 天，「估错 2.4 倍」）。
- B1 shop 8.6 ep/h / ETA ~5.6 天 → **§451** 修正为 6.8 ep/h / ~6.3 天。

**caveats**:
- §468.10: 「B0 是 API 推理, B1 是本地 4B 占 GPU, **两者不能互相外推**」；「B0 red 那两个 150h+ 的值可能含 stall/restart」；「reddit 是**双重贵**: token $22.10 vs $16.14 **且** wall-clock 5.6×」；未测 B2 与 WA 站。
- §449.2: 「B0 dom shopping 那条 `58.1h / 7.5 ep/h` 是**异常值** —— 它跑在 2026-08-04, 早于 B-1969 修复」。
- §446 / §447.1: 「**不要拿 cls/red 的 chain 节奏外推 shop**」；并行不可行（同 site 共享 docker + 同一 emma.lopez 账号）。
- §449.1: phantom 三个 mode **未实测**，外推为 ~$0.120/~$0.119/~$0.097，「要报 shop phantom 成本必须标外推」；「论文 'P-SoM cost ≈ DOM' … **不是因为文本更短, 是因为步数相近**」。
- §449.7 / §451: 「**用已完成部分的均价估剩余部分会低估** —— task 按 id 顺序跑, **剩余的不是随机样本**」；「**单点测速外推整条 chain 会偏**」；VWA shop 仍缺 3 个 phantom（约 **$146**），within-condition 状态不一致（前 374 / 后 61 跨 reset）。

**证据**: §446 / §447.1 / §449.1 / §449.2 / §449.7 / §451 / §468.10；`logs/queue_chain_b1_shop.log`、`docs/analysis/cross_sites/cost_per_mode.md`、chain `queue_chain_b0vis_b1shop_20260809.log`

**原文片段**: 「**站点间远大于 mode 间** … 而 cls→shop 差 **73%**」(§449.1)；「B0 在 reddit 上比 classifieds **慢 5.6 倍**」(§468.10)

---

# K. A100 运维与溯源（reset / git / 时区 / 并发 / stash）

**当前值**:
- **RESET_BEFORE 历史（A100 `logs/queue_chain_*.log` 全量 grep，§451.6）**: **13/13 全部 `RESET_BEFORE=1`**；「**项目里从来没有过 RESET_BEFORE=0 的 chain**」。§449.3: 2026-08-09 B0 vision shopping resume 374→435 用的 RESET_BEFORE=1 是 `queue_chain.sh:8` **默认值**，不是裁定。
- **容器共享（`docker ps`，§449.4）**: `vwa-shopping` 同时暴露 **7770 和 7780** ⇒ VWA shopping / WA shopping / WA shopping_admin **共用一个 Magento 实例**；代码层锁已是容器级（`.locks/p79_magento.lock`）。
- **时区（§449.6）**: DGX `spark-9ea3` = **BST (+0100)** · Condenser A100 = **UTC (+0000)** ⇒ 相差 **1 小时**。
- **A100 git 状态**: §446.5 (2026-08-09) `git status` **81 文件 / +7726 −1110** 未提交，HEAD `10f2569`；§471.7 (2026-08-19) HEAD `master@10f2569`（落后工作分支 **70 个 commit**），**84 files / 8195 insertions** 未提交，含 fire code `p79/agents/proxy_api_agent.py`；逐项核完「**没有该机器独有的工作**」。
- **A100 推送能力（§471.8）**: `user.name` **UNSET** · `credential.helper` **UNSET** · `git ls-remote` OK ⇒ **A100 不能 commit/push**。
- **SHA 缩写长度（§472.5）**: DGX **7** 位 · A100 **8** 位（同一 commit `3e200f3` / `3e200f39`）⇒ B-1983: bootstrap SHA pin 字符串比较「该闸自诞生起从未通过」。
- **只存在于 A100 工作树的证据绑定（§472.7）**: **6 条** shopping 绑定；已在 DGX 复现并提交；bound-clean **36 → 42**，unbound singleton **6 → 0**。
- **被 `git stash push -u` 吞掉的 quarantine 事件（§472.13）**: **9 条**（08-16 → 08-20），其中 **3 条**带完整论证的人工裁定，其余 6 条 quarantine 事件；已全部从 `stash@{1}` / `stash@{3}` 捞回提交 (B-1988)。
- **2026-08-16 自动开火并发（§466.1）**: `_b4_wa_watcher.sh` 与 `wa_chain_autolaunch.sh` 各自独立开火，一度并存 VWA-cls + WA-shop 两条 site chain；WA chain 15:26 因 B-1970 自杀，未污染数据；再发火风险 = **零**（对当前三个已解除的 watcher）。

**已作废**: §449.3「默认值恰好正确」→ **2026-08-11 推翻**（台账原文: 「canonical 组合是 **RESET_BEFORE=1 + FORCE_NEW=1**; chain 的 `FORCE_NEW:-0` 默认让它变成 **reset 容器 + 断点续跑**, 恰好制造 §280 要避免的 confound 镜像」；实证容器换代 B0 vision `b36ed7a7cccb→e338cc270773` / B1 dom `e338cc270773→f6f5c7016a3a`）。

**caveats**:
- §451.6: 「**区分开 FORCE_NEW** —— 那个才有 site 分歧 (cls/shop=1, reddit 破例 0, §352.3) 且**取决于启动路径**」。
- §449.4: 「『WA 不能和 VWA 混跑』的理由**不只是 latency 污染**, 更硬的是同容器同账号的 server-side session race + 购物车交叉污染 (hard rule #1)」。
- §449.6: 「夏令时结束后差值会变 0, **不要把 '差 1 小时' 硬记成常量, 要现打 date**」。
- §446.5 / §471.7: 「**A100 没有可用的 git 溯源** … **不能信它的 HEAD**」；「『这次没丢东西』不等于『这个状态是安全的』—— 它意味着此前每一次 fire 的代码版本都无法从 git 复原」。
- §471.8: 「fire 跑在 A100 而收尾跑在 DGX, 这个不对称是结构性的」。
- §472.5: 「更要紧的是它**制造了自己的误诊**」；修复 = `rev-parse --verify "<rev>^{commit}"` 归一到 40 位。
- §472.7 / §472.13: 「靠 03:33Z watchdog 重新绑回来才没丢 —— **这次是运气**」；「**只盘了当前还在的 4 个 stash** … 9 是**下界不是总数**」；「修复 (复制到 repo 外 + ntfy) **没有消除根因**」。
- §466.1: 「『零风险』只对**当前已解除的这三个**成立」；「锁是按容器分的 … 单 chain 规则是锁之上的**策略层**, 锁本身不阻止跨站并发」。

**证据**: §446.5 / §449.3 / §449.4 / §449.6 / §451.6 / §466.1 / §471.7 / §471.8 / §472.5 / §472.7 / §472.13；`scripts/queues/_reframe_bootstrap.sh`、`scripts/maintenance/reframe_finalize_poller.sh`、`docs/checkpoints/pre_run/fire_manifest.json`、`docs/checkpoints/quarantine_registry.jsonl`、`_lib_paper_grade_gates.sh:594`

**原文片段**: 「**13/13 全部 `RESET_BEFORE=1`** … **项目里从来没有过 RESET_BEFORE=0 的 chain**」(§451.6)；「`user.name` **UNSET** · `credential.helper` **UNSET** … ⇒ **A100 不能 commit/push**」(§471.8)

> §475 之后见 measured_D6.md

---

# L. 文稿工程

## L1. 证据链、交叉验证、引文、测试基线

**当前值**:
- **主证据的版本控制（§450.6）**: ① `results/phantom_paper/*` 被 **.gitignore:35 整个排除**，C1 主证据 `phase1_full_prereg_decision.json` 与 C2 主证据 `meta_phantom_lift.csv` 都在里面 ② `meta_phantom_lift.csv` mtime **05-17**，比其余分析产物老近三个月，支撑 drop-one **1.7–3.3pp**（C2 全部）。
- **图脚本独立解析 vs 产物散文（§450.17 累计）**: **7/7 全中**（F13 `0/8` · F8 `2.00–7.59pp` · F10b `2/2 在带内` + `3.82–4.15pp` · F4 `2 个 corpus 错别字` · F11 `5/6 格下降` + `reasoning_difficulty 5/6 最强` · F12 `1/8 存活` (Holm m=8 独立重算) · F1 dom/vision 截图 `md5 一致`）。§450.13 首轮 **3/3**。
- **GPT 搜索端引文核验（33 arXiv ID + 27 DOI，§450.9）**: **实质错误仅 1 处**（`arXiv:2502.11027` 标题错 + 与 2007 年 DOI 缝在一起）；Holm 1979 只有 JSTOR stable `4615733`（非编造）；P2 的 27 个 arXiv ID 全 clean。
- **测试基线（DGX, 2026-08-10，§450.8）**: `pytest -k 'router or triage or learnab'` = **3 failed / 115 passed / 1 skipped**；stash 后仍同样 ⇒ 预先存在。

**caveats**:
- §450.6: 「复现包必须**显式打包**这两个文件」；「**定稿前必核** meta_phantom_lift.csv 是否还对应当前 42-condition 口径」；「mtime 老 ≠ 数字错, 但**在没核之前也不能当它对**」。
- §450.13 / §450.17: 「只验证**解析路径与散文路径一致**, **不验证底层分析正确**」；「对硬编码散文这套检查会直接报不一致 —— 那才是它的价值」。
- §450.9: 「**我的核验脚本先制造了一批假错误** … ① 正则**必须锚 `arXiv:` 前缀** … ② **12 处初判 MISMATCH 里只有 1 处是真的**」；「真正危险的失效模式**不是编造 ID** 而是**真 ID 配错标题**」；未核文献是否支持方法学论点。
- §450.8 测试: 三个预先失败 `test_universe_consumption_lint::test_triage_list_is_a_ratchet` · `test_router_prior_baselines::test_literature_inventory_covers_named_router_families_with_sources` · `test_stress_a2_1_phantom_framing::test_section1_oracle_vs_realized_router_separation`；「**这三条本身是待修项** … ratchet 测试正是防止 universe 记账退化的」。

**证据**: §450.6 / §450.8 / §450.9 / §450.13 / §450.17；FIGURE_PLAN、CLAIM_EVIDENCE_MATRIX、VERIFICATION.md

**原文片段**: 「`results/phantom_paper/*` 被 **.gitignore:35 整个排除**」(§450.6)；「**12 处初判 MISMATCH 里只有 1 处是真的**」(§450.9)

---

## L2. 毕设可读性、deslop 与收尾审计

**当前值**:
- **deslop 前后（final_dissertation/tex 全 12 文件，§454.2）**: vale error **195 → 0**；散文 em-dash **222 → 0**；warning **108 → 104**；invariant gate **11/12 PASS**。
- **可读性双指标（7 章 + 4 附录, 改前快照，§474.1）**: 句长中位数 **15-22** 词 / 被动语态 **11-22%** / >45 词长句 **0-5%** ⇒ 句法层健康；outsider 名词密度 **4.3-13.7 个/100 词**（ch4 11.4 · 附录A 13.7 最高 · d_reproducibility 3.3 最低）。
- **vocabulary bill（codex 两轮零预设冷读，§474.2）**: **97** 个 — must-explain **55** / could-drop **32** / fine-in-context **10**；卡点 **258** (轮A) + **253** (轮B) 条。
- **de-jargon 改写后（§474.6）**: ch1 **7.9→6.1 (-23%)** · ch5 9.8→9.4 · 附录A 13.7→12.0 · ch3 7.6→7.5 / ch4 11.4→11.7 / ch6 9.4→9.3 / ch7 7.8→7.6；ch1 >45w 长句 **3%→6%**（已回头再拆 7 句）。
- **收尾 /stress（milestone, Stats methodologist，§474.9）**: **20 条**有效 findings = 3×P0 + 10×P1 + 3×P2 + 4 条并入；A 7 · B 7 · C 6；2-AI 重合 3 条；1-AI unique 12 条全部通过 Phase 4b 验真；丢弃 1 条 gemini 幻觉；「**20 条里没有一条是原稿固有缺陷, 全部是当天改写引入的**」。

**caveats**:
- §454.2: frontmatter 那 1 个 FAIL 全是**新增**；warning 只降 4 是**有意**（ContrastiveFormulas 8 条全留）。
- §474.1: 「jargon 词表是我 curate 的 (~70 条正则), 不是标准量表, 跨项目不可比」；prose 提取排除 tabular ⇒ 系统性低估表格密集章。
- §474.2: 两轮 Reader's summary test **都能填满 5 句** ⇒ 主线立得住，塌的是方法层；5 条卡点是 detex 损耗假阳性。
- §474.6: 「**这个指标衡量不了本轮大部分工作** … **并且指标抓到了退步** … 降 jargon 与降句长会互相拉扯」。
- §474.9: Mode C 首轮 51s/3 findings 不达标，retry 后 150s（略低于 180s 下限，判 marginal PASS）；只审改写是否改变 claim，不审原方法学。

**证据**: §454.2 / §474.1 / §474.2 / §474.6 / §474.9；commit 31515df、`docs/checkpoints/codex_outputs/thesis_outsider_readability_B_2026-08-21.md`、`docs/checkpoints/codex_outputs/dejargon_audit_B_2026-08-21_220310.md`

**原文片段**: 「**20 条里没有一条是原稿固有缺陷, 全部是当天改写引入的**」(§474.9)

> §475 之后见 measured_D6.md

---

## L3. VLM4RWD 移植与投稿

**当前值**:
- **wrapfigure 静默裁 caption（§473.1）**: `4_upperbound.tex` Figure 3 caption 四句，PDF 只渲染到第二句中间；丢失三段（含核心主张 'Most of the apparent headroom is not attributable to representation diversity'）在 38 页 PDF 中**零命中**。
- **ACL→NeurIPS 容量落差（§473.1）**: ACL 双栏正好 8 页（5674 词 + 4 图）→ 直接移植 **10 页**；三张图被放大 **1.8×**，缩回 0.62-0.66 后 10→9 页。
- **REALM `tab02.tex` caption 错误（§473.4）**: 三处 —— 引 `tab:t04` 而正主是 `tab:t05`；称 'non-separability result' 而 `tab05` 明写相反；阈值写 ≥83% 而实际 ≥7/8 = **87.5%**；VLM4RWD 副本已修，**REALM 在审版本未修**。
- **三家审阅命中（§473.6）**: Claude 7 / 3 OOB · codex 9 / 4 first-read-miss · gemini 6 / 2 first-read-miss；3-AI 重合仅 **1 条**；**最严重一条(caption 丢失)是 codex 1-AI 独占**；codex 9/9 属实 0 幻觉，gemini 4 条属实 + 1 条误判。
- **投稿参数（OpenReview 表单实测 2026-08-21，§473.7）**: deadline **2026-08-31 05:00**（官网 CFP 写 08-30）· notif 2026-09-29 · License **固定 CC BY 4.0** · TL;DR 上限 **250 字符** · 正文 8 页 · 双盲。
- **ghostscript 清 `/PTEX.FileName`（fig_overview.pdf，§473.8）**: 泄漏 1→**0**；1.79% 像素有差异，平均绝对差 **0.027/255**，差值>8 仅 **277 像素(0.016%)**；9 个字体仍嵌入；839KB→395KB。
- **Paper Checklist（§473.8）**: 16 项 **8 Yes / 6 NA / 2 No**（No = Open access to data and code / Experiments compute resources）。

**caveats**:
- §473.1: 「**丢失不报 warning** … 这类 bug 结构上不可能被『编译干净+页数对』发现, 因为页数对恰恰是字被吃掉的结果」；是否触发取决于 caption 长度；「少三到四成」是本稿实测不是通用换算。
- §473.4: 「**『在别处写下更正』不等于修好**。因标签 tab:t04 确实存在, LaTeX 不报 undefined」。
- §473.6: 「若按 Phase 4 的 30% 抽样极可能漏掉, 这是 **v7.10 Phase 4b『unique findings 全量核验』规则的一次正面验证**」；gemini 误判因 agy 只能读 inline 纯文本。
- §473.7: 「**官网 CFP 与 OpenReview 表单不一致时以表单为准**」；CC BY 4.0 **不可逆**。
- §473.8: 「**`mutool clean -ggg` 对此无效**」；gs 副作用 2 个 Type 1C 字体丢 ToUnicode map；「两处 No 是**诚实口径** … **编造 Yes 才危险**」。

**证据**: §473.1 / §473.4 / §473.6 / §473.7 / §473.8；`deliverables/vlm4rwd/README.md`、`deliverables/vlm4rwd/tables/tab02.tex`、`deliverables/vlm4rwd/figures/fig_overview.pdf`、`deliverables/vlm4rwd/checklist.tex`、`docs/checkpoints/codex_outputs/vlm4rwd_template_port_FINAL_2026-08-21_105414.md`

**原文片段**: 「丢失三段(含核心主张 …)在整个 38 页 PDF 中**零命中**」(§473.1)

> §475 之后见 measured_D6.md

---

# M. 元层与外部对照

## M1. 台账 / 结论层 / retracted 的规模

**当前值**:
- **Phase 0 台账（覆盖 §1–§397.10，§398.1）**: **2033** 条（MEASURED 905 / ADJUDICATED 831 / RETRACTED 156 / CLAIM_UNVERIFIED 92 / DATA 49）；1097 条带可核数字，**1093 = 99.6%** 可追溯。
- **结论层规模（§398.3）**: **253 节 / 9377 行** / 五批合计 **~591K token**；实测中文 **1.6 字符/token**。
- **作废归纳（§398.4）**: 156 条作废归纳为 **11 类 (M1–M11)**；其中 **6 类**原文自己写下过教训然后复发。

**已作废**: §398.1 首版核验器报 **92.3%** 可追溯 → 低估（只查被引 § 正文，漏指针型 §），以 99.6% 为准；§398.3 先前按 **3.5 字符/token** 估算，「错 2.2×」。

**caveats**: 4 条不可追溯全是 artifact 文件名时间戳；「253 节 / 9377 行」一句在 INDEX.md 与 REBUILD_PLAN.md，不在笔记 §398.3 正文；§231 最锋利（「我读过该 memory 仍犯」）。

**证据**: §398.1 / §398.3 / §398.4；`docs/reference/known/ledger.jsonl`、`docs/reference/known/conclusions/INDEX.md`、`docs/checkpoints/REBUILD_PLAN.md`、`docs/reference/known/conclusions/retracted.md`

**原文片段**: 「1097 条带可核数字, 1093 = 99.6% 可追溯」(§398.1)

---

## M2. MAG (arXiv 2607.10079 v3) 的可路由增量

**当前值（MAG 论文 §9.1, SoM=23 / coord=14 / overlap=8，§398.7）**: H_route = 29−23 = **6 tasks = 3.4pp**；论文正文只给 union **16.7%**（读起来大 5 倍）。

**caveats**: 「作者自承 ±2 point ⇒ 3.4pp 对 ±2 = 1.7×, 与 P79 的 1.35/2.09 vs 4.9-7.6pp 同构」；该文已做三-judge 交叉验证（435 verdicts, 一致率 94.9%）⇒ evaluator variation 不能再当攻击点。

**证据**: §398.7；`docs/reference/Web_Agent_MAG_Research_Agent_Note_Optimized.md`

**原文片段**: 「H_route = 29−23 = 6 tasks = 3.4pp; 论文正文只给 union 16.7% (读起来大 5 倍)」(§398.7)

---

# ⚠️ 矛盾清单

> 一律并列，不选边。

1. **triage / abstention AUROC 的新旧数字混用**
   - §450.8 已把 red·B2 **0.483/0.711 'below chance'** 与区间 **0.651–0.717** 判为过时（当前 0.615–0.864 / 0.526–0.758）。
   - 但更晚的 **§457.2** 仍写「与 §6 which-mode router (AUROC 0.651-0.717 in 5/6, 第 6 格 red·B2 = 0.483 **低于随机**) 的差别」—— 既引了已作废的数，又把 triage 产物称作 which-mode router；而 §457.2 自己的六个 abstention AUROC 与 §450.2 的 20 特征 triage AUROC **逐格相同**。
   - **§412** 的 visual_difficulty 增量以 red_B2 「0.48→0.53」为基线，同属修正前旧值。→ 见 C2 / C3 / C5。

2. **「always-cheapest 是成本下界」**
   - §401.1 证伪：Vision 在 **47.3-70.9%** 的 task 上不是最便宜，per-task cost oracle 便宜 **22.2-46.2%**。
   - §450.12 又写「always-cheapest 是**成本下界**(所有任务都送最便宜 mode), 任何保住 SR 的策略必然更贵」作为「0/8 部分是对照选择的产物」的理由。
   - 两者口径可能不同（per-mode 均值 vs per-task），但 §450.12 原文未声明。→ 见 C1 / C2。

3. **B1·WA-red 一臂边际 vs 重跑带**
   - §406: +4.81pp vs 重跑 **+2.00 ~ +4.00pp** ⇒「高出带子 0.81pp」。
   - §450.10: 同一格重跑 **0.00–10.00pp** ⇒ indistinguishable，且指出 WA 重跑行「不是 dom 的地板」。
   - §407.3 也仍按 2.00-4.00pp 地板读 WA（+ptext/+pprompt「出地板」）。→ 见 A4。

4. **过噪声门槛的两个版本**
   - §450.10: 正态 one-sided 95% **3.82–4.15pp**（§450.18 用它判 36 个 drop-one 值）。
   - §452.1: 另报精确 **4.02–4.46pp**。
   - 两者都只从 B0·VWA-cls 一格导出；§450.18 把它套到 6 cells 全部值上。→ 见 A5。

5. **fusion 「唯一显著区间为负」**
   - §418.2: 泄漏置零后 red_B2 SoM−DOM **−1.48pp [−3.45,+0.49] 跨零**。
   - §463.3（Kimi，未独立复核）: 「per-cell: 0/8 clear, 唯一显著区间为负」—— 未说明用的是泄漏前还是泄漏后的产物版本。→ 见 D1。

6. **最快表征是否稳健**
   - §408.4: raw vs canonical 下「每一格的最快模式完全相同」。
   - §463.3（Kimi，未独立复核）: 去掉容器时间后最快表征排名在 **4/8 cell 翻转**。
   - 不同 estimand（canonical 只扣 retry/busy-wait/screenshot；Kimi 扣容器时间），未调和。→ 见 F1。

7. **B1 是否确定性**
   - §470.7:「一刀切二分: B0 六臂 10.27-14.29% vs B1 两臂 0.00%」。
   - §472.3: B1.cls.dom **3.12%**；§406: B1 × WA-reddit **6.00%**（含环境漂移）。→ 见 A2。

8. **leak-kept vs leak-corrected 分母**
   - §462.1: `per_task_sr.csv` 是 leak-kept + n=205，canonical 是 leak-corrected + n=203；§455.3 / §457 / §467.2 / §468.5 的成本列继承 leak-kept。
   - §465.1: 影响面只有 red_B2 一格，「需重跑的产物为零」。但 §462.1 同时要求「camera-ready 前 0/36 与 0/8 分析必须在 primary 政策下重算」—— 两条处置口径并列。→ 见 E4。

9. **参考图占比的三套分母**
   - §406 / §445.3 / §450.16: 语料口径 VWA red **84/210 = 40.0%**（cls 68/234 = 29.1%, shop 169/466 = 36.3%）。
   - §407.17: 计分宇宙口径 cls **65/224 = 29.0%** · red **79/203 = 38.9%** · 合计 **432/1281 = 33.7%**。→ 见 D2 / H1。

10. **gemma-3-27b-it 图像通道**
    - §456.1 / §456.3: SAW_IT (+288, 0.3×)，「图像通道通」。
    - §468: 生产形状 payload 下「HTTP 200 但**看不到图**」。→ 见 I2。

---

# 对旧结论层的 supersede

| 旧 § | 新 § | 一句话 |
|---|---|---|
| §397.10(3)（D4 D8 / 矛盾清单 #1：manifest 「15 组」有 2 个 run） | §398.2 | 实测为 **19 组 ≥2-run (非 15)**，且多数第二个 run 目录不在磁盘；D4 矛盾清单 #1 由此闭合。 |
| §397.10(3) / D4 A3 caveat（「B0-MoE 上界、不可外推到本地确定性 backbone、真正需要的是本地格同测量而我们没跑」） | §406 / §470.5 / §470.7 / §472.3 | 本地格地板已测：B1·cls som 0.00% / vision 0.00% / dom 3.12%，B1·WA-red 6.00%（含漂移）；缺口被部分补上。 |
| §394.1 / §387.16.4（D4 B10：red·B2 AUROC **0.483**、最强单协变量 **0.711**、5/6 cell **0.65-0.72**、「唯一 AUROC < 0.5 的 cell」） | §450.8 | 这些是旧散文/旧产物数字；当前 red·B2 = **0.615/0.800**（6 格 20 特征）/ **0.526/0.790**（8 格 18 特征），「两个口径都高于随机」，区间 0.615–0.864 / 0.526–0.758。 |
| §388.4 / §392.2（D4 B10/B11 的 triage AUROC 绝对值） | §399.4 | producer 20 个特征里 2 个恒为 0；负结论偏保守不翻，但引用 AUROC 绝对值须先重跑（§450.8 重渲染后的数字见 C2）。 |
| D4 B10 「Holm at m=6, 阈值 0.008333」在 8 格产物里的沿用 | §450.4 / §450.17 | 8 格族最紧阈值 **6.25e-3**；按 m=8 独立重算为 **1/8 存活**（red·B2 p=0.0004）。 |
| §302.5（provider-dependent noise floor 只能停在可观测层） | §470.5 | 不是推翻而是坐实：12% 的 discordance 是 B0 serving 栈特有的，B1 完全可复现。 |
| §306（exclusive-solve 计数） | §470.3 | 不是推翻：all-A 结果逐格一致，但 phantom 臂的 unique-solve 计数落在 replicate 噪声包络内（下界 0-1）。 |
| 证据层 0b-extra（B1·cls routing AUROC 0.870 被当作 early-abort 有信号的依据） | §460.3 | 0b-extra 是整 episode 聚合，对 prefix 决策等于看未来，不可作 early-abort 依据。 |

---

# 覆盖性闭合

- **本批条目数 172 = 实际用到 172**；未归入任何主题的条目 **0** 条。
- 主题数 **35**（A 7 · B 2 · C 5 · D 3 · E 4 · F 2 · G 1 · H 1 · I 4 · J 1 · K 1 · L 3 · M 2）。
- 主题 ↔ 条目（§）对照：
  - A1: §398.2(dom pair) · §450.14 · §470.3(六臂 flip)
  - A2: §406(B1 WA 地板) · §470.5 · §470.7 · §472.3
  - A3: §398.2(manifest 19 组) · §467.1 · §468.5 · §469.5
  - A4: §406(B0·cls 边际) · §406(B1 WA 边际) · §407.3(×2) · §450.10(one-arm margin) · §452.1 · §455.2 · §455.3
  - A5: §407.8 · §450.10(门槛) · §450.18
  - A6: §470.3(包络)
  - A7: §407.4 · §407.22(富集) · §409.1 · §455.1 · §464.2
  - B1: §398.2(permutation null) · §398.2(Jaccard)
  - B2: §398.2(id-regime)
  - C1: §399.1 · §399.2 · §399.5/§399.6 · §401.1
  - C2: §399.4 · §412(visual_difficulty AUROC) · §450.2 · §450.4 · §450.8(5 个数字) · §450.12 · §450.17(AUROC/null) · §453.1
  - C3: §457.1 · §457.2 · §457.3 · §465 · §467.2 · §468.1 · §468.3
  - C4: §459.2 · §460.3
  - C5: §407.5(×2) · §408.1 · §419.3
  - D1: §407.2 · §407.21 · §418.2 · §463.3
  - D2: §406(VWA/WA 结构) · §406(B1 WA SR) · §407.17 · §407.25 · §412(axis 第七格) · §420.5
  - D3: §450.15 · §452.4
  - E1: §400.2 · §401.2 · §401.5 · §407.6 · §407.22(24 指标) · §408.2 · §409.5 · §412(26 指标)
  - E2: §400.1(×2) · §400.3
  - E3: §401.6(×2) · §402.1 · §402.3 · §407.23 · §411 · §412(text-wins 六候选) · §420.2 · §458.1
  - E4: §402.5 · §402.6 · §462.1 · §465.1
  - F1: §407.15 · §407.19 · §407.20 · §408.4(×2: canonical 比值 / claim 9 稳健性) · §409.2 · §413 · §418.3 · §458.2
  - F2: §408.4(未读字段) · §412(字段可用性)
  - G: §442 · §442.7(×2) · §442.8
  - H1: §445.1 · §445.2 · §445.3 · §447.2 · §448.2 · §450.16
  - I1: §444.1 · §446.1 · §448.3
  - I2: §444.4 · §456.1 · §456.3(×2) · §468 · §471.3
  - I3: §471.1 · §471.5 · §472.8 · §472.9(×2) · §472.11(×2)
  - I4: §466.2 · §469.2
  - J: §446 · §447.1 · §449.1(×2) · §449.2 · §449.7(×2) · §451 · §468.10
  - K: §446.5 · §449.3 · §449.4 · §449.6 · §451.6 · §466.1 · §471.7 · §471.8 · §472.5 · §472.7 · §472.13
  - L1: §450.6 · §450.8(测试基线) · §450.9 · §450.13 · §450.17(7/7 交叉验证)
  - L2: §454.2 · §474.1 · §474.2 · §474.6 · §474.9
  - L3: §473.1(×2) · §473.4 · §473.6 · §473.7 · §473.8(×2)
  - M1: §398.1 · §398.3 · §398.4
  - M2: §398.7
- 「(×2)」表示该 § 在批次文件里有两条独立 MEASURED 条目，均归入同一主题。

*本文件由 D5 批 172 条 MEASURED 记录聚合而成。数字一律原样抄写，未做任何算术（唯一的「聚合者注」是 C3 中指出 §457.2 与 §450.2 六个 AUROC 逐格相同，属比对非计算）。§475 之后的状态见 measured_D6.md。*
