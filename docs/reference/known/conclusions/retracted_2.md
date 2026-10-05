---
type: conclusions
batch: R2
status: done
created: 2026-10-06
source: scratchpad/batches/R2.jsonl（182 条 = RETRACTED 152 + CLAIM_UNVERIFIED 30；按 § 排序，RETRACTED 跨 §398.5–§527.4，CLAIM_UNVERIFIED 跨 §442–§524.1；日期 2026-08-02 → 09-15）
---

# 作废与待验 R2（§398.5–§527.4）

> **读法**：本批接在 `retracted.md`（B 批，§1–§397.10）之后。时期依次是：证据层自审
> （§407–§411），CLAIM_EVIDENCE_MATRIX 与跨 AI /stress 对账（§442–§453），proxy 价格与 B4/GPT 接入
> （§456–§471），毕设、REALM、VLM4RWD 的写作收尾（§473–§504），showcase 海报与演讲（§495–§527）。
> 作废的重心从「数字算错」移到了**「副本跟不上源」**（M12 + M15 = 30 条）、
> **「推理代替测量」**（M17 = 22 条）和**「改写措辞改变逻辑」**（M14 = 18 条）。
> 每条作废标 `R2#n`（=批次文件第 n 行），一条只归一个主模式，括号里是次要模式。
> 「已作废」的数字**禁引**；noise 类数字各带 scope 并列，**禁止相加相减**。

---

## 本批作废按错误模式计数

M1–M11 定义沿用 `retracted.md` §一；M12–M23 是本批新开，定义见 §一 各小节开头。

| 模式 | 一句话 | 本批条数 | 来源 |
|---|---|---:|---|
| M1 | 从代码文本/commit diff 推断，不落实证 | 2 | 旧 |
| M2 | 查代理量，不查真对象 | 8 | 旧 |
| M3 | 分子改了，分母没改 | 0 | 旧（本批分母类错误都归 M20） |
| M4 | 子串/关键词匹配冒充结构化提取 | 4 | 旧 |
| M5 | 小样本给出假确定性 | 8 | 旧 |
| M6 | 单臂测量制造幽灵 confound | 2 | 旧 |
| M7 | in-sample 估计冠以推断性名字 | 7 | 旧 |
| M8 | 指标的判定基准在各臂不是同一个 | 2 | 旧 |
| M9 | 照着别人报的表面问题修（或驳） | 1 | 旧 |
| M10 | 自己立的防线/禁令自己先违反 | 3 | 旧 |
| M11 | 看到「资源残留」就判泄漏 | 0 | 旧 |
| **M12** | **冻结副本漂移**：事实冻在散文/caption/硬编码/台账值里，源已前进 | **23** | 新 |
| **M13** | **缺证当证无**：在错处、用错探针、用不全的关键词查到 0，写成「不存在」 | **8** | 新 |
| **M14** | **措辞/概念层失真**：改写时改变逻辑强度、定义或范围；系统描述不忠实于代码 | **18** | 新 |
| **M15** | **作废没有传播**：已判死的说法在别的副本里存活并被再次搬运 | **7** | 新 |
| **M16** | **时变外部量当常量**：proxy 价格、可达性写成常量 | **4** | 新 |
| **M17** | **推理/外推代替测量**：没跑、没算、没打开就写成事实 | **22** | 新 |
| **M18** | **构造/代数必然当实证**（或反之）：缺 null/恒等对照 | **5** | 新 |
| **M19** | **统计机制误用**：抽样依赖、近似、破平、置换下限、Simpson、AUROC 性质 | **6** | 新 |
| **M20** | **口径/参照系错配**：分母集合、base、格、时区、baseline、量的种类对不上 | **9** | 新 |
| **M21** | **外部事实未查原始来源**：政策、日期、领域惯例凭转述进推理链 | **6** | 新 |
| **M22** | **非推理错误**：方案被 user 改判，或待验被确认关闭（不算犯错） | **3** | 新 |
| **M23** | **静默仪器**：工具/门/自检坏了仍输出「正常的样子」 | **4** | 新 |
| 合计 | | **152** | |

---

## 一、逐条按错误模式归档

### M1. 从代码文本推断，不落实证（2）

- **R2#1 §398.5** 「`preregistration_decision_test.py` 的注释 stale」→ 第 34 行明写 `⚠️ REWRITTEN 2026-05-13 (historical):`，Makefile:471 明写已 retired。原文自评：*「只看 grep 输出的孤立行就下断言 = M1 的实例, 而 M1 是自己归纳并排第一的那条」*。现值：注释是正确的历史记录，不需修改。
- **R2#113 §489.1** 「sanitize 结果落 `info["type_text_sanitized"]`，step record 可查」→ runner 只读 `action_executed` / `locator_route_meta`（main.py:4008/4064），该字段从未落盘。随后反向说成「静默丢字符什么都没记」又说重了（logger.warning 进 runner log）。现值：**结构化字段死了（聚合脚本读不到），log 行还在**；至今触发 0 次。

### M2. 查代理量，不查真对象（8）

- **R2#2 §406** 「B1 完全确定性（do_sample=False ⇒ 重跑 bit-identical）」→ step 级（§298.2 133/133，§397.10 组内一致性 1.000）被当成 episode 级；episode 级 3/50 翻转。*「解码确定性 ≠ episode 确定性」*。现值：B1 episode 地板 2.00/4.00pp。
- **R2#51 §455** 「A100 仅剩 41GB ⇒ VWA shop 18cond=33.5GB 装不下」→ 量错了盘：VWA fire 落 `/mnt/scratch`，avail 278G。现值：18cond 33.5GB / 两站 46.8GB 都装得下。正确探针 `df -h results/visualwebarena` 或 `readlink`。
- **R2#66 §469.5** halt marker「`validate_fire_manifest` 把 nothing to bind 误判 exit 1」→ 照 marker 推断；marker 只留拼接后最后 400 字符（`experiment_watchdog.py:1184`）。真因：R28065 被判 COMPLETE ghost。
- **R2#76 §473.8** 「VLM4RWD 匿名 0 泄露」（`pdftotext | grep` = 0）→ pdftotext 看不见 `/PTEX.FileName` 里的绝对路径。现值：四渠道审计（strings / pdfinfo / pdfinfo -meta / pdfdetach -list）。
- **R2#110 §482.3** 「SoM 对 Vision 溢价 8 格里 7 格为正」→ 照图上点目测；源表 6/8 严格为正，cls·B2 0.00，red·B2 -0.98。
- **R2#117 §490.4** 「two-arm 标签 z 每个任务都有定义，供给比 which-mode 好」→ 行数是代理，正类 base rate 只有 2.2%-14.4%（5-22 个正例/cell）。（此条的「稀缺」诊断后又被 R2#120 收窄。）
- **R2#134 §502.4** memory「毕设/REALM/VLM4RWD 三件均已交付」→ 从「截止 09-05 已过」推断完成。现值：毕设 2026-09-08 已交（119 页）；REALM 09-08 接受、camera-ready 09-14 未做；VLM4RWD 在审（notif 09-29）。*「deadline 过去 ≠ 事情做完」*。
- **R2#136 §504.1** 「REALM 提交稿 = `main_realm.tex` + `realm_*.tex`」及据此报的三条待修项 → 读的是废稿。现值：提交稿 = `main_restructured.tex` + `sections/*.tex`；三条待修项在真稿里均不成立。

### M4. 子串/关键词匹配冒充结构化提取（4）

- **R2#22 §411** R1 PREMATURE_FINISH_ON_FORM「WA 24 failed 命中」→ 24 是基条件的数，加上 prose 自述条件是 22；首次数出 26 是 glob 同时匹配了 pilot run。现值：22。
- **R2#23 §411** R3 PREMATURE_NEGATIVE_AFTER_SEARCH「WA 9 命中」→ sub-agent 措辞集比落地正则宽；WA 实际 0。现值：WA 0 / VWA reddit 10。
- **R2#83 §474.8** 「trainable 说法只藏在 ch1 和 ch6:260 两处」→ 只 grep 了完整短语，漏 ch5 小结与附录 C。现值：四处，已全部收口。教训：*「grep 词根 (`trainab`) 而不是完整短语」*。
- **R2#88 §475.6** 「二进制只需 `strings | grep -i 'author|creator'`」→ 关键词是猜的，泄漏在路径里。现值：`binary_probe()` 查路径片段；248 个二进制复扫 clean。（次：M2）

### M5. 小样本给出假确定性（8）

- **R2#7 §407.20** 「碳/能耗没有采集」→ 只看了 B0 一条 step record；B1 321/321 步、B2 542/542 步全部采满。现值：采了但因 NVML 回落 psutil 不报。（次：M13）
- **R2#9 §407.22** 「visibility_gap 上 Vision 一致最高」→ 只算 2 格（4.3/6.0%）；全 6 格 Vision 2/6 最高、4/6 最低 ⇒ 无信号。
- **R2#31 §449.1** 「B0 成本 $0.06/episode」→ 五分钟窗口恰落在最便宜的 vision condition；dom $0.1198 是它的 1.66 倍。现值：按 (site, mode) 分层实测单价表。
- **R2#32 §449.1** 「剩余预算 ≈ $315，申请 $500」→ 三项都派生自 R2#31。现值：≈ $546（shop $150 / WA $175 / B4 $221），低估约 2 倍。
- **R2#63 §468.4** 「leak 修正让 red_B2 区间跨零 ⇒ 证据靠累积站点状态」+「8/8 全含零」→ discordant d = 8→5，六个对照里三个 d<10；区间会动是因为没有 power。现值：**underpowered, not corrected**；「所有被审的格含零」（6 格，非 8 格）；`wa_red_B0`/`wa_red_B1` 从未被审。
- **R2#67 §470.2** 「纯 format 轴有正主效应（WA·B0 +8.66pp）」→ 全 8 格 2 正/2 零/4 负，median −0.45pp；WA·B0 是唯一一格且 WA cells unaudited。
- **R2#116 §489.4** 「33 ep 出 2 波 显著高于 0.74/100」→ Poisson P(X≥2)=0.1172。现值：「无法判定异于前次（Poisson p=0.117）」。*「未做假设检验不要写「显著」」*。
- **R2#131 §500.1** 「跨 side 覆盖差异稳健，视觉侧整体 > 文本侧整体」→ 只在 cls·B0 一格测过；reddit 上 Vision 下界 6→2、P-prompt 1→2，两侧接触。现值见 §二 T2。

### M6. 单臂测量制造幽灵 confound（2）

- **R2#11 §407.26** §407.25 的三条更正本身：P6/P16/P17 在 WA 0.0% 读成「机制缺席」，实为 site gate（VWA reddit 同样 0.0%）⇒ 拿闸门比测量；P43 定位效应不解释效应。现值：只有 WA-reddit vs VWA-reddit（站点类型固定）活下来；固化为 conditional_failure_attribution §4 七规则三站触发率表。
- **R2#115 §489.3** 「logout 全落 vision，坐标通道让事后归因失效」→ artifacts 里有 observation_dom.txt；登出在 step 8；§329 记的 B2·**dom** 同 task 同 step 同误点。现值：OSClass 把 My account 与 Logout 排在一起的 UI 陷阱，跨模型跨观测模式复现。（次：M13）

### M7. in-sample 估计冠以推断性名字（7）

- **R2#21 §409.5** 「P-SoM 在 7 个 (metric, cell) 上区别于两端 ⇒ 独立路由臂」→ 插值模式也区别于两端。现值：要求两腿反号后 6 个 off-segment。
- **R2#42 §450.10** C2「三个 phantom arm 各自独解一批任务，双轴独立」作正面结构性主张 → pooled 双轴效应 1.35 / 2.09pp 低于最宽松地板 2.0pp。判词 *「does not survive as a positive claim」*。
- **R2#46 §452.1** 「which-mode router 无从拟合，4/6 格 no trainable classifier」→ trainable 是 analyst 准入规则（N_MIN_CLASS_TRAIN=10）；去掉 filter 后 30/30 fold 全部拟合成功。现值：「4/6 格未通过 min_class_n=10 准入规则」；trainable→minted。
- **R2#49 §453.2** 「标签供给约束是 structural」→ 四个失败格只需 2.1–4.2× 现有语料。现值：从 impossibility 降为 specification；倍数是下界不是承诺。
- **R2#56 §462.2** abstention「held-out 0 损失省 0.8–24.7% … ≤5% 到 11.2–47.2%」→ 阈值用测试折真实标签选出，是 oracle-selected。现值：只能当乐观上界，不可与 §5 的 9.5–30.6% 并列。
- **R2#90 §476.1** 「$U(6)$ is an upper bound on repetition」→ 两个观测量三个参数，$p$ 不可识别；$p=1/2$ 落在可行区间第 12-22 百分位。现值：SoM $U(6)\in[35.72\%, 54.33\%]$。
- **R2#98 §477.4** 「More than half of the ceiling is repetition at every p the data admit」→ 端点是 plug-in 值不是置信界；临界 d=27.81 vs 实测 d=29。现值：只引 59%（bootstrap 下界 >1/2 的比例）。

### M8. 判定基准在各臂不是同一个（2）

- **R2#60 §467.3** 「red_B2 SoM−DOM = −2.96pp [−5.91, −0.49]，融合显著劣」→ 该臂 8 个成功里 3 个 leaked（37.5%）；primary leak 政策下 [−3.45, +0.49]。（其「8/8 全含零」随后被 R2#63 再修。）
- **R2#144 §508.1** 「B5 vision SR 7.4 / 12.1% 是 backbone 臂形状」→ GPT-5.6 不守 0-1000 坐标契约，B-1860 把像素当千分制，click 85% 落空。现值：B5 vision 在 B-1997 重跑前不进任何跨 backbone 比较。

### M9. 照着别人报的表面问题修（或驳）（1）

- **R2#121 §493.1** 「GPT 关于毕设 float 的整条判断为误」→ 它的修法确是空操作，但现象真（系统扫描 8 处）。现值：§493.2 真机制是 30 个 float 全 `[t]`-only，真修法 `[!b]`。原文：*「驳倒归因很容易顺手把现象一起丢掉」*。

### M10. 自己立的防线/禁令自己先违反（3）

- **R2#94 §476.4** 毕设 Conclusion「two benchmarks whose tasks differ in how much they depend on what is visible」→ 违反 §3.5 自设 scopebox（只主张任务规格差异，不主张模态差异）。现值：`across two benchmarks with different task specifications and annotation schemas`。（次：M14）
- **R2#109 §482.2** 「+2.23pp 恰好等于 rerun band 宽度」+ fig_fusion_forest → 正是 b_noise.tex:52 的反面教材 *「compares a draw to a draw」*，且跨 cell 借 band。现值：撤图；只报方向；只有 cls·B0 有实测阈值 3.52-4.15pp。
- **R2#149 §513.3** 片子并排「事后 +3.45 到 +16.35」与「重跑一次买 2.0–7.6」→ noise_floor_inventory.md §2 明写 not licensed。现值：仅 cls·B0 同臂数比较 +7.14 vs +4.46–7.59。

### M12（新）. 冻结副本漂移（23）

**定义**：一个事实被冻进散文、caption、硬编码字符串、台账旧值、memory 或讲稿，而它的源（产物、数据、臂数、规则版本）继续前进。原文自己计数为「缺口条比产物旧 / 引用比产物旧」，到 §450.18 已是*「第六次」*。修法一律是**让产出处从源现读**，而不是改那句话。

- **R2#5 §407.17** 「VWA 40.0% 的目标以参考图指定」→ §4:91 的旧数，继承时多加了 `.0`。现值：33.7%（classifieds 29.0% + reddit 38.9%）。
- **R2#6 §407.19** limitations「Two sites, one benchmark」→ WA 已在用。现值：两个 benchmark。
- **R2#13 §408.2** 「B1 axis-1 算不了，P-text pending」→ `axis1_microbehavior.py:834` 硬编码；数据 06-05 / 07-10 早已落地（229 / 210 个 summary）。
- **R2#14 §408.2** 「6 对对抗性机制配对」→ 硬编码数字，与同文件「没有一对达到 |0.1|」自相矛盾。现值：从 cancellations 现算。
- **R2#15 §408.5** 「Tier 2a partial，P-prompt 未到位」→ 36 格全齐，alt-path 早在 json 里。现值：Tier 2b diamond 表。
- **R2#27 §420.2** 「text-wins 侧 nothing clears 1.5×」（ruleset v9）→ v11 rescan 后 P49 达 3.61×，8 个 hit 全在 WA 两格。现值：WA 上有经因果验证的机制，VWA 六格原句保留。
- **R2#36 §450.3** C4「0/6 cell」→ 已是 0/8 跨两个 benchmark。
- **R2#37 §450.3** C4 置换「label-shuffle B=200」→ bundle-permutation，B=10000，plus-one (k+1)/(B+1)。
- **R2#38 §450.4** 散文「AUROC 0.651-0.717」→ 两个口径都不是。现值：20 特征 0.615–0.864；18 特征 0.526–0.723。
- **R2#43 §450.14** 「同模式重跑地板只有 vision 一格」（引 §302）→ B0·VWA-cls 已有 3 个 replicated arm，som pair 08-03 落地。（次：M13）
- **R2#44 §450.18** C2「1.7–3.3pp，22/24 arm 为正」→ 旧口径 `meta_phantom_lift.csv`（k_cells=3）。现值：0.00–2.68pp / 14/18；**新数字更弱**。
- **R2#54 §460.1** `layered_evidence_status.md` 的 claim 矩阵 C1/C2/C3/C6 全 ✅ → `layered_status.py:701-712` 硬编码，停在 pre-REALM framing。*「一个指向退役 claim 的索引比没有索引更糟」*。
- **R2#58 §464.2** `label_instability.md` 两臂数字（49/224，51.1% vs 2.9%，17.4×）→ som replicate 已落地，产物没重跑。现值见 §二 T7。
- **R2#81 §474.8** 「de-jargon 只有 ch1/ch5/附录A 改善」→ 是做到一半的快照（~230 条卡点只处理了 ~55 处）。现值：全稿 8.05 → 7.34 jargon/100 词（-9%）。
- **R2#97 §477.2** 毕设 band normal 3.82--4.15 / exact 4.02--4.46 → 三臂时代值。现值见 §二 T1；*「『exact 更保守』是三臂下的巧合不是性质」*。
- **R2#102 §479.4** Table 27 caption「three arms … no VWA-reddit cell」→ 现读：`B0.cls x6, B0.red x3, B1.cls x3, B5.cls x1`。
- **R2#103 §480.3** REALM #7「Only 2 of 8 cells carry a measured floor」→ 现 3/8。REALM #192 是已提交快照，不回改。
- **R2#104 §480.3** REALM #5「0.89-2.23pp」及「+2.23pp 恰好等于上沿」→ 六臂后 0.89-2.68pp。
- **R2#105 §481.2** fig_f9 本地 `C_SIDE` 与 fig_f5/f7 配色正好相反 → 统一到 `_style.py`（文本侧=橙 #E8720C / SoM=绿 #14855F / Vision=蓝 #1F5FD6）。
- **R2#108 §482.1** `_style.py` 注释「最小标签约 8pt」→ 按旧 16cm 版心写；实测图内标签 4.99pt。现值：PRINT_W_IN=5.12 按印刷宽度出图。
- **R2#137 §504.2** 在审稿 `3_noise.tex:5`「no VWA-reddit cell and no B2 cell carries a replicate」→ 覆盖已扩到 18 对 / 5 格。只剩「no B2 cell carries a replicate」成立。
- **R2#138 §504.3** 「camera-ready 换成三臂值 67/224 …」→ 产物已六臂。原文：*「与被我批评的那个 bug 同形(引一份不在流通的副本)」*。（次：M10）
- **R2#141 §507.1** SHOWCASE_PREP 讲稿与印刷海报一致 → 印的是 v9.10，讲稿还是 v8 结构。*「版本号更新 ≠ 内容更新」*。

### M13（新）. 缺证当证无（8）

**定义**：在错误的主机/目录、用错误的探针、用不全的关键词查到 0，就写成「不存在 / 不可能 / 没做过」。原文 §501.1 的不对称论证：*「误判『有』最多白查一次, 误判『没有』会去重新生产它」*。

- **R2#3 §407.7** 「cascade 部署期没有失败信号」→ confidence 每步都采（B0 4/6，B1/B2 6/6），paperB 引用 0 次。现值：confidence 作为 cascade 失败检测器（§407.7b）。
- **R2#10 §407.25** 「8 个 WA run step JSONL = 0 ⇒ 不可能」→ 本机 `find` 返回 0；数据在 paper-grade host，132.6 MB。原文：*「user 的反对理由是「不合理啊」—— 比那个 find 更好的仪器」*。
- **R2#35 §450.3** 「WA 只进了 oracle 层，没进 learnability 层」→ `router_triage_learnability_with_wa.md` 08-03 就存在（缺口条写于 08-09），`_with_wa` 后缀无索引。
- **R2#41 §450.10** 「台账 `pass@` 0 match ⇒ 项目从没做过」→ noise_floor_inventory.md 08-04 已做，叫 one-arm margin / rerun band。*「查关键词必须连同同义表述一起查」*。
- **R2#57 §464.1** Kimi K3「没有任何产物计算过这个交集」→ 产物存在；是我的 prompt 没内联它。教训：*「内联式跨 AI 审计中, 未内联的产物会被审计者读成不存在」*。
- **R2#72 §471.5** 「GPT-5.6 tools 墙要去找 proxy 要」→ `response_format` json_schema 在完整生产形状下一次通过。唯一真无解：logprobs。
- **R2#133 §501.1** 「两个 SoM run 的 artifacts 已被清，要重录」→ A100 上 6122 / 6076 张 png 完好；DGX 侧为 0 是因为 artifacts 从不在 rsync 范围内。
- **R2#139 §506.4** 「cron 仅三个任务，不碰 artifacts」→ DGX crontab 有六个任务，`sync_a100_results.sh` 每 15 分钟 `--delete-excluded`。

### M14（新）. 措辞/概念层失真（18）

**定义**：数字常常是对的，错在改写、压缩、de-jargon、起标题、写 caption 时改变了**逻辑强度**（联合→并列、限定词丢失、否定式说过头）、**定义**（triage、noise floor、image-only），或**系统描述不忠实于代码**。原文 §478.1：三家 AI 审计都审「证据是否支撑 claim」，而这类是「系统描述是否忠实于代码」，*「不同的层, 且这些句子不带数字正好落在审计盲区」*。

- **R2#30 §448.5** 「WA 无 reference image ⇒ 视觉 grounding 需求不存在」→ 混淆 visual matching 与 visual grounding。现值：只主张任务规格差异。
- **R2#75 §473.2** VLM4RWD intro：「three of the six modes are screenshot-free」/「grounding ablation」/ 用 AUROC 0.483 证「模型不知道自己看到了什么」→ 实为四个无图 mode；只有 SoM vs P-SoM 一对干净；0.483 与本稿附录直接对立。（次：M10）
- **R2#77 §474.3** 摘要「So does the retrospective oracle, in 7 of 8」→ 字面 = oracle 赢 7 格。现值：`fails to clear it in 7 of 8`。
- **R2#78 §474.3** ch2「any per-step cost is multiplied by one」→ 自 commit 63797b7 起的残句。
- **R2#79 §474.3** 「always-cheapest 是成本下界」（误删 `over fixed policies`）→ 见 §二 T5。
- **R2#84 §474.9** 「never beats always-cheapest — neither on success nor on cost」→ 七格里六格 SR 是赢的。原文：*「把联合条件拆成并列句会改变逻辑强度, 是 de-jargon 最危险的失效模式」*。
- **R2#85 §474.9** triage =「决定任务是否需要贵模式」→ 正文定义是「是否有任何模式能解出」。四处都是 de-jargon 时引入的。
- **R2#86 §474.9** 速查页 noise floor 单一定义 → 附录定义两个 floor。原文：*「原本教给读者的正是本文用整个附录来防止的读法」*。（其 replaced_by 里的阈值次日又被 R2#97 更新，见 §二 T1。）
- **R2#92 §476.4** 「no profitable (deployable) operating point exists」→ 论文从不定义汇率。现值：`no Pareto-improving operating point exists against always-cheapest under this cost boundary`。
- **R2#93 §476.4** Figure 6.1 图注「necessarily also spends more」→ 与同图注「oracle 7 of 8」冲突。现值：`cheapest fixed policy on average, not a per-episode cost floor`。
- **R2#99 §478.1** 开篇三分「结构化文本 / 带编号方框的截图 / 两者结合」→ 纯截图缺席。现值：结构化文本 / 纯截图 / 两者结合。
- **R2#100 §478.1-2** 8 处「interactable / candidate elements」→ 零 role 过滤，所有带 id 的 AXTree node 入选。
- **R2#106 §481.2** 表注「On the WebArena cells the winner is P-text and DOM」→ wa_red·B0 是 P-text 单独 35.58%；wa_red·B1 是 DOM / P-prompt / P-text 三路并列 16.35%。
- **R2#111 §484** 「All modes ground actions through element identifiers」→ vision 491/491 用 pixel coordinate。replaced_by：**待定**，范围决策未做。
- **R2#128 §499.10** 海报标题 `IT DOES NOT WORK` → `SO CHOOSE PER TASK? NOT SO FAST.`（user 指出）。
- **R2#130 §499.12** caption `None does.`（user 给出）→ 绿区内确有 oracle 点。现值：`Only the hindsight oracle enters the win region.`
- **R2#135 §503** 表 13 caption「element-id path 达到 89%」→ 单数 path 引的是 `id_locator` 一条子路径；审稿人 sVJH 据此读出 50.3pp。现值见 §二 T10。（次：M8）
- **R2#148 §513.2** 「image-only」→ image channel = som + vision。现值：LOOK and BOTH。

### M15（新）. 作废没有传播（7）

**定义**：推翻已经发生并入台账，但旧文字在别的副本（caveat、另一章、讲稿、CLAUDE.md、config 模板）里存活，后来又被原样搬运。与 M12 的区别：M12 是源在动、副本没动；M15 是结论已判死、副本没跟着死。原文连续三次写下同一条教训（§474.4 / §474.8 / §495）仍复发。

- **R2#61 §467.4** `abstention_learnability.md` §3 正文仍引 ORACLE_SELECTED 列的 11.2-47.2%，并说「into and past」oracle 区间（B-1971）→ *「§465 换掉了表没换表下面那句」*。现值见 §二 T6。（次：M10，docstring 明写 no number is hardcoded in prose）
- **R2#68 §470.4** CLAUDE.md / memory 两条 hero：「drop-one 1.7-3.3pp = THE principal hero metric」「AUROC ≥ baseline ⇒ 可学」→ 存活至 2026-08-19。现值：canonical framing 唯二来源 = `section1_intro.md` + `TERMS.md`。
- **R2#80 §474.4** ch1 / ch6:260 的 trainable 说法 → §452.1 早已推翻，ch5/ch7 改了、这两章漏改。*「RETRACTED 落地必须全稿 grep」*。
- **R2#82 §474.8** 结论段同一句式（`so does` + `cost lower bound`）→ 修摘要时没 grep 结论段。*「摘要与结论互为镜像」*。
- **R2#123 §495** 海报 v2 图注「Always-cheapest is a cost floor」→ 从 §450.12 散文搬运，未查已作废。
- **R2#124 §495** 根因：推翻发生在 §474.3/§476.4，被推翻的文字留在 §450.12（MEASURED）的 caveats 里。§450.12 caveat 其余部分仍成立。
- **R2#126 §498.3** SHOWCASE_PREP 讲稿「13.7–35.3% lower cost」「12–14% rerun discordance, three replicated modes」→ 两个都已作废。现值：1.6–35.3% / 10–14%（six replicated modes, B0·cls n=224）。

### M16（新）. 时变外部量当常量（4）

**定义**：proxy 价格、模型可达性被写进台账或 config 当常量。§478.5 的结论：*「问题不是某个数写错, 是任何写下来的价格都会过期」*。

- **R2#52 §456.2** §444.4「5 个 Anthropic 条目是死的」→ 08-12 重探九个候选全部 200。*「占位价→真实价本身就是上线信号」*。
- **R2#53 §456.2** §444.2「sonnet-4-6 与 B0 完全同价 0.001/0.005 ⇒ 12 conditions ≈ $154」→ 那是占位价。08-12 实测：sonnet-5/4-6 = 0.003/0.015 = B0 的 3 倍；真与 B0 同价的是 haiku-4-5。
- **R2#71 §471.2** §468「gpt-5.6-{sol,terra,luna} 与 B0 严格同价」→ 实测 sol 0.005/0.03 · terra 0.002/0.012 · luna 0.0002/0.0012。
- **R2#101 §478.5** §471.2 的替代值本身 → 08-26 terra 0.0025/0.015 · sol 0.00625/0.0375 · luna 0.00025/0.0015。现值：**不存可引用的价格**，每次现跑 `probe_proxy_model_registry.py`。

### M17（新）. 推理/外推代替测量（22）

**定义**：没跑、没算、没打开就把推断写成事实，包括估算、预测、「改一行就行」、「纯排版」。与 M1 的区别：M1 的推断源是代码/diff 文本；M17 的推断源是直觉、旧计划、类比或局部观察。

- **R2#4 §407.15** 「B0 latency 走共享 proxy 被排队污染」→ 实测 B0 每步 CV 0.15-0.22 vs B1 0.11-0.19，同量级。
- **R2#8 §407.22** 三条 ◆「架构下游」机制 → 全部测掉：scroll 前一步 no-op 比例在 6/6 组合都在基线或以下（Vision·cls 18.5% vs 36.4%）；no_change 的「误点下游」只对 85%；Vision click 失败仅 3/6 最高。现值：新增 UNADJUDICATED 注册表。
- **R2#25 §413** frame「图任务付延迟」→ 两个本地 backbone 上 Vision 每步更快（B1 +8.1% / B2 +3.7%），只有 B0 慢 1.8%；B1 区间 -5.9%~+13.4% 跨 0。
- **R2#34 §449.2** 「B1 shop 435×3 ETA ≈ 9.3 天」→ 实测 8.0 ep/h ⇒ ~5.6 天。
- **R2#39 §450.8** 「8 格族下 Holm 没重算」→ m 一直动态（0.0063 = 0.05/8）。*「该推断是在没读 :733/:736 之前下的」*。现值：m=8，1/8 reject。
- **R2#45 §451.5** 「shop/cls 中断后必须 FORCE_NEW=1」→ 查历史：13 条 chain log 全部 RESET_BEFORE=1；B1 dom 从起始就是 FORCE_NEW=0。现值：canonical 启动是 orchestrator（FORCE_NEW=1）；残留 R23934 的 435 个混两代容器，建议登记不重跑。
- **R2#55 §461.1** 「B4 = 改一行 config」→ 实际三处：`api_name`、`metrics.cost_api` 硬编码（不改则成本低估 3 倍）、`queue_baseline.sh` 白名单与 proxy key 按 baseline 名门控。（次：M15，§456.2 只 retract 了占位价的台账那句，没管 config 模板）
- **R2#59 §466.3** 「B4 落地仍是改一行，预算 $136–272」→ Anthropic-native 条目逐层 400。现值：协议分支 + tool_choice 枚举 + schema 摊平；*「不要按 $136 排」*。
- **R2#62 §468.2** 「abstention 迁移失败机制是 base-rate 漂移且单向」+ headline → 被自己的表推翻；headline 是两个极值的事后配对，null 下 P=1/6=0.17。现值：预算量子化。（次：M18）
- **R2#65 §469.3** 「并行 web_action 的 call 2+ 是推测，丢弃语义正确」→ 两个 call 基于同一 observation。现值：take-first 是确定性 wrapper 策略，可能惩罚多步规划。
- **R2#73 §472.3** AMENDMENT「B1 地板是个常数，剩余格也返回 0.00%」→ 第一次检验落在 3.12%（7/224）。
- **R2#74 §473.1** 「VLM4RWD 模板移植是纯排版，正文已压进 8 页」→ wrapfigure 吃掉 caption；撤掉后 9 页。现值：fig_ceilings 移附录 = 8 页。
- **R2#95 §477.1** 「异质 $p_i$ 会让区间更宽」→ *「没算就写了」*；优化后端点完全相同 `35.72%--54.33%`。
- **R2#107 §481.3** 「完全使用 Overleaf 官方模板 = 排版正确」→ UCL 模板自带两个缺陷。
- **R2#112 §485.1** 「fig_overview 保留横排」→ 会议稿本来就竖排发表（最小实词字高 0.96pt）。现值：全稿横排 3 处 → 0 处。
- **R2#114 §489.2** 「08-27 订阅那行熬过了每一次 reset」→ 时间戳不支持。现值：成因未确定；物证随 B-1997 修复被 DELETE。*「清脏数据前先 dump」*。
- **R2#122 §494.1** float 表首行「[t](原始) 116页/8处」→ 推断不是测量；还原重编译实测 10 处。现值：10→1。
- **R2#129 §499.11** 「screenshot 是动作后状态、obs_url 是动作前」→ 方向反了。现值：screenshot = 动作前，obs_url = 动作后。
- **R2#140 §506.6** 「DGX 起不了 VWA classifieds」→ 未实查；社区 arm64 镜像可运行。现值：DGX 独立一套（127.0.0.1:9981）。
- **R2#143 §507.5** 「口播 1,150–1,300 词 / 900–1,000 词」→ user SOP 实测 1,240 词讲成 11–12 分钟。现值：上限 1,000 词无下限；判据是彩排 ≤ 9:30。
- **R2#146 §511.3** 「浏览器缩放 125–150%」→ 1080p 下截图只剩 149px / 0px。现值：100% + F11。
- **R2#151 §515.1** 「shop_B0 三臂 replicate 约 ~15 h」→ 按原发车间隔每格 32-59 h。

### M18（新）. 构造/代数必然当实证（或反之）（5）

**定义**：结果在 null 下或按定义本来就会出现，却被读成发现；或只有一半有构造保证，却被称作定理。缺的总是一个随机对照或恒等式检查。

- **R2#16 §408.5** 「text + prompt + image 在容差内还原 SoM − DOM」→ 空表下的固定句子；即使有数据，在 mean_diff 上也是代数恒等式。现值：45/48 + 「这是恒等式不是交互检验」。（次：M23）
- **R2#18 §409.2** 「latency 让 Pareto 前沿 3/6 格变宽 ⇒ 第二个轴」→ 加轴只会弱增大前沿；精确 null 期望 4.70/6，P(≥3)=0.978。现值：ρ(cost, latency) 均值 -0.095。（次：M10，该对照自己列为「便宜且开着」却没跑）
- **R2#26 §419.3** 「P43 命中集 dom 9.9% → som 29.6%（+19.72pp）」→ `check_p43` 只在失败 episode 上命中，dom@P43 是构造出来的。现值：剥掉 outcome filter 的事前分区（同 71 任务，+22.54/+19.72pp）→ visual_intent_routing。
- **R2#91 §476.1** 附录 B.4「U(2) 是 out-of-sample check」→ $k=2$ 时式中无 $q$，零信息量。现值：残差只检验 union 算术与跑间稳定性。
- **R2#119 §491.2** 「oracle_two_arm 对 oracle_triage 的 8/8 严格 Pareto 压制是构造必然」→ 只有 SR 一半有构造保证；三行反例。现值：cost 上的 8/8 是这批 log 的经验性质。

### M19（新）. 统计机制误用（6）

**定义**：抽样依赖结构、正态近似、并列/破平、置换 p 的下限、跨层合并（Simpson）、对 AUROC 性质的误读。每条都让结论跨过了它本不该跨的门槛。

- **R2#19 §409.3** 「WA 唯一一格 cascade 打赢 always-rich」→ 两个点都是排序并列的产物；换 tie 顺序 SR 跨 8.65-14.42%。现值：暂挂。
- **R2#20 §409.4** 「SoM − Vision 池化 95% CI [+0.12, +2.75] 越过 0」→ 同 site 三 backbone 共享 task 噪声；task-clustered bootstrap [-0.01, +2.91]；Cochran Q I²=59% / 77%。replaced_by 原话：*「且 FE 本身是错的 estimand」*（见 §六）。
- **R2#28 §442.8** 「B-1969 对 cls Phase 1a 确有污染（p=0.024）」→ 跨 cell 合并基线 = Simpson's paradox；分层后 p=0.2397。另三处：denominator glob 混入（4257 vs 4032）、latency 条目数当超时次数、100 次 permutation 报 p=0.000。（次：M4、M20）
- **R2#47 §452.2** C1「省 13.7–35.3%」→ best single mode 平局按列表序破平选中更贵者。现值：1.6–35.3%。
- **R2#48 §452.3** 「单侧门槛 3.82–4.15pp，3 个过阈」→ 正态近似偏松。当时现值：exact 4.02 / 4.02 / 4.46pp，3→1（后被 R2#97 六臂更新）。
- **R2#120 §491.5** 「AUROC(z) 低归因于正类稀缺；换标签定义换不掉稀缺」→ AUROC 对不平衡在期望上不敏感；只试了 y、z 两种就外推是过度归纳。现值：稀缺与特征不足两个诊断都标 open。（次：M5）

### M20（新）. 口径/参照系错配（9）

**定义**：数字是真的，但来自另一个分母集合、另一个 base、另一个格、另一个时区、另一个 baseline，或另一种量（估算 vs 余额、model−observed vs observed−model）。这是 M3 的推广：M3 只管「分子改了分母没改」。

- **R2#17 §409.1** 「加一条臂买 1.97-8.65pp」→ 8.65 是 §407.3 里另一个 base 的数。现值：1.97-7.14pp（post-hoc 两臂 oracle-ceiling 增量）。
- **R2#29 §446.7** 「max_tokens=1 探针比真实 run 多活 74 分钟」→ DGX = BST、A100 = UTC，未换算就拼接。现值：盲区 13分43秒，约 1.4 个采样周期。
- **R2#33 §449.2** 「B0 vision shopping 停在 374/466」→ 466 是语料全量，run set 是 435。现值：374/435，剩 61 个。
- **R2#40 §450.8** C4 权衡点举例「cls·B0 +1.79pp 但 cost +11%」→ 三处错配。现值见 §二 T4。
- **R2#96 §477.1** 附录六个 U(2) 残差 → 公式是 model−observed，贴的是 observed−model，六个全取反。现值：`-1.12, +0.45, +1.34, +1.12, +0.67, +0.67`pp；表注 1.12pp → 1.34pp。
- **R2#125 §496.5** 「B0 reddit replicate 3 格 × $52/格 = $156，超预算」→ $52 是 B5 的单价。现值：B0 reddit $21–23/格，三格 $63–69。*「跨 baseline 的单位成本不可互推」*。
- **R2#132 §500.4** 「B0·red·Vision discordance 5.9%（12/205），gap 2.49pp」→ 工具取 205 个 common task，canonical 是 203 个 scored。现值：4.93%（10/203），gap 1.48pp。
- **R2#145 §510.2** `unique_solve_envelope_cross_cell.md` red_b0 列 n=205 → 含 AMENDMENT_08 协议排除的 task 58 与 160。现值：n=203；Vision 2–5，P-prompt 2–4。
- **R2#150 §515.1** 「预算 $546 实测」当余额排 fire 清单 → $546 是费用估算合计。现值：09-15 余额 $30.18，一律现查 `proxy_budget_watch.py --once`。

### M21（新）. 外部事实未查原始来源（6）

**定义**：领域惯例、会议政策、日期凭印象或转述进入推理链。推理本身没错，输入错。§475.8：*「结论(主 repo 不必转 private)碰巧对, 理由全错」*。

- **R2#24 §413** frame「融合默认只赢得过选错的通道」→ 前提「SoM 是领域默认」错；主流是纯截图（user 2026-08-02 纠正）。
- **R2#64 §468.10** REALM notif「2026-09-07」→ user 08-16 更正为 08-21。（本身随后被 R2#69 推翻）
- **R2#69 §470.9** §468.10 的「08-21」→ user 08-19 确认仍是 09-07；毕设截止延到 09-05。*「该日期已翻转两次(09-07 → 08-21 → 09-07), 再见到 08-21 一律 stale」*。
- **R2#70 §470.9** 「08-21 那天 A100 要空着」及据此否掉 B0×red×3phantom → 输入日期错。现值：19 天 GPU 窗口投方向无关的工作。
- **R2#87 §475.2** 「public repo 会撞 ARR anonymity window，要转 private」→ 政策形状说反。（其替代「09-12 后不 push」又被 R2#89 推翻）
- **R2#89 §475.8** 「ARR anonymity window 约 09-12 起算」→ ARR 自 2024-01 已取消 anonymity period。现值：无时间窗口约束；硬约束在提交件本身（不得含非匿名 repo 链接）。

### M22（新）. 非推理错误：改判与待验关闭（3）

- **R2#142 §507.3** 「slide 默认 pptx」→ user 澄清演讲用自己电脑。现值：HTML deck 为主。
- **R2#147 §511.5** 「登记标题可能还是 v8」（§511.1 CLAIM_UNVERIFIED）→ user 确认提交的是 v9。现值：登记标题 = v9.10 "When Is Expensive Perception Worth Paying For?"。
- **R2#152 §527.4** D24 演讲流程 → user 否掉，只活了一个晚上。现值：D25（opening → agents → demo → prize → question）。

### M23（新）. 静默仪器（4）

**定义**：工具、门禁或自检处在坏状态，却照样产出「正常的样子」：exit 0、空集上的 vacuous true、长期红着没人看的门、算出了不可能的值没人追问。

- **R2#12 §408.2** axis_effect_size 报告的全部否定结论 → 在空输入上生成，每个 contrast n=0。根因：`ModuleNotFoundError` 被 except 后返回空映射，exit 0；192 个 n_check 全是 pass:false 但没连到出口。现值：15 个 (metric, cell) 独立性组合。
- **R2#50 §454.3** 「deslopped.txt 的 ratchet 在保护 12 个文件」→ 条目指向已并走的路径，门一直红。*「an ignored gate protects nothing」*。
- **R2#118 §491.1** §490.4 two-arm nested policy 全部数字 → 阈值候选集缺 +inf，fallback 从 triage 脚本照抄了方向相反的 -inf；6/8 cell 的 observed saving 为负，*「当时照抄进笔记没有追问」*。现值见 §二 T4。
- **R2#127 §499.8** 海报 verify() 报的余量左 19.9mm / 右 9.0mm → PIL 量宽与 LibreOffice 排版差约 3%。现值：左 1.7mm / 右 0.1mm。

---

## 二、主题聚合：现在算什么

### T1. 重跑噪声地板 / band（最常被引、改得最多）

- **当前值**（各带 scope，禁止相加相减）：
  - B0·cls 六臂（n=224）：阈值 normal **3.52--4.15** / exact **3.12--4.46**；observed |ΔSR| **0.89--2.68**；SD **2.14--2.53**；set-difference floor **4.46--7.59**；discordance **10--14%**（§477.2）。六臂 discordance 写作 10.27–14.29%（§472.3、§498.3）。SoM 对口阈值 3.95/4.02（§477.2）。
  - B1：三臂地板 0.00 / 0.00 / 3.12%（§472.3）；episode 地板 2.00/4.00pp（§406）。
  - B0·red·Vision：discordance 4.93%（10/203）；API 组下界 4.93% 对 local 上界 3.45%，gap 1.48pp（§500.4）。
  - replicate 覆盖：B0·cls 6/6 臂 · B0·red 6/6 臂 · B1·cls 3 · B1·red 2 · B5·cls 1 · B1·wa-red 10-task×5mode；**no B2 cell carries a replicate**（§504.2）。
  - 同臂数比较（仅 cls·B0）：加一种不同看法 +7.14 vs 重跑一次 +4.46–7.59（§513.3；§477.4 caveat 同）。
- **演变**：§450.14 三臂（som 5.36–7.59pp，从 borrowed 变实测）→ §452.3 exact 4.02 / 4.02 / 4.46（3 个过阈 → 1）→ §474.9 速查页拆成两个 floor（写入 4.91-7.59 / 3.82-4.15）→ §477.2 六臂实况 → §480.3 / §504.2 覆盖扩到 5 格。
- **已作废**：「只有 vision 一格」（§450.14）· 3.82–4.15 及「3 个过阈」（§452.3）· exact 4.02--4.46 与 normal 3.82--4.15 的三臂 band（§477.2）· set-difference 4.91--7.59（§477.2）· 「exact 更保守」（§477.2）· 0.89-2.23 与「+2.23 恰好等于上沿」（§480.3、§482.2）· 12–14% three replicated modes（§498.3）· 「B1 地板恰好为零」（§472.3）· 5.9%（12/205）与 gap 2.49pp（§500.4）· 片子里「重跑一次买 2.0–7.6」（§513.3）· codex 的 4.91/4.91/5.36pp（§452.3）。
- **caveats**：*「band 是跨臂摘要, 具体效应按对口臂判」*（§477.2）；*「只有 cls·B0 有实测阈值 3.52-4.15pp, 其余七格不借不检验」*（§482.2）；d<10 的对照不得单独承载结论（§468.4）。
- **证据**：`docs/analysis/cross_sites/noise_floor_inventory.json` / `.md`；`final_dissertation/tex/appendix/b_noise.tex`。
- **原文**：§482.2 *「Reading a 2.2pp effect against a 2.23pp measured floor compares a draw to a draw. The main text uses the threshold, not the band.」*
- 相关待验：R2#164（2.0--7.6 → 0.0--7.6）、R2#165、R2#166，见 §三。
- 条目：R2#2 #43 #48 #63 #73 #86 #97 #102 #103 #104 #109 #126 #132 #137 #149。

### T2. 互补 / drop-one / 跨 side hero

- **当前值**：phantom arm drop-one **0.00–2.68pp / 14/18 为正**；全 6 mode 32/36 为正、0.00–4.46pp；P-prompt 只有 4/6 为正（§450.18，数据源 `fig0c_drop_one_bootstrap_ci.csv`）。C2 措辞：*「观察到方向一致的互补结构 … 但其幅度未超过重复采样地板」*（§450.10）。hero：**SoM 单臂**的不可替代覆盖跨两站稳健（cls 6 / red 4），视觉侧 vs 文本侧的整体表述**限定到 classifieds**（§500.1）；reddit 包络 n=203 时 Vision 2–5、P-prompt 2–4，下界与 separation 不变（§510.2）。P-SoM 6 个 off-segment（§409.5）。format / prompt 轴 SR 无主效应，task identity 有差异（§470.2）。canonical framing 只认 `section1_intro.md` + `TERMS.md`（§470.4）。
- **已作废**：1.7–3.3pp / 22/24（§450.18）· C2 作正面结构性主张（§450.10）· 「THE principal hero metric」（§470.4）· 「视觉侧整体 > 文本侧整体」跨站（§500.1）· 7 个 (metric, cell)（§409.5）· format 轴 +8.66pp（§470.2）· red_b0 Vision 2–6、P-prompt 2–5（§510.2）· `meta_phantom_lift.csv`（**不要再引**）。
- **caveats**：§450.10 的 C2 正确措辞里写的是「22/24 arm drop-one 为正」，但 §450.18 已把同一计数改为 14/18（phantom）/ 32/36（全 mode）——引那句措辞时数字要换。*「这是限定不是 hedge —— §496.4 意图书事前写死了这个落点」*（§500.1）。
- **证据**：`docs/analysis/cross_sites/unique_solve_envelope_cross_cell.md`；`fig0c_drop_one_bootstrap_ci.csv`。
- 条目：R2#21 #41 #42 #44 #67 #68 #131 #145；待验 R2#155 #174。

### T3. C1 oracle 上限与成本

- **当前值**：成本省 **1.6–35.3%**（cost-aware tie-break）；SR +3.45~+16.35pp、8/8 方向不受影响（§452.2）。「most of the ceiling is repetition」：SoM $U(6)\in[35.72\%, 54.33\%]$（$p\in[0.089,0.762]$），六跑拿走 headroom（16.07pp）的 52.8%-168.6%（§476.1）；异质 $p_i$ 不加宽区间（§477.1）；端点是 plug-in，bootstrap 下 >1/2 约 59%（§477.4）。arm gain 1.97-7.14pp，是 post-hoc 两臂 oracle-ceiling 增量（§409.1）。U(2) 残差（model−observed，SR 降序）`-1.12, +0.45, +1.34, +1.12, +0.67, +0.67`pp，最大 1.34pp（§477.1）。
- **已作废**：13.7–35.3%（§452.2）· 1.97-8.65pp（§409.1）· 「U(6) is an upper bound」（§476.1）· U(2) 是 out-of-sample check（§476.1）· 「异质更宽」（§477.1）· 残差 `+1.12, -0.45, …` 与「within 1.12pp」（§477.1）· 「at every p the data admit」（§477.4）。
- **caveats**：*「只引 59% 这个决策量, 不引那个两法不一致的精确区间」*（§477.4）；*「§4.7 真正硬的结果不受影响 —— one-arm like-for-like(+dom 7.14pp vs 同 mode rerun band 4.46--7.59pp, 落在 band 内)」*。
- **证据**：`docs/analysis/cross_sites/rerun_union_extrapolation.json`。
- 条目：R2#17 #47 #90 #91 #95 #96 #98。

### T4. C4 learned router / triage 可学性

- **当前值**：learned triage 真嵌套 CV 下 **0/8 cell** Pareto 胜过 always-cheapest（18 特征匹配集，跨两个 benchmark）（§450.3）。AUROC 按口径分别引：20 特征 VWA 六格 0.615–0.864，18 特征 8 格 0.526–0.723，clears best single feature in 5 of 8（§450.4）。Holm m=8，1/8 reject（§450.8）。置换：bundle-permutation，B=10000（§450.3）。权衡点：`wa_reddit·B1` +6.73pp/+41.7% · `cls·B1` +1.79pp/+36.1% · `red·B2` +2.46pp/+1.7%（§450.8）。learned(nested) ΔSR = +1.79/+5.42/+1.79/+3.94/−0.89/+2.46/+5.77，七格六格成功率赢、成本全输（§474.9）。which-mode：4/6 格未通过 min_class_n=10 准入规则；scale-up 2.1–4.2× 是下界（§452.1、§453.2）。triage 定义 = 预测是否有任何模式能解出（§474.9）。
- **two-arm（§491）**：修复后 Pareto-win = 1/8（对 cross-fitted 基线）或 3/8（对 whole-cell 基线）；多个 cell 精确塌成 always-cheapest。SR(two_arm) >= SR(triage) 有构造保证，cost 上的 8/8 是经验性质。AUROC(z) 低：稀缺与特征不足两个诊断都 open。
- **已作废**：0/6（§450.3）· B=200（§450.3）· 0.651-0.717（§450.4）· 「6 格族下没重算」（§450.8）· `cls·B0 +1.79pp 但 cost +11%`（§450.8）· 「nothing to fit」与全稿四处 trainable（§452.1、§474.4、§474.8）· 「structural」（§453.2）· 「neither on success nor on cost」（§474.9）· triage =「需要贵模式」（§474.9）· two-arm 的 0/8 与 §490.4 各格数字（§491.1）· 「8/8 压制是构造必然」（§491.2）· 「z 完整供给」（§490.4）· 「换标签换不掉稀缺」（§491.5）。
- **caveats**：*「这批特征分不开 z 编码的更细边界 … 稀缺 与 特征不足 是两个未分开的诊断」*（§491.5）。
- **证据**：`docs/analysis/cross_sites/router_triage_learnability_with_wa.md`（:124）；`router_label_supply_diagnosis.py:60,137`；`two_arm_action_learnability.py`。
- 条目：R2#35 #36 #37 #38 #39 #40 #46 #49 #80 #83 #84 #85 #117 #118 #119 #120；待验 R2#156 #157。

### T5. 「always-cheapest 是成本下界」这一句的死亡链

- **当前值**：*「cheapest fixed policy on average, not a per-episode floor」*（§476.4、§495）；摘要用 `no Pareto-improving operating point exists against always-cheapest under this cost boundary`（§476.4）；§450.12 caveat 里「在该成本口径下不存在划算的**可部署**点」仍成立，`可部署` 是承重词（§495）。learned 0/8 进赢区，hindsight oracle 1/8 进（§499.12）。
- **演变**：§474.3 改写误删 `over fixed policies` → §474.8 结论段第二份拷贝 → §476.4 Figure 6.1 图注 `necessarily` 与摘要 `profitable` → §495 海报 v2 从 §450.12 caveat 原样搬走 → §499.12 海报 caption `None does.` 与图矛盾。
- **已作废**：「always-cheapest 是成本下界 / per-episode 下界」（§474.3 / §476.4 / §495）· `necessarily`（§476.4）· `no profitable operating point`（§476.4）· `so does … in 7 of 8`（§474.3 / §474.8）· `None does.`（§499.12）· `IT DOES NOT WORK`（§499.10）。
- **原文**：§495 *「推翻发生在 §474.3/§476.4, 但被推翻的文字留在 §450.12 的 caveat 里 … 查台账的人查到 §450.12 仍会被误导」*。
- 条目：R2#77 #79 #82 #92 #93 #123 #124 #128 #130。

### T6. abstention

- **当前值**：零损失 **2.3–19.8%**、≤5% **6.0–24.6%**；措辞 *「overlaps but does not clear ≤30.6% (7/8)」*；嵌套阈值只保证内折达标，2/6 格（B2_cls / B2_red）外折超了 5% 预算（§467.4）。跨站迁移：真正成立的是**预算量子化**，24 个 nominal 预算档里 5 个是重复的（§468.2）。
- **已作废**：held-out 0.8–24.7% / 11.2–47.2%（§462.2，oracle-selected）· 「into and past」oracle 区间（§467.4）· base-rate 漂移机制与「排序最好的方向失败最惨」headline（§468.2）。
- **caveats**：§462.2 同节**未被推翻**：AUROC 0.615–0.864 / 6-of-6 cell 可拟合 / label 供给 2.3–14×。
- **证据**：`scripts/analysis/abstention_learnability.py`；`docs/analysis/cross_sites/abstention_site_transfer.md`；`master_bug_catalog.md#B-1971`。
- 条目：R2#56 #61 #62。

### T7. contested 集与重跑翻转集的重叠（`label_instability`）

- **当前值（六臂）**：86/224（38.4%）整格 flip；contested 88 tasks / 39.3% of cell；flip rate 81.8% vs 补集 10.3%；enrichment 7.9×；contested 承载 83.7% 的 flip（§504.3）。
- **演变**：两臂 49/224 · 51.1% vs 2.9% · 17.4× · 91.8%（产物 08-02）→ 三臂 67/224（29.9%）· 67.0% vs 5.9% · 11.4× · 88.1%（§464.2）→ 六臂（§504.3）。定性越来越强，enrichment 越来越低。
- **已作废**：两臂与三臂数字（§464.2、§504.3）；REALM 稿 §4 的「48–52% of those flip, against 2.9% of the rest」；Kimi 的「没有产物算过交集」（§464.1）。
- **caveats**：稿子的「three replicated arms」同时要改成六臂（§504.3）。
- 条目：R2#57 #58 #138；待验 R2#159。

### T8. B-1969 污染

- **当前值**：「77/4032 canonical episodes (1.91%) 记录了一次与 B-1969 一致的 reset-time timeout；全部一次重试后完成；**本分析未能识别其对 success 与 drop-one 的因果效应**」（§442.8）。
- **演变**：§442 CLAIM_UNVERIFIED → §442.7「污染真实存在但有界 — 78 episode / 1.83%, drop-one 期望偏移 <=0.45pp」→ §442.8 三家 /stress 推翻（Simpson；分层 p=0.2397）。
- **已作废**：「确有污染」、SR 5.13% vs 12.65%、p=0.024、78 episode / 1.83%、「156 次超时」、p=0.000。
- 条目：R2#28；待验 R2#153。

### T9. 融合 / SoM vs Vision

- **当前值**：SoM 对 Vision 溢价 6/8 严格为正，cls·B2 0.00，red·B2 -0.98（§482.3）。池化 task-clustered CI [-0.01, +2.91]，I²=59% / 77%（§409.4）。red_B2 SoM−DOM：underpowered，「所有被审的格含零」（6 格）；wa_red_B0 / wa_red_B1 未审（§468.4）。fig_fusion_forest 已撤（§482.2）。
- **已作废**：[+0.12, +2.75]（§409.4）· −2.96pp [−5.91, −0.49] 的「显著劣」（§467.3）· 「8/8 全含零」（§468.4）· 「7/8 为正」（§482.3）。
- 条目：R2#20 #60 #63 #109 #110。

### T10. grounding 通道与动作成功率

- **当前值**：element-id 两条路径合并 **64.2%**（13,421 次动作）；per-mode 动作成功率 P-text 75.0 / P-SoM 66.5 / SoM 61.8 / DOM 59.9 / P-prompt 57.7 / Vision 40.8；Vision 与最弱文本臂差 17.0pp（§503）。vision 491/491 用坐标，dom 511/511 与 som 436/438 用 element_id（§484）。SoM 标每个带 id 的 AXTree node，零 role 过滤（§478.1-2）。B5 vision 待 B-1997 重跑（§508.1）。并行 call 的 take-first 是确定性 wrapper 策略（§469.3）。
- **已作废**：caption 的「89%」与读出的 50.3pp（§503）· 「All modes ground through element identifiers」（§484，替代措辞**待定**）· 「interactable elements」（§478.1-2）· B5 vision 7.4 / 12.1%（§508.1）。
- 条目：R2#65 #100 #111 #135 #144。

### T11. proxy 价格、B4 与 GPT 接入

- **当前值**：**不存可引用的价格**，每次现跑 `probe_proxy_model_registry.py`（§478.5）。B4（Claude）落地 = 协议分支 + tool_choice 枚举 + schema 摊平，且摊平后语法约束强度与 B0 不同，§3.5 二分会变三档（§466.3）；config 侧另有 `cost_api` 与 queue 两处（§461.1）。GPT-5.6 = `response_format` json_schema（非 strict），无 logprobs（§471.5）。B0 reddit 单价 $21–23/格（§496.5）。
- **已作废**：「5 个 Anthropic 条目是死的」（§456.2）· sonnet-4-6 与 B0 同价及 $154（§456.2）· gpt-5.6 三档同价（§471.2）· §471.2 的替代价（§478.5）· 「改一行 config」（§461.1、§466.3）· B4 cls $136–272 作排期依据（§466.3）· tools 墙要去要（§471.5）· B0 $0.06/episode（§449.1）· 借 B5 的 $52/格（§496.5）。
- **caveats**：*「可达性是时变的, 每次排预算前重新拉 registry + 实测一发」*（§456.2）。
- 条目：R2#29 #31 #52 #53 #55 #59 #71 #72 #101 #125。

### T12. 预算、ETA、存储

- **当前值**：剩余实验费用估算 ≈ $546（§449.1），这是**需要花多少**不是余额；09-15 余额 $30.18（§515.1）。B1 shop ~5.6 天（§449.2）；B0 vision shopping 374/435，剩 61 个（§449.2）；B0 shop 每格 32-59 h（§515.1）。A100 scratch avail 278G，Phase 1b 真约束是 wallclock 与 B0 API 预算（§455）。
- **已作废**：$315（§449.1）· 374/466 与剩 92（§449.2）· 9.3 天（§449.2）· 41GB 装不下（§455）· $546 当余额（§515.1）· ~15 h（§515.1）· 「只够 2 格」（§496.5）。
- 条目：R2#32 #33 #34 #51 #150 #151。

### T13. 投稿、日期、匿名

- **当前值**：REALM notif 2026-09-07 / camera-ready 09-14（§470.9）；REALM 09-08 接受；毕设 2026-09-08 已交（119 页）；VLM4RWD 在审 notif 09-29（§502.4）。ARR 无 anonymity period，硬约束是提交件不得含非匿名 repo 链接（§475.8）。匿名性：四渠道审计 + `binary_probe()` 查路径片段（§473.8、§475.6）。REALM 提交稿 = `main_restructured.tex`（§504.1）。
- **已作废**：notif 08-21（§470.9）· 「08-21 A100 要空着」（§470.9）· 转 private / 09-12 后不 push（§475.2、§475.8）· pdftotext 判匿名（§473.8）· 「三件均已交付」（§502.4）· `main_realm.tex` 是提交稿（§504.1）。
- 条目：R2#64 #69 #70 #76 #87 #88 #89 #134 #136；待验 R2#161 #162。

### T14. 证据层与诊断指标自审（§407–§420）

- **当前值**：VWA 参考图目标 33.7%（§407.17）；WA 在用、两个 benchmark（§407.19）；碳采了但不报（§407.20）；◆ 标记按 UNADJUDICATED 注册表处理（§407.22）；visibility_gap 无信号（§407.22）；WA step JSONL 在 paper-grade host（§407.25）；跨站对比只认 WA-reddit vs VWA-reddit（§407.26）；axis_effect_size 改为 15 个 (metric, cell) 独立性组合（§408.2）；2x2 消融用 Tier 2b diamond 表，「还原端点」45/48 是恒等式不是交互检验（§408.5）；latency：ρ(cost, latency) 均值 -0.095，cheapest≠fastest 恰好三个 classifieds 格（§409.2）；P43 → visual_intent_routing（§419.3）；R1 = 22、R3 = WA 0 / VWA reddit 10（§411）；P49 3.61× 只在 WA（§420.2）；WA cascade 操作点暂挂（§409.3）；confidence 作为 cascade 失败检测器（§407.7）；B0 latency CV 与本地同量级（§407.15）。
- 条目：R2#3 #4 #5 #6 #7 #8 #9 #10 #11 #12 #13 #14 #15 #16 #18 #19 #22 #23 #26 #27。

### T15. 运维与基础设施

- **当前值**：shop canonical 启动是 orchestrator（FORCE_NEW=1），手工 queue_chain 默认会 glob-resume（§451.5）；watchdog marker 会截断 stderr，看完整输出（§469.5）；sanitize 的结构化字段未落盘、log 行在（§489.1）；订阅事故成因未确定（§489.2）；task 4 登出是 UI 陷阱（§489.3）；artifacts 在 A100 上完好，`results/` 下本地拉来的东西会被 sync cron 删（§501.1、§506.4）；DGX 可跑 arm64 classifieds（§506.6）；ratchet 已转绿（§454.3）；证据层 claim 矩阵已重写为四节（§460.1）。
- 条目：R2#1 #45 #50 #54 #66 #113 #114 #115 #116 #133 #139 #140。

### T16. 毕设排版、图、文字

- **当前值**：VLM4RWD 正文 8 页靠 fig_ceilings 移附录（§473.1）；UCL 模板两处 P79 FIX（§481.3）；按印刷宽度出图（§482.1）；fig_overview 竖排，全稿横排 0 处（§485.1）；float 用 `[!b]`，10→1（§493.2、§494.1）；全稿 jargon 8.05 → 7.34/100 词（§474.8）；配色统一 `_style.py`（§481.2）。文字修订见 M14 各条。
- 条目：R2#74 #75 #78 #81 #86 #94 #99 #105 #106 #107 #108 #112 #121 #122。

### T17. showcase 海报与演讲

- **当前值**：登记标题 = v9.10（§511.5）；演讲流程 D25（§527.4）；HTML deck 为主（§507.3）；口播上限 1,000 词、彩排 ≤ 9:30（§507.5）；缩放 100% + F11（§511.3）；截图侧称 LOOK and BOTH（§513.2）；screenshot = 动作前、obs_url = 动作后（§499.11）；海报余量左 1.7mm / 右 0.1mm（§499.8）；讲稿按 v9.10 六面板重写（§507.1）。
- 条目：R2#126 #127 #129 #141 #142 #143 #146 #147 #148 #149 #152。

### T18. frame 与定位

- **当前值**：领域主流是纯截图，DOM 是便宜退路，SoM 很少见（§413）；WA 与 VWA 的差异只主张任务规格（§448.5、§476.4）；frame 用 §5b。
- **已作废**：两个 frame 候选（§413）；WA 无视觉 grounding 需求（§448.5）。
- 条目：R2#24 #25 #30；待验 R2#158 #176–#180 #182。

---

## 三、CLAIM_UNVERIFIED（30 条）：缺什么证据，后来怎样

「批内后续」只写本批台账里能找到的后续条目；找不到写「批内无后续」。

| R2# | § | 待验说法（缩写） | 缺的证据 | 批内后续 / 现状 |
|---|---|---|---|---|
| 153 | §442 | cls Phase 1a 可能被 B-1969 污染 | cls episode 的 timeout 时刻与容器日志内部 POST 时刻对齐 | **先证实后推翻**：§442.7 判「污染真实存在但有界（78 episode / 1.83%）」→ §442.8（R2#28）撤回为「未能识别因果效应」，77/4032 |
| 154 | §446.3 | AWS proxy 按 `max_tokens` 预扣，解释 74 分钟盲区 | 余额接近空（<$1）但未空时跑 `--verify-reservation` | **证据基础被抽掉**：§446.7（R2#29）盲区实为 13分43秒（时区拼接）⇒ 假说降为纯防御性猜测；预留机制本身仍未测 |
| 155 | §448.5 | phantom drop-one 增益（1.7-3.3pp）是结构性的，不是 pass@K | 同 condition 重跑对照（`Best_Mode × 2 > Best_Mode + Phantom_Arm` 判据） | **推翻**：§450.10（R2#41/#42）对照早已做过，攻击成立，C2 *「does not survive as a positive claim」*；§450.18（R2#44）数字再降到 0.00–2.68pp |
| 156 | §448.5 | C4「routing 学不到是结构性的」 | 标签最多那格的学习曲线，或无正则化模型在训练集上也分不开 | **措辞被推翻、关键证据仍缺**：§452.1 / §453.2（R2#46/#49）降为「未通过准入规则」「specification，2.1–4.2× 下界」；学习曲线 / 训练误差检验批内无 |
| 157 | §450.9 | C4 的 0/8 是点估计，Pareto 序可能受噪声影响 | 每格 bootstrap dominance probability（约 1-2h） | 批内无后续（§491.1 改的是 two-arm policy，不是 triage 的 0/8） |
| 158 | §462.5 | codex 给的四篇相邻文献与据此收窄的 novelty 措辞 | curl arXiv API 核 ID 与内容；§315 五源措辞重跑零预设核 | 批内无后续（§505.17 核过的是另四篇 router 文献） |
| 159 | §463.3 | contested set 与 rerun-discordant set 高度重叠（REALM 结尾句与标题） | 三个复制臂上算交集 | **证实**：前提「没有产物算过」被 §464.1（R2#57）推翻；三臂 67.0% vs 5.9%（§464.2）→ 六臂 81.8% vs 10.3%、7.9×（§504.3） |
| 160 | §469.6 | G8 对 `error(code_bug)` 无 resolution 出口；`queue_chain.sh` 不查 marker ⇒ 两套 paper-grade 定义 | gate 要求的 matched-temporal-context 复现；修 gate | 批内无修复；§487.7（R2#171）再次引用「§469.6 已实证不查 marker」 |
| 161 | §473.7 | VLM4RWD topic fit 风险（gemini 判 reject） | 2026-09-29 notif | 批内无结果；§502.4 记「VLM4RWD 在审 (notif 09-29)」 |
| 162 | §473.8 | 在审 REALM 提交件带同一处 `/PTEX.FileName` 泄漏，后果未知 | reviewer 是否会 strings PDF；OpenReview 能否换件 | 泄漏本身已实证；批内无后果信息。§502.4 记 camera-ready 09-14 未做 |
| 163 | §474.8 | 附录 C 选 τ 的目标函数未写；1.35 / 2.09pp 哪个属哪个轴 | 查数据产物或作者确认 | 批内无后续；改写按「命名两个轴但不绑定顺序」保守处理 |
| 164 | §477.6 | 「one rerun buys 2.0--7.6pp」应为 0.0--7.6pp | 数据侧已实证（B1.cls.vision / som 的 d=0）；缺的是 framing 裁定 | **部分处理**：§513.3（R2#149）把片子/海报里的 2.0–7.6 换成 cls·B0 同臂数比较；毕设文字与 `fig_f8_oracle_ceiling` 未动（须一起改或一起不改） |
| 165 | §478.4 | B0 reddit P-text 地板 7.80% 真的低于 cls 带，不是站点漂移 | step-0 url_before 扫描排除漂移 | **证实**：§479.1 start_url_mismatch=0、205 个 step-0 landing 全一致 ⇒ 归 model nondeterminism |
| 166 | §479.6 | reddit 图像侧地板与文本侧（7.39-11.33%）同量级 | reddit 的 dom/som/vision replicate | **数据已补齐**（§500.1、§504.2：B0·red 6/6 臂）；量级比较的结果本批台账未给 |
| 167 | §480.1 | C1 分组来自 serving path 而非模型规模 | 同一 checkpoint 两种服务方式 | 批内无后续；不在算力包络内，限制已写进产物正文 |
| 168 | §481.2 | fig_sr_by_class 配色与全书相反 | 该图的生成脚本（不在仓库） | 批内无后续；处置 (a)/(b) 未选 |
| 169 | §482.5 | `_style.save()` 是死代码 | 已确认无调用方；缺的是修法裁定 | 批内无后续 |
| 170 | §483.1 | UCL / COMP0191 是否明文允许横排页 | COMP0191 handbook | 批内无直接后续。注（聚合者推断）：§485.1（R2#112）后全稿横排为 0 处，该问题已无适用对象 |
| 171 | §487.7 | 3 个 reddit GHOST 是 Phase B 有意产出的 replicate，应登记 CLEAN_PAIRS | 人裁定（改 canonical 噪声来源） | 批内无登记裁定 |
| 172 | §488.4 | R24364 若维持 2 波/33 ep，跑满 224 损失约 40 ep（18% 分母） | 该格跑满 | **速率前提被削弱**：§489.4（R2#116）2 波/33 ep 与前次无法区分（p=0.117）；最终损失批内无 |
| 173 | §489.7 | sanitize 表外字符 = 基建替模型纠错，应记 execution error | user 裁定（改 SR 定义）；跑完统计触发数 | 批内无裁定；§489.1 记触发 0 次 |
| 174 | §496.4 | 「跨 side 覆盖差异稳健」在 reddit 同样成立 | B0×reddit 五臂 replicate | **推翻（按事前写死的落点）**：§500.1（R2#131）Vision 下界 6→2、两侧接触 ⇒ 整体表述限定 classifieds，SoM 单臂跨两站稳健；§510.2 n=203 后下界不变 |
| 175 | §501.6 | DGX 上拉来的 artifacts 目录中途消失，原因未明 | 根因 | **查明**：§506.4（R2#139）sync cron 的 `--delete-excluded`；原「cron 仅三个任务」的排除理由是错的 |
| 176 | §505.6 | grounding dispatch 是唯一 per-step 标签稠密的路由轴 | 抽特征建模；动作级成功与 episode 成功的关系；范围裁定 | 批内无后续 |
| 177 | §505.10 | 「负结果预言自身在更强 agent 上反转」需收紧为「更强且非嵌套」 | 更多强格、>50% 的 agent、跨家族 | **证据基础部分作废**：§508.1（R2#144）把 §505.10 的 B5 union 51.3% / route-away 数标 pending（B5 vision 契约错配） |
| 178 | §505.12 | replicate 翻转对的第一分岔步可作 preference pair | 抽分岔步并验证可学 | 批内无后续；属另一篇 scope |
| 179 | §505.16 | 各家 CU 的观测形态（screenshot / AX / 混合），均无 cost-aware router | 读原始来源 | 批内无后续；不能当本项目证据 |
| 180 | §505.17 | 工业 router 三形态，CU 里无公开单独训的观测 router | GPT-5 router 与 Codex/Claude 训法的一手来源 | 四篇 arXiv 编号已核；其余批内无后续 |
| 181 | §511.1 | 主办方登记的海报标题可能还是 v8 | user / Zekun 确认 | **关闭（担忧被否）**：§511.5（R2#147）user 确认提交的是 v9 |
| 182 | §524.1 | GPT-6 Astra 的 computer use 只用截图 | Astra 发布页正文 | 批内无后续；片子只引 API 文档原话 |

**汇总**：被证实 3 条（#159 #165 #175）；被推翻 3 条（#153 先证实后推翻、#155、#174）；措辞被推翻但关键证据仍缺 1 条（#156）；前提被削弱 3 条（#154 #172 #177）；部分处理或数据已补齐 2 条（#164 #166）；关闭 1 条（#181）；批内无后续 17 条。

---

## 四、需要 user 判的（我无从验证）

- 本批的 user 纠正是推理链的输入：§413 领域默认是纯截图（R2#24）；REALM notif 日期两次翻转（R2#64 → R2#69）；§499.10 / §499.12 的海报措辞（R2#128 是 user 指出，R2#130 的 `None does.` 是 user 给出后又撤）；§507.3 / §507.5 / §511.5 / §527.4（R2#142 #143 #147 #152）。
- 明确留给人裁的待定项：§484 Vision 坐标通道的披露范围（R2#111）；§477.6 是否把 0.0--7.6 写进毕设（R2#164）；§487.7 CLEAN_PAIRS 登记（R2#171）；§489.7 sanitize 是否改 SR 定义（R2#173）；§505.6 grounding dispatch 是否算 paper 的 routing 主张（R2#176）；§482.5 `save()` 接上还是删（R2#169）。
- §476.1：GPT 建议把 U(6) 区间「降级为 sensitivity analysis 并停止使用」，原文判过保守、未采纳——这是取舍，不是对错。

---

## 五、本批的 meta 观察

- **副本问题压过计算问题**。M12 + M15 = 30 条，M14 = 18 条；而 M3（分母）0 条、M19（统计）6 条。数字算对之后，错误转移到「哪份文字跟着数字一起动了」。原文到 §450.18 自己数到*「第六次『引用比产物旧』」*，§504.3 又写下*「与被我批评的那个 bug 同形」*。
- **「全稿 grep」这条教训连写三次仍复发**：§474.4（R2#80）→ §474.8（R2#82、R2#83：grep 了短语没 grep 词根）→ §495（R2#123、R2#124：被推翻的文字躲在 MEASURED 条目的 caveats 字段里）→ §498.3（R2#126：讲稿）。说明作废的落点不止散文，还有**台账条目的 caveat 字段本身**。
- **作废的作废**在本批很常见，引用时要走到链尾：R2#10→#11（§407.25 的更正本身被 §407.26 推翻）、R2#60→#63、R2#64→#69、R2#71→#101、R2#87→#89、R2#48→#97、R2#86→#97（§474.9 写进的阈值次日过期）、R2#113 自我修正两次、R2#153 证实又推翻。
- **审计盲区有形状**：三家 AI 审「证据是否支撑 claim」，抓不到「系统描述是否忠实于代码」（§478.1-2）；内联式审计里没内联的产物会被读成不存在（§464.1）。这两类都需要换一个审计问题，而不是多审一轮。
- **M13 的代价不对称**（§501.1）：误判「有」最多白查一次，误判「没有」会去重新生产——本批两次差点因此占用正在跑 paper-grade 的主机（R2#133、R2#140）。

---

## 六、对旧结论层的 supersede

| 旧 § | 新 § | 一句话 |
|---|---|---|
| §4（:91，40% 参考图比例） | §407.17 | 40.0% 是旧数且多加了精度；现值 33.7%。 |
| §header 行动规划 + §97（CLAIM_UNVERIFIED「B1 完全确定性」） | §406 | episode 级 3/50 翻转；§298.2 的 step 级 133/133 仍成立，但不推出 episode 级。 |
| §302（「同模式重跑地板只有 vision 一格」，经 CLAIM_EVIDENCE_MATRIX 引用） | §450.14 | B0·VWA-cls 已有 3 个 replicated arm，后到 6/6（§504.2）。 |
| 旧 `retracted.md` §三 pooling estimand「FE inverse-variance (Decision 3A)」 | §409.4 | replaced_by 原话「且 FE 本身是错的 estimand」；改用 task-clustered bootstrap。台账本批**未给**替代 estimand 的正式裁定——两处并存，需人对齐。 |
| 旧 `retracted.md` §二「hero = drop-one oracle」与 §394「AUROC 0.65-0.72 in 5/6」 | §470.4 | 旧层已判死，但在 CLAUDE.md 与 `project_paper_hook.md` 里存活到 2026-08-19；本批清理，不是新推翻。 |
| §387.16.3（which-mode 标签稀缺） | §490.4 → §491.5 | 换成 two-arm 标签 z 没有解除瓶颈；但「稀缺」与「特征不足」是两个未分开的诊断，都标 open。 |
| 旧 `retracted.md` §六 待验「axis-2 (2.09pp) > axis-1 (1.35pp) 是否因跨 id-regime」 | §450.10、§474.8、§408.2 | 聚合者推断（台账未明说）：§450.10 判 1.35 / 2.09pp 低于 2.0pp 地板；§474.8 说这两个数的轴归属追不到；§408.2 的 axis_effect_size 报告曾在空输入上生成。该待验的数字基础被削弱。 |

---

### 六b、另一谱系补充：数字层的系统性 supersede

> 来源：2026-10-06 由 agy（`claude-opus-4-6-thinking`）对本批独立重做一遍聚合，其 supersede 表照录于下。
> 本节上表只收了「明文推翻旧结论」的条目；跨谱系对照（agy `gemini-3.1-pro-low` 审校）指出它漏了下表这类
> **数字层的成批更新**（C1 成本区间、C2 drop-one、C4 Pareto 与置换次数、噪声带六臂重算）。
> 核对：表中每个 § 都在台账里，每个数字都能在 `ledger.jsonl` 原文找到（脚本核对，2026-10-06）；内容未逐条人工复读。
> ⚠️ 最后一行（§394 AUROC）审校判为不准：§394 早已作废，§450.4 / §470.4 只是清理残留文本，不是本批的新推翻——见上表倒数第三行。

| 旧结论层 § | 新 § | 说明 |
|-----------|------|------|
| ≤§397 的 C1 成本区间 13.7–35.3% | §452.2 | 平局按列表序破平虚高; 新值 1.6–35.3% |
| ≤§397 的 C2 drop-one 1.7–3.3pp | §450.18 | 数据源过期; 新值 0.00–2.68pp / 14/18 |
| ≤§397 的 C2 正面结构性主张 | §450.10 | 幅度未超过重跑地板; 降为「方向一致但未过门槛」 |
| ≤§397 的 C4 Pareto 0/6 | §450.3 | 已扩到 0/8 (含 WA 两格) |
| ≤§397 的 C4 permutation B=200 | §450.3 | 已升到 B=10000 bundle-permutation |
| ≤§397 的 trainable 4/6 表述 | §452.1 | 改为「未达准入规则」; 30/30 fold 全拟合成功 |
| ≤§397 的噪声 band 3.82–4.15pp | §477.2 | 六臂后 normal 3.52–4.15 / exact 3.12–4.46 |
| ≤§397 的 band 0.89–2.23pp | §477.2 | 六臂后 0.89–2.68pp |
| ≤§397 的 discordance 12–14% | §477.2 | 六臂后 10–14% |
| ≤§397 的 label_instability 两臂值 | §464.2 → §504.3 | 三臂→六臂; flip 86/224, enrichment 7.9× |
| ≤§397 的 axis_effect_size 否定结论 | §408.2 | 空输入 vacuously true; 替代为 15 个独立性组合 |
| §394 的 AUROC 0.65–0.72 / 5/6 | §450.4 | 残留旧版; 18 特征 8 格 0.526–0.723 |

---

## 七、覆盖性闭合

- 本批条目 **182**，实际用到 **182**：RETRACTED 152 条全部在 §一（每条只归一个主模式，计数表合计 152）；CLAIM_UNVERIFIED 30 条全部在 §三。
- §二 主题覆盖：152 条 RETRACTED 全部至少挂在一个 T 下（T1–T18 末尾的「条目」行；一条可挂多个主题）。R2#29（时区拼接）挂在 T11，因为它作废的是 proxy 预扣假说的证据；R2#35、R2#41 分别挂在 T4、T2。
- RETRACTED 中未进 §二 主题的：**0**。CLAIM_UNVERIFIED 以 §三 表为准（30/30），其中 18 条另在相关主题里交叉引用，其余 12 条只在 §三。
- 未能归入任何主题、也未能归入任何错误模式的条目：**0**。
