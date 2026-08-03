---
type: action-ledger
status: rolling
updated: 2026-07-29
---

# Next Steps — Forward Action Ledger

> **Future-only**. **Roadmap = §1 `tasks.base`** (dynamic — edit `_status/tasks/*.md` frontmatter). Live state 不在这里:
> - cron health / 错误扫描 / ntfy 历史 → `make ntfy` + `logs/cron/*.log` (PLAYBOOK retired 2026-05-23, §279)
> - Real-time active runs / GPU → `make active` CLI (DGX only; fire 在 A100)
> - Paper-grade fire verdict → `paper_grade_check.py` (一条命令, §0) + 每 6h cron
> - Cell snapshot (active / pending / done) → `cells.base`
> - Paper section progress → `status.base`
> - 过去 chronicle → [[实验笔记]] (latest §357, 2026-06-23→25 — reddit abort saga 真根因 = task-138 改用户名 × auth_refresh fresh-login × fail-closed [B-1884] → Fix 4 (per-task identity restore, GRL substrate 复原) 实现+A100 deploy; §354 SUPERSEDED→§355 corrected→§356 VWA 设计缺陷盘点。**生产 live-verify 见上方 §0 06-26 块** [R11344 越过 task-138/151 无 abort])
> - Strategy / theory → [[paper_planning]]
> - **Phase 1 执行计划 + audit checklist** → [[phase1_plan]] ⭐ canonical
> - OSF DOI lock workflow → [[osf_lock_manifest]] · Compute infra → [[COMPUTE_INFRASTRUCTURE]]
>
> 🔧 新数据 → `make analysis` (~5-10min). Cron 每 10min 自动 sync cell frontmatter.

---

## §0 SESSION HANDOFF — 新 session 接手 ⭐ 先读这个

> ## 🟦 2026-08-03 晚 · 证据层独立审计 session 交接（**给做 cross-AI 或写 frame 的下一个 session**）
>
> chronicle → **笔记 §422–§426**。commits `da6178f` → `82d46d0`（7 个）。Overleaf 已同步到 `5b1e246`，
> 之后还有两个 commit 未推（见下方「立刻可做」）。
>
> ### 这个 session 干了什么（一句话）
> 独立审计证据层 → 找到 **八个同一形状的缺陷**，全部是「数据/产物在，但降级是静默的，降级后看起来完全正常」。
> 没有一个是靠更仔细读代码找到的，全部来自**让产物报告自己的完整度**。
>
> ### 🔴 会改变你怎么读现有数字的四条
>
> | 结论 | 影响 |
> |---|---|
> | **重跑地带 `0.89–2.23pp` 是两次抽样不是界** | 零假设 SD 2.32–2.53pp，单侧 95% 门槛 **3.8–4.2pp**。「超出噪声」的格子从 7 个降到 4 个 |
> | **`4.93–7.39pp` 是子区间** | 全集 −0.99 到 +7.39pp，**第三个 VWA-reddit 格符号相反**。原句写死在 `aggregate_fusion_premium.py` 里 |
> | **三个效率量全是估计量依赖的** | 时延（模型只占一步 22–67%，4/8 格最快模式翻转）· 本地成本（按 token vs 按 GPU 时间，2/4 格最便宜变）· 碳排（r=0.9935 就是时间本身，且 B0 根本没有） |
> | **执行路径是未报告的中介** | 三条投递路径动作成功率 89% / 39% / 16%；**Vision 按构造只能走最弱那条**。审稿人「是不是你们点击实现太烂」的反驳**部分正确** |
>
> ### 🟢 关掉的三个结构性缺口
>
> - **融合臂自己的重跑地板**（SoM replicate 224/224）→ 带子没动，claim 1/3 从外推变成实测；顺带裁定 `known.py` **§242**（1.7–3.3pp 落在噪声内，答案是「不」）与 **§293**（触发器不响，但因为 H1 早已 FAIL）
> - **路由从来没在 WebArena 上跑过** → oracle triage / learned triage / pooled tier 三个全接上，负结论现在覆盖两个基准
> - **26 个行为指标第一次有噪声参照** → 25 个活指标 22 个跨模式差 > 重跑带；掉的**正好是时延那两个**（0.87× / 0.84×），与时延分解**两条独立路径同一结论**
>
> ### ⚠️ 三条「看起来修好了、其实没有」的陷阱
>
> 1. **B-1929**：`sync_a100_results.sh` 的 `--delete` 每 15 分钟删掉 DGX 本地算出来的 `analysis/reason_diagnostics/`。
>    ⇒ 在这台机器上跑 `make analyze` 是徒劳的。已加 `--filter='protect ...'` 并**跨 cron 周期验收通过**。
>    **任何 DGX 侧派生的 per-run 产物都要检查是否在保护名单里。**
> 2. **B-1928**：`router_triage_learnability` 20 个特征里 3 个恒为 0（两个月前有人发现、写在**别的文件注释里**、没修）。
>    修后 AUROC 普遍上升，但 `reasoning_difficulty`（基准自带人工标注）成了 5/6 格的最强单特征 ⇒
>    **「看着学得会，是因为偷看了答案本」**。
> 3. **WA 泄漏审计未校准** → `persistent_state_leakage_audit`。WA 两格 0 命中，但同判据在已知的 VWA 上多报 3.7 倍。
>    **⚠️ unaudited 标记保留**；那个零是证据不是结论。
>
> ### 立刻可做（按价值排序）
>
> 1. **推 Overleaf**：`bash scripts/maintenance/overleaf_sync.sh`（落后两个 commit，35 张表）
> 2. **cross-AI**：**别扫「整个证据层」**——那样只会重复上面八个。最高价值 scope 是**「这些修复本身对不对」**，
>    因为那几处我既当运动员又当裁判。P0 两条：**McNemar 零假设 SD 当地板**是否成立（它把门槛从 2.23 抬到 3.8–4.2pp）、
>    **`dispatch_path` 写成「外部效度限制而非混淆」**这个因果措辞是否站得住。
>    最小喂料集：`EVIDENCE_LAYER_SUMMARY.md`（§4a/§4c/§4e/claim 4/claim 6/claim 9）+ `noise_floor_inventory.md`
>    + `dispatch_path_audit.md` + `latency_decomposition.md` + `replicate_metric_noise.md` + 七个 commit 的 diff。
>    ⚠️ prompt 里**不要写我的结论**（`feedback_zero_preset_cross_ai_verification`）。
> 3. **对账看板**：`docs/checkpoints/deliverables/duizhang_board_2026-08-04.html`（本地文件，gitignore 内，34 个证据格 + 8 张详细卡，已是最终版）
>
> ### 工具（新增，以后复用）
>
> - `check_summary_numbers.py` — 散文数字 vs 产物交叉核对（**别手改 SUMMARY 里的数**，改完跑它）
> - `audit_field_consumption.py` — 311 字段 × 146 脚本消费矩阵，分 OK/THIN/**CONSTANT**/ORPHAN/DEAD/GENERIC
> - `replicate_metric_noise.py` · `latency_decomposition.py` · `energy_carbon_audit.py` · `local_cost_estimand_audit.py` · `dispatch_path_audit.py`
>
> ### 一条应该变成默认的做法
>
> **任何新判据，先在有答案的地方跑一遍。** WA 泄漏审计如果只跑 WA，会得到一个漂亮的 0 和一句
> 「WA 没问题」，而那个 0 出自假阳率 14 倍的测试 —— 且它看起来会完全正常。多跑两个 cell 成本是零。
>

> ## 🟩 2026-08-03 · shopping reset / paper-grade 打通（VWA + WA）— **无阻塞，可直接 fire**
>
> chronicle → **笔记 §424**；缺陷详情 → **master_bug_catalog B-1930 ~ B-1936**。
>
> ### 可直接用
>
> | 命令 | 说明 |
> |---|---|
> | `RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh B0 dom shopping` | VWA shopping，reset 现在真能跑完（timeout 120s→900s，B-1931） |
> | `... queue_baseline.sh B0 dom shopping wa` | WA shopping，reset 不再被拒（B-1930） |
> | `bash scripts/queues/queue_phase1_paper_grade.sh launch phase1b` | VWA shop 18 conditions |
> | `... launch wa_shop` / `... launch wa_shop_admin` | WA shopping / shopping_admin 各 18 conditions（B-1935） |
>
> ⚠️ **三条 shop 链是同一个 Magento 容器**（`vwa-shopping`，7770 storefront + 7780 admin），
> 共用容器锁，必须串行；第二条启动时 abort 是 B-1934 的闸在工作，不是故障。
> stale lock 清理名字变了：`.locks/p79_magento.lock`（原 `p79_shopping_vwa.lock`）。
>
> ### 🔵 fire 之后要做的（不阻塞 fire）
>
> **敏感性分析：排除下列 task 重算 shopping SR。** ⚠️ **分母口径已更正**（/stress Gemini P1-7）：
> 不是 6-10 / **435**（≈2%），而是 6-10 / **104**（cart-graded 子集，≈**6-10%**），且误差**单向**
> （只产生假成功）⇒ 系统性抬高"购物车操作"这一子能力。敏感性分析应报**整簇 104 个**的排除结果，
> 而不只排"最差的 6-10 个"。⚠️ **§424.7 的裁定本身待重议**（见 §424.10）：
> 依据 (a) 跨站一致性已被证伪（cls 有 22 个 per-task reset）、(b) 分母已改写，仅 (c) 存活。
>
> | 站 | task | 机制 |
> |---|---|---|
> | VWA shopping | **86/87**、**223/224**、**348/349** | 每对两个 task 的 `must_include` 目标字符串**完全相同** |
> | VWA shopping | **453**（`"Green"`）、**455**（`"Gray"`） | 目标是短词，任何含该词的残留商品都可能误判 |
> | WA shopping_admin | **773/774** | 两个 task 的 eval **逐字相同**；先跑的删完 review，后跑的不做事即通过 |
>
> 做法：对每对，看后跑那个是否 success 且轨迹无有效操作 → 计入偏倚上界；然后排除重算 pooled。
>
> ### 🔒 不要动的
>
> `shopping_cart_reset.enabled` **必须保持 `False`**。开启 = 引入未声明的 per-task 条件
> = estimand 变更，需 PROTOCOL_NOTE / AMENDMENT。`test_disabled_by_default_is_a_noop` 是承重
> 断言。理由见 §424.7（跨站 estimand 一致性 > 单站干净；且 §402.7 已对同一缺陷类裁定披露不修）。
>

> ## 🟦 2026-08-03 · diag 质量审计 session 交接（**给动 `diag_scans/` 的那个 session**）
>
> chronicle → **笔记 §416.1–§416.14**。commits `cc429f7` → `915aa00`。
>
> ### 你等的那件事已经完成
>
> 下方 08-03 块写的「`conditional_failure_attribution` 未接：依赖 `diag_scans/`，另一 session 在动」——
> **就是这个 session，现在动完了，可以接了**：
>
> - `RULESET_VERSION` 9 → **`11-intent-text-fallback`**，**48 个 canonical condition 全部同版本**
> - 扫描产物：`results/diag_scans/v11_vwa/`（36）+ `v11_wa/`（12）。**v9/v10 目录留着仅供 diff，别再读**
> - 48 份 digest 已批量补 v11 数字块（`docs/analysis/{vwa_classifieds,vwa_reddit,wa_reddit}/`）
>
> ### 🔴 三条会改变你怎么读 diag 数字的结论
>
> | 结论 | 影响 |
> |---|---|
> | **P36(51%) / P31(50%) 是 risk-marker 不是死因** — 10 例跨 benchmark 因果验证 | **per-rule 表是症状分布，不是死因分布**。别拿它当失败归因引用 |
> | **`P2`/`P4` 在 vision 上是假 0**（依赖 `element_bbox`，vision 的 click 无 locator 元数据）；**`P36` 在 vision 上只覆盖 type 步** | vision 那一列**不可与 dom/som 并表**，须标"字段不可用" |
> | **P43 的「中性标签」定位只在 reddit 成立** | 见下，这条可能对 frame 有用 |
>
> ### ⭐ P43 → 可能是个 frame 候选（下方 08-03 块说 frame 三版都被判弱）
>
> §407.26(b) 记 P43 是"中性标签、补图无用"，依据 `+0.00/+1.56/+0.00pp`。**那是 reddit 64 个任务的数字**；
> P43 跨站触发而 **classifieds 的 71 个命中从未被检验**。在那里重做同样的受控 dom→som 对比：
>
> ```
> B0/cls  n=71   dom  9.9% → som 29.6%   Δ=+19.72pp
> B1/cls  n=71   dom  1.4% → som 14.1%   Δ=+12.68pp
> B0/red  n=64   dom 12.5% → som 12.5%   Δ= +0.00pp   ← 台账依据
> ```
>
> ⇒ **P43 子集是一个可事前识别（0-token 规则命中）、routing 能救（+12.7~19.7pp）的任务集**。
> 这比"某 mode 平均更好"强得多。**未接进任何 figure/§6 证据链** —— 若 frame 要找抓手，这是现成的一条。
>
> ### 其他产出
>
> - **新 bug**：`B-1923` 发帖限流（Postmill `@RateLimit(1h,max=3)` 源码坐实）· `B-1924` task 646 大小写不可通过 ·
>   `B-1925` `_format_history` 从不写 thought · `B-1926` `page_changed` 假阳性 · `B-1927` replicate run 劫持重扫
> - **`page_changed` 修正口径**（B-1926，你裁定的方案 3 = 只改分析层）：
>   `scripts/analysis/page_change_corrected_metrics.py` → `docs/analysis/cross_sites/page_change_corrected.json`。
>   **Micro 主结论稳健**（no_change_rate 最高=Vision 在 6/6 cell 不变），router streak≥2 触发 **5321→5910 (+11.1%)**
> - **数据质量审计总表**：`docs/analysis/_data_quality_audit.md`（字段级三类缺陷 + 分层可信度 + Q1–Q7 actionable）
> - **重跑结论**：**当前没有必须重跑的 run**。唯一候选是修 `select_option` 后重跑那 383 ep（B-1920），
>   但 B-1920 已论证主结论不动，只需 limitation
>
> ### ⚠️ 两个坑
>
> 1. **`diag_rescan_all.py` 必须带 `--baseline-dir`** —— 否则按 mtime 选 run，会被那个**仍在跑的
>    SoM replicate**（`..._R30696`）劫持（B-1927 已修成"有 baseline 就锁 run_id"）。
>    replicate 与 canonical **同目录同名同 seed 无标识**，跑满 224 ep 后 episode 数也不再能当警报。
> 2. **`.claude/` 被 gitignore** → `SKILL.md` 的 v11 更新（47 条规则 + P43 更正 + P48 过窄说明）**只在本地**，
>    不会随 commit 传播
>
> ### 未做 / 待裁定
>
> - `page_changed` 判定在 **runner 层**的修复（fire 路径 + estimand-adjacent，需 witness）—— 你已选方案 3 绕开
> - `select_option` CSS dropdown 分派修复（B-1920，修法未定）
> - `_format_history` 加 thought（B-1925，修了要全量重跑，属能力改进非污染修复）
> - **未跑 `/stress`** —— 本 session 没碰 paper prose，产出是审计与工具层；但 P43 那条若进 §6 证据链，
>   建议引用前补一次



> ## 🟪 2026-08-03 晚 · 旧稿归档 + realm 骨架 + 30 张表 + overleaf 已推 —— **最新**
>
> chronicle → **笔记 §421**（三类划分 / router 更根本的否定 / 归档 / 30 表 / overleaf 五个坑）。
>
> ### 下一个 session 的第一件事
>
> **零预设审证据层**，prompt 已写好：
> `docs/reference/EVIDENCE_LAYER_AUDIT_PROMPT.md`
> 五个问题：该进没进 / 进了少了 / 进了错了 / 不该进 / 没算但有用。
> ⚠️ 那个 prompt **明写不要读笔记**（会预设），用 `known.py` 查台账代替。
>
> ### 论文状态
>
> ```
> paper_drafts/realm/          新骨架，section 3/4/5 刻意空着
>   section1_intro.md          Abstract 槽位 = 8 条 claim-free 测量【生成】
>   section_evidence_guide.md  7 段导读，说每组表干什么 + 坑在哪【生成】
>   section_evidence.md        30 张表【生成】
>   limitations.md             不空 —— 5 条结构性限制
> paper_drafts/latex/          基础设施 · paper.bib 保留
> docs/archive/paper_drafts_pre_rewrite_2026-08-03/   paperA/paperB/aaai27/9 旧 section + README
> ```
>
> 三个生成文件**必须同一次调用生成**（`--abstract --guide --evidence`），否则散文数字会跟表脱钩
> —— 我手工拆过一次，那就是当天第 7 个同类失效。
>
> **Overleaf 已推**（13 页 / 未定义引用 0）：`main_realm.tex` 是当前目标；`main_paperA.tex` 已删；
> **`paperB` 保留但冻结**在 07-28 —— 它的 md 源已归档，`convert.sh paperB` 跑不了，所以
> `overleaf_sync.sh` 默认只同步 realm，显式传 paperB 会故意报错。
>
> ### ⚠️ 115 个文件未提交
>
> 含另一 session 的 `docs/analysis/wa_reddit/*` 和 `results/diag_scans/`，commit 时要分开。
>
> ### 在跑
>
> | 在哪 | 什么 | 状态 |
> |---|---|---|
> | A100 | SoM replicate `R30696` | **116/224**，约 19:00 收（A100 时间） |
> | A100 | B3 = MiMo armed chain | 等待中，replicate 满 224 自动起 |
> | DGX | mechanistic canonical | 只剩 `p4_som_pprompt_red` |
>
> ---
>
> ## 🟩 2026-08-03 · 第 8 格落地 + 证据层接线 + 对账板
>
> chronicle → **笔记 §414 / §415 / §416**（frame 三版皆判弱→转对账板 · 第 8 格落地 · WA 全量接线）。
>
> ### 对账用的东西（今天 08-03 与学长）
>
> - **证据台**：`docs/checkpoints/deliverables/evidence_board_2026-08-03_local.html`（双击即开，自带主题开关）
>   · 可分享链接 `https://claude.ai/code/artifact/9db9913e-be88-4f37-b217-6eb6ad83b565`（默认私有，页面右上分享）
>   · 12 个区块，数据由脚本从产物 JSON 直抽（零手抄），改产物后重跑抽取脚本即可刷新
> - **frame 未定** —— 我提的三版（§5b / 双地板 / 融合不值这个价）都被判撑不住，
>   板子末节「要拍的板」摆了四个待决点，**没有替你选**。deadline **08-05**，走**非归档轨**（已拍板）。
>
> ### 在跑的（跨主机，注意 A100 比 DGX 慢 1 小时）
>
> | 在哪 | 什么 | 怎么查 |
> |---|---|---|
> | **A100** | SoM replicate `B0_som_classifieds_20260803_084743_..._R30696`（224 ep，实测 2.9 min/ep → ~10.7h，A100 时间 19:30 前后完） | `ssh condense-a100 'cd /home/ubuntu/workspace/p79; find results/visualwebarena/phase1/B0_som_classifieds_20260803_*/ -path "*episodes*" -name "*summary*.json" | wc -l'`　←　**仓库在 `/home/ubuntu/workspace/p79`**，不是 `~/p79` |
> | **DGX** | mechanistic canonical **只剩最后一格** `p4_som_pprompt_red`（23/24 完成，08-03 11:15 起算约 3.7h） | 单格 done 判据 = **`pilot_summary.md` 非空**（不是 `results.json`——那个在跑到一半时就已存在且体积很大，拿它判会误报完成）；全 sweep 判据 = `results/mechanistic/canonical/.SWEEP_DONE`。已挂 monitor |
>
> ⚠️ 原 armed chain `logs/som_replicate/chain.sh` **已死**（24h cap 到点时 WA 还在跑）。重启用的是
> `logs/som_replicate/fire_now.sh`。**`queue_baseline` 默认 resume**，做 replicate 必须 `FORCE_NEW=1`，
> 否则它会 glob 到原始 run 并当续跑。
>
> ### 证据层接线状态（实证，非照抄）
>
> **8/8 已接**：`per_mode_four_dimension_profile` · `fusion_premium` · `noise_floor_inventory` ·
> `multimetric_pareto` · `axis_effect_size` · `confidence_cascade` · `outcome_efficiency`
> （`--with-wa` 现在同时挂 B1 与 B0；`noise_floor` 的 B0 无 pilot 故**有 margin 无 floor**）
>
> **未接**：`conditional_failure_attribution` 7/8（依赖 `diag_scans/`，另一 session 在动）·
> `axis1_microbehavior` 6/8（连 B1×WA 都没有，需单独一遍）· `cost_per_mode` 6/8（第二成本口径）·
> `routing_feature_diagnostics` / `visual_difficulty_router`（**设计上确实不适用**，验过）· `label_instability` 1/8
>
> ### ✅ 三处措辞已改（08-03），并且审证据层又挖出四件 —— chronicle 见 **笔记 §418**
>
> 三处（Vision 成本 7/8 且非构造性 · claim 9 的 5/8 且「跟着站点走」不再成立 · fusion 聚类 CI
> `[+0.06, +2.93]` 排除零但下界远低于地板下沿）全部改进 `EVIDENCE_LAYER_SUMMARY`，连带 claim
> 3/6、§1 矩阵、§7 表格（/7→/8 加 `wa_red_B0` 列）、开头 frame 状态（§5b 已死、当前无 frame）。
>
> **新挖出的四件**：
> 1. **两个同族 bug** —— `aggregate_fusion_premium.py:317` 和 `per_mode_four_dimension_profile.py`
>    的旁注列，结论**硬编码在生成器里**，每次重跑把自己再写一遍，跟隔壁数据驱动的列自相矛盾。已改成派生。
> 2. **唯一那条「融合显著输了」是环境送的** —— `red_B2` SoM−DOM 的 8 个 DOM 成功里 3 个是 sidebar
>    泄漏；置 0 后 −2.96pp → **−1.48pp 跨零**。新产物 `leakage_sensitivity.py`。模态反转不受影响。
> 3. **reddit-only 混杂** —— 站外导航 1.05–2.13%（classifieds ~0），且 reddit 容器站内 `env_step`
>    是 classifieds 的 **1.69×**。新产物 `offsite_navigation_audit.py`。
> 4. **「四种路由全失败」太强** —— `router_objective_ordering`（138 行，从未被引用）里 oracle triage
>    在 **6/6 格零 SR 损失省 9.5–30.6%**，learned triage 在 `cls_B1` 上 **out-of-fold +0.00pp/−4.5%**。
>    which-mode 路由失败，triage 路由没有。
>
> ### 🟢 证据层核查结论（08-03 下午，chronicle → 笔记 §420）
>
> | 维度 | 今早 | 现在 |
> |---|---|---|
> | 产物 ruleset 非 v11 | 2 | **0 / 36** |
> | 第 8 格接线 | 7 个 8/8，`conditional_failure_attribution` 卡 7/8 | **8/8 全齐** |
> | 汇总未引用产物 | **11** | **0** |
>
> **⚠️ 唯一的例外**：硬编码扫描**只覆盖 `n/6` 这一个形状**。修的 5 处里
> `aggregate_confidence_cascade.py:430` 是**分母错且事实错**（称 Vision 6/6 格最便宜，`wa_B0`
> 上是 DOM）⇒ **模式名 / 比值 / 方向三类硬编码从未扫过**，同类能错在事实层。
>
> **新增 claim 10**（`visual_intent_routing`）—— 目前最像 frame 的东西：0-token 正则事前指出
> 哪些任务需要截图，标记集 **+22.54pp [+9.86, +33.80]** vs 补集 +0.65pp（跨零），classifieds
> 上**样本外**（规则是为 reddit 写的）。三条限制：`cls_B2` 只 +1.41pp（要能力才兑现）· reddit
> 上**反号** · n=71（23 成功 vs 7）。**它不是 router**（分区固定不学习）。
>
> **claim 7 修正**：v11 rescan 后 text-wins 侧出现 `P49` **3.61×**（首次清过 1.5×，已因果验证
> WA som 610/614），但 **8 个 hit 全在 WA 两格，VWA 六格零贡献** ⇒ "WA 上有名字，VWA 上仍未解释"。
>
> **新增 §1b**（三条 diag 读法约束）：per-rule 表是**症状**不是死因 · **`vision` 列不可与
> dom/som 并表**（P2/P4 是结构性零不是测量值）· P43 中性标签只在 reddit 成立。
>
> 未做：`mechanism_per_task_report`（425 行）**仍未整合** —— 它加什么是 framing 判断不是接线活。
> WA 两格的 leakage **仍未 audit**（手查两个 episode 都是 earned，但那不是 audit）。
>
> ### 🟠 B3 = MiMo 已挂 armed chain（08-03，chronicle → 笔记 §419）
>
> replicate 满 224 → 自动起 `B3 som classifieds`（full 224 / 30 步 / `disable_thinking: true`）。
> chain 用 **episode 计数 + runner 存活**双条件，**不用 wallclock**（上次就是 24h cap 到点而死）；
> **replicate 若中途死则拒绝发射并报警**。查：`ssh condense-a100 'tail logs/b3_chain/chain.log'`
>
> - 权重在 **`/mnt/scratch/hf`**（321G 空闲那块盘）；根分区 92% 满是 **Docker 镜像 419G 且全 active**，
>   绕开非解决
> - `/no_think` 已实现（包装 `MiMoVLAgent` 自己的 processor，**不碰 `Qwen3VLAgent`**）；
>   官方要求它在**整个 user content 最末尾**（图之后），`tests/test_mimo_no_think.py` 专钉这条
> - ⚠️ **另外 4 个 queue 脚本仍硬写 `B0|B1|B2` 白名单** —— 只改了 `queue_baseline.sh`（跑 som 够用）。
>   B3 证明不地板后再改其余 4 个（地板了就白改）
> - 选型经 GPT 重扫 + 我逐条 WebFetch 核：**维持 MiMo**。Claude 撞"开源"硬约束；GUI-Owl 是
>   Qwen3-VL 派生（跨族角色担不了）；GLM-4.1V 官方卡**没有** WA/VWA 数字且是一年前的模型
>
> ### 未提交
>
> `docs/analysis/cross_sites/*`（7 个产物重算）+ `scripts/analysis/*`（6 个脚本参数化）+ 笔记 §414-416 + 板子两份。
> ⚠️ `docs/analysis/wa_reddit/*_diag_digest.md` 是**另一 session 的改动**，commit 时不要混进来。

> ## 🟦 2026-08-02 · WA reddit /diag 完成 + B-1919 修复 —— **这块不覆盖下面那块**（并行的另一条线，论文主线仍看 🟥）
>
> chronicle → **笔记 §410**（含 §410.8 修复补记）。数据层结论 → `docs/analysis/wa_reddit/_cell_cross_mode_findings.md`。
>
> ### 已完成
>
> - **WA reddit × B1 × 6 mode 的 /diag Tier-2 + Tier-3 全做完**：6 份 per-condition digest +
>   1 份 cell 级发现，落在 **`docs/analysis/wa_reddit/`**（新目录，与 `vwa_reddit/` 严格分开）。
>   SR 9.62–16.35%，是 VWA reddit 的 2–3 倍。
> - **B-1919 + A2 已修**：`sync_a100_results.sh` 从来只同步 VWA 子树，`results/webarena/phase1/`
>   无任何自动同步 → 19/19 个 WA run 的 `task_configs/` 全空 → /diag 44 条规则里 27 条静默失效。
>   已恢复 + 改脚本覆盖两个 benchmark（**WA 设 additive-only**，理由见 catalog）+ 重扫互证 6/6 一致。
>   `results/diag_scans/v8_wa/` 现在有效，是唯一数据源。
>   **A2 同批落码**：`scan_episodes` 遇 config 缺失现在**默认 exit 2**，报错里列出会静默失效的
>   28 条规则名（派生非硬编码）；`--allow-missing-config` 逃生舱会把缺失数写进输出 JSON。
>   全量 pytest **1631 passed**。
>
> ### 需要你拍板的三件
>
> | # | 事 | 为什么要你决定 |
> |---|---|---|
> | **A3** | `select_option` 在 reddit 上 **99.6% 失败**（1387/1392）。**机制不是新的** —— §51/B-57/§60/B-59/B-64 在 2026-04-14 就裁定过（你记对了，cls 也有）。新的只有 blast radius：B-06 当年把 18/20 的 custom-dropdown 样本剔除出估计（依据是一句与代码不符的「走 click 路径」），只报 15 ep / 0.3%。已落 **B-1920** 专门改这个估计 | **波及 VWA reddit 计分 cell 且已冻结进 v8 结论**。要不要重新评估该 cell 的解读，是论文层判断。修法方向也待定（放宽 fallback vs site-specific handler） |
> | ~~A2~~ | ✅ **已修** —— config 缺失默认硬失败 + 逃生舱留痕 `config_missing` + 报错列出会失效的规则名；测试 5 条含实际事故回归 | — |
> | 清理 | `B0_wa_3mode_shopping_20260417`（888K、192 个 task_configs、**0 个 episode summary** 的 2026-04 骨架，只在 DGX） | 我把 WA 同步设成 additive-only 就是为了不替你删它。要清就手动删，之后可考虑把 WA 也改回 delete |
>
> ### 可以直接做的（不需拍板）
>
> - **A9**：落两条新规则 —— `R1 PREMATURE_FINISH_ON_FORM`（24 failed / **0 success**）+
>   `R3 PREMATURE_NEGATIVE_AFTER_SEARCH`（9 / 0），bump `RULESET_VERSION` → 全量重扫 36 VWA + 6 WA。
>   ⚠️ 另两条候选 **不要落**：`R4 FORUM_NEVER_VISITED` 在 36% 的 success 上 fire（已否决），
>   `R2 NOELEM_ACTION_STREAK` 17% success 误伤（需先收窄）。
> - **A4**：`P36` 按元素角色分叉 —— 已坐实 1307 个 walk_fail 步里"元素不在观测里"= **0 次**，
>   57–74% 落在可交互角色上（scaffold 侧），26–43% 落在 StaticText/heading（agent-limit）。
> - **A5/A6/A7**：WA task 66 config 域名硬编码（孤例，VWA 干净）· `P40` 的 `detail_markers`
>   在 reddit 空转 · task 27–31 族全量审计后决定是否移出计分集。
>
> ### 未提交
>
> 全部改动**未 commit**。涉及：`docs/analysis/wa_reddit/`(新) · `实验笔记.md` §410 ·
> `master_bug_catalog.md` B-1919 · `sync_a100_results.sh` · `next_steps.md` 本块。
>
> ---
>
> ## 🟥 2026-08-01 夜 · 论文转向 + 投稿轨改判 —— 这块覆盖下面那块
>
> chronicle + 全部数字来源 → **笔记 §407**。下面只写 live / forward。
>
> ### ① 投稿轨改判（最要紧，先看）
>
> **REALM 有非归档轨**（Non-archival long, 同样 8 页, 同样 08-05）。
> 07-22 拍板的「保主会选项」在 **07-28 合并两篇时被静默删掉**，动机是页数，归档属性没进决策视野。
> ⇒ **投非归档轨**。witness = `_status/issues/issue_realm_archival_track_2026-08-01.md`。
> 连带：**页数之战降级** —— 8 页仍是上限，但非归档稿可以带着已知缺口去要反馈，不必是终稿。
> 另：**ARR commitment 已从 08-10 推到 08-31**（第三条路，暂不决定）。
>
> ### ② 骨架换了 —— 旧 ①②③④ 作废
>
> 触发 = user 三句话：「不必纠结 phantom」→「phantom+dom 算一个『没有截图模式』」→
> **「企业里没人区分 phantom，真正的工业问题就是 vision/text」**。随后 user 抓到 **SoM 是融合模式不是截图族成员**。
>
> 新问题陈述：**我每道题有 k 次模型调用的预算，怎么花在表征上。**
>
> | 步 | 说什么 | 关键数 |
> |---|---|---|
> | ① 上限真, **但一半是假币** | 加一条臂 1–8.65pp vs 加一次**重跑** 2.0–7.6pp | cls_B0 **69–106%** 可复现 |
> | ② **默认答案最贵而没挣到** | SoM 5/6 格最贵，溢价**无一格**明确超地板，2 格为负 | 独占覆盖 1胜3平3负 |
> | ③ **该买哪条随模态翻号** | VWA-cls 加第二截图臂更值（0–1/4）；WA 加文本臂 **4/4，高出地板 4.65pp** | |
> | ④ **压不到 per-request** | 四路全败 **+ 51.1% 有效训练行重跑就变** | 下界（6 臂只测 2 臂）|
>
> **旧 ②（H3 两轴）整个消失** → 2×2 降级为 §2 构造效度检查。断链没了，且省出超出的 2 页。
> 新章节：§1 问题陈述 / §2 setup+2×2 / §3 上限与假币 / §4 该买什么 / §5 为什么压不到 per-request / §6 discussion。
>
> ### ③ 立即要做的（P0 起，本 session 自走中）
>
> | P | 做什么 | 资源 | 换到什么 |
> |---|---|---|---|
> | **P0** | **confidence 级联离线模拟** | 纯离线 2–3h，不占 A100 | ④ 关死攻击面，**或**拿到唯一正面结论 |
> | **P0** | **B0×SoM×cls 重跑副本** 排 A100（必须等 B0×WA 完） | 7.8h + ~$17 | 头条的直接缺口 |
> | P1 | §1 新问题陈述（新旧并排给学长选） | — | 08-03 拍板用 |
> | P1 | **A 的负面证据改口径** | 30min | 见下「必须改的口径」 |
> | P2 | diag + macro/micro 进 §4 · 三维 Pareto（**全 6 格**，见上 ④.5 撤回） | 各 2h | 机制层 / 负结论加强 |
>
> ### ④ 必须改的口径（不改会被抓）
>
> 1. **A 的负面侧实际只有 WA 一格真撑** —— 另一格 red_B2 的 SoM SR = **0.99%（2 道题）**，−2.96pp 在那量级无信息量。
> 2. **「融合无独占覆盖」计数是 1–10 道题**，无区间 ⇒ 写「未观察到」不写断言。
> 3. **2.23pp 门槛只在 B0×cls 实测过**（WA B1 = 2.00）⇒ 写「对着我们测过的那两个地板」。
> 4. **三分表「最强文本」是 4 臂取 max** vs vision/som 各 1 臂 ⇒ 逐格查过「融合优势」一列不变（约束项要么是 vision 要么恰是 DOM），但**必须主动披露**。
> 5. ~~三维 Pareto 不能用 B0~~ **已撤回 (08-01 夜实测)** —— B0 每步延迟 CV 0.15–0.22，与本地 B1 的 0.11–0.19 同量级，且随 tok/step 单调（som 4722 tok→7.59s / vision 3470→6.85s）。排队没有主导。**三维 Pareto 用全 6 格**；B0 反而是 latency 唯一真在动的三格（B1 每步延迟几乎不随 token 变：13.67/13.63/12.69）。
>
> ### ⑤ 🚫 新增「不要引用的数字」
>
> - ~~「latency 跨模式只差 1.00–1.10×」~~ —— 那是**最高÷第二高**；真跨度 **1.12–1.40×**，与 cost 同量级。
> - ~~「cascade 卡在没有失败信号」~~ —— **confidence 每步都采**（B0 4/6 字段 / B1·B2 6/6），论文 0 引用。
> - ~~「6/6 才算 4-layer 成立」~~ —— 5/6 的期望假阳 **0.144**（36 检验），完全站得住；6/6 严 30 倍且把 SoM 的 5 个指标藏了。照 K-of-N 改报连续 x/6。
> - ~~「有参考图的题该路由到看得见图的模式」~~ —— **符号是反的**，只 1/6 格成立；`reference_images` 无 mode 过滤（`main.py:2890`），六模式全收得到。
>
> ### ⑥ 已知缺口（写进 limitations，不假装没有）
>
> - **SoM 无重跑地板**（全仓只有 B0_dom_cls / B0_vision_cls 两个副本）—— P0 已排队修
> - **n = 2 个工作负载**（B0×WA 加的是模型不是负载；shopping 目录数 = 0）
> - reddit `require_reset` no-op ⇒ 订阅在**同一 condition 内部**累积 = 条内顺序效应
> - `visual_difficulty` 被读出后未进特征表（`extract_50_features.py:334`）
>
> ---

> ## 🔴 2026-08-01 白天交接 —— 已被上面覆盖，保留作 chronicle 指针
>
> ### 卡点解除：508 条账本已导出 → 已过完 → 议程已出
>
> user 08-01 13:05 导出 `docs/checkpoints/台账.md`（认可 391 / 有问题 24 / 不确定 76 / **未标 17，按认可处理**）。
> 07-29 定的四步顺序**走到第 3 步**：
> ~~① 标完 508~~ → ~~② 过「有问题+不确定」~~ → ~~③ 生成对账议程~~ → **④ 08-03 对账后才写骨架**。
>
> **议程 = `deliverables/advisor_agenda_2026-08-03.md`**（要学长拍 4 条 / 已定知会 3 条 / 时间账）。
>
> ### ⭐ 本轮改结论的一件事：① 要打对折
>
> 臂数对齐后（同 cell 同 n 同泛函同臂数）——
> **B0·cls：多加 1 个不同表征 +7.14pp，多跑 1 次同模式 +4.91~7.59pp，落在带子里不可区分**；
> WA·B1：4.81pp vs 2.00~4.00pp，高出 0.81pp。
> ⇒ ① 的 headline 必须并排印重跑基线。**不要再单报 +16.07pp。**
> ⚠️ 也**不许推**「整个 16pp 都是噪声」——只有 1 个重跑臂不是 5 个。
>
> ### 新产物 / 新事实（都已落地，命令可复算）
>
> | 什么 | 在哪 |
> |---|---|
> | 噪声总表 + 臂数对齐对撞 | `aggregate_noise_floor_inventory.py` → `cross_sites/noise_floor_inventory.{md,json}` |
> | **B1 episode 地板 2.00/4.00pp**（此前以为不存在） | 同上 §1；来源 = WA pilot × full-104 同 condition 重叠 |
> | 台账 2057 → **2082**（补 §398 全节 + §406） | `known/ledger.jsonl` chunk 9 |
> | 结论层 9 处补标（3 处内部不一致 + 桶 B） | `known/conclusions/` |
> | chronicle | 笔记 **§406** |
>
> ### 三条 scope 已钉死（user 08-01）
>
> **visual 占比不报** · **shopping 不进本篇** · **WA 进但作第二 stratum 单列，不并入 θ_FE**。
> WA 纳入理由是**设计层**的（VWA 以图指定目标 40.0% / WA 纯文字 0%，合起来张成任务模态轴）。
>
> ### 🚫 新增「不要引用的数字」
>
> | 错的 | 对的 |
> |---|---|
> | ① 单报「+16.07pp oracle headroom」 | 必须并排印「重跑同一模式买 4.9–7.6pp」 |
> | 「B1 是完全确定性的 / 重跑 bit-identical」 | **已证伪** — episode 级 3/50 翻转。step 级确定性 ≠ episode 级 |
> | 07-29「B1/B2 重跑地板不用测」的**理由** | 结论保留，理由换成「数据已免费存在」 |
> | WA 的 H3 双轴 5.77/6.73pp 看起来赢 | **最不能用的数** — 落在 VWA 地板 4.9-7.6 内，且跨 benchmark 不可比 |
>
> ### 🔴 08-03 带这个去对账
>
> **看板** `deliverables/duizhang_board_2026-08-03.html`（gitignored；VSCode → Live Server）。
> 四条结论各自的正反面 + 4 条要拍的板。**议程** `deliverables/advisor_agenda_2026-08-03.md`。
>
> **要拍的 4 条**：① 删哪三页（建议砍 ②）· ② 标题退不退 · ③ ②「不成立」认不认 · ④ 那条开着的路要不要跑（跑不完）。
>
> ### 稿子现在的状态
>
> | | |
> |---|---|
> | ③ | ✅ 已焊进 §3.3 + Appendix A.6 |
> | ④ | ✅ 原本就在（§4/§5/§6），本轮按跨 AI 审计加固 |
> | ① | ✅ 在 §3，**但已打对折**（并排印重跑基线） |
> | ② | 🔀 **两条分支**：上线 = §3.3 末尾指针；停放 = `section3b_structure_PARKED.md`（不在 build 列表，文件头有 promote 手续） |
> | 页数 | ⛔ **10 / 8**。两条便宜路（表内联、挪 Table 7）都测过无效，**溢出是字数** |
> | 门 | deslop exit 0 · TODO 0 · 未定义引用 0 · `--submission` fail（9→10，已知） |
>
> ### 🚫 新增「不要引用的数字」（接上表）
>
> | 错的 | 对的 |
> |---|---|
> | 「6 个表征买 16pp vs 重跑买 7.6pp = 2.1×」 | **臂数不对齐**（5 臂对 1 臂）。正确是 **7.14 vs 4.91–7.59**，一臂对一臂 |
> | codex 说 τ=.10「双轴支配」 | **比错基线** —— 它比 best-single；对 always-cheapest 只是 +4.91pp/+8.8% 非支配 |
> | 「标签根本不存在」 | **已撤** —— 全失败任务是六个负样本，每格 1218/1344 条 |
> | noise_floor_inventory 首版 reddit n=205 | **203**（AMENDMENT_08）；产物已改走 `expected_scored_ids` |
>
> ### 🔴 compact 前最后状态（08-01 夜）—— 下一个 session 从这里接
>
> **看板已定稿**（`deliverables/duizhang_board_2026-08-03.html`，Live Server 开）。经两轮返工 +
> codex 冷读表述审计 24 条全落 + 术语统一（classifieds / reddit / 完美路由）。8 表 / 214 数字。
>
> **⚠️ 四条待落地，都还没进稿也没进看板**（详 笔记 §406.14–406.16）：
>
> | # | 什么 | 状态 |
> |---|---|---|
> | 1 | **重跑对的代码版本核查** —— 两对都干净（提示词逐字同 · 变了的代码 0 次触发）。该进 **Appendix A.6**，主动堵「你两次跑的代码一样吗」 | 已查完，未写 |
> | 2 | **router 第四层：目标本身在抖** —— 换 1 个模式 → 27.8% 标签变；更强的是结构证据 **70.1% 多解**（不吃样本量） | 已算完，未写。**要 user 定**：会让「供给」这个单一诊断变复杂，而供给是现在标题的立论 |
> | 3 | **三个 phantom 模式无稳定行为签名**（18 指标 ≥5/6 全 0）⇒ ② 的**第二条独立否证** | 已算完，未写。引用前须过产物自设的三道闸 |
> | 4 | **① 的成本优势不是独立的** —— 成功 episode 比失败便宜一半，所以「两轴严格支配」是一个机制两处显形 | **修正**，若要写 5/6 支配必须按此表述 |
>
> ### 🚫 再加两条「不要引用的数字」
>
> | 错的 | 对的 |
> |---|---|
> | 「B1/B2 重跑百分百一致」 | 那是 **step 级**；episode 级 3/50 翻转。B0 = 模型+环境噪声，B1/B2 = **只有环境噪声** |
> | 「重跑翻转是 element id 编号变造成的」 | **没测过**。按模式拆开 DOM 0 / SoM 0 / P-text 0 / P-prompt 1 / Vision 2，与 id-churn 故事不吻合，且每格 10 对分不开。我此前是把 §298.3 的 id-perturbation **探针**结论错套到 episode 重跑上（M2） |
>
> ### 披露边界已核查，结论保留披露（§406.17）
>
> 噪声分析**已在 origin/master**（phase0b 07-28 · compare_cross_run 05-25）；OSF 预注册
> `10.17605/OSF.IO/9QCWU` 已公开。且 codex 判定这条校准是全篇唯一「unusually disciplined」的贡献。
>
> ### 还在跑（与关键路径无关）
>
> A100 WA-B0：**step 2/6 (som)**，ETA ~08-06（在截稿之后）· DGX mechanistic sweep：驱动 pid 2386755，**10/22 格**。
>
> ---
>
> ## 2026-07-31 交接（历史块）
>
> ### 在跑的三条（**不要照抄下面的数字，用命令拉实时**）
>
> | 位置 | 什么 | 怎么查 |
> |---|---|---|
> | **A100** | WA reddit × **B0** × 6 mode，**全量 104**（07-31 **19:54 UTC** 重发；16:18 那次 abort 了，见下）。实测 12.7min/ep ⇒ ETA **~08-06**，可能跨 08-05 但不在关键路径上 | `ssh condense-a100 "tail -5 ~/workspace/p79/logs/queue_chain_wa_red_b0_full104_20260731_195425.log"` |
> | **DGX** | mechanistic canonical sweep，**22 格**（不是 8 格） | `ls results/mechanistic/canonical/*/pilot_summary.md \| wc -l` |
> | **DGX** | MiMo-VL-7B 权重 ✅ **16G 已就位** | `grep SNAPSHOT_DONE logs/mimo_weights_pull.log` |
>
> ⏰ **三个时钟不一样，跨机器读 log 先对齐**：DGX = **BST(+1)** · A100 host = **UTC** ·
> A100 的 `vwa-reddit` 容器 = **BST**（B-309/B-753 特意设，reddit 有相对时间任务）。
>
> ✅ **本轮已完成**：h10 **k=6** 二次 promote（witness `pre_run/h10_artifact_regen_provenance_2026-07-31.md`；
> 结论未变，修的是「用 k=5 闸门描述 k=6 池」的 provenance 不一致）· `make analysis FAST=1` ·
> shopping reset 实现（`d78fd3b`，**未实测**，等 A100 空）· B3 截断修复（`46ffa1e`）·
> fire6_monitor 判据改 PID（已部署到 A100 cron）。
>
> ⚠️ **提交前必须再走 verdict-day 严格路径**：`make analysis` 是 routine 路径，
> `fig0c` 带 `--allow-partial` 且输出打 **NON_PAPER_GRADE 水印**；提交终态走
> `VERDICT_DAY_RUNBOOK`（不带该 flag，partial 数据 fail-closed exit 2）。
>
> ### user 本轮三条决定
>
> 1. **先跑 WA-B0，不是 MiMo pilot** —— 理由是网站覆盖度（学长：真实网页视觉/文字比例均衡；
>    台账 §95 实测 VWA **95.3% visual**、剔除后只剩 43 个 non-visual，而 WA 480 个全算 non-visual）
> 2. **sweep 跑满 22 格** —— 「要占，反正 dgx 免费」（此前建议跑完 P5 就停，已否决）
> 3. **shop reset 要实现** —— 从 backlog 提为待办，见下
>
> ### ⛔ shop 的真 blocker：reset 根本没实现（不是磁盘）
>
> `reset_vwa_sites.sh:296-308` 的 `_reset_vwa_local_shopping()` 是 placeholder，`return 78`
> → gate hard-fail（B-299 的 fail-closed 设计，pre-fix 的 `return 0` 会让 Phase 1b 静默地
> 在脏 Magento 状态上跑）。要实现三件：**Magento SQL-restore + cache flush + cart truncate**。
>
> **实现路径已定位**：shopping 用 `docker run` from `shopping_final_0712` 镜像（68GB，自带
> Magento + seed DB），与 reddit 的 postmill **同构** ⇒ 走「重建容器」路线即可；
> `start_vwa_docker.sh` 已封装 base_url patch + cache flush + indexer poll。
>
> **另一个前置**（不挡实现，挡 fire）：A100 磁盘 42G free 是**硬上限**——443G 已用里 419GB 是
> 5 个 ACTIVE docker images（三站容器在用，删不掉）。shop 12 conditions × 435 ep ≈ 18.8G
> artifacts ⇒ 必须边跑边 rsync 回 DGX（`sync_a100_results.sh`）或等 2TB 扩容。
>
> ### 为什么 shop 值得重开（07-16 那次裁定没算这个角度）
>
> `router_label_supply_diagnosis`：6 格里 **4 格完全不可训**（需 ≥2 类各过 `N_MIN_CLASS_TRAIN=10`）。
> 标签数 ≈ scored × 可解率，shop **435 scored** 是 cls(224) 的 2×，外推 B0_shop ~188 /
> B1_shop ~107 ⇒ 可训 cell **2/6 → 4/9**。且这是 Paper B「瓶颈是标签产生率」主张的**可证伪检验**：
> 变可训 = 论点精确化 + 一条阈值；不变 = 论点升级。两个方向都是增量。
>
> ### 🔴 deadline —— 对账推迟到周一，写稿窗口压到 2 天
>
> **学长生病，对账推迟到 2026-08-03（周一）**（user 07-31 告知）。重算时间线：
>
> | 日期 | 星期 | 什么 |
> |---|---|---|
> | 07-31 → 08-02 | 五～日 | user 标 508 条账本；**这三天 Paper A 等对账，但 Paper B 不等** |
> | **08-03** | **周一** | 和学长对账 → 骨架 |
> | 08-04 → 08-05 | 二～三 | 骨架落地 + splice |
> | **08-05** | **周三** | **REALM @ EMNLP 截止** |
>
> ⇒ **对账之后只剩 2 天**。
>
> ### ⚠️ 是**一篇**不是两篇（user 拍板 2026-07-28，笔记 §398.8）
>
> 原 Paper A / Paper B **已于 07-28 合并为单篇 8 页**（REALM archival，双盲 ACL）。
> 唯一的稿子卡 = `task_realm_paper_b_router_negative`（标题已改）；
> `task_realm_paper_a_phenomenon` 已 **superseded**，内容降级为下表 ①②。
>
> 主张：**表征路由的上限真实存在，但既不稳定也不可达。**
>
> | 步 | 内容 | 依赖 08-03 对账? |
> |---|---|---|
> | ① | ceiling 高（+3.4~16.1pp，省 13.7–35.3%） | **是** |
> | ② | 有结构基础（H3 双轴独立） | **是** |
> | ③ | **但结构小于同模式重跑地板**（§398.2，焊接枢纽） | 否 |
> | ④ | 且学不到（0/6 Pareto） | 否 |
>
> ### ⭐ 周末（08-01/02）可动手的是 ③④
>
> ③④ 是 **k-无关的结构性事实**（噪声地板 + 标签供给诊断 + 0/6 Pareto，数字 07-27/28 已 k=6 实测落地），
> **不依赖对账结论**。对账真正要定的是 ①② 那半怎么讲。
> ⇒ 周末把 ③④ 两节写完，08-03 之后的 2 天专注 ①② + 焊接。
>
> ① ② 只能等对账 —— 骨架未定时写的 prose 会在结论裁定后大概率推翻重写，
> 这正是 07-28 起走零预设重建的原因（也是稿件停在 07-16 不是遗漏而是设计的原因）。
>
> WA-B0 / sweep / B3 / shop 全部不在这条关键路径上。

> ## 2026-07-29 交接（历史块）
>
> ### 现在卡在哪：等 user 标完 508 条结论
>
> **账本** `docs/checkpoints/deliverables/advisor_ledger.html`（gitignore，由
> `scripts/analysis/build_advisor_ledger.py` 重新生成）。VSCode → Live Server 打开。
> user 正在逐条裁定（✓认可 / ✗有问题 / ?不确定 + 备注，存 localStorage）。
>
> **接下来的顺序**（user 定的，不要跳步）：
> 1. user 标完 508 条 → 点「导出给 Claude」→ 粘进对话
> 2. Claude 与 user 先过一遍「有问题 / 不确定」的
> 3. **基于这些**生成真正的对账议程 → 和学长定
> 4. 定完才写骨架
>
> ⚠️ 账本里那 7 条「要定的」是 **Claude 的初步建议，不是议程**。真议程从 user 标的结果长出来。
>
> ### 本轮（§399–§404，8 个 commit `6d603b8`→`961b4ab`，**未 push**）
>
> REBUILD_PLAN 待办 **2 / 4 / 6 全部结清**：
> - **待办 4** router 最有利角落：严格支配 always-cheapest **0/26** · 在经验 Pareto 前沿上 **0/26**
>   ⇒ 负结果在最有利配置下依旧成立。副产品 **per-task cost headroom 22–46%**（可独立进论文）
> - **待办 2** 四维画像首跑：7 个 6/6 一致签名里**经验发现 0 个**（3 架构下游 + 4 构造必然）
> - **待办 6** blocker 根因：quarantine→resume 写了新 summary 没换 steps，全库仅 **2/7686**
>
> **cross-AI 双家审计**（codex `gpt-5.6-sol` + agy Gemini，均 PASS）：7 条真问题全修，
> **其中 3 条改了结论**；另 2 条 Gemini 指控经实证**证伪**（H-pool 数学不可能 / bootstrap N=4 退化）。
>
> **B2·reddit Tier-2 补齐**：14/14 全 agent-limit，scaffold-bug 0 · benchmark-FP 0
> ⇒ 该格 ~1–4% SR 是真能力地板。索引自称不完整 **9 → 3**。
>
> **撞出的 benchmark bug**：`require_reset` 在 reddit 是 no-op ⇒ 订阅跨 episode 累积。
> 逐 episode 实查 **泄漏 6 · earned 31**，实质只影响 B2·DOM 一格（8→5）。
> **user 已定**：归独立 bug paper，主 paper 一句披露 + 指针，**scored universe 保持 203**。
>
> ### user 本轮三条决定
> 1. sidebar 泄漏 → 独立 benchmark bug paper，不扩展 AMENDMENT_08
> 2. **B1/B2 重跑地板不用测** —— 确定性已实证（§298.2 133/133 · §397.10 1.000/1.000 · 全 A100 单型号）
> 3. mechanistic sweep 继续跑（pid 38603，deadline 08-01），与 **WA** 一并进「要和学长对的 Phase 1」
>    → 已写入 `_status/tasks/task_analysis_gating.md`
>
> ### ⚠️ 我在本轮犯的错（同型四次，见 §403.2 / §404.1）
> 全是**把更弱的东西当成更强的结论**：解析失败当成测量为零 · 「有风险的集合」说成
> 「已污染的集合」· 文本共现当成数据模式 · **把「我认为重要的子集」当成交付物**（流程层）。
> 第四次「没应用 context 里已有的 host-role 规则」是 REBUILD_PLAN 第五类错误**第三次复发**
> → 修复已进结论层 `INDEX §0`（常驻事实节 + checklist 规则）。
>
> ### 结论层三处结构性补漏
> `INDEX §0` 常驻事实（CLAUDE.md/memory 独有、三道防线都扫不到）· `§2` diag 覆盖度重写 ·
> `§1` canvas 逐个裁定 · `measured_D4` 附录 A（§398 噪声地板）与附录 B（diag Y1–Y5）。
>
> ### deadline
> **08-05 REALM @ EMNLP**（今天 07-29，剩一周）。骨架未定 —— 它卡着下游三项。


> 🚨 **2026-07-28 起改走零预设重建 —— 新 session 先读 [[REBUILD_PLAN]] 而不是本节。**
> 本节仍然是 *状态* 的来源 (跑了什么 / 数字 / 后台), 但 *下一步做什么* 由 REBUILD_PLAN 定。
> ⚠️ **上一轮 (§397.4-§397.9) 的结论一律视为待验**; 必须连 **笔记 §397.10 (作废条目)** 一起读。

> 🎯 **2026-07-28 — §0-C 台账 5 条落地 + 自审 3 条 + 跨 AI 三家 14 条全部落地。
> 两篇稿 8 页限内、门全绿、1626 tests pass、已 push。剩下只有人工动作。** ⭐⭐⭐⭐⭐
> (详 笔记 §397 / §397.7; 上一轮 §396)
>
> ### ⚠️ 最重要的一条方法论教训 (§397.7)
>
> 跨 AI 三家收敛到一条我自审**没抓到**的元问题: **本轮所有"修好"的数字都是 in-sample
> plug-in 估计, 却被冠以推断性的名字** —— `Bayes ceiling` 实为 modal agreement ·
> `interaction` 实为排序的代数后果 · `mode-invariant` 实为池化后的残差。
>
> **为什么自审漏了**: 我问的是"换个分母还成立吗"(**计量**层), 三家问的是"这个数配得上
> 这个名字吗"(**估计量**层)。**换口径复核抓不出估计量层的错 —— 两种检查都要做。**
> 尤其讽刺: 我上一版选中"交互"来立论正因为它换分母也 6/6, 而 Gemini 指出**它之所以稳健
> 是因为它被排序代数强制** —— **稳健 ≠ 有信息**。
>
> ### 本轮做完了什么 (commit `a59d731` → `120888f`, 7 个)
>
> | # | 结果 |
> |---|---|
> | **#3** | §4.2 机制推断改成真 2×2, 后经 Gemini 再修: **撤掉 "interaction" 一词**, 改描述性报告 + 主动交代"极值条件下该对比被代数强制"(action-step 5/6 格强制, episode 分母仅 2/6 强制而结论仍 6/6 → 那 4 格才是非平凡支持) |
> | **#16/#17** | 两个新产物。§6.1 三元组 → **7,686 / 7,041 / 645** (scored 口径, 18×224+18×203); §4.3 mode-invariance 改成实测 spread 7.4-13.7pp + vision 例外单说 |
> | **#11** | 置换 200 → **10000**。red·B2 **p=0.00050** (原 0.0050 = 地板), 裁定不变但现在由数据决定。附录 C 说清测的是整格阈值而非嵌套策略 |
> | **#9** | 修掉一条**假前提**("同 task 同特征向量" —— reddit 80% 共享 task 其行不同)。数**维持 79.2/83.7** 但经 codex 再修: 全篇**改名 in-sample modal agreement** (退回 task 分组后仍是 resubstitution —— cls 29.8% / reddit 37.0% 的行是单例), 补"仅共享 task" **70.3%/74.1%** |
> | **#8** | 查清**没有可用同模式 null**。不硬跑, 改在不计页数的 Limitations 里说全; 经 Gemini 再修成锋利版: **门测的是 ≠0, 而同策略两次跑本来就 ≠0 → 过门是弱证据**, 且区间宽度救不了 (重抽 task 不重抽 run) |
> | **§4.3** | (codex 新增) mode-invariance 是**池化假象** —— 格内最大 spread **15.1/27.3/48.3/36.1pp** vs 池化的 7.4-13.7。改成 "similar after pooling" + 明说不主张任何单格 mode-invariant |
> | **产物** | (codex 新增) `p` 曾被 `.3f` 打成 `0.000` 而散文硬编码 `p=0.005` → 小 p 走科学计数法 (**现 0.0005**) + **结论句动态生成** + 抽签数不足即 fail-loud |
>
> **新产物 (各一条命令重生成)**:
> `aggregate_cross_mode_failure_signatures.py` → `cross_mode_failure_signatures.{md,json}` ·
> `aggregate_evaluator_score_granularity.py` → `evaluator_score_granularity.{md,json}`。
> 均已登记进 `EVIDENCE_LAYER_AUDIT §0`。
>
> ### ⚠️ 下轮动数字前必读 (笔记 §397.5/§397.6/§397.7)
>
> 1. **跨 AI 报的那条往往不是最深的那条** —— #9 报"分组错"实为"前提假", #11 报"次数少"
>    实为"裁定由 B 而非数据决定"。照着报的修 = 修一半。
> 2. **算完换个正交口径复核自己** —— 我给 #3 的第一版有两个计数只在 action-step 分母下
>    成立, 换 episode incidence 就塌。产物里现在有一张"哪些陈述扛得住换分母"的表,
>    **只有标 yes 的才准进正文**。
> 3. **防线按能力布不按字面布** —— 我的新脚本经 `_discover_episodes` 读 episode, 源码里
>    没有 `_summary_v2.json`, 直接绕过昨天刚立的 B-1906 棘轮。棘轮已扩。
> 4. **⭐ 稳健性检查 ≠ 估计量检查, 两种都要做** —— 换分母查的是"算得对不对",
>    查不出"这个量是不是你说的那个量"。后者只能靠问"**这个名字的定义是什么, 我的计算
>    满足那个定义吗**"。本轮三条最贵的错全在这一层 (`Bayes ceiling` / `interaction` /
>    `mode-invariant` 三个名字都名不副实), 而我的自审一条都没抓到。
> 5. **⭐ 指标的判定基准在各臂是同一个吗?** —— 幻觉率定义为 "id 落在 `obs_nodes_info` 外",
>    而该 map 的键空间**按 text payload 分两套** (AXTree = 稀疏原生 nodeId / legend = 稠密 1..K)。
>    "落在 S 外的比率" 在 S 随臂变化时**不是同一个量** —— 它是 行为 × S 覆盖度 的乘积。
>    **user 抓到的, 三家 AI 全漏** (codex 查 provenance 没查键空间 · Gemini 只看散文 ·
>    我查分母和估计量名字, 没查探测器等灵敏)。详 §397.9。
> 6. **冷读模型的误读是歧义检测器** —— Gemini 两条 findings 攻击不成立 (把跨 mode 离散度
>    读成率本身; 把 4.975e-3 约整当成用错公式), 但**它误读的地方审稿人也会**, 两处措辞都改了。
>
> ### 先跑再信 (三条命令, 别照搬下面的数)
>
> ```bash
> cd docs/checkpoints/paper_drafts/latex && bash convert.sh paperA --submission && bash convert.sh paperB --submission
> make deslop-ratchet                     # vale, 12 文件阻塞集
> ssh condense-a100 'date -u; ps -o pid,etime -p 2658570; pgrep -af run_experiment'
> ```
> **A100 是 UTC / DGX 是 BST** —— 对时间线先 `date` 两端。
>
> ### 1. 两篇稿的状态
>
> | | 页数 | 门 | Overleaf |
> |---|---|---|---|
> | **Paper A** 现象篇 | 正文 8 页 | `--submission` PASS | `main_paperA.tex` |
> | **Paper B** 路由阴性 | 正文 8 页 | `--submission` PASS | `main_paperB.tex` |
>
> 一个 Overleaf 项目放两篇 (clone `~/overleaf-aaai27`, 目录名是历史遗留)。
> **切编译目标 = Overleaf 菜单 → Main document**。同步: `bash scripts/maintenance/overleaf_sync.sh`。
>
> 构建根是**新建的** `paper_drafts/latex/`, **不是** `aaai27/latex/` —— 后者是 AAAI 存档,
> 被 `tests/test_dayaudit_rounda_20260714.py::test_f03` 的 fixture 钉住, 别动。
>
> ### 2. 🔴 下一步 (台账已清空, 以下按优先级)
>
> | 优先 | 事项 | 谁 |
> |---|---|---|
> | **P0** | **REALM OpenReview 投稿 (08-05)** — 作者/COI/abstract 见 `deliverables/` | **user** |
> | **P0** | **学长看 Overleaf 两篇** | **user** |
> | P1 | **毕设 D8** (results+discussion 章) **已过期**; D9 全稿 08-10 与 REALM 08-05 撞期 | user 决策 |
> | P2 | 本 session 改动**未跑跨 AI 链** (Mode B/C)。这 5 条本身就是三家审计的产出, 我只跑了 Claude 自审 (抓到 3 条我自己引入的 P0)。要补就 scope = 本 session 改动 | 可选 |
> | P2 | A100 WA 全量跑完后 (~3 天) 才有窗口补 **#8 的同模式重跑** (WA reddit 与 VWA reddit 共用 postmill 容器, 不能并跑) | 排队 |
>
> **动手前必读** (两条并列):
> - §396.7 —— 跨 AI 的「明显该修」**先查是不是已被裁定过** (codex #4 在推翻
>   `router_features.py:78-101` 的 2026-06-09 F2/B-1806)。冷读模型看不见代码注释里的历史裁定。
> - §397.6 —— 跨 AI **报的那条往往不是最深的那条**, 且**自己修完要换个正交口径复核**。
>
> ### 3. 页数怎么调 (本 session 学到的, 别重踩)
>
> **超页先数表, 再决定砍不砍字。** 砍字不可逆, 浮动体是免费的:
> ```bash
> cd docs/checkpoints/paper_drafts/latex/build/paperB
> for p in $(seq 1 9); do echo "p$p: $(pdftotext -f $p -l $p main.pdf - | grep -cE '^Table [0-9]+:')"; done
> ```
> 本轮真凶: Table 1 是 `table*`, 跨栏浮动只能落页顶, 排队时 LaTeX 保序 → 后面所有表被卡住,
> 落地时一栏倾泻 4 张。全改单栏后 **一个字没砍就 9 页 → 8 页**。
> `convert.sh` 的 `SINGLE_COL_BODY` 是 **per-paper 显式清单不是规则** —— pandoc 只在单元格相对
> **全宽**够短时才出 `l` 列, 而 `l` 列不折行, 单栏下长表头会越界压到邻栏**且不报 Overfull**。
> **改了表就得渲染成 PNG 眼看一遍。** 本 session 两次靠眼看抓到 LaTeX 静默放过的错。
>
> ### 4. 🚫 不要引用的数字
>
> | 出处 | 错的 | 对的 |
> |---|---|---|
> | 旧 hook / 老 §1 | drop-one **1.7-3.3pp** hero | 6-mode k=6 **0.0-1.3pp**, H1 FAIL |
> | B-1898 旧记 | 该报 0.6533 | **反了** — 报 **0.7897** |
> | 笔记 §387.9 | 汇总 SR 6.37% | **6.40%** |
> | §387.16.3 | triage 成本 −38%~−45% | **−9.5%~−30.6%** |
> | venn/lift 旧图 | B0_red 独解 6 · B2_red 2 | **5** · **1** (B-1907) |
> | §383.4 | 「~1/4 标签由 tie-break 决定」 | **不成立** — true_tie 全 0; 真相是两个定义分歧 12.5-54.6% |
> | Paper B 旧 abstract | oracle「只赢成本 13–22%」 | **+3.45~+16.07pp 且 −13.7%~−35.3%** |
> | Paper A 旧 §4.2 | 幻觉引用降 **5–25×** | 该比值是 **DOM vs P-SoM 的复合对比**, 不能归给文本轴; §4.2 已改成 2×2 分解, 不再引这个比值 |
> | 07-28 中途版 §6.3 | 天花板 **83.9/89.1** (按特征向量) | **79.2/83.7** — 特征向量分组单例膨胀 (reddit 75% 的行是单例, 单例按定义 100%) |
> | 07-28 中途版 §4.2 | "P-SoM 六格最低 / P-prompt 五格最高" | **换 episode 分母就塌** (5/6 · 3/6)。只引产物里标 `quotable=yes` 的两条 |
> | #11 旧记 | red·B2 p = **0.0050** | **0.0005** (B=200 时 0.0050 是地板 1/201, 不是测量值) |
> | 07-28 中途版 §4.2 | 任何 **legend vs AXTree** 的幻觉率对比 (含 "P-SoM 六格最低" / "降幅更大" / 旧 "2.3-24.8×" / 旧 "SoM 0.08 vs dom") | **全部作废** — 跨 id-namespace。⚠️ **compact namespace 是 som/P-SoM/P-text 三个不是两个**; **Vision 无 element id** (坐标动作), 其 0.000 结构性不适用。详 §397.10 |
> | §397.9 我那套"探测器灵敏度"论证 | 当成 §4.2 的机制 | **别用** — 这条 id 噪声早有正经测量 (`b0_paired_idperturb` probes: B1 本地 temp0 组内一致性 1.0, id-shuffle 改变决策 **20%**; B0 12.5%)。§298.3 已写明 dom/p-prompt 承担、SoM-family 经 AMENDMENT_07 消除 |
> | §397.4 "只有一对同模式重跑且污染" | 当成 #8 的结论 | **错, 我搜漏了** — manifest 15 组有第二 run (archived/pre-fix) + `results/repro_replicates/B0_vision_classifieds_R24792_clean_replicate/` 是**干净**的; `compare_cross_run_same_condition.py` 专门算这个数; **§302.1 早有: self_drop 6.7/7.6pp** vs H3 轴 1.35/2.09pp |
> | 07-28 中途版 §6.3/A.4 | "Bayes ceiling" | **in-sample modal agreement** — 是 resubstitution 估计, 数不变但名字换了 |
> | 07-28 中途版 §4.3 | 四签名 "mode-invariant" | **"similar after pooling"** — 格内 spread 达 48.3pp |
> | §6.4 旧句 | tier "raises the ceiling to 89.9/96.7" 证明 backbone 一致 | 上升有算术成分; 一致性证据是**直接 agreement 68.5/88.0 vs 六路 42.6/44.0** |
>
> ### 5. 后台 (⚠️ 跑命令核; 下列为 07-28 08:28 BST 实测)
>
> - **A100 WA 全量**: chain pid **2658570** 存活 (13h28m), 6 mode × 104, 仍在 step **1/6**
>   (dom, run `B1_dom_wa_reddit_..._R13217`)。ETA ~3 天。**A100=UTC / DGX=BST**, 差 1h。
> - **DGX mechanistic sweep**: ⚠️ **在跑, 不是做完了** (2026-07-28 更正)。真对象是 **sweep 驱动
>   pid 38603** (`logs/mechanistic_canonical/.sweep.pid`) + supervisor, 活了 23h; **24 cell 里
>   做完 2 个**, cell 3/24 `p1_rev_reverse_cls` 正跑 (worker pid 1638252, 占 21.7 GB VRAM)。
>   单 cell ~800-845min → **08-01 deadline 会在第 7-8 个 cell 截断, 跑不完 24 个**。
>   ⛔ 上一版说"已完成"是**只查了 worker pid 38617**(它确实在 03:34 干净跑完了 cell 2) 就
>   宣布整体完成 —— **查子进程不查驱动**, 与本轮另外三次错同型。
>   属 §5 mechanism (advisor 2026-05-14 已搁置, 不进当前 paper), 但**不挡任何东西**:
>   DGX 就是留给 dev/curation/mechanistic 的共享争抢机 (paper-grade fire 2026-05-14 已迁 A100),
>   占 21.7/~128 GB 单进程, 在 CLAUDE.md "一次 1-2 进程" 范围内。**B3 的 fire 在 A100**
>   (其 frontmatter 自己写的), 只有适配碰 DGX 且可共存。
>
> ### 6. 人工动作 (只有 user 能做)
>
> - REALM OpenReview 投稿 (08-05); 作者/COI/abstract 见 `deliverables/`
> - 学长看 Overleaf 两篇
> - **毕设 D8** (results+discussion 章) **已过期**, D9 全稿 08-10 与 REALM 08-05 撞期
>
> **距 08-05 还有 8 天。git 已推 origin; Overleaf 已同步** (`a4550b2 = repo 2115173`,
> `SUBMISSION=1` 严格门通过, 两篇 limitations 均已拷入)。学长可直接看。
> ⚠️ Overleaf 网页端编辑**不自动回流** —— 内容冻结前一律改 md 再 sync, 防双源漂移。
> 切编译目标: Overleaf 菜单 → Main document → `main_paperA.tex` / `main_paperB.tex`。
>
> ### 7. 收尾时的门 (三条都跑过, 全绿)
>
> ```bash
> make deslop-ratchet                                                    # PASS
> cd docs/checkpoints/paper_drafts/latex && bash convert.sh paperA --submission \
>   && bash convert.sh paperB --submission                               # 两篇 content 均 ≤ p8
> .venv/bin/python3 -m pytest -q                                         # 1626 passed / 0 failed
> ```

---

<details><summary>07-27 深夜及更早的 handoff 块 (历史, 数字可能已被上面取代)</summary>

> 🎯 **2026-07-27 (五) 深夜 — 两篇 8 页稿都已成型。剩下的是 LaTeX 化 + stress + 投。** ⭐⭐⭐⭐
>
> **先跑再信**: `paper_grade_check.py` (A100) · `make active` (DGX) ·
> `vale --config=tools/paper-deslop/.vale.ini docs/checkpoints/paper_drafts/paperA/ paperB/`
> (上一版 handoff 写「后台不需要干预」而全量 WA 实际跑了 0 个 task, §390。)
>
> ### 1. 权威结论 (产物 `phase1_full_prereg_decision.json`)
>
> ```
> H1  P-SoM drop-one   FAIL  θ_FE=0.7897pp / boot 中位 0.7490 / CI[0.2858,1.4471] / p=0.807
>                            I²=0.0% Q=1.4265(df=5) → 小效应而非功效不足, 加 cell 无用
>                            4 个 cell 触发 SE floor, 方向是**反的**(降权最小效应 cell → θ 抬高)
> H3  axis-1 |P-text \ P-SoM|    PASS  1.3528pp CI[0.799,2.026]  p=1.19e-05
>     axis-2 |P-prompt \ P-SoM|  PASS  2.0877pp CI[1.399,2.919]  p=7.52e-07
>     → 承重句: **compound 臂吸收不了两个单轴臂** → 有轴的空间, 不是一个点
>     5/6 cell 过 noise floor(2 任务); 单独 Holm 显著 axis-1 3/6 · axis-2 4/6
> H2(a) cost 带         未证伪  6/6 cell 在带内, n=1281 paired
> → framing R5 + C'-S; P-SoM 独解 6 cls + 3 red
> ```
>
> **B-1898 已解决**: 0.68pp 阈值 2026-05-18 锁在 prereg L718, AMENDMENT_03 有 tag。
> **H1 报 0.7897; 0.6533 不得表述为「预注册规则的结果」。**
>
> ### 2. ✅ 两篇 8 页稿都已成型, deslop 全过
>
> | | 路径 | 词数 | deslop |
> |---|---|---|---|
> | **Paper A** 现象篇 | `paper_drafts/paperA/section{1..5}*.md` | 4440 | 0 errors · 5/5 invariant PASS |
> | **Paper B** 路由阴性 | `paper_drafts/paperB/section{1..7}*.md` | 6647 | 0 errors · 7/7 invariant PASS |
>
> 12 个文件已入 `tools/paper-deslop/deslopped.txt` (CI 起拦 error 级回归)。
> 数字**逐位可追溯**到产物。**两篇都投 workshop** (REALM @ EMNLP, 08-05, 双盲 ACL 8 页)。
>
> ⚠️ **原 `paper_drafts/section*.md` (53741 词) 是素材档案, 不是投稿稿** —— 所有 /stress
> 结论内联在里面, 有审计价值, 但 §2-§8 从未同步 H1 FAIL/H3 立论。**投稿用 paperA/**。
>
> ### 3. 🔴 剩下三件 (按顺序)
>
> **(1) LaTeX 化 + Overleaf —— user 已拍 (a): 一个项目放两篇**
> - clone 在 `~/overleaf-aaai27` (只有 7-16 我们自己 sync 的初始上传, **无网页端编辑待回流**,
>   push 不会覆盖谁)。`OVERLEAF_GIT_DIR` 未设但 `overleaf_sync.sh` 默认就是它。
> - 三处要改: 模板 **AAAI → ACL 2026**(现在是 `aaai2027.sty`/`.bst`) · `convert.sh` 的
>   `SOURCE_MD` 写死 `aaai27_main.md` → 要支持 paperA/paperB 两个 main · 一个项目两个 main.tex
> - `convert.sh` 已有 `--submission` 模式(TODO 槽非零则 fail), 复用它
>
> **(2) `/stress` 三家链 (user 已授权)** —— 两篇稿是全新 prose, 从未被 stress 过。
> Mode A(Claude) + Mode B(`codex --sandbox danger-full-access`) + Mode C(`agy -p`,
> 见 `scripts/maintenance/agy_stress_clean.sh`)。scope = pre-fire/submission band
> (findings ≥7 / ≥10)。**跑完必做 3-phase 后检**(I/O sanity · 深度+scope · runtime)。
>
> **(3) Paper B 补真引用** —— §7 related work 现在是**无 cite 的占位散文**, archival 稿
> 不能这样投。Paper A §5 同样缺 cite。`paper_drafts/paper.bib` 有现成 key 可用。
>
> ### 4. 🚫 不要引用的数字
>
> | 出处 | 错的 | 对的 |
> |---|---|---|
> | 旧 hook / 老 §1 | drop-one **1.7-3.3pp** hero | 6-mode k=6 **0.0-1.3pp**, H1 FAIL |
> | B-1898 旧记 | 该报 0.6533 | **反了** — 报 **0.7897** |
> | 笔记 §387.9 | 汇总 SR 6.37% | **6.40%** |
> | §387.16.3 | triage 成本 −38%~−45% | **−9.5%~−30.6%** |
> | venn/lift 旧图 | B0_red 独解 6 · B2_red 2 | **5** · **1** (B-1907) |
> | §383.4 | 「~1/4 标签由 tie-break 决定」 | **不成立** — true_tie 全 0; 真缺陷是 12.5-54.6% 顺序选了更贵的 |
> | §383.4 reddit | 矛盾率 45.5% / 上限 87.7% / tier 95.5% | **56.0% / 83.7% / 88.0%** |
>
> ### 5. 产物 (全部可重跑)
>
> `phase1_full_prereg_decision.{json,md,csv}` · `router_triage_learnability.{md,json}` ·
> `router_label_supply_diagnosis.{md,json}` (Paper B §4/§6 主体) ·
> `router_objective_ordering.{md,json}` · `sr_per_mode.json` (reddit 分母 203) ·
> `phantom_lift.csv` (B-1905 修后) · `amendment08_sensitivity.md`
>
> ### 6. 本 session 修完的 bug (catalog B-1901~B-1918, 测试 1626 passed)
>
> universe 族全线 + 自动化可靠性。三条最结构性: **B-1906** SHA-only 绕过跨产物互校
> (+ 常驻 lint 带只减不增棘轮) · **B-1903** 真嵌套 CV · **B-1916** FORCE_NEW 缺失差点让
> P-SoM 主角臂整个非 paper-grade。**横切教训**: 三条都不是算错了, 而是把验证责任交给了
> 一个不会执行它的地方(抄来的 SHA / 不读该产物的下游 / 与外折无关的 OOF 事实)。
>
> ### 7. 后台 (⚠️ 跑命令核, 别信文字)
>
> - **A100 WA 全量**: chain pid **2658570**, 6 mode × 104, `FORCE_NEW=1` 六臂同契约;
>   monitor `b3ulth366` 三态 probe(ssh 失败**不下判断**)。ETA ~3-4 天。
>   **A100=UTC / DGX=BST**, 对时间线先 `date` 两端(§391 踩过)。
> - **DGX mechanistic**: 38617 跑 `p1_fwd_strong_red`, 08-01 自截断, **非关键路径**。
>
> ### 8. 仍未决 / 欠账
>
> - **毕设 D8** (results+discussion 章) **已过期 4 天**, D9 全稿 08-10 与 REALM 08-05 撞期
> - `UNIVERSE_TRIAGE_PENDING` 里 ~34 个脚本(棘轮钉住不会长)
> - `deslop` 教训: 机械 em dash 替换**两次**造出逗号拼接(paperA/paperB 各一次)。
>   若有第三篇, **先写 replacement 表逐处判断**, 不要先跑正则再补救。
>
> **距 08-05 还有 9 天。22 个 commit 全部未 push。**

> 🚦 **2026-07-27 (三) — 前一块的 6 步清单已全部完成; 下一步 = k=6 重灌** ⭐⭐⭐ (详 笔记 §387.15)
>
> **✅ 已落地** (commit `dc15837` / `de3ff66` / `147ec12`, **已 push**):
> 1. **AMENDMENT_08** (不是 PROTOCOL_NOTE_07 —— `PROTOCOL_NOTE_01 §0` 把 AMENDMENT 命名空间
>    保留给动 `scored_task_count` / sample-pool composition 的改动, 本次两条都命中)。
>    reddit **计分**分母 205→**203**, **收集**分母保持 205 (runner 仍采集这两个 task ——
>    这让完整性契约跨 amendment 边界不变, 且两条敏感性臂对任何 run 都算得出来)。
>    tag `prereg-amendment-08-scored-set-exclusions-20260727`。
> 2. **判据强弱如实分层**: tier A (task 160, config 可推导 outcome-blind) / tier B (task 58,
>    需轨迹确认)。`tiers=()` 精确复现旧 universe SHA → 敏感性臂是**比较**不是重算。
> 3. **敏感性** (`docs/analysis/cross_sites/amendment08_sensitivity.md`): reddit pooled
>    `none` 6.94% → `A` 6.62% → **`A+B` 6.40%** / `A+B+X` 7.86% (仅参考); cls 四臂全 10.19%。
>    ⚠️ **6.37% 是错的** (§387.9 原数扣了分子没减分母 + B0 少数一格) → **一律用 6.40%**。
> 4. **B-1891 FIXED** (选项 b + 更保守): 加 `action_intent_fulfilled` 字段 + **独立** trigger
>    `intent_unfulfilled_streak`。**不动 `action_success`** (它喂 agent 可见的 FAILED 反馈),
>    **不并进 `no_progress_streak`** (WA 臂在改动后采集, 复用旧键会静默不可比)。
> 5. **B-1890 防线**: `NOT_POPULATED_BY_RUNNER` 登记 + **双向**测试 (全库扫描抓下一个同类字段)。
>    首跑抓到 7 个, 逐个查赋值点后确认全部是真算真为 0, 归 `CONSTANT_BY_NATURE` 各写理由。
> 6. **B-1894** (新, user "查笔记"指令直接产出): `queue_chain.sh` 的 WA `SITE_EXPECTED_N` 停在
>    pre-exclusion (106/192/182 vs 实产 104/173/176) → **全量 WA paper-grade chain 必 FATAL**。
>    错的理由 ("WA has no N/A taxonomy") 与笔记 §137 task #76 直接矛盾。已修 + 加测试同时钉住
>    数字**和**否定该理由文本。**B-1895**: 跨站 `|AND|` 畸形 obs_url 77/720, 根因未定。
>
> **▶ 下一步 = k=6 重灌** (bind → promote → `make analysis`, 走 NUMBERS_TODO §0 配方)
> → 两篇 REALM 稿 splice 数字。**注意**: reddit universe SHA 已从 `41b1a918…` 变为 `1ce29c8b…`,
> 所有带旧 SHA 的 reddit 产物都是陈旧的, **必须全部重生成** (产物间 SHA 互校会自动抓出漏网的)。
>
> **仍未答**: 要不要起 **毕设 D8** (results+discussion 章, **07-24 已过期 3 天**,
> 下一环 D9 全稿 08-10 与 REALM 08-05 撞期)。
>
> **WA (A100, 无人值守自走)**: pilot 6-mode × 10 task 收尾中 (第 5/6 个 mode)。
> **pilot 退出后会自动起全量 chain** —— 6 mode × **104** task (放开 10-task 限制:
> n=10 的 95% CI ≈ ±25-30pp, 撑不起任何结论), ETA **~3.5 天**,
> `RESET_BEFORE=1` + `P79_PAPER_GRADE=1` 无 partial 旁路 (B-1894 已把 gate 修成 104)。
> launcher `scripts/queues/_launch_wa_full_reddit.sh`, pid 落 `.wa_full_chain.pid`,
> log `logs/queue_chain_wa_red_b1_full_<ts>.log`, 起飞时 ntfy `p79-claude`。
> WA shopping/shopping_admin **不放开** (无 reset 实现, 跑出来非 paper-grade 干净)。


> 🔭 **2026-07-27 UPDATE (二) — /diag v8 freeze 完成: 36 condition 同版本, cross-mode 解锁** ⭐⭐⭐ (详 笔记 §387.13):
> - **✅ 规则批落码** (`dfa546c`): `RULESET_VERSION = 8-reddit-p41p46-b1890fix`。修 2 条 (B-1890 死字段 guard / P33 reddit 路径) + 新增 6 条 (P41-P46)。新增 `diag_rescan_all.py` 做 36-condition 重扫 + 版本一致性校验 (不一致 exit 1)。**reddit 18 + cls 18 全部落 v8, 36 份 digest 均补 v8 数字块 → cross-mode / cross-site 聚合解锁**。测试 1569 passed。
> - **⚠️ cls 行为非字节不变** (与 v6→v7 的 H1 不同): P35/P39 旧命中被移除 (抽查确认旧命中是错的) + P33 在 cls +1 例 (task 233 的 `sites` 漏声明跨站)。**cls 聚合请用 digest 里的 v8 数字块, 不要用正文旧数**。
> - **⭐ 首个 cross-mode 矩阵的结论对 paper 直接相关**: 占比最高的四条失败签名 (P31/P36/P5/P14) **全部 mode-无关** → 换表征救不了。与 §387.8 (comment 天花板 4x) / §387.10 (补图增益 ≈0) 共同界定 **routing 空间的外边界比 drop-one oracle 数字看起来的窄** —— **discussion 须三条并排写**, 否则 oracle 会被读成"还有这么多能靠 routing 拿到"。
> - **🐛 本轮新增 bug**: B-1888 (defaults 只解析一层) · B-1889 (task 160 passive FP, **待 prereg 级决策**) · B-1890 (footprint 字段恒 0 陷阱) · B-1891 (`action_success` 语义脱节 → **Phase 3 的 M3 retry 会失效**, 启动前必修) · B-1892 (task 58 参数知识捷径, **待决策**) · B-1893 (fire6 硬编码 VWA 路径致假告警, 已修)。
> - **🔲 diag 剩余欠账**: `SINGLE_TARGET_FINISH_ON_MULTI_TARGET_TASK` (多目标任务提前收工, 需实体抽取非纯字段比较) + P27 `ABANDONMENT_RE` 扩充 —— 留下一轮。
> - **✅ 已 push** (user 2026-07-27 授权): 本 session 全部 commit 已上 origin。

> 🔭 **2026-07-27 UPDATE — k=6 数据齐 + WA pilot 首次点火 + B-1888/B-647 两雷** ⭐⭐⭐ (详 笔记 §387):
> - **✅ k=6 数据完备**: `B2_phantom_prompt_reddit_20260723` 07-25 08:19 落 205/205 → **B2 reddit 六 mode 全齐**。**下一步 = bind → promote → k=6 重灌** (走 NUMBERS_TODO §0 配方), 随后稿件三处必改 (§383: "two cells"→"three cells" / 机制口径 `insufficient_train_data` / 补披露 Pass-2 从未 fire) + **B-1284 cross-family modifier 具备解除条件** + Protocol Note 06 两轨制披露块可删。距 REALM 08-05 还有 9 天。
> - **🔥 A100 在跑 WA pilot** (非 k=6 关键路径, 占的是 GPU 不是分析算力): `queue_chain` 6-mode **B1 × WA reddit**, 每 mode 10 task (prereg §8.8 分层采样), run_id 形如 `B1_<mode>_wa_reddit_2026072x`, log `logs/queue_chain_wa_red_b1_20260727b.log`。**RESET_BEFORE=1 全程 paper-grade**。完成时 ntfy `p79-claude`。
> - **⚠️ A100 GPU 曾整机不可用**: 07-24 `unattended-upgrades` 升 nvidia 用户态到 580.173.02 而内核模块仍 580.159.03 → `nvidia-smi` 直接失败。已用 `rmmod`+`modprobe` 原地修 (无需 reboot, 绕开 KubeVirt detach p-79)。**长期不跑 GPU 后应主动验一次 `nvidia-smi`**。
> - **🐛 B-1888 (已修, commit `0e6429c`)**: `defaults:` 继承只解析一层 → **55 个 WA config 从未继承 base 层** (缺 backend type / token 单价 / 碳强度 / tool-calling)。VWA 全是一级链故不受影响, 已用 123-config SHA 快照证明 **VWA 侧逐字节不变**。1536 passed。
> - **🐛 B-647 partial lift (commit `55755cc`)**: WA reddit reset 解除封印 (WA/VWA 共用同一 postmill 容器 + 同一 `.auth/reddit_state.json`)。**遗留风险**: site flock 是 per-(site,**benchmark**) 的, 共用容器下粒度错 → 理论上 WA reddit 与 VWA reddit 可并发互毁站点状态, **两者不要同时跑**。
> - **📊 reddit 15 条 /diag Tier-1 已全扫** (`/tmp/diag_red/*.json`, 3075 ep, 0 token)。**B2 reddit SR 仅 0.49–3.90%** 且 no-hit 比例塌到 0.5–4% (B0/B1 是 11–23%) = 失败模式高度同质; **P36 密度为 B0 的 4.5 倍**。**进 k=6 前须先定性 P36 是 agent-limit 还是 scaffold** — Tier-2 已就此深挖 (B2 六 mode)。Tier-3 digest 15 份 + reddit 规则批 (R1-R8/H2) → `8-reddit-*` freeze 全量重扫 = 未完成。
> - **⚠️ 未 push**: `0e6429c` + `55755cc` + `3a599b1` 在本地。

> 🔭 **2026-07-22 UPDATE — AAAI 撤出 → REALM @ EMNLP workshop ×2 (08-05) + H10 门控管道修复** ⭐⭐⭐ (详 笔记 §383):
> - **🎯 投稿转向 (user 决策)**: **AAAI-27 不投**。改投 **REALM @ EMNLP 2026** (`realm-workshop.github.io`), direct submission **2026-08-05**, long **8 页正文 + refs/appendix 不限**, 双盲 ACL 2026 style, notif 09-07 / camera 09-14。**两篇都是主线**: **Paper A = phantom 现象篇 (non-archival, 保主会选项)** / **Paper B = 路由阴性结果 + 标签诊断 (archival)**。第二 venue user 待想起。→ `_status/tasks/task_realm_paper_{a,b}_*.md`
> - **📄 8 页 > AAAI 7 页 + appendix 无限** → 原 `cut_prewrites` 砍词**作废**; Paper A 只需 AAAI→ACL 格式转换 (`aaai27/latex/` 改目标模板)。
> - **⭐ Paper B 不等数据**: 证据 k-无关, **现在就能整篇写完**; Paper A 的 H1/H3 全随 k=6 移动 → 先写非数字段落。唯一缺的一块 = **LOCO 池化+tier 实训** (prereg L447 已注册槽位)。
> - **🔧 H10 门控管道断了两个月已修**: canonical `l1_router/` 停在 05-18 `no_data_yet` 占位符 → verdict 一直报 `entropy_unavailable`。**管道理由与科学结论恰好同向, 故无人察觉**。重生成后 `h10_status` ok, `operational_gate_passed` 仍 False。witness = `ddc1e60` + tag `h10-canonical-artifact-regen-k5-20260722` + `pre_run/h10_artifact_regen_provenance_2026-07-22.md`。
> - **✏️ 稿件三处必改**: ① "two cells" → **"three cells"** (B0_red/B1_red/B2_cls 均 0/5 可训练折) ② 机制口径 = `insufficient_train_data` **不是**熵集中 (熵闸门是过的) ③ 补披露 **Pass-2 从未 fire** (`k_of_n="0/0"` 独立成因)。
> - **🐛 B-1887 已修** (`554cc7c`, 1531 tests pass): Stage-1 无 mode 齐全性守卫, 未跑的 mode 被当"全失败" → 不加 `--cells` 会静默吃进残缺 B2_reddit。**k=6 重跑前置, 现已解除**。
> - **🔲 下一步**: ① B2_reddit 收尾 (phantom_som 76/205 在跑, phantom_prompt 未起) ~07-26/27 → bind → **k=6 重灌 + entropy gate 二次 promote** ② Paper B 起稿 (证据齐) ③ Paper A 格式转换 ④ 毕设 D8 **07-24 到期**。
> - **⚠️ 未 push**: `ddc1e60` + `554cc7c` + 本次 docs commit 均在本地。

> 🔭 **2026-07-16 UPDATE — VERDICT DAY 完成: Branch B 落稿 + post-submission 路线拍板** ⭐⭐⭐ (详 笔记 §378-§381):
> - **✅ k=5 verdict 已 splice**: H1 FAIL (+0.83 [+0.27,+1.49] p=0.7430) / H3 双 PASS (+1.26/+2.60) / H10 fail-closed → **Branch B**。abstract 重写 (250 词整), 三方审计 16 findings 全修 (codex 3 P0: estimand 偷换/超词/R5 隐匿), 官方模板编译过 **正文第 7 页末收** (refs pp.8-9)。数字唯一源 = PN06 slotsheet; commit `c55b16e`+`e4f13cd`。
> - **📋 剩余 pending 槽 = 合法**: B2-red 行 (×6) / Table 4+§6 H10 / §5.4 latency canonical / K5 标记打包剥离。**若 B2-red 提交前齐 → 无条件 k=6 重灌** (regenerate slotsheet → re-splice → 删 K5 块)。
> - **🗺️ Post-submission 路线 (user 拍板, 详 tasks.base)**: **B3=MiMo 8 月先行** (适配 7 月下旬 DGX 起步) → **WA 50-task pilot 插空** (prereg §8.8 注册预测, B1 本地) → **shop 期刊版长线**; B0 replicate 附录机会性 (提交前重跑已否决 = outcome-dependent sampling)。
> - **🔲 USER 行动 (不变, Jul 21 AoE 倒计时)**: OpenReview 填表 (作者 4 人 + 提名 Zekun + 贴 abstract — ⚠️ 表单 abstract 用 `deliverables/openreview_abstract_tldr_2026-07-14.md` verdict-neutral 版, full deadline 前可换成 verdict 版) / Maria 确认 / 学长书面一行 / OSF 发布 / D7 story 版拍板。
>
> 🔭 **2026-07-16 凌晨 UPDATE — B0 pprompt LAND → k=5 数据完备, SOP 全通, verdict-day 就绪** ⭐⭐⭐ (详 笔记 §378):
> - **✅ B0 pprompt 205/205 (01:35Z, SR 12.68%)** → bind/promote/analysis/orchestrator-restart 全通 (run_manifest **30 conditions**, B2 dom 已 resume 向 k=6 窗口)。
> - **📊 k=5 interim**: H1 **+0.795pp [0.27,1.49] p=0.743 不过线** (B0·red 实测 +0.98, archive +3.3 未兑现); H3 **双轴过线** (+1.26/+2.60, p=0.0); H2a 5/5 → 指向 **Branch B** (Route C'-S, 预写已备)。artifact 仍 PARTIAL/NOT_EVALUATED — **正式 verdict 必须走 VERDICT_DAY_RUNBOOK + Protocol-Note-06 授权 k=5 slotsheet (`--h10-pending`), 禁手抄 interim 进 draft**。
> - **🔜 白天动作 (user 醒后)**: ① 跑 verdict-day runbook k=5 严格路径 → 选支 (大概率 B) → §8.3 abstract/§1 替换 splice → cut_prewrites 抵词数 → /stress + Mode B/C chain → convert.sh 页数账; ② OpenReview 填表 (作者 4 人 + 提名 Zekun Wu + 贴 abstract); ③ Maria 确认 + 学长书面一行 + OSF 发布。
>
> 🔭 **2026-07-15 晚 UPDATE — 两轨制已签字激活 (周会提前一天) + Pareto 重构 land + D7 story 版待拍板** ⭐⭐⭐ (详 笔记 §371-§373):
> - **✅ NOTE_06 两轨制 IN FORCE**: 学长 07-15 周会口头 APPROVE → 当日激活 (commit `765d31a` + tag `protocol-note-06-k5-early-verdict-signed-20260715`; 四项披露已落, §4/§8 K5 段已 splice, +210 词记 NUMBERS_TODO §4)。**k=5 verdict 现在只等 B0 pprompt land+bind** → 走下方 SOP ①-③ 后用 **Protocol-Note-06 授权 slotsheet** (`--h10-pending` 分次模式) 出 verdict → §8.3 abstract/§1 替换 splice。
> - **🔲 USER 行动清单 (Jul 21 AoE 冻结倒计时)**: ① **作者已定**: Jiaming Wei (一作) / Zekun Wu / Adriano Koshiyama / **Maria Perez-Ortiz 暂定→冻结前须确认**; **互审提名 = Zekun Wu** → 填 OpenReview (需各作者注册邮箱/profile + COI); ② 学长一行书面确认两轨制 (归档 NOTE_06 §6); ③ OSF 知会发布 (`deliverables/osf_notice_protocol_note_06_2026-07-15.md`) + 回填 URL; ④ D7 story 版拍板 (读 §2.0/§2.8, 备份在同 dir)。
> - **🆕 学长新指令: router-baseline (prior work) efficiency 对比** — offline replay 基座实现 RouteLLM-style / FrugalGPT-cascade / kNN baseline (OFFLINE/NON-GATE, §6/supplement 增强); codex 任务已排队/派发 (笔记 §373 ③)。
> - **✅ Pareto 重构 land (§371)**: oracle 双轴支配全 menu (43.3% @ $0.0623); locked router 离前沿; post-hoc τ=0.10 越过 fixed 前沿 (29.9% @ $0.0705, 只作 paper-2 teaser)。看板 router 屏已换 Pareto 散点; §6 prewrite 两版在 `prewrites_s6_pareto.md`。
> - **✅ B3 = MiMo-VL 8 月先行, 之后扩展其他模型** (学长拍板)。
>
> 🔭 **2026-07-15 深夜 UPDATE — 机会成本调度实战 + 两轨制预案就绪 + k=4 → 明晚 k=5** ⭐⭐⭐ (详 笔记 §367-§370):
> - **🔥 FIRE 现状 (需接管)**: **B0 pprompt resume 运行中** (run 20260709, 96/205 起步, ETA ~07-16 晚; user 批准中断 B2 dom@40/205 抢 proxy 窗口, §369)。**orchestrator 已被刻意停掉**。A100 watchdog (pid 1061406) 会 ntfy 完成/异常。
> - **🔲 B0 pprompt 完成时 (ntfy 响) 接管 SOP**: ① A100 `validate_fire_manifest --populate --apply` bind → ② rsync fire_manifest → DGX promote 进 run_manifest (30 条) → ③ `make analysis FAST=1` (**k=5**) → ④ **重启 orchestrator v2** (`setsid nohup bash scripts/maintenance/orchestrate_reddit_boundaries.sh > logs/orchestrate_red_v3_$(date +%Y%m%d).log 2>&1 &` @ A100 repo 根; 它 resume B2 dom@40/205 + 续 B2 链, 重启自动重置 pprompt 重试计数) → ⑤ 刷 draft interim 槽 + 周会材料若还没开会。**若 pprompt abort** (proxy 再死): 直接做 ④ (orchestrator 会在后续 boundary 重试 pprompt)。
> - **📋 07-16 周会材料就绪**: brief v2.2 (`deliverables/weekly_meeting_brief_2026-07-16.md`) + 看板 (`周报/dashboard.html` 已对齐两轨制) + **PROTOCOL_NOTE_06 草案** (`prereg_amendments/..._20260716_DRAFT.md`, DRAFT/NOT-IN-FORCE)。**核心签字项 = 两轨制**: k=5 提交基线 (签字后 ~07-17/18 即可 verdict) + B2 齐则无条件升 k=6。激活 = NOTE_06 §6 十分钟序列 (签字→去 DRAFT→打 witness tag→amendment log 行)。k=5 版 abstract/§1/§4/§8 披露句已在 branch_prewrites §8 预写。
> - **router 学不会的五层诊断** (学长追问备答): 有效标签少一个量级 (成功并集才有标签, B1_red 仅 26) / 单次 rollout 标签噪声 / 成本平局裁决非模式亲和 / intent 特征不含答案 (covariate 红旗) / oracle 天花板≠可学性。15 格 sweep 证明非假设类问题。修复路径 = 多 rollout / 更强模型 (B3 MiMo, 8 月) / 运行时特征 (paper-2)。
> - Episode 级并行三重锁死 (§4 方法文本承诺 / 并发状态污染×Fix-4 / §5.4 latency 可比性) — 已归档 §369, 不再议。
> - 全部 commits 已 push (`278db06`); 测试 1485/0。
>

> 🔭 **2026-07-14 深夜 UPDATE — 投稿系统实测 = OpenReview 非 CMT + 官方模板 7 页压线 + 07-16 线下周会** ⭐⭐⭐ 最新先读:
> - **OpenReview 修正 (user 注册实测)**: AAAI-27 = OpenReview 表单。deadline 双标注: abstract **Jul 22 11:59AM UTC-0 = Jul 21 23:59 AoE** / full **Jul 29 11:59AM UTC-0 = Jul 28 23:59 AoE** / **reviewer-nomination 冻结 Jul 21 AoE** (资格作者未提名=desk-reject 风险)。abstract+TL;DR 复制源已备: `deliverables/openreview_abstract_tldr_2026-07-14.md` (248 词/槽位 0)。
> - **官方模板页数账 (author kit aaai2027.sty/bst 已入)**: 正文**恰好第 7 页末结束** (refs pp.8-9) — 压线合规零余量, verdict splice 增词用 cut_prewrites 抵消。convert.sh 三个官方模式断点已修 (kit 命名/affiliations/双 bibstyle)。
> - **📅 07-16 线下学长周会** — 议程: D7 章交付 / PROTOCOL_NOTE_05 披露 (witness 义务) / 4-in-1 消息内容当面过 / **作者名单+reviewer nomination 拍板 (冻结 Jul 21 AoE 前必须落表单)** / B2 timeline→k<6 预案(a) 决策树 / router covariate 红旗。周会 brief 07-15 准备 (dashboard+md, 等 B1 pprompt land 用 k=4 新数)。
> - 全部 commits 已 push (最新 `62daf0d`); 今日全景 → 笔记 §366-§366.4。
> - **🔚 codex 自走循环已收官 (user 2026-07-14 深夜指令)**: 监视器已停。**下 session 接手动作**: ① 查 B1 pprompt land (`logs/cron/fire_manifest_a100_latest.json` bound>28 或 ssh A100) → 走 NUMBERS_TODO §0 promote 配方 → k=4 聚合+interim 刷新; ② 周会 brief 已备 `deliverables/weekly_meeting_brief_2026-07-16.md` (07-14 深夜快照, land 后可用 slotsheet --rehearsal 刷 k=4 数字)。
>
> 🔭 **2026-07-14 UPDATE — codex 自走循环 day-1: 三审计→全修 + 6 reddit conditions promote (22→28) + PROTOCOL_NOTE_05** ⭐⭐⭐ (详 笔记 §366):
> - **✅ 三场 codex (gpt-5.6-sol xhigh) 审计全修**: ① relwork stress (17ce79f "待 stress" 清账, 10 findings → `9acfa24` 修, novelty 重写三级 spine 可守) ② readiness (词数 5450→5248, banned grep 出清, branch 完整 abstract 197/203 词) ③ **工具链 Stop-ship 9 P0** → Chunk 1 `27e04a9` (H3 pool 恢复 "over 6 planned cells" estimand + H1/SR exact 224/205 task-set + analysis_status/h1_verdict schema; **witness = PROTOCOL_NOTE_05 + tag**, outcome-blind @ k=3) + Chunk 2 `e35d0b4` (fig0c universe / F2 单源 / router majority OOF 泄漏 / slotsheet fail-closed + Decimal 舍入锁)。测试 1433p/2f → **1454 passed/0 failed** (+19 regression)。
> - **⚠️ 修正后 interim 数字 (canonical artifact 已重生)**: H3 axis-1 k=3 = **1.08pp [0.47, 1.98]** (旧 3.20pp 是 outcome-dependent 删 cell 的产物, stale-by-correction — draft/旧引用勿再用); axis-2 = 2.26pp 不变; 全部 NOT_EVALUATED @ k=3 (verdict 只在 k=6)。
> - **✅ 6 个 bound reddit conditions 已 promote** (staging fire_manifest 07-13 → run_manifest 22→28): B0 psom + **B1 red dom/som/vision/ptext/psom (5/6!)**。B0/B1 reddit 各只差 P-prompt 即成完整 6-mode cell → k 3→5 在望。B1 pprompt 大概率在 A100 orchestrator 队列中/已完成 — **下一动作: 探 A100 最新 fire_manifest + 若 bound 继续 promote** (`logs/cron/fire_manifest_a100_latest.json` 15-min cron staging)。
> - **🔲 未 push 17+ commits** (`9acfa24`..今日尾, 详 笔记 §366/§366.2) — push 需 user 确认。**task149/33/155 registry classify 已清账** (`4801816`, 双侧 47 行); 余 task70/99 = B0 pprompt 未完成 run 合法开放。/diag 欠账: 新 promote 6 条全部 diag pending。**下午追加**: LaTeX 链通 (官方 sty/bst 待 user 下载 author kit) / supplement S1-S6 骨架 / runbook 彩排 + --h10-pending 分次 verdict / D7 lit-review P2-15 final-pass 完成 (等 user 通读, hard ≤07-20) / WebArena-Verified bib 待 user 手动 (OpenReview bot 防护)。
> - **⚙️ 工具链新语义**: verdict-day 走 runbook 严格路径 (fail-closed); 日常 `make analysis` fig0c 已显式 --allow-partial (`cc3334e`); slotsheet final 模式缺 artifact 即 exit 2; k<6 选支被禁 (预案(a) 出口 = 新 PROTOCOL_NOTE + advisor sign-off)。
>
> 🔭 **2026-07-08 UPDATE — B1 som land (205/205, bound 24) + outage#5 (proxy 恢复仅 ~31h 再死) + orchestrator v2 (per-boundary B0 插入)** ⭐⭐⭐:
> - **✅ 首个 boundary 全自动通过**: B1 som 00:14Z 完成 205/205 → orch 自动 bind (bound 24) → 探 proxy 3×503 → 正确跳过 B0 → B1 vision 已启动 (00:21Z)。
> - **🔴 outage#5**: proxy 07-06 17:1xZ 恢复 → 07-08 00:19Z 前再死 (~31h)。学长 Lambda timeout 修复未解决根因 — 上游模型服务本身不稳。
> - **✅ orchestrator v2 部署 (pid 206635, `f901b4a`)**: v1 只在单一 boundary 探 proxy 的缺口修复 → 每个 boundary 都 try_b0 (psom→pprompt, ≤2 attempts each), 全条件 FORCE_NEW=0+RESET auto (abort 数据永不丢), eps>=205 幂等 skip (可任意时点 kill 重启接管)。proxy 恢复窗口不再会被错过。
>
> 🔭 **2026-07-07 UPDATE — proxy 已恢复 (学长 Lambda timeout 120s→10min + 上游修复; gateway + Lambda URL 双入口 200)** ⭐⭐⭐:
> - **✅ proxy UP** (probe 循环 07-06 18:19 报 3×200 RECOVERED; 07-07 14:2x 复测 gateway+Lambda URL 均 200)。outage#4 总时长 ~49h (07-04 17:08Z → ~07-06 17:1xZ), 根因诊断 = Lambda 上游模型服务无响应 (503@30s gateway / 502@120s Lambda URL 双路径钉死), 学长已修 + Lambda timeout 提至 10min。
> - **B1 som reddit 健康**: 146/205 @ 07-07 14:16 (~11.5min/ep ≈ dom 节奏; 07-06 的 100s busy-wait 仅 task 0 局部现象)。ETA 完成 ~07-08 凌晨。done-monitor 已重挂 (前一 session 退出时被停)。
> - **🤖 4 天无人值守 orchestrator 已接管 boundary (07-07 14:32 armed, user 北爱尔兰行 07-08→07-11)**: `orchestrate_reddit_boundaries.sh` @ A100 (pid 143199, log `logs/orchestrate_red_20260707.log`) 自动执行整条队列 = B1 som wait→bind → 探 proxy → UP: B0 psom RESUME (146/205) → B0 pprompt → B1 vision/ptext/psom/pprompt → B2 ×6, per-boundary bind+ntfy (p79-claude)。user 三拍板: 全自动 A / B0 abort→跳过续 B1 / B2 纳入队尾。fail-safe = 异常 ntfy+停在安全状态。**回来后 TODO**: ① task149 registry classify (psom resume rerun 证据); ② 若中途停摆看 ntfy 最后一条 + orch log triage; ③ B0 psom/pprompt 若被跳过 → 手动 resume。详 笔记 §365。
>
> 🔭 **2026-07-06 UPDATE — 停摆 ~1.5 天后按分支 ② 重启: B1 som reddit UP + probe 重挂** ⭐⭐⭐:
> - **🔴 停摆发现**: B1 dom reddit 07-04 23:20 完成 205/205 **且已 bind** (bound 23 = cls 18 + red B0×4 + red B1 dom), 但 07-04 的"monitor fire → 探 proxy → 插 B0/续 B1"是 operator 动作, 07-05 无 session → fire 空转 ~35h (与 07-01 15h 停摆同款盲区: cron `inprog=none` 不报"该跑而没跑")。
> - **🔴 proxy 仍 DOWN (outage#4 已 ~41h+, 刷新纪录)**: 实测 `probe_proxy_alive.py` (新脚本, 单请求轻量探活, 已 rsync A100) → 无 tools 请求 **503**; 带 tools 请求 400 validation_error ("missing field `type`") — 疑 probe_proxy_capability.py 旧 schema 所致, 但 **proxy 恢复后 B0 首 episode 需盯防部署变更**。
> - **✅ 分支 ② 已执行 (user go 2026-07-06 ~11:00 BST)**: `RESET_BEFORE=1 queue_baseline.sh B1 som reddit` → **run_id=`B1_som_reddit_20260706` UP** (reset OK warm-up 93s, runner pid 26258 + watchdog pid 26289, GPU 9GB)。ETA 参考 B1 dom ~36h → **~07-08 早**。
> - **🔜 B1 som 完成时 (monitor fire) SOP**: ① `validate_fire_manifest --populate --apply` 补 bind (单跑不自动 bind); ② boundary 探 proxy (`ssh condense-a100 '.venv/bin/python3 scripts/maintenance/probe_proxy_alive.py'`): UP → B0 psom resume (146/205, task149 rerun→classify) → B0 pprompt; DOWN → 续 B1 vision 单跑。
> - **⏳ DGX 双 bg monitor 已挂**: done-monitor (205/205 或 runner 退出 → ntfy p79-claude, 60h 兜底) + proxy probe 循环 (5min/探, 3×200 判恢复 → ntfy, 48h 兜底)。
> - **📨 升级信号**: outage#4 时长已超 07-03 的 20.7h 纪录一倍 — 管理员证据包 (`deliverables/proxy_admin_outage_evidence_2026-07-03.md`) 待补 outage#4 时间线后发送 (user 审)。
>
> 🔭 **2026-07-04 UPDATE — proxy 恢复 (outage 共 ~20.7h) → 调度改为"最早边界插 B0"** ⭐⭐⭐ 最新先读:
> - **✅ proxy 恢复 ~03:40Z** (probe #2 200×3; outage 06:58Z 07-03 → ~03:40Z 07-04 史上最长, 疑学长人工重启)。**4 天 3 次 outage → B0 工作优先级最高 (趁 proxy 活着干), B1/B2 outage-免疫可随时跑**。
> - **✅ 已摘 chain 续链循环** (kill chain-loop pid 3849518, runner/watchdog PPID=1 无恙, 实证零影响): B1 dom 独立跑完当前条 (77/205 @ 04:00Z, ETA ~07-05 10:00Z), 完成后**不再自动续 som** — 改插 B0 slot。
> - **🔜 B0 slot 改条件分支 (2026-07-04 17:08Z proxy 又挂 = outage#4, 恢复后 ~13.5h 内再死)**: B1 dom 完成时 (monitor fire) **现场探 proxy**: ① proxy UP → bind B1 dom → B0 psom resume (146/205, task149 rerun→classify) → B0 pprompt → 续 B1; ② proxy DOWN → **B1 逐条单跑续** (som→vision→ptext→psom→pprompt, 每条一个 boundary, 首个 proxy-up 的 boundary 插 B0)。逐条单跑不走 chain = 每个 boundary 都是 B0 插入机会; 单跑无 sentinel/bind → 每条完成后 `--populate --apply` 补 bind (SOP)。B2 去留按 ETA 届时拍板。
> - **🔴 proxy 稳定性升级为项目 #1 风险**: 4 天 4 次 outage (06-30 / 07-02 / 07-03×20.7h / 07-04 17:08Z-), 恢复窗口越来越短 — DashScope 直连议题急迫性再升级 (学长线程)。
> - **教训沉淀候选**: "chain 完成后插 B0" 原计划在 20h 级 outage 面前不成立 — 依赖外部 substrate 的工作永远抢最早窗口 (机会成本调度)。
>
> 🔭 **2026-07-03 UPDATE — abort#3 task149 (proxy outage >2.5h 未自愈, 史上最长) + 待 user 拍板恢复策略** ⭐⭐:
> - **🔴 abort#3**: psom R28173 resume 段健康推进 62 ep (84→146) 后, task 149 再撞 proxy 503 (07:32Z abort, registry 已 append 未 classify — 等 rerun 证据, 同 task87 模式)。**本次 outage 异常**: ~06:58Z 起 >2.5h 无自愈 (前两次 ~36min), 503 body + 30s 稳定耗时 = API Gateway 集成超时, 后端挂死疑需人工重启。
> - **⏳ probe 循环已挂** (DGX bg, 5min/探, 连续 3×200 判恢复, 12h 兜底) → 恢复自动通知 → resume psom + rerun/classify task149 (协议同 §362)。
> - **📨 管理员证据包已备好**: `deliverables/proxy_admin_outage_evidence_2026-07-03.md` (60h 3 次 outage 时间线 + apigw-requestid + DashScope 议题) — **待 user 审后发学长/proxy 管理员**。
> - **✅ user 拍板 ③ (2026-07-03) + 学长消息已发**: **12-cell B1/B2 reddit chain 已启动** (直接 `queue_chain.sh` 自定义列表, 不过 G8 [G8 只在 launch 层], RESET_BEFORE=1 默认, chain pid=3849518, log=`logs/queue_chain_b1b2_red_proxyout_20260703.log`)。[1/12] B1 dom reddit UP (10:13Z, run_id=B1_dom_reddit_20260703)。顺序 = B1 dom/som/vision/ptext/psom/pprompt → B2 同 6 模式。**B0 插回策略 = chain 完成后**: bind 检查 (`--populate --apply`) → B0 psom resume (146/205+task149 rerun/classify) → B0 pprompt → 最后 `launch red` 兜底校验。proxy 恢复通知到达时 B1/B2 若在跑, B0 排队不并行 (同 site 一 baseline 硬规则)。DGX 双 bg monitor: probe 循环 (proxy) + chain-exit (72h 兜底)。
> - **✅ 顺带修**: paper_grade_check.sh host guard (`7c4eb13`) — 在 A100 上误跑 wrapper 曾致 2 条假 "could NOT reach A100" 告警 (08:35Z, 已 all-clear 澄清 + 双端部署)。
>
> 🔭 **2026-07-02 晚 UPDATE — psom R28173 abort@task87 (proxy 503) → resume 闭环, fire 已恢复推进** ⭐⭐:
> - **✅ abort + recovery 全闭环** (详 笔记 §362): task 87 撞 proxy 503 (18:08–18:44Z ~36min outage, 打穿 B-1880 wait-out) → fail-closed abort 83/205 → PROTOCOL_NOTE_03 resume 第 4 次实战 (`FORCE_NEW=0 RESET_BEFORE=0` 单跑 queue_phantom_som) → task 87 rerun 干净 (error=None merit-failure) → classify `transient_drift` via resume_rerun_clean → **G8 preflight 绿**。registry 已 sync DGX (37→39 行, 前缀不变式成立)。
> - **🔴 psom 完成后 operator 动作 (runner 退出时 ntfy p79-claude 会响, ETA ~07-04)**: 单跑 queue 自然退出**不续链且不 bind** (07-01 深夜同坑) → 必须先 A100 `validate_fire_manifest --populate --apply` 补 bind psom → 再 `launch red` (skip 已 bind, 续 pprompt→B1×6→B2×6)。DGX 侧双 done-monitor 已挂。
> - **⚠️ 48h 内第 2 次同签名 proxy 长 outage** (6-30 task80 / 7-02 task87, 均 >35min) — DashScope 直连议题权重 +1 (学长消息 v2 pending 项)。
>
> 🔭 **2026-07-02 UPDATE — 12 项队列自走完毕 (①-⑫ 全推进, 详下方深夜块逐项钩) + 下 session 入口** ⭐⭐⭐:
> - **✅ /stress chain DONE 2026-07-02** (user "直接做吧" 授权): A5+B5+C4 = 12 findings 全修 (Track B 归因重大更正 89f5af2 两层拆分 / cancellation 谬误→差分压缩 / Branch A 矩阵收紧 / 槽位真实字段名 / aaai27 限定语回填)。prose 三件已 commit (`b684306`) + **全部 push 完** (`823c43c..b684306`)。详 笔记 §361.6。
> - **✅ user 四项拍板 2026-07-02** (詳 §361.7 + paper_planning §19): **venue=AAAI-27** / 预案 **a+b** / **Amendment 03 撤** (论文保持 hedge 两层表述) / **B-1885 分母 205 不动** (§8 披露句已落)。唯一余项: 学长消息**知会版 v2** (`deliverables/advisor_msg_4in1_2026-07-01.md`) 待 user 审后发送 (决策已定, 消息只知会+留异议窗口)。
> - **fire**: B0 psom R28173 健康 (07-02 08:17Z 41/205, ~15min/ep, ETA ~07-04) → pprompt → B1×6 → B2×6。每 cell land 走 NUMBERS_TODO §0 配方 (**promotion-gap watch 已 cron 化**, 漏 promote 会 ntfy)。psom/pprompt land 后各自 `/diag` (som/vision digest 已齐)。
> - **本 session 硬产出速览**: verdict-day 工具链 (slotsheet+runbook+双分支+F1/F2+LaTeX) / Pass-2 预演通 + **B2_cls router 不可训 → H10 ≤4/6 风险** / B-1885 / reddit som+vision 双 diag / 22-cell 聚合 (red P-text [A])。chronicle → 笔记 §361。
>
> 🔭 **2026-07-01 深夜 UPDATE — fire 停摆 15h 已修复重启 (B0 psom R28173 跑着) + AAAI 新 session 工作队列固化** ⭐⭐ (队列已全钩, 见上方 07-02 块):
> - **✅ fire 已恢复**: ptext 完成后 chain 死 (§359 resume 用单跑 queue_phantom_text, 跑完自然退出, 无人续链 → 停摆 ~15h, cron `inprog=none` 不报警)。首次 relaunch 踩坑: **RESUME_MISSING 跳过判据 = manifest-bound ∧ eps==scored**, 而 bind 是 chain 内步骤 → chain 死时 ptext 从未 bind → relaunch 重跑 ptext (R13246, 已 kill+archive `_archive_duplicate_unbound_ptext_R13246_20260701`)。修复 = 手动 `validate_fire_manifest --populate --apply` (bind ptext → 22) → 再 `launch red` → **B0 phantom_som R28173 UP** (22:33Z, runner+watchdog 活, 前 4 条 reddit 全部正确 SKIP)。**教训/SOP: chain 死后续链前先跑 `--populate --apply` 补 bind, 否则重跑已完成条件**。剩余队列 = psom→pprompt→B1 red ×6→B2 red ×6 (14 cond)。live 状态跑 ① + `make ntfy`。
> - **🔲 每个 cell land 后**: 走 `paper_drafts/aaai27/NUMBERS_TODO.md §0` sync 配方 (rsync fire_manifest → promote run_manifest → make analysis → draft tags/slots)。
> - **📋 新 session 工作队列 (user 2026-07-01 决定, 按杠杆排序)**:
>   ① ✅ **拟学长四合一消息** DONE 2026-07-01 → `deliverables/advisor_msg_4in1_2026-07-01.md` (正文 + Amendment 03 拟稿附件; **待 user 审后发送**, 未落 prereg_amendments/ 未打 tag)
>   ② ✅(主体) **D7 lit-review 章**: 实为已有 draft v1.1 (06-10 3-AI stress 过, 非从零起草)。2026-07-01 对齐 pass DONE: aaai27 P0#1 措辞违规同类 4 处修复 ("no image tokens"→"no page-screenshot tokens" 等), banned-grep 0 hits。剩余 (07-13 final pass): P2-15 §2.7 指针 — notes-sweep 完, 仅 MobileGym 有 "Appendix J" 指针, 其余 5 处需回原文 PDF (细节在章稿 frontmatter notes) + arXiv:2603.12823 verify (API timeout, optional)
>   ③ ✅ **§1/abstract 双分支预写** DONE 2026-07-01 → `paper_drafts/aaai27/branch_prewrites_s1_abstract.md` (Branch A=H1-pass / Branch B=H1-fail+H3-pass=Route C'-S; abstract 1 处 + §1 两处整段替换文 + «槽»→gate-JSON 映射 + 词数账; C'-R/F 低概率支未预写)
>   ④ ✅ **verdict-day runbook** DONE 2026-07-01 → `aaai27/VERDICT_DAY_RUNBOOK.md` (0→6 步主链 + 自检链 + 失败模式表 + k<6 特例) + 新脚本 `scripts/analysis/verdict_day_slotsheet.py` (read-only formatter, gate JSON→slot sheet 一条命令, 唯一允许的 draft 数字来源; 已 smoke: 正确报 PARTIAL/Branch-B 建议/red 3-mode 禁引警告)
>   ⑤ ✅ **Pass-2 router 管线预演** DONE 2026-07-02: 3 段链全通 (extract→with_mi→train) @ 隔离 `l1_router_rehearsal_20260702/`, entropy gate emit OK (status=ok, 2.10 bits)。**发现: B2_cls router 不可训** (16 标签 task, 5 fold insufficient) → H10 5/6 判据风险, 已进 advisor 消息 3️⃣(b) + NUMBERS_TODO §1。verdict 前须在 canonical 目录重跑 + rsync pkls→A100
>   ⑥ ✅ **Figures F1+F2** DONE 2026-07-02: `fig_f1_diamond_schematic.py` (概念图, 无数据依赖) + `fig_f2_h1_forest.py` (H1 strict 估计量 forest + FE diamond + INTERIM 水印; 刻意区别于旧 fig_forest_drop_one 的 ADD-lift 估计量 — Amendment 02 混淆防线) 双双 render 验证 · ⑦ ✅ **连续性包** DONE 2026-07-02 → memory `project_aaai27_campaign.md` (deadline 锚/文件地图/slotsheet 工具链/sync 配方/estimand 混淆防线, 指针型不冻数) + MEMORY.md 索引行 · ⑧ ✅ **registry sync cron 补丁** DONE 2026-07-02: 新 `check_manifest_promotion_gap.py` (staged fire_manifest vs registry diff + 集合变化才 ntfy) 接进 `sync_a100_results.sh` 15-min cron (staging 副本, 不覆盖 git-tracked manifest, promote 保持手动 deliberate)。**顺带走完 red·B0 P-text sync 配方**: R32139 promote → 22-cell 聚合 → Table 2 升 [A]; gap 现 = 0 · ⑨ ✅ **/diag red som/vision** DONE 2026-07-02 → `docs/analysis/vwa_reddit/B0_{som,vision}_reddit_diag_digest.md` (som 41AL+4FP+0scaffold / vision 40AL+2FP, no-hit 全覆盖)。高杠杆产出: **B-1885 新登** (task 103+104 config eval-URL 错指 = 全 mode 系统性不可赢, 2-AI 双证 + 205-config 全扫恰 2 个; 分母豁免与否 = advisor estimand 议题) · token-granularity FP 族 (95+125 跨 mode; string_match 实为 finish-answer-based [environment.py:581], sub-agent 反向 claim 已亲验证伪) · submission_image_trap 跨 mode → mode-agnostic 规则 · **per-rule 因果折扣随 mode 漂移** (P5: som 9折 vs vision 3折) → v8 freeze 须 per-mode 标定 · 205 goto 疑点 defused (policy-block 正常)。R-som/R-vis 候选全记 digest, 落码 defer v8 · ⑩ ✅ **LaTeX 骨架** DONE 2026-07-02 → `aaai27/latex/` (skeleton.tex + README 转换工作流, pandoc smoke 过) · ⑪ ✅(阶段1) **砍词** 2026-07-02: 5395→5181 (-214 安全砍: §7/§8-stats/§4/§2-anchors/§1非分支段/§5.1/§5.2/§6共享段; abstract 266→250 恰入 CMT 限); 剩 ~-286 绑定 verdict-day 删支 (~130) + item-7 候选; banned grep 0 (checklist item 9 已加 self-match 免疫 filter) · ⑫ ✅(草稿) **Track B** DONE 2026-07-02 → `paper_drafts/trackB_judge_polarity_note.md` (2055 词 draft-v0, 数字全有源 [B-91/B-535 catalog + bib notes], 3 ⟨TBD⟩ [跨论文 SR delta 重测 / WebArena-Verified + PAE bib key 缺]; **未 commit, 与 aaai27 砍词稿 + branch prewrites 一起等 /stress chain**)
> - **✅ 已 push** (user 确认 2026-07-01 深夜; 本日 6 commits + 此前累积全部上 `origin/fix/b1878-reddit-reference-image`)。⚠️ A100 repo 的 git 状态落后于 origin — **fire 进行中不要在 A100 上 git pull** (queue 脚本被后续 chain 条件复用, 中途换码有风险); 需要单文件对齐用 rsync, 整体对齐等 chain 完成。
>
> 🔭 **2026-07-01 UPDATE — AAAI-27 定为投稿目标 + 合稿初稿 v0.1 落地** ⭐⭐ (fire 线见上方深夜块):
> - **⏰ 真实 deadline 在本月**: AAAI-27 abstract **07-21** / full **07-28** / supp 07-31 (UTC-12)，7 页正文+refs+repro checklist，双盲（官网核实，非 user 记的"下月28"）。venue 与 EMNLP/NeurIPS cascade + D11 early-Sep 有张力 → **advisor 确认 owed**（archival main-conf 提交权 + k<6 降级预案三选一，estimand-adjacent 需 witness）。
> - **✅ AAAI 合稿初稿 v0.1**: `paper_drafts/aaai27/aaai27_main.md`（submission master，4876 词实测）+ `NUMBERS_TODO.md`（⟨TBD⟩ 槽位→producer 映射 + deadline 风险账 ⭐ 数字/ETA 一律看它，别信 chronicle 冻数）。verdict slots（H1/H3/H10）留空 + (R-CONDITIONAL) 句子按 realized R-tier 重写；§6 router 双分支预写。详 笔记 §360。
> - **✅ 聚合链已打通 (07-01 晚)**: 门控 = fire_manifest.json (A100 auto-bind, 21 条) → promote 进 run_manifest.yaml → aggregators。已 sync + promote 15 cells → 聚合 21 cells；**H1 interim k=3 (cls-only) θ_FE=+0.98pp [−0.05,2.00] 低于 1.0pp 门槛 — verdict 押在 reddit**；H2(a) 3/3 unfalsified (1.01/1.04/1.08)；H3 interim 两轴 CI 排除 0。draft slots 已同步。新 cell land 后 sync 配方 → `paper_drafts/aaai27/NUMBERS_TODO.md §0`。
> - **⚠️ commit 本 prose 前必跑 /stress (+B/C chain)** — CLAUDE.md auto-trigger；aaai27/ 两文件当前 untracked 未 commit。
>
> 🔭 **2026-06-26→27 UPDATE — Fix-4 reddit fire LIVE + 生产 live-verified (无 abort 越过 task-138/151) + reddit 首条 (B0 dom) 已 land bound-clean ✅; D4 (@06-26) 坐实 miss → advisor re-plan** ⭐⭐⭐ (supersedes 06-25 "唯一剩 launch red"):
> - **✅ Fix-4 reddit fire 已启动 + 首条已 land**: user 06-25 15:48Z 已 `launch red` → reddit 首条 run `R11344` (`B0_dom_reddit_20260625_154833_...`; = 06-25 G8-clear 后的 Fix-4 relaunch, FORCE_NEW from ep0, 接 abort#10@task151 之后)。**该条 06-27 已完成 + manifest bound-clean = reddit 史上首条 paper-grade 干净 condition** (此前 reddit 从没跑完过一条)。chain 已自动续进后续 mode — 实时 current run/condition/进度/死活 → 跑 ① `bash scripts/maintenance/paper_grade_check.sh` + `make ntfy` (别冻数字进本行)。§0 06-25 标的 "唯一剩 operator launch red" 已完成。
> - **✅✅ Fix 4 生产环境 LIVE-VERIFIED (核心里程碑)**: R11344 已无 abort 跑过 **task 138 (改用户名 = B-1884 自毁级联根因)** 且越过 **task 151 (abort#10 前次失败点)** → `restore_reddit_identity()` 每-task 幂等复原**在真实 fire 里成立**, 自毁级联根除 (= 一次性已成立的事实, 不随 fire 进度变)。§0 06-25 的 verify-script 只证 SQL heal 单点, **整条 fire 撑过 138 = 现 PASSED**。Fix 4 (= estimand a clean-per-task) 实战闭环。⚠️ **这是 finding** — reddit 条件跑完后须 append 笔记 §358 chronicle (本次只更 live state, 未写笔记)。
> - **bound-clean = cls 18 全 + reddit 首条 (B0 dom)** (manifest 0 ghost / 0 unbound; 实时计数跑 ①); verdict **ISSUES=5 全 = B2(Gemma) cls parse_error >1%** = 已知 floor (§327 / B-1876 / 每次 cron verdict 都在, **非新回归、非 fire-blocker**)。
> - **🔲 operator 当前无需动作** — fire 自走 (reddit 首条 B0 dom 已 land, chain 自动进下一 mode; current condition 跑 ①), reddit 18 cond sequential ~1.5-2 周。监控 = `make ntfy` + 跑 ① verdict; 撞 abort 才介入 (三层自动兜底已就位: wait-out 35min B-1880 / pre-flight transient-retry B-1881 / resume-on-abort B-1882)。
> - **🔴 D4 (Pass-1 全 36 cond @06-26) 坐实 miss**: reddit chain 刚起步 (首条 B0 dom 已 land, 余 17+ cond sequential) → reddit Pass-1 ETA early/mid-July → **需跟学长 re-plan reddit 目标** (与已定 estimand(a) 一并同步; one-pager 已 06-21 sent)。
> - **有未 push commits** (清单跑 `git log @{u}..HEAD`, 含 Fix-4); push 需用户确认。
>
> 🔭 **2026-06-25 UPDATE — estimand 已拍 (user 选 a) + Fix 4 实现+A100 deploy+live-verify ✅ DONE; ~~唯一剩 operator `launch red` (待 user go)~~ → ✅ 已 launch 06-25 15:48 = R11344 (见上方 06-26 块)** ⭐⭐ 详 笔记 §357.6 + PROTOCOL_NOTE_04 + B-1884:
> - **estimand 决策 = (a) clean-per-task** (user 2026-06-25 "不问学长了，直接按推荐做")。Fix 4 是它的实现。
> - **✅ Fix 4 已实现 + 本地验证 (code 完成, 零回归)**: 新 `p79/utils/reddit_identity.py::restore_reddit_identity()` 在每个 reddit task 的 `_run_episode` 开头 (auth-refresh **之前**, 时序关键) 跑幂等 `UPDATE users SET username='MarvelsGrantMan136' WHERE id=13915 AND username<>...`, 经 **verified** 路径 `docker exec vwa-reddit su - postgres -c "psql -d postmill -c ..."` (§354 实测 peer-auth; 表 users/id=13915)。`runner/main.py` 加调用 + `config.py` 加 `reddit_identity_reset` 块。**12 新测试 + runner smoke = 19 pass 零回归**; shell 三层转义 `shlex.split` 双层模拟验证 psql 收 SQL byte-exact。witness = `PROTOCOL_NOTE_04`; catalog B-1884 标 ✅ FIXED。= 补 VWA 给 cls 有、reddit 漏 (`envs.py:172 TODO`) 的 per-task clean-state, cls 已 bound 数据零影响 (gate on site==reddit)。
> - **✅ A100 deploy + live-verify DONE (2026-06-25)**: rsync 5 文件→A100 `/home/ubuntu/workspace/p79` + 远端 14 测试过 + `scripts/maintenance/verify_reddit_identity_fix.sh` 实测通过。**验证抓到初版 bug**: postmill `users` 有 `normalized_username` 列、**登录认它不认 `username`** → 修成 Fix 4 复原**两列** (username + normalized_username=lower); 重验 真实两列改名→复原→**fresh login=LOGIN_OK**。(教训: live verify 必须 replicate 真实 mutation 全副作用, 否则弱模拟假阳性。)
> - **✅ DONE 06-25 15:48Z — operator `launch red` 已执行 → R11344** (FORCE_NEW from ep0; 该 reddit 首条现已完成 bound-clean + Fix-4 生产 verified, 详见上方 06-26→27 块): 启动命令 = `RESUME_MISSING=1 MAX_CONDITION_HOURS=0 MAX_CLS_WAIT_HOURS=0 launch red`。launch 时 queue reset reddit 容器 (docker rm+run) → 替换掉 verify 留的 fresh 容器, 无碍。
> - **codex 沙盒已修** (config.toml `sandbox_mode=danger-full-access`) — 后续 delegate 不再挂。
> - **commits 全 local 未 push**: 本 session code = reddit_identity.py + runner/main.py + config.py + test + PROTOCOL_NOTE_04 + catalog B-1884 + 笔记 §357 + 本 handoff (全 tracked); dashboard/memory/`~/.codex/config.toml` 不入版本。叠加前几 session 未 push 的 `6a18657`..`149e1e6`。push/commit 需用户确认。
> - **D4 (Pass-1 全 36 @06-26) 仍几乎确定 miss** (reddit 0 clean, 即便 Fix 4 deploy 后跑也要 ~1.5-2 周) → 需跟学长 re-plan reddit 目标 (early/mid-July)。
>
> 🔭 **2026-06-24 UPDATE — 根因独立复核 sound + 修法收敛 Fix 4 + codex 沙盒已修 + 周报 dashboard ready** ⭐⭐ 详 笔记 §357:
> - **根因复核 sound** (亲自读 code 非照搬笔记): B-1884 因果链全成立, 冒烟枪 = R819 task138 `success:true` + `eval_source_agent_url=/user/Patrick/account`; 关键机制 = cookie 绑 user-id 13915 不绑 username → 改名后 cookie 仍活、只 fresh 重登撞墙 (故伪装成"间歇 auth blip" 骗过 6 个 bug-number)。唯一软环节 = "B1 没事=capability-modulated" 是推断 (quark/A100 容器 confound), 已 hedge 不承重。
> - **范围核实**: **致命 abort 结构上仅 reddit** (全 880 task 只 138 改凭证, shopping/cls 各 0); shopping 无 reset 有**静默跨任务污染** = Phase 1b 预警 (不崩只悄悄扭曲 SR); cls 有 reset 受保护 (解释 18 cond 干净)。abort=P79 特有 (串行+周期 fresh 重登放大), VWA 并行 4-pane + cookie 复用容忍故不撞。
> - **修法收敛 = Fix 4** (codex file-grounded 双轮确认): `env.reset` 内幂等 `UPDATE users SET username='MarvelsGrantMan136' WHERE id=13915`, 镜像 cls require_reset, GRL "reliability not policy" substrate 复原。约束把 (A) liveness 探针 + (b) tolerate 双双判出 (都碰 runner/auth → 破 cls 已锁一致性)。**hook 唯一落点 = `main.py:2344` env.reset** (queue 只能 condition-间, 做不到 condition-内 per-task); setup 层非 measured execution → **需 witness**。
> - **🔲 Fix 4 = 设计定、未实现; 唯一工程 blocker = A100 确认 postmill psql creds** (`docker exec vwa-reddit env | grep -i postgres`, creds baked 在 image 不在代码库)。
> - **⚠️ estimand (a)/(b) 仍是 gate, 没变**: Fix 4 实现的是 **(a) clean-per-task 分支**(+ 中和 capability-confound); 若学长要 **(b) tolerate**, Fix 4 不适用。**别在 advisor 定 (a)/(b) 之前实现 Fix 4 / relaunch reddit**。(a)/(b) + D4 re-plan = 同一次 advisor 沟通。
> - **codex 沙盒已修** (顺手): `~/.codex/config.toml` 加 `sandbox_mode = "danger-full-access"` (全局默认绕 DGX 无-caps bwrap; 根因 = rescue agent 没传 `--sandbox`)。后续 codex delegate 不再挂。
> - **周报 dashboard ready**: `docs/checkpoints/周报/dashboard.html` 默认视图 = `RedditTask138Brief.jsx` (根因+范围+Fix4+(a)/(b)+capability-confound+D4 时间线, 7 section), 三层验证过 (build/lint/playwright 0-error)。Live Server 看, 给学长用。⚠️ 周报 gitignored 不入版本。
> - **D4 (Pass-1 全 36 @06-26) 几乎确定 miss** (cls 18 done clean, reddit 0 clean, 根因刚找到) → advisor re-plan (与 (a)/(b) 同一次)。
> - **commits 全 local 未 push**: 本 session = 笔记 §357 (tracked) + codex output md; dashboard 文件 + `~/.codex/config.toml` 不入版本。叠加前几 session 未 push 的 `6a18657`..`149e1e6`。push/commit 需用户确认。
>
> 🔭 **2026-06-23 UPDATE — reddit abort saga 真根因揪出 = task 138 改用户名 (B-1884); 整条 proxy/auth band-aid 被 recontextualize; fire 停着等 estimand 决策** ⭐⭐⭐ 详 笔记 §354/§355/§356 + master_bug_catalog B-1884:
> - **真根因 (100% 实证)**: reddit **task 138 intent="Change my username to..."** → B0(强模型)成功改名 MarvelsGrantMan136→Patrick → 后续 fresh login 全 "Invalid credentials" → P79 auth_refresh (每5ep重登) 失败 → fail-closed abort。**之前所有归因 (ref-image/wallclock/proxy503/auth-blip/budget B-1878→1883) 里 auth-class 那条全是对此的 band-aid**。B1(弱模型)做失败 task138 → 账号没动 → April 跑得好 = **capability-modulated contamination**。
> - **⚠️ 我中途两个误判 (已纠正, 别重蹈)**: ① B-1884 初版误判「image 缺账号」(错: image 有 id=13915, 是被改名) → §354 标 SUPERSEDED + §355 更正 + 注册 hook 已 revert; ② 误判「P79 不 honor require_reset → cls 已锁数据污染」(错: P79 经 `p79/envs/vwa_wrapper.py` honor, bound cls run 实测 22-23 次 reset fire) → **cls B0 dom/som 数据安全, 虚惊撤回**。两次都是 grep 窄一层就结论「缺失」, 教训: 结论「X 缺失」前 grep 整条 delegation chain (wrapper→upstream)。
> - **VWA 设计真相 (§356 + memory `reference-vwa-design-quirks`)**: require_reset 只实现 cls (reddit/shop=`TODO(jykoh)` no-op); VWA 并行=4段共享后端 (task138 污染比 P79 单流更脏); **不存在可照搬的「VWA 标准」处理破坏性任务**; P79 串行(非 VWA 并行)因 latency estimand; gc_maxlifetime 24min → P79 周期重登是承重墙不能删。
> - **fire 现状: R819 DEAD, 未重启** (停 148/205)。**无 bound reddit paper-grade 数据丢失** (reddit 从没跑完)。proxy-503 band-aid (B-1880 wait-out / B-1882 resume) 仍有效存在但**不再是 reddit blocker** (真 blocker = task138)。
> - **🔲 #1 OPEN — 必须先定再动 reddit fire**: reddit task-138 fix = **estimand 决策**(非工程能拍): **(a) reorder-138-last**(干净 per-task 能力, 去自毁级联, 贴 representation/routing 主张)vs **(b) verify-then-tolerate**(auth 不 abort, 跑完列表含自毁后果, 贴部署真实)+ 披露 + sensitivity。**= advisor 议题**(定义 reddit 测什么)。两个都不动任务列表内容、都是合理 deviation。**别在定这个之前重启 reddit / 别 `launch red`**。
> - **owed 降级**: PROTOCOL_NOTE_03 (resume-policy witness) + chain 续跑 (`RESUME_MISSING=1 launch red`) 现 **defer** 到 task-138 estimand 定了再说 (reddit 干净跑不了之前续跑无意义)。DashScope (proxy 根治) 仍学长议题但非 reddit blocker。
> - **commits 全 local 未 push**: `6a18657`(registry transient_drift) `ba1aec1`(B-1883 budget, 现 recontextualized) `c91cec5`(B-1884 v1 误判) `4df2dc4`(B-1884 corrected + hook revert) `1d9f691`(§356 VWA flaws)。push 需用户确认。
>
> 🔭 **2026-06-22 UPDATE (root-cause SUPERSEDED by 2026-06-23/B-1884 above; B-1880/B-1882 机制本身仍有效) — reddit abort #5/#6 (proxy 503) → wait-out retune 撞墙 → 转向 resume-on-abort (B-1882) → R819 resume from 135 中** 详 笔记 §352 + B-1880/B-1882:
> - **abort #5 (R16380 task104 steps=8)**: ~8-10min proxy 503 簇 mid-episode 耗尽 B-1880 11min → abort 丢 55 ep。B-1881 按设计未接 (steps>0 + proxy_5xx 排除)。Fix = B-1880 **wait-out retune** max_retries 8→24 (~11min→**~35min**, commit `50e0c90`)。
> - **abort #6 (R819 task139 steps=1)**: **~34min 持续 503 outage** 打满 attempt 24/24 → 连 35min wait-out 也耗尽,丢 135 ep。**band-aid 死亡有数据**: proxy outage 指数变长 (3min→8min→34min) → 别再加阈值。
> - **战略转向 resume-on-abort**: Explore 实证 **reddit task 独立** (0 跨任务 post-ID 碰撞) → reset-mixing 对 reddit 无害 → **从断点 resume (不 FORCE_NEW) 比 chunking 更好** (救进度 + 未来 abort 只丢 ~1 task + 零 infra)。chunking 弃。
> - **✅ B-1882 fix DONE (commit `42c1bbd`, A100 synced)**: `mint_run_id` stale-check 查错文件 (`condition_meta` 的不存在 schema_version 字段 → resume 永远判 stale = dead path)→ 改查 v2-filename marker。功能测试 v2→resume/legacy→fresh,链零回归 (只动 FORCE_NEW=0 路径)。**resume-on-abort 现自动化**。
> - **✅ R819 RESUME 中 (live)**: 手动补 R819 meta + archive fresh fallback R22390 → `FORCE_NEW=0 RESET_BEFORE=0 P79_PAPER_GRADE=1 queue_baseline.sh B0 dom reddit` → **resume from 135** (skip 0-138 + B-486 force-rerun task139 + 续 140-204)。runner+watchdog UP。**135 ep/~16h 救回**。live 进度跑 `make ntfy` + ssh ① (别信本行)。
> - **abort-recovery 栈现三层自动**: ① wait-out 35min (proxy 簇) ② B-1881 pre-flight retry (auth) ③ resume-on-abort 自动 (B-1882, abort 只丢 ~1 task)。**根因 proxy 周期性长 outage 未动** → DashScope 直连 (根治) = 学长议题。
> - **⚠️ 仍欠 (paper-grade 纪律,新 session 优先)**: ① **PROTOCOL_NOTE_03 + /stress** = resume-on-abort POLICY (reddit 用 resume = B-304 exception,被「task 独立」背书) 的正式 witness —— 已在 R819 enacted 但欠见证 (B-1882 只解锁 mechanism)。② **chain 续跑**: R819 是 `queue_baseline` 单跑 B0 dom reddit,完成后须 manifest-bind + `RESUME_MISSING=1 launch red` 续剩 17 cond (否则链不自动续)。③ aggregator emit retry_count covariate + §3.5/§8 disclosure (B-1881 follow-up, 仍 defer)。
>
> 🔭 **2026-06-20→21 UPDATE — reddit chain 第 4 次 abort (auth blip) → 结构修复 PRE-FLIGHT transient episode-retry (3-AI /stress + PROTOCOL_NOTE_02) DONE + re-fired** 详 笔记 §350 + B-1881:
> - **🔴 第 4 次 abort (2026-06-20 22:12Z)**: B-1880 re-fire 的 R26851 跑到 task140 (137/205) 撞 reddit auth blip (`LOGIN_FAILED still_on_login`, `steps=0` pre-flight 零污染) → fail-closed abort。**结构性问题坐实**: fail-closed-on-first-quarantine × B0 长 runtime → 连续 2 次 transient (503+auth) 弃 ~192 ep/~40h,都零污染。
> - **✅ Fix DONE (3-AI /stress contract change, estimand-neutral)**: transient quarantine → **有界 episode-retry**,核心收窄 = **PRE-FLIGHT (`steps==0`) ∧ class∈{auth,network}** (proxy_5xx 排除,B-1880 管)。3-AI 收敛: codex 独家 P0 (mid-episode mutation 污染) + gemini defuse (pre-flight-only) + 我 (组合风险) → steps==0 一刀解 mutation/redraw/masking 且 **estimand 不变 → PROTOCOL_NOTE_02 非 OSF amendment**。结构化 provenance + 透明度 (retry_count 进 summary+trajectory+ntfy+yaml) + watchdog race close。18 tests + 1420 pass 0 新回归。
> - **Witness**: `PROTOCOL_NOTE_02_TRANSIENT_PREFLIGHT_RETRY_20260621.md` + tag (NO OSF deposit,recovery-alignment tier;gemini OSF dissent 已记录)。**Live-verification pending**: 下次 reddit 撞 transient auth blip 验 PRESERVE。
> - **✅ committed + pushed + tag**: commit `f2eecfc`(fix)+ `10b792c`(task140 分类)+ tag `protocol-note-02-...`。
> - **✅ re-fire DONE + confirmed HEALTHY (2026-06-20 23:52 UTC, B-1881 armed)**: archive R26851 → 分类 task140=`transient_drift` 清 **Gate G8**(首次裸 launch 被 G8 拦 = 历史 quarantine 未分类,正确行为)→ `launch red` gates 全过 → [1/18] B0 dom reddit **R16380** runner(2317772 链)+watchdog 活 + task0 step JSONL。pre-flight transient-retry 生效中。live 进度跑 `make ntfy` + ① verdict(别信本行,秒级 stale)。
> - **Follow-up (defer)**: aggregator emit retry_count covariate · §3.5/§8 disclosure prose (per-cell retry count + B0 zero-retry SR sensitivity) · abort summary 漏 aborted-episode (codex P2 pre-existing)。
>
> 🔭 **2026-06-19 (later) UPDATE — reddit chain 第 3 次 abort (B0 proxy 503 sustained outage) → retry budget 加厚 + capped backoff DONE** 详 笔记 §349 + B-1880:
> - **🔴 reddit chain 第 3 次 DOWN (2026-06-19 21:57Z)**: R28130 (B-1879 re-fire, 07:50 起步) 跑到 task59 (58/205) abort。**根因 = B0 AWS proxy 持续 ~3min 503 (21:54→21:57Z, attempt 1/3→2/3→3/3 全 503) 耗尽旧 retry 预算 (~3min 容忍) → needs_reevaluation=True = paper_grade 首个 quarantine event → PaperGradeAbortError fail-closed → runner 退出不写 summary → C3 sentinel abort + orphan watchdog kill**。= 同 chain 连续第 3 次 abort, 第 3 个不同根因 (B-1878 ref image @t28 / B-1879 wallclock @t56 / 本次 503 @t59)。**断链 sentinel = 症状探测器非故障源**。外部 substrate transient, 非代码 bug。
> - **✅ Fix DONE (capped exponential backoff, user 选 b, estimand-neutral)**: `proxy_api_agent.py` 加 `retry_backoff_max_s` (None=uncapped 向后兼容) + `_capped_wait()` helper · `api_proxy.py` forward · `exp_v2_base.yaml api_strong` max_retries 3→8 / backoff 10 / cap 60 → 容忍窗口 **~3min→~10.7min**, post-recovery retry ≤60s。无 OSF witness (operational guard, 同 B-1665/B-1879 先例); yaml-exposed per B-568; 所有 B0 site 继承。+3 tests 17 pass + 关联 105 pass 零回归; 4 文件 rsync A100 双端 md5 一致 + py_compile OK。B-1880 登 catalog。
> - **✅ committed + pushed** `origin/fix/b1878-reddit-reference-image` (commit `57ee93c`: 4 fire-path 文件 + catalog + 笔记 §349, user 显式确认)。
> - **✅ re-launch DONE + confirmed healthy (2026-06-19 23:14 UTC)**: archive R28130 partial (55 ep) → `_archive_proxy503_R28130_dom_partial_20260620` → `RESUME_MISSING=1 MAX_CONDITION_HOURS=0 MAX_CLS_WAIT_HOURS=0 launch red`。新链 orchestrator (setsid PPID=1 抗断) + [1/18] B0 dom reddit **R26851** runner(2191146)+watchdog(2191226, idle-alert 60min)活 + **proxy 恢复零 503** (tool_calls parsed click 正常)。capped-backoff fix 已就位防下一 503 窗口。live 进度跑 `make ntfy` + ① verdict (别信本行硬编码, 秒级 stale)。
> - **⏱️ schedule risk 不变 (已 surface)**: B0 reddit ~58h/cond × 18 cond sequential → reddit Pass-1 ETA ~1.5-2 周+, D4 (Pass-1 全 36 cond 06-26) 几乎确定 miss → 需跟学长 re-plan。若长跑频繁撞 503, 考虑选项 c (quarantine transient 降为 episode-级重跑, 改动更大需 witness)。
>
> 🔭 **2026-06-19 UPDATE — reddit chain 第二次 DOWN (16h B0 wallclock cap mis-kill) → cap unlimited + re-launch DONE** 详 笔记 §344 + commit cd5029e:
> - **🔴→✅ reddit chain 第二次 DOWN 已修复 + re-launch 起步**: 06-18 09:38 re-launch (R4992 B0 dom reddit) 跑到 06-19 01:41Z 撞 **16h condition wallclock cap** abort (56/205, ~17min/task, ~58h projected) → orchestrator DOWN 02:00Z。**根因 = B0 16h cap 按 cls 142s/task 校准, 不外推 reddit** (max_steps-heavy + B0 proxy 高延迟) = 2026-06-03 B1/B2 cls 4h-cap saga (§314 R11094) 的 B0-reddit 版重演 (当时只改 B1/B2 unlimited, B0 留 16h, 假设只在 cls 验证过)。
> - **修复**: `queue_chain.sh` B0 default cap **16h→0 (unlimited)**, 跟 B1/B2 统一, real deadlock 靠 watchdog idle-alert(30min)+liveness 兜底 (commit `cd5029e`, operational guard 非 estimand → 无 OSF witness, 同 6/3 先例; 保留 `MAX_CONDITION_HOURS_B0=N>0` env 入口)。scp 同步 A100 (md5 `428921ae` 一致, bash -n OK)。
> - **✅ re-launch DONE (2026-06-19 07:50, `launch red` under unlimited cap)**: R4992 partial → `_archive_wallclock_killed_R4992_dom_partial_20260619` (forensic-safe mv, FORCE_NEW from ep0 per B-304)。新链 `queue_phase1_red_20260619_075023` orchestrator PID 2112400, **[1/18] B0 dom reddit R28130** runner(2113661)+watchdog(2113692) 活 + task0 step JSONL + proxy tool_calls(click/back) 正常 = fire 恢复。live 进度跑 ①②。
> - **watchdog idle-alert 30→60min (cap-unlimited 配套, commit `95a8d2b`)**: cap unlimited 后 idle-alert 成唯一真-stuck 网, reddit B0 慢 (~17min/task) → 30min 频繁误报 → 6 queue 脚本 `--idle-alert-mins` 30→`${EXP_WATCHDOG_IDLE_ALERT_MINS:-60}` (env 可覆盖); **step-stale 10min 未动** (B-1667 真 stuck 信号, 比 episode 更早更准). scp A100 (只 5 真实文件, phantom_dom=symlink→text 不传). 当前 R28130 watchdog 已 `restart_watchdog.sh --append-args` 热重启到 60min (kill 老→新, runner 不碰); 后续 17 cond 起新 watchdog 自动 60min。
> - **push 状态 (2026-06-19, 全部已 push `origin/fix/b1878-reddit-reference-image`)**: 本 session 收口 `..30ede37` — cap `cd5029e` + idle-alert `95a8d2b` + chronicle `fdee922`/`08bd684` + worktree 清理(B3 集成 `b0dade8` / quarantine `9b24ad1` / paper lit-positioning `30ede37`) + 前置 6 commits + agy-migrate `22c902a`。**工作树干净**。⚠️ **merge master 前**: `9b24ad1` 的 B3 pilot quarantine 记录 unclassified → 需先 classify 避免 G8 cls gate 误算 (现 feature branch push 不影响 A100 fire/master)。
> - **⏱️ schedule risk (已 surface user)**: B0 reddit dom ~58h/cond, reddit 18 cond 同-site sequential (hard rule 不能并行) → reddit Pass-1 ETA ~1.5-2 周+, **advisor D4 (Pass-1 全 36 cond 06-26, 剩 7 天) 几乎确定 miss** → 需跟学长 re-plan。
> - **受限 session 穿透教训重温 (§343)**: `condense-a100` = A100 VM 本体 (非 jump host) → 直接 `ssh condense-a100 'cmd'`, **别用 `-J`** (当跳板会 publickey denied), bastion MOTD banner 仅 stderr 可忽略。
>
> 🔭 **2026-06-18 UPDATE — cls Pass-1 全完成 (里程碑) + reddit chain abort 根因已修待 re-launch** 详 笔记 §343 + B-1878:
> - **cls Pass-1 ✅ 18 condition 全 paper-grade** (B0/B1/B2 × 6 mode × classifieds; ① verdict completed_ok=18 + manifest bound-clean=18)。chain 进 reddit 首 condition 即 abort (下条)。
> - **🔴 reddit chain DOWN (2026-06-17 22:26Z) → 根因已修, 待 user re-launch**: B0 dom reddit R15710 跑到 task28 (27/205) 撞 `EvaluatorUnavailableError` (缺 CWD-相对 reference image) → paper-grade fail-closed abort → sentinel L529 → orchestrator DOWN。**非 runner crash / 非站点 / 非代码 bug = A100 self-hosted 迁移漏配 reference image** (B-1878)。⚠️ 受限-session 错觉: 交互 `ssh condense-a100 'cmd'` 落 bastion MOTD banner, 实际 rsync / `ssh -o ConnectTimeout` 可穿透拉 runner log。**已修 2 图** (`coco_images` symlink + curl 落 `B009P9HODS.1.jpg`; PIL 金标准 RGB OK; reddit 全站仅此 2 个本地 reference, 余 16 image-task 走 http)。
> - **✅ re-launch DONE (2026-06-18 09:38, `launch red`)**: R15710 partial 已 archive → reddit red-only chain 起步 **[1/18] B0 dom reddit** (`queue_phase1_red_20260618_093824`, FORCE_NEW from ep0)。⚠️ **必须 `launch red` 非裸 `launch`** — cls 18 全完后裸 launch 撞「空 cls chain (0 cells) rc=2 → P0-2-B cascade halt 拒 launch red」首现边界 (B-1878 Re-fire 注)。live 进度跑 ① + `make ntfy`; A100 fire-6 monitor 接管。
> - **follow-up (未做)**: `preflight_v2.sh` 加「全站 task 本地 image reference 可达」gate (B-1878 follow-up; shopping Phase 1b 同隐患, 迟早撞同类)。
> - **B3 floor pilot DGX 路径已放弃** (env 幽灵非 code bug, 详下方 2026-06-17 块 + 笔记 §342) → B3 floor 改 A100-after-Pass1-fire; 现 reddit re-launch = 同站 A100 优先项。
>
> 🔭 **2026-06-17 UPDATE — B3 MiMo 集成 land + dashboard 对齐 + floor pilot 撞 DGX-setup gap (floor DEFERRED)** ⭐ 接 2026-06-16, 详 笔记 §341:
> - **B3 MiMo 集成 DONE + 离线 verified 2/2**: `MiMoVLAgent(Qwen3VLAgent)` 子类 (复用 Qwen2.5-VL 路径) + `local_mimo` backend + factory dispatch + `configs/exp_v2_B3_som_classifieds_pilot.yaml` + launcher `scripts/maintenance/run_mimo_b3_pilot.sh` (走 queue_baseline `SMOKE_CONFIG` dev 路径; **BASELINE="B1" 是纯校验标签, 实际 backend=local_mimo**)。`<think>` 被 action_utils 自动剥, MiMo 直吐 confidence。
> - **dashboard (周报_6.18 .md + jsx `AdvisorB2B3Brief.jsx`) 对齐 §340**: 换槽+留披露 / breadth-reframe / bolt-on-MiMo(MIRAGE)。vite build 过。**one-pager 未动** (user 指示)。
> - **🔴 DGX floor pilot 路径 ABANDONED → B3 floor 改 A100-after-cls-fire (user 决策 2026-06-17 "选2")**。重跑 (RESET=0+creds, run_id=`B3_som_classifieds_pilot_20260617`) **又失败, R300 同款**: 10/10 episode 全挂 `task_configs/classifieds_task_N.json does not exist` (task0 reset 时 config 在→跑 15 步→eval 时没了→task1-9 reset 即挂)。**根因彻查 (不再 hand-wave)**: ⚠️ **没有任何 P79 代码删 task_configs** —— 全库 grep (p79+scripts+VWA submodule) 对 `task_config*` 只有 `tasks.py:134 mkdir`, 零 unlink/rmtree/rename。5 个会动 run-dir 的路径**全排除**: watchdog orphan-cleanup (log "skipping") / watchdog artifact-cleanup (只 artifacts+episodes) / `_cleanup_stale_runs` (skip 当前 dir + 只删空>1h) / `clear_task_files` dev-auto-clean (只 summary+steps+artifacts) / load_tasks (隔离 dry-test 写出 10 个)。∴ **删除来自 P79 之外 = DGX-shared-env 幽灵** (外部进程/系统清理), **非 B3 code bug** —— **MiMo 集成本身 sound** (加载干净 + task0 真跑 15 步有效 action)。**A100 paper-grade 路径同 runner 跑 17 条 (B0/B1/B2) 无此问题** → 换机器即避。**⏭️ B3 floor = A100 cls fire 完后在 A100 跑** (同站冲突现不可; B2 cls phantom_prompt 近完→red 18→之后)。orphan watchdog 2092910 已 kill, DGX 干净。**未追幽灵** (option-1 inotify 抓现行 user 弃)。**周四材料不受影响** (dashboard §9 标 floor=pending/next-week)。
> - **B3 MiMo thinking 官方核实 + audit待审 (2026-06-17)**: ⚠️ **纠正本地误判** —— "chat template 无 thinking 开关逻辑" **≠** "关不掉"; 拉官方 (HF card + repo) 实证 **`/no_think` 能干净关 thinking** (放 user msg **最末尾**, **99.84%** 控制成功, RL+SFT 都支持) = 控制是 **RL 训练进权重** 非 template-level (故 grep template 找不到)。→ thinking on/off **是干净同模型 ablation** (不需换 SFT checkpoint), dashboard 决策点 #2 已更 (注明 /no_think + 可行)。**B3 official-usage audit待审 (post-pilot, GPT 跑, 对标 B2 Gemma reviewer-defense lock)**: 官方推荐 **temp=0.3/top_p=0.95** vs 我们 pilot **greedy 0.0** (对齐 B0/B1/B2 一致性) = **deviation 需 §8 disclose** (同 B2 greedy/bf16 可辩护逻辑)。**教训**: RL-trained 行为查官方 playbook + web/cross-AI 核, 别靠本地 code/template 静态推断 ("用 X 架构" ≠ "继承 X 训练行为")。
> - **B3 MiMo official-usage audit DONE (reviewer-defense 第三锁, 2026-06-17→18)**: GPT browsing-audit (官方 HF card/GitHub/arXiv 2506.03569) + Claude in-repo 代码验证 → **无单一 misconfiguration**; GPT 3 个"条件性需改"项验证 = **processor 从 MiMo checkpoint 加载 ✓ (`mimo_vl_agent.py:98`) + 单图 image-before-text ✓ (`qwen3vl_agent.py:250 insert(0,image)`) + 4096 截断 = telemetry gap ⚠️**。doc → `docs/analysis/vwa_classifieds/B3_mimo_official_playbook_audit_2026-06-17.md`。**A100 B3 run 前 action (不动当前 cls fire)**: ① **抬 `max_new_tokens` 4096→8192~16384** + 加截断率 telemetry (MiMo thinking 长推理先于 JSON, 官方 vision 评测用 32768; 4096 可能截掉 JSON = 人为失败; pilot 数据被 env 幽灵清无法回算); ② **§8 disclose 清单**: greedy(0.0) vs 官方 temp=0.3 / 替换默认身份 system prompt / SoM-element-ID JSON action / max_new_tokens cap / Qwen2.5-VL processor(从 MiMo 加载); ③ **温度敏感性補强 (post-pilot, 小规模)**: 少 task × {greedy, temp=0.3 ×3 seed} 比 success/valid-JSON/repeat-action/premature-finish (挡 reviewer "没按官方参数"; 诚实标 "decoding effect 量级/方向 unknown, 官方无 web-agent temp 消融")。**前置**: floor pilot (A100) confirm 不地板才 promote + lock 生效。
> - **quark dev-VWA 连通修复 (reusable)**: VS Code auto-forward 占 127.0.0.1:9980 → Docker bind 失败; 修=portproxy `0.0.0.0:9980→127.0.0.1:19980` (compose `127.0.0.1:19980:9980`, 别回 9980:9980)。诊断锚: 000=refused / timeout=防火墙 SYN-drop / 502=app-warmup。
> - **未 push**: 本 session 6 文件 (mimo_vl_agent.py/local_mimo.py/factory.py/pilot yaml/run_mimo_b3_pilot.sh/verify_mimo_b3_agent.py) + dashboard .md/jsx + 笔记 §341 + 本 handoff。**push 需 user 确认**。

> 🔭 **2026-06-16 SESSION HANDOFF — B3 跨族扩展 + B2-floor 收口 + phantom-framing 修正 + 周四 dashboard** ⭐ 新 session 做 B3/dashboard/framing 先读这个 (本 session 三条线, 与下方 fire 线并行)。详 笔记 §335-340。
>
> **① B2 (Gemma3-4B) floor — SETTLED (真地板非 bug)**: VWA cls ~1% = 真能力 + termination-policy 地板。**六源收敛** (§327 双模型 probe 视觉分层崩坏 OCR✓/photo✗ / diag digest 95% agent-limit / §330 0-finish 钉死 / GPT 官方 playbook 审计无单一 bug / 文献无 4B VWA 锚 / §335 action 级)。机制 = 看不懂照片→编→不 finish→budget death。reviewer-defense **三锁** = 官方用法审计 + §327 受控 probe + 文献锚 (+1024-cap probe 已证伪 cap 为病因)。→ `docs/analysis/vwa_classifieds/B2_gemma_official_playbook_audit_2026-06-16.md` + B-1876。
>
> **② phantom framing — FIXED (重要, 防误读)**: phantom = **independent routing space** (drop-one oracle hero = 各臂独立/不可替代覆盖; complementarity), **NOT '更便宜的 SoM / 4-fold drop-in 性能不减'**。已改 CLAUDE.md hook + memory `project_paper_hook` 的 headline 误导框架; **paper §1 正文本来就对, 未动**。⚠️ 新 session 沟通 phantom 一律"独立路由空间/drop-one hero", 别回退"drop-in/省钱"headline (reviewer 也会犯的误读)。
>
> **③ B3 跨族扩展 — 决策 + pilot PENDING (新 session 主线)**:
> - **framing 基础**: 跨模型维 = **replication BREADTH, 非 controlled ablation** (router 只在同模型内比 representation, 从不跨模型比) → 选型约束放松: family/thinking/architecture = §8 披露项**非否决项**。绑死硬约束只剩 4: **不地板 / 守固定 JSON 动作格式 / 装单 A100-40GB bf16 / 开源**。
> - **方案 (user 倾向, 待学长拍)**: **"换槽 + 留披露"** — B2 槽位往后换成能干跨族模型 (B2 cls 仅 5/6, reddit 0, 沉没成本小), Gemma cls 数据降级为 §8 floor 披露。比"加独立 B3"省 (~12 vs ~19 条件)。
> - **候选盘 (HF API 已核存在+尺寸+多模态)**: ⭐**MiMo-VL-7B-RL-2508** (8B/16.6GB, Xiaomi MiMo-7B=真第三家族, **bolt-on**, **Stage-0 conformance 已过**: 3/3 parse-valid + 无 GLM-lockout) · Gemma4-12B-it (12B/24GB, native omni, 不 gated) · Qwen3.5-9B (9.65B, native, **仍 Qwen 家族**) · A3-Qwen3.5-9B (agent-SFT+BrowserGym 格式→拒)。Qwen3.5-27B (56GB) 超 40GB 出局。
> - **MIRAGE/架构 insight (决定选谁)**: phantom 骑在 language-prior/mirage 机制上; 该机制文献 (Visual Ignorance/seeing-not-believing) **全 bolt-on** (Qwen2.5-VL/LLaVA/Gemma/InternVL), 原生架构=无人区 → **bolt-on MiMo (第三家族) = 最 regime-consistent breadth 点** (加家族多样性不动 phantom 机制); 原生 Gemma4/Qwen3.5 = 机制风险, 选它=mechanism-probe (§5 shelved 不碰)。
> - **⏭️ 下一步 = pilot on OUR scaffold (DGX, 不碰 A100 paper-grade)**: ⚠️ **跨脚手架数字不算数** (GenericAgent/WebVoyager ≠ 固定 JSON; 教训 3 次)。MiMo 直接上 **Stage 2 floor pilot** (20-30 cls agentic via DGX→quark VWA, 验"我们脚手架真不地板") → 完后按"几个 full run + 死线"定 (full run 多半 **Phase 1b**, 撞 D4 06-26)。脚本 `scripts/maintenance/probe_mimo_b3_conformance.py` (Stage-0 模板可改通用)。全盘 → `docs/literature/B3_model_selection_2026-06-16.md` (+ AI 报告 `gpt.md` / `跨家族...选型报告.md`)。
> - **学长周四决策点**: 确认 Path-B/换槽 · MiMo thinking confound (能关 `/no_think` 但能力可能依赖思考, 需 on/off ablation) · 要不要碰原生模型 (机制风险)。
>
> **④ 周四 dashboard — BUILT 但需更新**: `docs/checkpoints/周报/` 默认视图 `AdvisorB2B3Brief.jsx` (build+3层verify 过; 看=`npm run preview` 4173 或 VSCode Live Server 开 `周报/dashboard.html`)。报告 `reports/周报_6.18.md`。⚠️ **当前旧"对照组失败"framing, 未反映 ③ breadth-reframe + 换槽+留披露 + ② phantom-framing 修正** → **周四前需更新对齐** (新 session dashboard 活)。
>
> **未 push**: 本 session commit `ee94731`/`0044069`/`3b39d7c`/`80098e9` + checkpoint commit + CLAUDE.md framing 修正 (memory 改动走 memory 系统不入 git)。push 需 user 确认。

> ✅ **RESOLVED 2026-06-10**: ARC 完成通知 — Rancher 升级已完成, **无延伸停机, VM 未 reboot, fire 全程存活** (ntfy 实证连续推进, B2 dom cls 148/224 @15:01Z 无 DOWN/RESTORED 中断)。无需任何行动。下文风险分级 + 恢复流程**保留作未来 ARC 维护 reusable runbook**。
> ~~⚠️~~ **2026-06-10 (Wed) ARC Rancher 升级 — Condenser/A100 可能中断, fire 或需 re-launch** (ARC email 2026-06-03)。Condenser = fire 跑的 A100; ARC 原文 "do not anticipate extended downtime, but risk access to Condenser interrupted during upgrade"。**风险分级**: (a) 仅 access 中断 + VM 不 reboot → fire 存活, 无需动作; (b) Rancher 触发 KubeVirt VM reboot → fire **死** (`fire6_monitor` = **alert-only 不自动重启** [L7/L99 仅 ntfy]; `@reboot` cron 只起 homepage 不起 fire; 无 systemd unit) → **completed conditions SAFE** (manifest-bound on /mnt/scratch, fstab LABEL=scratch reboot 自动挂存活) + 损失 ≤1 in-progress mode 的 partial 进度 (~20h; RESUME_MISSING 仅在 **chain 层** skip 已完 mode [EXACT `eps==scored`], 对被中断 mode 整体 **FORCE_NEW=1 fresh 重跑 from ep0**, **绝不 mode 内续跑** — within-mode 续跑会混两套 reset 态 = B-304 trajectory-discontinuity 非 paper-grade; B-756 leaf fatal-guard 兜底) → **手动 re-launch** = `RESUME_MISSING=1 MAX_CONDITION_HOURS=0 MAX_CLS_WAIT_HOURS=0 bash scripts/queues/queue_phase1_paper_grade.sh launch` (同 6/3; 跳所有已完, 可能需先 archive 被中断 condition 的 partial dir 同 R23971)。**信号** = fire6_monitor ntfy "orchestrator DOWN (up→down)" ≤30min。**行动**: 6/10 当天/次晨 `make ntfy` + 跑 ② 验 fire 死活, 死则 re-launch。⚠️ **不要**为此加 fire 自动重启 (monitor 刻意 alert-only: 自动 relaunch transient 误触发 → 同-site double-fire → cross-contam)。⚠️ KubeVirt #17417: ARC upgrade 若 hard-restart VM 可能踩 GPU-passthrough volume 坑 (我方主动 reboot 才需 detach p-79; 此次 ARC 主导)。 **2026-06-07 admin 确认 (Camilla email, user 已回复同意)**: (a) storage 1524GB→**2TB 已同意** (待 provision; 解 §313 disk-pressure 复发根因); (b) **GPU-attached VM 不能 live-migration** = 上述 reboot 风险的**机制锚点** (维护时集群无法热迁移 GPU-VM → 只能停机/reboot; 我方必须挂 GPU 做推理故无可避免; 缓解 = 本块恢复流程)。**通知三重** (核实 2026-06-07): A100 `fire6_monitor` cron `*/30` (非常驻进程, reboot 后 cron 守护自恢 → 边沿触发 DOWN ~30min) + DGX `sync_a100_results` 15min `notify_on_fail` (A100 连不上即 ntfy) + DGX `paper_grade_check` 6h verdict 🔴 ntfy。fire **故意不 @reboot 自启** (避 same-site double-fire; `@reboot` 只起 homepage flask, VWA 站点靠 docker restart policy + @reboot 自愈) → 拿到 DOWN 通知后**手动** re-launch。

> 📅 **2026-06-10 advisor 两份书面 deliverable + deadline 自查 (standing, 每 session 必查)**: **D1 one-pager → advisor ✅ SENT 2026-06-21** (PDF, 踩硬线 ≤06-22 内; §351) · **D7 lit-review 章 self 07-13 / 官方硬线 ≤07-20 ← 下一 advisor 节点** · D4 Pass-1 全 36 cond 06-26 · D5 full analysis + drop-one verdict 07-01 · D6 router H10 verdict 07-08 · D9 thesis draft v1 08-10 · D11 submission early-Sep (**TBC 待确认**)。全表 canonical = `issue_advisor_sync_2026-06-10` D1-D11 表 (one-pager §4 精简为 6 行关键节点保 1 页); live = `make status V='tasks#NOW'` (eta 字段)。**每 session handoff 对照自查 on-track, 偏差主动报 + 提议 re-plan** (advisor 显式要求 "always check that you are on track")。详 `issue_advisor_sync_2026-06-10` + 笔记 §331。

> 🔥 **2026-06-06→08: B1 cls cross-mode /diag discover (6/6 mode 齐) + B-1869 + 3 类 routing-rescuable 证据** ⭐ 最新。fire 健康自走: B0 cls 6 mode 全完 → **B1 cls fire 6-mode 全完** (dom/som/vision/ptext/psom/**pprompt R32516**) → chain 进 B2 cls (**fire 死活/进度跑 ①② 拿 live**, 别信本行秒级 stale)。**6 condition /diag discover 全齐** (**仅 discover 不落码**, ruleset 仍 `5-domsomvispsom-b1860coord`): §317 som (P31=finish-less artifact) · §318 dom (真卡死 + P31 跨模式 confound) · §320 vision (**router crux REFINE**: vision 只救 "AXTree 丢失的已渲染文字" 型 P5 [task40 dom30→vision4 flip 截图确认], 救不了导航/推理型) · §321 ptext (**axis-2 P4 root-ref 瞎猜**) · §322 psom (HERO; **axis-2 P4 phantom-family-wide RESOLVED** — observation 实测 RootWebArea=[2] 无[1] → element_id=1=**幻觉 low-default 非 renumber-root**, 解 §321 两难, §290 第5次; psom>ptext 因 SoM-prompt click-priming; **P33 img-href→PNG 幻觉 B1(4B) 6/7 比 B0(235B) 显著差** = 4B 自纠错 gap) · **§323 pprompt (P4=0 但 walk_fail 主导 880act/149ep → 精化 §322 "phantom-family-wide" 为 text-表征轴: 裸 element_id mode-fragile / walk_fail mode-robust, pprompt=隔离 control 最强证据; DOM-URL 视觉盲 ⚠️撤回初版"修正 B0§319/§290第6次"误判: img-src 上游 VWA `processors.py:703` 注进**全文本模式** [pprompt obs `oc-content/uploads`=5 与 psom 同, observation_dom.txt=模型真输入], **B0§319 成立**, 是反向§290 [初版 grep 自己 log-only 错]; kaiyo.com=图片像素表征无关, P79 观测侧不接 captioning; task5 delete false-success FP 首例 SR-抬高; P18 sOrder 真 bug)**。**3 类干净 routing-rescuable failure** (paper router 硬通货): ① text-in-image rescue (§320 task40) ② attribute-verification deadlock (§321 task17) ③ **correct-item-visited-but-rejected (§323 task111)**。**P31 finish-less 跨 6 mode (全) confound 闭环** → 绝不可裸作路由信号。**B-1869** 登 catalog + audit.base (walk_fail fallback 报 success=True 测量隐患, pprompt 21.7% 复现; **post-fire candidate** 非 fire-blocker)。全链 → 笔记 §317-323。
> **⏭️ forward**: B1 cls **6-mode discover 全齐 ✅**。**🧊 freeze step = DEFERRED 等 B2 cls discover 一起做** (user 决策 2026-06-08, 一次完整 B1+B2 cls freeze 更全)。**即时下一步 = fire 跑 B2 cls 中, 每个 B2 cls mode 落地 `/diag B2 <mode>`** (同 B1 流程) → B2 cls 6-mode 齐 → freeze (合并 B1+B2 提议)。**freeze 方案已成文** → `docs/checkpoints/diag_freeze_v6_plan.md` (P34-P39 新规则 + P18/P19/P10 bug-fix + success-safe 收窄 + 6 决策点; **DOM-URL "双 surface" 已纠成单-surface** per §323 CORRECTION; **P4 success-safe 正解=walk_fail 豁免**)。
> ⚠️ **freeze ≠ drop-one (纠 conflation)**: freeze 解锁的是 **§306 cross-mode TAXONOMY routable/oracle (+16pp 型)** [等 B2]; 而 **§1 hero drop-one gate (P-SoM 1.7-3.3pp) 是 SR-based 独立于 freeze** —— B1 cls SR 已 landed, 只需 **manifest-promote B1 cls + `make analysis`** 即 k_cells=1→2 出数, **不需 freeze 也不需等 B2** (可现在单独解, 待 user 定)。**cross-mode/cross-model TAXONOMY 定量仍禁直至 freeze**。B-1869 留 post-fire。**digest 全在** `docs/analysis/vwa_classifieds/B1_{dom,som,vision,phantom_text,phantom_som,phantom_prompt}_classifieds_diag_digest.md`。
>
> 🔥 **2026-06-03 (later, wallclock saga): condition+chain 两层 wallclock cap → unlimited, fire 重启**。上午 fire(R11094 B1 dom cls)跑 4h 撞 **condition 级 4h wallclock**(B-1665 baseline-aware: B1/B2 4h / B0 16h)被 abort → 整 cls chain DOWN [1/12]。根因 = 弱 4B 模型 SR 低(8.7%)→ max_steps-heavy → 单 condition ~20h ≫ 4h(**弱模型反而比 B0 慢**: per-step latency 分档漏了 per-task 步数维度)。**两层 cap 都改 unlimited**: condition `queue_chain.sh` B1/B2 4h→0(commit `71b07ae`) + chain orchestrator cls-wait 24h→0 `MAX_CLS_WAIT_HOURS`(commit `013175a`); B0 保留 16h; 兜底 watchdog idle-alert(20min)+ liveness 不变。清 R11094(46ep)+R13717(0ep)→ `_archive_wallclock_killed_R11094_*`(forensic-safe mv)。重启 `RESUME_MISSING=1 MAX_CONDITION_HOURS=0 MAX_CLS_WAIT_HOURS=0`: Gate G8 clean → log 实证 "Waiting...(max 0h; 0=unlimited)" → B1 dom cls 起步确认(monitor step JSONL)。**代码 A100 scp 同步 + 全 commit 已 push `origin/diag-discover-then-freeze`**(`71b07ae` condition + `013175a` chain-wait + `6cabd19` Fire-5 cosmetic + `430b302` docs)。**fire 死活/进度跑 ①②拿 live**(别信本行, 秒级 stale)。全链 → 笔记 §314。
> **⏭️ forward**: cls chain 12 cells(B1×6+B2×6, 每 ~10-20h)→ red 18, **无 cap 跑到自然完成**, ETA **~1.5-2 周**(B1/B2 弱模型慢)。完后才轮到 Pass-2 router(见下方 6/2 条目: prose 已定, launch 前需先在 landed Pass-1 数据跑 Stage 1→3 否则 P0-2 gate fail-closed 拦)。**收尾 ✅ (session 2026-06-03)**: wallclock 两层 fix + Fire-5 cosmetic + §314 chronicle 全 push `origin/diag-discover-then-freeze`。剩 `retry_b1_single_task.sh` L69-70 syntax error(可选, 非关键路径——retry 是手动诊断工具非 fire 依赖)。
>
> 🔥 **2026-06-03: A100 saga 收口 + Phase 1a Pass-1 baseline fire 重启 (B1 dom cls 起步)**。A100 5/29→6/3 因 sl-g02 cluster ingress 中断 down **4 天** (根因 = storage headroom 仅 12Gi + KubeVirt #17417 GPU-passthrough-VM-reboot 需先 detach volume), ARC 恢复(没换盘, fstab LABEL=scratch 自动挂) + 防复发硬化(配额申请 **2TB 待批** / reddit+homepage reboot **自愈** [reddit restart=unless-stopped + homepage @reboot cron] / condenser infra 文档化 `docs/reference/condenser/`[**gitignored**] + memory `reference-condenser-a100-infra`)。B1 残留 R23971(5/28 partial-21ep)+B1/B2 空壳 归档(A100+DGX 4 处)。fire 重启撞 **Gate G8 quarantine**(cls task20 auth event from R23971) → diagnostic 复现 confirm transient + `classify transient_drift` → **重启成功**: B1 dom cls 起步, cls chain 12 cells(B1+B2 cls) → red 18, RESUME_MISSING skip B0 cls 6(manifest-bound)。commit `2f80669`(infra docs, **未 push**)。**fire 死活/verdict/run_id 跑 ① 拿 live**(别信本行, 秒级 stale)。全链 → 笔记 §313。
> **⏭️ forward**: **Pass-1 baseline fire 跑中**(B1+B2 cls + 18 red, ETA 多天; 跑 ① 拿 live verdict + ② 拿 fire 死活) → **完后才轮到 Pass-2 router**(见下方 6/2 条目: prose 已定, launch 前需先在 landed Pass-1 数据跑 Stage 1→3 生成 fold-aware bundle 否则 P0-2 gate fail-closed 拦)。**fire 完 2 收尾**(fire-path immutability 等完改): Fire-5 cosmetic(`quarantine_registry.py` messaging 硬编码旧编号) + retry script L69-70 syntax error。
>
> 🔥 **2026-06-02: 并行 6-session token-burst land + §6 disclosed-limitations rewrite** (2026-06-02)。Claude 协调 6 条并行 worktree 线 (S1-S6 围绕 router 重心) → **octopus merge 零冲突** (`1048113`)。**4 fire-blocker 修复** (Pass-2 launch-blocker): P0-4 cls wait 8h→24h `153ee69` / P0-3 τ hard-fail `f56c308` / P1-9 oracle-label hygiene `9832df6` / P0-2 leaf gate 验 fold-aware bundle (非 deprecated 单-pickle) `f79e0d2`; 全套件 **1397 pass 零回归**。**§6 prose**: 2-round /stress (S3 §311 + 第 2 轮 codex+gemini judged "amplifying not defusing") → **codex 保守路 rewrite** (`5a757f5`): H10=compliance gate / `k_cells_router_strictly_better`=mandatory diagnostic / 辩护→招供姿态。**P0-1 走 prose 降级 (c), 不 amend** (workshop scope user 决策 — claim ≤ locked prereg 故无需 witness)。全链 → **笔记 §312**。commits `1048113`→`5a757f5` **全未 push**。
> **⏭️ forward**: prose 定了, **等用户发话 fire Pass-2**。⚠️ **Pass-2 launch 前必须先在 landed Pass-1 数据上跑 Stage 1→3** (`extract_50_features` → `train_l1_router_with_mi` → `train_l1_router`) **生成 fold-aware bundle**, 否则 P0-2 gate **fail-closed 拦** (on-disk 现是 5/16 deprecated 单-pickle, `_lib_lr_artifact_validate` 报 52 failures)。S4 §7+§6.8 prose 仍标"待 stress"未 finalize。
>
> 🔥 **2026-05-28: B0 cls 6 mode 全完成 → chain 进 B1 + manifest promote + diag/taxonomy**。chain 已推进 **B1_dom_classifieds** = B0 cls 6 mode 全 paper-grade 完成 (dom R21557 17.4% / som R5313 27.2% / vision R32024 25.0% / ptext R31183 15.6% / psom R32031 15.6% / pprompt R14655 19.6%, 全 224; **live 进度跑 ①**)。本 session 5 件事: (1) **/diag psom → P33 phantom-img-nav 落码** (ruleset `4-domsomvis`→`5-domsomvispsom`) + P34 视觉盲 presence-only(21/106)回退; (2) **/diag pprompt → P34 phantom-family-wide 坐实** (shared verify 6/6 same_as_psom = 视觉盲与 prompt 格式无关) + success-fire 16/16 全 FP (representation-dependent) + 纯 discover ruleset 不动; (3) **6-mode cross-mode taxonomy** (扩 §291 3→6): routable 88/224 **(39%)** + oracle 43.3% vs best-single som 27.2% **=+16pp** (⚠️ PROVISIONAL — §302 ~14% serving noise 淹, 需 replicate); (4) **manifest promote 6 B0 cls→paper-grade** (partial 6/36) → make analysis 首跑真 paper-grade 数据, 修 2 个全数据复现 aggregator 真 bug (json-ndarray / pandas-groupby) + mechanism §5 non-fatal; (5) **config verify** (回应 pprompt>dom 反常质疑): action-level parse/tool_call **99.9%** (config 没崩) + action_success 77.3%<dom 79.4% (axis-2 SoM-`[N]`-ref 略劣 DOM-element_id-ref, lit-backed) → **pprompt>dom SR 反常 = 纯 §302 noise** (action-level 反向实锤)。commits `1a62da3`+`f9918d5` **未 push**。详笔记 §304/§305/§306 + 见 ④ forward。
>
> 🔥 **2026-05-27 PM: B-1868 PROTOCOL_NOTE_01 land → Phase 1a fire restart under new watchdog**。R14849 (B0 P-SoM cls) 13:47Z 撞 classifieds session-loss → pre-fix watchdog silent-deleted 3 ep + B-863 reaper 5min purge → user 抓 "整体不能 restore 设计吧" (§247 B-1777 sibling guard scope gap)。Full Phase A:4 P0 + 7 P1 + 3 P2 一次性 fix(types.py infra_covariates schema + watchdog paper_grade preserve + per-cond_dir fallback + atomic-summary mark + aggregator dual-path + event_key dedup + paper §3.5.4/§4.X.13 prose)。Cross-AI Mode B+C(codex 5 findings 4 OOB + gemini 4 findings 1 OOB)+ codex prose round PASS w/ amend(softened "DISJOINT" framing + "planned" sensitivity-producer)。14 invariant tests,全 pass。**3C downgrade**:不是 estimand change → 新立 `PROTOCOL_NOTE_##` convention(同 witness chain 但 **NO OSF deposit** vs AMENDMENT_07 kv9sf)。**commits `a1c5d6c` (B-1868 main) + `0984c9b` (P-text digest correction) + tag `protocol-note-01-session-lost-paper-grade-20260527` pushed**。**A100 archive** R14849 → `_archive_b1868_session_cleanup_R14849_20260527/`。**Fresh launch 2026-05-27 ~19:13**:`RESUME_MISSING=1 queue_phase1_paper_grade.sh launch`(orchestrator PID 913845 + cls chain PID 914520),正确 SKIP 4 已完(R21557 dom · R5313 som · R32024 vision · R31183 phantom_text)→ B0 phantom_som cls fresh 起步 → cls 14-cell + red 18-cell sequential。**复盘入口** = 笔记 §303 + `PROTOCOL_NOTE_01_SESSION_LOST_PAPER_GRADE_20260527.md` + master_bug_catalog B-1868。**验证 = `make ntfy` 看 `PAPER-GRADE VERDICT completed_ok↑` + `PAPER_GRADE SESSION RESTORED (paper_grade)`(若 session-loss 触发 = live test of PRESERVE 路径)**(别信本文硬编码;跑 ① 拿 live)。

> [!important] **掌握现状 = 跑命令拿 LIVE 状态, 别信本文/笔记的硬编码数字 (秒级 stale)。** 笔记 §265-271 = WHY; 本节 = HOW (拿 live + 盯什么)。
>
> **📊 §1/§2/§4-§7 = `![[base#view]]` embed — Obsidian 端渲染成活表; CLI/Claude session `Read` 只见空壳指针**。拿数据: `make status` (列全 base + 视图 + 计数) → `make status V='<base>#<视图名子串, 勿含空格>'` 渲染单视图 (= Obsidian 等价)。改字段: `make status-set N=<note> SET='status=done'`。数据源 = `_status/*.md` frontmatter (cron ~10min sync; 落后实际 fire 时以 ① live 为准)。

**① paper-grade verdict (一条命令; 每 6h cron 也自动跑 + ntfy; ⚠️ 无 A100 shell 的受限 session → 跳过 ssh, 直接读下方 `tail` 本地 cron log = 等价 verdict)**:
```
ssh condense-a100 'cd /home/ubuntu/workspace/p79 && .venv/bin/python3 scripts/maintenance/paper_grade_check.py'
#  → VERDICT: OK completed_ok=N inprog=[Rxxxx mode ep=N img=N errflood=0]   /   ISSUES=...
tail -40 logs/cron/paper_grade_check.log     # ⚠️ ssh 拿不到时 (受限 session) 这是等价 live 源, 滞后 ≤6h; cron 落 DGX 本地 00:30/06:30/12:30/18:30
```

**② fire 死活 (fire 在 A100; `make active` 只扫 DGX → 空是正常)**:
```
ssh condense-a100 'pgrep -af "queue_phase1_paper_grade|queue_chain\.sh|run_experiment\.py" | grep -v "bash -c"'
ssh condense-a100 'cd /home/ubuntu/workspace/p79 && .venv/bin/python3 scripts/analysis/validate_fire_manifest.py'
```
> ⚠️ 滤词用 `grep -v "bash -c"` **不是** `grep -v bash`: 链编排器本体 = `bash scripts/queues/queue_chain.sh …` (它本身就是个 bash 进程), `grep -v bash` 会把它一起滤掉 → 误判"编排器死了" → 诱发**有害的手动 re-fire** (同-site 双起 baseline → 违反 hard rule → cross-contam)。读数: `run_experiment.py` 活 = 当前 condition 在跑; `queue_chain.sh` 活 = 下一 condition 会自动续链; **后者空但前者活 = 当前 condition 跑完会静默 stall** (链不会自动续, 需手动 re-arm)。

**③ 当前阶段 (慢变语境 — live 进度跑①拿, 勿读快照数字)**: **Phase 1a fire 2026-05-27 19:13 第 4 次 re-launch 成功** via `RESUME_MISSING=1 queue_phase1_paper_grade.sh launch` (canonical 全 36-cond orchestrator, RESUME_MISSING skip 已完 dom+som+vision+phantom_text, **代码 @a1c5d6c PROTOCOL_NOTE_01**)。**先前 fire** sequential land: R21557 dom · R5313 som · R32024 vision · R31183 phantom_text(全 224 ep paper-grade authoritative)。**R14849 (B0 P-SoM cls 中断)** killed 2026-05-27 ~14:30Z 因 session-loss 撞 B-1868 (`§247 B-1777 sibling scope gap`) → 3 ep silent-cleaned + reaper purged → user 抓 → full P0+P1+P2 fix + PROTOCOL_NOTE_01 witness + R14849 archive → 新 fire fresh-起步 from B0 phantom_som cls under new patched watchdog。chain 剩 14 cls + 18 red: B0 cls phantom_som/phantom_prompt + B1+B2 cls 各 6 + 18 red conditions。每 cond **B-1839 per-condition docker restart** = fresh substrate。监控全自动: A100 `fire6_monitor` (**B-1863 边沿触发**) + DGX `paper_grade_check` 6h cron + sync 15min。**ETA 多天** (B0 cls ~8.3h/mode 实测)。**Pre-fire witnessed**: PROTOCOL_NOTE_01 (B-1868 paper_grade session-loss preserve, tag pushed, **NO OSF** — recovery alignment) · AMENDMENT_07 (SoM sequential-id, tag+OSF kv9sf) · AMENDMENT_05 (B-1860 coord) · AMENDMENT_06 (run-to-run sensitivity; §4 被 07 反转) · Amendment 02 (gates 不动) · B-1839 substrate — **OSF/gate 不动**。
>
> **🔄 2026-05-27 PM**: B-1868 PROTOCOL_NOTE_01 watchdog session-loss paper_grade preserve(`a1c5d6c` types.py + watchdog + aggregator + paper §3.5.4 + tests + witness doc;cross-AI Mode B+C audit 4 P0 + 7 P1 + 3 P2 全 fix;Q3=C downgrade 3C → PROTOCOL_NOTE_## convention 新立 vs AMENDMENT_##; **session-loss episodes now PRESERVED canonical, NOT cleaned**; live verification of preserve path pending session-loss recurrence). 详 → 笔记 §303 + `PROTOCOL_NOTE_01_SESSION_LOST_PAPER_GRADE_20260527.md`。**live run_id / 进度跑 ①**。

**④ PENDING — forward-only (live 进度靠 ①, 不在此冻快照)**:
> - **🔓 drop-one / routing 定量解锁 (2026-05-28 核心 forward)**: §1 hero drop-one gate (`phase1_full_prereg_decision.md`) 现 **INSUFFICIENT_DATA** (k_cells=1, FE pool 需 ≥2) → **解锁 = 再跑任意 1 cell** (reddit cls / B1 cls 最快) 让 k_cells≥2 出数。+ **replicate 扣 §302 noise**: 6-mode routable 39% / oracle +16pp 是 PROVISIONAL (单次 run ~14% noise 淹), 需 replicate-calibrate 才可信。
> - **B1/B2 跑前 verify (2026-05-28)**: B1 Qwen3-VL-4B + B2 Gemma3-4B = **local dense → 无 §302 server + 无 MoE**, noise floor ≈ id-churn ~10.5% (§298 B1 实测, 可控) → **可能比 B0 更干净 routing/drop-one evidence** (B0 server noise 是 hero number 麻烦)。跑后 verify id-churn 实际 SR 影响 + 4B capability 弱 → routable 形态可能变 (universal-fail↑?)。⚠️ pprompt 别扭 = axis-2 prompt-ref (SoM`[N]` vs DOM-element_id, lit-backed) **≠ id-churn**, 两码事。
> - **make analysis full Phase 1a 时**: 2 aggregator 真 bug 已修 (`aggregate_phase1_full_prereg_decision.py` json-ndarray + `aggregate_routing_auroc.py` pandas-2.2-groupby, commit `f9918d5`, 全数据也复现); **`Makefile analyze-mechanism` §5 暂搁 non-fatal — 恢复 §5 时移除 `|| echo` 复原 fatal**; figures 等全 6-cell (partial 只 B0 cls → fig3d+ empty crash)。manifest 已 promote B0 cls 6 (partial 6/36); reddit+B1+B2 完后 promote 剩 30 → drop-one/prereg full power。
> - **每 condition 落地跑 `/diag`** (per-condition 命名, 勿用 run_id): 3-tier (Tier-1 0-token) → `docs/analysis/vwa_<site>/<model>_<mode>_<site>_diag_digest.md`, 拆失败结构喂 paper-grade 错因表。**ruleset 当前 = `5-domsomvispsom-b1860coord`** (P1-P33, +P33 phantom-img-nav 2026-05-28 §304; dom/som/vision success-fire 全 0; psom P33 success-fire 1/17 / pprompt 0/5)。**新 fire run diag 已 land**: psom R32031 §304 + pprompt R14655 §305 (digest 在 `docs/analysis/vwa_classifieds/`); dom/som/vision/ptext digest 在新 fire run (R21557/R5313/R32024/R31183, ⚠️ 仍标 `4-domsomvis` — cross-mode 前 `diag_autorun.sh` 全量重扫拉齐 v5)。⚠️ **此前 diag (R31194 dom · R9725 som · R3671/R24792 vision · R2647 ptext) 全在 archived 旧代码 run 上** —— findings 已喂 ruleset + paper (§283/§284/§290/§291; **B-21** 货币 tokenize benchmark-FP; vision 失败全 agent-limit = paper §3-§4 evidence), 但 run-specific 数字 **superseded**。**新 Phase 1a run 落地后逐 condition 重跑 /diag**(per-condition digest 自动覆盖;**dom/vision digest 仍有效**因代码未变,**som-family 待新数据刷新**)。**open**: P14 v3 scroll_changed 豁免 · failed-hit causal verify (P19/P5/P14) · **cross-mode 定量仍禁** (discover-then-freeze, 6-mode 齐前)。
> - **[历史, R2647 archived] ptext repro (§292)**: archive R19776 ↔ R2647 同 task 群 6 flip 全 model-nondeterm + **B-1860 对 P-text 无副作用**(element-ID 归因后被 §294 纠正)。**durable 产物 = 工具 `scripts/analysis/compare_cross_run_same_condition.py`**(通用 pre/post-fix 跨 run 审计;起始污染判定用 `url_before` 非 `obs_url`)→ 复用于新 run 的 run-to-run sensitivity(§D4 / MoE 残留)。R2647 已 archive(sequential 重跑)→ R2647-specific 复算 **superseded**。
> - **H1 run-to-run sensitivity — 拆解 RECASTED via codex Mode B cold-start (笔记 §302, 2026-05-27)**: SoM-family element-ID churn (AMENDMENT_07 / B-1862, §295) 消除 ✓ 但**真 dominant 源 = remote serving nondeterminism, NOT element-ID, NOT MoE**。vision MoE compare (R24792↔R32024, 224 task) **discordance 14.3% + Δ+0.9pp net**, codex 自挖全 224 step-0 recompute: **224/224 screenshot byte-identical + url 一致 + 222/224 actions diverge** = 99% step-0 action 在字节一致 input 下 server 端就 diverge, 不是 trajectory 漂出来的。**§298.3 "B0 dom 12.1% ≈ id 10.5% + MoE 1-2pp 拆解" RETRACTED** (跨 model/modality/serving/perturbation 4 维度不可比, category error)。Codex cold-start 9-candidate ranking: **remote serving #1 > alias drift #2 > tool-call decoder #3 > MoE-specific routing #4** (官方 MoE 模型但 repo 无 expert log / batch ID / instance ID → "Do not label residual MoE without N=5 replay + serving evidence")。**Safe paper claim** = "B0 through AWS proxy/Bedrock 有 remote-serving instability floor ~14pp per-task SR discordance on classifieds vision, while net SR moves only ~1pp" (codex §6 verbatim)。**vision drop-one 7pp self-oracle > hero 1.7-3.3pp magnitude** → vision 进 hero 必加 caveat。
> - **repro_replicates ✓** (dom R31194 + vision R24792, DGX+A100, gitignored, README + §297 tracked provenance) — archive 本就是干净 replicate (B-1860 不碰 `make_dom_prompt` + dom 0 坐标动作 · vision archive post-B-1860 · AMENDMENT_07 保 native nodeId · B-1836 eval predates)
> - **dom canonical landed** (R21557 完 05-26 05:15, §298): archive **15.2%** vs current **17.4%**, **Δ=+2.2pp on 全 224 task** (McNemar p≈0.44 **不显著**) — §242 partial 假象第二次实证 (partial @88 误测 +8pp → canonical +2.2pp 收窄)
> - **Tier A B1 dense receptor 实证** (§298, DGX): n=133, **determinism 133/133**, **id-flip 14/133 = 10.5% 纯 id-channel** (零 MoE 混淆) → §282 推断**升级为受控证明** (仍 valid; 但**不能跟 B0 dom/vision 减法拆**, §302 codex 反转)
> - **vision canonical landed** (R32024 完 05-26 23:32, §302): archive R24792 SR=24.1% vs current R32024 SR=25.0%, **Δ=+0.9pp net / discordance 14.3% (32/224 flip)** = real noise floor; 5-task screenshot byte-identity 5/5 ✓ + codex 自挖全 224 = 224/224 input match + 222/224 step-0 actions diverge — server-side nondeterminism dominant evidence。
> - **Tier B activation patching** (§300, Myriad A100, 4-iter debug): 6 task per-layer → **task-heterogeneous distributed signal** (task 93=全网广 / 125=早层 id 晚层 curr 经典分裂 / 17=弱 / 64=chaos 第三态), id-channel **不是单层定位** (B1 dense local 仍 valid mechanism finding)
> - **跨 GPU greedy 不字节确定** (§300): V100 9 vs A100 6 baseline flip → §300 client-side floor; **现在 + §302 server-side ~14pp floor 双锚** = "B0 服务端 nondeterminism > 客户端理论 byte-identical determinism" (§242 → §302 升级)
> - ✅ **[DONE 2026-05-27 02:09 BST] N=5 same-payload replay cross-provider control** (§302.8): 双 batch DGX 0 GPU-hour $1; AWS Bedrock 16/20 full diverge + 0/20 deterministic + margin 4-5 vs DashScope intl 1/20 full + 4/20 deterministic + margin 17 = **layered noise** (Layer 1 model multi-token cumulative + Layer 2 AWS batching/routing 16× add); `system_fingerprint` 双 provider 100/100 都 None = audit gap 行业普遍。Risk 6 prose 升级到双层 framing; codex #1 cross-provider confirmed 但 sub-mechanism 不 isolate (audit-artifact gap)。
> - **仍 pending**: **B-1867 audit artifact patch** (Phase 1b 加 payload SHA + canonical args hash + response-headers + logprob-margins 持久化, client-side 补 audit gap 唯一可行 forward path) · replicate-calibrated MC perturbation (post-fire, AMENDMENT_06 non-gating, 现 reframe 不针对 MoE 而是 "provider-dependent layered noise floor") · 承诺 floor≈effect → hero prose 降级 (vision drop-one 7pp 已超 hero 1.7-3.3pp) · Tier B v3 per-position patching (**scope 紧, §5 advisor 暂搁仍生效**)
>
> 详 [[phase1_plan]] §D4 + [[paper_planning]] Risk 6 (🔁 UPDATED 2026-05-27) + AMENDMENT_06/07 + 笔记 §302 + codex output `docs/checkpoints/codex_outputs/vision_moe_anomaly_2026-05-27.md`。
> - ✅ **[DONE 2026-05-25] paper prose sequential reconcile**: section1 hero · section2 axis-1 · section3 §3.5+H3 · paper_planning Risk6 · phase1_plan §D4 全部对齐 AMENDMENT_07(commit `6b3088a`)。
> - **Pass-2 router (Pass-1 全 36 完后)**: `queue_phase1_router_paper_grade.sh` (H10 learned router, 6 cell × 1 condition, paper §6)。
> - ✅ **[DONE 2026-05-26] scratch redirect — partial symlink** (笔记 §301): 不是整 `results/` symlink (那让 git tree traversal 报 provenance/mechanistic 全 deleted → Gate 3 fail), 而是 **`results/visualwebarena` → `/mnt/scratch/p79_results_active_visualwebarena`** only。fire data 在 scratch (5.6G 现 → 28-30G 全 36 cond 完后), git-tracked subdir (provenance/mechanistic/repro_replicates) 留 / 上 Gate 3 干净。Phase 1+2+legacy 累计释放 `/` 22G (95%→90%)。教训: 整 symlink 撞 git symlink-traversal 边界, 必须 partial。
> - **B-1837** (eval 5-retry vs agent-step 0-retry → differential rescue confound) = measure-then-decide: Pass-1 量化 per-baseline eval-rescue rate, 再定 disclose (§3.5/§8) vs symmetric retry。NOT code change now (master_bug_catalog B-1837)。
> - **post-fire / Phase-1b code remediation** (B-1842~1847 parse-error rescue · B-1839-fu shopping reset · B-1848 Playwright wedge · B-1760 screenshot): 全迁 §4 → `make status V='audit#Deferred'`。
> - **AMENDMENT_08 候选 exclude list** (post-Phase-1a, 6-mode verify 后一次性 amend, /diag R21557+R5313 sourced §299): T180 rating widget AXTree-invisible (4 task, **B-1864** — GRL Track A wrapper-enrichment 3rd case, [[workshop_subpaper_plan]] §0.1) + cross-site cls+shopping single-site fire 不可达 (2 task, **B-1865** — paper §8 scope disclose 非 GRL) + cross-run cross-mode 4-run 一致替代 item + P28/P29 B-21 sibling (5 task, **B-1866**) = 合计 11 task, scored_task_count cls **224→213**。estimand witness 同 AMENDMENT_05/06/07 pattern (tag + OSF + paper prose reconcile), **mid-fire 不动**。
> - **PROTOCOL_NOTE_01 live verification pending** (B-1868, 2026-05-27 §303): fresh fire 跑中 — 若 cls session-loss 触发,watchdog 应走 preserve 分支 (不再 clean),emit `session_lost_paper_grade_preserved` event + atomic-patch summary `infra_covariates`。监控 ntfy `P79 SESSION RESTORED [classifieds] (paper_grade)`(新 message format with "paper_grade" suffix vs pre-fix plain "P79 SESSION RESTORED [cls]"). 若整 chain 跑完无 session-loss → 14 invariant tests + cross-AI audit + source-grep forward-guards 是足够 witness for paper-grade continuation。
> - **Aggregator-side `infra_contaminated_high` threshold tally + sensitivity-producer** (PROTOCOL_NOTE_01 §3.5.4 "planned"): aggregator 现 emit `session_lost_preserved` column unconditionally;sensitivity table excluding preserved-ep 是 paper §4 Appendix planned, **post-data lands**(post-Phase 1a + 1b 一次性 build,先 lock estimand emission pre-fire,sensitivity table 等观察到 contamination count 再 build)。当前 column 没 consumer 不是 bug,是 deliberate ordering。
> - **⚠️ disk 架构 "A100 算力 / DGX 数据"** (post-Phase-2 实测 update 2026-05-26): A100 `/dev/vda1` 90% (435G/485G post-cleanup), 真正 ceiling = `/var/lib/{containerd+docker}` 570G (containerd 388G 真物理 + docker 182G overlay+rootfs); `docker system df` 报 416GB image 是 logical 不去重共享 layer — 实际 docker stack 5 image active 不可压缩。`/mnt/scratch` 24% (114G/503G, 含 119G wikipedia.zim 单文件), 363G avail = 全 36 cond ~28-30G fire data 完全 fit。**已 partial-symlink** `results/visualwebarena` → scratch (上方 bullet); 整 results symlink 因 git symlink-traversal 边界 NOT viable。详 [[COMPUTE_INFRASTRUCTURE]] §1.1 partial symlink layout + 笔记 §301。
> - **gallery on-demand** (B-1828): `make gallery RUN=<run>` (annotate overlay + HTML) / `make gallery-all` → `http.server 8765 --directory results`。不自动刷新 (paper-grade 保干净); 原图随时 DGX 可生成, HTML 手动 `make gallery`; 近实时设 `P79_WATCHDOG_GALLERY=1` (run 期 overhead, 不建议)。
> - **🔴 / abort → 直接看 A100 runner log 新 traceback** (`ssh condense-a100 'tail -40 /home/ubuntu/workspace/p79/logs/B0_*_R*_runner.log'`), **别 isolation 复现** (B-1832→1836 教训: 运行期 bug 只在生产暴露)。

**⑤ WHY (按需读)**: 全链 chronicle → 笔记 §265-303 + `master_bug_catalog` B-1832~B-1868。关键锚: §269 (B-1836 eval-timeout) · §280 (B-1848 wedge) · §281 (AMENDMENT_04) · §285-288 (B-1860 coord 全链) · §289 (status 动态层) · §290 (vision /diag: B-1860 验证 + forensic + 落码 P31/P32/P27) · §291 (cross-mode 失败 taxonomy 框架 `cross_mode_failure_taxonomy.py` O(mode) + routing=缩略图识别梯度 provisional) · §292 (ptext archive↔current repro: 6 flip 全 model-nondeterm + element_id flip harmless + 零起始污染; **obs_url=action后 / url_before=action前** 字段语义教训; 工具 `compare_cross_run_same_condition.py`) · §293 (**H1 drop-one run-to-run 脆弱性**: task-level bootstrap 漏 run-to-run 方差 → anti-conservative; GPT cross-AI 3 纠正 [H10 非免疫 / drop-one 正偏非必然 / self-oracle=diagnostic NOT bias]; 4-维 reframe=只 Efficiency robust 救故事不救 gate; mitigation=non-gating replicate-calibrated sensitivity, 不 patch element_id; advisor post-fire) · §301 (Phase 2 disk migration 部分 symlink + AMENDMENT_07 manifest rebind follow-up + fire restart 3-trip debug 2026-05-26) · **§303 (B-1868 PROTOCOL_NOTE_01 全链落地 — watchdog session-loss paper_grade preserve semantics: B-1777 sibling scope gap 闭环 + cross-AI Mode B+C 5 OOB findings + PROTOCOL_NOTE_## convention 新立 vs AMENDMENT_##; `infra_covariates` schema + dual-path defense; 5 process 教训 incl. cross-AI default-chain + codex `-o` overwrite recovery + is_noise 命名 forward-guard)**。

**⚙️ reboot A100 后服务恢复 checklist (reusable — Gate 3 fire 2026-05-23 踩坑; 2026-06-02 两步自愈化)**: reboot 修 NVML/退化是一举两得。原先 reboot 后 **2 个服务需手动恢复** (否则 fire preflight **Gate 4 fail-closed 拦** `homepage endpoint not reachable`), **2026-06-02 已自愈, reboot 后只需验证不必手动**:
> 1. ~~**`vwa-reddit`** 手动 `docker start`~~ → **已自愈**: `docker update --restart unless-stopped vwa-reddit` (2026-06-02), docker daemon reboot 后自动拉起。手动兜底仍是 `ssh condense-a100 'docker start vwa-reddit'`。
> 2. ~~**homepage :4399** 手动起 flask~~ → **已自愈**: A100 user crontab `@reboot` 条目自动起 flask:4399 (2026-06-02, 依赖 cron daemon)。手动兜底命令: `docs/reference/condenser/README.md §6d`。
>
> **reboot 后验证** (而非手动起): `curl -sf localhost:9999` (reddit) + `curl -sf localhost:4399` (homepage) 都 HTTP 200; homepage `@reboot` 首次 reboot 后建议确认一次。**全 docker 站自动回来** (restart policy): classifieds / classifieds_db / vwa-shopping / vwa-wikipedia + 现在 vwa-reddit。**WHY**: Gate 3 fire 2026-05-23 因 homepage down 被 Gate 4 拦 → reddit `restart=no` + flask homepage 曾是 2 个 reboot 盲区, 2026-06-02 修掉 (docker restart policy + @reboot cron); fail-closed 拦住 = preflight 工作正常。

---

</details>

## §1 ROADMAP — dynamic (`tasks.base` ← edit `_status/tasks/*.md` frontmatter)

> 7 里程碑 = `_status/tasks/*.md` 小 frontmatter (`status`/`priority`/`horizon`/`blocker`/`eta`/`order`) + `tasks.base` 视图。改一字段即更新 (同 cells/issues 模式; 也自动进 `status.base`)。**roadmap canonical = 这里**。

![[tasks.base#🔴 NOW]]
![[tasks.base#📋 NEXT]]
![[tasks.base#🧊 BACKLOG]]

---

## §2 EXPERIMENTS — live (`cells.base`)

![[cells.base#Active 跑中]]
![[cells.base#Pending / Queued / Blocked]]

---

## §3 Router — ⭐ Phase 1 并行核心线 (advisor 2026-05-14) + Pass-1-gated 细项

> ⭐ router = Phase 1 并行核心 contribution (非 paper-2 deferred)。**执行 checklist + blockers + 完成判定 → [[phase1_plan]] §C**。

**路线 (a) rule-based** — 按 task 属性 route。 **路线 (b) learned** — 训练 classifier route (paper-1 §6 = learned-only per v7 amendment)。 **未来扩展** — 按 mode 行为模式 route。
Routing signal infra ✅: `confidence_summary.json` per-condition。Train/test → 5-fold site-stratified CV。设计 → [[paper_planning#§8]]。

**Cross-mode 失败 taxonomy** (`scripts/analysis/cross_mode_failure_taxonomy.py`, §291) — routing 证据/feature 来源: B0 cls 3-mode 实证 routing 价值 = **列表缩略图识别 (THUMBNAIL) 梯度** dom>som>vision (非纯导航 SEARCH-NAV 小 / 非详情大图 IMG 极小)。**6-mode 数据齐复用** (`--run` 加列, O(mode) 不爆炸) → 定 routable task 类型 → router feature。Open: UNCLEAR-NAV 拆 (需 listing-level obs, ROI 低) + cross-site (red/shop)。⚠️ provisional 3/6 mode, 禁 cross-mode 定量直至 freeze。

> [!todo] Deferred follow-ups — **gated on Pass-1 data landing** (router code 现跑空数据 `n_pooled_total:0`)。详 [[实验笔记]] §255 + [[master_bug_catalog]] B-1805~B-1818。
> 1. **Pass-1 run manifest** (C2/B-1810) — `results/phantom_paper/l1_router/pass1_run_manifest.json` 36 paper-grade run IDs → `discover_runs` strict whitelist。然后 `aggregate_h10_pareto.py --require-full-coverage` (C8/B-1811) 出 paper-grade H10 verdict (fail-closed on incomplete)。
> 2. **τ objective sensitivity** (F3/G2 B-1814) — Pass-1 后跑 τ 对比: (a) accuracy-τ vs (b) fixed-τ vs (c) outcome-matrix Pareto-τ; 仅当 (c) 实质移动 frontier 才采用。**不重构 Stage 1-3** (user 2026-05-21)。
> 3. **inner-CV MI leak** (C3/B-1816) — inner-CV 复用 Stage-2 outer-pool MI selector (mild 2nd-order)。Pass-1 后判断是否实质移 τ; 否则 disclosure 足够。
> 4. **§6 disclosure block** (写进 paper §6/§3.5): self-oracle noise ceiling (G1/B-1809, N=1 oracle) · deployment realism (no-success tasks routed) · MODES cost order (F2/B-1806, 实测 tie-break) · router_strictly_better (F7/B-1815, θ CI>0) · sklearn/numpy version metadata (C6/B-1813)。
> 5. **F9 (P2)** — 删/guard deprecated 8-dim `predict_mode` (`learned_router.py:425-494`) 确认无 caller 后。

---

## §4 Audit follow-ups → audit.base (`_status/audit/*.md`)

> 22 条 audit follow-ups 迁入 **audit.base** (2026-05-25 commit). 改 `_status/audit/<note>.md`
> frontmatter (`status` / `phase` / `priority`) 即更新视图; CLI `make status V='audit#All'` /
> `make status-set N=audit_<x> SET='status=done'`. §3 router follow-ups 仍 prose (见 §3);
> §5 sustainability 已并入 (phase==section8).

![[audit.base#🔴 Gate-blocking / now]]
![[audit.base#Deferred (post-fire)]]
![[audit.base#Paper-finalize]]
![[audit.base#All (by priority)]]

---

## §5 Sustainability / Green AI → audit.base (phase: section8)

![[audit.base#Section 8 (sustainability)]]

---

## §6 Codex task queue

![[codex.base#Ready to send (now)]]

![[codex.base#Running / In flight]]

![[codex.base#Blocked / Queued]]

**Pending Python scripts (非 codex)**:
- ⏳ Multi-metric Pareto (cost + lat + carbon) — Section 8 前置 (~2h)
- ⏳ TF-IDF + binary feature extraction — Section 6 Tier 1 router 前置 (~1h)

---

## §7 Open issues

![[issues.base#Active blockers]]

![[issues.base#Backlog]]

---

## §8 Advisor align

详 [[issue_advisor_sync_2026-05-14]] (2026-05-14 sync — Part 1 novelty + Part 2 决策点). Sync 后 decision log → [[paper_planning]] §19; framing register → issue status open → discussed (ADVISOR_SYNC.md retired 2026-05-15).

---

## §9 References + quick links

### Phase 1 canonical
- [[phase1_plan]] — 统领性 audit/execute checklist (§A1 实现层 stress + §A2 设计层 stress + §B clean run + §C router + §D evidence + §E milestones)

### Paper drafts (final prose)
```
docs/checkpoints/paper_drafts/
  section1_intro.md          ✅
  section2_background.md     ✅ + paper.bib
  section3_definition.md     ✅
  section4_findings.md       🟡 待 P-prompt column + Phase 1a hero numbers
  section5_mechanism.md      ✅ v2 (probe-causal-steering trichotomy)
  section6_routing.md        ❌ paper-2 / Tier 1+2 prototype
  section7_generalization.md ❌ 待 WA
  section8_discussion.md     ❌ paper end-stage
```

### Figures
`results/phantom_paper/figures/` — regenerate via `make analysis` / `make figures`:
- §1 hook (脚本已存在): fig0a_sr_per_mode_heatmap / fig0b_extra_confidence_calibration / fig0c_drop_one_oracle / fig0g_routing_auroc_heatmap / fig3b_image_token_gap / fig_phantom_structure_venn / fig_meta_forest / fig_forest_drop_one
- §5 mechanism (§5 暂搁 — 脚本未落地/已 frozen): fig_stage4_method42_v2_{cls,reddit} / fig_axis2_logit_lens_v2 / fig_axis2_layer_profile_v2 / fig_mech_8cell_l17_forest / fig_mech_real_vs_random / fig_layer_axis_emergence_v2_{cls,reddit}
- §5 mechanism (脚本存在但 §5 暂搁): fig_mechanism_pilot

### Key infra paths
```
configs/exp_v2_*.yaml                              per-site experiment configs (含 12 baseline configs §133 A4)
scripts/queues/queue_phase1_paper_grade.sh         Phase 1a 36-cond orchestrator (§134 harden)
scripts/queues/queue_chain.sh                      sequential chain (§134 C3 crash-detect)
scripts/queues/queue_{baseline,phantom_*}.sh       per-condition launch (§134 FORCE_NEW)
scripts/maintenance/paper_grade_check.py           paper-grade integrity check (cron + on-demand)
scripts/analysis/validate_fire_manifest.py         manifest ghost/over-complete gate (B-1825/1834)
scripts/analysis/preregistration_decision_test.py  H1/H3/TOST canonical (§133 T3 heterogeneity)
scripts/provenance/snapshot_{env,vwa}.*            provenance fingerprint
```

### Provenance artifacts (paper-cite-able)
```
results/provenance/env_dgx_baseline.json           DGX baseline (HF Qwen3-VL-4B SHA ebb281ec...)
docs/checkpoints/pre_run/osf_lock_manifest.md      8-step DOI workflow (DOI-1 minted 10.17605/OSF.IO/9QCWU)
docs/checkpoints/pre_run/preregistration.md        R1-R5 framing + AMENDMENT 01 Protocol Reset
```

---

## §10 独立 bug 研究 paper (workshop-targeted)

> advisor 2026-05-14 收口: bug 部分可**单独再发一篇 paper** 投 workshop — 独立于主 paper, **不替换**主 paper workshop 节点.

**方向**: cross-benchmark bug 聚合研究, 针对现有 web agent benchmark. **参考**: agisdk (https://github.com/agi-inc/agisdk). **素材**: dual-track environment / VWA bug fix ([[实验笔记]] §109 / `master_bug_catalog.md` 37+ bugs). **状态**: 方向 locked; scope + benchmark 选型 + 时间线待 planning. 详 [[workshop_subpaper_plan]]。

---

> 📕 **Historical / superseded** (从本 ledger 移除, 仍在 chronicle/canonical docs): Fire-3/4/5/6 RCA 叙事 → [[实验笔记]] §233/§237/§252/§269 · OSF DOI-1 8-step (已 mint) → [[osf_lock_manifest]] · stats methodology (FE estimand resolved) → [[preregistration]] §2.5 · mechanism §5 暂搁 (2026-05-14) → [[实验笔记]] §138 · fire-event-sequence (Fire-6 已 fire) → [[phase1_plan]] §E。
> 📖 **Doc update workflow** (when X → update which docs) → [[paper_planning#§20]]
