---
type: action-ledger
status: rolling
updated: 2026-06-03
---

# Next Steps — Forward Action Ledger

> **Future-only**. **Roadmap = §1 `tasks.base`** (dynamic — edit `_status/tasks/*.md` frontmatter). Live state 不在这里:
> - cron health / 错误扫描 / ntfy 历史 → `make ntfy` + `logs/cron/*.log` (PLAYBOOK retired 2026-05-23, §279)
> - Real-time active runs / GPU → `make active` CLI (DGX only; fire 在 A100)
> - Paper-grade fire verdict → `paper_grade_check.py` (一条命令, §0) + 每 6h cron
> - Cell snapshot (active / pending / done) → `cells.base`
> - Paper section progress → `status.base`
> - 过去 chronicle → [[实验笔记]] (latest §323, 2026-06-08 — §317-323 B1 cls cross-mode /diag discover **6-mode 齐**: som finish-less/dom 真卡死/vision **router-crux REFINE**/ptext+psom **axis-2 P4 phantom-family-wide RESOLVED**/psom **P33 4B>235B gap**/**pprompt P4=0 但 walk_fail 主导 → text-表征轴精化 + DOM-URL 视觉盲(⚠️撤回"修正B0§319"误判, img-src 全文本模式暴露 B0§319成立) + task5 false-success FP 首例**; §319 lit-digest)
> - Strategy / theory → [[paper_planning]]
> - **Phase 1 执行计划 + audit checklist** → [[phase1_plan]] ⭐ canonical
> - OSF DOI lock workflow → [[osf_lock_manifest]] · Compute infra → [[COMPUTE_INFRASTRUCTURE]]
>
> 🔧 新数据 → `make analysis` (~5-10min). Cron 每 10min 自动 sync cell frontmatter.

---

## §0 SESSION HANDOFF — 新 session 接手 ⭐ 先读这个

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

> 📅 **2026-06-10 advisor 两份书面 deliverable + deadline 自查 (standing, 每 session 必查)**: **D1 one-pager → advisor self 06-12 / 官方硬线 ≤06-22 (ASAP)** (`deliverables/advisor_onepager_2026-06.md` 已起草待 review) · **D7 lit-review 章 self 07-13 / 官方硬线 ≤07-20** · D4 Pass-1 全 36 cond 06-26 · D5 full analysis + drop-one verdict 07-01 · D6 router H10 verdict 07-08 · D9 thesis draft v1 08-10 · D11 submission early-Sep (**TBC 待确认**)。全表 canonical = `issue_advisor_sync_2026-06-10` D1-D11 表 (one-pager §4 精简为 6 行关键节点保 1 页); live = `make status V='tasks#NOW'` (eta 字段)。**每 session handoff 对照自查 on-track, 偏差主动报 + 提议 re-plan** (advisor 显式要求 "always check that you are on track")。详 `issue_advisor_sync_2026-06-10` + 笔记 §331。

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
