# Diag Freeze Step v6 — B1 cls 6-mode discover → freeze 合并落码方案

> **状态: 成文待 review, 未落码** (2026-06-08)。B1 cls 6-mode /diag discover 全齐 (§317 som / §318 dom / §320 vision / §321 ptext / §322 psom / §323 pprompt) → discover-then-freeze 协议进入 freeze: 合并 6 condition 的规则提议去重 → 落码 → bump version → 全量重扫拉齐 → **解锁 cross-mode 定量 + k_cells=2 drop-one**。
>
> **当前**: `RULESET_VERSION="5-domsomvispsom-b1860coord"`, 已码 P1-P33 (无 P9/P26, 31 条)。
> **目标**: `6-*`, 加新规则 (P34+) + 3 个确定性 bug-fix + 一批 success-safe 收窄。
>
> ⚠️ **immutability/witness**: `diag_pattern_match.py` 在 `scripts/analysis/` = **不在 fire import 路径** (memory [[feedback-analysis-layer-fire-immutability-and-witness]]) → fire 期间改安全, 不动 SR/eval estimand。diag = **失败 taxonomy 非 estimand** → **无需 OSF witness**。但它喂 cross-mode routable/oracle (§306 provisional 39%/+16pp 是 paper finding 非主 gate) → 必须**版本化 + 全量重扫**保可复现 (本方案 step D)。

---

## A. 新规则 (P34-P39)

> 编号沿用 discover 临时分配 (P34 视觉盲 §319/§305 / P35 mutation §318), P36+ 新增。每条标 `is_scaffold` + **内置 success-safe** (避免再造 presence-only 规则, 见 skill 陷阱#2)。

| P# | 名 | signal (0-token) | 类 | success-safe 内置 | source |
|---|---|---|---|---|---|
| **P34** | VISUAL_BLIND_IMAGE_TASK | `obs_mode in phantom_*/dom AND image-dependent` (`task_config.image≠null` OR intent 含 `in the image`/`picture`/`cover`/`color`/`looks like`) | agent-limit | ⚠️ **presence-only 风险高** (文本侥幸答对则 success-fire) → **硬 sub-signal**: `input_image=0 AND steps≤3 AND finish.answer 含 'cannot verify'/'image not visible'/'not listed'/'[]'` 才 fire causal; 否则标 soft-risk | §319/§305 P34 · §321/322/323 no-hit 主轴 |
| **P35** | MUTATION_MISSING | `eval_type=program_html AND (eval_source_agent_url 含 'item_edit' OR eval locator 含 '.comments_list') AND effective_mutating_action_count=0 AND agent_finished=true` | agent-limit | finish 但 0 mutation = 死因, success 不会触发 (success 必有 mutation) → 天然 success-safe | §318 P35 · §320/321/322/323 (75/76/208/213/9) **5-mode 收敛** |
| **P36** | WALK_FAIL_DEGENERATE | `locator_route_meta.error 含 'walk_fail'` (`no_input_within_walk` [type 变体] ∪ `no_actionable_within_walk` [click 变体]) | agent-limit | ⚠️ **B-1869**: walk_fail 21.7% 报 `action_success=True` → **success-fire 时标 presence-only** (walk_fail no-op 但 ep 靠别 action 成功, psom 4 success-ep 实证) | §321 (P4 success-safe 正解) · §322 (∪click) · **§323 最强 880act/149ep** |
| **P37** | URL_HALLUCINATION | `'example.com' in finish_answer AND any reference 含 'localhost'` | agent-limit | finish≠ref, 0 FP 风险 | §320 (task 35) · §322 |
| **P38** | DOM_URL_AS_IMAGE | `intent 含 'in the image'/'website...image' AND 该页有 img AND finish.answer 含 'localhost' (任一形态: page-URL/base-URL/img-src .png 路径 — **三者 obs 都在, 无 surface 分支**) AND reference 是真实外域 (非 localhost)` | agent-limit | finish≠ref | §321/322/**§323 (CORRECTED 单-surface, 见 §323 CORRECTION)** |
| **P39** | SPURIOUS_PASS (delete/mutation false-success) | **success=True** AND `eval_type=program_html (delete/mutation task) AND delete_remove_count=0 AND effective_mutating_action_count=0 AND agent_finished=false` | **benchmark-FP (SR-抬高)** | 本规则**专测 success 侧** (与其他失败规则相反) → 需独立 success-scan 路径 | §323 (task 5, B1 phantom sweep 首例) |

**P36 与 P4 关系 (关键决策, 见 §E-3)**: P4 (`element_id∈{0,1}` 裸号) = **`[SOM_MARKS]`-specific surface** (psom 63/ptext 41/pprompt **0**); P36 walk_fail = **mode-robust** (跨 text 表征抓退化引用)。§323 实证 "裸 element_id 信号 mode-fragile / walk_fail mode-robust"。**建议两者都保留** (P4 描述 [SOM_MARKS] 特有形态, P36 抓底层), 不 deprecate P4。

---

## B. 确定性 bug-fix (改现有规则正则/guard)

| 规则 | bug | fix | source |
|---|---|---|---|
| **P18** (cheapest 漏排序) | task 56: agent step3 已 `sOrder=i_price&iOrderType=asc` 排序, P18 仍 fire = **false positive** | 入口 guard: `if any obs_url 含 'sOrder=i_price' (或价格排序参数): return []` | §323 |
| **P19** (url_match 过早搜索页 finish) | task 23/28: eval 实为 string_match/program_html **非 url_match**, P19 走 fallback (无 finish 用 last obs_url) 误 fire | 加 guard: 仅 `eval_type == url_match` 才 fire; `has_finish=False` 时归 P31 不归 P19 | §320/321/323 收敛 |
| **P10** (跨步数值记忆失败) | task 45/87/93: `finish.answer` 是 URL 字符串, 端口 9980 + item id 被当"应记忆数字" → 系统性 FP | 加 guard: `if finish.answer.startswith('http'): return []` (URL 内嵌数字非记忆事实) | §320/321/322/323 收敛 |

---

## C. success-safe 收窄 (presence-only → causal, 0-token 非 success-label)

| 规则 | 现状 | 收窄 | source |
|---|---|---|---|
| **P4** | psom 24 hit / 4 success-ep 全 presence-only (点 ghost [1] 但靠别 action 成功) | walk_fail no-op 且 success=True 时静默 (用 P36 walk_fail 信号判, 非裸 element_id) | §322 |
| **P31** (budget 耗尽 incomplete) | **跨 6 mode (全) confound**: finish-less arrival artifact (到达正确 URL 空转, url_match/agent_page 自动 pass) | `eval url_match/agent_page AND agent_url≈reference_url` 豁免; success=True 强制静默 | §317/320/321/322/**323 全 mode** |
| **P5 / P14 / P12 / P17** | success-fire (productive-arrival / 点 img-PNG 循环 / 视觉盲验证式 oscillation 成功) | success=True → 这些标 presence-only; P14 已 v3 (productive 长停留) 范式可复用 | §320/321/322/323 |
| **P33** (裸 PNG 幻觉) | psom 7/7 causal, 但需分级 | sub-signal: 连续≥2 步 obs_url 含 `.png` 未 back → `severity=high` (PNG 卡死, 区别 B0 §304 单次 back) | §322 |

---

## D. 执行序列 (落码时按此)

1. **码 A (P34-P39) + B (3 guard) + C (收窄)** 进 `scripts/analysis/diag_pattern_match.py`; A 的每条 `check_pNN(steps, summary, config, mode)` + 注册进 `ALL_RULES`; scaffold 类标 `is_scaffold=True` (本批无, 全 agent-limit/benchmark-FP); **P39 走独立 success-scan** (其他是 failed-scan)。
2. **bump** `RULESET_VERSION` → **`6-b1clsfull-b1860coord`** (或 §E-1 定名)。
3. **同步 skill P-rule 列表** (`.claude/skills/diag/SKILL.md` 的 31 条清单 → 新条数) —— memory 教训: R31194 session 漏更停"13 条"半月。
4. **全量重扫**: `bash scripts/maintenance/diag_autorun.sh results/visualwebarena/phase1` → 所有 condition Tier-1 JSON 落同一 `ruleset_version`。
5. **验证**: (a) 新规则 fire 预期子集不误伤 success (跑 P34-P39 on B1 cls 6 mode 抽检); (b) 全 condition JSON 版本号一致; (c) bug-fix 后 P18 task56 / P19 task23-28 / P10 task45 不再 fire。
6. **解锁 cross-mode 定量**: `cross_mode_failure_taxonomy.py --run <6 mode dirs>` 读统一版本 JSON → routable/oracle 真数 (replace §306 provisional) + 喂 §1 drop-one (k_cells=2)。digest 的 `ruleset_version` header 手动 sync 到 `6-*`。

---

## E. 待你 review 的决策点

1. **版本命名**: `6-b1clsfull-b1860coord`? 还是别的 (反映 "6-mode + B1 cls sweep")？
2. **P39 (DELETE_FALSE_SUCCESS) 归属**: 是 diag-rule (本方案 A) 还是 **AMENDMENT_08 exclude 候选** (task 5 是 SR-抬高 benchmark-FP, 类似 §299 的 exclude list)? 同理 §320 的 `URL_MATCH_WRONG_ITEM` (task 20)。**两者可能更该进 exclude-list 而非 diag-rule** —— 它们是评测 FP 非失败模式。建议: diag 标记 + 同时登 AMENDMENT_08 候选, 不二选一。
3. **P4 是否保留** (vs 全用 P36 walk_fail): 建议**保留 P4** (描述 [SOM_MARKS] 特有 surface, 是 axis-2 证据), P36 补 mode-robust 底层。两者并存。
4. **P34 (VISUAL_BLIND) 的 presence-only 风险**: 这是最难的一条 (图像依赖任务文本侥幸答对 → success-fire)。是否只码硬 sub-signal 版 (input_image=0 + 放弃措辞), soft 版留 cross-mode taxonomy 工具处理? 建议: 只码硬 sub-signal, 避免污染。
5. **B-1869 联动**: P36 落码时内置"walk_fail-but-success → presence-only" = B-1869 measurement 隐患的 diag 侧 mitigation (action_success gating 本体仍 post-fire)。确认这样分工。
6. **何时落码**: 现在 (fire 跑 B2 cls 中, 不冲突, analysis-layer)? 还是等 B2/red 也 discover 完一起? **建议现在** —— 6-mode (B1 cls) 已够 freeze 一版解锁 B1 cls cross-mode; B2/red 落地后按协议再扩一版 (`7-*`) 全量重扫 (版本渐进, 非一次定死)。

---

## Cross-link
- discover 源: 笔记 §317-323 + `docs/analysis/vwa_classifieds/B1_{dom,som,vision,phantom_text,phantom_som,phantom_prompt}_classifieds_diag_digest.md`
- 协议: `.claude/skills/diag/SKILL.md` 「跨 condition / cross-mode 工作协议 (discover-then-freeze)」
- 工具: `scripts/analysis/diag_pattern_match.py` (ALL_RULES) · `scripts/maintenance/diag_autorun.sh` (全量重扫) · `scripts/analysis/cross_mode_failure_taxonomy.py` (§291 cross-mode 定量)
- 下游解锁: §1 hero drop-one gate (`phase1_full_prereg_decision.md`, 现 INSUFFICIENT_DATA k_cells=1 → B1 cls 第2 cell → k_cells=2)
