# B2 phantom_prompt classifieds — /diag failure attribution digest

**Run**: `B2_phantom_prompt_classifieds_20260616_142027_795794905_1801050_R10175` (manifest-bound authoritative)
**Condition**: phase1_phantom_prompt_router_0 · **Site**: classifieds · **Model**: B2 = Gemma3-4B · **Mode**: phantom_prompt (SoM prompt + AXTree 文本 + 无图; axis-2 control)
**N**: 224 ep · **SR**: 4/224 = **1.8%** · **ruleset_version**: `5-domsomvispsom-b1860coord`
**Diag date**: 2026-06-19 (Tier-1 全扫 + Tier-2 sonnet 深挖 9 ep)

> ⚠️ 单 condition digest，不下 cross-mode 结论。cross-mode 定量待 B1+B2 cls freeze。

## 1. 三分类统计

| 类别 | 占比 | 说明 |
|---|---|---|
| **agent-limit** | 6/9 Tier-2 | Gemma3-4B phantom_prompt 地板 |
| **benchmark-FP** | **3/9** (task 5/110/142) | ⭐ pprompt = B2 cls 最多 FP 的 mode |
| scaffold-bug | 0 | — |

## 2. Tier-1 规则分布 (failed per-rule, hit 总数)

`P5`(感知缺失)=288 · `P31`(budget耗尽)=207 · `P14`(URL自环)=153 · `P2`(容器节点误点)=65 · `P19`=55 · `P12`=55 · `P33`=52 · `P18`=34

→ P5+P31 主导 (Gemma 地板)。

## 3. Tier-2 深挖

**no-hit failed (5, 全 agent-limit)**:
- task 40 (搜索识别错 Whirlpool→LG) · 61 (误读任务 video game→bowling) · 108 (幻觉直跳 item 10542) · 111 (无图判球衣队名失败) · 174 (Black Friday logo 图像识别, 无图)

**success 审计 (4) — ⭐ B2 success 大量非真能力**:
- **task 5**: presence-only + FP (item 84144 run 前已 404，agent 从未删除，success 来自 eval program_html 对 404 返回 PASS) — 注意 ≠ ptext task 5 (那个 agent 真删了)
- **task 106**: 真成功 (constraint-skipping 侥幸：正确导航到 Photo 类最贵 Canon item，email 答对，但跳过 animal-image 约束 — phantom 无图无法验证，碰巧最贵 item 即正解)
- **task 110**: benchmark-FP (lucky numeric guess "0"，未访问正确 item 34406，ref "0|OR|zero" string_match 宽松通过)
- **task 142**: benchmark-FP (访问错误 item 65955 而非 ref 22310，但两者都在 Pennsylvania，string_match("Pennsylvania") 单点巧合命中，eval 不验 URL)

## 4. ⭐ 关键: B2 cls 名义 SR 严重虚高 (真实有效 ≈ 0)

pprompt SR 4/224 拆: **3 benchmark-FP + 1 constraint-skipping 侥幸** → **真实有效 SR ≈ 0**。
- FP 三源: presence-only (runner 救活) + lucky-guess (string_match 宽松短答案) + string-coincidence (不验 URL)
- **task 142 跨 psom+pprompt 都 FP** (string_match 不验 URL = 系统性 eval 宽松)
- → 印证 §335-338 B2 cls 真地板，且**更强**: 连那 1-4 个 success 都非可靠能力。

## 5. 🔁 Self-evolving — 提议 P-rule (post-fire candidates)

1. **P-vision-required** (phantom mode image-identification: intent 含 image/logo/jersey + obs_mode∈phantom*) 覆盖 61/111/174
2. **P-lucky-numeric-FP**: `eval_type=string_match + reference ∈ 短数字/否定词 + agent_url ≠ correct item URL` → benchmark-FP candidate (task 110)
3. **P-presence-only-delete**: `agent_finished=false + trajectory_incomplete=true + delete_remove_count=0 + success=true` (task 5)

→ ruleset 冻结待 B1+B2 cls freeze (§0 diag_freeze_v6_plan)。

## 6. Actionable

- **benchmark-FP task 5/110/142**: post-fire 评估是否排除 (string_match 宽松 + presence-only；142 同 psom = 系统性 eval-不验-URL)。
- **B2 pprompt cls = agent-limit 地板**；名义 SR 1.8% 虚高 (真实有效 ≈ 0)。
