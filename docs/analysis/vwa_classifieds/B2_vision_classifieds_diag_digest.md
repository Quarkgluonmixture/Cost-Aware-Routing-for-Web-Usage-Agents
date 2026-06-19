# B2 vision classifieds — /diag failure attribution digest

**Run**: `B2_vision_classifieds_20260612_221910_098760264_1351451_R9288` (manifest-bound authoritative)
**Condition**: phase1_vision_router_0 · **Site**: classifieds · **Model**: B2 = Gemma3-4B · **Mode**: vision (raw screenshot only, 无 AXTree 文本)
**N**: 224 ep · **SR**: 5/224 = **2.2%** · **ruleset_version**: `5-domsomvispsom-b1860coord`
**Diag date**: 2026-06-19 (Tier-1 全扫 + Tier-2 sonnet 深挖 8 ep)

> ⚠️ 单 condition digest，不下 cross-mode 结论。cross-mode 定量待 B1+B2 cls freeze。

## 1. 三分类统计

| 类别 | 占比 | 说明 |
|---|---|---|
| **agent-limit** | ~100% (219/219 failed) | Gemma3-4B vision grounding 地板。8 ep Tier-2 全 agent-limit |
| scaffold-bug | 0 | ⚠️ Tier-2 确认: **截图传递正常 + 坐标转换无误** = 无 vision scaffold bug |
| benchmark-FP | 0 | — |

## 2. Tier-1 规则分布 (failed per-rule, hit 总数)

`P5`(感知缺失循环)=**329** · `P31`(budget耗尽未完成)=197 · `P14`(URL自环)=119 · `P19`(url_match过早finish)=45 · `P18`(漏价格排序)=32 · `P20`(评测目标页从未访问)=25 · `P12`=13 · `P25`=13

→ **P5=329 是 6 mode 最高** = 纯像素 grounding 下感知缺失最严重 (无 AXTree 文本兜底 → Gemma 看不懂 screenshot 反复试)。

## 3. Tier-2 深挖

**no-hit failed (5, 全 agent-limit)**:
- **task 10**: early-abandon (到错误 item Canon 镜头 → 判 "task impossible, no furniture visible" → 2 步 finish)
- **task 50**: wrong-item navigation (读错 listing 的 email)
- **task 119**: thumbnail 视觉读取失败 (钞票面额读成 "$1" vs ref "$50")
- **task 215**: 无搜索策略 (随机点 Xbox 2 步 finish)
- **task 221**: 到达正确页但计数失败 (数碗答 "0" vs ref "6")

**success 审计 (3, 全 presence-only 伪成功 `hit_causal=false`)**:
- **task 121 / 123 / 193**: `agent_finished=false` + `trajectory_incomplete=true` + runner step-limit 截断时 url_match 通过，no_op_rate 64-83% (反复无效 click 偶然停在目标 URL) = **与 som 87/124 同模式跨 mode 复现**。

## 4. 🔁 Self-evolving — 提议 P-rule (post-fire candidates)

⭐ **P-presence-only (最高优先, vision 给出最精确信号)**:
```
agent_finished=false AND trajectory_incomplete=true AND success=true
AND eval_context_mode='agent_page'  →  presence_only_rescued
```
命中 vision 3/3 + som 87/124 = **跨 mode 系统性, 零 FP**。从 SR 剔除或单独报告 (剥离「runner 救活」vs「agent 完成」)。

→ ruleset 冻结待 B1+B2 cls freeze (§0 diag_freeze_v6_plan)。

## 5. Actionable

- 无 scaffold-bug · 无 benchmark-FP。
- **B2 vision cls = agent-limit 地板** (P5 感知缺失 6-mode 最严重 = 纯像素无文本兜底)。
- ⚠️ **presence-only 伪成功跨 som+vision 复现** = B2 url_match SR 系统性含 runner-救活成分 (paper 需注 + post-fire P-rule, B-1869 sibling)。
