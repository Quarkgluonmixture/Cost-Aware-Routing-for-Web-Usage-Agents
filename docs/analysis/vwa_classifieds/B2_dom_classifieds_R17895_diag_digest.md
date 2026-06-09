# B2 dom classifieds (R17895) — diag digest 【ARCHIVED — pas-off config-ablation arm】

> ⚠️ **SUPERSEDED 2026-06-09 (pan-and-scan amendment, 笔记 §328)**: R17895 跑于 `do_pan_and_scan=False` (Gemma 默认 256-tok squash 视觉配置)。A100 probe 证明该配置触发灾难模式 (复读崩溃/数量爆炸) → B2 全系改用 vendor-recommended `do_pan_and_scan=true` 重跑。**R17895 降级为 config-ablation arm** (pas-off 对照数据, 不进 paper-1 canonical), 本 digest 随之 run-suffix 归档。新 canonical digest 将在 pas-on rerun 完成后以默认名重建。本文 findings 中与配置无关的部分仍有效 (B-260 Magento gap / P33 cross-mode 重分类 / false-success 审计方法)。
>
> Per-condition failure attribution (site × model × **mode**)。**首次 B2 (Gemma3-VL 跨族) condition 诊断** — 2026-06-09。

## Header

| 字段 | 值 |
|---|---|
| **Run** | `B2_dom_classifieds_20260608_102408_385681148_799282_R17895` (manifest-bound authoritative) |
| Condition | `phase1_dom_router_0` |
| Site / Model / Mode | classifieds / **B2 = Gemma3-VL (`google/gemma-3-4b-it`, 跨族 4B)** / dom |
| N episodes | **223** (224 step-logs, 1 缺 summary) |
| **SR (raw)** | **2.2% (5/223)** |
| **SR (legit-corrected)** | **0.9–1.3% (2–3/223)** — 5 个 success 里 2 个 false-success (T5/T110, 见 §4) |
| Tier-1 coverage (failed) | **99.5% (217/218)** — P31 budget 规则吃满，no-hit 仅 1 |
| `ruleset_version` | **`5-domsomvispsom-b1860coord`** (与 B0 psom/pprompt 同版; dom/som/vision 旧 digest 标 `4-domsomvis` 待全量重扫拉齐) |
| Tier-2 深挖 | 27 ep (4 batch × sonnet)：success 审计 6 + Gemma 格式 7 + P33 7 + loop-to-death 7 |

> ⚠️ **discover 阶段 — 单 condition 不下 cross-mode 定量结论**。B2 其它 mode 未跑、ruleset 未 freeze 重扫。本 digest 只描述 B2 dom cls 自身。cross-mode 比较前须 `diag_autorun.sh` 全量重扫到同一 ruleset。

---

## 1. 一句话结论

**B2 (Gemma3-VL) 在 dom + cls 近乎全军覆没 (legit SR ~1%)，主因是 agent-limit（能力局限），NOT scaffold 解析 bug。** Gemma 的 action 98% 能被 Qwen-调的解析器正常解析 —— 灾难性 SR 来自模型行为本身：**从不 emit finish (224 ep 仅 23 次)、loop 到 budget 上限死亡 (P31 89%)、感知缺失循环 (P5 268 fires)**。**但** 父-context forensic 翻案了一条 sub-agent 误判为 agent-limit 的真 scaffold 问题 —— 经查重定为 **既有 B-260 的 Magento gap**：canonical `locator.fill()` 在 **Magento(7770) 搜索框上仍 query 累积**，与 B-260 "wrapper fill() → paper-grade data unaffected" 论断**矛盾**。范围限 shopping 站、对 cls SR 影响可忽略，但 **B-260 "数据无影响" 对 Phase 1b shop 不成立** = 高优先级 shop fire blocker。

---

## 2. 三分类统计 (Tier-1 hit + Tier-2 深挖合并)

| 类别 | 占比估计 | 说明 |
|---|---|---|
| **agent-limit** | **~95%** | 主导。loop-to-budget-death + 视觉盲 + URL 操纵不收敛 + 从不 finish。Gemma 跨族 4B 能力远低于 B0/B1 在 dom 的表现 |
| **scaffold-bug** | **~2% (4 ep)** | **B-260 Magento gap**：canonical `locator.fill()` 在 Magento(7770) 搜索框上仍 query 累积 (T200/207/225/230)。**仅 cross-site 漂到 shop 的任务触发**，cls 站(9980) 915 个 type 零累积 → 矛盾于 B-260 "data unaffected" |
| **benchmark-FP / false-success** | **~1% (2 ep)** | T5 (program_html isolated-context pass，agent 0 mutation)、T110 (string_match 答案侥幸匹配 reference=0)。**pipeline 未 auto-flag** (benchmark_noise=False, sr_excluded=False) |
| **unclear** | <1% | T45 edge（文字搜索撞对 item，DOM 看不到图有运气成分） |

> ⚠️ **scaffold/FP 占比是"事件计数"非"死因计数"** —— 4 个 type-累积 ep 的**主死因**仍是 agent-limit (loop)，scaffold 累积是叠加层。三分类按"该 ep 是否含此类信号"统计，非互斥。

---

## 3. Tier-1 规则分布 (failed-only per-rule, 多 fire 计数)

```
P5 :268  感知缺失循环 (重复 click 页面不变)        ← 主导行为签名
P31:198  budget耗尽未完成 (trajectory_incomplete)  ← 89% ep 终态
P14:129  URL自环 (stuck no-progress)
P6 :103  视觉任务DOM必败 (dom-gated)
P12: 75  从不翻页
P33: 71  导航至裸图URL幻觉 (dom 也大量 fire — 见 §4.3)
P2 : 51  容器节点误点
P16: 51  视觉图像内容 (dom-gated)
P19: 44  url_match过早搜索页finish
P18: 34  cheapest漏价格排序
P20: 26  评测目标页从未访问
P25: 11  跨站任务跳过其中一站
... (P17/P15/P10/P30/P22/P11/P24 长尾)
```

**success-fire (FP 源, 4 个 success ep)**: P5×3 / P6×3 / P14×2 / P12×2 / P31×1 / P16×1 — 全是 presence-only，与 B0 dom 同构 (P6 在 success 上仍 fire = "视觉任务"是风险标记非必死)。

**0-token scaffold 健康度全扫** (6206 steps)：
- `parse_valid`: **98% (6083/6206 True)** — Qwen-调解析器对 Gemma 输出**基本兼容**，排除"格式解析不了"粗 bug
- `parse_failure_reason` (123 fails): `invalid_select_option`×77 / `multiple_actions`×19 / `invalid_element_id`×18 / `invalid_action_type`×8 — **全经 Tier-2 判为 genuine model error 非 scaffold** (§4.2)
- `action_type`: click 3375 / type 1036 / **goto 1034 (16.7%, 异常高)** / select_option 403 / ... / **finish 仅 23**
- `goto` 1034 拆分：**search 753** (URL 拼搜索) / item_page 131 / home 84 / IMAGE_FILE 18

---

## 4. Tier-2 新发现

### 4.1 主导死因 = loop-to-budget-death（纯 agent-limit，无 scaffold 成分）

batch-4 forensic：7/7 ep 的 `page_changed`+`obs_url` **完整响应**了 Gemma 的 action（goto 真导航、type 真改 URL、click 在无效元素上 page_changed=False）。Runner 执行无误，环境无 scaffold bug。Gemma 两个典型行为失败模式：

1. **goto-thrash + 从不 finish**：7 ep 合计 finish=0。卡住时只会 (a) 重复 goto 同一 search URL (T4 goto user/items ×15；T56/36 同 search URL ×6-8)，或 (b) 反复 click 同一元素 (T28 点同一图片元素 ×26) → budget death。
2. **goto-search 不收敛**：Gemma 能拼对 search URL（T36 用对 sOrder+sPattern），但**从不读结果页价格/颜色**（T1/56 "can't filter by color" 后直接点第一个）、**从不排序** (0 次 sOrder=dt_price)、**从不翻页**，得到结果立刻离开 → search↔first-item 双节点环。cross-site 任务 (|AND|) thought 承认要换站但**行动层从不跨 port**。

### 4.2 Gemma 格式异常 = genuine model error，**B2 数据无需重跑**

batch-2 读 `raw_action` forensic：`invalid_select_option`×77 是 Gemma **JSON 输出不稳定**——同一 task 内 step0 输出完整 `{select_option, element_id, option_label}`、step4 随机漏 `option_label`、step6 又恢复 → **它知道格式，只是随机漏字段**。同 run 60+ 次 valid select_option 被正常解析 = 解析器无系统性失败。`multiple_actions`×19 是 Gemma 偶发一次输出 2 个 action JSON（解析器取第一个有效的），稀疏 <10% 非死因。**判定：agent-limit，解析器行为正确，不出 B-number。**
- 📌 minor infra 建议：`raw_output` 未存进 step_record → `multiple_actions` 无法 100% 复核。建议 `qwen3vl_agent.py step()` 把 `meta["raw_output"]` 写入 step_record（便于未来跨族格式审计）。

### 4.3 P33 在 dom 是**有效信号非 over-fire**，NO mode-gate — 但 docstring 需更正

batch-3 纠正了"18 goto vs 71 fire = over-fire"的**单位误判**：18 = `goto` 到 .png 的 **action 数**；71 = 着陆到 image 页的 **episode 数**。P33 靠 `obs_url` 着陆模式 fire（与 action_type 无关）。实际机制：
- **67 ep**：click AXTree 里的 `<a href=".../id.png">` image-thumbnail 链接 → image 页
- **11 ep (18 action)**：goto 直接构造 image URL

→ **dom AXTree 暴露 image-href 链接，Gemma-4B 误认作 item-detail 链接去点 → 困在裸 PNG**，与 phantom_som 的 [SOM_MARKS] img-href 是**结构同构的幻觉**。P33 success-fire **0/71 = success-safe**。5/7 真 image-hallucination；2/7 (T95/107) P33 在末步才 fire 非死因（真死因 = wrong-item）。
- ✅ **结论：P33 不需 mode-gate，dom 实证有效。** 但 P33 docstring + /diag SKILL.md 现写"phantom_som 特有 / phantom_som 结构诱发"是**不完整的** —— dom 经 AXTree img-href 同样诱发。**P33 实为 cross-mode 通用规则 (text-bearing modes 共有)，非 phantom_som-specific**。
- ⚠️ 但**不能从 B2 dom 单点断定** P33 的 cross-mode 分层 —— vision mode (无 AXTree/无 href) 理论上无可点 href → P33 应不 fire。需 B2 som/vision 数据验证"text-bearing modes 共有、vision 缺失"假设 (discover 阶段 follow-up)。

### 4.4 ★ 父-context forensic 翻案：type-累积是 scaffold 问题 (定为 **B-260 Magento gap**) 非 agent-limit

batch-4 sub-agent 把 T200/207 的 query 累积（`lion+pillow`→`lion+pillowlion+pillow`→…）判为 agent-limit("Gemma 不懂 type 是 append-not-replace")。**父 context deterministic forensic 翻案**：
- **Gemma 每步 type 值恒定 = `'lion pillow\n'`**（从不累积），但 URL q-param 增长 → 累积来自**环境没清空输入框**，Gemma 做的是对的。
- 定位：`dispatch_path=element_id_locator_route`, `fallback=false`, `locator_route_meta.success=true` —— 走的就是 **B-260 声称为 canonical fix 的 id-based `locator.fill()` 主路径**（注释号称 auto-clear `locator_dispatch.py:241`），但在 **Magento(7770) 服务端预填的搜索框**上实测**没清空**。
- **按 port 全扫界定范围**：query-doubling **只在 7770(Magento)** = 64 步 / 4 task (200/207/225/230)；**classifieds(9980) 915 个 type action 零累积**（OSClass 搜索框 fill() 正常）。
- → **这是既有 B-260 的 gap，非新 bug**。B-260 (DOC-DISCLOSED, 2026-05-16) 记 "重复输入会接在上一次后面" + 断言 "wrapper `locator.fill()` → paper-grade data unaffected"。本 forensic **反驳该断言在 Magento 成立**：fill() 主路径在 Magento 搜索框上仍累积。范围 = Magento 搜索框专属，对 B2 dom **cls** run 仅 4 个 cross-site 漂移任务触发 (cls SR 影响可忽略) —— **B-260 "data unaffected" 对 Phase 1a cls/red 成立，对 Phase 1b shop 不成立**。
- ⚠️ DOM 根因（fill() 为何在 Magento 不清空：bbox-center walk 命中 autocomplete 叠层？server `value=` attr 时序？secondary input？）须 live Magento 复现确认 —— archived run 状态已漂移，diag 协议禁 live 复现。本 digest 只断言 forensic 可证的 empirical fact（type 值恒定 + URL 增长 + fill 主路径 success=true + port-7770-only）。
- 📌 **方法论双重验证**：(1) log-only sub-agent 没做"type 值恒定 vs URL 增长"对照 → 误判 agent-limit；(2) 父 context **查重** master_bug_catalog 避免误标新 B-number + 撞见与既有条目矛盾的证据。印证协议"scaffold/FP 判定不信 log-only，须 deterministic forensic" + memory 纪律"加 entry 前查重"。

### 4.5 false-success 审计：raw SR 2.2% → legit ~1%

5 个 success 经 batch-1 + 父 context summary-field 复核：

| task | eval_context | 判定 | 依据 |
|---|---|---|---|
| **5** | isolated_program_html | **false-success** | agent 30 步从未访问 item 84144；`effective_mutating_action_count=0` 但 program_html 在 isolated context 对 pre-existing-404 item pass。**pipeline 未 flag** (benchmark_noise=False) |
| **110** | no_browser (string_match) | **false-success** | 推理错 item (JBL 音响→"0 games")，answer="0" 侥幸匹配 reference="0\|OR\|zero"；正确 item 34406 从未访问 |
| 25 | no_browser (string_match) | real-success | 正确找到 2023/11/16 的 boat，轨迹合理 |
| 106 | no_browser (string_match) | real-success | homepage DOM 直读正确 item，email 精确匹配 |
| 45 | agent_page (url_match) | edge/unclear | 文字搜索撞对 item 45196，url_match pass，有运气成分 |

→ **legit SR ≈ 2–3 / 223 = 0.9–1.3%**。⚠️ T5/T110 两个 false-success **pipeline 未 auto-flag**，直接抬高 B2 reported SR。SR 本就极低，2/5 假阳 = 真值约腰斩，**paper 报 B2 SR 时须用 legit-corrected 数**。

---

## 5. 代表 episode

| 类别 | task | 死因 | 证据 step |
|---|---|---|---|
| agent-limit (loop) | **T28** | click 同一裸 PNG 元素 ×26，零 goto/back 逃逸 | step1 click→.../50169.png; step2-29 全 click 同元素 page_changed=False |
| agent-limit (cross-site) | **T36** | thought 承认要去 OneStopMarket 但行动层从不跨 port | step4 找到 Luigi's Mansion; 0 步 obs_url 含 7770→实际卡 9980 |
| agent-limit (视觉盲+幻觉) | **T107** | 把 sofa 误认作自行车，finish "sofa has no rims" | step18 finish; eval 期待 purple/blue |
| **scaffold (B-260 Magento gap)** | **T200** | Gemma type 恒定"lion pillow"，Magento 框 fill() 仍不清空→query 指数累积 | step0 q=lion+pillow → step9 q=lion+pillow×6 |
| **false-success** | **T5** | 从未访问 item 84144，isolated program_html 对 404 item pass | effective_mutating_action_count=0, success=True |
| **false-success** | **T110** | 推理错 item，answer="0" 侥幸匹配 | step16 finish "speaker→no games→0" |
| no-hit blind spot | **T16** | start 搜索页立即 goto 错 item，3 步 finish 语法有效 email | step0 goto item 33795 (grinder≠coffee mug); step2 finish |

---

## 6. 🔁 Self-evolving — 提议 P34+ 规则 (discover 产物，**未落码**)

> ⚠️ 落码任一条须 bump `RULESET_VERSION` + `diag_autorun.sh` 全量重扫所有已扫 condition (保持 ruleset 一致才能 cross-mode 比较)。本 digest 仅提议，落码留作**批量 bump 单独 step** (避免半 freeze)。优先级排序：

| 候选 | signal (0-token) | 类别 | 优先级 | 来源 |
|---|---|---|---|---|
| **P34 image-URL-trap** | `obs_url` 匹配 `/oc-content/uploads/.+\.(png\|jpg\|jpeg)$` 连续 ≥5 步 | agent-limit | **高** | T28/199 (Gemma 困死裸图，比 P33 更窄更 causal) |
| **P35 search-accumulation** | `obs_url` q-param 含重复词 `([a-z]{3,})\1+` **AND** port=7770 | **scaffold (B-260-gap 检测器)** | **高** | T200/207 — 检测 Magento fill() 不清空污染 |
| **P36 cross-site-never-depart** | task start_url 含 `\|AND\|` 但全 ep obs_url 仅一个 port | agent-limit | 中 | T36/200/207 跨站导航失败 |
| **P37 never-finish-budget-death** | `trajectory_incomplete=True` AND finish action count=0 AND steps≥budget-2 | agent-limit | 中 | 224 ep 仅 23 finish (Gemma 签名) |
| **P38 skip-start-wrong-item** | step0 url_before 含 sCategory AND url_after 含 page=item AND steps≤5 AND string_match | agent-limit/FP | 中 | T16 (no-hit 盲区) |
| **P39 false-success-unflagged** | program_html + isolated_context_used AND effective_mutating_action_count=0 → flag 疑似 false-success | benchmark-FP | **高** | T5 — 检测未 flag 的 program_html 假阳 |

> P34/P35 与现有 P33/P14 部分重叠 (P33 着陆 image 页, P34 连续困死; P35 是 P14 自环的 scaffold-诱发子型) —— 落码时须 verify 不双重计数 + 加 success-safe 条件。

**P33 docstring 更正 (非新规则, 可立即改)**：`diag_pattern_match.py` P33 docstring + SKILL.md 把"phantom_som 特有 / phantom_som 结构诱发"改为"text-bearing modes (dom AXTree img-href / phantom_som SOM_MARKS img-href) 共有的结构同构幻觉"。

---

## 7. Actionable

| 项 | 类型 | 动作 | 优先级 |
|---|---|---|---|
| **B-260 follow-up (Magento gap)** | scaffold-bug | reopen/补注 B-260：canonical `locator.fill()` 在 Magento(7770) 搜索框 **未阻止累积** (反驳 "data unaffected")。复核 `dispatch_id_based_type` bbox-center walk 在 Magento autocomplete 搜索框上是否命中正确 input + fill 清空时序。**Phase 1b shop fire 前必修 + 须改 paper §3.5 "data unaffected" 表述加 shop scope caveat** | **高** (blocks shop) |
| P33 docstring | infra | 更正 `diag_pattern_match.py` + `SKILL.md` 的 P33 cross-mode 描述 | 低 (立即可做) |
| `raw_output` 入 step_record | infra | `qwen3vl_agent.py step()` 存 `meta["raw_output"]` 便于跨族格式审计 | 低 |
| B2 SR 报数 | paper | paper 报 B2 dom cls SR 须用 **legit-corrected 0.9-1.3%**，标注 raw 2.2% 含 2 个未 flag false-success | 中 |
| P34-P39 落码 | diag self-evolve | 批量 bump RULESET + 全量重扫 (与 B2 其它 mode 数据齐后一并做) | 中 (defer) |
| cross-mode 验证 | diag | B2 som/vision 跑完后验"P33/P6/P16 视觉类规则 vs vision grounding"分层 | defer |

---

## 8. paper-grade 洞察

1. ✅ **Gemma3-VL 跨族 4B 在 dom 几乎不可用 (legit SR ~1%)** — 不是 scaffold 不兼容（98% parse OK），是真实能力鸿沟。这是 cross-family matched-capability (4B 对齐 B1) 的**有效极端数据点**，支持 paper 的 representation×capability 交互叙事（B1 Qwen-4B vs B2 Gemma-4B 同量级行为差异）。
2. ✅ **Gemma 行为签名区别于 Qwen**：从不 finish (23/224) + goto-search 不收敛 + image-URL 幻觉 + cross-site 不跨站 —— 这些是 **model-specific agent-limit**，换 mode 救不了（behavioral 非 representation）→ 倾向 **需 module (finish-forcing / retry / memory) 非 routing**。但须 B2 多 mode 数据才能定 cross-mode 分层。
3. ⚠️ **diag 在 cls 数据里提前捞到 shop 的 scaffold 雷 (B-260 Magento gap)** — Phase 1b shop fire 前的关键 pre-flight 收获，且**反驳了 B-260 "paper-grade data unaffected" 的断言对 shop 成立**。
4. ⚠️ **deterministic forensic > log-only AI 归因** 再次验证 (type-累积翻案)。Tier-2 sub-agent 是深挖器非裁判，父 context 对 estimand-adjacent / scaffold-vs-limit 判定须独立 forensic 复核。

---

*生成 2026-06-09 · Tier-1 `diag_pattern_match.py` (ruleset `5-domsomvispsom-b1860coord`) + Tier-2 4×sonnet sub-agent (27 ep) + 父 opus forensic 复核 · /diag B2 dom*
