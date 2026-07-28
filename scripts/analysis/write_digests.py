#!/usr/bin/env python3
"""Write the 15 outstanding reddit /diag digests.

Tier-1 numbers come from gen_diag_digest.build() so every digest quotes one
source. Tier-2 prose is supplied per condition below; conditions whose Tier-2
has not been run say so explicitly rather than implying coverage.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gen_diag_digest import build, tier1_section, RUNS, OUT

DATE = "2026-07-27"
RULESET = "8-reddit-p41p46-b1890fix"

# Tier-2 findings, per condition. None = not deep-dived this round.
T2 = {
"B2_dom": dict(
  scope="7 ep（no-hit 全 3 + P36 因果审计 4）· 1 sonnet sub-agent",
  counts="agent-limit 7 · scaffold-bug 0 · benchmark-FP 0 · unclear 0",
  p36="**真死因**。walk_fail 信号真实准确 —— 被点的 element_id 确实存在于 observation（有真实 union_bound，非幻觉 ID），但确实不是可操作祖先（StaticText / 用 click 操作纯文本 input）。`locator_dispatch.py` 的 6 层 walk-up + ARIA 白名单按设计工作。致命的是模型**完全不响应** prompt 里 8-step 历史窗口中连续出现的 FAILED 反馈。walk_fail 后 episode **从不恢复**：task 145/100/40 是 100% 死锁，逐字重复同一 `(action_type, element_id)` 达 28–29 次直到预算耗尽。",
  findings=[
    "**task 145 / 100** — 对同一 StaticText 元素连续点击 29 / 28 次，`walk_fail:no_actionable_within_walk` 100% 复现，thought 逐字不变。",
    "**task 40** — 模型正确说出「要在搜索框搜 Pittsburgh」的计划，却连续 29 步用 `click` 而非 `type` 作用于搜索框；CLICK 解析器按设计不接受纯文本输入框 → 必然 walk_fail。计划对、动词错。",
    "**task 72** — DOM-only 无图，模型却臆造出具体视觉描述（「一只猫睡在沙发上」）并反复自我强化，写入文本框的也是幻觉内容。比「答不出」更危险的失败模式。",
    "**task 64 / 101 / 171（no-hit）** — 三种不同形态：世界知识错误+无谓导航 / 任务语义误解+过早放弃 / 精细定位失败+从未完成动作。共同点是「看起来在推进」（URL 或 DOM 有变化）从而躲过现有 no_progress 类触发器。",
  ],
  why="三层叠加，且规模效应明显（4B vs 235B）：(a) **无自我修正** —— 明知连续 FAILED 仍逐字重复直到 budget 耗尽（4/7 episode，30 步里 27–29 步纯浪费），这是 P36 密度达 B0 4.5× 的直接机制；(b) **动作空间语义混淆** —— 把 StaticText 当可点击、用 click 操作文本框；(c) **DOM-only 下的视觉幻觉**。B0/B1 在相同陷阱前也会失手一两步，但能靠更强的指令遵循跳出死循环，不会把整条预算耗在同一个死点。",
  rules=["`IDENTICAL_FAILED_ACTION_STREAK>=3`（同一 action_type+element_id 连续失败 ≥3 次即判 P36）—— 不是抓新模式，而是把现有 P36 的判定时机从 ~27 次**大幅提前**，可将此类 episode 的浪费步数从 ~28 压到 ~3。",
         "`EMPTY_FINISH_ON_MUTATION_TASK`（finish.answer=='' 且 eval 为 program_html/mutation 类）",
         "`NO_SEARCH_ATTEMPTED_FOR_PAGE_IMAGE_QUERY`（eval_types 含 page_image_query 且全轨迹 type 次数=0）"],
),
"B2_som": dict(
  scope="10 ep（no-hit 7 + success 全 3）· 1 sonnet sub-agent",
  counts="agent-limit 7 · **benchmark-FP 3（即本 condition 仅有的 3 个 success 全部可疑）** · scaffold-bug 0 · unclear 0",
  p36=None,
  findings=[
    "**3 个 success 无一被证实为真**：task 130（全程从未确认点中 Subscribe，在 /f/memes ↔ /forums 间震荡耗尽 30 步）· task 160（must_exclude-only eval，见 B-1889）· task 170（唯一一次语义正确的 Subscribe 点击 `action_success=false`，此后再无第二次尝试）。三者共同点：**全部 `agent_finished=false`**。",
    "⚠️ **跨 episode 状态泄漏嫌疑**：`require_reset` 在 reddit 上是 no-op（`external/visualwebarena/browser_env/envs.py:172` 只有 classifieds 分支真 reset），205 个 episode 顺序跑下来订阅列表是累积态 —— 对「检查 sidebar 订阅状态」这类 eval 构成泄漏风险。task 130 的 'memes' / task 170 的 'sports' 都可能是前序 episode 的残留。**本轮未独立证实，列为待查**。",
    "**task 1 / 13 / 153** — reference-image→目标帖子的视觉匹配失败：选中语义无关的 forum（AskReddit 而非 Newark、personalfinance 而非 dataisbeautiful）。",
    "**task 89 / 104** — 把「发评论」任务当纯文本 QA：1 步 finish 交答案，零 type 动作，评论从未真正发布。",
    "**task 113** — create-post 多图任务完全跑偏，30 步 thought 从未出现任何目标 forum 名或 create/post 关键词。",
  ],
  why="(a) 视觉-参考图匹配能力显著弱于 B0/B1；(b) 状态自我监控缺陷 —— 对同一 element_id 连点 5–7 次却意识不到在原地打转；(c) 长 instruction + 多模态下容易丢失任务目标。**SoM 的标注图对 Gemma3 未构成明显帮助** —— `element_id_coerced_from_string` 几乎每步出现（模型常把 element_id 输出成字符串需 runner 强转），点击失败率高，提示它对「编号↔元素」的对应关系把握也不稳。",
  rules=["`success==true AND agent_finished==false` → 强制人工复核，不直接计入 SR（本 condition 3/3 命中，全场性价比最高）",
         "`program_html.required_contents` 只含 must_exclude → 标记「可被纯 inaction 平凡满足」（→ B-1889）",
         "`intent` 含 'leave a comment' 且全 episode type 次数=0 → `premature-finish-no-comment-post`",
         "同一 element_id 连点 ≥5 次且 url 不变（不论 content_changed 真假）→ `stuck-click-loop`"],
),
"B2_vision": dict(
  scope="8 ep（no-hit 7 + success 1）· 1 sonnet sub-agent",
  counts="agent-limit 7 · benchmark-FP 1（task 160）· scaffold-bug 0 · unclear 0",
  p36=None,
  findings=[
    "**主导失败模式是「动作模态错误」，比 grounding/perception 更根本**：5 个要求「发一条真实评论」的任务（69/76/89/91/103）里，模型从未使用 `type`，一律用 `finish(answer=...)` 把答案当文字描述交上去。**task 103 的视觉判断完全正确**（'blue' 与 reference 字面精确一致），但答案没写进评论框，评测读站点内容时依然判失败。",
    "**坐标映射类 scaffold bug 已排除**：全部 8 个 episode `image_meta_recorded=True`、`input_image_tokens=768`（截图确实送进模型）；click 的 `coordinate_normalization` 全部 `recovered=true / true_oob=false / malformed=false`，无系统性错位。点击命中的是「错的」元素（帖子缩略图本身指向裸图文件），属语义级选错目标。",
    "**submission_images URL 陷阱**：点击帖内缩略图直接跳到 `/submission_images/*.jpg` 裸图页，丢失评论框上下文。vision mode 无语义标签只能靠坐标猜，比 dom/som 更容易踩中（站点结构 + 模型选择的复合问题，非 runner bug）。",
    "**task 160 = benchmark-FP** → B-1889。",
  ],
  why="vision mode（只有坐标、没有 element_id）下模型把「评论/发帖类操作任务」系统性误解成「纯 QA 问答任务」，跨 5/7 个 no-hit episode 可复现，比单个 perception 误判（数错 Jupiter 数量、少读一个零）更具规律性。",
  rules=["`never-posted-comment`：eval locator 含 `reddit_get_latest_comment_content_by_username` 类函数 且全程 `type` 成功次数=0 → 把「答案生成了但从未提交」从「感知错误」里精确剥离",
         "`success-via-inaction`：success 且 agent_finished=false 且全程 obs_url 唯一值=1（=start_url）→ 强制复核"],
),
"B2_phantom_text": dict(
  scope="6 ep（no-hit 3 + P36 因果审计 3）· 1 sonnet sub-agent",
  counts="agent-limit 6 · scaffold-bug 0（但发现一个真实检测缺陷，见下）· benchmark-FP 0 · unclear 0",
  p36="**真死因**。103/12/205 三条轨迹在触发 P36 后被完全钉死：对着已证实点不动的 element_id 逐字重复 20–28 次，耗尽全部预算，从未尝试换元素 / 滚动 / go-back / 改 URL。但**背后机制是 agent-limit**：element_id 均在当步 mark_count 范围内（非幻觉出界 ID）；task 205 甚至**已经成功到达目标页面**却仍在重复「我需要导航过去」的过时推理。",
  findings=[
    "🐛 **发现一个真实 scaffold 缺陷（非本 episode 根因，但值得单独修）**：task 103 的 `state_change_reason_distribution` 显示 `scroll_changed:29` —— **纯滚动位移被计入 `page_changed=True`**，导致 `no_progress_streak` / loop-trigger 全程哑火（`trigger_distribution={}` 空）。建议：`state_change_reason` 集合若只含 `scroll_changed` 不应计入 page_changed。否则这类「滚动但零实质进展」的 case 会继续被系统性漏记，影响一切基于 trigger 计数的分析。",
    "**task 104** — 把 'Notifications' 链接误判成通往论坛的路径，此后连续 8 次原样重发同一 click，url 恒为 /notifications。",
    "**task 179** — 图片识别正确（Missouri）、search 也成功，但陷入两段死循环：对真实 `<a>` 标签连点 7 次 url 不变；在 /forums 与 /search 间横跳 15+ 步，从未点击指向目标 forum 的链接。",
    "**task 12** — 从未表现出对参考图的任何识别，直接搜字面词 'Comments'，落在 NYC 版面（非目标 Pittsburgh），随后连点 26 次。",
  ],
  why="与 B2 其他 mode 同源：perseveration + 内部状态不随观测更新。task 205 是最清晰的例证 —— 已在目标页 20+ 步仍逐字重复「需要导航到该页」。",
  rules=["`IDENTICAL_ACTION_NO_STATE_CHANGE`（连续 ≥5 步 action_type+element_id 相同 且 url_before==url_after）—— 纯字段比较 0 token，命中本轮 4/6 episode",
         "修复 scroll-only 状态变化被误判为 progress（见上，是给现有 trigger 逻辑打补丁而非新规则）",
         "`URL_DIVERSITY_COLLAPSE`（trailing 10 步内 distinct url ≤2 且未 done）—— 覆盖比逐字重复更隐蔽的「两态乒乓」"],
),
"B2_phantom_som": dict(
  scope="8 ep（no-hit 5 + P36 审计 2 + success 1）· 1 sonnet sub-agent",
  counts="agent-limit 7 · scaffold-bug 0 · benchmark-FP 0 · unclear 1（task 58，成功但走捷径）",
  p36="判为**伴随症状而非独立根因** —— 与 B2_dom / B2_phantom_text 两位 sub-agent 的措辞有细微分歧，此处如实并陈：三方对**机制**的描述完全一致（walk_fail 只在 element_id 已在 `obs_nodes_info` 中找到、union_bound 存在之后才触发，从不代表幻觉引用；`_JS_RESOLVE_CLICK` 按设计不接受纯文本 `<input>`，task 181 在 step_13 换成 `type` 后立刻 `target_tag='INPUT'` 成功，直接实锤），分歧只在把 P36 称作「直接死因」还是「执行层症状」。**综合表述：P36 是失败的直接放大机制，根因是模型 perseveration。**",
  findings=[
    "✅ **[SOM_MARKS] 文本与可操作元素集一致，未发现错标** —— `som.py::build_som_text_from_obs_text` 每条 mark 直接取自 AXTree 行（仅去掉 `[N]` 前缀），role 信息（如 'textbox'）本就在文本里，模型有足够线索区分「该 type 还是该 click」。选错动词是纯推理问题。也未命中已知的 P33（点击图片 href 跳裸图页）。**P-SoM 作为 hero mode 在 scaffold 层是干净的** —— 对论文有利，但样本仅 8 例，建议在 B0/B1 同 mode 交叉核对后再写进正文。",
    "**task 181** — 前 13 步对搜索框（经 step_13 证实 `target_tag='INPUT'`）连续误用 click 而非 type；step_14 点了搜索结果跳出沙盒到真实站点 wfsb.com，后续在外站 DOM 上继续大量 walk_fail。",
    "**task 73** — 把「描述计划」当「执行计划」：step_0 的 thought 说 'I will search for...'，同一步直接输出 `finish`，episode 在 0 次真实导航后终止。",
    "**task 58（success，判 unclear）** — string_match 'Reki Kawahara' 精确匹配，判定本身无误；但 21 步全程只在 reddit 内打转，**从未访问任务要求的第二站点 wikipedia（localhost:8888）**，答案很可能来自模型参数知识而非页面取证。不影响 success 判定，但值得作为诚实性附注。",
  ],
  why="与 B2 其他 mode 同源（perseveration + 视觉语义映射弱）。P-SoM 特有的是**无图像通道**：task 89 即使成功导航到图片 URL 页面，`input_image` 仍为 0 token → 该任务在此 mode 下**结构性不可解**，不应记为 Gemma3 的能力弱点。",
  rules=["`MULTI_SITE_TASK_SINGLE_SITE_GROUNDING`（task.sites >1 但轨迹 obs_url 只覆盖 1 个站点）→ 标记「疑似参数知识捷径成功」，**直接关系 SR 数字的诚实性，且不只影响 P-SoM**",
         "`PHANTOM_IMAGE_BLIND`（全 episode input_image tokens==0 且任务本质需要看图）→ 把结构性不可解的任务从「模型能力不足」里摘出单独统计",
         "`STUCK_REPEAT_VALID_CLICK`（同一元组连续 ≥3 步、locator success=true 但 page_changed=false）—— 与 P36 walk_fail 型循环互补"],
),
"B2_phantom_prompt": dict(
  scope="6 ep（no-hit 1 + P36 审计 4 + 唯一 success 1）· 1 sonnet sub-agent",
  counts="agent-limit 5 · **benchmark-FP 1（唯一的 success）** · scaffold-bug 0 · unclear 0",
  p36="**真死因**。4 个抽样 episode 全部同一死法：命中一次 walk_fail 后连续 27–30 步原样重复，占该 episode 全部预算的 90–100%。sub-agent 另做了全 205 集结构扫描：**160/205 集（78%）至少命中一次 walk_fail，20/205 集单集内 ≥20/30 步被同一失败点击霸占**，总计 1458 次 step 级 walk_fail（与 Tier-1 的 1450 吻合）。",
  findings=[
    "⭐ **SR=0.49% 是真实能力崩溃，不是测量故障** —— 关键证据是**跨 baseline 的严格单调梯度**：B0(235B) 12.68% → B1(Qwen3-4B) 6.34% → B2(Gemma3-4B) 0.49%。若是 harness/infra 故障，三个 baseline 应**同等程度**失灵，而不是随模型规模/家族精确分级。token/延迟/cost 记账均正常，无 error 字段、无 auth 失败痕迹。",
    "⭐ **唯一那个 success（task 160）不可信** → B-1889，且该 task 已由 **AMENDMENT_08 tier A** 正式移出计分集。所以 **0/203 = 0.00%** 不再是本 digest 的事后修正，而就是 `sr_per_mode.json` 里的权威数字。**旧写法「0/205」把 scored rate 算在 collected 分母上** (B-1913)。",
    "**P-prompt 的 SoM-prompt × AXTree-text 组合是设计固有、非实现 bug**：代码确认（`_shared_vl_utils.py::build_mode_prompt_dispatch_table` + `som.py::prepare_observation_for_mode`）phantom_prompt 明确路由到 SoM system prompt + AXTree 原生文本 + 无图，element_id 用的是与 dom 模式**完全相同**的原生 AXTree id（`mark_count=0`，未走 seq 映射）→ walk_fail 与「提示-观测错配」正交，不是 ID 体系混乱导致点了不存在的编号。可归因于该刻意错配的是两个**间接**效应：(a) 无图像通道 → 失去独立视觉线索去发现自己卡死；(b) SoM prompt 反复宣称「你会收到标注截图」而实际没有，可能侵蚀 grounding 校准 —— 所有卡死点击的 confidence 都标 0.95（虚假自信）。**建议作为跨家族鲁棒性差异的证据写进分析，不建议改 harness。**",
  ],
  why="见上：perseveration 是主因，phantom_prompt 的无图像通道 + prompt/观测刻意错配放大了它。",
  rules=["把 `P35(MUTATION_MISSING)` 泛化为 `PASSIVE_MUST_EXCLUDE_FP`（去掉 `agent_finished==True` 与 locator 白名单限定）—— 当前 P35 恰好漏掉 task 160 这类 sidebar 场景。⚠️ 实现时**不要**用 `effective_mutating_action_count` 做判据（B-1890：该字段恒为 0）",
         "`P36-fatal`：同一 (action_type, element_id) 的 walk_fail 连续占满几乎整个预算 → 与「偶发可自愈」型 walk_fail 区分开，对路由信号设计也有用"],
),

# ---- B1 (Qwen3-VL-4B) reddit, Tier-2 2026-07-27 ----
"B1_dom_reddit": dict(
  scope="7 ep（no-hit 分层抽样 5 + success 审计 2）· 1 sonnet sub-agent",
  counts="agent-limit 5 · benchmark-FP 2（两个 success 均判 FP）· scaffold-bug 0 · unclear 0",
  p36="B1 **也有** perseveration，但形态是「谱系」而非 B2 那种单一死循环：task 160 在 step 3/4/5/6 连续 4 次同一 `walk_fail:no_actionable_within_walk`（同 element_id=3949），step 15/16 复发 2 次；task 114 则是语义级循环（连续 17/23 步在三个 forum 名的字面搜索间打转），最终靠切换到直接 URL 导航跳出，代价是耗掉 74% 预算。即「locator 层刚性重复（未能自纠）」到「策略层松散重复但最终自纠」的连续谱。",
  findings=[
    "⚠️ **P36 计数被系统性低估**：task 160 真实发生了 6 次 walk_fail，但因该 episode `success=true`、而 P36 对 success episode 直接 `return []`，这些完全没进 Tier-1 统计。（v8 未改此行为——success-safe 是刻意设计，但读 P36 数字时要知道它只覆盖 failed 侧。）",
    "**task 91 / 95 / 102** — dom 模式 `input_image=0`，任务要读帖子配图的颜色/计数。task 102 诚实认输，**task 95 则在零视觉输入下自信幻觉**（thought 称「I can see the snow... appears white」，真值 purple/pink，confidence 0.95）。",
    "**task 138** — 正确从参考图提取姓名 Patrick、正确导航到 account 页、正确输入用户名，**但直接 finish 未点任何 Save/提交**，修改未持久化。这是「差最后一步」类失败。",
    "**task 58 / 160** — 两个 success 均判 benchmark-FP（→ B-1892 / B-1889）。",
  ],
  why="B1 在「放弃」与「固执」之间偏向**过早放弃**（多个 episode 1-2 步内 confidence=0.0 直接 finish），而 B2 偏向**过度固执**。量化对照：B1_som 188 failed 中 P36 命中 54.8% / P31 命中 67.0%；B2_som 202 failed 中 64.9% / 83.7% —— 两项 B2 都显著更高。",
  rules=["`P-unsaved-form`（最后一个非 finish 动作是 type 表单字段，其后无提交类 click 即 finish，且 eval 要求字段持久化）—— 命中 task 138 这类「差最后一步」",
         "P27 `ABANDONMENT_RE` 扩充 'unable to determine' + 同时扫 `thought` 字段（现仅扫 answer/text）"],
),
"B1_som_reddit": dict(
  scope="7 ep（no-hit 5 + success 审计 2）· 1 sonnet sub-agent",
  counts="agent-limit 5 · **benchmark-FP 2（两个 success 全部可疑）** · scaffold-bug 0 · unclear 0",
  p36=None,
  findings=[
    "✅ **标注图确实送达模型**：每步 `image_meta` 的 `input_image` token 数非零（576 或 1344），`som.enabled=true`，`mark_count` 在 2–136 间正常变化 —— 机制层面没坏。",
    "**但抽样的 5 个 no-hit 没有一个是「看错/点错标注框」** —— 全部败在上游的视觉推理 / 计数 / OCR / 指令理解。",
    "**task 132** — 用真实 reddit 的 `/r/<sub>` 路径规范（本站应为 `/f/<sub>`）反复 goto，造成 4 次同构 404 循环。这是**预训练先验污染站内导航**的清晰例子。",
    "**task 175 / 203** — 「过早放弃」型：1-2 步内 confidence=0.0 直接 finish。task 203 的放弃措辞只写在 `thought` 里、`answer` 为空字符串，因此 P27 完全看不到。",
    "**task 58** 触发 P25 且判成功 → 本 digest 首次提出该跨站捷径疑点，后经跨 18-cell 复核确立为 **B-1892**。",
  ],
  why="标注图对 B1 的净增益边际：**som 7.39% (15/203) vs dom 5.91% (12/203)**，差 1.48pp = 3 个 task，n=203 下是噪声量级。真正撑住 B1 表现的是 **DOM/AXTree 文本本身** —— B1_vision（无文本纯截图）只有 **2.46% (5/203)**，远低于所有含文本的 mode。（B-1913：旧写法「扣除 task 160 后 som 7.80% vs dom 6.34%（17 vs 14 个成功，n=205）」有三处错 —— 分子扣了 task 160 而分母仍是 205；括号里的计数 17/14 与百分比 7.80/6.34（=16/205、13/205）自相矛盾；且 AMENDMENT_08 已把 160 移出计分集，无需再手工扣。数字一律以 `sr_per_mode.json` 为准。）",
  rules=["P27 `ABANDONMENT_RE` 加 'unable to determine' + 扫 `thought`（本 condition 5 个 no-hit 中 2 个因此漏检）",
         "`EMPTY_ANSWER_SURRENDER`（finish 且 answer=='' 且 confidence==0.0）",
         "`REAL_REDDIT_PATH_HALLUCINATION`（goto url 匹配 `/r/<name>` ≥2 次）"],
),
"B1_vision_reddit": dict(
  scope="6 ep（no-hit 5 + success 1）· 1 sonnet sub-agent + 全 condition 扩展扫描",
  counts="agent-limit 5 · benchmark-FP 1 · scaffold-bug 0 · unclear 0",
  p36=None,
  findings=[
    "⭐ **主导失败是「动作模态错误」而非 grounding 或 perception**：要求发真实评论的任务里，模型从不用 `type`，一律用 `finish(answer=...)` 把答案当文字描述交上去。**task 103 的视觉判断完全正确**（'blue' 与 reference 字面一致）却仍判失败 —— 答案没写进评论框。→ 这条观察催生了 **P46**。",
    "✅ **坐标映射无 scaffold bug**：全 condition 2386 个带坐标动作全量扫描，`x_regime`/`y_regime` 全为 `qwen_0_1000`，**0 例 `true_oob`、0 例 `malformed`**，仅 1 例 `dead_zone` 且仍 `recovered=true`。问题在「点哪」不在「点到哪去了」。",
    "**submission_images 陷阱**：点帖子缩略图直接跳裸图页（缩略图 href 就是图片文件本身）。vision 无语义标签只能靠坐标猜，比 dom/som 更易踩中。→ 这条催生了 **P33 的 reddit 路径扩展**。",
  ],
  why="动作模态错误 + 语义级选错目标的复合体，且 reddit 站本身评论/发帖类任务占比高，使 vision 在缺少文本结构辅助定位时被放大打击 —— 比 dom (6.83%) 低一半以上。",
  rules=["→ 已落码为 **P46**（COMMENT_INTENT_NO_TYPE）与 **P33 reddit 路径扩展**"],
),
"B1_phantom_text_reddit": dict(
  scope="7 ep（no-hit 5 + success 2）· 1 sonnet sub-agent + 全 run 扫描",
  counts="agent-limit 5 · benchmark-FP 2 · scaffold-bug 0（但复核出一个真实检测缺陷，见下）",
  p36=None,
  findings=[
    "🐛 **scroll-only 状态变化确实存在于 B1**：全 run 5226 步中 23 步的 `page_change_reasons` 恰好只含 `scroll_changed`，且全部仍记 `page_changed=True`。但**它不是这些 episode 的主因** —— 我复核后确立的真根因是 `action_success` 语义脱节（→ **B-1891**）：`no_progress_streak` 由 `prev_action_success` 驱动而非 `page_changed`，两个 trigger 是被**各自独立**压制的。",
    "**4/5 no-hit 是指令-观测错配**：任务显式或隐式要求看图，而 phantom_text 剥离页面截图。分两种子模式 —— 纯页面内嵌图（零信息，task 104 高置信度编造「0 kirbies」）vs 任务级参考图可见但页面帖子图不可见（task 133，能看懂参考图却无法在页面里核对是哪个帖子）。",
    "**task 104 的高置信度幻觉值得单独记**：模型**不知道自己看不见**，会把无信息状态包装成「已观察」的确定性陈述。",
  ],
  why="见上：指令-观测错配为主，叠加 perseveration。",
  rules=["→ 已落码为 **P43**（但按 §387.10 的受控对比结果改成了**中性标签**，不是 sub-agent 提议的「结构性不可解」）",
         "B-1891 的修复（`action_success` 语义）属 runner 层，未在本批规则内"],
),
"B1_phantom_som_reddit": dict(
  scope="7 ep（no-hit 5 + success 2）· 1 sonnet sub-agent（首次因 session limit 中断，已重放）+ 我的独立全量复算",
  counts="agent-limit 5 · benchmark-FP 2 · scaffold-bug 0 · unclear 0",
  p36="见下方两层核查 —— 结论是 walk_fail **既非 P-SoM 特有也不随能力单调**。",
  findings=[
    "⭐ **[SOM_MARKS] 两层核查（这是 hero mode 能否宣称 scaffold 干净的关键证据）**。我在 sub-agent 数字基础上补了 dom/som 对照，结果比原报告强得多：",
    # 🚨 2026-07-28: 这条 P-SoM vs dom 的比值 **不可比**, 不要再往 paper 搬。
    # 指标 = "element_id 不在 obs_nodes_info 里"。dom 的 map 用原生 CDP nodeId (稀疏,
    # median 7839-18729, max 691695), 而 P-SoM 被 build_som_text_from_obs_text 重键成
    # 1..K (稠密, median K=15-17, max 176)。稀疏空间下几乎任何笔误都落在有效集外 → 计数;
    # 稠密空间下**选错元素**通常仍命中有效 id → 不计数。所以 "P-SoM 干净 N×" 混了行为差异
    # 与**检测灵敏度**差异。paper A §4.2 已改成只报同 namespace 的两个 prompt 对比
    # (dom↔P-prompt / P-text↔P-SoM), 见 aggregate_cross_mode_failure_signatures.py。
    "  · **(a) 幻觉引用率**（引用了 observation 里不存在的 element_id）：P-SoM **B0 0.04% / B1 0.12% / B2 7.84%**；同 model 的 dom 是 **0.39% / 2.98% / 18.21%**。⚠️ **跨 id-namespace 不可比, 见上方注释**。→ **dom 在每个模型上都最差，P-SoM 干净 2.3–24.8×**（B0 9.75× / B1 24.83× / B2 2.32×；2026-07-27 /stress Mode A A-5 修正：旧文字写 “5–25×”，下界与自己列的 B2 一对数字矛盾，且 paper A §4.2 已照抄继承）。机制：dom 用原生 AXTree id（median 7839–18729，max 691695），P-SoM 用紧凑编号 1..N（median 15–17，max 176）—— 抄 5-6 位稀疏整数 vs 2-3 位紧凑编号。→ 这条催生了 **P44**。",
    "  · **(b) walk_fail 率**：P-SoM B0 13.3% / B1 29.5% / B2 21.9%；dom 23.5% / 18.5% / 35.2%。**3 个模型里 2 个是 dom 更差**，且不随能力单调 → 在 (model, mode) 格间就是噪声，**不能写成「walk 可执行性随能力劣化」**。",
    "⚠️ **同时修正了 4 个 sub-agent 的集体误判**：它们都断言 `obs_nodes_info missing union_bound`（幻觉引用分支）「一次都没出现」。在各自 6–8 个样本里成立，**总体上不成立**（B2 上 374 次 psom / 895 次 dom）。walk_fail 与幻觉引用是**并存**的两条分支。",
    "**task 19** — 点 [SOM_MARKS] 里的 img href 跳到 `/submission_images/*.jpg` 裸图页（reddit 版 P33），旧正则漏检。",
  ],
  why="P-SoM 的失败集中在「无页面截图 → 无法把参考图与页面缩略图做比对」（task 19/139），以及与其他 mode 共通的 perseveration。**scaffold 层在 element-引用维度不仅干净，而且优于 dom。**",
  rules=["→ 已落码为 **P44**（HALLUCINATED_ELEMENT_REF，此前零覆盖）与 **P33 reddit 路径扩展**"],
),
"B1_phantom_prompt_reddit": dict(
  scope="7 ep（no-hit 5 + success 2）· 1 sonnet sub-agent",
  counts="agent-limit 5 · benchmark-FP 2 · scaffold-bug 0 · unclear 0",
  p36="B1 在同一构造下也有 perseveration（task 142 连续 12 步猜不同 element_id），但**恢复能力明显更强** —— 最终靠 scroll 找回评论框并完成提交动作。",
  findings=[
    "⭐ **同一 P-prompt 构造下 B1 与 B2 的差异，正是「SR 梯度反映能力而非构造缺陷」的直接证据**。代码层确认两者面对**完全相同**的构造（`mark_count=0`、element_id 用原生 AXTree id、SoM prompt 仍宣称会给标注截图但从不发图）。差异在应对：(a) **B1 校准更诚实** —— task 152 直接给 confidence=0.0 并拒答，不像 B2 那样固定虚高 0.95；(b) **B1 会恢复** —— task 142 在 12 次误点后自行脱困。",
    "**幻觉措辞要分两类，不能混为一谈**：真幻觉（task 152 逐字出现「no image is visible in **the provided screenshot**」，而该 mode 从未提供 screenshot）vs 术语混用（task 132/138 说「the image」实指**真实存在**的任务级参考图 —— 参考图所有 mode 都发，不算幻觉）。",
    "**比幻觉更危险的模式**：task 142 编造具体日期、task 58 编造「评论里写着」的假引用来源，且配 0.95–1.0 高置信度。与 B2 的核心风险同质，只是发生率低得多。",
  ],
  why="构造缺陷是共同的，能力决定了伤害大小 —— 这正是 B0 12.68% > B1 6.34% > B2 0.49% 梯度的解释。",
  rules=["→ 已落码为 **P43**（中性标签版）"],
),
# ---- B0 (Qwen3-VL-235B-A22B) reddit phantom 系, Tier-2 2026-07-27 ----
"B0_phantom_text_reddit": dict(
  scope="7 ep（no-hit 5 + success 审计 2）· 1 sonnet sub-agent",
  counts="agent-limit 5 · scaffold-bug 1（P39 误报，见下）· benchmark-FP 1 · unclear 0",
  p36=None,
  findings=[
    "⭐ **B0 的失败形态与 B2 本质不同**：对「零信号」任务，B0 **快速优雅放弃**（task 120 仅 1 步就诚实承认无法判断）；即使编造错误答案（task 147/149）也在 5–7 步内干净收场，不做无意义重试。但 B0 **确实会**在「UI 反馈模糊」时短程重复（task 41 连续 17 步、task 129 连续 10 步点同一 toggle），关键区别是**规模减半**（10–17 步 vs B2 的 20–30）**且会自我打断**（task 129 第 11 步出现元推理「the button says Unsubscribe... I will assume... finish」主动跳出）。",
    "⭐ **4/5 no-hit 是「表征而非能力」的失败**（task 41/120/147/149）：所需信息在 phantom_text 的文本 substrate 里根本不存在。其中 2/5 命中同一个 `intent_template_id=60`（「数图中 X 数量」）→ **系统性任务族缺陷而非零散噪声**。只有 1/5（task 129）是即使有图也答错的语义粒度错误。⚠️ 但注意 §387.10 的受控对比显示，给这类任务补上截图的实测增益 ≈0 —— 所以「表征失败」不等于「换 mode 就能救」。",
    "🐛 **task 19 的 P39 命中是假警报** → 直接催生 **B-1890 的规则层修复**：P39 判据 `effective_mutating_action_count` 恒为 0，而逐步核查显示 step 2 有一次真实生效的点赞，且 eval 用 isolated context 直接查服务端状态。**v8 已把 P35/P39 改为从 step record 派生突变计数**，本 condition 的 P39 命中在 v8 下已消失。",
  ],
  why="B0 遇到结构性缺图任务时「快速合理化猜测后主动止损」，而非「卡死重复直到预算耗尽」。",
  rules=["→ 已落码：**P39/P35 的 B-1890 修复**、**P43**（中性标签版）"],
),
"B0_phantom_som_reddit": dict(
  scope="6 ep（no-hit 5 + success 1）· 1 sonnet sub-agent + 全 condition 0-token 结构扫描",
  counts="agent-limit 5 · benchmark-FP 1 · scaffold-bug 0 · unclear 0",
  p36="见 B1_phantom_som_reddit 的两层核查表 —— B0 是其中的干净端（幻觉 0.04% / walk_fail 13.3%）。",
  findings=[
    "⭐ **[SOM_MARKS] 一致性必须分两层说，不能一句「一致」带过**（这条方法论提醒来自本 agent，很到位）：",
    "  · **存在性层：非常干净** —— 2796 个带 element_id 的 action 里仅 1 例越界（0.036%）。**这一层可以放心写进论文正文。**",
    "  · **可执行性层：不能说零** —— walk_fail 覆盖 88/205 episode、356/4669 step（7.6%），其中 304 步最终失败。建议写法：可写「[SOM_MARKS] 编号幻觉率 <0.1%」，但**不要**无保留地写「零列了点不动」。",
    "**task 82 / 202** — 「多目标任务提前收工」：eval 要求 8 个 / 11 个不同目标，agent 只碰了 1 个就 finish 并自称全部完成。",
    "**task 120** — 严格结构性不可解：start_url 本身就是裸图片、无参考图、DOM 无内容。",
  ],
  why="P-SoM 在 B0 上的失败以「参考图↔页面缩略图无法比对」和「多目标覆盖不全」为主，scaffold 层干净。",
  rules=["`SINGLE_TARGET_FINISH_ON_MULTI_TARGET_TASK`（eval must_include 含 N>1 个实体但轨迹交互的 distinct target < N 即 finish）—— mode-agnostic，本批**未落码**，留待下一轮（需要实体抽取，非纯字段比较）"],
),
"B0_phantom_prompt_reddit": dict(
  scope="6 ep（no-hit 5 + success 1）· 1 sonnet sub-agent + 全 48 no-hit 的 task-config 级扫描",
  counts="agent-limit 5 · unclear 1 · scaffold-bug 0 · benchmark-FP 0",
  p36=None,
  findings=[
    "⭐ **本 condition 的 no-hit 是全 18 条里最多的（48/205 = 23.4%）**，扫描显示 **39/48（81%）命中「图像相关」信号**，本次抽样 5/5 全部落在该桶。→ 这是 P43 落码的最直接依据。",
    "⭐ **规则库的结构性偏置**：这类失败「过程干净利落，只是给错了答案」（短 episode、无循环、无预算耗尽、无 URL 自环），**恰好精确避开所有现有 P-rule 的触发条件** —— 现有规则大多是「过程性」病理探测器，而这一整类是「结局性」的。",
    "🔍 **一条重要的代码事实核实**（本 agent 主动查证，纠正了初始假设）：B0 proxy 的 `reference_images` **无视 observation_mode 一律真实发送**（task 109 实测 `image_payload_bytes_ref=172032`），只有**页面实时截图**才受 phantom 约束。→ 这条事实后来收窄了另外 4 个 agent 的「phantom = 完全无图」推断。",
    "**B0 未表现出 B2 的灾难性 perseveration**：5 个 episode 步数 2/13/8/5/23（上限 30），**全部主动 finish，无一跑满预算**。confidence 有起伏（0.7–1.0）而非 B2 的恒定 0.95，但在「盲猜终局」动作上依然普遍偏高。",
  ],
  why="信息在该 mode 的 substrate 里不存在 → B0 快速合理化猜测后止损。这是能力天花板与表征限制的叠加，但 §387.10 显示补图并不能兑现预期增益。",
  rules=["→ 已落码为 **P43**（中性标签版，命名刻意避开 sub-agent 提议的「guaranteed fail」）"],
),
}

BANNER = """> **定位声明**：本 digest 是**单 condition** 的失败归因，其中的 per-rule 分布只描述它自己。
>
> ✅ **discover-then-freeze 已完成**（2026-07-27）：reddit 规则批 P41–P46 + B-1890 修复 + P33
> reddit 路径扩展已落码，`RULESET_VERSION` = `8-reddit-p41p46-b1890fix`，**全部 36 个 canonical
> condition（reddit 18 + cls 18）已在该版本下重扫**，版本一致性由
> `scripts/analysis/diag_rescan_all.py` 校验 → **cross-mode / cross-model 定量聚合现已解锁**。
>
> ⚠️ v7→v8 的 cls 行为**不是**字节不变，差异全部经过定性核实：`P35`/`P39` 的旧命中因
> B-1890 死字段修复而移除（抽查确认那些 episode 确实有 6–8 个突变步，旧命中是错的）；
> `P33` 在 cls 上 +1 例（cls task 233 的 intent 实际要求访问 reddit，旧正则漏检）。
"""

def render(key):
    s = build(key)
    t2 = T2.get(key)
    L = [f"# /diag digest — {s['model']} × `{s['mode']}` × reddit\n",
         f"*生成 {DATE}（Tier-1 全量 + {'Tier-2 深挖' if t2 else 'Tier-2 未深挖'}）*\n",
         BANNER, "", tier1_section(s), ""]

    L.append("## 3. Tier-2 深挖\n")
    if not t2:
        L.append("**本轮未做 Tier-2 深挖。**\n")
        L.append("依 /diag skill 的跨-condition 预算纪律，Tier-2 只投给 (a) SR 异常低 / (b) 新 site-mode / "
                 "(c) no-hit 比例 >25% 的 condition。本 condition 的 SR 落在该 model 的常规区间、"
                 f"no-hit 比例为 {100*s['fn']/s['n']:.1f}%（<25%），故本轮排在 B2 六条之后。\n")
        L.append(f"**待深挖子集已就绪**：failed-NO-hit {s['fn']} 个（见 §2 列表）"
                 f"+ success-with-hits {s['sh']} 个（presence-only 误报审计）。\n")
        L.append("⚠️ 因此本 digest 的三分类**不完整** —— 未深挖不等于「无 scaffold-bug / 无 benchmark-FP」，"
                 "只代表本轮没有查。请勿据此下「pipeline 干净」结论。\n")
    else:
        L.append(f"**覆盖范围**：{t2['scope']}\n")
        L.append(f"**三分类**：{t2['counts']}\n")
        if t2.get("p36"):
            L.append(f"### P36 因果审计\n\n{t2['p36']}\n")
        L.append("### 具体发现\n")
        for f in t2["findings"]: L.append(f"- {f}")
        L.append("")
        L.append(f"### 为什么这个 cell 是 {s['sr']:.2f}%\n\n{t2['why']}\n")

    L.append("## 4. 🔁 Self-evolving — 提议规则\n")
    if t2:
        for r in t2["rules"]: L.append(f"- {r}")
        L.append("\n> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 "
                 "discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` "
                 "并全量重扫，而不是逐条落码逐次重扫。\n")
    else:
        L.append("待 Tier-2 深挖后补。\n")

    L.append("## 5. Actionable\n")
    if s["t160"]:
        L.append(f"- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。"
                 f"若排除，SR {s['sr']:.2f}% → {100*(s['succ']-1)/s['n']:.2f}%。"
                 f"排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。")
    else:
        L.append("- 本 cell 的 success 不含 task 160（B-1889 不影响本 cell 的 SR）。")
    L.append("- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。"
             if t2 else "- scaffold-bug 情况未知（Tier-2 未做）。")
    L.append("")
    L.append("---\n")
    L.append("**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / "
             "B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物\n")
    return "\n".join(L)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    # B0 dom/som/vision reddit 的 digest 是更早手写的 (含各自的 Tier-2 深挖记录),
    # 只在文件里追加 v8 补记, 不由本脚本整体重写 —— 见 --refresh-header。
    HANDWRITTEN = {"B0_dom_reddit", "B0_som_reddit", "B0_vision_reddit"}
    todo = [k for k in RUNS if k.endswith("_reddit") and k not in HANDWRITTEN]
    for k in todo:
        # key already carries the site suffix (`<model>_<mode>_<site>`), which IS the
        # digest basename — do not re-append it.
        p = OUT / f"{k}_diag_digest.md"
        p.write_text(render(k))
        print(f"wrote {p.relative_to(Path('/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents'))}")
    print(f"\n{len(todo)} digests written")
