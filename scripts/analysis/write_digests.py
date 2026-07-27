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
    "⭐ **唯一那个 success（task 160）不可信** → B-1889。**本 cell 修正后真实 SR = 0/205 = 0.00%**。",
    "**P-prompt 的 SoM-prompt × AXTree-text 组合是设计固有、非实现 bug**：代码确认（`_shared_vl_utils.py::build_mode_prompt_dispatch_table` + `som.py::prepare_observation_for_mode`）phantom_prompt 明确路由到 SoM system prompt + AXTree 原生文本 + 无图，element_id 用的是与 dom 模式**完全相同**的原生 AXTree id（`mark_count=0`，未走 seq 映射）→ walk_fail 与「提示-观测错配」正交，不是 ID 体系混乱导致点了不存在的编号。可归因于该刻意错配的是两个**间接**效应：(a) 无图像通道 → 失去独立视觉线索去发现自己卡死；(b) SoM prompt 反复宣称「你会收到标注截图」而实际没有，可能侵蚀 grounding 校准 —— 所有卡死点击的 confidence 都标 0.95（虚假自信）。**建议作为跨家族鲁棒性差异的证据写进分析，不建议改 harness。**",
  ],
  why="见上：perseveration 是主因，phantom_prompt 的无图像通道 + prompt/观测刻意错配放大了它。",
  rules=["把 `P35(MUTATION_MISSING)` 泛化为 `PASSIVE_MUST_EXCLUDE_FP`（去掉 `agent_finished==True` 与 locator 白名单限定）—— 当前 P35 恰好漏掉 task 160 这类 sidebar 场景。⚠️ 实现时**不要**用 `effective_mutating_action_count` 做判据（B-1890：该字段恒为 0）",
         "`P36-fatal`：同一 (action_type, element_id) 的 walk_fail 连续占满几乎整个预算 → 与「偶发可自愈」型 walk_fail 区分开，对路由信号设计也有用"],
),
}

BANNER = """> **定位声明**：本 digest 是**单 condition** 的失败归因，不下 cross-mode / cross-model 结论。
> 跨 mode 定量比较须等 reddit 规则批（R1–R8 + H2）落地、`RULESET_VERSION` 升到 `8-reddit-*`
> 并全量重扫后再做（/diag skill「discover-then-freeze」硬纪律）。
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
    todo = [k for k in RUNS if k not in ("B0_dom", "B0_som", "B0_vision")]
    for k in todo:
        model, mode = k.split("_", 1)
        p = OUT / f"{model}_{mode}_reddit_diag_digest.md"
        p.write_text(render(k))
        print(f"wrote {p.relative_to(Path('/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents'))}")
    print(f"\n{len(todo)} digests written")
