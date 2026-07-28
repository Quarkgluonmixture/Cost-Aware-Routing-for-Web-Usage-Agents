# 裁定层 A1（§5–§119，219 条 ADJUDICATED，2026-04-04 → 2026-05-09）

Claude 主 session 逐条通读产出，2026-07-28。**聚合非转写**：逐条索引见 `ledger.jsonl`。

这一片是**工程建设期 + paper framing 成形期**：scaffold bug 一个个被打掉、FP 体系四轮演化、
phantom space 从一个文献假说长成 paper hook、advisor 5/5 sync 把 paper 拆成四篇。
读这一片的用途 = 「这个设计当初为什么这么定」——**理由比结论重要**，本文件按理由组织。

> ⚠️ **跨批核对提示**：A 批按 type 切分，看不到 RETRACTED（在 B 批）与带数字 MEASURED（D1–D4）。
> 文中标 `⚠️ 待跨批核对` 处，合并阶段须按 § 号回标。

---

## 一、观测模式与 phantom space —— 定义是怎么长出来的

### 1.1 起点：Mirage Effect 不是 bug 是 feature

**当前状态**：Phantom-SoM 的理论根基锚定 Mirage Effect（Asadi et al., arXiv:2603.21687, Stanford,
2026-04）——VLM 无图时仍自信描述视觉特征，无图准确率达有图的 70-80%（mirage-mode > guess-mode）。
三层阶梯 = DOM(Guess-mode) → Phantom-SoM([SOM_MARKS] 文本 + 无图, Mirage-mode, 成本≈DOM)
→ Full SoM（真实多模态），**第一次升级零成本**。

**演化**：§18（04-09）锚定文献 → §23（04-10）**裁定 DOM vs SoM 差距的根因 = Mirage Effect 为主因，
不是文本压缩**（早期"文本压缩说"被此条修正；台账 flag 显示"文本压缩"版本已 RETRACTED）
→ §25（04-10）拆成三层混淆变量 → §102（04-26）phantom_som 落地为第 4 个 mode。

**证据**：§18 / §23 / §102；`docs/literature/phantom_som.md`；commit 59df818；
`p79/experiment/som.py::prepare_observation_for_mode`。

### 1.2 三层混淆变量 + 6 组消融（phantom space 的因果骨架）

**当前状态**：DOM↔SoM 之间存在**三层混淆**：① prompt 坐标 schema ② 文本格式（AXTree 层级树
vs SoM marks 扁平列表）③ 图片暗示。需 6 组消融 A(DOM baseline)/B(Phantom-SoM)/C(Full SoM broken)/
C'(Full SoM occlusion fixed)/D(DOM-text+SoM-prompt)/E(SoM-text+DOM-prompt)。
因果分解式：**D−A=纯 prompt，E−A=纯格式，C'−B=干净图片贡献，C−C'=occlusion cost，B−D−E+A≈mirage**。

**理由**：单一 DOM vs SoM 对比无法把 prompt / 格式 / 图片三个效应分开。

**样本量与统计**（§25 04-10）：A/C 复用 B1 已有数据（234 tasks），B+C'+D(+可能 E) 各 234 tasks；
统计用 paired McNemar test + FDR q=0.05；实现前置需**解耦 observation_mode**（当时是原子选择）
以支持 prompt/obs 交叉组合。

**§25 的自我修订**（04-26，因 §100 probe）：**B vs C 直接对比被 occlusion bug 污染，
不能据此 claim「无图更好」；必须引入 C' 才是干净对比**。依据 = §100 probe 实测 B0 在 reddit_task_6 上
OCR recall 78% → 18%，C 的图本身被破坏。

**证据**：§25 / §100。

### 1.3 SoM 标注范围：Universal 而非 Action-Affordance

**当前状态**：P79 采用 **Universal SoM**（标注所有元素含 StaticText/heading/image，所有元素有 ID，
可 click 任何元素），**不**过滤为 Interactable=True（VWA 原版 = Action-Affordance SoM）。
§101 定性：两者是**质的范式差异不是数量差异** —— P79 = trust model with role info，
VWA = structural constraint。

**理由三条**（§96 04-25）：
1. **路由研究控制变量** —— DOM 的 AXTree 含全部元素，SoM 若只留可交互则 DOM↔SoM 切换
   同时改格式与信息量两个变量，归因不干净；
2. **Phantom-SoM 消融前提（§25）** —— A(DOM) vs E(SoM marks + DOM prompt) 要隔离纯格式效应
   就要求文本内容对等；
3. **role 标签已提供区分信号**（`[id=87] StaticText 'Postmill'` vs `[id=79] link 'Home'`），
   误点非交互元素是模型能力问题不是数据设计问题。

**证据**：§96 / §101；`p79/experiment/som.py`。

### 1.4 SoM 历史设计决策定档（§101 四条）

| 决策 | 时间 | 内容 | 理由 |
|---|---|---|---|
| (a) 全元素不过滤 | §96 04-25 | 保留 Interactable=False 元素 | 控制变量（见 1.3） |
| (b) 标注色 | commit 5f0952b 04-06 | #D0021B 红 → #00BCD4 青 | 消除红框颜色 confound |
| (c) max_marks | §94 04-24 | 80 → 200 | Reddit p99 页面标签截断 |
| (d) placement 算法 | §101 首次显式分析 | simple 2-候选 vs VWA 8-候选 | **此前从未审视** |

**§101 对 P79 vs VWA 原版 SoM 差异的重新归类**（用户审视后修订之前列的 3 个 confound）：
**1 个 fundamental（标注范围）+ 2 个 minor（颜色 + placement）**。理由：数字标签本身是 high salience，
多色不消除 hijack 触发器；placement 只影响 occlusion 程度不改 hijack pattern。

**连带决定**：L4 标注范围 ablation **只做 fundamental 差异**（L4-a: VWA 原版仅 Interactable vs
P79 Universal × reddit B1），**取消 L4-b（颜色 categorical）与 L4-c（placement 8-corner）**。
定性：L4 不是颠覆性实验，是 mechanism robustness 严谨性补强 —— 即便 VWA 原版下反转消失主 claim 仍 robust。

### 1.5 关键代码事实：[SOM_MARKS] = AXTree 的 regex filter

**当前状态**（§103 04-27 定档）：`[SOM_MARKS]` 是 VWA AXTree 文本的 **regex filter**（取所有带 `[N]`
标记的行），**不需要 bbox 提取，不需要 image 处理** —— bbox 只在画 marked image 时才用。
**因此 Phantom-SoM 部署 cost ≈ DOM cost。**

vs SoM 的 saving 两项：(i) 跳 image draw ~30ms + $2e-5/step (ii) **省 image tokens ~600/1100/step（dominant）**。

**证据**：读 `p79/experiment/som.py::_extract_text_marks` line 24 确认。

> ⚠️ 台账给 §103 三条挂了 `named by RETRACTED §106` flag（§103 N=48 的「5/5 macro metrics
> P-text = P-SoM，representation 主导」结论被 §106 作废）。**被作废的是 N=48 那条 macro 结论，
> 不是本条代码事实**——但读 §103 必须连 §106 一起读。⚠️ 待跨批核对（§106 RETRACTED 记录在 B 批）。

### 1.6 SoM-Vision gap 不做线性分解

**裁定**（§101 04-26）：SoM-Vision gap **不能**简化为 `[SOM_MARKS]` 全文本贡献 —— 它包含
(a) 文本贡献 + (b) 标签覆盖在截图上的视觉差异 **两部分**。
理由：reddit B1 SoM-Vision +3.3pp = 同样文本贡献被 reddit 高密度标签的负面影响蚕食 ~3pp。**不做线性分解。**

> 这条与 PROGRESS 三条不可违反的第 1 条（§302 线性分解 = category error）**同型**，
> 是同一个方法论纪律在 A1 期的首次出现。

### 1.7 mode 命名与最终 scope

**Paper-facing naming 定档**（§106 04-29）：
- **P-text**（内部 `phantom_text`，legacy alias `phantom_dom`）= [SOM_MARKS] + DOM-prompt + 无图
- **P-prompt**（内部 `phantom_prompt`）= AXTree + SoM-prompt + 无图
- **P-SoM**（内部 `phantom_som`）= [SOM_MARKS] + SoM-prompt + 无图

理由：legacy value `phantom_dom` 误导（读起来像 DOM 的 phantom），但 run dir / condition_id 需向后兼容。

**P-DOM / Phantom-DOM 不存在**（§108.20 05-02，用户显性 confirm）。
**经验教训原文**：*"当 user message naming 跟 memory canonical 冲突时应先 confirm，不要 echo typo into doc"*。

**为什么加 P-prompt**（§106 04-29）：P-text 是 text 单 swap 但没有对应的 prompt 单 swap → cube 不对称，
**reviewer 必问 did you control prompt × text interaction**；当时 cascade 只能在 [SOM_MARKS]-text
context 下测 axis 2，无法分离交互项。5-mode design cube 因此升级为 **4-level Diamond**
（DOM → {P-text, P-prompt} → P-SoM → SoM）。

**Final scope confirm**（§108.20 05-02 用户）：Modes = DOM / SoM / Vision / P-text / P-prompt / P-SoM
= **6 modes**；Cells = 6 sites × 3 models × 6 modes = **108 baseline cells**；
**Cascade narrative（paper §1 hook）= 5-mode subset** DOM→P-text→P-SoM→SoM + Vision
（**P-prompt 不在 cascade**，因 prompt 复杂但 obs 简单，不在单调路径上）；B1 P-prompt = Tier 2 优先级。

> ⚠️ 待跨批核对：此处 6 sites / 108 cells 与当前 CLAUDE.md 的 Phase 1a = 42 conditions / 6 cells
> 差距极大，中间必有多次 scope 收缩（§110 的 16-cell 是其一，见 6.3）。演化链须在合并阶段补齐。

---

## 二、FP 体系四轮演化 —— 从逐 case 补丁到有文献锚点

这是 A1 片里**最清晰的一条演化路径**，也是最容易被重新提起的：每一轮都是在前一轮基础上"再加一条规则"，
最后第四轮**把前三轮的 magic number 全删了**。

### 2.1 演化路径

**§78（04-18/19）第一轮 —— 核心原则**
`agent_finished = final_action_type ∈ {finish, stop} ∧ ¬fallback_finish`；
agent 主动 finish → 真阳性（无论 N/A 还是普通 string_match）；
agent 非主动 finish + string_match + success → 评测器噪声 → `eval_fp`。
理由：runner 追加 fake stop（answer=''）时 GPT-4o-mini 评测器会把空串误判为正确（如 ref='0 order' 时空串判 correct）。
**同日扩展**：eval_fp 适用 eval_type 从 string_match 扩到 (string_match, program_html)；
**url_match 明确排除** —— 导航到正确页面而不 finish 是合法的。
触发案例：VWA reddit task 69 VISION，agent 3 次 click 全失败从未输入/提交，但 post 上已有旧评论匹配
`reddit_get_latest_comment_content_by_username()` → program_html 误判 score=1.0。

**§83（04-20）第二轮 —— 引入 PUR 阈值**
string_match 一律 E-FP；program_html **仅当 PUR > 0.5** 时标 E-FP。
理由：18 个 E-FP 实例的 PUR 呈**双峰** —— 低 PUR (<0.3) = agent 有实际操作偶然匹配不太可能；
高 PUR (>0.7) = agent 几乎无操作预存状态碰巧匹配。
触发案例：B1 classifieds task 5 SoM（program_html，agent 真执行了删除，PUR=0.23）被误标 E-FP。

**§88（04-22）第三轮 —— 补一个 OR 分支**
program_html E-FP 规则 = `PUR>0.5 OR (¬has_effective_action AND ¬require_reset AND url_unique_count≤2)`；
`has_effective_action` = episode 中用过 type/select_option；`require_reset` 来自 task config。
同时修 `has_image` task 在 visual_fp 检查中**提前 return 短路 eval_fp** 的 bug（改为 skip visual_fp 标记但继续走 eval_fp）。
理由：Reddit task 69/72 DOM program_html 误判 success=1.0（agent 全程 click/back 无 type/select/finish，
旧评论碰巧匹配），旧 PUR>0.5 规则漏掉因 **PUR≤0.11**（重复 click 触发 content_changed 压低 PUR）；
reddit 69/72 恰好 has_image=True 故修前被 visual_fp 路径短路。

**§95（04-24）第四轮 —— 推倒重来**
- **删除 visual_fp 层**，adjusted SR 只含 na_fp + eval_fp
- **eval_fp 简化为两条规则**，去掉 PUR 与 url_unique magic number：
  string_match → `agent_finished=False` 即 E-FP；program_html → `agent_finished=False ∧ ¬has_effective_action` 才 E-FP；url_match 不标。

### 2.2 §95 删 visual_fp 的四条理由（防重新提起的关键）

1. **无文献先例** —— 10 篇 WA/VWA 核心论文无一做跨 representation 的 visual FP 过滤；
   VWA 原文立场 *"all tasks are visually grounded"*；ExACT 用 text-only modality 子集做公平比较
2. **边界不可判** —— DOM 文本常泄露视觉信息（颜色在标题中 / 物品名暗含外观）
3. **过滤范围过大** —— Codex 手动审计确认 VWA **95.3%** task 为 visual（cls 96.2% / reddit 99.5% /
   shopping 92.9%），过滤几乎等于否定 DOM 全部成功
4. **与 routing 研究冲突** —— 剔除 visual task 后仅剩 **43 个 VWA task**，无法做有意义路由分析

**§95 删 PUR magic number 的理由**：原三层规则在逐 case 手动检查中**递增构建（§78→§83→§88），
有过拟合倾向**，PUR=0.5 / url_unique≤2 缺乏系统性验证。
文献依据 = WebArena Verified 用 network activity monitoring + backend state delta 验证 agent 主动完成，
`agent_finished + has_effective_action` 是其**轻量代理信号**。

### 2.3 文献锚点（§95）

| 论文 | 在 FP 体系中的角色 |
|---|---|
| WebArena Verified (2025) | eval_fp 哲学来源（active causal verification） |
| PAE (2025 ICML) | na_fp 实证（~50% FP 来自 N/A 猜测） |
| AgentOccam (2025 ICLR) | case-by-case evaluator 修补先例 |
| VWA (2024 ACL) | "all tasks visually grounded" 即**不做 visual FP 的依据** |
| ExACT (2025 ICLR) | text-only modality 子集即 non-visual subset 参考 |

来源 = 10 篇核心论文调研；原文档 `docs/literature/结果1.md` 与 `结果2.md` **现已不存在**。

### 2.4 论文呈现框架与向后兼容

- **主表** raw + adjusted（仅 na_fp + eval_fp）；**Robustness check** 用 non-visual subset SR
  （手动审计 43 VWA + 480 WA = **523 tasks**）；**Discussion** 承认 VWA 无 Verified 版本（文献空白）
  + 引 WebArena Verified 说明 eval_fp 设计思路 + 讨论 DOM 在 visual task 上的结构性劣势并指向 non-visual subset（§95）
- **敏感性分析**：无过滤 / 仅 agent_finished / 简化版 / 原三层，四种定义下 adjusted SR 变化幅度（§95）
- **三套指标同报**（§89 04-23，回应导师对 visual FP 过滤公平性的质疑）：raw SR + adjusted SR +
  non-visual subset SR；**DOM 在 visual task 上的结构性劣势当 finding（表征选择决定能力边界）而不是 confound**；
  说明引入 WA 的 fairness motivation（67%→44%）
- **向后兼容**（§78/§83/§88）：老数据无 agent_finished / eval_type / PUR 等列时传 None → 退化到原一刀切逻辑

### 2.5 ⚠️ 矛盾：prereg 与代码曾不一致

§95（04-24）已删 visual_fp，但 `preregistration.md` line 197 一度写
*"FP filter primary = na_fp + eval_fp + visual_fp combined"*，该表述由 **RETRACTED §115.1** 作废。
台账在 §95 的**四条**记录上都挂了这个 flag。
⇒ **结论层记录**：代码（§95）与 prereg（line 197）曾冲突约两周，以 §95 为准。⚠️ 待跨批核对 B 批 §115.1 全文。

### 2.6 相关但未采纳

- **§56（04-14）**：维持现行 adjusted SR 定义（B0 classifieds DOM 7.7%），**不采用严格定义（~0.4%）**。
  理由 = §56 DOM FP 分类体系深化后的取舍。
- **§100 Next steps #5**：`NON_VISUAL_TASK_IDS` 的 docstring 写 'manual' 但**实际是 codex 判定**，
  应由 codex 重新独立判断。该 list 是 §95 non-visual subset robustness check 的基础。**未决**。

---

## 三、B0/B1 对称性 —— scaffold confound 的系统性消除

### 3.1 为什么这件事被反复做

B1 本地推理天然结构化（`do_sample=False` + `max_new_tokens=384`），B0 走 proxy API。
两边任何 prompt / 解码 / 解析差异都会让 **capability 差异与 scaffold 差异混淆**。§44–§47 四节连续修。

### 3.2 已对齐的（改了）

| § | 内容 | 触发理由 |
|---|---|---|
| §44 | B0 改三模式独立 system prompt（`_get_system_prompts()` 返回 {dom,som,vision} dict）；conditions.py Phase1 metadata 写 model_name | 原 `_get_system_prompt()` **只有单一 DOM prompt**（Critical）；condition_id 同名 `phase1_dom_router_0` 无法区分 baseline |
| §45 | system prompt 嵌入 user content（`'System: {prompt}\n'`）并删 'system' 字段；图像位置改 `insert(0)`；click 历史优先输出 `[id=N]` | 消除 B0/B1 prompt 结构不对称 |
| §46 | obs_section 三模式格式与 `qwen3vl_agent.py` **完全一致**（vision→空串，som→obs_text 直传），不加额外标签前缀 | 修复前 B0 SoM 多 36 字符前缀 `'SOM_MARKS and annotated screenshot:'`，Vision 多固定标签 `'Screenshot (no text)'`，**使 B0 prompt 内容与 B1 不同，影响 B0/B1 对比有效性** |
| §47 | B0 SoM prompt 加降级 fallback（'If [SOM_MARKS] is empty... fall back to coordinate'）+ Vision type 描述改 'automatically clicks to focus'（删歧义的"先 click 再 type"） | §47 A2 设计不对称修复 |
| §36 | 参考图标签改为 `'[Reference image N] This image shows the target item described in the task...'`；三模式共用同一段注入代码 | 原标签 `[Input image 1]` 未说明用途，**4B 模型不会主动建立"该图=任务目标"关联** |
| §33 | 参考图**必须**传给 DOM 模式（删除 `observation_mode != 'dom'` 条件） | DOM 模式此前拿不到 reference image，视觉目标任务无从判断 |

### 3.3 **保留不改、只做论文披露的不对称**（防重新提起）

- **A1 解码策略**（§47 04-13）：B0 `temperature=0.1`（轻度随机，重跑可能产生不同 trajectory）
  vs B1 `do_sample=False` 贪婪解码（完全确定）。理由：**对 SR 影响极小，可接受**。
  → **后被 §107 C4 覆盖**：Phase A 把 18 个 B0 yaml 的 T=0.1 改为 **T=0.0** + RNG seeding。
  ⇒ 两条并存不矛盾（A1 期披露 → Phase A 期消除），但**引用 §47 A1 时必须连 §107 一起读**。
- **A3 max_new_tokens**（§47 04-13）：B0 4096（thought+JSON 从不截断）vs B1 384
  （verbose thought 可能截断 → parse_failed → wait action 浪费步数）；
  **B0/B1 SR 差距部分来自此非能力因素**。理由 = B1 384 是 GPU 时间 trade-off，
  §97 第九轮**再次确认有意保留**。
  → §97（04-26）把 max_new_tokens **默认**统一为 4096（agent QV-9 256→4096 / backend LQ-1 512→4096 /
  `configs/exp_v2_base.yaml` 512→4096，三处必须一致否则 yaml 默认值覆盖代码 default 使修复失效），
  但 **B1 baseline yaml 的 384 保持不动**。

### 3.4 解析链路与 tool calling 的公平性裁定

**链路定档**（§67 04-15）：`tool_use → text parse (json.loads → regex) → GLM extract → keyword fallback`。

**公平性裁定原文**（§67）：
- 方案 A（tool calling）是 **API 层输出格式切换，不改推理能力**；
- 方案 B（GLM 提取）**完全不改 235B prompt，且 prompt 不含任何 task context**（等价于更聪明的 regex），
  成本**不计入 `model_cost_usd`**，只记 `glm_fallback_*` metadata；
- **A+B 是消除 scaffold confound 使 B0/B1 可比** —— 不加这两层则 parse_error 变成 capability 差异的假象。

**scroll 语义离散化**（§67）：tool schema 把 `delta: [dx, dy]` 换成 `scroll_direction: enum('up','down')`，
agent 端再转回 delta 保持环境兼容；**判定为 format-only 不影响公平性**。
理由：消除训练数据中 deltaY 符号约定冲突（CSS dy>0=down vs Win32 dy>0=up vs macOS natural scrolling）；
`vwa_wrapper.py` 本来就只取方向丢弃量级。

**§70（04-16）方案 A 验证失败**：Proxy API（Bedrock 包装）**静默忽略 tools / tool_choice 参数**，
返回纯文本 `web_action('click','2')` 而非结构化 tool_use block → config 标 `use_tool_calling: false`，
改走方案 B。
> ⚠️ **本条是早期状态，已被反转**：台账条目自带注释 —— CLAUDE.md 记载后期（B-991 / Fire-6 2026-05-20）
> `use_tool_calling: true` + `tool_choice='required'` 已成 paper-grade default。⚠️ 待跨批核对反转发生在哪个 §。

### 3.5 B0 后端选型：不换 DashScope 也不换 Claude

**裁定**（§72 04-16）继续用 Bedrock proxy + Qwen3-VL-235B，四条理由：
1. proxy **免费**，DashScope 收费且免费额度已用 69%；
2. 两套 API scroll 表现一致（3.5% vs 3.2%），parse error 均 0，**API 不是差异来源**；
3. DOM 和 Vision 已跑完 234 tasks，切换意味全部重跑；
4. **同系列 weak→strong（Qwen-4B→235B）路由是原始研究设计**，换 Claude 改变实验变量。

**连带裁定**（§72）：**scroll 单向性与 stuck scroll = Qwen3-VL 模型行为**，不是 Bedrock proxy artifact，
也不是 prompt/环境通病。依据：DashScope 官方 3.2% ≈ proxy 6.9%（同模型不同 API 一致），Claude 36.2%
差约 10 倍；stuck scroll DashScope 19.4% ≈ proxy 26.4%，Claude 0%（到底部即停）；parse error 三条线均 0%。

**§50（04-14）**：B0 scroll 方向问题判定为**无可靠修复** —— 归为模型行为，论文作**已知局限披露**
（辅以 §66 定量统计 + §72 跨模型/跨 API 验证）。§66 为此在 `episode_reason_rows.csv` 新增 4 列
`scroll_up / scroll_down / scroll_direction_flips / scroll_wasted_steps`，理由：*"已有定性分析（§50）
但无定量统计，论文需数字支撑'已知局限'披露"*。

### 3.6 可复现性：seed=42 是假的

**§107（04-30）裁定**：paper 之前的 **seed=42 reproducibility claim 定性为 metadata-only（非真实现）**。
依据：`grep random.seed / np.random.seed / torch.manual_seed / set_seed` 在 `p79/` 全 **zero matches**；
18 个 B0 config 全用 T=0.1；payload 不传 seed；**Anthropic API 协议根本没有 seed 参数**。

**§107.1（04-30）**：B0 vs B1 reproducibility 不对称正式入 Section 4 disclosure，
定性为 **honest characterization，不是 retracted limitation**：
B1 byte-deterministic by construction（`do_sample=False` + `torch.manual_seed`）；
B0 **not byte-deterministic in probe，但 decision-level convergence empirically observed**。
理由：*"实证 probe 证据 > trust the proxy claim；cost ~$0.005 换 paper rigor leverage"*。
证据文件 `docs/analysis/cross_sites/probe_b37_api_determinism.md` **已缺失**。

---

## 四、动作执行层 —— scaffold bug 的根因谱系

这一节的价值不在"修了什么"，在**根因**：几乎每条都是"表面现象骗了第一次归因"。

### 4.1 最贵的一条：94.4% off-target click

**§107（04-30）Phase A 4-cluster ship**（commit 3c15cd7，~455 LOC 跨 12 文件，88/88 pytest pass）：
- **C1 locator-route dispatch**（新建 `p79/envs/locator_dispatch.py`，JS walk-up + element-handle dispatch，
  TYPE 改 `locator.fill()` 消除 Meta+A 全选变蓝）
- **C2 page_changed split**（新增 `agent_visible_changed` step field，`page_changed` 保留 12-reason union 用于 cycle 决策）
- **C3 fuzzy cycle hash**（第 3 个 cycle track on `(action_type, url_path_no_query)`，`min_reps=5`）
- **C4 RNG seeding + T=0.1→0.0**（18 个 B0 yaml）

理由：5-tier audit + Tier 10 probe 定位 **94.4% off-target click dispatch 为 #1 bug**；
SR derivation 需排除 `form_value_changed` / `dom_complexity` false trigger。

### 4.2 "AXTree 里的 [N] 在真实 DOM 里不存在"

**§51（04-14）**：新增 `select_option` action type；element_id 路径**必须**从
`self._last_obs_nodes_info[str(eid)]['union_bound']` 取像素中心坐标，再用 JS `elementFromPoint(x,y)`
定位 SELECT + `dispatchEvent('change')`，**不能用 `locator('[bid=N]')`**。
根因两条：(a) Playwright sync API 对原生 `<select>` 的 click 不弹出 option 列表；
(b) **bid 属性在真实 DOM 里不存在** —— VWA 仅在生成 AXTree 文本时写入 `[N]` 标记，
执行时靠 `obs_nodes_info union_bound` 像素坐标，**用 `[bid=N]` 定位必然 0 匹配**。

### 4.3 焦点被 CDP 快照重置

**§28/§29（04-11）**：Vision type+coordinate（无 element_id）**必须 pre-click 聚焦**，
且 pre-click 用 `page.mouse.click()` + `wait_for_timeout()` 而**不是** `env.step(click_action)`。
根因：`env.step` 触发完整 click→sleep→**CDP captureSnapshot** 流程，
**CDP 捕获把焦点从 INPUT 重置到 BODY**，后续 `keyboard.type` 输入到 BODY 无效（Playwright 直接对照验证）。

### 4.4 "全选变蓝"有两个不同根因，不要混

**§52/§64（04-14/04-15）**：
- **§52** = VWA 框架内置 Meta+A（scroll 后 input 全消失，模型把 link/span 当 input 用）
- **§64** = P79 自写的 `Control+a` + Backspace 清除逻辑在 click 未 focus input 时全选整页
- **修法不同**：§64 修法 = `Control+a` 前检查 `document.activeElement`
- 相关：**§30** Vision type 预清空用 `Control+A → Backspace` 不用 `Meta+A`（Linux 下 Meta+A 无效）

### 4.5 静默吞掉的异常

- **§30（04-11）**：VWA `actions.py` 的 `execute_mouse_click` 等函数统一加 `float()` 转换。
  根因链：`create_mouse_click_action` 存坐标为 **np.float32** → **NumPy 2.x 不再自动提升为 float64**
  → Playwright CDP JSON 序列化 float32 触发 TypeError → **被 VWA try/except 静默吞掉返回 reward=0**。
- **§63（04-15）**：`parse_action_text()` 入口先剥离 think 块
  （`re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()`）。
  根因：Qwen3-235B-A22B 部分步骤输出 extended thinking，**DOTALL 贪婪匹配从 think 块内第一个 `{`
  到文末最后一个 `}` 捕获非法 JSON**。

### 4.6 agent "感知不到自己操作生效"

- **§54（04-14）**：select 注入格式改为 `[OPTIONS: currently selected='Jewelry'] 'opt1', 'opt2', ...`
  （无 selected 时退化为原 `[OPTIONS]` 格式）。
  根因：VWA AXTree combobox 显示 `<label>` 文本而非当前选中值，执行 `el.value=opt.value` 后
  **AXTree 文本不变（text_sim=1.000）**，模型无法感知操作已生效 → 重复 select → cycle。
- **§61（04-15）**：`som.py::_build_som_result()` 构建 mark_lines 时需扫 obs_text，
  为每个 combobox/触发节点把紧随其后的 `[OPTIONS...]` / `[DROPDOWN OPTIONS...]` 行追加到对应 mark 行下方。
  根因：`_extract_text_marks()` 只提取有 `[N]` 编号的行，无编号的 `[OPTIONS]` 被过滤 →
  **SoM 模式 agent 从未看到 combobox 选项**（DOM 模式不受影响）→ **B0 SoM classifieds 234 个 episode 清除重跑**。
- **§60（04-14）**：新增 `_inject_css_dropdown_options()` 注入 `[DROPDOWN OPTIONS]`：JS 查询所有
  `getBoundingClientRect()=0` 的隐藏 `<ul>`（2-20 个 `<li><a>`），找最近可见祖先为触发器，
  匹配 `obs_nodes_info` 中 ≤150px 的节点。**与 §51 的区别：§51 控件在 AXTree 中可见，§60 连控件选项都不存在于 AXTree。**
- **§62（04-15）**：Vision `select_option` coordinate 路径在 native select 判断后追加 CSS dropdown fallback。
  根因：原路径仅判断 `el.tagName === 'SELECT'`，**CSS dropdown 条件永远 False → action 静默无效**；
  §60 只修了 DOM/SoM 的文本注入，**Vision 执行路径从未覆盖**。
- **§53（04-14）**：`reset()` 后在 page 上注册 dialog 监听器（confirm/alert → accept，prompt → dismiss），
  用 `_dialog_registered_page` 跟踪同一 page 对象避免跨 episode 重复累积。
  根因：VWA ScriptBrowserEnv 未注册 Playwright dialog 监听器，**原生 confirm 阻塞导航使所有 delete 类任务失败**。

### 4.7 状态变化检测的假阴性与假阳性

**§68（04-15）**三项改动：`build_page_state()` 接受 `form_snapshot` 参数替代**死代码 scroll**；
新增 `_form_fields_changed()` 按 `(tag,type,name,idx)` 匹配字段对比 value/checked/selectedIndex；
scroll 从无条件 True 改为检查 `scroll_y` 变化 ≥5px。开关 `state_change.form_snapshot_enabled: true`。
- **假阴性**：全页 AXTree 文本相似度阈值 0.95 无法察觉 type 小范围编辑 / select 原地切换 / checkbox toggle
- **假阳性**：scroll 在 L147-148 **无条件 return True**，而 `scroll_x/y` 从 `info.get()` 读取但
  **VWA 从未填充这两个字段（死代码）**

**§105（04-29）**：`state_change.py::_key` 对 radio/checkbox 加 value discriminator
（4-tuple → 5-tuple）。根因：Magento custom-option 把每个 radio 放进独立 `div.field choice` wrapper
（each sole child → idx=0），**3 个 same-name radio key 完全相同 → dict comprehension 互相覆盖只剩最后一个**
→ click 非最后那个时 before/after dict 完全相等，不报 `form_value_changed`。

**§68 的论文义务**：方法节必须披露 —— scroll 的 `action_success` 从无条件 True 变为实际检查 →
**新旧数据 `no_op_rate` 不完全对齐**；`form_value_changed` 是新增的 `page_change_reasons` 值，老数据不出现。

### 4.8 上游 VWA 的运算符优先级 bug（影响最广的一条）

**§80（04-19）**：修复 VWA `processors.py` 的 `in_viewport_ratio` 运算符优先级 bug
（`overlap_width * overlap_height / width * height` → 加括号成面积比），
并**重跑所有 DOM/SoM condition（B0+B1 全站点）**；Vision 不受影响不需重跑。

根因：原式实际是 `((ow*oh)/w)*h` 远超 1.0，使 `IN_VIEWPORT_RATIO_THRESHOLD=0.6` **形同虚设** ——
任何 1px 可见元素都保留完整文本；VWA 官方用 2048px viewport 无感，**P79 用 720px 影响放大**。

修后阈值 0.6 的数学保证：`ratio ≥ 0.6 → 元素中心必在 viewport 内`（`center_y ≤ 720 − 0.1h < 720`），
**功能性失败完全消除，残余不公平是 DOM 元素级全有/全无的结构性限制**。
论文注明：修正了上游 ratio bug，使用原始阈值 0.6，viewport 720px。

> 台账附实测（2026-07-28）：`external/visualwebarena/browser_env/processors.py:219` 已是加括号的修复版。

---

## 五、Cycle 检测与 early-stop —— 从"省步数"到"全面取消"

### 5.1 早期：加检测

| § | 规则 | 理由 |
|---|---|---|
| §5 (04-04) | `busy:1` 检查提到 LLM 调用前，且免费等待不计 max_steps | VWA networkidle timeout=2000 过短，远程站半加载，**LLM 推理已浪费后才覆盖 action 为 wait** |
| §12 (04-07) | scroll + `page_changed=True` 时不加入 strict cycle signatures，只留 soft 检测兜底 | 页面正常滚动、agent 浏览搜索结果时被 strict 检测误判死循环 |
| §33 (04-12) | 三条早停：scroll 交替死循环（连续 6 次 up/down 交替）/ URL stuck（连续 5 次 click 同 URL，**仅 click 计数**）/ about:blank 自动恢复 | Reddit 启动 3 个 episode 即暴露这些浪费模式 |
| §57 (04-14) | `_action_signature` 与 `_action_signature_soft` 在 `tab_focus` 时额外拼入 `\|pn={page_number}` | 未含 page_number 时所有 tab_focus 签名相同，多 Tab 切换被误判重复；**修后 task 229（pn=1→0→1）不再触发，task 150（pn=1→1→1 真死循环）仍正常 kill** |
| §107 C3 (04-30) | fuzzy cycle hash：第 3 个 cycle track on `(action_type, url_path_no_query)`，`min_reps=5` | Phase A |

**§31/§33/§45**：`baseline_retry_on_no_progress` **默认关闭（False）** ——
理由：§31 论证 **auto-scroll 副作用不可预测，retry 反而浪费步数**。

### 5.2 转折：early-stop 是 design layer 问题，不是 metric 问题

**§108.10（05-01）裁定**：Early-stop 定性为 **design layer 问题（agent system 是否包含 early-stop）**
而非 micro-metric measurement 问题。理由 —— **影响是 systemic，不止 micro layer**：
- **Outcome**：早停 task 没自然结束机会
- **Macro**：短 trajectory 是 **censored 数据**
- **Micro**：denominator confound
- **Efficiency**：不同 mode 触发率不同 → cost diff **partial 来自早停频率**

且 **Phase A Cluster 3 只是 partial mitigation 不是 full cancel**。

三候选：**A Full Cancel**（+$1300 14-cell rerun，全 dimension clean）/ **B Full Keep**
（cost-realistic，accept systemic confound）/ **C Hybrid**（main + 1-2 mechanism cells without，+$200）。
当时状态：用户 lean A，等学长 align。

**§110（05-05）advisor 5/5 sync verbal confirm：Option A 全 cancel。**
advisor 原话：*"影响到你对这分析，不想 early stop"*。

**§116.2（05-08）代码层真正关闭**：`runner/main.py` 加 config flag `_early_stop_enabled`
（default False per advisor 5/5），3 个 fire 点（line 1347 cycle detection / 1367 scroll alternation
`ALT_THRESHOLD=6` / 1386 URL stuck `URL_STUCK_THRESHOLD=5`）wrap 起来；
**保留 logging 作 paper-grade 诊断但不 truncate trajectory**；显式 opt-in `runtime.early_stop_enabled: true` 才能再启。
按 Protocol A 这是 **Tier 0 bug（eval logic 与 spec 不符）**，pre-lock 修法 = fix code + 笔记 chronicle，
不需 advisor 二次 confirm。

> **注意时间差**：advisor 05-05 confirm 方向，但**代码继续 fire 到 05-08** 才关。
> 05-05→05-08 之间产生的数据带 early-stop。⚠️ 待跨批核对该窗口是否有数据落盘。

---

## 六、Paper framing 演化 —— hook 是怎么变的

### 6.1 hook 的两次改写

**第一次（§108，05-01）**：从 *"P-SoM is hidden 4th routing arm"* 改为 **"phantom routing space（3 arms）"**：
边界 = **skip annotated image**，含 P-text / P-prompt / P-SoM 三个 arm 共享 4-fold drop-in property；
P-SoM（cube center，axis1+axis2 compound）是 representative arm。
理由：phantom_lift 数据显示 **3 个 phantom arm 都贡献 unique tasks**，旧 hook 字面不准。

**第二次（§108，05-01）boundary 论证改写**：改为 **no annotated image**（不是 matched parsing）。
- (a) cost ≈ DOM / (b) latency ~50% 由 boundary **definitional derive**
- (c) signal AUROC ≥ baseline / (d) drop-one positive 是 **emergent validation**
- 排除 cube 的 image-on phantom corners，因为加 image 拉齐 SoM cost **失去 drop-in property**

理由：旧 "mismatched parsing" 论证**有 counter-evidence** —— P-prompt 表面就是 mismatched
但实测 +2.86pp drop-one sig，有真 LLM 机制（visual prompting without image）。

### 6.2 M1/M2 双轴与"完备性"的论证方式

**§108（05-01）M1/M2 mechanism activation 2x2**：
Axis **M1（Image-mirage activation**，触发条件 prompt 期待 image）×
Axis **M2（Flat-list activation**，触发条件 obs text 是 flat indexed list）；
DOM (❌,❌) / P-text (❌,✅) / P-prompt (✅,❌) / **P-SoM (✅,✅ compound cube center)**。
理由：**Prompt textual coupling ≠ mechanism activation coupling**；LLM internal state 层两个 axis 正交；
P-SoM unique tasks（3 tasks B0 reddit，既不在 P-text 也不在 P-prompt）是 attention nonlinear combination
的 **emergent capability** —— P-SoM 作为 hero 的 mechanistic 理由。

**§108.5 完备性用 Approach 2 architectural elimination（deductive），不用 empirical residual 检查**：
- Approach 1（empirical）致命弱点 = **finite data 永远不能证明完备性（induction problem）**
  + fitting 现有数据 tautological
- Approach 2 前提链：phantom space 锁 image=✗ → input 只剩 prompt 文本 + obs 文本 →
  LLM 是 deterministic forward function on input tokens（T=0 greedy，Phase A 后真）→
  任何 differential 机制必由 input 差异 trigger → **M1/M2 exhaustive**

**§108.5 Caveat（一字不丢）**：Architectural completeness argument **只给 axis-level 完备性，
不给 axis 内部 sub-mechanism 完备性**（M1 内部 Mirage Effect / Scaffold Effect / Cross-modal flow
哪个 dominate 仍需 lit + empirical 区分）。**这个 caveat 直接引出 evidence/explanation separation。**

### 6.3 Evidence ⫨ Explanation 两层分离

**§108.6（05-01）**：Paper conceptual structure **严格 2 层分离**：
- **EVIDENCE LAYER（2D）**：4 测量类型（Outcome/Macro/Micro/Efficiency）× 4 cross-X comparison
  = **16 sub-cells**
- **EXPLANATION LAYER（1D zoom scale）**：Zoom 1 architectural / Zoom 2 M1M2 axis behavioral /
  Zoom 3 named cross-model phenomena / Zoom 4 model-internal mechanism

理由（用户 push back）：*"lit anchor 不在自己的 layer，是 explanation layer 内部 zoom scale 不同位置"*
（Asadi Mirage / Vu Scaffold / Sclar → **Zoom 3**；Kaduri Cross-modal flow / SteerMoE → **Zoom 4**）。

**§106（04-29）前置**：paper finding 组织从 **10 条 flat findings** 改为 **4-dimension Evidence + Mechanism
Framework**：Outcome（哪些 task 成功）/ Macro（agent 平均怎么 act）/ Micro（per-step decision）/
Efficiency（cost/latency/carbon）—— **四个正交 dimensions 不是 hierarchical layers**。
每 sub-evidence（0a-0g / 1a-1c / 2a-2e / 3a-3d）标 source artifact + live 数字 + cross-reference；
原 10 条 finding 全部 mapped 保留 legacy index。

**§108.7 cross-X taxonomy**：每 episode = (task, mode, site, model) 四元组，**固定 3 变 1 = 一种 cross-X**；
cross-mode 是 phantom space 主分析（§4/§5），cross-site/cross-model 是 §7 generalization。
诊断结论：*"§3 sub-codes 0a-3d 几乎都是 cross-mode paired metric —— **cross-site/cross-model 数据稀薄
是 paper §7 当前 ~40% 完成度的根因，不是 prose 写作问题**"*。

### 6.4 novelty 防线（双支柱 scope-defense）

**支柱 1 —— 防"industry already does this"**（§109.17，05-04）：
**artifact-existence 与 research-characterization 是不同 epistemic level**。
industry（agent-browser / Tarsier / Playwright MCP / Stagehand）已 deploy flat-text representation（cost driven），
但**production 单 mode 部署没有 controlled comparison harness，不知道 text-flattening 自身有独立 routing effect**。
paper claim 改为 *"first systematic peer-reviewed characterization of routing behavior across phantom
routing space configurations via controlled cross-mode comparison"*。
背景：用户 push back 纠正我 §109.16 的 **over-correction**（曾写"P-text 不 novel because agent-browser
default 是同 representation"）；**所有 4 phantom corner 在 research-novelty 层面 equally important，
缺哪个都失去某个 axis isolation 能力**。

**支柱 2 —— 防"why didn't you ablate SE module X"**（§109.19，05-04）：
**cognitive-routing vs SE-engineering** 区分。站点指纹库 / 短 action grammar（`click @7`）/ FPC fix /
Phase A 4-cluster / watchdog auto-clean **全部归为 SE deployment optimization，排除出 paper scope**
（Phase A 与 watchdog 保留为 evidence-layer prereq / Appendix D）。
用户自己 catch 的原话：*"站点知识注入也是 module 但不是 routing，没法做 routing，
否则我们就是在测一个 software engineering"*。
区别本质：Phantom 4-corner 跟 SE module 都是 fixed-mode + cost-saving，但
**research instrument vs deployment tool**。

**Scope 边界**（§109.19）：paper 限定在 **observation-representation axis**
（4-corner cube = text-payload × prompt-format × image-presence）；
**action-grammar axis**（LLM output serialization，`click @7` 短语法 vs verbose JSON action schema）
**正交且不 cover**，明写为 future work（8-cell extended cube = 4 observation × 2 action）。
理由：Phantom 4-corner 全程用 VWA default verbose action serialization 保持 observation-axis ablation control。

**§109.2（05-04）9-cell intervention taxonomy**：L1 Server-side / L2 Pipeline / L3 LLM-internal ×
(i) Bug fix / (ii) Synthesis / (iii) Channel addition；**我们 paper 占 (ii)×L3 的 inference-time-only sub-tier**
（对比 Magma pretraining-time / ScribeAgent fine-tuning-time / AppAgent-v2 RAG-offline-explore）。
理由原文：*"学长 5/3 push 的真正意义不是加 environment-side 新方向，是把已做 work 升级成 explicit framework"*；
笔记 §1-§108 ~40 条 § 映射进 9 cells（L1 ~6 / L2 ~28 已做）。

**§109.2 的连带用途**：(iii)×L2 self-perception channel-addition 的 7 条 §
（§52/§64 全选变蓝 / §55 delete 信号缺失 / §72 scroll_up / §72 scroll 到底不停 / §31 auto-scroll 不可预测 /
§32 Vision 坐标 misclick 零自纠正 / §96 B1 click 非交互）**共享同一 root cause，用来撑 paper §5
shared ceiling argument** —— phantom routing space 的 drop-in property 是测在**固定 (iii)×L2
channel-absent ceiling 下**的 (ii)×L3 substitution capability。

**§109.6（05-04）differentiator**：Magma 用 Qwen3-VL backbone 把 SoM+ToM integrate 进 **pretraining weights**；
ScribeAgent 用 Qwen 7B base **fine-tune** 6B token DOM corpus 到 WebArena 45.7→51.3%；
**我们用 non-pretrained non-fine-tuned Qwen3-VL 在 inference time —— clean experimental isolation**。
理由：同 model family 让 isolation 论证 reviewer 一眼 buy。

**§109.14（05-04）**：Playwright vs CDP substrate 是 **implementation detail 不影响 paper claim about
routing structure** —— paper §3 一句话 acknowledge。理由：CDP 是 lowest-level common substrate，
Playwright 是 CDP wrapper；agent-browser/OpenClaw skip Playwright for performance —— **两者都坐 (ii)×L2 cell**。

### 6.5 contribution 的增与删

**加**（§101 04-26）：Site visual quality **不是 uniform 整体属性，是 task-category 的函数**；
contribution 升级到 **capability × site × task-category 多维 routing**。
依据：classifieds 内 A(text-only) SoM-DOM = 0pp 但 C(page-screenshot) = +10.4pp（B1），
视觉收益强烈 task-category 异质。

**删**（§119 05-09）：删除 paper §1 **第三 contribution "capability × representation interaction"**
+ `fig_capability_b0_b1.png/.py` + `+43.7pp` 引用；§1 收紧到 **2 个 contribution**：
(1) Phantom routing space + 4-fold drop-in（HERO H1）(2) Structural axes P-text/P-prompt empirical witness（H3）；
**B1 改述为 cross-capability robustness check，not separate scientific claim**；
`disagreement_clusters.md` 5-bucket failure rubric 保留为 supplement candidate。
理由三条：capability gap 是 **sidebar diagnostic 不直接 support main claim**（figure 自身 docstring
已写 *"not part of the 4-dimension evidence framework"*）；reviewer 会问 *"这是 phantom paper 还是
model-comparison paper"*；**16-cell rerun 后 capability 数字会变但 phantom claim 不依赖 —— 削减风险面**。

**§106 的一条方法论 insight（cls 特有）**：cls 上 **aggregate macro 是 misleading** ——
routing arm 价值需 **outcome（Outcome 0d task-pool Jaccard）+ micro（Micro 2a page-id divergence）
一起证，不能只 cite macro**。依据：cls Macro 6/8 cells DOM-like（P-SoM aggregate ≈ DOM）看起来无行为差异，
但同时 task-pool Jaccard 0.53 + micro page-id divergence 34% 显示实际决策不同。

**§77（04-18）framing**：oracle headroom 当作 **well-defined ceiling**，论文用
*"approaches X% of headroom"* 的标准 routing 表述（得到 ceiling 24.79% / headroom 4.27pp 后的决定）。

**§118（05-09）external validity caveat 放宽**：从"效应 1 site evidence"改为
*"phantom mechanism in Qwen3-VL family on VWA-style tasks（cls + reddit, 2 sites）"*。
依据：Cell F 在 reddit 复制出 L11+L17 mid-layer 机制，p_Holm 比 cls 紧 2-3×。
> 台账 flag：§118 的 cross-site 对比表把 cls Cell A L17 = 0.011 与 reddit Cell F L17 = 0.004
> 放在同一 `overlap→tgt` 列标题下的表述被 **RETRACTED §120** 作废。⚠️ 读 §118 必连 §120。

### 6.6 advisor 5/5 sync 的六项裁定（§110，05-05）

| # | 裁定 | advisor 原话 / 理由 |
|---|---|---|
| 1 | **Early-stop Option A 全 cancel** | *"影响到你对这分析，不想 early stop"* |
| 2 | **Manifest 全 archive + 16-cell 重跑** | *"对这是一个问题我之前说的"*；5/4 manifest archive 8a9f595 已落地 |
| 3 | **Paper 拆开发**（不是一篇大 paper） | *"我觉得分开比较好... 人类的优雅是把东西给拆成几篇"* |
| 4 | **Routing paper 定位为 benchmark study** | *"把这个当做一个 benchmark，类似 benchmark study 去发，这个会稳一些"* |
| 5 | **Mechanistic interp 从 deferred 提升为现在就启动的 net-new paper** | *"你现在已经有 contrastive set 了... 说不定单是这个结果你就可以拿出来单独发一篇比较好的 paper。如果是个 cross-model 的话，这就是 golden feature"* |
| 6 | **Compute path 4-tier** | (1) advisor 5090 搬 AI Center (2) Rancher/Condenser H100 (3) RunPod 4090 self-fund $200 (4) Myriad 基本放弃 |

**tentative 4-paper shape**：P1 Phantom routing space（benchmark/D&B track）/ P2 Routing benchmark study /
P3 Mechanistic interp（ICLR-NeurIPS 候选 if cross-model golden feature hold）/
P4 **VWA bug ACL position paper** 或 survey + community repo（*"难度不高"* + 持续更新的 website/repo）。

**Mechanistic 3 工具**：activation patching（找 mirage critical layer）/ linear probe（hidden state 预测
mirage label）/ **SAE feature steering（defer —— Qwen3-VL-4B 公开 SAE 不存在需自训 1-2 周）**。
> 台账注：advisor 录音提的 *"truth-telling circuits"* 实指 Tool Calling Linear Circuit（ACL 2026, Qwen3-4B）。

**§110.3（student post-sync decide，非 sync 现场讨论）**：**N_cells = 16** =
B0 × {cls,red} × 3 phantom + B1 × {cls,red} × 3 phantom + B0 shop × 2 phantom + B1 shop × 2 phantom；
**K_h1 ≥ 12/16，K_h3 ≥ 11/16**；OSF DOI early upload（post-email-witness 而非 paper-submission-stage）；
Mechanistic Stage 1 立刻启动（不依赖 advisor confirm）。
理由：加 B1 shop × {phantom_text, phantom_som} 让 cross-capability shop coverage 完整；
DOI 时间戳 < data unblinding 时间戳让 audit trail ordering 明确。

> ⚠️ **待跨批核对（重要）**：此处 16 cells + K-of-N 门槛（K_h1≥12/16）与当前 CLAUDE.md 的
> **42 conditions / 6 cells + K-of-N 降级为 transparency-only（primary = FE inverse-variance pooled）**
> 完全不同。中间的降级链在 A2–A4 / B 批。**不要拿 §110.3 的 16-cell 当现状。**

**§112（05-06）Compute path 再变**：Tier 0 = **UCL Condenser A100 dedicated（allocated 5/6）**；
advisor 5090 demoted 到 fallback；Rancher H100 redundant；**RunPod $200 NOT NEEDED（经费可 save）**；
Myriad 已放弃（CGNAT block）。理由：A100 dedicated 让 16-cell rerun 从 ~3 周（DGX shared）/ ~1 周（RunPod）
压到 **~3-5 d**，且 mechanistic scale-up 可 exam 期间 parallel。

**§110.7 process 定档**：**Sync 三件套**（transcript + followup doc + outcomes registry），
因网卡断 mid-sync 而形成；后续 sync 都按此 pattern。
理由：后半段 forest plot + preregistration.md 没传过去（*"advisor 马德里山太多"*）；
**audit trail 完整比单口头 confirm 强 —— paper §1 footnote 可 cite explicit decision provenance**。
> 三个文件（`docs/reference/transcript.md` + `advisor_sync_5_5_{followup,outcomes}.md`）**当前均已不在原路径**。

### 6.7 §107 的自我防御：Phantom finding 在 dispatch bug 下仍 valid

**四条论证**（§107 04-30，防御"Phantom 是 framework bug 假象"假说）：
1. 4 个 text-bearing mode **共享同样 dispatch contamination** → cross-mode paired comparison cancel noise
2. **Vision mode 不受 B-33 影响**（用 normalized coords）但 reddit Vision 10.48% < P-SoM 13.81% **反向证明**
3. pilot wave-2 Δ=0pp on N=60
4. 3-axis cube + diamond 是 **well-defined factorial 不是 bug 涌现**

> 依据文件 `docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md` §5 **已于 §108.20 删除**。

---

## 七、预注册 / provenance / 变更协议

### 7.1 Provenance 5-gap（§114，05-07）

| Gap | 内容 |
|---|---|
| Gap1 | model + library SHA pin（`snapshot_env.py` hook 进 `run_experiment`） |
| Gap2 | cross-environment VWA fingerprint（`snapshot_vwa.sh`，docker image digests + per-site HTML SHA-256） |
| Gap3 | Stage 2B `run_manifest.json` emit |
| Gap4 | OSF DOI lock 8-step checklist |
| Gap5 | cross-machine numerical determinism check（threshold default 1e-2） |

理由：audit 现有 manifest coverage = 70%；**agent benchmark provenance 比传统 ML 难一档** ——
3 个 agent-specific drift vector：VWA Docker image silent drift（mutable tag）/ site state pre-test
mutability / cross-machine numerical drift（**SR-level invisible 但 hidden state magnitude 可见**）。

**§114.3**：OSF DOI lock **8-step workflow + 3-layer witness chain**（git tag + email PDF + OSF DOI page）定档；
**post-lock 变更纪律 = 不 amend 文件而是 v2 preregistration 重新 DOI，两份都 paper §3 cite**
（避免 moving goalpost critique）。

### 7.2 Protocol A —— evaluator change 4-tier（§115.3，05-07）

| Tier | 定义 | pre-lock | post-lock |
|---|---|---|---|
| **T0** | Bug（eval logic 与 spec 不符） | 直接修 | **允许**，但 paper §3 必须 disclose + double decision report |
| **T1** | FP rule expansion | 允许 | **forbidden in same paper**（need OSF v2 DOI） |
| **T2** | FP rule simplification | 允许 | **forbidden in same paper** |
| **T3** | Definition change | — | **不允许同 paper** |

commit prefix 纪律：`fix(eval):` / `fix(eval-postlock):` / `feat(eval-v2):`。
**retroactive 分类**：§95 = **T2**，§105 = **T0**，§78a-§88 = **T1**。
理由：*"没 protocol 就无法区分 paper-grade fix vs p-hacking —— reviewer 会问
'visual_fp 在 §95 删了为啥你用旧版 re-eval'"*。

### 7.3 Protocol B —— re-eval audit trail（§115.3）

每次 rederive 写 **append-only `rederive_metadata` entry**，含 5 mandatory fields
（`rederived_at` / `evaluator_code_sha` / `fp_rule_version` / `rewrite_set` / `trigger`）；
legacy dict 单 entry promote 为 list 第 0 元素再 append；**fail-soft**；
**OSF DOI lock 时 all cells 必须 `rederive_metadata` 非空且最新 SHA == lock SHA**。
核心抽象 = **provenance triangle**（代码层 SHA + 决策层 rule version + 执行层 metadata trail）闭环互锁。

### 7.4 pre_rerun_audit 与 bug catalog

**§116.3/§116.12/§116.13（05-08）**：`pre_rerun_audit.md` 建立并按 **5-phase lifecycle** 重构
（Setup / Run / Results / Analysis / Publication & Continuity），覆盖 **22 sub-section ~210 gate items ~640 行**；
每项带 verify 命令；之后每次 rerun 前必走一遍。
理由：*"没有 systematic audit，code drift / spec drift / quietly disabled detector / forgotten bug fix
都会潜伏到 rerun 数据里，reviewer 挑出来时 too late"*；按 topic 组织 reviewer 找 lifecycle phase 不直观，
改按「实验设置→run→结果→分析→发表」组织**直接映射 paper Methods §3**。

**§116.7/§116.8**：`master_bug_catalog` 全量回填 —— Phase 0 historical fixes（§5-§90 sweep）
从 compact table 展开为 proper subsections，**33 atomic entries B-39 to B-80** + umbrella sub-entries
（B-46a-g §33 Reddit / B-49a-c §39 B0 startup / B-50a-d §40 proxy_api / B-52a-h §42 audit /
B-54a-i §45 pre-launch / B-55a-b / B-56a-e / B-61a-b / B-67a-b）；catalog 从 ~960 → **1318 行**。
**未来纪律 = 每次写笔记 `[bug]` 立刻 add catalog subsection（backfill 是 lossy work）**。

**§116.9**：13 个 catalog entry 从 CONFIRMED 改 **FIXED**（代码在 commit 3c15cd7 已修但 catalog status stale）：
B-01/02/03/04/05/07/09/11/17/18/32/33/35。
**B-35 实际代码修复**：`auth_refresh.should_refresh()` 加 `seconds_since_refresh` kwarg + time-based check
（default **1200s，低于 PHP `gc_maxlifetime=1440s`**）；runner 加 `_auth_last_refresh_ts` per-site tracking。
理由：pre-fix 时 `max_step=30 × 60s/step` 的长 episode 可能 mid-episode PHP session 过期
→ long-episode auth expiry contamination。

**§116.14**：paper §3 mode definition **必须 explicit 写入** §94 的 `SoM max_marks=200`
（原 80，曾导致 B1 Reddit DOM>SoM mode reversal）与 `current_viewport_only=True`。
理由：这两条 setup 参数在 pre_rerun_audit 与 paper §3 都 **under-referenced，但直接决定 mode 语义**。

### 7.5 submodule 可复现性（§104，04-28）

VWA submodule fork 到 `Quarkgluonmixture/visualwebarena` 的 `p79-patches` branch
（**方案 A fork 而非 B patch file**），主 repo pin 到 `16b60d7`；两 commit 拆分
`3f9ceca` runtime patches + `16b60d7` setup + extra task configs。
理由：submodule 内 7 modified + 4 untracked configs **主 repo 完全不记录（ignore = dirty）**
→ 新机器 clone 后 patches 全丢，**paper-grade reproducibility 断**；
Myriad 加入 + paper supplemental code 投稿使 A 决胜。

---

## 八、实验编排 / watchdog / 数据安全

### 8.1 数据丢失事故与由此定的规矩

- **§34（04-12）**：**永久移除** `restart_queue_b1_serial.sh` 的 `--clean` 选项；
  删除 task 结果统一用 `clear_tasks.py` 精确清除。
  理由：`--clean` 对所有三站 RUN_ID 执行 `rm -rf`，**已造成 classifieds 702 episodes 全丢**。
- **§58（04-14）**：queue 脚本加全局 `ACTIVE_RUNNER_PID`，cleanup trap 时一并 kill runner；
  `is_condition_complete` 发现 summary 存在但 done < total 时**删除 summary 并返回未完成**。
  理由：kill 脚本时 cleanup 只杀 watchdog/gallery **不杀 runner**（job_pid 是局部变量）→ 孤儿 runner
  → 重启后两个并行 runner → **CUDA OOM / 重复数据**。
- **§104（04-28）**：`queue_baseline.sh` 的 resume-target glob 从 `${CFG_NAME}_*` 改为
  `${CFG_NAME}_[0-9]*`（仅 match digit suffix）；dirty run dirs 移到 `_archive/`。
  理由：glob 曾 match 到 rename 的 `B0_dom_shopping_20260428_dirty_no_reset` 当作 resume target，
  **runner 写入 dirty dir 再次污染**。
- **§76（04-18）**：summary **原子写入** —— 先写 `.json.tmp`，fsync 后 `os.replace()` 原子 rename。
  理由：直接 `open('w')` 在 OOM kill/SIGKILL 时文件不完整 → resume 时 `json.load` 失败 → task 被重跑
  → **step JSONL 追加重复记录**。
- **§85（04-20）**：`run_id` 追加 `_uuid4[:6]` 防秒级时间戳碰撞；`write_condition_meta` 改原子写入。

### 8.2 站点 reset 与 auth（P79 最长的承重墙）

- **§38（04-13）**：**condition 之间必须 reset 站点**（前一 condition 的写操作污染后续 condition 站点状态）。
  classifieds 有 **named volume** 需 `docker rm -f` + volume rm + compose up + 90s + `init_db.sh`
  + PHP session 86400s；reddit/shopping 数据烘焙进镜像，`docker rm -f` + `docker run` 即可。
  理由：B0 SR 预计更高，污染风险放大；**paper-grade ablation 需 fair 起始状态**。
- **§39（04-13）**：`reset_vwa.ps1` 追加手动执行 `init_db.sh` + 设置 `session.gc_maxlifetime=86400`（24h）。
  理由：`docker-entrypoint-initdb.d` **按字母序执行导致 schema 未建**；
  **PHP session 默认 24 分钟 GC 使 Playwright cookie 有效但服务器端不认**。
- **§75/§82（04-17/04-20）**：per-episode auth refresh —— `p79/utils/auth_refresh.py` 共享模块
  （从 watchdog 提取），runner `_run_episode()` 前按 `auth_refresh.interval`（默认 5 episode）per-site 计数刷新。
  §75 时默认 `enabled: false`，**§82 改 True 且全站**（classifieds/reddit/shopping/shopping_admin）。
  理由：原每 condition 刷新一次（192 task），Magento session ~5-12 min 过期 → **96% task 无登录态**；
  §82 前 classifieds/reddit 从未刷新 cookie → 被 watchdog 误判 NOT-LOGGED-IN 清除 →
  **queue 死循环（跑→清→重跑→清）**。
- **§75**：auth retry 改为**最多 5 次指数退避（10→20→40→80s）**并 ntfy 每次重试，不再 2 次失败即 fatal exit。
- **§104（04-28）**：**Post-reset proactive auth refresh** —— 3 个 queue script 加
  reset 之后 launch 之前的 `refresh_site_auth()`（reset → 15s settle → refresh → launch）。
  理由：reset 清 server-side session 但本地 `.auth/<site>_state.json` 还是 reset 前 cookies →
  runner load 后 server 不认 → **浪费 ~3-4 episode**；
  **双重防御 = structural（proactive）+ watchdog auto-clean（reactive）**。
- **§97 第九轮（04-26）**：`auth_refresh` AR-7 —— login 后检查 `still_on_login`
  （`login_path in final_url`），若仍在登录页则**不写 storage_state 并 `sys.exit(2)`**，
  caller 区分 rc=2 给 'LOGIN VERIFICATION FAILED' 警告。
  理由（真 bug）：**login 失败但 storage_state 仍写空 cookies，caller 误判 True，后续 episode 还是 NOT-LOGGED-IN**。

### 8.3 网络与域名（诊断规则值得单记）

- **§75（04-17）**：Chromium 必须注入
  `--host-resolver-rules=MAP metis.lti.cs.cmu.edu 100.95.81.103`（**6 处 Playwright launch 点全覆盖**）。
  根因：Magento base_url 配为 `metis.lti.cs.cmu.edu`，所有到 `100.95.81.103:7770` 的请求被 302 到该域名。
- **§75 诊断规则（可复用）**：**`curl` 返回 200/302 ≠ 站点正常**，必须检查 Location header 目标域名是否
  DGX 可达；`python http.client` 不 follow redirect 可精确判断。
  （原现象：curl 不 follow redirect 返回 302 造成**假象正常**，Playwright/urllib follow redirect 则域名不可解析连接失败。）
- **§103（04-27）**：Magento base_url 改为 IP（`web/secure/base_url` + `web/unsecure/base_url`
  = `http://100.95.81.103:7770/`，shopping_admin 同理 7780）+ `cache:flush`；
  4 个 queue script **revert 之前加的 metis hostname override**。
  根因：metis hostname 现解析到 CMU 公网 `128.2.205.52`（拒连）→ customer-data API 失败 →
  **Knockout 渲染 guest dropdown 即使 logged-in**。
- **§43（04-13）**：`OPENAI_API_KEY` 的 DUMMY 占位符必须**显式覆盖**
  （`if not _cur_key or _cur_key.startswith('DUMMY')`），**不能用 `os.environ.setdefault`**。
  根因：shell 脚本先 `export OPENAI_API_KEY=DUMMY`，setdefault 看到已有值不覆盖 →
  `.auth/openai_key` 真实 key 失效 → **ua_match evaluator 401**。

### 8.4 watchdog 的六次修

| § | 修什么 | 根因 |
|---|---|---|
| §14 (04-08) | 新增 step_000 DOM 登录标志检测，连续 ≥3 个 task 失登录触发 ntfy | session 运行中过期无检测手段，**数据静默污染** |
| §41 (04-13) | 孤儿 steps/artifacts 双重清理（`--clean-orphan-artifacts` + 10 min mtime 保护） | 旧 auto-retry **只删 summary**，遗留孤儿 steps JSONL + artifact；gallery 以 steps JSONL 为主索引 → **幽灵 episode 卡片** |
| §42 (04-13) | watchdog kill 必须加 OUTPUT_DIR 过滤 | 防误杀并行实验 |
| §84 (04-20) | 跨站 NOT-LOGGED-IN 检测**只查 active tab**（first_line 按 `' \| '` 分割取第一段） | 跨站 task 的 step_000 DOM 首行含所有 tab 标签，`any(kw in first_line)` 在**非活动 tab 标签**中命中 → Shopping 页的 'Sign In' 触发误判 → **反复删除正常 episode** |
| §87 (04-21) | 主循环**每轮 poll** 都调 `_prune_stale_condition_completions()`（原仅启动时一次） | stale `condition_summary_v2.json` 使 3 个 condition 全被标 COMPLETE → `_build_status_report()` 返回 None → **ntfy 30min 周期报告与 PERSISTENT-ERROR 通知全部静默** |
| §97 第七轮 (04-26) | 4 个 session 变量加入 `_save_state/_load_state`；`error_retry_counts` 在 condition 完成时按前缀 reset；state file 加 `_schema_version='v2'` | `session_contaminated` 只在进程内存，**watchdog 重启即丢** → 已追踪的 NOT-LOGGED-IN episodes 在 login restored 后不会被 auto-clean → **数据污染**；`error_retry_counts` 永久累积（state 已有 180 entries）使同 task 跨 condition 重跑时 **retry 配额已耗尽** |
| §104 (04-28) | `ZeroDivisionError` fix（`cond_rate = (cond_succ/cond_total) if cond_total else 0.0`） | auto-clean 11 个 NOT-LOGGED-IN episode 后 `cond_total=0`，下游无 guard 直接 crash（commit 4b1ff66） |

### 8.5 错误分类：什么算 noise，什么算 fatal

- **§76（04-18）零步错误分层**：`benchmark_noise=True` 的零步错误 runner **直接跳过**
  （交 watchdog `MAX_NOISE_RETRIES=3` 管理），仅非 noise 零步错误保留一次 runner 重试。
  理由：runner 无条件重试 `steps=0+error`，叠加 queue `MAX_RESUME_ATTEMPTS=10`
  可使同一持久错误（proxy 403 / Docker 缺陷）**被重试 10+ 次**。
- **§76**：**proxy API 403（额度耗尽）判为 fatal 不是 benchmark noise** —— `metrics.py` 增加
  `model-api` / `execute-api` 前置排除（原被 `'forbidden'` 关键词误分类为 `anti_bot_or_blocked`），
  runner 增加 proxy 403 专项 fatal 检测立即 raise；watchdog AUTO-RETRY 通知按 poll 周期合并（上限 20 行）防刷屏
  （如 35 个 403 × 3 = 105 条）。
- **§86（04-20）auto-retry 三漏洞**：runner health check `_error_title_patterns` 加 `'osclass error'`
  （step 0 拦截）+ mid-episode `_INFRA_TITLE_PATTERNS` title 检查抛 `RuntimeError('site_infra_error:')`；
  `metrics.detect_benchmark_noise` 加 `site_infra_error` → noise 类；
  runner resume **统一跳过所有零步错误**（不区分 noise/code_bug）全交 watchdog；
  `metrics.py:65-66` model-api/execute-api 从 `(False,None)` 改为 `(True,'api_infra')`。
  暴露案例：B0 classifieds task 154/155 —— Osclass DB 短暂宕机 → **agent 优雅退出 → error: null
  → 绕过重试链 → 永久卡在错误结果**。
- **§81（04-19）Wikipedia ZIM 三层修复**：`tasks.py::_placeholder_mapping()` 读 `WIKIPEDIA_ZIM_VERSION` env
  把 `wikipedia_en_all_maxi_2022-05` 重写为 `2025-08`；runner + vwa_wrapper 在 `reset()` 后检查所有 tab title
  是否含 'not found'/'404' 命中则抛 `RuntimeError('start_url_content_error')`；
  `metrics.py` 加对应噪声类别让 watchdog 自动重试。三个 queue 脚本均 export `WIKIPEDIA_ZIM_VERSION`。
- **§81 follow-up（04-29）**：B0 dom shopping **task 345** 的 `start_url_content_error`
  **不算 §81 regression**，论文 footnote 处理为
  *"1/466 task excluded due to upstream Wikipedia ZIM data drift independent of P79 setup"*。
  依据：task config start_url 已是 2025-08 路径（§81 修复仍工作），但 task 345 引用的 image asset
  `Country_calling_codes_map.svg.png.webp` **在 2025-08 ZIM dump 里也不存在**
  （所有路径变体 `.svg.png.webp` / `.svg.png` / `.svg` / base name / `I/m/` thumbnail prefix 均 HTTP 404）
  —— 是 Wikipedia upstream 2022→2025 删/改了该图片。
- **§87（04-21）Evaluator 脏 page 修复**：第 1 次用 agent 原始 page，失败后 `page.context.new_page()`
  创建 fresh page（**同 context 共享 cookies/auth 无脏状态**），第 2-3 次重试用 fresh page，
  finally 统一关闭防泄漏。根因：agent 交互后 page 残留脏状态（未提交表单 / beforeunload handler /
  pending XHR）→ evaluator `page.goto()` **永久 `net::ERR_ABORTED`**，
  代码内 3 次重试无效因**同一 page 状态不重置**；对 program_html 完全安全，url_match 第 1 次已用原始 page。
- **§74（04-17）**：WA `evaluator_unavailable` 修复 —— `VwaEvaluator.__init__()` 对缺失站点填充
  `https://example.com` dummy URL。根因：`env_config.py` 要求 WebArena **全部 7 站 URL 非空**，
  但只部署了 3 站 → **全部 episode `error=evaluator_unavailable` 评分 0**。

### 8.6 并发与调度

- **§74（04-17）调度策略**：B0 WA 先行（API 不占 GPU）与 B1 VWA reddit 并行 —— B0 WA 与 B1 共享
  shopping/reddit Docker 容器（同端口）但当前无站点冲突；B0 VWA 等 B1 和 B0 WA 都完成后再启动。
  **不同 site 可并行，同 site 不可** —— 后来固化为 CLAUDE.md **hard rule 1**。
- **§104（04-28）hard rule #3 引入**：禁止裸 `python scripts/run_experiment.py`，必须走 queue script。
  理由：queue 处理 reset（race-safe）+ env（PROXY_API_KEY / VWA endpoints / CUDA workaround / wikipedia zim）
  + watchdog 启动 + idempotent skip；**裸 runner 实证导致 B0 dom shopping pilot 0 reset 数据污染**。
- **§59（04-14）**：BLIP-2 懒加载 `_ensure_captioning_fn()` —— 仅 `page_image_query` 任务触发；
  CUDA 可用时轮询空闲 VRAM（阈值 18GB，间隔 30s），10 分钟内未满足则 fallback CPU。
  理由：GB10 共享 GPU 仅剩 ~5GB，BLIP-2 float16 ~15GB 无法加载；**强制 CPU 则 20+ min/task 会被 watchdog 杀死**。
- **§38（04-13）**：B0 三模式 DOM → reset → SOM → reset → Vision **共享同一 RUN_ID**；
  每模式用独立单模式 temp config（Python regex 替换 `observation_mode`）。理由：共享 RUN_ID 让分析脚本无需改动。

### 8.7 目录重构（§99，04-26）

顶层只留 `run_experiment.py` / `preflight_v2.sh` / `vwa_env*.sh*` / `README.md`；
新建 `queues/`（5）与 `maintenance/`（21）；删除 `scripts/dgx/ utils/ dev/ cloud/`
（cloud sagemaker **从未启用**，commit 7fce178 已标 archived）；**26 文件 `git mv` 保留 history + 33 文件 sed 替换路径**。
理由：`dgx/` 命名误导（里面没有 DGX 特化逻辑，混了 5 类内容）；子目录粒度不一。

**§99 明确不动的两处**：实验笔记 §42-§97 中对 `scripts/dgx/...` 旧路径的引用（**属事件记录的一部分**）；
`docs/checkpoints/B0_B1_call_chains.md` 中指向已删脚本的内容（属过期文档清理，独立工作）。
> 这条确立了一个原则：**笔记是 chronicle，旧路径是历史事实，不回改。**

---

## 九、分析管线与数据校验

### 9.1 canonical 单一源（最重要的一条）

**§97（04-26）**：`adjusted_success` **单一源（canonical）** —— runner 在 `_run_episode` 末尾计算
`adjusted_success` / `fp_reason` / `has_effective_action` 并写入 `episode_summary`，
**下游 5 处加 fast-path（已有 column 就跳过重算）**；`rederive_episode_summary.py` 也写 `adjusted_success`。
理由：**消除多处独立重算导致的口径漂移**。验证：B0 cls cross_rep 数字与 §97 第二轮一致。

### 9.2 脚本化取代手工

- **§91（04-24）**：`validate_run.py` **22 项检查分 8 组**（文件存在性 C01-02 / 结构一致性 C03-06 /
  episode 覆盖率 C07-08 / episode 完整性 C09-11 / step 完整性 C12-15 / scaffold 安全 C16-18 /
  artifact 完整性 C19-20 / 分析新鲜度 C21-22），退出码 0/1/2。
  SKILL.md Step 2 从 ~200 行手动流程精简为"跑脚本，FAIL 则阻断"，保留脚本不覆盖的手动检查
  （digest dry-run 比例 / GLM fallback 有效性）。理由：原由 Claude 手动多 agent 扫描，**慢且不可复现**。
- **§93（04-24）**：22→**27 checks** —— Group 9 Temporal Analysis（C23 时序 SR 退化 / C24 Auth 漂移 /
  C25 Reset 污染）+ Group 10 Data Consistency（C26 summary.steps vs JSONL 行数一致性 / C27 零成本 episode）
  + C22 增强（digest >105% 超额检测 + task_id 去重统计）。
- **§92（04-24）**：`diag_pattern_match.py` 实现 **13 条规则（P1-P8, P10-P14）**；
  **P9（16px 节点误触发）因需 DOM 级分析暂不实现**；新增 P12（从不翻页）/ P13（搜索代替浏览）/ P14（URL 自环）；
  DOM/SoM 用 `action.element_id` + `element_bbox` 像素坐标，Vision 用 `action.coordinate` 归一化 [0,1]，
  规则自动区分模式。理由：原由 Claude 逐 step 手动判断，耗时且不可批量。

### 9.3 §97 的五个真 bug 与三个口径修正

**cross_representation 5 个真 bug**（§97 04-26）：
- **F1** `_success_vector` **三态化（None/True/False）**——"未测"不再算"失败"
- **F2** global summary 透传整个 `a2_summary`
- **F3** `_build_oracle_rows` NaN cost + steps 守卫
- **F4** `_mark_false_positives` `ff_col` 缺失双守卫
- **F5** `_agent_finished_known` 列对齐 canonical（缺数据时 eval_fp 不过判）

症状：单站 run 的 `cross_representation_summary.json` 丢字段（`feature_oracle_ceiling` / `feature_gap` /
FP counts）；**A3 exclusive sets 把"未测"算成"失败"使 `only_X` 虚高**；oracle decomposition NaN cost / 0-step 行为未定义。

**latency 口径修正**（§97 第二轮）：`net_saving_latency` **不再加 `router_overhead`**（避免 double-count）
且改用新增的 `avg_total_latency_ms`（端到端 episode latency）而不是 **P95 单步 latency**；
`compute_wasted_cost` 加 `adjusted_success` 参数（**FP episode 也算 wasted**），新增 `wasted_cost_usd_adjusted` 列。
理由：P95 单步 latency 与 episode 总 router_overhead **单位不一致 + double-count**。

**router threshold 必须带 out-of-sample 估计**（§97 第二轮）：`_optimal_threshold` 加 **LOO-CV + bootstrap CI**
（`threshold_loo_mean/std`, `sensitivity_loo`, `specificity_loo`, `threshold_ci_lower/upper`, `validation` 字段）；
`c10_composite_signals` 加 `validation='in_sample'` + rank + `n_combinations_searched` 列 + stdout disclaimer。
理由：原 C10 composite 'best AUROC' 是 **in-sample 搜索最优，直接报会误导**。

**confidence 分析四修**（§97 第二轮 CC-4~CC-7）：benchmark 从 `run_dir.parts` 推断；
per-site adjustment 不再硬编码 fallback `'classifieds'`；`overall_usable` 加 `mode_invariant!=False` 守卫；
`_load_episode_summaries` task_id 缺失/无效则 skip **不 collide 到 -1**。
理由：之前硬编码 vwa 使 **WA confidence 分析的 `na_task_ids` 全错**。

### 9.4 命名混淆澄清（§98）

- `analyze_cross_representation` = **同一 run 内三模式（DOM/SoM/Vision）对比**，不是跨站
- 跨站 = `aggregate_cross_site`
- 跨 baseline = `compare_b0_b1`

**§98 watchdog 自动触发跨 run 分析**：condition 完成时判一次（幂等无副作用）——
`compare_b0_b1`（条件=兄弟 baseline 同 site run 也有 ≥1 condition 完成）+
`aggregate_cross_site`（条件=同 baseline 在 ≥2 个 site 都有 ≥1 condition 完成，self 也算）；
run_id 正则 `^(B[01])_(?:wa_)?3mode_(.+?)_(\d{8})$`；benchmark 从 `results/{benchmark}/phase1/` 路径段判定，
**VWA 与 WA 严格隔离**；同 site 多 run 取 date 最新。

**§98 未做的两项**：不自动触发 `/write-analysis`（`docs/analysis/` 文档生成仍手动，**避免跨写入冲突**）；
watchdog 仍只看单 run_dir，不会"看到 B1 reddit 完成 → 触发 B0 reddit run_dir 下的分析"（按需另启 watchdog）。

### 9.5 成本 / 延迟 / 能耗计量

- **§8（04-07）**：latency（`total_latency_ms` / `p95_step_latency_ms`）**不作为跨 condition 对比指标**。
  理由：受 GPU 争抢影响；**DOM 与 SoM 串行执行 —— DOM 先跑争抢少，SoM 后跑争抢多，两 condition 延迟不具可比性**。
  → **§31/§97 连带**：B1 latency **必须在独占 GPU（Myriad）上重跑 final B1**，DGX 共享 GPU 污染 latency。
  ⚠️ 待跨批核对：当前 canonical latency estimand 见 memory `project_cost_latency_canonical_estimand`（retry-adjusted），
  演化链在后批。
- **§35（04-12）**：成本模型补三类细分字段 —— `input_image_tokens` / `input_text_tokens`
  （用 `processor.image_token_id` 匹配分类）；`latency_ms` 拆 `obs_prepare` / `preprocessing` / `generate`；
  `estimate_step_flops()`（ViT encoder + LLM prefill + decode）**仅供分析**。新字段用 `meta.get(..., 0)` 向后兼容。
- **§48（04-14）**：`EpisodeSummaryV2` 加 `total_input_cost_usd` / `total_output_cost_usd`；
  `aggregate_condition_metrics` 加 `avg_input_cost_usd` / `avg_output_cost_usd`（**空返回也加**）；
  新建 `aggregate_cross_site.py` + `compare_b0_b1.py`（+ `mirage_effect.csv`）。
- **§69（04-16）`cost_usd` 双用途设计**：(A) router 决策用**统一 scalar**
  `step_total_cost = token_cost + router_overhead + obs_prepare_cost`；
  (B) 论文报告**拆分展示** `cost_usd.model` / `.router_overhead` / `.obs_prepare`。
  理由：**让读者看见"SoM 贵在哪"，同时 router 只需一个 scalar**。
- **§17（04-09）B0 计价**：`backend_type.startswith('api_')` 自动匹配 `cost_api` 定价
  = **$0.001/1K input + $0.005/1K output**；图片必须 base64 data URL（**不支持外部 HTTP URL**）；
  响应 content 支持 string/list 双格式。理由：学长提供的自定义代理 API 是 **Anthropic Messages 风格非 OpenAI 兼容**。
- **§17（04-13）**：B0 **thinking 模式不支持，不做代码改动** —— Bedrock 上 Qwen3 **静默忽略
  `enable_thinking: true`**，`meta['reasoning_content'] = None`。§45 Fix-1 顺带从 config 删除 `enable_thinking`。
- **§97（04-26）busy-wait**：B0 旧数据 `total_latency_ms` **不含 busy-wait stalls**，
  标记 `busy_wait_total_ms_unknown_pre_fix=True`，论文披露 *"B0 latency excludes busy-page wait stalls"*；
  新数据起带 `busy_wait_total_ms` 字段（**RU-4 只能补标志位不能追算**）。
- **§97 第三轮 energy_tracker 四修**：**ET-5** `__init__` prime `psutil.cpu_percent(interval=0.0)`
  丢弃首次返回的 0（**之前第一步能耗严重低估，修后仅新数据准确旧数据只能披露**）；
  ET-10 `measurement_source` 加 `kwh_per_step` 优先级；ET-12 删 `record_model_load` 死代码；
  ET-3 `_power_samples` 改 `deque(maxlen=600)` 把 O(N²) prune 降为 O(N)。

### 9.6 防回归基础设施（§97 第六轮）

- `runner.py`（1637L）拆为 `p79/experiment/runner/{__init__,helpers,main}.py`（278+27+**1477**L），
  `__init__.py` re-export 保 `from p79.experiment.runner import ExperimentRunner` 不破；
  `history_window` 从硬编码 8 提到 `cfg.agent.history_window`；reference image 由 runner **预 resize 一次**
  （避免 N_steps × resize）。
  > **拆分时引入隐藏 bug**：`vwa_root` path 计算少一层 `.parent` 导致 reference image 找不到（vision tasks 失败），已修。
- `tests/test_runner_smoke.py`（**7 测试保护 §97 invariants** —— PUR 排除 finish / busy_wait /
  energy_partial / aggregate 字段 / canonical import path）
- `p79/experiment/schema_migrations/`（v2 字段 catalog + migration 注册表，**下次 v3 直接 `@register('v2','v3')`**）
- `Makefile` 12 个常用命令一键化

### 9.7 §97 的审计边界（哪些查了、哪些没查）

**判定为干净、无需修的文件**：`som.py` / `router.py` / `state_change.py` / `io_utils.py`（restart dedup 正确）/
`conditions.py`（留 CD-1 注意）/ `action_utils.py` / `diag_pattern_match.py`（仅 cosmetic DPM-2）/
`experiment_watchdog.py`（经 §76/§82/§84/§86/§87 多次修复后稳定，`MAX_NOISE_RETRIES=3` +
`MAX_CODE_BUG_RETRIES=2` 防御 OK）。

**明确未审/跳过的范围**（判定只影响 supplementary，此说明**在 §97 中出现三次**）：
`validate_run.py` / `glm_*.py` / 各 `analyze_*.py` 专题脚本 / `scripts/dgx/*.sh` 部署脚本 /
utils 硬件 hack / reference image 与 SoM 工具脚本。

**CD-1 待办**（§97 第四轮）：`conditions.py` phase2 的 `_load_best_condition_from_phase1`
用 **raw success_rate** 排序，**Phase 2 启用前需切到 adjusted**。当时 phase2 未启用故不改。**未决**。

---

## 十、Routing 信号（Phase 2 的种子）

- **§7/§15（04-07/04-09）confidence 提取 6 指标**：`mean_logprob` / `min_logprob`（绝对信心）、
  `mean_margin` / `min_margin`（top-1 vs top-2 区分度）、`mean_entropy` / `max_entropy`（分布散度）。
  理由：**entropy 与 margin 互补 —— margin 只看 top-2 差距，entropy 捕获多候选散射场景**。
- **§15（04-09）**：routing 信号采用 **entropy + margin 互补而非 raw logprob**。
  依据 8 篇文献综述：**raw logprob 在小 VLM 上过度自信，calibration 后才可用**。
- **§24（04-10）Phase 2 路由信号优先级**：用**免费行为信号**（`page_unchanged_streak` /
  `no_progress_streak` / URL 重访 / action diversity）触发 Phantom-SoM ↔ Full SoM 升级；
  **PRM verifier(7B) 因太贵不用**；**learned router 因需训练数据搁后**。
  核心前提 = **token-level AUROC=0.497**（需要替代信号）；现有循环/停滞检测只需从"触发 retry"改为"触发模式升级"。
  > 台账明标：文献数字（R2D2 50% 错误减少 / TGPO AUROC=0.74 / WebArbiter +9.1pp / FrugalGPT 98% 成本削减）
  > **来自文献非 P79**。基于 ~1400 篇文献系统综述（2023-2026）。
- **§26（04-10）**：verbalized confidence 加入三模式 prompt JSON schema（`'confidence': 0.0-1.0`），
  但 **C5（logprob 模式不变性）与 C8（behavioral 累积）刻意不加 verbalized**；
  routing verdict 字段扩展为 `signal_discriminative = token ∨ behavioral ∨ verbalized`。
- **§19（04-09）**：把 **activation-level cosine gap** 列为 Phase 2 表征层路由信号候选（**仅 B1 白盒可行**），
  绕过输出层校准问题；M4 两阶段 = planner(cosine select) + grounder(generate)。
  > 台账明标：文献 'Tool Calling is a Linear, Steerable Circuit'（ACL 2026）在 Qwen3-4B 上的数字
  > （15 tool → 10 方向 PCA 90.2%；cosine gap 低→错误率高，捕获 92% 错误；steering L23+ 层 80-93% 准确切换 tool；
  > **内部 77-89% 正确但输出 3-61%（"知道但说不出"）**）—— **以上数字来自该论文，非 P79 测量**。

---

## 十一、WebArena 集成（§21 / §71 / §73 / §74）

**动机**（§21 04-09）：VWA **67% visual** 放大 DOM 劣势；WA 同框架同基础设施**零部署成本**。
引入免费三站 shopping 192 + shopping_admin 182 + reddit 106 = **480 tasks** 提升 non-visual 占比。

**唯一 key 约定**：VWA/WA 混跑必须用 **(benchmark, site, task_id) 三元组**作唯一 key ——
task_id 冲突 shopping **135 个** + reddit **9 个**，单靠 task_id 会碰撞。
**WA 5 个跨站任务（shopping+reddit）按 primary site 归入 shopping**，避免同一 task 在两站重复计数。

**集成关键约定**（§71 04-16）：
- `_site_matches` 用 `__shopping__` **精确匹配**（防 shopping/shopping_admin 碰撞，**末尾双下划线阻断子串误中**）
- `vwa_wrapper` 加 benchmark 参数动态切 DATASET + `required_vars`（WA 需 SHOPPING_ADMIN，不需 CLASSIFIEDS/WIKIPEDIA）
- `analysis.py` 三个 loader 参数化 benchmark，且 **WA visual 返回空集**（无 visual task，消除 FP 风险）
- `shopping_admin` 在 gallery site order/regex 中**列在 shopping 前**防贪婪匹配

**§73（04-16）DATASET 环境变量必须无条件覆盖**（`os.environ['DATASET'] = dataset`），
不能用 `if DATASET not in os.environ`；WA queue 脚本必须 `export DATASET=webarena`；
benchmark 从 run_dir 路径推断（`_infer_benchmark`）**而不是从 `pivot['benchmark']` 列**。
根因：`vwa_env_remote.sh` export 的 `visualwebarena` 会**毒化 WA 运行**；
CSV 不含 benchmark 列导致 **WA shopping 85 个任务被 VWA visual 列表误标**。

**§73 Gallery**：HTTP server 从 `results/{benchmark}/phase1` 提升到 `results/`，
新增 `generate_combined_gallery` 跨 VWA+WA 合并（站点标签加 `vwa:`/`wa:` 前缀 + 独立颜色 badge）；
只保留两个固定入口（B0/B1 各一）。**intent key 需加 `wa:` 前缀防碰撞。**

---

## 十二、Mechanistic（A1 期活跃 —— 现已整体搁置）

> **读这一节前先知道**：CLAUDE.md 记载 advisor 2026-05-14 *"mechanism 部分先不要管了"*，
> §5（activation patching / layer probe / logit lens / SAE）整个暂搁。**以下是搁置前的裁定，
> 对当前 paper scope 不生效**，保留是因为它们解释了"为什么当时那么做"。

- **§111.1（05-06）三件套 hand-rolled**（因 **nnsight wheel 在 aarch64/GB10 build 失败**）：
  `p79/mechanistic/{extract_hidden_states, linear_probe, activation_patching}.py` +
  `scripts/mechanistic/{run_stage1_pilot, run_stage2_patching_pilot, run_stage2b_continuation_pilot}.py`；
  **对齐 paper-grade prompt structure**（复用 `Qwen3VLAgent._make_dom_prompt/_make_som_prompt` +
  `_extract_text_marks` + image LANCZOS resize 1024）。
  裁定依据：**archived 数据虽 SR 受 Phase A bug 影响，但 hidden state 是 input-conditional 跟 reward 无关**，
  可用于 mechanistic。
  commits: e25d8f3 / 126977a / a771c90 / c49369b / 6ddd646 / 9fe3d84。
  > flag：§111 task 0 L11 patching 93% flip 作为 paper §5 representative finding 的说法已 **RETRACTED §117.4**。
- **§117.5（05-09）2x2 selection-bias control**（commit 9d67387）：Cell A forward×strong(24) +
  Cell B reverse×reverse(15) 已有，新增 Cell C forward×reverse-tier(15, qsub 335339) +
  Cell D reverse×strong-tier(24, qsub 335340)。
  **decision tree**：C+D 类似 L17 disruption → mechanism universal；C+D null/weak → task-class-specific 需 caveat。
  理由：composite 是 task-text-only（无 patching leak）但 **plausibly 与 patching effect size 相关** ——
  bidirectional finding 可能反映 mechanism universality **也可能是 task curation correlation**。
- **§108.9（05-01）SteerMoE**（Fayyaz 2026 ICLR）定位为 **paper §8 future work methodology template，不 self-probe**。
  3 个 methodological barrier：B0 expert activation 通过 proxy API **不可见**（不暴露 router logits）；
  local deploy Qwen3-VL-235B-A22B 需 **~120GB VRAM**（4×4090 ≈ $400-600）；
  用 Qwen3-30B-A3B（no vision）作 architectural proxy 但**不能直接 claim B0 phantom mechanism = expert routing**。
- **§108.19（05-01）Zoom 4 anchor**：加入 *Tool Calling is a Linear Steerable Circuit*（ACL 2026, Qwen3-4B）
  —— **唯一 4B-tier mechanistic anchor**，平衡此前全偏 B0 路径的 Kaduri + SteerMoE；
  paper.bib 56 → **57 entries**（`anon2026toolcalling`）。
  理由：hidden state PCA + cosine gap + L23+ steering 全部是 internal probe 方法直接 fit Zoom 4；
  B1 = Qwen3-VL-4B **同 base LM**；ACL 2026 finding（action selection 线性可分 / argument generation 非线性）
  给 §1 cascade 顺序 mechanistic 理由。

---

## 十三、SoM occlusion probe —— 一次 claim 边界划得极干净的实验

### 13.1 probe 设计（§100，04-26）

给 model 一张截图 + 一个 prompt（*"list all visible link/button/heading text content,
do NOT include numbered ID tags, do not guess"*），ground truth 从 axtree 提取所有 link/button/heading 的 label。
**三对照 mode** = mode-SoM（当前带标签截图，含 occlusion bug）/ mode-NoMarks（原始 screenshot.png）/
mode-WithText（SoM 截图 + prompt 附完整 `[SOM_MARKS]` 文本）。
**测试集 5 张图密度梯度**：classifieds_task_14 step1（33 marks）/ classifieds_task_15 step1（41）/
reddit_task_164 step14（54）/ reddit_task_6 step0（111）/ reddit_task_164 step0（128）。
动机：直接测量"模型从 SoM 截图能 extract 多少信号" + "标签是 destructive 还是中性"，以解释 B1 reddit SoM < DOM 反转。

### 13.2 honest claim 边界（§100 —— 五条可 claim + scope 限制，一字不丢）

1. 在 P79 SoM 设计下（标全部 + 固定青色 + simple placement）**SoM 标签对 B0/B1 的 OCR 都是 destructive**
   （-60pp on reddit_task_6）
2. B1 视觉 capability 在无标签下接近 B0
3. B1 在密集 SoM 截图上**倾向 attend 高对比度数字 ID 而非内容文字**（num_ids 0→446 随密度）
4. B1 给 `[SOM_MARKS]` 文本 fallback 后**完全 recover**（reddit_task_6 81%）且 attention 不再被数字 hijack
5. **text-over-vision bias 在 B1 上比 B0 更强**

**scope 限制**：仅适用 P79 SoM 设计（见 §101），**不能直接 generalize 到 VWA 原版 SoM**。

### 13.3 §101 的依赖度分析（哪些 claim 独立于设计选择）

**主 mechanism claim 全部独立于 P79 SoM 设计选择**：lazy minimization / B1 视觉 capability ≈ B0 /
text-over-vision bias / capability×density×task-category 交互 / visual_share 多样性。
理由：各 claim 的测量来源（NoMarks probe / DOM-Vision 对比 / 4 cell × 4 cat subset）**与 SoM 标注范围无关**。
**仅具体量级是 P79-specific**：reddit B1 -1.9pp / occlusion -60pp / §96 误点率。

---

## 十四、「不修，只披露」清单 —— 最容易被重新提起的一类

这一节单列，因为它是**防重做的最高频命中区**：每一条都是"看起来该修但已裁定不修"，理由不记住就会重提。

### 14.1 判定为模型行为 / 结构性限制

| § | 项 | 不修理由 |
|---|---|---|
| §50 | B0 scroll 方向混乱 | **无可靠修复**，归模型行为；论文已知局限披露（§66 定量 + §72 跨 API 验证） |
| §55 | delete 成功信号缺失（flash 消息时序） | 判定为**非结构性缺失** |
| §51 | `current_viewport_only=True` 下 combobox 需在 viewport 内才出现在 AXTree | 已知局限；agent click Publish Ad 后若未 scroll 可能看不到，需多走一步 scroll。publish 页 combobox 需 auth state（未登录 302 回首页，task config 已含 storage_state） |
| §80 | DOM 元素级全有/全无 | ratio bug 修后**残余不公平是结构性限制**，非 bug |

### 14.2 判定为"所有 condition 一致，不产生偏倚"（§97 第三/四轮）

`energy_tracker` **ET-1/2** `_average_measured_power` 跨 step 边界（所有 condition 一致）；
**ET-4** RAPL wrap-around 不防（**aarch64/GB10 无 RAPL 不触发**）；
`som.py` **SOM-1** max_marks 截断时 OPTIONS 注解丢失（§94 已 80→200 少触发）；
**SOM-3/8** 边界 ambiguity 与 `degraded_som` 双语义；
`environment.py` **EN-10** `_ensure_captioning_fn` 失败后 `captioning_fn=None` 静默 score=0；
`qwen3vl` **QV-1** type 自动加 `\n` 启发式过粗；**QV-4** `input_image_tokens` 含 reference image；
**QV-8** image position layout 不一致（§44/§46 已知设计）；
`state_change` **SC-7** dead filter `focus_blur` 无害。
统一理由：**影响 supplementary 而非主表，或所有 condition 一致不产生偏倚**。

### 14.3 判定为 blast radius 太小（§116.9，9 个 CONFIRMED bug 不修）

**B-15** finish_wrong_state（已由 §95 FP filter handle）/ **B-16** long step Playwright timeout（blast **1.85%**）/
**B-20** ua_match GPT drift（§3 limitation disclosure）/ **B-21** string_match GPT-judged（disclosure）/
**B-22** program_html selector brittleness **562/1598**（Section 4 cite + future work）/
**B-34** stale auth file mask / **B-36** image compression cliff /
**B-06** SELECT_OPTION arg-drop（blast **0.3%**）/ **B-08** SCROLL silent（blast **0.8%**）。
依据 = **Fix Scope Decision Matrix 按 blast radius 与 paper 影响权衡**。

### 14.4 判定为 FP 可接受（§92 diag 规则已知局限）

**P10** 数值提取含 element_id/坐标等噪声数字；**P11** location 正则过宽（`in\s+\w+` 匹配 'in the'）；
**P6** 颜色词会匹配非视觉语境（如 'Black Friday'）。理由：影响可控，FP 可接受。

### 14.5 §46 审计发现但判定无害

B0 meta 缺 `input_image_tokens`/`input_text_tokens` 细分（不影响 cost 计算，分析脚本不依赖）；
B1 config `include_sites` 需脚本 regex 覆盖（脚本始终覆盖）；
watchdog `.auth/glm` 不存在时 digest 不启动（`[[ -f ]]` 已保护）。

### 14.6 运行中不引入行为变更（§85）

代码审计 **63 个问题中本轮只修 9 项**（P0×3 + P1×3 + P2×3），且**全部限定为向后兼容改动**
（日志/防御性代码，无行为变更）。理由：**B1 正在运行中**。

### 14.7 ⚠️ SoM occlusion —— 三次裁定"不修"，但 C' 也没做

| § | 裁定 | 理由 |
|---|---|---|
| §25/§100 (04-26) | **不轻易改 production `som.py`**；若做 C' 用**独立渲染函数做 isolated 实验** | 标签色 #00BCD4 是 commit 5f0952b 调整后的设计（消除红框颜色 confound），**全局改会污染 in-flight 实验数据一致性** |
| §100 (04-26) | **不修 occlusion bug** | 颜色是 5f0952b 调整后的设计 + 已用 mode-NoMarks 覆盖"完全无标签"极端值，**C' 的边际收益小** |
| §102 (04-26) | **不修 occlusion bug**（标签实心填充覆盖元素文字） | mode-NoMarks probe 已 cover 极端值 + **production 一致性 trumps 修 bug** |

**⚠️ 张力（不调和，两侧并列）**：
- §25 修订（04-26）明写 *"B vs C 直接对比被 occlusion bug 污染，**必须引入 C'** 才是干净对比"*
- 但 §100 Next steps #2 **标 ❌**（不做 C'），理由是"边际收益小 + mode-NoMarks 已覆盖极端值"

⇒ 台账两侧同日并存。结论层记录为：**"必须 C'"是分析层的要求，"不做 C'"是执行层的取舍，
两者未在台账中被显式调和。** 若 reviewer 追问 B vs C 的干净性，这是暴露面。

### 14.8 其他"暂缓/未做"

- **§99**：4 个 queue 脚本的参数化合并**暂缓**（工作量约半天）。理由：它们互差 ~50%
  （API key / config 来源 / dataset / auth 逻辑差异真实），且 B1 shopping queue 即将启动是 **hot path**，
  这类 queue 是**阶段性产物**（Phase 1 跑完后大概率不复用），等 baseline 全收尾后再合并风险更小。
- **§94**：**排除 `max_tokens` 作为 B1 Reddit 反转的原因** —— B1 `max_new_tokens=384`，
  实际输出 mean=98-105 tokens，截断率 **0.0%-0.04%**，与反转无关。
- **§100 Next steps #3**（未做）：增加**任务导向 probe**（*"Where would you click to find X?"*），
  因为它比 OCR-recall probe **更接近 task SR 的能力测量**（当前 probe 只测 OCR recall，与 task SR 之间有 gap）。

---

## 十五、方法论纪律（A1 期确立、后续反复被引用）

1. **LLM hallucination signature 检测规则**（§109.18，05-04）—— 五条：
   (1) arXiv ID **编号年月 > 当前月**即 fabricated
   (2) **精确小数 + "基于内部测试" 无 source** 即 fabricated
   (3) **精确版本号 + 精确 release 日期无可点击 release URL** 即 fabricated
   (4) **政府机构/标准组织 + 具体接入数 + 时间** 三特征同现 **80%+ fabricated**
   (5) plausible 但 fetch 后无关的 arXiv ID **必须 actually fetch 才能 detect**
   触发背景：Round-2 6 处 + V2 **8/8 systematic fabrication**；
   风险原文：*"reviewer 一查发现整篇 arXiv table 全错 → **defensibility collapse**，比单点 hallucination 更危险"*。
2. **GLM 自动分析管线可信性**（§6，04-06）—— 接受 GLM batch digest 作为可信归因来源，
   依据 = AI 报告含完整失败类型分布 13 类 + 步数消耗 + 信息瓶颈证据链，**与人工分析结论一致**。
3. **不做线性分解**（§101，见 1.6）。
4. **笔记是 chronicle，历史路径不回改**（§99，见 8.7）。
5. **backfill 是 lossy work**（§116.7）—— 每次写笔记 `[bug]` 立刻 add catalog subsection。
6. **P79 novelty 定位**（§25，04-10）= 首个 **intra-model observation-mode routing 研究** +
   系统性四因素消融 + **把 mirage effect 当 feature 而非 bug**。

---

## 十六、⚠️ 本片矛盾与待核清单（合并阶段用）

| # | 事项 | 两侧 / 需核 |
|---|---|---|
| 1 | **SoM occlusion C'** | §25 修订"必须引入 C'" vs §100 Next#2 标 ❌ 不做（详见 14.7）——**未调和** |
| 2 | **`use_tool_calling`** | §70（04-16）验证失败设 false vs CLAUDE.md 后期 B-991/Fire-6 设 true + `tool_choice='required'`。⚠️ 反转发生在哪个 § 需跨批查 |
| 3 | **B0 温度** | §47 A1 保留 T=0.1 并披露 vs §107 C4 改 T=0.0（18 个 yaml）。演化非矛盾，但引用 §47 必连 §107 |
| 4 | **visual_fp** | §95 代码删除 vs prereg line 197 曾写 combined（由 RETRACTED §115.1 作废）。⚠️ 核 B 批 §115.1 |
| 5 | **N_cells / K-of-N** | §110.3 的 16 cells + K_h1≥12/16 vs 当前 42 conditions / 6 cells + K-of-N transparency-only。⚠️ 降级链在后批 |
| 6 | **latency estimand** | §8 "不跨 condition 可比" + §31/§97 "须独占 GPU 重跑" vs 当前 canonical retry-adjusted latency。⚠️ 演化链在后批 |
| 7 | **early-stop 关闭时间差** | advisor 05-05 confirm，代码 §116.2 05-08 才关。⚠️ 该窗口是否有数据落盘需核 |
| 8 | **§103 / §118 被点名** | §103 三条挂 `named by RETRACTED §106`；§118 挂 `named by RETRACTED §120`。⚠️ 读时必须连作废条一起读（B 批） |
| 9 | **§111 mechanistic** | §111 task 0 L11 patching 93% flip 作为 §5 representative finding 已 RETRACTED §117.4；且整个 §5 于 2026-05-14 搁置 |
| 10 | **消失的证据文件** | `probe_b37_api_determinism.md`（§107.1）/ `VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md`（§107，§108.20 删）/ `docs/literature/结果1.md`+`结果2.md`（§95）/ `transcript.md` + `advisor_sync_5_5_{followup,outcomes}.md`（§110.7）/ `docs/reference/transcript.md` —— **全部已不在原路径**，引用这些 § 的证据链时须注明 |
| 11 | **§56 含数字却在无数字批** | §56 记 "B0 cls DOM 7.7% vs 严格定义 ~0.4%"。切批的"无数字"判定不完全准，**不影响本文件（数字原样抄）**，但说明 A/D 批边界有渗漏 |

---

*本文件覆盖 A 批 4 片中的第 1 片（219/831 条）。A2（§121–§164）/ A3（§165–§240）/ A4（§241–§397）见同目录。*
