# B5 classifieds — 五个 mode 的 /diag 汇总（2026-09-11）

> B5 = GPT-5.6（`gpt-5.6-terra` via AWS proxy，`structured_output: response_format`），classifieds，
> dom / som / P-text / P-prompt / P-SoM 五个 condition。**vision 不在内**：B-1997 坐标契约错配，修好重跑后再补。
> ruleset `11-intent-text-fallback`，与 48 个既有 condition 同版本，可以并表。
> B5 在 `run_manifest.yaml` 的 `extension:` 节 —— **不在预注册 cell 集合里，不进任何 pooled / K-of-N / forest 统计**（笔记 §509）。
>
> 各 condition digest：[dom](B5_dom_classifieds_diag_digest.md) · [som](B5_som_classifieds_diag_digest.md) ·
> [P-text](B5_phantom_text_classifieds_diag_digest.md) · [P-prompt](B5_phantom_prompt_classifieds_diag_digest.md) ·
> [P-SoM](B5_phantom_som_classifieds_diag_digest.md)

## 1. 覆盖

| mode | SR | 失败 | Tier-1 有命中 | **no-hit** | Tier-2 覆盖 |
|---|---:|---:|---:|---:|---:|
| dom | 23.66% | 171 | 154 | 17 | 17/17 |
| som | **37.05%** | 141 | 87 | **54** | 54/54 |
| P-text | 24.11% | 170 | 136 | 34 | 34/34 |
| P-prompt | 21.88% | 175 | 138 | 37 | 37/37 |
| P-SoM | 22.77% | 173 | 136 | 37 | 37/37 |

no-hit 共 **179 个，全部深挖**（15 个 sonnet sub-agent，按 task 跨 mode 分批）；另审计 17 个 hit episode（P33 / P10 / P17）。
做法上和以往不同的两点：sub-agent 读的是**压缩轨迹**（每步一行：动作 / 目标 / 页面变没变 / URL / thought），
并附上从本机干净站点按 id 查到的 **agent 终态 item 与参考 item 的标题、价格、发布日期**，所以「选错了哪个 item」不用猜。
som 的 no-hit 最多：SR 最高的 mode，失败反而最少落进规则库擅长的「循环 / 卡住」形状，更多是「看起来合理、选错了」。

## 2. 三分类

| 类别 | no-hit 中 | 说明 |
|---|---:|---|
| agent-limit | **174** | Tier-1 命中的失败也全是 agent-limit 类规则 |
| scaffold-bug | **1** | dom task 210 → **B-1998**（机制见 §3.1，影响面远不止这一个 episode） |
| benchmark-FP 嫌疑 | **4** | 全是 task 41（四个 mode 各一），见 §3.2 |
| unclear | 0 | |

no-hit 子类（五个 mode 合计，sub-agent 标签归一后）：

| 子类 | 数 | 含义 |
|---|---:|---|
| `keyword-literal-proxy` | **55** | 把 Osclass 的字面关键词搜索当成语义 / 视觉判断；参考 item 标题里没有这个词，永远搜不到 |
| 视觉判断选错 | 33 | 颜色 / 外观 / 「图里那件」判断错；**som 有截图也常走「搜颜色词 → 价格排序 → 点第一个」** |
| 循环跑满 30 步 | 26 | 同一搜索词反复提交、scroll 方向来回、两页之间横跳 |
| 丢约束 | 13 | 丢类目参数、把手机壳当手机、把「画的是发明者」当「画的是飞机」 |
| 其他 | 10 | 已打开正确 item 又离开（string_match，P30 不覆盖）、编造佐证细节等 |
| 说了要核实却直接交 | 9 | thought 写「需要核实 X」，下一步不做任何核实就 finish |
| 该 mode 无图但任务要看图 | 7 | 结构性不可解（无图 mode 的颜色 / 封面题） |
| 答错数值 / 属性 | 7 | string_match |
| 放弃给定页 | 6 | intent 说「这一页」，第一步就搜索离开（§4.2 有全量测量） |
| 排序错 / 读错 intent | 8 | |
| task 41 / scaffold | 5 | 见 §3 |

**三个 task 五个 mode 全失败**：97（形状像动物 —— 参考 item 是「两只金属鹿形草坪装饰」，标题里没有 animal）、
117（和参考图同色的自行车 —— 四个 mode 选中同一个错 item）、123（和图中人物所用物品同色 —— 参考 item 是一辆 Mini Cooper）。
**task 40** 与 B0 已有定案一致（真正最新的不锈钢件标题里没有「stainless steel」），是跨 backbone 的固定陷阱。

## 3. 三件不是 agent 能力的事

### 3.1 B-1998：`multiple_actions` 被判无效，动作却照样执行

dom task 210 只跑 3 步：三步都被判 `parse_valid=False / multiple_actions`，撞 `max_consecutive_parse_errors=3` 被结束；
但三步页面都变了，URL 按 thought 推进。全量查：

| | `multiple_actions` 步数（每个 condition） | 其中页面变了 |
|---|---:|---:|
| B0（6 个 cls condition） | **0** | — |
| B1（6 个） | **0** | — |
| B2（5 个非 vision） | 2–24 | 大部分 |
| **B5**（5 个 + dom replicate） | **49–89** | 80–95% |

原因：解析器遇到多个不同的有效动作时返回「第一个 + 无效」，注释写着「如果 runner 按 parse_valid 把关就不会执行」；
runner 实际是在 `env.step()` 之后才计算 `parse_valid`，从没把关。结果是**动作执行了，记账却按「注入 wait、不耗预算」**：
B5 每个 condition 有 50–90 个不计入 30 步预算的真实动作（有 episode 跑到 31–32 步）。
B0/B1 不受影响 ⇒ 不改变任何预注册 cell 的数字；B5 / B2 的步数与空转类指标（§508.2）要带着这条读。

### 3.2 task 41：「第二行的价格区间」取决于页面每行放几个 item

sub-agent 起初判为「站点状态在 08-20 → 08-26 之间漂移」。0-token 核查推翻了这个说法：
- **5 月 25 日以来 32 个 run、所有 baseline 全部做错**；B0 在 5 月就读到了 $1,900；B5 八个 run（含 08-20 的 dom）答案完全相同（$1,900–$27,995）。站点这三个多月是稳定的。
- 本机干净站点上 boats gallery 的价格顺序：`2300, 2500, 3500, 5500, 23750, 1200, 1900, 27995, 24995, …`。
  参考答案 {1200, 23750} 对应**每行 3 个**时的第 4–6 位；agent 读到的是第 7–9 位。

结论：参考答案假设了一种排版，P79 的视口渲染出另一种。定为 **benchmark-FP 嫌疑（依赖布局）**；要确认需要 A100 上的截图看实际每行几个。
它在所有 cell 上都是常量 0，不影响 mode 间差分，只影响 SR 绝对值（上限 1/224 = 0.45pp）。

### 3.3 diag 规则自身的三个缺陷（都在 analysis 层，不影响 SR）

| | 缺陷 | 量 | 处置 |
|---|---|---|---|
| **B-1999** | P31「终态已到参考页」豁免只比 URL path；Osclass 每页都是 `/index.php` ⇒ cls 上的 url_match 任务一律被豁免 | 跑满预算却没标出：cls B0 203 · B1 436 · B2 453 · B5 147（约占 incomplete 的 40–60%）；reddit 全 0 | 改为比完整 URL；需 bump v12 |
| **B-2000** | 数字提取不认千分位（`$6,400` → 6 和 400），P10 多报 | 修后失败侧 B5 61→50、B1 39→29；**成功侧 31→30 基本不变** | 同批 v12；P10 在强模型上主要是语义误报，修逗号不够 |
| P33 | 「导航到裸图片 URL = 幻觉」的假设在有截图的 mode 上不成立 | 审计 som 4/4 非死因（点开大图是合理的看图策略）；无图 mode 5 个里 3 个是真编造 | 建议 som/vision 降为中性事件，同批 |

## 4. 两条被 sub-agent 反复提到的机制 —— 0-token 全量测量

### 4.1 字面关键词代理：真实的风险因素，但不能写成规则

B5 五个 mode 合计，url_match 的 item 任务，只看做过搜索的 episode：

| | 参考 item 标题**不含**任何搜过的词 | 含其中某个词 |
|---|---:|---:|
| 失败（353） | **66%** | 34% |
| 成功（164） | 32% | 68% |

失败组高一倍。但仍有 53 个成功 episode 满足条件（Osclass 搜索同时匹配描述；agent 也会搜完再按类目浏览）⇒ 当风险标记，不当死因。

### 4.2 放弃给定页：B5 特有，且伴随更低的 SR

intent 里有「on this page」这类说法的任务，第一个动作就是搜索的比例（5 个文本 mode，每个 backbone n=230）：

| backbone | 先搜索 | SR：先搜索 vs 留在页上 |
|---|---:|---|
| B0 | 14% | 12.1% vs 23.4% |
| B1 | 32% | 17.8% vs 16.6% |
| B2 | 10% | 0.0% vs 1.5% |
| **B5** | **47%** | 18.5% vs 27.9% |

B5 放弃给定页的频率是 B0 的 3.4 倍。⚠️ 这是描述性对比，没做同 task 配对：先搜索的 task 本身可能更难。成功侧 B5 也有 20 个 ⇒ 不做规则，
可以作为 §508 行为阶梯的新一行（「强模型更相信搜索框」）。

## 5. 规则提议（本轮都不落码）

| # | 提议 | 来源 | 0-token 复核 | 结论 |
|---|---|---|---|---|
| 1 | P31 豁免比完整 URL | 本轮 | 见 §3.3 | 修（B-1999），需 v12 |
| 2 | 数字提取认千分位 | hit 审计 | 见 §3.3 | 修（B-2000），同批 |
| 3 | P33 按 mode 收窄 | hit 审计 | som 4/4 误报 | 建议，同批 |
| 4 | url_match 版 P20：参考 item / 任何 item 页从未打开 | batch 02 / 13 | 「全程零 item 页」：B5 失败 154，**成功 1（string_match）**；限定 url_match 后成功 0 | 可做，但接近结果的同义反复（url_match 必须停在参考页），诊断价值低 |
| 5 | 字面关键词代理 | 多数批次 | 成功侧 53 个也符合 | 不做规则，当描述统计 |
| 6 | 放弃给定页 | batch 12 / 14 | 成功侧 20 个 | 不做规则，当行为指标 |
| 7 | 同一搜索词重复 ≥3 次且期间无 item 页 / scroll 方向来回 ≥4 次 | batch 00 / 07 / 10 / 14 | **未复核** | 待复核后再议 |
| 8 | thought 说要核实、下一步就 finish | batch 07 / 08 / 13 | **未复核**（措辞正则容易误伤） | 待复核后再议 |
| 9 | 物品名词错配（标题含 Case / Cover / Charger，intent 要的是物品本身） | batch 07 / 14 | **未复核** | 待复核后再议 |

> 本轮再次验证了 v8 freeze 时的教训：sub-agent 说「不会误伤成功 episode」的三条（#4、#5、#6）全量复核后**全都会误伤**；
> 它对 task 41 的「站点漂移」解释、对 P10 误报的「千分位」解释，也都只对了一部分或不对。

## 6. 和 §508.3「失败类型沿 backbone 迁移」表的关系

§508.3 说：弱模型的失败多是跑不完，强模型的失败多是答错。B5 补进来后（失败桶「占全部 episode」口径，B0 取同样 5 个文本 mode）：

| | 答错（early-finish / wrong-commit） | 跑不完（search-loop + click-loop + max-steps） |
|---|---:|---:|
| B0 | 47–58% | 25–31% |
| **B5** | **33–47%** | **29–35%** |

B5 把一大块「答错」变成了成功（约 −10pp），但**跑满预算的比例没有下降**（约 +4pp）。所以「越强越少绕路」只对前三级成立，到 B5 这一级，
失败构成里「跑不完」的份额反而上升。⚠️ 只有 cls 一个站点；B5 没有 vision；失败桶是 reason_bucket 的启发式映射；B-1998 让 B5 多拿到少量免费步数，
方向上应当**减少**跑不完，所以它解释不了这个结果。

## 7. 待 user 定

1. v12 规则批（B-1999 + B-2000 + P33 收窄）现在做还是以后：会动全部 53 份 digest 的数字和两个依赖 `diag_scans` 的下游。
2. B-1998 修法 (a) 派发前把关 / (b) 承认执行并记为有效动作；建议并进 B-1997 的同一次 witness。
3. task 41 要不要剔除：先要一张 A100 截图确认每行几个。
