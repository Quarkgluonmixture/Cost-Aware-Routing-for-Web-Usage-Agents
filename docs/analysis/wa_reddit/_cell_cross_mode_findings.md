# WA reddit × B1 — cell 级发现（6 mode 共用）

*生成 2026-08-02（/diag Tier-2 + Tier-3，6 condition 一次性做完）*

> 本文件收 **跨 mode 共有** 的发现。单 mode 的 per-rule 分布与 episode 明细见
> `B1_<mode>_wa_reddit_diag_digest.md`。六份 digest 全部引用本文件，不各自复述。

---

## 0. 先决条件：Tier-1 曾在空 config 下跑过 — **已于 2026-08-02 修复并重扫**

> ✅ **状态：已解决。** `results/diag_scans/v8_wa/` 现在是有效的唯一数据源，本文数字与它一致。
> 修复经过见本节末「修复记录」。以下保留问题描述，因为它解释了为什么 `P36`/`P40` 等结论
> 直到这一轮才浮现，也是 A2（`scan_episodes` fail-loud）仍然值得做的理由。

**原问题**：`results/diag_scans/v8_wa/B1_*_wa_reddit.json`（6 份）是在空 config 下跑的。

`scan_episodes()`（`scripts/analysis/diag_pattern_match.py:2001`）在
`task_configs/<stem>.json` 缺失时**静默退化**成 `config = {}`，不报错也不警告：

```python
config: Dict = {}
if config_path.exists():
    config = json.loads(config_path.read_text(encoding="utf-8"))
```

而 DGX 上**全部 19 个 WA run** 的 `task_configs/` 都是空目录（VWA run 全部齐全，对照组
`B1_dom_reddit_20260703` = 205 个文件）。44 条 P-rule 里 **28 条读 `config`**，空 config
下全部静默返回 `[]`。

**后果（重扫前 vs 重扫后）**：

| 量 | 修复前（空 config） | 修复后（现 canonical） |
|---|---|---|
| success 侧命中 | 0（六 mode 全 0） | **5**（`P40`，见 §3） |
| `P43` 命中 | 0 | 20 |
| `P46` 命中 | 0 | 1 |
| `P31` 命中 episode 数 | 406 | 390（404-carveout 需真实 URL 才生效；`P31` 每 episode 只发一次 hit，step 级同值） |
| failed-NO-hit 合计 | 54 | **56** |

`task_configs` 在 A100（source of truth）上完好：六个 run 各 104 个文件，且**字节级一致**
（`md5 8b0b30e35a67`）—— 所以是回传丢的，不是没生成。

⚠️ **不要用 `test_reddit.raw.json` 直接当 config**：raw 里 `start_url` 仍是占位符
`__REDDIT__`，runner 写盘时会做 placeholder 替换（`tasks.py:265`）。用 raw 会让所有比对
URL 的规则（`P14`/`P19`/`P20`/`P24`/`P30`）静默失配。实测 `P31` 因此虚高 16（406 vs 390）。

### 修复记录（2026-08-02，B-1919）

**根因**：`sync_a100_results.sh` 只同步 `visualwebarena/phase1/`，**`webarena/phase1/`
从来没有任何自动同步路径**（脚本第 49-50 行硬编码单一子树）。WA 数据到 DGX 全靠手动 rsync，
带什么过滤器就丢什么。

**修法**：

1. **定向恢复** —— 从 A100 拉回全部 WA run 的 `task_configs/`（19 个 run，含 5 个 B0 在跑的）。
   六个目标 run 逐 episode 覆盖 104/104，`md5 8b0b30e35a67` 与 A100 一致。
2. **堵根因** —— `sync_a100_results.sh` 改为循环 `BENCH_SUBTREES`，覆盖两个 benchmark 子树。
3. **按子树设 delete 策略** —— dry-run 发现直接给 WA 加上 VWA 那套 `--delete-after
   --delete-excluded` 会删掉 **26,207 个路径**。逐条核过：25,999 个是空的 `artifacts/`
   目录壳（无害，正是 `--delete-excluded` 的用途），但余下是整个
   `B0_wa_3mode_shopping_20260417`（192 个 task_configs + run_meta + 2 个 condition_meta，
   2026-04 归档骨架，**只在 DGX 上、A100 已无**）。为镜像一台从未持有该数据的机器而删除
   DGX 独有历史，正是本 patch 要修的那类错误 → **WA 子树设为 additive-only（`nodelete`）**，
   VWA 保留 delete（`clear_tasks` 传播需要，B-841）。
4. **重扫** —— `diag_pattern_match.py` 直接读盘上 config 重跑六个 condition，覆盖
   `results/diag_scans/v8_wa/`。与修复前用注入 config 算出的结果做**逐 hit 互证：
   6/6 完全一致**（hit 集合、success 数、N 全等）→ 两条独立路径互相验证。

**遗留**：`B0_wa_3mode_shopping_20260417` 那 888K 骨架仍在 DGX，是否清理由你定。
**A2 仍未做** —— `scan_episodes` 在 config 缺失时依然静默退化，下次同类丢失还是无声的。

---

## 1. 分母口径

| 量 | 值 | 依据 |
|---|---|---|
| collected | 104 / mode | episode 文件数 |
| **scored** | **104 / mode** | `sr_excluded` 字段在 624 个 summary 上**全部为 False** |
| raw config 总数 | 106 | `test_reddit.raw.json` |
| 未跑的 2 个 | task 723 / 726 | `_is_na_task` 判为 N/A，`exclude_na_tasks=True` 在 load 时排除（§139.8 预注册） |

WA reddit **不存在** VWA reddit 那种 203/205 分裂（B-1913）。SR 分母直接用 104，
分子分母同源，无需手工扣除。

---

## 2. SR 与三子集

| mode | N | success | SR | failed+hit | **failed-NO-hit** | success+hit |
|---|---|---|---|---|---|---|
| `dom` | 104 | 17 | **16.35%** | 80 | 7 | 2 |
| `som` | 104 | 14 | **13.46%** | 80 | 10 | 1 |
| `vision` | 104 | 10 | **9.62%** | 81 | 13 | 1 |
| `phantom_text` | 104 | 17 | **16.35%** | 75 | 12 | 1 |
| `phantom_som` | 104 | 12 | **11.54%** | 84 | 8 | 0 |
| `phantom_prompt` | 104 | 17 | **16.35%** | 81 | 6 | 0 |
| **合计** | 624 | 87 | 13.94% | 481 | **56** | 5 |

对照同 ruleset 的 VWA reddit B1（`results/diag_scans/v8_vwa/`）：SR 2.93–8.29%。
**WA reddit 的 SR 是 VWA reddit 的 2–3 倍**，与 WA 无图像任务（0/104 带 `image`）一致。

### 规则迁移缺口

44 条规则在 WA reddit 只有 **15 条** fire。逐条查过适用性后分三类：

| 类 | 规则 | 判定 |
|---|---|---|
| **不适用（0 fire 正确）** | `P34`/`P43` 的 image gate、`P25`、`P42` | WA reddit **0/104** 任务带 `image`；**104/104** 任务 `sites=["reddit"]` 单站 → 跨站/图像规则本就无对象 |
| **不适用（eval 形状不含）** | `P41` | `program_html.required_contents` 形态只有 `must_include`(131) 和 `exact_match`(12)，**没有一个** must_exclude-only |
| **cls site-gate** | `P6`/`P16` | v7 H1 显式加的 `benchmark_site!="classifieds"` gate |

`P46` 只 fire 1 次（dom task 409）不是规则失效：逐 gate 追踪显示 537 个失败 episode 里
51 个通过 intent gate，只有 12 个通过 "无成功 type" gate，1 个通过 "有 finish + answer" gate。
`P46` 要求 agent **交了答案**，而 WA reddit 的 comment 类失败绝大多数是**耗尽预算没交答案**
（`P31` 的地盘）。

---

## 3. success 侧 benchmark-FP：`P40` 命中的 5 个全部坐实

盘上扫描因空 config 完全看不到这批。重扫后 `P40 LUCKY_NUMERIC_FP` 在 5 个 **success**
episode 上 fire：dom 27/28、som 27、vision 30、phantom_text 31。

同一任务族，intent 同构：
> "Tell me the count of comments that have received more downvotes than upvotes for the
> user who made the latest post on the `<X>` forum."

`reference_answers = {"must_include": ["0"]}`。

> ⚠️ **机制更正（2026-08-03，B0 轮）**：此处原写"`must_include` 是**子串匹配**
> （`clean_ref in clean_pred`）"，**只覆盖了 else 分支**。实际是双分支
> （`evaluation_harness/evaluators.py:166-176`）：
>
> ```python
> if len(word_tokenize(clean_ref)) == 1:
>     return float(clean_ref in word_tokenize(clean_pred))   # 精确 token 成员 ← 漏了这支
> else:
>     return float(clean_ref in clean_pred)                  # 子串
> ```
>
> `"0"` 恰恰是单 token，走的是 **if 分支**。本节结论（该族送分）不受影响 —— agent 答
> "There are 0 comments" 的 tokenize 结果含 `"0"`，仍然命中，只是**命中方式是精确 token
> 而非子串**。
>
> 但这个误解有后果：它让本轮不会去查"为什么 `cheating` 不含 `cheat`"，因而**漏掉了
> 6 个 task / 74 个 episode 的 tokenize 假阴性**（`long`/`relation`/`headphone`/`break`/
> `product`/`cheat`）。详见 [[_benchmark_level_findings]] §B2。

Tier-2 逐个查轨迹：**5/5 `hit_causal=true`**，无一走完推理链（找到目标 forum → 该 forum
最新帖作者 → 该作者 `/user/<name>/comments` → 逐条比较票数）。5 个全是"逛到某个不相关帖子，
看到评论区为空，于是答 0"，恰好命中子串。som task 27 只用 5 步、3 个 URL。

**已知 `P40` 守卫在 reddit 上空转**：`detail_markers = ("page=item","product_id=","/product/")`
（`diag_pattern_match.py:1591`）全是 cls/shopping 的 URL 形态，Postmill 上永不出现 → 豁免
分支在 reddit 上**永远不触发**。本轮 5/5 恰好都确实没做工作，所以没暴露，但那是巧合不是设计保证。
reddit 的等价 marker 应是 `/user/<name>/comments`。

**影响**：这个族在 6 个 mode 上都在，每个 condition 无差别送分 → 系统性抬高 reddit SR 绝对值。
对 mode 间相对比较影响小（各 mode 同等受益），但给 drop-one oracle / routing 分析注入的是
**不区分 representation 的纯噪声**。

⚠️ **本轮只审计了 4 个 mode 各 1 例**。正式从计分集剔除前，需把 27/28/29/30/31 全族 ×
6 mode × B0/B1/B2 过一遍。

---

## 4. Tier-2 三分类合计（56 个 no-hit + 12 个因果验证 + 5 个 FP 审计 = 73 ep）

| 分类 | 数量 | 说明 |
|---|---|---|
| **agent-limit** | 49 | 主体。见各 mode digest |
| **benchmark-FP** | 6 | `P40` 族 5 + dom task 66（§5 F3） |
| **scaffold-bug** | 4 | phantom_text 601 / som 581 / phantom_prompt 731 / dom 29（后者见 §5 F2） |
| **unclear** | 2 | phantom_text 641、som(C2) 30 |
| 因果验证组 | 12 | 见 §6 |

---

## 5. Scaffold-bug — 总体级坐实的 3 条

以下每条都做过 **624 episode 全量 0-token 复核**，不是小样本外推。

### F1. `select_option` 在 reddit 上 99.6% 失败 —— **不是新 bug，是 B-06 剔除掉的那个桶**

> ⚠️ **先纠正定性**（2026-08-02 查台账后）。这个机制 **2026-04 就裁定过**，不是本轮发现：
> §51 / B-57（原生 `<select>` Playwright 点不开 → JS workaround）· §60（`_inject_css_dropdown_options`，
> 明写 "classifieds 'Sort by' 与 reddit sort 是 CSS/JS 自定义下拉不是原生 `<select>`"）·
> B-59（选中态反馈缺失）· B-64（vision 路径的 CSS 下拉不支持）。
>
> **本轮真正新增的只有 blast radius。** B-06 的 self-replay probe（20 例）当时就测出
> **18/20 是 `OTHER_CUSTOM_DROPDOWN`**，站点分布也点了 "reddit ARIA combobox"，但它把这
> 18/20 判为「走 click 路径不走 select_option dispatch」→ **从 blast radius 里剔除**，
> 只留 native arg-drop 的 2/20 → 结论 "15 ep / 0.3% of all ep, low priority"。
>
> **那句剔除依据与代码不符。** `p79/envs/vwa_wrapper.py:1194/1352` 的 CSS fallback
> 扫隐藏 `<ul>`（`getBoundingClientRect()` 宽高为 0、距点击点 ≤150px、含 `<li><a>`），
> 找不到就 `return {matched: false, ..., error: 'no_match_in_css_menus'}` —— **动作直接失败，
> 没有任何 click 兜底**。所以那 18/20 不该被剔除。

| 站点 / cell | 尝试 | 成功 | `no_match_in_css_menus` |
|---|---|---|---|
| **WA reddit（6 mode 合计）** | **1392** | **5 (0.4%)** | **1387** |
| VWA reddit B1 dom | 48 | 5 (10.4%) | 43 |
| VWA classifieds B1 dom | 285 | 86 (30.2%) | 0 |
| VWA classifieds B1 som | 155 | 99 (63.9%) | 9 |

**这是 reddit 站点特有，不是 WA 特有** —— Postmill 的 forum 选择器是自定义 CSS 下拉
（非原生 `<select>`），`p79/envs/vwa_wrapper.py` 的 CSS fallback 扫不到匹配菜单。

**与 B-06 旧估计的差距**：B-06 说 "15 ep / 0.3% of all ep"，那是**只数 native arg-drop**
的结果。按实际失败动作数重测，仅 WA reddit 六个 condition 就有 **1387 次失败动作、
覆盖 7–26 个 episode/mode**，VWA reddit 另有 43 次。差距不在测量精度，在**计数对象**：
B-06 数的是"upstream arg-drop 影响几个 episode"，这里数的是"select_option 这个动作实际
有多少次没成"。两个都对，但只有后者回答"agent 能不能选下拉框"。

WA reddit 上按 mode 分布均匀（dom 214/215、som 431/433、vision 156/156、phantom_text
106/106、phantom_som 226/227、phantom_prompt 254/255），覆盖 7–26 个 episode/mode。

**SR 关联**：用过 `select_option` 的 episode SR = **5.56%** (6/108)，没用过的 = **15.70%**
(81/516)。⚠️ 这是相关不是因果——需要下拉框的任务（建 forum、选版块发帖）本身更难，
两者混在一起。但它给"发帖/建版"整族任务设了一个天花板。

**⚠️ 影响 paper-1**：VWA reddit 也中招（43/48），而 VWA reddit 是**计分 cell** 且已冻结进
v8 结论。需评估这是否改变 reddit cell 的解读。

### F2. `walk_fail` ≠ 模型编造 id —— 元素 100% 在观测里，且在视口内

`P36 WALK_FAIL_DEGENERATE` 是 WA reddit 第二大失败质量（310 episode）。此前默认按
agent-limit（"模型报了不存在的 id"）理解。**在总体上不成立。**

方法：把模型请求的 `element_id` 与**该步落盘的观测文本**比对，并用**成功动作做阳性对照**
（成功动作必定用的是有效 id；若成功 id 也查不到，说明落盘的不是模型输入，该 mode 读数作废）。

| mode | 阳性对照：成功动作 id 在观测中 | walk_fail id 在观测中 | 探针有效？ |
|---|---|---|---|
| `dom` | 1425/1425 = **100%** | **367/367 = 100%** | ✅ |
| `som` | 1190/1190 = **100%** | **403/403 = 100%** | ✅ |
| `phantom_prompt` | 1247/1247 = **100%** | **537/537 = 100%** | ✅ |
| `phantom_text` | 44/1563 = 2.8% | — | ❌ 见 F4 |
| `phantom_som` | 48/1326 = 3.6% | — | ❌ 见 F4 |
| `vision` | 无 element_id 动作 | 0 | N/A |

**三个通过对照的 mode 里，1307 个 walk_fail 步中"元素不在观测里" = 0 次。** 模型一次
都没有编造 id。

按元素角色拆分（"在观测里"≠"本该可点"）：

| mode | 可交互角色（link/searchbox/textbox/…） | 不可交互（StaticText/heading/RootWebArea/…） |
|---|---|---|
| `dom` | 244 (66%) | 123 (34%) |
| `som` | 230 (57%) | 173 (43%) |
| `phantom_prompt` | 395 (74%) | 136 (26%) |

- **可交互角色上的 walk_fail = scaffold 侧**：框架列出了 link/searchbox，模型点了，派发失败。
  合计 **869 步**。`click|interactive|searchbox` 单项就有 106(dom)/64(som)/265(pprompt)。
- **不可交互角色上的 = agent-limit**：模型分不清"可读"与"可操作"。合计 432 步。
  其中 `type|non_interactive|RootWebArea`（58/19/48）是 `P4` 的地盘。

**viewport 假说已排除**：walk_fail 步的 `element_bbox` **全部有值**，且 362/367(dom)、
403/403(som)、528/537(pprompt) 落在视口内。不是"元素滚出屏幕所以够不着"。

代表证据（dom task 29，死锁 8 步）：
```
[36948] link 'Submissions' url: http://localhost:9999/user/MarvelsGrantMan136/submissions
```
该行在 step_022–029 的观测里**每一步都在**，模型 thought 正确识别（"there is a 'Submissions'
link that likely leads to all posts by this user"），`click(36948)` 连续 8 次
`walk_fail:no_actionable_within_walk`。phantom_text task 27 在同一个"Submissions"链接上
（id=15）复现同样失败 → 跨 mode 可复现。

**→ `P36` 应按元素角色分叉**：可交互角色 = scaffold（`is_scaffold=True`），不可交互 =
agent-limit。当前一条规则把两种机制混在一起。

### F3. WA reddit task 66 的 reference 硬编码生产域名 —— 结构性不可解

```json
"must_include": ["http://www.reddit.com/f/books/59396/apple-books-...",
                 "http://www.reddit.com/f/books/17445/i-just-finished-..."]
```
部署域名是 `localhost:9999`，`must_include` 是纯子串比对 → **任何 grounded agent 都不可能
匹配**。

全量复核（我扩查了 7 份 raw config，不止 WA reddit）：

| config | 命中 |
|---|---|
| `wa/test_reddit.raw.json` | **1/106**（task 66） |
| `wa/test_webarena.raw.json` | 1/812（同一 task 66） |
| `wa/test_shopping.raw.json` | 0/192 |
| `wa/test_shopping_admin.raw.json` | 0/182 |
| `vwa/test_classifieds.raw.json` | 0/234 |
| `vwa/test_reddit.raw.json` | 0/210 |
| `vwa/test_shopping.raw.json` | 0/466 |

**孤例，且不污染 VWA**。建议：把 task 66 标为 benchmark-config-defect 并从 WA reddit 计分集
剔除（或脚注），同时给 `validate_run` 加一条静态 config lint（0-token，不需要跑轨迹）。

### F4. artifact 落盘 ≠ 模型输入（phantom_text / phantom_som）

F2 的阳性对照顺带查出来的：这两个 mode 的 `artifacts/.../observation_dom.txt` 存的是
**原生 AXTree**（稀疏 id），而模型实际收到的是 [SOM_MARKS] 重编号（紧凑 1..N）。成功动作的
id 在落盘观测里只有 2.8% / 3.6% 能找到 —— 落盘的不是模型看到的东西。

`som` mode 反而是对的：它额外存了 `observation_som.txt`（`[id=N]` 格式）。

**后果**：这两个 mode 的任何"模型看到了什么"的事后核查都做不了。F2 对它们**无法判定**，
不是"判定为没问题"。→ 建议 phantom 系列比照 som 落盘 `observation_som.txt`。

---

## 6. 因果验证：Tier-1 主导规则是死因还是伴随

12 个 episode 跨 4 组抽样（`P31`-only / `P36`+`P45` / vision `P5`+`P14` / `P43`+`P44`）。

| 规则 | failed 覆盖 | 裁定 | 依据 |
|---|---|---|---|
| `P31` budget 耗尽 | 390 (72%) | **风险标记，非死因** | 3/3 "仅 P31" 样本的真实死因互不相同（表单反复按 Enter / 判断自相矛盾 / vision 全程不用搜索框），共同点只是"没做完"。反事实"多给步数能否救回"三个都是否 |
| `P36` walk_fail | 310 | **死因（对可交互角色）** | 见 F2。但需按角色分叉 |
| `P45` 连续同动作失败 | 203 | **死因，有盲区** | 精确描述死锁。盲区见 §7 R2 |
| `P5` 感知缺失循环 | 284 | **死因（vision 特有位置）** | vision 走 `coord_mouse_click` 绕开 element_id 通路，`P36`/`P45` 覆盖不到的地方由 `P5` 补上 |
| `P43` 无截图 | 20 | **中性标签（正向验证）** | 2/2 phantom 样本里目标图片**仅凭标题文本**在 step_0 一次命中 → 缺图没妨碍识别。与项目既有裁定一致 |
| `P44` 幻觉引用 | 20 | **样本不足** | 唯一样本（dom 611）是 30 步 episode 里的孤立插曲，真实死因是任务理解错误。n=1，不下结论 |

---

## 7. Self-evolving — 新规则候选（**已做 success-safe 全量检验**）

sub-agent 共提出十余条。下表只列做过 624 episode 全量复核的四条。**两条没通过。**

| 候选 | failed 命中 | success 命中 | 裁定 |
|---|---|---|---|
| **R1 `PREMATURE_FINISH_ON_FORM`** — finish 前最后一个非 finish 动作是 `type`，其后无 `click`，且 finish 时 `obs_url` 仍在 `/submit/`\|`/create_forum`\|`/-/edit` | **24** | **0** | ✅ **建议落码**。success 侧零误伤，跨 6 mode 均匀（2–6/mode） |
| **R2 `NOELEM_ACTION_STREAK`** — 连续 ≥3 次同一无 `element_id` 动作（`press`/`scroll`/`wait`）。补 `P45` 的盲区：`P45` 以 (action_type, element_id) 为键，无 id 的动作完全逃过 | 133 | **15** (17% of success) | ⚠️ **不可直接落码**。vision 独占 82/133（坐标点击天然连击），success 侧 15 个误伤。需先按 mode 分叉或加"页面无变化"副条件 |
| **R3 `PREMATURE_NEGATIVE_AFTER_SEARCH`** — ≤4 步、进过 `/search?q=`、finish 措辞含 "no results/no submissions" | **9** | **0** | ✅ 可落码但覆盖低（1.7%）。som task 724 vs 730 提供了硬反证：同一用户 Hrekires 同一版，724 经 `/user/Hrekires/submissions` 找到了真实帖子，730 却断言"无投稿" |
| **R4 `FORUM_NEVER_VISITED`** — intent 点名的 forum 在整条 URL 轨迹里从未出现 | 133 | **31** (36% of success) | ❌ **否决**。这是本轮被 sub-agent 最看好的候选（phantom_prompt agent 列为首选），但它在 36% 的成功 episode 上 fire = 典型 presence-only。**不落码** |

**未做全量复核、仅小样本的提议**（列出以便下轮跟进，**不得据此落码**）：
`body-vs-comment-misplacement`（vision 2/13）· `vote-action-never-attempted`（vision 4/13）·
`form-abandoned-via-navbar-click`（phantom_som 1/8、phantom_prompt 2/6，bbox y=0 的顶栏
"Submit" 链接）· `type() 用 .fill() 是替换语义而 prompt 文案暗示追加`（phantom_prompt
task 731，代码级断言，需核 `locator_dispatch.py` 与 `_shared_vl_utils.py` 的一致性）。

---

## 8. 可 actionable 项

| # | 事项 | 类型 | 优先级 |
|---|---|---|---|
| A1 | ~~WA run 的 `task_configs` 未回传到 DGX（19/19 全空）→ 28 条规则静默失效~~ | scaffold / infra | ✅ **已修 2026-08-02（B-1919）** — 根因是 `sync_a100_results.sh` 从不同步 WA 子树；已恢复 + 堵根因 + 重扫互证，见 §0 |
| A2 | `scan_episodes` config 缺失时静默退化成 `{}` → 应 fail-loud 或至少 warn + 在输出 JSON 里记 `config_missing` 计数 | scaffold | **P0，未做** — 这是 A1 能潜伏半个月的原因；不修则下次同类丢失依然无声 |
| A3 | `select_option` 在 reddit 99.6% 失败（F1），**波及 VWA reddit 计分 cell** | scaffold | **P0** |
| A4 | `P36` 按元素角色分叉（可交互 = scaffold / 不可交互 = agent-limit）（F2） | 规则 | P1 |
| A5 | WA reddit task 66 reference 硬编码 `www.reddit.com`（F3）+ 给 `validate_run` 加静态 config lint | benchmark-FP | P1 |
| A6 | `P40` 的 `detail_markers` 在 reddit 空转，应按 site 分（reddit → `/user/<name>/comments`）（§3） | 规则 | P1 |
| A7 | task 27/28/29/30/31 族全量审计后决定是否移出计分集（§3） | benchmark-FP | P1 |
| A8 | phantom_text / phantom_som 落盘 `observation_som.txt`（F4） | infra | P2 |
| A9 | 落 R1 + R3 两条新规则，bump `RULESET_VERSION`，**全量重扫 36 VWA + 6 WA condition** | 规则 | P2 —— A1 已修，现可动手 |

---

## 9. 定位声明

- 本轮 = **单 cell（WA reddit × B1）6 mode**，无 B0/B2 对照。cross-model 结论不在此下。
- `ruleset_version` = `8-reddit-p41p46-b1890fix`，与 36 个 VWA canonical condition 同版本，
  但**本轮数字是注入 config 后重算的**，与盘上 `results/diag_scans/v8_wa/` 不一致（§0）。
  A1 修完并重扫前，**不要把本 cell 的数字与 VWA 的直接并表**。
- Tier-2 覆盖 73/624 episode（56 no-hit 全覆盖 + 12 因果验证 + 5 FP 审计），8 个 sonnet
  sub-agent。sub-agent 的定性判定与规则命中在文中分别标注。
- 凡标 "✅ 建议落码 / 已坐实" 的，均经 624 episode 全量 0-token 复核；小样本提议单列于 §7 末。
