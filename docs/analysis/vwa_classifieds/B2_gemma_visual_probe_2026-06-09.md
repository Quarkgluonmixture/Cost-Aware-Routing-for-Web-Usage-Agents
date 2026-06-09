# B2 Gemma3 视觉分层 probe — 同图同问双模型对照 (2026-06-09)

> 回答 "B2 SR 为何这么低 + som 为何无增益"。probe 在 DGX (GB10) 跑，模型/管道与 A100 实验同款
> （Gemma: `apply_chat_template` 同 `gemma3vl_agent.py:245`；Qwen: `qwen_vl_utils.process_vision_info` 同 `qwen3vl_agent.py:268`）。
> 脚本: `/tmp/som_probe.py` + `/tmp/qwen_probe_fix.py`（一次性 probe，方法全文如下）。

## 设置

- **输入图**: B2 som run R23029 `classifieds_task_22/som/step_000_som.png` (1280×720) 经 P79 链（`image_max_size=1024` LANCZOS → 1024×576）。各模型 processor 自行做后续预处理 —— 与实验完全一致。
- **场景**: OSClass Cars+trucks gallery，56 个 SoM 标号，6 个 listing 缩略图（task 22 = "How many miles does the red car in the second row have?"，reference=103K）。
- **Ground truth**（[SOM_MARKS] 文本 + 1280 原图 2× 放大人工判读）: Apply 按钮=26；Publish Ad=6；第二行车色 = 银轿车 / 银灰(白) SUV / **红色跑车 (Porsche 911 形)**；首行第一 listing 价格 = 7000.00。
- 封闭式三问（targeted 标号读取 / 缩略图颜色 / 小字价格 OCR），greedy，max_new_tokens=256。

## 结果

| 问题 | 测的层 | **Gemma3-4b (image_tokens=256)** | **Qwen3-VL-4B (image_tokens=576)** |
|---|---|---|---|
| Q1 'Apply'/'Publish Ad' 标号 | SoM 标号可读性 | ✅ **26, 6** | ✅ **26 6** |
| Q2 第二行车颜色 | **自然照片内容** | ❌ **"Dark Blue" ×4**（行内 3 辆说成 4 辆；全部同色复读 = 幻觉签名） | ✅ **Silver, White, Red**（且自发认出 "Hyundai Elantra" / "**Porsche 911**" — 无文本可抄，纯像素） |
| Q3 首 listing 价格 | 小字 OCR | ✅ **7000.00** | ✅ **7000.00** |

## 判读 — Gemma 视觉是"分层崩坏"不是"全盲"

1. **UI 文字层（标号 / 价格 OCR）在 256 token 下可读** — 初始假设"SoM 标号物理不可读"被 Q1/Q3 **证伪**。
2. **自然照片内容层崩坏** — 缩略图物体/颜色识别失败且**自信幻觉**（Dark Blue×4；行结构都数错）。与实验轨迹互证：B2 som task22 实跑把 JBL 音响页连续 8 步说成 "Xbox Series X"；B2 dom 把 sofa 说成自行车 ("no rims")。
3. **零弃权风格放大伤害**：B2 som 861 个 thought 里 0 次 "can't see"（B1 4033 个里 409 次）。读不出 → 编 → string_match 必错 → 不 finish → budget death。

## 与 condition 级数据闭环

**B1 (Qwen-4B) cls 四点 ablation 链**（图的真实边际）:

| condition | SR | 含义 |
|---|---|---|
| dom (AXTree, 无图) | 6.2% | 文本基线 |
| phantom_som ([SOM_MARKS] 文本, 无图) | 6.7% | **[SOM_MARKS] 格式效应 ≈ +0.5pp** |
| som ([SOM_MARKS]+标注图) | 14.3% | **标注图边际 = +7.6pp** |
| vision (裸图) | 12.5% | 图单独 ≈ 2× dom |

**B2 (Gemma-4b)**: dom 2.2% (224ep, legit ~1%) → som 3.1% (32ep partial) = **图边际 ≈ +0.9pp ≈ 0**。
→ som 增益的载体本来就是"照片内容 + 空间布局"（标号 id 在 [SOM_MARKS] 文本里都有，图上标号读出来没有增量），**恰好是 Gemma 唯一读不出的层** → som 对 B2 结构性无效。

**可检验预测**（B2 后续 condition 自动检验）: B2 phantom_som ≈ B2 dom ≈ B2 som（全平 ~2-3%）；B2 vision ≤ som。若 B2 vision 显著 > dom，本判读错。

## 为什么 dom 也低（2.2% vs B1 6.2%）— 与图无关的行为差距

dom 无截图（仅 image-task 的 reference image，30% 步），差距在文本侧行为，diag digest 已证：
- agent_finished **10% vs B1 41%**（90% budget death；som 下 9% vs 51% — som 完全没动 Gemma 行为）
- goto-search 不收敛（753 个 goto-search；从不读结果价格、从不 sOrder=dt_price、从不翻页）
- click→page_changed 37% vs B1 56%
- AXTree img-href 误当 item 链接点进裸 PNG（71 ep）
- 已排除: 解析器不兼容（98% parse OK）/ 输出截断（max 166≪4096）/ prompt 构造差异（shared module 逐行同构）/ 部署 bug（transformers 5.8.1, revision pinned, greedy 同 B1）

## 追加 (同日晚): pan&scan A/B — 配置次优真实存在但非主因

用户质疑 "确定不是代码原因吗" → 验证唯一未排除的配置层嫌疑: `do_pan_and_scan`（Gemma3 官方推荐用于文字密集/非方形图，transformers 默认关，P79 未开）。同图同三问 A/B (`/tmp/gemma_pas_probe.py`):

| | P&S=False (256 tok) | P&S=True (**768 tok** = 3 crop×256) | truth |
|---|---|---|---|
| Q1 标号 | ✅ 26,6 | ✅ 26,6 | 26/6 |
| Q2 车色 | ❌ "White,Blue,Gray,White" (4辆全乱) | ⚠️ "**Gray, White**, White" — 数量对+前2对, **红车仍 White** | 银灰/白/**红** |
| Q3 价格 | ✅ 7000.00 | ✅ 7000.00 | 7000.00 |

判读: **(a) P&S 有真实改善** (3× 预算 → 数量+部分颜色翻对) = 当前 B2 部署是**配置次优** (没用官方推荐高分辨率路径), paper 须 disclosure; **(b) 但开了 P&S 关键判别仍失败** (红色跑车认不出 — 恰是 task22 "red car" 的判别信息) → 配置次优是次要因素, 照片认知上限仍低于任务需求, 主因维持"模型层"。**(c)** 256-tok 下两次 Q2 答案不同 (Dark Blue×4 vs White/Blue/Gray/White×4, 均 greedy, 预处理路径微差即翻转) = 信息不足时混沌输出的直接签名。

**裁决 (对"是不是代码原因")**: 代码 bug = 无 (管道通/解析对/图在); **配置次优 = 有且可量化** (P&S 关, 768 vs 256); 主因 = 模型 (agent 行为层 0-finish 与图无关 + 照片认知开 P&S 仍不过线)。可选 follow-up: B2 som+P&S 小 pilot (~30 task) 看 SR 是否动, 决定是否值得改配重跑 (user 决策)。

256-token 固定预算（896×896 squash + 4×4 avg-pool）对高对比规则结构（文字边缘）保留可分性，但自然图像的细粒度颜色/纹理在网格化+池化下混叠；叠加 Gemma3 预训练 OCR 数据占比高 vs Qwen3-VL 的 GUI/grounding 特化。**"为什么 OCR 活了照片死了"的层内机制不在本 probe 证据范围内**。

## 追加 (同日深夜): A100 扩展 probe — 20 图 × 3 臂, 实验同环境 (transformers 5.8.1)

R23029 被废弃清理后 (chain abort, 另一 session 有意清除 + DGX mirror 同步跟删), 图源改用 **B1 som R31705 artifacts** (SoM 标注图 = som.py 环境管线生成, 与跑的模型无关, 同任务同页内容等同)。20 张 step_000 标注图, **实际只 3 个独特场景** (首页 ×18 / 油画页 task14 / 汽车页 task22 — start_url 高度重复, 多样性受限 = 本 probe known limitation)。3 臂: Gemma P&S=off (部署配置) / Gemma P&S=on / Qwen 锚 (实验同款 qwen_vl_utils 路径)。判分 = Claude 对原图人工核对 + [SOM_MARKS] 标题文本。

**稳定性**: 18 张相同首页图三臂各 18/18 答案完全一致 — A100 greedy 确定 (DGX 上的输出漂移 = transformers 5.3 vs 5.8 + fast-processor 路径差, 非模型内在不稳定)。

**判分表 (3 独特场景)**:

| 场景 | Gemma P&S=off (256 tok) | Gemma P&S=on (768 tok) | Qwen (576 tok) |
|---|---|---|---|
| 首页 4 缩略图 | 物体大错: Xbox→"Laptop", Canon→"Telescope" (形状瞎猜, 标题 OCR 都没读出) | 物体名 4/4 (= **标题 OCR 解锁带动**), 颜色弱 (证书纸→"Gold") | 物体 4/4 (标题驱动), 颜色贴照片 (证书纸→"silver/white", 鸟→"gray") |
| 油画页 (5-6 图) | **"Wooden Frame - Brown" 复读 ×31 至 token 耗尽** — 灾难性重复退化 | 5 项全部合理 (Ocean Painting Blue/White ✓) | 4 项合理, 自发引用 SoM 标号 |
| 汽车页 (6 图) | **6 辆报成 12 辆** — 数量爆炸 + 编造幻影车 | 6 辆数量 ✓, 4/6 大致对, 1 幻影黄车, Porsche→"Truck" | **6/6 全对** 含 "Red sports car" |

**关键发现**:
1. **P&S 真实效果比单图 A/B 估计大得多**: 消除两类**灾难模式** (复读崩溃 + 数量爆炸 — 3 场景中 2 个在 256-tok 下出现)。这不是"认知质量差一点", 是 agent loop 幻觉决策的直接源头。物体名从形状瞎猜→全对 (主要由标题 OCR 解锁带动)。
2. **照片真内容三臂都未达**: 首页两个"照片≠标题"冲突项 (Canon listing 缩略图实际是一只**鸟**的长焦样片 / 戒指 listing 实际是 **IGI 证书纸**), 三臂无一报告照片实物, 全报标题物体 — 网页缩略图场景模型普遍被旁侧 OCR 文字 anchor; Qwen 优势在颜色仍忠于像素 + 零灾难 + 无冲突页全对。
3. **修正后的层次**: Gemma 256 = OCR 部分可用 + 照片崩坏 + 灾难模式频发; Gemma 768 = OCR 解锁 + 照片仍弱 (颜色/幻影) + 灾难消除; Qwen 576 = OCR + 照片颜色 + 稳定。

**配置结论更新**: P&S=off 的 B2 在"灾难模式区"运行 — 影响的不只是感知质量, 是 agent 决策输入的稳定性。**som+P&S agent pilot (~50 task) 从"可选"升级为"推荐"** (P&S 可能改变行为签名, 不只感知)。disclosure 无论如何必写。配置层最终归因: 代码 bug 无 / **配置次优 = 实质性** (灾难模式可由 P&S 消除) / 模型差距仍在 (768 仍逊 Qwen 576: 颜色/幻影/照片内容)。

## Probe 方法教训（防复踩）

1. **共享 GPU + `device_map="auto"` 的 meta-device 陷阱**: 初版 probe Qwen 臂权重部分 offload 到 meta device（log: "Some parameters are on the meta device"），vision tower 输出垃圾 → Qwen 报告 "image severely corrupted, heavy pixelation" 并三问全错。**模型对损坏输入的诚实弃权差点被误读成"Qwen 视觉差"**。修复 = 重跑（实验同款 qwen_vl_utils 路径）后三问全对。教训: probe 前 grep log 里 offload/meta 警告；两路径 pixel_values/keys 实测等价（`image_grid_thw` 都在），路径本身无 bug。
2. **单点 thought ≠ 视觉证据**: B1 task22 s0 "red car = Porsche 911" 曾被当作 Qwen 读图证据，实际 `[id=54] link '1987 Porsche 911 Carrera Coupe'` 就在 [SOM_MARKS] 文本里（文本+世界知识即可）。condition 级 ablation（psom vs som）才是干净证据。本 probe Q2 无文本可抄，才构成纯像素证据。
3. **spatial-intent 切片无判别力**: "second row" 类任务 B1 0/6、B2 0/3 全挂 — 对两个 4B 都是天花板，不构成 B1/B2 对比证据。

## 关联

- B2 dom diag digest: `B2_dom_classifieds_diag_digest.md`（agent-limit 主因 + B-260 Magento gap）
- 实验笔记 §325（B2 dom diag）/ §326（本 probe）
- B2 som run: `B2_som_classifieds_..._R23029`（截至 probe 时 32 ep，仍在 A100 跑）
