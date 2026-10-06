# LLM router pilot — 2026-09-09 raw data

**状态: 数据已落盘, 结论未定。** user 2026-09-09 判定分析尚有缺陷, 留新 session 想清楚再写
笔记/台账。本文件只记录**做了什么 + 参数 + 已知缺陷**, 不下结论。

## 跑了什么

Router 模型 = **GPT-5.6**, 两个 tier (`global.openai.gpt-5.6-luna` / `-terra`), 走 B0 那条 AWS proxy。
被路由的 agent = B0 Qwen3-VL-235B / B1 Qwen3-VL-4B / B2 Gemma-3-4B / **B5 = GPT-5.6-terra**。

| 版本 | 输入 | 决策空间 | 覆盖 |
|---|---|---|---|
| `llm_router.py` (v1) | 只有 task intent, zero-shot | 6 选 1 | 4 site × 2 tier, backbone-agnostic |
| `llm_router_v2.py` | + 全 40 条 failure rule + few-shot 标签 + base rate | 6 选 1 | cls_B0 × {含/不含 base rate} |
| `llm_router_v3.py` | + 三轴画像 (SR/cost/latency) + cost-aware 目标 + few-shot 带成本 | 6 选 1 | **11 cell**; `--group` 版另跑 4 cell |
| `llm_router_3class.py` | 同 v3, READ/LOOK/BOTH, arm-matched, 代表臂 fold 内选 | 3 选 1 | 11 cell |
| `llm_router_ablate.py` | position/描述 ablation | — | **写了但未跑** |

分析脚本 (不调 API): `build_full_table.py` (全量 per-task 表, 自检 7722/7722 对上
`per_task_sr.csv`) · `pareto_profile.py` · `strict_dom.py` · `eval_pareto.py` ·
`eval_final.py` · `class3_headroom.py` · `regex_costaware.py` · `need_image.py` ·
`b5_splice_window.py` · `router_headroom.py`

## 复现

```bash
source scripts/queues/_lib_paper_grade_gates.sh && load_proxy_api_key "$PWD" r && export PROXY_API_KEY
.venv/bin/python3 scripts/.../build_full_table.py           # -> tables/full_table2.json
.venv/bin/python3 .../llm_router_v3.py --cell cls_B0 --site classifieds --out x.json
.venv/bin/python3 .../eval_final.py
```
`full_table2.json` 的每条 = `[success, cost_usd, latency_ms, tokens, steps]`, key `<cell>|<mode>`。
run 选择规则 = 每 (cell,mode) 取 episode 最多的 run, 并列取最早; 与 canonical `per_task_sr.csv`
在重叠 6 格逐 task 100% 一致。

## ⚠️ 已知缺陷 (新 session 必读, 逐条决定要不要修)

1. **position bias 未排除** — 六个选项里 `som` 固定排第 2, 且它的描述是唯一提到
   "annotated screenshot **plus** numbered list" 的, 听上去信息最全。"85–95% 选 som"
   有多少是排版造成的**未测**。`llm_router_ablate.py` 已写 (rev / alpha 两变体) 但**没跑**。
2. **`cls_B5` 是自路由** — router 与 agent 同为 GPT-5.6。该格任何结论都带这个混淆。
3. **`cheap=no-image` 是 post-hoc 选的** — `regex_costaware.py` 两个变体里, 事后挑了表现好的
   那个报。要进 paper 必须预先声明或另找 held-out。
4. **GroupKFold 只跑 4 格** (cls_B0/cls_B1/red_B0/red_B2), 其余 7 格只有随机 fold。
   ⚠️ 术语: user 2026-09-09 裁定 template sibling **不是 leakage** —— 随机 fold 与
   GroupKFold 回答两个不同的部署问题(稳态 vs 冷启动), 都合法。项目自己的措辞见
   `router_model_sweep.py:1087`「template sibling 相关性」。
5. **`corr(best−2nd gap, Δbest) = −0.696` 的 n = 10** — 相关系数极不稳, 不能当结论。
6. **降本三源分解 (A 26.9% / B 3.3% / C 1.0%) 全是 oracle 上界**, 不是可实现值。
   A 需判定「无解」, B/C 需判定「便宜臂也能解」, 判定器的实际精度未测。
7. **router overhead** 按 luna 实测均摊 (`$0.0539/224` + 3s) 计入 cost/latency 轴;
   terra 贵约 5×, 未分别计。
8. **`shop_B1` 的 psom/ptext 是部分数据** (161–435 episodes), 交集后 n=163。
9. **所有 SR 差值未做显著性检验**, 只对着 `noise_floor_inventory.json` 的 rerun band 读;
   6 个 cell 没有 replicate, 那几格的差值读不了。
10. 🔴 **per-step splice 窗口那条结论是错的, 字段用错了** (2026-09-09 当天更正)。
    `b5_splice_window.py` 用 `obs_url` 判分岔, 但 **`obs_url` 是动作执行之后的 URL**
    (对照同一条 step record 的 `state_digest.url_before` / `url_after`)。路由决策发生在
    **看到页面之后、动作之前**, 该用 `url_before`。重算 (B0·cls, dom/som/vision/ptext 四臂):
    - **step0 的 `url_before` 四 mode 一致 = 224/224 = 100.0%** ⇒ **第 0 步 splice 严格合法**
    - 中位分岔步 **2.0** (不是 1.0); 决策窗口 ≥1 步 **100%**, ≥2 步 **64%**, ≥3 步 24–37%
    ⇒ 「per-step / online routing 是死路」这个判断**作废**。可行的形态是
    **one-step-lookahead**: 先用便宜 mode 跑第 0 步, 拿它的 `action.thought`
    (中位 132–135 chars) + `confidence` (`mean_logprob`/`mean_margin`) + observation,
    再决定要不要升级; 全部可用现有 18,294 episodes 离线评估, 不需新 fire。
    ⚠️ 未被绕开的约束: 路由**目标**的标签仍只在 (便宜失败 ∧ 贵的成功) 上有定义 ——
    输入端信息量涨了, 输出端监督稀疏性没变。
    ⚠️ `b5_splice_window.py` 的数字**未重跑**, 文件里仍是 `obs_url` 口径, 引用前先改字段。

## 数据来源
- `results/phantom_paper/per_task_sr.csv` (canonical, 6 格, 用于自检)
- `results/{visualwebarena,webarena}/phase1/*/` episode summaries (11 格)
- `docs/analysis/cross_sites/noise_floor_inventory.json` (rerun band)
- `docs/analysis/cross_sites/cross_mode_failure_signatures.json` (40 条 failure rule)
- task intent + difficulty 字段: `*/task_configs/*.json`

## 2026-09-09 晚, 新 session 的复算 (不调 API, 不点火)

产物 → **`docs/analysis/cross_sites/one_step_lookahead_2026-09-09.md`** (tracked digest, 表全在那) ·
笔记 §505 · 台账 §505 ×9。脚本 `scripts/{extract_step0.py,lookahead_eval.py,bandit_replay.py}`,
原始输出 `lookahead/` (含 21,291-episode 的 step-0 特征表 `step0.jsonl.gz`)。

对上面「已知缺陷」的处置:
- **#5 (corr −0.696)**: 复现为 −0.708 (v3, n=11), 但 bootstrap 95% [−0.94, +0.63]; 且结构上必负
  (router 一偏离 best 就付 gap)。能支持「两个相近臂 ⇒ 有活路」的量是 oracle 加一臂增益 vs gap =
  +0.37 [−0.10, +0.69]。**弃, 不是 finding。**
- **#10 (决策窗口)**: 「≥2 步 64%」是 URL 相等, 不是状态相等; cheap 与 rich 在 step 0 的 url_after
  就有 27–70% 不同。**只有 step-0 推理-peek 严格合法**, 本次的 lookahead 全在这一点上评。
- **新 #11**: 重跑翻转在 step 0 不可见 —— 18 对 replicate / 326 个翻转 task 的配对检验, 所有
  step-0 信号 P(succ>fail) 0.47–0.52。lookahead 能看到的只有 task 级信息, 而那些 intent + 页面
  统计 (§457) 已经免费给了: MODEL0 对 OBS0 的 AUROC 增量中位 −0.007…+0.021。
- **新 #12**: 三臂 (dom/vision/som) 收敛不改 verdict: which-mode 最小类 ≥10 行仍只 2/11;
  3 臂 bandit 11/11 输 fixed-best; WA 两格天花板丢 60–80%。
- **新 #13**: contextual bandit (6/2/3 臂, LinUCB/Thompson) 11/11 格低于 fixed-best —— sVJH 的
  bandit / online-after-partial-interaction 两条已用数据关掉。
