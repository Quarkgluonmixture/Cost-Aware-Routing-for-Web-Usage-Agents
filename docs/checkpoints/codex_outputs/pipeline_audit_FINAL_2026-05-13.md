## Verdict (one sentence)
这 5 个脚本还不是 paper-grade：核心问题不是小统计瑕疵，而是 **aggregate geometry / in-sample steering / formatter mismatch / silent partial extraction** 会直接污染 §5 mechanism claims。

## Critical findings (P0 — must fix before next extraction)

- **Script + line**: `scripts/analysis/stage4_axis2_layer_profile.py:60-72`
- **Bug**: 先按 mode 做全局 mean，再算 cosine gap：`mean → cosine`。这是 naive of-means，不是 per-task paired geometry；任务内容差异会被混进“axis-2 layer profile”。
- **Impact**: §5.7 “axis-2 在 L23 emergence/peak” 可能只是任务分布/step 分布的 aggregate artifact。
- **Fix**: 改成 `(task_id, step)` matched pairs：每个 pair 先算 per-sample / per-task gap，再报告 mean/median + paired bootstrap CI；全局 mean curve 只能做 supplement。
- **Effort**: 1-2h

- **Script + line**: `scripts/mechanistic/run_stage4_method44_v2_sweep.py:104-112`, `:115-117`, `:125-179`
- **Bug**: steering direction 用同一个 NPZ 全样本均值构造，然后在同 tier/tasks 上评估；这是 in-sample causal steering，不是 held-out test。
- **Impact**: Method 4.4 “steering works / HDMI reliability” claim 会被 reviewer 当成 leakage。你测的是 memorized cohort direction，不是 transferable mechanism。
- **Fix**: 增加 `--direction-npz/--eval-manifest/--heldout-task-ids`，默认 split by task；direction 只从 train tasks 算，sweep 只在 held-out tasks/steps 上跑。
- **Effort**: 2-3h

- **Script + line**: `scripts/mechanistic/run_stage4_format_variation_extract.py:8-11`, `:169-171`
- **Bug**: docstring 说 variants 要和 “baseline P-text” 聚类，但 `ALL_MODES` 只有 8 variants + `dom` + `som`，没有 `phantom_text` / `phantom_som` text-only baseline。
- **Impact**: H1 test 的 anchor 错了。你无法证明 “marks-like variants cluster with P-text”，因为 P-text 根本没抽。
- **Fix**: 把 `phantom_text` 和 `phantom_som` 加入 baselines；保存 mode schema/version；下游分析必须显式比较 variants↔phantom_text、variants↔dom、variants↔som。
- **Effort**: 45-90min

- **Script + line**: `scripts/mechanistic/run_stage4_format_variation_extract.py:181-196`
- **Bug**: extraction failure 被 `continue` 静默吞掉，最后仍 `np.stack` 输出 partial NPZ；missing obs 也只是 warning+skip。
- **Impact**: matched-N v2 一落地就可能变成 ragged N，各 mode/task/step 不平衡，后续 cosine/logit/steering 全部不可解释。
- **Fix**: 建 expected grid `(task, step, mode)`；任何 cell failure 默认 raise。若允许 resume，写 failure manifest，并要求 `--allow-partial` 才能输出 NPZ。
- **Effort**: 1h

## Medium findings (P1)

- **Script + line**: `scripts/analysis/stage4_axis2_per_task_fragility.py:73-83`
- **Bug**: 每个 task/mode 只要有任意样本就参与，未 assert exactly 2 steps / complete modes。
- **Impact**: “24 tasks × 2 steps” 的 fragility 表可能实际混入 partial tasks。
- **Fix**: enforce complete grid；输出 excluded cells；默认 fail closed。
- **Effort**: 30-45min

- **Script + line**: `scripts/analysis/stage4_axis2_per_task_fragility.py:89-111`
- **Bug**: 只报 mean/median/IQR，没有 bootstrap CI；也没有 paired test / multiple-comparison correction。
- **Impact**: “broad vs sparse axis-2” 判读没有不确定性，阈值列不够 paper-grade。
- **Fix**: task-level bootstrap 10k；报告 CI for median/mean/frac_gt；多 pair/layer 做 Holm or FDR。
- **Effort**: 1h

- **Script + line**: `scripts/mechanistic/run_stage4_method44_v2_sweep.py:87-89`, `:197-211`
- **Bug**: `is_json_valid` 只检查 startswith `{` 或 `"`，但 markdown 写 “JSON valid”。这不是 JSON validity，是 first-character heuristic。
- **Impact**: HDMI selectivity inflated；over-steered broken generations 会被记成 valid。
- **Fix**: `json.loads` parse；若只想 envelope metric，改名 `json_envelope_rate`，不要叫 valid。
- **Effort**: 15min

- **Script + line**: `scripts/mechanistic/run_stage2b_continuation_pilot.py:237-259`
- **Bug**: provenance snapshot failure non-fatal。
- **Impact**: paper extraction 可以在无 git/model/env pin 的情况下继续跑，后面无法复现。
- **Fix**: 默认 raise；只在 `--allow-missing-provenance` 下 warning。
- **Effort**: 20min

- **Script + line**: `scripts/mechanistic/run_stage4_method44_v2_sweep.py:55`
- **Bug**: 使用 `hidden_states.npz`，而其他 stage4 scripts 默认 `hidden_states_v2_fixed.npz`。
- **Impact**: steering direction 和 layer-profile/PCA/logit-lens 可能不是同一批 hidden states。
- **Fix**: default 改成 v2 fixed；CLI 显式传 `--npz` 并写入 output config。
- **Effort**: 20min

## Low / cosmetic (P2)

- **Script + line**: `scripts/analysis/stage4_axis2_layer_profile.py:101`, `:114`
- **Bug**: markdown 硬编码 “288 ex”，不从 NPZ 读取。
- **Impact**: matched-N v2 / reverse-tier 运行后报告会撒谎。
- **Fix**: 从 `len(H)`、mode counts、task counts 动态生成。
- **Effort**: 15min

- **Script + line**: `scripts/mechanistic/run_stage4_format_variation_extract.py:210-217`
- **Bug**: 没有 dtype normalization / shape assert。
- **Impact**: extractor 返回 dtype 一变，NPZ 精度和 downstream cosine 数值漂移。
- **Fix**: `np.asarray(h, dtype=np.float32)`，assert `(n_layers, hidden_dim)` stable。
- **Effort**: 20min

- **Script + line**: `scripts/mechanistic/run_stage2b_continuation_pilot.py:441-451`
- **Bug**: plot 用 mean ± std，不是 CI。
- **Impact**: pilot 图可读，但不能作为 paper magnitude。
- **Fix**: bootstrap CI band。
- **Effort**: 45min

## Out-of-box callout
最容易被初读 reviewer 漏掉的是 formatter anchor 错误：`run_stage4_format_variation_extract.py:8-11` 声称 variants should cluster with baseline P-text，但 `run_stage4_format_variation_extract.py:169-171` 实际没有抽 `phantom_text`。这不是统计问题，是 experiment object 不存在；H1 的主要对照组在数据里缺席。

## Cross-pipeline coherence flag
我预期 Claude 那边会看到 layer-index/dtype/NPZ coherence 问题；你这边已经有明显 mismatch：analysis scripts 默认读 `hidden_states_v2_fixed.npz`，但 Method 4.4 sweep 读 `hidden_states.npz`。另一个风险是 layer convention：Method 4.4 明写 `patcher.layers[L] ↔ npz[:, L+1, :]`，而 layer-profile 直接把 `H[:, L, :]` 标成 `L0 embedding, L36 final block`。如果 PCA/logit-lens 用另一套 convention，L17/L23 claim 会错一层。

## One thing to fix tonight (1-3h)
先修 **format-variation extractor grid/provenance/baselines**：加入 `phantom_text`/`phantom_som`，expected-grid fail-closed，float32 cast，output provenance/config hash。这个改动最杠杆，因为下一轮 matched-N v2 一旦抽错，后面所有 analysis 都是在垃圾 NPZ 上做漂亮图。

=== DONE ===