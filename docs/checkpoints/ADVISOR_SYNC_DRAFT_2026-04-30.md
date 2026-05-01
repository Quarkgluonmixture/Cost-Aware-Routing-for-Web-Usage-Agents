# 学长 sync 草稿（2026-04-30）

**目的**: 简略 sync 进展 + 约会议讨论细节
**长度**: ~250 字

---

## 学长好，

想跟您简略同步一下，具体细节希望能开个会聊一下。

**1. Phantom-SoM 主线**：finding 出来了，4-fold drop-in property（cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one oracle 1.7-3.3pp）。Section 1-3 paper 草稿已写完，evidence 95% ready。

**2. VWA framework bugs 发现**：今天系统 audit 发现了 37 个 scaffold-level bugs（5-tier audit + 文献支撑 / Gemini DR 综述对照），其中最严重的 dispatch bug 让 94.4% 的失败 click 落在错误 DOM target 上。今天已经修完（Phase A 4-cluster patch），pilot 验证 PASS。这个 bug 影响 absolute SR 数字，但不影响 cross-mode 比较（symmetric contamination + Vision counter-evidence），所以 Phantom finding 仍 valid。这本身可能也是一个值得披露的副 contribution。

**3. 资源障碍**：UCL Myriad 申请下来了我也试着用了，但 UCL 防火墙 drop Tailscale CGNAT 段，**Myriad 不能 reach 我家里 quark Windows 的 VWA docker**，物理级 blocked，没法绕过（详见 `docs/reference/MYRIAD_SMOKE_REPORT.md`）。

DGX GPU 争抢的实测数据：B1 P-text cls 现在跑 30h 完成 225/234 ep (96%)，**~8 min/ep average**（peak 时段降到 4 ep/h ≈ 15 min/ep）。对比 B0 proxy（无 GPU 争抢）~3.5 min/ep。即 **DGX shared 比 dedicated 资源慢 2-3× average，peak 时段 5-10×**。Phase A 14-cell × 234 ep 全跑：DGX shared 需 ~437h（~18 天 24/7 wallclock），独占 4090 估 ~87-145h（3-6 天）。

**4. 提议**：调研 RunPod 4090 dedicated 只要 $0.6/h。基于实测吞吐:
- 4090 dedicated 估 ~87-145 GPU hours × $0.6/h = **~$52-87 actual**
- + 30% buffer (crash/retry/idle): **~$70-115 reasonable estimate**
- 申请 **$200 budget** 留 head-room for additional probes (Q3 B0 multi-call extended verify ~$10, Tier 5 evaluator probe ~$20, P-prompt diamond shop ~$30, Section 5 ad-hoc query ~$20).

**Wallclock 影响是 deal-breaker**: paper data ready 时间从 ~3 周 (DGX) 缩到 ~1 周 (4090). paper writing + 学长 review 时间从此 unblocked.

**问您几件事**：
- RunPod 经费可不可以走？走什么流程？
- Phantom finding 方向您觉得 OK 吗？
- VWA bug 这一发现要不要单独成文？
- **paper-strategy 4 个 framing 决定** (我 audit + 自己 sanity check + 05-01 deep discussion 后, 这四个真 ask 您):
  1. **Early-stop mechanism — design decision (我 lean Option A 全 cancel)** 🆕 重写 2026-05-01:
     - 之前 framing 是 "early-stop bias on micro metrics, A/B/C measurement options" 实际逃避了 design decision
     - 真问题: agent system 是否包含 early-stop? 它影响**全 4 dimension** (Outcome/Macro/Micro/Efficiency), 不只 micro layer (e.g. SR 上 task 没机会"自然结束", Macro action freq 是 censored 数据, Efficiency cost diff 部分来自早停 frequency)
     - Phase A Cluster 3 (fuzzy cycle hash min_reps=5) 是 partial mitigation 不是 full cancel
     - **Option A (lean)**: 14-cell rerun 全 cancel early-stop, +$1300 cost, paper rigor 全 dim clean
     - Option B: 全保留, accept cross-dim systemic confound
     - Option C: hybrid — main 14-cell with early-stop + 1-2 mechanism cells without, +$200
     - **学长这个 design decision 您 lean 哪个?** Option A 我倾向, 但 paper rigor vs cost 是真 trade-off
  2. **B0 pre/post Phase A 数据 sampling 不对称**: Phase A 之前 B0 用 T=0.1 stochastic, 之后 T=0 greedy. archived data + 14-cell re-run 是不同 sampling regime. 直接合 vs 只用 post-Phase-A 重跑 数据 (弃 archived)?
  3. **Paper hook reframe — phantom routing space (3 arms) vs Phantom-SoM (1 arm)** 🆕 2026-05-01:
     - **数据 trigger**: B0 reddit (唯一 6-mode 完整 cell) 显示 P-text +3.81pp / P-SoM +3.33pp / P-prompt +2.86pp 4-mode drop-one 全 sig；6-mode oracle vs 3-mode +7.14pp [3.81, 10.48] sig；P-prompt marginal +1.90pp sig 验证不冗余。即 **3 个 phantom arms 都贡献 unique tasks**, 旧 hook "Phantom-SoM is hidden 4th routing arm" 字面不准 (实际 3 arms / 6 modes total)
     - **新 hook 候选**: "**hidden phantom routing space**, boundary = 'no annotated image', 内含 3 routing arms 共享 4-fold drop-in property"。P-SoM 仍是 paper hero (cube center, axis 1+2 compound, representative arm)
     - **§2 cube boundary 重新定义**: 旧 "why 5 not 8" 用 mismatched parsing 论证；新 framing 用 "no annotated image" boundary 论证 — 排除的 3 个 cube corners (#2/#4/#6) 都因 image cost 拉齐 SoM 而 violate phantom space boundary，跟 parsing 是否 matched 无关。P-prompt 不是 mismatched-redundant 因为有真 LLM 机制
     - **LLM 机制 layer 升级**:
       - Axis 1 (P-text): AXTree → `[SOM_MARKS]` 把任务 ontology 从 "web browsing" reframe 到 "indexed selection" (browse → select 切换)
       - Axis 2 (P-prompt): SoM-style prompt 不用图仍 activate "visual-mark referencing" mental model, agent 不用图能 recover ~70-80% visual structure info from textual cues (lit anchors: **Mirage Effect** Asadi et al. 2026 arXiv:2603.21687 Stanford 70-80% accuracy of with-image; **Scaffold Effect** Vu & Balloccu 2026 prompt mentioning modality alone explains 70-80% perf shift; **Cross-modal flow** Kaduri et al. middle-layer query-token storage; 详 phantom_som.md + 实验笔记 §18 §25)
     - **Strategic value**: 把 contribution 从 "1 个 mode" 升级到 "1 个 routing dimension"，dimension discovery 比 single-arm finding 顶刊概率高
     - **paper conceptual structure 同步升级 (笔记 §108 详 chronicle)**:
       - **Evidence vs Explanation 严格分两层**: Evidence = 2D organize (4 测量类型 × 4 cross-X 比较 axis), Explanation = 1D zoom scale (Zoom 1 architectural / Zoom 2 behavioral M1/M2 / Zoom 3 named phenomena Mirage/Scaffold/Sclar / Zoom 4 model-internal Cross-modal flow/SteerMoE)
       - **M1/M2 mechanism activation 2x2 framework**: 4 phantom corners 是 M1 (Image-mirage) × M2 (Flat-list) 2x2 activation pattern。P-SoM 是 M1+M2 compound state, transformer attention nonlinear combination produces emergent capability (3 unique tasks B0 reddit), 不是简单叠加
       - **Approach 2 architectural completeness (Zoom 1)**: phantom space 锁 image=✗ 只 vary 2 input dim, M1/M2 by design exhaustive (deductive 不依赖 finite data verify)
     - **学长这个 reframe OK 吗?** 关键是 (a) phantom space 命名 / (b) 3-arm framing 取代 4th-arm framing / (c) "no annotated image" boundary 论证 / (d) evidence-explanation 分层 + Zoom 1-4 / (e) M1/M2 activation 2x2 framework
  4. **SteerMoE scope decision (您发的 ICLR 2026 paper)** 🆕 2026-05-01:
     - 谢谢您发 SteerMoE (Fayyaz 2026). 我读完后认为它在 Zoom 4 (model-internal mechanism) 层, 跟我们的 Zoom 1+2+3 主 paper 是 layered complementary
     - **Architectural fit**: B0 = Qwen3-VL-235B-A22B 是 MoE, 跟 SteerMoE 实验用的 Qwen3-30B-A3B 是 architectural cousin. Methodology 几乎可直接 transfer (paired examples → expert RD → router logits steering)
     - **B0 vision-grounding 可能 concentrated in subset of experts 假说**: SteerMoE "Alignment Faking" 框架 → phantom routing 通过 obs/prompt config 绕过 vision-grounding experts
     - **3 个 scope option**:
       - (i) **不 self-probe, paper §8 future work 列举 SteerMoE methodology** (我 lean): paper 主线 Zoom 1+2+3 已 solid, Zoom 4 留 follow-up paper 自然 sequence
       - (ii) Self-probe small-scale: local deploy Qwen3-30B-A3B (no vision) 作 architectural proxy, 跑 paired phantom prompts → expert RD. 但 paper 不能直接 claim "B0 phantom mechanism = expert routing" (跨 model class)
       - (iii) Self-probe full-scale: local deploy Qwen3-VL-235B-A22B 4×4090 ~$400-600 cost, 直接 probe B0. 但超 RunPod $200 budget + 增加 paper scope
     - **学长这个 scope 您 lean 哪个?** option (i) 我倾向, 让 paper sequence 自然成 trilogy; option (ii)/(iii) 是把 SteerMoE 整合进主 paper

**🆕 实证 anchor for B0/B1 reproducibility 不对称** (我跑了 cheap probe 验证):
- Probe: 5 calls × T=0 + top_p=1.0 + seed=42 forwarded × 同 prompt × proxy API
- Result: 5/5 byte-level distinct outputs **但** 5/5 same action (`click element_id=5`)
- **Conclusion**: B0 token-level non-deterministic, decision-level convergent
- Section 4 disclosure paragraph drafted in `docs/analysis/cross_sites/probe_b37_api_determinism.md`
- Cost: $0.005, 给 paper 实证 anchor 而不是 "trust the proxy" claim

详 `docs/reference/PAPER_STRATEGY_OPEN_QUESTIONS.md` (audit 9 个 issues, 7 个我自决 disclose, 上面 2 个真 ask 您)

具体细节希望能开个会聊聊（30-45 分钟应该够），您方便的时候定个时间。

详细文档我都整理在 repo 里了：
- `docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md`
- `docs/reference/MYRIAD_SMOKE_REPORT.md`
- `docs/checkpoints/paper_planning.md`

谢谢学长！

—— 嘉名
