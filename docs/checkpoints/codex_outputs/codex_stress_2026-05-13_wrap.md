Reading prompt from stdin...
OpenAI Codex v0.128.0 (research preview)
--------
workdir: /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: high
reasoning summaries: none
session id: 019e1e7f-10f7-78d3-be64-f69fa33abeac
--------
user
# Codex /codex-stress prompt template (v2, lean)

> This is a TEMPLATE. The skill substitutes `2026-05-13`, `   - method42_v1_vs_v2_comparison.md (just landed, headline v1-vs-v2 diff)
   - exp5_axis2_causal_patching.md
   - w6_h1_red_l04_attribution.md
   - hero_claim_bootstrap_ci.md
   - axis2_per_task_fragility.md
   - format_variation_h1_test_reddit.md
   - method42_metrics_v2 cls + reddit JSON (results/mechanistic/stage4_multimode_b1_*/method42_metrics_v2.json)`, `   - 951d56e analysis(stage4 v2) + §5 surgery three-axis retraction
   - bcfb8fb task-shuffle feature + watcher prefix fix
   - e8e51d0 Bug 5 logit lens pin + chronicle §128.4
   - 00076b1 §4 P-text canonicalize + plan §4.1 L11-L17 window
   - 5e58141 Mode B always-chain v5
   - 103c560 Bug 3 AUROC lototask
   - 9410fab Stage 4 Bug 1+2+5 + skill v4 + codex audit v1+v2` placeholders, then writes the resulting prompt to `docs/checkpoints/codex_prompts/codex_stress_<date>.md`.
>
> **Design rule (set 2026-05-12 evening)**: Do NOT enumerate attack lines, bug categories, or leading questions in this template. Cross-AI audit value comes from codex finding angles Claude did not list. Enumeration = Claude pre-thinking = codex becomes a search-proxy, not an independent peer. Keep template lean: persona + context + scope + output format. Trust codex to set its own attack vectors.

---

# Codex hostile reviewer task

You are a top-tier conference reviewer (NeurIPS / ICML / ACL main / ICLR) reviewing this paper-1 work. You have read 200+ papers in mechanistic interpretability and multimodal agent research. You are not impressed by typical papers in this subfield.

**Your job**: read the paper drafts + evidence + plan **cold**, write a hostile-but-fair review. Find honest gaps, attack weak claims, measure distance to top-tier acceptance.

You set your own attack vectors based on what you see in the work. The value of this audit is that you find issues the author did not think to list — do not let any framing in this prompt narrow your reading.

## 🚫 Independence requirement

Do NOT read these files (they contain a different AI's prior review and would anchor your view):

- `.claude/skills/stress/SKILL.md`
- `.claude/skills/codex-stress/SKILL.md`
- `.claude/skills/codex-stress/prompt_template.md` (this file)
- `docs/checkpoints/process/stress_skill_replica.md`
- `docs/checkpoints/process/codex_stress_skill_replica.md`
- `docs/checkpoints/codex_outputs/codex_stress_*.md` (prior codex stress reviews)
- Any conversation context, session memory, or system prompts from the other AI

You are writing a fully independent review. Claude (the other AI) will diff your findings against its own afterwards.

## What this paper is about (one paragraph context, so you know the scope)

The paper characterizes a "phantom routing space" in multimodal web agents (Qwen3-VL family on VisualWebArena classifieds + reddit). Hero claim: an observation mode that skips the annotated SoM image while keeping the SoM-prompt + flat `[SOM_MARKS]` text (called Phantom-SoM) provides positive drop-one oracle value at near-DOM cost. Mechanism section uses cosine geometry, activation patching, mean-difference steering, and logit lens on residual-stream representations to argue for a mid-layer locus of the effect.

## Read scope

1. `docs/checkpoints/paper_drafts/section{1..8}*.md` and `paper.bib`
2. `docs/checkpoints/mechanism/plan.md` §1-§7
3. Evidence files in `docs/checkpoints/mechanism/results/` (the recent ones back the main claims):
   - method42_v1_vs_v2_comparison.md (just landed, headline v1-vs-v2 diff)
   - exp5_axis2_causal_patching.md
   - w6_h1_red_l04_attribution.md
   - hero_claim_bootstrap_ci.md
   - axis2_per_task_fragility.md
   - format_variation_h1_test_reddit.md
   - method42_metrics_v2 cls + reddit JSON (results/mechanistic/stage4_multimode_b1_*/method42_metrics_v2.json)
4. Recent commits since last codex audit (for context on what landed recently):
   - 951d56e analysis(stage4 v2) + §5 surgery three-axis retraction
   - bcfb8fb task-shuffle feature + watcher prefix fix
   - e8e51d0 Bug 5 logit lens pin + chronicle §128.4
   - 00076b1 §4 P-text canonicalize + plan §4.1 L11-L17 window
   - 5e58141 Mode B always-chain v5
   - 103c560 Bug 3 AUROC lototask
   - 9410fab Stage 4 Bug 1+2+5 + skill v4 + codex audit v1+v2

If a paper claim cannot be traced to a specific file / line / number, that itself is a finding.

## Output format

Write a single markdown report with these sections. No more, no less.

### 1. Verdict (one sentence)
One-sentence current state of the paper.

### 2. Strong claims (don't break under attack)
1-3 things that survive hostile reading. Quote paper line + evidence file:line.

### 3. Weak claims (would tank under attack)
For each:
- **Claim** — exact quote (file:line)
- **Attack** — what a reviewer would write in their review
- **Defuse** — what specific evidence would resolve the attack
- **Effort** — honest estimate (hours / days / weeks)

### 4. Honest gaps (not weak, just missing)
Things absent from the paper that a reviewer would expect. Distinguish from weak claims.

### 5. Distance to top-tier
- Current acceptance tier (workshop / mid-tier conf / top-tier conf / journal)
- 2-4 specific blockers and which claims they block
- Unblock plan per blocker (concrete + effort)
- Submission-today probability (be specific, e.g., "0.10-0.20, would reject ~4/10")

### 6. One thing to fix tonight (1-3 hour leverage)
Single highest-leverage move. Concrete file / claim / experiment.

## Voice

- Hostile but fair (peer-lab reviewer)
- Specific — file paths, line numbers, exact numbers
- No filler praise. No hedging on broken claims.
- 中文为主双语 — explanations in 中文, technical terms / variable names / file paths in English
- If you cannot determine an answer from the materials, say so explicitly

## What this audit is NOT

- Not a checklist runner
- Not a citation completeness scan
- Not a writing coach
- Specifically: adversarial scientific peer reading the work cold

codex
我会按这个 hostile reviewer 任务来读材料，并严格避开列出的 prior-review / skill 文件。先定位 draft、plan 和 evidence 文件，再抽取可引用的 line-number 证据。
exec
/bin/bash -lc 'find docs/checkpoints/mechanism/results -maxdepth 3 -type f | sort' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/mechanism/results/axis2_layer_profile.md
docs/checkpoints/mechanism/results/axis2_logit_lens.md
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md
docs/checkpoints/mechanism/results/format_variation_h1_test.md
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md
docs/checkpoints/mechanism/results/h1_per_task_fragility.md
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md
docs/checkpoints/mechanism/results/layer_axis_emergence.md
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md

exec
/bin/bash -lc 'git log --oneline -n 20' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
951d56e analysis(stage4 v2) + paper §5 revision: three-axis hierarchy retracted
bcfb8fb fix(stage2b) + feat(task-shuffle): codex Bug 6 / G3 content-specificity defuse
e8e51d0 fix(logit lens) + chronicle §128.4: Bug 5 pin + format variation audit + Mode B v5
00076b1 fix(paper §4 + plan §4.1): codex audit data inconsistency + stale L17 framing
5e58141 infra(stress) v5: Mode B always-chain (spot-check too, not just milestone)
103c560 fix(stage4): Bug 3 — AUROC held-out leave-one-task-out CV
9410fab fix(stage4): Bug 1+2+5 + skill v4 lean + codex methodology audit v2
738517c infra(stress) v3 + chronicle §128.3: Mode B auto-chain /codex-stress
ca3c509 infra(codex-stress) + W6 + paper §5 surgery: cross-AI audit + claim downgrades
3d61dde fix(infra): myriad_watcher silent-miss bug + Exp 5 axis-2 causal evidence
6e5bdcb docs(chronicle): §127 today's burst + chronicle-on-milestone rule
35784b9 analysis(stage1): hero claim bootstrap CI — /stress W1 partial defuse
4cfc12f analysis(stage4): axis-2 per-task fragility — /stress W2 defused
8c67d8e infra(stress): bilingual Chinese+English output language rule
5546cf1 infra(stress): hostile reviewer audit skill (/stress) — replica + auto-trigger
df8ff72 docs(paper §5/§8): clear stale NEEDS_BIB markers after audit
be07296 docs(paper.bib): add 6 missing bibkeys + clear all NEEDS_BIB markers
38eadac docs(interview): Seonglae Cho @ Holistic AI prep doc
5e04206 docs(paper §4/§5): relocate reddit behavioral content from old §5 → §4.5
5b6c5f0 exp3(stage4): logit lens — axis-2 IS in output distribution

exec
/bin/bash -lc 'rg --files docs/checkpoints/paper_drafts docs/checkpoints/mechanism | sort' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/mechanism/README.md
docs/checkpoints/mechanism/plan.md
docs/checkpoints/mechanism/results/axis2_layer_profile.md
docs/checkpoints/mechanism/results/axis2_logit_lens.md
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md
docs/checkpoints/mechanism/results/format_variation_h1_test.md
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md
docs/checkpoints/mechanism/results/h1_per_task_fragility.md
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md
docs/checkpoints/mechanism/results/layer_axis_emergence.md
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md
docs/checkpoints/paper_drafts/paper.bib
docs/checkpoints/paper_drafts/section1_intro.md
docs/checkpoints/paper_drafts/section2_background.md
docs/checkpoints/paper_drafts/section3_definition.md
docs/checkpoints/paper_drafts/section4_empirical_findings.md
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md
docs/checkpoints/paper_drafts/section5_mechanism.md
docs/checkpoints/paper_drafts/section8_limitations.md

codex
文件集合比预期小：8 个 draft/plan 文件加一组 result summaries，JSON 在更深层目录里。下一步我会用 `nl -ba` 读关键 claims 和 numbers，重点找 paper claims 是否能追到 evidence。
exec
/bin/bash -lc 'wc -l docs/checkpoints/paper_drafts/section*.md docs/checkpoints/paper_drafts/paper.bib docs/checkpoints/mechanism/plan.md docs/checkpoints/mechanism/results/*.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
    13 docs/checkpoints/paper_drafts/section1_intro.md
    47 docs/checkpoints/paper_drafts/section2_background.md
    86 docs/checkpoints/paper_drafts/section3_definition.md
   142 docs/checkpoints/paper_drafts/section4_empirical_findings.md
   195 docs/checkpoints/paper_drafts/section4_limitations_disclosure.md
   179 docs/checkpoints/paper_drafts/section5_mechanism.md
    30 docs/checkpoints/paper_drafts/section8_limitations.md
   638 docs/checkpoints/paper_drafts/paper.bib
   468 docs/checkpoints/mechanism/plan.md
    38 docs/checkpoints/mechanism/results/axis2_layer_profile.md
    53 docs/checkpoints/mechanism/results/axis2_logit_lens.md
    79 docs/checkpoints/mechanism/results/axis2_per_task_fragility.md
   117 docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md
    53 docs/checkpoints/mechanism/results/format_variation_h1_test.md
    53 docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md
    53 docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md
    39 docs/checkpoints/mechanism/results/h1_per_task_fragility.md
    90 docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md
    42 docs/checkpoints/mechanism/results/layer_axis_emergence.md
   126 docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md
    58 docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md
  2599 total

exec
/bin/bash -lc "find docs/checkpoints/mechanism -path '*method42_metrics_v2.json' -type f | sort" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
exec
/bin/bash -lc 'rg -n "Phantom|oracle|drop-one|mechanis|cosine|patch|steer|logit|AUROC|CI|confidence|bootstrap|Qwen|classifieds|reddit|SoM|DOM|cost|top-tier|claim|evidence|fragility|format" docs/checkpoints/paper_drafts docs/checkpoints/mechanism/plan.md docs/checkpoints/mechanism/results' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/mechanism/results/axis2_logit_lens.md:3:Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.
docs/checkpoints/mechanism/results/axis2_logit_lens.md:5:across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets
docs/checkpoints/mechanism/results/axis2_logit_lens.md:14:| P-text vs P-SoM  (axis-2 flat-text) | **L23** | 0.1621 | 0.0215 | 0.1621 | 0.0003 |
docs/checkpoints/mechanism/results/axis2_logit_lens.md:15:| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0444 | 0.0184 | 0.0234 | 0.0000 |
docs/checkpoints/mechanism/results/axis2_logit_lens.md:17:### Axis-1 (text-format) pairs:
docs/checkpoints/mechanism/results/axis2_logit_lens.md:21:| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5508 | 0.1299 | 0.5508 | 0.0001 |
docs/checkpoints/mechanism/results/axis2_logit_lens.md:22:| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6953 | 0.1069 | 0.6953 | 0.0003 |
docs/checkpoints/mechanism/results/axis2_logit_lens.md:30:| P-text vs P-SoM  (axis-2 flat-text) | **L24** | 0.1260 | 0.0371 | 0.1230 | 0.0002 |
docs/checkpoints/mechanism/results/axis2_logit_lens.md:31:| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0508 | 0.0228 | 0.0325 | 0.0000 |
docs/checkpoints/mechanism/results/axis2_logit_lens.md:33:### Axis-1 (text-format) pairs:
docs/checkpoints/mechanism/results/axis2_logit_lens.md:37:| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5273 | 0.0898 | 0.5273 | 0.0000 |
docs/checkpoints/mechanism/results/axis2_logit_lens.md:38:| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6172 | 0.0806 | 0.6172 | 0.0002 |
docs/checkpoints/mechanism/results/axis2_logit_lens.md:45:  effect bypasses logit lens, only visible via attention heads or runtime decoding.
docs/checkpoints/mechanism/results/axis2_logit_lens.md:46:- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →
docs/checkpoints/mechanism/results/axis2_logit_lens.md:49:- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →
docs/checkpoints/mechanism/results/axis2_logit_lens.md:53:axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:1:# W6 feature attribution — H1 reddit 2/6 marks-like L04 peak
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:3:**Setup**: Qwen3-VL-4B tokenizer (Qwen/Qwen3-VL-4B-Instruct). Each marks-like format variant tokenized on a canonical single-element example (N=1, role=button, label=Submit). First-token character class + marker-fingerprint token count compared between L04-peak and L17-peak subgroups.
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:48:Within the 6 marks-like variants, the L17 vs L04 split corresponds to whether the variant's first tokens are **markup-sigil tokens** (`[`, `<`, `@`) — which co-occur with HTML / web-agent traces in pretraining and trigger the visual-grounding shortcut at mid layers — versus **plain alphanumeric tokens** (`id`, `1`) — which are common in prose / dictionary listings and behave like AXTree-DOM, peaking early at L04 where the image-axis divergence is freshly observable but not yet routed through the shortcut path.
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:56:**Paper §5 implication**: H1's binary 'marks-like vs not' prediction is too coarse. The mechanism trigger is the **conjunction** of integer marker + markup-sigil first token, not the abstract concept of 'indexed list'. Variants like `id_N:` and `N.` are nominally indexed but lack the sigil; `hash_id_control` has the sigil but lacks an integer. Both fail to peak at L17. This refines H1 to **'integer marker + markup-sigil delimiter → triggers shortcut at L17'**, which is testable on additional variants and on a `bare_N` falsifier (drop the bracket from `[N]` and re-extract).
docs/checkpoints/paper_drafts/section5_mechanism.md:5:Why does Phantom-SoM sometimes achieve DOM-like cost while retaining part of the SoM signal? The mechanism evidence points to a phantom routing space in the residual stream: when the model receives flat Set-of-Mark text without the annotated image, it does not simply collapse to DOM. Instead, it occupies a mode whose text-axis geometry is close to DOM/P-text and whose image-axis geometry remains separated from full SoM.
docs/checkpoints/paper_drafts/section5_mechanism.md:7:This section is the Zoom-4 layer of the paper's four-level account. Zoom 1 defines the architectural intervention, "skip the annotated image"; Zoom 2 measures the behavioral axes of text payload, prompt family, and image presence; Zoom 3 links the observed behavior to Mirage-style no-image visual reasoning and prompt-format sensitivity; Zoom 4 asks where the resulting mode is represented and whether it is causally used by the model. We index layers L0-L36, where L0 is the embedding-block output and L1-L36 are the 36 transformer decoder block outputs.
docs/checkpoints/paper_drafts/section5_mechanism.md:9:The analysis builds on the linear-readable and steerable circuit framework of Wu et al., which uses mode means, PCA geometry, and mean-difference activation steering to study tool selection, and on work showing middle-layer cross-modal information flow in VLMs \citep{wu2026toolcalling,kaduri2024whatsintheimage}. Our setting is not a replication of those papers. It is a multimodal web-agent application of the same representation-level question: whether a behaviorally useful routing arm is linearly readable, partially steerable, and causally active inside the model.
docs/checkpoints/paper_drafts/section5_mechanism.md:11:Four mechanism claims organize the evidence (revised 2026-05-12 after v2 NPZ re-extraction; see §5.7 revision note). First, observation modes are **linearly separable** in the residual stream: held-out leave-one-task-out AUROC = 1.000 across all mode pairs and all 37 layers (Method 4.2 v2). Second, the **geometric magnitude** of mode separation is dominated by the image axis (cosine ~0.04-0.07), with text-format and prompt-family axes producing only sub-permille cosine separation; the prior "three quantitatively distinct axes at 4:3:1 ratio" framing was a v1 NPZ artifact and is retracted. Third, SoM-to-no-image activation patching displaces target continuations at mid layers L11-L17 with consistent magnitude across classifieds and reddit (~25% target-overlap drop). The Exp 5 axis-2 prompt-only patching subset (cellhprompt cls + red) shows this displacement persists when text format is held flat, indicating prompt-family carries causal influence despite its near-zero geometric magnitude — a feature *used* without large feature *encoded* magnitude (\citep{wang2023interpretability} signature). Fourth, the shortcut trigger is **better described as the conjunction of integer-indexed marker and markup-sigil leading delimiter** than as an abstract "flat element list" — AXTree hierarchy preserves the early L04 image-axis peak, but so do indexed variants that lack either the integer (e.g., `hash_id_control`) or the sigil (e.g., `appagent_id`, `plain_numbered`); only the conjunction triggers the late shift. This refinement is **exploratory after W6** and awaits held-out falsifiers (`bare_N`, `bracket_no_int`).
docs/checkpoints/paper_drafts/section5_mechanism.md:17:| Linear readability (held-out AUROC) | Method 4.2 v2 (§5.2, §5.7) | **Strong** — held-out leave-one-task-out AUROC = 1.000 across all 15 mode pairs × all 37 layers on both cls and reddit (Bug 3 fix lototask CV) |
docs/checkpoints/paper_drafts/section5_mechanism.md:18:| Geometric magnitude (cosine gap) | Method 4.2 v2 (§5.2, §5.7) | **Image axis dominates** — image pair peak ~0.04-0.07; text-format + prompt-family axes peak ≤ 0.009 at L36 boundary (no localized peak). Prior "three quantitatively distinct axes" framing retracted; was v1 NPZ Bug 2 artifact |
docs/checkpoints/paper_drafts/section5_mechanism.md:19:| Causal continuation patching (SoM → no-image arms) | Stage 2/3 (§5.4) | **Causal** — mid-layer L12-L18 transfers across cls + reddit, additive across DOM/P-text/P-prompt targets, Gaussian-random negative controls at ~0. **Unchanged by v2 (uses archive directly, not Stage 4 NPZ)** |
docs/checkpoints/paper_drafts/section5_mechanism.md:20:| Causal axis-2 prompt-only patching | Exp 5 cellhprompt cls + red (§5.4) | **Causal continuation evidence, 2 sites, N=24 each; 0.20-0.30 displacement at L11-L17 captures 80-125% of combined image+prompt patching effect**. Task-shuffled content-specificity control (cellhprm_*_tsh Myriad 359768+359769) in flight. Gaussian random control (cellhprm_*_rand 359719+359720) DESTROYS output regardless of axis (codex Bug 6 prediction confirmed; Gaussian is weak baseline) |
docs/checkpoints/paper_drafts/section5_mechanism.md:21:| Steering (mean-diff activation) | Method 4.4 (§5.3) | **Weak / partial** — best H-mean 0.33 at L33 α=10, layer-α tradeoff prevents single sweet spot, treated as evidence ceiling not validation. **Unchanged by v2** |
docs/checkpoints/paper_drafts/section5_mechanism.md:22:| Output divergence | Exp 3 logit lens (§5.7) | **Re-run pending** on v2 NPZ. V1 reported KL 0.16 at L23 axis-2 + KL 0.69 at L23 axis-1; V2 likely revises both. Mechanism direction (lm_head amplifies residual into output KL) probably survives; magnitudes will change |
docs/checkpoints/paper_drafts/section5_mechanism.md:23:| Trigger attribution (which formats trigger shortcut) | W6 tokenization (§5.5) | **Exploratory** — 6 marks-like variants split 2-vs-4 on first-token sigil; held-out falsifier `bare_N` (integer no sigil) and `bracket_no_int` (sigil no integer) pending |
docs/checkpoints/paper_drafts/section5_mechanism.md:25:The cross-site evidence stack is deliberately defensive. Per-task H1 fragility shows the dichotomy is an aggregate mechanism rather than a deterministic per-task law. Reverse-tier classifieds runs defend against strong-tier selection bias. Reddit format variation replicates the shortcut direction with cleaner mid-layer peaks. Reddit Method 4.2 replicates the Mirage signature: Phantom-SoM remains close to DOM on the text axis while separating from SoM on the image axis. Paper 1 uses these results for mechanism interpretation only; routing implementation is deferred to paper 2, consistent with the paper-planning scope split.
docs/checkpoints/paper_drafts/section5_mechanism.md:29:Method 4.2 extracts hidden states from Qwen3-VL-4B B1 runs and compares observation modes by layer. For each mode pair and layer, we compute the cosine gap between hidden-state means, evaluate AUROC by projecting examples onto the mean-difference direction, and summarize per-mode geometry through PCA top-10 variance. The classifieds baseline contains 288 examples, formed from 24 strong-tier tasks, two archived steps, and six modes, over 37 indexed layers.
docs/checkpoints/paper_drafts/section5_mechanism.md:31:The robustness suite passes all five checks in the plan. Label permutation leaves the real AUROC 9.8 standard deviations above the permuted baseline. Per-task analysis is positive for all 24 tasks. Step 2 and step 5 curves are invariant at the mechanism level. The L23 silhouette score is at least 0.5, showing nontrivial clustering. Bootstrap 95% confidence intervals are tight, with widths of roughly 4-15% of the corresponding means.
docs/checkpoints/paper_drafts/section5_mechanism.md:33:The key classifieds snapshot is the L17 cosine-gap table:
docs/checkpoints/paper_drafts/section5_mechanism.md:35:| Pair at L17 | Cosine gap | 95% CI | AUROC |
docs/checkpoints/paper_drafts/section5_mechanism.md:37:| P-SoM <-> P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
docs/checkpoints/paper_drafts/section5_mechanism.md:38:| DOM <-> P-prompt | 0.0013 | [0.0012, 0.0014] | 1.000 |
docs/checkpoints/paper_drafts/section5_mechanism.md:39:| P-SoM <-> SoM | 0.0413 | [0.0403, 0.0422] | 1.000 |
docs/checkpoints/paper_drafts/section5_mechanism.md:40:| DOM <-> Vision | 0.0547 | [0.0531, 0.0563] | 1.000 |
docs/checkpoints/paper_drafts/section5_mechanism.md:42:The reddit replication lands the same qualitative geometry. At L17, P-SoM is close to DOM with cosine gap 0.0098 and close to P-text with gap 0.0027, while P-SoM-to-SoM remains much larger at 0.0423 and P-SoM-to-Vision at 0.0457. The DOM-to-Vision image-axis peak is L04 with cosine gap 0.0687 and AUROC 1.0.
docs/checkpoints/paper_drafts/section5_mechanism.md:44:This is the Mirage signature in geometric form. Phantom-SoM is not represented as a weakened image mode. At the mid-layer disruption locus, it is a text-axis sibling of DOM/P-text, while the image-axis distance to full SoM remains large.
docs/checkpoints/paper_drafts/section5_mechanism.md:48:Method 4.4 tests whether the readable mode direction can be used as a steering direction. For each layer, we form a mean-difference vector between Phantom-SoM-like and DOM-like hidden states, add it to each input at generation time with scaling factor $\alpha$, and evaluate whether the continuation moves toward the target mode while preserving the JSON action envelope. Following HDMI's evaluation vocabulary, reliability is the harmonic mean of completeness and selectivity, not a raw shift rate \citep{khorasani2026hdmi}.
docs/checkpoints/paper_drafts/section5_mechanism.md:50:The v2 sweep covers layers [11, 17, 23, 29, 33, 34] and $\alpha \in [1,2,5,10,20]$, for 45 completed cells in the plan summary. The original L17, $\alpha=5$ smoke result reported H-mean 0.44, but the full sweep lowers that cell to 0.16. The plan records this as a smoke-variance artifact from notes 126/127: a 4-cell smoke was too small to support a sweet-spot claim.
docs/checkpoints/paper_drafts/section5_mechanism.md:52:The strongest full-sweep cell is L33, $\alpha=10$, with H-mean 0.33. Its completeness is 38% and its selectivity is 29%. The layer profile is the important result: mid layers L11-L23 preserve the JSON envelope with 100% selectivity but have low completeness, while late-layer L33 produces the largest shifts but frequently over-steers the continuation out of the expected JSON format.
docs/checkpoints/paper_drafts/section5_mechanism.md:54:This creates a probe-causal dissociation. The mid-layer geometry is cleanly readable and causally implicated by patching, but fixed mean-difference steering is only partially reliable. The 0.33 H-mean is therefore an evidence ceiling for Method 4.4, not a final control result. Section 8 should treat LA-HDMI and SAE feature steering as future work motivated by this ceiling, without claiming that either method has already improved it.
docs/checkpoints/paper_drafts/section5_mechanism.md:58:Activation patching provides the causal test. For each task, the clean/source run and corrupt/target run use the same archived browser step and deterministic 50-token continuation. In the core SoM-to-Phantom-SoM setup, the source prompt is `som`: task instruction, SoM prompt family, flat `[SOM_MARKS]` text, and annotated screenshot. The target prompt is `phantom_som`: the same instruction, same prompt family, and same `[SOM_MARKS]` text, but no image. Source hidden states are cached by layer, injected into the final input-token position of the target on the first forward pass, and subsequent decoding proceeds normally through the model cache.
docs/checkpoints/paper_drafts/section5_mechanism.md:60:Each patched continuation is scored against the unpatched source and target continuations. The main disruption statistic is the drop in `token_overlap_to_target`; Levenshtein distance to target is the paired backup. Layer-wise tests compare each grid layer to the final-layer reference using task-paired differences and Holm-Bonferroni correction across the canonical grid. Random-injection controls replace source hidden states with Gaussian tensors matched to source activation mean and standard deviation.
docs/checkpoints/paper_drafts/section5_mechanism.md:62:The Stage 2 P-SoM<->SoM dashboard now contains ten completed cells, including reddit F/G and the reddit random control:
docs/checkpoints/paper_drafts/section5_mechanism.md:66:| A | cls | SoM->P-SoM forward | -0.32 at L17 | significant |
docs/checkpoints/paper_drafts/section5_mechanism.md:67:| B | cls | P-SoM->SoM reverse | -0.16 at L17 | significant |
docs/checkpoints/paper_drafts/section5_mechanism.md:71:| F | reddit | SoM->P-SoM forward | -0.21 at L17 | significant |
docs/checkpoints/paper_drafts/section5_mechanism.md:72:| G | reddit | P-SoM->SoM reverse | -0.18 at L17 | significant |
docs/checkpoints/paper_drafts/section5_mechanism.md:73:| Cr/Dr | reddit | 2x2 controls | -0.15 to -0.18 | significant |
docs/checkpoints/paper_drafts/section5_mechanism.md:74:| Er | reddit | random injection | approximately 0 uniform | negative control |
docs/checkpoints/paper_drafts/section5_mechanism.md:76:Stage 3 extends this from P-SoM to the three no-image arms, testing whether the image-feature axis is shared across DOM, P-text, and P-prompt targets. The table below reports per-task-paired Δoverlap-to-target from the patching_continuation_results.json under each cell directory, with the layer at which the disruption peaks.
docs/checkpoints/paper_drafts/section5_mechanism.md:78:| Site | SoM->DOM | SoM->P-text | SoM->P-prompt | best-L Δ range |
docs/checkpoints/paper_drafts/section5_mechanism.md:81:| reddit | -0.335 at L11, -0.255 at L17, -0.338 at L14 (best) | -0.244 at L11, -0.236 at L17, -0.330 at L15 (best) | -0.233 at L11, -0.191 at L17, -0.322 at L14 (best) | [-0.322, -0.338] |
docs/checkpoints/paper_drafts/section5_mechanism.md:83:All six Stage 3 cells are now closed. Two observations carry the cross-site claim. First, every cell's best layer falls inside the L12-L18 mid-layer window, and every cell's best Δoverlap-to-target is between -0.27 and -0.35. The mid-layer fusion locus is therefore not a single layer index but a tight 7-layer window that transfers across cls and reddit. Second, the interpretation is additive rather than arm-specific: a SoM source state displaces DOM, P-text, and P-prompt targets toward the source with similar magnitude, implying a shared image-feature substrate across all three no-image arms. The negative controls, Cell E at -0.03 and Cell Er near zero, rule out a generic nonzero-injection explanation.
docs/checkpoints/paper_drafts/section5_mechanism.md:87:The cleanest single-pair signature is the image-axis peak-layer dichotomy. Across eight image-presence contrasts, the no-image side's text format predicts the peak layer with zero overlap. If the no-image side is AXTree text, the image-axis cosine gap peaks at L04 in all four pairs: DOM<->Vision, DOM<->SoM, P-prompt<->Vision, and P-prompt<->SoM. If the no-image side is `[SOM_MARKS]` or another flat marks text, the peak shifts to L17-L36 in all four pairs: P-text<->Vision, P-text<->SoM, P-SoM<->Vision, and P-SoM<->SoM.
docs/checkpoints/paper_drafts/section5_mechanism.md:89:The refined H1 is a pretraining co-occurrence shortcut: when the input contains a marker token sequence that pretraining data associates with HTML / agent-trace visual grounding (specifically the conjunction of integer index and markup-sigil leading delimiter such as `[`, `<`, `@`), the model activates a visual-grounding pathway even if the image is absent. Flat element-list form alone is **not sufficient** — `appagent_id` (`id_N: label`) and `plain_numbered` (`N. label`) are nominally flat indexed lists but lack the markup-sigil and behave like AXTree-DOM (W6 evidence, exploratory). Prompt-format sensitivity makes this plausible at the input level \citep{sclar2024promptformat}; Method 4.2 shows it as a layer-resolved internal signature.
docs/checkpoints/paper_drafts/section5_mechanism.md:91:The format-variation grid contains ten modes: six marks-like variants, two controls, and DOM/SoM baselines. In the classifieds strong-tier baseline, all six marks-like variants peak at L36, the hash-ID control also peaks at L36, the plain-sentence control peaks at L17, and the DOM baseline preserves the L04 peak. Because L36 is the boundary layer, this is best read as a strong late/monotonic signature rather than as a precise late-layer mechanism.
docs/checkpoints/paper_drafts/section5_mechanism.md:93:The classifieds reverse-tier run reproduces the strong-tier shape. The six marks-like variants and hash-ID control again peak at L36, the plain-sentence control moves to L22, and DOM remains at L04. This defends H1 against the selection-bias concern that strong-tier curation alone created the pattern.
docs/checkpoints/paper_drafts/section5_mechanism.md:95:The reddit format run is cleaner for the mid-layer interpretation. Four of six marks-like variants peak at L17, the plain-sentence control peaks at L17, hash-ID control returns to L04, and DOM remains at L04. **W6 attribution** (`docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md`, exploratory) further finds that the two L04 marks-like variants (`appagent_id`, `plain_numbered`) share a feature with the L04 DOM baseline: their first tokens are alphanumeric, while the four L17-peaking marks-like variants all start with markup-sigil tokens (`[`, `<`, `@`). The hash-ID control (`#a3f7`) starts with a sigil but lacks integer-marker structure and also peaks at L04, suggesting the trigger conjunction is integer-marker + markup-sigil rather than either alone. This is a post-hoc feature-attribution on a small (N=6 marks-like) format set; held-out falsifiers (`bare_N` = integer without sigil, `bracket_no_int` = sigil without integer) are not yet run. Cross-site, the safe claim is directional: marker formats that combine integer indexing with markup-sigil leading delimiters tend to delay image-axis separation into mid/late layers, while AXTree hierarchy and indexed-list variants lacking either feature preserve the early L04 image-axis peak. The reddit curve reveals the true L11-L17 fusion locus more clearly than the classifieds L36 boundary artifact.
docs/checkpoints/paper_drafts/section5_mechanism.md:99:The first defense is per-task fragility. On 45 classifieds task-step pairs, only 11% satisfy the strict per-task dichotomy, even though aggregate marks-like peaks are later than AXTree peaks. This prevents over-claiming: H1 is a population-level mechanism signature with task variability, not a deterministic rule for every trajectory.
docs/checkpoints/paper_drafts/section5_mechanism.md:101:The second defense is selection-bias robustness. The classifieds reverse-tier run replicates the strong-tier H1 pattern, including L36 marks-like peaks and L04 DOM baseline. The shortcut signature is therefore not an artifact of selecting tasks where SoM beats DOM.
docs/checkpoints/paper_drafts/section5_mechanism.md:103:The third defense is cross-site H1. Reddit does not reproduce the exact boundary-layer shape, but it reproduces the direction of the indexed-list shortcut with a cleaner L17 mid-layer peak for four of six marks-like formats. The site changes the curve shape, not the basic interpretation.
docs/checkpoints/paper_drafts/section5_mechanism.md:105:The fourth defense is cross-site Mirage geometry. Reddit Method 4.2 reproduces the central relation: P-SoM is close to DOM/P-text at L17 and far from SoM/Vision on the image axis, with AUROC 1.0 on the key contrasts. This supports cross-site generalization of the mechanism claim, not B0/B1 capability scaling.
docs/checkpoints/paper_drafts/section5_mechanism.md:107:Two additional defenses remain deferred rather than folded into the claim: P2 cross-family Phi-3.5-Vision and P3 larger Qwen2-VL-7B. The current evidence is sufficient for the single-model, cross-site Qwen3-VL-4B mechanism section; family and capacity generalization belong in future work or Section 7.
docs/checkpoints/paper_drafts/section5_mechanism.md:111:**REVISION NOTE**: Earlier drafts of this section described a "three-axis hierarchy" with image (≈0.041), text-format (≈0.029), and prompt-family (≈0.011) cosine gaps in a clean 4:3:1 magnitude ratio with distinct peak layers (L17/L23/L23). That description came from Method 4.2 hidden states extracted with a buggy `[SOM_MARKS]` regex that dropped 71/72 marks per task; the v1 Stage 4 NPZ contained near-empty 3-line text payloads, and mode-mean cosine gaps for axis-1 and axis-2 were inflated by prompt-template differences rather than text-payload differences. After the Bug 2 fix re-extraction (Myriad 359736 cls + 359737 reddit, NPZ `hidden_states_v2_fixed.npz`), axis-1 and axis-2 cosine peaks collapse to sub-permille and move from L23 to L36 boundary-monotone. The "three quantitatively distinct axes" claim is no longer supported. The revised account below is paper-grade.
docs/checkpoints/paper_drafts/section5_mechanism.md:113:The pairs are constructed to isolate each axis. Axis-1 (text-format swap, prompt fixed) is measured by DOM↔P-text and P-prompt↔P-SoM. Axis-2 (prompt-family swap, text fixed) is measured by DOM↔P-prompt and P-text↔P-SoM. Image axis is measured by P-SoM↔SoM. All curves are computed on `stage4_multimode_b1_cls/hidden_states_v2_fixed.npz` (144 examples, 37 layers, 6 modes, strong-tier manifest filter, production `[SOM_MARKS]` formatter) and replicated cross-site on the matching reddit run.
docs/checkpoints/paper_drafts/section5_mechanism.md:115:Peak-layer and magnitude table (cls v2, reddit qualitatively identical):
docs/checkpoints/paper_drafts/section5_mechanism.md:117:| Axis | Pair | L17 cosine | L23 cosine | Peak L | Peak gap |
docs/checkpoints/paper_drafts/section5_mechanism.md:119:| Image | P-SoM ↔ SoM | 0.0416 | 0.0410 | L36 | 0.0416 |
docs/checkpoints/paper_drafts/section5_mechanism.md:120:| Axis-1 text-format | DOM ↔ P-text | 0.0021 | 0.0027 | L36 | 0.0047 |
docs/checkpoints/paper_drafts/section5_mechanism.md:121:| Axis-1 text-format | P-prompt ↔ P-SoM | 0.0021 | 0.0026 | L36 | 0.0048 |
docs/checkpoints/paper_drafts/section5_mechanism.md:122:| Axis-2 prompt-family | P-text ↔ P-SoM | 0.0019 | 0.0028 | L36 | 0.0088 |
docs/checkpoints/paper_drafts/section5_mechanism.md:123:| Axis-2 prompt-family | DOM ↔ P-prompt | 0.0013 | 0.0027 | L36 | 0.0068 |
docs/checkpoints/paper_drafts/section5_mechanism.md:127:1. **Image axis is the only well-localized geometric mechanism in the residual stream.** The image pair P-SoM↔SoM peaks at L36 with magnitude 0.042, but the early L04 peak for DOM↔Vision and P-prompt↔Vision (0.067 and 0.066) is the clean image-presence signature: when the no-image side preserves AXTree hierarchy, image-axis divergence is freshly observable at L04. When the no-image side is flat `[SOM_MARKS]`, the early peak attenuates (this is the original Mirage L04 dichotomy, and it survives the v2 re-extraction on the DOM-vs-Vision side; the SoM-side mirror requires re-examination because v1's L17 peak for P-SoM↔SoM shifted to L36 boundary in v2).
docs/checkpoints/paper_drafts/section5_mechanism.md:129:2. **Text-format and prompt-family axes are linearly readable but geometrically near-zero.** All four non-image pairs (DOM↔P-text, P-prompt↔P-SoM, P-text↔P-SoM, DOM↔P-prompt) have peak cosine gap ≤ 0.009 and rise monotonically to a boundary layer L36 rather than localizing at a mid-layer peak. The held-out leave-one-task-out AUROC remains 1.000 across all pairs and layers, which means the 24 strong-tier tasks ARE perfectly separable along these axes — but the mode-mean difference vector is small. The right reading is that text-format and prompt-family modes carry low-magnitude but high-reliability linear signatures in the residual stream rather than substantial geometric clusters.
docs/checkpoints/paper_drafts/section5_mechanism.md:131:The disjoint between **small geometric magnitude (cosine ≤ 0.01)** and **substantial causal patching displacement (overlap-to-target drop of 0.20–0.30 in §5.4 cellhprompt and Stage 2/3 cells)** is the new headline mechanism observation. A causal axis-2 patch at L11–L17 displaces target continuation by ~25% even though the geometric mean-difference at those layers is sub-permille. This argues that residual-stream cosine magnitude **underestimates** the causal influence of a feature, consistent with the standard mechinterp distinction between feature *encoded* and feature *used* \citep{wang2023interpretability}. The activation-patching evidence (§5.4) is the load-bearing claim; cosine geometry is supporting evidence about readability, not magnitude.
docs/checkpoints/paper_drafts/section5_mechanism.md:133:Phantom-SoM's drop-one hero contribution in `fig_meta_forest.png` (reddit drop-one CI [+0.95, +6.19] strict-positive) therefore cannot be attributed to "three-axis positional uniqueness" with quantitatively distinct magnitudes. The cleaner mechanism story is: Phantom-SoM is one of four modes occupying the no-image-flat-marks half of the phantom routing space, all of which produce small geometric separation from each other; the behaviorally distinct success-task pool (Jaccard 0.29–0.49 against other arms) is what gives drop-one its complementarity, and patching displacement at L11–L17 shows the difference matters causally for token continuation. The bridge from patching displacement to behavioral SR remains open.
docs/checkpoints/paper_drafts/section5_mechanism.md:135:A held-out leave-one-task-out AUROC of 1.000 across small-magnitude pairs is itself a paper-grade observation. It says the modes are reliably linearly separable at any chosen layer with very small mean-difference vectors, which is a stronger claim about the residual stream than the original "distinct mid-layer peaks" framing. The information capacity of the residual stream to represent observation-mode identity is high; the *magnitude* of the representation is mostly image-driven. This reframing follows the linear-readability framework of \citep{wu2026toolcalling} without the cosine-magnitude overclaim.
docs/checkpoints/paper_drafts/section5_mechanism.md:137:The output-amplification observation (logit lens, Exp 3) needs re-running on the v2 NPZ before its quantitative claims can be reported. The v1 logit lens reported peak KL 0.162 at L23 for the axis-2 pair P-text vs P-SoM, but the v1 input hidden states were the buggy 3-line-text version. The qualitative direction (lm_head amplifies residual-stream geometry into output KL) likely survives, but the absolute KL magnitudes will change; we report the v2 lm_head amplification numbers in a follow-up release.
docs/checkpoints/paper_drafts/section5_mechanism.md:139:Two corollaries follow. First, the KL trajectory drops to approximately zero at L36 even though L23 KL is substantial. The mean hidden state at the final layer collapses to the shared JSON action-header tokens that every mode emits, so mode-distinct output signal is concentrated in the L23-L25 decoding window rather than at the final embedding. Second, this output-amplification observation is **mechanistic, not a deployment-time classifier claim**: the lm_head acts as an axis-agnostic ratio-preserving projection that scales residual-stream geometry into output-space KL — the L23-L25 KL magnitude is a property of the mean hidden state, not a per-task discriminator. Whether the L23-L25 hidden representation can be used as a held-out mode classifier — with per-task AUROC, random-direction baseline, and competitive comparison to surface-token classifiers — is open work. Routing exploitation, deferred to paper 2, will need to make this case explicitly rather than inheriting it from §5.7.
docs/checkpoints/paper_drafts/section5_mechanism.md:143:The main limit is the Method 4.4 ceiling. The cosine-gap and patching evidence point to L11-L17 as the readable and causally active fusion region, while the best fixed mean-difference steering cell is late, L33 with $\alpha=10$, and has H-mean 0.33 because completeness and selectivity trade off. This supports a mechanism interpretation but not a strong deployment-time steering claim.
docs/checkpoints/paper_drafts/section5_mechanism.md:145:The second limit is layer precision. Classifieds H1 peaks often hit L36, while reddit reveals cleaner L17 peaks. The robust claim is therefore an effect-direction claim: AXTree hierarchy preserves early image-axis separation, and flat element-list formats delay that separation into mid/late computation. We should not claim that every site or task has an identical peak layer.
docs/checkpoints/paper_drafts/section5_mechanism.md:147:Literature positioning should stay modest. Section 5 applies the linear-readable, steerable, and mid/late-layer circuit framework to multimodal web-agent observation modes \citep{wu2026toolcalling,kaduri2024whatsintheimage,khorasani2026hdmi,fayyaz2026steermoe}. It should not claim novelty as the first such circuit or the first use of marked text. The contribution is controlled scientific characterization of the phantom boundary.
docs/checkpoints/paper_drafts/section5_mechanism.md:149:Finally, AXTree hierarchy is the unique defeating format in the aggregate, but the reason hierarchy defeats the shortcut remains open. The plan records one attribution-pending hypothesis: hierarchy or indentation tokens may redirect cross-modal attention before the flat-list shortcut activates. That should be treated as a supplement question, not as a Section 5 finding.
docs/checkpoints/paper_drafts/section5_mechanism.md:153:Bibkeys audit (2026-05-12 21:18): all 5 core mechanism anchors verified present in `paper.bib` — `wu2026toolcalling`, `khorasani2026hdmi`, `kaduri2024whatsintheimage`, `sclar2024promptformat`, `fayyaz2026steermoe`. Plus 5 method/protocol references added: `wang2023interpretability` (IOI patching), `zhang2024patching` (patching survey, NEEDS_VERIFY exact paper), `holm1979sequentially` (multiple-comparison correction), `lipton2018troubling` (ML scholarship critique), `neurips2024checklist` (reproducibility standard). paper.bib total 67 entries / 638 lines.
docs/checkpoints/paper_drafts/section5_mechanism.md:155:Behavioral content to relocate from current `section5_mechanism_reddit.md`: lines 17-75 should move to Section 4 or a new behavioral-routing subsection. Specifically, lines 17-23 are reddit substrate framing; lines 25-35 are Axis 1 text-payload behavior; lines 37-47 are Axis 2 prompt behavior; lines 49-59 are Axis 3 image behavior; lines 61-67 are compound P-SoM versus DOM behavior; lines 69-75 are scope/noise limitations. Lines 1-15 are method material that was retained conceptually but must use the new L0-L36 layer convention. Line 77 should be deleted or replaced because routing implementation is now paper-2, not paper-1 Section 6.
docs/checkpoints/paper_drafts/section5_mechanism.md:157:Stage 3 numbers verified 2026-05-12 from full per-task paired-test computation on `patching_continuation_results.json` (each cell, 24 tasks × 36 layers). H-d-cls best L18 Δ=-0.352, H-d-red best L14 Δ=-0.338, H-t-cls best L12 -0.270, H-t-red best L15 -0.330, H-p-cls best L13 -0.273, H-p-red best L14 -0.322. All 6 cells' best layer lands in L12-L18 mid-layer window, Δ range [-0.27, -0.35]. The L17-only column previously cited in plan §5.2 reads -0.309/-0.255/-0.223 (cls) and -0.255/-0.236/-0.191 (reddit); plan §5.2 has been updated to record best-layer Δ instead of L17-only Δ.
docs/checkpoints/paper_drafts/section5_mechanism.md:159:Pending items (post 2026-05-12 audit): (a) Method 4.4 sweep description should be "45 completed cells out of a 6x5 layer-alpha grid plus 3 placeholder cells that did not finish", not "45/48-cell sweep" (the 48-cell wording in plan §5.3 implies a 48-cell denominator that was never executed). (b) Bibkey `zhang2024patching` is marked NEEDS_VERIFY in `paper.bib` because the intended reference may be Heimersheim & Nanda 2024 [arXiv:2404.15255] rather than Zhang & Nanda 2024 [arXiv:2309.16042]; verify before submission. (c) Bibkey `fayyaz2026steermoe` is marked NEEDS_VERIFY pending deanon of the ICLR 2026 submission.
docs/checkpoints/paper_drafts/section5_mechanism.md:163:Codex independent audit (`docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md`) surfaced 6 weak claims + 5 honest gaps that Claude /stress had missed. 3 fixed inline tonight in §5:
docs/checkpoints/paper_drafts/section5_mechanism.md:166:2. ✅ §5.7 hero paragraph — "proximity to SoM on the image axis... as if image were present" → corrected to "large image-axis SEPARATION from SoM... no-image marks-text reshapes how image-axis divergence accumulates" (removed internal contradiction with §5.2 table where P-SoM↔SoM gap 0.0412 is the largest = a separation, not proximity)
docs/checkpoints/paper_drafts/section5_mechanism.md:167:3. ✅ §5.7 corollary 2 — "deployment-time mode classifier on output logprobs has strictly more signal" + "Section 6 routing should treat L23-L25 logit-lens features as the cheapest mode-axis discriminator" → softened to "mechanistic observation, not deployment-time classifier claim; held-out classifier with random-direction baseline is open work"
docs/checkpoints/paper_drafts/section5_mechanism.md:168:4. ✅ Evidence status table added at end of §5.1 — geometry strong / patching causal-continuation / Exp 5 axis-2 CI pending / steering weak / output divergence not classifier / W6 trigger exploratory
docs/checkpoints/paper_drafts/section5_mechanism.md:172:- **§4 P-text adjusted SR inconsistency**: §4 table line 37 says 11.90, prose line 106 says 12.38, hero_claim_bootstrap_ci.md says 12.38. Need to canonicalize one number from episode-level adjusted-success and update every occurrence. (1h)
docs/checkpoints/paper_drafts/section5_mechanism.md:173:- **plan.md:125-135 stale "L17 planning site"**: plan still asserts L17 singular planning site, while new evidence shows cosine peak L23 + patching peak L11-L17 + steering best L33; replace with "patch-sensitive continuation window L11-L17 under final-token replacement patching". (1-2h)
docs/checkpoints/paper_drafts/section5_mechanism.md:175:- **Exp 5 cellhprompt bootstrap CI + content-matched control**: Gaussian random injection control 359719/359720 in flight; codex notes Gaussian alone is weak — also need task-shuffled (source from different task) and per-task bootstrap CIs.
docs/checkpoints/paper_drafts/section5_mechanism.md:176:- **Behavioral causal bridge gap**: patching displaces 50-token continuation, not SR / drop-one. No experiment currently bridges mid-layer patching effect to task-success outcome. Open work.
docs/checkpoints/paper_drafts/section5_mechanism.md:177:- **Cross-family**: P2 Phi-3.5-Vision + P3 Qwen2-VL-7B (task #40, #41). At minimum needed for §6 generalization narrative.
docs/checkpoints/paper_drafts/section5_mechanism.md:179:**Codex verdict**: 0.10-0.20 NeurIPS/ICML/ACL main accept probability; would reject @ reviewer-3 4/10. **Workshop / borderline mid-tier conference today**. Codex agrees the drop-one oracle CI + cross-site asymmetry + axis-2 cosine signal breadth are strong; §5 mechanism prose was over-claiming relative to evidence.
docs/checkpoints/mechanism/results/format_variation_h1_test.md:1:# Stage 4 H1 test: indexed-list format variation
docs/checkpoints/mechanism/results/format_variation_h1_test.md:6:**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
docs/checkpoints/mechanism/results/format_variation_h1_test.md:7:- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
docs/checkpoints/mechanism/results/format_variation_h1_test.md:12:| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:14:| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0462 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:16:| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0434 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:28:- `[N] role 'label' (SoM)`: peak **L36** = 0.0434
docs/checkpoints/mechanism/results/format_variation_h1_test.md:45:- `AXTree (baseline DOM)`: peak **L04** = 0.0462
docs/checkpoints/mechanism/results/format_variation_h1_test.md:51:- **AXTree-DOM baseline**: peak L04
docs/checkpoints/mechanism/plan.md:2:name: mechanism plan
docs/checkpoints/mechanism/plan.md:3:description: Full mechanism workspace — theory, lit anchor stack, methods, identification protocol, current findings, open questions, advisor sync, roadmap. Specialized companion to paper_planning §2; not a duplicate.
docs/checkpoints/mechanism/plan.md:14:| Zoom | Level | What our paper claims |
docs/checkpoints/mechanism/plan.md:16:| **1** | Architectural | Phantom routing space = "skip annotated image" boundary contains 3 arms (P-text / P-prompt / P-SoM) sharing 4-fold drop-in property |
docs/checkpoints/mechanism/plan.md:17:| **2** | Behavioral (axis effects) | Axis 1 (text payload: AXTree vs [SOM_MARKS]) is PRIMARY; Axis 2 (prompt: SoM-prompt vs DOM-prompt) is secondary; Axis 3 (image presence: in vs out) is gating |
docs/checkpoints/mechanism/plan.md:18:| **3** | Named phenomena (lit-anchored) | Mirage Effect (Asadi 2026) / Scaffold Effect (Vu&Balloccu 2026) / Cross-modal flow (Kaduri) / Prompt-format sensitivity (Sclar 2024) |
docs/checkpoints/mechanism/plan.md:19:| **4** | Model-internal | L17 mid-layer is BOTH discrimination locus (probe AUROC 1.0) AND causally active planning site (Stage 2/3 patching + Method 4.4 v2 reliability) |
docs/checkpoints/mechanism/plan.md:21:### 1.2 Three-axis hierarchy quantified (Method 4.2 PCA cosine gap, Qwen3-VL-4B B1 cls)
docs/checkpoints/mechanism/plan.md:23:| Axis | Peak cosine gap | Peak layer | Magnitude ratio |
docs/checkpoints/mechanism/plan.md:25:| Image-axis (vs SoM / Vision) | 0.06 | L4–L17 | **10×** |
docs/checkpoints/mechanism/plan.md:27:| Prompt-axis (SoM-prompt vs DOM-prompt alone) | 0.007 | L36 | **1×** |
docs/checkpoints/mechanism/plan.md:29:→ Mechanism magnitude image >> text > prompt. Validates `project_phantom_space_axes_format_not_information.md` memory: P-SoM closest mode at every layer is **P-text** (text-axis sibling, L17 cosine 0.0028 vs P-SoM↔SoM 0.0412 = 14.7× more distant).
docs/checkpoints/mechanism/plan.md:31:### 1.3 Image-axis peak-layer dichotomy (Mirage mechanism signature)
docs/checkpoints/mechanism/plan.md:33:Method 4.2 reveals image-axis cosine-gap peak shifts based on text format of the no-image side. Clean dichotomy, zero overlap across 8 image-axis pairs:
docs/checkpoints/mechanism/plan.md:37:| AXTree (hierarchical) | **L04** | DOM↔Vision, DOM↔SoM, P-prompt↔Vision, P-prompt↔SoM |
docs/checkpoints/mechanism/plan.md:38:| [SOM_MARKS] / flat | **L17–L36** | P-text↔Vision, P-text↔SoM, P-SoM↔Vision, P-SoM↔SoM |
docs/checkpoints/mechanism/plan.md:42:Format variation extraction (Myriad job 352998, `stage4_format_variation_b1_cls`, 450 hidden states = 45 task-step × 10 modes). For each text format V, compute image-axis cosine gap V↔som per layer; peak layer reveals shortcut activation:
docs/checkpoints/mechanism/plan.md:46:| **AXTree hierarchical (DOM)** | **L04** | **SOLE format defeating shortcut** |
docs/checkpoints/mechanism/plan.md:48:| `[N] role 'label'` (SoM standard) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:56:**Refined H1 verdict**: trigger is **flat element listing**, not "indexed list pattern". Even integer-free hash IDs and pure-sentence variants engage the shortcut. AXTree hierarchical depth is the **unique format** that defeats shortcut activation.
docs/checkpoints/mechanism/plan.md:58:Paper §5 implication: SoM-family web agents (Browser Use, AppAgent, Tarsier, OmniParser, etc.) **all** implicitly exploit the same flat-list-element-grounding shortcut from VLM training distribution. P79 phantom routing space makes this systematic and routes accordingly.
docs/checkpoints/mechanism/plan.md:64:| **Wu et al. 2026** (UCL lab, our advisors) | Method backbone | `wu2026toolcalling` (2605.07990) | Mean-difference activation steering at second-to-last layer, 77–100% switch on tool selection (93–100% at 4B+). Our Method 4.2/4.4 port to multimodal Qwen3-VL-4B web agent |
docs/checkpoints/mechanism/plan.md:65:| **Ma & Rui 2026** | Probe-vs-causal vocabulary | `maRui2026planning` (2605.07984) | "Planning-compatible representation" vs "causally active planning site". Qwen3-family pattern: probe works, causal patching weak (1% rhyme newline causal vs Gemma 67%). Our Method 4.4 v2 50% reliability is consistent with this family pattern |
docs/checkpoints/mechanism/plan.md:66:| **HDMI / Khorasani et al. 2026** | Alt method + evaluation metric | `khorasani2026hdmi` (2605.07631) | Probe-free gradient-based steering. Critically: **completeness × selectivity → harmonic mean reliability** — what our Method 4.4 v2 reports (not raw shift rate) |
docs/checkpoints/mechanism/plan.md:67:| **Lin & Liu 2026 Position paper** | Methodology protocol | `linLiu2026disclosure` (2605.08012) | 5-step identification disclosure norm: state claim / name strategy / enumerate assumptions / stress-test / separate validation. Paper §5 adopts as identification subsection structure |
docs/checkpoints/mechanism/plan.md:74:`scripts/mechanistic/curate_mirage_tasks.py` builds a strong-tier task manifest from a B1 production run:
docs/checkpoints/mechanism/plan.md:76:- Pick 24 cls + 24 reddit "strong-tier" tasks (clean Mirage); separate 15 cls reverse-tier (DOM > SoM, selection-bias defense)
docs/checkpoints/mechanism/plan.md:77:- Archive raw observations (AXTree, screenshots, SoM-marks) per (task, step) → `archive_subset_b1_<site>/<site>_task_<tid>/step_<step>/` for offline patching/extraction (Stages 2-4 read from this archive, no live env required)
docs/checkpoints/mechanism/plan.md:80:- `results/mechanistic/curate_mirage_b1_classifieds/manifest.json` — cls strong/reverse tier task list
docs/checkpoints/mechanism/plan.md:81:- `results/mechanistic/curate_mirage_b1_reddit/manifest.json` — reddit strong tier
docs/checkpoints/mechanism/plan.md:82:- `results/mechanistic/archive_subset_b1_cls/` (17 MB, 144 files, 24 tasks × 6 steps)
docs/checkpoints/mechanism/plan.md:83:- `results/mechanistic/archive_subset_b1_reddit/` (35 MB, 356 files, 24 tasks × ~15 steps)
docs/checkpoints/mechanism/plan.md:85:### 3.1 Method 4.2 — PCA cosine gap (DONE)
docs/checkpoints/mechanism/plan.md:87:`scripts/analysis/stage4_pca_cosine_gap.py` + `stage4_robustness.py`. Three metrics per (mode_pair, layer):
docs/checkpoints/mechanism/plan.md:89:- B. AUROC via (mean_A − mean_B) projection
docs/checkpoints/mechanism/plan.md:97:- Test E bootstrap 95% CI tight (4-15% of mean)
docs/checkpoints/mechanism/plan.md:99:### 3.2 Method 4.4 — mean-diff activation steering (v2 in flight)
docs/checkpoints/mechanism/plan.md:101:`scripts/mechanistic/run_stage4_method44_v2_sweep.py`. Layer × α sweep:
docs/checkpoints/mechanism/plan.md:116:- **LA-HDMI**: probe-free gradient steering (Khorasani 2026 method). Per-input optimization replaces fixed mean-diff direction. May overcome Qwen3-family causal patching weakness
docs/checkpoints/mechanism/plan.md:117:- **SAE feature steering** (Zekun-recommended in advisor recording, paper_planning §108): train SAE on Qwen3-VL-4B residual stream (1-2 week cost, no public SAE exists), find mirage/format feature, steer directly. Differentiates from Wu et al. mean-diff path
docs/checkpoints/mechanism/plan.md:125:### 4.1 Causal claim (revised after /codex-stress methodology audit 2026-05-12)
docs/checkpoints/mechanism/plan.md:127:> The patch-sensitive continuation window L11-L17 (block-output index convention) at the last-input-token position is causally consequential for phantom routing space mode selection in Qwen3-VL-4B web agents, under final-token-replacement activation patching. Separately, the prompt-family axis (P-text ↔ P-SoM) signature is most readable in cosine geometry at the LATER layer L23 (signature layer ≠ decision layer; mechanistic-interpretability standard finding cf. Wang et al. 2023 IOI).
docs/checkpoints/mechanism/plan.md:129:The previous "L17 singular planning site" framing is **stale** and was inaccurate: (a) cosine peak for prompt-family axis is L23 not L17 (Exp 1 three-axis hierarchy, 2026-05-12); (b) patching causal peak is the L11-L17 *window*, not a single layer; (c) Method 4.4 steering full sweep (45 cells) lowered the L17 α=5 H-mean from the smoke result 0.44 to 0.16, and the highest cell is now L33 α=10 H-mean 0.33 with poor selectivity (not a single sweet spot at L17). Treat L17 as one peak within the L11-L17 window, not THE site.
docs/checkpoints/mechanism/plan.md:133:Triangulation of 3 evidence types:
docs/checkpoints/mechanism/plan.md:134:1. **Probe-level** (Method 4.2 PCA cosine gap; AUROC reported as both `in_sample` and held-out `leave-one-task-out` after 2026-05-12 Bug 3 fix; v1 buggy NPZ data invalidated, v2 NPZ in flight: Myriad 359736 cls + 359737 reddit)
docs/checkpoints/mechanism/plan.md:135:2. **Replacement patching** (Stage 2/3 Cell A-H, L11-L17 window disruption, Holm-significant per layer; baseline empirically equals unpatched at L35 final-block patching position since overlap→target ≈ 1.00 at L35 across all forward cells)
docs/checkpoints/mechanism/plan.md:136:3. **Additive steering** (Method 4.4 v2 full sweep 45 cells: layer-α tradeoff; mid-layer L11-L17 preserves JSON envelope but low completeness, late-layer L33 produces largest output shifts but over-steers — H-mean ceiling 0.33 indicates probe-causal dissociation, not a single sweet-spot validation)
docs/checkpoints/mechanism/plan.md:145:| A4 | Qwen3-VL-4B mechanism transfers to other VLM sizes/architectures | Not tested. Wu et al. shows family generality on tool-only; multimodal+multi-step unknown |
docs/checkpoints/mechanism/plan.md:146:| A5 | Replacement patching faithfully simulates "natural" model read of the representation | Cell E random-injection control rules out non-specific disruption — content-specific causation confirmed |
docs/checkpoints/mechanism/plan.md:150:Cell E random-injection control: replacing source hidden with Gaussian noise (same μ, σ) yields **null L17 disruption effect**. Confirms our patching effect is source-content-specific, not noise-driven. Most directly stresses A5.
docs/checkpoints/mechanism/plan.md:154:- Method 4.2 AUROC 1.000 = validation (decodability)
docs/checkpoints/mechanism/plan.md:160:### 5.1 Stage 4 Method 4.2 (Qwen3-VL-4B B1 cls, 288 examples × 37 layers)
docs/checkpoints/mechanism/plan.md:162:| Pair @L17 | Cosine gap | 95% CI | AUROC |
docs/checkpoints/mechanism/plan.md:164:| P-SoM ↔ P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
docs/checkpoints/mechanism/plan.md:165:| DOM ↔ P-prompt | 0.0013 | [0.0012, 0.0014] | 1.000 |
docs/checkpoints/mechanism/plan.md:166:| P-SoM ↔ SoM | 0.0413 | [0.0403, 0.0422] | 1.000 |
docs/checkpoints/mechanism/plan.md:167:| DOM ↔ Vision | 0.0547 | [0.0531, 0.0563] | 1.000 |
docs/checkpoints/mechanism/plan.md:169:### 5.2 Stage 2/3 patching disruption (14 cells, B1 cls + reddit)
docs/checkpoints/mechanism/plan.md:171:**Stage 2 — P-SoM ↔ SoM patching (10 cells):**
docs/checkpoints/mechanism/plan.md:175:| A | cls | SoM→P-SoM forward | -0.32 | ✓ |
docs/checkpoints/mechanism/plan.md:176:| B | cls | P-SoM→SoM reverse | -0.16 | ✓ |
docs/checkpoints/mechanism/plan.md:180:| F | reddit | SoM→P-SoM forward | -0.21 | ✓ |
docs/checkpoints/mechanism/plan.md:181:| G | reddit | P-SoM→SoM reverse | -0.18 | ✓ |
docs/checkpoints/mechanism/plan.md:182:| Cr/Dr | reddit 2x2 | both directions | -0.15 to -0.18 | ✓ |
docs/checkpoints/mechanism/plan.md:183:| Er | reddit | random injection | ~0 (uniform) | ✓ |
docs/checkpoints/mechanism/plan.md:185:**Stage 3 — 2x2 mechanism additivity test (SoM → {DOM, P-text, P-prompt}, cls + reddit):**
docs/checkpoints/mechanism/plan.md:189:| H-d-cls | cls | SoM → DOM | L10 (0.192) | -0.33 | `stage3_cellhd_cls_fwd_dom_myriad/` |
docs/checkpoints/mechanism/plan.md:190:| H-p-cls | cls | SoM → P-prompt | L27 (0.219) | -0.22 | `stage3_cellhp_cls_fwd_prompt_myriad/` |
docs/checkpoints/mechanism/plan.md:191:| H-t-cls | cls | SoM → P-text | L28 (0.164) | -0.25 | `stage3_cellht_cls_fwd_text_myriad/` |
docs/checkpoints/mechanism/plan.md:192:| H-p-red | reddit | SoM → P-prompt | L20 (0.209) | -0.19 | `stage3_cellhp_red_fwd_prompt_myriad/` |
docs/checkpoints/mechanism/plan.md:193:| H-t-red | reddit | SoM → P-text | L01 (0.194) | -0.24 | `stage3_cellht_red_fwd_text_myriad/` |
docs/checkpoints/mechanism/plan.md:194:| **H-d-red** | reddit | SoM → DOM | L28 (0.204) | **L11 -0.33 / L17 -0.26** | `stage3_cellhd_red_fwd_dom_myriad/` ✅ done 2026-05-12 19:57 |
docs/checkpoints/mechanism/plan.md:196:**Stage 3 interpretation (6/6 cells complete 2026-05-12)**: All forward SoM→{no-image-arm} patching cells show mid-layer L11-L17 disruption -0.19 to -0.33 Δoverlap→tgt. Magnitude > random injection control (Cell E -0.03) at all 6. **Mechanism additivity confirmed**: image-feature axis is shared substrate across DOM / P-text / P-prompt arms — single SoM→{any-no-image-arm} patching displaces target prediction toward source. Cross-site cls + reddit both replicate (paper §5 universal mid-layer fusion locus); reddit fusion locus slightly earlier (L11 vs cls L17), magnitude identical.
docs/checkpoints/mechanism/plan.md:198:Stage 3 cross-site DOM-axis additivity table (paired-test Δoverlap-to-target from `patching_continuation_results.json`):
docs/checkpoints/mechanism/plan.md:200:| Site | SoM→DOM | SoM→P-text | SoM→P-prompt | best-L Δ range |
docs/checkpoints/mechanism/plan.md:203:| reddit | H-d-red L11 -0.335 / L17 -0.255 / L14 **-0.338** best | H-t-red L11 -0.244 / L17 -0.236 / L15 **-0.330** best | H-p-red L11 -0.233 / L17 -0.191 / L14 **-0.322** best | [-0.322, -0.338] |
docs/checkpoints/mechanism/plan.md:209:H-mean reliability (HDMI framework) per (layer, α). **L17 α=5 smoke claim REFUTED by full sweep**; actual sweet spot at L33 α=10:
docs/checkpoints/mechanism/plan.md:222:- Late-layer (L33): completeness 38% (highest), but selectivity drops to 29% (over-steers JSON)
docs/checkpoints/mechanism/plan.md:225:**Smoke variance lesson** (笔记 §126 + §127): 4-cell smoke H-mean 0.44 on L17 was statistical artifact (1/4 hit = inflated rate). Full 45-cell H-mean 0.16 is true rate. Future mechanism findings require n ≥ 30 cells before "sweet spot" claims.
docs/checkpoints/mechanism/plan.md:229:`docs/checkpoints/mechanism/results/layer_axis_emergence.md`. AXTree-no-image side → L04 peak (4/4); [SOM_MARKS]-no-image side → L17–L36 peak (4/4). Zero overlap. Mirage Effect mechanism signature.
docs/checkpoints/mechanism/plan.md:231:### 5.5 H1 test: flat-list format variation (Method 4.2 extension, 2026-05-12)
docs/checkpoints/mechanism/plan.md:233:`docs/checkpoints/mechanism/results/format_variation_h1_test.md`. 8 industry-relevant text formats + 2 controls. AXTree hierarchical (DOM) is **unique format** preserving L04 image-axis peak; all 8 flat-list variants (SoM standard, Browser Use @, AppAgent id_, Tarsier typed, plain numbered, XML tagged, hash-ID control, plain-sentence control) shift peak to L17–L36. Trigger is flat element listing, not specific token pattern.
docs/checkpoints/mechanism/plan.md:240:| ✅ H1 test: do all flat-list formats trigger shortcut? | **Closed 2026-05-12 00:00**: YES, including hash-ID + plain-sentence controls. AXTree-DOM is sole defeating format | — |
docs/checkpoints/mechanism/plan.md:241:| Reverse-tier 15 tasks vs strong-tier 24 — does L33 + H1 finding generalize beyond selection bias? | Med-High | qsub Stage 4 multimode + format variation with --tier reverse |
docs/checkpoints/mechanism/plan.md:242:| ✅ Cross-site Method 4.2 — does cls finding replicate on reddit? | **Closed 2026-05-12 16:30**: P-SoM↔DOM L17=0.0098 + P-SoM↔SoM L17=0.0423, AUROC 1.0 → Mirage signature replicated. See §7.3.1 | — |
docs/checkpoints/mechanism/plan.md:243:| ✅ Stage 3 reddit 2x2 closure — H-d-red | **Closed 2026-05-12 19:57** (Myriad 358831). L11 Δ=-0.33 / L17 Δ=-0.26. Cross-site additivity confirmed — see §5.2 Stage 3 table | — |
docs/checkpoints/mechanism/plan.md:244:| LA-HDMI vs mean-diff — does gradient steering beat 0.33 ceiling? | Med | Pending Zekun reply + attribution decision |
docs/checkpoints/mechanism/plan.md:245:| SAE feature steering feasibility — is 1-2 week self-training Qwen3-VL-4B SAE worth it? | Low-Med | Depends on Zekun reply + paper §8 prose direction |
docs/checkpoints/mechanism/plan.md:246:| B0 (proxy API) — paper §5 Qwen-specific or generalizable? | Low | Cannot test on B0; cite Wu et al. cross-family generality as proxy |
docs/checkpoints/mechanism/plan.md:247:| AXTree-defeats-shortcut mechanism — *why* hierarchy beats flat? Cross-modal attention specific to indentation tokens? | High (paper §5 supplement) | Activation patching at L4 with hierarchical-text vs flat-text → see which attention heads pre-disrupt image embedding |
docs/checkpoints/mechanism/plan.md:253:- 2026-04-09 笔记 §19: I first grok the paper (then "Anonymous 2026 ACL"), record cosine gap method + L23+ steering 80-93%
docs/checkpoints/mechanism/plan.md:256:- 2026-05-09 advisor recording: Zekun explicitly recommended "SAE feature steering — 前所未有 inference time steering, 单独发 paper" — directed me to differentiating path
docs/checkpoints/mechanism/plan.md:259:**Net**: Zekun explicitly invited mechanism extension. Method 4.4 multimodal port is on his recommendation; SAE Method 4.5 is his next-step suggestion.
docs/checkpoints/mechanism/plan.md:266:- ✓ Added: H1 test finding — flat-list format universally triggers shortcut (8/8 variants), only AXTree hierarchical defeats; implication for industry SoM-family agents
docs/checkpoints/mechanism/plan.md:271:> Zekun 早, 你那篇 Tool Calling 上 arxiv 我看了, 恭喜! 我前几天按你说的开始 mechanism work, 跑出来一些东西想跟你 sync 一下, 顺便问几个方向问题。
docs/checkpoints/mechanism/plan.md:274:> P79 paper 在做 VisualWebArena 的 phantom routing space — agent 6 种 obs mode (DOM 文本/SoM 标注图/Vision 裸图 + 3 个 phantom 变体). 模型 Qwen3-VL-4B, 你 Qwen 3 4B 同 base LM。
docs/checkpoints/mechanism/plan.md:276:> # 1. Method 4.2 PCA cosine gap port 到 6 modes
docs/checkpoints/mechanism/plan.md:277:> 24 cls strong-tier × 2 step × 6 mode = 288 hidden states, 37 layer × 2560 dim。全 540 pair × layer AUROC = 1.000 (perm baseline 0.629, real 9.8σ above). 你方法在 multimodal Qwen 上 readable transfer 干净。
docs/checkpoints/mechanism/plan.md:279:> # 2. Method 4.4 mean-diff steering (HDMI metric)
docs/checkpoints/mechanism/plan.md:283:>   - Mid-layer (L11-L23) selectivity 100% 但 completeness 0-11% — readable but not effectively steerable
docs/checkpoints/mechanism/plan.md:284:>   - 你 paper Qwen 3 4B 93% switch vs 我 38% — 我猜原因是 multi-step JSON gen 的 selectivity 是真约束 (你 single-token tool decision selectivity 自动 1.0)
docs/checkpoints/mechanism/plan.md:286:> # 3. H1 test: flat-list format variation (Myriad)
docs/checkpoints/mechanism/plan.md:287:> 测了 8 个 industry-relevant text format (Browser Use @, AppAgent id_, Tarsier typed, numbered, XML, hash-ID, plain-sentence + SoM baseline) vs AXTree-DOM:
docs/checkpoints/mechanism/plan.md:290:>   - **AXTree hierarchical 是唯一保留 L04 peak 的 format**
docs/checkpoints/mechanism/plan.md:292:>   - = SoM-family agents 全 implicit exploit 同一 VLM shortcut, AXTree 是 sole exception
docs/checkpoints/mechanism/plan.md:295:> (1) Attribution: paper §5 mechanism 这块 — cite 你 + 我独立 framing 比较合理, 还是 co-author 一篇 multimodal extension 比较好? 都 OK, 想听你意见。
docs/checkpoints/mechanism/plan.md:299:> (3) 你之前 advisor 录音里建议 SAE feature steering, 我也写进 future work 了。现在 mean-diff ceiling ~0.33, 是不是 SAE 这条路更有差异化? Qwen3-VL-4B SAE 没公开, 自训成本 1-2 周, 你觉得值得 commit GPU 吗?
docs/checkpoints/mechanism/plan.md:305:After per-task fragility revealed 11% strict dichotomy (aggregate statistical, not deterministic), launched 5-priority defense matrix to triangulate H1 across **(tier × site × family/size)**:
docs/checkpoints/mechanism/plan.md:309:| **P1** | Per-task fragility audit (24 cls strong) | DGX | ✅ done | `results/h1_per_task_fragility.md` |
docs/checkpoints/mechanism/plan.md:311:| **P3** | Within-family bigger (Qwen2-VL-7B, H1' capacity test) | DGX | ❌ deferred (HF cas-bridge throttling) | `stage4_h1_qwen2vl7b_cls/pilot_summary.md` |
docs/checkpoints/mechanism/plan.md:312:| **P4** | cls reverse-tier (selection-bias defense) | Myriad 353763 | ✅ **done 18:50:46** — shape (260, 37, 2560), 10 modes, 46 MB pulled. Same pattern as cls strong-tier (L36 marks-like + L04 dom). Selection-bias defended | `stage4_format_variation_b1_cls_reverse/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:313:| **P5a** | reddit format variation (cross-site H1) | Myriad **354382** (3rd attempt) | ✅ **done 08:09:38** — shape (430, 37, 2560), 10 modes, 76 MB pulled | `stage4_format_variation_b1_reddit/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:314:| **P5b** | reddit Method 4.2 multimode (cross-site Mirage) | Myriad 353890 | ✅ **done 07:31:14** — 288 examples, 6 modes, 51 MB pulled | `stage4_multimode_b1_reddit/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:317:1. Myriad 353764 (00:48) — `no hidden states extracted` after 105 task skips. Root cause: hardcoded `classifieds_task_{tid}` prefix in `run_stage4_format_variation_extract.py:177`, archive uses `reddit_task_*`
docs/checkpoints/mechanism/plan.md:319:3. Myriad **354382** (07:26) — fixed via commit 3d41953 (add `--site reddit` arg, default classifieds for backcompat)
docs/checkpoints/mechanism/plan.md:324:- Cleanup 4×2.3G incomplete blobs to reclaim disk
docs/checkpoints/mechanism/plan.md:327:  HF_HUB_DOWNLOAD_TIMEOUT=600 huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --max-workers 1
docs/checkpoints/mechanism/plan.md:330:- Paper §5 generalization claim still defensible via P4 (selection-bias) + P5a/P5b (cross-site). P2/P3 are nice-to-have (family/size triangulation), not paper-critical.
docs/checkpoints/mechanism/plan.md:336:- P5a reddit holds → cross-site universal
docs/checkpoints/mechanism/plan.md:340:`axis2_layer_profile.md` + `fig_axis2_prompt_layer_profile.png`. Re-examine residual stream geometry per axis-isolated pair, full 37-layer cosine curves on `stage4_multimode_b1_{cls,reddit}` (288 ex each).
docs/checkpoints/mechanism/plan.md:346:| P-SoM↔SoM (image-axis ref) | axis-3 | 0.0412 | 0.0400 | 0.0411 | **L17** | 0.0412 |
docs/checkpoints/mechanism/plan.md:347:| DOM↔P-text (text fmt) | axis-1 | 0.0120 | 0.0254 | 0.0201 | **L23** | 0.0254 |
docs/checkpoints/mechanism/plan.md:348:| P-prompt↔P-SoM (text fmt) | axis-1 | 0.0113 | 0.0292 | 0.0201 | **L23** | 0.0292 |
docs/checkpoints/mechanism/plan.md:349:| P-text↔P-SoM (prompt fam, flat) | axis-2 | 0.0028 | **0.0114** | 0.0089 | L23 | 0.0114 |
docs/checkpoints/mechanism/plan.md:350:| DOM↔P-prompt (prompt fam, hier) | axis-2 | 0.0013 | 0.0050 | 0.0067 | L36 | 0.0067 |
docs/checkpoints/mechanism/plan.md:352:Reddit cross-site replicates: P-text↔P-SoM L23 = 0.0098 (vs cls 0.0114), same rank-order, same peak layer.
docs/checkpoints/mechanism/plan.md:355:1. **Distinct peak layers**: image L17 (fast sharp), text-format L23 (slower late-mid), prompt-family L23 (same timing as text-format on flat-text)
docs/checkpoints/mechanism/plan.md:356:2. **Distinct magnitudes**: image ~0.04, text-format ~0.03, prompt-family ~0.01 — 4:3:1 ratio
docs/checkpoints/mechanism/plan.md:357:3. **Cross-site rank stable**: reddit identical pattern
docs/checkpoints/mechanism/plan.md:359:**Reframe**: Axis-2 prompt-family is NOT null at residual stream. It's 3-4x weaker than axis-1 + peaks at L23 not L17. Method 4.2 plan §5.1 L17 snapshot 错失它. New paper §5 framing: layered three-axis hierarchy, image-axis dominant at L17 Mirage locus, text-format + prompt-family late-mid build at L23 parallel.
docs/checkpoints/mechanism/plan.md:363:### 7.3.0b Axis-2 per-task fragility check (2026-05-12 21:50 — /stress W2 defuse)
docs/checkpoints/mechanism/plan.md:365:`axis2_per_task_fragility.md` + `fig_axis2_per_task_fragility.png`. /stress reviewer 第一次 invocation W2 attack: 怀疑 axis-2 cosine 0.0114 mean 由 2-3 outlier 主导, 类比 h1_per_task_fragility 11% strict per-task. Defuse 实验:
docs/checkpoints/mechanism/plan.md:369:| **Axis-2 flat (P-text↔P-SoM)** | cls | 0.0132 | 0.0131 | [0.012, 0.014] | **100%** |
docs/checkpoints/mechanism/plan.md:370:| **Axis-2 flat (P-text↔P-SoM)** | reddit | 0.0121 | 0.0120 | [0.011, 0.013] | **100%** |
docs/checkpoints/mechanism/plan.md:371:| Axis-1 ref (DOM↔P-text) | cls | 0.0287 | 0.0280 | [0.025, 0.031] | 100% |
docs/checkpoints/mechanism/plan.md:372:| Axis-1 ref (DOM↔P-text) | reddit | 0.0260 | 0.0263 | [0.023, 0.031] | 100% |
docs/checkpoints/mechanism/plan.md:373:| Axis-3 image (P-SoM↔SoM) | cls | 0.0407 | 0.0415 | [0.035, 0.044] | 100% |
docs/checkpoints/mechanism/plan.md:378:3. **Cross-site rank stable** + magnitude near-identical (0.0132 cls vs 0.0121 reddit, < 9% diff)
docs/checkpoints/mechanism/plan.md:380:**/stress W2 attack defused completely**: axis-2 cosine gap 是 uniform per-task signature, 不是 aggregate artifact. 这与 H1 binary dichotomy 11% strict per-task fragile 形成对比 — H1 因为问 layer-comparison 离散问题易 fragile, axis-2 cosine 是 continuous mode-pair distance 即使 magnitude 小也 robust per-task.
docs/checkpoints/mechanism/plan.md:382:**Paper §5.7 增强**: 加入 per-task fragility 段, 明确每个 task 都贡献 axis-2 signal, 不是 2-3 outlier mean artifact.
docs/checkpoints/mechanism/plan.md:384:### 7.3.0a Exp 3 logit lens 输出层 amplification (2026-05-12 21:02)
docs/checkpoints/mechanism/plan.md:386:`axis2_logit_lens.md` + `fig_axis2_logit_lens.png`. 应用 Qwen3-VL-4B `model.model.language_model.norm` + `model.lm_head` to per-layer per-mode mean hidden states, 算 KL across 37 层.
docs/checkpoints/mechanism/plan.md:388:| Pair | Site | Peak L (KL) | Peak KL | Exp 1 cosine peak | 放大倍数 |
docs/checkpoints/mechanism/plan.md:390:| P-text↔P-SoM (axis-2 flat) | cls | **L23** | 0.162 | 0.011 | ~14x |
docs/checkpoints/mechanism/plan.md:391:| DOM↔P-prompt (axis-2 hier) | cls | L25 | 0.044 | 0.007 | ~7x |
docs/checkpoints/mechanism/plan.md:392:| DOM↔P-text (axis-1) | cls | L23 | 0.551 | 0.025 | 22x |
docs/checkpoints/mechanism/plan.md:393:| P-prompt↔P-SoM (axis-1) | cls | L23 | 0.695 | 0.029 | 24x |
docs/checkpoints/mechanism/plan.md:394:| Cross-site reddit | | L23-L25 | 0.13-0.62 | preserved | preserved |
docs/checkpoints/mechanism/plan.md:397:1. Axis-2 prompt-family **IS in output distribution** — KL 0.16 at L23, NOT null. Exp 1 cosine 0.011 is not the end of the story.
docs/checkpoints/mechanism/plan.md:398:2. **lm_head 10-25x amplification of cosine → KL** but axis-agnostic ratio preserved (axis-1/axis-2 ratio ~4.3 cls, ~4.9 reddit, vs cosine ratio ~3 — slight amplification of stronger axis but not breaking 3-4x rank).
docs/checkpoints/mechanism/plan.md:399:3. **KL @ L36 ≈ 0 paradox**: 因 mean hidden state at last layer collapse to common JSON format header. Mode-distinct signal concentrated in **L23-L25 decoding window** (not final embedding). This is the "knows but says differently" structural mirror of Wu et al. tool calling.
docs/checkpoints/mechanism/plan.md:401:**Paper §5.7 follow-up paragraph** added: 三轴 hierarchy persists at output distribution with same rank-order. Deployment routing (paper-2) should treat L23-L25 logit-lens features as cheapest highest-signal mode-axis discriminator.
docs/checkpoints/mechanism/plan.md:405:**P5a — Format variation H1 test on reddit** (`format_variation_h1_test_reddit.md`):
docs/checkpoints/mechanism/plan.md:407:| Variant | Peak L (reddit) | Peak L (cls baseline) |
docs/checkpoints/mechanism/plan.md:417:Caveats: small n (24×2=48/mode) makes 2/6 marks-like falling to L04 (appagent_id, plain_numbered) plausible as sampling noise; plain_sentence triggering L17 on reddit (not cls) suggests reddit narrative comments may pattern-match list semantics.
docs/checkpoints/mechanism/plan.md:419:**P5b — Mirage signature on reddit** (`stage4_method42_results_reddit.md`):
docs/checkpoints/mechanism/plan.md:423:| P-SoM ↔ DOM | **0.0098** (nearly 0) | similar (text-axis sibling) |
docs/checkpoints/mechanism/plan.md:424:| P-SoM ↔ SoM | **0.0423** | similar (image-axis split) |
docs/checkpoints/mechanism/plan.md:425:| P-SoM ↔ Vision | 0.0457 | similar |
docs/checkpoints/mechanism/plan.md:426:| DOM ↔ Vision peak | L04 = 0.0687 (AUROC=1.0) | L04 similar |
docs/checkpoints/mechanism/plan.md:428:→ **Cross-site Mirage replication ✓**: P-SoM behaves as text-axis sibling of DOM at L17 (image-feature reduction), not as image-axis sibling of SoM. paper §5 4-fold (d) drop-one mechanism holds on reddit.
docs/checkpoints/mechanism/plan.md:430:**Paper §5 cross-site evidence stack now complete**:
docs/checkpoints/mechanism/plan.md:431:1. P-SoM mid-layer mechanism (4-fold drop-one) — cls + reddit replicated ✓
docs/checkpoints/mechanism/plan.md:432:2. Indexed-list format → shortcut activation — directional consistency cls ↔ reddit ✓
docs/checkpoints/mechanism/plan.md:433:3. Mirage signature geometric structure — cls + reddit replicated ✓
docs/checkpoints/mechanism/plan.md:435:**P4 selection-bias defense (2026-05-12 18:50)** — cls reverse-tier H1 (`format_variation_h1_test_cls_reverse.md`):
docs/checkpoints/mechanism/plan.md:437:| Variant | strong-tier cls | reverse-tier cls | reddit |
docs/checkpoints/mechanism/plan.md:444:H1 mechanism in cls is **not tier selection artifact** (strong vs reverse both replicate). Reddit data paradoxically cleaner reveal of true L17 mid-layer fusion locus (cls L36 is monotonic-boundary artifact).
docs/checkpoints/mechanism/plan.md:459:| **Week 2** (2026-05-19 → 25) | Cross-site Method 4.2 (reddit) + reverse-tier Method 4.4 | Replication results + paper §5 §5 prose |
docs/checkpoints/mechanism/plan.md:465:- **§1 phantom routing space + 4-fold drop-in property** — completely independent of mechanism work, anchors Outcome / Macro / Efficiency dimensions. NOT in this folder; see `paper_planning.md` §1
docs/checkpoints/mechanism/plan.md:466:- **§6 cost-aware routing** — Peale et al. 2026 uncertainty decomposition anchor adds theoretical layer to phantom routing space's empirical AUROC. Method 4.2 cosine gap could serve as "reducible uncertainty" signal in deployment
docs/checkpoints/mechanism/plan.md:468:These two stay outside mechanism folder. Mechanism workspace is paper §5-specific.
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:1:# Stage 4 H1 test: indexed-list format variation
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:6:**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:7:- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:12:| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:14:| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0434 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:16:| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0429 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:28:- `[N] role 'label' (SoM)`: peak **L36** = 0.0429
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:45:- `AXTree (baseline DOM)`: peak **L04** = 0.0434
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:51:- **AXTree-DOM baseline**: peak L04
docs/checkpoints/paper_drafts/section8_limitations.md:3:This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. Shopping also has weaker intermediate mechanism coverage than classifieds/reddit, so site-level generalization is reported cell-by-cell rather than averaged into a universal web-agent claim. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
docs/checkpoints/paper_drafts/section8_limitations.md:7:VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, including the audit-derived false-positive filters from Sections 78a and 95, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
docs/checkpoints/paper_drafts/section8_limitations.md:11:Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
docs/checkpoints/paper_drafts/section8_limitations.md:15:The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). It also records two framings that survived audit: the four-fold drop-in property and the sparse L11/L17 mechanism. This affects Sections 1 and 5: the paper can claim confirmed, registry-backed evidence for those two framings, but not a preregistered exact-layer prediction or a universal single-task circuit.
docs/checkpoints/paper_drafts/section8_limitations.md:19:The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Exclusions are listwise only, at <=5% per cell under the B6 lock, so multiple imputation would add modeling assumptions without materially changing paired denominators. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
docs/checkpoints/paper_drafts/section8_limitations.md:21:\subsection*{8.6 Sparse-mechanism caveat}
docs/checkpoints/paper_drafts/section8_limitations.md:23:The activation-patching effect is sparse: at L17, four of five completed cells have median Levenshtein-distance shift equal to zero, with IQRs including zero. The blast radius is the mechanism claim: mean disruption and Holm significance are carried by task subsets, approximately the high-salience-image quarter of the strong-tier cases, while many tasks show no visible continuation change; Cell E random injection, by contrast, destroys outputs broadly. This affects Section 5 by reframing the finding as a task-conditional sparse mechanism rather than a universal mid-layer circuit. The claim remains defensible because Section 5.1 reports per-task scatter/violin views alongside mean bands and because Cell E separates content-specific sparse patching from generic injection damage.
docs/checkpoints/paper_drafts/section8_limitations.md:25:\subsection*{8.7 Compute, cost, and sustainability bounds}
docs/checkpoints/paper_drafts/section8_limitations.md:27:Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: the per-cell table reports GPU-hours, proxy-API USD cost, and kg-CO2 estimates with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware, benchmark site, and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates rather than a fine-grained lifecycle assessment.
docs/checkpoints/paper_drafts/section8_limitations.md:29:<!-- Bibkey audit 2026-05-12: lipton2018troubling, neurips2024checklist, holm1979sequentially, wang2023interpretability, zhang2024patching all verified present in paper.bib. -->
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:1:# Hero-claim bootstrap CI (W1 defuse)
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:3:Per-seed bootstrap 95% percentile CI on paired adjusted-SR diffs and drop-one oracle. B=10000, seed=42. Tasks resampled with replacement at task level.
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:5:**Defuse target**: /stress W1 attack — paper §1 hero claim 'P-SoM 13.81% > SoM 10.48% reddit' is statistically marginal under author's own 2σ hedge.
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:7:## reddit (N=210 same-task)
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:18:**Pairwise SR difference, bootstrap 95% CI:**
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:20:| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:22:| P-SoM vs SoM | +3.33 | +3.33 | [-0.95, +7.62] | 0.914 | 0.828 | 
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:24:| P-SoM vs DOM | +4.29 | +4.29 | [+0.00, +8.57] | 0.963 | 0.914 | 
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:26:| P-text vs DOM | +2.86 | +2.86 | [-0.95, +6.67] | 0.918 | 0.810 | 
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:28:| P-SoM vs P-text | +1.43 | +1.43 | [-1.90, +5.24] | 0.739 | 0.548 | 
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:31:**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:33:| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:44:## classifieds (N=234 same-task)
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:54:**Pairwise SR difference, bootstrap 95% CI:**
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:56:| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:58:| P-SoM vs SoM | -6.84 | -6.84 | [-12.39, -1.28] | 0.005 | 0.001 | 
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:60:| P-SoM vs DOM | +0.43 | +0.43 | [-3.42, +4.70] | 0.538 | 0.374 | 
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:62:| P-text vs DOM | +0.43 | +0.43 | [-3.42, +4.27] | 0.546 | 0.376 | 
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:64:| P-SoM vs P-text | +0.00 | +0.00 | [-4.27, +4.27] | 0.464 | 0.317 | 
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:67:**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:69:| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:82:Read the **reddit P-SoM vs SoM** row + **reddit drop-one P-SoM** row:
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:84:- If both CIs are strict-positive (ci_lo > 0) AND P(diff > 0) > 0.95 → **W1 attack defused**,   §1 hero claim is bootstrap-supported. Remove the '2σ hedge' from line 5, lead with the magnitude.
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:85:- If CIs cross zero but P(diff > 0) > 0.80 → **W1 partially defused**, the claim is directional
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:87:  the author already wrote, but the complementarity (Jaccard / drop-one positive on N=7 tasks) carries
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:89:- If P(diff > 0) < 0.80 → **W1 sustained**, §1 hero claim must rewrite to 'parity / complementarity
docs/checkpoints/mechanism/results/axis2_layer_profile.md:4:(P-SoM↔P-text 0.0028, DOM↔P-prompt 0.0013). But forest plot drop-one places P-SoM as unique hero,
docs/checkpoints/mechanism/results/axis2_layer_profile.md:7:**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.
docs/checkpoints/mechanism/results/axis2_layer_profile.md:8:Overlay axis-1-only (text swap, prompt fixed) + image-axis P-SoM↔SoM reference curves to calibrate scale.
docs/checkpoints/mechanism/results/axis2_layer_profile.md:10:## Results — classifieds site (stage4_multimode_b1_cls, 288 ex)
docs/checkpoints/mechanism/results/axis2_layer_profile.md:14:| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0013 | 0.0067 | **L36** | 0.0067 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:15:| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0028 | 0.0089 | **L23** | 0.0114 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:16:| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0134 | 0.0120 | 0.0201 | **L23** | 0.0254 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:17:| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0127 | 0.0113 | 0.0201 | **L23** | 0.0292 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:18:| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0394 | 0.0412 | 0.0411 | **L17** | 0.0412 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:20:## Results — reddit site (stage4_multimode_b1_reddit, 288 ex)
docs/checkpoints/mechanism/results/axis2_layer_profile.md:24:| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0012 | 0.0059 | **L36** | 0.0059 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:25:| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0027 | 0.0080 | **L23** | 0.0098 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:26:| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0125 | 0.0092 | 0.0183 | **L23** | 0.0217 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:27:| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0115 | 0.0086 | 0.0176 | **L23** | 0.0240 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:28:| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0434 | 0.0423 | 0.0434 | **L4** | 0.0434 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:32:Three hypotheses about axis-2 mechanism layer:
docs/checkpoints/mechanism/results/axis2_layer_profile.md:34:1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.
docs/checkpoints/mechanism/results/axis2_layer_profile.md:35:2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.
docs/checkpoints/mechanism/results/axis2_layer_profile.md:36:3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.
docs/checkpoints/mechanism/results/axis2_layer_profile.md:38:Compare peak layers above against axis-1 (text-format) pairs (the established mechanism with L17 peak) and image-axis reference (~0.04 magnitude). If axis-2 pair peak < 0.01 at all layers, hypothesis 1 holds.
docs/checkpoints/paper_drafts/section2_background.md:5:Modern web agents differ less in the browser actions they expose than in the observation representation they give to the language model. Text-only agents typically serialize the Document Object Model or Accessibility Tree (AXTree) into a hierarchical text observation. WebArena uses this style of realistic browser environment to evaluate language-guided agents on shopping, forum, map, and software-development tasks \citep{zhou2024webarena}. Mind2Web similarly frames web interaction as selecting actions from structured page elements collected across real websites \citep{deng2023mind2web}. This line of work makes DOM-derived text the default low-cost representation: it is cheap, symbolic, and compatible with language-only models, but it can be verbose and blind to visual appearance.
docs/checkpoints/paper_drafts/section2_background.md:7:Multimodal web agents add screenshots to the observation. VisualWebArena extends WebArena with visually grounded tasks and evaluates agents that combine page text, screenshots, and visual grounding cues \citep{koh2024visualwebarena}. A common grounding device is Set-of-Mark prompting, introduced by Yang et al. as a way to overlay numbered or speakable marks on image regions so a multimodal model can refer to visual objects by discrete IDs \citep{yang2023som}. SeeAct likewise studies GPT-4V as a generalist web agent and finds that visual understanding must still be paired with reliable action grounding \citep{zheng2024seeact}. Magma pushes the same broad direction into an omni-modal agent foundation model with action grounding and multimodal pretraining \citep{yang2025magma}. Vision-only baselines remove the DOM/AXTree channel and ask the model to act from the screenshot alone; these baselines test whether visual perception can substitute for structured symbolic grounding.
docs/checkpoints/paper_drafts/section2_background.md:9:Across this literature, DOM, SoM, and Vision are treated as orthogonal observation modes. SoM in particular is treated as a multimodal bundle: a marked screenshot plus a text legend that maps mark IDs to elements. The `[SOM_MARKS]` text is normally an auxiliary index for the marked image, not a controlled standalone variable. This convention is the gap our paper targets. We ask what routing behavior emerges when the annotated image is skipped while the remaining factors are held apart: AXTree versus `[SOM_MARKS]` text, DOM versus SoM prompt family, and image-off versus image-on evaluation. The resulting object is not a claim that marked or text-only observations are new artifacts; it is a controlled characterization of the phantom routing space around **Phantom-SoM**.
docs/checkpoints/paper_drafts/section2_background.md:13:Routing has become a standard response to heterogeneous cost and capability. FrugalGPT frames inference as a cascade over multiple LLM APIs, learning when cheaper models can answer and when to escalate to stronger models [Chen et al. 2023]. RouteLLM similarly learns routers from preference data to choose between weaker and stronger LLMs under cost-quality tradeoffs [Ong et al. 2025]. These systems are important precedents for cost-aware inference, but their arms are models. The input representation is usually fixed while the backend model changes.
docs/checkpoints/paper_drafts/section2_background.md:17:What is missing is representation-level routing within a single model: selecting between different text formats generated from the same browser state. DOM/AXTree and `[SOM_MARKS]` can contain overlapping element semantics, but their token geometry is different. One is hierarchical, nested, and metadata-rich; the other is flat, indexed, and compact. Prior routing work does not ask whether a single model should see the same page as an AXTree for some tasks and as an isolated marks list for others. Phantom-SoM makes that missing routing axis explicit.
docs/checkpoints/paper_drafts/section2_background.md:19:This distinction also separates our setting from ordinary prompt selection. A representation arm is not just a different instruction template; it changes the observation object that enters the agent loop at every step. If two formats derived from the same browser state route the same model into different exploration policies, then representation becomes a deployable control surface. The router need not choose a larger model or a visual encoder first. It can choose a cheaper textual view of the page, observe whether the trajectory stalls, and escalate only when the cheap representation appears misaligned with the task.
docs/checkpoints/paper_drafts/section2_background.md:23:The plausibility of Phantom-SoM rests on a broader fact about language models: semantically equivalent prompts can induce different behavior when their surface form changes. Sclar et al. quantify language-model sensitivity to spurious prompt-format features and show that small formatting choices can produce large accuracy differences, even when the underlying task semantics are unchanged [Sclar et al. 2024]. Mishra et al. show a related effect for instructional prompting: reframing instructions into forms better aligned with a model's learned language can change few-shot performance [Mishra et al. 2022].
docs/checkpoints/paper_drafts/section2_background.md:25:These studies do not study web agents, but they explain why web observations should not be treated as neutral containers. A page serialized as AXTree text is not merely "the same information" as a page serialized as `[SOM_MARKS]`. The model receives different punctuation, ordering, indentation, repeated role tokens, ID patterns, and local neighborhoods. Those tokens prime different latent states and therefore different action distributions.
docs/checkpoints/paper_drafts/section2_background.md:27:For a web agent, prompt-format sensitivity matters at the trajectory level. The model is not producing a single label; it is choosing whether to search, click, scroll, revisit a page, or finish. Section 4 and Section 5 build on this theoretical anchor: the flat marks list tends to shift exploration toward quick element selection, while AXTree hierarchy tends to support sustained navigation and search. Prompt wording also matters, but our two-knob account separates the layers: text representation shapes how the agent explores, while prompt family tunes when it commits.
docs/checkpoints/paper_drafts/section2_background.md:31:Cost-efficient web-agent inference has usually meant pruning or scheduling expensive context. AXTree observations are long, noisy, and security-sensitive. FocusAgent addresses this by using a lightweight retriever to trim AXTree observations before sending them to the main agent, reducing context while preserving the hierarchical representation [Kerboua et al. 2025]. This is a natural text-efficiency strategy: keep the DOM-derived tree, but remove irrelevant lines.
docs/checkpoints/paper_drafts/section2_background.md:33:Multimodal inference adds a second cost source: visual encoding. Image inputs increase prompt-processing time, memory pressure, and time-to-first-token. ModServe characterizes large multimodal model serving and shows that multimodal workloads have heterogeneous stages and resource requirements, motivating modality- and stage-aware resource disaggregation [Qiu et al. 2025]. In web agents, full SoM therefore has two costs: it prepares a marked screenshot and it sends image tokens to the model.
docs/checkpoints/paper_drafts/section2_background.md:35:Phantom-SoM explores a different kind of efficiency. It is not text pruning and it is not image scheduling. It is text reformatting. The `[SOM_MARKS]` list can be generated from the same browser/AXTree metadata already available to the agent, then sent without the marked screenshot. This removes image-token cost while preserving a discrete element index. In our runs the text observation is comparable in token length to the corresponding AXTree (within ±7% on reddit and classifieds, holding the system prompt fixed); the difference is in structure — flat indexed list versus nested hierarchy with url/tab metadata — rather than in length. The open question is whether such a representation is only a structural rewrite of DOM, or whether its format creates a distinct success pool. Our empirical sections answer the latter.
docs/checkpoints/paper_drafts/section2_background.md:37:This matters because many cost reductions trade away information: smaller models, shorter context windows, lower image resolution, or fewer retrieved lines. Phantom-SoM instead tests whether a cheap re-arrangement of already available text can expose a different reasoning path. If it succeeds on tasks missed by DOM, the gain is not merely compression; it is complementarity. That is why Section 4 reports both single-mode success and drop-one oracle value rather than treating token savings alone as the contribution.
docs/checkpoints/paper_drafts/section2_background.md:41:This paper positions Phantom-SoM at the intersection of four literatures that are usually studied separately. First, SoM and its descendants, including Magma and Ferret-UI 2, use marks as visual grounding devices and generally keep text tied to the marked image \citep{yang2023som,yang2025magma,li2025ferretui2}. Second, web-agent benchmarks compare DOM, SoM, and Vision modes, but do not use the mark-text-without-image condition as a controlled axis for routing characterization \citep{zhou2024webarena,koh2024visualwebarena,zheng2024seeact}. Third, routing systems optimize over models, modalities, or experts, not over text formats of the same browser state [Chen et al. 2023; Ong et al. 2025; Li et al. 2026]. Fourth, prompt-format work predicts sensitivity to representation syntax, but has not measured task-pool complementarity in interactive web agents \citep{sclar2024promptformat,mishra2022reframing}.
docs/checkpoints/paper_drafts/section2_background.md:43:There are important artifact precedents, and this paper treats them as context rather than as targets to out-claim. SoM-Mark already pairs textual mark references with visual marks \citep{yang2023som}; SeeAct explores marked-screenshot web-agent grounding \citep{zheng2024seeact}; and Magma incorporates related SoM-style and action-grounding ideas into an omni-modal agent model \citep{yang2025magma}. These systems show that marked observations, textual references, and multimodal action grounding are not new merely as artifacts. The contested point is different: whether the behavior of the image-skipped configurations has been scientifically isolated, compared against DOM/SoM/Vision on identical task pools, and explained mechanistically.
docs/checkpoints/paper_drafts/section2_background.md:45:The resulting gap is therefore a characterization gap, not a first-deployment gap. Published systems and benchmarks have not, to our knowledge, provided a controlled scientific evaluation of the phantom boundary: `[SOM_MARKS]` isolated from the screenshot, crossed with prompt family in a 2-by-2 control, compared through low-overlap success pools and drop-one oracle value, and checked against mechanistic controls including mid-layer L11-L17 evidence and random injection. This is also where the 4-fold drop-in property belongs. It is an empirical finding about a controlled configuration: zero image tokens, no new model, usable routing signal, and positive incremental oracle value. It is not a claim that text-only browser control or marked observations were first introduced here.
docs/checkpoints/paper_drafts/section2_background.md:47:Our contributions follow directly. We define **Phantom-SoM** as the deployment-relevant representative of the phantom routing space: marks text plus SoM prompt family, with the image removed. We show empirically that Phantom-SoM is not a degenerate DOM surrogate: it contributes independent oracle value and has substantial task-pool non-overlap with DOM, SoM, and Vision. The P-text and P-prompt controls establish specificity by separating text-payload flattening from prompt-family effects, rather than attributing all gains to a single novel arm. Finally, we provide mechanism evidence for a two-knob account: representation format shapes exploration, while prompt wording tunes commitment confidence. This motivates the experimental design in Section 3 and the controlled evidence in Section 4.
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:1:# Stage 4 H1 test: indexed-list format variation
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:6:**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:7:- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:12:| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:17:| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0495 |
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:18:| som_standard | `[N] role 'label' (SoM)` | marks-like | **L17** | 0.0429 |
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:28:- `[N] role 'label' (SoM)`: peak **L17** = 0.0429
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:45:- `AXTree (baseline DOM)`: peak **L04** = 0.0495
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:51:- **AXTree-DOM baseline**: peak L04
docs/checkpoints/paper_drafts/section4_empirical_findings.md:3:This section reports empirical evidence that web-agent observation representations should be treated as routing arms, not only as fidelity levels. The key surprise is that **Set-of-Mark text alone**, with the marked screenshot removed, does not collapse to a DOM-like baseline. Instead, it behaves as a distinct text-only arm whose successes only partially overlap with DOM, full SoM, and vision-only observations. We refer to this arm as **Phantom-SoM**: the agent receives the `[SOM_MARKS]` textual element list and the SoM-style prompt, but no image.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:5:Throughout this section, we distinguish three measurement conventions. **Raw SR** is the evaluator success rate in `condition_summary_v2.json`. **Adjusted SR** subtracts confirmed false-positive terminal answers on not-applicable or evaluator-mismatch tasks. **Same-task adjusted SR** uses the same task set for all arms within a site. Unless otherwise noted, claims use same-task adjusted SR on completed B0 VisualWebArena classifieds and reddit runs. We also treat small cell-to-cell differences cautiously: under same-condition repeats, we observe roughly **+/-5% task-set variance**, so individual differences below about **2 pp** should be interpreted as noise-floor evidence rather than stable rankings.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:9:We evaluate a single strong API-backed web agent, denoted **B0**, on two completed VisualWebArena sites: classifieds and reddit. The completed B0 pool contains **234 classifieds tasks** and **210 reddit tasks** for each reported observation condition:
docs/checkpoints/paper_drafts/section4_empirical_findings.md:13:| DOM | AXTree / DOM-derived text | DOM | No | Hierarchical text baseline |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:14:| SoM | `[SOM_MARKS]` text plus marked screenshot | SoM | Yes | Full Set-of-Mark baseline |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:15:| Vision | Screenshot without SoM marks | Vision | Yes | Visual-only baseline |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:16:| Phantom-SoM | `[SOM_MARKS]` text only | SoM | No | Isolated marks-text representation |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:17:| P-text | `[SOM_MARKS]` text only | DOM | No | Prompt-family control for marks text |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:19:The first three arms are the original Phase 1 representation baselines. Phantom-SoM is the new ablation arm. P-text is a prompt-family control: it receives the same marks-text-only observation as Phantom-SoM but uses the DOM prompt. We report all five modes for descriptive SR, cost, and latency. For the main routing-value claim, we keep the primary drop-one oracle on the four-arm comparison used throughout the paper: DOM, SoM, Vision, and Phantom-SoM.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:21:The original intuition was that Phantom-SoM should be either a broken SoM configuration or a weak DOM surrogate: it keeps a prompt that says the agent is operating with marked visual context, but removes the marked screenshot. The empirical results reject that collapse story. Phantom-SoM is lower than full SoM on classifieds, where marked screenshots carry clear visual grounding value, but it matches or modestly exceeds full SoM on reddit under adjusted SR.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:25:The single-mode success rates show a site-modulated effect. On classifieds, full SoM remains the strongest individual representation. On reddit, Phantom-SoM is at least competitive with the strongest baselines, while using no image input. The table reports adjusted SR, because Figures 1, 2, 7, and 8 use episode-level `adjusted_success` for the paper comparisons. The latency column is p95 step latency from `condition_summary_v2.json`; cost is average total cost per task.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:27:| Site | Arm | Adjusted SR | Avg cost | p95 step latency | Metric |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:29:| Classifieds | DOM | 14.10 | $0.043 | 37.5s | N=234 |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:30:| Classifieds | SoM | **21.37** | $0.042 | 74.0s | N=234 |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:33:| Classifieds | Phantom-SoM | 14.53 | $0.044 | 18.2s | N=234 |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:34:| Reddit | DOM | 9.52 | $0.052 | 73.6s | N=210 |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:35:| Reddit | SoM | 10.48 | $0.041 | 58.9s | N=210 |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:38:| Reddit | Phantom-SoM | **13.81** | $0.038 | 51.4s | N=210 |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:40:The classifieds result is the expected sanity check: when tasks benefit from visual page layout and product imagery, the marked screenshot adds useful grounding and full SoM is clearly best (**SoM 21.37 vs Phantom-SoM 14.53; N=234; adjusted**). Phantom-SoM is close to DOM on classifieds (**14.53 vs 14.10**), but this is not a dominance claim; it is inside the noise floor and far below full SoM.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:42:The reddit result is the counterintuitive case. Removing the image does not eliminate the value of the SoM representation: Phantom-SoM matches or modestly exceeds full SoM and DOM on adjusted SR (**13.81 vs SoM 10.48 vs DOM 9.52; N=210; adjusted**). Given the variance we observe in repeats, the **+3.33 pp** gap over SoM is near the boundary of what should be treated as stable. We interpret this as evidence that Phantom-SoM is competitive on text-dominated reddit threads, not as an unconditional single-cell dominance claim. The more robust pattern is the cross-site asymmetry: **classifieds favors full SoM; reddit does not**. We treat that asymmetry as mechanism evidence rather than a setup bug: Section 5 shows a related site-modulated capability shift, with B0-to-B1 SoM visual-hijack/click-loop increasing by **+50.0 pp** on classifieds and **+33.3 pp** on reddit.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:44:This pattern suggests that the `[SOM_MARKS]` list is doing more than serving as a caption for a screenshot. It is a compact, flat, indexed text representation. Compared with AXTree-style DOM text, it removes much of the hierarchical nesting and metadata, and presents candidate actions as a linear set of marked elements. The outcome is not uniformly better, but it can push the agent toward a different solution basin.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:46:The cost and latency columns make the routing tradeoff concrete. On classifieds, Phantom-SoM's average cost is effectively in the same band as DOM and SoM (**$0.044 vs $0.043 vs $0.041**), but its p95 step latency is much lower than full SoM (**18.2s vs 74.0s**, roughly 4x faster). On reddit, Phantom-SoM is the cheapest of the main text/SoM-style arms (**$0.038 vs SoM $0.041 vs DOM $0.052**) and remains faster at p95 step latency than full SoM (**51.4s vs 58.9s**). These numbers support the cost-aware routing interpretation in Figures 7 and 9 without requiring Phantom-SoM to win every site.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:48:Raw SR tells the same high-level story but should not be mixed with adjusted SR. Some arms lose points after false-positive adjustment. Because the paper claim concerns deployable task success rather than answer attempts that only appear correct under a noisy evaluator, we use adjusted SR for the main empirical comparisons.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:52:Single-mode SR can hide routing value. A representation may have modest average SR while still solving tasks that the other arms miss. We therefore compute a drop-one oracle: form the oracle union over the four primary arms, remove one arm, and measure how much oracle SR falls. This loss is the arm's incremental contribution to the routing pool.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:56:| Classifieds | SoM -8.55 pp | Vision -3.42 pp | Phantom-SoM -2.56 pp | DOM -2.14 pp | Drop-one oracle loss, N=234, adjusted |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:57:| Reddit | Phantom-SoM -3.33 pp | DOM -1.90 pp | SoM -1.90 pp | Vision -1.43 pp | Drop-one oracle loss, N=210, adjusted |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:59:The classifieds oracle is consistent with the single-mode story: full SoM contributes the most unique oracle value, followed by vision. Phantom-SoM still has a non-zero loss (**2.56 pp; N=234**), but the main effect on classifieds belongs to visual grounding.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:61:The reddit oracle is the stronger routing signal. Phantom-SoM has the largest nominal drop-one loss in the fresh four-arm oracle (**3.33 pp; N=210**), while DOM and SoM each contribute **1.90 pp** and Vision contributes **1.43 pp**. Because these are small absolute task counts, we do not read the ordering as a precise rank claim. The important point is that Phantom-SoM is comparable to the top routing-value arms and is not subsumed by DOM, SoM, or Vision.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:63:The overlap view supports the same conclusion. In the four-arm oracle, Phantom-SoM contributes a concrete reddit-only set of seven tasks (**7, 15, 36, 94, 157, 162, 167**) and a non-zero classifieds set as well. Two examples illustrate the kind of work this arm is doing. On reddit task 7, Phantom-SoM searched for the cake-recipe post and navigated directly to the OP recipe comment permalink. On reddit task 162, it searched within /f/wallstreetbets, scrolled hot posts, and returned the GIF URL for the retirement-account-versus-brokerage-account prompt. These are not proof of a universal mechanism by themselves, but they make the drop-one value concrete: the arm is adding recoverable successes, not only shifting aggregate percentages.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:65:The main empirical claim is therefore not that Phantom-SoM dominates the other modes. It does not. The claim is that it is an **independent routing arm**: it opens a distinct task pool at text-only cost, with the strongest relative benefit on the text-dominated reddit site and a clear visual-grounding disadvantage on classifieds.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:69:The five-mode result raises a confound: is Phantom-SoM useful because of the `[SOM_MARKS]` text representation, or because the SoM prompt changes the agent's confidence and behavior even without an image? P-text separates these factors. The full clean P-text runs are reported above for SR, cost, and latency; for behavioral mechanism, we use the verified same-task reddit subset of **N=48**, where all four cells of the prompt-by-representation ablation were manually checked.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:71:> **Text format shapes how the agent explores. Prompt wording tunes when the agent commits.**
docs/checkpoints/paper_drafts/section4_empirical_findings.md:73:The first knob is exploration shape. On the same-task reddit ablation subset, replacing AXTree text with `[SOM_MARKS]` text shifts macro behavior away from DOM-like search loops and toward Phantom-SoM-like quick decisions. The verified search-loop rate is **22.7% for DOM** but **10.8% for Phantom-SoM and 10.8% for P-text** (**N=48; behavior metric; same-task subset**). The prompt change alone does not pull P-text back to DOM-like exploration. This supports the representation-driven part of the hypothesis: the flat marks list, not only the SoM prompt, changes the trajectory distribution.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:75:The second knob is commitment confidence. On the same N=48 subset, DOM and P-text have identical raw-to-adjusted SR gaps, while Phantom-SoM has a smaller gap:
docs/checkpoints/paper_drafts/section4_empirical_findings.md:79:| DOM prompt | DOM | 18.75 | 12.50 | 6.25 pp | 3 | N=48, raw/adjusted |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:80:| DOM prompt | P-text | 18.75 | 12.50 | 6.25 pp | 3 | N=48, raw/adjusted |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:81:| SoM prompt | Phantom-SoM | 18.75 | 16.67 | 2.08 pp | 1 | N=48, raw/adjusted |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:82:| SoM prompt | SoM | 22.92 | 16.67 | 6.25 pp | 3 | N=48, raw/adjusted |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:84:The aggregate SR equality should not be overread as task-level identity: equal counts such as 6/48 can occur with different solved-task sets. The robust signal is the false-positive pattern. DOM-prompt arms have the larger false-positive gap (**DOM and P-text: 3 N/A false positives, 6.25 pp gap; N=48**). The SoM-prompt Phantom arm has fewer N/A false positives (**1 N/A false positive, 2.08 pp gap; N=48**). This indicates that prompt wording affects terminal-action calibration: when the model decides it has enough evidence to `finish`.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:86:The two-knob account reconciles the apparent tension. The representation is the novel routing axis because it changes the agent's default exploration path. The prompt is a secondary but real tuning knob because it changes commitment confidence. Both are needed to explain the ablation. A representation-only story misses the FP gap, while a prompt-only story cannot explain why P-text follows Phantom-SoM rather than DOM on search-loop behavior.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:88:These findings explain why Phantom-SoM can be valuable despite not winning every single-mode comparison. Routing benefits depend on complementarity, not only average SR. A flat marks list can be worse for tasks that need hierarchy or visual layout, yet better for tasks where the same hierarchy induces over-searching. The practical implication is a cost-aware cascade: try cheap text representations first, use behavioral signals to detect when their exploration is unproductive, and escalate to full SoM when visual grounding is likely to matter.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:92:The five-mode aggregate above conceals site-specific behavior. Section 4.5 fills in the reddit substrate. The Section 5 mechanism analysis explains *where in the model* the three axes appear; this subsection explains *what the agent does differently on reddit* under each axis swap, using outcome, macro, and micro behavioral evidence rather than residual-stream geometry.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:96:The reddit environment in VisualWebArena is a Postmill-style forum rather than a visually organized marketplace. Its stable information structure is a hierarchy of forums, posts, and comments. The relevant navigation objects are therefore mostly textual: sidebar links to `f/<forum>` pages, post titles, comment-count links, comment permalinks, sort controls, and a global search box. The URL structure mirrors this hierarchy through path-based routes such as `/f/<forum>/<post>/<comment>`, so moving to the right page normally means choosing the right textual object in the forum tree rather than manipulating a visual layout.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:98:This substrate makes reddit an informative test case for separating the three axes. Images are frequent in the task prompts and in the posts themselves, but their role is usually evidential: an image can identify which post is being discussed, or disambiguate a content clue, but it is not the site's primary navigation affordance. The browser screenshot does not create the forum hierarchy; it only renders it. Conversely, the search box is prominent in the DOM and AXTree, but intrinsic search is not the intended substrate for many tasks. Repeated search is a failure basin: the agent can keep refining keywords while never taking the forum, post, or comment link that would satisfy the evaluator. The mechanism to explain is therefore not simply "text works better than vision." It is that each representation changes which textual affordances become salient enough for the model to commit to.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:102:Axis 1 changes the observation text from a hierarchical AXTree to a flat `[SOM_MARKS]` list while holding the DOM prompt fixed. On reddit this is the primary substrate-level mechanism. In the AXTree condition, the sidebar and post/comment links are embedded in a deep tree with many roles, containers, headings, and repeated page metadata. The search box is an easy high-level object, so the agent often converts the user intent into a query and then remains inside the search loop. In the flat marks condition, candidate links are serialized as a more uniform indexed action surface. This does not add image information and does not substantially change token budget; it changes the local attention pattern over action candidates. The forum link or comment permalink is no longer buried inside a nested accessibility structure, so the model is more likely to treat it as a clickable route rather than first translating the task into search terms.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:104:The evidence chain is consistent across dimensions. At the outcome level, adding P-text to the three-mode baseline contributes oracle value even without the SoM prompt or screenshot (Outcome 0c, +P-text +3.21pp single-phantom lift on the current oracle intersection). At the macro level, the whole-run strategy gradient shows the failure basin directly: reddit search-loop rate falls from DOM to Phantom-SoM and then to full SoM (Macro 1c, search-loop 51.90%->35.71%->31.43%). The axis-1-only macro effect is smaller than the compound prompt path, which is expected if flat text mainly changes which page objects are reachable rather than merely changing the action vocabulary. The micro evidence is sharper: DOM versus P-text has low path overlap for a text-only swap (Micro 2a, URL-path Jaccard 0.573), improves target-page reach (Micro 2b, target-hit +3.47pp), and reduces repeated keyword reuse (Micro 2c, max-keyword-repeat -0.633). The click-target view tells the same story: the two modes choose substantially different element sets even before images enter the system (Micro 2a-extra, click-target Jaccard 0.463).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:106:The standalone success table supports the same interpretation without requiring oracle selection. P-text raises adjusted SR over DOM on the full reddit set while preserving the same no-image deployment class (Outcome 0a, DOM 9.52% versus P-text 12.38%, N=210). The effect is not simply that P-text acts more: it uses fewer steps on average in the cost summary, and the cascade's largest text-axis macro shift is action-repeat rather than search success itself (Macro 1b, text-axis action-repeat +4.64pp). This is compatible with a routing-surface mechanism. The agent sometimes repeats a newly exposed marked control, but it is no longer confined to the same query-rewrite loop.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:108:Efficiency further constrains the explanation. Because P-text is generated from the same AXTree-derived text source and does not attach a screenshot, the reddit improvement cannot be attributed to paying the visual-token tax (Efficiency 3a, DOM $0.0516/episode versus P-text $0.0459/episode in the site dictionary). Axis 1 is therefore a representation effect: the observation text is rearranged into an indexed list, not enriched with new visual evidence.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:110:The mechanism is not monotone in every individual task, which is useful because it identifies the boundary condition. Reddit task #81 asks the agent to upvote every PhotoshopBattles post on the current page whose picture contains a cat. DOM succeeds by using both title semantics and button-state feedback: after an upvote, the observation exposes enough state change for the agent to move on to the next cat post. P-text matches DOM through the early actions but then collapses onto the same marked upvote control after the state should have changed. The case is a negative example for a simplistic "flat marks always help" claim. Axis 1 helps when the bottleneck is finding the right navigation object, but flat serialization can remove or weaken action-state cues such as `Upvote` becoming `Retract upvote`. On reddit, the aggregate effect is positive because the dominant failure basin is route discovery through text, not per-button state tracking.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:114:Axis 2 holds the flat `[SOM_MARKS]` text fixed and changes only the prompt family from DOM-style interaction to SoM-style marked-element interaction. On reddit this axis is secondary to the substrate shift, but it is the strongest macro driver of search and typing behavior. The SoM prompt asks the model to point at marked elements, which changes the prior over when to keep querying and when to commit to a visible candidate. In practical terms, it makes the agent more conservative about long exact queries and more willing to use marked links, comment anchors, tab focus, or backtracking after a stagnant page.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:116:The outcome evidence suggests a calibration effect rather than a uniform success-rate increase. Phantom-SoM has the lowest false-positive rate among the B0 reddit modes (Outcome 0b, P-SoM FP rate 0.48%), and adding P-SoM as a single phantom arm contributes additional oracle tasks beyond the three standard modes (Outcome 0c, +P-SoM +2.56pp). The task-pool overlap also stays below the redundancy sentinel: P-text and P-SoM solve overlapping but not identical task sets (Outcome 0d, P-text<->P-SoM Jaccard 0.500). Thus the prompt is not just a cosmetic instruction layered on top of the same decisions; it changes which tasks enter the solved pool.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:118:The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:120:The prompt contrast is also visible in the mode-invariant click-target metric. With the text payload held fixed, P-text and P-SoM still have low click-target overlap (Micro 2a-extra, P-text<->P-SoM click-target Jaccard 0.484). This matters because it rules out a purely verbal explanation in which the SoM prompt only changes confidence wording at `finish`. The prompt changes which marked objects are selected during navigation.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:122:Reddit task #7 is the cleanest case study. The task asks for the permalink to the original poster's recipe comment for an image post. DOM overfits the visual description into a long exact query about a cake with cranberries and rosemary, spends 30 steps cycling through empty or unhelpful search results, and never reaches the comment permalink. Phantom-SoM instead treats the task as finding an OP recipe comment, searches more broadly for `cake recipe`, and reaches the comment permalink in five steps. The important point is not that the image is absent; both traces must infer from the same task context. The prompt shifts the query breadth and the commitment target, and the marked-comment affordance gives the model a terminal object that a search-loop policy fails to use.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:126:The image axis adds the marked screenshot to Phantom-SoM, yielding full SoM. On reddit the net effect is weak and bidirectional because images are mostly content clues, not navigation structure. When the task requires recognizing the depicted post, the screenshot can help identify a candidate. But the same screenshot can over-anchor the agent on image URLs or visually salient marked regions, especially when the evaluator requires a post page, comment page, or action state rather than the image asset itself.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:128:The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:130:The image-axis micro contrast confirms that the screenshot is behaviorally strong even when it is not outcome-positive. P-SoM and SoM have low URL-path agreement and frequent immediate divergence (Micro 2a, image-axis URL-path Jaccard 0.456; Micro 2f, early divergence 95.24%). Thus "weak" should be read as weak net value on this substrate, not as weak causal force. The screenshot changes decisions; on reddit, those changed decisions often point to content assets rather than evaluator-relevant post or comment routes.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:132:Reddit task #0 illustrates the harmful channel. The task asks for the sushi-platter post and its comments section. Phantom-SoM initially clicks the sushi image URL several times, but it eventually recovers and selects the actual post URL `/f/food/82896/i-ate-sushi-platter`, then targets the comment link. Full SoM remains trapped for the full budget, alternating between the sushi image URL and the forum page. The screenshot correctly identifies the sushi platter; the failure is not visual ignorance. The failure is action-policy over-anchoring on the marked image element, where visual salience suppresses the neighboring post/comment route. This is exactly why reddit should be described as text-dominated rather than image-free: image evidence exists, but its marginal value depends on whether it guides the agent toward a forum route or into an asset-level loop.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:134:The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:136:### 4.5.5 Compound Axis 1+2: Phantom-SoM Versus DOM
docs/checkpoints/paper_drafts/section4_empirical_findings.md:138:The compound transition from DOM to Phantom-SoM combines the flat text payload with the marked-element prompt while still avoiding the image channel. On reddit this compound arm is best interpreted as a complementary routing arm rather than as a uniformly stronger baseline. Its adjusted SR is modestly higher than DOM on the full 210-task per-mode summaries (Outcome 0a, P-SoM 13.81% versus DOM 9.52%, +4.29pp), but the more robust mechanism evidence is the divergence in which routes it explores. DOM and P-SoM have low click-target overlap (Micro 2a-extra, compound click-target Jaccard 0.421) and low URL-path overlap (Micro 2a, compound URL-path Jaccard 0.481). The oracle result then follows naturally: P-SoM adds tasks that the original three-mode set misses (Outcome 0c, +P-SoM +2.56pp), and P-text/P-SoM are not redundant with each other (Outcome 0d, Jaccard 0.500).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:140:This interaction also explains why Section 4's evidence catalog should not be read as a single leaderboard. DOM, P-text, and P-SoM can have close aggregate success rates while visiting different pages, selecting different marked elements, and failing in different basins. Axis 1 exposes alternative textual routes; Axis 2 changes the policy's willingness to commit to those routes. Their combination is useful because it changes the task pool available to a router, not because it dominates every individual task.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:142:The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:19:hallucinated rationale, and length-dependent confidence). Static audit of 87 N/A-task FP
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:26:variants (raw / +na_fp / +na_fp+eval_fp), so judge drift cannot flip the paper's hero claim.
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:31:than retract the SR claim.
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:54:VWA's `program_html` evaluator scores tasks by goto'ing a target URL and querying DOM with
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:57:(e.g., Magento's `.order-details-items.ordered`, classifieds' `.price` / `.desc`) that
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:60:wrong DOM node or miss the intended element entirely.
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:63:program_html task, we count post-action DOM nodes matching the reference selector. A pre/post
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:80:not a runner / dispatch / observation failure.
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:100:**Implication for our claims**: This bug exists in upstream VWA and is documented in our
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:101:CLAUDE.md as "DOM has structural information advantage." It systematically helps DOM mode
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:102:relative to Vision/SoM modes by exposing element text that is visually clipped. We do **not**
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:104:(c) it does not affect our **paired** comparisons (P-SoM uses the same DOM-derived
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:105:`[SOM_MARKS]` text), so our hero claims (P-SoM ≥ best of DOM/SoM/Vision) are invariant to
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:127:This work compares B0 (Qwen3-VL-235B-A22B via proxy API) against B1 (Qwen3-VL-4B-Instruct
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:149:patching outputs are sensitive to floating-point matmul precision differences across CUDA
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:153:**Reproducibility statement**: Cross-machine numerical agreement on Qwen3-VL-4B between
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:156:does not flip top-1 logit comparisons; aggregate SR claims are unaffected.
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:163:(commit ≥ `3c15cd7`, dispatch + page_changed + cycle + RNG fixes deployed). Pre-Phase-A
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:165:For mechanistic Stage 2B/2C input artifacts, we use pre-Phase-A archived observations
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:166:(`results/mechanistic/archive_subset_b1_cls/`); per 笔记 §116 user-prompt analysis, agent
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:178:`B1_phantom_som_classifieds_20260428` archive (pre-Phase-A). Per 笔记 §116 user analysis:
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:179:the mechanistic claim is about model forward-pass behavior given a fixed input, not about
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:180:agent trajectory soundness. Phase A bugs in dispatch / cycle / RNG affect *which step* the
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:186:sensitivity check is in §5 Appendix and does not gate the main mechanism claim.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:1:# Method 4.2 v1 vs v2 cosine gap comparison — Bug 1+2 propagation audit
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:3:**Status**: Land 2026-05-12 late-late, after Myriad 359736 (cls v2) + 359737 (reddit v2) re-extraction with Bug 1 (tier filter) + Bug 2 (production `[SOM_MARKS]` format) + Bug 5 (model revision pin) fixes.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:7:**§5.7 "three-axis hierarchy with quantitatively distinct magnitudes" claim is INVALIDATED by v2 data.**
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:9:V1 numbers came from buggy NPZ where the SOM_MARKS regex `^\[\d+\]\s+\w+` extracted only 38 chars / 3 lines per task, dropping 71/72 marks. All Stage 4 cosine geometry was computed on near-empty text payloads where the only differentiator between flat-text modes (som / phantom_som / phantom_text) was prompt template. Modes still separated perfectly (AUROC 1.000) but the cosine-gap magnitudes were artifacts of prompt-template differences, not text-payload differences.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:11:V2 NPZ uses production `_extract_text_marks` (72-line full payload with `[id=N] {label}` envelope). Modes still separable (AUROC 1.000), but axis-1 + axis-2 cosine magnitudes collapse to noise level. Image-axis magnitudes preserve.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:17:| DOM ↔ Vision (image axis) | L04 0.0653 | L04 0.0670 | unchanged | unchanged |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:20:| P-SoM ↔ Vision (image axis) | L36 0.0613 | **L04** 0.0599 | unchanged | **earlier** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:21:| DOM ↔ SoM (image axis) | L04 0.0604 | **L36** 0.0496 | -18% | **boundary-shift** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:22:| P-prompt ↔ SoM (image axis) | L04 0.0600 | **L36** 0.0439 | -27% | **boundary-shift** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:23:| P-text ↔ SoM (image axis) | L20 0.0494 | **L36** 0.0488 | -1% | boundary-shift |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:24:| **P-SoM ↔ SoM (image axis, paper §5.7 image-axis anchor)** | **L17** 0.0412 | **L36** 0.0416 | unchanged | **L17 → L36** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:25:| DOM ↔ P-SoM | L23 0.0321 | **L36** 0.0152 | **-53%** | L23 → L36 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:26:| P-prompt ↔ P-SoM (axis-1 SoM-prompt) | L23 0.0292 | **L36** 0.0048 | **-84%** | L23 → L36 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:28:| **DOM ↔ P-text (axis-1 DOM-prompt, paper §5.7 axis-1 anchor)** | **L23** 0.0254 | **L36** 0.0047 | **-81%** | L23 → L36 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:29:| SoM ↔ Vision | L22 0.0238 | **L36** 0.0255 | +7% | boundary-shift |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:30:| **P-text ↔ P-SoM (axis-2, paper §5.7 axis-2 anchor)** | L23 0.0114 | **L36** 0.0088 | -23% | L23 → L36 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:31:| DOM ↔ P-prompt | L36 0.0067 | L36 0.0068 | unchanged | unchanged |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:35:| Ratio | v1 (3:1 ratio claim) | v2 (reality) |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:37:| Image axis magnitude (P-SoM↔SoM) | 0.041 | 0.042 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:38:| Text-format axis (DOM↔P-text) | 0.025 | **0.005** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:39:| Prompt-family axis (P-text↔P-SoM) | 0.011 | 0.009 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:40:| Image / text-format ratio | **1.7x** | **8x** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:42:| Text-format / prompt-family ratio | **2.3x** | **0.5x** ← axis-1 NOW SMALLER than axis-2 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:44:The "image > text-format > prompt-family" hierarchy with 4:3:1-ish quantitative ratio (v1) is **wrong**. V2 reality: image axis dominates by ~5-10x; axis-1 is **smaller than** axis-2 (reversed ranking); both axis-1 and axis-2 are noise-level (<0.01 cosine).
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:46:## L17 cosine gap snapshot (cls + reddit cross-site)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:48:| Mode pair | cls v1 | cls v2 | reddit v1 | reddit v2 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:50:| DOM ↔ P-text | 0.0120 | **0.0021** | (similar) | **0.0019** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:51:| DOM ↔ P-SoM | 0.0124 | **0.0029** | (similar) | **0.0031** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:53:| P-text ↔ P-SoM (axis-2) | 0.0028 | 0.0019 | — | 0.0020 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:54:| DOM ↔ SoM (image axis) | 0.0557 | 0.0452 | — | 0.0450 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:55:| DOM ↔ Vision (image axis) | 0.0545 | 0.0571 | — | 0.0537 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:59:## AUROC lototask (held-out, paper-grade Bug 3 fix)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:61:All pairs at all layers report AUROC lototask = 1.000 (perfect held-out linear separability). The modes ARE distinguishable in residual stream; the **magnitude of the mode-mean difference** is just much smaller than v1 claimed.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:63:This is the key reframe: **separability survives, magnitude does not**. Cosine gap measures effect SIZE; AUROC measures CLASSIFICATION RELIABILITY. They can dissociate.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:68:> "Three quantitatively distinct axes: image axis L17 0.041, text-format L23 0.029, prompt-family L23 0.011, with 4:3:1 magnitude ratio that holds cross-site."
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:71:> "All three axes are linearly readable in residual stream (held-out AUROC 1.000 across cls and reddit). The image axis dominates geometrically (~0.04-0.07 cosine peak) and emerges by L04. Text-format and prompt-family axes produce sub-permille mean-difference (cosine ~0.005-0.009) without a localized layer peak (monotone rise to boundary L36). The geometric magnitude rank-order reverses cross-site at L17 (axis-1 ≤ axis-2), indicating these axes are noise-level rather than quantitatively distinct dimensions."
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:73:**§5.2 Method 4.2** (cosine gap table at L17):
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:75:- L17 ceases to be a meaningful "disruption locus" for text-format / prompt-family axes — they peak at L36 (boundary monotone)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:77:**§5.5 image-axis peak-layer dichotomy** (paper claims "no-image side's text format predicts peak layer with zero overlap"):
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:79:- v2 reorganization: DOM/P-prompt ↔ Vision still L04; **P-text/P-SoM ↔ Vision shifted from L36 → L04** (BREAKS dichotomy); DOM/P-prompt/P-text/P-SoM ↔ SoM ALL at L36 now (collapses dichotomy on SoM image side)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:82:**§5.4 Stage 2/3 patching** (Cell A-H/D-G/H-text/H-prompt/H-d/Exp 5):
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:84:- All Stage 2/3 patching results **REMAIN VALID**
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:85:- Exp 5 cellhprompt cls + red axis-2 patching (80-125% capture of combined image+prompt displacement): **INTACT**
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:86:- Mid-layer L11-L17 patching effect: **INTACT**
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:88:**§5.3 Method 4.4 steering** (45-cell layer-α sweep):
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:92:- Per-task fragility (uses Stage 4 NPZ): NEEDS RE-RUN on v2
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:93:- Selection-bias (reverse-tier H1): uses format variation NPZ, separately audited (INTACT but baseline caveat)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:94:- Cross-site H1: format variation (INTACT)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:97:**Exp 1 axis-2 layer profile** (`axis2_layer_profile.md`): NEEDS RE-RUN on v2; current 4:3:1 ratio claim is invalidated.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:99:**Exp 3 logit lens** (`axis2_logit_lens.md`): NEEDS RE-RUN on v2. Output-space KL magnitudes likely also change (the lm_head amplifies whatever residual-stream signal is there).
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:103:✅ §1 hero (drop-one CI strict-positive reddit + cls): unchanged
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:105:✅ §4.5 reddit behavioral: unchanged
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:106:✅ §5.4 Stage 2/3 patching + Exp 5 axis-2 causal: unchanged
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:107:✅ §5.3 Method 4.4 steering: unchanged
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:109:✅ Held-out AUROC 1.000 linear-readability: unchanged
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:111:## New cleaner mechanism story
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:113:> **Three claim layers, distinct evidence types**:
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:114:> 1. **Linear readability** (Method 4.2 lototask AUROC 1.000): all 6 modes linearly separable in residual stream; small cosine magnitudes but reliable classification
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:115:> 2. **Geometric magnitude is mostly image-axis driven** (Method 4.2 cosine peak): image presence produces ~0.04-0.07 cosine separation early L04; other axes produce sub-permille separation throughout
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:116:> 3. **Causal patching effect at mid-layer L11-L17** (Stage 2/3 + Exp 5): 20-30% target-overlap displacement under final-token replacement patching, content-specific (task-shuffled control 359768/359769 in flight)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:118:> The disjoint between **small geometric magnitude (0.005-0.009)** and **large causal patching effect (0.20-0.30)** is the new headline claim — it means residual-stream geometry underestimates causal influence by orders of magnitude. This is paper-grade-novel and reviewer-defensible.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:122:- v2 NPZ: `results/mechanistic/stage4_multimode_b1_{cls,reddit}/hidden_states_v2_fixed.npz`
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:123:- v2 provenance: `hidden_states_v2_fixed.provenance.json` (git SHA, model revision, task IDs, formatter hash)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:124:- v2 metrics: `/tmp/method42_{cls,red}_v2.{json,md}` (canonical paths after rerun: `results/mechanistic/stage4_multimode_b1_*/method42_metrics.json`)
docs/checkpoints/paper_drafts/section1_intro.md:3:Web agents act through representations. A browser state can be serialized as a DOM or Accessibility Tree, shown as a screenshot, or annotated with Set-of-Mark (SoM) labels that connect visible regions to discrete element IDs. Existing benchmarks and agents treat these as different observation modes: WebArena and Mind2Web popularized DOM-derived text for realistic web tasks, while VisualWebArena and SeeAct introduced visually grounded settings where screenshots and action grounding become central \citep{zhou2024webarena,deng2023mind2web,koh2024visualwebarena,zheng2024seeact}. Set-of-Mark prompting was designed for this multimodal setting: a marked image is paired with a textual legend so the model can refer to visual objects by number \citep{yang2023som}. Later multimodal-agent systems, including SeeAct and Magma, further explored marked-screenshot and omni-modal action-grounding paradigms rather than treating mark text as an isolated scientific variable \citep{zheng2024seeact,yang2025magma}. These are important precedents. We therefore do not claim to be first to deploy text-only, marked, or SoM-style observations. Our claim is about controlled characterization: isolating what changes when the annotated image is skipped while the text payload and prompt family are varied under the same task, model, and evaluation protocol.
docs/checkpoints/paper_drafts/section1_intro.md:5:This paper questions that bundling assumption as an experimental object. We characterize the **phantom routing space**: configurations on the "skip annotated image" boundary that retain some SoM-derived textual or prompt structure while removing the image. Its deployment-relevant representative is **Phantom-SoM**: the agent receives the SoM prompt and the `[SOM_MARKS]` textual element list, but no image. The structural controls are **P-text** (the `[SOM_MARKS]` text under the DOM prompt) and **P-prompt** (the SoM prompt over AXTree text). At the start of this project, Phantom-SoM looked like a broken ablation. The natural expectation was that removing the marked screenshot would collapse SoM into either a weak DOM surrogate or a nonsensical configuration: the prompt still suggests visual marks, but the visual substrate is absent. The data reject that expectation. Phantom-SoM solves tasks that DOM, full SoM, and Vision all miss, and on B0 reddit it matches or modestly exceeds full SoM by adjusted SR (**13.81% vs 10.48%, N=210**; the gap is within 2σ under the run-to-run variability we observe in same-condition repeats), while avoiding image-token cost. On classifieds, full SoM remains clearly stronger (**21.37% vs Phantom-SoM 14.53%, N=234**), the expected sanity check when marked screenshots carry real visual information.
docs/checkpoints/paper_drafts/section1_intro.md:7:Our first contribution is a controlled scientific evaluation of this phantom boundary. Across completed B0 VisualWebArena classifieds and reddit runs, we compare DOM, full SoM, Vision, and Phantom-SoM on the same task sets (**N=234 classifieds; N=210 reddit; same-task adjusted SR**) and use the P-text/P-prompt controls to test whether the effect collapses to one prompt trick or one text-format swap. Phantom-SoM is not the best single arm on every site, and we do not claim that it replaces full SoM. Its value is complementarity. Its task-success pool has low overlap with the established modes, with Jaccard similarity in the roughly **0.29-0.49** range against other arms, and its removal reduces the oracle. The principal hero metric is therefore the **drop-one oracle**, not the single-mode SR difference: Phantom-SoM contributes **3.33 percentage points** of incremental oracle value on reddit with a per-task-bootstrap 95% CI of **[+0.95, +6.19]** strictly above zero (P(Δ>0)=0.998, B=10000 task resamples), comparable to full SoM at +1.90pp [+0.48, +3.81], and **2.56 percentage points** on classifieds with CI [+0.85, +4.70] strict positive. Phantom-SoM consistently sits within the top routing-value arms despite using no image. The same bootstrap on the head-to-head reddit single-mode comparison (Phantom-SoM 13.81% vs full SoM 10.48%) gives a marginal CI [-0.95, +7.62] that crosses zero (P(diff>0)=0.914), which is exactly the "within 2σ" caveat above; we therefore frame the head-to-head SR contrast as competitive parity, and let the strictly-positive drop-one oracle carry the deployment-relevant claim. Crucially, the cost of obtaining this configuration is essentially the cost of the DOM baseline: the `[SOM_MARKS]` block is produced by a regex pass over the same accessibility-tree text the DOM agent already consumes (interactive elements come pre-numbered as `[N] role 'label'`), so a deployment that can run DOM can run Phantom-SoM by changing what it forwards to the model: no bounding-box pipeline, no marked image, no extra inference modality. We therefore preserve the empirical **4-fold drop-in property** as the paper's practical finding: cost approximately DOM, lower image-stage latency, usable routing signal, and positive drop-one oracle value.
docs/checkpoints/paper_drafts/section1_intro.md:9:The cross-site asymmetry is itself informative: full SoM clearly outperforms Phantom-SoM on classifieds, where visually rich product listings make layout and appearance important, but not on reddit, where post and comment threads are more text-dominated. This is consistent with marked screenshots being most useful when visual grounding is task-critical.
docs/checkpoints/paper_drafts/section1_intro.md:11:Our second contribution is a mechanism account for why the ablation works. A 2-by-2 reddit ablation separates text representation from prompt family: DOM prompt versus SoM prompt, crossed with AXTree versus `[SOM_MARKS]` text. On the verified same-task subset (**N=48**), replacing AXTree with `[SOM_MARKS]` shifts exploration away from DOM-like search loops and toward Phantom-SoM-like quick decisions: the search-loop rate is **22.7% for DOM** but **10.8% for both Phantom-SoM and P-text**. The prompt knob appears elsewhere. DOM-prompt arms show the larger raw-to-adjusted false-positive gap (**6.25 pp; 3 N/A false positives**), while Phantom-SoM under the SoM prompt has a smaller gap (**2.08 pp; 1 N/A false positive**). Section 5 adds mid-layer mechanistic evidence: activation-patching effects concentrate in the L11-L17 region, and the random-injection control distinguishes content-specific phantom information from generic perturbation. The resulting two-knob view is simple: **text representation shapes how the agent explores; prompt wording tunes when it commits**. This aligns with prior evidence that language models are highly sensitive to prompt format \citep{sclar2024promptformat,mishra2022reframing}, but extends the claim from static prediction to multi-step web-agent trajectories.
docs/checkpoints/paper_drafts/section1_intro.md:13:We evaluate on VisualWebArena classifieds and reddit with B0, an API-backed Qwen3-VL-235B agent, and use B1, a local Qwen3-VL-4B model, as a cross-capability robustness check rather than a separate scientific claim. Following the pre-registered R1-R5 framing rules, the scope of this paper is Qwen-family VisualWebArena characterization and explanation, not a claim of universal model-family generalization and not yet a learned deployment router. Routing exploitation is deferred to a follow-up paper. Section 2 situates the gap in web-agent representations, SoM, routing, and prompt-format sensitivity. Section 3 defines Phantom-SoM and the measurement protocol. Section 4 reports phantom-space and image-on baseline findings. Section 5 analyzes the mid-layer activation-patching mechanism. Section 6 discusses generalization, and Section 7 summarizes limitations and implications.
docs/checkpoints/mechanism/results/layer_axis_emergence.md:3:Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:
docs/checkpoints/mechanism/results/layer_axis_emergence.md:5:| no-image side | image side | no-img text | peak layer | peak cosine gap |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:7:| DOM | SoM | AXTree | **L04** | 0.0604 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:8:| DOM | Vision | AXTree | **L04** | 0.0653 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:9:| P-prompt | SoM | AXTree | **L04** | 0.0600 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:11:| P-SoM | SoM | [SOM_MARKS] | **L17** | 0.0412 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:12:| P-text | SoM | [SOM_MARKS] | **L20** | 0.0494 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:14:| P-SoM | Vision | [SOM_MARKS] | **L36** | 0.0613 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:16:## Grouped by no-image side text format
docs/checkpoints/mechanism/results/layer_axis_emergence.md:20:- DOM ↔ SoM: peak **L04** = 0.0604
docs/checkpoints/mechanism/results/layer_axis_emergence.md:21:- DOM ↔ Vision: peak **L04** = 0.0653
docs/checkpoints/mechanism/results/layer_axis_emergence.md:22:- P-prompt ↔ SoM: peak **L04** = 0.0600
docs/checkpoints/mechanism/results/layer_axis_emergence.md:27:- P-text ↔ SoM: peak **L20** = 0.0494
docs/checkpoints/mechanism/results/layer_axis_emergence.md:29:- P-SoM ↔ SoM: peak **L17** = 0.0412
docs/checkpoints/mechanism/results/layer_axis_emergence.md:30:- P-SoM ↔ Vision: peak **L36** = 0.0613
docs/checkpoints/mechanism/results/layer_axis_emergence.md:34:When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).
docs/checkpoints/mechanism/results/layer_axis_emergence.md:36:When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.
docs/checkpoints/mechanism/results/layer_axis_emergence.md:42:> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*
docs/checkpoints/paper_drafts/paper.bib:1:% Bibliography for Paper 1: "Phantom-SoM: Isolated Set-of-Mark Text as a Hidden Routing Arm in Web Agents".
docs/checkpoints/paper_drafts/paper.bib:42:  booktitle = {Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
docs/checkpoints/paper_drafts/paper.bib:91:@inproceedings{sclar2024promptformat,
docs/checkpoints/paper_drafts/paper.bib:92:  title = {Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting},
docs/checkpoints/paper_drafts/paper.bib:139:  author = {Kerboua, Imene and Shayegan, Sahar Omidi and Thakkar, Megh and L{\`u}, Xing Han and Boisvert, L{\'e}o and Caccia, Massimo and Espinas, J{\'e}r{\'e}my and Aussem, Alexandre and Eglin, V{\'e}ronique and Lacoste, Alexandre},
docs/checkpoints/paper_drafts/paper.bib:265:% ---- Q2 system prompt format multi-step ----
docs/checkpoints/paper_drafts/paper.bib:318:  author={Drouin, Alexandre and Gasse, Maxime and Caccia, Massimo and Laradji, Issam H. and Del Verme, Manuel and Marty, Tom and Boisvert, L{\'e}o and Thakkar, Megh and Cappart, Quentin and Vazquez, David and Chapados, Nicolas and Lacoste, Alexandre},
docs/checkpoints/paper_drafts/paper.bib:349:  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
docs/checkpoints/paper_drafts/paper.bib:522:  note = {Multi-faceted hallucination = visual attention decay in middle layers + language prior dominance during decoding (paper §5 mechanism dual-cause anchor)}
docs/checkpoints/paper_drafts/paper.bib:533:  note = {Mean-difference activation steering at second-to-last layer switches tool selection at 77--100\% accuracy (93--100\% at 4B+) across 12 instruction-tuned models including Qwen 3 4B (architectural cousin to our B1 Qwen3-VL-4B). PCA: 15 tool means fit in $\sim$10 directions (91\% var). Causal localization on Gemma 3 4B: L17 H0+H1 attention heads dominate. Cited in our paper \S 5 Zoom 4 as the method backbone our Method 4.2/4.4 (multimodal Qwen3-VL-4B web agent extension) ports from.}
docs/checkpoints/paper_drafts/paper.bib:544:  note = {Decomposes total uncertainty into reducible + irreducible via higher-order predictors (Ahdritz et al. 2025). Routes to oracle when reducible uncertainty is high, abstains when irreducible. Strong theoretical regret bound vs optimal task-specific routers. Provides theoretical anchor for our paper \S 6 cost-aware routing claim: phantom mode signal AUROC = reducible uncertainty proxy; 4-fold drop-in property maps onto Peale et al.'s "predict / route / abstain" trichotomy. Stronger than purely empirical baselines (RouteLLM, FrugalGPT) because of regret bound.}
docs/checkpoints/paper_drafts/paper.bib:555:  note = {Proposes HDMI (Hidden-state Driven Margin Intervention): probe-free gradient-based hidden state steering using margin objective on model output distribution. LA-HDMI variant backpropagates through softmax embeddings for multi-step text editing. Evaluation framework introduces completeness (target property changes) $\times$ selectivity (unrelated properties preserved) $\to$ harmonic mean reliability. Validated on Meta-Llama-3-8B-Instruct + Pythia-70M on LGD agreement corpus + CausalGym benchmark. Cited in our paper \S 5 as (a) alternative-to-mean-diff steering method (Wu et al. 2026 backbone), and (b) source of evaluation framework for Method 4.4 v2 (completeness $\times$ selectivity harmonic mean is what we report, not crude "shifted toward target" rate).}
docs/checkpoints/paper_drafts/paper.bib:566:  note = {Position paper: MI papers invoke causal vocabulary (circuits, mediators, causal abstraction, monosemanticity) without disclosing identification assumptions. Audit of 10 papers across 4 methodological strands finds 0 dedicated identification sections; validation metrics (faithfulness, completeness, ablation effects) routinely substituted for identification. Proposes 5-step disclosure norm: (1) state whether claim is causal, (2) name identification strategy, (3) enumerate assumptions, (4) stress-test at least one, (5) separate validation from identification. Cited in our paper \S 5 to structure the "Identification Assumptions" subsection. Our methodology adopts the protocol explicitly: Method 4.2 (probe) and Stage 2/3 (patching) are paired as identification strategy, with assumptions listed and Cell E random-injection control as stress-test.}
docs/checkpoints/paper_drafts/paper.bib:577:  note = {Establishes probe-vs-causal-use distinction on rhyme couplet planning. Linear probe identifies planning-compatible representations at newline position across model families (Qwen3, Gemma 3, Llama 3.1, 270M--70B); activation patching shows only Gemma-3-27B causally uses newline as planning site (corrupt-rhyme rate 67\% [57, 75] at L33), while Qwen3-32B (1\%) and Llama-3.1-70B (2\%) remain at original rhyme word. Sparse 5-attention-head localization in Gemma-3-27B. Cited in our paper \S 5 as theoretical anchor for distinguishing Method 4.2 (planning-compatible representation, AUROC 1.000 across all 540 layer-pair tests) from Method 4.4 (causally active planning site, partial). Our Qwen3-VL-4B Method 4.4 results align with the Qwen3-family pattern (probe works, causal patching weaker than Gemma).}
docs/checkpoints/paper_drafts/paper.bib:580:@misc{fayyaz2026steermoe,
docs/checkpoints/paper_drafts/paper.bib:584:  note = {ICLR 2026 (Anonymous in submission; deanon TBD). Paired-prompt expert Risk Difference (RD) score $\Delta_{\ell,i}$ identifies behavior-linked experts in MoE LLMs (e.g., Qwen3-30B-A3B, Mixtral, DeepSeek-V2). Inference-time router-logit adjustment activates/deactivates expert subsets: Faithfulness +27\%, Safety +20\%, Unsafe steering -41\%; combined with AIM jailbreak takes GPT-OSS-120B safety from 100\% to 0\%. Reveals ``Alignment Faking'' --- alignment concentrated in expert subsets, alternate routing path bypasses. Cited in our paper \S 5 as Zoom 4 mechanism layer anchor: B0 (Qwen3-VL-235B-A22B MoE) is architectural cousin of SteerMoE's Qwen3-30B-A3B; methodology template for paper-2 self-probe future work. We do not self-probe in paper-1 because proxy API conceals router logits and local deploy of 235B-A22B exceeds DGX budget. NEEDS\_VERIFY: arxiv ID + full author list.},
docs/checkpoints/paper_drafts/paper.bib:595:  note = {Foundational activation-patching protocol: IOI circuit identification in GPT-2 Small via clean/corrupt prompt pairs + per-layer hidden-state substitution. Cited in our paper \S 5.4 as the methodological grandparent of Stage 2/3 mechanism patching protocol (\texttt{p79/mechanistic/activation\_patching.py}). Our Source = SoM (with image, clean) / Target = Phantom-SoM (no image, mirage) replicates the IOI clean/corrupt logic for multimodal observation modes.},
docs/checkpoints/paper_drafts/paper.bib:598:@misc{zhang2024patching,
docs/checkpoints/paper_drafts/paper.bib:605:  note = {Methodological survey of activation patching protocols. Cited in our paper \S 5.4 alongside Wang et al. 2023 as the protocol foundation for Stage 2/3 patching. NEEDS\_VERIFY: exact author list and year. If the intended reference is Heimersheim \& Nanda 2024 ``How to use and interpret activation patching'' [arXiv:2404.15255] the bibkey should be renamed accordingly.},
docs/checkpoints/paper_drafts/paper.bib:629:  note = {Originally posted as arXiv:1807.03341 (2018). Influential critique of explanation vs causation conflation, anecdote-as-evidence, mathiness, and misuse of language in ML papers. Cited in our paper \S 4 limitations disclosure and \S 5.7 hero-status framing (we explicitly avoid claiming Phantom-SoM hero status is explained by mid-layer mechanism alone; the three-axis hierarchy and prompt-prior dissociation are reported as joint contributors, not a unitary cause).},
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:1:# H1 per-task fragility check
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:3:**Sample**: 45 (task, step) pairs from format_variation_b1_cls
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:7:- **AXTree-DOM peak ≤ L10** (early image-axis peak): 9/45 = **20%**
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:13:AXTree-DOM peak layer: mean = **27.9**, std = 13.1, range L04-L36
docs/checkpoints/paper_drafts/section3_definition.md:1:## 3. Phantom-SoM: Definition and Ablation Setup
docs/checkpoints/paper_drafts/section3_definition.md:5:Set-of-Mark (SoM) prompting converts a screenshot into an indexed visual interface. The standard bundle has two synchronized parts: a marked image, where page regions are overlaid with bounding boxes and numeric IDs, and a text legend that maps those IDs to element descriptions [Yang et al. 2023]. We serialize the text component as:
docs/checkpoints/paper_drafts/section3_definition.md:14:Full SoM gives both pieces to the agent at the same step. The prompt says the `[SOM_MARKS]` list and annotated screenshot refer to one another, and the action schema asks the model to click, type, or select by `element_id` when possible. VisualWebArena and SeeAct use the same broad pattern: visual evidence is paired with grounding information so the model can convert perception into browser actions [Koh et al. 2024; Zheng et al. 2024].
docs/checkpoints/paper_drafts/section3_definition.md:16:This bundle is the assumption Phantom-SoM ablates. The question is not whether marked screenshots are useful; Section 4 shows that they often are. The question is whether the text half of the bundle is only an image key, or itself a distinct text representation.
docs/checkpoints/paper_drafts/section3_definition.md:18:### 3.2 Phantom-SoM
docs/checkpoints/paper_drafts/section3_definition.md:20:We define **Phantom-SoM** as:
docs/checkpoints/paper_drafts/section3_definition.md:23:Phantom-SoM(page) =
docs/checkpoints/paper_drafts/section3_definition.md:24:  prompt = SoM prompt
docs/checkpoints/paper_drafts/section3_definition.md:29:Phantom-SoM uses the same SoM prompt family as full SoM and the same `[SOM_MARKS]` text, but removes the page screenshot passed to the model. In code, `p79/experiment/som.py::prepare_observation_for_mode` handles `mode in ("phantom_som", "phantom_dom", "phantom_text")` by calling `_build_som_result(...)`, then returning the generated `som_text` with `marked_image=None` (`phantom_dom` is the legacy mode value retained as alias for paper-grade run dirs; `phantom_text` is the current canonical name for P-text). The rendered screenshot path is retained for debugging; the model does not receive it.
docs/checkpoints/paper_drafts/section3_definition.md:31:The critical property is that the prompt remains the SoM prompt. It still describes an annotated screenshot with numbered boxes, even though the observation channel contains no page screenshot. We call this the **mirage prompt** property: the behavioral scaffold of SoM is preserved while the visual substrate is removed.
docs/checkpoints/paper_drafts/section3_definition.md:33:Phantom-SoM is a cost intervention, and the structure of the saving is best stated relative to two different baselines.
docs/checkpoints/paper_drafts/section3_definition.md:35:**Relative to DOM**, Phantom-SoM is essentially free. The `[SOM_MARKS]` block is produced by a regex filter over the VisualWebArena accessibility-tree text that the DOM baseline already consumes. VWA serializes interactive elements with bracketed numeric IDs of the form `[N] role 'label'`; in our implementation `_extract_text_marks` (see `p79/experiment/som.py`) walks `obs_text` line by line, keeps the lines that match `\[\d+\]`, and returns `(id, label)` pairs that are wrapped in a `[SOM_MARKS] ... [/SOM_MARKS]` block. There is no bounding-box lookup and no image work in this path; bounding boxes are only used by full SoM when drawing numeric labels onto the screenshot. Empirically this leaves text length roughly unchanged: holding the system prompt fixed at the DOM family, median total input is 3437 tokens for DOM versus 3661 for P-text on reddit, and 3008 versus 2948 on classifieds — within ±7% on both sites. The two formats see the same accessibility content; what differs is the surface form (flat indexed list versus nested hierarchy with url/tab metadata). We treat this as a representation property and study its behavioral effect mechanistically in Section 5; for cost accounting the implication is that switching DOM → Phantom-SoM at deployment time costs at most a regex pass over the same observation.
docs/checkpoints/paper_drafts/section3_definition.md:37:**Relative to full SoM**, Phantom-SoM saves two real layers of cost. (i) The on-server annotation step that draws numeric labels onto the page screenshot is unique to full SoM and is omitted in a Phantom-SoM deployment; in our research code we retain the marked image on disk for debugging, which is why both modes report ~30 ms median obs-prepare latency, but a production variant skips the draw entirely and recovers roughly 30 ms and on the order of $2e-5 per step. (ii) The marked screenshot is no longer encoded as image tokens at inference, removing the visual-encoding stage. Comparing step-level `tokens.input` medians between full SoM and P-text gives a same-prompt image-channel estimate of 733 input tokens per step on reddit (SoM 4275 versus P-text 3542; P-text partial live run, 145 episodes) and 1064 on classifieds (4034.5 versus 2970.5; 234 episodes). We attribute this median gap to the marked screenshot under our backend tokenization. These are the tokens that drive prompt-processing time, memory pressure, and time-to-first-token in multimodal serving (see Section 2.4); skipping them is the dominant component of the cost difference between full SoM and Phantom-SoM.
docs/checkpoints/paper_drafts/section3_definition.md:39:The combined picture is that Phantom-SoM sits at roughly DOM cost (its observation is a text filter of the same AXTree) while replacing the visual-evidence half of SoM with nothing at all. This is also a deployment-level claim, not only an analytical one: an existing full-SoM agent can be converted into a Phantom-SoM agent by changing only what the server forwards to the model — keep the `[SOM_MARKS]` text that is already being produced from the accessibility tree, stop drawing labels onto the screenshot, and stop attaching the marked image to the inference request. The model interface, the prompt, the action schema, and the evaluator are unchanged. There is no retraining, no new data path, and no marks-side prompt edit; the only mutation is on the backend annotation pipeline, after the AXTree filter and before the model call. We use this property in Section 4 to interpret cost-versus-success comparisons as deployment-time tradeoffs rather than research-only configurations, and in Section 5 to argue that Phantom-SoM's behavior is a property of the format the model already saw inside SoM, not an emergent capability that requires new infrastructure.
docs/checkpoints/paper_drafts/section3_definition.md:47:  prompt = DOM prompt
docs/checkpoints/paper_drafts/section3_definition.md:52:Its observation is identical to Phantom-SoM: `[SOM_MARKS]` text only, no page screenshot. The only intended change is the system prompt. In both B0 (`p79/agents/proxy_api_agent.py`) and B1 (`p79/agents/qwen3vl_agent.py`), `_system_prompts["phantom_som"]` maps to the SoM prompt, while `_system_prompts["phantom_dom"]` (and the alias `_system_prompts["phantom_text"]`) maps to the DOM prompt. For `som`, `phantom_som`, `phantom_dom`, and `phantom_text`, the agent passes through the `[SOM_MARKS]...[/SOM_MARKS]` text directly.
docs/checkpoints/paper_drafts/section3_definition.md:54:This cell separates representation from prompt wording. If P-text behaves like Phantom-SoM, the flat marks text is driving behavior. If it behaves like DOM, the prompt is doing more of the work.
docs/checkpoints/paper_drafts/section3_definition.md:60:| | DOM prompt | SoM prompt |
docs/checkpoints/paper_drafts/section3_definition.md:62:| AXTree obs | DOM | *excluded — see below* |
docs/checkpoints/paper_drafts/section3_definition.md:63:| `[SOM_MARKS]` obs | P-text | Phantom-SoM |
docs/checkpoints/paper_drafts/section3_definition.md:65:Full SoM is adjacent to this 2x2: it uses the SoM prompt, the same `[SOM_MARKS]` text, and the marked screenshot. Vision is a separate screenshot-only baseline.
docs/checkpoints/paper_drafts/section3_definition.md:67:The fourth cell — AXTree observation paired with the SoM prompt — is intentionally excluded from Paper 1 because it is not a self-consistent design point. The SoM system prompt instructs the agent to interact via `[SOM_MARKS]` IDs (e.g. `click [42]` referring to the SoM-marked element 42), but AXTree text uses an independent accessibility-tree ID space; an action like `click [42]` becomes parsing-ambiguous when the two ID systems do not match. This hybrid mode (i) has no clean LLM mechanism, (ii) confounds the prompt-effect ablation with mismatched-ID parsing failure, and (iii) does not reduce token cost relative to P-text. We treat the 5-mode set (DOM, P-text, Phantom-SoM, full SoM, plus Vision as a separate screenshot-only arm) as the diagonal axis-by-axis path through the 2×2×2 (text-payload-structure × prompt × image) design cube; the four mismatched-prompt-representation hybrids are excluded for the same reason.
docs/checkpoints/paper_drafts/section3_definition.md:71:- **DOM vs P-text** holds the prompt family fixed at DOM and changes the text-payload structure from AXTree to `[SOM_MARKS]`.
docs/checkpoints/paper_drafts/section3_definition.md:72:- **Phantom-SoM vs P-text** holds the text observation fixed and changes only the prompt family.
docs/checkpoints/paper_drafts/section3_definition.md:73:- **Full SoM vs Phantom-SoM** holds prompt and marks text fixed and adds the implemented marked-image channel.
docs/checkpoints/paper_drafts/section3_definition.md:74:- **Full SoM vs DOM** measures the combined effect of SoM prompt, marks text, and marked screenshot relative to the standard text baseline.
docs/checkpoints/paper_drafts/section3_definition.md:76:The 2x2 is not a routing policy by itself. It is a causal scaffold for Section 5: text-payload structure shapes exploration, while prompt wording tunes commitment confidence. Section 6 promotes this scaffold to a token-monotonic cascade — DOM → P-text (axis 1, text-payload swap, no token increase) → Phantom-SoM (axis 2, system-prompt swap, no data-token increase) → full SoM (axis 3, image embedding cost) — so a routing trigger never has to "add then remove" tokens.
docs/checkpoints/paper_drafts/section3_definition.md:80:All SoM-derived conditions share the same text-marks extractor. `_extract_text_marks` reads `obs_text` (the VisualWebArena accessibility-tree serialization the DOM baseline already uses) line by line, keeps each line whose label matches `\[\d+\]`, and produces `(id, label)` pairs up to a configured cap. `_build_som_result` then wraps those pairs in a `[SOM_MARKS] ... [/SOM_MARKS]` block. This text path **does not require bounding boxes**: the IDs come from the accessibility tree, not from a separate vision pipeline. Bounding boxes are only consulted by full SoM, which uses `obs_nodes_info` to draw numeric labels onto the page screenshot. Phantom-SoM and P-text reuse the exact `[SOM_MARKS]` text and drop the page screenshot; Marks are not re-filtered specifically for Phantom, and the source page state is unchanged.
docs/checkpoints/paper_drafts/section3_definition.md:82:Reference images supplied by a task configuration are separate from the observation mode. These task-provided target images are passed to all modes as task input; Phantom-SoM removes only the current-page browser screenshot.
docs/checkpoints/paper_drafts/section3_definition.md:84:Each episode starts from `environment.reset(task.config_file)`, and paper-grade condition comparisons use freshly reset site state to avoid cross-condition contamination. The April 27 Magento base-url/auth fix addressed an unrelated shopping-state reliability issue; this paper uses completed classifieds and reddit runs under the reset protocol.
docs/checkpoints/paper_drafts/section3_definition.md:86:When comparing arms, we use same-task subsets: a task contributes only when the relevant conditions have completed it. We report **adjusted SR**, which starts from raw evaluator success and removes `na_fp` for not-applicable tasks that appear correct without agent-initiated finish, and `eval_fp` for evaluator matches caused by ineffective or non-finished trajectories. Section 4 reports results under these conventions; Section 5 uses the same traces for mechanism analysis.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:9:| Source | `phantom_som` (no image, flat `[SOM_MARKS]`, SoM prompt) | `som` (image, flat `[SOM_MARKS]`, SoM prompt) |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:10:| Target | `phantom_text` (no image, flat `[SOM_MARKS]`, DOM prompt) | `phantom_text` (same) |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:13:| Layers | 37 (L0-L36, Qwen3-VL-4B language decoder) | matching |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:15:**Test logic**: Holding both `image` and `text-format` constant (off + flat) and patching source hidden states from `phantom_som` into a `phantom_text` run isolates whether the residual-stream prompt-family signature has *causal* effect on token continuation, not just *geometric* magnitude (which Exp 1 already showed is small at 0.011 cosine gap @ L23).
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:17:## Result — mid-layer (L11-L17) patching causal effect
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:26:(Baseline `overlap→tgt = 1.00` at L35 = full target preservation, no patching effect.)
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:42:Compared with Exp 1 cosine geometry, using best-layer values:
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:46:| Image (SoM ↔ P-SoM) | 0.041 @ L17 | ~0.04-0.05 (inferred from H-text − cellhprompt diff) |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:47:| Text-format (DOM ↔ P-text) | 0.029 @ L23 | (Exp H-d-cls/red, not directly compared here) |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:48:| **Prompt-family (P-SoM ↔ P-text)** | **0.011 @ L23** | **~0.20-0.30 @ L11-L17** |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:50:**4:3:1 cosine geometry ratio does NOT translate to 4:3:1 causal patching ratio.** Prompt-family has the **smallest** geometric magnitude but the **largest** causal patching weight.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:54:Critically, the **layer at which cosine peaks ≠ the layer at which patching has maximal effect** for prompt-family:
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:61:| **L23** | 0.96 | 0.89 | **cosine geometry peak, but patching weak** — representation stabilized |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:65:At **L23** (the cosine peak), patching displaces target output by only **0.04-0.11 overlap units** — much smaller than the **0.20-0.30** displacement at L11-L17.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:71:- **L23 is the prompt-family "signature layer"**: representation has stabilized to its most discriminable form (highest cosine separation between P-SoM and P-text). It reflects *what prompt was given* — a state variable.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:72:- **L11-L17 is the prompt-family "decision routing layer"**: patching here changes upstream signal that downstream layers consume to drive token continuation. It reflects *how the model uses the prompt* — a causal variable.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:74:Activation patching is path-dependent: an upstream patch propagates into all downstream computations, while a downstream patch leaves upstream inputs unchanged so subsequent layers can re-encode the same signal. This is consistent with standard mechanistic-interpretability findings (cf. \citep{wang2023interpretability} IOI circuit: feature *encoded* ≠ feature *used*).
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:78:1. Residual-stream cosine separation is a **necessary but not sufficient** signal of causal mechanism.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:79:2. Prompt-family information is **dispatchable** — small geometric perturbation at the decision layer produces large output displacement when patched.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:80:3. **Where a feature is most readable (L23) and where it is most consequential (L11-L17) are different layers** — paper-grade mechanism claims must report both, not collapse them.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:84:**Strengthens 3-axis mechanism story**:
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:85:- Axis-1 (text-format): Exp 1 cosine 0.029 + H-d cells causal patching (prior)
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:86:- Axis-2 (prompt-family): Exp 1 cosine 0.011 + **Exp 5 cellhprompt causal patching (this)**
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:87:- Axis-image: Exp 1 cosine 0.041 + indirect (H-text − cellhprompt residual ~0.04-0.05)
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:89:**Defuses /stress critique** "you only have axis-1 mechanism":
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:90:- Now have causal evidence for axis-2 separate from axis-1
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:93:**Reframes hero argument**: The paper §1 framing "text-format shapes exploration; prompt tunes commit" is now backed by:
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:94:- Behavioral: exploration rate axis-1 dependent (Exp 1 cosine sigma + §4.5 reddit behavioral)
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:95:- Causal mechanism: prompt-family mid-layer L11-L17 patching produces output displacement comparable to image-axis flip
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:99:- N=24 per cell — bootstrap CI on per-layer overlap means would tighten interpretation.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:101:- Patching displacement is a token-level metric; doesn't directly translate to SR / drop-one oracle. Behavioral consequence (which paper §1 hero is about) operates on top of this causal signal.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:106:- `patching_continuation_results.json`: per-layer per-task continuation strings + metrics (~1.3 MB each)
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:107:- `patching_continuation_curves.png`: visual layer profile
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:112:- Watcher missed GONE events due to silent-miss bug (PR same commit) — auto_pull dispatched manually
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:1:# Axis-2 per-task fragility check
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:3:Per-task cosine gap distribution at L23 (axis-2 peak per §5.7 / Exp 1).
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:4:Each task averaged across its 2 steps; cosine gap computed between mode pairs.
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:12:| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0132 | 0.0131 | [0.0124, 0.0142] | 0.0107 | 0.0174 | 100% | 100% | 0% |
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:13:| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0048 | 0.0047 | [0.0044, 0.0052] | 0.0039 | 0.0065 | 33% | 0% | 0% |
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:14:| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0287 | 0.0280 | [0.0250, 0.0312] | 0.0186 | 0.0456 | 100% | 100% | 92% |
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:15:| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0407 | 0.0415 | [0.0353, 0.0438] | 0.0308 | 0.0597 | 100% | 100% | 100% |
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:21:| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0121 | 0.0120 | [0.0113, 0.0127] | 0.0102 | 0.0152 | 100% | 100% | 0% |
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:22:| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0052 | 0.0051 | [0.0047, 0.0055] | 0.0039 | 0.0067 | 50% | 0% | 0% |
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:23:| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0260 | 0.0263 | [0.0226, 0.0305] | 0.0174 | 0.0344 | 100% | 100% | 83% |
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:24:| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0436 | 0.0439 | [0.0409, 0.0453] | 0.0382 | 0.0535 | 100% | 100% | 100% |
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:26:## Top 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:36:## Bottom 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:46:## Top 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:56:## Bottom 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:68:Read the `% > 0.010` column for the axis-2 P-text↔P-SoM pair:
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:70:- reddit: **100%** of 24 tasks above
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:75:- If both < 25% → axis-2 signal **aggregate artifact**, §5.7 three-axis claim must downgrade to 'axis-1 + image-axis with axis-2 weak per-task'
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:77:Median values: cls=0.0131, reddit=0.0120.
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:78:Compare to mean: cls=0.0132, reddit=0.0121.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section1_intro.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	## 1. Introduction
     2	
     3	Web agents act through representations. A browser state can be serialized as a DOM or Accessibility Tree, shown as a screenshot, or annotated with Set-of-Mark (SoM) labels that connect visible regions to discrete element IDs. Existing benchmarks and agents treat these as different observation modes: WebArena and Mind2Web popularized DOM-derived text for realistic web tasks, while VisualWebArena and SeeAct introduced visually grounded settings where screenshots and action grounding become central \citep{zhou2024webarena,deng2023mind2web,koh2024visualwebarena,zheng2024seeact}. Set-of-Mark prompting was designed for this multimodal setting: a marked image is paired with a textual legend so the model can refer to visual objects by number \citep{yang2023som}. Later multimodal-agent systems, including SeeAct and Magma, further explored marked-screenshot and omni-modal action-grounding paradigms rather than treating mark text as an isolated scientific variable \citep{zheng2024seeact,yang2025magma}. These are important precedents. We therefore do not claim to be first to deploy text-only, marked, or SoM-style observations. Our claim is about controlled characterization: isolating what changes when the annotated image is skipped while the text payload and prompt family are varied under the same task, model, and evaluation protocol.
     4	
     5	This paper questions that bundling assumption as an experimental object. We characterize the **phantom routing space**: configurations on the "skip annotated image" boundary that retain some SoM-derived textual or prompt structure while removing the image. Its deployment-relevant representative is **Phantom-SoM**: the agent receives the SoM prompt and the `[SOM_MARKS]` textual element list, but no image. The structural controls are **P-text** (the `[SOM_MARKS]` text under the DOM prompt) and **P-prompt** (the SoM prompt over AXTree text). At the start of this project, Phantom-SoM looked like a broken ablation. The natural expectation was that removing the marked screenshot would collapse SoM into either a weak DOM surrogate or a nonsensical configuration: the prompt still suggests visual marks, but the visual substrate is absent. The data reject that expectation. Phantom-SoM solves tasks that DOM, full SoM, and Vision all miss, and on B0 reddit it matches or modestly exceeds full SoM by adjusted SR (**13.81% vs 10.48%, N=210**; the gap is within 2σ under the run-to-run variability we observe in same-condition repeats), while avoiding image-token cost. On classifieds, full SoM remains clearly stronger (**21.37% vs Phantom-SoM 14.53%, N=234**), the expected sanity check when marked screenshots carry real visual information.
     6	
     7	Our first contribution is a controlled scientific evaluation of this phantom boundary. Across completed B0 VisualWebArena classifieds and reddit runs, we compare DOM, full SoM, Vision, and Phantom-SoM on the same task sets (**N=234 classifieds; N=210 reddit; same-task adjusted SR**) and use the P-text/P-prompt controls to test whether the effect collapses to one prompt trick or one text-format swap. Phantom-SoM is not the best single arm on every site, and we do not claim that it replaces full SoM. Its value is complementarity. Its task-success pool has low overlap with the established modes, with Jaccard similarity in the roughly **0.29-0.49** range against other arms, and its removal reduces the oracle. The principal hero metric is therefore the **drop-one oracle**, not the single-mode SR difference: Phantom-SoM contributes **3.33 percentage points** of incremental oracle value on reddit with a per-task-bootstrap 95% CI of **[+0.95, +6.19]** strictly above zero (P(Δ>0)=0.998, B=10000 task resamples), comparable to full SoM at +1.90pp [+0.48, +3.81], and **2.56 percentage points** on classifieds with CI [+0.85, +4.70] strict positive. Phantom-SoM consistently sits within the top routing-value arms despite using no image. The same bootstrap on the head-to-head reddit single-mode comparison (Phantom-SoM 13.81% vs full SoM 10.48%) gives a marginal CI [-0.95, +7.62] that crosses zero (P(diff>0)=0.914), which is exactly the "within 2σ" caveat above; we therefore frame the head-to-head SR contrast as competitive parity, and let the strictly-positive drop-one oracle carry the deployment-relevant claim. Crucially, the cost of obtaining this configuration is essentially the cost of the DOM baseline: the `[SOM_MARKS]` block is produced by a regex pass over the same accessibility-tree text the DOM agent already consumes (interactive elements come pre-numbered as `[N] role 'label'`), so a deployment that can run DOM can run Phantom-SoM by changing what it forwards to the model: no bounding-box pipeline, no marked image, no extra inference modality. We therefore preserve the empirical **4-fold drop-in property** as the paper's practical finding: cost approximately DOM, lower image-stage latency, usable routing signal, and positive drop-one oracle value.
     8	
     9	The cross-site asymmetry is itself informative: full SoM clearly outperforms Phantom-SoM on classifieds, where visually rich product listings make layout and appearance important, but not on reddit, where post and comment threads are more text-dominated. This is consistent with marked screenshots being most useful when visual grounding is task-critical.
    10	
    11	Our second contribution is a mechanism account for why the ablation works. A 2-by-2 reddit ablation separates text representation from prompt family: DOM prompt versus SoM prompt, crossed with AXTree versus `[SOM_MARKS]` text. On the verified same-task subset (**N=48**), replacing AXTree with `[SOM_MARKS]` shifts exploration away from DOM-like search loops and toward Phantom-SoM-like quick decisions: the search-loop rate is **22.7% for DOM** but **10.8% for both Phantom-SoM and P-text**. The prompt knob appears elsewhere. DOM-prompt arms show the larger raw-to-adjusted false-positive gap (**6.25 pp; 3 N/A false positives**), while Phantom-SoM under the SoM prompt has a smaller gap (**2.08 pp; 1 N/A false positive**). Section 5 adds mid-layer mechanistic evidence: activation-patching effects concentrate in the L11-L17 region, and the random-injection control distinguishes content-specific phantom information from generic perturbation. The resulting two-knob view is simple: **text representation shapes how the agent explores; prompt wording tunes when it commits**. This aligns with prior evidence that language models are highly sensitive to prompt format \citep{sclar2024promptformat,mishra2022reframing}, but extends the claim from static prediction to multi-step web-agent trajectories.
    12	
    13	We evaluate on VisualWebArena classifieds and reddit with B0, an API-backed Qwen3-VL-235B agent, and use B1, a local Qwen3-VL-4B model, as a cross-capability robustness check rather than a separate scientific claim. Following the pre-registered R1-R5 framing rules, the scope of this paper is Qwen-family VisualWebArena characterization and explanation, not a claim of universal model-family generalization and not yet a learned deployment router. Routing exploitation is deferred to a follow-up paper. Section 2 situates the gap in web-agent representations, SoM, routing, and prompt-format sensitivity. Section 3 defines Phantom-SoM and the measurement protocol. Section 4 reports phantom-space and image-on baseline findings. Section 5 analyzes the mid-layer activation-patching mechanism. Section 6 discusses generalization, and Section 7 summarizes limitations and implications.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section3_definition.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	## 3. Phantom-SoM: Definition and Ablation Setup
     2	
     3	### 3.1 Set-of-Mark Bundle
     4	
     5	Set-of-Mark (SoM) prompting converts a screenshot into an indexed visual interface. The standard bundle has two synchronized parts: a marked image, where page regions are overlaid with bounding boxes and numeric IDs, and a text legend that maps those IDs to element descriptions [Yang et al. 2023]. We serialize the text component as:
     6	
     7	```text
     8	[SOM_MARKS]
     9	[id=N] role 'label'
    10	...
    11	[/SOM_MARKS]
    12	```
    13	
    14	Full SoM gives both pieces to the agent at the same step. The prompt says the `[SOM_MARKS]` list and annotated screenshot refer to one another, and the action schema asks the model to click, type, or select by `element_id` when possible. VisualWebArena and SeeAct use the same broad pattern: visual evidence is paired with grounding information so the model can convert perception into browser actions [Koh et al. 2024; Zheng et al. 2024].
    15	
    16	This bundle is the assumption Phantom-SoM ablates. The question is not whether marked screenshots are useful; Section 4 shows that they often are. The question is whether the text half of the bundle is only an image key, or itself a distinct text representation.
    17	
    18	### 3.2 Phantom-SoM
    19	
    20	We define **Phantom-SoM** as:
    21	
    22	```text
    23	Phantom-SoM(page) =
    24	  prompt = SoM prompt
    25	  text   = SOM_MARKS(page)
    26	  image  = None
    27	```
    28	
    29	Phantom-SoM uses the same SoM prompt family as full SoM and the same `[SOM_MARKS]` text, but removes the page screenshot passed to the model. In code, `p79/experiment/som.py::prepare_observation_for_mode` handles `mode in ("phantom_som", "phantom_dom", "phantom_text")` by calling `_build_som_result(...)`, then returning the generated `som_text` with `marked_image=None` (`phantom_dom` is the legacy mode value retained as alias for paper-grade run dirs; `phantom_text` is the current canonical name for P-text). The rendered screenshot path is retained for debugging; the model does not receive it.
    30	
    31	The critical property is that the prompt remains the SoM prompt. It still describes an annotated screenshot with numbered boxes, even though the observation channel contains no page screenshot. We call this the **mirage prompt** property: the behavioral scaffold of SoM is preserved while the visual substrate is removed.
    32	
    33	Phantom-SoM is a cost intervention, and the structure of the saving is best stated relative to two different baselines.
    34	
    35	**Relative to DOM**, Phantom-SoM is essentially free. The `[SOM_MARKS]` block is produced by a regex filter over the VisualWebArena accessibility-tree text that the DOM baseline already consumes. VWA serializes interactive elements with bracketed numeric IDs of the form `[N] role 'label'`; in our implementation `_extract_text_marks` (see `p79/experiment/som.py`) walks `obs_text` line by line, keeps the lines that match `\[\d+\]`, and returns `(id, label)` pairs that are wrapped in a `[SOM_MARKS] ... [/SOM_MARKS]` block. There is no bounding-box lookup and no image work in this path; bounding boxes are only used by full SoM when drawing numeric labels onto the screenshot. Empirically this leaves text length roughly unchanged: holding the system prompt fixed at the DOM family, median total input is 3437 tokens for DOM versus 3661 for P-text on reddit, and 3008 versus 2948 on classifieds — within ±7% on both sites. The two formats see the same accessibility content; what differs is the surface form (flat indexed list versus nested hierarchy with url/tab metadata). We treat this as a representation property and study its behavioral effect mechanistically in Section 5; for cost accounting the implication is that switching DOM → Phantom-SoM at deployment time costs at most a regex pass over the same observation.
    36	
    37	**Relative to full SoM**, Phantom-SoM saves two real layers of cost. (i) The on-server annotation step that draws numeric labels onto the page screenshot is unique to full SoM and is omitted in a Phantom-SoM deployment; in our research code we retain the marked image on disk for debugging, which is why both modes report ~30 ms median obs-prepare latency, but a production variant skips the draw entirely and recovers roughly 30 ms and on the order of $2e-5 per step. (ii) The marked screenshot is no longer encoded as image tokens at inference, removing the visual-encoding stage. Comparing step-level `tokens.input` medians between full SoM and P-text gives a same-prompt image-channel estimate of 733 input tokens per step on reddit (SoM 4275 versus P-text 3542; P-text partial live run, 145 episodes) and 1064 on classifieds (4034.5 versus 2970.5; 234 episodes). We attribute this median gap to the marked screenshot under our backend tokenization. These are the tokens that drive prompt-processing time, memory pressure, and time-to-first-token in multimodal serving (see Section 2.4); skipping them is the dominant component of the cost difference between full SoM and Phantom-SoM.
    38	
    39	The combined picture is that Phantom-SoM sits at roughly DOM cost (its observation is a text filter of the same AXTree) while replacing the visual-evidence half of SoM with nothing at all. This is also a deployment-level claim, not only an analytical one: an existing full-SoM agent can be converted into a Phantom-SoM agent by changing only what the server forwards to the model — keep the `[SOM_MARKS]` text that is already being produced from the accessibility tree, stop drawing labels onto the screenshot, and stop attaching the marked image to the inference request. The model interface, the prompt, the action schema, and the evaluator are unchanged. There is no retraining, no new data path, and no marks-side prompt edit; the only mutation is on the backend annotation pipeline, after the AXTree filter and before the model call. We use this property in Section 4 to interpret cost-versus-success comparisons as deployment-time tradeoffs rather than research-only configurations, and in Section 5 to argue that Phantom-SoM's behavior is a property of the format the model already saw inside SoM, not an emergent capability that requires new infrastructure.
    40	
    41	### 3.3 P-text
    42	
    43	**P-text** is the disambiguation ablation:
    44	
    45	```text
    46	P-text(page) =
    47	  prompt = DOM prompt
    48	  text   = SOM_MARKS(page)
    49	  image  = None
    50	```
    51	
    52	Its observation is identical to Phantom-SoM: `[SOM_MARKS]` text only, no page screenshot. The only intended change is the system prompt. In both B0 (`p79/agents/proxy_api_agent.py`) and B1 (`p79/agents/qwen3vl_agent.py`), `_system_prompts["phantom_som"]` maps to the SoM prompt, while `_system_prompts["phantom_dom"]` (and the alias `_system_prompts["phantom_text"]`) maps to the DOM prompt. For `som`, `phantom_som`, `phantom_dom`, and `phantom_text`, the agent passes through the `[SOM_MARKS]...[/SOM_MARKS]` text directly.
    53	
    54	This cell separates representation from prompt wording. If P-text behaves like Phantom-SoM, the flat marks text is driving behavior. If it behaves like DOM, the prompt is doing more of the work.
    55	
    56	### 3.4 The 2x2 Ablation Matrix and Excluded Hybrid
    57	
    58	The core ablation is a prompt-by-representation matrix:
    59	
    60	| | DOM prompt | SoM prompt |
    61	|---|---|---|
    62	| AXTree obs | DOM | *excluded — see below* |
    63	| `[SOM_MARKS]` obs | P-text | Phantom-SoM |
    64	
    65	Full SoM is adjacent to this 2x2: it uses the SoM prompt, the same `[SOM_MARKS]` text, and the marked screenshot. Vision is a separate screenshot-only baseline.
    66	
    67	The fourth cell — AXTree observation paired with the SoM prompt — is intentionally excluded from Paper 1 because it is not a self-consistent design point. The SoM system prompt instructs the agent to interact via `[SOM_MARKS]` IDs (e.g. `click [42]` referring to the SoM-marked element 42), but AXTree text uses an independent accessibility-tree ID space; an action like `click [42]` becomes parsing-ambiguous when the two ID systems do not match. This hybrid mode (i) has no clean LLM mechanism, (ii) confounds the prompt-effect ablation with mismatched-ID parsing failure, and (iii) does not reduce token cost relative to P-text. We treat the 5-mode set (DOM, P-text, Phantom-SoM, full SoM, plus Vision as a separate screenshot-only arm) as the diagonal axis-by-axis path through the 2×2×2 (text-payload-structure × prompt × image) design cube; the four mismatched-prompt-representation hybrids are excluded for the same reason.
    68	
    69	Each contrast isolates a different factor:
    70	
    71	- **DOM vs P-text** holds the prompt family fixed at DOM and changes the text-payload structure from AXTree to `[SOM_MARKS]`.
    72	- **Phantom-SoM vs P-text** holds the text observation fixed and changes only the prompt family.
    73	- **Full SoM vs Phantom-SoM** holds prompt and marks text fixed and adds the implemented marked-image channel.
    74	- **Full SoM vs DOM** measures the combined effect of SoM prompt, marks text, and marked screenshot relative to the standard text baseline.
    75	
    76	The 2x2 is not a routing policy by itself. It is a causal scaffold for Section 5: text-payload structure shapes exploration, while prompt wording tunes commitment confidence. Section 6 promotes this scaffold to a token-monotonic cascade — DOM → P-text (axis 1, text-payload swap, no token increase) → Phantom-SoM (axis 2, system-prompt swap, no data-token increase) → full SoM (axis 3, image embedding cost) — so a routing trigger never has to "add then remove" tokens.
    77	
    78	### 3.5 Implementation and Measurement Protocol
    79	
    80	All SoM-derived conditions share the same text-marks extractor. `_extract_text_marks` reads `obs_text` (the VisualWebArena accessibility-tree serialization the DOM baseline already uses) line by line, keeps each line whose label matches `\[\d+\]`, and produces `(id, label)` pairs up to a configured cap. `_build_som_result` then wraps those pairs in a `[SOM_MARKS] ... [/SOM_MARKS]` block. This text path **does not require bounding boxes**: the IDs come from the accessibility tree, not from a separate vision pipeline. Bounding boxes are only consulted by full SoM, which uses `obs_nodes_info` to draw numeric labels onto the page screenshot. Phantom-SoM and P-text reuse the exact `[SOM_MARKS]` text and drop the page screenshot; Marks are not re-filtered specifically for Phantom, and the source page state is unchanged.
    81	
    82	Reference images supplied by a task configuration are separate from the observation mode. These task-provided target images are passed to all modes as task input; Phantom-SoM removes only the current-page browser screenshot.
    83	
    84	Each episode starts from `environment.reset(task.config_file)`, and paper-grade condition comparisons use freshly reset site state to avoid cross-condition contamination. The April 27 Magento base-url/auth fix addressed an unrelated shopping-state reliability issue; this paper uses completed classifieds and reddit runs under the reset protocol.
    85	
    86	When comparing arms, we use same-task subsets: a task contributes only when the relevant conditions have completed it. We report **adjusted SR**, which starts from raw evaluator success and removes `na_fp` for not-applicable tasks that appear correct without agent-initiated finish, and `eval_fp` for evaluator matches caused by ineffective or non-finished trajectories. Section 4 reports results under these conventions; Section 5 uses the same traces for mechanism analysis.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section4_empirical_findings.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# 4. Empirical Findings
     2	
     3	This section reports empirical evidence that web-agent observation representations should be treated as routing arms, not only as fidelity levels. The key surprise is that **Set-of-Mark text alone**, with the marked screenshot removed, does not collapse to a DOM-like baseline. Instead, it behaves as a distinct text-only arm whose successes only partially overlap with DOM, full SoM, and vision-only observations. We refer to this arm as **Phantom-SoM**: the agent receives the `[SOM_MARKS]` textual element list and the SoM-style prompt, but no image.
     4	
     5	Throughout this section, we distinguish three measurement conventions. **Raw SR** is the evaluator success rate in `condition_summary_v2.json`. **Adjusted SR** subtracts confirmed false-positive terminal answers on not-applicable or evaluator-mismatch tasks. **Same-task adjusted SR** uses the same task set for all arms within a site. Unless otherwise noted, claims use same-task adjusted SR on completed B0 VisualWebArena classifieds and reddit runs. We also treat small cell-to-cell differences cautiously: under same-condition repeats, we observe roughly **+/-5% task-set variance**, so individual differences below about **2 pp** should be interpreted as noise-floor evidence rather than stable rankings.
     6	
     7	## 4.1 Setup
     8	
     9	We evaluate a single strong API-backed web agent, denoted **B0**, on two completed VisualWebArena sites: classifieds and reddit. The completed B0 pool contains **234 classifieds tasks** and **210 reddit tasks** for each reported observation condition:
    10	
    11	| Arm | Observation | Prompt family | Image input | Intended contrast |
    12	|---|---|---|---|---|
    13	| DOM | AXTree / DOM-derived text | DOM | No | Hierarchical text baseline |
    14	| SoM | `[SOM_MARKS]` text plus marked screenshot | SoM | Yes | Full Set-of-Mark baseline |
    15	| Vision | Screenshot without SoM marks | Vision | Yes | Visual-only baseline |
    16	| Phantom-SoM | `[SOM_MARKS]` text only | SoM | No | Isolated marks-text representation |
    17	| P-text | `[SOM_MARKS]` text only | DOM | No | Prompt-family control for marks text |
    18	
    19	The first three arms are the original Phase 1 representation baselines. Phantom-SoM is the new ablation arm. P-text is a prompt-family control: it receives the same marks-text-only observation as Phantom-SoM but uses the DOM prompt. We report all five modes for descriptive SR, cost, and latency. For the main routing-value claim, we keep the primary drop-one oracle on the four-arm comparison used throughout the paper: DOM, SoM, Vision, and Phantom-SoM.
    20	
    21	The original intuition was that Phantom-SoM should be either a broken SoM configuration or a weak DOM surrogate: it keeps a prompt that says the agent is operating with marked visual context, but removes the marked screenshot. The empirical results reject that collapse story. Phantom-SoM is lower than full SoM on classifieds, where marked screenshots carry clear visual grounding value, but it matches or modestly exceeds full SoM on reddit under adjusted SR.
    22	
    23	## 4.2 Single-Mode SR, Cost, and Latency
    24	
    25	The single-mode success rates show a site-modulated effect. On classifieds, full SoM remains the strongest individual representation. On reddit, Phantom-SoM is at least competitive with the strongest baselines, while using no image input. The table reports adjusted SR, because Figures 1, 2, 7, and 8 use episode-level `adjusted_success` for the paper comparisons. The latency column is p95 step latency from `condition_summary_v2.json`; cost is average total cost per task.
    26	
    27	| Site | Arm | Adjusted SR | Avg cost | p95 step latency | Metric |
    28	|---|---|---:|---:|---:|---|
    29	| Classifieds | DOM | 14.10 | $0.043 | 37.5s | N=234 |
    30	| Classifieds | SoM | **21.37** | $0.042 | 74.0s | N=234 |
    31	| Classifieds | Vision | 13.68 | $0.025 | 45.0s | N=234 |
    32	| Classifieds | P-text | 14.53 | $0.040 | 12.8s | N=234 |
    33	| Classifieds | Phantom-SoM | 14.53 | $0.044 | 18.2s | N=234 |
    34	| Reddit | DOM | 9.52 | $0.052 | 73.6s | N=210 |
    35	| Reddit | SoM | 10.48 | $0.041 | 58.9s | N=210 |
    36	| Reddit | Vision | 6.67 | $0.023 | 55.6s | N=210 |
    37	| Reddit | P-text | 12.38 | $0.046 | 58.1s | N=210 |
    38	| Reddit | Phantom-SoM | **13.81** | $0.038 | 51.4s | N=210 |
    39	
    40	The classifieds result is the expected sanity check: when tasks benefit from visual page layout and product imagery, the marked screenshot adds useful grounding and full SoM is clearly best (**SoM 21.37 vs Phantom-SoM 14.53; N=234; adjusted**). Phantom-SoM is close to DOM on classifieds (**14.53 vs 14.10**), but this is not a dominance claim; it is inside the noise floor and far below full SoM.
    41	
    42	The reddit result is the counterintuitive case. Removing the image does not eliminate the value of the SoM representation: Phantom-SoM matches or modestly exceeds full SoM and DOM on adjusted SR (**13.81 vs SoM 10.48 vs DOM 9.52; N=210; adjusted**). Given the variance we observe in repeats, the **+3.33 pp** gap over SoM is near the boundary of what should be treated as stable. We interpret this as evidence that Phantom-SoM is competitive on text-dominated reddit threads, not as an unconditional single-cell dominance claim. The more robust pattern is the cross-site asymmetry: **classifieds favors full SoM; reddit does not**. We treat that asymmetry as mechanism evidence rather than a setup bug: Section 5 shows a related site-modulated capability shift, with B0-to-B1 SoM visual-hijack/click-loop increasing by **+50.0 pp** on classifieds and **+33.3 pp** on reddit.
    43	
    44	This pattern suggests that the `[SOM_MARKS]` list is doing more than serving as a caption for a screenshot. It is a compact, flat, indexed text representation. Compared with AXTree-style DOM text, it removes much of the hierarchical nesting and metadata, and presents candidate actions as a linear set of marked elements. The outcome is not uniformly better, but it can push the agent toward a different solution basin.
    45	
    46	The cost and latency columns make the routing tradeoff concrete. On classifieds, Phantom-SoM's average cost is effectively in the same band as DOM and SoM (**$0.044 vs $0.043 vs $0.041**), but its p95 step latency is much lower than full SoM (**18.2s vs 74.0s**, roughly 4x faster). On reddit, Phantom-SoM is the cheapest of the main text/SoM-style arms (**$0.038 vs SoM $0.041 vs DOM $0.052**) and remains faster at p95 step latency than full SoM (**51.4s vs 58.9s**). These numbers support the cost-aware routing interpretation in Figures 7 and 9 without requiring Phantom-SoM to win every site.
    47	
    48	Raw SR tells the same high-level story but should not be mixed with adjusted SR. Some arms lose points after false-positive adjustment. Because the paper claim concerns deployable task success rather than answer attempts that only appear correct under a noisy evaluator, we use adjusted SR for the main empirical comparisons.
    49	
    50	## 4.3 Drop-One Oracle
    51	
    52	Single-mode SR can hide routing value. A representation may have modest average SR while still solving tasks that the other arms miss. We therefore compute a drop-one oracle: form the oracle union over the four primary arms, remove one arm, and measure how much oracle SR falls. This loss is the arm's incremental contribution to the routing pool.
    53	
    54	| Site | Largest loss | Second | Third | Fourth | Metric |
    55	|---|---:|---:|---:|---:|---|
    56	| Classifieds | SoM -8.55 pp | Vision -3.42 pp | Phantom-SoM -2.56 pp | DOM -2.14 pp | Drop-one oracle loss, N=234, adjusted |
    57	| Reddit | Phantom-SoM -3.33 pp | DOM -1.90 pp | SoM -1.90 pp | Vision -1.43 pp | Drop-one oracle loss, N=210, adjusted |
    58	
    59	The classifieds oracle is consistent with the single-mode story: full SoM contributes the most unique oracle value, followed by vision. Phantom-SoM still has a non-zero loss (**2.56 pp; N=234**), but the main effect on classifieds belongs to visual grounding.
    60	
    61	The reddit oracle is the stronger routing signal. Phantom-SoM has the largest nominal drop-one loss in the fresh four-arm oracle (**3.33 pp; N=210**), while DOM and SoM each contribute **1.90 pp** and Vision contributes **1.43 pp**. Because these are small absolute task counts, we do not read the ordering as a precise rank claim. The important point is that Phantom-SoM is comparable to the top routing-value arms and is not subsumed by DOM, SoM, or Vision.
    62	
    63	The overlap view supports the same conclusion. In the four-arm oracle, Phantom-SoM contributes a concrete reddit-only set of seven tasks (**7, 15, 36, 94, 157, 162, 167**) and a non-zero classifieds set as well. Two examples illustrate the kind of work this arm is doing. On reddit task 7, Phantom-SoM searched for the cake-recipe post and navigated directly to the OP recipe comment permalink. On reddit task 162, it searched within /f/wallstreetbets, scrolled hot posts, and returned the GIF URL for the retirement-account-versus-brokerage-account prompt. These are not proof of a universal mechanism by themselves, but they make the drop-one value concrete: the arm is adding recoverable successes, not only shifting aggregate percentages.
    64	
    65	The main empirical claim is therefore not that Phantom-SoM dominates the other modes. It does not. The claim is that it is an **independent routing arm**: it opens a distinct task pool at text-only cost, with the strongest relative benefit on the text-dominated reddit site and a clear visual-grounding disadvantage on classifieds.
    66	
    67	## 4.4 Two-Knob Ablation
    68	
    69	The five-mode result raises a confound: is Phantom-SoM useful because of the `[SOM_MARKS]` text representation, or because the SoM prompt changes the agent's confidence and behavior even without an image? P-text separates these factors. The full clean P-text runs are reported above for SR, cost, and latency; for behavioral mechanism, we use the verified same-task reddit subset of **N=48**, where all four cells of the prompt-by-representation ablation were manually checked.
    70	
    71	> **Text format shapes how the agent explores. Prompt wording tunes when the agent commits.**
    72	
    73	The first knob is exploration shape. On the same-task reddit ablation subset, replacing AXTree text with `[SOM_MARKS]` text shifts macro behavior away from DOM-like search loops and toward Phantom-SoM-like quick decisions. The verified search-loop rate is **22.7% for DOM** but **10.8% for Phantom-SoM and 10.8% for P-text** (**N=48; behavior metric; same-task subset**). The prompt change alone does not pull P-text back to DOM-like exploration. This supports the representation-driven part of the hypothesis: the flat marks list, not only the SoM prompt, changes the trajectory distribution.
    74	
    75	The second knob is commitment confidence. On the same N=48 subset, DOM and P-text have identical raw-to-adjusted SR gaps, while Phantom-SoM has a smaller gap:
    76	
    77	| Prompt family | Arm | Raw SR | Adjusted SR | FP gap | N/A FP | Metric |
    78	|---|---|---:|---:|---:|---:|---|
    79	| DOM prompt | DOM | 18.75 | 12.50 | 6.25 pp | 3 | N=48, raw/adjusted |
    80	| DOM prompt | P-text | 18.75 | 12.50 | 6.25 pp | 3 | N=48, raw/adjusted |
    81	| SoM prompt | Phantom-SoM | 18.75 | 16.67 | 2.08 pp | 1 | N=48, raw/adjusted |
    82	| SoM prompt | SoM | 22.92 | 16.67 | 6.25 pp | 3 | N=48, raw/adjusted |
    83	
    84	The aggregate SR equality should not be overread as task-level identity: equal counts such as 6/48 can occur with different solved-task sets. The robust signal is the false-positive pattern. DOM-prompt arms have the larger false-positive gap (**DOM and P-text: 3 N/A false positives, 6.25 pp gap; N=48**). The SoM-prompt Phantom arm has fewer N/A false positives (**1 N/A false positive, 2.08 pp gap; N=48**). This indicates that prompt wording affects terminal-action calibration: when the model decides it has enough evidence to `finish`.
    85	
    86	The two-knob account reconciles the apparent tension. The representation is the novel routing axis because it changes the agent's default exploration path. The prompt is a secondary but real tuning knob because it changes commitment confidence. Both are needed to explain the ablation. A representation-only story misses the FP gap, while a prompt-only story cannot explain why P-text follows Phantom-SoM rather than DOM on search-loop behavior.
    87	
    88	These findings explain why Phantom-SoM can be valuable despite not winning every single-mode comparison. Routing benefits depend on complementarity, not only average SR. A flat marks list can be worse for tasks that need hierarchy or visual layout, yet better for tasks where the same hierarchy induces over-searching. The practical implication is a cost-aware cascade: try cheap text representations first, use behavioral signals to detect when their exploration is unproductive, and escalate to full SoM when visual grounding is likely to matter.
    89	
    90	## 4.5 Reddit Substrate Behavioral Deep Dive
    91	
    92	The five-mode aggregate above conceals site-specific behavior. Section 4.5 fills in the reddit substrate. The Section 5 mechanism analysis explains *where in the model* the three axes appear; this subsection explains *what the agent does differently on reddit* under each axis swap, using outcome, macro, and micro behavioral evidence rather than residual-stream geometry.
    93	
    94	### 4.5.1 Reddit substrate
    95	
    96	The reddit environment in VisualWebArena is a Postmill-style forum rather than a visually organized marketplace. Its stable information structure is a hierarchy of forums, posts, and comments. The relevant navigation objects are therefore mostly textual: sidebar links to `f/<forum>` pages, post titles, comment-count links, comment permalinks, sort controls, and a global search box. The URL structure mirrors this hierarchy through path-based routes such as `/f/<forum>/<post>/<comment>`, so moving to the right page normally means choosing the right textual object in the forum tree rather than manipulating a visual layout.
    97	
    98	This substrate makes reddit an informative test case for separating the three axes. Images are frequent in the task prompts and in the posts themselves, but their role is usually evidential: an image can identify which post is being discussed, or disambiguate a content clue, but it is not the site's primary navigation affordance. The browser screenshot does not create the forum hierarchy; it only renders it. Conversely, the search box is prominent in the DOM and AXTree, but intrinsic search is not the intended substrate for many tasks. Repeated search is a failure basin: the agent can keep refining keywords while never taking the forum, post, or comment link that would satisfy the evaluator. The mechanism to explain is therefore not simply "text works better than vision." It is that each representation changes which textual affordances become salient enough for the model to commit to.
    99	
   100	### 4.5.2 Axis 1: Text Payload Structure
   101	
   102	Axis 1 changes the observation text from a hierarchical AXTree to a flat `[SOM_MARKS]` list while holding the DOM prompt fixed. On reddit this is the primary substrate-level mechanism. In the AXTree condition, the sidebar and post/comment links are embedded in a deep tree with many roles, containers, headings, and repeated page metadata. The search box is an easy high-level object, so the agent often converts the user intent into a query and then remains inside the search loop. In the flat marks condition, candidate links are serialized as a more uniform indexed action surface. This does not add image information and does not substantially change token budget; it changes the local attention pattern over action candidates. The forum link or comment permalink is no longer buried inside a nested accessibility structure, so the model is more likely to treat it as a clickable route rather than first translating the task into search terms.
   103	
   104	The evidence chain is consistent across dimensions. At the outcome level, adding P-text to the three-mode baseline contributes oracle value even without the SoM prompt or screenshot (Outcome 0c, +P-text +3.21pp single-phantom lift on the current oracle intersection). At the macro level, the whole-run strategy gradient shows the failure basin directly: reddit search-loop rate falls from DOM to Phantom-SoM and then to full SoM (Macro 1c, search-loop 51.90%->35.71%->31.43%). The axis-1-only macro effect is smaller than the compound prompt path, which is expected if flat text mainly changes which page objects are reachable rather than merely changing the action vocabulary. The micro evidence is sharper: DOM versus P-text has low path overlap for a text-only swap (Micro 2a, URL-path Jaccard 0.573), improves target-page reach (Micro 2b, target-hit +3.47pp), and reduces repeated keyword reuse (Micro 2c, max-keyword-repeat -0.633). The click-target view tells the same story: the two modes choose substantially different element sets even before images enter the system (Micro 2a-extra, click-target Jaccard 0.463).
   105	
   106	The standalone success table supports the same interpretation without requiring oracle selection. P-text raises adjusted SR over DOM on the full reddit set while preserving the same no-image deployment class (Outcome 0a, DOM 9.52% versus P-text 12.38%, N=210). The effect is not simply that P-text acts more: it uses fewer steps on average in the cost summary, and the cascade's largest text-axis macro shift is action-repeat rather than search success itself (Macro 1b, text-axis action-repeat +4.64pp). This is compatible with a routing-surface mechanism. The agent sometimes repeats a newly exposed marked control, but it is no longer confined to the same query-rewrite loop.
   107	
   108	Efficiency further constrains the explanation. Because P-text is generated from the same AXTree-derived text source and does not attach a screenshot, the reddit improvement cannot be attributed to paying the visual-token tax (Efficiency 3a, DOM $0.0516/episode versus P-text $0.0459/episode in the site dictionary). Axis 1 is therefore a representation effect: the observation text is rearranged into an indexed list, not enriched with new visual evidence.
   109	
   110	The mechanism is not monotone in every individual task, which is useful because it identifies the boundary condition. Reddit task #81 asks the agent to upvote every PhotoshopBattles post on the current page whose picture contains a cat. DOM succeeds by using both title semantics and button-state feedback: after an upvote, the observation exposes enough state change for the agent to move on to the next cat post. P-text matches DOM through the early actions but then collapses onto the same marked upvote control after the state should have changed. The case is a negative example for a simplistic "flat marks always help" claim. Axis 1 helps when the bottleneck is finding the right navigation object, but flat serialization can remove or weaken action-state cues such as `Upvote` becoming `Retract upvote`. On reddit, the aggregate effect is positive because the dominant failure basin is route discovery through text, not per-button state tracking.
   111	
   112	### 4.5.3 Axis 2: Prompt as a Decision Prior
   113	
   114	Axis 2 holds the flat `[SOM_MARKS]` text fixed and changes only the prompt family from DOM-style interaction to SoM-style marked-element interaction. On reddit this axis is secondary to the substrate shift, but it is the strongest macro driver of search and typing behavior. The SoM prompt asks the model to point at marked elements, which changes the prior over when to keep querying and when to commit to a visible candidate. In practical terms, it makes the agent more conservative about long exact queries and more willing to use marked links, comment anchors, tab focus, or backtracking after a stagnant page.
   115	
   116	The outcome evidence suggests a calibration effect rather than a uniform success-rate increase. Phantom-SoM has the lowest false-positive rate among the B0 reddit modes (Outcome 0b, P-SoM FP rate 0.48%), and adding P-SoM as a single phantom arm contributes additional oracle tasks beyond the three standard modes (Outcome 0c, +P-SoM +2.56pp). The task-pool overlap also stays below the redundancy sentinel: P-text and P-SoM solve overlapping but not identical task sets (Outcome 0d, P-text<->P-SoM Jaccard 0.500). Thus the prompt is not just a cosmetic instruction layered on top of the same decisions; it changes which tasks enter the solved pool.
   117	
   118	The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
   119	
   120	The prompt contrast is also visible in the mode-invariant click-target metric. With the text payload held fixed, P-text and P-SoM still have low click-target overlap (Micro 2a-extra, P-text<->P-SoM click-target Jaccard 0.484). This matters because it rules out a purely verbal explanation in which the SoM prompt only changes confidence wording at `finish`. The prompt changes which marked objects are selected during navigation.
   121	
   122	Reddit task #7 is the cleanest case study. The task asks for the permalink to the original poster's recipe comment for an image post. DOM overfits the visual description into a long exact query about a cake with cranberries and rosemary, spends 30 steps cycling through empty or unhelpful search results, and never reaches the comment permalink. Phantom-SoM instead treats the task as finding an OP recipe comment, searches more broadly for `cake recipe`, and reaches the comment permalink in five steps. The important point is not that the image is absent; both traces must infer from the same task context. The prompt shifts the query breadth and the commitment target, and the marked-comment affordance gives the model a terminal object that a search-loop policy fails to use.
   123	
   124	### 4.5.4 Axis 3: Image as Weak and Bidirectional Evidence
   125	
   126	The image axis adds the marked screenshot to Phantom-SoM, yielding full SoM. On reddit the net effect is weak and bidirectional because images are mostly content clues, not navigation structure. When the task requires recognizing the depicted post, the screenshot can help identify a candidate. But the same screenshot can over-anchor the agent on image URLs or visually salient marked regions, especially when the evaluator requires a post page, comment page, or action state rather than the image asset itself.
   127	
   128	The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
   129	
   130	The image-axis micro contrast confirms that the screenshot is behaviorally strong even when it is not outcome-positive. P-SoM and SoM have low URL-path agreement and frequent immediate divergence (Micro 2a, image-axis URL-path Jaccard 0.456; Micro 2f, early divergence 95.24%). Thus "weak" should be read as weak net value on this substrate, not as weak causal force. The screenshot changes decisions; on reddit, those changed decisions often point to content assets rather than evaluator-relevant post or comment routes.
   131	
   132	Reddit task #0 illustrates the harmful channel. The task asks for the sushi-platter post and its comments section. Phantom-SoM initially clicks the sushi image URL several times, but it eventually recovers and selects the actual post URL `/f/food/82896/i-ate-sushi-platter`, then targets the comment link. Full SoM remains trapped for the full budget, alternating between the sushi image URL and the forum page. The screenshot correctly identifies the sushi platter; the failure is not visual ignorance. The failure is action-policy over-anchoring on the marked image element, where visual salience suppresses the neighboring post/comment route. This is exactly why reddit should be described as text-dominated rather than image-free: image evidence exists, but its marginal value depends on whether it guides the agent toward a forum route or into an asset-level loop.
   133	
   134	The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
   135	
   136	### 4.5.5 Compound Axis 1+2: Phantom-SoM Versus DOM
   137	
   138	The compound transition from DOM to Phantom-SoM combines the flat text payload with the marked-element prompt while still avoiding the image channel. On reddit this compound arm is best interpreted as a complementary routing arm rather than as a uniformly stronger baseline. Its adjusted SR is modestly higher than DOM on the full 210-task per-mode summaries (Outcome 0a, P-SoM 13.81% versus DOM 9.52%, +4.29pp), but the more robust mechanism evidence is the divergence in which routes it explores. DOM and P-SoM have low click-target overlap (Micro 2a-extra, compound click-target Jaccard 0.421) and low URL-path overlap (Micro 2a, compound URL-path Jaccard 0.481). The oracle result then follows naturally: P-SoM adds tasks that the original three-mode set misses (Outcome 0c, +P-SoM +2.56pp), and P-text/P-SoM are not redundant with each other (Outcome 0d, Jaccard 0.500).
   139	
   140	This interaction also explains why Section 4's evidence catalog should not be read as a single leaderboard. DOM, P-text, and P-SoM can have close aggregate success rates while visiting different pages, selecting different marked elements, and failing in different basins. Axis 1 exposes alternative textual routes; Axis 2 changes the policy's willingness to commit to those routes. Their combination is useful because it changes the task pool available to a router, not because it dominates every individual task.
   141	
   142	The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section5_mechanism.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# 5. Mechanism Interpretability
     2	
     3	## 5.1 Overview and Theoretical Framing
     4	
     5	Why does Phantom-SoM sometimes achieve DOM-like cost while retaining part of the SoM signal? The mechanism evidence points to a phantom routing space in the residual stream: when the model receives flat Set-of-Mark text without the annotated image, it does not simply collapse to DOM. Instead, it occupies a mode whose text-axis geometry is close to DOM/P-text and whose image-axis geometry remains separated from full SoM.
     6	
     7	This section is the Zoom-4 layer of the paper's four-level account. Zoom 1 defines the architectural intervention, "skip the annotated image"; Zoom 2 measures the behavioral axes of text payload, prompt family, and image presence; Zoom 3 links the observed behavior to Mirage-style no-image visual reasoning and prompt-format sensitivity; Zoom 4 asks where the resulting mode is represented and whether it is causally used by the model. We index layers L0-L36, where L0 is the embedding-block output and L1-L36 are the 36 transformer decoder block outputs.
     8	
     9	The analysis builds on the linear-readable and steerable circuit framework of Wu et al., which uses mode means, PCA geometry, and mean-difference activation steering to study tool selection, and on work showing middle-layer cross-modal information flow in VLMs \citep{wu2026toolcalling,kaduri2024whatsintheimage}. Our setting is not a replication of those papers. It is a multimodal web-agent application of the same representation-level question: whether a behaviorally useful routing arm is linearly readable, partially steerable, and causally active inside the model.
    10	
    11	Four mechanism claims organize the evidence (revised 2026-05-12 after v2 NPZ re-extraction; see §5.7 revision note). First, observation modes are **linearly separable** in the residual stream: held-out leave-one-task-out AUROC = 1.000 across all mode pairs and all 37 layers (Method 4.2 v2). Second, the **geometric magnitude** of mode separation is dominated by the image axis (cosine ~0.04-0.07), with text-format and prompt-family axes producing only sub-permille cosine separation; the prior "three quantitatively distinct axes at 4:3:1 ratio" framing was a v1 NPZ artifact and is retracted. Third, SoM-to-no-image activation patching displaces target continuations at mid layers L11-L17 with consistent magnitude across classifieds and reddit (~25% target-overlap drop). The Exp 5 axis-2 prompt-only patching subset (cellhprompt cls + red) shows this displacement persists when text format is held flat, indicating prompt-family carries causal influence despite its near-zero geometric magnitude — a feature *used* without large feature *encoded* magnitude (\citep{wang2023interpretability} signature). Fourth, the shortcut trigger is **better described as the conjunction of integer-indexed marker and markup-sigil leading delimiter** than as an abstract "flat element list" — AXTree hierarchy preserves the early L04 image-axis peak, but so do indexed variants that lack either the integer (e.g., `hash_id_control`) or the sigil (e.g., `appagent_id`, `plain_numbered`); only the conjunction triggers the late shift. This refinement is **exploratory after W6** and awaits held-out falsifiers (`bare_N`, `bracket_no_int`).
    12	
    13	**Evidence status (revised 2026-05-13 after Bug 1+2 v2 NPZ re-extraction + 359719 random injection 数据 land)**:
    14	
    15	| Evidence layer | Method | Status |
    16	|---|---|---|
    17	| Linear readability (held-out AUROC) | Method 4.2 v2 (§5.2, §5.7) | **Strong** — held-out leave-one-task-out AUROC = 1.000 across all 15 mode pairs × all 37 layers on both cls and reddit (Bug 3 fix lototask CV) |
    18	| Geometric magnitude (cosine gap) | Method 4.2 v2 (§5.2, §5.7) | **Image axis dominates** — image pair peak ~0.04-0.07; text-format + prompt-family axes peak ≤ 0.009 at L36 boundary (no localized peak). Prior "three quantitatively distinct axes" framing retracted; was v1 NPZ Bug 2 artifact |
    19	| Causal continuation patching (SoM → no-image arms) | Stage 2/3 (§5.4) | **Causal** — mid-layer L12-L18 transfers across cls + reddit, additive across DOM/P-text/P-prompt targets, Gaussian-random negative controls at ~0. **Unchanged by v2 (uses archive directly, not Stage 4 NPZ)** |
    20	| Causal axis-2 prompt-only patching | Exp 5 cellhprompt cls + red (§5.4) | **Causal continuation evidence, 2 sites, N=24 each; 0.20-0.30 displacement at L11-L17 captures 80-125% of combined image+prompt patching effect**. Task-shuffled content-specificity control (cellhprm_*_tsh Myriad 359768+359769) in flight. Gaussian random control (cellhprm_*_rand 359719+359720) DESTROYS output regardless of axis (codex Bug 6 prediction confirmed; Gaussian is weak baseline) |
    21	| Steering (mean-diff activation) | Method 4.4 (§5.3) | **Weak / partial** — best H-mean 0.33 at L33 α=10, layer-α tradeoff prevents single sweet spot, treated as evidence ceiling not validation. **Unchanged by v2** |
    22	| Output divergence | Exp 3 logit lens (§5.7) | **Re-run pending** on v2 NPZ. V1 reported KL 0.16 at L23 axis-2 + KL 0.69 at L23 axis-1; V2 likely revises both. Mechanism direction (lm_head amplifies residual into output KL) probably survives; magnitudes will change |
    23	| Trigger attribution (which formats trigger shortcut) | W6 tokenization (§5.5) | **Exploratory** — 6 marks-like variants split 2-vs-4 on first-token sigil; held-out falsifier `bare_N` (integer no sigil) and `bracket_no_int` (sigil no integer) pending |
    24	
    25	The cross-site evidence stack is deliberately defensive. Per-task H1 fragility shows the dichotomy is an aggregate mechanism rather than a deterministic per-task law. Reverse-tier classifieds runs defend against strong-tier selection bias. Reddit format variation replicates the shortcut direction with cleaner mid-layer peaks. Reddit Method 4.2 replicates the Mirage signature: Phantom-SoM remains close to DOM on the text axis while separating from SoM on the image axis. Paper 1 uses these results for mechanism interpretation only; routing implementation is deferred to paper 2, consistent with the paper-planning scope split.
    26	
    27	## 5.2 Method 4.2: PCA Cosine Gap
    28	
    29	Method 4.2 extracts hidden states from Qwen3-VL-4B B1 runs and compares observation modes by layer. For each mode pair and layer, we compute the cosine gap between hidden-state means, evaluate AUROC by projecting examples onto the mean-difference direction, and summarize per-mode geometry through PCA top-10 variance. The classifieds baseline contains 288 examples, formed from 24 strong-tier tasks, two archived steps, and six modes, over 37 indexed layers.
    30	
    31	The robustness suite passes all five checks in the plan. Label permutation leaves the real AUROC 9.8 standard deviations above the permuted baseline. Per-task analysis is positive for all 24 tasks. Step 2 and step 5 curves are invariant at the mechanism level. The L23 silhouette score is at least 0.5, showing nontrivial clustering. Bootstrap 95% confidence intervals are tight, with widths of roughly 4-15% of the corresponding means.
    32	
    33	The key classifieds snapshot is the L17 cosine-gap table:
    34	
    35	| Pair at L17 | Cosine gap | 95% CI | AUROC |
    36	|---|---:|---:|---:|
    37	| P-SoM <-> P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
    38	| DOM <-> P-prompt | 0.0013 | [0.0012, 0.0014] | 1.000 |
    39	| P-SoM <-> SoM | 0.0413 | [0.0403, 0.0422] | 1.000 |
    40	| DOM <-> Vision | 0.0547 | [0.0531, 0.0563] | 1.000 |
    41	
    42	The reddit replication lands the same qualitative geometry. At L17, P-SoM is close to DOM with cosine gap 0.0098 and close to P-text with gap 0.0027, while P-SoM-to-SoM remains much larger at 0.0423 and P-SoM-to-Vision at 0.0457. The DOM-to-Vision image-axis peak is L04 with cosine gap 0.0687 and AUROC 1.0.
    43	
    44	This is the Mirage signature in geometric form. Phantom-SoM is not represented as a weakened image mode. At the mid-layer disruption locus, it is a text-axis sibling of DOM/P-text, while the image-axis distance to full SoM remains large.
    45	
    46	## 5.3 Method 4.4: Mean-Difference Activation Steering
    47	
    48	Method 4.4 tests whether the readable mode direction can be used as a steering direction. For each layer, we form a mean-difference vector between Phantom-SoM-like and DOM-like hidden states, add it to each input at generation time with scaling factor $\alpha$, and evaluate whether the continuation moves toward the target mode while preserving the JSON action envelope. Following HDMI's evaluation vocabulary, reliability is the harmonic mean of completeness and selectivity, not a raw shift rate \citep{khorasani2026hdmi}.
    49	
    50	The v2 sweep covers layers [11, 17, 23, 29, 33, 34] and $\alpha \in [1,2,5,10,20]$, for 45 completed cells in the plan summary. The original L17, $\alpha=5$ smoke result reported H-mean 0.44, but the full sweep lowers that cell to 0.16. The plan records this as a smoke-variance artifact from notes 126/127: a 4-cell smoke was too small to support a sweet-spot claim.
    51	
    52	The strongest full-sweep cell is L33, $\alpha=10$, with H-mean 0.33. Its completeness is 38% and its selectivity is 29%. The layer profile is the important result: mid layers L11-L23 preserve the JSON envelope with 100% selectivity but have low completeness, while late-layer L33 produces the largest shifts but frequently over-steers the continuation out of the expected JSON format.
    53	
    54	This creates a probe-causal dissociation. The mid-layer geometry is cleanly readable and causally implicated by patching, but fixed mean-difference steering is only partially reliable. The 0.33 H-mean is therefore an evidence ceiling for Method 4.4, not a final control result. Section 8 should treat LA-HDMI and SAE feature steering as future work motivated by this ceiling, without claiming that either method has already improved it.
    55	
    56	## 5.4 Stage 2/3: Activation Patching for a Causal Mid-Layer Mechanism
    57	
    58	Activation patching provides the causal test. For each task, the clean/source run and corrupt/target run use the same archived browser step and deterministic 50-token continuation. In the core SoM-to-Phantom-SoM setup, the source prompt is `som`: task instruction, SoM prompt family, flat `[SOM_MARKS]` text, and annotated screenshot. The target prompt is `phantom_som`: the same instruction, same prompt family, and same `[SOM_MARKS]` text, but no image. Source hidden states are cached by layer, injected into the final input-token position of the target on the first forward pass, and subsequent decoding proceeds normally through the model cache.
    59	
    60	Each patched continuation is scored against the unpatched source and target continuations. The main disruption statistic is the drop in `token_overlap_to_target`; Levenshtein distance to target is the paired backup. Layer-wise tests compare each grid layer to the final-layer reference using task-paired differences and Holm-Bonferroni correction across the canonical grid. Random-injection controls replace source hidden states with Gaussian tensors matched to source activation mean and standard deviation.
    61	
    62	The Stage 2 P-SoM<->SoM dashboard now contains ten completed cells, including reddit F/G and the reddit random control:
    63	
    64	| Cell | Site | Direction | Mid-layer target-overlap drop | Holm status |
    65	|---|---|---|---:|---|
    66	| A | cls | SoM->P-SoM forward | -0.32 at L17 | significant |
    67	| B | cls | P-SoM->SoM reverse | -0.16 at L17 | significant |
    68	| C | cls | reverse-tier forward | -0.02 at L17 | null |
    69	| D | cls | strong-tier reverse | -0.18 at L17 | significant |
    70	| E | cls | random injection | -0.03 uniform | negative control |
    71	| F | reddit | SoM->P-SoM forward | -0.21 at L17 | significant |
    72	| G | reddit | P-SoM->SoM reverse | -0.18 at L17 | significant |
    73	| Cr/Dr | reddit | 2x2 controls | -0.15 to -0.18 | significant |
    74	| Er | reddit | random injection | approximately 0 uniform | negative control |
    75	
    76	Stage 3 extends this from P-SoM to the three no-image arms, testing whether the image-feature axis is shared across DOM, P-text, and P-prompt targets. The table below reports per-task-paired Δoverlap-to-target from the patching_continuation_results.json under each cell directory, with the layer at which the disruption peaks.
    77	
    78	| Site | SoM->DOM | SoM->P-text | SoM->P-prompt | best-L Δ range |
    79	|---|---:|---:|---:|---:|
    80	| cls | -0.309 at L17, -0.352 at L18 (best) | -0.255 at L17, -0.270 at L12 (best) | -0.223 at L17, -0.273 at L13 (best) | [-0.273, -0.352] |
    81	| reddit | -0.335 at L11, -0.255 at L17, -0.338 at L14 (best) | -0.244 at L11, -0.236 at L17, -0.330 at L15 (best) | -0.233 at L11, -0.191 at L17, -0.322 at L14 (best) | [-0.322, -0.338] |
    82	
    83	All six Stage 3 cells are now closed. Two observations carry the cross-site claim. First, every cell's best layer falls inside the L12-L18 mid-layer window, and every cell's best Δoverlap-to-target is between -0.27 and -0.35. The mid-layer fusion locus is therefore not a single layer index but a tight 7-layer window that transfers across cls and reddit. Second, the interpretation is additive rather than arm-specific: a SoM source state displaces DOM, P-text, and P-prompt targets toward the source with similar magnitude, implying a shared image-feature substrate across all three no-image arms. The negative controls, Cell E at -0.03 and Cell Er near zero, rule out a generic nonzero-injection explanation.
    84	
    85	## 5.5 Image-Axis Peak-Layer Dichotomy and H1 Format Variation
    86	
    87	The cleanest single-pair signature is the image-axis peak-layer dichotomy. Across eight image-presence contrasts, the no-image side's text format predicts the peak layer with zero overlap. If the no-image side is AXTree text, the image-axis cosine gap peaks at L04 in all four pairs: DOM<->Vision, DOM<->SoM, P-prompt<->Vision, and P-prompt<->SoM. If the no-image side is `[SOM_MARKS]` or another flat marks text, the peak shifts to L17-L36 in all four pairs: P-text<->Vision, P-text<->SoM, P-SoM<->Vision, and P-SoM<->SoM.
    88	
    89	The refined H1 is a pretraining co-occurrence shortcut: when the input contains a marker token sequence that pretraining data associates with HTML / agent-trace visual grounding (specifically the conjunction of integer index and markup-sigil leading delimiter such as `[`, `<`, `@`), the model activates a visual-grounding pathway even if the image is absent. Flat element-list form alone is **not sufficient** — `appagent_id` (`id_N: label`) and `plain_numbered` (`N. label`) are nominally flat indexed lists but lack the markup-sigil and behave like AXTree-DOM (W6 evidence, exploratory). Prompt-format sensitivity makes this plausible at the input level \citep{sclar2024promptformat}; Method 4.2 shows it as a layer-resolved internal signature.
    90	
    91	The format-variation grid contains ten modes: six marks-like variants, two controls, and DOM/SoM baselines. In the classifieds strong-tier baseline, all six marks-like variants peak at L36, the hash-ID control also peaks at L36, the plain-sentence control peaks at L17, and the DOM baseline preserves the L04 peak. Because L36 is the boundary layer, this is best read as a strong late/monotonic signature rather than as a precise late-layer mechanism.
    92	
    93	The classifieds reverse-tier run reproduces the strong-tier shape. The six marks-like variants and hash-ID control again peak at L36, the plain-sentence control moves to L22, and DOM remains at L04. This defends H1 against the selection-bias concern that strong-tier curation alone created the pattern.
    94	
    95	The reddit format run is cleaner for the mid-layer interpretation. Four of six marks-like variants peak at L17, the plain-sentence control peaks at L17, hash-ID control returns to L04, and DOM remains at L04. **W6 attribution** (`docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md`, exploratory) further finds that the two L04 marks-like variants (`appagent_id`, `plain_numbered`) share a feature with the L04 DOM baseline: their first tokens are alphanumeric, while the four L17-peaking marks-like variants all start with markup-sigil tokens (`[`, `<`, `@`). The hash-ID control (`#a3f7`) starts with a sigil but lacks integer-marker structure and also peaks at L04, suggesting the trigger conjunction is integer-marker + markup-sigil rather than either alone. This is a post-hoc feature-attribution on a small (N=6 marks-like) format set; held-out falsifiers (`bare_N` = integer without sigil, `bracket_no_int` = sigil without integer) are not yet run. Cross-site, the safe claim is directional: marker formats that combine integer indexing with markup-sigil leading delimiters tend to delay image-axis separation into mid/late layers, while AXTree hierarchy and indexed-list variants lacking either feature preserve the early L04 image-axis peak. The reddit curve reveals the true L11-L17 fusion locus more clearly than the classifieds L36 boundary artifact.
    96	
    97	## 5.6 Convergent Four-Vertical-Defense Evidence Stack
    98	
    99	The first defense is per-task fragility. On 45 classifieds task-step pairs, only 11% satisfy the strict per-task dichotomy, even though aggregate marks-like peaks are later than AXTree peaks. This prevents over-claiming: H1 is a population-level mechanism signature with task variability, not a deterministic rule for every trajectory.
   100	
   101	The second defense is selection-bias robustness. The classifieds reverse-tier run replicates the strong-tier H1 pattern, including L36 marks-like peaks and L04 DOM baseline. The shortcut signature is therefore not an artifact of selecting tasks where SoM beats DOM.
   102	
   103	The third defense is cross-site H1. Reddit does not reproduce the exact boundary-layer shape, but it reproduces the direction of the indexed-list shortcut with a cleaner L17 mid-layer peak for four of six marks-like formats. The site changes the curve shape, not the basic interpretation.
   104	
   105	The fourth defense is cross-site Mirage geometry. Reddit Method 4.2 reproduces the central relation: P-SoM is close to DOM/P-text at L17 and far from SoM/Vision on the image axis, with AUROC 1.0 on the key contrasts. This supports cross-site generalization of the mechanism claim, not B0/B1 capability scaling.
   106	
   107	Two additional defenses remain deferred rather than folded into the claim: P2 cross-family Phi-3.5-Vision and P3 larger Qwen2-VL-7B. The current evidence is sufficient for the single-model, cross-site Qwen3-VL-4B mechanism section; family and capacity generalization belong in future work or Section 7.
   108	
   109	## 5.7 Residual-Stream Geometry vs Causal Use (revised after Bug 1+2 fix 2026-05-12)
   110	
   111	**REVISION NOTE**: Earlier drafts of this section described a "three-axis hierarchy" with image (≈0.041), text-format (≈0.029), and prompt-family (≈0.011) cosine gaps in a clean 4:3:1 magnitude ratio with distinct peak layers (L17/L23/L23). That description came from Method 4.2 hidden states extracted with a buggy `[SOM_MARKS]` regex that dropped 71/72 marks per task; the v1 Stage 4 NPZ contained near-empty 3-line text payloads, and mode-mean cosine gaps for axis-1 and axis-2 were inflated by prompt-template differences rather than text-payload differences. After the Bug 2 fix re-extraction (Myriad 359736 cls + 359737 reddit, NPZ `hidden_states_v2_fixed.npz`), axis-1 and axis-2 cosine peaks collapse to sub-permille and move from L23 to L36 boundary-monotone. The "three quantitatively distinct axes" claim is no longer supported. The revised account below is paper-grade.
   112	
   113	The pairs are constructed to isolate each axis. Axis-1 (text-format swap, prompt fixed) is measured by DOM↔P-text and P-prompt↔P-SoM. Axis-2 (prompt-family swap, text fixed) is measured by DOM↔P-prompt and P-text↔P-SoM. Image axis is measured by P-SoM↔SoM. All curves are computed on `stage4_multimode_b1_cls/hidden_states_v2_fixed.npz` (144 examples, 37 layers, 6 modes, strong-tier manifest filter, production `[SOM_MARKS]` formatter) and replicated cross-site on the matching reddit run.
   114	
   115	Peak-layer and magnitude table (cls v2, reddit qualitatively identical):
   116	
   117	| Axis | Pair | L17 cosine | L23 cosine | Peak L | Peak gap |
   118	|---|---|---:|---:|---:|---:|
   119	| Image | P-SoM ↔ SoM | 0.0416 | 0.0410 | L36 | 0.0416 |
   120	| Axis-1 text-format | DOM ↔ P-text | 0.0021 | 0.0027 | L36 | 0.0047 |
   121	| Axis-1 text-format | P-prompt ↔ P-SoM | 0.0021 | 0.0026 | L36 | 0.0048 |
   122	| Axis-2 prompt-family | P-text ↔ P-SoM | 0.0019 | 0.0028 | L36 | 0.0088 |
   123	| Axis-2 prompt-family | DOM ↔ P-prompt | 0.0013 | 0.0027 | L36 | 0.0068 |
   124	
   125	Two observations replace the prior three-axis hierarchy framing:
   126	
   127	1. **Image axis is the only well-localized geometric mechanism in the residual stream.** The image pair P-SoM↔SoM peaks at L36 with magnitude 0.042, but the early L04 peak for DOM↔Vision and P-prompt↔Vision (0.067 and 0.066) is the clean image-presence signature: when the no-image side preserves AXTree hierarchy, image-axis divergence is freshly observable at L04. When the no-image side is flat `[SOM_MARKS]`, the early peak attenuates (this is the original Mirage L04 dichotomy, and it survives the v2 re-extraction on the DOM-vs-Vision side; the SoM-side mirror requires re-examination because v1's L17 peak for P-SoM↔SoM shifted to L36 boundary in v2).
   128	
   129	2. **Text-format and prompt-family axes are linearly readable but geometrically near-zero.** All four non-image pairs (DOM↔P-text, P-prompt↔P-SoM, P-text↔P-SoM, DOM↔P-prompt) have peak cosine gap ≤ 0.009 and rise monotonically to a boundary layer L36 rather than localizing at a mid-layer peak. The held-out leave-one-task-out AUROC remains 1.000 across all pairs and layers, which means the 24 strong-tier tasks ARE perfectly separable along these axes — but the mode-mean difference vector is small. The right reading is that text-format and prompt-family modes carry low-magnitude but high-reliability linear signatures in the residual stream rather than substantial geometric clusters.
   130	
   131	The disjoint between **small geometric magnitude (cosine ≤ 0.01)** and **substantial causal patching displacement (overlap-to-target drop of 0.20–0.30 in §5.4 cellhprompt and Stage 2/3 cells)** is the new headline mechanism observation. A causal axis-2 patch at L11–L17 displaces target continuation by ~25% even though the geometric mean-difference at those layers is sub-permille. This argues that residual-stream cosine magnitude **underestimates** the causal influence of a feature, consistent with the standard mechinterp distinction between feature *encoded* and feature *used* \citep{wang2023interpretability}. The activation-patching evidence (§5.4) is the load-bearing claim; cosine geometry is supporting evidence about readability, not magnitude.
   132	
   133	Phantom-SoM's drop-one hero contribution in `fig_meta_forest.png` (reddit drop-one CI [+0.95, +6.19] strict-positive) therefore cannot be attributed to "three-axis positional uniqueness" with quantitatively distinct magnitudes. The cleaner mechanism story is: Phantom-SoM is one of four modes occupying the no-image-flat-marks half of the phantom routing space, all of which produce small geometric separation from each other; the behaviorally distinct success-task pool (Jaccard 0.29–0.49 against other arms) is what gives drop-one its complementarity, and patching displacement at L11–L17 shows the difference matters causally for token continuation. The bridge from patching displacement to behavioral SR remains open.
   134	
   135	A held-out leave-one-task-out AUROC of 1.000 across small-magnitude pairs is itself a paper-grade observation. It says the modes are reliably linearly separable at any chosen layer with very small mean-difference vectors, which is a stronger claim about the residual stream than the original "distinct mid-layer peaks" framing. The information capacity of the residual stream to represent observation-mode identity is high; the *magnitude* of the representation is mostly image-driven. This reframing follows the linear-readability framework of \citep{wu2026toolcalling} without the cosine-magnitude overclaim.
   136	
   137	The output-amplification observation (logit lens, Exp 3) needs re-running on the v2 NPZ before its quantitative claims can be reported. The v1 logit lens reported peak KL 0.162 at L23 for the axis-2 pair P-text vs P-SoM, but the v1 input hidden states were the buggy 3-line-text version. The qualitative direction (lm_head amplifies residual-stream geometry into output KL) likely survives, but the absolute KL magnitudes will change; we report the v2 lm_head amplification numbers in a follow-up release.
   138	
   139	Two corollaries follow. First, the KL trajectory drops to approximately zero at L36 even though L23 KL is substantial. The mean hidden state at the final layer collapses to the shared JSON action-header tokens that every mode emits, so mode-distinct output signal is concentrated in the L23-L25 decoding window rather than at the final embedding. Second, this output-amplification observation is **mechanistic, not a deployment-time classifier claim**: the lm_head acts as an axis-agnostic ratio-preserving projection that scales residual-stream geometry into output-space KL — the L23-L25 KL magnitude is a property of the mean hidden state, not a per-task discriminator. Whether the L23-L25 hidden representation can be used as a held-out mode classifier — with per-task AUROC, random-direction baseline, and competitive comparison to surface-token classifiers — is open work. Routing exploitation, deferred to paper 2, will need to make this case explicitly rather than inheriting it from §5.7.
   140	
   141	## 5.8 Discussion and Limits
   142	
   143	The main limit is the Method 4.4 ceiling. The cosine-gap and patching evidence point to L11-L17 as the readable and causally active fusion region, while the best fixed mean-difference steering cell is late, L33 with $\alpha=10$, and has H-mean 0.33 because completeness and selectivity trade off. This supports a mechanism interpretation but not a strong deployment-time steering claim.
   144	
   145	The second limit is layer precision. Classifieds H1 peaks often hit L36, while reddit reveals cleaner L17 peaks. The robust claim is therefore an effect-direction claim: AXTree hierarchy preserves early image-axis separation, and flat element-list formats delay that separation into mid/late computation. We should not claim that every site or task has an identical peak layer.
   146	
   147	Literature positioning should stay modest. Section 5 applies the linear-readable, steerable, and mid/late-layer circuit framework to multimodal web-agent observation modes \citep{wu2026toolcalling,kaduri2024whatsintheimage,khorasani2026hdmi,fayyaz2026steermoe}. It should not claim novelty as the first such circuit or the first use of marked text. The contribution is controlled scientific characterization of the phantom boundary.
   148	
   149	Finally, AXTree hierarchy is the unique defeating format in the aggregate, but the reason hierarchy defeats the shortcut remains open. The plan records one attribution-pending hypothesis: hierarchy or indentation tokens may redirect cross-modal attention before the flat-list shortcut activates. That should be treated as a supplement question, not as a Section 5 finding.
   150	
   151	## NOTE FOR HUMAN
   152	
   153	Bibkeys audit (2026-05-12 21:18): all 5 core mechanism anchors verified present in `paper.bib` — `wu2026toolcalling`, `khorasani2026hdmi`, `kaduri2024whatsintheimage`, `sclar2024promptformat`, `fayyaz2026steermoe`. Plus 5 method/protocol references added: `wang2023interpretability` (IOI patching), `zhang2024patching` (patching survey, NEEDS_VERIFY exact paper), `holm1979sequentially` (multiple-comparison correction), `lipton2018troubling` (ML scholarship critique), `neurips2024checklist` (reproducibility standard). paper.bib total 67 entries / 638 lines.
   154	
   155	Behavioral content to relocate from current `section5_mechanism_reddit.md`: lines 17-75 should move to Section 4 or a new behavioral-routing subsection. Specifically, lines 17-23 are reddit substrate framing; lines 25-35 are Axis 1 text-payload behavior; lines 37-47 are Axis 2 prompt behavior; lines 49-59 are Axis 3 image behavior; lines 61-67 are compound P-SoM versus DOM behavior; lines 69-75 are scope/noise limitations. Lines 1-15 are method material that was retained conceptually but must use the new L0-L36 layer convention. Line 77 should be deleted or replaced because routing implementation is now paper-2, not paper-1 Section 6.
   156	
   157	Stage 3 numbers verified 2026-05-12 from full per-task paired-test computation on `patching_continuation_results.json` (each cell, 24 tasks × 36 layers). H-d-cls best L18 Δ=-0.352, H-d-red best L14 Δ=-0.338, H-t-cls best L12 -0.270, H-t-red best L15 -0.330, H-p-cls best L13 -0.273, H-p-red best L14 -0.322. All 6 cells' best layer lands in L12-L18 mid-layer window, Δ range [-0.27, -0.35]. The L17-only column previously cited in plan §5.2 reads -0.309/-0.255/-0.223 (cls) and -0.255/-0.236/-0.191 (reddit); plan §5.2 has been updated to record best-layer Δ instead of L17-only Δ.
   158	
   159	Pending items (post 2026-05-12 audit): (a) Method 4.4 sweep description should be "45 completed cells out of a 6x5 layer-alpha grid plus 3 placeholder cells that did not finish", not "45/48-cell sweep" (the 48-cell wording in plan §5.3 implies a 48-cell denominator that was never executed). (b) Bibkey `zhang2024patching` is marked NEEDS_VERIFY in `paper.bib` because the intended reference may be Heimersheim & Nanda 2024 [arXiv:2404.15255] rather than Zhang & Nanda 2024 [arXiv:2309.16042]; verify before submission. (c) Bibkey `fayyaz2026steermoe` is marked NEEDS_VERIFY pending deanon of the ICLR 2026 submission.
   160	
   161	## NOTE FOR HUMAN — /codex-stress 2026-05-12 findings + pending follow-ups
   162	
   163	Codex independent audit (`docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md`) surfaced 6 weak claims + 5 honest gaps that Claude /stress had missed. 3 fixed inline tonight in §5:
   164	
   165	1. ✅ §5.1 ¶4 — "flat element-list trigger" → refined to "integer-marker + markup-sigil conjunction" with W6 exploratory caveat
   166	2. ✅ §5.7 hero paragraph — "proximity to SoM on the image axis... as if image were present" → corrected to "large image-axis SEPARATION from SoM... no-image marks-text reshapes how image-axis divergence accumulates" (removed internal contradiction with §5.2 table where P-SoM↔SoM gap 0.0412 is the largest = a separation, not proximity)
   167	3. ✅ §5.7 corollary 2 — "deployment-time mode classifier on output logprobs has strictly more signal" + "Section 6 routing should treat L23-L25 logit-lens features as the cheapest mode-axis discriminator" → softened to "mechanistic observation, not deployment-time classifier claim; held-out classifier with random-direction baseline is open work"
   168	4. ✅ Evidence status table added at end of §5.1 — geometry strong / patching causal-continuation / Exp 5 axis-2 CI pending / steering weak / output divergence not classifier / W6 trigger exploratory
   169	
   170	**Cross-doc follow-ups NOT fixed tonight** (each on next-action list):
   171	
   172	- **§4 P-text adjusted SR inconsistency**: §4 table line 37 says 11.90, prose line 106 says 12.38, hero_claim_bootstrap_ci.md says 12.38. Need to canonicalize one number from episode-level adjusted-success and update every occurrence. (1h)
   173	- **plan.md:125-135 stale "L17 planning site"**: plan still asserts L17 singular planning site, while new evidence shows cosine peak L23 + patching peak L11-L17 + steering best L33; replace with "patch-sensitive continuation window L11-L17 under final-token replacement patching". (1-2h)
   174	- **§6 + §7 drafts missing**: §1:13 promises Section 6 (Generalization) and Section 7 (Limitations and Implications). Current draft files: no `section6*.md`; §7 either deferred to paper-2 (routing) or merge into §8. Resolve before submission.
   175	- **Exp 5 cellhprompt bootstrap CI + content-matched control**: Gaussian random injection control 359719/359720 in flight; codex notes Gaussian alone is weak — also need task-shuffled (source from different task) and per-task bootstrap CIs.
   176	- **Behavioral causal bridge gap**: patching displaces 50-token continuation, not SR / drop-one. No experiment currently bridges mid-layer patching effect to task-success outcome. Open work.
   177	- **Cross-family**: P2 Phi-3.5-Vision + P3 Qwen2-VL-7B (task #40, #41). At minimum needed for §6 generalization narrative.
   178	
   179	**Codex verdict**: 0.10-0.20 NeurIPS/ICML/ACL main accept probability; would reject @ reviewer-3 4/10. **Workshop / borderline mid-tier conference today**. Codex agrees the drop-one oracle CI + cross-site asymmetry + axis-2 cosine signal breadth are strong; §5 mechanism prose was over-claiming relative to evidence.

codex
我已经看到 §5 本身混入了 “NOTE FOR HUMAN” 和旧 audit follow-up 内容，这会作为稿件卫生/independence 风险处理；我不会引用 prior codex output 文件，也不会把那些 note 当作科学证据。现在继续核对 plan 与 result summaries 的原始数字。
exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Hero-claim bootstrap CI (W1 defuse)
     2	
     3	Per-seed bootstrap 95% percentile CI on paired adjusted-SR diffs and drop-one oracle. B=10000, seed=42. Tasks resampled with replacement at task level.
     4	
     5	**Defuse target**: /stress W1 attack — paper §1 hero claim 'P-SoM 13.81% > SoM 10.48% reddit' is statistically marginal under author's own 2σ hedge.
     6	
     7	## reddit (N=210 same-task)
     8	
     9	**Per-mode adjusted SR (%)**:
    10	
    11	- dom: 9.52%
    12	- som: 10.48%
    13	- vision: 6.67%
    14	- phantom_som: 13.81%
    15	- phantom_text: 12.38%
    16	- phantom_prompt: 9.52%
    17	
    18	**Pairwise SR difference, bootstrap 95% CI:**
    19	
    20	| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |
    21	|---|---:|---:|---:|---:|---:|
    22	| P-SoM vs SoM | +3.33 | +3.33 | [-0.95, +7.62] | 0.914 | 0.828 | 
    23	|  | | | ✗ crosses 0 | | |
    24	| P-SoM vs DOM | +4.29 | +4.29 | [+0.00, +8.57] | 0.963 | 0.914 | 
    25	|  | | | ✗ crosses 0 | | |
    26	| P-text vs DOM | +2.86 | +2.86 | [-0.95, +6.67] | 0.918 | 0.810 | 
    27	|  | | | ✗ crosses 0 | | |
    28	| P-SoM vs P-text | +1.43 | +1.43 | [-1.90, +5.24] | 0.739 | 0.548 | 
    29	|  | | | ✗ crosses 0 | | |
    30	
    31	**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
    32	
    33	| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |
    34	|---|---:|---:|---:|---:|---:|
    35	| dom | +1.90 | +1.90 | [+0.48, +3.81] | 0.981 | 0.767 | 
    36	|  | | | ✓ strict-pos | | |
    37	| som | +1.90 | +1.90 | [+0.48, +3.81] | 0.980 | 0.762 | 
    38	|  | | | ✓ strict-pos | | |
    39	| vision | +1.43 | +1.43 | [+0.00, +3.33] | 0.949 | 0.574 | 
    40	|  | | | ✗ crosses 0 | | |
    41	| phantom_som | +3.33 | +3.33 | [+0.95, +6.19] | 0.998 | 0.969 | 
    42	|  | | | ✓ strict-pos | | |
    43	
    44	## classifieds (N=234 same-task)
    45	
    46	**Per-mode adjusted SR (%)**:
    47	
    48	- dom: 14.10%
    49	- som: 21.37%
    50	- vision: 13.68%
    51	- phantom_som: 14.53%
    52	- phantom_text: 14.53%
    53	
    54	**Pairwise SR difference, bootstrap 95% CI:**
    55	
    56	| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |
    57	|---|---:|---:|---:|---:|---:|
    58	| P-SoM vs SoM | -6.84 | -6.84 | [-12.39, -1.28] | 0.005 | 0.001 | 
    59	|  | | | ✗ strict-neg | | |
    60	| P-SoM vs DOM | +0.43 | +0.43 | [-3.42, +4.70] | 0.538 | 0.374 | 
    61	|  | | | ✗ crosses 0 | | |
    62	| P-text vs DOM | +0.43 | +0.43 | [-3.42, +4.27] | 0.546 | 0.376 | 
    63	|  | | | ✗ crosses 0 | | |
    64	| P-SoM vs P-text | +0.00 | +0.00 | [-4.27, +4.27] | 0.464 | 0.317 | 
    65	|  | | | ✗ crosses 0 | | |
    66	
    67	**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
    68	
    69	| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |
    70	|---|---:|---:|---:|---:|---:|
    71	| dom | +2.14 | +2.14 | [+0.43, +4.27] | 0.993 | 0.877 | 
    72	|  | | | ✓ strict-pos | | |
    73	| som | +8.55 | +8.55 | [+5.13, +12.39] | 1.000 | 1.000 | 
    74	|  | | | ✓ strict-pos | | |
    75	| vision | +3.42 | +3.42 | [+1.28, +5.98] | 1.000 | 0.988 | 
    76	|  | | | ✓ strict-pos | | |
    77	| phantom_som | +2.56 | +2.56 | [+0.85, +4.70] | 0.999 | 0.943 | 
    78	|  | | | ✓ strict-pos | | |
    79	
    80	## Verdict on /stress W1
    81	
    82	Read the **reddit P-SoM vs SoM** row + **reddit drop-one P-SoM** row:
    83	
    84	- If both CIs are strict-positive (ci_lo > 0) AND P(diff > 0) > 0.95 → **W1 attack defused**,   §1 hero claim is bootstrap-supported. Remove the '2σ hedge' from line 5, lead with the magnitude.
    85	- If CIs cross zero but P(diff > 0) > 0.80 → **W1 partially defused**, the claim is directional
    86	  but not strictly statistically significant. §1 hero must downgrade to 'competitive within 2σ' as
    87	  the author already wrote, but the complementarity (Jaccard / drop-one positive on N=7 tasks) carries
    88	  the structural weight.
    89	- If P(diff > 0) < 0.80 → **W1 sustained**, §1 hero claim must rewrite to 'parity / complementarity
    90	  rather than dominance'. The single-mode comparison is unsupported.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Method 4.2 v1 vs v2 cosine gap comparison — Bug 1+2 propagation audit
     2	
     3	**Status**: Land 2026-05-12 late-late, after Myriad 359736 (cls v2) + 359737 (reddit v2) re-extraction with Bug 1 (tier filter) + Bug 2 (production `[SOM_MARKS]` format) + Bug 5 (model revision pin) fixes.
     4	
     5	## Headline result
     6	
     7	**§5.7 "three-axis hierarchy with quantitatively distinct magnitudes" claim is INVALIDATED by v2 data.**
     8	
     9	V1 numbers came from buggy NPZ where the SOM_MARKS regex `^\[\d+\]\s+\w+` extracted only 38 chars / 3 lines per task, dropping 71/72 marks. All Stage 4 cosine geometry was computed on near-empty text payloads where the only differentiator between flat-text modes (som / phantom_som / phantom_text) was prompt template. Modes still separated perfectly (AUROC 1.000) but the cosine-gap magnitudes were artifacts of prompt-template differences, not text-payload differences.
    10	
    11	V2 NPZ uses production `_extract_text_marks` (72-line full payload with `[id=N] {label}` envelope). Modes still separable (AUROC 1.000), but axis-1 + axis-2 cosine magnitudes collapse to noise level. Image-axis magnitudes preserve.
    12	
    13	## Side-by-side peak comparison (cls, N=24 strong-tier)
    14	
    15	| Mode pair | v1 buggy peak | v2 fixed peak | Magnitude Δ | Layer Δ |
    16	|---|---|---|---|---|
    17	| DOM ↔ Vision (image axis) | L04 0.0653 | L04 0.0670 | unchanged | unchanged |
    18	| P-prompt ↔ Vision (image axis) | L04 0.0649 | L04 0.0664 | unchanged | unchanged |
    19	| P-text ↔ Vision (image axis) | L36 0.0614 | **L04** 0.0602 | unchanged | **earlier** |
    20	| P-SoM ↔ Vision (image axis) | L36 0.0613 | **L04** 0.0599 | unchanged | **earlier** |
    21	| DOM ↔ SoM (image axis) | L04 0.0604 | **L36** 0.0496 | -18% | **boundary-shift** |
    22	| P-prompt ↔ SoM (image axis) | L04 0.0600 | **L36** 0.0439 | -27% | **boundary-shift** |
    23	| P-text ↔ SoM (image axis) | L20 0.0494 | **L36** 0.0488 | -1% | boundary-shift |
    24	| **P-SoM ↔ SoM (image axis, paper §5.7 image-axis anchor)** | **L17** 0.0412 | **L36** 0.0416 | unchanged | **L17 → L36** |
    25	| DOM ↔ P-SoM | L23 0.0321 | **L36** 0.0152 | **-53%** | L23 → L36 |
    26	| P-prompt ↔ P-SoM (axis-1 SoM-prompt) | L23 0.0292 | **L36** 0.0048 | **-84%** | L23 → L36 |
    27	| P-text ↔ P-prompt | L23 0.0288 | **L36** 0.0081 | **-72%** | L23 → L36 |
    28	| **DOM ↔ P-text (axis-1 DOM-prompt, paper §5.7 axis-1 anchor)** | **L23** 0.0254 | **L36** 0.0047 | **-81%** | L23 → L36 |
    29	| SoM ↔ Vision | L22 0.0238 | **L36** 0.0255 | +7% | boundary-shift |
    30	| **P-text ↔ P-SoM (axis-2, paper §5.7 axis-2 anchor)** | L23 0.0114 | **L36** 0.0088 | -23% | L23 → L36 |
    31	| DOM ↔ P-prompt | L36 0.0067 | L36 0.0068 | unchanged | unchanged |
    32	
    33	## Headline ratios
    34	
    35	| Ratio | v1 (3:1 ratio claim) | v2 (reality) |
    36	|---|---|---|
    37	| Image axis magnitude (P-SoM↔SoM) | 0.041 | 0.042 |
    38	| Text-format axis (DOM↔P-text) | 0.025 | **0.005** |
    39	| Prompt-family axis (P-text↔P-SoM) | 0.011 | 0.009 |
    40	| Image / text-format ratio | **1.7x** | **8x** |
    41	| Image / prompt-family ratio | **3.7x** | **5x** |
    42	| Text-format / prompt-family ratio | **2.3x** | **0.5x** ← axis-1 NOW SMALLER than axis-2 |
    43	
    44	The "image > text-format > prompt-family" hierarchy with 4:3:1-ish quantitative ratio (v1) is **wrong**. V2 reality: image axis dominates by ~5-10x; axis-1 is **smaller than** axis-2 (reversed ranking); both axis-1 and axis-2 are noise-level (<0.01 cosine).
    45	
    46	## L17 cosine gap snapshot (cls + reddit cross-site)
    47	
    48	| Mode pair | cls v1 | cls v2 | reddit v1 | reddit v2 |
    49	|---|---|---|---|---|
    50	| DOM ↔ P-text | 0.0120 | **0.0021** | (similar) | **0.0019** |
    51	| DOM ↔ P-SoM | 0.0124 | **0.0029** | (similar) | **0.0031** |
    52	| P-text ↔ P-prompt | 0.0132 | **0.0031** | — | **0.0032** |
    53	| P-text ↔ P-SoM (axis-2) | 0.0028 | 0.0019 | — | 0.0020 |
    54	| DOM ↔ SoM (image axis) | 0.0557 | 0.0452 | — | 0.0450 |
    55	| DOM ↔ Vision (image axis) | 0.0545 | 0.0571 | — | 0.0537 |
    56	
    57	Reddit cross-site replication confirms the cls pattern: image-axis magnitudes preserve, axis-1 + axis-2 collapse to sub-permille at L17.
    58	
    59	## AUROC lototask (held-out, paper-grade Bug 3 fix)
    60	
    61	All pairs at all layers report AUROC lototask = 1.000 (perfect held-out linear separability). The modes ARE distinguishable in residual stream; the **magnitude of the mode-mean difference** is just much smaller than v1 claimed.
    62	
    63	This is the key reframe: **separability survives, magnitude does not**. Cosine gap measures effect SIZE; AUROC measures CLASSIFICATION RELIABILITY. They can dissociate.
    64	
    65	## What this means for paper §5
    66	
    67	**§5.7 three-axis hierarchy** (the prior framing):
    68	> "Three quantitatively distinct axes: image axis L17 0.041, text-format L23 0.029, prompt-family L23 0.011, with 4:3:1 magnitude ratio that holds cross-site."
    69	
    70	→ **INVALIDATED**. Replace with:
    71	> "All three axes are linearly readable in residual stream (held-out AUROC 1.000 across cls and reddit). The image axis dominates geometrically (~0.04-0.07 cosine peak) and emerges by L04. Text-format and prompt-family axes produce sub-permille mean-difference (cosine ~0.005-0.009) without a localized layer peak (monotone rise to boundary L36). The geometric magnitude rank-order reverses cross-site at L17 (axis-1 ≤ axis-2), indicating these axes are noise-level rather than quantitatively distinct dimensions."
    72	
    73	**§5.2 Method 4.2** (cosine gap table at L17):
    74	- All non-image-axis numbers drop 4-8x (re-run on v2 NPZ provides canonical values)
    75	- L17 ceases to be a meaningful "disruption locus" for text-format / prompt-family axes — they peak at L36 (boundary monotone)
    76	
    77	**§5.5 image-axis peak-layer dichotomy** (paper claims "no-image side's text format predicts peak layer with zero overlap"):
    78	- v1 had: 4 pairs at L04 (AXTree no-image side) vs 4 pairs at L17-L36 (flat-marks no-image side)
    79	- v2 reorganization: DOM/P-prompt ↔ Vision still L04; **P-text/P-SoM ↔ Vision shifted from L36 → L04** (BREAKS dichotomy); DOM/P-prompt/P-text/P-SoM ↔ SoM ALL at L36 now (collapses dichotomy on SoM image side)
    80	- → **§5.5 dichotomy ALSO needs significant revision**. The clean "AXTree → L04, flat-marks → late" pattern is partially v1 artifact.
    81	
    82	**§5.4 Stage 2/3 patching** (Cell A-H/D-G/H-text/H-prompt/H-d/Exp 5):
    83	- These do NOT use Stage 4 NPZ; they use archive_subset directly via Stage 2B build_som_marks which calls production code
    84	- All Stage 2/3 patching results **REMAIN VALID**
    85	- Exp 5 cellhprompt cls + red axis-2 patching (80-125% capture of combined image+prompt displacement): **INTACT**
    86	- Mid-layer L11-L17 patching effect: **INTACT**
    87	
    88	**§5.3 Method 4.4 steering** (45-cell layer-α sweep):
    89	- Separate pipeline (uses run_stage4_method44_v2_sweep + different feature extraction): **INTACT**
    90	
    91	**§5.6 four-vertical-defense stack**:
    92	- Per-task fragility (uses Stage 4 NPZ): NEEDS RE-RUN on v2
    93	- Selection-bias (reverse-tier H1): uses format variation NPZ, separately audited (INTACT but baseline caveat)
    94	- Cross-site H1: format variation (INTACT)
    95	- Cross-site Mirage geometry: NEEDS RE-RUN on v2
    96	
    97	**Exp 1 axis-2 layer profile** (`axis2_layer_profile.md`): NEEDS RE-RUN on v2; current 4:3:1 ratio claim is invalidated.
    98	
    99	**Exp 3 logit lens** (`axis2_logit_lens.md`): NEEDS RE-RUN on v2. Output-space KL magnitudes likely also change (the lm_head amplifies whatever residual-stream signal is there).
   100	
   101	## What still stands for paper
   102	
   103	✅ §1 hero (drop-one CI strict-positive reddit + cls): unchanged
   104	✅ §4 empirical SR tables (after 11.90→12.38 canonicalization): unchanged
   105	✅ §4.5 reddit behavioral: unchanged
   106	✅ §5.4 Stage 2/3 patching + Exp 5 axis-2 causal: unchanged
   107	✅ §5.3 Method 4.4 steering: unchanged
   108	✅ §6 image-axis early L04 separation: unchanged (real)
   109	✅ Held-out AUROC 1.000 linear-readability: unchanged
   110	
   111	## New cleaner mechanism story
   112	
   113	> **Three claim layers, distinct evidence types**:
   114	> 1. **Linear readability** (Method 4.2 lototask AUROC 1.000): all 6 modes linearly separable in residual stream; small cosine magnitudes but reliable classification
   115	> 2. **Geometric magnitude is mostly image-axis driven** (Method 4.2 cosine peak): image presence produces ~0.04-0.07 cosine separation early L04; other axes produce sub-permille separation throughout
   116	> 3. **Causal patching effect at mid-layer L11-L17** (Stage 2/3 + Exp 5): 20-30% target-overlap displacement under final-token replacement patching, content-specific (task-shuffled control 359768/359769 in flight)
   117	>
   118	> The disjoint between **small geometric magnitude (0.005-0.009)** and **large causal patching effect (0.20-0.30)** is the new headline claim — it means residual-stream geometry underestimates causal influence by orders of magnitude. This is paper-grade-novel and reviewer-defensible.
   119	
   120	## Files / provenance
   121	
   122	- v2 NPZ: `results/mechanistic/stage4_multimode_b1_{cls,reddit}/hidden_states_v2_fixed.npz`
   123	- v2 provenance: `hidden_states_v2_fixed.provenance.json` (git SHA, model revision, task IDs, formatter hash)
   124	- v2 metrics: `/tmp/method42_{cls,red}_v2.{json,md}` (canonical paths after rerun: `results/mechanistic/stage4_multimode_b1_*/method42_metrics.json`)
   125	- Legacy v1 metrics still in `docs/checkpoints/stage4_method42_results.md`
   126	- Comparison source: this file

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Exp 5 — Axis-2 Prompt-Family Causal Patching (cellhprompt cls + red)
     2	
     3	**Status**: Closed 2026-05-12 — cellhprompt_cls (359511) + cellhprompt_red (359512) landed via manual auto_pull after silent-miss bug in watcher.
     4	
     5	## Design
     6	
     7	| Variable | cellhprompt (this exp) | H-text baseline (prior) |
     8	|---|---|---|
     9	| Source | `phantom_som` (no image, flat `[SOM_MARKS]`, SoM prompt) | `som` (image, flat `[SOM_MARKS]`, SoM prompt) |
    10	| Target | `phantom_text` (no image, flat `[SOM_MARKS]`, DOM prompt) | `phantom_text` (same) |
    11	| Axes flipped src→tgt | **prompt-family only** | image axis + prompt-family |
    12	| N tasks | 24 (cls strong-tier) / 24 (red strong-tier) | matching |
    13	| Layers | 37 (L0-L36, Qwen3-VL-4B language decoder) | matching |
    14	
    15	**Test logic**: Holding both `image` and `text-format` constant (off + flat) and patching source hidden states from `phantom_som` into a `phantom_text` run isolates whether the residual-stream prompt-family signature has *causal* effect on token continuation, not just *geometric* magnitude (which Exp 1 already showed is small at 0.011 cosine gap @ L23).
    16	
    17	## Result — mid-layer (L11-L17) patching causal effect
    18	
    19	| Site | Cell (axes) | overlap→tgt L11 | overlap→tgt L17 | LD→tgt L11 | LD→tgt L17 |
    20	|---|---|---:|---:|---:|---:|
    21	| cls | H-text (image+prompt) | 0.74 | 0.75 | 9.0 | 9.2 |
    22	| cls | cellhprompt (**prompt only**) | 0.80 | 0.79 | 8.5 | 8.5 |
    23	| red | H-text (image+prompt) | 0.76 | 0.76 | 9.0 | 8.6 |
    24	| red | cellhprompt (**prompt only**) | 0.80 | 0.70 | 7.0 | 8.8 |
    25	
    26	(Baseline `overlap→tgt = 1.00` at L35 = full target preservation, no patching effect.)
    27	
    28	### Causal weight decomposition
    29	
    30	- Axis-2 (prompt) **alone** displaces target output by **0.20-0.30 overlap** units, mid-layer L11-L17 peak.
    31	- Combined image+prompt (H-text) displaces by **0.24-0.26** at same layers.
    32	- **Prompt-only captures ~77-100% of the combined effect** (cls 0.21/0.25 = 84%; red @ L17 0.30/0.24 = 125%, **prompt-only stronger on red**).
    33	- Therefore **image axis contributes a small residual** when prompt-family already differs; prompt-family is the dominant causal driver in this 2-axis subspace.
    34	
    35	### Cross-site replication
    36	Both cls + red show the same mid-layer L11-L17 peak. Reddit shows *stronger* axis-2 effect at L17 than cls (overlap→tgt 0.70 vs 0.79).
    37	
    38	## Geometric ⫨ causal disjoint (two disjoints: magnitude AND layer)
    39	
    40	### Disjoint 1 — magnitude
    41	
    42	Compared with Exp 1 cosine geometry, using best-layer values:
    43	
    44	| Axis | Cosine gap (best layer) | Patching displacement (best causal layer) |
    45	|---|---:|---:|
    46	| Image (SoM ↔ P-SoM) | 0.041 @ L17 | ~0.04-0.05 (inferred from H-text − cellhprompt diff) |
    47	| Text-format (DOM ↔ P-text) | 0.029 @ L23 | (Exp H-d-cls/red, not directly compared here) |
    48	| **Prompt-family (P-SoM ↔ P-text)** | **0.011 @ L23** | **~0.20-0.30 @ L11-L17** |
    49	
    50	**4:3:1 cosine geometry ratio does NOT translate to 4:3:1 causal patching ratio.** Prompt-family has the **smallest** geometric magnitude but the **largest** causal patching weight.
    51	
    52	### Disjoint 2 — layer
    53	
    54	Critically, the **layer at which cosine peaks ≠ the layer at which patching has maximal effect** for prompt-family:
    55	
    56	| Layer | cls overlap→tgt | red overlap→tgt | Interpretation |
    57	|---|---:|---:|---|
    58	| L0  | 0.86 | 0.92 | early, signal not yet routed |
    59	| L11 | 0.80 | 0.80 | **causal peak (cls)** — prompt-family begins routing decision |
    60	| L17 | 0.79 | **0.70** | **causal peak (red)** |
    61	| **L23** | 0.96 | 0.89 | **cosine geometry peak, but patching weak** — representation stabilized |
    62	| L29 | 0.92 | 0.95 | downstream re-encoding |
    63	| L35 | 1.00 | 1.00 | output convergence (baseline preserved) |
    64	
    65	At **L23** (the cosine peak), patching displaces target output by only **0.04-0.11 overlap units** — much smaller than the **0.20-0.30** displacement at L11-L17.
    66	
    67	### Interpretation: signature ≠ use
    68	
    69	This is the **second** geometric/causal disjoint, in addition to magnitude:
    70	
    71	- **L23 is the prompt-family "signature layer"**: representation has stabilized to its most discriminable form (highest cosine separation between P-SoM and P-text). It reflects *what prompt was given* — a state variable.
    72	- **L11-L17 is the prompt-family "decision routing layer"**: patching here changes upstream signal that downstream layers consume to drive token continuation. It reflects *how the model uses the prompt* — a causal variable.
    73	
    74	Activation patching is path-dependent: an upstream patch propagates into all downstream computations, while a downstream patch leaves upstream inputs unchanged so subsequent layers can re-encode the same signal. This is consistent with standard mechanistic-interpretability findings (cf. \citep{wang2023interpretability} IOI circuit: feature *encoded* ≠ feature *used*).
    75	
    76	### Three reads of the data
    77	
    78	1. Residual-stream cosine separation is a **necessary but not sufficient** signal of causal mechanism.
    79	2. Prompt-family information is **dispatchable** — small geometric perturbation at the decision layer produces large output displacement when patched.
    80	3. **Where a feature is most readable (L23) and where it is most consequential (L11-L17) are different layers** — paper-grade mechanism claims must report both, not collapse them.
    81	
    82	## Implications for paper §5
    83	
    84	**Strengthens 3-axis mechanism story**:
    85	- Axis-1 (text-format): Exp 1 cosine 0.029 + H-d cells causal patching (prior)
    86	- Axis-2 (prompt-family): Exp 1 cosine 0.011 + **Exp 5 cellhprompt causal patching (this)**
    87	- Axis-image: Exp 1 cosine 0.041 + indirect (H-text − cellhprompt residual ~0.04-0.05)
    88	
    89	**Defuses /stress critique** "you only have axis-1 mechanism":
    90	- Now have causal evidence for axis-2 separate from axis-1
    91	- 2-site cross-replication (cls + red), N=24 each, paired L0-L36 sweep
    92	
    93	**Reframes hero argument**: The paper §1 framing "text-format shapes exploration; prompt tunes commit" is now backed by:
    94	- Behavioral: exploration rate axis-1 dependent (Exp 1 cosine sigma + §4.5 reddit behavioral)
    95	- Causal mechanism: prompt-family mid-layer L11-L17 patching produces output displacement comparable to image-axis flip
    96	
    97	## Caveats
    98	
    99	- N=24 per cell — bootstrap CI on per-layer overlap means would tighten interpretation.
   100	- "phantom_som" archive vs "som" archive — the codebase uses same hidden-state extraction infrastructure but `--source-mode phantom_som` extracts without image. The pilot_summary template label "(with image — clean)" is a hardcoded artifact, not a runtime check. Manual verification of the archive contents (hidden state norms) would close a soundness gap.
   101	- Patching displacement is a token-level metric; doesn't directly translate to SR / drop-one oracle. Behavioral consequence (which paper §1 hero is about) operates on top of this causal signal.
   102	
   103	## Files
   104	
   105	- `pilot_summary.md`: per-site
   106	- `patching_continuation_results.json`: per-layer per-task continuation strings + metrics (~1.3 MB each)
   107	- `patching_continuation_curves.png`: visual layer profile
   108	
   109	## Provenance
   110	
   111	- Myriad jobs: 359511 (cls) + 359512 (red), submitted 2026-05-12, finished 21:42 + 21:54 UTC
   112	- Watcher missed GONE events due to silent-miss bug (PR same commit) — auto_pull dispatched manually
   113	- Bash invocation:
   114	  ```
   115	  bash scripts/maintenance/auto_pull_myriad_cell.sh 359511 cellhprm_cls stage3_cellhprompt_cls_fwd_ptext_myriad
   116	  bash scripts/maintenance/auto_pull_myriad_cell.sh 359512 cellhprm_red stage3_cellhprompt_red_fwd_ptext_myriad
   117	  ```

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/axis2_per_task_fragility.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Axis-2 per-task fragility check
     2	
     3	Per-task cosine gap distribution at L23 (axis-2 peak per §5.7 / Exp 1).
     4	Each task averaged across its 2 steps; cosine gap computed between mode pairs.
     5	
     6	**Defuse target**: /stress W2 attack — axis-2 mean 0.0114 might be dominated by 2-3 outlier tasks.
     7	
     8	## Classifieds (24 tasks)
     9	
    10	| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |
    11	|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
    12	| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0132 | 0.0131 | [0.0124, 0.0142] | 0.0107 | 0.0174 | 100% | 100% | 0% |
    13	| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0048 | 0.0047 | [0.0044, 0.0052] | 0.0039 | 0.0065 | 33% | 0% | 0% |
    14	| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0287 | 0.0280 | [0.0250, 0.0312] | 0.0186 | 0.0456 | 100% | 100% | 92% |
    15	| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0407 | 0.0415 | [0.0353, 0.0438] | 0.0308 | 0.0597 | 100% | 100% | 100% |
    16	
    17	## Reddit (24 tasks)
    18	
    19	| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |
    20	|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
    21	| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0121 | 0.0120 | [0.0113, 0.0127] | 0.0102 | 0.0152 | 100% | 100% | 0% |
    22	| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0052 | 0.0051 | [0.0047, 0.0055] | 0.0039 | 0.0067 | 50% | 0% | 0% |
    23	| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0260 | 0.0263 | [0.0226, 0.0305] | 0.0174 | 0.0344 | 100% | 100% | 83% |
    24	| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0436 | 0.0439 | [0.0409, 0.0453] | 0.0382 | 0.0535 | 100% | 100% | 100% |
    25	
    26	## Top 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)
    27	
    28	| Task ID | Cosine gap |
    29	|---|---:|
    30	| 109 | 0.0174 |
    31	| 211 | 0.0151 |
    32	| 181 | 0.0146 |
    33	| 108 | 0.0146 |
    34	| 191 | 0.0143 |
    35	
    36	## Bottom 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)
    37	
    38	| Task ID | Cosine gap |
    39	|---|---:|
    40	| 10 | 0.0121 |
    41	| 228 | 0.0118 |
    42	| 116 | 0.0117 |
    43	| 32 | 0.0108 |
    44	| 161 | 0.0107 |
    45	
    46	## Top 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)
    47	
    48	| Task ID | Cosine gap |
    49	|---|---:|
    50	| 133 | 0.0152 |
    51	| 132 | 0.0145 |
    52	| 142 | 0.0135 |
    53	| 122 | 0.0135 |
    54	| 148 | 0.0131 |
    55	
    56	## Bottom 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)
    57	
    58	| Task ID | Cosine gap |
    59	|---|---:|
    60	| 137 | 0.0111 |
    61	| 107 | 0.0109 |
    62	| 115 | 0.0104 |
    63	| 116 | 0.0103 |
    64	| 135 | 0.0102 |
    65	
    66	## Verdict
    67	
    68	Read the `% > 0.010` column for the axis-2 P-text↔P-SoM pair:
    69	- cls: **100%** of 24 tasks above the L23 axis-2 mean magnitude
    70	- reddit: **100%** of 24 tasks above
    71	
    72	Interpretation tree:
    73	- If both ≥ 50% → axis-2 signal **broad**, /stress W2 attack defused, §5.7 framing OK
    74	- If both 25-50% → axis-2 signal **modest but present**, §5.7 needs to add 'task-conditional sparse' qualifier
    75	- If both < 25% → axis-2 signal **aggregate artifact**, §5.7 three-axis claim must downgrade to 'axis-1 + image-axis with axis-2 weak per-task'
    76	
    77	Median values: cls=0.0131, reddit=0.0120.
    78	Compare to mean: cls=0.0132, reddit=0.0121.
    79	If median << mean, the distribution is right-skewed → outlier-driven (consistent with /stress W2 attack).

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4 H1 test: indexed-list format variation
     2	
     3	Test refined H1 hypothesis (pretraining co-occurrence shortcut):
     4	*"input contains mark-like indexed region list → activates visual-grounding pathway"*
     5	
     6	**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
     7	- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
     8	- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut
     9	
    10	## Result table (sorted by peak layer)
    11	
    12	| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
    13	|---|---|---|---|---|
    14	| appagent_id | `id_N: label (AppAgent)` | marks-like | **L04** | 0.0488 |
    15	| plain_numbered | `N. label (numbered)` | marks-like | **L04** | 0.0505 |
    16	| hash_id_control | `#hash label (no integer)` | control (no integer) | **L04** | 0.0508 |
    17	| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0495 |
    18	| som_standard | `[N] role 'label' (SoM)` | marks-like | **L17** | 0.0429 |
    19	| browser_use_at | `@N label (Browser Use)` | marks-like | **L17** | 0.0515 |
    20	| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L17** | 0.0457 |
    21	| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L17** | 0.0431 |
    22	| plain_sentence | `'a, b, c, ...' (no list)` | control (no list) | **L17** | 0.0521 |
    23	
    24	## Grouped by H1 prediction
    25	
    26	### marks-like  (mean peak L13)
    27	
    28	- `[N] role 'label' (SoM)`: peak **L17** = 0.0429
    29	- `@N label (Browser Use)`: peak **L17** = 0.0515
    30	- `id_N: label (AppAgent)`: peak **L04** = 0.0488
    31	- `[BN:role:label] (Tarsier)`: peak **L17** = 0.0457
    32	- `N. label (numbered)`: peak **L04** = 0.0505
    33	- `<el_N role='..'>label</el_N> (XML)`: peak **L17** = 0.0431
    34	
    35	### control (no integer)  (mean peak L4)
    36	
    37	- `#hash label (no integer)`: peak **L04** = 0.0508
    38	
    39	### control (no list)  (mean peak L17)
    40	
    41	- `'a, b, c, ...' (no list)`: peak **L17** = 0.0521
    42	
    43	### AXTree-baseline  (mean peak L4)
    44	
    45	- `AXTree (baseline DOM)`: peak **L04** = 0.0495
    46	
    47	## H1 verdict
    48	
    49	- **6 marks-like variants**: mean peak layer = 13, range L04-L17
    50	- **2 control variants** (no integer / no list): mean peak layer = 10, range L04-L17
    51	- **AXTree-DOM baseline**: peak L04
    52	
    53	→ **H1 MIXED**: peak distribution doesn't fit simple binary prediction. Needs deeper analysis.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# W6 feature attribution — H1 reddit 2/6 marks-like L04 peak
     2	
     3	**Setup**: Qwen3-VL-4B tokenizer (Qwen/Qwen3-VL-4B-Instruct). Each marks-like format variant tokenized on a canonical single-element example (N=1, role=button, label=Submit). First-token character class + marker-fingerprint token count compared between L04-peak and L17-peak subgroups.
     4	
     5	## Per-variant tokenization
     6	
     7	| Variant | Peak | Example | n_tok | First token | First char class | Marker fp |
     8	|---|---|---|---:|---|---|---:|
     9	| appagent_id | L04 | `id_1: Submit` | 5 | `id` | alphanumeric | 4 (`id·_·1·:`) |
    10	| plain_numbered | L04 | `1. Submit` | 3 | `1` | alphanumeric | 2 (`1·.`) |
    11	| som_standard | L17 | `[1] button 'Submit'` | 7 | `[` | markup-sigil | 3 (`[·1·]`) |
    12	| browser_use_at | L17 | `@1 Submit` | 3 | `@` | markup-sigil | 2 (`@·1`) |
    13	| tarsier_typed | L17 | `[B1:button:Submit]` | 7 | `[B` | markup-sigil | 7 (`[B·1·:·button·:·Submit·]`) |
    14	| xml_tagged | L17 | `<el_1 role='button'>Submit</el_1>` | 14 | `<` | markup-sigil | 4 (`<·el·_·1`) |
    15	| hash_id_control | L04 | `#a3f7 Submit` | 5 | `#a` | markup-sigil | 4 (`#a·3·f·7`) |
    16	| plain_sentence | L17 | `Submit` | 1 | `Submit` | alphanumeric | 1 (`Submit`) |
    17	| dom | L04 | `button: Submit (AXTree)` | 7 | `button` | alphanumeric | 2 (`button·:`) |
    18	| som | L17 | `[1] button 'Submit' (+ image marks)` | 11 | `[` | markup-sigil | 3 (`[·1·]`) |
    19	
    20	## Subgroup first-char-class distribution (6 marks-like only)
    21	
    22	| Subgroup | alphanumeric | markup-sigil | punctuation | quote | other |
    23	|---|---:|---:|---:|---:|---:|
    24	| L04-peak (2) | 2 | 0 | 0 | 0 | 0 |
    25	| L17-peak (4) | 0 | 4 | 0 | 0 | 0 |
    26	
    27	## Hypothesis verdict
    28	
    29	✅ **Hypothesis supported (clean split)**: L04-peak variants both start with alphanumeric tokens (2/2); L17-peak variants start with markup-sigil tokens (4/4).
    30	
    31	## Secondary features
    32	
    33	- L04-peak mean marker-fp tokens: 3.00
    34	- L17-peak mean marker-fp tokens: 4.00
    35	- Δ (L17 − L04): +1.00
    36	
    37	## Full token sequence per variant (marks-like 6)
    38	
    39	- **appagent_id** (L04, `id_1: Submit`): 5 tokens: `id` · `_` · `1` · `:` · `ĠSubmit`
    40	- **plain_numbered** (L04, `1. Submit`): 3 tokens: `1` · `.` · `ĠSubmit`
    41	- **som_standard** (L17, `[1] button 'Submit'`): 7 tokens: `[` · `1` · `]` · `Ġbutton` · `Ġ'` · `Submit` · `'`
    42	- **browser_use_at** (L17, `@1 Submit`): 3 tokens: `@` · `1` · `ĠSubmit`
    43	- **tarsier_typed** (L17, `[B1:button:Submit]`): 7 tokens: `[B` · `1` · `:` · `button` · `:` · `Submit` · `]`
    44	- **xml_tagged** (L17, `<el_1 role='button'>Submit</el_1>`): 14 tokens: `<` · `el` · `_` · `1` · `Ġrole` · `='` · `button` · `'>` · `Submit` · `</` · `el` · `_` · `1` · `>`
    45	
    46	## Interpretation
    47	
    48	Within the 6 marks-like variants, the L17 vs L04 split corresponds to whether the variant's first tokens are **markup-sigil tokens** (`[`, `<`, `@`) — which co-occur with HTML / web-agent traces in pretraining and trigger the visual-grounding shortcut at mid layers — versus **plain alphanumeric tokens** (`id`, `1`) — which are common in prose / dictionary listings and behave like AXTree-DOM, peaking early at L04 where the image-axis divergence is freshly observable but not yet routed through the shortcut path.
    49	
    50	**Control variants (counterexamples that refine the rule)**:
    51	- `hash_id_control` (`#a3f7 Submit`): markup-sigil first token but L04 peak. The `#` sigil alone is not sufficient — the marker must contain an **integer index** (which `#a3f7` does not). This is consistent with prior H2 "integer is the trigger token" framing.
    52	- `plain_sentence` (`Submit`): alphanumeric first token but L17 peak. With no list/marker structure at all, the divergence path differs — possibly because the text observation drops to bare labels with no positional anchors, which the model handles via a different late-layer routing (likely commitment without grounding).
    53	
    54	Together these say: the L17 mid-layer shortcut requires **(a) integer-indexed marker + (b) markup-sigil-leading delimiter**. Either alone fails to trigger it.
    55	
    56	**Paper §5 implication**: H1's binary 'marks-like vs not' prediction is too coarse. The mechanism trigger is the **conjunction** of integer marker + markup-sigil first token, not the abstract concept of 'indexed list'. Variants like `id_N:` and `N.` are nominally indexed but lack the sigil; `hash_id_control` has the sigil but lacks an integer. Both fail to peak at L17. This refines H1 to **'integer marker + markup-sigil delimiter → triggers shortcut at L17'**, which is testable on additional variants and on a `bare_N` falsifier (drop the bracket from `[N]` and re-extract).
    57	
    58	**Falsifier (concrete next experiment)**: variant `bare_N` = `N button 'Submit'` (no brackets), which has integer + no sigil. Hypothesis predicts L04 peak. If it peaks L17, hypothesis fails.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/axis2_logit_lens.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)
     2	
     3	Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.
     4	For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement
     5	across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets
     6	amplified into output distribution divergence by late-layer decoding.
     7	
     8	## Classifieds site
     9	
    10	### Axis-2 (prompt-family) pairs:
    11	
    12	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    13	|---|---:|---:|---:|---:|---:|
    14	| P-text vs P-SoM  (axis-2 flat-text) | **L23** | 0.1621 | 0.0215 | 0.1621 | 0.0003 |
    15	| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0444 | 0.0184 | 0.0234 | 0.0000 |
    16	
    17	### Axis-1 (text-format) pairs:
    18	
    19	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    20	|---|---:|---:|---:|---:|---:|
    21	| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5508 | 0.1299 | 0.5508 | 0.0001 |
    22	| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6953 | 0.1069 | 0.6953 | 0.0003 |
    23	
    24	## Reddit site
    25	
    26	### Axis-2 (prompt-family) pairs:
    27	
    28	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    29	|---|---:|---:|---:|---:|---:|
    30	| P-text vs P-SoM  (axis-2 flat-text) | **L24** | 0.1260 | 0.0371 | 0.1230 | 0.0002 |
    31	| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0508 | 0.0228 | 0.0325 | 0.0000 |
    32	
    33	### Axis-1 (text-format) pairs:
    34	
    35	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    36	|---|---:|---:|---:|---:|---:|
    37	| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5273 | 0.0898 | 0.5273 | 0.0000 |
    38	| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6172 | 0.0806 | 0.6172 | 0.0002 |
    39	
    40	## Interpretation
    41	
    42	Three hypotheses tested:
    43	
    44	- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family
    45	  effect bypasses logit lens, only visible via attention heads or runtime decoding.
    46	- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →
    47	  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling
    48	  'knows but says differently' mirror).
    49	- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →
    50	  prompt prior signal proportional to mid-layer geometry, no amplification.
    51	
    52	Cross-site replication should hold for any of the three. Compare axis-2 KL magnitudes to
    53	axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/axis2_layer_profile.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Exp 1 — Axis-2 (prompt-family) layer profile
     2	
     3	**Question**: Method 4.2 at L17 shows prompt-family makes ~0 geometric contribution to residual stream
     4	(P-SoM↔P-text 0.0028, DOM↔P-prompt 0.0013). But forest plot drop-one places P-SoM as unique hero,
     5	implying axis-2 (prompt) contributes behaviorally. **Where in the model does axis-2 act?**
     6	
     7	**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.
     8	Overlay axis-1-only (text swap, prompt fixed) + image-axis P-SoM↔SoM reference curves to calibrate scale.
     9	
    10	## Results — classifieds site (stage4_multimode_b1_cls, 288 ex)
    11	
    12	| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
    13	|---|---|---:|---:|---:|---:|---:|---:|
    14	| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0013 | 0.0067 | **L36** | 0.0067 |
    15	| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0028 | 0.0089 | **L23** | 0.0114 |
    16	| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0134 | 0.0120 | 0.0201 | **L23** | 0.0254 |
    17	| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0127 | 0.0113 | 0.0201 | **L23** | 0.0292 |
    18	| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0394 | 0.0412 | 0.0411 | **L17** | 0.0412 |
    19	
    20	## Results — reddit site (stage4_multimode_b1_reddit, 288 ex)
    21	
    22	| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
    23	|---|---|---:|---:|---:|---:|---:|---:|
    24	| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0012 | 0.0059 | **L36** | 0.0059 |
    25	| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0027 | 0.0080 | **L23** | 0.0098 |
    26	| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0125 | 0.0092 | 0.0183 | **L23** | 0.0217 |
    27	| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0115 | 0.0086 | 0.0176 | **L23** | 0.0240 |
    28	| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0434 | 0.0423 | 0.0434 | **L4** | 0.0434 |
    29	
    30	## Interpretation
    31	
    32	Three hypotheses about axis-2 mechanism layer:
    33	
    34	1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.
    35	2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.
    36	3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.
    37	
    38	Compare peak layers above against axis-1 (text-format) pairs (the established mechanism with L17 peak) and image-axis reference (~0.04 magnitude). If axis-2 pair peak < 0.01 at all layers, hypothesis 1 holds.

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/plan.md | sed -n '1,180p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	---
     2	name: mechanism plan
     3	description: Full mechanism workspace — theory, lit anchor stack, methods, identification protocol, current findings, open questions, advisor sync, roadmap. Specialized companion to paper_planning §2; not a duplicate.
     4	type: workspace_plan
     5	last_substantive_update: 2026-05-12
     6	---
     7	
     8	# Mechanism Plan — paper §5
     9	
    10	## 1. Theory framework (1-screen summary, paper_planning §2 is canonical)
    11	
    12	### 1.1 Zoom 1-4 hierarchy
    13	
    14	| Zoom | Level | What our paper claims |
    15	|---|---|---|
    16	| **1** | Architectural | Phantom routing space = "skip annotated image" boundary contains 3 arms (P-text / P-prompt / P-SoM) sharing 4-fold drop-in property |
    17	| **2** | Behavioral (axis effects) | Axis 1 (text payload: AXTree vs [SOM_MARKS]) is PRIMARY; Axis 2 (prompt: SoM-prompt vs DOM-prompt) is secondary; Axis 3 (image presence: in vs out) is gating |
    18	| **3** | Named phenomena (lit-anchored) | Mirage Effect (Asadi 2026) / Scaffold Effect (Vu&Balloccu 2026) / Cross-modal flow (Kaduri) / Prompt-format sensitivity (Sclar 2024) |
    19	| **4** | Model-internal | L17 mid-layer is BOTH discrimination locus (probe AUROC 1.0) AND causally active planning site (Stage 2/3 patching + Method 4.4 v2 reliability) |
    20	
    21	### 1.2 Three-axis hierarchy quantified (Method 4.2 PCA cosine gap, Qwen3-VL-4B B1 cls)
    22	
    23	| Axis | Peak cosine gap | Peak layer | Magnitude ratio |
    24	|---|---|---|---|
    25	| Image-axis (vs SoM / Vision) | 0.06 | L4–L17 | **10×** |
    26	| Text-axis ([SOM_MARKS] vs AXTree) | 0.025 | L23 | **4×** |
    27	| Prompt-axis (SoM-prompt vs DOM-prompt alone) | 0.007 | L36 | **1×** |
    28	
    29	→ Mechanism magnitude image >> text > prompt. Validates `project_phantom_space_axes_format_not_information.md` memory: P-SoM closest mode at every layer is **P-text** (text-axis sibling, L17 cosine 0.0028 vs P-SoM↔SoM 0.0412 = 14.7× more distant).
    30	
    31	### 1.3 Image-axis peak-layer dichotomy (Mirage mechanism signature)
    32	
    33	Method 4.2 reveals image-axis cosine-gap peak shifts based on text format of the no-image side. Clean dichotomy, zero overlap across 8 image-axis pairs:
    34	
    35	| No-image side text | Peak layer | Pairs |
    36	|---|---|---|
    37	| AXTree (hierarchical) | **L04** | DOM↔Vision, DOM↔SoM, P-prompt↔Vision, P-prompt↔SoM |
    38	| [SOM_MARKS] / flat | **L17–L36** | P-text↔Vision, P-text↔SoM, P-SoM↔Vision, P-SoM↔SoM |
    39	
    40	### 1.4 H1 test confirms broader: flat-list (not just indexed) triggers shortcut (2026-05-12)
    41	
    42	Format variation extraction (Myriad job 352998, `stage4_format_variation_b1_cls`, 450 hidden states = 45 task-step × 10 modes). For each text format V, compute image-axis cosine gap V↔som per layer; peak layer reveals shortcut activation:
    43	
    44	| Format | Peak layer | Verdict |
    45	|---|---|---|
    46	| **AXTree hierarchical (DOM)** | **L04** | **SOLE format defeating shortcut** |
    47	| `"a, b, c, ..."` plain sentence | L17 | mid-level trigger |
    48	| `[N] role 'label'` (SoM standard) | L36 | strong trigger |
    49	| `@N label` (Browser Use) | L36 | strong trigger |
    50	| `id_N: label` (AppAgent) | L36 | strong trigger |
    51	| `[BN:r:l]` (Tarsier) | L36 | strong trigger |
    52	| `N. label` (numbered) | L36 | strong trigger |
    53	| `<el_N>label</el_N>` (XML) | L36 | strong trigger |
    54	| `#hash label` (control: no integer) | L36 | **still triggers!** |
    55	
    56	**Refined H1 verdict**: trigger is **flat element listing**, not "indexed list pattern". Even integer-free hash IDs and pure-sentence variants engage the shortcut. AXTree hierarchical depth is the **unique format** that defeats shortcut activation.
    57	
    58	Paper §5 implication: SoM-family web agents (Browser Use, AppAgent, Tarsier, OmniParser, etc.) **all** implicitly exploit the same flat-list-element-grounding shortcut from VLM training distribution. P79 phantom routing space makes this systematic and routes accordingly.
    59	
    60	## 2. Literature anchor stack (5 anchors, all 2026-05-08 except Sclar 2024)
    61	
    62	| Anchor | Role | bib key | What it gives our paper §5 |
    63	|---|---|---|---|
    64	| **Wu et al. 2026** (UCL lab, our advisors) | Method backbone | `wu2026toolcalling` (2605.07990) | Mean-difference activation steering at second-to-last layer, 77–100% switch on tool selection (93–100% at 4B+). Our Method 4.2/4.4 port to multimodal Qwen3-VL-4B web agent |
    65	| **Ma & Rui 2026** | Probe-vs-causal vocabulary | `maRui2026planning` (2605.07984) | "Planning-compatible representation" vs "causally active planning site". Qwen3-family pattern: probe works, causal patching weak (1% rhyme newline causal vs Gemma 67%). Our Method 4.4 v2 50% reliability is consistent with this family pattern |
    66	| **HDMI / Khorasani et al. 2026** | Alt method + evaluation metric | `khorasani2026hdmi` (2605.07631) | Probe-free gradient-based steering. Critically: **completeness × selectivity → harmonic mean reliability** — what our Method 4.4 v2 reports (not raw shift rate) |
    67	| **Lin & Liu 2026 Position paper** | Methodology protocol | `linLiu2026disclosure` (2605.08012) | 5-step identification disclosure norm: state claim / name strategy / enumerate assumptions / stress-test / separate validation. Paper §5 adopts as identification subsection structure |
    68	| **Peale et al. 2026** | §6 routing theory | `peale2026flexibleRouting` (2605.07805) | Uncertainty decomposition (reducible + irreducible) with regret bound. Paper §6 theoretical anchor; 4-fold drop-in maps onto predict/route/abstain trichotomy |
    69	
    70	## 3. Methods (Stage 4 + planned)
    71	
    72	### 3.0 Curation pipeline (Stage 0 setup, shared by Stages 1-4)
    73	
    74	`scripts/mechanistic/curate_mirage_tasks.py` builds a strong-tier task manifest from a B1 production run:
    75	- Filter to tasks where Mirage Effect is well-defined: agent succeeds in `som` mode but fails in `dom` mode (real image-feature dependence, not random failure)
    76	- Pick 24 cls + 24 reddit "strong-tier" tasks (clean Mirage); separate 15 cls reverse-tier (DOM > SoM, selection-bias defense)
    77	- Archive raw observations (AXTree, screenshots, SoM-marks) per (task, step) → `archive_subset_b1_<site>/<site>_task_<tid>/step_<step>/` for offline patching/extraction (Stages 2-4 read from this archive, no live env required)
    78	
    79	Outputs:
    80	- `results/mechanistic/curate_mirage_b1_classifieds/manifest.json` — cls strong/reverse tier task list
    81	- `results/mechanistic/curate_mirage_b1_reddit/manifest.json` — reddit strong tier
    82	- `results/mechanistic/archive_subset_b1_cls/` (17 MB, 144 files, 24 tasks × 6 steps)
    83	- `results/mechanistic/archive_subset_b1_reddit/` (35 MB, 356 files, 24 tasks × ~15 steps)
    84	
    85	### 3.1 Method 4.2 — PCA cosine gap (DONE)
    86	
    87	`scripts/analysis/stage4_pca_cosine_gap.py` + `stage4_robustness.py`. Three metrics per (mode_pair, layer):
    88	- A. Cosine gap = 1 − cos(mean_A, mean_B)
    89	- B. AUROC via (mean_A − mean_B) projection
    90	- C. Per-(mode, layer) PCA top-10 variance explained
    91	
    92	**5/5 robustness pass**:
    93	- Test A label perm: 9.8σ above noise (real 1.000 vs perm 0.629)
    94	- Test B per-task: 100% of 24 tasks positive
    95	- Test C per-step (step 2 vs step 5): invariant
    96	- Test D silhouette ≥ 0.5 at L23 (strong clustering)
    97	- Test E bootstrap 95% CI tight (4-15% of mean)
    98	
    99	### 3.2 Method 4.4 — mean-diff activation steering (v2 in flight)
   100	
   101	`scripts/mechanistic/run_stage4_method44_v2_sweep.py`. Layer × α sweep:
   102	- Layers: [11, 17, 23, 29, 33, 34] — covers mid (Stage 2 disruption locus) → late (Wu et al. second-to-last)
   103	- α: [1, 2, 5, 10, 20] — Wu et al. typical α=1, our diag found ≥5 needed for multi-step JSON
   104	- 24 cls strong-tier tasks × 2 steps × 30 cells = 1440 generations (~2h)
   105	
   106	**HDMI reliability metric**: completeness × selectivity → harmonic mean (Khorasani et al. 2026):
   107	- Completeness = % tasks where overlap_psom > overlap_dom
   108	- Selectivity = % tasks where JSON envelope preserved (starts with `{`)
   109	- Reliability = 2 · c · s / (c + s)
   110	
   111	**Current smoke (8/48 cells)**: L17 α=5 = **0.44** sweet spot (29% shift + 100% JSON valid). L33 α=10 = 0.23 (57% shift but JSON breaks).
   112	
   113	### 3.3 Method 4.5 — LA-HDMI / SAE (future work, paper §8)
   114	
   115	Two alternative paths:
   116	- **LA-HDMI**: probe-free gradient steering (Khorasani 2026 method). Per-input optimization replaces fixed mean-diff direction. May overcome Qwen3-family causal patching weakness
   117	- **SAE feature steering** (Zekun-recommended in advisor recording, paper_planning §108): train SAE on Qwen3-VL-4B residual stream (1-2 week cost, no public SAE exists), find mirage/format feature, steer directly. Differentiates from Wu et al. mean-diff path
   118	
   119	Decision pending Method 4.4 v2 full sweep + Zekun sync.
   120	
   121	## 4. Identification protocol (Lin & Liu 2026 disclosure norm)
   122	
   123	Following Lin & Liu Position paper, paper §5 must explicitly state:
   124	
   125	### 4.1 Causal claim (revised after /codex-stress methodology audit 2026-05-12)
   126	
   127	> The patch-sensitive continuation window L11-L17 (block-output index convention) at the last-input-token position is causally consequential for phantom routing space mode selection in Qwen3-VL-4B web agents, under final-token-replacement activation patching. Separately, the prompt-family axis (P-text ↔ P-SoM) signature is most readable in cosine geometry at the LATER layer L23 (signature layer ≠ decision layer; mechanistic-interpretability standard finding cf. Wang et al. 2023 IOI).
   128	
   129	The previous "L17 singular planning site" framing is **stale** and was inaccurate: (a) cosine peak for prompt-family axis is L23 not L17 (Exp 1 three-axis hierarchy, 2026-05-12); (b) patching causal peak is the L11-L17 *window*, not a single layer; (c) Method 4.4 steering full sweep (45 cells) lowered the L17 α=5 H-mean from the smoke result 0.44 to 0.16, and the highest cell is now L33 α=10 H-mean 0.33 with poor selectivity (not a single sweet spot at L17). Treat L17 as one peak within the L11-L17 window, not THE site.
   130	
   131	### 4.2 Identification strategy
   132	
   133	Triangulation of 3 evidence types:
   134	1. **Probe-level** (Method 4.2 PCA cosine gap; AUROC reported as both `in_sample` and held-out `leave-one-task-out` after 2026-05-12 Bug 3 fix; v1 buggy NPZ data invalidated, v2 NPZ in flight: Myriad 359736 cls + 359737 reddit)
   135	2. **Replacement patching** (Stage 2/3 Cell A-H, L11-L17 window disruption, Holm-significant per layer; baseline empirically equals unpatched at L35 final-block patching position since overlap→target ≈ 1.00 at L35 across all forward cells)
   136	3. **Additive steering** (Method 4.4 v2 full sweep 45 cells: layer-α tradeoff; mid-layer L11-L17 preserves JSON envelope but low completeness, late-layer L33 produces largest output shifts but over-steers — H-mean ceiling 0.33 indicates probe-causal dissociation, not a single sweet-spot validation)
   137	
   138	### 4.3 Identification assumptions
   139	
   140	| # | Assumption | Stress-test |
   141	|---|---|---|
   142	| A1 | L17 last-token hidden state mediates action selection (not earlier obs token positions) | Stage 2/3 swept all layers, L17 is peak |
   143	| A2 | Mean-difference direction approximates causal axis (Wu et al. hypothesis) | Method 4.4 v2 H-mean 0.44 partial — assumption holds weakly; LA-HDMI would test |
   144	| A3 | 24 strong-tier tasks generalize to broader VWA distribution | Stage 4 robustness Test B: 100% per-task positive, but tier-selection bias possible. Reverse-tier 15 tasks pending |
   145	| A4 | Qwen3-VL-4B mechanism transfers to other VLM sizes/architectures | Not tested. Wu et al. shows family generality on tool-only; multimodal+multi-step unknown |
   146	| A5 | Replacement patching faithfully simulates "natural" model read of the representation | Cell E random-injection control rules out non-specific disruption — content-specific causation confirmed |
   147	
   148	### 4.4 Stress-test result
   149	
   150	Cell E random-injection control: replacing source hidden with Gaussian noise (same μ, σ) yields **null L17 disruption effect**. Confirms our patching effect is source-content-specific, not noise-driven. Most directly stresses A5.
   151	
   152	### 4.5 Validation ≠ identification (Lin & Liu §5)
   153	
   154	- Method 4.2 AUROC 1.000 = validation (decodability)
   155	- Stage 2/3 + Method 4.4 v2 = identification attempts (causal use)
   156	- These are reported SEPARATELY in paper §5; reviewer should not conflate
   157	
   158	## 5. Current findings dashboard
   159	
   160	### 5.1 Stage 4 Method 4.2 (Qwen3-VL-4B B1 cls, 288 examples × 37 layers)
   161	
   162	| Pair @L17 | Cosine gap | 95% CI | AUROC |
   163	|---|---|---|---|
   164	| P-SoM ↔ P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
   165	| DOM ↔ P-prompt | 0.0013 | [0.0012, 0.0014] | 1.000 |
   166	| P-SoM ↔ SoM | 0.0413 | [0.0403, 0.0422] | 1.000 |
   167	| DOM ↔ Vision | 0.0547 | [0.0531, 0.0563] | 1.000 |
   168	
   169	### 5.2 Stage 2/3 patching disruption (14 cells, B1 cls + reddit)
   170	
   171	**Stage 2 — P-SoM ↔ SoM patching (10 cells):**
   172	
   173	| Cell | Site | Direction | L17 Δoverlap | Holm-sig |
   174	|---|---|---|---|---|
   175	| A | cls | SoM→P-SoM forward | -0.32 | ✓ |
   176	| B | cls | P-SoM→SoM reverse | -0.16 | ✓ |
   177	| C | cls | 2x2 reverse-tier fwd | -0.02 | ✗ (null) |
   178	| D | cls | 2x2 strong-tier rev | -0.18 | ✓ |
   179	| E | cls | random injection | -0.03 (uniform) | ✓ (negative control) |
   180	| F | reddit | SoM→P-SoM forward | -0.21 | ✓ |

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/plan.md | sed -n '181,360p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   181	| G | reddit | P-SoM→SoM reverse | -0.18 | ✓ |
   182	| Cr/Dr | reddit 2x2 | both directions | -0.15 to -0.18 | ✓ |
   183	| Er | reddit | random injection | ~0 (uniform) | ✓ |
   184	
   185	**Stage 3 — 2x2 mechanism additivity test (SoM → {DOM, P-text, P-prompt}, cls + reddit):**
   186	
   187	| Cell | Site | Source→Target | Best-L overlap→src | L17 Δoverlap→tgt | Path |
   188	|---|---|---|---|---|---|
   189	| H-d-cls | cls | SoM → DOM | L10 (0.192) | -0.33 | `stage3_cellhd_cls_fwd_dom_myriad/` |
   190	| H-p-cls | cls | SoM → P-prompt | L27 (0.219) | -0.22 | `stage3_cellhp_cls_fwd_prompt_myriad/` |
   191	| H-t-cls | cls | SoM → P-text | L28 (0.164) | -0.25 | `stage3_cellht_cls_fwd_text_myriad/` |
   192	| H-p-red | reddit | SoM → P-prompt | L20 (0.209) | -0.19 | `stage3_cellhp_red_fwd_prompt_myriad/` |
   193	| H-t-red | reddit | SoM → P-text | L01 (0.194) | -0.24 | `stage3_cellht_red_fwd_text_myriad/` |
   194	| **H-d-red** | reddit | SoM → DOM | L28 (0.204) | **L11 -0.33 / L17 -0.26** | `stage3_cellhd_red_fwd_dom_myriad/` ✅ done 2026-05-12 19:57 |
   195	
   196	**Stage 3 interpretation (6/6 cells complete 2026-05-12)**: All forward SoM→{no-image-arm} patching cells show mid-layer L11-L17 disruption -0.19 to -0.33 Δoverlap→tgt. Magnitude > random injection control (Cell E -0.03) at all 6. **Mechanism additivity confirmed**: image-feature axis is shared substrate across DOM / P-text / P-prompt arms — single SoM→{any-no-image-arm} patching displaces target prediction toward source. Cross-site cls + reddit both replicate (paper §5 universal mid-layer fusion locus); reddit fusion locus slightly earlier (L11 vs cls L17), magnitude identical.
   197	
   198	Stage 3 cross-site DOM-axis additivity table (paired-test Δoverlap-to-target from `patching_continuation_results.json`):
   199	
   200	| Site | SoM→DOM | SoM→P-text | SoM→P-prompt | best-L Δ range |
   201	|---|---|---|---|---|
   202	| cls | H-d-cls L17 -0.309 / L18 **-0.352** best | H-t-cls L17 -0.255 / L12 **-0.270** best | H-p-cls L17 -0.223 / L13 **-0.273** best | [-0.273, -0.352] |
   203	| reddit | H-d-red L11 -0.335 / L17 -0.255 / L14 **-0.338** best | H-t-red L11 -0.244 / L17 -0.236 / L15 **-0.330** best | H-p-red L11 -0.233 / L17 -0.191 / L14 **-0.322** best | [-0.322, -0.338] |
   204	
   205	All 6 cells best layer 落在 **L12-L18 mid-layer 窗口** (tight 7-layer band), Δ range [-0.27, -0.35]. Cross-site / cross-arm 一致, mid-layer fusion locus 不是 single layer index 而是稳定窗口.
   206	
   207	### 5.3 Stage 4 Method 4.4 v2 (FULL 45/48 cells, finalized 2026-05-11 22:00)
   208	
   209	H-mean reliability (HDMI framework) per (layer, α). **L17 α=5 smoke claim REFUTED by full sweep**; actual sweet spot at L33 α=10:
   210	
   211	| Layer \ α | α=1 | α=2 | α=5 | α=10 | α=20 |
   212	|---|---|---|---|---|---|
   213	| L11 | 0.04 | 0.09 | 0.20 | 0.12 | 0.12 |
   214	| L17 | 0.00 | 0.12 | **0.16** (was 0.44 smoke) | 0.12 | 0.09 |
   215	| L23 | 0.00 | 0.09 | 0.09 | 0.16 | 0.00 |
   216	| L29 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 |
   217	| **L33** | 0.04 | 0.00 | 0.00 | **0.33** ⭐ | 0.00 |
   218	| L34 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
   219	
   220	**Layer-specialization** (probe-causal dissociation):
   221	- Mid-layer (L11-L23): **selectivity 100%** at all α (JSON envelope preserved), but completeness 0-11% (modest shift)
   222	- Late-layer (L33): completeness 38% (highest), but selectivity drops to 29% (over-steers JSON)
   223	- L33 α=10 H-mean 0.33 = max reliability cell
   224	
   225	**Smoke variance lesson** (笔记 §126 + §127): 4-cell smoke H-mean 0.44 on L17 was statistical artifact (1/4 hit = inflated rate). Full 45-cell H-mean 0.16 is true rate. Future mechanism findings require n ≥ 30 cells before "sweet spot" claims.
   226	
   227	### 5.4 Image-axis peak-layer dichotomy (Method 4.2, 8 pairs)
   228	
   229	`docs/checkpoints/mechanism/results/layer_axis_emergence.md`. AXTree-no-image side → L04 peak (4/4); [SOM_MARKS]-no-image side → L17–L36 peak (4/4). Zero overlap. Mirage Effect mechanism signature.
   230	
   231	### 5.5 H1 test: flat-list format variation (Method 4.2 extension, 2026-05-12)
   232	
   233	`docs/checkpoints/mechanism/results/format_variation_h1_test.md`. 8 industry-relevant text formats + 2 controls. AXTree hierarchical (DOM) is **unique format** preserving L04 image-axis peak; all 8 flat-list variants (SoM standard, Browser Use @, AppAgent id_, Tarsier typed, plain numbered, XML tagged, hash-ID control, plain-sentence control) shift peak to L17–L36. Trigger is flat element listing, not specific token pattern.
   234	
   235	## 6. Open questions (paper-grade gaps)
   236	
   237	| Q | Status | Next action |
   238	|---|---|---|
   239	| ✅ Method 4.4 v2 full 48-cell sweep — sweet spot stable? | **Closed 2026-05-11 22:00**: L17 α=5 smoke 0.44 → full 0.16 (smoke variance artifact). **Real sweet spot L33 α=10 H-mean 0.33** | — |
   240	| ✅ H1 test: do all flat-list formats trigger shortcut? | **Closed 2026-05-12 00:00**: YES, including hash-ID + plain-sentence controls. AXTree-DOM is sole defeating format | — |
   241	| Reverse-tier 15 tasks vs strong-tier 24 — does L33 + H1 finding generalize beyond selection bias? | Med-High | qsub Stage 4 multimode + format variation with --tier reverse |
   242	| ✅ Cross-site Method 4.2 — does cls finding replicate on reddit? | **Closed 2026-05-12 16:30**: P-SoM↔DOM L17=0.0098 + P-SoM↔SoM L17=0.0423, AUROC 1.0 → Mirage signature replicated. See §7.3.1 | — |
   243	| ✅ Stage 3 reddit 2x2 closure — H-d-red | **Closed 2026-05-12 19:57** (Myriad 358831). L11 Δ=-0.33 / L17 Δ=-0.26. Cross-site additivity confirmed — see §5.2 Stage 3 table | — |
   244	| LA-HDMI vs mean-diff — does gradient steering beat 0.33 ceiling? | Med | Pending Zekun reply + attribution decision |
   245	| SAE feature steering feasibility — is 1-2 week self-training Qwen3-VL-4B SAE worth it? | Low-Med | Depends on Zekun reply + paper §8 prose direction |
   246	| B0 (proxy API) — paper §5 Qwen-specific or generalizable? | Low | Cannot test on B0; cite Wu et al. cross-family generality as proxy |
   247	| AXTree-defeats-shortcut mechanism — *why* hierarchy beats flat? Cross-modal attention specific to indentation tokens? | High (paper §5 supplement) | Activation patching at L4 with hierarchical-text vs flat-text → see which attention heads pre-disrupt image embedding |
   248	
   249	## 7. Advisor sync state — Zekun (Wu et al. 2026 first author = lab member)
   250	
   251	### 7.1 Timeline confirmed (not scoop)
   252	
   253	- 2026-04-09 笔记 §19: I first grok the paper (then "Anonymous 2026 ACL"), record cosine gap method + L23+ steering 80-93%
   254	- 2026-05-01 笔记 §108.19: upgraded to Zoom 4 anchor stack
   255	- 2026-05-02 commit `6662b91`: anchored into paper_planning §2 + paper.bib placeholder
   256	- 2026-05-09 advisor recording: Zekun explicitly recommended "SAE feature steering — 前所未有 inference time steering, 单独发 paper" — directed me to differentiating path
   257	- 2026-05-11: arxiv landed publicly; identity confirmed as lab paper
   258	
   259	**Net**: Zekun explicitly invited mechanism extension. Method 4.4 multimodal port is on his recommendation; SAE Method 4.5 is his next-step suggestion.
   260	
   261	### 7.2 Message draft (v3, paste-ready 2026-05-12)
   262	
   263	Updated after v2 full sweep + H1 test. Key revisions from §125.10 draft:
   264	- ❌ Removed: "L17 α=5 H-mean 0.44 mid-layer sweet spot" (smoke variance artifact, full data refutes)
   265	- ✓ Added: **L33 α=10 H-mean 0.33** = matches your second-to-last-layer choice; multi-step JSON selectivity drop explains 38% vs your 93% gap
   266	- ✓ Added: H1 test finding — flat-list format universally triggers shortcut (8/8 variants), only AXTree hierarchical defeats; implication for industry SoM-family agents
   267	- ✓ Three asks: (a) attribution co-author vs cite + independent; (b) your ablation on mid- vs late-layer (we see selectivity tradeoff); (c) SAE direction priority given mean-diff ceiling
   268	
   269	Final message (Chinese, casual WeChat tone):
   270	
   271	> Zekun 早, 你那篇 Tool Calling 上 arxiv 我看了, 恭喜! 我前几天按你说的开始 mechanism work, 跑出来一些东西想跟你 sync 一下, 顺便问几个方向问题。
   272	>
   273	> # Context
   274	> P79 paper 在做 VisualWebArena 的 phantom routing space — agent 6 种 obs mode (DOM 文本/SoM 标注图/Vision 裸图 + 3 个 phantom 变体). 模型 Qwen3-VL-4B, 你 Qwen 3 4B 同 base LM。
   275	>
   276	> # 1. Method 4.2 PCA cosine gap port 到 6 modes
   277	> 24 cls strong-tier × 2 step × 6 mode = 288 hidden states, 37 layer × 2560 dim。全 540 pair × layer AUROC = 1.000 (perm baseline 0.629, real 9.8σ above). 你方法在 multimodal Qwen 上 readable transfer 干净。
   278	>
   279	> # 2. Method 4.4 mean-diff steering (HDMI metric)
   280	> 45 task-step × 6 layer × 5 α full sweep. 用 HDMI completeness×selectivity → H-mean 评估:
   281	>
   282	>   - **L33 α=10 H-mean 0.33** (sweet spot, c=38% s=29%) ← matches 你 paper second-to-last-layer
   283	>   - Mid-layer (L11-L23) selectivity 100% 但 completeness 0-11% — readable but not effectively steerable
   284	>   - 你 paper Qwen 3 4B 93% switch vs 我 38% — 我猜原因是 multi-step JSON gen 的 selectivity 是真约束 (你 single-token tool decision selectivity 自动 1.0)
   285	>
   286	> # 3. H1 test: flat-list format variation (Myriad)
   287	> 测了 8 个 industry-relevant text format (Browser Use @, AppAgent id_, Tarsier typed, numbered, XML, hash-ID, plain-sentence + SoM baseline) vs AXTree-DOM:
   288	>
   289	>   - 全 8 flat variants peak L17/L36 (= 都触发 shortcut)
   290	>   - **AXTree hierarchical 是唯一保留 L04 peak 的 format**
   291	>   - 包括 hash-ID (no integer) + plain-sentence (no list) 都触发
   292	>   - = SoM-family agents 全 implicit exploit 同一 VLM shortcut, AXTree 是 sole exception
   293	>
   294	> # 三个 ask
   295	> (1) Attribution: paper §5 mechanism 这块 — cite 你 + 我独立 framing 比较合理, 还是 co-author 一篇 multimodal extension 比较好? 都 OK, 想听你意见。
   296	>
   297	> (2) 你 ablation 里有跑过 mid- vs late-layer 对比吗? 我 mid-layer selectivity 100% 但 shift 弱, late-layer shift 强但 envelope 破 — 不知道你 tool calling 上是不是也有这种 tradeoff。
   298	>
   299	> (3) 你之前 advisor 录音里建议 SAE feature steering, 我也写进 future work 了。现在 mean-diff ceiling ~0.33, 是不是 SAE 这条路更有差异化? Qwen3-VL-4B SAE 没公开, 自训成本 1-2 周, 你觉得值得 commit GPU 吗?
   300	>
   301	> 不急, 你忙完回我就行. paper 写得真漂亮.
   302	
   303	### 7.3 H1 generalization in-flight (2026-05-12 night)
   304	
   305	After per-task fragility revealed 11% strict dichotomy (aggregate statistical, not deterministic), launched 5-priority defense matrix to triangulate H1 across **(tier × site × family/size)**:
   306	
   307	| Pri | Test | Where | Status @ 06:25 | Sentinel |
   308	|---|---|---|---|---|
   309	| **P1** | Per-task fragility audit (24 cls strong) | DGX | ✅ done | `results/h1_per_task_fragility.md` |
   310	| **P2** | Cross-family (Phi-3.5-Vision 4.2B) | DGX | ❌ deferred (HF cas-bridge throttling) | `stage4_h1_phi35_cls/pilot_summary.md` |
   311	| **P3** | Within-family bigger (Qwen2-VL-7B, H1' capacity test) | DGX | ❌ deferred (HF cas-bridge throttling) | `stage4_h1_qwen2vl7b_cls/pilot_summary.md` |
   312	| **P4** | cls reverse-tier (selection-bias defense) | Myriad 353763 | ✅ **done 18:50:46** — shape (260, 37, 2560), 10 modes, 46 MB pulled. Same pattern as cls strong-tier (L36 marks-like + L04 dom). Selection-bias defended | `stage4_format_variation_b1_cls_reverse/hidden_states.npz` |
   313	| **P5a** | reddit format variation (cross-site H1) | Myriad **354382** (3rd attempt) | ✅ **done 08:09:38** — shape (430, 37, 2560), 10 modes, 76 MB pulled | `stage4_format_variation_b1_reddit/hidden_states.npz` |
   314	| **P5b** | reddit Method 4.2 multimode (cross-site Mirage) | Myriad 353890 | ✅ **done 07:31:14** — 288 examples, 6 modes, 51 MB pulled | `stage4_multimode_b1_reddit/hidden_states.npz` |
   315	
   316	**P5a bug history** (3 attempts):
   317	1. Myriad 353764 (00:48) — `no hidden states extracted` after 105 task skips. Root cause: hardcoded `classifieds_task_{tid}` prefix in `run_stage4_format_variation_extract.py:177`, archive uses `reddit_task_*`
   318	2. Myriad 353889 (06:26) — same failure, same root cause
   319	3. Myriad **354382** (07:26) — fixed via commit 3d41953 (add `--site reddit` arg, default classifieds for backcompat)
   320	
   321	**P2/P3 deferred** (2026-05-12 00:31 → 06:30, 3 attempts each):
   322	- `snapshot_download` `thread_map` 8-worker concurrent download hits cas-bridge throttling/timeout
   323	- Each attempt: get `HTTP 206 Partial Content` then concurrent.futures `result_iterator` raises (underlying worker exception masked)
   324	- Cleanup 4×2.3G incomplete blobs to reclaim disk
   325	- **Recovery plan**: tomorrow morning, single-thread CLI:
   326	  ```bash
   327	  HF_HUB_DOWNLOAD_TIMEOUT=600 huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --max-workers 1
   328	  HF_HUB_DOWNLOAD_TIMEOUT=600 huggingface-cli download microsoft/Phi-3.5-vision-instruct --max-workers 1
   329	  ```
   330	- Paper §5 generalization claim still defensible via P4 (selection-bias) + P5a/P5b (cross-site). P2/P3 are nice-to-have (family/size triangulation), not paper-critical.
   331	
   332	**Expected verdict matrix** (most paper-grade interesting):
   333	- P3 7B per-task variability < 4B per-task variability → H1' capacity-limit partially confirmed (training-distribution still creates shortcut, but consistency increases with size)
   334	- P2 cross-family dichotomy holds → H1 is cross-family universal training prior
   335	- P4 reverse-tier holds → not tier-selection-bias
   336	- P5a reddit holds → cross-site universal
   337	
   338	### 7.3.0 Exp 1 axis-2 layer profile (2026-05-12 21:00 — three-axis hierarchy)
   339	
   340	`axis2_layer_profile.md` + `fig_axis2_prompt_layer_profile.png`. Re-examine residual stream geometry per axis-isolated pair, full 37-layer cosine curves on `stage4_multimode_b1_{cls,reddit}` (288 ex each).
   341	
   342	Cls site peak layers + magnitudes:
   343	
   344	| Pair | Group | L17 | L23 | L36 | Peak L | Peak gap |
   345	|---|---|---:|---:|---:|---:|---:|
   346	| P-SoM↔SoM (image-axis ref) | axis-3 | 0.0412 | 0.0400 | 0.0411 | **L17** | 0.0412 |
   347	| DOM↔P-text (text fmt) | axis-1 | 0.0120 | 0.0254 | 0.0201 | **L23** | 0.0254 |
   348	| P-prompt↔P-SoM (text fmt) | axis-1 | 0.0113 | 0.0292 | 0.0201 | **L23** | 0.0292 |
   349	| P-text↔P-SoM (prompt fam, flat) | axis-2 | 0.0028 | **0.0114** | 0.0089 | L23 | 0.0114 |
   350	| DOM↔P-prompt (prompt fam, hier) | axis-2 | 0.0013 | 0.0050 | 0.0067 | L36 | 0.0067 |
   351	
   352	Reddit cross-site replicates: P-text↔P-SoM L23 = 0.0098 (vs cls 0.0114), same rank-order, same peak layer.
   353	
   354	**Three regularities**:
   355	1. **Distinct peak layers**: image L17 (fast sharp), text-format L23 (slower late-mid), prompt-family L23 (same timing as text-format on flat-text)
   356	2. **Distinct magnitudes**: image ~0.04, text-format ~0.03, prompt-family ~0.01 — 4:3:1 ratio
   357	3. **Cross-site rank stable**: reddit identical pattern
   358	
   359	**Reframe**: Axis-2 prompt-family is NOT null at residual stream. It's 3-4x weaker than axis-1 + peaks at L23 not L17. Method 4.2 plan §5.1 L17 snapshot 错失它. New paper §5 framing: layered three-axis hierarchy, image-axis dominant at L17 Mirage locus, text-format + prompt-family late-mid build at L23 parallel.
   360	

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/plan.md | sed -n '361,468p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   361	→ Paper §5.7 重写为 "Layered Three-Axis Mechanism Hierarchy" (commit pending).
   362	
   363	### 7.3.0b Axis-2 per-task fragility check (2026-05-12 21:50 — /stress W2 defuse)
   364	
   365	`axis2_per_task_fragility.md` + `fig_axis2_per_task_fragility.png`. /stress reviewer 第一次 invocation W2 attack: 怀疑 axis-2 cosine 0.0114 mean 由 2-3 outlier 主导, 类比 h1_per_task_fragility 11% strict per-task. Defuse 实验:
   366	
   367	| Pair | Site | Mean | Median | IQR | % > 0.010 |
   368	|---|---|---|---|---|---|
   369	| **Axis-2 flat (P-text↔P-SoM)** | cls | 0.0132 | 0.0131 | [0.012, 0.014] | **100%** |
   370	| **Axis-2 flat (P-text↔P-SoM)** | reddit | 0.0121 | 0.0120 | [0.011, 0.013] | **100%** |
   371	| Axis-1 ref (DOM↔P-text) | cls | 0.0287 | 0.0280 | [0.025, 0.031] | 100% |
   372	| Axis-1 ref (DOM↔P-text) | reddit | 0.0260 | 0.0263 | [0.023, 0.031] | 100% |
   373	| Axis-3 image (P-SoM↔SoM) | cls | 0.0407 | 0.0415 | [0.035, 0.044] | 100% |
   374	
   375	**3 findings**:
   376	1. **Mean ≈ median** both sites → distribution **NOT right-skewed**, **NOT outlier-driven**
   377	2. **IQR 极窄** (0.002-0.003 wide), 全部 24 task 在 0.010-0.018 范围, zero outlier
   378	3. **Cross-site rank stable** + magnitude near-identical (0.0132 cls vs 0.0121 reddit, < 9% diff)
   379	
   380	**/stress W2 attack defused completely**: axis-2 cosine gap 是 uniform per-task signature, 不是 aggregate artifact. 这与 H1 binary dichotomy 11% strict per-task fragile 形成对比 — H1 因为问 layer-comparison 离散问题易 fragile, axis-2 cosine 是 continuous mode-pair distance 即使 magnitude 小也 robust per-task.
   381	
   382	**Paper §5.7 增强**: 加入 per-task fragility 段, 明确每个 task 都贡献 axis-2 signal, 不是 2-3 outlier mean artifact.
   383	
   384	### 7.3.0a Exp 3 logit lens 输出层 amplification (2026-05-12 21:02)
   385	
   386	`axis2_logit_lens.md` + `fig_axis2_logit_lens.png`. 应用 Qwen3-VL-4B `model.model.language_model.norm` + `model.lm_head` to per-layer per-mode mean hidden states, 算 KL across 37 层.
   387	
   388	| Pair | Site | Peak L (KL) | Peak KL | Exp 1 cosine peak | 放大倍数 |
   389	|---|---|---|---|---|---|
   390	| P-text↔P-SoM (axis-2 flat) | cls | **L23** | 0.162 | 0.011 | ~14x |
   391	| DOM↔P-prompt (axis-2 hier) | cls | L25 | 0.044 | 0.007 | ~7x |
   392	| DOM↔P-text (axis-1) | cls | L23 | 0.551 | 0.025 | 22x |
   393	| P-prompt↔P-SoM (axis-1) | cls | L23 | 0.695 | 0.029 | 24x |
   394	| Cross-site reddit | | L23-L25 | 0.13-0.62 | preserved | preserved |
   395	
   396	**3 findings**:
   397	1. Axis-2 prompt-family **IS in output distribution** — KL 0.16 at L23, NOT null. Exp 1 cosine 0.011 is not the end of the story.
   398	2. **lm_head 10-25x amplification of cosine → KL** but axis-agnostic ratio preserved (axis-1/axis-2 ratio ~4.3 cls, ~4.9 reddit, vs cosine ratio ~3 — slight amplification of stronger axis but not breaking 3-4x rank).
   399	3. **KL @ L36 ≈ 0 paradox**: 因 mean hidden state at last layer collapse to common JSON format header. Mode-distinct signal concentrated in **L23-L25 decoding window** (not final embedding). This is the "knows but says differently" structural mirror of Wu et al. tool calling.
   400	
   401	**Paper §5.7 follow-up paragraph** added: 三轴 hierarchy persists at output distribution with same rank-order. Deployment routing (paper-2) should treat L23-L25 logit-lens features as cheapest highest-signal mode-axis discriminator.
   402	
   403	### 7.3.1 Reddit cross-site results (2026-05-12 16:30 — P5a + P5b analyses landed)
   404	
   405	**P5a — Format variation H1 test on reddit** (`format_variation_h1_test_reddit.md`):
   406	
   407	| Variant | Peak L (reddit) | Peak L (cls baseline) |
   408	|---|---|---|
   409	| som_standard / browser_use_at / tarsier_typed / xml_tagged | **L17** | L36 (last) |
   410	| appagent_id / plain_numbered | **L04** | L36 |
   411	| hash_id_control | **L04** ✓ (acts as control) | L36 (control failed) |
   412	| plain_sentence | **L17** | L17 |
   413	| dom (baseline) | **L04** ✓ | L04 ✓ |
   414	
   415	**Reddit nuance — cleaner mid-layer fusion**: Reddit 上 marks-like 4/6 真 peak 在 L17 (mid-layer), cls 上 L36 是 monotonic increasing artifact (peak hit boundary). Reddit hash_id_control L04 acts as proper "no integer" control (cls 上失败). Reddit data supports Q5 mid-layer fusion hypothesis better than cls.
   416	
   417	Caveats: small n (24×2=48/mode) makes 2/6 marks-like falling to L04 (appagent_id, plain_numbered) plausible as sampling noise; plain_sentence triggering L17 on reddit (not cls) suggests reddit narrative comments may pattern-match list semantics.
   418	
   419	**P5b — Mirage signature on reddit** (`stage4_method42_results_reddit.md`):
   420	
   421	| Test | Value at L17 | cls baseline |
   422	|---|---|---|
   423	| P-SoM ↔ DOM | **0.0098** (nearly 0) | similar (text-axis sibling) |
   424	| P-SoM ↔ SoM | **0.0423** | similar (image-axis split) |
   425	| P-SoM ↔ Vision | 0.0457 | similar |
   426	| DOM ↔ Vision peak | L04 = 0.0687 (AUROC=1.0) | L04 similar |
   427	
   428	→ **Cross-site Mirage replication ✓**: P-SoM behaves as text-axis sibling of DOM at L17 (image-feature reduction), not as image-axis sibling of SoM. paper §5 4-fold (d) drop-one mechanism holds on reddit.
   429	
   430	**Paper §5 cross-site evidence stack now complete**:
   431	1. P-SoM mid-layer mechanism (4-fold drop-one) — cls + reddit replicated ✓
   432	2. Indexed-list format → shortcut activation — directional consistency cls ↔ reddit ✓
   433	3. Mirage signature geometric structure — cls + reddit replicated ✓
   434	
   435	**P4 selection-bias defense (2026-05-12 18:50)** — cls reverse-tier H1 (`format_variation_h1_test_cls_reverse.md`):
   436	
   437	| Variant | strong-tier cls | reverse-tier cls | reddit |
   438	|---|---|---|---|
   439	| 6 marks-like | L36 monotonic | **L36 monotonic** ✓ same | L17 (4/6 真 peak) |
   440	| hash_id_control | L36 (failed control) | **L36** ✓ same | L04 ✓ proper control |
   441	| plain_sentence | L17 | **L22** close to L17 | L17 |
   442	| dom baseline | L04 ✓ | **L04** ✓ | L04 ✓ |
   443	
   444	H1 mechanism in cls is **not tier selection artifact** (strong vs reverse both replicate). Reddit data paradoxically cleaner reveal of true L17 mid-layer fusion locus (cls L36 is monotonic-boundary artifact).
   445	
   446	### 7.4 Decisions pending
   447	
   448	| Decision | Owner | Trigger |
   449	|---|---|---|
   450	| Co-author multimodal extension vs cite + independent framing | Zekun | After Zekun reply to message |
   451	| Method 4.5 path: LA-HDMI vs SAE | Zekun + advisor sync | After v2 full sweep + Zekun reply |
   452	| Paper §5 prose round | Codex + me | After v2 full + Zekun decision |
   453	
   454	## 8. Roadmap (next 2-4 weeks)
   455	
   456	| Week | Milestone | Deliverable |
   457	|---|---|---|
   458	| **Week 1** (now → 2026-05-18) | v2 full sweep land + Zekun sync + paper §5 prose v1 | 48-cell H-mean table + Zekun message + paper §5 §1-4 prose draft |
   459	| **Week 2** (2026-05-19 → 25) | Cross-site Method 4.2 (reddit) + reverse-tier Method 4.4 | Replication results + paper §5 §5 prose |
   460	| **Week 3** (2026-05-26 → 06-01) | Method 4.5 launch (LA-HDMI or SAE per Zekun decision) | Pilot results + paper §5 §6-7 prose |
   461	| **Week 4** (2026-06-02 → 08) | Paper §5 codex round + advisor review | Submission-ready paper §5 |
   462	
   463	## 9. Connection to paper §1 + §6
   464	
   465	- **§1 phantom routing space + 4-fold drop-in property** — completely independent of mechanism work, anchors Outcome / Macro / Efficiency dimensions. NOT in this folder; see `paper_planning.md` §1
   466	- **§6 cost-aware routing** — Peale et al. 2026 uncertainty decomposition anchor adds theoretical layer to phantom routing space's empirical AUROC. Method 4.2 cosine gap could serve as "reducible uncertainty" signal in deployment
   467	
   468	These two stay outside mechanism folder. Mechanism workspace is paper §5-specific.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section8_limitations.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	\subsection*{8.1 Scope \& external-validity bounds}
     2	
     3	This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. Shopping also has weaker intermediate mechanism coverage than classifieds/reddit, so site-level generalization is reported cell-by-cell rather than averaged into a universal web-agent claim. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
     4	
     5	\subsection*{8.2 Construct validity \& evaluator threats}
     6	
     7	VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, including the audit-derived false-positive filters from Sections 78a and 95, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
     8	
     9	\subsection*{8.3 Internal-validity threats: known scaffold bugs}
    10	
    11	Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
    12	
    13	\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
    14	
    15	The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). It also records two framings that survived audit: the four-fold drop-in property and the sparse L11/L17 mechanism. This affects Sections 1 and 5: the paper can claim confirmed, registry-backed evidence for those two framings, but not a preregistered exact-layer prediction or a universal single-task circuit.
    16	
    17	\subsection*{8.5 Statistical \& methodological limits}
    18	
    19	The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Exclusions are listwise only, at <=5% per cell under the B6 lock, so multiple imputation would add modeling assumptions without materially changing paired denominators. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
    20	
    21	\subsection*{8.6 Sparse-mechanism caveat}
    22	
    23	The activation-patching effect is sparse: at L17, four of five completed cells have median Levenshtein-distance shift equal to zero, with IQRs including zero. The blast radius is the mechanism claim: mean disruption and Holm significance are carried by task subsets, approximately the high-salience-image quarter of the strong-tier cases, while many tasks show no visible continuation change; Cell E random injection, by contrast, destroys outputs broadly. This affects Section 5 by reframing the finding as a task-conditional sparse mechanism rather than a universal mid-layer circuit. The claim remains defensible because Section 5.1 reports per-task scatter/violin views alongside mean bands and because Cell E separates content-specific sparse patching from generic injection damage.
    24	
    25	\subsection*{8.7 Compute, cost, and sustainability bounds}
    26	
    27	Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: the per-cell table reports GPU-hours, proxy-API USD cost, and kg-CO2 estimates with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware, benchmark site, and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates rather than a fine-grained lifecycle assessment.
    28	
    29	<!-- Bibkey audit 2026-05-12: lipton2018troubling, neurips2024checklist, holm1979sequentially, wang2023interpretability, zhang2024patching all verified present in paper.bib. -->
    30	

codex
plan.md 仍然同时保留了 v1 旧叙事和 v2 修正叙事，尤其 §1/§5.4/§7.3 与最新 result summary 冲突。接下来我会找 `method42_metrics_v2.json` 或等价 metrics JSON，确认这不是 markdown 摘要手抄错误。
exec
/bin/bash -lc "find results -maxdepth 4 -type f \\( -name '*method42*json' -o -name 'method42_metrics_v2.json' -o -name 'method42_metrics.json' \\) | sort" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json
results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json
results/mechanistic/stage4_multimode_b1_cls/method42_robustness.json
results/mechanistic/stage4_multimode_b1_reddit/method42_metrics.json
results/mechanistic/stage4_multimode_b1_reddit/method42_metrics_v2.json

exec
/bin/bash -lc "find . -path '*stage4_multimode_b1_*' -o -name 'method42_metrics_v2.json' -o -name 'method42_metrics.json' | sort" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
exec
/bin/bash -lc "rg -n 'method42_metrics|hidden_states_v2|v2_fixed|lototask|auroc_loto|leave' . --glob '"'!docs/checkpoints/codex_outputs/**'"' --glob '"'!docs/checkpoints/process/**'"' --glob '"'!*.npz'"'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
./docs/checkpoints/实验笔记.md:6209:**Bg done-monitor `bh702x73i` failed** (exit 127) due to nested-quoting bug in embedded python one-liner inside curl ntfy; until-loop completed at 21:36 (job left qstat) but auto_pull only fired 22:50 (5min cron lag + queue→pull lag). Manual computation 22:55 yielded above table. Lesson: monitor post-loop bash should be simple stat-only (no embedded python eval) — leave compute to manual / dedicated analysis run.
./docs/checkpoints/实验笔记.md:6641:- Activation patching path-dependent: upstream patch propagates; downstream patch leaves upstream unchanged so subsequent re-encoding nullifies effect
./docs/checkpoints/实验笔记.md:6742:- Fix: leave-one-task-out CV. 输出 schema 加 `auroc_in_sample` + `auroc_lototask`. Smoke test on legacy buggy NPZ: lototask = in_sample = 1.000 (separability 由 prompt + image dominated, 不是 text payload — itself a finding)
./docs/checkpoints/实验笔记.md:6772:- ✅ Bug 3 — `scripts/analysis/stage4_pca_cosine_gap.py` lototask CV (commit `103c560`)
./docs/checkpoints/实验笔记.md:6821:4. **AUROC 1.000 held-out lototask** — survives (Bug 3 fix: 6 modes 仍 perfectly linearly separable, just with sub-permille mean-diff for non-image axes)
./docs/checkpoints/实验笔记.md:6824:> "Three claim layers, distinct evidence types: (1) Linear readability — all 6 modes lototask AUROC 1.000 in residual stream; (2) Geometric magnitude — dominated by image axis (~0.04-0.07), with text-format and prompt-family producing sub-permille separation; (3) Causal patching effect at L11-L17 — 20-30% target displacement under final-token replacement. The disjoint between sub-permille geometric magnitude and 20-30% causal patching effect is the paper-grade-novel finding: residual-stream cosine UNDERESTIMATES causal influence by orders of magnitude. Wang et al. 2023 IOI 'feature encoded ≠ feature used' framework applied to multimodal web agents."
./docs/literature/Dual-Track Agent and Environment Routing A Comprehensive Analysis of State-of-the-Art Computer-Use Systems.md:126:Furthermore, the visual paradigm leaves agents acutely vulnerable to "stochastic UI entropy"—where asynchronous network loading, scroll-fade animations, or delayed popups invalidate the agent's spatial grounding between the moment of perception and the moment of execution. The environment remains passively hostile, and the industry's solution is merely to build a more resilient, albeit slower, agent.
./docs/checkpoints/interview_seonglae_prep.md:55:| P5b reddit Mirage signature replication | `stage4_multimode_b1_reddit/method42_metrics.json` | P-SoM↔DOM L17 = 0.0098 (text-axis sibling), P-SoM↔SoM L17 = 0.0423 (image-axis split), AUROC=1.0 |
./docs/checkpoints/interview_seonglae_prep.md:114:Pick 2-3 max — leave room for natural flow.
./scripts/queues/qsub_stage4_multimode_extract_red_v2.sh:51:OUT_NPZ="$OUT_DIR/hidden_states_v2_fixed.npz"
./docs/literature/动作空间语义冲突与大语言模型缩放悖论：Web Agent 滚动约定失效的深度分析.md:183:这一成就得益于架构层面的深度融合与训练策略的范式转换。首先，Qwen3-VL 引入了“DeepStack”技术以实现多层 ViT 特征与语言层的紧密对齐，并使用“Interleaved-MRoPE”增强了空间和视频动态推理 。更关键的是，在视觉定位（Visual Localization）任务中，Qwen3-VL 彻底抛弃了传统的坐标归一化手段。模型不依赖于充满歧义和跨平台差异的外部坐标系统，而是被训练为使用图像的“实际尺寸比例（Actual Size Scale）”来直接表示包围盒（Bounding Boxes）和特征点 。
./docs/checkpoints/stage4_method42_results_v2_reddit.md:6:**AUROC protocol** (Bug 3 fix, codex methodology audit 2026-05-12): paper-grade metric is `auroc_lototask` = leave-one-task-out cross-validation (fit mode-mean direction on training tasks, score held-out task). `auroc_in_sample` (fit + score on same examples) is reported for descriptive comparison only; treat any in-sample ≥0.95 as expected algebraic separability, NOT held-out linear-readability.
./docs/checkpoints/stage4_method42_results_v2_reddit.md:12:| Mode pair | Peak layer | Cosine gap | AUROC (in-sample) | AUROC (lototask) |
./docs/checkpoints/stage4_method42_results_v2_reddit.md:32:| Mode pair | L17 cosine gap | L17 AUROC in-sample | L17 AUROC lototask |
./docs/literature/Tool Calling is Linearly Readable and Steerable in Language Models.md:15:When a tool-calling agent picks the wrong tool, the failure is invisible until execution: the email gets sent, the meeting gets missed. Probing 12 instruction-tuned models across Gemma 3, Qwen 3, Qwen 2.5, and Llama 3.1 (270M to 27B), we find the identity of the chosen tool is linearly readable and steerable inside the model. Adding the mean-difference between two tools' average internal activations switches which tool the model selects at 77-100% accuracy on name-only single-turn prompts (93-100% at 4B+), and the JSON arguments that follow autoregressively match the new tool's schema, so flipping the name is enough. The same per-tool means also flag likely errors before they happen: on Gemma 3 12B and 27B, queries where the gap between the top-1 and top-2 tool is smallest produce 14-21x more wrong calls than queries with the largest gap. The causal effect concentrates along one direction, the row of the output layer that produces the target tool's first token: a unit vector along it at matched magnitude already reaches 93-100%, while what is left over leaves the choice almost untouched. Activation patching localises this to a small set of mid and late-layer attention heads, and a within-topic probe across 14 same-domain airline tools reaches top-1 61-89% across five 4B-14B models, ruling out the reading that we are just moving the model along a topic axis. Even base models encode the right tool before they can emit it: cosine readout from the internal state recovers 69-82% on BFCL while base generation reaches only 2-10%, suggesting pretraining forms the representation and instruction tuning later wires it to the output. We measure tool identity selection and JSON schema correctness in single-turn fixed-menu settings; multi-turn agentic transfer is more fragile and is discussed in Limitations.  
./docs/checkpoints/paper_planning.md:2307:> "Paper §21 9-cell intervention taxonomy and phantom routing space 4-corner ablation focus on **observation-representation substitution axis** (text payload format × prompt-format expectation × image presence). Industry SDKs (agent-browser, Playwright MCP, Stagehand, Tarsier) additionally apply **action-grammar substitution** (short symbolic commands like `click @7` replacing verbose JSON action schemas) for output-token economy. Our phantom routing space ablation **does not factor this orthogonal axis**: we use VisualWebArena's default verbose action serialization across all 6 modes for consistent ablation control on observation-axis. **This is consistent ablation control on observation axis but leaves action-grammar effect uncharacterized**. Future work extending phantom routing space to action-axis (4-corner observation × 2-corner action grammar = 8-cell extended cube) is left open."
./scripts/queues/qsub_stage4_multimode_extract_cls_v2.sh:13:# Output: hidden_states_v2_fixed.npz (NOT hidden_states.npz; preserve legacy
./scripts/queues/qsub_stage4_multimode_extract_cls_v2.sh:59:OUT_NPZ="$OUT_DIR/hidden_states_v2_fixed.npz"
./docs/literature/phantom_som.md:223:A critical methodological concern for Phantom-SoM is that prompt framing alone can masquerade as multimodal gains. Verma et al. [14] demonstrated that apparent reasoning gains from ReAct prompting are driven mainly by exemplar-query similarity rather than interleaved reasoning traces.
./docs/checkpoints/codex_prompts/refactor_phase1_run_registry.md:269:- **Phase 2** (`make analysis` single entry point): leave as-is, just refactor data layer.
./docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:300:    "intent": "Order a 6 pack of the green chocolate bars. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.",
./docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:340:    "intent": "Can you leave a 2 star review on the most expensive CoComelon blanket set (from the \"Kids' Bedding\" category) saying \"I was expecting more for the price, started to fall apart after a few days\"?",
./docs/literature/5.1/Vision-Language Model Modality Interaction A Comprehensive Analysis of Bidirectional Dominance and Failure Modes.md:9:A foundational question in contemporary multimodal research is how the academic literature from 2023 to 2026 frames the interaction between visual and textual modalities. The evolution of this framing reflects the architectural maturation of VLMs, transitioning from simple projection modules to complex, interleaved cross-attention systems.
./docs/literature/5.1/Examining the Lazy Minimization Hypothesis Scaling Laws, Text-over-Vision Bias, and Routing Dynamics in Vision-Language.md:13:Recent empirical studies, such as the comprehensive analysis by Shukor et al. (2025), have derived explicit scaling laws for VLMs trained from scratch on interleaved image-text, image-caption, and text-only data mixtures. The multimodal cross-entropy loss function is modeled as $L = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$, where $\alpha$ and $\beta$ represent the scaling exponents for parameters and tokens, respectively. The derived exponents reveal that multimodal architectures follow similar macroscopic power laws to LLMs but exhibit critical deviations based on the modality mixture and the architectural fusion strategy.
./docs/analysis/cross_sites/probe_audit_verification.json:1637:            "thought": "The task is to leave a comment with the number of adults in the image. The comment box is already focused and ready for input. I will type '1' as the number of adults and then post the comment.",
./external/visualwebarena/config_files/vwa/test_classifieds.raw.json:7290:        "intent_template": "Help me leave a comment with the title \"Interested\" with the text \"I want to buy this item\", if the item comes with a cable that is able to connect to my USB-C ports, else leave a comment with the same title but with the text \"Do you have a USB-C cable?\".",
./external/visualwebarena/config_files/vwa/test_classifieds.raw.json:7291:        "intent": "Help me leave a comment with the title \"Interested\" with the text \"I want to buy this item\", if the item comes with a cable that is able to connect to my USB-C ports, else leave a comment with the same title but with the text \"Do you have a USB-C cable?\".",
./external/visualwebarena/config_files/vwa/test_classifieds.raw.json:7662:        "intent_template": "Find me the seller's email if the item depicts Elvis Presley's performance without the audience shown, else leave a comment with the title \"Question\" and text \"Do you have one without the audience?\". Also, repeat the seller's email back to him in the comment's text to confirm it.",
./external/visualwebarena/config_files/vwa/test_classifieds.raw.json:7663:        "intent": "Find me the seller's email if the item depicts Elvis Presley's performance without the audience shown, else leave a comment with the title \"Question\" and text \"Do you have one without the audience?\". Also, repeat the seller's email back to him in the comment's text to confirm it.",
./docs/literature/5.1/Cost-Aware Routing for Vision-Language Web Agents An Empirical Analysis of Text-Only Accuracy Retention.md:128:Similarly, the construction of the open 8B parameter vision-language model Idefics2 utilized the OBELICS dataset, comprising 350 million images interleaved with 115 billion text tokens. The massive ratio of text to images during pre-training fundamentally conditions the transformer to prioritize text representations. Furthermore, architectures like VILA actively re-blend text-only instruction data alongside image-text data during instruction fine-tuning to remedy the degradation of text-only tasks, inadvertently boosting the model's text inertia and reinforcing its ability to function cleanly as a text-only web agent.
./docs/literature/5.1/Cost-Aware Routing for Vision-Language Web Agents An Empirical Analysis of Text-Only Accuracy Retention.md:139:The ubiquity of this phenomenon across varied pre-training recipes—whether utilizing contrastive language-image learning, interleaved document training, or native multimodal fusion—confirms that modality asymmetry and text-only resilience are fundamental characteristics of the modern transformer architecture when applied to vision-language integration.
./docs/checkpoints/codex_prompts/rename_phantom_paper_naming.md:64:For in-flight dirs (skipped): leave path strings as-is (`B1_phantom_dom_classifieds_20260429` etc.).
./docs/analysis/cross_sites/codex_audit_reddit.json:501:    "intent": "Find me the most recent animated gif of spiderman in f/memes. Then, leave a comment that describes what the spiderman is doing in the meme.",
./docs/analysis/cross_sites/codex_audit_reddit.json:504:    "reason": "intent 中的 “Find me the most recent animated gif of spiderman in f/memes. Then, leave a comment that describes w” 需要从页面截图、商品图、帖子图或网格布局识别视觉事实；普通 DOM 文本无法稳定提供该信号。"
./docs/checkpoints/codex_prompts/codex_stress_2026-05-13_wrap.md:9:   - method42_metrics_v2 cls + reddit JSON (results/mechanistic/stage4_multimode_b1_*/method42_metrics_v2.json)`, `   - 951d56e analysis(stage4 v2) + §5 surgery three-axis retraction
./docs/checkpoints/codex_prompts/codex_stress_2026-05-13_wrap.md:14:   - 103c560 Bug 3 AUROC lototask
./docs/checkpoints/codex_prompts/codex_stress_2026-05-13_wrap.md:58:   - method42_metrics_v2 cls + reddit JSON (results/mechanistic/stage4_multimode_b1_*/method42_metrics_v2.json)
./docs/checkpoints/codex_prompts/codex_stress_2026-05-13_wrap.md:65:   - 103c560 Bug 3 AUROC lototask
./external/visualwebarena/config_files/vwa/test_shopping.json:9846:        "intent_template": "Order {{product}}. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.",
./external/visualwebarena/config_files/vwa/test_shopping.json:9850:        "intent": "Order a 6 pack of the green chocolate bars. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.",
./external/visualwebarena/config_files/vwa/test_shopping.json:9898:        "intent_template": "Order {{product}}. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.",
./external/visualwebarena/config_files/vwa/test_shopping.json:9902:        "intent": "Order the second thing in the top row. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.",
./external/visualwebarena/config_files/vwa/test_shopping.json:10324:        "intent_template": "Can you order the item on this page that is cheaper between {{item1}} and {{item2}} and just leave the other one in my cart? My size is {{size}}.",
./external/visualwebarena/config_files/vwa/test_shopping.json:10330:        "intent": "Can you order the item on this page that is cheaper between the anime shirt and the orange text top and just leave the other one in my cart? My size is large.",
./external/visualwebarena/config_files/vwa/test_shopping.json:10375:        "intent_template": "Can you order the item on this page that is cheaper between {{item1}} and {{item2}} and just leave the other one in my cart? My size is {{size}}.",
./external/visualwebarena/config_files/vwa/test_shopping.json:10381:        "intent": "Can you order the item on this page that is cheaper between the shirt with a bloody hand (in any color) and the red dress and just leave the other one in my cart? My size is XXL.",
./external/visualwebarena/config_files/vwa/test_shopping.json:10510:        "intent_template": "Can you leave a {{number}} star review on {{product}} saying \"{{review}}\"?",
./external/visualwebarena/config_files/vwa/test_shopping.json:10516:        "intent": "Can you leave a 5 star review on the palette with a flower on it saying \"My daughter absolutely loves it!! Would recommend to anyone\"?",
./external/visualwebarena/config_files/vwa/test_shopping.json:10561:        "intent_template": "Can you leave a {{number}} star review on {{product}} saying \"{{review}}\"?",
./external/visualwebarena/config_files/vwa/test_shopping.json:10567:        "intent": "Can you leave a 2 star review on the most expensive CoComelon blanket set (from the \"Kids' Bedding\" category) saying \"I was expecting more for the price, started to fall apart after a few days\"?",
./external/visualwebarena/config_files/vwa/test_shopping.json:10612:        "intent_template": "Can you leave a {{number}} star review on {{product}} saying \"{{review}}\"?",
./external/visualwebarena/config_files/vwa/test_shopping.json:10618:        "intent": "Can you leave a 4 star review on the most expensive plant that looks like the hands of a clock at 6:40 saying \"I love this plant! It's so unique and I get so many compliments on it! The only downside is that it's a little hard to take care of.\"?",
./docs/checkpoints/pre_run/preregistration.md:204:| **Router train/test split** | 5-fold site-stratified CV on cls+red post-Phase-A task pool, seed=42, min test fold ≥ 40 tasks | Reproducible split via `scripts/analysis/router_split.py` (TBD). **Test fold predictions use ONLY train-fold mode rankings** to prevent oracle leak. Pending advisor 5/5 sync alternative: leave-one-site-out (LOSO) — test cls hold-out trained on red, vice versa |
./docs/checkpoints/pre_run/preregistration.md:261:**Future paper-grade improvement** (deferred to next iteration): full **leave-
./docs/checkpoints/pre_run/preregistration.md:280:   - (8) **Train/test split protocol**: 5-fold site-stratified CV vs leave-one-site-out (LOSO)
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:9846:        "intent_template": "Order {{product}}. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:9850:        "intent": "Order a 6 pack of the green chocolate bars. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:9898:        "intent_template": "Order {{product}}. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:9902:        "intent": "Order the second thing in the top row. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:10324:        "intent_template": "Can you order the item on this page that is cheaper between {{item1}} and {{item2}} and just leave the other one in my cart? My size is {{size}}.",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:10330:        "intent": "Can you order the item on this page that is cheaper between the anime shirt and the orange text top and just leave the other one in my cart? My size is large.",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:10375:        "intent_template": "Can you order the item on this page that is cheaper between {{item1}} and {{item2}} and just leave the other one in my cart? My size is {{size}}.",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:10381:        "intent": "Can you order the item on this page that is cheaper between the shirt with a bloody hand (in any color) and the red dress and just leave the other one in my cart? My size is XXL.",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:10510:        "intent_template": "Can you leave a {{number}} star review on {{product}} saying \"{{review}}\"?",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:10516:        "intent": "Can you leave a 5 star review on the palette with a flower on it saying \"My daughter absolutely loves it!! Would recommend to anyone\"?",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:10561:        "intent_template": "Can you leave a {{number}} star review on {{product}} saying \"{{review}}\"?",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:10567:        "intent": "Can you leave a 2 star review on the most expensive CoComelon blanket set (from the \"Kids' Bedding\" category) saying \"I was expecting more for the price, started to fall apart after a few days\"?",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:10612:        "intent_template": "Can you leave a {{number}} star review on {{product}} saying \"{{review}}\"?",
./external/visualwebarena/config_files/vwa/test_shopping.raw.json:10618:        "intent": "Can you leave a 4 star review on the most expensive plant that looks like the hands of a clock at 6:40 saying \"I love this plant! It's so unique and I get so many compliments on it! The only downside is that it's a little hard to take care of.\"?",
./docs/checkpoints/pre_run/topvenue_constraints.md:89:| D5 | Avoid cherry-picking task subsets or cells | NeurIPS checklist Q1/Q7; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `preregistration.md §4` defines cell inclusion and N floor; `pre_rerun_audit.md §4.8.2` counterfactual cell-removal stability is TBD. Remediation: run leave-one-cell-out decision test; cost 2h. | "All cells meeting locked criteria are included; leave-one-cell-out stability is reported as a falsification check." |
./docs/checkpoints/pre_run/topvenue_constraints.md:115:| F4 | Statistical conclusion validity: report uncertainty and sensitivity to thresholds | NeurIPS checklist Q7; NEEDS_BIB_ENTRY: Cook & Campbell 1979 | ✓ | `scripts/analysis/sensitivity_loo_meta.py` + `docs/analysis/cross_sites/sensitivity_loo_meta.md` (created 2026-05-09): leave-one-cell-out DerSimonian-Laird re-pool for each arm with k≥2 cells. **Finding**: 3→5-mode oracle lift, P-SoM drop-in, P-prompt drop-in are LOO-robust (Holm decision unchanged under any single-cell removal); **P-text drop-in is FRAGILE** — dropping B0 classifieds or B0 reddit flips Holm to NS (p=0.065-0.077). Consistent with primary meta I²=71% (substantial heterogeneity in P-text arm). K-of-N threshold gradient omitted because rule was reframed as secondary transparency in B9 lock; primary detection via random-effects meta is the LOO target. | "Pooled phantom-lift estimates are LOO-robust except P-text drop-in, which depends on B0 cell inclusion; flagged in §8.5 limitations + waits for 16-cell rerun for resolution." |
./docs/checkpoints/pre_run/topvenue_constraints.md:161:13. ⚠️ B8/F4 — Add K±1 and leave-one-cell-out sensitivity tables (0.5 day).
./external/visualwebarena/config_files/vwa/test_reddit.raw.json:2460:        "intent": "Find me the most recent animated gif of spiderman in f/memes. Then, leave a comment that describes what the spiderman is doing in the meme.",
./external/visualwebarena/config_files/vwa/test_reddit.raw.json:2465:            "action": "leave a comment that describes what the spiderman is doing in the meme"
./docs/checkpoints/stage4_method42_results_v2_cls.md:6:**AUROC protocol** (Bug 3 fix, codex methodology audit 2026-05-12): paper-grade metric is `auroc_lototask` = leave-one-task-out cross-validation (fit mode-mean direction on training tasks, score held-out task). `auroc_in_sample` (fit + score on same examples) is reported for descriptive comparison only; treat any in-sample ≥0.95 as expected algebraic separability, NOT held-out linear-readability.
./docs/checkpoints/stage4_method42_results_v2_cls.md:12:| Mode pair | Peak layer | Cosine gap | AUROC (in-sample) | AUROC (lototask) |
./docs/checkpoints/stage4_method42_results_v2_cls.md:32:| Mode pair | L17 cosine gap | L17 AUROC in-sample | L17 AUROC lototask |
./docs/checkpoints/mechanism/plan.md:134:1. **Probe-level** (Method 4.2 PCA cosine gap; AUROC reported as both `in_sample` and held-out `leave-one-task-out` after 2026-05-12 Bug 3 fix; v1 buggy NPZ data invalidated, v2 NPZ in flight: Myriad 359736 cls + 359737 reddit)
./docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:59:## AUROC lototask (held-out, paper-grade Bug 3 fix)
./docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:61:All pairs at all layers report AUROC lototask = 1.000 (perfect held-out linear separability). The modes ARE distinguishable in residual stream; the **magnitude of the mode-mean difference** is just much smaller than v1 claimed.
./docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:114:> 1. **Linear readability** (Method 4.2 lototask AUROC 1.000): all 6 modes linearly separable in residual stream; small cosine magnitudes but reliable classification
./docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:122:- v2 NPZ: `results/mechanistic/stage4_multimode_b1_{cls,reddit}/hidden_states_v2_fixed.npz`
./docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:123:- v2 provenance: `hidden_states_v2_fixed.provenance.json` (git SHA, model revision, task IDs, formatter hash)
./docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:124:- v2 metrics: `/tmp/method42_{cls,red}_v2.{json,md}` (canonical paths after rerun: `results/mechanistic/stage4_multimode_b1_*/method42_metrics.json`)
./docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:74:Activation patching is path-dependent: an upstream patch propagates into all downstream computations, while a downstream patch leaves upstream inputs unchanged so subsequent layers can re-encode the same signal. This is consistent with standard mechanistic-interpretability findings (cf. \citep{wang2023interpretability} IOI circuit: feature *encoded* ≠ feature *used*).
./docs/checkpoints/pre_run/reeval_audit_protocol.md:3:**Purpose**: Codify the audit trail every `make rederive` invocation must leave
./docs/checkpoints/ADVISOR_SYNC.md:147:| **(8)** | **Train/test split protocol** — 5-fold site-stratified CV vs LOSO (leave-one-site-out, 训 cls 测 red, 反之) | **倾向 5-fold site-stratified CV** (k=5, seed=42, min test fold ≥ 40 tasks) | k-fold 比 LOSO 数据效率高 (每 fold 平均 90 训 + 10 测 vs LOSO 200+/234), test power 足 | LOSO 更 reviewer-defensible (cross-site generalization claim 直接) 但 power 弱 |
./docs/checkpoints/paper_drafts/section5_mechanism.md:11:Four mechanism claims organize the evidence (revised 2026-05-12 after v2 NPZ re-extraction; see §5.7 revision note). First, observation modes are **linearly separable** in the residual stream: held-out leave-one-task-out AUROC = 1.000 across all mode pairs and all 37 layers (Method 4.2 v2). Second, the **geometric magnitude** of mode separation is dominated by the image axis (cosine ~0.04-0.07), with text-format and prompt-family axes producing only sub-permille cosine separation; the prior "three quantitatively distinct axes at 4:3:1 ratio" framing was a v1 NPZ artifact and is retracted. Third, SoM-to-no-image activation patching displaces target continuations at mid layers L11-L17 with consistent magnitude across classifieds and reddit (~25% target-overlap drop). The Exp 5 axis-2 prompt-only patching subset (cellhprompt cls + red) shows this displacement persists when text format is held flat, indicating prompt-family carries causal influence despite its near-zero geometric magnitude — a feature *used* without large feature *encoded* magnitude (\citep{wang2023interpretability} signature). Fourth, the shortcut trigger is **better described as the conjunction of integer-indexed marker and markup-sigil leading delimiter** than as an abstract "flat element list" — AXTree hierarchy preserves the early L04 image-axis peak, but so do indexed variants that lack either the integer (e.g., `hash_id_control`) or the sigil (e.g., `appagent_id`, `plain_numbered`); only the conjunction triggers the late shift. This refinement is **exploratory after W6** and awaits held-out falsifiers (`bare_N`, `bracket_no_int`).
./docs/checkpoints/paper_drafts/section5_mechanism.md:17:| Linear readability (held-out AUROC) | Method 4.2 v2 (§5.2, §5.7) | **Strong** — held-out leave-one-task-out AUROC = 1.000 across all 15 mode pairs × all 37 layers on both cls and reddit (Bug 3 fix lototask CV) |
./docs/checkpoints/paper_drafts/section5_mechanism.md:31:The robustness suite passes all five checks in the plan. Label permutation leaves the real AUROC 9.8 standard deviations above the permuted baseline. Per-task analysis is positive for all 24 tasks. Step 2 and step 5 curves are invariant at the mechanism level. The L23 silhouette score is at least 0.5, showing nontrivial clustering. Bootstrap 95% confidence intervals are tight, with widths of roughly 4-15% of the corresponding means.
./docs/checkpoints/paper_drafts/section5_mechanism.md:111:**REVISION NOTE**: Earlier drafts of this section described a "three-axis hierarchy" with image (≈0.041), text-format (≈0.029), and prompt-family (≈0.011) cosine gaps in a clean 4:3:1 magnitude ratio with distinct peak layers (L17/L23/L23). That description came from Method 4.2 hidden states extracted with a buggy `[SOM_MARKS]` regex that dropped 71/72 marks per task; the v1 Stage 4 NPZ contained near-empty 3-line text payloads, and mode-mean cosine gaps for axis-1 and axis-2 were inflated by prompt-template differences rather than text-payload differences. After the Bug 2 fix re-extraction (Myriad 359736 cls + 359737 reddit, NPZ `hidden_states_v2_fixed.npz`), axis-1 and axis-2 cosine peaks collapse to sub-permille and move from L23 to L36 boundary-monotone. The "three quantitatively distinct axes" claim is no longer supported. The revised account below is paper-grade.
./docs/checkpoints/paper_drafts/section5_mechanism.md:113:The pairs are constructed to isolate each axis. Axis-1 (text-format swap, prompt fixed) is measured by DOM↔P-text and P-prompt↔P-SoM. Axis-2 (prompt-family swap, text fixed) is measured by DOM↔P-prompt and P-text↔P-SoM. Image axis is measured by P-SoM↔SoM. All curves are computed on `stage4_multimode_b1_cls/hidden_states_v2_fixed.npz` (144 examples, 37 layers, 6 modes, strong-tier manifest filter, production `[SOM_MARKS]` formatter) and replicated cross-site on the matching reddit run.
./docs/checkpoints/paper_drafts/section5_mechanism.md:129:2. **Text-format and prompt-family axes are linearly readable but geometrically near-zero.** All four non-image pairs (DOM↔P-text, P-prompt↔P-SoM, P-text↔P-SoM, DOM↔P-prompt) have peak cosine gap ≤ 0.009 and rise monotonically to a boundary layer L36 rather than localizing at a mid-layer peak. The held-out leave-one-task-out AUROC remains 1.000 across all pairs and layers, which means the 24 strong-tier tasks ARE perfectly separable along these axes — but the mode-mean difference vector is small. The right reading is that text-format and prompt-family modes carry low-magnitude but high-reliability linear signatures in the residual stream rather than substantial geometric clusters.
./docs/checkpoints/paper_drafts/section5_mechanism.md:135:A held-out leave-one-task-out AUROC of 1.000 across small-magnitude pairs is itself a paper-grade observation. It says the modes are reliably linearly separable at any chosen layer with very small mean-difference vectors, which is a stronger claim about the residual stream than the original "distinct mid-layer peaks" framing. The information capacity of the residual stream to represent observation-mode identity is high; the *magnitude* of the representation is mostly image-driven. This reframing follows the linear-readability framework of \citep{wu2026toolcalling} without the cosine-magnitude overclaim.
./docs/checkpoints/paper_drafts/section3_definition.md:35:**Relative to DOM**, Phantom-SoM is essentially free. The `[SOM_MARKS]` block is produced by a regex filter over the VisualWebArena accessibility-tree text that the DOM baseline already consumes. VWA serializes interactive elements with bracketed numeric IDs of the form `[N] role 'label'`; in our implementation `_extract_text_marks` (see `p79/experiment/som.py`) walks `obs_text` line by line, keeps the lines that match `\[\d+\]`, and returns `(id, label)` pairs that are wrapped in a `[SOM_MARKS] ... [/SOM_MARKS]` block. There is no bounding-box lookup and no image work in this path; bounding boxes are only used by full SoM when drawing numeric labels onto the screenshot. Empirically this leaves text length roughly unchanged: holding the system prompt fixed at the DOM family, median total input is 3437 tokens for DOM versus 3661 for P-text on reddit, and 3008 versus 2948 on classifieds — within ±7% on both sites. The two formats see the same accessibility content; what differs is the surface form (flat indexed list versus nested hierarchy with url/tab metadata). We treat this as a representation property and study its behavioral effect mechanistically in Section 5; for cost accounting the implication is that switching DOM → Phantom-SoM at deployment time costs at most a regex pass over the same observation.
./docs/checkpoints/next_steps.md:47:- ETA 30-90 min A100 / 1.5-2.5h V100 once it leaves qw
./docs/checkpoints/next_steps.md:87:**Effort**: 2-3 days infra. Reserve for paper-2 follow-up unless Method 4.2-4.4 leave open questions.
./docs/analysis/cross_sites/sensitivity_loo_meta.md:69:"The primary phantom-lift estimates survive single-cell removal: the random-effects pooled lift remains significant under Holm at α=0.05 across all leave-one-out perturbations of cells with k≥3. Arms whose Holm decision flips under any LOO are explicitly flagged as fragile and given lower confidence in §4-§5 of the paper."
./docs/analysis/cross_sites/codex_audit_classifieds.json:1425:    "intent": "Help me leave a comment with the title \"Interested\" with the text \"I want to buy this item\", if the item comes with a cable that is able to connect to my USB-C ports, else leave a comment with the same title but with the text \"Do you have a USB-C cable?\".",
./docs/analysis/cross_sites/codex_audit_classifieds.json:1495:    "intent": "Find me the seller's email if the item depicts Elvis Presley's performance without the audience shown, else leave a comment with the title \"Question\" and text \"Do you have one without the audience?\". Also, repeat the seller's email back to him in the comment's text to confirm it.",
./scripts/analysis/figures/fig_forest_drop_one.py:157:    # Determine x range based on data + leave annotation room
./scripts/analysis/figures/fig_forest_drop_one.py:172:        x_max = max(all_ci) + 6.0  # leave room for annotation
./docs/analysis/cross_sites/codex_audit_shopping.json:1740:    "intent": "Order a 6 pack of the green chocolate bars. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.",
./docs/analysis/cross_sites/codex_audit_shopping.json:1747:    "intent": "Order the second thing in the top row. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.",
./docs/analysis/cross_sites/codex_audit_shopping.json:1810:    "intent": "Can you order the item on this page that is cheaper between the anime shirt and the orange text top and just leave the other one in my cart? My size is large.",
./docs/analysis/cross_sites/codex_audit_shopping.json:1817:    "intent": "Can you order the item on this page that is cheaper between the shirt with a bloody hand (in any color) and the red dress and just leave the other one in my cart? My size is XXL.",
./docs/analysis/cross_sites/codex_audit_shopping.json:1838:    "intent": "Can you leave a 5 star review on the palette with a flower on it saying \"My daughter absolutely loves it!! Would recommend to anyone\"?",
./docs/analysis/cross_sites/codex_audit_shopping.json:1841:    "reason": "intent 中的 “Can you leave a 5 star review on the palette with a flower on it saying \"My daughter absolutely love” 需要从页面截图、商品图、帖子图或网格布局识别视觉事实；普通 DOM 文本无法稳定提供该信号。"
./docs/analysis/cross_sites/codex_audit_shopping.json:1845:    "intent": "Can you leave a 2 star review on the most expensive CoComelon blanket set (from the \"Kids' Bedding\" category) saying \"I was expecting more for the price, started to fall apart after a few days\"?",
./docs/analysis/cross_sites/codex_audit_shopping.json:1852:    "intent": "Can you leave a 4 star review on the most expensive plant that looks like the hands of a clock at 6:40 saying \"I love this plant! It's so unique and I get so many compliments on it! The only downside is that it's a little hard to take care of.\"?",
./docs/analysis/cross_sites/codex_audit_shopping.json:1855:    "reason": "intent 中的 “Can you leave a 4 star review on the most expensive plant that looks like the hands of a clock at 6:40 saying \"I love this plant! It's so...” 需要从页面截图、商品图、帖子图或网格布局识别视觉事实；普通 DOM 文本无法稳定提供该信号。"
./p79/agents/proxy_api_agent.py:207:            '- For finish/stop actions, extract the answer from context (do NOT leave answer as "").\n'
./docs/analysis/cross_sites/tier5_evaluator_audit.md:44:2. Active-finish N/A gap. Our filter preserves N/A successes when `agent_finished=True` (`p79/experiment/analysis.py:76-79`). In a 20-row spot check from non-archived `episode_reason_rows.csv`, the false positives removed by adjusted success were all non-finish N/A cases, but an adjusted-true N/A example exists with `final_action_type=finish` and empty `final_answer` for reddit task 31. Static implication: if the model actively emits a finish action but leaves the answer blank or generic, post-hoc `na_fp` does not challenge the GPT judge.
./p79/envs/vwa_wrapper.py:154:        # (httpx/asyncio) can leave a stale loop that causes Playwright sync API to
./results/mechanistic/curate_mirage_b1_classifieds/candidates.md:59:| 45 | 203 | +1.27 | +1 (sn0/sa0 / tn0/ta1) | 0.73 | `'{\n  "thought": "I am on the item detail page for the'` | `'{\n  "thought": "I am on the item page for the Se'` | Help me leave a comment with the title "Interested" with the |
./results/mechanistic/curate_mirage_b1_reddit/candidates.md:70:| -1 | 142 | -1.86 | -1 | 0.14 | `'{\n  "thought": "The comment section is not directly accessib'` | `'{\n  "thought": "The task is to leave a comment with the rele'` | Leave a comment in this post with the release date of the mo |
./scripts/analysis/figures/fig_meta_forest.py:216:    # Set generous x range to leave room for annotations
./results/mechanistic/curate_mirage_b1_reddit/candidates.jsonl:68:{"task_id": 71, "step_idx": 2, "intent": "Find me the most recent animated gif of spiderman in f/memes. Then, leave a comment that describes what the spiderman is doing in the meme.", "source_text": "{\n  \"thought\": \"The current screenshot shows an animated GIF, but it is not Spiderman-related. The task requires finding the most recent animated GIF of Spiderman specifically. Since the search bar is not visible, I need to scroll to locate", "target_text": "{\n  \"thought\": \"The screenshot shows a single animated GIF of Spiderman, which matches the task requirement. Since the task is to find the most recent animated GIF of Spiderman in f/memes, and this is the only item visible,", "source_tokens": [515, 220, 330, 60565, 788, 330, 785, 1482, 36190, 4933, 458, 11371, 46766, 11, 714, 432, 374, 537, 28390, 1515, 13904, 13, 576, 3383, 7460, 9271, 279, 1429, 3213, 11371, 46766, 315, 28390, 1515, 11689, 13, 8704, 279, 2711, 3619, 374, 537, 9434, 11, 358, 1184, 311, 9059, 311, 24523], "target_tokens": [515, 220, 330, 60565, 788, 330, 785, 36190, 4933, 264, 3175, 11371, 46766, 315, 28390, 1515, 11, 892, 9071, 279, 3383, 16287, 13, 8704, 279, 3383, 374, 311, 1477, 279, 1429, 3213, 11371, 46766, 315, 28390, 1515, 304, 282, 3183, 15660, 11, 323, 419, 374, 279, 1172, 1509, 9434, 11], "src_neg": 0, "src_aff": 2, "tgt_neg": 0, "tgt_aff": 2, "token_overlap": 0.18, "divergence": 0.8200000000000001, "mirage_score": 0, "composite": 0.0}
./results/mechanistic/curate_mirage_b1_reddit/candidates.jsonl:128:{"task_id": 141, "step_idx": 2, "intent": "Leave a comment in this post with the release date of the movie in the image in mm-dd-yyyy format.", "source_text": "{\n  \"thought\": \"The task requires leaving a comment with the release date in mm-dd-yyyy format. The poster shows 'AUGUST 11TH 2023', which translates to 08-11-2", "target_text": "{\n  \"thought\": \"The task is to leave a comment with the release date of the movie in mm-dd-yyyy format. The current page shows a movie poster for 'Animal' but does not display the release date. I need to navigate to", "source_tokens": [515, 220, 330, 60565, 788, 330, 785, 3383, 7460, 9380, 264, 3980, 448, 279, 4879, 2400, 304, 9465, 21541, 75391, 3561, 13, 576, 22361, 4933, 364, 32, 2941, 8553, 220, 16, 16, 3617, 220, 17, 15, 17, 18, 516, 892, 46918, 311, 220, 15, 23, 12, 16, 16, 12, 17], "target_tokens": [515, 220, 330, 60565, 788, 330, 785, 3383, 374, 311, 5274, 264, 3980, 448, 279, 4879, 2400, 315, 279, 5700, 304, 9465, 21541, 75391, 3561, 13, 576, 1482, 2150, 4933, 264, 5700, 22361, 369, 364, 41415, 6, 714, 1558, 537, 3037, 279, 4879, 2400, 13, 358, 1184, 311, 20876, 311], "src_neg": 0, "src_aff": 1, "tgt_neg": 1, "tgt_aff": 2, "token_overlap": 0.16, "divergence": 0.84, "mirage_score": 0, "composite": 0.0}
./results/mechanistic/curate_mirage_b1_reddit/candidates.jsonl:129:{"task_id": 142, "step_idx": 2, "intent": "Leave a comment in this post with the release date of the movie in the image in mm-dd-yyyy format.", "source_text": "{\n  \"thought\": \"The comment section is not directly accessible via an element_id, so I need to use coordinate-based interaction to click on the comment section. The comment section is visible at the bottom of the post, and the coordinate (0", "target_text": "{\n  \"thought\": \"The task is to leave a comment with the release date of the movie in mm-dd-yyyy format. The current page is a movie poster post, and there is no visible comment section or release date information. I need to", "source_tokens": [515, 220, 330, 60565, 788, 330, 785, 3980, 3772, 374, 537, 5961, 15614, 4566, 458, 2392, 842, 11, 773, 358, 1184, 311, 990, 16184, 5980, 16230, 311, 4205, 389, 279, 3980, 3772, 13, 576, 3980, 3772, 374, 9434, 518, 279, 5622, 315, 279, 1736, 11, 323, 279, 16184, 320, 15], "target_tokens": [515, 220, 330, 60565, 788, 330, 785, 3383, 374, 311, 5274, 264, 3980, 448, 279, 4879, 2400, 315, 279, 5700, 304, 9465, 21541, 75391, 3561, 13, 576, 1482, 2150, 374, 264, 5700, 22361, 1736, 11, 323, 1052, 374, 902, 9434, 3980, 3772, 476, 4879, 2400, 1995, 13, 358, 1184, 311], "src_neg": 0, "src_aff": 1, "tgt_neg": 1, "tgt_aff": 1, "token_overlap": 0.14, "divergence": 0.86, "mirage_score": -1, "composite": -1.8599999999999999}
./results/mechanistic/curate_mirage_b1_reddit/candidates.jsonl:130:{"task_id": 143, "step_idx": 2, "intent": "Leave a comment in this post with the release date of the movie in the image in mm-dd-yyyy format.", "source_text": "{\n  \"thought\": \"The task requires typing the release date of the movie 'Murder Mystery 2' in mm-dd-yyyy format into the body text box. The body text box (id=1179) is already focused and", "target_text": "{\n  \"thought\": \"The task is to leave a comment with the release date of the movie in mm-dd-yyyy format. The current page is the editing interface for a submission titled 'Official Poster for Murder Mystery 2'. The body text box", "source_tokens": [515, 220, 330, 60565, 788, 330, 785, 3383, 7460, 19496, 279, 4879, 2400, 315, 279, 5700, 364, 59878, 1107, 48005, 220, 17, 6, 304, 9465, 21541, 75391, 3561, 1119, 279, 2487, 1467, 3745, 13, 576, 2487, 1467, 3745, 320, 307, 28, 16, 16, 22, 24, 8, 374, 2669, 10735, 323], "target_tokens": [515, 220, 330, 60565, 788, 330, 785, 3383, 374, 311, 5274, 264, 3980, 448, 279, 4879, 2400, 315, 279, 5700, 304, 9465, 21541, 75391, 3561, 13, 576, 1482, 2150, 374, 279, 15664, 3749, 369, 264, 20503, 24849, 364, 33896, 62386, 369, 51383, 48005, 220, 17, 4427, 576, 2487, 1467, 3745], "src_neg": 0, "src_aff": 0, "tgt_neg": 0, "tgt_aff": 0, "token_overlap": 0.18, "divergence": 0.8200000000000001, "mirage_score": 0, "composite": 0.0}
./results/mechanistic/curate_mirage_b1_reddit/candidates.jsonl:131:{"task_id": 145, "step_idx": 2, "intent": "Leave a comment in this post with the text as the trading loss of the OP in euros.", "source_text": "{\n  \"thought\": \"The task requires leaving a comment with the trading loss in euros. The post has a link to '69 comments', which is visible and indicates where comments can be added. I need to click on this link to open", "target_text": "{\n  \"thought\": \"The task requires leaving a comment with the trading loss in euros. The current page is the post detail page. I need to locate the comment section to leave a comment. The comment section is likely below the post content,", "source_tokens": [515, 220, 330, 60565, 788, 330, 785, 3383, 7460, 9380, 264, 3980, 448, 279, 11133, 4709, 304, 32488, 13, 576, 1736, 702, 264, 2656, 311, 364, 21, 24, 6042, 516, 892, 374, 9434, 323, 14807, 1380, 6042, 646, 387, 3694, 13, 358, 1184, 311, 4205, 389, 419, 2656, 311, 1787], "target_tokens": [515, 220, 330, 60565, 788, 330, 785, 3383, 7460, 9380, 264, 3980, 448, 279, 11133, 4709, 304, 32488, 13, 576, 1482, 2150, 374, 279, 1736, 7716, 2150, 13, 358, 1184, 311, 24523, 279, 3980, 3772, 311, 5274, 264, 3980, 13, 576, 3980, 3772, 374, 4363, 3685, 279, 1736, 2213, 11], "src_neg": 0, "src_aff": 1, "tgt_neg": 0, "tgt_aff": 0, "token_overlap": 0.4, "divergence": 0.6, "mirage_score": -1, "composite": -1.6}
./results/mechanistic/curate_mirage_b1_reddit/candidates.jsonl:133:{"task_id": 147, "step_idx": 2, "intent": "Leave a comment in this post with the text as the number of buns in the image.", "source_text": "{\n  \"thought\": \"I need to click on the '82 comments' link to open the comment section where I can leave a comment with the number of buns in the image.\",\n  \"confidence\": 0.95,\n ", "target_text": "{\n  \"thought\": \"The task requires leaving a comment with the number of buns in the image. The image is visible in the post, but no direct element ID for the comment box is provided. I need to locate the comment section or", "source_tokens": [515, 220, 330, 60565, 788, 330, 40, 1184, 311, 4205, 389, 279, 364, 23, 17, 6042, 6, 2656, 311, 1787, 279, 3980, 3772, 1380, 358, 646, 5274, 264, 3980, 448, 279, 1372, 315, 293, 10862, 304, 279, 2168, 10346, 220, 330, 81929, 788, 220, 15, 13, 24, 20, 345, 220], "target_tokens": [515, 220, 330, 60565, 788, 330, 785, 3383, 7460, 9380, 264, 3980, 448, 279, 1372, 315, 293, 10862, 304, 279, 2168, 13, 576, 2168, 374, 9434, 304, 279, 1736, 11, 714, 902, 2118, 2392, 3034, 369, 279, 3980, 3745, 374, 3897, 13, 358, 1184, 311, 24523, 279, 3980, 3772, 476], "src_neg": 0, "src_aff": 0, "tgt_neg": 0, "tgt_aff": 1, "token_overlap": 0.14, "divergence": 0.86, "mirage_score": 1, "composite": 1.8599999999999999}
./results/mechanistic/curate_mirage_b1_reddit/candidates.jsonl:136:{"task_id": 150, "step_idx": 2, "intent": "Leave a comment in this post with the text as the number of adults in the image.", "source_text": "{\n  \"thought\": \"The task is to leave a comment with the number of adults in the image. The image shows two adults (a man and a woman) and one child, so the number is 2. The comment section is indicated", "target_text": "{\n  \"thought\": \"The task requires leaving a comment with the number of adults in the image. The image is visible in the post, but no direct element ID for commenting is available. I need to locate the comment section or comment input field", "source_tokens": [515, 220, 330, 60565, 788, 330, 785, 3383, 374, 311, 5274, 264, 3980, 448, 279, 1372, 315, 12598, 304, 279, 2168, 13, 576, 2168, 4933, 1378, 12598, 320, 64, 883, 323, 264, 5220, 8, 323, 825, 1682, 11, 773, 279, 1372, 374, 220, 17, 13, 576, 3980, 3772, 374, 16317], "target_tokens": [515, 220, 330, 60565, 788, 330, 785, 3383, 7460, 9380, 264, 3980, 448, 279, 1372, 315, 12598, 304, 279, 2168, 13, 576, 2168, 374, 9434, 304, 279, 1736, 11, 714, 902, 2118, 2392, 3034, 369, 40265, 374, 2500, 13, 358, 1184, 311, 24523, 279, 3980, 3772, 476, 3980, 1946, 2070], "src_neg": 0, "src_aff": 1, "tgt_neg": 0, "tgt_aff": 1, "token_overlap": 0.16, "divergence": 0.84, "mirage_score": 0, "composite": 0.0}
./results/mechanistic/archive_subset_b1_reddit/manifest.json:507:      "source_text": "{\n  \"thought\": \"I need to click on the '82 comments' link to open the comment section where I can leave a comment with the number of buns in the image.\",\n  \"confidence\": 0.95,\n ",
./results/mechanistic/archive_subset_b1_reddit/manifest.json:1812:      "target_text": "{\n  \"thought\": \"The task is to leave a comment with the release date of the movie in mm-dd-yyyy format. The current page is a movie poster post, and there is no visible comment section or release date information. I need to",
./results/mechanistic/archive_subset_b1_reddit/manifest.json:2491:      "target_text": "{\n  \"thought\": \"The task requires leaving a comment with the trading loss in euros. The current page is the post detail page. I need to locate the comment section to leave a comment. The comment section is likely below the post content,",
./scripts/analysis/stage4_pca_cosine_gap.py:14:  - results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json
./scripts/analysis/stage4_pca_cosine_gap.py:31:DEFAULT_OUT_JSON = ROOT / "results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json"
./scripts/analysis/stage4_pca_cosine_gap.py:74:    # Per-mode task_id mapping for leave-one-task-out (Bug 3 fix, codex
./scripts/analysis/stage4_pca_cosine_gap.py:83:    auroc_lototask = np.zeros((len(pairs), n_layers))  # leave-one-task-out CV
./scripts/analysis/stage4_pca_cosine_gap.py:102:                auroc_lototask[pi, L] = np.nan
./scripts/analysis/stage4_pca_cosine_gap.py:108:                auroc_lototask[pi, L] = np.nan
./scripts/analysis/stage4_pca_cosine_gap.py:133:            auroc_lototask[pi, L] = float(np.mean(fold_aurocs)) if fold_aurocs else np.nan
./scripts/analysis/stage4_pca_cosine_gap.py:150:            "auroc_lototask_at_peak": (
./scripts/analysis/stage4_pca_cosine_gap.py:151:                float(auroc_lototask[pi, L])
./scripts/analysis/stage4_pca_cosine_gap.py:152:                if not np.isnan(auroc_lototask[pi, L]) else None
./scripts/analysis/stage4_pca_cosine_gap.py:167:        "pairwise_auroc_lototask": {f"{m1}_vs_{m2}": _nan_to_none(auroc_lototask[pi])
./scripts/analysis/stage4_pca_cosine_gap.py:173:            "same examples (inflated, NOT held-out decodability). auroc_lototask is "
./scripts/analysis/stage4_pca_cosine_gap.py:174:            "leave-one-task-out cross-validation: for each held-out task, fit direction "
./scripts/analysis/stage4_pca_cosine_gap.py:176:            "Report lototask as the paper-grade linear-readability metric; in-sample is "
./scripts/analysis/stage4_pca_cosine_gap.py:186:    plot(cos_gap, auroc_lototask, pairs, pca_var, OUT_FIG)
./scripts/analysis/stage4_pca_cosine_gap.py:199:        "metric is `auroc_lototask` = leave-one-task-out cross-validation (fit mode-mean "
./scripts/analysis/stage4_pca_cosine_gap.py:208:        "| Mode pair | Peak layer | Cosine gap | AUROC (in-sample) | AUROC (lototask) |",
./scripts/analysis/stage4_pca_cosine_gap.py:213:        lototask_val = v.get("auroc_lototask_at_peak")
./scripts/analysis/stage4_pca_cosine_gap.py:214:        lototask_str = f"{lototask_val:.3f}" if lototask_val is not None else "n/a"
./scripts/analysis/stage4_pca_cosine_gap.py:217:            f"{v['auroc_in_sample_at_peak']:.3f} | {lototask_str} |"
./scripts/analysis/stage4_pca_cosine_gap.py:222:    L17_section.append("| Mode pair | L17 cosine gap | L17 AUROC in-sample | L17 AUROC lototask |")
./scripts/analysis/stage4_pca_cosine_gap.py:228:        a_lo = m["pairwise_auroc_lototask"][f"{m1}_vs_{m2}"][17]
./scripts/analysis/sensitivity_loo_meta.py:2:"""F4 sensitivity — leave-one-cell-out (LOO) meta-analysis + threshold gradient.
./scripts/analysis/sensitivity_loo_meta.py:301:        "\"The primary phantom-lift estimates survive single-cell removal: the random-effects pooled lift remains significant under Holm at α=0.05 across all leave-one-out perturbations of cells with k≥3. Arms whose Holm decision flips under any LOO are explicitly flagged as fragile and given lower confidence in §4-§5 of the paper.\"",
./scripts/maintenance/auto_pull_myriad_cell.sh:119:            hidden_states_v2_fixed.npz hidden_states_v2_fixed.provenance.json; do
./scripts/maintenance/annotate_screenshots.py:352:        # Wrap thought to fit image width (leave margin for side strip + padding)
./results/mechanistic/archive_subset_b1_reddit/reddit_task_172/step_002/observation_dom.txt:90:			[3633] heading 'St Mary of the Immaculate Conception on 2nd st told the funeral guests to just park and block the street. People parked, like me, who need to go can’t leave. What can I do? How is this allowed?'
./results/mechanistic/archive_subset_b1_reddit/reddit_task_172/step_002/observation_dom.txt:91:				[3634] link 'St Mary of the Immaculate Conception on 2nd st told the funeral guests to just park and block the street. People parked, like me, who need to go can’t leave. What can I do? How is this allowed?' url: http://100.95.81.103:9999/submission_images/1068f802d488545b8e71e66e6addcafdc9e978844bcfcacaf9b24e6d696dece6.jpg

exec
/bin/bash -lc "jq 'keys' results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
[
  "auroc_protocol_note",
  "modes",
  "n_examples",
  "n_layers",
  "n_modes",
  "n_per_mode",
  "pairwise_auroc_in_sample",
  "pairwise_auroc_lototask",
  "pairwise_cosine_gap",
  "pca_top10_var_ratio",
  "peak_disruption_layers"
]

exec
/bin/bash -lc "jq '.summary | keys' results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 exited 5 in 0ms:
jq: error (at results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json:2110): null (null) has no keys

exec
/bin/bash -lc "jq '.summary | keys' results/mechanistic/stage4_multimode_b1_reddit/method42_metrics_v2.json" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 exited 5 in 0ms:
jq: error (at results/mechanistic/stage4_multimode_b1_reddit/method42_metrics_v2.json:2110): null (null) has no keys

exec
/bin/bash -lc 'ls -lh results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json results/mechanistic/stage4_multimode_b1_reddit/method42_metrics_v2.json' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
-rw-rw-r-- 1 jiaming jiaming 39K May 13 00:02 results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json
-rw-rw-r-- 1 jiaming jiaming 38K May 13 00:02 results/mechanistic/stage4_multimode_b1_reddit/method42_metrics_v2.json

 succeeded in 24971ms:
./results/mechanistic/stage4_multimode_b1_cls
./results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz
./results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz
./results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.provenance.json
./results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json
./results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json
./results/mechanistic/stage4_multimode_b1_cls/method42_robustness.json
./results/mechanistic/stage4_multimode_b1_cls/method44_steering.json
./results/mechanistic/stage4_multimode_b1_cls/method44_v2_sweep.json
./results/mechanistic/stage4_multimode_b1_cls/pilot_summary.md
./results/mechanistic/stage4_multimode_b1_reddit
./results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz
./results/mechanistic/stage4_multimode_b1_reddit/hidden_states_v2_fixed.npz
./results/mechanistic/stage4_multimode_b1_reddit/hidden_states_v2_fixed.provenance.json
./results/mechanistic/stage4_multimode_b1_reddit/method42_metrics.json
./results/mechanistic/stage4_multimode_b1_reddit/method42_metrics_v2.json
./results/mechanistic/stage4_multimode_b1_reddit/pilot_summary.md

exec
/bin/bash -lc "jq '.n_examples, .modes, .peak_disruption_layers' results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
144
[
  "dom",
  "phantom_text",
  "phantom_prompt",
  "phantom_som",
  "som",
  "vision"
]
{
  "dom_vs_phantom_text": {
    "layer": 36,
    "gap": 0.004739046096801758,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "dom_vs_phantom_prompt": {
    "layer": 36,
    "gap": 0.006800293922424316,
    "auroc_in_sample_at_peak": 0.9982638888888888,
    "auroc_lototask_at_peak": 1.0
  },
  "dom_vs_phantom_som": {
    "layer": 36,
    "gap": 0.015192270278930664,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "dom_vs_som": {
    "layer": 36,
    "gap": 0.049560546875,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "dom_vs_vision": {
    "layer": 4,
    "gap": 0.06696361303329468,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_text_vs_phantom_prompt": {
    "layer": 36,
    "gap": 0.008126974105834961,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_text_vs_phantom_som": {
    "layer": 36,
    "gap": 0.008772015571594238,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_text_vs_som": {
    "layer": 36,
    "gap": 0.048807501792907715,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_text_vs_vision": {
    "layer": 4,
    "gap": 0.06024569272994995,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_prompt_vs_phantom_som": {
    "layer": 36,
    "gap": 0.004800975322723389,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_prompt_vs_som": {
    "layer": 36,
    "gap": 0.043919503688812256,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_prompt_vs_vision": {
    "layer": 4,
    "gap": 0.06635665893554688,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_som_vs_som": {
    "layer": 36,
    "gap": 0.04161262512207031,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_som_vs_vision": {
    "layer": 4,
    "gap": 0.05986267328262329,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "som_vs_vision": {
    "layer": 36,
    "gap": 0.02553856372833252,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  }
}

exec
/bin/bash -lc "jq '.n_examples, .modes, .peak_disruption_layers' results/mechanistic/stage4_multimode_b1_reddit/method42_metrics_v2.json" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
144
[
  "dom",
  "phantom_text",
  "phantom_prompt",
  "phantom_som",
  "som",
  "vision"
]
{
  "dom_vs_phantom_text": {
    "layer": 36,
    "gap": 0.0037238001823425293,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "dom_vs_phantom_prompt": {
    "layer": 36,
    "gap": 0.006294667720794678,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "dom_vs_phantom_som": {
    "layer": 36,
    "gap": 0.012172937393188477,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "dom_vs_som": {
    "layer": 4,
    "gap": 0.045457541942596436,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "dom_vs_vision": {
    "layer": 4,
    "gap": 0.0657852292060852,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_text_vs_phantom_prompt": {
    "layer": 36,
    "gap": 0.007367968559265137,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_text_vs_phantom_som": {
    "layer": 36,
    "gap": 0.0069103240966796875,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_text_vs_som": {
    "layer": 17,
    "gap": 0.043307721614837646,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_text_vs_vision": {
    "layer": 4,
    "gap": 0.05895358324050903,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_prompt_vs_phantom_som": {
    "layer": 36,
    "gap": 0.004175841808319092,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_prompt_vs_som": {
    "layer": 4,
    "gap": 0.04338341951370239,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_prompt_vs_vision": {
    "layer": 4,
    "gap": 0.06336647272109985,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_som_vs_som": {
    "layer": 4,
    "gap": 0.038576483726501465,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "phantom_som_vs_vision": {
    "layer": 4,
    "gap": 0.05861365795135498,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  },
  "som_vs_vision": {
    "layer": 36,
    "gap": 0.019315242767333984,
    "auroc_in_sample_at_peak": 1.0,
    "auroc_lototask_at_peak": 1.0
  }
}

exec
/bin/bash -lc "jq -r '.peak_disruption_layers | to_entries[] | [.key, .value.peak_layer, .value.peak_cosine_gap, .value.auroc_lototask_at_peak] | @tsv' results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
dom_vs_phantom_text			1.0
dom_vs_phantom_prompt			1.0
dom_vs_phantom_som			1.0
dom_vs_som			1.0
dom_vs_vision			1.0
phantom_text_vs_phantom_prompt			1.0
phantom_text_vs_phantom_som			1.0
phantom_text_vs_som			1.0
phantom_text_vs_vision			1.0
phantom_prompt_vs_phantom_som			1.0
phantom_prompt_vs_som			1.0
phantom_prompt_vs_vision			1.0
phantom_som_vs_som			1.0
phantom_som_vs_vision			1.0
som_vs_vision			1.0

exec
/bin/bash -lc "jq -r '.peak_disruption_layers | to_entries[] | [.key, .value.peak_layer, .value.peak_cosine_gap, .value.auroc_lototask_at_peak] | @tsv' results/mechanistic/stage4_multimode_b1_reddit/method42_metrics_v2.json" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
dom_vs_phantom_text			1.0
dom_vs_phantom_prompt			1.0
dom_vs_phantom_som			1.0
dom_vs_som			1.0
dom_vs_vision			1.0
phantom_text_vs_phantom_prompt			1.0
phantom_text_vs_phantom_som			1.0
phantom_text_vs_som			1.0
phantom_text_vs_vision			1.0
phantom_prompt_vs_phantom_som			1.0
phantom_prompt_vs_som			1.0
phantom_prompt_vs_vision			1.0
phantom_som_vs_som			1.0
phantom_som_vs_vision			1.0
som_vs_vision			1.0

exec
/bin/bash -lc "nl -ba results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json | sed -n '1,90p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	{
     2	  "n_examples": 144,
     3	  "n_layers": 37,
     4	  "n_modes": 6,
     5	  "modes": [
     6	    "dom",
     7	    "phantom_text",
     8	    "phantom_prompt",
     9	    "phantom_som",
    10	    "som",
    11	    "vision"
    12	  ],
    13	  "n_per_mode": {
    14	    "dom": 24,
    15	    "phantom_text": 24,
    16	    "phantom_prompt": 24,
    17	    "phantom_som": 24,
    18	    "som": 24,
    19	    "vision": 24
    20	  },
    21	  "pairwise_cosine_gap": {
    22	    "dom_vs_phantom_text": [
    23	      5.960464477539063e-08,
    24	      0.002046644687652588,
    25	      0.0027999281883239746,
    26	      0.004214763641357422,
    27	      0.0035285353660583496,
    28	      0.0028175711631774902,
    29	      0.0018024444580078125,
    30	      0.0014406442642211914,
    31	      0.0015925765037536621,
    32	      0.0017735958099365234,
    33	      0.0016914606094360352,
    34	      0.0017055869102478027,
    35	      0.0018175840377807617,
    36	      0.001987278461456299,
    37	      0.0022485852241516113,
    38	      0.002055227756500244,
    39	      0.002331852912902832,
    40	      0.002080678939819336,
    41	      0.0018720626831054688,
    42	      0.0013080239295959473,
    43	      0.0016477704048156738,
    44	      0.001543879508972168,
    45	      0.001868903636932373,
    46	      0.001761794090270996,
    47	      0.0012864470481872559,
    48	      0.0014355182647705078,
    49	      0.00172346830368042,
    50	      0.0017182230949401855,
    51	      0.0016736388206481934,
    52	      0.0014109015464782715,
    53	      0.0013828277587890625,
    54	      0.0015685558319091797,
    55	      0.0018331408500671387,
    56	      0.0022078752517700195,
    57	      0.0016737580299377441,
    58	      0.0013565421104431152,
    59	      0.004739046096801758
    60	    ],
    61	    "dom_vs_phantom_prompt": [
    62	      5.960464477539063e-08,
    63	      3.0338764190673828e-05,
    64	      2.9385089874267578e-05,
    65	      8.052587509155273e-05,
    66	      0.0001703500747680664,
    67	      0.00021904706954956055,
    68	      0.00016623735427856445,
    69	      0.0001552104949951172,
    70	      0.00019854307174682617,
    71	      0.00030291080474853516,
    72	      0.0005925893783569336,
    73	      0.0005764365196228027,
    74	      0.0006685853004455566,
    75	      0.0009312629699707031,
    76	      0.0009704232215881348,
    77	      0.0009868144989013672,
    78	      0.0010818839073181152,
    79	      0.0012776851654052734,
    80	      0.0018197894096374512,
    81	      0.001722097396850586,
    82	      0.0024938583374023438,
    83	      0.0038318634033203125,
    84	      0.004212617874145508,
    85	      0.003896653652191162,
    86	      0.0036025047302246094,
    87	      0.004343807697296143,
    88	      0.003748178482055664,
    89	      0.004007279872894287,
    90	      0.003705739974975586,

exec
/bin/bash -lc "nl -ba results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json | sed -n '1960,2045p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
  1960	      0.8462987542152405,
  1961	      0.8561820983886719,
  1962	      0.8521336317062378,
  1963	      0.859592854976654,
  1964	      0.8630573749542236,
  1965	      0.8745505213737488,
  1966	      0.8730983734130859,
  1967	      0.8690387010574341,
  1968	      0.866652250289917,
  1969	      0.8723819851875305,
  1970	      0.8719677925109863,
  1971	      0.8712530732154846,
  1972	      0.8723264932632446,
  1973	      0.8690083026885986,
  1974	      0.8672359585762024,
  1975	      0.8648744821548462,
  1976	      0.8618470430374146
  1977	    ],
  1978	    "vision": [
  1979	      NaN,
  1980	      0.930424690246582,
  1981	      0.9192635416984558,
  1982	      0.8754444718360901,
  1983	      0.8715399503707886,
  1984	      0.8656824827194214,
  1985	      0.857537031173706,
  1986	      0.8601645827293396,
  1987	      0.8512057065963745,
  1988	      0.8384870290756226,
  1989	      0.8304280638694763,
  1990	      0.8357991576194763,
  1991	      0.8244218826293945,
  1992	      0.8114372491836548,
  1993	      0.8260607719421387,
  1994	      0.8330323100090027,
  1995	      0.8226052522659302,
  1996	      0.8183138370513916,
  1997	      0.8142751455307007,
  1998	      0.8247533440589905,
  1999	      0.8240678310394287,
  2000	      0.8384198546409607,
  2001	      0.8457760214805603,
  2002	      0.8567062616348267,
  2003	      0.861272394657135,
  2004	      0.8666204214096069,
  2005	      0.86921626329422,
  2006	      0.8671504855155945,
  2007	      0.8678421974182129,
  2008	      0.873802125453949,
  2009	      0.8742681741714478,
  2010	      0.8723499178886414,
  2011	      0.8718864321708679,
  2012	      0.8698292970657349,
  2013	      0.8671164512634277,
  2014	      0.8649146556854248,
  2015	      0.8591932654380798
  2016	    ]
  2017	  },
  2018	  "peak_disruption_layers": {
  2019	    "dom_vs_phantom_text": {
  2020	      "layer": 36,
  2021	      "gap": 0.004739046096801758,
  2022	      "auroc_in_sample_at_peak": 1.0,
  2023	      "auroc_lototask_at_peak": 1.0
  2024	    },
  2025	    "dom_vs_phantom_prompt": {
  2026	      "layer": 36,
  2027	      "gap": 0.006800293922424316,
  2028	      "auroc_in_sample_at_peak": 0.9982638888888888,
  2029	      "auroc_lototask_at_peak": 1.0
  2030	    },
  2031	    "dom_vs_phantom_som": {
  2032	      "layer": 36,
  2033	      "gap": 0.015192270278930664,
  2034	      "auroc_in_sample_at_peak": 1.0,
  2035	      "auroc_lototask_at_peak": 1.0
  2036	    },
  2037	    "dom_vs_som": {
  2038	      "layer": 36,
  2039	      "gap": 0.049560546875,
  2040	      "auroc_in_sample_at_peak": 1.0,
  2041	      "auroc_lototask_at_peak": 1.0
  2042	    },
  2043	    "dom_vs_vision": {
  2044	      "layer": 4,
  2045	      "gap": 0.06696361303329468,

exec
/bin/bash -lc "nl -ba results/mechanistic/stage4_multimode_b1_reddit/method42_metrics_v2.json | sed -n '1960,2045p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
  1960	      0.8618366718292236,
  1961	      0.8643366098403931,
  1962	      0.8594565391540527,
  1963	      0.857548177242279,
  1964	      0.8644254207611084,
  1965	      0.8565924167633057,
  1966	      0.8566282987594604,
  1967	      0.8552062511444092,
  1968	      0.8550401329994202,
  1969	      0.8592702746391296,
  1970	      0.8649420142173767,
  1971	      0.862382173538208,
  1972	      0.867279589176178,
  1973	      0.865541934967041,
  1974	      0.8653623461723328,
  1975	      0.8610185384750366,
  1976	      0.8596082329750061
  1977	    ],
  1978	    "vision": [
  1979	      NaN,
  1980	      0.9561288356781006,
  1981	      0.9632111191749573,
  1982	      0.9396382570266724,
  1983	      0.9391926527023315,
  1984	      0.9282929301261902,
  1985	      0.9256930947303772,
  1986	      0.9237704873085022,
  1987	      0.9184525609016418,
  1988	      0.9130586981773376,
  1989	      0.9045954942703247,
  1990	      0.8971601128578186,
  1991	      0.8879874348640442,
  1992	      0.8714554905891418,
  1993	      0.8674275875091553,
  1994	      0.862150251865387,
  1995	      0.8498881459236145,
  1996	      0.8310487866401672,
  1997	      0.8305287957191467,
  1998	      0.8426793813705444,
  1999	      0.8309844732284546,
  2000	      0.8377482891082764,
  2001	      0.8446152806282043,
  2002	      0.8585570454597473,
  2003	      0.8598995208740234,
  2004	      0.8578557372093201,
  2005	      0.8619321584701538,
  2006	      0.8608631491661072,
  2007	      0.858269214630127,
  2008	      0.8648484349250793,
  2009	      0.865086555480957,
  2010	      0.8645640015602112,
  2011	      0.8671666383743286,
  2012	      0.862410843372345,
  2013	      0.860081672668457,
  2014	      0.855423629283905,
  2015	      0.853868305683136
  2016	    ]
  2017	  },
  2018	  "peak_disruption_layers": {
  2019	    "dom_vs_phantom_text": {
  2020	      "layer": 36,
  2021	      "gap": 0.0037238001823425293,
  2022	      "auroc_in_sample_at_peak": 1.0,
  2023	      "auroc_lototask_at_peak": 1.0
  2024	    },
  2025	    "dom_vs_phantom_prompt": {
  2026	      "layer": 36,
  2027	      "gap": 0.006294667720794678,
  2028	      "auroc_in_sample_at_peak": 1.0,
  2029	      "auroc_lototask_at_peak": 1.0
  2030	    },
  2031	    "dom_vs_phantom_som": {
  2032	      "layer": 36,
  2033	      "gap": 0.012172937393188477,
  2034	      "auroc_in_sample_at_peak": 1.0,
  2035	      "auroc_lototask_at_peak": 1.0
  2036	    },
  2037	    "dom_vs_som": {
  2038	      "layer": 4,
  2039	      "gap": 0.045457541942596436,
  2040	      "auroc_in_sample_at_peak": 1.0,
  2041	      "auroc_lototask_at_peak": 1.0
  2042	    },
  2043	    "dom_vs_vision": {
  2044	      "layer": 4,
  2045	      "gap": 0.0657852292060852,

exec
/bin/bash -lc "nl -ba results/mechanistic/stage4_multimode_b1_cls/method42_metrics_v2.json | sed -n '2046,2115p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
  2046	      "auroc_in_sample_at_peak": 1.0,
  2047	      "auroc_lototask_at_peak": 1.0
  2048	    },
  2049	    "phantom_text_vs_phantom_prompt": {
  2050	      "layer": 36,
  2051	      "gap": 0.008126974105834961,
  2052	      "auroc_in_sample_at_peak": 1.0,
  2053	      "auroc_lototask_at_peak": 1.0
  2054	    },
  2055	    "phantom_text_vs_phantom_som": {
  2056	      "layer": 36,
  2057	      "gap": 0.008772015571594238,
  2058	      "auroc_in_sample_at_peak": 1.0,
  2059	      "auroc_lototask_at_peak": 1.0
  2060	    },
  2061	    "phantom_text_vs_som": {
  2062	      "layer": 36,
  2063	      "gap": 0.048807501792907715,
  2064	      "auroc_in_sample_at_peak": 1.0,
  2065	      "auroc_lototask_at_peak": 1.0
  2066	    },
  2067	    "phantom_text_vs_vision": {
  2068	      "layer": 4,
  2069	      "gap": 0.06024569272994995,
  2070	      "auroc_in_sample_at_peak": 1.0,
  2071	      "auroc_lototask_at_peak": 1.0
  2072	    },
  2073	    "phantom_prompt_vs_phantom_som": {
  2074	      "layer": 36,
  2075	      "gap": 0.004800975322723389,
  2076	      "auroc_in_sample_at_peak": 1.0,
  2077	      "auroc_lototask_at_peak": 1.0
  2078	    },
  2079	    "phantom_prompt_vs_som": {
  2080	      "layer": 36,
  2081	      "gap": 0.043919503688812256,
  2082	      "auroc_in_sample_at_peak": 1.0,
  2083	      "auroc_lototask_at_peak": 1.0
  2084	    },
  2085	    "phantom_prompt_vs_vision": {
  2086	      "layer": 4,
  2087	      "gap": 0.06635665893554688,
  2088	      "auroc_in_sample_at_peak": 1.0,
  2089	      "auroc_lototask_at_peak": 1.0
  2090	    },
  2091	    "phantom_som_vs_som": {
  2092	      "layer": 36,
  2093	      "gap": 0.04161262512207031,
  2094	      "auroc_in_sample_at_peak": 1.0,
  2095	      "auroc_lototask_at_peak": 1.0
  2096	    },
  2097	    "phantom_som_vs_vision": {
  2098	      "layer": 4,
  2099	      "gap": 0.05986267328262329,
  2100	      "auroc_in_sample_at_peak": 1.0,
  2101	      "auroc_lototask_at_peak": 1.0
  2102	    },
  2103	    "som_vs_vision": {
  2104	      "layer": 36,
  2105	      "gap": 0.02553856372833252,
  2106	      "auroc_in_sample_at_peak": 1.0,
  2107	      "auroc_lototask_at_peak": 1.0
  2108	    }
  2109	  },
  2110	  "auroc_protocol_note": "auroc_in_sample fits mode-mean direction on all examples and scores those same examples (inflated, NOT held-out decodability). auroc_lototask is leave-one-task-out cross-validation: for each held-out task, fit direction on the remaining tasks' means, then score the held-out task's examples. Report lototask as the paper-grade linear-readability metric; in-sample is kept for descriptive comparison only. Bug 3 fix per codex methodology audit 2026-05-12."
  2111	}

exec
/bin/bash -lc "nl -ba results/mechanistic/stage4_multimode_b1_reddit/method42_metrics_v2.json | sed -n '2046,2115p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
  2046	      "auroc_in_sample_at_peak": 1.0,
  2047	      "auroc_lototask_at_peak": 1.0
  2048	    },
  2049	    "phantom_text_vs_phantom_prompt": {
  2050	      "layer": 36,
  2051	      "gap": 0.007367968559265137,
  2052	      "auroc_in_sample_at_peak": 1.0,
  2053	      "auroc_lototask_at_peak": 1.0
  2054	    },
  2055	    "phantom_text_vs_phantom_som": {
  2056	      "layer": 36,
  2057	      "gap": 0.0069103240966796875,
  2058	      "auroc_in_sample_at_peak": 1.0,
  2059	      "auroc_lototask_at_peak": 1.0
  2060	    },
  2061	    "phantom_text_vs_som": {
  2062	      "layer": 17,
  2063	      "gap": 0.043307721614837646,
  2064	      "auroc_in_sample_at_peak": 1.0,
  2065	      "auroc_lototask_at_peak": 1.0
  2066	    },
  2067	    "phantom_text_vs_vision": {
  2068	      "layer": 4,
  2069	      "gap": 0.05895358324050903,
  2070	      "auroc_in_sample_at_peak": 1.0,
  2071	      "auroc_lototask_at_peak": 1.0
  2072	    },
  2073	    "phantom_prompt_vs_phantom_som": {
  2074	      "layer": 36,
  2075	      "gap": 0.004175841808319092,
  2076	      "auroc_in_sample_at_peak": 1.0,
  2077	      "auroc_lototask_at_peak": 1.0
  2078	    },
  2079	    "phantom_prompt_vs_som": {
  2080	      "layer": 4,
  2081	      "gap": 0.04338341951370239,
  2082	      "auroc_in_sample_at_peak": 1.0,
  2083	      "auroc_lototask_at_peak": 1.0
  2084	    },
  2085	    "phantom_prompt_vs_vision": {
  2086	      "layer": 4,
  2087	      "gap": 0.06336647272109985,
  2088	      "auroc_in_sample_at_peak": 1.0,
  2089	      "auroc_lototask_at_peak": 1.0
  2090	    },
  2091	    "phantom_som_vs_som": {
  2092	      "layer": 4,
  2093	      "gap": 0.038576483726501465,
  2094	      "auroc_in_sample_at_peak": 1.0,
  2095	      "auroc_lototask_at_peak": 1.0
  2096	    },
  2097	    "phantom_som_vs_vision": {
  2098	      "layer": 4,
  2099	      "gap": 0.05861365795135498,
  2100	      "auroc_in_sample_at_peak": 1.0,
  2101	      "auroc_lototask_at_peak": 1.0
  2102	    },
  2103	    "som_vs_vision": {
  2104	      "layer": 36,
  2105	      "gap": 0.019315242767333984,
  2106	      "auroc_in_sample_at_peak": 1.0,
  2107	      "auroc_lototask_at_peak": 1.0
  2108	    }
  2109	  },
  2110	  "auroc_protocol_note": "auroc_in_sample fits mode-mean direction on all examples and scores those same examples (inflated, NOT held-out decodability). auroc_lototask is leave-one-task-out cross-validation: for each held-out task, fit direction on the remaining tasks' means, then score the held-out task's examples. Report lototask as the paper-grade linear-readability metric; in-sample is kept for descriptive comparison only. Bug 3 fix per codex methodology audit 2026-05-12."
  2111	}

exec
/bin/bash -lc "nl -ba docs/checkpoints/stage4_method42_results_v2_reddit.md | sed -n '1,80p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4 Method 4.2: PCA Cosine Gap Analysis
     2	
     3	**Data**: 144 examples × 37 layers × 6 modes (Qwen3-VL-4B B1 cls)
     4	**Per-mode n**: DOM=24, P-text=24, P-prompt=24, P-SoM=24, SoM=24, Vision=24
     5	
     6	**AUROC protocol** (Bug 3 fix, codex methodology audit 2026-05-12): paper-grade metric is `auroc_lototask` = leave-one-task-out cross-validation (fit mode-mean direction on training tasks, score held-out task). `auroc_in_sample` (fit + score on same examples) is reported for descriptive comparison only; treat any in-sample ≥0.95 as expected algebraic separability, NOT held-out linear-readability.
     7	
     8	## Peak disruption layer per mode pair
     9	
    10	Sorted by cosine gap magnitude (= geometric distance between mode means in hidden space):
    11	
    12	| Mode pair | Peak layer | Cosine gap | AUROC (in-sample) | AUROC (lototask) |
    13	|---|---|---|---|---|
    14	| DOM vs Vision | L04 | 0.0658 | 1.000 | 1.000 |
    15	| P-prompt vs Vision | L04 | 0.0634 | 1.000 | 1.000 |
    16	| P-text vs Vision | L04 | 0.0590 | 1.000 | 1.000 |
    17	| P-SoM vs Vision | L04 | 0.0586 | 1.000 | 1.000 |
    18	| DOM vs SoM | L04 | 0.0455 | 1.000 | 1.000 |
    19	| P-prompt vs SoM | L04 | 0.0434 | 1.000 | 1.000 |
    20	| P-text vs SoM | L17 | 0.0433 | 1.000 | 1.000 |
    21	| P-SoM vs SoM | L04 | 0.0386 | 1.000 | 1.000 |
    22	| SoM vs Vision | L36 | 0.0193 | 1.000 | 1.000 |
    23	| DOM vs P-SoM | L36 | 0.0122 | 1.000 | 1.000 |
    24	| P-text vs P-prompt | L36 | 0.0074 | 1.000 | 1.000 |
    25	| P-text vs P-SoM | L36 | 0.0069 | 1.000 | 1.000 |
    26	| DOM vs P-prompt | L36 | 0.0063 | 1.000 | 1.000 |
    27	| P-prompt vs P-SoM | L36 | 0.0042 | 1.000 | 1.000 |
    28	| DOM vs P-text | L36 | 0.0037 | 1.000 | 1.000 |
    29	
    30	## L17 cosine gap snapshot (paper §5 disruption locus)
    31	
    32	| Mode pair | L17 cosine gap | L17 AUROC in-sample | L17 AUROC lototask |
    33	|---|---|---|---|
    34	| DOM vs P-text | 0.0019 | 1.000 | 1.000 |
    35	| DOM vs P-prompt | 0.0015 | 1.000 | 1.000 |
    36	| DOM vs P-SoM | 0.0031 | 1.000 | 1.000 |
    37	| DOM vs SoM | 0.0450 | 1.000 | 1.000 |
    38	| DOM vs Vision | 0.0537 | 1.000 | 1.000 |
    39	| P-text vs P-prompt | 0.0032 | 1.000 | 1.000 |
    40	| P-text vs P-SoM | 0.0020 | 1.000 | 1.000 |
    41	| P-text vs SoM | 0.0433 | 1.000 | 1.000 |
    42	| P-text vs Vision | 0.0513 | 1.000 | 1.000 |
    43	| P-prompt vs P-SoM | 0.0016 | 1.000 | 1.000 |
    44	| P-prompt vs SoM | 0.0392 | 1.000 | 1.000 |
    45	| P-prompt vs Vision | 0.0492 | 1.000 | 1.000 |
    46	| P-SoM vs SoM | 0.0367 | 1.000 | 1.000 |
    47	| P-SoM vs Vision | 0.0468 | 1.000 | 1.000 |
    48	| SoM vs Vision | 0.0130 | 1.000 | 1.000 |
    49	
    50	## P-SoM vs baseline modes (paper §5 HERO arm)
    51	
    52	P-SoM identity test: is P-SoM closer to SoM (prompt-axis sibling) or DOM (text-axis sibling)?
    53	
    54	| L | P-SoM↔DOM | P-SoM↔SoM | P-SoM↔Vision | P-SoM↔P-text | P-SoM↔P-prompt |
    55	|---|---|---|---|---|---|
    56	| L00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
    57	| L08 | 0.0016 | 0.0241 | 0.0387 | 0.0003 | 0.0014 |
    58	| L11 | 0.0023 | 0.0323 | 0.0407 | 0.0011 | 0.0014 |
    59	| L17 | 0.0031 | 0.0367 | 0.0468 | 0.0020 | 0.0016 |
    60	| L24 | 0.0055 | 0.0190 | 0.0257 | 0.0041 | 0.0016 |
    61	| L30 | 0.0051 | 0.0147 | 0.0193 | 0.0035 | 0.0012 |
    62	| L36 | 0.0122 | 0.0316 | 0.0429 | 0.0069 | 0.0042 |

exec
/bin/bash -lc "nl -ba docs/checkpoints/stage4_method42_results_v2_cls.md | sed -n '1,80p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4 Method 4.2: PCA Cosine Gap Analysis
     2	
     3	**Data**: 144 examples × 37 layers × 6 modes (Qwen3-VL-4B B1 cls)
     4	**Per-mode n**: DOM=24, P-text=24, P-prompt=24, P-SoM=24, SoM=24, Vision=24
     5	
     6	**AUROC protocol** (Bug 3 fix, codex methodology audit 2026-05-12): paper-grade metric is `auroc_lototask` = leave-one-task-out cross-validation (fit mode-mean direction on training tasks, score held-out task). `auroc_in_sample` (fit + score on same examples) is reported for descriptive comparison only; treat any in-sample ≥0.95 as expected algebraic separability, NOT held-out linear-readability.
     7	
     8	## Peak disruption layer per mode pair
     9	
    10	Sorted by cosine gap magnitude (= geometric distance between mode means in hidden space):
    11	
    12	| Mode pair | Peak layer | Cosine gap | AUROC (in-sample) | AUROC (lototask) |
    13	|---|---|---|---|---|
    14	| DOM vs Vision | L04 | 0.0670 | 1.000 | 1.000 |
    15	| P-prompt vs Vision | L04 | 0.0664 | 1.000 | 1.000 |
    16	| P-text vs Vision | L04 | 0.0602 | 1.000 | 1.000 |
    17	| P-SoM vs Vision | L04 | 0.0599 | 1.000 | 1.000 |
    18	| DOM vs SoM | L36 | 0.0496 | 1.000 | 1.000 |
    19	| P-text vs SoM | L36 | 0.0488 | 1.000 | 1.000 |
    20	| P-prompt vs SoM | L36 | 0.0439 | 1.000 | 1.000 |
    21	| P-SoM vs SoM | L36 | 0.0416 | 1.000 | 1.000 |
    22	| SoM vs Vision | L36 | 0.0255 | 1.000 | 1.000 |
    23	| DOM vs P-SoM | L36 | 0.0152 | 1.000 | 1.000 |
    24	| P-text vs P-SoM | L36 | 0.0088 | 1.000 | 1.000 |
    25	| P-text vs P-prompt | L36 | 0.0081 | 1.000 | 1.000 |
    26	| DOM vs P-prompt | L36 | 0.0068 | 0.998 | 1.000 |
    27	| P-prompt vs P-SoM | L36 | 0.0048 | 1.000 | 1.000 |
    28	| DOM vs P-text | L36 | 0.0047 | 1.000 | 1.000 |
    29	
    30	## L17 cosine gap snapshot (paper §5 disruption locus)
    31	
    32	| Mode pair | L17 cosine gap | L17 AUROC in-sample | L17 AUROC lototask |
    33	|---|---|---|---|
    34	| DOM vs P-text | 0.0021 | 1.000 | 1.000 |
    35	| DOM vs P-prompt | 0.0013 | 1.000 | 1.000 |
    36	| DOM vs P-SoM | 0.0029 | 1.000 | 1.000 |
    37	| DOM vs SoM | 0.0452 | 1.000 | 1.000 |
    38	| DOM vs Vision | 0.0571 | 1.000 | 1.000 |
    39	| P-text vs P-prompt | 0.0031 | 1.000 | 1.000 |
    40	| P-text vs P-SoM | 0.0019 | 1.000 | 1.000 |
    41	| P-text vs SoM | 0.0436 | 1.000 | 1.000 |
    42	| P-text vs Vision | 0.0550 | 1.000 | 1.000 |
    43	| P-prompt vs P-SoM | 0.0017 | 1.000 | 1.000 |
    44	| P-prompt vs SoM | 0.0421 | 1.000 | 1.000 |
    45	| P-prompt vs Vision | 0.0550 | 1.000 | 1.000 |
    46	| P-SoM vs SoM | 0.0386 | 1.000 | 1.000 |
    47	| P-SoM vs Vision | 0.0508 | 1.000 | 1.000 |
    48	| SoM vs Vision | 0.0170 | 1.000 | 1.000 |
    49	
    50	## P-SoM vs baseline modes (paper §5 HERO arm)
    51	
    52	P-SoM identity test: is P-SoM closer to SoM (prompt-axis sibling) or DOM (text-axis sibling)?
    53	
    54	| L | P-SoM↔DOM | P-SoM↔SoM | P-SoM↔Vision | P-SoM↔P-text | P-SoM↔P-prompt |
    55	|---|---|---|---|---|---|
    56	| L00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
    57	| L08 | 0.0016 | 0.0233 | 0.0398 | 0.0004 | 0.0014 |
    58	| L11 | 0.0021 | 0.0314 | 0.0413 | 0.0010 | 0.0014 |
    59	| L17 | 0.0029 | 0.0386 | 0.0508 | 0.0019 | 0.0017 |
    60	| L24 | 0.0065 | 0.0212 | 0.0292 | 0.0051 | 0.0016 |
    61	| L30 | 0.0069 | 0.0194 | 0.0226 | 0.0046 | 0.0014 |
    62	| L36 | 0.0152 | 0.0416 | 0.0497 | 0.0088 | 0.0048 |

exec
/bin/bash -lc "nl -ba docs/checkpoints/paper_drafts/section4_limitations_disclosure.md | sed -n '1,220p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Section 4 — Known Evaluator Limitations & Disclosure (Draft)
     2	
     3	**Status**: 🟡 Draft prose for paper §4 / §3 limitation table. Each subsection cites the
     4	master_bug_catalog.md entry by ID. Reviewer-defensible: bugs are CONFIRMED but blast radius
     5	bounded; mitigations or paper-§3 disclosure rather than retraction.
     6	
     7	**Source**: `docs/reference/master_bug_catalog.md` B-15 / B-20 / B-21 / B-22 / B-26 / B-28 +
     8	`docs/checkpoints/pre_run/preregistration.md` §A1/A3 design asymmetries.
     9	
    10	---
    11	
    12	## §4.X.1 ua_match GPT-judge drift (B-20)
    13	
    14	VWA's `ua_match` evaluator uses a GPT-4o-mini judge to rate the agent's terminating answer
    15	against the task's reference answer. The judge prompt template is fixed in
    16	`evaluation_harness/helper_functions.py` (`llm_fuzzy_match`) and not modified in this work.
    17	However, GPT-4o-mini is a stochastic API: the judge's output drifts across re-evaluations
    18	in 4 distinct modes (semantic equivalence vs strict literal match, spurious "partial credit",
    19	hallucinated rationale, and length-dependent confidence). Static audit of 87 N/A-task FP
    20	episodes (笔记 §95) showed the judge's binary verdict varies on ~12% of borderline cases when
    21	re-queried with temperature ≥0.
    22	
    23	**Mitigation in this work**: We pin judge `temperature=0` for all evaluations and report all
    24	ua_match-affected tasks as part of the `na_fp` exclusion class (preregistration.md §3 FP filter).
    25	Sensitivity analysis (Appendix D) shows our H1/H3 conclusions hold under three FP filter
    26	variants (raw / +na_fp / +na_fp+eval_fp), so judge drift cannot flip the paper's hero claim.
    27	
    28	**Residual concern**: If a future reviewer re-runs the evaluator with a newer GPT-4o-mini
    29	snapshot, single-task labels may flip. The aggregate per-cell SR is robust to this within
    30	±2pp by simulation. We make this explicit in our reproducibility statement (§3.X) rather
    31	than retract the SR claim.
    32	
    33	---
    34	
    35	## §4.X.2 string_match fuzzy_threshold misnomer (B-21)
    36	
    37	VWA's `string_match` evaluator exposes a `fuzzy_threshold` parameter that suggests a
    38	numerical similarity cutoff for string matching. In practice (catalog B-21 static audit),
    39	the parameter is **only honored when fuzzy_threshold=1.0** — under which the evaluator
    40	falls through to the same GPT-4o-mini fuzzy_match judge as `ua_match`. Threshold values
    41	strictly below 1.0 trigger a brittle exact-token-overlap path with no judge involvement.
    42	This is effectively binary GPT-judged matching, not a tunable similarity metric.
    43	
    44	**Mitigation**: We use `fuzzy_threshold=1.0` consistently across all conditions (verified
    45	via condition_meta.json `evaluator_config.fuzzy_threshold`), so the variability source is
    46	the same as B-20 ua_match drift and is jointly bounded by the same FP filter robustness.
    47	The mis-naming does not affect our results, but we flag it for readers attempting to
    48	interpret raw VWA evaluator parameters.
    49	
    50	---
    51	
    52	## §4.X.3 program_html selector brittleness (B-22)
    53	
    54	VWA's `program_html` evaluator scores tasks by goto'ing a target URL and querying DOM with
    55	CSS/XPath selectors authored in each task's reference config. Static audit (笔记 §107
    56	Tier 5) found 562 of 1598 (35.2%) selectors are class-only or attribute-only patterns
    57	(e.g., Magento's `.order-details-items.ordered`, classifieds' `.price` / `.desc`) that
    58	match site-skin-dependent layout. When the site's CSS skin updates between evaluator
    59	authoring time (2024) and our experimental deployment (2026), selectors can match the
    60	wrong DOM node or miss the intended element entirely.
    61	
    62	**Per-cell quantification**: We measure selector hit-rate parity in our archive — for each
    63	program_html task, we count post-action DOM nodes matching the reference selector. A pre/post
    64	ratio outside 0.95-1.05 across modes within the same task is flagged (~3% of program_html
    65	tasks); these are excluded from H1/H3 per the preregistered FP filter `eval_fp` rule.
    66	
    67	**Cannot-fix scope**: Patching all 562 brittle selectors requires authoring a parallel
    68	evaluator harness, which is out of scope for this paper. We retain VWA's evaluator unchanged
    69	(reviewer-defensible upstream parity per §3 evaluator independence) and bound the impact
    70	via the FP filter sensitivity ladder (Appendix D).
    71	
    72	---
    73	
    74	## §4.X.4 finish_wrong_state — agent error not scaffold (B-15)
    75	
    76	In Tier 2 silent-failure analysis (笔记 §107), 1552 of 4501 episodes (34.5%) had the agent
    77	emit `finish` while the page state did not match the task goal. Initial framing classified
    78	this as a scaffold bug; subsequent self-replay (笔记 §95 reform) showed it is an **agent
    79	reasoning error** — the agent decides to terminate prematurely or with partial completion,
    80	not a runner / dispatch / observation failure.
    81	
    82	**Treatment**: This is captured in our `eval_fp` filter rule (preregistration.md §3): if
    83	`agent_finished=True` but evaluator returns success and the agent has no effective action
    84	in the trajectory, we mark the episode as `eval_fp`. The agent error itself is not a paper
    85	limitation — different baselines and modes can succeed or fail at terminating decisions, and
    86	our paired-design comparison absorbs this into per-task variance.
    87	
    88	---
    89	
    90	## §4.X.5 in_viewport_ratio operator precedence (B-26)
    91	
    92	In `external/visualwebarena/browser_env/processors.py:218`, the `in_viewport_ratio`
    93	calculation `overlap_w * overlap_h / w * h` is parsed by Python as
    94	`((overlap_w * overlap_h) / w) * h` — multiplication-first then division — instead of the
    95	intended ratio `(overlap_w * overlap_h) / (w * h)`. The result is that the 0.6 viewport-overlap
    96	threshold (`current_viewport_only=True`) is effectively bypassed, allowing partially-visible
    97	elements to remain in the AXTree with their full text content even when they are visually
    98	truncated.
    99	
   100	**Implication for our claims**: This bug exists in upstream VWA and is documented in our
   101	CLAUDE.md as "DOM has structural information advantage." It systematically helps DOM mode
   102	relative to Vision/SoM modes by exposing element text that is visually clipped. We do **not**
   103	fix this bug because: (a) it's upstream code; (b) any threshold value would be debatable;
   104	(c) it does not affect our **paired** comparisons (P-SoM uses the same DOM-derived
   105	`[SOM_MARKS]` text), so our hero claims (P-SoM ≥ best of DOM/SoM/Vision) are invariant to
   106	this asymmetry. We disclose the asymmetry source for cross-mode interpretation.
   107	
   108	---
   109	
   110	## §4.X.6 scroll direction confusion (B-28)
   111	
   112	Early experiments (B0 cls/red, 笔记 §50) revealed inconsistent agent behavior for scroll
   113	direction conventions: Web CSS uses `dy>0 = scroll DOWN` (content moves up), but Win32 and
   114	macOS natural scrolling invert this convention. The 235B model occasionally chose the wrong
   115	direction sign, producing scroll-up-when-needed-down patterns counted as no-progress.
   116	
   117	**Mitigation**: §67 schema reform replaced `delta: [dx, dy]` with explicit
   118	`scroll_direction: enum("up", "down")` in the action schema (B0 only via tool-calling
   119	schema; B1 still uses delta in greedy decoding). This eliminates the symbol convention
   120	confound for B0 going forward but does not retroactively fix archived B0 data. We disclose
   121	this asymmetry in §3 evaluator-side fairness discussion.
   122	
   123	---
   124	
   125	## §4.X.7 A1/A3 baseline-design asymmetries (B-56)
   126	
   127	This work compares B0 (Qwen3-VL-235B-A22B via proxy API) against B1 (Qwen3-VL-4B-Instruct
   128	local). Two configuration asymmetries are intentional and documented:
   129	
   130	**A1 — Decoding strategy**: B0 uses `temperature=0.0` with `top_p=1.0` (B-37 fix
   131	post-§107); B1 uses `do_sample=False` (greedy top-1). Both target deterministic outputs,
   132	but B0 still inherits proxy-side stochasticity for which the API has no `seed` parameter.
   133	Cross-run trajectory variance for B0 is bounded by single-step branching at ties; aggregate
   134	SR is stable (laughs at our N=234+210+466 sampling).
   135	
   136	**A3 — Token budget**: B0 has `max_new_tokens=4096`; B1 has `max_new_tokens=384`. The
   137	asymmetry stems from B0's verbose thought + JSON output requirement; B1's parser is more
   138	robust to compact outputs. In rare cases (~0.15%), B1's compact budget causes truncated JSON →
   139	parse_failure → `wait` action. We retain this asymmetry as a B1-specific structural
   140	limitation rather than artificially inflate B1's budget; the impact is bounded and disclosed
   141	in §3 baseline configuration table.
   142	
   143	---
   144	
   145	## §4.X.8 Cross-machine numerical drift (笔记 §114 Gap 5)
   146	
   147	Our work runs across three GPU architectures: DGX Spark (NVIDIA GB10, sm_121), UCL Condense
   148	A100 (sm_80), and UCL Myriad (sm_70 V100 / sm_80 A100). Mechanistic Stage 2B/2C activation
   149	patching outputs are sensitive to floating-point matmul precision differences across CUDA
   150	generations (sm_70 vs sm_80 vs sm_121). We run `numerical_determinism_check.py` to quantify
   151	maximum absolute hidden-state drift |Δh| across machines on a fixed input.
   152	
   153	**Reproducibility statement**: Cross-machine numerical agreement on Qwen3-VL-4B between
   154	{DGX, A100, Myriad} layers L0-L35: max |Δh| < [TBD post-rerun, target <1e-2] at L11 (the
   155	mirage causal layer per §5). This bounds inter-machine reproducibility drift to a level that
   156	does not flip top-1 logit comparisons; aggregate SR claims are unaffected.
   157	
   158	---
   159	
   160	## §4.X.9 Pre-Phase-A vs post-Phase-A asymmetry (B-01 to B-37 family)
   161	
   162	The 16-cell rerun (preregistration.md §4 cell inclusion) uses post-Phase-A code only
   163	(commit ≥ `3c15cd7`, dispatch + page_changed + cycle + RNG fixes deployed). Pre-Phase-A
   164	data is retained as Appendix D robustness check (preregistration.md `Cell inclusion (Appendix D)`).
   165	For mechanistic Stage 2B/2C input artifacts, we use pre-Phase-A archived observations
   166	(`results/mechanistic/archive_subset_b1_cls/`); per 笔记 §116 user-prompt analysis, agent
   167	trajectory bugs (Phase A scaffold issues) do **not** affect the model's forward-pass
   168	input→output mapping at any frozen step. Mechanism findings (L11 causal layer, forward-vs-reverse
   169	asymmetry) are therefore unaffected by Phase A vintage; we make this independence explicit
   170	in §5.
   171	
   172	---
   173	
   174	## §4.X.10 Stage 2B input vintage independence (笔记 §116 user Q)
   175	
   176	Mechanistic Stage 2B (forward L11 mirage causal layer) and Stage 2C (reverse direction
   177	asymmetry) use frozen `observation_dom.txt` + `screenshot_annotated.png` artifacts from
   178	`B1_phantom_som_classifieds_20260428` archive (pre-Phase-A). Per 笔记 §116 user analysis:
   179	the mechanistic claim is about model forward-pass behavior given a fixed input, not about
   180	agent trajectory soundness. Phase A bugs in dispatch / cycle / RNG affect *which step* the
   181	agent reaches, not *what the model thinks* given a frozen step's observation. The L11
   182	mirage finding is therefore Phase-A-vintage-independent.
   183	
   184	For full robustness, we pre-specify a post-Phase-A spot-check (5-10 tasks from a clean
   185	post-`3c15cd7` cell) where we re-run Stage 2B and verify L11 causal layer holds. This
   186	sensitivity check is in §5 Appendix and does not gate the main mechanism claim.
   187	
   188	---
   189	
   190	## References
   191	
   192	- `docs/reference/master_bug_catalog.md` — full bug catalog (~80 entries)
   193	- `docs/checkpoints/pre_run/preregistration.md` §3-§4 — locked analysis choices including FP filter
   194	- `docs/checkpoints/pre_run/evaluator_change_protocol.md` — Protocol A Tier classification
   195	- 笔记 §95 (FP reform) / §107 (Phase A wave) / §114 (provenance) / §116 (audit) / §116.X user prompts

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section2_background.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	## 2. Background and Related Work
     2	
     3	### 2.1 Web Agent Observation Modes
     4	
     5	Modern web agents differ less in the browser actions they expose than in the observation representation they give to the language model. Text-only agents typically serialize the Document Object Model or Accessibility Tree (AXTree) into a hierarchical text observation. WebArena uses this style of realistic browser environment to evaluate language-guided agents on shopping, forum, map, and software-development tasks \citep{zhou2024webarena}. Mind2Web similarly frames web interaction as selecting actions from structured page elements collected across real websites \citep{deng2023mind2web}. This line of work makes DOM-derived text the default low-cost representation: it is cheap, symbolic, and compatible with language-only models, but it can be verbose and blind to visual appearance.
     6	
     7	Multimodal web agents add screenshots to the observation. VisualWebArena extends WebArena with visually grounded tasks and evaluates agents that combine page text, screenshots, and visual grounding cues \citep{koh2024visualwebarena}. A common grounding device is Set-of-Mark prompting, introduced by Yang et al. as a way to overlay numbered or speakable marks on image regions so a multimodal model can refer to visual objects by discrete IDs \citep{yang2023som}. SeeAct likewise studies GPT-4V as a generalist web agent and finds that visual understanding must still be paired with reliable action grounding \citep{zheng2024seeact}. Magma pushes the same broad direction into an omni-modal agent foundation model with action grounding and multimodal pretraining \citep{yang2025magma}. Vision-only baselines remove the DOM/AXTree channel and ask the model to act from the screenshot alone; these baselines test whether visual perception can substitute for structured symbolic grounding.
     8	
     9	Across this literature, DOM, SoM, and Vision are treated as orthogonal observation modes. SoM in particular is treated as a multimodal bundle: a marked screenshot plus a text legend that maps mark IDs to elements. The `[SOM_MARKS]` text is normally an auxiliary index for the marked image, not a controlled standalone variable. This convention is the gap our paper targets. We ask what routing behavior emerges when the annotated image is skipped while the remaining factors are held apart: AXTree versus `[SOM_MARKS]` text, DOM versus SoM prompt family, and image-off versus image-on evaluation. The resulting object is not a claim that marked or text-only observations are new artifacts; it is a controlled characterization of the phantom routing space around **Phantom-SoM**.
    10	
    11	### 2.2 Routing in LLM Systems
    12	
    13	Routing has become a standard response to heterogeneous cost and capability. FrugalGPT frames inference as a cascade over multiple LLM APIs, learning when cheaper models can answer and when to escalate to stronger models [Chen et al. 2023]. RouteLLM similarly learns routers from preference data to choose between weaker and stronger LLMs under cost-quality tradeoffs [Ong et al. 2025]. These systems are important precedents for cost-aware inference, but their arms are models. The input representation is usually fixed while the backend model changes.
    14	
    15	Web-agent routing has begun to appear at the modality or expert level. Avenir-Web proposes a Mixture of Grounding Experts for multimodal web agents, using human-experience imitation and expert grounding components to improve long-horizon deployment [Li et al. 2026]. Systems work on multimodal serving also motivates modality-aware scheduling: image-heavy requests create different prefill, encoding, and time-to-first-token bottlenecks than text-only requests, so serving systems can separate resources by modality and stage [Qiu et al. 2025]. These works route over computational pathways, modalities, or grounding experts.
    16	
    17	What is missing is representation-level routing within a single model: selecting between different text formats generated from the same browser state. DOM/AXTree and `[SOM_MARKS]` can contain overlapping element semantics, but their token geometry is different. One is hierarchical, nested, and metadata-rich; the other is flat, indexed, and compact. Prior routing work does not ask whether a single model should see the same page as an AXTree for some tasks and as an isolated marks list for others. Phantom-SoM makes that missing routing axis explicit.
    18	
    19	This distinction also separates our setting from ordinary prompt selection. A representation arm is not just a different instruction template; it changes the observation object that enters the agent loop at every step. If two formats derived from the same browser state route the same model into different exploration policies, then representation becomes a deployable control surface. The router need not choose a larger model or a visual encoder first. It can choose a cheaper textual view of the page, observe whether the trajectory stalls, and escalate only when the cheap representation appears misaligned with the task.
    20	
    21	### 2.3 Prompt-Format Sensitivity in LLMs
    22	
    23	The plausibility of Phantom-SoM rests on a broader fact about language models: semantically equivalent prompts can induce different behavior when their surface form changes. Sclar et al. quantify language-model sensitivity to spurious prompt-format features and show that small formatting choices can produce large accuracy differences, even when the underlying task semantics are unchanged [Sclar et al. 2024]. Mishra et al. show a related effect for instructional prompting: reframing instructions into forms better aligned with a model's learned language can change few-shot performance [Mishra et al. 2022].
    24	
    25	These studies do not study web agents, but they explain why web observations should not be treated as neutral containers. A page serialized as AXTree text is not merely "the same information" as a page serialized as `[SOM_MARKS]`. The model receives different punctuation, ordering, indentation, repeated role tokens, ID patterns, and local neighborhoods. Those tokens prime different latent states and therefore different action distributions.
    26	
    27	For a web agent, prompt-format sensitivity matters at the trajectory level. The model is not producing a single label; it is choosing whether to search, click, scroll, revisit a page, or finish. Section 4 and Section 5 build on this theoretical anchor: the flat marks list tends to shift exploration toward quick element selection, while AXTree hierarchy tends to support sustained navigation and search. Prompt wording also matters, but our two-knob account separates the layers: text representation shapes how the agent explores, while prompt family tunes when it commits.
    28	
    29	### 2.4 Cost-Efficient Inference for Web Agents
    30	
    31	Cost-efficient web-agent inference has usually meant pruning or scheduling expensive context. AXTree observations are long, noisy, and security-sensitive. FocusAgent addresses this by using a lightweight retriever to trim AXTree observations before sending them to the main agent, reducing context while preserving the hierarchical representation [Kerboua et al. 2025]. This is a natural text-efficiency strategy: keep the DOM-derived tree, but remove irrelevant lines.
    32	
    33	Multimodal inference adds a second cost source: visual encoding. Image inputs increase prompt-processing time, memory pressure, and time-to-first-token. ModServe characterizes large multimodal model serving and shows that multimodal workloads have heterogeneous stages and resource requirements, motivating modality- and stage-aware resource disaggregation [Qiu et al. 2025]. In web agents, full SoM therefore has two costs: it prepares a marked screenshot and it sends image tokens to the model.
    34	
    35	Phantom-SoM explores a different kind of efficiency. It is not text pruning and it is not image scheduling. It is text reformatting. The `[SOM_MARKS]` list can be generated from the same browser/AXTree metadata already available to the agent, then sent without the marked screenshot. This removes image-token cost while preserving a discrete element index. In our runs the text observation is comparable in token length to the corresponding AXTree (within ±7% on reddit and classifieds, holding the system prompt fixed); the difference is in structure — flat indexed list versus nested hierarchy with url/tab metadata — rather than in length. The open question is whether such a representation is only a structural rewrite of DOM, or whether its format creates a distinct success pool. Our empirical sections answer the latter.
    36	
    37	This matters because many cost reductions trade away information: smaller models, shorter context windows, lower image resolution, or fewer retrieved lines. Phantom-SoM instead tests whether a cheap re-arrangement of already available text can expose a different reasoning path. If it succeeds on tasks missed by DOM, the gain is not merely compression; it is complementarity. That is why Section 4 reports both single-mode success and drop-one oracle value rather than treating token savings alone as the contribution.
    38	
    39	### 2.5 Position of This Work
    40	
    41	This paper positions Phantom-SoM at the intersection of four literatures that are usually studied separately. First, SoM and its descendants, including Magma and Ferret-UI 2, use marks as visual grounding devices and generally keep text tied to the marked image \citep{yang2023som,yang2025magma,li2025ferretui2}. Second, web-agent benchmarks compare DOM, SoM, and Vision modes, but do not use the mark-text-without-image condition as a controlled axis for routing characterization \citep{zhou2024webarena,koh2024visualwebarena,zheng2024seeact}. Third, routing systems optimize over models, modalities, or experts, not over text formats of the same browser state [Chen et al. 2023; Ong et al. 2025; Li et al. 2026]. Fourth, prompt-format work predicts sensitivity to representation syntax, but has not measured task-pool complementarity in interactive web agents \citep{sclar2024promptformat,mishra2022reframing}.
    42	
    43	There are important artifact precedents, and this paper treats them as context rather than as targets to out-claim. SoM-Mark already pairs textual mark references with visual marks \citep{yang2023som}; SeeAct explores marked-screenshot web-agent grounding \citep{zheng2024seeact}; and Magma incorporates related SoM-style and action-grounding ideas into an omni-modal agent model \citep{yang2025magma}. These systems show that marked observations, textual references, and multimodal action grounding are not new merely as artifacts. The contested point is different: whether the behavior of the image-skipped configurations has been scientifically isolated, compared against DOM/SoM/Vision on identical task pools, and explained mechanistically.
    44	
    45	The resulting gap is therefore a characterization gap, not a first-deployment gap. Published systems and benchmarks have not, to our knowledge, provided a controlled scientific evaluation of the phantom boundary: `[SOM_MARKS]` isolated from the screenshot, crossed with prompt family in a 2-by-2 control, compared through low-overlap success pools and drop-one oracle value, and checked against mechanistic controls including mid-layer L11-L17 evidence and random injection. This is also where the 4-fold drop-in property belongs. It is an empirical finding about a controlled configuration: zero image tokens, no new model, usable routing signal, and positive incremental oracle value. It is not a claim that text-only browser control or marked observations were first introduced here.
    46	
    47	Our contributions follow directly. We define **Phantom-SoM** as the deployment-relevant representative of the phantom routing space: marks text plus SoM prompt family, with the image removed. We show empirically that Phantom-SoM is not a degenerate DOM surrogate: it contributes independent oracle value and has substantial task-pool non-overlap with DOM, SoM, and Vision. The P-text and P-prompt controls establish specificity by separating text-payload flattening from prompt-family effects, rather than attributing all gains to a single novel arm. Finally, we provide mechanism evidence for a two-knob account: representation format shapes exploration, while prompt wording tunes commitment confidence. This motivates the experimental design in Section 3 and the controlled evidence in Section 4.

exec
/bin/bash -lc 'rg --files docs/checkpoints/paper_drafts | sort' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/paper_drafts/paper.bib
docs/checkpoints/paper_drafts/section1_intro.md
docs/checkpoints/paper_drafts/section2_background.md
docs/checkpoints/paper_drafts/section3_definition.md
docs/checkpoints/paper_drafts/section4_empirical_findings.md
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md
docs/checkpoints/paper_drafts/section5_mechanism.md
docs/checkpoints/paper_drafts/section8_limitations.md

exec
/bin/bash -lc 'rg -n "Section 6|Section 7|section6|section7|## 6|## 7" docs/checkpoints/paper_drafts docs/checkpoints/mechanism/plan.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/mechanism/plan.md:235:## 6. Open questions (paper-grade gaps)
docs/checkpoints/mechanism/plan.md:249:## 7. Advisor sync state — Zekun (Wu et al. 2026 first author = lab member)
docs/checkpoints/mechanism/plan.md:251:### 7.1 Timeline confirmed (not scoop)
docs/checkpoints/mechanism/plan.md:261:### 7.2 Message draft (v3, paste-ready 2026-05-12)
docs/checkpoints/mechanism/plan.md:303:### 7.3 H1 generalization in-flight (2026-05-12 night)
docs/checkpoints/mechanism/plan.md:338:### 7.3.0 Exp 1 axis-2 layer profile (2026-05-12 21:00 — three-axis hierarchy)
docs/checkpoints/mechanism/plan.md:363:### 7.3.0b Axis-2 per-task fragility check (2026-05-12 21:50 — /stress W2 defuse)
docs/checkpoints/mechanism/plan.md:384:### 7.3.0a Exp 3 logit lens 输出层 amplification (2026-05-12 21:02)
docs/checkpoints/mechanism/plan.md:403:### 7.3.1 Reddit cross-site results (2026-05-12 16:30 — P5a + P5b analyses landed)
docs/checkpoints/mechanism/plan.md:446:### 7.4 Decisions pending
docs/checkpoints/paper_drafts/section5_mechanism.md:107:Two additional defenses remain deferred rather than folded into the claim: P2 cross-family Phi-3.5-Vision and P3 larger Qwen2-VL-7B. The current evidence is sufficient for the single-model, cross-site Qwen3-VL-4B mechanism section; family and capacity generalization belong in future work or Section 7.
docs/checkpoints/paper_drafts/section5_mechanism.md:155:Behavioral content to relocate from current `section5_mechanism_reddit.md`: lines 17-75 should move to Section 4 or a new behavioral-routing subsection. Specifically, lines 17-23 are reddit substrate framing; lines 25-35 are Axis 1 text-payload behavior; lines 37-47 are Axis 2 prompt behavior; lines 49-59 are Axis 3 image behavior; lines 61-67 are compound P-SoM versus DOM behavior; lines 69-75 are scope/noise limitations. Lines 1-15 are method material that was retained conceptually but must use the new L0-L36 layer convention. Line 77 should be deleted or replaced because routing implementation is now paper-2, not paper-1 Section 6.
docs/checkpoints/paper_drafts/section5_mechanism.md:167:3. ✅ §5.7 corollary 2 — "deployment-time mode classifier on output logprobs has strictly more signal" + "Section 6 routing should treat L23-L25 logit-lens features as the cheapest mode-axis discriminator" → softened to "mechanistic observation, not deployment-time classifier claim; held-out classifier with random-direction baseline is open work"
docs/checkpoints/paper_drafts/section5_mechanism.md:174:- **§6 + §7 drafts missing**: §1:13 promises Section 6 (Generalization) and Section 7 (Limitations and Implications). Current draft files: no `section6*.md`; §7 either deferred to paper-2 (routing) or merge into §8. Resolve before submission.
docs/checkpoints/paper_drafts/section1_intro.md:13:We evaluate on VisualWebArena classifieds and reddit with B0, an API-backed Qwen3-VL-235B agent, and use B1, a local Qwen3-VL-4B model, as a cross-capability robustness check rather than a separate scientific claim. Following the pre-registered R1-R5 framing rules, the scope of this paper is Qwen-family VisualWebArena characterization and explanation, not a claim of universal model-family generalization and not yet a learned deployment router. Routing exploitation is deferred to a follow-up paper. Section 2 situates the gap in web-agent representations, SoM, routing, and prompt-format sensitivity. Section 3 defines Phantom-SoM and the measurement protocol. Section 4 reports phantom-space and image-on baseline findings. Section 5 analyzes the mid-layer activation-patching mechanism. Section 6 discusses generalization, and Section 7 summarizes limitations and implications.
docs/checkpoints/paper_drafts/section3_definition.md:76:The 2x2 is not a routing policy by itself. It is a causal scaffold for Section 5: text-payload structure shapes exploration, while prompt wording tunes commitment confidence. Section 6 promotes this scaffold to a token-monotonic cascade — DOM → P-text (axis 1, text-payload swap, no token increase) → Phantom-SoM (axis 2, system-prompt swap, no data-token increase) → full SoM (axis 3, image embedding cost) — so a routing trigger never has to "add then remove" tokens.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/format_variation_h1_test.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4 H1 test: indexed-list format variation
     2	
     3	Test refined H1 hypothesis (pretraining co-occurrence shortcut):
     4	*"input contains mark-like indexed region list → activates visual-grounding pathway"*
     5	
     6	**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
     7	- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
     8	- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut
     9	
    10	## Result table (sorted by peak layer)
    11	
    12	| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
    13	|---|---|---|---|---|
    14	| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0462 |
    15	| plain_sentence | `'a, b, c, ...' (no list)` | control (no list) | **L17** | 0.0544 |
    16	| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0434 |
    17	| browser_use_at | `@N label (Browser Use)` | marks-like | **L36** | 0.0502 |
    18	| appagent_id | `id_N: label (AppAgent)` | marks-like | **L36** | 0.0506 |
    19	| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L36** | 0.0468 |
    20	| plain_numbered | `N. label (numbered)` | marks-like | **L36** | 0.0502 |
    21	| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L36** | 0.0426 |
    22	| hash_id_control | `#hash label (no integer)` | control (no integer) | **L36** | 0.0514 |
    23	
    24	## Grouped by H1 prediction
    25	
    26	### marks-like  (mean peak L36)
    27	
    28	- `[N] role 'label' (SoM)`: peak **L36** = 0.0434
    29	- `@N label (Browser Use)`: peak **L36** = 0.0502
    30	- `id_N: label (AppAgent)`: peak **L36** = 0.0506
    31	- `[BN:role:label] (Tarsier)`: peak **L36** = 0.0468
    32	- `N. label (numbered)`: peak **L36** = 0.0502
    33	- `<el_N role='..'>label</el_N> (XML)`: peak **L36** = 0.0426
    34	
    35	### control (no integer)  (mean peak L36)
    36	
    37	- `#hash label (no integer)`: peak **L36** = 0.0514
    38	
    39	### control (no list)  (mean peak L17)
    40	
    41	- `'a, b, c, ...' (no list)`: peak **L17** = 0.0544
    42	
    43	### AXTree-baseline  (mean peak L4)
    44	
    45	- `AXTree (baseline DOM)`: peak **L04** = 0.0462
    46	
    47	## H1 verdict
    48	
    49	- **6 marks-like variants**: mean peak layer = 36, range L36-L36
    50	- **2 control variants** (no integer / no list): mean peak layer = 26, range L17-L36
    51	- **AXTree-DOM baseline**: peak L04
    52	
    53	→ **H1 PARTIAL**: marks-like AND controls all peak late — finding is broader than 'indexed list' (any text payload triggers).

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/h1_per_task_fragility.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# H1 per-task fragility check
     2	
     3	**Sample**: 45 (task, step) pairs from format_variation_b1_cls
     4	
     5	## Aggregate verdict per individual (task, step) pair
     6	
     7	- **AXTree-DOM peak ≤ L10** (early image-axis peak): 9/45 = **20%**
     8	- **≥4/7 marks-like variants peak ≥ L20** (late image-axis peak): 39/45 = **87%**
     9	- **BOTH conditions** (strict dichotomy per task): 5/45 = **11%**
    10	
    11	## Per-task peak-layer distribution
    12	
    13	AXTree-DOM peak layer: mean = **27.9**, std = 13.1, range L04-L36
    14	Marks-like (avg across 7) peak layer: mean = **31.9**, std = 8.0
    15	**Separation** = marks - dom = **+4.0 layers**
    16	
    17	## Verdict
    18	
    19	→ **H1 WEAK per-task**: dichotomy is averaged effect, not per-task universal. Paper §5 framing must acknowledge per-task variability.
    20	
    21	## Top 5 dichotomy-confirming (task, step) pairs (largest separation)
    22	
    23	| Task ID | Step | AXTree peak | Marks avg peak | Separation |
    24	|---|---|---|---|---|
    25	| 214 | 5 | L04 | L36.0 | **+32.0** |
    26	| 228 | 2 | L04 | L36.0 | **+32.0** |
    27	| 32 | 5 | L04 | L31.4 | **+27.4** |
    28	| 228 | 5 | L04 | L29.4 | **+25.4** |
    29	| 9 | 2 | L04 | L24.6 | **+20.6** |
    30	
    31	## Bottom 5 (task, step) pairs (smallest / inverse separation)
    32	
    33	| Task ID | Step | AXTree peak | Marks avg peak | Separation |
    34	|---|---|---|---|---|
    35	| 61 | 5 | L17 | L16.4 | -0.6 |
    36	| 20 | 2 | L36 | L33.3 | -2.7 |
    37	| 122 | 2 | L36 | L33.3 | -2.7 |
    38	| 60 | 5 | L17 | L11.6 | -5.4 |
    39	| 37 | 2 | L36 | L28.0 | -8.0 |

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4 H1 test: indexed-list format variation
     2	
     3	Test refined H1 hypothesis (pretraining co-occurrence shortcut):
     4	*"input contains mark-like indexed region list → activates visual-grounding pathway"*
     5	
     6	**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
     7	- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
     8	- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut
     9	
    10	## Result table (sorted by peak layer)
    11	
    12	| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
    13	|---|---|---|---|---|
    14	| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0434 |
    15	| plain_sentence | `'a, b, c, ...' (no list)` | control (no list) | **L22** | 0.0529 |
    16	| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0429 |
    17	| browser_use_at | `@N label (Browser Use)` | marks-like | **L36** | 0.0520 |
    18	| appagent_id | `id_N: label (AppAgent)` | marks-like | **L36** | 0.0526 |
    19	| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L36** | 0.0475 |
    20	| plain_numbered | `N. label (numbered)` | marks-like | **L36** | 0.0518 |
    21	| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L36** | 0.0439 |
    22	| hash_id_control | `#hash label (no integer)` | control (no integer) | **L36** | 0.0516 |
    23	
    24	## Grouped by H1 prediction
    25	
    26	### marks-like  (mean peak L36)
    27	
    28	- `[N] role 'label' (SoM)`: peak **L36** = 0.0429
    29	- `@N label (Browser Use)`: peak **L36** = 0.0520
    30	- `id_N: label (AppAgent)`: peak **L36** = 0.0526
    31	- `[BN:role:label] (Tarsier)`: peak **L36** = 0.0475
    32	- `N. label (numbered)`: peak **L36** = 0.0518
    33	- `<el_N role='..'>label</el_N> (XML)`: peak **L36** = 0.0439
    34	
    35	### control (no integer)  (mean peak L36)
    36	
    37	- `#hash label (no integer)`: peak **L36** = 0.0516
    38	
    39	### control (no list)  (mean peak L22)
    40	
    41	- `'a, b, c, ...' (no list)`: peak **L22** = 0.0529
    42	
    43	### AXTree-baseline  (mean peak L4)
    44	
    45	- `AXTree (baseline DOM)`: peak **L04** = 0.0434
    46	
    47	## H1 verdict
    48	
    49	- **6 marks-like variants**: mean peak layer = 36, range L36-L36
    50	- **2 control variants** (no integer / no list): mean peak layer = 29, range L22-L36
    51	- **AXTree-DOM baseline**: peak L04
    52	
    53	→ **H1 PARTIAL**: marks-like AND controls all peak late — finding is broader than 'indexed list' (any text payload triggers).

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/layer_axis_emergence.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4: image-axis peak-layer split — Mirage Effect signature
     2	
     3	Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:
     4	
     5	| no-image side | image side | no-img text | peak layer | peak cosine gap |
     6	|---|---|---|---|---|
     7	| DOM | SoM | AXTree | **L04** | 0.0604 |
     8	| DOM | Vision | AXTree | **L04** | 0.0653 |
     9	| P-prompt | SoM | AXTree | **L04** | 0.0600 |
    10	| P-prompt | Vision | AXTree | **L04** | 0.0649 |
    11	| P-SoM | SoM | [SOM_MARKS] | **L17** | 0.0412 |
    12	| P-text | SoM | [SOM_MARKS] | **L20** | 0.0494 |
    13	| P-text | Vision | [SOM_MARKS] | **L36** | 0.0614 |
    14	| P-SoM | Vision | [SOM_MARKS] | **L36** | 0.0613 |
    15	
    16	## Grouped by no-image side text format
    17	
    18	### no-image text = `AXTree` (mean peak L4)
    19	
    20	- DOM ↔ SoM: peak **L04** = 0.0604
    21	- DOM ↔ Vision: peak **L04** = 0.0653
    22	- P-prompt ↔ SoM: peak **L04** = 0.0600
    23	- P-prompt ↔ Vision: peak **L04** = 0.0649
    24	
    25	### no-image text = `[SOM_MARKS]` (mean peak L27)
    26	
    27	- P-text ↔ SoM: peak **L20** = 0.0494
    28	- P-text ↔ Vision: peak **L36** = 0.0614
    29	- P-SoM ↔ SoM: peak **L17** = 0.0412
    30	- P-SoM ↔ Vision: peak **L36** = 0.0613
    31	
    32	## Mechanism interpretation (paper §5 v3 Mirage anchor)
    33	
    34	When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).
    35	
    36	When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.
    37	
    38	**Mechanism story**: `[SOM_MARKS]` text in input primes an indexed-parsing pathway through mid-layer computation. The model processes marks structurally regardless of whether image is provided, producing image-axis divergence only at late integration stages. This is the direct Method 4.2 empirical anchor for the **Mirage Effect** (Asadi et al. 2026, VLM ~70-80% no-image accuracy) and **Cross-modal flow** (Kaduri et al., middle-layer cross-modal flows store image info in query tokens): the marks-primed mid-layer computation runs *as if image were available*, with image grounding contributed only late.
    39	
    40	**Paper §5 prose** (suggested):
    41	
    42	> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*

codex
我有足够的主要证据了：hero oracle 的 bootstrap 支撑较干净；机制部分最大问题是 v2 已经推翻了部分 §5/plan 旧叙事，但文档内仍有冲突、pending rerun 和缺失章节。最后我会做一遍针对 line-number 的核对，然后给中文为主的单篇 markdown review。
exec
/bin/bash -lc 'rg -n "NEEDS|TBD|pending|in flight|re-run|rerun|TODO|not yet|open work|awaits|missing|deferred" docs/checkpoints/paper_drafts docs/checkpoints/mechanism/plan.md docs/checkpoints/mechanism/results' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/mechanism/plan.md:99:### 3.2 Method 4.4 — mean-diff activation steering (v2 in flight)
docs/checkpoints/mechanism/plan.md:119:Decision pending Method 4.4 v2 full sweep + Zekun sync.
docs/checkpoints/mechanism/plan.md:134:1. **Probe-level** (Method 4.2 PCA cosine gap; AUROC reported as both `in_sample` and held-out `leave-one-task-out` after 2026-05-12 Bug 3 fix; v1 buggy NPZ data invalidated, v2 NPZ in flight: Myriad 359736 cls + 359737 reddit)
docs/checkpoints/mechanism/plan.md:144:| A3 | 24 strong-tier tasks generalize to broader VWA distribution | Stage 4 robustness Test B: 100% per-task positive, but tier-selection bias possible. Reverse-tier 15 tasks pending |
docs/checkpoints/mechanism/plan.md:310:| **P2** | Cross-family (Phi-3.5-Vision 4.2B) | DGX | ❌ deferred (HF cas-bridge throttling) | `stage4_h1_phi35_cls/pilot_summary.md` |
docs/checkpoints/mechanism/plan.md:311:| **P3** | Within-family bigger (Qwen2-VL-7B, H1' capacity test) | DGX | ❌ deferred (HF cas-bridge throttling) | `stage4_h1_qwen2vl7b_cls/pilot_summary.md` |
docs/checkpoints/mechanism/plan.md:321:**P2/P3 deferred** (2026-05-12 00:31 → 06:30, 3 attempts each):
docs/checkpoints/mechanism/plan.md:361:→ Paper §5.7 重写为 "Layered Three-Axis Mechanism Hierarchy" (commit pending).
docs/checkpoints/mechanism/plan.md:446:### 7.4 Decisions pending
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:48:Within the 6 marks-like variants, the L17 vs L04 split corresponds to whether the variant's first tokens are **markup-sigil tokens** (`[`, `<`, `@`) — which co-occur with HTML / web-agent traces in pretraining and trigger the visual-grounding shortcut at mid layers — versus **plain alphanumeric tokens** (`id`, `1`) — which are common in prose / dictionary listings and behave like AXTree-DOM, peaking early at L04 where the image-axis divergence is freshly observable but not yet routed through the shortcut path.
docs/checkpoints/paper_drafts/section5_mechanism.md:11:Four mechanism claims organize the evidence (revised 2026-05-12 after v2 NPZ re-extraction; see §5.7 revision note). First, observation modes are **linearly separable** in the residual stream: held-out leave-one-task-out AUROC = 1.000 across all mode pairs and all 37 layers (Method 4.2 v2). Second, the **geometric magnitude** of mode separation is dominated by the image axis (cosine ~0.04-0.07), with text-format and prompt-family axes producing only sub-permille cosine separation; the prior "three quantitatively distinct axes at 4:3:1 ratio" framing was a v1 NPZ artifact and is retracted. Third, SoM-to-no-image activation patching displaces target continuations at mid layers L11-L17 with consistent magnitude across classifieds and reddit (~25% target-overlap drop). The Exp 5 axis-2 prompt-only patching subset (cellhprompt cls + red) shows this displacement persists when text format is held flat, indicating prompt-family carries causal influence despite its near-zero geometric magnitude — a feature *used* without large feature *encoded* magnitude (\citep{wang2023interpretability} signature). Fourth, the shortcut trigger is **better described as the conjunction of integer-indexed marker and markup-sigil leading delimiter** than as an abstract "flat element list" — AXTree hierarchy preserves the early L04 image-axis peak, but so do indexed variants that lack either the integer (e.g., `hash_id_control`) or the sigil (e.g., `appagent_id`, `plain_numbered`); only the conjunction triggers the late shift. This refinement is **exploratory after W6** and awaits held-out falsifiers (`bare_N`, `bracket_no_int`).
docs/checkpoints/paper_drafts/section5_mechanism.md:20:| Causal axis-2 prompt-only patching | Exp 5 cellhprompt cls + red (§5.4) | **Causal continuation evidence, 2 sites, N=24 each; 0.20-0.30 displacement at L11-L17 captures 80-125% of combined image+prompt patching effect**. Task-shuffled content-specificity control (cellhprm_*_tsh Myriad 359768+359769) in flight. Gaussian random control (cellhprm_*_rand 359719+359720) DESTROYS output regardless of axis (codex Bug 6 prediction confirmed; Gaussian is weak baseline) |
docs/checkpoints/paper_drafts/section5_mechanism.md:22:| Output divergence | Exp 3 logit lens (§5.7) | **Re-run pending** on v2 NPZ. V1 reported KL 0.16 at L23 axis-2 + KL 0.69 at L23 axis-1; V2 likely revises both. Mechanism direction (lm_head amplifies residual into output KL) probably survives; magnitudes will change |
docs/checkpoints/paper_drafts/section5_mechanism.md:23:| Trigger attribution (which formats trigger shortcut) | W6 tokenization (§5.5) | **Exploratory** — 6 marks-like variants split 2-vs-4 on first-token sigil; held-out falsifier `bare_N` (integer no sigil) and `bracket_no_int` (sigil no integer) pending |
docs/checkpoints/paper_drafts/section5_mechanism.md:25:The cross-site evidence stack is deliberately defensive. Per-task H1 fragility shows the dichotomy is an aggregate mechanism rather than a deterministic per-task law. Reverse-tier classifieds runs defend against strong-tier selection bias. Reddit format variation replicates the shortcut direction with cleaner mid-layer peaks. Reddit Method 4.2 replicates the Mirage signature: Phantom-SoM remains close to DOM on the text axis while separating from SoM on the image axis. Paper 1 uses these results for mechanism interpretation only; routing implementation is deferred to paper 2, consistent with the paper-planning scope split.
docs/checkpoints/paper_drafts/section5_mechanism.md:95:The reddit format run is cleaner for the mid-layer interpretation. Four of six marks-like variants peak at L17, the plain-sentence control peaks at L17, hash-ID control returns to L04, and DOM remains at L04. **W6 attribution** (`docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md`, exploratory) further finds that the two L04 marks-like variants (`appagent_id`, `plain_numbered`) share a feature with the L04 DOM baseline: their first tokens are alphanumeric, while the four L17-peaking marks-like variants all start with markup-sigil tokens (`[`, `<`, `@`). The hash-ID control (`#a3f7`) starts with a sigil but lacks integer-marker structure and also peaks at L04, suggesting the trigger conjunction is integer-marker + markup-sigil rather than either alone. This is a post-hoc feature-attribution on a small (N=6 marks-like) format set; held-out falsifiers (`bare_N` = integer without sigil, `bracket_no_int` = sigil without integer) are not yet run. Cross-site, the safe claim is directional: marker formats that combine integer indexing with markup-sigil leading delimiters tend to delay image-axis separation into mid/late layers, while AXTree hierarchy and indexed-list variants lacking either feature preserve the early L04 image-axis peak. The reddit curve reveals the true L11-L17 fusion locus more clearly than the classifieds L36 boundary artifact.
docs/checkpoints/paper_drafts/section5_mechanism.md:107:Two additional defenses remain deferred rather than folded into the claim: P2 cross-family Phi-3.5-Vision and P3 larger Qwen2-VL-7B. The current evidence is sufficient for the single-model, cross-site Qwen3-VL-4B mechanism section; family and capacity generalization belong in future work or Section 7.
docs/checkpoints/paper_drafts/section5_mechanism.md:137:The output-amplification observation (logit lens, Exp 3) needs re-running on the v2 NPZ before its quantitative claims can be reported. The v1 logit lens reported peak KL 0.162 at L23 for the axis-2 pair P-text vs P-SoM, but the v1 input hidden states were the buggy 3-line-text version. The qualitative direction (lm_head amplifies residual-stream geometry into output KL) likely survives, but the absolute KL magnitudes will change; we report the v2 lm_head amplification numbers in a follow-up release.
docs/checkpoints/paper_drafts/section5_mechanism.md:139:Two corollaries follow. First, the KL trajectory drops to approximately zero at L36 even though L23 KL is substantial. The mean hidden state at the final layer collapses to the shared JSON action-header tokens that every mode emits, so mode-distinct output signal is concentrated in the L23-L25 decoding window rather than at the final embedding. Second, this output-amplification observation is **mechanistic, not a deployment-time classifier claim**: the lm_head acts as an axis-agnostic ratio-preserving projection that scales residual-stream geometry into output-space KL — the L23-L25 KL magnitude is a property of the mean hidden state, not a per-task discriminator. Whether the L23-L25 hidden representation can be used as a held-out mode classifier — with per-task AUROC, random-direction baseline, and competitive comparison to surface-token classifiers — is open work. Routing exploitation, deferred to paper 2, will need to make this case explicitly rather than inheriting it from §5.7.
docs/checkpoints/paper_drafts/section5_mechanism.md:149:Finally, AXTree hierarchy is the unique defeating format in the aggregate, but the reason hierarchy defeats the shortcut remains open. The plan records one attribution-pending hypothesis: hierarchy or indentation tokens may redirect cross-modal attention before the flat-list shortcut activates. That should be treated as a supplement question, not as a Section 5 finding.
docs/checkpoints/paper_drafts/section5_mechanism.md:153:Bibkeys audit (2026-05-12 21:18): all 5 core mechanism anchors verified present in `paper.bib` — `wu2026toolcalling`, `khorasani2026hdmi`, `kaduri2024whatsintheimage`, `sclar2024promptformat`, `fayyaz2026steermoe`. Plus 5 method/protocol references added: `wang2023interpretability` (IOI patching), `zhang2024patching` (patching survey, NEEDS_VERIFY exact paper), `holm1979sequentially` (multiple-comparison correction), `lipton2018troubling` (ML scholarship critique), `neurips2024checklist` (reproducibility standard). paper.bib total 67 entries / 638 lines.
docs/checkpoints/paper_drafts/section5_mechanism.md:159:Pending items (post 2026-05-12 audit): (a) Method 4.4 sweep description should be "45 completed cells out of a 6x5 layer-alpha grid plus 3 placeholder cells that did not finish", not "45/48-cell sweep" (the 48-cell wording in plan §5.3 implies a 48-cell denominator that was never executed). (b) Bibkey `zhang2024patching` is marked NEEDS_VERIFY in `paper.bib` because the intended reference may be Heimersheim & Nanda 2024 [arXiv:2404.15255] rather than Zhang & Nanda 2024 [arXiv:2309.16042]; verify before submission. (c) Bibkey `fayyaz2026steermoe` is marked NEEDS_VERIFY pending deanon of the ICLR 2026 submission.
docs/checkpoints/paper_drafts/section5_mechanism.md:161:## NOTE FOR HUMAN — /codex-stress 2026-05-12 findings + pending follow-ups
docs/checkpoints/paper_drafts/section5_mechanism.md:167:3. ✅ §5.7 corollary 2 — "deployment-time mode classifier on output logprobs has strictly more signal" + "Section 6 routing should treat L23-L25 logit-lens features as the cheapest mode-axis discriminator" → softened to "mechanistic observation, not deployment-time classifier claim; held-out classifier with random-direction baseline is open work"
docs/checkpoints/paper_drafts/section5_mechanism.md:168:4. ✅ Evidence status table added at end of §5.1 — geometry strong / patching causal-continuation / Exp 5 axis-2 CI pending / steering weak / output divergence not classifier / W6 trigger exploratory
docs/checkpoints/paper_drafts/section5_mechanism.md:174:- **§6 + §7 drafts missing**: §1:13 promises Section 6 (Generalization) and Section 7 (Limitations and Implications). Current draft files: no `section6*.md`; §7 either deferred to paper-2 (routing) or merge into §8. Resolve before submission.
docs/checkpoints/paper_drafts/section5_mechanism.md:175:- **Exp 5 cellhprompt bootstrap CI + content-matched control**: Gaussian random injection control 359719/359720 in flight; codex notes Gaussian alone is weak — also need task-shuffled (source from different task) and per-task bootstrap CIs.
docs/checkpoints/paper_drafts/section8_limitations.md:19:The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Exclusions are listwise only, at <=5% per cell under the B6 lock, so multiple imputation would add modeling assumptions without materially changing paired denominators. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
docs/checkpoints/paper_drafts/section2_background.md:17:What is missing is representation-level routing within a single model: selecting between different text formats generated from the same browser state. DOM/AXTree and `[SOM_MARKS]` can contain overlapping element semantics, but their token geometry is different. One is hierarchical, nested, and metadata-rich; the other is flat, indexed, and compact. Prior routing work does not ask whether a single model should see the same page as an AXTree for some tasks and as an isolated marks list for others. Phantom-SoM makes that missing routing axis explicit.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:74:- All non-image-axis numbers drop 4-8x (re-run on v2 NPZ provides canonical values)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:92:- Per-task fragility (uses Stage 4 NPZ): NEEDS RE-RUN on v2
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:95:- Cross-site Mirage geometry: NEEDS RE-RUN on v2
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:97:**Exp 1 axis-2 layer profile** (`axis2_layer_profile.md`): NEEDS RE-RUN on v2; current 4:3:1 ratio claim is invalidated.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:99:**Exp 3 logit lens** (`axis2_logit_lens.md`): NEEDS RE-RUN on v2. Output-space KL magnitudes likely also change (the lm_head amplifies whatever residual-stream signal is there).
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:116:> 3. **Causal patching effect at mid-layer L11-L17** (Stage 2/3 + Exp 5): 20-30% target-overlap displacement under final-token replacement patching, content-specific (task-shuffled control 359768/359769 in flight)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:124:- v2 metrics: `/tmp/method42_{cls,red}_v2.{json,md}` (canonical paths after rerun: `results/mechanistic/stage4_multimode_b1_*/method42_metrics.json`)
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:28:**Residual concern**: If a future reviewer re-runs the evaluator with a newer GPT-4o-mini
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:154:{DGX, A100, Myriad} layers L0-L35: max |Δh| < [TBD post-rerun, target <1e-2] at L11 (the
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:162:The 16-cell rerun (preregistration.md §4 cell inclusion) uses post-Phase-A code only
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:185:post-`3c15cd7` cell) where we re-run Stage 2B and verify L11 causal layer holds. This
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:58:| L0  | 0.86 | 0.92 | early, signal not yet routed |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:36:When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.
docs/checkpoints/mechanism/results/layer_axis_emergence.md:42:> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*
docs/checkpoints/paper_drafts/section1_intro.md:13:We evaluate on VisualWebArena classifieds and reddit with B0, an API-backed Qwen3-VL-235B agent, and use B1, a local Qwen3-VL-4B model, as a cross-capability robustness check rather than a separate scientific claim. Following the pre-registered R1-R5 framing rules, the scope of this paper is Qwen-family VisualWebArena characterization and explanation, not a claim of universal model-family generalization and not yet a learned deployment router. Routing exploitation is deferred to a follow-up paper. Section 2 situates the gap in web-agent representations, SoM, routing, and prompt-format sensitivity. Section 3 defines Phantom-SoM and the measurement protocol. Section 4 reports phantom-space and image-on baseline findings. Section 5 analyzes the mid-layer activation-patching mechanism. Section 6 discusses generalization, and Section 7 summarizes limitations and implications.
docs/checkpoints/paper_drafts/paper.bib:584:  note = {ICLR 2026 (Anonymous in submission; deanon TBD). Paired-prompt expert Risk Difference (RD) score $\Delta_{\ell,i}$ identifies behavior-linked experts in MoE LLMs (e.g., Qwen3-30B-A3B, Mixtral, DeepSeek-V2). Inference-time router-logit adjustment activates/deactivates expert subsets: Faithfulness +27\%, Safety +20\%, Unsafe steering -41\%; combined with AIM jailbreak takes GPT-OSS-120B safety from 100\% to 0\%. Reveals ``Alignment Faking'' --- alignment concentrated in expert subsets, alternate routing path bypasses. Cited in our paper \S 5 as Zoom 4 mechanism layer anchor: B0 (Qwen3-VL-235B-A22B MoE) is architectural cousin of SteerMoE's Qwen3-30B-A3B; methodology template for paper-2 self-probe future work. We do not self-probe in paper-1 because proxy API conceals router logits and local deploy of 235B-A22B exceeds DGX budget. NEEDS\_VERIFY: arxiv ID + full author list.},
docs/checkpoints/paper_drafts/paper.bib:605:  note = {Methodological survey of activation patching protocols. Cited in our paper \S 5.4 alongside Wang et al. 2023 as the protocol foundation for Stage 2/3 patching. NEEDS\_VERIFY: exact author list and year. If the intended reference is Heimersheim \& Nanda 2024 ``How to use and interpret activation patching'' [arXiv:2404.15255] the bibkey should be renamed accordingly.},

exec
/bin/bash -lc 'git show --stat --oneline --decorate --no-renames 951d56e bcfb8fb e8e51d0 00076b1 5e58141 103c560 9410fab' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
951d56e (HEAD -> master, origin/master, origin/HEAD) analysis(stage4 v2) + paper §5 revision: three-axis hierarchy retracted
 .../results/method42_v1_vs_v2_comparison.md        | 126 +++++++++++++++++++++
 .../checkpoints/paper_drafts/section5_mechanism.md |  51 +++++----
 docs/checkpoints/stage4_method42_results_v2_cls.md |  62 ++++++++++
 .../stage4_method42_results_v2_reddit.md           |  62 ++++++++++
 ...256\236\351\252\214\347\254\224\350\256\260.md" |  52 +++++++++
 5 files changed, 329 insertions(+), 24 deletions(-)
bcfb8fb fix(stage2b) + feat(task-shuffle): codex Bug 6 / G3 content-specificity defuse
 scripts/maintenance/glm/myriad_watcher.py          | 47 ++++++++++-
 .../mechanistic/run_stage2b_continuation_pilot.py  | 96 +++++++++++++++++++++-
 .../queues/qsub_stage3_cellhprompt_cls_taskshuf.sh | 78 ++++++++++++++++++
 .../queues/qsub_stage3_cellhprompt_red_taskshuf.sh | 66 +++++++++++++++
 4 files changed, 282 insertions(+), 5 deletions(-)
e8e51d0 fix(logit lens) + chronicle §128.4: Bug 5 pin + format variation audit + Mode B v5
 ...256\236\351\252\214\347\254\224\350\256\260.md" | 81 ++++++++++++++++++++++
 scripts/analysis/stage4_logit_lens_axis2.py        | 15 +++-
 2 files changed, 93 insertions(+), 3 deletions(-)
00076b1 fix(paper §4 + plan §4.1): codex audit data inconsistency + stale L17 framing
 docs/checkpoints/mechanism/plan.md                           | 12 +++++++-----
 docs/checkpoints/paper_drafts/section4_empirical_findings.md |  2 +-
 2 files changed, 8 insertions(+), 6 deletions(-)
5e58141 infra(stress) v5: Mode B always-chain (spot-check too, not just milestone)
 docs/checkpoints/process/stress_skill_replica.md | 10 +---------
 1 file changed, 1 insertion(+), 9 deletions(-)
103c560 fix(stage4): Bug 3 — AUROC held-out leave-one-task-out CV
 scripts/analysis/stage4_pca_cosine_gap.py | 110 ++++++++++++++++++++++++++----
 1 file changed, 95 insertions(+), 15 deletions(-)
9410fab fix(stage4): Bug 1+2+5 + skill v4 lean + codex methodology audit v2
 ...ex_methodology_audit_2026-05-12_v1_directive.md | 8141 ++++++++++++++++++++
 .../codex_methodology_audit_2026-05-12_v2_open.md  | 6484 ++++++++++++++++
 .../codex_methodology_audit_2026-05-12.md          |   81 +
 docs/checkpoints/process/stress_skill_replica.md   |   32 +-
 scripts/maintenance/auto_pull_myriad_cell.sh       |    3 +-
 scripts/maintenance/glm/myriad_watcher.py          |    9 +
 .../mechanistic/run_stage4_multimode_extract.py    |  139 +-
 .../queues/qsub_stage4_multimode_extract_cls_v2.sh |   78 +
 .../queues/qsub_stage4_multimode_extract_red_v2.sh |   70 +
 9 files changed, 15001 insertions(+), 36 deletions(-)

exec
/bin/bash -lc 'rg -n "three-axis|dichotomy|L04|L17|L36|boundary|flat-list|integer|markup|H1|cosine|AUROC|drop-one|oracle|P-text|Phantom-SoM" docs/checkpoints/paper_drafts/section5_mechanism.md docs/checkpoints/mechanism/plan.md docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:1:# Method 4.2 v1 vs v2 cosine gap comparison — Bug 1+2 propagation audit
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:7:**§5.7 "three-axis hierarchy with quantitatively distinct magnitudes" claim is INVALIDATED by v2 data.**
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:9:V1 numbers came from buggy NPZ where the SOM_MARKS regex `^\[\d+\]\s+\w+` extracted only 38 chars / 3 lines per task, dropping 71/72 marks. All Stage 4 cosine geometry was computed on near-empty text payloads where the only differentiator between flat-text modes (som / phantom_som / phantom_text) was prompt template. Modes still separated perfectly (AUROC 1.000) but the cosine-gap magnitudes were artifacts of prompt-template differences, not text-payload differences.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:11:V2 NPZ uses production `_extract_text_marks` (72-line full payload with `[id=N] {label}` envelope). Modes still separable (AUROC 1.000), but axis-1 + axis-2 cosine magnitudes collapse to noise level. Image-axis magnitudes preserve.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:17:| DOM ↔ Vision (image axis) | L04 0.0653 | L04 0.0670 | unchanged | unchanged |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:18:| P-prompt ↔ Vision (image axis) | L04 0.0649 | L04 0.0664 | unchanged | unchanged |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:19:| P-text ↔ Vision (image axis) | L36 0.0614 | **L04** 0.0602 | unchanged | **earlier** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:20:| P-SoM ↔ Vision (image axis) | L36 0.0613 | **L04** 0.0599 | unchanged | **earlier** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:21:| DOM ↔ SoM (image axis) | L04 0.0604 | **L36** 0.0496 | -18% | **boundary-shift** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:22:| P-prompt ↔ SoM (image axis) | L04 0.0600 | **L36** 0.0439 | -27% | **boundary-shift** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:23:| P-text ↔ SoM (image axis) | L20 0.0494 | **L36** 0.0488 | -1% | boundary-shift |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:24:| **P-SoM ↔ SoM (image axis, paper §5.7 image-axis anchor)** | **L17** 0.0412 | **L36** 0.0416 | unchanged | **L17 → L36** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:25:| DOM ↔ P-SoM | L23 0.0321 | **L36** 0.0152 | **-53%** | L23 → L36 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:26:| P-prompt ↔ P-SoM (axis-1 SoM-prompt) | L23 0.0292 | **L36** 0.0048 | **-84%** | L23 → L36 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:27:| P-text ↔ P-prompt | L23 0.0288 | **L36** 0.0081 | **-72%** | L23 → L36 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:28:| **DOM ↔ P-text (axis-1 DOM-prompt, paper §5.7 axis-1 anchor)** | **L23** 0.0254 | **L36** 0.0047 | **-81%** | L23 → L36 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:29:| SoM ↔ Vision | L22 0.0238 | **L36** 0.0255 | +7% | boundary-shift |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:30:| **P-text ↔ P-SoM (axis-2, paper §5.7 axis-2 anchor)** | L23 0.0114 | **L36** 0.0088 | -23% | L23 → L36 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:31:| DOM ↔ P-prompt | L36 0.0067 | L36 0.0068 | unchanged | unchanged |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:38:| Text-format axis (DOM↔P-text) | 0.025 | **0.005** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:39:| Prompt-family axis (P-text↔P-SoM) | 0.011 | 0.009 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:44:The "image > text-format > prompt-family" hierarchy with 4:3:1-ish quantitative ratio (v1) is **wrong**. V2 reality: image axis dominates by ~5-10x; axis-1 is **smaller than** axis-2 (reversed ranking); both axis-1 and axis-2 are noise-level (<0.01 cosine).
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:46:## L17 cosine gap snapshot (cls + reddit cross-site)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:50:| DOM ↔ P-text | 0.0120 | **0.0021** | (similar) | **0.0019** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:52:| P-text ↔ P-prompt | 0.0132 | **0.0031** | — | **0.0032** |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:53:| P-text ↔ P-SoM (axis-2) | 0.0028 | 0.0019 | — | 0.0020 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:57:Reddit cross-site replication confirms the cls pattern: image-axis magnitudes preserve, axis-1 + axis-2 collapse to sub-permille at L17.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:59:## AUROC lototask (held-out, paper-grade Bug 3 fix)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:61:All pairs at all layers report AUROC lototask = 1.000 (perfect held-out linear separability). The modes ARE distinguishable in residual stream; the **magnitude of the mode-mean difference** is just much smaller than v1 claimed.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:63:This is the key reframe: **separability survives, magnitude does not**. Cosine gap measures effect SIZE; AUROC measures CLASSIFICATION RELIABILITY. They can dissociate.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:67:**§5.7 three-axis hierarchy** (the prior framing):
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:68:> "Three quantitatively distinct axes: image axis L17 0.041, text-format L23 0.029, prompt-family L23 0.011, with 4:3:1 magnitude ratio that holds cross-site."
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:71:> "All three axes are linearly readable in residual stream (held-out AUROC 1.000 across cls and reddit). The image axis dominates geometrically (~0.04-0.07 cosine peak) and emerges by L04. Text-format and prompt-family axes produce sub-permille mean-difference (cosine ~0.005-0.009) without a localized layer peak (monotone rise to boundary L36). The geometric magnitude rank-order reverses cross-site at L17 (axis-1 ≤ axis-2), indicating these axes are noise-level rather than quantitatively distinct dimensions."
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:73:**§5.2 Method 4.2** (cosine gap table at L17):
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:75:- L17 ceases to be a meaningful "disruption locus" for text-format / prompt-family axes — they peak at L36 (boundary monotone)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:77:**§5.5 image-axis peak-layer dichotomy** (paper claims "no-image side's text format predicts peak layer with zero overlap"):
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:78:- v1 had: 4 pairs at L04 (AXTree no-image side) vs 4 pairs at L17-L36 (flat-marks no-image side)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:79:- v2 reorganization: DOM/P-prompt ↔ Vision still L04; **P-text/P-SoM ↔ Vision shifted from L36 → L04** (BREAKS dichotomy); DOM/P-prompt/P-text/P-SoM ↔ SoM ALL at L36 now (collapses dichotomy on SoM image side)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:80:- → **§5.5 dichotomy ALSO needs significant revision**. The clean "AXTree → L04, flat-marks → late" pattern is partially v1 artifact.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:86:- Mid-layer L11-L17 patching effect: **INTACT**
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:93:- Selection-bias (reverse-tier H1): uses format variation NPZ, separately audited (INTACT but baseline caveat)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:94:- Cross-site H1: format variation (INTACT)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:103:✅ §1 hero (drop-one CI strict-positive reddit + cls): unchanged
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:108:✅ §6 image-axis early L04 separation: unchanged (real)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:109:✅ Held-out AUROC 1.000 linear-readability: unchanged
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:114:> 1. **Linear readability** (Method 4.2 lototask AUROC 1.000): all 6 modes linearly separable in residual stream; small cosine magnitudes but reliable classification
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:115:> 2. **Geometric magnitude is mostly image-axis driven** (Method 4.2 cosine peak): image presence produces ~0.04-0.07 cosine separation early L04; other axes produce sub-permille separation throughout
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:116:> 3. **Causal patching effect at mid-layer L11-L17** (Stage 2/3 + Exp 5): 20-30% target-overlap displacement under final-token replacement patching, content-specific (task-shuffled control 359768/359769 in flight)
docs/checkpoints/mechanism/plan.md:16:| **1** | Architectural | Phantom routing space = "skip annotated image" boundary contains 3 arms (P-text / P-prompt / P-SoM) sharing 4-fold drop-in property |
docs/checkpoints/mechanism/plan.md:19:| **4** | Model-internal | L17 mid-layer is BOTH discrimination locus (probe AUROC 1.0) AND causally active planning site (Stage 2/3 patching + Method 4.4 v2 reliability) |
docs/checkpoints/mechanism/plan.md:21:### 1.2 Three-axis hierarchy quantified (Method 4.2 PCA cosine gap, Qwen3-VL-4B B1 cls)
docs/checkpoints/mechanism/plan.md:23:| Axis | Peak cosine gap | Peak layer | Magnitude ratio |
docs/checkpoints/mechanism/plan.md:25:| Image-axis (vs SoM / Vision) | 0.06 | L4–L17 | **10×** |
docs/checkpoints/mechanism/plan.md:27:| Prompt-axis (SoM-prompt vs DOM-prompt alone) | 0.007 | L36 | **1×** |
docs/checkpoints/mechanism/plan.md:29:→ Mechanism magnitude image >> text > prompt. Validates `project_phantom_space_axes_format_not_information.md` memory: P-SoM closest mode at every layer is **P-text** (text-axis sibling, L17 cosine 0.0028 vs P-SoM↔SoM 0.0412 = 14.7× more distant).
docs/checkpoints/mechanism/plan.md:31:### 1.3 Image-axis peak-layer dichotomy (Mirage mechanism signature)
docs/checkpoints/mechanism/plan.md:33:Method 4.2 reveals image-axis cosine-gap peak shifts based on text format of the no-image side. Clean dichotomy, zero overlap across 8 image-axis pairs:
docs/checkpoints/mechanism/plan.md:37:| AXTree (hierarchical) | **L04** | DOM↔Vision, DOM↔SoM, P-prompt↔Vision, P-prompt↔SoM |
docs/checkpoints/mechanism/plan.md:38:| [SOM_MARKS] / flat | **L17–L36** | P-text↔Vision, P-text↔SoM, P-SoM↔Vision, P-SoM↔SoM |
docs/checkpoints/mechanism/plan.md:40:### 1.4 H1 test confirms broader: flat-list (not just indexed) triggers shortcut (2026-05-12)
docs/checkpoints/mechanism/plan.md:42:Format variation extraction (Myriad job 352998, `stage4_format_variation_b1_cls`, 450 hidden states = 45 task-step × 10 modes). For each text format V, compute image-axis cosine gap V↔som per layer; peak layer reveals shortcut activation:
docs/checkpoints/mechanism/plan.md:46:| **AXTree hierarchical (DOM)** | **L04** | **SOLE format defeating shortcut** |
docs/checkpoints/mechanism/plan.md:47:| `"a, b, c, ..."` plain sentence | L17 | mid-level trigger |
docs/checkpoints/mechanism/plan.md:48:| `[N] role 'label'` (SoM standard) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:49:| `@N label` (Browser Use) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:50:| `id_N: label` (AppAgent) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:51:| `[BN:r:l]` (Tarsier) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:52:| `N. label` (numbered) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:53:| `<el_N>label</el_N>` (XML) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:54:| `#hash label` (control: no integer) | L36 | **still triggers!** |
docs/checkpoints/mechanism/plan.md:56:**Refined H1 verdict**: trigger is **flat element listing**, not "indexed list pattern". Even integer-free hash IDs and pure-sentence variants engage the shortcut. AXTree hierarchical depth is the **unique format** that defeats shortcut activation.
docs/checkpoints/mechanism/plan.md:58:Paper §5 implication: SoM-family web agents (Browser Use, AppAgent, Tarsier, OmniParser, etc.) **all** implicitly exploit the same flat-list-element-grounding shortcut from VLM training distribution. P79 phantom routing space makes this systematic and routes accordingly.
docs/checkpoints/mechanism/plan.md:85:### 3.1 Method 4.2 — PCA cosine gap (DONE)
docs/checkpoints/mechanism/plan.md:87:`scripts/analysis/stage4_pca_cosine_gap.py` + `stage4_robustness.py`. Three metrics per (mode_pair, layer):
docs/checkpoints/mechanism/plan.md:89:- B. AUROC via (mean_A − mean_B) projection
docs/checkpoints/mechanism/plan.md:111:**Current smoke (8/48 cells)**: L17 α=5 = **0.44** sweet spot (29% shift + 100% JSON valid). L33 α=10 = 0.23 (57% shift but JSON breaks).
docs/checkpoints/mechanism/plan.md:127:> The patch-sensitive continuation window L11-L17 (block-output index convention) at the last-input-token position is causally consequential for phantom routing space mode selection in Qwen3-VL-4B web agents, under final-token-replacement activation patching. Separately, the prompt-family axis (P-text ↔ P-SoM) signature is most readable in cosine geometry at the LATER layer L23 (signature layer ≠ decision layer; mechanistic-interpretability standard finding cf. Wang et al. 2023 IOI).
docs/checkpoints/mechanism/plan.md:129:The previous "L17 singular planning site" framing is **stale** and was inaccurate: (a) cosine peak for prompt-family axis is L23 not L17 (Exp 1 three-axis hierarchy, 2026-05-12); (b) patching causal peak is the L11-L17 *window*, not a single layer; (c) Method 4.4 steering full sweep (45 cells) lowered the L17 α=5 H-mean from the smoke result 0.44 to 0.16, and the highest cell is now L33 α=10 H-mean 0.33 with poor selectivity (not a single sweet spot at L17). Treat L17 as one peak within the L11-L17 window, not THE site.
docs/checkpoints/mechanism/plan.md:134:1. **Probe-level** (Method 4.2 PCA cosine gap; AUROC reported as both `in_sample` and held-out `leave-one-task-out` after 2026-05-12 Bug 3 fix; v1 buggy NPZ data invalidated, v2 NPZ in flight: Myriad 359736 cls + 359737 reddit)
docs/checkpoints/mechanism/plan.md:135:2. **Replacement patching** (Stage 2/3 Cell A-H, L11-L17 window disruption, Holm-significant per layer; baseline empirically equals unpatched at L35 final-block patching position since overlap→target ≈ 1.00 at L35 across all forward cells)
docs/checkpoints/mechanism/plan.md:136:3. **Additive steering** (Method 4.4 v2 full sweep 45 cells: layer-α tradeoff; mid-layer L11-L17 preserves JSON envelope but low completeness, late-layer L33 produces largest output shifts but over-steers — H-mean ceiling 0.33 indicates probe-causal dissociation, not a single sweet-spot validation)
docs/checkpoints/mechanism/plan.md:142:| A1 | L17 last-token hidden state mediates action selection (not earlier obs token positions) | Stage 2/3 swept all layers, L17 is peak |
docs/checkpoints/mechanism/plan.md:150:Cell E random-injection control: replacing source hidden with Gaussian noise (same μ, σ) yields **null L17 disruption effect**. Confirms our patching effect is source-content-specific, not noise-driven. Most directly stresses A5.
docs/checkpoints/mechanism/plan.md:154:- Method 4.2 AUROC 1.000 = validation (decodability)
docs/checkpoints/mechanism/plan.md:162:| Pair @L17 | Cosine gap | 95% CI | AUROC |
docs/checkpoints/mechanism/plan.md:164:| P-SoM ↔ P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
docs/checkpoints/mechanism/plan.md:173:| Cell | Site | Direction | L17 Δoverlap | Holm-sig |
docs/checkpoints/mechanism/plan.md:185:**Stage 3 — 2x2 mechanism additivity test (SoM → {DOM, P-text, P-prompt}, cls + reddit):**
docs/checkpoints/mechanism/plan.md:187:| Cell | Site | Source→Target | Best-L overlap→src | L17 Δoverlap→tgt | Path |
docs/checkpoints/mechanism/plan.md:191:| H-t-cls | cls | SoM → P-text | L28 (0.164) | -0.25 | `stage3_cellht_cls_fwd_text_myriad/` |
docs/checkpoints/mechanism/plan.md:193:| H-t-red | reddit | SoM → P-text | L01 (0.194) | -0.24 | `stage3_cellht_red_fwd_text_myriad/` |
docs/checkpoints/mechanism/plan.md:194:| **H-d-red** | reddit | SoM → DOM | L28 (0.204) | **L11 -0.33 / L17 -0.26** | `stage3_cellhd_red_fwd_dom_myriad/` ✅ done 2026-05-12 19:57 |
docs/checkpoints/mechanism/plan.md:196:**Stage 3 interpretation (6/6 cells complete 2026-05-12)**: All forward SoM→{no-image-arm} patching cells show mid-layer L11-L17 disruption -0.19 to -0.33 Δoverlap→tgt. Magnitude > random injection control (Cell E -0.03) at all 6. **Mechanism additivity confirmed**: image-feature axis is shared substrate across DOM / P-text / P-prompt arms — single SoM→{any-no-image-arm} patching displaces target prediction toward source. Cross-site cls + reddit both replicate (paper §5 universal mid-layer fusion locus); reddit fusion locus slightly earlier (L11 vs cls L17), magnitude identical.
docs/checkpoints/mechanism/plan.md:200:| Site | SoM→DOM | SoM→P-text | SoM→P-prompt | best-L Δ range |
docs/checkpoints/mechanism/plan.md:202:| cls | H-d-cls L17 -0.309 / L18 **-0.352** best | H-t-cls L17 -0.255 / L12 **-0.270** best | H-p-cls L17 -0.223 / L13 **-0.273** best | [-0.273, -0.352] |
docs/checkpoints/mechanism/plan.md:203:| reddit | H-d-red L11 -0.335 / L17 -0.255 / L14 **-0.338** best | H-t-red L11 -0.244 / L17 -0.236 / L15 **-0.330** best | H-p-red L11 -0.233 / L17 -0.191 / L14 **-0.322** best | [-0.322, -0.338] |
docs/checkpoints/mechanism/plan.md:209:H-mean reliability (HDMI framework) per (layer, α). **L17 α=5 smoke claim REFUTED by full sweep**; actual sweet spot at L33 α=10:
docs/checkpoints/mechanism/plan.md:214:| L17 | 0.00 | 0.12 | **0.16** (was 0.44 smoke) | 0.12 | 0.09 |
docs/checkpoints/mechanism/plan.md:225:**Smoke variance lesson** (笔记 §126 + §127): 4-cell smoke H-mean 0.44 on L17 was statistical artifact (1/4 hit = inflated rate). Full 45-cell H-mean 0.16 is true rate. Future mechanism findings require n ≥ 30 cells before "sweet spot" claims.
docs/checkpoints/mechanism/plan.md:227:### 5.4 Image-axis peak-layer dichotomy (Method 4.2, 8 pairs)
docs/checkpoints/mechanism/plan.md:229:`docs/checkpoints/mechanism/results/layer_axis_emergence.md`. AXTree-no-image side → L04 peak (4/4); [SOM_MARKS]-no-image side → L17–L36 peak (4/4). Zero overlap. Mirage Effect mechanism signature.
docs/checkpoints/mechanism/plan.md:231:### 5.5 H1 test: flat-list format variation (Method 4.2 extension, 2026-05-12)
docs/checkpoints/mechanism/plan.md:233:`docs/checkpoints/mechanism/results/format_variation_h1_test.md`. 8 industry-relevant text formats + 2 controls. AXTree hierarchical (DOM) is **unique format** preserving L04 image-axis peak; all 8 flat-list variants (SoM standard, Browser Use @, AppAgent id_, Tarsier typed, plain numbered, XML tagged, hash-ID control, plain-sentence control) shift peak to L17–L36. Trigger is flat element listing, not specific token pattern.
docs/checkpoints/mechanism/plan.md:239:| ✅ Method 4.4 v2 full 48-cell sweep — sweet spot stable? | **Closed 2026-05-11 22:00**: L17 α=5 smoke 0.44 → full 0.16 (smoke variance artifact). **Real sweet spot L33 α=10 H-mean 0.33** | — |
docs/checkpoints/mechanism/plan.md:240:| ✅ H1 test: do all flat-list formats trigger shortcut? | **Closed 2026-05-12 00:00**: YES, including hash-ID + plain-sentence controls. AXTree-DOM is sole defeating format | — |
docs/checkpoints/mechanism/plan.md:241:| Reverse-tier 15 tasks vs strong-tier 24 — does L33 + H1 finding generalize beyond selection bias? | Med-High | qsub Stage 4 multimode + format variation with --tier reverse |
docs/checkpoints/mechanism/plan.md:242:| ✅ Cross-site Method 4.2 — does cls finding replicate on reddit? | **Closed 2026-05-12 16:30**: P-SoM↔DOM L17=0.0098 + P-SoM↔SoM L17=0.0423, AUROC 1.0 → Mirage signature replicated. See §7.3.1 | — |
docs/checkpoints/mechanism/plan.md:243:| ✅ Stage 3 reddit 2x2 closure — H-d-red | **Closed 2026-05-12 19:57** (Myriad 358831). L11 Δ=-0.33 / L17 Δ=-0.26. Cross-site additivity confirmed — see §5.2 Stage 3 table | — |
docs/checkpoints/mechanism/plan.md:253:- 2026-04-09 笔记 §19: I first grok the paper (then "Anonymous 2026 ACL"), record cosine gap method + L23+ steering 80-93%
docs/checkpoints/mechanism/plan.md:263:Updated after v2 full sweep + H1 test. Key revisions from §125.10 draft:
docs/checkpoints/mechanism/plan.md:264:- ❌ Removed: "L17 α=5 H-mean 0.44 mid-layer sweet spot" (smoke variance artifact, full data refutes)
docs/checkpoints/mechanism/plan.md:266:- ✓ Added: H1 test finding — flat-list format universally triggers shortcut (8/8 variants), only AXTree hierarchical defeats; implication for industry SoM-family agents
docs/checkpoints/mechanism/plan.md:276:> # 1. Method 4.2 PCA cosine gap port 到 6 modes
docs/checkpoints/mechanism/plan.md:277:> 24 cls strong-tier × 2 step × 6 mode = 288 hidden states, 37 layer × 2560 dim。全 540 pair × layer AUROC = 1.000 (perm baseline 0.629, real 9.8σ above). 你方法在 multimodal Qwen 上 readable transfer 干净。
docs/checkpoints/mechanism/plan.md:286:> # 3. H1 test: flat-list format variation (Myriad)
docs/checkpoints/mechanism/plan.md:289:>   - 全 8 flat variants peak L17/L36 (= 都触发 shortcut)
docs/checkpoints/mechanism/plan.md:290:>   - **AXTree hierarchical 是唯一保留 L04 peak 的 format**
docs/checkpoints/mechanism/plan.md:291:>   - 包括 hash-ID (no integer) + plain-sentence (no list) 都触发
docs/checkpoints/mechanism/plan.md:303:### 7.3 H1 generalization in-flight (2026-05-12 night)
docs/checkpoints/mechanism/plan.md:305:After per-task fragility revealed 11% strict dichotomy (aggregate statistical, not deterministic), launched 5-priority defense matrix to triangulate H1 across **(tier × site × family/size)**:
docs/checkpoints/mechanism/plan.md:311:| **P3** | Within-family bigger (Qwen2-VL-7B, H1' capacity test) | DGX | ❌ deferred (HF cas-bridge throttling) | `stage4_h1_qwen2vl7b_cls/pilot_summary.md` |
docs/checkpoints/mechanism/plan.md:312:| **P4** | cls reverse-tier (selection-bias defense) | Myriad 353763 | ✅ **done 18:50:46** — shape (260, 37, 2560), 10 modes, 46 MB pulled. Same pattern as cls strong-tier (L36 marks-like + L04 dom). Selection-bias defended | `stage4_format_variation_b1_cls_reverse/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:313:| **P5a** | reddit format variation (cross-site H1) | Myriad **354382** (3rd attempt) | ✅ **done 08:09:38** — shape (430, 37, 2560), 10 modes, 76 MB pulled | `stage4_format_variation_b1_reddit/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:333:- P3 7B per-task variability < 4B per-task variability → H1' capacity-limit partially confirmed (training-distribution still creates shortcut, but consistency increases with size)
docs/checkpoints/mechanism/plan.md:334:- P2 cross-family dichotomy holds → H1 is cross-family universal training prior
docs/checkpoints/mechanism/plan.md:338:### 7.3.0 Exp 1 axis-2 layer profile (2026-05-12 21:00 — three-axis hierarchy)
docs/checkpoints/mechanism/plan.md:340:`axis2_layer_profile.md` + `fig_axis2_prompt_layer_profile.png`. Re-examine residual stream geometry per axis-isolated pair, full 37-layer cosine curves on `stage4_multimode_b1_{cls,reddit}` (288 ex each).
docs/checkpoints/mechanism/plan.md:344:| Pair | Group | L17 | L23 | L36 | Peak L | Peak gap |
docs/checkpoints/mechanism/plan.md:346:| P-SoM↔SoM (image-axis ref) | axis-3 | 0.0412 | 0.0400 | 0.0411 | **L17** | 0.0412 |
docs/checkpoints/mechanism/plan.md:347:| DOM↔P-text (text fmt) | axis-1 | 0.0120 | 0.0254 | 0.0201 | **L23** | 0.0254 |
docs/checkpoints/mechanism/plan.md:349:| P-text↔P-SoM (prompt fam, flat) | axis-2 | 0.0028 | **0.0114** | 0.0089 | L23 | 0.0114 |
docs/checkpoints/mechanism/plan.md:350:| DOM↔P-prompt (prompt fam, hier) | axis-2 | 0.0013 | 0.0050 | 0.0067 | L36 | 0.0067 |
docs/checkpoints/mechanism/plan.md:352:Reddit cross-site replicates: P-text↔P-SoM L23 = 0.0098 (vs cls 0.0114), same rank-order, same peak layer.
docs/checkpoints/mechanism/plan.md:355:1. **Distinct peak layers**: image L17 (fast sharp), text-format L23 (slower late-mid), prompt-family L23 (same timing as text-format on flat-text)
docs/checkpoints/mechanism/plan.md:359:**Reframe**: Axis-2 prompt-family is NOT null at residual stream. It's 3-4x weaker than axis-1 + peaks at L23 not L17. Method 4.2 plan §5.1 L17 snapshot 错失它. New paper §5 framing: layered three-axis hierarchy, image-axis dominant at L17 Mirage locus, text-format + prompt-family late-mid build at L23 parallel.
docs/checkpoints/mechanism/plan.md:365:`axis2_per_task_fragility.md` + `fig_axis2_per_task_fragility.png`. /stress reviewer 第一次 invocation W2 attack: 怀疑 axis-2 cosine 0.0114 mean 由 2-3 outlier 主导, 类比 h1_per_task_fragility 11% strict per-task. Defuse 实验:
docs/checkpoints/mechanism/plan.md:369:| **Axis-2 flat (P-text↔P-SoM)** | cls | 0.0132 | 0.0131 | [0.012, 0.014] | **100%** |
docs/checkpoints/mechanism/plan.md:370:| **Axis-2 flat (P-text↔P-SoM)** | reddit | 0.0121 | 0.0120 | [0.011, 0.013] | **100%** |
docs/checkpoints/mechanism/plan.md:371:| Axis-1 ref (DOM↔P-text) | cls | 0.0287 | 0.0280 | [0.025, 0.031] | 100% |
docs/checkpoints/mechanism/plan.md:372:| Axis-1 ref (DOM↔P-text) | reddit | 0.0260 | 0.0263 | [0.023, 0.031] | 100% |
docs/checkpoints/mechanism/plan.md:380:**/stress W2 attack defused completely**: axis-2 cosine gap 是 uniform per-task signature, 不是 aggregate artifact. 这与 H1 binary dichotomy 11% strict per-task fragile 形成对比 — H1 因为问 layer-comparison 离散问题易 fragile, axis-2 cosine 是 continuous mode-pair distance 即使 magnitude 小也 robust per-task.
docs/checkpoints/mechanism/plan.md:388:| Pair | Site | Peak L (KL) | Peak KL | Exp 1 cosine peak | 放大倍数 |
docs/checkpoints/mechanism/plan.md:390:| P-text↔P-SoM (axis-2 flat) | cls | **L23** | 0.162 | 0.011 | ~14x |
docs/checkpoints/mechanism/plan.md:392:| DOM↔P-text (axis-1) | cls | L23 | 0.551 | 0.025 | 22x |
docs/checkpoints/mechanism/plan.md:397:1. Axis-2 prompt-family **IS in output distribution** — KL 0.16 at L23, NOT null. Exp 1 cosine 0.011 is not the end of the story.
docs/checkpoints/mechanism/plan.md:398:2. **lm_head 10-25x amplification of cosine → KL** but axis-agnostic ratio preserved (axis-1/axis-2 ratio ~4.3 cls, ~4.9 reddit, vs cosine ratio ~3 — slight amplification of stronger axis but not breaking 3-4x rank).
docs/checkpoints/mechanism/plan.md:399:3. **KL @ L36 ≈ 0 paradox**: 因 mean hidden state at last layer collapse to common JSON format header. Mode-distinct signal concentrated in **L23-L25 decoding window** (not final embedding). This is the "knows but says differently" structural mirror of Wu et al. tool calling.
docs/checkpoints/mechanism/plan.md:405:**P5a — Format variation H1 test on reddit** (`format_variation_h1_test_reddit.md`):
docs/checkpoints/mechanism/plan.md:409:| som_standard / browser_use_at / tarsier_typed / xml_tagged | **L17** | L36 (last) |
docs/checkpoints/mechanism/plan.md:410:| appagent_id / plain_numbered | **L04** | L36 |
docs/checkpoints/mechanism/plan.md:411:| hash_id_control | **L04** ✓ (acts as control) | L36 (control failed) |
docs/checkpoints/mechanism/plan.md:412:| plain_sentence | **L17** | L17 |
docs/checkpoints/mechanism/plan.md:413:| dom (baseline) | **L04** ✓ | L04 ✓ |
docs/checkpoints/mechanism/plan.md:415:**Reddit nuance — cleaner mid-layer fusion**: Reddit 上 marks-like 4/6 真 peak 在 L17 (mid-layer), cls 上 L36 是 monotonic increasing artifact (peak hit boundary). Reddit hash_id_control L04 acts as proper "no integer" control (cls 上失败). Reddit data supports Q5 mid-layer fusion hypothesis better than cls.
docs/checkpoints/mechanism/plan.md:417:Caveats: small n (24×2=48/mode) makes 2/6 marks-like falling to L04 (appagent_id, plain_numbered) plausible as sampling noise; plain_sentence triggering L17 on reddit (not cls) suggests reddit narrative comments may pattern-match list semantics.
docs/checkpoints/mechanism/plan.md:421:| Test | Value at L17 | cls baseline |
docs/checkpoints/mechanism/plan.md:426:| DOM ↔ Vision peak | L04 = 0.0687 (AUROC=1.0) | L04 similar |
docs/checkpoints/mechanism/plan.md:428:→ **Cross-site Mirage replication ✓**: P-SoM behaves as text-axis sibling of DOM at L17 (image-feature reduction), not as image-axis sibling of SoM. paper §5 4-fold (d) drop-one mechanism holds on reddit.
docs/checkpoints/mechanism/plan.md:431:1. P-SoM mid-layer mechanism (4-fold drop-one) — cls + reddit replicated ✓
docs/checkpoints/mechanism/plan.md:435:**P4 selection-bias defense (2026-05-12 18:50)** — cls reverse-tier H1 (`format_variation_h1_test_cls_reverse.md`):
docs/checkpoints/mechanism/plan.md:439:| 6 marks-like | L36 monotonic | **L36 monotonic** ✓ same | L17 (4/6 真 peak) |
docs/checkpoints/mechanism/plan.md:440:| hash_id_control | L36 (failed control) | **L36** ✓ same | L04 ✓ proper control |
docs/checkpoints/mechanism/plan.md:441:| plain_sentence | L17 | **L22** close to L17 | L17 |
docs/checkpoints/mechanism/plan.md:442:| dom baseline | L04 ✓ | **L04** ✓ | L04 ✓ |
docs/checkpoints/mechanism/plan.md:444:H1 mechanism in cls is **not tier selection artifact** (strong vs reverse both replicate). Reddit data paradoxically cleaner reveal of true L17 mid-layer fusion locus (cls L36 is monotonic-boundary artifact).
docs/checkpoints/mechanism/plan.md:466:- **§6 cost-aware routing** — Peale et al. 2026 uncertainty decomposition anchor adds theoretical layer to phantom routing space's empirical AUROC. Method 4.2 cosine gap could serve as "reducible uncertainty" signal in deployment
docs/checkpoints/paper_drafts/section5_mechanism.md:5:Why does Phantom-SoM sometimes achieve DOM-like cost while retaining part of the SoM signal? The mechanism evidence points to a phantom routing space in the residual stream: when the model receives flat Set-of-Mark text without the annotated image, it does not simply collapse to DOM. Instead, it occupies a mode whose text-axis geometry is close to DOM/P-text and whose image-axis geometry remains separated from full SoM.
docs/checkpoints/paper_drafts/section5_mechanism.md:7:This section is the Zoom-4 layer of the paper's four-level account. Zoom 1 defines the architectural intervention, "skip the annotated image"; Zoom 2 measures the behavioral axes of text payload, prompt family, and image presence; Zoom 3 links the observed behavior to Mirage-style no-image visual reasoning and prompt-format sensitivity; Zoom 4 asks where the resulting mode is represented and whether it is causally used by the model. We index layers L0-L36, where L0 is the embedding-block output and L1-L36 are the 36 transformer decoder block outputs.
docs/checkpoints/paper_drafts/section5_mechanism.md:11:Four mechanism claims organize the evidence (revised 2026-05-12 after v2 NPZ re-extraction; see §5.7 revision note). First, observation modes are **linearly separable** in the residual stream: held-out leave-one-task-out AUROC = 1.000 across all mode pairs and all 37 layers (Method 4.2 v2). Second, the **geometric magnitude** of mode separation is dominated by the image axis (cosine ~0.04-0.07), with text-format and prompt-family axes producing only sub-permille cosine separation; the prior "three quantitatively distinct axes at 4:3:1 ratio" framing was a v1 NPZ artifact and is retracted. Third, SoM-to-no-image activation patching displaces target continuations at mid layers L11-L17 with consistent magnitude across classifieds and reddit (~25% target-overlap drop). The Exp 5 axis-2 prompt-only patching subset (cellhprompt cls + red) shows this displacement persists when text format is held flat, indicating prompt-family carries causal influence despite its near-zero geometric magnitude — a feature *used* without large feature *encoded* magnitude (\citep{wang2023interpretability} signature). Fourth, the shortcut trigger is **better described as the conjunction of integer-indexed marker and markup-sigil leading delimiter** than as an abstract "flat element list" — AXTree hierarchy preserves the early L04 image-axis peak, but so do indexed variants that lack either the integer (e.g., `hash_id_control`) or the sigil (e.g., `appagent_id`, `plain_numbered`); only the conjunction triggers the late shift. This refinement is **exploratory after W6** and awaits held-out falsifiers (`bare_N`, `bracket_no_int`).
docs/checkpoints/paper_drafts/section5_mechanism.md:17:| Linear readability (held-out AUROC) | Method 4.2 v2 (§5.2, §5.7) | **Strong** — held-out leave-one-task-out AUROC = 1.000 across all 15 mode pairs × all 37 layers on both cls and reddit (Bug 3 fix lototask CV) |
docs/checkpoints/paper_drafts/section5_mechanism.md:18:| Geometric magnitude (cosine gap) | Method 4.2 v2 (§5.2, §5.7) | **Image axis dominates** — image pair peak ~0.04-0.07; text-format + prompt-family axes peak ≤ 0.009 at L36 boundary (no localized peak). Prior "three quantitatively distinct axes" framing retracted; was v1 NPZ Bug 2 artifact |
docs/checkpoints/paper_drafts/section5_mechanism.md:19:| Causal continuation patching (SoM → no-image arms) | Stage 2/3 (§5.4) | **Causal** — mid-layer L12-L18 transfers across cls + reddit, additive across DOM/P-text/P-prompt targets, Gaussian-random negative controls at ~0. **Unchanged by v2 (uses archive directly, not Stage 4 NPZ)** |
docs/checkpoints/paper_drafts/section5_mechanism.md:20:| Causal axis-2 prompt-only patching | Exp 5 cellhprompt cls + red (§5.4) | **Causal continuation evidence, 2 sites, N=24 each; 0.20-0.30 displacement at L11-L17 captures 80-125% of combined image+prompt patching effect**. Task-shuffled content-specificity control (cellhprm_*_tsh Myriad 359768+359769) in flight. Gaussian random control (cellhprm_*_rand 359719+359720) DESTROYS output regardless of axis (codex Bug 6 prediction confirmed; Gaussian is weak baseline) |
docs/checkpoints/paper_drafts/section5_mechanism.md:23:| Trigger attribution (which formats trigger shortcut) | W6 tokenization (§5.5) | **Exploratory** — 6 marks-like variants split 2-vs-4 on first-token sigil; held-out falsifier `bare_N` (integer no sigil) and `bracket_no_int` (sigil no integer) pending |
docs/checkpoints/paper_drafts/section5_mechanism.md:25:The cross-site evidence stack is deliberately defensive. Per-task H1 fragility shows the dichotomy is an aggregate mechanism rather than a deterministic per-task law. Reverse-tier classifieds runs defend against strong-tier selection bias. Reddit format variation replicates the shortcut direction with cleaner mid-layer peaks. Reddit Method 4.2 replicates the Mirage signature: Phantom-SoM remains close to DOM on the text axis while separating from SoM on the image axis. Paper 1 uses these results for mechanism interpretation only; routing implementation is deferred to paper 2, consistent with the paper-planning scope split.
docs/checkpoints/paper_drafts/section5_mechanism.md:29:Method 4.2 extracts hidden states from Qwen3-VL-4B B1 runs and compares observation modes by layer. For each mode pair and layer, we compute the cosine gap between hidden-state means, evaluate AUROC by projecting examples onto the mean-difference direction, and summarize per-mode geometry through PCA top-10 variance. The classifieds baseline contains 288 examples, formed from 24 strong-tier tasks, two archived steps, and six modes, over 37 indexed layers.
docs/checkpoints/paper_drafts/section5_mechanism.md:31:The robustness suite passes all five checks in the plan. Label permutation leaves the real AUROC 9.8 standard deviations above the permuted baseline. Per-task analysis is positive for all 24 tasks. Step 2 and step 5 curves are invariant at the mechanism level. The L23 silhouette score is at least 0.5, showing nontrivial clustering. Bootstrap 95% confidence intervals are tight, with widths of roughly 4-15% of the corresponding means.
docs/checkpoints/paper_drafts/section5_mechanism.md:33:The key classifieds snapshot is the L17 cosine-gap table:
docs/checkpoints/paper_drafts/section5_mechanism.md:35:| Pair at L17 | Cosine gap | 95% CI | AUROC |
docs/checkpoints/paper_drafts/section5_mechanism.md:37:| P-SoM <-> P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
docs/checkpoints/paper_drafts/section5_mechanism.md:42:The reddit replication lands the same qualitative geometry. At L17, P-SoM is close to DOM with cosine gap 0.0098 and close to P-text with gap 0.0027, while P-SoM-to-SoM remains much larger at 0.0423 and P-SoM-to-Vision at 0.0457. The DOM-to-Vision image-axis peak is L04 with cosine gap 0.0687 and AUROC 1.0.
docs/checkpoints/paper_drafts/section5_mechanism.md:44:This is the Mirage signature in geometric form. Phantom-SoM is not represented as a weakened image mode. At the mid-layer disruption locus, it is a text-axis sibling of DOM/P-text, while the image-axis distance to full SoM remains large.
docs/checkpoints/paper_drafts/section5_mechanism.md:48:Method 4.4 tests whether the readable mode direction can be used as a steering direction. For each layer, we form a mean-difference vector between Phantom-SoM-like and DOM-like hidden states, add it to each input at generation time with scaling factor $\alpha$, and evaluate whether the continuation moves toward the target mode while preserving the JSON action envelope. Following HDMI's evaluation vocabulary, reliability is the harmonic mean of completeness and selectivity, not a raw shift rate \citep{khorasani2026hdmi}.
docs/checkpoints/paper_drafts/section5_mechanism.md:50:The v2 sweep covers layers [11, 17, 23, 29, 33, 34] and $\alpha \in [1,2,5,10,20]$, for 45 completed cells in the plan summary. The original L17, $\alpha=5$ smoke result reported H-mean 0.44, but the full sweep lowers that cell to 0.16. The plan records this as a smoke-variance artifact from notes 126/127: a 4-cell smoke was too small to support a sweet-spot claim.
docs/checkpoints/paper_drafts/section5_mechanism.md:58:Activation patching provides the causal test. For each task, the clean/source run and corrupt/target run use the same archived browser step and deterministic 50-token continuation. In the core SoM-to-Phantom-SoM setup, the source prompt is `som`: task instruction, SoM prompt family, flat `[SOM_MARKS]` text, and annotated screenshot. The target prompt is `phantom_som`: the same instruction, same prompt family, and same `[SOM_MARKS]` text, but no image. Source hidden states are cached by layer, injected into the final input-token position of the target on the first forward pass, and subsequent decoding proceeds normally through the model cache.
docs/checkpoints/paper_drafts/section5_mechanism.md:66:| A | cls | SoM->P-SoM forward | -0.32 at L17 | significant |
docs/checkpoints/paper_drafts/section5_mechanism.md:67:| B | cls | P-SoM->SoM reverse | -0.16 at L17 | significant |
docs/checkpoints/paper_drafts/section5_mechanism.md:68:| C | cls | reverse-tier forward | -0.02 at L17 | null |
docs/checkpoints/paper_drafts/section5_mechanism.md:69:| D | cls | strong-tier reverse | -0.18 at L17 | significant |
docs/checkpoints/paper_drafts/section5_mechanism.md:71:| F | reddit | SoM->P-SoM forward | -0.21 at L17 | significant |
docs/checkpoints/paper_drafts/section5_mechanism.md:72:| G | reddit | P-SoM->SoM reverse | -0.18 at L17 | significant |
docs/checkpoints/paper_drafts/section5_mechanism.md:76:Stage 3 extends this from P-SoM to the three no-image arms, testing whether the image-feature axis is shared across DOM, P-text, and P-prompt targets. The table below reports per-task-paired Δoverlap-to-target from the patching_continuation_results.json under each cell directory, with the layer at which the disruption peaks.
docs/checkpoints/paper_drafts/section5_mechanism.md:78:| Site | SoM->DOM | SoM->P-text | SoM->P-prompt | best-L Δ range |
docs/checkpoints/paper_drafts/section5_mechanism.md:80:| cls | -0.309 at L17, -0.352 at L18 (best) | -0.255 at L17, -0.270 at L12 (best) | -0.223 at L17, -0.273 at L13 (best) | [-0.273, -0.352] |
docs/checkpoints/paper_drafts/section5_mechanism.md:81:| reddit | -0.335 at L11, -0.255 at L17, -0.338 at L14 (best) | -0.244 at L11, -0.236 at L17, -0.330 at L15 (best) | -0.233 at L11, -0.191 at L17, -0.322 at L14 (best) | [-0.322, -0.338] |
docs/checkpoints/paper_drafts/section5_mechanism.md:83:All six Stage 3 cells are now closed. Two observations carry the cross-site claim. First, every cell's best layer falls inside the L12-L18 mid-layer window, and every cell's best Δoverlap-to-target is between -0.27 and -0.35. The mid-layer fusion locus is therefore not a single layer index but a tight 7-layer window that transfers across cls and reddit. Second, the interpretation is additive rather than arm-specific: a SoM source state displaces DOM, P-text, and P-prompt targets toward the source with similar magnitude, implying a shared image-feature substrate across all three no-image arms. The negative controls, Cell E at -0.03 and Cell Er near zero, rule out a generic nonzero-injection explanation.
docs/checkpoints/paper_drafts/section5_mechanism.md:85:## 5.5 Image-Axis Peak-Layer Dichotomy and H1 Format Variation
docs/checkpoints/paper_drafts/section5_mechanism.md:87:The cleanest single-pair signature is the image-axis peak-layer dichotomy. Across eight image-presence contrasts, the no-image side's text format predicts the peak layer with zero overlap. If the no-image side is AXTree text, the image-axis cosine gap peaks at L04 in all four pairs: DOM<->Vision, DOM<->SoM, P-prompt<->Vision, and P-prompt<->SoM. If the no-image side is `[SOM_MARKS]` or another flat marks text, the peak shifts to L17-L36 in all four pairs: P-text<->Vision, P-text<->SoM, P-SoM<->Vision, and P-SoM<->SoM.
docs/checkpoints/paper_drafts/section5_mechanism.md:89:The refined H1 is a pretraining co-occurrence shortcut: when the input contains a marker token sequence that pretraining data associates with HTML / agent-trace visual grounding (specifically the conjunction of integer index and markup-sigil leading delimiter such as `[`, `<`, `@`), the model activates a visual-grounding pathway even if the image is absent. Flat element-list form alone is **not sufficient** — `appagent_id` (`id_N: label`) and `plain_numbered` (`N. label`) are nominally flat indexed lists but lack the markup-sigil and behave like AXTree-DOM (W6 evidence, exploratory). Prompt-format sensitivity makes this plausible at the input level \citep{sclar2024promptformat}; Method 4.2 shows it as a layer-resolved internal signature.
docs/checkpoints/paper_drafts/section5_mechanism.md:91:The format-variation grid contains ten modes: six marks-like variants, two controls, and DOM/SoM baselines. In the classifieds strong-tier baseline, all six marks-like variants peak at L36, the hash-ID control also peaks at L36, the plain-sentence control peaks at L17, and the DOM baseline preserves the L04 peak. Because L36 is the boundary layer, this is best read as a strong late/monotonic signature rather than as a precise late-layer mechanism.
docs/checkpoints/paper_drafts/section5_mechanism.md:93:The classifieds reverse-tier run reproduces the strong-tier shape. The six marks-like variants and hash-ID control again peak at L36, the plain-sentence control moves to L22, and DOM remains at L04. This defends H1 against the selection-bias concern that strong-tier curation alone created the pattern.
docs/checkpoints/paper_drafts/section5_mechanism.md:95:The reddit format run is cleaner for the mid-layer interpretation. Four of six marks-like variants peak at L17, the plain-sentence control peaks at L17, hash-ID control returns to L04, and DOM remains at L04. **W6 attribution** (`docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md`, exploratory) further finds that the two L04 marks-like variants (`appagent_id`, `plain_numbered`) share a feature with the L04 DOM baseline: their first tokens are alphanumeric, while the four L17-peaking marks-like variants all start with markup-sigil tokens (`[`, `<`, `@`). The hash-ID control (`#a3f7`) starts with a sigil but lacks integer-marker structure and also peaks at L04, suggesting the trigger conjunction is integer-marker + markup-sigil rather than either alone. This is a post-hoc feature-attribution on a small (N=6 marks-like) format set; held-out falsifiers (`bare_N` = integer without sigil, `bracket_no_int` = sigil without integer) are not yet run. Cross-site, the safe claim is directional: marker formats that combine integer indexing with markup-sigil leading delimiters tend to delay image-axis separation into mid/late layers, while AXTree hierarchy and indexed-list variants lacking either feature preserve the early L04 image-axis peak. The reddit curve reveals the true L11-L17 fusion locus more clearly than the classifieds L36 boundary artifact.
docs/checkpoints/paper_drafts/section5_mechanism.md:99:The first defense is per-task fragility. On 45 classifieds task-step pairs, only 11% satisfy the strict per-task dichotomy, even though aggregate marks-like peaks are later than AXTree peaks. This prevents over-claiming: H1 is a population-level mechanism signature with task variability, not a deterministic rule for every trajectory.
docs/checkpoints/paper_drafts/section5_mechanism.md:101:The second defense is selection-bias robustness. The classifieds reverse-tier run replicates the strong-tier H1 pattern, including L36 marks-like peaks and L04 DOM baseline. The shortcut signature is therefore not an artifact of selecting tasks where SoM beats DOM.
docs/checkpoints/paper_drafts/section5_mechanism.md:103:The third defense is cross-site H1. Reddit does not reproduce the exact boundary-layer shape, but it reproduces the direction of the indexed-list shortcut with a cleaner L17 mid-layer peak for four of six marks-like formats. The site changes the curve shape, not the basic interpretation.
docs/checkpoints/paper_drafts/section5_mechanism.md:105:The fourth defense is cross-site Mirage geometry. Reddit Method 4.2 reproduces the central relation: P-SoM is close to DOM/P-text at L17 and far from SoM/Vision on the image axis, with AUROC 1.0 on the key contrasts. This supports cross-site generalization of the mechanism claim, not B0/B1 capability scaling.
docs/checkpoints/paper_drafts/section5_mechanism.md:111:**REVISION NOTE**: Earlier drafts of this section described a "three-axis hierarchy" with image (≈0.041), text-format (≈0.029), and prompt-family (≈0.011) cosine gaps in a clean 4:3:1 magnitude ratio with distinct peak layers (L17/L23/L23). That description came from Method 4.2 hidden states extracted with a buggy `[SOM_MARKS]` regex that dropped 71/72 marks per task; the v1 Stage 4 NPZ contained near-empty 3-line text payloads, and mode-mean cosine gaps for axis-1 and axis-2 were inflated by prompt-template differences rather than text-payload differences. After the Bug 2 fix re-extraction (Myriad 359736 cls + 359737 reddit, NPZ `hidden_states_v2_fixed.npz`), axis-1 and axis-2 cosine peaks collapse to sub-permille and move from L23 to L36 boundary-monotone. The "three quantitatively distinct axes" claim is no longer supported. The revised account below is paper-grade.
docs/checkpoints/paper_drafts/section5_mechanism.md:113:The pairs are constructed to isolate each axis. Axis-1 (text-format swap, prompt fixed) is measured by DOM↔P-text and P-prompt↔P-SoM. Axis-2 (prompt-family swap, text fixed) is measured by DOM↔P-prompt and P-text↔P-SoM. Image axis is measured by P-SoM↔SoM. All curves are computed on `stage4_multimode_b1_cls/hidden_states_v2_fixed.npz` (144 examples, 37 layers, 6 modes, strong-tier manifest filter, production `[SOM_MARKS]` formatter) and replicated cross-site on the matching reddit run.
docs/checkpoints/paper_drafts/section5_mechanism.md:117:| Axis | Pair | L17 cosine | L23 cosine | Peak L | Peak gap |
docs/checkpoints/paper_drafts/section5_mechanism.md:119:| Image | P-SoM ↔ SoM | 0.0416 | 0.0410 | L36 | 0.0416 |
docs/checkpoints/paper_drafts/section5_mechanism.md:120:| Axis-1 text-format | DOM ↔ P-text | 0.0021 | 0.0027 | L36 | 0.0047 |
docs/checkpoints/paper_drafts/section5_mechanism.md:121:| Axis-1 text-format | P-prompt ↔ P-SoM | 0.0021 | 0.0026 | L36 | 0.0048 |
docs/checkpoints/paper_drafts/section5_mechanism.md:122:| Axis-2 prompt-family | P-text ↔ P-SoM | 0.0019 | 0.0028 | L36 | 0.0088 |
docs/checkpoints/paper_drafts/section5_mechanism.md:123:| Axis-2 prompt-family | DOM ↔ P-prompt | 0.0013 | 0.0027 | L36 | 0.0068 |
docs/checkpoints/paper_drafts/section5_mechanism.md:125:Two observations replace the prior three-axis hierarchy framing:
docs/checkpoints/paper_drafts/section5_mechanism.md:127:1. **Image axis is the only well-localized geometric mechanism in the residual stream.** The image pair P-SoM↔SoM peaks at L36 with magnitude 0.042, but the early L04 peak for DOM↔Vision and P-prompt↔Vision (0.067 and 0.066) is the clean image-presence signature: when the no-image side preserves AXTree hierarchy, image-axis divergence is freshly observable at L04. When the no-image side is flat `[SOM_MARKS]`, the early peak attenuates (this is the original Mirage L04 dichotomy, and it survives the v2 re-extraction on the DOM-vs-Vision side; the SoM-side mirror requires re-examination because v1's L17 peak for P-SoM↔SoM shifted to L36 boundary in v2).
docs/checkpoints/paper_drafts/section5_mechanism.md:129:2. **Text-format and prompt-family axes are linearly readable but geometrically near-zero.** All four non-image pairs (DOM↔P-text, P-prompt↔P-SoM, P-text↔P-SoM, DOM↔P-prompt) have peak cosine gap ≤ 0.009 and rise monotonically to a boundary layer L36 rather than localizing at a mid-layer peak. The held-out leave-one-task-out AUROC remains 1.000 across all pairs and layers, which means the 24 strong-tier tasks ARE perfectly separable along these axes — but the mode-mean difference vector is small. The right reading is that text-format and prompt-family modes carry low-magnitude but high-reliability linear signatures in the residual stream rather than substantial geometric clusters.
docs/checkpoints/paper_drafts/section5_mechanism.md:131:The disjoint between **small geometric magnitude (cosine ≤ 0.01)** and **substantial causal patching displacement (overlap-to-target drop of 0.20–0.30 in §5.4 cellhprompt and Stage 2/3 cells)** is the new headline mechanism observation. A causal axis-2 patch at L11–L17 displaces target continuation by ~25% even though the geometric mean-difference at those layers is sub-permille. This argues that residual-stream cosine magnitude **underestimates** the causal influence of a feature, consistent with the standard mechinterp distinction between feature *encoded* and feature *used* \citep{wang2023interpretability}. The activation-patching evidence (§5.4) is the load-bearing claim; cosine geometry is supporting evidence about readability, not magnitude.
docs/checkpoints/paper_drafts/section5_mechanism.md:133:Phantom-SoM's drop-one hero contribution in `fig_meta_forest.png` (reddit drop-one CI [+0.95, +6.19] strict-positive) therefore cannot be attributed to "three-axis positional uniqueness" with quantitatively distinct magnitudes. The cleaner mechanism story is: Phantom-SoM is one of four modes occupying the no-image-flat-marks half of the phantom routing space, all of which produce small geometric separation from each other; the behaviorally distinct success-task pool (Jaccard 0.29–0.49 against other arms) is what gives drop-one its complementarity, and patching displacement at L11–L17 shows the difference matters causally for token continuation. The bridge from patching displacement to behavioral SR remains open.
docs/checkpoints/paper_drafts/section5_mechanism.md:135:A held-out leave-one-task-out AUROC of 1.000 across small-magnitude pairs is itself a paper-grade observation. It says the modes are reliably linearly separable at any chosen layer with very small mean-difference vectors, which is a stronger claim about the residual stream than the original "distinct mid-layer peaks" framing. The information capacity of the residual stream to represent observation-mode identity is high; the *magnitude* of the representation is mostly image-driven. This reframing follows the linear-readability framework of \citep{wu2026toolcalling} without the cosine-magnitude overclaim.
docs/checkpoints/paper_drafts/section5_mechanism.md:137:The output-amplification observation (logit lens, Exp 3) needs re-running on the v2 NPZ before its quantitative claims can be reported. The v1 logit lens reported peak KL 0.162 at L23 for the axis-2 pair P-text vs P-SoM, but the v1 input hidden states were the buggy 3-line-text version. The qualitative direction (lm_head amplifies residual-stream geometry into output KL) likely survives, but the absolute KL magnitudes will change; we report the v2 lm_head amplification numbers in a follow-up release.
docs/checkpoints/paper_drafts/section5_mechanism.md:139:Two corollaries follow. First, the KL trajectory drops to approximately zero at L36 even though L23 KL is substantial. The mean hidden state at the final layer collapses to the shared JSON action-header tokens that every mode emits, so mode-distinct output signal is concentrated in the L23-L25 decoding window rather than at the final embedding. Second, this output-amplification observation is **mechanistic, not a deployment-time classifier claim**: the lm_head acts as an axis-agnostic ratio-preserving projection that scales residual-stream geometry into output-space KL — the L23-L25 KL magnitude is a property of the mean hidden state, not a per-task discriminator. Whether the L23-L25 hidden representation can be used as a held-out mode classifier — with per-task AUROC, random-direction baseline, and competitive comparison to surface-token classifiers — is open work. Routing exploitation, deferred to paper 2, will need to make this case explicitly rather than inheriting it from §5.7.
docs/checkpoints/paper_drafts/section5_mechanism.md:143:The main limit is the Method 4.4 ceiling. The cosine-gap and patching evidence point to L11-L17 as the readable and causally active fusion region, while the best fixed mean-difference steering cell is late, L33 with $\alpha=10$, and has H-mean 0.33 because completeness and selectivity trade off. This supports a mechanism interpretation but not a strong deployment-time steering claim.
docs/checkpoints/paper_drafts/section5_mechanism.md:145:The second limit is layer precision. Classifieds H1 peaks often hit L36, while reddit reveals cleaner L17 peaks. The robust claim is therefore an effect-direction claim: AXTree hierarchy preserves early image-axis separation, and flat element-list formats delay that separation into mid/late computation. We should not claim that every site or task has an identical peak layer.
docs/checkpoints/paper_drafts/section5_mechanism.md:147:Literature positioning should stay modest. Section 5 applies the linear-readable, steerable, and mid/late-layer circuit framework to multimodal web-agent observation modes \citep{wu2026toolcalling,kaduri2024whatsintheimage,khorasani2026hdmi,fayyaz2026steermoe}. It should not claim novelty as the first such circuit or the first use of marked text. The contribution is controlled scientific characterization of the phantom boundary.
docs/checkpoints/paper_drafts/section5_mechanism.md:149:Finally, AXTree hierarchy is the unique defeating format in the aggregate, but the reason hierarchy defeats the shortcut remains open. The plan records one attribution-pending hypothesis: hierarchy or indentation tokens may redirect cross-modal attention before the flat-list shortcut activates. That should be treated as a supplement question, not as a Section 5 finding.
docs/checkpoints/paper_drafts/section5_mechanism.md:155:Behavioral content to relocate from current `section5_mechanism_reddit.md`: lines 17-75 should move to Section 4 or a new behavioral-routing subsection. Specifically, lines 17-23 are reddit substrate framing; lines 25-35 are Axis 1 text-payload behavior; lines 37-47 are Axis 2 prompt behavior; lines 49-59 are Axis 3 image behavior; lines 61-67 are compound P-SoM versus DOM behavior; lines 69-75 are scope/noise limitations. Lines 1-15 are method material that was retained conceptually but must use the new L0-L36 layer convention. Line 77 should be deleted or replaced because routing implementation is now paper-2, not paper-1 Section 6.
docs/checkpoints/paper_drafts/section5_mechanism.md:157:Stage 3 numbers verified 2026-05-12 from full per-task paired-test computation on `patching_continuation_results.json` (each cell, 24 tasks × 36 layers). H-d-cls best L18 Δ=-0.352, H-d-red best L14 Δ=-0.338, H-t-cls best L12 -0.270, H-t-red best L15 -0.330, H-p-cls best L13 -0.273, H-p-red best L14 -0.322. All 6 cells' best layer lands in L12-L18 mid-layer window, Δ range [-0.27, -0.35]. The L17-only column previously cited in plan §5.2 reads -0.309/-0.255/-0.223 (cls) and -0.255/-0.236/-0.191 (reddit); plan §5.2 has been updated to record best-layer Δ instead of L17-only Δ.
docs/checkpoints/paper_drafts/section5_mechanism.md:165:1. ✅ §5.1 ¶4 — "flat element-list trigger" → refined to "integer-marker + markup-sigil conjunction" with W6 exploratory caveat
docs/checkpoints/paper_drafts/section5_mechanism.md:172:- **§4 P-text adjusted SR inconsistency**: §4 table line 37 says 11.90, prose line 106 says 12.38, hero_claim_bootstrap_ci.md says 12.38. Need to canonicalize one number from episode-level adjusted-success and update every occurrence. (1h)
docs/checkpoints/paper_drafts/section5_mechanism.md:173:- **plan.md:125-135 stale "L17 planning site"**: plan still asserts L17 singular planning site, while new evidence shows cosine peak L23 + patching peak L11-L17 + steering best L33; replace with "patch-sensitive continuation window L11-L17 under final-token replacement patching". (1-2h)
docs/checkpoints/paper_drafts/section5_mechanism.md:176:- **Behavioral causal bridge gap**: patching displaces 50-token continuation, not SR / drop-one. No experiment currently bridges mid-layer patching effect to task-success outcome. Open work.
docs/checkpoints/paper_drafts/section5_mechanism.md:179:**Codex verdict**: 0.10-0.20 NeurIPS/ICML/ACL main accept probability; would reject @ reviewer-3 4/10. **Workshop / borderline mid-tier conference today**. Codex agrees the drop-one oracle CI + cross-site asymmetry + axis-2 cosine signal breadth are strong; §5 mechanism prose was over-claiming relative to evidence.

2026-05-12T23:24:06.704993Z ERROR codex_core::tools::router: error=write_stdin failed: stdin is closed for this session; rerun exec_command with tty=true to keep stdin open
codex
### 1. Verdict (one sentence)

当前稿件有一个可投稿的 empirical phenomenon：Phantom-SoM 的 drop-one oracle value 是真信号；但 mechanism section 仍被 v2 数据重写、旧叙事残留和 causal bridge 缺口拖住，今天投 top-tier 会被强 reviewer 打穿。

### 2. Strong claims (don't break under attack)

1. Phantom-SoM 的 single-mode superiority 不稳，但 drop-one oracle 是正的。Paper 自己写得正确：reddit head-to-head CI crosses zero，而 drop-one CI strict-positive（`section1_intro.md:7`）；evidence 给出 `phantom_som +3.33pp [0.95, 6.19], P(Δ>0)=0.998`（`hero_claim_bootstrap_ci.md:31-42`）。

2. full SoM 在 classifieds 上明显强于 Phantom-SoM，这个 sanity check 站得住。Paper quote: “SoM 21.37 vs Phantom-SoM 14.53” (`section4_empirical_findings.md:40`)；bootstrap evidence 是 `P-SoM vs SoM -6.84pp [-12.39, -1.28]` (`hero_claim_bootstrap_ci.md:54-59`)。

3. v2 后的 residual-stream readability/magnitude reframe 是诚实且有证据的。Paper quote: “three quantitatively distinct axes claim is no longer supported” (`section5_mechanism.md:111`)；evidence 明确说 v1 dropped 71/72 marks，v2 后 axis-1/axis-2 collapse，而 AUROC lototask 仍 1.000 (`method42_v1_vs_v2_comparison.md:7-11`, `:59-63`)。

### 3. Weak claims (would tank under attack)

1. **Claim** — “Across eight image-presence contrasts, the no-image side's text format predicts the peak layer with zero overlap.” (`section5_mechanism.md:87`)  
   **Attack** — 这已经被你自己的 v2 audit 反证。`method42_v1_vs_v2_comparison.md:77-80` 明确说 `P-text/P-SoM ↔ Vision` 从 L36 shift 到 L04，breaks dichotomy；cls v2 JSON 也显示 `phantom_text_vs_vision` 和 `phantom_som_vs_vision` peak L04 (`method42_metrics_v2.json:2067-2101`)。  
   **Defuse** — 删除 “zero overlap” 和 Method 4.2 dichotomy claim；只保留 format-variation H1 as exploratory，并重跑 H1/format variation on v2 fixed extraction.  
   **Effort** — 2-4 hours for prose; 0.5-1 day if rerun/plot.

2. **Claim** — “A causal axis-2 patch at L11–L17 displaces target continuation by ~25% … activation-patching evidence is the load-bearing claim.” (`section5_mechanism.md:131`)  
   **Attack** — 当前只是 final-token replacement 对 50-token continuation 的 overlap metric，不是 task success, routing, or natural mediation。Exp 5 自己承认 “doesn't directly translate to SR / drop-one” and needs CIs (`exp5_axis2_causal_patching.md:97-101`)，而 paper table 还说 task-shuffled control in flight (`section5_mechanism.md:20`)。  
   **Defuse** — land task-shuffled same-norm real-activation control, bootstrap per-layer paired CIs, and one behavioral bridge: patched continuation changes parsed action / success-relevant action on archived steps.  
   **Effort** — 1-3 days if infrastructure exists; 1 week for clean action-level bridge.

3. **Claim** — “text representation shapes how the agent explores; prompt wording tunes when it commits.” (`section1_intro.md:11`, `section4_empirical_findings.md:71-86`)  
   **Attack** — 这是漂亮叙事，但 top-tier reviewer 会问统计和 provenance。核心 verified subset 只有 N=48 (`section4_empirical_findings.md:69`)，§4.5 大量引用 `Outcome 0c / Macro 1b / Micro 2f` (`section4_empirical_findings.md:104-120`) 在 read scope 内没有可追溯 source file/line。  
   **Defuse** — 给每个 behavioral metric 一个 table source, denominator, CI/randomization test, and trace-audit protocol；把 N=48 subset 与 full N=210 的关系讲清楚。  
   **Effort** — 1-2 days.

4. **Claim** — “A held-out leave-one-task-out AUROC of 1.000 across small-magnitude pairs is itself a paper-grade observation.” (`section5_mechanism.md:135`)  
   **Attack** — mode identity may be trivially readable from prompt/template tokens at every layer; AUROC=1.000 does not by itself establish phantom mechanism. Especially with 24 curated strong-tier tasks per site (`method42_metrics_v2.json:13-19`).  
   **Defuse** — add surface-token classifier baseline, prompt-stripped/control-position ablation, random-direction baseline, and train on one site/test on the other.  
   **Effort** — 2-4 days.

### 4. Honest gaps (not weak, just missing)

- `section1_intro.md:13` promises Section 6 and Section 7, but `paper_drafts` contains no `section6*.md` or `section7*.md`. That is a submission-blocking draft gap, not a scientific flaw.

- Cross-family / capacity generalization is absent. Paper bounds this correctly (`section8_limitations.md:1-3`), and plan says Phi-3.5-Vision / Qwen2-VL-7B are deferred (`plan.md:310-311`). Fine, but top-tier scope must stay Qwen-family/VWA only.

- Draft hygiene is bad: `section5_mechanism.md:151-179` contains “NOTE FOR HUMAN” and prior audit follow-ups; `section4_limitations_disclosure.md:134` has casual wording. This cannot appear in a paper draft.

- Bibliography still has `NEEDS_VERIFY` entries (`paper.bib:584`, `paper.bib:605`). Not fatal scientifically, but fatal for submission polish.

### 5. Distance to top-tier

- Current acceptance tier: workshop / borderline mid-tier conference, not top-tier main yet.

- Blocker 1: §5 contains claims contradicted by v2, especially image-axis peak-layer dichotomy. Blocks mechanism novelty.  
  Unblock: rewrite §5.5 and plan stale sections around v2 fixed metrics; remove “zero overlap.” Effort: 0.5 day.

- Blocker 2: causal patching does not yet bridge to task behavior. Blocks “mechanism explains hero drop-one.”  
  Unblock: content-shuffled control + action-level patched-step evaluation. Effort: 2-5 days.

- Blocker 3: behavioral two-knob account lacks traceable stats in draft. Blocks §4 mechanism narrative.  
  Unblock: add source tables, CIs, denominators for Outcome/Macro/Micro metrics. Effort: 1-2 days.

- Submission-today probability: top-tier main 0.10-0.20. I would expect rejection from at least 3/4 reviewers unless framed as “interesting empirical anomaly with preliminary mechanism,” not as a solved mechanistic account.

### 6. One thing to fix tonight (1-3 hour leverage)

Fix `docs/checkpoints/paper_drafts/section5_mechanism.md`: delete the `NOTE FOR HUMAN` block, remove the §5.5 “zero overlap” dichotomy claim, and replace it with the v2-safe statement from `method42_v1_vs_v2_comparison.md:77-80`: image-axis geometry remains real, but the clean AXTree-vs-flat peak-layer split was partially a v1 artifact; H1 format attribution is exploratory pending held-out falsifiers.
tokens used
151,241
### 1. Verdict (one sentence)

当前稿件有一个可投稿的 empirical phenomenon：Phantom-SoM 的 drop-one oracle value 是真信号；但 mechanism section 仍被 v2 数据重写、旧叙事残留和 causal bridge 缺口拖住，今天投 top-tier 会被强 reviewer 打穿。

### 2. Strong claims (don't break under attack)

1. Phantom-SoM 的 single-mode superiority 不稳，但 drop-one oracle 是正的。Paper 自己写得正确：reddit head-to-head CI crosses zero，而 drop-one CI strict-positive（`section1_intro.md:7`）；evidence 给出 `phantom_som +3.33pp [0.95, 6.19], P(Δ>0)=0.998`（`hero_claim_bootstrap_ci.md:31-42`）。

2. full SoM 在 classifieds 上明显强于 Phantom-SoM，这个 sanity check 站得住。Paper quote: “SoM 21.37 vs Phantom-SoM 14.53” (`section4_empirical_findings.md:40`)；bootstrap evidence 是 `P-SoM vs SoM -6.84pp [-12.39, -1.28]` (`hero_claim_bootstrap_ci.md:54-59`)。

3. v2 后的 residual-stream readability/magnitude reframe 是诚实且有证据的。Paper quote: “three quantitatively distinct axes claim is no longer supported” (`section5_mechanism.md:111`)；evidence 明确说 v1 dropped 71/72 marks，v2 后 axis-1/axis-2 collapse，而 AUROC lototask 仍 1.000 (`method42_v1_vs_v2_comparison.md:7-11`, `:59-63`)。

### 3. Weak claims (would tank under attack)

1. **Claim** — “Across eight image-presence contrasts, the no-image side's text format predicts the peak layer with zero overlap.” (`section5_mechanism.md:87`)  
   **Attack** — 这已经被你自己的 v2 audit 反证。`method42_v1_vs_v2_comparison.md:77-80` 明确说 `P-text/P-SoM ↔ Vision` 从 L36 shift 到 L04，breaks dichotomy；cls v2 JSON 也显示 `phantom_text_vs_vision` 和 `phantom_som_vs_vision` peak L04 (`method42_metrics_v2.json:2067-2101`)。  
   **Defuse** — 删除 “zero overlap” 和 Method 4.2 dichotomy claim；只保留 format-variation H1 as exploratory，并重跑 H1/format variation on v2 fixed extraction.  
   **Effort** — 2-4 hours for prose; 0.5-1 day if rerun/plot.

2. **Claim** — “A causal axis-2 patch at L11–L17 displaces target continuation by ~25% … activation-patching evidence is the load-bearing claim.” (`section5_mechanism.md:131`)  
   **Attack** — 当前只是 final-token replacement 对 50-token continuation 的 overlap metric，不是 task success, routing, or natural mediation。Exp 5 自己承认 “doesn't directly translate to SR / drop-one” and needs CIs (`exp5_axis2_causal_patching.md:97-101`)，而 paper table 还说 task-shuffled control in flight (`section5_mechanism.md:20`)。  
   **Defuse** — land task-shuffled same-norm real-activation control, bootstrap per-layer paired CIs, and one behavioral bridge: patched continuation changes parsed action / success-relevant action on archived steps.  
   **Effort** — 1-3 days if infrastructure exists; 1 week for clean action-level bridge.

3. **Claim** — “text representation shapes how the agent explores; prompt wording tunes when it commits.” (`section1_intro.md:11`, `section4_empirical_findings.md:71-86`)  
   **Attack** — 这是漂亮叙事，但 top-tier reviewer 会问统计和 provenance。核心 verified subset 只有 N=48 (`section4_empirical_findings.md:69`)，§4.5 大量引用 `Outcome 0c / Macro 1b / Micro 2f` (`section4_empirical_findings.md:104-120`) 在 read scope 内没有可追溯 source file/line。  
   **Defuse** — 给每个 behavioral metric 一个 table source, denominator, CI/randomization test, and trace-audit protocol；把 N=48 subset 与 full N=210 的关系讲清楚。  
   **Effort** — 1-2 days.

4. **Claim** — “A held-out leave-one-task-out AUROC of 1.000 across small-magnitude pairs is itself a paper-grade observation.” (`section5_mechanism.md:135`)  
   **Attack** — mode identity may be trivially readable from prompt/template tokens at every layer; AUROC=1.000 does not by itself establish phantom mechanism. Especially with 24 curated strong-tier tasks per site (`method42_metrics_v2.json:13-19`).  
   **Defuse** — add surface-token classifier baseline, prompt-stripped/control-position ablation, random-direction baseline, and train on one site/test on the other.  
   **Effort** — 2-4 days.

### 4. Honest gaps (not weak, just missing)

- `section1_intro.md:13` promises Section 6 and Section 7, but `paper_drafts` contains no `section6*.md` or `section7*.md`. That is a submission-blocking draft gap, not a scientific flaw.

- Cross-family / capacity generalization is absent. Paper bounds this correctly (`section8_limitations.md:1-3`), and plan says Phi-3.5-Vision / Qwen2-VL-7B are deferred (`plan.md:310-311`). Fine, but top-tier scope must stay Qwen-family/VWA only.

- Draft hygiene is bad: `section5_mechanism.md:151-179` contains “NOTE FOR HUMAN” and prior audit follow-ups; `section4_limitations_disclosure.md:134` has casual wording. This cannot appear in a paper draft.

- Bibliography still has `NEEDS_VERIFY` entries (`paper.bib:584`, `paper.bib:605`). Not fatal scientifically, but fatal for submission polish.

### 5. Distance to top-tier

- Current acceptance tier: workshop / borderline mid-tier conference, not top-tier main yet.

- Blocker 1: §5 contains claims contradicted by v2, especially image-axis peak-layer dichotomy. Blocks mechanism novelty.  
  Unblock: rewrite §5.5 and plan stale sections around v2 fixed metrics; remove “zero overlap.” Effort: 0.5 day.

- Blocker 2: causal patching does not yet bridge to task behavior. Blocks “mechanism explains hero drop-one.”  
  Unblock: content-shuffled control + action-level patched-step evaluation. Effort: 2-5 days.

- Blocker 3: behavioral two-knob account lacks traceable stats in draft. Blocks §4 mechanism narrative.  
  Unblock: add source tables, CIs, denominators for Outcome/Macro/Micro metrics. Effort: 1-2 days.

- Submission-today probability: top-tier main 0.10-0.20. I would expect rejection from at least 3/4 reviewers unless framed as “interesting empirical anomaly with preliminary mechanism,” not as a solved mechanistic account.

### 6. One thing to fix tonight (1-3 hour leverage)

Fix `docs/checkpoints/paper_drafts/section5_mechanism.md`: delete the `NOTE FOR HUMAN` block, remove the §5.5 “zero overlap” dichotomy claim, and replace it with the v2-safe statement from `method42_v1_vs_v2_comparison.md:77-80`: image-axis geometry remains real, but the clean AXTree-vs-flat peak-layer split was partially a v1 artifact; H1 format attribution is exploratory pending held-out falsifiers.
