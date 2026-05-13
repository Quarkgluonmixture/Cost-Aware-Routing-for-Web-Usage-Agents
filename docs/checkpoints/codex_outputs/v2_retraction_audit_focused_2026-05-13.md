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
session id: 019e204c-1782-7b43-affe-cbb5ad37d92e
--------
user
# Codex focused audit — P79 v2 retraction (3 specific claims)

Independent hostile reviewer mode. **Do NOT** read prior codex outputs or Claude reviews. Read **only** the files listed below — don't go exploring.

## Context

P79 paper-1, Qwen3-VL-4B B1, mechanism §5. v2 NPZ migration just landed (Bug 1 tier filter + Bug 2 SOM_MARKS regex + Bug 5 model revision). Author rewrote `docs/checkpoints/mechanism/plan.md` with v2 numbers. User is **skeptical** of the new framing and wants a brutal cross-AI sanity check.

## Read these 4 files only

1. `docs/checkpoints/mechanism/plan.md` §0 + §1.2 + §1.3 (the 3 most-changed sections)
2. `docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md`
3. `scripts/analysis/stage4_pca_cosine_gap.py` (lines 40-180 — cosine_gap function + main loop)
4. `docs/checkpoints/mechanism/results/axis2_logit_lens_v2.md`

## Attack these 3 claims (don't explore beyond)

### Claim A: "Cosine-causal disjoint" hero (plan.md §1.2)

Author argues:
- cosine gap 0.5-1% (residual stream geometric)
- KL ~0.05-0.09 (logit lens output)
- patching Δoverlap 20-30% (causal behavior)
- → "geometry underestimates causal by orders of magnitude"

Attack: are these comparable? Cosine measures mean-distance, KL measures distribution divergence, Δoverlap measures behavior change after intervention. Calling this a "disjoint" — is this defensible or category error?

### Claim B: V1 magnitude → v2 magnitude collapse is "artifact correction"

`method42_v1_vs_v2_comparison.md` shows axis-1 text-format cosine collapsed -81% (0.025 → 0.005). Author calls v1 "buggy artifact" and v2 "correct".

Attack: alternative interpretation — v2's properly-included `[SOM_MARKS]` HOMOGENIZES text across `som / phantom_som / phantom_text / phantom_prompt` modes, all carrying same `[id=N] {label}` block. V1 may have **accidentally exposed real signal** by selectively dropping marks differently per mode. How would author distinguish "v2 corrects bug" from "v2 masks real signal"?

### Claim C: Logit lens "amplification 8-44×"

`axis2_logit_lens_v2.md` line 3: "Apply Qwen3-VL-4B's final_norm + lm_head to per-layer **per-mode mean** hidden states." Then computes KL between modes.

Attack: per-mode means are averaged representations that don't correspond to any actual forward pass. Reviewer-3 will say "decoded an average that doesn't exist; KL between two averaged distributions is not a meaningful 'amplification' of the underlying signal — it could be averaging artifact". Is there per-task KL computation (decode each task's actual hidden state through lm_head then average task KLs) instead?

## Deliverable

Output **only**:

```markdown
## Verdict line
[one sentence: do the 3 claims survive hostile reading?]

## Claim A: Cosine-causal disjoint
Strength: [strong | weak | broken]
Attack: [specific quote + what's wrong]
Defuse: [what evidence/reframing would save it]
Effort: [hours]

## Claim B: v1→v2 magnitude collapse
Strength: [strong | weak | broken]
Attack: ...
Defuse: ...
Effort: ...

## Claim C: Logit lens amplification
Strength: [strong | weak | broken]
Attack: ...
Defuse: ...
Effort: ...

## Single highest-leverage move tonight
[1-2 sentences]
```

Brutal but fair, ~500-800 words total. **Do NOT** explore other files. **Do NOT** re-read full plan.md (just the 3 sections cited). Output goes to stdout — the wrapper captures it.

End your response with the literal token: `=== AUDIT COMPLETE ===`

exec
/bin/bash -lc "awk '
/"'^## 0'"\\.|"'^# 0'"\\.|"'^### 0'"\\./ {p=1}
/"'^## 1'"\\.|"'^# 1'"\\.|"'^### 1'"\\./ {if (p==1) p=0}
p
/"'^## 1'"\\.2|"'^# 1'"\\.2|"'^### 1'"\\.2/ {p=2}
/"'^## 1'"\\.3|"'^# 1'"\\.3|"'^### 1'"\\.3/ {if (p==2) p=0; p=3}
/"'^## 1'"\\.4|"'^# 1'"\\.4|"'^### 1'"\\.4|"'^## 2'"\\.|"'^# 2'"\\.|"'^### 2'"\\./ {if (p==2 || p==3) p=0}
p==2 || p==3
' docs/checkpoints/mechanism/plan.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
## 0. v2 retraction summary (2026-05-13)

V1 Stage 4 NPZ regex `^\[\d+\]\s+\w+` extracted only 38 chars / 3 lines per task, dropping 71/72 SOM_MARKS. Affected: Method 4.2 cosine geometry, Exp 1 axis-2 layer profile, Exp 3 logit lens, per-task fragility. V2 NPZ uses production `_extract_text_marks` (full 72-line `[id=N] {label}` payload). Re-extraction Myriad 359736 (cls) + 359737 (reddit) landed 2026-05-12 late, v2 metrics 2026-05-13 02:52.

**What changed**:
- ✗ V1 "three-axis hierarchy 4:3:1 magnitude ratio" → INVALIDATED. V2: image dominates ~5-10×; axis-1 and axis-2 both noise-level (cosine ~0.005-0.009); axis-1 magnitude is now ≤ axis-2 (reversed ranking).
- ✗ V1 "AXTree → L04 vs flat → L17-L36" no-image-side dichotomy → REORGANIZED. V2: dichotomy is image-side-based (Vision→L04, SoM→L36), not text-format-based.
- ✓ AUROC linear-readability 1.000 cross-site → preserved.
- ✓ Image-axis cosine peaks (~0.04-0.07) → preserved.
- ✓ Stage 2/3 patching (uses archive_subset, not Stage 4 NPZ) → unchanged.
- ✓ Method 4.4 steering (separate pipeline) → unchanged.
- ✓ Exp 5 axis-2 causal patching → unchanged.

**New hero claim** (replaces v1 three-axis hierarchy): **cosine-causal disjoint** — geometric magnitude is sub-permille (0.005-0.009) but causal patching displaces overlap 20-30% AND lm_head amplifies cosine→KL by 8-25×. Residual-stream geometry underestimates causal influence by orders of magnitude; cosine gap measures effect SIZE while AUROC measures CLASSIFICATION RELIABILITY and they dissociate. Paper-grade novel + reviewer-defensible.

Provenance: `docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md` (canonical v1↔v2 diff). V2 NPZ at `results/mechanistic/stage4_multimode_b1_{cls,reddit}/hidden_states_v2_fixed.npz`.

### 1.2 Cosine-causal disjoint (Method 4.2 v2 + Stage 2/3 + Exp 3 logit lens)


V2 NPZ-corrected geometry (paper-grade canonical, 2026-05-13):
V2 NPZ-corrected geometry (paper-grade canonical, 2026-05-13):


| Axis | Pair | L17 cos gap | Peak L / gap | Notes |
| Axis | Pair | L17 cos gap | Peak L / gap | Notes |
|---|---|---:|---:|---|
|---|---|---:|---:|---|
| **Image-axis (Vision)** | DOM ↔ Vision | 0.057 | L04 0.067 | early visual encoder |
| **Image-axis (Vision)** | DOM ↔ Vision | 0.057 | L04 0.067 | early visual encoder |
| **Image-axis (SoM)** | P-SoM ↔ SoM | 0.003 | L36 0.042 | late integration |
| **Image-axis (SoM)** | P-SoM ↔ SoM | 0.003 | L36 0.042 | late integration |
| Axis-1 text-format | DOM ↔ P-text | 0.002 | L36 0.005 | sub-permille, monotone-to-boundary |
| Axis-1 text-format | DOM ↔ P-text | 0.002 | L36 0.005 | sub-permille, monotone-to-boundary |
| Axis-1 text-format | P-prompt ↔ P-SoM | 0.002 | L36 0.005 | sub-permille |
| Axis-1 text-format | P-prompt ↔ P-SoM | 0.002 | L36 0.005 | sub-permille |
| Axis-2 prompt-family flat | P-text ↔ P-SoM | 0.002 | L36 0.009 | sub-permille |
| Axis-2 prompt-family flat | P-text ↔ P-SoM | 0.002 | L36 0.009 | sub-permille |
| Axis-2 prompt-family hier | DOM ↔ P-prompt | 0.001 | L36 0.007 | sub-permille |
| Axis-2 prompt-family hier | DOM ↔ P-prompt | 0.001 | L36 0.007 | sub-permille |


**Geometric magnitudes** v2: image 0.04-0.07 / text-format 0.005 / prompt-family 0.009 → image dominates **5-10×**, axis-1 ≤ axis-2 (sub-permille).
**Geometric magnitudes** v2: image 0.04-0.07 / text-format 0.005 / prompt-family 0.009 → image dominates **5-10×**, axis-1 ≤ axis-2 (sub-permille).


**Causal patching magnitudes** (Stage 2/3 mid-layer L11-L17 window, 6/6 cells cross-site):
**Causal patching magnitudes** (Stage 2/3 mid-layer L11-L17 window, 6/6 cells cross-site):
- Δoverlap-to-target: -0.27 to -0.35 (cls + reddit, all SoM→{no-image-arm} forward cells)
- Δoverlap-to-target: -0.27 to -0.35 (cls + reddit, all SoM→{no-image-arm} forward cells)
- Random injection control (E + Er): null effect
- Random injection control (E + Er): null effect
- → **causal patching effect magnitude 20-30%** vs **geometric magnitude 0.5-1%**
- → **causal patching effect magnitude 20-30%** vs **geometric magnitude 0.5-1%**


**Logit lens amplification** (Exp 3 v2, Qwen3-VL-4B `norm + lm_head` on per-layer means):
**Logit lens amplification** (Exp 3 v2, Qwen3-VL-4B `norm + lm_head` on per-layer means):
- Axis-2 P-text↔P-SoM cosine 0.002 at L17 → KL **0.088 at L25** (cls), 0.057 at L25 (reddit)
- Axis-2 P-text↔P-SoM cosine 0.002 at L17 → KL **0.088 at L25** (cls), 0.057 at L25 (reddit)
- Cosine→KL amplification: **8-44× depending on pair**, peak amplification at L21-L25 decoding window
- Cosine→KL amplification: **8-44× depending on pair**, peak amplification at L21-L25 decoding window
- KL collapses to ~0 at L36 (mean hidden collapses to common JSON-header prefix) → mode-distinct signal lives in **L23-L25 window**, not final embedding
- KL collapses to ~0 at L36 (mean hidden collapses to common JSON-header prefix) → mode-distinct signal lives in **L23-L25 window**, not final embedding


**Interpretive disjoint**: residual-stream cosine geometry severely underestimates causal influence. Three converging numbers:
**Interpretive disjoint**: residual-stream cosine geometry severely underestimates causal influence. Three converging numbers:
- Cosine gap 0.5-1% (geometric magnitude small)
- Cosine gap 0.5-1% (geometric magnitude small)
- Δoverlap 20-30% (causal effect large)
- Δoverlap 20-30% (causal effect large)
- KL ~0.05-0.09 (output divergence intermediate, amplified 8-44× from cosine)
- KL ~0.05-0.09 (output divergence intermediate, amplified 8-44× from cosine)


This is the new paper §5 hero claim. AUROC linear-readability 1.000 holds throughout — modes ARE distinguishable in residual stream; the magnitude of the mode-mean difference is just much smaller than v1 claimed.
This is the new paper §5 hero claim. AUROC linear-readability 1.000 holds throughout — modes ARE distinguishable in residual stream; the magnitude of the mode-mean difference is just much smaller than v1 claimed.


### 1.3 Image-axis peak-layer signature (v2 — cross-site DIVERGENT, needs further work)
### 1.3 Image-axis peak-layer signature (v2 — cross-site DIVERGENT, needs further work)


V2 NPZ data shows the dichotomy **does NOT replicate cleanly cross-site**. This is a v2-revealed paper-grade nuance not present in v1:
V2 NPZ data shows the dichotomy **does NOT replicate cleanly cross-site**. This is a v2-revealed paper-grade nuance not present in v1:


**Cls v2**: clean image-side-based dichotomy
**Cls v2**: clean image-side-based dichotomy


| Image side | Peak layer | All 4 pairs cos gap |
| Image side | Peak layer | All 4 pairs cos gap |
|---|---|---:|
|---|---|---:|
| Vision (naked) | **L04** | 0.060-0.067 |
| Vision (naked) | **L04** | 0.060-0.067 |
| SoM (annotated) | **L36** | 0.042-0.050 |
| SoM (annotated) | **L36** | 0.042-0.050 |


**Reddit v2**: peak layer mostly L04 across the board (7/8 pairs), only P-text↔SoM at L17.
**Reddit v2**: peak layer mostly L04 across the board (7/8 pairs), only P-text↔SoM at L17.


| Image side | Peak layer | Pairs |
| Image side | Peak layer | Pairs |
|---|---|---|
|---|---|---|
| Vision (naked) | L04 (all 4) | DOM/P-text/P-prompt/P-SoM ↔ Vision |
| Vision (naked) | L04 (all 4) | DOM/P-text/P-prompt/P-SoM ↔ Vision |
| SoM (annotated) | L04 (3/4) | DOM↔SoM 0.046, P-prompt↔SoM 0.043, P-SoM↔SoM 0.039 |
| SoM (annotated) | L04 (3/4) | DOM↔SoM 0.046, P-prompt↔SoM 0.043, P-SoM↔SoM 0.039 |
| SoM (annotated) | **L17 (1/4)** | P-text↔SoM 0.043 |
| SoM (annotated) | **L17 (1/4)** | P-text↔SoM 0.043 |


**Cross-site disagreement is real**: cls SoM-image pairs all defer to L36 late integration; reddit SoM-image pairs mostly emerge at L04 with one exception. Possible explanations:
**Cross-site disagreement is real**: cls SoM-image pairs all defer to L36 late integration; reddit SoM-image pairs mostly emerge at L04 with one exception. Possible explanations:
1. Reddit's smaller/sparser SoM overlay produces clearer early visual discrepancy regardless of text-payload format
1. Reddit's smaller/sparser SoM overlay produces clearer early visual discrepancy regardless of text-payload format
2. Cls listing-heavy DOM trees push annotated SoM cosine peak past mid-layers; reddit comment-thread DOM doesn't
2. Cls listing-heavy DOM trees push annotated SoM cosine peak past mid-layers; reddit comment-thread DOM doesn't
3. V2 NPZ sampling variance (288 ex each is borderline for layer-peak precision at 0.04 magnitude)
3. V2 NPZ sampling variance (288 ex each is borderline for layer-peak precision at 0.04 magnitude)


**v1 framing retraction**: v1 said the dichotomy was no-image-side-text-based (AXTree → L04 vs `[SOM_MARKS]` → L17-L36) and cross-site stable. V2 data on cls reorganizes to image-side-based; v2 data on reddit collapses to L04 dominant. Neither v1 nor a single v2 reorganized framing replicates cross-site.
**v1 framing retraction**: v1 said the dichotomy was no-image-side-text-based (AXTree → L04 vs `[SOM_MARKS]` → L17-L36) and cross-site stable. V2 data on cls reorganizes to image-side-based; v2 data on reddit collapses to L04 dominant. Neither v1 nor a single v2 reorganized framing replicates cross-site.


**Paper §5 prose implication**: do NOT make a "peak-layer dichotomy is universal mechanism" claim. Honest framing: image-axis cosine peak structure varies by site (cls late-integration on SoM, reddit early-integration), with **AUROC linear-readability 1.000 preserved cross-site at all layers**. The "Mirage signature" claim must be reframed around AUROC + cosine magnitude rank-order (image > text-format ≈ prompt-family), not peak-layer location.
**Paper §5 prose implication**: do NOT make a "peak-layer dichotomy is universal mechanism" claim. Honest framing: image-axis cosine peak structure varies by site (cls late-integration on SoM, reddit early-integration), with **AUROC linear-readability 1.000 preserved cross-site at all layers**. The "Mirage signature" claim must be reframed around AUROC + cosine magnitude rank-order (image > text-format ≈ prompt-family), not peak-layer location.


### 1.4 H1 test confirms broader: flat-list (not just indexed) triggers shortcut (2026-05-12)

exec
/bin/bash -lc "sed -n '40,180p' scripts/analysis/stage4_pca_cosine_gap.py" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
def cosine_gap(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))


def pair_key(a: str, b: str) -> str:
    """Canonical pair key using MODES index order (matches itertools.combinations output)."""
    i, j = MODES.index(a), MODES.index(b)
    return f"{MODES[min(i, j)]}_vs_{MODES[max(i, j)]}"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--output-fig", type=Path, default=DEFAULT_OUT_FIG)
    args = parser.parse_args()
    NPZ = args.input
    OUT_JSON = args.output_json
    OUT_MD = args.output_md
    OUT_FIG = args.output_fig

    d = np.load(NPZ, allow_pickle=True)
    H = d["hidden_states"]
    mode_labels = d["mode_labels_str"]
    task_ids = d["task_ids"] if "task_ids" in d.files else None
    n_layers = H.shape[1]
    print(f"[stage4] loaded {H.shape[0]} examples × {n_layers} layers × {H.shape[2]} dim")

    states = {m: H[mode_labels == m] for m in MODES}
    means = {m: states[m].mean(axis=0) for m in MODES}  # each (37, 2560)
    print(f"[stage4] per-mode counts: " + ", ".join(f"{m}={len(states[m])}" for m in MODES))

    # Per-mode task_id mapping for leave-one-task-out (Bug 3 fix, codex
    # methodology audit 2026-05-12: previous AUROC fit direction on the
    # same examples used to evaluate → inflated, not held-out decodability).
    mode_task_ids = {m: task_ids[mode_labels == m] if task_ids is not None else None
                     for m in MODES}

    pairs = list(combinations(MODES, 2))
    cos_gap = np.zeros((len(pairs), n_layers))
    auroc_in_sample = np.zeros((len(pairs), n_layers))
    auroc_lototask = np.zeros((len(pairs), n_layers))  # leave-one-task-out CV
    for pi, (m1, m2) in enumerate(pairs):
        for L in range(n_layers):
            c1, c2 = means[m1][L], means[m2][L]
            cos_gap[pi, L] = cosine_gap(c1, c2)
            direction = (c1 - c2) / (np.linalg.norm(c1 - c2) + 1e-9)
            s1 = states[m1][:, L, :] @ direction
            s2 = states[m2][:, L, :] @ direction
            y = np.concatenate([np.ones(len(s1)), np.zeros(len(s2))])
            scores = np.concatenate([s1, s2])
            try:
                auroc_in_sample[pi, L] = roc_auc_score(y, scores)
            except Exception:
                auroc_in_sample[pi, L] = 0.5

            # Leave-one-task-out CV — only when task_ids are available
            tids_m1 = mode_task_ids[m1]
            tids_m2 = mode_task_ids[m2]
            if tids_m1 is None or tids_m2 is None:
                auroc_lototask[pi, L] = np.nan
                continue
            # Tasks that appear in BOTH modes (paper-grade design has all
            # tasks in all modes, so this is usually all 24)
            common_tasks = sorted(set(tids_m1.tolist()) & set(tids_m2.tolist()))
            if len(common_tasks) < 3:
                auroc_lototask[pi, L] = np.nan
                continue
            fold_aurocs = []
            for held_out_tid in common_tasks:
                # Train: all examples whose task_id != held_out_tid
                train_mask_m1 = tids_m1 != held_out_tid
                train_mask_m2 = tids_m2 != held_out_tid
                test_mask_m1 = tids_m1 == held_out_tid
                test_mask_m2 = tids_m2 == held_out_tid
                if (train_mask_m1.sum() == 0 or train_mask_m2.sum() == 0 or
                        test_mask_m1.sum() == 0 or test_mask_m2.sum() == 0):
                    continue
                train_c1 = states[m1][train_mask_m1, L, :].mean(0)
                train_c2 = states[m2][train_mask_m2, L, :].mean(0)
                train_dir = (train_c1 - train_c2) / (np.linalg.norm(train_c1 - train_c2) + 1e-9)
                test_s1 = states[m1][test_mask_m1, L, :] @ train_dir
                test_s2 = states[m2][test_mask_m2, L, :] @ train_dir
                test_y = np.concatenate([np.ones(len(test_s1)), np.zeros(len(test_s2))])
                test_scores = np.concatenate([test_s1, test_s2])
                if len(np.unique(test_y)) < 2:
                    continue
                try:
                    fold_aurocs.append(roc_auc_score(test_y, test_scores))
                except Exception:
                    pass
            auroc_lototask[pi, L] = float(np.mean(fold_aurocs)) if fold_aurocs else np.nan

    pca_var = np.zeros((len(MODES), n_layers))
    for mi, mode in enumerate(MODES):
        X = states[mode]  # (n, 37, 2560)
        for L in range(n_layers):
            if X.shape[0] >= 11:
                n_comp = min(10, X.shape[0] - 1)
                pca_var[mi, L] = PCA(n_components=n_comp).fit(X[:, L, :]).explained_variance_ratio_.sum()

    peak = {}
    for pi, (m1, m2) in enumerate(pairs):
        L = int(np.argmax(cos_gap[pi]))
        peak[f"{m1}_vs_{m2}"] = {
            "layer": L,
            "gap": float(cos_gap[pi, L]),
            "auroc_in_sample_at_peak": float(auroc_in_sample[pi, L]),
            "auroc_lototask_at_peak": (
                float(auroc_lototask[pi, L])
                if not np.isnan(auroc_lototask[pi, L]) else None
            ),
        }

    # Replace NaN with None for JSON serializability
    def _nan_to_none(arr):
        return [None if np.isnan(x) else float(x) for x in arr]

    metrics = {
        "n_examples": int(H.shape[0]), "n_layers": int(n_layers), "n_modes": len(MODES),
        "modes": MODES, "n_per_mode": {m: int(len(states[m])) for m in MODES},
        "pairwise_cosine_gap": {f"{m1}_vs_{m2}": cos_gap[pi].tolist()
                                  for pi, (m1, m2) in enumerate(pairs)},
        "pairwise_auroc_in_sample": {f"{m1}_vs_{m2}": auroc_in_sample[pi].tolist()
                                       for pi, (m1, m2) in enumerate(pairs)},
        "pairwise_auroc_lototask": {f"{m1}_vs_{m2}": _nan_to_none(auroc_lototask[pi])
                                      for pi, (m1, m2) in enumerate(pairs)},
        "pca_top10_var_ratio": {m: pca_var[mi].tolist() for mi, m in enumerate(MODES)},
        "peak_disruption_layers": peak,
        "auroc_protocol_note": (
            "auroc_in_sample fits mode-mean direction on all examples and scores those "
            "same examples (inflated, NOT held-out decodability). auroc_lototask is "
            "leave-one-task-out cross-validation: for each held-out task, fit direction "
            "on the remaining tasks' means, then score the held-out task's examples. "
            "Report lototask as the paper-grade linear-readability metric; in-sample is "
            "kept for descriptive comparison only. Bug 3 fix per codex methodology audit "
            "2026-05-12."
        ),
    }

exec
/bin/bash -lc "sed -n '1,260p' docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
# Method 4.2 v1 vs v2 cosine gap comparison — Bug 1+2 propagation audit

**Status**: Land 2026-05-12 late-late, after Myriad 359736 (cls v2) + 359737 (reddit v2) re-extraction with Bug 1 (tier filter) + Bug 2 (production `[SOM_MARKS]` format) + Bug 5 (model revision pin) fixes.

## Headline result

**§5.7 "three-axis hierarchy with quantitatively distinct magnitudes" claim is INVALIDATED by v2 data.**

V1 numbers came from buggy NPZ where the SOM_MARKS regex `^\[\d+\]\s+\w+` extracted only 38 chars / 3 lines per task, dropping 71/72 marks. All Stage 4 cosine geometry was computed on near-empty text payloads where the only differentiator between flat-text modes (som / phantom_som / phantom_text) was prompt template. Modes still separated perfectly (AUROC 1.000) but the cosine-gap magnitudes were artifacts of prompt-template differences, not text-payload differences.

V2 NPZ uses production `_extract_text_marks` (72-line full payload with `[id=N] {label}` envelope). Modes still separable (AUROC 1.000), but axis-1 + axis-2 cosine magnitudes collapse to noise level. Image-axis magnitudes preserve.

## Side-by-side peak comparison (cls, N=24 strong-tier)

| Mode pair | v1 buggy peak | v2 fixed peak | Magnitude Δ | Layer Δ |
|---|---|---|---|---|
| DOM ↔ Vision (image axis) | L04 0.0653 | L04 0.0670 | unchanged | unchanged |
| P-prompt ↔ Vision (image axis) | L04 0.0649 | L04 0.0664 | unchanged | unchanged |
| P-text ↔ Vision (image axis) | L36 0.0614 | **L04** 0.0602 | unchanged | **earlier** |
| P-SoM ↔ Vision (image axis) | L36 0.0613 | **L04** 0.0599 | unchanged | **earlier** |
| DOM ↔ SoM (image axis) | L04 0.0604 | **L36** 0.0496 | -18% | **boundary-shift** |
| P-prompt ↔ SoM (image axis) | L04 0.0600 | **L36** 0.0439 | -27% | **boundary-shift** |
| P-text ↔ SoM (image axis) | L20 0.0494 | **L36** 0.0488 | -1% | boundary-shift |
| **P-SoM ↔ SoM (image axis, paper §5.7 image-axis anchor)** | **L17** 0.0412 | **L36** 0.0416 | unchanged | **L17 → L36** |
| DOM ↔ P-SoM | L23 0.0321 | **L36** 0.0152 | **-53%** | L23 → L36 |
| P-prompt ↔ P-SoM (axis-1 SoM-prompt) | L23 0.0292 | **L36** 0.0048 | **-84%** | L23 → L36 |
| P-text ↔ P-prompt | L23 0.0288 | **L36** 0.0081 | **-72%** | L23 → L36 |
| **DOM ↔ P-text (axis-1 DOM-prompt, paper §5.7 axis-1 anchor)** | **L23** 0.0254 | **L36** 0.0047 | **-81%** | L23 → L36 |
| SoM ↔ Vision | L22 0.0238 | **L36** 0.0255 | +7% | boundary-shift |
| **P-text ↔ P-SoM (axis-2, paper §5.7 axis-2 anchor)** | L23 0.0114 | **L36** 0.0088 | -23% | L23 → L36 |
| DOM ↔ P-prompt | L36 0.0067 | L36 0.0068 | unchanged | unchanged |

## Headline ratios

| Ratio | v1 (3:1 ratio claim) | v2 (reality) |
|---|---|---|
| Image axis magnitude (P-SoM↔SoM) | 0.041 | 0.042 |
| Text-format axis (DOM↔P-text) | 0.025 | **0.005** |
| Prompt-family axis (P-text↔P-SoM) | 0.011 | 0.009 |
| Image / text-format ratio | **1.7x** | **8x** |
| Image / prompt-family ratio | **3.7x** | **5x** |
| Text-format / prompt-family ratio | **2.3x** | **0.5x** ← axis-1 NOW SMALLER than axis-2 |

The "image > text-format > prompt-family" hierarchy with 4:3:1-ish quantitative ratio (v1) is **wrong**. V2 reality: image axis dominates by ~5-10x; axis-1 is **smaller than** axis-2 (reversed ranking); both axis-1 and axis-2 are noise-level (<0.01 cosine).

## L17 cosine gap snapshot (cls + reddit cross-site)

| Mode pair | cls v1 | cls v2 | reddit v1 | reddit v2 |
|---|---|---|---|---|
| DOM ↔ P-text | 0.0120 | **0.0021** | (similar) | **0.0019** |
| DOM ↔ P-SoM | 0.0124 | **0.0029** | (similar) | **0.0031** |
| P-text ↔ P-prompt | 0.0132 | **0.0031** | — | **0.0032** |
| P-text ↔ P-SoM (axis-2) | 0.0028 | 0.0019 | — | 0.0020 |
| DOM ↔ SoM (image axis) | 0.0557 | 0.0452 | — | 0.0450 |
| DOM ↔ Vision (image axis) | 0.0545 | 0.0571 | — | 0.0537 |

Reddit cross-site replication confirms the cls pattern: image-axis magnitudes preserve, axis-1 + axis-2 collapse to sub-permille at L17.

## AUROC lototask (held-out, paper-grade Bug 3 fix)

All pairs at all layers report AUROC lototask = 1.000 (perfect held-out linear separability). The modes ARE distinguishable in residual stream; the **magnitude of the mode-mean difference** is just much smaller than v1 claimed.

This is the key reframe: **separability survives, magnitude does not**. Cosine gap measures effect SIZE; AUROC measures CLASSIFICATION RELIABILITY. They can dissociate.

## What this means for paper §5

**§5.7 three-axis hierarchy** (the prior framing):
> "Three quantitatively distinct axes: image axis L17 0.041, text-format L23 0.029, prompt-family L23 0.011, with 4:3:1 magnitude ratio that holds cross-site."

→ **INVALIDATED**. Replace with:
> "All three axes are linearly readable in residual stream (held-out AUROC 1.000 across cls and reddit). The image axis dominates geometrically (~0.04-0.07 cosine peak) and emerges by L04. Text-format and prompt-family axes produce sub-permille mean-difference (cosine ~0.005-0.009) without a localized layer peak (monotone rise to boundary L36). The geometric magnitude rank-order reverses cross-site at L17 (axis-1 ≤ axis-2), indicating these axes are noise-level rather than quantitatively distinct dimensions."

**§5.2 Method 4.2** (cosine gap table at L17):
- All non-image-axis numbers drop 4-8x (re-run on v2 NPZ provides canonical values)
- L17 ceases to be a meaningful "disruption locus" for text-format / prompt-family axes — they peak at L36 (boundary monotone)

**§5.5 image-axis peak-layer dichotomy** (paper claims "no-image side's text format predicts peak layer with zero overlap"):
- v1 had: 4 pairs at L04 (AXTree no-image side) vs 4 pairs at L17-L36 (flat-marks no-image side)
- v2 reorganization: DOM/P-prompt ↔ Vision still L04; **P-text/P-SoM ↔ Vision shifted from L36 → L04** (BREAKS dichotomy); DOM/P-prompt/P-text/P-SoM ↔ SoM ALL at L36 now (collapses dichotomy on SoM image side)
- → **§5.5 dichotomy ALSO needs significant revision**. The clean "AXTree → L04, flat-marks → late" pattern is partially v1 artifact.

**§5.4 Stage 2/3 patching** (Cell A-H/D-G/H-text/H-prompt/H-d/Exp 5):
- These do NOT use Stage 4 NPZ; they use archive_subset directly via Stage 2B build_som_marks which calls production code
- All Stage 2/3 patching results **REMAIN VALID**
- Exp 5 cellhprompt cls + red axis-2 patching (80-125% capture of combined image+prompt displacement): **INTACT**
- Mid-layer L11-L17 patching effect: **INTACT**

**§5.3 Method 4.4 steering** (45-cell layer-α sweep):
- Separate pipeline (uses run_stage4_method44_v2_sweep + different feature extraction): **INTACT**

**§5.6 four-vertical-defense stack**:
- Per-task fragility (uses Stage 4 NPZ): NEEDS RE-RUN on v2
- Selection-bias (reverse-tier H1): uses format variation NPZ, separately audited (INTACT but baseline caveat)
- Cross-site H1: format variation (INTACT)
- Cross-site Mirage geometry: NEEDS RE-RUN on v2

**Exp 1 axis-2 layer profile** (`axis2_layer_profile.md`): NEEDS RE-RUN on v2; current 4:3:1 ratio claim is invalidated.

**Exp 3 logit lens** (`axis2_logit_lens.md`): NEEDS RE-RUN on v2. Output-space KL magnitudes likely also change (the lm_head amplifies whatever residual-stream signal is there).

## What still stands for paper

✅ §1 hero (drop-one CI strict-positive reddit + cls): unchanged
✅ §4 empirical SR tables (after 11.90→12.38 canonicalization): unchanged
✅ §4.5 reddit behavioral: unchanged
✅ §5.4 Stage 2/3 patching + Exp 5 axis-2 causal: unchanged
✅ §5.3 Method 4.4 steering: unchanged
✅ §6 image-axis early L04 separation: unchanged (real)
✅ Held-out AUROC 1.000 linear-readability: unchanged

## New cleaner mechanism story

> **Three claim layers, distinct evidence types**:
> 1. **Linear readability** (Method 4.2 lototask AUROC 1.000): all 6 modes linearly separable in residual stream; small cosine magnitudes but reliable classification
> 2. **Geometric magnitude is mostly image-axis driven** (Method 4.2 cosine peak): image presence produces ~0.04-0.07 cosine separation early L04; other axes produce sub-permille separation throughout
> 3. **Causal patching effect at mid-layer L11-L17** (Stage 2/3 + Exp 5): 20-30% target-overlap displacement under final-token replacement patching, content-specific (task-shuffled control 359768/359769 in flight)
>
> The disjoint between **small geometric magnitude (0.005-0.009)** and **large causal patching effect (0.20-0.30)** is the new headline claim — it means residual-stream geometry underestimates causal influence by orders of magnitude. This is paper-grade-novel and reviewer-defensible.

## Files / provenance

- v2 NPZ: `results/mechanistic/stage4_multimode_b1_{cls,reddit}/hidden_states_v2_fixed.npz`
- v2 provenance: `hidden_states_v2_fixed.provenance.json` (git SHA, model revision, task IDs, formatter hash)
- v2 metrics: `/tmp/method42_{cls,red}_v2.{json,md}` (canonical paths after rerun: `results/mechanistic/stage4_multimode_b1_*/method42_metrics.json`)
- Legacy v1 metrics still in `docs/checkpoints/stage4_method42_results.md`
- Comparison source: this file

exec
/bin/bash -lc "sed -n '1,260p' docs/checkpoints/mechanism/results/axis2_logit_lens_v2.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)

Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.
For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement
across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets
amplified into output distribution divergence by late-layer decoding.

## Classifieds site

### Axis-2 (prompt-family) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L25** | 0.0879 | 0.0134 | 0.0520 | 0.0000 |
| DOM vs P-prompt  (axis-2 hierarchical) | **L21** | 0.0459 | 0.0026 | 0.0240 | 0.0000 |

### Axis-1 (text-format) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L3** | 0.0425 | 0.0164 | 0.0096 | 0.0000 |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L3** | 0.0393 | 0.0167 | 0.0242 | 0.0000 |

## Reddit site

### Axis-2 (prompt-family) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L25** | 0.0574 | 0.0192 | 0.0391 | 0.0000 |
| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0488 | 0.0106 | 0.0415 | 0.0000 |

### Axis-1 (text-format) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L2** | 0.0330 | 0.0126 | 0.0322 | 0.0000 |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L3** | 0.0391 | 0.0136 | 0.0214 | 0.0000 |

## Interpretation

Three hypotheses tested:

- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family
  effect bypasses logit lens, only visible via attention heads or runtime decoding.
- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →
  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling
  'knows but says differently' mirror).
- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →
  prompt prior signal proportional to mid-layer geometry, no amplification.

Cross-site replication should hold for any of the three. Compare axis-2 KL magnitudes to
axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.

codex
## Verdict line
The three claims do not survive hostile reading as written: A is a plausible dissociation overclaimed as magnitude comparability, B is directionally credible but under-controlled, and C is broken as an “amplification” claim.

## Claim A: Cosine-causal disjoint
Strength: weak  
Attack: Quote: “**causal patching effect magnitude 20-30%** vs **geometric magnitude 0.5-1%**” and “KL ~0.05-0.09 … amplified 8-44× from cosine.” This compares three different objects: cosine distance between per-mode mean residual vectors, KL between decoded distributions, and behavioral overlap change under intervention. Those are not commensurate units, so “orders of magnitude” is mathematically soft at best. “Disjoint” is defensible as a qualitative dissociation, not as a ratio claim.  
Defuse: Reframe as “small mean residual-stream displacement can coexist with high linear decodability and large causal patch effects.” To save the stronger version, report matched-layer, matched-pair normalized effect sizes: per-example residual displacement relative to within-mode variance, intervention effect relative to random/task-shuffled patch controls, and bootstrap CIs. Avoid cosine/KL/overlap ratios unless converted into a shared predictive or causal scale.  
Effort: 3-6 hours

## Claim B: v1→v2 magnitude collapse
Strength: weak  
Attack: Quote: “V1 numbers came from buggy NPZ where the SOM_MARKS regex … extracted only 38 chars / 3 lines per task, dropping 71/72 marks” and “V2 reality: image axis dominates … axis-1 and axis-2 are noise-level.” The regex bug proves v1 was not production-faithful, but it does not by itself prove v2 reveals the true mechanistic magnitude. The skeptical alternative is live: if v2 injects nearly identical full `[id=N] {label}` blocks into `som / phantom_som / phantom_text / phantom_prompt`, it may intentionally or accidentally homogenize the very text-format contrast being measured. V1 could have exposed sensitivity to which marks survive truncation, prompt placement, or local text salience, even if it was not the intended pipeline.  
Defuse: Distinguish “bug correction” from “signal masking” with explicit prompt-delta controls. Run a small factorial: full marks vs consistently truncated marks across all modes; identical mark payload with only template varied; shuffled labels/ids; labels removed; length-matched DOM text; and production-real prompts. Show that v1’s large axis-1 appears only under asymmetric extraction failure, while task behavior/patching tracks v2 production prompts. Then call v1 an artifact confidently.  
Effort: 6-10 hours

## Claim C: Logit lens amplification
Strength: broken  
Attack: Quote: “Apply Qwen3-VL-4B’s final_norm + lm_head to per-layer **per-mode mean** hidden states.” This decodes averaged hidden states that do not correspond to actual forward-pass states. Because `final_norm`, softmax, and KL are nonlinear, `KL(decode(mean h_a), decode(mean h_b))` is not the average of task-level output divergences. The “8-44× amplification” also compares cosine units to KL units, so the ratio is not meaningful. As written, this is an exploratory mean-state visualization, not evidence that the model amplifies the underlying per-task signal.  
Defuse: Compute paired per-task/per-example logit-lens KL instead: for each task and layer, decode the actual hidden state for each mode, compute KL for matched mode pairs, then average KLs across tasks with bootstrap CIs. Also report the decoded-mean KL beside the mean of decoded KLs to quantify the averaging artifact. Only call it amplification if per-task KL rises in the same L21-L25 window and exceeds controls.  
Effort: 4-8 hours

## Single highest-leverage move tonight
Run the per-task logit-lens KL and one prompt-delta audit table. If C collapses, downgrade logit lens to exploratory; if B’s prompt controls show v2 is production-faithful without homogenization artifacts, the paper can still keep a cleaner “small geometry, large causal intervention” story without the unsafe amplification math.

=== AUDIT COMPLETE ===
tokens used
27,391
## Verdict line
The three claims do not survive hostile reading as written: A is a plausible dissociation overclaimed as magnitude comparability, B is directionally credible but under-controlled, and C is broken as an “amplification” claim.

## Claim A: Cosine-causal disjoint
Strength: weak  
Attack: Quote: “**causal patching effect magnitude 20-30%** vs **geometric magnitude 0.5-1%**” and “KL ~0.05-0.09 … amplified 8-44× from cosine.” This compares three different objects: cosine distance between per-mode mean residual vectors, KL between decoded distributions, and behavioral overlap change under intervention. Those are not commensurate units, so “orders of magnitude” is mathematically soft at best. “Disjoint” is defensible as a qualitative dissociation, not as a ratio claim.  
Defuse: Reframe as “small mean residual-stream displacement can coexist with high linear decodability and large causal patch effects.” To save the stronger version, report matched-layer, matched-pair normalized effect sizes: per-example residual displacement relative to within-mode variance, intervention effect relative to random/task-shuffled patch controls, and bootstrap CIs. Avoid cosine/KL/overlap ratios unless converted into a shared predictive or causal scale.  
Effort: 3-6 hours

## Claim B: v1→v2 magnitude collapse
Strength: weak  
Attack: Quote: “V1 numbers came from buggy NPZ where the SOM_MARKS regex … extracted only 38 chars / 3 lines per task, dropping 71/72 marks” and “V2 reality: image axis dominates … axis-1 and axis-2 are noise-level.” The regex bug proves v1 was not production-faithful, but it does not by itself prove v2 reveals the true mechanistic magnitude. The skeptical alternative is live: if v2 injects nearly identical full `[id=N] {label}` blocks into `som / phantom_som / phantom_text / phantom_prompt`, it may intentionally or accidentally homogenize the very text-format contrast being measured. V1 could have exposed sensitivity to which marks survive truncation, prompt placement, or local text salience, even if it was not the intended pipeline.  
Defuse: Distinguish “bug correction” from “signal masking” with explicit prompt-delta controls. Run a small factorial: full marks vs consistently truncated marks across all modes; identical mark payload with only template varied; shuffled labels/ids; labels removed; length-matched DOM text; and production-real prompts. Show that v1’s large axis-1 appears only under asymmetric extraction failure, while task behavior/patching tracks v2 production prompts. Then call v1 an artifact confidently.  
Effort: 6-10 hours

## Claim C: Logit lens amplification
Strength: broken  
Attack: Quote: “Apply Qwen3-VL-4B’s final_norm + lm_head to per-layer **per-mode mean** hidden states.” This decodes averaged hidden states that do not correspond to actual forward-pass states. Because `final_norm`, softmax, and KL are nonlinear, `KL(decode(mean h_a), decode(mean h_b))` is not the average of task-level output divergences. The “8-44× amplification” also compares cosine units to KL units, so the ratio is not meaningful. As written, this is an exploratory mean-state visualization, not evidence that the model amplifies the underlying per-task signal.  
Defuse: Compute paired per-task/per-example logit-lens KL instead: for each task and layer, decode the actual hidden state for each mode, compute KL for matched mode pairs, then average KLs across tasks with bootstrap CIs. Also report the decoded-mean KL beside the mean of decoded KLs to quantify the averaging artifact. Only call it amplification if per-task KL rises in the same L21-L25 window and exceeds controls.  
Effort: 4-8 hours

## Single highest-leverage move tonight
Run the per-task logit-lens KL and one prompt-delta audit table. If C collapses, downgrade logit lens to exploratory; if B’s prompt controls show v2 is production-faithful without homogenization artifacts, the paper can still keep a cleaner “small geometry, large causal intervention” story without the unsafe amplification math.

=== AUDIT COMPLETE ===
