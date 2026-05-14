### Finding 1 — Format-variation still uses v1-ish SOM text [P0]
**Claim** — `fmt_som_standard` says it is “same as run_stage2b build_som_marks”, `scripts/mechanistic/run_stage4_format_variation_extract.py:82`.  
**Code reality** — it filters raw bracket lines with `startswith("[")`, while variant extraction uses `MARK_LINE_RE`, `:60-85`; it never calls production `_extract_text_marks`.  
**Attack** — This reintroduces Bug 2 into H1/format-variation baselines: `som`, `phantom_text`, `phantom_som`, and all variants are not mechanically comparable to Stage 4 v2 multimode SOM payloads. H1 numbers can mix v2 multimode with format-variation text vintage.  
**Defuse** — Replace mark extraction with production `_extract_text_marks` and re-extract.  
**Effort** — 30-60 minutes plus extraction.  
**Confidence** — high.

### Finding 2 — Grid check missed smaller-than-claimed cohorts [P1]
**Claim** — P0-2 fail-closed grid should prevent ragged/silent partial outputs, `run_stage4_multimode_extract.py:246`; `run_stage4_format_variation_extract.py:254`.  
**Code reality** — expected grid was derived from `selected` / `intents_by_tid`, so a 20-task run with `--n-tasks 24` could pass if 4 tasks were never selected. Duplicate mode cells also escaped set-diff checks.  
**Attack** — A partial archive or short manifest ships “complete” internally but not complete relative to the claimed N.  
**Defuse** — Added target-N guard and cardinality/duplicate checks.  
**Effort** — applied.  
**Confidence** — high.

### Finding 3 — Stage 2B mislabeled patcher layers as embedding-indexed [P1]
**Claim** — `ActivationPatcher` documents `layers[L]` as decoder block output, not embedding, `p79/mechanistic/activation_patching.py:39-43`.  
**Code reality** — Stage 2B plot label said `Layer index (0=embedding, ≥1=post-block)`, while patching uses `source_cache[L]` into `patcher.layers[L]`, `activation_patching.py:405-435`.  
**Attack** — Cross-pipeline prose can read Stage2B `L17` as NPZ `H[:,17,:]`; correct NPZ coordinate is `H[:,18,:]`.  
**Defuse** — Relabeled Stage2B outputs as block indices and recorded the convention.  
**Effort** — applied.  
**Confidence** — high.

### Finding 4 — Provenance was not strong enough outside multimode [P1, OOB]
**Claim** — §5 claims same pinned Qwen3-VL-4B revision across Stage 2B / Stage 4.  
**Code reality** — multimode had `--model-revision` and sidecar; format-variation and Stage2B relied on default constructor behavior and did not record revision in their primary outputs.  
**Attack** — Even if today’s default is pinned, downstream JSON/NPZ cannot prove the run used that SHA.  
**Defuse** — Added explicit revision args and output provenance.  
**Effort** — applied.  
**Confidence** — high.

Inline fixes applied: stricter grid guards in `run_stage4_multimode_extract.py` and `run_stage4_format_variation_extract.py`; `--model-revision` plus provenance sidecar for format-variation; `--model-revision`, config/run-manifest revision fields, and block-index labels for Stage2B.

Verification: `python3 -m py_compile` passes for all three edited scripts. Only an existing docstring escape warning remains in multimode.