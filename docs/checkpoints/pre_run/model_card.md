# Model Card

> Per Mitchell et al. 2019 "Model Cards for Model Reporting". Addresses
> audit constraint **A12** (model-card-like disclosure for B0/B1 model
> identity, access, decoding, and limitations).

This paper studies **three web-agent baselines**: **B0** (Qwen3-VL-235B-A22B
proxy API), **B1** (Qwen3-VL-4B local), and **B2** (Gemma3-VL `google/gemma-3-4b-it`
local, added 2026-05-14 as a cross-family robustness-check (4B parity) control vs B1 at
4B parity). Below are their full identity / capability / limitation cards.

---

## B0 — Qwen3-VL-235B-A22B (proxy API)

### Identity

| Field | Value |
|---|---|
| Provider | HolisticAI internal proxy → upstream Qwen API |
| Model name | `qwen3-vl-235b-a22b` (proxy `api_name`: `qwen.qwen3-vl-235b-a22b`) |
| Architecture | Mixture-of-Experts (235B total params, ~22B active — the "A22B" suffix) |
| Modality | Vision-language: text + image |
| Access | Closed-weight, API-only via HolisticAI proxy at `${PROXY_BASE_URL}` (gitignored) |
| Authentication | `${PROXY_API_KEY}` env var, sourced from `.env` (gitignored) |
| Cost (per token) | Subsidized by HolisticAI lab budget; replicators with separate Qwen API access pay vendor rates |

### Decoding

| Parameter | Value |
|---|---|
| Temperature | **0.0** (greedy server-side decoding) |
| max_tokens | per-task, default 4096 |
| Top-p / top-k | not used (temperature=0 → deterministic) |
| Stop tokens | per-task config, including `</finish>` |
| Server-side determinism | **best-effort**: API providers don't strictly guarantee bitwise determinism even at temperature=0 (load-balancing across model replicas, MoE routing variance, kernel selection may introduce <1‰ output drift). Reported as "one controlled stochastic sample" with task-level bootstrap uncertainty per `preregistration.md §7`. |

### Capability profile (intended use)

- Web-agent multi-step reasoning over DOM / SoM / Vision observations
- VWA classifieds / reddit / shopping (910 tasks across 3 sites in scope)
- 2-step lookahead reasoning for action selection
- N/A task detection (terminating with `finish.answer="N/A"` when site contains no valid candidate)

### Limitations

- **Stochasticity**: server-side may drift on identical input (see decoding above)
- **Action grammar**: VWA verbose action format; agent occasionally emits malformed action JSON requiring fallback to no-op
- **No system role**: paper-grade convention uses user-role-only chat template (`memory/MEMORY.md` guard rail) — preserves hundreds of episode comparison consistency
- **Cost asymmetry vs B1**: B0 is API-only, scales with usage; B1 is fully local. Asymmetry disclosed as design constraint per `paper_planning.md` and `preregistration.md`
- **⚠️ B0 vs B1/B2 deployment-stack confounder (B-127 disclosure 2026-05-15 per gemini Mode C P2-4)**: The B0 (API) vs B1/B2 (local) split is more than just a scale comparison — B0 has a GLM-5.1 parse-error rescue scaffold (`proxy_api_agent.py::_call_glm_extract`, enabled by `use_glm_fallback: true`) that converts malformed B0 responses into valid actions; B1 + B2 have no such fallback path. This is a hidden infrastructure confounder for any B0-vs-B1 scale claim — observed SR differences mix capability-scale with API-scaffold rescue. Section 8 (limitations) explicitly admits this confounder; scale claims in paper §1 are appropriately softened. (See also section3 §3.5.1 cross-baseline disclosure.)
- **Closed weights**: cannot probe internals (logits / attention / hidden states); SteerMoE-style mechanism analysis deferred to future work (see `negative_results_registry.md` entry #9)
- **Proxy availability**: API endpoint may degrade or sunset post-publication; replicators verify via released traces (`preregistration.md §7` reproducibility tier)

---

## B1 — Qwen3-VL-4B-Instruct (local)

### Identity

| Field | Value |
|---|---|
| Provider | Alibaba (Qwen team) |
| Model name | `Qwen/Qwen3-VL-4B-Instruct` |
| HuggingFace revision SHA | **`ebb281ec70b05090aa6165b016eac8ec08e71b17`** (pinned, see `locked_versions.md`) |
| Architecture | Dense decoder-only transformer (4B params) + vision encoder (~700M params for image embedding) |
| Modality | Multimodal text + image (paper-1: VWA agent role; Stage 2 mechanistic analysis is **paper-2 scope** per advisor 2026-05-14, B-132 banner update 2026-05-15) |
| Access | **Open weights** — fully reproducible, replicators download from HuggingFace |
| License | Apache 2.0 (per HF model card) |
| Storage | ~10GB bf16 weights cached at `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec.../` |

### Decoding

| Parameter | Value |
|---|---|
| Decoding mode | **Greedy** (`do_sample=False`), bitwise deterministic per (HF SHA + seed + GPU class) |
| Temperature | n/a (greedy) |
| max_new_tokens | per-task (4096 baseline; 50 for Stage 2 patching continuations) |
| Seed | **42** (`configs/exp_v2_base.yaml`) — propagated via `_seed_global_rng()` per (cond, seed) |
| Numerical determinism | bitwise across **A100 80GB / V100 / GB10 within tolerance** (≤1e-6 hidden-state delta per `numerical_determinism_check.py` audit F6). Math SDPA backend forced for cross-architecture portability per B-81h fix. |

### Capability profile

- Same VWA agent role as B0 (DOM / SoM / Vision / phantom_som / phantom_dom modes)
- Smaller model → expected weaker SR than B0 (capability contrast is **part of the scientific design**, not a limitation)
- Mechanistic Stage 2 analysis (activation patching) requires open-weights → **only B1 supports paper §5 mechanism claims**
- Layers: 36 transformer blocks (L0 = embedding output, L35 = final post-block); 24 attention heads; hidden dim 2048

### Limitations

- **Smaller model**: SR is lower than B0 on most tasks; visual-hijack / click-loop patterns are dominant failure mode (per `section1_intro.md`: B1 70.4% visual-hijack vs B0 53.3% early-finish on disagreement slice)
- **No CoT prefix supplied** (paper-grade convention): agent reasons in `thought` field per VWA prompt template, no scratch-pad CoT trick
- **GPU dependency**: bf16 inference requires sm_70+ (V100 OK, T4 OK with reduced batch); CPU inference too slow for paper-scale runs
- **Cross-machine numerical determinism caveat**: minor (<1e-6) hidden-state drift across V100 vs A100 due to cuDNN kernel selection — bounded under tolerance but worth noting (audit F6)

---

## B2 — Gemma3-VL `google/gemma-3-4b-it` (local, added 2026-05-14)

### Identity

| Field | Value |
|---|---|
| Provider | Google (Gemma team) |
| Model name | `google/gemma-3-4b-it` |
| HuggingFace revision SHA | ⏳ **pending lock** (pin at A100 bring-up time, matching B1 protocol; recorded into `env_snapshot.json`) |
| Architecture | Dense decoder-only transformer (4B params) + vision encoder (multimodal) |
| Modality | Multimodal text + image |
| Access | **Open weights** — fully reproducible, replicators download from HuggingFace |
| License | Gemma Terms of Use (per HF model card) |
| Storage | bf16 fits A100-PCIE-40GB unquantized |
| Capability rationale | 4B params parity with B1 → **cross-family robustness check at 4B-parameter parity** (Google Gemma vs Alibaba Qwen lineage at matched scale); advisor discussion 2026-05-14 |

### Decoding

| Parameter | Value |
|---|---|
| Decoding mode | **Greedy** (`do_sample=False`), bitwise deterministic per (HF SHA + seed + GPU class) |
| Temperature | n/a (greedy) |
| max_new_tokens | per-task (matching B1 protocol) |
| Seed | **42** (`configs/exp_v2_base.yaml` — shared with B0/B1) |
| Numerical determinism | bitwise across A100 PCIE-40GB (canonical host); cross-architecture portability TBD per first run |

### Capability profile

- Same VWA agent role as B0/B1 (DOM / SoM / Vision / phantom_som / phantom_dom / phantom_text / phantom_prompt modes)
- Cross-family control at matched 4B capability — comparison to B1 (4B Qwen) tests **family** effect at controlled scale
- Open-weights enables future mechanism analysis (paper-2 scope per advisor 2026-05-14; not part of paper-1)

### Limitations

- **HF SHA pin pending** — lock at A100 bring-up time per phase1_plan §B0 #11 prereq
- **Smaller model**: SR is expected lower than B0; failure mode patterns TBD (Phantom-SoM cross-family hold rate is one of the empirical questions)
- **No system role** (paper-grade convention) — same as B0/B1
- **Mechanism §5 paper-2 deferred per advisor 2026-05-14** — section5_mechanism.md remains paper-2 working draft, mechanism extension to B2 future work
- **GPU dependency**: bf16 inference requires sm_70+ A100/V100 class; CPU inference too slow for paper-scale runs
- **Host migration interaction**: B2 only runs on canonical A100 stack (no DGX pre-fix archive); cross-host comparison N/A by design

---

## Summary Comparison

| Dimension | B0 (API) | B1 (local) | B2 (local, added 2026-05-14) |
|---|---|---|---|
| Open weights | ❌ | ✓ | ✓ |
| Reproducibility | Verifiable from traces | **Byte-identical** | **Byte-identical** (pending HF SHA pin) |
| Mechanism analysis (paper-2 scope) | Not possible (closed) | Stage 2 patching available | Stage 2 future work (open weights enable) |
| SR (typical) | Higher | Lower | TBD (cross-family control) |
| Cost per episode | $0.01-0.10 | ~free (local compute) | ~free (local compute) |
| Latency | Network-bound (~5-20s) | GPU-bound (~2-10s on A100) | GPU-bound TBD on A100 |
| Determinism | Best-effort (server-side) | Bitwise (greedy + seed) | Bitwise (greedy + seed) |
| In-scope for paper §5 mechanism | ❌ Future work | ⏸️ Paper-2 deferred (advisor 2026-05-14) | ⏸️ Paper-2 future work |
| Family | Alibaba Qwen | Alibaba Qwen | **Google Gemma** (cross-family) |
| Param scale | 235B (A22B MoE) | 4B | 4B (matched with B1) |

The **B0 vs B1 vs B2 design**: B0 supplies the high-capability behavioral
characterization (Qwen-family 235B); B1 supplies the open-weight matched 4B
within-family comparison; **B2 supplies the cross-family robustness-check (4B parity)
control at 4B parity with B1** (Google Gemma vs Alibaba Qwen lineage). Together
the 3 baselines disambiguate (a) capability scale effects (B0 vs B1) from (b)
family-level differences at the same 4B parameter scale (B1 vs B2 — note: 4B parity is parameter count only, not full capability anchor; no MMMU/VQA zero-shot benchmark established at preregistration time, so this is a robustness check rather than a strict matched-capability control. See model_card §"Capability rationale" 2026-05-15 downgrade note per gemini Mode C P0-5). Mechanism analysis is paper-2
scope per advisor 2026-05-14.

## References

- Mitchell et al. 2019 "Model Cards for Model Reporting" (NEEDS_BIB_ENTRY)
- `pre_run/locked_versions.md` for the exact pinned revision SHAs
- `preregistration.md §7` for reproducibility scope per model
- `negative_results_registry.md` entry #9 for SteerMoE / B0 self-probe retract
