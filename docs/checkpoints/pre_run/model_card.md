# Model Card

> Per Mitchell et al. 2019 "Model Cards for Model Reporting". Addresses
> audit constraint **A12** (model-card-like disclosure for B0/B1 model
> identity, access, decoding, and limitations).

This paper studies two web-agent baselines, **B0** and **B1**. Below are
their full identity / capability / limitation cards.

---

## B0 — Qwen3-Omni-235B-Thinking (proxy API)

### Identity

| Field | Value |
|---|---|
| Provider | HolisticAI internal proxy → upstream Qwen API |
| Model name | `qwen3-omni-235b-thinking` |
| Architecture | Mixture-of-Experts (235B total params, ~22B active) |
| Modality | Multimodal: text, image, audio (we use text + image only) |
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
- **Cost asymmetry vs B1**: B0 is API-only, scales with usage; B1 is fully local. Asymmetry disclosed as design constraint per `paper_planning.md` and `preregistration.md` (no remediation, intended scientific design)
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
| Modality | Multimodal text + image (used for both VWA agent + Stage 2 mechanistic analysis) |
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

## Summary Comparison

| Dimension | B0 (API) | B1 (local) |
|---|---|---|
| Open weights | ❌ | ✓ |
| Reproducibility | Verifiable from traces | **Byte-identical** |
| Mechanism analysis | Not possible (closed) | Stage 2 patching available |
| SR (typical) | Higher | Lower |
| Cost per episode | $0.01-0.10 | ~free (local compute) |
| Latency | Network-bound (~5-20s) | GPU-bound (~2-10s on A100) |
| Determinism | Best-effort (server-side) | Bitwise (greedy + seed) |
| In-scope for paper §5 mechanism | ❌ Future work (Zoom 4) | ✓ Primary |

The **B0 vs B1 asymmetry is part of the scientific design**: B0 supplies the
high-capability behavioral characterization, B1 supplies the open-weight
mechanism evidence. Cross-validating mechanism findings on B0 is future
work pending open-weight 235B class models or proxy access to B0 internals.

## References

- Mitchell et al. 2019 "Model Cards for Model Reporting" (NEEDS_BIB_ENTRY)
- `pre_run/locked_versions.md` for the exact pinned revision SHAs
- `preregistration.md §7` for reproducibility scope per model
- `negative_results_registry.md` entry #9 for SteerMoE / B0 self-probe retract
