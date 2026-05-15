# Ethics / License / COI / Misuse-Safety Statements (paper §8)

> Pre-written ethics / license / COI / IRB / misuse-safety paragraphs ready
> for paper §8 Discussion section. Addresses audit constraints **F7** (license,
> COI, IRB) and **H3** (misuse / safety scope for autonomous web agents).

## License + Code Release

The agent code (`p79/`), experiment runner, and analysis scripts are released
under the **MIT License**, consistent with VisualWebArena (Koh et al. 2024)
and WebArena (Zhou et al. 2024) licensing. The released artifact includes:

- All experiment scripts at the locked git commit SHA
- Configuration YAMLs with seed=42 and inference parameters (3 baselines:
  B0 Qwen3-VL-235B-A22B via proxy API, B1 Qwen3-VL-4B local, B2 Gemma3-VL
  `google/gemma-3-4b-it` local)
- Analysis pipeline: `aggregate_phantom_lift.py` (paired bootstrap),
  `power_analysis.py` (MDE at k=6), `preregistration_decision_test.py`
  (H1+H3 R1-R5 framing rule verdict at 6 statistical cells),
  `generate_per_task_sr.py` (wide-format CSV producer for decision test)
- Pre-registration document (`preregistration.md`) with locked git SHA and OSF DOI
  (filled at advisor witness time)
- **NOT included** (B-132 fix per pre_run/ residual audit 2026-05-15 evening):
  mechanism §5 artifacts (`results/mechanistic/archive_subset_b1_*` + curated
  mirage candidates + `stage2_layer_significance.py` outputs). Mechanism §5
  deferred to paper-2 per advisor 2026-05-14; paper-1 OSF DOI does NOT cite
  paper-2 forward stub artifacts. User will mint fresh mirage curation +
  Stage 2 outputs on new data for paper-2 own release.

The released artifact does **not** include:
- B0 proxy API credentials (`scripts/vwa_env_remote.sh`, `.env` — gitignored)
- Site authentication state (`.auth/` — gitignored)
- Raw model weights for Qwen3-VL-4B (HuggingFace revision SHA pinned in
  `env_snapshot.json`; replicators download from HF)

## Conflict of Interest

The authors declare **no conflicts of interest**. This research was conducted
as part of the lead author's MSc thesis at University College London (UCL),
supervised by faculty in the UCL AI Centre. Compute resources were provided by:

- **UCL Condense A100 (A100-PCIE-40GB dedicated allocation, VM `a100-jiaming-test`)** — institutional research allocation; canonical paper-grade rerun host post-2026-05-15 (self-hosted VWA Docker, no Tailscale)
- UCL Myriad HPC cluster (mixed V100/A100 nodes) — institutional research allocation; used for cross-arch numerical determinism check (audit F6)
- DGX Spark workstation (NVIDIA GB10) — shared research workstation, HolisticAI lab; **archive-only post-2026-05-15** (pre-fix Phase 1a archive data was DGX→quark Tailscale → VWA Docker stack; canonical paper-grade run migrated to A100 self-host)
- Tailscale (Personal Plan) for cross-machine networking — used for DGX→quark archive era only; A100 canonical run no longer uses Tailscale

Model API costs (B0 proxy access to Qwen3-VL-235B-A22B) were covered by lab
research budget. No external industry funding, no consulting relationships
with model providers, no equity in benchmark / agent-framework companies.

## Institutional Review / IRB

This work uses **synthetic web environments only** (VisualWebArena classifieds /
reddit / shopping benchmarks, all hosted in self-controlled Docker containers
with synthetic seed users `emma.lopez` / `MarvelsGrantMan136` / `blake.sullivan`).

- **No human subjects** participate in the experiments.
- **No personal data** is collected, stored, or analyzed.
- **No real-user data** is used; benchmark task observations are synthetic
  HTML/DOM snapshots from controlled VWA Docker images.
- IRB approval is **not required** under UCL Research Ethics Framework
  Section II.4 (synthetic data / no human subjects).

This work therefore does not require IRB review. The authors confirm this
classification with UCL Research Ethics Office before publication if requested.

## Misuse / Safety Scope (autonomous web agents)

The contribution of this paper is a **scientific characterization of
representation-routing in web agents** (specifically the phantom routing space
boundary), not a deployment-ready autonomous agent system. However, autonomous
web-agent research has known dual-use risks worth disclosing:

1. **Lower automation cost broadens deployment surface**: The 4-fold drop-in
   property we characterize (cost ≈ DOM, latency ~50% lower) reduces the
   financial barrier to running web agents at scale. This includes legitimate
   uses (accessibility tools, structured-task automation) and potential
   misuse (large-scale account creation, automated content posting,
   benchmark Sybil attacks, misinformation distribution). Our paper does
   not enable these directly, but our routing-arm characterization makes
   cheaper deployment feasible.

2. **Visual hijack patterns surfaced in this work** (Mirage / Scaffold /
   Cross-modal flow lit anchors per `paper.bib` `asadi2026mirageillusionvisualunderstanding` /
   `vu2026scaffold` / `kaduri2024whatsintheimage`) describe failure modes of
   VLM web agents under controlled SoM-style perturbations. Public
   characterization of these failure modes could in principle inform
   adversarial prompt-injection attacks; we mitigate by (a) limiting
   demonstration to benchmark synthetic tasks, (b) not releasing exploit
   templates, and (c) discussing failure modes in the context of
   robustness improvement rather than attack.

3. **Web-agent benchmark contamination risk**: Cheap inference-time routing
   could accelerate gaming of public benchmarks (VWA / WA / Mind2Web /
   AgentBench leaderboards). We disclose this risk; our paper does not
   benchmark-game (we report all 36 conditions / 6 statistical cells per
   locked preregistration — Phase 1a B0+B1+B2 × cls+red × 6 modes — not
   cherry-picked), but readers building on our work should be aware that
   benchmark selection bias is a documented hazard in this space.

4. **Cross-bench generalization not yet established**: Mechanism claims are
   scoped to Qwen-family VWA per `preregistration.md §7`. Deploying our
   findings to safety-critical web tasks (financial transactions, medical
   information retrieval, government services) would require additional
   evaluation per CLAW-Safety guidelines (`paper.bib` `wei2026clawsafety`)
   and is **not endorsed by this paper**.

The authors recommend that downstream practitioners deploying phantom-style
routing in production conduct domain-specific safety review, especially for
sites with material consequences. Our cost / latency drop-in property
arguments are not a license to deploy without domain-specific safety eval.

## Provenance + Reproducibility (audit A14 cross-reference)

See `preregistration.md §7` for the full reproducibility scope statement
covering paper-1 component tiers (B1 byte-identical / B2 byte-identical /
B0 verifiable from traces / VWA env A100 self-host / Evaluator p79-patches
f0c835b). Paper §5 mechanism analysis (Stage 2 activation patching, layer
probes, logit lens) is **paper-2 scope** per advisor 2026-05-14 — paper-1
OSF DOI does NOT cover mechanism §5 reproducibility; paper-2 will mint
its own DOI for fresh mirage curation + Stage 2 outputs (B-132 per
pre_run/ residual audit 2026-05-15).

## References

- `paper.bib` keys: `koh2024visualwebarena`, `zhou2024webarena`,
  `wei2026clawsafety`, `asadi2026mirageillusionvisualunderstanding`,
  `vu2026scaffold`, `kaduri2024whatsintheimage`
- `docs/checkpoints/pre_run/preregistration.md` §7 (reproducibility scope)
- `docs/checkpoints/pre_run/negative_results_registry.md` (open-science
  retraction registry)
