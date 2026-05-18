# NeurIPS 2025 Reproducibility Checklist (paper-time submission artifact)

> **Provenance**: Created /stress A2.9 P0-1-AC* (Claude Mode A + gemini Mode C
> 2-AI OOB overlap) 2026-05-18 — B-1506. Submission-ready scope gate per
> NeurIPS 2025 Paper Checklist published at `https://neurips.cc/Conferences/2025/PaperChecklist`.
> Replaces ad-hoc internal coverage in `topvenue_constraints.md` (78-constraint
> internal audit format ✓/⚠️/❌, NOT the paper-time 15-Q yes/no/justify
> submission format). Q1-Q15 = NeurIPS 2024+2025 standard; Q16 = 2025-new
> LLM Use Disclosure.
>
> Each question: **Answer** {Yes / No / NA} + **Justify** with concrete artifact
> citation (file:line or §pointer). Replicators verify Yes-claims against the
> cited artifact; No / NA cannot be hand-waved.

---

## Q1. Claims — do the abstract & introduction claims accurately reflect the paper's contributions?

**Answer**: Yes
**Justify**: Paper §1 `section1_intro.md` claims are bounded by:
(a) phantom routing space empirical phenomenon (3 sibling arms: P-text, P-prompt,
P-SoM) with 4-fold drop-in property — pre-registered as H1+H2(a) gating
hypotheses in `pre_run/preregistration.md §2`;
(b) cost-aware router empirical evaluation — pre-registered as H10 Pareto
non-dominance gate in `pre_run/preregistration.md §2`. Cross-site (cls+red)
+ cross-baseline (B0+B1+B2) external-validity scope bounded by `pre_run/preregistration.md §2.7`
workshop_R1 vs main_R1 bifurcation matrix (B-1267 /stress A2.6a P0-8-C* 2026-05-18).
No abstract claim exceeds within-cell evidence at the prereg-locked 6-cell
fixed-effects estimand per decision "3A" 2026-05-14.

## Q2. Limitations — does the paper discuss limitations?

**Answer**: Yes
**Justify**: `paper_drafts/section8_limitations.md` is a dedicated §8 with 8
subsections covering: 8.1 scope/external-validity (B-1280 scaffolding-class
scope + B-1236 N=3 underdeterminedness + B-1284 B2 cross-family claim-tier
gate), 8.2 construct validity (VWA evaluator threats — `ua_match` + `string_match`
+ `program_html` + `finish_wrong_state`), 8.3 internal-validity / scaffold bugs
(viewport B-26, scroll-direction, action-vocabulary B-406 + cross-family
B-916/917/918), 8.4 pre vs post-hoc + negative results registry, 8.5 statistical
limits (Holm-Bonferroni + HKSJ small-k caveat B-1308 + H1+H10 FWER disclosure
B-1310), 8.6 mechanism §5 paper-2 deferral, 8.7 compute/cost/sustainability +
cross-baseline cost unit-basis collision B-1505, 8.8 phantom-space concept-
domain speculation B-1294+B-1296.

## Q3. Theory — for theoretical contributions, do assumptions hold and proofs exist?

**Answer**: NA
**Justify**: Paper-1 is empirical characterization + router evaluation; no
mathematical theorems are proven. The phantom routing space construct is
empirical (validated within 6 prereg-locked cells, NOT proven over a
super-population — see §8.1 N=3 underdeterminedness disclosure). H2(a) cost
≈ DOM is a by-construction property (regex filtering on shared AXTree) +
empirical falsification check per cell (>1.20× cost ratio threshold per
`preregistration.md §H2(a)` falsification rule). Mid-layer mechanism patching
(activation patching / logit lens / SAE) is paper-2 scope per advisor 2026-05-14.

## Q4. Experimental result reproducibility — clear code/data/protocol?

**Answer**: Yes
**Justify**: `pre_run/preregistration.md §7` reproducibility scope table (7
component tiers) + `pre_run/locked_versions.md` (HF SHA / submodule SHA /
seed=42 / decoding params) + `pre_run/osf_lock_manifest.md` (8-step OSF DOI
workflow + artifact freeze registry) + `phase1_plan.md` (canonical execution
protocol). VWA submodule HEAD `ac33d2fcd9cec2fcbeddd56d0fa3da58b4c7e927`
pinned via 3-layer SBOM (HEAD commit + upstream base + tree-hash chain
sha256 `752caebdc6bd84761b2f308331f21241a9b4a28de65b46ff0007ef27d8c72778`
per prereg §7 L626-L630). Per-task config materialization via
`make vwa-generate-configs` (byte-deterministic per B-604+B-615).

## Q5. Open access to data and code — is everything publicly available with reasonable license?

**Answer**: Yes (release-time gated; OSF DOI mint at advisor witness +
submission day)
**Justify**: `pre_run/release_redaction_checklist.md` defines public release
artifact scope (✗ credentials / ✗ auth state / ✓ scripts / ✓ p79/ / ✓
configs / ✓ pre_run/ / ✓ paper_drafts/ / ✓ results/provenance/) + redaction
gate `make pre-release-check` (B-1512 wires Makefile target). OSF DOI mint
~1 week before submission per `osf_lock_manifest.md §3` 8-step workflow.
P79 code MIT, VWA MIT, WA MIT — see `ethics_license_coi_statements.md`
release license matrix (B-1501 /stress A2.9 P1-1-ABC* 2026-05-18).

## Q6. Experimental setting / details — full hyperparameters + configs?

**Answer**: Yes
**Justify**: `configs/exp_v2_base.yaml` + `configs/exp_v2_phase1_{B0,B1,B2}_{dom,som,vision,phantom_*}_{classifieds,reddit,shopping}.yaml` per (baseline × mode × site) cell explicit YAML;
`pre_run/preregistration.md §4` locks mode definitions + inclusion rules + 6-cell pool.
Per-cell seed=42 + greedy decoding + `_seed_global_rng()` per (cond, seed) iteration.

## Q7. Experiment statistical significance — error bars / CIs / multiple-comparison correction?

**Answer**: Yes
**Justify**: Paired bootstrap CIs (1000 resamples per `aggregate_phantom_lift.py`
+ seed=42) on hero P-SoM-vs-DOM drop-one oracle; one-sided FE inverse-variance
pooled superiority test at α=0.05 H0: θ_FE ≤ +1.0pp (PRIMARY H1 gate per
`preregistration.md §2 L85`). Holm-Bonferroni correction across canonical
6-layer grid for paper-2 mechanism patching. HKSJ-modified non-shrink-guarded
random-effects + DerSimonian-Laird as Appendix-D-bis sensitivity per B-1308+B-1309
/stress A2.3d. H1+H10 joint family-wise Type I disclosed at 9.75% explicitly
per B-1310 (NeurIPS-style m=1-family disclosure). K-of-N transparency-only
(no threshold) per B-1264.

## Q8. Dataset description / source / preprocessing?

**Answer**: Yes
**Justify**: `pre_run/dataset_card.md` covers VisualWebArena 234 cls + 210 reddit
tasks (paper-1 scope) + 466 shop tasks (Phase 1b deferred main paper).
N/A task exclusion at task-load (`exclude_na_tasks: true` default per B-91
fix family). Per-site task-pool sha256 in `pre_run/locked_versions.md`.
912 gitignored per-task config files (`config_files/vwa/test_{site}/{0..N}.json`)
deterministically regenerated from `config_files/vwa/test_{site}.raw.json`
templates via `scripts/generate_test_data.py` (B-604+B-615 clean-idempotent).

## Q9. License + terms of use — paper / code / models / data?

**Answer**: Yes
**Justify**: Release license matrix in `ethics_license_coi_statements.md` §License
(B-1501 /stress A2.9 P1-1-ABC* 2026-05-18) — P79 MIT / VWA MIT / WA MIT /
Qwen3-VL-4B Apache 2.0 / Qwen3-VL-235B Apache 2.0 + proxy ToS /
**Gemma3-VL `google/gemma-3-4b-it` Gemma Terms of Use** (gated, requires
HF EULA acceptance, prohibits weight redistribution) / OpenAI judge `gpt-4o-mini`
OpenAI ToS. Replicator compliance checklist provided.

## Q10. Reproducibility / public artifacts — full disclosure?

**Answer**: Yes
**Justify**: Same as Q4 + Q5. Submission-time OSF DOI mint covers all listed
artifacts. Provenance snapshot per Phase 1a paper-grade run re-evaluates
the 3-layer SBOM contract; divergence aborts the run (per
`scripts/provenance/snapshot_vwa.sh` + caller integration B-822 /stress A1.16
2026-05-17 + B-823 caller bypass fix).

## Q11. Theoretical claims (mathematical) — assumptions + proofs?

**Answer**: NA (same as Q3 — empirical paper)

## Q12. Compute resources — type, time, total?

**Answer**: Yes (post-Phase-1a-fire fill)
**Justify**: `section8_limitations.md §8.7` provides framework (energy via
NVIDIAPowerReader pynvml on A100; 220 g/kWh UK national grid 2024 average;
PUE 1.0 lower-bound + 1.5 upper-bound range per B-1510 /stress A2.9 P1-5-AC
2026-05-18); per-cell + total table `compute_cost_carbon_table.csv/md`
skeleton landed B-1508 /stress A2.9 P0-4-AB pre-fire + numerical fill
post-Phase-1a-fire via `aggregate_phase1_full_prereg_decision.py` new
output target. Compute fleet 3-tier breakout (A100 canonical / Myriad
cross-arch F6 audit / DGX Spark archive ref) — per-host GPU-hours column
B-1509 /stress A2.9 P1-4-AC. **LLM evaluator API cost (3rd-party)**:
GPT-4o-mini judge calls via `VWA_EVAL_MODEL` env (default per B-833
/stress A1.16) — ~150K API calls estimate aggregate disclosed in §8.7
B-1507 /stress A2.9 P1-2-ABC 2026-05-18.

## Q13. Negative impact / risks?

**Answer**: Yes
**Justify**: `ethics_license_coi_statements.md` §Misuse/Safety Scope (4
disclosed risks: lower automation cost broadens deployment surface /
visual hijack patterns Mirage+Scaffold / web-agent benchmark contamination /
cross-bench generalization not yet established). 4-fold drop-in property
making cheap deployment feasible disclosed; benchmark cherry-picking risk
disclosed (we report all 42 conditions / 6 cells per locked preregistration
per B-1502 /stress A2.9 P0-5-ABC* 2026-05-18).

## Q14. Crowdsourcing / human subjects?

**Answer**: No / NA
**Justify**: `ethics_license_coi_statements.md` §Institutional Review IRB:
this work uses synthetic web environments only (VWA cls/red/shop hosted in
self-controlled Docker containers with synthetic seed users); no human
subjects; no personal data; no real-user data; IRB approval NOT required
under UCL Research Ethics Framework Section II.4 (synthetic data /
no human subjects).

## Q15. Safeguards / risk mitigation for high-risk releases?

**Answer**: NA
**Justify**: Paper-1 is scientific characterization of representation routing
in web agents, NOT a deployment-ready autonomous agent system. We do NOT
release exploit templates, adversarial prompt templates, or weaponized
prompts — failure mode analyses are discussed for robustness improvement
per `ethics_license_coi_statements.md` §Misuse (item 2).

## Q16. LLM Use Disclosure (NeurIPS 2025-new) — acknowledgement of generative-AI assistance?

**Answer**: Yes
**Justify**: `ethics_license_coi_statements.md` §"LLM Use Disclosure"
(B-1507 /stress A2.9 P0-2-ABC* 2026-05-18) declares scope of LLM assistance:
Claude Code (Anthropic) for primary code review + debugging + audit;
codex CLI (OpenAI) for independent code audit cross-check; gemini CLI (Google)
for independent broad-reviewer audit + literature pointer; Gemini Deep Research
for literature DR (`docs/literature/`). The authors verified all generated
content and accept responsibility for any errors. **Final scientific decisions,
hypothesis framing, hypothesis-tier gating (H1-H10), R1-R5 framing rule
calls, and analysis interpretation were made by human authors**. No LLM
is listed as a co-author. Per NeurIPS 2025 policy boundary, LLM use for
writing/editing/formatting that does NOT affect core methodology is exempt;
this paper's LLM use partially exceeds that boundary (e.g., audit findings
landed code edits) and is therefore disclosed explicitly here per the
2025 policy spirit.

---

## Submission gate

This checklist is the canonical NeurIPS 2025 Paper Checklist artifact;
`topvenue_constraints.md` is the internal-audit cross-check (78 constraints
✓/⚠️/❌ format), NOT a submission-time substitute. At paper submission
time the author copies this Q1-Q16 yes/no/justify list into the NeurIPS
submission portal's required Paper Checklist field per the venue
template, with the specific file:line + §pointer citations preserved.

**Status banner**:
- **A2.9 audit closure 2026-05-18** /stress P0-1-AC* B-1506: file created
- Per-question artifact citations live; no Q1-Q16 is "TBD"
- Workshop submission: fully covered by Phase 1a closure
- Main paper submission: Q1+Q5+Q9 R-tier may shift per B2 outcome
  (cross-family claim-tier gate B-1284) + Phase 1b shop scope expansion

