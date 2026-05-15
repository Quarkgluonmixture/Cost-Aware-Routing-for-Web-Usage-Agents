# Dataset Card

> Per Gebru et al. 2018 "Datasheets for Datasets". Addresses audit
> constraint **A13** (datasheet-style disclosure for benchmark composition).

## Benchmark identity

| Field | Value |
|---|---|
| Name | **VisualWebArena (VWA)** |
| Citation | Koh et al. 2024 (`paper.bib` `koh2024visualwebarena`) |
| Repo | https://github.com/web-arena-x/visualwebarena |
| Submodule SHA in this work | **`832f037e2cc7ebda4a41831443a3fc9b79d06cd6`** (per `locked_versions.md`) |
| License | Apache 2.0 |

## Sites in scope

This paper uses **3 of 3** VWA sites. All sites are self-hosted Docker
containers (no external HTTP traffic to commercial sites — paper-grade
privacy + reproducibility).

| Site | N tasks | Task-pool sha256 | Site type | Hosted at |
|---|---|---|---|---|
| **classifieds** | 234 | `d36a20c1eaa1f5da...` | Visual-rich product listings (Magento-based) | **Canonical 2026-05-15+**: self-hosted on A100 (Condenser VM `a100-jiaming-test`) `:9980`. Archive (pre-2026-05-15): DGX Tailscale → quark Docker `:9980` |
| **reddit** | 210 | `ecd4ed4370740fd6...` | Text-dominated forum threads (Postmill) | **Canonical 2026-05-15+**: self-hosted on A100 `:9999`. Archive (pre-2026-05-15): DGX Tailscale → quark `:9999` |
| **shopping** | 466 | `07889e3646ee10e3...` | Mixed text + image product pages (Magento) | **Canonical 2026-05-15+ (Phase 1b deferred)**: self-hosted on A100 `:7770`. Archive (pre-2026-05-15): DGX Tailscale → quark `:7770` |
| **Total** | **910** | — | | |

**Host migration (2026-05-15)**: VWA Docker stack migrated from DGX→quark Tailscale tunnel
to A100 (Condenser VM) self-hosted Docker. Cross-host comparison NOT made per
`preregistration.md` Infrastructure migration note + Appendix A 2026-05-15 entry.
#11 A100 VM VWA Docker bring-up gates canonical Phase 1a launch.

WebArena (480 tasks across shopping/shopping_admin/reddit) is **out of scope**
for this paper per `preregistration.md §7` external validity scope. Cross-bench
generalization is explicit future work.

## Task curation methodology

### Inclusion

All in-scope VWA tasks (Phase 1a = cls + red = 444 tasks; Phase 1b deferred adds shop = 466 tasks) are **included by default** in the locked **36-condition Phase 1a rerun** (3 baselines × 2 sites × 6 modes per `preregistration.md §4` cell inclusion criteria):

- Phase A post-fix code only (commit ≥ `3c15cd7`)
- **A100 self-hosted Docker canonical** (DGX→quark archive pre-2026-05-15 reference only, NO cross-host comparison)
- **3 baselines**: B0 Qwen3-VL-235B-A22B (proxy API) / B1 Qwen3-VL-4B (local) / **B2 Gemma3-VL `google/gemma-3-4b-it`** (local, matched-capability cross-family control, added 2026-05-14)
- Per-cell N inclusion floor ≥ 100 episodes
- All 6 modes per (site, baseline) cell: DOM / SoM / Vision / P-text / P-prompt / P-SoM

### Exclusion (RESTRUCTURED 2026-05-14 — post-hoc FP filter retired in favour of source-level fixes)

Per `preregistration.md §4` row "FP filter architecture" (revised 2026-05-14):
the post-hoc `compute_adjusted_success` layer with `fp_reason ∈ {'', 'na_fp', 'eval_fp'}`
is **retired**. Replaced by source-level fixes at the evaluator + task-load layers:

| Layer | Mechanism | What it does | Audited at |
|---|---|---|---|
| **VWA evaluator (B-91 fix, upstream)** | `llm_fuzzy_match` / `llm_ua_match` empty-prediction guard | Returns 0.0 when prediction empty/whitespace — fixes the GPT-4o-mini-scoring-empty-as-correct root cause that was the dominant FP source. Patch in VWA submodule branch `p79-patches` commit `f0c835b`. | 笔记 §139.8 + master_bug_catalog §139 B-91 |
| **N/A task-load exclusion** | `task.exclude_na_tasks: true` default in `p79/experiment/tasks.py::load_tasks` | 73 N/A tasks (cls 10 / red 5 / shop 31 / wa-shop 19 / wa-admin 6 / wa-red 2 = 5.3% of 1390) excluded at selection time — pre-registered scope decision, NOT post-hoc denominator drop. WONDERBREAD precedent for filtering impossible workflows. | preregistration.md §4 row "N/A task exclusion" |
| **program_html eval_fp branch** | **DROPPED entirely** | `has_effective_action` heuristic had no scalable boundary across WA+6 sites; contamination prevented upstream by `RESET_BEFORE=1` per-cell reset protocol. | 笔记 §139.8 |
| **`visual_fp`** | **RETIRED 2026-05-09** (boundary-undecidable, over-filtered 95.3% VWA tasks) | Replaced by manually-audited non-visual subset (43 VWA + 480 WA = 523 tasks) for Appendix D robustness | `evaluator_change_protocol.md §7` |

### Robustness sensitivity (post-§139.8)

Post-§139.8, **`adjusted_success ≡ success`** — raw `success` from the (fixed)
evaluator is the single primary metric. The prior 3-variant sensitivity ladder
(raw / +na_fp / +na_fp+eval_fp) collapses to one variant. The pre-§139.8 ladder
is retained ONLY for Appendix D pre-§139.8 archived data contamination disclosure
(does NOT apply to canonical post-fix A100 rerun).

`scored_task_count` (cls 224 / red 205 / shop 435 post-N/A-exclusion) replaces
the prior hardcoded `EXPECTED_N` per benchmark. See `memory/reference_fp_architecture_2026-05-14.md`
for the canonical framework.

### Manual non-visual subset (Appendix D robustness)

Replaces deprecated `visual_fp` with **manually-audited 43 VWA + 480 WA = 523
non-visual tasks** (`docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py`).
Used as Appendix D robustness check to confirm paper §1 hero finding holds
on tasks where image is not necessary (rules out "phantom modes work because
target tasks don't need image" trivial alternative).

## Per-site task characteristics

### classifieds (visual-rich)

- Product listings with thumbnails, prices, location, category breadcrumbs
- 234 tasks, N/A rate ~5-8%, image visibility critical for ~60-70% of tasks
- SR pattern (per `section1_intro.md`): full SoM **21.37%** > Phantom-SoM **14.53%** (image-grounded advantage clear)
- Common failure modes: visual hijack (B1 70.4%), early finish on stale page (B0 53.3%)

### reddit (text-dominated)

- Forum threads with text posts, comments, upvote counts; image attachments occasional
- 210 tasks, N/A rate ~10-12%, image visibility critical for ~30-40% of tasks
- SR pattern: Phantom-SoM **13.81%** ≈ full SoM **10.48%** (text-only modes competitive)
- Common failure modes: navigate-to-comments confusion, comment-vs-post target ambiguity

### shopping (mixed)

- Product detail pages, search results, cart pages
- 466 tasks (largest), N/A rate ~6-9%, image visibility critical for ~50% of tasks
- Pilot pending (currently DOM-only B0 cls + B1 cls done; full 5-mode shop in 16-cell rerun)
- Expected SR pattern: similar to classifieds (visual-rich) but more text-rich than classifieds

## Auxiliary images

Some VWA tasks reference images via `[Input image 1]` placeholders in prompts.
These are loaded from `external/visualwebarena/config_files/vwa/<site>/`
and hashed per-image into `env_snapshot.json` `extra.reference_images_sha256`
per run, ensuring byte-identical reference imagery across reruns.

## Data sharing & redaction

The released artifact (per `preregistration.md §7` reproducibility scope) includes:

- **Task config files** (full 910 task pool): MIT license inherited from VWA
- **Curated mirage subset** (`results/mechanistic/archive_subset_b1_{cls,reddit}/`):
  47-48 + 24-15 task subsets with cached observations + `screenshot_annotated.png`
- **Run artifacts** (per-cell episode JSONLs, summaries, env_snapshots): paper-grade

Excluded from release:
- VWA Docker images (replicators pull from VWA repo themselves)
- Site auth state (`.auth/`, gitignored — synthetic seed users only, no real user data)
- B0 proxy API credentials

## Privacy & ethics

VWA tasks are **synthetic**. The Docker containers host:

- Synthetic seed accounts: `emma.lopez` (shopping), `MarvelsGrantMan136` (reddit), `blake.sullivan` (classifieds) — generated by VWA team, no real user data
- Synthetic content: catalog, posts, listings — none reference real people or real-world events
- No PII, no real transactions, no scraped real user data

IRB review **not required** per `pre_run/ethics_license_coi_statements.md`.

## Known issues / caveats

| Issue | Catalog ID | Status |
|---|---|---|
| `ua_match` GPT-4o-mini judge drift | `master_bug_catalog.md` B-20 | DISCLOSED — temperature=0 pin + na_fp class |
| `string_match fuzzy_threshold` only honors 1.0 | B-21 | DISCLOSED — fixed at 1.0 across all conditions |
| VWA site state contamination (cart/listing accumulation) | `experiment_launch_rules.md` | MITIGATED — `RESET_BEFORE=1` protocol per cell |
| Reference image loading bug pre-Phase-A | B-36 | FIXED — Phase A post-fix `3c15cd7+` |

## Per-task ID stability

`(benchmark, site, task_id)` triple is the canonical unique key per
`master_bug_catalog.md` B-80. Cross-benchmark task IDs cannot collide
because benchmark / site is explicit in manifests.

## References

- Gebru et al. 2018 "Datasheets for Datasets" (NEEDS_BIB_ENTRY)
- `paper.bib` `koh2024visualwebarena` (VWA itself)
- `pre_run/locked_versions.md` for VWA SHA + per-site task-pool hashes
- `pre_run/preregistration.md §4` for FP filter + cell inclusion locks
- `master_bug_catalog.md` for known evaluator issues
