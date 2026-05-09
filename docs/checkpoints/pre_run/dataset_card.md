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
| **classifieds** | 234 | `d36a20c1eaa1f5da...` | Visual-rich product listings (Magento-based) | DGX Tailscale → quark Docker container `:9980` (cf. `memory/MEMORY.md` "Docker 容器端口") |
| **reddit** | 210 | `ecd4ed4370740fd6...` | Text-dominated forum threads (Postmill) | DGX Tailscale → quark `:9999` |
| **shopping** | 466 | `07889e3646ee10e3...` | Mixed text + image product pages (Magento) | DGX Tailscale → quark `:7770` |
| **Total** | **910** | — | | |

WebArena (480 tasks across shopping/shopping_admin/reddit) is **out of scope**
for this paper per `preregistration.md §7` external validity scope. Cross-bench
generalization is explicit future work.

## Task curation methodology

### Inclusion

All 910 VWA tasks are **included by default** in the locked 16-cell rerun
(per `preregistration.md §4` cell inclusion criteria):

- Phase A post-fix code only (commit ≥ `3c15cd7`)
- Per-cell N inclusion floor ≥ 100 episodes
- All locked modes per cell (DOM / SoM / Vision / phantom_som / phantom_dom
  for B0/B1, plus phantom_text / phantom_prompt for select cells)

### Exclusion (post-FP-filter)

Per `preregistration.md §4` FP filter primary spec (`na_fp + eval_fp` combined),
per-task exclusions are:

| Filter | What it excludes | Rationale | Audited at |
|---|---|---|---|
| `na_fp` | Tasks marked N/A by ground truth where agent emitted `finish.answer="X"` (non-N/A) and was scored success by mistake | Mirage prevention; agent overconfidence on impossible tasks should not be rewarded | 笔记 §78a |
| `eval_fp` | Tasks where evaluator's program_html / string_match scoring is logged as ambiguous-pass (e.g., URL prefix match where agent navigated past target) | Evaluator-reliability defense | 笔记 §95 |
| `visual_fp` | **DEPRECATED** — boundary-undecidable, over-filters 95.3% VWA tasks | Removed per evaluator change protocol T2 simplification | `evaluator_change_protocol.md §7` |

### Robustness sensitivity

Per `preregistration.md §4` FP filter sensitivity ladder, **3 variants reported**
in paper appendix:

1. `raw_SR` (no FP filter)
2. `+na_fp only` (just N/A exclusion)
3. `+na_fp+eval_fp combined` (primary)

H1/H3 conclusions must hold under all 3 variants (paper §3 sensitivity disclosure).

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
