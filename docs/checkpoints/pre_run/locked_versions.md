# Locked Versions Manifest

> Pinned versions of every external dependency that affects experimental
> outcomes. Recorded at the latest paper-grade lock point (2026-05-09)
> as part of audit constraint **A5** (freeze benchmark/task pools and
> dataset versions) and **F8** (temporal validity / software drift).
>
> Future paper-grade reruns must verify these pins still match (`make
> verify-version-locks` — to be wired into `make pre-launch-check`,
> audit C10).

## VisualWebArena

| Component | Pin | Source |
|---|---|---|
| **VWA submodule SHA** | **`832f037e2cc7ebda4a41831443a3fc9b79d06cd6`** | `git -C external/visualwebarena rev-parse HEAD` (locked 2026-05-09) |
| Repo URL | https://github.com/web-arena-x/visualwebarena | Submodule URL in `.gitmodules` |
| WebArena upstream | (not used in this paper, scoped out per audit F3 / preregistration §7) | — |

### Task pool hashes per site

Hash = `find <site>/*.json | sort | xargs sha256sum | sha256sum`. Verifies
that the task config files have not drifted since the paper's lock point.

| Site | N tasks | Task-pool sha256 (first 16 hex) |
|---|---|---|
| classifieds (`test_classifieds`) | **234** | `d36a20c1eaa1f5da...` |
| reddit (`test_reddit`) | **210** | `ecd4ed4370740fd6...` |
| shopping (`test_shopping`) | **466** | `07889e3646ee10e3...` |

Total VWA tasks in scope: **910 tasks** across 3 sites.

### Reference image hashes

VWA tasks reference auxiliary images (e.g., `[Input image 1]` for
visual-question tasks) loaded from `external/visualwebarena/config_files/vwa/`.
Per-image sha256 hashes are recorded by `scripts/provenance/snapshot_env.py`
into `env_snapshot.json` `extra.reference_images_sha256` per run, ensuring
that paper-grade reruns use byte-identical reference imagery.

## Browser substrate

| Component | Pin | How verified |
|---|---|---|
| **Playwright** | **1.58.0** | `pip show playwright` (locked 2026-05-09) |
| **Chromium** | **revision 1208** (`~/.cache/ms-playwright/chromium-1208/`) | bundled by Playwright 1.58.0; pinned by Playwright version |
| Browser policy | `--ignore-certificate-errors --disable-web-security` per VWA Docker convention | Per `scripts/vwa_env_remote.sh` (gitignored) + Docker compose files |

Browser version is pinned **transitively** through Playwright pin: Playwright
1.58.0 ships Chromium revision 1208. Upgrading Playwright forces a Chromium
upgrade, which is treated as a T0 evaluator-relevant change per
`evaluator_change_protocol.md`.

## Model substrate

### B1 — local Qwen3-VL-4B (mechanistic + B1 baseline)

| Component | Pin |
|---|---|
| **HF revision SHA** | **`ebb281ec70b05090aa6165b016eac8ec08e71b17`** |
| HF model name | `Qwen/Qwen3-VL-4B-Instruct` |
| Dtype | `torch.bfloat16` |
| Decoding | greedy (`do_sample=False`), `max_new_tokens` per task / 50 for Stage 2 |
| Seed | 42 (`configs/exp_v2_base.yaml`) |

Recorded automatically per run via `snapshot_env.py` → `env_snapshot.json`
`extra.hf_model_revision_pinned`.

### B0 — proxy API Qwen3-VL-235B-A22B

| Component | Pin |
|---|---|
| Provider | Internal HolisticAI proxy → upstream Qwen API |
| Model name | `qwen3-vl-235b-a22b` (proxy `api_name`: `qwen.qwen3-vl-235b-a22b`) |
| Decoding | `temperature=0`, `max_tokens` per task |
| Stochasticity | Server-side determinism best-effort under `temperature=0` (not strictly guaranteed; see `preregistration.md §7` reproducibility scope) |

API version drift is a known risk — disclosed in `preregistration.md §7`
"verifiable from traces, replayable subject to API access". Test-time
spot-check via `validate_run.py --strict` re-evaluation of N=20 archived
B0 episodes (audit F2 sensitivity) detects upstream model drift.

### B2 — local Gemma3-VL `google/gemma-3-4b-it` (added 2026-05-14, matched-capability cross-family control vs B1)

| Component | Pin |
|---|---|
| **HF revision SHA** | ⏳ **pending lock** (matching B1 protocol — pin at A100 bring-up time, recorded into env_snapshot.json) |
| HF model name | `google/gemma-3-4b-it` |
| Dtype | `torch.bfloat16` (fits A100-PCIE-40GB unquantized) |
| Decoding | greedy (`do_sample=False`), `max_new_tokens` per task |
| Seed | 42 (`configs/exp_v2_base.yaml` — shared with B1) |
| Capability rationale | 4B params parity with B1 (4B Qwen3-VL); cross-family control (Google Gemma vs Alibaba Qwen lineage) at matched scale; advisor discussion 2026-05-14 (see `preregistration.md` Appendix A 2026-05-14 entry + 笔记 §138 + §142) |

Recorded automatically per run via `snapshot_env.py` → `env_snapshot.json`
`extra.hf_model_revision_pinned`. HF SHA lock pending #11 A100 VM VWA
Docker bring-up + first B2 smoke run.

## Python / library substrate

Per `pyproject.toml` + `.venv/bin/pip show <pkg>`:

| Library | Pin / version | Notes |
|---|---|---|
| Python | 3.12 (DGX) / 3.9.6 (Myriad) / 3.11 (Condense, future) | Multi-environment, behavior parity verified per audit F6 |
| torch | 2.11.0+cu128 (DGX) / 2.1.0 (Myriad module) | `sitecustomize.py` shim for Myriad's torch 2.1 missing `register_pytree_node` + `torch.compiler.is_compiling` (B-81b, B-81f) |
| transformers | 4.57.6 | Required for Qwen3-VL support (4.55 missing it; 4.49 lacks Qwen3-VL entirely) |
| numpy | <2 | torch 2.1 binary compiled against NumPy 1.x (Myriad B-81 stack) |
| urllib3 | <2 | RHEL 7 OpenSSL 1.0.2k incompatible with v2 (Myriad B-81c) |
| scipy | 1.17.1 (DGX) | Stat tests (Holm, Welch, Wilcoxon) |
| matplotlib | (per pyproject) | Figure rendering |

Full transitive lock recorded in `env_snapshot.json` `libraries` field per run.

## Evaluator code (RESTRUCTURED 2026-05-14 — post-hoc adjusted_success layer retired in favour of source-level fixes; see preregistration.md §4 "FP filter architecture" row + Appendix A 2026-05-14 entry)

| Component | Pin | Hash |
|---|---|---|
| `evaluation_harness/helper_functions.py` | per `evaluator_code.combined_sha256` in `env_snapshot.json` (VWA submodule branch `p79-patches` commit `f0c835b` post-B-91 fix) | Recorded automatically per run |
| **Primary metric** | **raw `success` from evaluator** (canonical, single metric post-§139.8) | `adjusted_success ≡ success` post-fix; legacy `compute_adjusted_success` retained ONLY for Appendix D pre-§139.8 archive contamination disclosure (paper-grade rerun does not use it) |
| **Source-level FP fixes** | (B-91) `llm_fuzzy_match` / `llm_ua_match` evaluator-level empty-prediction guard | VWA submodule `p79-patches` branch commit `f0c835b` adds `if not pred or not pred.strip(): return 0.0` upstream guard. Closes na_fp at the evaluator boundary. |
| **N/A task exclusion** | At task-load time (`task.exclude_na_tasks: true`, `p79/experiment/tasks.py::load_tasks`) | 73 N/A tasks excluded across VWA+WA (5.3% of 1390) — pre-registered scope decision, NOT post-hoc denominator drop |
| **`scored_task_count`** | Replaces hardcoded `EXPECTED_N` per benchmark / site (cls 224 / red 205 / shop 435 post-N/A-exclusion) | Computed at run time from in-scope task pool |
| **program_html eval_fp branch** | **DROPPED** entirely (no scalable boundary across WA+6 sites; contamination prevented upstream by `RESET_BEFORE` protocol) | `has_effective_action` heuristic removed |
| **`visual_fp`** | **RETIRED** earlier (2026-05-09, boundary-undecidable, over-filtered 95.3% VWA tasks) | Replaced by manually-audited non-visual subset (43 VWA + 480 WA = 523 tasks) for Appendix D robustness |

**Pre-§139.8 history (retained for Appendix D contamination disclosure only)**: The post-hoc `compute_adjusted_success` FP filter returned `fp_reason ∈ {'', 'na_fp', 'eval_fp'}`; 3-variant sensitivity ladder (raw / +na_fp / +na_fp+eval_fp) was the paper-grade reporting protocol. Retired 2026-05-14 in favor of the source-level fixes above — see `memory/reference_fp_architecture_2026-05-14.md` (canonical) + 笔记 §139.8 + master_bug_catalog §139 B-83~B-91.

T0/T1/T2/T3 changes governed by `evaluator_change_protocol.md`. Same paper
must dual-report under any post-lock T0 fix per protocol. The B-91 source-level
fix itself is a T0 change applied **pre-lock** (paper-grade rerun has not happened
yet under §139.8 architecture); locking happens after the A100 rerun completes.

## Hardware / Host substrate (added 2026-05-15)

| Component | Pin |
|---|---|
| **Canonical paper-grade rerun host** | **A100 (Condenser VM `a100-jiaming-test` @ 10.134.51.2, A100-PCIE-40GB)** with self-hosted VWA Docker stack on A100 (no Tailscale tunnel) |
| Pre-2026-05-15 archive host (reference only) | DGX Spark (`spark-9ea3` aarch64 GB10) with VWA Docker on remote (quark Windows home) via Tailscale tunnel |
| Migration date | 2026-05-15 (see `preregistration.md` Appendix A 2026-05-15 entry; #11 A100 VM VWA Docker bring-up gates canonical launch per `phase1_plan.md §B0`) |
| Cross-host comparison policy | **NOT made** — archive vs canonical-run on different hardware/network stacks; archive retained for §139.8 FP sensitivity + Appendix D contamination disclosure only |
| CUDA toolkit | A100 PCIE 40GB host — pending CUDA pin at bring-up |
| Docker runtime | Self-hosted on A100 (no Tailscale dependency); image SHA + container fingerprint snapshot pending |

## Git lock

| Component | Pin |
|---|---|
| **p79 repo HEAD at lock** | filled at advisor witness time → `preregistration.md` frontmatter `registered_git_sha` |
| Git tag | `paper-grade-rerun-launch-{date}` (TBD when 36-condition Phase 1a launches on A100) |

## Verification command (planned, audit C10)

```bash
make verify-version-locks
```

This will:
1. Check `git -C external/visualwebarena rev-parse HEAD` matches `832f037e2cc7ebda4a41831443a3fc9b79d06cd6`
2. Hash each per-site task pool and compare to entries above
3. Check `pip show playwright` is `1.58.0`
4. Check `~/.cache/ms-playwright/chromium-1208/` exists
5. Check HF model snapshot exists at the pinned revision SHA
6. Exit non-zero on any mismatch with diagnostic output

To be wired into `make pre-launch-check` per audit C10 and B7 stopping rule (a).

## Update protocol

When a pin changes (intentional upgrade or unavoidable drift):
1. Document the change in `master_bug_catalog.md` with severity
2. Decide T0/T1/T2/T3 classification per `evaluator_change_protocol.md`
3. Update this file with new pin + reason in changelog below
4. Re-run smoke test gate (B7) to verify no behavior drift
5. If T0+ and post-lock: dual-report in paper appendix per protocol

## Changelog

- **2026-05-09**: Initial lock at VWA `832f037e`, Playwright 1.58.0, Chromium 1208,
  HF Qwen3-VL-4B `ebb281ec70b0...`, transformers 4.57.6 + Myriad shims (B-81 umbrella).
- **2026-05-14**: B2 = Gemma3-VL `google/gemma-3-4b-it` added as 3rd baseline (matched-capability
  cross-family control vs B1 4B). HF SHA pin pending A100 bring-up; preregistration §4 cell scope
  expanded from 24 cond / 4 cells → 36 cond / 6 cells.
- **2026-05-15**: Canonical paper-grade rerun host migrated DGX Spark → A100 Condenser VM
  `a100-jiaming-test` (A100-PCIE-40GB, self-hosted VWA Docker). DGX→quark Tailscale stack retained
  for pre-2026-05-15 archive reference only. New Hardware/Host substrate section added above. #11
  A100 VM VWA Docker bring-up gates canonical launch.
