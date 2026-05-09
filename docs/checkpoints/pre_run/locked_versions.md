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

### B0 — proxy API Qwen3-Omni-235B-Thinking

| Component | Pin |
|---|---|
| Provider | Internal HolisticAI proxy → upstream Qwen API |
| Model name | `qwen3-omni-235b-thinking` |
| Decoding | `temperature=0`, `max_tokens` per task |
| Stochasticity | Server-side determinism best-effort under `temperature=0` (not strictly guaranteed; see `preregistration.md §7` reproducibility scope) |

API version drift is a known risk — disclosed in `preregistration.md §7`
"verifiable from traces, replayable subject to API access". Test-time
spot-check via `validate_run.py --strict` re-evaluation of N=20 archived
B0 episodes (audit F2 sensitivity) detects upstream model drift.

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

## Evaluator code

| Component | Pin | Hash |
|---|---|---|
| `evaluation_harness/helper_functions.py` | per `evaluator_code.combined_sha256` in `env_snapshot.json` | Recorded automatically per run |
| Adjusted SR computation | `p79.experiment.analysis.compute_adjusted_success` | Pinned at git commit SHA |
| FP filter | `na_fp + eval_fp combined` per `preregistration.md §4` | Code path: `compute_adjusted_success()` returns `fp_reason ∈ {'', 'na_fp', 'eval_fp'}` |

T0/T1/T2/T3 changes governed by `evaluator_change_protocol.md`. Same paper
must dual-report under any post-lock T0 fix per protocol.

## Git lock

| Component | Pin |
|---|---|
| **p79 repo HEAD at lock** | filled at advisor witness time → `preregistration.md` frontmatter `registered_git_sha` |
| Git tag | `paper-grade-rerun-launch-{date}` (TBD when 16-cell launches) |

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
