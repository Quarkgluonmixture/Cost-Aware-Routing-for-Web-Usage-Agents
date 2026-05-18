# Aggregate Compute / Cost / Carbon Table (Phase 1a + Phase 1b)

> **Provenance**: Created /stress A2.9 Chunk 2 — B-1508 P0-4-AB
> (Claude Mode A F3 + codex Mode B F6 OOB) 2026-05-18.
> Pre-fire skeleton; numerical fill aggregated post-Phase-1a-fire via
> `aggregate_phase1_full_prereg_decision.py` new output target.
> Supports paper §8.7 + NeurIPS 2025 Paper Checklist Q12 (Compute Resources).

## Status (2026-05-18 pre-fire)

- ⏳ **Phase 1a Pass-1 baseline**: 36 conditions × 6 cells — TBD post-fire
- ⏳ **Phase 1a Pass-2 learned router**: 6 conditions × 6 cells — TBD post-Pass-1
- ⏳ **Phase 1b shop expansion (main paper)**: +21 conditions × 3 cells — TBD post-workshop submission

## Schema (per-cell row)

| Column | Type | Source / derivation |
|---|---|---|
| `cell_id` | string | "{site}_{model}" e.g., `cls_B0`, `red_B1`, `cls_B2` |
| `host` | enum | A100 / Myriad / DGX |
| `gpu_hours` | float | aggregator: `sum(step_record.wall_clock_sec) / 3600` |
| `total_energy_kwh` | float | `sum(step_record.total_energy_kwh)` via NVIDIAPowerReader pynvml (A100 host) |
| `pue_lower` | float | 1.0 (dock-power only) |
| `pue_upper` | float | 1.5 (university-HPC typical per Strubell 2019) |
| `co2e_kg_lower` | float | `total_energy_kwh × 0.220 kg/kWh × 1.0` |
| `co2e_kg_upper` | float | `total_energy_kwh × 0.220 kg/kWh × 1.5` |
| `cost_basis` | enum | `api_usd` (B0) / `electricity_usd_derived` (B1+B2) |
| `agent_inference_usd` | float | B0: `sum(step_record.cost_usd.model)` (proxy API margin); B1+B2: `total_energy_kwh × electricity_rate` |
| `judge_calls_count` | int | per-episode `judge_calls` log via patched VWA evaluator (B-1509 wires) |
| `judge_tokens_input` | int | aggregate VWA `gpt-4o-mini` input tokens |
| `judge_tokens_output` | int | aggregate VWA `gpt-4o-mini` output tokens |
| `judge_usd` | float | `judge_tokens_input × 0.15/1M + judge_tokens_output × 0.60/1M` (OpenAI 2026-05 pricing) |
| `total_usd_lower` | float | `agent_inference_usd + judge_usd` (using PUE=1.0 for electricity cells) |
| `total_usd_upper` | float | `agent_inference_usd + judge_usd` (using PUE=1.5 for electricity cells) |

## Per-cell table (post-fire fill)

| cell_id | host | gpu_hours | total_energy_kwh | co2e_kg_lower | co2e_kg_upper | cost_basis | agent_usd | judge_usd | total_usd_lower | total_usd_upper |
|---|---|---|---|---|---|---|---|---|---|---|
| cls_B0 | A100 | TBD | TBD | TBD | TBD | api_usd | TBD | TBD | TBD | TBD |
| cls_B1 | A100 | TBD | TBD | TBD | TBD | electricity_usd_derived | TBD | TBD | TBD | TBD |
| cls_B2 | A100 | TBD | TBD | TBD | TBD | electricity_usd_derived | TBD | TBD | TBD | TBD |
| red_B0 | A100 | TBD | TBD | TBD | TBD | api_usd | TBD | TBD | TBD | TBD |
| red_B1 | A100 | TBD | TBD | TBD | TBD | electricity_usd_derived | TBD | TBD | TBD | TBD |
| red_B2 | A100 | TBD | TBD | TBD | TBD | electricity_usd_derived | TBD | TBD | TBD | TBD |
| **Pass-1 Phase 1a total** | A100 | TBD | TBD | TBD | TBD | mixed | TBD | TBD | TBD | TBD |
| **Pass-2 learned router (6 cond)** | A100 | TBD | TBD | TBD | TBD | mixed | TBD | TBD | TBD | TBD |
| **Phase 1a Grand total (42 cond)** | A100 | TBD | TBD | TBD | TBD | mixed | TBD | TBD | TBD | TBD |

## Phase 1b shop expansion (main paper, post-workshop)

| cell_id | host | gpu_hours | total_energy_kwh | co2e_kg_lower | co2e_kg_upper | cost_basis | agent_usd | judge_usd | total_usd_lower | total_usd_upper |
|---|---|---|---|---|---|---|---|---|---|---|
| shop_B0 | A100 | TBD | TBD | TBD | TBD | api_usd | TBD | TBD | TBD | TBD |
| shop_B1 | A100 | TBD | TBD | TBD | TBD | electricity_usd_derived | TBD | TBD | TBD | TBD |
| shop_B2 | A100 | TBD | TBD | TBD | TBD | electricity_usd_derived | TBD | TBD | TBD | TBD |

## Cross-cluster (Myriad cross-arch F6 audit + DGX archive)

| Source | host | gpu_hours | total_energy_kwh | co2e_kg_lower | co2e_kg_upper | rationale |
|---|---|---|---|---|---|---|
| Audit F6 numerical determinism (paper-2 scope) | Myriad | TBD | TBD | TBD | TBD | cross-arch reproducibility check; per advisor 2026-05-14 paper-2 deferred |
| Pre-2026-05-15 archive reference | DGX Spark | TBD | TBD | TBD | TBD | §139.8 FP-architecture sensitivity ladder + Appendix D contamination disclosure; NOT used for canonical paper-grade claims |

## Cross-references

- Paper §8.7 paragraphs covering: cost basis collision warning (B-1505) /
  aggregate compute table framework (B-1508) / LLM judge cost band
  (B-1509) / 3-tier compute fleet breakdown (B-1511) / PUE range
  reporting (B-1510)
- `pre_run/preregistration.md §7` reproducibility scope (per-component tier)
- `pre_run/neurips_checklist.md Q12` (Compute Resources)
- `p79/experiment/energy_tracker.py:48-72` (REGION_INTENSITY_G_PER_KWH UK 220)
- `configs/exp_v2_base.yaml` (carbon_intensity_g_per_kwh: 220 override + cost_api rates)
- `external/visualwebarena/evaluation_harness/helper_functions.py:611-615`
  (VWA_EVAL_MODEL env, GPT-4o-mini judge call site)
- Aggregator entry point: `aggregate_phase1_full_prereg_decision.py` post-fire
  build_full_decision output target (B-1508 wires implementation;
  B-1509 wires judge_usd column; B-1510 wires PUE range column;
  B-1511 wires per-host gpu_hours stratification)

