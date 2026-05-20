"""Invariant tests for /stress A1.6b cold-start fixes (B-650 ~ B-661).

Cross-AI 3-AI cycle (Claude self / codex Mode B / gemini Mode C) on
`p79/experiment/analysis.py` lines 900-2012 (Pareto + decision-test +
analyze_run body half).

12 fixes:
- B-650: heatmap cmap `set_bad("#cccccc")` so N/A → gray (matches caption)
- B-651: Holm-Bonferroni family key is (test, metric) per locked prereg §3
  (cell-scoping reverted Q1=A /stress 2026-05-20; cell_key is transparency-only)
- B-652: episode + step gate symmetric with B-601 condition gate
- B-653: TOST + SR-Wilcoxon + wilcoxon_skipped.csv emitted
- B-654: per-site SR double-column (observed + scored_set)
- B-655: paired bootstrap CI on SR lift emitted
- B-656: per-site separate heatmap (no multi-site task_id collision)
- B-657: cross-baseline grouped bar (no B0/B1/B2 mean mixing)
- B-658: per-condition rng SeedSequence (order-independent reproducibility)
- B-659: partial-cell skip in `_analyze_condition` (stub session_summary)
- B-660: Pareto preserves tied points (val <= best_min + last_max tie branch)
- B-661: phase2 fixed_best/routed iloc[0] non-uniqueness fail-loud
"""
from __future__ import annotations

import json
import pytest


# ─── B-660 ──────────────────────────────────────────────────────────────────
def test_b660_pareto_preserves_true_ties():
    """When 2 points share BOTH maximize + minimize → both on frontier."""
    from p79.experiment.analysis import _compute_pareto_front

    # A and B exact tie on (sr=0.6, cost=$0.1). C dominated (sr=0.5, cost=$0.1).
    points = [
        {"success_rate": 0.6, "avg_total_cost_usd": 0.1},  # A
        {"success_rate": 0.6, "avg_total_cost_usd": 0.1},  # B (tied with A)
        {"success_rate": 0.5, "avg_total_cost_usd": 0.1},  # C (dominated)
    ]
    indices = _compute_pareto_front(
        points, maximize="success_rate", minimize="avg_total_cost_usd",
    )
    # A + B both on frontier, C dropped.
    assert set(indices) == {0, 1}, f"expected A+B on frontier, got {indices}"


def test_b660_pareto_drops_dominated_same_cost():
    """Same cost but lower SR → dominated, must be dropped."""
    from p79.experiment.analysis import _compute_pareto_front

    points = [
        {"success_rate": 0.7, "avg_total_cost_usd": 0.1},  # dominator
        {"success_rate": 0.5, "avg_total_cost_usd": 0.1},  # dominated (same cost, lower SR)
    ]
    indices = _compute_pareto_front(
        points, maximize="success_rate", minimize="avg_total_cost_usd",
    )
    assert indices == [0], f"expected only A on frontier, got {indices}"


def test_b660_pareto_strict_improvement_path():
    """Classic strict-< sweep should still work for non-tied points."""
    from p79.experiment.analysis import _compute_pareto_front

    points = [
        {"success_rate": 0.7, "avg_total_cost_usd": 0.3},  # high SR, high cost
        {"success_rate": 0.5, "avg_total_cost_usd": 0.1},  # mid SR, low cost — frontier
        {"success_rate": 0.6, "avg_total_cost_usd": 0.2},  # mid SR, mid cost — frontier
    ]
    indices = _compute_pareto_front(
        points, maximize="success_rate", minimize="avg_total_cost_usd",
    )
    # All 3 are mutually non-dominated → all on frontier.
    assert set(indices) == {0, 1, 2}, f"expected all 3 on frontier, got {indices}"


# ─── B-661 ──────────────────────────────────────────────────────────────────
def test_b661_phase2_fixed_best_non_unique_raises(tmp_path):
    """`phase2_fixed_best` matching multiple rows must raise paper-grade fail-loud."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    import pandas as pd
    from p79.experiment.analysis import _plot_phase2

    # Two `phase2_fixed_best` rows (e.g., multi-site contamination).
    cond_df = pd.DataFrame([
        {
            "condition_id": "phase2_fixed_best", "success_rate": 0.6,
            "avg_total_cost_usd": 0.1, "avg_total_model_cost_usd": 0.08,
            "avg_router_overhead_cost_usd": 0.02,
        },
        {
            "condition_id": "phase2_fixed_best", "success_rate": 0.7,
            "avg_total_cost_usd": 0.15, "avg_total_model_cost_usd": 0.12,
            "avg_router_overhead_cost_usd": 0.03,
        },
        {
            "condition_id": "phase2_routed", "success_rate": 0.65,
            "avg_total_cost_usd": 0.09, "avg_total_model_cost_usd": 0.07,
            "avg_router_overhead_cost_usd": 0.02,
        },
    ])
    plots_dir = tmp_path / "plots"
    tables_dir = tmp_path / "tables"
    reports_dir = tmp_path / "reports"
    for d in (plots_dir, tables_dir, reports_dir):
        d.mkdir(parents=True)

    with pytest.raises(ValueError, match=r"B-661 phase2_fixed_best matched 2 rows"):
        _plot_phase2(cond_df, plots_dir, tables_dir, reports_dir)


# ─── B-658 ──────────────────────────────────────────────────────────────────
def test_b658_per_condition_rng_order_independent():
    """Bootstrap CI numbers must not depend on cond_ids glob order."""
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    import pandas as pd
    from pathlib import Path
    from p79.experiment.analysis import _compute_statistical_tests

    # Build ep_df with 2 conditions, then run twice with reversed cond_df order.
    ep_rows = []
    for cid, sr in [("phase1_dom_router_0", [1, 1, 0, 1, 0]),
                    ("phase1_som_router_0", [0, 1, 0, 0, 1])]:
        for task_id, s in enumerate(sr):
            ep_rows.append({
                "condition_id": cid, "task_id": task_id,
                "benchmark_site": "classifieds", "benchmark": "visualwebarena",
                "success": bool(s), "total_cost_usd": 0.01, "p95_step_latency_ms": 100.0,
            })
    ep_df = pd.DataFrame(ep_rows)

    def _ci_for(order):
        cond_df = pd.DataFrame([{"condition_id": cid} for cid in order])
        out_dir = Path("/tmp") / f"a1_6b_b658_{'_'.join(order)}"
        reports_dir = out_dir / "reports"
        tables_dir = out_dir / "tables"
        for d in (reports_dir, tables_dir):
            d.mkdir(parents=True, exist_ok=True)
        _compute_statistical_tests(cond_df, ep_df, reports_dir, tables_dir)
        with open(reports_dir / "statistical_tests.json") as f:
            res = json.load(f)
        return res["bootstrap_ci"]

    ci_fwd = _ci_for(["phase1_dom_router_0", "phase1_som_router_0"])
    ci_rev = _ci_for(["phase1_som_router_0", "phase1_dom_router_0"])
    # Per-condition rng → same CI regardless of cond_ids order.
    for cid in ("phase1_dom_router_0", "phase1_som_router_0"):
        assert abs(ci_fwd[cid]["ci_lower_95"] - ci_rev[cid]["ci_lower_95"]) < 1e-9, (
            f"B-658 violated: CI lower for {cid} differs across cond_ids order "
            f"(forward {ci_fwd[cid]['ci_lower_95']:.6f} vs reverse {ci_rev[cid]['ci_lower_95']:.6f})"
        )
        assert abs(ci_fwd[cid]["ci_upper_95"] - ci_rev[cid]["ci_upper_95"]) < 1e-9


# ─── B-654 ──────────────────────────────────────────────────────────────────
def test_b654_per_site_emits_scored_set_denominator(tmp_path):
    """per_site_metrics.csv must carry success_rate_scored_set + scored_set_n."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    import pandas as pd
    from p79.experiment.analysis import _analyze_per_site

    # Mock ep_df with 2 conditions × 2 sites.
    ep_rows = []
    for cid in ("phase1_dom_router_0", "phase1_som_router_0"):
        for site in ("classifieds", "reddit"):
            for task_id in range(5):
                ep_rows.append({
                    "condition_id": cid, "benchmark_site": site,
                    "benchmark": "visualwebarena", "task_id": task_id,
                    "success": (task_id % 2 == 0), "steps": 5,
                    "total_cost_usd": 0.01, "total_energy_kwh": 0.001,
                })
    ep_df = pd.DataFrame(ep_rows)

    plots_dir = tmp_path / "plots"
    tables_dir = tmp_path / "tables"
    for d in (plots_dir, tables_dir):
        d.mkdir(parents=True)

    _analyze_per_site(ep_df, plots_dir, tables_dir)

    csv_path = tables_dir / "per_site_metrics.csv"
    assert csv_path.exists(), "B-654: per_site_metrics.csv not emitted"
    df = pd.read_csv(csv_path)
    # Required columns from B-654 disclosure.
    for col in ("success_rate_observed", "success_rate_scored_set",
                "n_episodes_observed", "scored_set_n", "n_success",
                "estimand_note"):
        assert col in df.columns, f"B-654 missing column: {col}"
    # B-1599 (/stress A2.10 P2-8-A 2026-05-18): replace hardcoded N=5 with
    # computed expectation from ep_df fixture, so fixture changes don't
    # silently break this test. observed denominator = N tasks per
    # (cond_id × site); success_rate_observed = (task_id % 2 == 0).mean().
    for _, row in df.iterrows():
        cond_rows = ep_df[
            (ep_df["condition_id"] == row["condition_id"])
            & (ep_df["benchmark_site"] == row["benchmark_site"])
        ]
        expected_n = len(cond_rows)
        expected_sr = float(cond_rows["success"].mean())
        assert row["n_episodes_observed"] == expected_n, (
            f"expected n_episodes_observed={expected_n}, got {row['n_episodes_observed']}"
        )
        assert abs(row["success_rate_observed"] - expected_sr) < 1e-9, (
            f"expected SR={expected_sr}, got {row['success_rate_observed']}"
        )


# ─── B-650 ──────────────────────────────────────────────────────────────────
def test_b650_heatmap_uses_masked_array_and_set_bad():
    """Source-level check: `cmap.set_bad(` + `masked_invalid` invocations exist."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] /
           "p79" / "experiment" / "analysis.py").read_text()
    assert "cmap.set_bad(" in src, "B-650: cmap.set_bad missing"
    assert "masked_invalid(" in src, "B-650: np.ma.masked_invalid missing"
    assert "B-650" in src, "B-650: stamp comment missing"


# ─── B-659 ──────────────────────────────────────────────────────────────────
def test_b659_partial_cell_skip_emits_stub(tmp_path):
    """`_analyze_condition` skipped for `_synthesized=True` row; stub written."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    from p79.experiment.analysis import analyze_run

    # Derive the episode summary from EPISODE_SUMMARY_V2_DEFAULTS so it carries
    # every paper-grade key (incl. total_latency_minus_retry_ms, which
    # aggregate_condition_metrics requires present) and won't drift on future
    # schema additions.
    from conftest import complete_episode_summary

    cond = tmp_path / "phase1_dom_router_0"
    eps = cond / "episodes"
    eps.mkdir(parents=True)
    # NO condition_summary_v2.json (so condition is synthesized from meta + ep).
    (cond / "condition_meta.json").write_text(json.dumps({
        "condition_id": "phase1_dom_router_0",
        "seed": 42, "phase": "phase1", "backend_id": "local_qwen3vl_4b",
        "som_on": False, "observation_mode": "dom", "router_on": False,
        "modules": {},
    }))
    (eps / "1_summary_v2.json").write_text(json.dumps(complete_episode_summary(**{
        "schema_version": "2.0", "run_id": "r1",
        "condition_id": "phase1_dom_router_0",
        "benchmark": "vwa", "benchmark_site": "classifieds",
        "task_id": 1, "seed": 42, "success": True, "score": 1.0,
        "steps": 1, "retries": 0, "no_op_rate": 0.0, "page_unchanged_rate": 0.0,
        "total_latency_ms": 100.0, "p95_step_latency_ms": 100.0,
        "total_tokens": 0, "total_model_cost_usd": 0.0, "total_cost_usd": 0.0,
        "total_router_overhead_cost_usd": 0.0, "total_router_overhead_ms": 0.0,
        "total_energy_kwh": None, "total_co2e_kg": None,
        "escalation_count": 0, "trigger_distribution": {},
        "benchmark_noise": False, "benchmark_noise_category": None,
        "artifacts_dir": "",
    })))
    (eps / "1_steps_v2.jsonl").write_text('{"step_idx": 0, "x": 1}\n')

    analyze_run(str(tmp_path))
    stub_path = tmp_path / "analysis" / "results" / "phase1_dom_router_0" / "session_summary.json"
    assert stub_path.exists(), "B-659: partial-cell stub session_summary.json not emitted"
    stub = json.loads(stub_path.read_text())
    assert stub["partial"] is True, "B-659: stub.partial must be True"
    assert "B-659" in stub["skip_reason"], "B-659: stub.skip_reason should cite B-659"


# ─── B-653 (post-B-1051 TOST retire 2026-05-18) ───────────────────────────
def test_b653_tost_retired_paired_bootstrap_still_emitted(tmp_path):
    """B-1051 (/stress A2.3c Mode B P0-1-B*): TOST framework RETIRED per B-957;
    statistical_tests.json must NOT carry tost_equivalence section. Paired
    bootstrap lift (B-655) still emitted unaffected by TOST retirement."""
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    import pandas as pd
    from p79.experiment.analysis import _compute_statistical_tests

    ep_rows = []
    for cid, sr in [("phase1_dom_router_0", [1, 1, 0, 1, 0, 1, 0, 0]),
                    ("phase1_som_router_0", [0, 1, 0, 0, 1, 0, 1, 1])]:
        for task_id, s in enumerate(sr):
            ep_rows.append({
                "condition_id": cid, "task_id": task_id,
                "benchmark_site": "classifieds", "benchmark": "visualwebarena",
                "backend_id": "local_qwen3vl_4b",
                "success": bool(s), "total_cost_usd": 0.01,
                "p95_step_latency_ms": 100.0,
            })
    ep_df = pd.DataFrame(ep_rows)
    cond_df = pd.DataFrame([
        {"condition_id": "phase1_dom_router_0"},
        {"condition_id": "phase1_som_router_0"},
    ])

    reports_dir = tmp_path / "reports"
    tables_dir = tmp_path / "tables"
    for d in (reports_dir, tables_dir):
        d.mkdir(parents=True)

    _compute_statistical_tests(cond_df, ep_df, reports_dir, tables_dir)

    with open(reports_dir / "statistical_tests.json") as f:
        res = json.load(f)
    # Post-B-1051 contract: TOST section MUST NOT exist (retired per B-957).
    assert "tost_equivalence" not in res, (
        "B-1051: tost_equivalence section MUST be absent post-B-957 TOST retire"
    )
    # B-655 paired bootstrap lift still emitted (unaffected by TOST retirement).
    assert "bootstrap_paired_lift" in res, "B-655: bootstrap_paired_lift section missing"
    pair_key = "phase1_dom_router_0_vs_phase1_som_router_0"
    lift_entry = res["bootstrap_paired_lift"][pair_key]
    assert lift_entry["estimand"] == "paired_same_task_lift_b_minus_a"
    assert lift_entry["n_paired"] == 8


# ─── B-651 ──────────────────────────────────────────────────────────────────
def test_b651_holm_family_test_metric_not_cell_scoped(tmp_path):
    """Holm family key is (test, metric) — NOT cell-scoped.

    B-651 cell-scoping was REVERTED (Q1=A /stress 2026-05-20): the DOI-1-locked
    preregistration §3 family declaration + paper section3_definition.md:132
    define the Holm family as `(test, metric)` ("avoids over-correcting tests
    that probe distinct estimands"). A short-lived `(test, metric, cell_key)`
    grouping (commit ac925a1) contradicted that registered family + paper prose
    and was walked back. `cell_key` survives as a per-row transparency stratum
    label but must NOT enter the holm_family key. This test guards against
    re-introducing cell-scoping. See /stress B-651 walk-back 2026-05-20.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    import pandas as pd
    from p79.experiment.analysis import _compute_statistical_tests

    # 3 conds same site, same baseline → all pairs share one (site,model) stratum.
    ep_rows = []
    for cid in ("phase1_dom_router_0", "phase1_som_router_0", "phase1_vision_router_0"):
        for task_id in range(8):
            ep_rows.append({
                "condition_id": cid, "task_id": task_id,
                "benchmark_site": "classifieds", "benchmark": "visualwebarena",
                "backend_id": "local_qwen3vl_4b",
                "success": (task_id + hash(cid)) % 3 != 0,
                "total_cost_usd": 0.01, "p95_step_latency_ms": 100.0,
            })
    ep_df = pd.DataFrame(ep_rows)
    cond_df = pd.DataFrame([{"condition_id": c} for c in (
        "phase1_dom_router_0", "phase1_som_router_0", "phase1_vision_router_0"
    )])

    reports_dir = tmp_path / "reports"
    tables_dir = tmp_path / "tables"
    for d in (reports_dir, tables_dir):
        d.mkdir(parents=True)

    _compute_statistical_tests(cond_df, ep_df, reports_dir, tables_dir)

    csv_df = pd.read_csv(tables_dir / "statistical_tests.csv")
    # cell_key survives as a transparency stratum-label column (intra_<site>_<baseline>
    # for same-site same-baseline comparisons) — but is NOT the Holm family key.
    assert "cell_key" in csv_df.columns, "B-651: cell_key transparency column missing"
    intra = csv_df[csv_df["cell_key"].astype(str).str.startswith("intra_")]
    assert len(intra) > 0, "B-651: no intra-cell stratum-label entries found"
    # holm_family must be the bare (test, metric) form per locked prereg §3 —
    # it must NOT encode cell_key (no intra_/crossbaseline_/crosssite_ tokens).
    holm_families = set(csv_df["holm_family"].dropna().astype(str))
    assert holm_families, "B-651: no holm_family values emitted"
    expected_forms = {
        "mcnemar_exact_success",
        "wilcoxon_signed_rank_total_cost_usd",
        "wilcoxon_signed_rank_p95_step_latency_ms",
    }
    assert holm_families <= expected_forms, (
        f"B-651: holm_family must be (test, metric) form per prereg §3; "
        f"got unexpected families={holm_families - expected_forms}"
    )
    assert not any(
        tok in f for f in holm_families
        for tok in ("intra_", "crossbaseline_", "crosssite_", "unknown")
    ), f"B-651: holm_family must NOT be cell-scoped (reverted Q1=A); got {holm_families}"
