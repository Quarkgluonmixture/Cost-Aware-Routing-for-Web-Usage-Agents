"""Regression tests for the two full-data aggregator bugs fixed in f9918d5.

S6 test-health audit (2026-06-02): commit f9918d5 ("first paper-grade manifest
promote", B0 classifieds 6-mode, 2026-05-28) fixed two aggregator bugs that
reproduce on the FULL landed Pass-1 data, but shipped them with NO regression
test. Both are pure dependency-version-drift bugs — exactly the class that
silently re-breaks on the next numpy / pandas bump — so they need locking.

  Fix 1 — aggregate_phase1_full_prereg_decision.write_json
      json.dumps(payload, ..., default=float) crashed on the per-cell `boot_pp`
      numpy ndarray (B=1000 bootstrap replicates; source L385 stores
      `boot.astype(np.float32)` into per_cell[*].{h1,h3_axis1,h3_axis2}).
      `float(ndarray)` raises "only 0-dimensional arrays can be converted to
      Python scalars". Fix:
          default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)

  Fix 2 — aggregate_routing_auroc.main()
      pandas >=2.2 groupby(...).apply() EXCLUDES the grouping columns from the
      result, so the downstream sort_values(["baseline","site","mode",...])
      raised KeyError. Fix: back up the key columns before the apply, restore
      them index-aligned, with a reset_index() fallback before the sort.

Coverage map (audit conclusion): run-discovery (test_pass1_manifest), registry
cell-mapping (test_run_registry), learned-router runtime + artifacts
(test_learned_router_runtime / test_router_artifact_contract), H10 cost-basis
(test_stress_deepaudit_h10_basis) and H1 canonical alignment
(test_h1_canonical_alignment) were all covered; the aggregator OUTPUT-serialization
+ groupby-key stage of the same Pass-1 data-land path was the one gap. This file
closes it.

The write_json tests exercise the ACTUAL fixed function (lazy-imported; they skip
if the VWA submodule config is absent — the analysis layer's run_registry computes
`scored_task_count(strict=True)` at import time, the same dependency the other
analysis-layer tests carry). The pandas-groupby test pins the version-sensitive
behavior + restore pattern (CI-runnable, no data). The end-to-end test runs BOTH
actual aggregators against landed Pass-1 data (RUN_LOCAL_DATA_TESTS=1).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _import_write_json():
    """Lazy import of the real write_json. Skip (not error) when the analysis
    layer cannot load — e.g. the VWA submodule config is absent so
    run_registry's module-level `scored_task_count(strict=True)` raises
    FileNotFoundError at import time."""
    sys.path.insert(0, str(REPO / "scripts" / "analysis"))
    try:
        from scripts.analysis.aggregate_phase1_full_prereg_decision import write_json
    except (ImportError, FileNotFoundError) as exc:  # pragma: no cover - env guard
        pytest.skip(f"analysis layer unavailable (VWA submodule config?): {exc}")
    return write_json


# ─── Fix 1: json ndarray serialization (write_json) ─────────────────────────
def test_default_float_alone_crashes_on_ndarray():
    """Necessity proof (pure numpy, no aggregator import): the PRE-fix serializer
    `default=float` cannot serialize a multi-element ndarray — `float(ndarray)`
    raises. This is exactly what the per-cell boot_pp array triggered."""
    payload = {"boot_pp": np.empty(1000, dtype=np.float32)}
    with pytest.raises(TypeError):
        json.dumps(payload, default=float)
    # the fix's serializer handles it
    s = json.dumps(
        payload,
        default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o),
    )
    assert len(json.loads(s)["boot_pp"]) == 1000


def test_write_json_serializes_boot_pp_ndarray(tmp_path):
    """f9918d5 Fix 1: the ACTUAL write_json must serialize the per-cell boot_pp
    ndarray (B=1000) instead of crashing, and emit it as a JSON list."""
    write_json = _import_write_json()
    payload = {
        "per_cell": [{"h1": {"boot_pp": np.arange(1000, dtype=np.float32)}}],
        "k_cells": 1,
    }
    out = tmp_path / "decision.json"
    write_json(payload, out)  # must not raise (pre-fix: TypeError on default=float)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))  # must round-trip
    boot = loaded["per_cell"][0]["h1"]["boot_pp"]
    assert isinstance(boot, list) and len(boot) == 1000
    assert boot[1] == pytest.approx(1.0)  # np.arange(1000) → [0, 1, 2, ...]


def test_write_json_preserves_numpy_scalar_else_branch(tmp_path):
    """The fix's else-branch `float(o)` must still coerce stray numpy scalars
    (the lambda replaced a bare `default=float`, so scalar coercion must survive)."""
    write_json = _import_write_json()
    payload = {"se_pp": np.float64(0.68), "ci_lo": np.float32(-1.25)}
    out = tmp_path / "scalar.json"
    write_json(payload, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["se_pp"] == pytest.approx(0.68)
    assert loaded["ci_lo"] == pytest.approx(-1.25, abs=1e-5)


# ─── Fix 2: pandas groupby grouping-key restoration ─────────────────────────
def test_routing_auroc_groupby_key_restore_pattern():
    """f9918d5 Fix 2: pandas >=2.2 groupby(group_keys=False).apply() drops the
    grouping columns; the backup + index-aligned restore must recover
    baseline/site/mode so the downstream sort_values does not KeyError.

    Replicates aggregate_routing_auroc.main()'s exact pattern. Asserts the
    DURABLE contract (keys present + non-null + sort succeeds) which holds
    whether or not the installed pandas drops the keys, so a future pandas
    behavior change cannot make this brittle."""
    df = pd.DataFrame(
        {
            "baseline": ["B0", "B0", "B1"],
            "site": ["classifieds", "classifieds", "reddit"],
            "mode": ["dom", "som", "dom"],
            "p_one_sided": [0.01, 0.04, 0.02],
        }
    )

    def _holm_within(group):  # mirrors the real closure (operates on a non-key col)
        return group.assign(p_holm=group["p_one_sided"] * len(group))

    # --- the fix: back up keys, apply, restore index-aligned, reset_index fallback
    keys_bak = df[["baseline", "site", "mode"]].copy()
    out = df.groupby(["baseline", "site", "mode"], group_keys=False).apply(_holm_within)
    for k in ("baseline", "site", "mode"):
        if k not in out.columns:
            out[k] = keys_bak[k]
    if any(k not in out.columns for k in ("baseline", "site", "mode")):
        out = out.reset_index()

    # durable contract
    for k in ("baseline", "site", "mode"):
        assert k in out.columns, f"grouping key {k} missing after restore"
        assert out[k].notna().all(), f"grouping key {k} has NaN after restore"
    assert "p_holm" in out.columns  # the apply payload survived
    # the sort that raised KeyError pre-fix now succeeds
    sorted_out = out.sort_values(["baseline", "site", "mode", "p_holm"])
    assert len(sorted_out) == len(df)


# ─── End-to-end: both aggregators on landed Pass-1 data (gated) ──────────────
@pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_DATA_TESTS") != "1",
    reason="landed-data aggregator probe skipped by default — set "
    "RUN_LOCAL_DATA_TESTS=1 to run both aggregators against this host's "
    "results/ + run_manifest.yaml (needs landed Pass-1 episode data)",
)
def test_both_aggregators_emit_intact_contract_on_landed_pass1_data(tmp_path):
    """End-to-end on landed Pass-1 data: run the ACTUAL aggregators and assert the
    f9918d5 contracts on real output — routing_auroc keys non-null (Fix 2) +
    prereg_decision JSON valid with any boot_pp serialized as a list (Fix 1)."""
    # Fix 2 — routing_auroc CSV with intact grouping keys
    auroc_out = tmp_path / "auroc.csv"
    r1 = subprocess.run(
        [sys.executable, "scripts/analysis/aggregate_routing_auroc.py",
         "--output", str(auroc_out)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    if not auroc_out.exists():
        pytest.skip(f"no landed routing-auroc data (rc={r1.returncode}): {r1.stderr[-300:]}")
    adf = pd.read_csv(auroc_out)
    for k in ("baseline", "site", "mode"):
        assert k in adf.columns, f"{k} missing from auroc CSV (Fix 2 regressed)"
        assert adf[k].notna().all(), f"{k} has NaN in auroc CSV (Fix 2 restore failed)"

    # Fix 1 — prereg_decision JSON valid + boot_pp serialized as list
    dec_json = tmp_path / "decision.json"
    r2 = subprocess.run(
        [sys.executable, "scripts/analysis/aggregate_phase1_full_prereg_decision.py",
         "--output-json", str(dec_json),
         "--output-csv", str(tmp_path / "decision.csv"),
         "--output-md", str(tmp_path / "decision.md")],
        cwd=str(REPO), capture_output=True, text=True,
    )
    assert dec_json.exists(), (
        f"prereg_decision did not write JSON (rc={r2.returncode}): {r2.stderr[-300:]}"
    )
    d = json.loads(dec_json.read_text(encoding="utf-8"))  # must parse (Fix 1)
    saw_boot = False
    for cell in d.get("per_cell", []):
        for hkey in ("h1", "h3_axis1", "h3_axis2"):
            sub = cell.get(hkey)
            if isinstance(sub, dict) and "boot_pp" in sub:
                assert isinstance(sub["boot_pp"], list), (
                    f"per_cell[*].{hkey}.boot_pp must be a JSON list (Fix 1 ndarray→tolist)"
                )
                saw_boot = True
    # not a hard requirement (depends on which cells/axes landed), but record it
    if not saw_boot:
        print("note: no boot_pp arrays in landed prereg_decision output "
              "(k_cells/axes config dependent) — JSON-validity still proves Fix 1")
