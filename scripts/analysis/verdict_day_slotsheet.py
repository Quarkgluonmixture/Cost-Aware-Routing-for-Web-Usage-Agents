#!/usr/bin/env python3
"""Fail-closed verdict-day formatter over canonical analysis artifacts.

Final mode emits copyable slots/tables only when every required artifact is
present, non-empty, exact, and mutually consistent.  ``--rehearsal`` emits a
diagnostic sheet headed ``INVALID_FOR_DRAFT``; it never emits copyable slot or
table blocks.  ``--h10-pending`` is a COMPLETE-only final mode for the interval
between Pass-1 and Pass-2: H1/H3 slots and Tables 2/3 remain copyable, while all
H10/router values fail closed behind explicit pending markers.  Verdict logic
reads ``analysis_status`` and ``h1_verdict``; legacy ``gate_status`` is
display-only fallback metadata.  ``--protocol-note-06`` is a separately tagged,
fixed-five authorization with the same Pass-1-only open surface; it never accepts
PARTIAL or ordinary COMPLETE artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DECISION = ROOT / "results/phantom_paper/phase1_full_prereg_decision.json"
H10 = ROOT / "results/phantom_paper/h10_pareto_verdict.json"
SR = ROOT / "docs/analysis/cross_sites/sr_per_mode.json"
FIG0C = ROOT / "results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv"
ROUTER = ROOT / "results/phantom_paper/l1_router/covariate_baseline.json"

MODE_ORDER = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]
PLANNED_CELL_IDS = {
    f"{baseline}_{site}"
    for baseline in ("B0", "B1", "B2")
    for site in ("classifieds", "reddit")
}
CAPTURE_WINDOW_SECONDS = 24 * 60 * 60
H10_PENDING_NOTICE = (
    "H10 PENDING (Pass-2 not landed; deployability fail-closed per prereg)"
)
H10_PENDING_ABSTRACT = (
    "Learned-router deployability remains pending until the preregistered "
    "Pass-2 evaluation is complete."
)
PROTOCOL_NOTE_06_STATUS = "COMPLETE_K5_PROTOCOL_NOTE_06"
PROTOCOL_NOTE_06_QUALIFIER = "on the five landed cells"
PROTOCOL_NOTE_06_WITNESS_TAG = (
    "protocol-note-06-k5-early-verdict-signed-20260715"
)
PROTOCOL_NOTE_06_CELL_IDS = {
    "B0_classifieds",
    "B1_classifieds",
    "B2_classifieds",
    "B0_reddit",
    "B1_reddit",
}


def g(d: Any, *keys: str, default: Any = "MISSING") -> Any:
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def decimal_format(value: Any, places: int = 2, *, signed: bool = True) -> str:
    """Format with the locked Decimal ROUND_HALF_UP rule."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "MISSING"
    try:
        number = Decimal(str(value))
        quantum = Decimal(1).scaleb(-places)
        rounded = number.quantize(quantum, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return "MISSING"
    prefix = "+" if signed and rounded >= 0 else ""
    return f"{prefix}{rounded:.{places}f}"


def scalars(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if not isinstance(v, (dict, list))}


def scalar_display(value: Any, *, places: int = 4) -> str:
    """Display arbitrary producer scalars without bypassing the rounding lock."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (float, Decimal)):
        return decimal_format(value, places, signed=False)
    return str(value)


def _cell_id(baseline: Any, site: Any) -> str:
    return f"{baseline}_{site}"


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_number(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> bool:
    """Type-strict finite numeric validation (``bool`` is never numeric)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    if minimum is not None and number < minimum:
        return False
    if maximum is not None and number > maximum:
        return False
    return not integer or number.is_integer()


def _is_csv_number(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> bool:
    if isinstance(value, bool) or value in (None, ""):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(number):
        return False
    if minimum is not None and number < minimum:
        return False
    if maximum is not None and number > maximum:
        return False
    return not integer or number.is_integer()


def _is_required_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _load_json(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        errors.append(f"missing or empty {label}: {path}")
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label}: {exc}")
        return {}
    if not isinstance(value, dict) or not value:
        errors.append(f"{label} must be a non-empty JSON object")
        return {}
    return value


def _load_csv(path: Path, label: str, errors: list[str]) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        errors.append(f"missing or empty {label}: {path}")
        return []
    try:
        rows = list(csv.DictReader(path.open()))
    except (OSError, csv.Error) as exc:
        errors.append(f"invalid {label}: {exc}")
        return []
    if not rows:
        errors.append(f"{label} has an empty table")
    return rows


def validate_artifacts(
    dec: dict[str, Any],
    h10: dict[str, Any],
    sr: dict[str, Any],
    fig0c: list[dict[str, str]],
    router: dict[str, Any],
    *,
    final: bool,
    h10_pending: bool = False,
    protocol_note_06: bool = False,
) -> tuple[list[str], list[str]]:
    """Return hard validation errors and explicitly disclosed provenance gaps."""
    errors: list[str] = []
    gaps: list[str] = []
    status = dec.get("analysis_status")
    pass1_only = h10_pending or protocol_note_06
    expected_cell_ids = (
        PROTOCOL_NOTE_06_CELL_IDS if protocol_note_06 else PLANNED_CELL_IDS
    )
    if protocol_note_06:
        if status != PROTOCOL_NOTE_06_STATUS:
            errors.append(
                "--protocol-note-06 requires decision artifact "
                f"analysis_status={PROTOCOL_NOTE_06_STATUS}, got {status!r}"
            )
        if dec.get("protocol_note") != "PROTOCOL_NOTE_06":
            errors.append("--protocol-note-06 requires protocol_note=PROTOCOL_NOTE_06")
        if dec.get("verdict_qualifier") != PROTOCOL_NOTE_06_QUALIFIER:
            errors.append(
                "--protocol-note-06 requires verdict_qualifier="
                f"{PROTOCOL_NOTE_06_QUALIFIER!r}"
            )
        if dec.get("b1284_one_tier_downgrade") is not True:
            errors.append("--protocol-note-06 requires b1284_one_tier_downgrade=true")
        if dec.get("r_tier_cap") != "R2":
            errors.append("--protocol-note-06 requires r_tier_cap=R2")
        if dec.get("witness_tag") != PROTOCOL_NOTE_06_WITNESS_TAG:
            errors.append("--protocol-note-06 witness_tag is missing or unauthorized")
        fixed_cell_set = dec.get("fixed_cell_set")
        if (
            not isinstance(fixed_cell_set, list)
            or len(fixed_cell_set) != len(PROTOCOL_NOTE_06_CELL_IDS)
            or set(fixed_cell_set) != PROTOCOL_NOTE_06_CELL_IDS
        ):
            errors.append("--protocol-note-06 fixed_cell_set is not the authorized five")
    elif status not in {"COMPLETE", "PARTIAL", "INSUFFICIENT"}:
        errors.append(f"decision analysis_status invalid or missing: {status!r}")
    if pass1_only:
        if status != "COMPLETE":
            if not protocol_note_06:
                errors.append(
                    "--h10-pending requires decision artifact "
                    f"analysis_status=COMPLETE, got {status!r}; it cannot bypass completeness"
                )
        if dec.get("h1_verdict") not in {"PASS", "FAIL"}:
            errors.append(
                f"{'--protocol-note-06' if protocol_note_06 else '--h10-pending'} "
                "requires decision artifact h1_verdict in {PASS, FAIL}"
            )
    elif final and status != "COMPLETE":
        errors.append(f"final slotsheet requires analysis_status=COMPLETE, got {status!r}")
    if final and not h10_pending and dec.get("h1_verdict") not in {"PASS", "FAIL"}:
        errors.append("final slotsheet requires h1_verdict in {PASS, FAIL}")
    if final:
        required_numeric_paths = (
            (("pooled_h1_fe", "theta_FE_pp"), -100.0, 100.0, False),
            (("pooled_h1_bootstrap", "ci95_lo_pp_bootstrap"), -100.0, 100.0, False),
            (("pooled_h1_bootstrap", "ci95_hi_pp_bootstrap"), -100.0, 100.0, False),
            (("pooled_h1_bootstrap", "p_one_sided_bootstrap"), 0.0, 1.0, False),
            (("pooled_h1_bootstrap", "k_cells"), 1.0, 6.0, True),
            (("h1_heterogeneity", "I_squared_pct"), 0.0, 100.0, False),
            (("h3_axis1_pooled_fe", "theta_FE_pp"), -100.0, 100.0, False),
            (("h3_axis1_pooled_fe", "ci95_lo_pp_bootstrap"), -100.0, 100.0, False),
            (("h3_axis1_pooled_fe", "ci95_hi_pp_bootstrap"), -100.0, 100.0, False),
            (("h3_axis2_pooled_fe", "theta_FE_pp"), -100.0, 100.0, False),
            (("h3_axis2_pooled_fe", "ci95_lo_pp_bootstrap"), -100.0, 100.0, False),
            (("h3_axis2_pooled_fe", "ci95_hi_pp_bootstrap"), -100.0, 100.0, False),
        )
        for path, minimum, maximum, integer in required_numeric_paths:
            value = g(dec, *path, default=None)
            if not _is_number(
                value, minimum=minimum, maximum=maximum, integer=integer
            ):
                errors.append(
                    "decision required numeric field invalid: "
                    f"{'.'.join(path)}={value!r}"
                )
        if _parse_time(dec.get("captured_at")) is None:
            errors.append("decision required captured_at timestamp missing or invalid")
        if protocol_note_06 and g(dec, "pooled_h1_bootstrap", "k_cells", default=None) != 5:
            errors.append("--protocol-note-06 requires pooled_h1_bootstrap.k_cells=5")
        if protocol_note_06:
            for axis_key in ("h3_axis1_pooled_fe", "h3_axis2_pooled_fe"):
                verdict = g(dec, axis_key, "axis_verdict", default=None)
                if verdict not in {"PASS", "FAIL"}:
                    errors.append(
                        f"--protocol-note-06 requires {axis_key}.axis_verdict in "
                        "{PASS, FAIL}"
                    )

    decision_cells: dict[str, dict[str, Any]] = {}
    per_cell = dec.get("per_cell")
    if not isinstance(per_cell, list) or not per_cell:
        errors.append("decision per_cell is missing or empty")
    else:
        for cell in per_cell:
            cid = _cell_id(cell.get("baseline"), cell.get("site"))
            if cid in decision_cells:
                errors.append(f"decision duplicate cell: {cid}")
            decision_cells[cid] = cell
            h1 = cell.get("h1", {})
            if h1.get("complete_exact") is not True:
                errors.append(f"decision {cid} h1.complete_exact is not true")
            if not isinstance(h1.get("observed_n"), dict) or set(h1["observed_n"]) != set(MODE_ORDER):
                errors.append(f"decision {cid} does not carry six unique canonical modes")
            if final and not _is_required_string(h1.get("task_set_sha256")):
                errors.append(f"decision {cid} required task_set_sha256 missing")
        if final and set(decision_cells) != expected_cell_ids:
            errors.append(
                "decision cells are not the exact authorized scope: "
                f"missing={sorted(expected_cell_ids - set(decision_cells))} "
                f"extra={sorted(set(decision_cells) - expected_cell_ids)}"
            )

    sr_rows = sr.get("summary_table")
    if not isinstance(sr_rows, list) or not sr_rows:
        errors.append("SR summary_table is missing or empty")
        sr_rows = []
    sr_keys: set[tuple[str, str, str]] = set()
    if final and _parse_time(sr.get("captured_at")) is None:
        errors.append("SR required captured_at timestamp missing or invalid")
    for row in sr_rows:
        key = (str(row.get("baseline")), str(row.get("site")), str(row.get("mode")))
        if key in sr_keys:
            errors.append(f"SR duplicate row: {key}")
        sr_keys.add(key)
        exact_value = row.get("complete_exact")
        if exact_value is not True and (final or row.get("complete") is not True):
            errors.append(f"SR non-exact row: {key}")
        cid = _cell_id(row.get("baseline"), row.get("site"))
        expected_sha = g(decision_cells.get(cid, {}), "h1", "task_set_sha256", default=None)
        row_sha = row.get("task_set_sha256")
        if final and not _is_required_string(row_sha):
            errors.append(f"SR required task_set_sha256 missing for {key}")
        if expected_sha is not None and row_sha != expected_sha:
            errors.append(f"task_set_sha256 mismatch decision↔SR for {key}")
        if final:
            sr_numeric_fields = (
                ("n_total", 0.0, None, True),
                ("observed_n", 0.0, None, True),
                ("expected_n", 1.0, None, True),
                ("n_success", 0.0, None, True),
                ("sr_denominator_n", 1.0, None, True),
                ("sr_pct", 0.0, 100.0, False),
                ("completeness_ratio", 0.0, 1.0, False),
            )
            for field, minimum, maximum, integer in sr_numeric_fields:
                value = row.get(field)
                if not _is_number(
                    value, minimum=minimum, maximum=maximum, integer=integer
                ):
                    errors.append(f"SR invalid numeric field for {key}: {field}={value!r}")
    expected_sr_keys = {
        (cid.split("_", 1)[0], cid.split("_", 1)[1], mode)
        for cid in expected_cell_ids for mode in MODE_ORDER
    }
    if final and sr_keys != expected_sr_keys:
        errors.append(
            f"SR table must contain exact {len(expected_sr_keys)} authorized rows; "
            f"got {len(sr_keys)} unique rows"
        )

    scoped_fig = [
        row for row in fig0c
        if _cell_id(row.get("baseline"), row.get("site")) in expected_cell_ids
    ]
    numeric_fig = [r for r in scoped_fig if r.get("row_type", "numeric") == "numeric"]
    error_fig = [r for r in scoped_fig if r.get("row_type") == "panel_error"]
    if error_fig:
        errors.append(f"fig0c contains {len(error_fig)} panel-level error rows")
    fig_by_cell: dict[str, list[dict[str, str]]] = {}
    for row in numeric_fig:
        cid = _cell_id(row.get("baseline"), row.get("site"))
        fig_by_cell.setdefault(cid, []).append(row)
    for cid, rows in fig_by_cell.items():
        modes = [r.get("mode") for r in rows]
        if set(modes) != set(MODE_ORDER) or len(modes) != len(set(modes)):
            errors.append(f"fig0c {cid} is not exact six unique modes")
        if any(str(r.get("complete_exact")).lower() != "true" for r in rows):
            errors.append(f"fig0c {cid} contains non-exact numeric rows")
        # The ordinary six-panel fig0c producer globally labels every row
        # NON_PAPER_GRADE when the absent B2 Reddit panel forces --allow-partial.
        # NOTE_06 authorizes only the five named panels, so their row-level
        # complete_exact/is_partial/task-SHA checks below are the binding grade
        # contract.  Every existing mode continues to reject the global label.
        if not protocol_note_06 and any(
            r.get("grade") != "PAPER_GRADE" for r in rows
        ):
            errors.append(f"fig0c {cid} contains NON_PAPER_GRADE rows")
        if any(str(r.get("is_partial")).lower() == "true" for r in rows):
            errors.append(f"fig0c {cid} contains partial numeric rows")
        for row in rows:
            if final:
                fig_numeric_fields = (
                    ("drop_one_loss_pp", -100.0, 100.0, False),
                    ("ci95_low_pp", -100.0, 100.0, False),
                    ("ci95_high_pp", -100.0, 100.0, False),
                    ("n_common", 1.0, None, True),
                    ("n_expected", 1.0, None, True),
                    ("n_modes_unique", 6.0, 6.0, True),
                )
                for field, minimum, maximum, integer in fig_numeric_fields:
                    value = row.get(field)
                    if not _is_csv_number(
                        value, minimum=minimum, maximum=maximum, integer=integer
                    ):
                        errors.append(
                            f"fig0c {cid} invalid numeric field: {field}={value!r}"
                        )
                if not _is_required_string(row.get("task_set_sha256")):
                    errors.append(f"fig0c {cid} required task_set_sha256 missing")
                if _parse_time(row.get("captured_at")) is None:
                    errors.append(f"fig0c {cid} required captured_at missing or invalid")
            try:
                portfolio = json.loads(row.get("portfolio_modes", "[]"))
                n_modes_unique = int(row.get("n_modes_unique", "0"))
                n_common = int(row.get("n_common", "0"))
                n_expected = int(row.get("n_expected", "0"))
            except (json.JSONDecodeError, TypeError, ValueError):
                errors.append(f"fig0c {cid} has malformed completeness metadata")
                continue
            if set(portfolio) != set(MODE_ORDER) or n_modes_unique != len(MODE_ORDER):
                errors.append(f"fig0c {cid} portfolio metadata is not exact six modes")
            if n_common != n_expected:
                errors.append(f"fig0c {cid} n_common={n_common} != n_expected={n_expected}")
        shas = {r.get("task_set_sha256") for r in rows}
        if len(shas) != 1:
            errors.append(f"fig0c {cid} has inconsistent task_set_sha256 values")
        expected_sha = g(decision_cells.get(cid, {}), "h1", "task_set_sha256", default=None)
        if expected_sha is not None and shas != {expected_sha}:
            errors.append(f"task_set_sha256 mismatch decision↔fig0c for {cid}")
        psom_rows = [row for row in rows if row.get("mode") == "P-SoM"]
        if len(psom_rows) == 1 and cid in decision_cells:
            h1 = decision_cells[cid].get("h1", {})
            comparable = (
                ("drop_one_loss_pp", "theta_pp"),
                ("ci95_low_pp", "ci95_lo_pp"),
                ("ci95_high_pp", "ci95_hi_pp"),
            )
            for fig_field, decision_field in comparable:
                try:
                    fig_value = float(psom_rows[0][fig_field])
                    decision_value = float(h1[decision_field])
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        f"decision↔fig0c numeric join field missing for {cid}: "
                        f"{fig_field}/{decision_field}"
                    )
                    continue
                # fig0c sidecar stores four decimal places.
                if abs(fig_value - decision_value) > 1e-4:
                    errors.append(
                        f"decision↔fig0c numeric mismatch for {cid}: "
                        f"{fig_field}={fig_value} vs {decision_field}={decision_value}"
                    )
    if final and set(fig_by_cell) != expected_cell_ids:
        errors.append(
            "fig0c numeric panels must equal the authorized scope; "
            f"got {sorted(fig_by_cell)}"
        )

    if not pass1_only:
        h10_cells = h10.get("per_cell")
        if not isinstance(h10_cells, dict) or not h10_cells:
            errors.append("H10 per_cell is missing or empty")
        elif final and set(h10_cells) != PLANNED_CELL_IDS:
            errors.append("H10 per_cell must contain exact planned six")
        elif final and any(
            not isinstance(value, dict) or not value for value in h10_cells.values()
        ):
            errors.append("H10 per_cell contains an empty/non-object table row")
        if final and (
            not isinstance(h10.get("operational_deployment_gate"), dict)
            or not h10.get("operational_deployment_gate")
        ):
            errors.append("H10 operational_deployment_gate is missing")
        if final and _parse_time(h10.get("captured_at")) is None:
            errors.append("H10 required captured_at timestamp missing or invalid")
        if isinstance(h10_cells, dict):
            for cid, cell in h10_cells.items():
                h10_sha = cell.get("task_set_sha256") if isinstance(cell, dict) else None
                expected_sha = g(
                    decision_cells.get(cid, {}), "h1", "task_set_sha256",
                    default=None,
                )
                if final and not _is_required_string(h10_sha):
                    errors.append(f"H10 {cid} required task_set_sha256 missing")
                if expected_sha is not None and h10_sha != expected_sha:
                    errors.append(f"task_set_sha256 mismatch decision↔H10 for {cid}")

        router_cells = set(str(c) for c in router.get("cells", []))
        if not router:
            errors.append("router artifact is missing or empty")
        else:
            if final and router.get("grade") != "PAPER_GRADE":
                errors.append(f"router grade is not PAPER_GRADE: {router.get('grade')!r}")
            if final and router.get("analysis_status") != "COMPLETE":
                errors.append(
                    f"router analysis_status is not COMPLETE: {router.get('analysis_status')!r}"
                )
            if final and router_cells != PLANNED_CELL_IDS:
                errors.append("router cells must contain exact planned six")
            if final and _parse_time(router.get("captured_at")) is None:
                errors.append("router required captured_at timestamp missing or invalid")
            contrasts = router.get("paired_contrasts")
            if not isinstance(contrasts, list) or not contrasts:
                errors.append("router paired_contrasts is missing or empty")
            elif final:
                expected_contrast_ids = {
                    "full-vs-scalar:standard",
                    "full-vs-scalar:template_disjoint",
                    "standard-vs-template-disjoint:full_lr",
                }
                keys = {
                    (str(record.get("cell_id")), str(record.get("contrast_id")))
                    for record in contrasts if isinstance(record, dict)
                }
                expected_keys = {
                    (cell_id, contrast_id)
                    for cell_id in PLANNED_CELL_IDS
                    for contrast_id in expected_contrast_ids
                }
                if keys != expected_keys or len(contrasts) != len(expected_keys):
                    errors.append("router paired_contrasts must contain exact 18 predefined rows")
                for record in contrasts:
                    if not isinstance(record, dict):
                        errors.append("router paired_contrasts contains a non-object row")
                        continue
                    rid = f"{record.get('cell_id')}/{record.get('contrast_id')}"
                    if not _is_number(
                        record.get("delta_auroc"), minimum=-1.0, maximum=1.0
                    ):
                        errors.append(f"router {rid} invalid delta_auroc")
                    ci = record.get("ci95")
                    if (
                        not isinstance(ci, list)
                        or len(ci) != 2
                        or any(not _is_number(v, minimum=-1.0, maximum=1.0) for v in ci)
                    ):
                        errors.append(f"router {rid} invalid ci95")
                    if not _is_number(record.get("n_common"), minimum=1.0, integer=True):
                        errors.append(f"router {rid} invalid n_common")
            if not isinstance(router.get("results"), list) or not router.get("results"):
                errors.append("router results table is missing or empty")
            router_validation = g(router, "canonical_input_validation", "cells", default={})
            if isinstance(router_validation, dict):
                for cid, validation in router_validation.items():
                    router_sha = (
                        validation.get("task_set_sha256")
                        if isinstance(validation, dict) else None
                    )
                    expected_sha = g(
                        decision_cells.get(cid, {}), "h1", "task_set_sha256",
                        default=None,
                    )
                    if final and not _is_required_string(router_sha):
                        errors.append(f"router {cid} required task_set_sha256 missing")
                    if expected_sha is not None and router_sha != expected_sha:
                        errors.append(f"task_set_sha256 mismatch decision↔router for {cid}")
            elif final:
                errors.append("router canonical_input_validation.cells is missing")

    dec_time = _parse_time(dec.get("captured_at"))
    fig_times = {_parse_time(r.get("captured_at")) for r in numeric_fig}
    fig_times.discard(None)
    provenance_times = [("fig0c", fig_times)]
    if not pass1_only:
        router_time = _parse_time(router.get("captured_at"))
        provenance_times.append(("router", {router_time} if router_time else set()))
    for label, times in provenance_times:
        if dec_time is not None and times:
            for other in times:
                if abs((other - dec_time).total_seconds()) > CAPTURE_WINDOW_SECONDS:
                    errors.append(f"captured_at window exceeded decision↔{label} (>24h)")
        else:
            gaps.append(f"decision↔{label}: captured_at unavailable on one side")

    # Final mode treats the required timestamps/SHA joins above as hard schema
    # errors. Rehearsal retains gap-shaped diagnostics for older artifacts.
    if not sr.get("captured_at") and not final:
        gaps.append("SR: captured_at absent; time-window join cannot be enforced")
    if not pass1_only:
        if not h10.get("captured_at") and not final:
            gaps.append("H10: captured_at absent; time-window join cannot be enforced")
        h10_cells = h10.get("per_cell")
        if isinstance(h10_cells, dict) and not final:
            for cid, cell in h10_cells.items():
                if not isinstance(cell, dict) or not cell.get("task_set_sha256"):
                    gaps.append(
                        f"H10 {cid}: task_set_sha256 absent; "
                        "task-universe join cannot be enforced"
                    )
    return errors, sorted(set(gaps))


def _router_slot_id(record: dict[str, Any]) -> str:
    raw = f"ROUTER_{record.get('cell_id')}_{record.get('contrast_id')}"
    return re.sub(r"[^A-Z0-9]+", "_", raw.upper()).strip("_")


def build_sheet(
    dec: dict[str, Any],
    h10: dict[str, Any],
    sr: dict[str, Any],
    fig0c: list[dict[str, str]],
    router: dict[str, Any],
    *,
    rehearsal: bool,
    h10_pending: bool = False,
    protocol_note_06: bool = False,
    errors: list[str],
    gaps: list[str],
) -> str:
    lines: list[str] = []
    add = lines.append
    pass1_only = h10_pending or protocol_note_06
    expected_cell_ids = (
        PROTOCOL_NOTE_06_CELL_IDS if protocol_note_06 else PLANNED_CELL_IDS
    )
    qualifier_suffix = (
        f" — {PROTOCOL_NOTE_06_QUALIFIER}" if protocol_note_06 else ""
    )
    if rehearsal:
        add("# INVALID_FOR_DRAFT — verdict-day rehearsal diagnostics")
    elif protocol_note_06:
        add(
            "# PROTOCOL_NOTE_06 k=5 verdict-day slot sheet "
            f"(captured_at={g(dec, 'captured_at')})"
        )
    else:
        add(f"# Verdict-day slot sheet (captured_at={g(dec, 'captured_at')})")
    add("")
    add("> Rounding lock: decimal.Decimal + ROUND_HALF_UP. Final mode is the only copyable source.")
    if pass1_only:
        add(f"> **{H10_PENDING_NOTICE}**")
        add("> Copyable scope in this sheet: H1/H3 slots and Tables 2/3 only.")
    if protocol_note_06:
        add(
            "> Authorized temporary verdict **on the five landed cells**; "
            "B-1284 automatic one-tier downgrade; R-tier cap **R2**."
        )
    add("")

    analysis_status = dec.get("analysis_status")
    h1_verdict = dec.get("h1_verdict", "NOT_EVALUATED")
    legacy_gate = dec.get("gate_status", "MISSING")
    display_status = analysis_status if analysis_status is not None else f"LEGACY_DISPLAY_ONLY:{legacy_gate}"
    add("## A. Gate status (producer fields)")
    add(f"- analysis_status=`{display_status}`")
    add(f"- h1_verdict=`{h1_verdict}`")
    add(f"- legacy gate_status=`{legacy_gate}` (display/fallback only; never branch logic)")
    if protocol_note_06:
        add("- protocol_note=`PROTOCOL_NOTE_06`")
        add(f"- verdict_qualifier=`{PROTOCOL_NOTE_06_QUALIFIER}`")
        add("- B-1284 one-tier downgrade=`true`; r_tier_cap=`R2`")
        add(f"- fixed_cell_set=`{sorted(PROTOCOL_NOTE_06_CELL_IDS)}`")
        add(f"- witness_tag=`{dec.get('witness_tag', 'MISSING')}`")
    boot = g(dec, "pooled_h1_bootstrap", default={})
    fe = g(dec, "pooled_h1_fe", default={})
    add(
        "- H1: "
        f"theta_FE={decimal_format(g(fe, 'theta_FE_pp'))}pp; "
        f"CI95=[{decimal_format(g(boot, 'ci95_lo_pp_bootstrap'))}, "
        f"{decimal_format(g(boot, 'ci95_hi_pp_bootstrap'))}]pp; "
        f"k={g(boot, 'k_cells')}{qualifier_suffix}"
    )
    heterogeneity = g(dec, "h1_heterogeneity", default={})
    add(
        "- H1 heterogeneity: "
        f"I²={scalar_display(g(heterogeneity, 'I_squared_pct'))}% | "
        f"cap_at_R3={g(heterogeneity, 'heterogeneity_cap_at_r3')}"
    )
    for axis_key in ("h3_axis1_pooled_fe", "h3_axis2_pooled_fe"):
        axis = g(dec, axis_key, default={})
        add(
            f"- {axis_key}: "
            + " | ".join(f"{key}={scalar_display(value)}" for key, value in scalars(axis).items())
            + qualifier_suffix
        )
    add("- H2(a) per-cell:")
    for cell in dec.get("per_cell", []):
        h2 = cell.get("h2a", {})
        add(
            f"  - {cell.get('baseline')}·{cell.get('site')}: "
            + " | ".join(f"{key}={scalar_display(value)}" for key, value in scalars(h2).items())
        )
    if pass1_only:
        add(f"- H10 operational gate: **{H10_PENDING_NOTICE}**")
    else:
        operational = g(h10, "operational_deployment_gate", default={})
        add(
            "- H10 operational gate: "
            + " | ".join(
                f"{key}={scalar_display(value)}"
                for key, value in scalars(operational).items()
            )
        )
    if errors:
        add("- validation diagnostics:")
        for error in errors:
            add(f"  - {error}")
    add("")

    add("## B. Branch suggestion")
    complete_for_mode = (
        analysis_status == PROTOCOL_NOTE_06_STATUS
        if protocol_note_06 else analysis_status == "COMPLETE"
    )
    if rehearsal or not complete_for_mode:
        add("- **NO_BRANCH** — non-COMPLETE/rehearsal artifacts cannot select or splice a branch.")
    elif h1_verdict == "PASS":
        add(f"- **Branch A** (h1_verdict=PASS{qualifier_suffix}).")
    elif h1_verdict == "FAIL":
        ax1 = g(dec, "h3_axis1_pooled_fe", "passed")
        ax2 = g(dec, "h3_axis2_pooled_fe", "passed")
        add(
            f"- H1 FAIL{qualifier_suffix}; H3 axis1_pass={ax1}{qualifier_suffix}, "
            f"axis2_pass={ax2}{qualifier_suffix}."
        )
        if protocol_note_06 and ax1 is True and ax2 is True:
            add(
                f"- **Branch B** — both H3 axes pass {PROTOCOL_NOTE_06_QUALIFIER}."
            )
        else:
            add("- Both H3 axes pass → Branch B; otherwise use the Amendment-02 ladder.")
    else:
        add("- **NO_BRANCH** — canonical h1_verdict is not evaluated.")
    add("")

    add("## Provenance validation")
    if gaps:
        add("### 剩余 provenance gap")
        for gap in gaps:
            add(f"- {gap}")
    else:
        add("- All currently comparable provenance fields joined successfully.")
    add("")

    if rehearsal:
        add("## C'. Router covariate diagnostics (REHEARSAL — NON-COPYABLE, 禁止进 draft)")
        add(
            "> ⚠ Diagnostic visibility only. This warning-shaped table is not a "
            "canonical slot source and must not be copied into the draft."
        )
        add(
            "- input artifact markers: "
            f"grade=`{router.get('grade', 'MISSING')}`; "
            f"analysis_status=`{router.get('analysis_status', 'MISSING')}`; "
            f"captured_at=`{router.get('captured_at', 'MISSING')}`"
        )
        add("| ⚠ diagnostic only | cell | contrast | delta_AUROC | CI95 | n_common |")
        add("|---|---|---|---:|---:|---:|")
        contrasts = router.get("paired_contrasts", [])
        if isinstance(contrasts, list):
            for contrast in contrasts:
                if not isinstance(contrast, dict):
                    continue
                ci = contrast.get("ci95")
                ci_text = (
                    f"[{decimal_format(ci[0], 3)}, {decimal_format(ci[1], 3)}]"
                    if isinstance(ci, list) and len(ci) == 2 else "undefined"
                )
                add(
                    f"| ⚠ NON-COPYABLE | {contrast.get('cell_id')} | "
                    f"{contrast.get('contrast_id')} | "
                    f"{decimal_format(contrast.get('delta_auroc'), 3)} | "
                    f"{ci_text} | {contrast.get('n_common')} |"
                )
        add("")

    if rehearsal or not complete_for_mode:
        add("Copyable §C–§F slots/tables intentionally suppressed.")
        return "\n".join(lines) + "\n"

    add("## C. Canonical slot values")
    add("| slot | value | producer field |")
    add("|---|---:|---|")
    if protocol_note_06:
        add(
            f"| H1_VERDICT | {h1_verdict}{qualifier_suffix} | h1_verdict |"
        )
    add(
        f"| THETA | {decimal_format(g(fe, 'theta_FE_pp'))}{qualifier_suffix} | "
        "pooled_h1_fe.theta_FE_pp |"
    )
    add(
        f"| CI_LO / CI_HI | {decimal_format(g(boot, 'ci95_lo_pp_bootstrap'))} / "
        f"{decimal_format(g(boot, 'ci95_hi_pp_bootstrap'))}{qualifier_suffix} | "
        "pooled_h1_bootstrap percentile CI |"
    )
    add(f"| P_BOOT | {decimal_format(g(boot, 'p_one_sided_bootstrap'), 4, signed=False)}{qualifier_suffix} | pooled_h1_bootstrap.p_one_sided_bootstrap |")
    add(f"| K | {g(boot, 'k_cells')}{qualifier_suffix} | pooled_h1_bootstrap.k_cells |")
    for axis_name, axis_key in (("AX1", "h3_axis1_pooled_fe"), ("AX2", "h3_axis2_pooled_fe")):
        axis = g(dec, axis_key, default={})
        if protocol_note_06:
            add(
                f"| {axis_name}_VERDICT | {g(axis, 'axis_verdict')}{qualifier_suffix} | "
                f"{axis_key}.axis_verdict |"
            )
        add(
            f"| {axis_name} | {decimal_format(g(axis, 'theta_FE_pp'))} "
            f"[{decimal_format(g(axis, 'ci95_lo_pp_bootstrap'))}, "
            f"{decimal_format(g(axis, 'ci95_hi_pp_bootstrap'))}]{qualifier_suffix} | {axis_key} |"
        )
    if pass1_only:
        add("")
        add("### H10/router slots")
        add(f"- **{H10_PENDING_NOTICE}**")
    else:
        for contrast in router.get("paired_contrasts", []):
            slot = _router_slot_id(contrast)
            ci = contrast.get("ci95")
            ci_text = (
                f"[{decimal_format(ci[0], 3)}, {decimal_format(ci[1], 3)}]"
                if isinstance(ci, list) and len(ci) == 2 else "undefined"
            )
            add(
                f"| {slot} | ΔAUROC={decimal_format(contrast.get('delta_auroc'), 3)}; "
                f"CI95={ci_text}; n={contrast.get('n_common')} | "
                f"paired_contrasts[{contrast.get('contrast_id')}] |"
            )
    add("")

    add("## D. Table 2 regen (SR)")
    cells: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in sr["summary_table"]:
        if _cell_id(row.get("baseline"), row.get("site")) not in expected_cell_ids:
            continue
        cells.setdefault((row["site"], row["baseline"]), {})[row["mode"]] = row
    add("| cell | " + " | ".join(MODE_ORDER) + " |")
    add("|---|" + "---:|" * len(MODE_ORDER))
    for (site, baseline), modes in sorted(cells.items()):
        values = [decimal_format(modes[mode]["sr_pct"], 1, signed=False) for mode in MODE_ORDER]
        add(f"| {baseline}·{site} | " + " | ".join(values) + " |")
    add("")

    add("## E. Table 3 regen (fig0c strict drop-one)")
    add("| panel | mode | drop-one pp | CI95 |")
    add("|---|---|---:|---:|")
    for row in fig0c:
        if row.get("row_type", "numeric") != "numeric":
            continue
        if _cell_id(row.get("baseline"), row.get("site")) not in expected_cell_ids:
            continue
        add(
            f"| {row['site_baseline']} | {row['mode']} | "
            f"{decimal_format(float(row['drop_one_loss_pp']))} | "
            f"[{decimal_format(float(row['ci95_low_pp']))}, "
            f"{decimal_format(float(row['ci95_high_pp']))}] |"
        )
    add("")

    add("## F. Table 4 regen (H10)")
    if pass1_only:
        add(f"- **{H10_PENDING_NOTICE}**")
        add("- Table 4 numeric rows are intentionally withheld until Pass-2 lands.")
        add(f"- Suggested abstract `<H10-VERDICT>` phrase: “{H10_PENDING_ABSTRACT}”")
    else:
        for cid, cell in h10["per_cell"].items():
            add(
                f"- {cid}: "
                + " | ".join(
                    f"{key}={scalar_display(value)}"
                    for key, value in scalars(cell).items()
                )
            )
    add("")

    add("## G. Post-splice checklist")
    add("1. Run banned-phrase and residual-slot greps from VERDICT_DAY_RUNBOOK.md.")
    if pass1_only:
        add("2. Leave §6/Table 4 pending; rerun full final mode after Pass-2 lands.")
    else:
        add("2. Verify [P]→[A] provenance lifts and the router contrast cohort counts.")
    add("3. Run the required stress chain before paper-prose commit.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--decision", type=Path, default=DECISION)
    ap.add_argument("--h10", type=Path, default=H10)
    ap.add_argument("--sr", type=Path, default=SR)
    ap.add_argument("--fig0c", type=Path, default=FIG0C)
    ap.add_argument("--router", type=Path, default=ROUTER)
    ap.add_argument("--out", type=Path, default=None)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--rehearsal", action="store_true",
        help="emit INVALID_FOR_DRAFT diagnostics without copyable slots/tables",
    )
    mode.add_argument(
        "--h10-pending", action="store_true",
        help=(
            "COMPLETE-only Pass-1 final sheet: emit copyable H1/H3 and Tables 2/3 "
            "while withholding H10/router values until Pass-2"
        ),
    )
    mode.add_argument(
        "--protocol-note-06", action="store_true",
        help=(
            "accept only the authorized NOTE_06 k=5 decision artifact; emit H1/H3 "
            "and Tables 2/3 with mandatory five-landed-cell qualifiers while "
            "withholding H10/router/Table 4"
        ),
    )
    args = ap.parse_args(argv)

    load_errors: list[str] = []
    dec = _load_json(args.decision, "decision artifact", load_errors)
    sr = _load_json(args.sr, "SR artifact", load_errors)
    fig0c = _load_csv(args.fig0c, "fig0c artifact", load_errors)
    if args.h10_pending or args.protocol_note_06:
        h10: dict[str, Any] = {}
        router: dict[str, Any] = {}
    else:
        h10 = _load_json(args.h10, "H10 artifact", load_errors)
        router = _load_json(args.router, "router artifact", load_errors)
    validation_errors, gaps = validate_artifacts(
        dec, h10, sr, fig0c, router,
        final=not args.rehearsal,
        h10_pending=args.h10_pending,
        protocol_note_06=args.protocol_note_06,
    )
    errors = load_errors + validation_errors
    if errors and not args.rehearsal:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2

    output = build_sheet(
        dec, h10, sr, fig0c, router,
        rehearsal=args.rehearsal, h10_pending=args.h10_pending,
        protocol_note_06=args.protocol_note_06,
        errors=errors, gaps=gaps,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
        print(f"written: {args.out}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
