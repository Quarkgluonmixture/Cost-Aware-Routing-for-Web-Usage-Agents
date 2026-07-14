#!/usr/bin/env python3
"""Paper F2 — H1 forest from the canonical decision JSON only.

The canonical producer already emits each cell's strict six-mode H1 point,
task-paired bootstrap CI, exact task-set diagnostics, and the pooled FE result.
F2 therefore performs no second per-cell computation and does not join fig0c.

Final mode is fail-closed: exactly six planned cells, six unique canonical modes
per cell, ``complete_exact=True``, and ``analysis_status=COMPLETE`` are required.
``--interim`` permits a validated proper subset and stamps the render INTERIM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from scripts.analysis.lib.canonical_task_universe import expected_scored_ids
    from scripts.analysis.lib.run_registry import PAPER_MODES
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.canonical_task_universe import expected_scored_ids
    from scripts.analysis.lib.run_registry import PAPER_MODES

ROOT = Path(__file__).resolve().parents[3]
DECISION = ROOT / "results/phantom_paper/phase1_full_prereg_decision.json"
OUT = ROOT / "results/phantom_paper/figures/fig_f2_h1_forest"

DELTA_PP = 1.0
ARM = "P-SoM"
C_CELL = "#0072B2"
C_POOL = "#D55E00"
PLANNED_CELL_IDS = {
    f"{baseline}_{site}"
    for baseline in ("B0", "B1", "B2")
    for site in ("classifieds", "reddit")
}


def _cell_id(cell: dict) -> str:
    return f"{cell.get('baseline')}_{cell.get('site')}"


def validate_decision(dec: dict, *, interim: bool) -> tuple[list[dict], list[str]]:
    """Validate the canonical per-cell H1 schema and return ordered cells/errors."""
    errors: list[str] = []
    analysis_status = dec.get("analysis_status")
    if analysis_status not in {"COMPLETE", "PARTIAL", "INSUFFICIENT"}:
        errors.append(f"invalid or missing analysis_status={analysis_status!r}")
    if not interim and analysis_status != "COMPLETE":
        errors.append(
            f"final render requires analysis_status=COMPLETE, got {analysis_status!r}"
        )
    if not interim and dec.get("h1_verdict") not in {"PASS", "FAIL"}:
        errors.append("final render requires h1_verdict in {PASS, FAIL}")

    raw_cells = dec.get("per_cell")
    if not isinstance(raw_cells, list) or not raw_cells:
        return [], errors + ["per_cell is missing or empty"]

    by_id: dict[str, dict] = {}
    for cell in raw_cells:
        cid = _cell_id(cell)
        if cid in by_id:
            errors.append(f"duplicate per_cell identity: {cid}")
            continue
        if cid not in PLANNED_CELL_IDS:
            errors.append(f"unexpected per_cell identity: {cid}")
        by_id[cid] = cell
        h1 = cell.get("h1") if isinstance(cell.get("h1"), dict) else {}
        if h1.get("complete_exact") is not True:
            errors.append(f"{cid}: h1.complete_exact is not true")
        observed_n = h1.get("observed_n")
        if not isinstance(observed_n, dict) or set(observed_n) != set(PAPER_MODES):
            got = sorted(observed_n) if isinstance(observed_n, dict) else observed_n
            errors.append(
                f"{cid}: observed_n must contain six unique canonical modes; got {got}"
            )
        try:
            expected_ids, expected_sha = expected_scored_ids(str(cell.get("site")))
        except Exception as exc:  # fail closed with a useful schema error
            errors.append(f"{cid}: cannot resolve canonical task universe: {exc}")
            continue
        if h1.get("n_tasks") != len(expected_ids):
            errors.append(
                f"{cid}: n_tasks={h1.get('n_tasks')} != canonical {len(expected_ids)}"
            )
        if h1.get("task_set_sha256") != expected_sha:
            errors.append(f"{cid}: task_set_sha256 mismatch")
        for field in ("theta_pp", "ci95_lo_pp", "ci95_hi_pp"):
            if not isinstance(h1.get(field), (int, float)):
                errors.append(f"{cid}: missing numeric h1.{field}")

    ids = set(by_id)
    if not interim and ids != PLANNED_CELL_IDS:
        errors.append(
            "final render requires exact six planned cells: "
            f"missing={sorted(PLANNED_CELL_IDS - ids)} "
            f"extra={sorted(ids - PLANNED_CELL_IDS)}"
        )
    if interim and not ids.issubset(PLANNED_CELL_IDS):
        errors.append("interim cell IDs are not a subset of the planned six")
    return [by_id[cid] for cid in sorted(by_id)], errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--decision", type=Path, default=DECISION)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--interim", action="store_true",
        help="Render a validated proper subset with an INTERIM watermark.",
    )
    args = ap.parse_args(argv)

    if not args.decision.is_file():
        print(f"error: missing decision JSON: {args.decision}", file=sys.stderr)
        return 2
    try:
        dec = json.loads(args.decision.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: invalid decision JSON: {exc}", file=sys.stderr)
        return 2

    cell_records, errors = validate_decision(dec, interim=args.interim)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2

    cells = [
        (
            f"{cell['baseline']} {cell['site']}",
            float(cell["h1"]["theta_pp"]),
            float(cell["h1"]["ci95_lo_pp"]),
            float(cell["h1"]["ci95_hi_pp"]),
        )
        for cell in cell_records
    ]
    fe = dec.get("pooled_h1_fe", {})
    boot = dec.get("pooled_h1_bootstrap", {})
    theta = fe.get("theta_FE_pp")
    ci_lo = boot.get("ci95_lo_pp_bootstrap")
    ci_hi = boot.get("ci95_hi_pp_bootstrap")
    k = boot.get("k_cells")
    if not all(isinstance(v, (int, float)) for v in (theta, ci_lo, ci_hi)):
        print("error: pooled H1 point/CI fields are missing or non-numeric", file=sys.stderr)
        return 2
    if k != len(cells):
        print(
            f"error: pooled k_cells={k!r} does not match per_cell count={len(cells)}",
            file=sys.stderr,
        )
        return 2

    n = len(cells)
    fig, ax = plt.subplots(figsize=(6.6, 0.62 * (n + 2) + 1.4), dpi=300)
    ys = list(range(n, 0, -1))
    for (label, d, lo, hi), y in zip(cells, ys):
        ax.plot([lo, hi], [y, y], color=C_CELL, lw=1.8, zorder=2)
        ax.plot([lo, lo], [y - 0.12, y + 0.12], color=C_CELL, lw=1.8)
        ax.plot([hi, hi], [y - 0.12, y + 0.12], color=C_CELL, lw=1.8)
        ax.scatter([d], [y], s=54, color=C_CELL, zorder=3)
        ax.annotate(
            f"{d:+.2f} [{lo:.2f}, {hi:.2f}]", (hi, y), xytext=(8, 0),
            textcoords="offset points", va="center", fontsize=8, color="#333333",
        )

    y0 = 0
    ax.fill(
        [ci_lo, theta, ci_hi, theta], [y0, y0 + 0.22, y0, y0 - 0.22],
        color=C_POOL, alpha=0.85, zorder=3,
    )
    verdict = dec.get("h1_verdict", "NOT_EVALUATED")
    ax.annotate(
        f"FE pool (k={k}): {theta:+.2f} [{ci_lo:.2f}, {ci_hi:.2f}]  "
        f"gate({DELTA_PP:+.1f}pp): {verdict}",
        (max(ci_hi, DELTA_PP), y0), xytext=(8, 0), textcoords="offset points",
        va="center", fontsize=8, fontweight="bold", color=C_POOL,
    )

    ax.axvline(0.0, color="#555555", lw=0.9, ls="--", zorder=1)
    ax.axvline(DELTA_PP, color=C_POOL, lw=0.9, ls=":", zorder=1)
    ax.text(
        DELTA_PP, n + 0.75, f"gate {DELTA_PP:+.1f}pp", fontsize=7.5,
        color=C_POOL, ha="center",
    )
    ax.set_yticks(ys + [0])
    ax.set_yticklabels([c[0] for c in cells] + ["FE pool"], fontsize=9)
    ax.set_xlabel(
        f"{ARM} drop-one loss from 6-mode portfolio "
        "(pp; task-paired bootstrap 95% CI)", fontsize=9,
    )
    ax.set_ylim(-0.8, n + 1.1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(
        "H1: pooled P-SoM drop-one vs +1.0pp substantive threshold", fontsize=10,
    )

    if args.interim:
        fig.text(
            0.5, 0.5,
            f"INTERIM ({dec.get('analysis_status')}) — NOT A VERDICT",
            fontsize=22, color="#CC0000", alpha=0.25, ha="center",
            va="center", rotation=18, zorder=10,
        )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = args.out.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
