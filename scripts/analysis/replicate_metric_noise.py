#!/usr/bin/env python3
"""How much of a behavioural-metric difference can a rerun produce? — 2026-08-03

`per_mode_four_dimension_profile` reports 26 metrics per (cell, mode) and the claim that
rests on them — the four image-free modes are behaviourally non-separable, Vision reaches
the consistency bar on 9 metrics and SoM on 5 — has never been read against run-to-run
noise. Every SR-scale claim in this project is judged against the rerun band
(`noise_floor_inventory`); the behavioural claims are not, and there was no reason for the
asymmetry beyond the fact that nobody had computed a per-metric band.

Three same-condition replicate pairs now exist on `B0 x classifieds` (dom, vision, som —
the SoM arm landed 2026-08-03). This measures, for each of the 26 metrics, how far the
metric moves when the SAME condition is run twice, and sets that beside how far it moves
BETWEEN modes in the same cell.

Method: the profile machinery is reused verbatim rather than reimplemented. A cell spec
carries `modes: {display_name -> episodes_dir}`, so swapping one arm's directory to its
replicate and re-running `profile_cell` yields that arm's 26 metrics computed exactly as
the paper computes them, on the same paired task set. The run-to-run band for a metric is
`|metric(run A) - metric(run B)|`, and the reported band is the max over the three arms.

Two metrics are excluded and named: `n_unique_solves` is cross-mode by construction (it
asks what no OTHER mode solved), so swapping one arm changes it for reasons that are not
run-to-run noise.

Usage
-----
    .venv/bin/python3 scripts/analysis/replicate_metric_noise.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.aggregate_phantom_lift import CELLS  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402
from scripts.analysis.per_mode_four_dimension_profile import (  # noqa: E402
    DIMENSIONS, DISPLAY_MODES, profile_cell,
)

OUT_MD = REPO / "docs/analysis/cross_sites/replicate_metric_noise.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/replicate_metric_noise.json"

# (display mode, replicate episodes dir) — all six arms replicated on B0 x classifieds.
# Listed three until 2026-08-22; the three phantom replicates had landed in
# CLEAN_PAIRS but never reached this dict, so the published band ignored them.
P1 = "results/visualwebarena/phase1"
REPLICATES = {
    "DOM": f"{P1}/../../repro_replicates/B0_dom_classifieds_R31194_clean_replicate/phase1_dom_router_0/episodes",
    "Vision": f"{P1}/../../repro_replicates/B0_vision_classifieds_R24792_clean_replicate/phase1_vision_router_0/episodes",
    "SoM": f"{P1}/B0_som_classifieds_20260803_084743_413015398_3677519_R30696/phase1_som_router_0/episodes",
    "P-text": f"{P1}/B0_phantom_text_classifieds_20260817_092244_763693821_1962797_R20043/phase1_phantom_text_router_0/episodes",
    "P-prompt": f"{P1}/B0_phantom_prompt_classifieds_20260817_184335_813828144_2037698_R12207/phase1_phantom_prompt_router_0/episodes",
    "P-SoM": f"{P1}/B0_phantom_som_classifieds_20260818_040525_430521618_2113605_R13257/phase1_phantom_som_router_0/episodes",
}
# Cross-mode by construction: it asks what no OTHER mode solved, so swapping one arm moves
# it for a reason that is not run-to-run noise.
CROSS_MODE_METRICS = {"n_unique_solves"}
ALL_METRICS = [(dim, key, label)
               for dim, items in DIMENSIONS.items() for key, label in items]


class MissingInput(RuntimeError):
    """Fail loud rather than report a band over a partial arm set."""


def _base_spec() -> dict:
    for c in CELLS:
        if c.get("baseline") == "B0" and c.get("site") == "classifieds":
            spec = dict(c)
            ids, _sha = expected_scored_ids("classifieds")
            spec["universe"] = set(ids)
            spec["steps_glob"] = "classifieds_task_*_steps_v2.jsonl"
            spec["modes"] = dict(spec.get("modes") or {})
            if not spec["modes"]:
                raise MissingInput("B0 x classifieds cell carries no `modes` map")
            return spec
    raise MissingInput("B0 x classifieds not present in the run registry")


def main() -> int:
    base = _base_spec()
    canonical = profile_cell(base)

    bands: dict[str, dict[str, float]] = {}     # metric -> {arm: |delta|}
    for arm, rel in REPLICATES.items():
        rep = (REPO / rel).resolve()
        if not rep.is_dir():
            raise MissingInput(f"{arm}: replicate episodes dir absent: {rep}")
        swapped = dict(base)
        swapped["modes"] = {**base["modes"], arm: rep}
        alt = profile_cell(swapped)
        a = canonical["per_mode"][arm]
        b = alt["per_mode"][arm]
        for _dim, key, _lab in ALL_METRICS:
            if key in CROSS_MODE_METRICS:
                continue
            va, vb = a.get(key), b.get(key)
            if va is None or vb is None:
                continue
            bands.setdefault(key, {})[arm] = abs(float(va) - float(vb))

    rows = []
    for dim, key, label in ALL_METRICS:
        if key in CROSS_MODE_METRICS:
            rows.append({"dimension": dim, "metric": key, "label": label,
                         "excluded": "cross-mode by construction"})
            continue
        per_arm = bands.get(key) or {}
        if not per_arm:
            rows.append({"dimension": dim, "metric": key, "label": label,
                         "excluded": "not populated on this cell"})
            continue
        vals = [canonical["per_mode"][m].get(key) for m in DISPLAY_MODES]
        vals = [float(v) for v in vals if v is not None]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else None
        band = max(per_arm.values())
        rows.append({
            "dimension": dim, "metric": key, "label": label,
            "band_per_arm": per_arm, "band_max": band,
            "cross_mode_spread": spread,
            "ratio": (spread / band) if (spread is not None and band > 0) else None,
            "exceeds_noise": (spread is not None and band > 0 and spread > band),
        })

    live = [r for r in rows if "excluded" not in r]
    n_exceed = sum(1 for r in live if r["exceeds_noise"])
    out = {"schema": "2026-08-03-replicate-metric-noise-v1",
           "post_hoc_exploratory": True, "h10_eligible": False,
           "cell": "B0·classifieds", "arms_replicated": sorted(REPLICATES),
           "n_metrics_total": len(ALL_METRICS), "n_metrics_live": len(live),
           "n_exceeding_noise": n_exceed, "metrics": rows}

    L = ["---", "type: analysis", "status: complete",
         "purpose: per-metric run-to-run band for the 26 behavioural metrics, and which "
         "cross-mode differences survive it",
         "post_hoc_exploratory: true",
         "scope_warning: one cell (B0 x classifieds) and one rerun per arm. A band from a "
         "single rerun is a point estimate, not a bound — the same caveat noise_floor_inventory "
         "carries for the SR-scale band.",
         "producer: scripts/analysis/replicate_metric_noise.py", "---", "",
         "# Can a rerun produce the behavioural differences?", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/replicate_metric_noise.py`", "",
         "Every success-rate claim in this project is judged against the rerun band. The "
         "26-metric behavioural claims never were — not for a reason, but because no "
         "per-metric band existed. Three replicated arms on `B0·classifieds` (dom, vision, "
         "**som**, the last landing 2026-08-03) now allow one.", "",
         f"**{n_exceed} of {len(live)} metrics** have a cross-mode spread larger than the "
         "largest run-to-run movement of the same metric.", "",
         "| dimension | metric | cross-mode spread | rerun band | ratio | bigger than a rerun? |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        if "excluded" in r:
            L.append(f"| {r['dimension']} | `{r['metric']}` | — | — | — | *{r['excluded']}* |")
            continue
        sp = "—" if r["cross_mode_spread"] is None else f"{r['cross_mode_spread']:.3f}"
        rt = "—" if r["ratio"] is None else f"{r['ratio']:.2f}×"
        L.append(f"| {r['dimension']} | `{r['metric']}` | {sp} | {r['band_max']:.3f} | {rt} "
                 f"| {'**yes**' if r['exceeds_noise'] else 'no'} |")
    fails = [r for r in live if not r["exceeds_noise"]]
    if fails:
        L += ["", "## The metrics a rerun can reproduce", ""]
        for r in sorted(fails, key=lambda r: r["ratio"] or 0):
            L.append(f"- **`{r['metric']}`** ({r['label']}) — spread "
                     f"{r['cross_mode_spread']:.3f} against a band of {r['band_max']:.3f}, "
                     f"ratio **{r['ratio']:.2f}×**.")
        lat = [r for r in fails if "latency" in r["metric"]]
        if lat:
            L += ["", "⚠️ **Both latency metrics are in that list, and that is not a "
                  "coincidence.** Independently of this table, `latency_decomposition` "
                  "measured that only 22–67% of a step is the model call — the rest is the "
                  "browser and the container — and that removing the container changes which "
                  "mode is fastest in 4 of 8 cells. Two unrelated routes reach the same "
                  "place: **on this cell the latency axis does not resolve modes above "
                  "run-to-run movement.** Claim 9's safe form (*the cost ordering and the "
                  "latency ordering disagree*) is a statement about two rankings and "
                  "survives; any sentence naming a mode as fastest does not."]
    L += ["", "`cross-mode spread` = max − min of that metric over the six modes in the "
          "canonical cell. `rerun band` = the largest |metric(run A) − metric(run B)| over "
          "the three replicated arms. A ratio near or below 1 means the differences the "
          "profile reports between modes are the size a rerun of one mode produces on its "
          "own.", "",
          "## What this does and does not settle", "",
          "**It is one cell and one rerun per arm.** The band is a point estimate of a "
          "random quantity, exactly as `noise_floor_inventory` §1b says of the SR-scale "
          "band — a second rerun would move it. Nothing here should be read as a threshold.",
          "",
          "**It does not touch the non-separability result directly.** That claim is about "
          "which mode is *extreme* on a metric across 8 cells, not about the size of a gap "
          "in one cell. A metric can have a small spread and still put the same mode at the "
          "top in every cell — consistency and magnitude are different questions, and the "
          "≥83% bar is a consistency bar. What this table adds is the magnitude the "
          "consistency is about, which the profile never printed.", ""]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[replicate_metric_noise] {len(live)} live metrics; "
          f"{n_exceed} have a cross-mode spread bigger than the rerun band")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
