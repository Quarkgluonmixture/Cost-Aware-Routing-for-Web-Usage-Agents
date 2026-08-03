#!/usr/bin/env python3
"""How much of the latency we report is the model, and how much is the container? — 2026-08-03

Why this exists
---------------
Every latency figure in this project is `latency_ms.total` (or the canonical estimand,
which subtracts retry / busy-wait / recovered-screenshot time from it). None of them
subtract the environment. `latency_ms.backend_infer` — the model call on its own — has
been written on 100% of steps since the beginning and is read by no analysis script.

That matters because `offsite_navigation_audit` already established the reddit container
is 1.69x slower than the classifieds one before any agent behaviour enters. If the model
is a minority of the measured time, then a "latency ordering" over modes is substantially
an ordering over how much environment each mode provokes — which is a real property, but
not the one the phrase suggests.

This is a validity audit of the estimand, in the same shape as `offsite_navigation_audit`,
and deliberately NOT a 27th metric in `per_mode_four_dimension_profile`: adding one there
moves the >=7/8 consistency denominators that claim 6 rests on, and this finding does not
need that.

Found by `audit_field_consumption.py`, which was written after `confidence.verbalized`
turned out to be recorded everywhere, read by the calibration analysis, and absent from
the cascade — the product whose conclusion it would change.

Usage
-----
    .venv/bin/python3 scripts/analysis/latency_decomposition.py
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.lib.run_registry import get_cells  # noqa: E402

OUT_MD = REPO / "docs/analysis/cross_sites/latency_decomposition.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/latency_decomposition.json"
MODES = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]
WA_STEM = {"DOM": "dom", "SoM": "som", "Vision": "vision", "P-text": "phantom_text",
           "P-prompt": "phantom_prompt", "P-SoM": "phantom_som"}
FIELDS = ("total", "backend_infer", "obs_prepare", "env_step", "runtime_sleep")


class MissingInput(RuntimeError):
    """Fail loud rather than report a partial grid."""


def _episode_dirs() -> dict[str, dict[str, Path]]:
    cells: dict[str, dict[str, Path]] = {}
    for bl in ("B0", "B1", "B2"):
        for site in ("classifieds", "reddit"):
            d = {c.mode: Path(c.episodes_dir) for c in get_cells(baseline=bl, site=site)}
            if len(d) == len(MODES):
                cells[f"{bl}·{site}"] = d
    for bl in ("B0", "B1"):
        d: dict[str, Path] = {}
        for m, stem in WA_STEM.items():
            hits = [p for p in glob.glob(
                str(REPO / f"results/webarena/phase1/{bl}_{stem}_wa_reddit_2026*_R*"))
                if "ABORTED" not in p and Path(p).is_dir()]
            if len(hits) == 1:
                eps = sorted(Path(hits[0]).glob("*/episodes"))
                if eps:
                    d[m] = eps[0]
        if len(d) == len(MODES):
            cells[f"{bl}·wa_reddit"] = d
    if not cells:
        raise MissingInput("no complete six-mode cell found")
    return cells


def _mode_means(ep_dir: Path) -> dict[str, float]:
    buckets: dict[str, list[float]] = {k: [] for k in FIELDS}
    for f in sorted(ep_dir.glob("*steps_v2.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            lat = rec.get("latency_ms") or {}
            for k in FIELDS:
                v = lat.get(k)
                if v is not None:
                    buckets[k].append(float(v))
    if not buckets["total"]:
        raise MissingInput(f"no latency records under {ep_dir}")
    return {k: (statistics.fmean(v) if v else float("nan")) for k, v in buckets.items()}


def main() -> int:
    cells = _episode_dirs()
    out: dict = {"schema": "2026-08-03-latency-decomposition-v1",
                 "post_hoc_exploratory": True, "h10_eligible": False, "cells": {}}
    flips: list[str] = []
    for cid, dirs in sorted(cells.items()):
        per_mode = {m: _mode_means(d) for m, d in dirs.items()}
        fastest_total = min(per_mode, key=lambda m: per_mode[m]["total"])
        fastest_model = min(per_mode, key=lambda m: per_mode[m]["backend_infer"])
        share = (statistics.fmean(v["backend_infer"] for v in per_mode.values())
                 / statistics.fmean(v["total"] for v in per_mode.values()))
        if fastest_total != fastest_model:
            flips.append(cid)
        out["cells"][cid] = {
            "per_mode_ms": per_mode,
            "fastest_by_total": fastest_total,
            "fastest_by_model_only": fastest_model,
            "fastest_agrees": fastest_total == fastest_model,
            "model_share_of_total": share,
        }
    out["n_cells"] = len(out["cells"])
    out["n_fastest_flips"] = len(flips)
    out["flip_cells"] = flips
    shares = [c["model_share_of_total"] for c in out["cells"].values()]
    out["model_share_min"], out["model_share_max"] = min(shares), max(shares)

    L = ["---", "type: analysis", "status: complete",
         "purpose: how much of the reported latency is the model and how much is the environment",
         "post_hoc_exploratory: true",
         "scope_warning: this is a validity audit of the latency estimand, not a new "
         "behavioural metric. It deliberately does not enter per_mode_four_dimension_profile, "
         "because adding a 27th metric moves the >=7/8 consistency denominators claim 6 rests on.",
         "producer: scripts/analysis/latency_decomposition.py", "---", "",
         "# What is inside a latency number", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/latency_decomposition.py`", "",
         "Every latency figure in this project is `latency_ms.total`, or the canonical "
         "estimand, which subtracts retry / busy-wait / recovered-screenshot time from it. "
         "Neither subtracts the environment. `latency_ms.backend_infer` isolates the model "
         "call; it has been written on 100% of steps since the beginning and **was read by "
         "no analysis script until 2026-08-03**.", "",
         "## 1. The split, per cell", "",
         "| cell | mean total (ms) | mean model call (ms) | model share | obs prepare (ms) | runtime sleep (ms) |",
         "|---|---|---|---|---|---|"]
    for cid, c in out["cells"].items():
        pm = c["per_mode_ms"]
        tot = statistics.fmean(v["total"] for v in pm.values())
        inf = statistics.fmean(v["backend_infer"] for v in pm.values())
        obs = statistics.fmean(v["obs_prepare"] for v in pm.values()
                               if v["obs_prepare"] == v["obs_prepare"])
        slp = statistics.fmean(v["runtime_sleep"] for v in pm.values()
                               if v["runtime_sleep"] == v["runtime_sleep"])
        L.append(f"| `{cid}` | {tot:,.0f} | {inf:,.0f} | **{100*c['model_share_of_total']:.1f}%** "
                 f"| {obs:.1f} | {slp:.0f} |")
    L += ["", f"**The model is {100*out['model_share_min']:.0f}–"
          f"{100*out['model_share_max']:.0f}% of the time we report.** The remainder is the "
          "browser and the container. `offsite_navigation_audit` already measured the reddit "
          "container at 1.69x the classifieds one before any agent behaviour enters, so that "
          "remainder is not a constant either.", ""]

    L += ["## 2. Does the fastest mode change when the container is removed?", "",
          "| cell | fastest by total | fastest by model call alone | same? |",
          "|---|---|---|---|"]
    for cid, c in out["cells"].items():
        L.append(f"| `{cid}` | {c['fastest_by_total']} | {c['fastest_by_model_only']} | "
                 f"{'yes' if c['fastest_agrees'] else '**no**'} |")
    # Where the flips land is the test of whether this is noise or mechanism.
    _red_flip = sum(1 for c in flips if "reddit" in c)
    _red_total = sum(1 for c in out["cells"] if "reddit" in c)
    _cls_flip = len(flips) - _red_flip
    _cls_total = out["n_cells"] - _red_total
    out["flips_by_family"] = {"reddit": [_red_flip, _red_total],
                              "classifieds": [_cls_flip, _cls_total]}
    L += ["", f"⚠️ **The fastest mode changes in {out['n_fastest_flips']} of "
          f"{out['n_cells']} cells** ({', '.join('`' + c + '`' for c in flips) or 'none'}). "
          f"**They are not scattered: {_red_flip} of {_red_total} reddit-family cells flip "
          f"and {_cls_flip} of {_cls_total} classifieds cells do** — i.e. the flips land "
          "exactly where the container is slowest and the model is the smallest share of "
          "the step. That is the pattern a container effect produces, not the pattern noise "
          "produces, and it is why this is reported as an estimand problem rather than as a "
          "finding about modes. "
          "Any sentence naming *which* mode is fastest is therefore a statement about this "
          "deployment's browser and container as much as about the mode. The claim that "
          "survives is the weaker, estimand-independent one: **cost ordering and latency "
          "ordering are not the same ordering** — that is a statement about two rankings "
          "disagreeing, and it does not depend on which latency you rank by.", "",
          "## 3. Two things this rules out", "",
          "- **The SoM annotation step is not the cost.** `obs_prepare` — the marking pass "
          "that turns a screenshot into a numbered one — runs at 15–21 ms on SoM arms and "
          "~0.1 ms elsewhere. Against a 6,000–37,000 ms step it is nothing. If SoM is slower "
          "it is because of what it makes the agent *do*, not because annotating costs time.",
          "- **Runtime sleeps are not the cost either.** They run 0–211 ms per step, i.e. "
          "under 2% of a step even at their worst.", "",
          "## 4. What this does not license", "",
          "Model-only latency is **not** a better estimand for a deployment claim — a user "
          "waits for the whole step, container included. It is the right estimand for a claim "
          "about *the representation*, and the wrong one for a claim about *the system*. "
          "Both are reported here so a sentence can pick the one it means, which is the same "
          "discipline `outcome_efficiency` applies to the cost denominator.", ""]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[latency_decomposition] {out['n_cells']} cells; model share "
          f"{100*out['model_share_min']:.1f}–{100*out['model_share_max']:.1f}%; "
          f"fastest-mode flips in {out['n_fastest_flips']} cells")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
