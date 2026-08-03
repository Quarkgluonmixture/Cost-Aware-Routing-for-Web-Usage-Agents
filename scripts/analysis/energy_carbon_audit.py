#!/usr/bin/env python3
"""Is the carbon column an axis, or wall-clock in other units? — 2026-08-03

`EVIDENCE_LAYER_SUMMARY` §4a has asserted since it was written that the energy column is
"wall-clock in other units and is reported as uninformative rather than as an axis".
That is almost certainly right, and it had never been measured. An unmeasured limitation
is still a claim, and this one is easy to check: if energy is a near-constant power times
elapsed time, then per-episode energy correlates with per-episode latency at r ~ 1.

Checking it turned up two things the assertion does not mention:

  * `configs/exp_v2_base.yaml:95` sets `use_pynvml: true`, and every step records
    `source: psutil_profile`. The configuration asks for the GPU counter and the runtime
    used the CPU estimate. Nothing failed loudly, because no product ever read the field.
  * **B0 has no energy data at all** (`source: disabled`, 0% populated). It is API-served,
    so there is nothing local to measure and disabling it is correct — but it means any
    carbon statement covers two of the three backbones, and the absent one is the
    strongest and the one every headline number uses.

Usage
-----
    .venv/bin/python3 scripts/analysis/energy_carbon_audit.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.lib.run_registry import get_cells  # noqa: E402

OUT_MD = REPO / "docs/analysis/cross_sites/energy_carbon_audit.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/energy_carbon_audit.json"
MAX_EP = 25          # episodes per condition; r is already saturated well below this


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx, sy = statistics.pstdev(x), statistics.pstdev(y)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (n * sx * sy)


def main() -> int:
    rows: list[dict] = []
    by_backbone: dict[str, Counter] = {}
    for bl in ("B0", "B1", "B2"):
        src = Counter()
        for site in ("classifieds", "reddit"):
            for cell in get_cells(baseline=bl, site=site):
                kwh, lat, watt = [], [], []
                for f in sorted(Path(cell.episodes_dir).glob("*steps_v2.jsonl"))[:MAX_EP]:
                    for line in f.read_text().splitlines():
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        e = rec.get("energy") or {}
                        L = rec.get("latency_ms") or {}
                        src[e.get("source") or "<absent>"] += 1
                        if e.get("kwh") is not None and L.get("total") is not None:
                            kwh.append(float(e["kwh"]))
                            lat.append(float(L["total"]))
                        if e.get("power_watts") is not None:
                            watt.append(float(e["power_watts"]))
                if len(kwh) > 10:
                    rows.append({
                        "cell": f"{bl}·{site}", "mode": cell.mode, "n_steps": len(kwh),
                        "r_energy_latency": _pearson(kwh, lat),
                        "mean_power_w": statistics.fmean(watt) if watt else None,
                        "sd_power_w": statistics.pstdev(watt) if len(watt) > 1 else 0.0,
                    })
        by_backbone[bl] = src

    rs = [r["r_energy_latency"] for r in rows]
    ws = [r["mean_power_w"] for r in rows if r["mean_power_w"] is not None]
    out = {"schema": "2026-08-03-energy-carbon-audit-v1", "post_hoc_exploratory": True,
           "h10_eligible": False, "conditions": rows,
           "source_counts": {k: dict(v) for k, v in by_backbone.items()},
           "r_min": min(rs), "r_max": max(rs), "r_mean": statistics.fmean(rs),
           "power_mean_w": statistics.fmean(ws), "power_sd_w": statistics.pstdev(ws),
           "backbones_with_energy": sorted(
               b for b, c in by_backbone.items() if any(k == "psutil_profile" for k in c))}

    L = ["---", "type: analysis", "status: complete",
         "purpose: measure whether the energy/carbon column carries information beyond elapsed time",
         "post_hoc_exploratory: true",
         "producer: scripts/analysis/energy_carbon_audit.py", "---", "",
         "# Is carbon an axis?", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/energy_carbon_audit.py`", "",
         "`EVIDENCE_LAYER_SUMMARY` §4a has always said the energy column is \"wall-clock in "
         "other units\". It is — and until 2026-08-03 nobody had measured it. An unmeasured "
         "limitation is still a claim.", "",
         "## 1. Energy against elapsed time", "",
         f"Over **{len(rows)} conditions**, per-step energy correlates with per-step latency "
         f"at **r = {out['r_min']:.4f} to {out['r_max']:.4f}** (mean "
         f"**{out['r_mean']:.4f}**). Recorded power is **{out['power_mean_w']:.1f} W** with a "
         f"cross-condition SD of {out['power_sd_w']:.2f} W.", "",
         "| cell | mode | steps | r(energy, latency) | mean W | SD W |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['cell']}` | {r['mode']} | {r['n_steps']} | {r['r_energy_latency']:.4f} "
                 f"| {r['mean_power_w']:.1f} | {r['sd_power_w']:.2f} |")
    L += ["", "A near-constant wattage times elapsed time **is** elapsed time. The column "
          "should be reported as uninformative, never as a third axis beside cost and "
          "latency, and no mode-ordering may be read off it that is not already the latency "
          "ordering.", "",
          "## 2. Two things the standing limitation does not say", "",
          "**The configuration asked for the GPU counter and did not get it.** "
          "`configs/exp_v2_base.yaml:95` sets `use_pynvml: true`; every step records "
          "`source: psutil_profile`. That is a CPU-package estimate, which is why the number "
          "sits at ~66 W on an accelerator rated several times that. Nothing failed loudly "
          "because no product read the field — the same shape as `confidence.verbalized` and "
          "`latency_ms.backend_infer`.", ""]
    L += ["**B0 has no energy data at all.** Per-backbone `energy.source` counts:", "",
          "| backbone | source values |", "|---|---|"]
    for b, c in out["source_counts"].items():
        L.append(f"| {b} | " + ", ".join(f"`{k}` × {v:,}" for k, v in sorted(c.items())) + " |")
    L += ["", "B0 is served through an API, so there is no local draw to measure and "
          "`disabled` is the correct setting. The consequence still has to be stated: "
          f"**energy exists for {', '.join(out['backbones_with_energy'])} only**, and the "
          "backbone it is missing on is the strongest one and the one every headline number "
          "is computed on. A carbon comparison across the three backbones cannot be made at "
          "all — not merely 'uncalibrated', but absent.", "",
          "## 3. What would make it an axis", "",
          "A per-accelerator counter (`pynvml` actually engaged, or a wall meter) on the "
          "locally-served arms, plus a defensible figure for the API-served arm's remote "
          "draw — which the provider does not publish. The second half is not obtainable, so "
          "the honest position is that this project cannot report carbon as a comparable "
          "quantity across its backbones, and reporting it per-backbone adds nothing over "
          "reporting latency.", ""]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[energy_carbon_audit] {len(rows)} conditions; "
          f"r {out['r_min']:.4f}–{out['r_max']:.4f} (mean {out['r_mean']:.4f}); "
          f"power {out['power_mean_w']:.1f} W; energy present on "
          f"{','.join(out['backbones_with_energy'])}")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
