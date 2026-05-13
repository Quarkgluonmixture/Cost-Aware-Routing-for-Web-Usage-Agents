#!/usr/bin/env python3
"""Re-aggregate Method 4.4 v2 sweep JSON with HDMI H-mean reliability metric.

Run after `run_stage4_method44_v2_sweep.py` completes. The v2 sweep script
writes a JSON with per-task per-layer per-α raw cells (shifted_toward_psom,
json_valid, overlap_dom, overlap_psom). This script re-aggregates with the
HDMI completeness × selectivity → harmonic mean reliability metric
(Khorasani et al. 2026 arXiv:2605.07631).

Idempotent — can re-run any time; just reads the JSON and writes the md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "results/mechanistic/stage4_multimode_b1_cls/method44_v2_sweep.json"
OUT_MD = ROOT / "docs/checkpoints/stage4_method44_v2_results.md"


def main():
    d = json.loads(JSON_PATH.read_text())
    cfg = d["config"]
    layers = cfg["layers"]
    alphas = cfg["alphas"]
    per_task = d["results"]

    agg = {}
    for L in layers:
        for alpha in alphas:
            cells = [r["per_layer"][str(L)][str(alpha)] for r in per_task
                      if str(L) in r["per_layer"] and str(alpha) in r["per_layer"][str(L)]]
            if not cells:
                continue
            completeness = float(np.mean([c["shifted_toward_psom"] for c in cells]))
            selectivity = float(np.mean([c["json_valid"] for c in cells]))
            hmean = (2 * completeness * selectivity / (completeness + selectivity + 1e-9)
                      if (completeness + selectivity) > 0 else 0.0)
            agg[f"L{L:02d}_a{alpha}"] = {
                "n": len(cells),
                "mean_overlap_dom": float(np.mean([c["overlap_dom"] for c in cells])),
                "mean_overlap_psom": float(np.mean([c["overlap_psom"] for c in cells])),
                "completeness": completeness,
                "selectivity": selectivity,
                "reliability": hmean,
                "first_token_psom_match_rate": float(np.mean([c["first_token_psom_match"] for c in cells])),
            }

    missing = [f"L{L:02d}_a{alpha}" for L in layers for alpha in alphas
               if f"L{L:02d}_a{alpha}" not in agg]
    if missing:
        raise RuntimeError(
            "Missing Method 4.4 v2 aggregate cells; refusing to render absent "
            f"layer/alpha cells as 0.0: {', '.join(missing[:20])}"
            + (" ..." if len(missing) > 20 else "")
        )

    # Also save back to JSON so other tools see H-mean
    d["aggregate"] = agg
    JSON_PATH.write_text(json.dumps(d, indent=2))

    write_md(d, OUT_MD, layers, alphas)


def write_md(d, out, layers, alphas):
    cfg = d["config"]
    n_cells = len(d["results"])

    def metric(L, a, name):
        key = f"L{L:02d}_a{a}"
        if key not in d["aggregate"]:
            raise KeyError(f"missing aggregate cell {key}")
        return d["aggregate"][key][name]

    lines = [
        "# Stage 4 Method 4.4 v2: Layer × α Sweep (HDMI reliability framework)",
        "",
        f"**Config**: tier={cfg['tier']}, n_task×step={n_cells}, max_new_tokens={cfg['max_new_tokens']}",
        f"**Direction norms per layer**: " + ", ".join(f"L{k}={v:.2f}" for k, v in cfg.get('direction_norms', {}).items()),
        "",
        "## HDMI Reliability — harmonic mean (completeness × selectivity)",
        "",
        "Following Khorasani et al. 2026 (arXiv:2605.07631): reliability = 2·c·s/(c+s).",
        "Penalizes \"shift target but break envelope\" failure mode. Higher = better.",
        "",
        "| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |",
        "|---|" + "|".join(["---"] * len(alphas)) + "|",
    ]
    for L in layers:
        row = [f"L{L:02d}"]
        for a in alphas:
            v = metric(L, a, "reliability")
            row.append(f"**{v:.2f}**")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Completeness (shifted-toward-P-SoM rate)")
    lines.append("")
    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
    for L in layers:
        row = [f"L{L:02d}"]
        for a in alphas:
            v = metric(L, a, "completeness")
            row.append(f"{v:.0%}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Selectivity (JSON envelope valid rate)")
    lines.append("")
    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
    for L in layers:
        row = [f"L{L:02d}"]
        for a in alphas:
            v = metric(L, a, "selectivity")
            row.append(f"{v:.0%}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Token overlap to DOM baseline")
    lines.append("")
    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
    for L in layers:
        row = [f"L{L:02d}"]
        for a in alphas:
            v = metric(L, a, "mean_overlap_dom")
            row.append(f"{v:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Token overlap to P-SoM baseline")
    lines.append("")
    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
    for L in layers:
        row = [f"L{L:02d}"]
        for a in alphas:
            v = metric(L, a, "mean_overlap_psom")
            row.append(f"{v:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Peak cell summary
    cells_with_rel = [(k, v["reliability"]) for k, v in d["aggregate"].items()]
    cells_with_rel.sort(key=lambda x: -x[1])
    lines.append("## Top-5 cells by reliability")
    lines.append("")
    lines.append("| Rank | (Layer, α) | Reliability | Completeness | Selectivity |")
    lines.append("|---|---|---|---|---|")
    for i, (k, _) in enumerate(cells_with_rel[:5], 1):
        v = d["aggregate"][k]
        lines.append(f"| {i} | {k} | {v['reliability']:.2f} | {v['completeness']:.0%} | {v['selectivity']:.0%} |")
    lines.append("")

    out.write_text("\n".join(lines) + "\n")
    print(f"summary → {out}")


if __name__ == "__main__":
    main()
