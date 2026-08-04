#!/usr/bin/env python3
"""Freeze the §5 mechanism results into one cited product.

Why this exists (2026-08-03). The mechanism work was shelved by the advisor on 2026-05-14
and `realm/section5.md` has been an empty placeholder since. The *data* never went
anywhere: `results/mechanistic/` still holds the canonical hidden-state metrics and the
three-arm patching sweep, and the write-ups sit in `docs/checkpoints/mechanism/results/`
as prose Markdown that no table generator can read. A cross-AI coverage audit found them
uncited, which is a different state from absent — a shelved result and a missing result
look identical on disk and completely different to a reviewer.

This reads the **canonical JSON**, never the prose write-ups, so the numbers cannot drift
from a hand-typed summary. Two products come out:

  * **readability** — Method 4.2 v2: leave-one-task-out AUROC between every mode pair, and
    the cosine gaps those AUROCs are computed against. The point of putting them together
    is that the AUROCs are ~1.0 while the gaps are sub-permille on two of three axes.
  * **patching** — the Stage-3 prompt-family sweep with **both** of its controls
    (`_rand` = random injection, `_taskshuf` = task-shuffled source). A displacement
    number without those two is not evidence of anything content-specific.

Usage:
    .venv/bin/python3 scripts/analysis/aggregate_mechanism_evidence.py
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]
MECH = REPO / "results/mechanistic"
OUT_JSON = REPO / "docs/analysis/cross_sites/mechanism_evidence.json"
OUT_MD = REPO / "docs/analysis/cross_sites/mechanism_evidence.md"

SITES = {"cls": "classifieds", "red": "reddit"}
ARMS = {"real": "", "random_injection": "_rand", "task_shuffled": "_taskshuf"}

# The three axis contrasts §5 is about, in Method 4.2's pair naming.
AXES = {
    "image": "phantom_som_vs_som",
    "text_format": "dom_vs_phantom_text",
    "prompt_family": "phantom_text_vs_phantom_som",
}


def _pair(d: dict, key: str):
    """Method 4.2 stores each unordered pair once; try both orderings."""
    if key in d:
        return d[key]
    a, b = key.split("_vs_")
    return d.get(f"{b}_vs_{a}")


def _peak(series) -> tuple[float | None, int | None]:
    """Peak value of a per-layer series and the layer it peaks at.

    Every Method 4.2 entry is a 37-long per-layer list, not a scalar — the write-ups
    quote best-layer values, and the *layer* is half the claim (cosine peaks late,
    patching peaks mid), so both come back.
    """
    if not isinstance(series, list):
        return (series if isinstance(series, (int, float)) else None), None
    scored = [(v, i) for i, v in enumerate(series) if isinstance(v, (int, float))]
    if not scored:
        return None, None
    val, layer = max(scored)
    return val, layer


def readability() -> dict:
    out = {}
    for short, site in SITES.items():
        p = MECH / f"stage4_multimode_b1_{'cls' if short == 'cls' else 'reddit'}/method42_metrics_v2.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        loto = d.get("pairwise_auroc_lototask", {})
        gaps = d.get("pairwise_cosine_gap", {})
        # Best-layer AUROC per pair: the claim is "separable somewhere", not "at every layer"
        # (layer 0 sits at chance 0.5 for every pair, by construction).
        best = {k: _peak(v) for k, v in loto.items()}
        vals = [v for v, _ in best.values() if v is not None]
        axis_gap = {}
        for axis, key in AXES.items():
            val, layer = _peak(_pair(gaps, key))
            axis_gap[axis] = {"peak_gap": val, "peak_layer": layer}
        out[site] = {
            "n_examples": d.get("n_examples"),
            "n_modes": d.get("n_modes"),
            "n_layers": d.get("n_layers"),
            "n_pairs": len(vals),
            "auroc_lototask_best_layer_min": min(vals) if vals else None,
            "auroc_lototask_best_layer_mean": mean(vals) if vals else None,
            "n_pairs_at_auroc_1": sum(1 for v in vals if v >= 0.9999),
            # best-layer over 37 layers x 15 pairs = 555 chances; reporting only the peak makes
            # "15/15 perfectly separable" read as 15 independent separations. This says how
            # BROAD each separation is, which is the part a max hides.
            "median_layers_at_auroc_1": (
                sorted(sum(1 for x in v if isinstance(x, (int, float)) and x >= 0.9999)
                       for v in loto.values())[len(loto) // 2] if loto else None),
            "n_layers_total": d.get("n_layers"),
            "axis_cosine_gap": axis_gap,
            "source": str(p.relative_to(REPO)),
        }
    return out


def _arm_curve(path: Path) -> dict | None:
    """Per-layer displacement AND convergence for one patching arm.

    Two quantities, because one of them cannot carry the causal claim on its own:

      * ``displacement`` = 1 − overlap with the *unpatched target* — how much the output
        moved. Wrecking the residual stream maximises this: the random-injection arm
        reaches 0.99, which is destruction, not steering.
      * ``convergence``  = overlap with the *source* continuation — whether it moved
        **toward the source**. This is the direction the causal claim is about.

    Reporting displacement alone is what made the task-shuffled control look like it
    replicated the effect (0.30 against real's 0.23 on classifieds): a source drawn from
    an unrelated task perturbs just as hard, and only convergence separates them.
    """
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    tasks = d.get("per_task") or []
    if not tasks:
        return None
    n_layers = len(tasks[0].get("per_layer") or [])

    def _curve(field: str, transform=lambda v: v):
        out = []
        for li in range(n_layers):
            vals = [transform(t["per_layer"][li][field])
                    for t in tasks
                    if li < len(t.get("per_layer") or [])
                    and isinstance(t["per_layer"][li].get(field), (int, float))]
            out.append(mean(vals) if vals else None)
        return out

    disp = _curve("token_overlap_to_target", lambda v: 1.0 - v)
    conv = _curve("token_overlap_to_source")
    d_scored = [(v, i) for i, v in enumerate(disp) if v is not None]
    c_scored = [(v, i) for i, v in enumerate(conv) if v is not None]
    d_peak, d_layer = max(d_scored) if d_scored else (None, None)
    c_peak, c_layer = max(c_scored) if c_scored else (None, None)
    return {
        "n_tasks": len(tasks),
        "config": {k: d["config"].get(k) for k in
                   ("source_mode", "target_mode", "step", "random_inject", "tier")},
        "displacement_by_layer": disp,
        "convergence_by_layer": conv,
        "peak_displacement": d_peak,
        "peak_displacement_layer": d_layer,
        "peak_convergence": c_peak,
        "peak_convergence_layer": c_layer,
        "convergence_at_L0": conv[0] if conv else None,
        # L23 is where Method 4.2's cosine gap peaks; the layer disjoint is the claim.
        "displacement_at_L23": disp[23] if len(disp) > 23 else None,
        "source": str(path.relative_to(REPO)),
    }


def patching() -> dict:
    out = {}
    for short, site in SITES.items():
        arms = {}
        for arm, suffix in ARMS.items():
            p = MECH / f"stage3_cellhprompt_{short}_fwd_ptext{suffix}_myriad/patching_continuation_results.json"
            got = _arm_curve(p)
            if got:
                arms[arm] = got
        if arms:
            out[site] = arms
    return out


def main() -> None:
    payload = {
        "schema": "2026-08-03-mechanism-evidence-v1",
        "note": (
            "Frozen §5 evidence, read from canonical results/mechanistic JSON rather than "
            "from the prose write-ups in docs/checkpoints/mechanism/results/. Mechanism work "
            "was shelved 2026-05-14; this product exists so the shelved results are citable "
            "and distinguishable from results that were never obtained."
        ),
        "readability": readability(),
        "patching": patching(),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1))

    lines = ["# Mechanism evidence (frozen §5)", "", payload["note"], "",
             "## Linear readability vs geometric magnitude (Method 4.2 v2)", "",
             "| site | modes | examples | pairs | pairs at AUROC 1.000 | worst pair | "
             "image gap | text-format gap | prompt-family gap |",
             "|---|---|---|---|---|---|---|---|---|"]
    for site, r in payload["readability"].items():
        g = r["axis_cosine_gap"]
        lines.append(
            f"| {site} | {r['n_modes']} | {r['n_examples']} | {r['n_pairs']} | "
            f"{r['n_pairs_at_auroc_1']}/{r['n_pairs']} | "
            f"{r['auroc_lototask_best_layer_min']:.3f} | "
            + " | ".join(
                f"{g[a]['peak_gap']:.4f} (L{g[a]['peak_layer']:02d})"
                if isinstance(g[a].get("peak_gap"), (int, float)) else "—"
                for a in ("image", "text_format", "prompt_family")) + " |")
    lines += ["", "## Causal patching with both controls (Stage 3, prompt-family axis)", "",
              "`displacement` = 1 − overlap with the unpatched target (how far the output moved). ",
              "`convergence` = overlap with the source continuation (whether it moved *toward "
              "the source*). Displacement alone cannot separate steering from destruction.", "",
              "| site | arm | n | peak displacement | disp layer | peak convergence | conv layer | "
              "displacement at L23 |", "|---|---|---|---|---|---|---|---|"]
    for site, arms in payload["patching"].items():
        for arm, a in arms.items():
            l23 = a["displacement_at_L23"]
            lines.append(
                f"| {site} | {arm} | {a['n_tasks']} | {a['peak_displacement']:.3f} | "
                f"L{a['peak_displacement_layer']:02d} | {a['peak_convergence']:.3f} | "
                f"L{a['peak_convergence_layer']:02d} | "
                + (f"{l23:.3f} |" if isinstance(l23, (int, float)) else "— |"))
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"[json] {OUT_JSON}")
    print(f"[md]   {OUT_MD}")


if __name__ == "__main__":
    main()
