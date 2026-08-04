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
        # An argmax is only meaningful if the curve has a peak. On several arms it does
        # not: `p2_psom_ptext_cls` reaches its maximum convergence at SIX layers exactly
        # (L00, L24, L26, L27, L29, L30) over a total curve range of 0.014, so "the peak
        # layer" is whichever tied index the implementation happens to return. The layer is
        # what §5's content-specific reading is argued from, so these two fields ship
        # beside it and any statement about a peak layer must be read against them.
        "peak_convergence_n_tied": sum(1 for v in conv if abs(v - c_peak) < 1e-12) if conv else None,
        "convergence_spread": (max(conv) - min(conv)) if conv else None,
        "peak_displacement_n_tied": sum(1 for v in disp if abs(v - d_peak) < 1e-12) if disp else None,
        "displacement_spread": (max(disp) - min(disp)) if disp else None,
        "convergence_at_L0": conv[0] if conv else None,
        # L23 is where Method 4.2's cosine gap peaks; the layer disjoint is the claim.
        "displacement_at_L23": disp[23] if len(disp) > 23 else None,
        "source": str(path.relative_to(REPO)),
    }


def patching() -> dict:
    """The 2026-05 Myriad sweep. Kept unchanged: two tables cite these numbers."""
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


# --- the 2026-08 DGX sweep -----------------------------------------------------------
# 24 cells finished on DGX 2026-07-30 → 08-03 (`logs/mechanistic_canonical/`, marker
# "24/24, 18 ran, 6 skipped"). Nothing read them: this product was regenerated at
# 2026-08-04 01:25, *after* the sweep ended, and still pointed only at the May Myriad
# directories. A finished run that no product reads is indistinguishable, on disk, from a
# run that was never launched — the same failure mode this file was written to prevent,
# one directory over.
CANON = MECH / "canonical"

# Which cells re-run a May arm under an identical config. Verified field-by-field before
# pairing (model, n_layers, n_tasks, step, max_new_tokens, source/target mode, tier): these
# are the SAME experiment executed twice, months apart, on different hardware — so their
# disagreement is run-to-run variance of the mechanism layer, measured the same way this
# paper measures it everywhere else.
REPLICATE_OF = {
    "p2_psom_ptext_cls": ("classifieds", "real"),
    "p2_psom_ptext_red": ("reddit", "real"),
    "p2_taskshuf_cls": ("classifieds", "task_shuffled"),
    "p2_taskshuf_red": ("reddit", "task_shuffled"),
    "p5_psom_ptext_rand_cls": ("classifieds", "random_injection"),
    "p5_psom_ptext_rand_red": ("reddit", "random_injection"),
}

# Which axis each cell's (source → target) contrast probes. Named from the config rather
# than from the directory name, so a renamed cell cannot silently change axis.
def _axis_of(cfg: dict) -> str:
    s, t = cfg.get("source_mode"), cfg.get("target_mode")
    if cfg.get("random_inject"):
        return "control:random_injection"
    pair = frozenset({s, t})
    if pair == frozenset({"som", "phantom_som"}):
        return "image"                      # only the screenshot differs
    if pair == frozenset({"som", "dom"}):
        return "image+text-format+prompt"   # compound
    if pair == frozenset({"phantom_som", "phantom_text"}):
        return "prompt-style"
    if pair == frozenset({"phantom_som", "phantom_prompt"}):
        return "text-format"
    if pair == frozenset({"som", "phantom_text"}) or pair == frozenset({"som", "phantom_prompt"}):
        return "image+one-text-axis"
    return f"{s}->{t}"


def patching_canonical() -> dict:
    """Every cell of the 2026-08 sweep, tagged by axis and by what it replicates."""
    out = {}
    if not CANON.is_dir():
        return out
    for d in sorted(CANON.iterdir()):
        f = d / "patching_continuation_results.json"
        if not f.is_file():
            continue
        got = _arm_curve(f)
        if not got:
            continue
        cfg = json.loads(f.read_text())["config"]
        got["axis"] = _axis_of(cfg)
        got["reverse"] = bool(cfg.get("reverse"))
        got["n_tasks"] = cfg.get("n_tasks")
        got["replicates"] = REPLICATE_OF.get(d.name)
        out[d.name] = got
    return out


def replication(old: dict, new: dict) -> list:
    """Same config, two runs — what moved.

    The peak LAYER is reported next to the peak VALUE because §5's content-specific claim
    rests on the layer, not the magnitude: the caption of the patching table argues that the
    real arm peaks mid-stack while the shuffled arm collapses to the boundary. If the layer
    is not stable across a re-run, that argument cannot carry the claim on its own.
    """
    rows = []
    for cell, (site, arm) in REPLICATE_OF.items():
        a = (old.get(site) or {}).get(arm)
        b = new.get(cell)
        if not a or not b:
            continue
        rows.append({
            "site": site, "arm": arm, "cell_2026_08": cell,
            "peak_convergence": [a["peak_convergence"], b["peak_convergence"]],
            "peak_convergence_layer": [a["peak_convergence_layer"], b["peak_convergence_layer"]],
            "peak_displacement": [a["peak_displacement"], b["peak_displacement"]],
            "peak_displacement_layer": [a["peak_displacement_layer"], b["peak_displacement_layer"]],
            "conv_layer_moved": a["peak_convergence_layer"] != b["peak_convergence_layer"],
            "n_tied": [a.get("peak_convergence_n_tied"), b.get("peak_convergence_n_tied")],
            "spread": [a.get("convergence_spread"), b.get("convergence_spread")],
        })
    return rows


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
        "patching_canonical_2026_08": patching_canonical(),
    }
    payload["replication_2026_05_vs_2026_08"] = replication(
        payload["patching"], payload["patching_canonical_2026_08"])
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
    # --- 2026-08 sweep, grouped by the axis each contrast probes
    canon = payload["patching_canonical_2026_08"]
    if canon:
        lines += ["", "## The 2026-08 sweep (24 cells, DGX) — grouped by axis", "",
                  "Finished 2026-08-03 and read by nothing until 2026-08-04. Random injection "
                  "is the destruction control: it maximises displacement while converging on "
                  "nothing, which is why displacement alone proves no steering.", "",
                  "| axis | cell | src to tgt | n | peak disp | L | peak conv | L |",
                  "|---|---|---|---|---|---|---|---|"]
        for cell, a in sorted(canon.items(), key=lambda kv: (kv[1]["axis"], kv[0])):
            c = a["config"]
            arrow = f"{c['source_mode']} to {c['target_mode']}" + (" (rev)" if a["reverse"] else "")
            lines.append(
                f"| {a['axis']} | `{cell}` | {arrow} | {a['n_tasks']} | "
                f"{a['peak_displacement']:.3f} | L{a['peak_displacement_layer']:02d} | "
                f"{a['peak_convergence']:.3f} | L{a['peak_convergence_layer']:02d} |")

    # --- same config, run twice
    rep = payload.get("replication_2026_05_vs_2026_08") or []
    if rep:
        moved = sum(1 for r in rep if r["conv_layer_moved"])
        lines += ["", "## Same configuration, run twice (2026-05 Myriad vs 2026-08 DGX)", "",
                  f"Six arms re-ran under a field-identical config. **The convergence peak "
                  f"layer moved in {moved} of {len(rep)}.** This matters more than the value "
                  f"movement: the content-specific reading of the patching result is argued "
                  f"from the peak LAYER (real mid-stack, shuffled at the boundary), so a peak "
                  f"layer that is not reproducible cannot carry that argument by itself. The "
                  f"same rerun discipline this paper applies to success rates applies here.",
                  "",
                  "| site | arm | peak conv 05 to 08 | conv layer 05 to 08 | moved? | tied layers 05/08 | curve range 05/08 |",
                  "|---|---|---|---|---|---|---|"]
        for r in rep:
            c0, c1 = r["peak_convergence"]
            l0, l1 = r["peak_convergence_layer"]
            t0, t1 = r.get("n_tied", [None, None])
            sp0, sp1 = r.get("spread", [None, None])
            tie = (f"{t0}/{t1}" if t0 is not None else "—")
            spr = (f"{sp0:.3f}/{sp1:.3f}" if isinstance(sp0, float) else "—")
            lines.append(
                f"| {r['site']} | {r['arm']} | {c0:.3f} to {c1:.3f} | "
                f"L{l0:02d} to L{l1:02d} | {'**yes**' if r['conv_layer_moved'] else 'no'} | "
                f"{tie} | {spr} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"[json] {OUT_JSON}")
    print(f"[md]   {OUT_MD}")


if __name__ == "__main__":
    main()
