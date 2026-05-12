#!/usr/bin/env python3
"""Stage 4 H1 test analysis: do all indexed-list formats trigger Mirage signature?

Loads format variation hidden states (8 variants + dom + som baselines).
For each variant V, compute per-layer cosine gap V↔som (image-axis test).
Peak layer for variant indicates whether variant triggers marks-shortcut
(peak L17+) or behaves like AXTree no-image (peak L04).

H1 prediction:
  marks-like variants (som_standard / browser_use_at / appagent_id /
    tarsier_typed / plain_numbered / xml_tagged) → V↔som peak L17+
  controls (hash_id_control / plain_sentence) → V↔som peak L04 (like dom)

Outputs:
  docs/checkpoints/mechanism/results/format_variation_h1_test.md
  results/phantom_paper/figures/fig_stage4_format_variation_h1.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = ROOT / "results/mechanistic/stage4_format_variation_b1_cls/hidden_states.npz"
DEFAULT_OUT_MD = ROOT / "docs/checkpoints/mechanism/results/format_variation_h1_test.md"
DEFAULT_OUT_FIG = ROOT / "results/phantom_paper/figures/fig_stage4_format_variation_h1.png"

# Order: 6 marks-like variants + 2 controls + 2 baselines
VARIANTS = ["som_standard", "browser_use_at", "appagent_id", "tarsier_typed",
             "plain_numbered", "xml_tagged",
             "hash_id_control", "plain_sentence",  # 2 controls
             "dom", "som"]

DISPLAY = {
    "som_standard": "[N] role 'label' (SoM)",
    "browser_use_at": "@N label (Browser Use)",
    "appagent_id": "id_N: label (AppAgent)",
    "tarsier_typed": "[BN:role:label] (Tarsier)",
    "plain_numbered": "N. label (numbered)",
    "xml_tagged": "<el_N role='..'>label</el_N> (XML)",
    "hash_id_control": "#hash label (no integer)",
    "plain_sentence": "'a, b, c, ...' (no list)",
    "dom": "AXTree (baseline DOM)",
    "som": "marks + image (baseline SoM)",
}

# H1 prediction class
H1_CLASS = {
    "som_standard": "marks-like",
    "browser_use_at": "marks-like",
    "appagent_id": "marks-like",
    "tarsier_typed": "marks-like",
    "plain_numbered": "marks-like",
    "xml_tagged": "marks-like",
    "hash_id_control": "control (no integer)",
    "plain_sentence": "control (no list)",
    "dom": "AXTree-baseline",
    "som": "image-baseline",
}


def cosine_gap(a, b):
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_NPZ,
                        help="hidden_states.npz path (default: cls)")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--output-fig", type=Path, default=DEFAULT_OUT_FIG)
    args = parser.parse_args()

    d = np.load(args.input, allow_pickle=True)
    H = d["hidden_states"]
    ml = d["mode_labels_str"]
    n_layers = H.shape[1]
    print(f"loaded {H.shape} from {args.input}, n modes = {len(set(ml.tolist()))}")

    means = {m: H[ml == m].mean(axis=0) for m in VARIANTS}

    results = {}
    for v in VARIANTS:
        if v == "som":
            continue
        curve = np.array([cosine_gap(means[v][L], means["som"][L]) for L in range(n_layers)])
        peak_L = int(np.argmax(curve))
        peak_gap = float(curve[peak_L])
        results[v] = {"curve": curve, "peak_L": peak_L, "peak_gap": peak_gap}
        print(f"  {v:20s} | peak L{peak_L:02d} = {peak_gap:.4f} | class = {H1_CLASS[v]}")

    write_md(results, args.output_md)
    plot(results, args.output_fig, n_layers)


def write_md(results, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 4 H1 test: indexed-list format variation",
        "",
        "Test refined H1 hypothesis (pretraining co-occurrence shortcut):",
        "*\"input contains mark-like indexed region list → activates visual-grounding pathway\"*",
        "",
        "**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:",
        "- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)",
        "- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut",
        "",
        "## Result table (sorted by peak layer)",
        "",
        "| Variant | Format example | H1 class | Peak layer | Peak cosine gap |",
        "|---|---|---|---|---|",
    ]
    for v, r in sorted(results.items(), key=lambda x: x[1]["peak_L"]):
        lines.append(f"| {v} | `{DISPLAY[v]}` | {H1_CLASS[v]} | **L{r['peak_L']:02d}** | {r['peak_gap']:.4f} |")
    lines.append("")

    # Group by H1 class
    lines.append("## Grouped by H1 prediction")
    lines.append("")
    groups = {}
    for v, r in results.items():
        groups.setdefault(H1_CLASS[v], []).append((v, r))
    for cls in ["marks-like", "control (no integer)", "control (no list)", "AXTree-baseline"]:
        pairs = groups.get(cls, [])
        if not pairs:
            continue
        mean_L = np.mean([r["peak_L"] for _, r in pairs])
        lines.append(f"### {cls}  (mean peak L{mean_L:.0f})")
        lines.append("")
        for v, r in pairs:
            lines.append(f"- `{DISPLAY[v]}`: peak **L{r['peak_L']:02d}** = {r['peak_gap']:.4f}")
        lines.append("")

    # H1 verdict
    lines.append("## H1 verdict")
    lines.append("")
    marks_like_peaks = [r["peak_L"] for v, r in results.items() if H1_CLASS[v] == "marks-like"]
    control_peaks = [r["peak_L"] for v, r in results.items() if H1_CLASS[v].startswith("control")]
    dom_peak = results["dom"]["peak_L"]
    lines.append(f"- **6 marks-like variants**: mean peak layer = {np.mean(marks_like_peaks):.0f}, range L{min(marks_like_peaks):02d}-L{max(marks_like_peaks):02d}")
    lines.append(f"- **2 control variants** (no integer / no list): mean peak layer = {np.mean(control_peaks):.0f}, range L{min(control_peaks):02d}-L{max(control_peaks):02d}")
    lines.append(f"- **AXTree-DOM baseline**: peak L{dom_peak:02d}")
    lines.append("")
    if np.mean(marks_like_peaks) > 15 and np.mean(control_peaks) <= 10:
        lines.append("→ **H1 CONFIRMED**: marks-like variants peak at mid/late layers (shortcut triggered), controls peak early (no shortcut, behave like AXTree).")
    elif np.mean(marks_like_peaks) > 15 and np.mean(control_peaks) > 15:
        lines.append("→ **H1 PARTIAL**: marks-like AND controls all peak late — finding is broader than 'indexed list' (any text payload triggers).")
    elif np.mean(marks_like_peaks) <= 10:
        lines.append("→ **H1 REFUTED**: marks-like variants don't show delayed peak. Mechanism is more specific than indexed-list-presence.")
    else:
        lines.append("→ **H1 MIXED**: peak distribution doesn't fit simple binary prediction. Needs deeper analysis.")
    out.write_text("\n".join(lines) + "\n")
    print(f"summary → {out}")


def plot(results, out, n_layers):
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.35, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, :])

    # Panel (a): AXTree-DOM alone — highlight L04 peak
    r_dom = results["dom"]
    ax_a.plot(range(n_layers), r_dom["curve"], color="#888888", linewidth=2.5,
                label="AXTree-DOM (baseline)")
    ax_a.scatter([r_dom["peak_L"]], [r_dom["peak_gap"]], color="#888888", s=200,
                  marker="*", zorder=5, edgecolor="black", linewidth=1)
    ax_a.annotate(f"L{r_dom['peak_L']:02d} = {r_dom['peak_gap']:.4f}\n← PEAK",
                    xy=(r_dom["peak_L"], r_dom["peak_gap"]),
                    xytext=(8, r_dom["peak_gap"] - 0.005),
                    fontsize=10, color="black",
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1})
    ax_a.axvline(4, color="#888888", linestyle="--", alpha=0.6)
    ax_a.axvline(36, color="gray", linestyle=":", alpha=0.3)
    ax_a.set_xlabel("Layer index")
    ax_a.set_ylabel("Cosine gap to SoM baseline")
    ax_a.set_title("(a) AXTree hierarchical (DOM) — L04 PEAK\n"
                    "early fresh image-axis detection", fontsize=10, fontweight="bold")
    ax_a.grid(alpha=0.3)

    # Panel (b): 8 flat-list variants — all peak L36 (mostly)
    flat_variants = ["som_standard", "browser_use_at", "appagent_id", "tarsier_typed",
                       "plain_numbered", "xml_tagged", "hash_id_control", "plain_sentence"]
    flat_colors = plt.cm.viridis(np.linspace(0, 0.85, len(flat_variants)))
    for color, v in zip(flat_colors, flat_variants):
        r = results[v]
        ax_b.plot(range(n_layers), r["curve"], color=color, linewidth=1.5,
                    label=DISPLAY[v], alpha=0.85)
        ax_b.scatter([r["peak_L"]], [r["peak_gap"]], color=color, s=80,
                      marker="*", zorder=5, edgecolor="black", linewidth=0.5)
    ax_b.axvline(4, color="gray", linestyle=":", alpha=0.3)
    ax_b.axvline(17, color="gray", linestyle="--", alpha=0.4)
    ax_b.axvline(36, color="gray", linestyle="--", alpha=0.6)
    ax_b.text(33, 0.005, "L36\nlate peak", color="gray", fontsize=9)
    ax_b.set_xlabel("Layer index")
    ax_b.set_title("(b) 8 flat-list formats — L17/L36 PEAK\n"
                    "(SoM / Browser Use / AppAgent / Tarsier / numbered / XML\n"
                    "+ hash_id_control + plain_sentence)", fontsize=10, fontweight="bold")
    ax_b.legend(loc="lower right", fontsize=7, framealpha=0.85, ncol=2)
    ax_b.grid(alpha=0.3)

    # Panel (c): Bar chart of peak layer per variant
    ordered_variants = ["dom"] + flat_variants
    bar_colors = ["#888888"] + list(flat_colors)
    peak_layers = [results[v]["peak_L"] for v in ordered_variants]
    labels = [DISPLAY[v] for v in ordered_variants]

    bars = ax_c.barh(range(len(ordered_variants)), peak_layers, color=bar_colors, alpha=0.85,
                       edgecolor="black", linewidth=0.5)
    for i, (v, L) in enumerate(zip(ordered_variants, peak_layers)):
        ax_c.text(L + 0.5, i, f"L{L:02d}", va="center", fontsize=9,
                    fontweight="bold" if v == "dom" else "normal")
        # Annotate H1 prediction
        if v == "dom":
            note = "← unique: hierarchical defeats shortcut"
            ax_c.text(L + 4, i, note, va="center", fontsize=8.5, color="#444444", fontstyle="italic")
        elif v in ("hash_id_control", "plain_sentence"):
            note = "← control: still triggers!" if L > 10 else ""
            ax_c.text(L + 4, i, note, va="center", fontsize=8.5, color="#cc4444", fontstyle="italic")
    ax_c.set_yticks(range(len(ordered_variants)))
    ax_c.set_yticklabels(labels, fontsize=9)
    ax_c.invert_yaxis()
    ax_c.axvline(4, color="#888888", linestyle="--", alpha=0.5, label="L4 (AXTree-DOM peak)")
    ax_c.axvline(36, color="#4477aa", linestyle="--", alpha=0.5, label="L36 (flat-list peak)")
    ax_c.set_xlabel("Peak layer of image-axis cosine gap to SoM baseline")
    ax_c.set_title("(c) Peak layer per format — AXTree (L04) vs flat-list (L17/L36)",
                     fontsize=10, fontweight="bold")
    ax_c.set_xlim(-1, n_layers + 7)
    ax_c.grid(alpha=0.3, axis="x")
    ax_c.legend(loc="lower right", fontsize=8)

    fig.suptitle("H1 test: flat-list element representation triggers visual-grounding shortcut\n"
                  "(broader than indexed-list — hash IDs + plain sentence also trigger; only AXTree hierarchical defeats)",
                  fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"figure → {out}")


if __name__ == "__main__":
    main()
