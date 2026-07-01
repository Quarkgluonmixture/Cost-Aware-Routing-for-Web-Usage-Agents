#!/usr/bin/env python3
"""Verdict-day slot sheet — read-only formatter over canonical gate artifacts.

Emits ONE markdown sheet aggregating everything aaai27_main.md slot-filling needs:
  (A) gate verdicts verbatim from producers  (B) branch suggestion for
  branch_prewrites_s1_abstract.md  (C) «slot» values  (D/E/F) Tables 2/3/4
  regenerated  (G) checklist reminders.

DISCIPLINE: this script contains NO estimand logic. Every verdict boolean and
number is copied verbatim from producer output (aggregate_phase1_full_prereg_decision.py,
aggregate_h10_pareto.py, sr_per_mode aggregator, fig0c). Missing fields print as
MISSING — never recomputed here. Not on the fire import path (pure reader).

Usage:  .venv/bin/python3 scripts/analysis/verdict_day_slotsheet.py [--out FILE]
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECISION = ROOT / "results/phantom_paper/phase1_full_prereg_decision.json"
H10 = ROOT / "results/phantom_paper/h10_pareto_verdict.json"
SR = ROOT / "docs/analysis/cross_sites/sr_per_mode.json"
FIG0C = ROOT / "results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv"

MODE_ORDER = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]


def g(d, *keys, default="MISSING"):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def f2(x):
    return f"{x:+.2f}" if isinstance(x, (int, float)) else str(x)


def scalars(d):
    return {k: v for k, v in d.items() if not isinstance(v, (dict, list))} if isinstance(d, dict) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dec = json.loads(DECISION.read_text())
    h10 = json.loads(H10.read_text())
    sr = json.loads(SR.read_text())
    fig0c = list(csv.DictReader(FIG0C.open())) if FIG0C.exists() else []

    L = []
    add = L.append
    add(f"# Verdict-day slot sheet  (generated from artifacts captured_at={g(dec,'captured_at')})")
    add("")
    add("> 所有数字/verdict 逐字来自 producer 产物 — 本 sheet 之外禁止手抄任何 gate 数字进 draft。")
    add("")

    # ---- A. gate verdicts ------------------------------------------------
    add("## A. Gate verdicts (verbatim)")
    gate_status = g(dec, "gate_status")
    add(f"- **gate_status = `{gate_status}`**" + ("  ⚠️ PARTIAL — interim only, NOT a verdict; 不得 splice 分支" if gate_status != "COMPLETE" else ""))
    boot = g(dec, "pooled_h1_bootstrap", default={})
    fe = g(dec, "pooled_h1_fe", default={})
    het = g(dec, "h1_heterogeneity", default={})
    add(f"- **H1 PRIMARY (bootstrap percentile, prereg B-1009/B-1303)**: gate_passed_bootstrap = "
        f"**{g(boot,'gate_passed_bootstrap')}** | k={g(boot,'k_cells')} | p_one_sided_bootstrap={g(boot,'p_one_sided_bootstrap')} "
        f"| percentile CI95=[{f2(g(boot,'ci95_lo_pp_bootstrap'))}, {f2(g(boot,'ci95_hi_pp_bootstrap'))}]pp "
        f"| bootstrap median={f2(g(boot,'theta_fe_bootstrap_median_pp'))}pp")
    add(f"- H1 point estimate (FE pool): θ_FE={f2(g(fe,'theta_FE_pp'))}pp | normal CI95=[{f2(g(fe,'ci95_FE_lo_pp'))}, {f2(g(fe,'ci95_FE_hi_pp'))}]pp "
        f"| transparency z={g(fe,'z_one_sided')} p_norm={g(fe,'p_one_sided')} (appendix column, NOT the gate)")
    add(f"- H1 heterogeneity: I²={g(het,'I_squared_pct')}% | cap_at_R3={g(het,'heterogeneity_cap_at_r3')} (threshold {g(het,'heterogeneity_threshold_pct')}%)")
    for ax in ("h3_axis1_pooled_fe", "h3_axis2_pooled_fe"):
        d = g(dec, ax, default={})
        add(f"- **{ax}**: " + " | ".join(f"{k}={f2(v) if isinstance(v,float) else v}" for k, v in scalars(d).items()))
    add("- **H2(a) per-cell** (ANY cell median ratio > 1.20× ⇒ falsified):")
    for c in dec.get("per_cell", []):
        h2 = c.get("h2a", {})
        add(f"    - {c.get('baseline')}·{c.get('site')}: " + " | ".join(f"{k}={v}" for k, v in scalars(h2).items()))
    if dec.get("skipped_cells"):
        add("- ⚠️ skipped_cells: " + "; ".join(f"{c.get('baseline')}·{c.get('site')} ({c.get('reason')})" for c in dec["skipped_cells"]))
    op = g(h10, "operational_deployment_gate", default={})
    add("- **H10 operational gate**: " + " | ".join(f"{k}={v}" for k, v in scalars(op).items()))
    add("")

    # ---- B. branch suggestion -------------------------------------------
    add("## B. Branch suggestion (branch_prewrites_s1_abstract.md) — 人工对照 prereg §2.5 + Amendment 02 确认")
    h1_pass = g(boot, "gate_passed_bootstrap")
    ax1_pass = g(dec, "h3_axis1_pooled_fe", "passed")
    ax2_pass = g(dec, "h3_axis2_pooled_fe", "passed")
    if gate_status != "COMPLETE":
        add("- ⚠️ gate_status ≠ COMPLETE → **不选支**。本 sheet 仅供预演。")
    if h1_pass is True:
        add("- → **Branch A** (H1-PASS)。R-tier 由 H3 两轴 + B2 claim-tier gate 定 (prereg §2.5 step 8)。")
    elif h1_pass is False:
        add(f"- H1 FAIL; H3 axis1_pass={ax1_pass} axis2_pass={ax2_pass}")
        add("- 两轴均 pass → **Branch B** (Route C'-S)。否则回 Amendment 02 ladder (C'-R / F, 未预写)。")
    else:
        add("- H1 verdict MISSING。")
    if g(het, "heterogeneity_cap_at_r3") is True:
        add("- ⚠️ I² cap 触发 → framing 最高 R3 (cap-only)。")
    add("")

    # ---- C. slot values ---------------------------------------------------
    add("## C. «slot» values (branch_prewrites §槽位约定)")
    add(f"| «THETA» | {f2(g(fe,'theta_FE_pp'))} | pooled_h1_fe.theta_FE_pp (point estimate) |")
    add(f"| «CI_LO» «CI_HI» | {f2(g(boot,'ci95_lo_pp_bootstrap'))} , {f2(g(boot,'ci95_hi_pp_bootstrap'))} | bootstrap percentile CI (PRIMARY per B-1009) |")
    add(f"| «P_BOOT» | {g(boot,'p_one_sided_bootstrap')} | pooled_h1_bootstrap.p_one_sided_bootstrap |")
    add(f"| «K» | {g(boot,'k_cells')} | 若 <6 → advisor 预案(a) k<6 透明披露句必须同时进 §4/§8 |")
    ax1, ax2 = g(dec, "h3_axis1_pooled_fe", default={}), g(dec, "h3_axis2_pooled_fe", default={})
    add(f"| «AX1» | {f2(g(ax1,'theta_FE_pp'))} (k={g(ax1,'k_cells')}) | h3_axis1_pooled_fe |")
    add(f"| «AX2» | {f2(g(ax2,'theta_FE_pp'))} (k={g(ax2,'k_cells')}) | h3_axis2_pooled_fe |")
    add("| «UNIQ_CLS» «UNIQ_RED» | 见 per_cell h1 unique-pass 字段 / drop-one digest | canonical [A] 计数, 禁用 archive 7+6 |")
    add("")

    # ---- D. Table 2 -------------------------------------------------------
    add("## D. Table 2 regen (sr_per_mode.json)")
    rows = sr.get("summary_table", [])
    cells = {}
    for r in rows:
        cells.setdefault((r["site"], r["baseline"]), {})[r["mode"]] = r
    add("| cell | " + " | ".join(MODE_ORDER) + " |")
    add("|---|" + "---|" * len(MODE_ORDER))
    for (site, base), modes in sorted(cells.items()):
        vals = []
        for m in MODE_ORDER:
            r = modes.get(m)
            if r is None:
                vals.append("⟨TBD⟩")
            else:
                flag = "" if r.get("complete") else " ⚠️partial"
                vals.append(f"{r.get('sr_pct', float('nan')):.1f}{flag}")
        add(f"| {base}·{site} | " + " | ".join(vals) + " |")
    add("")

    # ---- E. Table 3 -------------------------------------------------------
    add("## E. Table 3 regen (fig0c drop-one, per panel)")
    add("| panel | mode | drop-one pp | CI95 | flag |")
    add("|---|---|---|---|---|")
    panel_nmodes = {}
    for r in fig0c:
        panel_nmodes[r["site_baseline"]] = panel_nmodes.get(r["site_baseline"], 0) + 1
    for r in fig0c:
        warn = ""
        if r.get("is_partial") == "True":
            warn += " ⚠️episode-partial"
        if panel_nmodes[r["site_baseline"]] < 6:
            warn += f" ⚠️{panel_nmodes[r['site_baseline']]}-mode portfolio ≠ 6-mode 定义, draft 禁引 (NUMBERS_TODO §0)"
        add(f"| {r['site_baseline']} | {r['mode']} | {float(r['drop_one_loss_pp']):+.2f} | [{float(r['ci95_low_pp']):.2f}, {float(r['ci95_high_pp']):.2f}] |{warn} |")
    add("")

    # ---- F. Table 4 -------------------------------------------------------
    add("## F. Table 4 regen (h10 per_cell)")
    for cid, c in (g(h10, "per_cell", default={}) or {}).items():
        add(f"- {cid}: " + " | ".join(f"{k}={v}" for k, v in scalars(c).items()))
    add("")

    # ---- G. checklist -----------------------------------------------------
    add("## G. Post-splice checklist (aaai27_main items 7-9)")
    add('1. banned grep (必须 0 hits): `grep -nE "image-free|image-off|no image tokens|text-only cost|both Qwen cells|most of the.*mass" docs/checkpoints/paper_drafts/aaai27/aaai27_main.md`')
    add("2. 残留槽位检查 (必须 0 hits): `grep -nE '<(H1|H3|H10)-VERDICT>|R-CONDITIONAL|«|⟨TBD⟩' aaai27_main.md`")
    add("3. 词数: strip HTML comments 后 wc -w (strip 命令别贴进 MD comment 块 — checklist item 7 教训)")
    add("4. [P]→[A] provenance lift 核对 (Table 2 注)")
    add("5. /stress + codex + gemini chain (CLAUDE.md auto-trigger)")

    out = "\n".join(L) + "\n"
    if args.out:
        args.out.write_text(out)
        print(f"written: {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
