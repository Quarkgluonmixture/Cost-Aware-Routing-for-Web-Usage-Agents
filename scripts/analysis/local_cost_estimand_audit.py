#!/usr/bin/env python3
"""What is the locally-served dollar figure actually measuring? — 2026-08-03

`EVIDENCE_LAYER_SUMMARY` §4a carried this as a limitation for weeks:

    "Absolute local cost. The per-token constant for the locally-served backbones was
     derived for a different accelerator than the runs were served on. Within-cell ratios
     are unaffected because it is a single multiplier; absolute dollar figures for B1/B2
     are uncalibrated."

Like the carbon entry beside it, that had never been checked. Checking it confirms the
first sentence, corrects the reasoning in the second, and **refutes its conclusion**.

Three findings, in order of how much they matter:

  1. The constant's provenance is in the config itself (`exp_v2_base.yaml:66-81`):
     "DGX Spark GB10 ~$0.20/hr, ~60 tok/s -> ~$0.00093/1k tokens". Every paper-grade run
     was served on `a100-jiaming-test`. The same file's ENERGY block was migrated to
     `hardware_profile: a100_pcie_40gb` (B-118, "dgx_spark profile retired per A100
     migration") and the COST block, four lines above it, was not.
  2. The assumed throughput is ~60 tok/s. Measured throughput is 248-551 tok/s, and it
     varies by mode -- which is precisely why a fixed price-per-token cannot preserve the
     relative ordering of modes that differ in how token-dense their steps are.
  3. So "within-cell ratios are unaffected" is wrong. It is not one multiplier, it is two
     (input and output), and more importantly the token basis is itself a proxy for a time
     basis: local inference is paid for in GPU-seconds. Pricing the same episodes by
     GPU-time instead of by token **changes which mode is cheapest in 2 of 4 local cells**.
     (The two-constants worry turns out to be harmless on its own -- output is 2-4% of
     tokens, so the output:input ratio can move 2x to 10x without reordering anything.
     Reported because it is the objection a reader will raise first.)

B0 is unaffected: it pays a real API bill in real dollars.

Usage
-----
    .venv/bin/python3 scripts/analysis/local_cost_estimand_audit.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.lib.run_registry import get_cells  # noqa: E402

OUT_MD = REPO / "docs/analysis/cross_sites/local_cost_estimand_audit.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/local_cost_estimand_audit.json"
LOCAL_BACKBONES = ("B1", "B2")
IN_PER_1K = 0.00093        # exp_v2_base.yaml metrics.cost.input_cost_per_1k
OUT_PER_1K = 0.00185       # exp_v2_base.yaml metrics.cost.output_cost_per_1k
DOLLARS_PER_HOUR = 0.20    # the $/hr the two constants above were derived FROM
ASSUMED_TOK_PER_S = 60.0   # the throughput assumed in that derivation
MAX_EP = 20


def _cell_rows() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for bl in LOCAL_BACKBONES:
        for site in ("classifieds", "reddit"):
            for cell in get_cells(baseline=bl, site=site):
                tin, tout, tms = [], [], []
                for f in sorted(Path(cell.episodes_dir).glob("*steps_v2.jsonl"))[:MAX_EP]:
                    for line in f.read_text().splitlines():
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        t = rec.get("tokens") or {}
                        lat = rec.get("latency_ms") or {}
                        if t.get("input") is None or lat.get("backend_infer") is None:
                            continue
                        tin.append(float(t["input"]))
                        tout.append(float(t.get("output") or 0))
                        tms.append(float(lat["backend_infer"]))
                if not tin:
                    continue
                mi, mo, mt = (statistics.fmean(tin), statistics.fmean(tout),
                              statistics.fmean(tms))
                out.setdefault(f"{bl}·{site}", {})[cell.mode] = {
                    "in_tok": mi, "out_tok": mo, "model_ms": mt,
                    "measured_tok_per_s": (mi + mo) / (mt / 1000.0),
                    "cost_by_token": mi / 1000 * IN_PER_1K + mo / 1000 * OUT_PER_1K,
                    "cost_by_gpu_time": mt / 1000 / 3600 * DOLLARS_PER_HOUR,
                    "output_share": mo / (mi + mo),
                }
    return out


def main() -> int:
    cells = _cell_rows()
    if not cells:
        raise SystemExit("no local-backbone cells found")

    flips, ratio_flips = [], []
    for cid, modes in cells.items():
        by_tok = min(modes, key=lambda m: modes[m]["cost_by_token"])
        by_time = min(modes, key=lambda m: modes[m]["cost_by_gpu_time"])
        if by_tok != by_time:
            flips.append(cid)
        # sensitivity of the ordering to the output:input price ratio alone
        winners = {min(modes, key=lambda m: modes[m]["in_tok"] + modes[m]["out_tok"] * r)
                   for r in (2, 3, 4, 6, 8, 10)}
        if len(winners) > 1:
            ratio_flips.append(cid)
        for m in modes:
            modes[m]["cheapest_by_token"] = (m == by_tok)
            modes[m]["cheapest_by_gpu_time"] = (m == by_time)
        cells[cid] = modes

    tps = [v["measured_tok_per_s"] for m in cells.values() for v in m.values()]
    shares = [v["output_share"] for m in cells.values() for v in m.values()]
    out = {
        "schema": "2026-08-03-local-cost-estimand-v1",
        "post_hoc_exploratory": True, "h10_eligible": False,
        "config_constants": {"input_per_1k": IN_PER_1K, "output_per_1k": OUT_PER_1K,
                             "derived_from_dollars_per_hour": DOLLARS_PER_HOUR,
                             "derived_from_assumed_tok_per_s": ASSUMED_TOK_PER_S,
                             "derived_for_device": "DGX Spark GB10",
                             "runs_served_on": "a100-jiaming-test (A100-PCIE-40GB)"},
        "measured_tok_per_s_min": min(tps), "measured_tok_per_s_max": max(tps),
        "throughput_underestimate_min": min(tps) / ASSUMED_TOK_PER_S,
        "throughput_underestimate_max": max(tps) / ASSUMED_TOK_PER_S,
        "output_share_min": min(shares), "output_share_max": max(shares),
        "n_cells": len(cells), "cheapest_flips_token_vs_time": flips,
        "cheapest_flips_on_output_ratio": ratio_flips,
        "cells": cells,
    }

    L = ["---", "type: analysis", "status: complete",
         "purpose: what the locally-served dollar figure measures, and whether its per-mode "
         "ordering survives a change of basis",
         "post_hoc_exploratory: true",
         "producer: scripts/analysis/local_cost_estimand_audit.py", "---", "",
         "# The locally-served dollar figure", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/local_cost_estimand_audit.py`", "",
         "`EVIDENCE_LAYER_SUMMARY` §4a said the local per-token constant \"was derived for a "
         "different accelerator\" and that \"within-cell ratios are unaffected because it is "
         "a single multiplier\". The first half is right and now has a citation; **the second "
         "half is wrong**.", "",
         "## 1. Where the constant came from, and where the runs ran", "",
         "`configs/exp_v2_base.yaml:66-81`, verbatim: *\"DGX Spark GB10: $4,699 / 3yr + 140W "
         "× $0.12/kWh ≈ $0.20/hr\"* and *\"DGX Spark GB10 ~$0.20/hr, ~60 tok/s → "
         f"~${IN_PER_1K}/1k tokens\"*. Every paper-grade condition was served on "
         "**`a100-jiaming-test`** (36/36 by `env_snapshot.json`).", "",
         "The same file's **energy** block *was* migrated — `hardware_profile: "
         "\"a100_pcie_40gb\"`, commented *\"B-118 (2026-05-15): canonical paper-grade rerun "
         "host = A100 Condenser VM; dgx_spark profile retired per A100 migration\"*. The "
         "**cost** block sits four lines above it and still says DGX Spark. One half of the "
         "hardware assumption was migrated and the other was not, in the same file.", "",
         f"A second inconsistency inside the same pipeline: the cost derivation assumes "
         f"**140 W**; the energy telemetry records **66.3 W** (`energy_carbon_audit`). Two "
         f"power figures, one pipeline.", "",
         "## 2. The assumed throughput is off by 4–9×", "",
         f"The derivation assumes **{ASSUMED_TOK_PER_S:.0f} tok/s**. Measured throughput on "
         f"the actual host is **{out['measured_tok_per_s_min']:.0f}–"
         f"{out['measured_tok_per_s_max']:.0f} tok/s** "
         f"({out['throughput_underestimate_min']:.1f}–"
         f"{out['throughput_underestimate_max']:.1f}× the assumption), **and it varies by "
         "mode** — which is exactly why a fixed price-per-token cannot preserve an ordering "
         "over modes whose steps differ in token density.", "",
         "| cell | mode | tokens/step | model ms/step | measured tok/s | $ by token | $ by GPU-time |",
         "|---|---|---|---|---|---|---|"]
    for cid, modes in cells.items():
        for m, v in modes.items():
            tok_mark = " ⬅" if v["cheapest_by_token"] else ""
            time_mark = " ⬅" if v["cheapest_by_gpu_time"] else ""
            L.append(f"| `{cid}` | {m} | {v['in_tok'] + v['out_tok']:,.0f} | "
                     f"{v['model_ms']:,.0f} | {v['measured_tok_per_s']:.0f} | "
                     f"{v['cost_by_token']:.6f}{tok_mark} | "
                     f"{v['cost_by_gpu_time']:.6f}{time_mark} |")
    L += ["", "⬅ marks the cheapest mode under each basis. `$ by GPU-time` prices the same "
          f"episodes at the same ${DOLLARS_PER_HOUR:.2f}/hr the token constant was derived "
          "from, applied to measured model time instead of to a token count.", ""]

    L += ["## 3. The ordering does not survive the change of basis", "",
          f"**The cheapest mode changes in {len(flips)} of {out['n_cells']} local cells** "
          f"({', '.join('`' + c + '`' for c in flips) or 'none'}) between the token basis and "
          "the GPU-time basis. That is a *within-cell* reordering, which is what §4a said "
          "could not happen.", "",
          "Which basis is right depends on the claim. A local deployment rents or owns the "
          "accelerator by the second, so GPU-time is the deployment-facing quantity; the "
          "token basis is a proxy for it that the config's own comment derives *from* it via "
          "an assumed throughput. Neither is reported here as correct — the point is that the "
          "per-mode cost ordering on locally-served backbones is **estimand-dependent**, the "
          "same conclusion `latency_decomposition` reached for latency and "
          "`outcome_efficiency` reached for the denominator.", "",
          "## 4. One worry that turns out to be harmless", "",
          f"The constant is really two (input {IN_PER_1K}, output {OUT_PER_1K}), so a reader "
          "will ask whether the output:input price ratio drives the ordering. It does not: "
          f"output tokens are **{100*out['output_share_min']:.1f}–"
          f"{100*out['output_share_max']:.1f}%** of the total, and sweeping the ratio from 2× "
          f"to 10× reorders **{len(ratio_flips)} of {out['n_cells']}** cells. §4a's "
          "\"effectively a single multiplier\" reasoning is therefore right on this point — "
          "just not sufficient for its conclusion, because the token basis itself is the "
          "problem.", "",
          "## 5. Scope", "",
          "**B0 is unaffected.** It pays a real API bill at published per-token rates, so its "
          "dollars are dollars. This page is about B1 and B2 only, where a \"cost\" is a "
          "modelling choice rather than an invoice — and where, per `cost_per_mode`, those "
          "figures were already flagged as belonging to a different class from B0's and never "
          "combined into one ratio.", ""]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[local_cost_estimand] {out['n_cells']} local cells; measured "
          f"{out['measured_tok_per_s_min']:.0f}-{out['measured_tok_per_s_max']:.0f} tok/s vs "
          f"assumed {ASSUMED_TOK_PER_S:.0f}; cheapest-mode flips token-vs-time in "
          f"{len(flips)}/{out['n_cells']}, on output-ratio in {len(ratio_flips)}/{out['n_cells']}")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
