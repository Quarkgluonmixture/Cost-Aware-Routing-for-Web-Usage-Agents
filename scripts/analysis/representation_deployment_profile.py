#!/usr/bin/env python3
"""Two deployment properties of a representation that success rate does not show.

A team choosing an observation representation reads success rate and price. Neither
answers the two questions that actually decide the operational cost of running it:

  1. WHEN IT FAILS, CAN YOU TELL WHY?  A failure that lands in a named bucket
     (committed too early, looping on search, misgrounded the element) is triage you
     can act on. A failure that only says "ran out of steps" or "errored" is not. The
     ratio between those is an on-call cost, and it differs by representation.

  2. WHAT DOES THE TAIL COST, NOT THE MEAN?  Whether a representation fits a context
     window, and whether it needs truncation, is decided by p95/p99 input tokens, not
     the mean. §449.1's finding that cost is driven by STEP COUNT rather than per-step
     token volume makes the per-step tail an independent quantity worth its own look.

Neither needs a new run. (1) reads the existing `failure_modes_per_cell.json` product;
(2) reads the per-step `tokens.input` already recorded in every step_v2 record.

*** DIAGNOSABILITY, DEFINED ***

`failure_modes_per_cell.json` maps fine-grained reason buckets onto a 5-bucket paper
taxonomy. Two of its buckets name no mechanism:
  - `max-steps-other`  = the generic `fail_max_steps`, i.e. "the budget ran out and no
                         rule fired"
  - `error/noise`      = env error / parse error / benchmark noise / summary error
Everything else names a mechanism. Diagnosability is the share of FAILURES that land in
a mechanism bucket. It is a property of this ruleset, not a law of nature: a richer
ruleset would move failures out of `max-steps-other`, so the number is a floor on how
diagnosable a representation is, and comparisons across modes are only valid because
every mode is scored by the SAME ruleset.

*** SCOPE ***

- Diagnosability: whatever cells the product carries (35 as of 2026-08-12).
- Token tail: B0 only, by default. B0 is the paying backbone, and per-step token counts
  from a hosted endpoint are the provider's own accounting; B1/B2 count locally and the
  two are not the same instrument (cf. the six-fold tokenizer spread across providers in
  笔记 §456.1). Pass --cells to widen it deliberately.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LOG = logging.getLogger("depprofile")
FAILURE_PRODUCT = REPO / "docs/analysis/cross_sites/failure_modes_per_cell.json"
PHASE1 = REPO / "results/visualwebarena/phase1"

# Buckets that name no mechanism -- see the docstring.
UNDIAGNOSED = ("max-steps-other", "error/noise")
MODE_DIRS = {"DOM": "dom", "SoM": "som", "Vision": "vision",
             "P-text": "phantom_text", "P-prompt": "phantom_prompt", "P-SoM": "phantom_som"}


def diagnosability() -> list[dict]:
    if not FAILURE_PRODUCT.is_file():
        raise RuntimeError(f"failure-mode product absent: {FAILURE_PRODUCT}")
    prod = json.loads(FAILURE_PRODUCT.read_text())
    rows = []
    for key, c in sorted(prod["cells"].items()):
        failed = c.get("failed_count") or 0
        if not failed:
            continue
        buckets = c.get("buckets", {})
        undiag = sum(buckets.get(b, {}).get("count", 0) for b in UNDIAGNOSED)
        named = failed - undiag
        rows.append({
            "cell": key, "baseline": c["baseline"], "site": c["site"], "mode": c["mode"],
            "failed": failed,
            "named_mechanism": named,
            "undiagnosed": undiag,
            "diagnosable_pct": 100 * named / failed,
            "top_bucket": max(
                ((b, v.get("count", 0)) for b, v in buckets.items()),
                key=lambda kv: kv[1], default=("-", 0))[0],
        })
    return rows


def token_tail(cells: list[tuple[str, str]]) -> list[dict]:
    """p50/p95/p99/max of per-step `tokens.input`, per (baseline, site, mode)."""
    rows = []
    for baseline, site in cells:
        for mode_label, mode_dir in MODE_DIRS.items():
            pat = f"{baseline}_{mode_dir}_{site}_*"
            runs = sorted(p for p in PHASE1.glob(pat)
                          if p.is_dir() and "ABORTED" not in p.name)
            if not runs:
                LOG.warning("no run dir for %s", pat)
                continue
            vals: list[int] = []
            n_steps = n_missing = 0
            for run in runs:
                for jf in run.glob("*/episodes/*_steps_v2.jsonl"):
                    for line in jf.read_text(errors="replace").splitlines():
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        tok = rec.get("tokens") or {}
                        v = tok.get("input")
                        n_steps += 1
                        if isinstance(v, (int, float)) and v > 0:
                            vals.append(int(v))
                        else:
                            n_missing += 1
            if not vals:
                LOG.warning("%s: no tokens.input values in %d steps -- skipping rather "
                            "than reporting zeros", pat, n_steps)
                continue
            vals.sort()

            def q(p):
                if len(vals) == 1:
                    return vals[0]
                i = min(len(vals) - 1, int(round(p * (len(vals) - 1))))
                return vals[i]

            rows.append({
                "baseline": baseline, "site": site, "mode": mode_label,
                "n_steps_with_tokens": len(vals),
                "n_steps_missing_tokens": n_missing,
                "p50": q(0.50), "p95": q(0.95), "p99": q(0.99), "max": vals[-1],
                "mean": round(statistics.fmean(vals), 1),
                "tail_ratio_p99_over_p50": round(q(0.99) / q(0.50), 2) if q(0.50) else None,
                "runs": [r.name for r in runs],
            })
    return rows


def render_md(diag: list[dict], tail: list[dict], cells) -> str:
    L = ["---", "type: analysis", "status: complete",
         "purpose: two deployment properties of an observation representation that success "
         "rate does not show -- whether its failures are diagnosable, and what its token "
         "tail costs",
         "scope_warning: diagnosability is defined against THIS P-rule ruleset and is a "
         "floor, not a law -- a richer ruleset moves failures out of the generic bucket. "
         "Cross-mode comparison is valid only because every mode is scored by the same "
         "ruleset. The token tail is B0-only by default: per-step counts from a hosted "
         "endpoint are the provider's accounting, and B1/B2 count locally.",
         "producer: scripts/analysis/representation_deployment_profile.py", "---", "",
         "# Deployment profile of a representation", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/representation_deployment_profile.py`", "",
         "## 1. When it fails, can you tell why?", "",
         "A failure in a named bucket (committed early, search loop, misgrounded element, "
         "missing context) is triage an on-call engineer can act on. `max-steps-other` "
         "(\"budget ran out, no rule fired\") and `error/noise` name no mechanism. "
         "Diagnosability below is the share of failures that name one.", "",
         "| cell | mode | failures | named mechanism | undiagnosed | **diagnosable** | biggest bucket |",
         "|---|---|---:|---:|---:|---:|---|"]
    for r in sorted(diag, key=lambda r: (r["baseline"], r["site"], -r["diagnosable_pct"])):
        L.append(f"| `{r['baseline']}/{r['site']}` | {r['mode']} | {r['failed']} | "
                 f"{r['named_mechanism']} | {r['undiagnosed']} | "
                 f"**{r['diagnosable_pct']:.1f}%** | {r['top_bucket']} |")
    spread = {}
    for r in diag:
        spread.setdefault((r["baseline"], r["site"]), []).append(r["diagnosable_pct"])
    worst = min(diag, key=lambda r: r["diagnosable_pct"])
    best = max(diag, key=lambda r: r["diagnosable_pct"])
    L += ["", f"Within a cell the spread across modes reaches "
          f"{max(max(v) - min(v) for v in spread.values()):.1f} points, and across the "
          f"whole table it runs from **{worst['diagnosable_pct']:.1f}%** "
          f"(`{worst['baseline']}/{worst['site']}` {worst['mode']}) to "
          f"**{best['diagnosable_pct']:.1f}%** (`{best['baseline']}/{best['site']}` "
          f"{best['mode']}). Two representations with the same success rate can therefore "
          "differ substantially in how much of their failure mass is actionable -- a cost "
          "that lands on the operator, not on the benchmark scoreboard.", ""]
    if tail:
        L += ["## 2. What does the tail cost, not the mean?", "",
              "Per-step `tokens.input` as the provider counted it. Context-window fit and "
              "the decision to truncate are set by the tail, and §449.1 already showed that "
              "episode cost is driven by step count rather than per-step volume -- so the "
              "per-step tail is an independent quantity, not a restatement of price.", "",
              "| cell | mode | runs | steps | p50 | p95 | p99 | max | p99/p50 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sorted(tail, key=lambda r: (r["baseline"], r["site"], -r["p99"])):
            L.append(f"| `{r['baseline']}/{r['site']}` | {r['mode']} | {len(r['runs'])} | "
                     f"{r['n_steps_with_tokens']} | {r['p50']} | {r['p95']} | "
                     f"**{r['p99']}** | {r['max']} | {r['tail_ratio_p99_over_p50']} |")
        multi = [r for r in tail if len(r["runs"]) > 1]
        if multi:
            L += ["", "⚠️ **Pooled over runs where more than one exists.** "
                  + "; ".join(f"`{r['mode']}` on `{r['baseline']}/{r['site']}` pools "
                              f"{len(r['runs'])}" for r in multi)
                  + " — a same-condition replicate lives under `phase1/` for those, so their "
                  "step counts are correspondingly larger. Quantiles of one condition pooled "
                  "across its own reruns are still quantiles of that condition, but the step "
                  "counts are not comparable across rows without this column."]
        L += ["", "⚠️ `tokens.input` is the TOTAL input; the `input_text` / `input_image` "
              "split is null on B0 (the hosted endpoint does not itemise it), so a "
              "screenshot-bearing mode's tail cannot be attributed between text and image "
              "here. 台账 §260 estimated the image share from a som-vs-dom median "
              "difference instead.", ""]
    L += ["## What this is for", "",
          "Both columns are properties a single-mode deployment cannot measure about "
          "itself: diagnosability needs a shared ruleset applied across representations, "
          "and a token tail is only interpretable next to the alternatives'. They belong "
          "with the other deployment-facing numbers -- fusion's premium clearing the rerun "
          "threshold in 0 of 8 cells (`fusion_premium.md`), the cost of unstable element "
          "ids, and the abstention frontier (`abstention_learnability.md`).", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="B0:classifieds,B0:reddit",
                    help="comma-separated baseline:site for the token tail")
    ap.add_argument("--skip-tail", action="store_true", help="diagnosability only (fast)")
    ap.add_argument("--out-dir", default="docs/analysis/cross_sites")
    ap.add_argument("--from-json", default=None,
                    help="re-render the report from an existing product (no recompute)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.from_json:
        d = json.loads(Path(args.from_json).read_text())
        out = Path(args.from_json).with_suffix(".md")
        out.write_text(render_md(d["diagnosability"], d.get("token_tail") or [], []))
        print(f"re-rendered {out} from {args.from_json} (no recomputation)")
        return 0

    diag = diagnosability()
    print(f"\n=== 1. failure diagnosability ({len(diag)} cell x mode) ===")
    print(f"{'cell':<22} {'mode':<9} {'failed':>7} {'undiag':>7} {'diagnosable':>12}  top bucket")
    for r in sorted(diag, key=lambda r: (r["baseline"], r["site"], -r["diagnosable_pct"])):
        print(f"{r['baseline'] + '/' + r['site']:<22} {r['mode']:<9} {r['failed']:>7} "
              f"{r['undiagnosed']:>7} {r['diagnosable_pct']:>11.1f}%  {r['top_bucket']}")

    cells = []
    if not args.skip_tail:
        for tok in args.cells.split(","):
            b, s = tok.split(":")
            cells.append((b.strip(), s.strip()))
    tail = token_tail(cells) if cells else []
    if tail:
        print(f"\n=== 2. per-step input-token tail ===")
        print(f"{'cell':<22} {'mode':<9} {'steps':>7} {'p50':>7} {'p95':>7} {'p99':>7} "
              f"{'max':>7} {'p99/p50':>8}")
        for r in sorted(tail, key=lambda r: (r["baseline"], r["site"], -r["p99"])):
            print(f"{r['baseline'] + '/' + r['site']:<22} {r['mode']:<9} "
                  f"{r['n_steps_with_tokens']:>7} {r['p50']:>7} {r['p95']:>7} "
                  f"{r['p99']:>7} {r['max']:>7} {r['tail_ratio_p99_over_p50']:>8}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "representation_deployment_profile.json"
    js.write_text(json.dumps({"undiagnosed_buckets": list(UNDIAGNOSED),
                              "diagnosability": diag, "token_tail": tail}, indent=2))
    md = js.with_suffix(".md")
    md.write_text(render_md(diag, tail, cells))
    print(f"\nwrote {js}\nwrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
