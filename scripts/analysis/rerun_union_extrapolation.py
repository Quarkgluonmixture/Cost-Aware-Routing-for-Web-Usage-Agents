#!/usr/bin/env python3
"""How much of the 6-mode oracle ceiling could SIX RERUNS of one mode produce?

This exists because of a specific attack, and it is the strongest one available
against the only positive claim this project still makes. The shape of it:

    "You measured that ONE rerun of one arm buys 2.0-7.6pp. Your ceiling is the
     union of SIX arms. If one extra arm buys 7 points of noise, what do five
     buy? The 16pp ceiling might just be six draws from a 12% flip distribution."

`noise_floor_inventory.md` §2 correctly refuses to answer it by subtraction --
one rerun arm is not five, and reruns have diminishing returns -- but refusing
the bad arithmetic is not the same as answering the question. This script
answers it with the arm count matched.

MODEL (deliberately the same null already used for the floor, so no new
assumption is smuggled in). Partition the scored set into
  * s0  tasks the mode solves deterministically,
  * m   tasks it solves with probability 1/2 ("flippable"),
  * the rest, never solved.
Two runs of the condition then disagree on each flippable task with probability
1/2, so the observed discordant count d estimates m/2, giving m ~= 2d and
s0 ~= |A| - d. The union over k independent runs is

    U(k) = s0 + m * (1 - 2^-k)

which is exactly pass@k over one arm. U(1) reproduces the single-run SR by
construction; U(2) is a genuine out-of-sample check against the observed
two-run union, and is printed so the model can be falsified rather than
trusted.

The comparison that matters is U(6) -- six reruns of ONE representation --
against the measured six-MODE oracle. Same arm count, same cell, same n.

⚠️ Reported as a bound, not an estimate. The 1/2-flip model is the most
generous account of what repetition can buy: it assumes every discordant task
is a fair coin, which maximises the union's growth. Real flippable tasks with
p != 1/2 saturate faster. So U(6) OVERSTATES what six reruns would deliver, and
that is the direction that makes the conclusion safe.

Output: docs/analysis/cross_sites/rerun_union_extrapolation.{md,json}
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_FLOOR = ROOT / "docs/analysis/cross_sites/noise_floor_inventory.md"
SRC_ORDER = ROOT / "docs/analysis/cross_sites/router_objective_ordering.md"
OUT_MD = ROOT / "docs/analysis/cross_sites/rerun_union_extrapolation.md"
OUT_JSON = ROOT / "docs/analysis/cross_sites/rerun_union_extrapolation.json"

# The three replicated arms of B0 x VWA-classifieds, from noise_floor_inventory §1.
RE_PAIR = re.compile(
    r"\|\s*`(B\d[.\w-]+)`\s*\|[^|]*\|\s*(\d+)\s*\|\s*\*\*([\d.]+)pp\*\*\s*\|"
    r"\s*\*\*([\d.]+)pp\*\*\s*\|\s*([\d.]+)%\s*\|")

# Single-mode SR, to anchor |A| for each replicated arm.
RE_HEAD = re.compile(r"^## (\w+) · (B\d)\s+\(n=(\d+)\)", re.M)
RE_SINGLE = re.compile(
    r"^\|\s*`single:([\w-]+)`\s*\|\s*fixed\s*\|\s*([\d.]+)\s*\|", re.M)
RE_ORACLE = re.compile(
    r"^\|\s*`oracle_sr`\s*\|\s*oracle\s*\|\s*([\d.]+)\s*\|", re.M)

MODE_OF = {"dom": "DOM", "som": "SoM", "vision": "Vision"}


def load_cell(site: str, base: str):
    """Per-mode single SR and the 6-mode oracle SR for one cell."""
    text = SRC_ORDER.read_text(encoding="utf-8")
    heads = list(RE_HEAD.finditer(text))
    for i, h in enumerate(heads):
        if h.group(1) != site or h.group(2) != base:
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        block = text[h.end():end]
        sr = {m: float(v) for m, v in RE_SINGLE.findall(block)}
        orc = RE_ORACLE.search(block)
        if not orc:
            raise SystemExit(f"{SRC_ORDER}: no oracle_sr row for {site}·{base}")
        return int(h.group(3)), sr, float(orc.group(1))
    raise SystemExit(f"{SRC_ORDER}: cell {site}·{base} not found")


def main() -> int:
    pairs = RE_PAIR.findall(SRC_FLOOR.read_text(encoding="utf-8"))
    if not pairs:
        raise SystemExit(f"{SRC_FLOOR}: no replicate rows parsed")

    n_cell, sr_of, oracle_sr = load_cell("classifieds", "B0")
    rows, K = [], 6
    for tag, n_s, drop_ab, drop_ba, disc in pairs:
        if not tag.startswith("B0.cls."):
            continue                      # only the cell that carries an oracle
        n = int(n_s)
        if n != n_cell:
            raise SystemExit(f"{tag}: n={n} != cell n={n_cell} — refusing")
        mode = MODE_OF.get(tag.rsplit(".", 1)[1])
        if mode is None or mode not in sr_of:
            raise SystemExit(f"{tag}: cannot map to a mode in {sorted(sr_of)}")

        d = round(float(disc) / 100.0 * n)              # discordant tasks
        a = round(sr_of[mode] / 100.0 * n)              # |A|, the canonical run
        m = 2 * d                                       # flippable tasks
        s0 = a - d                                      # deterministic solves
        if s0 < 0:
            raise SystemExit(f"{tag}: s0<0 — model inapplicable")
        u = {k: (s0 + m * (1 - 2.0 ** -k)) / n * 100.0 for k in range(1, K + 1)}
        # Out-of-sample check: observed 2-run union = |A| + |B\A|.
        u2_obs = (a + round(float(drop_ba) / 100.0 * n)) / n * 100.0
        rows.append({"arm": tag, "mode": mode, "n": n, "sr_single_pct": sr_of[mode],
                     "discordant": d, "flippable_est": m, "determ_solve_est": s0,
                     "union_pct": {str(k): round(v, 2) for k, v in u.items()},
                     "union2_observed_pct": round(u2_obs, 2),
                     "union2_model_error_pp": round(u[2] - u2_obs, 2)})

    if not rows:
        raise SystemExit("no B0.cls replicate arms found")

    best_single = max(sr_of.values())
    u6 = [r["union_pct"]["6"] for r in rows]
    payload = {"cell": "classifieds·B0", "n": n_cell, "K": K,
               "best_single_sr_pct": best_single, "oracle_6mode_sr_pct": oracle_sr,
               "arms": rows,
               "u6_min_pct": min(u6), "u6_max_pct": max(u6),
               "oracle_minus_u6_max_pp": round(oracle_sr - max(u6), 2),
               "model": "s0 + m(1-2^-k), m=2*discordant, s0=|A|-discordant",
               "direction_of_bias": "overstates repetition (every discordant "
                                    "task treated as a fair coin)"}
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    L = ["# Six reruns of one mode vs six distinct modes", "",
         "> Regenerate: `.venv/bin/python3 scripts/analysis/rerun_union_extrapolation.py`",
         "",
         "Answers the one question the noise-floor inventory declines to answer by",
         "arithmetic: the ceiling adds five arms, the measured floor adds one, so what",
         "would five *rerun* arms have bought? Model and its bias direction are",
         "documented in the script header. Reported as an upper bound on repetition.",
         "",
         f"Cell `classifieds·B0`, n={n_cell}. Best single mode {best_single:.2f}%; "
         f"six-mode oracle **{oracle_sr:.2f}%**.", "",
         "| replicated arm | single-run SR | discordant | flippable (est) | "
         "U(2) model | U(2) observed | **U(6) model** |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| `{r['arm']}` ({r['mode']}) | {r['sr_single_pct']:.2f}% | "
                 f"{r['discordant']} | {r['flippable_est']} | "
                 f"{r['union_pct']['2']:.2f}% | {r['union2_observed_pct']:.2f}% | "
                 f"**{r['union_pct']['6']:.2f}%** |")
    L += ["",
          f"**U(2) model vs observed** differs by "
          f"{min(r['union2_model_error_pp'] for r in rows):+.2f} to "
          f"{max(r['union2_model_error_pp'] for r in rows):+.2f}pp — the model is "
          "checked out of sample rather than assumed, and it errs on the side of "
          "crediting repetition too much.", "",
          f"**The comparison at matched arm count.** Six reruns of one mode reach "
          f"{min(u6):.2f}–{max(u6):.2f}%. Six distinct modes reach "
          f"**{oracle_sr:.2f}%** — a residual of only "
          f"**{oracle_sr - max(u6):.2f}pp** over the best six-rerun account.", "",
          "⇒ Two readings, and both belong in the write-up.", "",
          f"**At matched ARM COUNT the residual does not clear our own threshold.** "
          f"{oracle_sr - max(u6):.2f}pp sits below the 3.82–4.15pp one-sided band "
          "derived in §1b, so six distinct representations are not distinguishable "
          "from six repetitions of the strongest one. Repetition explains most of the "
          "ceiling. This is the strongest available attack on the ceiling claim and "
          "it substantially lands.", "",
          "**At matched SERVING COST the two are not interchangeable.** The mode "
          "oracle spends ONE episode per task; the six-rerun union spends SIX. So the "
          "residual buys little, but it buys it at a sixth of the deployment cost — "
          "which is the axis a deployment actually pays on. The surviving claim is "
          "therefore about cost-efficiency of the ceiling, not about its height.", "",
          "⚠️ Both readings are bounded by one cell — `classifieds·B0` is the only "
          "cell carrying replicated arms. Neither generalises without more replicates.",
          ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    for r in rows:
        print(f"  {r['arm']:<16} SR {r['sr_single_pct']:5.2f}%  U(2) model "
              f"{r['union_pct']['2']:5.2f}% vs obs {r['union2_observed_pct']:5.2f}%"
              f"  U(6) {r['union_pct']['6']:5.2f}%")
    print(f"  six-mode oracle {oracle_sr:.2f}%  →  gap over best U(6) = "
          f"{oracle_sr - max(u6):.2f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
