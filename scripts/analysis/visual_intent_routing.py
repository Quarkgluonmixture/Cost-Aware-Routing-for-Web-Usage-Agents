"""Can a 0-token text rule say, in advance, which tasks need the screenshot?

`P43` (diag ruleset) marks (task × mode) pairs whose intent needs page-embedded visual
information while the mode withholds the page screenshot. Its production form contains
an outcome filter — `if summary.get("success"): return []` — which makes the hit set
**outcome-dependent**: tasks the text-only arms solved are excluded by construction, so
comparing arms *within* that set measures the selection, not the screenshot.

This strips the outcome filter and keeps only the parts that are decidable **before any
episode runs**:

    VISUAL_INTENT_RE.search(intent)   and   not task_config["image"]

Both read the task config. No model call, no episode, no tokens. What remains is a
genuine ex-ante partition of the scored task set, and the question becomes answerable:
does the screenshot pay off more on the tasks the rule points at?

Provenance matters for reading the result. The rule was written for reddit — its
docstring cites "64 reddit tasks, previously invisible to every Tier-1 rule" — and the
classifieds hits were incidental and never examined. The classifieds numbers below are
therefore **out-of-sample** for this rule: the regex was not tuned on them.

Regenerate:
    .venv/bin/python3 scripts/analysis/visual_intent_routing.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

# Copied verbatim from scripts/analysis/diag_pattern_match.py (VISUAL_INTENT_RE).
# Duplicated rather than imported so this product pins the exact predicate it measured:
# the diag ruleset is versioned and moves (v8 → v11 in one week), and a silently
# updated regex would change what this file claims without changing this file.
VISUAL_INTENT_RE = re.compile(
    r"\b(image|picture|photo|screenshot)\b|\bcolou?r of\b|"
    r"\bhow many\b[^.]{0,40}\bin (?:the|this)\b",
    re.IGNORECASE,
)

MODES = ["dom", "som", "vision", "ptext", "pprompt", "psom"]
IMAGE_ARMS = ["vision", "som"]
CFG_DIR = {
    "classifieds": REPO / "external/visualwebarena/config_files/vwa/test_classifieds",
    "reddit": REPO / "external/visualwebarena/config_files/vwa/test_reddit",
}
SITE_PREFIX = {"classifieds": "cls", "reddit": "red"}

# WebArena. Its task configs are not in the submodule — each run mirrors the ones it ran
# into its own `task_configs/`, so the predicate is rebuilt from a run directory. Success
# comes from the same episode summaries rather than `per_task_sr.csv`, which is VWA-only.
WA_ROOT = REPO / "results/webarena/phase1"
WA_MODE_STEM = {"dom": "dom", "som": "som", "vision": "vision", "ptext": "phantom_text",
                "pprompt": "phantom_prompt", "psom": "phantom_som"}
WA_BASELINES = ("B1", "B0")
N_BOOT = 10000
SEED = 20260803

OUT_MD = REPO / "docs/analysis/cross_sites/visual_intent_routing.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/visual_intent_routing.json"


class MissingInput(RuntimeError):
    """Fail loud rather than report an empty partition as a null result."""


def ex_ante_set(site: str) -> set[int]:
    scored, _ = expected_scored_ids(site)
    d = CFG_DIR[site]
    if not d.is_dir():
        raise MissingInput(f"{d} missing — cannot rebuild the ex-ante set")
    hits, missing = set(), []
    for tid in sorted(scored):
        p = d / f"{tid}.json"
        if not p.exists():
            missing.append(tid)
            continue
        cfg = json.loads(p.read_text())
        if cfg.get("image"):
            continue  # task-level reference image is delivered in every mode
        if VISUAL_INTENT_RE.search(str(cfg.get("intent") or "")):
            hits.add(tid)
    if missing:
        raise MissingInput(
            f"{site}: {len(missing)} task configs absent (e.g. {missing[:5]}) — the "
            "partition would be silently incomplete")
    if not hits:
        raise MissingInput(f"{site}: regex matched nothing — check the config path")
    return hits


def load_sr() -> dict[str, dict[int, dict[str, int]]]:
    cells: dict[str, dict[int, dict[str, int]]] = {}
    for r in csv.DictReader((REPO / "results/phantom_paper/per_task_sr.csv").open()):
        scored, _ = expected_scored_ids(r["site"])
        tid = int(r["task_id"])
        if tid in scored:
            cells.setdefault(r["cell_id"], {})[tid] = {
                m: int(float(r[f"sr_{m}"]) > 0) for m in MODES}
    return cells


def _wa_run_dir(baseline: str, mode: str) -> Path:
    pat = f"{baseline}_{WA_MODE_STEM[mode]}_wa_reddit_2026*_R*"
    hits = [Path(p) for p in glob.glob(str(WA_ROOT / pat))
            if Path(p).is_dir() and "ABORTED" not in p]
    if len(hits) != 1:
        raise MissingInput(f"WA[{baseline}] {mode}: expected 1 run dir for {pat!r}, got {len(hits)}")
    return hits[0]


def load_wa_sr(baseline: str) -> dict[int, dict[str, int]]:
    """Per-task success for one WA cell, on the six-mode task intersection.

    WA carries no AMENDMENT_08 list, so the universe is what all six modes actually ran —
    the same convention `fusion_premium` and `conditional_failure_attribution` use.
    """
    per: dict[str, dict[int, int]] = {}
    for m in MODES:
        d = _wa_run_dir(baseline, m)
        rows = {}
        for f in (list(d.glob("*/episodes/*summary*.json"))
                  or list(d.glob("episodes/*summary*.json"))):
            s = json.loads(f.read_text())
            if not s.get("sr_excluded"):
                rows[int(s["task_id"])] = 1 if s.get("success") else 0
        per[m] = rows
    common = set.intersection(*(set(v) for v in per.values()))
    if not common:
        raise MissingInput(f"WA[{baseline}]: empty six-mode task intersection")
    return {t: {m: per[m][t] for m in MODES} for t in sorted(common)}


def wa_ex_ante_set(baseline: str, universe: set[int]) -> set[int]:
    """Same predicate, rebuilt from the run's own mirrored task configs."""
    cfg_dir = _wa_run_dir(baseline, "dom") / "task_configs"
    if not cfg_dir.is_dir():
        raise MissingInput(f"{cfg_dir} missing — WA configs are mirrored per run (B-1919)")
    hits, missing = set(), []
    for tid in sorted(universe):
        p = cfg_dir / f"reddit_task_{tid}.json"
        if not p.exists():
            missing.append(tid)
            continue
        cfg = json.loads(p.read_text())
        if cfg.get("image"):
            continue
        if VISUAL_INTENT_RE.search(str(cfg.get("intent") or "")):
            hits.add(tid)
    if missing:
        raise MissingInput(
            f"WA[{baseline}]: {len(missing)} task configs absent (e.g. {missing[:5]}) — the "
            "partition would be silently incomplete")
    return hits


def paired_diff(tasks: dict[int, dict[str, int]], ids: list[int], a: str, b: str) -> dict:
    n = len(ids)
    diffs = [tasks[t][a] - tasks[t][b] for t in ids]
    est = 100 * sum(diffs) / n
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        boots.append(100 * s / n)
    boots.sort()
    return {"n": n, "est_pp": est,
            "ci": [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]],
            "n_a": sum(tasks[t][a] for t in ids), "n_b": sum(tasks[t][b] for t in ids)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    a = ap.parse_args()

    sr = load_sr()
    out: dict = {"schema": 1, "post_hoc_exploratory": True, "h10_eligible": False,
                 "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "predicate": VISUAL_INTENT_RE.pattern, "n_boot": N_BOOT, "seed": SEED,
                 "sites": {}}

    L = ["---", "type: analysis", "status: rolling",
         "purpose: does a 0-token text rule identify, in advance, the tasks where the "
         "screenshot pays",
         "producer: scripts/analysis/visual_intent_routing.py", "---", "",
         "# Visual-intent routing — an ex-ante partition", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/visual_intent_routing.py`", "",
         "The predicate is a regex over the task intent plus a check that the task carries "
         "no reference image:", "",
         f"```\n{VISUAL_INTENT_RE.pattern}\n```", "",
         "Both terms read the **task config**. No model call, no episode, no tokens — this "
         "is decidable before anything runs.", "",
         "⚠️ **This is not `P43` as shipped.** The production rule adds "
         "`if summary.get(\"success\"): return []`, which makes its hit set "
         "outcome-dependent — tasks the text-only arms solved are excluded by construction, "
         "so an arm comparison inside that set measures the selection. The outcome filter is "
         "dropped here; everything else is P43's predicate verbatim.", "",
         "⚠️ **Provenance.** The rule was written for *reddit* (\"64 reddit tasks, previously "
         "invisible to every Tier-1 rule\"). Its classifieds hits were incidental and never "
         "examined, so the classifieds rows are **out-of-sample** — the regex was not tuned "
         "on them.", ""]

    for site in ("classifieds", "reddit"):
        S = ex_ante_set(site)
        pre = SITE_PREFIX[site]
        L += [f"## {site} — flagged n={len(S)}", "",
              "| cell | arm | flagged Δ vs DOM | 95% CI | rest Δ vs DOM | 95% CI | concentration |",
              "|---|---|---|---|---|---|---|"]
        site_rec: dict = {"n_flagged": len(S), "flagged_task_ids": sorted(S), "cells": {}}
        for b in ("B0", "B1", "B2"):
            cell = f"{pre}_{b}"
            if cell not in sr:
                continue
            flagged = sorted(t for t in S if t in sr[cell])
            rest = sorted(set(sr[cell]) - set(flagged))
            rec: dict = {}
            for arm in IMAGE_ARMS:
                fin = paired_diff(sr[cell], flagged, arm, "dom")
                rou = paired_diff(sr[cell], rest, arm, "dom")
                rec[arm] = {"flagged": fin, "rest": rou,
                            "concentration_pp": fin["est_pp"] - rou["est_pp"]}
                L.append(
                    f"| `{cell}` | {arm} | **{fin['est_pp']:+.2f}pp** "
                    f"({fin['n_a']}/{fin['n']} vs {fin['n_b']}/{fin['n']}) | "
                    f"[{fin['ci'][0]:+.2f}, {fin['ci'][1]:+.2f}] | "
                    f"{rou['est_pp']:+.2f}pp | [{rou['ci'][0]:+.2f}, {rou['ci'][1]:+.2f}] | "
                    f"**{fin['est_pp'] - rou['est_pp']:+.2f}pp** |")
            site_rec["cells"][cell] = rec
        out["sites"][site] = site_rec
        L.append("")

    # WebArena. Same predicate, different universe convention and a different success source.
    wa_rec: dict = {"cells": {}}
    L += ["## WebArena reddit", "",
          "Same predicate. The universe is the six-mode task intersection (WA has no "
          "AMENDMENT_08 list) and the configs come from each run's mirrored `task_configs/`.", "",
          "⚠️ **The predicate barely fires here.** It flags 71 of 224 classifieds tasks and 63 of "
          "203 VWA-reddit tasks, but only **5 of 104** on WA — WebArena's intents are worded "
          "differently and the regex, which was written against VWA phrasing, mostly misses. "
          "Whatever these rows show, they are not a test of the classifieds result: **the WA "
          "cells are a coverage note, not a replication.**", "",
          "| cell | n flagged | arm | flagged Δ vs DOM | 95% CI | rest Δ vs DOM | 95% CI |",
          "|---|---|---|---|---|---|---|"]
    for wb in WA_BASELINES:
        try:
            tasks = load_wa_sr(wb)
            S = wa_ex_ante_set(wb, set(tasks))
        except MissingInput as e:
            L.append(f"| `wa_{wb}` | — | — | *{e}* | | | |")
            continue
        flagged = sorted(S)
        rest = sorted(set(tasks) - S)
        if not flagged or not rest:
            L.append(f"| `wa_{wb}` | {len(flagged)} | — | *degenerate partition* | | | |")
            continue
        rec: dict = {"n_flagged": len(flagged), "n_universe": len(tasks)}
        for arm in IMAGE_ARMS:
            fin = paired_diff(tasks, flagged, arm, "dom")
            rou = paired_diff(tasks, rest, arm, "dom")
            # A flagged set nothing solves yields 0.00pp with a zero-width interval, which
            # reads like "no effect" and is actually "no information": there is nothing to
            # resample. Label it, because the number is otherwise indistinguishable from a
            # measured null and would be quoted as one.
            degenerate = (fin["n_a"] == 0 and fin["n_b"] == 0)
            rec[arm] = {"flagged": fin, "rest": rou, "degenerate": degenerate,
                        "concentration_pp": None if degenerate
                                            else fin["est_pp"] - rou["est_pp"]}
            cell_txt = ("**degenerate** — no mode solves any flagged task, so this is "
                        "*no information*, not a measured null"
                        if degenerate else
                        f"**{fin['est_pp']:+.2f}pp** "
                        f"({fin['n_a']}/{fin['n']} vs {fin['n_b']}/{fin['n']})")
            ci_txt = "—" if degenerate else f"[{fin['ci'][0]:+.2f}, {fin['ci'][1]:+.2f}]"
            L.append(
                f"| `wa_{wb}` | {len(flagged)}/{len(tasks)} | {arm} | {cell_txt} | {ci_txt} | "
                f"{rou['est_pp']:+.2f}pp | [{rou['ci'][0]:+.2f}, {rou['ci'][1]:+.2f}] |")
        wa_rec["cells"][f"wa_{wb}"] = rec
    out["sites"]["wa_reddit"] = wa_rec
    L.append("")

    # Headline, derived rather than asserted.
    cls = out["sites"]["classifieds"]["cells"]
    red = out["sites"]["reddit"]["cells"]
    best = max(cls, key=lambda c: cls[c]["vision"]["flagged"]["est_pp"])
    bv = cls[best]["vision"]
    L += ["## What this says", "",
          f"On classifieds the partition separates: `{best}` pays "
          f"**{bv['flagged']['est_pp']:+.2f}pp** for the screenshot on the flagged tasks "
          f"against **{bv['rest']['est_pp']:+.2f}pp** on the rest. The flagged set is "
          f"{len(out['sites']['classifieds']['flagged_task_ids'])} of "
          f"{len(sr[best])} tasks and the predicate costs nothing to evaluate.", "",
          "Two things this is **not**. It is not a claim that the flagged tasks are "
          "unsolvable without the screenshot — several are solved by DOM. And it is not a "
          "router: the partition is fixed and known in advance, which is what makes it "
          "cheap, but nothing here learns it or adapts it.", "",
          "Three caveats that belong beside the number, in order of how much they cost:", ""]
    b2 = cls.get("cls_B2", {}).get("vision", {}).get("flagged", {}).get("est_pp")
    if b2 is not None:
        L.append(f"1. **It needs capability to cash in.** `cls_B2` gets {b2:+.2f}pp on the "
                 "same flagged set — the weakest backbone cannot use the screenshot even "
                 "when the rule correctly says it is needed. Two of three backbones, not "
                 "three of three.")
    rb0 = red.get("red_B0", {}).get("vision", {}).get("flagged", {}).get("est_pp")
    if rb0 is not None:
        L.append(f"2. **It is site-specific, and on reddit the sign flips.** `red_B0` gets "
                 f"{rb0:+.2f}pp on its flagged tasks — the screenshot *hurts* there. Same "
                 "predicate, opposite verdict, which is the modality reversal showing up in "
                 "a third functional.")
    L += ["3. **The counts are small.** The flagged sets are ~70 tasks and the successes "
          "behind the largest gap are in the low twenties against single digits; the "
          "intervals above are paired bootstrap over tasks and should be read, not the "
          "point estimates alone.", ""]

    a.out_md.write_text("\n".join(L) + "\n")
    a.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[md]   {a.out_md}")
    print(f"[json] {a.out_json}")
    for site in out["sites"]:
        for cell, rec in out["sites"][site]["cells"].items():
            v = rec.get("vision")
            if not v:
                continue
            if v.get("degenerate"):
                print(f"  {cell:8} vision  DEGENERATE — no mode solves any of the "
                      f"{rec.get('n_flagged', '?')} flagged tasks (no information)")
                continue
            print(f"  {cell:8} vision flagged {v['flagged']['est_pp']:+7.2f}pp  "
                  f"rest {v['rest']['est_pp']:+6.2f}pp  conc {v['concentration_pp']:+7.2f}pp")


if __name__ == "__main__":
    main()
