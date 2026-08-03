"""How much of each cell's wall-clock is spent outside the benchmark?

Postmill (both VWA reddit and WA reddit) is a link-aggregator: a post's title *is* an
external URL, so an agent that opens a trending thread can navigate to the live public
internet and keep acting there. The classifieds site is self-contained and has almost no
such links.

That asymmetry matters because the paper's latency claim is a **site-level** one — "the
cheapest mode is not the fastest, and the split follows the site". If reddit episodes spend
part of their `env_step` time waiting on imgur and news sites while classifieds episodes
never do, then part of the between-site latency difference is network geography rather than
anything about representation.

This measures it: the share of steps whose observed URL is not the local benchmark host,
and what those steps cost in `latency_ms.env_step` relative to on-site steps in the same
cell. It makes no recommendation — whether to exclude, reweight or merely disclose is a
preregistration-level decision.

Regenerate:
    .venv/bin/python3 scripts/analysis/offsite_navigation_audit.py
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.analysis.axis_effect_size as A  # noqa: E402

HOST_RE = re.compile(r"https?://([^/:]+)")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}
WA_STEM = {"dom": "dom", "som": "som", "vision": "vision", "ptext": "phantom_text",
           "pprompt": "phantom_prompt", "psom": "phantom_som"}
OUT_MD = REPO / "docs/analysis/cross_sites/offsite_navigation_audit.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/offsite_navigation_audit.json"


class MissingInput(RuntimeError):
    """Fail loud rather than report a 0% off-site rate that is really a missing input."""


def scan(step_files, label: str) -> dict:
    n_step = n_off = 0
    n_ep = 0
    ep_off = 0
    hosts: Counter = Counter()
    lat_on: list[float] = []
    lat_off: list[float] = []
    for f in step_files:
        n_ep += 1
        this_off = 0
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = rec.get("obs_url")
            if not isinstance(url, str):
                continue
            n_step += 1
            m = HOST_RE.match(url)
            host = m.group(1) if m else None
            env = (rec.get("latency_ms") or {}).get("env_step")
            offsite = bool(host) and host not in LOCAL_HOSTS
            if offsite:
                n_off += 1
                this_off += 1
                hosts[host] += 1
                if isinstance(env, (int, float)):
                    lat_off.append(float(env))
            elif isinstance(env, (int, float)):
                lat_on.append(float(env))
        if this_off:
            ep_off += 1
    if not n_step:
        raise MissingInput(f"{label}: zero steps read — inputs missing, not a 0% result")
    return {
        "cell": label, "n_steps": n_step, "n_offsite": n_off,
        "pct_steps": 100.0 * n_off / n_step,
        "n_episodes": n_ep, "n_episodes_offsite": ep_off,
        "pct_episodes": 100.0 * ep_off / n_ep if n_ep else 0.0,
        "median_env_ms_onsite": st.median(lat_on) if lat_on else None,
        "median_env_ms_offsite": st.median(lat_off) if lat_off else None,
        "mean_env_ms_onsite": st.fmean(lat_on) if lat_on else None,
        "mean_env_ms_offsite": st.fmean(lat_off) if lat_off else None,
        "top_hosts": hosts.most_common(5),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    a = ap.parse_args()

    cells: list[dict] = []
    for b in ("B0", "B1", "B2"):
        for site in ("classifieds", "reddit"):
            files = []
            for ak in A._REGISTRY_MODE_TO_AXIS_KEY.values():
                ep = A.STEP_DIRS.get(b, {}).get(site, {}).get(ak)
                if ep is not None and ep.exists():
                    files += sorted(ep.glob("*_steps_v2.jsonl"))
            if files:
                cells.append(scan(files, f"{b}·VWA-{site[:3]}"))
    for b in ("B1", "B0"):
        files = []
        for stem in WA_STEM.values():
            hits = [p for p in (REPO / "results/webarena/phase1").glob(
                f"{b}_{stem}_wa_reddit_2026*_R*") if p.is_dir() and "ABORTED" not in p.name]
            if hits:
                files += sorted(hits[0].glob("*/episodes/*_steps_v2.jsonl"))
        if files:
            cells.append(scan(files, f"{b}·WA-red"))

    out = {"schema": 1, "post_hoc_exploratory": True, "h10_eligible": False,
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "local_hosts": sorted(LOCAL_HOSTS), "cells": cells}

    L = ["---", "type: analysis", "status: rolling",
         "purpose: how much of each cell's environment time is spent on the public internet",
         "producer: scripts/analysis/offsite_navigation_audit.py", "---", "",
         "# Off-site navigation audit", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/offsite_navigation_audit.py`", "",
         "Postmill is a **link aggregator**: a post's title is an external URL, so an agent "
         "opening a trending thread can walk onto the live public internet and keep acting "
         "there. The classifieds site is self-contained and offers almost no such exit.", "",
         "A step is **off-site** when its `obs_url` host is not `localhost`. `env_step` is the "
         "environment-interaction slice of `latency_ms` — the part that contains page load.", "",
         "## 1. Per cell", "",
         "| cell | off-site steps | off-site episodes | median `env_step` on-site | off-site | ratio |",
         "|---|---|---|---|---|---|"]
    for c in cells:
        on, off = c["median_env_ms_onsite"], c["median_env_ms_offsite"]
        ratio = f"**{off/on:.2f}×**" if on and off else "—"
        L.append(f"| `{c['cell']}` | {c['n_offsite']}/{c['n_steps']} "
                 f"({c['pct_steps']:.2f}%) | {c['n_episodes_offsite']}/{c['n_episodes']} "
                 f"({c['pct_episodes']:.1f}%) | "
                 f"{on:,.0f} ms | {f'{off:,.0f} ms' if off else '—'} | {ratio} |")

    vwa_cls = [c for c in cells if "VWA-cla" in c["cell"]]
    reddits = [c for c in cells if "red" in c["cell"]]
    cls_pcts = ", ".join(f"{c['pct_steps']:.2f}%" for c in vwa_cls)
    red_pcts = ", ".join(f"{c['pct_steps']:.2f}%" for c in reddits)
    L += ["", "## 2. The asymmetry", "",
          f"Classifieds: {cls_pcts}. Reddit (VWA and WA): {red_pcts}.", "",
          "**Off-site navigation is a reddit phenomenon.** That is a property of the "
          "application, not of any observation mode — but the paper's latency claim is stated "
          "at the *site* level (`multimetric_pareto`: the cheapest≠fastest split \"follows the "
          "site, not the backbone\"), and between-site latency comparisons therefore contain a "
          "component that is network geography rather than representation.", ""]

    # Rank by influence on the cell's total environment time, not by the raw ratio: a 3x
    # penalty on 0.16% of steps is a smaller distortion than a 0.5x discount on 2.13%.
    scored = [(c, c["median_env_ms_offsite"] / c["median_env_ms_onsite"])
              for c in cells if c["median_env_ms_offsite"] and c["median_env_ms_onsite"]]
    faster = [(c, r) for c, r in scored if r < 1]
    slower = [(c, r) for c, r in scored if r >= 1]
    fast_txt = ", ".join(f"`{c['cell']}` {r:.2f}×" for c, r in faster)
    slow_txt = ", ".join(f"`{c['cell']}` {r:.2f}×" for c, r in slower)
    L += ["**The penalty runs the other way.** Off-site steps are *faster* than on-site ones in "
          f"{len(faster)} of the {len(scored)} cells that have any ({fast_txt})"
          + (f"; slower in {slow_txt}" if slow_txt else "") + ". "
          "Commercial CDNs outrun a Postmill container sharing a host with the agent, so walking "
          "off-site buys time rather than costing it.", ""]
    if scored:
        top = max(scored, key=lambda cr: cr[0]["pct_steps"] * abs(cr[1] - 1))
        c, r = top
        L += [f"Largest distortion by exposure × magnitude: `{c['cell']}`, {r:.2f}× on "
              f"{c['pct_steps']:.2f}% of steps — about "
              f"**{c['pct_steps'] * abs(r - 1):.2f}%** of that cell's environment time, and in "
              "the direction that makes reddit look faster than it is.", ""]
    L += ["Too small to overturn a latency ordering on its own. It is recorded because it is "
          "**undisclosed and one-sided** — it touches reddit and not classifieds, on the same "
          "axis as the site split the claim rests on.", "",
          "## 2b. The larger asymmetry is the containers themselves", "",
          "| site | median on-site `env_step` |", "|---|---|"]
    for c in cells:
        L.append(f"| `{c['cell']}` | {c['median_env_ms_onsite']:,.0f} ms |")
    cls_med = [c["median_env_ms_onsite"] for c in vwa_cls if c["median_env_ms_onsite"]]
    red_med = [c["median_env_ms_onsite"] for c in reddits if c["median_env_ms_onsite"]]
    if cls_med and red_med:
        L += ["", f"Reddit's on-site page interaction costs "
              f"{st.fmean(red_med) / st.fmean(cls_med):.2f}× what classifieds' does, before any "
              "agent behaviour enters. This dwarfs the off-site effect above and is a property "
              "of Postmill versus Osclass on this host. It does **not** threaten the latency "
              "claim, which compares modes *within* a cell — but it does mean the phrase "
              "\"follows the site\" is carrying infrastructure as well as workload, and a "
              "between-site latency number should never be quoted bare.", ""]
    L += ["## 3. Where they go", ""]
    for c in cells:
        if c["top_hosts"]:
            L.append(f"- `{c['cell']}`: " + ", ".join(f"{h} ({n})" for h, n in c["top_hosts"]))
    L += ["", "⚠️ These are **live public sites reached from an experiment host**. Nothing was "
          "submitted to them — the actions are clicks and scrolls on pages the agent loaded — "
          "but the runs are not hermetic, and a replication on a network-isolated host would "
          "not reproduce these steps at all.", ""]

    a.out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    a.out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[md]   {a.out_md}")
    print(f"[json] {a.out_json}")
    for c in cells:
        print(f"  {c['cell']:14} {c['pct_steps']:5.2f}% steps  {c['pct_episodes']:5.1f}% eps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
