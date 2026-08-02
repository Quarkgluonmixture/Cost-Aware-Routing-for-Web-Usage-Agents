#!/usr/bin/env python3
"""Do the features a router would obviously reach for point the way anyone would expect?

Two questions, both about the feature side rather than the label side, and both with answers
that are useful precisely because they are not the expected ones.

1. `has_reference_image` is in the router's binary feature set and is the feature a human would
   route on: the task ships a picture, so send it to a mode that can see pictures. We stratify
   by it and measure whether image-bearing modes actually gain there.

2. `visual_difficulty` is a VWA-native annotation of how much the PAGE has to be read visually.
   `extract_50_features.py` reads it out of the task config and then does not put it in the
   feature table (`feature_names_numeric` carries `reasoning_difficulty` only). We measure what
   it would have been worth.

The mechanism behind (1) is in the runner, not the data: `main.py` passes `reference_images`
into `BackendStepContext` with no per-mode filter, so all six modes receive the task's own
picture. What the image-free modes lack is the PAGE screenshot. A task that ships its own
reference image is therefore exactly the task that does not need one.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

LOG = logging.getLogger("feature_diagnostics")
VWA_CFG = REPO / "external/visualwebarena/config_files/vwa"
SR = REPO / "results/phantom_paper/per_task_sr.csv"
FEAT_JSON = REPO / "results/phantom_paper/l1_router/raw_features_phase1a.json"
OUT_MD = REPO / "docs/analysis/cross_sites/routing_feature_diagnostics.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/routing_feature_diagnostics.json"

MODES = ["dom", "som", "vision", "ptext", "pprompt", "psom"]
IMAGE = {"som", "vision"}
TEXT = {"dom", "ptext", "pprompt", "psom"}
DIFF_ORDER = ["easy", "medium", "hard"]
DIFF_FIX = {"mediun": "medium"}      # three tasks carry this typo in the VWA configs


class MissingInput(RuntimeError):
    """Fail loud rather than stratify over a partial task set."""


def load() -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = {}
    for r in csv.DictReader(SR.open()):
        site = r["site"]
        scored, _ = expected_scored_ids(site)
        tid = int(r["task_id"])
        if tid not in scored:
            continue
        f = VWA_CFG / f"test_{site}" / f"{tid}.json"
        if not f.exists():
            raise MissingInput(f"no task config for {site}/{tid}")
        cfg = json.loads(f.read_text())
        img = cfg.get("image")
        vd = cfg.get("visual_difficulty")
        cells.setdefault(r["cell_id"], []).append({
            "tid": tid,
            "has_img": img not in (None, "None", "", []),
            "vis_diff": DIFF_FIX.get(vd, vd),
            "solve": {m for m in MODES if float(r[f"sr_{m}"]) > 0},
        })
    for cid, rows in cells.items():
        site = "classifieds" if cid.startswith("cls") else "reddit"
        if len(rows) != len(expected_scored_ids(site)[0]):
            raise MissingInput(f"{cid}: {len(rows)} rows against the canonical universe")
    return cells


def marginal(rows: list[dict], cands: set[str], base_mode: str) -> float:
    """Best gain from adding ONE arm out of `cands` on top of `base_mode`. Arm-count matched."""
    n = len(rows)
    base = {r["tid"] for r in rows if base_mode in r["solve"]}
    best = 0.0
    for m in cands:
        if m == base_mode:
            continue
        best = max(best, 100 * len({r["tid"] for r in rows if m in r["solve"]} - base) / n)
    return best


def build() -> dict:
    cells = load()
    feat = json.loads(FEAT_JSON.read_text()) if FEAT_JSON.exists() else {}
    out = {"schema": "2026-08-02-routing-feature-diagnostics-v1", "post_hoc_exploratory": True,
           "visual_difficulty_in_feature_table":
               "visual_difficulty" in (feat.get("feature_names_numeric") or []),
           "numeric_features": feat.get("feature_names_numeric"),
           "has_reference_image": {}, "visual_difficulty": {}, "corpus": {}}

    n_img = n_tot = 0
    for cid, rows in sorted(cells.items()):
        img = [r for r in rows if r["has_img"]]
        txt = [r for r in rows if not r["has_img"]]
        n_img += len(img); n_tot += len(rows)
        entry = {"n_img": len(img), "n_txt": len(txt), "sr": {}}
        for label, grp in (("with_ref_image", img), ("without_ref_image", txt)):
            entry["sr"][label] = {m: 100 * sum(m in r["solve"] for r in grp) / len(grp)
                                  for m in MODES} if grp else {}
            if grp:
                strongest_text = max(TEXT, key=lambda m: sum(m in r["solve"] for r in grp))
                entry.setdefault("marginal", {})[label] = {
                    "base_text_mode": strongest_text,
                    "add_image_arm": marginal(grp, IMAGE, strongest_text),
                    "add_text_arm": marginal(grp, TEXT, strongest_text),
                }
        m = entry["marginal"]
        entry["intuition_holds"] = (
            m["with_ref_image"]["add_image_arm"] - m["with_ref_image"]["add_text_arm"]
            > m["without_ref_image"]["add_image_arm"] - m["without_ref_image"]["add_text_arm"])
        out["has_reference_image"][cid] = entry

        vd: dict[str, dict] = {}
        for lvl in DIFF_ORDER:
            grp = [r for r in rows if r["vis_diff"] == lvl]
            if not grp:
                continue
            sr = {m: 100 * sum(m in r["solve"] for r in grp) / len(grp) for m in MODES}
            vd[lvl] = {"n": len(grp),
                       "gap_best_image_minus_best_text":
                           max(sr[m] for m in IMAGE) - max(sr[m] for m in TEXT)}
        out["visual_difficulty"][cid] = vd
    out["corpus"] = {"n_scored": n_tot, "n_with_reference_image": n_img,
                     "pct_with_reference_image": 100 * n_img / n_tot}
    LOG.info("reference-image share over the scored universe: %.1f%%",
             out["corpus"]["pct_with_reference_image"])
    return out


def render(d: dict) -> str:
    c = d["corpus"]
    L = ["---", "type: analysis", "status: complete", "created: 2026-08-02",
         "purpose: whether the two obvious routing features point the way a practitioner would "
         "assume", "post_hoc_exploratory: true",
         "scope_warning: VWA only; WebArena ships no reference images and has no "
         "visual_difficulty annotation. Stratum sizes are small in the low-success cells.",
         "producer: scripts/analysis/aggregate_routing_feature_diagnostics.py", "---", "",
         "# Two routing features, and what they actually predict", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_routing_feature_diagnostics.py`",
         "",
         f"**{c['n_with_reference_image']} of {c['n_scored']} scored VWA tasks "
         f"({c['pct_with_reference_image']:.1f}%) ship a reference image.**", "",
         "## 1. `has_reference_image`: the sign is backwards", "",
         "The intuition is that a task shipping a picture should be routed to a mode that can see "
         "pictures. Solve rate by mode, split on the feature:", "",
         "| cell | stratum | n | " + " | ".join(MODES) + " |",
         "|---|---|---|" + "---|" * len(MODES)]
    for cid, e in d["has_reference_image"].items():
        for label, nk in (("with_ref_image", "n_img"), ("without_ref_image", "n_txt")):
            if not e["sr"].get(label):
                continue
            L.append(f"| `{cid}` | {label.replace('_', ' ')} | {e[nk]} | "
                     + " | ".join(f"{e['sr'][label][m]:.1f}" for m in MODES) + " |")
    L += ["", "Arm-count matched: on top of that stratum's strongest text mode, the best gain "
          "from adding **one** image-bearing arm against adding **one** other text arm.", "",
          "| cell | with a reference image | without one | intuition holds? |",
          "|---|---|---|---|"]
    holds = 0
    for cid, e in d["has_reference_image"].items():
        a, b = e["marginal"]["with_ref_image"], e["marginal"]["without_ref_image"]
        da = a["add_image_arm"] - a["add_text_arm"]
        db = b["add_image_arm"] - b["add_text_arm"]
        holds += e["intuition_holds"]
        L.append(f"| `{cid}` | {a['add_image_arm']:.2f} vs {a['add_text_arm']:.2f} "
                 f"({da:+.2f}) | {b['add_image_arm']:.2f} vs {b['add_text_arm']:.2f} "
                 f"({db:+.2f}) | {'yes' if e['intuition_holds'] else '**no**'} |")
    n_cells = len(d["has_reference_image"])
    L += ["", f"**The intuition holds in {holds} of {n_cells} cells.** In the two largest it is "
          "reversed: the image-bearing arm is worth more on the tasks that do *not* ship a "
          "reference image.", "",
          "The mechanism is in the harness, not the data. `reference_images` is passed into "
          "`BackendStepContext` with no per-mode filter, so all six modes receive the task's own "
          "picture; what the image-free modes lack is the **page** screenshot. A task that ships "
          "its own reference image is therefore precisely the task that does not need one, and "
          "the feature measures the opposite of what a router needs.", "",
          "## 2. `visual_difficulty`: the right feature, read and then dropped", "",
          f"`extract_50_features.py` reads this VWA-native annotation out of each task config and "
          f"does not place it in the feature table. Present in `feature_names_numeric`: "
          f"**{d['visual_difficulty_in_feature_table']}**. The table carries "
          f"`{d['numeric_features']}`.", "",
          "Best image-bearing mode minus best text-only mode, by annotated visual difficulty:", "",
          "| cell | " + " | ".join(f"{l} (n)" for l in DIFF_ORDER) + " |",
          "|---|" + "---|" * len(DIFF_ORDER)]
    means = {l: [] for l in DIFF_ORDER}
    for cid, vd in d["visual_difficulty"].items():
        cells_ = []
        for l in DIFF_ORDER:
            if l in vd:
                means[l].append(vd[l]["gap_best_image_minus_best_text"])
                cells_.append(f"{vd[l]['gap_best_image_minus_best_text']:+.2f} ({vd[l]['n']})")
            else:
                cells_.append("—")
        L.append(f"| `{cid}` | " + " | ".join(cells_) + " |")
    L.append("| **mean** | " + " | ".join(
        f"**{st.fmean(means[l]):+.2f}**" if means[l] else "—" for l in DIFF_ORDER) + " |")
    L += ["", "The mean gap rises monotonically with annotated visual difficulty, which is the "
          "direction a router would want, **but it is carried by classifieds and reverses on "
          "reddit**. It is a better feature than `has_reference_image` and it is not a rescue: "
          "adding it changes no cell's supply arithmetic, since the constraint there is the "
          "number of labelled rows and not their separability.", "",
          "## 3. What this does and does not show", "",
          "It shows that the prior a practitioner would bring to this problem is wrong-signed, "
          "which matters because few-shot learning leans on priors: a hand-written rule of the "
          "obvious form would actively hurt, and an L1-regularised model fitted on 15 to 97 rows "
          "is unlikely to recover a counterintuitive coefficient. It does **not** show that "
          "better features would fix routing. The supply argument is about row counts and is "
          "untouched by anything on this page."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    d = build()
    OUT_JSON.write_text(json.dumps(d, indent=2))
    OUT_MD.write_text(render(d))
    print(f"✓ {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
