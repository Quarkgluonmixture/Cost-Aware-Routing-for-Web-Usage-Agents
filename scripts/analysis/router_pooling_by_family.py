#!/usr/bin/env python3
"""Does pooling labels ACROSS BACKBONES break identifiability everywhere, or only
across model families?

`router_label_supply_diagnosis.py` reports one pooled conflict rate per site
(classifieds 57.4%, reddit 56.0%) computed over every task shared by 2+ cells.
That statistic mixes two very different pairings:

  same-family, different scale  : B0 (Qwen3-VL-235B-A22B) + B1 (Qwen3-VL-4B)
  cross-family                  : anything involving B2 (Gemma-3-4B)

If the conflict is driven by family rather than by pooling per se, the aggregate
understates how usable a Qwen-only pool would be — and Paper B's claim that the
binding constraint is *label supply* would need the qualifier "across families".

Reported per backbone subset, per site, for both label definitions:
  which-mode   6-way  (the preregistered label)
  cost-tier    binary (image-consuming vs text-only; see _tier)

    router_pooling_by_family.py
    router_pooling_by_family.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import router_label_supply_diagnosis as D  # noqa: E402

FAMILY = {"B0": "qwen", "B1": "qwen", "B2": "gemma"}
MIN_CLASS_N = 10  # same filter router_label_supply_diagnosis uses


def backbone(cell_id: str) -> str:
    return cell_id.split("_", 1)[0]


def analyse(pool: dict, site: str, keep: set[str]) -> dict:
    """Conflict + plug-in ceiling over tasks shared by 2+ of `keep`'s cells."""
    labels, cells, tids = pool["labels"], pool["cell_ids"], pool["task_ids"]
    per_task: dict[int, list[str]] = defaultdict(list)
    for lab, cid, tid in zip(labels, cells, tids):
        if D._site_of(str(cid)) != site or backbone(str(cid)) not in keep:
            continue
        per_task[int(tid)].append(str(lab))

    shared = {t: ls for t, ls in per_task.items() if len(ls) >= 2}
    out = {"n_labels": sum(len(v) for v in per_task.values()),
           "n_tasks_with_label": len(per_task), "n_shared": len(shared)}
    if not shared:
        return out | {"conflict_rate": None, "ceiling_mode": None,
                      "ceiling_tier": None, "conflict_rate_tier": None}

    # which-mode
    conf = {t: ls for t, ls in shared.items() if len(set(ls)) > 1}
    out["n_conflicting"] = len(conf)
    out["conflict_rate"] = len(conf) / len(shared)
    # plug-in ceiling = pick each task's modal label, count how many rows it covers
    hit = sum(Counter(ls).most_common(1)[0][1] for ls in shared.values())
    tot = sum(len(ls) for ls in shared.values())
    out["ceiling_mode"] = hit / tot

    # cost tier
    tier = {t: [D._tier(l) for l in ls] for t, ls in shared.items()}
    conf_t = {t: ls for t, ls in tier.items() if len(set(ls)) > 1}
    out["conflict_rate_tier"] = len(conf_t) / len(tier)
    hit_t = sum(Counter(ls).most_common(1)[0][1] for ls in tier.values())
    out["ceiling_tier"] = hit_t / tot

    # trainability of the pooled set under the same min-class filter
    flat = [l for ls in per_task.values() for l in ls]
    out["classes_surviving"] = sum(1 for _, n in Counter(flat).items()
                                   if n >= MIN_CLASS_N)
    out["classes_present"] = len(set(flat))
    tiers = [D._tier(l) for l in flat]
    out["tier_classes_surviving"] = sum(1 for _, n in Counter(tiers).items()
                                        if n >= MIN_CLASS_N)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    pool = D.load_pool()
    all_bb = sorted({backbone(str(c)) for c in pool["all_cell_ids"]})
    subsets = [frozenset(s) for k in (2, 3)
               for s in combinations(all_bb, k) if k <= len(all_bb)]

    res = {}
    for site in sorted({D._site_of(str(c)) for c in pool["all_cell_ids"]}):
        print(f"\n{'='*96}\n{site}\n{'='*96}")
        print(f"{'subset':16s} {'family':10s} {'shared':>7s} {'conflict':>9s} "
              f"{'ceil(mode)':>11s} {'tier-conf':>10s} {'ceil(tier)':>11s} "
              f"{'cls≥10':>7s} {'tier≥10':>8s}")
        print("-" * 96)
        for ss in subsets:
            fam = "same-family" if len({FAMILY[b] for b in ss}) == 1 else "cross-family"
            r = analyse(pool, site, set(ss))
            res[f"{site}|{'+'.join(sorted(ss))}"] = r | {"family": fam}
            if r["conflict_rate"] is None:
                print(f"{'+'.join(sorted(ss)):16s} {fam:10s}  no shared tasks")
                continue
            print(f"{'+'.join(sorted(ss)):16s} {fam:10s} "
                  f"{r['n_shared']:7d} {r['conflict_rate']*100:8.1f}% "
                  f"{r['ceiling_mode']*100:10.1f}% {r['conflict_rate_tier']*100:9.1f}% "
                  f"{r['ceiling_tier']*100:10.1f}% "
                  f"{r['classes_surviving']:4d}/{r['classes_present']:<2d} "
                  f"{r['tier_classes_surviving']:7d}")

    print("\n判读：same-family 若 conflict 显著低于 cross-family，则 Paper B 的"
          "\n『binding constraint = 标签供给』需加限定『跨族』。"
          "\ncls≥10 = 池化后活过 min-class 过滤的类数（which-mode，需 ≥2 才可训）；"
          "\ntier≥10 同理但用二值 cost tier。")
    if a.json:
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
