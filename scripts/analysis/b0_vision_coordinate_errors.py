#!/usr/bin/env python3
"""B0 Vision 坐标错误定量统计 — 含 DOM/SoM 对比."""

import sys
import os
from pathlib import Path
from collections import Counter, defaultdict
import statistics

# 确保项目可导入
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from p79.experiment.io_utils import read_jsonl_dedup

BASE = Path("results/visualwebarena/phase1/B0_3mode_classifieds_20260413")
CONDITIONS = {
    "vision": "phase1_vision_router_0",
    "dom":    "phase1_dom_router_0",
    "som":    "phase1_som_router_0",
}


def load_episodes(condition_dir):
    eps_dir = condition_dir / "episodes"
    episodes = {}
    for f in sorted(eps_dir.glob("*_steps_v2.jsonl")):
        parts = f.stem.replace("_steps_v2", "").split("_task_")
        task_id = int(parts[1])
        records = read_jsonl_dedup(f)
        episodes[task_id] = records
    return episodes


def flatten(episodes):
    return [r for recs in episodes.values() for r in recs]


def get_action_type(r):
    return r.get("action_type") or r.get("action", {}).get("action_type", "unknown")


# 1
def report_action_distribution(steps, label=""):
    print(f"\n{'='*70}")
    print(f"  {label} — 总步数和 action_type 分布")
    print(f"{'='*70}")
    total = len(steps)
    type_counts = Counter(get_action_type(r) for r in steps)
    print(f"总步数: {total}")
    for at, cnt in type_counts.most_common():
        print(f"  {at:20s}: {cnt:5d}  ({100*cnt/total:5.1f}%)")
    return type_counts


# 2
def report_coordinate_failure_rate(steps, label=""):
    print(f"\n{'='*70}")
    print(f"  {label} — click/type 的 action_success=false 率")
    print(f"{'='*70}")
    results = {}
    for at in ["click", "type"]:
        subset = [r for r in steps if get_action_type(r) == at]
        if not subset:
            print(f"  {at}: 无此类步骤")
            continue
        fail = [r for r in subset if r.get("action_success") is False]
        rate = len(fail) / len(subset)
        print(f"  {at}: {len(fail)}/{len(subset)} = {100*rate:.1f}%")
        results[at] = {"total": len(subset), "fail": len(fail), "rate": rate}
    coord_steps = [r for r in steps if get_action_type(r) in ("click", "type")]
    coord_fail = [r for r in coord_steps if r.get("action_success") is False]
    if coord_steps:
        rate = len(coord_fail) / len(coord_steps)
        print(f"  合计(click+type): {len(coord_fail)}/{len(coord_steps)} = {100*rate:.1f}%")
        results["combined"] = {"total": len(coord_steps), "fail": len(coord_fail), "rate": rate}
    return results


# 3
def report_false_but_page_changed(steps, label=""):
    print(f"\n{'='*70}")
    print(f"  {label} — action_success=false 中 page_changed 的比例")
    print(f"{'='*70}")
    fail_steps = [r for r in steps if r.get("action_success") is False]
    if not fail_steps:
        print("  无 action_success=false 的步骤")
        return
    page_changed = [r for r in fail_steps if r.get("page_changed") is True]
    rate = len(page_changed) / len(fail_steps)
    print(f"  action_success=false 且 page_changed=true: {len(page_changed)}/{len(fail_steps)} = {100*rate:.1f}%")
    for at in ["click", "type", "scroll"]:
        sub = [r for r in fail_steps if get_action_type(r) == at]
        if not sub:
            continue
        pc = [r for r in sub if r.get("page_changed") is True]
        print(f"    {at}: page_changed {len(pc)}/{len(sub)} = {100*len(pc)/len(sub):.1f}%")


# 4
def report_failure_streaks(episodes, label=""):
    print(f"\n{'='*70}")
    print(f"  {label} — 连续 action_success=false streak 分布")
    print(f"{'='*70}")
    max_streaks = []
    all_streaks = []
    for task_id, steps in sorted(episodes.items()):
        current_streak = 0
        ep_max = 0
        for r in steps:
            if r.get("action_success") is False:
                current_streak += 1
                ep_max = max(ep_max, current_streak)
            else:
                if current_streak > 0:
                    all_streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            all_streaks.append(current_streak)
        max_streaks.append(ep_max)

    if not max_streaks:
        print("  无数据")
        return

    streak_dist = Counter(max_streaks)
    print(f"  每 episode 最大连续失败 streak 分布 (共 {len(max_streaks)} episodes):")
    for s in sorted(streak_dist.keys()):
        print(f"    max_streak={s:2d}: {streak_dist[s]:3d} episodes ({100*streak_dist[s]/len(max_streaks):.1f}%)")

    if all_streaks:
        all_dist = Counter(all_streaks)
        print(f"\n  所有 streak 长度分布 (共 {len(all_streaks)} streaks):")
        for s in sorted(all_dist.keys()):
            print(f"    streak={s:2d}: {all_dist[s]:3d} ({100*all_dist[s]/len(all_streaks):.1f}%)")

        print(f"\n  streak 统计: mean={statistics.mean(all_streaks):.2f}, "
              f"median={statistics.median(all_streaks):.1f}, "
              f"max={max(all_streaks)}, "
              f"total_streaks={len(all_streaks)}")


# 5
def report_failed_coordinates(steps, label=""):
    print(f"\n{'='*70}")
    print(f"  {label} — action_success=false 的坐标分布 (type + click)")
    print(f"{'='*70}")
    x_labels = ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"]

    for at in ["type", "click"]:
        fail_sub = [
            r for r in steps
            if get_action_type(r) == at and r.get("action_success") is False
        ]
        if not fail_sub:
            print(f"\n  {at}: 无 action_success=false 的操作")
            continue

        xs, ys = [], []
        for r in fail_sub:
            coord = r.get("action", {}).get("coordinate")
            if coord and len(coord) == 2:
                xs.append(coord[0])
                ys.append(coord[1])

        if not xs:
            print(f"\n  {at}: 无有效坐标数据")
            continue

        print(f"\n  --- {at} (action_success=false) ---")
        print(f"  样本数: {len(xs)}")
        if len(xs) > 1:
            print(f"  X: mean={statistics.mean(xs):.3f}, median={statistics.median(xs):.3f}, "
                  f"min={min(xs):.3f}, max={max(xs):.3f}, stdev={statistics.stdev(xs):.3f}")
            print(f"  Y: mean={statistics.mean(ys):.3f}, median={statistics.median(ys):.3f}, "
                  f"min={min(ys):.3f}, max={max(ys):.3f}, stdev={statistics.stdev(ys):.3f}")
        else:
            print(f"  X: {xs[0]:.3f},  Y: {ys[0]:.3f}")

        grid = Counter()
        for x, y in zip(xs, ys):
            bx = min(int(x * 5), 4)
            by = min(int(y * 5), 4)
            grid[(bx, by)] += 1

        print(f"\n  {at} 坐标 5x5 热力分布 (行=Y区间, 列=X区间):")
        print(f"  {'Y\\X':>8s}  " + "  ".join(f"{l:>6s}" for l in x_labels))
        for by in range(5):
            y_label = x_labels[by]
            row_cells = []
            for bx in range(5):
                cnt = grid.get((bx, by), 0)
                row_cells.append(f"{cnt:6d}")
            print(f"  {y_label:>8s}  " + "  ".join(row_cells))


# 6
def report_cross_mode_comparison():
    print(f"\n{'='*70}")
    print(f"  B0 三模式 action_success=false 率对比")
    print(f"{'='*70}")

    rows = []
    for mode, cond_id in CONDITIONS.items():
        cond_dir = BASE / cond_id
        if not cond_dir.exists():
            print(f"  {mode}: 目录不存在 ({cond_dir})")
            continue
        episodes = load_episodes(cond_dir)
        steps = flatten(episodes)
        total = len(steps)
        n_eps = len(episodes)

        all_fail = sum(1 for r in steps if r.get("action_success") is False)
        coord_steps = [r for r in steps if get_action_type(r) in ("click", "type")]
        coord_fail = sum(1 for r in coord_steps if r.get("action_success") is False)

        for at in ["click", "type"]:
            sub = [r for r in steps if get_action_type(r) == at]
            fail = sum(1 for r in sub if r.get("action_success") is False)
            rows.append({"mode": mode, "action_type": at,
                         "total": len(sub), "fail": fail,
                         "rate": fail / len(sub) if sub else 0,
                         "n_eps": n_eps})

        rows.append({"mode": mode, "action_type": "click+type",
                      "total": len(coord_steps), "fail": coord_fail,
                      "rate": coord_fail / len(coord_steps) if coord_steps else 0,
                      "n_eps": n_eps})
        rows.append({"mode": mode, "action_type": "ALL",
                      "total": total, "fail": all_fail,
                      "rate": all_fail / total if total else 0,
                      "n_eps": n_eps})

    print(f"\n  {'Mode':>8s}  {'Episodes':>8s}  {'ActionType':>12s}  {'Total':>6s}  {'Fail':>6s}  {'Rate':>7s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*7}")
    for r in rows:
        print(f"  {r['mode']:>8s}  {r['n_eps']:>8d}  {r['action_type']:>12s}  {r['total']:6d}  {r['fail']:6d}  {100*r['rate']:6.1f}%")

    # error_category 对比
    print(f"\n  error_category 分布对比:")
    for mode, cond_id in CONDITIONS.items():
        cond_dir = BASE / cond_id
        if not cond_dir.exists():
            continue
        episodes = load_episodes(cond_dir)
        steps = flatten(episodes)
        error_cats = Counter(r.get("error_category") for r in steps if r.get("error_category"))
        total_errors = sum(error_cats.values())
        print(f"\n  {mode} (共 {total_errors} 个有 error_category 的步骤):")
        for cat, cnt in error_cats.most_common():
            print(f"    {cat:30s}: {cnt:4d} ({100*cnt/total_errors:.1f}%)")


def main():
    os.chdir(Path(__file__).resolve().parents[2])

    vision_dir = BASE / CONDITIONS["vision"]
    if not vision_dir.exists():
        print(f"Vision 目录不存在: {vision_dir}")
        return

    episodes = load_episodes(vision_dir)
    steps = flatten(episodes)
    print(f"B0 Vision: 加载 {len(episodes)} episodes, {len(steps)} steps")

    report_action_distribution(steps, "B0 Vision")
    report_coordinate_failure_rate(steps, "B0 Vision")
    report_false_but_page_changed(steps, "B0 Vision")
    report_failure_streaks(episodes, "B0 Vision")
    report_failed_coordinates(steps, "B0 Vision")
    report_cross_mode_comparison()


if __name__ == "__main__":
    main()
