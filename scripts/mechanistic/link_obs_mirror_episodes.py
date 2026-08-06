#!/usr/bin/env python3
"""把 `_obs_mirror` 的 artifacts 与本地 `phase1/` 的 episodes 接起来。

WHY 这个脚本必须存在
====================
观测数据被 **同步策略切成了两半**，而每个下游 mechanistic 脚本都需要两半同时在场：

  * `results/{visualwebarena,webarena}/phase1/<run>/<cond>/episodes/`
    —— 本地有。`sync_a100_results.sh` 只 exclude `artifacts/`，episodes 照常同步。
  * `results/mechanistic/_obs_mirror/<bm>/<run>/<cond>/artifacts/`
    —— 2026-08-06 从 A100 拉的文本观测。**刻意放在 `phase1/` 之外**：那棵树带
    `--exclude artifacts/ + --delete-excluded`，拉进去会被下一次 cron sync 清掉
    (`results/repro_replicates/README.md` 记录过同一个坑)。

于是 `b0_paired_idperturb_replay.py` 这类脚本会失败得**很安静**：它从
`CURR/episodes/*_summary_v2.json` 取 task 池，池为空就跑 0 个 task、
打印 `=== AGG === {}`、**退出码 0**。实证 2026-08-06：第一次 M1 pilot 就是这样
「成功」的 —— 只有 `tasks=0` 一行透露了真相。

所以这里给 mirror 侧补一个指向本地 episodes 的相对符号链接。相对而非绝对：
`_obs_mirror` 将来若整体搬走或 rsync 到别的机器，`rsync -l` / `tar` 保留相对
目标仍然可解析。

用法
====
    .venv/bin/python3 scripts/mechanistic/link_obs_mirror_episodes.py            # 全部
    .venv/bin/python3 scripts/mechanistic/link_obs_mirror_episodes.py --check    # 只报告不写
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MIRROR = REPO / "results" / "mechanistic" / "_obs_mirror"
BENCH_ROOT = {
    "visualwebarena": REPO / "results" / "visualwebarena" / "phase1",
    "webarena": REPO / "results" / "webarena" / "phase1",
}


def iter_conditions():
    """yield (benchmark, run_id, cond_id, mirror_cond_dir)."""
    if not MIRROR.exists():
        return
    for bm_dir in sorted(MIRROR.iterdir()):
        if not bm_dir.is_dir() or bm_dir.name not in BENCH_ROOT:
            continue
        for run_dir in sorted(bm_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for cond_dir in sorted(run_dir.iterdir()):
                if cond_dir.is_dir() and (cond_dir / "artifacts").is_dir():
                    yield bm_dir.name, run_dir.name, cond_dir.name, cond_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只报告状态，不创建任何链接")
    args = ap.parse_args()

    linked = already = missing = broken = 0
    for bm, run_id, cond_id, cond_dir in iter_conditions():
        local_eps = BENCH_ROOT[bm] / run_id / cond_id / "episodes"
        mirror_eps = cond_dir / "episodes"

        if not local_eps.is_dir():
            # 本地这条 run 没同步下来，或 run_id 在两棵树里不同名。
            # 不猜、不 fuzzy-match：报出来让人看。
            print(f"  ✗ MISSING local episodes  {bm}/{run_id}/{cond_id}")
            missing += 1
            continue

        if mirror_eps.is_symlink():
            if mirror_eps.resolve() == local_eps.resolve():
                already += 1
                continue
            print(f"  ! 已存在但指向别处 {mirror_eps} -> {os.readlink(mirror_eps)}")
            broken += 1
            if args.check:
                continue
            mirror_eps.unlink()
        elif mirror_eps.exists():
            # 真目录：不是我们建的，绝不动它。
            print(f"  ! {mirror_eps} 是真目录而非链接 — 跳过，请人工裁定")
            broken += 1
            continue

        if args.check:
            print(f"  + 待建 {bm}/{run_id}/{cond_id}")
            linked += 1
            continue

        rel = os.path.relpath(local_eps, cond_dir)
        mirror_eps.symlink_to(rel, target_is_directory=True)
        n = len(list(mirror_eps.glob("*_summary_v2.json")))
        if n == 0:
            # 链接建成了但对面是空的 —— 与「没建链接」在下游表现完全一样
            # （task 池为空 → 静默跑 0 个 task）。必须区分开报出来。
            print(f"  ⚠ {bm}/{run_id}/{cond_id}: 链接已建但 episodes 里 0 个 summary")
        linked += 1

    verb = "待建" if args.check else "已建"
    print(f"\n{verb}={linked}  已就绪={already}  本地缺 episodes={missing}  需人工={broken}")
    # 本地缺 episodes 不算失败（那条 run 可能本来就没同步），但要让调用者能判别。
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
