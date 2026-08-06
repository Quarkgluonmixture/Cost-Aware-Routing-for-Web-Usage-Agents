#!/usr/bin/env python3
"""B-1966 重跑验收 — 双向断言：该变的变了，不该变的没变。

只检查「修复后与修复前不同」是不够的：那只能证明代码路径被碰过，不能排除
修复顺手改坏了别的东西。B-1966 的形状恰好允许一个更强的断言 ——

  修复前 source 侧**无条件**带图、target 侧**无条件**不带图。于是：
    * source_mode ∈ {som, vision} 的 cell     → 修复前就是对的 → 新旧必须**逐位相同**
    * source_mode ∉ {som, vision} 的 cell     → 修复前多喂了图 → 新旧必须**不同**
  target 侧同理（target_mode ∈ {som, vision} 才需要图，本批 24 cell 无此情形）。

实测 8/24 受污染（全是 source=phantom_som），16/24 未受污染。
16 个「必须相同」是这次验收真正的承重部分：它证明修复没有副作用。

用法:  .venv/bin/python3 scripts/mechanistic/verify_b1966_rerun.py
       (在 Sparks 上跑, 或把 canonical_b1966fix/ 同步回 DGX 后跑)
"""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OLD = REPO / "results/mechanistic/canonical"
NEW = REPO / "results/mechanistic/canonical_b1966fix"
MODES_WITH_IMAGE = {"som", "vision"}


def cells() -> list[tuple[str, str, str]]:
    """(name, source_mode, target_mode) — 从单一真相源读，不复制清单。"""
    out = subprocess.run(
        ["bash", "-c",
         f'source "{REPO}/scripts/queues/_mechanistic_cells.sh"; printf "%s\\n" "${{CELLS[@]}}"'],
        capture_output=True, text=True, check=True).stdout.strip().split("\n")
    res = []
    for line in out:
        name, _site, _key, extra = line.split("|", 3)
        sm = re.search(r"--source-mode (\S+)", extra)
        tm = re.search(r"--target-mode (\S+)", extra)
        # argparse 默认值，与 run_stage2b_continuation_pilot.py:170-171 对齐
        res.append((name, sm.group(1) if sm else "som", tm.group(1) if tm else "phantom_som"))
    return res


def payload_hash(root: Path, name: str) -> str | None:
    f = root / name / "patching_continuation_results.json"
    if not f.exists():
        return None
    return hashlib.md5(
        json.dumps(json.load(open(f))["per_task"], sort_keys=True).encode()).hexdigest()


def main() -> int:
    rows, bad = [], 0
    for name, sm, tm in cells():
        # 修复前 source 一律带图 / target 一律不带图
        was_correct = (sm in MODES_WITH_IMAGE) and (tm not in MODES_WITH_IMAGE)
        expect = "same" if was_correct else "diff"
        o, n = payload_hash(OLD, name), payload_hash(NEW, name)
        if o is None or n is None:
            verdict, ok = f"缺产物(old={'有' if o else '无'} new={'有' if n else '无'})", None
        else:
            actual = "same" if o == n else "diff"
            ok = (actual == expect)
            verdict = ("✓" if ok else "❌") + f" 预期{expect} 实际{actual}"
        if ok is False:
            bad += 1
        rows.append((name, sm, expect, verdict))

    w = max(len(r[0]) for r in rows) + 2
    print(f"{'cell':<{w}}{'source':<14}{'预期':<6}{'结果'}")
    for name, sm, expect, verdict in rows:
        print(f"{name:<{w}}{sm:<14}{expect:<6}{verdict}")

    n_diff = sum(1 for _, sm, e, _ in rows if e == "diff")
    print(f"\n受污染(预期变) {n_diff}/{len(rows)}   未受污染(预期不变) {len(rows)-n_diff}/{len(rows)}")
    if bad:
        print(f"❌ {bad} 个 cell 与预期不符 —— 修复要么没生效, 要么有副作用")
        return 1
    missing = sum(1 for r in rows if "缺产物" in r[3])
    if missing:
        print(f"⏳ {missing} 个 cell 尚无产物 (重跑未完成)")
        return 2
    print("✅ 全部符合预期: 该变的变了, 不该变的一位没动")
    return 0


if __name__ == "__main__":
    sys.exit(main())
