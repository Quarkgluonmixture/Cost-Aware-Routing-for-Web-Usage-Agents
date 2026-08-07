#!/usr/bin/env python3
"""把已生成的 `pilot_summary.md` 里那行说反话的 Setup 描述改对（不重算任何数字）。

背景 — B-1966 后续，2026-08-07 /stress P0-1
========================================
`run_stage2b_continuation_pilot.py` 的 summary 模板曾把「有没有喂页面截图」写死：

    - Source: `{mode}` (with image — clean) / Target: `{mode}` (no image — mirage)

修复前 source 侧无条件带图，这行**碰巧**是对的。B-1966 修好之后 source 按 mode 决定，
这行就开始说反话 —— 实测 `p2_psom_ptext_cls/pilot_summary.md` 写着
`Source: phantom_som (with image — clean)`，而 `phantom_som` 的定义就是**不带图**。

数据没错，**描述错了**。而 `pilot_summary.md` 同时是完成标记和人读入口，
留着它等于给未来任何读这批数据的人（包括自己）埋一个与实际相反的结论。

为什么只改一行、不重新生成整份
==============================
summary 里错的**只有这一行**。Result / block-resolved curves / qualitative 三段都是从
真实 per_task 算出来的，正确。重新生成整份需要从 `per_task` 重算 agg —— 那等于把聚合
逻辑再抄一遍，就是这个 bug 的病根（同一契约写在多处）的第四次复发。改一行不碰其余，
既不重算也不复制。

幂等 + 保守
==========
* 已经是新格式 → 跳过（可反复跑）
* 找不到 Setup 行、或找不到同目录的 results json → 跳过并报出，不猜
* 只认那一行的正则；文件其余字节不动

用法:
    .venv/bin/python3 scripts/mechanistic/rerender_pilot_summary_setup_line.py --check
    .venv/bin/python3 scripts/mechanistic/rerender_pilot_summary_setup_line.py
    .venv/bin/python3 scripts/mechanistic/rerender_pilot_summary_setup_line.py --root <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from p79.experiment.som import mode_receives_page_image  # noqa: E402

# ⚠️ 默认**只**扫修复后的目录。
#
# `results/mechanistic/canonical/`（修复前）里那行 `(with image — clean)` 对当时的行为
# 是**准确的** —— 修复前 source 侧无条件带图，所以每个 cell 的 source 都真的有图。
# 那些文件是 B-1966 的**证据**：它们记录了「当时确实喂了图」。用现在的对错去改过去的
# 记录 = 销毁证据。同 §440「旧目录是 B-1966 的证据，不删」。
#
# 想改旧目录必须显式 `--root results/mechanistic/canonical`，而且你大概率不该这么做。
DEFAULT_ROOTS = [
    REPO / "results/mechanistic/canonical_b1966fix",
]

# 只匹配 Setup 段那一行。两种历史写法都认：
#   - Source: `X` (with image — clean) / Target: `Y` (no image — mirage)      ← 旧硬编码
#   - Source: `X` (with page image) / Target: `Y` (no page image)             ← 新派生
SETUP_RE = re.compile(
    r"^- Source: `(?P<src>[^`]+)` \([^)]*\) / Target: `(?P<tgt>[^`]+)` \([^)]*\)$",
    re.MULTILINE,
)


def tag(mode: str) -> str:
    return "with page image" if mode_receives_page_image(mode) else "no page image"


def rerender_one(summary: Path, check: bool) -> str:
    """returns: 'fixed' | 'already' | 'skip:<reason>'"""
    results = summary.parent / "patching_continuation_results.json"
    if not results.exists():
        return "skip:无 results json（无法确认真实 mode）"

    cfg = json.load(open(results)).get("config", {})
    # `source_mode_raw` / `target_mode_raw` 是 --reverse 前的原始参数，正是 input 构造
    # 用的那两个（reverse 只交换 patch 方向，不改构造）。summary 的 Setup 段描述的
    # 就是构造，所以这里必须用 _raw，不能用被 reverse 交换过的 logged 版本。
    src = cfg.get("source_mode_raw") or cfg.get("source_mode")
    tgt = cfg.get("target_mode_raw") or cfg.get("target_mode")
    if not src or not tgt:
        return "skip:config 里没有 source/target mode"

    text = summary.read_text(encoding="utf-8")
    m = SETUP_RE.search(text)
    if not m:
        return "skip:找不到 Setup 的 Source/Target 行"
    if m.group("src") != src or m.group("tgt") != tgt:
        return f"skip:summary 写 {m.group('src')}→{m.group('tgt')} 与 config {src}→{tgt} 不符"

    correct = f"- Source: `{src}` ({tag(src)}) / Target: `{tgt}` ({tag(tgt)})"
    if m.group(0) == correct:
        return "already"
    if check:
        return f"fixed(dry):  {m.group(0).strip()}  →  {correct.strip()}"
    summary.write_text(text[:m.start()] + correct + text[m.end():], encoding="utf-8")
    return "fixed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只报告不写")
    ap.add_argument("--root", action="append", default=None,
                    help="指定扫描根目录（可重复）；默认扫 canonical_b1966fix + canonical")
    args = ap.parse_args()

    roots = [Path(r) for r in args.root] if args.root else DEFAULT_ROOTS
    n_fixed = n_already = n_skip = 0
    for root in roots:
        if not root.exists():
            print(f"  (跳过不存在的根目录 {root})")
            continue
        summaries = sorted(root.glob("*/pilot_summary.md"))
        print(f"\n=== {root.relative_to(REPO) if root.is_relative_to(REPO) else root} — {len(summaries)} 份")
        for s in summaries:
            r = rerender_one(s, args.check)
            cell = s.parent.name
            if r == "already":
                n_already += 1
            elif r.startswith("fixed"):
                n_fixed += 1
                print(f"  ✏️  {cell:<26} {r if r != 'fixed' else '已改'}")
            else:
                n_skip += 1
                print(f"  ⚠️  {cell:<26} {r}")

    verb = "待改" if args.check else "已改"
    print(f"\n{verb}={n_fixed}  本就正确={n_already}  跳过={n_skip}")
    return 1 if n_skip else 0


if __name__ == "__main__":
    sys.exit(main())
