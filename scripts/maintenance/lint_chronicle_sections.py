#!/usr/bin/env python3
"""实验笔记 § 号 lint —— 并发 session 撞号的检测与取号入口。

**为什么需要**: § 号是全局资源, 但「取最大值 +1」这个分配动作**不是原子的**。
两个并发 session 各自 `grep | tail` 拿到同一个号, 再各自 append —— git **不会报冲突**,
因为两次 append 落在文件的不同位置, 文本层无重叠。**冲突只存在于语义层**。
2026-08-04 已是第二次 (前一次 §424, 本次 §428)。

用法:
  lint_chronicle_sections.py            # 检查; 重复 → exit 1
  lint_chronicle_sections.py --next     # 打印下一个可用 §, append 前先跑这个
  lint_chronicle_sections.py --file X   # 指定别的 chronicle

注: 笔记历史上存在**乱序**(号小的 § 出现在文件更后面), 那是并发写入的既成事实,
不是错误 —— 所以顺序只报告不 fail; **只有重复号才 fail**。
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

# CJK 文件名: 用 pathlib 直读, 不经 shell glob (见 memory reference_cjk_filename_shell_glob_trap)
DEFAULT = (pathlib.Path(__file__).resolve().parents[2]
           / "docs" / "checkpoints" / "实验笔记.md")

SEC_RE = re.compile(r"^## (\d+)\.\s*(.*)$")
SUB_RE = re.compile(r"^#{3,4} (\d+)\.(\d+)")

# Ratchet: lint 上线 (2026-08-04) 时既已存在的撞号。它们发生在 6 月, 可能已被其他文档
# 交叉引用, 改号风险大于收益 → 豁免, 但**每次运行都列出来**提醒未清。
# **新增撞号一律 fail** —— 这才是 lint 的用途 (同 index_bug_catalog.py 的 "going forward" 模式)。
# 清理任一组后, 从这里删掉对应号即可收紧 ratchet。
KNOWN_DUPES = {328, 330}


def parse(path: pathlib.Path):
    secs: list[tuple[int, int, str]] = []      # (lineno, num, title)
    subs: list[tuple[int, int, int]] = []      # (lineno, parent, child)
    for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if m := SEC_RE.match(line):
            secs.append((i, int(m.group(1)), m.group(2).strip()))
        elif m := SUB_RE.match(line):
            subs.append((i, int(m.group(1)), int(m.group(2))))
    return secs, subs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(DEFAULT))
    ap.add_argument("--next", action="store_true", help="打印下一个可用 § 号后退出")
    args = ap.parse_args()

    path = pathlib.Path(args.file)
    if not path.exists():
        print(f"✗ 找不到 {path}", file=sys.stderr)
        return 2
    secs, subs = parse(path)
    if not secs:
        print(f"✗ {path} 里没有匹配 '## N. 标题' 的 §", file=sys.stderr)
        return 2

    nums = [n for _, n, _ in secs]
    if args.next:
        print(max(nums) + 1)
        return 0

    rc = 0

    # 1. 重复号 —— 硬失败。这是并发撞号的唯一可靠信号。
    dup = {n: [(ln, t) for ln, m, t in secs if m == n]
           for n, c in collections.Counter(nums).items() if c > 1}
    new_dup = {n: v for n, v in dup.items() if n not in KNOWN_DUPES}
    old_dup = {n: v for n, v in dup.items() if n in KNOWN_DUPES}

    if new_dup:
        rc = 1
        print(f"✗ 新增重复 § 号 {len(new_dup)} 组:")
        for n, entries in sorted(new_dup.items()):
            print(f"  §{n} 出现 {len(entries)} 次:")
            for ln, t in entries:
                print(f"    L{ln}: {t[:64]}")
        print("  → 按**先到先得**处理: 保留日期早的, 后写的改成 max+1, 内容不动,")
        print("    并在改号的 § 下加一行说明 (它的 commit message 里引用的是旧号)。")
    else:
        print(f"✓ 无新增重复 § 号 ({len(secs)} 个, 最大 §{max(nums)})")

    if old_dup:
        print(f"⚠ 历史遗留撞号 {len(old_dup)} 组 (ratchet 豁免, 未清): "
              + ", ".join(f"§{n}×{len(v)}" for n, v in sorted(old_dup.items())))
        print("    改号需先核查交叉引用; 清理后从 KNOWN_DUPES 删掉对应号以收紧 ratchet。")
    missing = sorted(KNOWN_DUPES - set(dup))
    if missing:
        print(f"✓ ratchet 可收紧: §{', §'.join(map(str, missing))} 已不再重复, "
              "请从 KNOWN_DUPES 移除")

    # 2. 孤儿子章节 —— ### N.x 找不到对应的 ## N.
    orphan = sorted({p for _, p, _ in subs} - set(nums))
    if orphan:
        rc = 1
        print(f"✗ 孤儿子章节: §{', §'.join(map(str, orphan))}.x 没有对应的 '## N.' 父节")
    else:
        print(f"✓ 子章节归属正确 ({len(subs)} 个)")

    # 3. 乱序 —— 只报告。历史遗留的并发写入既成事实, 不是错误。
    inv = [(secs[i - 1], secs[i]) for i in range(1, len(secs)) if secs[i][1] < secs[i - 1][1]]
    if inv:
        print(f"⚠ 文件顺序与 § 号不单调 ({len(inv)} 处, 仅提示不失败):")
        for (l1, n1, _), (l2, n2, _) in inv[-3:]:
            print(f"    L{l1} §{n1} → L{l2} §{n2}")

    print(f"\n下一个可用: §{max(nums) + 1}"
          "   (append 前跑 `--next`; 并发 session 仍可能撞, 靠本 lint 在 commit 前捕获)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
