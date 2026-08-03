#!/usr/bin/env python3
"""Magento 状态面分析 — shopping reset 替代方案的实证基础。

两个互补的视角 (见 docs/reference/shopping_reset_state_surface.md):

  closure  外键反向闭包 → 「回滚必须覆盖的表」**上界** (理论可能被牵连)
  probe    UPDATE_TIME 时间轴分段 → 实验**实际**改动了哪些表 (穷举, 含无 FK 的表)

闭包给上界, probe 给实际。两者对照才能判定有无遗漏 —— 替代 reset 方案的成立条件
恰恰是「没有遗漏」, 所以枚举式验证 (我列表再核对) 不够格: 漏掉的永远不会暴露。

用法:
  analyze_magento_state_surface.py closure
  analyze_magento_state_surface.py probe <probe.tsv> [--since 'YYYY-MM-DD HH:MM:SS']
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "docs" / "reference" / "data"

# 评测可见的状态面 —— 从 VWA/WA shopping 的 program_html eval URL 反推。
# 数字 = 引用该 URL 的 eval 条目数 (VWA/WA)。
SEEDS = {
    "quote": "cart (/checkout/cart, VWA 104× / WA 5×)",
    "sales_order": "order (shopping_get_latest_order_url, VWA 116× / WA 10×)",
    "wishlist": "wishlist (/wishlist/, VWA 57× / WA 15×)",
    "catalog_compare_item": "compare (/catalog/product_compare, VWA 4×)",
    "customer_entity": "account (/customer/account/, VWA 2×)",
    "customer_address_entity": "address (/customer/address/, VWA 5× / WA 10×)",
    "newsletter_subscriber": "newsletter (/newsletter/manage/, WA 1×)",
}

# 容器启动期就会被写的表 (base_url patch / reindex / cron), 与实验无关。
# probe 分析时从「实验改动面」里扣掉, 但**必须显式列出**而不是静默过滤 ——
# 静默过滤会让「其实是实验写的」被误算成启动噪声。
STARTUP_NOISE = {
    "core_config_data", "queue_poison_pill", "design_config_grid_flat",
    "customer_grid_flat", "catalog_category_product_index_store1",
    "catalog_product_index_eav", "cataloginventory_stock_status",
    "cataloginventory_stock_status_tmp", "catalog_product_index_website",
    "catalog_product_index_price", "indexer_state", "cron_schedule",
}


def _load_fk() -> tuple[dict[str, set[str]], dict[tuple[str, str], str]]:
    edges_f, rules_f = DATA / "magento_fk_edges.tsv", DATA / "magento_fk_rules.tsv"
    if not edges_f.exists():
        sys.exit(f"缺少 {edges_f} — 见 docs/reference/shopping_reset_state_surface.md §4 重新导出")
    rev: dict[str, set[str]] = collections.defaultdict(set)
    for line in edges_f.read_text().strip().split("\n"):
        child, parent = line.split("\t")[:2]
        rev[parent].add(child)
    rules = {}
    for line in rules_f.read_text().strip().split("\n"):
        p = line.split("\t")
        if len(p) >= 3:
            rules[(p[0], p[1])] = p[2]
    return rev, rules


def cmd_closure(_args) -> None:
    rev, rules = _load_fk()
    union: set[str] = set()
    for seed, why in SEEDS.items():
        seen, stack = set(), [seed]
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            stack.extend(rev.get(t, ()))
        union |= seen
        cascade = sorted(c for c in seen - {seed} if rules.get((c, seed)) == "CASCADE")
        other = sorted(c for c in seen - {seed}
                       if rules.get((c, seed)) not in ("CASCADE", None))
        deep = sorted(seen - {seed} - set(cascade) - set(other))
        print(f"\n=== {seed}  [{why}]  闭包 {len(seen)} 张 ===")
        print(f"  直接 CASCADE ({len(cascade)}): {', '.join(cascade) or '(无)'}")
        if other:
            print(f"  非 CASCADE ({len(other)}): "
                  + ", ".join(f"{c}[{rules[(c, seed)]}]" for c in other))
        if deep:
            print(f"  间接 ({len(deep)}): {', '.join(deep)}")
    print(f"\n{'=' * 70}\n并集 {len(union)} 张 (全库 370)")
    print(", ".join(sorted(union)))
    print("\n⚠️ customer_entity 的闭包大是因为闭包答的是「**删掉**这行会牵连谁」, "
          "而实验从不删 seed 用户 —— 见文档 §4.1 删除型 vs 恢复型。")


def cmd_probe(args) -> None:
    path = pathlib.Path(args.tsv)
    if not path.exists():
        sys.exit(f"找不到 {path} (A100: ~/workspace/p79/logs/magento_table_probe.tsv)")
    rows = [l.split("\t") for l in path.read_text().strip().split("\n")[1:]]
    rows = [r for r in rows if len(r) >= 4 and r[2] != "(container-absent)"]
    if not rows:
        sys.exit("probe 文件里还没有有效样本 —— 这表示还没采到, 不表示「测出为零」")

    starts = sorted({r[1] for r in rows})
    print(f"样本 {len(rows)} 行, 容器实例 {len(starts)} 个: {', '.join(starts)}")

    latest: dict[str, str] = {}
    for r in rows:
        if r[3] > latest.get(r[2], ""):
            latest[r[2]] = r[3]

    if args.since:
        mutated = {t: u for t, u in latest.items() if u > args.since}
        print(f"\n=== {args.since} 之后被写的表 ({len(mutated)}) ===")
    else:
        mutated = latest
        print(f"\n=== 全部被写过的表 ({len(mutated)}) ===")

    noise = {t: u for t, u in mutated.items() if t in STARTUP_NOISE}
    real = {t: u for t, u in mutated.items() if t not in STARTUP_NOISE}
    for t, u in sorted(real.items(), key=lambda kv: kv[1]):
        print(f"  {t:<44} {u}")
    print(f"\n启动期噪声 ({len(noise)}, 已按名单归类, 未静默丢弃): "
          + ", ".join(sorted(noise)))

    rev, _ = _load_fk()
    closure: set[str] = set()
    for seed in SEEDS:
        seen, stack = set(), [seed]
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            stack.extend(rev.get(t, ()))
        closure |= seen
    outside = sorted(set(real) - closure)
    print(f"\n{'=' * 70}")
    print(f"落在 66 张外键闭包**外**的 ({len(outside)}) — 即评测盲区的潜在污染通道:")
    print("  " + (", ".join(outside) if outside else "(无)"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("closure").set_defaults(fn=cmd_closure)
    p = sub.add_parser("probe")
    p.add_argument("tsv")
    p.add_argument("--since", help="只看该时刻之后被写的表 (runner 启动时刻)")
    p.set_defaults(fn=cmd_probe)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
