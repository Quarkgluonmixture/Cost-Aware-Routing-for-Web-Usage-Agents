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

# 2026-08-03 首次容器启动时观测到被写的表 (base_url patch / reindex / cron)。
#
# ⚠️ **这是参考名单, 不是过滤器** (P1-1-A 修正 2026-08-04, /stress milestone)。
# 早先版本拿它把「启动噪声」从结果里扣掉 —— 那等于用一个 12 元素的硬编码集合
# 把 370 张表的穷举结论过滤回了枚举, 而本文档 §5 的核心主张恰恰是「只有穷举够格」。
# 更要命的是这里面 `core_config_data` 是 storefront 可写的、`cron_schedule` 是后台任务表:
# 若实验真写了它们, 过滤会**静默扣掉一条污染通道**。
# 现在只用于交叉标注 (`[启动期也见过]`), 绝不从输出里删除任何表。
STARTUP_OBSERVED = {
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
    raw = [l.split("\t") for l in path.read_text().strip().split("\n")[1:]]
    raw = [r for r in raw if len(r) >= 4]

    # sentinel 行不是数据, 但它们的**存在**是数据: 标记时间轴上容器缺席 / DB 不可达的窗口。
    # 不静默丢弃 —— 「没采到」和「测出为零」必须可区分。
    absent = [r for r in raw if r[2] == "(container-absent)"]
    dbdown = [r for r in raw if r[2] == "(db-unavailable)"]
    rows = [r for r in raw if r[2] not in ("(container-absent)", "(db-unavailable)")]
    if not rows:
        sys.exit("probe 文件里还没有有效样本 —— 这表示还没采到, 不表示「测出为零」")

    # ── 按容器实例分组 (P1-2-A 修正 2026-08-04) ────────────────────────────
    # MariaDB 的 InnoDB `UPDATE_TIME` 是**内存态 table stats, 容器重建即归零**。
    # 每个 condition 的 reset 都 `docker rm -f` 重建容器, 所以时间轴天然是分段的。
    # 早先版本对全体样本取 max, 会把 condition A 期间的写入显示在 condition B 的
    # 结果里 —— 正好模糊掉这份实证要回答的「跨 condition 污染」问题。
    by_inst: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_inst[r[1]].append(r)

    print(f"有效样本 {len(rows)} 行 / 容器实例 {len(by_inst)} 个"
          f" / sentinel: container-absent {len(absent)}, db-unavailable {len(dbdown)}")
    if absent or dbdown:
        print("  ⚠️ 上述 sentinel 时刻**没有观测**, 不可读作「该时段无表变动」")

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

    for inst, irows in sorted(by_inst.items()):
        first_probe = min(r[0] for r in irows)
        latest: dict[str, str] = {}
        for r in irows:
            if r[3] > latest.get(r[2], ""):
                latest[r[2]] = r[3]
        shown = ({t: u for t, u in latest.items() if u > args.since}
                 if args.since else latest)
        print(f"\n{'=' * 70}")
        print(f"容器实例 started={inst}  (首次采样 {first_probe}, 被写表 {len(shown)}"
              + (f", 已按 --since {args.since} 过滤" if args.since else "") + ")")
        for t, u in sorted(shown.items(), key=lambda kv: kv[1]):
            tags = []
            if t in STARTUP_OBSERVED:
                tags.append("启动期也见过")
            if t not in closure:
                tags.append("★闭包外")
            print(f"  {t:<44} {u}  {' '.join(tags)}")
        outside = sorted(set(shown) - closure)
        print(f"  ── 落在 66 张外键闭包**外** ({len(outside)}) = 评测盲区的潜在污染通道: "
              + (", ".join(outside) if outside else "(无)"))

    print(f"\n{'=' * 70}")
    print("注: `启动期也见过` 仅是与 2026-08-03 首次启动观测的交叉标注, **不代表已排除** ——")
    print("    core_config_data / cron_schedule 都是 storefront 可写的表。要判定某张表是")
    print("    启动写的还是实验写的, 用 --since <runner 启动时刻> 按时间切, 不要按表名切。")


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
