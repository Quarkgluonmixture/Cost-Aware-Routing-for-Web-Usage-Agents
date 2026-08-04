# Shopping (Magento) reset — 状态面实证

> **这份文档回答一个问题**: shopping 的 reset 现在要 60+ 分钟重建整个容器,
> 其中**多少是协议要求的, 多少是实现选择的副作用**?
>
> 前置裁定见 `master_bug_catalog.md` B-1954 + 实验笔记 §428.13:
> **重建容器不是协议要求**。预注册只写 `RESET_BEFORE=1` "ensures clean start state" —— 规定的是**结果**,
> 不是手段; AMENDMENT_01 无任何容器条款。本文档给出替换手段所需的实证基础。
>
> **状态**: 静态分析已完成; 实证采集进行中 (见 §5)。**不要把 §5 的空白读成「测出是零」** ——
> 那只表示还没采到。

---

## 1. 三站的 reset 手段差异 (为什么只有 shopping 贵)

| 站 | 手段 | 代价 | 为什么能这么做 |
|---|---|---|---|
| classifieds | HTTP `POST page=reset` + 5 表 SQL 哨兵 + PHP 缓存清 + `docker **restart**` | ~30-60s | 应用自带 reset 端点 |
| reddit | `docker rm -f` + `run` | ~60-120s | postmill 镜像自 seed, **无需重算的派生结构** |
| shopping | `docker rm -f` + 重建 + **全量 reindex** | **60+ min** | 同一手段搬到 141GB Magento 上, 代价放大两个量级 |

reddit 的手段之所以成立, 是因为它没有「需要重算一小时的派生结构」。Magento 有, 所以同一个动作的语义完全不同。

## 2. 重建代价的去向 (实测)

容器 `vwa-shopping` 2026-08-03 23:12:02 新建。MariaDB 10.6 的 InnoDB `UPDATE_TIME` 实时,
且容器创建后未重启 → **370 张表里只有 12 张有 `UPDATE_TIME`**, 其余 358 张从未被这个 mysqld 实例写过
(镜像里的数据文件是 baked-in 的)。这 12 张即「重建后必须重算的东西」:

| 表 | rows | MB | 性质 |
|---|---|---|---|
| `catalog_category_product_index_store1` | 368,734 | 19.5 | 商品目录派生索引 |
| `catalog_product_index_price` | 107,592 | 13.5 | 商品目录派生索引 |
| `cataloginventory_stock_status` | 102,090 | 4.5 | 商品目录派生索引 |
| `catalog_product_index_eav` | 104,720 | 4.5 | 商品目录派生索引 |
| `catalog_product_index_website` / `cataloginventory_stock_status_tmp` | — | — | 同上 |
| `core_config_data` | 54 | — | base_url patch 写入 |
| `design_config_grid_flat` / `customer_grid_flat` | 3 / 27 | — | 启动期 grid 重建 |
| `indexer_state` / `cron_schedule` / `queue_poison_pill` | — | — | 运行时状态 |

**`catalogsearch_fulltext` 不在 MySQL 里** —— 容器内跑着 Elasticsearch (`%fulltext%` 表查询为空,
`ps` 见 `org.elasticsearch.bootstrap.Elasticsearch`)。**那 60 分钟的大头是往 ES 灌 10 万+ 商品**,
`indexer_state.catalogsearch_fulltext` 是 11 个 indexer 里最后 valid 的一个。

> **关键**: 这些全是**商品目录的派生物**。实验 (加购物车 / 下单 / 改地址) **从不改商品目录**,
> 所以它们在实验前后**恒等**。现状 = 「重建整个数据库来清一个购物车」。
>
> ⚠️ 例外: **WA `shopping_admin`** 任务会改商品 (价格 / 库存 / 名称) → 索引真的失效。
> 本文档的结论**不覆盖 admin**, 当前这轮 fire (VWA shopping + WA shopping storefront) 不含 admin。

## 3. 评测可见的状态面 (静态, 下界)

从 `program_html` eval 的 URL 反推评测**读**哪些状态:

| 状态 | VWA | WA | 承载表 |
|---|---|---|---|
| `func:shopping_get_latest_order_url()` | 116 | 10 | `sales_order*` |
| `/checkout/cart` | 104 | 5 | `quote` / `quote_item` |
| `/wishlist/` | 57 | 15 | `wishlist` / `wishlist_item` |
| `/customer/address/` | 5 | 10 | `customer_address_entity` |
| `/catalog/product_compare` | 4 | — | `catalog_compare_item` |
| `/customer/account/` | 2 | — | `customer_entity` |
| `/newsletter/manage/` | — | 1 | `newsletter_subscriber` |

> **这只是下界**。评测**读**的状态面 ≠ 实验**改**的状态面 —— agent 完全可能改了评测不读的东西
> (写 review / 改账户设置), 那些改动照样污染下一个 condition。所以必须有 §5 的穷举实证。

**另**: VWA shopping 有 **19 个任务标 `require_reset=true`**, 清一色是「加购物车」任务
(224-229, 244-245, 267-273, 289, 320-322)。语义完全吻合「这任务会改购物车, 跑完要清」——
但 `require_reset` 在代码里**只对 classifieds 实现** (`envs.py:172`, 带 `TODO(jykoh)`),
在 shopping 上是 **no-op**。

## 4. 外键闭包 (静态, 上界)

从 §3 的 seed 出发做反向外键闭包 (392 条 FK 边), 得**并集 66 张表**。分析脚本见
`scripts/maintenance/probe_table_mutations.sh` 同目录的调查记录。

| seed | 闭包 | 直接 CASCADE |
|---|---|---|
| `quote` | 9 | `quote_address` / `quote_id_mask` / `quote_item` / `quote_payment` |
| `sales_order` | 24 | 12 张 (invoice / shipment / creditmemo / payment / ...) |
| `wishlist` | 3 | `wishlist_item` |
| `catalog_compare_item` | 1 | (无) |
| `customer_address_entity` | 6 | 5 张 EAV 属性表 |
| `newsletter_subscriber` | 3 | `newsletter_problem` / `newsletter_queue_link` |
| `customer_entity` | **54** | 21 张 — 见下方警告 |

### 4.1 两类回滚语义 (难度差一个量级)

`customer_entity` 闭包炸到 54 张是个**误导** —— 闭包答的是「**删掉**这行会牵连谁」,
而实验**从不删** emma.lopez。于是回滚天然分两类:

- **删除型** (`quote` / `wishlist` / `catalog_compare_item` / `sales_order` 簇):
  实验**新建**的行 → 直接 `DELETE`, **不需要知道 seed 原值**
- **恢复型** (`customer_entity` + 5 张 EAV 属性表 + `customer_address_entity`):
  实验**改写**了 seed 行 → **必须有 seed 基线**才能还原

第二类正是历史上踩过的坑 (任务把用户名/地址改掉 → 后续任务全废)。它需要一份 seed dump,
而不只是一条 DELETE 语句。

### 4.2 闭包的已知盲区

**没有外键的行为记录表不在任何闭包里**, 但会被 storefront 写入 —— 例如
`search_query` (agent 大量搜索必然写) / `report_event` / `report_viewed_product_index` /
`customer_log` / `customer_visitor` / `persistent_session`。

这正是 §5 穷举实证要补的部分: 闭包给上界, `UPDATE_TIME` 给实际。

## 5. 穷举实证 (采集中)

`scripts/maintenance/probe_table_mutations.sh` — cron `*/10`, 只读
(`SELECT information_schema`), 对 fire 无副作用。输出 `logs/magento_table_probe.tsv` (A100)。

每行 `probe_ts | container_started | table_name | update_time | table_rows | data_mb`。
`container_started` 用于识别跨 condition 的容器重建; 容器缺席 (reset 窗口) 记
`(container-absent)` sentinel 保持时间轴连续。

**为什么用穷举而非枚举**: 「哪些表会被改」如果靠我列举再验证, 只能确认我想到的,
**漏掉的永远不会暴露**。替代 reset 方案的成立条件恰恰是「**没有遗漏**」, 所以只有
扫全部 370 张表、按时间戳分段的证据够格。

分析方法 (数据落地后) —— **用脚本, 不要手写 awk**:
```bash
scripts/maintenance/analyze_magento_state_surface.py probe logs/magento_table_probe.tsv \
  [--since '<runner 启动时刻>']
```

> ⚠️ **两条曾经写错的分析纪律** (P1-1-A / P1-2-A, /stress 2026-08-04):
>
> 1. **必须按容器实例分段, 不能对全体样本取 max**。MariaDB 的 InnoDB `UPDATE_TIME` 是
>    **内存态 table stats, 容器重建即归零**; 每个 condition 的 reset 都 `docker rm -f` 重建,
>    所以时间轴天然分段。全局取 max 会把 condition A 的写入显示在 condition B 的结果里 ——
>    正好模糊掉这份实证要回答的「跨 condition 污染」问题。脚本已按 `container_started` 分组。
> 2. **不能按表名过滤「启动噪声」**。早先版本用一个 12 元素硬编码集合把 §2 那些表扣掉,
>    等于用枚举把 370 张表的穷举结论过滤了回去。实测反例: `core_config_data` 在实例 2 的
>    `UPDATE_TIME` 是 **01:38:46**, 而 runner 01:10 就启动了 —— 它在**实验期**也被写。
>    要区分启动写 vs 实验写, **按时间切 (`--since`), 不要按表名切**。

**待查 (2026-08-04 首轮数据, 勿当结论)**: storefront-only 的 condition 里出现了
`admin_user`(02:25:40) / `admin_user_session`(05:00:02) / `authorization_rule` / `flag` /
`queue_lock`。可能来自 watchdog 的 auth_refresh 或 Magento 后台 cron, **尚未查证** ——
在确认来源前不可读作「实验污染了 admin 表」。

**待确证** (不要读成「已测出为零」):
- [ ] 实验期实际被写的表全集
- [ ] 其中有多少落在 §4 的 66 张闭包外 (= 评测盲区的污染通道)
- [ ] 恢复型回滚需要的 seed 基线字段范围
- [ ] 「回滚这些表 ≡ 重建容器」的等价性验证 (需比对 fresh 容器 vs 跑过任务的容器的表级 diff)

## 6. 当前裁定

**本轮 fire 按重建跑**, 不改手段。7 × 60min ≈ 7h 纯 reset, 占 11 天的 2.6% —— 可接受,
且 estimand 干净。

替代方案 (保留容器 + 只回滚受影响表) **不违反预注册**, 但属 **estimand-adjacent**:
必须先完成 §5 的等价性验证才能换, 且换的时机应在 fire 之间而非 fire 中。
