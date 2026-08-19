---
type: reference
status: living
purpose: operational gotchas — the ways tooling, machines and monitoring lie to you here
created: 2026-08-09
scope: 操作 / 基建 / 工具调用层。分析与写作层在 paper_process_pitfalls.md；代码缺陷在 master_bug_catalog.md
---

# 操作层的坑 — 工具、机器与监控骗你的方式

> **三份文档的分工**（`paper_process_pitfalls.md` §9 位置约定表的操作层扩展）：
>
> | 文档 | 收什么 | 症状 |
> |---|---|---|
> | [[master_bug_catalog]] | 代码缺陷 | 有 B-number，能修 |
> | [[paper_process_pitfalls]] | 分析与写作过程 | 编译通过、log 干净、**结论是错的** |
> | **本文档** | 操作 / 基建 / 工具调用 | 命令退出码 0、**做的不是你以为的那件事** |
>
> **来源**：台账 `known.py` 里 **137 条祈使型 ADJUDICATED**（§17 → §448）归纳而成。
> 这里只放**可复用的判据**；具体实例用 §号指路，全文用
> `.venv/bin/python3 scripts/maintenance/known.py <关键词>` 查。
>
> **用法**：不要通读。做下面某件事之前，跳到对应那节。

---

## 0. 元规律 — 三个形状

### 0.1 ⭐ 拿字符串出现当结构判断

**出现频率最高，且每次都静默。** 你要判断的是「结构里有没有这个东西」，写出来的却是
「文本里有没有这几个字」——而注释、命令行、日志、别的标识符里都可能有这几个字。

| 实例 | 你以为在判断 | 实际匹配到 |
|---|---|---|
| §446.6 | 目标进程 | **自己的 shell**（pattern 出现在自己的 bash 命令串里）→ exit 144 |
| §446.6 二次 | `budget_watch.py` | 也匹配 `proxy_budget_watch.py`（**新名是旧名的超集**），`[c]` 技巧救不了 |
| §448.6 | config 有没有 `include_sites` 键 | 注释里的 `"only include_sites differs"` → 三个 config 静默漏改，而缺该键会让 `load_tasks` 载入 **0 个 task** |
| §183 | grep 到的残留数 | pandoc 在 `+` 后插了断点，模式只匹配连续形 → 报「0 残留」而实际 118 处仍在 |

> **判据：判断结构，就用结构解析。** `yaml.safe_load` / `ast` / `pgrep` 取 PID 再纯数字 kill。
> **判据：写下任何 pattern 之前，问「我这条命令自己的文本会不会被它匹配」**——
> 包括超集关系（新脚本名含旧脚本名）。
> 可靠动作：先跑一条**纯只读、命令里不含任何相关名字**的快照（`ps -eo pid,args > /tmp/x`），
> 人看一眼，再用纯数字操作。

### 0.2 检查通过 ≠ 被检查的东西健康

探针、健康检查、完成判据，测的往往是**它自己**能不能跑通。

- **§446.3 探针形状**：`max_tokens=1` 的探针在真实负载（4096）已被拒绝后仍报 healthy。
  一个不按真实负载形状发请求的健康检查，证明的是探针能活，不是被监控的东西能活。
- **§391.2 探测失败 vs 条件不满足**：monitor 里任何 `if ! <check>` 都必须能回答
  「这是条件不满足，还是我没测到」。远端探测尤其——网络错误和「任务没完成」长得一样。
- **§128.2 SSH chain 的 returncode 不可信**（经 Windows PowerShell 中转）：
  必须用 stdout sentinel（`__QSTAT_OK__`）+ double-probe guard。
- **§448.4 测试只验形状**：断言 `PROD_MAX_TOKENS == 4096` 与签名默认值，
  而把 HTTP payload 硬编码回 1 照样全绿——**那正是它声称防的回归**。

> **判据：健康检查必须与它所认证的负载同形状。**
> **判据：断言行为，不断言形状**——测线上 payload，不测模块常量；
> 测 AST 的实参绑定，不测源码里字符串出现几次。

### 0.3 两个同名的东西不是同一个东西

- **§446.7 / §449.6 时区**（**同一天踩了两次**）：DGX 是 BST(+0100)、A100 是 UTC(+0000)。
  第一次把两边日志时刻并排排序，**从 14 分钟的间隔编造出 74 分钟**，据此建了一个机制假说
  写进笔记和台账；第二次在诊断别的东西时顺手拿「DGX 现在 21:39」减「A100 log 20:36」，
  **把正常的 10 分钟 reset 读成卡死 63 分钟**，转头去查死锁。
  ⚠️ **本条当时就写在 GOTCHAS 里，没挡住** —— 因为两次都不是「要做时间线分析」的场合，
  是**顺手做了一次减法**。判据必须覆盖减法本身，不能只覆盖「分析」。
- **§445.1 两种分母**：run set（语料 − N/A，load 时）vs scored set（再 − protocol，分析时）。
  reddit 既是 205 又是 203，shopping 既是 435 又是 432，**两个都对**。
- **§393.3 硬规则**：success **rate** → 计分集；episode 级**覆盖率/计数** → 采集集。
- **§396.7 / §397.7**：稳健性检查 ≠ 估计量检查。换分母查「算得对不对」，
  查不出「这个量是不是你说的那个量」——后者只能问「这个名字的定义是什么」。

> **判据：减号两边来自不同主机，就先并排打两边 `date`，再做减法。**
> 不限于「时间线分析」——**任何一次跨机时刻相减都算**，包括顺手估个「跑了多久」。
> 单机内比较不受影响。差值会随夏令时变，**别把「差 1 小时」硬记成常量**。
> **判据：相除、相减、并排之前，先确认两边的定义一致**（memory `feedback-same-name-not-comparable`）。

---

## 1. 起一个长任务之前

- [ ] **done-condition 选 Tier 1**（文件 marker / log sentinel）> Tier 2（远端 job 状态）
      > Tier 3（PID `kill -0`）。**绝不用 pattern 自匹配**（见本文「元规律 0.1」）
- [ ] **有 max-wait 兜底吗**——done-condition 写错时不会静默死循环
- [ ] **STALL 阈值要大于最慢的正常静默段**。实测 Magento reset **33 分钟**不写 log，
      而 watchdog 的 step-stale 默认 10 分钟（§447）
- [ ] **fire-and-forget 必须配三件套**（§205.7）：launch-time `.lock` marker +
      success-path `.done` marker + 下个 cron tick 的 stale-lock 检测
- [ ] **monitor 的收尾动作若是「启动另一个任务」，必须校验那个任务活着**（§390.3）——
      fire 后 `sleep 60` + 真 PID `kill -0` 复验
- [ ] **until-loop 之后的 bash 只能 stat-only**，不嵌 python one-liner（嵌套引号在 `bash -c` 里必炸）
- [ ] **`grep` 无匹配返回 exit 1**，`set -euo pipefail` 下会杀掉脚本，且**恰在检查通过时**触发。加 `|| true`

## 2. 停一个任务之前

- [ ] **先只读快照，再纯数字 kill**（见本文「元规律 0.1」）。不要一条命令里既提脚本名又 pkill
- [ ] **watchdog 要跟 runner 一起杀**，否则孤儿 watchdog 一直 spam
      （memory `feedback-kill-watchdog-alongside-runner`）
- [ ] **杀完确认**：`ps` 复查 + GPU 显存归零 + 无残留 chromium

## 3. 跨机部署一个改动之前

- [ ] **`rsync` 打印的成功不算部署验证**。要在**远端**验：符号在不在、测试跑不跑得过（§448）
- [ ] **A100 的 git HEAD 不可信**：81 文件 / +7726 行「未提交修改」其实是历来 rsync 的结果，
      HEAD 停在很旧的 commit。**要知道远端跑的是哪版代码，只能比对文件内容**（§446.5）
- [ ] **bastion 的 banner 不等于受限 session**——`rsync` / `ssh -o ConnectTimeout` 能穿透
      （memory `reference-condenser-a100-infra`）
- [ ] **两块盘**：VWA fire 写 `/mnt/scratch`，WA 写 `vda1`。报磁盘警报前先问 fire 写哪块

## 4. 改 config 之前

- [ ] **验 `load_experiment_config()` 的解析结果，不验文件内容**（§405）。
      WA chain 那次 abort 就是跨层继承分叉——逐文件 diff 每个都正确
- [ ] **显式空列表 ≠ 选零个**：`if selected_ids and ...` 让 `task_ids: []` 静默变成**全选**，
      而六格的 `task_ids_sha` 还会「一致」——**直接击穿用 sha 做的 parity 验证**（§448 P1-5，已 fail-loud）
- [ ] **同一 run_dir 换过 config 就有 provenance 断裂**：`run_meta.json` 只记一个。
      中途换 config 后，旧 config 落下的 episode **在新集内的保留、在集外的必须清**（§447.3），
      否则 episode 数 ≠ expected_n，B-1834 gate 会在收尾时判 abort
- [ ] **子集 run 会撞死写的分母**：`queue_chain.sh` 的 `SITE_EXPECTED_N` 与
      `paper_scored_task_count` 都按全量算（§448.2，**未修，backlog**）

## 5. 调外部 API / 探测未知接口之前

- [ ] **候选表里放一个已知好的控制 + 一个故意错的探针**（§444.3）。
      控制组失败 = 是你的调用方式错了，不是对方不支持；故意错的那发，
      错误消息常常直接告诉你正确用法（`Use GET /model-api/models`）
- [ ] **注册表列出 ≠ 能调用**：5 个 Anthropic 条目标着价格却全部 400。
      **价格推不出可用性，加新模型前逐个打一发**（§444.4）
- [ ] **按 body 分类，不按状态码**：`403` 既可能是 `Budget exceeded` 也可能是
      `invalid api key`；只看状态码会把凭证过期报成没钱（§448.3）
- [ ] **控制组要先打，不是等结果诡异了才补**（§470.8）。arXiv API 在 DGX 上
      `http` 返 **301 未跟随**（curl 无 `-L` → 0 字节）、`https` 连打几发就
      **429 Rate exceeded**（14 字节）。两种失败在解析器眼里都是「没有 entry」，
      **和「这篇论文不存在」逐字节一样**。我据此一度报了四篇「查无此文」，
      是拿已 verified 的 ID 当控制才发现是自己的调用坏了。
      ⇒ 查文献前先打一发**已知存在**的 ID；控制组不过就别读结果。
      ⇒ 换出口重试（quark / A100 / 让 codex 代查），别在同一 IP 上硬刷 —— 退避
      7 分钟仍 429。**「未核成」必须报成未核成，不能报成查无此文。**

- [ ] **凭证占位符要无条件覆盖**：`if not key or key.startswith("DUMMY")`，
      不能用 `if KEY not in os.environ`（§43 / §73）

## 6. 相信另一个 agent 的结论之前

- [ ] **cross-AI 输出先验证再用**：大小 >2KB / 无「File not found」/ 发现数达 scope 下限 /
      ≥1 条 OOB / 引用了具体行号数字
- [ ] **1-AI unique 的 P0/P1 必须 100% 事实核验**（不抽样）。2-AI/3-AI 重合自带交叉确认，
      unique 没有那层过滤，而它们偏偏 blast 最大（/stress v7.10 Phase 4b）
- [ ] **它们的正确 finding + 我的错误实现 = 一个只在真实例上暴露的新 bug**（§187）
- [ ] **fixer-bias**：修 fixer 的工作必须由独立 reviewer 验，
      不接受「自己抓 flaw → 自己改 → 自己 self-confirm」（§133.3）
- [ ] **hallucination 降级之前用更宽的 grep 复验**（§205.1）

## 7. 修完一处之后

- [ ] **同形错误要一起找**。修完立刻问「这个形状还在哪里」，
      **用 `grep` 普查代替顺着调用链走**（§388.7.1，同型第三次）
- [ ] **sibling propagation 是 hard gate**：任一 site 加 atomic-write，
      必须 `grep write_text|open..w` 扫 siblings（§204.5 / §136.retro）
- [ ] **注释描述的契约必须有会红的东西守着**（§391.5）——
      三条同形 bug 的修法都是加测试/加断言，不是改注释
- [ ] **做 mutation 验证**：把修回退，确认测试真的会红。没红的测试等于没有

## 8. 宣布「完成」之前

- [ ] **测试基线要对照**：本次 23 failed 全部预先存在（`git stash` 逐个复跑核对）。
      不核对就不知道是不是自己弄坏的
- [ ] **`echo "done"` 不是完成证据**——它无论前一条命令成不成功都会打印
- [ ] **跑 `/stress`**（milestone 前强制）+ chain Mode B (codex) + Mode C (agy)
- [ ] **写笔记 + 台账**：过去与 WHY → 笔记；「这个量测过吗」→ 台账；
      live 状态 → `next_steps §0`（memory `feedback-chronicle-on-milestone`）

---

## 9. 本项目特有的硬规则（违反会毁数据，不只是麻烦）

- **同 site 同时只能跑一个 baseline**（B0 XOR B1 XOR B2）——共享 docker + 同一用户账号，
  并发会造成 server-side session race + cart/comment 交叉污染
- **同一物理 host 同时只能跑一条 site chain**（cls XOR red XOR shop）——§A2.11 P0-5
- **禁止裸 `python scripts/run_experiment.py`**，必须走 queue script（§104）——
  裸 runner 实证造成过 paper-grade 数据污染
- **condition 之间必须 reset 站点**（§38）
- **删 task 结果统一用 `clear_tasks.py`**，不手动 `rm`
- **分析脚本一律经 `pass1_run_manifest.json` 白名单取 run，禁止裸 glob**（§442.8）

---

> **准入判据**：这个坑**未来还会遇到**，而且**当时不会报错**。
> 只发生一次、或者会被现有测试抓到的，不进这里——进 `master_bug_catalog.md`。
