#!/usr/bin/env python3
"""B-1969 污染面分析 — cls Phase 1a 是否被「站点周期性不回答」污染。

v2 (2026-08-08) — 三家 /stress 审计 (Claude + codex + gemini) 推翻 v1 后重写。
v1 的四个错误都由这一版结构性堵死:

  1. **denominator 用 glob**。v1 `glob("B*_classifieds_2026*")` 扫到 4257-4259 个
     episode, 混进了 B0_som replicate(20260803) / B1_3mode stale / B3_som。canonical
     是 18 run × 224 = 4032。仓库**早就有**这个白名单 (`pass1_run_manifest.json`,
     B-1896 / 笔记 §367), 它的 `_why` 原话就是防 glob 混入 stale run —— v1 绕过了它。
     → 本版一律经 `load_canonical()`。

  2. **把「每 episode 两条延迟记录」读成「两次超时」**。v1 报「156 次超时尝试」;
     实际 `sum(reset_goto_timeout_count)=78`、`sum(reset_goto_retry_count)=78`,
     而 `reset_goto_latency_ms_per_attempt` 每 episode 有 2 个条目 (1 次超时 + 1 次
     成功重试), 156 是**条目数**。→ 本版分别报三个量, 不再从条目数推事件数。

  3. **反事实基线跨 cell 混合 → Simpson's paradox**。v1 拿「同 task id、未触发的
     episode」跨全部 18 cell 汇总得 12.65%, 再与受影响组的 5.13% 比, 得 p=0.024。
     但受影响 episode 53% 来自 B2 (cell SR <2%), 基线却被 B0 (15-29%) 拉高。
     分层后效应消失 (见 `stratified_test`)。→ 本版只做 cell 分层比较,
     混合口径仅作为**演示偏倚**保留并显式标注 BIASED。

  4. **p=0.000 与未落盘的蒙特卡洛**。v1 用 100 次 permutation 却报 p=0.000
     (有效 Monte Carlo p 的下限是 plus-one 估计 1/(N+1)); drop-one 敏感性分析
     则完全没进版本控制, 数字无法复现。→ 本版 N_PERM=20000、一律 plus-one,
     且敏感性分析就在本文件里。

⚠️ 本脚本**不能**识别 B-1969 的因果效应, 且这是设计上的天花板而非实现不足:
   - `reset_goto_timeout_count` 只在 env.reset 阶段记账 (`vwa_wrapper.py:404`),
     episode 中段撞上窗口它看不到 ⇒ 77 是**探测下界**, 不是发生率。
   - 唯一能直接定因的证据 (episode 请求时刻 ↔ 容器内部 POST 时刻对齐) **已永久不可得**:
     Phase 1a 期间的 cls 容器日志随 Gate-3 逐 condition `docker restart` 滚掉了。
   - `wallclock_start` 在 `_run_episode` 之前打点 (`runner/main.py:1861`), 即 **reset 之前**,
     而 flagged episode 的 reset 耗时 median ~50s、max ~205s ⇒ 用它重建的 step 时刻
     系统性偏移, 且偏移量大于聚集分析测到的效应本身。故 `clustering_analysis()`
     标注为 NON-IDENTIFYING, 只描述「长延迟在时间上有结构」, 不主张那是 B-1969。

用法: python3 scripts/analysis/scan_b1969_contamination.py [--out results/b1969_scan.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import statistics
import sys
from datetime import datetime, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(REPO, "results/phantom_paper/l1_router/pass1_run_manifest.json")
PHASE1 = os.path.join(REPO, "results/visualwebarena/phase1")
TAIL_MS = 12000.0          # B-1969 实测的「站点不回答」窗口下界
N_PERM = 20000
SEED = 20260808


def plus_one_p(hits: int, n: int) -> float:
    """(hits+1)/(n+1) — Monte Carlo p 的无偏下界。永远不会返回 0."""
    return (hits + 1) / (n + 1)


def load_canonical():
    """只取 pass1_run_manifest 白名单里的 cls run。防 glob 混入 replicate/stale。"""
    man = json.load(open(MANIFEST))
    runs = []
    for key, lst in man["pass1"].items():
        if key.endswith("_classifieds"):
            runs.extend(lst)
    return sorted(runs)


def load_episodes(runs):
    """→ list[dict]: run/model/mode/task/success/flagged/reset_latencies/env_steps"""
    out = []
    for run in runs:
        m = re.match(r"(B\d)_(.+?)_classifieds_", run)
        model, mode = m.group(1), m.group(2)
        for sf in glob.glob(os.path.join(PHASE1, run, "*", "episodes", "*_summary_v2.json")):
            try:
                s = json.load(open(sf))
            except Exception:
                continue
            task = int(re.search(r"task_(\d+)_", os.path.basename(sf)).group(1))
            envs, base = [], None
            stf = sf.replace("_summary_v2.json", "_steps_v2.jsonl")
            if os.path.exists(stf):
                try:
                    base = datetime.fromisoformat(s["wallclock_start"])
                except Exception:
                    base = None
                acc = 0.0
                for line in open(stf, errors="replace"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    lat = d.get("latency_ms") or {}
                    if not isinstance(lat, dict) or lat.get("env_step") is None:
                        continue
                    envs.append({
                        "idx": d.get("step_idx", -1),
                        "env": lat["env_step"],
                        # ⚠️ 近似时刻: 不含 reset 耗时, 见模块 docstring
                        "t": (base + timedelta(milliseconds=acc)) if base else None,
                    })
                    acc += (lat.get("total") or 0.0)
            out.append({
                "run": run, "model": model, "mode": mode, "task": task,
                "success": bool(s.get("success")),
                "flagged": bool(s.get("reset_goto_timeout_count")),
                "timeout_count": s.get("reset_goto_timeout_count") or 0,
                "retry_count": s.get("reset_goto_retry_count") or 0,
                "reset_latencies": s.get("reset_goto_latency_ms_per_attempt") or [],
                "recovered": s.get("reset_goto_recovered"),
                "error": s.get("error"),
                "benchmark_noise": s.get("benchmark_noise"),
                "env_steps": envs,
            })
    return out


def incidence(eps):
    """发生率 — 明确区分「超时事件数」「重试数」「延迟条目数」(v1 把三者搞混)。"""
    flagged = [e for e in eps if e["flagged"]]
    firsts = [e["reset_latencies"][0] for e in flagged if len(e["reset_latencies"]) >= 2]
    lasts = [e["reset_latencies"][-1] for e in flagged if len(e["reset_latencies"]) >= 2]
    with_tail = sum(1 for e in flagged if any(s["env"] >= TAIL_MS for s in e["env_steps"]))
    any_tail = sum(1 for e in eps if any(s["env"] >= TAIL_MS for s in e["env_steps"]))
    return {
        "episodes_total": len(eps),
        "flagged_episodes": len(flagged),
        "flagged_pct": 100 * len(flagged) / len(eps),
        "timeout_events": sum(e["timeout_count"] for e in flagged),
        "retry_events": sum(e["retry_count"] for e in flagged),
        "latency_entries": sum(len(e["reset_latencies"]) for e in flagged),
        "first_attempt_median_s": statistics.median(firsts) / 1000 if firsts else None,
        "successful_retry_median_s": statistics.median(lasts) / 1000 if lasts else None,
        "all_recovered": all(e["recovered"] for e in flagged),
        "any_error": any(e["error"] for e in flagged),
        "any_benchmark_noise": any(e["benchmark_noise"] for e in flagged),
        # 关键: flagged 里真正有 post-reset 长 step 的只是少数 → 不能说「整个 episode 降级」
        "flagged_with_post_reset_tail": with_tail,
        "flagged_without_post_reset_tail": len(flagged) - with_tail,
        "episodes_with_any_tail": any_tail,
        "episodes_with_any_tail_pct": 100 * any_tail / len(eps),
    }


def stratified_test(eps, rng):
    """核心检验: 在每个 (model,mode) cell 内部比 flagged vs 同-task 未 flagged。

    v1 的混合口径同时输出, 标注 BIASED, 用于展示 Simpson's paradox 的量级。
    """
    hit_tasks = {e["task"] for e in eps if e["flagged"]}
    cells = collections.defaultdict(lambda: {"hit": [], "clean": []})
    for e in eps:
        if e["flagged"]:
            cells[e["run"]]["hit"].append(e["success"])
        elif e["task"] in hit_tasks:
            cells[e["run"]]["clean"].append(e["success"])

    O = sum(sum(c["hit"]) for c in cells.values())
    E = sum(len(c["hit"]) * (sum(c["clean"]) / len(c["clean"]))
            for c in cells.values() if c["hit"] and c["clean"])

    # 分层 permutation: 每个 cell 内把 flagged 标签重新随机分配给同 cell 同-task 池
    hits = 0
    for _ in range(N_PERM):
        tot = 0
        for c in cells.values():
            if not c["hit"] or not c["clean"]:
                continue
            pool = c["hit"] + c["clean"]
            tot += sum(rng.sample(pool, len(c["hit"])))
        if tot <= O:
            hits += 1

    biased_hit = [e["success"] for e in eps if e["flagged"]]
    biased_clean = [e["success"] for e in eps if not e["flagged"] and e["task"] in hit_tasks]
    return {
        "observed_successes": O,
        "expected_successes_stratified": E,
        "gap": E - O,
        "p_stratified_plus_one": plus_one_p(hits, N_PERM),
        "n_perm": N_PERM,
        "BIASED_pooled_flagged_sr": 100 * sum(biased_hit) / len(biased_hit),
        "BIASED_pooled_clean_sr": 100 * sum(biased_clean) / len(biased_clean),
        "_biased_note": "混合口径仅供展示 Simpson's paradox 量级; 勿引用",
        "cell_baseline": {r: (sum(c["clean"]) / len(c["clean"]) if c["clean"] else None)
                          for r, c in cells.items()},
    }


def drop_one_sensitivity(eps, cell_baseline, rng, n_sim=10000):
    """drop-one oracle 对「受影响 episode 反事实成功」的敏感性。

    v1 此段从未落盘 (在 heredoc 里跑的) → 数字不可复现, 这是 codex F4 的核心指控。
    反事实概率用**同 cell 同-task** 基线, 不用 v1 那个跨 cell 混合的 12.65%。
    报 p50 + p95 + p99, 不把中位数写成 bound。
    """
    modes = sorted({e["mode"] for e in eps})
    out = {}
    for model in sorted({e["model"] for e in eps}):
        sub = [e for e in eps if e["model"] == model]
        tasks = {e["task"] for e in sub}
        base_succ = collections.defaultdict(set)
        for e in sub:
            if e["success"]:
                base_succ[e["mode"]].add(e["task"])
        flagged = [(e["run"], e["mode"], e["task"]) for e in sub if e["flagged"]]

        def drop_one(extra):
            succ = {m: set(base_succ[m]) for m in modes}
            for (_, md, t) in extra:
                succ[md].add(t)
            full = len(set().union(*succ.values())) if succ else 0
            res = {}
            for m in modes:
                u = set().union(*[succ[k] for k in modes if k != m]) if len(modes) > 1 else set()
                res[m] = (full - len(u)) / len(tasks) * 100
            return res

        obs = drop_one([])
        sims = collections.defaultdict(list)
        for _ in range(n_sim):
            flip = [f for f in flagged if rng.random() < (cell_baseline.get(f[0]) or 0.0)]
            d = drop_one(flip)
            for m in modes:
                sims[m].append(abs(d[m] - obs[m]))
        out[model] = {
            m: {
                "observed_pp": obs[m],
                "shift_p50_pp": sorted(sims[m])[n_sim // 2],
                "shift_p95_pp": sorted(sims[m])[int(0.95 * n_sim)],
                "shift_p99_pp": sorted(sims[m])[int(0.99 * n_sim)],
            } for m in modes
        }
    return out


def clustering_analysis(eps, rng):
    """长延迟 step 在时间上是否成簇。

    ⚠️ NON-IDENTIFYING — 三重限制 (见模块 docstring):
      (a) 时刻由 wallclock_start + 累积 step total 重建, **不含 reset 耗时**
          (median ~50s), 该偏移大于本分析测到的效应本身;
      (b) episode 内重排这个对照会把「该 episode 有很多长 step」这类站点窗口效应
          一并条件掉, 因此偏保守 —— 它只检验残余的位置结构;
      (c) 长延迟本身不特异于 B-1969 (页面重也会长)。
    结论只能是「长延迟在时间上有结构」, 不能是「B-1969 造成了它」。
    """
    flat = []
    for i, e in enumerate(eps):
        for s in e["env_steps"]:
            if s["t"]:
                flat.append((s["t"], s["env"], i))
    flat.sort(key=lambda x: x[0])
    if len(flat) < 100:
        return {"skipped": "insufficient timestamped steps"}
    n = len(flat)
    obs_flags = [x[1] >= TAIL_MS for x in flat]
    k = sum(obs_flags)

    def med_gap(flags):
        ts = [flat[j][0] for j in range(n) if flags[j]]
        if len(ts) < 3:
            return None
        g = [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
        return statistics.median(g)

    obs = med_gap(obs_flags)
    by_ep = collections.defaultdict(list)
    for j in range(n):
        by_ep[flat[j][2]].append(j)

    hits = 0
    n_iter = 2000  # 该分析 non-identifying, 用较少迭代即可
    for _ in range(n_iter):
        f = [False] * n
        for idxs in by_ep.values():
            c = sum(1 for j in idxs if obs_flags[j])
            if c:
                for j in rng.sample(idxs, c):
                    f[j] = True
        m = med_gap(f)
        if m is not None and m <= obs:
            hits += 1
    return {
        "IDENTIFYING": False,
        "_why_not": "时刻不含 reset 耗时(median ~50s > 本效应); episode 内重排把站点窗口效应条件掉了; 长延迟不特异",
        "observed_median_gap_s": obs,
        "p_within_episode_null_plus_one": plus_one_p(hits, n_iter),
        "n_perm": n_iter,
        "tail_steps": k,
        "total_steps": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="写 JSON 结果到此路径")
    a = ap.parse_args()
    rng = random.Random(SEED)

    runs = load_canonical()
    eps = load_episodes(runs)
    print(f"canonical runs: {len(runs)}   episodes: {len(eps)}")
    if len(eps) != len(runs) * 224:
        print(f"  ⚠️ 期望 {len(runs)*224}, 实得 {len(eps)}", file=sys.stderr)

    inc = incidence(eps)
    print("\n=== 发生率 (探测下界, 非真实发生率) ===")
    for k in ("flagged_episodes", "episodes_total", "flagged_pct", "timeout_events",
              "retry_events", "latency_entries", "first_attempt_median_s",
              "successful_retry_median_s", "all_recovered",
              "flagged_with_post_reset_tail", "flagged_without_post_reset_tail",
              "episodes_with_any_tail", "episodes_with_any_tail_pct"):
        print(f"  {k:34s} {inc[k]}")

    st = stratified_test(eps, rng)
    print("\n=== cell-stratified 检验 (主结论) ===")
    print(f"  实测成功 O = {st['observed_successes']}")
    print(f"  分层期望 E = {st['expected_successes_stratified']:.2f}   缺口 {st['gap']:.2f}")
    print(f"  分层 permutation plus-one p = {st['p_stratified_plus_one']:.4f}  (N={st['n_perm']})")
    print(f"  [BIASED 仅供对照] 混合口径 {st['BIASED_pooled_flagged_sr']:.2f}% vs "
          f"{st['BIASED_pooled_clean_sr']:.2f}%")

    ds = drop_one_sensitivity(eps, st["cell_baseline"], rng)
    print("\n=== drop-one oracle 敏感性 (cell-specific 反事实; 报 p50/p95/p99) ===")
    for model, per in ds.items():
        worst = max(per.items(), key=lambda kv: kv[1]["shift_p95_pp"])
        print(f"  {model}: 最大 p95 偏移 {worst[1]['shift_p95_pp']:.2f}pp (mode={worst[0]}, "
              f"p50={worst[1]['shift_p50_pp']:.2f}, p99={worst[1]['shift_p99_pp']:.2f})")

    cl = clustering_analysis(eps, rng)
    print("\n=== 聚集分析 (NON-IDENTIFYING, 见 docstring) ===")
    print(f"  中位间隔 {cl.get('observed_median_gap_s')}s   "
          f"plus-one p = {cl.get('p_within_episode_null_plus_one')}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"runs": runs, "incidence": inc, "stratified": st,
                   "drop_one_sensitivity": ds, "clustering": cl},
                  open(a.out, "w"), indent=2, default=str)
        print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
