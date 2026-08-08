#!/usr/bin/env python3
"""B-1969 v4 — 把 v3 的成簇结论去掉 episode confound。

v3 证明长尾成簇 (p=0.000), 但成簇有两个互斥的解释:
  (甲) 站点侧时间窗口 —— 一个 cron 窗口拖慢**该时刻附近的所有 step**,
       无视它们属于哪个 episode ⇒ 成簇会**跨 episode 边界**
  (乙) task/页面因素 —— 某些 task 的页面本来就重 (大 DOM / 复杂表单),
       该 episode 的 step 普遍慢 ⇒ 成簇**完全在 episode 内部**

检验: 把长尾标签**只在各自 episode 内部**随机重排 (保持每个 episode 的长尾
数量与 step 时刻不变)。这样 (乙) 被完整保留, (甲) 被打散。
  若实测成簇 ≈ 该基线 ⇒ 成簇由 (乙) 解释, 无站点侧证据
  若实测成簇 ≫ 该基线 ⇒ 存在跨 episode 的时间窗口, 支持 (甲)
"""
import json, glob, os, re, collections, statistics, random
from datetime import datetime, timedelta

random.seed(20260808)
RUNS = sorted(glob.glob("results/visualwebarena/phase1/B*_classifieds_2026*"))
THRESH = 12000.0

eps = []  # 每个 episode: list[(abs_t, env)]
for r in RUNS:
    for sf in glob.glob(os.path.join(r, "*", "episodes", "*_summary_v2.json")):
        try:
            s = json.load(open(sf))
            base = datetime.fromisoformat(s["wallclock_start"])
        except Exception:
            continue
        stf = sf.replace("_summary_v2.json", "_steps_v2.jsonl")
        if not os.path.exists(stf):
            continue
        acc, cur = 0.0, []
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
            cur.append((base + timedelta(milliseconds=acc), lat["env_step"]))
            acc += (lat.get("total") or 0.0)
        if cur:
            eps.append(cur)

allst = [(t, e, i) for i, ep in enumerate(eps) for (t, e) in ep]
allst.sort(key=lambda x: x[0])
n, k = len(allst), sum(1 for x in allst if x[1] >= THRESH)
print(f"episodes {len(eps)}  steps {n}  长尾 {k} ({100*k/n:.2f}%)")

# 用索引化表示: 每个 step 一个全局 id, 记录 (时刻, episode_id)
gid = [(t, i) for (t, e, i) in allst]
obs_flags = [x[1] >= THRESH for x in allst]

def cluster_stats(flag_bools):
    ts = [gid[j][0] for j in range(n) if flag_bools[j]]
    ts.sort()
    if len(ts) < 3:
        return None, None
    g = [(ts[i+1]-ts[i]).total_seconds() for i in range(len(ts)-1)]
    return statistics.median(g), sum(1 for x in g if x <= 60)/len(g)

obs_med, obs_frac = cluster_stats(obs_flags)
print(f"\n实测: 中位间隔 {obs_med:.1f}s, <=60s 占 {100*obs_frac:.1f}%")

# 按 episode 分组全局下标
by_ep = collections.defaultdict(list)
for j in range(n):
    by_ep[gid[j][1]].append(j)

print("\n=== 基线 1: 完全随机重排 (v3 的做法, 同时打散甲和乙) ===")
meds, fracs = [], []
for _ in range(100):
    f = [False]*n
    for j in random.sample(range(n), k):
        f[j] = True
    m, fr = cluster_stats(f)
    meds.append(m); fracs.append(fr)
meds.sort(); fracs.sort()
print(f"  中位间隔 p50 {meds[50]:.1f}s   <=60s 占比 p50 {100*fracs[50]:.1f}%")

print("\n=== 基线 2: episode 内重排 (保留乙, 打散甲) ⭐ 关键对照 ===")
meds2, fracs2 = [], []
for _ in range(100):
    f = [False]*n
    for ep_id, idxs in by_ep.items():
        cnt = sum(1 for j in idxs if obs_flags[j])
        if cnt:
            for j in random.sample(idxs, cnt):
                f[j] = True
    m, fr = cluster_stats(f)
    meds2.append(m); fracs2.append(fr)
meds2.sort(); fracs2.sort()
print(f"  中位间隔: p5 {meds2[5]:.1f}s  p50 {meds2[50]:.1f}s  p95 {meds2[95]:.1f}s")
print(f"  <=60s 占比: p5 {100*fracs2[5]:.1f}%  p50 {100*fracs2[50]:.1f}%  p95 {100*fracs2[95]:.1f}%")
p2m = sum(1 for m in meds2 if m <= obs_med)/len(meds2)
p2f = sum(1 for fr in fracs2 if fr >= obs_frac)/len(fracs2)
print(f"  → 实测 {obs_med:.1f}s 相对该基线的单侧 p = {p2m:.3f}")
print(f"  → 实测 {100*obs_frac:.1f}% 相对该基线的单侧 p = {p2f:.3f}")

print("\n=== 补充: 长尾在 episode 之间的集中度 ===")
per_ep = [sum(1 for j in idxs if obs_flags[j]) for idxs in by_ep.values()]
per_ep_nonzero = [c for c in per_ep if c]
print(f"  有长尾的 episode: {len(per_ep_nonzero)} / {len(by_ep)} = {100*len(per_ep_nonzero)/len(by_ep):.1f}%")
print(f"  每 episode 长尾数: 中位 {statistics.median(per_ep_nonzero):.0f}, "
      f"max {max(per_ep_nonzero)}, 均值 {statistics.mean(per_ep_nonzero):.2f}")
top = sorted(per_ep_nonzero, reverse=True)
print(f"  最集中的 20 个 episode 吃掉: {sum(top[:20])} / {k} = {100*sum(top[:20])/k:.1f}% 的长尾")
