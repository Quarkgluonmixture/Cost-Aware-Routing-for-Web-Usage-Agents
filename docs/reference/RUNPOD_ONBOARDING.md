# RunPod Onboarding — 14-cell Phase A Paper-Grade Re-run

**Status**: 🟡 plan ready, blocked on advisor RunPod 经费 approval (2026-04-30 sync)
**Companion docs**:
- `MYRIAD_SMOKE_REPORT.md` (why not Myriad — §4.4 5-barrier rejection)
- `VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md` (Phase A patches that re-run requires)
- `ADVISOR_SYNC_DRAFT_2026-04-30.md` (sync framing for cost / wallclock asks)

---

## 1. Architecture — Hub-spoke 3-machine setup

```
[quark Windows]  ─Tailscale─  [DGX Spark]  ─Tailscale─  [RunPod 4090]
  100.95.81.103                spark-9ea3                runpod-4090 (TBD)
  │                            │                          │
  └─ VWA Docker (sites)         ├─ B0 实验 (proxy API)      ├─ B1 实验 (4B local)
     cls:9980                   ├─ Analysis + paper writing  └─ Results rsync to DGX
     red:9999                   │
     shop:7770                  └─ git master (truth source)
     wiki:8888
```

**Key insight**: RunPod 容器允许装 Tailscale → RunPod 在 Tailnet 内 reach quark VWA Docker, 跟 DGX 一样的 network position. UCL Myriad 不行 (firewall block CGNAT, see Myriad report §4 — 物理级 blocked).

**为啥 hub-spoke 不是 mirror**:
- B0 (proxy API) 不要 GPU → 留 DGX shared CPU 完全够 (no $0.6/h GPU 浪费)
- B1 (4B local) 严重受 DGX GPU 争抢 → 移 RunPod 4090 dedicated (~3-5× faster)
- Analysis + paper writing 在 DGX hub (results 全 rsync 回来)
- 1 个 VS Code Remote-SSH 到 DGX 就够, RunPod 通过 tmux/ssh 触发

---

## 2. 14-cell 分发 — 7+7 strategy

| Cell | 跑哪里 | ETA per cell | 总 ETA |
|---|---|---:|---:|
| **B0 cls 5-mode** (DOM/SoM/Vision/P-text/P-SoM) | DGX (proxy API) | ~3.5 min/ep × 234 ≈ 14h | 70h (parallel cells share CPU) |
| **B0 red 5-mode** | DGX | ~3.5 min/ep × 210 ≈ 12h | 60h |
| **B0 shop 5-mode** | DGX (cautious — site bug 排错先) | ~3.5 min/ep × 466 ≈ 27h | 135h |
| **B0 P-prompt diamond** cls/shop | DGX | ~14h + 27h | 41h |
| **B1 cls 5-mode** | RunPod 4090 | ~3.5 min/ep × 234 ≈ 14h | 70h |
| **B1 red 5-mode** | RunPod | ~12h | 60h |
| **B1 shop 5-mode** | RunPod | ~27h | 135h |
| **B1 P-prompt diamond** | RunPod | ~41h | 41h |

**Wallclock estimate**:
- DGX cells (B0 总 ~306h) — 多 cell 并行 (~5-7 cells parallel, B0 不抢 GPU) → ~50-70h wallclock
- RunPod cells (B1 总 ~306h) — single 4090 sequential → ~50-60h wallclock × $0.6/h = $30-36
- Combined (parallel): ~3-5 days wallclock total

**Cost recalc** (per advisor sync §3-4):
- $52-87 actual GPU hours for B1
- + 30% buffer (crash/retry/idle) = $70-115
- $200 ask = head-room for additional probes (Q3 extended / Tier 5 evaluator / Section 5 ad-hoc)

---

## 3. RunPod 7-step onboarding (post-advisor approval)

```bash
# 1. RunPod web → pick 4090 instance ($0.6/h), choose PyTorch 2.x container
#    (24GB VRAM, 4B Qwen3-VL ~10GB bf16 fits with plenty of headroom)

# 2. SSH in, install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname runpod-4090
# Browser to authenticate, link to your Tailscale account

# 3. 验证 Tailscale 通了 (核心 unlock)
ping -c 3 100.95.81.103   # quark VWA host
curl -s -o /dev/null -w "%{http_code}\n" --max-time 10 http://100.95.81.103:9980/  # cls
curl -s -o /dev/null -w "%{http_code}\n" --max-time 10 http://100.95.81.103:9999/  # red
curl -s -o /dev/null -w "%{http_code}\n" --max-time 10 http://100.95.81.103:7770/  # shop
# 都应返回 200

# 4. Clone repo + checkout 当前 master
git clone https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents
cd Cost-Aware-Routing-for-Web-Usage-Agents
# Pin commit hash for paper-grade integrity (current master after sync push)
git checkout <pinned-hash-after-push>

# 5. Setup .venv + deps
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis]"
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128

# 6. 复制 auth files (manual scp from DGX, NOT via git — gitignored)
mkdir -p .auth
# from DGX side:
#   scp .auth/qwen_api runpod-4090:Cost-Aware-Routing-for-Web-Usage-Agents/.auth/
#   scp .auth/glm runpod-4090:Cost-Aware-Routing-for-Web-Usage-Agents/.auth/
# from RunPod side: verify
ls -la .auth/  # qwen_api + glm should be there

# 7. Smoke test B1 cls 1 task (验证 chain 完整)
export VWA_REMOTE_HOST=100.95.81.103
bash scripts/queues/queue_baseline.sh B1 dom classifieds  # full cell
# OR: just run 1 task to smoke-test:
.venv/bin/python3 scripts/run_experiment.py \
    --config configs/exp_v2_B1_dom_classifieds.yaml \
    --run_id smoke_test_B1_cls \
    --task_id 0 \
    --max_steps 5 \
    --log_path logs/smoke_test_B1_cls.log
# Expect: ep done, no errors, action selection sane
```

---

## 4. Sync mechanism — rsync over Tailscale

**On DGX** (hub):
```bash
# Pull single cell
rsync -avz --progress \
    runpod-4090:Cost-Aware-Routing-for-Web-Usage-Agents/results/visualwebarena/phase1/B1_phantom_text_classifieds_20260501/ \
    /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results/visualwebarena/phase1/B1_phantom_text_classifieds_20260501/

# Pull all RunPod B1 cells (after 14-cell wave done)
rsync -avz --progress \
    runpod-4090:Cost-Aware-Routing-for-Web-Usage-Agents/results/visualwebarena/phase1/B1_*/ \
    /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results/visualwebarena/phase1/

# 一键: make rsync-from-runpod RUN=B1_phantom_text_classifieds_20260501
# (TBD Makefile target — add when RunPod live)
```

**Why rsync not git**:
- Results JSONL + screenshots ~100MB-1GB per cell
- `.gitignore` already excludes `results/visualwebarena/`
- rsync over Tailscale = LAN-speed (~100MB/s+)
- 一次性传输, 不需要 version history

---

## 5. Watchdog 跨 host 配置

每个 cell 自动启动 watchdog (queue scripts 处理). 区分 host:

```bash
# DGX watchdog ntfy topic (现有)
NTFY_TOPIC=p79-exp-dgx-spark

# RunPod watchdog ntfy topic (新增, 在 RunPod 上 export)
NTFY_TOPIC=p79-exp-runpod
```

→ 你 phone 上区分两边 alert. 不需要 cross-host watchdog coordination (each cell 独立).

---

## 6. Failure modes + mitigation

| 风险 | 概率 | Mitigation |
|---|---|---|
| **quark Windows wsl --shutdown** (VWA 全 stop) | 中 | RunPod 跑实验前**桌面贴便签**: "RunPod B1 in-flight, 不要 wsl --shutdown". DGX 有 watchdog auto-recovery. |
| RunPod 容器被 reboot (provider-side) | 低 | RunPod dedicated instances 稳定; 偶尔 evict. 实验 resume 机制存在 (queue scripts idempotent skip). |
| Tailscale 重连失败 | 低 | RunPod auto-reconnect; manual `tailscale up` 修. |
| API key (.auth) 没复制 | 高 if 忘 | 第 6 步 scp 是唯一 manual step, 失败 runner 会立即 error out. |
| RunPod billing 超 budget | 中 | RunPod 有 max-budget cap, 设 $200 hard limit. |

---

## 7. Compatibility — current code 直接跑 RunPod

✅ Phase A patches (commit `3c15cd7` + later) 全 site-host-agnostic:
- `VWA_REMOTE_HOST` env var 控制 site 解析 (`scripts/vwa_env_remote.sh` 模式)
- B1 model 通过 transformers 加载 (no host-specific path)
- Watchdog ntfy 通过 env var 区分
- queue scripts 自动检测 reset / cleanup / log dir

✅ 不需要 RunPod-specific 代码 fork — 同 master, 同 commit hash, 不同 environment.

---

## 8. New session checkpoint (2026-04-30 EOD)

**为啥写这个 doc**: hub-spoke 架构 + 7-step onboarding 在 conversation 里讨论但没落档. New session 启动后 read this + `MYRIAD_SMOKE_REPORT.md §4.4` (why not Myriad) + `ADVISOR_SYNC_DRAFT_2026-04-30.md` (advisor framing) → 完整 picture.

**Triggering action**: 等学长会议 approve RunPod 经费 → 走第 3 节 7-step onboarding.

**Other checkpoint deliverables**:
- 5 unpushed commits 待 push (含 b37 probe + paper-strategy refinements + Myriad rejection)
- B1 P-text cls 仍跑 (PID 2280869, ~198/234 ep, ETA 1-2 天)
- next_steps.md §0 TL;DR + §6 advisor checklist 是 new session 主入口

---

## 9. References

- Myriad rejected: `MYRIAD_SMOKE_REPORT.md` (UCL firewall §4 + Docker rejection §4.4)
- Phase A code: `VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md` + `master_bug_catalog.md`
- Pilot validation: `docs/analysis/cross_sites/pilot_t0_wave3_final.md` (60 ep PASS)
- Advisor sync: `ADVISOR_SYNC_DRAFT_2026-04-30.md` (~250 字 with 4 asks + empirical anchors)
- Paper strategy: `PAPER_STRATEGY_OPEN_QUESTIONS.md` (9-issue audit)
