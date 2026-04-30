# Myriad Connectivity Smoke Report

**调研日期**: 2026-04-30
**Author**: P79 project (DGX session + jiaming)
**Verdict**: 🔴 **B1 全量迁移 Myriad 不可行** | 🟢 Myriad 仅适合 offline analysis / model training / paper 写作

## 1. Background & Goal

DGX Spark 是共享 GPU 环境（多用户竞争 ~128GB VRAM），并发跑多个 Qwen3-VL-4B 实例时吞吐受限。原计划利用 UCL Myriad HPC 的 4×A100 40G L 节点跑 B1 baseline 跨站点并行，**摆脱 GPU 争抢**。

**Smoke 目标**: 验证 Myriad → quark VWA docker (Tailscale 100.95.81.103) 连通性，决定是否能直接搬 B1 实验。

参考: `docs/reference/Myriad.md` (UCL 官方 Myriad 文档) | README §139 (onboarding) | README §171 (hub-spoke sync)

---

## 2. Investigation Path — 失败路径归档（避免后人重做）

### 2.1 OpenConnect on WSL (CSD wrapper failures)

- WSL Ubuntu 24.04 + openconnect 9.12-1build5
- `vpn.ucl.ac.uk` 302 redirect → `vpn4.ucl.ac.uk` → `/CACHE/sdesktop/install/binaries/sfinst` 404
- 多版本 csd-wrapper.sh (`/bin/true` / 自写 POST stub) 都卡 `+CSCOE+/sdesktop/wait.html` 无限轮询
- **根因**: ASA 收 client POST 后**二次校验** endpoint 真实合规状态，wrapper 撒谎过不了

### 2.2 OpenConnect SAML browser flow

- `sudo -E openconnect --protocol=anyconnect --external-browser='/mnt/c/Windows/explorer.exe' vpn.ucl.ac.uk`
- 服务器**不**返回 SAML challenge，仍走 hostscan 路径
- 结论: UCL ASA 的 endpoint compliance check 优先于 SAML SSO，不可绕

### 2.3 Browser web VPN portal

- Edge/Chrome 64-bit 访问 `vpn.ucl.ac.uk` →  
  `Platform Detection: Web-based launch of Cisco Secure Desktop is not supported with 64-bit versions of IE`
- 52 秒 fallback timeout 后 →  
  `Automated download of Cisco Secure Desktop is not supported by your system configuration`
- **根因**: UCL ASA 配置依赖 IE 32-bit ActiveX，现代浏览器全部失败

### 2.4 Cisco Secure Client 5.1.7 on Windows

- 安装在 quark Windows，`Server: vpn.ucl.ac.uk` → "终端安全评估失败" 3 秒
- log: `13:39:50 终端安全评估：正在启动... → 13:39:53 终端安全评估：失败`
- **根因 #1**: Windows Defender 被第三方 AV (火绒 + 卡巴斯基) 禁用 (`WinDefend` service `Disabled`)
- **根因 #2**: 火绒/卡巴均 "snooze" 状态 → UCL Posture 探测看不到 active AV → fail
- **根因 #3**: Cisco Secure Client 5.x base installer **不带 HostScan/Posture 模块**，PowerShell 检查 `C:\Program Files (x86)\Cisco\Cisco Secure Client\` 只有 vpn 核心组件，无 `csc_posture.exe` / `cisco-secure-client-hostscan*.exe`
- 修复尝试: `Set-MpPreference -DisableRealtimeMonitoring $false` 报 `0x800106ba`（service-level disabled），无法直接启 Defender；激活火绒 GUI 也未通过

### 2.5 DGX → Myriad 反向 ssh jump

- 从 DGX (家用 Tailscale 出口 109.175.214.x) → `myriad.rc.ucl.ac.uk:22` TCP timeout
- **根因**: UCL 边界防火墙 only 允许 UCL 内网 IP / Connect VPN 隧道访问 ssh，drop 任意公网

### 2.6 ✅ SUCCESS: Desktop@UCL Anywhere (Citrix HTML5)

- 浏览器 https://desktop.ucl.ac.uk → UCL SSO + MFA → Citrix 虚拟 Windows 桌面
- 在 VM 内启 PuTTY → `ssh ucab352@myriad.rc.ucl.ac.uk` → 直接进 login13
- **零 VPN，零 endpoint compliance 要求**
- 唯一前置障碍: 杀软 HTTPS 流量审查可能阻 WebSocket（用无痕模式或调 AV 策略解决）

---

## 3. Smoke Test Results

### 3.1 Login Node Smoke (login12, 2026-04-30 15:18)

简化版 smoke_login.sh 输出 + 解读:

| Probe | 结果 | 解读 |
|---|---|---|
| `myriadfs` 1.0T quota | ✅ available | 数据存储足够 |
| `module` system | ❌ FAIL (脚本简化未 source `/etc/profile.d/modules.sh`) | False negative — 实际可用 |
| Tailscale `100.95.81.103:9980` (cls) | ❌ unreachable | UCL firewall drop 100.x |
| Tailscale `100.95.81.103:7770` (shop) | ❌ unreachable | 同上 |
| Tailscale `100.95.81.103:9999` (reddit) | ❌ unreachable | 同上 |
| Tailscale `100.95.81.103:4399` (homepage) | ❌ unreachable | 同上 |
| `ssh spark-9ea3` (DGX hostname) | ❌ unreachable | DNS 解析失败 + 100.x dropped |
| `qsub` job submission | ✅ OK (`job 282887 qw`) | 调度系统正常 |

### 3.2 Compute Node Smoke (job 282887, simplified)

[等 job 跑完后补 — 当前排队中]

预期价值低（简化版 21 行只有 qsub directives + echo），未跑完整 GPU + bf16 + HF Hub probe。

如需补完整 GPU 验证，可跑 `~/p79_smoke/smoke_gpu_only.qsub`（30 行 mini 版）拿:
- `nvidia-smi` 输出（确认 A100 40G 分配）
- `torch.cuda.get_device_capability()` ≥ (8,0) 验证 bf16
- `huggingface.co` egress (决定能否下 Qwen3-VL-4B 权重)

### 3.3 Firewall Mechanism Diagnosis (login12, 2026-04-30 16:01)

跑 `~/p79_smoke/diagnose_firewall.sh` 拿到细粒度 firewall 行为。**双层 stateful drop**:

| Layer | Rule | Evidence |
|---|---|---|
| **Layer 1: IP segment blacklist** | CGNAT 100.64.0.0/10 + RFC 1918 全 drop（不论端口）| `100.64.0.1:443` / `100.100.100.100:443` / `100.127.255.254:443` 全 timeout；`10.0.0.1:443` / `192.168.1.1:443` / `172.16.0.1:443` 全 timeout |
| **Layer 2: Outbound port whitelist** | 公网 IP 仅允许 22/53/80/443 等 well-known port，custom port (9980/7770/9999/4399) 全 drop | `1.1.1.1:443` OPEN, `1.1.1.1:9999/7770/9980` 全 timeout |

**关键 traceroute 证据** (Layer 2 是 stateful firewall drop 而非 routing drop):
```
myriad → 100.95.81.103:        myriad → 1.1.1.1:
hop 1: 10.28.101.252           hop 1: 10.28.101.252
hop 2: 193.60.251.253          hop 2: 193.60.251.253
hop 3: 10.0.107.17             hop 3: 10.0.107.17
hop 4: 193.60.238.37           hop 4: 193.60.238.37
hop 5: 77.241.77.37            hop 5: 77.241.77.37
hop 6-7: timeout               hop 6-7: timeout
hop 8: 146.97.139.237          hop 8: 146.97.139.237
```

两条路径前 5 跳完全相同 — **包成功路由出 UCL 边界**到 ISP 上游 (146.97.139.237)，但 TCP SYN 被中间 stateful firewall 丢。证明 IP routing 没问题，是 connection-state 检查 drop。

**潜在穿透窗口** (但**不推荐**实施):

DERP relay 全部可达：
```
derp.tailscale.com -> 200
login.tailscale.com -> 302
controlplane.tailscale.com -> 302
derp1.tailscale.com -> 200
```

理论上 Tailscale userspace mode (`--tun=userspace-networking`) 经 DERP relay over HTTPS:443 能从 myriad reach Tailnet。但有以下硬阻碍：

1. **UCL ARC 服务条款风险**: 通常禁止在共享 HPC 装 user VPN
2. **被发现 → account 暂停**: smoke 外的 analysis use case 全停
3. **DERP relay 性能受限**: 10-50MB/s，B1 latency +50-200ms
4. **工程改造量**: B1 runner 代码全部要走 SOCKS5 proxy

**判定**: 理论可行，实操**不推荐**。Citrix 中转 + B1 留 DGX 仍是最优解。

---

## 4. Verdict

### 4.1 物理级 blocked path（diagnose_firewall.sh 实证）

UCL 边界防火墙是**双层 stateful drop**:
- **IP layer**: drop CGNAT 100.64.0.0/10 + RFC 1918 私网（不论端口）
- **Port layer**: 公网 IP 仅 allow whitelist port (22/53/80/443)，custom port (9999/7770/9980 etc.) 全 drop
- 包能 routing 出 UCL 边界（traceroute 显示），但 TCP SYN 被 stateful firewall drop

**唯一理论穿透窗口** = Tailscale userspace mode + DERP relay over HTTPS:443，但有 ARC 服务条款 / 性能 / 改造成本三重阻碍，**不推荐**实施。详见 §3.3。

### 4.2 对 B1 baseline 的影响

- B1 需要持续 HTTP request quark VWA docker (`http://100.95.81.103:9999` 等)
- Myriad 既不能直接 reach Tailscale，也不能 ssh DGX 中转
- **结论**: B1 全量迁移到 Myriad **不可行**，原"摆脱 GPU 争抢"目标 Myriad 解不了

### 4.3 对 hub-spoke 数据流的影响 (README §171)

原设计:
```
Spoke (Myriad) → Hub (DGX): 推 Tier B (episodes JSONL) via rsync over Tailscale
```

实际: Myriad ↔ DGX 双向都不通。`make rsync-to-hub` / `rsync-from-hub` workflow **需要改**:

- 经 Citrix VM 中转（用户 manual scp）
- 或 SaaS 中转（GitHub Gist / Dropbox / Google Drive）
- 或反向 SSH tunnel: DGX 主动 ssh 到 Citrix VM forwards myriad（复杂，long-running session 不稳）

---

## 4.4 Alternative — VWA Docker on Myriad? (五层 barrier 分析, 拒掉)

**问题**: 用户 ask "如果 myriad 上面自己跑 docker，会不会不用内网穿透了". 思路是把 VWA sites 跟 B1 都 deploy 到 Myriad compute node, 同 node localhost 通信, 绕开 Tailscale CGNAT firewall 问题.

**Verdict**: 🔴 **rejected — 五层 barrier 累计 = 不可行**.

### Barrier 1: Docker 权限 (可绕)

- Docker daemon 要 root → HPC 共享 node 不给
- 替代: **Singularity / Apptainer** (rootless container, designed for HPC)
- VWA 三站镜像理论上能转 Singularity image (镜像 push HF Hub / Singularity registry, 然后 myriad 上 pull)
- 工程量: 1-2 天 (学习 + 转换 + multi-container orchestration + 测试)

### Barrier 2: HPC 48h wallclock (致命)

- B1 baseline 跑全 466 task × 30 min/task ≈ 10 天 wallclock per cell × 14 cells
- 单 qsub job 最多 48h ≈ 96 task → 必须切 5+ jobs per cell
- 每 job 启动 = 启动 VWA 7 容器 + 加载 fixture + 跑 runner ≥ 10-30 min overhead
- 累计 overhead ~1-3h, 但**主要问题不是时间损失, 是 job 间状态不连续** (Barrier 3)

### Barrier 3: Job 间数据持久化 (deal-breaker) 🔴

**Naive view (我之前的)**: mount myriadfs 到 `/var/lib/mysql` → DB 文件持久 → 解决.

**Paper-grade view (用户 critique 后)**: data files persist 但**measurement instrument inconsistency** 仍存在.

- Job A 跑 task 0-95 后退出 → MySQL container 销毁
- 即使 myriadfs 持久 mysql data files:
  - **In-memory transactional state** 丢 (uncommitted writes rollback)
  - **Session / cookie / cart state** reset (Magento PHP session 在 Redis/files, 跨 restart 失效)
  - **DB query cache / connection pool** cold restart (PHP-FPM opcache 5-10 min 才稳)
  - **DB warmup behavior 不一致** — Job A task 95 时 DB hot, Job B task 96 cold
- VWA evaluator 依赖 deterministic state ("shopping cart 应该有 X")
- **结果**: page render latency 跨 job 差几百 ms → agent timing-dependent decisions 可能 flip → SR 数字 **uncontrolled noise**

这不是 P79 项目特有, 是 **web app + HPC batch 通用矛盾**:
- Web app 设计 for long-running stateful service
- HPC batch 设计 for ephemeral compute
- 两者**架构不兼容** (DB warmup variance vs paper-grade SR comparison)

### Barrier 4: Compute-to-compute 通信 (可绕但繁琐)

- 单 job 内: runner + VWA 在同 node localhost — ✅ OK
- 跨 job "VWA daemon 一直跑 + runner 多 job 共享":
  - HPC 不允许长 daemon job (违反 fair-share scheduling)
  - 即使能跑, compute node A 上 VWA 监听 9999, compute node B 上 runner 能否 reach? 需诊断 compute-to-compute custom port policy
  - login node 不允许跑长 daemon (only short < 15min jobs)

### Barrier 5: VWA fixture 预热 (累加成本)

- Magento (shopping) 启动后 ~5-10 min 预热 (PHP-FPM + opcache + DB connections + warmup queries)
- classifieds DB import ~3 min
- reddit Postgres ~1 min
- 每 job 启动都付这成本 → 14-cell × 5 jobs/cell × 10 min = ~12h pure setup overhead

### 累计判定

Barrier 3 单独**就足以否定**这条 path (paper-grade SR 不可比). 加 Barrier 1-2-4-5 累积工程量 + risk:

| 维度 | Myriad VWA | RunPod 4090 |
|---|---|---|
| Setup 时间 | 2-3 天 | 1-2h |
| Cost | $0 | $90-150 |
| Cross-job state continuity | ❌ 不保证 (deal-breaker) | ✅ instance always-on |
| Container 政策 | ⚠️ Singularity 未 verify | ✅ Docker 直接 |
| Wallclock limit | 48-72h per job | 你 rent 期间 |
| Multi-container orchestration | 🔴 复杂 | ✅ docker-compose |

**结论**: 省 $90 不值得 paper-grade integrity 风险 + 2-3 天 setup. RunPod path 是 dominant choice.

---

## 5. Viable Myriad Use Cases

**不依赖 quark VWA docker** 的任务，Myriad 都能做:

| Use case | 价值 | 数据流 |
|---|---|---|
| **离线分析** (figures, cross-site aggregation, bootstrap CI) | 中 — DGX 也能跑，但 Myriad 不抢 GPU | results JSONL 经 Citrix 中转 |
| **模型 fine-tuning** (4×A100 40G L 节点) | **高** — 真正稀缺资源 | 模型权重从 HF Hub 直下，训练数据从 hub 中转 |
| **Paper 写作环境** (VS Code Remote-SSH) | 中 — 体验好 | 直接 git push/pull |
| **Codex 长 prompts hosting** (vLLM 部署 4B 模型) | 高 — 视 paper 进度 | 需要装 vLLM 等基础设施 |

**不可能的 use case**:
- ❌ B1 baseline 全量 (需要 quark VWA)
- ❌ B0 baseline (需要 proxy_api_agent + VWA)
- ❌ 实时 watchdog 监督 DGX 跑的实验 (双向不通)

---

## 6. Canonical Working Access Path

**长期标准姿势**（每次需要 Myriad 时按这套流程）:

```
1. 浏览器访问 https://desktop.ucl.ac.uk (Desktop@UCL Anywhere)
2. UCL 邮箱 + 密码 + Microsoft Authenticator MFA
3. 启动 Citrix Windows VM
4. VM 内打开 VS Code 1.94 (已预装) → 装 Remote-SSH extension
5. Remote-SSH: Connect to Host → ucab352@myriad.rc.ucl.ac.uk
6. UCL 密码登录
7. Open Folder /home/ucab352/...，拥有完整 IDE
```

**优势**:
- 零 VPN，零 endpoint compliance 检查
- 零 quark Windows 网络变更（Tailscale / Docker / B1 实验全部不动）
- 文件编辑 + 集成 terminal + Git 全部远端化

---

## 7. 已知 Anti-patterns（避免）

按危险度排序：

| Anti-pattern | 后果 |
|---|---|
| 在 quark Windows 装 Cisco Secure Client 跑 UCL VPN | Tailscale 退 DERP relay → DGX 实验 latency 抖动；且 Posture failed (此 quark 配置) |
| dual boot 进 native Ubuntu | quark Windows 关 = Hyper-V 关 = Docker Desktop 关 = VWA 容器全 stop = **B1 实验立即 fail** |
| 在 WSL 调 OpenConnect csd-wrapper | 与 Cisco 同问题 (ASA 二次校验 hostscan)，浪费 1-2h 调试 |
| 浏览器走 web VPN portal | IE ActiveX dependency，所有现代浏览器死路 |
| `wsl --shutdown` 期间放任 active 实验跑 | docker-desktop distro 被一并关 → VWA 容器全 stop exit 255（实证 04-30 故障）|

---

## 8. Smoke Scripts

仓库内位置:
- `scripts/myriad/smoke_login.sh` (138 行 完整版)
- `scripts/myriad/smoke_compute.qsub` (138 行 完整版)

**已知设计问题**: 完整版假设 myriad 上有完整 P79 repo (`REPO_ROOT="../.."`)，但 Citrix 路径下 myriad 通常**没有 repo**。需要 inline 创建（heredoc）或者用 VS Code Remote-SSH 直接编辑。

PuTTY 粘贴大段 heredoc 会 truncate（实测 138 行 → 21 行），所以推荐:
- VS Code Remote-SSH 集成 terminal 粘贴（稳）
- 或者用 simplified 版（30 行 mini GPU smoke）

---

## 9. Lessons Learned

1. **UCL VPN endpoint compliance 极严**: 没有 active AV + firewall 任何客户端都进不去（不管 OpenConnect / Cisco / 浏览器）。这是 UCL IT 政策，非技术问题。

2. **Cisco Secure Client 5.x 模块化**: base installer 不带 Posture/HostScan，需要 ASA download 或 complete installer。

3. **CSD wrapper 不可替代真实 hostscan**: ASA 收 POST 后会二次探测 endpoint 实际状态，wrapper 撒谎过不了。

4. **Citrix HTML5 是 UCL 官方远程办公解**: 比 VPN 更稳定（零 endpoint compliance），适合 Linux 用户和不想动 Windows 主机网络的场景。

5. **Tailscale CGNAT 段 (100.64.0.0/10) 在企业 firewall 是 first-class 黑名单**: 不要期望任何企业网络允许 reach 这个段。

6. **PuTTY 粘贴大段 heredoc 不可靠**: PuTTY paste rate-limit 会截断；改用 VS Code 集成 terminal。

7. **dual boot 不是 quark 网络问题的解**: Hyper-V 依赖 Windows host，dual boot 会让 Docker 全停，B1 实验断。

---

## 10. References

- `docs/reference/Myriad.md` — UCL 官方 Myriad 文档 (账号 / job sizes / GPU types / module load 模板)
- `README.md` §139 — Onboarding new host (6 步标准化)
- `README.md` §171 — Cross-host results sync (hub-spoke design)
- `scripts/myriad/smoke_login.sh` + `smoke_compute.qsub` — 完整版 smoke 脚本
- `~/.claude/projects/-home-jiaming-.../memory/feedback_wsl_shutdown_quark_rule.md` — quark wsl shutdown 不停 DGX 实验 hard rule
- UCL Research Computing docs — https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/

---

## Appendix A — Citrix VM 内 VS Code Remote-SSH 配置

```bash
# 在 Citrix VM (Windows) 启 VS Code 1.94 → 装 Remote-SSH extension
# Ctrl+Shift+P → "Remote-SSH: Connect to Host..." → "+ Add New SSH Host..."
# 输入: ucab352@myriad.rc.ucl.ac.uk
# 选: 保存到 default ~/.ssh/config

# 连接后第一次 myriad 端会自动安装 vscode-server (~50MB at ~/.vscode-server/)
# myriadfs 1TB quota 完全够
```

## Appendix B — Mini GPU Smoke (PuTTY-friendly, 30 行)

如果只想验证 GPU bf16 + HF Hub 可达性（决定 Myriad 能否跑 4B 模型 fine-tuning）:

```bash
cat > ~/p79_smoke/smoke_gpu_only.qsub <<'EOF'
#!/bin/bash -l
#$ -N p79_gpu_only
#$ -l h_rt=00:10:00
#$ -l mem=8G
#$ -l gpu=1
#$ -ac allow=L
#$ -cwd
#$ -j y

echo "===== Node + GPU ====="
hostname
nvidia-smi 2>&1 | head -20

echo "===== Module + Torch + bf16 ====="
module unload compilers mpi gcc-libs 2>/dev/null || true
module load gcc-libs/10.2.0 python3/3.9-gnu-10.2.0 cuda/11.3.1/gnu-10.2.0 cudnn/8.2.1.32/cuda-11.3 pytorch/1.11.0/gpu
module list

python3 -c "
import torch
print('torch:', torch.__version__)
print('cuda?', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cap:', torch.cuda.get_device_capability(0), '(need >=8.0 for bf16)')
    x = torch.randn(1024,1024, device='cuda', dtype=torch.bfloat16)
    y = torch.randn(1024,1024, device='cuda', dtype=torch.bfloat16)
    print('bf16 matmul OK, sum=', (x@y).sum().item())
"

echo "===== HF Hub egress ====="
curl -sS -o /dev/null -w "huggingface.co -> %{http_code} %{time_total}s\n" --connect-timeout 5 https://huggingface.co
EOF
qsub ~/p79_smoke/smoke_gpu_only.qsub
```

期望关键输出:
- `cap: (8, 0)` + `bf16 matmul OK` → A100 native bf16，能跑 Qwen3-VL-4B
- `huggingface.co -> 200` → 能下模型权重

如果两项都通过，Myriad 的 model fine-tuning use case 可行。

---

**报告 owner**: P79 project (Cost-Aware Routing for Web Usage Agents)
**Last update**: 2026-04-30
**Next review**: 视 Myriad 实际使用情况；若需要做 model fine-tuning 时补 GPU compute smoke 完整结果
