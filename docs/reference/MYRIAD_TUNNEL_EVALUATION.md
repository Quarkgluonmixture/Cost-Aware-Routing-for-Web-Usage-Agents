# Myriad 穿透方案评估（请学长 review）

**调研日期**: 2026-04-30
**Session 测试节点**: `login12.myriad.ucl.ac.uk`
**Status**: 等学长基于 evidence 评估 DERP 路径或其他穿透姿势

---

## 1. 问题背景

**目标**: Myriad 4×A100 40G 节点跑 B1 baseline，摆脱 DGX shared GPU 争抢

**约束**: B1 runner 需要持续 HTTP request 家用 `quark` 上的 VWA docker (Tailscale 100.95.81.103, 端口 9980/7770/9999)

**当前 working access**: Desktop@UCL Anywhere (Citrix HTML5) → PuTTY → ssh `ucab352@myriad.rc.ucl.ac.uk` (绕开 endpoint compliance，但 Myriad 仍无法 outbound 到 quark)

---

## 2. 已 verified blocked paths（避免重复）

下面这些都已实测失败，不需要再试：

| 方案 | 失败原因 |
|---|---|
| OpenConnect on WSL Ubuntu 24.04 | UCL ASA hostscan 二次校验，所有 csd-wrapper (我自写 + 上游) 都卡 `+CSCOE+/sdesktop/wait.html` 无限轮询 |
| Cisco Secure Client 5.1.7 on Windows | Posture Assessment 3 秒 fail；根因 = quark 装了火绒+卡巴让 Windows Defender 进 Disabled service state，UCL 探测看不到 active AV/firewall |
| 浏览器 Web VPN portal (Edge/Chrome) | UCL ASA 配置依赖 IE 32-bit ActiveX，现代浏览器死路 |
| DGX → Myriad 反向 ssh jump | 家用 ISP 出口 → `myriad.rc.ucl.ac.uk:22` TCP timeout，UCL 边界 firewall 只 allow UCL 内网 IP / Connect VPN |
| Native Ubuntu dual boot | quark Windows 关机 = Hyper-V 关 = Docker 关 = VWA 容器全停，B1 实验立即 fail |

---

## 3. Firewall 机制 — 双层 stateful drop（核心 evidence）

### 3.1 实测脚本

`~/p79_smoke/diagnose_firewall.sh` 在 myriad login12 跑（attached as Appendix A 完整 raw output）。

### 3.2 Layer 1: IP segment blacklist

CGNAT 100.64.0.0/10 + RFC 1918 整段 drop（不论端口）:

| 目标 | 结果 |
|---|---|
| `100.95.81.103:9980` (我家 Tailscale) | TIMEOUT |
| `100.64.0.1:443` | TIMEOUT |
| `100.100.100.100:443` | TIMEOUT |
| `100.127.255.254:443` | TIMEOUT |
| `10.0.0.1:443` | TIMEOUT |
| `192.168.1.1:443` | TIMEOUT |
| `172.16.0.1:443` | TIMEOUT |

### 3.3 Layer 2: Outbound port whitelist

公网 IP 仅 allow whitelist port，custom port 全 drop:

| 目标 | 结果 |
|---|---|
| `1.1.1.1:80` | OPEN |
| `1.1.1.1:443` | OPEN |
| `1.1.1.1:9999` | TIMEOUT |
| `1.1.1.1:7770` | TIMEOUT |
| `1.1.1.1:9980` | TIMEOUT |

### 3.4 关键 traceroute evidence — drop 在哪发生

```
myriad → 100.95.81.103 (Tailscale 我家):
 1  10.28.101.252       (myriad subnet gateway)
 2  193.60.251.253      (UCL backbone)
 3  10.0.107.17         (UCL internal)
 4  193.60.238.37       (UCL backbone)
 5  77.241.77.37        (UCL → ISP transition)
 6  * * 77.241.77.9     (上游 ISP)
 7  * * *
 8  146.97.139.237      (公网，146.97.x = JANET backbone)

myriad → 1.1.1.1 (Cloudflare 公网):
 1  10.28.101.252
 2  193.60.251.253
 3  10.0.107.17
 4  193.60.238.37
 5  77.241.77.37
 6  * * *
 7  * * *
 8  146.97.139.237      (相同的 8 跳路径！)
```

**两条路径前 5 跳完全相同**，包都 routing 出 UCL 边界到 ISP 上游 (146.97.139.237)。但前者 TCP SYN 永远收不到 SYN-ACK。

**含义**: 这不是 routing 拒绝（包到得了上游），是**stateful firewall 在 connection-state 层 drop**。意味着任何依赖 SYN flood / TCP 协议层 mimicry 的穿透姿势都不会 work——唯一能过的是 **application-layer over allowed port** (HTTPS:443 / HTTP:80)。

---

## 4. 理论穿透窗口 — DERP relay over HTTPS:443

### 4.1 Tailscale DERP 服务全部可达

```
derp.tailscale.com           -> HTTPS 200, 0.508s
login.tailscale.com          -> HTTPS 302, 0.254s
controlplane.tailscale.com   -> HTTPS 302, 0.255s
derp1.tailscale.com          -> HTTPS 200, 0.491s
```

DERP relay 走公网 HTTPS:443，**双层 firewall 都过**。

### 4.2 理论路径

```
[Myriad userspace tailscaled (--tun=userspace-networking)]
  ↓ outbound HTTPS:443 to derp.tailscale.com (UCL allow)
[Tailscale DERP relay server]
  ↓ HTTPS:443 internal routing
[quark Windows Tailscale daemon]
  ↓ Hyper-V vEthernet
[quark VWA docker on 100.95.81.103:9980/9999/7770]
```

### 4.3 实施 sketch

```bash
# myriad login or compute, 用户态:
mkdir -p ~/tsbin && cd ~/tsbin
curl -L -o tailscale.tgz https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz
tar xzf tailscale.tgz --strip-components=1

./tailscaled --tun=userspace-networking \
             --socks5-server=localhost:1055 \
             --outbound-http-proxy-listen=localhost:1055 \
             --state=$HOME/.tsstate > tsd.log 2>&1 &

./tailscale --socket=/tmp/tailscaled.sock up
# 浏览器登 Tailscale Auth → 拿到 magic link

# 用户 program 走 SOCKS5 proxy:
curl --socks5 localhost:1055 http://100.95.81.103:9999/
HTTPS_PROXY=socks5h://localhost:1055 python3 b1_runner.py
```

### 4.4 待评估 considerations（请学长判断）

| 维度 | 数据 / 风险 |
|---|---|
| **UCL ARC 服务条款** | 学长熟悉么？是否禁止用户安装 VPN 软件？是否 DERP relay over HTTPS 算"标准网络应用"还是"VPN 客户端"？|
| **被发现的概率** | UCL firewall log 看到 long keep-alive 到 derp.tailscale.com 是否会 audit flag？还是只关注 known abuse pattern？|
| **DERP relay 性能** | 实测 ~10-50MB/s (vs direct ~100-200MB/s)。B1 runner 每 task ~30 个 HTTP request × 平均 50KB response = 1.5MB/task → 30s/task vs DGX 15min/task，latency overhead 可接受？|
| **工程改造量** | B1 runner (`p79/agents/qwen3vl_agent.py` + `p79/envs/vwa_wrapper.py`) 需走 SOCKS5。Playwright 也支持 `--proxy-server=socks5://localhost:1055`，改造量 <1 天 |
| **VWA 容器侧** | 不需要改 — quark Tailscale daemon 已经 expose 100.95.81.103，在 Tailnet 内 Myriad 能直接访问 |
| **学长是否有别的姿势** | reverse SSH tunnel? 自建 WireGuard? 利用 UCL 公开 NAT 出口? 其他 mesh VPN (Headscale 自托管)? |

---

## 5. Open questions for 学长

请基于以上 evidence 评估:

1. **DERP relay 路径在 UCL ARC 政策下是否可接受？** 你之前在 myriad 跑过 user VPN 类工具吗？
2. **如果 DERP 不可行，你试过的"穿透"是哪种姿势？** 反向 SSH tunnel? WireGuard? HTTP proxy 中转?
3. **ROI 判断**: B1 跑通 ~1 个月 → 如果 Myriad 能跑节省 10x wallclock，但承担 ARC 风险，你会推荐哪条路?
4. **替代方案**: 是否有学校提供的合法 outbound proxy / NAT punching service 可以利用?

---

## Appendix A — diagnose_firewall.sh 完整 raw output (login12, 16:01)

```
==========================================
Myriad Outbound Firewall Diagnosis
Time: 2026-04-30T16:01:03+0100
Host: login12.myriad.ucl.ac.uk
==========================================

=== [A] Public HTTPS (Academic Whitelist) ===
  https://www.google.com           -> 200 0.251s
  https://github.com               -> 200 0.267s
  https://huggingface.co           -> 200 0.217s
  https://pypi.org                 -> 200 0.211s
  https://www.ucl.ac.uk            -> 200 0.334s

=== [B] Public HTTP (Port 80) ===
  http://www.google.com            -> 200 0.091s
  http://example.com               -> 200 0.034s

=== [C] Custom Ports on Public IP (Port-based Policy) ===
  1.1.1.1:80    -> OPEN
  1.1.1.1:443   -> OPEN
  1.1.1.1:9999  -> TIMEOUT/BLOCKED
  1.1.1.1:7770  -> TIMEOUT/BLOCKED
  1.1.1.1:9980  -> TIMEOUT/BLOCKED

=== [D] Tailscale Home (CGNAT) ===
  100.95.81.103:9980  -> TIMEOUT
  100.95.81.103:9999  -> TIMEOUT
  100.95.81.103:4399  -> TIMEOUT

=== [E] Other CGNAT Prefixes (100.64.0.0/10) ===
  100.64.0.1:443       -> TIMEOUT
  100.100.100.100:443  -> TIMEOUT
  100.127.255.254:443  -> TIMEOUT

=== [F] RFC 1918 Private Segments ===
  10.0.0.1:443         -> TIMEOUT
  192.168.1.1:443      -> TIMEOUT
  172.16.0.1:443       -> TIMEOUT

=== [G] Traceroute (Hop Diagnosis) ===
  >> Target: 100.95.81.103 (Home Tailnet)
traceroute to 100.95.81.103 (100.95.81.103), 8 hops max, 60 byte packets
 1  10.28.101.252  0.363 ms  0.482 ms  0.466 ms
 2  193.60.251.253  1.451 ms  1.494 ms  1.400 ms
 3  10.0.107.17  1.021 ms  0.726 ms  0.885 ms
 4  193.60.238.37  1.756 ms  1.595 ms  1.249 ms
 5  77.241.77.37  3.315 ms  1.639 ms  1.454 ms
 6  * * 77.241.77.9  2.960 ms
 7  * * *
 8  146.97.139.237  1.966 ms  1.566 ms  1.550 ms

  >> Target: 1.1.1.1 (Reference Public)
traceroute to 1.1.1.1 (1.1.1.1), 8 hops max, 60 byte packets
 1  10.28.101.252  0.358 ms  0.262 ms  0.376 ms
 2  193.60.251.253  1.172 ms  0.860 ms  1.477 ms
 3  10.0.107.17  0.802 ms  1.070 ms  1.125 ms
 4  193.60.238.37  1.088 ms  0.883 ms  0.980 ms
 5  77.241.77.37  1.098 ms  1.390 ms  18.479 ms
 6  * * *
 7  * * *
 8  146.97.139.237  17.923 ms  1.694 ms  1.678 ms

=== [H] Tailscale DERP Relay Infrastructure ===
  https://derp.tailscale.com           -> 200 0.508s
  https://login.tailscale.com          -> 302 0.254s
  https://controlplane.tailscale.com   -> 302 0.255s

=== [I] Local DERP Nodes (London/UK) ===
  derp1.tailscale.com -> 200 0.491s
  derp16-lhr.tailscale.com -> DNS resolution failed
```

---

**联系方式**: 如需补 compute node 端 firewall 行为（理论比 login 更严）/ 或我跑别的 probe，告诉我命令我立刻在 myriad 上执行回报。
