---
type: reference
status: live
created: 2026-05-07
last_updated: 2026-05-14
audience: self + advisor sync prep
---

# Compute Infrastructure Landscape (P79)

> **Living document** — update when compute access changes (new accounts, network policy, deprecations).
>
> **Last major update**: 2026-05-14. **Condenser A100 now operational** — VM `a100-jiaming-test` @ `10.134.51.2` (replaces earlier `a100-jiaming-mech` @ `10.52.6.89`), Ubuntu 22.04.5, SSH user `ubuntu`, PyTorch smoke test passed (`torch 2.11.0+cu128`, cuda True, capability (8,0), driver 580.126.20 / CUDA 13.0). GPU confirmed **A100-PCIE-40GB** (NOT 80GB). Access: quark → UCL VPN → condenser bastion (`cloud-user`, key `id_condenser_private_fixed` + cert `id_condenser_new.signed`) → VM (`ssh condense-a100`). Earlier (2026-05-07): Myriad passwordless SSH working (VSCode Remote-SSH ❌ glibc 2.17), Myriad HPC account activated.

---

## §0 TL;DR

| Tier | Platform | Status | Use case |
|---|---|---|---|
| **0** | UCL Condenser **A100 40GB dedicated** | ✅ **operational 2026-05-14** (VM `a100-jiaming-test` @ `10.134.51.2`, PyTorch verified; VWA docker self-hosted LIVE since 2026-05-14, fired Fire-3 2026-05-18) | **Paper-grade primary**: Stage 2B scale-up + Llama-4 cross-arch (small variants only, 40GB constraint) + 42-cond / 6-cell Phase 1a (VWA self-hosted on VM) |
| **1** | UCL **Myriad HPC** (V/U-type 4× A100 80GB / L-type 4× A100 40GB / E/F-type 2× V100) | ✅ account activated 5/6 / **passwordless SSH 5/7 evening** / ⚠️ VSCode Remote-SSH NOT viable (RHEL 7 glibc 2.17 < required 2.28) | Terminal-only batch + cross-arch (qsub on V/U-type) + future SAE training (4-GPU data-parallel). Workflow: dev on quark/A100 → git push → ssh myriad git pull → qsub. |
| **2** | **DGX Spark** (`spark-9ea3`) shared lab | ✅ stable, no admin/sudo, lab Tailscale `981526092.github` | Archived data source / VWA Docker Tailscale bridge / curation done (笔记 §113) |
| **3** | Advisor 5090 (post AI Center 搬运) | ⏳ pending | Backup if Condense fails, advisor offered 5/5 sync |
| ❌ | RunPod 4090 self-fund $200 | deprecated by Tier 0 | not needed |
| ❌ | Myriad pre-5/6 | superseded by 5/6 reactivation | no longer "abandoned" — see §1.2 |

---

## §1 Platforms Detail

### §1.1 UCL Condenser A100 40GB dedicated ⭐ Tier 0

**Status (2026-05-14)**: ✅ **operational** — PyTorch smoke test passed (`torch 2.11.0+cu128`, `cuda available: True`, device `NVIDIA A100-PCIE-40GB`, capability `(8, 0)`, CUDA matmul OK).
**Provider**: UCL ARC Condenser (Harvester KubeVirt over Rancher).
**Namespace**: `arc-proj-webarena-ns`.
**VM**: `a100-jiaming-test` — IP **`10.134.51.2`**, Ubuntu 22.04.5 LTS, SSH user `ubuntu`, 500 GB disk persistent. (Supersedes earlier `a100-jiaming-mech` @ `10.52.6.89`.)
**GPU**: 1× NVIDIA **A100-PCIE-40GB** (40960 MiB) — driver `580.126.20`, CUDA 13.0, MIG disabled, persistence mode enabled. **40GB, NOT 80GB** — correct any old "80GB" notes.
**Cost**: $0 (UCL allocation, NOT student-funded; supersedes RunPod budget plan).
**Wallclock**: unlimited (dedicated, single-tenant, no queue).
**Persistent state**: 500 GB disk persistent.

**Access path** (current, 2026-05-14): quark (Windows) → UCL VPN → condenser bastion `ssh.condenser.arc.ucl.ac.uk` → VM. From quark: `ssh condense-a100` (alias `condense-test` also points at `10.134.51.2`). quark `~/.ssh/config`:
```
Host condenser
    HostName ssh.condenser.arc.ucl.ac.uk
    User cloud-user
    IdentityFile ~/.ssh/id_condenser_private_fixed
    CertificateFile ~/.ssh/id_condenser_new.signed
    StrictHostKeyChecking accept-new
Host condense-a100        # (and alias condense-test)
    HostName 10.134.51.2
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519_dgx
    ProxyJump condenser
    StrictHostKeyChecking accept-new
    ConnectTimeout 20
```
From DGX: `ssh -i ~/.ssh/vwa_windows Quark@100.95.81.103 'ssh condense-a100 "<cmd>"'`.

**Status**: P79 repo + venv set up on VM; **VWA docker stack self-hosted ON the VM** (cls/red/shop @ A100 localhost, see §1.1.2) — VWA self-host is LIVE since 2026-05-14 and has fired paper-grade (Fire-3 2026-05-18). Paper-grade Phase 1a/1b/Pass-2 runs are migrated here from DGX/Myriad.

**40GB capacity analysis** (paper-strategic):
| Model | Params | bf16 size | fp16+activations buffer | Fits 40GB? |
|---|---|---|---|---|
| Qwen3-VL-4B (B1) | 4B | ~8 GB | ~12-15 GB | ✅ comfortable, 25 GB headroom |
| Qwen2-VL-7B | 7B | ~14 GB | ~20 GB | ✅ comfortable |
| Llama-3.2-11B-Vision | 11B | ~22 GB | ~30 GB | ✅ tight but OK |
| Llama-4 Scout (~17B) | 17B | ~34 GB | ~38 GB | ⚠️ borderline OOM, need 4-bit quant or shorter sequences |
| Llama-4 Maverick (~70B) | 70B | ~140 GB | n/a | ❌ **does not fit** — use Llama-3.2-90B 4-bit on Myriad instead |
| SAE training (paper v2) | varies | typically need 2-4× model size | | ❌ single-card 40GB infeasible — defer to Myriad 4×80GB data-parallel |

**Reachability from A100 outbound**:
- ✅ Internet (HuggingFace download / pip / GitHub clone)
- ✅ UCL services on UCL backbone (10.32.x.x bastion etc.)
- ❌ **quark VWA Docker (100.95.81.103:9980/9999/7770)** — NOT reachable directly; would need Tailscale install + lab tailnet membership.
- ❌ DGX (`100.99.92.18`) Tailscale — same as above.

**Note**: For the 42-cond / 6-cell Phase 1a, see §1.1.2 "VWA self-host on A100" — preferred path bypasses the reach issue entirely by co-locating VWA stack with agent on the A100 VM itself.

#### §1.1.2 VWA self-host on A100 (LIVE — Phase 1a paper-grade host)

Instead of reaching out to quark VWA, **deploy VWA Docker stack on A100 VM itself**. Agent + VWA both localhost-only on A100. Decided 2026-05-07; LIVE since 2026-05-14, fired paper-grade Fire-3 2026-05-18.

**Why preferred over Tailscale-to-quark path**:
| Dimension | A100 self-host | A100 → Tailscale → quark |
|---|---|---|
| Network latency | localhost ~0.1ms | overlay 50-100ms+ |
| Lab admin gating | ❌ none | ✅ needs admin approval |
| Quark sleep risk | not applicable | ⚠️ quark sleep kills run |
| Reproducibility | ⭐ single VM | "agent A100 + VWA quark" complex |
| Tear-down portability | `docker compose down` clean | needs Tailscale teardown coordination |
| Disk cost | +~110GB (cls+red+shop, no wiki) | 0 |
| Setup time | ~1-2h one-time | ~30min Tailscale + admin wait |

**Disk budget on 500GB A100 disk**:
- Ubuntu OS + apt cache: ~10GB
- Docker engine + image overhead: ~5GB
- VWA cls (osclass + MySQL DB): ~10-20GB
- VWA red (Postmill + DB): ~5-10GB
- VWA shop (Magento + DB): ~30-40GB
- VWA homepage (Python): ~1GB
- VWA wikipedia (skip — paper §5 mechanistic + Phase 1a don't use Wikipedia tasks): 0
- Qwen3-VL-4B model cache: ~10GB
- Llama-3.2-11B-Vision cache (cross-arch): ~22GB
- Mechanistic + Phase 1a results: ~30GB
- pip site-packages: ~10GB
- **Total**: ~135-160GB ⇒ 500GB - 160GB = ~340GB headroom ⭐ comfortable

**Actual disk layout (post-Phase 2 migration 2026-05-26, 笔记 §301)** — original 500GB headroom budget proved optimistic; real usage 2026-05-26 pre-migration was `/dev/vda1` 457G/485G **95%** because:
- `/var/lib/containerd` = 388G (docker image storage委托给 containerd, 真物理大头, 不是 `/var/lib/docker`)
- `/var/lib/docker/rootfs` = 182G (overlay layer + ~13.76G container write-layer during fire)
- `docker system df` 报 416GB "Images" 是 logical size (不去重共享 layer), 真实需看 `du /var/lib/{docker,containerd}` ≈ 570G total docker stack — paper-grade fire 不可压缩 baseline
- VM 有 2 个独立物理盘 (虚拟): `/dev/vda1` 485G (`/`, docker stack 主占) + `/dev/sda` 503G (`/mnt/scratch`, 含 119G `wikipedia.zim`)

**Partial symlink layout** (Phase 2 disk migration result, fire data → scratch):
```
/home/ubuntu/workspace/p79/results/
├── B0_3mode / B1_3mode / B2_3mode / ...       (small, git-tracked or analysis aggregate)
├── provenance/        (256K, git-tracked: env_a100_baseline.json + vwa_a100_*.json — Gate 3 fingerprint)
├── mechanistic/       (52M, git-tracked: §5 mechanism archive, paper §5 暂搁但 data 保留)
├── repro_replicates/  (1.5G, gitignored: dom R31194 + vision R24792 clean replicates, 笔记 §297)
├── diagnostic_replay/ (3.4M)
├── phantom_paper/     (2.7M)
├── phase1_paper_grade/(12K)
└── visualwebarena → /mnt/scratch/p79_results_active_visualwebarena   ⭐ symlink (fire data)

/mnt/scratch/
├── p79_results_active_visualwebarena/    ← active fire run dirs (B0/B1/B2 × 6 mode × cls/red)
├── wikipedia_en_all_maxi_2025-08.zim     119G (VWA wikipedia data, single static file)
└── lost+found/
```

**Why partial symlink (not full `results/` symlink)**: git 不 follow symlink for tree traversal — 整 `results/` symlink 会让 `git status` 把整 subtree 看成 deleted (因为 git 看 `results` 是 lrwxrwxrwx 类型, 当成 single file 不进 scratch tree). Paper-grade preflight Gate 3 fail-closed on `git diff results/provenance/*.json` showing DELETED → fire abort. **教训**: 只 symlink 不进 git tracking 的子树 (i.e., `results/visualwebarena/<run_id>/...` 是 fire 写, gitignored); git-tracked subdir (provenance / mechanistic / repro_replicates) 必须 留 / 上 git-friendly. Phase 2 第一次 launch 踩这个坑, restructure 后第三次 launch 14:18 飞起。详 [[实验笔记]] §301。

**Disk monitoring**: `ntfy` alert at `/` ≥95% (~每 1-2h scan); paper-grade fire 期间 container write-layer 涨 ~2-3G/cond (B-1839 per-cond docker restart 设计应 reset 但 ~13.76G accumulated 实测可能 partial). scratch 24% (365G avail) 跑全 36 cond ~28-30G data 完全够。

**Setup script**: `scripts/setup/a100_self_host_vwa.sh` (added 2026-05-07).

**Paper §3 method disclosure required**:
> "Phase 1 paper-grade runs were executed against VWA Docker stack hosted on Windows machine via Docker Desktop WSL2 backend, accessed by agent through Tailscale. For the post-Phase-A 42-cond / 6-cell Phase 1a paper-grade fire, the same Docker stack is deployed on the A100-equipped UCL Condense VM (Ubuntu 22.04 native Docker), with agent and stack co-located on same host. Byte-equivalence of HTML responses verified across deployments via per-site checksums."

**Status (2026-05-14)**: ✅ VWA self-host LIVE on the A100 VM (cls/red/shop @ localhost), fired paper-grade Fire-3 2026-05-18.

**Access path** (post-Steve 5/7 clarification):
- ❌ Direct SSH not allowed
- ✅ Via condenser bastion `ssh.condenser.arc.ucl.ac.uk` (10.32.0.32, UCL-internal-only)
- ✅ Bastion needs **CA-signed SSH cert** generated at https://condenser.arc.ucl.ac.uk/portal/ (7-day validity, regenerate as needed)
- ✅ Bastion does ProxyJump only — actual SSH session lands on `ubuntu@10.52.6.89` with `~/.ssh/id_ed25519_condense` (the KeyPair private key that pairs `jiaming-dgx-spark`)

**SSH config (target machine: quark Windows or Desk@Anywhere)**:

```
Host condenser
    HostName ssh.condenser.arc.ucl.ac.uk
    User cloud-user
    CertificateFile ~/.ssh/id_arc.signed
    IdentityFile ~/.ssh/id_arc
    StrictHostKeyChecking accept-new

Host condense-a100
    HostName 10.52.6.89
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519_dgx        # Quark side: Windows copy of DGX key (same KeyPair)
    ProxyJump condenser
    StrictHostKeyChecking accept-new
```

**Status (5/7 morning)**:
- ✅ VM running
- ✅ A100 attached (Steve confirmed via email)
- ⏳ SSH cert pending (user generates on quark via SSH Portal)
- ⏳ Cloud-init verification (pciutils + nvidia-utils-535 installed?)
- ⏳ NVIDIA driver install pending (Ubuntu cloud image doesn't auto-install driver, run `sudo ubuntu-drivers autoinstall`)

### §1.2 UCL Myriad HPC ⭐ Tier 1

**Provider**: UCL ARC Research Computing (traditional HPC cluster, SGE batch scheduler).
**Login node**: `myriad.rc.ucl.ac.uk` (DNS round-robin login12 / login13).
**OS**: RHEL 7 / CentOS 7 family (glibc 2.17, GLIBCXX 3.4.24). ⚠️ **Affects VSCode Remote-SSH** — see "Tooling caveat" below.
**Account**: `ucab352` activated 2026-05-06 (per "We are happy to confirm" email, retroactively re-applied — previous account may have lapsed).
**Auth**: ✅ **Passwordless SSH works as of 2026-05-07 evening** (`id_rsa_myriad` RSA key from quark, `Authenticated to myriad.rc.ucl.ac.uk using "publickey"`). Just type `ssh myriad` — no password, no Cisco VPN needed beyond initial UCL routing.

**⚠️ Tooling caveat — VSCode Remote-SSH NOT viable** (discovered 2026-05-07):
- VSCode Server requires **glibc ≥ 2.28** + **GLIBCXX ≥ 3.4.25**, but Myriad login node only has glibc 2.17 (`osReleaseId == rhel`).
- Server installs but cannot start. Symptom: `Missing GLIBC >= 2.28` / `Missing GLIBCXX >= 3.4.25` in VSCode Remote-SSH log.
- **Implication**: Myriad workflow is **terminal SSH only** (`ssh` / `qsub` / `scp` / `rsync`); no in-IDE file editing on Myriad.
- **Workarounds (priority order)**:
  1. ✅ **Recommended**: edit code on quark (VSCode local) or A100 (VSCode Remote-SSH) → `git push` → `ssh myriad && cd ~/Scratch/p79 && git pull` → `qsub job.sh`. Pure terminal flow.
  2. **code-server** in `$HOME` user space (no glibc dependency, browser-based VSCode). Run on Myriad login node, port-forward via SSH `-L 8080:localhost:8080`. Setup ~30 min.
  3. JetBrains Gateway (Toolbox-based, sometimes ships own libstdc++) — untested on Myriad.
  4. NOT viable: pinning old VSCode Server version that targets glibc 2.17 — Microsoft removed official support.

**Node types relevant to us**:
| Type | GPU | VRAM | Count | Notes |
|---|---|---|---|---|
| **V** | 4× A100 80GB | 80 GB | 2 nodes | ⭐ matches Condense GPU spec |
| U | 4× A100 80GB | 80 GB | 1 node | similar V |
| L | 4× A100 40GB | 40 GB | 6 nodes | smaller VRAM but Qwen3-VL-4B fits |
| E/F | 2× V100 | 32 GB | 19 nodes | older but plentiful |

**Quotas**: 1 TB home (= scratch), `gquota` to check. Backed-up `~/ACFS` 1 TB read-only from compute nodes.
**Wallclock**: 72h single-core, 48h parallel (chunkable). Phase 1a (~24h per condition) fits comfortably.
**Pre-built modules**: PyTorch 1.11 GPU (load module list per `~/.rc.ucl.ac.uk/docs`), CUDA 11.x, Python 3.7-3.11 stack.

**Job submission paradigm**: SGE qsub batch scripts:
```bash
#$ -l h_rt=24:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -ac allow=V    # request V-type (A100 80GB)
#$ -wd /home/ucab352/scratch/jobs
```

**Old "abandoned" status retracted**: Pre-2026-05-06 we deprecated Myriad due to (a) Tailscale CGNAT block (Myriad → home VWA Docker blocked), (b) wallclock fits single-block Phase 1a. Updated assessment 2026-05-07:
- (a) Still applies — Myriad cannot reach a live VWA env → **Phase 1a paper-grade fire NOT feasible on Myriad** (needs live VWA env)
- (b) Wallclock fine for chunked Phase 1a — but Tier 0 A100 dedicated has no wallclock so still primary
- ✅ **Mechanistic / cross-arch / SAE on archived data is fine on Myriad** (no VWA needed, just LLM forward pass)

**Use cases (5/7 update)**:
1. ⭐ Mechanistic Stage 2B parallel (if Condense busy + Myriad V-type queue available)
2. ⭐ Llama-4 cross-arch (4-GPU data-parallel on V-type)
3. ⭐ SAE training (paper v2, defer) — V-type 4× A100 80GB perfect for SAE training scale
4. ⏳ Analysis batch jobs (CPU-only on D-type) without burning A100 quota
5. ❌ Phase 1a paper-grade fire (CGNAT VWA block)
6. ❌ Interactive dev (qsub batch only, no real-time iteration)

### §1.3 DGX Spark (`spark-9ea3`) — Tier 2

**Provider**: lab shared (seonglae / 981526092 lab).
**Hardware**: aarch64 + NVIDIA GB10 (sm_121), single GPU ~128 GB unified VRAM.
**Tailnet**: `981526092.github` (lab tailnet, **user is NOT admin**).
**Tailscale IP**: `100.99.92.18` (in lab tailnet).
**Local LAN**: `10.1.17.116/24` (`enP7s7`, lab subnet — separate from UCL Condense 10.52.x.x).
**No sudo**: cannot install system packages, system VPN client (e.g. openconnect), or modify network routes.
**No GPU contention guarantee**: shared with seonglae sweep jobs, latency varies 1-5×.

**Roles (post-2026-05-06)**:
- **Archived B1 phantom_som data source** (`results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428/` 1.8 GB)
- **VWA Docker Tailscale bridge** (DGX → quark `100.95.81.103:9980/9999/7770` — **NOT used by Condense or Myriad**, only by historic Phase 1 + future fresh runs on lab GPU)
- **Curation host** (笔记 §113 — `curate_mirage_tasks.py` ran 5/6 evening, 24+11 candidates dataset)
- **Lab GPU jobs** (any work that doesn't need to reach UCL infrastructure)

**Cannot do**:
- Reach UCL Condense A100 (lab tailnet blocks 10.52.x.x cross-VLAN)
- Reach Myriad / bastion (no UCL VPN, no admin to install)
- Generate SSH Portal cert (cert needs UCL VPN, DGX has none)

### §1.4 Quark (Windows home) — gateway / dev workstation

**Hardware**: User's home Windows machine.
**OS**: Windows 11 (or 10), Tailscale + Cisco AnyConnect both installed.
**Tailnet**: `981526092.github` (same lab tailnet as DGX, user owned; Tailscale IP `100.95.81.103`).
**UCL VPN**: Cisco AnyConnect to `vpn.ucl.ac.uk` (active 2026-05-07).
**Admin**: yes (own machine).

**Why quark is the central gateway**:
- ✅ Tailscale → DGX (lab tailnet member)
- ✅ Cisco AnyConnect → UCL VPN (Myriad + bastion + Condense reachable)
- ✅ Local Docker Desktop WSL2 backend running VWA stack (see §1.4.1 below)
- ✅ User has admin (can install OpenSSH Server / generate keys / configure)
- ✅ VSCode for Windows + Remote-SSH extension → A100 dev workstation
- ✅ `(ai_learning)` conda env present (cert/scp scripts use existing Python)

**Roles**:
- **Primary SSH client** for A100 Condense + Myriad
- ~~**VWA Docker host** for paper-grade runs~~ (superseded 2026-05-14: VWA now self-hosted on the A100 VM; quark Docker no longer in paper-grade critical path)
- **Cert generation host** (UCL VPN + browser → SSH Portal)
- **Data transfer pivot** (rsync DGX → quark via Tailscale, scp quark → A100 via UCL VPN+bastion)

#### §1.4.1 VWA Docker stack (verified state 2026-05-07)

**Docker engine**: Docker Desktop with **WSL2 backend** (modern default, NVIDIA GPU passthrough capable).
- Server: `linux / amd64 / 6.6.87.2-microsoft-standard-WSL2` (kernel string indicates WSL2 distro)
- Engine endpoint: `npipe:////./pipe/dockerDesktopLinuxEngine` (Windows ↔ WSL2 named pipe)
- Default context: `desktop-linux *` (auto-selected by CLI)
- Daemon process: lives in WSL2 special distro `docker-desktop` (NOT in user's Ubuntu WSL distro)
- Disk: container images stored in WSL2 VM disk (typically `C:\Users\<user>\AppData\Local\Docker\wsl\disk\` virtual disk image)

**WSL distros** (`wsl --list -v`):
| Distro | Version | State | Role |
|---|---|---|---|
| Ubuntu | 2 | Running | User's Ubuntu workspace (independent of Docker Desktop) |
| docker-desktop | 2 | Running | Docker engine VM (auto-managed by Docker Desktop) |

**VWA stack** (6 containers, all up 2026-05-07 after Docker Desktop wake from sleep):

| Container Name | Image | Port (host:container) | Site / Role |
|---|---|---|---|
| `classifieds` | osclass-* | 9980:9980 | OSClass classifieds frontend (paper-grade VWA cls) |
| `classifieds-com` | osclass-postgres / mysql | (internal) | classifieds DB backend |
| `vwa-reddit` | postmill-pop | 9999:80 | Postmill (Reddit-like, paper-grade VWA red) |
| `vwa-shopping` | shopping_fin | 7770:80 | Magento frontend (paper-grade VWA shop) |
| `shopping_admin` | shopping_admin | 7780:80 | Magento admin panel (rarely used in agent runs) |
| `vwa-homepage` | python:3.10-* | 4399:4399 | VWA hub (start page) |
| `vwa-wikipedia` | kiwix/kiwix-serve | 8888:80 | Wikipedia knowledge base |

**Network exposure**: containers bind to `0.0.0.0:<port>` on quark Windows host, accessible via:
- Quark localhost: `http://localhost:9980` (PowerShell debug)
- Tailscale lab tailnet: `http://100.95.81.103:9980` (DGX access via Tailscale, used in paper-grade runs)
- UCL VPN tunnel: NOT exposed (firewall, irrelevant for our work)

**Bringing up / restarting** (PowerShell):
```powershell
# Check state
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Restart all VWA containers (if Docker Desktop slept and containers stopped)
docker compose -f <path-to-vwa-compose.yml> up -d
# OR individually:
docker start classifieds classifieds-com vwa-reddit vwa-shopping shopping_admin vwa-homepage vwa-wikipedia
```

**Site reset (paper-grade between conditions)**:
- Reset script: `scripts/maintenance/reset_vwa_sites.sh` (run from DGX over Tailscale to quark, see CLAUDE.md)
- Implementation detail: PowerShell on quark restarts container + restores DB snapshot via Docker volume reset
- Auth file: DGX-side `.auth/` (gitignored) holds Playwright session state per site

**Implications for compute path**:
- ✅ **DGX → quark VWA** path used for historic Phase 1 paper-grade runs (B1 dom/som/vision + B0 phantom_*)
- ✅ Same path will be used for **future fresh paper-grade runs from lab GPU** (e.g. DGX-side B1 work post-Phase-A if A100 Condense busy)
- ❌ **Myriad → quark VWA** blocked by CGNAT (Myriad outbound firewall denies Tailscale 100.x.x.x range)
- ✅ **A100 Condense paper-grade** does NOT use quark VWA — VWA Docker is self-hosted on the A100 VM itself (cls/red/shop @ A100 localhost, see §1.1.2), so the Tailscale-to-quark reach issue is moot for Phase 1a.

**Quark sleep/Docker Desktop quirk**: when Windows sleeps, Docker Desktop suspends → containers Stopped. Wake from sleep → Docker Desktop auto-restarts but containers may need manual `docker start` (some compose configs auto-restart, depends on `restart: unless-stopped` policy). VWA containers verified to auto-restart 2026-05-07 (50 seconds after Docker Desktop wake — likely from auto-restart policy).

**Ref**: `feedback_wsl_shutdown_quark_rule.md` (memory) — quark主机 sleep/restart 之前必须 stop DGX 实验, 否则 VWA 容器全 stop 致 timeout.

### §1.5 Desk@Anywhere — fallback gateway

**UCL-managed Windows virtual desktop**, browser-accessible from anywhere.
**UCL backbone**: yes (UCL internal services reachable without separate VPN, but **CANNOT reach Condense subnet 10.52.x.x** — empirically tested 5/7).
**Use case**: backup if quark unavailable.
**Disadvantage**: session-managed (may close on idle), can't host long-running SSH tunnels reliably.

### §1.6 Advisor 5090 — Tier 3 backup

**Status**: advisor offered 2026-05-05 sync. Pending advisor moving 5090 from home gaming setup to AI Center for long-running availability.
**Hardware**: 1× NVIDIA RTX 5090 (32 GB VRAM, sm_120).
**Use case**: backup if A100 Condense fails. 32 GB VRAM enough for B1 4B + Llama-4 (small variants) but not full SAE training.

---

## §2 Network Access Matrix

Y = direct path works. * = via gateway (multi-hop). N = blocked.

| From ↓ / To → | DGX | Quark | A100 (10.52.6.89) | Myriad | Bastion | Internet |
|---|---|---|---|---|---|---|
| **DGX** | self | Y (Tailscale 6-16ms) | N (no UCL routes, no admin to add) | N | N | Y |
| **Quark** | Y (Tailscale) | self | * (Cisco VPN + bastion ProxyJump, cert needed) | Y (Cisco VPN, password SSH ✅) | Y (Cisco VPN) | Y |
| **A100 VM** | N | * (would need Tailscale install on A100 + lab admin approval, defer) | self | N | (used as jump only) | partial (UCL routed) |
| **Myriad** | N | (login terminal only) | N (CGNAT block legacy) | self | (peer UCL service) | partial (UCL routed) |
| **Desk@Anywhere** | N | N | N (firewall, tested 5/7) | Y (UCL backbone) | (presumably Y) | Y |

**Key insight**: **Quark is the only machine that can reach BOTH lab Tailscale AND UCL VPN resources.** All cross-cluster work flows through quark.

---

## §3 SSH paths + setup

### §3.1 Quark → A100 Condense (PRIMARY)

**Pre-req**: SSH cert from https://condenser.arc.ucl.ac.uk/portal/ (UCL VPN active), saved to `$HOME\.ssh\id_arc` + `id_arc.signed`. Cert valid 7 days, regenerate via portal.

```powershell
# Quark PowerShell, after generating cert + lock permissions via icacls:
ssh condense-a100
# Or for VSCode: F1 → "Remote-SSH: Connect to Host" → condense-a100
```

`~/.ssh/config` block: see §1.1.

### §3.2 Quark → Myriad ✅ passwordless (2026-05-07 evening)

```powershell
ssh myriad
# No password prompt — id_rsa_myriad RSA key authenticated
```

Setup recap (already done):
- Generated `id_rsa_myriad` on quark, public key copied to `~/.ssh/authorized_keys` on Myriad
- `~/.ssh/config` on quark has `Host myriad` block with `User ucab352` + `HostName myriad.rc.ucl.ac.uk` + `IdentityFile ~/.ssh/id_rsa_myriad`
- ✅ Confirmed: `Server accepts key: ... id_rsa_myriad RSA` + `Authenticated to myriad.rc.ucl.ac.uk using "publickey"`

⚠️ **VSCode Remote-SSH NOT viable** on Myriad (glibc 2.17 vs required 2.28+ — see §1.2 Tooling caveat). Terminal SSH only; for IDE workflow develop on quark / A100 + `git push` + `ssh myriad git pull`.

### §3.3 Quark → DGX (lab Tailscale)

```bash
ssh jiaming@spark-9ea3   # via Tailscale, no UCL VPN needed
# or via IP: ssh jiaming@100.99.92.18
```

### §3.4 Myriad key auth setup ✅ DONE (2026-05-07 evening)

Already configured. For reference / future re-setup:
```bash
# On Myriad (was done from initial password login):
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "<quark id_rsa_myriad.pub content>" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Quark side: ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_myriad
# ~/.ssh/config:
#   Host myriad
#       HostName myriad.rc.ucl.ac.uk
#       User ucab352
#       IdentityFile ~/.ssh/id_rsa_myriad
```

Test: `ssh myriad hostname` should print login12 or login13 without password prompt.

### §3.5 DGX → A100 (BLOCKED, do not configure)

DGX cannot reach A100. Workaround: data transfers via DGX → quark → A100, or commit small subsets to git (see §5 archived-data flow).

---

## §4 SSH key + cert inventory

| Location | File | Purpose |
|---|---|---|
| DGX `~/.ssh/id_ed25519_condense` | private key | KeyPair `jiaming-dgx-spark` private, pairs with public on Rancher |
| DGX `~/.ssh/id_ed25519_condense.pub` | public key | Pasted into Rancher KeyPair 5/6 |
| Quark `$HOME\.ssh\id_ed25519_dgx` | private key | **Same KeyPair as DGX** (copied from DGX 5/6 evening) |
| Quark `$HOME\.ssh\id_ed25519_dgx.pub` | public key | (generated alongside) |
| Quark `$HOME\.ssh\id_arc` | private key | **TBD** — generated by SSH Portal at cert request time |
| Quark `$HOME\.ssh\id_arc.signed` | signed cert | **TBD** — bastion auth, 7-day validity |
| Rancher (UCL Condense) KeyPair `jiaming-dgx-spark` | public key + fingerprint `26:53:bb:...:fe` | namespace `arc-proj-webarena-ns`, validated |

**Rotation**:
- `id_arc` + `id_arc.signed`: regenerate every 7 days (or when cert expires)
- `id_ed25519_condense` / `id_ed25519_dgx`: long-lived (months); regenerate if compromised

---

## §5 Archived data flow

**Source**: DGX `results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428/` (1.8 GB total).
**Need on A100 / Myriad**: 24-task strong + 11-task reverse subset (~25 MB) for Stage 2B scale-up + reverse asymmetry.

**Path A (recommended for small subset)**: DGX → git commit + push → A100/Myriad git pull. ~25 MB fits in repo, .gitignore whitelist exception.

**Path B (large data, DGX archive bulk)**:
```
DGX                  Quark                   A100/Myriad
 │ rsync via         │ scp via UCL VPN        │
 │ Tailscale ~6-16ms │ + bastion ProxyJump    │
 ├──────────────────►│───────────────────────►│
```

**Path C (cloud bucket transit)**: only if quark disk space tight. Uploads + 2× transfer time.

---

## §6 Compute path priority (P79 paper-grade work)

| Workload | Primary | Backup | Reason / 40GB-aware |
|---|---|---|---|
| Mechanistic Stage 2B scale-up (24-task curated, B1 Qwen3-VL-4B) | Tier 0 A100 40GB Condense | Tier 1 Myriad V-type 80GB | B1 4B (~10GB) fits 40GB comfortably; Condense faster iteration |
| Llama-3.2-11B-Vision cross-arch | Tier 0 A100 40GB Condense | Tier 1 Myriad V-type 80GB | ~22GB bf16 fits 40GB tight, fine |
| Llama-4 Scout (~17B) cross-arch | ⚠️ Tier 0 borderline (use 4-bit quant) | Tier 1 Myriad V-type 80GB | 40GB headroom thin, prefer Myriad if fp16 needed |
| Llama-4 Maverick (~70B) cross-arch | ❌ NOT viable on Tier 0 | Tier 1 Myriad V-type 80GB (4-bit single-card) or 4× data-parallel | 40GB single-card insufficient |
| 42-cond / 6-cell Phase 1a paper-grade fire | ✅ Tier 0 **primary** (VWA self-hosted on A100 VM, LIVE since 2026-05-14, fired Fire-3 2026-05-18) | Fallback: DGX shared (lab tailnet, GPU contention slow) | VWA co-located on A100 localhost |
| SAE training (paper v2 deferred) | ❌ Tier 0 single 40GB infeasible | Tier 1 Myriad V-type 4× A100 80GB data-parallel | data-parallel scales SAE training |
| CPU analysis batch (figures, aggregation) | DGX local | Tier 1 Myriad D-type | no GPU needed |
| Smoke tests / curation (small) | DGX (if shared GPU available) | Tier 0 A100 | already done 5/6, 笔记 §113 |

---

## §7 Active workflows + status

### §7.1 Mechanistic Stage 1+2 (笔记 §111-§113)

- ✅ Stage 1 linear probe (Stage 1A/1B/1C) — trivial separability proven, infra works
- ✅ Stage 2A first-token logit shift — L17 peak (5-task aggregate)
- ✅ Stage 2B continuation (forward) — task 0 L11 mirage causal layer (case study)
- ✅ Stage 2B reverse — null at all layers (paper-grade asymmetry, §111.5b)
- ✅ Curation 234 task → 24 strong + 11 reverse candidates (笔记 §113)
- ⏳ Stage 2B curated scale-up on A100 — pending SSH setup
- ⏳ Llama-4 cross-arch — pending SSH setup
- ⏳ Position-resolved patching — pending SSH setup

### §7.2 42-cond / 6-cell Phase 1a paper-grade fire

- ✅ Pre-registration framework draft (`preregistration.md` status:draft)
- ⏳ Threshold witness from advisor (K_h1=0.75 / K_h3=0.67 / TOST δ=1.0pp)
- ⏳ Paper split decision (3 vs 4 papers)
- ✅ Compute path lock: A100 Condense, VWA self-hosted on VM
- ✅ Fired on A100 (Fire-3 2026-05-18)

### §7.3 Paper drafts

- ✅ §1-§3 done (~3163 words, paper_planning §4)
- ⏳ §5 mechanism (post-Stage 2B scale-up)
- ⏳ §6 routing (post-router experiments, paper 2 if split)
- ⏳ Other sections paced by data + advisor email

---

## §8 Maintenance protocol

**Update this doc when**:
- New compute platform allocated / deprecated (e.g. SAE GPU access)
- Network policy change (e.g. firewall rule changes affecting SSH paths)
- SSH cert lifecycle event (regeneration, key rotation)
- Quota changes (Myriad 1TB exceeded, Condense disk full)
- VWA Docker location / state changes (e.g. quark hardware migration)
- Advisor sync results affecting compute decisions

**Don't update for**:
- Per-job temporary state (use `_status/cells/` cell files)
- Daily progress (笔记 chronicle)
- Paper strategy decisions (paper_planning §19 decision log)

**Cross-references**:
- `docs/checkpoints/实验笔记.md §110-§113` — chronicles 5/5+5/6+5/7
- `docs/checkpoints/advisor_sync_5_5_outcomes.md §A.8` — compute path original 5/5 sync table
- `docs/checkpoints/pre_run/preregistration.md` — paper-grade gating
- `docs/reference/DGX_SPARK_MACHINE_QUIRKS.md` — DGX-specific quirks (sm_121 nvrtc fallback, etc.)
