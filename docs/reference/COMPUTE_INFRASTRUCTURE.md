---
type: reference
status: live
created: 2026-05-07
last_updated: 2026-05-07
audience: self + advisor sync prep
---

# Compute Infrastructure Landscape (P79)

> **Living document** — update when compute access changes (new accounts, network policy, deprecations).
>
> **Last major update**: 2026-05-07 morning. UCL Condense A100 allocated 5/6 + Myriad HPC account activated 5/6 evening + GPU attached to A100 5/7 morning. SSH path via condenser bastion still pending cert generation on quark.

---

## §0 TL;DR

| Tier | Platform | Status | Use case |
|---|---|---|---|
| **0** | UCL Condense **A100 80GB dedicated** | ✅ allocated 5/6 / GPU attached 5/7 by Steve / SSH path pending cert | **Paper-grade primary**: Stage 2B scale-up + Llama-4 cross-arch + 16-cell rerun |
| **1** | UCL **Myriad HPC** (V/U-type 4× A100 80GB / L-type 4× A100 40GB / E/F-type 2× V100) | ✅ account activated 5/6, password SSH from quark works 5/7 | Backup + CPU batch + parallel cross-arch + future SAE training (4-GPU data-parallel) |
| **2** | **DGX Spark** (`spark-9ea3`) shared lab | ✅ stable, no admin/sudo, lab Tailscale `981526092.github` | Archived data source / VWA Docker Tailscale bridge / curation done (笔记 §113) |
| **3** | Advisor 5090 (post AI Center 搬运) | ⏳ pending | Backup if Condense fails, advisor offered 5/5 sync |
| ❌ | RunPod 4090 self-fund $200 | deprecated by Tier 0 | not needed |
| ❌ | Myriad pre-5/6 | superseded by 5/6 reactivation | no longer "abandoned" — see §1.2 |

---

## §1 Platforms Detail

### §1.1 UCL Condense A100 80GB dedicated ⭐ Tier 0

**Provider**: UCL ARC Condense (Harvester KubeVirt over Rancher).
**Namespace**: `arc-proj-webarena-ns`.
**VM**: `a100-jiaming-mech` — 16 CPU, 64 GB RAM, 500 GB disk, Ubuntu 22.04 cloud image, KeyPair `jiaming-dgx-spark` (fingerprint `26:53:bb:2f:26:e8:6c:9b:56:ff:14:aa:2a:3a:fe:fe`).
**GPU**: 1× NVIDIA A100 80GB (PCI passthrough, attached by Steve 5/7 morning after VM created — VM moved to host with GPU, IP changed 10.52.12.75 → **10.52.6.89**).
**Cost**: $0 (UCL allocation, NOT student-funded; supersedes RunPod budget plan).
**Wallclock**: unlimited (dedicated, single-tenant, no queue).
**Persistent state**: 500 GB disk persistent.

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
**Account**: `ucab352` activated 2026-05-06 (per "We are happy to confirm" email, retroactively re-applied — previous account may have lapsed).
**Auth**: UCL password (key auth setup pending, see §3.4).

**Node types relevant to us**:
| Type | GPU | VRAM | Count | Notes |
|---|---|---|---|---|
| **V** | 4× A100 80GB | 80 GB | 2 nodes | ⭐ matches Condense GPU spec |
| U | 4× A100 80GB | 80 GB | 1 node | similar V |
| L | 4× A100 40GB | 40 GB | 6 nodes | smaller VRAM but Qwen3-VL-4B fits |
| E/F | 2× V100 | 32 GB | 19 nodes | older but plentiful |

**Quotas**: 1 TB home (= scratch), `gquota` to check. Backed-up `~/ACFS` 1 TB read-only from compute nodes.
**Wallclock**: 72h single-core, 48h parallel (chunkable). 16-cell rerun (~24h per cell) fits comfortably.
**Pre-built modules**: PyTorch 1.11 GPU (load module list per `~/.rc.ucl.ac.uk/docs`), CUDA 11.x, Python 3.7-3.11 stack.

**Job submission paradigm**: SGE qsub batch scripts:
```bash
#$ -l h_rt=24:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -ac allow=V    # request V-type (A100 80GB)
#$ -wd /home/ucab352/scratch/jobs
```

**Old "abandoned" status retracted**: Pre-2026-05-06 we deprecated Myriad due to (a) Tailscale CGNAT block (Myriad → home VWA Docker blocked), (b) wallclock fits single-block 16-cell rerun. Updated assessment 2026-05-07:
- (a) Still applies — Myriad cannot reach quark VWA Docker via Tailscale → **16-cell rerun NOT feasible on Myriad** (needs live VWA env)
- (b) Wallclock fine for chunked 16-cell — but Tier 0 A100 dedicated has no wallclock so still primary
- ✅ **Mechanistic / cross-arch / SAE on archived data is fine on Myriad** (no VWA needed, just LLM forward pass)

**Use cases (5/7 update)**:
1. ⭐ Mechanistic Stage 2B parallel (if Condense busy + Myriad V-type queue available)
2. ⭐ Llama-4 cross-arch (4-GPU data-parallel on V-type)
3. ⭐ SAE training (paper v2, defer) — V-type 4× A100 80GB perfect for SAE training scale
4. ⏳ Analysis batch jobs (CPU-only on D-type) without burning A100 quota
5. ❌ 16-cell rerun (CGNAT VWA block)
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
- ✅ Local Docker Desktop WSL2 backend running VWA stack (5 containers, ports 9980/9999/7770/7780/8888/4399)
- ✅ User has admin (can install OpenSSH Server / generate keys / configure)
- ✅ VSCode for Windows + Remote-SSH extension → A100 dev workstation

**Roles**:
- **Primary SSH client** for A100 Condense + Myriad
- **VWA Docker host** for paper-grade 16-cell rerun (DGX or A100 needs to reach via Tailscale)
- **Cert generation host** (UCL VPN + browser → SSH Portal)
- **Data transfer pivot** (rsync DGX → quark via Tailscale, scp quark → A100 via UCL VPN+bastion)

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

### §3.2 Quark → Myriad

```powershell
ssh ucab352@myriad.rc.ucl.ac.uk
# Password prompt — UCL password
```

Pending key auth setup (§3.4). Until then, password every login (Cisco VPN must be active).

### §3.3 Quark → DGX (lab Tailscale)

```bash
ssh jiaming@spark-9ea3   # via Tailscale, no UCL VPN needed
# or via IP: ssh jiaming@100.99.92.18
```

### §3.4 Myriad key auth setup (pending)

```bash
# On Myriad (after first password login):
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "<quark public key>" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Quark side: cat $HOME\.ssh\id_ed25519_dgx.pub  (or generate dedicated myriad key)
```

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

| Workload | Primary | Backup | Reason |
|---|---|---|---|
| Mechanistic Stage 2B scale-up (24-task curated) | Tier 0 A100 Condense | Tier 1 Myriad V-type | Both work; Condense faster iteration |
| Llama-4 cross-arch validation | Tier 0 A100 Condense | Tier 1 Myriad V-type 4× A100 | Cross-arch needs ≥1 GPU 80GB |
| 16-cell paper-grade rerun (post-advisor email + threshold lock) | Tier 0 A100 Condense | ⚠️ Myriad blocked (CGNAT VWA) | needs VWA Docker reach via Tailscale |
| SAE training (paper v2 deferred) | Tier 1 Myriad V-type 4× A100 data-parallel | Tier 0 single A100 (slower) | data-parallel scales SAE training |
| CPU analysis batch (figures, aggregation) | DGX local | Tier 1 Myriad D-type | no GPU needed, save A100 quota |
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

### §7.2 16-cell paper-grade rerun

- ✅ Pre-registration framework draft (`preregistration.md` status:draft)
- ⏳ Threshold witness from advisor (K_h1=0.75 / K_h3=0.67 / TOST δ=1.0pp)
- ⏳ Paper split decision (3 vs 4 papers)
- ⏳ Compute path lock (planned: A100 Condense, ~3-5d)
- ⏳ Launch on A100 (post-advisor email reply + SSH setup)

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
- `docs/checkpoints/preregistration.md` — paper-grade gating
- `docs/reference/MYRIAD_SMOKE_REPORT.md` — historical CGNAT investigation (now partly obsolete per §1.2)
- `docs/reference/RUNPOD_ONBOARDING.md` — deprecated by Tier 0
- `docs/reference/DGX_SPARK_MACHINE_QUIRKS.md` — DGX-specific quirks (sm_121 nvrtc fallback, etc.)
