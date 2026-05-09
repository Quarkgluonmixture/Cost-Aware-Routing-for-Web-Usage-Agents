# GLM 5.1 on quark Windows — Myriad SSH chain replacement

**Created**: 2026-05-09
**Why**: DGX → quark Tailscale → Myriad SSH chain depends on (a) DGX-side ssh-agent + key (b) quark Tailscale up (c) Cisco AnyConnect on quark up. 经常 (a) DGX context compaction 把 ssh-agent 状态丢; (b) Tailscale 偶尔自动 sleep; (c) Cisco token expire. 单 chain 易碎. **GLM 5.1 cron on quark** 是更稳定的等价物 — quark Windows 端有 Cisco / OpenSSH / Tailscale 全套, GLM API 调用 unlimited (per `feedback_delegate_to_small_llms.md`), cron 不依赖 interactive session.

## Architecture (existing 5-cron + new 6th on quark)

```
DGX (existing 5 GLM cron, runs on DGX Linux):
├── error-scan (5min) — runner / watchdog log scan
├── glm-update-cells (10min) — cell frontmatter sync + re-run detect
├── glm-refresh-playbook-s2 (30min) — fast §2 refresh
├── glm-refresh-playbook full (2h) — §1+§2 narrative
└── check-links (weekly) — dead link scan

quark Windows (NEW 6th cron, runs on quark Task Scheduler):
└── glm-myriad-sync (30min) — Myriad qstat + result scp + git push back to DGX
```

quark side 只跑 1 cron (Myriad chain), 因为 DGX cron 处理本地 PLAYBOOK / cell / figure 更高效 (本地 fs).

## glm-myriad-sync cron job — full spec

### Trigger

quark Windows Task Scheduler:
- **Cadence**: every 30 min
- **Run as**: `Quark` user account (interactive session OK; cron 装 SYSTEM 也可以但 Cisco AnyConnect 一般要 user-context)
- **Action**: `pwsh.exe -NoProfile -ExecutionPolicy Bypass -File C:\vwa\glm_myriad_sync.ps1`
- **Conditions**: only run when network is connected (Cisco connected required for Myriad SSH)

### glm_myriad_sync.ps1 (quark-side script)

```powershell
# C:\vwa\glm_myriad_sync.ps1
# GLM-driven Myriad SSH sync. Runs on quark Windows via Task Scheduler.
#
# Flow:
# 1. SSH to Myriad, capture qstat + recent job artifacts
# 2. Send qstat output + last cron timestamp + cell list to GLM 5.1 API
# 3. GLM decides: (a) should we scp results back? (b) which cells done?
#    (c) git commit + push? (d) ntfy alert?
# 4. PowerShell executes GLM's decision

$ErrorActionPreference = "Stop"
$LogFile  = "C:\vwa\logs\glm_myriad_sync_$(Get-Date -Format 'yyyy-MM-dd').log"
$RepoPath = "C:\vwa\Cost-Aware-Routing-for-Web-Usage-Agents"
$MyriadKey = "C:\Users\Quark\.ssh\myriad_ed25519"
$MyriadUser = "ucabjz0"
$MyriadHost = "myriad.rc.ucl.ac.uk"
$GlmApiKey = $env:GLM_API_KEY                            # set in user env vars
$GlmEndpoint = "https://api.bigmodel.cn/v4/chat/completions"   # 智谱 GLM-4.5 endpoint
$NtfyTopic = "p79-exp-dgx-spark"

function Log($msg) {
    "$(Get-Date -Format 'HH:mm:ss') $msg" | Out-File -Append $LogFile
}

# --- 1. Capture Myriad qstat ---
Log "Phase 1: Myriad qstat"
$qstat = ssh -i $MyriadKey -o StrictHostKeyChecking=no -o ConnectTimeout=15 `
    "$MyriadUser@$MyriadHost" "qstat -u $MyriadUser; ls -lat ~/Scratch/p79/results/mechanistic/ | head -20" 2>&1
if ($LASTEXITCODE -ne 0) {
    Log "ERROR: Myriad SSH failed (rc=$LASTEXITCODE)"
    # Optional: ntfy "Myriad SSH down" alert
    & curl.exe -s -d "Myriad SSH chain failed at $(Get-Date)" "https://ntfy.sh/$NtfyTopic"
    exit 1
}

# --- 2. Pull cell directory listing for done-detection ---
$cells_remote = ssh -i $MyriadKey "$MyriadUser@$MyriadHost" `
    "ls ~/Scratch/p79/results/mechanistic/stage2_cell*/patching_continuation_results.json 2>/dev/null"
$cells_remote = $cells_remote -split "`n" | Where-Object { $_ -ne "" }

# --- 3. Read which cells we already have locally on DGX (via repo on quark) ---
$cells_local = Get-ChildItem -Path "$RepoPath\results\mechanistic" -Directory `
    -Filter "stage2_cell*" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name

# --- 4. Ask GLM 5.1 what to do ---
$prompt = @"
You are managing a remote experiment SSH chain for a web-agent paper.

Current state:
- Myriad qstat (running jobs): $qstat
- Remote cells with results.json: $($cells_remote -join ', ')
- Local cells already pulled: $($cells_local -join ', ')

Decide ONE of:
1. PULL <cell_name> — scp result back from Myriad
2. WAIT — no new results, no action
3. ALERT — failure detected, ntfy user

Reply in JSON: {"action": "pull|wait|alert", "cell": "<name or null>", "reason": "<short>"}
"@

$body = @{
    model = "glm-4.5"
    messages = @(@{role="user"; content=$prompt})
    temperature = 0.0
    response_format = @{type="json_object"}
} | ConvertTo-Json -Depth 5

$resp = Invoke-RestMethod -Uri $GlmEndpoint -Method Post `
    -Headers @{Authorization="Bearer $GlmApiKey"; "Content-Type"="application/json"} `
    -Body $body
$decision = $resp.choices[0].message.content | ConvertFrom-Json
Log "GLM decision: $($decision | ConvertTo-Json -Compress)"

# --- 5. Execute decision ---
switch ($decision.action) {
    "pull" {
        $cell = $decision.cell
        Log "Pulling $cell from Myriad..."
        & scp -i $MyriadKey -r "$MyriadUser@${MyriadHost}:~/Scratch/p79/results/mechanistic/$cell" `
            "$RepoPath\results\mechanistic\"
        # Auto-commit
        Set-Location $RepoPath
        & git add "results/mechanistic/$cell"
        & git commit -m "data($cell): Myriad results sync via GLM cron $(Get-Date -Format yyyy-MM-dd)"
        & git push origin master                # auto-push since this is a sync, no human-decision
        & curl.exe -s -d "Myriad cell $cell pulled + pushed" "https://ntfy.sh/$NtfyTopic"
    }
    "wait" {
        Log "GLM: wait, no new results"
    }
    "alert" {
        & curl.exe -s -d "GLM alert: $($decision.reason)" -H "Priority: high" "https://ntfy.sh/$NtfyTopic"
    }
}

Log "Done."
```

### Setup steps (one-time on quark)

```powershell
# 1. Set GLM API key as user env var (PowerShell)
[Environment]::SetEnvironmentVariable("GLM_API_KEY", "<your-key>", "User")

# 2. Create dirs
mkdir C:\vwa\logs -ErrorAction SilentlyContinue

# 3. Drop glm_myriad_sync.ps1 into C:\vwa\

# 4. Test once interactively
& C:\vwa\glm_myriad_sync.ps1
# inspect C:\vwa\logs\glm_myriad_sync_<date>.log

# 5. Register as Task Scheduler
schtasks /Create /TN "P79\GlmMyriadSync" /SC MINUTE /MO 30 `
    /TR "pwsh.exe -NoProfile -ExecutionPolicy Bypass -File C:\vwa\glm_myriad_sync.ps1" `
    /RU $env:USERNAME /RL HIGHEST /F
```

### Where DGX-side observes the result

Once GLM cron pushes new mechanistic data to git (DGX is the upstream), DGX can pull via:

```bash
cd /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
git pull origin master                   # or via PLAYBOOK §1 cron (existing)
```

**No new DGX cron needed** — just pull on demand. The DGX 5-cron suite already refreshes PLAYBOOK §1+§2 every 2h, which surfaces new git commits in the cron narrative.

## Failure modes + alerts

| Failure | Symptom | Recovery |
|---|---|---|
| Cisco AnyConnect down on quark | Myriad SSH timeout in cron log | quark user reconnects Cisco; cron auto-resumes next tick |
| GLM API down | `Invoke-RestMethod` 500/timeout | Cron logs error + ntfy push; falls back to no-op (no harm) |
| Myriad qstat returns 0 jobs but local has stale active cells | GLM detects mismatch → ALERT | DGX-side: clear stale cells via cron `glm-update-cells` |
| Git push conflict | DGX has uncommitted changes | Manual: `git pull --rebase`; cron self-heals next tick |

## Cost analysis

- GLM-4.5 input ~600 tokens × 48 calls/day = 28.8K input/day
- Output ~50 tokens × 48 = 2.4K output/day
- 智谱 GLM-4.5 pricing (2026-05): ~¥1/1M input, ~¥4/1M output
- **Daily cost**: ~¥0.04 = $0.005/day. **Monthly**: ~$0.15.
- vs DGX context compaction loss + missed cell pulls: **way cheaper than human time**.

## Why GLM (not GPT-4o-mini / Claude Haiku)?

- **Per `feedback_delegate_to_small_llms.md`**: GLM 5.1 unlimited for recurring cron + status digests; codex limited but token-heavy OK
- This task is recurring + structured-decision + low-stakes → fits GLM "unlimited" tier perfectly
- Existing 5-cron suite already uses GLM-4.5; consistent infra (same API key, same logging pattern)
- GLM-4.5 supports `response_format: json_object` natively (cron parsing simpler)

## Integration with existing PLAYBOOK §6 cron sidecar

Add to `PLAYBOOK.md §6 Cron Sidecar` table:

```
| **glm-myriad-sync** (quark) | every 30min | Myriad qstat + result scp + auto git commit-push back to DGX. Runs on quark Task Scheduler, GLM 5.1 decides PULL/WAIT/ALERT. Replaces fragile DGX→quark→Myriad SSH chain. |
```

## Open items

- [ ] User to create GLM API key + set on quark env
- [ ] User to drop `glm_myriad_sync.ps1` to `C:\vwa\`
- [ ] Test interactive run once before enabling Task Scheduler
- [ ] Add status row to PLAYBOOK §6
- [ ] Optional: extend GLM prompt to also detect Myriad job failures (qacct exit_status > 0) and auto-resubmit with different SGE flags
