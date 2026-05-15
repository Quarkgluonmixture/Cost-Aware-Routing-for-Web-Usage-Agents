---
description: Cross-AI audit (Mode C) — invoke gemini CLI as independent hostile reviewer focused on prose/claim/design-layer attacks. Gemini reads paper drafts + preregistration + planning prose cold (no SCRIPTS, no Claude/codex prior context), writes its own claim-by-claim attack. Best for §A2 design-layer / framing / methodology / statistical-design audits where Anthropic-Claude + OpenAI-codex two AI lineage may share blind spots Google-Gemini doesn't.
---

# /gemini-stress — Cross-AI Audit (Mode C)

## Why this skill exists

User has Gemini subscription (paid) → free-quota high. Empirical pilot 2026-05-15 on P79 §A2 design-layer audit caught:
- **#1 Self-Oracle baseline (P0)** — drop-one oracle lift may be stochastic-noise artifact in low-SR regime; needs DOM_seed2 vs DOM_seed1 ablation. Claude+codex 7-day audit history did NOT catch.
- **#2 FE meta on opposing mechanisms (P0)** — fixed-effects pooling structurally heterogeneous cells (site-modulated) = uninterpretable estimand. Sharper than prior K-of-N reclassification.
- **#4 Trajectory total-cost confound (P1)** — per-step token ≈ DOM ≠ total episode cost; failure-loop steps eat the savings.
- **#7 P-prompt graceful-degradation (P1)** — may not be a real axis, just B0 robustness to mismatched format.

Three independent AI lineages (Anthropic / OpenAI / Google) → blind spots stack additively, not multiplicatively. Mode C closes the third corner.

## Scope discipline — prose/design, NOT code

Gemini's empirical strength = **paper claim-audit / framing / statistical design / external validity reasoning**. Empirical weakness (per user 2026-05-15) = code-level audit.

**Default scope**: §A2 design-layer surfaces (paper_drafts + preregistration + paper_planning prose + claim-formation logic). Read-only by construction (`--approval-mode plan`).

**Discouraged scope**: code/pipeline audit (use `/codex-stress` instead — codex's reproducibility-auditor persona calibrated for code). If user explicitly requests Gemini on code, allow but flag the lineage mismatch.

## Preflight — verify paths BEFORE dispatch

Empirical 2026-05-15: a single wrong path (`section4_findings.md` vs `section4_empirical_findings.md`) wasted one full audit run. Hard rule:

```bash
PROMPT=docs/checkpoints/gemini_prompts/<scope>_<date>.md
BAD=0
grep -oE 'docs/[a-zA-Z0-9_/.-]+\.md' "$PROMPT" | sort -u | while read path; do
  [ ! -f "$path" ] && { echo "✗ MISSING: $path"; BAD=1; }
done
[ "$BAD" = 1 ] && { echo "Fix paths in prompt before dispatch"; exit 2; }
```

Better: use `find docs/ -name 'section4*'` to **discover** real file names instead of hardcoding from memory.

## Invocation

### Filename convention v7.1 (HHMMSS for concurrent-session safety)

Empirical 2026-05-15: parallel `/stress` sessions observed (commits `b1a31a2` + `1a2ff9b` ran while we worked). Date-only would collide. **Use HHMMSS**:

```bash
DATE=$(date +%Y-%m-%d_%H%M%S)
PROMPT=docs/checkpoints/gemini_prompts/<scope>_${DATE}.md
OUTPUT=docs/checkpoints/gemini_outputs/<scope>_${DATE}.md
DEBUG=docs/checkpoints/gemini_outputs/<scope>_${DATE}.debug.log
```

### Smoke test (v7.1, 2026-05-15)

Before dispatch verify CLI + auth + quota healthy:

```bash
echo "" | timeout 15 gemini -p "Reply exactly OK" --approval-mode plan 2>&1 | grep -qE "^OK\b"
if [ $? -ne 0 ]; then
  echo "⚠️  gemini CLI unhealthy — Mode C skipped, surface to user"
  echo "Continue with Claude /stress + Mode B (codex) if applicable"
  exit 1
fi
```

### Dispatch

```bash
# Preflight: path-existence check (see above)
# Smoke test: see above

# Dispatch — backgrounded; --debug captures routing info for model identification
gemini --approval-mode plan --debug -p "$(cat "$PROMPT")" > "$OUTPUT" 2> "$DEBUG" &
GEMINI_PID=$!
```

Use `run_in_background: true` on the Bash call so harness notifies on completion. **Do NOT poll.**

### Parallel with Mode B

When chained from master `/stress`, Mode C + Mode B dispatch in **parallel** (two separate `run_in_background: true` Bash calls). Total wallclock = **max(B, C), not sum**.

### Why `--approval-mode plan` (not `--yolo`)

`plan` mode = read-only. Gemini can read project files via its tools but cannot write to working tree. For audit (read prose, output critique to stdout), this is the correct minimum permission. `--yolo` is for agent tasks that need to edit.

### Model identification + silent-fallback warning (v7.1)

**Default routing**: `--approval-mode plan` auto-routes to Pro tier. Verified 2026-05-15: `gemini-3.1-pro-preview` per debug log line "[Routing] Selected model: gemini-3.1-pro-preview (Source: agent-router/approval-mode)".

**Record actual model**:

```bash
grep -m1 "Selected model:" "$DEBUG" | head -1 >> "$OUTPUT.meta"
```

**Don't pass `-m`** unless you've verified the model name via `--debug` first. Silent fallback observed 2026-05-15: `-m gemini-3-pro-preview` (no dot between `3` and `1`) silently downgrades to `gemini-2.5-pro` **without error**. Typo → quality degradation no one sees.

If `-m` is needed, run a 1-line smoke first: `gemini --debug -m <name> -p "say ok" 2>&1 | grep "Selected model:"` to confirm the requested name is honored.

### Auth

User OAuth via `GOOGLE_GENAI_USE_GCA=true gemini` (once, persists to `~/.gemini/`). Subsequent runs no env var needed. Falls back to `GEMINI_API_KEY` env var if OAuth not present.

## Prompt design

Gemini prompt MUST:

1. **Establish persona** — top-tier ML/NLP reviewer (NeurIPS/ICML/ACL/ICLR), implements methodology rigorously, debugs students' pipelines. **No rotation menu** (unlike Mode B codex which has 4 personas). Empirical 2026-05-15 pilot: Gemini's strength = broad framing reasoning across §A2 surfaces. Persona rotation would narrow that breadth. If user wants deep persona-specific dive → invoke /codex-stress with rotation menu instead.
2. **Establish cold-read constraint** — explicitly: "You have NOT seen any prior Claude or codex analysis"
3. **Name artifacts to read** — file paths verified by preflight; tell Gemini to use its file tools
4. **Force read-prose-not-code** — "Do not read code. This is claim-audit, not code-audit. Implementation audit is a separate pass."
5. **DO NOT enumerate attack categories** — per `feedback_lean_audit_prompts`, listing "check power / check confounds / check framing" creates list-shaped blind spot. Let Gemini find what's actually there.
6. **Force output structure** — Top-N weakest claims (P0/P1/P2) + **Bug Table (REQUIRED v7.2, 2026-05-15)** + cross-cutting concerns + distance-to-top-tier + ONE highest-leverage move

   **Bug Table spec** (consolidated user-facing actionable summary after Weak claims, 3 cols only grouped by severity P0/P1/P2):

   ```markdown
   ### 🔴 P0 (lock 前必须 fix)
   | # | Bug | Blast Radius | Launch 卡? |
   ```

   - **Bug**: short id (`Pn-i`) + `file:line` (paper draft section + line) + 1-sentence diagnosis (which wording / claim / framing is broken)
   - **Blast Radius** (人话, 2-4 sentences): (a) what role this prose / claim / preregistration field plays in the paper, (b) what happens concretely if reviewer reads paper without this fix (specific reviewer attack quoted), (c) which paper section / OSF artifact is corrupted. For abstract methodology bugs (statistical estimand / experimental design) include 1-line generalization analogy.
   - **Launch 卡?**: `不卡` / `不卡 launch,block OSF lock` / `不卡 launch,卡 paper write` (Mode C bugs rarely block actual launch — usually block OSF lock cleanliness or paper-grade claim defensibility)
   - Rationale: severity tag alone (P0/P1/P2) not actionable; Bug Table converts each finding into "this is what reviewer will catch if you submit without this fix"
7. **Independence reminder at end** — "Your value = blind spots the other two miss. Don't try to be comprehensive; try to be independent."

Output language: 中文为主双语 (per project rule); technical critique specifics in English.

## Post-flight verification — MANDATORY (NEW v1, 2026-05-15)

Per user directive 2026-05-15: "gemini, codex 每次 stress 都需要看是不是完备 / 没 bug。如果不行就针对某一个部分重试。"

**Claude must verify Gemini output BEFORE showing user the diff.** No silent partial-audit surfacing.

### Phase 1 — I/O sanity (cheap, automated)

```bash
OUT=docs/checkpoints/gemini_outputs/<scope>_<date>.md
SIZE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
ISSUES=""

# Too small → likely early termination / auth fail
[ "$SIZE" -lt 2000 ] && ISSUES+="output <2KB likely failure; "

# File-read errors inside output → some artifact missed
grep -iE "File not found|Error executing tool|cannot read|permission denied|No such file" "$OUT" \
  && ISSUES+="missing-file error in output; "

# Truncation (ends mid-sentence)
LAST=$(tail -c 200 "$OUT" | tr -d '\n' | tail -c 1)
echo "$LAST" | grep -qE "[.!?。！？\]\)）」』]" || ISSUES+="ends mid-sentence (truncation risk); "

# Structure markers
grep -qE "P0|P1|P2|Severity" "$OUT" || ISSUES+="no severity tags; "
grep -qE "Distance to top.?tier|distance-to-top-tier|leverage|highest.?leverage" "$OUT" \
  || ISSUES+="missing required output sections; "
```

### Phase 2 — depth/scope sanity (Claude reads + judges)

Read the output. Judge against scope target:

| Scope | Finding count min | OOB count min |
|---|---|---|
| spot-check | 3 | 1 |
| milestone | 5 | 2 |
| pre-fire | 7 | 3 |
| submission-ready | 10 | exhaustive |

Also check:
- **Specificity** — each finding quotes specific claim / file / number, not generic "could be improved"
- **Persona drift** — Gemini sometimes writes code or summarizes instead of attacking. If output looks like a friendly summary, fail.
- **Cold-read integrity** — Gemini didn't accidentally reference codex / Claude content (would mean it found and read prior audit files)

### Phase 3 — runtime sanity

| Scope | Expected wallclock |
|---|---|
| spot-check | 1-5 min |
| milestone | 3-10 min |
| pre-fire | 8-20 min |
| submission-ready | 15-40 min |

- <1 min = near-certain auth / quota / model error
- >30 min on smaller scope = possible hang

### Retry decision matrix

| Failure mode | Action |
|---|---|
| Single missing-file in output | Identify path via grep + `find docs/ -name <pattern>` → patch prompt → rerun (cheap) |
| Truncation | Continuation prompt: "Continue from finding #N. Output ended mid-sentence; pick up where you left off." → append output |
| Persona drift / off-topic | Re-prompt with stronger persona anchor + explicit "this is hostile claim-audit, attack-by-attack, not summary" + scope declaration |
| Too few findings | Re-prompt with explicit "Required minimum: N findings (≥M OOB). Previous attempt produced K. Find more." |
| Suspicious-fast (<1 min) | Re-run as-is (transient error usually). Check `~/.gemini/` quota log if persistent. |
| Scope undeclared | Re-prompt to declare scope at output top + restate output structure |

### Retry budget

**Max 2 retries per audit invocation.** After 2 failed retries, surface failure mode to user:

> ⚠️ /gemini-stress failed verification 3× (failure mode: <X>). Options:
> (a) try different scope band (e.g. shrink milestone → spot-check)
> (b) swap to /codex-stress (different model lineage may handle this prose better)
> (c) accept partial output (Phase 1 passed, Phase 2 marginal)
> (d) manual review

## Diff integration (after verification passes)

If Claude /stress already ran in same session, produce **3-way diff section**:

```markdown
### What Gemini caught that Claude+codex missed (highest-value, this is why we run Mode C)
- <finding>: <severity> — <quote / why weak / defuse evidence / effort>

### What Claude+codex caught that Gemini missed (sanity check, Gemini lineage blind spots)
- <finding>: <severity> — usually code-level or implementation-detail attacks

### Where all three agree (highest-confidence weak claims, top priority defuse)
- <finding>: <severity>
```

If Claude /stress did NOT run (Gemini standalone invocation), produce 1-section Gemini findings dump + recommendation to chain Claude /stress for diff.

## Auto-chain via Mode B+C

When invoked inside `/stress`, master `/stress` skill (Claude self-review) → auto-chain `/codex-stress` (Mode B) AND `/gemini-stress` (Mode C) **in parallel** by default. Master /stress orchestrates 3-way diff after both verifications pass.

## Chronicle trigger

Per memory `feedback_chronicle_on_milestone.md` trigger 4: completion of `/gemini-stress` (alongside `/stress` + `/codex-stress`) is itself a milestone — append to `实验笔记.md` under `[infra]` tag if new findings surface.

## Bypass conditions

→ **Canonical at `.claude/skills/stress/SKILL.md` "Bypass conditions (canonical, v7.1)"** (master /stress). If this skill's bypass refs drift from master, **master wins**.

Quick reference (mirror only):
- "skip gemini" / "no gemini" → Mode C skipped
- "prose only" → keep C, skip B
- "claude only" / "no cross-AI" → Mode C skipped
- gemini smoke test failed → Mode C skipped, surface

## What this skill is NOT

- NOT a code reviewer (Gemini's empirical weakness; use `/codex-stress` for code)
- NOT a fact-checker on numerical claims (it'll hallucinate numbers; use it for framing not arithmetic)
- NOT a comprehensive audit (designed to be **independent**, not exhaustive — the value is blind spot discovery)
- NOT a substitute for `/codex-stress` (different lineage, different blind spots, complementary)

## Trust model

- Output git-status: gitignored (`docs/checkpoints/gemini_outputs/`), Mirror of `codex_outputs/` policy
- Prompt git-status: gitignored (`docs/checkpoints/gemini_prompts/`)
- `--approval-mode plan` = read-only, cannot modify working tree → safer than codex `--sandbox danger-full-access`
- Auth: OAuth Google personal account or `GEMINI_API_KEY` — both persist to user-owned files only

## Versioning

- v1 (2026-05-15): Initial. Mode C standalone + auto-chain from /stress. Built atop user 2026-05-14 advisor 收口 (router + Phase 1 critical path) + Gemini subscription. Pilot empirical 2026-05-15 §A2 design-layer audit caught 4 P0/P1 attacks Claude+codex missed in 7-day audit history. Post-flight verification protocol mandatory from v1 (user directive 2026-05-15 after pilot exposed silent partial-audit risk via section4_findings.md path typo).
- **v7.1 (2026-05-15 reflexive audit, parity with master v7.1)**: HHMMSS filename convention for concurrent-session safety / smoke test pre-dispatch / `--debug` model identification + record routing line / silent-fallback warning for `-m` typo (e.g. `gemini-3-pro-preview` no-dot → `gemini-2.5-pro` silent) / parallel-with-Mode-B explicit / persona "no rotation by design" (empirical Gemini broad-framing strength) / bypass conditions delegated to master canonical / chronicle trigger 4 referenced. Driver: user "整体审计下 skill" 2026-05-15 → /stress-on-skill reflexive audit caught 3-skill component family version-sync gaps.
