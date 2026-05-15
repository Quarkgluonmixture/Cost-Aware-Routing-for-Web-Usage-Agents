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

```bash
DATE=$(date +%Y-%m-%d)
PROMPT=docs/checkpoints/gemini_prompts/<scope>_${DATE}.md
OUTPUT=docs/checkpoints/gemini_outputs/<scope>_${DATE}.md

# Preflight: path-existence check (see above)

# Dispatch — backgrounded, harness will notify on completion
gemini --approval-mode plan -p "$(cat "$PROMPT")" > "$OUTPUT" 2>&1 &
GEMINI_PID=$!
```

Use `run_in_background: true` on the Bash call so harness tracks completion automatically. **Do NOT poll** — harness notifies when done.

### Why `--approval-mode plan` (not `--yolo`)

`plan` mode = read-only. Gemini can read project files via its tools but cannot write to working tree. For audit (read prose, output critique to stdout), this is the correct minimum permission. `--yolo` is for agent tasks that need to edit.

### Auth

User OAuth via `GOOGLE_GENAI_USE_GCA=true gemini` (once, persists to `~/.gemini/`). Subsequent runs no env var needed. Falls back to `GEMINI_API_KEY` env var if OAuth not present.

## Prompt design

Gemini prompt MUST:

1. **Establish persona** — top-tier ML/NLP reviewer (NeurIPS/ICML/ACL/ICLR), implements methodology rigorously, debugs students' pipelines
2. **Establish cold-read constraint** — explicitly: "You have NOT seen any prior Claude or codex analysis"
3. **Name artifacts to read** — file paths verified by preflight; tell Gemini to use its file tools
4. **Force read-prose-not-code** — "Do not read code. This is claim-audit, not code-audit. Implementation audit is a separate pass."
5. **DO NOT enumerate attack categories** — per `feedback_lean_audit_prompts`, listing "check power / check confounds / check framing" creates list-shaped blind spot. Let Gemini find what's actually there.
6. **Force output structure** — Top-N weakest claims (P0/P1/P2) + cross-cutting concerns + distance-to-top-tier + ONE highest-leverage move
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

When invoked inside `/stress`:
- Master `/stress` skill (Claude self-review) → auto-chain `/codex-stress` (Mode B) AND `/gemini-stress` (Mode C) by default
- Bypass with: "skip gemini" / "skip codex" / "claude only" / "no cross-AI"
- Bypass with: "code only" → skip Mode C (Gemini); keep Mode B (codex code-audit)
- Bypass with: "prose only" → skip Mode B (codex); keep Mode C (Gemini prose-audit)

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
