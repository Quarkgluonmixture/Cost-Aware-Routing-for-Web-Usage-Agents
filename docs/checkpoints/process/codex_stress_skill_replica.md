---
description: Cross-AI audit (Mode B) — invoke codex CLI as independent hostile reviewer focused on code/pipeline/reproducibility-audit attacks. Codex reads scripts + pipeline + recent commits cold (no Claude/Gemini prior context), writes its own claim-by-claim attack. Default-chained from /stress at all scope bands. Best for §A1 implementation-layer code-audit where OpenAI codex lineage's reproducibility-auditor strength complements Anthropic Claude + Google Gemini.
---

# /codex-stress — Cross-AI Audit (Mode B)

## Why this skill exists

User explicit feedback 2026-05-12 after 2 consecutive missed paper-grade gaps (L23 cosine vs L17 patching layer disjoint; axis-2 random-injection control absence). Single-model self-audit has systematic blind spots — Claude reading its own narrative locks priors. Codex (GPT-derived lineage) reads paper drafts + code + pipeline cold; finds different angles.

**Mode B status: LIVE** (auto-chain from `/stress` since 2026-05-12 evening, always-chain at any scope since 2026-05-12 late evening). Component of cross-AI three-lineage architecture (Claude / codex / Gemini, see master `/stress` v7).

## Scope discipline — code/pipeline/reproducibility lineage

Empirical placement of codex strength (per cumulative `/codex-stress` runs Mode A 2026-05-12 → pre-fire audit 2026-05-13/14):

| Surface | codex strength | Notes |
|---|---|---|
| Pipeline correctness (`scripts/queues/`, `scripts/maintenance/`, queue chain) | ⭐⭐⭐ | reproducibility-auditor persona calibrated for this |
| Code↔prose mismatch (function does X, paper says Y) | ⭐⭐⭐ | sibling-script propagation detection (v6) |
| Schema / config / migration consistency | ⭐⭐ | reads YAML + python flat |
| Statistical methodology (estimand definition, meta-analysis) | ⭐⭐ | covered but Gemini stronger on framing layer |
| Claim formation / external validity reasoning | ⭐ | Gemini stronger on prose claim-audit |

**Default scope**: §A1 implementation-layer code-audit, pipeline / script / data-manipulation surfaces. Master `/stress` chains Mode B at **all scope bands** (spot / milestone / pre-fire / submission). For prose-only audit → user can `prose only` to skip Mode B and keep Mode C (Gemini) only.

## Preflight — verify paths BEFORE dispatch (v7)

Empirical 2026-05-15 (Gemini Mode C pilot): a single wrong path (`section4_findings.md` vs actual `section4_empirical_findings.md`) caused silent partial-audit. Hard rule for both Mode B + C:

```bash
PROMPT=docs/checkpoints/codex_prompts/<scope>_<datetime>.md
BAD=0
grep -oE 'docs/[a-zA-Z0-9_/.-]+\.md|scripts/[a-zA-Z0-9_/.-]+\.py|p79/[a-zA-Z0-9_/.-]+\.py' "$PROMPT" \
  | sort -u | while read path; do
  [ ! -f "$path" ] && { echo "✗ MISSING: $path"; BAD=1; }
done
[ "$BAD" = 1 ] && { echo "Fix paths in prompt before dispatch"; exit 2; }
```

For code surfaces, **always discover via `find` / `grep -r`**, never hardcode from memory:

```bash
# Wrong: hardcode "scripts/analysis/aggregate_phantom_lift.py"
# Right: find scripts/ -name "aggregate*phantom*" → confirm exact name
```

## Smoke test (v7, 2026-05-15)

Before dispatch, verify codex CLI healthy:

```bash
echo "Reply with exactly OK" | codex exec --sandbox danger-full-access 2>&1 | grep -qE "^OK$|^OK\b"
if [ $? -ne 0 ]; then
  echo "⚠️  codex CLI unhealthy — Mode B skipped, surface to user"
  echo "Continue with Claude /stress + Mode C (Gemini) if applicable"
  exit 1
fi
```

Closes 2026-05-13 failure mode where 3 codex fires returned exit 0 but produced no actual review.

## Invocation

### Filename convention v7 (HHMMSS for concurrent-session safety)

Empirical 2026-05-15: parallel `/stress` sessions observed (commits `b1a31a2` + `1a2ff9b` from another session while we worked). Date-only filename `<scope>_<YYYY-MM-DD>.md` would collide. Use HHMMSS:

```bash
DATE=$(date +%Y-%m-%d_%H%M%S)
PROMPT=docs/checkpoints/codex_prompts/<scope>_${DATE}.md
OUTPUT_FINAL=docs/checkpoints/codex_outputs/<scope>_FINAL_${DATE}.md
OUTPUT_TRACE=docs/checkpoints/codex_outputs/<scope>_trace_${DATE}.log
```

### Dispatch

```bash
# Preflight: path-existence check (see above)
# Smoke test: see above

# Dispatch — backgrounded; `-o` for atomic final-message write
codex exec --sandbox danger-full-access \
  -o "$OUTPUT_FINAL" \
  < "$PROMPT" > "$OUTPUT_TRACE" 2>&1 &
CODEX_PID=$!
```

Use `run_in_background: true` on the Bash call so harness notifies on completion. **Do NOT poll.**

### Parallel with Mode C

When chained from master `/stress`, Mode B + Mode C dispatch in **parallel** (two separate `run_in_background: true` Bash calls). Both share preflight + smoke + post-flight verification. Harness notifies each independently. Total wallclock ≈ max(B, C), not sum. **Do NOT serialize unless explicit reason** (e.g., Mode C depends on Mode B output, which is never the case in current architecture).

### Why `-o` flag (codex-specific)

`codex exec < file.md > out.md` has three failure modes (per v5 spec):
- (a) stdin EOF immediately after file read → codex may treat as "session ending, finalize ASAP"
- (b) stdout redirect = block-buffered → final flush lost on premature exit
- (c) isatty=false codepath in some codex versions → simplified non-interactive output

`-o` writes structured final message via atomic file write, independent of stdout stream. **Always use -o** for /codex-stress.

### Filename convention details

- `<scope>_FINAL_<HHMMSS-date>.md` — atomic final message via `-o`; primary audit trail
- `<scope>_trace_<HHMMSS-date>.log` — raw stdout (reasoning + bash exec); debug only

Both gitignored at `docs/checkpoints/codex_outputs/` per commit `16179b8`. Replica of skill at `docs/checkpoints/process/codex_stress_skill_replica.md` is git-tracked.

### Model identification (v7, 2026-05-15)

Codex uses GPT-5-codex tier per OpenAI account. Record actual model from trace log:

```bash
grep -m1 -iE "^model: |using model:|gpt-[0-9]" "$OUTPUT_TRACE" | head -1 >> "$OUTPUT_FINAL.meta"
```

Skill metadata for current verification baseline: `codex-cli v0.130.0` (2026-05-15), model = `gpt-5-codex` per OpenAI subscription tier. Protects audit reproducibility if OpenAI changes routing.

## Prompt design (v7)

The codex prompt MUST:

1. **Establish independence**: "You have NOT seen any prior review of this paper. Do NOT read `docs/checkpoints/stress*.md` / `gemini_outputs/*` / any Claude analysis files. Write your review cold from paper drafts + code + recent commits only."
2. **Set hostile reviewer persona** with rotation menu (v6, retained for codex):
   - default: peer-lab top-tier reviewer
   - "be brutal" → reviewer-3-skeptical-3/10
   - rotation candidates: mechinterp implementer / ML systems engineer / stats methodologist / reproducibility auditor
3. **DO NOT enumerate attack categories** — per memory `feedback_lean_audit_prompts`, listing "check power / check confounds" creates list-shaped blind spot. Let codex find what's actually there.
4. **Force output format**:
   - **Scope declaration line** (state spot/milestone/pre-fire/submission at top)
   - **Verdict line** (one sentence on current state)
   - **Strong claims (survive attack)** — 1-3 things that hold up
   - **Weak claims (would tank under attack)** — claim quoted + attack + defuse evidence + effort estimate + P0/P1/P2 severity
   - **Honest gaps** — things not in paper that reviewer expects
   - **Distance to top-tier** — current-tier / dominant gap / unblock plan
   - **ONE highest-leverage move** (24h)
5. **Output language**: 中文为主双语 (per project rule); criticism specifics in technical English
6. **Read order** (suggested, not strict):
   - Code-heavy scope: scripts/<scope>/ → p79/<module>/ → recent commits → relevant paper §
   - Prose-heavy scope: paper_drafts/section*.md → preregistration → paper_planning § → recent commits

Template at `.claude/skills/codex-stress/prompt_template.md`. v6 added persona rotation + scope-handoff file from Claude → codex (complementary coverage structurally enforced).

## Post-flight verification — MANDATORY (v7)

**Three-phase verification protocol shared with Mode C. Canonical source: master `/stress` SKILL.md "Post-flight verification" section.** Brief summary below; if drift suspected, master wins.

### Phase 1 — I/O sanity (automated)

```bash
OUT="$OUTPUT_FINAL"
SIZE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
ISSUES=""

[ "$SIZE" -lt 2000 ] && ISSUES+="output <2KB; "
grep -iE "File not found|Error executing tool|cannot read|permission denied" "$OUT" && ISSUES+="file-read errors; "
tail -c 200 "$OUT" | tr -d '\n' | tail -c 1 | grep -qE "[.!?。！？\]\)）」』]" || ISSUES+="truncation risk; "
grep -qE "P0|P1|P2|Severity" "$OUT" || ISSUES+="no severity tags; "
grep -qE "Distance to top.?tier|leverage" "$OUT" || ISSUES+="missing required sections; "
```

**codex-specific failure mode**: `-o` file empty but trace has partial reasoning → extract verbatim critique from `^codex$` markers in trace, label as "partial codex output", **don't fake the diff**.

### Phase 2 — depth/scope sanity (Claude judges)

| Scope | Finding count min | OOB count min |
|---|---|---|
| spot-check | 3 | 1 |
| milestone | 5 | 2 |
| pre-fire | **8** (codex code-audit benefits from higher floor than baseline 7) | **3** |
| submission-ready | 10 | exhaustive |

Plus: specificity (file:line / function name / commit hash), persona drift check (codex sometimes drifts to friendly summary), cold-read integrity (no Claude/Gemini context bleed).

### Phase 3 — runtime sanity

| Scope | Expected wallclock |
|---|---|
| spot-check | 1-5 min |
| milestone | 3-10 min |
| pre-fire | 8-20 min |
| submission-ready | 15-40 min |

### Retry budget

Max **2 retries**. After 3rd failure surface to user per master /stress retry matrix. Option (b) "swap model" → regenerate prompt for swap target (codex → emphasis on code paths in scope; gemini → emphasis on prose paths).

## Fix-verification mandate (v6, retained)

For each inline fix codex applies:
- Run `python3 -m py_compile <fixed_file>` — error → revert, document as "fix attempt failed"
- Data-altering fixes: codex must NOT auto-apply — document only
- Non-data-altering (grid check, label fix, provenance add): codex applies + verify `git diff --check` + py_compile
- Verification status in diff section: `PATCHED + verified` / `PATCHED but py_compile failed` / `DEFUSE PENDING (data-altering)`

## Retrospective hook (v6, retained — now covers Mode B+C)

Within **7 days** after each /codex-stress + /gemini-stress, append to 实验笔记 retro entry:
- For each finding [P0/P1]: did it surface a real bug? (YES / NO / UNKNOWN)
- Spec drift suggestion: pattern across audits that vN+1 should encode?

## Output integration

After codex completes + verification passes:

1. Read `$OUTPUT_FINAL`
2. If running inside `/stress` chain → contribute to 3-way diff (master stress orchestrates):
   - **What codex caught that Claude+Gemini missed**: codex unique catches
   - **What Claude+Gemini caught that codex missed**: codex blind spots (sanity)
   - **Where all three agree**: highest-confidence weak claims, top priority defuse
3. If standalone invocation → 1-section codex findings + recommend chaining Claude + Gemini for full 3-way

## Chronicle trigger

Per memory `feedback_chronicle_on_milestone.md` trigger 4: completion of `/codex-stress` (alongside `/stress` + `/gemini-stress`) is itself a milestone — append to `实验笔记.md` under `[infra]` tag if new findings surface.

## Bypass conditions

→ **Canonical at `.claude/skills/stress/SKILL.md` "Bypass conditions"** (master /stress). Quick reference:
- "skip codex" / "no codex" → Mode B skipped
- "claude only" / "no cross-AI" → all cross-AI skipped
- "prose only" → skip B (keep C Gemini)
- codex smoke test failed → Mode B skipped + surface

If this list drifts from master, **master wins**.

## What this skill is NOT

- NOT a citation checker (separate skill candidate: `/codex-cite-check`)
- NOT a paper-grade scorer (qualitative reviewer voice, not numeric)
- NOT a substitute for `/gemini-stress` (different lineage, complementary)
- NOT a "double-confirm Claude is right" tool — value = blind spot discovery, not validation

## Trust model

- Outputs **gitignored** at `docs/checkpoints/codex_outputs/` per commit `16179b8` (correcting v1 claim of "git-tracked")
- Replica git-tracked at `docs/checkpoints/process/codex_stress_skill_replica.md`
- `--sandbox danger-full-access` per CLAUDE.md Codex section; risk = trust prompt content (prompt is reviewable)
- Codex applies inline fixes within sandbox; fix-verification mandate (v6) catches broken fixes via py_compile

## Versioning

- v1 (2026-05-12): Initial Mode A standalone skill. Mode B chain planned but not yet live.
- **v7 (2026-05-15)**: Parity sync with master `/stress` v7 + `/gemini-stress` v1. Driver: 2026-05-15 self-audit caught 3 direct contradictions between codex-stress v1 and master stress v7 (Mode B "planned" vs live ~3 days; outputs "git-tracked" vs actually gitignored; codex "not a code reviewer" vs master defaulting codex to code-heavy scope). codex-stress was fossilized because Mode B auto-chain consumed its evolution driver — nobody re-read SKILL.md after chain auto-fired. v7 additions: (1) scope discipline explicit (§A1 code lineage, table of strengths); (2) preflight path-check + smoke test + post-flight 3-phase verification mandatory; (3) bypass canonical delegated to master /stress; (4) HHMMSS filename to prevent concurrent-session collision (2026-05-15 observed parallel `/stress` runs `b1a31a2` + `1a2ff9b`); (5) model identification recorded for reproducibility; (6) parallel-with-Mode-C declared explicit (max-not-sum wallclock); (7) chronicle trigger inclusion; (8) retry option (b) "swap model" mechanics with prompt regeneration spec.
