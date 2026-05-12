# Process rule: /codex-stress — cross-AI audit skill

User set 2026-05-12 after 2 missed paper-grade gaps surfaced (L23-vs-L17 layer disjoint; axis-2 random-injection control absence). Single-AI self-audit has systematic blind spots.

This is the git-tracked replica of `.claude/skills/codex-stress/SKILL.md` (gitignored). Cross-machine recovery: if `.claude/` is reset, re-create the skill from this file + `.claude/skills/codex-stress/prompt_template.md` (also git-tracked at `docs/checkpoints/process/codex_stress_prompt_template_replica.md`).

## Skill purpose

Invoke codex CLI as **independent hostile reviewer** that has NOT seen Claude's prior analysis. Codex (GPT-derived) has different prior; running on same paper drafts surfaces issues Claude missed.

**Mode A** (this skill): standalone `/codex-stress` user-invocable.

**Mode B** (planned): auto-chain inside Claude's `/stress` at milestone triggers (paper commit, push, milestone declaration). Output diff shown to user as part of /stress final report.

## When to invoke

### Manual (`/codex-stress`)
- User wants a second opinion on current paper state
- Before submission / advisor sync / external review, after Claude /stress
- Spot-check any decision point

### Auto-chain (Mode B, planned)
After Claude /stress completes at milestone triggers, automatically run /codex-stress on same scope.

## How it works

1. **Scope assembly**: Claude collects paper drafts + mechanism plan + recent evidence files + recent commits since last codex audit
2. **Prompt generation**: write codex prompt to `docs/checkpoints/codex_prompts/codex_stress_<date>.md` from template (independence instruction + hostile reviewer persona + output format + read order)
3. **Codex invocation**: `codex exec --sandbox danger-full-access < <prompt> > <output> 2>&1` foreground + PID monitor (Tier 3 per CLAUDE.md long-task rule)
4. **Diff surfacing**: Claude reads codex output, compares to its own /stress findings, produces sections:
   - "What codex caught that I missed"
   - "What I caught that codex missed"
   - "Where we agree (highest confidence)"
5. **Chronicle**: per 阶段性成果 rule, /codex-stress completion is itself a milestone → append to `实验笔记.md`

## Codex prompt design constraints

- **Independence**: prompt explicitly forbids codex reading `.claude/skills/stress/SKILL.md` or claude's prior /stress output
- **Hostile reviewer persona**: NeurIPS/ICML/ACL reviewer who has read 200+ papers in this space, brutal but specific, claim-by-claim attack, distance-to-top-tier framing
- **Output language**: 中文为主双语 (per project rule)
- **Output format**: verdict + strong claims + weak claims (quote+attack+defuse+effort) + honest gaps + distance-to-top-tier + one-tonight-fix

## Calibration

- Default: hostile but fair (peer-lab reviewer voice)
- "be brutal" → escalate to skeptical-reviewer-3 mode (use brutal template variant)
- "be gentle" → refuse, suggest skip /codex-stress

## Trust model

- Codex output git-tracked (`docs/checkpoints/codex_outputs/`), audit log reviewable
- User can read before acting
- Codex runs `--sandbox danger-full-access` per project default; risk = trust the prompt content (we write the prompt, so reviewable)

## Versioning

- v1 (2026-05-12): Initial Mode A standalone skill, after user explicitly proposed cross-AI cross-audit pattern. Built atop existing `/stress` (Claude hostile reviewer) and existing codex CLI infrastructure (CLAUDE.md "Codex CLI on DGX" section).
