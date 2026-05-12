# Process artifacts (git-tracked replicas of `.claude/` config)

`.claude/` is gitignored by user preference (settings.local.json contains per-machine config). This directory holds **git-tracked replicas** of the parts of `.claude/` that are project-content (not personal settings), so they survive cross-machine setup and are versioned alongside paper drafts.

## Files

### `stress_skill_replica.md`
Replica of `.claude/skills/stress/SKILL.md` — the **hostile reviewer audit** skill (`/stress` slash command). Provides a top-tier conference reviewer persona that reads paper drafts + evidence and writes brutal-but-fair criticism, with distance-to-top-tier framing.

**Auto-trigger rule** (in `.claude/CLAUDE.md` "Hostile reviewer audit" section): MUST run `/stress` before milestone declaration, paper prose commits, paper push, codex prose rounds, advisor sync, interview prep, or ultrareview.

**To restore on new machine**:
```bash
mkdir -p .claude/skills/stress
cp docs/checkpoints/process/stress_skill_replica.md .claude/skills/stress/SKILL.md
```

The auto-trigger rule lives in `.claude/CLAUDE.md` "Hostile reviewer audit (`/stress`)" section, which would need to be re-added if `.claude/CLAUDE.md` is reset.

## Versioning

- v1 (2026-05-12 early): PRA-10 checklist runner — deprecated for being too mechanical
- v2 (2026-05-12 late): hostile reviewer persona (current) — captures the value of adversarial reading rather than checkbox-ticking. Reframed after user feedback that genuine reviewer mode catches gaps a checklist misses.
