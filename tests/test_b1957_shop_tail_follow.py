"""B-1957 follow-on + B-1958 — the shop_b0 resume chain and its two broken links.

The 2026-08-04 shop_b0 fire aborted at cell 1 (dom) 320/435. Recovering it needs
two mutually exclusive settings in one chain (dom must resume with
RESET_BEFORE=0 per B-304; the other six are fresh and must reset), which
`queue_chain.sh` cannot express — its RESET_FLAG is one value for the whole
chain. So dom is resumed by hand and a follow-on cron launches the rest.

Two defects found while building that path, pinned here:

  B-1958 — `_cron_wa_shop_follow.sh` read its sentinel with a bare `stat -c %Y`
  on a SYMLINK. GNU stat does not follow symlinks without -L, so it returned the
  link's own mtime (stamped at chain launch, never updated) and the "sentinel
  updated?" branch — the P0-1-A fix that made sentinel the PRIMARY criterion —
  could never be true. The cron logged a tidy refusal every 10 minutes for 20h
  while being structurally incapable of ever firing.

  B-1959 — `_condition_complete` keys on `site|baseline|mode`, but shop_b0's
  cell 1 and cell 7 are byte-identical (`queue_baseline.sh B0 dom shopping`; the
  replicate arm that gives the site its stochastic noise floor). Under
  RESUME_MISSING=1 a completed dom would mark BOTH lines done and silently drop
  the replicate. Latent today only because fire_manifest.json has no shopping
  conditions. Pinned as a documented expectation so the tail builder is not
  "fixed" by routing it through the filter.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH = REPO_ROOT / "scripts/queues/queue_phase1_paper_grade.sh"
WA_FOLLOW = REPO_ROOT / "scripts/queues/_cron_wa_shop_follow.sh"
TAIL_FOLLOW = REPO_ROOT / "scripts/queues/_cron_shop_b0_tail_follow.sh"


def _bash_n(path: Path):
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, f"{path.name} syntax error: {r.stderr}"


class TestScriptsParse:
    @pytest.mark.parametrize("script", [ORCH, WA_FOLLOW, TAIL_FOLLOW])
    def test_bash_n(self, script):
        assert script.exists(), f"missing {script}"
        _bash_n(script)


class TestB1958SentinelDereference:
    """A bare `stat` on the sentinel symlink is the bug — it must stay dead."""

    def test_no_bare_stat_on_the_done_symlink(self):
        src = WA_FOLLOW.read_text(encoding="utf-8")
        # Any `stat` reading ${DONE} must pass -L. Comments are stripped so the
        # explanatory block describing the bug does not trip its own test.
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        offenders = [
            ln for ln in code.splitlines()
            if "${DONE}" in ln and "stat" in ln and "stat -L" not in ln
        ]
        assert not offenders, f"bare stat on the DONE symlink returned: {offenders}"

    def test_dereferencing_helper_is_used_for_both_reads(self):
        code = WA_FOLLOW.read_text(encoding="utf-8")
        assert "_done_mtime()" in code, "helper missing"
        assert "stat -L -c %Y" in code, "helper must dereference"
        # arm-time write and check-time read must both go through it, else the
        # two sides compare different quantities.
        assert code.count("$(_done_mtime)") >= 2, (
            "both the armed-file write and the NOW_MTIME read must use the helper"
        )

    def test_symlink_vs_target_mtime_actually_differ(self, tmp_path):
        """The premise: GNU stat really does not follow symlinks by default."""
        target = tmp_path / "real.done"
        target.write_text("rc=0\n")
        link = tmp_path / "latest.done"
        link.symlink_to(target.name)
        import os, time
        # Age the link, then touch the target so the two mtimes must differ.
        os.utime(link, (1, 1), follow_symlinks=False)
        time.sleep(0.01)
        target.write_text("rc=0 again\n")
        bare = subprocess.run(["stat", "-c", "%Y", str(link)],
                              capture_output=True, text=True).stdout.strip()
        deref = subprocess.run(["stat", "-L", "-c", "%Y", str(link)],
                               capture_output=True, text=True).stdout.strip()
        assert bare != deref, (
            "premise broken: this platform's stat follows symlinks by default, "
            "so the B-1958 diagnosis needs re-checking"
        )


class TestTailBuilder:
    """cells 2-7 = full chain minus the dom main arm, replicate arm intact."""

    def _run_builders(self, resume_missing="0"):
        harness = f"""
        log() {{ :; }}
        _condition_complete() {{ return 0; }}   # pretend EVERYTHING is done
        _resume_filter_done() {{
          if [[ "${{RESUME_MISSING:-0}}" != "1" ]]; then cat; return 0; fi
          local cmd
          while IFS= read -r cmd; do
            [[ -z "${{cmd// }}" ]] && continue
            if _condition_complete "${{cmd}}"; then log skip >&2; else echo "${{cmd}}"; fi
          done
        }}
        eval "$(sed -n '/^build_shop_b0_chain() {{/,/^}}/p' {ORCH})"
        eval "$(sed -n '/^build_shop_b0_tail_chain() {{/,/^}}/p' {ORCH})"
        echo "---FULL---"
        RESUME_MISSING={resume_missing} build_shop_b0_chain
        echo "---TAIL---"
        RESUME_MISSING={resume_missing} build_shop_b0_tail_chain
        """
        out = subprocess.run(["bash", "-c", harness],
                             capture_output=True, text=True).stdout
        full_s, tail_s = out.split("---TAIL---")
        full = [l for l in full_s.replace("---FULL---", "").strip().splitlines() if l.strip()]
        tail = [l for l in tail_s.strip().splitlines() if l.strip()]
        return full, tail

    def test_tail_is_full_minus_first_cell(self):
        full, tail = self._run_builders()
        assert len(full) == 7, f"shop_b0 should be 7 cells, got {full}"
        assert len(tail) == 6, f"tail should be 6 cells, got {tail}"
        assert tail == full[1:], "tail must be exactly cells 2-7 of the full chain"

    def test_replicate_arm_survives(self):
        """The last cell is a deliberate duplicate of the first (noise floor for
        §242/§293). Dropping the FIRST occurrence must keep the last."""
        full, tail = self._run_builders()
        assert full[0] == full[-1], (
            "premise: shop_b0 cell 1 and cell 7 are the same command "
            "(replicate arm). If this changed, revisit the tail builder."
        )
        assert tail[-1] == full[0], "replicate arm must remain in the tail chain"
        assert "dom" in tail[-1]

    def test_tail_supports_resume_but_never_drops_the_replicate(self):
        """B-1959 fix changed this deliberately: the DERIVATION still runs with
        RESUME_MISSING off (so `tail -n +2` always drops the dom main arm, not
        whatever the filter happened to leave first), but the derived cells then
        DO go through the filter — a 9.7-day chain that cannot resume would
        re-run everything after any interruption.

        The replicate arm stays exempt: once the dom MAIN arm is bound its key is
        indistinguishable from the replicate's, so any filter reads the replicate
        as already done."""
        full, tail = self._run_builders(resume_missing="1")
        assert full == [], "stub says everything is done, so the full chain filters to empty"
        assert tail == ["queue_baseline.sh B0 dom shopping"], (
            f"with everything complete, only the exempt replicate arm may remain, got {tail}"
        )


class TestB1959ReplicateKeyCollision:
    """Document the latent collision rather than let it be discovered in a fire."""

    def test_first_and_last_cell_share_a_manifest_key(self):
        code = ORCH.read_text(encoding="utf-8")
        body = re.search(r"^build_shop_b0_chain\(\) \{.*?^\}", code, re.S | re.M).group(0)
        cells = [l.strip() for l in body.splitlines()
                 if l.strip().startswith(("queue_baseline.sh", "queue_phantom"))]
        assert cells[0] == cells[-1], "replicate arm premise changed"

    def test_manifest_has_no_shopping_conditions_yet(self):
        """Why B-1959 is latent, not active. If this ever fails, the tail chain's
        assumptions need re-checking BEFORE the next shop resume."""
        mf = REPO_ROOT / "docs/checkpoints/pre_run/fire_manifest.json"
        conds = json.loads(mf.read_text(encoding="utf-8")).get("conditions", {})
        shopping = [k for k in conds if k.startswith("shopping|")]
        assert not shopping, (
            f"shopping entered the fire manifest ({shopping}) — RESUME_MISSING=1 "
            "would now mark BOTH dom cells done and silently drop the replicate arm "
            "(B-1959). Fix _condition_complete before relying on resume here."
        )


class TestTailFollowGuards:
    def test_launches_the_tail_label_not_the_full_chain(self):
        code = TAIL_FOLLOW.read_text(encoding="utf-8")
        assert "launch shop_b0_tail" in code
        assert "launch shop_b0\"" not in code and "launch shop_b0 " not in code, (
            "must never launch the full 7-cell chain — dom would be re-run from zero"
        )

    def test_orchestrator_accepts_the_label(self):
        code = ORCH.read_text(encoding="utf-8")
        assert "shop_b0_tail)" in code, "label case branch missing"
        assert "shop_b0_tail" in re.search(r"Unknown site filter[^\n]*", code).group(0), (
            "usage string not updated — operators would not discover the label"
        )

    def test_fires_only_on_complete(self):
        """aborted / partial / missing must all hold, including the nasty case
        where the episode count is satisfied but the condition aborted."""
        code = TAIL_FOLLOW.read_text(encoding="utf-8")
        assert 'if [ "${STATE}" != "complete" ]; then' in code
        # abort is checked BEFORE the count, so a 435-episode aborted condition
        # is still refused.
        py = re.search(r'if d\.get\("condition_aborted"\).*?print\(f"partial', code, re.S)
        assert py, "verdict block not found"
        assert py.group(0).index("condition_aborted") < py.group(0).index("eps >= want"), (
            "abort must be tested before the episode count"
        )

    def test_uses_bracket_trick_for_process_scan(self):
        """CLAUDE.md records three separate incidents of pgrep self-matching."""
        code = TAIL_FOLLOW.read_text(encoding="utf-8")
        assert "[r]un_experiment" in code, "must use the bracket trick"
        assert not re.search(r'pgrep -f ["\']run_experiment', code), "bare pgrep self-matches"

    def test_is_idempotent_and_fails_closed(self):
        """B-1964 UPDATED this test's contract on purpose.

        It used to assert `FIRED flag must be set before launching`, i.e. it
        pinned the very defect codex P1-1 flagged: writing the terminal flag
        before knowing whether the launch survived the gates turned every
        transient failure into a permanent self-lock. A test that encodes a bug's
        behaviour actively defends that bug when someone tries to fix it.

        The new contract: idempotent via FIRED, concurrency via flock, and FIRED
        written only after a live chain pid is confirmed."""
        code = TAIL_FOLLOW.read_text(encoding="utf-8")
        assert '[ -f "${FIRED}" ] && exit 0' in code, "must stay idempotent"
        assert "flock -n 9" in code, "concurrency must be guarded by flock, not flag ordering"
        launch_at = code.index("launching shop_b0_tail")
        after = code[launch_at:]
        assert "kill -0" in after[:after.index("${FIRED}")], (
            "FIRED must come AFTER confirming the chain is alive"
        )


class TestB1960EveryLabelRegisteredAtEveryGate:
    """A launch label must be named by EVERY gate that branches on SITE_FILTER.

    B-1960: `shop_b0` and `wa_shop_b0` were added to the launch dispatcher and to
    Gate 8, but never to Gate 7 — so they hit Gate 7's `*)` fallback and the gate
    verified the cls+red configs instead. "All chain configs exist" was answered
    about a chain nobody was launching, and the 2026-08-04 shop_b0 fire passed it
    that way. A fallback that silently inspects the wrong thing is worse than no
    gate, because it reports success.

    This test makes the registration matrix explicit: labels come from the usage
    string (the operator-facing contract) and every one must appear in both
    gates' case statements.
    """

    def _labels(self):
        code = ORCH.read_text(encoding="utf-8")
        usage = re.search(r"expected: ([a-z0-9_|]+)\)", code).group(1)
        return [l for l in usage.split("|") if l]

    def _case_patterns(self, marker):
        """Collect every pattern named in the case block following `marker`."""
        code = ORCH.read_text(encoding="utf-8")
        start = code.index(marker)
        case_start = code.index('case "$', start)
        block = code[case_start:code.index("esac", case_start)]
        pats = set()
        # Patterns may be quoted (Gate 8 uses `all|"")` to also catch the empty
        # filter), so allow quotes and strip them off each alternative.
        for m in re.finditer(r'^\s*([a-z0-9_|*"]+)\)', block, re.M):
            for p in m.group(1).split("|"):
                pats.add(p.strip('"'))
        return pats

    @staticmethod
    def _dispatcher_builder(code, label):
        """Builder that the LAUNCH dispatcher runs for `label`.

        The dispatcher's branch is `<label>)` alone on its line; Gate 7's branch
        for the same label is `<label>)  builders_to_check=...` on one line. Match
        only the former, or the search walks forward from Gate 7 and reports
        whatever `launch_chain` comes next (which is how this test first failed).
        """
        m = re.search(rf"^\s*{label}\)\s*$", code, re.M)
        assert m, f"no dispatcher branch for {label}"
        tail = code[m.end():]
        return re.search(r"launch_chain \"[a-z]+\" (\w+)", tail).group(1)

    def test_usage_lists_the_tail_label(self):
        assert "shop_b0_tail" in self._labels()

    def test_gate7_names_every_label(self):
        pats = self._case_patterns("Gate 7: All chain configs exist")
        missing = [l for l in self._labels() if l not in pats]
        assert not missing, (
            f"Gate 7 does not name {missing} — they fall through to the `*)` "
            "fallback and the gate verifies the WRONG chain's configs (B-1960)"
        )

    def test_gate8_names_every_label(self):
        pats = self._case_patterns("Gate 8: cross-fire quarantine")
        missing = [l for l in self._labels() if l not in pats]
        assert not missing, f"Gate 8 has no quarantine policy for {missing}"

    def test_gate7_checks_the_chain_being_launched(self):
        """Each label's Gate 7 builder must be the one its dispatcher launches."""
        code = ORCH.read_text(encoding="utf-8")
        start = code.index("Gate 7: All chain configs exist")
        g7 = code[start:code.index("esac", code.index('case "$', start))]
        for label in ("shop_b0", "shop_b0_tail", "wa_shop_b0"):
            m = re.search(rf"^\s*(?:[a-z0-9_|]*\|)?{label}(?:\|[a-z0-9_|]*)?\)\s*"
                          rf"builders_to_check=\"([^\"]+)\"", g7, re.M)
            assert m, f"Gate 7 has no explicit entry for {label}"
            dispatch = self._dispatcher_builder(code, label)
            assert dispatch in m.group(1), (
                f"{label}: Gate 7 checks {m.group(1)!r} but the dispatcher "
                f"launches {dispatch!r} — the gate is inspecting a different chain"
            )


class TestB1963ResetFloorAssignsWhatItPrints:
    """The Magento reset floor must clamp UP to the number it announces.

    codex Mode B P1-4: the branch printed "clamping to 6000s" and then assigned
    2400. §429 measured reindex alone at ~40 min (2400s), so the effective floor
    was shorter than a legitimate reset. Blast radius is shop_b0_tail: its first
    cell dies during reset while the follower has already written FIRED.
    """

    def test_floor_assigns_6000(self):
        lib = (REPO_ROOT / "scripts/queues/_lib_paper_grade_gates.sh").read_text(encoding="utf-8")
        i = lib.index("below the Magento rebuild floor")
        block = lib[i:i + 1200]
        m = re.search(r"_reset_timeout=(\d+)", block)
        assert m, "no assignment found after the clamp message"
        assert m.group(1) == "6000", (
            f"clamp announces 6000s but assigns {m.group(1)}s — the floor must not "
            "silently clamp DOWN below a measured legitimate reset"
        )

    def test_announced_and_assigned_agree(self):
        """Guard the general shape, not just this one number."""
        lib = (REPO_ROOT / "scripts/queues/_lib_paper_grade_gates.sh").read_text(encoding="utf-8")
        i = lib.index("below the Magento rebuild floor")
        block = lib[i:i + 1200]
        announced = re.search(r"clamping to (\d+)s", block)
        assigned = re.search(r"_reset_timeout=(\d+)", block)
        assert announced and assigned
        assert announced.group(1) == assigned.group(1), (
            f"message says {announced.group(1)}s, code assigns {assigned.group(1)}s"
        )


class TestB1959OneBindingRetiresOneLine:
    """A single manifest binding may retire ONE chain line, not every line that
    happens to parse to the same key.

    shop_b0 lists `queue_baseline.sh B0 dom shopping` twice on purpose (cell 7 is
    the replicate arm carrying the site's stochastic noise floor; §242/§293 hang
    on it). `_condition_complete` keys on site|baseline|mode, so a completed dom
    used to match BOTH lines and drop the replicate silently — under the
    appearance of a successful resume. The watchdog auto-binds the moment dom
    finishes (`_auto_bind_manifest`), so this was days away from firing, not
    hypothetical.
    """

    def _filter(self, lines, complete_pred="return 1"):
        harness = f"""
        log() {{ :; }}
        _condition_complete() {{ {complete_pred}; }}
        eval "$(sed -n '/^_resume_filter_done() {{/,/^}}/p' {ORCH})"
        printf '%s\\n' {' '.join(repr(l) for l in lines)} | RESUME_MISSING=1 _resume_filter_done
        """
        out = subprocess.run(["bash", "-c", harness], capture_output=True, text=True).stdout
        return [l for l in out.strip().splitlines() if l.strip()]

    DOM = "queue_baseline.sh B0 dom shopping"
    SOM = "queue_baseline.sh B0 som shopping"

    def test_duplicate_line_survives_a_single_binding(self):
        """The load-bearing assertion: dom bound → first line goes, replicate stays."""
        kept = self._filter([self.DOM, self.SOM, self.DOM],
                            complete_pred='case "$1" in *"dom shopping"*) return 0;; *) return 1;; esac')
        assert kept.count(self.DOM) == 1, f"replicate arm lost: {kept}"
        assert self.SOM in kept, "unbound line must be kept"

    def test_distinct_lines_behave_exactly_as_before(self):
        """cls/red chains have 18 distinct lines — their behaviour must not change."""
        kept = self._filter([self.DOM, self.SOM], complete_pred="return 0")
        assert kept == [], "all-distinct + all-complete must filter to empty"

    def test_no_filtering_when_resume_missing_is_off(self):
        harness = f"""
        log() {{ :; }}
        _condition_complete() {{ return 0; }}
        eval "$(sed -n '/^_resume_filter_done() {{/,/^}}/p' {ORCH})"
        printf '%s\\n' '{self.DOM}' '{self.SOM}' | _resume_filter_done
        """
        out = subprocess.run(["bash", "-c", harness], capture_output=True, text=True).stdout
        assert len([l for l in out.strip().splitlines() if l.strip()]) == 2


class TestB1964ThreeStateFollower:
    """"Should not fire right now" must not be written as "never fire again"."""

    def _src(self):
        return TAIL_FOLLOW.read_text(encoding="utf-8")

    def test_held_is_separate_from_fired(self):
        src = self._src()
        assert "HELD=" in src, "HELD state missing"
        # the not-complete branch must write HELD, never FIRED
        i = src.index("dom runner 已消失但数据未完成")
        branch = src[i - 600:i + 600]
        assert 'touch "${FIRED}"' not in branch, (
            "the data-incomplete branch must NOT write FIRED — that turns an "
            "operator-initiated stop into a permanent self-lock (observed "
            "2026-08-06 08:30)"
        )
        assert '"${HELD}"' in branch

    def test_held_clears_once_data_completes(self):
        src = self._src()
        assert 'rm -f "${HELD}"' in src, "HELD must clear when the run completes"

    def test_fired_written_only_after_chain_confirmed(self):
        """FIRED must follow a live chain pid, not merely a spawn."""
        src = self._src()
        launch_at = src.index("launching shop_b0_tail")
        after = src[launch_at:]
        fired_at = after.index("${FIRED}")
        assert "kill -0" in after[:fired_at], (
            "must verify the chain pid is alive before writing FIRED"
        )
        assert "chain_pid=" in src

    def test_failed_launch_stays_retryable(self):
        src = self._src()
        i = src.index("launch 未确认")
        tail = src[i:i + 500]
        assert 'touch "${FIRED}"' not in tail and '> "${FIRED}"' not in tail, (
            "an unconfirmed launch must remain retryable"
        )

    def test_concurrency_guarded_by_flock_not_by_flag_ordering(self):
        src = self._src()
        assert "flock -n 9" in src, "must use flock for the launch critical section"

    def test_running_chain_is_adopted(self):
        """Else a 10-day chain finishes and the next tick launches a second one."""
        src = self._src()
        assert "补记 LAUNCHED" in src
        assert "queue_chain\\.sh" in src or "queue_chain" in src


class TestB1965LockDocumentation:
    def test_no_rm_lock_advice(self):
        src = (REPO_ROOT / "scripts/queues/queue_chain.sh").read_text(encoding="utf-8")
        assert "to force-release" not in src, (
            "removing a flock'd lock file cannot release it and lets a second "
            "process lock a different inode — the exact collision it guards"
        )
        assert "lslocks" in src, "should point at the real diagnostic instead"


class TestCronIsVersionControlled:
    """The recovery must be reproducible from committed state (codex P2-2)."""

    def test_both_follow_crons_are_declared(self):
        ct = (REPO_ROOT / "scripts/maintenance/crontab.txt").read_text(encoding="utf-8")
        assert "_cron_shop_b0_tail_follow.sh" in ct
        assert "_cron_wa_shop_follow.sh" in ct
