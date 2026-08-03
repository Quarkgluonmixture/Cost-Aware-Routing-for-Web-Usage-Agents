#!/usr/bin/env python3
"""Paper-grade integrity check of the A100 Phase-1a fire (cron-invoked).

Runs ON the A100 (where the results live). Validates, for every paper-grade
condition_summary_v2.json on disk + the in-progress run:

  1. episodes == scored_task_count  (EXACT — B-1834; over/under = contamination)
  2. parse_error_rate <= 1%  AND  benchmark_noise_rate == 0
  3. som/vision conditions: artifact images present (PNG > 0)
     — the B-1828→B-1832→B-1835 regression class (deferred-save losing all images);
     this is THE check that the silent 4× image-loss would have tripped.
  4. B0 conditions: cost coverage present (cost_column_coverage_rate ~ 1)
  5. in-progress run health: episodes progressing, images landing (som/vision),
     no UnboundLocalError / deferred-save-failed flood in the runner log.

Prints a one-line VERDICT (consumed by paper_grade_check.sh → ntfy) + details.
Exit 0 = paper-grade clean; 1 = issue(s) found; 2 = usage/internal error.

Usage: .venv/bin/python3 scripts/maintenance/paper_grade_check.py [--min-date YYYYMMDD]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results/visualwebarena/phase1"
MANIFEST = REPO / "docs/checkpoints/pre_run/fire_manifest.json"
LOGS = REPO / "logs"
DATE_RE = re.compile(r"(\d{8})")
IMAGE_MODES = ("som", "vision")  # modes that MUST produce per-step artifact images
# 2026-06-11 (user): sub-1% parse errors are expected LLM output flakiness, not a
# data-integrity signal — at >0 the ISSUE list was 9/10 entries of 0.05-0.3% noise
# drowning the one real outlier (B2-dom @2%). 1% keeps that class visible.
PARSE_ERR_THRESHOLD = 0.01


# B-1933 (2026-08-03): WA scored-task counts. The manifest's `scored_task_count`
# covers only the three VWA sites because it is the Phase 1a fire lock, and Phase
# 1a is VWA-only — adding WA rows there would edit a pre-registration artifact to
# describe runs it does not bind. These live here instead, and they are not a
# fresh estimate: 173 / 176 / 104 are the post-N/A-exclusion counts adjudicated in
# B-1894, the same numbers `queue_chain.sh`'s SITE_EXPECTED_N asserts against at
# launch. Keep the two in sync; a disagreement means one of them drifted.
WA_SCORED_FALLBACK = {
    "wa_shopping": 173,
    "wa_shopping_admin": 176,
    "wa_reddit": 104,
}


def _site_of(run_name: str) -> str:
    """Resolve run_id → site key, longest-prefix first.

    B-1933: the previous version tested membership in ("classifieds", "reddit",
    "shopping") and returned the first hit, so every WA run resolved to its VWA
    namesake — `B0_dom_wa_shopping_<ts>` contains the substring "shopping" and
    came back as `shopping`, and `..._wa_shopping_admin_...` did too. The run was
    then checked against the VWA expectation (435 vs its real 173, or 205 vs 104
    for reddit), so a complete WA condition read as incomplete and a bound one
    raised a false BOUND-run-incomplete issue. `queue_chain.sh:476` already fixed
    this same substring collision at launch time (B-1894); this is the analysis
    side of it, which was missed.

    Order matters: `wa_shopping_admin` must be tested before `wa_shopping`,
    which must be tested before `shopping`.
    """
    for s in ("wa_shopping_admin", "wa_shopping", "wa_reddit",
              "classifieds", "reddit", "shopping"):
        if s in run_name:
            return s
    return "?"


def _expected_scored(site: str, scored: dict) -> int | None:
    """Expected scored-task count for a site, or None when genuinely unknown.

    B-1933: returns None rather than a 10**9 sentinel. The old sentinel made an
    unknown site indistinguishable from a hugely-incomplete one — `ep < 10**9` is
    always true, so an unrecognised run either vanished from the report (unbound)
    or produced a nonsense "incomplete (224 < scored 1000000000)" line. An
    unmeasured expectation must be reported as unmeasured, not silently coerced
    into a comparison that always fails.
    """
    if site in scored:
        return int(scored[site])
    if site in WA_SCORED_FALLBACK:
        return WA_SCORED_FALLBACK[site]
    return None


def _mode_of(cond_id: str) -> str:
    # phase1_<mode>_router_0  →  <mode>
    return cond_id.replace("phase1_", "").replace("_router_0", "")


def _img_count(cond_dir: Path) -> int:
    n = 0
    for pat in ("**/*_som.png", "**/screenshot.png"):
        n += len(glob.glob(str(cond_dir / pat), recursive=True))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-date", type=int, default=20260521,
                    help="only check runs at-or-after this YYYYMMDD as paper-grade "
                         "(20260521 = Fire-6 start; includes dom R9755 + 0522 re-fires)")
    args = ap.parse_args()

    try:
        _man = json.loads(MANIFEST.read_text())
        scored = _man.get("scored_task_count", {})
        bound_runs = {c["run_id"] for c in _man.get("conditions", {}).values() if "run_id" in c}
    except Exception as e:
        print(f"VERDICT: FAIL — cannot read manifest scored_task_count: {e}")
        return 1

    issues: list[str] = []
    completed_ok = 0

    if not RESULTS.is_dir():
        print("VERDICT: FAIL — results dir missing")
        return 1

    for summ in sorted(glob.glob(str(RESULTS / "B*/*/condition_summary_v2.json"))):
        run_name = Path(summ).parts[-3]
        if "smoke" in run_name or "_test" in run_name:
            continue
        m = DATE_RE.search(run_name)
        if not m or int(m.group(1)) < args.min_date:
            continue
        cond_id = Path(summ).parts[-2]
        site = _site_of(run_name)
        mode = _mode_of(cond_id)
        baseline = run_name.split("_")[0]
        tag = f"{baseline}-{mode}-{site}"
        try:
            s = json.loads(Path(summ).read_text())
        except Exception:
            issues.append(f"{tag}: condition_summary unreadable")
            continue
        ep = int(s.get("episodes", 0))
        exp = _expected_scored(site, scored)
        is_bound = run_name in bound_runs
        if exp is None:
            # B-1933: surface it instead of comparing against a sentinel. A run
            # whose site we cannot name is exactly the case where a silent pass
            # is most dangerous — nothing else in this loop would flag it.
            issues.append(
                f"{tag}: cannot determine expected scored count "
                f"(site={site!r} from run_name={run_name!r}) — episode-count check SKIPPED"
            )
            continue
        # 1. episode count vs scored
        if ep > exp:
            # over-complete = dedup-failure / double-run contamination (always an issue)
            issues.append(f"{tag}: OVER-COMPLETE episodes {ep} > scored {exp} (contamination)")
            continue
        if ep < exp:
            # partial. Only an issue if it's the manifest-AUTHORITATIVE run (corrupted
            # binding). An UNbound partial = a harmless aborted/dead re-fire attempt —
            # below scored so aggregators' scored-count gate already excludes it; the
            # manifest validator handles ghosts separately. Skip it here.
            if is_bound:
                issues.append(f"{tag}: BOUND run incomplete ({ep} < scored {exp})")
            continue
        # 2. parse-error / benchmark-noise
        per = float(s.get("parse_error_rate", 0) or 0)
        if per > PARSE_ERR_THRESHOLD:
            issues.append(f"{tag}: parse_error_rate={per:.2%} > {PARSE_ERR_THRESHOLD:.0%}")
        if float(s.get("benchmark_noise_rate", 0) or 0) > 0:
            issues.append(f"{tag}: benchmark_noise_rate={s.get('benchmark_noise_rate')}")
        # 3. image presence (B-1828/1832/1835 regression class)
        if mode in IMAGE_MODES:
            png = _img_count(Path(summ).parent)
            if png == 0:
                issues.append(f"{tag}: 0 artifact images (B-1832/1835 image-loss regression!)")
        # 4. B0 cost coverage
        if baseline == "B0":
            cov = s.get("cost_column_coverage_rate", s.get("cost_coverage_rate"))
            if cov is not None and float(cov) < 0.99:
                issues.append(f"{tag}: cost coverage {cov} < 0.99")
        completed_ok += 1

    # 5. in-progress run health (newest run dir lacking a condition_summary)
    inprog = None
    for d in sorted(RESULTS.glob("B*_*"), key=lambda p: p.name, reverse=True):
        if "smoke" in d.name or "_test" in d.name:
            continue
        m = DATE_RE.search(d.name)
        if not m or int(m.group(1)) < args.min_date:
            continue
        cond_dirs = [c for c in d.glob("phase1_*_router_0")]
        if cond_dirs and not (cond_dirs[0] / "condition_summary_v2.json").exists():
            inprog = (d, cond_dirs[0])
            break

    inprog_note = "none"
    if inprog:
        d, cdir = inprog
        mode = _mode_of(cdir.name)
        ep = len(glob.glob(str(cdir / "episodes" / "*summary*.json")))
        _rm = re.search(r"R\d+", d.name)
        rid = _rm.group(0) if _rm else "?"
        png = _img_count(cdir) if mode in IMAGE_MODES else -1
        # runner log error flood
        rlogs = sorted(glob.glob(str(LOGS / f"*{rid}*runner.log")), key=os.path.getmtime, reverse=True)
        err = 0
        if rlogs:
            txt = Path(rlogs[0]).read_text(errors="ignore")
            err = txt.count("UnboundLocalError") + txt.count("deferred image save failed")
        inprog_note = f"{rid} {mode} ep={ep} img={png} errflood={err}"
        if mode in IMAGE_MODES and ep >= 2 and png == 0:
            issues.append(f"in-progress {rid} {mode}: ep={ep} but 0 images (regression!)")
        if err > 0:
            issues.append(f"in-progress {rid}: {err} UnboundLocalError/deferred-save-failed (regression!)")

    # Verdict
    if issues:
        print(f"VERDICT: ISSUES={len(issues)} (completed_ok={completed_ok}, inprog={inprog_note})")
        for i in issues[:10]:
            print(f"  ISSUE: {i}")
        return 1
    print(f"VERDICT: OK completed_ok={completed_ok} inprog=[{inprog_note}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
