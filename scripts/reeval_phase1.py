"""
Offline re-evaluation for phase1 tasks that failed with evaluator_error:'answer'.

The bug was: runner appended a trailing observation after the stop action, so
trajectory[-1] was an observation dict instead of the VWA stop Action, causing
StringEvaluator to raise KeyError('answer').

This script re-evaluates affected tasks using data already stored in the JSONL logs:
  - agent answer  → last step's action.answer
  - final URL     → last step's state_digest.url_after
  - reference     → task config file

Supports: string_match (exact_match / must_include / must_exclude / one_of / fuzzy_N/A)
          url_match (EXACT / GOLD in PRED)
Skips:    fuzzy_match (non-N/A) and ua_match — require LLM calls
          program_html / page_image_query — require live browser

Usage:
    python scripts/reeval_phase1.py [--dry-run] [--phase-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from nltk.tokenize import word_tokenize
except ImportError:
    def word_tokenize(s):  # type: ignore[misc]
        return s.split()

VWA_CONFIGS_ROOT = Path(__file__).parent.parent / "external/visualwebarena/config_files/vwa"

# ── helpers ──────────────────────────────────────────────────────────────────

def clean_answer(answer: str) -> str:
    a = str(answer)
    if a.startswith("'") and a.endswith("'"):
        a = a[1:-1]
    elif a.startswith('"') and a.endswith('"'):
        a = a[1:-1]
    return a.lower()


def exact_match(ref: str, pred: str) -> float:
    return float(clean_answer(pred) == clean_answer(ref))


def must_include(ref: str, pred: str) -> float:
    cr, cp = clean_answer(ref), clean_answer(pred)
    if len(word_tokenize(cr)) == 1:
        return float(cr in word_tokenize(cp))
    return float(cr in cp)


def must_exclude(ref: str, pred: str) -> float:
    cr, cp = clean_answer(ref), clean_answer(pred)
    if len(word_tokenize(cr)) == 1:
        return float(cr not in word_tokenize(cp))
    return float(cr not in cp)


def clean_url(url: str) -> str:
    url = str(url).replace("localhost", "127.0.0.1")
    if url.endswith("/"):
        url = url[:-1]
    return url


def eval_string_match(pred_answer: str, eval_cfg: dict) -> tuple[float, str | None]:
    """Returns (score, skip_reason). skip_reason is set if we can't eval offline."""
    ref_answers = eval_cfg.get("reference_answers", {})
    score = 1.0
    pred = clean_answer(pred_answer)
    for approach, value in ref_answers.items():
        if approach == "exact_match":
            score *= exact_match(ref=value, pred=pred)
        elif approach == "must_include":
            assert isinstance(value, list)
            for must_val in value:
                options = must_val.split(" |OR| ")
                score *= float(any(must_include(ref=v, pred=pred) for v in options))
        elif approach == "must_exclude":
            assert isinstance(value, list)
            for excl_val in value:
                score *= must_exclude(ref=excl_val, pred=pred)
        elif approach == "one_of":
            assert isinstance(value, list)
            found = any(clean_answer(v) in pred for v in value)
            score *= float(found)
        elif approach == "fuzzy_match":
            if value == "N/A":
                # exact match for N/A tasks (no LLM needed)
                score *= exact_match(ref=value, pred=pred)
            else:
                return 0.0, f"fuzzy_match_requires_llm"
        elif approach == "required_values":
            # numeric — best effort
            try:
                import re
                nums = re.findall(r"-?\d+(?:\.\d+)?", pred)
                pred_num = int(float(nums[0])) if nums else None
                if pred_num is None:
                    score = 0.0
                else:
                    assert isinstance(value, list)
                    for v in value:
                        options = v.split(" |OR| ")
                        ok = any(_compare_numeric(pred_num, opt) for opt in options)
                        score *= float(ok)
            except Exception:
                return 0.0, "required_values_parse_error"
        else:
            return 0.0, f"unknown_approach:{approach}"
    return score, None


def _compare_numeric(pred: int, expr: str) -> bool:
    expr = expr.strip()
    import re
    m = re.match(r"([<>]=?|==|=)\s*(-?\d+(?:\.\d+)?)", expr)
    if not m:
        return False
    op, val = m.group(1), float(m.group(2))
    ops = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
           ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
           "=": lambda a, b: a == b, "==": lambda a, b: a == b}
    return ops.get(op, lambda a, b: False)(pred, val)


def eval_url_match(final_url: str | None, eval_cfg: dict) -> tuple[float, str | None]:
    if not final_url:
        return 0.0, "no_url_in_logs"
    ref_raw = eval_cfg.get("reference_url", "")
    if not ref_raw:
        return 1.0, None  # no reference URL → pass
    ref_urls = [clean_url(u) for u in ref_raw.split(" |OR| ")]
    pred = clean_url(final_url)
    rule = eval_cfg.get("url_note", "EXACT")
    if rule == "EXACT":
        return float(pred in ref_urls), None
    elif rule == "GOLD in PRED":
        return float(any(r in pred for r in ref_urls)), None
    return 0.0, f"unknown_url_note:{rule}"


# ── task config loader ────────────────────────────────────────────────────────

_site_configs: dict[str, dict] = {}

def get_task_cfg(site: str, task_id: int) -> dict | None:
    if site not in _site_configs:
        p = VWA_CONFIGS_ROOT / f"test_{site}.json"
        if not p.exists():
            return None
        tasks = json.loads(p.read_text())
        _site_configs[site] = {t["task_id"]: t for t in tasks}
    return _site_configs[site].get(task_id)


# ── per-summary re-evaluation ─────────────────────────────────────────────────

def reeval_summary(summary_path: Path, dry_run: bool) -> dict:
    """Returns a result dict describing what happened."""
    summary = json.loads(summary_path.read_text())

    if not (summary.get("error") or "").startswith("evaluator_error"):
        return {"path": str(summary_path), "status": "skipped_no_error"}

    site = summary.get("benchmark_site", "")
    task_id = summary.get("task_id")
    task_cfg = get_task_cfg(site, task_id)
    if task_cfg is None:
        return {"path": str(summary_path), "status": "skipped_no_config"}

    eval_types = task_cfg["eval"].get("eval_types", [])
    unsupported = [t for t in eval_types if t not in ("string_match", "url_match")]
    if unsupported:
        return {"path": str(summary_path), "status": f"skipped_unsupported:{unsupported}"}

    # Load steps to extract answer + final URL
    steps_path = summary_path.with_name(
        summary_path.name.replace("summary_v2.json", "steps_v2.jsonl")
    )
    if not steps_path.exists():
        return {"path": str(summary_path), "status": "skipped_no_steps_file"}

    steps = [json.loads(l) for l in steps_path.read_text().splitlines() if l.strip()]
    if not steps:
        return {"path": str(summary_path), "status": "skipped_empty_steps"}

    last_step = steps[-1]
    action = last_step.get("action", {})
    pred_answer = str(action.get("answer", ""))
    state_digest = last_step.get("state_digest") or {}
    final_url = state_digest.get("url_after") or state_digest.get("url_before")

    # Compute score
    combined_score = 1.0
    skip_reason = None

    for eval_type in eval_types:
        if eval_type == "string_match":
            s, reason = eval_string_match(pred_answer, task_cfg["eval"])
            combined_score *= s
            if reason:
                skip_reason = reason
                break
        elif eval_type == "url_match":
            s, reason = eval_url_match(final_url, task_cfg["eval"])
            combined_score *= s
            if reason:
                skip_reason = reason
                break

    if skip_reason:
        return {
            "path": str(summary_path), "status": f"skipped:{skip_reason}",
            "task_id": task_id, "site": site,
        }

    old_score = summary.get("score", 0.0)
    new_score = combined_score
    changed = abs(new_score - old_score) > 1e-6

    result = {
        "path": str(summary_path),
        "status": "reevaluated",
        "task_id": task_id,
        "site": site,
        "old_score": old_score,
        "new_score": new_score,
        "changed": changed,
        "pred_answer": pred_answer,
    }

    if not dry_run:
        # Backup original
        backup = summary_path.with_suffix(".json.bak")
        if not backup.exists():
            shutil.copy2(summary_path, backup)
        # Update summary
        summary["score"] = new_score
        summary["success"] = new_score >= 1.0
        summary["error"] = None
        summary["offline_reevaluated"] = True
        summary["offline_reeval_answer"] = pred_answer
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Offline re-evaluation for phase1 evaluator_error tasks")
    parser.add_argument("--phase-dir", default=None, help="Path to phase dir (default: results/visualwebarena/phase1)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    phase_dir = Path(args.phase_dir) if args.phase_dir else base / "results/visualwebarena/phase1"

    summaries = sorted(phase_dir.glob("**/episodes/*summary_v2.json"))
    print(f"Found {len(summaries)} summary files in {phase_dir}")

    results = []
    for s in summaries:
        r = reeval_summary(s, dry_run=args.dry_run)
        results.append(r)

    # Print summary
    from collections import Counter
    statuses = Counter(r["status"] for r in results)
    reevaluated = [r for r in results if r["status"] == "reevaluated"]
    changed = [r for r in reevaluated if r.get("changed")]
    newly_correct = [r for r in changed if r["new_score"] >= 1.0 and r["old_score"] < 1.0]

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Results:")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")
    print(f"\nRe-evaluated: {len(reevaluated)}")
    print(f"Score changed: {len(changed)}")
    print(f"Newly correct (0→1): {len(newly_correct)}")

    if newly_correct:
        print("\nNewly correct tasks:")
        for r in sorted(newly_correct, key=lambda x: (x["site"], x["task_id"])):
            print(f"  [{r['site']}] task {r['task_id']}: answer='{r['pred_answer']}'")

    if args.dry_run:
        print("\n(dry run — no files modified)")
    else:
        print(f"\nSummary files updated. Backups saved as *.json.bak")


if __name__ == "__main__":
    main()
