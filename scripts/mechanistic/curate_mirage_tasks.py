"""Curate mirage task candidates for Stage 2B scale-up.

Scans archived B1 phantom_som run, for each (task_id, step_idx) generates
{SoM-with-image, P-SoM-no-image} continuations, flags pairs where outputs
diverge specifically along mirage axis (source negation vs target affirmative
hallucination). Outputs human-reviewable candidate list.

Use case: Stage 2B Stage 1 pilot found 1/3 tasks (cls task 0) showed clean
mirage signal (source 'do not show any blue kayak' vs target 'show items
related to blue'). Other tasks were null because source/target diverged for
non-mirage reasons. Manual curation of 10-20 paper-grade clean mirage cases
required for cross-task aggregate paper §5 claim.

Usage:
    # Default: scan 234 cls tasks at step_002, output sorted by mirage_score
    python3 scripts/mechanistic/curate_mirage_tasks.py \
      --site classifieds \
      --archived-run-dir results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428

    # Sample subset for quick test:
    python3 scripts/mechanistic/curate_mirage_tasks.py --site classifieds --n-tasks 30

ETA: 234 tasks × ~10-15s per task pair ≈ 40-60 min on B1 4B local + GPU contention.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from p79.mechanistic.extract_hidden_states import HiddenStateExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("curate-mirage")

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_TO_CONFIG_DIR = {
    "classifieds": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_classifieds",
    "reddit": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_reddit",
    "shopping": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_shopping",
}

# Heuristic patterns for mirage axis detection
# Mirage signature: source (with image) sees ground truth absence → emits negation
#                   target (no image, hallucinates) → emits affirmative
NEGATION_PATTERNS = [
    r"\bdo[e]?s? ?not\b", r"\bdoesn'?t\b", r"\bdon'?t\b",
    r"\bno such\b", r"\bnot show", r"\bnot ?(yet)? ?(a|the)?(vailable|ppear)",
    r"\bcannot\b", r"\bcan'?t (find|see|locate)\b",
    r"\bnot found\b", r"\bnone\b", r"\bno (items?|results?|listings?|matches?)\b",
    r"\bunable to\b", r"\bnowhere\b", r"\bempty\b",
    r"\bno (\w+ )?(found|visible|available|displayed|listed)\b",
]

AFFIRMATIVE_PATTERNS = [
    r"\bshow(s|ing|n)?\b", r"\bsee[ns]?\b", r"\bfound\b",
    r"\bappears? to\b", r"\bresults? (show|display)\b",
    r"\b(item|listing)s? (show|appear|display)\b",
    r"\bdisplay(s|ing|ed)?\b", r"\bvisible\b", r"\blisted\b",
    r"\b(I (can|see))? on (the )?\w+ page\b",
]


def load_intents(site: str, n_tasks: int | None = None) -> list[tuple[int, str]]:
    config_dir = SITE_TO_CONFIG_DIR[site]
    json_files = sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem))
    if n_tasks is not None:
        json_files = json_files[:n_tasks]
    intents = []
    for jf in json_files:
        d = json.loads(jf.read_text())
        if d.get("intent"):
            intents.append((int(jf.stem), d["intent"]))
    return intents


def find_artifacts_dir(run_dir: Path) -> Path:
    for child in run_dir.iterdir():
        if child.is_dir() and (child / "artifacts").is_dir():
            return child / "artifacts"
    raise FileNotFoundError(f"No condition subdir with artifacts/ in {run_dir}")


def build_som_marks(obs_text: str, max_marks: int = 200) -> str:
    from p79.experiment.som import _extract_text_marks
    marks = _extract_text_marks(obs_text, max_marks=max_marks)
    if not marks:
        return "[SOM_MARKS]\n[/SOM_MARKS]"
    return "\n".join(["[SOM_MARKS]"] + [f"[id={m['id']}] {m['label']}" for m in marks] + ["[/SOM_MARKS]"])


def build_inputs(extractor: HiddenStateExtractor, intent: str, mode: str, obs_text: str, image_path):
    user_text = extractor._build_user_text(intent, mode, obs_text)
    content = []
    if image_path is not None:
        img = HiddenStateExtractor._load_resize_image(image_path)
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "user", "content": content}]
    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image_path is not None:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = extractor.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
    else:
        inputs = extractor.processor(text=[text], padding=True, return_tensors="pt")
    return {k: v.to(extractor.model.device) for k, v in inputs.items()}


def generate_continuation(extractor, inputs, max_new_tokens: int = 15) -> tuple[list[int], str]:
    """Greedy generate, return (token_ids, decoded_text)."""
    import torch
    with torch.no_grad():
        out = extractor.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
        )
    input_len = inputs["input_ids"].shape[1]
    tokens = out.sequences[0, input_len:].cpu().tolist()
    text = extractor.processor.tokenizer.decode(tokens, skip_special_tokens=True)
    return tokens, text


def count_pattern_matches(text: str, patterns: list[str]) -> int:
    return sum(bool(re.search(p, text, re.I)) for p in patterns)


def token_overlap_ratio(a: list[int], b: list[int]) -> float:
    n = min(len(a), len(b))
    return sum(int(a[i] == b[i]) for i in range(n)) / n if n else 0.0


def score_mirage_candidate(source_text: str, target_text: str, source_tokens: list[int], target_tokens: list[int]) -> dict:
    """Score how 'mirage-like' source-target divergence is.

    Strong mirage signal:
    - source has negation (sees ground truth absence)
    - target has affirmative (hallucinates)
    - low token overlap (real divergence not envelope only)
    """
    src_neg = count_pattern_matches(source_text, NEGATION_PATTERNS)
    src_aff = count_pattern_matches(source_text, AFFIRMATIVE_PATTERNS)
    tgt_neg = count_pattern_matches(target_text, NEGATION_PATTERNS)
    tgt_aff = count_pattern_matches(target_text, AFFIRMATIVE_PATTERNS)
    overlap = token_overlap_ratio(source_tokens, target_tokens)

    # mirage_score: positive = source-negative + target-affirmative pattern
    # negative = reverse pattern (also interesting but unusual)
    # 0 = neutral
    mirage_score = (src_neg - tgt_neg) + (tgt_aff - src_aff)

    # Divergence: low overlap means real content diff (not envelope only)
    # Baseline overlap from §111 cls observed ~0.47-0.60 from shared envelope
    # < 0.4 = strong divergence
    divergence = 1.0 - overlap

    # Composite: mirage_score scaled by divergence (so envelope-only matches don't pollute)
    composite = mirage_score * (1.0 + divergence)

    return {
        "src_neg": src_neg, "src_aff": src_aff,
        "tgt_neg": tgt_neg, "tgt_aff": tgt_aff,
        "token_overlap": overlap,
        "divergence": divergence,
        "mirage_score": mirage_score,
        "composite": composite,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="classifieds", choices=list(SITE_TO_CONFIG_DIR))
    p.add_argument("--n-tasks", type=int, default=None, help="Default None = all tasks (234 for cls)")
    p.add_argument("--step", type=int, default=2, help="Step idx to sample observation from")
    p.add_argument("--max-new-tokens", type=int, default=15)
    p.add_argument("--archived-run-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--source-mode", default="som")
    p.add_argument("--target-mode", default="phantom_som")
    p.add_argument("--min-free-vram-gb", type=float, default=0.0)
    p.add_argument(
        "--artifacts-subdir", default=None,
        help="Override condition subdir name. For multi-mode archived runs "
             "(e.g. B1_3mode_reddit_20260413 has phase1_{dom,som,vision}_router_0), "
             "find_artifacts_dir picks first-iterated which may be wrong condition. "
             "Set explicitly: e.g. --artifacts-subdir phase1_som_router_0.",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / f"results/mechanistic/curate_mirage_b1_{args.site}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    intents = load_intents(args.site, args.n_tasks)
    logger.info(f"Loaded {len(intents)} task intents")

    archived_dir = Path(args.archived_run_dir)
    if args.artifacts_subdir:
        artifacts_dir = archived_dir / args.artifacts_subdir / "artifacts"
        if not artifacts_dir.is_dir():
            raise FileNotFoundError(f"--artifacts-subdir resolved to {artifacts_dir} which doesn't exist")
    else:
        artifacts_dir = find_artifacts_dir(archived_dir)
    logger.info(f"Archived artifacts: {artifacts_dir}")

    extractor = HiddenStateExtractor(model_path=args.model_path, min_free_vram_gb=args.min_free_vram_gb)
    logger.info(f"Model loaded")

    candidates = []
    skipped_no_artifact = 0

    for i, (task_id, intent) in enumerate(intents):
        step_dir = artifacts_dir / f"{args.site}_task_{task_id}" / f"step_{args.step:03d}"
        obs_file = step_dir / "observation_dom.txt"
        screenshot_annotated = step_dir / "screenshot_annotated.png"
        if not obs_file.exists() or not screenshot_annotated.exists():
            skipped_no_artifact += 1
            continue

        obs_text = obs_file.read_text()
        som_marks_text = build_som_marks(obs_text)

        try:
            source_inputs = build_inputs(extractor, intent, args.source_mode, som_marks_text, str(screenshot_annotated))
            source_tokens, source_text = generate_continuation(extractor, source_inputs, args.max_new_tokens)

            target_inputs = build_inputs(extractor, intent, args.target_mode, som_marks_text, None)
            target_tokens, target_text = generate_continuation(extractor, target_inputs, args.max_new_tokens)
        except Exception as e:
            logger.warning(f"task {task_id}: forward pass error: {e}")
            continue

        scores = score_mirage_candidate(source_text, target_text, source_tokens, target_tokens)
        candidates.append({
            "task_id": task_id,
            "step_idx": args.step,
            "intent": intent,
            "source_text": source_text,
            "target_text": target_text,
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            **scores,
        })

        if (i + 1) % 10 == 0:
            logger.info(
                f"[{i + 1}/{len(intents)}] task {task_id}: "
                f"composite={scores['composite']:+.2f} (mirage={scores['mirage_score']:+}, overlap={scores['token_overlap']:.2f})"
            )

        # Incremental save (recoverable)
        with (out_dir / "candidates.jsonl").open("w") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")

    if not candidates:
        logger.error("No candidates collected; aborting")
        return

    logger.info(f"Scored {len(candidates)} task pairs (skipped {skipped_no_artifact} missing artifacts)")

    # Sort by composite score (descending = most mirage-like first)
    candidates.sort(key=lambda c: c["composite"], reverse=True)

    # Human-readable markdown
    md_lines = [
        f"# Mirage Task Candidates — {args.site} N={len(candidates)}",
        "",
        f"**Setup**: source `{args.source_mode}` (with image) vs target `{args.target_mode}` (no image), "
        f"step {args.step:03d}, max_new_tokens={args.max_new_tokens}.",
        "",
        "**Composite score**: `(src_neg - tgt_neg) + (tgt_aff - src_aff)` × `(1 + divergence)`. "
        "Higher = stronger mirage axis (source 看到 ground-truth 无 → 否定 / target 幻觉 → 肯定).",
        "",
        "**Token overlap baseline ~0.47-0.60** (from §111 envelope-only) — entries below 0.4 are real content divergence.",
        "",
        "**Manual review**: pick top 10-20 entries with composite ≥ 1.0 AND overlap < 0.5 AND obvious mirage qualitative read.",
        "",
        "## Top candidates (sorted by composite, descending)",
        "",
        "| rank | task_id | composite | mirage_score | overlap | source | target | intent |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rank, c in enumerate(candidates[:50], 1):  # top 50
        md_lines.append(
            f"| {rank} | {c['task_id']} | {c['composite']:+.2f} | "
            f"{c['mirage_score']:+} (sn{c['src_neg']}/sa{c['src_aff']} / tn{c['tgt_neg']}/ta{c['tgt_aff']}) | "
            f"{c['token_overlap']:.2f} | "
            f"`{c['source_text'][:60]!r}` | "
            f"`{c['target_text'][:60]!r}` | "
            f"{c['intent'][:60]} |"
        )
    md_lines.append("")
    md_lines.append("## Bottom candidates (potential reverse-direction interest)")
    md_lines.append("")
    md_lines.append("| rank | task_id | composite | mirage_score | overlap | source | target | intent |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for rank, c in enumerate(candidates[-15:], 1):
        md_lines.append(
            f"| -{rank} | {c['task_id']} | {c['composite']:+.2f} | "
            f"{c['mirage_score']:+} | {c['token_overlap']:.2f} | "
            f"`{c['source_text'][:60]!r}` | "
            f"`{c['target_text'][:60]!r}` | "
            f"{c['intent'][:60]} |"
        )

    md_lines.append("")
    md_lines.append("## Summary stats")
    md_lines.append("")
    md_lines.append(f"- N candidates: {len(candidates)}")
    md_lines.append(f"- skipped (missing artifacts): {skipped_no_artifact}")
    composite_strong = [c for c in candidates if c["composite"] >= 1.0 and c["token_overlap"] < 0.5]
    md_lines.append(f"- **Strong mirage candidates** (composite ≥ 1.0 AND overlap < 0.5): **{len(composite_strong)}**")
    md_lines.append(f"- composite distribution: min={candidates[-1]['composite']:+.2f}, "
                    f"median={candidates[len(candidates) // 2]['composite']:+.2f}, "
                    f"max={candidates[0]['composite']:+.2f}")

    (out_dir / "candidates.md").write_text("\n".join(md_lines))

    logger.info(
        f"Curation done: {len(candidates)} scored, {len(composite_strong)} strong mirage candidates. "
        f"Review {out_dir}/candidates.md"
    )


if __name__ == "__main__":
    main()
