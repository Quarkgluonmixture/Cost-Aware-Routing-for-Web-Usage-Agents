"""Shared VL agent utilities — cross-baseline contract source of truth.

B-146 (/stress A1.2 v8 codex B4, 2026-05-16): hosts the prompt templates,
history formatting, confidence computation, and VRAM helper that Qwen3VLAgent,
Gemma3VLAgent, and ProxyApiAgent all consume. Previously Gemma3VLAgent
imported these via Qwen3VLAgent classmethods, which transitively pulled in
``transformers.Qwen3VLForConditionalGeneration`` and ``qwen_vl_utils`` —
Gemma runs failed at first launch in environments without Qwen deps.

This module deliberately has minimal dependencies (``torch`` only, for VRAM
inspection and confidence math). All prompt strings and history formatting
work without any torch/transformers/qwen_vl_utils imports.

Single-source contract: cross-baseline byte-identical prompts + identical
confidence schema are paper-grade reproducibility requirements; future
baselines (B3/B4) MUST consume from here.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# VRAM gating helper
# ----------------------------------------------------------------------


def wait_for_vram(min_free_gb: float, poll_interval: int = 30, timeout: int = 0) -> None:
    """Block until at least *min_free_gb* GPU memory is available.

    Args:
        min_free_gb: Minimum free VRAM in GB before proceeding.
        poll_interval: Seconds between checks.
        timeout: Max seconds to wait (0 = unlimited).
    """
    if not torch.cuda.is_available():
        return
    start = time.time()
    while True:
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        if free_gb >= min_free_gb:
            logger.info(
                "VRAM check passed: %.1f GB free / %.1f GB total (need %.1f GB)",
                free_gb, total_gb, min_free_gb,
            )
            return
        elapsed = time.time() - start
        if timeout > 0 and elapsed >= timeout:
            raise RuntimeError(
                f"VRAM wait timeout after {elapsed:.0f}s: "
                f"{free_gb:.1f} GB free < {min_free_gb:.1f} GB required"
            )
        logger.warning(
            "Waiting for VRAM: %.1f GB free / %.1f GB total (need %.1f GB). "
            "Retrying in %ds... (elapsed %.0fs)",
            free_gb, total_gb, min_free_gb, poll_interval, elapsed,
        )
        time.sleep(poll_interval)


# ----------------------------------------------------------------------
# System prompts (must remain byte-identical across baselines)
# ----------------------------------------------------------------------


def make_dom_prompt() -> str:
    return """You are a precise web navigation agent.
Output ONLY valid JSON. No markdown blocks, no explanations.

You receive the full Accessibility Tree of the current page.
Use element IDs from the Accessibility Tree to interact with elements.

Core Rules:
1) Do NOT answer or finish immediately. You MUST navigate to find the item.
2) You are logged in as a user. For tasks involving your own content (e.g., "my listing", "my post", "my message"),
   navigate to account/profile sections instead of searching publicly.
3) If the target category is not visible, look for a parent category or use the search bar.
4) NEVER give up early. If you don't see the item, SEARCH for it using the search bar.
5) Only use "finish" when you have successfully completed the task or after EXHAUSTIVE search.
6) For single-item tasks (find and navigate to ONE specific item/page), you MUST open that item's detail page before "finish".
   For collection tasks (return links/info for MULTIPLE items), you MAY "finish" from a list/search page
   after recording the required items in your answer.
7) If you are on the homepage, DO NOT go back. Start by searching or clicking a category.
8) If you are stuck, use scroll or try a different category/search.

Response Format (JSON):
{
  "thought": "Brief reasoning about what to do next.",
  "confidence": 0.0 to 1.0,
  "action_type": "click" | "type" | "select_option" | "scroll" | "wait" | "back" | "forward" | "finish" | "tab_focus",
  ... (other action parameters) ...
}
"confidence": your self-assessed probability (0.0–1.0) that this action makes meaningful progress toward the task goal.

Action Schema:
1. Click: {"action_type": "click", "element_id": N}
   - N is the numeric ID from the Accessibility Tree (e.g., [175] link 'Comments' -> element_id: 175).
   - ALWAYS prefer element_id. Only use coordinate as last resort.
2. Type: {"action_type": "type", "text": "string", "element_id": N}
   - This action automatically clicks the target to focus it, then types the text.
   - ALWAYS use "type" (not "click") when you want to enter text into an input field.
   - "click" is for buttons, links, and navigation only — it cannot enter text.
   - ALWAYS specify element_id to target the correct input field.
   - To submit, append "\\n" to the text.
2.5. Select Option: {"action_type": "select_option", "element_id": N, "option_label": "Option Name"}
   - Use ONLY for <select> dropdown elements (shown as "combobox" in the Accessibility Tree).
   - Clicking a combobox does NOT open the dropdown. Use select_option instead.
   - option_label must match the visible option text exactly.
3. Scroll: {"action_type": "scroll", "delta": [dx, dy], "coordinate_type": "normalized"}
   - dy>0 scrolls DOWN, dy<0 scrolls UP. Use scroll up when the target is above the current view.
4. Wait: {"action_type": "wait"}
5. Back: {"action_type": "back"} — WARNING: Do NOT use on the first/homepage.
6. Forward: {"action_type": "forward"}
7. Finish: {"action_type": "finish", "answer": "optional string"}
8. Tab focus: {"action_type": "tab_focus", "page_number": int}

Tab Rule:
- Multi-site tasks may open multiple tabs (different websites).
- If the target website is in another tab, switch with {"action_type":"tab_focus","page_number":N} BEFORE clicking.
- Element IDs are page-local to the current tab. Do NOT reuse IDs from another tab/site.
- Do NOT try to find a cross-site navigation link on the current page when the site is already in another tab.

CRITICAL:
- You MUST include a "thought" field.
- ALWAYS use element_id for click and type. Do NOT guess coordinates.
- Do NOT output literal newlines inside JSON strings. Use \\n.
- Avoid repeating the same action. Change strategy if stuck.
"""


def make_som_prompt() -> str:
    return """You are a precise web navigation agent.
Output ONLY valid JSON. No markdown blocks, no explanations.

You receive:
  1. A [SOM_MARKS] list: flat index of interactive elements, each with [id=N] and a short description.
  2. A screenshot — normally with bounding boxes labeled by ID, matching the [SOM_MARKS] list.

Note: If [SOM_MARKS] is empty (no elements detected), no bounding boxes will appear in the screenshot.
In that case, fall back to coordinate-based interaction using what you can see in the screenshot.

Use the element IDs from [SOM_MARKS] to interact. Use the screenshot to understand spatial layout and locate elements not in the list.

Core Rules:
1) Do NOT answer or finish immediately. You MUST navigate to find the item.
2) You are logged in as a user. For tasks involving your own content (e.g., "my listing", "my post", "my message"),
   navigate to account/profile sections instead of searching publicly.
3) Prefer element_id for clicks and typing. Use coordinate only when the target is visible in the image but has no ID in [SOM_MARKS].
4) NEVER give up early. If you don't see the item, SEARCH for it using the search bar.
5) Only use "finish" when you have successfully completed the task or after EXHAUSTIVE search.
6) For single-item tasks (find and navigate to ONE specific item/page), you MUST open that item's detail page before "finish".
   For collection tasks (return links/info for MULTIPLE items), you MAY "finish" from a list/search page
   after recording the required items in your answer.
7) If you are on the homepage, DO NOT go back. Start by searching or clicking a category.
8) If you are stuck, scroll or try a different approach.

Response Format (JSON):
{
  "thought": "Brief reasoning about what to do next.",
  "confidence": 0.0 to 1.0,
  "action_type": "click" | "type" | "select_option" | "scroll" | "wait" | "back" | "forward" | "finish" | "tab_focus",
  ... (other action parameters) ...
}
"confidence": your self-assessed probability (0.0–1.0) that this action makes meaningful progress toward the task goal.

Action Schema:
1. Click by element_id (preferred): {"action_type": "click", "element_id": N}
   - N is from [SOM_MARKS], e.g. [id=175] link 'Comments' -> element_id: 175.
2. Click by coordinate (fallback): {"action_type": "click", "coordinate": [x, y], "coordinate_type": "normalized"}
   - x, y are floats 0.0–1.0. Use only when no element_id is available.
3. Type: {"action_type": "type", "text": "string", "element_id": N}
   - This action automatically clicks the target to focus it, then types the text.
   - ALWAYS use "type" (not "click") when you want to enter text into an input field.
   - "click" is for buttons, links, and navigation only — it cannot enter text.
   - Prefer element_id. To submit, append "\\n" to the text.
3.5. Select Option: {"action_type": "select_option", "element_id": N, "option_label": "Option Name"}
   - Use ONLY for <select> dropdown elements (shown as "combobox" in the SOM_MARKS list).
   - Clicking a combobox does NOT open the dropdown. Use select_option instead.
   - option_label must match the visible option text exactly.
4. Scroll: {"action_type": "scroll", "delta": [dx, dy], "coordinate_type": "normalized"}
   - dy>0 scrolls DOWN, dy<0 scrolls UP. Use scroll up when the target is above the current view.
5. Wait: {"action_type": "wait"}
6. Back: {"action_type": "back"} — WARNING: Do NOT use on the first/homepage.
7. Forward: {"action_type": "forward"}
8. Finish: {"action_type": "finish", "answer": "optional string"}
9. Tab focus: {"action_type": "tab_focus", "page_number": int}

Tab Rule:
- Multi-site tasks may open multiple tabs (different websites).
- If the target website is in another tab, switch with {"action_type":"tab_focus","page_number":N} BEFORE clicking.
- Element IDs are page-local to the current tab. Do NOT reuse IDs from another tab/site.
- Do NOT try to find a cross-site navigation link on the current page when the site is already in another tab.

CRITICAL:
- You MUST include a "thought" field.
- Prefer element_id over coordinate when the element appears in [SOM_MARKS].
- Do NOT output literal newlines inside JSON strings. Use \\n.
- Avoid repeating the same action. Change strategy if stuck.
"""


# ----------------------------------------------------------------------
# Canonical mode → prompt dispatch table (single source of truth)
# ----------------------------------------------------------------------


def build_mode_prompt_dispatch_table() -> Dict[str, str]:
    """Single source of truth for the 7-key mode → system-prompt dispatch.

    B-451 (/stress A1.4 P0-5-A* OOB, 2026-05-17): pre-fix each of B0
    (`proxy_api_agent.py::_get_system_prompts`), B1 (`qwen3vl_agent.py::
    __init__._system_prompts`), B2 (`gemma3vl_agent.py::__init__._system_prompts`),
    and the mechanistic extractor (`p79/mechanistic/extract_hidden_states.py::
    __init__._mode_to_prompt`) hand-rolled the same 7-key dict locally. Four
    copies of the same string mapping = four silent-drift surfaces. If the
    agent dispatch grew a new key (e.g. a future `phantom_axtree_random_perm`
    paper-2 arm), the extractor's `.get(mode, dom_prompt)` would silently
    fall back — NPZ extraction prompt diverges from production agent prompt
    without any error surface. B-103 (the `Accessibility Tree:\\n` prefix
    missing fix) was exactly this class of drift, caught only after
    mechanistic NPZ data had already been generated. This function makes
    the dispatch byte-identical at the function-call boundary; consumers
    that need a customized variant should override AFTER the canonical
    base call rather than re-listing the keys.

    Mode semantics:
      - "dom":             DOM prompt + AXTree text + no image
      - "som":             SoM prompt + [SOM_MARKS] text + marked screenshot
      - "phantom_som":     SoM prompt + [SOM_MARKS] text + NO image (P-SoM)
      - "phantom_dom":     DOM prompt + [SOM_MARKS] text + NO image (legacy
                           alias of phantom_text; archive run dirs still use)
      - "phantom_text":    DOM prompt + [SOM_MARKS] text + NO image (P-text,
                           current canonical name)
      - "phantom_prompt":  SoM prompt + AXTree text + NO image (P-prompt,
                           symmetric prompt-axis swap from DOM)
      - "vision":          Vision prompt + empty text + raw screenshot
    """
    dom_prompt = make_dom_prompt()
    som_prompt = make_som_prompt()
    vision_prompt = make_vision_prompt()
    return {
        "dom": dom_prompt,
        "som": som_prompt,
        "phantom_som": som_prompt,     # P-SoM: SoM prompt + [SOM_MARKS] text + no image
        "phantom_dom": dom_prompt,     # P-text (legacy alias)
        "phantom_text": dom_prompt,    # P-text (current canonical name)
        "phantom_prompt": som_prompt,  # P-prompt: SoM prompt + AXTree text + no image
        "vision": vision_prompt,
    }


def make_vision_prompt() -> str:
    return """You are a precise web navigation agent.
Output ONLY valid JSON. No markdown blocks, no explanations.

You receive only a raw screenshot of the current page. No element IDs are available.
Use normalized coordinates (x, y as floats 0.0–1.0, origin top-left) to interact.

Core Rules:
1) Do NOT answer or finish immediately. You MUST navigate to find the item.
2) You are logged in as a user. For tasks involving your own content (e.g., "my listing", "my post", "my message"),
   navigate to account/profile sections instead of searching publicly.
3) Use coordinates to click visible elements. Estimate the center of the target element.
4) NEVER give up early. Scroll to find content not visible, then search if needed.
5) Only use "finish" when you have successfully completed the task or after EXHAUSTIVE search.
6) For single-item tasks (find and navigate to ONE specific item/page), you MUST open that item's detail page before "finish".
   For collection tasks (return links/info for MULTIPLE items), you MAY "finish" from a list/search page
   after recording the required items in your answer.
7) If you are on the homepage, DO NOT go back. Start by searching or clicking a category.
8) If you are stuck, scroll or try a different approach.

Response Format (JSON):
{
  "thought": "Brief reasoning about what to do next.",
  "confidence": 0.0 to 1.0,
  "action_type": "click" | "type" | "select_option" | "scroll" | "wait" | "back" | "forward" | "finish" | "tab_focus",
  ... (other action parameters) ...
}
"confidence": your self-assessed probability (0.0–1.0) that this action makes meaningful progress toward the task goal.

Action Schema:
1. Click: {"action_type": "click", "coordinate": [x, y], "coordinate_type": "normalized"}
   - x, y are floats 0.0–1.0. Estimate the center of the target element in the screenshot.
2. Type: {"action_type": "type", "text": "string", "coordinate": [x, y], "coordinate_type": "normalized"}
   - This action automatically clicks the target coordinate to focus it, then types the text.
   - ALWAYS use "type" (not "click") when you want to enter text into an input field.
   - "click" is for buttons, links, and navigation only — it cannot enter text.
   - Include coordinate to specify the input field location. To submit, append "\\n" to the text.
2.5. Select Option: {"action_type": "select_option", "coordinate": [x, y], "option_label": "Option Name"}
   - Use ONLY for <select> dropdown visible in the screenshot.
   - Clicking a dropdown does NOT open it. Use select_option to set the value directly.
   - option_label must match the visible option text exactly.
3. Scroll: {"action_type": "scroll", "delta": [dx, dy], "coordinate_type": "normalized"}
   - dy>0 scrolls DOWN, dy<0 scrolls UP. Use scroll up when the target is above the current view.
4. Wait: {"action_type": "wait"}
5. Back: {"action_type": "back"} — WARNING: Do NOT use on the first/homepage.
6. Forward: {"action_type": "forward"}
7. Finish: {"action_type": "finish", "answer": "optional string"}
8. Tab focus: {"action_type": "tab_focus", "page_number": int}

Tab Rule:
- Multi-site tasks may open multiple tabs (different websites).
- If the target website is in another tab, switch with {"action_type":"tab_focus","page_number":N} BEFORE clicking.
- Do NOT try to find a cross-site navigation link on the current page when the site is already in another tab.

CRITICAL:
- You MUST include a "thought" field.
- DO NOT use element_id — there are no element IDs in this mode.
- Do NOT output literal newlines inside JSON strings. Use \\n.
- Avoid repeating the same action. Change strategy if stuck.
"""


# ----------------------------------------------------------------------
# History formatting (cross-baseline schema)
# ----------------------------------------------------------------------


def format_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = []
    for rec in history:
        act = rec.get("action", {})
        atype = act.get("action_type", "?")
        detail = ""
        if atype == "click":
            if "element_id" in act:
                detail = f" [id={act['element_id']}]"
            elif "coordinate" in act:
                detail = f" coord={act['coordinate']}"
            else:
                detail = " ?"
        elif atype == "type":
            detail = f' "{act.get("text", "")}"'
        elif atype == "scroll":
            detail = f' delta={act.get("delta", "?")}'
        success = rec.get("action_success", None)
        changed = rec.get("page_changed", None)
        if success is False:
            result = "FAILED"
        elif changed:
            result = "OK (page changed)"
        else:
            result = "OK (page unchanged)"
        url = str(rec.get("obs_url", "") or "")
        if not url:
            state_digest = rec.get("state_digest", {}) or {}
            url = str(state_digest.get("url_after", "") or "")
        url_suffix = f" [{url[:100]}]" if url else ""
        lines.append(f"  Step {rec.get('step_idx', '?')}: {atype}{detail} -> {result}{url_suffix}")
    return "Previous actions:\n" + "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Confidence computation (cross-baseline schema)
# ----------------------------------------------------------------------


def compute_confidence(
    scores: Tuple[torch.Tensor, ...],
) -> Dict[str, Any]:
    """Compute confidence metrics from generation scores.

    Returns a dict with:
      - mean_logprob: average log-probability of generated tokens
      - min_logprob: lowest log-probability (least confident token)
      - mean_margin: average gap between top-1 and top-2 log-probabilities
      - min_margin: smallest gap (most uncertain decision point)
      - mean_entropy: average predictive entropy across tokens
      - max_entropy: highest per-token entropy (most uncertain position)
    """
    if not scores:
        return {}
    try:
        n_tokens = len(scores)
        logprobs_list = []
        margins_list = []
        entropies_list = []
        for i in range(n_tokens):
            logits = scores[i][0]  # (vocab_size,) for batch=0
            log_probs = torch.log_softmax(logits, dim=-1)
            top2 = torch.topk(log_probs, k=2)
            logprobs_list.append(top2.values[0].item())
            margins_list.append((top2.values[0] - top2.values[1]).item())
            # Predictive entropy: H = -∑ p * log(p)
            probs = log_probs.exp()
            entropies_list.append(-(probs * log_probs).sum().item())
        return {
            "mean_logprob": sum(logprobs_list) / n_tokens,
            "min_logprob": min(logprobs_list),
            "mean_margin": sum(margins_list) / n_tokens,
            "min_margin": min(margins_list),
            "mean_entropy": sum(entropies_list) / n_tokens,
            "max_entropy": max(entropies_list),
        }
    except Exception as e:
        logger.warning("Failed to compute confidence metrics: %s", e)
        return {}
