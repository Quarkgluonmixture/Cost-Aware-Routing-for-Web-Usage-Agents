import logging
import time
import torch
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from p79.utils.torch_cuda_workarounds import apply_nvrtc_prod_fallback_if_needed
from p79.backends.action_utils import parse_action_text

logger = logging.getLogger(__name__)


def _wait_for_vram(min_free_gb: float, poll_interval: int = 30, timeout: int = 0) -> None:
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


class Qwen3VLAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get("model", {}).get("path", "Qwen/Qwen3-VL-4B-Instruct")
        # B-136 (/stress A1.1 v8 Claude F5, 2026-05-15): revision STRICT mode.
        # Removed _DEFAULT_REVISION hardcoded fallback. yaml is the single
        # source-of-truth for paper-grade reproducibility — silent default
        # (even with logger.warning) is a provenance lie:run_meta records
        # the SHA but config does not, so OSF artifact ≠ commit history.
        # Now: explicit revision required; missing key raises immediately.
        # B-83 historical context: backend wrapper sometimes passes
        # `revision=None` literally (key present, value None); `or` handles
        # both missing-key and explicit-None as "unset" → strict raise.
        self.model_revision = config.get("model", {}).get("revision")
        if not self.model_revision:
            raise RuntimeError(
                "model.revision must be pinned in config for paper-grade "
                "reproducibility. Expected an HF SHA (e.g. "
                "'ebb281ec70b05090aa6165b016eac8ec08e71b17'). See "
                "configs/exp_v2_base.yaml model.revision."
            )
        self.device = config.get("model", {}).get("device", "cuda")
        self.quantization = config.get("model", {}).get("quantization", "none")

        # DGX Spark GB10 (sm_121) can hit NVRTC arch errors with some torch builds.
        # This installs a targeted fallback for prod reductions when needed.
        apply_nvrtc_prod_fallback_if_needed()

        # Wait for sufficient VRAM before loading model
        min_free_gb = float(config.get("model", {}).get("min_free_vram_gb", 0))
        if min_free_gb > 0:
            _wait_for_vram(min_free_gb)

        logger.info(f"Loading model from {self.model_path} with quantization={self.quantization}")

        # Load Model
        quantization_config = None
        model_dtype = torch.bfloat16
        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            model_dtype = "auto"
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_dtype = "auto"

        logger.info(f"Loading model {self.model_path} (revision={self.model_revision[:12]}...)")
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                revision=self.model_revision,
                torch_dtype=model_dtype,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                revision=self.model_revision,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

        # Prompts are selected per observation mode at inference time.
        # Phantom-SoM (§25): same SoM prompt + same SoM marks text, but no image.
        # Tests whether the model can complete tasks using SoM textual labels alone
        # (a.k.a. "mirage" mode — preserves prompt that mentions screenshot).
        som_prompt = self._make_som_prompt()
        dom_prompt = self._make_dom_prompt()
        self._system_prompts = {
            "dom": dom_prompt,
            "som": som_prompt,
            "phantom_som": som_prompt,     # P-SoM: SoM prompt + [SOM_MARKS] text + no image (image-mismatched)
            "phantom_dom": dom_prompt,     # P-text (legacy alias): DOM prompt + [SOM_MARKS] text + no image (text-mismatched)
            "phantom_text": dom_prompt,    # P-text (current name): same dispatch as phantom_dom
            "phantom_prompt": som_prompt,  # P-prompt: SoM prompt + AXTree text + no image (prompt-only swap from DOM)
            "vision": self._make_vision_prompt(),
        }
        # Default (backward compat / unknown mode)
        self.system_prompt = self._system_prompts["dom"]

    # ------------------------------------------------------------------
    # Mode-specific system prompts
    # ------------------------------------------------------------------

    @staticmethod
    def _make_dom_prompt() -> str:
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

    @staticmethod
    def _make_som_prompt() -> str:
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

    @staticmethod
    def _make_vision_prompt() -> str:
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

    @staticmethod
    def _compute_confidence(
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

    @staticmethod
    def _format_history(history: List[Dict[str, Any]]) -> str:
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

    def step(
        self,
        instruction: str,
        obs: Any,
        history: Optional[List[Dict[str, Any]]] = None,
        observation_mode: str = "dom",
        reference_images: Optional[List[Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Takes instruction and observation, returns action dict and metadata.

        Args:
            observation_mode: One of "dom", "som", "vision". Selects the
                appropriate system prompt and text label.
            reference_images: Optional list of PIL Images provided by the task
                config (e.g. product photos for "find this item" tasks).
        """
        image = obs.image
        obs_text = ""
        if hasattr(obs, "text") and obs.text:
            obs_text = obs.text
        # B-84: no max_obs_chars truncation. It fired on ~0.2% of steps but only
        # on AXTree modes (marks modes derive from the untruncated text), an
        # axis-1 page-coverage asymmetry. The viewport filter is the real input
        # bound (empirically median 3306 / p99 7656 / max 46592 chars).

        # /stress A1.4 F2: strict mode validation (was silent .get(mode, dom_default)).
        # som.py's prepare_observation_for_mode already raises on unknown mode at the
        # observation-pipeline entry; this is defense-in-depth for any code path that
        # reaches the agent directly with a typo'd mode.
        if observation_mode not in self._system_prompts:
            raise ValueError(
                f"Unknown observation_mode {observation_mode!r}; expected one of "
                f"{sorted(self._system_prompts)}"
            )
        system_prompt = self._system_prompts[observation_mode]

        # Label the text section according to mode
        if observation_mode == "vision":
            obs_section = ""  # no text — screenshot only
        elif observation_mode in ("som", "phantom_som", "phantom_dom", "phantom_text"):
            # obs_text already contains the [SOM_MARKS]...[/SOM_MARKS] block
            # (or "[SOM_MARKS]\n[/SOM_MARKS]" when degraded). Pass it through directly.
            # phantom_som receives the same text but no image (see som.py).
            # phantom_text is the current name for phantom_dom (legacy alias preserved).
            obs_section = obs_text if obs_text else ""
        else:
            # "dom" or "phantom_prompt": full AXTree text. phantom_prompt has the SoM
            # prompt above but the obs is AXTree (no [SOM_MARKS] block in obs_text).
            obs_section = f"Accessibility Tree:\n{obs_text}"

        history_text = self._format_history(history or [])
        content = [
            {
                "type": "text",
                "text": (
                    f"Task: {instruction}\nSystem: {system_prompt}\n"
                    f"{history_text}"
                    f"{obs_section}"
                ),
            }
        ]

        # B-133 (/stress A1.1 v8 3-AI overlap P0-5, 2026-05-15): cross-baseline
        # image-encode lenient alignment. Previously B0 wrapped encode in
        # try/except + counted + continued text-only, while B1/B2 raised →
        # same root cause (corrupt PIL / OOM during resize) produced
        # asymmetric episode outcomes (B0 SoM step degraded to P-SoM step
        # vs B1/B2 episode-killed). All 3 baselines now match: log + count
        # + continue without that image. `aggregate_*.py` must
        # symmetric-exclude steps with image_encode_error > 0 (paper-grade
        # contamination flag, watchdog-auto-clean parallel).
        max_size = self.config.get("agent", {}).get("image_max_size", 1024)
        _image_encode_error_count = 0

        # Inject task reference images (e.g. product photos) before the screenshot
        if reference_images:
            for idx, ref_img in enumerate(reference_images):
                try:
                    if max(ref_img.size) > max_size:
                        ratio = max_size / max(ref_img.size)
                        new_size = (int(ref_img.size[0] * ratio), int(ref_img.size[1] * ratio))
                        ref_img = ref_img.resize(new_size, Image.Resampling.LANCZOS)
                    label = (
                        f"[Reference image {idx + 1}] "
                        f"This image shows the target item described in the task. "
                        f"Use it to identify which element to interact with."
                    )
                    content.append({"type": "text", "text": label})
                    content.append({"type": "image", "image": ref_img})
                except Exception:
                    _image_encode_error_count += 1
                    logger.warning(
                        "B1 failed to encode reference image %d; skipping.",
                        idx + 1, exc_info=True,
                    )

        if image is not None:
            try:
                # Resize if necessary
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                if reference_images:
                    # With reference images: append screenshot at end with label
                    content.append({"type": "text", "text": "[Current screenshot]"})
                    content.append({"type": "image", "image": image})
                else:
                    # No reference images: preserve original position (before text)
                    content.insert(0, {"type": "image", "image": image})
            except Exception:
                _image_encode_error_count += 1
                logger.warning(
                    "B1 failed to encode screenshot; continuing without image.",
                    exc_info=True,
                )
                image = None  # ensure downstream image-bool checks see None

        messages = [{"role": "user", "content": content}]

        # Prepare for inference
        preprocess_start = time.time()
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        has_images = image is not None or bool(reference_images)
        if has_images:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.model.device)
        preprocess_ms = (time.time() - preprocess_start) * 1000.0

        # Count image tokens (expanded from vision patches by the processor)
        image_token_count = int((inputs.input_ids == self.processor.image_token_id).sum().item())

        # Generate. Default raised from 256 → 4096 (§45 alignment): 256 is
        # below typical thought+JSON envelope (~400-1500 tok), causing silent
        # truncation that produces parse errors rather than valid actions.
        max_new_tokens = int(self.config.get("model", {}).get("max_new_tokens", 4096))
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        # B-37 fix: torch.manual_seed already called per-condition in runner.run(),
        # but transformers.generate() can introduce its own RNG state during
        # tokenization padding / mask. Force fresh seed before each generate call
        # to guarantee strict per-step reproducibility on B1 greedy path.
        _b37_seed = self.config.get("model", {}).get("seed")
        if _b37_seed is not None:
            try:
                import torch as _torch_b37
                _torch_b37.manual_seed(int(_b37_seed))
                if _torch_b37.cuda.is_available():
                    _torch_b37.cuda.manual_seed_all(int(_b37_seed))
            except ImportError:
                pass

        generate_start = time.time()
        gen_output = self.model.generate(**inputs, **gen_kwargs)
        generate_ms = (time.time() - generate_start) * 1000.0
        generated_ids = gen_output.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Extract confidence metrics from logprobs
        confidence_metrics = self._compute_confidence(gen_output.scores)

        # Parse
        action, valid, fail_reason = parse_action_text(output_text)
        
        # Enforce newline for search queries if missing
        if action.get("action_type") == "type":
            text = action.get("text", "") or ""
            # action.get("thought") may be None (e.g. agent omitted the field
            # despite system-prompt requirement) — guard before .lower().
            thought = (action.get("thought") or "").lower()
            # If thought mentions search/find, or if text looks like a query (short, no newlines)
            if ("search" in thought or "find" in thought or "look for" in thought) and not text.endswith("\n"):
                action["text"] = text + "\n"
                logger.info(f"Auto-appended newline to search query: {repr(action['text'])}")

        meta = {
            "raw_output": output_text,
            "valid": valid,
            "failure_reason": fail_reason,
            "input_tokens": inputs.input_ids.shape[1],  # Exact count
            "input_image_tokens": image_token_count,
            "input_text_tokens": inputs.input_ids.shape[1] - image_token_count,
            "output_tokens": len(generated_ids_trimmed[0]),  # Exact count
            "preprocess_ms": preprocess_ms,
            "generate_ms": generate_ms,
            # B-133 (/stress A1.1 v8 3-AI overlap P0-5, 2026-05-15): align
            # with B0 + B2 — count of image-encode failures this step
            # (0 = clean; >0 = N images silently dropped). Persisted via
            # runner step_record.image_meta (B-112 wiring). aggregate_*.py
            # MUST symmetric-exclude steps with image_encode_error > 0 for
            # paper-grade cross-baseline SR comparability.
            "image_encode_error": _image_encode_error_count if _image_encode_error_count else None,
            **confidence_metrics,
        }

        return action, meta
