from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# B-146 (/stress A1.2 v8 codex B4, 2026-05-16): cross-baseline VL helpers
# moved from Qwen3VLAgent classmethods into ``_shared_vl_utils`` so this
# module no longer transitively pulls in Qwen3VLAgent's heavy deps
# (``transformers.Qwen3VLForConditionalGeneration`` + ``qwen_vl_utils``).
# Cross-baseline byte-identical prompts + identical confidence schema are
# still enforced — single source of truth is the shared module.
from p79.agents._shared_vl_utils import (
    build_mode_prompt_dispatch_table as _shared_build_mode_prompt_dispatch_table,
    compute_confidence as _shared_compute_confidence,
    format_history as _shared_format_history,
    make_dom_prompt as _shared_make_dom_prompt,
    make_som_prompt as _shared_make_som_prompt,
    make_vision_prompt as _shared_make_vision_prompt,
    wait_for_vram as _wait_for_vram,
)
from p79.backends.action_utils import parse_action_text
from p79.utils.torch_cuda_workarounds import apply_nvrtc_prod_fallback_if_needed

logger = logging.getLogger(__name__)

# Gemma 3 normalizes every input image to 896x896 and encodes it to a fixed
# 256 tokens (Gemma 3 technical report / HF model card). Unlike Qwen3-VL's
# variable patch count, this is constant per image.
GEMMA3_IMAGE_TOKENS = 256

_DOM_PROMPT = _shared_make_dom_prompt()
_SOM_PROMPT = _shared_make_som_prompt()
_VISION_PROMPT = _shared_make_vision_prompt()


class Gemma3VLAgent:
    """Local Gemma 3 vision-language agent — the cross-family third baseline.

    Mirrors Qwen3VLAgent's ``step()`` contract (identical action/meta shape) so
    the backend and runner stay model-agnostic. Gemma-specific differences are
    isolated here: the model class, a single-step processor call (Gemma's
    AutoProcessor handles vision preprocessing — no qwen_vl_utils), and fixed
    256-token-per-image accounting.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_cfg = config.get("model", {})
        self.model_path = model_cfg.get("path", "google/gemma-3-4b-it")
        # B-136 (/stress A1.1 v8 Claude F5, 2026-05-15): revision STRICT mode
        # aligned with B-136 in qwen3vl_agent.py. Previously this agent
        # warned-and-loaded-HEAD on missing revision; now matches Qwen path
        # by raising. Single cross-baseline policy: paper-grade configs MUST
        # pin revision explicitly; no silent default / no HF-HEAD fallback.
        self.model_revision = model_cfg.get("revision")
        if not self.model_revision:
            raise RuntimeError(
                "model.revision must be pinned in config for paper-grade "
                "reproducibility. Loading at HF HEAD breaks the OSF lock "
                "manifest — run_meta records the loaded SHA but config does "
                "not, so reviewers cannot replay. See "
                "configs/exp_v2_base.yaml local_gemma.revision."
            )
        self.device = model_cfg.get("device", "cuda")
        self.quantization = model_cfg.get("quantization", "none")

        # DGX Spark GB10 (sm_121) NVRTC arch fallback — harmless no-op on A100.
        apply_nvrtc_prod_fallback_if_needed()

        min_free_gb = float(model_cfg.get("min_free_vram_gb", 0))
        if min_free_gb > 0:
            _wait_for_vram(min_free_gb)

        quantization_config = None
        model_dtype: Any = torch.bfloat16
        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model_dtype = "auto"
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_dtype = "auto"

        rev = self.model_revision
        logger.info(
            "Loading Gemma3 model %s (revision=%s, quantization=%s)",
            self.model_path,
            (rev[:12] + "...") if rev else "<unset>",
            self.quantization,
        )
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": model_dtype,
            "device_map": "auto",
            "quantization_config": quantization_config,
        }
        if rev:
            load_kwargs["revision"] = rev
        try:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_path, **load_kwargs
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, **({"revision": rev} if rev else {})
            )
        except Exception as e:
            logger.error("Failed to load Gemma3 model: %s", e)
            raise

        # bf16 input cast only when running unquantized (matches loaded dtype);
        # quantized loads manage input dtype internally.
        self._input_dtype = torch.bfloat16 if quantization_config is None else None

        # B-451 (/stress A1.4 P0-5-A* OOB, 2026-05-17): use the canonical
        # dispatch table from `_shared_vl_utils` (single source of truth across
        # B0/B1/B2 + mechanistic extractor). Module-level `_DOM_PROMPT` / `_SOM_PROMPT`
        # / `_VISION_PROMPT` constants retained for backwards-compat callers /
        # external imports but the agent now consumes the canonical dict directly.
        self._system_prompts = _shared_build_mode_prompt_dispatch_table()
        self.system_prompt = self._system_prompts["dom"]

    def step(
        self,
        instruction: str,
        obs: Any,
        history: Optional[List[Dict[str, Any]]] = None,
        observation_mode: str = "dom",
        reference_images: Optional[List[Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        image = getattr(obs, "image", None)
        obs_text = ""
        if hasattr(obs, "text") and obs.text:
            obs_text = obs.text
        # B-84: no max_obs_chars truncation — it created an axis-1 page-coverage
        # asymmetry (AXTree modes truncated while marks modes derive from the
        # untruncated text). The viewport filter is the real input bound. Kept
        # in lockstep with the Qwen agent so the 3 baselines stay comparable.

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

        # Mode -> text-section labelling, identical to the Qwen agent.
        if observation_mode == "vision":
            obs_section = ""
        elif observation_mode in ("som", "phantom_som", "phantom_dom", "phantom_text"):
            obs_section = obs_text if obs_text else ""
        else:  # "dom" or "phantom_prompt": full AXTree text
            obs_section = f"Accessibility Tree:\n{obs_text}"

        history_text = _shared_format_history(history or [])

        # The system prompt is embedded in the user turn's text — NOT a separate
        # `system` role — even though Gemma 3 supports system roles natively.
        # This keeps prompt structure identical to B0/B1 (project guard rail:
        # prompt placement must not vary across baselines).
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Task: {instruction}\nSystem: {system_prompt}\n"
                    f"{history_text}{obs_section}"
                ),
            }
        ]

        # B-133 (/stress A1.1 v8 3-AI overlap P0-5, 2026-05-15): cross-baseline
        # image-encode lenient alignment. See qwen3vl_agent.py for full
        # rationale — paper-grade requirement is "same root cause yields same
        # cross-baseline outcome". B2 now matches B0 + B1 lenient pattern.
        max_size = self.config.get("agent", {}).get("image_max_size", 1024)
        _image_encode_error_count = 0

        if reference_images:
            for idx, ref_img in enumerate(reference_images):
                try:
                    if max(ref_img.size) > max_size:
                        ratio = max_size / max(ref_img.size)
                        ref_img = ref_img.resize(
                            (int(ref_img.size[0] * ratio), int(ref_img.size[1] * ratio)),
                            Image.Resampling.LANCZOS,
                        )
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                f"[Reference image {idx + 1}] This image shows the "
                                f"target item described in the task. Use it to "
                                f"identify which element to interact with."
                            ),
                        }
                    )
                    content.append({"type": "image", "image": ref_img})
                except Exception:
                    _image_encode_error_count += 1
                    logger.warning(
                        "B2 failed to encode reference image %d; skipping.",
                        idx + 1, exc_info=True,
                    )

        if image is not None:
            try:
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    image = image.resize(
                        (int(image.size[0] * ratio), int(image.size[1] * ratio)),
                        Image.Resampling.LANCZOS,
                    )
                if reference_images:
                    content.append({"type": "text", "text": "[Current screenshot]"})
                    content.append({"type": "image", "image": image})
                else:
                    content.insert(0, {"type": "image", "image": image})
            except Exception:
                _image_encode_error_count += 1
                logger.warning(
                    "B2 failed to encode screenshot; continuing without image.",
                    exc_info=True,
                )
                image = None  # ensure downstream image-bool checks see None

        messages = [{"role": "user", "content": content}]

        # Gemma's AutoProcessor handles all vision preprocessing in a single
        # apply_chat_template call — no separate process_vision_info step.
        preprocess_start = time.time()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if self._input_dtype is not None:
            inputs = inputs.to(self.model.device, dtype=self._input_dtype)
        else:
            inputs = inputs.to(self.model.device)
        preprocess_ms = (time.time() - preprocess_start) * 1000.0

        # Image-token accounting. Gemma 3 = fixed 256 tokens/image; prefer an
        # exact count from the processor's image-token id when it is exposed.
        # B-139 (/stress A1.1 v8 Claude F6, 2026-05-15): track which method
        # the processor allowed so silent transformers-version drift between
        # estimate-256/img and exact-id-match doesn't change cost-accounting
        # semantics without an audit trail. Meta emits the chosen method.
        n_images = (1 if image is not None else 0) + (
            len(reference_images) if reference_images else 0
        )
        image_token_count = n_images * GEMMA3_IMAGE_TOKENS
        image_token_count_method = "estimate_256_per_image"
        img_tok_id = getattr(self.processor, "image_token_id", None)
        if img_tok_id is not None:
            image_token_count = int((inputs["input_ids"] == img_tok_id).sum().item())
            image_token_count_method = "exact_id_match"

        max_new_tokens = int(self.config.get("model", {}).get("max_new_tokens", 4096))
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        # Per-step seed reset for strict greedy reproducibility (mirrors the
        # B-37 fix in the Qwen agent).
        _seed = self.config.get("model", {}).get("seed")
        if _seed is not None:
            torch.manual_seed(int(_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(_seed))

        generate_start = time.time()
        gen_output = self.model.generate(**inputs, **gen_kwargs)
        generate_ms = (time.time() - generate_start) * 1000.0

        input_len = inputs["input_ids"].shape[1]
        generated_ids_trimmed = [seq[input_len:] for seq in gen_output.sequences]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        confidence_metrics = _shared_compute_confidence(gen_output.scores)

        action, valid, fail_reason = parse_action_text(output_text)

        # Enforce a trailing newline on search-query typing (identical to the
        # Qwen agent's behaviour).
        if action.get("action_type") == "type":
            typed = action.get("text", "") or ""
            thought = (action.get("thought") or "").lower()
            if (
                ("search" in thought or "find" in thought or "look for" in thought)
                and not typed.endswith("\n")
            ):
                action["text"] = typed + "\n"
                logger.info("Auto-appended newline to search query: %r", action["text"])

        meta = {
            "raw_output": output_text,
            "valid": valid,
            "failure_reason": fail_reason,
            "input_tokens": input_len,
            "input_image_tokens": image_token_count,
            "input_text_tokens": input_len - image_token_count,
            "output_tokens": len(generated_ids_trimmed[0]),
            "preprocess_ms": preprocess_ms,
            "generate_ms": generate_ms,
            # B-133 (/stress A1.1 v8 3-AI overlap P0-5, 2026-05-15): align
            # with B0 + B1 — count of image-encode failures this step.
            # Persisted via runner step_record.image_meta (B-112 wiring).
            # aggregate_*.py MUST symmetric-exclude image_encode_error > 0.
            "image_encode_error": _image_encode_error_count if _image_encode_error_count else None,
            # B-139 (/stress A1.1 v8 Claude F6, 2026-05-15): which method
            # gave image_token_count above. "exact_id_match" = transformers
            # exposed processor.image_token_id and we counted token instances;
            # "estimate_256_per_image" = older transformers, n_images*256
            # static cost. Reviewer auditing cost ≈ DOM hero claim across
            # transformers versions needs to see the method, not just the count.
            "image_token_count_method": image_token_count_method,
            **confidence_metrics,
        }

        return action, meta
