import logging
import time
import torch
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from p79.utils.torch_cuda_workarounds import apply_nvrtc_prod_fallback_if_needed
from p79.backends.action_utils import parse_action_text
# B-146 (/stress A1.2 v8 codex B4, 2026-05-16): cross-baseline VL helpers
# moved to ``_shared_vl_utils`` so Gemma3VLAgent (and any future B3/B4) can
# consume them without transitively importing this module's heavy
# ``transformers.Qwen3VLForConditionalGeneration`` + ``qwen_vl_utils`` deps.
from p79.agents._shared_vl_utils import (
    compute_confidence as _shared_compute_confidence,
    format_history as _shared_format_history,
    make_dom_prompt as _shared_make_dom_prompt,
    make_som_prompt as _shared_make_som_prompt,
    make_vision_prompt as _shared_make_vision_prompt,
    wait_for_vram as _wait_for_vram,
)

logger = logging.getLogger(__name__)


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
    # B-146 (/stress A1.2 v8 codex B4, 2026-05-16): bodies live in
    # ``p79/agents/_shared_vl_utils.py`` so Gemma3VLAgent + cross-family
    # mechanistic scripts can consume them without transitively importing
    # Qwen-specific deps. Classmethods preserved as backward-compatible
    # delegates so external callers (proxy_api_agent, extract_hidden_states,
    # run_stage4_h1_*, tests/test_agents_prompt_parity) keep working.

    @staticmethod
    def _make_dom_prompt() -> str:
        return _shared_make_dom_prompt()

    @staticmethod
    def _make_som_prompt() -> str:
        return _shared_make_som_prompt()

    @staticmethod
    def _make_vision_prompt() -> str:
        return _shared_make_vision_prompt()

    @staticmethod
    def _compute_confidence(
        scores: Tuple[torch.Tensor, ...],
    ) -> Dict[str, Any]:
        return _shared_compute_confidence(scores)

    @staticmethod
    def _format_history(history: List[Dict[str, Any]]) -> str:
        return _shared_format_history(history)

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
