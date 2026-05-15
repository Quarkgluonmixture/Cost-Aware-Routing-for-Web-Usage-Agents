"""Forward-pass-only hidden state extractor for Qwen3-VL-4B (B1) contrastive set.

Reuses Qwen3VLAgent prompt construction (system_prompt inlined into user content
per agent line 436) so hidden states reflect identical prompt structure as the
agent saw during paper-grade runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from p79.utils.torch_cuda_workarounds import apply_nvrtc_prod_fallback_if_needed

logger = logging.getLogger(__name__)

# Match qwen3vl_agent.py default image_max_size for production parity
IMAGE_MAX_SIZE_DEFAULT = 1024


class HiddenStateExtractor:
    """Extract last-input-token hidden states from Qwen3-VL-4B forward pass.

    Stage 1 pilot scope: empty observation, system prompt + task intent only.
    Later stages: load actual archived observations (DOM AXTree / SoM marks).
    """

    # System prompts — copied from p79.agents.qwen3vl_agent (must keep in sync; if
    # the agent prompts drift, hidden states will not reflect production conditions).
    # Source: qwen3vl_agent.py::_make_dom_prompt / _make_som_prompt as of 2026-05-05.

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-4B-Instruct",
        # Paper-grade: pin HF revision SHA — DGX baseline lock 2026-05-07 (笔记 §114)
        model_revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        min_free_vram_gb: float = 12.0,
    ):
        apply_nvrtc_prod_fallback_if_needed()

        if min_free_vram_gb > 0 and torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
            if free_gb < min_free_vram_gb:
                raise RuntimeError(
                    f"Insufficient VRAM: {free_gb:.1f} GB free < {min_free_vram_gb:.1f} GB required. "
                    f"Wait for other GPU jobs to finish or set min_free_vram_gb=0 to skip check."
                )

        logger.info(f"Loading {model_path} (revision={model_revision[:12]}...) for hidden state extraction (dtype={dtype})")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            revision=model_revision,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, revision=model_revision, trust_remote_code=True
        )
        self.device = device
        self.model_revision = model_revision

        # Load system prompts from the agent — single source of truth.
        # /stress A1.1 B-92 propagation fix (2026-05-15): _make_*_prompt are
        # @staticmethod since commit 11d6fd9, so the previous `(self)` argument
        # would raise TypeError. Drop the arg.
        from p79.agents.qwen3vl_agent import Qwen3VLAgent
        self._dom_prompt = Qwen3VLAgent._make_dom_prompt()
        self._som_prompt = Qwen3VLAgent._make_som_prompt()
        self._mode_to_prompt = {
            "dom": self._dom_prompt,
            "som": self._som_prompt,
            "phantom_som": self._som_prompt,
            "phantom_text": self._dom_prompt,
            "phantom_dom": self._dom_prompt,
            "phantom_prompt": self._som_prompt,
            "vision": Qwen3VLAgent._make_vision_prompt(),
        }

    def _build_user_text(
        self,
        intent: str,
        mode: str,
        observation_text: str = "",
    ) -> str:
        """Replicate agent's user content text format (qwen3vl_agent.py:441-450).

        Format: f"Task: {instruction}\\nSystem: {system_prompt}\\n[obs_section]"
        where obs_section is mode-conditional:
          - vision: ""
          - som / phantom_som / phantom_dom / phantom_text: obs_text (no
            prefix — text already wrapped in [SOM_MARKS]...[/SOM_MARKS])
          - dom / phantom_prompt: "Accessibility Tree:\\n" + obs_text

        /stress A1.4 B-103 fix (2026-05-15): the `Accessibility Tree:\\n`
        prefix for DOM-style modes was missing, so NPZ extraction for
        `dom` and `phantom_prompt` modes was not byte-identical to the
        production agent input. Mechanism §5 is paused per advisor §138
        (frozen archive, not future-scheduled) but the byte-divergence is
        fixed in place so any non-mechanism reuse of this builder stays
        production-consistent.
        """
        system_prompt = self._mode_to_prompt.get(mode, self._dom_prompt)
        text = f"Task: {intent}\nSystem: {system_prompt}\n"
        if observation_text:
            if mode in ("dom", "phantom_prompt"):
                text += f"Accessibility Tree:\n{observation_text}"
            else:
                text += observation_text
        return text

    @staticmethod
    def _load_resize_image(image_path: Union[str, Path], max_size: int = IMAGE_MAX_SIZE_DEFAULT) -> Image.Image:
        """Load + LANCZOS-resize image to max_size (matches qwen3vl_agent.py:447-450)."""
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        return img

    @torch.no_grad()
    def extract(
        self,
        intent: str,
        mode: str,
        observation_text: str = "",
        image_path: Optional[Union[str, Path]] = None,
    ) -> torch.Tensor:
        """Forward pass with output_hidden_states=True. Return last-token hidden states.

        Args:
            intent: task instruction
            mode: observation mode (dom / som / phantom_som / phantom_text / phantom_prompt / vision)
            observation_text: full AXTree or [SOM_MARKS] text (mode-conditional)
            image_path: if provided, load image and add to messages content
                (multimodal forward pass; for SoM / Vision modes)

        Returns:
            Tensor of shape (n_layers + 1, hidden_dim). Layer 0 is embedding output;
            layer L for L >= 1 is post-transformer-block-L hidden state.
        """
        user_text = self._build_user_text(intent, mode, observation_text)

        # Build content. For multimodal: image first, then text (matches agent line 471).
        content = []
        if image_path is not None:
            img = self._load_resize_image(image_path)
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if image_path is not None:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(text=[text], padding=True, return_tensors="pt")

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        # outputs.hidden_states is tuple of (n_layers + 1) tensors of shape
        # (batch=1, seq_len, hidden_dim). Extract last token of each.
        hidden_states = torch.stack(
            [h[0, -1, :].detach().float().cpu() for h in outputs.hidden_states],
            dim=0,
        )  # (n_layers + 1, hidden_dim)
        return hidden_states

    def extract_batch(
        self,
        items: list[tuple[str, str, Optional[str], Optional[Union[str, Path]]]],
    ) -> tuple[torch.Tensor, list[str]]:
        """Sequential extraction over (intent, mode, observation_text, image_path) tuples.

        Args:
            items: list of (intent, mode, observation_text or None, image_path or None)

        Returns:
            (hidden_states, mode_labels)
            - hidden_states: Tensor (n_items, n_layers + 1, hidden_dim)
            - mode_labels: list of mode strings (for label encoding downstream)
        """
        hs_list = []
        labels = []
        for i, item in enumerate(items):
            # Backward-compat: support 3-tuple (without image_path)
            if len(item) == 3:
                intent, mode, obs = item
                image_path = None
            else:
                intent, mode, obs, image_path = item
            hs = self.extract(intent, mode, obs or "", image_path=image_path)
            hs_list.append(hs)
            labels.append(mode)
            if (i + 1) % 20 == 0:
                logger.info(f"Extracted {i + 1}/{len(items)} hidden states")
        return torch.stack(hs_list, dim=0), labels
