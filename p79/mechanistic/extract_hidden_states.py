"""Forward-pass-only hidden state extractor for Qwen3-VL-4B (B1) contrastive set.

Reuses Qwen3VLAgent prompt construction (system_prompt inlined into user content
per agent line 436) so hidden states reflect identical prompt structure as the
agent saw during paper-grade runs.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from p79.utils.torch_cuda_workarounds import apply_nvrtc_prod_fallback_if_needed

logger = logging.getLogger(__name__)


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

        logger.info(f"Loading {model_path} for hidden state extraction (dtype={dtype})")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.device = device

        # Load system prompts from the agent — single source of truth.
        from p79.agents.qwen3vl_agent import Qwen3VLAgent
        self._dom_prompt = Qwen3VLAgent._make_dom_prompt(self)
        self._som_prompt = Qwen3VLAgent._make_som_prompt(self)
        self._mode_to_prompt = {
            "dom": self._dom_prompt,
            "som": self._som_prompt,
            "phantom_som": self._som_prompt,
            "phantom_text": self._dom_prompt,
            "phantom_dom": self._dom_prompt,
            "phantom_prompt": self._som_prompt,
            "vision": Qwen3VLAgent._make_vision_prompt(self),
        }

    def _build_user_text(
        self,
        intent: str,
        mode: str,
        observation_text: str = "",
    ) -> str:
        """Replicate agent's user content text format (qwen3vl_agent.py:436).

        Format: f"Task: {instruction}\\nSystem: {system_prompt}\\n[observation if any]"
        """
        system_prompt = self._mode_to_prompt.get(mode, self._dom_prompt)
        text = f"Task: {intent}\nSystem: {system_prompt}\n"
        if observation_text:
            text += observation_text
        return text

    @torch.no_grad()
    def extract(
        self,
        intent: str,
        mode: str,
        observation_text: str = "",
    ) -> torch.Tensor:
        """Forward pass with output_hidden_states=True. Return last-token hidden states.

        Returns:
            Tensor of shape (n_layers + 1, hidden_dim). Layer 0 is embedding output;
            layer L for L >= 1 is post-transformer-block-L hidden state.
        """
        user_text = self._build_user_text(intent, mode, observation_text)
        messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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
        items: list[tuple[str, str, Optional[str]]],
    ) -> tuple[torch.Tensor, list[str]]:
        """Sequential extraction over (intent, mode, observation_text) tuples.

        Args:
            items: list of (intent, mode, observation_text or None)

        Returns:
            (hidden_states, mode_labels)
            - hidden_states: Tensor (n_items, n_layers + 1, hidden_dim)
            - mode_labels: list of mode strings (for label encoding downstream)
        """
        hs_list = []
        labels = []
        for i, (intent, mode, obs) in enumerate(items):
            hs = self.extract(intent, mode, obs or "")
            hs_list.append(hs)
            labels.append(mode)
            if (i + 1) % 20 == 0:
                logger.info(f"Extracted {i + 1}/{len(items)} hidden states")
        return torch.stack(hs_list, dim=0), labels
