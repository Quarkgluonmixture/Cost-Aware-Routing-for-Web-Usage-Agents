"""B3 dev-pilot agent — Xiaomi MiMo-VL-7B-RL-2508 (cross-family, DEV ONLY).

MiMo-VL ships as the transformers Qwen2.5-VL deployment class
(``Qwen2_5_VLForConditionalGeneration``; we load via the generic
``AutoModelForImageTextToText`` exactly as the Stage-0 probe did, which is the
proven path on this transformers build). Its processing stack — AutoProcessor +
``qwen_vl_utils.process_vision_info`` — is isomorphic to B1's Qwen3-VL, so
``MiMoVLAgent`` SUBCLASSES :class:`Qwen3VLAgent` and inherits ``step()`` /
prompt dispatch / confidence verbatim. Only the model load differs (different
HF class + default path + lenient revision). The ``-RL`` thinking checkpoint
emits ``<think>...</think>`` before the JSON action; that prefix is stripped
downstream by ``action_utils.parse_action_text`` (re.IGNORECASE, B-800) so no
parser change is needed.

DEV-ONLY scaffold for the §340 B3 cross-family floor pilot — NOT in the
paper-grade fire import path (B0/B1/B2). Revision is lenient (warn, not the
B-136 strict raise) because this is a dev floor-pilot; still pin the SHA in the
pilot config for reproducibility. If B3 is promoted to a paper-grade baseline,
re-home this as a standalone class + restore strict-revision per repo convention.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from p79.agents.qwen3vl_agent import Qwen3VLAgent
from p79.agents._shared_vl_utils import (
    build_mode_prompt_dispatch_table as _shared_build_mode_prompt_dispatch_table,
    wait_for_vram as _wait_for_vram,
)
from p79.utils.torch_cuda_workarounds import apply_nvrtc_prod_fallback_if_needed

logger = logging.getLogger(__name__)


class MiMoVLAgent(Qwen3VLAgent):
    """MiMo-VL-7B-RL agent. Inherits Qwen3VLAgent.step()/prompts/confidence;
    overrides only model loading (Qwen2.5-VL deployment class)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get("model", {}).get("path", "XiaomiMiMo/MiMo-VL-7B-RL-2508")
        # Lenient revision (dev pilot). B1/B2 raise on missing revision (B-136
        # paper-grade reproducibility); here we warn and load the cached
        # snapshot so a missing SHA can't block a dev floor-pilot. Pin it in
        # the pilot config for reproducibility.
        self.model_revision = config.get("model", {}).get("revision") or None
        if self.model_revision is None:
            logger.warning(
                "MiMoVLAgent: model.revision not pinned — loading latest cached "
                "snapshot (DEV pilot; pin the HF SHA in config for reproducibility)."
            )
        self.device = config.get("model", {}).get("device", "cuda")
        self.quantization = config.get("model", {}).get("quantization", "none")

        # DGX Spark GB10 (sm_121) NVRTC arch fallback — MiMo's Qwen2.5-VL vision
        # path calls image_grid_thw.prod(-1) which crashes without this patch.
        apply_nvrtc_prod_fallback_if_needed()

        min_free_gb = float(config.get("model", {}).get("min_free_vram_gb", 0))
        if min_free_gb > 0:
            _wait_for_vram(min_free_gb)

        logger.info(
            "Loading MiMo-VL from %s (quant=%s, revision=%s)",
            self.model_path, self.quantization, self.model_revision,
        )

        quantization_config = None
        model_dtype = torch.bfloat16
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

        try:
            # AutoModelForImageTextToText resolves to Qwen2_5_VLForConditionalGeneration
            # for this checkpoint (Stage-0 probe proven path on this transformers build).
            self.model = AutoModelForImageTextToText.from_pretrained(
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
            logger.error("Failed to load MiMo-VL model: %s", e)
            raise

        # The inherited step() counts image tokens via self.processor.image_token_id
        # (Qwen3-VL exposes it). Qwen2.5-VL's processor should too; monkeypatch a
        # fallback on the instance if a variant lacks it so the inherited step()
        # cannot crash mid-pilot. <|image_pad|> is the Qwen-VL image placeholder.
        if getattr(self.processor, "image_token_id", None) is None:
            iid = None
            tok = getattr(self.processor, "tokenizer", None)
            if tok is not None:
                try:
                    iid = tok.convert_tokens_to_ids("<|image_pad|>")
                except Exception:
                    iid = None
            self.processor.image_token_id = iid if iid not in (None, -1) else -1
            logger.warning(
                "MiMoVLAgent: processor lacked image_token_id; set fallback=%s "
                "(image-token accounting may be approximate for this dev pilot).",
                self.processor.image_token_id,
            )

        # Canonical 7-key mode→prompt dispatch (byte-identical with B0/B1/B2).
        self._system_prompts = _shared_build_mode_prompt_dispatch_table()
        self.system_prompt = self._system_prompts["dom"]
