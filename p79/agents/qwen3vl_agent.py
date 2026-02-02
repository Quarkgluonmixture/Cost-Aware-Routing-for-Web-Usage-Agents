import json
import re
import logging
import torch
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

class Qwen3VLAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get("model", {}).get("path", "/mnt/d/(Gluons)/hf_models/Qwen3-VL-4B-Instruct")
        self.device = config.get("model", {}).get("device", "cuda")
        self.quantization = config.get("model", {}).get("quantization", "4bit")
        
        logger.info(f"Loading model from {self.model_path} with quantization={self.quantization}")
        
        # Load Model
        quantization_config = None
        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.quantization != "none" else "auto",
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True # Often needed for new models
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        return """You are a precise web navigation agent.
Output ONLY valid JSON. No markdown blocks, no explanations.

Core Rules:
1) Do NOT answer or finish immediately. First gather evidence by scrolling or inspecting visible results.
2) If the answer is not visible, you MUST scroll down or navigate to search/category pages.
3) Avoid repeating the same click coordinate. If an action seems ineffective, scroll instead.
4) Only use "finish" after confirming the target item is present or absent.
5) Prefer clicking search bars, category filters, or product listings instead of empty background.
6) If you are stuck, use scroll or back; do not repeat identical clicks.

Action Schema:
1. Click: {"action_type": "click", "coordinate": [x, y], "coordinate_type": "normalized"}
   - x, y are floats 0.0-1.0.
2. Type: {"action_type": "type", "text": "string"}
3. Scroll: {"action_type": "scroll", "delta": [dx, dy], "coordinate_type": "normalized"}
4. Wait: {"action_type": "wait"}
5. Back: {"action_type": "back"}
6. Forward: {"action_type": "forward"}
7. Finish: {"action_type": "finish", "answer": "optional string"}

If unsure, verify the screen content by scrolling and exploring.
"""

    def step(self, instruction: str, obs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Takes instruction and observation, returns action dict and metadata.
        """
        image = obs.image
        obs_text = ""
        if hasattr(obs, "text") and obs.text:
            obs_text = obs.text
            max_chars = self.config.get("agent", {}).get("max_obs_chars", 8000)
            if len(obs_text) > max_chars:
                obs_text = obs_text[:max_chars] + "\n[TRUNCATED]"
        
        # Resize if necessary
        max_size = self.config.get("agent", {}).get("image_max_size", 1024)
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Task: {instruction}\nSystem: {self.system_prompt}\n"
                            f"Accessibility Tree:\n{obs_text}"
                        ),
                    },
                ],
            }
        ]

        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate
        gen_kwargs = {
            "max_new_tokens": self.config.get("model", {}).get("max_new_tokens", 256),
            "temperature": self.config.get("model", {}).get("temperature", 0.1),
            "top_p": self.config.get("model", {}).get("top_p", 0.9),
            "do_sample": True
        }
        
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Parse
        action, valid, fail_reason = self._parse_and_validate(output_text)
        
        meta = {
            "raw_output": output_text,
            "valid": valid,
            "failure_reason": fail_reason,
            "input_tokens": inputs.input_ids.shape[1], # Exact count
            "output_tokens": len(generated_ids_trimmed[0]) # Exact count
        }
        
        return action, meta

    def _parse_and_validate(self, text: str) -> Tuple[Dict[str, Any], bool, str]:
        text = text.strip()
        lower_text = text.lower()
        
        # 1. Try direct JSON parse
        try:
            action = json.loads(text)
            return self._validate_schema(action), True, None
        except json.JSONDecodeError:
            pass
            
        # 2. Try regex extraction
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                action = json.loads(match.group(0))
                return self._validate_schema(action), True, "repaired_regex"
            except json.JSONDecodeError:
                pass
        
        # 3. Fallback
        if "scroll" in lower_text:
            return {
                "action_type": "scroll",
                "delta": [0, 0.8],
                "coordinate_type": "normalized",
            }, False, "keyword_scroll"
        if "back" in lower_text:
            return {"action_type": "back"}, False, "keyword_back"
        if "finish" in lower_text or "stop" in lower_text:
            return {"action_type": "finish", "answer": ""}, False, "keyword_finish"
        if "wait" in lower_text:
            return {"action_type": "wait"}, False, "keyword_wait"

        logger.warning(f"Failed to parse action from: {text}")
        return {"action_type": "wait"}, False, "parse_failed"

    def _validate_schema(self, action: Dict[str, Any]) -> Dict[str, Any]:
        # Basic schema validation
        if "action_type" not in action:
            return {"action_type": "wait"} # Invalid schema
        
        # Ensure coordinate exists for click
        if action["action_type"] == "click":
            if "coordinate" not in action and "element_id" not in action:
                return {"action_type": "wait"}
            if "coordinate" in action and "coordinate_type" not in action:
                action["coordinate_type"] = "normalized" # Default
                
        return action
