import json
import re
import logging
import time
import torch
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from p79.utils.torch_cuda_workarounds import apply_nvrtc_prod_fallback_if_needed

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

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=model_dtype,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True,
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
1) Do NOT answer or finish immediately. You MUST navigate to find the item.
2) If the target category (e.g., "Blankets & Throws") is not visible, look for a parent category (e.g., "Home & Kitchen") or use the search bar.
3) NEVER give up early. If you don't see the item, SEARCH for it using the search bar.
4) Only use "finish" when you have successfully completed the task (e.g., found the item, placed order) or if you have searched everywhere and are 100% sure it's missing.
5) If you are on the homepage, DO NOT go back. Start by searching or clicking a category.
6) If you are stuck, use scroll or try a different category/search.

Response Format (JSON):
{
  "thought": "Brief reasoning about what to do next. Why are you choosing this action? What is your plan?",
  "action_type": "click" | "type" | "scroll" | "wait" | "back" | "forward" | "finish" | "tab_focus",
  ... (other action parameters) ...
}

Action Schema:
1. Click: {"action_type": "click", "element_id": N}
   - N is the numeric ID from the Accessibility Tree (e.g., [175] link 'Comments' -> element_id: 175).
   - This is the PREFERRED way to click. Use element IDs from the Accessibility Tree.
   - Alternative (only if no element ID): {"action_type": "click", "coordinate": [x, y], "coordinate_type": "normalized"} with x, y as floats 0.0-1.0.
2. Type: {"action_type": "type", "text": "string", "element_id": N}
   - ALWAYS specify element_id to target the correct input field (e.g., search box [397], text field [132]).
   - To submit a search or form, append "\\n" to the text (e.g., "red blanket\\n").
   - Without element_id, text goes to whatever is focused, which is often WRONG.
3. Scroll: {"action_type": "scroll", "delta": [dx, dy], "coordinate_type": "normalized"}
4. Wait: {"action_type": "wait"}
5. Back: {"action_type": "back"}
   - WARNING: Do NOT use "back" if you are on the first page (homepage). Going back from the first page leads to a blank page (about:blank) and you will be stuck.
6. Forward: {"action_type": "forward"}
7. Finish: {"action_type": "finish", "answer": "optional string"}
8. Tab focus: {"action_type": "tab_focus", "page_number": int}

Tab Rule:
- If the Accessibility Tree lists tabs like "Tab 0" / "Tab 1", use tab_focus to switch to the tab that matches the site you need (e.g., Wikipedia). Do NOT click random coordinates to switch tabs.
- If the task says "Wikipedia site in the second tab", immediately use {"action_type":"tab_focus","page_number":1} before any clicks.

CRITICAL:
- You MUST include a "thought" field to explain your reasoning.
- DO NOT use "finish" to report failure. "finish" is ONLY for success or after EXHAUSTIVE search (at least 3 different search queries/attempts).
- If you are in the wrong category, use the search bar or click a navigation link. Avoid "back" unless you are sure it won't lead to about:blank.
- PREFER clicking on Categories (e.g., "Home & Kitchen" -> "Blankets & Throws") over searching if search results are poor.
- If search returns unrelated items (e.g. seafood instead of blankets), STOP searching immediately. Navigate via Categories.
- Do NOT output literal newlines inside JSON strings. Use \\n for newline.
- If search results appear, CLICK on the most promising item to verify details (price, color). Do not just stare at the list.
- Avoid repeating the same search query or action. If something doesn't work, change your strategy.
- ALWAYS use element_id from the Accessibility Tree for click and type actions. Do NOT guess coordinates or type blindly.
"""

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
                coord = act.get("coordinate", "?")
                detail = f" {coord}"
            elif atype == "type":
                detail = f' "{act.get("text", "")}"'
            elif atype == "scroll":
                detail = f' delta={act.get("delta", "?")}'
            success = rec.get("action_success", None)
            changed = rec.get("page_changed", None)
            if success is False or changed is False:
                result = "FAILED (page unchanged)"
            elif changed:
                result = "OK (page changed)"
            else:
                result = "OK"
            lines.append(f"  Step {rec.get('step_idx', '?')}: {atype}{detail} -> {result}")
        return "Previous actions:\n" + "\n".join(lines) + "\n"

    def step(self, instruction: str, obs: Any, history: Optional[List[Dict[str, Any]]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

        history_text = self._format_history(history or [])

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
                            f"{history_text}"
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
        
        # Enforce newline for search queries if missing
        if action.get("action_type") == "type":
            text = action.get("text", "")
            thought = action.get("thought", "").lower()
            # If thought mentions search/find, or if text looks like a query (short, no newlines)
            if ("search" in thought or "find" in thought or "look for" in thought) and not text.endswith("\n"):
                action["text"] = text + "\n"
                logger.info(f"Auto-appended newline to search query: {repr(action['text'])}")

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
