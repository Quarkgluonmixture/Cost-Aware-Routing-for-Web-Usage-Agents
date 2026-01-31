import argparse
import json
import logging
import os
import time
import re
from typing import Dict, Any, List

import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# Import from our codebase
from p79.envs.vwa_wrapper import VWAWrapper, P79Observation
from p79.agents.qwen3vl_agent import Qwen3VLAgent
from p79.utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise web navigation agent.
Output ONLY valid JSON. No markdown blocks, no explanations.

Action Schema:
1. Click: {"action_type": "click", "element_id": int}
   - Use the element_id from the Accessibility Tree.
2. Type: {"action_type": "type", "element_id": int, "text": "string"}
3. Scroll: {"action_type": "scroll", "direction": "down"}
   - Use this if you cannot find the element or need to see more.
4. Stop: {"action_type": "stop", "answer": "optional string"}

Observation provided below includes a screenshot and an Accessibility Tree (text).
Match elements in the screenshot to the Accessibility Tree IDs.
"""

def validate_action(action: Dict[str, Any], obs_text: str) -> Dict[str, Any]:
    """
    Validate that element_id exists in obs_text (Accessibility Tree).
    If invalid or missing, fallback to scroll down.
    """
    action_type = action.get("action_type", "").lower()
    
    if action_type in ["click", "type"]:
        eid = action.get("element_id")
        if eid is None:
            logger.warning(f"Action {action_type} missing element_id. Fallback to scroll.")
            return {"action_type": "scroll", "direction": "down"}
        
        # Simple check: looks for "[ID]" in the text
        # Accessibility tree format usually involves IDs like [123]
        # Regex to find `[eid]`
        pattern = f"\[{eid}\]"
        if not re.search(pattern, obs_text):
            logger.warning(f"Element ID {eid} not found in observation. Fallback to scroll.")
            return {"action_type": "scroll", "direction": "down"}
            
    return action

def run_episode(
    agent_wrapper: Qwen3VLAgent,
    env: VWAWrapper,
    task_config: Dict[str, Any],
    max_steps: int = 30
):
    # 1. Reset
    # We need to save task_config to a file for env.reset if it's a dict
    # But for this script, we assume task_config is loaded from a list file, 
    # so we dump it to a temporary file.
    temp_config_path = f"temp_task_{time.time()}.json"
    
    # Pre-process task config (placeholder replacement)
    # This matches the smoke test logic
    if "start_url" in task_config:
        # Check environment variables for placeholders
        # VWA usually uses env vars, but if they are missing/dummy, we might need manual fix
        # p79/envs/vwa_wrapper.py now sets defaults to example.com if missing
        pass 

    with open(temp_config_path, "w") as f:
        json.dump(task_config, f)
        
    try:
        obs, info = env.reset(temp_config_path)
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    instruction = task_config.get("intent", "Browse the web.")
    logger.info(f"Task Instruction: {instruction}")

    history: List[Dict[str, Any]] = []
    total_tokens = 0
    start_time = time.time()
    
    # JSONL Logger
    log_filename = f"episode_{task_config.get('task_id', 'unknown')}.jsonl"
    logger.info(f"Logging to {log_filename}")
    
    with open(log_filename, "w") as log_file:
        for step_idx in range(max_steps):
            logger.info(f"--- Step {step_idx} ---")
            
            # 2. Prepare Input
            # Resize image if needed (Agent helper does this, but we are doing raw generation)
            # We reuse agent.processor and agent.model
            
            # Construct Prompt
            # User: Image + Text (Instruction + Obs + System)
            
            # Truncate obs text if too long to fit context?
            # Qwen2-VL has large context, but let's be safe or just pass it.
            
            prompt_text = f"System: {SYSTEM_PROMPT}\n\nTask: {instruction}\n\nAccessibility Tree:\n{obs.text}"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": obs.image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            # 3. Inference
            text_inputs = agent_wrapper.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = agent_wrapper.processor(
                text=[text_inputs],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(agent_wrapper.model.device)
            
            input_tokens = inputs.input_ids.shape[1]
            
            gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": True
            }
            
            generated_ids = agent_wrapper.model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = agent_wrapper.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            gen_tokens = len(generated_ids_trimmed[0])
            total_tokens += (input_tokens + gen_tokens)
            
            logger.info(f"Raw Output: {output_text}")
            
            # 4. Parse & Validate
            action = {"action_type": "wait"}
            try:
                # Try JSON parse
                clean_text = output_text.strip()
                # Remove markdown code blocks if present
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()
                
                action = json.loads(clean_text)
            except Exception:
                # Regex fallback
                match = re.search(r"\{.*\}", output_text, re.DOTALL)
                if match:
                    try:
                        action = json.loads(match.group(0))
                    except:
                        pass
            
            # Validate ID
            final_action = validate_action(action, obs.text)
            if final_action != action:
                logger.info(f"Action modified by validator: {action} -> {final_action}")
            
            # 5. Step Env
            if final_action.get("action_type") == "stop":
                logger.info("Agent decided to stop.")
                # Log final state
                step_log = {
                    "step": step_idx,
                    "action": final_action,
                    "reward": 0.0,
                    "terminated": True,
                    "truncated": False,
                    "latency": time.time() - start_time, # approximate total
                    "prompt_tokens": input_tokens,
                    "gen_tokens": gen_tokens
                }
                log_file.write(json.dumps(step_log) + "\n")
                break
                
            obs, reward, terminated, truncated, info = env.step(final_action)
            
            # Log
            step_log = {
                "step": step_idx,
                "action": final_action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "latency": 0.0, # We calculate per-step? tracking start/end of step would be better
                "prompt_tokens": input_tokens,
                "gen_tokens": gen_tokens
            }
            log_file.write(json.dumps(step_log) + "\n")
            log_file.flush()
            
            if terminated or truncated:
                logger.info(f"Episode ended. Terminated: {terminated}, Truncated: {truncated}")
                break
                
    # Summary
    end_time = time.time()
    logger.info("=== Episode Summary ===")
    logger.info(f"Success Proxy: {terminated and reward > 0}") # VWA reward is 1.0 on success usually
    logger.info(f"Total Steps: {step_idx + 1}")
    logger.info(f"Total Tokens: {total_tokens}")
    logger.info(f"Total Wall Time: {end_time - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config", type=str, required=True, help="Path to raw tasks json")
    parser.add_argument("--task_id", type=int, default=0, help="Index of task to run")
    parser.add_argument("--model_path", type=str, default="/mnt/d/(Gluons)/hf_models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    # 1. Load Task Config
    with open(args.task_config, "r") as f:
        tasks = json.load(f)
    
    if args.task_id >= len(tasks):
        raise ValueError(f"Task ID {args.task_id} out of range (0-{len(tasks)-1})")
    
    task = tasks[args.task_id]
    
    # 2. Init Agent (Model)
    # We use Qwen3VLAgent class mainly for loading logic
    agent_config = {
        "model": {
            "path": args.model_path,
            "quantization": "4bit", # Default to 4bit for 7B model
            "device": "cuda"
        }
    }
    logger.info("Loading Agent/Model...")
    agent = Qwen3VLAgent(agent_config)
    
    # 3. Init Env
    logger.info("Initializing VWA Environment...")
    env = VWAWrapper(
        headless=args.headless,
        observation_type="accessibility_tree",
        dry_run=False
    )
    
    # 4. Run
    try:
        run_episode(agent, env, task)
    finally:
        env.close()

if __name__ == "__main__":
    main()
