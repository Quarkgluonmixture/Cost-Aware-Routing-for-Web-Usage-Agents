import argparse
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from p79.utils.config import ConfigLoader
from p79.utils.timers import Timer
from p79.logging.logger import EpisodeLogger
from p79.envs.vwa_wrapper import VWAWrapper
from p79.agents.qwen3vl_agent import Qwen3VLAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run a single episode")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--task_id", type=str, default="test_0", help="Task ID")
    parser.add_argument("--instruction", type=str, default="Find the price of the item.", help="Task instruction")
    args = parser.parse_args()

    # Load Config
    cfg_loader = ConfigLoader(args.config)
    config = cfg_loader.config

    # Init Components
    logger.info("Initializing components...")
    try:
        env = VWAWrapper(config["env"])
        agent = Qwen3VLAgent(config)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return

    episode_logger = EpisodeLogger(config["logging"]["log_dir"], args.task_id, args.instruction)
    timer = Timer()

    # Reset Env
    logger.info(f"Resetting environment for task {args.task_id}")
    obs = env.reset({"task_id": args.task_id, "instruction": args.instruction})
    
    max_steps = config["agent"]["max_steps"]
    done = False
    step = 0

    while not done and step < max_steps:
        logger.info(f"Step {step} | Instruction: {args.instruction[:50]}...")
        
        # 1. Agent Inference
        timer.start()
        action, agent_meta = agent.step(args.instruction, obs)
        inference_ms = timer.stop()
        logger.info(f"Inference: {inference_ms:.2f}ms | Action: {action.get('action_type')}")
        
        # 2. Env Step
        timer.start()
        obs, reward, done, info = env.step(action)
        env_ms = timer.stop()
        
        # 3. Logging
        obs_meta = {
            "image_size": obs["image"].size,
            "dom_len": len(obs.get("dom", ""))
        }
        
        latency_info = {
            "inference": inference_ms,
            "env": env_ms,
            "total": inference_ms + env_ms
        }
        
        token_info = {
            "input": agent_meta.get("input_tokens", 0),
            "output": agent_meta.get("output_tokens", 0)
        }
        
        # Simple cost placeholder
        cost_info = {
            "total": 0.0 
        }
        
        episode_logger.log_step(
            step_id=step,
            obs_meta=obs_meta,
            action_info=action,
            latency_info=latency_info,
            token_info=token_info,
            cost_info=cost_info,
            done=done,
            success=done and reward > 0 # Simple success check
        )
        
        step += 1

    summary = episode_logger.save_summary()
    logger.info(f"Episode finished. Success: {summary['success']}. Summary saved to {summary['log_file']}")

if __name__ == "__main__":
    main()
