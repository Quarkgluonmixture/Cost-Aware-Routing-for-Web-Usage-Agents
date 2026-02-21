import argparse
import json
import logging
import os
import sys
import time
import yaml
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add external/visualwebarena to path
sys.path.append(os.path.join(os.getcwd(), "external/visualwebarena"))

# Ensure env vars are set before importing browser_env
if "DATASET" not in os.environ:
    os.environ["DATASET"] = "visualwebarena"
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "dummy_key"
required_vars = ["REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE", "CLASSIFIEDS", "CLASSIFIEDS_RESET_TOKEN"]
for var in required_vars:
    if var not in os.environ:
        os.environ[var] = "https://example.com"

# VWA Imports
import browser_env.env_config as env_config
from browser_env.actions import (
    create_stop_action,
    ActionTypes,
)
from evaluation_harness import evaluator_router

# P79 Imports
from p79.envs.vwa_wrapper import VWAWrapper, P79Observation
from p79.agents.qwen_api_agent import QwenApiAgent

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle simple inheritance via 'defaults' list
    if "defaults" in config:
        for default_path in config["defaults"]:
            # Try finding default path
            if not os.path.exists(default_path):
                # try relative to config file directory
                p = Path(config_path).parent / default_path
                if p.exists():
                    default_path = str(p)
                else:
                    logger.warning(f"Default config {default_path} not found")
                    continue

            with open(default_path, "r") as f:
                base_config = yaml.safe_load(f)

            # Recursive merge: config overrides base_config
            def merge(base, update):
                for k, v in update.items():
                    if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                        merge(base[k], v)
                    else:
                        base[k] = v
                return base

            # Merge into base_config, then apply update
            config = merge(base_config, config)

        del config["defaults"]

    return config


def resolve_placeholders(text: str) -> str:
    """Replace VWA placeholders with actual URLs."""
    replace_map = {
        "__REDDIT__": getattr(env_config, "REDDIT", ""),
        "__SHOPPING__": getattr(env_config, "SHOPPING", ""),
        "__SHOPPING_ADMIN__": getattr(env_config, "SHOPPING_ADMIN", ""),
        "__GITLAB__": getattr(env_config, "GITLAB", ""),
        "__WIKIPEDIA__": getattr(env_config, "WIKIPEDIA", ""),
        "__MAP__": getattr(env_config, "MAP", ""),
        "__HOMEPAGE__": getattr(env_config, "HOMEPAGE", ""),
        "__CLASSIFIEDS__": getattr(env_config, "CLASSIFIEDS", ""),
    }
    for k, v in replace_map.items():
        if v:  # Only replace if value is set
            text = text.replace(k, v)
    return text


def prepare_task_config(task_raw: Dict[str, Any], output_dir: Path) -> Path:
    """
    Resolves placeholders in the task config and saves it to a file.
    Returns the path to the resolved config file.
    """
    # Deep copy to avoid modifying original
    task = copy.deepcopy(task_raw)

    # Recursively resolve strings in the dict
    def recursive_replace(obj):
        if isinstance(obj, str):
            return resolve_placeholders(obj)
        if isinstance(obj, list):
            return [recursive_replace(i) for i in obj]
        if isinstance(obj, dict):
            return {k: recursive_replace(v) for k, v in obj.items()}
        return obj

    task_resolved = recursive_replace(task)

    task_id = task_resolved["task_id"]
    config_dir = output_dir / "task_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / f"{task_id}.json"
    with open(config_path, "w") as f:
        json.dump(task_resolved, f, indent=2)

    return config_path


def save_episode_artifacts(
    output_dir: Path,
    task_id: int,
    trajectory: List[Any],
    summary: Dict[str, Any],
    step_logs: List[Dict[str, Any]],
):
    """Save trajectory, summary, and step logs."""
    # 1. Save Summary
    with open(output_dir / f"episode_{task_id}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 2. Save Step Logs (JSONL)
    with open(output_dir / f"episode_{task_id}_log.jsonl", "w") as f:
        for log in step_logs:
            f.write(json.dumps(log) + "\n")


def run_episode(
    env: VWAWrapper,
    agent: QwenApiAgent,
    task_config_path: Path,
    intent: str,
    max_steps: int,
    task_id: int,
    run_id: str,
    sites: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs a single episode and evaluates it."""

    # Reset Env
    obs, info = env.reset(config_file=str(task_config_path))

    trajectory = []
    # VWA expects the first element of trajectory to be state info
    trajectory.append({"observation": obs.raw, "info": info})

    step_logs = []
    total_reward = 0.0

    start_time = time.time()

    for step_idx in range(max_steps):
        step_start = time.time()

        # Agent Act
        if hasattr(agent, "step"):
            action_json, meta = agent.step(intent, obs)
        else:
            action_json = {"action_type": "wait"}
            meta = {}

        # Env Step
        next_obs, reward, terminated, truncated, next_info = env.step(action_json)

        # Record for VWA Trajectory
        if "raw_action" in next_info:
            trajectory.append(next_info["raw_action"])
        else:
            logger.warning("raw_action not found in info, using mock")
            trajectory.append(create_stop_action(""))

        # Record State
        trajectory.append({"observation": next_obs.raw, "info": next_info})

        # Logging
        latency = (time.time() - step_start) * 1000
        step_log = {
            "run_id": run_id,
            "task_id": task_id,
            "step": step_idx,
            "agent_action": action_json,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "latency_ms": latency,
        }
        step_logs.append(step_log)

        obs = next_obs
        total_reward += reward

        if terminated or truncated:
            break

    def ensure_stop_action(traj: List[Any]) -> None:
        if not traj:
            traj.append(create_stop_action(""))
            return

        last_item = traj[-1]
        if not (isinstance(last_item, dict) and "action_type" in last_item):
            traj.append(create_stop_action(""))
            return

        if last_item.get("action_type") != ActionTypes.STOP:
            traj.append(create_stop_action(""))
            return

        if "answer" not in last_item:
            last_item["answer"] = ""

    # Always ensure a STOP action for VWA eval consistency
    ensure_stop_action(trajectory)

    # Evaluation
    score = 0.0
    try:
        if env._env:
            evaluator = evaluator_router(task_config_path)
            score = evaluator(
                trajectory=trajectory,
                config_file=task_config_path,
                page=env._env.page,
            )
            logger.info(f"Task {task_id} Score: {score}")
        else:
            logger.warning("Env not initialized, skipping eval")
    except Exception as e:
        logger.error(f"Evaluation failed for task {task_id}: {e}")
        import traceback

        traceback.print_exc()

    duration = time.time() - start_time

    summary = {
        "task_id": task_id,
        "success": score == 1.0,
        "score": score,
        "steps": len(step_logs),
        "total_reward": total_reward,
        "duration_s": duration,
        "run_id": run_id,
    }

    return summary, trajectory, step_logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config (yaml)")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load Config
    cfg = load_config(args.config)

    # Overrides
    if args.shard_id is not None:
        cfg["run"]["shard_id"] = args.shard_id
    if args.num_shards is not None:
        cfg["run"]["num_shards"] = args.num_shards
    if args.resume:
        cfg["run"]["resume"] = True

    run_id = args.run_id or f"run_{int(time.time())}"
    output_dir = Path(cfg["run"]["output_dir"]) / "strong_only" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Run Meta
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "config": cfg,
                "timestamp": time.time(),
                "args": vars(args),
            },
            f,
            indent=2,
        )

    # Load Tasks
    task_config_file = cfg["task"]["config_file"]
    with open(task_config_file, "r") as f:
        all_tasks = json.load(f)

    # Filter by task_ids if specified
    if "task_ids" in cfg["task"]:
        target_ids = set(cfg["task"]["task_ids"])
        tasks_to_run = [t for t in all_tasks if t["task_id"] in target_ids]
    else:
        tasks_to_run = all_tasks

    # Sharding
    num_shards = cfg["run"]["num_shards"]
    shard_id = cfg["run"]["shard_id"]
    if num_shards > 1:
        tasks_to_run = [t for i, t in enumerate(tasks_to_run) if i % num_shards == shard_id]

    logger.info(f"Starting Run {run_id}. Tasks: {len(tasks_to_run)} (Shard {shard_id}/{num_shards})")

    # Initialize Env & Agent
    env_cfg = cfg["env"]
    wrapper = VWAWrapper(
        headless=env_cfg.get("headless", True),
        observation_type="accessibility_tree",
        viewport_width=env_cfg.get("viewport_width", 1280),
        viewport_height=env_cfg.get("viewport_height", 720),
        dry_run=env_cfg.get("dry_run", False),
    )

    logger.info("Initializing Qwen API Agent...")
    agent = QwenApiAgent(cfg)

    try:
        for task in tasks_to_run:
            task_id = task["task_id"]

            # Resume Check
            summary_path = output_dir / f"episode_{task_id}_summary.json"
            if cfg["run"]["resume"] and summary_path.exists():
                logger.info(f"Skipping Task {task_id} (Already completed)")
                continue

            logger.info(f"Running Task {task_id}")

            # Prepare Config
            task_config_path = prepare_task_config(task, output_dir)

            # Get Intent
            with open(task_config_path, "r") as f:
                task_c = json.load(f)
                intent = task_c["intent"]

            # Run
            try:
                summary, trajectory, step_logs = run_episode(
                    env=wrapper,
                    agent=agent,
                    task_config_path=task_config_path,
                    intent=intent,
                    max_steps=cfg.get("agent", {}).get("max_steps", 20),
                    task_id=task_id,
                    run_id=run_id,
                )

                # Save
                save_episode_artifacts(output_dir, task_id, trajectory, summary, step_logs)

            except Exception as e:
                logger.error(f"Error running task {task_id}: {e}")
                summary = {
                    "task_id": task_id,
                    "success": False,
                    "score": 0.0,
                    "error": str(e),
                    "run_id": run_id,
                }
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)

    finally:
        wrapper.close()
        logger.info("Run finished.")


if __name__ == "__main__":
    main()
