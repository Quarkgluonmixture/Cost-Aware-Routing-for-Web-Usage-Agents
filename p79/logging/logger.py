import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class EpisodeLogger:
    def __init__(self, log_dir: str, episode_id: str, task_text: str):
        self.log_dir = log_dir
        self.episode_id = episode_id
        self.task_text = task_text
        self.steps = []
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Files
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"episode_{episode_id}_{timestamp_str}.jsonl")
        self.summary_file_path = os.path.join(log_dir, f"summary_{episode_id}_{timestamp_str}.json")

    def log_step(self, 
                 step_id: int,
                 obs_meta: Dict[str, Any],
                 action_info: Dict[str, Any],
                 latency_info: Dict[str, float],
                 token_info: Dict[str, int],
                 cost_info: Dict[str, float],
                 done: bool,
                 success: bool = False,
                 energy_info: Optional[Dict[str, float]] = None):
        
        entry = {
            "episode_id": self.episode_id,
            "step_id": step_id,
            "timestamp": datetime.now().isoformat(),
            "task_text": self.task_text,
            "obs_meta": obs_meta,
            "action": action_info, # includes json, valid, repair info
            "latency_ms": latency_info,
            "tokens": token_info,
            "cost_usd": cost_info,
            "energy": energy_info or {"kwh": 0.0, "co2e_kg": 0.0},
            "done": done,
            "success": success
        }
        
        self.steps.append(entry)
        
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save_summary(self):
        # Calculate totals
        total_steps = len(self.steps)
        total_latency = sum(s["latency_ms"].get("total", 0) for s in self.steps)
        total_input_tokens = sum(s["tokens"].get("input", 0) for s in self.steps)
        total_output_tokens = sum(s["tokens"].get("output", 0) for s in self.steps)
        total_cost = sum(s["cost_usd"].get("total", 0) for s in self.steps)
        
        # Check if success in last step
        is_success = self.steps[-1]["success"] if self.steps else False

        summary = {
            "episode_id": self.episode_id,
            "task_text": self.task_text,
            "total_steps": total_steps,
            "success": is_success,
            "total_latency_ms": total_latency,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": total_cost,
            "log_file": self.log_file_path
        }

        with open(self.summary_file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        return summary
