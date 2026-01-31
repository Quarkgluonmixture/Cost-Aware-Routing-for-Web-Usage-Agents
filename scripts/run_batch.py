import argparse
import json
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Run a batch of tasks")
    parser.add_argument("--tasks_file", type=str, default="data/tasks/sample_tasks.jsonl")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.tasks_file):
        print(f"Tasks file not found: {args.tasks_file}")
        return

    with open(args.tasks_file, 'r') as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Found {len(tasks)} tasks.")

    for i, task in enumerate(tasks):
        print(f"\n=== Running Task {i+1}/{len(tasks)}: {task['task_id']} ===")
        
        # Determine python executable
        python_exe = sys.executable
        
        cmd = [
            python_exe, "scripts/run_episode.py",
            "--config", args.config,
            "--task_id", str(task['task_id']),
            "--instruction", task['instruction']
        ]
        
        try:
            # Pass environment variables
            env = os.environ.copy()
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Task {task['task_id']} failed with error: {e}")
        except KeyboardInterrupt:
            print("Batch run interrupted.")
            break

if __name__ == "__main__":
    main()
