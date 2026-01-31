import json
import os
from p79.envs.vwa_wrapper import VWAWrapper

def main():
    # 1. Prepare a single config file
    raw_config_path = "external/visualwebarena/config_files/vwa/test_classifieds.raw.json"
    if not os.path.exists(raw_config_path):
        # Fallback if VWA repo structure changed
        print(f"Warning: {raw_config_path} not found. Trying to find any json in config_files...")
        # (Simplified for now, assuming it exists based on previous `ls`)
        return

    with open(raw_config_path, "r") as f:
        tasks = json.load(f)
    
    # Take the first task
    single_task = tasks[0]
    
    # Remove storage_state to avoid FileNotFoundError during smoke test
    if "storage_state" in single_task:
        print("Removing storage_state from config for smoke test.")
        del single_task["storage_state"]
        
    # Replace placeholders with valid URL
    if "start_url" in single_task:
        print(f"Replacing placeholder in start_url: {single_task['start_url']}")
        single_task["start_url"] = single_task["start_url"].replace("__CLASSIFIEDS__", "https://example.com")
        print(f"New start_url: {single_task['start_url']}")

    # We need to make sure start_url is valid or mocked if we want to avoid 404s/timeouts
    # But for smoke test, even if it fails to load the page, reset might succeed or throw.
    # The dummy env vars map __CLASSIFIEDS__ to localhost:9999.
    # So it will try to open http://localhost:9999/...
    # This might fail in playwright navigation.
    
    # To avoid navigation error crashing the script, we might want to catch it?
    # Or just let it run. If localhost:9999 is not listening, playwright might error.
    # But let's see.
    
    temp_config_path = "temp_smoke_config.json"
    with open(temp_config_path, "w") as f:
        json.dump(single_task, f)

    print(f"Created temp config: {temp_config_path}")

    # 2. Init Env
    env = VWAWrapper(
        headless=True,
        observation_type="accessibility_tree", # VWA supports this
        dry_run=False,
    )

    print("Initializing env (lazy)...")

    try:
        # 3. Reset
        print("Calling reset...")
        obs, info = env.reset(temp_config_path)
        print("Reset successful!")
        print("Obs text len:", len(obs.text))
        print("Info keys:", list(info.keys()))
        if obs.image is not None:
             print("Obs has image!")
        
        # 4. Step
        print("Calling step...")
        # Just click something random, we expect it might fail or do nothing, but return obs
        action = {"action_type": "click", "element_id": 0} 
        obs, reward, terminated, truncated, info = env.step(action)
        print("Step successful!")
        print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print("Cleaned up temp config.")

if __name__ == "__main__":
    main()
