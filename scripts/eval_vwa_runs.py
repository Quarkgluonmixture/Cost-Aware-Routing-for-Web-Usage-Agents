import argparse
import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="Directory containing episode summaries")
    parser.add_argument("--output_csv", type=str, default="eval_results.csv", help="Output CSV filename")
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    summary_files = list(result_dir.glob("episode_*_summary.json"))
    
    if not summary_files:
        logger.warning(f"No summary files found in {result_dir}")
        return

    results = []
    for f in summary_files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
                results.append(data)
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")
            
    df = pd.DataFrame(results)
    
    # Calculate Metrics
    total_tasks = len(df)
    success_rate = df["success"].mean() if "success" in df.columns else 0.0
    avg_score = df["score"].mean() if "score" in df.columns else 0.0
    avg_steps = df["steps"].mean() if "steps" in df.columns else 0.0
    
    logger.info("="*30)
    logger.info(f"Evaluation Summary for {result_dir}")
    logger.info("="*30)
    logger.info(f"Total Episodes: {total_tasks}")
    logger.info(f"Success Rate:   {success_rate:.2%}")
    logger.info(f"Average Score:  {avg_score:.4f}")
    logger.info(f"Average Steps:  {avg_steps:.2f}")
    logger.info("="*30)
    
    # Save CSV
    output_path = result_dir / args.output_csv
    df.to_csv(output_path, index=False)
    logger.info(f"Detailed results saved to {output_path}")

if __name__ == "__main__":
    main()
