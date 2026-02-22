import argparse
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_vwa_results(results_dir="results"):
    """Load VWA episode summary files from results directory."""
    data = []
    datasets = ["shopping", "reddit", "wikipedia", "classifieds"]
    
    for dataset in datasets:
        dataset_dir = Path(results_dir) / dataset
        if not dataset_dir.exists():
            continue
            
        # Find all run directories
        for run_dir in dataset_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            # Find all episode summary files
            summary_files = list(run_dir.glob("episode_*_summary.json"))
            
            for summary_file in summary_files:
                with open(summary_file, 'r') as fh:
                    episode_data = json.load(fh)
                    # Add dataset and run_id info
                    episode_data["dataset"] = dataset
                    episode_data["run_id"] = run_dir.name
                    data.append(episode_data)
    
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", help="Directory containing VWA results")
    parser.add_argument("--output_dir", default="results_summary", help="Directory to save summary outputs")
    args = parser.parse_args()
    
    # Load results
    data = load_vwa_results(args.results_dir)
    
    if not data:
        print(f"No summary files found in {args.results_dir}")
        return
    
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data in summary files.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute Overall Metrics
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    sr = df["success"].mean()
    avg_score = df["score"].mean()
    avg_steps = df["steps"].mean()
    avg_duration = df["duration_s"].mean()
    
    print(f"Total Episodes: {len(df)}")
    print(f"Success Rate: {sr:.2%}")
    print(f"Average Score: {avg_score:.3f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Duration: {avg_duration:.2f}s")
    
    # Compute Metrics per Dataset
    print("\n" + "=" * 60)
    print("RESULTS BY DATASET")
    print("=" * 60)
    
    dataset_stats = []
    for dataset in df["dataset"].unique():
        df_dataset = df[df["dataset"] == dataset]
        dataset_sr = df_dataset["success"].mean()
        dataset_score = df_dataset["score"].mean()
        dataset_steps = df_dataset["steps"].mean()
        dataset_duration = df_dataset["duration_s"].mean()
        
        print(f"\n{dataset.upper()}:")
        print(f"  Episodes: {len(df_dataset)}")
        print(f"  Success Rate: {dataset_sr:.2%}")
        print(f"  Average Score: {dataset_score:.3f}")
        print(f"  Average Steps: {dataset_steps:.2f}")
        print(f"  Average Duration: {dataset_duration:.2f}s")
        
        dataset_stats.append({
            "dataset": dataset,
            "episodes": len(df_dataset),
            "success_rate": dataset_sr,
            "avg_score": dataset_score,
            "avg_steps": dataset_steps,
            "avg_duration": dataset_duration
        })
    
    # Save CSV
    csv_path = output_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")
    
    # Save dataset stats
    stats_df = pd.DataFrame(dataset_stats)
    stats_path = output_dir / "dataset_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved dataset stats to {stats_path}")
    
    # Plots
    # 1. Success Rate by Dataset
    plt.figure(figsize=(10, 6))
    stats_df.plot(x="dataset", y="success_rate", kind="bar", color="steelblue", legend=False)
    plt.title("Success Rate by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = output_dir / "success_rate_by_dataset.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    # 2. Steps Distribution
    plt.figure(figsize=(10, 6))
    if len(df["steps"].unique()) > 1:
        bins = range(int(df["steps"].min()), int(df["steps"].max()) + 2, 1)
        plt.hist(df["steps"], bins=bins, alpha=0.7, edgecolor="black")
    else:
        plt.hist(df["steps"], alpha=0.7, edgecolor="black")
        
    plt.title("Steps per Episode")
    plt.xlabel("Steps")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = output_dir / "steps_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    # 3. Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["score"], bins=20, alpha=0.7, edgecolor="black")
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = output_dir / "score_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
