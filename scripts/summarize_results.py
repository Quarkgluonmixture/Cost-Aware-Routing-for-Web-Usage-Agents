import argparse
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="logs")
    args = parser.parse_args()
    
    summary_files = glob.glob(os.path.join(args.log_dir, "summary_*.json"))
    
    if not summary_files:
        print("No summary files found.")
        return
        
    data = []
    for f in summary_files:
        with open(f, 'r') as fh:
            data.append(json.load(fh))
            
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data in summary files.")
        return
    
    # Compute Metrics
    sr = df["success"].mean()
    avg_steps = df["total_steps"].mean()
    p95_latency = df["total_latency_ms"].quantile(0.95)
    
    print(f"Success Rate: {sr:.2%}")
    print(f"Avg Steps: {avg_steps:.2f}")
    print(f"P95 Latency: {p95_latency:.2f} ms")
    
    # Save CSV
    df.to_csv("results_summary.csv", index=False)
    print("Saved results_summary.csv")
    
    # Plots
    # 1. Steps Distribution
    plt.figure(figsize=(10, 6))
    if len(df["total_steps"].unique()) > 1:
        bins = range(int(df["total_steps"].min()), int(df["total_steps"].max()) + 2, 1)
        plt.hist(df["total_steps"], bins=bins, alpha=0.7)
    else:
        plt.hist(df["total_steps"], alpha=0.7)
        
    plt.title("Steps per Episode")
    plt.xlabel("Steps")
    plt.ylabel("Count")
    plt.savefig("steps_hist.png")
    plt.close()
    
    # 2. Latency vs Tokens
    plt.figure(figsize=(10, 6))
    total_tokens = df["total_input_tokens"] + df["total_output_tokens"]
    plt.scatter(total_tokens, df["total_latency_ms"], alpha=0.6)
    plt.title("Latency vs Total Tokens")
    plt.xlabel("Tokens")
    plt.ylabel("Latency (ms)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("latency_tokens.png")
    plt.close()
    
    print("Saved plots: steps_hist.png, latency_tokens.png")

if __name__ == "__main__":
    main()
