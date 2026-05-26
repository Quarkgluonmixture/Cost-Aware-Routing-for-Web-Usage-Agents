#!/bin/bash -l
# Tier B activation patching — layer-by-layer find id-effect emergence layer
# 承 Tier A (笔记 §298): B1 dense 14/133=10.5% pure id-channel step-0 flip.
# 本 qsub 对 14 flip task 跑 activation patching layer sweep, 找哪层 patching 翻转决策。
#
# Submission (from Myriad login node):
#   cd ~/Scratch/p79 && mkdir -p logs && qsub scripts/queues/qsub_tier_b_id_patching_myriad.sh
#
# Monitor:
#   qstat -u $USER
#   tail -f ~/Scratch/p79/logs/qsub_tier_b_id_patching.*.out
#
# Output: results/tier_b_id_patching_<timestamp>/tier_b_task_<tid>.json + tier_b_summary.json

#$ -l h_rt=4:0:0
#$ -l mem=48G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N tier_b_id_patching
#$ -o /home/ucab352/Scratch/p79/logs/qsub_tier_b_id_patching.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_tier_b_id_patching.$JOB_ID.err
#$ -j n

mkdir -p /home/ucab352/Scratch/p79/logs

set -euo pipefail
REPO_DIR="/home/ucab352/Scratch/p79"
cd "$REPO_DIR"

echo "[$(date '+%H:%M:%S')] Job $JOB_ID start on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Myriad pytorch 2.1.0/gpu module (auto-loads python/3.9.6 + cuda/11.8 + cudnn + gcc-libs/10.2.0)
module unload gcc-libs python python3 2>/dev/null || true
module load pytorch/2.1.0/gpu

# pip user-site (avoid Home quota)
export PYTHONUSERBASE="$HOME/Scratch/python_user"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:${PYTHONPATH:-}"

# HF offline (compute nodes firewalled, use cached model)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# CUDA workaround (defensive, mostly DGX GB10 — A100 unaffected but harmless)
export PYTORCH_NVML_BASED_CUDA_CHECK=1

# Tier A paths (Myriad-side, tar-piped from DGX 2026-05-26)
export P79_TIER_A_ARCH="$REPO_DIR/results/repro_replicates/B0_dom_classifieds_R31194_clean_replicate/phase1_dom_router_0"
export P79_TIER_A_CURR="$REPO_DIR/results/visualwebarena/phase1/B0_dom_classifieds_20260525_194618_553890342_530647_R21557/phase1_dom_router_0"

# Output dir (timestamped per-run)
export P79_TIER_B_OUT="$REPO_DIR/results/tier_b_id_patching_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$P79_TIER_B_OUT"

echo "[$(date '+%H:%M:%S')] Python: $(python3 --version)"
echo "[$(date '+%H:%M:%S')] Repo HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo N/A)"
echo "[$(date '+%H:%M:%S')] OUT: $P79_TIER_B_OUT"

# 14 flip task (Tier A 已坐实 B1 翻转, 笔记 §298)
TASKS="${TIER_B_TASKS:-10 11 12 16 17 59 64 92 93 94 107 108 118 125}"

python3 scripts/analysis/tier_b_id_patching.py $TASKS

echo "[$(date '+%H:%M:%S')] Tier B done; outputs in $P79_TIER_B_OUT"
ls "$P79_TIER_B_OUT"
