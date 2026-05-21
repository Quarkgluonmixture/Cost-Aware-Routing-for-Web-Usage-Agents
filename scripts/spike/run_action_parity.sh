#!/bin/bash
# Orchestrate the action-parity spike end-to-end: HF (main .venv) -> vLLM
# (.venv-vllm, flashinfer-free) -> compare. Writes /tmp/spike/ap_report.txt and
# /tmp/spike/ap.DONE (Tier-1 marker for the remote monitor).
set -u
cd /home/ubuntu/workspace/p79 || exit 2
RUN=results/visualwebarena/phase1/B2_dom_classifieds_20260520/phase1_dom_router_0
N=${N:-40}
S=scripts/spike/spike_action_parity.py
rm -f /tmp/spike/ap.DONE /tmp/spike/ap.FAIL
trap 'echo "[ap] FAILED"; touch /tmp/spike/ap.FAIL' ERR

echo "[ap] $(date +%H:%M:%S) HF pass (main .venv)"
for M in qwen gemma; do
  echo "[ap]   hf $M"
  PYTORCH_NVML_BASED_CUDA_CHECK=1 .venv/bin/python "$S" --engine hf --model "$M" \
    --run-dir "$RUN" --n "$N" --max-new-tokens 256 \
    --out /tmp/spike/ap_hf_$M.jsonl > /tmp/spike/ap_hf_$M.log 2>&1
done

echo "[ap] $(date +%H:%M:%S) vLLM pass (.venv-vllm, FLASH_ATTN + no flashinfer sampler)"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
for M in qwen gemma; do
  echo "[ap]   vllm $M"
  .venv-vllm/bin/python "$S" --engine vllm --model "$M" \
    --paired /tmp/spike/ap_hf_$M.jsonl --max-new-tokens 256 --gpu-mem-util 0.5 \
    --out /tmp/spike/ap_vllm_$M.jsonl > /tmp/spike/ap_vllm_$M.log 2>&1
done

echo "[ap] $(date +%H:%M:%S) compare -> /tmp/spike/ap_report.txt"
{
  echo "================ ACTION-PARITY REPORT (HF eager vs vLLM) ================"
  echo "real dom AXTree inputs, N=$N stratified by length, same input_ids both engines"
  for M in qwen gemma; do
    echo; echo "### $M"
    .venv/bin/python "$S" --compare /tmp/spike/ap_hf_$M.jsonl /tmp/spike/ap_vllm_$M.jsonl
  done
} > /tmp/spike/ap_report.txt 2>&1

cat /tmp/spike/ap_report.txt
touch /tmp/spike/ap.DONE
echo "[ap] $(date +%H:%M:%S) DONE"
