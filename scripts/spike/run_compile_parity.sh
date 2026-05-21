#!/bin/bash
# torch.compile vs HF-eager action parity. Reuses the SAME 40 real dom steps as
# the vLLM run (collect_steps is deterministic per run-dir+N), so apc_hf_<M>
# (compile) is directly comparable to ap_hf_<M> (eager) by idx. gemma compile
# is expected to hit transformers #42440 — handled gracefully.
set -u
cd /home/ubuntu/workspace/p79 || exit 2
RUN=results/visualwebarena/phase1/B2_dom_classifieds_20260520/phase1_dom_router_0
N=${N:-40}
S=scripts/spike/spike_action_parity.py
rm -f /tmp/spike/apc.DONE /tmp/spike/apc.FAIL

for M in qwen gemma; do
  echo "[apc] $(date +%H:%M:%S) compile $M (first steps compile, may be slow)"
  PYTORCH_NVML_BASED_CUDA_CHECK=1 .venv/bin/python "$S" --engine hf --model "$M" --compile \
    --run-dir "$RUN" --n "$N" --max-new-tokens 256 \
    --out /tmp/spike/apc_hf_$M.jsonl > /tmp/spike/apc_hf_$M.log 2>&1 \
    || echo "[apc] $M compile FAILED (see /tmp/spike/apc_hf_$M.log)"
done

{
  echo "============ torch.compile vs HF-eager ACTION PARITY ============"
  echo "same 40 real dom steps as vLLM run; eager=ap_hf_<M> compile=apc_hf_<M>"
  for M in qwen gemma; do
    echo; echo "### $M"
    if [ -s /tmp/spike/apc_hf_$M.jsonl ]; then
      .venv/bin/python "$S" --compare /tmp/spike/ap_hf_$M.jsonl /tmp/spike/apc_hf_$M.jsonl
    else
      echo "  compile produced NO output (likely transformers #42440 for gemma)"
      tail -6 /tmp/spike/apc_hf_$M.log 2>/dev/null
    fi
  done
} > /tmp/spike/apc_report.txt 2>&1

cat /tmp/spike/apc_report.txt
touch /tmp/spike/apc.DONE
echo "[apc] $(date +%H:%M:%S) DONE"
