#!/bin/bash
# Run final benchmarks on AutoDL RTX 4090
# Addresses W1 (real workload), W3 (quality), W4 (E2E pipeline), W5 (cache validation)

set -e

export TMPDIR=/tmp
export VLLM_USE_TRITON_FLASH_ATTN=0
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# Install dependencies
pip install datasets scikit-learn scipy 2>/dev/null || true

cd /root/ragcache_pp_project
mkdir -p /root/ragcache_pp_project/results

echo "=== Starting final benchmarks at $(date) ==="

# Run all 4 experiments
python ragcache_pp/vllm_integration/benchmark_final.py \
    --model /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/$(ls /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/ | head -1) \
    --max-model-len 4096 \
    --gpu-mem 0.90 \
    --enforce-eager \
    --experiments all \
    --output /root/ragcache_pp_project/results/final_results.json

echo "=== All experiments completed at $(date) ==="
echo "Results saved to /root/ragcache_pp_project/results/final_results.json"
