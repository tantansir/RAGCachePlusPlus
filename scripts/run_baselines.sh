#!/bin/bash
set -e
export TMPDIR=/tmp
export VLLM_USE_TRITON_FLASH_ATTN=0
export HF_ENDPOINT=https://hf-mirror.com
cd /root/ragcache_pp_project
mkdir -p /root/ragcache_pp_project/results
QWEN7B=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
echo "=== Baselines single-run at $(date) ==="
python ragcache_pp/vllm_integration/benchmark_baselines_rerun.py \
    --model "$QWEN7B" --max-model-len 4096 --gpu-mem 0.90 --enforce-eager \
    --num-docs 500 --num-queries 200 --top-k 5 --overlap 0.6 \
    --output /root/ragcache_pp_project/results/baselines_single_run.json
echo "=== Done at $(date) ==="
