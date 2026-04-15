#!/bin/bash
set -e
export TMPDIR=/tmp
export VLLM_USE_TRITON_FLASH_ATTN=0
export HF_ENDPOINT=https://hf-mirror.com
cd /root/ragcache_pp_project
QWEN7B=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
echo "=== Round 7 (multihop + sensitivity) at $(date) ==="
python ragcache_pp/vllm_integration/benchmark_round6.py \
    --model "$QWEN7B" --max-model-len 4096 --gpu-mem 0.90 --enforce-eager \
    --experiments multihop_quality,sensitivity \
    --output /root/ragcache_pp_project/round7_results.json
echo "=== Done at $(date) ==="
