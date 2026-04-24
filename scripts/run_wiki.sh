#!/bin/bash
set -e
export TMPDIR=/tmp
export VLLM_USE_TRITON_FLASH_ATTN=0
cd /root/ragcache_pp_project
mkdir -p /root/ragcache_pp_project/results
QWEN7B=/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
echo "=== Wikipedia corpus run at $(date) ==="
python ragcache_pp/vllm_integration/benchmark_wiki_corpus.py \
    --model "$QWEN7B" --max-model-len 4096 --gpu-mem 0.90 --enforce-eager \
    --output /root/ragcache_pp_project/results/wiki_corpus_results.json
echo "=== Done at $(date) ==="
