#!/bin/bash
# Run reviewer-requested benchmarks on AutoDL RTX 4090
# Usage: bash run_reviewer_benchmarks.sh [experiments]
# Examples:
#   bash run_reviewer_benchmarks.sh              # Run all experiments
#   bash run_reviewer_benchmarks.sh concurrent   # Run only concurrent
#   bash run_reviewer_benchmarks.sh "concurrent,baselines"  # Run specific experiments

set -e

EXPERIMENTS="${1:-all}"
OUTPUT="/root/autodl-tmp/reviewer_results.json"
MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "=================================================="
echo "RAGCache++ Reviewer Benchmarks"
echo "Experiments: $EXPERIMENTS"
echo "Output: $OUTPUT"
echo "=================================================="

# Ensure dependencies
pip install datasets --quiet 2>/dev/null || true

cd /root/autodl-tmp/ragcache_pp/vllm_integration

export TMPDIR=/root/autodl-tmp
export VLLM_USE_TRITON_FLASH_ATTN=0

python benchmark_reviewer.py \
  --model "$MODEL" \
  --num-docs 500 \
  --num-queries 200 \
  --top-k 5 \
  --max-model-len 4096 \
  --gpu-mem 0.90 \
  --enforce-eager \
  --overlap 0.6 \
  --experiments "$EXPERIMENTS" \
  --output "$OUTPUT"

echo ""
echo "Done! Results saved to $OUTPUT"
echo "Download with: scp -P PORT root@HOST:$OUTPUT ."
