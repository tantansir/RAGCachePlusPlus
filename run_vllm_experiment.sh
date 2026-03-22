#!/bin/bash
#SBATCH --job-name=ragcache_vllm
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=02:00:00
#SBATCH --mem=60G
#SBATCH --output=vllm_bench_%j.out
#SBATCH --error=vllm_bench_%j.err

set -uo pipefail

module load anaconda3/2024.10-1
module load cuda/12.4.0

PROJECT=/ocean/projects/cis260009p/ktan4/final_project
source $PROJECT/.venv/bin/activate
export PYTHONPATH=$PROJECT
export HF_HOME=$PROJECT/.hf_cache
export TRANSFORMERS_CACHE=$PROJECT/.hf_cache

cd $PROJECT
export VLLM_USE_TRITON_FLASH_ATTN=0
# Force triton to use system ptxas instead of built-in LLVM backend
# to avoid "Failed to compute parent layout for slice layout" on V100
export TRITON_PTXAS_PATH=/opt/packages/cuda/v12.4.0/bin/ptxas

echo "============================================================"
echo "RAGCache++ Real GPU Benchmark"
echo "============================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Python: $(python --version)"
echo "TRITON_PTXAS_PATH=$TRITON_PTXAS_PATH"
nvidia-smi
echo "============================================================"

python -m ragcache_pp.vllm_integration.benchmark_real \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-docs 500 \
    --num-queries 200 \
    --top-k 5 \
    --output benchmark_v100_results.json \
    --gpu-mem 0.90 \
    --max-model-len 4096 \
    --dtype half \
    --enforce-eager

echo "============================================================"
echo "Exit code: $?"
echo "Benchmark Complete: $(date)"
echo "Final results:"
cat benchmark_v100_results.json 2>/dev/null || echo "(no results file)"
echo "============================================================"
