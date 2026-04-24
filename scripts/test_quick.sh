#!/bin/bash
#SBATCH --job-name=quick_test
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=00:10:00
#SBATCH --mem=60G
#SBATCH --output=quick_test_%j.out
#SBATCH --error=quick_test_%j.err

source /ocean/projects/cis260009p/ktan4/final_project/.venv/bin/activate
export PYTHONPATH=/ocean/projects/cis260009p/ktan4/final_project
export HF_HOME=/ocean/projects/cis260009p/ktan4/final_project/.hf_cache
export VLLM_USE_TRITON_FLASH_ATTN=0

python -c "
import triton; print('triton=' + triton.__version__)
import os; print('VLLM_USE_TRITON_FLASH_ATTN=' + os.environ.get('VLLM_USE_TRITON_FLASH_ATTN','unset'))
from vllm import LLM, SamplingParams
import time

print('Loading Qwen2.5-7B...')
t0 = time.time()
llm = LLM(
    model='Qwen/Qwen2.5-7B-Instruct',
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    enable_prefix_caching=True,
    trust_remote_code=True,
    enforce_eager=True,
    dtype='half',
)
print('Loaded in %.1fs' % (time.time()-t0))

params = SamplingParams(max_tokens=1, temperature=0.0)
for i in range(10):
    t0 = time.time()
    out = llm.generate(['Hello world test query number %d. ' % i * 50], params)
    print('Query %d: %.1fms' % (i, (time.time()-t0)*1000))

print('SUCCESS: V100 + APC test passed!')
"
echo "Exit: $?"
