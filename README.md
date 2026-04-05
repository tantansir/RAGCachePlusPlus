# RAGCache++

**Cache-Aware Document Ordering for Low-Latency RAG Serving**

CMU 11-868 Large Language Model Systems (Spring 2026) Final Project

## Overview

RAGCache++ is a lightweight prompt-level optimization that reorders documents in RAG prompts to maximize prefix sharing with vLLM's Automatic Prefix Caching (APC). It requires **zero modifications** to the serving engine — operating entirely at the prompt-construction layer.

**Key insight:** vLLM's APC uses chained hashes, so a single-token prefix mismatch invalidates all subsequent cached blocks. By reordering documents so that shared subsets appear first in a consistent order, document-level overlap is converted into token-level prefix hits.

### How it works

1. A **knowledge tree** (trie indexed by document-ID sequences) tracks which orderings are currently in the KV cache.
2. For each new request, a **greedy algorithm** walks the trie to find the longest cached prefix, places those documents first, and appends remaining documents in retrieval-rank order.
3. The reordered prompt is sent to vLLM, which sees a longer prefix match and skips more prefill computation.

## Results

Evaluated on RTX 4060 Ti (Qwen2.5-1.5B) and RTX 4090 (Qwen2.5-7B):

| Metric | Value |
|--------|-------|
| Median TTFT reduction | **20–33%** |
| Prefill computation saved (p50) | **66%** |
| Greedy vs. oracle (exhaustive search) | **97.5%** optimal |
| Overhead | < 0.13%, zero GPU memory cost |
| Concurrent load (batch 1–8) | 7–10% improvement |
| Cache pressure robustness (gpu_mem 0.78–0.90) | Stable 29–31% |
| Statistical significance | 3 seeds, 95% CI excludes zero |

## Project Structure

```
ragcache_pp/                    # Core library
├── cache/
│   ├── knowledge_tree.py       # Trie for tracking cached document orderings
│   ├── cache_manager.py        # Cache management with PGDSF eviction
│   ├── pgdsf_policy.py         # Priority-based eviction policy
│   └── spatial_index.py        # Geospatial index for location-aware RAG
├── vllm_integration/
│   ├── prompt_builder.py       # Prompt construction with document reordering
│   ├── benchmark_real.py       # Core benchmark: overlap sweep, TTFT measurement
│   ├── benchmark_reviewer.py   # Extended benchmarks: concurrent, baselines, eviction, multi-seed
│   ├── benchmark_systems.py    # Systems benchmarks: throughput, memory, profiling
│   └── benchmark_hotpotqa.py   # Quality evaluation on HotpotQA
├── evaluation/
│   ├── benchmark.py            # Simulation-mode benchmarks
│   └── workload_generator.py   # Synthetic RAG workload generation
├── serving/
│   └── rag_controller.py       # RAG serving controller
└── config.py                   # Configuration

paper/                          # LaTeX source (MLSys 2024 template)
├── main.tex                    # Paper source
├── main.pdf                    # Compiled paper
├── references.bib              # Bibliography
├── figures/                    # Generated figures
└── gen_figures.py              # Figure generation script

*.json                          # Experiment results
├── reviewer_results.json       # Concurrent, baselines, multi-seed, eviction results
├── systems_benchmark_4090.json # Systems benchmark results (RTX 4090)
├── benchmark_4090_results.json # Core benchmark results (RTX 4090)
├── benchmark_results_v5.json   # Overlap sweep results (RTX 4060 Ti)
└── hotpotqa_results.json       # HotpotQA quality evaluation

*.sh                            # Run scripts
├── run_reviewer_benchmarks.sh  # Run reviewer-requested experiments
├── run_vllm_experiment.sh      # Run core vLLM experiments
└── test_quick.sh               # Quick smoke test
```

## Quick Start

### Simulation mode (no GPU)

```python
from ragcache_pp.cache.knowledge_tree import KnowledgeTree

tree = KnowledgeTree()
# Insert a cached document ordering
tree.insert(["doc_1", "doc_3", "doc_5"])
# Find longest cached prefix for a new query
match = tree.prefix_match(["doc_1", "doc_3", "doc_7", "doc_5"])
# match.prefix_ids = ["doc_1", "doc_3"], match length = 2
```

### GPU mode (vLLM)

```bash
# Install dependencies
pip install vllm torch transformers datasets numpy

# Run core benchmarks
python -m ragcache_pp.vllm_integration.benchmark_real \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-queries 200 --top-k 5

# Run reviewer benchmarks (concurrent, baselines, eviction, multi-seed)
python -m ragcache_pp.vllm_integration.benchmark_reviewer \
    --model Qwen/Qwen2.5-7B-Instruct
```

## Citation

```bibtex
@misc{ragcacheplusplus2026,
  title   = {RAGCache++: Cache-Aware Document Ordering for Low-Latency RAG Serving},
  author  = {Kaizhen Tang},
  year    = {2026},
  note    = {CMU 11-868 LLM Systems, Spring 2026}
}
```
