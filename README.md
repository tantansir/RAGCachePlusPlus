# RAGCache++

**Cache-Aware Document Ordering for Low-Latency RAG Serving**

CMU 11-868 Large Language Model Systems (Spring 2026) Final Project

Authors: Kaizhen Tan, Rong Gu, Mingyuan Li

## Overview

RAGCache++ is a lightweight prompt-level optimization that reorders documents in RAG prompts to maximize prefix sharing with vLLM's Automatic Prefix Caching (APC). It requires **zero modifications** to the serving engine — operating entirely at the prompt-construction layer.

**Key insight:** vLLM's APC uses chained hashes, so a single-token prefix mismatch invalidates all subsequent cached blocks. By reordering documents so that shared subsets appear first in a consistent order, document-level overlap is converted into token-level prefix hits.

### How it works

1. A **knowledge tree** (trie indexed by document-ID sequences) tracks which orderings are currently in the KV cache.
2. For each new request, a **greedy algorithm** walks the trie to find the longest cached prefix, places those documents first, and appends remaining documents in retrieval-rank order.
3. A **TTFT-based feedback loop** infers vLLM's actual cache state and prunes stale trie paths when the predicted reuse does not match the observed cold-start TTFT.
4. The reordered prompt is sent to vLLM, which sees a longer prefix match and skips more prefill computation.

## Results

Evaluated on RTX 4060 Ti (Qwen2.5-1.5B), RTX 4090 (Qwen2.5-7B), plus Qwen2.5-0.5B and Microsoft Phi-3-mini-4k-instruct (3.8B) for cross-architecture validation.

| Metric | Value |
|--------|-------|
| Median TTFT reduction | **20–33%** (synthetic bursty) |
| Prefill computation saved (p50) | **66%** |
| Greedy vs. oracle (exhaustive search) | **99%** of oracle TTFT reduction (prefix-length optimality ratio 0.997) |
| Overhead | < 0.13%, zero GPU memory cost |
| Concurrent load (batch 1–8) | 7–10% improvement |
| Cache pressure robustness (gpu_mem 0.78–0.90) | Stable 29–31% |
| Real Wikipedia corpus (1000 passages, 20 topics, TF-IDF) | **+26.8%** p50, **+45.9%** vs no-cache |
| Cross-architecture (Phi-3-mini) | 18.3% |
| Multi-hop reasoning quality | ΔEM = 0, ΔF1 = 0 (reordering is safe) |
| Feedback-loop trie-reality correlation | Pearson r = 0.97 |
| Statistical significance | 3 seeds, 95% CI excludes zero |

## Project Structure

```
ragcache_pp/                           # Full integrated system (~2,200 LOC)
├── cache/
│   ├── knowledge_tree.py              # Trie for tracking cached document orderings
│   ├── cache_manager.py               # Multi-tier cache manager (GPU / host)
│   ├── pgdsf_policy.py                # PGDSF priority eviction policy
│   └── spatial_index.py               # Geohash index for spatial RAG
├── vllm_integration/
│   ├── serving.py                     # VLLMCacheProxy + CacheStateFeedback + FastAPI middleware
│   ├── prompt_builder.py              # Greedy ordering + prompt assembly
│   ├── benchmark_real.py              # Core benchmark: overlap sweep, TTFT measurement
│   ├── benchmark_reviewer.py          # Extended: concurrent, baselines, eviction, multi-seed
│   ├── benchmark_systems.py           # Systems: throughput, memory, profiling
│   ├── benchmark_final.py             # NQ-Open + quality + E2E + cache validation
│   ├── benchmark_round3.py            # Proxy baselines, mixed workload, second model
│   ├── benchmark_round4.py            # Online cold-start, quality embedded
│   ├── benchmark_round5.py            # Cross-architecture (Phi-3), optimality gap
│   ├── benchmark_round6.py            # MS MARCO real, multi-hop, sensitivity, freq vs trie
│   └── benchmark_wiki_corpus.py       # Wikipedia 20-topic real-corpus TF-IDF benchmark
├── evaluation/
│   ├── benchmark.py                   # Simulation-mode benchmarks
│   └── workload_generator.py          # Synthetic, geospatial, NQ, mixed-topic traces
├── serving/
│   └── rag_controller.py              # RAG serving controller with pipelining
└── config.py                          # Configuration

paper/                                 # LaTeX source (MLSys 2024 template)
├── paper.tex                 # Paper source (14 pages)
├── paper.pdf                 # Compiled paper
├── references.bib                     # Bibliography
├── figures/                           # Generated figures
└── gen_figures.py                     # Figure generation script

*.json                                 # Experiment results
├── benchmark_4090_results.json        # Main RTX 4090 TTFT (Table 1)
├── benchmark_isolated_results.json    # Main RTX 4060 Ti TTFT (Table 1)
├── overlap_sweep_results.json         # Overlap sensitivity (Figure 3, Appx B)
├── systems_benchmark_4090.json        # Profiling, throughput, cache_eff, concurrent, pipelining
├── reviewer_results.json              # Recency, Oracle, multi-seed, eviction
├── round3_results.json                # Proxy baselines, mixed workload
├── round4_results.json                # Qwen-0.5B, online cold-start
├── round5_results.json                # Phi-3 cross-architecture, optimality ratio 0.997
├── round6_results.json                # MS MARCO real corpus, freq vs trie
├── round7_results.json                # Multi-hop quality, top-k sensitivity
├── wiki_corpus_results.json           # Wikipedia 20-topic real corpus
├── final_results.json                 # NQ-Open, quality=EM 1.000, E2E pipeline, cache validation
└── integrated_results(2).json         # VLLMCacheProxy feedback loop validation

*.sh                                   # Run scripts
├── run_final_benchmarks.sh            # Core benchmarks
├── run_reviewer_benchmarks.sh         # Reviewer-requested experiments
├── run_round6.sh                      # Round 6: MS MARCO + multi-hop + sensitivity
├── run_round7.sh                      # Round 7: multi-hop + sensitivity (re-run)
├── run_wiki.sh                        # Wikipedia real-corpus benchmark
└── run_vllm_experiment.sh             # Core vLLM experiments

RAGCachePP_Poster.pptx                 # Conference-style poster
RAGCachePP_Presentation.pptx           # Final presentation slides
speaker_notes.md                       # Presentation script (12–15 min)
```

## Quick Start

### Simulation mode (no GPU)

```python
from ragcache_pp.cache.knowledge_tree import KnowledgeTree, KVCacheMetadata

tree = KnowledgeTree()
metas = [KVCacheMetadata(doc_id=d, num_tokens=200, num_blocks=13, tier="gpu")
         for d in ["doc_1", "doc_3", "doc_5"]]
tree.insert(["doc_1", "doc_3", "doc_5"], metas)

matched, length = tree.prefix_match(["doc_1", "doc_3", "doc_7", "doc_5"])
# length = 2 (doc_1, doc_3 matched)
```

### Integrated vLLM serving

```python
from ragcache_pp.vllm_integration.serving import VLLMCacheProxy, RAGCacheMiddleware

proxy = VLLMCacheProxy(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_mem=0.90, max_model_len=4096,
    enable_feedback=True,
)
app = RAGCacheMiddleware(proxy).create_app()   # FastAPI: POST /v1/rag/generate
```

### Run benchmarks (GPU)

```bash
pip install vllm torch transformers datasets scikit-learn numpy

# Core benchmark (Main Results table)
bash run_final_benchmarks.sh

# Reviewer-requested: concurrent, baselines, multi-seed
bash run_reviewer_benchmarks.sh

# Multi-hop quality + top-k sensitivity
bash run_round7.sh

# Wikipedia real-corpus TF-IDF benchmark
bash run_wiki.sh
```

## Paper

`paper/paper.pdf` — 14 pages, MLSys 2024 template. See the paper for full experimental details, related-work discussion, and system-level analysis.

## Citation

```bibtex
@misc{ragcacheplusplus2026,
  title   = {RAGCache++: Cache-Aware Document Ordering for Low-Latency RAG Serving},
  author  = {Kaizhen Tan and Rong Gu and Mingyuan Li},
  year    = {2026},
  note    = {CMU 11-868 LLM Systems, Spring 2026},
  url     = {https://github.com/tantansir/RAGCachePlusPlus}
}
```
