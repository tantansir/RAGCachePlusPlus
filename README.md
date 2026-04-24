# RAGCache++

Cache-aware document ordering for low-latency RAG serving.

CMU 11-868 Large Language Model Systems, Spring 2026 final project.

Authors: Kaizhen Tan, Rong Gu, Mingyuan Li

## Overview

RAGCache++ is a prompt-layer optimization for RAG serving with vLLM Automatic Prefix Caching (APC). It reorders retrieved documents so repeated document subsets appear earlier in prompts, which turns document overlap into longer token-prefix cache hits. The system does not modify the serving engine.

The key idea is simple: vLLM APC uses chained hashes, so one early token mismatch can invalidate reuse for all later cached blocks. RAGCache++ tracks cached document orderings with a trie, greedily chooses the longest cached prefix for each request, then appends the remaining documents in retrieval-rank order.

## Results

Evaluated on RTX 4060 Ti and RTX 4090 with Qwen2.5 models, plus Phi-3-mini for cross-architecture validation.

| Metric | Result |
| --- | --- |
| Median TTFT reduction | 20-23% on synthetic bursty workloads |
| Prefill computation saved | 66% p50 |
| Greedy vs. oracle | 99% of oracle TTFT reduction |
| Overhead | < 0.13%, no GPU memory cost |
| Real Wikipedia corpus | +26.8% p50, +45.9% vs. no-cache |
| Cross-architecture validation | 18.3% improvement on Phi-3-mini |
| Feedback-loop validation | Pearson r = 0.97 between predicted and observed reuse |

## Repository Structure

```text
.
├── deliverables/                 # Final submitted artifacts
│   ├── final_report.pdf
│   ├── final_poster.pptx
│   └── speaker_notes.md
├── ragcache_pp/                  # Python package
│   ├── cache/
│   ├── evaluation/
│   ├── retrieval/
│   ├── serving/
│   └── vllm_integration/
├── results/                      # Saved benchmark outputs
├── scripts/                      # Benchmark and cluster run scripts
├── requirements.txt
└── README.md
```

## Quick Start

Install the base Python dependencies:

```bash
pip install -r requirements.txt
```

For vLLM experiments, install the serving stack expected by your GPU environment:

```bash
pip install vllm torch transformers datasets scikit-learn numpy scipy
```

Simulation-mode usage:

```python
from ragcache_pp.cache.knowledge_tree import KnowledgeTree, KVCacheMetadata

tree = KnowledgeTree()
metas = [
    KVCacheMetadata(doc_id=d, num_tokens=200, num_blocks=13, tier="gpu")
    for d in ["doc_1", "doc_3", "doc_5"]
]
tree.insert(["doc_1", "doc_3", "doc_5"], metas)

matched, length = tree.prefix_match(["doc_1", "doc_3", "doc_7", "doc_5"])
```

Integrated vLLM serving:

```python
from ragcache_pp.vllm_integration.serving import VLLMCacheProxy, RAGCacheMiddleware

proxy = VLLMCacheProxy(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_mem=0.90,
    max_model_len=4096,
    enable_feedback=True,
)
app = RAGCacheMiddleware(proxy).create_app()
```

Run benchmarks from the repository root:

```bash
bash scripts/run_final_benchmarks.sh
bash scripts/run_reviewer_benchmarks.sh
bash scripts/run_round7.sh
bash scripts/run_wiki.sh
```

New benchmark outputs should be written under `results/`.

## Deliverables

The final report and poster are in `deliverables/`:

- `deliverables/final_report.pdf`
- `deliverables/final_poster.pptx`
- `deliverables/speaker_notes.md`

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
