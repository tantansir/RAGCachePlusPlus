"""Real GPU benchmark for RAGCache++ using vLLM.

Experiment design:
1. Generate a corpus of synthetic documents with known spatial overlap patterns
2. Generate RAG queries that retrieve overlapping document subsets
3. Compare strategies:
   a. no_cache: vLLM with prefix caching disabled
   b. apc_random: vLLM APC enabled, random document ordering
   c. apc_retrieval: vLLM APC enabled, retrieval-rank ordering
   d. apc_optimized: vLLM APC enabled, knowledge-tree-optimized ordering
4. Measure: TTFT (prefill latency), total latency, cache block hit rate
"""

from __future__ import annotations

import os
import sys
# Disable triton flash attention on V100 (set via env or here as fallback)
os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")

import gc
import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from ragcache_pp.cache.knowledge_tree import KnowledgeTree
from ragcache_pp.vllm_integration.prompt_builder import (
    build_rag_prompt,
    optimize_doc_order,
)


# ---------------------------------------------------------------------------
# Corpus and workload generation
# ---------------------------------------------------------------------------

LOREM = (
    "The urban landscape of New York City presents unique challenges for "
    "location-based information retrieval. Points of interest are densely "
    "packed in neighborhoods like Midtown Manhattan, the Upper East Side, "
    "and Williamsburg in Brooklyn. Visitors frequently query nearby "
    "restaurants, museums, parks, and transit options within a small radius. "
    "This spatial locality in query patterns means that retrieval-augmented "
    "generation systems can benefit significantly from caching document "
    "representations that are likely to be re-accessed by subsequent queries "
    "from the same geographic area. The key insight is that KV cache blocks "
    "computed for one query's retrieved documents can be reused when a "
    "nearby query retrieves overlapping documents. "
)


def generate_corpus(num_docs: int = 500, tokens_per_doc: int = 200) -> dict[str, str]:
    """Generate a synthetic document corpus with known overlap patterns.

    Documents are grouped into 'regions' (10 docs each) to create
    predictable overlap when nearby queries retrieve from the same region.
    """
    corpus: dict[str, str] = {}
    # Repeat LOREM to reach desired length (approximate tokens ~ words * 1.3)
    target_words = int(tokens_per_doc / 1.3)
    base_words = LOREM.split()

    for i in range(num_docs):
        region = i // 10
        # Each doc has a unique identifier + region-specific content + filler
        header = (
            f"Document ID: doc_{i:04d}. Region: region_{region:03d}. "
            f"This document covers point of interest #{i} in region {region}. "
        )
        # Pad to target length
        words_needed = max(0, target_words - len(header.split()))
        filler = " ".join(base_words[j % len(base_words)] for j in range(words_needed))
        corpus[f"doc_{i:04d}"] = header + filler

    return corpus


@dataclass
class RAGQuery:
    """A single RAG benchmark query."""

    query_id: str
    query_text: str
    doc_ids: list[str]  # retrieved documents in rank order
    region: int = 0


def generate_rag_trace(
    corpus: dict[str, str],
    num_queries: int = 200,
    top_k: int = 5,
    overlap_fraction: float = 0.6,
    seed: int = 42,
) -> list[RAGQuery]:
    """Generate a RAG query trace with controlled document overlap.

    Queries are generated in bursts from the same region to create
    prefix-sharing opportunities.  `overlap_fraction` controls how many
    docs are shared between consecutive queries in a burst.
    """
    rng = random.Random(seed)
    doc_ids = sorted(corpus.keys())
    num_regions = len(doc_ids) // 10

    trace: list[RAGQuery] = []
    burst_size = 10  # queries per region burst

    for q_idx in range(num_queries):
        # Pick region (bursts of `burst_size` queries from same region)
        burst_region = (q_idx // burst_size) % num_regions
        region_docs = [d for d in doc_ids if int(d.split("_")[1]) // 10 == burst_region]

        # Position within burst determines overlap with previous
        pos_in_burst = q_idx % burst_size
        if pos_in_burst == 0 or not trace:
            # First query in burst — random selection from region
            selected = rng.sample(region_docs, min(top_k, len(region_docs)))
        else:
            # Overlap with previous query
            prev_docs = trace[-1].doc_ids
            num_shared = max(1, int(top_k * overlap_fraction))
            shared = prev_docs[:num_shared]
            # Fill remaining from region
            candidates = [d for d in region_docs if d not in shared]
            new_docs = rng.sample(candidates, min(top_k - num_shared, len(candidates)))
            selected = shared + new_docs

        query_text = (
            f"What are the notable features and visiting tips for the "
            f"points of interest in region {burst_region}?"
        )

        trace.append(RAGQuery(
            query_id=f"q_{q_idx:04d}",
            query_text=query_text,
            doc_ids=selected[:top_k],
            region=burst_region,
        ))

    return trace


# ---------------------------------------------------------------------------
# Measurement utilities
# ---------------------------------------------------------------------------

@dataclass
class RequestTiming:
    query_id: str
    strategy: str
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0
    prefix_hit: bool = False  # whether APC likely reused prefix


@dataclass
class ExperimentResult:
    strategy: str
    timings: list[RequestTiming] = field(default_factory=list)

    def summary(self) -> dict:
        ttfts = sorted(t.ttft_ms for t in self.timings)
        totals = sorted(t.total_ms for t in self.timings)
        n = len(ttfts)
        if n == 0:
            return {"strategy": self.strategy, "n": 0}

        def pct(data: list[float], p: float) -> float:
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        prompt_tokens = [t.num_prompt_tokens for t in self.timings]
        return {
            "strategy": self.strategy,
            "n": n,
            "ttft_p50_ms": round(pct(ttfts, 50), 2),
            "ttft_p95_ms": round(pct(ttfts, 95), 2),
            "ttft_p99_ms": round(pct(ttfts, 99), 2),
            "ttft_mean_ms": round(sum(ttfts) / n, 2),
            "total_p50_ms": round(pct(totals, 50), 2),
            "total_p95_ms": round(pct(totals, 95), 2),
            "avg_prompt_tokens": round(sum(prompt_tokens) / n, 1),
            "prefix_hit_rate": round(
                sum(1 for t in self.timings if t.prefix_hit) / n, 3
            ),
        }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_single_strategy(
    llm: LLM,
    trace: list[RAGQuery],
    corpus: dict[str, str],
    strategy: str,
    knowledge_tree: Optional[KnowledgeTree] = None,
    max_output_tokens: int = 1,
    warmup: int = 5,
) -> ExperimentResult:
    """Run one strategy over the full trace.

    We generate only 1 token per request (max_output_tokens=1) so that
    the measured latency is dominated by prefill (TTFT).
    """
    sampling_params = SamplingParams(
        max_tokens=max_output_tokens,
        temperature=0.0,
    )

    result = ExperimentResult(strategy=strategy)
    warmup_latencies: list[float] = []

    for i, query in enumerate(trace):
        # Build prompt with the chosen ordering strategy
        if strategy == "apc_optimized" and knowledge_tree is not None:
            prompt, ordered_ids = build_rag_prompt(
                query.query_text, query.doc_ids, corpus,
                doc_order="optimized", knowledge_tree=knowledge_tree,
            )
        elif strategy == "apc_random":
            prompt, ordered_ids = build_rag_prompt(
                query.query_text, query.doc_ids, corpus,
                doc_order="random",
            )
        elif strategy == "apc_sorted":
            prompt, ordered_ids = build_rag_prompt(
                query.query_text, query.doc_ids, corpus,
                doc_order="sorted",
            )
        else:
            # "no_cache" and "apc_retrieval" both use original order
            prompt, ordered_ids = build_rag_prompt(
                query.query_text, query.doc_ids, corpus,
                doc_order="original",
            )

        # Measure generation time (TTFT ≈ total for max_tokens=1)
        t0 = time.perf_counter()
        outputs: list[RequestOutput] = llm.generate([prompt], sampling_params)
        t1 = time.perf_counter()

        output = outputs[0]
        elapsed_ms = (t1 - t0) * 1000.0

        # Extract metrics from vLLM output
        num_prompt_tokens = (
            len(output.prompt_token_ids) if output.prompt_token_ids else 0
        )

        # Heuristic: compare against warmup latencies (first requests, likely cold)
        if i < warmup:
            warmup_latencies.append(elapsed_ms)
        if warmup_latencies:
            baseline_ms = sorted(warmup_latencies)[len(warmup_latencies) // 2]
        else:
            baseline_ms = elapsed_ms
        prefix_hit = elapsed_ms < baseline_ms * 0.6

        timing = RequestTiming(
            query_id=query.query_id,
            strategy=strategy,
            ttft_ms=elapsed_ms,
            total_ms=elapsed_ms,
            num_prompt_tokens=num_prompt_tokens,
            num_generated_tokens=max_output_tokens,
            prefix_hit=prefix_hit,
        )

        # Update knowledge tree with the served sequence (for optimized strategy)
        if knowledge_tree is not None:
            from ragcache_pp.cache.knowledge_tree import KVCacheMetadata
            metadata_list = []
            for doc_id in ordered_ids:
                meta = KVCacheMetadata(
                    doc_id=doc_id,
                    num_tokens=200,  # approximate
                    num_blocks=13,
                    tier="gpu",
                    created_at=i,
                    last_accessed_at=i,
                    access_count=1,
                )
                metadata_list.append(meta)
            knowledge_tree.insert(ordered_ids, metadata_list)

        if i >= warmup:
            result.timings.append(timing)

        if (i + 1) % 50 == 0:
            print(f"  [{strategy}] {i + 1}/{len(trace)} queries done")

    return result


def run_benchmark(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_docs: int = 500,
    num_queries: int = 200,
    top_k: int = 5,
    output_path: str = "results/benchmark_real_results.json",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    enforce_eager: bool = False,
    dtype: str = "auto",
    strategy_filter: str | None = None,
) -> dict:
    """Run the full real GPU benchmark.

    If strategy_filter is set, run only that single strategy.
    Results are merged with any existing results in output_path.
    """
    print("=" * 60)
    print("RAGCache++ Real GPU Benchmark")
    print("=" * 60)
    sys.stdout.flush()

    # Step 1: Generate corpus and trace
    print(f"\n[Step 1] Generating corpus ({num_docs} docs) and trace ({num_queries} queries)...")
    corpus = generate_corpus(num_docs=num_docs)
    trace = generate_rag_trace(corpus, num_queries=num_queries, top_k=top_k)

    # Compute overlap statistics
    total_overlap = 0
    for i in range(1, len(trace)):
        prev = set(trace[i - 1].doc_ids)
        curr = set(trace[i].doc_ids)
        total_overlap += len(prev & curr) / max(len(prev | curr), 1)
    avg_overlap = total_overlap / max(len(trace) - 1, 1)
    print(f"  Average Jaccard overlap between consecutive queries: {avg_overlap:.3f}")
    sys.stdout.flush()

    # Load any existing results (for incremental strategy runs)
    results = {}
    if strategy_filter and os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            results = existing.get("results", {})
            print(f"  Loaded {len(results)} existing strategy results from {output_path}")
        except (json.JSONDecodeError, KeyError):
            pass

    # Helper to create a fresh LLM instance (isolates APC cache state between strategies)
    def make_llm(enable_apc: bool) -> LLM:
        kwargs = dict(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=enable_apc,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
        )
        if dtype != "auto":
            kwargs["dtype"] = dtype
        return LLM(**kwargs)

    # Each strategy gets a FRESH LLM instance to avoid cache warm-state confounding.
    # Order: most important first (results survive if later strategy triggers LLVM abort).
    # apc_random excluded: triggers Triton LLVM bug on V100 (compute capability 7.0).
    all_strategies = [
        ("no_cache", False, "original", None),
        ("apc_retrieval", True, "original", None),
        ("apc_optimized", True, "optimized", KnowledgeTree()),
        ("apc_sorted", True, "sorted", None),
    ]

    if strategy_filter:
        strategies = [s for s in all_strategies if s[0] == strategy_filter]
        if not strategies:
            print(f"ERROR: unknown strategy '{strategy_filter}'")
            print(f"  Available: {[s[0] for s in all_strategies]}")
            return {}
    else:
        strategies = all_strategies

    import torch

    def _save_interim(path: str, results: dict, avg_ov: float, cfg: dict):
        """Write results to disk with fsync so data survives a crash."""
        out = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": cfg,
            "workload": {"avg_consecutive_overlap": round(avg_ov, 3)},
            "results": results,
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    _cfg = {"model_name": model_name, "num_docs": num_docs,
            "num_queries": num_queries, "top_k": top_k,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len}

    for step_idx, (strategy, enable_apc, doc_order, kt) in enumerate(strategies, 2):
        print(f"\n[Step {step_idx}] Running strategy: {strategy} (APC={'on' if enable_apc else 'off'})")
        sys.stdout.flush()
        print(f"  Loading fresh LLM instance...")
        sys.stdout.flush()
        llm = make_llm(enable_apc)

        r = run_single_strategy(llm, trace, corpus, strategy, knowledge_tree=kt)
        results[strategy] = r.summary()
        s = r.summary()
        print(f"  Result: TTFT p50={s['ttft_p50_ms']:.1f}ms, mean={s['ttft_mean_ms']:.1f}ms")
        sys.stdout.flush()

        # Save after each strategy — survives LLVM abort in later strategies
        _save_interim(output_path, results, avg_overlap, _cfg)
        print(f"  (results saved to {output_path})")
        sys.stdout.flush()

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Strategy':<20} {'TTFT p50':>10} {'TTFT p95':>10} {'TTFT mean':>10} {'Hit Rate':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['ttft_p50_ms']:>9.1f}ms {r['ttft_p95_ms']:>9.1f}ms "
              f"{r['ttft_mean_ms']:>9.1f}ms {r.get('prefix_hit_rate', 0):>9.3f}")

    # Improvement
    if "no_cache" in results and "apc_optimized" in results:
        baseline = results["no_cache"]["ttft_p50_ms"]
        optimized = results["apc_optimized"]["ttft_p50_ms"]
        improvement = (baseline - optimized) / baseline * 100
        print(f"\nTTFT improvement (optimized vs no_cache): {improvement:.1f}%")

    if "apc_retrieval" in results and "apc_optimized" in results:
        baseline_apc = results["apc_retrieval"]["ttft_p50_ms"]
        optimized_apc = results["apc_optimized"]["ttft_p50_ms"]
        improvement_apc = (baseline_apc - optimized_apc) / baseline_apc * 100
        print(f"TTFT improvement (optimized vs retrieval-order APC): {improvement_apc:.1f}%")

    if "apc_sorted" in results and "apc_optimized" in results:
        sorted_apc = results["apc_sorted"]["ttft_p50_ms"]
        optimized_apc = results["apc_optimized"]["ttft_p50_ms"]
        improvement_sorted = (sorted_apc - optimized_apc) / sorted_apc * 100
        print(f"TTFT improvement (optimized vs sorted APC): {improvement_sorted:.1f}%")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model_name": model_name,
            "num_docs": num_docs,
            "num_queries": num_queries,
            "top_k": top_k,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
        },
        "workload": {
            "avg_consecutive_overlap": round(avg_overlap, 3),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
    sys.stdout.flush()

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGCache++ Real GPU Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-docs", type=int, default=500)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="results/benchmark_real_results.json")
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs (needed when Triton/gcc unavailable)")
    parser.add_argument("--dtype", default="auto",
                        help="Model dtype (auto, half, float16, bfloat16)")
    parser.add_argument("--strategy", default=None,
                        help="Run only this strategy (for process isolation)")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        num_docs=args.num_docs,
        num_queries=args.num_queries,
        top_k=args.top_k,
        output_path=args.output,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        strategy_filter=args.strategy,
    )
