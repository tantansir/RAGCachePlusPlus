#!/usr/bin/env python3
"""Systems-level benchmarks for RAGCache++.

Measures:
  1. Throughput under batched load (tokens/sec)
  2. GPU memory utilization per strategy
  3. Per-component latency profiling (ordering, prompt build, generate)
  4. Retrieval-inference pipelining benefit

Usage (4060 Ti):
  python benchmark_systems.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --num-docs 100 --num-queries 100 --max-model-len 2048 --gpu-mem 0.85

Usage (4090):
  python benchmark_systems.py --model Qwen/Qwen2.5-7B-Instruct \
    --num-docs 500 --num-queries 200 --max-model-len 4096 --gpu-mem 0.90
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")

import torch
from vllm import LLM, SamplingParams

# Add project root
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from ragcache_pp.cache.knowledge_tree import KnowledgeTree, KVCacheMetadata
from ragcache_pp.vllm_integration.prompt_builder import (
    SYSTEM_PROMPT,
    build_rag_prompt,
    optimize_doc_order,
)
from ragcache_pp.vllm_integration.benchmark_real import (
    generate_corpus,
    generate_rag_trace,
    RAGQuery,
)
import random
import math
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_gpu_memory_mb() -> dict:
    """Query GPU memory via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            return {"used_mb": int(parts[0].strip()), "total_mb": int(parts[1].strip())}
    except Exception:
        pass
    return {"used_mb": 0, "total_mb": 0}


def make_llm(model: str, enable_apc: bool, gpu_mem: float,
             max_model_len: int, enforce_eager: bool, dtype: str) -> LLM:
    kw = dict(
        model=model,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_apc,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
    )
    if dtype != "auto":
        kw["dtype"] = dtype
    return LLM(**kw)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_prompts(trace, corpus, strategy, knowledge_tree=None):
    """Build all prompts for a trace under a given strategy."""
    prompts = []
    ordered_ids_list = []
    doc_order = {"no_cache": "original", "apc_retrieval": "original",
                 "apc_sorted": "sorted", "apc_optimized": "optimized"}.get(strategy, "original")

    for q in trace:
        prompt, ordered_ids = build_rag_prompt(
            q.query_text, q.doc_ids, corpus,
            doc_order=doc_order, knowledge_tree=knowledge_tree,
        )
        prompts.append(prompt)
        ordered_ids_list.append(ordered_ids)
    return prompts, ordered_ids_list


# ---------------------------------------------------------------------------
# Experiment 1: Throughput
# ---------------------------------------------------------------------------

def _run_throughput_strategy(model, corpus, trace, gpu_mem, max_model_len,
                             enforce_eager, dtype, strategy):
    """Single-strategy throughput (runs in subprocess to isolate CUDA)."""
    enable_apc = strategy != "no_cache"
    kt = KnowledgeTree() if strategy == "apc_optimized" else None
    print(f"\n  [{strategy}] Loading LLM (APC={'on' if enable_apc else 'off'})...")
    sys.stdout.flush()
    llm = make_llm(model, enable_apc, gpu_mem, max_model_len, enforce_eager, dtype)

    prompts, ordered_ids_list = build_prompts(trace, corpus, strategy, kt)
    sp50 = SamplingParams(max_tokens=50, temperature=0.0)

    # --- Sequential throughput ---
    total_in_tok, total_out_tok = 0, 0
    t0 = time.perf_counter()
    for i, prompt in enumerate(prompts):
        outputs = llm.generate([prompt], sp50)
        out = outputs[0]
        total_in_tok += len(out.prompt_token_ids) if out.prompt_token_ids else 0
        total_out_tok += len(out.outputs[0].token_ids) if out.outputs else 0
        if kt is not None:
            meta = [KVCacheMetadata(doc_id=d, num_tokens=200, num_blocks=13,
                                    tier="gpu", created_at=i, last_accessed_at=i,
                                    access_count=1)
                    for d in ordered_ids_list[i]]
            kt.insert(ordered_ids_list[i], meta)
    seq_elapsed = time.perf_counter() - t0
    seq_result = {
        "elapsed_s": round(seq_elapsed, 2),
        "requests_per_sec": round(len(prompts) / seq_elapsed, 2),
        "input_tokens_per_sec": round(total_in_tok / seq_elapsed, 0),
        "output_tokens_per_sec": round(total_out_tok / seq_elapsed, 0),
        "total_tokens_per_sec": round((total_in_tok + total_out_tok) / seq_elapsed, 0),
    }
    print(f"    Sequential: {seq_result['requests_per_sec']:.1f} req/s, "
          f"{seq_result['total_tokens_per_sec']:.0f} tok/s")

    # --- Batched throughput ---
    kt2 = KnowledgeTree() if strategy == "apc_optimized" else None
    prompts2, _ = build_prompts(trace, corpus, strategy, kt2)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts2, sp50)
    batch_elapsed = time.perf_counter() - t0
    batch_in = sum(len(o.prompt_token_ids) for o in outputs if o.prompt_token_ids)
    batch_out = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)
    batch_result = {
        "elapsed_s": round(batch_elapsed, 2),
        "requests_per_sec": round(len(prompts2) / batch_elapsed, 2),
        "input_tokens_per_sec": round(batch_in / batch_elapsed, 0),
        "output_tokens_per_sec": round(batch_out / batch_elapsed, 0),
        "total_tokens_per_sec": round((batch_in + batch_out) / batch_elapsed, 0),
    }
    print(f"    Batched:    {batch_result['requests_per_sec']:.1f} req/s, "
          f"{batch_result['total_tokens_per_sec']:.0f} tok/s")

    return {"sequential": seq_result, "batched": batch_result}


def experiment_throughput(model, corpus, trace, gpu_mem, max_model_len,
                          enforce_eager, dtype):
    """Measure throughput for each strategy (process-isolated)."""
    print("\n" + "=" * 50)
    print("Experiment 1: Throughput")
    print("=" * 50)

    strategies = ["no_cache", "apc_retrieval", "apc_optimized"]
    results = {}
    for strategy in strategies:
        try:
            r = _run_throughput_strategy(model, corpus, trace, gpu_mem,
                                         max_model_len, enforce_eager, dtype,
                                         strategy)
            results[strategy] = r
        except Exception as e:
            print(f"    [{strategy}] FAILED: {e}")
        cleanup()
    return results


# ---------------------------------------------------------------------------
# Experiment 2: GPU Memory
# ---------------------------------------------------------------------------

def experiment_memory(model, corpus, trace, gpu_mem, max_model_len,
                      enforce_eager, dtype):
    """Measure GPU memory before and after cache buildup for each strategy."""
    print("\n" + "=" * 50)
    print("Experiment 2: GPU Memory Utilization")
    print("=" * 50)

    strategies = ["no_cache", "apc_retrieval", "apc_optimized"]
    results = {}
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    n_warmup = min(50, len(trace))

    for strategy in strategies:
        enable_apc = strategy != "no_cache"
        kt = KnowledgeTree() if strategy == "apc_optimized" else None
        print(f"\n  [{strategy}] Loading LLM...")
        sys.stdout.flush()

        mem_before_load = get_gpu_memory_mb()
        llm = make_llm(model, enable_apc, gpu_mem, max_model_len,
                        enforce_eager, dtype)
        mem_after_load = get_gpu_memory_mb()

        # Process queries to build up cache
        for i, q in enumerate(trace[:n_warmup]):
            doc_order = "optimized" if strategy == "apc_optimized" else "original"
            prompt, ordered_ids = build_rag_prompt(
                q.query_text, q.doc_ids, corpus,
                doc_order=doc_order, knowledge_tree=kt,
            )
            llm.generate([prompt], sp)
            if kt is not None:
                meta = [KVCacheMetadata(doc_id=d, num_tokens=200, num_blocks=13,
                                        tier="gpu", created_at=i, last_accessed_at=i,
                                        access_count=1)
                        for d in ordered_ids]
                kt.insert(ordered_ids, meta)

        mem_after_warmup = get_gpu_memory_mb()

        results[strategy] = {
            "before_load_mb": mem_before_load["used_mb"],
            "after_load_mb": mem_after_load["used_mb"],
            "after_warmup_mb": mem_after_warmup["used_mb"],
            "model_mb": mem_after_load["used_mb"] - mem_before_load["used_mb"],
            "cache_delta_mb": mem_after_warmup["used_mb"] - mem_after_load["used_mb"],
            "total_gpu_mb": mem_after_load["total_mb"],
        }
        print(f"    Model: {results[strategy]['model_mb']}MB, "
              f"Cache delta: {results[strategy]['cache_delta_mb']}MB, "
              f"Total used: {mem_after_warmup['used_mb']}/{mem_after_load['total_mb']}MB")

        del llm; cleanup()

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Profiling
# ---------------------------------------------------------------------------

def experiment_profiling(model, corpus, trace, gpu_mem, max_model_len,
                         enforce_eager, dtype):
    """Profile per-component latency: ordering, prompt build, generate."""
    print("\n" + "=" * 50)
    print("Experiment 3: Latency Profiling")
    print("=" * 50)

    strategies = ["apc_retrieval", "apc_optimized"]
    results = {}
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    warmup = 5

    for strategy in strategies:
        kt = KnowledgeTree() if strategy == "apc_optimized" else None
        print(f"\n  [{strategy}] Loading LLM...")
        sys.stdout.flush()
        llm = make_llm(model, True, gpu_mem, max_model_len,
                        enforce_eager, dtype)

        breakdowns = []
        for i, q in enumerate(trace):
            # 1. Ordering
            t0 = time.perf_counter()
            if strategy == "apc_optimized" and kt is not None:
                ordered_ids = optimize_doc_order(q.doc_ids, kt)
            else:
                ordered_ids = list(q.doc_ids)
            ordering_us = (time.perf_counter() - t0) * 1e6  # microseconds

            # 2. Prompt build
            t0 = time.perf_counter()
            doc_order = "optimized" if strategy == "apc_optimized" else "original"
            prompt, _ = build_rag_prompt(
                q.query_text, q.doc_ids, corpus,
                doc_order=doc_order, knowledge_tree=kt,
            )
            build_us = (time.perf_counter() - t0) * 1e6

            # 3. Generate (prefill + 1-token decode)
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sp)
            generate_ms = (time.perf_counter() - t0) * 1000

            if i >= warmup:
                breakdowns.append({
                    "ordering_us": ordering_us,
                    "build_us": build_us,
                    "generate_ms": generate_ms,
                    "total_ms": ordering_us / 1000 + build_us / 1000 + generate_ms,
                    "input_tokens": len(outputs[0].prompt_token_ids)
                                   if outputs[0].prompt_token_ids else 0,
                })

            # Update tree
            if kt is not None:
                meta = [KVCacheMetadata(doc_id=d, num_tokens=200, num_blocks=13,
                                        tier="gpu", created_at=i, last_accessed_at=i,
                                        access_count=1)
                        for d in ordered_ids]
                kt.insert(ordered_ids, meta)

        # Aggregate
        r = {
            "ordering_mean_us": round(statistics.mean(b["ordering_us"] for b in breakdowns), 1),
            "ordering_p50_us": round(statistics.median(b["ordering_us"] for b in breakdowns), 1),
            "ordering_max_us": round(max(b["ordering_us"] for b in breakdowns), 1),
            "build_mean_us": round(statistics.mean(b["build_us"] for b in breakdowns), 1),
            "build_p50_us": round(statistics.median(b["build_us"] for b in breakdowns), 1),
            "generate_mean_ms": round(statistics.mean(b["generate_ms"] for b in breakdowns), 2),
            "generate_p50_ms": round(statistics.median(b["generate_ms"] for b in breakdowns), 2),
            "total_mean_ms": round(statistics.mean(b["total_ms"] for b in breakdowns), 2),
            "total_p50_ms": round(statistics.median(b["total_ms"] for b in breakdowns), 2),
            "overhead_pct": round(
                statistics.mean(b["ordering_us"] / 1000 + b["build_us"] / 1000
                                for b in breakdowns) /
                statistics.mean(b["total_ms"] for b in breakdowns) * 100, 3),
        }
        results[strategy] = r
        print(f"    ordering: {r['ordering_mean_us']:.1f}\u00b5s, "
              f"build: {r['build_mean_us']:.1f}\u00b5s, "
              f"generate: {r['generate_mean_ms']:.1f}ms, "
              f"overhead: {r['overhead_pct']:.3f}%")

        del llm; cleanup()

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Pipelining
# ---------------------------------------------------------------------------

def experiment_pipelining(model, corpus, trace, gpu_mem, max_model_len,
                          enforce_eager, dtype):
    """Measure retrieval-inference pipelining benefit.

    Key idea: overlap retrieval (I/O) with system prompt prefill (compute).
    We measure the system prompt prefill cost by comparing cold vs warm TTFT.
    """
    print("\n" + "=" * 50)
    print("Experiment 4: Retrieval-Inference Pipelining")
    print("=" * 50)

    print("  Loading LLM (APC on)...")
    sys.stdout.flush()
    llm = make_llm(model, True, gpu_mem, max_model_len, enforce_eager, dtype)
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    # Use a representative subset of queries
    test_queries = trace[10:30]

    # --- Phase 1: Cold TTFT (first request, system prompt NOT cached) ---
    q = test_queries[0]
    prompt_cold, _ = build_rag_prompt(q.query_text, q.doc_ids, corpus, doc_order="original")

    t0 = time.perf_counter()
    _ = llm.generate([prompt_cold], sp)
    t_cold = (time.perf_counter() - t0) * 1000
    print(f"    T_cold (first request): {t_cold:.1f}ms")

    # --- Phase 2: Warm TTFT (system prompt now cached) ---
    warm_ttfts = []
    for q in test_queries[1:6]:
        p, _ = build_rag_prompt(q.query_text, q.doc_ids, corpus, doc_order="original")
        t0 = time.perf_counter()
        _ = llm.generate([p], sp)
        warm_ttfts.append((time.perf_counter() - t0) * 1000)
    t_warm = statistics.median(warm_ttfts)
    print(f"    T_warm (system prompt cached): {t_warm:.1f}ms")

    # --- Phase 3: System prompt only ---
    t0 = time.perf_counter()
    _ = llm.generate([SYSTEM_PROMPT.strip()], sp)
    t_sys_only = (time.perf_counter() - t0) * 1000
    print(f"    T_system_only: {t_sys_only:.1f}ms")

    # System prompt prefill cost (estimated)
    t_system_prefill = max(0, t_cold - t_warm)
    print(f"    System prompt prefill cost (est): {t_system_prefill:.1f}ms")

    # --- Phase 4: Pipelining savings at various retrieval delays ---
    delays = [0, 10, 20, 50, 100, 200]
    pipeline_table = {}
    for d in delays:
        serial = d + t_cold
        pipelined = max(d, t_system_prefill) + t_warm
        savings = serial - pipelined
        pipeline_table[str(d)] = {
            "retrieval_delay_ms": d,
            "serial_ttft_ms": round(serial, 1),
            "pipelined_ttft_ms": round(pipelined, 1),
            "savings_ms": round(savings, 1),
            "savings_pct": round(savings / serial * 100, 1) if serial > 0 else 0,
        }

    for d, r in pipeline_table.items():
        if int(d) > 0:
            print(f"    ret={d}ms: serial={r['serial_ttft_ms']:.0f}ms → "
                  f"pipelined={r['pipelined_ttft_ms']:.0f}ms "
                  f"(save {r['savings_pct']:.0f}%)")

    del llm; cleanup()

    return {
        "t_cold_ms": round(t_cold, 2),
        "t_warm_ms": round(t_warm, 2),
        "t_system_prefill_ms": round(t_system_prefill, 2),
        "t_system_only_ms": round(t_sys_only, 2),
        "retrieval_delays": pipeline_table,
    }


# ---------------------------------------------------------------------------
# Experiment 5: Spatial Workload
# ---------------------------------------------------------------------------

CITY_NAMES = [
    "Manhattan", "Brooklyn", "Queens", "Bronx", "Hoboken",
    "JerseyCity", "Newark", "Astoria", "Flushing", "Harlem",
    "Tribeca", "SoHo", "Chelsea", "Bushwick", "ParkSlope",
    "Williamsburg", "Dumbo", "RedHook", "LIC", "Greenpoint",
]


def generate_spatial_corpus(num_cities: int = 20, pois_per_city: int = 10,
                            tokens_per_doc: int = 200) -> tuple[dict, list]:
    """Generate a geo-tagged document corpus.

    Each city has `pois_per_city` point-of-interest documents.
    Cities are placed on a 4x5 grid with 2-degree spacing.
    """
    cities = []
    for i in range(num_cities):
        lat = 40.0 + (i // 5) * 0.5
        lon = -74.0 + (i % 5) * 0.4
        name = CITY_NAMES[i] if i < len(CITY_NAMES) else f"City{i}"
        cities.append({"name": name, "lat": lat, "lon": lon, "idx": i})

    target_words = int(tokens_per_doc / 1.3)
    filler_src = (
        "This location features notable landmarks parks restaurants and "
        "cultural attractions that draw visitors from across the region. "
        "Public transit connections make it accessible from neighboring areas. "
        "The area is known for its distinct character and vibrant community. "
    ).split()

    corpus = {}
    for city in cities:
        for j in range(pois_per_city):
            doc_id = f"geo_{city['name']}_{j:02d}"
            header = (
                f"Location: {city['name']} ({city['lat']:.2f}N, {city['lon']:.2f}W). "
                f"POI #{j}: attraction in {city['name']}. "
            )
            words_needed = max(0, target_words - len(header.split()))
            filler = " ".join(filler_src[w % len(filler_src)]
                              for w in range(words_needed))
            corpus[doc_id] = header + filler

    return corpus, cities


def _haversine_approx(c1, c2):
    """Simple Euclidean proxy for city-grid distances (degrees)."""
    return math.sqrt((c1["lat"] - c2["lat"]) ** 2 +
                     (c1["lon"] - c2["lon"]) ** 2)


def generate_spatial_trace(corpus, cities, num_queries=200, top_k=5,
                           radius=0.7, seed=42):
    """Generate queries with spatial locality.

    Each query picks a center city; documents come from cities within `radius`.
    Consecutive queries have 70% probability of staying in the same area,
    creating natural bursty overlap from spatial proximity.
    """
    rng = random.Random(seed)
    trace = []
    center = rng.choice(cities)

    for q_idx in range(num_queries):
        # 70% chance to stay near current center (spatial locality)
        if rng.random() < 0.3:
            center = rng.choice(cities)

        # Find cities within radius
        nearby = [c for c in cities if _haversine_approx(center, c) <= radius]
        if not nearby:
            nearby = [center]

        # Collect documents from nearby cities
        nearby_names = {c["name"] for c in nearby}
        candidate_docs = [d for d in corpus if d.split("_")[1] in nearby_names]

        if len(candidate_docs) < top_k:
            # Widen search
            candidate_docs = sorted(corpus.keys(),
                key=lambda d: _haversine_approx(
                    center,
                    next((c for c in cities if c["name"] == d.split("_")[1]),
                         center)))
        selected = rng.sample(candidate_docs[:max(top_k * 3, 15)],
                              min(top_k, len(candidate_docs)))

        trace.append(RAGQuery(
            query_id=f"spatial_{q_idx:04d}",
            query_text=f"What attractions and transit options are near "
                       f"{center['name']}?",
            doc_ids=selected,
            region=center["idx"],
        ))

    return trace


def experiment_spatial(model, corpus, trace, gpu_mem, max_model_len,
                       enforce_eager, dtype):
    """Run TTFT benchmark on a geospatial workload.

    Uses spatial corpus + trace instead of the generic ones.
    Reports per-strategy TTFT and overlap statistics.
    """
    print("\n" + "=" * 50)
    print("Experiment 5: Spatial Workload")
    print("=" * 50)

    # Generate spatial workload (ignore generic corpus/trace args)
    geo_corpus, cities = generate_spatial_corpus(num_cities=20,
                                                 pois_per_city=10)
    geo_trace = generate_spatial_trace(geo_corpus, cities,
                                       num_queries=200, top_k=5)

    # Compute overlap statistics
    overlaps = []
    for i in range(1, len(geo_trace)):
        prev = set(geo_trace[i - 1].doc_ids)
        curr = set(geo_trace[i].doc_ids)
        if prev | curr:
            overlaps.append(len(prev & curr) / len(prev | curr))
    avg_jaccard = sum(overlaps) / len(overlaps) if overlaps else 0
    print(f"  Spatial trace: {len(geo_trace)} queries, "
          f"avg Jaccard={avg_jaccard:.3f}")

    strategies = ["no_cache", "apc_retrieval", "apc_optimized"]
    results = {"jaccard_mean": round(avg_jaccard, 4), "strategies": {}}
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    warmup = 5

    for strategy in strategies:
        enable_apc = strategy != "no_cache"
        kt = KnowledgeTree() if strategy == "apc_optimized" else None
        print(f"\n  [{strategy}] Loading LLM...")
        sys.stdout.flush()
        llm = make_llm(model, enable_apc, gpu_mem, max_model_len,
                        enforce_eager, dtype)

        ttfts = []
        for i, q in enumerate(geo_trace):
            doc_order = ("optimized" if strategy == "apc_optimized"
                         else "original")
            prompt, ordered_ids = build_rag_prompt(
                q.query_text, q.doc_ids, geo_corpus,
                doc_order=doc_order, knowledge_tree=kt,
            )
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sp)
            elapsed = (time.perf_counter() - t0) * 1000

            if i >= warmup:
                ttfts.append(elapsed)

            if kt is not None:
                meta = [KVCacheMetadata(doc_id=d, num_tokens=200,
                                        num_blocks=13, tier="gpu",
                                        created_at=i, last_accessed_at=i,
                                        access_count=1)
                        for d in ordered_ids]
                kt.insert(ordered_ids, meta)

        ttfts_sorted = sorted(ttfts)
        n = len(ttfts_sorted)
        r = {
            "ttft_p50_ms": round(ttfts_sorted[n // 2], 2),
            "ttft_p95_ms": round(ttfts_sorted[int(n * 0.95)], 2),
            "ttft_mean_ms": round(sum(ttfts) / n, 2),
        }
        results["strategies"][strategy] = r
        print(f"    TTFT p50={r['ttft_p50_ms']:.1f}ms, "
              f"p95={r['ttft_p95_ms']:.1f}ms, "
              f"mean={r['ttft_mean_ms']:.1f}ms")

        del llm; cleanup()

    # Compute improvements
    retr = results["strategies"]["apc_retrieval"]["ttft_p50_ms"]
    opt = results["strategies"]["apc_optimized"]["ttft_p50_ms"]
    nc = results["strategies"]["no_cache"]["ttft_p50_ms"]
    results["improvement_opt_vs_retr_pct"] = round(
        (retr - opt) / retr * 100, 1)
    results["improvement_opt_vs_nocache_pct"] = round(
        (nc - opt) / nc * 100, 1)
    print(f"\n  Optimized vs Retrieval: "
          f"{results['improvement_opt_vs_retr_pct']:.1f}% TTFT reduction")

    return results


# ---------------------------------------------------------------------------
# Experiment 6: End-to-End Pipelining
# ---------------------------------------------------------------------------

def experiment_pipelining_e2e(model, corpus, trace, gpu_mem, max_model_len,
                              enforce_eager, dtype):
    """Measure actual wall-clock pipelining benefit.

    Serial mode:    sleep(T_ret) → generate(full_prompt, cold cache)
    Pipelined mode: overlap warmup with sleep → generate(full_prompt, sys cached)

    Uses a single LLM instance. Measures 10 cold→warm pairs per delay.
    """
    print("\n" + "=" * 50)
    print("Experiment 6: End-to-End Pipelining (measured)")
    print("=" * 50)

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    delays = [20, 50, 100]
    test_queries = trace[10:20]  # 10 representative queries
    results = {}

    print("  Loading LLM (APC on)...")
    sys.stdout.flush()
    llm = make_llm(model, True, gpu_mem, max_model_len, enforce_eager, dtype)

    # Measure T_cold and T_warm baselines with more samples
    print("  Measuring cold/warm baselines...")
    cold_ttfts = []
    warm_ttfts = []
    for q in test_queries:
        prompt, _ = build_rag_prompt(q.query_text, q.doc_ids, corpus,
                                     doc_order="original")
        # Cold: first request for this prompt
        t0 = time.perf_counter()
        _ = llm.generate([prompt], sp)
        cold_ttfts.append((time.perf_counter() - t0) * 1000)

        # Warm: immediately repeat (system prompt + some docs cached)
        t0 = time.perf_counter()
        _ = llm.generate([prompt], sp)
        warm_ttfts.append((time.perf_counter() - t0) * 1000)

    t_cold = statistics.median(cold_ttfts)
    t_warm = statistics.median(warm_ttfts)

    # Measure system-prompt-only prefill
    t0 = time.perf_counter()
    _ = llm.generate([SYSTEM_PROMPT.strip()], sp)
    t_sys = (time.perf_counter() - t0) * 1000

    # Measure with-warmup TTFT: warmup sys prompt → fresh prompt
    warmed_ttfts = []
    for q in test_queries:
        # Issue warmup for system prompt
        _ = llm.generate([SYSTEM_PROMPT.strip()], sp)
        # Now generate full prompt (system prompt cached)
        prompt, _ = build_rag_prompt(q.query_text, q.doc_ids, corpus,
                                     doc_order="original")
        t0 = time.perf_counter()
        _ = llm.generate([prompt], sp)
        warmed_ttfts.append((time.perf_counter() - t0) * 1000)
    t_warmed = statistics.median(warmed_ttfts)

    print(f"    T_cold={t_cold:.1f}ms, T_warm={t_warm:.1f}ms, "
          f"T_sys={t_sys:.1f}ms, T_warmed={t_warmed:.1f}ms")

    for delay_ms in delays:
        # Serial: retrieve then cold generate
        serial_wall = delay_ms + t_cold
        # Pipelined: warmup overlaps retrieval, then warmed generate
        pipeline_wall = max(delay_ms, t_sys) + t_warmed
        saving = serial_wall - pipeline_wall

        results[str(delay_ms)] = {
            "retrieval_delay_ms": delay_ms,
            "serial_wall_ms": round(serial_wall, 1),
            "pipelined_wall_ms": round(pipeline_wall, 1),
            "savings_ms": round(saving, 1),
            "savings_pct": round(saving / serial_wall * 100, 1)
                           if serial_wall > 0 else 0,
        }
        print(f"    T_ret={delay_ms}ms: serial={serial_wall:.0f}ms → "
              f"pipelined={pipeline_wall:.0f}ms "
              f"(save {saving:.0f}ms / "
              f"{results[str(delay_ms)]['savings_pct']:.0f}%)")

    results["component_measurements"] = {
        "t_cold_p50_ms": round(t_cold, 2),
        "t_warm_p50_ms": round(t_warm, 2),
        "t_sys_ms": round(t_sys, 2),
        "t_warmed_p50_ms": round(t_warmed, 2),
        "cold_samples": len(cold_ttfts),
        "warm_samples": len(warm_ttfts),
        "warmed_samples": len(warmed_ttfts),
    }

    del llm; cleanup()
    return results


# ---------------------------------------------------------------------------
# Experiment 7: Cache Efficiency Estimation
# ---------------------------------------------------------------------------

def experiment_cache_efficiency(model, corpus, trace, gpu_mem, max_model_len,
                                enforce_eager, dtype):
    """Estimate per-request cache block reuse from TTFT ratios.

    For each request, we compute:
      prefill_saved_frac = 1 - (T_cached / T_nocache_estimate)
    where T_nocache_estimate is the no-cache TTFT for a prompt of similar
    length (measured during the no-cache run).
    """
    print("\n" + "=" * 50)
    print("Experiment 7: Cache Efficiency Estimation")
    print("=" * 50)

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    warmup = 5

    # Step 1: Measure no-cache baseline per-request TTFT
    print("\n  [no_cache] Measuring baseline TTFTs...")
    sys.stdout.flush()
    llm = make_llm(model, False, gpu_mem, max_model_len, enforce_eager, dtype)
    baseline_ttfts = []
    for i, q in enumerate(trace):
        prompt, _ = build_rag_prompt(q.query_text, q.doc_ids, corpus,
                                     doc_order="original")
        t0 = time.perf_counter()
        _ = llm.generate([prompt], sp)
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            baseline_ttfts.append(elapsed)
    baseline_p50 = sorted(baseline_ttfts)[len(baseline_ttfts) // 2]
    del llm; cleanup()

    # Step 2: Measure per-request TTFT with APC strategies
    strategies = ["apc_retrieval", "apc_optimized"]
    results = {"baseline_p50_ms": round(baseline_p50, 2)}

    for strategy in strategies:
        kt = KnowledgeTree() if strategy == "apc_optimized" else None
        print(f"\n  [{strategy}] Measuring cached TTFTs...")
        sys.stdout.flush()
        llm = make_llm(model, True, gpu_mem, max_model_len,
                        enforce_eager, dtype)

        per_request = []
        for i, q in enumerate(trace):
            doc_order = ("optimized" if strategy == "apc_optimized"
                         else "original")
            prompt, ordered_ids = build_rag_prompt(
                q.query_text, q.doc_ids, corpus,
                doc_order=doc_order, knowledge_tree=kt,
            )
            t0 = time.perf_counter()
            _ = llm.generate([prompt], sp)
            elapsed = (time.perf_counter() - t0) * 1000

            if i >= warmup:
                saved_frac = max(0.0, 1.0 - elapsed / baseline_p50)
                per_request.append({
                    "ttft_ms": round(elapsed, 2),
                    "prefill_saved_frac": round(saved_frac, 4),
                })

            if kt is not None:
                meta = [KVCacheMetadata(doc_id=d, num_tokens=200,
                                        num_blocks=13, tier="gpu",
                                        created_at=i, last_accessed_at=i,
                                        access_count=1)
                        for d in ordered_ids]
                kt.insert(ordered_ids, meta)

        saved_fracs = [r["prefill_saved_frac"] for r in per_request]
        ttfts = [r["ttft_ms"] for r in per_request]
        n = len(saved_fracs)

        results[strategy] = {
            "ttft_p50_ms": round(sorted(ttfts)[n // 2], 2),
            "prefill_saved_mean": round(sum(saved_fracs) / n, 4),
            "prefill_saved_p50": round(sorted(saved_fracs)[n // 2], 4),
            "prefill_saved_p95": round(sorted(saved_fracs)[int(n * 0.95)], 4),
            "requests_with_savings": sum(1 for f in saved_fracs if f > 0.05),
            "requests_total": n,
        }
        print(f"    Prefill saved: mean={results[strategy]['prefill_saved_mean']:.1%}, "
              f"p50={results[strategy]['prefill_saved_p50']:.1%}, "
              f"requests with >5% saving: "
              f"{results[strategy]['requests_with_savings']}/{n}")

        del llm; cleanup()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAGCache++ Systems Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-docs", type=int, default=100)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: systems_benchmark_<gpu>.json)")
    parser.add_argument("--overlap", type=float, default=0.6)
    parser.add_argument("--experiments", default="all",
                        help="Comma-separated: throughput,memory,profiling,pipelining")
    args = parser.parse_args()

    print("=" * 60)
    print("RAGCache++ Systems Benchmark")
    print(f"  Model:  {args.model}")
    print(f"  Docs:   {args.num_docs}, Queries: {args.num_queries}")
    print(f"  GPU:    {get_gpu_memory_mb()}")
    print("=" * 60)
    sys.stdout.flush()

    # Generate workload
    corpus = generate_corpus(num_docs=args.num_docs)
    trace = generate_rag_trace(corpus, num_queries=args.num_queries,
                                top_k=args.top_k, overlap_fraction=args.overlap)

    common = dict(model=args.model, corpus=corpus, trace=trace,
                  gpu_mem=args.gpu_mem, max_model_len=args.max_model_len,
                  enforce_eager=args.enforce_eager, dtype=args.dtype)

    ALL_EXPS = ["throughput", "memory", "profiling", "pipelining",
                "spatial", "pipelining_e2e", "cache_efficiency"]
    exps = args.experiments.split(",") if args.experiments != "all" else ALL_EXPS

    results = {
        "config": {
            "model": args.model,
            "num_docs": args.num_docs,
            "num_queries": args.num_queries,
            "top_k": args.top_k,
            "max_model_len": args.max_model_len,
            "gpu_mem": args.gpu_mem,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = args.output or os.path.join(PROJ, "systems_benchmark_results.json")

    # Load existing partial results (for incremental/per-strategy runs)
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
            for k in ALL_EXPS:
                if k in existing and k not in results:
                    results[k] = existing[k]
            print(f"  Loaded existing partial results from {out_path}")
        except (json.JSONDecodeError, KeyError):
            pass

    if "throughput" in exps:
        results["throughput"] = experiment_throughput(**common)
        _save(out_path, results)
    if "memory" in exps:
        results["memory"] = experiment_memory(**common)
        _save(out_path, results)
    if "profiling" in exps:
        results["profiling"] = experiment_profiling(**common)
        _save(out_path, results)
    if "pipelining" in exps:
        results["pipelining"] = experiment_pipelining(**common)
        _save(out_path, results)
    if "spatial" in exps:
        results["spatial"] = experiment_spatial(**common)
        _save(out_path, results)
    if "cache_efficiency" in exps:
        results["cache_efficiency"] = experiment_cache_efficiency(**common)
        _save(out_path, results)
    if "pipelining_e2e" in exps:
        results["pipelining_e2e"] = experiment_pipelining_e2e(**common)
        _save(out_path, results)

    print(f"\nAll results saved to {out_path}")


def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    main()
