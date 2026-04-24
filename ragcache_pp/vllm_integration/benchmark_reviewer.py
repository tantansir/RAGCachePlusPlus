#!/usr/bin/env python3
"""Reviewer-requested benchmarks for RAGCache++.

Addresses critical/major reviewer weaknesses:
  W1:  Concurrent multi-tenant load test
  W4:  Quality check with 7B model (HotpotQA, 200 examples)
  W6:  Zero-overlap anomaly investigation
  W7:  Stronger ordering baselines (recency, oracle)
  W10: Multi-seed statistical rigor with confidence intervals
  W1b: Cache eviction pressure (reduced GPU memory)

Usage (4090):
  python benchmark_reviewer.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-docs 500 --num-queries 200 \
    --max-model-len 4096 --gpu-mem 0.90 \
    --enforce-eager \
    --experiments all \
    --output /root/autodl-tmp/ragcache_pp/results/reviewer_results.json
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from itertools import permutations

os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")

import torch
from vllm import LLM, SamplingParams

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
RESULTS_DIR = os.path.join(PROJ, "results")

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


# ===================================================================
# Utility functions
# ===================================================================

def get_gpu_memory_mb() -> dict:
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


def make_llm(model, enable_apc, gpu_mem, max_model_len, enforce_eager, dtype):
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


def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


# ===================================================================
# New ordering strategies
# ===================================================================

def _order_recency(doc_ids: list[str], doc_recency: dict[str, int]) -> list[str]:
    """Order documents by most recently seen first, fallback to original order."""
    idx_map = {d: i for i, d in enumerate(doc_ids)}
    return sorted(doc_ids, key=lambda d: (-doc_recency.get(d, -1), idx_map[d]))


def _order_oracle(doc_ids: list[str], knowledge_tree: KnowledgeTree) -> list[str]:
    """Exhaustive search for permutation with longest prefix match.
    Feasible for k <= 7 (5040 permutations max)."""
    if len(doc_ids) > 7:
        return list(doc_ids)

    best_perm = list(doc_ids)
    best_len = 0

    for perm in permutations(doc_ids):
        _, match_len = knowledge_tree.prefix_match(list(perm))
        if match_len > best_len:
            best_len = match_len
            best_perm = list(perm)
        if best_len == len(doc_ids):
            break  # Can't do better than full match
    return best_perm


def compute_ordering(doc_ids, strategy, knowledge_tree=None, doc_recency=None):
    """Compute document ordering for any strategy."""
    if strategy in ("no_cache", "apc_retrieval"):
        return list(doc_ids)
    elif strategy == "apc_sorted":
        return sorted(doc_ids)
    elif strategy == "apc_recency":
        return _order_recency(doc_ids, doc_recency or {})
    elif strategy == "apc_oracle":
        if knowledge_tree:
            return _order_oracle(doc_ids, knowledge_tree)
        return list(doc_ids)
    elif strategy == "apc_optimized":
        if knowledge_tree:
            return optimize_doc_order(doc_ids, knowledge_tree)
        return list(doc_ids)
    return list(doc_ids)


def update_tree(kt, ordered_ids, query_idx):
    """Insert served sequence into knowledge tree."""
    if kt is None:
        return
    meta = [KVCacheMetadata(doc_id=d, num_tokens=200, num_blocks=13,
                            tier="gpu", created_at=query_idx,
                            last_accessed_at=query_idx, access_count=1)
            for d in ordered_ids]
    kt.insert(ordered_ids, meta)


def run_trace_sequential(llm, trace, corpus, strategy, sp=None, warmup=5):
    """Run trace sequentially and return per-request TTFTs."""
    if sp is None:
        sp = SamplingParams(max_tokens=1, temperature=0.0)

    needs_tree = strategy in ("apc_optimized", "apc_oracle", "apc_recency")
    kt = KnowledgeTree() if needs_tree else None
    doc_recency = {} if strategy == "apc_recency" else None
    ttfts = []

    for i, q in enumerate(trace):
        ordered = compute_ordering(q.doc_ids, strategy, kt, doc_recency)
        prompt, _ = build_rag_prompt(q.query_text, ordered, corpus, doc_order="original")

        t0 = time.perf_counter()
        _ = llm.generate([prompt], sp)
        elapsed = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            ttfts.append(elapsed)

        update_tree(kt, ordered, i)
        if doc_recency is not None:
            for d in ordered:
                doc_recency[d] = i

    return ttfts


def ttft_stats(ttfts):
    """Compute p50, p95, p99, mean from a list of TTFTs."""
    s = sorted(ttfts)
    n = len(s)
    if n == 0:
        return {}
    return {
        "n": n,
        "p50_ms": round(s[n // 2], 2),
        "p95_ms": round(s[int(n * 0.95)], 2),
        "p99_ms": round(s[int(n * 0.99)], 2),
        "mean_ms": round(sum(s) / n, 2),
        "std_ms": round(statistics.stdev(s), 2) if n > 1 else 0,
    }


# ===================================================================
# Experiment 1: Concurrent Load Test
# ===================================================================

def experiment_concurrent(model, corpus, trace, gpu_mem, max_model_len,
                          enforce_eager, dtype):
    """Measure TTFT at different concurrency (batch) levels.

    For each batch size C, the trace is processed in windows of C requests.
    All C requests in a window are submitted together via llm.generate().
    The knowledge tree is updated BETWEEN windows, not within.
    This simulates concurrent arrivals where ordering uses stale cache state.
    """
    print("\n" + "=" * 60)
    print("Experiment: Concurrent Load Test")
    print("=" * 60)

    batch_sizes = [1, 2, 4, 8]
    strategies = ["no_cache", "apc_retrieval", "apc_optimized"]
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    warmup_batches = 2
    results = {}

    for strategy in strategies:
        results[strategy] = {}
        for bs in batch_sizes:
            enable_apc = strategy != "no_cache"
            needs_tree = strategy in ("apc_optimized",)
            kt = KnowledgeTree() if needs_tree else None

            print(f"\n  [{strategy}] batch_size={bs}, loading LLM...")
            sys.stdout.flush()

            try:
                llm = make_llm(model, enable_apc, gpu_mem, max_model_len,
                               enforce_eager, dtype)
            except Exception as e:
                print(f"    FAILED to load: {e}")
                results[strategy][str(bs)] = {"error": str(e)}
                continue

            batch_ttfts = []  # per-batch wall clock times
            per_request_ttfts = []  # estimated per-request TTFTs

            num_batches = len(trace) // bs
            for batch_idx in range(num_batches):
                batch_start = batch_idx * bs
                batch_queries = trace[batch_start:batch_start + bs]

                # Build all prompts for this batch
                batch_prompts = []
                batch_ordered = []
                for q in batch_queries:
                    ordered = compute_ordering(q.doc_ids, strategy, kt)
                    prompt, _ = build_rag_prompt(
                        q.query_text, ordered, corpus, doc_order="original")
                    batch_prompts.append(prompt)
                    batch_ordered.append(ordered)

                # Submit batch
                t0 = time.perf_counter()
                _ = llm.generate(batch_prompts, sp)
                batch_wall = (time.perf_counter() - t0) * 1000

                if batch_idx >= warmup_batches:
                    batch_ttfts.append(batch_wall)
                    per_request_ttfts.append(batch_wall / bs)

                # Update tree after batch completes
                for j, ordered in enumerate(batch_ordered):
                    update_tree(kt, ordered, batch_start + j)

            # Aggregate
            if per_request_ttfts:
                s = sorted(per_request_ttfts)
                n = len(s)
                results[strategy][str(bs)] = {
                    "batch_size": bs,
                    "num_batches": n,
                    "avg_per_request_ms": round(sum(s) / n, 2),
                    "p50_per_request_ms": round(s[n // 2], 2),
                    "p95_per_request_ms": round(s[int(n * 0.95)], 2),
                    "batch_wall_p50_ms": round(sorted(batch_ttfts)[len(batch_ttfts) // 2], 2),
                    "throughput_req_s": round(bs / (sum(batch_ttfts) / len(batch_ttfts) / 1000), 2),
                }
                print(f"    bs={bs}: avg TTFT={results[strategy][str(bs)]['avg_per_request_ms']:.1f}ms, "
                      f"throughput={results[strategy][str(bs)]['throughput_req_s']:.1f} req/s")

            del llm; cleanup()

    # Compute improvement at each batch size
    improvements = {}
    for bs in batch_sizes:
        bs_key = str(bs)
        if (bs_key in results.get("apc_retrieval", {}) and
            bs_key in results.get("apc_optimized", {}) and
            "error" not in results["apc_retrieval"][bs_key] and
            "error" not in results["apc_optimized"][bs_key]):
            retr = results["apc_retrieval"][bs_key]["avg_per_request_ms"]
            opt = results["apc_optimized"][bs_key]["avg_per_request_ms"]
            improvements[bs_key] = round((retr - opt) / retr * 100, 1)
            print(f"  Improvement at bs={bs}: {improvements[bs_key]:.1f}%")

    results["improvements"] = improvements
    return results


# ===================================================================
# Experiment 2: Stronger Baselines
# ===================================================================

def experiment_baselines(model, corpus, trace, gpu_mem, max_model_len,
                         enforce_eager, dtype):
    """Compare all ordering strategies including recency and oracle."""
    print("\n" + "=" * 60)
    print("Experiment: Stronger Baselines")
    print("=" * 60)

    strategies = ["no_cache", "apc_retrieval", "apc_sorted",
                  "apc_recency", "apc_oracle", "apc_optimized"]
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    results = {}

    for strategy in strategies:
        enable_apc = strategy != "no_cache"
        print(f"\n  [{strategy}] Loading LLM...")
        sys.stdout.flush()

        try:
            llm = make_llm(model, enable_apc, gpu_mem, max_model_len,
                           enforce_eager, dtype)
        except Exception as e:
            print(f"    FAILED: {e}")
            results[strategy] = {"error": str(e)}
            continue

        ttfts = run_trace_sequential(llm, trace, corpus, strategy, sp)
        results[strategy] = ttft_stats(ttfts)
        print(f"    p50={results[strategy]['p50_ms']:.1f}ms, "
              f"mean={results[strategy]['mean_ms']:.1f}ms")

        del llm; cleanup()

    # Compute all pairwise improvements vs retrieval-order
    if "apc_retrieval" in results and "p50_ms" in results.get("apc_retrieval", {}):
        base = results["apc_retrieval"]["p50_ms"]
        improvements = {}
        for s in strategies:
            if s != "apc_retrieval" and "p50_ms" in results.get(s, {}):
                improvements[s] = round((base - results[s]["p50_ms"]) / base * 100, 1)
        results["improvements_vs_retrieval"] = improvements
        print(f"\n  Improvements vs retrieval-order:")
        for s, imp in improvements.items():
            print(f"    {s}: {imp:+.1f}%")

    return results


# ===================================================================
# Experiment 3: Multi-Seed Statistical Rigor
# ===================================================================

def experiment_multi_seed(model, corpus_args, gpu_mem, max_model_len,
                          enforce_eager, dtype, num_docs, num_queries, top_k,
                          overlap):
    """Run main comparison with multiple seeds, report mean +/- std and 95% CI."""
    print("\n" + "=" * 60)
    print("Experiment: Multi-Seed Statistical Rigor")
    print("=" * 60)

    seeds = [42, 123, 456]
    strategies = ["no_cache", "apc_retrieval", "apc_optimized"]
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    # Collect per-seed results
    seed_results = {s: [] for s in strategies}

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        corpus = generate_corpus(num_docs=num_docs)
        trace = generate_rag_trace(corpus, num_queries=num_queries,
                                   top_k=top_k, overlap_fraction=overlap,
                                   seed=seed)

        for strategy in strategies:
            enable_apc = strategy != "no_cache"
            print(f"  [{strategy}] Loading LLM (seed={seed})...")
            sys.stdout.flush()

            llm = make_llm(model, enable_apc, gpu_mem, max_model_len,
                           enforce_eager, dtype)
            ttfts = run_trace_sequential(llm, trace, corpus, strategy, sp)
            stats = ttft_stats(ttfts)
            seed_results[strategy].append(stats)
            print(f"    p50={stats['p50_ms']:.1f}ms")

            del llm; cleanup()

    # Aggregate across seeds
    results = {}
    for strategy in strategies:
        p50s = [r["p50_ms"] for r in seed_results[strategy]]
        means = [r["mean_ms"] for r in seed_results[strategy]]
        n = len(p50s)
        p50_mean = sum(p50s) / n
        p50_std = statistics.stdev(p50s) if n > 1 else 0
        mean_mean = sum(means) / n
        mean_std = statistics.stdev(means) if n > 1 else 0

        # 95% CI: mean +/- t * std / sqrt(n), t=4.303 for df=2
        t_val = 4.303  # t-distribution, 95% CI, df=2
        p50_ci = t_val * p50_std / math.sqrt(n)
        mean_ci = t_val * mean_std / math.sqrt(n)

        results[strategy] = {
            "per_seed": seed_results[strategy],
            "p50_mean": round(p50_mean, 2),
            "p50_std": round(p50_std, 2),
            "p50_ci95": round(p50_ci, 2),
            "mean_mean": round(mean_mean, 2),
            "mean_std": round(mean_std, 2),
            "mean_ci95": round(mean_ci, 2),
        }
        print(f"\n  [{strategy}] p50: {p50_mean:.1f} ± {p50_std:.1f}ms "
              f"(95% CI: ±{p50_ci:.1f}ms)")

    # Improvement with CI
    if "apc_retrieval" in results and "apc_optimized" in results:
        retr_p50s = [r["p50_ms"] for r in seed_results["apc_retrieval"]]
        opt_p50s = [r["p50_ms"] for r in seed_results["apc_optimized"]]
        diffs = [r - o for r, o in zip(retr_p50s, opt_p50s)]
        diff_mean = sum(diffs) / len(diffs)
        diff_std = statistics.stdev(diffs) if len(diffs) > 1 else 0
        diff_ci = 4.303 * diff_std / math.sqrt(len(diffs))
        results["improvement"] = {
            "diff_mean_ms": round(diff_mean, 2),
            "diff_std_ms": round(diff_std, 2),
            "diff_ci95_ms": round(diff_ci, 2),
            "pct_improvement": round(
                diff_mean / results["apc_retrieval"]["p50_mean"] * 100, 1),
            "significant": diff_mean - diff_ci > 0,  # CI doesn't cross 0
        }
        print(f"\n  Improvement: {results['improvement']['diff_mean_ms']:.1f} ± "
              f"{results['improvement']['diff_ci95_ms']:.1f}ms "
              f"({results['improvement']['pct_improvement']:.1f}%), "
              f"significant={results['improvement']['significant']}")

    return results


# ===================================================================
# Experiment 4: Cache Eviction Pressure
# ===================================================================

def experiment_eviction(model, corpus, trace, max_model_len, enforce_eager,
                        dtype):
    """Test ordering benefit under reduced GPU memory (cache pressure)."""
    print("\n" + "=" * 60)
    print("Experiment: Cache Eviction Pressure")
    print("=" * 60)

    gpu_mem_levels = [0.78, 0.82, 0.86, 0.90]
    strategies = ["apc_retrieval", "apc_optimized"]
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    results = {}

    for gm in gpu_mem_levels:
        results[str(gm)] = {}
        print(f"\n  --- gpu_mem={gm} ---")

        for strategy in strategies:
            print(f"  [{strategy}] Loading LLM (gpu_mem={gm})...")
            sys.stdout.flush()

            try:
                llm = make_llm(model, True, gm, max_model_len,
                               enforce_eager, dtype)
            except Exception as e:
                print(f"    FAILED (OOM likely): {e}")
                results[str(gm)][strategy] = {"error": str(e)}
                cleanup()
                continue

            ttfts = run_trace_sequential(llm, trace, corpus, strategy, sp)
            results[str(gm)][strategy] = ttft_stats(ttfts)
            print(f"    p50={results[str(gm)][strategy]['p50_ms']:.1f}ms")

            del llm; cleanup()

        # Compute improvement at this memory level
        r = results[str(gm)]
        if ("apc_retrieval" in r and "apc_optimized" in r and
            "p50_ms" in r.get("apc_retrieval", {}) and
            "p50_ms" in r.get("apc_optimized", {})):
            base = r["apc_retrieval"]["p50_ms"]
            opt = r["apc_optimized"]["p50_ms"]
            r["improvement_pct"] = round((base - opt) / base * 100, 1)
            print(f"  Improvement at gpu_mem={gm}: {r['improvement_pct']:.1f}%")

    return results


# ===================================================================
# Experiment 5: Quality Check with 7B Model
# ===================================================================

def experiment_quality_7b(model, gpu_mem, max_model_len, enforce_eager, dtype,
                          num_examples=200):
    """HotpotQA quality check with larger model."""
    print("\n" + "=" * 60)
    print("Experiment: Quality Check (7B, HotpotQA)")
    print("=" * 60)

    # Try loading HotpotQA
    try:
        from datasets import load_dataset
        print(f"  Loading HotpotQA ({num_examples} examples)...")
        ds = load_dataset("hotpot_qa", "fullwiki",
                          split=f"validation[:{num_examples}]")
    except ImportError:
        print("  ERROR: 'datasets' package not installed.")
        print("  Install with: pip install datasets")
        return {"error": "datasets not installed"}
    except Exception as e:
        print(f"  ERROR loading HotpotQA: {e}")
        return {"error": str(e)}

    import re
    import string
    from collections import Counter

    def normalize(s):
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = s.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(s.split())

    def em(pred, gold):
        return float(normalize(pred) == normalize(gold))

    def f1(pred, gold):
        pt = normalize(pred).split()
        gt = normalize(gold).split()
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            return 0.0
        p = ns / len(pt) if pt else 0
        r = ns / len(gt) if gt else 0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    # Prepare examples
    examples = []
    for item in ds:
        ctx = item["context"]
        doc_ids = []
        doc_contents = {}
        for title, sentences in zip(ctx["title"][:5], ctx["sentences"][:5]):
            doc_id = title.replace(" ", "_")[:50]
            if doc_id in doc_contents:
                doc_id = f"{doc_id}_{len(doc_ids)}"
            doc_ids.append(doc_id)
            text = f"{title}\n" + " ".join(sentences)
            words = text.split()
            if len(words) > 150:  # Slightly longer for 7B
                text = " ".join(words[:150]) + "..."
            doc_contents[doc_id] = text
        examples.append({
            "question": item["question"],
            "answer": item["answer"],
            "doc_ids": doc_ids,
            "doc_contents": doc_contents,
        })
    print(f"  Prepared {len(examples)} examples")

    strategies = ["no_cache", "apc_retrieval", "apc_optimized"]
    sp = SamplingParams(max_tokens=80, temperature=0.0)
    warmup = 3
    results = {}

    for strategy in strategies:
        enable_apc = strategy != "no_cache"
        kt = KnowledgeTree() if strategy == "apc_optimized" else None
        print(f"\n  [{strategy}] Loading LLM...")
        sys.stdout.flush()

        llm = make_llm(model, enable_apc, gpu_mem, max_model_len,
                        enforce_eager, dtype)

        em_scores, f1_scores, ttfts = [], [], []
        for i, ex in enumerate(examples):
            ordered = compute_ordering(ex["doc_ids"], strategy, kt)
            prompt, _ = build_rag_prompt(
                ex["question"], ordered, ex["doc_contents"],
                doc_order="original")

            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sp)
            elapsed = (time.perf_counter() - t0) * 1000

            pred = outputs[0].outputs[0].text.strip()

            if i >= warmup:
                em_scores.append(em(pred, ex["answer"]))
                f1_scores.append(f1(pred, ex["answer"]))
                ttfts.append(elapsed)

            update_tree(kt, ordered, i)

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(examples)} done")

        n = len(em_scores)
        results[strategy] = {
            "n": n,
            "em": round(sum(em_scores) / n, 4),
            "f1": round(sum(f1_scores) / n, 4),
            "ttft_p50_ms": round(sorted(ttfts)[n // 2], 2),
            "ttft_mean_ms": round(sum(ttfts) / n, 2),
        }
        print(f"    EM={results[strategy]['em']:.3f}, "
              f"F1={results[strategy]['f1']:.3f}, "
              f"TTFT p50={results[strategy]['ttft_p50_ms']:.1f}ms")

        del llm; cleanup()

    # Quality delta
    if "apc_retrieval" in results and "apc_optimized" in results:
        results["quality_delta"] = {
            "em_diff": round(results["apc_optimized"]["em"] -
                           results["apc_retrieval"]["em"], 4),
            "f1_diff": round(results["apc_optimized"]["f1"] -
                           results["apc_retrieval"]["f1"], 4),
        }

    return results


# ===================================================================
# Experiment 6: Zero-Overlap Investigation
# ===================================================================

def experiment_overlap_debug(model, gpu_mem, max_model_len, enforce_eager,
                             dtype, num_docs=500, top_k=5):
    """Investigate the zero-overlap anomaly.

    The original trace generator uses max(1, int(k * overlap)) which forces
    at least 1 shared doc even at overlap=0.0. We test:
      (a) Standard overlap=0.0 (actually Jaccard > 0 due to region bursts)
      (b) True zero overlap (queries from DIFFERENT regions, no shared docs)
    to isolate the ordering effect from natural region-based overlap.
    """
    print("\n" + "=" * 60)
    print("Experiment: Zero-Overlap Investigation")
    print("=" * 60)

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    corpus = generate_corpus(num_docs=num_docs)
    results = {}

    # (a) Standard overlap=0.0 (region-burst, max(1,...) forces sharing)
    print("\n  --- Part A: Standard overlap=0.0 (with region bursts) ---")
    trace_std = generate_rag_trace(corpus, num_queries=200, top_k=top_k,
                                   overlap_fraction=0.0, seed=42)
    # Measure actual Jaccard
    overlaps = []
    for i in range(1, len(trace_std)):
        prev = set(trace_std[i - 1].doc_ids)
        curr = set(trace_std[i].doc_ids)
        if prev | curr:
            overlaps.append(len(prev & curr) / len(prev | curr))
    jaccard_std = sum(overlaps) / len(overlaps) if overlaps else 0
    print(f"  Actual Jaccard (overlap=0.0): {jaccard_std:.3f}")
    results["standard_overlap_0"] = {"jaccard": round(jaccard_std, 4)}

    for strategy in ["apc_retrieval", "apc_optimized"]:
        llm = make_llm(model, True, gpu_mem, max_model_len, enforce_eager, dtype)
        ttfts = run_trace_sequential(llm, trace_std, corpus, strategy, sp)
        results["standard_overlap_0"][strategy] = ttft_stats(ttfts)
        print(f"  [{strategy}] p50={results['standard_overlap_0'][strategy]['p50_ms']:.1f}ms")
        del llm; cleanup()

    # Compute improvement
    if ("apc_retrieval" in results["standard_overlap_0"] and
        "apc_optimized" in results["standard_overlap_0"]):
        r = results["standard_overlap_0"]
        base = r["apc_retrieval"]["p50_ms"]
        opt = r["apc_optimized"]["p50_ms"]
        r["improvement_pct"] = round((base - opt) / base * 100, 1)
        print(f"  Improvement (standard overlap=0.0): {r['improvement_pct']:.1f}%")

    # (b) True zero overlap: consecutive queries from DIFFERENT regions
    print("\n  --- Part B: True zero overlap (different regions) ---")
    rng = random.Random(42)
    doc_ids_all = sorted(corpus.keys())
    num_regions = len(doc_ids_all) // 10
    true_zero_trace = []

    for q_idx in range(200):
        # Each query picks a DIFFERENT region than previous
        region = q_idx % num_regions
        region_docs = [d for d in doc_ids_all
                       if int(d.split("_")[1]) // 10 == region]
        selected = rng.sample(region_docs, min(top_k, len(region_docs)))
        true_zero_trace.append(RAGQuery(
            query_id=f"tz_{q_idx:04d}",
            query_text=f"Tell me about region {region}",
            doc_ids=selected,
            region=region,
        ))

    # Measure actual Jaccard
    overlaps_tz = []
    for i in range(1, len(true_zero_trace)):
        prev = set(true_zero_trace[i - 1].doc_ids)
        curr = set(true_zero_trace[i].doc_ids)
        if prev | curr:
            overlaps_tz.append(len(prev & curr) / len(prev | curr))
    jaccard_tz = sum(overlaps_tz) / len(overlaps_tz) if overlaps_tz else 0
    print(f"  Actual Jaccard (true zero): {jaccard_tz:.3f}")
    results["true_zero_overlap"] = {"jaccard": round(jaccard_tz, 4)}

    for strategy in ["apc_retrieval", "apc_optimized"]:
        llm = make_llm(model, True, gpu_mem, max_model_len, enforce_eager, dtype)
        ttfts = run_trace_sequential(llm, true_zero_trace, corpus, strategy, sp)
        results["true_zero_overlap"][strategy] = ttft_stats(ttfts)
        print(f"  [{strategy}] p50={results['true_zero_overlap'][strategy]['p50_ms']:.1f}ms")
        del llm; cleanup()

    # Compute improvement
    if ("apc_retrieval" in results["true_zero_overlap"] and
        "apc_optimized" in results["true_zero_overlap"]):
        r = results["true_zero_overlap"]
        base = r["apc_retrieval"]["p50_ms"]
        opt = r["apc_optimized"]["p50_ms"]
        r["improvement_pct"] = round((base - opt) / base * 100, 1)
        print(f"  Improvement (true zero overlap): {r['improvement_pct']:.1f}%")

    # Explanation
    results["finding"] = (
        "The 'overlap=0.0' setting in generate_rag_trace still forces "
        "max(1, int(k * overlap)) = 1 shared doc plus region-burst sampling, "
        "resulting in Jaccard > 0. True zero overlap (different regions per query) "
        "shows the genuine baseline."
    )
    return results


# ===================================================================
# Main
# ===================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAGCache++ Reviewer Benchmarks")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-docs", type=int, default=500)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--overlap", type=float, default=0.6)
    parser.add_argument("--output", default=None)
    parser.add_argument("--experiments", default="all",
                        help="Comma-separated: concurrent,baselines,multi_seed,"
                             "eviction,quality_7b,overlap_debug")
    args = parser.parse_args()

    print("=" * 60)
    print("RAGCache++ Reviewer Benchmarks")
    print(f"  Model:  {args.model}")
    print(f"  GPU:    {get_gpu_memory_mb()}")
    print(f"  Docs:   {args.num_docs}, Queries: {args.num_queries}")
    print("=" * 60)
    sys.stdout.flush()

    # Generate shared workload
    corpus = generate_corpus(num_docs=args.num_docs)
    trace = generate_rag_trace(corpus, num_queries=args.num_queries,
                               top_k=args.top_k,
                               overlap_fraction=args.overlap)

    common = dict(model=args.model, corpus=corpus, trace=trace,
                  gpu_mem=args.gpu_mem, max_model_len=args.max_model_len,
                  enforce_eager=args.enforce_eager, dtype=args.dtype)

    ALL_EXPS = ["concurrent", "baselines", "multi_seed", "eviction",
                "quality_7b", "overlap_debug"]
    exps = args.experiments.split(",") if args.experiments != "all" else ALL_EXPS

    out_path = args.output or os.path.join(RESULTS_DIR, "reviewer_results.json")
    results = {
        "config": {
            "model": args.model,
            "num_docs": args.num_docs,
            "num_queries": args.num_queries,
            "top_k": args.top_k,
            "max_model_len": args.max_model_len,
            "gpu_mem": args.gpu_mem,
            "overlap": args.overlap,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if "concurrent" in exps:
        results["concurrent"] = experiment_concurrent(**common)
        _save(out_path, results)
        print(f"  [saved to {out_path}]")

    if "baselines" in exps:
        results["baselines"] = experiment_baselines(**common)
        _save(out_path, results)
        print(f"  [saved to {out_path}]")

    if "multi_seed" in exps:
        results["multi_seed"] = experiment_multi_seed(
            model=args.model, corpus_args=None,
            gpu_mem=args.gpu_mem, max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager, dtype=args.dtype,
            num_docs=args.num_docs, num_queries=args.num_queries,
            top_k=args.top_k, overlap=args.overlap,
        )
        _save(out_path, results)
        print(f"  [saved to {out_path}]")

    if "eviction" in exps:
        results["eviction"] = experiment_eviction(
            model=args.model, corpus=corpus, trace=trace,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager, dtype=args.dtype,
        )
        _save(out_path, results)
        print(f"  [saved to {out_path}]")

    if "quality_7b" in exps:
        results["quality_7b"] = experiment_quality_7b(
            model=args.model, gpu_mem=args.gpu_mem,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager, dtype=args.dtype,
            num_examples=200,
        )
        _save(out_path, results)
        print(f"  [saved to {out_path}]")

    if "overlap_debug" in exps:
        results["overlap_debug"] = experiment_overlap_debug(
            model=args.model, gpu_mem=args.gpu_mem,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager, dtype=args.dtype,
            num_docs=args.num_docs, top_k=args.top_k,
        )
        _save(out_path, results)
        print(f"  [saved to {out_path}]")

    print(f"\n{'=' * 60}")
    print(f"All results saved to {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
