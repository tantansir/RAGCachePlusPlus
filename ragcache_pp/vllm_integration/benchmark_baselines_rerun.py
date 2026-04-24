"""Single-run baselines benchmark — produces Table 8 from one consistent run.

Runs 6 strategies (no_cache, apc_sorted, apc_retrieval, apc_recency,
apc_optimized, apc_oracle) on the same synthetic bursty trace. All
percentages are computed against the same APC+Retrieval baseline.
"""
from __future__ import annotations
import argparse, gc, itertools, json, os, random, sys, time
from typing import Optional
import numpy as np

PROJ = "/root/ragcache_pp_project"
sys.path.insert(0, PROJ)

from vllm import LLM, SamplingParams
from ragcache_pp.cache.knowledge_tree import KnowledgeTree, KVCacheMetadata
from ragcache_pp.vllm_integration.prompt_builder import (
    build_rag_prompt, optimize_doc_order)
from ragcache_pp.vllm_integration.benchmark_real import (
    generate_corpus, generate_rag_trace, RAGQuery)


def make_llm(model, enable_apc, gpu_mem, max_model_len, enforce_eager, dtype):
    return LLM(model=model, gpu_memory_utilization=gpu_mem,
               max_model_len=max_model_len, enforce_eager=enforce_eager,
               enable_prefix_caching=enable_apc, trust_remote_code=True,
               disable_log_stats=True, dtype=dtype)


def cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def order_recency(doc_ids, recency):
    # Documents with most-recent access first, then unseen in retrieval order.
    seen = [(d, recency.get(d, -1)) for d in doc_ids]
    seen.sort(key=lambda p: -p[1])
    return [d for d, _ in seen]


def order_oracle(doc_ids, kt):
    """Exhaustive k! search for the best prefix match in the trie."""
    best_order, best_len = doc_ids, 0
    for perm in itertools.permutations(doc_ids):
        node = kt.root; length = 0
        for d in perm:
            ch = node.get_child(d)
            if (ch is not None and ch.kv_metadata is not None
                    and ch.kv_metadata.tier != "none"):
                node = ch; length += 1
            else:
                break
        if length > best_len:
            best_len = length; best_order = list(perm)
            if best_len == len(doc_ids):
                return best_order
    return best_order


def compute_ordering(doc_ids, strategy, kt=None, recency=None):
    if strategy == "apc_sorted":
        return sorted(doc_ids)
    if strategy == "apc_recency":
        return order_recency(doc_ids, recency or {})
    if strategy == "apc_optimized" and kt is not None:
        return optimize_doc_order(doc_ids, kt)
    if strategy == "apc_oracle" and kt is not None:
        return order_oracle(doc_ids, kt)
    return list(doc_ids)


def update_tree(kt, ordered_ids, query_idx):
    if kt is None:
        return
    metas = [KVCacheMetadata(doc_id=d, num_tokens=200, num_blocks=13,
                             tier="gpu", created_at=float(query_idx),
                             last_accessed_at=float(query_idx))
             for d in ordered_ids]
    kt.insert(ordered_ids, metas)


def p(x, q):
    return float(np.percentile(np.array(x, dtype=float), q)) if x else 0.0


def run_strategy(llm, trace, corpus, strategy, warmup=5):
    kt = KnowledgeTree() if strategy in ("apc_optimized", "apc_oracle") else None
    recency = {} if strategy == "apc_recency" else None
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    ttfts = []
    for i, q in enumerate(trace):
        ordered = compute_ordering(q.doc_ids, strategy, kt, recency)
        prompt, _ = build_rag_prompt(q.query_text, ordered, corpus,
                                     doc_order="original")
        t0 = time.perf_counter()
        _ = llm.generate([prompt], sp)
        ttft_ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            ttfts.append(ttft_ms)
        if recency is not None:
            for d in q.doc_ids:
                recency[d] = i
        if kt is not None:
            update_tree(kt, ordered, i)
    return {"n": len(ttfts),
            "p50_ms": round(p(ttfts, 50), 2),
            "p95_ms": round(p(ttfts, 95), 2),
            "mean_ms": round(float(np.mean(ttfts)) if ttfts else 0.0, 2),
            "std_ms": round(float(np.std(ttfts)) if ttfts else 0.0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--num-docs", type=int, default=500)
    ap.add_argument("--num-queries", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--overlap", type=float, default=0.6)
    ap.add_argument("--output",
                    default="/root/ragcache_pp_project/results/baselines_single_run.json")
    a = ap.parse_args()

    print("=" * 60)
    print("Single-Run Baselines: 6 strategies on one shared trace")
    print("=" * 60); sys.stdout.flush()

    corpus = generate_corpus(num_docs=a.num_docs)
    trace = generate_rag_trace(corpus, num_queries=a.num_queries,
                                top_k=a.top_k, overlap_fraction=a.overlap)
    print(f"  Trace: {len(trace)} queries, top-k={a.top_k}, overlap={a.overlap}")

    strategies = ["no_cache", "apc_sorted", "apc_retrieval",
                  "apc_recency", "apc_optimized", "apc_oracle"]

    results = {}
    for s in strategies:
        enable_apc = s != "no_cache"
        print(f"\n  [{s}] Loading LLM..."); sys.stdout.flush()
        try:
            llm = make_llm(a.model, enable_apc, a.gpu_mem, a.max_model_len,
                            a.enforce_eager, a.dtype)
        except Exception as e:
            print(f"    FAILED: {e}"); results[s] = {"error": str(e)}; continue
        results[s] = run_strategy(llm, trace, corpus, s)
        print(f"    p50={results[s]['p50_ms']} mean={results[s]['mean_ms']}")
        del llm; cleanup()

    # Derived %'s vs apc_retrieval baseline
    retr_p50 = results["apc_retrieval"]["p50_ms"]
    retr_mean = results["apc_retrieval"]["mean_ms"]
    improvements = {}
    for s in strategies:
        if s == "apc_retrieval" or "p50_ms" not in results.get(s, {}):
            continue
        improvements[s] = {
            "p50_pct": round(100 * (retr_p50 - results[s]["p50_ms"]) / retr_p50, 2),
            "mean_pct": round(100 * (retr_mean - results[s]["mean_ms"]) / retr_mean, 2),
        }
    greedy = improvements.get("apc_optimized", {}).get("p50_pct", 0)
    oracle = improvements.get("apc_oracle", {}).get("p50_pct", 1)
    greedy_vs_oracle_pct = round(100 * greedy / oracle, 1) if oracle else 0

    output = {
        "config": {"model": a.model, "num_docs": a.num_docs,
                   "num_queries": a.num_queries, "top_k": a.top_k,
                   "overlap": a.overlap, "gpu_mem": a.gpu_mem,
                   "max_model_len": a.max_model_len},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategies": results,
        "improvements_vs_retrieval": improvements,
        "greedy_vs_oracle_pct": greedy_vs_oracle_pct,
    }
    with open(a.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {a.output}")
    print(f"\n  Summary (vs APC+Retrieval p50):")
    for s, d in improvements.items():
        print(f"    {s}: p50 {d['p50_pct']:+.2f}%, mean {d['mean_pct']:+.2f}%")
    print(f"  Greedy vs Oracle: {greedy_vs_oracle_pct}%")


if __name__ == "__main__":
    main()
