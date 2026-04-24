#!/usr/bin/env python3
"""Integrated system benchmark: compares bare vLLM vs simple wrapper vs full pipeline.

Demonstrates that the full integrated pipeline with cache-state feedback
achieves better accuracy and latency than the simple wrapper, especially
under eviction pressure.

Experiments:
  1. full_pipeline    -- bare vLLM vs trie wrapper vs VLLMCacheProxy
  2. feedback_loop    -- eviction pressure: feedback-enabled vs stale-trie
  3. overhead_profile -- per-component latency breakdown (<1ms overhead)

Usage:
  python benchmark_integrated.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-model-len 4096 --gpu-mem 0.90 \
    --enforce-eager --experiments all \
    --output /path/to/results/integrated_results.json
"""
from __future__ import annotations
import gc, json, os, statistics, subprocess, sys, time
from typing import Optional

os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from vllm import LLM, SamplingParams

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
RESULTS_DIR = os.path.join(PROJ, "results")
from ragcache_pp.cache.knowledge_tree import KnowledgeTree, KVCacheMetadata
from ragcache_pp.vllm_integration.prompt_builder import build_rag_prompt, optimize_doc_order
from ragcache_pp.vllm_integration.benchmark_real import generate_corpus, generate_rag_trace, RAGQuery
from ragcache_pp.vllm_integration.serving import VLLMCacheProxy, CacheStateFeedback

# ── Utilities (same conventions as other benchmarks) ──────────────────

def get_gpu_memory_mb() -> dict:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,nounits,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            return {"used_mb": int(parts[0].strip()), "total_mb": int(parts[1].strip())}
    except Exception:
        pass
    return {"used_mb": 0, "total_mb": 0}

def make_llm(model, gpu_mem, max_model_len, enforce_eager):
    return LLM(model=model, gpu_memory_utilization=gpu_mem,
               max_model_len=max_model_len, enable_prefix_caching=True,
               trust_remote_code=True, enforce_eager=enforce_eager,
               disable_log_stats=True)

def cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2); f.flush(); os.fsync(f.fileno())

def ttft_stats(ttfts):
    s = sorted(ttfts); n = len(s)
    if n == 0: return {}
    return {"n": n, "p50_ms": round(s[n//2], 2), "p95_ms": round(s[int(n*0.95)], 2),
            "p99_ms": round(s[int(n*0.99)], 2), "mean_ms": round(sum(s)/n, 2),
            "std_ms": round(statistics.stdev(s), 2) if n > 1 else 0}

def update_tree(kt, ordered_ids, qi):
    if kt is None: return
    meta = [KVCacheMetadata(doc_id=d, num_tokens=200, num_blocks=13, tier="gpu",
                            created_at=qi, last_accessed_at=qi, access_count=1)
            for d in ordered_ids]
    kt.insert(ordered_ids, meta)

def _measure_prefix_len(kt, ordered):
    """Walk the trie to count predicted prefix length."""
    node, plen = kt.root, 0
    for d in ordered:
        child = node.get_child(d)
        if child and child.kv_metadata and child.kv_metadata.tier != "none":
            plen += 1; node = child
        else: break
    return plen

# ── Experiment 1: Full Pipeline Comparison ────────────────────────────

def experiment_full_pipeline(model, gpu_mem=0.90, max_model_len=4096,
                             enforce_eager=True, num_queries=200, overlap=0.6):
    """A: bare vLLM  B: simple trie wrapper  C: full VLLMCacheProxy."""
    print("\n" + "="*60 + "\nExperiment 1: Full Pipeline Comparison"
          f"\n  {num_queries} queries, overlap={overlap}\n" + "="*60); sys.stdout.flush()
    corpus = generate_corpus(500, 200)
    trace = generate_rag_trace(corpus, num_queries, 5, overlap)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    results = {}

    # Strategy A: bare vLLM (APC on, retrieval order)
    print("\n  [A] Bare vLLM (apc_retrieval) ..."); sys.stdout.flush()
    llm_a = make_llm(model, gpu_mem, max_model_len, enforce_eager)
    ttfts_a = []
    for q in trace:
        prompt, _ = build_rag_prompt(q.query_text, q.doc_ids, corpus)
        t0 = time.perf_counter(); llm_a.generate([prompt], sp)
        ttfts_a.append((time.perf_counter()-t0)*1000)
    results["bare_vllm"] = ttft_stats(ttfts_a)
    del llm_a; cleanup()
    print(f"    p50={results['bare_vllm']['p50_ms']:.1f}ms  mean={results['bare_vllm']['mean_ms']:.1f}ms")

    # Strategy B: simple trie wrapper
    print("\n  [B] Simple trie wrapper (apc_optimized) ..."); sys.stdout.flush()
    llm_b = make_llm(model, gpu_mem, max_model_len, enforce_eager)
    kt_b = KnowledgeTree(); ttfts_b = []; prefix_lens_b = []
    for qi, q in enumerate(trace):
        ordered = optimize_doc_order(q.doc_ids, kt_b)
        prefix_lens_b.append(_measure_prefix_len(kt_b, ordered))
        prompt, _ = build_rag_prompt(q.query_text, ordered, corpus)
        t0 = time.perf_counter(); llm_b.generate([prompt], sp)
        ttfts_b.append((time.perf_counter()-t0)*1000)
        update_tree(kt_b, ordered, qi)
    results["simple_wrapper"] = ttft_stats(ttfts_b)
    results["simple_wrapper"]["avg_predicted_prefix"] = round(sum(prefix_lens_b)/max(len(prefix_lens_b),1), 2)
    del llm_b; cleanup()
    print(f"    p50={results['simple_wrapper']['p50_ms']:.1f}ms  mean={results['simple_wrapper']['mean_ms']:.1f}ms  "
          f"avg_prefix={results['simple_wrapper']['avg_predicted_prefix']:.1f}")

    # Strategy C: full integrated proxy
    print("\n  [C] Full integrated proxy (VLLMCacheProxy) ..."); sys.stdout.flush()
    proxy = VLLMCacheProxy(model=model, gpu_mem=gpu_mem, max_model_len=max_model_len,
                           enforce_eager=enforce_eager, enable_feedback=True)
    ttfts_c = []; prefix_lens_c = []; mismatches_c = 0
    for q in trace:
        r = proxy.serve_request(query=q.query_text, doc_ids=q.doc_ids,
                                doc_contents=corpus, max_tokens=1)
        ttfts_c.append(r["ttft_ms"]); prefix_lens_c.append(r["predicted_prefix_len"])
        if r["mismatch_detected"]: mismatches_c += 1
    fb = proxy.feedback.get_stats() if proxy.feedback else {}
    ps = proxy.get_stats()
    results["integrated_proxy"] = ttft_stats(ttfts_c)
    results["integrated_proxy"].update({
        "avg_predicted_prefix": round(sum(prefix_lens_c)/max(len(prefix_lens_c),1), 2),
        "mismatches": mismatches_c,
        "feedback_accuracy": fb.get("accuracy"), "pruned_paths": fb.get("pruned_paths", 0),
        "cache_stats": ps.get("cache_stats", {}), "memory_profile": ps.get("memory_profile", {}),
    })
    proxy.cleanup(); cleanup()
    print(f"    p50={results['integrated_proxy']['p50_ms']:.1f}ms  mean={results['integrated_proxy']['mean_ms']:.1f}ms  "
          f"avg_prefix={results['integrated_proxy']['avg_predicted_prefix']:.1f}")
    print(f"    mismatches={mismatches_c}  feedback_accuracy={fb.get('accuracy','N/A')}")

    # Speedup summary
    base = results["bare_vllm"]["mean_ms"]
    for lbl in ("simple_wrapper", "integrated_proxy"):
        m = results[lbl]["mean_ms"]
        results[lbl]["speedup_vs_bare"] = round(base / m, 3) if base > 0 else 0
    return results

# ── Experiment 2: Feedback Loop Under Eviction Pressure ───────────────

def experiment_feedback_loop(model, gpu_mem=0.78, max_model_len=4096,
                             enforce_eager=True, num_queries=200, overlap=0.6):
    """Stale trie (no feedback) vs integrated proxy (feedback pruning) under tight memory."""
    print("\n" + "="*60 + "\nExperiment 2: Feedback Loop Under Eviction Pressure"
          f"\n  gpu_mem={gpu_mem} (tight), {num_queries} queries\n" + "="*60); sys.stdout.flush()
    corpus = generate_corpus(500, 200)
    trace = generate_rag_trace(corpus, num_queries, 5, overlap)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    results = {}

    # A: Simple wrapper with passive mismatch observation
    print("\n  [A] Simple wrapper (stale trie) ..."); sys.stdout.flush()
    llm_a = make_llm(model, gpu_mem, max_model_len, enforce_eager)
    kt_a = KnowledgeTree(); fb_a = CacheStateFeedback()
    ttfts_a = []; mismatches_a = 0
    for qi, q in enumerate(trace):
        ordered = optimize_doc_order(q.doc_ids, kt_a)
        plen = _measure_prefix_len(kt_a, ordered)
        prompt, _ = build_rag_prompt(q.query_text, ordered, corpus)
        t0 = time.perf_counter(); llm_a.generate([prompt], sp)
        ttft = (time.perf_counter()-t0)*1000; ttfts_a.append(ttft)
        fb_a.update_cold_estimate(ttft, plen, len(q.doc_ids))
        if fb_a.check_mismatch(plen, len(q.doc_ids), ttft): mismatches_a += 1
        update_tree(kt_a, ordered, qi)
    results["stale_trie"] = ttft_stats(ttfts_a)
    results["stale_trie"]["mismatches_observed"] = mismatches_a
    results["stale_trie"]["prediction_accuracy"] = round(fb_a.get_accuracy(), 4)
    del llm_a; cleanup()
    print(f"    p50={results['stale_trie']['p50_ms']:.1f}ms  mismatches={mismatches_a}  "
          f"accuracy={results['stale_trie']['prediction_accuracy']:.3f}")

    # B: Integrated proxy with feedback pruning
    print("\n  [B] Integrated proxy (feedback pruning) ..."); sys.stdout.flush()
    proxy = VLLMCacheProxy(model=model, gpu_mem=gpu_mem, max_model_len=max_model_len,
                           enforce_eager=enforce_eager, enable_feedback=True)
    ttfts_b = []; mismatches_b = 0
    for q in trace:
        r = proxy.serve_request(query=q.query_text, doc_ids=q.doc_ids,
                                doc_contents=corpus, max_tokens=1)
        ttfts_b.append(r["ttft_ms"])
        if r["mismatch_detected"]: mismatches_b += 1
    fb = proxy.feedback.get_stats() if proxy.feedback else {}
    results["feedback_proxy"] = ttft_stats(ttfts_b)
    results["feedback_proxy"]["mismatches_observed"] = mismatches_b
    results["feedback_proxy"]["prediction_accuracy"] = fb.get("accuracy")
    results["feedback_proxy"]["pruned_paths"] = fb.get("pruned_paths", 0)
    proxy.cleanup(); cleanup()
    print(f"    p50={results['feedback_proxy']['p50_ms']:.1f}ms  mismatches={mismatches_b}  "
          f"accuracy={results['feedback_proxy']['prediction_accuracy']}")

    results["comparison"] = {
        "stale_mismatches": mismatches_a, "feedback_mismatches": mismatches_b,
        "mismatch_reduction": round(1.0 - mismatches_b/max(mismatches_a,1), 3) if mismatches_a > 0 else 0.0,
        "stale_accuracy": results["stale_trie"]["prediction_accuracy"],
        "feedback_accuracy": results["feedback_proxy"]["prediction_accuracy"],
    }
    return results

# ── Experiment 3: System Overhead Profiling ───────────────────────────

def experiment_overhead_profile(model, gpu_mem=0.90, max_model_len=4096,
                                enforce_eager=True, num_queries=100, overlap=0.6):
    """Measure per-component latency; show <1ms overhead."""
    print("\n" + "="*60 + "\nExperiment 3: System Overhead Profiling"
          f"\n  {num_queries} queries\n" + "="*60); sys.stdout.flush()
    corpus = generate_corpus(500, 200)
    trace = generate_rag_trace(corpus, num_queries, 5, overlap)
    proxy = VLLMCacheProxy(model=model, gpu_mem=gpu_mem, max_model_len=max_model_len,
                           enforce_eager=enforce_eager, enable_feedback=True)
    ordering_t, build_t, inference_t = [], [], []
    for q in trace:
        r = proxy.serve_request(query=q.query_text, doc_ids=q.doc_ids,
                                doc_contents=corpus, max_tokens=1)
        ordering_t.append(r["ordering_ms"]); build_t.append(r["build_ms"])
        inference_t.append(max(0.0, r["ttft_ms"] - r["ordering_ms"] - r["build_ms"]))

    def _s(t):
        s = sorted(t); n = len(s)
        if n == 0: return {}
        return {"mean_ms": round(sum(s)/n, 4), "p50_ms": round(s[n//2], 4),
                "p99_ms": round(s[int(n*0.99)], 4), "max_ms": round(s[-1], 4)}

    oo, ob, oi = _s(ordering_t), _s(build_t), _s(inference_t)
    overhead = oo["mean_ms"] + ob["mean_ms"]
    results = {"ordering": oo, "prompt_build": ob, "inference": oi,
               "total_system_overhead_ms": round(overhead, 4),
               "overhead_fraction_of_ttft": round(overhead / max(oi["mean_ms"], 0.001), 6)}
    proxy.cleanup(); cleanup()

    print(f"\n  {'Component':<18} {'Mean':>10}  {'P50':>10}  {'P99':>10}")
    print(f"  {'-'*18} {'-'*10}  {'-'*10}  {'-'*10}")
    for lbl, d in [("Ordering", oo), ("Prompt build", ob), ("Inference", oi)]:
        print(f"  {lbl:<18} {d['mean_ms']:>10.4f}  {d['p50_ms']:>10.4f}  {d['p99_ms']:>10.4f}")
    print(f"\n  Total overhead: {overhead:.4f} ms/request  "
          f"({results['overhead_fraction_of_ttft']:.6f} of TTFT)")
    return results

# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="RAGCache++ Integrated System Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--overlap", type=float, default=0.6)
    parser.add_argument("--output", default=None,
                        help="Output JSON (default: <project>/results/integrated_results.json)")
    parser.add_argument("--experiments", default="all",
                        help="Comma-separated: full_pipeline,feedback_loop,overhead_profile")
    args = parser.parse_args()

    print("="*60 + "\nRAGCache++ Integrated System Benchmark")
    print(f"  Model: {args.model}\n  GPU: {get_gpu_memory_mb()}\n  GPU mem: {args.gpu_mem}")
    print(f"  Max len: {args.max_model_len}\n  Queries: {args.num_queries}\n  Overlap: {args.overlap}")
    print("="*60); sys.stdout.flush()

    ALL = ["full_pipeline", "feedback_loop", "overhead_profile"]
    exps = args.experiments.split(",") if args.experiments != "all" else ALL
    out_path = args.output or os.path.join(RESULTS_DIR, "integrated_results.json")
    results = {"config": {"model": args.model, "max_model_len": args.max_model_len,
                          "gpu_mem": args.gpu_mem, "enforce_eager": args.enforce_eager,
                          "num_queries": args.num_queries, "overlap": args.overlap},
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    for exp in ALL:
        if exp not in exps: continue
        fn = {"full_pipeline": experiment_full_pipeline,
              "feedback_loop": experiment_feedback_loop,
              "overhead_profile": experiment_overhead_profile}[exp]
        kw = dict(model=args.model, max_model_len=args.max_model_len,
                  enforce_eager=args.enforce_eager,
                  num_queries=min(args.num_queries, 100) if exp == "overhead_profile" else args.num_queries,
                  overlap=args.overlap)
        kw["gpu_mem"] = 0.78 if exp == "feedback_loop" else args.gpu_mem
        try:
            results[exp] = fn(**kw)
        except Exception as e:
            print(f"\n  ERROR in {exp}: {e}"); results[exp] = {"error": str(e)}
        _save(out_path, results); print(f"  [saved to {out_path}]")

    print(f"\n{'='*60}\nFinal Summary\n{'='*60}")
    for exp in ALL:
        if exp in results:
            ed = results[exp]
            status = "ERROR" if isinstance(ed, dict) and "error" in ed and len(ed) == 1 else "OK"
        else:
            status = "SKIPPED"
        print(f"  {exp}: {status}")
    print(f"\nAll results saved to {out_path}\n{'='*60}")

if __name__ == "__main__":
    main()
