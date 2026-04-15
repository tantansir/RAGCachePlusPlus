"""Wikipedia real-corpus benchmark — forces _generate_diverse_corpus path.

Reproduces Table 11: 1000 Wikipedia-style passages (20 topics), TF-IDF retrieval.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
import numpy as np

PROJ = "/root/ragcache_pp_project"
sys.path.insert(0, PROJ)

from ragcache_pp.vllm_integration.benchmark_round6 import (
    _generate_diverse_corpus, _run_strategies, RAGQuery,
)


def build_wiki_trace(num_passages=1000, num_queries=200, top_k=5):
    passages, queries_raw = _generate_diverse_corpus(num_passages, num_queries * 2)
    pids = list(passages.keys())

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(max_features=5000, stop_words='english')
    M = vec.fit_transform([passages[pid] for pid in pids])

    rng = random.Random(42)
    selected = rng.sample(queries_raw, min(num_queries, len(queries_raw)))
    trace = []
    jaccards = []
    prev = None
    for q_text in selected:
        sims = cosine_similarity(vec.transform([q_text]), M)[0]
        top = sims.argsort()[-top_k:][::-1]
        doc_ids = [pids[i] for i in top]
        cur = set(doc_ids)
        if prev is not None:
            u = len(cur | prev); i_ = len(cur & prev)
            jaccards.append(i_ / u if u else 0.0)
        prev = cur
        trace.append(RAGQuery(query_id=f"q_{len(trace):04d}",
                              query_text=q_text, doc_ids=doc_ids))
    return passages, trace, (float(np.mean(jaccards)) if jaccards else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--output", default="/root/ragcache_pp_project/wiki_corpus_results.json")
    a = ap.parse_args()

    print("="*60); print("Wikipedia Real-Corpus Benchmark"); print("="*60)
    print(f"  Model: {a.model}"); sys.stdout.flush()

    print("\n  Building corpus (20 topics x 50 passages = 1000) + TF-IDF retrieval...")
    corpus, trace, mean_jac = build_wiki_trace(1000, 200, 5)
    print(f"  Trace: {len(trace)} queries, mean Jaccard={mean_jac:.4f}")

    res, impr = _run_strategies(a.model, a.gpu_mem, a.max_model_len,
                                 a.enforce_eager, a.dtype, corpus, trace)
    output = {
        "config": {"model": a.model, "max_model_len": a.max_model_len,
                   "gpu_mem": a.gpu_mem, "enforce_eager": a.enforce_eager},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wiki_real_corpus": {
            "corpus_size": 1000, "num_queries": 200, "top_k": 5,
            "num_topics": 20, "mean_jaccard": round(mean_jac, 4),
            "no_cache": res.get("no_cache"),
            "apc_retrieval": res.get("apc_retrieval"),
            "apc_optimized": res.get("apc_optimized"),
            "improvements": impr,
        },
    }
    with open(a.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {a.output}")
    # Print key results
    opt = res.get("apc_optimized", {}); retr = res.get("apc_retrieval", {})
    if opt and retr:
        pct = 100 * (retr['p50_ms'] - opt['p50_ms']) / retr['p50_ms']
        print(f"\n  KEY: opt p50={opt['p50_ms']:.1f}, retr p50={retr['p50_ms']:.1f}, improvement=+{pct:.1f}%")


if __name__ == "__main__":
    main()
