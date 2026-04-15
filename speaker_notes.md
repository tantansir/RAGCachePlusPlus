# RAGCache++ — Speaker Script
## CMU 11-868 LLM Systems | Final Presentation | Spring 2026

**Estimated duration: 12–15 minutes**

---

### Slide 1: Title (30s)

Good morning/afternoon, everyone. I'm Kaizhen Tan, and together with Rong Gu and Mingyuan Li, we'll present RAGCache++, a prompt-layer scheduling system that reduces LLM serving latency in RAG applications through intelligent document ordering.

Our system achieves 20 to 33 percent TTFT reduction, reaches 99% of oracle performance, and requires only about 150 lines of Python — with absolutely zero modifications to the serving engine.

---

### Slide 2: The RAG Latency Problem (60s)

RAG improves LLM accuracy by injecting retrieved passages into the prompt, but this comes at a significant cost. Prompts become 2 to 10 times longer, and since prefill computation scales linearly with input length, this directly inflates Time-To-First-Token — the metric users feel most.

In a typical RAG pipeline, query embedding and vector retrieval take about 10 to 50 milliseconds. But the LLM prefill — step 4 — can take 50 to 200 or more milliseconds, dominating total latency. Our goal is to reduce this prefill cost without modifying the serving engine itself.

---

### Slide 3: vLLM APC Background (60s)

Modern serving engines like vLLM provide Automatic Prefix Caching. It works by managing KV cache memory in fixed-size blocks via PagedAttention, and reusing cached blocks when a new request shares an identical token-level prefix.

The critical detail is that APC uses chained hashes. If even a single token differs at position j, ALL subsequent blocks are invalidated. So look at this example: two RAG queries retrieve 60% of the same documents — A, B, C appear in both. But because the documents are in different positions, the token-level prefix diverges at the very first document, and ZERO blocks are reused.

This is the fundamental gap we target: RAG retrieval works at the document level, but prefix caching works at the token level. Different document ordering equals different prefix equals no cache reuse.

---

### Slide 4: RAGCache++ Approach (60s)

Our approach is straightforward. Since APC requires exact token-level prefix matches, we can maximize cache reuse simply by reordering documents to align with cached prefixes.

RAGCache++ has three core components. First, a knowledge tree — a trie indexed by document-ID sequences — that tracks which orderings are currently in the KV cache. Second, a greedy O(k) algorithm that walks the trie to find the longest cached prefix for each new request. Third, the entire system requires zero modifications to vLLM — it's about 150 lines of Python operating purely at the prompt-construction layer.

The analogy is a disk I/O scheduler: it reorders requests to exploit spatial locality without modifying the disk firmware. We do the same for KV cache locality.

---

### Slide 5: Greedy Ordering Algorithm (45s)

Here's the algorithm in detail. Starting at the trie root, we greedily extend the cached prefix. At each level, we look for a child node whose document ID is in our remaining candidate set. If we find a cached continuation, we append that document to our ordering and move deeper into the trie. When no cached continuation exists, we fall back to retrieval-rank order to preserve relevance ranking.

The algorithm is O(k) per request where k is typically 3 to 10 documents. On our 200-query trace, it achieves a mean optimality ratio of 0.997 — solving 98.5% of queries optimally.

---

### Slide 6: Cache-State Feedback Loop (60s)

A key challenge is that the trie tracks what we've served, not what's actually resident in vLLM's cache. Under memory pressure, vLLM may evict blocks that the trie thinks are cached, leading to stale predictions.

We close this gap with a TTFT-based feedback loop. The system maintains a running estimate of cold-start TTFT using an exponential moving average. After each request, it compares the trie's predicted reuse fraction with the TTFT-inferred actual reuse. When the trie predicts a significant cache hit but the observed TTFT is near-cold, we detect a mismatch and prune the stale path.

This achieves 99.5% prediction accuracy with only 0.049 milliseconds of overhead — less than 0.13% of TTFT. The trie-reality correlation is Pearson r = 0.972, and it remains stable even under memory pressure.

---

### Slide 7: System Architecture (45s)

Here's the full system architecture. The pipeline flows from the retriever, which produces top-k documents, through the knowledge tree for trie-based ordering, to the prompt builder for reordering and construction, then to vLLM with APC enabled for inference.

After inference, the TTFT-based feedback loop compares predicted and actual cache reuse, and feeds corrections back to the knowledge tree. The cache manager with PGDSF eviction policy tracks multi-tier cache state.

All of this is wrapped in VLLMCacheProxy, an integrated serving proxy, with a FastAPI middleware layer providing a standard API endpoint. Zero engine modifications.

---

### Slide 8: Main Results (60s)

Our main results show consistent TTFT reductions across two GPU configurations. The headline numbers: 20 to 33% TTFT reduction, 66% prefill saved, 99% of oracle performance, and less than 0.13% overhead.

Looking at the table: on the RTX 4060 Ti with Qwen2.5-1.5B, optimized ordering reduces median TTFT from 72.5 to 57.9 milliseconds — a 20% improvement. On the RTX 4090 with Qwen2.5-7B, from 59.6 to 40.1 — a 33% improvement.

Critically, these results are statistically significant. Across three independent trace seeds on the 4090, the improvement is 16.8 plus or minus 1.6 milliseconds, with the 95% confidence interval excluding zero.

---

### Slide 9: Key Finding — Depth > Count (60s)

One of our most interesting findings is counterintuitive: higher prefix hit rate does NOT always mean lower latency.

Look at the data: APC+Sorted achieves the highest hit rate at 25.6%, yet it has the highest TTFT among APC strategies — 81.2 milliseconds. Our optimized ordering has a lower hit rate — 19.0% — but achieves the lowest TTFT at 40.1 milliseconds.

Why? Because sorted ordering creates many short, fragmented prefix matches — matching only 1 or 2 documents before diverging. Our tree-based ordering creates fewer but much longer contiguous matches — matching 3 to 4 documents consecutively. Since prefill savings grow with prefix length, longer prefixes are disproportionately valuable.

This insight — that prefix DEPTH, not hit COUNT, drives latency reduction — is a key takeaway from our work.

---

### Slide 10: Robustness & Generality (45s)

Our approach is robust across multiple dimensions.

Concurrent load: ordering benefits persist at 7 to 10% across all batch sizes from 1 to 8.

GPU memory levels: improvement stays stable at 29 to 31% as we sweep gpu memory utilization from 0.78 to 0.90.

Cross-architecture: on the Qwen2.5 family — 0.5B, 1.5B, and 7B — we see 19%, 20%, and 33% improvements. On Phi-3-mini — a completely different architecture — we still get 18%.

Overlap sensitivity: peak improvement of 30% at moderate overlap, scaling up to 60% at k=10.

---

### Slide 11: Quality Preservation (45s)

A natural concern is whether reordering degrades answer quality. Our evaluation confirms it does not.

On extractive QA with 97 questions, both orderings achieve perfect exact match of 1.000. Delta EM equals zero.

For multi-hop reasoning — 40 two-hop examples with evidence chains — within each evidence configuration, our reordering produces identical quality. Delta EM equals zero, Delta F1 equals zero.

This is because we change only the order of passages, not their content. The model can still locate and extract answers regardless of document position.

---

### Slide 12: Diverse Workloads (45s)

Beyond synthetic benchmarks, we evaluate on four diverse workloads.

Geospatial with spatial locality: 4.5% improvement at Jaccard 0.17. NQ-Open with very high overlap: 2.3% — already near-optimal alignment. Mixed-topic with 30% topic switching: 16.3% at Jaccard 0.31. And a real corpus of 1000 Wikipedia passages with TF-IDF retrieval: 26.8% with no controlled overlap.

The pattern is clear: ordering benefits scale with moderate overlap — exactly the regime of real multi-topic RAG deployments.

---

### Slide 13: Systems Analysis (45s)

From a systems perspective, our overhead is minimal: 66 microseconds total, less than 0.13% of latency. Meanwhile, inference p50 drops by 29%.

Throughput is completely unaffected — ordering changes the prefill path, not decode.

GPU memory is identical — we change which blocks are reused, not the pool size.

For pipelining, we overlap system prompt prefill with retrieval, saving up to 26% on cold-start requests with a simple ThreadPoolExecutor — no engine modifications.

---

### Slide 14: Related Work (45s)

We position our work within a landscape of concurrent systems. RAGCache modifies the serving engine — we achieve comparable benefits at the prompt layer. ContextPilot independently converges on a similar approach, validating cache-aware ordering as a systems abstraction. CacheBlend and TurboRAG handle position-independent reuse — complementary to our approach.

Our specific contribution is the deployment-ready realization: engine-agnostic, formally near-optimal, with a closed-loop feedback mechanism and comprehensive evaluation.

---

### Slide 15: Conclusion (30s)

In conclusion, RAGCache++ demonstrates that a thin prompt-layer scheduling policy can recover substantial cache locality in RAG serving. 20 to 33% TTFT reduction, 66% prefill saved, 99% of oracle — all with 150 lines of Python and zero engine modifications.

Our key systems insight: prefix depth, not hit count, drives latency reduction in prefix-caching systems.

---

### Slide 16: Thank You (open)

Thank you for your attention. We're happy to take any questions.

---

**Total estimated time: ~12 minutes presentation + 3 minutes Q&A**
