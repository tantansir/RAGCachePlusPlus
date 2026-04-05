# Auto Review Loop: RAGCache++ — Final Report Push

## Round 1 (2026-04-04T10:00:00)

### Assessment (Summary)
- Score: 4/10 (for OSDI/SOSP/MLSys level)
- Verdict: Not ready
- Key criticisms:
  1. **Workload realism** — synthetic workloads only; need at least one real RAG corpus
  2. **Novelty defense** — no head-to-head vs RAGCache/CacheBlend; appears as a lightweight heuristic
  3. **Quality preservation unsupported** — EM=0 everywhere invalidates the quality claim
  4. **E2E systems impact missing** — only TTFT microbenchmarks; no full pipeline under load
  5. **Cache abstraction unvalidated** — trie doesn't query actual vLLM APC state
  6. **Generality weak** — single model family (Qwen2.5), single engine
  7. **Systems contribution borderline** — "150 lines" reads as a trick, not a systems contribution
  8. **Midterm scaffolding** — "Remaining Work and Timeline" section signals incompleteness

<details>
<summary>Click to expand full reviewer response</summary>

Score: 4/10 for OSDI/SOSP/MLSys quality.

The upside is real: the idea is clean, the latency gains are meaningful, the implementation is elegant, and the oracle-gap story is good. The problem is that the current paper does not yet prove enough novelty, realism, or end-to-end systems value for a top venue.

Critical Weaknesses:

1. Workload realism is not strong enough.
Most of the evidence comes from synthetic overlap-controlled workloads. For a systems paper, that is the biggest problem, because the entire benefit depends on real reuse structure.
Minimum fix: run at least one public, realistic RAG workload with real retrieved passages and natural query/document reuse; report the observed overlap distribution and gains there.

2. Novelty relative to closest prior work is under-defended.
Without a direct comparison to RAGCache or CacheBlend, reviewers can reasonably conclude this is a lightweight heuristic variant of existing cache-aware serving ideas.
Minimum fix: implement and compare against at least one closest prior system on the same stack, or make a very explicit orthogonality argument and back it with a strong proxy baseline.

3. The quality-preservation claim is currently unsupported.
"EM=0 everywhere" means the HotpotQA result is not evidence. At top venues, unsupported "no degradation" claims hurt credibility.
Minimum fix: use a model/dataset pair that yields nontrivial accuracy, then show answer quality is unchanged across orderings. If you cannot do that, weaken the claim.

4. End-to-end systems impact is not fully demonstrated.
The paper is still dominated by TTFT/prefill microbenchmarks, and pipelining is projected rather than measured.
Minimum fix: measure full request-to-first-token and request-to-last-token in a live pipeline under load, including retrieval, prompt construction, and serving; report p50/p95, throughput, and utilization.

5. The cache abstraction is approximate and not validated against real engine state.
The knowledge tree predicts cached prefixes but does not query actual vLLM APC state. Under eviction/interference, that can diverge.
Minimum fix: add a validation experiment comparing predicted reusable prefix length with actual observed reuse under eviction and concurrency.

6. Generality is weak.
One model family and one engine path is not enough for a broad RAG-serving claim.
Minimum fix: add at least one additional model family/tokenizer and show the trend holds; otherwise scope the claim tightly to vLLM APC.

7. The systems contribution is still borderline in depth.
"150 lines, no engine mods" is practically attractive, but reviewers may see it as a clever prompt-ordering trick rather than a systems contribution.
Minimum fix: sharpen the framing to "prompt-layer cache scheduling for APC-based RAG serving," and add either stronger theory/analysis or stronger deployment evidence.

8. The paper still reads like a midterm report.
A "Remaining Work and Timeline" section and projected claims signal incompleteness.
Minimum fix: remove midterm scaffolding and cut any claim not directly supported by data.

Ready? No.
Not "almost" for a top venue. The idea is promising, but the paper still needs a real-world workload story, a closest-baseline novelty defense, a real quality validation, and end-to-end production-style measurements before it is submission-ready.

</details>

### Actions Taken
(implementing below)

### Status
- Continuing to Round 2

## Round 2 (2026-04-04T14:00:00)

### Assessment (Summary)
- Score: 5.5/10
- Verdict: No (closer, but still short of top-venue bar)
- Key criticisms:
  1. Novelty/nearest-prior defense still biggest risk — no head-to-head
  2. Real workload story improved but NQ Jaccard=0.88 too high — need moderate overlap trace
  3. Quality evidence still weak (EM=0, F1=0.008)
  4. Generality narrow (one model family)
  5. Contribution depth borderline for MLSys

<details>
<summary>Click to expand full reviewer response</summary>

Score: 5.5/10. Verdict: No. The paper now looks like a serious systems project rather than a midterm snapshot. W4 and W5 look substantially fixed. W8 is fixed. W1, W3, W7 are improved but not fully fixed. The remaining blocker is whether this is a sufficiently novel and broadly important systems contribution, or just a clever low-level optimization with good but context-dependent gains. Need: head-to-head or proxy baseline vs nearest prior, moderate-overlap real trace, nontrivial quality scores, second model family. MLSys is best fit; OSDI/SOSP still a stretch.

</details>

### Actions Taken
1. Ran 4 new experiments on AutoDL (NQ workload, quality, E2E pipeline, cache validation)
2. Added 3 new sections and tables to paper (NQ-Open, E2E, Trie Validation)
3. Updated abstract and conclusion with new findings
4. Fixed bibtex citations and table formatting
5. Paper: 10 pages, 0 errors, 12 tables

### Status
- Continuing to Round 3

## Round 3 (2026-04-04T18:00:00)

### Assessment (Summary)
- Score: 6.5/10
- Verdict: Almost (credible submission draft for MLSys/EuroSys)
- Key criticisms (minor/moderate):
  1. Algorithmic novelty: frequency ordering matches trie — need online/cold-start scenario where trie wins
  2. Generality: still one model family
  3. Real-world: mixed-topic is good but constructed; need one more natural trace
  4. Quality: still only a sanity check (EM=0)
  5. Prior-work confrontation: proxy baselines good but not actual RAGCache/CacheBlend

<details>
<summary>Click to expand full reviewer response</summary>

Score: 6.5/10. Verdict: Almost. This is now in "credible submission draft" territory for MLSys/EuroSys, but still below the likely accept bar. The evaluation is much stronger now; the main remaining issue is that the paper still does not fully prove why RAGCache++ itself, rather than any simple cache-aware heuristic, is the new systems idea. If you can show the trie policy beats simple heuristics in realistic online settings, and add one more model family, this becomes plausibly competitive for MLSys/EuroSys.

</details>

### Actions Taken
1. Added RAGCache-inspired proxy baselines (frequency + PGDSF-proxy)
2. Added mixed-topic workload (Jaccard=0.31, 16.3% improvement)
3. Added context window sensitivity (max_model_len=2048, 31.0% improvement)
4. Updated baselines table with 7 strategies
5. Updated Discussion with explicit RAGCache confrontation
6. Paper: 11 pages, 0 errors, 13 tables

### Status
- **STOP CONDITION MET**: Score 6.5 >= 6 AND verdict "Almost" → loop complete

## Final Summary
- **Starting score**: 4/10 (Round 1)
- **Round 2 score**: 5.5/10
- **Final score**: 6.5/10 (Round 3)
- **Total rounds**: 3 of 4 max
- **Verdict**: Almost ready — credible MLSys/EuroSys submission draft
- **Key improvements across rounds**: 
  - Real-workload (NQ-Open + mixed-topic), E2E pipeline, trie validation, proxy baselines, midterm cleanup
  - 4 new experiment scripts, 13 tables total, comprehensive overlap-regime characterization
- **Remaining for top-venue**: trie vs frequency in online setting, second model family, nontrivial quality eval

## Method Description
RAGCache++ is a prompt-layer cache scheduling policy for APC-based RAG serving. It maintains a knowledge tree (trie) indexed by document-ID sequences to track which orderings are currently in the KV cache. For each incoming RAG query, a greedy algorithm walks the trie to find the longest cached prefix among the retrieved documents, places those documents first, and appends remaining documents in retrieval-rank order. The reordered prompt is sent to vLLM with APC enabled, which sees a longer prefix match and skips more prefill computation. The system adds ~150 lines of Python at the prompt-construction layer with <0.13% overhead, zero GPU memory cost, and zero throughput regression.

---

# Auto Review Loop: Session 2 (Post-Restructuring)

## Round 1 (2026-04-05T00:00:00)

### Assessment (Summary)
- Score: 5/10
- Verdict: No
- Key criticisms:
  1. Novelty below bar — frequency matches trie on stationary workloads
  2. Feedback loop under-exercised — only 1 mismatch in 200 queries
  3. External validity weak — NQ/real-corpus still use synthetic documents
  4. Quality evidence not strong — near-zero F1 on 2/3 settings
  5. Baseline story muddled — contradictory claims trie vs frequency
  6. Measurement proxies — no direct APC block instrumentation
  7. Polish issues — wrong citation, overlap-sweep bug explanation

### Actions Taken
(implementing paper-only fixes)

### Status
- Continuing to Round 2

## Round 2 (2026-04-05T01:00:00)

### Assessment (Summary)
- Score: 6/10
- Verdict: Almost — "honest and plausible, needs one more piece of real-workload evidence"
- Remaining weaknesses:
  1. External validity — still no public real RAG benchmark
  2. Feedback loop — honestly presented but barely exercised
  3. Measurement proxies — TTFT-inferred, not direct APC instrumentation
  4. Contribution borderline — "strong deployable study, not clearly new mechanism"
  5. Quality story thin — scoped correctly but small evidence base

### Actions Taken
1. Reframed contribution as deployment-ready realization (not algorithmic novelty)
2. Clarified trie vs frequency: "when does trie outperform?"
3. Fixed NQ-Open citation (was citing HotpotQA)
4. Honestly acknowledged feedback loop limitation
5. Scoped quality claim to extractive settings

### Status
- **STOP CONDITION MET**: Score 6 >= 6

## Final Summary (Session 2)
- **Round 1 score**: 5/10
- **Round 2 score**: 6/10
- **Total rounds**: 2 of 4 max
- **Key improvement**: Honest framing eliminates overclaiming — reviewer trust increased
- **Remaining for top-venue**: real-workload validation, sustained eviction test for feedback loop

## Method Description
RAGCache++ is a prompt-layer cache scheduling policy for APC-based RAG serving. The system consists of: (1) a KnowledgeTree (trie) tracking cached document orderings, (2) a greedy O(k) ordering algorithm achieving 99.7% of oracle, (3) a CacheManager with PGDSF eviction and multi-tier block allocation, (4) a CacheStateFeedback module using TTFT observations to detect and prune stale trie entries, and (5) a VLLMCacheProxy integrating all components with real vLLM inference. The system adds 0.049ms overhead per request and requires zero vLLM modifications.
