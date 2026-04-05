"""RAGCache++ integrated serving layer over vLLM.

Connects all system components (cache_manager, pgdsf_policy, knowledge_tree)
with real vLLM inference and a TTFT-based cache-state feedback loop.

The feedback loop infers vLLM's actual cache state from TTFT observations:
when the knowledge tree predicts a prefix hit but the measured TTFT is close
to cold-start, the trie path is pruned to stay synchronized with reality.
"""
from __future__ import annotations
import gc, time
from typing import Optional

from ragcache_pp.cache.cache_manager import CacheManager
from ragcache_pp.cache.knowledge_tree import KnowledgeTree, KVCacheMetadata
from ragcache_pp.cache.pgdsf_policy import PGDSFPolicy
from ragcache_pp.config import CacheConfig

# ---------------------------------------------------------------------------
# Cache-state feedback: TTFT-based mismatch detection
# ---------------------------------------------------------------------------

class CacheStateFeedback:
    """Infer vLLM's actual cache state from TTFT observations.

    The knowledge tree tracks what we THINK is cached. But vLLM manages
    eviction internally (LRU-based). When the trie predicts a prefix hit
    but the actual TTFT is close to cold-start, we know vLLM evicted
    those blocks. This class detects such mismatches and triggers trie
    pruning to keep the trie synchronized with reality.
    """
    def __init__(self, cold_ttft_estimate_ms: float = 100.0,
                 mismatch_threshold: float = 0.7,
                 ema_alpha: float = 0.1, window_size: int = 20):
        self.cold_ttft = cold_ttft_estimate_ms
        self.mismatch_threshold = mismatch_threshold
        self.ema_alpha = ema_alpha
        self.window_size = window_size
        self.history: list[tuple[float, float]] = []
        self.mismatches = 0
        self.total_observations = 0
        self.pruned_paths = 0

    def update_cold_estimate(self, ttft_ms: float,
                             predicted_prefix_len: int, top_k: int) -> None:
        """Update running cold-start TTFT estimate from observed cache misses."""
        if predicted_prefix_len == 0:
            self.cold_ttft = (1 - self.ema_alpha) * self.cold_ttft + self.ema_alpha * ttft_ms

    def check_mismatch(self, predicted_prefix_len: int, top_k: int,
                       actual_ttft_ms: float) -> bool:
        """Return True if predicted prefix hit mismatches observed cold TTFT."""
        self.total_observations += 1
        predicted_reuse = predicted_prefix_len / max(top_k, 1)
        actual_reuse = max(0.0, 1.0 - actual_ttft_ms / max(self.cold_ttft, 1.0))
        self.history.append((predicted_reuse, actual_reuse))
        if len(self.history) > self.window_size:
            self.history.pop(0)
        if predicted_reuse > 0.3 and actual_reuse < 0.15:
            self.mismatches += 1
            return True
        return False

    def get_accuracy(self) -> float:
        if self.total_observations == 0:
            return 1.0
        return 1.0 - self.mismatches / self.total_observations

    def get_stats(self) -> dict:
        return {"cold_ttft_estimate_ms": round(self.cold_ttft, 2),
                "total_observations": self.total_observations,
                "mismatches": self.mismatches,
                "accuracy": round(self.get_accuracy(), 4),
                "pruned_paths": self.pruned_paths}

# ---------------------------------------------------------------------------
# Main integration proxy
# ---------------------------------------------------------------------------

class VLLMCacheProxy:
    """Integrated RAGCache++ serving proxy over vLLM.

    Connects: knowledge_tree + cache_manager + pgdsf_policy + vLLM
    with a TTFT-based cache-state feedback loop.

    Integration of all system components:
    1. cache_manager tracks multi-tier cache state
    2. knowledge_tree provides prefix-aware ordering
    3. pgdsf_policy drives eviction decisions
    4. CacheStateFeedback keeps trie synchronized with vLLM's actual state
    5. vLLM handles actual inference with APC
    """
    def __init__(self, model: str, gpu_mem: float = 0.9,
                 max_model_len: int = 4096, enforce_eager: bool = True,
                 enable_feedback: bool = True,
                 cache_config: Optional[CacheConfig] = None):
        from vllm import LLM  # noqa: F401
        self.llm = LLM(model=model, gpu_memory_utilization=gpu_mem,
                       max_model_len=max_model_len, enforce_eager=enforce_eager,
                       enable_prefix_caching=True, trust_remote_code=True,
                       disable_log_stats=True)
        if cache_config is None:
            cache_config = CacheConfig()
        self.cache_manager = CacheManager(
            config=cache_config, enable_spatial=False, enable_non_prefix_reuse=False)
        self.knowledge_tree: KnowledgeTree = self.cache_manager.tree
        self.feedback: Optional[CacheStateFeedback] = (
            CacheStateFeedback() if enable_feedback else None)
        self._query_count = 0
        self._request_log: list[dict] = []

    def serve_request(self, query: str, doc_ids: list[str],
                      doc_contents: dict[str, str], max_tokens: int = 1,
                      sampling_params=None) -> dict:
        """Serve a RAG request through the full integrated pipeline.

        Pipeline: greedy trie walk -> prompt build -> vLLM inference ->
        TTFT feedback -> trie pruning on mismatch -> cache state update.
        """
        from vllm import SamplingParams as SP
        if sampling_params is None:
            sampling_params = SP(max_tokens=max_tokens, temperature=0.0)
        self._query_count += 1

        # Step 1: Greedy trie walk for optimal ordering
        t0 = time.perf_counter()
        remaining = set(doc_ids)
        ordered: list[str] = []
        node = self.knowledge_tree.root
        predicted_prefix_len = 0
        while remaining:
            found = False
            for d in list(remaining):
                child = node.get_child(d)
                if (child is not None and child.kv_metadata is not None
                        and child.kv_metadata.tier != "none"):
                    ordered.append(d); remaining.discard(d)
                    node = child; predicted_prefix_len += 1
                    found = True; break
            if not found:
                for d in doc_ids:
                    if d in remaining:
                        ordered.append(d); remaining.discard(d)
                break
        ordered_ids = ordered
        ordering_ms = (time.perf_counter() - t0) * 1000

        # Step 2: Build prompt
        t0 = time.perf_counter()
        from ragcache_pp.vllm_integration.prompt_builder import build_rag_prompt
        prompt, _ = build_rag_prompt(query, ordered_ids, doc_contents,
                                     doc_order="original")
        build_ms = (time.perf_counter() - t0) * 1000

        # Step 3: vLLM inference
        t0 = time.perf_counter()
        outputs = self.llm.generate([prompt], sampling_params)
        ttft_ms = (time.perf_counter() - t0) * 1000
        output_text = outputs[0].outputs[0].text if outputs else ""

        # Step 4: Cache-state feedback
        mismatch = False; actual_reuse_frac = 0.0
        if self.feedback is not None:
            self.feedback.update_cold_estimate(ttft_ms, predicted_prefix_len,
                                               len(doc_ids))
            mismatch = self.feedback.check_mismatch(
                predicted_prefix_len, len(doc_ids), ttft_ms)
            actual_reuse_frac = max(
                0.0, 1.0 - ttft_ms / max(self.feedback.cold_ttft, 1.0))

        # Step 5: Trie pruning on mismatch
        if mismatch:
            prune_node = self.knowledge_tree.root
            for d in ordered_ids[:predicted_prefix_len]:
                child = prune_node.get_child(d)
                if child is not None and child.kv_metadata is not None:
                    child.kv_metadata.tier = "none"
                    child.kv_metadata.block_ids = []
                    prune_node = child
                else:
                    break
            if self.feedback:
                self.feedback.pruned_paths += 1

        # Step 6: Admit this sequence into cache
        tokens_per_doc = 200
        metadata_list: list[KVCacheMetadata] = []
        for i, d in enumerate(ordered_ids):
            meta = self.cache_manager.admit(d, tokens_per_doc, retrieval_rank=i + 1)
            if meta is None:
                meta = KVCacheMetadata(doc_id=d, num_tokens=tokens_per_doc,
                                       num_blocks=(tokens_per_doc + 15) // 16,
                                       tier="none")
            metadata_list.append(meta)
        self.cache_manager.insert_sequence(ordered_ids, metadata_list)
        self.cache_manager.step()

        # Collect metrics
        metrics = {
            "query_id": self._query_count,
            "ttft_ms": round(ttft_ms, 2),
            "ordering_ms": round(ordering_ms, 4),
            "build_ms": round(build_ms, 4),
            "predicted_prefix_len": predicted_prefix_len,
            "actual_reuse_fraction": round(actual_reuse_frac, 4),
            "mismatch_detected": mismatch,
            "output_text": output_text[:100],
            "cache_stats": {
                "tree_nodes": self.knowledge_tree.node_count,
                "gpu_util": self.cache_manager.gpu_allocator.utilization,
                "feedback_accuracy": (
                    self.feedback.get_accuracy() if self.feedback else None),
            },
        }
        self._request_log.append(metrics)
        return metrics

    def get_stats(self) -> dict:
        """Get aggregate system statistics."""
        return {
            "total_requests": self._query_count,
            "cache_stats": self.cache_manager.stats.as_dict(),
            "memory_profile": self.cache_manager.get_memory_profile(),
            "feedback_stats": self.feedback.get_stats() if self.feedback else None,
        }

    def cleanup(self) -> None:
        """Release resources."""
        del self.llm; gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------

class RAGCacheMiddleware:
    """FastAPI middleware wrapping VLLMCacheProxy as a deployable service.

    POST /v1/rag/generate  -- inference with cache-aware ordering
    GET  /v1/rag/stats     -- system statistics
    GET  /v1/rag/health    -- health check
    """
    def __init__(self, proxy: VLLMCacheProxy):
        self.proxy = proxy

    def create_app(self):
        """Create and return a FastAPI application."""
        try:
            from fastapi import FastAPI
            from pydantic import BaseModel
        except ImportError:
            print("FastAPI not installed. Install with: pip install fastapi uvicorn")
            return None

        app = FastAPI(title="RAGCache++ Middleware",
                     description="Cache-aware document ordering for low-latency RAG serving")

        class RAGRequest(BaseModel):
            query: str
            documents: list[dict]
            max_tokens: int = 128

        class RAGResponse(BaseModel):
            text: str
            ttft_ms: float
            predicted_prefix_len: int
            mismatch_detected: bool

        proxy = self.proxy

        @app.post("/v1/rag/generate", response_model=RAGResponse)
        async def generate(request: RAGRequest):
            doc_ids = [d["id"] for d in request.documents]
            doc_contents = {d["id"]: d["content"] for d in request.documents}
            result = proxy.serve_request(
                query=request.query, doc_ids=doc_ids,
                doc_contents=doc_contents, max_tokens=request.max_tokens)
            return RAGResponse(
                text=result["output_text"], ttft_ms=result["ttft_ms"],
                predicted_prefix_len=result["predicted_prefix_len"],
                mismatch_detected=result["mismatch_detected"])

        @app.get("/v1/rag/stats")
        async def stats():
            return proxy.get_stats()

        @app.get("/v1/rag/health")
        async def health():
            return {"status": "ok", "requests_served": proxy._query_count}

        return app
