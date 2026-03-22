"""RAG Controller: orchestrates the RAGCache++ serving pipeline.

Handles the full request lifecycle:
1. Query embedding + spatial constraint extraction
2. Retrieval (vector search + optional spatial filter)
3. Spatial prefetch trigger
4. Cache lookup (prefix match + non-prefix reuse)
5. Selective recomputation (for non-prefix cached chunks)
6. Prefill + decode
7. Cache update (admit new KV, update stats)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ragcache_pp.cache.cache_manager import CacheManager, CacheLookupResult
from ragcache_pp.cache.knowledge_tree import KVCacheMetadata
from ragcache_pp.cache.spatial_index import SpatialIndex
from ragcache_pp.config import RAGCachePPConfig


@dataclass
class RequestMetrics:
    """Per-request latency breakdown (all times in ms)."""

    retrieval_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    kv_movement_ms: float = 0.0
    selective_recompute_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    queueing_ms: float = 0.0
    spatial_policy_ms: float = 0.0

    # Derived
    total_ms: float = 0.0
    ttft_ms: float = 0.0  # time to first token = everything except decode

    # Cache info
    num_prefix_hits: int = 0
    num_non_prefix_hits: int = 0
    num_misses: int = 0
    num_prefetch_hits: int = 0
    tokens_recomputed: int = 0
    tokens_skipped: int = 0  # tokens served from cache
    flops_saved_fraction: float = 0.0

    def finalize(self) -> None:
        self.ttft_ms = (
            self.retrieval_ms + self.cache_lookup_ms + self.kv_movement_ms
            + self.selective_recompute_ms + self.prefill_ms + self.queueing_ms
            + self.spatial_policy_ms
        )
        self.total_ms = self.ttft_ms + self.decode_ms

    def as_dict(self) -> dict:
        return {
            "retrieval_ms": self.retrieval_ms,
            "cache_lookup_ms": self.cache_lookup_ms,
            "kv_movement_ms": self.kv_movement_ms,
            "selective_recompute_ms": self.selective_recompute_ms,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "queueing_ms": self.queueing_ms,
            "spatial_policy_ms": self.spatial_policy_ms,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "num_prefix_hits": self.num_prefix_hits,
            "num_non_prefix_hits": self.num_non_prefix_hits,
            "num_misses": self.num_misses,
            "num_prefetch_hits": self.num_prefetch_hits,
            "tokens_recomputed": self.tokens_recomputed,
            "tokens_skipped": self.tokens_skipped,
            "flops_saved_fraction": self.flops_saved_fraction,
        }


@dataclass
class RAGRequest:
    """A RAG serving request."""

    query_id: str
    query_text: str
    query_embedding: Optional[list[float]] = None

    # Spatial metadata (optional)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    spatial_radius_m: Optional[float] = None

    # Retrieved documents (filled by retrieval)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_doc_tokens: list[int] = field(default_factory=list)  # token counts
    retrieved_doc_geohashes: list[Optional[str]] = field(default_factory=list)
    retrieval_ranks: list[int] = field(default_factory=list)

    # Timing
    arrival_time: float = 0.0


class SimulatedLatencyModel:
    """Simulates latency for components not yet connected to real vLLM.

    Provides realistic timing based on RAGCache and CacheBlend measurements.
    Used for evaluation harness before full vLLM integration.
    """

    def __init__(self) -> None:
        # Base latencies from RAGCache paper measurements (ms)
        self.prefill_ms_per_token = 0.25  # ~250ms for 1000 tokens on A10G
        self.decode_ms_per_token = 5.0  # autoregressive decode
        self.retrieval_base_ms = 20.0  # vector search base
        self.retrieval_ms_per_doc = 2.0  # per-document scoring
        self.kv_movement_ms_per_block = 0.1  # host->GPU via PCIe
        self.recompute_ms_per_token = 0.3  # selective recompute per token
        self.cache_lookup_ms = 0.05  # tree traversal
        self.spatial_policy_ms = 0.02  # geohash + prefetch decision

    def estimate_prefill(self, num_tokens: int) -> float:
        return num_tokens * self.prefill_ms_per_token

    def estimate_decode(self, num_output_tokens: int) -> float:
        return num_output_tokens * self.decode_ms_per_token

    def estimate_retrieval(self, num_docs: int) -> float:
        return self.retrieval_base_ms + num_docs * self.retrieval_ms_per_doc

    def estimate_kv_movement(self, num_blocks: int) -> float:
        return num_blocks * self.kv_movement_ms_per_block

    def estimate_selective_recompute(self, num_tokens: int) -> float:
        return num_tokens * self.recompute_ms_per_token


class RAGController:
    """Main controller orchestrating the RAGCache++ pipeline.

    In simulation mode, uses SimulatedLatencyModel for timing.
    In production mode, integrates with vLLM engine.
    """

    def __init__(
        self,
        config: RAGCachePPConfig,
        cache_manager: CacheManager,
        spatial_index: Optional[SpatialIndex] = None,
        simulation_mode: bool = True,
    ):
        self.config = config
        self.cache_manager = cache_manager
        self.spatial_index = spatial_index
        self.simulation_mode = simulation_mode
        self.latency_model = SimulatedLatencyModel()

        self._request_count = 0
        self._all_metrics: list[RequestMetrics] = []

    def process_request(self, request: RAGRequest, output_tokens: int = 50) -> RequestMetrics:
        """Process a single RAG request through the full pipeline.

        Args:
            request: The RAG request with retrieved documents.
            output_tokens: Number of tokens to generate.

        Returns:
            Per-request latency breakdown.
        """
        metrics = RequestMetrics()
        self._request_count += 1

        # Step 1: Simulate retrieval (already done — timing only)
        metrics.retrieval_ms = self.latency_model.estimate_retrieval(len(request.retrieved_doc_ids))

        # Step 2: Spatial prefetch (concurrent with retrieval in real system)
        t0 = time.perf_counter()
        if (
            self.config.enable_spatial_policies
            and request.latitude is not None
            and request.longitude is not None
        ):
            self.cache_manager.trigger_prefetch(
                request.latitude, request.longitude
            )
        metrics.spatial_policy_ms = (time.perf_counter() - t0) * 1000

        # Step 3: Cache lookup (also updates policy priorities for hits)
        t0 = time.perf_counter()
        lookup_result = self.cache_manager.lookup(request.retrieved_doc_ids)
        metrics.cache_lookup_ms = (time.perf_counter() - t0) * 1000

        metrics.num_prefix_hits = lookup_result.prefix_length
        metrics.num_non_prefix_hits = len(lookup_result.non_prefix_reusable)
        metrics.num_misses = lookup_result.total_uncached
        metrics.num_prefetch_hits = len(lookup_result.prefetch_hits)

        # Record spatial cache hits (actual hits, not admissions)
        for node in lookup_result.prefix_matched:
            if node.kv_metadata and self.spatial_index is not None:
                savings = self.latency_model.estimate_prefill(node.kv_metadata.num_tokens)
                self.spatial_index.record_cache_hit(node.doc_id, savings)
                self.cache_manager.stats.total_prefill_savings_ms += savings
        for doc_id, node in lookup_result.non_prefix_reusable.items():
            if node.kv_metadata and self.spatial_index is not None:
                savings = self.latency_model.estimate_prefill(node.kv_metadata.num_tokens)
                self.spatial_index.record_cache_hit(doc_id, savings)
                self.cache_manager.stats.total_prefill_savings_ms += savings

        # Step 4: KV movement (host -> GPU for host-tier hits)
        # If promotion fails, downgrade that chunk to uncached for this request.
        host_blocks_to_move = 0
        failed_promotions: set[str] = set()
        for node in list(lookup_result.prefix_matched):
            if node.kv_metadata and node.kv_metadata.tier == "host":
                if self.cache_manager.promote_to_gpu(node):
                    host_blocks_to_move += node.kv_metadata.num_blocks
                else:
                    # Promotion failed — treat as uncached for this request
                    lookup_result.prefix_matched.remove(node)
                    lookup_result.prefix_length -= 1
                    failed_promotions.add(node.doc_id)
        for doc_id, node in list(lookup_result.non_prefix_reusable.items()):
            if node.kv_metadata and node.kv_metadata.tier == "host":
                if self.cache_manager.promote_to_gpu(node):
                    host_blocks_to_move += node.kv_metadata.num_blocks
                else:
                    del lookup_result.non_prefix_reusable[doc_id]
                    failed_promotions.add(doc_id)
        # Add failed promotions to uncached
        if failed_promotions:
            lookup_result.uncached_doc_ids.extend(failed_promotions)
            metrics.num_prefix_hits = lookup_result.prefix_length
            metrics.num_non_prefix_hits = len(lookup_result.non_prefix_reusable)
            metrics.num_misses = lookup_result.total_uncached
        metrics.kv_movement_ms = self.latency_model.estimate_kv_movement(host_blocks_to_move)
        self.cache_manager.stats.total_movement_overhead_ms += metrics.kv_movement_ms

        # Step 5: Selective recomputation (non-prefix chunks)
        tokens_recomputed = 0
        non_prefix_tokens_reused = 0
        if self.config.enable_selective_recompute:
            for doc_id, node in lookup_result.non_prefix_reusable.items():
                if node.kv_metadata:
                    chunk_tokens = node.kv_metadata.num_tokens
                    recompute_tokens = int(chunk_tokens * self.config.cache.recompute_budget)
                    tokens_recomputed += recompute_tokens
                    non_prefix_tokens_reused += chunk_tokens - recompute_tokens
        metrics.tokens_recomputed = tokens_recomputed
        metrics.selective_recompute_ms = self.latency_model.estimate_selective_recompute(tokens_recomputed)

        # Step 6: Prefill for uncached tokens
        uncached_tokens = 0
        for doc_id in lookup_result.uncached_doc_ids:
            idx = request.retrieved_doc_ids.index(doc_id)
            if idx < len(request.retrieved_doc_tokens):
                uncached_tokens += request.retrieved_doc_tokens[idx]

        # Prefix-matched tokens fully skip prefill (no double-counting)
        prefix_tokens_skipped = sum(
            n.kv_metadata.num_tokens for n in lookup_result.prefix_matched
            if n.kv_metadata
        )
        metrics.tokens_skipped = prefix_tokens_skipped + non_prefix_tokens_reused
        metrics.prefill_ms = self.latency_model.estimate_prefill(uncached_tokens + tokens_recomputed)

        # Compute FLOPs saved fraction and track token-level hit rate
        total_tokens = sum(request.retrieved_doc_tokens) if request.retrieved_doc_tokens else 0
        actual_cost = uncached_tokens + tokens_recomputed
        metrics.flops_saved_fraction = (
            1.0 - actual_cost / total_tokens if total_tokens > 0 else 0.0
        )
        self.cache_manager.stats.total_tokens_requested += total_tokens
        self.cache_manager.stats.total_tokens_reused += (prefix_tokens_skipped + non_prefix_tokens_reused)

        # Step 7: Decode
        metrics.decode_ms = self.latency_model.estimate_decode(output_tokens)

        # Step 8: Admit uncached documents to cache (collect real metadata)
        admitted_metadata: dict[str, KVCacheMetadata] = {}
        for doc_id in lookup_result.uncached_doc_ids:
            idx = request.retrieved_doc_ids.index(doc_id)
            num_tokens = request.retrieved_doc_tokens[idx] if idx < len(request.retrieved_doc_tokens) else 256
            rank = request.retrieval_ranks[idx] if idx < len(request.retrieval_ranks) else idx + 1
            geohash = request.retrieved_doc_geohashes[idx] if idx < len(request.retrieved_doc_geohashes) else None

            meta = self.cache_manager.admit(doc_id, num_tokens, rank, geohash)
            if meta is not None:
                admitted_metadata[doc_id] = meta

        # Step 9: Insert sequence into knowledge tree with CORRECT metadata
        # - Admitted docs get real metadata (with actual block IDs)
        # - Non-prefix reused docs: copy their existing cached metadata so the
        #   new path also has live KV (enables future prefix hits on this path)
        # - Already prefix-cached docs: tree.insert() preserves existing metadata
        # - Rejected docs get tier="none" placeholder (no blocks wasted)
        # Non-prefix reused docs are NOT persisted under the new path.
        # Each non-prefix reuse is per-request: the original node's blocks stay
        # at their original tree location. Persisting would require either block
        # aliasing (dual ownership bug) or copy-on-write (2x memory cost).
        # This is noted as a known simplification in the report.
        metadata_list: list[KVCacheMetadata] = []
        for i, doc_id in enumerate(request.retrieved_doc_ids):
            if doc_id in admitted_metadata:
                metadata_list.append(admitted_metadata[doc_id])
            else:
                metadata_list.append(self._make_placeholder_meta(request, i))
        self.cache_manager.insert_sequence(request.retrieved_doc_ids, metadata_list)

        # Step 10: Advance time step
        self.cache_manager.step()

        # Finalize metrics
        metrics.finalize()
        self._all_metrics.append(metrics)

        return metrics

    def _make_placeholder_meta(self, request: RAGRequest, idx: int) -> KVCacheMetadata:
        """Create a tier='none' placeholder for docs not admitted to cache."""
        doc_id = request.retrieved_doc_ids[idx]
        num_tokens = request.retrieved_doc_tokens[idx] if idx < len(request.retrieved_doc_tokens) else 256
        num_blocks = (num_tokens + 15) // 16
        geohash = request.retrieved_doc_geohashes[idx] if idx < len(request.retrieved_doc_geohashes) else None
        return KVCacheMetadata(
            doc_id=doc_id, num_tokens=num_tokens, num_blocks=num_blocks,
            tier="none", created_at=0, last_accessed_at=0,
            access_count=0, geohash=geohash,
        )

    def get_aggregate_metrics(self) -> dict:
        """Compute aggregate metrics across all processed requests."""
        if not self._all_metrics:
            return {}

        n = len(self._all_metrics)
        ttfts = sorted([m.ttft_ms for m in self._all_metrics])
        totals = sorted([m.total_ms for m in self._all_metrics])

        def percentile(data: list[float], p: float) -> float:
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        return {
            "num_requests": n,
            "ttft_p50_ms": percentile(ttfts, 50),
            "ttft_p95_ms": percentile(ttfts, 95),
            "ttft_p99_ms": percentile(ttfts, 99),
            "ttft_mean_ms": sum(ttfts) / n,
            "e2e_p50_ms": percentile(totals, 50),
            "e2e_p95_ms": percentile(totals, 95),
            "e2e_p99_ms": percentile(totals, 99),
            "e2e_mean_ms": sum(totals) / n,
            "throughput_rps_simulated": n / (sum(totals) / 1000) if sum(totals) > 0 else 0,  # single-request simulation only
            "avg_flops_saved": sum(m.flops_saved_fraction for m in self._all_metrics) / n,
            "avg_tokens_recomputed": sum(m.tokens_recomputed for m in self._all_metrics) / n,
            "avg_prefetch_hits": sum(m.num_prefetch_hits for m in self._all_metrics) / n,
            "cache_stats": self.cache_manager.stats.as_dict(),
            "memory_profile": self.cache_manager.get_memory_profile(),
        }
