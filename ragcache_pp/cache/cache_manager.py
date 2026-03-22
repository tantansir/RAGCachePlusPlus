"""Multi-tier KV cache manager for RAGCache++.

Manages KV cache blocks across GPU and host memory tiers, coordinating
with the knowledge tree for lookup, the PGDSF policy for eviction, and
the spatial index for metadata-aware admission and prefetch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ragcache_pp.cache.knowledge_tree import KnowledgeTree, KnowledgeTreeNode, KVCacheMetadata
from ragcache_pp.cache.pgdsf_policy import PGDSFPolicy
from ragcache_pp.cache.spatial_index import SpatialIndex
from ragcache_pp.config import CacheConfig


@dataclass
class CacheStats:
    """Accumulated cache statistics for profiling."""

    total_lookups: int = 0
    prefix_hits: int = 0
    non_prefix_hits: int = 0
    misses: int = 0
    gpu_hits: int = 0
    host_hits: int = 0
    evictions_gpu: int = 0
    evictions_host: int = 0
    admissions: int = 0
    admission_rejections: int = 0
    prefetch_issued: int = 0
    prefetch_used: int = 0  # prefetched KV that was actually accessed
    total_tokens_requested: int = 0
    total_tokens_reused: int = 0
    total_blocks_moved_gpu_to_host: int = 0
    total_blocks_moved_host_to_gpu: int = 0
    total_prefill_savings_ms: float = 0.0
    total_movement_overhead_ms: float = 0.0
    total_policy_overhead_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return (self.prefix_hits + self.non_prefix_hits) / self.total_lookups

    @property
    def byte_hit_rate(self) -> float:
        """Token-level hit rate (tokens reused / tokens requested)."""
        if self.total_tokens_requested == 0:
            return 0.0
        return self.total_tokens_reused / self.total_tokens_requested

    @property
    def prefetch_hit_rate(self) -> float:
        if self.prefetch_issued == 0:
            return 0.0
        return self.prefetch_used / self.prefetch_issued

    def as_dict(self) -> dict:
        return {
            "total_lookups": self.total_lookups,
            "prefix_hits": self.prefix_hits,
            "non_prefix_hits": self.non_prefix_hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "byte_hit_rate": self.byte_hit_rate,
            "gpu_hits": self.gpu_hits,
            "host_hits": self.host_hits,
            "evictions_gpu": self.evictions_gpu,
            "evictions_host": self.evictions_host,
            "admissions": self.admissions,
            "admission_rejections": self.admission_rejections,
            "prefetch_issued": self.prefetch_issued,
            "prefetch_used": self.prefetch_used,
            "prefetch_hit_rate": self.prefetch_hit_rate,
            "total_prefill_savings_ms": self.total_prefill_savings_ms,
            "total_movement_overhead_ms": self.total_movement_overhead_ms,
            "total_policy_overhead_ms": self.total_policy_overhead_ms,
        }


class BlockAllocator:
    """Simple block allocator simulating vLLM's paged memory.

    Tracks allocated and free blocks for a single tier (GPU or host).
    """

    def __init__(self, capacity: int, tier_name: str = "gpu"):
        self.capacity = capacity
        self.tier_name = tier_name
        self._free_blocks: list[int] = list(range(capacity))
        self._allocated: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free_blocks)

    @property
    def num_allocated(self) -> int:
        return len(self._allocated)

    @property
    def utilization(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.num_allocated / self.capacity

    def allocate(self, num_blocks: int) -> Optional[list[int]]:
        """Allocate num_blocks contiguous-free blocks. Returns block IDs or None."""
        if num_blocks > self.num_free:
            return None
        blocks = self._free_blocks[:num_blocks]
        self._free_blocks = self._free_blocks[num_blocks:]
        self._allocated.update(blocks)
        return blocks

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            if bid in self._allocated:
                self._allocated.discard(bid)
                self._free_blocks.append(bid)


@dataclass
class CacheLookupResult:
    """Result of a cache lookup for a document sequence."""

    # Prefix-matched documents (KV fully reusable)
    prefix_matched: list[KnowledgeTreeNode] = field(default_factory=list)
    prefix_length: int = 0

    # Non-prefix reusable documents (need selective recomputation)
    non_prefix_reusable: dict[str, KnowledgeTreeNode] = field(default_factory=dict)

    # Documents that need full computation
    uncached_doc_ids: list[str] = field(default_factory=list)

    # Prefetched documents that were used
    prefetch_hits: list[str] = field(default_factory=list)

    @property
    def total_cached(self) -> int:
        return self.prefix_length + len(self.non_prefix_reusable)

    @property
    def total_uncached(self) -> int:
        return len(self.uncached_doc_ids)


class CacheManager:
    """Multi-tier KV cache manager for RAGCache++.

    Coordinates knowledge tree, PGDSF policy, spatial index, and block
    allocators to manage KV cache across GPU and host memory.
    """

    def __init__(
        self,
        config: CacheConfig,
        spatial_index: Optional[SpatialIndex] = None,
        enable_spatial: bool = True,
        enable_non_prefix_reuse: bool = True,
    ):
        self.config = config
        self.enable_spatial = enable_spatial and spatial_index is not None
        self.enable_non_prefix_reuse = enable_non_prefix_reuse

        # Core components
        self.tree = KnowledgeTree()
        self.spatial_index = spatial_index

        self.gpu_policy = PGDSFPolicy(
            spatial_index=spatial_index if enable_spatial else None,
            spatial_lambda=config.spatial_lambda,
            age_decay=config.pgdsf_age_decay,
        )
        self.host_policy = PGDSFPolicy(
            spatial_index=spatial_index if enable_spatial else None,
            spatial_lambda=config.spatial_lambda,
            age_decay=config.pgdsf_age_decay,
        )

        # Block allocators
        self.gpu_allocator = BlockAllocator(config.gpu_cache_capacity, "gpu")
        self.host_allocator = BlockAllocator(config.host_cache_capacity, "host")

        # Tracking
        self.stats = CacheStats()
        self._prefetched_docs: set[str] = set()  # docs currently prefetched to GPU
        self._time_step: int = 0  # logical clock for PGDSF age consistency

    def lookup(self, doc_sequence: list[str]) -> CacheLookupResult:
        """Look up a document sequence in the cache.

        1. Try prefix match in the knowledge tree.
        2. If non-prefix reuse is enabled, find reusable chunks beyond prefix.
        3. Record stats.

        Args:
            doc_sequence: Ordered list of retrieved document IDs.

        Returns:
            CacheLookupResult with matched/unmatched documents.
        """
        t_start = time.perf_counter()
        result = CacheLookupResult()

        # Step 1: Prefix match
        matched_nodes, prefix_len = self.tree.prefix_match(doc_sequence)
        result.prefix_matched = [n for n in matched_nodes if n.kv_metadata and n.kv_metadata.tier != "none"]
        result.prefix_length = len(result.prefix_matched)

        matched_ids = set()
        for node in result.prefix_matched:
            matched_ids.add(node.doc_id)
            self.stats.prefix_hits += 1
            if node.kv_metadata and node.kv_metadata.tier == "gpu":
                self.stats.gpu_hits += 1
            elif node.kv_metadata and node.kv_metadata.tier == "host":
                self.stats.host_hits += 1
            # Credit prefetch hits for prefix matches too
            if node.doc_id in self._prefetched_docs:
                result.prefetch_hits.append(node.doc_id)
                self.stats.prefetch_used += 1
                self._prefetched_docs.discard(node.doc_id)

        # Step 2: Non-prefix reuse (CacheBlend-style)
        if self.enable_non_prefix_reuse:
            remaining = [d for d in doc_sequence[prefix_len:] if d not in matched_ids]
            reusable = self.tree.find_reusable_chunks(remaining)
            for doc_id, nodes in reusable.items():
                # Pick the best cached node (prefer GPU tier)
                best = None
                for n in nodes:
                    if n.kv_metadata and n.kv_metadata.tier == "gpu":
                        best = n
                        break
                    if n.kv_metadata and n.kv_metadata.tier == "host" and best is None:
                        best = n
                if best is not None:
                    result.non_prefix_reusable[doc_id] = best
                    matched_ids.add(doc_id)
                    self.stats.non_prefix_hits += 1
                    # Track per-tier hits for non-prefix too
                    if best.kv_metadata.tier == "gpu":
                        self.stats.gpu_hits += 1
                    elif best.kv_metadata.tier == "host":
                        self.stats.host_hits += 1
                    # Check if this was a prefetched doc
                    if doc_id in self._prefetched_docs:
                        result.prefetch_hits.append(doc_id)
                        self.stats.prefetch_used += 1
                        self._prefetched_docs.discard(doc_id)

        # Step 3: Uncached documents
        result.uncached_doc_ids = [d for d in doc_sequence if d not in matched_ids]
        self.stats.misses += len(result.uncached_doc_ids)
        self.stats.total_lookups += len(doc_sequence)

        # Step 4: Update policy priorities for all cache hits (Bug fix #6)
        for node in result.prefix_matched:
            if node.kv_metadata:
                node.kv_metadata.record_access(self._time_step)
                if node.kv_metadata.tier == "gpu":
                    self.gpu_policy.update(node)
                elif node.kv_metadata.tier == "host":
                    self.host_policy.update(node)
        for doc_id, node in result.non_prefix_reusable.items():
            if node.kv_metadata:
                node.kv_metadata.record_access(self._time_step)
                if node.kv_metadata.tier == "gpu":
                    self.gpu_policy.update(node)
                elif node.kv_metadata.tier == "host":
                    self.host_policy.update(node)

        t_elapsed = (time.perf_counter() - t_start) * 1000
        self.stats.total_policy_overhead_ms += t_elapsed

        return result

    def admit(
        self,
        doc_id: str,
        num_tokens: int,
        retrieval_rank: int,
        geohash: Optional[str] = None,
    ) -> Optional[KVCacheMetadata]:
        """Attempt to admit a document's KV cache into GPU (or host as fallback).

        Uses spatial-aware admission if enabled.

        Returns:
            KVCacheMetadata if admitted, None if rejected.
        """
        t_start = time.perf_counter()

        # Admission check
        if self.enable_spatial and self.spatial_index is not None:
            should_admit = self.spatial_index.check_admission(
                doc_id, retrieval_rank, self.config.admission_tau, self.config.admission_k_hot
            )
        else:
            # Without spatial: admit top-k_hot
            should_admit = retrieval_rank <= self.config.admission_k_hot * 2

        if not should_admit:
            self.stats.admission_rejections += 1
            t_elapsed = (time.perf_counter() - t_start) * 1000
            self.stats.total_policy_overhead_ms += t_elapsed
            return None

        # Calculate blocks needed
        block_size = 16  # tokens per block (vLLM default)
        num_blocks = (num_tokens + block_size - 1) // block_size

        # Try GPU first
        tier = "gpu"
        blocks = self.gpu_allocator.allocate(num_blocks)
        if blocks is None:
            # GPU full — evict until space available
            blocks = self._evict_and_allocate(num_blocks, "gpu")

        if blocks is None:
            # GPU eviction insufficient — try host
            tier = "host"
            blocks = self.host_allocator.allocate(num_blocks)
            if blocks is None:
                blocks = self._evict_and_allocate(num_blocks, "host")

        if blocks is None:
            # Both tiers full, cannot admit
            self.stats.admission_rejections += 1
            t_elapsed = (time.perf_counter() - t_start) * 1000
            self.stats.total_policy_overhead_ms += t_elapsed
            return None

        # Create metadata (use logical time for PGDSF age consistency)
        meta = KVCacheMetadata(
            doc_id=doc_id,
            num_tokens=num_tokens,
            num_blocks=num_blocks,
            block_ids=blocks,
            tier=tier,
            created_at=float(self._time_step),
            last_accessed_at=float(self._time_step),
            access_count=1,
            geohash=geohash,
        )

        self.stats.admissions += 1

        # Update spatial index footprint
        if self.spatial_index is not None and geohash:
            self.spatial_index.update_footprint(doc_id, num_blocks, delta=+1)

        t_elapsed = (time.perf_counter() - t_start) * 1000
        self.stats.total_policy_overhead_ms += t_elapsed

        return meta

    def insert_sequence(
        self,
        doc_sequence: list[str],
        metadata_list: list[KVCacheMetadata],
    ) -> list[KnowledgeTreeNode]:
        """Insert a full document sequence with their KV metadata into the tree.

        Only registers genuinely new or re-admitted nodes in the eviction policy.
        Existing cached nodes are already tracked by the policy (updated in lookup).
        """
        nodes, is_new = self.tree.insert(doc_sequence, metadata_list)
        for node, new in zip(nodes, is_new):
            if new and node.kv_metadata and node.kv_metadata.tier != "none":
                if node.kv_metadata.tier == "gpu":
                    self.gpu_policy.insert(node)
                elif node.kv_metadata.tier == "host":
                    self.host_policy.insert(node)
        return nodes

    def trigger_prefetch(self, query_lat: float, query_lon: float) -> list[str]:
        """Trigger spatial prefetch for adjacent regions.

        Finds cached docs in neighboring geohash cells that are on the host tier
        and promotes them to GPU. In a real system this is async PCIe DMA;
        here we simulate the decision and movement.

        Returns:
            List of doc_ids actually prefetched (promoted to GPU).
        """
        if not self.enable_spatial or self.spatial_index is None:
            return []

        candidates = self.spatial_index.get_prefetch_candidates(
            query_lat, query_lon, self.config.prefetch_budget
        )

        prefetched: list[str] = []
        for doc_id in candidates:
            if doc_id in self._prefetched_docs:
                continue
            # Find cached nodes for this doc and try to promote host→GPU
            nodes = self.tree._all_nodes.get(doc_id, [])
            for node in nodes:
                if node.kv_metadata and node.kv_metadata.tier == "host":
                    if self.promote_to_gpu(node):
                        self._prefetched_docs.add(doc_id)
                        prefetched.append(doc_id)
                        self.stats.prefetch_issued += 1
                        break

        return prefetched

    def promote_to_gpu(self, node: KnowledgeTreeNode) -> bool:
        """Promote a host-tier node to GPU tier.

        Returns True if successful.
        """
        meta = node.kv_metadata
        if meta is None or meta.tier != "host":
            return False

        gpu_blocks = self.gpu_allocator.allocate(meta.num_blocks)
        if gpu_blocks is None:
            gpu_blocks = self._evict_and_allocate(meta.num_blocks, "gpu")
        if gpu_blocks is None:
            return False

        # Free host blocks
        self.host_allocator.free(meta.block_ids)
        self.host_policy.remove(node)

        # Assign GPU blocks
        meta.block_ids = gpu_blocks
        meta.tier = "gpu"
        self.gpu_policy.insert(node)

        self.stats.total_blocks_moved_host_to_gpu += meta.num_blocks
        return True

    def demote_to_host(self, node: KnowledgeTreeNode) -> bool:
        """Demote a GPU-tier node to host tier.

        Returns True if successful.
        """
        meta = node.kv_metadata
        if meta is None or meta.tier != "gpu":
            return False

        host_blocks = self.host_allocator.allocate(meta.num_blocks)
        if host_blocks is None:
            host_blocks = self._evict_and_allocate(meta.num_blocks, "host")
        if host_blocks is None:
            return False

        # Free GPU blocks
        self.gpu_allocator.free(meta.block_ids)
        self.gpu_policy.remove(node)

        # Assign host blocks
        meta.block_ids = host_blocks
        meta.tier = "host"
        self.host_policy.insert(node)

        self.stats.total_blocks_moved_gpu_to_host += meta.num_blocks
        return True

    def step(self) -> None:
        """Advance one time step (call once per request)."""
        self._time_step += 1
        self.gpu_policy.tick()
        self.host_policy.tick()
        if self.spatial_index is not None:
            self.spatial_index.step()

    def _evict_and_allocate(self, num_blocks: int, tier: str) -> Optional[list[int]]:
        """Evict entries until enough space, then allocate.

        For GPU evictions: tries to demote victim to host tier first.
        Only hard-evicts (tier="none") when host is also full.
        """
        allocator = self.gpu_allocator if tier == "gpu" else self.host_allocator
        policy = self.gpu_policy if tier == "gpu" else self.host_policy
        max_attempts = 50  # prevent infinite loop

        for _ in range(max_attempts):
            if allocator.num_free >= num_blocks:
                return allocator.allocate(num_blocks)

            victim = policy.evict()
            if victim is None:
                break

            meta = victim.kv_metadata
            if meta is None:
                continue

            if tier == "gpu":
                # Try demoting to host first (victim already removed from gpu_policy)
                host_blocks = self.host_allocator.allocate(meta.num_blocks)
                if host_blocks is None:
                    # Host full — try evicting from host to make room for demotion
                    host_blocks = self._evict_host_only(meta.num_blocks)
                if host_blocks is not None:
                    self.gpu_allocator.free(meta.block_ids)
                    meta.block_ids = host_blocks
                    meta.tier = "host"
                    self.host_policy.insert(victim)
                    self.stats.total_blocks_moved_gpu_to_host += meta.num_blocks
                    self.stats.evictions_gpu += 1
                    continue

            # Hard evict — free blocks, mark as uncached
            allocator.free(meta.block_ids)
            meta.block_ids = []
            meta.tier = "none"

            if tier == "gpu":
                self.stats.evictions_gpu += 1
            else:
                self.stats.evictions_host += 1

            # Clear prefetched state for evicted doc
            self._prefetched_docs.discard(meta.doc_id)

            # Update spatial index
            if self.spatial_index is not None and meta.geohash:
                self.spatial_index.update_footprint(meta.doc_id, meta.num_blocks, delta=-1)

        if allocator.num_free >= num_blocks:
            return allocator.allocate(num_blocks)
        return None

    def _evict_host_only(self, num_blocks: int) -> Optional[list[int]]:
        """Evict from host tier only (used during GPU→host demotion)."""
        max_attempts = 20
        for _ in range(max_attempts):
            if self.host_allocator.num_free >= num_blocks:
                return self.host_allocator.allocate(num_blocks)
            victim = self.host_policy.evict()
            if victim is None:
                break
            meta = victim.kv_metadata
            if meta is None:
                continue
            self.host_allocator.free(meta.block_ids)
            meta.block_ids = []
            meta.tier = "none"
            self.stats.evictions_host += 1
            self._prefetched_docs.discard(meta.doc_id)
            if self.spatial_index is not None and meta.geohash:
                self.spatial_index.update_footprint(meta.doc_id, meta.num_blocks, delta=-1)
        if self.host_allocator.num_free >= num_blocks:
            return self.host_allocator.allocate(num_blocks)
        return None

    def allocate_path_copy(
        self, doc_id: str, num_tokens: int, num_blocks: int, geohash: Optional[str] = None
    ) -> Optional[KVCacheMetadata]:
        """Allocate fresh blocks for a path-specific copy of cached KV.

        Used when non-prefix reused KV is persisted under a new tree path.
        Unlike aliasing, this allocates distinct blocks so each node owns its memory.
        In a real system, KV data would be copied into the new blocks.

        Returns metadata with fresh block IDs, or None if allocation failed.
        """
        blocks = self.gpu_allocator.allocate(num_blocks)
        if blocks is None:
            blocks = self._evict_and_allocate(num_blocks, "gpu")
        if blocks is None:
            # Try host as fallback
            blocks = self.host_allocator.allocate(num_blocks)
            if blocks is None:
                blocks = self._evict_and_allocate(num_blocks, "host")
            if blocks is None:
                return None
            tier = "host"
        else:
            tier = "gpu"

        meta = KVCacheMetadata(
            doc_id=doc_id,
            num_tokens=num_tokens,
            num_blocks=num_blocks,
            block_ids=blocks,
            tier=tier,
            created_at=float(self._time_step),
            last_accessed_at=float(self._time_step),
            access_count=1,
            geohash=geohash,
        )

        if self.spatial_index is not None and geohash:
            self.spatial_index.update_footprint(doc_id, num_blocks, delta=+1)

        return meta

    def get_memory_profile(self) -> dict:
        """Return current memory utilization profile."""
        return {
            "gpu_capacity_blocks": self.gpu_allocator.capacity,
            "gpu_allocated_blocks": self.gpu_allocator.num_allocated,
            "gpu_free_blocks": self.gpu_allocator.num_free,
            "gpu_utilization": self.gpu_allocator.utilization,
            "host_capacity_blocks": self.host_allocator.capacity,
            "host_allocated_blocks": self.host_allocator.num_allocated,
            "host_free_blocks": self.host_allocator.num_free,
            "host_utilization": self.host_allocator.utilization,
            "tree_node_count": self.tree.node_count,
            "prefetched_pending": len(self._prefetched_docs),
        }
