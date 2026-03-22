"""PGDSF replacement policy with spatial-aware extensions.

Prefix-aware Greedy-Dual-Size-Frequency (PGDSF) replacement policy from
RAGCache, extended with spatial locality bonus for RAGCache++.

Priority(d) = PGDSF_base(d) + lambda_s * U(geohash(d))

Where PGDSF_base = (frequency * log(size) * age_penalty) / recomputation_cost
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Optional

from ragcache_pp.cache.knowledge_tree import KnowledgeTreeNode, KVCacheMetadata
from ragcache_pp.cache.spatial_index import SpatialIndex


@dataclass
class PriorityEntry:
    """An entry in the eviction priority queue."""

    priority: float
    node: KnowledgeTreeNode
    timestamp: float  # for tie-breaking (older = lower priority)

    def __lt__(self, other: PriorityEntry) -> bool:
        # Min-heap: lower priority = evict first
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp  # older first


class PGDSFPolicy:
    """PGDSF eviction policy with optional spatial-aware extension.

    Maintains a min-heap of (priority, node) pairs. When eviction is needed,
    the lowest-priority entry is removed.

    Base PGDSF priority:
        P(d) = (freq(d) * log(1 + size(d)) * age_penalty(d)) / recompute_cost(d)

    Spatial extension:
        P(d) = P_base(d) + lambda_s * U(geohash(d))

    Where U(geohash(d)) is the region utility from the spatial index.
    """

    def __init__(
        self,
        spatial_index: Optional[SpatialIndex] = None,
        spatial_lambda: float = 0.3,
        age_decay: float = 0.95,
        size_weight: float = 1.0,
    ):
        self.spatial_index = spatial_index
        self.spatial_lambda = spatial_lambda
        self.age_decay = age_decay
        self.size_weight = size_weight

        self._heap: list[PriorityEntry] = []
        self._entries: dict[int, PriorityEntry] = {}  # id(node) -> entry
        self._clock = 0.0  # logical clock for age computation

    def compute_priority(self, node: KnowledgeTreeNode) -> float:
        """Compute the eviction priority for a node.

        Higher priority = less likely to be evicted.
        """
        meta = node.kv_metadata
        if meta is None:
            return 0.0

        freq = max(1, meta.access_count)
        size = max(1, meta.num_tokens)
        # Age penalty: exponential decay based on time since last access
        age = self._clock - meta.last_accessed_at
        age_penalty = self.age_decay ** max(0, age)
        # Recomputation cost: proportional to chunk size (prefill cost)
        recompute_cost = max(1.0, float(size))

        base_priority = (freq * math.log(1 + size) * age_penalty) / recompute_cost

        # Spatial bonus
        spatial_bonus = 0.0
        if self.spatial_index is not None and meta.geohash:
            spatial_bonus = self.spatial_lambda * self.spatial_index.get_region_utility(meta.geohash)

        return base_priority + spatial_bonus

    def insert(self, node: KnowledgeTreeNode) -> None:
        """Insert a node into the priority queue."""
        # Invalidate old entry if exists (prevent stale heap duplicates)
        old_entry = self._entries.pop(id(node), None)
        if old_entry is not None:
            old_entry.node = None  # type: ignore[assignment]  # mark stale

        priority = self.compute_priority(node)
        entry = PriorityEntry(priority=priority, node=node, timestamp=self._clock)
        self._entries[id(node)] = entry
        heapq.heappush(self._heap, entry)

    def update(self, node: KnowledgeTreeNode) -> None:
        """Update priority for a node (e.g., after access)."""
        # Mark old entry as stale (lazy deletion)
        old_entry = self._entries.pop(id(node), None)
        if old_entry is not None:
            old_entry.node = None  # type: ignore[assignment]  # mark stale

        # Insert with updated priority
        self.insert(node)

    def evict(self) -> Optional[KnowledgeTreeNode]:
        """Remove and return the lowest-priority node.

        Returns None if the queue is empty.
        """
        while self._heap:
            entry = heapq.heappop(self._heap)
            # Skip stale entries (node set to None in update())
            if entry.node is None:
                continue
            node = entry.node
            self._entries.pop(id(node), None)
            return node
        return None

    def remove(self, node: KnowledgeTreeNode) -> None:
        """Remove a specific node from the queue (lazy deletion)."""
        entry = self._entries.pop(id(node), None)
        if entry is not None:
            entry.node = None  # type: ignore[assignment]  # mark stale

    def tick(self) -> None:
        """Advance logical clock (call once per request)."""
        self._clock += 1.0

    def peek_lowest(self) -> Optional[tuple[float, KnowledgeTreeNode]]:
        """Peek at the lowest-priority entry without removing it."""
        while self._heap:
            if self._heap[0].node is not None:
                return self._heap[0].priority, self._heap[0].node
            heapq.heappop(self._heap)  # remove stale
        return None

    @property
    def size(self) -> int:
        """Number of active entries."""
        return len(self._entries)
