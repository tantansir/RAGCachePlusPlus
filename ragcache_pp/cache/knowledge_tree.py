"""Knowledge Tree: document-sequence-indexed tree for KV cache organization.

RAGCache organizes cached KV states in a knowledge tree where each path from
root to leaf represents a specific sequence of retrieved documents. Nodes store
metadata pointing to KV cache pages in vLLM's paged memory, not the KV data
itself. Prefix matching traverses the tree in O(h) time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVCacheMetadata:
    """Metadata for a document's KV cache blocks."""

    doc_id: str
    num_tokens: int
    num_blocks: int  # number of vLLM paged-attention blocks
    block_ids: list[int] = field(default_factory=list)  # GPU or host block IDs
    tier: str = "none"  # "gpu", "host", or "none" (not cached)
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    access_count: int = 0
    geohash: Optional[str] = None  # spatial metadata

    @property
    def size(self) -> int:
        """Size in blocks."""
        return self.num_blocks

    def record_access(self, logical_time: float = 0.0) -> None:
        self.last_accessed_at = logical_time
        self.access_count += 1


class KnowledgeTreeNode:
    """A node in the knowledge tree.

    Each node represents one document in a document sequence. The path from
    root to this node defines the prefix sequence of documents.
    """

    __slots__ = ("doc_id", "kv_metadata", "children", "parent")

    def __init__(
        self,
        doc_id: str,
        kv_metadata: Optional[KVCacheMetadata] = None,
        parent: Optional[KnowledgeTreeNode] = None,
    ):
        self.doc_id = doc_id
        self.kv_metadata = kv_metadata
        self.children: dict[str, KnowledgeTreeNode] = {}
        self.parent = parent

    def get_child(self, doc_id: str) -> Optional[KnowledgeTreeNode]:
        return self.children.get(doc_id)

    def add_child(self, doc_id: str, kv_metadata: Optional[KVCacheMetadata] = None) -> KnowledgeTreeNode:
        if doc_id in self.children:
            return self.children[doc_id]
        child = KnowledgeTreeNode(doc_id=doc_id, kv_metadata=kv_metadata, parent=self)
        self.children[doc_id] = child
        return child

    def remove_child(self, doc_id: str) -> Optional[KnowledgeTreeNode]:
        return self.children.pop(doc_id, None)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def path_from_root(self) -> list[str]:
        """Return the document sequence from root to this node."""
        path: list[str] = []
        node: Optional[KnowledgeTreeNode] = self
        while node is not None and node.doc_id != "__ROOT__":
            path.append(node.doc_id)
            node = node.parent
        path.reverse()
        return path


class KnowledgeTree:
    """Document-sequence-indexed tree for KV cache lookup.

    The root node is a sentinel; each first-level child represents the first
    document in a request's retrieved set. Deeper nodes represent subsequent
    documents.

    Prefix matching: given a document sequence [D1, D2, D3], traverse the tree
    to find the longest matching prefix. Cached KV for matched documents can
    be reused; unmatched suffix requires prefill.
    """

    def __init__(self) -> None:
        self.root = KnowledgeTreeNode(doc_id="__ROOT__")
        self._node_count = 0
        self._all_nodes: dict[str, list[KnowledgeTreeNode]] = {}  # doc_id -> nodes

    @property
    def node_count(self) -> int:
        return self._node_count

    def insert(
        self, doc_sequence: list[str], kv_metadata_list: list[KVCacheMetadata]
    ) -> tuple[list[KnowledgeTreeNode], list[bool]]:
        """Insert a document sequence into the tree.

        Returns:
            nodes: list of nodes (one per document)
            is_new: list of booleans — True if the node was newly created or
                    re-admitted (evicted node replaced with live metadata)
        """
        assert len(doc_sequence) == len(kv_metadata_list)
        nodes: list[KnowledgeTreeNode] = []
        is_new: list[bool] = []
        current = self.root

        for doc_id, kv_meta in zip(doc_sequence, kv_metadata_list):
            child = current.get_child(doc_id)
            if child is None:
                child = current.add_child(doc_id, kv_meta)
                self._node_count += 1
                self._all_nodes.setdefault(doc_id, []).append(child)
                is_new.append(True)
            else:
                # Re-admit: replace evicted metadata with live metadata
                if (
                    (child.kv_metadata is None or child.kv_metadata.tier == "none")
                    and kv_meta.tier != "none"
                ):
                    child.kv_metadata = kv_meta
                    is_new.append(True)
                else:
                    is_new.append(False)
            nodes.append(child)
            current = child

        return nodes, is_new

    def prefix_match(self, doc_sequence: list[str]) -> tuple[list[KnowledgeTreeNode], int]:
        """Find the longest matching prefix in the tree.

        Stops at evicted nodes (tier="none") because prefix KV reuse
        requires all prior KV states in the sequence to be resident.

        Returns:
            matched_nodes: list of matched KnowledgeTreeNodes (all resident)
            match_length: number of documents matched (prefix length)
        """
        matched: list[KnowledgeTreeNode] = []
        current = self.root

        for doc_id in doc_sequence:
            child = current.get_child(doc_id)
            if child is None:
                break
            # Stop at evicted nodes — prefix dependency requires all prior KV
            if child.kv_metadata is None or child.kv_metadata.tier == "none":
                break
            # Don't call record_access here; CacheManager handles it with logical time
            matched.append(child)
            current = child

        return matched, len(matched)

    def find_reusable_chunks(self, doc_sequence: list[str]) -> dict[str, list[KnowledgeTreeNode]]:
        """Find all cached chunks that appear in the sequence, regardless of position.

        This supports non-prefix reuse (CacheBlend-style). Returns a mapping
        from doc_id to nodes where that document's KV is cached.

        For non-prefix chunks, selective recomputation is needed to restore
        cross-attention accuracy.
        """
        reusable: dict[str, list[KnowledgeTreeNode]] = {}
        for doc_id in doc_sequence:
            if doc_id in self._all_nodes:
                cached_nodes = [
                    n for n in self._all_nodes[doc_id]
                    if n.kv_metadata is not None and n.kv_metadata.tier != "none"
                ]
                if cached_nodes:
                    reusable[doc_id] = cached_nodes
        return reusable

    def remove_node(self, node: KnowledgeTreeNode) -> None:
        """Remove a leaf node from the tree.

        Only leaf nodes can be removed. For internal nodes, remove children first.
        """
        if not node.is_leaf:
            raise ValueError("Cannot remove non-leaf node; remove children first.")
        if node.parent is not None:
            node.parent.remove_child(node.doc_id)
        # Remove from _all_nodes index
        if node.doc_id in self._all_nodes:
            self._all_nodes[node.doc_id] = [
                n for n in self._all_nodes[node.doc_id] if n is not node
            ]
            if not self._all_nodes[node.doc_id]:
                del self._all_nodes[node.doc_id]
        self._node_count -= 1

    def get_all_leaf_nodes(self) -> list[KnowledgeTreeNode]:
        """Return all leaf nodes (candidates for eviction)."""
        leaves: list[KnowledgeTreeNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf and node is not self.root:
                leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def get_all_cached_nodes(self) -> list[KnowledgeTreeNode]:
        """Return all nodes that have cached KV (on GPU or host)."""
        cached: list[KnowledgeTreeNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.kv_metadata is not None and node.kv_metadata.tier != "none":
                cached.append(node)
            stack.extend(node.children.values())
        return cached
