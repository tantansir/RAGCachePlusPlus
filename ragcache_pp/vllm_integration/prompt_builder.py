"""RAG prompt builder with document ordering optimization.

Key insight: vLLM's Automatic Prefix Caching (APC) reuses KV blocks when
the token-level prefix matches a previous request. By ordering documents
in a canonical sequence that maximizes prefix sharing, we convert document
overlap into prefix overlap — dramatically increasing APC hit rates.

The knowledge tree tracks which document sequences are currently cached.
The optimizer greedily extends the longest cached prefix.
"""

from __future__ import annotations

import random
from typing import Optional

from ragcache_pp.cache.knowledge_tree import KnowledgeTree


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based only on "
    "the provided documents. If the documents don't contain enough "
    "information, say so.\n\n"
)

DOC_TEMPLATE = "[Document {rank}: {doc_id}]\n{content}\n\n"

QUERY_TEMPLATE = "Question: {query}\nAnswer:"


def build_rag_prompt(
    query: str,
    doc_ids: list[str],
    doc_contents: dict[str, str],
    doc_order: str = "original",
    knowledge_tree: Optional[KnowledgeTree] = None,
) -> tuple[str, list[str]]:
    """Build a RAG prompt with the given documents.

    Args:
        query: User query text.
        doc_ids: Retrieved document IDs (in retrieval rank order).
        doc_contents: Mapping from doc_id to document text.
        doc_order: "original" (retrieval order), "random", or "optimized".
        knowledge_tree: Required when doc_order == "optimized".

    Returns:
        (prompt_text, ordered_doc_ids)
    """
    if doc_order == "random":
        ordered = list(doc_ids)
        random.shuffle(ordered)
    elif doc_order == "sorted":
        ordered = sorted(doc_ids)
    elif doc_order == "optimized" and knowledge_tree is not None:
        ordered = optimize_doc_order(doc_ids, knowledge_tree)
    else:
        ordered = list(doc_ids)

    parts = [SYSTEM_PROMPT]
    for rank, doc_id in enumerate(ordered, 1):
        content = doc_contents.get(doc_id, f"[Content for {doc_id}]")
        parts.append(DOC_TEMPLATE.format(rank=rank, doc_id=doc_id, content=content))
    parts.append(QUERY_TEMPLATE.format(query=query))

    return "".join(parts), ordered


def optimize_doc_order(
    doc_ids: list[str],
    knowledge_tree: KnowledgeTree,
) -> list[str]:
    """Greedily order documents to maximize prefix sharing with cached sequences.

    At each position, choose the document that extends the longest cached
    prefix in the knowledge tree.  Greedy is optimal for a trie (the
    longest cached path is unique at each node).
    """
    ordered: list[str] = []
    remaining = set(doc_ids)
    current_node = knowledge_tree.root

    while remaining:
        found = False
        # Check children of current node for a cached continuation
        for doc_id in list(remaining):
            child = current_node.get_child(doc_id)
            if (
                child is not None
                and child.kv_metadata is not None
                and child.kv_metadata.tier != "none"
            ):
                ordered.append(doc_id)
                remaining.remove(doc_id)
                current_node = child
                found = True
                break

        if not found:
            # No cached continuation — preserve retrieval rank for the rest
            for doc_id in doc_ids:
                if doc_id in remaining:
                    ordered.append(doc_id)
            break

    return ordered
