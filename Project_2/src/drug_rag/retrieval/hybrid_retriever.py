from __future__ import annotations

from typing import Any

from drug_rag.retrieval.advanced_retriever import retrieve_advanced


def hybrid_search(
    vector_store: Any,
    question: str,
    summary_store: Any | None = None,
    top_k: int = 6,
    rerank_top_k: int = 5,
) -> list[Any]:
    trace = retrieve_advanced(
        vector_store=vector_store,
        summary_store=summary_store,
        question=question,
        mode="hybrid" if summary_store is not None else "metadata",
        rerank="keyword",
        candidate_k=top_k,
        final_k=rerank_top_k,
    )
    return trace.docs


def _dedupe(docs: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        key = metadata.get("chunk_id") or getattr(doc, "page_content", str(doc))[:80]
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique
