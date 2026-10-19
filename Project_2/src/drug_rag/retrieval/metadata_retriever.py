from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from drug_rag.retrieval.intent_router import build_metadata_filter
from drug_rag.retrieval.query_rewriter import QueryRewrite, build_query_rewrite


@dataclass(slots=True)
class RetrievalItem:
    rank: int
    score: float | None
    vector_score: float | None
    rerank_score: float | None
    retrieval_stage: str
    chunk_id: str
    section: str
    drug_name: str
    generic_name: str
    source_file: str
    page_start: int | str
    page_end: int | str
    preview: str


@dataclass(slots=True)
class RetrievalTrace:
    question: str
    rewrite: QueryRewrite
    metadata_filter: dict[str, Any]
    k: int
    fallback_used: bool
    search_query: str
    results: list[RetrievalItem]
    docs: list[Any] = field(default_factory=list, repr=False)
    error: str | None = None

    def to_dict(self, include_preview: bool = True) -> dict[str, Any]:
        data = {
            "question": self.question,
            "search_query": self.search_query,
            "route": self.rewrite.route.to_dict(),
            "rewrite": self.rewrite.to_dict(),
            "metadata_filter": self.metadata_filter,
            "k": self.k,
            "fallback_used": self.fallback_used,
            "error": self.error,
            "results": [asdict(item) for item in self.results],
        }
        if not include_preview:
            for item in data["results"]:
                item.pop("preview", None)
        return data


def retrieve_with_trace(
    vector_store: Any,
    question: str,
    k: int = 6,
    drug_name: str | None = None,
    generic_name: str | None = None,
    category: str | None = None,
    use_rewrite: bool = True,
    fallback_without_filter: bool = True,
) -> RetrievalTrace:
    rewrite = build_query_rewrite(question, drug_name=drug_name, generic_name=generic_name)
    metadata_filter = build_metadata_filter(
        rewrite.route,
        drug_name=drug_name,
        generic_name=generic_name,
        category=category,
    )
    search_query = rewrite.rewritten_question if use_rewrite else question
    fallback_used = False
    error = None

    try:
        docs_with_scores = _similarity_search(vector_store, search_query, k=k, metadata_filter=metadata_filter)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        docs_with_scores = []

    if not docs_with_scores and metadata_filter and fallback_without_filter:
        fallback_used = True
        try:
            docs_with_scores = _similarity_search(vector_store, search_query, k=k, metadata_filter={})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            docs_with_scores = []

    docs = [doc for doc, _ in docs_with_scores]
    results = [_to_item(rank, doc, score) for rank, (doc, score) in enumerate(docs_with_scores, start=1)]
    return RetrievalTrace(
        question=question,
        rewrite=rewrite,
        metadata_filter=metadata_filter,
        k=k,
        fallback_used=fallback_used,
        search_query=search_query,
        results=results,
        docs=docs,
        error=error,
    )


def similarity_search_with_route(vector_store: Any, question: str, k: int = 6):
    return retrieve_with_trace(vector_store, question, k=k).docs


def _similarity_search(vector_store: Any, query: str, k: int, metadata_filter: dict[str, Any]) -> list[tuple[Any, float | None]]:
    kwargs = {"k": k}
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    if hasattr(vector_store, "similarity_search_with_score"):
        return vector_store.similarity_search_with_score(query, **kwargs)
    docs = vector_store.similarity_search(query, **kwargs)
    return [(doc, None) for doc in docs]


def _to_item(rank: int, doc: Any, score: float | None) -> RetrievalItem:
    metadata = getattr(doc, "metadata", {}) or {}
    text = getattr(doc, "page_content", str(doc)).replace("\n", " ")
    preview = text[:240].rstrip() + ("..." if len(text) > 240 else "")
    return RetrievalItem(
        rank=rank,
        score=round(float(score), 6) if score is not None else None,
        vector_score=_round_optional(metadata.get("vector_score", score)),
        rerank_score=_round_optional(metadata.get("rerank_score")),
        retrieval_stage=str(metadata.get("retrieval_stage", metadata.get("summary_type", "chunk"))),
        chunk_id=str(metadata.get("chunk_id", "")),
        section=str(metadata.get("section", "")),
        drug_name=str(metadata.get("drug_name", "")),
        generic_name=str(metadata.get("generic_name", "")),
        source_file=str(metadata.get("source_file", "")),
        page_start=metadata.get("page_start", "?"),
        page_end=metadata.get("page_end", metadata.get("page_start", "?")),
        preview=preview,
    )


def document_to_retrieval_item(rank: int, doc: Any, score: float | None = None) -> RetrievalItem:
    return _to_item(rank, doc, score)


def _round_optional(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None
