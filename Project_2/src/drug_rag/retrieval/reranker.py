from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


def rerank_documents(
    query: str,
    docs: list[Any],
    strategy: str = "none",
    top_k: int = 5,
    preferred_sections: list[str] | None = None,
    model_provider: str = "none",
    model_name: str = "gte-rerank-v2",
    model_endpoint: str | None = None,
    model_api_key_env: str = "DASHSCOPE_API_KEY",
    model_timeout: int = 30,
    preserve_preferred_sections: bool = True,
    errors: list[str] | None = None,
) -> list[Any]:
    strategy = (strategy or "none").lower()
    if strategy == "none":
        for rank, doc in enumerate(docs[:top_k], start=1):
            _set_rerank_metadata(doc, strategy=strategy, score=None, rank=rank)
        return docs[:top_k]

    if strategy == "model":
        try:
            selected = _model_rerank_documents(
                query=query,
                docs=docs,
                top_k=top_k,
                provider=model_provider,
                model=model_name,
                endpoint=model_endpoint,
                api_key_env=model_api_key_env,
                timeout=model_timeout,
            )
            if preserve_preferred_sections:
                selected = _preserve_preferred_section_docs(selected, docs, preferred_sections, top_k)
            for rank, doc in enumerate(selected, start=1):
                metadata = getattr(doc, "metadata", {}) or {}
                _set_rerank_metadata(
                    doc,
                    strategy="model",
                    score=_optional_float(metadata.get("model_rerank_score")),
                    rank=rank,
                )
            return selected[:top_k]
        except Exception as exc:
            if errors is not None:
                errors.append(f"rerank_model: {type(exc).__name__}: {exc}")
            # 模型重排是增强层，不能因为外部服务不可用导致检索链路失败。
            return rerank_documents(
                query,
                docs,
                strategy="keyword",
                top_k=top_k,
                preferred_sections=preferred_sections,
                errors=errors,
            )

    if strategy == "keyword":
        scored = [(keyword_score(query, doc, preferred_sections=preferred_sections), doc) for doc in docs]
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        for rank, (score, doc) in enumerate(scored[:top_k], start=1):
            _set_rerank_metadata(doc, strategy=strategy, score=score, rank=rank)
            selected.append(doc)
        return selected
    raise ValueError(f"Unsupported rerank strategy: {strategy}")


def keyword_rerank(question: str, docs: list[Any], top_k: int = 5) -> list[Any]:
    return rerank_documents(question, docs, strategy="keyword", top_k=top_k)


def keyword_score(question: str, doc: Any, preferred_sections: list[str] | None = None) -> float:
    terms = set(re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]{1,2}", question.lower()))
    text = getattr(doc, "page_content", str(doc)).lower()
    metadata = getattr(doc, "metadata", {}) or {}
    section = str(metadata.get("section", ""))
    overlap = sum(1 for term in terms if term and term in text)
    section_bonus = 2 if any(term in section for term in terms) else 0
    preferred_bonus = 5 if preferred_sections and section in preferred_sections else 0
    exact_bonus = 3 if question.strip() and question.strip() in text else 0
    return float(overlap + section_bonus + preferred_bonus + exact_bonus)


def _model_rerank_documents(
    query: str,
    docs: list[Any],
    top_k: int,
    provider: str,
    model: str,
    endpoint: str | None,
    api_key_env: str,
    timeout: int,
) -> list[Any]:
    if not docs:
        return []
    provider = (provider or "none").lower()
    if provider in {"none", "off", "false"}:
        raise RuntimeError("model rerank provider is disabled")
    if provider != "dashscope":
        raise ValueError(f"Unsupported rerank provider: {provider}")

    scores = _dashscope_rerank(
        query=query,
        documents=[_doc_text(doc) for doc in docs],
        model=model,
        endpoint=endpoint or "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        api_key_env=api_key_env,
        timeout=timeout,
    )
    scored_docs = []
    for index, score in scores:
        if 0 <= index < len(docs):
            doc = docs[index]
            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata["model_rerank_score"] = score
            try:
                doc.metadata = metadata
            except Exception:
                pass
            scored_docs.append((score, doc))
    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


def _dashscope_rerank(
    query: str,
    documents: list[str],
    model: str,
    endpoint: str,
    api_key_env: str,
    timeout: int,
) -> list[tuple[int, float]]:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing environment variable: {api_key_env}")
    payload = {
        "model": model,
        "input": {
            "query": query,
            "documents": documents,
        },
        "parameters": {
            # 返回全部候选的模型分数，最终 top_k 和安全章节保留由本地代码控制。
            "top_n": len(documents),
            "return_documents": False,
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Rerank request failed: HTTP {exc.code} {detail}") from exc

    data = json.loads(body)
    results = (data.get("output") or {}).get("results") or data.get("results") or []
    parsed: list[tuple[int, float]] = []
    for item in results:
        index = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if index is None or score is None:
            continue
        parsed.append((int(index), float(score)))
    if not parsed:
        raise RuntimeError(f"Rerank response has no results: {body[:300]}")
    return parsed


def _preserve_preferred_section_docs(
    selected: list[Any],
    candidates: list[Any],
    preferred_sections: list[str] | None,
    top_k: int,
) -> list[Any]:
    if not preferred_sections or not selected:
        return selected[:top_k]
    selected_keys = {_doc_key(doc) for doc in selected}
    output = list(selected[:top_k])
    for section in preferred_sections:
        if any((getattr(doc, "metadata", {}) or {}).get("section") == section for doc in output):
            continue
        replacement = next(
            (
                doc
                for doc in candidates
                if (getattr(doc, "metadata", {}) or {}).get("section") == section
                and _doc_key(doc) not in selected_keys
            ),
            None,
        )
        if replacement is None:
            continue
        if len(output) < top_k:
            output.append(replacement)
        else:
            output[-1] = replacement
        selected_keys.add(_doc_key(replacement))
    return output[:top_k]


def _doc_key(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("chunk_id") or f"{metadata.get('doc_id')}:{metadata.get('section')}" or id(doc))


def _doc_text(doc: Any) -> str:
    return getattr(doc, "page_content", str(doc)).replace("\n", " ").strip()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _set_rerank_metadata(doc: Any, strategy: str, score: float | None, rank: int) -> None:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    metadata["rerank_strategy"] = strategy
    metadata["rerank_score"] = score
    metadata["rerank_rank"] = rank
    try:
        doc.metadata = metadata
    except Exception:
        pass
