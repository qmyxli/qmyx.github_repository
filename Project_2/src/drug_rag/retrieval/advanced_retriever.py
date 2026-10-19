from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from langchain_core.documents import Document

from drug_rag.retrieval.intent_router import build_metadata_filter
from drug_rag.retrieval.metadata_retriever import (
    RetrievalItem,
    _similarity_search,
    document_to_retrieval_item,
)
from drug_rag.retrieval.query_rewriter import QueryRewrite, build_query_rewrite
from drug_rag.retrieval.retrieval_policy import AUTO_MODE, RetrievalDecision, build_retrieval_decision
from drug_rag.retrieval.reranker import rerank_documents

# 前端/接口允许使用的检索模式。
# auto：生产主路径，由系统根据问题意图自动选择检索策略。
# vector：只做普通向量检索，主要用于实验对比。
# metadata：只做带 metadata filter 的向量检索，主要用于验证章节/药品过滤效果。
# summary：只查摘要索引，主要用于观察摘要索引本身的召回效果。
# hybrid：手动混合 metadata + summary + summary_guided + vector，主要用于实验对比。
RETRIEVAL_MODES = ("auto", "vector", "metadata", "summary", "hybrid")

# none：不重排；keyword：规则重排；model：外部重排模型；auto：由业务策略决定。
RERANK_STRATEGIES = ("auto", "none", "keyword", "model")


@dataclass(slots=True)
class AdvancedRetrievalTrace:
    """高级检索的可观察 Trace。

    这个对象有两个用途：
    1. 给 QA 链路继续使用：docs 字段保存最终入选的 LangChain Document。
    2. 给前端/评测使用：to_dict() 会把路由、改写、过滤器、各阶段命中数、最终来源都暴露出来。

    注意：docs 里是原始 Document 对象，不适合直接 JSON 序列化，所以 repr=False，
    API 返回时主要使用 results 里的结构化字段。
    """

    question: str
    mode: str
    rerank: str
    rewrite: QueryRewrite
    metadata_filter: dict[str, Any]
    search_query: str
    candidate_k: int
    final_k: int
    stage_counts: dict[str, int]
    fallback_used: bool
    results: list[RetrievalItem]
    decision: RetrievalDecision | None = None
    docs: list[Any] = field(default_factory=list, repr=False)
    errors: list[str] = field(default_factory=list)

    def to_dict(self, include_preview: bool = True) -> dict[str, Any]:
        """转换为可返回给前端的字典。

        include_preview=False 时会移除 preview，适合评测日志或更小的接口响应。
        """
        payload = {
            "question": self.question,
            "mode": self.mode,
            "rerank": self.rerank,
            "search_query": self.search_query,
            "route": self.rewrite.route.to_dict(),
            "rewrite": self.rewrite.to_dict(),
            "metadata_filter": self.metadata_filter,
            "candidate_k": self.candidate_k,
            "final_k": self.final_k,
            "stage_counts": self.stage_counts,
            "fallback_used": self.fallback_used,
            "decision": self.decision.to_dict() if self.decision else None,
            "errors": self.errors,
            "results": [asdict(item) for item in self.results],
        }
        if not include_preview:
            for item in payload["results"]:
                item.pop("preview", None)
        return payload


def retrieve_advanced(
    vector_store: Any,
    question: str,
    summary_store: Any | None = None,
    mode: str = AUTO_MODE,
    rerank: str = "auto",
    candidate_k: int | None = None,
    final_k: int = 6,
    drug_name: str | None = None,
    generic_name: str | None = None,
    category: str | None = None,
    use_rewrite: bool = True,
    fallback_without_filter: bool = True,
    rerank_model_provider: str = "none",
    rerank_model_name: str = "gte-rerank-v2",
    rerank_model_endpoint: str | None = None,
    rerank_model_api_key_env: str = "DASHSCOPE_API_KEY",
    rerank_model_timeout: int = 30,
    rerank_model_strategies: list[str] | None = None,
    preserve_preferred_sections: bool = True,
) -> AdvancedRetrievalTrace:
    """执行可观察的高级检索流程。

    入口会先根据 mode 决定是否使用自动策略。auto 模式会产生 RetrievalDecision，
    再按策略执行 metadata、summary、summary_guided、vector 等阶段。最后统一
    去重、rerank，并把每个阶段数量和最终结果写入 Trace，供前端展示和评测。

    整体执行顺序：
    1. 校验 mode/rerank，避免非法参数进入检索链路。
    2. 如果是 auto，则调用 build_retrieval_decision 做“问题重写 + 意图路由 + 策略选择”。
    3. 如果是手动模式，则只做基础问题重写和 metadata filter 构造。
    4. 按策略执行一个或多个检索阶段，每个阶段都会给 Document 标记 retrieval_stage。
    5. 多路召回结果去重，避免同一个 chunk 被重复用于答案生成。
    6. 对候选文档 rerank，得到最终 Top K。
    7. 把最终结果和中间过程封装为 AdvancedRetrievalTrace。
    """
    # 1. 参数校验：前端理论上会限制选项，但后端仍然要兜底，避免未知模式导致链路分支异常。
    mode = _validate(mode, RETRIEVAL_MODES, "retrieval mode")
    rerank = _validate(rerank, RERANK_STRATEGIES, "rerank strategy")

    # 手动模式没有策略层帮忙设置 candidate_k，所以这里给一个保守默认值。
    # candidate_k 是“先召回多少候选”，final_k 是“最终给答案层多少证据”。
    if candidate_k is None and mode != AUTO_MODE:
        candidate_k = max(final_k, 10)

    decision: RetrievalDecision | None = None
    if mode == AUTO_MODE:
        # auto 是真实业务主路径：系统根据问题选择检索方式，而不是让用户手动切换。
        # build_retrieval_decision 内部会完成：
        # - route_question：识别意图和目标章节
        # - rewrite_question：把自然语言改写成更利于检索的查询
        # - build_metadata_filter：把章节/药品/分类约束变成 Chroma filter
        # - strategy：选择 metadata_first、safety_hybrid、summary_guided_hybrid 或 hybrid
        rewrite, decision = build_retrieval_decision(
            question=question,
            drug_name=drug_name,
            generic_name=generic_name,
            category=category,
            top_k=final_k,
            candidate_k=candidate_k,
            rerank=rerank,
            use_rewrite=use_rewrite,
        )
        # 策略层已经给出最终可执行参数，后续检索阶段只消费这些参数。
        metadata_filter = decision.metadata_filter
        search_query = decision.rewritten_query
        rerank = _resolve_auto_rerank_strategy(
            requested_rerank=rerank,
            decision=decision,
            model_provider=rerank_model_provider,
            model_strategies=rerank_model_strategies,
        )
        candidate_k = decision.candidate_k
        final_k = decision.final_k
    else:
        # 手动模式主要用于前期验证阶段。
        # 这里仍然保留问题重写和 metadata filter，保证 vector/metadata/summary/hybrid
        # 可以在同样的问题理解结果上做横向比较。
        rewrite = build_query_rewrite(question, drug_name=drug_name, generic_name=generic_name)
        metadata_filter = build_metadata_filter(
            rewrite.route,
            drug_name=drug_name,
            generic_name=generic_name,
            category=category,
        )
        search_query = rewrite.rewritten_question if use_rewrite else question
        rerank = "keyword" if rerank == "auto" else rerank

    # stage_docs 保存所有阶段召回的原始 Document，最后统一去重和重排。
    # stage_counts 用于 Trace 展示，例如 metadata 命中几个、summary_guided 命中几个。
    # errors 不直接中断主流程，而是记录在 Trace 中，方便观察某个辅助检索阶段是否失败。
    stage_docs: list[Any] = []
    stage_counts: dict[str, int] = {}
    errors: list[str] = []
    fallback_used = False

    if mode == AUTO_MODE:
        # 自动模式根据 decision.strategy 分发到对应检索组合。
        stage_docs, stage_counts, fallback_used = _retrieve_by_decision(
            decision=decision,
            vector_store=vector_store,
            summary_store=summary_store,
            query=search_query,
            metadata_filter=metadata_filter,
            candidate_k=candidate_k,
            final_k=final_k,
            errors=errors,
            fallback_without_filter=fallback_without_filter,
        )

    elif mode == "vector":
        # 纯向量检索：不使用章节/药品过滤。适合观察 embedding 语义召回的基础能力。
        docs = _search_stage(vector_store, search_query, candidate_k, {}, "vector", errors)
        stage_docs.extend(docs)
        stage_counts["vector"] = len(docs)

    elif mode == "metadata":
        # metadata 模式：只查满足 metadata_filter 的 chunk。
        # 例如问题命中“贮藏”，就只在 section=贮藏/有效期/包装 中检索。
        docs = _search_stage(vector_store, search_query, candidate_k, metadata_filter, "metadata", errors)
        if not docs and metadata_filter and fallback_without_filter:
            # 如果过滤太严格导致 0 命中，可以退回无过滤检索，避免直接无答案。
            # fallback 会在 Trace 中标记，便于课程中说明“过滤召回失败”的场景。
            fallback_used = True
            docs = _search_stage(vector_store, search_query, candidate_k, {}, "metadata_fallback", errors)
        stage_docs.extend(docs)
        stage_counts["metadata"] = len(docs)

    elif mode == "summary":
        # 摘要索引检索：检索的是章节摘要 collection，不是原文 chunk。
        # 通常不直接作为最终证据，更多用于“先定位章节”的实验观察。
        docs = _search_summary_stage(summary_store, search_query, candidate_k, metadata_filter, errors)
        stage_docs.extend(docs)
        stage_counts["summary"] = len(docs)

    elif mode == "hybrid":
        # 手动 hybrid：同时跑多条检索路径，便于对比多路召回覆盖面。
        # metadata_docs：带过滤的原文 chunk 检索。
        metadata_docs = _search_stage(vector_store, search_query, candidate_k, metadata_filter, "metadata", errors)
        if not metadata_docs and metadata_filter and fallback_without_filter:
            fallback_used = True
            metadata_docs = _search_stage(vector_store, search_query, candidate_k, {}, "metadata_fallback", errors)
        # summary_docs：从摘要索引里找可能相关章节。
        summary_docs = _search_summary_stage(summary_store, search_query, max(2, candidate_k // 2), metadata_filter, errors)
        # guided_docs：根据 summary_docs 命中的章节，再回原文 chunk collection 取证。
        guided_docs = _search_summary_guided_chunks(
            vector_store=vector_store,
            query=search_query,
            summary_docs=summary_docs,
            metadata_filter=metadata_filter,
            candidate_k=max(2, candidate_k // 2),
            errors=errors,
        )
        # vector_docs：无过滤向量兜底，防止 metadata 或 summary 对问题理解不完整。
        vector_docs = _search_stage(vector_store, search_query, max(2, candidate_k // 2), {}, "vector", errors)
        stage_docs.extend(metadata_docs)
        stage_docs.extend(guided_docs)
        stage_docs.extend(summary_docs)
        stage_docs.extend(vector_docs)
        stage_counts["metadata"] = len(metadata_docs)
        stage_counts["summary_guided"] = len(guided_docs)
        stage_counts["summary"] = len(summary_docs)
        stage_counts["vector"] = len(vector_docs)

    # 多路检索经常会召回同一个 chunk。这里先去重，再做最终重排。
    candidates = _dedupe(stage_docs)

    # keyword rerank 会根据问题词、章节偏好等给候选文档重新打分。
    # preferred_sections 来自意图路由，能让“用法用量”问题优先保留用法用量章节。
    selected_docs = rerank_documents(
        search_query,
        candidates,
        strategy=rerank,
        top_k=final_k,
        preferred_sections=rewrite.route.sections,
        model_provider=rerank_model_provider,
        model_name=rerank_model_name,
        model_endpoint=rerank_model_endpoint,
        model_api_key_env=rerank_model_api_key_env,
        model_timeout=rerank_model_timeout,
        preserve_preferred_sections=preserve_preferred_sections,
        errors=errors,
    )

    # 把最终 Document 转成前端可展示的结果项：
    # rank、score、retrieval_stage、chunk_id、section、source_file、page 等。
    results = [
        document_to_retrieval_item(
            rank=rank,
            doc=doc,
            score=(getattr(doc, "metadata", {}) or {}).get("vector_score"),
        )
        for rank, doc in enumerate(selected_docs, start=1)
    ]

    # Trace 是这个函数最重要的产物：既包含最终 docs，也包含检索过程。
    # QA 层用 trace.docs 生成回答；前端用 trace.to_dict() 展示检索路径。
    return AdvancedRetrievalTrace(
        question=question,
        mode=mode,
        rerank=rerank,
        rewrite=rewrite,
        metadata_filter=metadata_filter,
        search_query=search_query,
        candidate_k=candidate_k,
        final_k=final_k,
        stage_counts=stage_counts,
        fallback_used=fallback_used,
        results=results,
        decision=decision,
        docs=selected_docs,
        errors=errors,
    )


def _resolve_auto_rerank_strategy(
    requested_rerank: str,
    decision: RetrievalDecision,
    model_provider: str,
    model_strategies: list[str] | None,
) -> str:
    """根据业务检索策略决定 auto rerank 的实际执行方式。

    固定章节问题已经由 metadata filter 精确定位，keyword 足够稳定。
    安全/泛问/混合路径候选更多、语义差异更细，优先使用模型重排；如果模型不可用，
    rerank_documents 内部会退回 keyword。
    """
    requested = (requested_rerank or "auto").lower()
    if requested != "auto":
        return requested
    if (model_provider or "none").lower() in {"none", "off", "false"}:
        return decision.rerank
    enabled_strategies = set(model_strategies or ["safety_hybrid", "summary_guided_hybrid", "hybrid"])
    if decision.strategy in enabled_strategies:
        return "model"
    return decision.rerank


def _retrieve_by_decision(
    decision: RetrievalDecision | None,
    vector_store: Any,
    summary_store: Any | None,
    query: str,
    metadata_filter: dict[str, Any],
    candidate_k: int,
    final_k: int,
    errors: list[str],
    fallback_without_filter: bool,
) -> tuple[list[Any], dict[str, int], bool]:
    """按 RetrievalDecision 执行具体召回阶段，并记录每个阶段命中数量。

    这里是 auto 模式的核心分发函数。策略不是用户选择的，而是由 retrieval_policy
    根据问题意图产生：
    - metadata_first：固定章节问题，先用 metadata filter 精准召回。
    - safety_hybrid：安全/禁忌/特殊人群/购药问题，多路召回降低漏掉风险信息的概率。
    - summary_guided_hybrid：泛问或章节不明确，先用摘要定位章节，再回原文取证。
    - hybrid：已有药品/分类约束但意图不清晰，综合多路召回。
    """
    if decision is None:
        return [], {}, False

    strategy = decision.strategy
    # stage_docs/stage_counts/fallback_used 与 retrieve_advanced 主函数中的含义一致，
    # 只是在这里按自动策略填充。
    stage_docs: list[Any] = []
    stage_counts: dict[str, int] = {}
    fallback_used = False

    if strategy == "metadata_first":
        # 固定章节问题优先精确过滤；如果命中不足，再用无过滤向量检索兜底。
        metadata_docs = _metadata_get_stage(vector_store, candidate_k, metadata_filter, "metadata", errors)
        if not metadata_docs:
            metadata_docs = _search_stage(vector_store, query, candidate_k, metadata_filter, "metadata_vector", errors)
        if not metadata_docs and metadata_filter and fallback_without_filter:
            fallback_used = True
            metadata_docs = _search_stage(vector_store, query, candidate_k, {}, "metadata_fallback", errors)
        stage_docs.extend(metadata_docs)
        stage_counts["metadata"] = len(metadata_docs)

        # 如果精准过滤后证据数量不够，再做一次无过滤向量检索兜底。
        # 这样既保留固定章节问题的高精度，也避免章节识别错误时完全无答案。
        if len(_dedupe(stage_docs)) < final_k:
            vector_docs = _search_stage(vector_store, query, max(2, candidate_k // 2), {}, "vector_fallback", errors)
            stage_docs.extend(vector_docs)
            stage_counts["vector_fallback"] = len(vector_docs)
        return stage_docs, stage_counts, fallback_used

    if strategy == "safety_hybrid":
        # 高风险问题多章节分散，metadata/summary/vector 三路一起召回降低漏召回。
        metadata_docs = _search_stage(vector_store, query, candidate_k, metadata_filter, "metadata", errors)
        if not metadata_docs and metadata_filter and fallback_without_filter:
            fallback_used = True
            metadata_docs = _search_stage(vector_store, query, candidate_k, {}, "metadata_fallback", errors)

        # 安全类问题的信息可能分散在禁忌、注意事项、特殊人群、相互作用、不良反应等章节。
        # summary_docs 用于找到相关章节，guided_docs 再回到原文 chunk 取证。
        summary_docs = _search_summary_stage(summary_store, query, max(2, candidate_k // 2), metadata_filter, errors)
        guided_docs = _search_summary_guided_chunks(
            vector_store=vector_store,
            query=query,
            summary_docs=summary_docs,
            metadata_filter=metadata_filter,
            candidate_k=max(2, candidate_k // 2),
            errors=errors,
        )
        # 无过滤 vector 是安全问题的兜底召回，避免 metadata 或摘要索引漏掉原文中的风险句。
        vector_docs = _search_stage(vector_store, query, max(2, candidate_k // 2), {}, "vector", errors)
        stage_docs.extend(metadata_docs)
        stage_docs.extend(guided_docs)
        stage_docs.extend(summary_docs)
        stage_docs.extend(vector_docs)
        stage_counts["metadata"] = len(metadata_docs)
        stage_counts["summary_guided"] = len(guided_docs)
        stage_counts["summary"] = len(summary_docs)
        stage_counts["vector"] = len(vector_docs)
        return stage_docs, stage_counts, fallback_used

    if strategy == "summary_guided_hybrid":
        # 泛问先查摘要确定章节，再回原文 chunk 检索证据。
        summary_docs = _search_summary_stage(summary_store, query, candidate_k, metadata_filter, errors)

        # 例如用户问“这个药安全吗”，不一定命中明确章节。
        # 摘要索引可能先命中“禁忌/注意事项/不良反应”，然后 guided 检索用这些章节过滤原文。
        guided_docs = _search_summary_guided_chunks(
            vector_store=vector_store,
            query=query,
            summary_docs=summary_docs,
            metadata_filter=metadata_filter,
            candidate_k=candidate_k,
            errors=errors,
        )
        # 泛问仍保留无过滤向量检索，防止摘要索引覆盖不全。
        vector_docs = _search_stage(vector_store, query, candidate_k, {}, "vector", errors)
        stage_docs.extend(guided_docs)
        stage_docs.extend(summary_docs)
        stage_docs.extend(vector_docs)
        stage_counts["summary_guided"] = len(guided_docs)
        stage_counts["summary"] = len(summary_docs)
        stage_counts["vector"] = len(vector_docs)
        return stage_docs, stage_counts, fallback_used

    # 默认 hybrid：适合“有约束但意图不明”的情况。
    # 比如识别出了药品名，但用户只问“这个怎么样”，此时需要兼顾过滤、摘要定位和向量兜底。
    metadata_docs = _search_stage(vector_store, query, candidate_k, metadata_filter, "metadata", errors)
    summary_docs = _search_summary_stage(summary_store, query, max(2, candidate_k // 2), metadata_filter, errors)
    guided_docs = _search_summary_guided_chunks(
        vector_store=vector_store,
        query=query,
        summary_docs=summary_docs,
        metadata_filter=metadata_filter,
        candidate_k=max(2, candidate_k // 2),
        errors=errors,
    )
    vector_docs = _search_stage(vector_store, query, max(2, candidate_k // 2), {}, "vector", errors)
    stage_docs.extend(metadata_docs)
    stage_docs.extend(guided_docs)
    stage_docs.extend(summary_docs)
    stage_docs.extend(vector_docs)
    stage_counts["metadata"] = len(metadata_docs)
    stage_counts["summary_guided"] = len(guided_docs)
    stage_counts["summary"] = len(summary_docs)
    stage_counts["vector"] = len(vector_docs)
    return stage_docs, stage_counts, fallback_used


def _search_stage(
    store: Any,
    query: str,
    k: int,
    metadata_filter: dict[str, Any],
    stage: str,
    errors: list[str],
) -> list[Any]:
    """执行一次 Chroma similarity_search，并给返回文档标记 retrieval_stage。

    参数说明：
    - store：Chroma collection 包装对象，可以是原文 chunk store，也可以是摘要 store。
    - query：最终用于向量检索的查询，一般是重写后的问题。
    - k：本阶段召回数量。
    - metadata_filter：Chroma filter，例如 {"section": "用法用量"}。
    - stage：当前阶段名称，会写入 doc.metadata["retrieval_stage"]，供 Trace 和来源展示使用。
    - errors：异常不抛出到上层，而是记录在这里，保证其他检索阶段还能继续执行。
    """
    try:
        docs_with_scores = _similarity_search(store, query, k=k, metadata_filter=metadata_filter)
    except Exception as exc:
        errors.append(f"{stage}: {type(exc).__name__}: {exc}")
        return []
    return [_tag_doc(doc, score=score, stage=stage) for doc, score in docs_with_scores]


def _metadata_get_stage(
    store: Any,
    k: int,
    metadata_filter: dict[str, Any],
    stage: str,
    errors: list[str],
) -> list[Any]:
    """执行确定性的 metadata-only 查询。

    固定章节问题，例如“奥美拉唑肠溶胶囊一天吃几次”，本质上已经有非常强的结构化条件：
    section=用法用量 + drug_name/generic_name=标准药品名。此时不应该完全依赖
    similarity_search_with_score，因为它仍然要先做 query embedding 和向量相似度查询。

    这里直接调用 Chroma collection.get(where=...)，只按 metadata 取文档。
    取到后再交给后续 keyword rerank 排序，保证固定章节问题不会因为向量检索阶段异常或相似度
    行为差异而返回 0 条。
    """
    if not metadata_filter:
        return []
    collection = getattr(store, "_collection", None)
    if collection is None or not hasattr(collection, "get"):
        return []
    try:
        payload = collection.get(where=metadata_filter, limit=k, include=["documents", "metadatas"])
    except Exception as exc:
        errors.append(f"{stage}_get: {type(exc).__name__}: {exc}")
        return []

    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    ids = payload.get("ids") or []
    docs: list[Any] = []
    for idx, text in enumerate(documents):
        metadata = dict(metadatas[idx] or {}) if idx < len(metadatas) else {}
        if idx < len(ids) and ids[idx] and "chunk_id" not in metadata:
            metadata["chunk_id"] = ids[idx]
        docs.append(_tag_doc(Document(page_content=text or "", metadata=metadata), score=None, stage=stage))
    return docs


def _search_summary_stage(
    summary_store: Any | None,
    query: str,
    k: int,
    metadata_filter: dict[str, Any],
    errors: list[str],
) -> list[Any]:
    """检索章节摘要 collection；metadata filter 会裁剪为摘要索引支持的字段。

    摘要索引不是完整原文，而是按章节生成的短摘要。它适合解决：
    - 用户问题很泛，难以直接定位原文章节。
    - 原文 chunk 很长或表述分散，直接向量检索不稳定。

    注意：summary_store 可能没有配置，所以这里会把错误写入 Trace，而不是中断问答。
    """
    if summary_store is None:
        errors.append("summary: summary_store is not configured")
        return []
    return _search_stage(summary_store, query, k, _summary_metadata_filter(metadata_filter), "summary", errors)


def _search_summary_guided_chunks(
    vector_store: Any,
    query: str,
    summary_docs: list[Any],
    metadata_filter: dict[str, Any],
    candidate_k: int,
    errors: list[str],
) -> list[Any]:
    """根据摘要命中的 section 回到原文 chunk collection 取证。

    summary_guided 的思想是“两跳检索”：
    1. 第一跳查 drug_summaries，找到可能相关的章节，例如“贮藏”“禁忌”。
    2. 第二跳查 drug_chunks，并把第一跳命中的 section 作为 metadata filter。

    这样既利用摘要帮助定位章节，又保证最终证据仍来自说明书原文 chunk。
    """
    sections = _sections_from_docs(summary_docs)
    if not sections:
        return []
    if len(sections) == 1:
        section_filter: dict[str, Any] = {"section": sections[0]}
    else:
        section_filter = {"section": {"$in": sections}}
    guided_filter = _combine_stage_filters(section_filter, _identity_metadata_filter(metadata_filter))
    return _search_stage(vector_store, query, candidate_k, guided_filter, "summary_guided", errors)


def _tag_doc(doc: Any, score: float | None, stage: str) -> Any:
    """给召回到的 Document 补充检索过程元数据。

    LangChain/Chroma 返回的 Document metadata 来自入库时的 chunk metadata。
    这里额外加：
    - vector_score：当前检索阶段返回的相似度分数。
    - retrieval_stage：来自哪个检索阶段，前端 Trace 和来源展示会用到。
    """
    metadata = dict(getattr(doc, "metadata", {}) or {})
    metadata["vector_score"] = score
    metadata["retrieval_stage"] = stage
    try:
        doc.metadata = metadata
    except Exception:
        pass
    return doc


def _dedupe(docs: Iterable[Any]) -> list[Any]:
    """多路召回会产生重复结果，这里按 chunk_id 或摘要键去重并保留首次出现。"""
    seen: set[str] = set()
    unique: list[Any] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        # 原文 chunk 优先用 chunk_id 去重；摘要文档没有 chunk_id 时，
        # 用 doc_id + summary_type + section 作为摘要级别的唯一键。
        key = (
            metadata.get("chunk_id")
            or f"{metadata.get('doc_id')}:{metadata.get('summary_type')}:{metadata.get('section')}"
            or getattr(doc, "page_content", str(doc))[:120]
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _sections_from_docs(docs: list[Any]) -> list[str]:
    """从摘要检索结果中提取 section，并保持首次出现顺序。"""
    sections = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        section = metadata.get("section")
        if section:
            sections.append(str(section))
    return list(dict.fromkeys(sections))


def _summary_metadata_filter(metadata_filter: dict[str, Any]) -> dict[str, Any]:
    """把通用 metadata_filter 裁剪成摘要索引支持的字段。

    原文 chunk 和摘要 collection 的 metadata 字段不完全一样。
    如果把摘要索引不支持的字段传给 Chroma filter，可能导致查询失败或 0 命中。
    """
    return _filter_metadata_keys(
        metadata_filter,
        supported_keys={"doc_id", "section", "drug_name", "generic_name", "source_file", "summary_type"},
    )


def _identity_metadata_filter(metadata_filter: dict[str, Any]) -> dict[str, Any]:
    """保留药品身份相关过滤条件，供 summary_guided 回原文取证时使用。

    summary_guided 已经会用摘要命中的 section 做章节过滤；
    这里再叠加 doc_id/drug_name/generic_name/category 等身份过滤，避免跨药品取证。
    """
    return _filter_metadata_keys(
        metadata_filter,
        supported_keys={"doc_id", "drug_name", "generic_name", "category", "source_file"},
    )


def _filter_metadata_keys(metadata_filter: dict[str, Any], supported_keys: set[str]) -> dict[str, Any]:
    """递归裁剪 metadata filter，只保留目标 collection 支持的字段。

    支持 Chroma 常见的组合过滤结构：
    - {"$and": [filter1, filter2]}
    - {"$or": [filter1, filter2]}

    如果某个子过滤器被裁剪为空，就丢弃它；如果组合条件只剩一个子条件，则拍平成普通 filter。
    """
    if not metadata_filter:
        return {}

    filtered: dict[str, Any] = {}
    for key, value in metadata_filter.items():
        if key in {"$and", "$or"} and isinstance(value, list):
            children = [_filter_metadata_keys(item, supported_keys) for item in value if isinstance(item, dict)]
            children = [item for item in children if item]
            if not children:
                continue
            if len(children) == 1:
                filtered.update(children[0])
            else:
                filtered[key] = children
        elif key in supported_keys:
            filtered[key] = value
    return filtered


def _combine_stage_filters(*filters: dict[str, Any]) -> dict[str, Any]:
    """合并多个 Chroma metadata filter。

    空 filter 会被忽略；多个非空 filter 用 $and 组合。
    例如 summary_guided 中会合并：
    - {"section": {"$in": ["禁忌", "注意事项"]}}
    - {"generic_name": "坎地沙坦酯片"}
    """
    non_empty = [item for item in filters if item]
    if not non_empty:
        return {}
    if len(non_empty) == 1:
        return non_empty[0]
    return {"$and": non_empty}


def _validate(value: str, allowed: tuple[str, ...], label: str) -> str:
    """校验枚举型参数，并统一转小写。"""
    normalized = (value or "").lower()
    if normalized not in allowed:
        raise ValueError(f"Unsupported {label}: {value}. Expected one of: {', '.join(allowed)}")
    return normalized
