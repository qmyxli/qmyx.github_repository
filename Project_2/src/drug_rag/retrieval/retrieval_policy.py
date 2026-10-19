from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from drug_rag.retrieval.intent_router import build_metadata_filter
from drug_rag.retrieval.query_rewriter import QueryRewrite, build_query_rewrite


AUTO_MODE = "auto"
AUTO_RERANK = "auto"

FIXED_SECTION_INTENTS = {"dosage", "indication", "storage"}
SAFETY_INTENTS = {"special_population", "contraindication", "interaction", "adverse_reaction", "purchase_guidance"}


@dataclass(slots=True)
class RetrievalDecision:
    intent: str
    strategy: str
    metadata_filter: dict[str, Any]
    rewritten_query: str
    candidate_k: int
    final_k: int
    use_metadata: bool
    use_summary: bool
    use_vector: bool
    rerank: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_retrieval_decision(
    question: str,
    drug_name: str | None = None,
    generic_name: str | None = None,
    category: str | None = None,
    top_k: int = 6,
    candidate_k: int | None = None,
    rerank: str = AUTO_RERANK,
    use_rewrite: bool = True,
) -> tuple[QueryRewrite, RetrievalDecision]:
    """根据问题意图选择自动检索策略。

    - 固定章节问题，如用法用量/适应症/贮藏，走 metadata_first。
    - 高风险或购药决策问题，走 safety_hybrid，避免漏掉禁忌和注意事项。
    - 有药品/分类约束但意图不明确，走 hybrid。
    - 泛问且无明确章节，走 summary_guided_hybrid，先用摘要定位章节。
    """
    rewrite = build_query_rewrite(question, drug_name=drug_name, generic_name=generic_name)
    metadata_filter = build_metadata_filter(
        rewrite.route,
        drug_name=drug_name,
        generic_name=generic_name,
        category=category,
    )
    intent = rewrite.route.intent
    final_k = max(1, top_k)
    chosen_rerank = "keyword" if rerank in {AUTO_RERANK, "", None} else rerank
    rewritten_query = rewrite.rewritten_question if use_rewrite else question

    if intent in FIXED_SECTION_INTENTS:
        strategy = "metadata_first"
        default_candidate_k = max(final_k + 2, 8)
        use_metadata = True
        use_summary = False
        use_vector = True
        reason = (
            f"问题命中 {intent} 意图，答案通常位于固定说明书章节；"
            "优先使用章节与药品元数据过滤，命中不足时再用向量检索兜底。"
        )
    elif intent in SAFETY_INTENTS:
        strategy = "safety_hybrid"
        default_candidate_k = max(final_k * 2, 12)
        use_metadata = True
        use_summary = True
        use_vector = True
        reason = (
            f"问题命中 {intent} 高风险或购药决策意图，单一路径容易漏掉禁忌、注意事项或特殊人群章节；"
            "采用 metadata + summary + vector 混合召回，并对意图章节加权重排。"
        )
    elif metadata_filter:
        strategy = "hybrid"
        default_candidate_k = max(final_k * 2, 10)
        use_metadata = True
        use_summary = True
        use_vector = True
        reason = "问题已有药品或分类约束，但意图不够明确；采用混合检索兼顾精准过滤和语义召回。"
    else:
        strategy = "summary_guided_hybrid"
        default_candidate_k = max(final_k * 2, 10)
        use_metadata = False
        use_summary = True
        use_vector = True
        reason = "问题未命中明确章节，先用摘要索引定位可能相关章节，再回到原文 chunk 取证并用向量检索兜底。"

    return rewrite, RetrievalDecision(
        intent=intent,
        strategy=strategy,
        metadata_filter=metadata_filter,
        rewritten_query=rewritten_query,
        candidate_k=candidate_k or default_candidate_k,
        final_k=final_k,
        use_metadata=use_metadata,
        use_summary=use_summary,
        use_vector=use_vector,
        rerank=chosen_rerank,
        reason=reason,
    )
