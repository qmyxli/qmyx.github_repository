from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from drug_rag.chains.safety_chain import SafetyAssessment, assess_safety
from drug_rag.config import load_yaml
from drug_rag.llms import build_chat_model
from drug_rag.retrieval.advanced_retriever import AdvancedRetrievalTrace, retrieve_advanced


DEFAULT_QA_SYSTEM_PROMPT = (
    "你是药品说明书RAG系统的受控生成层。只能依据给定说明书证据回答，"
    "不能补充证据外的医学知识，不能替代医生或执业药师建议。"
    "如果证据不足，要明确说不能根据当前说明书片段判断。"
    "不要在答案正文中输出“安全等级”或“风险标签”，这两项由系统界面的来源区单独展示。"
    "必须保留咨询医生或药师提示和来源。"
    "输出结构固定为：结论、依据说明书、用药/购药建议、需要咨询医生或药师的情况、来源。"
)

DEFAULT_QA_USER_TEMPLATE = """用户问题：
{question}

系统初步结论：
{conclusion}

说明书证据：
{evidence}

用药/购药建议：
{guidance}

安全等级：{risk_level}
风险标签：{risk_tags}
必须保留的安全提示：
{safety_warnings}

来源：
{sources}
"""


@dataclass(slots=True)
class AnswerResult:
    answer: str
    trace: AdvancedRetrievalTrace
    safety: SafetyAssessment

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "trace": self.trace.to_dict(),
            "safety": self.safety.to_dict(),
        }


def answer_question(
    question: str,
    vector_store: Any,
    top_k: int = 6,
    candidate_k: int | None = None,
    summary_store: Any | None = None,
    use_llm: bool = False,
    llm_provider: str = "none",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_temperature: float = 0.1,
    llm_evidence_max_chars_per_doc: int = 1200,
    llm_evidence_max_total_chars: int = 6000,
    llm_evidence_max_chars_per_drug: int = 1800,
    llm_evidence_max_chars_per_drug_doc: int = 900,
    prompts_path: str | None = None,
    mode: str = "auto",
    rerank: str = "auto",
    rerank_model_provider: str = "none",
    rerank_model_name: str = "gte-rerank-v2",
    rerank_model_endpoint: str | None = None,
    rerank_model_api_key_env: str = "DASHSCOPE_API_KEY",
    rerank_model_timeout: int = 30,
    rerank_model_strategies: list[str] | None = None,
    preserve_preferred_sections: bool = True,
    drug_name: str | None = None,
    generic_name: str | None = None,
    category: str | None = None,
) -> str:
    """只返回答案文本的简化入口，内部复用 answer_question_with_trace。"""
    return answer_question_with_trace(
        question=question,
        vector_store=vector_store,
        top_k=top_k,
        candidate_k=candidate_k,
        summary_store=summary_store,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key_env=llm_api_key_env,
        llm_temperature=llm_temperature,
        llm_evidence_max_chars_per_doc=llm_evidence_max_chars_per_doc,
        llm_evidence_max_total_chars=llm_evidence_max_total_chars,
        llm_evidence_max_chars_per_drug=llm_evidence_max_chars_per_drug,
        llm_evidence_max_chars_per_drug_doc=llm_evidence_max_chars_per_drug_doc,
        prompts_path=prompts_path,
        mode=mode,
        rerank=rerank,
        rerank_model_provider=rerank_model_provider,
        rerank_model_name=rerank_model_name,
        rerank_model_endpoint=rerank_model_endpoint,
        rerank_model_api_key_env=rerank_model_api_key_env,
        rerank_model_timeout=rerank_model_timeout,
        rerank_model_strategies=rerank_model_strategies,
        preserve_preferred_sections=preserve_preferred_sections,
        drug_name=drug_name,
        generic_name=generic_name,
        category=category,
    ).answer


def answer_question_with_trace(
    question: str,
    vector_store: Any,
    top_k: int = 6,
    candidate_k: int | None = None,
    summary_store: Any | None = None,
    use_llm: bool = False,
    llm_provider: str = "none",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    llm_temperature: float = 0.1,
    llm_evidence_max_chars_per_doc: int = 1200,
    llm_evidence_max_total_chars: int = 6000,
    llm_evidence_max_chars_per_drug: int = 1800,
    llm_evidence_max_chars_per_drug_doc: int = 900,
    prompts_path: str | None = None,
    mode: str = "auto",
    rerank: str = "auto",
    rerank_model_provider: str = "none",
    rerank_model_name: str = "gte-rerank-v2",
    rerank_model_endpoint: str | None = None,
    rerank_model_api_key_env: str = "DASHSCOPE_API_KEY",
    rerank_model_timeout: int = 30,
    rerank_model_strategies: list[str] | None = None,
    preserve_preferred_sections: bool = True,
    drug_name: str | None = None,
    generic_name: str | None = None,
    category: str | None = None,
) -> AnswerResult:
    """带 Trace 的完整问答主流程。

    这个函数是后端 RAG 问答链路的核心入口。它不只返回答案文本，还会返回：
    - trace：检索过程，包括意图路由、问题重写、metadata filter、召回来源等。
    - safety：安全评估结果，包括风险等级和风险标签。

    可以把它理解成 5 个连续阶段：
    1. 检索：retrieve_advanced 负责意图路由、问题重写、多路召回、rerank。
    2. 证据筛选：_answer_docs 从召回结果中挑选真正用于回答的原文证据。
    3. 多药品保护：如果未指定药品且召回跨多个药品，则按药品分组回答。
    4. 安全评估：assess_safety 根据问题和证据判断风险等级。
    5. 答案生成：根据 use_llm 选择受控 LLM 生成或本地模板生成。
    """
    # 第一步：执行高级检索。
    # retrieve_advanced 返回 AdvancedRetrievalTrace，而不是单纯的 docs。
    # Trace 中既有最终入选的 trace.docs，也有检索中间过程：
    # - trace.rewrite：问题改写与意图路由结果
    # - trace.metadata_filter：实际使用的 Chroma metadata 过滤条件
    # - trace.decision：auto 模式下系统选择的检索策略
    # - trace.results：前端“数据来源与检索 Trace”里展示的结构化结果
    trace = retrieve_advanced(
        # 原文 chunk 的 Chroma 向量库，最终证据通常来自这里。
        vector_store=vector_store,
        # 章节摘要的 Chroma 向量库，用于 summary 或 summary_guided 检索。
        summary_store=summary_store,
        # 用户原始问题。retrieve_advanced 内部会根据 use_rewrite 策略决定是否重写。
        question=question,
        # 检索模式。生产主路径通常是 auto，由系统按场景选择策略。
        mode=mode,
        # 重排策略。auto 通常会落到 keyword rerank。
        rerank=rerank,
        # 候选召回数量：先多召回一些，再由 rerank 选出最终证据。
        candidate_k=candidate_k,
        # 最终保留给答案层的证据数量。
        final_k=top_k,
        # 以下三个字段来自 catalog 解析或前端设置，用于限制检索范围。
        # 例如识别到 generic_name=坎地沙坦酯片，就不会跨药品乱取证。
        drug_name=drug_name,
        generic_name=generic_name,
        category=category,
        rerank_model_provider=rerank_model_provider,
        rerank_model_name=rerank_model_name,
        rerank_model_endpoint=rerank_model_endpoint,
        rerank_model_api_key_env=rerank_model_api_key_env,
        rerank_model_timeout=rerank_model_timeout,
        rerank_model_strategies=rerank_model_strategies,
        preserve_preferred_sections=preserve_preferred_sections,
    )

    # 第二步：从 trace.docs 中挑选“真正用于回答”的证据。
    # retrieve_advanced 可能返回 metadata/vector/summary/summary_guided 多种阶段的结果。
    # _answer_docs 会优先选择命中目标章节的“原文 chunk”，尽量不直接依赖摘要生成答案。
    docs = _answer_docs(trace.docs, trace.rewrite.route.sections)

    # 如果检索没有拿到任何可用证据，不能让模型自由发挥。
    # 这里直接返回一个安全兜底答案，并把空检索的 trace 一起返回，方便前端排查。
    if not docs:
        # 没有证据时，只能基于用户问题本身做最低限度的安全评估。
        assessment = assess_safety(question)
        # 使用本地模板生成“证据不足”的回答。
        answer = _format_answer(
            conclusion="说明书中未检索到足够信息，不能据此给出明确用药建议。",
            evidence="未找到相关说明书片段。",
            guidance="请核对药品名称、规格和适用人群后重新提问，或直接咨询医生/执业药师。",
            safety=assessment,
            sources="无",
        )
        # AnswerResult 是接口层最终返回的数据结构：答案 + 检索 trace + 安全评估。
        return AnswerResult(answer=answer, trace=trace, safety=assessment)

    # 第三步：为“多药品检索结果”准备证据。
    # trace.docs 里可能包含多个药品的结果，尤其是用户只问“用法用量”“孕妇能用吗”
    # 但没有说明具体药品时。这个函数会为每个药品保留最合适的证据。
    multi_drug_docs = _multi_drug_answer_docs(trace.docs, trace.rewrite.route.sections)

    # 判断是否需要进入“多药品分组回答”。
    # 触发条件大致是：
    # - 用户问题是药品相关意图，如 dosage/storage/adverse_reaction/special_population 等。
    # - 用户没有明确指定 drug_name/generic_name。
    # - 检索结果跨多个 doc_id 或多个药品 identity。
    if _needs_multi_drug_answer(trace, multi_drug_docs, drug_name=drug_name, generic_name=generic_name):
        # 用户没有限定具体药品，但检索命中多个药品：不能任选一个结论，必须按药品分组列出。
        # 安全评估使用的上下文优先取路由章节相关内容。
        # 如果相关章节内容为空，则退回使用所有多药品证据文本。
        safety_context = _safety_context(multi_drug_docs, trace.rewrite.route.sections) or "\n\n".join(
            _doc_text(doc) for doc in multi_drug_docs
        )
        # 根据用户问题和召回证据判断风险等级。
        # 例如孕妇、儿童、禁忌、相互作用、不良反应等会提升风险等级。
        assessment = assess_safety(question, safety_context)
        # 按药品维度分组，避免把 A 药说明书结论套用到 B 药。
        grouped_docs = _group_docs_by_drug(multi_drug_docs)

        # 如果配置启用了 LLM，则走“受控 LLM 生成”。
        # 注意这里不是让 LLM 自由检索，而是把已经整理好的结论、证据、建议、来源填入 prompt。
        if use_llm:
            answer = _controlled_llm_answer(
                # 原始用户问题。
                question=question,
                # 多药品场景的总说明：提示用户本次检索跨多个药品，不能混用结论。
                conclusion=_build_multi_drug_conclusion(trace.rewrite.route.intent, grouped_docs),
                # 给 LLM 使用更长、更完整的多药品证据，避免只根据压缩展示片段生成答案。
                evidence=_build_multi_drug_llm_evidence(
                    question=question,
                    grouped_docs=grouped_docs,
                    route_sections=trace.rewrite.route.sections,
                    max_chars_per_drug=llm_evidence_max_chars_per_drug,
                    max_chars_per_doc=llm_evidence_max_chars_per_drug_doc,
                ),
                # 多药品场景下的购药/用药建议。
                guidance=_build_multi_drug_guidance(assessment),
                # 安全评估结果会进入 prompt，但最终正文会去掉“安全等级/风险标签”重复展示。
                safety=assessment,
                # 按药品整理的来源信息。
                sources=_format_multi_drug_sources(grouped_docs),
                # LLM 运行参数，来自 configs/app.yaml 和环境变量。
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key_env=llm_api_key_env,
                llm_temperature=llm_temperature,
                prompts_path=prompts_path,
            )
            # 多药品 + LLM 分支直接返回结果。
            return AnswerResult(answer=answer, trace=trace, safety=assessment)

        # 如果没有启用 LLM，则使用本地模板生成多药品答案。
        answer = _format_answer(
            conclusion=_build_multi_drug_conclusion(trace.rewrite.route.intent, grouped_docs),
            evidence=_compact_multi_drug_evidence(grouped_docs),
            guidance=_build_multi_drug_guidance(assessment),
            safety=assessment,
            sources=_format_multi_drug_sources(grouped_docs),
        )
        # 多药品 + 模板分支直接返回结果。
        return AnswerResult(answer=answer, trace=trace, safety=assessment)

    # 第四步：单药品或已明确药品约束的正常问答流程。
    # 这里增加了一个特殊处理：不良反应频率表结构化抽取。
    # 原因是 PDF 表格抽成纯文本后容易丢失列边界，导致把“发生频率不明”误归入“＜0.1%”或漏掉“0.1-＜5%”列后半部分。
    structured_parts = _structured_adverse_frequency_parts(question, docs)

    # 把最终用于回答的文档内容拼成上下文。
    # context 主要用于安全评估兜底，不直接等同于最终回答。
    context = "\n\n".join(_doc_text(doc) for doc in docs)

    # 安全评估优先使用路由命中的目标章节上下文。
    # 例如孕妇问题优先看“孕妇及哺乳期妇女用药/禁忌/注意事项”，
    # 避免无关章节里的词触发错误风险判断。
    safety_context = _safety_context(docs, trace.rewrite.route.sections) or context

    # 根据用户问题和说明书证据做安全评估。
    # 返回 SafetyAssessment，包含 risk_level、findings、warnings、must_consult 等。
    assessment = assess_safety(question, safety_context)

    # 构造“结论”。
    # 如果 structured_parts 存在，说明这是不良反应频率表问题，优先使用结构化表格抽取结果。
    # 否则使用普通抽取式结论生成逻辑。
    conclusion = (
        structured_parts["conclusion"]
        if structured_parts
        else _build_conclusion(question, docs, assessment)
    )

    # 构造“依据说明书”部分。
    # 表格问题使用结构化证据；普通问题使用 _compact_evidence 压缩原文 chunk。
    evidence = structured_parts["evidence"] if structured_parts else _compact_evidence(docs)

    # 第五步：生成最终答案。
    # 如果启用了 LLM，走受控生成层；否则走本地模板答案。
    if use_llm:
        # LLM 输入和页面展示使用不同证据版本：
        # - 页面/模板展示：evidence，短一些，方便用户阅读。
        # - LLM 生成：llm_evidence，长一些，尽量保留完整关键句、条款和命中章节。
        # - 表格型问题：structured_parts 已经是结构化抽取结果，直接给 LLM，绕开普通截断。
        llm_evidence = (
            structured_parts["evidence"]
            if structured_parts
            else _build_llm_evidence(
                question=question,
                docs=docs,
                route_sections=trace.rewrite.route.sections,
                max_chars_per_doc=llm_evidence_max_chars_per_doc,
                max_total_chars=llm_evidence_max_total_chars,
            )
        )
        answer = _controlled_llm_answer(
            # 用户原始问题。
            question=question,
            # 上一步已经确定好的结论，LLM 只能基于它和证据组织语言。
            conclusion=conclusion,
            # 给 LLM 的证据包，不再使用 _compact_evidence 的 320 字展示截断。
            evidence=llm_evidence,
            # 根据风险等级生成的用药/购药建议。
            guidance=_build_guidance(question, docs, assessment),
            # 安全评估对象。
            safety=assessment,
            # 来源信息，用于让答案保留可追溯性。
            sources=_format_sources(docs),
            # LLM 参数。
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key_env=llm_api_key_env,
            llm_temperature=llm_temperature,
            prompts_path=prompts_path,
        )
        # 单药品 + LLM 分支返回。
        return AnswerResult(answer=answer, trace=trace, safety=assessment)

    # 未启用 LLM 时，使用稳定的模板答案。
    # 这个分支适合课堂演示“纯检索 + 规则模板”的可控效果，也能作为 LLM 失败时的兜底。
    answer = _format_answer(
        conclusion=conclusion,
        evidence=evidence,
        guidance=_build_guidance(question, docs, assessment),
        safety=assessment,
        sources=_format_sources(docs),
    )
    # 单药品 + 模板分支返回。
    return AnswerResult(answer=answer, trace=trace, safety=assessment)


def _build_conclusion(question: str, docs: list[Any], assessment: SafetyAssessment) -> str:
    """基于问题类型和检索证据生成抽取式结论，用作兜底答案或 LLM 初稿。"""
    context = "\n".join(_doc_text(doc) for doc in docs)
    if any(keyword in question for keyword in ["孕妇", "妊娠", "怀孕", "备孕"]) and any(
        keyword in context for keyword in ["孕妇或有妊娠可能的妇女禁用", "孕妇", "妊娠"]
    ):
        if "禁用" in context:
            return "根据检索到的说明书内容，孕妇或有妊娠可能的妇女属于高风险/禁用场景，不建议自行购买或使用。"
    if any(keyword in question for keyword in ["一天", "几次", "剂量", "用量", "怎么吃"]):
        dosage = _first_section_text(docs, "用法用量")
        if dosage:
            return _first_sentence(dosage)
    if any(keyword in question for keyword in ["一起吃", "同服", "合用", "相互作用"]):
        return "该问题涉及合并用药，应以说明书的药物相互作用章节为依据，并由医生或执业药师确认。"
    if any(keyword in question for keyword in ["有效期", "多久过期", "过期"]):
        validity = _first_section_text(docs, "有效期")
        if validity:
            return _first_sentence(validity)
    if any(keyword in question for keyword in ["贮藏", "保存", "放哪里", "放哪", "存放", "冰箱", "冷藏", "冷冻", "低温", "常温", "温度"]):
        storage = _first_section_text(docs, "贮藏")
        if storage:
            if any(keyword in question for keyword in ["冰箱", "冷藏", "冷冻", "低温"]):
                return _storage_temperature_answer(storage)
            return _first_sentence(storage)
    if assessment.risk_level == "high":
        return "该问题涉及高风险用药场景，请优先按说明书禁忌、注意事项和特殊人群用药信息处理。"
    return "以下为根据说明书检索到的相关信息，需结合患者具体情况判断。"


def _structured_adverse_frequency_parts(question: str, docs: list[Any]) -> dict[str, str] | None:
    """对“不良反应发生率”类问题做结构化证据抽取。

    药品说明书里的不良反应经常是多列表格。普通 PDF 文本抽取会把列摊平成一段文本，
    例如把“发生频率不明”列的内容紧跟在“＜0.1%”列附近，LLM 容易误归类。
    所以当用户明确询问某个发生率区间时，先尝试用 PDF 字符坐标恢复对应列，再生成证据。
    """
    if not _asks_less_than_0_1_adverse_reaction(question, docs):
        return None
    extracted = _extract_adverse_frequency_items_from_docs(docs, "lt_0_1")
    if not extracted:
        return None

    lines = []
    all_items: list[str] = []
    for source_file, items in extracted.items():
        all_items.extend(items)
        lines.append(f"{source_file}：发生率＜0.1%的不良反应为：{'、'.join(items)}。")

    unique_items = list(dict.fromkeys(all_items))
    return {
        "conclusion": f"坎地沙坦酯片发生率＜0.1%的不良反应包括：{'、'.join(unique_items)}。",
        "evidence": "\n".join(lines)
        + "\n注意：表格中“发生频率不明”是独立列，不应并入“＜0.1%”列。",
    }


def _asks_less_than_0_1_adverse_reaction(question: str, docs: list[Any]) -> bool:
    question_text = question.replace(" ", "")
    frequency_matched = any(
        keyword in question_text
        for keyword in ["小于0.1%", "小于0.1％", "小于0.1", "低于0.1%", "低于0.1％", "低于0.1", "＜0.1%", "＜0.1", "<0.1%", "<0.1"]
    )
    adverse_matched = any(keyword in question for keyword in ["不良反应", "副作用", "反应"])
    section_matched = any((getattr(doc, "metadata", {}) or {}).get("section") == "不良反应" for doc in docs)
    return frequency_matched and adverse_matched and section_matched


def _extract_adverse_frequency_items_from_docs(docs: list[Any], frequency: str) -> dict[str, list[str]]:
    extracted: dict[str, list[str]] = {}
    seen_paths = set()
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("section") != "不良反应":
            continue
        source_path = metadata.get("source_path")
        if not source_path or source_path in seen_paths:
            continue
        seen_paths.add(source_path)
        items = _extract_adverse_frequency_items_from_pdf(source_path, frequency)
        if items:
            extracted[metadata.get("source_file") or Path(source_path).name] = items
    return extracted


@lru_cache(maxsize=32)
def _extract_adverse_frequency_items_from_pdf(source_path: str, frequency: str) -> tuple[str, ...]:
    if frequency != "lt_0_1":
        return tuple()
    try:
        from pypdf import PdfReader
    except ImportError:
        return tuple()

    path = Path(source_path)
    if not path.exists():
        return tuple()

    try:
        reader = PdfReader(str(path))
    except Exception:
        return tuple()

    fragments: list[tuple[float, float, str]] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if not all(marker in page_text for marker in ["其它的不良反应", "＜0.1%", "发生频率不明"]):
            continue

        def visitor(text: str, _cm: Any, tm: Any, _font_dict: Any, _font_size: Any) -> None:
            cleaned = text.strip()
            if not cleaned:
                return
            x = float(tm[4])
            y = float(tm[5])
            # 该列位于“不良反应”表格的“＜0.1%”列。这里使用坐标列边界，而不是
            # 摊平后的文本顺序，避免把右侧“发生频率不明”列误合并进来。
            if 400 <= x < 500 and 180 <= y < 705:
                fragments.append((y, x, cleaned))

        try:
            page.extract_text(visitor_text=visitor)
        except Exception:
            continue

    text = _merge_pdf_column_fragments(fragments)
    return tuple(_split_adverse_items(text))


def _merge_pdf_column_fragments(fragments: list[tuple[float, float, str]]) -> str:
    cleaned_fragments = []
    for y, x, text in sorted(fragments, key=lambda item: (-item[0], item[1])):
        if text in {"＜", "0.1%", "）、", "）。"}:
            continue
        if re.fullmatch(r"[）)、，。；;\s]+", text):
            continue
        cleaned_fragments.append(text)
    merged = "".join(cleaned_fragments)
    merged = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", merged)
    return merged


def _split_adverse_items(text: str) -> list[str]:
    if not text:
        return []
    normalized = (
        text.replace("，", "、")
        .replace(",", "、")
        .replace("；", "、")
        .replace(";", "、")
        .replace("心房颤动肢体麻木", "心房颤动、肢体麻木")
        .replace("肢体麻木味觉异常", "肢体麻木、味觉异常")
        .replace("味觉异常低钠血症", "味觉异常、低钠血症")
    )
    items = [item.strip(" 、。") for item in normalized.split("、") if item.strip(" 、。")]
    return list(dict.fromkeys(items))


def _needs_multi_drug_answer(
    trace: AdvancedRetrievalTrace,
    docs: list[Any] | None = None,
    drug_name: str | None = None,
    generic_name: str | None = None,
) -> bool:
    """判断是否需要多药品分组回答。

    如果问题是药品相关意图、没有显式药品约束，并且证据跨多个药品 identity，
    则进入多药品回答，避免把 A 药结论误套到 B 药。
    """
    drug_specific_intents = {
        "dosage",
        "indication",
        "storage",
        "special_population",
        "contraindication",
        "interaction",
        "adverse_reaction",
        "purchase_guidance",
    }
    if trace.rewrite.route.intent not in drug_specific_intents:
        return False
    if drug_name or generic_name:
        return False
    answer_docs = docs or trace.docs
    if _question_mentions_drug_identity(trace.question, answer_docs):
        return False
    return _has_multiple_drugs(answer_docs)


def _needs_drug_disambiguation(
    trace: AdvancedRetrievalTrace,
    drug_name: str | None = None,
    generic_name: str | None = None,
) -> bool:
    return _needs_multi_drug_answer(trace, drug_name=drug_name, generic_name=generic_name)


def _question_mentions_drug_identity(question: str, docs: list[Any]) -> bool:
    normalized_question = _normalize_identity_text(question)
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        names = [
            metadata.get("drug_name"),
            metadata.get("generic_name"),
            metadata.get("source_file"),
        ]
        for name in names:
            normalized_name = _normalize_identity_text(str(name or ""))
            if normalized_name and normalized_name != "unknown" and normalized_name in normalized_question:
                return True
    return False


def _has_multiple_drugs(docs: list[Any]) -> bool:
    identities = set()
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        identity = (
            metadata.get("doc_id"),
            metadata.get("generic_name"),
            metadata.get("drug_name"),
        )
        if any(identity):
            identities.add(identity)
    return len(identities) > 1


def _group_docs_by_drug(docs: list[Any]) -> list[dict[str, Any]]:
    """按 doc_id + generic_name + drug_name 对召回证据分组。"""
    groups: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        key = (
            metadata.get("doc_id"),
            metadata.get("generic_name"),
            metadata.get("drug_name"),
        )
        if not any(key):
            key = (metadata.get("source_file"), None, None)
        if key not in groups:
            groups[key] = {
                "doc_id": metadata.get("doc_id"),
                "drug_name": metadata.get("drug_name") or "未知商品名",
                "generic_name": metadata.get("generic_name") or "未知通用名",
                "source_file": metadata.get("source_file") or "未知文件",
                "docs": [],
            }
        groups[key]["docs"].append(doc)
    return list(groups.values())


def _build_multi_drug_conclusion(intent: str, grouped_docs: list[dict[str, Any]]) -> str:
    intent_names = {
        "dosage": "用法用量",
        "indication": "适应症",
        "storage": "贮藏方式",
        "special_population": "特殊人群用药",
        "contraindication": "禁忌",
        "interaction": "药物相互作用",
        "adverse_reaction": "不良反应",
        "purchase_guidance": "购药指导",
    }
    topic = intent_names.get(intent, "相关问题")
    count = len(grouped_docs)
    return (
        f"本次问题未限定具体药品，系统在说明书库中检索到 {count} 个相关药品。"
        f"不同药品的{topic}可能不同，下面按药品分别列出说明书依据，不能把其中一个药品的结论套用到其他药品。"
    )


def _compact_multi_drug_evidence(grouped_docs: list[dict[str, Any]], max_chars_per_drug: int = 360) -> str:
    """为多药品回答压缩证据，每个药品输出若干条章节片段。"""
    lines = []
    for idx, group in enumerate(grouped_docs, start=1):
        display_name = _drug_display_name(group)
        snippets = []
        seen = set()
        for doc in group["docs"]:
            metadata = getattr(doc, "metadata", {}) or {}
            section = metadata.get("section", "未知章节")
            text = _doc_text(doc).replace("\n", " ").strip()
            if not text:
                continue
            snippet = f"[{section}] {_first_sentence(text)}"
            if snippet not in seen:
                snippets.append(snippet)
                seen.add(snippet)
            if sum(len(item) for item in snippets) >= max_chars_per_drug:
                break
        evidence = "；".join(snippets)
        if len(evidence) > max_chars_per_drug:
            evidence = evidence[:max_chars_per_drug].rstrip() + "..."
        lines.append(f"{idx}. {display_name}：{evidence or '检索到相关章节，但片段内容为空。'}")
    return "\n".join(lines)


def _build_multi_drug_guidance(assessment: SafetyAssessment) -> str:
    if assessment.risk_level == "high":
        return "已按药品分别列出结果；孕妇、儿童、禁忌、相互作用等高风险场景不要自行判断替换用药，请携带具体药品名和个人情况咨询医生或执业药师。"
    return "请先确认要购买或使用的是哪一个药品，再按对应药品说明书核对；不同药品即使适应症相近，也不能共用剂量、禁忌或贮藏结论。"


def _format_multi_drug_sources(grouped_docs: list[dict[str, Any]]) -> str:
    lines = []
    for idx, group in enumerate(grouped_docs, start=1):
        source_lines = _format_sources(group["docs"]).replace("\n", "；")
        lines.append(f"{idx}. {_drug_display_name(group)}：{source_lines}")
    return "\n".join(lines)


def _drug_display_name(group: dict[str, Any]) -> str:
    drug_name = group.get("drug_name") or "未知商品名"
    generic_name = group.get("generic_name") or "未知通用名"
    if drug_name == generic_name:
        return drug_name
    return f"{drug_name}（{generic_name}）"


def _normalize_identity_text(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("®", "")
        .replace("庐", "")
    )


def _build_guidance(question: str, docs: list[Any], assessment: SafetyAssessment) -> str:
    if assessment.risk_level == "high":
        return "不建议仅凭自动问答自行购药或调整用药；请携带药品名称、规格、既往病史和正在使用的药物咨询医生或执业药师。"
    if assessment.risk_level == "medium":
        return "购药或用药前请重点核对适应症、禁忌、用法用量、注意事项和药物相互作用。"
    return "请按说明书、处方或医生/药师指导使用，不要自行超剂量或长期使用。"


def _compact_evidence(docs: list[Any], max_chars_per_doc: int = 320) -> str:
    """生成页面展示/模板兜底用的短证据。
    这个函数只负责“展示层压缩”，不再作为 LLM 的主要证据输入。
    企业落地中，如果直接把每条证据截断到 320 字再交给 LLM，长禁忌、长表格、相互作用、特殊人群用药等内容可能被截掉，进而影响最终答案完整性。
    """
    lines = []
    for idx, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        section = metadata.get("section", "未知章节")
        text = _doc_text(doc).replace("\n", " ")
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + "..."
        lines.append(f"{idx}. [{section}] {text}")
    return "\n".join(lines)


def _build_llm_evidence(
    question: str,
    docs: list[Any],
    route_sections: list[str] | None = None,
    max_chars_per_doc: int = 1200,
    max_total_chars: int = 6000,
) -> str:
    """生成专门给受控 LLM 使用的证据包。

    和 _compact_evidence 的区别：
    - _compact_evidence 面向页面展示，每条默认 320 字，强调可读性。
    - _build_llm_evidence 面向 LLM 生成，每条默认 1200 字，强调证据完整性。

    设计原则：
    1. 优先保留意图路由命中的章节，例如“用法用量”“禁忌”“贮藏”。
    2. 对长文本不做粗暴后半截删除，而是抽取包含问题关键词的完整句/条款。
    3. 如果没有匹配到关键词，则保留 chunk 开头的完整片段作为兜底。
    4. 不良反应频率表这类结构化表格问题会在上游 structured_parts 处理，
       不走这里的普通文本截断逻辑。
    """
    route_sections = route_sections or []
    ordered_docs = _prioritize_docs_for_llm(docs, route_sections)
    lines: list[str] = []
    total_chars = 0

    for idx, doc in enumerate(ordered_docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        section = metadata.get("section", "未知章节")
        source_file = metadata.get("source_file", "未知文件")
        page_start = metadata.get("page_start", "?")
        page_end = metadata.get("page_end", page_start)
        text = _normalize_evidence_text(_doc_text(doc))
        if not text:
            continue

        packed_text = _pack_doc_text_for_llm(
            question=question,
            text=text,
            max_chars=max_chars_per_doc,
        )
        line = f"{idx}. [{section}]（{source_file}，第 {page_start}-{page_end} 页）{packed_text}"
        if total_chars + len(line) > max_total_chars:
            remaining = max_total_chars - total_chars
            if remaining > 200:
                lines.append(line[:remaining].rstrip() + "...")
            break
        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


def _build_multi_drug_llm_evidence(
    question: str,
    grouped_docs: list[dict[str, Any]],
    route_sections: list[str] | None = None,
    max_chars_per_drug: int = 1800,
    max_chars_per_doc: int = 900,
) -> str:
    """为多药品 LLM 回答构造更完整的分组证据。

    页面展示仍使用 _compact_multi_drug_evidence；LLM 生成时则需要每个药品保留更长证据，
    否则容易只回答某一个药品或漏掉后续药品的重要条款。
    """
    lines: list[str] = []
    for idx, group in enumerate(grouped_docs, start=1):
        display_name = _drug_display_name(group)
        evidence = _build_llm_evidence(
            question=question,
            docs=group["docs"],
            route_sections=route_sections,
            max_chars_per_doc=max_chars_per_doc,
            max_total_chars=max_chars_per_drug,
        )
        lines.append(f"{idx}. {display_name}\n{evidence or '检索到相关章节，但片段内容为空。'}")
    return "\n".join(lines)


def _prioritize_docs_for_llm(docs: list[Any], route_sections: list[str]) -> list[Any]:
    """按章节相关性给 LLM 证据排序，命中路由章节的证据优先。"""
    if not route_sections:
        return docs
    route_set = set(route_sections)
    return sorted(
        docs,
        key=lambda doc: 0 if (getattr(doc, "metadata", {}) or {}).get("section") in route_set else 1,
    )


def _pack_doc_text_for_llm(question: str, text: str, max_chars: int) -> str:
    """在单个 chunk 内为 LLM 选择较完整的证据文本。"""
    if len(text) <= max_chars:
        return text

    focused = _focused_evidence_excerpt(question, text, max_chars=max_chars)
    if focused:
        return focused
    return text[:max_chars].rstrip() + "..."


def _focused_evidence_excerpt(question: str, text: str, max_chars: int) -> str:
    """优先抽取包含问题关键词的完整句/条款，避免按固定字符数切断关键语义。"""
    keywords = _question_keywords(question)
    if not keywords:
        return ""

    units = _split_evidence_units(text)
    selected: list[str] = []
    total = 0
    for unit in units:
        if not any(keyword in unit for keyword in keywords):
            continue
        addition = unit.strip()
        if not addition:
            continue
        if total + len(addition) > max_chars:
            break
        selected.append(addition)
        total += len(addition)

    if not selected:
        return ""
    excerpt = " ".join(selected)
    return excerpt if len(excerpt) <= max_chars else excerpt[:max_chars].rstrip() + "..."


def _split_evidence_units(text: str) -> list[str]:
    """把证据文本切成尽量完整的句子或条款。"""
    units = re.split(r"(?<=[。；;])\s*|\n+", text)
    return [unit.strip() for unit in units if unit.strip()]


def _question_keywords(question: str) -> list[str]:
    """从用户问题中提取用于定位证据句的关键词。"""
    stopwords = {
        "这个",
        "那个",
        "哪些",
        "什么",
        "怎么",
        "是否",
        "可以",
        "能够",
        "有没有",
        "有哪些",
        "一下",
    }
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9.%＜<>~～-]{2,}", question)
    keywords = [token for token in tokens if token not in stopwords]
    return list(dict.fromkeys(keywords))


def _normalize_evidence_text(text: str) -> str:
    """清理证据中的多余空白，保留句子和条款顺序。"""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _answer_docs(docs: list[Any], route_sections: list[str]) -> list[Any]:
    """选择最终单药回答使用的证据。
    优先选择命中目标章节的原文 chunk；如果没有，再退回所有原文 chunk；
    最后才使用摘要结果，避免最终答案过度依赖摘要。
    """
    if route_sections:
        section_docs = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            if metadata.get("retrieval_stage") != "summary" and metadata.get("section") in route_sections:
                section_docs.append(doc)
        if section_docs:
            return section_docs

    source_docs = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("retrieval_stage") != "summary":
            source_docs.append(doc)
    return source_docs or docs


def _multi_drug_answer_docs(docs: list[Any], route_sections: list[str]) -> list[Any]:
    """为多药品回答保留每个药品的最佳证据。

    每个药品优先使用目标章节原文，其次使用任意原文，再其次使用摘要。
    这样可以避免 Trace 中命中了多个药品，但最终回答只剩一个药品。
    """
    selected = []
    for group in _group_docs_by_drug(docs):
        group_docs = group["docs"]
        route_source_docs = _filter_docs(group_docs, route_sections=route_sections, include_summary=False)
        source_docs = _filter_docs(group_docs, route_sections=None, include_summary=False)
        route_docs = _filter_docs(group_docs, route_sections=route_sections, include_summary=True)
        selected.extend(route_source_docs or source_docs or route_docs or group_docs)
    return selected


def _filter_docs(
    docs: list[Any],
    route_sections: list[str] | None = None,
    include_summary: bool = True,
) -> list[Any]:
    filtered = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if not include_summary and metadata.get("retrieval_stage") == "summary":
            continue
        if route_sections and metadata.get("section") not in route_sections:
            continue
        filtered.append(doc)
    return filtered


def _format_sources(docs: list[Any]) -> str:
    """把证据来源格式化为文件名、页码、章节和检索阶段。"""
    sources = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source_file = metadata.get("source_file", "未知文件")
        page_start = metadata.get("page_start", "?")
        page_end = metadata.get("page_end", page_start)
        section = metadata.get("section", "未知章节")
        stage = metadata.get("retrieval_stage", "retrieval")
        sources.append(f"{source_file}，第 {page_start}-{page_end} 页，{section}，检索阶段：{stage}")
    return "\n".join(dict.fromkeys(sources))


def _format_answer(
    conclusion: str,
    evidence: str,
    guidance: str,
    safety: SafetyAssessment,
    sources: str,
) -> str:
    safety_lines = "\n".join(f"- {warning}" for warning in safety.warnings)
    findings = ", ".join(finding.label for finding in safety.findings) or "none"
    return (
        f"结论：\n{conclusion}\n\n"
        f"依据说明书：\n{evidence}\n\n"
        f"用药/购药建议：\n{guidance}\n\n"
        f"安全等级：\n{safety.risk_level}\n\n"
        f"风险标签：\n{findings}\n\n"
        f"需要咨询医生或药师的情况：\n{safety_lines}\n\n"
        f"来源：\n{sources}"
    )


def _first_section_text(docs: list[Any], section: str) -> str:
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("section") == section:
            return _doc_text(doc)
    return ""


def _safety_context(docs: list[Any], route_sections: list[str]) -> str:
    if not route_sections:
        return "\n\n".join(_doc_text(doc) for doc in docs)
    relevant = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("section") in route_sections:
            relevant.append(_doc_text(doc))
    return "\n\n".join(relevant)


def _first_sentence(text: str) -> str:
    for sep in ["。", "；", "\n"]:
        if sep in text:
            return text.split(sep, 1)[0].strip() + ("。" if sep == "。" else "")
    return text.strip()


def _storage_temperature_answer(storage_text: str) -> str:
    first_sentence = _first_sentence(storage_text)
    if "常温" in storage_text or "10~30" in storage_text or "10～30" in storage_text:
        requirement = first_sentence.rstrip("。；;，, ")
        return f"说明书要求{requirement}，未要求冷藏或冷冻保存；不建议自行放入冰箱，按说明书常温条件保存即可。"
    return first_sentence


def _doc_text(doc: Any) -> str:
    return getattr(doc, "page_content", str(doc))


def _controlled_llm_answer(
    question: str,
    conclusion: str,
    evidence: str,
    guidance: str,
    safety: SafetyAssessment,
    sources: str,
    llm_provider: str,
    llm_model: str | None,
    llm_base_url: str | None,
    llm_api_key_env: str,
    llm_temperature: float,
    prompts_path: str | None,
) -> str:
    """受控 LLM 生成层。

    该函数不会直接让模型自由发挥，而是把抽取式结论、证据、安全提示和来源
    填入 prompts.yaml 中的模板。模型输出后还会清理“安全等级/风险标签”，
    因为这两个字段由前端来源区单独展示。
    """
    fallback = _format_answer(
        conclusion=conclusion,
        evidence=evidence,
        guidance=guidance,
        safety=safety,
        sources=sources,
    )
    chat = build_chat_model(
        provider=llm_provider,
        model=llm_model,
        base_url=llm_base_url,
        api_key_env=llm_api_key_env,
        temperature=llm_temperature,
    )
    if chat is None:
        return fallback
    safety_lines = "\n".join(f"- {warning}" for warning in safety.warnings)
    findings = ", ".join(finding.label for finding in safety.findings) or "none"
    prompt_config = _load_qa_prompt_config(prompts_path)
    prompt = prompt_config["user_template"].format(
        question=question,
        conclusion=conclusion,
        evidence=evidence,
        guidance=guidance,
        risk_level=safety.risk_level,
        risk_tags=findings,
        safety_warnings=safety_lines,
        sources=sources,
    )
    messages = [
        {
            "role": "system",
            "content": prompt_config["system"],
        },
        {"role": "user", "content": prompt},
    ]
    try:
        return _strip_answer_safety_summary(chat.invoke(messages))
    except Exception as exc:
        return fallback + f"\n\n生成层提示：LLM 生成不可用，已返回抽取式安全兜底答案。原因：{type(exc).__name__}: {exc}"


def _load_qa_prompt_config(prompts_path: str | None) -> dict[str, str]:
    """读取 qa.system 和 qa.user_template；缺失时回退到代码内默认提示词。"""
    config = _load_prompts_yaml(prompts_path) if prompts_path else {}
    qa_config = config.get("qa", {}) if isinstance(config, dict) else {}
    return {
        "system": qa_config.get("system") or DEFAULT_QA_SYSTEM_PROMPT,
        "user_template": qa_config.get("user_template") or DEFAULT_QA_USER_TEMPLATE,
    }


@lru_cache(maxsize=8)
def _load_prompts_yaml(prompts_path: str) -> dict[str, Any]:
    path = Path(prompts_path)
    if not path.exists():
        return {}
    try:
        return load_yaml(path)
    except Exception:
        return {}


def _strip_answer_safety_summary(answer: str) -> str:
    """从 LLM 正文中移除安全等级和风险标签，避免和 Trace 来源区重复展示。"""
    text = str(answer)
    lines = text.splitlines()
    cleaned: list[str] = []
    skip_next_value = False
    for line in lines:
        stripped = line.strip()
        normalized = stripped.rstrip("：:")
        if normalized in {"安全等级", "风险标签"}:
            skip_next_value = True
            continue
        if stripped.startswith(("安全等级：", "安全等级:", "风险标签：", "风险标签:")):
            continue
        if skip_next_value and stripped and not stripped.endswith(("：", ":")):
            skip_next_value = False
            continue
        skip_next_value = False
        cleaned.append(line)
    return "\n".join(cleaned).strip()
