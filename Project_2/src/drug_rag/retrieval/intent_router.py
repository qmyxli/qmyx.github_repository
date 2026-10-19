from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class IntentRule:
    intent: str
    keywords: list[str]
    sections: list[str]
    priority: float = 1.0


@dataclass(slots=True)
class QueryRoute:
    intent: str
    sections: list[str]
    metadata_filter: dict[str, Any]
    matched_keywords: list[str]
    confidence: float
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


INTENT_RULES: list[IntentRule] = [
    IntentRule(
        intent="indication",
        keywords=["治什么", "适应症", "能治", "用于", "高血压", "什么病", "适合什么病"],
        sections=["适应症"],
        priority=1.0,
    ),
    IntentRule(
        intent="storage",
        keywords=[
            "保存",
            "贮藏",
            "有效期",
            "放哪里",
            "放哪",
            "存放",
            "过期",
            "包装",
            "冰箱",
            "放冰箱",
            "冷藏",
            "冷冻",
            "低温",
            "常温",
            "温度",
            "阴凉",
            "避光",
        ],
        sections=["贮藏", "有效期", "包装"],
        priority=1.0,
    ),
    IntentRule(
        intent="purchase_guidance",
        keywords=["能买吗", "买药", "药店", "处方", "推荐", "购买", "用药指导"],
        sections=["适应症", "禁忌", "注意事项", "用法用量"],
        priority=1.1,
    ),
    IntentRule(
        intent="adverse_reaction",
        keywords=["副作用", "不良反应", "头晕", "恶心", "不舒服", "风险", "反应"],
        sections=["不良反应"],
        priority=1.2,
    ),
    IntentRule(
        intent="contraindication",
        keywords=["不能吃", "禁忌", "过敏", "不适合", "慎用", "哪些人不能", "不建议"],
        sections=["禁忌", "注意事项"],
        priority=1.3,
    ),

    IntentRule(
        intent="interaction",
        keywords=["一起吃", "同服", "相互作用", "合用", "搭配", "联用", "其他药", "降压药"],
        sections=["药物相互作用"],
        priority=1.3,
    ),
    IntentRule(
        intent="dosage",
        keywords=["怎么吃", "吃几次", "剂量", "用量", "服用", "一次几片", "一天几次", "一天吃几次", "每日", "每次", "口服"],
        sections=["用法用量"],
        priority=1.4,
    ),
    IntentRule(
        intent="special_population",
        keywords=["孕妇", "妊娠", "怀孕", "备孕", "哺乳", "儿童", "小孩", "孩子", "婴儿", "宝宝", "未成年", "老人", "老年", "高龄", "肝功能", "肾功能", "肝损伤", "肾损伤"],
        sections=["孕妇及哺乳期妇女用药", "儿童用药", "老年用药", "老年患者用药", "注意事项"],
        priority=1.6,
    ),

]


def route_question(question: str) -> QueryRoute:
    """根据用户自然语言问题判断业务意图，并给出对应说明书章节。

    当前项目没有用 LLM 来判断意图，而是用“关键词规则 + 优先级打分”的方式。
    这样做的好处是稳定、可解释、低成本，并且适合药品问答这种高风险场景。

    返回的 QueryRoute 会被后续链路继续使用：
    - intent：问题意图，例如 dosage、storage、special_population、general_qa。
    - sections：该意图通常应该检索的说明书章节。
    - metadata_filter：基于 sections 生成的 Chroma metadata filter。
    - matched_keywords：命中的关键词，方便前端 Trace 展示和调试。
    - confidence/scores：规则打分结果，用于观察路由是否明确。
    """
    # 先做基础归一化：去掉首尾空白，并转小写。
    # 中文不受 lower 影响，但如果问题里有英文或缩写，转小写可以提高匹配稳定性。
    normalized = question.strip().lower()

    # 对每一条 IntentRule 计算匹配分数。
    # _score_rule 会检查该规则的 keywords 是否出现在用户问题中，
    # 并结合关键词长度和规则 priority 计算一个分数。
    scored = [_score_rule(normalized, rule) for rule in INTENT_RULES]

    # 过滤掉 0 分规则，只保留真正命中的意图候选。
    # 每个元素结构是：(score, rule, matched_keywords)。
    scored = [item for item in scored if item[0] > 0]

    # 如果没有任何关键词命中，就进入 general_qa 兜底意图。
    # general_qa 不绑定固定章节，也不生成 section filter。
    # 后续 retrieval_policy 会把它路由到 summary_guided_hybrid：
    # 先用摘要索引找可能相关章节，再回原文 chunk 取证，并用向量检索兜底。
    if not scored:
        return QueryRoute(
            intent="general_qa",
            sections=[],
            metadata_filter={},
            matched_keywords=[],
            confidence=0.0,
            scores={},
        )

    # 如果命中了多个意图规则，按分数从高到低排序。
    # 例如“孕妇能不能吃”可能同时命中“不能吃”(contraindication)
    # 和“孕妇”(special_population)，排序后选择分数最高的意图。
    scored.sort(key=lambda item: item[0], reverse=True)

    # 取分数最高的规则作为最终路由。
    best_score, best_rule, best_keywords = scored[0]

    # 所有命中规则的总分，用来计算一个相对置信度。
    # 如果只有一个规则命中，confidence 通常为 1.0。
    # 如果多个规则竞争，confidence 会降低，说明该问题有多重意图。
    total_score = sum(score for score, _, _ in scored)
    confidence = round(best_score / total_score, 3) if total_score else 0.0

    # 保存每个命中 intent 的得分，供 Trace 展示。
    # 这能帮助我们解释为什么系统选择了某个意图。
    scores = {rule.intent: round(score, 3) for score, rule, _ in scored}

    # 默认使用最佳规则里预设的说明书章节。
    # 例如 dosage -> ["用法用量"]，storage -> ["贮藏", "有效期", "包装"]。
    sections = best_rule.sections

    # 特殊人群问题需要进一步细分章节。
    # 因为 special_population 里包含孕妇、哺乳、儿童、老人、肝肾功能异常等多种场景，
    # 不同关键词应该映射到不同章节，避免一次性检索过宽。
    if best_rule.intent == "special_population":
        sections = _special_population_sections(best_keywords)

    # 组装最终路由结果。
    # _section_filter 会把 sections 转成 Chroma 可用的 metadata filter：
    # - 单章节：{"section": "用法用量"}
    # - 多章节：{"section": {"$in": ["贮藏", "有效期", "包装"]}}
    return QueryRoute(
        intent=best_rule.intent,
        sections=sections,
        metadata_filter=_section_filter(sections),
        matched_keywords=best_keywords,
        confidence=confidence,
        scores=scores,
    )


def build_metadata_filter(
    route: QueryRoute,
    drug_name: str | None = None,
    generic_name: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    if route.metadata_filter:
        filters.append(route.metadata_filter)
    if drug_name:
        filters.append({"drug_name": drug_name})
    if generic_name:
        filters.append({"generic_name": generic_name})
    if category:
        filters.append({"category": category})
    return combine_metadata_filters(filters)


def combine_metadata_filters(filters: list[dict[str, Any]]) -> dict[str, Any]:
    non_empty = [item for item in filters if item]
    if not non_empty:
        return {}
    if len(non_empty) == 1:
        return non_empty[0]
    return {"$and": non_empty}


def _score_rule(question: str, rule: IntentRule) -> tuple[float, IntentRule, list[str]]:
    matched = [keyword for keyword in rule.keywords if keyword.lower() in question]
    if not matched:
        return 0.0, rule, []
    keyword_score = sum(1.0 + min(len(keyword), 6) * 0.1 for keyword in matched)
    return keyword_score * rule.priority, rule, matched


def _section_filter(sections: list[str]) -> dict[str, Any]:
    if not sections:
        return {}
    if len(sections) == 1:
        return {"section": sections[0]}
    return {"section": {"$in": sections}}


def _special_population_sections(matched_keywords: list[str]) -> list[str]:
    sections: list[str] = []
    keyword_text = " ".join(matched_keywords)
    if any(keyword in keyword_text for keyword in ["孕妇", "妊娠", "怀孕", "备孕"]):
        sections.extend(["孕妇及哺乳期妇女用药", "禁忌", "注意事项"])
    if any(keyword in keyword_text for keyword in ["哺乳", "母乳", "喂奶"]):
        sections.extend(["孕妇及哺乳期妇女用药", "注意事项"])
    if any(keyword in keyword_text for keyword in ["儿童", "小孩", "孩子", "婴儿", "宝宝", "未成年"]):
        sections.extend(["儿童用药", "注意事项"])
    if any(keyword in keyword_text for keyword in ["老人", "老年", "高龄"]):
        sections.extend(["老年用药", "老年患者用药", "注意事项"])
    if any(keyword in keyword_text for keyword in ["肝功能", "肾功能", "肝损伤", "肾损伤"]):
        sections.extend(["注意事项", "禁忌"])
    return list(dict.fromkeys(sections)) or ["孕妇及哺乳期妇女用药", "儿童用药", "老年用药", "老年患者用药", "注意事项"]
