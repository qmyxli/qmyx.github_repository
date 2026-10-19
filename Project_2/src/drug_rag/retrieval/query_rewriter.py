from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from drug_rag.retrieval.intent_router import QueryRoute, route_question


INTENT_ANCHORS = {
    "dosage": ["成人", "给药频次", "剂量调整"],
    "indication": ["治疗疾病", "适应症"],
    "contraindication": ["禁用", "慎用", "风险人群"],
    "adverse_reaction": ["常见不良反应", "严重不良反应"],
    "interaction": ["合并用药", "相互作用", "降压药"],
    "special_population": ["特殊人群", "孕妇", "儿童", "老年患者"],
    "storage": ["保存条件", "贮藏", "常温", "冷藏", "冷冻", "温度", "有效期"],
    "purchase_guidance": ["购药指导", "适应症", "禁忌", "用法用量"],
    "general_qa": [],
}


@dataclass(slots=True)
class QueryRewrite:
    original_question: str
    rewritten_question: str
    route: QueryRoute
    anchors: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["route"] = self.route.to_dict()
        return data


def build_query_rewrite(
    question: str,
    drug_name: str | None = None,
    generic_name: str | None = None,
) -> QueryRewrite:
    route = route_question(question)
    anchors = []
    if drug_name and drug_name != "unknown":
        anchors.append(drug_name)
    if generic_name and generic_name != "unknown":
        anchors.append(generic_name)
    anchors.extend(route.sections)
    if route.intent == "special_population":
        anchors.extend(_special_population_anchors(route.sections))
    else:
        anchors.extend(INTENT_ANCHORS.get(route.intent, []))
    anchors.extend(route.matched_keywords)
    anchors.append(question)
    unique_anchors = list(dict.fromkeys(anchor for anchor in anchors if anchor))
    return QueryRewrite(
        original_question=question,
        rewritten_question=" ".join(unique_anchors),
        route=route,
        anchors=unique_anchors,
    )


def rewrite_question(question: str, drug_name: str | None = None, generic_name: str | None = None) -> str:
    return build_query_rewrite(question, drug_name=drug_name, generic_name=generic_name).rewritten_question


def _special_population_anchors(sections: list[str]) -> list[str]:
    anchors = ["特殊人群"]
    if "孕妇及哺乳期妇女用药" in sections:
        anchors.extend(["孕妇", "妊娠", "哺乳"])
    if "儿童用药" in sections:
        anchors.extend(["儿童", "儿童安全性"])
    if "老年用药" in sections or "老年患者用药" in sections:
        anchors.extend(["老年患者", "老年用药"])
    if sections == ["注意事项", "禁忌"] or "禁忌" in sections:
        anchors.extend(["慎用", "禁忌"])
    return anchors
