from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SafetyFinding:
    label: str
    level: str
    keywords: list[str]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SafetyAssessment:
    risk_level: str
    findings: list[SafetyFinding]
    must_consult: bool
    emergency: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "must_consult": self.must_consult,
            "emergency": self.emergency,
            "warnings": self.warnings,
            "findings": [finding.to_dict() for finding in self.findings],
        }


RISK_RULES = [
    {
        "label": "pregnancy",
        "level": "high",
        "keywords": ["孕妇", "妊娠", "怀孕", "备孕", "有妊娠可能"],
        "message": "涉及孕妇、备孕或可能妊娠人群，必须严格核对说明书禁忌和特殊人群用药信息。",
    },
    {
        "label": "lactation",
        "level": "high",
        "keywords": ["哺乳", "母乳", "喂奶"],
        "message": "涉及哺乳期用药，应先咨询医生或执业药师，不要自行判断。",
    },
    {
        "label": "child",
        "level": "high",
        "keywords": ["儿童", "小孩", "孩子", "婴儿", "宝宝", "未成年"],
        "message": "涉及儿童用药，需确认儿童安全性、剂量和适用年龄。",
    },
    {
        "label": "elderly",
        "level": "medium",
        "keywords": ["老人", "老年", "高龄"],
        "message": "涉及老年患者，应结合血压、肾功能、合并疾病和正在使用的药物谨慎评估。",
    },
    {
        "label": "renal_hepatic",
        "level": "high",
        "keywords": ["肝功能", "肾功能", "肝损伤", "肾损伤", "肾衰", "肝病", "肾病"],
        "message": "涉及肝肾功能异常，药物清除和不良反应风险可能改变，应咨询医生或药师。",
    },
    {
        "label": "allergy",
        "level": "high",
        "keywords": ["过敏", "皮疹", "荨麻疹", "呼吸困难"],
        "message": "涉及过敏或疑似严重过敏反应，应停止自行用药并及时就医。",
    },
    {
        "label": "interaction",
        "level": "medium",
        "keywords": ["一起吃", "同服", "合用", "联用", "相互作用", "搭配", "其他药"],
        "message": "涉及合并用药，应核对药物相互作用，避免自行叠加用药。",
    },
    {
        "label": "adverse_reaction",
        "level": "medium",
        "keywords": ["副作用", "不良反应", "头晕", "恶心", "不舒服", "低血压"],
        "message": "涉及不良反应，应观察严重程度，必要时停药并咨询医生或药师。",
    },
    {
        "label": "overdose",
        "level": "high",
        "keywords": ["吃多了", "过量", "多吃", "误服", "一次吃了很多"],
        "message": "涉及过量或误服，应尽快联系医生、药师或急救机构。",
    },
    {
        "label": "contraindication",
        "level": "high",
        "keywords": ["禁忌", "不能吃", "禁用", "不适合"],
        "message": "涉及禁忌或不适用人群，应以说明书禁忌为准，不要自行用药。",
    },
]

CONTEXT_HIGH_RISK_KEYWORDS = ["禁用", "禁止", "尽快停止", "胎儿", "新生儿死亡", "肾衰", "高钾血症", "休克"]
EMERGENCY_KEYWORDS = ["呼吸困难", "休克", "昏迷", "严重过敏", "胸痛", "大量误服"]
LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}


def assess_safety(question: str, context: str = "") -> SafetyAssessment:
    findings = [_finding_from_rule(rule, question) for rule in RISK_RULES if _matched_keywords(rule, question)]
    if any(keyword in context for keyword in CONTEXT_HIGH_RISK_KEYWORDS):
        findings.append(
            SafetyFinding(
                label="source_high_risk",
                level="high",
                keywords=[keyword for keyword in CONTEXT_HIGH_RISK_KEYWORDS if keyword in context],
                message="检索到的说明书片段包含禁用、停用或严重风险表述，回答必须突出风险。",
            )
        )

    emergency = any(keyword in question for keyword in EMERGENCY_KEYWORDS)
    risk_level = _max_level([finding.level for finding in findings] + (["high"] if emergency else []))
    must_consult = risk_level in {"medium", "high"} or emergency
    warnings = _build_warnings(risk_level, findings, emergency)
    return SafetyAssessment(
        risk_level=risk_level,
        findings=findings,
        must_consult=must_consult,
        emergency=emergency,
        warnings=warnings,
    )


def safety_notice(question: str, context: str = "") -> str:
    assessment = assess_safety(question, context)
    return "\n".join(assessment.warnings)


def _finding_from_rule(rule: dict[str, Any], text: str) -> SafetyFinding:
    return SafetyFinding(
        label=rule["label"],
        level=rule["level"],
        keywords=_matched_keywords(rule, text),
        message=rule["message"],
    )


def _matched_keywords(rule: dict[str, Any], text: str) -> list[str]:
    return [keyword for keyword in rule["keywords"] if keyword in text]


def _max_level(levels: list[str]) -> str:
    if not levels:
        return "low"
    return max(levels, key=lambda level: LEVEL_ORDER[level])


def _build_warnings(risk_level: str, findings: list[SafetyFinding], emergency: bool) -> list[str]:
    warnings = []
    if emergency:
        warnings.append("如出现呼吸困难、休克、昏迷、严重过敏或大量误服等紧急情况，请立即就医或联系急救。")
    if risk_level == "high":
        warnings.append("该问题涉及高风险用药场景，请不要仅凭自动回答自行用药或购药，务必咨询医生或执业药师。")
    elif risk_level == "medium":
        warnings.append("该问题涉及需要谨慎判断的用药场景，建议在医生或执业药师指导下使用。")
    else:
        warnings.append("请按说明书、处方和医生或药师指导使用；如症状未改善或出现不适，应及时咨询专业人员。")

    for finding in findings:
        if finding.message not in warnings:
            warnings.append(finding.message)
    return warnings
