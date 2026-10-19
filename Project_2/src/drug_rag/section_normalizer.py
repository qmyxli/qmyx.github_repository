from __future__ import annotations

import re


SECTION_ALIASES = {
    "药品名称": "药品名称",
    "成份": "成份",
    "成分": "成份",
    "性状": "性状",
    "适应症": "适应症",
    "规格": "规格",
    "用法用量": "用法用量",
    "不良反应": "不良反应",
    "禁忌": "禁忌",
    "注意事项": "注意事项",
    "孕妇及哺乳期妇女用药": "孕妇及哺乳期妇女用药",
    "儿童用药": "儿童用药",
    "老年用药": "老年用药",
    "老年患者用药": "老年患者用药",
    "药物相互作用": "药物相互作用",
    "药物过量": "药物过量",
    "临床药理": "临床药理",
    "临床试验": "临床试验",
    "药理毒理": "药理毒理",
    "药代动力学": "药代动力学",
    "贮藏": "贮藏",
    "包装": "包装",
    "有效期": "有效期",
    "执行标准": "执行标准",
    "批准文号": "批准文号",
    "上市许可持有人": "上市许可持有人",
    "生产企业": "生产企业",
    "境内联系人": "境内联系人",
}


def compact_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_section_name(raw: str) -> str:
    name = raw.replace("【", "").replace("】", "")
    name = re.sub(r"\s+", "", name)
    return SECTION_ALIASES.get(name, name)


def normalize_body_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\S)-\n(\S)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return compact_text(text)
