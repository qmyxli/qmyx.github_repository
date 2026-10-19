from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drug_rag.io_utils import ensure_parent, read_jsonl


@dataclass(slots=True)
class CatalogResolution:
    drug_name: str | None = None
    generic_name: str | None = None
    category: str | None = None
    matched_drugs: list[dict[str, Any]] | None = None
    matched_categories: list[str] | None = None
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_metadata_catalog(jsonl_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    """从 master JSONL 聚合出药品目录 catalog。

    catalog 不是向量索引，而是结构化目录：记录药品名、通用名、分类、
    来源文件、章节列表和页码。问答前会用它识别用户问题中的药品/分类约束。
    """
    grouped: dict[str, list] = defaultdict(list)
    for chunk in read_jsonl(jsonl_path):
        grouped[chunk.doc_id].append(chunk)

    drugs = []
    for doc_id, chunks in grouped.items():
        first = chunks[0]
        sections = sorted({chunk.section for chunk in chunks})
        section_pages: dict[str, list[int]] = defaultdict(list)
        for chunk in chunks:
            section_pages[chunk.section].extend(range(chunk.source.page_start, chunk.source.page_end + 1))
        drugs.append(
            {
                "doc_id": doc_id,
                "drug_name": first.drug_name,
                "generic_name": first.generic_name,
                "category": first.category,
                "source_file": first.source.source_file,
                "source_path": first.source.source_path,
                "sections": sections,
                "section_pages": {
                    section: sorted(set(pages)) for section, pages in section_pages.items()
                },
                "chunk_count": len(chunks),
            }
        )

    categories = sorted({item["category"] for item in drugs if item.get("category")})
    category_counts = {
        category: sum(1 for item in drugs if item.get("category") == category)
        for category in categories
    }
    catalog = {
        "drug_count": len(drugs),
        "categories": categories,
        "category_counts": category_counts,
        "drugs": sorted(drugs, key=lambda item: (item["category"], item["generic_name"], item["drug_name"])),
    }
    path = ensure_parent(out_path)
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


def load_metadata_catalog(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_drugs(
    catalog: dict[str, Any],
    drug_name: str | None = None,
    generic_name: str | None = None,
    category: str | None = None,
    section: str | None = None,
) -> list[dict[str, Any]]:
    """按商品名、通用名、分类或章节在 catalog 中查药品。"""
    results = []
    for item in catalog.get("drugs", []):
        if drug_name and drug_name not in item.get("drug_name", ""):
            continue
        if generic_name and generic_name not in item.get("generic_name", ""):
            continue
        if category and category != item.get("category"):
            continue
        if section and section not in item.get("sections", []):
            continue
        results.append(item)
    return results


def resolve_query_constraints(catalog: dict[str, Any], question: str) -> CatalogResolution:
    """从用户问题中解析药品约束。

    如果问题唯一命中某个药品，则返回 drug_name/generic_name/category；
    如果只命中一个分类，则返回 category；如果命中多个药品，则标记 ambiguous，
    避免系统误把多个药品混成一个答案。
    """
    normalized_question = _normalize(question)
    matched_drugs = []
    for item in catalog.get("drugs", []):
        names = _drug_match_names(item)
        if any(name and name in normalized_question for name in names):
            matched_drugs.append(item)

    matched_categories = [
        category
        for category in catalog.get("categories", [])
        if category and _normalize(category) in normalized_question
    ]

    unique_doc_ids = {item.get("doc_id") for item in matched_drugs}
    if len(unique_doc_ids) == 1 and matched_drugs:
        item = matched_drugs[0]
        return CatalogResolution(
            drug_name=item.get("drug_name"),
            generic_name=item.get("generic_name"),
            category=item.get("category"),
            matched_drugs=matched_drugs,
            matched_categories=matched_categories,
            ambiguous=False,
        )

    if not matched_drugs and len(matched_categories) == 1:
        return CatalogResolution(
            category=matched_categories[0],
            matched_drugs=[],
            matched_categories=matched_categories,
            ambiguous=False,
        )

    return CatalogResolution(
        matched_drugs=matched_drugs,
        matched_categories=matched_categories,
        ambiguous=len(unique_doc_ids) > 1 or len(matched_categories) > 1,
    )


def _drug_match_names(item: dict[str, Any]) -> list[str]:
    names = [
        item.get("drug_name", ""),
        item.get("generic_name", ""),
        Path(item.get("source_file", "")).stem,
    ]
    return [_normalize(name) for name in names if name and name != "unknown"]


def _normalize(text: str) -> str:
    return (
        str(text)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("®", "")
        .replace("（", "(")
        .replace("）", ")")
    )
