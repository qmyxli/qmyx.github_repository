from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from drug_rag.loaders import PdfPage, load_pdf_pages
from drug_rag.parsers.section_normalizer import normalize_body_text, normalize_section_name
from drug_rag.schemas import DrugChunk, DrugDocument, SourceSpan

SECTION_PATTERN = re.compile(r"【([^】]{1,40})】")


@dataclass(slots=True)
class ParsedSection:
    name: str
    text: str
    page_start: int
    page_end: int
    index: int


def parse_drug_pdf(
    pdf_path: str | Path,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
    category: str = "unknown",
) -> tuple[DrugDocument, list[DrugChunk]]:
    """把单个药品说明书 PDF 解析成文档元信息和可入库 chunk。

    解析策略是“章节优先，章节内滑窗”：先识别【用法用量】等说明书章节，
    再在章节内部按 chunk_size/chunk_overlap 切分。这样后续可以用 section
    metadata 精准过滤，并保留页码来源。
    """
    pages = load_pdf_pages(pdf_path)
    path = Path(pdf_path)
    doc_id = _make_doc_id(path, pages)
    sections = _extract_sections(pages)
    drug_name, generic_name = _extract_drug_names(sections, path)
    approval_number = _extract_field(sections, "批准文号") or "unknown"

    document = DrugDocument(
        doc_id=doc_id,
        source_file=path.name,
        source_path=str(path),
        pages=len(pages),
        drug_name=drug_name,
        generic_name=generic_name,
        approval_number=approval_number,
        category=category,
    )

    chunks: list[DrugChunk] = []
    for section in sections:
        text_chunks = _split_text(section.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_id = f"{doc_id}:{section.index:03d}:{chunk_index:03d}"
            chunks.append(
                DrugChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_text,
                    section=section.name,
                    source=SourceSpan(
                        source_file=path.name,
                        source_path=str(path),
                        page_start=section.page_start,
                        page_end=section.page_end,
                    ),
                    drug_name=drug_name,
                    generic_name=generic_name,
                    category=category,
                    section_index=section.index,
                    chunk_index=chunk_index,
                    metadata={
                        "approval_number": approval_number,
                        "pages": len(pages),
                    },
                )
            )
    return document, chunks


def _extract_sections(pages: list[PdfPage]) -> list[ParsedSection]:
    """从整份 PDF 文本中识别【章节名】，并记录每个章节的页码范围。"""
    page_offsets: list[tuple[int, int, int]] = []
    full_text_parts: list[str] = []
    cursor = 0
    for page in pages:
        text = normalize_body_text(page.text)
        start = cursor
        full_text_parts.append(text)
        cursor += len(text)
        page_offsets.append((page.page_number, start, cursor))
        full_text_parts.append("\n")
        cursor += 1

    full_text = "\n".join(part for part in full_text_parts if part)
    matches = list(SECTION_PATTERN.finditer(full_text))
    if not matches:
        return [
            ParsedSection(
                name="全文",
                text=normalize_body_text(full_text),
                page_start=1,
                page_end=pages[-1].page_number if pages else 1,
                index=0,
            )
        ]

    sections: list[ParsedSection] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(full_text)
        name = normalize_section_name(match.group(1))
        body = normalize_body_text(full_text[start:end])
        if not body:
            continue
        page_start = _page_for_offset(page_offsets, match.start())
        page_end = _page_for_offset(page_offsets, max(start, end - 1))
        sections.append(
            ParsedSection(
                name=name,
                text=body,
                page_start=page_start,
                page_end=page_end,
                index=len(sections),
            )
        )
    return sections


def _page_for_offset(page_offsets: list[tuple[int, int, int]], offset: int) -> int:
    for page_number, start, end in page_offsets:
        if start <= offset <= end:
            return page_number
    return page_offsets[-1][0] if page_offsets else 1


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """在单个章节内部切 chunk，优先在换行、句号、分号附近断开。"""
    text = normalize_body_text(text)
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        window = text[start:end]
        if end < len(text):
            cut = max(window.rfind("\n"), window.rfind("。"), window.rfind("；"))
            if cut > chunk_size * 0.5:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _extract_drug_names(sections: list[ParsedSection], path: Path) -> tuple[str, str]:
    """优先从“药品名称”章节提取商品名和通用名，失败时从文件名猜测。"""
    name_text = _extract_field(sections, "药品名称") or ""
    generic = _search_name(name_text, ["通用名称", "通 用 名 称"])
    trade = _search_name(name_text, ["商品名称", "商 品 名 称"])
    if not generic:
        generic = _guess_generic_name_from_filename(path.name)
    drug_name = trade or generic or path.stem
    return drug_name or "unknown", generic or "unknown"


def _extract_field(sections: list[ParsedSection], section_name: str) -> str | None:
    for section in sections:
        if section.name == section_name:
            return section.text.strip()
    return None


def _search_name(text: str, labels: list[str]) -> str:
    for label in labels:
        compact_label = re.sub(r"\s+", "", label)
        flexible_label = r"\s*".join(re.escape(char) for char in compact_label)
        pattern = re.compile(rf"{flexible_label}\s*[:：]\s*([^\n]+)")
        match = pattern.search(text)
        if match:
            return match.group(1).strip(" ；;。")
    return ""


def _guess_generic_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    for sep in ["__", "_", "-", " "]:
        parts = [p for p in stem.split(sep) if p]
        for part in parts:
            if part.endswith("片") or part.endswith("胶囊") or part.endswith("颗粒"):
                return part
    return ""


def _make_doc_id(path: Path, pages: list[PdfPage]) -> str:
    """用文件原始字节 MD5 作为 doc_id，用于重复上传识别和版本更新。"""
    return hashlib.md5(path.read_bytes()).hexdigest()
