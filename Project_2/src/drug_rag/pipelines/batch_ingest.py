from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from drug_rag.io_utils import ensure_parent
from drug_rag.parsers import parse_drug_pdf
from drug_rag.pipelines.manifest import iter_manifest_documents, load_batch_manifest, resolve_category


@dataclass(slots=True)
class BatchDocumentReport:
    path: str
    category: str
    doc_id: str = ""
    drug_name: str = ""
    generic_name: str = ""
    chunk_count: int = 0
    status: str = "ok"
    error: str = ""


@dataclass(slots=True)
class BatchParseReport:
    doc_count: int = 0
    chunk_count: int = 0
    failed_count: int = 0
    out_jsonl: str = ""
    documents: list[BatchDocumentReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_count": self.doc_count,
            "chunk_count": self.chunk_count,
            "failed_count": self.failed_count,
            "out_jsonl": self.out_jsonl,
            "documents": [asdict(item) for item in self.documents],
        }


def parse_pdf_batch(
    input_dir: str | Path | None,
    out_jsonl: str | Path,
    category: str = "unknown",
    chunk_size: int = 900,
    chunk_overlap: int = 120,
    recursive: bool = True,
    manifest_path: str | Path | None = None,
    category_from_dir: bool = False,
    fail_fast: bool = False,
) -> BatchParseReport:
    manifest = load_batch_manifest(manifest_path)
    pdfs = _collect_pdfs(input_dir=input_dir, recursive=recursive, manifest_path=manifest_path)
    out_path = ensure_parent(out_jsonl)
    report = BatchParseReport(out_jsonl=str(out_path))

    with out_path.open("w", encoding="utf-8") as f:
        for pdf in pdfs:
            resolved_category = resolve_category(
                pdf,
                manifest=manifest,
                default_category=category,
                category_from_dir=category_from_dir,
            )
            item = BatchDocumentReport(path=str(pdf), category=resolved_category)
            try:
                document, chunks = parse_drug_pdf(
                    pdf,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    category=resolved_category,
                )
                item.doc_id = document.doc_id
                item.drug_name = document.drug_name
                item.generic_name = document.generic_name
                item.chunk_count = len(chunks)
                if not chunks:
                    item.status = "empty"
                else:
                    report.doc_count += 1
                    report.chunk_count += len(chunks)
                    for chunk in chunks:
                        f.write(json.dumps(chunk.to_json(), ensure_ascii=False) + "\n")
            except Exception as exc:
                item.status = "failed"
                item.error = f"{type(exc).__name__}: {exc}"
                report.failed_count += 1
                if fail_fast:
                    report.documents.append(item)
                    raise
            report.documents.append(item)
    return report


def parse_pdf_directory(
    input_dir: str | Path,
    out_jsonl: str | Path,
    category: str = "unknown",
    chunk_size: int = 900,
    chunk_overlap: int = 120,
    recursive: bool = True,
) -> tuple[int, int]:
    report = parse_pdf_batch(
        input_dir=input_dir,
        out_jsonl=out_jsonl,
        category=category,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        recursive=recursive,
    )
    return report.doc_count, report.chunk_count


def write_batch_report(report: BatchParseReport, out_path: str | Path) -> None:
    path = ensure_parent(out_path)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_pdfs(
    input_dir: str | Path | None,
    recursive: bool,
    manifest_path: str | Path | None,
) -> list[Path]:
    manifest = load_batch_manifest(manifest_path)
    manifest_docs = [doc.path for doc in iter_manifest_documents(manifest)]
    if manifest_docs:
        return sorted(manifest_docs)
    if not input_dir:
        raise ValueError("input_dir is required when manifest does not list documents")
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(input_path.glob(pattern))
