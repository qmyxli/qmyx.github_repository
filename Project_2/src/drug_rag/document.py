from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceSpan:
    source_file: str
    source_path: str
    page_start: int
    page_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DrugDocument:
    doc_id: str
    source_file: str
    source_path: str
    pages: int
    drug_name: str = "unknown"
    generic_name: str = "unknown"
    approval_number: str = "unknown"
    category: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("extra", None)
        data.update(self.extra)
        return data


@dataclass(slots=True)
class DrugChunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str
    source: SourceSpan
    drug_name: str = "unknown"
    generic_name: str = "unknown"
    category: str = "unknown"
    section_index: int = 0
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "section": self.section,
            "source": self.source.to_dict(),
            "drug_name": self.drug_name,
            "generic_name": self.generic_name,
            "category": self.category,
            "section_index": self.section_index,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }
        return payload

    def flat_metadata(self) -> dict[str, Any]:
        metadata = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section": self.section,
            "drug_name": self.drug_name,
            "generic_name": self.generic_name,
            "category": self.category,
            "section_index": self.section_index,
            "chunk_index": self.chunk_index,
            **self.source.to_dict(),
            **self.metadata,
        }
        return {k: v for k, v in metadata.items() if v is not None}
