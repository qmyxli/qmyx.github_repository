from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from drug_rag.schemas import DrugChunk, SourceSpan


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(chunks: Iterable[DrugChunk], path: str | Path) -> int:
    path = ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[DrugChunk]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            source = SourceSpan(**payload["source"])
            yield DrugChunk(
                chunk_id=payload["chunk_id"],
                doc_id=payload["doc_id"],
                text=payload["text"],
                section=payload["section"],
                source=source,
                drug_name=payload.get("drug_name", "unknown"),
                generic_name=payload.get("generic_name", "unknown"),
                category=payload.get("category", "unknown"),
                section_index=payload.get("section_index", 0),
                chunk_index=payload.get("chunk_index", 0),
                metadata=payload.get("metadata", {}),
            )
