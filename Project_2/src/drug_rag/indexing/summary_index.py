from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from drug_rag.embeddings import build_embeddings
from drug_rag.indexing.vector_index import _chroma_class
from drug_rag.io_utils import read_jsonl


def build_summary_index(
    jsonl_path: str | Path,
    persist_dir: str | Path = "data/chroma",
    collection: str = "drug_summaries",
    embedding_provider: str = "hash",
    embedding_model: str | None = None,
    dimensions: int | None = 512,
    embedding_base_url: str | None = None,
    embedding_api_key_env: str = "DASHSCOPE_API_KEY",
) -> int:
    """构建章节摘要索引。

    原文 chunk 按 doc_id + section 聚合，每个药品章节生成一条摘要向量。
    摘要索引用来“先定位相关章节”，最终回答仍优先回到原文 chunk 取证。
    """
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for chunk in read_jsonl(jsonl_path):
        grouped[(chunk.doc_id, chunk.section)].append(chunk)

    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    for (doc_id, section), chunks in grouped.items():
        combined = "\n".join(chunk.text for chunk in chunks)
        summary = _extractive_summary(combined)
        first = chunks[0]
        texts.append(f"{first.drug_name} {first.generic_name} {section}\n{summary}")
        metadatas.append(
            {
                "doc_id": doc_id,
                "section": section,
                "drug_name": first.drug_name,
                "generic_name": first.generic_name,
                "category": first.category,
                "source_file": first.source.source_file,
                "summary_type": "section",
            }
        )
        ids.append(f"{doc_id}:summary:{section}")

    if not texts:
        return 0

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    embeddings = build_embeddings(
        embedding_provider,
        embedding_model,
        dimensions,
        base_url=embedding_base_url,
        api_key_env=embedding_api_key_env,
    )
    Chroma = _chroma_class()
    vector_store = Chroma(
        collection_name=collection,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    try:
        vector_store.delete(ids=ids)
    except Exception:
        pass
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    if hasattr(vector_store, "persist"):
        vector_store.persist()
    return len(texts)


def _extractive_summary(text: str, max_chars: int = 420) -> str:
    """轻量抽取式摘要：截取章节前部内容，并尽量在句号/分号处结束。"""
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) <= max_chars:
        return text
    stop = max(text.rfind("。", 0, max_chars), text.rfind("；", 0, max_chars))
    if stop > max_chars * 0.5:
        return text[: stop + 1]
    return text[:max_chars].rstrip() + "..."
