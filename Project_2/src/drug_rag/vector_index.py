from __future__ import annotations

from pathlib import Path

from drug_rag.embeddings import build_embeddings
from drug_rag.io_utils import read_jsonl


def _chroma_class():
    try:
        from langchain_chroma import Chroma

        return Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma

            return Chroma
        except ImportError as exc:
            raise RuntimeError(
                "Missing Chroma dependencies. Install with `pip install -r requirements.txt`."
            ) from exc


def build_vector_index(
    jsonl_path: str | Path,
    persist_dir: str | Path = "data/chroma",
    collection: str = "drug_chunks",
    embedding_provider: str = "hash",
    embedding_model: str | None = None,
    dimensions: int | None = 512,
    embedding_base_url: str | None = None,
    embedding_api_key_env: str = "DASHSCOPE_API_KEY",
) -> int:
    """把 JSONL 中的原文 chunk 写入 Chroma 原文 collection。

    每个 chunk 的 text 会被 embedding，flat_metadata 会作为 Chroma metadata，
    chunk_id 作为向量 ID。写入前会按相同 ID 删除旧记录，支持重复构建和更新。
    """
    chunks = list(read_jsonl(jsonl_path))
    if not chunks:
        raise ValueError(f"No chunks found in {jsonl_path}")

    embeddings = build_embeddings(
        provider=embedding_provider,
        model=embedding_model,
        dimensions=dimensions,
        base_url=embedding_base_url,
        api_key_env=embedding_api_key_env,
    )
    Chroma = _chroma_class()
    texts = [chunk.text for chunk in chunks]
    metadatas = [chunk.flat_metadata() for chunk in chunks]
    ids = [chunk.chunk_id for chunk in chunks]

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=collection,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    _delete_existing(vector_store, ids)
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    if hasattr(vector_store, "persist"):
        vector_store.persist()
    return len(chunks)


def load_vector_store(
    persist_dir: str | Path = "data/chroma",
    collection: str = "drug_chunks",
    embedding_provider: str = "hash",
    embedding_model: str | None = None,
    dimensions: int | None = 512,
    embedding_base_url: str | None = None,
    embedding_api_key_env: str = "DASHSCOPE_API_KEY",
):
    """按配置加载一个 Chroma collection，供检索或删除旧向量使用。"""
    embeddings = build_embeddings(
        provider=embedding_provider,
        model=embedding_model,
        dimensions=dimensions,
        base_url=embedding_base_url,
        api_key_env=embedding_api_key_env,
    )
    Chroma = _chroma_class()
    return Chroma(
        collection_name=collection,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def _delete_existing(vector_store, ids: list[str]) -> None:
    """写入前删除同 ID 文档；Chroma 中不存在这些 ID 时直接忽略。"""
    try:
        vector_store.delete(ids=ids)
    except Exception:
        pass
