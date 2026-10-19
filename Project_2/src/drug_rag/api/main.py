from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from drug_rag.chains import answer_question_with_trace
from drug_rag.config import deep_get, load_yaml
from drug_rag.indexing import build_vector_index, load_vector_store
from drug_rag.indexing.metadata_index import (
    build_metadata_catalog,
    load_metadata_catalog,
    resolve_query_constraints,
)
from drug_rag.indexing.summary_index import build_summary_index
from drug_rag.io_utils import ensure_parent, read_jsonl, write_jsonl
from drug_rag.parsers import parse_drug_pdf
from drug_rag.retrieval import retrieve_advanced
from drug_rag.retrieval.advanced_retriever import RERANK_STRATEGIES, RETRIEVAL_MODES
from drug_rag.schemas import DrugChunk, DrugDocument, SourceSpan

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = Path(os.getenv("DRUG_RAG_WEB_DIR", PROJECT_ROOT / "web"))
UPLOAD_DIR = Path(os.getenv("DRUG_RAG_UPLOAD_DIR", PROJECT_ROOT / "data" / "uploads"))
UPLOAD_JSONL_DIR = Path(os.getenv("DRUG_RAG_UPLOAD_JSONL_DIR", PROJECT_ROOT / "data" / "processed" / "uploads"))
MASTER_JSONL = Path(os.getenv("DRUG_RAG_MASTER_JSONL", PROJECT_ROOT / "data" / "processed" / "all_drug_chunks.jsonl"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


class RagRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    mode: str = Field("auto")
    rerank: str = Field("auto")
    top_k: int = Field(6, ge=1, le=20)
    candidate_k: int | None = Field(None, ge=1, le=50)
    drug_name: str | None = None
    generic_name: str | None = None
    category: str | None = None
    use_catalog: bool = True


class UploadResponse(BaseModel):
    filename: str
    saved_path: str
    doc_id: str
    drug_name: str
    generic_name: str
    category: str
    chunk_count: int
    summary_count: int
    catalog_drug_count: int
    supported_extensions: list[str]
    action: str = "indexed"
    message: str = ""


class AppSettings(BaseModel):
    config_path: str = "configs/app.yaml"
    persist_dir: str = "data/chroma"
    collection: str = "drug_chunks"
    summary_collection: str = "drug_summaries"
    catalog_path: str = "data/processed/drug_catalog.json"


def create_app() -> FastAPI:
    """创建 FastAPI 应用，并把前端页面、问答接口和上传入库接口挂载起来。"""
    app = FastAPI(title="Drug Instruction RAG", version="0.1.0")

    if WEB_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")

    @app.get("/")
    def index():
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend index.html not found.")
        return FileResponse(index_path)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        settings = get_settings()
        catalog_exists = Path(settings.catalog_path).exists()
        return {
            "status": "ok",
            "persist_dir": settings.persist_dir,
            "collection": settings.collection,
            "summary_collection": settings.summary_collection,
            "catalog_path": settings.catalog_path,
            "catalog_exists": catalog_exists,
            "upload_limit_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "supported_upload_extensions": sorted(SUPPORTED_UPLOAD_EXTENSIONS),
        }

    @app.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        return get_catalog()

    @app.get("/api/catalog/resolve")
    def resolve_catalog(query: str = Query(..., min_length=1)) -> dict[str, Any]:
        catalog_data = get_catalog()
        return resolve_query_constraints(catalog_data, query).to_dict()

    @app.post("/api/retrieve")
    def retrieve(request: RagRequest) -> dict[str, Any]:
        """只执行检索链路并返回 Trace，用于观察路由、改写、filter 和召回结果。"""
        # 校验前端传来的参数是否合法，比如
        # top_k、检索模式、rerank
        # 选项等，避免非法配置进入后面的检索流程。
        _validate_options(request)
        # 解析用户问题里的药品约束和分类约束。
        # 例如用户问“坎地沙坦酯片怎么吃”，这里会从
        # drug_catalog.json
        # 里识别出
        # generic_name = 坎地沙坦酯片，后面检索时就能只查这个药品相关chunk。
        constraints = _resolve_constraints(request)
        # 获取运行时对象，包括：
        # Chroma
        # 向量库、摘要索引、LLM
        # 配置、embedding
        # 配置、prompt
        # 配置等。
        # 这是为了避免每次请求都重新加载数据库和模型配置。
        runtime = get_runtime()
        rerank_config = runtime["rerank_config"]

        trace = retrieve_advanced(
            vector_store=runtime["vector_store"],
            summary_store=runtime["summary_store"] if request.mode in {"auto", "summary", "hybrid"} else None,
            question=request.question,
            mode=request.mode,
            rerank=request.rerank,
            candidate_k=request.candidate_k,
            final_k=request.top_k,
            drug_name=constraints["drug_name"],
            generic_name=constraints["generic_name"],
            category=constraints["category"],
            rerank_model_provider=rerank_config.get("provider", "none"),
            rerank_model_name=rerank_config.get("model", "gte-rerank-v2"),
            rerank_model_endpoint=rerank_config.get("endpoint"),
            rerank_model_api_key_env=rerank_config.get("api_key_env", "DASHSCOPE_API_KEY"),
            rerank_model_timeout=int(rerank_config.get("timeout", 30)),
            rerank_model_strategies=rerank_config.get("use_for_strategies"),
            preserve_preferred_sections=bool(rerank_config.get("preserve_preferred_sections", True)),
        )
        return {
            "trace": trace.to_dict(),
            "catalog_resolution": constraints["catalog_resolution"],
        }

    @app.post("/api/ask")
    def ask(request: RagRequest) -> dict[str, Any]:
        """完整问答入口：先解析药品/分类约束，再检索，最后生成安全回答。"""
        _validate_options(request)
        constraints = _resolve_constraints(request)
        runtime = get_runtime()
        llm_evidence_config = runtime["llm_config"].get("evidence", {})
        rerank_config = runtime["rerank_config"]
        # 执行RAG问答：
        # 先做意图路由、问题重写、metadata
        # filter、摘要 / 混合检索、rerank；
        # 再根据召回证据做安全风险判断；最后用模板或受控LLM生成答案，并返回trace。
        result = answer_question_with_trace(
            question=request.question,
            vector_store=runtime["vector_store"],
            summary_store=runtime["summary_store"] if request.mode in {"auto", "summary", "hybrid"} else None,
            use_llm=runtime["llm_config"].get("provider", "none") not in {"none", "off", "false"},
            llm_provider=runtime["llm_config"].get("provider", "none"),
            llm_model=runtime["llm_config"].get("model"),
            llm_base_url=runtime["llm_config"].get("base_url"),
            llm_api_key_env=runtime["llm_config"].get("api_key_env", "DASHSCOPE_API_KEY"),
            llm_temperature=float(runtime["llm_config"].get("temperature", 0.1)),
            llm_evidence_max_chars_per_doc=int(llm_evidence_config.get("max_chars_per_doc", 1200)),
            llm_evidence_max_total_chars=int(llm_evidence_config.get("max_total_chars", 6000)),
            llm_evidence_max_chars_per_drug=int(llm_evidence_config.get("max_chars_per_drug", 1800)),
            llm_evidence_max_chars_per_drug_doc=int(llm_evidence_config.get("max_chars_per_drug_doc", 900)),
            prompts_path=runtime["prompts_path"],
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            mode=request.mode,
            rerank=request.rerank,
            rerank_model_provider=rerank_config.get("provider", "none"),
            rerank_model_name=rerank_config.get("model", "gte-rerank-v2"),
            rerank_model_endpoint=rerank_config.get("endpoint"),
            rerank_model_api_key_env=rerank_config.get("api_key_env", "DASHSCOPE_API_KEY"),
            rerank_model_timeout=int(rerank_config.get("timeout", 30)),
            rerank_model_strategies=rerank_config.get("use_for_strategies"),
            preserve_preferred_sections=bool(rerank_config.get("preserve_preferred_sections", True)),
            drug_name=constraints["drug_name"],
            generic_name=constraints["generic_name"],
            category=constraints["category"],
        )
        payload = result.to_dict()
        payload["catalog_resolution"] = constraints["catalog_resolution"]
        return payload

    @app.post("/api/upload")
    async def upload_document(
        file: UploadFile = File(...),
        category: str = Form("上传文档"),
        drug_name: str | None = Form(None),
        generic_name: str | None = Form(None),
    ) -> dict[str, Any]:
        """文件上传入口：保存文件、解析 chunk、写入 Chroma、更新摘要索引和 catalog。"""  #  ctrl+鼠标
        payload = await _process_upload(
            file=file,
            category=category,
            drug_name=drug_name,
            generic_name=generic_name,
        )
        return payload.model_dump()

    return app


def _validate_options(request: RagRequest) -> None:
    """校验前端传入的检索模式和 rerank 策略是否在白名单中。"""
    if request.mode not in RETRIEVAL_MODES:
        raise HTTPException(status_code=422, detail=f"Unsupported retrieval mode: {request.mode}")
    if request.rerank not in RERANK_STRATEGIES:
        raise HTTPException(status_code=422, detail=f"Unsupported rerank strategy: {request.rerank}")


def _resolve_constraints(request: RagRequest) -> dict[str, Any]:
    """用 catalog 从用户问题中识别药品名、通用名和分类，作为检索 metadata 约束。"""
    drug_name = request.drug_name
    generic_name = request.generic_name
    category = request.category
    resolution_payload = None
    if request.use_catalog and Path(get_settings().catalog_path).exists():
        resolution = resolve_query_constraints(get_catalog(), request.question)
        resolution_payload = resolution.to_dict()
        drug_name = drug_name or resolution.drug_name
        generic_name = generic_name or resolution.generic_name
        category = category or resolution.category
    return {
        "drug_name": drug_name,
        "generic_name": generic_name,
        "category": category,
        "catalog_resolution": resolution_payload,
    }


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """读取应用配置；环境变量优先，YAML 配置作为默认值。"""
    config_path = _project_path(os.getenv("DRUG_RAG_CONFIG", "configs/app.yaml"))
    config = load_yaml(config_path) if config_path.exists() else {}
    return AppSettings(
        config_path=str(config_path),
        persist_dir=str(_project_path(os.getenv("DRUG_RAG_CHROMA_DIR", deep_get(config, "paths.chroma_dir", "data/chroma")))),
        collection=os.getenv("DRUG_RAG_COLLECTION", deep_get(config, "index.chunk_collection", "drug_chunks")),
        summary_collection=os.getenv(
            "DRUG_RAG_SUMMARY_COLLECTION",
            deep_get(config, "index.summary_collection", "drug_summaries"),
        ),
        catalog_path=str(_project_path(os.getenv("DRUG_RAG_CATALOG", "data/processed/drug_catalog.json"))),
    )


def _project_path(path: str | Path) -> Path:
    """把配置中的相对路径解析到项目根目录。

    FastAPI 在不同启动方式下工作目录可能不同。如果直接使用 data/chroma 这类相对路径，
    可能会落到 src/drug_rag/api/data/chroma，导致前端仍读取旧 catalog/旧 Chroma。
    """
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_runtime() -> dict[str, Any]:
    """加载运行期对象，包括原文向量库、摘要向量库、LLM 和 prompt 配置。"""
    settings = get_settings()
    config = load_yaml(settings.config_path) if Path(settings.config_path).exists() else {}
    embedding_cfg = config.get("embedding", {})
    llm_cfg = config.get("llm", {})
    rerank_cfg = config.get("rerank", {})
    vector_store = load_vector_store(
        persist_dir=settings.persist_dir,
        collection=settings.collection,
        embedding_provider=embedding_cfg.get("provider", "hash"),
        embedding_model=embedding_cfg.get("model"),
        dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
        embedding_base_url=embedding_cfg.get("base_url"),
        embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
    )
    summary_store = load_vector_store(
        persist_dir=settings.persist_dir,
        collection=settings.summary_collection,
        embedding_provider=embedding_cfg.get("provider", "hash"),
        embedding_model=embedding_cfg.get("model"),
        dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
        embedding_base_url=embedding_cfg.get("base_url"),
        embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
    )
    prompts_path = deep_get(config, "prompts.path", "configs/prompts.yaml")
    return {
        "vector_store": vector_store,
        "summary_store": summary_store,
        "llm_config": llm_cfg,
        "rerank_config": rerank_cfg,
        "prompts_path": prompts_path,
    }

# least recently used 最近最少使用 maxsize 缓存中的个数
@lru_cache(maxsize=1)
def get_catalog() -> dict[str, Any]:
    """读取药品 catalog；如果还没有入库文件，则返回空目录。"""
    catalog_path = Path(get_settings().catalog_path)
    if not catalog_path.exists():
        return {"drug_count": 0, "categories": [], "category_counts": {}, "drugs": []}
    return load_metadata_catalog(catalog_path)


async def _process_upload(
    file: UploadFile,
    category: str,
    drug_name: str | None = None,
    generic_name: str | None = None,
) -> UploadResponse:
    """执行上传入库主流程。

    流程：
    1. 校验文件类型和大小。
    2. 用文件字节 MD5 生成 doc_id，判断是否重复上传。
    3. 若同名路径已有旧版本，先删除旧 JSONL 记录和 Chroma 向量。
    4. 解析文件为 DrugChunk，写入单文件 JSONL 和 master JSONL。
    5. 构建原文 chunk 索引、章节摘要索引，并重建 catalog。
    """
    original_name = file.filename or "uploaded-document"
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"暂不支持该文件类型，请上传：{supported}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件超过 10MB 限制。")

    incoming_doc_id = _content_md5(content)
    target_path = _upload_target_path(original_name)
    existing_chunks = _load_master_chunks(MASTER_JSONL)
    duplicate_chunks = [chunk for chunk in existing_chunks if chunk.doc_id == incoming_doc_id]
    if duplicate_chunks:
        # 相同字节内容已经入库，不再重复生成向量，避免 Chroma 中出现重复药品。
        catalog = build_metadata_catalog(MASTER_JSONL, get_settings().catalog_path) if MASTER_JSONL.exists() else get_catalog()
        first = duplicate_chunks[0]
        return UploadResponse(
            filename=original_name,
            saved_path=first.source.source_path,
            doc_id=incoming_doc_id,
            drug_name=first.drug_name,
            generic_name=first.generic_name,
            category=first.category,
            chunk_count=len(duplicate_chunks),
            summary_count=len({chunk.section for chunk in duplicate_chunks}),
            catalog_drug_count=int(catalog.get("drug_count", 0)),
            supported_extensions=sorted(SUPPORTED_UPLOAD_EXTENSIONS),
            action="skipped",
            message="相同文件内容已入库，本次上传已跳过。",
        )

    same_path_chunks = [
        chunk
        for chunk in existing_chunks
        if _same_upload_source(chunk, target_path)
    ]

    saved_path = _save_upload_bytes(original_name, content)
    document, chunks = _parse_uploaded_document(
        saved_path=saved_path,
        suffix=suffix,
        category=category.strip() or "上传文档",
        drug_name=drug_name.strip() if drug_name else None,
        generic_name=generic_name.strip() if generic_name else None,
    )
    if not chunks:
        raise HTTPException(status_code=422, detail="文件已上传，但没有解析出可入库文本。")

    if same_path_chunks:
        # 同名上传路径但内容不同，按“文件更新”处理：先删旧版本，再写新版本。
        _rewrite_master_without_chunks(MASTER_JSONL, same_path_chunks)
        _delete_chunks_from_chroma(same_path_chunks)

    upload_jsonl = UPLOAD_JSONL_DIR / f"{document.doc_id}.jsonl"
    write_jsonl(chunks, upload_jsonl)
    _append_chunks(chunks, MASTER_JSONL)

    settings = get_settings()
    config = load_yaml(settings.config_path) if Path(settings.config_path).exists() else {}
    embedding_cfg = config.get("embedding", {})
    chunk_count = build_vector_index(
        upload_jsonl,
        persist_dir=settings.persist_dir,
        collection=settings.collection,
        embedding_provider=embedding_cfg.get("provider", "hash"),
        embedding_model=embedding_cfg.get("model"),
        dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
        embedding_base_url=embedding_cfg.get("base_url"),
        embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
    )
    summary_count = build_summary_index(
        upload_jsonl,
        persist_dir=settings.persist_dir,
        collection=settings.summary_collection,
        embedding_provider=embedding_cfg.get("provider", "hash"),
        embedding_model=embedding_cfg.get("model"),
        dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
        embedding_base_url=embedding_cfg.get("base_url"),
        embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
    )
    catalog = build_metadata_catalog(MASTER_JSONL, settings.catalog_path)
    get_runtime.cache_clear()
    get_catalog.cache_clear()

    action = "replaced" if same_path_chunks else "indexed"
    message = "同名上传路径已存在且内容不同，已覆盖旧版本并更新索引。" if same_path_chunks else "文件已解析并入库。"
    return UploadResponse(
        filename=original_name,
        saved_path=str(saved_path),
        doc_id=document.doc_id,
        drug_name=document.drug_name,
        generic_name=document.generic_name,
        category=document.category,
        chunk_count=chunk_count,
        summary_count=summary_count,
        catalog_drug_count=int(catalog.get("drug_count", 0)),
        supported_extensions=sorted(SUPPORTED_UPLOAD_EXTENSIONS),
        action=action,
        message=message,
    )


def _save_upload_bytes(original_name: str, content: bytes) -> Path:
    path = _upload_target_path(original_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _upload_target_path(original_name: str) -> Path:
    return UPLOAD_DIR / _safe_filename(original_name)


def _parse_uploaded_document(
    saved_path: Path,
    suffix: str,
    category: str,
    drug_name: str | None,
    generic_name: str | None,
) -> tuple[DrugDocument, list[DrugChunk]]:
    """按文件类型选择解析器，并应用前端传入的药品名、通用名、分类覆盖值。"""
    if suffix == ".pdf":
        document, chunks = parse_drug_pdf(saved_path, category=category)
    elif suffix == ".docx":
        text = _extract_docx_text(saved_path)
        document, chunks = _parse_plain_text_document(saved_path, text, category=category)
    else:
        text = _decode_text(saved_path.read_bytes())
        document, chunks = _parse_plain_text_document(saved_path, text, category=category)

    if drug_name:
        document.drug_name = drug_name
    if generic_name:
        document.generic_name = generic_name
    document.category = category
    for chunk in chunks:
        if drug_name:
            chunk.drug_name = drug_name
        if generic_name:
            chunk.generic_name = generic_name
        chunk.category = category
    return document, chunks


def _parse_plain_text_document(path: Path, text: str, category: str) -> tuple[DrugDocument, list[DrugChunk]]:
    """解析 TXT/MD/DOCX 文本：按【章节名】切段，再按窗口切成 chunk。"""
    normalized = _normalize_upload_text(text)
    doc_id = _make_upload_doc_id(path, normalized)
    name = path.stem
    sections = _extract_upload_sections(normalized)
    document = DrugDocument(
        doc_id=doc_id,
        source_file=path.name,
        source_path=str(path),
        pages=1,
        drug_name=name,
        generic_name=name,
        approval_number="unknown",
        category=category,
    )
    chunks = []
    for section_index, (section_name, section_text) in enumerate(sections):
        for chunk_index, chunk_text in enumerate(_split_upload_text(section_text)):
            chunks.append(
                DrugChunk(
                    chunk_id=f"{doc_id}:{section_index:03d}:{chunk_index:03d}",
                    doc_id=doc_id,
                    text=chunk_text,
                    section=section_name,
                    source=SourceSpan(
                        source_file=path.name,
                        source_path=str(path),
                        page_start=1,
                        page_end=1,
                    ),
                    drug_name=document.drug_name,
                    generic_name=document.generic_name,
                    category=category,
                    section_index=section_index,
                    chunk_index=chunk_index,
                    metadata={"upload_type": path.suffix.lower().lstrip(".")},
                )
            )
    return document, chunks


def _append_chunks(chunks: list[DrugChunk], path: Path) -> None:
    """把新解析出的 chunk 追加写入 master JSONL，作为上传历史和 catalog 数据源。"""
    path = ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_json(), ensure_ascii=False) + "\n")


def _extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as docx:
            xml_bytes = docx.read("word/document.xml")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"DOCX 解析失败：{exc}") from exc
    root = ElementTree.fromstring(xml_bytes)
    texts = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            texts.append(node.text)
    return "\n".join(texts)


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _split_upload_text(text: str, chunk_size: int = 900, chunk_overlap: int = 120) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        window = text[start:end]
        if end < len(text):
            cut = max(window.rfind("\n"), window.rfind("。"), window.rfind("；"))
            if cut > chunk_size * 0.45:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _extract_upload_sections(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"【([^】]{1,40})】", text))
    if not matches:
        return [("全文", text)] if text else []

    sections: list[tuple[str, str]] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        sections.append(("全文", prefix))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        name = match.group(1).strip()
        body = text[start:end].strip()
        if body:
            sections.append((name, body))
    return sections


def _normalize_upload_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def _make_upload_doc_id(path: Path, text: str) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _content_md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def _load_master_chunks(path: Path) -> list[DrugChunk]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _same_upload_source(chunk: DrugChunk, target_path: Path) -> bool:
    target = str(target_path)
    return chunk.source.source_path == target or chunk.source.source_file == target_path.name


def _rewrite_master_without_chunks(path: Path, removed_chunks: list[DrugChunk]) -> None:
    if not path.exists() or not removed_chunks:
        return
    removed_ids = {chunk.chunk_id for chunk in removed_chunks}
    remaining = [chunk for chunk in read_jsonl(path) if chunk.chunk_id not in removed_ids]
    write_jsonl(remaining, path)


def _delete_chunks_from_chroma(chunks: list[DrugChunk]) -> None:
    """从原文 collection 和摘要 collection 中删除旧版本文档。"""
    if not chunks:
        return
    settings = get_settings()
    config = load_yaml(settings.config_path) if Path(settings.config_path).exists() else {}
    embedding_cfg = config.get("embedding", {})
    vector_store = load_vector_store(
        persist_dir=settings.persist_dir,
        collection=settings.collection,
        embedding_provider=embedding_cfg.get("provider", "hash"),
        embedding_model=embedding_cfg.get("model"),
        dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
        embedding_base_url=embedding_cfg.get("base_url"),
        embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
    )
    summary_store = load_vector_store(
        persist_dir=settings.persist_dir,
        collection=settings.summary_collection,
        embedding_provider=embedding_cfg.get("provider", "hash"),
        embedding_model=embedding_cfg.get("model"),
        dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
        embedding_base_url=embedding_cfg.get("base_url"),
        embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
    )
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    summary_ids = sorted({f"{chunk.doc_id}:summary:{chunk.section}" for chunk in chunks})
    try:
        vector_store.delete(ids=chunk_ids)
    except Exception:
        pass
    try:
        summary_store.delete(ids=summary_ids)
    except Exception:
        pass


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(filename).name).strip(" .")
    if not cleaned:
        cleaned = "uploaded-document"
    if len(cleaned) > 120:
        stem = Path(cleaned).stem[:80]
        suffix = Path(cleaned).suffix[:12]
        cleaned = f"{stem}{suffix}"
    return cleaned


def _optional_int(value: Any) -> int | None:
    if value in {None, "", "none", "null"}:
        return None
    return int(value)


app = create_app()
# 功能类似
#   python -m uvicorn drug_rag.api.main:app --host 127.0.0.1 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("drug_rag.api.main:app", host="127.0.0.1", port=8000, reload=True)

