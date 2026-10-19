from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from drug_rag.chains import answer_question
from drug_rag.config import deep_get, load_yaml
from drug_rag.indexing import build_vector_index, load_vector_store
from drug_rag.indexing.metadata_index import (
    build_metadata_catalog,
    find_drugs,
    load_metadata_catalog,
    resolve_query_constraints,
)
from drug_rag.indexing.summary_index import build_summary_index
from drug_rag.io_utils import write_jsonl
from drug_rag.parsers import parse_drug_pdf
from drug_rag.pipelines import parse_pdf_batch, parse_pdf_directory, write_batch_report
from drug_rag.retrieval import build_metadata_filter, build_query_rewrite, retrieve_advanced
from drug_rag.retrieval.advanced_retriever import RERANK_STRATEGIES, RETRIEVAL_MODES


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(prog="drug-rag")
    parser.add_argument("--config", default="configs/app.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse", help="Parse one drug instruction PDF into section chunks.")
    parse_cmd.add_argument("--pdf", required=True)
    parse_cmd.add_argument("--out", required=True)
    parse_cmd.add_argument("--category", default="unknown")
    parse_cmd.add_argument("--chunk-size", type=int, default=900)
    parse_cmd.add_argument("--chunk-overlap", type=int, default=120)

    batch_parse_cmd = subparsers.add_parser("batch-parse", help="Parse all PDFs in a directory into one JSONL.")
    batch_parse_cmd.add_argument("--input-dir", default=None)
    batch_parse_cmd.add_argument("--out", required=True)
    batch_parse_cmd.add_argument("--category", default="unknown")
    batch_parse_cmd.add_argument("--chunk-size", type=int, default=900)
    batch_parse_cmd.add_argument("--chunk-overlap", type=int, default=120)
    batch_parse_cmd.add_argument("--no-recursive", action="store_true")
    batch_parse_cmd.add_argument("--manifest", default=None)
    batch_parse_cmd.add_argument("--category-from-dir", action="store_true")
    batch_parse_cmd.add_argument("--fail-fast", action="store_true")
    batch_parse_cmd.add_argument("--report-out", default=None)

    batch_build_cmd = subparsers.add_parser("batch-build", help="Parse many PDFs and build chunk, summary, and catalog indexes.")
    batch_build_cmd.add_argument("--input-dir", default=None)
    batch_build_cmd.add_argument("--manifest", default=None)
    batch_build_cmd.add_argument("--jsonl", default="data/processed/all_drug_chunks.jsonl")
    batch_build_cmd.add_argument("--persist-dir", default=None)
    batch_build_cmd.add_argument("--collection", default=None)
    batch_build_cmd.add_argument("--summary-collection", default=None)
    batch_build_cmd.add_argument("--catalog-out", default="data/processed/drug_catalog.json")
    batch_build_cmd.add_argument("--report-out", default="data/processed/batch_report.json")
    batch_build_cmd.add_argument("--category", default="unknown")
    batch_build_cmd.add_argument("--category-from-dir", action="store_true")
    batch_build_cmd.add_argument("--chunk-size", type=int, default=900)
    batch_build_cmd.add_argument("--chunk-overlap", type=int, default=120)
    batch_build_cmd.add_argument("--no-recursive", action="store_true")
    batch_build_cmd.add_argument("--fail-fast", action="store_true")

    ingest_cmd = subparsers.add_parser("ingest", help="Build Chroma vector index from JSONL chunks.")
    ingest_cmd.add_argument("--jsonl", required=True)
    ingest_cmd.add_argument("--persist-dir", default=None)
    ingest_cmd.add_argument("--collection", default=None)
    ingest_cmd.add_argument("--summary", action="store_true", help="Also build section summary index.")
    ingest_cmd.add_argument("--catalog-out", default=None, help="Optional metadata catalog JSON path.")

    catalog_cmd = subparsers.add_parser("catalog", help="Build metadata catalog JSON from parsed chunks.")
    catalog_cmd.add_argument("--jsonl", required=True)
    catalog_cmd.add_argument("--out", required=True)

    catalog_search_cmd = subparsers.add_parser("catalog-search", help="Search drugs in the metadata catalog.")
    catalog_search_cmd.add_argument("--catalog", required=True)
    catalog_search_cmd.add_argument("--query", default=None)
    catalog_search_cmd.add_argument("--drug-name", default=None)
    catalog_search_cmd.add_argument("--generic-name", default=None)
    catalog_search_cmd.add_argument("--category", default=None)
    catalog_search_cmd.add_argument("--section", default=None)
    catalog_search_cmd.add_argument("--json", action="store_true")

    route_cmd = subparsers.add_parser("route", help="Show intent routing, query rewrite, and metadata filter.")
    route_cmd.add_argument("--question", required=True)
    route_cmd.add_argument("--drug-name", default=None)
    route_cmd.add_argument("--generic-name", default=None)
    route_cmd.add_argument("--category", default=None)
    route_cmd.add_argument("--catalog", default=None)
    route_cmd.add_argument("--json", action="store_true")

    retrieve_cmd = subparsers.add_parser("retrieve", help="Run observable metadata-aware retrieval.")
    retrieve_cmd.add_argument("--question", required=True)
    retrieve_cmd.add_argument("--persist-dir", default=None)
    retrieve_cmd.add_argument("--collection", default=None)
    retrieve_cmd.add_argument("--summary-collection", default=None)
    retrieve_cmd.add_argument("--top-k", type=int, default=6)
    retrieve_cmd.add_argument("--candidate-k", type=int, default=None)
    retrieve_cmd.add_argument("--final-k", type=int, default=None)
    retrieve_cmd.add_argument("--mode", choices=RETRIEVAL_MODES, default="auto")
    retrieve_cmd.add_argument("--rerank", choices=RERANK_STRATEGIES, default="auto")
    retrieve_cmd.add_argument("--drug-name", default=None)
    retrieve_cmd.add_argument("--generic-name", default=None)
    retrieve_cmd.add_argument("--category", default=None)
    retrieve_cmd.add_argument("--catalog", default=None)
    retrieve_cmd.add_argument("--json", action="store_true")

    ask_cmd = subparsers.add_parser("ask", help="Ask a question against the Chroma index.")
    ask_cmd.add_argument("--question", required=True)
    ask_cmd.add_argument("--persist-dir", default=None)
    ask_cmd.add_argument("--collection", default=None)
    ask_cmd.add_argument("--summary-collection", default=None)
    ask_cmd.add_argument("--top-k", type=int, default=6)
    ask_cmd.add_argument("--mode", choices=RETRIEVAL_MODES, default="auto")
    ask_cmd.add_argument("--rerank", choices=RERANK_STRATEGIES, default="auto")
    ask_cmd.add_argument("--drug-name", default=None)
    ask_cmd.add_argument("--generic-name", default=None)
    ask_cmd.add_argument("--category", default=None)
    ask_cmd.add_argument("--catalog", default=None)

    args = parser.parse_args()

    if args.command == "parse":
        document, chunks = parse_drug_pdf(
            args.pdf,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            category=args.category,
        )
        count = write_jsonl(chunks, args.out)
        print(f"Parsed {document.source_file}")
        print(f"doc_id={document.doc_id}")
        print(f"drug_name={document.drug_name}")
        print(f"generic_name={document.generic_name}")
        print(f"approval_number={document.approval_number}")
        print(f"chunks={count}")
        print(f"out={args.out}")
        return

    if args.command == "batch-parse":
        report = parse_pdf_batch(
            input_dir=args.input_dir,
            out_jsonl=args.out,
            category=args.category,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            recursive=not args.no_recursive,
            manifest_path=args.manifest,
            category_from_dir=args.category_from_dir,
            fail_fast=args.fail_fast,
        )
        if args.report_out:
            write_batch_report(report, args.report_out)
            print(f"report={args.report_out}")
        print(f"Parsed pdfs={report.doc_count}")
        print(f"chunks={report.chunk_count}")
        print(f"failed={report.failed_count}")
        print(f"out={args.out}")
        return

    if args.command == "batch-build":
        config = _load_config_if_exists(args.config)
        persist_dir = args.persist_dir or deep_get(config, "paths.chroma_dir", "data/chroma")
        collection = args.collection or deep_get(config, "index.chunk_collection", "drug_chunks")
        summary_collection = args.summary_collection or deep_get(config, "index.summary_collection", "drug_summaries")
        embedding_cfg = config.get("embedding", {})
        report = parse_pdf_batch(
            input_dir=args.input_dir,
            out_jsonl=args.jsonl,
            category=args.category,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            recursive=not args.no_recursive,
            manifest_path=args.manifest,
            category_from_dir=args.category_from_dir,
            fail_fast=args.fail_fast,
        )
        write_batch_report(report, args.report_out)
        chunk_count = build_vector_index(
            args.jsonl,
            persist_dir=persist_dir,
            collection=collection,
            embedding_provider=embedding_cfg.get("provider", "hash"),
            embedding_model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            embedding_base_url=embedding_cfg.get("base_url"),
            embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
        )
        summary_count = build_summary_index(
            args.jsonl,
            persist_dir=persist_dir,
            collection=summary_collection,
            embedding_provider=embedding_cfg.get("provider", "hash"),
            embedding_model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            embedding_base_url=embedding_cfg.get("base_url"),
            embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
        )
        catalog = build_metadata_catalog(args.jsonl, args.catalog_out)
        print(f"Parsed pdfs={report.doc_count}")
        print(f"failed={report.failed_count}")
        print(f"Indexed chunks={chunk_count}")
        print(f"Indexed summaries={summary_count}")
        print(f"drug_count={catalog['drug_count']}")
        print(f"categories={catalog.get('category_counts', {})}")
        print(f"jsonl={args.jsonl}")
        print(f"catalog={args.catalog_out}")
        print(f"report={args.report_out}")
        return

    if args.command == "catalog":
        catalog = build_metadata_catalog(args.jsonl, args.out)
        print(f"metadata_catalog={args.out}")
        print(f"drug_count={catalog['drug_count']}")
        print(f"categories={catalog.get('category_counts', {})}")
        return

    if args.command == "catalog-search":
        catalog = load_metadata_catalog(args.catalog)
        if args.query:
            resolution = resolve_query_constraints(catalog, args.query)
            payload = resolution.to_dict()
        else:
            payload = {
                "matches": find_drugs(
                    catalog,
                    drug_name=args.drug_name,
                    generic_name=args.generic_name,
                    category=args.category,
                    section=args.section,
                )
            }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_catalog_search(payload)
        return

    if args.command == "route":
        drug_name, generic_name, category, resolution = _resolve_catalog_args(args)
        rewrite = build_query_rewrite(
            args.question,
            drug_name=drug_name,
            generic_name=generic_name,
        )
        metadata_filter = build_metadata_filter(
            rewrite.route,
            drug_name=drug_name,
            generic_name=generic_name,
            category=category,
        )
        payload = {
            "question": args.question,
            "route": rewrite.route.to_dict(),
            "rewrite": rewrite.to_dict(),
            "metadata_filter": metadata_filter,
            "catalog_resolution": resolution.to_dict() if resolution else None,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_route(payload)
        return

    if args.command == "ingest":
        config = _load_config_if_exists(args.config)
        persist_dir = args.persist_dir or deep_get(config, "paths.chroma_dir", "data/chroma")
        collection = args.collection or deep_get(config, "index.chunk_collection", "drug_chunks")
        embedding_cfg = config.get("embedding", {})
        count = build_vector_index(
            args.jsonl,
            persist_dir=persist_dir,
            collection=collection,
            embedding_provider=embedding_cfg.get("provider", "hash"),
            embedding_model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            embedding_base_url=embedding_cfg.get("base_url"),
            embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
        )
        print(f"Indexed chunks={count}")
        print(f"persist_dir={persist_dir}")
        print(f"collection={collection}")
        if args.catalog_out:
            catalog = build_metadata_catalog(args.jsonl, args.catalog_out)
            print(f"metadata_catalog={args.catalog_out}")
            print(f"drug_count={catalog['drug_count']}")
        if args.summary:
            summary_collection = deep_get(config, "index.summary_collection", "drug_summaries")
            summary_count = build_summary_index(
                args.jsonl,
                persist_dir=persist_dir,
                collection=summary_collection,
                embedding_provider=embedding_cfg.get("provider", "hash"),
                embedding_model=embedding_cfg.get("model"),
                dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
                embedding_base_url=embedding_cfg.get("base_url"),
                embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
            )
            print(f"Indexed summaries={summary_count}")
            print(f"summary_collection={summary_collection}")
        return

    if args.command == "retrieve":
        config = _load_config_if_exists(args.config)
        persist_dir = args.persist_dir or deep_get(config, "paths.chroma_dir", "data/chroma")
        collection = args.collection or deep_get(config, "index.chunk_collection", "drug_chunks")
        summary_collection = args.summary_collection or deep_get(config, "index.summary_collection", "drug_summaries")
        embedding_cfg = config.get("embedding", {})
        llm_cfg = config.get("llm", {})
        rerank_cfg = config.get("rerank", {})
        prompts_path = deep_get(config, "prompts.path", "configs/prompts.yaml")
        vector_store = load_vector_store(
            persist_dir=persist_dir,
            collection=collection,
            embedding_provider=embedding_cfg.get("provider", "hash"),
            embedding_model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            embedding_base_url=embedding_cfg.get("base_url"),
            embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
        )
        summary_store = None
        if args.mode in {"auto", "summary", "hybrid"}:
            summary_store = load_vector_store(
                persist_dir=persist_dir,
                collection=summary_collection,
                embedding_provider=embedding_cfg.get("provider", "hash"),
                embedding_model=embedding_cfg.get("model"),
                dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
                embedding_base_url=embedding_cfg.get("base_url"),
                embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
            )
        drug_name, generic_name, category, resolution = _resolve_catalog_args(args)
        trace = retrieve_advanced(
            vector_store=vector_store,
            summary_store=summary_store,
            question=args.question,
            mode=args.mode,
            rerank=args.rerank,
            candidate_k=args.candidate_k,
            final_k=args.final_k or args.top_k,
            drug_name=drug_name,
            generic_name=generic_name,
            category=category,
            rerank_model_provider=rerank_cfg.get("provider", "none"),
            rerank_model_name=rerank_cfg.get("model", "gte-rerank-v2"),
            rerank_model_endpoint=rerank_cfg.get("endpoint"),
            rerank_model_api_key_env=rerank_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
            rerank_model_timeout=int(rerank_cfg.get("timeout", 30)),
            rerank_model_strategies=rerank_cfg.get("use_for_strategies"),
            preserve_preferred_sections=bool(rerank_cfg.get("preserve_preferred_sections", True)),
        )
        if args.json:
            print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_retrieval_trace(trace.to_dict())
        return

    if args.command == "ask":
        config = _load_config_if_exists(args.config)
        persist_dir = args.persist_dir or deep_get(config, "paths.chroma_dir", "data/chroma")
        collection = args.collection or deep_get(config, "index.chunk_collection", "drug_chunks")
        summary_collection = args.summary_collection or deep_get(config, "index.summary_collection", "drug_summaries")
        embedding_cfg = config.get("embedding", {})
        llm_cfg = config.get("llm", {})
        vector_store = load_vector_store(
            persist_dir=persist_dir,
            collection=collection,
            embedding_provider=embedding_cfg.get("provider", "hash"),
            embedding_model=embedding_cfg.get("model"),
            dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
            embedding_base_url=embedding_cfg.get("base_url"),
            embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
        )
        summary_store = None
        if args.mode in {"auto", "summary", "hybrid"}:
            summary_store = load_vector_store(
                persist_dir=persist_dir,
                collection=summary_collection,
                embedding_provider=embedding_cfg.get("provider", "hash"),
                embedding_model=embedding_cfg.get("model"),
                dimensions=_optional_int(embedding_cfg.get("dimensions", 512)),
                embedding_base_url=embedding_cfg.get("base_url"),
                embedding_api_key_env=embedding_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
            )
        drug_name, generic_name, category, _ = _resolve_catalog_args(args)
        llm_evidence_cfg = llm_cfg.get("evidence", {})
        rerank_cfg = config.get("rerank", {})
        print(
            answer_question(
                args.question,
                vector_store,
                top_k=args.top_k,
                summary_store=summary_store,
                mode=args.mode,
                rerank=args.rerank,
                use_llm=llm_cfg.get("provider", "none") not in {"none", "off", "false"},
                llm_provider=llm_cfg.get("provider", "none"),
                llm_model=llm_cfg.get("model"),
                llm_base_url=llm_cfg.get("base_url"),
                llm_api_key_env=llm_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
                llm_temperature=float(llm_cfg.get("temperature", 0.1)),
                llm_evidence_max_chars_per_doc=int(llm_evidence_cfg.get("max_chars_per_doc", 1200)),
                llm_evidence_max_total_chars=int(llm_evidence_cfg.get("max_total_chars", 6000)),
                llm_evidence_max_chars_per_drug=int(llm_evidence_cfg.get("max_chars_per_drug", 1800)),
                llm_evidence_max_chars_per_drug_doc=int(llm_evidence_cfg.get("max_chars_per_drug_doc", 900)),
                prompts_path=prompts_path,
                rerank_model_provider=rerank_cfg.get("provider", "none"),
                rerank_model_name=rerank_cfg.get("model", "gte-rerank-v2"),
                rerank_model_endpoint=rerank_cfg.get("endpoint"),
                rerank_model_api_key_env=rerank_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
                rerank_model_timeout=int(rerank_cfg.get("timeout", 30)),
                rerank_model_strategies=rerank_cfg.get("use_for_strategies"),
                preserve_preferred_sections=bool(rerank_cfg.get("preserve_preferred_sections", True)),
                drug_name=drug_name,
                generic_name=generic_name,
                category=category,
            )
        )
        return


def _load_config_if_exists(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        return load_yaml(config_path)
    except RuntimeError as exc:
        print(f"Warning: {exc} Falling back to CLI defaults.", file=sys.stderr)
        return {}


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _print_route(payload: dict) -> None:
    route = payload["route"]
    print(f"intent={route['intent']}")
    print(f"confidence={route['confidence']}")
    print(f"matched_keywords={', '.join(route['matched_keywords']) or '-'}")
    print(f"sections={', '.join(route['sections']) or '-'}")
    print(f"rewritten_question={payload['rewrite']['rewritten_question']}")
    print(f"metadata_filter={json.dumps(payload['metadata_filter'], ensure_ascii=False)}")
    if payload.get("catalog_resolution"):
        print(f"catalog_resolution={json.dumps(payload['catalog_resolution'], ensure_ascii=False)}")


def _print_retrieval_trace(payload: dict) -> None:
    route = payload["route"]
    print(f"mode={payload.get('mode', 'metadata')} rerank={payload.get('rerank', 'none')}")
    print(f"intent={route['intent']} confidence={route['confidence']}")
    if payload.get("decision"):
        decision = payload["decision"]
        print(f"strategy={decision.get('strategy')} reason={decision.get('reason')}")
    print(f"search_query={payload['search_query']}")
    print(f"metadata_filter={json.dumps(payload['metadata_filter'], ensure_ascii=False)}")
    if "stage_counts" in payload:
        print(f"stage_counts={json.dumps(payload['stage_counts'], ensure_ascii=False)}")
    print(f"fallback_used={payload['fallback_used']}")
    if payload.get("error"):
        print(f"error={payload['error']}")
    if payload.get("errors"):
        print(f"errors={json.dumps(payload['errors'], ensure_ascii=False)}")
    for item in payload["results"]:
        print(
            f"{item['rank']}. stage={item.get('retrieval_stage', '-')} "
            f"section={item['section']} score={item['score']} "
            f"rerank_score={item.get('rerank_score')} page={item['page_start']}-{item['page_end']} "
            f"chunk_id={item['chunk_id']}"
        )
        print(f"   {item['preview']}")


def _resolve_catalog_args(args) -> tuple[str | None, str | None, str | None, object | None]:
    drug_name = getattr(args, "drug_name", None)
    generic_name = getattr(args, "generic_name", None)
    category = getattr(args, "category", None)
    resolution = None
    catalog_path = getattr(args, "catalog", None)
    question = getattr(args, "question", None)
    if catalog_path and question:
        catalog = load_metadata_catalog(catalog_path)
        resolution = resolve_query_constraints(catalog, question)
        drug_name = drug_name or resolution.drug_name
        generic_name = generic_name or resolution.generic_name
        category = category or resolution.category
    return drug_name, generic_name, category, resolution


def _print_catalog_search(payload: dict) -> None:
    if "matches" in payload:
        matches = payload["matches"]
    else:
        matches = payload.get("matched_drugs") or []
        print(f"category={payload.get('category') or '-'} ambiguous={payload.get('ambiguous')}")
    if not matches:
        print("No catalog matches.")
        return
    for item in matches:
        print(
            f"{item.get('drug_name')} | {item.get('generic_name')} | "
            f"category={item.get('category')} | chunks={item.get('chunk_count')} | "
            f"source={item.get('source_file')}"
        )


def _optional_int(value) -> int | None:
    if value in {None, "", "none", "null"}:
        return None
    return int(value)


if __name__ == "__main__":
    main()
