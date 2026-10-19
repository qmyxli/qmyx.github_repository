from __future__ import annotations

import csv
import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentSpec:
    path: Path
    category: str = "unknown"
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CategoryRule:
    category: str
    patterns: list[str]


@dataclass(slots=True)
class BatchManifest:
    defaults: dict[str, Any] = field(default_factory=dict)
    documents: list[DocumentSpec] = field(default_factory=list)
    category_rules: list[CategoryRule] = field(default_factory=list)


def load_batch_manifest(path: str | Path | None) -> BatchManifest:
    if not path:
        return BatchManifest()
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    suffix = manifest_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from exc
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    elif suffix == ".json":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif suffix == ".csv":
        data = _load_csv_manifest(manifest_path)
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")

    return _manifest_from_mapping(data, base_dir=manifest_path.parent)


def resolve_category(
    pdf_path: str | Path,
    manifest: BatchManifest,
    default_category: str = "unknown",
    category_from_dir: bool = False,
) -> str:
    path = Path(pdf_path)
    explicit = _explicit_document_category(path, manifest)
    if explicit:
        return explicit

    for rule in manifest.category_rules:
        for pattern in rule.patterns:
            if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern):
                return rule.category

    if category_from_dir and path.parent.name:
        return path.parent.name
    return manifest.defaults.get("category") or default_category


def iter_manifest_documents(manifest: BatchManifest) -> list[DocumentSpec]:
    return list(manifest.documents)


def _load_csv_manifest(path: Path) -> dict[str, Any]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return {"documents": rows}


def _manifest_from_mapping(data: dict[str, Any], base_dir: Path) -> BatchManifest:
    defaults = data.get("defaults", {}) or {}
    documents = []
    for item in data.get("documents", []) or []:
        raw_path = item.get("path") or item.get("file") or item.get("pdf")
        if not raw_path:
            continue
        doc_path = Path(raw_path)
        if not doc_path.is_absolute():
            doc_path = base_dir / doc_path
        documents.append(
            DocumentSpec(
                path=doc_path,
                category=item.get("category") or defaults.get("category") or "unknown",
                tags=_split_tags(item.get("tags")),
            )
        )

    category_rules = []
    categories = data.get("categories", {}) or {}
    if isinstance(categories, dict):
        for category, value in categories.items():
            if isinstance(value, dict):
                patterns = value.get("patterns", []) or []
            elif isinstance(value, list):
                patterns = value
            else:
                patterns = [str(value)]
            category_rules.append(CategoryRule(category=category, patterns=[str(p) for p in patterns]))

    return BatchManifest(defaults=defaults, documents=documents, category_rules=category_rules)


def _explicit_document_category(path: Path, manifest: BatchManifest) -> str:
    resolved = _safe_resolve(path)
    for doc in manifest.documents:
        if _safe_resolve(doc.path) == resolved:
            return doc.category
    return ""


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _split_tags(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]
