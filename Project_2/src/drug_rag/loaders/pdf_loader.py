from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PdfPage:
    page_number: int
    text: str


def load_pdf_pages(pdf_path: str | Path) -> list[PdfPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pypdf. Install with `pip install pypdf`.") from exc

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(path)

    reader = PdfReader(str(path))
    pages: list[PdfPage] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PdfPage(page_number=idx, text=text))
    return pages
