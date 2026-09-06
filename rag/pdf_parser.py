"""Text extraction from text-based PDF files using PyMuPDF."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    # Newer PyMuPDF releases expose the preferred import name directly.
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover - compatibility with older releases
    try:
        import fitz  # type: ignore
    except ImportError:  # pragma: no cover - exercised in a missing-dependency environment
        fitz = None  # type: ignore[assignment]


class PDFParserError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed."""


def _clean_text(text: str) -> str:
    """Remove extraction noise while retaining paragraph boundaries."""

    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join words split at a line boundary, but keep ordinary line breaks.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if cleaned and not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        cleaned.append(line)
        previous_blank = False
    return "\n".join(cleaned).strip()


def parse_pdf(path: str | Path, source: str | None = None) -> list[dict[str, Any]]:
    """Extract non-empty pages from a text-based PDF.

    Page numbers are one-based because they are intended for user-facing
    citations. The returned ``source`` is the supplied display name or the
    PDF filename.
    """

    if fitz is None:
        raise PDFParserError("PyMuPDF is not installed; install pymupdf first.")
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise PDFParserError(f"PDF file not found: {pdf_path}")

    display_source = source or pdf_path.name
    pages: list[dict[str, Any]] = []
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:  # fitz exposes several backend-specific errors
        raise PDFParserError(f"Could not open {pdf_path.name}: {exc}") from exc

    try:
        for page_number, page in enumerate(document, start=1):
            text = _clean_text(page.get_text("text"))
            if text:
                pages.append({"source": display_source, "page": page_number, "text": text})
    finally:
        document.close()
    return pages


def scan_pdfs(papers_dir: str | Path) -> list[Path]:
    """Return all PDFs below ``papers_dir`` in deterministic order."""

    directory = Path(papers_dir)
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: str(path).lower(),
    )
