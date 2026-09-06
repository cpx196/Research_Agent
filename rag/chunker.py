"""Simple character-based chunking for extracted paper pages."""

from __future__ import annotations

from typing import Any


def _validate_parameters(chunk_size: int, overlap: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 2000,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """Split pages into overlapping character windows.

    The task specification allows characters as a token approximation. A
    chunk never crosses a page boundary, so every result has an unambiguous
    user-facing page citation.
    """

    _validate_parameters(chunk_size, overlap)
    chunks: list[dict[str, Any]] = []
    step = chunk_size - overlap
    for page in pages:
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        source = str(page.get("source", ""))
        page_number = page.get("page")
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "source": source,
                        "page": page_number,
                        "text": chunk_text,
                        "start_char": start,
                        "end_char": end,
                    }
                )
            if end >= len(text):
                break
            start += step
    return chunks
