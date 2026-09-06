"""Offline PDF -> chunks -> embeddings -> FAISS ingestion command."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

from .chunker import chunk_pages
from .embedding import EmbeddingModel, EmbeddingModelError, HashingEmbeddingModel
from .pdf_parser import PDFParserError, parse_pdf, scan_pdfs
from .vector_store import VectorStore, VectorStoreError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPERS_DIR = PROJECT_ROOT / "papers"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "vector_db"


def build_index(
    papers_dir: str | Path = DEFAULT_PAPERS_DIR,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    model_name: str | None = None,
    backend: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> dict[str, Any]:
    """Build and persist a local index, returning reproducibility statistics."""

    pdfs = scan_pdfs(papers_dir)
    print(f"[RAG] Found {len(pdfs)} PDFs")
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {Path(papers_dir)}")

    pages: list[dict[str, Any]] = []
    warnings: list[str] = []
    for pdf_path in pdfs:
        try:
            parsed_pages = parse_pdf(pdf_path)
        except PDFParserError as exc:
            warning = f"Warning: {exc}"
            warnings.append(warning)
            print(warning)
            continue
        if not parsed_pages:
            warning = f"Warning: No extractable text found in {pdf_path.name}"
            warnings.append(warning)
            print(warning)
            continue
        pages.extend(parsed_pages)

    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    print(f"[RAG] Total pages with text: {len(pages)}")
    print(f"[RAG] Total chunks: {len(chunks)}")
    if not chunks:
        raise ValueError("No extractable text chunks were produced from the PDF directory.")

    started = time.perf_counter()
    print("[RAG] Encoding chunks (CPU; this may take a while on first run)...")
    selected_backend = (backend or os.getenv("RAG_EMBEDDING_BACKEND", "sentence-transformers")).strip().lower()
    if selected_backend in {"hashing", "local-hash", "offline"}:
        model = HashingEmbeddingModel()
    elif selected_backend in {"sentence-transformers", "sentence_transformers", "bge"}:
        model = EmbeddingModel(
            model_name=model_name,
            batch_size=batch_size,
            device=device or os.getenv("RAG_EMBEDDING_DEVICE", "cpu"),
        )
    else:
        raise ValueError(f"Unknown embedding backend {selected_backend}")
    embeddings = model.encode_documents([chunk["text"] for chunk in chunks])
    embedding_seconds = time.perf_counter() - started
    print(f"[RAG] Document embedding finished in {embedding_seconds:.2f}s")

    target_dir = Path(index_dir)
    print("[RAG] Saving FAISS index...")
    store = VectorStore.build(
        embeddings,
        chunks,
        index_path=target_dir / "index.faiss",
        metadata_path=target_dir / "metadata.json",
    )
    config = {
        "backend": selected_backend,
        "model_name": model.model_name,
        "device": getattr(model, "device", "cpu"),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_dimension": int(embeddings.shape[1]),
    }
    with (target_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    print("[RAG] Done.")
    return {
        "pdf_count": len(pdfs),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "embedding_backend": selected_backend,
        "embedding_model": model.model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_seconds": round(embedding_seconds, 3),
        "index_path": str(store.index_path),
        "metadata_path": str(store.metadata_path),
        "index_size_bytes": store.file_size_bytes(),
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Research Agent V2 local paper index")
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--model", default=None, help="Sentence-transformers model name")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["sentence-transformers", "hashing"],
        help="Embedding backend; hashing is offline and useful when Hugging Face is unavailable",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Default is cpu on macOS")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--overlap", type=int, default=200)
    return parser


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()
    args = build_parser().parse_args()
    try:
        stats = build_index(
            papers_dir=args.papers_dir,
            index_dir=args.index_dir,
            model_name=args.model,
            backend=args.backend,
            batch_size=args.batch_size,
            device=args.device,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    except (FileNotFoundError, PDFParserError, EmbeddingModelError, VectorStoreError, ValueError) as exc:
        print(f"ToolError: RAG ingestion failed: {exc}")
        return 2
    print(f"[RAG] Index size: {stats['index_size_bytes']} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
