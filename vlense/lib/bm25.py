from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from ..core.types import ChunkRetrievalResult, IndexedChunk
from .retrieval import build_document_id

_WORD_RE = re.compile(r"[a-zA-Z0-9]+", re.UNICODE)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _split_oversized(text: str, max_chars: int, overlap: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return parts


def build_chunks_from_pdf(
    pdf_path: str | Path,
    *,
    max_chars: int = 4000,
    overlap: int = 400,
    merge_small_pages: bool = False,
) -> List[IndexedChunk]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "BM25 PDF extraction dependencies are missing. Install project dependencies with `uv sync`."
        ) from exc

    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(path)
    try:
        document_id = build_document_id(str(path))
        toc = doc.get_toc(simple=True) or []
        toc_entries = []
        for entry in toc:
            if len(entry) < 3:
                continue
            title = str(entry[1]).strip()
            page_number = int(entry[2])
            if title:
                toc_entries.append((page_number, title))
        toc_entries.sort(key=lambda item: item[0])

        def section_for_page(page_number: int) -> str:
            current = ""
            for toc_page, title in toc_entries:
                if toc_page > page_number:
                    break
                current = title
            return current

        chunks: List[IndexedChunk] = []
        page_buffer: List[tuple[int, str, str]] = []
        buffer_length = 0

        def flush_buffer() -> None:
            nonlocal page_buffer, buffer_length
            if not page_buffer:
                return

            merged = "\n\n".join(text for _, text, _ in page_buffer if text.strip())
            merged = _normalize_whitespace(merged)
            section_hint = next((section for _, _, section in page_buffer if section), "")
            page_start = page_buffer[0][0]
            page_end = page_buffer[-1][0]
            page_buffer = []
            buffer_length = 0

            if not merged:
                return

            for piece in _split_oversized(merged, max_chars, overlap):
                chunks.append(
                    IndexedChunk(
                        document_id=document_id,
                        source_path=str(path),
                        file_name=path.name,
                        page_start=page_start,
                        page_end=page_end,
                        text=piece,
                        section_hint=section_hint,
                    )
                )

        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = _normalize_whitespace(page.get_text("text") or "")
            if not text:
                continue

            page_number = index + 1
            section_hint = section_for_page(page_number)

            if not merge_small_pages:
                for piece in _split_oversized(text, max_chars, overlap):
                    chunks.append(
                        IndexedChunk(
                            document_id=document_id,
                            source_path=str(path),
                            file_name=path.name,
                            page_start=page_number,
                            page_end=page_number,
                            text=piece,
                            section_hint=section_hint,
                        )
                    )
                continue

            if len(text) > max_chars:
                flush_buffer()
                for piece in _split_oversized(text, max_chars, overlap):
                    chunks.append(
                        IndexedChunk(
                            document_id=document_id,
                            source_path=str(path),
                            file_name=path.name,
                            page_start=page_number,
                            page_end=page_number,
                            text=piece,
                            section_hint=section_hint,
                        )
                    )
                continue

            piece_length = len(text) + 2
            if page_buffer and buffer_length + piece_length > max_chars:
                flush_buffer()
            page_buffer.append((page_number, text, section_hint))
            buffer_length += piece_length
            if buffer_length >= max_chars:
                flush_buffer()

        flush_buffer()
        return chunks
    finally:
        doc.close()


def retrieve_chunks(
    chunks: Iterable[IndexedChunk],
    query: str,
    *,
    top_k: int = 8,
) -> List[ChunkRetrievalResult]:
    try:
        from rank_bm25 import BM25Plus
    except ImportError as exc:
        raise ImportError(
            "BM25 dependencies are missing. Install project dependencies with `uv sync`."
        ) from exc

    chunk_list = list(chunks)
    if not chunk_list:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    tokenized_corpus = []
    for chunk in chunk_list:
        tokens = _tokenize(chunk.text)
        tokenized_corpus.append(tokens or ["empty"])

    # BM25Plus behaves better than BM25Okapi on tiny corpora, where exact-match
    # terms can otherwise collapse to zero IDF and produce arbitrary ties.
    scores = BM25Plus(tokenized_corpus).get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    return [
        ChunkRetrievalResult(
            chunk=chunk_list[index],
            score=float(scores[index]),
        )
        for index in ranked_indices
    ]


def serialize_chunks(chunks: Iterable[IndexedChunk]) -> List[dict]:
    return [asdict(chunk) for chunk in chunks]
