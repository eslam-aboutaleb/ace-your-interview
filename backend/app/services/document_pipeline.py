"""Document ingestion pipeline: extract → chunk → embed → persist.

Blocking work (extraction, chunking, persistence) runs on a
dedicated single-worker executor thread; embedding calls are
awaited asynchronously. Failures mark the document ``failed``
with the error message.
"""

from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
MAX_CHUNKS_PER_DOCUMENT = 2000
# Hard guard consistent with chat's 200k-character input guard.
MAX_DOCUMENT_TEXT_CHARS = 200_000
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

ALLOWED_MIME_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "md",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}
ALLOWED_EXTENSIONS = {"pdf", "txt", "md", "markdown", "docx"}

SCANNED_PDF_ERROR = "scanned PDFs not supported in v1"

_SEPARATORS = ["\n\n", "\n", " ", ""]

T = TypeVar("T")


class DocumentProcessingError(RuntimeError):
    """Raised when extraction or chunking fails."""


def file_extension(filename: str) -> str:
    name = (filename or "").rsplit(".", 1)
    return name[-1].strip().lower() if len(name) == 2 else ""


def resolve_document_kind(filename: str, mime_type: str) -> str:
    """Return the canonical kind (pdf/txt/md/docx) for an upload."""
    mime = (mime_type or "").strip().lower()
    if mime in ALLOWED_MIME_TYPES:
        return ALLOWED_MIME_TYPES[mime]
    extension = file_extension(filename)
    if extension in ALLOWED_EXTENSIONS:
        if extension == "markdown":
            return "md"
        return extension
    raise DocumentProcessingError(
        f"unsupported file type: {mime_type or mime or 'unknown mime'} "
        f"({filename or 'unknown filename'})"
    )


def extract_text(filename: str, mime_type: str, data: bytes) -> str:
    kind = resolve_document_kind(filename, mime_type)
    if kind == "pdf":
        return _extract_pdf(data)
    if kind == "docx":
        return _extract_docx(data)
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts = [
        page.extract_text() or "" for page in reader.pages
    ]
    text = "\n".join(parts).strip()
    if not text:
        # Image-only / scanned PDFs yield no text layer.
        raise DocumentProcessingError(SCANNED_PDF_ERROR)
    return text


def _extract_docx(data: bytes) -> str:
    import mammoth

    result = mammoth.extract_raw_text(io.BytesIO(data))
    return (result.value or "").strip()


def _split_recursive(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    for separator in _SEPARATORS:
        if separator and separator in text:
            pieces: list[str] = []
            for part in text.split(separator):
                pieces.extend(
                    _split_recursive(part, chunk_size)
                )
            return pieces
    return [
        text[i : i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


def split_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Recursive character split with a fixed overlap."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))
    segments = _split_recursive(text, chunk_size)
    chunks: list[str] = []
    buffer = ""
    for segment in segments:
        if not segment.strip():
            continue
        if buffer and len(buffer) + len(segment) + 1 > chunk_size:
            chunks.append(buffer)
            buffer = buffer[-overlap:] if overlap else ""
        buffer = f"{buffer}\n{segment}" if buffer else segment
        while len(buffer) > chunk_size:
            chunks.append(buffer[:chunk_size])
            buffer = (
                buffer[chunk_size - overlap :]
                if overlap
                else buffer[chunk_size:]
            )
    if buffer.strip():
        chunks.append(buffer)
    return chunks[:MAX_CHUNKS_PER_DOCUMENT]


class DocumentPipeline:
    """Extract, chunk, embed and persist uploaded documents."""

    def __init__(self, document_store, embedding_client):
        self._store = document_store
        self._embedding_client = embedding_client
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="document-pipeline"
        )

    async def run_async(
        self, fn: Callable[..., T], /, *args, **kwargs
    ) -> T:
        """Run blocking work on the pipeline executor thread."""
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(
                self._executor,
                lambda: fn(*args, **kwargs),
            )
        return await loop.run_in_executor(self._executor, fn, *args)

    def schedule(
        self,
        user_id: str,
        document_id: str,
        filename: str,
        mime_type: str,
        data: bytes,
    ) -> asyncio.Task:
        """Schedule processing from an async request handler."""
        return asyncio.create_task(
            self.process(
                user_id=user_id,
                document_id=document_id,
                filename=filename,
                mime_type=mime_type,
                data=data,
            )
        )

    async def process(
        self,
        user_id: str,
        document_id: str,
        filename: str,
        mime_type: str,
        data: bytes,
    ) -> None:
        try:
            await self._store.run_async(
                self._store.update_document_status,
                user_id=user_id,
                document_id=document_id,
                status="processing",
            )
            text = await self.run_async(
                extract_text, filename, mime_type, data
            )
            if len(text) > MAX_DOCUMENT_TEXT_CHARS:
                text = text[:MAX_DOCUMENT_TEXT_CHARS]
            chunks = await self.run_async(split_chunks, text)
            if not chunks:
                raise DocumentProcessingError(
                    "document contains no extractable text"
                )
            # Rebuild the vec index first when the embedding
            # provider/dimension changed.
            dimension = self._embedding_client.dimension
            await self._store.run_async(
                self._store.ensure_vec_index, dimension
            )
            embeddings = await self._embedding_client.embed(
                chunks
            )
            await self._store.run_async(
                self._store.persist_chunks,
                user_id=user_id,
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings,
            )
            await self._store.run_async(
                self._store.update_document_status,
                user_id=user_id,
                document_id=document_id,
                status="ready",
                chunk_count=len(chunks),
            )
            logger.info(
                "Document %s processed: %d chunks",
                document_id,
                len(chunks),
            )
        except Exception as e:
            logger.exception(
                "Document %s processing failed", document_id
            )
            await self._store.run_async(
                self._store.update_document_status,
                user_id=user_id,
                document_id=document_id,
                status="failed",
                error=str(e)[:500],
            )
