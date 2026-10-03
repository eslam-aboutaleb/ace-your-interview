"""Coverage for app/services/document_pipeline.py.

Complements ``tests/test_document_store.py`` (which only drives ``split_chunks``
through the happy path) with kind resolution, both extraction backends, the
blank-segment edge of the chunker, and every branch of ``DocumentPipeline.process``
including the disabled-ingestion contract (the pipeline never runs at all when
``STUDY_ENABLE_RAG_V1`` is false, which the router gate enforces).

``pypdf`` and ``mammoth`` are imported lazily *inside* the extraction helpers,
so a stand-in module in ``sys.modules`` exercises this module's own logic (the
page join, the scanned-PDF rejection, the ``.strip()``) without requiring the
native dependency to be installed.
"""

import asyncio
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from app.services.document_pipeline import (
    ALLOWED_MIME_TYPES,
    CHUNK_SIZE,
    MAX_CHUNKS_PER_DOCUMENT,
    MAX_DOCUMENT_TEXT_CHARS,
    SCANNED_PDF_ERROR,
    DocumentPipeline,
    DocumentProcessingError,
    extract_text,
    file_extension,
    resolve_document_kind,
    split_chunks,
)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ── Stand-ins for the lazily imported extraction libraries ──────────────


def _fake_pypdf(page_texts):
    """``pypdf`` double: a PdfReader whose pages yield ``page_texts``."""

    class _Page:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class PdfReader:
        def __init__(self, stream):
            self.stream = stream
            self.pages = [_Page(text) for text in page_texts]

    module = types.ModuleType("pypdf")
    module.PdfReader = PdfReader
    return module


def _fake_mammoth(value):
    module = types.ModuleType("mammoth")

    class _Result:
        def __init__(self):
            self.value = value

    module.extract_raw_text = lambda stream: _Result()
    return module


def _patched_modules(**modules):
    return patch.dict(sys.modules, modules)


# ── Kind resolution ─────────────────────────────────────────────────────


class FileExtensionTests(unittest.TestCase):
    def test_extension_is_lowercased(self):
        self.assertEqual(file_extension("Resume.PDF"), "pdf")
        self.assertEqual(file_extension("notes.tar.gz"), "gz")

    def test_a_name_without_a_dot_has_no_extension(self):
        self.assertEqual(file_extension("README"), "")
        self.assertEqual(file_extension(""), "")
        self.assertEqual(file_extension(None), "")

    def test_a_leading_dot_is_not_an_extension(self):
        self.assertEqual(file_extension(".gitignore"), "gitignore")


class ResolveDocumentKindTests(unittest.TestCase):
    def test_mime_type_wins_over_the_extension(self):
        self.assertEqual(resolve_document_kind("weird.xyz", "text/plain"), "txt")

    def test_mime_type_is_case_and_space_insensitive(self):
        self.assertEqual(resolve_document_kind("a.txt", "  TEXT/PLAIN  "), "txt")

    def test_every_allow_listed_mime_type_resolves(self):
        for mime, expected in ALLOWED_MIME_TYPES.items():
            self.assertEqual(
                resolve_document_kind(f"a.{expected}", mime), expected, mime
            )

    def test_markdown_extension_maps_onto_md(self):
        self.assertEqual(resolve_document_kind("notes.markdown", ""), "md")
        self.assertEqual(resolve_document_kind("notes.md", ""), "md")

    def test_extension_is_used_when_the_mime_is_unknown(self):
        self.assertEqual(resolve_document_kind("paper.pdf", "application/x-thing"), "pdf")
        self.assertEqual(resolve_document_kind("paper.docx", ""), "docx")

    def test_an_unknown_type_is_rejected(self):
        with self.assertRaises(DocumentProcessingError) as ctx:
            resolve_document_kind("archive.zip", "application/zip")
        message = str(ctx.exception)
        self.assertIn("unsupported file type", message)
        self.assertIn("application/zip", message)
        self.assertIn("archive.zip", message)

    def test_a_rejection_names_the_unknown_inputs(self):
        with self.assertRaises(DocumentProcessingError) as ctx:
            resolve_document_kind("", "")
        self.assertIn("unknown mime", str(ctx.exception))
        self.assertIn("unknown filename", str(ctx.exception))


# ── Extraction ──────────────────────────────────────────────────────────


class ExtractTextTests(unittest.TestCase):
    def test_plain_text_and_markdown_decode_as_utf8(self):
        self.assertEqual(extract_text("a.txt", "text/plain", b"hello"), "hello")
        self.assertEqual(extract_text("a.md", "text/markdown", b"# hi"), "# hi")

    def test_undecodable_bytes_are_replaced_not_raised(self):
        self.assertIn("�", extract_text("a.txt", "text/plain", b"ok\xffbad"))

    def test_pdf_dispatch_joins_the_page_texts(self):
        with _patched_modules(pypdf=_fake_pypdf(["page one", "page two"])):
            text = extract_text("a.pdf", "application/pdf", b"%PDF-1.4")
        self.assertEqual(text, "page one\npage two")

    def test_pdf_pages_without_a_text_layer_become_empty(self):
        with _patched_modules(pypdf=_fake_pypdf([None, "only this"])):
            text = extract_text("a.pdf", "application/pdf", b"%PDF-1.4")
        self.assertEqual(text, "only this")

    def test_a_scanned_pdf_is_rejected(self):
        with _patched_modules(pypdf=_fake_pypdf([None, "", "   "])):
            with self.assertRaises(DocumentProcessingError) as ctx:
                extract_text("scan.pdf", "application/pdf", b"%PDF-1.4")
        self.assertEqual(str(ctx.exception), SCANNED_PDF_ERROR)

    def test_docx_dispatch_reads_the_raw_text_value(self):
        with _patched_modules(mammoth=_fake_mammoth("  docx body  ")):
            self.assertEqual(
                extract_text("a.docx", DOCX_MIME, b"PK"), "docx body"
            )

    def test_an_empty_docx_yields_the_empty_string(self):
        with _patched_modules(mammoth=_fake_mammoth(None)):
            self.assertEqual(extract_text("a.docx", DOCX_MIME, b"PK"), "")

    def test_the_kind_check_runs_before_any_backend(self):
        with self.assertRaises(DocumentProcessingError):
            extract_text("a.exe", "application/x-msdownload", b"MZ")


# ── Chunking edges ──────────────────────────────────────────────────────


class SplitChunkEdgeTests(unittest.TestCase):
    def test_blank_segments_are_skipped(self):
        text = "\n\n\n" + ("word " * 300) + "\n\n\n   \n\n" + ("other " * 300)
        chunks = split_chunks(text, chunk_size=200)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.strip(), "a blank chunk survived")
            self.assertLessEqual(len(chunk), 200)

    def test_a_trailing_blank_buffer_is_not_emitted(self):
        # The separator split yields a final whitespace-only piece; it must not
        # become a chunk of its own.
        text = "alpha beta gamma delta. " * 60 + "\n\n   \n\n"
        chunks = split_chunks(text, chunk_size=120)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertTrue(chunk.strip())

    def test_text_of_exactly_the_chunk_size_is_one_chunk(self):
        text = "a" * CHUNK_SIZE
        self.assertEqual(split_chunks(text), [text])

    def test_an_overlap_larger_than_the_chunk_size_is_clamped(self):
        text = "z" * (CHUNK_SIZE * 2)
        chunks = split_chunks(text, chunk_size=50, overlap=500)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 50)

    def test_a_zero_overlap_is_honoured(self):
        text = "b" * 40
        chunks = split_chunks(text, chunk_size=10, overlap=0)
        self.assertEqual(chunks, ["b" * 10] * 4)

    def test_negative_overlap_is_clamped_to_zero(self):
        chunks = split_chunks("c" * 30, chunk_size=10, overlap=-5)
        self.assertEqual(chunks, ["c" * 10] * 3)

    def test_the_chunk_cap_is_always_applied(self):
        text = "q" * (CHUNK_SIZE * (MAX_CHUNKS_PER_DOCUMENT + 5))
        self.assertEqual(len(split_chunks(text)), MAX_CHUNKS_PER_DOCUMENT)

    def test_whitespace_only_text_yields_nothing(self):
        self.assertEqual(split_chunks("\t\n  \r"), [])
        self.assertEqual(split_chunks(None), [])


# ── Pipeline doubles ────────────────────────────────────────────────────


class _RecordingStore:
    """Records the ordered pipeline calls against an in-memory state."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] = []
        self.status = "uploaded"
        self.error = ""
        self.chunk_count = 0
        self.ensure_vec_index_result = False
        self.ensure_vec_index_calls: list[int] = []

    async def run_async(self, fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    def update_document_status(
        self, *, user_id, document_id, status, error="", chunk_count=None
    ):
        self.calls.append(("status", status))
        self.status = status
        if error:
            self.error = error
        if chunk_count is not None:
            self.chunk_count = chunk_count
        return {"status": status}

    def ensure_vec_index(self, dimension):
        self.calls.append(("ensure_vec_index", dimension))
        self.ensure_vec_index_calls.append(dimension)
        return self.ensure_vec_index_result

    def persist_chunks(self, *, user_id, document_id, chunks, embeddings):
        self.calls.append(("persist_chunks", len(chunks)))
        self.chunks = list(chunks)
        self.embeddings = list(embeddings)
        return len(chunks)


class _RecordingEmbedder:
    def __init__(self, dimension=4, error: Exception | None = None):
        self.dimension = dimension
        self.embedded: list[list[str]] = []
        self._error = error

    async def embed(self, texts, user_identity=None):
        if self._error is not None:
            raise self._error
        self.embedded.append(list(texts))
        return [[float(len(text))] * self.dimension for text in texts]


class PipelineFixture(unittest.TestCase):
    def setUp(self):
        self.store = _RecordingStore()
        self.embedder = _RecordingEmbedder()
        self.pipeline = DocumentPipeline(self.store, self.embedder)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(self.pipeline._executor.shutdown, wait=False)

    def _process(self, data=b"some plain study notes", filename="a.txt", mime="text/plain"):
        return asyncio.run(
            self.pipeline.process(
                user_id="u1",
                document_id="doc-1",
                filename=filename,
                mime_type=mime,
                data=data,
            )
        )

    def _write(self, name, text):
        path = os.path.join(self._tmp.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path


class RunAsyncTests(PipelineFixture):
    def test_positional_arguments_are_forwarded(self):
        result = asyncio.run(self.pipeline.run_async(lambda a, b: a + b, 2, 3))
        self.assertEqual(result, 5)

    def test_keyword_arguments_are_forwarded(self):
        result = asyncio.run(
            self.pipeline.run_async(
                lambda a, b=0: (a, b), 2, b=7
            )
        )
        self.assertEqual(result, (2, 7))

    def test_the_work_runs_off_the_event_loop_thread(self):
        import threading

        seen: list[int] = []

        def capture():
            seen.append(threading.get_ident())
            return threading.get_ident()

        loop_thread = threading.get_ident()
        worker = asyncio.run(self.pipeline.run_async(capture))
        self.assertNotEqual(worker, loop_thread)
        self.assertEqual(seen, [worker])


class ProcessTests(PipelineFixture):
    def test_a_good_document_reaches_ready_with_its_chunks(self):
        self._process(data=b"Caching stores data so repeated reads are fast.")
        self.assertEqual(self.store.status, "ready")
        self.assertEqual(self.store.error, "")
        self.assertGreater(self.store.chunk_count, 0)
        self.assertEqual(len(self.store.chunks), self.store.chunk_count)
        self.assertEqual(len(self.store.embeddings), self.store.chunk_count)
        self.assertEqual(self.embedder.dimension, self.store.ensure_vec_index_calls[0])

    def test_the_status_sequence_is_processing_then_ready(self):
        self._process()
        statuses = [call for call in self.store.calls if call[0] == "status"]
        self.assertEqual(statuses, [("status", "processing"), ("status", "ready")])

    def test_the_vector_index_is_ensured_before_the_chunks_are_persisted(self):
        self._process()
        order = [call[0] for call in self.store.calls]
        self.assertEqual(
            order, ["status", "ensure_vec_index", "persist_chunks", "status"]
        )

    def test_a_document_with_no_extractable_text_fails(self):
        self._process(data=b"   \n\t  ")
        self.assertEqual(self.store.status, "failed")
        self.assertIn("no extractable text", self.store.error)
        self.assertEqual(self.store.chunks, [])
        # Nothing was embedded, so no embedding cost was paid.
        self.assertEqual(self.embedder.embedded, [])

    def test_an_unsupported_type_fails_with_the_resolution_error(self):
        self._process(data=b"MZ", filename="a.exe", mime="application/x-msdownload")
        self.assertEqual(self.store.status, "failed")
        self.assertIn("unsupported file type", self.store.error)

    def test_a_failed_embedding_call_marks_the_document_failed(self):
        self.embedder = _RecordingEmbedder(error=RuntimeError("provider down"))
        self.pipeline = DocumentPipeline(self.store, self.embedder)
        self._process()
        self.assertEqual(self.store.status, "failed")
        self.assertIn("provider down", self.store.error)

    def test_a_long_document_is_clipped_to_the_hard_character_ceiling(self):
        body = "HEAD-SENTINEL " + ("filler " * 40_000) + " TAIL-SENTINEL"
        self.assertGreater(len(body), MAX_DOCUMENT_TEXT_CHARS)
        self._process(data=body.encode("utf-8"))
        self.assertEqual(self.store.status, "ready")
        everything = "\n".join(self.store.chunks)
        self.assertIn("HEAD-SENTINEL", everything)
        self.assertNotIn("TAIL-SENTINEL", everything)

    def test_the_stored_error_is_clipped_to_500_characters(self):
        self.embedder = _RecordingEmbedder(
            error=RuntimeError("e" * 5000)
        )
        self.pipeline = DocumentPipeline(self.store, self.embedder)
        self._process()
        self.assertEqual(self.store.status, "failed")
        self.assertLessEqual(len(self.store.error), 500)

    def test_a_multi_page_pdf_is_ingested(self):
        with _patched_modules(pypdf=_fake_pypdf(["alpha", "beta", "gamma"])):
            self._process(
                data=b"%PDF-1.4", filename="paper.pdf", mime="application/pdf"
            )
        self.assertEqual(self.store.status, "ready")
        self.assertIn("alpha", " ".join(self.store.chunks))

    def test_a_scanned_pdf_fails_the_document(self):
        with _patched_modules(pypdf=_fake_pypdf([None, None])):
            self._process(
                data=b"%PDF-1.4", filename="scan.pdf", mime="application/pdf"
            )
        self.assertEqual(self.store.status, "failed")
        self.assertEqual(self.store.error, SCANNED_PDF_ERROR)

    def test_processing_a_document_that_never_existed_still_reports_failed(self):
        # The store's status update targets a row that is not there; the
        # pipeline must not raise, it only records what it can.
        self._process(data=b"\x00\x01 binary junk")
        self.assertIn(self.store.status, ("ready", "failed"))


class ScheduleTests(PipelineFixture):
    def test_schedule_returns_a_task_that_processes_the_document(self):
        async def _drive():
            task = self.pipeline.schedule(
                user_id="u1",
                document_id="doc-1",
                filename="a.txt",
                mime_type="text/plain",
                data=b"scheduled content",
            )
            self.assertIsInstance(task, asyncio.Task)
            self.assertFalse(task.done())
            await task
            return task

        task = asyncio.run(_drive())
        self.assertTrue(task.done())
        self.assertEqual(task.exception(), None)
        self.assertEqual(self.store.status, "ready")
        self.assertIn("scheduled content", " ".join(self.store.chunks))

    def test_a_scheduled_failure_is_swallowed_inside_the_task(self):
        async def _drive():
            task = self.pipeline.schedule(
                user_id="u1",
                document_id="doc-1",
                filename="a.exe",
                mime_type="application/x-msdownload",
                data=b"MZ",
            )
            await task
            return task

        task = asyncio.run(_drive())
        self.assertTrue(task.done())
        self.assertIsNone(task.exception())
        self.assertEqual(self.store.status, "failed")


class RagFlagGateTests(unittest.TestCase):
    """The pipeline is only ever reached when ``STUDY_ENABLE_RAG_V1`` is on."""

    def test_documents_router_refuses_before_the_pipeline_is_reachable(self):
        from fastapi import HTTPException

        from app.routers import documents
        from app.config import get_settings

        settings = get_settings()
        original = settings.enable_rag_v1
        settings.enable_rag_v1 = False
        try:
            with self.assertRaises(HTTPException) as ctx:
                documents._require_rag_enabled()
        finally:
            settings.enable_rag_v1 = original
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("STUDY_ENABLE_RAG_V1=false", str(ctx.exception.detail))

    def test_the_gate_is_a_noop_when_the_flag_is_on(self):
        from app.routers import documents
        from app.config import get_settings

        settings = get_settings()
        original = settings.enable_rag_v1
        settings.enable_rag_v1 = True
        try:
            self.assertIsNone(documents._require_rag_enabled())
        finally:
            settings.enable_rag_v1 = original


if __name__ == "__main__":
    unittest.main()