"""Tests for the documents router: caps, mime allowlist, CRUD."""

import asyncio
import io
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import documents
from app.services.document_pipeline import DocumentPipeline
from app.services.document_store import DocumentStore


class _FakePipeline:
    """Records scheduled processing without running it."""

    def __init__(self):
        self.scheduled = []

    def schedule(self, *, user_id, document_id, filename, mime_type, data):
        self.scheduled.append(
            {
                "user_id": user_id,
                "document_id": document_id,
                "filename": filename,
                "mime_type": mime_type,
                "data": data,
            }
        )
        return None


def _make_client(store, pipeline):
    documents.init(
        document_store=store,
        pipeline=pipeline,
        learning_store=None,
    )
    application = FastAPI()
    application.include_router(documents.router)
    application.dependency_overrides[require_auth] = lambda: {
        "user": "u1",
        "provider": "google",
    }
    return TestClient(application)


class DocumentsRouterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "router.db")
        self.store = DocumentStore(self.db_path)
        self.pipeline = _FakePipeline()
        self.client = _make_client(self.store, self.pipeline)
        self._settings_patches = []

    def tearDown(self):
        for patcher in self._settings_patches:
            patcher.stop()
        self.store._executor.shutdown(wait=False)
        self._tmp.cleanup()

    def _patch_settings(self, **overrides):
        from app.config import get_settings

        settings = get_settings()
        for key, value in overrides.items():
            setattr(settings, key, value)
        return settings

    def test_upload_returns_202_and_schedules_pipeline(self):
        self._patch_settings(enable_rag_v1=True)
        response = self.client.post(
            "/api/documents",
            files={
                "file": ("notes.txt", b"hello world", "text/plain")
            },
        )
        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertIn("document_id", payload)
        self.assertEqual(payload["status"], "processing")
        self.assertEqual(len(self.pipeline.scheduled), 1)
        self.assertEqual(
            self.pipeline.scheduled[0]["filename"], "notes.txt"
        )
        self.assertEqual(
            self.pipeline.scheduled[0]["data"], b"hello world"
        )
        # The document row exists and is processing.
        detail = self.client.get(
            f"/api/documents/{payload['document_id']}"
        )
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["status"], "processing")

    def test_upload_rejects_disallowed_mime(self):
        self._patch_settings(enable_rag_v1=True)
        response = self.client.post(
            "/api/documents",
            files={
                "file": ("payload.zip", b"PK\x03\x04", "application/zip")
            },
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(self.pipeline.scheduled, [])

    def test_upload_rejects_disallowed_extension(self):
        self._patch_settings(enable_rag_v1=True)
        response = self.client.post(
            "/api/documents",
            files={
                "file": ("script.exe", b"MZ\x90\x00", "application/octet-stream")
            },
        )
        self.assertEqual(response.status_code, 415)

    def test_upload_allows_markdown_and_docx(self):
        self._patch_settings(enable_rag_v1=True)
        for filename, mime, data in [
            ("readme.md", "text/markdown", b"# title"),
            ("readme.markdown", "text/markdown", b"# title"),
            (
                "doc.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                b"PK\x03\x04docx",
            ),
            ("notes.pdf", "application/pdf", b"%PDF-1.4"),
        ]:
            response = self.client.post(
                "/api/documents",
                files={"file": (filename, data, mime)},
            )
            self.assertEqual(response.status_code, 202, filename)

    def test_upload_rejects_oversize_file(self):
        self._patch_settings(enable_rag_v1=True)
        with patch(
            "app.routers.documents.MAX_FILE_SIZE", 1024
        ):
            response = self.client.post(
                "/api/documents",
                files={
                    "file": ("big.txt", b"x" * 2048, "text/plain")
                },
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.pipeline.scheduled, [])

    def test_upload_rejects_empty_file(self):
        self._patch_settings(enable_rag_v1=True)
        response = self.client.post(
            "/api/documents",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_enforces_daily_cap(self):
        self._patch_settings(enable_rag_v1=True, documents_daily_cap=1)
        first = self.client.post(
            "/api/documents",
            files={"file": ("a.txt", b"a", "text/plain")},
        )
        self.assertEqual(first.status_code, 202)
        second = self.client.post(
            "/api/documents",
            files={"file": ("b.txt", b"b", "text/plain")},
        )
        self.assertEqual(second.status_code, 429)
        self.assertEqual(len(self.pipeline.scheduled), 1)

    def test_endpoints_require_rag_flag(self):
        self._patch_settings(enable_rag_v1=False)
        response = self.client.post(
            "/api/documents",
            files={"file": ("a.txt", b"a", "text/plain")},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.get("/api/documents").status_code, 503)
        self.assertEqual(
            self.client.get("/api/documents/anything").status_code,
            503,
        )
        self.assertEqual(
            self.client.delete("/api/documents/anything").status_code,
            503,
        )

    def test_list_documents(self):
        self._patch_settings(enable_rag_v1=True)
        for name in ["a.txt", "b.txt"]:
            self.client.post(
                "/api/documents",
                files={"file": (name, b"content", "text/plain")},
            )
        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["documents"]), 2)
        titles = {doc["title"] for doc in payload["documents"]}
        self.assertEqual(titles, {"a", "b"})

    def test_get_document_404_for_missing(self):
        self._patch_settings(enable_rag_v1=True)
        response = self.client.get("/api/documents/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_delete_document(self):
        self._patch_settings(enable_rag_v1=True)
        uploaded = self.client.post(
            "/api/documents",
            files={"file": ("a.txt", b"a", "text/plain")},
        )
        document_id = uploaded.json()["document_id"]
        response = self.client.delete(f"/api/documents/{document_id}")
        self.assertEqual(response.status_code, 204)
        self.assertEqual(
            self.client.get(f"/api/documents/{document_id}").status_code,
            404,
        )
        self.assertEqual(
            self.client.delete(f"/api/documents/{document_id}").status_code,
            404,
        )

    def test_other_users_documents_are_invisible(self):
        self._patch_settings(enable_rag_v1=True)
        uploaded = self.client.post(
            "/api/documents",
            files={"file": ("a.txt", b"a", "text/plain")},
        )
        document_id = uploaded.json()["document_id"]
        # Simulate another user's store view: the row belongs
        # to u1, so a different user must not see it.
        other_store = DocumentStore(self.db_path)
        try:
            self.assertIsNone(
                other_store.get_document("u2", document_id)
            )
        finally:
            other_store._executor.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
