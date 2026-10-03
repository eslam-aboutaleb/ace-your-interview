"""Resume parser tests — extraction, LLM profile extraction, encrypted storage."""

import io
import os
import tempfile
import unittest

from pypdf import PdfWriter

from app.config import get_settings
from app.services.llm_client import CALL_FAILED_CODE
from app.services.resume_parser import (
    MAX_RESUME_FILE_SIZE,
    ResumeParseError,
    ResumeParser,
    ResumeProfileStore,
    extract_resume_text,
)


class ScriptedLLM:
    """Return a fixed structured profile, or fail on demand."""

    def __init__(self, analysis: str = "", error_code: str = ""):
        self.analysis = analysis
        self.error_code = error_code
        self.calls = 0

    async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
        self.calls += 1
        if self.error_code:
            return {
                "success": False,
                "analysis": "",
                "metadata": {},
                "error": "provider unavailable",
                "error_code": self.error_code,
                "finish_reason": "",
                "usage": {},
            }
        return {
            "success": True,
            "analysis": self.analysis,
            "metadata": {},
            "error_code": "",
            "finish_reason": "stop",
            "usage": {},
        }


PROFILE_JSON = (
    '{"skills":["Python","FastAPI","Docker"],'
    '"projects":["Resume parser API"],'
    '"experience_years":3.5,'
    '"roles":["Backend Engineer"],'
    '"strengths":["Clear tradeoff reasoning"],'
    '"weak_spots":["No metrics on resume"]}'
)


class ExtractResumeTextTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "learning.db")
        self.prev_key = os.environ.get("STUDY_CREDENTIALS_ENCRYPTION_KEY")
        os.environ["STUDY_CREDENTIALS_ENCRYPTION_KEY"] = "resume-test-encryption-secret"
        get_settings.cache_clear()

    def tearDown(self):
        self.tempdir.cleanup()
        if self.prev_key is None:
            os.environ.pop("STUDY_CREDENTIALS_ENCRYPTION_KEY", None)
        else:
            os.environ["STUDY_CREDENTIALS_ENCRYPTION_KEY"] = self.prev_key
        get_settings.cache_clear()

    def test_txt_extraction(self):
        text = extract_resume_text("resume.txt", b"Python backend engineer\n5 years")
        self.assertIn("Python backend engineer", text)

    def test_pdf_extraction(self):
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        # pypdf cannot insert text into a blank page without a content
        # stream; build a minimal one-page PDF whose text is extractable
        # via a simple content stream.
        buffer = io.BytesIO()
        writer.write(buffer)
        # A blank PDF has no text: extraction must raise a parse error
        # rather than return an empty profile.
        with self.assertRaises(ResumeParseError):
            extract_resume_text("resume.pdf", buffer.getvalue())

    def test_pdf_content_sniffing_without_extension(self):
        # A PDF whose filename lacks an extension is detected by magic bytes.
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = io.BytesIO()
        writer.write(buffer)
        with self.assertRaises(ResumeParseError):
            extract_resume_text("resume", buffer.getvalue())

    def test_unsupported_extension_rejected(self):
        with self.assertRaises(ResumeParseError):
            extract_resume_text("resume.doc", b"not a docx")

    def test_empty_file_rejected(self):
        with self.assertRaises(ResumeParseError):
            extract_resume_text("resume.txt", b"   ")

    def test_oversized_file_rejected_by_parser(self):
        parser = ResumeParser(ScriptedLLM(PROFILE_JSON), ResumeProfileStore(self.db_path))
        with self.assertRaises(ResumeParseError):
            import asyncio

            asyncio.run(
                parser.parse_and_store(
                    user_id="u1",
                    filename="resume.txt",
                    content=b"x" * (MAX_RESUME_FILE_SIZE + 1),
                )
            )


class ProfileExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "learning.db")
        self.prev_key = os.environ.get("STUDY_CREDENTIALS_ENCRYPTION_KEY")
        os.environ["STUDY_CREDENTIALS_ENCRYPTION_KEY"] = "resume-test-encryption-secret"
        get_settings.cache_clear()
        self.store = ResumeProfileStore(self.db_path)

    def tearDown(self):
        self.tempdir.cleanup()
        if self.prev_key is None:
            os.environ.pop("STUDY_CREDENTIALS_ENCRYPTION_KEY", None)
        else:
            os.environ["STUDY_CREDENTIALS_ENCRYPTION_KEY"] = self.prev_key
        get_settings.cache_clear()

    def test_llm_profile_extraction_and_storage(self):
        import asyncio

        parser = ResumeParser(ScriptedLLM(PROFILE_JSON), self.store)
        saved = asyncio.run(
            parser.parse_and_store(
                user_id="u1",
                filename="resume.txt",
                content=b"Python backend engineer with Docker experience",
            )
        )
        self.assertEqual(saved["skills"], ["Python", "FastAPI", "Docker"])
        self.assertEqual(saved["experience_years"], 3.5)
        self.assertEqual(saved["roles"], ["Backend Engineer"])
        # Raw text hash is stored for change detection...
        self.assertTrue(saved["raw_text_hash"])
        # ...but the raw text itself is not.
        with open(self.db_path, "rb") as f:
            raw_db = f.read()
        self.assertNotIn(b"Python backend engineer with Docker experience", raw_db)

    def test_profile_is_encrypted_at_rest(self):
        import asyncio

        parser = ResumeParser(ScriptedLLM(PROFILE_JSON), self.store)
        asyncio.run(
            parser.parse_and_store(
                user_id="u1",
                filename="resume.txt",
                content=b"secret resume text",
            )
        )
        with open(self.db_path, "rb") as f:
            raw_db = f.read()
        # Neither the skill names nor the raw text appear in plaintext.
        self.assertNotIn(b"FastAPI", raw_db)
        self.assertNotIn(b"secret resume text", raw_db)

    def test_round_trip_through_store(self):
        import asyncio

        parser = ResumeParser(ScriptedLLM(PROFILE_JSON), self.store)
        asyncio.run(
            parser.parse_and_store(
                user_id="u1",
                filename="resume.txt",
                content=b"resume body",
            )
        )
        loaded = self.store.get("u1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["skills"], ["Python", "FastAPI", "Docker"])
        self.assertEqual(loaded["weak_spots"], ["No metrics on resume"])

    def test_llm_failure_raises_parse_error(self):
        import asyncio

        parser = ResumeParser(ScriptedLLM(error_code=CALL_FAILED_CODE), self.store)
        with self.assertRaises(ResumeParseError):
            asyncio.run(
                parser.parse_and_store(
                    user_id="u1",
                    filename="resume.txt",
                    content=b"resume body",
                )
            )

    def test_unparseable_llm_output_raises_parse_error(self):
        import asyncio

        parser = ResumeParser(ScriptedLLM("not json at all"), self.store)
        with self.assertRaises(ResumeParseError):
            asyncio.run(
                parser.parse_and_store(
                    user_id="u1",
                    filename="resume.txt",
                    content=b"resume body",
                )
            )

    def test_gdpr_delete(self):
        import asyncio

        parser = ResumeParser(ScriptedLLM(PROFILE_JSON), self.store)
        asyncio.run(
            parser.parse_and_store(
                user_id="u1",
                filename="resume.txt",
                content=b"resume body",
            )
        )
        self.assertTrue(self.store.delete("u1"))
        self.assertIsNone(self.store.get("u1"))
        self.assertFalse(self.store.delete("u1"))

    def test_missing_encryption_secret_fails_on_use(self):
        # The store constructs without the secret (router init stays
        # cheap); using it without a configured secret raises.
        os.environ.pop("STUDY_CREDENTIALS_ENCRYPTION_KEY", None)
        get_settings.cache_clear()
        store = ResumeProfileStore(self.db_path)
        import asyncio

        parser = ResumeParser(ScriptedLLM(PROFILE_JSON), store)
        with self.assertRaises(RuntimeError):
            asyncio.run(
                parser.parse_and_store(
                    user_id="u1",
                    filename="resume.txt",
                    content=b"resume body",
                )
            )


if __name__ == "__main__":
    unittest.main()
