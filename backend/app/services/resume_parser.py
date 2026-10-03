"""Resume parsing — text extraction + LLM structured profile extraction.

PII minimization: the raw resume text is never persisted. Only the
extracted :class:`ResumeProfile` (skills, projects, roles, …) and a
SHA-256 hash of the raw text (for change detection) are stored in
``resume_profiles``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings
from app.schemas.models import ResumeProfile
from app.services.llm_client import LLMClient
from app.services.llm_policy import raise_if_policy_blocked_result
from app.services.user_settings_store import resolve_credentials_encryption_secret

logger = logging.getLogger(__name__)

#: Upper bound on accepted resume payloads (8 MiB).
MAX_RESUME_FILE_SIZE = 8 * 1024 * 1024

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class ResumeParseError(Exception):
    """Raised when the resume cannot be parsed or extracted."""

    def __init__(self, detail: str, *, field: str = ""):
        super().__init__(detail)
        self.detail = detail
        self.field = field


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def extract_resume_text(filename: str, content: bytes) -> str:
    """Extract plain text from a PDF, DOCX, or TXT resume.

    The file type is resolved from the filename extension (with a
    MIME-type-free fallback to content sniffing for PDFs).
    """
    name = (filename or "").strip().lower()
    ext = ""
    if "." in name:
        ext = name.rsplit(".", 1)[-1]
        ext = f".{ext}"

    if ext == ".pdf" or (not ext and content[:5] == b"%PDF-"):
        return _extract_pdf(content)
    if ext == ".docx":
        return _extract_docx(content)
    if ext == ".txt" or not ext:
        return _extract_txt(content)
    raise ResumeParseError(
        f"Unsupported resume file type '{ext or 'unknown'}'. "
        "Upload a PDF, DOCX, or TXT file.",
        field="file",
    )


def _extract_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ResumeParseError("PDF parsing is unavailable on this server.") from exc

    import io

    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [
            (page.extract_text() or "") for page in reader.pages
        ]
    except Exception as exc:
        raise ResumeParseError(
            f"Could not read the PDF resume: {exc}", field="file"
        ) from exc

    text = "\n".join(page for page in pages if page.strip())
    if not text.strip():
        raise ResumeParseError(
            "The PDF resume contains no extractable text. "
            "Upload a text-based PDF (not a scanned image) or paste your resume manually.",
            field="file",
        )
    return text


def _extract_docx(content: bytes) -> str:
    try:
        import mammoth
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ResumeParseError("DOCX parsing is unavailable on this server.") from exc

    import io

    try:
        result = mammoth.extract_text(io.BytesIO(content))
    except Exception as exc:
        raise ResumeParseError(
            f"Could not read the DOCX resume: {exc}", field="file"
        ) from exc

    text = (result.value or "").strip()
    if not text:
        raise ResumeParseError(
            "The DOCX resume contains no extractable text.",
            field="file",
        )
    return text


def _extract_txt(content: bytes) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1", errors="replace")
    if not text.strip():
        raise ResumeParseError("The resume file is empty.", field="file")
    return text


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class ResumeProfileStore:
    """SQLite persistence for extracted resume profiles.

    PII minimization: the raw resume text is never written — only
    the extracted profile, encrypted at rest with the same Fernet
    derivation as ``UserSettingsStore`` (the credentials encryption
    secret), plus the SHA-256 hash of the raw text for change
    detection.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Built on first use: constructing the store must not require
        # the encryption secret (the app resolves it at startup), so
        # router init stays cheap and feature-independent.
        self._cipher: Fernet | None = None
        self._init_db()

    def _get_cipher(self) -> Fernet:
        if self._cipher is None:
            secret = resolve_credentials_encryption_secret(get_settings())
            digest = hashlib.sha256(secret.encode("utf-8")).digest()
            key = base64.urlsafe_b64encode(digest)
            self._cipher = Fernet(key)
        return self._cipher

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_blob TEXT NOT NULL,
                    raw_text_hash TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _encrypt(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        return self._get_cipher().encrypt(raw.encode("utf-8")).decode("utf-8")

    def _decrypt(self, blob: str) -> dict[str, Any]:
        try:
            raw = self._get_cipher().decrypt(blob.encode("utf-8")).decode("utf-8")
            data = json.loads(raw)
        except (InvalidToken, json.JSONDecodeError, ValueError):
            logger.warning("Resume profile blob failed to decrypt; treating as absent")
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _dumps(items: list[str]) -> str:
        return json.dumps([str(x) for x in items], ensure_ascii=True)

    @staticmethod
    def _clean_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(x).strip() for x in value if str(x).strip()]

    def _row_to_profile(self, row: sqlite3.Row) -> dict[str, Any]:
        data = self._decrypt(row["profile_blob"])
        return {
            "user_id": row["user_id"],
            "skills": self._clean_list(data.get("skills")),
            "projects": self._clean_list(data.get("projects")),
            "experience_years": (
                float(data["experience_years"])
                if isinstance(data.get("experience_years"), (int, float))
                and not isinstance(data.get("experience_years"), bool)
                else None
            ),
            "roles": self._clean_list(data.get("roles")),
            "strengths": self._clean_list(data.get("strengths")),
            "weak_spots": self._clean_list(data.get("weak_spots")),
            "raw_text_hash": row["raw_text_hash"],
            "updated_at": row["updated_at"],
        }

    def save(self, user_id: str, profile: ResumeProfile) -> dict[str, Any]:
        now = _utc_now_iso()
        payload = {
            "skills": [str(x) for x in profile.skills],
            "projects": [str(x) for x in profile.projects],
            "experience_years": profile.experience_years,
            "roles": [str(x) for x in profile.roles],
            "strengths": [str(x) for x in profile.strengths],
            "weak_spots": [str(x) for x in profile.weak_spots],
        }
        blob = self._encrypt(payload)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO resume_profiles(
                        user_id, profile_blob, raw_text_hash, updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        profile_blob = excluded.profile_blob,
                        raw_text_hash = excluded.raw_text_hash,
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id,
                        blob,
                        profile.raw_text_hash or "",
                        now,
                    ),
                )
        saved = self.get(user_id)
        assert saved is not None
        return saved

    def get(self, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM resume_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_profile(row)

    def delete(self, user_id: str) -> bool:
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM resume_profiles WHERE user_id = ?",
                    (user_id,),
                )
        return bool(cursor.rowcount)


class ResumeParser:
    """Extract a structured :class:`ResumeProfile` from resume text.

    Uses the shared :class:`LLMClient` with a strict JSON output
    contract. A deterministic token-overlap fallback is *not* used for
    the profile itself: per the plan, LLM extraction failure surfaces
    as a 422 so the user can paste their resume manually instead.
    """

    def __init__(self, llm: LLMClient, store: ResumeProfileStore):
        self.llm = llm
        self.store = store

    async def parse_and_store(
        self,
        *,
        user_id: str,
        filename: str,
        content: bytes,
        llm_config: Any = None,
        user_identity: dict | None = None,
    ) -> dict[str, Any]:
        if len(content) > MAX_RESUME_FILE_SIZE:
            raise ResumeParseError(
                f"Resume file exceeds the "
                f"{MAX_RESUME_FILE_SIZE // (1024 * 1024)} MB limit.",
                field="file",
            )

        text = extract_resume_text(filename, content)
        profile = await self.extract_profile(
            text,
            llm_config=llm_config,
            user_identity=user_identity,
        )
        profile.raw_text_hash = _sha256_hex(text)
        return self.store.save(user_id, profile)

    async def extract_profile(
        self,
        text: str,
        *,
        llm_config: Any = None,
        user_identity: dict | None = None,
    ) -> ResumeProfile:
        prompt = self._build_prompt(text)
        result = await self.llm.completion(
            prompt,
            llm_config,
            user_identity=user_identity,
            structured=True,
        )
        raise_if_policy_blocked_result(result)

        payload: dict[str, Any] = {}
        if result.get("success"):
            payload = self._extract_json(result.get("analysis", ""))

        if not payload:
            raise ResumeParseError(
                "Could not extract a structured profile from the resume. "
                "Paste your resume summary manually instead.",
                field="resume",
            )

        return self._normalise_profile(payload)

    @staticmethod
    def _build_prompt(text: str) -> str:
        bounded = text[:24000]
        return f"""You are a resume parser. Extract a structured profile from the resume text below.

Return ONLY a JSON object with this exact shape:
{{
  "skills": ["skill1", "skill2"],
  "projects": ["project1", "project2"],
  "experience_years": 3.5,
  "roles": ["role1", "role2"],
  "strengths": ["strength1", "strength2"],
  "weak_spots": ["weak_spot1", "weak_spot2"]
}}

Rules:
- skills: concrete technologies, languages, frameworks, tools (e.g. "Python", "FastAPI", "Docker").
- projects: short project names or one-line descriptions.
- experience_years: total years of professional experience as a number, or null when not stated.
- roles: job titles held, most recent last.
- strengths: demonstrated capabilities evidenced by the resume.
- weak_spots: areas the resume does NOT evidence (missing fundamentals, no metrics, shallow scope).
- Only extract what the resume states. Do not invent skills, roles, or experience.
- Keep every string concise (<= 80 characters).

Resume text:
{bounded}
"""

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.DOTALL)
        if fenced:
            try:
                data = json.loads(fenced.group(1))
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _clean_string_list(value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            cleaned = str(item).strip()
            if not cleaned:
                continue
            if cleaned.lower() in {"n/a", "none", "null", "-"}:
                continue
            out.append(cleaned[:80])
            if len(out) >= limit:
                break
        return out

    @classmethod
    def _normalise_profile(cls, payload: dict[str, Any]) -> ResumeProfile:
        experience_years: float | None = None
        raw_years = payload.get("experience_years")
        if isinstance(raw_years, (int, float)) and not isinstance(raw_years, bool):
            experience_years = max(0.0, min(60.0, float(raw_years)))
        elif isinstance(raw_years, str):
            match = re.search(r"\d+(?:\.\d+)?", raw_years)
            if match:
                experience_years = max(0.0, min(60.0, float(match.group(0))))

        return ResumeProfile(
            skills=cls._clean_string_list(payload.get("skills"), 200),
            projects=cls._clean_string_list(payload.get("projects"), 100),
            experience_years=experience_years,
            roles=cls._clean_string_list(payload.get("roles"), 50),
            strengths=cls._clean_string_list(payload.get("strengths"), 50),
            weak_spots=cls._clean_string_list(payload.get("weak_spots"), 50),
            raw_text_hash="",
            updated_at=_utc_now_iso(),
        )
