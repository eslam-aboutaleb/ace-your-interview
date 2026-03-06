"""Parse owner-handbook markdown files into structured topics."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.schemas.models import TopicDetail, TopicSummary


class DocParser:
    """Reads and caches all owner-handbook .md files."""

    def __init__(self, docs_path: Optional[str] = None):
        configured = docs_path or get_settings().docs_path
        self.docs_path = self._resolve_docs_path(configured)
        self._topics: dict[str, TopicDetail] = {}
        self._load()

    # ── public API ──────────────────────────────────────────
    def list_topics(self) -> list[TopicSummary]:
        summaries: list[TopicSummary] = []
        for tid, td in sorted(self._topics.items()):
            summaries.append(
                TopicSummary(
                    id=tid,
                    title=td.title,
                    description=td.description,
                    section_count=len(td.sections),
                    estimated_questions=max(3, len(td.sections) * 2),
                )
            )
        return summaries

    def get_topic(self, topic_id: str) -> Optional[TopicDetail]:
        return self._topics.get(topic_id)

    def get_topic_content(self, topic_id: str) -> str:
        """Return raw markdown content for LLM prompt."""
        topic = self._topics.get(topic_id)
        return topic.raw_content if topic else ""

    # ── internals ───────────────────────────────────────────
    @staticmethod
    def _resolve_docs_path(configured_path: str) -> str:
        """Resolve docs path robustly for local and container runtimes."""
        p = Path(configured_path)
        if p.is_absolute():
            return str(p)

        # Prefer cwd-relative if it exists, otherwise backend-root relative.
        cwd_candidate = (Path.cwd() / p).resolve()
        if cwd_candidate.is_dir():
            return str(cwd_candidate)

        backend_root = Path(__file__).resolve().parents[2]
        backend_candidate = (backend_root / p).resolve()
        return str(backend_candidate)

    def _load(self):
        handbook_dir = os.path.join(self.docs_path, "owner-handbook")
        if not os.path.isdir(handbook_dir):
            # Fallback: maybe docs_path IS the owner-handbook dir
            if os.path.isdir(self.docs_path):
                handbook_dir = self.docs_path
            else:
                return

        for fname in sorted(os.listdir(handbook_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(handbook_dir, fname)
            topic_id = fname.replace(".md", "")
            raw = self._read_file(fpath)
            title = self._extract_title(raw, fname)
            description = self._extract_description(raw)
            sections = self._extract_sections(raw)
            self._topics[topic_id] = TopicDetail(
                id=topic_id,
                title=title,
                description=description,
                sections=sections,
                raw_content=raw,
            )

    @staticmethod
    def _read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if m:
            return m.group(1).strip()
        return fallback.replace(".md", "").replace("-", " ").title()

    @staticmethod
    def _extract_description(content: str) -> str:
        """First non-heading, non-empty paragraph after the title."""
        lines = content.split("\n")
        capture = False
        buf: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                if capture and buf:
                    break
                capture = True
                continue
            if capture and stripped:
                buf.append(stripped)
            elif capture and not stripped and buf:
                break
        return " ".join(buf)[:300]

    @staticmethod
    def _extract_sections(content: str) -> list[dict[str, str]]:
        sections: list[dict[str, str]] = []
        parts = re.split(r"^(#{2,3})\s+(.+)$", content, flags=re.MULTILINE)

        # parts = [pre, level, heading, body, level, heading, body, ...]
        i = 1
        while i < len(parts) - 2:
            heading = parts[i + 1].strip()
            body = parts[i + 2].strip() if i + 2 < len(parts) else ""
            sections.append({"heading": heading, "content": body[:12000]})
            i += 3

        return sections
