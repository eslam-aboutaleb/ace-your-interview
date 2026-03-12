"""Load static curriculum topics from the app catalog or a markdown override."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.schemas.models import TopicDetail, TopicSummary

_VALID_TRACKS = {"backend", "frontend", "system_design", "ai_stack"}
_VALID_LEVELS = {"junior", "mid", "senior"}
_DEFAULT_LEVELS = ["junior", "mid", "senior"]


class DocParser:
    """Read and cache static curriculum topics for the learning platform."""

    def __init__(
        self,
        docs_path: Optional[str] = None,
        curriculum_path: Optional[str] = None,
    ):
        self._topics: dict[str, TopicDetail] = {}
        self.source_kind = "curriculum_json"
        self.source_path = ""

        if docs_path:
            self.docs_path = self._resolve_markdown_path(docs_path)
            self.source_kind = "markdown_override"
            self.source_path = self.docs_path
            self._load_from_markdown_dir()
            return

        configured = curriculum_path or get_settings().curriculum_path
        self.curriculum_path = self._resolve_curriculum_path(configured)
        self.source_path = self.curriculum_path
        self._load_from_curriculum_json()

    def list_topics(
        self,
        track: Optional[str] = None,
        level: Optional[str] = None,
        q: Optional[str] = None,
    ) -> list[TopicSummary]:
        track_filter = self._normalise_track(track)
        level_filter = self._normalise_level(level)
        query = (q or "").strip().lower()
        summaries: list[TopicSummary] = []
        for tid, td in sorted(self._topics.items()):
            if track_filter and td.track != track_filter:
                continue
            if level_filter and level_filter not in td.levels:
                continue
            if query and query not in f"{td.title} {td.description}".lower():
                continue
            summaries.append(
                TopicSummary(
                    id=tid,
                    title=td.title,
                    description=td.description,
                    track=td.track,
                    levels=td.levels,
                    section_count=len(td.sections),
                    estimated_questions=max(3, len(td.sections) * 2),
                )
            )
        return summaries

    def get_topic(self, topic_id: str) -> Optional[TopicDetail]:
        return self._topics.get(topic_id)

    def get_topic_content(self, topic_id: str) -> str:
        topic = self._topics.get(topic_id)
        return topic.raw_content if topic else ""

    @staticmethod
    def _resolve_markdown_path(configured_path: str) -> str:
        p = Path(configured_path)
        if p.is_absolute():
            return str(p)

        cwd_candidate = (Path.cwd() / p).resolve()
        if cwd_candidate.exists():
            return str(cwd_candidate)

        backend_root = Path(__file__).resolve().parents[2]
        return str((backend_root / p).resolve())

    @staticmethod
    def _resolve_curriculum_path(configured_path: str) -> str:
        p = Path(configured_path)
        if p.is_absolute():
            return str(p)

        cwd_candidate = (Path.cwd() / p).resolve()
        if cwd_candidate.is_file():
            return str(cwd_candidate)

        backend_root = Path(__file__).resolve().parents[2]
        return str((backend_root / p).resolve())

    def _load_from_curriculum_json(self) -> None:
        curriculum_file = Path(self.curriculum_path)
        if not curriculum_file.is_file():
            return

        payload = json.loads(curriculum_file.read_text(encoding="utf-8"))
        raw_topics = payload.get("topics")
        if not isinstance(raw_topics, list):
            return

        for item in raw_topics:
            if not isinstance(item, dict):
                continue
            topic_id = str(item.get("id", "")).strip()
            title = str(item.get("title", "")).strip()
            if not topic_id or not title:
                continue
            description = str(item.get("description", "")).strip()
            track = self._normalise_track(str(item.get("track", ""))) or self._infer_track(
                topic_id,
                title,
            )
            levels = self._normalise_levels(item.get("levels"))
            sections = self._normalise_sections(item.get("sections"))
            raw_content = str(item.get("raw_content", "")).strip() or self._render_raw_content(
                title=title,
                description=description,
                sections=sections,
            )
            self._topics[topic_id] = TopicDetail(
                id=topic_id,
                title=title,
                description=description,
                track=track or "",
                levels=levels,
                sections=sections,
                raw_content=raw_content,
            )

    def _load_from_markdown_dir(self) -> None:
        handbook_dir = os.path.join(self.docs_path, "owner-handbook")
        if not os.path.isdir(handbook_dir):
            if os.path.isdir(self.docs_path):
                handbook_dir = self.docs_path
            else:
                return

        for fname in sorted(os.listdir(handbook_dir)):
            if not fname.endswith(".md"):
                continue
            if fname.lower() == "readme.md":
                continue
            fpath = os.path.join(handbook_dir, fname)
            topic_id = fname.replace(".md", "")
            raw = self._read_file(fpath)
            metadata, content = self._extract_frontmatter(raw)
            title = self._extract_title(content, fname)
            description = self._extract_description(content)
            sections = self._extract_sections(content)
            track = self._normalise_track(metadata.get("track")) or self._infer_track(
                topic_id,
                title,
            )
            levels = self._normalise_levels(metadata.get("levels"))
            self._topics[topic_id] = TopicDetail(
                id=topic_id,
                title=title,
                description=description,
                track=track or "",
                levels=levels,
                sections=sections,
                raw_content=content,
            )

    @staticmethod
    def _render_raw_content(
        *,
        title: str,
        description: str,
        sections: list[dict[str, str]],
    ) -> str:
        parts = [f"# {title.strip()}", "", description.strip()]
        for section in sections:
            parts.extend(["", f"## {section['heading'].strip()}", "", section["content"].strip()])
        return "\n".join(parts).strip()

    @staticmethod
    def _normalise_sections(raw_sections: object) -> list[dict[str, str]]:
        if not isinstance(raw_sections, list):
            return []
        sections: list[dict[str, str]] = []
        for item in raw_sections:
            if not isinstance(item, dict):
                continue
            heading = str(item.get("heading", "")).strip()
            content = str(item.get("content", "")).strip()
            if not heading or not content:
                continue
            sections.append({"heading": heading, "content": content})
        return sections

    @staticmethod
    def _read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _extract_frontmatter(content: str) -> tuple[dict[str, str | list[str]], str]:
        if not content.startswith("---"):
            return {}, content

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, flags=re.DOTALL)
        if not match:
            return {}, content

        meta_raw = match.group(1)
        body = match.group(2)
        metadata: dict[str, str | list[str]] = {}
        levels: list[str] = []
        in_levels_list = False

        for line in meta_raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if in_levels_list:
                if stripped.startswith("-"):
                    levels.append(stripped[1:].strip())
                    continue
                in_levels_list = False

            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "levels":
                if not value:
                    in_levels_list = True
                    continue
                if value.startswith("[") and value.endswith("]"):
                    items = [v.strip() for v in value[1:-1].split(",")]
                    levels.extend([i for i in items if i])
                else:
                    levels.extend([v.strip() for v in value.split(",") if v.strip()])
            else:
                metadata[key] = value

        if levels:
            metadata["levels"] = levels
        return metadata, body

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if m:
            return m.group(1).strip()
        return fallback.replace(".md", "").replace("-", " ").title()

    @staticmethod
    def _extract_description(content: str) -> str:
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

        i = 1
        while i < len(parts) - 2:
            heading = parts[i + 1].strip()
            body = parts[i + 2].strip() if i + 2 < len(parts) else ""
            sections.append({"heading": heading, "content": body[:12000]})
            i += 3

        if not sections:
            snippet = content.strip()
            if snippet:
                sections.append({"heading": "Overview", "content": snippet[:12000]})
        return sections

    @staticmethod
    def _normalise_track(track: Optional[str]) -> str:
        if not track:
            return ""
        t = track.strip().lower().replace("-", "_").replace(" ", "_")
        return t if t in _VALID_TRACKS else ""

    @staticmethod
    def _normalise_level(level: Optional[str]) -> str:
        if not level:
            return ""
        l = level.strip().lower()
        return l if l in _VALID_LEVELS else ""

    def _normalise_levels(self, levels_meta: object) -> list[str]:
        if isinstance(levels_meta, str):
            raw_levels = [levels_meta]
        elif isinstance(levels_meta, list):
            raw_levels = [str(v) for v in levels_meta]
        else:
            raw_levels = []

        normalised: list[str] = []
        for item in raw_levels:
            l = self._normalise_level(item)
            if l and l not in normalised:
                normalised.append(l)
        return normalised or list(_DEFAULT_LEVELS)

    def _infer_track(self, topic_id: str, title: str) -> str:
        text = f"{topic_id} {title}".lower()
        if "system design" in text or "system_design" in text:
            return "system_design"
        if "ai" in text or "llm" in text or "rag" in text or "agent" in text:
            return "ai_stack"
        if "frontend" in text or "ui" in text:
            return "frontend"
        if "backend" in text or "api" in text or "http" in text:
            return "backend"
        return "backend"
