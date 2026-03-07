"""Generate user-defined custom topic roadmaps via LLM with validation/fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from app.schemas.models import LLMConfigRequest, TopicDetail
from app.services.llm_client import LLMClient
from app.services.llm_policy import raise_if_policy_blocked_result

logger = logging.getLogger(__name__)

_VALID_TRACKS = {"backend", "frontend", "system_design", "ai_stack"}
_VALID_LEVELS = {"junior", "mid", "senior"}
_DEFAULT_LEVELS = ["junior", "mid", "senior"]
_MIN_SECTIONS = 100
_MAX_SECTIONS = 150
_DEFAULT_SECTIONS = 120
_MAX_ATTEMPTS = 4


def _slugify_topic(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug or "topic"


def _clamp_target_sections(value: int) -> int:
    if value < _MIN_SECTIONS:
        return _MIN_SECTIONS
    if value > _MAX_SECTIONS:
        return _MAX_SECTIONS
    return value


def _normalise_track(track: str) -> str:
    t = (track or "").strip().lower().replace("-", "_").replace(" ", "_")
    if t in _VALID_TRACKS:
        return t
    return ""


def _infer_track(topic: str) -> str:
    text = topic.lower()
    if any(k in text for k in ("system design", "distributed", "scaling", "reliability")):
        return "system_design"
    if any(k in text for k in ("frontend", "react", "vue", "css", "javascript", "web")):
        return "frontend"
    if any(k in text for k in ("ai", "llm", "prompt", "rag", "agent", "ml", "machine learning")):
        return "ai_stack"
    return "backend"


def _normalise_levels(levels: Any) -> list[str]:
    if isinstance(levels, list):
        raw = [str(x).strip().lower() for x in levels]
    elif isinstance(levels, str):
        raw = [x.strip().lower() for x in levels.split(",")]
    else:
        raw = []
    out: list[str] = []
    for item in raw:
        if item in _VALID_LEVELS and item not in out:
            out.append(item)
    return out or list(_DEFAULT_LEVELS)


def _level_rank(level: str) -> int:
    if level == "junior":
        return 0
    if level == "mid":
        return 1
    return 2


def _level_for_position(index: int, total: int) -> str:
    if total <= 0:
        return "junior"
    junior_cutoff = max(1, total // 3)
    mid_cutoff = max(junior_cutoff + 1, (2 * total) // 3)
    if index < junior_cutoff:
        return "junior"
    if index < mid_cutoff:
        return "mid"
    return "senior"


def _format_heading(level: str, heading: str) -> str:
    stripped = heading.strip()
    if not stripped:
        return ""
    prefix = f"{level.title()}:"
    if stripped.lower().startswith(("junior:", "mid:", "senior:")):
        return stripped[:200]
    return f"{prefix} {stripped}"[:200]


def _autofill_section(topic: str, index: int, total: int) -> dict[str, str]:
    level = _level_for_position(index, total)
    level_templates = {
        "junior": [
            "Fundamentals and terminology",
            "Setup and development environment",
            "Core syntax and mental models",
            "Data structures and primitives",
            "Control flow and basic patterns",
            "Error handling basics",
            "Testing fundamentals",
            "Debugging and troubleshooting basics",
            "Common beginner interview pitfalls",
            "Hands-on practice exercises",
        ],
        "mid": [
            "Architecture patterns and tradeoffs",
            "Performance optimization techniques",
            "State management and data flow",
            "API and integration design",
            "Concurrency and parallelism in practice",
            "Security and reliability considerations",
            "Code quality and maintainability",
            "Observability and production diagnostics",
            "Advanced testing strategies",
            "Scenario-based interview drills",
        ],
        "senior": [
            "System-level design decisions",
            "Scalability and capacity planning",
            "Resilience and failure mode strategy",
            "Cost, performance, and risk tradeoffs",
            "Governance and technical leadership",
            "Migration and modernization strategy",
            "Cross-team collaboration patterns",
            "Operational excellence at scale",
            "Deep architecture interview cases",
            "Senior-level decision communication",
        ],
    }
    templates = level_templates[level]
    template = templates[index % len(templates)]
    facets = [
        "Core Concepts",
        "Hands-on Practice",
        "Design Tradeoffs",
        "Debugging Patterns",
        "Performance Lens",
        "Reliability Lens",
        "Security Lens",
        "Testing Lens",
        "Interview Scenarios",
        "Anti-patterns",
        "Production Notes",
        "Case Study",
    ]
    facet = facets[index % len(facets)]
    heading = _format_heading(level, f"{topic} {template} - {facet}")
    content = (
        f"Focus on {template.lower()} for {topic}, with practical implementation details, "
        "tradeoff analysis, and interview-ready reasoning depth for this level."
    )
    return {"heading": heading, "content": content[:1800]}


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _normalise_sections(sections: Any, topic: str, target_sections: int) -> list[dict[str, str]]:
    ranked_sections: list[tuple[int, int, dict[str, str]]] = []
    seen: set[str] = set()
    if isinstance(sections, list):
        for idx, item in enumerate(sections):
            if not isinstance(item, dict):
                continue
            heading = str(item.get("heading", "")).strip()
            content = str(item.get("content", "")).strip()
            if len(heading) < 3 or len(content) < 10:
                continue
            level_raw = str(item.get("level", "")).strip().lower()
            level = level_raw if level_raw in _VALID_LEVELS else _level_for_position(idx, target_sections)
            formatted_heading = _format_heading(level, heading)
            heading_key = formatted_heading.lower()
            if heading_key in seen:
                continue
            seen.add(heading_key)
            ranked_sections.append(
                (
                    _level_rank(level),
                    idx,
                    {
                        "heading": formatted_heading,
                        "content": content[:1800],
                    },
                )
            )
            if len(ranked_sections) >= target_sections:
                break

    ranked_sections.sort(key=lambda item: (item[0], item[1]))
    normalised = [item[2] for item in ranked_sections]

    if len(normalised) < target_sections:
        for idx in range(len(normalised), target_sections):
            fill = _autofill_section(topic, idx, target_sections)
            heading_key = fill["heading"].lower()
            if heading_key in seen:
                continue
            seen.add(heading_key)
            normalised.append(fill)
            if len(normalised) >= target_sections:
                break

    return normalised[:target_sections]


def _build_raw_content(title: str, description: str, sections: list[dict[str, str]]) -> str:
    parts = [f"# {title}", "", description.strip()]
    for sec in sections:
        parts.extend(["", f"## {sec['heading']}", "", sec["content"]])
    return "\n".join(parts).strip()


def _build_prompt(topic: str, target_sections: int) -> str:
    junior_count = target_sections // 3
    mid_count = target_sections // 3
    senior_count = target_sections - junior_count - mid_count
    return f"""You are an expert curriculum architect and technical interview coach.

Create a deep learning roadmap for the custom topic: "{topic}".

Return ONLY valid JSON object with this exact schema:
{{
  "title": "string",
  "description": "string",
  "track": "backend|frontend|system_design|ai_stack",
  "levels": ["junior", "mid", "senior"],
  "sections": [
    {{
      "heading": "specific subtopic title",
      "content": "2-4 concise sentences with practical learning notes",
      "level": "junior|mid|senior"
    }}
  ]
}}

Hard constraints:
- sections length must be exactly {target_sections}.
- Ensure level progression and ordering:
  - first {junior_count} sections are junior,
  - next {mid_count} sections are mid,
  - final {senior_count} sections are senior.
- Subtopics must be granular and non-duplicative.
- Each heading must be a real, concrete topic name.
- Do NOT use generic headings like "Module 1", "Part A", "Topic X", or placeholders.
- Cover fundamentals through advanced architecture and interview tradeoffs.
- Keep each section content concise and practical.
- Output JSON only with escaped newlines/quotes/backslashes.
"""


def _fallback_topic_detail(topic: str, topic_id: str, target_sections: int) -> TopicDetail:
    title = f"{topic.strip().title()} Interview Roadmap"
    description = (
        f"Comprehensive custom roadmap for {topic} covering fundamentals, implementation, "
        "architecture, performance, and interview-oriented tradeoffs."
    )
    track = _infer_track(topic)
    levels = list(_DEFAULT_LEVELS)
    sections = _normalise_sections([], topic, target_sections)
    return TopicDetail(
        id=topic_id,
        title=title,
        description=description,
        track=track,
        levels=levels,
        sections=sections,
        raw_content=_build_raw_content(title, description, sections),
    )


class CustomTopicGenerator:
    """Generate custom roadmap content with deterministic topic IDs."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def generate_topic(
        self,
        *,
        topic: str,
        target_sections: int = _DEFAULT_SECTIONS,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> TopicDetail:
        source_topic = (topic or "").strip()
        if not source_topic:
            source_topic = "Custom Topic"
        target = _clamp_target_sections(target_sections)
        topic_id = f"custom-{_slugify_topic(source_topic)}"
        prompt = _build_prompt(source_topic, target)

        for _ in range(_MAX_ATTEMPTS):
            result = await self.llm.completion(
                prompt,
                llm_config,
                user_identity=user_identity,
            )
            raise_if_policy_blocked_result(result)
            if not result.get("success"):
                continue
            payload = _parse_json_object(result.get("analysis", ""))
            if not payload:
                continue

            title = str(payload.get("title", "")).strip() or f"{source_topic.title()} Interview Roadmap"
            description = str(payload.get("description", "")).strip()
            if not description:
                description = (
                    f"Comprehensive custom roadmap for {source_topic} with interview-focused "
                    "subtopics and practical depth."
                )
            track = _normalise_track(str(payload.get("track", ""))) or _infer_track(source_topic)
            levels = _normalise_levels(payload.get("levels"))
            sections = _normalise_sections(payload.get("sections"), source_topic, target)
            if len(sections) < _MIN_SECTIONS:
                continue
            return TopicDetail(
                id=topic_id,
                title=title[:200],
                description=description[:600],
                track=track,
                levels=levels,
                sections=sections,
                raw_content=_build_raw_content(title, description, sections),
            )

        logger.warning("Falling back to synthetic custom roadmap for topic=%s", source_topic)
        return _fallback_topic_detail(source_topic, topic_id, target)
