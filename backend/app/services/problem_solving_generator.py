"""Generate language-specific problem-solving curriculum."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from app.schemas.models import LLMConfigRequest, TopicDetail
from app.services.llm_client import LLMClient
from app.services.llm_policy import raise_if_policy_blocked_result
from app.services.mcp_gateway import MCPGateway
from app.services.topic_catalog import (
    PROBLEM_SOLVING_DEFAULT_LANGUAGE,
    PROBLEM_SOLVING_DESCRIPTION,
    PROBLEM_SOLVING_LANGUAGE_OPTIONS,
    PROBLEM_SOLVING_LEVELS,
    PROBLEM_SOLVING_TARGET_SECTIONS,
    PROBLEM_SOLVING_TITLE,
    PROBLEM_SOLVING_TOPIC_ID,
    PROBLEM_SOLVING_TRACK,
)

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 4
_MIN_CONTENT_CHARS = 90


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _normalise_language(language: str) -> str:
    lang = str(language or "").strip().lower()
    if lang in PROBLEM_SOLVING_LANGUAGE_OPTIONS:
        return lang
    return PROBLEM_SOLVING_DEFAULT_LANGUAGE


def _level_for_index(index: int, total: int) -> str:
    if total <= 0:
        return "junior"
    one_third = max(1, total // 3)
    two_third = max(one_third + 1, (2 * total) // 3)
    if index < one_third:
        return "junior"
    if index < two_third:
        return "mid"
    return "senior"


def _format_heading(level: str, heading: str) -> str:
    prefix = f"{level.title()}: "
    plain = str(heading or "").strip()
    if plain.lower().startswith(("junior:", "mid:", "senior:")):
        return plain[:160]
    return f"{prefix}{plain[:150]}"


def _normalise_sections(sections: Any, target_sections: int, language: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    if not isinstance(sections, list):
        sections = []

    for idx, item in enumerate(sections):
        if len(out) >= target_sections:
            break
        if not isinstance(item, dict):
            continue
        level = _level_for_index(idx, target_sections)
        heading = _format_heading(level, str(item.get("heading", "")).strip())
        content = str(item.get("content", "")).strip()
        if not heading or len(content) < _MIN_CONTENT_CHARS:
            continue
        key = heading.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"heading": heading, "content": content[:3000]})

    if len(out) >= target_sections:
        return out[:target_sections]
    return _fallback_sections(language, target_sections, seed=out)


def _fallback_sections(
    language: str,
    target_sections: int,
    seed: Optional[list[dict[str, str]]] = None,
) -> list[dict[str, str]]:
    sections = list(seed or [])
    seen = {str(item.get("heading", "")).lower() for item in sections}
    junior_topics = [
        "Complexity Analysis and Big-O",
        "Arrays and Two-Pointer Techniques",
        "Hash Maps and Frequency Counting",
        "Strings and Sliding Window",
        "Stacks, Queues, and Monotonic Variants",
        "Recursion and Backtracking Basics",
        "Sorting and Binary Search",
        "Linked Lists Fundamentals",
        "Trees and DFS/BFS Traversal",
        "Heaps and Priority Queues",
    ]
    mid_topics = [
        "Dynamic Programming Patterns",
        "Greedy Strategy and Proof Intuition",
        "Graph Algorithms: Shortest Path",
        "Graph Algorithms: Topological Ordering",
        "Union-Find and Connectivity",
        "Prefix Sums and Difference Arrays",
        "Interval Problems and Sweep Line",
        "Bit Manipulation Techniques",
        "Advanced Binary Search on Answer",
        "State Compression and Memoization",
    ]
    senior_topics = [
        "Complexity Tradeoffs Under Constraints",
        "Problem Decomposition and Invariants",
        "Interview Communication Under Time Pressure",
        "Designing Optimal Data Structures",
        "Hard Graph and DP Hybrids",
        "Proof of Correctness and Edge Cases",
        "Performance Tuning and Micro-optimizations",
        "Amortized Analysis and Tradeoffs",
        "Robust Testing Strategy for Algorithms",
        "Pattern Selection Strategy for New Problems",
    ]
    libraries = {
        "junior": junior_topics,
        "mid": mid_topics,
        "senior": senior_topics,
    }

    idx = len(sections)
    while len(sections) < target_sections:
        level = _level_for_index(idx, target_sections)
        lib = libraries[level]
        base = lib[(idx // 2) % len(lib)]
        heading = _format_heading(level, f"{base} ({language}) #{idx + 1}")
        if heading.lower() in seen:
            idx += 1
            continue
        seen.add(heading.lower())
        content = (
            f"Explain {base.lower()} for {language} interview problem solving. Cover how to identify "
            "the pattern from constraints, how to reason about correctness, and the time/space tradeoffs. "
            "Include common pitfalls and how a candidate should communicate the approach in an interview."
        )
        sections.append({"heading": heading, "content": content})
        idx += 1
    return sections[:target_sections]


def _build_raw_content(title: str, description: str, sections: list[dict[str, str]]) -> str:
    lines = [f"# {title}", "", description.strip()]
    for section in sections:
        lines.extend(["", f"## {section['heading']}", "", section["content"].strip()])
    return "\n".join(lines).strip()


def _build_prompt(language: str, target_sections: int, mcp_context: str = "") -> str:
    mcp_block = (
        f"\n\nExternal context (optional, use only if factual and useful):\n{mcp_context[:2500]}"
        if mcp_context
        else ""
    )
    return f"""You are an expert programming interview instructor.

Create a complete problem-solving curriculum for language "{language}".
Return ONLY valid JSON object:
{{
  "title": "Problem Solving and Algorithms ({language})",
  "description": "One short paragraph",
  "track": "backend",
  "levels": ["junior", "mid", "senior"],
  "sections": [
    {{"heading": "section name", "content": "concise but useful explanation"}}
  ]
}}

Rules:
- Produce exactly {target_sections} sections.
- Order sections from beginner (junior) to intermediate (mid) to advanced (senior).
- Every heading must be concrete and interview-relevant (not generic).
- Every content field should include:
  - problem-reading strategy,
  - algorithm approach intuition,
  - complexity tradeoff guidance,
  - common mistakes.
- Keep each content concise and practical.
- Use language-specific framing for {language} idioms and tradeoffs.
- No markdown, no prose outside JSON.
{mcp_block}
"""


class ProblemSolvingGenerator:
    def __init__(self, llm_client: LLMClient, mcp_gateway: MCPGateway | None = None):
        self.llm = llm_client
        self.mcp = mcp_gateway

    async def generate_topic(
        self,
        *,
        preferred_language: str,
        target_sections: int = PROBLEM_SOLVING_TARGET_SECTIONS,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> TopicDetail:
        language = _normalise_language(preferred_language)
        target = PROBLEM_SOLVING_TARGET_SECTIONS if int(target_sections or 0) <= 0 else int(target_sections)
        if target != PROBLEM_SOLVING_TARGET_SECTIONS:
            target = PROBLEM_SOLVING_TARGET_SECTIONS

        mcp_context = ""
        if self.mcp is not None:
            mcp_context = await self.mcp.gather_context(
                flow="custom_topic",
                query=f"{language} coding interview roadmap algorithms data structures",
                topic_id=PROBLEM_SOLVING_TOPIC_ID,
                topic_title=PROBLEM_SOLVING_TITLE,
            )
        prompt = _build_prompt(language=language, target_sections=target, mcp_context=mcp_context)

        payload: dict[str, Any] = {}
        for _ in range(_MAX_ATTEMPTS):
            result = await self.llm.completion(
                prompt,
                llm_config,
                user_identity=user_identity,
            )
            raise_if_policy_blocked_result(result)
            if not result.get("success"):
                continue
            parsed = _parse_json_object(str(result.get("analysis", "")))
            if not parsed:
                continue
            sections = _normalise_sections(parsed.get("sections"), target, language)
            if len(sections) != target:
                continue
            payload = parsed
            payload["sections"] = sections
            break

        if not payload:
            logger.warning("problem_solving_generator_fallback language=%s", language)
            payload = {
                "title": f"{PROBLEM_SOLVING_TITLE} ({language})",
                "description": (
                    f"Structured {language} roadmap for coding interview problem solving from "
                    "beginner to senior depth."
                ),
                "track": PROBLEM_SOLVING_TRACK,
                "levels": list(PROBLEM_SOLVING_LEVELS),
                "sections": _fallback_sections(language, target),
            }

        title = str(payload.get("title", "")).strip() or f"{PROBLEM_SOLVING_TITLE} ({language})"
        description = str(payload.get("description", "")).strip() or PROBLEM_SOLVING_DESCRIPTION
        sections = _normalise_sections(payload.get("sections"), target, language)
        raw_content = _build_raw_content(title, description, sections)

        return TopicDetail(
            id=PROBLEM_SOLVING_TOPIC_ID,
            title=title,
            description=description,
            track=PROBLEM_SOLVING_TRACK,
            levels=list(PROBLEM_SOLVING_LEVELS),
            sections=sections,
            raw_content=raw_content,
            requires_programming=True,
            language_options=list(PROBLEM_SOLVING_LANGUAGE_OPTIONS),
            selected_language=language,
            response_detail="concise",
            is_dynamic_topic=True,
            content_ready=True,
        )

