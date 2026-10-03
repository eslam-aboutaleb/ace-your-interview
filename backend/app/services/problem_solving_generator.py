"""Generate language-specific problem-solving curriculum."""

from __future__ import annotations

import inspect
import logging
import math
import re
from typing import Any, AsyncIterator, Optional

from app.schemas.models import LLMConfigRequest, TopicDetail
from app.services.llm_client import (
    TERMINAL_ERROR_CODES,
    LLMClient,
    parse_json_object,
)
from app.services.llm_policy import raise_if_policy_blocked_result
from app.services.mcp_gateway import MCPGateway
from app.services.prompt_blocks import (
    UNTRUSTED_CLAUSE,
    render_contract,
    untrusted_block,
)
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
_STREAM_BATCH_SIZE = 12
_MAX_STREAM_HEADINGS_CONTEXT = 120
_MAX_TOKENS_CAP = 1500
_INSTRUCTOR_SYSTEM_PROMPT = (
    "You are an expert programming interview instructor. You write language-specific "
    "problem-solving curricula that move from beginner to advanced, and you always "
    "respond with a single JSON object and never with prose."
)


def _supported_options(client: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Drop call options ``client.completion`` does not accept.

    ``LLMClient`` takes the full Stage-2 option set, but the client is injected,
    so a narrower implementation may be in use. Filter against the bound
    signature rather than assuming one shape.
    """
    try:
        params = inspect.signature(client.completion).parameters
    except (TypeError, ValueError):
        return dict(options)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return options
    return {key: value for key, value in options.items() if key in params}



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


def _build_stream_batch_prompt(
    *,
    language: str,
    target_sections: int,
    start_index: int,
    batch_size: int,
    existing_headings: list[str],
    mcp_context: str = "",
) -> str:
    """Build the user turn. The instructor persona lives in ``system`` (Stage 2.1).

    The rules here carry the whole intent of the deleted non-streaming
    ``_build_prompt``: exact batch size, level progression, interview-relevance
    per heading, and the four content dimensions. The coverage mandate below is
    the part the live prompt used to be missing.
    """
    start_no = start_index + 1
    end_no = min(target_sections, start_index + batch_size)
    level_start = _level_for_index(start_index, target_sections)
    level_end = _level_for_index(max(start_index, end_no - 1), target_sections)
    existing_blob = "\n".join(
        [f"- {h}" for h in existing_headings[-_MAX_STREAM_HEADINGS_CONTEXT:]]
    )
    existing_block = existing_blob if existing_blob else "- (none yet)"
    mcp_block = untrusted_block("external_context", mcp_context, 2500)
    contract = render_contract(
        schema_label="Return ONLY valid JSON object with this exact schema",
        schema_block="""{
  "title": "string",
  "description": "string",
  "sections": [
    {"heading": "section name", "content": "concise but useful explanation"}
  ]
}""",
        rules=[
            f"sections length must be exactly {batch_size}.",
            f"These sections correspond to positions {start_no}..{end_no} of {target_sections}.",
            f"Expected level progression in this batch: {level_start} -> {level_end}.",
            "Do not repeat or paraphrase these existing headings:",
            existing_block,
            "Every heading must be concrete and interview-relevant, not a placeholder.",
            f'Every heading must be genuinely specific to {language}.',
            "Every content field must cover problem-reading strategy, algorithm approach intuition, complexity tradeoff guidance, and common mistakes.",
            "Keep content concise and practical.",
            f"Use {language}-specific framing for idioms and tradeoffs, not generic advice.",
            "Across the whole roadmap the batches together must cover fundamentals, data structures, algorithmic patterns, and interview communication.",
            "No markdown and no prose outside JSON.",
            UNTRUSTED_CLAUSE,
        ],
    )
    return f"""{contract}{mcp_block}
"""


def _normalise_batch_sections(
    *,
    sections: Any,
    target_sections: int,
    language: str,
    start_index: int,
    batch_size: int,
    seen_headings: set[str],
) -> tuple[list[dict[str, str]], set[str]]:
    """Normalise a batch and report the headings it consumed.

    The consumed headings are returned instead of being committed here so the
    caller can commit them only once the batch is known good; a rejected
    attempt must not poison the de-dup set for the next attempt.
    """
    _ = language
    out: list[dict[str, str]] = []
    consumed: set[str] = set()
    if not isinstance(sections, list):
        return out, consumed

    for item in sections:
        if len(out) >= batch_size:
            break
        if not isinstance(item, dict):
            continue
        global_index = start_index + len(out)
        level = _level_for_index(global_index, target_sections)
        heading = _format_heading(level, str(item.get("heading", "")).strip())
        content = str(item.get("content", "")).strip()
        if not heading or len(content) < _MIN_CONTENT_CHARS:
            continue
        key = heading.lower()
        if key in seen_headings:
            continue
        consumed.add(key)
        out.append({"heading": heading, "content": content[:3000]})
    return out, consumed


def _fallback_batch_sections(
    *,
    language: str,
    target_sections: int,
    start_index: int,
    batch_size: int,
    seen_headings: set[str],
) -> list[dict[str, str]]:
    fallback_pool = _fallback_sections(language, target_sections, seed=[])
    out: list[dict[str, str]] = []
    while len(out) < batch_size:
        global_index = start_index + len(out)
        base = fallback_pool[global_index % len(fallback_pool)]
        base_heading = re.sub(
            r"^\s*(junior|mid|senior):\s*",
            "",
            str(base.get("heading", "")).strip(),
            flags=re.IGNORECASE,
        )
        level = _level_for_index(global_index, target_sections)
        heading = _format_heading(level, f"{base_heading} ({language}) #{global_index + 1}")
        suffix = 2
        while heading.lower() in seen_headings:
            heading = _format_heading(
                level,
                f"{base_heading} ({language}) #{global_index + 1}.{suffix}",
            )
            suffix += 1
        seen_headings.add(heading.lower())
        out.append(
            {
                "heading": heading,
                "content": str(base.get("content", "")).strip()[:3000],
            }
        )
    return out


def _build_topic_detail(
    *,
    language: str,
    title: str,
    description: str,
    sections: list[dict[str, str]],
) -> TopicDetail:
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
        final_topic: TopicDetail | None = None
        async for event in self.generate_topic_stream(
            preferred_language=preferred_language,
            target_sections=target_sections,
            llm_config=llm_config,
            user_identity=user_identity,
        ):
            if event.get("type") != "done":
                continue
            maybe_topic = event.get("topic")
            if isinstance(maybe_topic, TopicDetail):
                final_topic = maybe_topic
                break
        if final_topic is not None:
            return final_topic

        language = _normalise_language(preferred_language)
        target = PROBLEM_SOLVING_TARGET_SECTIONS if int(target_sections or 0) <= 0 else int(target_sections)
        if target != PROBLEM_SOLVING_TARGET_SECTIONS:
            target = PROBLEM_SOLVING_TARGET_SECTIONS
        logger.warning("problem_solving_generator_stream_missing_done language=%s", language)
        fallback_sections = _fallback_sections(language, target)
        return _build_topic_detail(
            language=language,
            title=f"{PROBLEM_SOLVING_TITLE} ({language})",
            description=PROBLEM_SOLVING_DESCRIPTION,
            sections=fallback_sections,
        )

    async def generate_topic_stream(
        self,
        *,
        preferred_language: str,
        target_sections: int = PROBLEM_SOLVING_TARGET_SECTIONS,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
    ) -> AsyncIterator[dict[str, Any]]:
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

        title = f"{PROBLEM_SOLVING_TITLE} ({language})"
        description = PROBLEM_SOLVING_DESCRIPTION
        sections: list[dict[str, str]] = []
        seen_headings: set[str] = set()
        batch_count = max(1, int(math.ceil(target / _STREAM_BATCH_SIZE)))

        for batch_index in range(batch_count):
            start_index = len(sections)
            if start_index >= target:
                break
            remaining = target - start_index
            batch_size = min(_STREAM_BATCH_SIZE, remaining)
            yield {
                "type": "progress",
                "batch_index": batch_index + 1,
                "batch_count": batch_count,
                "generated_sections": start_index,
                "target_sections": target,
                "message": (
                    f"Generating sections {start_index + 1}-{start_index + batch_size} "
                    f"of {target}."
                ),
            }

            prompt = _build_stream_batch_prompt(
                language=language,
                target_sections=target,
                start_index=start_index,
                batch_size=batch_size,
                existing_headings=[s["heading"] for s in sections],
                mcp_context=mcp_context,
            )
            batch_sections: list[dict[str, str]] = []
            accepted_headings: set[str] = set()
            attempt_title = title
            attempt_description = description
            for _ in range(_MAX_ATTEMPTS):
                result = await self.llm.completion(
                    prompt,
                    llm_config,
                    user_identity=user_identity,
                    **_supported_options(
                        self.llm,
                        {
                            "task": "final",
                            "system": _INSTRUCTOR_SYSTEM_PROMPT,
                            "structured": True,
                            "max_tokens_cap": _MAX_TOKENS_CAP,
                        },
                    ),
                )
                raise_if_policy_blocked_result(result)
                if not result.get("success"):
                    # A truncated response or an exhausted budget cannot be fixed
                    # by re-sending the identical prompt.
                    if str(result.get("error_code") or "") in TERMINAL_ERROR_CODES:
                        break
                    continue
                parsed = parse_json_object(str(result.get("analysis", "")))
                if not parsed:
                    continue
                attempt_title = str(parsed.get("title", "")).strip() or title
                attempt_description = str(parsed.get("description", "")).strip() or description
                # Each attempt de-dupes against its own copy so a rejected
                # attempt leaves no trace in the shared heading set.
                attempt_sections, attempt_headings = _normalise_batch_sections(
                    sections=parsed.get("sections"),
                    target_sections=target,
                    language=language,
                    start_index=start_index,
                    batch_size=batch_size,
                    seen_headings=set(seen_headings),
                )
                if len(attempt_sections) >= batch_size:
                    batch_sections = attempt_sections
                    accepted_headings = attempt_headings
                    break
                # Keep the longest partial batch seen so far so a later failure
                # does not discard work that already validated.
                if len(attempt_sections) > len(batch_sections):
                    batch_sections = attempt_sections
                    accepted_headings = attempt_headings

            if len(batch_sections) >= batch_size:
                title = attempt_title
                description = attempt_description
                seen_headings.update(accepted_headings)

            if len(batch_sections) < batch_size:
                logger.warning(
                    "problem_solving_generator_batch_fallback language=%s batch=%s size=%s got=%s",
                    language,
                    batch_index + 1,
                    batch_size,
                    len(batch_sections),
                )
                batch_sections.extend(
                    _fallback_batch_sections(
                        language=language,
                        target_sections=target,
                        start_index=start_index + len(batch_sections),
                        batch_size=batch_size - len(batch_sections),
                        seen_headings=seen_headings,
                    )
                )

            for section in batch_sections[:batch_size]:
                sections.append(section)
                yield {
                    "type": "section",
                    "index": len(sections),
                    "total_sections": target,
                    "heading": section["heading"],
                    "content": section["content"],
                }

        if len(sections) < target:
            sections = _fallback_sections(language, target, seed=sections)
        else:
            sections = sections[:target]
        title = title or f"{PROBLEM_SOLVING_TITLE} ({language})"
        description = description or PROBLEM_SOLVING_DESCRIPTION
        topic = _build_topic_detail(
            language=language,
            title=title,
            description=description,
            sections=sections,
        )
        yield {"type": "done", "topic": topic}
