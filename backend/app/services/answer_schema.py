"""Validation, bounded repair and legacy conversion for rich answers (issue #3).

Generation calls :func:`repair_rich_answer` on any structured ``sections``
payload a model returns. The rule is strict: **only content that fully
validates as a :class:`~app.schemas.answers.RichAnswer` is ever published**;
anything that cannot be repaired within the bounded repair budget is dropped
back to the legacy flat-markdown answer, never served as structured content.

Repairs are deterministic and recorded, so the eval tier can count them:
id assignment, heading demotion, fence balancing, unsafe-HTML escaping,
table padding, unknown-type fallback to escaped text, and dropping sections
that still fail.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.schemas.answers import (
    ANSWER_SCHEMA_VERSION,
    RichAnswer,
    UNSAFE_HTML_RE,
)

MAX_REPAIR_ROUNDS = 3
MAX_MARKDOWN_LENGTH = 50000

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class AnswerValidationError(ValueError):
    """Raised when a payload cannot validate as a RichAnswer."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__(";".join(issues))


@dataclass
class RepairResult:
    answer: RichAnswer | None
    repairs: list[str] = field(default_factory=list)
    dropped_sections: int = 0


def stable_section_id(section_type: str, title: str, index: int) -> str:
    """Deterministic slug ID for a section missing one."""
    base = _SLUG_RE.sub("-", (title or section_type).strip().lower()).strip("-")
    if not base:
        base = section_type
    return f"{base[:60]}-{index}"


def _escape_unsafe_html(text: str) -> str:
    """Neutralize executable markup by escaping the angle brackets in tags."""

    def _escape_tag(match: re.Match[str]) -> str:
        # Entity-escape the angle brackets: the tag becomes visible inert
        # text and no longer matches the executable-markup detector.
        return match.group(0).replace("<", "&lt;").replace(">", "&gt;")

    text = re.sub(
        r"<\s*/?\s*(script|iframe|object|embed|form|input|button|textarea|select|"
        r"style|link|meta|base|svg|math)[^>]*>?",
        _escape_tag,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bon([a-z]+)\s*=", r"on-\1=", text, flags=re.IGNORECASE)
    text = re.sub(r"javascript\s*:", "javascript-", text, flags=re.IGNORECASE)
    text = re.sub(r"vbscript\s*:", "vbscript-", text, flags=re.IGNORECASE)
    text = re.sub(r"data\s*:\s*text/html", "data:text-plain", text, flags=re.IGNORECASE)
    return text


def _fix_heading_skips(text: str) -> str:
    """Demote headings so levels never skip (## -> #### becomes ## -> ###)."""
    lines = text.splitlines()
    previous = 0
    fixed = []
    for line in lines:
        match = re.match(r"^(#{1,6})(\s+\S.*)$", line)
        if match:
            level = len(match.group(1))
            if previous and level > previous + 1:
                level = previous + 1
                line = "#" * level + match.group(2)
            previous = level
        fixed.append(line)
    return "\n".join(fixed)


def _balance_code_fences(text: str) -> str:
    """Close an unterminated fenced code block."""
    fence_count = len(re.findall(r"^```", text, flags=re.MULTILINE))
    if fence_count % 2 == 1:
        return text.rstrip("\n") + "\n```\n"
    return text


def _repair_text(text: Any, repairs: list[str], *, label: str) -> str:
    value = str(text if text is not None else "")
    if UNSAFE_HTML_RE.search(value):
        value = _escape_unsafe_html(value)
        repairs.append(f"escaped_unsafe_html:{label}")
    fixed = _fix_heading_skips(value)
    if fixed != value:
        repairs.append(f"demoted_headings:{label}")
        value = fixed
    balanced = _balance_code_fences(value)
    if balanced != value:
        repairs.append(f"closed_code_fence:{label}")
        value = balanced
    if len(value) > MAX_MARKDOWN_LENGTH:
        value = value[:MAX_MARKDOWN_LENGTH]
        repairs.append(f"truncated:{label}")
    return value


def _section_text_keys(section: dict[str, Any]) -> list[str]:
    """Fields of a section payload that carry markdown or free text."""
    keys = []
    for key in ("text", "title", "alt", "quote"):
        if key in section:
            keys.append(key)
    return keys


def _repair_section_payload(
    section: Any, index: int, repairs: list[str]
) -> dict[str, Any]:
    """Best-effort repair of one raw section; may still fail validation."""
    if not isinstance(section, dict):
        return {
            "type": "explanation",
            "text": f"_(unsupported section payload dropped: {type(section).__name__})_",
        }
    fixed = dict(section)
    section_type = str(fixed.get("type", "")).strip() or "explanation"
    if section_type not in (
        "summary",
        "explanation",
        "code",
        "table",
        "diagram",
        "tradeoffs",
        "common_mistakes",
        "references",
        "follow_up",
    ):
        # Unsupported sections fall back to escaped text, never raw HTML.
        escaped = json.dumps(fixed, ensure_ascii=False, indent=2)[:5000]
        repairs.append(f"unknown_section_type:{section_type}")
        fixed = {
            "type": "explanation",
            "title": f"Unsupported section ({section_type[:40]})",
            "text": f"```json\n{escaped}\n```",
        }
        section_type = "explanation"

    for key in _section_text_keys(fixed):
        fixed[key] = _repair_text(fixed[key], repairs, label=f"section[{index}].{key}")
    for container in ("items", "prompts", "pros", "cons"):
        if isinstance(fixed.get(container), list):
            fixed[container] = [
                (
                    {
                        k: _repair_text(
                            v, repairs, label=f"section[{index}].{container}.{k}"
                        )
                        for k, v in entry.items()
                    }
                    if isinstance(entry, dict)
                    else _repair_text(
                        entry, repairs, label=f"section[{index}].{container}"
                    )
                )
                for entry in fixed[container]
            ]

    if section_type == "table":
        columns = fixed.get("columns") or []
        width = len(columns)
        rows = fixed.get("rows") or []
        repaired_rows = []
        for row in rows:
            row = [str(cell) for cell in (row or [])]
            if len(row) < width:
                row = row + [""] * (width - len(row))
                repairs.append(f"padded_table_row:section[{index}]")
            elif len(row) > width:
                row = row[:width]
                repairs.append(f"truncated_table_row:section[{index}]")
            repaired_rows.append(row)
        fixed["rows"] = repaired_rows

    if section_type == "diagram":
        if not str(fixed.get("alt", "")).strip():
            fixed["alt"] = str(fixed.get("title", "")).strip() or "Diagram"
            repairs.append(f"added_diagram_alt:section[{index}]")
        fixed["format"] = "mermaid"

    if section_type == "code":
        language = str(fixed.get("language", "")).strip().lower()
        if not language or not re.match(r"^[a-z0-9][a-z0-9+#.-]{0,40}$", language):
            fixed["language"] = "text"
            repairs.append(f"normalised_language:section[{index}]")
        else:
            fixed["language"] = language

    section_id = str(fixed.get("id", "")).strip()
    if not section_id or not re.match(r"^[a-z0-9][a-z0-9-]{0,80}$", section_id):
        fixed["id"] = stable_section_id(
            section_type, str(fixed.get("title", "")), index
        )
        repairs.append(f"assigned_section_id:section[{index}]")
    return fixed


def parse_rich_answer(payload: dict[str, Any]) -> RichAnswer:
    """Validate a raw payload; raise AnswerValidationError with coded issues."""
    try:
        return RichAnswer.model_validate(payload)
    except ValueError as exc:
        raise AnswerValidationError([str(exc)]) from exc


def repair_rich_answer(
    payload: dict[str, Any], *, max_rounds: int = MAX_REPAIR_ROUNDS
) -> RepairResult:
    """Repair a raw sections payload into a valid RichAnswer, or fail.

    Bounded: at most ``max_rounds`` repair/validate cycles; sections that
    still fail are dropped one by one. Returns ``answer=None`` when nothing
    salvageable remains - callers must then fall back to the legacy answer.
    """
    repairs: list[str] = []
    dropped = 0
    if not isinstance(payload, dict):
        return RepairResult(answer=None, repairs=["payload_not_an_object"])
    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        return RepairResult(answer=None, repairs=["no_sections"])

    working = list(raw_sections)
    for _round in range(max_rounds):
        repaired_sections = [
            _repair_section_payload(section, index, repairs)
            for index, section in enumerate(working)
        ]
        # De-duplicate IDs deterministically.
        seen: dict[str, int] = {}
        for index, section in enumerate(repaired_sections):
            sid = str(section.get("id", ""))
            if sid in seen:
                section["id"] = f"{sid}-{index}"
                repairs.append(f"deduplicated_section_id:section[{index}]")
            seen[str(section.get("id", ""))] = index

        candidate = {
            "answer_schema_version": str(
                payload.get("answer_schema_version") or ANSWER_SCHEMA_VERSION
            ),
            "sections": repaired_sections,
        }
        try:
            answer = RichAnswer.model_validate(candidate)
            return RepairResult(
                answer=answer, repairs=repairs, dropped_sections=dropped
            )
        except ValueError:
            # Drop the first section that still fails on its own.
            failing_index = None
            for index, section in enumerate(repaired_sections):
                try:
                    single = RichAnswer.model_validate(
                        {**candidate, "sections": [section]}
                    )
                    del single
                except ValueError:
                    failing_index = index
                    break
            if failing_index is None:
                failing_index = 0
            working = [s for i, s in enumerate(repaired_sections) if i != failing_index]
            dropped += 1
            repairs.append(f"dropped_section:{failing_index}")
            if not working:
                break
    return RepairResult(answer=None, repairs=repairs, dropped_sections=dropped)


# ----------------------------------------------------------------------
# Legacy markdown compatibility
# ----------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```([a-zA-Z0-9+#.-]*)\s*$")
_ATX_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def markdown_to_rich_answer(markdown: str) -> RichAnswer:
    """Convert a legacy flat-markdown answer into typed sections.

    Deterministic split: leading prose becomes a ``summary`` section, each
    ATX heading opens an ``explanation`` section, fenced blocks become
    ``code`` sections (``mermaid`` fences become ``diagram`` sections).
    """
    text = str(markdown or "")
    sections: list[dict[str, Any]] = []
    current_heading = ""
    buffer: list[str] = []
    fence_language: str | None = None
    fence_lines: list[str] = []

    def flush_prose() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        buffer = []
        if not body:
            return
        if current_heading:
            sections.append(
                {"type": "explanation", "title": current_heading, "text": body}
            )
        else:
            sections.append({"type": "summary", "text": body})

    for line in text.splitlines():
        if fence_language is not None:
            if line.strip().startswith("```"):
                code_body = "\n".join(fence_lines).strip("\n")
                if fence_language == "mermaid":
                    sections.append(
                        {
                            "type": "diagram",
                            "format": "mermaid",
                            "source": code_body,
                            "alt": current_heading or "Diagram",
                        }
                    )
                else:
                    sections.append(
                        {
                            "type": "code",
                            "language": fence_language or "text",
                            "code": code_body,
                        }
                    )
                fence_language = None
                fence_lines = []
            else:
                fence_lines.append(line)
            continue
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            flush_prose()
            fence_language = (fence_match.group(1) or "").strip().lower()
            fence_lines = []
            continue
        heading_match = _ATX_RE.match(line)
        if heading_match:
            flush_prose()
            current_heading = heading_match.group(2).strip()
            continue
        buffer.append(line)

    if fence_language is not None:
        # Unterminated fence: treat the remainder as prose of the open fence.
        buffer.append("```" + (fence_language or ""))
        buffer.extend(fence_lines)
    flush_prose()

    if not sections:
        sections = [{"type": "summary", "text": text.strip() or "_(empty answer)_"}]

    result = repair_rich_answer({"sections": sections})
    if result.answer is None:  # pragma: no cover - repair of our own output cannot fail
        raise AnswerValidationError(["legacy_conversion_failed"])
    return result.answer


def rich_answer_to_markdown(answer: RichAnswer) -> str:
    """Render a validated RichAnswer back to legacy flat markdown."""
    parts: list[str] = []
    for section in answer.sections:
        kind = section.type
        if kind == "summary":
            parts.append(section.text.strip())
        elif kind == "explanation":
            heading = f"## {section.title}" if section.title else ""
            parts.append(f"{heading}\n\n{section.text.strip()}".strip())
        elif kind == "code":
            label = f"**{section.filename}**\n\n" if section.filename else ""
            parts.append(f"{label}```{section.language}\n{section.code.rstrip()}\n```")
        elif kind == "table":
            header = "| " + " | ".join(section.columns) + " |"
            divider = "| " + " | ".join("---" for _ in section.columns) + " |"
            body = ["| " + " | ".join(row) + " |" for row in section.rows]
            parts.append("\n".join([header, divider, *body]))
        elif kind == "diagram":
            parts.append(f"```{section.format}\n{section.source.strip()}\n```")
        elif kind == "tradeoffs":
            lines = [f"## {section.title or 'Tradeoffs'}", ""]
            for item in section.items:
                lines.append(f"**{item.option}**")
                lines.extend(f"- Pros: {pro}" for pro in item.pros)
                lines.extend(f"- Cons: {con}" for con in item.cons)
                lines.append("")
            parts.append("\n".join(lines).strip())
        elif kind == "common_mistakes":
            lines = [f"## {section.title or 'Common mistakes'}", ""]
            for item in section.items:
                lines.append(
                    f"- **Mistake:** {item.mistake} **Fix:** {item.correction}"
                )
            parts.append("\n".join(lines).strip())
        elif kind == "references":
            lines = [f"## {section.title or 'References'}", ""]
            lines.extend(f"- {item.label}: {item.source_ref}" for item in section.items)
            parts.append("\n".join(lines).strip())
        elif kind == "follow_up":
            lines = [f"## {section.title or 'Follow-up questions'}", ""]
            lines.extend(f"- {prompt}" for prompt in section.prompts)
            parts.append("\n".join(lines).strip())
    return "\n\n".join(part for part in parts if part)
