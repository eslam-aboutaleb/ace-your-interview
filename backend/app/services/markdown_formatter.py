"""Utilities for cleaning markdown readability without breaking structure."""

from __future__ import annotations

import re

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_BULLET_LINE_RE = re.compile(r"^\s*([-*]|\d+\.)\s+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


def _split_long_paragraph(text: str) -> str:
    paragraph = " ".join(text.strip().split())
    if len(paragraph) <= 360:
        return paragraph

    # Prefer splitting by sentence boundaries.
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    if len(sentences) <= 1:
        return paragraph

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        tentative = " ".join([*current, sentence]).strip()
        if current and len(tentative) > 280:
            chunks.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current).strip())
    return "\n\n".join(chunk for chunk in chunks if chunk)


def _format_non_code_markdown(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    paragraph_buf: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_buf:
            return
        joined = " ".join(part.strip() for part in paragraph_buf if part.strip())
        if joined:
            out.append(_split_long_paragraph(joined))
        paragraph_buf.clear()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            if not out or out[-1] != "":
                out.append("")
            continue

        if _HEADING_RE.match(stripped) or _TABLE_LINE_RE.match(stripped) or _BULLET_LINE_RE.match(stripped):
            flush_paragraph()
            out.append(stripped)
            continue

        paragraph_buf.append(stripped)

    flush_paragraph()

    # Remove repeated blank lines while preserving section separation.
    compact: list[str] = []
    for line in out:
        if line == "" and compact and compact[-1] == "":
            continue
        compact.append(line)
    return "\n".join(compact).strip()


def format_markdown_readable(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""

    segments: list[str] = []
    cursor = 0
    for match in _CODE_FENCE_RE.finditer(text):
        if match.start() > cursor:
            segments.append(_format_non_code_markdown(text[cursor : match.start()]))
        segments.append(match.group(0).strip())
        cursor = match.end()
    if cursor < len(text):
        segments.append(_format_non_code_markdown(text[cursor:]))

    return "\n\n".join(seg for seg in segments if seg.strip()).strip()
