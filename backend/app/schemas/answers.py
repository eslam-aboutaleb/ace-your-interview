"""Typed rich learning answer schema (issue #3).

Versioned, discriminated answer sections that generation validates before
publishing and the reader renders without ever executing untrusted HTML.

The contract:

* ``answer_schema_version`` identifies the schema; ``"1.0"`` is the first.
* Every section carries a stable ``id`` (lowercase slug) so the reader can
  deep-link anchors and checkpoints can reference positions.
* Markdown text fields never contain executable HTML: script/embed tags,
  inline event handlers and ``javascript:`` URLs are rejected. Renderers must
  still escape everything - validation is the second line of defense, not
  the first.
* Heading levels inside a section's markdown may not skip (``##`` then
  ``####`` is invalid) so the composed page hierarchy stays correct.
* Diagrams are Mermaid-only and must carry an accessible ``alt`` text.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ANSWER_SCHEMA_VERSION = "1.0"

_SECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
_LANGUAGE_RE = re.compile(r"^[a-z0-9][a-z0-9+#.-]{0,40}$")

#: Markup that must never appear in a text field, because a renderer mistake
#: would turn it into script execution. Covers raw tags that can execute or
#: smuggle content, inline event handlers and dangerous URL schemes.
UNSAFE_HTML_RE = re.compile(
    r"<\s*/?\s*(script|iframe|object|embed|form|input|button|textarea|select|"
    r"style|link|meta|base|svg|math)\b"
    r"|\son[a-z]+\s*="
    r"|javascript\s*:"
    r"|data\s*:\s*text/html"
    r"|vbscript\s*:",
    re.IGNORECASE,
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)


def markdown_safety_issues(text: str, *, field: str) -> list[str]:
    """Coded issues for executable-HTML content inside a markdown field."""
    issues = []
    match = UNSAFE_HTML_RE.search(text)
    if match:
        issues.append(f"unsafe_html:{field}")
    return issues


def heading_hierarchy_issues(text: str, *, field: str) -> list[str]:
    """Coded issues for skipped heading levels inside a markdown field."""
    issues = []
    previous = 0
    for match in _HEADING_RE.finditer(text):
        level = len(match.group(1))
        if previous and level > previous + 1:
            issues.append(f"heading_level_skip:{field}")
            break
        previous = level
    return issues


def markdown_field_issues(text: str, *, field: str) -> list[str]:
    return markdown_safety_issues(text, field=field) + heading_hierarchy_issues(
        text, field=field
    )


def _check_markdown(value: str, *, field: str) -> str:
    issues = markdown_field_issues(value, field=field)
    if issues:
        raise ValueError(";".join(issues))
    return value


def _check_safe_text(value: str, *, field: str) -> str:
    issues = markdown_safety_issues(value, field=field)
    if issues:
        raise ValueError(";".join(issues))
    return value


class SectionBase(BaseModel):
    """Common section fields; ``id`` may be empty pre-repair."""

    id: str = ""
    title: str = ""

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if value and not _SECTION_ID_RE.match(value):
            raise ValueError(f"invalid_section_id:{value!r}")
        return value

    @field_validator("title")
    @classmethod
    def _title_safe(cls, value: str) -> str:
        return _check_safe_text(value, field="title")


class SummarySection(SectionBase):
    type: Literal["summary"] = "summary"
    text: str = Field(min_length=1, max_length=20000)

    @field_validator("text")
    @classmethod
    def _text_rules(cls, value: str) -> str:
        return _check_markdown(value, field="summary.text")


class ExplanationSection(SectionBase):
    type: Literal["explanation"] = "explanation"
    text: str = Field(min_length=1, max_length=50000)

    @field_validator("text")
    @classmethod
    def _text_rules(cls, value: str) -> str:
        return _check_markdown(value, field="explanation.text")


class CodeSection(SectionBase):
    type: Literal["code"] = "code"
    language: str = Field(min_length=1, max_length=42)
    code: str = Field(min_length=1, max_length=100000)
    filename: str = ""
    line_highlights: list[int] = Field(default_factory=list)

    @field_validator("language")
    @classmethod
    def _language_shape(cls, value: str) -> str:
        normalised = value.strip().lower()
        if not _LANGUAGE_RE.match(normalised):
            raise ValueError(f"invalid_language:{value!r}")
        return normalised

    @field_validator("line_highlights")
    @classmethod
    def _lines_positive(cls, value: list[int]) -> list[int]:
        if any(line < 1 for line in value):
            raise ValueError("invalid_line_highlight")
        return value


class TableSection(SectionBase):
    type: Literal["table"] = "table"
    columns: list[str] = Field(min_length=1, max_length=20)
    rows: list[list[str]] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def _row_widths(self) -> "TableSection":
        for index, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(f"table_row_width:{index}")
        for cell_row in [self.columns, *self.rows]:
            for cell in cell_row:
                _check_safe_text(str(cell), field="table.cell")
        return self


class DiagramSection(SectionBase):
    type: Literal["diagram"] = "diagram"
    format: Literal["mermaid"] = "mermaid"
    source: str = Field(min_length=1, max_length=50000)
    alt: str = Field(min_length=1, max_length=500)

    @field_validator("source")
    @classmethod
    def _source_safe(cls, value: str) -> str:
        # Mermaid `click ... javascript:` callbacks execute in the page; the
        # renderer also runs with securityLevel "strict" as a second barrier.
        if re.search(
            r"javascript\s*:|vbscript\s*:|\bon[a-z]+\s*=", value, re.IGNORECASE
        ):
            raise ValueError("unsafe_html:diagram.source")
        return value


class TradeoffItem(BaseModel):
    option: str = Field(min_length=1, max_length=300)
    pros: list[str] = Field(default_factory=list, max_length=50)
    cons: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("option")
    @classmethod
    def _option_safe(cls, value: str) -> str:
        return _check_safe_text(value, field="tradeoffs.option")

    @field_validator("pros", "cons")
    @classmethod
    def _items_safe(cls, value: list[str]) -> list[str]:
        for item in value:
            _check_safe_text(item, field="tradeoffs.item")
        return value


class TradeoffsSection(SectionBase):
    type: Literal["tradeoffs"] = "tradeoffs"
    items: list[TradeoffItem] = Field(min_length=1, max_length=50)


class MistakeItem(BaseModel):
    mistake: str = Field(min_length=1, max_length=2000)
    correction: str = Field(min_length=1, max_length=2000)

    @field_validator("mistake", "correction")
    @classmethod
    def _safe(cls, value: str) -> str:
        return _check_markdown(value, field="common_mistakes.item")


class CommonMistakesSection(SectionBase):
    type: Literal["common_mistakes"] = "common_mistakes"
    items: list[MistakeItem] = Field(min_length=1, max_length=50)


class ReferenceItem(BaseModel):
    label: str = Field(min_length=1, max_length=300)
    source_ref: str = Field(min_length=1, max_length=500)
    quote: str = ""

    @field_validator("label", "source_ref", "quote")
    @classmethod
    def _safe(cls, value: str) -> str:
        return _check_safe_text(value, field="references.item")


class ReferencesSection(SectionBase):
    type: Literal["references"] = "references"
    items: list[ReferenceItem] = Field(min_length=1, max_length=100)


class FollowUpSection(SectionBase):
    type: Literal["follow_up"] = "follow_up"
    prompts: list[str] = Field(min_length=1, max_length=20)

    @field_validator("prompts")
    @classmethod
    def _safe(cls, value: list[str]) -> list[str]:
        for prompt in value:
            if not prompt.strip():
                raise ValueError("empty_follow_up_prompt")
            _check_safe_text(prompt, field="follow_up.prompts")
        return value


AnswerSection = Annotated[
    SummarySection
    | ExplanationSection
    | CodeSection
    | TableSection
    | DiagramSection
    | TradeoffsSection
    | CommonMistakesSection
    | ReferencesSection
    | FollowUpSection,
    Field(discriminator="type"),
]

SECTION_TYPES: tuple[str, ...] = (
    "summary",
    "explanation",
    "code",
    "table",
    "diagram",
    "tradeoffs",
    "common_mistakes",
    "references",
    "follow_up",
)


class RichAnswer(BaseModel):
    """A validated, render-safe structured answer."""

    answer_schema_version: str = ANSWER_SCHEMA_VERSION
    sections: list[AnswerSection] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _section_ids_unique(self) -> "RichAnswer":
        ids = [section.id for section in self.sections if section.id]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate_section_id")
        return self

    def to_legacy_markdown(self) -> str:
        """Render to the legacy flat-markdown ``answer`` representation."""
        from app.services.answer_schema import rich_answer_to_markdown

        return rich_answer_to_markdown(self)


def section_types() -> list[str]:
    return list(SECTION_TYPES)


def empty_answer_schema_version() -> str:
    return ""


def parse_section(payload: dict[str, Any]) -> AnswerSection:
    """Parse one raw section payload through the discriminated union."""
    from pydantic import TypeAdapter

    return TypeAdapter(AnswerSection).validate_python(payload)
