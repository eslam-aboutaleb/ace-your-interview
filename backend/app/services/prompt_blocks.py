"""Reusable prompt-building blocks shared across generation flows."""

from __future__ import annotations

from typing import Iterable


def bullet_lines(lines: Iterable[str]) -> str:
    cleaned = [str(line).strip() for line in lines if str(line).strip()]
    return "\n".join(f"- {line}" for line in cleaned)


def section(title: str, lines: Iterable[str]) -> str:
    body = bullet_lines(lines)
    if not body:
        return ""
    return f"{title}:\n{body}"


def render_contract(
    *,
    schema_label: str = "Return ONLY valid JSON",
    schema_block: str,
    rules: Iterable[str],
) -> str:
    rules_block = section("Rules", rules)
    return f"{schema_label}:\n{schema_block.strip()}\n\n{rules_block}".strip()


def optional_context_block(label: str, value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f"\n\n{label}:\n{text[:max_chars]}"
