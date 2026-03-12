"""Shared built-in topic catalog helpers."""

from __future__ import annotations

PROBLEM_SOLVING_TOPIC_ID = "00-problem-solving-and-algorithms"
PROBLEM_SOLVING_TITLE = "Problem Solving and Algorithms"
PROBLEM_SOLVING_DESCRIPTION = (
    "Master algorithmic problem solving from beginner to senior with language-specific roadmaps, "
    "patterns, and interview tradeoffs."
)
PROBLEM_SOLVING_TRACK = "backend"
PROBLEM_SOLVING_LEVELS = ["junior", "mid", "senior"]
PROBLEM_SOLVING_LANGUAGE_OPTIONS = [
    "python",
    "java",
    "cpp",
    "javascript",
    "csharp",
    "go",
]
PROBLEM_SOLVING_DEFAULT_LANGUAGE = "python"
PROBLEM_SOLVING_TARGET_SECTIONS = 120


def is_problem_solving_topic(topic_id: str) -> bool:
    return str(topic_id or "").strip() == PROBLEM_SOLVING_TOPIC_ID


def problem_solving_base_topic() -> dict:
    return {
        "id": PROBLEM_SOLVING_TOPIC_ID,
        "title": PROBLEM_SOLVING_TITLE,
        "description": PROBLEM_SOLVING_DESCRIPTION,
        "track": PROBLEM_SOLVING_TRACK,
        "levels": list(PROBLEM_SOLVING_LEVELS),
        "sections": [],
        "raw_content": "",
        "requires_programming": True,
        "language_options": list(PROBLEM_SOLVING_LANGUAGE_OPTIONS),
        "selected_language": PROBLEM_SOLVING_DEFAULT_LANGUAGE,
        "response_detail": "very_detailed",
        "is_dynamic_topic": True,
        "content_ready": False,
    }
