"""Generate a per-topic progress summary, with a deterministic offline fallback.

The summary is the semantic half of the no-repeat guarantee: the raw
``questions_asked`` list stops at 200 entries and the prompt only shows a slice
of it, while the summary covers the whole history. Because of that, summary
generation must never block or fail a save — every error path degrades to a
deterministic summary built from the same statistics the AI would have used.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.schemas.models import LLMConfigRequest
from app.services.llm_client import LLMClient
from app.services.progress_store import MAX_SUMMARY_CHARS

logger = logging.getLogger(__name__)

MAX_SUMMARY_WORDS = 200
MAX_PROMPT_QUESTIONS = 40
MAX_PROMPT_SECTIONS = 60

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+\-]*\s*\n(?P<body>[\s\S]*?)\n?```$")

_VALID_DIFFICULTIES = ("easy", "medium", "hard")


def _clamp_choices(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= MAX_PROMPT_SECTIONS:
            break
    return out


def build_attempt_stats(questions: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Aggregate self-assessed outcomes from the learner's current session.

    ``confidence`` is 0 when the learner has not submitted an attempt for that
    question, which is what separates "not answered" from "answered wrong".
    """
    items = [q for q in (questions or []) if isinstance(q, dict)]
    by_difficulty: dict[str, dict[str, int]] = {
        level: {"total": 0, "submitted": 0, "correct": 0} for level in _VALID_DIFFICULTIES
    }
    confidence_total = 0
    confidence_count = 0
    revealed = 0
    submitted = 0
    correct = 0
    incorrect = 0

    for item in items:
        level = str(item.get("difficulty") or "").strip().lower()
        if level not in by_difficulty:
            level = "medium"
        by_difficulty[level]["total"] += 1
        if bool(item.get("revealed")):
            revealed += 1
        is_correct = item.get("is_correct")
        confidence_raw = item.get("confidence")
        try:
            confidence = int(confidence_raw or 0)
        except (TypeError, ValueError):
            confidence = 0
        if is_correct is None and confidence <= 0:
            continue
        submitted += 1
        by_difficulty[level]["submitted"] += 1
        if is_correct is True:
            correct += 1
            by_difficulty[level]["correct"] += 1
        else:
            incorrect += 1
        if confidence > 0:
            confidence_total += confidence
            confidence_count += 1

    return {
        "total": len(items),
        "revealed": revealed,
        "submitted": submitted,
        "correct": correct,
        "incorrect": incorrect,
        "not_submitted": len(items) - submitted,
        "avg_confidence": round(confidence_total / confidence_count, 2) if confidence_count else 0.0,
        "by_difficulty": by_difficulty,
    }


def _difficulty_breakdown(stats: dict[str, Any]) -> str:
    by_difficulty = stats.get("by_difficulty") or {}
    parts: list[str] = []
    for level in _VALID_DIFFICULTIES:
        bucket = by_difficulty.get(level) or {}
        total = int(bucket.get("total", 0) or 0)
        if total <= 0:
            continue
        parts.append(f"{level} {total}")
    return ", ".join(parts)


def build_fallback_summary(
    *,
    topic_title: str,
    sections: list[str] | None = None,
    questions: list[str] | None = None,
    attempt_stats: dict[str, Any] | None = None,
) -> str:
    """Deterministic summary used whenever the LLM path is unavailable.

    Built from the same inputs the model would have seen so the injected
    progress block is still specific enough to steer the next generation.
    """
    stats = attempt_stats if isinstance(attempt_stats, dict) else {}
    section_list = _clamp_choices(sections)
    question_count = len(questions or [])
    total = int(stats.get("total", 0) or 0) or question_count
    submitted = int(stats.get("submitted", 0) or 0)
    correct = int(stats.get("correct", 0) or 0)
    incorrect = int(stats.get("incorrect", 0) or 0)
    avg_confidence = float(stats.get("avg_confidence", 0.0) or 0.0)

    title = str(topic_title or "").strip() or "this topic"
    lines: list[str] = [f'Topic: {title}']

    if total > 0:
        breakdown = _difficulty_breakdown(stats)
        count_line = f"Questions covered in the latest session: {total}"
        if breakdown:
            count_line += f" ({breakdown})"
        lines.append(f"{count_line}.")
    elif question_count > 0:
        lines.append(f"Questions covered so far: {question_count}.")

    if submitted > 0:
        result_line = (
            f"Self-assessed results: {submitted} of {total} answered — "
            f"{correct} mostly right, {incorrect} needing more practice"
        )
        if avg_confidence > 0:
            result_line += f"; average confidence {avg_confidence}/5"
        lines.append(f"{result_line}.")
        weakest = _weakest_difficulty(stats)
        if weakest:
            lines.append(weakest)
    elif total > 0:
        lines.append("No questions have been self-assessed yet, so coverage is unknown.")

    if section_list:
        shown = section_list[:12]
        lines.append(f"Sections already worked through: {'; '.join(shown)}.")
        if len(section_list) > len(shown):
            lines.append(f"Plus {len(section_list) - len(shown)} more section(s).")
    else:
        lines.append("No sections recorded yet.")

    lines.append(
        "Next questions should build on this history: avoid the questions above and "
        "target the sections and difficulties with the thinnest coverage."
    )
    return "\n".join(lines)[:MAX_SUMMARY_CHARS]


def _weakest_difficulty(stats: dict[str, Any]) -> str:
    by_difficulty = stats.get("by_difficulty") or {}
    scored: list[tuple[int, int, str]] = []
    for level in _VALID_DIFFICULTIES:
        bucket = by_difficulty.get(level) or {}
        submitted = int(bucket.get("submitted", 0) or 0)
        total = int(bucket.get("total", 0) or 0)
        correct = int(bucket.get("correct", 0) or 0)
        if submitted <= 0:
            continue
        ratio = correct / submitted
        scored.append((ratio, total, level))
    if not scored:
        return ""
    scored.sort()
    ratio, total, level = scored[0]
    return (
        f"Weakest area: {level} questions ({correct} of {submitted} mostly right across {total} asked)."
    )


def _clean_model_text(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    match = _CODE_FENCE_RE.match(text)
    if match:
        text = match.group("body").strip()
    return text[:MAX_SUMMARY_CHARS]


class ProgressSummarizer:
    """Produce one cumulative progress note per (user, topic)."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def build_summary_prompt(
        self,
        *,
        topic_title: str,
        sections: list[str] | None = None,
        questions: list[str] | None = None,
        attempt_stats: dict[str, Any] | None = None,
        previous_summary: str = "",
    ) -> str:
        section_list = _clamp_choices(sections)
        question_list = [str(q or "").strip() for q in (questions or []) if str(q or "").strip()]
        shown_questions = question_list[:MAX_PROMPT_QUESTIONS]
        stats = attempt_stats if isinstance(attempt_stats, dict) else {}
        previous = str(previous_summary or "").strip()

        sections_text = "\n".join(f"- {item}" for item in section_list) or "(none recorded)"
        questions_text = (
            "\n".join(f"- {item}" for item in shown_questions) or "(none recorded)"
        )
        previous_text = previous or "(no previous note — this is the first one)"

        return f"""You maintain one cumulative study-progress note per learner per topic.

A question generator reads this note at the start of the next session so it can
continue where the learner stopped instead of repeating the same questions.

Topic: {str(topic_title or "").strip() or "this topic"}

Sections the learner has already worked through:
{sections_text}

Questions the learner has already been asked (newest first):
{questions_text}

Self-assessed attempt stats for the latest session:
{stats}

Previous progress note (update it — never restart from scratch):
{previous_text}

Write the updated progress note now, at most {MAX_SUMMARY_WORDS} words.
Cover: what has been covered and answered; which sections are still untouched;
where the learner was unsure (low confidence or "needs more practice"); and what
the next questions should target to move them forward.
Write plain prose or short bullets. No preamble, no meta-commentary, no headings
other than the topic title."""

    async def generate(
        self,
        *,
        user_id: str = "",
        topic_title: str = "",
        sections: list[str] | None = None,
        questions: list[str] | None = None,
        attempt_stats: dict[str, Any] | None = None,
        previous_summary: str = "",
        user_identity: dict[str, Any] | None = None,
        llm_config: LLMConfigRequest | None = None,
    ) -> dict[str, Any]:
        """Return ``{text, provider_used, model_used, source}``; never raises.

        ``task="final"`` routes to the learner's configured model instead of an
        eval/planner route, and a policy block is just another failure mode that
        falls back — a learner without a personal credential still gets a usable
        progress note.
        """
        fallback = build_fallback_summary(
            topic_title=topic_title,
            sections=sections,
            questions=questions,
            attempt_stats=attempt_stats,
        )
        prompt = self.build_summary_prompt(
            topic_title=topic_title,
            sections=sections,
            questions=questions,
            attempt_stats=attempt_stats,
            previous_summary=previous_summary,
        )

        provider_used = ""
        model_used = ""
        try:
            result = await self.llm_client.completion(
                prompt,
                llm_config,
                user_identity=user_identity,
                task="final",
            )
            metadata = result.get("metadata") or {}
            provider_used = str(metadata.get("provider", "") or "")
            model_used = str(metadata.get("model", "") or "")
            if not result.get("success"):
                logger.info(
                    "Progress summary fell back for user=%s topic=%s code=%s error=%s",
                    user_id,
                    topic_title,
                    result.get("error_code", ""),
                    str(result.get("error", ""))[:200],
                )
                return {
                    "text": fallback,
                    "provider_used": provider_used,
                    "model_used": model_used,
                    "source": "fallback",
                }
            text = _clean_model_text(result.get("analysis", ""))
            if not text:
                logger.info(
                    "Progress summary returned empty analysis for user=%s topic=%s; using fallback",
                    user_id,
                    topic_title,
                )
                return {
                    "text": fallback,
                    "provider_used": provider_used,
                    "model_used": model_used,
                    "source": "fallback",
                }
            return {
                "text": text,
                "provider_used": provider_used,
                "model_used": model_used,
                "source": "ai",
            }
        except Exception:
            logger.exception(
                "Progress summary generation failed for user=%s topic=%s; using fallback",
                user_id,
                topic_title,
            )
            return {
                "text": fallback,
                "provider_used": "",
                "model_used": "",
                "source": "fallback",
            }
