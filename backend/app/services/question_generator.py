"""Generate interview questions & answers via LLM with strict validation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, AsyncIterator, Callable, Optional

from app.schemas.models import (
    GenerateQuestionsResponse,
    GenerateQuestionsV2Response,
    GenerateQuizResponse,
    GenerateQuizV2Response,
    LLMConfigRequest,
    QuestionAnswer,
    QuestionAnswerV2,
    QuizChoice,
    QuizDistractorRationale,
    QuizQuestion,
    QuizQuestionType,
    QuizQuestionV2,
)
from app.services.llm_client import (
    TERMINAL_ERROR_CODES,
    LLMClient,
    parse_json_object,
    supported_options,
)
from app.services.llm_policy import (
    APPROVAL_REQUIRED_CODE,
    APPROVAL_REQUIRED_MESSAGE,
    PERSONAL_CREDENTIAL_REQUIRED_CODE,
    PERSONAL_CREDENTIAL_REQUIRED_MESSAGE,
    STUDY_APP_NOT_ASSIGNED_CODE,
    STUDY_APP_NOT_ASSIGNED_MESSAGE,
    LLMServiceApprovalRequiredError,
    PersonalCredentialRequiredError,
    StudyAppLLMNotAssignedError,
    raise_if_policy_blocked_result,
)
from app.services.llm_skills import normalise_track, skill_context_block
from app.services.learning_store import record_generation_run_safe
from app.services.markdown_formatter import format_markdown_readable
from app.services.mcp_gateway import MCPGateway
from app.services.question_text import normalise_question
from app.services.prompt_blocks import (
    UNTRUSTED_CLAUSE,
    render_contract,
    untrusted_block,
)
from app.services.prompt_registry import default_registry
from app.services.topic_catalog import (
    PROBLEM_SOLVING_DEFAULT_LANGUAGE,
    is_problem_solving_topic,
)

logger = logging.getLogger(__name__)

_MAX_DOC_CONTEXT = 45000
_MAX_TOPIC_CONTEXT = 9000
# The progress block rides along on every generation prompt, so it is capped
# hard. It is the mitigation for the question-list caps below: once the
# normalizer's 80-item cap and the uniqueness block's 60-item slice are reached,
# the oldest history drops out of the prompt, but this block still describes it.
_MAX_PROGRESS_BLOCK_CHARS = 1200
# Skill-taxonomy context injected into every generation prompt. The
# taxonomy is module-authored (trusted), so it is not fenced like the
# untrusted blocks, but it is still capped so the injection stays a
# fixed-size addition to the prompt. The full three-domain block
# (technical + interview + educational) measures ~6.4k-7.9k chars
# depending on level/session, so 8000 keeps every domain intact for
# every combination while bounding the injection.
_MAX_SKILL_CONTEXT_CHARS = 8000
_SKILL_CONTEXT_LABEL = (
    "Skill context (apply these when writing questions, explanations, and expected signals)"
)
# Per-flow output ceiling. A caller/config/preference value may lower it but
# never raise it (enforced by ``LLMClient``).
_MIN_QUESTIONS_MAX_TOKENS_CAP = 600
_MAX_QUESTIONS_MAX_TOKENS_CAP = 8000
_TOKENS_PER_QUESTION = 140


def _questions_max_tokens_cap(count: int) -> int:
    """Scale the questions/quiz output budget with the requested item count."""
    scaled = int(count or 0) * _TOKENS_PER_QUESTION
    return max(
        _MIN_QUESTIONS_MAX_TOKENS_CAP,
        min(_MAX_QUESTIONS_MAX_TOKENS_CAP, scaled),
    )


#: The option-narrowing shim now lives in :mod:`app.services.llm_client`
#: as :func:`supported_options`; the module-level alias keeps the
#: historical call sites (and their tests) working against the one
#: shared implementation.
_supported_options = supported_options



_PROGRESS_BLOCK_LABEL = (
    "Learner progress on this topic so far "
    "(use it to continue from where they stopped, never to repeat covered material)"
)
_TARGET_OBJECTIVES_BLOCK_LABEL = (
    "Learning objectives the learner is weakest on "
    "(prioritize questions that assess these objectives)"
)
_MAX_TARGET_OBJECTIVES_BLOCK_CHARS = 800
_BASE_MAX_ATTEMPTS = 5
_MAX_TOTAL_ATTEMPTS = 18
_MAX_RECOVERY_ATTEMPTS = 24
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_LEVELS = {"junior", "mid", "senior"}
#: Requested levels that delegate depth to the learner's attempt history:
#: a level the client left out, and the explicit ``"auto"`` opt-in.
_AUTO_LEVEL_ALIASES = {"", "auto"}
_PROBLEM_SOLVING_SECTION_HEADINGS = (
    "Problem",
    "Solution Walkthrough",
    "Complexity",
    "Code",
)
_PROBLEM_SOLVING_CODE_FENCE_RE = re.compile(
    r"```(?P<language>[a-zA-Z0-9_#+-]+)\s*\n(?P<code>[\s\S]*?)```",
    re.MULTILINE,
)
_PROBLEM_SOLVING_H3_RE = re.compile(r"^\s*###\s+(.+?)\s*$", re.MULTILINE)
_QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "why",
    "with",
    "would",
    "you",
}
_GROUNDING_GENERIC_TOKENS = {
    "algorithm",
    "application",
    "approach",
    "architecture",
    "backend",
    "concept",
    "data",
    "design",
    "engineer",
    "engineering",
    "implementation",
    "interview",
    "pattern",
    "problem",
    "programming",
    "service",
    "software",
    "solution",
    "solving",
    "structure",
    "system",
    "technical",
    "topic",
    "tradeoff",
}


def _question_id(topic_id: str, question: str) -> str:
    digest = hashlib.sha1(f"{topic_id}:{question.strip().lower()}".encode("utf-8")).hexdigest()
    return f"{topic_id}:{digest[:14]}"


def _normalise_question(text: str) -> str:
    return normalise_question(text)


def _question_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", _normalise_question(text))
    normalized: set[str] = set()
    for word in words:
        if len(word) <= 2 or word in _QUESTION_STOPWORDS:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = f"{word[:-3]}y"
        elif word.endswith("s") and len(word) > 3 and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        normalized.add(word)
    return normalized


def _grounding_tokens(text: str) -> set[str]:
    tokens = _question_tokens(text)
    filtered = {token for token in tokens if token not in _GROUNDING_GENERIC_TOKENS}
    return filtered or tokens


def _is_near_duplicate_question(candidate: str, seen_questions: list[str]) -> bool:
    cand_norm = _normalise_question(candidate)
    if not cand_norm:
        return True
    cand_tokens = _question_tokens(candidate)
    for seen in seen_questions:
        seen_norm = _normalise_question(seen)
        if not seen_norm:
            continue
        if cand_norm == seen_norm:
            return True
        if (cand_norm in seen_norm or seen_norm in cand_norm) and min(len(cand_norm), len(seen_norm)) >= 24:
            return True
        if SequenceMatcher(None, cand_norm, seen_norm).ratio() >= 0.87:
            return True
        seen_tokens = _question_tokens(seen)
        if cand_tokens and seen_tokens:
            overlap = len(cand_tokens & seen_tokens) / max(1, len(cand_tokens | seen_tokens))
            if overlap >= 0.65:
                return True
    return False


def _clamp_content(doc_content: str) -> str:
    return doc_content[:_MAX_DOC_CONTEXT]


def _normalise_level(level: Optional[str]) -> str:
    lv = (level or "mid").strip().lower()
    return lv if lv in _VALID_LEVELS else "mid"


def resolve_level(level: Optional[str], recommended_level: Optional[str] = None) -> str:
    """Resolve the level a generation flow runs at.

    An explicit level (``junior``/``mid``/``senior``) always wins, so a
    client that pins its level — including the level this defaulted to
    before adaptive difficulty existed — is unaffected. An omitted level
    or the literal ``"auto"`` delegates to ``recommended_level``, the
    history-derived level from
    :meth:`LearningStore.get_level_recommendation`, and falls back to
    ``"mid"`` when there is no recommendation (no user context, or a
    learner with no attempt history for the topic yet).
    """
    explicit = str(level or "").strip().lower()
    if explicit in _VALID_LEVELS:
        return explicit
    if explicit in _AUTO_LEVEL_ALIASES:
        recommendation = str(recommended_level or "").strip().lower()
        if recommendation in _VALID_LEVELS:
            return recommendation
    return "mid"


def _level_is_auto(level: Optional[str]) -> bool:
    """True when a request left its level to the learner's history."""
    return str(level or "").strip().lower() in _AUTO_LEVEL_ALIASES


def is_auto_level(level: Optional[str]) -> bool:
    """Public form of :func:`_level_is_auto` for the other flow modules.

    ``InterviewGenerator`` needs the same "would this level otherwise be
    defaulted?" question the question and quiz flows ask, and the alias set
    must stay in one place.
    """
    return _level_is_auto(level)


def _attempt_budget(target_count: int) -> int:
    target = max(1, int(target_count or 1))
    return min(_MAX_TOTAL_ATTEMPTS, max(_BASE_MAX_ATTEMPTS, target + 3))


def _recovery_budget(missing_count: int) -> int:
    missing = max(1, int(missing_count or 1))
    return min(_MAX_RECOVERY_ATTEMPTS, max(3, missing * 3))


def _normalise_existing_questions(existing_questions: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in existing_questions or []:
        text = str(raw or "").strip()
        if not text:
            continue
        norm = _normalise_question(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
        if len(out) >= 80:
            break
    return out


def _merge_existing_questions(
    existing_questions: list[str] | None,
    additional_existing_questions: list[str] | None,
) -> list[str]:
    """Merge the client's per-checkpoint list with the server-stored history.

    Client-first: that preserves today's per-checkpoint behaviour exactly (the
    client list wins the budget), and the stored history then fills what is left
    of the 80-item cap. Documented limitation: past the cap the oldest stored
    questions are dropped from the prompt, which is what the progress block is
    there to compensate for.
    """
    return _normalise_existing_questions(
        [*(existing_questions or []), *(additional_existing_questions or [])]
    )


def _progress_block(prior_progress: str) -> str:
    # ``prior_progress`` is model-generated text derived from stored learner
    # history, so it is fenced like any other untrusted block.
    return untrusted_block(
        _PROGRESS_BLOCK_LABEL,
        prior_progress,
        _MAX_PROGRESS_BLOCK_CHARS,
    )


def _target_objectives_block(target_objectives: list[str] | None) -> str:
    # ``target_objectives`` are learning objectives derived from stored
    # learner history, so they are fenced like any other untrusted block.
    objectives = [
        str(obj).strip()
        for obj in (target_objectives or [])
        if str(obj).strip()
    ]
    if not objectives:
        return ""
    blob = "\n".join(f"- {obj}" for obj in objectives[:20])
    return untrusted_block(
        _TARGET_OBJECTIVES_BLOCK_LABEL,
        blob,
        _MAX_TARGET_OBJECTIVES_BLOCK_CHARS,
    )


def _learner_state_block(learner_state: str) -> str:
    # ``learner_state`` arrives already gathered, bounded, and fenced by
    # learner_memory.build_learner_state_block; the only job here is the
    # separating newline, so an empty learner keeps the prompt unchanged.
    text = str(learner_state or "").strip()
    if not text:
        return ""
    return f"\n{text}\n"


def _skill_context_section(level: str, session_type: str, track: str = "") -> str:
    """Render the bounded skill-taxonomy context for a generation prompt.

    The taxonomy is module-authored (trusted), so unlike the untrusted
    blocks it is not fenced; it is still capped so the injection stays
    a fixed-size addition to every prompt.

    ``track`` selects the per-track technical pack (backend, frontend,
    system design, AI stack). An empty or unknown track keeps the DSA
    pack the problem-solving flow is generated against.
    """
    block = skill_context_block(
        level=level,
        session_type=session_type,
        track=track,
        include_educational=True,
    )
    if not block:
        return ""
    return f"{_SKILL_CONTEXT_LABEL}:\n{block[:_MAX_SKILL_CONTEXT_CHARS]}"


def _skill_track_for_topic(topic_id: str, track: str) -> str:
    """Return the track that selects the technical skill pack.

    The problem-solving topic is catalogue-tracked as ``backend`` but is
    generated against the DSA pack, so its track is deliberately dropped
    to keep that flow byte-identical.
    """
    if is_problem_solving_topic(topic_id):
        return ""
    return normalise_track(track)


def _skill_track_for_topics(topics_content: list[dict]) -> str:
    """Return the dominant track across the requested topics.

    A multi-topic quiz can mix tracks; the first topic that declares one
    picks the pack, and a quiz with no declared track keeps DSA.
    """
    for topic in topics_content or []:
        track = normalise_track(topic.get("track"))
        if track:
            return track
    return ""


def _problem_solving_language(preferred_language: str) -> str:
    language = (preferred_language or "").strip().lower()
    return language or PROBLEM_SOLVING_DEFAULT_LANGUAGE


def _study_code_language(
    preferred_language: str,
    *,
    problem_solving_mode: bool,
    requires_programming: bool,
) -> str:
    language = (preferred_language or "").strip().lower()
    if problem_solving_mode:
        return _problem_solving_language(language)
    if language:
        return language
    if requires_programming:
        return PROBLEM_SOLVING_DEFAULT_LANGUAGE
    return ""


_CONCEPTUAL_COACH_LABELS = (
    "**Answer:**",
    "**Detailed explanation:**",
    "**Common mistake:**",
)
_PROBLEM_SOLVING_WALKTHROUGH_LABELS = (
    "**Direct answer:**",
    "**Detailed explanation:**",
    "**Why this works:**",
    "**Common mistake:**",
)


def _question_tutoring_arc_rules(problem_solving_mode: bool, language: str = "") -> list[str]:
    if problem_solving_mode:
        walkthrough_labels = ", ".join([f"`{label}`" for label in _PROBLEM_SOLVING_WALKTHROUGH_LABELS])
        return [
            "Write each answer as a direct, detailed study answer for the learner.",
            *_problem_solving_required_markdown(language),
            "Answer the actual question in full; do not turn the response into generic coaching about how to solve problems.",
            "In `### Problem`, begin with `**What the problem is asking:**`.",
            f"In `### Solution Walkthrough`, include these bold labels in this order: {walkthrough_labels}.",
            (
                "In `### Complexity`, include `**Time and space:**` followed by "
                "`**Tradeoff / scaling caveat:**`."
            ),
            "Under `### Solution Walkthrough`, make the first sentence under `**Direct answer:**` name the final algorithm or solution directly.",
            "Use `**Detailed explanation:**` to walk through the actual solution logic, not generic interview advice.",
            "Keep the answer generic and topic-grounded; do not inject named canned techniques unless the course content naturally justifies them.",
        ]

    coach_labels = ", ".join([f"`{label}`" for label in _CONCEPTUAL_COACH_LABELS])
    return [
        "Write each answer as a direct, detailed study answer for the learner.",
        f"For non-problem-solving answers, use these bold lead-ins in this exact order: {coach_labels}.",
        "Do not add `###` headings to non-problem-solving answers.",
        "The first sentence after `**Answer:**` must directly answer the question.",
        "Answer the actual question itself, not how a learner should approach answering it.",
        "Do not use meta-guideline phrasing such as `the candidate should`, `you should answer by`, or `when discussing this in an interview` anywhere in the answer.",
        "Keep all required answer segments regardless of depth setting; `response_detail` only changes how much detail lives inside them.",
    ]


def _question_auxiliary_field_rules(problem_solving_mode: bool) -> list[str]:
    reasoning_rule = (
        "Set `reasoning_summary` to a short mental model or invariant that explains how to choose the approach before coding."
        if problem_solving_mode
        else "Set `reasoning_summary` to a short mental-model takeaway, not answer-writing advice."
    )
    return [
        "Set `learning_objective` to a sentence that starts with `After this question, the learner should be able to...`.",
        reasoning_rule,
    ]


def _question_retry_guidance(problem_solving_mode: bool) -> str:
    if problem_solving_mode:
        return "\n".join(
            [
                "- Regenerate each item as a fresh, detailed answer to the actual problem.",
                "- Keep exactly these H3 headings in order: `### Problem`, `### Solution Walkthrough`, `### Complexity`, `### Code`.",
                "- In `### Problem`, start with `**What the problem is asking:**` and summarize the inputs, outputs, and key constraint.",
                "- In `### Solution Walkthrough`, use `**Direct answer:**`, `**Detailed explanation:**`, `**Why this works:**`, and `**Common mistake:**` in that order.",
                "- Do not fill the answer with generic coaching phrases like `start by`, `first clarify`, or `the interviewer wants` unless they are directly tied to the actual solution.",
                "- Use numbered lists only when they describe the actual algorithm or data flow, and use a compact comparison table when showing why the chosen approach beats a weaker alternative.",
                "- In `### Complexity`, include `**Time and space:**` and `**Tradeoff / scaling caveat:**`.",
                "- In `### Code`, include exactly one commented fenced code block in the selected language, then `**Invariant note:**`.",
                "- Keep the answer specific enough that a learner could study from it without extra notes.",
            ]
        )
    return "\n".join(
        [
            "- Regenerate each item as a fresh, detailed answer to the actual question.",
            "- Use only bold lead-ins in this exact order: `**Answer:**`, `**Detailed explanation:**`, `**Common mistake:**`.",
            "- The first sentence after `**Answer:**` must directly answer the question.",
            "- Do not spend the answer telling the learner how to answer. Explain the concept, mechanism, tradeoff, or failure mode directly.",
            "- Use numbered lists for actual steps or flows, bullet lists for pros/cons or failure modes, and a compact table when the answer is mainly comparing alternatives.",
            "- Include the tradeoff, bottleneck, or failure mode that would matter in a real interview follow-up.",
        ]
    )


def _common_interview_question_rules(problem_solving_mode: bool, *, topic_scope: str) -> list[str]:
    if problem_solving_mode:
        return [
            f"Questions must sound like common coding interview questions for {topic_scope}, not obscure puzzle variants or textbook trivia.",
            "Prefer high-frequency interviewer prompts first: clarify the problem, state constraints, choose the algorithm or data structure, justify tradeoffs, and reason about edge cases.",
            "Start with the standard interview patterns for this topic before unusual twists unless the documentation makes the twist central.",
        ]
    return [
        f"Questions must sound like common interview questions for {topic_scope}, not arbitrary trivia or internal prompt exercises.",
        "Prefer the questions interviewers repeatedly ask for this topic: what it is, why it matters, how it works, when to use it, tradeoffs, failure modes, debugging, testing, and scaling.",
        "Use the documentation to make those common interview questions concrete for the current topic instead of inventing niche hypotheticals.",
    ]


def _difficulty_generation_rules(problem_solving_mode: bool, difficulty: Optional[str]) -> list[str]:
    diff = str(difficulty or "").strip().lower()
    if diff == "easy":
        return [
            (
                "Easy questions should stay foundational: prefer high-frequency interview questions, direct "
                "tradeoff explanations, and the first practical implementation concern."
            )
        ]
    if diff == "medium":
        return [
            (
                "Medium questions should focus on common architecture or implementation scenarios, practical "
                "tradeoffs, bottlenecks, and realistic follow-up reasoning."
            )
        ]
    if diff == "hard":
        if problem_solving_mode:
            return [
                (
                    "Hard questions should probe tricky edge cases, follow-up twists, tighter constraints, or "
                    "alternative designs that break simpler solutions."
                )
            ]
        return [
            (
                "Hard questions should probe tricky edge cases, distributed-system anomalies, scaling bottlenecks, "
                "failure modes, or multi-region consistency tradeoffs."
            )
        ]
    if problem_solving_mode:
        return [
            (
                "When difficulty is not fixed, keep the batch naturally progressive: start with common interview "
                "problem shapes, then move toward tougher edge cases or follow-up twists later in the batch. "
                "Do not enforce exact easy/medium/hard quotas."
            )
        ]
    return [
        (
            "When difficulty is not fixed, keep the batch naturally progressive: lead with common interview "
            "questions, then move toward harder tradeoffs, bottlenecks, failure modes, or scale problems later "
            "in the batch. Do not enforce exact easy/medium/hard quotas."
        )
    ]


def _study_markdown_rules(*, code_language: str, problem_solving_mode: bool) -> list[str]:
    rules = [
        "When the answer compares options, tradeoffs, or categories, use a compact GFM table instead of burying the comparison in prose.",
        "If the question is mainly asking for a comparison, prefer a compact GFM table for the main side-by-side contrast.",
        "If you use a table, output valid GFM table syntax.",
        "one row per line.",
        "include a separator row (e.g. `| --- | --- |`).",
        "When the answer includes request flow, debugging flow, implementation sequence, or evaluation criteria, use numbered lists instead of dense prose.",
        "Use bullet lists for components, pros/cons, bottlenecks, failure modes, or checklist-style explanations.",
        "When the topic touches storage, caching, messaging, coordination, retries, replication, or multi-service flows, explicitly cover bottlenecks, scaling constraints, failure modes, and relevant consistency/CAP implications.",
        "You may include fenced Mermaid diagrams or clear ASCII diagrams when architecture or data flow is genuinely easier to understand visually.",
        "If you include fences, always use explicit language tags (for example: ```python, ```mermaid).",
    ]
    if problem_solving_mode:
        rules.append("If you compare a brute-force approach with the chosen approach, summarize that tradeoff clearly, using a compact table when it improves clarity.")
    else:
        if code_language:
            rules.append(
                f'If code materially clarifies the concept, use a concise fenced "{code_language}" example; otherwise keep the answer conceptual.'
            )
        else:
            rules.append("Use code examples only when they materially improve clarity.")
    return rules


def _grounding_anchor_phrases(
    *,
    topic_title: str,
    section_title: Optional[str],
    content: str,
    limit: int = 10,
) -> list[str]:
    candidates = [str(section_title or "").strip(), str(topic_title or "").strip()]
    candidates.extend(_extract_content_fragments(content, limit=max(limit * 3, 18)))

    anchors: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        text = re.sub(r"\s+", " ", str(raw or "").strip())
        if len(text) < 4:
            continue
        if len(text) > 90:
            text = text[:90].rsplit(" ", 1)[0].strip() or text[:90]
        norm = _normalise_question(text)
        if not norm or norm in seen:
            continue
        if not _question_tokens(text):
            continue
        seen.add(norm)
        anchors.append(text)
        if len(anchors) >= limit:
            break
    return anchors


def _render_grounding_anchors_block(anchors: list[str]) -> str:
    if not anchors:
        return ""
    # Anchors are phrases lifted verbatim from the topic title, section title, and
    # course content, so the whole list is untrusted. Fenced, because an anchor
    # containing a closing tag would otherwise end the list and turn the rest of
    # the prompt into instructions the model should follow.
    return (
        "\nGrounding anchors from the topic content. Each generated question must "
        "clearly target one of these:\n"
        + untrusted_block(
            "grounding_anchors",
            "\n".join([f"- {anchor}" for anchor in anchors]),
        )
    )


def _normalise_content_fragment(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _question_is_grounded_to_anchors(
    question: str,
    *,
    topic_title: str,
    anchors: list[str],
    problem_solving_mode: bool,
) -> bool:
    question_tokens = _question_tokens(question)
    if not question_tokens:
        return False

    topic_tokens = _grounding_tokens(topic_title)
    if question_tokens & topic_tokens:
        return True

    for anchor in anchors:
        anchor_tokens = _grounding_tokens(anchor)
        if not anchor_tokens:
            continue
        overlap = question_tokens & anchor_tokens
        if any(len(token) >= 5 for token in overlap) or len(overlap) >= 2:
            return True
    return False


def _validate_topic_grounding(
    item: dict[str, Any],
    *,
    topic_title: str,
    anchors: list[str],
    problem_solving_mode: bool,
) -> tuple[bool, str]:
    question = str(item.get("question", "")).strip()
    if _question_is_grounded_to_anchors(
        question,
        topic_title=topic_title,
        anchors=anchors,
        problem_solving_mode=problem_solving_mode,
    ):
        return True, ""
    return False, "question_not_grounded_to_topic"


def _fallback_difficulty_for_index(index: int, total: int, requested: Optional[str]) -> str:
    diff = str(requested or "").strip().lower()
    if diff in _VALID_DIFFICULTIES:
        return diff
    if total <= 1:
        return "medium"
    bucket = (max(0, index) * 3) // max(1, total)
    if bucket <= 0:
        return "easy"
    if bucket == 1:
        return "medium"
    return "hard"


def _default_learning_objective(problem_solving_mode: bool) -> str:
    if problem_solving_mode:
        return (
            "After this question, the learner should be able to explain the core invariant and map it to a working implementation."
        )
    return (
        "After this question, the learner should be able to explain the current checkpoint and justify its practical tradeoffs."
    )


def _default_reasoning_summary(problem_solving_mode: bool) -> str:
    if problem_solving_mode:
        return "State the invariant first, then choose the structure that preserves it with the least extra work."
    return "Start from the core constraint, then connect it to the tradeoff that makes the answer correct."


def _problem_solving_required_markdown(language: str) -> list[str]:
    headings = ", ".join([f"`### {heading}`" for heading in _PROBLEM_SOLVING_SECTION_HEADINGS])
    return [
        f"Use exactly these markdown headings in this exact order: {headings}.",
        "Do not add extra `###` headings before, between, or after those sections.",
        f'Under `### Code`, include exactly one fenced `{language}` block.',
        "The code block must contain practical code, not pseudocode.",
        "Add comments that explain each key step or block in the code.",
        "After the code block, add `**Invariant note:**` describing what part of the code preserves the core invariant.",
    ]


_PROBLEM_SOLVING_DEFAULT_SCENARIO_ID = "two_sum_hash_map"
_PROBLEM_SOLVING_FALLBACK_SCENARIOS: tuple[dict[str, str], ...] = (
    {
        "id": "two_sum_hash_map",
        "question": (
            "In {source_scope}, consider an array-based input related to {focus}. Return the two indices "
            "whose values satisfy a target-sum constraint. Explain your approach, complexity, and "
            "edge-case handling as you would in a live interview."
        ),
        "problem": (
            "Translate the prompt into inputs, outputs, and constraints for {focus}. You need a one-pass "
            "strategy that quickly checks whether the complement of the current value has already appeared."
        ),
        "walkthrough": (
            "Reject O(n^2) brute force once the complement relationship is clear. Use a hash map from "
            "value to index; for each number, look up its complement in O(1), then store the current "
            "value after the check so the same element is never reused."
        ),
        "complexity": (
            "The array is scanned once, so time is O(n). The hash map can store up to n values, so extra "
            "space is O(n). The tradeoff is additional memory for linear-time performance."
        ),
        "reasoning_summary": (
            "Identify the complement invariant first, then choose a hash map for one-pass lookup speed."
        ),
    },
    {
        "id": "longest_substring_sliding_window",
        "question": (
            "In {source_scope}, analyze a string scenario around {focus} and return the length of the "
            "longest substring without repeating characters. Explain how you reason about pointer movement."
        ),
        "problem": (
            "You need the best window length with all unique characters for {focus}. Rechecking every "
            "substring is too slow, so you need an incremental way to maintain validity while scanning."
        ),
        "walkthrough": (
            "Use a sliding window with a left pointer and a map of last-seen indices. When a repeated "
            "character appears inside the active window, move the left pointer past the previous index. "
            "Update window length on each step and keep the maximum."
        ),
        "complexity": (
            "Each character is processed at most twice (by right/left movement), so time is O(n). The "
            "last-seen map stores up to unique characters, so space is O(min(n, alphabet_size))."
        ),
        "reasoning_summary": (
            "Maintain a valid window invariant and move pointers only when the invariant is violated."
        ),
    },
    {
        "id": "merge_intervals_sorting",
        "question": (
            "Using the context of {focus} in {source_scope}, merge all overlapping intervals `[start, end]` "
            "and return the result. Explain your sorting and merge decisions clearly."
        ),
        "problem": (
            "Overlaps can appear in many positions, so local pair checks are unreliable unless intervals "
            "are ordered. The task is to collapse intersecting ranges into canonical disjoint intervals."
        ),
        "walkthrough": (
            "Sort intervals by start. Iterate once while building an output list: if the current interval "
            "does not overlap the last merged interval, append it; otherwise extend the previous end with "
            "the maximum endpoint."
        ),
        "complexity": (
            "Sorting dominates at O(n log n). The merge pass is O(n). Extra output storage is O(n) in the "
            "worst case when no intervals overlap."
        ),
        "reasoning_summary": (
            "Sort to linearize overlap checks, then maintain a single merged frontier while scanning."
        ),
    },
    {
        "id": "top_k_frequent_heap",
        "question": (
            "Given data from {source_scope} related to {focus}, return the `k` most frequent elements. "
            "Explain why your data structure choices fit interview constraints."
        ),
        "problem": (
            "You must rank elements by frequency without sorting every element-value pair when only top `k` "
            "results are needed. The challenge is balancing counting and extraction cost."
        ),
        "walkthrough": (
            "Count frequencies with a hash map, then use a heap to extract the top `k` entries by frequency. "
            "A max-heap gives direct top retrieval; alternatively, a size-limited min-heap can improve memory "
            "when `k` is much smaller than the number of unique values."
        ),
        "complexity": (
            "Frequency counting is O(n). Building and popping from a heap is O(m + k log m) for max-heap "
            "(or O(m log k) with a size-k min-heap), where m is number of unique values."
        ),
        "reasoning_summary": (
            "Separate counting from ranking, then use heap operations to avoid full-frequency sorting."
        ),
    },
    {
        "id": "search_rotated_binary_search",
        "question": (
            "In {source_scope}, you receive a rotated sorted array tied to {focus} and a target value. "
            "Return the index or -1 if not found, and explain how you adapt binary search under rotation."
        ),
        "problem": (
            "Standard binary search assumes full ordering, but rotation breaks global monotonicity. You "
            "must still use O(log n) decisions by identifying which half is sorted at each step."
        ),
        "walkthrough": (
            "At each midpoint, detect whether left or right half is sorted. Check whether the target lies "
            "within the sorted half's bounds; if yes, keep that half, otherwise search the other half. "
            "Repeat until found or bounds cross."
        ),
        "complexity": (
            "Each step halves the search interval, so time is O(log n). Space is O(1) with iterative bounds."
        ),
        "reasoning_summary": (
            "Recover binary-search pruning by exploiting the sorted half that still exists after rotation."
        ),
    },
    {
        "id": "number_of_islands_dfs",
        "question": (
            "Using {focus} as context in {source_scope}, given a 2D grid of `'1'` (land) and `'0'` (water), "
            "return the number of islands. Explain traversal strategy and how you avoid double-counting."
        ),
        "problem": (
            "Connected land cells form components. You need to count components once each, while ensuring "
            "visited land is not counted again from adjacent cells."
        ),
        "walkthrough": (
            "Scan each cell; when unvisited land is found, increment island count and run DFS/BFS to mark "
            "all connected land cells visited. Continue scanning until all cells are processed."
        ),
        "complexity": (
            "Every cell is visited at most once, so time is O(rows * cols). Visited tracking and recursion/"
            "queue storage are O(rows * cols) in the worst case."
        ),
        "reasoning_summary": (
            "Model the grid as connected components and mark each component exactly once."
        ),
    },
)
_PROBLEM_SOLVING_SCENARIOS_BY_ID = {
    item["id"]: item for item in _PROBLEM_SOLVING_FALLBACK_SCENARIOS
}
_PROBLEM_SOLVING_PYTHON_CODE_TEMPLATES: dict[str, str] = {
    "two_sum_hash_map": """def two_sum(nums, target):
    # Store each seen value with its index so complement lookups stay O(1).
    seen = {}

    # Scan once and check complement before storing the current value.
    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            # Return indices as soon as a valid pair is found.
            return [seen[complement], index]
        seen[value] = index

    # Return an empty result when no pair exists.
    return []
""",
    "longest_substring_sliding_window": """def length_of_longest_substring(s):
    # Track the latest index of each character for fast window fixes.
    last_seen = {}
    left = 0
    best = 0

    # Expand the window with `right` and shrink from `left` on duplicates.
    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        best = max(best, right - left + 1)

    # The best window length is the answer.
    return best
""",
    "merge_intervals_sorting": """def merge_intervals(intervals):
    # Sort by interval start so overlap checks become local.
    intervals.sort(key=lambda pair: pair[0])
    merged = []

    # Merge into the latest interval when ranges overlap.
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        merged[-1][1] = max(merged[-1][1], end)

    # Return merged disjoint intervals.
    return merged
""",
    "top_k_frequent_heap": """from collections import Counter
import heapq


def top_k_frequent(nums, k):
    # Count frequency of each value in O(n).
    freq = Counter(nums)

    # Use a max-heap via negative counts for top-k extraction.
    heap = [(-count, value) for value, count in freq.items()]
    heapq.heapify(heap)

    result = []
    for _ in range(min(k, len(heap))):
        _, value = heapq.heappop(heap)
        result.append(value)
    return result
""",
    "search_rotated_binary_search": """def search_rotated(nums, target):
    # Keep classic binary-search boundaries.
    left, right = 0, len(nums) - 1

    # One half is always sorted even after rotation.
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    # Return -1 when target is absent.
    return -1
""",
    "number_of_islands_dfs": """def num_islands(grid):
    # Guard empty input to avoid index errors.
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    visited = set()

    def dfs(r, c):
        # Stop at water, bounds, or previously visited cells.
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if grid[r][c] != "1" or (r, c) in visited:
            return
        visited.add((r, c))

        # Explore all four directions for the current island.
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            dfs(r + dr, c + dc)

    islands = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visited:
                islands += 1
                dfs(r, c)
    return islands
""",
}

_PROBLEM_SOLVING_JAVA_CODE_TEMPLATES: dict[str, str] = {
    "two_sum_hash_map": """import java.util.HashMap;
import java.util.Map;

class Solution {
    public int[] twoSum(int[] nums, int target) {
        // Store each seen value with its index so complement lookups stay O(1).
        Map<Integer, Integer> seen = new HashMap<>();

        // Scan the array once and look for the complement before storing the current value.
        for (int index = 0; index < nums.length; index++) {
            int value = nums[index];
            int complement = target - value;
            if (seen.containsKey(complement)) {
                // Return the earlier index and the current index as soon as the pair is found.
                return new int[] {seen.get(complement), index};
            }
            seen.put(value, index);
        }

        // Return an empty result when the input does not contain a valid pair.
        return new int[0];
    }
}
""",
    "longest_substring_sliding_window": """import java.util.HashMap;
import java.util.Map;

class Solution {
    public int lengthOfLongestSubstring(String s) {
        // Track the latest index of each character for fast window fixes.
        Map<Character, Integer> lastSeen = new HashMap<>();
        int left = 0;
        int best = 0;

        // Expand the window with `right` and shrink from `left` on duplicates.
        for (int right = 0; right < s.length(); right++) {
            char current = s.charAt(right);
            Integer previous = lastSeen.get(current);
            if (previous != null && previous >= left) {
                left = previous + 1;
            }
            lastSeen.put(current, right);
            best = Math.max(best, right - left + 1);
        }

        // The best window length is the answer.
        return best;
    }
}
""",
    "merge_intervals_sorting": """import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

class Solution {
    public int[][] merge(int[][] intervals) {
        // Sort by start so overlap checks become local.
        Arrays.sort(intervals, Comparator.comparingInt(pair -> pair[0]));
        List<int[]> merged = new ArrayList<>();

        // Merge into the latest interval when ranges overlap.
        for (int[] interval : intervals) {
            if (merged.isEmpty() || interval[0] > merged.get(merged.size() - 1)[1]) {
                merged.add(new int[] {interval[0], interval[1]});
                continue;
            }
            int[] last = merged.get(merged.size() - 1);
            last[1] = Math.max(last[1], interval[1]);
        }

        // Return merged disjoint intervals.
        return merged.toArray(new int[0][]);
    }
}
""",
    "top_k_frequent_heap": """import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        // Count frequency of each value in O(n).
        Map<Integer, Integer> counts = new HashMap<>();
        for (int value : nums) {
            counts.merge(value, 1, Integer::sum);
        }

        // Use a max-heap ordered by frequency for top-k extraction.
        PriorityQueue<Map.Entry<Integer, Integer>> heap =
                new PriorityQueue<>(Comparator.comparingInt(Map.Entry::getValue));
        heap.addAll(counts.entrySet());

        List<Integer> result = new ArrayList<>();
        while (result.size() < k && !heap.isEmpty()) {
            result.add(heap.poll().getKey());
        }

        // Return the most frequent values in descending order.
        int[] top = new int[result.size()];
        for (int i = 0; i < top.length; i++) {
            top[i] = result.get(i);
        }
        return top;
    }
}
""",
    "search_rotated_binary_search": """class Solution {
    public int search(int[] nums, int target) {
        // Keep classic binary-search boundaries.
        int left = 0;
        int right = nums.length - 1;

        // One half is always sorted even after rotation.
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else if (nums[mid] < target && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        // Return -1 when the target is absent.
        return -1;
    }
}
""",
    "number_of_islands_dfs": """class Solution {
    public int numIslands(char[][] grid) {
        // Guard empty input to avoid index errors.
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }

        int rows = grid.length;
        int cols = grid[0].length;
        int islands = 0;

        // Count each connected component once, marking its cells as water.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] == '1') {
                    islands++;
                    sink(grid, r, c, rows, cols);
                }
            }
        }
        return islands;
    }

    private void sink(char[][] grid, int r, int c, int rows, int cols) {
        // Stop at water or bounds so each cell is visited once.
        if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != '1') {
            return;
        }
        grid[r][c] = '0';

        // Explore all four directions for the current island.
        sink(grid, r + 1, c, rows, cols);
        sink(grid, r - 1, c, rows, cols);
        sink(grid, r, c + 1, rows, cols);
        sink(grid, r, c - 1, rows, cols);
    }
}
""",
}

_PROBLEM_SOLVING_CPP_CODE_TEMPLATES: dict[str, str] = {
    "two_sum_hash_map": """#include <unordered_map>
#include <vector>

using namespace std;

vector<int> twoSum(const vector<int>& nums, int target) {
    // Store each seen value with its index so complement lookups stay O(1).
    unordered_map<int, int> seen;

    // Scan the array once and look for the complement before storing the current value.
    for (int index = 0; index < static_cast<int>(nums.size()); index++) {
        int value = nums[index];
        int complement = target - value;
        auto found = seen.find(complement);
        if (found != seen.end()) {
            // Return the earlier index and the current index as soon as the pair is found.
            return {found->second, index};
        }
        seen[value] = index;
    }

    // Return an empty result when the input does not contain a valid pair.
    return {};
}
""",
    "longest_substring_sliding_window": """#include <algorithm>
#include <string>
#include <unordered_map>

using namespace std;

int lengthOfLongestSubstring(const string& s) {
    // Track the latest index of each character for fast window fixes.
    unordered_map<char, int> lastSeen;
    int left = 0;
    int best = 0;

    // Expand the window with `right` and shrink from `left` on duplicates.
    for (int right = 0; right < static_cast<int>(s.size()); right++) {
        auto found = lastSeen.find(s[right]);
        if (found != lastSeen.end() && found->second >= left) {
            left = found->second + 1;
        }
        lastSeen[s[right]] = right;
        best = max(best, right - left + 1);
    }

    // The best window length is the answer.
    return best;
}
""",
    "merge_intervals_sorting": """#include <algorithm>
#include <vector>

using namespace std;

vector<vector<int>> mergeIntervals(vector<vector<int>>& intervals) {
    // Sort by start so overlap checks become local.
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> merged;

    // Merge into the latest interval when ranges overlap.
    for (const auto& interval : intervals) {
        if (merged.empty() || interval[0] > merged.back()[1]) {
            merged.push_back(interval);
            continue;
        }
        merged.back()[1] = max(merged.back()[1], interval[1]);
    }

    // Return merged disjoint intervals.
    return merged;
}
""",
    "top_k_frequent_heap": """#include <algorithm>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

vector<int> topKFrequent(const vector<int>& nums, int k) {
    // Count frequency of each value in O(n).
    unordered_map<int, int> counts;
    for (int value : nums) {
        counts[value]++;
    }

    // Use a max-heap ordered by frequency for top-k extraction.
    priority_queue<pair<int, int>> heap;
    for (const auto& entry : counts) {
        heap.push({entry.second, entry.first});
    }

    vector<int> result;
    while (!heap.empty() && static_cast<int>(result.size()) < k) {
        result.push_back(heap.top().second);
        heap.pop();
    }

    // Return the most frequent values in descending order.
    sort(result.begin(), result.end(), greater<int>());
    return result;
}
""",
    "search_rotated_binary_search": """#include <vector>

using namespace std;

int searchRotated(const vector<int>& nums, int target) {
    // Keep classic binary-search boundaries.
    int left = 0;
    int right = static_cast<int>(nums.size()) - 1;

    // One half is always sorted even after rotation.
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            return mid;
        }
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else if (nums[mid] < target && target <= nums[right]) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    // Return -1 when the target is absent.
    return -1;
}
""",
    "number_of_islands_dfs": """#include <vector>

using namespace std;

int numIslands(vector<vector<char>>& grid) {
    // Guard empty input to avoid index errors.
    if (grid.empty() || grid[0].empty()) {
        return 0;
    }

    int rows = static_cast<int>(grid.size());
    int cols = static_cast<int>(grid[0].size());
    int islands = 0;

    // Count each connected component once, sinking its cells into water.
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == '1') {
                islands++;
                sink(grid, r, c, rows, cols);
            }
        }
    }
    return islands;
}

void sink(vector<vector<char>>& grid, int r, int c, int rows, int cols) {
    // Stop at water or bounds so each cell is visited once.
    if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != '1') {
        return;
    }
    grid[r][c] = '0';

    // Explore all four directions for the current island.
    sink(grid, r + 1, c, rows, cols);
    sink(grid, r - 1, c, rows, cols);
    sink(grid, r, c + 1, rows, cols);
    sink(grid, r, c - 1, rows, cols);
}
""",
}

_PROBLEM_SOLVING_JAVASCRIPT_CODE_TEMPLATES: dict[str, str] = {
    "two_sum_hash_map": """function twoSum(nums, target) {
  // Store each seen value with its index so complement lookups stay O(1).
  const seen = new Map();

  // Scan the array once and look for the complement before storing the current value.
  for (let index = 0; index < nums.length; index += 1) {
    const value = nums[index];
    const complement = target - value;
    if (seen.has(complement)) {
      // Return the earlier index and the current index as soon as the pair is found.
      return [seen.get(complement), index];
    }
    seen.set(value, index);
  }

  // Return an empty result when the input does not contain a valid pair.
  return [];
}
""",
    "longest_substring_sliding_window": """function lengthOfLongestSubstring(s) {
  // Track the latest index of each character for fast window fixes.
  const lastSeen = new Map();
  let left = 0;
  let best = 0;

  // Expand the window with `right` and shrink from `left` on duplicates.
  for (let right = 0; right < s.length; right += 1) {
    const current = s[right];
    const previous = lastSeen.get(current);
    if (previous !== undefined && previous >= left) {
      left = previous + 1;
    }
    lastSeen.set(current, right);
    best = Math.max(best, right - left + 1);
  }

  // The best window length is the answer.
  return best;
}
""",
    "merge_intervals_sorting": """function mergeIntervals(intervals) {
  // Sort by start so overlap checks become local.
  const sorted = intervals.slice().sort((a, b) => a[0] - b[0]);
  const merged = [];

  // Merge into the latest interval when ranges overlap.
  for (const [start, end] of sorted) {
    if (merged.length === 0 || start > merged[merged.length - 1][1]) {
      merged.push([start, end]);
      continue;
    }
    merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], end);
  }

  // Return merged disjoint intervals.
  return merged;
}
""",
    "top_k_frequent_heap": """function topKFrequent(nums, k) {
  // Count frequency of each value in O(n).
  const counts = new Map();
  for (const value of nums) {
    counts.set(value, (counts.get(value) || 0) + 1);
  }

  // Sort the (value, count) pairs by descending frequency and take the top k.
  const ranked = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);

  // Return the most frequent values in descending order.
  return ranked.slice(0, k).map(([value]) => value);
}
""",
    "search_rotated_binary_search": """function searchRotated(nums, target) {
  // Keep classic binary-search boundaries.
  let left = 0;
  let right = nums.length - 1;

  // One half is always sorted even after rotation.
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    if (nums[mid] === target) {
      return mid;
    }
    if (nums[left] <= nums[mid]) {
      if (nums[left] <= target && target < nums[mid]) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    } else if (nums[mid] < target && target <= nums[right]) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  // Return -1 when the target is absent.
  return -1;
}
""",
    "number_of_islands_dfs": """function numIslands(grid) {
  // Guard empty input to avoid index errors.
  if (!grid || grid.length === 0 || grid[0].length === 0) {
    return 0;
  }

  const rows = grid.length;
  const cols = grid[0].length;
  let islands = 0;

  // Count each connected component once, sinking its cells into water.
  const sink = (r, c) => {
    // Stop at water or bounds so each cell is visited once.
    if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] !== "1") {
      return;
    }
    grid[r][c] = "0";

    // Explore all four directions for the current island.
    sink(r + 1, c);
    sink(r - 1, c);
    sink(r, c + 1);
    sink(r, c - 1);
  };

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      if (grid[r][c] === "1") {
        islands += 1;
        sink(r, c);
      }
    }
  }
  return islands;
}
""",
}

_PROBLEM_SOLVING_CSHARP_CODE_TEMPLATES: dict[str, str] = {
    "two_sum_hash_map": """using System.Collections.Generic;

public class Solution
{
    public int[] TwoSum(int[] nums, int target)
    {
        // Store each seen value with its index so complement lookups stay O(1).
        var seen = new Dictionary<int, int>();

        // Scan the array once and look for the complement before storing the current value.
        for (var index = 0; index < nums.Length; index++)
        {
            var value = nums[index];
            var complement = target - value;
            if (seen.TryGetValue(complement, out var previousIndex))
            {
                // Return the earlier index and the current index as soon as the pair is found.
                return new[] { previousIndex, index };
            }
            seen[value] = index;
        }

        // Return an empty result when the input does not contain a valid pair.
        return new int[0];
    }
}
""",
    "longest_substring_sliding_window": """using System;
using System.Collections.Generic;

public class Solution
{
    public int LengthOfLongestSubstring(string s)
    {
        // Track the latest index of each character for fast window fixes.
        var lastSeen = new Dictionary<char, int>();
        var left = 0;
        var best = 0;

        // Expand the window with `right` and shrink from `left` on duplicates.
        for (var right = 0; right < s.Length; right++)
        {
            var current = s[right];
            if (lastSeen.TryGetValue(current, out var previous) && previous >= left)
            {
                left = previous + 1;
            }
            lastSeen[current] = right;
            best = Math.Max(best, right - left + 1);
        }

        // The best window length is the answer.
        return best;
    }
}
""",
    "merge_intervals_sorting": """using System.Collections.Generic;
using System.Linq;

public class Solution
{
    public int[][] Merge(int[][] intervals)
    {
        // Sort by start so overlap checks become local.
        var sorted = intervals.OrderBy(pair => pair[0]).ToList();
        var merged = new List<int[]>();

        // Merge into the latest interval when ranges overlap.
        foreach (var interval in sorted)
        {
            if (merged.Count == 0 || interval[0] > merged[merged.Count - 1][1])
            {
                merged.Add(new[] { interval[0], interval[1] });
                continue;
            }
            merged[merged.Count - 1][1] = System.Math.Max(merged[merged.Count - 1][1], interval[1]);
        }

        // Return merged disjoint intervals.
        return merged.ToArray();
    }
}
""",
    "top_k_frequent_heap": """using System.Collections.Generic;
using System.Linq;

public class Solution
{
    public int[] TopKFrequent(int[] nums, int k)
    {
        // Count frequency of each value in O(n).
        var counts = new Dictionary<int, int>();
        foreach (var value in nums)
        {
            counts[value] = counts.TryGetValue(value, out var seen) ? seen + 1 : 1;
        }

        // Order the (value, count) pairs by descending frequency and take the top k.
        return counts
            .OrderByDescending(entry => entry.Value)
            .Take(k)
            .Select(entry => entry.Key)
            .ToArray();
    }
}
""",
    "search_rotated_binary_search": """public class Solution
{
    public int Search(int[] nums, int target)
    {
        // Keep classic binary-search boundaries.
        var left = 0;
        var right = nums.Length - 1;

        // One half is always sorted even after rotation.
        while (left <= right)
        {
            var mid = left + (right - left) / 2;
            if (nums[mid] == target)
            {
                return mid;
            }
            if (nums[left] <= nums[mid])
            {
                if (nums[left] <= target && target < nums[mid])
                {
                    right = mid - 1;
                }
                else
                {
                    left = mid + 1;
                }
            }
            else if (nums[mid] < target && target <= nums[right])
            {
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }

        // Return -1 when the target is absent.
        return -1;
    }
}
""",
    "number_of_islands_dfs": """public class Solution
{
    public int NumIslands(char[][] grid)
    {
        // Guard empty input to avoid index errors.
        if (grid == null || grid.Length == 0 || grid[0].Length == 0)
        {
            return 0;
        }

        var rows = grid.Length;
        var cols = grid[0].Length;
        var islands = 0;

        // Count each connected component once, sinking its cells into water.
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                if (grid[r][c] == '1')
                {
                    islands++;
                    Sink(grid, r, c, rows, cols);
                }
            }
        }
        return islands;
    }

    private static void Sink(char[][] grid, int r, int c, int rows, int cols)
    {
        // Stop at water or bounds so each cell is visited once.
        if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != '1')
        {
            return;
        }
        grid[r][c] = '0';

        // Explore all four directions for the current island.
        Sink(grid, r + 1, c, rows, cols);
        Sink(grid, r - 1, c, rows, cols);
        Sink(grid, r, c + 1, rows, cols);
        Sink(grid, r, c - 1, rows, cols);
    }
}
""",
}

_PROBLEM_SOLVING_GO_CODE_TEMPLATES: dict[str, str] = {
    "two_sum_hash_map": """package main

func twoSum(nums []int, target int) []int {
    // Store each seen value with its index so complement lookups stay O(1).
    seen := make(map[int]int)

    // Scan the array once and look for the complement before storing the current value.
    for index, value := range nums {
        complement := target - value
        if previousIndex, ok := seen[complement]; ok {
            // Return the earlier index and the current index as soon as the pair is found.
            return []int{previousIndex, index}
        }
        seen[value] = index
    }

    // Return an empty result when the input does not contain a valid pair.
    return []int{}
}
""",
    "longest_substring_sliding_window": """package main

func lengthOfLongestSubstring(s string) int {
    // Track the latest index of each character for fast window fixes.
    lastSeen := make(map[rune]int)
    left := 0
    best := 0

    // Expand the window with `right` and shrink from `left` on duplicates.
    for right, current := range []rune(s) {
        if previous, ok := lastSeen[current]; ok && previous >= left {
            left = previous + 1
        }
        lastSeen[current] = right
        if right-left+1 > best {
            best = right - left + 1
        }
    }

    // The best window length is the answer.
    return best
}
""",
    "merge_intervals_sorting": """package main

import "sort"

func mergeIntervals(intervals [][2]int) [][2]int {
    // Sort by start so overlap checks become local.
    sort.Slice(intervals, func(a, b int) bool { return intervals[a][0] < intervals[b][0] })
    merged := make([][2]int, 0, len(intervals))

    // Merge into the latest interval when ranges overlap.
    for _, interval := range intervals {
        last := len(merged) - 1
        if last < 0 || interval[0] > merged[last][1] {
            merged = append(merged, interval)
            continue
        }
        if interval[1] > merged[last][1] {
            merged[last][1] = interval[1]
        }
    }

    // Return merged disjoint intervals.
    return merged
}
""",
    "top_k_frequent_heap": """package main

import (
    "container/heap"
)

type countEntry struct {
    value int
    count int
}

type countHeap []countEntry

func (h countHeap) Len() int           { return len(h) }
func (h countHeap) Less(a, b int) bool { return h[a].count > h[b].count }
func (h countHeap) Swap(a, b int)      { h[a], h[b] = h[b], h[a] }
func (h *countHeap) Push(x any)        { *h = append(*h, x.(countEntry)) }
func (h *countHeap) Pop() any {
    // Remove the lowest-priority entry so the root stays the most frequent value.
    old := *h
    last := old[len(old)-1]
    *h = old[:len(old)-1]
    return last
}

func topKFrequent(nums []int, k int) []int {
    // Count frequency of each value in O(n).
    counts := make(map[int]int)
    for _, value := range nums {
        counts[value]++
    }

    // Use a max-heap ordered by frequency for top-k extraction.
    h := &countHeap{}
    for value, count := range counts {
        *h = append(*h, countEntry{value: value, count: count})
    }
    heap.Init(h)

    result := make([]int, 0, k)
    for len(result) < k && h.Len() > 0 {
        result = append(result, heap.Pop(h).(countEntry).value)
    }

    // Return the most frequent values.
    return result
}
""",
    "search_rotated_binary_search": """package main

func searchRotated(nums []int, target int) int {
    // Keep classic binary-search boundaries.
    left, right := 0, len(nums)-1

    // One half is always sorted even after rotation.
    for left <= right {
        mid := left + (right-left)/2
        if nums[mid] == target {
            return mid
        }
        if nums[left] <= nums[mid] {
            if nums[left] <= target && target < nums[mid] {
                right = mid - 1
            } else {
                left = mid + 1
            }
        } else if nums[mid] < target && target <= nums[right] {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }

    // Return -1 when the target is absent.
    return -1
}
""",
    "number_of_islands_dfs": """package main

func numIslands(grid [][]byte) int {
    // Guard empty input to avoid index errors.
    if len(grid) == 0 || len(grid[0]) == 0 {
        return 0
    }

    rows, cols := len(grid), len(grid[0])
    var sink func(r, c int)
    islands := 0

    // Count each connected component once, sinking its cells into water.
    sink = func(r, c int) {
        // Stop at water or bounds so each cell is visited once.
        if r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != '1' {
            return
        }
        grid[r][c] = '0'

        // Explore all four directions for the current island.
        sink(r+1, c)
        sink(r-1, c)
        sink(r, c+1)
        sink(r, c-1)
    }

    for r := 0; r < rows; r++ {
        for c := 0; c < cols; c++ {
            if grid[r][c] == '1' {
                islands++
                sink(r, c)
            }
        }
    }
    return islands
}
""",
}

#: Genuine, hand-written source per language and scenario.
#:
#: A language is deliberately *absent* from a scenario rather than inheriting
#: another language's source. Relabelling the Python two-sum implementation
#: under a ```java fence is a correctness defect, so "no entry
#: here" means "no code block at all", not "reuse something close enough".
_PROBLEM_SOLVING_CODE_TEMPLATES_BY_LANGUAGE: dict[str, dict[str, str]] = {
    "python": _PROBLEM_SOLVING_PYTHON_CODE_TEMPLATES,
    "java": _PROBLEM_SOLVING_JAVA_CODE_TEMPLATES,
    "cpp": _PROBLEM_SOLVING_CPP_CODE_TEMPLATES,
    "javascript": _PROBLEM_SOLVING_JAVASCRIPT_CODE_TEMPLATES,
    "csharp": _PROBLEM_SOLVING_CSHARP_CODE_TEMPLATES,
    "go": _PROBLEM_SOLVING_GO_CODE_TEMPLATES,
}


def _problem_solving_scenario_ids_for_language(language: str) -> list[str]:
    """Scenario ids that have a genuine code template in ``language``.

    Empty for an unsupported language, which lets the caller fall back to a
    language-agnostic question instead of borrowing another language's source.
    """
    templates = _PROBLEM_SOLVING_CODE_TEMPLATES_BY_LANGUAGE.get(
        str(language or "").strip().lower()
    )
    if not templates:
        return []
    return list(templates)


def _problem_solving_question_seed(
    focus: str,
    source_scope: str,
    *,
    scenario_id: str = _PROBLEM_SOLVING_DEFAULT_SCENARIO_ID,
) -> dict[str, str]:
    scenario = _PROBLEM_SOLVING_SCENARIOS_BY_ID.get(scenario_id) or _PROBLEM_SOLVING_SCENARIOS_BY_ID[
        _PROBLEM_SOLVING_DEFAULT_SCENARIO_ID
    ]
    focus_hint = (focus or source_scope or "the current checkpoint").strip()
    scope_hint = (source_scope or "this topic").strip() or "this topic"
    return {
        "question": scenario["question"].format(focus=focus_hint, source_scope=scope_hint),
        "problem": scenario["problem"].format(focus=focus_hint, source_scope=scope_hint),
        "walkthrough": scenario["walkthrough"].format(focus=focus_hint, source_scope=scope_hint),
        "complexity": scenario["complexity"].format(focus=focus_hint, source_scope=scope_hint),
        "reasoning_summary": scenario["reasoning_summary"].format(focus=focus_hint, source_scope=scope_hint),
    }


def _split_first_sentence(text: str) -> tuple[str, str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "", ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    first = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    return first, rest


def _split_last_sentence(text: str) -> tuple[str, str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "", ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(parts) == 1:
        return cleaned, cleaned
    return " ".join(parts[:-1]).strip(), parts[-1].strip()


def _fallback_learning_objective(*, focus: str, source_scope: str, problem_solving_mode: bool) -> str:
    if problem_solving_mode:
        return (
            f"After this question, the learner should be able to explain the invariant behind {focus} "
            f"and turn it into a working solution in {source_scope}."
        )
    return (
        f"After this question, the learner should be able to explain {focus} in {source_scope} "
        "and justify the practical tradeoff behind the answer."
    )


def _build_conceptual_fallback_answer(
    *,
    focus: str,
    source_scope: str,
    angle: str,
    difficulty: str,
    code_language: str = "",
    requires_programming: bool = False,
) -> tuple[str, str]:
    reasoning_summary = (
        f"Name the main constraint around {focus}, then justify the choice that best protects {angle}."
    )
    impact_table = "\n".join(
        [
            "| Concern | What matters for this topic |",
            "| --- | --- |",
            f"| Core role | {focus} should solve a concrete problem inside {source_scope}, not exist as an abstract pattern. |",
            f"| Main tradeoff | The design choice should protect {angle} first, while making the downside explicit. |",
            "| Failure risk | Edge cases, scaling pressure, and operational failure paths change whether the same choice still makes sense. |",
        ]
    )
    hard_follow_up = ""
    if difficulty == "hard":
        hard_follow_up = (
            "\n\n- At harder interview levels, also call out the bottleneck you expect first under scale."
            "\n- If consistency, retries, caching, or coordination are involved, mention the failure mode that changes the design."
        )
    code_example = ""
    if requires_programming and code_language:
        code_example = (
            f"\n\n```{code_language}\n"
            "# Example: keep the guardrail that protects the main requirement explicit.\n"
            "def apply_constraint(state, limit):\n"
            "    if state > limit:\n"
            "        return limit\n"
            "    return state\n"
            "```"
        )
    answer = "\n\n".join(
        [
            (
                f"**Answer:** {focus} in {source_scope} matters because it controls whether the system stays correct, maintainable, and predictable under real constraints. A strong answer explains what problem it solves, why that solution is chosen, and what tradeoff or failure mode limits it."
            ),
            (
                f"**Detailed explanation:** In practical terms, {focus} should be explained through the real engineering pressure around it: the requirement it satisfies, the tradeoff it introduces, and the failure path that would force a different design. That makes the answer useful for both study and interview follow-ups.\n\n"
                f"{impact_table}"
                f"{hard_follow_up}"
                f"{code_example}"
            ),
            (
                f"**Common mistake:** A weak answer for {focus} jumps straight to a favorite pattern and never "
                "shows why it fits the real constraints or what could go wrong."
            ),
        ]
    ).strip()
    return answer, reasoning_summary


def _problem_solving_code_template(
    language: str,
    *,
    scenario_id: str = _PROBLEM_SOLVING_DEFAULT_SCENARIO_ID,
) -> str:
    """Return genuine source for ``language``/``scenario_id``, or an empty string.

    Never falls back to another language's implementation: relabelling Python
    source under a ```java fence produces an answer that looks right and is
    wrong. An empty result means the caller must omit the code block, which in
    turn fails ``_validate_problem_solving_answer`` rather than shipping a
    mismatch.
    """
    templates = _PROBLEM_SOLVING_CODE_TEMPLATES_BY_LANGUAGE.get(
        str(language or "").strip().lower()
    )
    if not templates:
        return ""
    code = templates.get(scenario_id, "")
    return code.strip()


def _build_problem_solving_fallback_answer(
    *,
    focus: str,
    source_scope: str,
    language: str,
    scenario_id: str = _PROBLEM_SOLVING_DEFAULT_SCENARIO_ID,
) -> tuple[str, str]:
    seed = _problem_solving_question_seed(
        focus,
        source_scope,
        scenario_id=scenario_id,
    )
    code = _problem_solving_code_template(
        language,
        scenario_id=scenario_id,
    )
    short_answer, _remaining_walkthrough = _split_first_sentence(seed["walkthrough"])
    time_and_space, tradeoff = _split_last_sentence(seed["complexity"])
    approach_table = "\n".join(
        [
            "| Approach | Why interviewers move past it | Why the chosen approach wins |",
            "| --- | --- | --- |",
            "| Brute force | Repeats work and usually misses the core invariant | Useful only as a baseline for correctness reasoning |",
            "| Chosen approach | Preserves the exact state the next step needs | Reduces repeated work while keeping the explanation easy to defend |",
        ]
    )
    blocks = [
        "### Problem",
        (
            f"**What the problem is asking:** {seed['problem']}"
        ),
        "### Solution Walkthrough",
        f"**Direct answer:** {short_answer}",
        (
            f"**Detailed explanation:** {seed['walkthrough']}\n\n"
            f"{approach_table}"
        ),
        (
            f"**Why this works:** {seed['walkthrough']}\n"
            "- The invariant tells you what must stay true after every update.\n"
            "- The chosen structure stores exactly the information the next step needs.\n"
            "- The update order prevents invalid reuse or stale state."
        ),
        (
            f"**Common mistake:** A weak answer for {focus} starts coding the brute-force idea before "
            "stating the invariant and why the chosen structure fits the constraint."
        ),
        "### Complexity",
        f"**Time and space:** {time_and_space}",
        f"**Tradeoff / scaling caveat:** {tradeoff}",
    ]
    if code:
        # Only ever emit a code block whose source genuinely is ``language``.
        # With no template the heading is omitted, so validation rejects the
        # answer instead of a mismatched fence reaching the learner.
        blocks.extend(
            [
                "### Code",
                f"```{language}\n{code}\n```",
                (
                    "**Invariant note:** The state update inside the main scan is what preserves the core invariant, "
                    "so each step keeps the partial solution valid before the next iteration."
                ),
            ]
        )
    answer = "\n\n".join(blocks).strip()
    return answer, seed["reasoning_summary"]


def _validate_problem_solving_answer(answer: str, preferred_language: str) -> tuple[bool, str]:
    language = _problem_solving_language(preferred_language)
    answer_text = (answer or "").strip()
    headings = [
        match.group(1).strip()
        for match in _PROBLEM_SOLVING_H3_RE.finditer(
            _PROBLEM_SOLVING_CODE_FENCE_RE.sub("", answer_text)
        )
    ]
    if headings != list(_PROBLEM_SOLVING_SECTION_HEADINGS):
        return False, "invalid_problem_solving_heading_sequence"

    code_matches = list(_PROBLEM_SOLVING_CODE_FENCE_RE.finditer(answer_text))
    if len(code_matches) != 1:
        return False, "invalid_problem_solving_code_block_count"

    code_match = code_matches[0]
    if code_match.start() <= answer_text.find("### Code"):
        return False, "problem_solving_code_block_before_code_heading"

    fence_language = code_match.group("language").strip().lower()
    if fence_language != language:
        return False, "problem_solving_code_language_mismatch"

    code = code_match.group("code").strip()
    if not code:
        return False, "empty_problem_solving_code_block"

    comment_count = 0
    for raw_line in code.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if language == "python":
            if stripped.startswith("#"):
                comment_count += 1
        elif (
            stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith("*")
            or stripped.endswith("*/")
        ):
            comment_count += 1
    if comment_count < 2:
        return False, "problem_solving_code_missing_comments"

    return True, ""


def _problem_solving_pattern_signature(question: str, answer: str) -> str:
    text = f"{question}\n{answer}".strip().lower()
    if not text:
        return ""
    if (
        "two sum" in text
        or "two_sum" in text
        or "complement" in text
        or "def two_sum" in text
        or "twosum(" in text
        or ("target" in text and "indices" in text and "hash map" in text)
    ):
        return "two_sum_hash_map"
    if (
        "substring without repeating" in text
        or ("substring" in text and "repeating" in text and "window" in text)
        or "def length_of_longest_substring" in text
    ):
        return "longest_substring_sliding_window"
    if (
        "merge intervals" in text
        or ("interval" in text and "overlap" in text and "sort" in text)
        or "def merge_intervals" in text
    ):
        return "merge_intervals_sorting"
    if (
        "top k frequent" in text
        or "most frequent elements" in text
        or ("heap" in text and "frequency" in text)
        or "def top_k_frequent" in text
    ):
        return "top_k_frequent_heap"
    if (
        "rotated sorted array" in text
        or ("rotated" in text and "binary search" in text)
        or "def search_rotated" in text
    ):
        return "search_rotated_binary_search"
    if (
        "number of islands" in text
        or ("grid" in text and "island" in text and ("dfs" in text or "bfs" in text))
        or "def num_islands" in text
    ):
        return "number_of_islands_dfs"
    return ""


def _problem_solving_unique_pattern_target(target_count: int) -> int:
    return min(max(0, int(target_count or 0)), len(_PROBLEM_SOLVING_SCENARIOS_BY_ID))


def _accept_problem_pattern_signature(
    *,
    signature: str,
    seen_signatures: set[str],
    unique_target: int,
) -> bool:
    if not signature:
        return True
    if signature in seen_signatures and len(seen_signatures) < unique_target:
        return False
    seen_signatures.add(signature)
    return True


def _problem_solving_fallback_count_with_headroom(missing_count: int, problem_solving_mode: bool) -> int:
    missing = max(0, int(missing_count or 0))
    if missing <= 0:
        return 0
    if not problem_solving_mode:
        return missing
    scenario_count = max(1, len(_PROBLEM_SOLVING_SCENARIOS_BY_ID))
    return min(24, max(missing, missing * 3, missing + scenario_count))


def _question_system_prompt(problem_solving_mode: bool) -> str:
    """The ``system`` role for question generation.

    The persona used to be the first line of the user turn, which gave
    externally sourced course text the same authority as the instruction.
    The contract lives in the prompt registry; this selects the
    registered template for the requested mode.
    """
    name = (
        "question_generation_problem_solving"
        if problem_solving_mode
        else "question_generation"
    )
    return default_registry.get(name).system_prompt


def _problem_solving_few_shot_example(language: str) -> str:
    """Worked example showing the exact answer shape the validator enforces."""
    code = _problem_solving_code_template(
        language,
        scenario_id=_PROBLEM_SOLVING_DEFAULT_SCENARIO_ID,
    )
    fence = language if code else "python"
    sample = code or _PROBLEM_SOLVING_PYTHON_CODE_TEMPLATES[
        _PROBLEM_SOLVING_DEFAULT_SCENARIO_ID
    ]
    return (
        "Worked example showing the required shape (its content is illustrative, "
        "not reusable content):\n"
        "[\n"
        '  {\n'
        '    "question": "Given an array of integers and a target, return the two indices whose values sum to the target, or an empty array when no pair exists.",\n'
        '    "answer": "### Problem\\n**What the problem is asking:** Return two indices i and j where nums[i] + nums[j] equals target.\\n\\n'
        '### Solution Walkthrough\\n**Direct answer:** Use a hash map from value to index and check each value\\u0027s complement before storing it.\\n\\n'
        "**Detailed explanation:** A one-pass scan keeps every earlier value reachable in O(1), which removes the nested loop entirely.\\n\\n"
        "**Why this works:** Checking the complement before inserting the current value guarantees the same element is never reused.\\n\\n"
        "**Common mistake:** Inserting the current value first, which lets one element satisfy both slots.\\n\\n"
        "### Complexity\\n**Time and space:** Time is O(n) for the single scan; extra space is O(n) for the map.\\n\\n"
        "**Tradeoff / scaling caveat:** The map trades memory for linear time; under a hard memory cap, sort-then-two-pointers is the fallback.\\n\\n"
        f"### Code\\n```{fence}\\n{sample}\\n```\\n\\n"
        '**Invariant note:** Storing the value only after the complement check is what keeps the partial solution valid.\\",\\n'
        '    "difficulty": "medium",\n'
        '    "learning_objective": "After this question, the learner should be able to choose a one-pass hash-map strategy and justify it.",\n'
        '    "source_section": "Hash maps",\n'
        '    "source_quote": "A hash map keeps every earlier value reachable in constant time.",\n'
        '    "reasoning_summary": "Identify the complement invariant first, then pick the structure that preserves it.",\n'
        '    "target_level": "mid"\n'
        "  }\n"
        "]"
    )


def _conceptual_few_shot_example() -> str:
    """Worked example showing the exact conceptual answer shape."""
    return (
        "Worked example showing the required shape (its content is illustrative, "
        "not reusable content):\n"
        "[\n"
        '  {\n'
        '    "question": "Why does an idempotency key make a retried payment request safe?",\n'
        '    "answer": "**Answer:** An idempotency key makes a retried payment safe because the server stores the first response for that key and replays it instead of executing the charge twice.\\n\\n'
        "**Detailed explanation:**\\n\\n"
        "| Concern | What matters for this topic |\\n"
        "| --- | --- |\\n"
        "| Core role | The key identifies one logical request, not one HTTP call. |\\n"
        "| Main tradeoff | Storing the first response costs memory and needs an expiry policy. |\\n"
        "| Failure risk | A key with no expiry grows the store; a key reused for a different body breaks the guarantee. |\\n\\n"
        '**Common mistake:** Deduplicating in the client only, which cannot protect against a network-level retry after the response was lost.\\n",\n'
        '    "difficulty": "medium",\n'
        '    "learning_objective": "After this question, the learner should be able to explain where the idempotency guarantee is enforced and why.",\n'
        '    "source_section": "Reliable retries",\n'
        '    "source_quote": "The server stores the first response for a key and replays it on retry.",\n'
        '    "reasoning_summary": "Name the retry hazard first, then the store that removes it.",\n'
        '    "target_level": "mid"\n'
        "  }\n"
        "]"
    )


def _build_prompt(
    topic_id: str,
    topic_title: str,
    doc_content: str,
    count: int = 5,
    difficulty: Optional[str] = None,
    level: Optional[str] = None,
    section_title: Optional[str] = None,
    section_content: Optional[str] = None,
    response_detail: str = "very_detailed",
    preferred_language: str = "",
    requires_programming: bool = False,
    requested_total_count: Optional[int] = None,
    existing_questions: Optional[list[str]] = None,
    additional_existing_questions: Optional[list[str]] = None,
    prior_progress: str = "",
    mcp_context: str = "",
    track: str = "",
    learner_state: str = "",
) -> str:
    problem_solving_mode = is_problem_solving_topic(topic_id)
    target_level = _normalise_level(level)
    session_type = "coding" if problem_solving_mode else "general"
    skill_section = _skill_context_section(
        target_level,
        session_type,
        _skill_track_for_topic(topic_id, track),
    )
    requested_total = max(1, int(requested_total_count or count or 1))
    existing_seed = _merge_existing_questions(existing_questions, additional_existing_questions)
    diff_clause = ""
    if difficulty:
        diff_clause = f' All questions should be "{difficulty}" difficulty.'

    if section_title and section_content:
        scope = f'the section titled "{section_title}"'
        content = _clamp_content(section_content)
    else:
        scope = "the current topic"
        content = _clamp_content(doc_content)
    grounding_anchors = _grounding_anchor_phrases(
        topic_title=topic_title,
        section_title=section_title,
        content=section_content if (section_title and section_content) else doc_content,
    )
    grounding_block = _render_grounding_anchors_block(grounding_anchors)
    common_interview_rules = _common_interview_question_rules(
        problem_solving_mode,
        topic_scope=scope,
    )

    detail_clause = (
        "Be detailed enough to help the learner pass the interview: explain the direct answer, reasoning, tradeoffs, edge cases, practical examples, and the follow-up points an interviewer is likely to probe. Do not be terse or vague."
        if response_detail != "very_detailed"
        else "Be maximally detailed with layered explanation depth, concrete examples, edge cases, tradeoffs, and enough substance that the learner could study from the answer alone."
    )
    selected_language = (preferred_language or "").strip().lower()
    study_code_language = _study_code_language(
        selected_language,
        problem_solving_mode=problem_solving_mode,
        requires_programming=requires_programming,
    )
    problem_solving_language = _problem_solving_language(study_code_language)
    code_clause = ""
    if problem_solving_mode:
        code_clause = (
            f'Under `### Code`, include exactly one fenced "{problem_solving_language}" example with comments '
            "that explain each key step or block."
        )
    elif requires_programming and study_code_language:
        code_clause = (
            f'Include one practical fenced code example in "{study_code_language}" when code clarifies the answer.'
        )
    elif not requires_programming:
        code_clause = "Avoid code blocks unless code is explicitly required by the question."
    mcp_block = untrusted_block("external_context", mcp_context, 2500)

    problem_scope_rules = (
        [
            "Questions must be true problem-solving prompts (algorithmic/coding style), not generic theory prompts.",
            "Use a template-driven question structure: concrete input/output, explicit constraints, and interview-style reasoning expectations.",
            "Do not copy canonical LeetCode wording; adapt each question to the current topic scope and source content.",
            "Across generated items, vary the kinds of constraints, data-structure choices, and solution shapes.",
            "Every answer must include commented code in the selected language.",
            *common_interview_rules,
            *_question_tutoring_arc_rules(True, problem_solving_language),
        ]
        if problem_solving_mode
        else [
            *common_interview_rules,
            "Questions must cover conceptual + practical angles.",
            *_question_tutoring_arc_rules(False),
        ]
    )
    answer_shape = (
        "Markdown with exactly ### Problem, ### Solution Walkthrough, ### Complexity, and ### Code; include the required bold labels inside those sections and exactly one fenced code block"
        if problem_solving_mode
        else "Markdown with bold sections **Answer:**, **Detailed explanation:**, and **Common mistake:**"
    )
    uniqueness_block = ""
    if existing_seed:
        existing_blob = "\n".join([f"- {q}" for q in existing_seed[:60]])
        uniqueness_block = (
            "\nAlready generated questions for this checkpoint. Do NOT repeat or rephrase these:\n"
            f"{existing_blob}\n"
        )
    progress_block = _progress_block(prior_progress)
    learner_state_block = _learner_state_block(learner_state)
    untrusted_blocks = "\n\n".join(
        block
        for block in (
            untrusted_block("topic_title", topic_title),
            untrusted_block("section_title", section_title or ""),
            untrusted_block(
                "section_content" if (section_title and section_content) else "documentation",
                content,
                _MAX_DOC_CONTEXT,
            ),
        )
        if block
    )
    few_shot_block = (
        _problem_solving_few_shot_example(problem_solving_language)
        if problem_solving_mode
        else _conceptual_few_shot_example()
    )
    prompt_contract = render_contract(
        schema_label="Return ONLY valid JSON in this shape",
        schema_block=f"""[
  {{
    "question": "Question text",
    "answer": "{answer_shape}",
    "difficulty": "easy|medium|hard"
  }}
]""",
        rules=[
            f"Return exactly {count} items; never return fewer.",
            "Questions must be standalone and non-duplicative.",
            "Questions must be NEW relative to already generated checkpoint questions listed above.",
            "Each question must be explicitly grounded to the current topic, section, or one of the grounding anchors below.",
            "Do not ask generic interview questions that could fit many unrelated topics with only minor wording changes.",
            "The question itself should make the current topic recognizable before the learner reads the answer.",
            *problem_scope_rules,
            "expected_signals should reference the relevant interview skills (clarify before coding, verbalise complexity, enumerate edge cases, discuss tradeoffs) where the question calls for them.",
            *_difficulty_generation_rules(problem_solving_mode, difficulty),
            *_question_auxiliary_field_rules(problem_solving_mode),
            "Answers must be grounded in the provided documentation.",
            detail_clause,
            "Format answers as markdown, but keep structure adaptive.",
            "long answers must be split into short readable paragraphs with blank lines.",
            "Do not compress complex topics into a few generic sentences; explain the mechanism, tradeoffs, edge cases, and the follow-up points a real interviewer would probe.",
            "Do not turn the answer into generic advice about how to answer. Give the actual explanation, decision, comparison, or solution directly.",
            *_study_markdown_rules(
                code_language=study_code_language or PROBLEM_SOLVING_DEFAULT_LANGUAGE,
                problem_solving_mode=problem_solving_mode,
            ),
            (
                "Use only the required headings for problem-solving answers."
                if problem_solving_mode
                else "Do not add extra headings to conceptual answers."
            ),
            (
                "Problem Solving answers must always include the required fenced code block."
                if problem_solving_mode
                else "You may include fenced code blocks when code clarifies an implementation detail."
            ),
            code_clause if code_clause else "Use code examples only when they materially improve clarity.",
            "Because output must be valid JSON, escape newlines, quotes, and backslashes correctly inside string values.",
            "source_quote must be factual text from the provided course content.",
            "question, source_quote, and answer must stay aligned to the same topic concept; do not mix unrelated concepts.",
            f'target_level must match "{target_level}" exactly.',
            "Do not wrap JSON with prose; return raw JSON only.",
            "Complexity must match target_level.",
            "junior: fundamentals, definitions, and straightforward tradeoffs.",
            "mid: implementation details, constraints, and moderate tradeoffs.",
            "senior: architecture, scaling, risk, and deep tradeoff decisions.",
            "Keep markdown practical and interview-usable, not terse.",
            UNTRUSTED_CLAUSE,
        ],
    )
    return f"""Generate exactly {count} interview-style questions with direct, detailed study answers.{diff_clause}
Target candidate level: "{target_level}".
Target scope: {scope}.
User requested total questions for this checkpoint: {requested_total}.
This call is generating {count} new questions to fill remaining slots.
{progress_block}{learner_state_block}{uniqueness_block}
{grounding_block}

{skill_section}

{prompt_contract}

Optional fields (recommended when available): learning_objective, source_section, source_quote,
reasoning_summary, target_level.

{few_shot_block}

{untrusted_blocks}{mcp_block}"""


#: The quiz-generation system prompt contract, migrated verbatim into
#: the prompt registry as ``quiz_generation@1``.
_QUIZ_SYSTEM_PROMPT = default_registry.get("quiz_generation").system_prompt

_QUIZ_FEW_SHOT_EXAMPLE = (
    "Worked example showing the required shape (its content is illustrative, not reusable "
    "content):\n"
    "[\n"
    '  {\n'
    '    "question": "Which statement about connection pooling is correct?",\n'
    '    "type": "mcq",\n'
    '    "choices": [{"label":"A","text":"A pool fixes the cost of acquiring a connection"}, '
    '{"label":"B","text":"A pool bounds concurrent connections and reuses warm ones"}, '
    '{"label":"C","text":"A pool removes the need for timeouts"}, '
    '{"label":"D","text":"A pool guarantees transaction isolation"}],\n'
    '    "correct_answer": "B",\n'
    '    "explanation": "Pooling reuses already-established connections, which removes most of '
    'the handshake cost, and caps how many can be open at once. It does not remove the need for '
    'statement timeouts, and isolation is still chosen per transaction.\\n\\n'
    "The wrong options fail for specific reasons worth naming: A describes what pooling does not "
    'remove, and D confuses capacity control with correctness guarantees.",\n'
    '    "difficulty": "medium",\n'
    '    "topic_id": "topic-databases",\n'
    '    "source_quote": "A pool bounds concurrent connections and reuses warm ones.",\n'
    '    "reasoning_summary": "Pooling is a capacity and latency control, not a correctness guarantee.",\n'
    '    "target_level": "mid"\n'
    "  },\n"
    '  {\n'
    '    "question": "A statement-level timeout guarantees a query cannot hold locks indefinitely.",\n'
    '    "type": "true_false",\n'
    '    "choices": [{"label":"A","text":"true"}, {"label":"B","text":"false"}],\n'
    '    "correct_answer": "B",\n'
    '    "explanation": "A timeout bounds how long the server waits, but a blocked transaction can '
    'still hold locks beyond it unless cancellation is honoured and the transaction is rolled back.",\n'
    '    "difficulty": "hard",\n'
    '    "topic_id": "topic-databases",\n'
    '    "source_quote": "Cancellation must be honoured and the transaction rolled back.",\n'
    '    "reasoning_summary": "The timeout bounds waiting, not lock ownership.",\n'
    '    "target_level": "mid"\n'
    "  }\n"
    "]"
)


def _build_quiz_prompt(
    topics_content: list[dict],
    count: int = 10,
    question_types: list[str] | None = None,
    difficulty: Optional[str] = None,
    level: Optional[str] = None,
    response_detail: str = "concise",
    preferred_language: str = "",
    existing_questions: Optional[list[str]] = None,
    additional_existing_questions: Optional[list[str]] = None,
    prior_progress: str = "",
    mcp_context: str = "",
    target_objectives: list[str] | None = None,
    learner_state: str = "",
) -> str:
    """Build the quiz user turn. The creator persona lives in ``system``."""
    target_level = _normalise_level(level)
    types = question_types or ["mcq", "true_false"]
    skill_section = _skill_context_section(
        target_level,
        "general",
        _skill_track_for_topics(topics_content),
    )
    existing_seed = _merge_existing_questions(existing_questions, additional_existing_questions)
    diff_clause = ""
    if difficulty:
        diff_clause = f' All questions should be "{difficulty}" difficulty.'

    if "mcq" in types and "true_false" in types:
        type_instructions = "Mix multiple-choice (exactly 4 options A/B/C/D) and true_false (exactly A=True, B=False)."
    elif "mcq" in types:
        type_instructions = "All questions must be MCQ with exactly 4 options A/B/C/D."
    else:
        type_instructions = "All questions must be true_false with exactly 2 options: A=True, B=False."

    detail_clause = (
        "Keep explanations concise and high-signal."
        if response_detail != "very_detailed"
        else "Provide very detailed explanations with practical depth."
    )
    docs_blocks: list[str] = []
    for tc in topics_content:
        topic_requires_programming = bool(tc.get("requires_programming"))
        topic_language = str(tc.get("preferred_language", "")).strip().lower()
        topic_detail = str(tc.get("response_detail", response_detail)).strip().lower()
        topic_notes = [f"detail={topic_detail or response_detail}"]
        if topic_requires_programming and topic_language:
            topic_notes.append(f"preferred_language={topic_language}")
        docs_blocks.append(
            untrusted_block(
                f"topic id={tc['id']}",
                f"Topic title: {tc['title']}\n"
                f"Notes: {', '.join(topic_notes)}\n\n"
                f"{tc['content']}",
                _MAX_TOPIC_CONTEXT,
            )
        )
    mcp_block = untrusted_block("external_context", mcp_context, 2500)
    progress_block = _progress_block(prior_progress)
    targeting_block = _target_objectives_block(target_objectives)
    learner_state_block = _learner_state_block(learner_state)
    uniqueness_block = ""
    if existing_seed:
        existing_blob = "\n".join([f"- {q}" for q in existing_seed[:60]])
        uniqueness_block = (
            "\nAlready asked for these topics. Do NOT repeat or rephrase these questions:\n"
            f"{existing_blob}\n"
        )
    prompt_contract = render_contract(
        schema_block="""[
  {
    "question": "Question text",
    "type": "mcq|true_false",
    "choices": [{"label":"A","text":"..."}, {"label":"B","text":"..."}, ...],
    "correct_answer": "A|B|C|D",
    "explanation": "2-4 sentence explanation",
    "difficulty": "easy|medium|hard",
    "topic_id": "one of provided topic ids",
    "source_quote": "short direct quote from course content",
    "reasoning_summary": "1-2 sentence why answer is correct",
    "target_level": "junior|mid|senior",
    "learning_objective": "what the learner should be able to do after answering",
    "expected_answer": "the full text of the correct choice",
    "distractor_rationales": [{"label": "B", "rationale": "why B is wrong"}]
  }
]""",
        rules=[
            f"Return exactly {count} items; never return fewer.",
            "topic_id must be one of the provided ids.",
            "correct_answer must match one choice label.",
            "Avoid trick ambiguity; one clearly correct answer.",
            "Exactly one choice is defensibly correct; every other choice must be wrong on the documented facts.",
            "Distractors must be plausible and technically distinct from each other and from the correct answer.",
            "Vary the position of the correct answer across items; do not default to A.",
            "source_quote must be a verbatim excerpt of the topic content; never invent a quotation.",
            "learning_objective states the skill the question assesses.",
            "expected_answer must repeat the correct choice text exactly.",
            "distractor_rationales must cover every non-correct choice label with a reason of at least 12 characters.",
            "Distribute questions across topics as evenly as possible.",
            (
                "Prioritize questions that assess the provided target learning objectives."
                if target_objectives
                else ""
            ),
            detail_clause,
            "Format explanations as markdown, but keep structure adaptive.",
            "long explanations must be split into short readable paragraphs with blank lines.",
            "use bullets only when listing steps/checklists/categories.",
            "use headings only when the explanation naturally has sections.",
            "use tables only for direct comparisons/category matrices.",
            "If you use a table, output valid GFM table syntax.",
            "one row per line.",
            "include a separator row (e.g. `| --- | --- |`).",
            "You may include fenced code blocks when code clarifies an implementation detail.",
            "Avoid unnecessary code blocks for non-programming topics.",
            (
                f'If code materially clarifies an explanation, prefer fenced "{preferred_language}" snippets.'
                if preferred_language
                else ""
            ),
            "You may include fenced Mermaid diagrams when architecture or flows are better shown visually.",
            "If you include fences, always use explicit language tags (for example: ```yaml, ```mermaid).",
            "Because output must be valid JSON, escape newlines, quotes, and backslashes correctly inside string values.",
            f'target_level must match "{target_level}" exactly.',
            "Complexity must match target_level.",
            "junior: direct recall and simple application.",
            "mid: applied reasoning with concrete constraints.",
            "senior: architecture-level reasoning and tradeoff depth.",
            "Keep markdown compact and practical.",
            UNTRUSTED_CLAUSE,
        ],
    )
    return f"""Create exactly {count} quiz questions from the documentation below.{diff_clause}
Target candidate level: "{target_level}".
{type_instructions}
{progress_block}{learner_state_block}{targeting_block}{uniqueness_block}
{skill_section}

{prompt_contract}

{_QUIZ_FEW_SHOT_EXAMPLE}

{chr(10).join(docs_blocks)[:_MAX_DOC_CONTEXT]}{mcp_block}"""


def _build_retry_prompt(
    base_prompt: str,
    missing_count: int,
    issues: str,
    existing_questions: list[str],
    hard_requirements: Optional[list[str]] = None,
    recovery_guidance: str = "",
    *,
    include_base_prompt: bool = False,
) -> str:
    """Build a short correction turn for one retry.

    The base prompt can carry up to ``_MAX_DOC_CONTEXT`` (45000) characters, so
    embedding it made every retry roughly twice the cost of the original call
    and let the worst case transmit the same context eighteen times. The base
    prompt was already sent on the first attempt of this cycle, so a retry now
    states only the correction. ``include_base_prompt`` restores the old
    behaviour for the single case that still wants a self-contained turn: the
    very first attempt after a prompt is rebuilt.
    """
    existing_blob = "\n".join([f"- {q}" for q in existing_questions[:50]])
    hard_requirements_block = ""
    if hard_requirements:
        hard_requirements_block = "Hard format requirements:\n" + "\n".join(
            [f"- {item}" for item in hard_requirements]
        )
    recovery_guidance_block = ""
    if recovery_guidance.strip():
        recovery_guidance_block = f"Recovery guidance:\n{recovery_guidance.strip()}\n"
    base_block = ""
    if include_base_prompt and base_prompt:
        base_block = (
            "\nThe full task contract and documentation were already sent with the "
            "previous attempt. Do not repeat that material here.\n"
        )
    return f"""Your previous output did not satisfy the schema or quality constraints.
The original task, its contract, and its documentation were already provided. Keep
answering that same task; only fix what the issues below describe.

Missing items needed: {missing_count}
Validation issues:
{issues}

{hard_requirements_block}
{recovery_guidance_block}

Do not repeat any of these existing questions:
{existing_blob if existing_blob else "- (none)"}

Return ONLY a JSON array with exactly {missing_count} NEW valid items.
{base_block}"""


def _strip_outer_code_fence(candidate: str) -> str:
    stripped = candidate.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _parse_text_qa_pairs(candidate: str) -> list[dict]:
    """Read ``Question:``/``Answer:`` prose into items.

    Kept deliberately: providers outside ``JSON_MODE_CAPABLE_PROVIDERS``
    (Anthropic) get no ``response_format``, so a prose response is a real
    outcome rather than a malformed one.
    """
    cleaned = candidate.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _strip_outer_code_fence(cleaned)

    q_pattern = re.compile(
        r"(?im)^\s*(?:\d+\s*[\).:\-]\s*)?(?:\*\*)?\s*(?:question|q)\s*\d*\s*[:\-]\s*"
    )
    a_pattern = re.compile(r"(?im)^\s*(?:\*\*)?\s*(?:answer|a)\s*\d*\s*[:\-]\s*")
    q_matches = list(q_pattern.finditer(cleaned))
    items: list[dict] = []

    if q_matches:
        for idx, q_match in enumerate(q_matches):
            block_end = q_matches[idx + 1].start() if idx + 1 < len(q_matches) else len(cleaned)
            block = cleaned[q_match.end() : block_end].strip()
            if not block:
                continue
            answer_match = a_pattern.search(block)
            if answer_match:
                question = block[: answer_match.start()].strip(" -*\t\n")
                answer = block[answer_match.end() :].strip(" \t\n")
            else:
                parts = [p.strip() for p in block.split("\n", 1)]
                question = parts[0].strip(" -*\t")
                answer = parts[1].strip() if len(parts) > 1 else ""
            if question and answer:
                items.append({"question": question, "answer": answer})
        if items:
            return items

    numbered_pattern = re.compile(
        r"(?ims)^\s*(\d+)[\).:\-]\s*(.+?)(?:\n\s*(?:answer|a)\s*[:\-]\s*(.+?))(?=^\s*\d+[\).:\-]|\Z)"
    )
    for match in numbered_pattern.finditer(cleaned):
        question = match.group(2).strip(" -*\t\n")
        answer = match.group(3).strip()
        if question and answer:
            items.append({"question": question, "answer": answer})
    return items


def _parse_questions_json(raw: str) -> list[dict]:
    """Extract a JSON array of items from an LLM response.

    The salvage chain is collapsed. The local sliding ``raw_decode``
    scan and the duplicated fenced re-parse existed only to re-read near-JSON,
    which the shared ``parse_json_object`` already handles (direct, then fenced,
    then first/last delimiter slice) and which ``structured=True`` makes rare.
    What stays is genuinely schema repair: unwrapping a ``{"questions": [...]}``
    envelope or a single-item object, and the plain-text reader that providers
    without JSON mode (Anthropic) still need.
    """
    text = (raw or "").strip()
    if not text:
        return []

    def _extract_items(payload: Any) -> list[dict]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            nested = payload.get("questions") or payload.get("items")
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
            if payload.get("question") and payload.get("answer"):
                return [payload]
        return []

    def _try_load_json(candidate: str) -> Any:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    candidates: list[Any] = [_try_load_json(text)]
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(_try_load_json(fenced.group(1)))
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(_try_load_json(text[start : end + 1]))
    # Shared salvage, so a dict envelope survives the same way every other flow.
    candidates.append(parse_json_object(text))

    for candidate in candidates:
        items = _extract_items(candidate)
        if items:
            return items

    return _parse_text_qa_pairs(text)


def _extract_content_fragments(content: str, *, limit: int = 48) -> list[str]:
    text = (content or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return []

    fragments: list[str] = []
    seen: set[str] = set()
    lines = text.splitlines()
    for line in lines:
        chunk = line.strip()
        if not chunk:
            continue
        if chunk.startswith("```"):
            continue
        chunk = re.sub(r"^#{1,6}\s*", "", chunk)
        chunk = re.sub(r"^[\-\*\d\.\)\s]+", "", chunk).strip()
        chunk = re.sub(r"\s+", " ", chunk)
        if len(chunk) < 6:
            continue
        if len(chunk) > 140:
            chunk = chunk[:140].rsplit(" ", 1)[0].strip() or chunk[:140]
        norm = _normalise_question(chunk)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        fragments.append(chunk.rstrip(":;,. "))
        if len(fragments) >= limit:
            break

    if not fragments:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in paragraphs[:limit]:
            cleaned = re.sub(r"\s+", " ", p)
            if len(cleaned) < 6:
                continue
            fragments.append(cleaned[:140].rsplit(" ", 1)[0].strip() or cleaned[:140])
            if len(fragments) >= limit:
                break
    return fragments


def _build_fallback_question_items(
    *,
    topic_id: str,
    topic_title: str,
    doc_content: str,
    count: int,
    difficulty: Optional[str],
    level: Optional[str],
    section_title: Optional[str] = None,
    section_content: Optional[str] = None,
    existing_questions: Optional[list[str]] = None,
    preferred_language: str = "",
    requires_programming: bool = False,
) -> list[dict[str, Any]]:
    remaining = max(0, int(count or 0))
    if remaining <= 0:
        return []

    problem_solving_mode = is_problem_solving_topic(topic_id)
    source_scope = section_title or topic_title or "this topic"
    source_content = section_content if (section_title and section_content) else doc_content
    fragments = _extract_content_fragments(source_content, limit=64)
    if not fragments:
        fragments = [source_scope, topic_title or "core concepts", "practical implementation"]
    grounding_anchors = _grounding_anchor_phrases(
        topic_title=topic_title,
        section_title=section_title,
        content=source_content,
    )

    existing_seed = _normalise_existing_questions(existing_questions)
    dedup: set[str] = {_normalise_question(q) for q in existing_seed}
    seen_questions: list[str] = [*existing_seed]
    selected_language = _study_code_language(
        preferred_language,
        problem_solving_mode=problem_solving_mode,
        requires_programming=requires_programming,
    )
    scenario_ids = (
        _problem_solving_scenario_ids_for_language(selected_language)
        if problem_solving_mode
        else []
    )
    if problem_solving_mode and not scenario_ids:
        # No genuine code template exists for this language. Emit a
        # language-agnostic problem prompt with no code template rather than
        # borrowing another language's source; the answer then fails
        # ``_validate_problem_solving_answer`` and is dropped, which is the
        # intended outcome (a wrong fence is worse than a short answer).
        logger.warning(
            "problem_solving_fallback_has_no_template_for_language language=%s",
            selected_language,
        )

    templates_by_difficulty = {
        "easy": [
            "What is {focus} in {topic}, and why does it matter in practice?",
            "How does {focus} work in {topic}, and what requirement is it protecting?",
            "When would you use {focus} in {topic}, and what first tradeoff would you mention?",
            "What problem does {focus} solve in {topic}, and what goes wrong when teams ignore it?",
            "How would you explain {focus} to a new backend engineer working on {topic}?",
            "What is the simplest correct way to reason about {focus} in {topic} before adding optimizations?",
        ],
        "medium": [
            "How would you implement {focus} in {topic} while protecting {angle}?",
            "If {focus} caused a production issue in {topic}, how would you debug it and what would you check first?",
            "How would you test {focus} in {topic} so you can defend the design under follow-up questions?",
            "What tradeoffs would you compare when choosing {focus} in {topic} with emphasis on {angle}?",
            "How would you roll out or migrate {focus} in {topic} safely, and what signal would you watch first?",
            "What bottleneck or edge case would you expect first when {focus} is used in {topic}, and how would you mitigate it?",
        ],
        "hard": [
            "If {focus} needed to scale in {topic}, what would you change first and what tradeoff would get worse?",
            "What failure modes or edge cases would you call out when discussing {focus} in {topic} with emphasis on {angle}?",
            "How would you explain the consistency, coordination, or bottleneck risk around {focus} in {topic} when the load grows sharply?",
            "If {focus} failed during a regional outage in {topic}, what recovery tradeoff would you make first?",
            "How would you redesign {focus} in {topic} when latency, correctness, and cost are all in tension?",
            "What interviewer follow-up would expose a weak answer about {focus} in {topic}, and how would you answer it at senior level?",
        ],
    }
    angles = [
        "correctness",
        "reliability",
        "performance",
        "security",
        "maintainability",
        "scalability",
        "observability",
        "cost efficiency",
        "testing",
    ]

    items: list[dict[str, Any]] = []
    idx = 0
    max_rounds = max(remaining * 12, 24)
    while len(items) < remaining and idx < max_rounds:
        focus = fragments[idx % len(fragments)]
        angle = angles[(idx // max(1, len(fragments))) % len(angles)]
        difficulty_value = _fallback_difficulty_for_index(len(items), remaining, difficulty)
        scenario_id = _PROBLEM_SOLVING_DEFAULT_SCENARIO_ID
        if problem_solving_mode and scenario_ids:
            scenario_id = scenario_ids[idx % len(scenario_ids)]
            question = _problem_solving_question_seed(
                focus,
                source_scope,
                scenario_id=scenario_id,
            )["question"]
        elif problem_solving_mode:
            # Language-agnostic prompt with no scenario template.
            question = (
                f"In {source_scope}, design an interview-grade algorithm problem about {focus}. "
                "State the input, output, and constraints, then explain the invariant your "
                "solution preserves, its complexity, and how you would test it."
            )
        else:
            templates = templates_by_difficulty.get(
                difficulty_value,
                templates_by_difficulty["medium"],
            )
            template = templates[idx % len(templates)]
            question = template.format(
                focus=focus,
                topic=topic_title or "this topic",
                angle=angle,
            ).strip()
        q_norm = _normalise_question(question)
        idx += 1
        if not q_norm or q_norm in dedup or _is_near_duplicate_question(question, seen_questions):
            continue

        if problem_solving_mode:
            answer, reasoning_summary = _build_problem_solving_fallback_answer(
                focus=focus,
                source_scope=source_scope,
                language=selected_language,
                scenario_id=scenario_id,
            )
        else:
            answer, reasoning_summary = _build_conceptual_fallback_answer(
                focus=focus,
                source_scope=source_scope,
                angle=angle,
                difficulty=difficulty_value,
                code_language=selected_language,
                requires_programming=requires_programming,
            )
        item: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "difficulty": difficulty_value,
            "learning_objective": _fallback_learning_objective(
                focus=focus,
                source_scope=source_scope,
                problem_solving_mode=problem_solving_mode,
            ),
            "source_section": source_scope,
            "source_quote": focus,
            "reasoning_summary": reasoning_summary,
            "target_level": _normalise_level(level),
            "topic_id": topic_id,
        }
        valid, _issue = _validate_question_item(
            item,
            topic_id,
            difficulty,
            level,
            preferred_language=selected_language,
            requires_programming=requires_programming,
            topic_title=topic_title,
            grounding_anchors=grounding_anchors,
            topic_content=source_content,
        )
        if not valid:
            continue
        dedup.add(q_norm)
        seen_questions.append(question)
        items.append(item)

    return items[:remaining]


def _validate_question_item(
    item: dict[str, Any],
    topic_id: str,
    difficulty: Optional[str],
    level: Optional[str],
    preferred_language: str = "",
    requires_programming: bool = False,
    topic_title: str = "",
    grounding_anchors: Optional[list[str]] = None,
    topic_content: str = "",
) -> tuple[bool, str]:
    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()
    if not question:
        return False, "missing_or_empty_question"
    if not answer:
        return False, "missing_or_empty_answer"

    diff = str(item.get("difficulty", "")).strip().lower()
    if difficulty:
        diff = str(difficulty).strip().lower()
    if diff not in _VALID_DIFFICULTIES:
        diff = "medium"

    target_level = _normalise_level(level)
    problem_solving_mode = is_problem_solving_topic(topic_id)
    item["question"] = question
    item["answer"] = answer
    item["learning_objective"] = (
        str(item.get("learning_objective", "")).strip()
        or _default_learning_objective(problem_solving_mode)
    )
    item["source_section"] = str(item.get("source_section", "")).strip() or "Topic section"
    source_quote = str(item.get("source_quote", "")).strip()
    if topic_content and not problem_solving_mode:
        if (
            not source_quote
            or _normalise_content_fragment(source_quote)
            not in _normalise_content_fragment(topic_content)
        ):
            return False, "source_quote_not_in_content"
        item["source_quote"] = source_quote
    else:
        item["source_quote"] = source_quote or "Generated from the current topic content."
    item["reasoning_summary"] = (
        str(item.get("reasoning_summary", "")).strip()
        or _default_reasoning_summary(problem_solving_mode)
    )
    item["target_level"] = target_level
    item["topic_id"] = topic_id
    item["difficulty"] = diff
    if problem_solving_mode:
        valid_answer, issue = _validate_problem_solving_answer(
            answer,
            preferred_language,
        )
        if not valid_answer:
            return False, issue
    if (topic_title or grounding_anchors) and not problem_solving_mode:
        valid_grounding, issue = _validate_topic_grounding(
            item,
            topic_title=topic_title,
            anchors=grounding_anchors or [],
            problem_solving_mode=problem_solving_mode,
        )
        if not valid_grounding:
            return False, issue
    return True, ""


def _validate_quiz_item(
    item: dict[str, Any],
    allowed_topics: set[str],
    allowed_types: set[str],
    difficulty: Optional[str],
    level: Optional[str],
    topic_content: str = "",
) -> tuple[bool, str]:
    required = [
        "question",
        "type",
        "choices",
        "correct_answer",
        "explanation",
        "topic_id",
        "source_quote",
        "reasoning_summary",
        "target_level",
        "learning_objective",
        "expected_answer",
        "distractor_rationales",
    ]
    for key in required:
        if key not in item:
            return False, f"missing_{key}"

    q_text = item.get("question")
    q_type = str(item.get("type", "")).lower()
    choices = item.get("choices")
    correct = str(item.get("correct_answer", "")).strip()
    explanation = item.get("explanation")
    topic_id = str(item.get("topic_id", "")).strip()
    source_quote = item.get("source_quote")
    reasoning_summary = item.get("reasoning_summary")
    target_level = str(item.get("target_level", "")).strip().lower()
    learning_objective = str(item.get("learning_objective", "")).strip()
    expected_answer = str(item.get("expected_answer", "")).strip()
    distractor_rationales = item.get("distractor_rationales")

    if not isinstance(q_text, str) or len(q_text.strip()) < 8:
        return False, "invalid_question"
    if q_type not in {"mcq", "true_false"}:
        return False, "invalid_type"
    if q_type not in allowed_types:
        return False, "disallowed_type"
    if topic_id not in allowed_topics:
        return False, "invalid_topic_id"
    if not isinstance(explanation, str) or len(explanation.strip()) < 15:
        return False, "invalid_explanation"
    if not isinstance(source_quote, str) or len(source_quote.strip()) < 8:
        return False, "invalid_source_quote"
    if topic_content and not is_problem_solving_topic(topic_id):
        if _normalise_content_fragment(source_quote) not in _normalise_content_fragment(
            topic_content
        ):
            return False, "source_quote_not_in_content"
    if not isinstance(reasoning_summary, str) or len(reasoning_summary.strip()) < 8:
        return False, "invalid_reasoning_summary"
    if target_level not in _VALID_LEVELS:
        return False, "invalid_target_level"
    if _normalise_level(level) != target_level:
        return False, "level_mismatch"
    if not learning_objective:
        return False, "invalid_learning_objective"
    if not expected_answer:
        return False, "invalid_expected_answer"
    if not isinstance(distractor_rationales, list):
        return False, "invalid_distractor_rationales"

    if not isinstance(choices, list):
        return False, "invalid_choices_type"
    labels = []
    for c in choices:
        if not isinstance(c, dict):
            return False, "invalid_choice_item"
        label = str(c.get("label", "")).strip()
        text = c.get("text")
        if label not in {"A", "B", "C", "D"}:
            return False, "invalid_choice_label"
        if not isinstance(text, str) or not text.strip():
            return False, "invalid_choice_text"
        labels.append(label)

    if q_type == "mcq":
        if labels != ["A", "B", "C", "D"]:
            return False, "mcq_labels_must_be_abcd"
        if len(choices) != 4:
            return False, "mcq_must_have_4_choices"
        if correct not in {"A", "B", "C", "D"}:
            return False, "invalid_correct_answer_mcq"
    else:
        if len(choices) != 2:
            return False, "true_false_must_have_2_choices"
        if labels != ["A", "B"]:
            return False, "true_false_labels_must_be_ab"
        true_false_texts = [str(choices[0].get("text", "")).strip().lower(), str(choices[1].get("text", "")).strip().lower()]
        if true_false_texts != ["true", "false"]:
            return False, "true_false_text_must_be_true_false"
        if correct not in {"A", "B"}:
            return False, "invalid_correct_answer_true_false"

    correct_text = ""
    for c in choices:
        if str(c.get("label", "")).strip() == correct:
            correct_text = str(c.get("text", "")).strip()
            break
    if q_type == "mcq" and expected_answer != correct_text:
        return False, "expected_answer_mismatch"

    rationale_by_label: dict[str, str] = {}
    for entry in distractor_rationales:
        if not isinstance(entry, dict):
            return False, "invalid_distractor_rationales"
        rationale_label = str(entry.get("label", "")).strip()
        rationale = str(entry.get("rationale", "")).strip()
        if rationale_label not in {"A", "B", "C", "D"} or len(rationale) < 12:
            return False, "invalid_distractor_rationales"
        rationale_by_label[rationale_label] = rationale
    for label in labels:
        if label != correct and label not in rationale_by_label:
            return False, "missing_distractor_rationale"

    diff = str(item.get("difficulty", "medium")).lower()
    if diff not in _VALID_DIFFICULTIES:
        return False, "invalid_difficulty"
    requested = str(difficulty or "").strip().lower()
    if requested and diff != requested:
        return False, "difficulty_mismatch"
    item["difficulty"] = diff
    item["type"] = q_type
    return True, ""


def _topic_content_for(
    item: dict[str, Any],
    topics_content: list[dict],
) -> str:
    topic_id = str(item.get("topic_id", "")).strip()
    for topic in topics_content:
        if str(topic.get("id", "")).strip() == topic_id:
            return str(topic.get("content", ""))
    return ""


def _normalise_answer_position(item: dict[str, Any]) -> dict[str, Any]:
    choices = item.get("choices")
    correct = str(item.get("correct_answer", "")).strip()
    if (
        not isinstance(choices, list)
        or len(choices) != 4
        or correct not in {"A", "B", "C", "D"}
    ):
        return item
    correct_index = next(
        (
            index
            for index, choice in enumerate(choices)
            if isinstance(choice, dict)
            and str(choice.get("label", "")).strip() == correct
        ),
        None,
    )
    if correct_index is None:
        return item
    question = _normalise_question(str(item.get("question", "")))
    if not question:
        return item
    digest = hashlib.sha256(question.encode("utf-8")).digest()
    target = digest[0] % len(choices)
    shift = (target - correct_index) % len(choices)
    rotated = choices[-shift:] + choices[:-shift]
    item["choices"] = [
        {**choice, "label": chr(ord("A") + index)}
        for index, choice in enumerate(rotated)
    ]
    item["correct_answer"] = chr(ord("A") + target)
    return item


async def _collect_with_retries(
    *,
    llm: LLMClient,
    base_prompt: str,
    llm_config: Optional[LLMConfigRequest],
    user_identity: Optional[dict] = None,
    target_count: int,
    validator: Callable[[dict[str, Any]], tuple[bool, str]],
    existing_questions: Optional[list[str]] = None,
    hard_requirements: Optional[list[str]] = None,
    retry_guidance: str = "",
    system_prompt: str = "",
    prompt_name: str = "",
    prompt_version: str = "",
    flow: str = "",
    run_store: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing_seed = _normalise_existing_questions(existing_questions)
    dedup: set[str] = {_normalise_question(q) for q in existing_seed}
    seen_questions: list[str] = [*existing_seed]
    valid_items: list[dict[str, Any]] = []
    malformed_items = 0
    retries_used = 0
    attempt_count = 0
    issue_counter: Counter[str] = Counter()
    metadata: dict[str, Any] = {}
    max_tokens_cap = _questions_max_tokens_cap(target_count)
    prompt = base_prompt
    last_error = ""
    last_error_code = ""
    last_finish_reason = ""
    terminal_reason = ""

    for _ in range(_attempt_budget(target_count)):
        attempt_count += 1
        result = await llm.completion(
            prompt,
            llm_config,
            user_identity=user_identity,
            **_supported_options(
                llm,
                {
                    "task": "final",
                    "system": system_prompt,
                    "structured": True,
                    "max_tokens_cap": max_tokens_cap,
                    "attempts": attempt_count,
                    "terminal_reason": terminal_reason,
                    "prompt_name": prompt_name,
                    "prompt_version": prompt_version,
                },
            ),
        )
        raise_if_policy_blocked_result(result)
        metadata = result.get("metadata", {})
        last_finish_reason = str(result.get("finish_reason", "") or "")
        last_error_code = str(result.get("error_code", "") or "")
        if not result.get("success"):
            last_error = result.get("error", "unknown_error")
            error_code = str(result.get("error_code") or "")
            if error_code in TERMINAL_ERROR_CODES:
                # A truncated response or an exhausted budget cannot be fixed by
                # re-sending the same prompt at the same cap.
                terminal_reason = error_code
                break
            retries_used += 1
            prompt = _build_retry_prompt(
                base_prompt=base_prompt,
                missing_count=max(1, target_count - len(valid_items)),
                issues=f"transport_or_provider_error: {last_error}",
                existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
                hard_requirements=hard_requirements,
                recovery_guidance=retry_guidance,
            )
            continue

        parsed = _parse_questions_json(result.get("analysis", ""))
        if not parsed:
            retries_used += 1
            issue_counter["json_parse_failed"] += 1
            prompt = _build_retry_prompt(
                base_prompt=base_prompt,
                missing_count=max(1, target_count - len(valid_items)),
                issues="json_parse_failed",
                existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
                hard_requirements=hard_requirements,
                recovery_guidance=retry_guidance,
            )
            continue

        for item in parsed:
            if not isinstance(item, dict):
                malformed_items += 1
                issue_counter["non_dict_item"] += 1
                continue

            is_valid, issue = validator(item)
            if not is_valid:
                malformed_items += 1
                issue_counter[issue] += 1
                continue

            item = _normalise_answer_position(item)
            q_norm = _normalise_question(str(item.get("question", "")))
            question_text = str(item.get("question", "")).strip()
            if not q_norm or q_norm in dedup or _is_near_duplicate_question(question_text, seen_questions):
                malformed_items += 1
                issue_counter["duplicate_question"] += 1
                continue
            dedup.add(q_norm)
            seen_questions.append(question_text)
            valid_items.append(item)
            if len(valid_items) >= target_count:
                break

        if len(valid_items) >= target_count:
            break

        retries_used += 1
        top_issues = ", ".join([f"{k}:{v}" for k, v in issue_counter.most_common(5)]) or "insufficient_valid_items"
        prompt = _build_retry_prompt(
            base_prompt=base_prompt,
            missing_count=target_count - len(valid_items),
            issues=top_issues,
            existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
            hard_requirements=hard_requirements,
            recovery_guidance=retry_guidance,
        )

    if len(valid_items) < target_count and not terminal_reason:
        # Final pass: request one item at a time to reduce duplicate/schema drift.
        for _ in range(_recovery_budget(target_count - len(valid_items))):
            if len(valid_items) >= target_count:
                break

            single_prompt = _build_retry_prompt(
                base_prompt=base_prompt,
                missing_count=1,
                issues="final_recovery_fill_missing_items",
                existing_questions=[*existing_seed, *[x.get("question", "") for x in valid_items]],
                hard_requirements=hard_requirements,
                recovery_guidance=retry_guidance,
            )
            attempt_count += 1
            result = await llm.completion(
                single_prompt,
                llm_config,
                user_identity=user_identity,
                **_supported_options(
                    llm,
                    {
                        "task": "final",
                        "system": system_prompt,
                        "structured": True,
                        "max_tokens_cap": max_tokens_cap,
                        "attempts": attempt_count,
                        "terminal_reason": terminal_reason,
                        "prompt_name": prompt_name,
                        "prompt_version": prompt_version,
                    },
                ),
            )
            raise_if_policy_blocked_result(result)
            metadata = result.get("metadata", {})
            last_finish_reason = str(result.get("finish_reason", "") or "")
            last_error_code = str(result.get("error_code", "") or "")
            if not result.get("success"):
                error_code = str(result.get("error_code") or "")
                if error_code in TERMINAL_ERROR_CODES:
                    # Single-item recovery cannot fix a truncated response or an
                    # exhausted budget either.
                    terminal_reason = error_code
                    break
                retries_used += 1
                issue_counter["transport_or_provider_error"] += 1
                continue

            parsed = _parse_questions_json(result.get("analysis", ""))
            if not parsed:
                retries_used += 1
                malformed_items += 1
                issue_counter["json_parse_failed"] += 1
                continue

            added = False
            for item in parsed:
                if not isinstance(item, dict):
                    malformed_items += 1
                    issue_counter["non_dict_item"] += 1
                    continue
                is_valid, issue = validator(item)
                if not is_valid:
                    malformed_items += 1
                    issue_counter[issue] += 1
                    continue

                item = _normalise_answer_position(item)
                q_norm = _normalise_question(str(item.get("question", "")))
                question_text = str(item.get("question", "")).strip()
                if not q_norm or q_norm in dedup or _is_near_duplicate_question(question_text, seen_questions):
                    malformed_items += 1
                    issue_counter["duplicate_question"] += 1
                    continue

                dedup.add(q_norm)
                seen_questions.append(question_text)
                valid_items.append(item)
                added = True
                break

            if not added:
                retries_used += 1

    if len(valid_items) < target_count:
        logger.warning(
            "Generation produced %d/%d valid items after retries=%d issues=%s last_error=%s terminal=%s",
            len(valid_items),
            target_count,
            retries_used,
            dict(issue_counter),
            last_error,
            terminal_reason,
        )

    stats = {
        "metadata": dict(
            metadata,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
        ),
        "retries_used": retries_used,
        "malformed_items_dropped": malformed_items,
        "terminal_reason": terminal_reason,
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "attempts": attempt_count,
    }
    await record_generation_run_safe(
        run_store,
        user_identity=user_identity,
        metadata=metadata,
        flow=flow,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        attempts=attempt_count,
        terminal_reason=terminal_reason,
        finish_reason=last_finish_reason,
        error_code=last_error_code,
    )
    return valid_items[:target_count], stats


class QuestionGenerator:
    """Uses LiteLLM to generate interview Q&A and quizzes."""

    def __init__(
        self,
        llm_client: LLMClient,
        mcp_gateway: MCPGateway | None = None,
        run_store: Any = None,
    ):
        self.llm = llm_client
        self.mcp = mcp_gateway
        #: Optional :class:`~app.services.learning_store.LearningStore`
        #: (or any object exposing ``record_generation_run``) that
        #: persists one telemetry row per completed generation flow. When
        #: it also exposes ``get_level_recommendation`` it is the source
        #: for adaptive difficulty (see :meth:`_resolve_generation_level`).
        self.run_store = run_store

    def _recommended_level(self, *, user_id: str, topic_id: str) -> str | None:
        """History-derived level for one topic, or ``None`` with no signal.

        Best-effort by design: a recommendation failure must never fail
        the generation, so any error degrades to ``None`` (no signal) and
        the caller keeps its own default level.
        """
        store = self.run_store
        lookup = (
            getattr(store, "get_level_recommendation", None)
            if store is not None
            else None
        )
        if lookup is None:
            return None
        try:
            recommended = lookup(user_id, topic_id)
        except Exception:
            logger.warning(
                "level recommendation failed user=%s topic=%s",
                user_id,
                topic_id,
                exc_info=True,
            )
            return None
        return str(recommended) if recommended else None

    def _resolve_generation_level(
        self,
        *,
        level: Optional[str],
        user_identity: Optional[dict],
        topic_id: str,
    ) -> tuple[str, str]:
        """Return ``(effective_level, resolved_level)`` for a flow.

        ``effective_level`` is what the prompts and validators run at.
        ``resolved_level`` is the history-derived level surfaced to the
        client so the UI can mark it as recommended; it is empty when the
        request pinned its own level, because then nothing was resolved.
        """
        if not _level_is_auto(level):
            # An explicit level never consults the learner's history.
            return resolve_level(level), ""
        user_id = str((user_identity or {}).get("user", "") or "")
        recommended = (
            self._recommended_level(user_id=user_id, topic_id=topic_id)
            if user_id and topic_id
            else None
        )
        resolved_level = resolve_level(level, recommended)
        return resolved_level, resolved_level

    async def _mcp_context_for_flow(
        self,
        *,
        flow: str,
        query: str,
        topic_id: str = "",
        topic_title: str = "",
    ) -> str:
        if self.mcp is None:
            return ""
        return await self.mcp.gather_context(
            flow=flow,
            query=query,
            topic_id=topic_id,
            topic_title=topic_title,
        )

    @staticmethod
    def _to_question_answer_v2(topic_id: str, item: dict[str, Any]) -> QuestionAnswerV2:
        return QuestionAnswerV2(
            question_id=_question_id(topic_id, item["question"]),
            topic_id=topic_id,
            question=item["question"].strip(),
            answer=format_markdown_readable(item["answer"].strip()),
            difficulty=item["difficulty"],
            learning_objective=item["learning_objective"].strip(),
            source_section=item["source_section"].strip(),
            source_quote=item["source_quote"].strip(),
            reasoning_summary=item["reasoning_summary"].strip(),
        )

    @staticmethod
    def _to_quiz_question_v2(item: dict[str, Any]) -> QuizQuestionV2:
        topic_id = str(item.get("topic_id", "")).strip()
        qid = _question_id(topic_id or "quiz", str(item.get("question", "")))
        choices = [QuizChoice(label=c["label"], text=c["text"]) for c in item["choices"]]
        return QuizQuestionV2(
            question_id=qid,
            question=str(item.get("question", "")).strip(),
            type=QuizQuestionType(str(item.get("type", "mcq")).strip().lower()),
            choices=choices,
            correct_answer=str(item.get("correct_answer", "")).strip(),
            explanation=format_markdown_readable(str(item.get("explanation", "")).strip()),
            difficulty=str(item.get("difficulty", "medium")).strip().lower() or "medium",
            topic_id=topic_id,
            source_quote=str(item.get("source_quote", "")).strip(),
            reasoning_summary=str(item.get("reasoning_summary", "")).strip(),
            learning_objective=str(item.get("learning_objective", "")).strip(),
            expected_answer=str(item.get("expected_answer", "")).strip(),
            distractor_rationales=[
                QuizDistractorRationale(
                    label=str(rationale.get("label", "")).strip(),
                    rationale=str(rationale.get("rationale", "")).strip(),
                )
                for rationale in item.get("distractor_rationales", [])
                if isinstance(rationale, dict)
            ],
        )

    @staticmethod
    def _policy_error_payload(exc: Exception) -> tuple[str, str]:
        if isinstance(exc, LLMServiceApprovalRequiredError):
            return APPROVAL_REQUIRED_CODE, APPROVAL_REQUIRED_MESSAGE
        if isinstance(exc, StudyAppLLMNotAssignedError):
            return STUDY_APP_NOT_ASSIGNED_CODE, STUDY_APP_NOT_ASSIGNED_MESSAGE
        if isinstance(exc, PersonalCredentialRequiredError):
            return PERSONAL_CREDENTIAL_REQUIRED_CODE, PERSONAL_CREDENTIAL_REQUIRED_MESSAGE
        return "generation_failed", str(exc).strip() or "Question generation failed"

    async def generate_v2_stream(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 5,
        requested_total_count: Optional[int] = None,
        existing_questions: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
        response_detail: str = "very_detailed",
        preferred_language: str = "",
        requires_programming: bool = False,
        prior_progress: str = "",
        additional_existing_questions: Optional[list[str]] = None,
        topic_track: str = "",
        learner_state: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        target_count = max(0, int(count))
        requested_total = max(1, int(requested_total_count or count or 1))
        effective_level, resolved_level = self._resolve_generation_level(
            level=level, user_identity=user_identity, topic_id=topic_id
        )
        seed_existing = _merge_existing_questions(
            existing_questions, additional_existing_questions
        )
        problem_solving_mode = is_problem_solving_topic(topic_id)
        selected_language = _problem_solving_language(preferred_language) if problem_solving_mode else (
            (preferred_language or "").strip().lower()
        )
        hard_requirements = (
            _problem_solving_required_markdown(selected_language) if problem_solving_mode else None
        )
        retry_guidance = _question_retry_guidance(problem_solving_mode)
        mcp_context = await self._mcp_context_for_flow(
            flow="questions",
            query=f"{topic_title} {section_title or ''} interview questions and practical examples",
            topic_id=topic_id,
            topic_title=topic_title,
        )
        dedup_norm: set[str] = {_normalise_question(q) for q in seed_existing}
        generated_question_texts: list[str] = [*seed_existing]
        problem_pattern_signatures: set[str] = set()
        max_unique_problem_patterns = (
            _problem_solving_unique_pattern_target(target_count)
            if problem_solving_mode
            else 0
        )

        async def _generate_one(*, include_section: bool, relaxed: bool = False) -> dict[str, Any]:
            retries_used = 0
            attempt_count = 0
            malformed_dropped = 0
            metadata: dict[str, Any] = {}
            issue_counter: Counter[str] = Counter()
            last_error = ""
            active_section_title = section_title if include_section else None
            active_section_content = section_content if include_section else None
            active_grounding_anchors = _grounding_anchor_phrases(
                topic_title=topic_title,
                section_title=active_section_title,
                content=active_section_content or doc_content,
            )
            base_prompt = _build_prompt(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=1,
                difficulty=None if relaxed else difficulty,
                level=effective_level,
                section_title=active_section_title,
                section_content=active_section_content,
                response_detail=response_detail,
                preferred_language=selected_language if (problem_solving_mode or not relaxed) else "",
                requires_programming=requires_programming if (problem_solving_mode or not relaxed) else False,
                requested_total_count=requested_total,
                existing_questions=generated_question_texts,
                additional_existing_questions=additional_existing_questions,
                prior_progress=prior_progress,
                mcp_context=mcp_context,
                track=topic_track,
                learner_state=learner_state,
            )
            prompt = base_prompt
            system_prompt = _question_system_prompt(problem_solving_mode)
            max_tokens_cap = _questions_max_tokens_cap(1)
            terminal_reason = ""

            try:
                for _ in range(_attempt_budget(1)):
                    attempt_count += 1
                    result = await self.llm.completion(
                        prompt,
                        llm_config,
                        user_identity=user_identity,
                        **_supported_options(
                            self.llm,
                            {
                                "task": "final",
                                "system": system_prompt,
                                "structured": True,
                                "max_tokens_cap": max_tokens_cap,
                                "attempts": attempt_count,
                                "terminal_reason": terminal_reason,
                                "prompt_name": (
                                    "question_generation_problem_solving"
                                    if problem_solving_mode
                                    else "question_generation"
                                ),
                                "prompt_version": default_registry.latest_version(
                                    "question_generation_problem_solving"
                                    if problem_solving_mode
                                    else "question_generation"
                                ),
                            },
                        ),
                    )
                    raise_if_policy_blocked_result(result)
                    metadata = result.get("metadata", {})

                    if not result.get("success"):
                        error_code = str(result.get("error_code") or "")
                        if error_code in TERMINAL_ERROR_CODES:
                            # Retrying a truncated response or an exhausted
                            # budget re-sends the same prompt at the same cap.
                            last_error = f"{error_code}: {result.get('error', '')}".strip(": ")
                            terminal_reason = error_code
                            break
                        retries_used += 1
                        last_error = str(result.get("error", "unknown_error"))
                        issue_counter["transport_or_provider_error"] += 1
                        prompt = _build_retry_prompt(
                            base_prompt=base_prompt,
                            missing_count=1,
                            issues=f"transport_or_provider_error: {last_error}",
                            existing_questions=generated_question_texts,
                            hard_requirements=hard_requirements,
                            recovery_guidance=retry_guidance,
                        )
                        continue

                    parsed = _parse_questions_json(result.get("analysis", ""))
                    if not parsed:
                        retries_used += 1
                        malformed_dropped += 1
                        issue_counter["json_parse_failed"] += 1
                        prompt = _build_retry_prompt(
                            base_prompt=base_prompt,
                            missing_count=1,
                            issues="json_parse_failed",
                            existing_questions=generated_question_texts,
                            hard_requirements=hard_requirements,
                            recovery_guidance=retry_guidance,
                        )
                        continue

                    selected_item: dict[str, Any] | None = None
                    for item in parsed:
                        if not isinstance(item, dict):
                            malformed_dropped += 1
                            issue_counter["non_dict_item"] += 1
                            continue

                        valid, issue = _validate_question_item(
                            item,
                            topic_id,
                            difficulty,
                            effective_level,
                            preferred_language=selected_language,
                            requires_programming=requires_programming,
                            topic_title=topic_title,
                            grounding_anchors=active_grounding_anchors,
                            topic_content=doc_content,
                        )
                        if not valid:
                            malformed_dropped += 1
                            issue_counter[issue] += 1
                            continue

                        q_text = str(item.get("question", "")).strip()
                        q_norm = _normalise_question(q_text)
                        if not q_norm:
                            malformed_dropped += 1
                            issue_counter["invalid_question"] += 1
                            continue

                        if q_norm in dedup_norm or _is_near_duplicate_question(q_text, generated_question_texts):
                            malformed_dropped += 1
                            issue_counter["duplicate_question"] += 1
                            continue

                        if problem_solving_mode:
                            signature = _problem_solving_pattern_signature(
                                q_text,
                                str(item.get("answer", "")),
                            )
                            if not _accept_problem_pattern_signature(
                                signature=signature,
                                seen_signatures=problem_pattern_signatures,
                                unique_target=max_unique_problem_patterns,
                            ):
                                malformed_dropped += 1
                                issue_counter["duplicate_problem_pattern"] += 1
                                continue

                        dedup_norm.add(q_norm)
                        generated_question_texts.append(q_text)
                        selected_item = item
                        break

                    if selected_item is not None:
                        qa = self._to_question_answer_v2(topic_id, selected_item)
                        return {
                            "question": qa.model_dump(),
                            "metadata": metadata,
                            "retries_used": retries_used,
                            "malformed_items_dropped": malformed_dropped,
                        }

                    retries_used += 1
                    top_issues = (
                        ", ".join(f"{k}:{v}" for k, v in issue_counter.most_common(5))
                        or "insufficient_valid_items"
                    )
                    prompt = _build_retry_prompt(
                        base_prompt=base_prompt,
                        missing_count=1,
                        issues=top_issues,
                        existing_questions=generated_question_texts,
                        hard_requirements=hard_requirements,
                        recovery_guidance=retry_guidance,
                    )

                if issue_counter:
                    logger.warning(
                        "Streaming slot failed to produce valid question after retries=%d issues=%s last_error=%s terminal=%s",
                        retries_used,
                        dict(issue_counter),
                        last_error,
                        terminal_reason,
                    )
                return {
                    "question": None,
                    "metadata": metadata,
                    "retries_used": retries_used,
                    "malformed_items_dropped": malformed_dropped,
                    "terminal_reason": terminal_reason,
                }
            except (
                LLMServiceApprovalRequiredError,
                StudyAppLLMNotAssignedError,
                PersonalCredentialRequiredError,
            ) as exc:
                code, message = self._policy_error_payload(exc)
                return {
                    "error": {"code": code, "message": message},
                    "metadata": metadata,
                    "retries_used": retries_used,
                    "malformed_items_dropped": malformed_dropped,
                }
            except Exception as exc:
                logger.exception("Streaming question generation failed: %s", exc)
                return {
                    "error": {
                        "code": "generation_failed",
                        "message": str(exc).strip() or "Question generation failed",
                    },
                    "metadata": metadata,
                    "retries_used": retries_used,
                    "malformed_items_dropped": malformed_dropped,
                }

        yield {
            "type": "start",
            "topic_id": topic_id,
            "topic_title": topic_title,
            "target_count": target_count,
        }

        if target_count == 0:
            yield {
                "type": "done",
                "topic_id": topic_id,
                "topic_title": topic_title,
                "generated_count": 0,
                "provider_used": "",
                "model_used": "",
                "retries_used": 0,
                "malformed_items_dropped": 0,
                "degraded": False,
                "requested_count": 0,
                "returned_count": 0,
                "resolved_level": resolved_level,
            }
            return

        provider_used = ""
        model_used = ""
        retries_used = 0
        malformed_items_dropped = 0
        generated_count = 0

        section_first = bool(section_title and section_content)
        phase_modes: list[tuple[bool, bool]] = (
            [(True, False), (False, False)]
            if section_first
            else [(False, False)]
        )
        if not problem_solving_mode:
            phase_modes = (
                [(True, False), (False, False), (False, True)]
                if section_first
                else [(False, False), (False, True)]
            )
        terminal_reason = ""
        for include_section, relaxed in phase_modes:
            while generated_count < target_count:
                result = await _generate_one(include_section=include_section, relaxed=relaxed)
                retries_used += int(result.get("retries_used", 0))
                malformed_items_dropped += int(result.get("malformed_items_dropped", 0))
                metadata = result.get("metadata", {}) or {}
                if metadata.get("provider"):
                    provider_used = str(metadata.get("provider"))
                if metadata.get("model"):
                    model_used = str(metadata.get("model"))
                terminal_reason = str(result.get("terminal_reason") or "")
                if terminal_reason:
                    # A truncated response or an exhausted budget will not change
                    # on the next slot, so stop and fall back deterministically.
                    break

                error = result.get("error")
                if error:
                    yield {
                        "type": "error",
                        "code": str(error.get("code", "generation_failed")),
                        "message": str(error.get("message", "Question generation failed")),
                    }
                    return

                question = result.get("question")
                if not isinstance(question, dict):
                    break

                generated_count += 1
                yield {
                    "type": "question",
                    "question": question,
                }

            if terminal_reason:
                break

        if generated_count < target_count and not terminal_reason:
            for _ in range(_recovery_budget(target_count - generated_count)):
                if generated_count >= target_count:
                    break

                result = await _generate_one(
                    include_section=False,
                    relaxed=False if problem_solving_mode else True,
                )
                retries_used += int(result.get("retries_used", 0))
                malformed_items_dropped += int(result.get("malformed_items_dropped", 0))
                metadata = result.get("metadata", {}) or {}
                if metadata.get("provider"):
                    provider_used = str(metadata.get("provider"))
                if metadata.get("model"):
                    model_used = str(metadata.get("model"))
                if str(result.get("terminal_reason") or ""):
                    # A truncated response or an exhausted budget will not change
                    # on the next recovery attempt; fall back deterministically.
                    break

                error = result.get("error")
                if error:
                    yield {
                        "type": "error",
                        "code": str(error.get("code", "generation_failed")),
                        "message": str(error.get("message", "Question generation failed")),
                    }
                    return

                question = result.get("question")
                if not isinstance(question, dict):
                    continue

                generated_count += 1
                yield {
                    "type": "question",
                    "question": question,
                }

        fallback_injected = False
        if generated_count < target_count:
            missing_count = target_count - generated_count
            fallback_items = _build_fallback_question_items(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=_problem_solving_fallback_count_with_headroom(
                    missing_count,
                    problem_solving_mode,
                ),
                difficulty=difficulty,
                level=effective_level,
                section_title=None,
                section_content=None,
                existing_questions=generated_question_texts,
                preferred_language=selected_language,
                requires_programming=requires_programming,
            )
            for item in fallback_items:
                q_text = str(item.get("question", "")).strip()
                q_norm = _normalise_question(q_text)
                if not q_norm or q_norm in dedup_norm or _is_near_duplicate_question(q_text, generated_question_texts):
                    continue
                if problem_solving_mode:
                    signature = _problem_solving_pattern_signature(
                        q_text,
                        str(item.get("answer", "")),
                    )
                    if not _accept_problem_pattern_signature(
                        signature=signature,
                        seen_signatures=problem_pattern_signatures,
                        unique_target=max_unique_problem_patterns,
                    ):
                        continue
                dedup_norm.add(q_norm)
                generated_question_texts.append(q_text)
                qa = self._to_question_answer_v2(topic_id, item)
                generated_count += 1
                fallback_injected = True
                yield {
                    "type": "question",
                    "question": qa.model_dump(),
                }
                if generated_count >= target_count:
                    break

        yield {
            "type": "done",
            "topic_id": topic_id,
            "topic_title": topic_title,
            "generated_count": generated_count,
            "provider_used": provider_used,
            "model_used": model_used,
            "retries_used": retries_used,
            "malformed_items_dropped": malformed_items_dropped,
            "degraded": fallback_injected,
            "requested_count": target_count,
            "returned_count": generated_count,
            "resolved_level": resolved_level,
        }

    async def generate(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 5,
        requested_total_count: Optional[int] = None,
        existing_questions: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
        response_detail: str = "very_detailed",
        preferred_language: str = "",
        requires_programming: bool = False,
        prior_progress: str = "",
        additional_existing_questions: Optional[list[str]] = None,
        topic_track: str = "",
        learner_state: str = "",
    ) -> GenerateQuestionsResponse:
        problem_solving_mode = is_problem_solving_topic(topic_id)
        effective_level, resolved_level = self._resolve_generation_level(
            level=level, user_identity=user_identity, topic_id=topic_id
        )
        selected_language = _problem_solving_language(preferred_language) if problem_solving_mode else (
            (preferred_language or "").strip().lower()
        )
        v2 = await self.generate_v2(
            topic_id=topic_id,
            topic_title=topic_title,
            doc_content=doc_content,
            count=count,
            requested_total_count=requested_total_count,
            existing_questions=existing_questions,
            difficulty=difficulty,
            level=effective_level,
            llm_config=llm_config,
            user_identity=user_identity,
            section_title=section_title,
            section_content=section_content,
            response_detail=response_detail,
            preferred_language=selected_language,
            requires_programming=requires_programming,
            prior_progress=prior_progress,
            additional_existing_questions=additional_existing_questions,
            topic_track=topic_track,
            learner_state=learner_state,
        )
        merged_seed = _merge_existing_questions(
            existing_questions, additional_existing_questions
        )
        target_count = max(1, int(count or 1))
        questions = [
            QuestionAnswer(question=q.question, answer=q.answer, difficulty=q.difficulty)
            for q in v2.questions
        ]
        problem_pattern_signatures: set[str] = set()
        max_unique_problem_patterns = (
            _problem_solving_unique_pattern_target(target_count)
            if problem_solving_mode
            else 0
        )
        if problem_solving_mode:
            for q in questions:
                signature = _problem_solving_pattern_signature(q.question, q.answer)
                if signature:
                    problem_pattern_signatures.add(signature)
        if len(questions) < target_count:
            missing_count = target_count - len(questions)
            fallback_items = _build_fallback_question_items(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=_problem_solving_fallback_count_with_headroom(
                    missing_count,
                    problem_solving_mode,
                ),
                difficulty=difficulty,
                level=effective_level,
                section_title=section_title,
                section_content=section_content,
                existing_questions=[
                    *merged_seed,
                    *[q.question for q in questions],
                ],
                preferred_language=selected_language,
                requires_programming=requires_programming,
            )
            for item in fallback_items:
                if problem_solving_mode:
                    signature = _problem_solving_pattern_signature(
                        str(item.get("question", "")).strip(),
                        str(item.get("answer", "")),
                    )
                    if not _accept_problem_pattern_signature(
                        signature=signature,
                        seen_signatures=problem_pattern_signatures,
                        unique_target=max_unique_problem_patterns,
                    ):
                        continue
                questions.append(
                    QuestionAnswer(
                        question=str(item.get("question", "")).strip(),
                        answer=str(item.get("answer", "")).strip(),
                        difficulty=str(item.get("difficulty", "medium")).strip().lower() or "medium",
                    )
                )
                if len(questions) >= target_count:
                    break
        return GenerateQuestionsResponse(
            topic_id=topic_id,
            topic_title=topic_title,
            questions=questions[:target_count],
            provider_used=v2.provider_used,
            model_used=v2.model_used,
            resolved_level=resolved_level,
        )

    async def generate_v2(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 5,
        requested_total_count: Optional[int] = None,
        existing_questions: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
        response_detail: str = "very_detailed",
        preferred_language: str = "",
        requires_programming: bool = False,
        prior_progress: str = "",
        additional_existing_questions: Optional[list[str]] = None,
        topic_track: str = "",
        learner_state: str = "",
    ) -> GenerateQuestionsV2Response:
        target_count = max(1, int(count or 1))
        problem_solving_mode = is_problem_solving_topic(topic_id)
        effective_level, resolved_level = self._resolve_generation_level(
            level=level, user_identity=user_identity, topic_id=topic_id
        )
        selected_language = _problem_solving_language(preferred_language) if problem_solving_mode else (
            (preferred_language or "").strip().lower()
        )
        retry_guidance = _question_retry_guidance(problem_solving_mode)
        mcp_context = await self._mcp_context_for_flow(
            flow="questions",
            query=f"{topic_title} {section_title or ''} interview questions and practical examples",
            topic_id=topic_id,
            topic_title=topic_title,
        )
        provider_used = ""
        model_used = ""
        retries_used = 0
        malformed_items_dropped = 0
        generation_version = ""
        raw_items: list[dict[str, Any]] = []
        existing_seed = _merge_existing_questions(
            existing_questions, additional_existing_questions
        )
        dedup_norm: set[str] = {_normalise_question(q) for q in existing_seed}
        seen_questions: list[str] = [*existing_seed]
        problem_pattern_signatures: set[str] = set()
        max_unique_problem_patterns = (
            _problem_solving_unique_pattern_target(target_count)
            if problem_solving_mode
            else 0
        )

        pass_modes: list[dict[str, Any]] = [
            {
                "section_title": section_title,
                "section_content": section_content,
                "difficulty": difficulty,
                "preferred_language": selected_language,
                "requires_programming": requires_programming,
            }
        ]
        if section_title and section_content:
            pass_modes.append(
                {
                    "section_title": None,
                    "section_content": None,
                    "difficulty": difficulty,
                    "preferred_language": selected_language,
                    "requires_programming": requires_programming,
                }
            )
        if not problem_solving_mode:
            pass_modes.append(
                {
                    "section_title": None,
                    "section_content": None,
                    "difficulty": None,
                    "preferred_language": "",
                    "requires_programming": False,
                }
            )

        for mode in pass_modes:
            if len(raw_items) >= target_count:
                break
            remaining = target_count - len(raw_items)
            existing_for_pass = [
                *existing_seed,
                *[str(item.get("question", "")) for item in raw_items],
            ]
            grounding_anchors = _grounding_anchor_phrases(
                topic_title=topic_title,
                section_title=mode["section_title"],
                content=mode["section_content"] or doc_content,
            )
            def validator(
                item,
                mode=mode,
                grounding_anchors=grounding_anchors,
            ):
                return _validate_question_item(
                    item,
                    topic_id,
                    mode["difficulty"],
                    effective_level,
                    preferred_language=str(mode["preferred_language"]),
                    requires_programming=bool(mode["requires_programming"]),
                    topic_title=topic_title,
                    grounding_anchors=grounding_anchors,
                    topic_content=str(mode["section_content"] or doc_content),
                )
            prompt = _build_prompt(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=remaining,
                requested_total_count=requested_total_count,
                existing_questions=existing_for_pass,
                additional_existing_questions=additional_existing_questions,
                prior_progress=prior_progress,
                difficulty=mode["difficulty"],
                level=effective_level,
                section_title=mode["section_title"],
                section_content=mode["section_content"],
                response_detail=response_detail,
                preferred_language=mode["preferred_language"],
                requires_programming=mode["requires_programming"],
                mcp_context=mcp_context,
                track=topic_track,
                learner_state=learner_state,
            )
            pass_items, pass_stats = await _collect_with_retries(
                llm=self.llm,
                base_prompt=prompt,
                llm_config=llm_config,
                user_identity=user_identity,
                target_count=remaining,
                validator=validator,
                existing_questions=existing_for_pass,
                system_prompt=_question_system_prompt(problem_solving_mode),
                hard_requirements=(
                    _problem_solving_required_markdown(selected_language)
                    if problem_solving_mode
                    else None
                ),
                retry_guidance=retry_guidance,
                prompt_name=(
                    "question_generation_problem_solving"
                    if problem_solving_mode
                    else "question_generation"
                ),
                prompt_version=default_registry.latest_version(
                    "question_generation_problem_solving"
                    if problem_solving_mode
                    else "question_generation"
                ),
                flow="questions",
                run_store=self.run_store,
            )
            retries_used += int(pass_stats.get("retries_used", 0))
            malformed_items_dropped += int(pass_stats.get("malformed_items_dropped", 0))
            pass_metadata = pass_stats.get("metadata", {}) or {}
            if pass_metadata.get("provider"):
                provider_used = str(pass_metadata.get("provider"))
            if pass_metadata.get("model"):
                model_used = str(pass_metadata.get("model"))
            pass_prompt_name = str(pass_stats.get("prompt_name", "") or "")
            pass_prompt_version = str(pass_stats.get("prompt_version", "") or "")
            if pass_prompt_name:
                generation_version = f"{pass_prompt_name}@{pass_prompt_version}"
            if str(pass_stats.get("terminal_reason") or ""):
                # A truncated response or an exhausted budget will not change on
                # the next pass, so stop here and fall back deterministically.
                break

            for item in pass_items:
                question_text = str(item.get("question", "")).strip()
                q_norm = _normalise_question(question_text)
                if not q_norm or q_norm in dedup_norm or _is_near_duplicate_question(question_text, seen_questions):
                    continue
                if problem_solving_mode:
                    signature = _problem_solving_pattern_signature(
                        question_text,
                        str(item.get("answer", "")),
                    )
                    if not _accept_problem_pattern_signature(
                        signature=signature,
                        seen_signatures=problem_pattern_signatures,
                        unique_target=max_unique_problem_patterns,
                    ):
                        continue
                dedup_norm.add(q_norm)
                seen_questions.append(question_text)
                raw_items.append(item)
                if len(raw_items) >= target_count:
                    break

        fallback_injected = False
        if len(raw_items) < target_count:
            missing_count = target_count - len(raw_items)
            fallback_items = _build_fallback_question_items(
                topic_id=topic_id,
                topic_title=topic_title,
                doc_content=doc_content,
                count=_problem_solving_fallback_count_with_headroom(
                    missing_count,
                    problem_solving_mode,
                ),
                difficulty=difficulty,
                level=level,
                section_title=None,
                section_content=None,
                existing_questions=[
                    *existing_seed,
                    *[str(item.get("question", "")) for item in raw_items],
                ],
                preferred_language=selected_language,
                requires_programming=requires_programming,
            )
            for item in fallback_items:
                question_text = str(item.get("question", "")).strip()
                q_norm = _normalise_question(question_text)
                if not q_norm or q_norm in dedup_norm or _is_near_duplicate_question(question_text, seen_questions):
                    continue
                if problem_solving_mode:
                    signature = _problem_solving_pattern_signature(
                        question_text,
                        str(item.get("answer", "")),
                    )
                    if not _accept_problem_pattern_signature(
                        signature=signature,
                        seen_signatures=problem_pattern_signatures,
                        unique_target=max_unique_problem_patterns,
                    ):
                        continue
                dedup_norm.add(q_norm)
                seen_questions.append(question_text)
                raw_items.append(item)
                fallback_injected = True
                if len(raw_items) >= target_count:
                    break

        questions = [self._to_question_answer_v2(topic_id, item) for item in raw_items[:target_count]]

        return GenerateQuestionsV2Response(
            topic_id=topic_id,
            topic_title=topic_title,
            questions=questions,
            provider_used=provider_used,
            model_used=model_used,
            retries_used=retries_used,
            malformed_items_dropped=malformed_items_dropped,
            degraded=fallback_injected,
            requested_count=target_count,
            returned_count=len(questions),
            generation_version=generation_version,
            resolved_level=resolved_level,
        )

    async def generate_quiz_v2_stream(
        self,
        topics_content: list[dict],
        count: int = 10,
        question_types: list[str] | None = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        response_detail: str | None = None,
        preferred_language: str = "",
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        prior_progress: str = "",
        additional_existing_questions: Optional[list[str]] = None,
        target_objectives: list[str] | None = None,
        learner_state: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        target_count = max(1, int(count or 1))
        effective_detail = "very_detailed" if response_detail == "very_detailed" else "concise"
        mcp_context = await self._mcp_context_for_flow(
            flow="quiz",
            query=" ".join(
                [
                    " ".join(str(tc.get("title", "")).strip() for tc in topics_content[:4]),
                    "technical quiz generation",
                ]
            ).strip(),
        )
        allowed_topics = {str(tc["id"]) for tc in topics_content}
        allowed_types = set(question_types or ["mcq", "true_false"])
        if not allowed_types:
            allowed_types = {"mcq", "true_false"}
        # A quiz spans topics but runs at one level, so the history of
        # the first requested topic picks the level.
        quiz_topic_id = str(topics_content[0].get("id", "")) if topics_content else ""
        effective_level, resolved_level = self._resolve_generation_level(
            level=level, user_identity=user_identity, topic_id=quiz_topic_id
        )

        dedup_norm: set[str] = set()
        generated_question_texts: list[str] = _merge_existing_questions(
            None, additional_existing_questions
        )
        dedup_norm.update(_normalise_question(q) for q in generated_question_texts)
        provider_used = ""
        model_used = ""
        retries_used = 0
        malformed_items_dropped = 0
        generated_count = 0

        async def _generate_one() -> dict[str, Any]:
            local_retries = 0
            local_attempts = 0
            local_malformed = 0
            metadata: dict[str, Any] = {}
            issue_counter: Counter[str] = Counter()
            prompt = _build_quiz_prompt(
                topics_content,
                1,
                question_types,
                difficulty,
                effective_level,
                response_detail=effective_detail,
                preferred_language=(preferred_language or "").strip().lower(),
                additional_existing_questions=additional_existing_questions,
                prior_progress=prior_progress,
                mcp_context=mcp_context,
                target_objectives=target_objectives,
                learner_state=learner_state,
            )
            base_prompt = prompt
            max_tokens_cap = _questions_max_tokens_cap(1)
            terminal_reason = ""
            try:
                for _ in range(_attempt_budget(1)):
                    local_attempts += 1
                    result = await self.llm.completion(
                        prompt,
                        llm_config,
                        user_identity=user_identity,
                        **_supported_options(
                            self.llm,
                            {
                                "task": "final",
                                "system": _QUIZ_SYSTEM_PROMPT,
                                "structured": True,
                                "max_tokens_cap": max_tokens_cap,
                                "attempts": local_attempts,
                                "terminal_reason": terminal_reason,
                                "prompt_name": "quiz_generation",
                                "prompt_version": default_registry.latest_version(
                                    "quiz_generation"
                                ),
                            },
                        ),
                    )
                    raise_if_policy_blocked_result(result)
                    metadata = result.get("metadata", {})
                    if not result.get("success"):
                        if str(result.get("error_code") or "") in TERMINAL_ERROR_CODES:
                            terminal_reason = str(result.get("error_code") or "")
                            break
                        local_retries += 1
                        issue_counter["transport_or_provider_error"] += 1
                        prompt = _build_retry_prompt(
                            base_prompt=base_prompt,
                            missing_count=1,
                            issues="transport_or_provider_error",
                            existing_questions=generated_question_texts,
                        )
                        continue

                    parsed = _parse_questions_json(result.get("analysis", ""))
                    if not parsed:
                        local_retries += 1
                        local_malformed += 1
                        issue_counter["json_parse_failed"] += 1
                        prompt = _build_retry_prompt(
                            base_prompt=base_prompt,
                            missing_count=1,
                            issues="json_parse_failed",
                            existing_questions=generated_question_texts,
                        )
                        continue

                    selected_item: dict[str, Any] | None = None
                    for item in parsed:
                        if not isinstance(item, dict):
                            local_malformed += 1
                            issue_counter["non_dict_item"] += 1
                            continue
                        is_valid, issue = _validate_quiz_item(
                            item,
                            allowed_topics,
                            allowed_types,
                            difficulty,
                            effective_level,
                            topic_content=_topic_content_for(item, topics_content),
                        )
                        if not is_valid:
                            local_malformed += 1
                            issue_counter[issue] += 1
                            continue
                        item = _normalise_answer_position(item)
                        question_text = str(item.get("question", "")).strip()
                        q_norm = _normalise_question(question_text)
                        if (
                            not q_norm
                            or q_norm in dedup_norm
                            or _is_near_duplicate_question(question_text, generated_question_texts)
                        ):
                            local_malformed += 1
                            issue_counter["duplicate_question"] += 1
                            continue
                        dedup_norm.add(q_norm)
                        generated_question_texts.append(question_text)
                        selected_item = item
                        break

                    if selected_item is not None:
                        question = self._to_quiz_question_v2(selected_item)
                        return {
                            "question": question.model_dump(),
                            "metadata": metadata,
                            "retries_used": local_retries,
                            "malformed_items_dropped": local_malformed,
                        }

                    local_retries += 1
                    top_issues = (
                        ", ".join(f"{k}:{v}" for k, v in issue_counter.most_common(5))
                        or "insufficient_valid_items"
                    )
                    prompt = _build_retry_prompt(
                        base_prompt=base_prompt,
                        missing_count=1,
                        issues=top_issues,
                        existing_questions=generated_question_texts,
                    )

                return {
                    "question": None,
                    "metadata": metadata,
                    "retries_used": local_retries,
                    "malformed_items_dropped": local_malformed,
                    "terminal_reason": terminal_reason,
                }
            except (
                LLMServiceApprovalRequiredError,
                StudyAppLLMNotAssignedError,
                PersonalCredentialRequiredError,
            ) as exc:
                code, message = self._policy_error_payload(exc)
                return {
                    "error": {"code": code, "message": message},
                    "metadata": metadata,
                    "retries_used": local_retries,
                    "malformed_items_dropped": local_malformed,
                    "terminal_reason": "",
                }
            except Exception as exc:
                logger.exception("Streaming quiz generation failed: %s", exc)
                return {
                    "error": {
                        "code": "generation_failed",
                        "message": str(exc).strip() or "Quiz generation failed",
                    },
                    "metadata": metadata,
                    "retries_used": local_retries,
                    "malformed_items_dropped": local_malformed,
                }

        yield {
            "type": "start",
            "target_count": target_count,
            "topics_used": sorted(allowed_topics),
            "question_types": sorted(allowed_types),
        }

        terminal_reason = ""
        while generated_count < target_count:
            yield {
                "type": "progress",
                "stage": "generating",
                "message": f"Generating quiz question {generated_count + 1} of {target_count}.",
                "generated_count": generated_count,
                "target_count": target_count,
            }
            result = await _generate_one()
            retries_used += int(result.get("retries_used", 0))
            malformed_items_dropped += int(result.get("malformed_items_dropped", 0))
            metadata = result.get("metadata", {}) or {}
            if metadata.get("provider"):
                provider_used = str(metadata.get("provider"))
            if metadata.get("model"):
                model_used = str(metadata.get("model"))
            terminal_reason = str(result.get("terminal_reason") or "")
            if terminal_reason:
                # A truncated response or an exhausted budget will not change on
                # the next slot, so stop and fall back deterministically.
                break

            error = result.get("error")
            if error:
                yield {
                    "type": "error",
                    "code": str(error.get("code", "generation_failed")),
                    "message": str(error.get("message", "Quiz generation failed")),
                }
                return

            question = result.get("question")
            if not isinstance(question, dict):
                break
            generated_count += 1
            yield {"type": "question", "question": question}

        if generated_count < target_count and not terminal_reason:
            for _ in range(_recovery_budget(target_count - generated_count)):
                if generated_count >= target_count:
                    break
                yield {
                    "type": "progress",
                    "stage": "recovering",
                    "message": (
                        f"Recovering missing quiz questions: {generated_count}/{target_count} generated."
                    ),
                    "generated_count": generated_count,
                    "target_count": target_count,
                }
                result = await _generate_one()
                retries_used += int(result.get("retries_used", 0))
                malformed_items_dropped += int(result.get("malformed_items_dropped", 0))
                metadata = result.get("metadata", {}) or {}
                if metadata.get("provider"):
                    provider_used = str(metadata.get("provider"))
                if metadata.get("model"):
                    model_used = str(metadata.get("model"))
                error = result.get("error")
                if error:
                    yield {
                        "type": "error",
                        "code": str(error.get("code", "generation_failed")),
                        "message": str(error.get("message", "Quiz generation failed")),
                    }
                    return
                question = result.get("question")
                if not isinstance(question, dict):
                    continue
                generated_count += 1
                yield {"type": "question", "question": question.model_dump()}

        yield {
            "type": "done",
            "generated_count": generated_count,
            "target_count": target_count,
            "topics_used": sorted(allowed_topics),
            "provider_used": provider_used,
            "model_used": model_used,
            "retries_used": retries_used,
            "malformed_items_dropped": malformed_items_dropped,
            "degraded": generated_count < target_count,
            "requested_count": target_count,
            "returned_count": generated_count,
            "resolved_level": resolved_level,
        }

    async def generate_quiz(
        self,
        topics_content: list[dict],
        count: int = 10,
        question_types: list[str] | None = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        response_detail: str | None = None,
        preferred_language: str = "",
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        prior_progress: str = "",
        additional_existing_questions: Optional[list[str]] = None,
        target_objectives: list[str] | None = None,
        learner_state: str = "",
    ) -> GenerateQuizResponse:
        # A quiz spans topics but runs at one level, so the history of
        # the first requested topic picks the level.
        quiz_topic_id = str(topics_content[0].get("id", "")) if topics_content else ""
        effective_level, resolved_level = self._resolve_generation_level(
            level=level, user_identity=user_identity, topic_id=quiz_topic_id
        )
        v2 = await self.generate_quiz_v2(
            topics_content=topics_content,
            count=count,
            question_types=question_types,
            difficulty=difficulty,
            level=effective_level,
            response_detail=response_detail,
            preferred_language=preferred_language,
            llm_config=llm_config,
            user_identity=user_identity,
            prior_progress=prior_progress,
            additional_existing_questions=additional_existing_questions,
            target_objectives=target_objectives,
            learner_state=learner_state,
        )
        legacy_questions = [
            QuizQuestion(
                question=q.question,
                type=q.type,
                choices=q.choices,
                correct_answer=q.correct_answer,
                explanation=q.explanation,
                difficulty=q.difficulty,
                topic_id=q.topic_id,
            )
            for q in v2.questions
        ]
        return GenerateQuizResponse(
            questions=legacy_questions,
            topics_used=v2.topics_used,
            provider_used=v2.provider_used,
            model_used=v2.model_used,
            resolved_level=resolved_level,
        )

    async def generate_quiz_v2(
        self,
        topics_content: list[dict],
        count: int = 10,
        question_types: list[str] | None = None,
        difficulty: Optional[str] = None,
        level: Optional[str] = None,
        response_detail: str | None = None,
        preferred_language: str = "",
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        prior_progress: str = "",
        additional_existing_questions: Optional[list[str]] = None,
        target_objectives: list[str] | None = None,
        learner_state: str = "",
    ) -> GenerateQuizV2Response:
        effective_detail = "very_detailed" if response_detail == "very_detailed" else "concise"
        # A quiz spans topics but runs at one level, so the history of
        # the first requested topic picks the level.
        quiz_topic_id = str(topics_content[0].get("id", "")) if topics_content else ""
        effective_level, resolved_level = self._resolve_generation_level(
            level=level, user_identity=user_identity, topic_id=quiz_topic_id
        )
        mcp_context = await self._mcp_context_for_flow(
            flow="quiz",
            query=" ".join(
                [
                    " ".join(str(tc.get("title", "")).strip() for tc in topics_content[:4]),
                    "technical quiz generation",
                ]
            ).strip(),
        )
        prompt = _build_quiz_prompt(
            topics_content,
            count,
            question_types,
            difficulty,
            effective_level,
            response_detail=effective_detail,
            preferred_language=(preferred_language or "").strip().lower(),
            additional_existing_questions=additional_existing_questions,
            prior_progress=prior_progress,
            mcp_context=mcp_context,
            target_objectives=target_objectives,
            learner_state=learner_state,
        )
        allowed_topics = {str(tc["id"]) for tc in topics_content}
        allowed_types = set(question_types or ["mcq", "true_false"])
        def validator(item):
            return _validate_quiz_item(
                item,
                allowed_topics,
                allowed_types,
                difficulty,
                effective_level,
                topic_content=_topic_content_for(item, topics_content),
            )

        raw_items, stats = await _collect_with_retries(
            llm=self.llm,
            base_prompt=prompt,
            llm_config=llm_config,
            user_identity=user_identity,
            target_count=count,
            validator=validator,
            existing_questions=additional_existing_questions,
            prompt_name="quiz_generation",
            prompt_version=default_registry.latest_version("quiz_generation"),
            flow="quiz",
            run_store=self.run_store,
        )

        questions = []
        for item in raw_items:
            item = _normalise_answer_position(item)
            topic_id = str(item.get("topic_id", "")).strip()
            qid = _question_id(topic_id or "quiz", item["question"])
            choices = [QuizChoice(label=c["label"], text=c["text"]) for c in item["choices"]]
            questions.append(
                QuizQuestionV2(
                    question_id=qid,
                    question=item["question"].strip(),
                    type=QuizQuestionType(item["type"]),
                    choices=choices,
                    correct_answer=str(item["correct_answer"]).strip(),
                    explanation=format_markdown_readable(item["explanation"].strip()),
                    difficulty=item["difficulty"],
                    topic_id=topic_id,
                    source_quote=item["source_quote"].strip(),
                    reasoning_summary=item["reasoning_summary"].strip(),
                    learning_objective=str(item.get("learning_objective", "")).strip(),
                    expected_answer=str(item.get("expected_answer", "")).strip(),
                    distractor_rationales=[
                        QuizDistractorRationale(
                            label=str(rationale.get("label", "")).strip(),
                            rationale=str(rationale.get("rationale", "")).strip(),
                        )
                        for rationale in item.get("distractor_rationales", [])
                        if isinstance(rationale, dict)
                    ],
                )
            )

        metadata = stats.get("metadata", {})
        topic_ids = sorted({tc["id"] for tc in topics_content})
        prompt_name = str(stats.get("prompt_name", "") or "")
        prompt_version = str(stats.get("prompt_version", "") or "")
        generation_version = f"{prompt_name}@{prompt_version}" if prompt_name else ""
        return GenerateQuizV2Response(
            questions=questions,
            topics_used=topic_ids,
            provider_used=metadata.get("provider", ""),
            model_used=metadata.get("model", ""),
            retries_used=stats.get("retries_used", 0),
            malformed_items_dropped=stats.get("malformed_items_dropped", 0),
            degraded=len(questions) < count,
            requested_count=count,
            returned_count=len(questions),
            generation_version=generation_version,
            resolved_level=resolved_level,
        )

    async def generate_cards(
        self,
        topic_id: str,
        topic_title: str,
        doc_content: str,
        count: int = 10,
        section_title: Optional[str] = None,
        section_content: Optional[str] = None,
        response_detail: str = "very_detailed",
        existing_cards: Optional[list[str]] = None,
        llm_config: Optional[LLMConfigRequest] = None,
        user_identity: Optional[dict] = None,
        level: Optional[str] = None,
        topic_track: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Generate flashcards ``{front, back, source_quote, source_section}``.

        Reuses the v2 question machinery: the same
        grounding-anchor validation, the same
        ``_collect_with_retries`` retry/repair loop, and
        the same ``LLMClient`` budget policy. ``count`` is
        capped at ``MAX_CARD_GENERATION_COUNT``.
        """
        target_count = max(1, min(int(count or 10), MAX_CARD_GENERATION_COUNT))
        effective_level, resolved_level = self._resolve_generation_level(
            level=level, user_identity=user_identity, topic_id=topic_id
        )
        mcp_context = await self._mcp_context_for_flow(
            flow="questions",
            query=f"{topic_title} {section_title or ''} flashcards",
            topic_id=topic_id,
            topic_title=topic_title,
        )
        grounding_anchors = _grounding_anchor_phrases(
            topic_title=topic_title,
            section_title=section_title,
            content=section_content if (section_title and section_content) else doc_content,
        )
        prompt = _build_card_prompt(
            topic_id=topic_id,
            topic_title=topic_title,
            doc_content=doc_content,
            count=target_count,
            section_title=section_title,
            section_content=section_content,
            response_detail=response_detail,
            existing_cards=existing_cards,
            mcp_context=mcp_context,
            level=effective_level,
            track=topic_track,
        )

        def validator(item: dict[str, Any]) -> tuple[bool, str]:
            return _validate_card_item(
                item,
                topic_title=topic_title,
                grounding_anchors=grounding_anchors,
            )

        raw_items, stats = await _collect_with_retries(
            llm=self.llm,
            base_prompt=prompt,
            llm_config=llm_config,
            user_identity=user_identity,
            target_count=target_count,
            validator=validator,
            existing_questions=existing_cards,
            system_prompt=_CARD_SYSTEM_PROMPT,
            prompt_name="card_generation",
            prompt_version=default_registry.latest_version("card_generation"),
            flow="cards",
            run_store=self.run_store,
        )

        cards: list[dict[str, Any]] = []
        seen_fronts: set[str] = set()
        for item in raw_items:
            front = str(item.get("front", "")).strip()
            norm = _normalise_question(front)
            if not norm or norm in seen_fronts:
                continue
            seen_fronts.add(norm)
            cards.append(
                {
                    "card_id": _card_id(topic_id, front),
                    "front": front,
                    "back": str(item.get("back", "")).strip(),
                    "source_section": str(
                        item.get("source_section", "")
                    ).strip(),
                    "source_quote": str(
                        item.get("source_quote", "")
                    ).strip(),
                }
            )
        # Same contract as the question/quiz flows: the level these cards were
        # auto-resolved to, empty when the caller pinned its own.
        return cards[:target_count], {**stats, "resolved_level": resolved_level}


# ── Flashcard generation (STUDY_ENABLE_FLASHCARDS_V1) ──

#: Hard cap on cards generated per request.
MAX_CARD_GENERATION_COUNT = 50

#: The flashcard-generation system prompt contract, migrated verbatim
#: into the prompt registry as ``card_generation@1``.
_CARD_SYSTEM_PROMPT = default_registry.get("card_generation").system_prompt


def _build_card_prompt(
    topic_id: str,
    topic_title: str,
    doc_content: str,
    count: int = 10,
    section_title: Optional[str] = None,
    section_content: Optional[str] = None,
    response_detail: str = "very_detailed",
    existing_cards: Optional[list[str]] = None,
    mcp_context: str = "",
    level: Optional[str] = None,
    track: str = "",
) -> str:
    """Build the flashcard generation user turn."""
    skill_section = _skill_context_section(
        _normalise_level(level),
        "general",
        _skill_track_for_topic(topic_id, track),
    )
    if section_title and section_content:
        scope = f'the "{section_title}" section of "{topic_title}"'
        content = _clamp_content(section_content)
    else:
        scope = f'"{topic_title}"'
        content = _clamp_content(doc_content)
    grounding_anchors = _grounding_anchor_phrases(
        topic_title=topic_title,
        section_title=section_title,
        content=section_content if (section_title and section_content) else doc_content,
    )
    grounding_block = _render_grounding_anchors_block(grounding_anchors)
    detail_clause = (
        "Keep back answers concise but complete: the direct answer first, "
        "then the key mechanism, tradeoff, or failure mode an interviewer "
        "would probe."
        if response_detail != "very_detailed"
        else "Be maximally detailed on the back: the direct answer first, "
        "then layered explanation, concrete examples, edge cases, and the "
        "tradeoff or failure mode an interviewer would probe."
    )
    uniqueness_block = ""
    seed = _normalise_existing_questions(existing_cards)
    if seed:
        existing_blob = "\n".join([f"- {q}" for q in seed[:60]])
        uniqueness_block = (
            "\nAlready generated cards for this topic. Do NOT repeat or "
            f"rephrase these:\n{existing_blob}\n"
        )
    mcp_block = untrusted_block("external_context", mcp_context, 2500)
    untrusted_blocks = "\n\n".join(
        block
        for block in (
            untrusted_block("topic_title", topic_title),
            untrusted_block("section_title", section_title or ""),
            untrusted_block(
                "section_content" if (section_title and section_content) else "documentation",
                content,
                _MAX_DOC_CONTEXT,
            ),
        )
        if block
    )
    prompt_contract = render_contract(
        schema_label="Return ONLY valid JSON in this shape",
        schema_block="""[
  {
    "front": "Concise flashcard question",
    "back": "Direct, detailed answer",
    "source_section": "Section name",
    "source_quote": "short direct quote from course content"
  }
]""",
        rules=[
            f"Return exactly {count} items; never return fewer.",
            "Front text must be a self-contained question that names the topic concept.",
            "Back text must directly answer the front; no meta-guidance about how to answer.",
            "Each card must be explicitly grounded to the current topic, section, or one of the grounding anchors below.",
            "Do not ask generic questions that could fit many unrelated topics with only minor wording changes.",
            "Front and back must stay aligned to the same concept.",
            detail_clause,
            "source_quote must be factual text from the provided course content.",
            "Because output must be valid JSON, escape newlines, quotes, and backslashes correctly inside string values.",
            "Do not wrap JSON with prose; return raw JSON only.",
            UNTRUSTED_CLAUSE,
        ],
    )
    return f"""Generate exactly {count} interview-grade flashcards for {scope}.
{uniqueness_block}
{grounding_block}

{skill_section}

{prompt_contract}

{untrusted_blocks}{mcp_block}"""


def _validate_card_item(
    item: dict[str, Any],
    *,
    topic_title: str,
    grounding_anchors: list[str],
) -> tuple[bool, str]:
    """Validate one generated flashcard item.

    Mirrors ``front`` into ``question`` so the shared
    retry machinery's near-duplicate detection works
    unchanged on the card front text.
    """
    front = str(item.get("front", "")).strip()
    back = str(item.get("back", "")).strip()
    if not front:
        return False, "missing_or_empty_front"
    if len(front) < 8:
        return False, "front_too_short"
    if not back:
        return False, "missing_or_empty_back"
    if len(back) < 15:
        return False, "back_too_short"
    source_quote = str(item.get("source_quote", "")).strip()
    if not source_quote:
        return False, "missing_source_quote"
    source_section = str(item.get("source_section", "")).strip()
    if not source_section:
        return False, "missing_source_section"
    item["question"] = front
    item["front"] = front
    item["back"] = back
    item["source_quote"] = source_quote
    item["source_section"] = source_section
    if topic_title or grounding_anchors:
        if not _question_is_grounded_to_anchors(
            front,
            topic_title=topic_title,
            anchors=grounding_anchors,
            problem_solving_mode=False,
        ):
            return False, "card_not_grounded_to_topic"
    return True, ""


def _card_id(topic_id: str, front: str) -> str:
    digest = hashlib.sha1(
        f"card:{topic_id}:{front.strip().lower()}".encode("utf-8")
    ).hexdigest()
    return f"{topic_id}:{digest[:14]}"
