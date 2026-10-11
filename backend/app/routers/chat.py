"""Chat router — interactive follow-up conversations on highlighted words."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    ChatAskRequest,
    ChatAskResponse,
    ChatFollowUpRequest,
    ChatFollowUpResponse,
    ChatHighlightActionRequest,
    ChatHighlightActionResponse,
    HighlightActionEnum,
)
from app.services import fsrs_scheduler
from app.services.card_store import DECK_NAME_MAX, CardStore
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore
from app.services.learner_memory import build_learner_state_block
from app.services.llm_client import BUDGET_EXCEEDED_CODE, LLMClient
from app.services.llm_policy import raise_if_policy_blocked_result
from app.services.llm_skills import normalise_track, skill_context_block
from app.services.markdown_formatter import format_markdown_readable
from app.services.mcp_gateway import MCPGateway
from app.services.prompt_blocks import (
    UNTRUSTED_CLAUSE,
    render_contract,
    untrusted_block,
)
from app.services.prompt_registry import default_registry

router = APIRouter(prefix="/api/chat", tags=["chat"])

_llm_client: LLMClient | None = None
_parser: DocParser | None = None
_learning_store: LearningStore | None = None
_mcp_gateway: MCPGateway | None = None
_card_store: CardStore | None = None

# ── Prompt bounds ────────────────────────────────────────────────
# Every externally sourced string is bounded before it is fenced, so a single
# request cannot inflate the prompt beyond the caps below. The Pydantic
# `max_length` values requested for `user_message` / `ChatMessage.content`
# match these so a schema-level rejection and a router-level truncation agree.
MAX_USER_MESSAGE_CHARS = 4000
MAX_WORD_CHARS = 200
MAX_CONTEXT_QUESTION_CHARS = 2000
MAX_CONTEXT_ANSWER_CHARS = 6000
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_MESSAGE_CHARS = 2000
MAX_HISTORY_CHARS = 12000
MAX_TOPIC_METADATA_CHARS = 1200
MAX_MEMORY_PROMPT_CHARS = 900
MAX_MCP_CONTEXT_CHARS = 2500
#: Bound on the fenced learner-state block. The learner's weak areas,
#: objectives, error patterns, and interview weaknesses are gathered by
#: learner_memory, which never raises: an empty block leaves this prompt
#: byte-identical to one built without it.
MAX_LEARNER_STATE_CHARS = 1200
#: Absolute pre-fence ceiling, well above every per-field cap, so one pathological
#: payload cannot cost unbounded string work before truncation.
HARD_INPUT_GUARD_CHARS = 200_000

#: Per-flow output ceiling. Chat is a free-form markdown flow, so
#: the cap is a ceiling on the answer, not on a JSON document.
CHAT_MAX_TOKENS_CAP = 1200
#: Bound on the handbook-section name carried by a highlighted Q&A.
MAX_SOURCE_SECTION_CHARS = 200
#: Bound on the one verbatim quote the explain path cites.
MAX_SOURCE_QUOTE_CHARS = 400
#: Bound on the injected skill taxonomies. The taxonomy is
#: module-authored (trusted), so it is capped rather than fenced,
#: exactly like the generation flows that inject it.
MAX_SKILL_CONTEXT_CHARS = 4000
#: Deck that gathers flashcards created from highlighted terms when the
#: caller does not name one.
HIGHLIGHT_DECK_FALLBACK_NAME = "Highlights"
#: Card-id prefix for review items scheduled from a highlighted term, so
#: the FSRS review queue names the term instead of a bare uuid.
HIGHLIGHT_REVIEW_CARD_PREFIX = "highlight:"
#: Bound on the generated FSRS card id for a highlighted review item.
HIGHLIGHT_REVIEW_CARD_ID_MAX = 120

CHAT_SYSTEM_PROMPT = (
    "You are a helpful study assistant. The user is studying technical "
    "documentation and has a follow-up question about a highlighted term. "
    f"{UNTRUSTED_CLAUSE}"
)

#: Persona for the highlight ``explain`` action: same assistant voice as
#: the follow-up flow, addressed to a highlighted term in one Q&A card.
HIGHLIGHT_EXPLAIN_SYSTEM_PROMPT = (
    "You are a helpful study assistant. The user highlighted a term in a "
    "technical Q&A and wants it explained in that context. "
    f"{UNTRUSTED_CLAUSE}"
)

#: Persona for the highlight ``interview_angle`` action: the registered
#: interview-coach template, with its JSON-only output contract overridden
#: for a markdown answer.
HIGHLIGHT_INTERVIEW_SYSTEM_PROMPT = (
    f"{default_registry.get('interview_coach').system_prompt}\n\n"
    "This flow overrides the output-format sentence above: answer as one "
    "markdown response, never as JSON. "
    f"{UNTRUSTED_CLAUSE}"
)


def _fence(label: str, value: str, max_chars: int) -> str:
    """De-fang and fence one externally sourced value.

    ``untrusted_block`` owns the sanitiser (delimiter stripping plus injection
    marker neutralisation) so it cannot drift between flows. The value is only
    clipped at ``HARD_INPUT_GUARD_CHARS`` here: clipping at the per-field cap is
    left to ``untrusted_block`` so the prompt still carries the
    "[truncated at N characters]" marker the model needs to know the context is
    partial.
    """
    return untrusted_block(label, str(value or "").strip()[:HARD_INPUT_GUARD_CHARS], max_chars)


def _conversation_id(body: ChatFollowUpRequest) -> str:
    raw = (body.conversation_id or "").strip()
    if raw:
        return raw[:120]
    fallback = f"{body.topic_id}:{body.section_title}:{body.word}".strip(":")
    return (fallback or "chat").lower()[:120]


def _memory_prompt_block(memory: dict | None) -> str:
    if not isinstance(memory, dict):
        return ""
    summary = memory.get("summary")
    if not isinstance(summary, dict):
        return ""
    return _fence(
        "Conversation memory (assistant summary from prior turns; use as soft context)",
        str(summary.get("summary", "")),
        MAX_MEMORY_PROMPT_CHARS,
    )


def _learner_state_prompt_block(user_id: str, topic_id: str, track: str) -> str:
    """Render the learner-state block, or "" when there is nothing to show.

    ``build_learner_state_block`` gathers and fences the block itself, so
    the only job here is the separating newline: an empty learner keeps
    this prompt byte-identical to one built without it.
    """
    learner_state = build_learner_state_block(
        user_id,
        topic_id=topic_id,
        track=track,
        max_chars=MAX_LEARNER_STATE_CHARS,
    )
    if not learner_state:
        return ""
    return f"\n{learner_state}"


def _history_block(history: list | None) -> str:
    """Fence the recent conversation turns as one untrusted block."""
    if not history:
        return ""
    lines: list[str] = []
    for msg in list(history)[-MAX_HISTORY_MESSAGES:]:
        role_label = "User" if getattr(msg, "role", "") == "user" else "Assistant"
        content = str(getattr(msg, "content", "") or "").strip()[:MAX_HISTORY_MESSAGE_CHARS]
        if not content:
            continue
        lines.append(f"{role_label}: {content}")
    if not lines:
        return ""
    return _fence("Conversation history", "\n".join(lines), MAX_HISTORY_CHARS)


def _topic_metadata_block(
    *,
    topic_id: str,
    topic_title: str,
    topic_track: str,
    section_title: str,
    mode: str,
) -> str:
    """Fence the topic metadata the answer is grounded in."""
    lines: list[str] = []
    if topic_id:
        lines.append(f"- Topic ID: {topic_id}")
    if topic_title:
        lines.append(f"- Topic Title: {topic_title}")
    if topic_track:
        lines.append(f"- Topic Track: {topic_track}")
    if section_title:
        lines.append(f"- Section: {section_title}")
    if mode:
        lines.append(f"- Learning Mode: {mode}")
    return _fence(
        "Topic metadata",
        "\n".join(lines) if lines else "- Topic metadata not provided.",
        MAX_TOPIC_METADATA_CHARS,
    )


async def _resolve_topic_ai_settings(
    user: dict,
    *,
    topic_id: str,
    response_detail,
    preferred_language,
    requires_programming,
) -> tuple[str, str, bool]:
    """Resolve response detail, language, and code rules for a topic.

    An explicit request value always wins; otherwise the topic's stored
    AI settings decide. Shared by the follow-up and highlight flows so
    both answers honour the same per-topic preferences.
    """
    resolved_detail = response_detail.value if response_detail else "concise"
    resolved_language = (preferred_language or "").strip().lower()
    resolved_requires_programming = bool(requires_programming)

    if topic_id and _learning_store is not None:
        topic_detail = None
        if _parser is not None:
            static = _parser.get_topic(topic_id)
            if static:
                topic_detail = static.model_dump(mode="json")
        if topic_detail is None:
            custom = await _learning_store.run_async(
                _learning_store.get_custom_topic,
                user_id=user["user"],
                topic_id=topic_id,
            )
            if custom:
                topic_detail = custom
        if topic_detail is not None:
            resolved = await _learning_store.run_async(
                _learning_store.resolve_topic_ai_settings,
                user_id=user["user"],
                topic_id=topic_id,
                topic_detail=topic_detail,
            )
            if response_detail is None:
                resolved_detail = resolved.get("response_detail", "concise")
            if preferred_language is None:
                resolved_language = resolved.get("preferred_language", "")
            if requires_programming is None:
                resolved_requires_programming = bool(
                    resolved.get("requires_programming", False)
                )

    return resolved_detail, resolved_language, resolved_requires_programming


async def _conversation_blocks(
    user: dict,
    *,
    topic_id: str,
    topic_track: str,
    conversation_id: str,
    use_memory: bool,
) -> tuple[str, str]:
    """Render the per-conversation memory and cross-session learner state."""
    memory_block = ""
    if bool(use_memory) and _learning_store is not None:
        memory = await _learning_store.run_async(
            _learning_store.get_assistant_memory,
            user_id=user["user"],
            conversation_id=conversation_id,
            flow="chat",
        )
        memory_block = _memory_prompt_block(memory)

    learner_state_block = ""
    if bool(use_memory):
        # Independent of the conversation memory: this is the learner's
        # cross-session weak areas, objectives, error patterns, and
        # interview weaknesses.
        learner_state_block = _learner_state_prompt_block(
            user["user"], topic_id, topic_track
        )
    return memory_block, learner_state_block


async def _persist_conversation_memory(
    user: dict,
    *,
    conversation_id: str,
    topic_title: str,
    topic_id: str,
    word: str,
    user_message: str,
    reply: str,
    updated_by: str,
) -> None:
    """Fold one turn into the conversation's assistant-memory digest."""
    previous = (
        await _learning_store.run_async(
            _learning_store.get_assistant_memory,
            user_id=user["user"],
            conversation_id=conversation_id,
            flow="chat",
        )
        or {}
    )
    prev_summary = ""
    if isinstance(previous.get("summary"), dict):
        prev_summary = str(previous["summary"].get("summary", "")).strip()
    merged_summary = (
        f"{prev_summary}\n"
        f"Topic: {topic_title or topic_id or 'unknown'} | Word: {word}\n"
        f"User asked: {user_message[:240]}\n"
        f"Assistant: {reply[:380]}"
    ).strip()
    # One shared bound for every stored memory digest.
    memory_cap = max(
        MAX_MEMORY_PROMPT_CHARS,
        int(getattr(get_settings(), "memory_summary_max_chars", 1200) or 1200),
    )
    await _learning_store.run_async(
        _learning_store.upsert_assistant_memory,
        user_id=user["user"],
        conversation_id=conversation_id,
        flow="chat",
        summary={
            "summary": merged_summary[:memory_cap],
            "topic_id": topic_id,
            "word": word,
            "updated_by": updated_by,
        },
    )


def init(
    llm_client: LLMClient,
    parser: DocParser | None = None,
    learning_store: LearningStore | None = None,
    mcp_gateway: MCPGateway | None = None,
    card_store: CardStore | None = None,
):
    global _llm_client, _parser, _learning_store, _mcp_gateway, _card_store
    _llm_client = llm_client
    _parser = parser
    _learning_store = learning_store
    _mcp_gateway = mcp_gateway
    _card_store = card_store


@router.post("/follow-up", response_model=ChatFollowUpResponse)
async def chat_follow_up(
    body: ChatFollowUpRequest,
    user: dict = Depends(require_auth),
):
    """Answer a follow-up question about a highlighted word/phrase in a Q&A."""
    if _llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    conv_id = _conversation_id(body)
    resolved_detail, resolved_language, resolved_requires_programming = (
        await _resolve_topic_ai_settings(
            user,
            topic_id=body.topic_id,
            response_detail=body.response_detail,
            preferred_language=body.preferred_language,
            requires_programming=body.requires_programming,
        )
    )
    memory_block, learner_state_block = await _conversation_blocks(
        user,
        topic_id=body.topic_id,
        topic_track=body.topic_track,
        conversation_id=conv_id,
        use_memory=bool(body.use_memory),
    )

    # Build conversation history for context
    history_block = _history_block(body.history)

    topic_block = _topic_metadata_block(
        topic_id=body.topic_id,
        topic_title=body.topic_title,
        topic_track=body.topic_track,
        section_title=body.section_title,
        mode=body.mode,
    )
    mcp_context = ""
    if _mcp_gateway is not None:
        mcp_context = await _mcp_gateway.gather_context(
            flow="chat",
            query=f"{body.topic_title or body.topic_id} {body.word} {body.user_message}",
            topic_id=body.topic_id,
            topic_title=body.topic_title,
        )
    mcp_block = _fence("External web context", mcp_context, MAX_MCP_CONTEXT_CHARS)
    detail_clause = (
        "Use concise responses by default."
        if resolved_detail == "concise"
        else "Provide very detailed responses with layered explanation depth."
    )
    language_rule = ""
    if resolved_requires_programming and resolved_language:
        language_rule = (
            f'Include one practical fenced code example in "{resolved_language}" when code adds clarity.'
        )
    elif not resolved_requires_programming:
        language_rule = "Avoid code blocks unless the user explicitly asks for code."
    contract = render_contract(
        schema_label="Return markdown only",
        schema_block='(single markdown response, no JSON wrapper)',
        rules=[
            "Be concise but thorough.",
            detail_clause,
            "Keep the explanation oriented to the provided topic context and this Q&A.",
            "If the question drifts outside topic scope, answer briefly and connect back to the current topic.",
            "If the highlighted word is technical, define it and explain why it matters here.",
            "Include one short quote from the context when possible.",
            "If the answer is uncertain from context, explicitly say what is uncertain and avoid inventing facts.",
            "Format as markdown with adaptive structure.",
            "use short paragraphs with blank lines for long responses.",
            "use bullets only when listing steps/checklists/categories.",
            "use headings only when sections improve clarity.",
            "use tables only for direct comparisons/category matrices.",
            "If you use a table, output valid GFM table syntax.",
            "one row per line.",
            "include a separator row (e.g. `| --- | --- |`).",
            "You may include fenced code blocks when code clarifies the explanation.",
            language_rule,
            "You may include fenced Mermaid diagrams when system behavior is easier to explain visually.",
            "If you include fences, always use explicit language tags (for example: ```python, ```mermaid).",
            "End with one practical takeaway sentence.",
        ],
    )

    prompt = f"""The user is reading a Q&A in technical documentation and has a follow-up question.

Context — the Q&A the user was reading:
{_fence("Context question", body.context_question, MAX_CONTEXT_QUESTION_CHARS)}
{_fence("Context answer", body.context_answer, MAX_CONTEXT_ANSWER_CHARS)}

Topic context:
{topic_block}

The user highlighted this word/phrase:
{_fence("Highlighted word", body.word, MAX_WORD_CHARS)}
{history_block}
{memory_block}{learner_state_block}

User's question:
{_fence("User question", body.user_message, MAX_USER_MESSAGE_CHARS)}

Provide a clear, educational explanation grounded in the context above.
{contract}.{mcp_block}"""

    result = await _llm_client.completion(
        prompt,
        body.llm_config,
        user_identity=user,
        task="final",
        system=CHAT_SYSTEM_PROMPT,
        structured=False,
        max_tokens_cap=CHAT_MAX_TOKENS_CAP,
    )
    raise_if_policy_blocked_result(result)
    if str(result.get("error_code", "")).strip().lower() == BUDGET_EXCEEDED_CODE:
        # A distinct, retryable outcome: the per-user LLM call budget
        # (config.llm_user_call_budget) is exhausted for this window. Must not
        # degrade into a generic 500 or a silent "sorry" reply.
        raise HTTPException(
            status_code=429,
            detail={
                "code": BUDGET_EXCEEDED_CODE,
                "message": str(result.get("error") or "LLM call budget exceeded."),
            },
        )

    if result["success"] and result["analysis"]:
        metadata = result.get("metadata", {})
        reply = format_markdown_readable(result["analysis"])
        if bool(body.use_memory) and _learning_store is not None:
            await _persist_conversation_memory(
                user,
                conversation_id=conv_id,
                topic_title=body.topic_title,
                topic_id=body.topic_id,
                word=body.word,
                user_message=body.user_message,
                reply=reply,
                updated_by="chat_follow_up",
            )
        return ChatFollowUpResponse(
            reply=reply,
            provider_used=metadata.get("provider", ""),
            model_used=metadata.get("model", ""),
        )

    error_msg = result.get("error", "Unknown error")
    return ChatFollowUpResponse(
        reply=f"Sorry, I couldn't process that request. Error: {error_msg}",
    )


# ── Document RAG ask endpoint (STUDY_ENABLE_RAG_V1) ────────
_rag_service = None


def init_rag_service(rag_service):
    """Wire the citation-grounded RAG service (document ask)."""
    global _rag_service
    _rag_service = rag_service


@router.post("/ask", response_model=ChatAskResponse)
async def chat_ask(
    body: ChatAskRequest,
    user: dict = Depends(require_auth),
):
    """Answer a question grounded in the user's uploaded documents."""
    if _rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    settings = get_settings()
    if not settings.enable_rag_v1:
        raise HTTPException(
            status_code=503,
            detail="Document RAG is not enabled (STUDY_ENABLE_RAG_V1=false)",
        )
    result = await _rag_service.ask(
        user_id=user["user"],
        message=body.message,
        document_ids=body.document_ids,
        topic_id=body.topic_id,
        conversation_id=body.conversation_id,
        llm_config=body.llm_config,
        user_identity=user,
    )
    return ChatAskResponse(
        answer=result["answer"],
        citations=result["citations"],
        conversation_id=result["conversation_id"],
    )


# ── Highlight deep actions ───────────────────────────────────────
# One endpoint, one action per call: four outcomes for the same
# highlighted term, so the selection popover can turn a reading
# question into an explanation, a flashcard, a scheduled review, or an
# interview-ready answer without a second endpoint per intent.


def _normalise_fragment(value: str) -> str:
    """Collapse whitespace so quotes compare and render predictably."""
    return " ".join(str(value or "").split())


def _short_verbatim_quote(quote: str) -> str:
    """Bound one provided quote to a short, still-verbatim excerpt.

    Only whole sentences survive, so the cited text is never a partial
    sentence the handbook does not contain. A quote whose sentences all
    exceed the cap is dropped rather than clipped: inventing or
    truncating a quotation is worse than citing none.
    """
    text = _normalise_fragment(quote)
    if not text:
        return ""
    if len(text) <= MAX_SOURCE_QUOTE_CHARS:
        return text
    kept: list[str] = []
    total = 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        cost = len(sentence) + (1 if kept else 0)
        if total + cost > MAX_SOURCE_QUOTE_CHARS:
            break
        kept.append(sentence)
        total += cost
    return " ".join(kept)


def _grounded_citation(body: ChatHighlightActionRequest) -> tuple[str, str, str]:
    """Return ``(section, quote, citation_line)`` for the context.

    The citation is built from what the context actually carries. When
    the context names no section the line says so, so the answer never
    invents a handbook reference.
    """
    section = _normalise_fragment(body.source_section or body.section_title)[
        :MAX_SOURCE_SECTION_CHARS
    ]
    quote = _short_verbatim_quote(body.source_quote)
    if not section and not quote:
        return "", "", (
            "The highlighted context names no handbook section, so this answer cites none."
        )
    parts: list[str] = []
    if section:
        parts.append(f'section "{section}"')
    if quote:
        parts.append(f'quote "{quote}"')
    return section, quote, "Grounded in the highlighted context: " + " and ".join(parts) + "."


def _citation_prompt_block(body: ChatHighlightActionRequest) -> str:
    """Fence the grounding the answer must cite (section + one quote)."""
    section, quote, _line = _grounded_citation(body)
    if not section and not quote:
        return _fence(
            "Context source",
            "This context carries no handbook section or quote; say so instead of citing one.",
            MAX_SOURCE_SECTION_CHARS + MAX_SOURCE_QUOTE_CHARS,
        )
    lines: list[str] = []
    if section:
        lines.append(f"Handbook section: {section}")
    if quote:
        lines.append(f"Verbatim quote from the context: {quote}")
    return _fence(
        "Context source",
        "\n".join(lines),
        MAX_SOURCE_SECTION_CHARS + MAX_SOURCE_QUOTE_CHARS,
    )


def _highlight_skill_block(topic_track: str) -> str:
    """Render the topic track's skill taxonomies for the coach persona.

    The taxonomy is module-authored (trusted), so it is capped rather
    than fenced, matching the generation flows that inject it.
    """
    block = skill_context_block(
        level="mid",
        session_type="general",
        track=normalise_track(topic_track),
        include_educational=False,
    )
    return block[:MAX_SKILL_CONTEXT_CHARS] if block else ""


HIGHLIGHT_EXPLAIN_SYSTEM_PROMPT = (
    "You are a helpful study assistant. The user highlighted a term in a "
    "technical Q&A and wants it explained in that context. "
    f"{UNTRUSTED_CLAUSE}"
)


def _detail_clause(resolved_detail: str) -> str:
    return (
        "Use concise responses by default."
        if resolved_detail == "concise"
        else "Provide very detailed responses with layered explanation depth."
    )


def _language_rule(resolved_requires_programming: bool, resolved_language: str) -> str:
    if resolved_requires_programming and resolved_language:
        return (
            f'Include one practical fenced code example in "{resolved_language}" '
            "when code adds clarity."
        )
    if not resolved_requires_programming:
        return "Avoid code blocks unless the user explicitly asks for code."
    return ""


async def _highlight_mcp_block(
    body: ChatHighlightActionRequest, *, query: str
) -> str:
    if _mcp_gateway is None:
        return ""
    mcp_context = await _mcp_gateway.gather_context(
        flow="chat",
        query=query,
        topic_id=body.topic_id,
        topic_title=body.topic_title,
    )
    return _fence("External web context", mcp_context, MAX_MCP_CONTEXT_CHARS)


async def _complete_highlight_prompt(
    *,
    system: str,
    prompt: str,
    body: ChatHighlightActionRequest,
    user: dict,
    conv_id: str,
) -> ChatHighlightActionResponse:
    """Run one highlight prompt and fold the turn into memory."""
    result = await _llm_client.completion(
        prompt,
        body.llm_config,
        user_identity=user,
        task="final",
        system=system,
        structured=False,
        max_tokens_cap=CHAT_MAX_TOKENS_CAP,
    )
    raise_if_policy_blocked_result(result)
    if str(result.get("error_code", "")).strip().lower() == BUDGET_EXCEEDED_CODE:
        # Same distinct, retryable outcome as the follow-up flow: the
        # per-user LLM call budget is exhausted for this window.
        raise HTTPException(
            status_code=429,
            detail={
                "code": BUDGET_EXCEEDED_CODE,
                "message": str(result.get("error") or "LLM call budget exceeded."),
            },
        )

    metadata = result.get("metadata", {})
    if not (result["success"] and result["analysis"]):
        error_msg = result.get("error", "Unknown error")
        return ChatHighlightActionResponse(
            action=body.action.value,
            reply=f"Sorry, I couldn't process that request. Error: {error_msg}",
            provider_used=metadata.get("provider", ""),
            model_used=metadata.get("model", ""),
        )

    reply = format_markdown_readable(result["analysis"])
    if bool(body.use_memory) and _learning_store is not None:
        await _persist_conversation_memory(
            user,
            conversation_id=conv_id,
            topic_title=body.topic_title,
            topic_id=body.topic_id,
            word=body.word,
            user_message=body.user_message or body.word,
            reply=reply,
            updated_by="chat_highlight_action",
        )
    return ChatHighlightActionResponse(
        action=body.action.value,
        reply=reply,
        provider_used=metadata.get("provider", ""),
        model_used=metadata.get("model", ""),
    )


async def _explain_highlight(
    body: ChatHighlightActionRequest, user: dict
) -> ChatHighlightActionResponse:
    """Explain the highlighted term in the context of the Q&A card."""
    conv_id = _conversation_id(body)
    resolved_detail, resolved_language, resolved_requires_programming = (
        await _resolve_topic_ai_settings(
            user,
            topic_id=body.topic_id,
            response_detail=body.response_detail,
            preferred_language=body.preferred_language,
            requires_programming=body.requires_programming,
        )
    )
    memory_block, learner_state_block = await _conversation_blocks(
        user,
        topic_id=body.topic_id,
        topic_track=body.topic_track,
        conversation_id=conv_id,
        use_memory=bool(body.use_memory),
    )
    section, quote, citation_line = _grounded_citation(body)
    language_rule = _language_rule(resolved_requires_programming, resolved_language)
    contract = render_contract(
        schema_label="Return markdown only",
        schema_block="(single markdown response, no JSON wrapper)",
        rules=[
            "Be concise but thorough.",
            _detail_clause(resolved_detail),
            "Keep the explanation oriented to the provided topic context and this Q&A.",
            "If the question drifts outside topic scope, answer briefly and connect back to the current topic.",
            "Define the highlighted term and explain why it matters in this context.",
            "Cite the handbook section named in the context source block, and include one short verbatim quote from it.",
            "When the context source block says there is no section or quote, state that the context has none instead of inventing one.",
            "If the answer is uncertain from context, explicitly say what is uncertain and avoid inventing facts.",
            "Format as markdown with adaptive structure.",
            "use short paragraphs with blank lines for long responses.",
            "use bullets only when listing steps/checklists/categories.",
            "use headings only when sections improve clarity.",
            "use tables only for direct comparisons/category matrices.",
            "If you use a table, output valid GFM table syntax.",
            "one row per line.",
            "include a separator row (e.g. `| --- | --- |`).",
            "You may include fenced code blocks when code clarifies the explanation.",
            language_rule,
            "You may include fenced Mermaid diagrams when system behavior is easier to explain visually.",
            "If you include fences, always use explicit language tags (for example: ```python, ```mermaid).",
            "End with one practical takeaway sentence.",
        ],
    )
    question = _normalise_fragment(body.user_message) or (
        f'Explain "{_normalise_fragment(body.word)}" in the context of this Q&A.'
    )
    mcp_block = await _highlight_mcp_block(
        body,
        query=f"{body.topic_title or body.topic_id} {body.word} {question}",
    )
    topic_block = _topic_metadata_block(
        topic_id=body.topic_id,
        topic_title=body.topic_title,
        topic_track=body.topic_track,
        section_title=body.section_title,
        mode=body.mode,
    )
    citation_block = _citation_prompt_block(body)
    prompt = f"""The user highlighted a term in a technical Q&A and wants it explained.

Context — the Q&A the user was reading:
{_fence("Context question", body.context_question, MAX_CONTEXT_QUESTION_CHARS)}
{_fence("Context answer", body.context_answer, MAX_CONTEXT_ANSWER_CHARS)}
{citation_block}

Topic context:
{topic_block}

The user highlighted this word/phrase:
{_fence("Highlighted word", body.word, MAX_WORD_CHARS)}
{_history_block(body.history)}
{memory_block}{learner_state_block}

User's question:
{_fence("User question", question, MAX_USER_MESSAGE_CHARS)}

Provide a clear, educational explanation grounded in the context above.
{contract}.{mcp_block}"""

    response = await _complete_highlight_prompt(
        system=HIGHLIGHT_EXPLAIN_SYSTEM_PROMPT,
        prompt=prompt,
        body=body,
        user=user,
        conv_id=conv_id,
    )
    response.reply = f"> **Source:** {citation_line}\n\n{response.reply}"
    response.source_section = section
    response.source_quote = quote
    return response


async def _interview_angle_highlight(
    body: ChatHighlightActionRequest, user: dict
) -> ChatHighlightActionResponse:
    """Coach the highlighted term as an interview answer."""
    conv_id = _conversation_id(body)
    resolved_detail, resolved_language, resolved_requires_programming = (
        await _resolve_topic_ai_settings(
            user,
            topic_id=body.topic_id,
            response_detail=body.response_detail,
            preferred_language=body.preferred_language,
            requires_programming=body.requires_programming,
        )
    )
    memory_block, learner_state_block = await _conversation_blocks(
        user,
        topic_id=body.topic_id,
        topic_track=body.topic_track,
        conversation_id=conv_id,
        use_memory=bool(body.use_memory),
    )
    section, quote, citation_line = _grounded_citation(body)
    language_rule = _language_rule(resolved_requires_programming, resolved_language)
    contract = render_contract(
        schema_label="Return markdown only",
        schema_block="(single markdown response, no JSON wrapper)",
        rules=[
            "Answer how you would explain the highlighted term in a real interview.",
            "Use a STAR-ish structure: a one-sentence Situation/Task framing, the Action you take, and the Result you aim for.",
            "Ground every claim in the provided context Q&A and the context source block; do not invent facts beyond them.",
            "Name the main tradeoff an interviewer would probe, and end with the follow-up question they are likely to ask next.",
            "Be concise but thorough.",
            _detail_clause(resolved_detail),
            "Format as markdown with adaptive structure: short paragraphs, headings only where they help.",
            language_rule,
            "Close with a 20-second spoken version of the answer.",
        ],
    )
    skill_block = _highlight_skill_block(body.topic_track)
    skill_section = f"\n\n{skill_block}" if skill_block else ""
    question = _normalise_fragment(body.user_message) or (
        f'How would you explain "{_normalise_fragment(body.word)}" in an interview?'
    )
    mcp_block = await _highlight_mcp_block(
        body,
        query=f"{body.topic_title or body.topic_id} {body.word} {question}",
    )
    topic_block = _topic_metadata_block(
        topic_id=body.topic_id,
        topic_title=body.topic_title,
        topic_track=body.topic_track,
        section_title=body.section_title,
        mode=body.mode,
    )
    citation_block = _citation_prompt_block(body)
    prompt = f"""The user is preparing for interviews and wants a coached answer for a highlighted term.

Context — the Q&A the user was reading:
{_fence("Context question", body.context_question, MAX_CONTEXT_QUESTION_CHARS)}
{_fence("Context answer", body.context_answer, MAX_CONTEXT_ANSWER_CHARS)}
{citation_block}

Topic context:
{topic_block}

The user highlighted this word/phrase:
{_fence("Highlighted word", body.word, MAX_WORD_CHARS)}
{_history_block(body.history)}
{memory_block}{learner_state_block}

Interview question to coach:
{_fence("User question", question, MAX_USER_MESSAGE_CHARS)}

Provide a coached, interview-ready markdown answer grounded in the context above.
{contract}.{skill_section}{mcp_block}"""

    response = await _complete_highlight_prompt(
        system=HIGHLIGHT_INTERVIEW_SYSTEM_PROMPT,
        prompt=prompt,
        body=body,
        user=user,
        conv_id=conv_id,
    )
    response.reply = f"> **Source:** {citation_line}\n\n{response.reply}"
    response.source_section = section
    response.source_quote = quote
    return response


def _require_card_store() -> CardStore:
    if _card_store is None:
        raise HTTPException(status_code=503, detail="Card store not initialised")
    return _card_store


def _require_learning_store() -> LearningStore:
    if _learning_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return _learning_store


def _highlight_deck_name(body: ChatHighlightActionRequest) -> str:
    """Name of the per-topic deck that gathers highlighted-term cards."""
    topic = _normalise_fragment(body.topic_title) or _normalise_fragment(body.topic_id)
    name = f"{HIGHLIGHT_DECK_FALLBACK_NAME} - {topic}" if topic else (
        HIGHLIGHT_DECK_FALLBACK_NAME
    )
    return name[:DECK_NAME_MAX]


def _highlight_card_texts(body: ChatHighlightActionRequest) -> tuple[str, str]:
    """Build the flashcard front/back from the term and its context."""
    word = _normalise_fragment(body.word) or "this term"
    scope = (
        _normalise_fragment(body.source_section)
        or _normalise_fragment(body.section_title)
        or _normalise_fragment(body.topic_title)
        or "this topic"
    )
    front = f'What does "{word}" mean in the context of {scope}?'
    back = (
        _normalise_fragment(body.context_answer)
        or f'Explain "{word}" and why it matters in {scope}.'
    )
    quote = _short_verbatim_quote(body.source_quote)
    if quote:
        back = f'{back}\n\nSource ({scope}): "{quote}"'
    return front, back


async def _resolve_highlight_deck(
    store: CardStore, body: ChatHighlightActionRequest, user: dict
) -> dict:
    """Resolve the deck that receives the card, creating it once per topic."""
    requested = _normalise_fragment(body.deck_id)
    if requested:
        deck = await store.run_async(
            store.get_deck, user_id=user["user"], deck_id=requested
        )
        if deck is None:
            raise HTTPException(status_code=404, detail="deck not found")
        return deck

    name = _highlight_deck_name(body)
    decks = await store.run_async(store.list_decks, user_id=user["user"])
    for deck in decks:
        if str(deck.get("name", "")) == name:
            return deck
    return await store.run_async(
        store.create_deck,
        user_id=user["user"],
        name=name,
        description=(
            "Flashcards created from highlighted terms in "
            f"{_normalise_fragment(body.topic_title) or 'your topics'}."
        ),
    )


async def _create_highlight_flashcard(
    body: ChatHighlightActionRequest, user: dict
) -> ChatHighlightActionResponse:
    """Create a flashcard for the highlighted term via the card store."""
    if not get_settings().enable_flashcards_v1:
        raise HTTPException(
            status_code=503,
            detail="Flashcards disabled (STUDY_ENABLE_FLASHCARDS_V1=false)",
        )
    store = _require_card_store()
    deck = await _resolve_highlight_deck(store, body, user)
    front, back = _highlight_card_texts(body)
    source_ref = (
        _normalise_fragment(body.source_section)
        or _normalise_fragment(body.section_title)
        or _normalise_fragment(body.topic_id)
    )
    card = await store.run_async(
        store.create_card,
        user_id=user["user"],
        deck_id=deck["deck_id"],
        front=front,
        back=back,
        tags=["highlight", "flashcard"],
        source_ref=source_ref,
    )
    return ChatHighlightActionResponse(
        action=body.action.value,
        card_id=card["card_id"],
        deck_id=card["deck_id"],
        deck_name=deck["name"],
        due_at=str(card.get("fsrs", {}).get("due_at", "")),
    )


def _highlight_review_card_id(body: ChatHighlightActionRequest) -> str:
    """Stable FSRS card id for a highlighted review item.

    Re-scheduling the same term in the same topic updates that card
    instead of piling a duplicate into the review queue.
    """
    topic = _normalise_fragment(body.topic_id) or "highlights"
    slug = re.sub(r"[^a-z0-9]+", "-", _normalise_fragment(body.word).lower()).strip("-")
    card_id = f"{HIGHLIGHT_REVIEW_CARD_PREFIX}{topic}:{slug or 'term'}"
    return card_id[:HIGHLIGHT_REVIEW_CARD_ID_MAX]


async def _schedule_highlight_review(
    body: ChatHighlightActionRequest, user: dict
) -> ChatHighlightActionResponse:
    """Schedule a review item for the highlighted term on the FSRS scheduler."""
    if not getattr(get_settings(), "enable_fsrs_v1", False):
        raise HTTPException(status_code=404, detail="FSRS scheduling disabled")
    store = _require_learning_store()
    card_id = _highlight_review_card_id(body)
    result = await store.run_async(
        store.record_review,
        user_id=user["user"],
        card_id=card_id,
        topic_id=_normalise_fragment(body.topic_id) or "highlights",
        rating=fsrs_scheduler.RATING_GOOD,
        response_time_ms=0,
        source_type="flashcard",
    )
    return ChatHighlightActionResponse(
        action=body.action.value,
        card_id=card_id,
        due_at=str(result["card"].get("due_at", "")),
    )


@router.post("/highlight-action", response_model=ChatHighlightActionResponse)
async def chat_highlight_action(
    body: ChatHighlightActionRequest,
    user: dict = Depends(require_auth),
):
    """Run one deep action on a highlighted term inside a Q&A card.

    ``explain`` answers in grounded markdown, ``flashcard`` and
    ``review`` write a card through the existing card and FSRS APIs, and
    ``interview_angle`` coaches the term as an interview answer. Every
    externally sourced string is fenced and bounded exactly as the
    follow-up flow fences it.
    """
    if _llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    action = body.action
    if action == HighlightActionEnum.FLASHCARD:
        return await _create_highlight_flashcard(body, user)
    if action == HighlightActionEnum.REVIEW:
        return await _schedule_highlight_review(body, user)
    if action == HighlightActionEnum.INTERVIEW_ANGLE:
        return await _interview_angle_highlight(body, user)
    return await _explain_highlight(body, user)
