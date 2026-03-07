"""Chat router — interactive follow-up conversations on highlighted words."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import require_auth
from app.schemas.models import ChatFollowUpRequest, ChatFollowUpResponse
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore
from app.services.llm_client import LLMClient
from app.services.llm_policy import raise_if_policy_blocked_result
from app.services.markdown_formatter import format_markdown_readable
from app.services.mcp_gateway import MCPGateway
from app.services.prompt_blocks import optional_context_block, render_contract

router = APIRouter(prefix="/api/chat", tags=["chat"])

_llm_client: LLMClient | None = None
_parser: DocParser | None = None
_learning_store: LearningStore | None = None
_mcp_gateway: MCPGateway | None = None


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
    summary_text = str(summary.get("summary", "")).strip()
    if not summary_text:
        return ""
    return optional_context_block(
        "Conversation memory (assistant summary from prior turns; use as soft context)",
        summary_text,
        900,
    )


def init(
    llm_client: LLMClient,
    parser: DocParser | None = None,
    learning_store: LearningStore | None = None,
    mcp_gateway: MCPGateway | None = None,
):
    global _llm_client, _parser, _learning_store, _mcp_gateway
    _llm_client = llm_client
    _parser = parser
    _learning_store = learning_store
    _mcp_gateway = mcp_gateway


@router.post("/follow-up", response_model=ChatFollowUpResponse)
async def chat_follow_up(
    body: ChatFollowUpRequest,
    user: dict = Depends(require_auth),
):
    """Answer a follow-up question about a highlighted word/phrase in a Q&A."""
    if _llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    resolved_detail = body.response_detail.value if body.response_detail else "concise"
    resolved_language = (body.preferred_language or "").strip().lower()
    resolved_requires_programming = bool(body.requires_programming)
    conv_id = _conversation_id(body)
    memory_block = ""

    if body.topic_id and _learning_store is not None:
        topic_detail = None
        if _parser is not None:
            static = _parser.get_topic(body.topic_id)
            if static:
                topic_detail = static.model_dump(mode="json")
        if topic_detail is None:
            custom = await _learning_store.run_async(
                _learning_store.get_custom_topic,
                user_id=user["user"],
                topic_id=body.topic_id,
            )
            if custom:
                topic_detail = custom
        if topic_detail is not None:
            resolved = await _learning_store.run_async(
                _learning_store.resolve_topic_ai_settings,
                user_id=user["user"],
                topic_id=body.topic_id,
                topic_detail=topic_detail,
            )
            if body.response_detail is None:
                resolved_detail = resolved.get("response_detail", "concise")
            if body.preferred_language is None:
                resolved_language = resolved.get("preferred_language", "")
            if body.requires_programming is None:
                resolved_requires_programming = bool(resolved.get("requires_programming", False))

    if bool(body.use_memory) and _learning_store is not None:
        memory = await _learning_store.run_async(
            _learning_store.get_assistant_memory,
            user_id=user["user"],
            conversation_id=conv_id,
            flow="chat",
        )
        memory_block = _memory_prompt_block(memory)

    # Build conversation history for context
    history_text = ""
    if body.history:
        for msg in body.history[-6:]:  # Last 6 messages for context window
            role_label = "User" if msg.role == "user" else "Assistant"
            history_text += f"\n{role_label}: {msg.content}"

    topic_context_lines: list[str] = []
    if body.topic_id:
        topic_context_lines.append(f"- Topic ID: {body.topic_id}")
    if body.topic_title:
        topic_context_lines.append(f"- Topic Title: {body.topic_title}")
    if body.topic_track:
        topic_context_lines.append(f"- Topic Track: {body.topic_track}")
    if body.section_title:
        topic_context_lines.append(f"- Section: {body.section_title}")
    if body.mode:
        topic_context_lines.append(f"- Learning Mode: {body.mode}")
    topic_context = (
        "\n".join(topic_context_lines)
        if topic_context_lines
        else "- Topic metadata not provided."
    )
    mcp_context = ""
    if _mcp_gateway is not None:
        mcp_context = await _mcp_gateway.gather_context(
            flow="chat",
            query=f"{body.topic_title or body.topic_id} {body.word} {body.user_message}",
            topic_id=body.topic_id,
            topic_title=body.topic_title,
        )
    mcp_block = optional_context_block(
        "External context (optional, use only if helpful and factual)",
        mcp_context,
        2500,
    )
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

    prompt = f"""You are a helpful study assistant. The user is studying technical documentation and has a follow-up question.

Context — the user was reading this Q&A:
Question: {body.context_question}
Answer: {body.context_answer}

Topic context:
{topic_context}

The user highlighted the word/phrase: "{body.word}"
{f"Previous conversation:{history_text}" if history_text else ""}
{memory_block}

User's question: {body.user_message}

Provide a clear, educational explanation grounded in the context above.
{contract}.{mcp_block}"""

    result = await _llm_client.completion(
        prompt,
        body.llm_config,
        user_identity=user,
    )
    raise_if_policy_blocked_result(result)

    if result["success"] and result["analysis"]:
        metadata = result.get("metadata", {})
        reply = format_markdown_readable(result["analysis"])
        if bool(body.use_memory) and _learning_store is not None:
            previous = await _learning_store.run_async(
                _learning_store.get_assistant_memory,
                user_id=user["user"],
                conversation_id=conv_id,
                flow="chat",
            ) or {}
            prev_summary = ""
            if isinstance(previous.get("summary"), dict):
                prev_summary = str(previous["summary"].get("summary", "")).strip()
            merged_summary = (
                f"{prev_summary}\n"
                f"Topic: {body.topic_title or body.topic_id or 'unknown'} | "
                f"Word: {body.word}\n"
                f"User asked: {body.user_message[:240]}\n"
                f"Assistant: {reply[:380]}"
            ).strip()
            await _learning_store.run_async(
                _learning_store.upsert_assistant_memory,
                user_id=user["user"],
                conversation_id=conv_id,
                flow="chat",
                summary={
                    "summary": merged_summary[:1200],
                    "topic_id": body.topic_id,
                    "word": body.word,
                    "updated_by": "chat_follow_up",
                },
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
