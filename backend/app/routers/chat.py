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

router = APIRouter(prefix="/api/chat", tags=["chat"])

_llm_client: LLMClient | None = None
_parser: DocParser | None = None
_learning_store: LearningStore | None = None
_mcp_gateway: MCPGateway | None = None


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

    if body.topic_id and _learning_store is not None:
        topic_detail = None
        if _parser is not None:
            static = _parser.get_topic(body.topic_id)
            if static:
                topic_detail = static.model_dump(mode="json")
        if topic_detail is None:
            custom = _learning_store.get_custom_topic(user_id=user["user"], topic_id=body.topic_id)
            if custom:
                topic_detail = custom
        if topic_detail is not None:
            resolved = _learning_store.resolve_topic_ai_settings(
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
    mcp_block = (
        f"\n\nExternal context (optional, use only if helpful and factual):\n{mcp_context}"
        if mcp_context
        else ""
    )
    detail_clause = (
        "Use concise responses by default."
        if resolved_detail == "concise"
        else "Provide very detailed responses with layered explanation depth."
    )
    language_clause = ""
    if resolved_requires_programming and resolved_language:
        language_clause = (
            f'\n- Include one practical fenced code example in "{resolved_language}" when code adds clarity.'
        )
    elif not resolved_requires_programming:
        language_clause = "\n- Avoid code blocks unless the user explicitly asks for code."

    prompt = f"""You are a helpful study assistant. The user is studying technical documentation and has a follow-up question.

Context — the user was reading this Q&A:
Question: {body.context_question}
Answer: {body.context_answer}

Topic context:
{topic_context}

The user highlighted the word/phrase: "{body.word}"
{f"Previous conversation:{history_text}" if history_text else ""}

User's question: {body.user_message}

Provide a clear, educational explanation grounded in the context above.
Rules:
- Be concise but thorough.
- {detail_clause}
- Keep the explanation oriented to the provided topic context and this Q&A.
- If the question drifts outside topic scope, answer briefly and connect back to the current topic.
- If the highlighted word is technical, define it and explain why it matters here.
- Include one short quote from the context when possible.
- If the answer is uncertain from context, explicitly say what is uncertain and avoid inventing facts.
- Format as markdown with adaptive structure:
  - use short paragraphs with blank lines for long responses,
  - use bullets only when listing steps/checklists/categories,
  - use headings only when sections improve clarity,
  - use tables only for direct comparisons/category matrices.
- If you use a table, output valid GFM table syntax:
  - one row per line,
  - include a separator row (e.g. `| --- | --- |`).
- You may include fenced code blocks when code clarifies the explanation.{language_clause}
- You may include fenced Mermaid diagrams when system behavior is easier to explain visually.
- If you include fences, always use explicit language tags (for example: ```python, ```mermaid).
- End with one practical takeaway sentence.
Return markdown only.{mcp_block}"""

    result = await _llm_client.completion(
        prompt,
        body.llm_config,
        user_identity=user,
    )
    raise_if_policy_blocked_result(result)

    if result["success"] and result["analysis"]:
        metadata = result.get("metadata", {})
        reply = format_markdown_readable(result["analysis"])
        return ChatFollowUpResponse(
            reply=reply,
            provider_used=metadata.get("provider", ""),
            model_used=metadata.get("model", ""),
        )

    error_msg = result.get("error", "Unknown error")
    return ChatFollowUpResponse(
        reply=f"Sorry, I couldn't process that request. Error: {error_msg}",
    )
