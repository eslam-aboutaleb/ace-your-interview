"""Chat router — interactive follow-up conversations on highlighted words."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.models import ChatFollowUpRequest, ChatFollowUpResponse
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/api/chat", tags=["chat"])

_llm_client: LLMClient | None = None


def init(llm_client: LLMClient):
    global _llm_client
    _llm_client = llm_client


@router.post("/follow-up", response_model=ChatFollowUpResponse)
async def chat_follow_up(body: ChatFollowUpRequest):
    """Answer a follow-up question about a highlighted word/phrase in a Q&A."""
    if _llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    # Build conversation history for context
    history_text = ""
    if body.history:
        for msg in body.history[-6:]:  # Last 6 messages for context window
            role_label = "User" if msg.role == "user" else "Assistant"
            history_text += f"\n{role_label}: {msg.content}"

    prompt = f"""You are a helpful study assistant. The user is studying technical documentation and has a follow-up question.

Context — the user was reading this Q&A:
Question: {body.context_question}
Answer: {body.context_answer}

The user highlighted the word/phrase: "{body.word}"
{f"Previous conversation:{history_text}" if history_text else ""}

User's question: {body.user_message}

Provide a clear, educational explanation. Be concise but thorough (2-5 sentences). If the highlighted word is a technical term, define it and explain its relevance in this context. Use examples where helpful."""

    result = await _llm_client.completion(prompt, body.llm_config)

    if result["success"] and result["analysis"]:
        metadata = result.get("metadata", {})
        return ChatFollowUpResponse(
            reply=result["analysis"],
            provider_used=metadata.get("provider", ""),
            model_used=metadata.get("model", ""),
        )

    error_msg = result.get("error", "Unknown error")
    return ChatFollowUpResponse(
        reply=f"Sorry, I couldn't process that request. Error: {error_msg}",
    )
