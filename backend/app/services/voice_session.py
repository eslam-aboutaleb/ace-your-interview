"""Voice session orchestrator — STT → LLM → TTS pipeline.

Manages a single voice conversation session, routing user speech through
the appropriate flow (interview, chat, or Q&A) and returning synthesized
audio responses.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from app.config import get_settings
from app.schemas.models import LLMConfigRequest
from app.services.llm_client import LLMClient
from app.services.voice_providers import (
    NullSTTProvider,
    NullTTSProvider,
    STTProvider,
    TTSProvider,
    create_stt_provider,
    create_tts_provider,
)

logger = logging.getLogger(__name__)


class VoiceSession:
    """Orchestrates a single voice interaction turn: audio-in → text → LLM → text → audio-out."""

    def __init__(
        self,
        llm_client: LLMClient,
        stt: STTProvider,
        tts: TTSProvider,
        session_type: str = "chat",
        session_id: str = "",
        user_identity: dict | None = None,
        llm_config: LLMConfigRequest | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ):
        self.llm = llm_client
        self.stt = stt
        self.tts = tts
        self.session_type = session_type
        self.session_id = session_id
        self.user_identity = user_identity or {}
        self.llm_config = llm_config
        # The persona is server-owned. ``system_prompt`` exists for server-side
        # callers only (it is never populated from a wire message); when omitted
        # the session type's default persona is used.
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.conversation_history: list[dict[str, str]] = conversation_history or []
        self.turn_index = 0

    def _default_system_prompt(self) -> str:
        if self.session_type == "interview":
            return (
                "You are a professional interview coach conducting a mock interview. "
                "Ask clear, focused questions and provide constructive feedback. "
                "Keep responses concise and conversational since this is a voice interaction. "
                "Limit responses to 2-3 sentences."
            )
        elif self.session_type == "qa":
            return (
                "You are a knowledgeable study tutor helping a student learn. "
                "Ask questions to test understanding and provide clear explanations. "
                "Keep responses brief and conversational for voice interaction. "
                "Limit responses to 2-3 sentences."
            )
        return (
            "You are a helpful study assistant. Answer questions clearly and concisely. "
            "Keep responses brief since this is a voice conversation. "
            "Limit responses to 2-3 sentences."
        )

    async def process_audio_turn(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/webm",
    ) -> dict[str, Any]:
        """Process one voice turn: transcribe → LLM → synthesize.

        Returns:
            {
                "transcript": str,           # what the user said
                "response_text": str,        # LLM reply text
                "audio_bytes": bytes,        # TTS audio (MP3)
                "stt_latency_ms": int,
                "llm_latency_ms": int,
                "tts_latency_ms": int,
                "total_latency_ms": int,
            }
        """
        total_start = time.monotonic()

        # 1) STT — transcribe user speech
        stt_start = time.monotonic()
        try:
            transcript = await self.stt.transcribe(audio_bytes, mime_type=mime_type)
        except Exception as e:
            logger.error("STT failed: %s", e)
            return {"error": f"Speech recognition failed: {e}", "transcript": "", "response_text": "", "audio_bytes": b""}
        stt_ms = int((time.monotonic() - stt_start) * 1000)

        if not transcript.strip():
            return {
                "transcript": "",
                "response_text": "",
                "audio_bytes": b"",
                "stt_latency_ms": stt_ms,
                "llm_latency_ms": 0,
                "tts_latency_ms": 0,
                "total_latency_ms": int((time.monotonic() - total_start) * 1000),
            }

        # 2) LLM — generate response
        self.conversation_history.append({"role": "user", "content": transcript})

        llm_start = time.monotonic()
        try:
            response_text = await self._call_llm(transcript)
        except Exception as e:
            logger.error("LLM failed: %s", e)
            response_text = "I'm sorry, I had trouble processing that. Could you repeat?"
        llm_ms = int((time.monotonic() - llm_start) * 1000)

        self.conversation_history.append({"role": "assistant", "content": response_text})
        self.turn_index += 1

        # 3) TTS — synthesize response
        tts_start = time.monotonic()
        try:
            audio_out = await self.tts.synthesize(response_text)
        except Exception as e:
            logger.error("TTS failed: %s", e)
            audio_out = b""
        tts_ms = int((time.monotonic() - tts_start) * 1000)

        total_ms = int((time.monotonic() - total_start) * 1000)

        return {
            "transcript": transcript,
            "response_text": response_text,
            "audio_bytes": audio_out,
            "stt_latency_ms": stt_ms,
            "llm_latency_ms": llm_ms,
            "tts_latency_ms": tts_ms,
            "total_latency_ms": total_ms,
        }

    async def generate_speech(self, text: str) -> bytes:
        """TTS-only: convert text to audio (for speaking questions aloud)."""
        try:
            return await self.tts.synthesize(text)
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            return b""

    async def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        """STT-only: convert audio to text (for voice input without full pipeline)."""
        try:
            return await self.stt.transcribe(audio_bytes, mime_type=mime_type)
        except Exception as e:
            logger.error("STT transcribe failed: %s", e)
            return ""

    async def _call_llm(self, user_text: str) -> str:
        """Send conversation to LLM and get response."""
        # Build the user turn from the trimmed history (last 10 turns = 20
        # messages). The persona is sent as a real system message, never
        # flattened into the user text.
        history_window = self.conversation_history[-20:]
        parts = [f"{m['role'].upper()}: {m['content']}" for m in history_window]
        if not parts:
            parts.append(f"USER: {user_text}")
        prompt = "\n".join(parts)

        result = await self.llm.completion(
            prompt=prompt,
            llm_config=self.llm_config,
            user_identity=self.user_identity,
            system=self.system_prompt,
            task="final",
            max_tokens_cap=None,
        )

        if result.get("success") and result.get("analysis"):
            return str(result["analysis"]).strip()

        return "I'm sorry, I couldn't generate a response. Could you try again?"

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt mid-session."""
        self.system_prompt = prompt

    def clear_history(self) -> None:
        """Reset conversation history for a new topic."""
        self.conversation_history.clear()
        self.turn_index = 0


class UnsupportedVoiceTierError(ValueError):
    """Raised when a voice tier is not one of the implemented tiers."""


def create_voice_session(
    llm_client: LLMClient,
    tier: str,
    session_type: str = "chat",
    session_id: str = "",
    user_identity: dict | None = None,
    llm_config: LLMConfigRequest | None = None,
    system_prompt: str | None = None,
) -> VoiceSession:
    """Factory to create a VoiceSession with the right STT/TTS for the given tier.

    ``system_prompt`` is a server-side-only override. The WebSocket router never
    passes it, so a client cannot redefine the assistant's persona.
    """
    settings = get_settings()

    if tier == "cloud":
        # Mirror the STT key selection for TTS: only OpenAI TTS needs a key, and
        # omitting it makes every cloud-tier synthesis fail at request time.
        stt_key = os.getenv("GROQ_API_KEY", "") if settings.voice_stt_provider == "groq" else os.getenv("OPENAI_API_KEY", "")
        stt = create_stt_provider(settings.voice_stt_provider, api_key=stt_key, model=settings.voice_stt_model)
        tts_key = os.getenv("OPENAI_API_KEY", "") if settings.voice_tts_provider == "openai" else ""
        tts = create_tts_provider(settings.voice_tts_provider, voice=settings.voice_tts_voice, api_key=tts_key)
    elif tier == "browser":
        # Server-side STT/TTS are genuinely unused on this tier: the browser
        # handles audio locally. Inert providers make an accidental server-side
        # call fail loudly instead of quietly spending an STT/TTS request.
        stt = NullSTTProvider()
        tts = NullTTSProvider()
    else:
        raise UnsupportedVoiceTierError(f"Unknown voice tier: {tier!r}")

    return VoiceSession(
        llm_client=llm_client,
        stt=stt,
        tts=tts,
        session_type=session_type,
        session_id=session_id,
        user_identity=user_identity,
        llm_config=llm_config,
        system_prompt=system_prompt,
    )
