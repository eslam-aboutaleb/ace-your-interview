"""Voice session orchestrator — STT → LLM → TTS pipeline.

Manages a single voice conversation session, routing user speech through
the appropriate flow (interview, chat, or Q&A) and returning synthesized
audio responses.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator

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
        # The in-flight turn runs as a cancellable task so a
        # client ``interrupt`` can cut it mid-pipeline.
        self._turn_task: asyncio.Task[dict[str, Any]] | None = None
        # Set True once a turn reaches the LLM stage. The router
        # reads it after an interrupt to decide whether the turn
        # consumed an LLM turn of the connection budget.
        self.llm_reached = False

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

        The pipeline runs as a cancellable task so a client
        ``interrupt`` can cut it mid-flight. Returns:
            {
                "transcript": str,           # what the user said
                "response_text": str,        # LLM reply text
                "audio_bytes": bytes,        # TTS audio (MP3)
                "stt_latency_ms": int,
                "llm_latency_ms": int,
                "tts_latency_ms": int,
                "total_latency_ms": int,
                "interrupted": bool,
            }
        """
        self.llm_reached = False
        # The turn runs as its own task (the router wraps this
        # coroutine in one). ``interrupt`` cancels that task, so
        # the pipeline is cut at whatever stage is in flight and
        # no inner work is orphaned.
        self._turn_task = asyncio.current_task()
        try:
            return await self._run_audio_turn(audio_bytes, mime_type=mime_type)
        except asyncio.CancelledError:
            return {
                "transcript": "",
                "response_text": "",
                "audio_bytes": b"",
                "stt_latency_ms": 0,
                "llm_latency_ms": 0,
                "tts_latency_ms": 0,
                "total_latency_ms": 0,
                "interrupted": True,
            }
        finally:
            self._turn_task = None

    def interrupt(self) -> None:
        """Cancel the in-flight turn, if any."""
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()

    async def _run_audio_turn(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str = "audio/webm",
    ) -> dict[str, Any]:
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
                "interrupted": False,
            }

        # 2) LLM — generate response
        self.conversation_history.append({"role": "user", "content": transcript})

        llm_start = time.monotonic()
        self.llm_reached = True
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
            "interrupted": False,
        }

    async def generate_speech_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks, falling back to full synthesis.

        Wraps the provider's ``synthesize_stream`` so a long
        utterance can be cut mid-stream by an interrupt. When the
        provider cannot stream, the fully synthesized audio is
        yielded as a single chunk (the ``synthesize`` fallback).
        """
        try:
            async for chunk in self.tts.synthesize_stream(text):
                yield chunk
            return
        except Exception as e:
            logger.error("Streaming TTS failed, falling back: %s", e)
        audio = await self.generate_speech(text)
        if audio:
            yield audio

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
            flow="voice_turn",
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


class VoiceInterviewSession:
    """Voice-to-voice mock interview orchestrator.

    Wraps a :class:`VoiceSession` (STT/TTS) with the interview
    store and generator: the first question is generated and
    spoken on start, and each spoken answer is transcribed,
    evaluated, recorded, and spoken back as feedback. The
    session is capped at the configured ``turn_count``; when
    the last turn is recorded the report is built and saved.
    """

    def __init__(
        self,
        *,
        voice_session: "VoiceSession",
        store: Any,
        generator: Any,
        session: dict[str, Any],
        user_id: str,
        user_identity: dict | None = None,
        llm_config: LLMConfigRequest | None = None,
    ):
        self.voice = voice_session
        self.store = store
        self.generator = generator
        self.session = dict(session)
        self.user_identity = user_identity or {}
        self.llm_config = llm_config
        # ``create_session`` does not echo user_id back, so the
        # caller (which authenticated the user) supplies it.
        self.user_id = str(user_id or "")
        self.session_id = str(self.session.get("session_id") or "")
        self.turn_count = int(self.session.get("turn_count") or 0)
        self.completed = False
        # The in-flight answer runs as a cancellable task so a
        # client ``interrupt`` (barge-in) can cut it mid-pipeline.
        self._turn_task: asyncio.Task[dict[str, Any]] | None = None
        # Set True once an answer reaches the evaluation (LLM)
        # stage; the router reads it after an interrupt to decide
        # whether the turn consumed an LLM turn of the budget.
        self.llm_reached = False

    @property
    def turns_completed(self) -> int:
        return int(self.session.get("turns_completed") or 0)

    @property
    def turn_limit_reached(self) -> bool:
        return self.turn_count > 0 and self.turns_completed >= self.turn_count

    async def start(self) -> dict[str, Any]:
        """Generate the first question and speak it aloud."""
        context = self.store.get_session_context(
            user_id=self.user_id,
            session_id=self.session_id,
        )
        if not context:
            raise RuntimeError("Interview session context unavailable")
        self.session = context
        question = await self.generator.generate_question(
            session=self.session,
            turns=[],
            llm_config=self.llm_config,
            user_identity=self.user_identity,
        )
        self.store.set_current_question(
            user_id=self.user_id,
            session_id=self.session_id,
            question=question["question"],
            contract=question,
        )
        self.session["current_question"] = question["question"]
        self.session["current_question_contract"] = question
        audio = await self.voice.generate_speech(question["question"])
        return {
            "session_id": self.session_id,
            "question": question["question"],
            "competency_focus": question.get("competency_focus", ""),
            "expected_signals": question.get("expected_signals", []),
            "audio_bytes": audio,
        }

    async def answer(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/webm",
        response_time_ms: int = 0,
    ) -> dict[str, Any]:
        """Transcribe a spoken answer, evaluate it, speak feedback.

        ``response_time_ms`` is the client-measured duration of
        the spoken answer (mic-start → mic-stop); it is persisted
        on the turn. The pipeline runs as a cancellable task so a
        client ``interrupt`` (barge-in) can cut it mid-flight.
        """
        self.llm_reached = False
        # The answer runs as its own task (the router wraps
        # this coroutine in one); ``interrupt`` cancels it,
        # cutting the pipeline at whatever stage is in flight.
        self._turn_task = asyncio.current_task()
        try:
            return await self._answer(
                audio_bytes,
                mime_type=mime_type,
                response_time_ms=response_time_ms,
            )
        except asyncio.CancelledError:
            return {
                "transcript": "",
                "feedback_text": "",
                "audio_bytes": b"",
                "rubric": None,
                "degraded": False,
                "completed": False,
                "report": None,
                "response_time_ms": max(0, int(response_time_ms)),
                "latency": {"stt_ms": 0, "llm_ms": 0, "tts_ms": 0, "total_ms": 0},
                "interrupted": True,
            }
        finally:
            self._turn_task = None

    def interrupt(self) -> None:
        """Cancel the in-flight answer, if any."""
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()

    async def _answer(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str = "audio/webm",
        response_time_ms: int = 0,
    ) -> dict[str, Any]:
        stt_start = time.monotonic()
        transcript = await self.voice.transcribe_audio(
            audio_bytes, mime_type=mime_type
        )
        stt_ms = int((time.monotonic() - stt_start) * 1000)
        if not transcript.strip():
            return {
                "transcript": "",
                "feedback_text": "",
                "audio_bytes": b"",
                "rubric": None,
                "degraded": False,
                "completed": False,
                "report": None,
                "response_time_ms": max(0, int(response_time_ms)),
                "latency": {
                    "stt_ms": stt_ms,
                    "llm_ms": 0,
                    "tts_ms": 0,
                    "total_ms": stt_ms,
                },
            }
        return await self._evaluate_answer(
            transcript,
            response_time_ms=response_time_ms,
            stt_ms=stt_ms,
        )

    async def answer_text(self, text: str) -> dict[str, Any]:
        """Evaluate a typed answer (text fallback for the interview)."""
        if not str(text or "").strip():
            return {
                "transcript": "",
                "feedback_text": "",
                "audio_bytes": b"",
                "rubric": None,
                "degraded": False,
                "completed": False,
                "report": None,
                "response_time_ms": 0,
                "latency": {"stt_ms": 0, "llm_ms": 0, "tts_ms": 0, "total_ms": 0},
            }
        return await self._evaluate_answer(str(text).strip())

    async def _evaluate_answer(
        self,
        user_answer: str,
        *,
        response_time_ms: int = 0,
        stt_ms: int = 0,
    ) -> dict[str, Any]:
        question = str(self.session.get("current_question") or "").strip()
        # The question's assessment contract was persisted when the
        # question was asked; legacy sessions have none and evaluate
        # against the question text only.
        question_contract = self.session.get("current_question_contract") or None
        turn_index = self.turns_completed + 1
        llm_start = time.monotonic()
        self.llm_reached = True
        eval_payload = await self.generator.evaluate_answer(
            session=self.session,
            question=question,
            user_answer=user_answer,
            turn_index=turn_index,
            llm_config=self.llm_config,
            user_identity=self.user_identity,
            question_contract=question_contract,
        )
        llm_ms = int((time.monotonic() - llm_start) * 1000)
        # Internal drift signal for the eval harness; never spoken.
        eval_payload.pop("follow_up_note_repaired", None)
        degraded = bool(eval_payload.get("degraded"))

        _turn_dict, updated_session = self.store.record_turn(
            user_id=self.user_id,
            session_id=self.session_id,
            turn_index=turn_index,
            question=question,
            user_answer=user_answer,
            rubric=eval_payload["rubric"],
            strengths=eval_payload["strengths"],
            improvements=eval_payload["improvements"],
            follow_up_note=eval_payload["follow_up_note"],
            response_time_ms=response_time_ms,
            degraded=degraded,
            contract=question_contract,
        )
        if updated_session:
            self.session = updated_session
        self.session["turns_completed"] = turn_index

        prior_memory = str(self.session.get("memory_summary") or "").strip()
        memory_update = (
            f"{prior_memory}\n"
            f"Turn {turn_index} question: {question[:220]}\n"
            f"Candidate answer summary: {user_answer[:320]}\n"
            f"Strengths: {'; '.join(eval_payload['strengths'][:2])}\n"
            f"Improvements: {'; '.join(eval_payload['improvements'][:2])}"
        ).strip()
        self.store.set_memory_summary(
            user_id=self.user_id,
            session_id=self.session_id,
            summary=memory_update[:2400],
        )
        self.session["memory_summary"] = memory_update[:2400]

        feedback_text = self._feedback_text(eval_payload)
        tts_start = time.monotonic()
        audio = await self.voice.generate_speech(feedback_text)
        tts_ms = int((time.monotonic() - tts_start) * 1000)

        report = None
        if self.turn_limit_reached:
            self.completed = True
            turns = self.store.get_turns(
                user_id=self.user_id,
                session_id=self.session_id,
            )
            report = self.generator.build_report(session=self.session, turns=turns)
            self.store.save_report(
                user_id=self.user_id,
                session_id=self.session_id,
                report=report,
            )

        return {
            "transcript": user_answer,
            "feedback_text": feedback_text,
            "audio_bytes": audio,
            "rubric": eval_payload["rubric"],
            "degraded": degraded,
            "completed": self.completed,
            "report": report,
            "response_time_ms": max(0, int(response_time_ms)),
            "latency": {
                "stt_ms": stt_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
                "total_ms": stt_ms + llm_ms + tts_ms,
            },
        }

    @staticmethod
    def _feedback_text(eval_payload: dict[str, Any]) -> str:
        rubric = eval_payload.get("rubric") or {}
        parts: list[str] = []
        overall = rubric.get("overall")
        if isinstance(overall, (int, float)):
            parts.append(f"Overall score: {overall} out of 100.")
        strengths = [
            str(s) for s in (eval_payload.get("strengths") or []) if str(s).strip()
        ]
        if strengths:
            parts.append("Strengths: " + " ".join(strengths[:2]))
        improvements = [
            str(i)
            for i in (eval_payload.get("improvements") or [])
            if str(i).strip()
        ]
        if improvements:
            parts.append("To improve: " + " ".join(improvements[:2]))
        return " ".join(parts) if parts else "Thanks for your answer."


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
