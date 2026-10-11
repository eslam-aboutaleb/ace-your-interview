"""Voice router — WebSocket streaming + REST config endpoints."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from typing import Any

from app.config import get_settings
from app.dependencies import (
    is_dev_auth_bypass_enabled,
    require_admin,
    require_auth,
)
from app.schemas.models import (
    CreateInterviewSessionRequestExtended,
    LLMConfigRequest,
    VoiceConfigResponse,
    VoiceSettingsUpdateRequest,
)
from app.services.auth import decode_jwt_token
from app.services.llm_client import LLMClient
from app.services.user_settings_store import UserSettingsStore, identity_key_for_user
from app.services.voice_session import (
    VoiceInterviewSession,
    VoiceSession,
    create_voice_session,
)
from app.routers import interview_sessions as interview_sessions_router

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

_llm_client: LLMClient | None = None
_user_settings_store: UserSettingsStore | None = None

# Close code for a connection that exhausted its per-connection LLM turn budget.
# Application-defined range (4000-4999), distinct from 4001 (unauthenticated).
_TURN_BUDGET_CLOSE_CODE = 4429


def init(llm_client: LLMClient, user_settings_store: UserSettingsStore | None = None):
    global _llm_client, _user_settings_store
    _llm_client = llm_client
    _user_settings_store = user_settings_store


# ── REST Endpoints ─────────────────────────────────────────


@router.get("/config", response_model=VoiceConfigResponse)
async def get_voice_config(user: dict = Depends(require_auth)):
    """Return current voice config for the authenticated user."""
    settings = get_settings()
    enabled_tiers = [t.strip() for t in settings.voice_tiers_enabled.split(",") if t.strip()]

    # Per-user tier override: a saved preference wins when it names a
    # tier that is currently enabled; otherwise the admin default applies.
    user_tier = settings.voice_default_tier
    if _user_settings_store is not None:
        identity_key = identity_key_for_user(user)
        prefs, _ = _user_settings_store.get_user_state(identity_key)
        saved_tier = str(prefs.get("voice_tier") or "").strip()
        if saved_tier and saved_tier in enabled_tiers:
            user_tier = saved_tier

    return VoiceConfigResponse(
        enabled=settings.enable_voice_agent,
        available_tiers=enabled_tiers,
        default_tier=settings.voice_default_tier,
        user_tier=user_tier,
        stt_provider=settings.voice_stt_provider,
        tts_provider=settings.voice_tts_provider,
    )


@router.put("/settings")
async def update_voice_settings(
    body: VoiceSettingsUpdateRequest,
    user: dict = Depends(require_admin),
):
    """Admin-only: update voice agent settings (writes to env; restarts apply)."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Map request fields → config env var names
    field_to_env = {
        "enable_voice_agent": "STUDY_ENABLE_VOICE_AGENT",
        "voice_tiers_enabled": "STUDY_VOICE_TIERS_ENABLED",
        "voice_default_tier": "STUDY_VOICE_DEFAULT_TIER",
        "voice_stt_provider": "STUDY_VOICE_STT_PROVIDER",
        "voice_stt_model": "STUDY_VOICE_STT_MODEL",
        "voice_tts_provider": "STUDY_VOICE_TTS_PROVIDER",
        "voice_tts_voice": "STUDY_VOICE_TTS_VOICE",
    }

    applied = {}
    for field, value in updates.items():
        env_key = field_to_env.get(field)
        if env_key:
            str_val = str(value).lower() if isinstance(value, bool) else str(value)
            os.environ[env_key] = str_val
            applied[field] = str_val

    # Force settings to reload from env
    get_settings.cache_clear()

    return {"status": "ok", "applied": applied}


# ── WebSocket Endpoint ─────────────────────────────────────


async def _ws_authenticate(token: str) -> dict | None:
    """Validate a session token presented as a WebSocket query parameter.

    Uses the same ``decode_jwt_token`` helper as the HTTP dependency chain so
    the WebSocket and HTTP auth paths cannot diverge.
    """
    if not token:
        return None
    try:
        payload = decode_jwt_token(token)
        return {"user": payload["sub"], "provider": payload["provider"]}
    except Exception as exc:
        # WARNING + traceback: a swallowed auth failure here means every voice
        # WebSocket silently rejects every user, which must never be invisible.
        logger.warning(
            "Voice WebSocket token validation failed: %s",
            exc,
            exc_info=True,
        )
        return None


async def _ws_authenticate_from_cookie(websocket: WebSocket) -> dict | None:
    """Try to authenticate via session cookie (same as HTTP requests)."""
    token = websocket.cookies.get("session")
    if not token:
        return None
    return await _ws_authenticate(token)


async def _consume_turn_budget(
    websocket: WebSocket,
    *,
    turns_used: int,
    max_turns: int,
) -> bool:
    """Return True when the connection may spend another LLM turn.

    WebSockets bypass the HTTP rate-limit middleware, so this is the only thing
    bounding LLM spend per connection. On exhaustion it emits an error frame and
    closes with :data:`_TURN_BUDGET_CLOSE_CODE`, and returns False so the caller
    stops reading from a closed socket.
    """
    if max_turns <= 0 or turns_used < max_turns:
        return True

    await websocket.send_json({
        "type": "error",
        "message": (
            f"Turn budget exhausted: this connection has used its limit of "
            f"{max_turns} LLM turns. Reconnect to start a new session."
        ),
    })
    await websocket.close(
        code=_TURN_BUDGET_CLOSE_CODE,
        reason="LLM turn budget exhausted",
    )
    return False


async def _run_turn_interruptible(
    websocket: WebSocket,
    turn_task: "asyncio.Task[dict]",
    interrupt_target: Any,
    pending: "list[dict]",
) -> "dict | None":
    """Await a turn task while listening for an ``interrupt``.

    The turn runs as a cancellable task. While it is in
    flight, incoming messages are read: an ``interrupt``
    cancels the turn through the session's ``interrupt``
    method, and any other message is stashed in ``pending``
    for the main loop to process once the turn settles.
    Returns the turn result, or ``None`` when the turn was
    interrupted before it produced one.
    """
    reader: "asyncio.Task[str] | None" = None
    try:
        while True:
            reader = asyncio.create_task(websocket.receive_text())
            done, _ = await asyncio.wait(
                {turn_task, reader},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if turn_task in done:
                return turn_task.result()
            raw = reader.result()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {}
            if msg.get("type") == "interrupt":
                interrupt_target.interrupt()
                # The cancelled turn settles with an interrupted
                # result; await it so the task is reaped.
                try:
                    return await turn_task
                except asyncio.CancelledError:
                    return None
            pending.append(msg)
    finally:
        if reader is not None and not reader.done():
            reader.cancel()
        if not turn_task.done():
            turn_task.cancel()


@router.websocket("/stream")
async def voice_stream(
    websocket: WebSocket,
    token: str = Query(default=""),
):
    """WebSocket voice streaming endpoint.

    Protocol:
        Client → Server:
            1. {"type": "start", "tier": "cloud", "session_type": "chat",
                "session_id": "...",
                "llm_config": {"provider": "groq", "model": "..."}}
            2. {"type": "audio", "data": "<base64 audio>", "mime": "audio/webm",
                "response_time_ms": 1234}
            3. {"type": "text", "content": "typed text fallback"}
            4. {"type": "synthesize", "text": "read aloud", "stream": true}
            5. {"type": "interrupt"}  # cancel the in-flight turn (barge-in)
            6. {"type": "stop"}

        Server → Client:
            1. {"type": "ready", "session_id": "..."}
            2. {"type": "transcript", "text": "...", "final": true}
            3. {"type": "response", "text": "...", "audio": "<base64 mp3>",
                "latency": {"stt_ms": .., "llm_ms": .., "tts_ms": .., "total_ms": ..}}
            4. {"type": "audio", "audio": "<base64 chunk>", "text": "...",
                "final": false}
            5. {"type": "error", "message": "..."}

    Each connection is limited to ``settings.voice_max_turns_per_connection``
    LLM-producing turns; exceeding it sends ``{"type": "error"}`` and closes
    with code 4429.

    An ``interrupt`` cancels the in-flight turn (barge-in): the turn's
    STT/LLM/TTS pipeline runs as a cancellable task, so a client that
    starts speaking while the assistant is talking can cut the turn short.
    An interrupted turn that never reached the LLM does not consume an LLM
    turn of the per-connection budget.
    """
    settings = get_settings()

    if not settings.enable_voice_agent:
        await websocket.close(code=4003, reason="Voice agent is disabled")
        return

    # Auth: try token query param first, then cookie
    user_identity = await _ws_authenticate(token) or await _ws_authenticate_from_cookie(websocket)

    # Dev bypass — gated on the same four conditions as require_auth.
    if not user_identity and is_dev_auth_bypass_enabled(
        settings=settings,
        request_host=websocket.url.hostname or "",
        client_host=websocket.client.host if websocket.client else "",
    ):
        user_identity = {"user": "local-dev", "provider": "local"}

    if not user_identity:
        await websocket.close(code=4001, reason="Authentication required")
        return

    await websocket.accept()

    if _llm_client is None:
        await websocket.send_json({"type": "error", "message": "Service not initialised"})
        await websocket.close(code=4503)
        return

    session: VoiceSession | None = None
    interview: VoiceInterviewSession | None = None
    session_id = ""
    # HTTP middleware rate limiting does not cover WebSockets, so each connection
    # carries its own budget of LLM-producing turns (audio + text). A value <= 0
    # disables the budget.
    llm_turns_used = 0
    max_llm_turns = int(settings.voice_max_turns_per_connection or 0)
    # Messages received while a turn is in flight are stashed
    # here and processed once the turn settles.
    pending: "list[dict]" = []

    try:
        while True:
            if pending:
                msg = pending.pop(0)
            else:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

            msg_type = msg.get("type", "")

            if msg_type == "start":
                tier = msg.get("tier", settings.voice_default_tier)
                enabled_tiers = [t.strip() for t in settings.voice_tiers_enabled.split(",")]
                if tier not in enabled_tiers:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Tier '{tier}' is not enabled. Available: {enabled_tiers}",
                    })
                    continue

                session_type = msg.get("session_type", "chat")
                session_id = msg.get("session_id", str(uuid.uuid4()))
                llm_config = None
                if msg.get("llm_config"):
                    try:
                        llm_config = LLMConfigRequest(**msg["llm_config"])
                    except Exception:
                        llm_config = None

                # The persona is server-owned: a client-supplied system_prompt is
                # ignored so the wire cannot redefine what the assistant is.
                session = create_voice_session(
                    llm_client=_llm_client,
                    tier=tier,
                    session_type=session_type,
                    session_id=session_id,
                    user_identity=user_identity,
                    llm_config=llm_config,
                )

                await websocket.send_json({
                    "type": "ready",
                    "session_id": session_id,
                    "tier": tier,
                })

            elif msg_type == "interview_start":
                if not get_settings().enable_voice_interview_v1:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Voice interviews are disabled.",
                    })
                    continue

                if not await _consume_turn_budget(
                    websocket,
                    turns_used=llm_turns_used,
                    max_turns=max_llm_turns,
                ):
                    return
                llm_turns_used += 1

                services = interview_sessions_router.get_interview_services()
                if services is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Interview services not initialised.",
                    })
                    continue
                store, generator = services

                config_payload = msg.get("config") or {}
                try:
                    config = CreateInterviewSessionRequestExtended(**config_payload)
                except Exception as exc:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid interview config: {exc}",
                    })
                    continue

                resume_summary_text, focus_areas = (
                    await interview_sessions_router.personalize_session_inputs(
                        config,
                        user_id=user_identity["user"],
                        llm_config=config.llm_config,
                        user_identity=user_identity,
                    )
                )
                job_description_text = config.job_description_text.strip()
                if not job_description_text:
                    job_description_text = config.jd_text.strip()

                session_row = store.create_session(
                    user_id=user_identity["user"],
                    track=config.track.value,
                    level=(
                        config.level.value
                        if config.level is not None
                        else generator.resolve_session_level(
                            track=config.track.value,
                            user_identity=user_identity,
                        )
                    ),
                    interview_type=config.interview_type.value,
                    turn_count=config.turn_count,
                    target_role=config.target_role.strip(),
                    interviewer_style=config.interviewer_style.value,
                    feedback_mode=config.feedback_mode.value,
                    job_description_text=job_description_text,
                    resume_summary_text=resume_summary_text,
                    focus_areas=focus_areas,
                    company=config.company,
                )

                interview_tier = msg.get("tier", settings.voice_default_tier)
                enabled_tiers = [
                    t.strip() for t in settings.voice_tiers_enabled.split(",")
                ]
                if interview_tier not in enabled_tiers:
                    await websocket.send_json({
                        "type": "error",
                        "message": (
                            f"Tier '{interview_tier}' is not enabled. "
                            f"Available: {enabled_tiers}"
                        ),
                    })
                    continue

                voice_session = create_voice_session(
                    llm_client=_llm_client,
                    tier=interview_tier,
                    session_type="interview",
                    session_id=session_row["session_id"],
                    user_identity=user_identity,
                    llm_config=config.llm_config,
                )
                interview = VoiceInterviewSession(
                    voice_session=voice_session,
                    store=store,
                    generator=generator,
                    session=session_row,
                    user_id=user_identity["user"],
                    user_identity=user_identity,
                    llm_config=config.llm_config,
                )

                try:
                    started = await interview.start()
                except Exception as exc:
                    logger.error(
                        "Voice interview start failed: %s", exc, exc_info=True
                    )
                    interview = None
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Interview start failed: {exc}",
                    })
                    continue

                started_audio_b64 = (
                    base64.b64encode(started["audio_bytes"]).decode()
                    if started["audio_bytes"]
                    else ""
                )
                await websocket.send_json({
                    "type": "interview_ready",
                    "session_id": started["session_id"],
                    "question": started["question"],
                    "competency_focus": started["competency_focus"],
                    "expected_signals": started["expected_signals"],
                    "audio": started_audio_b64,
                    "turn_count": interview.turn_count,
                })

            elif msg_type == "audio":
                if interview is not None:
                    # Voice interview: a spoken answer is transcribed,
                    # evaluated, recorded, and spoken back as feedback.
                    audio_b64 = msg.get("data", "")
                    mime = msg.get("mime", "audio/webm")
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                    except Exception:
                        await websocket.send_json({"type": "error", "message": "Invalid base64 audio"})
                        continue

                    if len(audio_bytes) < 100:
                        await websocket.send_json({"type": "error", "message": "Audio too short"})
                        continue

                    if not await _consume_turn_budget(
                        websocket,
                        turns_used=llm_turns_used,
                        max_turns=max_llm_turns,
                    ):
                        return
                    llm_turns_used += 1

                    response_time_ms = int(msg.get("response_time_ms") or 0)
                    turn_task = asyncio.create_task(
                        interview.answer(
                            audio_bytes,
                            mime_type=mime,
                            response_time_ms=response_time_ms,
                        )
                    )
                    result = await _run_turn_interruptible(
                        websocket, turn_task, interview, pending
                    )
                    # An interrupted turn that never reached the
                    # evaluation LLM did not spend an LLM turn.
                    if (
                        result is not None
                        and result.get("interrupted")
                        and not interview.llm_reached
                    ):
                        llm_turns_used -= 1
                    if result is None or result.get("interrupted"):
                        # Barge-in cut the turn short; the client
                        # is already recording the next answer.
                        continue

                    if result["transcript"]:
                        await websocket.send_json({
                            "type": "transcript",
                            "text": result["transcript"],
                            "final": True,
                        })

                    answer_audio_b64 = (
                        base64.b64encode(result["audio_bytes"]).decode()
                        if result["audio_bytes"]
                        else ""
                    )
                    latency = result.get("latency") or {}
                    await websocket.send_json({
                        "type": "interview_response",
                        "text": result["feedback_text"],
                        "audio": answer_audio_b64,
                        "rubric": result["rubric"],
                        "degraded": result["degraded"],
                        "completed": result["completed"],
                        "response_time_ms": result.get("response_time_ms", 0),
                        "latency": {
                            "stt_ms": latency.get("stt_ms", 0),
                            "llm_ms": latency.get("llm_ms", 0),
                            "tts_ms": latency.get("tts_ms", 0),
                            "total_ms": latency.get("total_ms", 0),
                        },
                    })

                    if result["completed"] and result["report"]:
                        await websocket.send_json({
                            "type": "interview_completed",
                            "session_id": interview.session_id,
                            "report": result["report"],
                        })
                        interview = None
                    continue

                if session is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Session not started. Send {type: 'start'} first.",
                    })
                    continue

                audio_b64 = msg.get("data", "")
                mime = msg.get("mime", "audio/webm")
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Invalid base64 audio"})
                    continue

                if len(audio_bytes) < 100:
                    await websocket.send_json({"type": "error", "message": "Audio too short"})
                    continue

                if not await _consume_turn_budget(
                    websocket,
                    turns_used=llm_turns_used,
                    max_turns=max_llm_turns,
                ):
                    return

                llm_turns_used += 1

                turn_task = asyncio.create_task(
                    session.process_audio_turn(audio_bytes, mime_type=mime)
                )
                result = await _run_turn_interruptible(
                    websocket, turn_task, session, pending
                )
                # An interrupted turn that never reached the LLM
                # did not spend an LLM turn.
                if (
                    result is not None
                    and result.get("interrupted")
                    and not session.llm_reached
                ):
                    llm_turns_used -= 1
                if result is None or result.get("interrupted"):
                    continue

                if result.get("error"):
                    await websocket.send_json({"type": "error", "message": result["error"]})
                    continue

                # Send transcript first so UI can show it immediately
                if result["transcript"]:
                    await websocket.send_json({
                        "type": "transcript",
                        "text": result["transcript"],
                        "final": True,
                    })

                # Send response with audio
                audio_out_b64 = ""
                if result["audio_bytes"]:
                    audio_out_b64 = base64.b64encode(result["audio_bytes"]).decode()

                await websocket.send_json({
                    "type": "response",
                    "text": result["response_text"],
                    "audio": audio_out_b64,
                    "latency": {
                        "stt_ms": result["stt_latency_ms"],
                        "llm_ms": result["llm_latency_ms"],
                        "tts_ms": result["tts_latency_ms"],
                        "total_ms": result["total_latency_ms"],
                    },
                })

            elif msg_type == "text":
                # Text fallback — user typed instead of speaking
                if interview is not None:
                    text = msg.get("content", "").strip()
                    if not text:
                        continue

                    if not await _consume_turn_budget(
                        websocket,
                        turns_used=llm_turns_used,
                        max_turns=max_llm_turns,
                    ):
                        return
                    llm_turns_used += 1

                    result = await interview.answer_text(text)

                    answer_audio_b64 = (
                        base64.b64encode(result["audio_bytes"]).decode()
                        if result["audio_bytes"]
                        else ""
                    )
                    latency = result.get("latency") or {}
                    await websocket.send_json({
                        "type": "interview_response",
                        "text": result["feedback_text"],
                        "audio": answer_audio_b64,
                        "rubric": result["rubric"],
                        "degraded": result["degraded"],
                        "completed": result["completed"],
                        "response_time_ms": result.get("response_time_ms", 0),
                        "latency": {
                            "stt_ms": latency.get("stt_ms", 0),
                            "llm_ms": latency.get("llm_ms", 0),
                            "tts_ms": latency.get("tts_ms", 0),
                            "total_ms": latency.get("total_ms", 0),
                        },
                    })

                    if result["completed"] and result["report"]:
                        await websocket.send_json({
                            "type": "interview_completed",
                            "session_id": interview.session_id,
                            "report": result["report"],
                        })
                        interview = None
                    continue

                if session is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Session not started. Send {type: 'start'} first.",
                    })
                    continue

                text = msg.get("content", "").strip()
                if not text:
                    continue

                if not await _consume_turn_budget(
                    websocket,
                    turns_used=llm_turns_used,
                    max_turns=max_llm_turns,
                ):
                    return

                llm_turns_used += 1

                session.conversation_history.append({"role": "user", "content": text})

                llm_start = time.monotonic()
                response_text = await session._call_llm(text)
                llm_ms = int((time.monotonic() - llm_start) * 1000)

                session.conversation_history.append({"role": "assistant", "content": response_text})
                session.turn_index += 1

                # Optionally synthesize
                audio_out_b64 = ""
                tts_ms = 0
                if msg.get("speak", True):
                    tts_start = time.monotonic()
                    audio_bytes = await session.generate_speech(response_text)
                    tts_ms = int((time.monotonic() - tts_start) * 1000)
                    if audio_bytes:
                        audio_out_b64 = base64.b64encode(audio_bytes).decode()

                await websocket.send_json({
                    "type": "response",
                    "text": response_text,
                    "audio": audio_out_b64,
                    "latency": {
                        "stt_ms": 0,
                        "llm_ms": llm_ms,
                        "tts_ms": tts_ms,
                        "total_ms": llm_ms + tts_ms,
                    },
                })

            elif msg_type == "synthesize":
                # TTS-only: speak arbitrary text (e.g., reading a question aloud)
                if session is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Session not started.",
                    })
                    continue

                text = msg.get("text", "").strip()
                if not text:
                    continue

                if msg.get("stream"):
                    # Chunked playback: stream TTS audio as it is
                    # synthesized so an interrupt can cut it
                    # mid-utterance. A final (possibly empty)
                    # frame always terminates the stream.
                    streamed = False
                    try:
                        async for chunk in session.generate_speech_stream(text):
                            if not chunk:
                                continue
                            streamed = True
                            await websocket.send_json({
                                "type": "audio",
                                "audio": base64.b64encode(chunk).decode(),
                                "text": text,
                                "final": False,
                            })
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.error("Streaming synthesis failed: %s", exc)
                    await websocket.send_json({
                        "type": "audio",
                        "audio": "",
                        "text": text,
                        "final": True,
                        "streamed": streamed,
                    })
                    continue

                audio_bytes = await session.generate_speech(text)
                audio_out_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else ""
                await websocket.send_json({
                    "type": "audio",
                    "audio": audio_out_b64,
                    "text": text,
                })

            elif msg_type == "interrupt":
                # Cancel the in-flight turn (barge-in). During
                # an active turn the interrupt is consumed by
                # _run_turn_interruptible; this handler covers
                # the race where the turn has already settled,
                # and defensively cancels any stray task.
                if session is not None:
                    session.interrupt()
                if interview is not None:
                    interview.interrupt()

            elif msg_type == "stop":
                if session:
                    session.clear_history()
                    session = None
                interview = None
                await websocket.send_json({"type": "stopped", "session_id": session_id})

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected: user=%s session=%s", user_identity.get("user"), session_id)
    except Exception as e:
        logger.error("Voice WebSocket error: %s", e, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": "Internal server error"})
            await websocket.close(code=4500)
        except Exception:
            pass
