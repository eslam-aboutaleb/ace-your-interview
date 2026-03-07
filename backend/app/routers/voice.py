"""Voice router — WebSocket streaming + REST config endpoints."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.dependencies import is_admin_identity, require_admin, require_auth
from app.schemas.models import (
    LLMConfigRequest,
    VoiceConfigResponse,
    VoiceSettingsUpdateRequest,
    VoiceTierEnum,
)
from app.services.llm_client import LLMClient
from app.services.voice_session import VoiceSession, create_voice_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

_llm_client: LLMClient | None = None


def init(llm_client: LLMClient):
    global _llm_client
    _llm_client = llm_client


# ── REST Endpoints ─────────────────────────────────────────


@router.get("/config", response_model=VoiceConfigResponse)
async def get_voice_config(user: dict = Depends(require_auth)):
    """Return current voice config for the authenticated user."""
    settings = get_settings()
    enabled_tiers = [t.strip() for t in settings.voice_tiers_enabled.split(",") if t.strip()]

    return VoiceConfigResponse(
        enabled=settings.enable_voice_agent,
        available_tiers=enabled_tiers,
        default_tier=settings.voice_default_tier,
        user_tier=settings.voice_default_tier,  # TODO: per-user override from UserSettingsStore
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
    """Validate session token from WebSocket query params.

    Re-uses the same JWT validation logic as require_auth but works
    outside of the Depends() chain.
    """
    if not token:
        return None
    try:
        import jwt as pyjwt
        settings = get_settings()
        payload = pyjwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return {"user": payload.get("sub", ""), "provider": payload.get("provider", "")}
    except Exception:
        return None


async def _ws_authenticate_from_cookie(websocket: WebSocket) -> dict | None:
    """Try to authenticate via session cookie (same as HTTP requests)."""
    token = websocket.cookies.get("session")
    if not token:
        return None
    return await _ws_authenticate(token)


@router.websocket("/stream")
async def voice_stream(
    websocket: WebSocket,
    token: str = Query(default=""),
):
    """WebSocket voice streaming endpoint.

    Protocol:
        Client → Server:
            1. {"type": "start", "tier": "cloud", "session_type": "chat",
                "session_id": "...", "system_prompt": "...",
                "llm_config": {"provider": "groq", "model": "..."}}
            2. {"type": "audio", "data": "<base64 audio>", "mime": "audio/webm"}
            3. {"type": "text", "content": "typed text fallback"}
            4. {"type": "stop"}

        Server → Client:
            1. {"type": "ready", "session_id": "..."}
            2. {"type": "transcript", "text": "...", "final": true}
            3. {"type": "response", "text": "...", "audio": "<base64 mp3>",
                "latency": {"stt_ms": .., "llm_ms": .., "tts_ms": .., "total_ms": ..}}
            4. {"type": "error", "message": "..."}
    """
    settings = get_settings()

    if not settings.enable_voice_agent:
        await websocket.close(code=4003, reason="Voice agent is disabled")
        return

    # Auth: try token query param first, then cookie
    user_identity = await _ws_authenticate(token) or await _ws_authenticate_from_cookie(websocket)

    # Dev bypass
    if not user_identity and settings.environment.strip().lower() == "development":
        user_identity = {"user": "dev-user", "provider": "dev"}

    if not user_identity:
        await websocket.close(code=4001, reason="Authentication required")
        return

    await websocket.accept()

    if _llm_client is None:
        await websocket.send_json({"type": "error", "message": "Service not initialised"})
        await websocket.close(code=4503)
        return

    session: VoiceSession | None = None
    session_id = ""

    try:
        while True:
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
                system_prompt = msg.get("system_prompt", "")
                llm_config = None
                if msg.get("llm_config"):
                    try:
                        llm_config = LLMConfigRequest(**msg["llm_config"])
                    except Exception:
                        llm_config = None

                session = create_voice_session(
                    llm_client=_llm_client,
                    tier=tier,
                    session_type=session_type,
                    session_id=session_id,
                    user_identity=user_identity,
                    llm_config=llm_config,
                    system_prompt=system_prompt,
                )

                await websocket.send_json({
                    "type": "ready",
                    "session_id": session_id,
                    "tier": tier,
                })

            elif msg_type == "audio":
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

                result = await session.process_audio_turn(audio_bytes, mime_type=mime)

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
                if session is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Session not started. Send {type: 'start'} first.",
                    })
                    continue

                text = msg.get("content", "").strip()
                if not text:
                    continue

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

                audio_bytes = await session.generate_speech(text)
                audio_out_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else ""
                await websocket.send_json({
                    "type": "audio",
                    "audio": audio_out_b64,
                    "text": text,
                })

            elif msg_type == "stop":
                if session:
                    session.clear_history()
                    session = None
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
