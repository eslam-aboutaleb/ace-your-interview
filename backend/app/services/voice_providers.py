"""Voice STT and TTS provider implementations.

Three tiers:
  - browser:  No server-side audio processing; handled entirely in the browser.
  - cloud:    Groq Whisper (STT) + Edge TTS (TTS) – near-zero cost.
  - realtime: OpenAI Realtime API – bidirectional WebSocket audio proxy.
"""

from __future__ import annotations

import io
import logging
import struct
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


# ── STT Providers ────────────────────────────────────────────

class STTProvider(ABC):
    """Base class for speech-to-text providers."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, *, mime_type: str = "audio/webm") -> str:
        """Convert audio bytes to text transcript."""


class GroqSTTProvider(STTProvider):
    """Groq-hosted Whisper – free tier, high quality."""

    def __init__(self, api_key: str, model: str = "whisper-large-v3"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"

    async def transcribe(self, audio_bytes: bytes, *, mime_type: str = "audio/webm") -> str:
        ext_map = {
            "audio/webm": "audio.webm",
            "audio/wav": "audio.wav",
            "audio/mp4": "audio.mp4",
            "audio/ogg": "audio.ogg",
            "audio/mpeg": "audio.mp3",
        }
        filename = ext_map.get(mime_type, "audio.webm")

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (filename, io.BytesIO(audio_bytes), mime_type)},
                data={"model": self.model, "response_format": "text", "language": "en"},
            )
            resp.raise_for_status()
            return resp.text.strip()


class OpenAISTTProvider(STTProvider):
    """OpenAI Whisper STT – pay-per-use, highest accuracy."""

    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"

    async def transcribe(self, audio_bytes: bytes, *, mime_type: str = "audio/webm") -> str:
        ext_map = {
            "audio/webm": "audio.webm",
            "audio/wav": "audio.wav",
            "audio/mp4": "audio.mp4",
            "audio/ogg": "audio.ogg",
            "audio/mpeg": "audio.mp3",
        }
        filename = ext_map.get(mime_type, "audio.webm")

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (filename, io.BytesIO(audio_bytes), mime_type)},
                data={"model": self.model, "response_format": "text", "language": "en"},
            )
            resp.raise_for_status()
            return resp.text.strip()


# ── TTS Providers ────────────────────────────────────────────

class TTSProvider(ABC):
    """Base class for text-to-speech providers."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (MP3)."""

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks as they're generated."""


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS – completely free, high quality.

    Uses the edge-tts Python package which connects to Microsoft's
    public speech synthesis endpoint.
    """

    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice

    async def synthesize(self, text: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]


class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS – pay-per-use, premium voices."""

    def __init__(self, api_key: str, model: str = "tts-1", voice: str = "alloy"):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.base_url = "https://api.openai.com/v1"

    async def synthesize(self, text: str) -> bytes:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/audio/speech",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": text,
                    "voice": self.voice,
                    "response_format": "mp3",
                },
            )
            resp.raise_for_status()
            return resp.content

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/audio/speech",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": text,
                    "voice": self.voice,
                    "response_format": "mp3",
                },
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(4096):
                    yield chunk


# ── Factory ──────────────────────────────────────────────────

def create_stt_provider(provider: str, api_key: str, model: str = "") -> STTProvider:
    """Create an STT provider based on configuration."""
    if provider == "openai":
        return OpenAISTTProvider(api_key=api_key, model=model or "whisper-1")
    # Default to Groq (free tier)
    return GroqSTTProvider(api_key=api_key, model=model or "whisper-large-v3")


def create_tts_provider(provider: str, voice: str = "", api_key: str = "") -> TTSProvider:
    """Create a TTS provider based on configuration."""
    if provider == "openai":
        return OpenAITTSProvider(api_key=api_key, voice=voice or "alloy")
    # Default to Edge TTS (free)
    return EdgeTTSProvider(voice=voice or "en-US-AriaNeural")


# ── Audio Utilities ──────────────────────────────────────────

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Convert raw PCM bytes to WAV format."""
    buf = io.BytesIO()
    data_size = len(pcm_bytes)
    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))   # PCM format
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * channels * sample_width))  # byte rate
    buf.write(struct.pack("<H", channels * sample_width))  # block align
    buf.write(struct.pack("<H", sample_width * 8))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_bytes)
    return buf.getvalue()
