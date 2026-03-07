"""Tests for voice module — providers, session, config endpoint."""

import asyncio
import json
import os
import unittest

from app.config import get_settings
from app.schemas.models import (
    VoiceConfigResponse,
    VoiceSettingsUpdateRequest,
    VoiceTierEnum,
)
from app.services.voice_providers import (
    EdgeTTSProvider,
    GroqSTTProvider,
    OpenAISTTProvider,
    OpenAITTSProvider,
    create_stt_provider,
    create_tts_provider,
    pcm_to_wav,
)
from app.services.voice_session import VoiceSession, create_voice_session


# ── Fake LLM Client ──────────────────────────────────────


class FakeLLM:
    def __init__(self, response_text: str = "Test response."):
        self.response_text = response_text
        self.last_prompt = ""

    async def completion(self, prompt, llm_config=None, user_identity=None):
        self.last_prompt = prompt
        return {
            "success": True,
            "analysis": self.response_text,
            "metadata": {"provider": "groq", "model": "test"},
            "error": "",
        }


# ── Fake Providers ────────────────────────────────────────


class FakeSTT:
    async def transcribe(self, audio_bytes, **kwargs):
        return "hello world"


class FakeTTS:
    async def synthesize(self, text, **kwargs):
        return b"\x00\x01\x02"  # fake audio bytes


# ── Provider Factory Tests ────────────────────────────────


class TestProviderFactory(unittest.TestCase):
    def test_create_stt_groq(self):
        provider = create_stt_provider("groq", api_key="test-key")
        self.assertIsInstance(provider, GroqSTTProvider)

    def test_create_stt_openai(self):
        provider = create_stt_provider("openai", api_key="test-key")
        self.assertIsInstance(provider, OpenAISTTProvider)

    def test_create_stt_unknown_defaults_to_groq(self):
        provider = create_stt_provider("unknown", api_key="test-key")
        self.assertIsInstance(provider, GroqSTTProvider)

    def test_create_tts_edge(self):
        provider = create_tts_provider("edge", voice="en-US-AriaNeural")
        self.assertIsInstance(provider, EdgeTTSProvider)

    def test_create_tts_openai(self):
        provider = create_tts_provider("openai", voice="alloy", api_key="test-key")
        self.assertIsInstance(provider, OpenAITTSProvider)

    def test_create_tts_unknown_defaults_to_edge(self):
        provider = create_tts_provider("unknown", voice="test")
        self.assertIsInstance(provider, EdgeTTSProvider)


# ── PCM to WAV Utility ───────────────────────────────────


class TestPCMtoWAV(unittest.TestCase):
    def test_pcm_to_wav_header(self):
        pcm_data = b"\x00" * 100
        wav = pcm_to_wav(pcm_data, sample_rate=16000, channels=1, sample_width=2)
        # WAV files start with RIFF header
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav[:12])
        # Should be larger than input (has 44-byte header)
        self.assertGreater(len(wav), len(pcm_data))


# ── Voice Session Tests ──────────────────────────────────


class TestVoiceSession(unittest.TestCase):
    def _make_session(self, **kwargs) -> VoiceSession:
        llm = kwargs.pop("llm", FakeLLM())
        stt = kwargs.pop("stt", FakeSTT())
        tts = kwargs.pop("tts", FakeTTS())
        return VoiceSession(
            llm_client=llm,
            stt=stt,
            tts=tts,
            session_type=kwargs.pop("session_type", "chat"),
            **kwargs,
        )

    def test_process_audio_turn_success(self):
        session = self._make_session()
        result = asyncio.get_event_loop().run_until_complete(
            session.process_audio_turn(b"\x00" * 200, mime_type="audio/webm")
        )
        self.assertEqual(result["transcript"], "hello world")
        self.assertEqual(result["response_text"], "Test response.")
        self.assertEqual(result["audio_bytes"], b"\x00\x01\x02")
        self.assertGreaterEqual(result["total_latency_ms"], 0)

    def test_process_audio_turn_empty_transcript(self):
        class EmptySTT:
            async def transcribe(self, audio_bytes, **kwargs):
                return ""

        session = self._make_session(stt=EmptySTT())
        result = asyncio.get_event_loop().run_until_complete(
            session.process_audio_turn(b"\x00" * 200)
        )
        self.assertEqual(result["transcript"], "")
        self.assertEqual(result["response_text"], "")

    def test_conversation_history_tracked(self):
        session = self._make_session()
        asyncio.get_event_loop().run_until_complete(
            session.process_audio_turn(b"\x00" * 200)
        )
        self.assertEqual(len(session.conversation_history), 2)
        self.assertEqual(session.conversation_history[0]["role"], "user")
        self.assertEqual(session.conversation_history[1]["role"], "assistant")
        self.assertEqual(session.turn_index, 1)

    def test_clear_history(self):
        session = self._make_session()
        asyncio.get_event_loop().run_until_complete(
            session.process_audio_turn(b"\x00" * 200)
        )
        session.clear_history()
        self.assertEqual(len(session.conversation_history), 0)
        self.assertEqual(session.turn_index, 0)

    def test_set_system_prompt(self):
        session = self._make_session()
        session.set_system_prompt("Custom prompt")
        self.assertEqual(session.system_prompt, "Custom prompt")

    def test_default_system_prompts(self):
        for stype in ("chat", "interview", "qa"):
            session = self._make_session(session_type=stype)
            self.assertTrue(len(session.system_prompt) > 10)

    def test_generate_speech(self):
        session = self._make_session()
        audio = asyncio.get_event_loop().run_until_complete(
            session.generate_speech("Hello")
        )
        self.assertEqual(audio, b"\x00\x01\x02")

    def test_transcribe_audio(self):
        session = self._make_session()
        text = asyncio.get_event_loop().run_until_complete(
            session.transcribe_audio(b"\x00" * 200)
        )
        self.assertEqual(text, "hello world")

    def test_stt_failure_returns_error(self):
        class FailSTT:
            async def transcribe(self, audio_bytes, **kwargs):
                raise RuntimeError("STT down")

        session = self._make_session(stt=FailSTT())
        result = asyncio.get_event_loop().run_until_complete(
            session.process_audio_turn(b"\x00" * 200)
        )
        self.assertIn("error", result)
        self.assertIn("STT down", result["error"])


# ── Schema Tests ─────────────────────────────────────────


class TestVoiceSchemas(unittest.TestCase):
    def test_voice_config_response_construction(self):
        cfg = VoiceConfigResponse(
            enabled=True,
            available_tiers=["browser", "cloud"],
            default_tier="browser",
            user_tier="cloud",
            stt_provider="groq",
            tts_provider="edge",
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(len(cfg.available_tiers), 2)

    def test_voice_settings_update_partial(self):
        req = VoiceSettingsUpdateRequest(enable_voice_agent=True)
        dump = req.model_dump(exclude_none=True)
        self.assertIn("enable_voice_agent", dump)
        self.assertNotIn("voice_tts_voice", dump)

    def test_voice_tier_enum(self):
        self.assertEqual(VoiceTierEnum.BROWSER.value, "browser")
        self.assertEqual(VoiceTierEnum.CLOUD.value, "cloud")
        self.assertEqual(VoiceTierEnum.REALTIME.value, "realtime")


# ── Create Voice Session Factory ─────────────────────────


class TestCreateVoiceSession(unittest.TestCase):
    def test_create_browser_session(self):
        os.environ.setdefault("GROQ_API_KEY", "test")
        session = create_voice_session(
            llm_client=FakeLLM(),
            tier="browser",
            session_type="chat",
        )
        self.assertIsInstance(session, VoiceSession)
        self.assertEqual(session.session_type, "chat")

    def test_create_cloud_session(self):
        os.environ.setdefault("GROQ_API_KEY", "test")
        session = create_voice_session(
            llm_client=FakeLLM(),
            tier="cloud",
            session_type="interview",
        )
        self.assertIsInstance(session, VoiceSession)
        self.assertEqual(session.session_type, "interview")


if __name__ == "__main__":
    unittest.main()
