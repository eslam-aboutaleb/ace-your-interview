"""Tests for voice module — providers, session, tiers, config endpoint."""

import asyncio
import json
import os
import unittest
from unittest import mock

from app.config import get_settings
from app.schemas.models import (
    VoiceConfigResponse,
    VoiceSettingsUpdateRequest,
    VoiceTierEnum,
)
from app.services.voice_providers import (
    EdgeTTSProvider,
    GroqSTTProvider,
    NullSTTProvider,
    NullTTSProvider,
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
        self.last_system = ""
        self.last_task = None
        self.calls = 0

    async def completion(
        self,
        prompt,
        llm_config=None,
        user_identity=None,
        system="",
        task=None,
        **kwargs,
    ):
        self.calls += 1
        self.last_prompt = prompt
        self.last_system = system
        self.last_task = task
        return {
            "success": True,
            "analysis": self.response_text,
            "metadata": {"provider": "groq", "model": "test"},
            "error": "",
        }


class RecordingLLM(FakeLLM):
    """FakeLLM that records the full conversation history per call."""

    def __init__(self, response_text: str = "Test response."):
        super().__init__(response_text)
        self.prompts: list[str] = []

    async def completion(self, *args, **kwargs):
        self.prompts.append(kwargs.get("prompt", ""))
        return await super().completion(*args, **kwargs)


# ── Fake Providers ────────────────────────────────────────


class FakeSTT:
    async def transcribe(self, audio_bytes, **kwargs):
        return "hello world"


class FakeTTS:
    async def synthesize(self, text, **kwargs):
        return b"\x00\x01\x02"  # fake audio bytes


def _run(coro):
    """Drive a coroutine to completion.

    ``asyncio.get_event_loop()`` is not usable here: it raises once another test
    in the session has closed the thread's event loop.
    """
    return asyncio.run(coro)


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
        result = _run(
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
        result = _run(
            session.process_audio_turn(b"\x00" * 200)
        )
        self.assertEqual(result["transcript"], "")
        self.assertEqual(result["response_text"], "")

    def test_conversation_history_tracked(self):
        session = self._make_session()
        _run(
            session.process_audio_turn(b"\x00" * 200)
        )
        self.assertEqual(len(session.conversation_history), 2)
        self.assertEqual(session.conversation_history[0]["role"], "user")
        self.assertEqual(session.conversation_history[1]["role"], "assistant")
        self.assertEqual(session.turn_index, 1)

    def test_clear_history(self):
        session = self._make_session()
        _run(
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
        audio = _run(
            session.generate_speech("Hello")
        )
        self.assertEqual(audio, b"\x00\x01\x02")

    def test_transcribe_audio(self):
        session = self._make_session()
        text = _run(
            session.transcribe_audio(b"\x00" * 200)
        )
        self.assertEqual(text, "hello world")

    def test_stt_failure_returns_error(self):
        class FailSTT:
            async def transcribe(self, audio_bytes, **kwargs):
                raise RuntimeError("STT down")

        session = self._make_session(stt=FailSTT())
        result = _run(
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
        values = {t.value for t in VoiceTierEnum}
        self.assertIn("browser", values)
        self.assertIn("cloud", values)


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

    # ── Plan 1.6: the realtime tier is gone ────────────────

    def test_realtime_tier_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            create_voice_session(llm_client=FakeLLM(), tier="realtime")
        self.assertIn("Unknown voice tier", str(ctx.exception))
        self.assertIn("realtime", str(ctx.exception))

    def test_unknown_tier_is_rejected(self):
        for tier in ("", "nonsense", "REALTIME", "CLOUD", None):
            with self.subTest(tier=tier):
                with self.assertRaises(ValueError) as ctx:
                    create_voice_session(llm_client=FakeLLM(), tier=tier)
                self.assertIn("Unknown voice tier", str(ctx.exception))

    def test_unsupported_tier_error_is_a_value_error(self):
        from app.services.voice_session import UnsupportedVoiceTierError

        self.assertTrue(issubclass(UnsupportedVoiceTierError, ValueError))

    # ── Plan 1.6: the browser tier is genuinely inert ──────

    def test_browser_tier_uses_inert_providers(self):
        session = create_voice_session(llm_client=FakeLLM(), tier="browser")
        self.assertIsInstance(session.stt, NullSTTProvider)
        self.assertIsInstance(session.tts, NullTTSProvider)

    def test_null_stt_provider_raises_when_called(self):
        provider = NullSTTProvider()
        with self.assertRaises(RuntimeError):
            _run(
                provider.transcribe(b"\x00" * 200)
            )

    def test_null_tts_provider_raises_when_called(self):
        provider = NullTTSProvider()
        with self.assertRaises(RuntimeError):
            _run(
                provider.synthesize("hello")
            )

    def test_null_tts_provider_stream_raises_when_iterated(self):
        provider = NullTTSProvider()

        async def drain():
            async for _ in provider.synthesize_stream("hello"):
                pass

        with self.assertRaises(RuntimeError):
            _run(drain())

    def test_browser_tier_session_surfaces_stt_failure_as_error(self):
        session = create_voice_session(llm_client=FakeLLM(), tier="browser")
        result = _run(
            session.process_audio_turn(b"\x00" * 200)
        )
        self.assertIn("error", result)
        self.assertIn("browser tier", result["error"])

    # ── Plan 1.6: cloud TTS receives an api key ───────────

    def test_cloud_tier_passes_api_key_to_tts_factory(self):
        os.environ["STUDY_VOICE_TTS_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "openai-test-key"
        get_settings.cache_clear()
        try:
            with mock.patch(
                "app.services.voice_session.create_tts_provider",
                wraps=create_tts_provider,
            ) as spy:
                create_voice_session(llm_client=FakeLLM(), tier="cloud")
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(spy.call_args.kwargs.get("api_key"), "openai-test-key")
        finally:
            os.environ["STUDY_VOICE_TTS_PROVIDER"] = "edge"
            get_settings.cache_clear()

    def test_cloud_tier_edge_tts_needs_no_key(self):
        os.environ["STUDY_VOICE_TTS_PROVIDER"] = "edge"
        get_settings.cache_clear()
        session = create_voice_session(llm_client=FakeLLM(), tier="cloud")
        self.assertIsInstance(session.tts, EdgeTTSProvider)


# ── Plan 1.7 / 2.1: server-owned persona, real system role ──


class TestServerOwnedPersona(unittest.TestCase):
    def _session(self, llm, session_type: str = "chat") -> VoiceSession:
        return VoiceSession(
            llm_client=llm,
            stt=FakeSTT(),
            tts=FakeTTS(),
            session_type=session_type,
        )

    def test_default_persona_is_used_when_no_override(self):
        session = self._session(FakeLLM(), "interview")
        self.assertIn("interview coach", session.system_prompt)

    def test_server_side_override_is_accepted(self):
        session = VoiceSession(
            llm_client=FakeLLM(),
            stt=FakeSTT(),
            tts=FakeTTS(),
            session_type="chat",
            system_prompt="Server-owned override.",
        )
        self.assertEqual(session.system_prompt, "Server-owned override.")

    def test_llm_call_sends_persona_as_a_system_message(self):
        llm = FakeLLM()
        session = self._session(llm, "interview")
        _run(
            session.process_audio_turn(b"\x00" * 200)
        )
        self.assertIn("interview coach", llm.last_system)
        # The persona must not be flattened into the user turn.
        self.assertNotIn("SYSTEM:", llm.last_prompt)
        self.assertEqual(llm.last_task, "final")

    def test_history_is_trimmed_to_the_last_20_messages(self):
        llm = RecordingLLM()
        session = self._session(llm)
        session.conversation_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i}"}
            for i in range(40)
        ]
        _run(session._call_llm("ignored"))
        prompt = llm.last_prompt
        self.assertNotIn("turn-0\n", prompt)
        self.assertIn("USER: turn-20\n", prompt)
        self.assertIn("ASSISTANT: turn-39", prompt)
        # 20 history messages only.
        self.assertEqual(len(prompt.splitlines()), 20)


if __name__ == "__main__":
    unittest.main()
