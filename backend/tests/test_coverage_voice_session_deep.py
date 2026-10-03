"""Coverage for app/services/voice_session.py.

The orchestrator's contract is that a failure in *any* of the three stages
degrades one turn rather than raising into the WebSocket handler. These tests
pin each degradation path: STT failure short-circuits before the LLM, an LLM
failure still produces a speakable apology, and a TTS failure yields text with
no audio. They also pin the persona selection per session type and the
server-side-only ``system_prompt`` override.

No provider is constructed: every test injects fakes, so nothing here can open
a socket.
"""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.llm_client import BUDGET_EXCEEDED_CODE
from app.services.voice_providers import NullSTTProvider, NullTTSProvider
from app.services import voice_session
from app.services.voice_session import (
    UnsupportedVoiceTierError,
    VoiceSession,
    create_voice_session,
)


class RecordingLLM:
    def __init__(self, analysis="A clear answer.", success=True, error_code="", boom=False):
        self.analysis = analysis
        self.success = success
        self.error_code = error_code
        self.boom = boom
        self.calls: list[dict] = []

    async def completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.boom:
            raise RuntimeError("llm exploded")
        return {
            "success": self.success,
            "analysis": self.analysis,
            "metadata": {},
            "error": "" if self.success else "failed",
            "error_code": self.error_code,
        }


class FakeSTT:
    def __init__(self, transcript="Tell me about idempotency.", boom=False):
        self.transcript = transcript
        self.boom = boom
        self.calls: list[tuple[int, str]] = []

    async def transcribe(self, audio_bytes: bytes, *, mime_type: str = "audio/webm") -> str:
        self.calls.append((len(audio_bytes), mime_type))
        if self.boom:
            raise RuntimeError("stt provider unreachable")
        return self.transcript


class FakeTTS:
    def __init__(self, audio=b"ID3-mp3", boom=False):
        self.audio = audio
        self.boom = boom
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.spoken.append(text)
        if self.boom:
            raise RuntimeError("tts quota exhausted")
        return self.audio


def _session(llm=None, stt=None, tts=None, **kwargs) -> VoiceSession:
    return VoiceSession(
        llm_client=llm or RecordingLLM(),
        stt=stt or FakeSTT(),
        tts=tts or FakeTTS(),
        **kwargs,
    )


class DefaultPersonaTests(unittest.TestCase):
    def test_each_session_type_gets_its_own_default_persona(self):
        for session_type, marker in (
            ("interview", "mock interview"),
            ("qa", "study tutor"),
            ("chat", "study assistant"),
        ):
            with self.subTest(session_type=session_type):
                self.assertIn(marker, _session(session_type=session_type).system_prompt)

    def test_an_explicit_server_side_persona_wins_over_the_default(self):
        session = _session(session_type="chat", system_prompt="You are a Rust mentor.")
        self.assertEqual(session.system_prompt, "You are a Rust mentor.")

    def test_set_system_prompt_replaces_the_persona_mid_session(self):
        session = _session()
        session.set_system_prompt("You are a Go mentor.")
        self.assertEqual(session.system_prompt, "You are a Go mentor.")

    def test_the_persona_is_sent_as_a_real_system_message(self):
        llm = RecordingLLM()
        session = _session(llm=llm, session_type="qa", system_prompt="You are a Rust mentor.")
        asyncio.run(session.process_audio_turn(b"audio"))
        self.assertEqual(llm.calls[0]["system"], "You are a Rust mentor.")
        self.assertEqual(llm.calls[0]["task"], "final")
        self.assertEqual(llm.calls[0]["flow"], "voice_turn")
        # The persona must not be flattened into the user turn.
        self.assertNotIn("You are a Rust mentor", llm.calls[0]["prompt"])


class ProcessAudioTurnTests(unittest.TestCase):
    def test_a_complete_turn_reports_every_stage_latency(self):
        llm = RecordingLLM(analysis="  Idempotency means the retry is safe.  ")
        stt = FakeSTT()
        tts = FakeTTS()
        out = asyncio.run(_session(llm=llm, stt=stt, tts=tts).process_audio_turn(b"12345"))

        self.assertEqual(out["transcript"], "Tell me about idempotency.")
        self.assertEqual(out["response_text"], "Idempotency means the retry is safe.")
        self.assertEqual(out["audio_bytes"], b"ID3-mp3")
        for key in ("stt_latency_ms", "llm_latency_ms", "tts_latency_ms", "total_latency_ms"):
            self.assertIsInstance(out[key], int, key)
        self.assertGreaterEqual(out["total_latency_ms"], out["stt_latency_ms"])
        # The trimmed response is what gets spoken.
        self.assertEqual(tts.spoken, ["Idempotency means the retry is safe."])
        self.assertEqual(stt.calls, [(5, "audio/webm")])

    def test_the_turn_appends_both_sides_to_the_conversation_history(self):
        session = _session()
        asyncio.run(session.process_audio_turn(b"audio"))
        self.assertEqual(
            session.conversation_history,
            [
                {"role": "user", "content": "Tell me about idempotency."},
                {"role": "assistant", "content": "A clear answer."},
            ],
        )
        self.assertEqual(session.turn_index, 1)

    def test_clear_history_resets_the_turn_counter(self):
        session = _session()
        asyncio.run(session.process_audio_turn(b"audio"))
        session.clear_history()
        self.assertEqual(session.conversation_history, [])
        self.assertEqual(session.turn_index, 0)

    def test_a_stt_outage_short_circuits_before_the_llm_is_spent(self):
        llm = RecordingLLM()
        tts = FakeTTS()
        out = asyncio.run(
            _session(llm=llm, stt=FakeSTT(boom=True), tts=tts).process_audio_turn(b"audio")
        )
        self.assertIn("stt provider unreachable", out["error"])
        self.assertTrue(out["error"].startswith("Speech recognition failed:"))
        self.assertEqual(out["transcript"], "")
        self.assertEqual(out["audio_bytes"], b"")
        self.assertEqual(llm.calls, [])
        self.assertEqual(tts.spoken, [])

    def test_a_blank_transcription_skips_the_llm_and_tts_entirely(self):
        llm = RecordingLLM()
        tts = FakeTTS()
        out = asyncio.run(
            _session(llm=llm, stt=FakeSTT(transcript="   \n "), tts=tts).process_audio_turn(b"a")
        )
        self.assertEqual(out["transcript"], "")
        self.assertEqual(out["response_text"], "")
        self.assertEqual(out["audio_bytes"], b"")
        self.assertEqual(out["llm_latency_ms"], 0)
        self.assertEqual(out["tts_latency_ms"], 0)
        self.assertEqual(llm.calls, [])

    def test_a_raised_llm_error_becomes_a_speakable_apology(self):
        tts = FakeTTS()
        out = asyncio.run(
            _session(llm=RecordingLLM(boom=True), tts=tts).process_audio_turn(b"audio")
        )
        self.assertIn("trouble processing", out["response_text"])
        self.assertEqual(out["audio_bytes"], b"ID3-mp3")
        self.assertEqual(tts.spoken, [out["response_text"]])

    def test_an_unsuccessful_llm_result_becomes_a_speakable_apology(self):
        llm = RecordingLLM(success=False, error_code=BUDGET_EXCEEDED_CODE)
        out = asyncio.run(_session(llm=llm).process_audio_turn(b"audio"))
        self.assertEqual(
            out["response_text"],
            "I'm sorry, I couldn't generate a response. Could you try again?",
        )
        self.assertEqual(out["audio_bytes"], b"ID3-mp3")

    def test_a_tts_outage_still_returns_the_transcript_and_reply_text(self):
        out = asyncio.run(_session(tts=FakeTTS(boom=True)).process_audio_turn(b"audio"))
        self.assertEqual(out["response_text"], "A clear answer.")
        self.assertEqual(out["audio_bytes"], b"")
        self.assertGreaterEqual(out["tts_latency_ms"], 0)

    def test_a_custom_mime_type_is_forwarded_to_the_transcriber(self):
        stt = FakeSTT()
        asyncio.run(_session(stt=stt).process_audio_turn(b"abc", mime_type="audio/mp4"))
        self.assertEqual(stt.calls, [(3, "audio/mp4")])


class HelperFailureTests(unittest.TestCase):
    def test_generate_speech_returns_empty_audio_when_tts_fails(self):
        session = _session(tts=FakeTTS(boom=True))
        self.assertEqual(asyncio.run(session.generate_speech("hello")), b"")

    def test_generate_speech_returns_the_provider_audio_on_success(self):
        session = _session(tts=FakeTTS(audio=b"wav-bytes"))
        self.assertEqual(asyncio.run(session.generate_speech("hello")), b"wav-bytes")

    def test_transcribe_audio_returns_the_empty_string_when_stt_fails(self):
        session = _session(stt=FakeSTT(boom=True))
        self.assertEqual(asyncio.run(session.transcribe_audio(b"audio")), "")

    def test_transcribe_audio_returns_the_transcript_on_success(self):
        session = _session(stt=FakeSTT(transcript="hello there"))
        self.assertEqual(
            asyncio.run(session.transcribe_audio(b"audio", mime_type="audio/wav")), "hello there"
        )


class ConversationWindowTests(unittest.TestCase):
    def test_only_the_last_twenty_messages_reach_the_llm(self):
        llm = RecordingLLM()
        session = VoiceSession(
            llm_client=llm,
            stt=FakeSTT(),
            tts=FakeTTS(),
            conversation_history=[
                {"role": "user" if index % 2 == 0 else "assistant", "content": f"m{index}"}
                for index in range(40)
            ],
        )
        asyncio.run(session.process_audio_turn(b"audio"))
        prompt_lines = llm.calls[0]["prompt"].splitlines()
        # 41 messages exist after the append; only the last 20 are sent, and the
        # freshly appended user turn is the last of them.
        self.assertEqual(len(prompt_lines), 20)
        self.assertEqual(prompt_lines[0], "ASSISTANT: m21")
        self.assertTrue(prompt_lines[-1].startswith("USER: Tell me about idempotency."))


class FactoryTests(unittest.TestCase):
    def test_the_browser_tier_uses_inert_providers(self):
        session = create_voice_session(RecordingLLM(), tier="browser")
        self.assertIsInstance(session.stt, NullSTTProvider)
        self.assertIsInstance(session.tts, NullTTSProvider)

    def test_the_cloud_tier_mirrors_the_stt_key_selection_for_tts(self):
        stt = FakeSTT()
        tts = FakeTTS()
        with patch.object(voice_session, "create_stt_provider", return_value=stt) as stt_factory, \
                patch.object(voice_session, "create_tts_provider", return_value=tts) as tts_factory, \
                patch.dict(
                    os.environ,
                    {"GROQ_API_KEY": "groq-key", "OPENAI_API_KEY": "openai-key"},
                    clear=False,
                ):
            session = create_voice_session(RecordingLLM(), tier="cloud", session_type="qa")

        self.assertIs(session.stt, stt)
        self.assertIs(session.tts, tts)
        stt_args, tts_args = stt_factory.call_args, tts_factory.call_args
        # Groq STT reads the Groq key; the default edge TTS needs no key at all,
        # so none is passed and no bogus OpenAI key is spent.
        self.assertEqual(stt_args.kwargs["api_key"], "groq-key")
        self.assertEqual(tts_args.kwargs["api_key"], "")
        self.assertEqual(session.system_prompt.count("study tutor"), 1)

    def test_an_openai_tts_provider_receives_the_openai_key(self):
        with patch.object(voice_session, "create_stt_provider") as stt_factory, \
                patch.object(voice_session, "create_tts_provider") as tts_factory, \
                patch.object(voice_session, "get_settings") as settings_factory, \
                patch.dict(
                    os.environ,
                    {"GROQ_API_KEY": "groq-key", "OPENAI_API_KEY": "openai-key"},
                    clear=False,
                ):
                    settings_factory.return_value = SimpleNamespace(
                        voice_stt_provider="openai",
                        voice_stt_model="whisper-1",
                        voice_tts_provider="openai",
                        voice_tts_voice="alloy",
                    )
                    create_voice_session(RecordingLLM(), tier="cloud")

        self.assertEqual(stt_factory.call_args.kwargs["api_key"], "openai-key")
        self.assertEqual(stt_factory.call_args.kwargs["model"], "whisper-1")
        # Only OpenAI TTS needs a key; omitting it makes every cloud-tier
        # synthesis fail at request time, so the factory must pass it through.
        self.assertEqual(tts_factory.call_args.kwargs["api_key"], "openai-key")
        self.assertEqual(tts_factory.call_args.kwargs["voice"], "alloy")

    def test_an_inert_browser_turn_returns_no_audio_and_no_text(self):
        session = create_voice_session(RecordingLLM(), tier="browser", session_type="chat")
        out = asyncio.run(session.process_audio_turn(b"audio"))
        self.assertEqual(out["transcript"], "")
        self.assertEqual(out["audio_bytes"], b"")

    def test_an_unknown_tier_is_rejected(self):
        with self.assertRaises(UnsupportedVoiceTierError) as ctx:
            create_voice_session(RecordingLLM(), tier="premium")
        self.assertIn("premium", str(ctx.exception))

    def test_the_server_side_persona_override_is_threaded_through_the_factory(self):
        session = create_voice_session(
            RecordingLLM(), tier="browser", session_type="interview", system_prompt="You are SRE."
        )
        self.assertEqual(session.system_prompt, "You are SRE.")
        self.assertEqual(session.session_type, "interview")
