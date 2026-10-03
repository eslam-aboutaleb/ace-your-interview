"""Coverage for ``app.services.voice_providers`` — provider HTTP and factory paths.

The HTTP providers are driven with a fake ``httpx`` namespace patched onto the
module, and the Edge TTS providers get a stub ``edge_tts`` module in
``sys.modules``, so nothing leaves the process.
"""

import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services import voice_providers
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


def _run(coro):
    return asyncio.run(coro)


class _FakeResponse:
    def __init__(self, text="", content=b"", chunks=(), raises=None):
        self.text = text
        self.content = content
        self._chunks = list(chunks)
        self._raises = raises

    def raise_for_status(self):
        if self._raises is not None:
            raise self._raises

    async def aiter_bytes(self, size=None):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc_info):
        return False


class _FakeAsyncClient:
    def __init__(self, response=None, **kwargs):
        self._response = response if response is not None else _FakeResponse()
        self.init_kwargs = kwargs
        self.posts: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self._response

    def stream(self, method, url, **kwargs):
        self.posts.append((url, dict(kwargs, method=method)))
        return _FakeStreamContext(self._response)


def _namespace(client: _FakeAsyncClient):
    def factory(**kwargs):
        client.init_kwargs = kwargs
        return client

    return SimpleNamespace(AsyncClient=factory)


class SttProviderTests(unittest.TestCase):
    def _transcribe(self, provider, client, audio=b"\x01" * 200, mime="audio/webm"):
        with patch.object(voice_providers, "httpx", _namespace(client)):
            return _run(provider.transcribe(audio, mime_type=mime))

    def test_groq_transcribe_posts_multipart_and_returns_stripped_text(self):
        client = _FakeAsyncClient(_FakeResponse(text="  hello world \n"))
        provider = GroqSTTProvider(api_key="gsk-test", model="whisper-large-v3")

        transcript = self._transcribe(provider, client)

        self.assertEqual(transcript, "hello world")
        url, kwargs = client.posts[0]
        self.assertEqual(url, "https://api.groq.com/openai/v1/audio/transcriptions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer gsk-test")
        self.assertEqual(kwargs["data"]["model"], "whisper-large-v3")
        self.assertEqual(kwargs["data"]["response_format"], "text")
        filename, _stream, content_type = kwargs["files"]["file"]
        self.assertEqual(filename, "audio.webm")
        self.assertEqual(content_type, "audio/webm")
        self.assertEqual(client.init_kwargs["timeout"], 30.0)

    def test_groq_transcribe_maps_every_known_mime_type_to_a_filename(self):
        expected = {
            "audio/webm": "audio.webm",
            "audio/wav": "audio.wav",
            "audio/mp4": "audio.mp4",
            "audio/ogg": "audio.ogg",
            "audio/mpeg": "audio.mp3",
            "application/ogg": "audio.webm",
        }
        for mime, filename in expected.items():
            with self.subTest(mime=mime):
                client = _FakeAsyncClient(_FakeResponse(text="ok"))
                self._transcribe(GroqSTTProvider(api_key="k"), client, mime=mime)
                self.assertEqual(client.posts[0][1]["files"]["file"][0], filename)

    def test_groq_transcribe_propagates_http_errors(self):
        client = _FakeAsyncClient(_FakeResponse(raises=RuntimeError("429 rate limited")))
        with self.assertRaises(RuntimeError):
            self._transcribe(GroqSTTProvider(api_key="k"), client)

    def test_openai_transcribe_uses_the_openai_endpoint_and_default_model(self):
        client = _FakeAsyncClient(_FakeResponse(text="transcribed"))
        provider = OpenAISTTProvider(api_key="sk-test")

        self.assertEqual(self._transcribe(provider, client), "transcribed")
        url, kwargs = client.posts[0]
        self.assertEqual(url, "https://api.openai.com/v1/audio/transcriptions")
        self.assertEqual(provider.model, "whisper-1")
        self.assertEqual(kwargs["data"]["model"], "whisper-1")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-test")

    def test_openai_transcribe_maps_wav_and_propagates_http_errors(self):
        client = _FakeAsyncClient(_FakeResponse(text=" wav "))
        self.assertEqual(
            self._transcribe(OpenAISTTProvider(api_key="k"), client, mime="audio/wav"),
            "wav",
        )
        self.assertEqual(client.posts[0][1]["files"]["file"][0], "audio.wav")

        failing = _FakeAsyncClient(_FakeResponse(raises=RuntimeError("401")))
        with self.assertRaises(RuntimeError):
            self._transcribe(OpenAISTTProvider(api_key="k"), failing)


class NullProviderTests(unittest.TestCase):
    def test_null_stt_raises_with_a_tier_explanation(self):
        with self.assertRaises(RuntimeError) as ctx:
            _run(NullSTTProvider().transcribe(b"\x01" * 10))
        self.assertIn("browser tier", str(ctx.exception))

    def test_null_tts_synthesize_raises_with_a_tier_explanation(self):
        with self.assertRaises(RuntimeError) as ctx:
            _run(NullTTSProvider().synthesize("hello"))
        self.assertIn("browser tier", str(ctx.exception))

    def test_null_tts_stream_raises_only_on_first_iteration(self):
        provider = NullTTSProvider()
        stream = provider.synthesize_stream("hello")
        self.assertTrue(hasattr(stream, "__anext__"))
        with self.assertRaises(RuntimeError):
            _run(stream.__anext__())


class FakeEdgeCommunicate:
    """Records construction and replays a fixed chunk sequence."""

    last_init: dict = {}

    def __init__(self, text, voice):
        FakeEdgeCommunicate.last_init = {"text": text, "voice": voice}
        self._text = text

    async def stream(self):
        yield {"type": "WordBoundary", "text": self._text}
        yield {"type": "audio", "data": b"chunk-1"}
        yield {"type": "audio", "data": b"chunk-2"}


class EdgeTtsProviderTests(unittest.TestCase):
    def setUp(self):
        self.prev_module = sys.modules.get("edge_tts")
        sys.modules["edge_tts"] = SimpleNamespace(Communicate=FakeEdgeCommunicate)
        FakeEdgeCommunicate.last_init = {}

    def tearDown(self):
        if self.prev_module is None:
            sys.modules.pop("edge_tts", None)
        else:
            sys.modules["edge_tts"] = self.prev_module

    def test_synthesize_concatenates_only_audio_chunks(self):
        audio = _run(EdgeTTSProvider(voice="en-GB-SoniaNeural").synthesize("hello"))
        self.assertEqual(audio, b"chunk-1chunk-2")
        self.assertEqual(
            FakeEdgeCommunicate.last_init,
            {"text": "hello", "voice": "en-GB-SoniaNeural"},
        )

    def test_synthesize_stream_yields_audio_chunks_individually(self):
        provider = EdgeTTSProvider(voice="en-US-AriaNeural")

        async def collect():
            return [chunk async for chunk in provider.synthesize_stream("hello")]

        self.assertEqual(_run(collect()), [b"chunk-1", b"chunk-2"])
        self.assertEqual(
            FakeEdgeCommunicate.last_init,
            {"text": "hello", "voice": "en-US-AriaNeural"},
        )

    def test_default_voice_is_aria(self):
        _run(EdgeTTSProvider().synthesize("hi"))
        self.assertEqual(FakeEdgeCommunicate.last_init["voice"], "en-US-AriaNeural")


class OpenAiTtsProviderTests(unittest.TestCase):
    def test_synthesize_posts_json_and_returns_raw_mp3_bytes(self):
        client = _FakeAsyncClient(_FakeResponse(content=b"ID3-mp3-bytes"))
        provider = OpenAITTSProvider(api_key="sk-test", model="tts-1-hd", voice="nova")

        audio = _run_with_client(provider, client, "speak")

        self.assertEqual(audio, b"ID3-mp3-bytes")
        url, kwargs = client.posts[0]
        self.assertEqual(url, "https://api.openai.com/v1/audio/speech")
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(
            kwargs["json"],
            {
                "model": "tts-1-hd",
                "input": "speak",
                "voice": "nova",
                "response_format": "mp3",
            },
        )
        self.assertEqual(client.init_kwargs["timeout"], 30.0)

    def test_synthesize_propagates_http_errors(self):
        client = _FakeAsyncClient(_FakeResponse(raises=RuntimeError("402 quota")))
        with self.assertRaises(RuntimeError):
            _run_with_client(OpenAITTSProvider(api_key="sk"), client, "speak")

    def test_synthesize_stream_yields_chunked_bytes(self):
        client = _FakeAsyncClient(_FakeResponse(chunks=[b"a", b"b", b"c"]))
        provider = OpenAITTSProvider(api_key="sk-test", voice="alloy")

        async def collect():
            return [chunk async for chunk in provider.synthesize_stream("speak")]

        with patch.object(voice_providers, "httpx", _namespace(client)):
            chunks = _run(collect())

        self.assertEqual(chunks, [b"a", b"b", b"c"])
        _method, kwargs = client.posts[0]
        self.assertEqual(kwargs["method"], "POST")
        self.assertEqual(kwargs["json"]["voice"], "alloy")

    def test_synthesize_stream_propagates_http_errors(self):
        client = _FakeAsyncClient(_FakeResponse(raises=RuntimeError("500")))

        async def collect():
            return [chunk async for chunk in OpenAITTSProvider(api_key="sk").synthesize_stream("x")]

        with patch.object(voice_providers, "httpx", _namespace(client)):
            with self.assertRaises(RuntimeError):
                _run(collect())


def _run_with_client(provider, client, text):
    with patch.object(voice_providers, "httpx", _namespace(client)):
        return _run(provider.synthesize(text))


class ProviderFactoryTests(unittest.TestCase):
    def test_stt_factory_keeps_an_explicit_model_and_defaults_otherwise(self):
        groq = create_stt_provider("groq", api_key="k", model="whisper-large-v3")
        self.assertIsInstance(groq, GroqSTTProvider)
        self.assertEqual(groq.model, "whisper-large-v3")
        self.assertEqual(groq.base_url, "https://api.groq.com/openai/v1")

        openai = create_stt_provider("openai", api_key="k")
        self.assertIsInstance(openai, OpenAISTTProvider)
        self.assertEqual(openai.model, "whisper-1")

    def test_tts_factory_defaults_to_edge_with_aria(self):
        edge = create_tts_provider("")
        self.assertIsInstance(edge, EdgeTTSProvider)
        self.assertEqual(edge.voice, "en-US-AriaNeural")

        openai = create_tts_provider("openai", voice="", api_key="k")
        self.assertIsInstance(openai, OpenAITTSProvider)
        self.assertEqual(openai.voice, "alloy")
        self.assertEqual(openai.model, "tts-1")

    def test_factory_wires_the_configured_voice_and_key(self):
        provider = create_tts_provider("openai", voice="nova", api_key="sk-x")
        self.assertEqual(provider.api_key, "sk-x")
        self.assertEqual(provider.voice, "nova")


class PcmToWavTests(unittest.TestCase):
    def test_header_describes_the_requested_format(self):
        wav = pcm_to_wav(b"\x00" * 32, sample_rate=24000, channels=2, sample_width=2)

        self.assertEqual(wav[:4], b"RIFF")
        self.assertEqual(wav[8:12], b"WAVE")
        self.assertEqual(wav[12:16], b"fmt ")
        self.assertEqual(int.from_bytes(wav[16:20], "little"), 16)  # fmt chunk size
        self.assertEqual(int.from_bytes(wav[20:22], "little"), 1)  # PCM
        self.assertEqual(int.from_bytes(wav[22:24], "little"), 2)  # channels
        self.assertEqual(int.from_bytes(wav[24:28], "little"), 24000)  # sample rate
        self.assertEqual(int.from_bytes(wav[28:32], "little"), 24000 * 2 * 2)  # byte rate
        self.assertEqual(int.from_bytes(wav[32:34], "little"), 4)  # block align
        self.assertEqual(int.from_bytes(wav[34:36], "little"), 16)  # bits per sample
        self.assertEqual(wav[36:40], b"data")
        self.assertEqual(int.from_bytes(wav[40:44], "little"), 32)
        self.assertEqual(wav[44:], b"\x00" * 32)

    def test_empty_pcm_produces_a_bare_header(self):
        wav = pcm_to_wav(b"")
        self.assertEqual(len(wav), 44)
        self.assertEqual(int.from_bytes(wav[4:8], "little"), 36)

    def test_eight_bit_mono_format_is_encoded(self):
        wav = pcm_to_wav(b"\xff" * 8, sample_rate=8000, channels=1, sample_width=1)
        self.assertEqual(int.from_bytes(wav[34:36], "little"), 8)
        self.assertEqual(int.from_bytes(wav[32:34], "little"), 1)


class EdgeTtsImportFailureTests(unittest.TestCase):
    def test_missing_edge_tts_package_surfaces_as_import_error(self):
        prev = sys.modules.get("edge_tts")
        sys.modules["edge_tts"] = None  # ``import edge_tts`` raises ImportError
        try:
            with self.assertRaises(ImportError):
                _run(EdgeTTSProvider().synthesize("hello"))
        finally:
            if prev is None:
                sys.modules.pop("edge_tts", None)
            else:
                sys.modules["edge_tts"] = prev


if __name__ == "__main__":
    unittest.main()