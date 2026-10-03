"""Coverage for app/services/grpc_client.py.

No socket is ever opened: ``grpc.aio.insecure_channel`` is patched with a fake
channel that hands the real generated ``AnalysisServiceStub`` a recording async
callable. The protobuf messages themselves are real, so request/response
marshalling is exercised for real.

Requires ``grpcio`` and the vendored ``app/generated`` stubs, both of which are
tracked on main.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import grpc

from app.generated import analysis_pb2
from app.schemas.models import LLMConfigRequest
from app.services import grpc_client as grpc_client_module
from app.services.grpc_client import GRPCClient


def _run(coro):
    return asyncio.run(coro)


def _settings_stub():
    """The four gRPC targets ``GRPCClient.__init__`` reads.

    ``app/config.py`` declares these as ``STUDY_GRPC_*`` settings;
    the stub keeps the channel tests independent of the defaults
    declared there. See ``GRPCClientSettingsContractTests`` for the
    contract against the real ``Settings``.
    """
    return SimpleNamespace(
        grpc_llm_chain_host="llm-chain",
        grpc_llm_chain_port=50051,
        grpc_cli_agent_host="cli-agent",
        grpc_cli_agent_port=50052,
    )


class _FakeChannel:
    """Minimal stand-in for ``grpc.aio.Channel``.

    ``unary_unary`` returns the same shape the real stub expects: an async
    callable taking ``(request, timeout=...)``. Keeping the real
    ``AnalysisServiceStub`` on top of this means the stub's method wiring
    (``/polymarket.analysis.AnalysisService/...``) is still asserted.
    """

    def __init__(self, handler):
        self._handler = handler
        self.method_paths: list[str] = []
        self.calls: list[tuple[str, object, float | None]] = []
        self.closed = False
        self.close_count = 0

    def unary_unary(
        self,
        method,
        request_serializer=None,
        response_deserializer=None,
        _registered_method=False,
    ):
        self.method_paths.append(method)

        async def _call(request, timeout=None, **kwargs):
            self.calls.append((method, request, timeout))
            outcome = self._handler(method, request)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return _call

    async def close(self, grace=None):
        self.closed = True
        self.close_count += 1


def _grpc_error(code=grpc.StatusCode.UNAVAILABLE, details="connection refused"):
    return grpc.aio.AioRpcError(code, None, (), details, "debug_error_string=stub")


class GRPCClientSettingsContractTests(unittest.TestCase):
    """The client is built from the real ``Settings`` object."""

    def test_real_settings_expose_the_grpc_target_fields(self):
        from app.config import Settings

        fields = Settings.model_fields
        for name in (
            "grpc_llm_chain_host",
            "grpc_llm_chain_port",
            "grpc_cli_agent_host",
            "grpc_cli_agent_port",
        ):
            self.assertIn(name, fields)

    def test_real_settings_construct_the_client(self):
        from app.config import Settings

        with patch.object(grpc_client_module, "get_settings", return_value=Settings()):
            client = GRPCClient()
        self.assertEqual(client._llm_chain_target, "localhost:50051")
        self.assertEqual(client._cli_agent_target, "localhost:50052")


class GRPCClientChannelTests(unittest.TestCase):
    def _client(self):
        with patch.object(grpc_client_module, "get_settings", return_value=_settings_stub()):
            return GRPCClient()

    def test_init_derives_targets_from_settings(self):
        client = self._client()
        self.assertEqual(client._llm_chain_target, "llm-chain:50051")
        self.assertEqual(client._cli_agent_target, "cli-agent:50052")
        self.assertIsNone(client._llm_chain_channel)
        self.assertIsNone(client._cli_agent_channel)

    def test_stub_channels_are_created_once_and_reused(self):
        client = self._client()
        channels = {}

        def fake_channel(target):
            channels.setdefault(target, _FakeChannel(lambda method, request: None))
            return channels[target]

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", side_effect=fake_channel):
            _run(client._get_llm_chain_stub())
            _run(client._get_llm_chain_stub())

        # One channel, but a fresh stub each call, and the real stub registers
        # both RPCs on the channel it is handed.
        self.assertEqual(len(channels), 1)
        self.assertEqual(
            channels[client._llm_chain_target].method_paths,
            [
                "/polymarket.analysis.AnalysisService/QuickAnalysis",
                "/polymarket.analysis.AnalysisService/HealthCheck",
            ]
            * 2,
        )

    def test_cli_agent_channel_is_cached_across_stub_requests(self):
        client = self._client()
        channels = []

        def fake_channel(target):
            channel = _FakeChannel(lambda method, request: None)
            channels.append(channel)
            return channel

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", side_effect=fake_channel):
            _run(client._get_cli_agent_stub())
            _run(client._get_cli_agent_stub())

        self.assertEqual(len(channels), 1)

    def test_cli_agent_and_llm_chain_use_distinct_targets(self):
        client = self._client()
        seen: list[str] = []

        def fake_channel(target):
            seen.append(target)
            return _FakeChannel(lambda method, request: None)

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", side_effect=fake_channel):
            _run(client._get_cli_agent_stub())
            _run(client._get_llm_chain_stub())

        self.assertEqual(seen, [client._cli_agent_target, client._llm_chain_target])

    def test_resolve_backend_routes_github_to_cli_agent(self):
        client = self._client()
        self.assertEqual(client._resolve_backend("github"), "cli-agent")
        self.assertEqual(client._resolve_backend("GitHub"), "cli-agent")
        self.assertEqual(client._resolve_backend("openai"), "llm-chain")
        self.assertEqual(client._resolve_backend("default"), "llm-chain")
        self.assertEqual(client._resolve_backend(""), "llm-chain")

    def test_get_stub_dispatches_by_provider(self):
        client = self._client()
        targets: list[str] = []

        def fake_channel(target):
            targets.append(target)
            return _FakeChannel(lambda method, request: None)

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", side_effect=fake_channel):
            _run(client._get_stub("github"))
            _run(client._get_stub("anthropic"))

        self.assertEqual(targets, [client._cli_agent_target, client._llm_chain_target])

    def test_close_is_a_noop_when_no_channel_was_opened(self):
        client = self._client()
        _run(client.close())
        self.assertIsNone(client._llm_chain_channel)
        self.assertIsNone(client._cli_agent_channel)

    def test_close_closes_only_the_channels_that_were_opened(self):
        client = self._client()
        llm_channel = _FakeChannel(lambda method, request: None)
        cli_channel = _FakeChannel(lambda method, request: None)

        def fake_channel(target):
            return cli_channel if target == client._cli_agent_target else llm_channel

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", side_effect=fake_channel):
            _run(client._get_cli_agent_stub())
            _run(client.close())

        self.assertTrue(cli_channel.closed)
        self.assertEqual(cli_channel.close_count, 1)
        self.assertFalse(llm_channel.closed)

    def test_close_closes_both_channels(self):
        client = self._client()
        channels = []

        def fake_channel(target):
            channel = _FakeChannel(lambda method, request: None)
            channels.append(channel)
            return channel

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", side_effect=fake_channel):
            _run(client._get_llm_chain_stub())
            _run(client._get_cli_agent_stub())
            _run(client.close())

        self.assertEqual([c.closed for c in channels], [True, True])


class GRPCClientQuickAnalysisTests(unittest.TestCase):
    def _client(self):
        with patch.object(grpc_client_module, "get_settings", return_value=_settings_stub()):
            return GRPCClient()

    def test_default_config_sends_zero_proto_provider_and_default_user(self):
        client = self._client()
        response = analysis_pb2.AnalysisResponse(
            success=True,
            analysis="looks good",
            metadata={"provider": "openai", "model": "gpt-4o-mini"},
        )
        channel = _FakeChannel(lambda method, request: response)

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", return_value=channel):
            out = _run(client.quick_analysis("Why is my p99 spiking?"))

        self.assertEqual(
            out,
            {
                "success": True,
                "analysis": "looks good",
                "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
                "error": "",
            },
        )
        method, request, timeout = channel.calls[0]
        self.assertEqual(method, "/polymarket.analysis.AnalysisService/QuickAnalysis")
        self.assertEqual(timeout, 120)
        self.assertEqual(request.question, "Why is my p99 spiking?")
        self.assertEqual(request.current_price, 0.0)
        self.assertEqual(request.user_id, "study-app")
        # No llm_config supplied -> the proto default (provider 0).
        self.assertEqual(request.llm_config.provider, 0)
        self.assertEqual(request.llm_config.model, "")

    def test_llm_config_is_marshalled_into_the_proto_message(self):
        client = self._client()
        channel = _FakeChannel(
            lambda method, request: analysis_pb2.AnalysisResponse(success=True, analysis="ok")
        )
        config = LLMConfigRequest(
            provider="anthropic", model="claude-3-5-sonnet", temperature=0.25, max_tokens=512
        )

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", return_value=channel):
            out = _run(client.quick_analysis("hello", config))

        self.assertTrue(out["success"])
        self.assertEqual(out["metadata"], {})
        request = channel.calls[0][1]
        self.assertEqual(request.llm_config.provider, 2)  # LLM_PROVIDER_ANTHROPIC
        self.assertEqual(request.llm_config.model, "claude-3-5-sonnet")
        self.assertAlmostEqual(request.llm_config.temperature, 0.25, places=5)
        self.assertEqual(request.llm_config.max_tokens, 512)

    def test_github_provider_is_marshalled_and_routed_to_cli_agent(self):
        client = self._client()
        targets: list[str] = []
        channel = _FakeChannel(lambda method, request: analysis_pb2.AnalysisResponse(success=True))

        def fake_channel(target):
            targets.append(target)
            return channel

        config = LLMConfigRequest(provider="github", model="gpt-5-copilot", max_tokens=64)
        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", side_effect=fake_channel):
            _run(client.quick_analysis("hello", config))

        self.assertEqual(targets, [client._cli_agent_target])
        self.assertEqual(channel.calls[0][1].llm_config.provider, 6)  # LLM_PROVIDER_GITHUB

    def test_null_temperature_falls_back_to_proto_zero(self):
        client = self._client()
        channel = _FakeChannel(lambda method, request: analysis_pb2.AnalysisResponse(success=True))
        config = LLMConfigRequest(provider="groq", model="llama-3.3-70b", max_tokens=128)

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", return_value=channel):
            _run(client.quick_analysis("hi", config))

        request = channel.calls[0][1]
        self.assertEqual(request.llm_config.provider, 4)  # LLM_PROVIDER_GROQ
        self.assertEqual(request.llm_config.temperature, 0.0)

    def test_aio_rpc_error_is_reported_with_code_and_details(self):
        client = self._client()
        channel = _FakeChannel(
            lambda method, request: _grpc_error(grpc.StatusCode.DEADLINE_EXCEEDED, "deadline exceeded")
        )

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", return_value=channel):
            out = _run(client.quick_analysis("slow question"))

        self.assertEqual(out["success"], False)
        self.assertEqual(out["analysis"], "")
        self.assertEqual(out["metadata"], {})
        self.assertEqual(out["error"], "gRPC error: DEADLINE_EXCEEDED – deadline exceeded")

    def test_unexpected_exception_is_reported_with_its_message(self):
        client = self._client()
        channel = _FakeChannel(
            lambda method, request: RuntimeError("stub raised before dispatch")
        )

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", return_value=channel):
            out = _run(client.quick_analysis("boom"))

        self.assertEqual(out["success"], False)
        self.assertEqual(out["error"], "stub raised before dispatch")

    def test_channel_construction_failure_is_contained(self):
        client = self._client()

        with patch.object(
            grpc_client_module.grpc.aio, "insecure_channel", side_effect=OSError("dns failure")
        ):
            out = _run(client.quick_analysis("q"))

        self.assertEqual(out["success"], False)
        self.assertEqual(out["error"], "dns failure")
        # Nothing was cached, so a later attempt can still construct a channel.
        self.assertIsNone(client._llm_chain_channel)


class GRPCClientHealthCheckTests(unittest.TestCase):
    def _client(self):
        with patch.object(grpc_client_module, "get_settings", return_value=_settings_stub()):
            return GRPCClient()

    def test_llm_chain_health_is_returned_with_capabilities(self):
        client = self._client()
        response = analysis_pb2.HealthResponse(
            healthy=True,
            service_name="llm-chain",
            version="1.4.2",
            capabilities={"analysis": "yes", "streaming": "no"},
        )
        channel = _FakeChannel(lambda method, request: response)

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", return_value=channel):
            out = _run(client.health_check())

        self.assertEqual(
            out,
            {
                "healthy": True,
                "service_name": "llm-chain",
                "version": "1.4.2",
                "capabilities": {"analysis": "yes", "streaming": "no"},
            },
        )
        method, request, timeout = channel.calls[0]
        self.assertEqual(method, "/polymarket.analysis.AnalysisService/HealthCheck")
        self.assertEqual(timeout, 10)
        self.assertEqual(request, analysis_pb2.HealthRequest())

    def test_cli_agent_health_uses_the_cli_agent_channel(self):
        client = self._client()
        targets: list[str] = []
        channel = _FakeChannel(
            lambda method, request: analysis_pb2.HealthResponse(
                healthy=True, service_name="cli-agent", version="2.0.0"
            )
        )

        def fake_channel(target):
            targets.append(target)
            return channel

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", side_effect=fake_channel):
            out = _run(client.health_check("cli-agent"))

        self.assertEqual(targets, [client._cli_agent_target])
        self.assertTrue(out["healthy"])
        # An empty capabilities map collapses to {} rather than staying a map.
        self.assertEqual(out["capabilities"], {})

    def test_unknown_backend_name_falls_back_to_llm_chain(self):
        client = self._client()
        targets: list[str] = []
        channel = _FakeChannel(lambda method, request: analysis_pb2.HealthResponse(healthy=True))

        def fake_channel(target):
            targets.append(target)
            return channel

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", side_effect=fake_channel):
            out = _run(client.health_check("something-else"))

        self.assertEqual(targets, [client._llm_chain_target])
        self.assertTrue(out["healthy"])

    def test_health_check_degrades_instead_of_raising(self):
        client = self._client()
        channel = _FakeChannel(
            lambda method, request: _grpc_error(grpc.StatusCode.UNAVAILABLE, "backend down")
        )

        with patch.object(grpc_client_module.grpc.aio, "insecure_channel", return_value=channel):
            out = _run(client.health_check("cli-agent"))

        self.assertEqual(
            out,
            {
                "healthy": False,
                "service_name": "cli-agent",
                "version": "",
                "capabilities": {},
            },
        )

    def test_health_check_degrades_when_the_channel_cannot_be_built(self):
        client = self._client()
        with patch.object(
            grpc_client_module.grpc.aio, "insecure_channel", side_effect=OSError("no route")
        ):
            out = _run(client.health_check())

        self.assertFalse(out["healthy"])
        self.assertEqual(out["service_name"], "llm-chain")


if __name__ == "__main__":
    unittest.main()
