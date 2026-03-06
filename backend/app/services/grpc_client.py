"""Async gRPC client for llm-chain and cli-agent services."""

from __future__ import annotations

import logging
from typing import Optional

import grpc

from app.config import get_settings
from app.schemas.models import PROVIDER_TO_PROTO, LLMConfigRequest

logger = logging.getLogger(__name__)

# Proto stubs are generated at build time into app/generated/
from app.generated import analysis_pb2, analysis_pb2_grpc  # noqa: E402


class GRPCClient:
    """Manages async gRPC channels to both LLM backends."""

    def __init__(self):
        settings = get_settings()
        self._llm_chain_target = f"{settings.grpc_llm_chain_host}:{settings.grpc_llm_chain_port}"
        self._cli_agent_target = f"{settings.grpc_cli_agent_host}:{settings.grpc_cli_agent_port}"
        self._llm_chain_channel: Optional[grpc.aio.Channel] = None
        self._cli_agent_channel: Optional[grpc.aio.Channel] = None

    # ── Channel management ──────────────────────────────────
    async def _get_llm_chain_stub(self) -> analysis_pb2_grpc.AnalysisServiceStub:
        if self._llm_chain_channel is None:
            self._llm_chain_channel = grpc.aio.insecure_channel(self._llm_chain_target)
        return analysis_pb2_grpc.AnalysisServiceStub(self._llm_chain_channel)

    async def _get_cli_agent_stub(self) -> analysis_pb2_grpc.AnalysisServiceStub:
        if self._cli_agent_channel is None:
            self._cli_agent_channel = grpc.aio.insecure_channel(self._cli_agent_target)
        return analysis_pb2_grpc.AnalysisServiceStub(self._cli_agent_channel)

    def _resolve_backend(self, provider: str) -> str:
        """Route to the correct backend, mirroring LLMGateway logic."""
        if provider.lower() == "github":
            return "cli-agent"
        return "llm-chain"

    async def _get_stub(self, provider: str) -> analysis_pb2_grpc.AnalysisServiceStub:
        backend = self._resolve_backend(provider)
        if backend == "cli-agent":
            return await self._get_cli_agent_stub()
        return await self._get_llm_chain_stub()

    # ── Public API ──────────────────────────────────────────
    async def quick_analysis(
        self,
        question: str,
        llm_config: Optional[LLMConfigRequest] = None,
    ) -> dict:
        """Send a freeform question to the LLM via QuickAnalysis RPC."""
        provider_str = "default"
        proto_config = analysis_pb2.LLMConfig()

        if llm_config:
            provider_str = llm_config.provider.value
            proto_config = analysis_pb2.LLMConfig(
                provider=PROVIDER_TO_PROTO.get(provider_str, 0),
                model=llm_config.model or "",
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
            )

        request = analysis_pb2.QuickAnalysisRequest(
            question=question,
            current_price=0.0,
            user_id="study-app",
            llm_config=proto_config,
        )

        try:
            stub = await self._get_stub(provider_str)
            response = await stub.QuickAnalysis(request, timeout=120)
            return {
                "success": response.success,
                "analysis": response.analysis,
                "metadata": dict(response.metadata) if response.metadata else {},
                "error": response.error,
            }
        except grpc.aio.AioRpcError as e:
            logger.error(f"gRPC error: {e.code()} – {e.details()}")
            return {
                "success": False,
                "analysis": "",
                "metadata": {},
                "error": f"gRPC error: {e.code().name} – {e.details()}",
            }
        except Exception as e:
            logger.error(f"Unexpected error calling gRPC: {e}")
            return {
                "success": False,
                "analysis": "",
                "metadata": {},
                "error": str(e),
            }

    async def health_check(self, backend: str = "llm-chain") -> dict:
        """Check health of a specific backend."""
        try:
            if backend == "cli-agent":
                stub = await self._get_cli_agent_stub()
            else:
                stub = await self._get_llm_chain_stub()

            response = await stub.HealthCheck(
                analysis_pb2.HealthRequest(), timeout=10
            )
            return {
                "healthy": response.healthy,
                "service_name": response.service_name,
                "version": response.version,
                "capabilities": dict(response.capabilities) if response.capabilities else {},
            }
        except Exception as e:
            logger.warning(f"Health check failed for {backend}: {e}")
            return {
                "healthy": False,
                "service_name": backend,
                "version": "",
                "capabilities": {},
            }

    async def close(self):
        if self._llm_chain_channel:
            await self._llm_chain_channel.close()
        if self._cli_agent_channel:
            await self._cli_agent_channel.close()
