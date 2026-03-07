import asyncio
import unittest
from types import SimpleNamespace

from app.services.mcp_gateway import MCPGateway


def _settings(**overrides):
    base = {
        "enable_mcp_gateway": True,
        "mcp_rollout_stage": 1,
        "mcp_timeout_seconds": 3.0,
        "mcp_max_context_chars": 4000,
        "mcp_enable_custom_topic": True,
        "mcp_enable_chat": True,
        "mcp_enable_questions": True,
        "mcp_enable_quiz": True,
        "mcp_enable_interview": True,
        "mcp_agentic_loop_enabled": False,
        "mcp_agentic_max_steps": 2,
        "mcp_tavily_enabled": False,
        "tavily_api_key": "",
        "mcp_firecrawl_enabled": False,
        "firecrawl_api_key": "",
        "mcp_github_enabled": False,
        "github_token": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class MCPGatewayTests(unittest.TestCase):
    def test_stage_gating(self):
        gateway = MCPGateway(_settings(mcp_rollout_stage=1))
        self.assertTrue(gateway.is_enabled_for_flow("custom_topic"))
        self.assertTrue(gateway.is_enabled_for_flow("chat"))
        self.assertFalse(gateway.is_enabled_for_flow("questions"))
        self.assertFalse(gateway.is_enabled_for_flow("quiz"))
        self.assertFalse(gateway.is_enabled_for_flow("interview"))

        gateway2 = MCPGateway(_settings(mcp_rollout_stage=3))
        self.assertTrue(gateway2.is_enabled_for_flow("questions"))
        self.assertTrue(gateway2.is_enabled_for_flow("quiz"))
        self.assertTrue(gateway2.is_enabled_for_flow("interview"))

    def test_gather_context_soft_fallback(self):
        gateway = MCPGateway(_settings(enable_mcp_gateway=False))
        out = asyncio.run(gateway.gather_context(flow="chat", query="java retry patterns"))
        self.assertEqual(out, "")

    def test_agentic_loop_uses_multiple_query_variants(self):
        class StubGateway(MCPGateway):
            def __init__(self):
                super().__init__(
                    _settings(
                        mcp_agentic_loop_enabled=True,
                        mcp_agentic_max_steps=2,
                    )
                )
                self.queries: list[str] = []

            async def _tavily_context(self, query: str) -> tuple[str, str]:
                self.queries.append(query)
                return f"Tavily web context:\n- row for {query}", ""

            async def _firecrawl_context(self, url: str) -> str:
                return ""

            async def _github_context(self, query: str) -> str:
                return ""

            def _context_is_adequate(self, *, query: str, merged: str, step: int, max_steps: int) -> bool:
                return step >= max_steps

        gateway = StubGateway()
        out = asyncio.run(
            gateway.gather_context(
                flow="chat",
                query="retry safety idempotency keys",
                topic_id="backend",
                topic_title="Retry Patterns",
            )
        )
        self.assertIn("Tavily web context", out)
        self.assertGreaterEqual(len(gateway.queries), 2)


if __name__ == "__main__":
    unittest.main()
