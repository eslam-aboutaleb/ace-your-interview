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


if __name__ == "__main__":
    unittest.main()
