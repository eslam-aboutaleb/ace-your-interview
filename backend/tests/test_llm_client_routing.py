import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.schemas.models import LLMConfigRequest, LLMProviderEnum
from app.services.llm_client import LLMClient


class _FakeResp:
    class _Choice:
        class _Msg:
            content = "ok"

        message = _Msg()

    choices = [_Choice()]


class LLMClientRoutingTests(unittest.TestCase):
    def test_task_routing_applies_when_model_not_locked(self):
        client = LLMClient(user_settings_store=None)
        captured: dict[str, str] = {}

        async def fake_completion(**kwargs):
            captured["model"] = kwargs.get("model", "")
            return _FakeResp()

        settings = SimpleNamespace(
            default_temperature=0.2,
            default_provider="openai",
            default_model="gpt-4o-mini",
            llm_task_routing_enabled=True,
            llm_task_route_map_json=json.dumps({"eval": "gpt-4o-mini", "final": "gpt-4o"}),
        )
        with patch("app.services.llm_client.get_settings", return_value=settings):
            with patch("app.services.llm_client.litellm.acompletion", new=fake_completion):
                out = asyncio.run(
                    client.completion("Evaluate the candidate answer and return rubric JSON")
                )
        self.assertTrue(out["success"])
        self.assertEqual(captured.get("model"), "gpt-4o-mini")

    def test_explicit_model_overrides_task_routing(self):
        client = LLMClient(user_settings_store=None)
        captured: dict[str, str] = {}

        async def fake_completion(**kwargs):
            captured["model"] = kwargs.get("model", "")
            return _FakeResp()

        settings = SimpleNamespace(
            default_temperature=0.2,
            default_provider="openai",
            default_model="gpt-4o-mini",
            llm_task_routing_enabled=True,
            llm_task_route_map_json=json.dumps({"eval": "gpt-4o-mini"}),
        )
        cfg = LLMConfigRequest(provider=LLMProviderEnum.OPENAI, model="gpt-4o")
        with patch("app.services.llm_client.get_settings", return_value=settings):
            with patch("app.services.llm_client.litellm.acompletion", new=fake_completion):
                out = asyncio.run(
                    client.completion(
                        "Evaluate the candidate answer and return rubric JSON",
                        llm_config=cfg,
                    )
                )
        self.assertTrue(out["success"])
        self.assertEqual(captured.get("model"), "gpt-4o")


if __name__ == "__main__":
    unittest.main()
