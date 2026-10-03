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
    #: The route map every test in this class shares. `eval` and `retrieval`
    #: deliberately point at models distinct from `final` and from the default,
    #: so a routing decision is always observable in the model that was sent.
    _ROUTE_MAP = {
        "eval": "gpt-4o-mini",
        "retrieval": "gpt-4o-search",
        "final": "gpt-4o",
    }

    def _settings(self, **overrides):
        base = dict(
            default_temperature=0.2,
            default_provider="openai",
            default_model="gpt-4o-mini",
            llm_task_routing_enabled=True,
            llm_task_route_map_json=json.dumps(self._ROUTE_MAP),
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _run_completion(self, client, settings, prompt, **call_kwargs):
        captured: dict[str, str] = {}

        async def fake_completion(**kwargs):
            captured["model"] = kwargs.get("model", "")
            return _FakeResp()

        with patch("app.services.llm_client.get_settings", return_value=settings):
            with patch("app.services.llm_client.litellm.acompletion", new=fake_completion):
                out = asyncio.run(client.completion(prompt, **call_kwargs))
        return out, captured

    def test_task_routing_applies_when_model_not_locked(self):
        """Stage 2.4: the task must be passed explicitly to route on it."""
        client = LLMClient(user_settings_store=None)
        out, captured = self._run_completion(
            client,
            self._settings(),
            "Score the candidate answer against the rubric and return JSON",
            task="eval",
        )

        self.assertTrue(out["success"])
        self.assertEqual(out["metadata"].get("task"), "eval")
        self.assertEqual(captured.get("model"), "gpt-4o-mini")

    def test_explicit_model_overrides_task_routing(self):
        """An explicitly configured model wins over the task route map."""
        client = LLMClient(user_settings_store=None)
        cfg = LLMConfigRequest(provider=LLMProviderEnum.OPENAI, model="gpt-4o")
        out, captured = self._run_completion(
            client,
            self._settings(),
            "Score the candidate answer against the rubric and return JSON",
            llm_config=cfg,
            task="eval",
        )

        self.assertTrue(out["success"])
        self.assertEqual(out["metadata"].get("task"), "eval")
        self.assertEqual(captured.get("model"), "gpt-4o")

    def test_prompt_text_alone_no_longer_infers_the_task(self):
        """Regression: prompt wording must not steer the task route.

        The prompt here contains both the eval wording ("Evaluate the candidate
        answer") and retrieval wording ("search"), so under the old
        text-sniffing router it would have been routed to one of those instead
        of the default. `task=None` now means "final", and only the explicit
        `task=` argument can select another route.
        """
        client = LLMClient(user_settings_store=None)
        hostile_prompt = (
            "Evaluate the candidate answer and return rubric JSON; "
            "also search the retrieval index for supporting quotes."
        )
        out, captured = self._run_completion(
            client,
            self._settings(),
            hostile_prompt,
            task=None,
        )

        self.assertTrue(out["success"])
        self.assertEqual(out["metadata"].get("task"), "final")
        # Routed via the "final" entry, not via eval/retrieval/default.
        self.assertEqual(captured.get("model"), "gpt-4o")
        self.assertNotIn("search", str(captured.get("model")))

    def test_unset_task_falls_back_to_final_when_routing_is_disabled(self):
        """Routing off + no task still resolves to "final" and the default model."""
        client = LLMClient(user_settings_store=None)
        out, captured = self._run_completion(
            client,
            self._settings(llm_task_routing_enabled=False),
            "Evaluate the candidate answer and return rubric JSON",
        )

        self.assertTrue(out["success"])
        self.assertEqual(out["metadata"].get("task"), "final")
        self.assertEqual(captured.get("model"), "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
