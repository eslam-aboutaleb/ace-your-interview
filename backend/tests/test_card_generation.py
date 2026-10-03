"""Unit tests for LLM flashcard generation."""

from __future__ import annotations

import json
import unittest

from app.services.question_generator import (
    MAX_CARD_GENERATION_COUNT,
    QuestionGenerator,
    _validate_card_item,
)


class _FakeLLMClient:
    """Minimal LLM client stub returning a fixed JSON array."""

    def __init__(self, items: list[dict]):
        self._items = items
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, **kwargs):
        self.prompts.append(prompt)
        return {
            "success": True,
            "analysis": json.dumps(self._items),
            "content": json.dumps(self._items),
            "metadata": {"model": "fake"},
        }


def _valid_card(front: str, back: str) -> dict:
    return {
        "front": front,
        "back": back,
        "source_quote": "Authentication verifies the caller identity.",
        "source_section": "Authentication",
    }


class CardGenerationTests(unittest.TestCase):
    def _generator(self, items):
        return QuestionGenerator(_FakeLLMClient(items))

    def test_generate_cards_returns_validated_cards(self):
        items = [
            _valid_card(
                "What is Authentication?",
                "Authentication verifies who the caller is before access.",
            ),
            _valid_card(
                "Why does Authentication matter?",
                "Authentication prevents anonymous access to protected data.",
            ),
        ]
        generator = self._generator(items)
        import asyncio

        cards, stats = asyncio.run(
            generator.generate_cards(
                topic_id="topic-auth",
                topic_title="Authentication",
                doc_content="Authentication verifies the caller identity.",
                count=2,
            )
        )
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0]["front"], "What is Authentication?")
        self.assertEqual(
            cards[0]["back"],
            "Authentication verifies who the caller is before access.",
        )
        self.assertEqual(cards[0]["source_section"], "Authentication")
        self.assertEqual(
            cards[0]["source_quote"],
            "Authentication verifies the caller identity.",
        )
        self.assertTrue(cards[0]["card_id"].startswith("topic-auth:"))
        self.assertIn("retries_used", stats)
        self.assertIn("malformed_items_dropped", stats)

    def test_generate_cards_dedupes_against_existing(self):
        items = [
            _valid_card(
                "What is Authentication?",
                "Authentication verifies who the caller is before access.",
            ),
            _valid_card(
                "How does Authentication verify callers?",
                "Authentication verifies callers by checking their credentials before granting access.",
            ),
        ]
        generator = self._generator(items)
        import asyncio

        cards, _ = asyncio.run(
            generator.generate_cards(
                topic_id="topic-auth",
                topic_title="Authentication",
                doc_content="Authentication verifies the caller identity.",
                count=2,
                existing_cards=["What is Authentication?"],
            )
        )
        fronts = [c["front"] for c in cards]
        self.assertNotIn("What is Authentication?", fronts)
        self.assertIn("How does Authentication verify callers?", fronts)

    def test_generate_cards_caps_count(self):
        items = [
            _valid_card(
                f"What is Authentication topic {i}?",
                "Authentication verifies who the caller is before access.",
            )
            for i in range(MAX_CARD_GENERATION_COUNT + 10)
        ]
        generator = self._generator(items)
        import asyncio

        cards, _ = asyncio.run(
            generator.generate_cards(
                topic_id="topic-auth",
                topic_title="Authentication",
                doc_content="Authentication verifies the caller identity.",
                count=MAX_CARD_GENERATION_COUNT + 10,
            )
        )
        self.assertLessEqual(len(cards), MAX_CARD_GENERATION_COUNT)

    def test_generate_cards_drops_malformed(self):
        items = [
            {"front": "short", "back": "ok back long enough"},  # front too short
            {"front": "What is Authentication?", "back": ""},  # empty back
            _valid_card(
                "What is Authentication?",
                "Authentication verifies who the caller is before access.",
            ),
        ]
        generator = self._generator(items)
        import asyncio

        cards, stats = asyncio.run(
            generator.generate_cards(
                topic_id="topic-auth",
                topic_title="Authentication",
                doc_content="Authentication verifies the caller identity.",
                count=1,
            )
        )
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["front"], "What is Authentication?")
        self.assertGreater(stats["malformed_items_dropped"], 0)

    def test_validate_card_item_rules(self):
        topic_title = "Authentication"
        anchors = ["Authentication verifies the caller identity."]

        valid, issue = _validate_card_item(
            _valid_card(
                "What is Authentication?",
                "Authentication verifies who the caller is before access.",
            ),
            topic_title=topic_title,
            grounding_anchors=anchors,
        )
        self.assertTrue(valid, issue)

        missing_quote, _ = _validate_card_item(
            {
                "front": "What is Authentication?",
                "back": "Authentication verifies who the caller is before access.",
                "source_section": "Authentication",
            },
            topic_title=topic_title,
            grounding_anchors=anchors,
        )
        self.assertFalse(missing_quote)

        ungrounded, _ = _validate_card_item(
            {
                "front": "What is Kubernetes networking?",
                "back": "Kubernetes networking covers pods and services.",
                "source_quote": "Authentication verifies the caller identity.",
                "source_section": "Authentication",
            },
            topic_title=topic_title,
            grounding_anchors=anchors,
        )
        self.assertFalse(ungrounded)


if __name__ == "__main__":
    unittest.main()
