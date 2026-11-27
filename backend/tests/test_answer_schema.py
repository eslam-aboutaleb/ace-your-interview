"""Tests for the typed rich answer schema (issue #3).

Covers schema round-trips for every section type, the safety validators
(XSS, heading hierarchy, mermaid click handlers), bounded repair of
malformed model outputs, and legacy markdown conversion in both directions.
"""

import unittest

from app.schemas.answers import ANSWER_SCHEMA_VERSION
from app.services.answer_schema import (
    AnswerValidationError,
    markdown_to_rich_answer,
    parse_rich_answer,
    repair_rich_answer,
    rich_answer_to_markdown,
)


def _full_payload():
    return {
        "answer_schema_version": ANSWER_SCHEMA_VERSION,
        "sections": [
            {
                "type": "summary",
                "id": "summary-0",
                "text": "HTTP is a request/response protocol.",
            },
            {
                "type": "explanation",
                "id": "lifecycle-1",
                "title": "Request lifecycle",
                "text": "A request travels through DNS, TCP, TLS and the app server.",
            },
            {
                "type": "code",
                "id": "example-2",
                "title": "Example",
                "language": "python",
                "code": "print('hello')\n",
                "filename": "main.py",
                "line_highlights": [1],
            },
            {
                "type": "table",
                "id": "verbs-3",
                "columns": ["Verb", "Idempotent"],
                "rows": [["GET", "yes"], ["POST", "no"]],
            },
            {
                "type": "diagram",
                "id": "flow-4",
                "format": "mermaid",
                "source": "sequenceDiagram\nClient->>Server: GET /",
                "alt": "Request flow",
            },
            {
                "type": "tradeoffs",
                "id": "tradeoffs-5",
                "items": [
                    {
                        "option": "HTTP/1.1",
                        "pros": ["simple"],
                        "cons": ["head-of-line blocking"],
                    }
                ],
            },
            {
                "type": "common_mistakes",
                "id": "mistakes-6",
                "items": [
                    {
                        "mistake": "Using POST for reads",
                        "correction": "Use GET for safe reads",
                    }
                ],
            },
            {
                "type": "references",
                "id": "refs-7",
                "items": [
                    {
                        "label": "RFC 9110",
                        "source_ref": "https://www.rfc-editor.org/rfc/rfc9110",
                        "quote": "",
                    }
                ],
            },
            {
                "type": "follow_up",
                "id": "follow-8",
                "prompts": ["How does HTTP/2 improve on this?"],
            },
        ],
    }


class SchemaValidationTests(unittest.TestCase):
    def test_full_payload_round_trips(self):
        answer = parse_rich_answer(_full_payload())
        self.assertEqual(answer.answer_schema_version, ANSWER_SCHEMA_VERSION)
        self.assertEqual(len(answer.sections), 9)
        again = parse_rich_answer(answer.model_dump(mode="json"))
        self.assertEqual(again.model_dump(mode="json"), answer.model_dump(mode="json"))

    def test_language_is_normalised(self):
        payload = {
            "sections": [
                {"type": "code", "id": "c-0", "language": "Python", "code": "x = 1"}
            ],
        }
        answer = parse_rich_answer(payload)
        self.assertEqual(answer.sections[0].language, "python")

    def test_duplicate_section_ids_rejected(self):
        payload = {
            "sections": [
                {"type": "summary", "id": "a", "text": "one"},
                {"type": "summary", "id": "a", "text": "two"},
            ]
        }
        with self.assertRaises(AnswerValidationError):
            parse_rich_answer(payload)

    def test_empty_sections_rejected(self):
        with self.assertRaises(AnswerValidationError):
            parse_rich_answer({"sections": []})

    def test_invalid_section_id_rejected(self):
        with self.assertRaises(AnswerValidationError):
            parse_rich_answer(
                {"sections": [{"type": "summary", "id": "Bad ID!", "text": "hi"}]}
            )


class SafetyTests(unittest.TestCase):
    def test_script_tag_rejected(self):
        payload = {
            "sections": [
                {
                    "type": "explanation",
                    "id": "x-0",
                    "text": "hi <script>alert(1)</script>",
                }
            ]
        }
        with self.assertRaises(AnswerValidationError):
            parse_rich_answer(payload)

    def test_event_handler_rejected(self):
        payload = {
            "sections": [
                {
                    "type": "summary",
                    "id": "x-0",
                    "text": '<img src=x onerror="alert(1)">',
                }
            ]
        }
        with self.assertRaises(AnswerValidationError):
            parse_rich_answer(payload)

    def test_javascript_url_rejected(self):
        payload = {
            "sections": [
                {"type": "summary", "id": "x-0", "text": "[click](javascript:alert(1))"}
            ]
        }
        with self.assertRaises(AnswerValidationError):
            parse_rich_answer(payload)

    def test_mermaid_click_javascript_rejected(self):
        payload = {
            "sections": [
                {
                    "type": "diagram",
                    "id": "d-0",
                    "source": 'graph TD\nA-->B\nclick A href "javascript:alert(1)"',
                    "alt": "diagram",
                }
            ]
        }
        with self.assertRaises(AnswerValidationError):
            parse_rich_answer(payload)

    def test_heading_level_skip_rejected(self):
        payload = {
            "sections": [
                {"type": "explanation", "id": "x-0", "text": "## One\n\n#### Skipped"}
            ]
        }
        with self.assertRaises(AnswerValidationError):
            parse_rich_answer(payload)

    def test_table_row_width_mismatch_rejected(self):
        payload = {
            "sections": [
                {
                    "type": "table",
                    "id": "t-0",
                    "columns": ["a", "b"],
                    "rows": [["only-one"]],
                }
            ]
        }
        with self.assertRaises(AnswerValidationError):
            parse_rich_answer(payload)


class RepairTests(unittest.TestCase):
    def test_missing_ids_are_assigned_stably(self):
        result = repair_rich_answer(
            {
                "sections": [
                    {"type": "summary", "text": "Overview."},
                    {"type": "explanation", "title": "Deep dive", "text": "Details."},
                ]
            }
        )
        self.assertIsNotNone(result.answer)
        ids = [s.id for s in result.answer.sections]
        self.assertTrue(all(ids))
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(
            any(r.startswith("assigned_section_id") for r in result.repairs)
        )

    def test_unsafe_html_is_escaped_not_served_raw(self):
        result = repair_rich_answer(
            {
                "sections": [
                    {
                        "type": "explanation",
                        "id": "x-0",
                        "text": "ok <script>alert(1)</script> done",
                    }
                ]
            }
        )
        self.assertIsNotNone(result.answer)
        text = result.answer.sections[0].text
        self.assertNotIn("<script>", text)
        self.assertTrue(
            any(r.startswith("escaped_unsafe_html") for r in result.repairs)
        )
        # The repaired answer re-validates.
        parse_rich_answer(result.answer.model_dump(mode="json"))

    def test_heading_skip_is_demoted(self):
        result = repair_rich_answer(
            {
                "sections": [
                    {
                        "type": "explanation",
                        "id": "x-0",
                        "text": "## One\n\n#### Skipped",
                    }
                ]
            }
        )
        self.assertIsNotNone(result.answer)
        self.assertIn("### Skipped", result.answer.sections[0].text)

    def test_unbalanced_code_fence_is_closed(self):
        result = repair_rich_answer(
            {
                "sections": [
                    {
                        "type": "explanation",
                        "id": "x-0",
                        "text": "intro\n\n```python\nx = 1",
                    }
                ]
            }
        )
        self.assertIsNotNone(result.answer)
        self.assertTrue(result.answer.sections[0].text.rstrip().endswith("```"))

    def test_unknown_section_type_falls_back_to_escaped_text(self):
        result = repair_rich_answer(
            {
                "sections": [
                    {"type": "interactive_widget", "html": "<script>alert(1)</script>"}
                ]
            }
        )
        self.assertIsNotNone(result.answer)
        section = result.answer.sections[0]
        self.assertEqual(section.type, "explanation")
        self.assertNotIn("<script>", section.text)
        self.assertTrue(
            any(r.startswith("unknown_section_type") for r in result.repairs)
        )

    def test_table_rows_are_padded(self):
        result = repair_rich_answer(
            {
                "sections": [
                    {
                        "type": "table",
                        "id": "t-0",
                        "columns": ["a", "b"],
                        "rows": [["1"]],
                    }
                ]
            }
        )
        self.assertIsNotNone(result.answer)
        self.assertEqual(result.answer.sections[0].rows, [["1", ""]])

    def test_diagram_gets_alt_text(self):
        result = repair_rich_answer(
            {"sections": [{"type": "diagram", "source": "graph TD\nA-->B"}]}
        )
        self.assertIsNotNone(result.answer)
        self.assertTrue(result.answer.sections[0].alt)

    def test_bad_language_normalised(self):
        result = repair_rich_answer(
            {
                "sections": [
                    {"type": "code", "language": "C++20!!", "code": "int main(){}"}
                ]
            }
        )
        self.assertIsNotNone(result.answer)
        self.assertEqual(result.answer.sections[0].language, "text")

    def test_unsalvageable_sections_are_dropped(self):
        result = repair_rich_answer(
            {
                "sections": [
                    {"type": "summary", "id": "s-0", "text": "good"},
                    {"type": "code", "id": "c-1", "language": "python", "code": ""},
                ]
            }
        )
        self.assertIsNotNone(result.answer)
        self.assertGreaterEqual(result.dropped_sections, 1)
        self.assertEqual([s.type for s in result.answer.sections], ["summary"])

    def test_duplicate_ids_are_deduplicated(self):
        result = repair_rich_answer(
            {
                "sections": [
                    {"type": "summary", "id": "s-0", "text": "one"},
                    {"type": "summary", "id": "s-0", "text": "two"},
                ]
            }
        )
        self.assertIsNotNone(result.answer)
        ids = [s.id for s in result.answer.sections]
        self.assertEqual(len(set(ids)), 2)

    def test_completely_broken_payload_returns_none(self):
        self.assertIsNone(repair_rich_answer({"sections": "not-a-list"}).answer)
        self.assertIsNone(repair_rich_answer({"sections": []}).answer)
        self.assertIsNone(repair_rich_answer("nope").answer)

    def test_repair_is_bounded(self):
        # Even pathological payloads terminate within the round budget.
        payload = {"sections": [{"type": "summary", "id": "s-0", "text": "ok"}]}
        result = repair_rich_answer(payload, max_rounds=1)
        self.assertIsNotNone(result.answer)


class LegacyConversionTests(unittest.TestCase):
    LEGACY = (
        "HTTP is a request/response protocol.\n"
        "\n"
        "## Request lifecycle\n"
        "\n"
        "DNS, then TCP, then TLS.\n"
        "\n"
        "```python\n"
        "import http.client\n"
        "```\n"
        "\n"
        "## Sequence\n"
        "\n"
        "```mermaid\n"
        "sequenceDiagram\n"
        "Client->>Server: GET /\n"
        "```\n"
    )

    def test_markdown_splits_into_typed_sections(self):
        answer = markdown_to_rich_answer(self.LEGACY)
        kinds = [s.type for s in answer.sections]
        # The "Sequence" heading has no prose body, so it contributes no
        # explanation section; it becomes the diagram's alt text instead.
        self.assertEqual(kinds, ["summary", "explanation", "code", "diagram"])
        self.assertEqual(answer.sections[1].title, "Request lifecycle")
        self.assertEqual(answer.sections[2].language, "python")
        self.assertEqual(answer.sections[3].format, "mermaid")
        self.assertEqual(answer.sections[3].alt, "Sequence")
        # Every converted section carries a stable ID.
        self.assertTrue(all(s.id for s in answer.sections))
        # And the result validates as publishable.
        parse_rich_answer(answer.model_dump(mode="json"))

    def test_legacy_round_trip_is_stable(self):
        answer = markdown_to_rich_answer(self.LEGACY)
        markdown = rich_answer_to_markdown(answer)
        again = markdown_to_rich_answer(markdown)
        self.assertEqual(
            [s.type for s in again.sections],
            [s.type for s in answer.sections],
        )
        self.assertEqual(rich_answer_to_markdown(again), markdown)

    def test_markdown_render_covers_all_section_types(self):
        answer = parse_rich_answer(_full_payload())
        markdown = rich_answer_to_markdown(answer)
        self.assertIn("HTTP is a request/response protocol.", markdown)
        self.assertIn("## Request lifecycle", markdown)
        self.assertIn("```python", markdown)
        self.assertIn("| Verb | Idempotent |", markdown)
        self.assertIn("```mermaid", markdown)
        self.assertIn("**HTTP/1.1**", markdown)
        self.assertIn("Mistake:", markdown)
        self.assertIn("RFC 9110", markdown)
        self.assertIn("How does HTTP/2 improve on this?", markdown)

    def test_empty_markdown_yields_single_summary(self):
        answer = markdown_to_rich_answer("")
        self.assertEqual(len(answer.sections), 1)
        self.assertEqual(answer.sections[0].type, "summary")


class GeneratorIntegrationTests(unittest.TestCase):
    """_validate_question_item attaches or drops sections correctly."""

    def _item(self, sections):
        return {
            "question": "What is HTTP?",
            "answer": "HTTP is a protocol.",
            "difficulty": "easy",
            "sections": sections,
        }

    def test_valid_sections_are_attached(self):
        from app.services.question_generator import _validate_question_item

        item = self._item([{"type": "summary", "text": "Short answer."}])
        valid, issue = _validate_question_item(item, "01-backend", None, "junior")
        self.assertTrue(valid, issue)
        self.assertEqual(item["answer_schema_version"], ANSWER_SCHEMA_VERSION)
        self.assertEqual(len(item["sections"]), 1)
        # Attached sections validate again at the response boundary.
        from app.schemas.models import QuestionAnswerV2

        response = QuestionAnswerV2(
            question_id="q1",
            topic_id="01-backend",
            question=item["question"],
            answer=item["answer"],
            answer_schema_version=item["answer_schema_version"],
            sections=item["sections"],
        )
        self.assertEqual(response.sections[0].type, "summary")

    def test_broken_sections_fall_back_to_legacy_answer(self):
        from app.services.question_generator import _validate_question_item

        item = self._item("this is not a section list")
        valid, issue = _validate_question_item(item, "01-backend", None, "junior")
        self.assertTrue(valid, issue)
        self.assertNotIn("sections", item)
        self.assertEqual(item["answer_schema_version"], "")

    def test_xss_sections_repaired_before_publish(self):
        from app.services.question_generator import _validate_question_item

        item = self._item(
            [{"type": "explanation", "text": "ok <script>alert(1)</script>"}]
        )
        valid, issue = _validate_question_item(item, "01-backend", None, "junior")
        self.assertTrue(valid, issue)
        self.assertNotIn("<script>", item["sections"][0]["text"])


if __name__ == "__main__":
    unittest.main()
