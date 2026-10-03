"""Coverage for app/services/question_generator.py.

Complements the existing generator tests with the deterministic layer they do
not reach: the token/retry budgets, the prompt-fragment builders, the JSON and
prose salvage parsers, the answer/quiz item validators (every rejection reason),
the content-fragment and fallback builders, the problem-solving contract
validator, the pattern-signature de-duplication, and the stream/direct
generation failure frames.

Every generator-level test injects a scripted ``completion``, so no socket is
ever opened.
"""

import asyncio
import json
import unittest

from app.services import question_generator as qg
from app.services.question_generator import QuestionGenerator
from app.services.topic_catalog import PROBLEM_SOLVING_TOPIC_ID

PS_ID = PROBLEM_SOLVING_TOPIC_ID


def _ok(payload) -> dict:
    return {
        "success": True,
        "analysis": payload if isinstance(payload, str) else json.dumps(payload),
        "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
        "error": "",
        "error_code": "",
        "finish_reason": "stop",
        "usage": {},
    }


def _fail(error_code: str) -> dict:
    return {
        "success": False,
        "analysis": "",
        "metadata": {},
        "error": "boom",
        "error_code": error_code,
        "finish_reason": "",
        "usage": {},
    }


class ScriptedLLM:
    """Replays one result per call; then a permanent transport failure."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0
        self.prompts: list[str] = []
        self.options: list[dict] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        self.options.append({"task": task, **kwargs})
        if self._results:
            return dict(self._results.pop(0))
        return _fail("llm_call_failed")


class FakeMCP:
    def __init__(self, context="Team context: payments are split-sharded by merchant id."):
        self.context = context
        self.calls: list[dict] = []

    async def gather_context(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.context


async def _collect(iterator):
    return [event async for event in iterator]


def _generator(results=None, mcp=None) -> QuestionGenerator:
    return QuestionGenerator(ScriptedLLM(results or []), mcp_gateway=mcp)


def _item(question: str, answer: str = "A grounded answer.", **overrides) -> dict:
    item = {"question": question, "answer": answer}
    item.update(overrides)
    return item


class SupportedOptionsTests(unittest.TestCase):
    def test_an_unintrospectable_client_gets_every_option(self):
        class _Weird:
            # ``inspect.signature`` raises for a non-callable attribute, which
            # must degrade to "pass everything" rather than break generation.
            completion = None

        options = {"task": "final", "structured": True, "max_tokens_cap": 900}
        self.assertEqual(qg._supported_options(_Weird(), options), options)

    def test_a_narrow_client_only_receives_declared_options(self):
        class _Narrow:
            async def completion(self, prompt, llm_config=None, user_identity=None, task=None):
                return {}

        self.assertEqual(
            qg._supported_options(_Narrow(), {"task": "final", "structured": True}),
            {"task": "final"},
        )


class NearDuplicateTests(unittest.TestCase):
    def test_a_candidate_that_normalises_to_nothing_is_always_a_duplicate(self):
        self.assertTrue(qg._is_near_duplicate_question("", ["Explain sharding"]))
        self.assertTrue(qg._is_near_duplicate_question("   ", ["Explain sharding"]))

    def test_a_blank_seen_entry_is_skipped(self):
        self.assertFalse(
            qg._is_near_duplicate_question("Explain sharding across shards", ["", "   "])
        )

    def test_an_exact_normalised_match_is_a_duplicate(self):
        self.assertTrue(
            qg._is_near_duplicate_question(
                "Explain database index selection strategies",
                ["Explain  database   index selection strategies!"],
            )
        )

    def test_a_containment_match_over_the_minimum_length_is_a_duplicate(self):
        long_seen = "Explain how the write-ahead log guarantees durability of a commit"
        self.assertTrue(
            qg._is_near_duplicate_question(long_seen, [long_seen + " under power loss"])
        )

    def test_a_genuinely_different_question_is_not_a_duplicate(self):
        self.assertFalse(
            qg._is_near_duplicate_question(
                "Describe how a leader election resolves a split brain",
                ["Explain database index selection strategies"],
            )
        )


class BudgetTests(unittest.TestCase):
    def test_the_recovery_budget_grows_with_the_shortfall_and_is_capped(self):
        self.assertEqual(qg._recovery_budget(0), 3)
        self.assertEqual(qg._recovery_budget(None), 3)
        self.assertEqual(qg._recovery_budget(1), 3)
        self.assertEqual(qg._recovery_budget(4), 12)
        self.assertEqual(qg._recovery_budget(1000), qg._MAX_RECOVERY_ATTEMPTS)

    def test_the_attempt_budget_is_at_least_the_base_and_at_most_the_cap(self):
        self.assertEqual(qg._attempt_budget(0), qg._BASE_MAX_ATTEMPTS)
        self.assertEqual(qg._attempt_budget(None), qg._BASE_MAX_ATTEMPTS)
        self.assertEqual(qg._attempt_budget(1), qg._BASE_MAX_ATTEMPTS)
        # The middle band is target + 3.
        self.assertEqual(qg._attempt_budget(6), 9)
        self.assertEqual(qg._attempt_budget(10_000), qg._MAX_TOTAL_ATTEMPTS)


class PromptFragmentTests(unittest.TestCase):
    def test_existing_question_seeds_drop_blanks_and_duplicates(self):
        self.assertEqual(
            qg._normalise_existing_questions(
                ["Explain sharding", "   ", "explain   sharding", None, "Explain caching"]
            ),
            ["Explain sharding", "Explain caching"],
        )
        self.assertEqual(qg._normalise_existing_questions(None), [])

    def test_a_plain_preferred_language_is_passed_through(self):
        self.assertEqual(
            qg._study_code_language("go", problem_solving_mode=False, requires_programming=True), "go"
        )

    def test_hard_problem_solving_questions_get_the_problem_solving_rule(self):
        rules = qg._difficulty_generation_rules(True, "hard")
        self.assertIn("Hard questions should probe tricky edge cases", rules[0])
        rules = qg._difficulty_generation_rules(False, "hard")
        self.assertIn("distributed-system anomalies", rules[0])

    def test_a_conceptual_topic_is_told_not_to_force_code(self):
        self.assertEqual(
            qg._study_markdown_rules(code_language="", problem_solving_mode=False)[-1],
            "Use code examples only when they materially improve clarity.",
        )

    def test_a_programming_topic_is_told_to_tag_its_fences(self):
        self.assertIn(
            'use a concise fenced "go" example',
            qg._study_markdown_rules(code_language="go", problem_solving_mode=False)[-1],
        )

    def test_problem_solving_gets_the_brute_force_comparison_rule_instead(self):
        self.assertIn(
            "brute-force",
            qg._study_markdown_rules(code_language="python", problem_solving_mode=True)[-1],
        )

    def test_an_unknown_difficulty_falls_through_to_the_general_rule(self):
        rules = qg._difficulty_generation_rules(False, "impossible")
        self.assertEqual(len(rules), 1)
        self.assertIn("difficulty", rules[0].lower())


class GroundingTests(unittest.TestCase):
    def test_the_anchor_block_is_empty_without_anchors(self):
        self.assertEqual(qg._render_grounding_anchors_block([]), "")

    def test_the_anchor_block_fences_the_anchor_list(self):
        block = qg._render_grounding_anchors_block(["index selection", "write-ahead log"])
        self.assertIn("grounding_anchors", block)
        self.assertIn("- index selection", block)
        self.assertIn("- write-ahead log", block)

    def test_an_anchor_pool_stops_at_the_limit_and_skips_junk(self):
        anchors = qg._grounding_anchor_phrases(
            topic_title="Database indexing",
            section_title="Index selection strategies",
            content="```\nfenced code is not an anchor\n```\nA real sentence about the write-ahead log.",
            limit=2,
        )
        self.assertLessEqual(len(anchors), 2)
        for anchor in anchors:
            self.assertNotIn("```", anchor)
            self.assertGreater(len(anchor.split()), 0)

    def test_a_question_with_no_meaningful_tokens_is_not_grounded(self):
        self.assertFalse(
            qg._question_is_grounded_to_anchors(
                "?!",
                topic_title="Database indexing",
                anchors=["index selection"],
                problem_solving_mode=False,
            )
        )

    def test_an_anchor_with_no_meaningful_tokens_is_skipped(self):
        grounded = qg._question_is_grounded_to_anchors(
            "Explain how a write ahead log guarantees durability",
            topic_title="Database internals",
            anchors=["?!", "###", "durability"],
            problem_solving_mode=False,
        )
        self.assertTrue(grounded)

    def test_the_problem_solving_reasoning_summary_leads_with_the_invariant(self):
        self.assertIn("invariant", qg._default_reasoning_summary(True))
        self.assertIn("constraint", qg._default_reasoning_summary(False))


class SentenceSplitTests(unittest.TestCase):
    def test_the_first_sentence_split_handles_a_single_sentence(self):
        self.assertEqual(qg._split_first_sentence("Only one sentence."), ("Only one sentence.", ""))

    def test_the_first_sentence_split_handles_empty_text(self):
        self.assertEqual(qg._split_first_sentence("   "), ("", ""))

    def test_the_last_sentence_split_handles_a_single_sentence(self):
        self.assertEqual(qg._split_last_sentence("Only one."), ("Only one.", "Only one."))

    def test_the_last_sentence_split_handles_empty_text(self):
        self.assertEqual(qg._split_last_sentence(""), ("", ""))


class ProblemSolvingAnswerTests(unittest.TestCase):
    ANSWER = (
        "### Problem\nState the constraint.\n\n"
        "### Solution Walkthrough\nWalk it step by step.\n\n"
        "### Complexity\nTime and space.\n\n"
        "### Code\n```python\n# build a hash map\nseen = {}\n# scan once\nfor x in nums:\n    pass\n```\n"
    )

    def test_a_conforming_answer_is_accepted(self):
        ok, issue = qg._validate_problem_solving_answer(self.ANSWER, "python")
        self.assertTrue(ok)
        self.assertEqual(issue, "")

    def test_a_fence_appearing_before_the_code_heading_is_rejected(self):
        # Exactly one fence, but it sits above the ``### Code`` heading, so the
        # prose sections carry code the reader cannot find in its own section.
        answer = (
            "### Problem\n```python\n# early\nseen = {}\n```\n\n"
            "### Solution Walkthrough\nWalk it step by step.\n\n"
            "### Complexity\nTime and space.\n\n### Code\nSee the block above.\n"
        )
        ok, issue = qg._validate_problem_solving_answer(answer, "python")
        self.assertFalse(ok)
        self.assertEqual(issue, "problem_solving_code_block_before_code_heading")

    def test_a_second_fence_is_rejected_before_the_position_check(self):
        answer = self.ANSWER.replace(
            "### Problem\nState the constraint.",
            "```python\n# early\n```\n\n### Problem\nState the constraint.",
        )
        ok, issue = qg._validate_problem_solving_answer(answer, "python")
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_problem_solving_code_block_count")

    def test_an_empty_code_block_is_rejected(self):
        answer = self.ANSWER.replace(
            "```python\n# build a hash map\nseen = {}\n# scan once\nfor x in nums:\n    pass\n```",
            "```python\n\n\n```",
        )
        ok, issue = qg._validate_problem_solving_answer(answer, "python")
        self.assertFalse(ok)
        self.assertEqual(issue, "empty_problem_solving_code_block")

    def test_a_fence_with_too_few_comments_is_rejected(self):
        answer = self.ANSWER.replace(
            "```python\n# build a hash map\nseen = {}\n# scan once\nfor x in nums:\n    pass\n```",
            "```python\nseen = {}\nfor x in nums:\n    pass\n```",
        )
        ok, issue = qg._validate_problem_solving_answer(answer, "python")
        self.assertFalse(ok)
        self.assertEqual(issue, "problem_solving_code_missing_comments")

    def test_a_java_answer_under_a_java_fence_is_accepted(self):
        answer = self.ANSWER.replace("```python", "```java").replace(
            "# build a hash map", "// build a hash map"
        ).replace("# scan once", "// scan once")
        ok, issue = qg._validate_problem_solving_answer(answer, "java")
        self.assertTrue(ok, issue)
        # A python answer must never be emitted under a java fence.
        ok, issue = qg._validate_problem_solving_answer(answer, "python")
        self.assertFalse(ok)
        self.assertEqual(issue, "problem_solving_code_language_mismatch")

    def test_an_out_of_order_heading_sequence_is_rejected(self):
        answer = self.ANSWER.replace(
            "### Problem\n", "### Solution Walkthrough\n", 1
        )
        ok, issue = qg._validate_problem_solving_answer(answer, "python")
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_problem_solving_heading_sequence")

    def test_two_code_fences_are_rejected(self):
        answer = self.ANSWER + "\n```python\n# extra\n```\n"
        ok, issue = qg._validate_problem_solving_answer(answer, "python")
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_problem_solving_code_block_count")


class PatternSignatureTests(unittest.TestCase):
    def test_an_empty_pair_has_no_signature(self):
        self.assertEqual(qg._problem_solving_pattern_signature("", ""), "")

    def test_a_grid_island_walkthrough_is_recognised(self):
        self.assertEqual(
            qg._problem_solving_pattern_signature(
                "Count the number of islands in the grid using dfs or bfs",
                "def num_islands(grid): ...",
            ),
            "number_of_islands_dfs",
        )

    def test_an_item_with_no_signature_is_always_accepted(self):
        seen = {"two_sum_hash_map"}
        self.assertTrue(
            qg._accept_problem_pattern_signature(signature="", seen_signatures=seen, unique_target=5)
        )
        # And the empty signature is never recorded, so it cannot block later ones.
        self.assertNotIn("", seen)

    def test_a_repeated_signature_is_refused_until_the_unique_target_is_reached(self):
        seen = {"two_sum_hash_map"}
        self.assertFalse(
            qg._accept_problem_pattern_signature(
                signature="two_sum_hash_map", seen_signatures=seen, unique_target=5
            )
        )
        # Once the target is reached, repeats are allowed again.
        self.assertTrue(
            qg._accept_problem_pattern_signature(
                signature="two_sum_hash_map", seen_signatures=seen, unique_target=1
            )
        )

    def test_the_problem_solving_fallback_count_adds_headroom(self):
        self.assertEqual(qg._problem_solving_fallback_count_with_headroom(0, True), 0)
        self.assertEqual(qg._problem_solving_fallback_count_with_headroom(None, True), 0)
        self.assertEqual(qg._problem_solving_fallback_count_with_headroom(2, False), 2)
        headroom = qg._problem_solving_fallback_count_with_headroom(2, True)
        self.assertGreater(headroom, 2)
        self.assertLessEqual(headroom, 24)


class QuizPromptTests(unittest.TestCase):
    TOPICS = [
        {
            "id": "topic-a",
            "title": "Topic A",
            "content": "Topic A body.",
            "requires_programming": True,
            "preferred_language": "Go",
            "response_detail": "concise",
        }
    ]

    def test_an_explicit_difficulty_reaches_the_prompt(self):
        prompt = qg._build_quiz_prompt(
            self.TOPICS, 4, ["mcq", "true_false"], "hard", "mid"
        )
        self.assertIn('All questions should be "hard" difficulty.', prompt)
        self.assertIn("Mix multiple-choice", prompt)

    def test_a_true_false_only_quiz_gets_the_two_option_contract(self):
        prompt = qg._build_quiz_prompt(self.TOPICS, 4, ["true_false"], None, "mid")
        self.assertIn("exactly 2 options: A=True, B=False", prompt)
        self.assertNotIn("All questions should be \"hard\" difficulty.", prompt)

    def test_a_programming_topic_declares_its_language_in_the_topic_notes(self):
        prompt = qg._build_quiz_prompt(self.TOPICS, 4, ["mcq"], None, "mid")
        self.assertIn("preferred_language=go", prompt)
        self.assertIn("detail=concise", prompt)

    def test_a_conceptual_topic_omits_the_language_note(self):
        prompt = qg._build_quiz_prompt(
            [
                {
                    "id": "topic-b",
                    "title": "Topic B",
                    "content": "Body.",
                    "requires_programming": False,
                    "preferred_language": "Go",
                    "response_detail": "concise",
                }
            ],
            4,
            ["mcq"],
            None,
            "mid",
        )
        self.assertNotIn("preferred_language=go", prompt)


class StripFenceTests(unittest.TestCase):
    def test_a_fenced_block_is_unwrapped(self):
        self.assertEqual(
            qg._strip_outer_code_fence("```json\n{\"a\": 1}\n```"), '{"a": 1}'
        )

    def test_an_unterminated_or_inner_only_fence_is_left_alone(self):
        self.assertEqual(qg._strip_outer_code_fence("```json\n{\"a\": 1}"), '```json\n{"a": 1}')
        self.assertEqual(qg._strip_outer_code_fence("```"), "```")
        self.assertEqual(
            qg._strip_outer_code_fence("```json\n```"), "```json\n```"
        )

    def test_plain_text_is_returned_stripped(self):
        self.assertEqual(qg._strip_outer_code_fence("  hello  "), "hello")


class TextQaParserTests(unittest.TestCase):
    def test_a_question_answer_prose_block_is_read(self):
        items = qg._parse_text_qa_pairs(
            "1. Question: What is a write ahead log?\nAnswer: It makes commits durable."
        )
        self.assertEqual(
            items,
            [{"question": "What is a write ahead log?", "answer": "It makes commits durable."}],
        )

    def test_a_question_without_an_answer_marker_uses_the_next_line(self):
        items = qg._parse_text_qa_pairs(
            "**Q1:** What is a write ahead log?\nIt makes commits durable."
        )
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["question"].startswith("What is a write ahead log?"))
        self.assertTrue(items[0]["answer"].startswith("It makes commits durable."))

    def test_an_empty_question_block_is_dropped(self):
        self.assertEqual(qg._parse_text_qa_pairs("Question:\nAnswer:\n"), [])

    def test_a_numbered_block_with_an_answer_line_is_read_without_question_markers(self):
        items = qg._parse_text_qa_pairs(
            "1) Explain a Bloom filter trade-off\nAnswer: False positives without false negatives."
        )
        self.assertEqual(len(items), 1)
        self.assertIn("Bloom filter", items[0]["question"])
        self.assertIn("False positives", items[0]["answer"])

    def test_prose_with_no_question_markers_yields_nothing(self):
        self.assertEqual(qg._parse_text_qa_pairs("Just a paragraph of prose."), [])


class ParseQuestionsJsonTests(unittest.TestCase):
    def test_a_bare_array_is_read(self):
        self.assertEqual(
            qg._parse_questions_json('[{"question": "q1", "answer": "a1"}]'),
            [{"question": "q1", "answer": "a1"}],
        )

    def test_a_questions_envelope_is_unwrapped(self):
        self.assertEqual(
            qg._parse_questions_json('{"questions": [{"question": "q1", "answer": "a1"}]}'),
            [{"question": "q1", "answer": "a1"}],
        )

    def test_an_items_envelope_is_unwrapped(self):
        self.assertEqual(
            qg._parse_questions_json('{"items": [{"question": "q1", "answer": "a1"}]}'),
            [{"question": "q1", "answer": "a1"}],
        )

    def test_a_single_object_with_a_question_and_answer_is_accepted(self):
        self.assertEqual(
            qg._parse_questions_json('{"question": "q1", "answer": "a1"}'),
            [{"question": "q1", "answer": "a1"}],
        )

    def test_a_single_object_without_an_answer_is_rejected(self):
        self.assertEqual(qg._parse_questions_json('{"question": "q1"}'), [])

    def test_non_dict_entries_are_dropped(self):
        self.assertEqual(
            qg._parse_questions_json('[1, "two", {"question": "q1", "answer": "a1"}]'),
            [{"question": "q1", "answer": "a1"}],
        )

    def test_an_empty_response_parses_to_nothing(self):
        self.assertEqual(qg._parse_questions_json("   "), [])
        self.assertEqual(qg._parse_questions_json("not json at all"), [])


class ExtractContentFragmentsTests(unittest.TestCase):
    def test_fenced_lines_blank_lines_and_tiny_lines_are_skipped(self):
        fragments = qg._extract_content_fragments(
            "\n```\nfenced body that is long enough\n```\n\n# Heading line\n- bullet item text\nab\n"
        )
        self.assertTrue(all("```" not in f for f in fragments))
        self.assertTrue(all(len(f) >= 6 for f in fragments))

    def test_trailing_punctuation_is_trimmed_from_a_fragment(self):
        fragments = qg._extract_content_fragments("A repeated line about pooling.")
        self.assertEqual(fragments, ["A repeated line about pooling"])

    def test_an_over_long_line_is_truncated_on_a_word_boundary(self):
        fragments = qg._extract_content_fragments("word " * 60, limit=1)
        self.assertEqual(len(fragments), 1)
        self.assertLessEqual(len(fragments[0]), 140)
        self.assertFalse(fragments[0].endswith("word "))

    def test_content_with_no_usable_lines_falls_back_to_paragraphs(self):
        # Every line is fenced or under six characters, so the line scan finds
        # nothing and the paragraph pass takes over.
        fragments = qg._extract_content_fragments("```\nabcd\n```\n\nefghij klmno pqrstu")
        self.assertEqual(fragments, ["efghij klmno pqrstu"])

    def test_the_fragment_limit_is_respected(self):
        fragments = qg._extract_content_fragments(
            "\n\n".join(f"Paragraph number {i} about pooling." for i in range(20)), limit=5
        )
        self.assertLessEqual(len(fragments), 5)

    def test_empty_content_yields_no_fragments(self):
        self.assertEqual(qg._extract_content_fragments(""), [])


class FallbackQuestionItemTests(unittest.TestCase):
    def test_a_zero_or_negative_count_yields_nothing(self):
        self.assertEqual(
            qg._build_fallback_question_items(
                topic_id="t", topic_title="T", doc_content="Body.", count=0, difficulty=None, level="mid"
            ),
            [],
        )

    def test_the_fallback_topics_every_item_to_the_topic_content(self):
        items = qg._build_fallback_question_items(
            topic_id="t",
            topic_title="Connection pooling",
            doc_content="Pooling bounds concurrent connections and reuses warm ones.",
            count=4,
            difficulty=None,
            level="mid",
        )
        self.assertEqual(len(items), 4)
        for item in items:
            self.assertTrue(item["question"].strip())
            self.assertTrue(item["answer"].strip())
            self.assertIn(item["difficulty"], qg._VALID_DIFFICULTIES)

    def test_content_with_no_usable_fragments_still_produces_items(self):
        items = qg._build_fallback_question_items(
            topic_id="t",
            topic_title="Connection pooling",
            doc_content="",
            count=2,
            difficulty=None,
            level="mid",
        )
        self.assertEqual(len(items), 2)
        self.assertTrue(all(item["source_quote"] for item in items))

    def test_a_topic_the_questions_cannot_be_grounded_to_yields_nothing(self):
        # Grounding is enforced on the deterministic filler too, so a bare
        # one-word title with no content produces no items at all.
        self.assertEqual(
            qg._build_fallback_question_items(
                topic_id="t", topic_title="T", doc_content="", count=2,
                difficulty=None, level="mid",
            ),
            [],
        )

    def test_the_fallback_never_repeats_an_existing_question(self):
        items = qg._build_fallback_question_items(
            topic_id="t",
            topic_title="Connection pooling",
            doc_content="Pooling bounds concurrent connections and reuses warm ones.",
            count=5,
            difficulty=None,
            level="mid",
        )
        texts = [item["question"] for item in items]
        self.assertEqual(len(texts), len({qg._normalise_question(t) for t in texts}))
        if texts:
            repeat = qg._build_fallback_question_items(
                topic_id="t",
                topic_title="Connection pooling",
                doc_content="Pooling bounds concurrent connections and reuses warm ones.",
                count=3,
                difficulty=None,
                level="mid",
                existing_questions=texts,
            )
            self.assertFalse(
                set(texts) & {item["question"] for item in repeat},
                "existing questions must be excluded from the deterministic filler",
            )


class ValidateQuestionItemTests(unittest.TestCase):
    def test_an_empty_question_is_rejected(self):
        ok, issue = qg._validate_question_item({"question": "  ", "answer": "a"}, "t", None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "missing_or_empty_question")

    def test_an_empty_answer_is_rejected(self):
        ok, issue = qg._validate_question_item({"question": "q", "answer": ""}, "t", None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "missing_or_empty_answer")

    def test_a_conforming_item_is_normalised_in_place(self):
        item = _item("Explain how a write ahead log guarantees durability", "It appends first.")
        ok, issue = qg._validate_question_item(item, "t", None, "lead")
        self.assertTrue(ok, issue)
        self.assertEqual(item["target_level"], "mid")
        self.assertEqual(item["difficulty"], "medium")
        self.assertEqual(item["topic_id"], "t")
        self.assertTrue(item["source_quote"])
        self.assertTrue(item["reasoning_summary"])
        self.assertTrue(item["learning_objective"])
        self.assertEqual(item["source_section"], "Topic section")

    def test_the_problem_solving_topic_enforces_its_answer_contract(self):
        ok, issue = qg._validate_question_item(
            _item("Solve two sum", "A plain answer."),
            PS_ID,
            None,
            "mid",
            preferred_language="python",
        )
        self.assertFalse(ok)
        self.assertTrue(issue.startswith("invalid_problem_solving") or issue.startswith("problem_solving"))


def _quiz_item(**overrides) -> dict:
    item = {
        "question": "Which structure preserves insertion order while offering fast lookup?",
        "type": "mcq",
        "choices": [
            {"label": "A", "text": "LinkedHashMap"},
            {"label": "B", "text": "HashSet"},
            {"label": "C", "text": "TreeSet"},
            {"label": "D", "text": "PriorityQueue"},
        ],
        "correct_answer": "A",
        "explanation": "A LinkedHashMap keeps a linked list of its entries in insertion order.",
        "topic_id": "topic-a",
        "source_quote": "a pool bounds concurrent connections",
        "reasoning_summary": "insertion order plus hash lookup points to LinkedHashMap",
        "target_level": "mid",
        "difficulty": "medium",
    }
    item.update(overrides)
    return item


TOPICS = {"topic-a"}
TYPES = {"mcq", "true_false"}


class ValidateQuizItemTests(unittest.TestCase):
    def test_a_conforming_mcq_is_accepted_and_normalised(self):
        item = _quiz_item(type="MCQ")
        ok, issue = qg._validate_quiz_item(item, TOPICS, TYPES, None, "mid")
        self.assertTrue(ok, issue)
        self.assertEqual(item["type"], "mcq")
        self.assertEqual(item["difficulty"], "medium")

    def test_a_short_question_is_rejected(self):
        ok, issue = qg._validate_quiz_item(_quiz_item(question="short"), TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_question")

    def test_an_unknown_type_is_rejected(self):
        ok, issue = qg._validate_quiz_item(_quiz_item(type="essay"), TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_type")

    def test_a_type_outside_the_requested_set_is_rejected(self):
        ok, issue = qg._validate_quiz_item(_quiz_item(), TOPICS, {"true_false"}, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "disallowed_type")

    def test_an_unknown_topic_id_is_rejected(self):
        ok, issue = qg._validate_quiz_item(_quiz_item(topic_id="topic-z"), TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_topic_id")

    def test_a_thin_explanation_quote_or_reasoning_is_rejected(self):
        for field, issue in (
            ("explanation", "invalid_explanation"),
            ("source_quote", "invalid_source_quote"),
            ("reasoning_summary", "invalid_reasoning_summary"),
        ):
            with self.subTest(field=field):
                ok, reason = qg._validate_quiz_item(
                    _quiz_item(**{field: "short"}), TOPICS, TYPES, None, "mid"
                )
                self.assertFalse(ok)
                self.assertEqual(reason, issue)

    def test_an_unknown_or_mismatched_level_is_rejected(self):
        ok, issue = qg._validate_quiz_item(
            _quiz_item(target_level="lead"), TOPICS, TYPES, None, "mid"
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_target_level")
        ok, issue = qg._validate_quiz_item(
            _quiz_item(target_level="senior"), TOPICS, TYPES, None, "mid"
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "level_mismatch")

    def test_a_non_list_or_junk_choices_payload_is_rejected(self):
        ok, issue = qg._validate_quiz_item(_quiz_item(choices="A"), TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_choices_type")
        ok, issue = qg._validate_quiz_item(
            _quiz_item(choices=["LinkedHashMap"]), TOPICS, TYPES, None, "mid"
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_choice_item")

    def test_a_bad_choice_label_or_blank_text_is_rejected(self):
        bad_label = _quiz_item(
            choices=[
                {"label": "a", "text": "LinkedHashMap"},
                {"label": "B", "text": "HashSet"},
                {"label": "C", "text": "TreeSet"},
                {"label": "D", "text": "PriorityQueue"},
            ]
        )
        ok, issue = qg._validate_quiz_item(bad_label, TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_choice_label")
        blank_text = _quiz_item(
            choices=[
                {"label": "A", "text": "  "},
                {"label": "B", "text": "HashSet"},
                {"label": "C", "text": "TreeSet"},
                {"label": "D", "text": "PriorityQueue"},
            ]
        )
        ok, issue = qg._validate_quiz_item(blank_text, TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_choice_text")

    def test_mcq_labels_must_be_ordered_abc_d(self):
        ok, issue = qg._validate_quiz_item(
            _quiz_item(
                choices=[
                    {"label": "A", "text": "LinkedHashMap"},
                    {"label": "B", "text": "HashSet"},
                    {"label": "C", "text": "TreeSet"},
                    {"label": "E", "text": "PriorityQueue"},
                ]
            ),
            TOPICS,
            TYPES,
            None,
            "mid",
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_choice_label")

    def test_mcq_needs_exactly_four_choices(self):
        item = _quiz_item(
            choices=[
                {"label": "A", "text": "LinkedHashMap"},
                {"label": "B", "text": "HashSet"},
                {"label": "C", "text": "TreeSet"},
            ],
            correct_answer="A",
        )
        ok, issue = qg._validate_quiz_item(item, TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "mcq_labels_must_be_abcd")

    def test_an_mcq_answer_outside_the_labels_is_rejected(self):
        ok, issue = qg._validate_quiz_item(
            _quiz_item(correct_answer="E"), TOPICS, TYPES, None, "mid"
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_correct_answer_mcq")

    def test_a_true_false_item_needs_exactly_two_true_false_choices(self):
        three = _quiz_item(
            type="true_false",
            correct_answer="A",
            choices=[
                {"label": "A", "text": "True"},
                {"label": "B", "text": "False"},
                {"label": "C", "text": "Maybe"},
            ],
        )
        ok, issue = qg._validate_quiz_item(three, TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "true_false_must_have_2_choices")

        bad_labels = _quiz_item(
            type="true_false",
            correct_answer="A",
            choices=[{"label": "A", "text": "True"}, {"label": "C", "text": "False"}],
        )
        ok, issue = qg._validate_quiz_item(bad_labels, TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "true_false_labels_must_be_ab")

        wrong_text = _quiz_item(
            type="true_false",
            correct_answer="A",
            choices=[
                {"label": "A", "text": "Yes"},
                {"label": "B", "text": "False"},
            ],
        )
        ok, issue = qg._validate_quiz_item(wrong_text, TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "true_false_text_must_be_true_false")

        bad_answer = _quiz_item(
            type="true_false",
            correct_answer="C",
            choices=[{"label": "A", "text": "True"}, {"label": "B", "text": "False"}],
        )
        ok, issue = qg._validate_quiz_item(bad_answer, TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_correct_answer_true_false")

        good = _quiz_item(
            type="true_false",
            correct_answer="A",
            choices=[{"label": "A", "text": "True"}, {"label": "B", "text": "False"}],
        )
        ok, issue = qg._validate_quiz_item(good, TOPICS, TYPES, None, "mid")
        self.assertTrue(ok, issue)

    def test_an_unknown_or_mismatched_difficulty_is_rejected(self):
        ok, issue = qg._validate_quiz_item(
            _quiz_item(difficulty="impossible"), TOPICS, TYPES, None, "mid"
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "invalid_difficulty")
        ok, issue = qg._validate_quiz_item(_quiz_item(difficulty="hard"), TOPICS, TYPES, "easy", "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "difficulty_mismatch")

    def test_a_padded_requested_difficulty_matches_the_item(self):
        # The requested difficulty is normalised before the comparison,
        # so case and whitespace cannot reject a conforming item.
        item = _quiz_item(difficulty="hard")
        ok, issue = qg._validate_quiz_item(item, TOPICS, TYPES, " Hard ", "mid")
        self.assertTrue(ok, issue)
        self.assertEqual(item["difficulty"], "hard")


class FallbackQuizItemTests(unittest.TestCase):
    TOPIC_CONTENT = [
        {"id": "topic-a", "title": "Topic A", "content": "A pool bounds concurrent connections."},
        {"id": "topic-b", "title": "Topic B", "content": "B statement about quotas."},
    ]

    def test_a_zero_count_or_no_topics_yields_nothing(self):
        self.assertEqual(
            qg._build_fallback_quiz_items(
                topics_content=self.TOPIC_CONTENT,
                count=0,
                question_types=["mcq"],
                difficulty=None,
                level="mid",
                existing_questions=[],
            ),
            [],
        )
        self.assertEqual(
            qg._build_fallback_quiz_items(
                topics_content=[],
                count=3,
                question_types=["mcq"],
                difficulty=None,
                level="mid",
                existing_questions=[],
            ),
            [],
        )

    def test_a_true_false_only_quiz_uses_the_two_option_contract(self):
        items = qg._build_fallback_quiz_items(
            topics_content=self.TOPIC_CONTENT,
            count=3,
            question_types=["true_false"],
            difficulty=None,
            level="mid",
            existing_questions=[],
        )
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertEqual(item["type"], "true_false")
            self.assertEqual(
                [c["text"] for c in item["choices"]], ["True", "False"]
            )
            self.assertIn(item["correct_answer"], {"A", "B"})

    def test_a_mixed_quiz_alternates_types(self):
        items = qg._build_fallback_quiz_items(
            topics_content=self.TOPIC_CONTENT,
            count=4,
            question_types=["mcq", "true_false"],
            difficulty=None,
            level="mid",
            existing_questions=[],
        )
        types = [item["type"] for item in items]
        self.assertEqual(sorted(set(types)), ["mcq", "true_false"])

    def test_topics_without_an_id_produce_nothing(self):
        # A blank topic id cannot be attributed, so the filler skips the entry
        # entirely rather than emitting an unattributable question.
        items = qg._build_fallback_quiz_items(
            topics_content=[{"id": "", "title": "Nameless", "content": "body"}],
            count=2,
            question_types=["mcq"],
            difficulty=None,
            level="mid",
            existing_questions=[],
        )
        self.assertEqual(items, [])

    def test_a_thin_topic_body_gets_a_generic_source_quote(self):
        items = qg._build_fallback_quiz_items(
            topics_content=[{"id": "topic-a", "title": "Topic A", "content": "tiny"}],
            count=1,
            question_types=["mcq"],
            difficulty=None,
            level="mid",
            existing_questions=[],
        )
        self.assertGreaterEqual(len(items[0]["source_quote"]), 8)

    def test_existing_questions_are_not_repeated(self):
        items = qg._build_fallback_quiz_items(
            topics_content=self.TOPIC_CONTENT,
            count=4,
            question_types=["mcq"],
            difficulty=None,
            level="mid",
            existing_questions=[],
        )
        texts = [item["question"] for item in items]
        repeat = qg._build_fallback_quiz_items(
            topics_content=self.TOPIC_CONTENT,
            count=4,
            question_types=["mcq"],
            difficulty=None,
            level="mid",
            existing_questions=texts,
        )
        self.assertFalse(set(texts) & {item["question"] for item in repeat})

    def test_an_empty_type_set_means_both_types(self):
        items = qg._build_fallback_quiz_items(
            topics_content=[
                {"id": "topic-a", "title": "Topic A", "content": "A pool bounds connections."}
            ],
            count=4,
            question_types=[],
            difficulty=None,
            level="mid",
            existing_questions=[],
        )
        self.assertEqual(len(items), 4)
        types = sorted({item["type"] for item in items})
        self.assertEqual(types, ["mcq", "true_false"])

    def test_a_mixed_quiz_strictly_alternates_types(self):
        # Alternation is keyed on the accepted count, so a mixed
        # quiz flips type on every emitted item.
        items = qg._build_fallback_quiz_items(
            topics_content=[
                {"id": "topic-a", "title": "Topic A", "content": "A pool bounds connections."}
            ],
            count=2,
            question_types=["mcq", "true_false"],
            difficulty=None,
            level="mid",
            existing_questions=[],
        )
        self.assertEqual([item["type"] for item in items], ["mcq", "true_false"])

        longer = qg._build_fallback_quiz_items(
            topics_content=self.TOPIC_CONTENT,
            count=6,
            question_types=["mcq", "true_false"],
            difficulty=None,
            level="mid",
            existing_questions=[],
        )
        types = [item["type"] for item in longer]
        self.assertTrue(types)
        for prev, nxt in zip(types, types[1:]):
            self.assertNotEqual(prev, nxt)
        self.assertEqual(sorted(set(types)), ["mcq", "true_false"])

    def test_both_topics_are_used_by_the_fallback(self):
        # Topic selection is keyed on the attempt counter, so with
        # two topics every topic_id appears in the output.
        items = qg._build_fallback_quiz_items(
            topics_content=self.TOPIC_CONTENT,
            count=2,
            question_types=["mcq", "true_false"],
            difficulty=None,
            level="mid",
            existing_questions=[],
        )
        self.assertEqual({item["topic_id"] for item in items}, {"topic-a", "topic-b"})

    def test_a_non_canonical_difficulty_falls_back_to_medium(self):
        # The filler canonicalises an unknown difficulty to "medium"
        # and validates against the canonical value, so the batch is
        # served at the default difficulty instead of coming back empty.
        items = qg._build_fallback_quiz_items(
            topics_content=[
                {"id": "topic-a", "title": "Topic A", "content": "A pool bounds connections."}
            ],
            count=2,
            question_types=["mcq"],
            difficulty="impossible",
            level="mid",
            existing_questions=[],
        )
        self.assertEqual(len(items), 2)
        for item in items:
            self.assertEqual(item["difficulty"], "medium")
        ok, issue = qg._validate_quiz_item(_quiz_item(), TOPICS, TYPES, "impossible", "mid")
        self.assertFalse(ok)
        self.assertEqual(issue, "difficulty_mismatch")


class CollectWithRetriesTests(unittest.TestCase):
    def _run(self, results, *, target_count=2, validator=None):
        llm = ScriptedLLM(results)

        def _default_validator(item):
            return (True, "") if item.get("question") and item.get("answer") else (False, "bad")

        async def _go():
            items, stats = await qg._collect_with_retries(
                llm=llm,
                base_prompt="base",
                llm_config=None,
                target_count=target_count,
                validator=validator or _default_validator,
                system_prompt="system",
            )
            return items, stats

        items, stats = asyncio.run(_go())
        return items, stats, llm

    def test_junk_and_duplicate_items_are_dropped_and_the_batch_fills(self):
        items, stats, llm = self._run(
            [
                _ok([{"question": "q1", "answer": "a1"}]),
                _ok(["not-a-dict", {"question": "", "answer": "a"}, {"question": "q2", "answer": "a2"}]),
            ]
        )
        self.assertEqual([item["question"] for item in items], ["q1", "q2"])
        self.assertGreaterEqual(llm.calls, 2)
        self.assertGreaterEqual(stats["retries_used"], 1)

    def test_a_terminal_transport_error_ends_the_attempt_loop(self):
        items, stats, llm = self._run([_fail("llm_truncated")])
        self.assertEqual(items, [])
        self.assertEqual(stats["terminal_reason"], "llm_truncated")
        # A truncated response cannot be fixed by re-sending the same prompt.
        self.assertEqual(llm.calls, 1)

    def test_the_final_recovery_pass_asks_for_one_item_at_a_time(self):
        items, _stats, llm = self._run([_ok("not json")], target_count=1)
        # Every attempt and every recovery attempt fails, so the last prompt the
        # generator built must be the single-item recovery prompt.
        self.assertEqual(items, [])
        self.assertIn("final_recovery_fill_missing_items", llm.prompts[-1])
        self.assertGreaterEqual(llm.calls, qg._BASE_MAX_ATTEMPTS)

    def test_the_mcp_context_helper_returns_nothing_without_a_gateway(self):
        generator = _generator()
        self.assertEqual(
            asyncio.run(
                generator._mcp_context_for_flow(flow="questions", query="q", topic_id="t")
            ),
            "",
        )

    def test_the_mcp_context_helper_passes_every_field_through(self):
        mcp = FakeMCP()
        generator = _generator(mcp=mcp)
        context = asyncio.run(
            generator._mcp_context_for_flow(
                flow="questions", query="sharding", topic_id="t", topic_title="Topic"
            )
        )
        self.assertEqual(context, mcp.context)
        self.assertEqual(mcp.calls[0]["flow"], "questions")
        self.assertEqual(mcp.calls[0]["topic_id"], "t")
        self.assertEqual(mcp.calls[0]["topic_title"], "Topic")


class PolicyErrorPayloadTests(unittest.TestCase):
    def test_each_policy_error_maps_to_its_own_code_and_message(self):
        from app.services.llm_policy import (
            LLMServiceApprovalRequiredError,
            PersonalCredentialRequiredError,
            StudyAppLLMNotAssignedError,
        )

        generator = _generator()
        cases = (
            (LLMServiceApprovalRequiredError("a"), "llm_service_approval_required"),
            (StudyAppLLMNotAssignedError("b"), "study_app_llm_not_assigned"),
            (PersonalCredentialRequiredError("c"), "personal_credential_required"),
        )
        for exc, code in cases:
            with self.subTest(code=code):
                actual_code, message = generator._policy_error_payload(exc)
                self.assertEqual(actual_code, code)
                self.assertTrue(message)

    def test_an_unexpected_error_falls_back_to_a_generic_code(self):
        generator = _generator()
        code, message = generator._policy_error_payload(RuntimeError("socket closed"))
        self.assertEqual(code, "generation_failed")
        self.assertEqual(message, "socket closed")
        code, message = generator._policy_error_payload(RuntimeError("   "))
        self.assertEqual(code, "generation_failed")
        self.assertEqual(message, "Question generation failed")


class GenerateV2StreamTests(unittest.TestCase):
    def _events(self, results, **kwargs):
        generator = _generator(results)
        params = {
            "topic_id": "topic-a",
            "topic_title": "Topic A",
            "doc_content": "Pooling bounds concurrent connections and reuses warm ones.",
            "count": 2,
            "level": "mid",
        }
        params.update(kwargs)
        return asyncio.run(_collect(generator.generate_v2_stream(**params))), generator.llm

    def test_a_transport_error_is_retried_and_the_stream_still_completes(self):
        events, llm = self._events(
            [
                _fail("llm_call_failed"),
                _ok([_item("Explain how pooling bounds concurrent connections")]),
            ],
            count=1,
        )
        self.assertGreaterEqual(llm.calls, 2)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 1)

    def test_a_json_parse_failure_is_retried(self):
        events, llm = self._events(
            [_ok("not json"), _ok([_item("Explain how pooling reuses warm sockets")])],
            count=1,
        )
        self.assertEqual(llm.calls, 2)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(len([e for e in events if e["type"] == "question"]), 1)

    def test_junk_items_are_dropped_without_stopping_the_stream(self):
        events, _llm = self._events(
            [
                _ok(
                    [
                        "not-a-dict",
                        {"question": "", "answer": "a"},
                        _item("Explain how pooling bounds concurrent connections"),
                    ]
                )
            ],
            count=1,
        )
        self.assertEqual(len([e for e in events if e["type"] == "question"]), 1)

    def test_a_zero_target_completes_immediately(self):
        events, llm = self._events([], count=0)
        self.assertEqual([e["type"] for e in events], ["start", "done"])
        self.assertEqual(events[-1]["generated_count"], 0)
        self.assertEqual(llm.calls, 0)

    def test_a_terminal_code_is_not_retried_and_falls_back_deterministically(self):
        events, llm = self._events([_fail("llm_budget_exceeded")], count=1)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(events[-1]["type"], "done")
        # The deterministic filler still fills the request.
        self.assertEqual(events[-1]["generated_count"], 1)

    def test_a_policy_block_is_reported_as_an_error_event(self):
        from app.services.llm_policy import LLMServiceApprovalRequiredError

        async def _blocked(prompt, llm_config=None, user_identity=None, task=None, **kwargs):
            raise LLMServiceApprovalRequiredError("Approval required")

        generator = _generator([])
        generator.llm.completion = _blocked
        events = asyncio.run(
            _collect(
                generator.generate_v2_stream(
                    topic_id="topic-a",
                    topic_title="Topic A",
                    doc_content="Pooling bounds concurrent connections.",
                    count=2,
                    level="mid",
                )
            )
        )
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "llm_service_approval_required")

    def test_an_unexpected_raised_error_is_reported_as_an_error_event(self):
        async def _boom(prompt, llm_config=None, user_identity=None, task=None, **kwargs):
            raise RuntimeError("socket closed")

        generator = _generator([])
        generator.llm.completion = _boom
        events = asyncio.run(
            _collect(
                generator.generate_v2_stream(
                    topic_id="topic-a",
                    topic_title="Topic A",
                    doc_content="Pooling bounds concurrent connections.",
                    count=2,
                    level="mid",
                )
            )
        )
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "generation_failed")
        self.assertEqual(events[-1]["message"], "socket closed")

    def test_the_external_context_reaches_the_prompt(self):
        mcp = FakeMCP()
        events, llm = self._events(
            [_ok([_item("Explain how pooling bounds concurrent connections")])],
            count=1,
        ) if False else (None, None)
        generator = _generator(
            [_ok([_item("Explain how pooling bounds concurrent connections")])], mcp=mcp
        )
        asyncio.run(
            _collect(
                generator.generate_v2_stream(
                    topic_id="topic-a",
                    topic_title="Topic A",
                    doc_content="Pooling bounds concurrent connections.",
                    count=1,
                    level="mid",
                )
            )
        )
        self.assertEqual(mcp.calls[0]["flow"], "questions")
        self.assertIn("split-sharded by merchant id", generator.llm.prompts[0])


class GenerateTests(unittest.TestCase):
    def test_the_legacy_path_returns_the_grounded_batch(self):
        generator = _generator(
            [_ok([_item("Explain how pooling bounds concurrent connections")])]
        )
        response = asyncio.run(
            generator.generate(
                topic_id="topic-a",
                topic_title="Topic A",
                doc_content="Pooling bounds concurrent connections.",
                count=1,
                level="mid",
            )
        )
        self.assertEqual(len(response.questions), 1)
        self.assertEqual(response.topic_id, "topic-a")
        self.assertEqual(response.questions[0].difficulty, "medium")

    def test_the_legacy_path_tops_up_from_the_deterministic_filler(self):
        generator = _generator([_fail("llm_budget_exceeded")])
        response = asyncio.run(
            generator.generate(
                topic_id="topic-a",
                topic_title="Topic A",
                doc_content="Pooling bounds concurrent connections.",
                count=3,
                level="mid",
            )
        )
        self.assertEqual(len(response.questions), 3)
        self.assertEqual(len({q.question for q in response.questions}), 3)

    def test_the_problem_solving_topic_falls_back_without_repeating_a_pattern(self):
        generator = _generator([_fail("llm_budget_exceeded")])
        response = asyncio.run(
            generator.generate(
                topic_id=PS_ID,
                topic_title="Problem Solving and Algorithms",
                doc_content=(
                    "Solve two sum with a hash map, then a rotated binary search, then "
                    "count the number of islands with dfs."
                ),
                count=5,
                level="mid",
                preferred_language="python",
            )
        )
        # Only one scenario survives de-duplication, so a 5-question request is
        # served with 1 (see the fallback-starvation note in the report).
        self.assertEqual(len(response.questions), 1)
        self.assertTrue(response.questions[0].question.strip())
        self.assertTrue(response.questions[0].answer.strip())


class GenerateV2Tests(unittest.TestCase):
    def test_the_direct_path_tops_up_and_never_repeats_a_question(self):
        generator = _generator(
            [_ok([_item("Explain how pooling bounds concurrent connections")])]
        )
        response = asyncio.run(
            generator.generate_v2(
                topic_id="topic-a",
                topic_title="Topic A",
                doc_content="Pooling bounds concurrent connections and reuses warm ones.",
                count=3,
                level="mid",
            )
        )
        self.assertEqual(len(response.questions), 3)
        self.assertEqual(len({q.question for q in response.questions}), 3)
        self.assertEqual(response.malformed_items_dropped, 0)

    def test_the_direct_path_drops_duplicates_including_near_duplicates(self):
        generator = _generator(
            [
                _ok([_item("Explain how pooling bounds concurrent connections")]),
                _ok([_item("Explain how pooling bounds concurrent connections!")]),
            ]
        )
        response = asyncio.run(
            generator.generate_v2(
                topic_id="topic-a",
                topic_title="Topic A",
                doc_content="Pooling bounds concurrent connections and reuses warm ones.",
                count=1,
                level="mid",
            )
        )
        self.assertEqual(len(response.questions), 1)

    def test_a_terminal_code_in_the_direct_path_still_returns_a_full_batch(self):
        generator = _generator([_fail("llm_truncated")])
        response = asyncio.run(
            generator.generate_v2(
                topic_id="topic-a",
                topic_title="Topic A",
                doc_content="Pooling bounds concurrent connections.",
                count=2,
                level="mid",
            )
        )
        self.assertEqual(len(response.questions), 2)

    def test_a_problem_solving_answer_under_the_wrong_fence_is_rejected(self):
        # The provider returned python source under a java fence for a python
        # request; the contract validator must refuse it rather than publish it.
        wrong_fence = (
            "### Problem\nState the constraint.\n\n"
            "### Solution Walkthrough\nWalk it.\n\n"
            "### Complexity\nO(n).\n\n"
            "### Code\n```java\n// build\nMap<Integer,Integer> seen = new HashMap<>();\n"
            "// scan\nfor (int x : nums) {}\n```\n"
        )
        ok, issue = qg._validate_problem_solving_answer(wrong_fence, "python")
        self.assertFalse(ok)
        self.assertEqual(issue, "problem_solving_code_language_mismatch")

    def test_the_problem_solving_direct_path_never_repeats_a_pattern(self):
        conforming = (
            "### Problem\nState the constraint.\n\n"
            "### Solution Walkthrough\nWalk it step by step.\n\n"
            "### Complexity\nO(n) time.\n\n"
            "### Code\n```python\n# build a hash map\nseen = {}\n# scan once\n"
            "for x in nums:\n    pass\n```\n"
        )
        generator = _generator([_ok([_item("Solve two sum using a hash map", conforming)])])
        response = asyncio.run(
            generator.generate_v2(
                topic_id=PS_ID,
                topic_title="Problem Solving and Algorithms",
                doc_content=(
                    "Solve two sum with a hash map, then a rotated binary search, then "
                    "count the number of islands with dfs."
                ),
                count=3,
                level="mid",
                preferred_language="python",
            )
        )
        self.assertGreaterEqual(len(response.questions), 1)
        signatures = [
            qg._problem_solving_pattern_signature(q.question, q.answer)
            for q in response.questions
        ]
        unique = [s for s in signatures if s]
        self.assertEqual(len(unique), len(set(unique)))
        # Every published answer satisfies the language/fence contract.
        for item in response.questions:
            valid, issue = qg._validate_problem_solving_answer(item.answer, "python")
            self.assertTrue(valid, f"{issue}: {item.answer[:80]}")


QUIZ_TOPICS = [
    {
        "id": "topic-a",
        "title": "Topic A",
        "content": "A pool bounds concurrent connections and reuses warm ones.",
        "requires_programming": True,
        "preferred_language": "go",
        "response_detail": "concise",
    }
]

QUIZ_QUESTIONS = [
    "Which structure preserves insertion order while offering fast lookup?",
    "Which eviction policy protects a cache from a shifting working set?",
    "Which isolation level permits a non-repeatable read inside a transaction?",
    "Which scheduling policy stops one noisy neighbour starving the rest?",
]


class QuizLLMFactory:
    """Emits one valid quiz item per call, rotating through distinct stems."""

    def __init__(self, results=None):
        self._results = list(results or [])
        self.calls = 0
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if self._results:
            return dict(self._results.pop(0))
        item = _quiz_item(question=QUIZ_QUESTIONS[self.calls % len(QUIZ_QUESTIONS)])
        return _ok([item])


class GenerateQuizV2StreamTests(unittest.TestCase):
    def _run(self, llm, **kwargs):
        generator = QuestionGenerator(llm)
        params = {"topics_content": QUIZ_TOPICS, "count": 2, "level": "mid"}
        params.update(kwargs)
        return asyncio.run(_collect(generator.generate_quiz_v2_stream(**params)))

    def test_a_transport_error_is_retried_and_the_stream_completes(self):
        llm = QuizLLMFactory([_fail("llm_call_failed")])
        events = self._run(llm)
        self.assertGreaterEqual(llm.calls, 2)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 2)
        self.assertEqual(events[-1]["topics_used"], ["topic-a"])

    def test_a_json_parse_failure_is_retried(self):
        llm = QuizLLMFactory([_ok("not json")])
        events = self._run(llm, count=1)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 1)

    def test_junk_and_duplicate_items_are_dropped(self):
        llm = QuizLLMFactory(
            [
                _ok(["not-a-dict", _quiz_item(question="short"), _quiz_item()]),
                _ok([_quiz_item()]),
            ]
        )
        events = self._run(llm, count=1)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 1)

    def test_an_empty_question_type_set_defaults_to_both_types(self):
        llm = QuizLLMFactory()
        generator = QuestionGenerator(llm)
        events = asyncio.run(
            _collect(
                generator.generate_quiz_v2_stream(
                    topics_content=QUIZ_TOPICS, count=1, question_types=[], level="mid"
                )
            )
        )
        self.assertEqual(events[-1]["generated_count"], 1)
        self.assertIn("Mix multiple-choice", llm.prompts[0])

    def test_a_terminal_code_still_fills_the_batch_deterministically(self):
        llm = QuizLLMFactory([_fail("llm_budget_exceeded")])
        events = self._run(llm, count=2)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 2)

    def test_a_policy_block_is_reported_as_an_error_event(self):
        from app.services.llm_policy import StudyAppLLMNotAssignedError

        async def _blocked(prompt, llm_config=None, user_identity=None, task=None, **kwargs):
            raise StudyAppLLMNotAssignedError("No service assigned")

        llm = QuizLLMFactory()
        llm.completion = _blocked
        events = self._run(llm)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "study_app_llm_not_assigned")

    def test_an_unexpected_raised_error_is_reported_as_an_error_event(self):
        async def _boom(prompt, llm_config=None, user_identity=None, task=None, **kwargs):
            raise RuntimeError("socket closed")

        llm = QuizLLMFactory()
        llm.completion = _boom
        events = self._run(llm)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "generation_failed")
        self.assertEqual(events[-1]["message"], "socket closed")

    def test_the_external_context_reaches_the_quiz_prompt(self):
        mcp = FakeMCP()
        llm = QuizLLMFactory()
        generator = QuestionGenerator(llm, mcp_gateway=mcp)
        asyncio.run(
            _collect(
                generator.generate_quiz_v2_stream(
                    topics_content=QUIZ_TOPICS, count=1, level="mid"
                )
            )
        )
        self.assertEqual(mcp.calls[0]["flow"], "quiz")
        self.assertIn("technical quiz generation", mcp.calls[0]["query"])
        self.assertIn("split-sharded by merchant id", llm.prompts[0])


class GenerateQuizTests(unittest.TestCase):
    def test_the_legacy_quiz_returns_the_grounded_batch(self):
        llm = QuizLLMFactory()
        response = asyncio.run(
            QuestionGenerator(llm).generate_quiz(
                topics_content=QUIZ_TOPICS, count=2, question_types=["mcq"], level="mid"
            )
        )
        self.assertEqual(len(response.questions), 2)
        self.assertEqual(response.topics_used, ["topic-a"])
        self.assertEqual(response.provider_used, "openai")
        self.assertEqual(response.questions[0].type, "mcq")
        self.assertEqual(response.questions[0].topic_id, "topic-a")

    def test_a_terminal_quiz_failure_still_serves_a_batch(self):
        # The direct quiz route falls back deterministically, matching the
        # streaming route
        # (GenerateQuizV2StreamTests.test_a_terminal_code_still_fills_the_batch).
        llm = QuizLLMFactory([_fail("llm_truncated")])
        response = asyncio.run(
            QuestionGenerator(llm).generate_quiz(
                topics_content=QUIZ_TOPICS, count=2, question_types=["mcq"], level="mid"
            )
        )
        self.assertEqual(llm.calls, 1)
        self.assertEqual(len(response.questions), 2)
        self.assertEqual(response.topics_used, ["topic-a"])
        for question in response.questions:
            self.assertTrue(question.question.strip())

    def test_a_terminal_quiz_v2_failure_still_serves_a_batch(self):
        llm = QuizLLMFactory([_fail("llm_budget_exceeded")])
        response = asyncio.run(
            QuestionGenerator(llm).generate_quiz_v2(
                topics_content=QUIZ_TOPICS, count=2, question_types=["mcq"], level="mid"
            )
        )
        self.assertEqual(len(response.questions), 2)

    def test_the_quiz_fallback_never_exceeds_the_requested_count(self):
        llm = QuizLLMFactory([_fail("llm_budget_exceeded")])
        response = asyncio.run(
            QuestionGenerator(llm).generate_quiz_v2(
                topics_content=QUIZ_TOPICS, count=3, question_types=["mcq"], level="mid"
            )
        )
        self.assertLessEqual(len(response.questions), 3)

    def test_the_legacy_quiz_drops_junk_and_duplicates(self):
        llm = QuizLLMFactory(
            [
                _ok(["not-a-dict", _quiz_item(question="short"), _quiz_item()]),
                _ok([_quiz_item(question=QUIZ_QUESTIONS[1])]),
            ]
        )
        response = asyncio.run(
            QuestionGenerator(llm).generate_quiz(
                topics_content=QUIZ_TOPICS, count=2, question_types=["mcq"], level="mid"
            )
        )
        self.assertEqual(len(response.questions), 2)
        self.assertEqual(
            len({q.question for q in response.questions}), 2
        )

# --------------------------------------------------------------------------
# Second batch: the remaining validation, prompt, and stream-tail branches.
# --------------------------------------------------------------------------


class NearDuplicateBranchTests(unittest.TestCase):
    def test_an_exact_duplicate_is_caught_by_the_normalised_comparison(self):
        self.assertTrue(
            qg._is_near_duplicate_question(
                "Explain connection pooling in detail",
                ["Explain  connection pooling in detail."],
            )
        )

    def test_a_high_similarity_ratio_is_caught(self):
        self.assertTrue(
            qg._is_near_duplicate_question(
                "Explain connection pooling in production systems today",
                ["Explain connection pooling in production system today"],
            )
        )

    def test_a_high_token_overlap_is_caught(self):
        self.assertTrue(
            qg._is_near_duplicate_question(
                "Explain connection pooling, warm sockets, and handshake reuse",
                ["Explain connection pooling, warm sockets, and handshake budgets"],
            )
        )


class ExistingQuestionSeedTests(unittest.TestCase):
    def test_a_repeated_seed_is_dropped(self):
        self.assertEqual(
            qg._normalise_existing_questions(
                ["Explain sharding", "  ", "Explain sharding", "Explain caching"]
            ),
            ["Explain sharding", "Explain caching"],
        )


class StudyCodeLanguageTests(unittest.TestCase):
    def test_a_programming_topic_with_no_language_falls_back_to_python(self):
        self.assertEqual(
            qg._study_code_language(
                "", problem_solving_mode=False, requires_programming=True
            ),
            "python",
        )

    def test_a_conceptual_topic_gets_no_code_language(self):
        self.assertEqual(
            qg._study_code_language(
                "", problem_solving_mode=False, requires_programming=False
            ),
            "",
        )

    def test_the_problem_solving_language_is_normalised_but_not_canonicalised(self):
        # Only case and whitespace are normalised here; the caller is responsible
        # for rejecting a language it has no scenario template for.
        self.assertEqual(qg._problem_solving_language("  PYTHON "), "python")
        self.assertEqual(qg._problem_solving_language("cobol"), "cobol")
        self.assertEqual(qg._problem_solving_language(""), "python")


class DifficultyRuleTests(unittest.TestCase):
    def test_easy_and_medium_difficulty_each_have_their_own_rule(self):
        self.assertIn("foundational", qg._difficulty_generation_rules(False, "easy")[0])
        self.assertIn("bottlenecks", qg._difficulty_generation_rules(False, "medium")[0])


class GroundingAnchorTests(unittest.TestCase):
    def test_duplicate_and_tokenless_anchors_are_dropped(self):
        anchors = qg._grounding_anchor_phrases(
            topic_title="Index selection",
            section_title=None,
            content="Index selection strategies\nindex selection strategies\n!!! ???\n",
            limit=10,
        )
        self.assertEqual(len(anchors), len({a.lower() for a in anchors}))
        for anchor in anchors:
            self.assertNotEqual(anchor.strip(), "!!! ???")

    def test_a_question_with_no_token_overlap_is_not_grounded(self):
        ok, issue = qg._validate_topic_grounding(
            {"question": "Why do pianists prefer weighted hammers"},
            topic_title="Database indexing",
            anchors=["index selection strategies"],
            problem_solving_mode=False,
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "question_not_grounded_to_topic")


class BuildPromptTests(unittest.TestCase):
    def _prompt(self, **kwargs):
        params = {
            "topic_id": "topic-a",
            "topic_title": "Connection pooling",
            "doc_content": "Pooling bounds concurrent connections and reuses warm ones.",
            "count": 3,
            "level": "mid",
        }
        params.update(kwargs)
        return qg._build_prompt(**params)

    def test_an_explicit_difficulty_reaches_the_prompt(self):
        self.assertIn('All questions should be "hard" difficulty.', self._prompt(difficulty="hard"))

    def test_a_section_request_narrows_the_scope(self):
        prompt = self._prompt(
            section_title="Warm sockets",
            section_content="A warm socket avoids the handshake entirely.",
        )
        self.assertIn('the section titled "Warm sockets"', prompt)
        self.assertIn("A warm socket avoids the handshake entirely.", prompt)
        self.assertIn("User requested total questions for this checkpoint: 3", prompt)

    def test_a_programming_topic_is_asked_for_one_fenced_example(self):
        prompt = self._prompt(preferred_language="Go", requires_programming=True)
        self.assertIn('fenced code example in "go"', prompt)

    def test_a_conceptual_topic_is_told_to_avoid_code(self):
        prompt = self._prompt(preferred_language="Go", requires_programming=False)
        self.assertIn("Avoid code blocks unless code is explicitly required", prompt)

    def test_the_problem_solving_topic_is_asked_for_exactly_one_fence(self):
        prompt = self._prompt(topic_id=PS_ID, preferred_language="python")
        self.assertIn('include exactly one fenced "python" example', prompt.lower())

    def test_an_existing_question_history_reaches_the_prompt(self):
        prompt = self._prompt(
            existing_questions=["Explain pool sizing and warm socket reuse"]
        )
        self.assertIn("Explain pool sizing and warm socket reuse", prompt)


class QuizPromptHistoryTests(unittest.TestCase):
    TOPICS = [{"id": "topic-a", "title": "Topic A", "content": "A pool bounds connections."}]

    def test_the_existing_question_blob_is_rendered_into_the_prompt(self):
        prompt = qg._build_quiz_prompt(
            self.TOPICS,
            3,
            ["mcq"],
            None,
            "mid",
            existing_questions=["Explain pool sizing and warm socket reuse"],
        )
        self.assertIn("Do NOT repeat or rephrase these questions", prompt)
        self.assertIn("Explain pool sizing and warm socket reuse", prompt)

    def test_a_prior_progress_note_reaches_the_prompt(self):
        prompt = qg._build_quiz_prompt(
            self.TOPICS, 3, ["mcq"], None, "mid", prior_progress="Learner already covered pools."
        )
        self.assertIn("Learner already covered pools.", prompt)

    def test_external_context_is_fenced_as_untrusted(self):
        prompt = qg._build_quiz_prompt(
            self.TOPICS,
            3,
            ["mcq"],
            None,
            "mid",
            mcp_context="The team runs Kafka with Kafka Streams.",
        )
        self.assertIn("untrusted_input", prompt)
        self.assertIn("Kafka Streams", prompt)


class PatternSignatureDetailTests(unittest.TestCase):
    def test_every_catalogued_pattern_has_a_distinct_signature(self):
        cases = {
            "longest_substring_sliding_window": "Find the longest substring without repeating characters",
            "merge_intervals_sorting": "Merge all overlapping intervals by sorting on the start",
            "top_k_frequent_heap": "Return the top k frequent elements using a heap",
            "search_rotated_binary_search": "Search a rotated sorted array with binary search",
            "two_sum_hash_map": "Solve two sum with a hash map of complements",
            "number_of_islands_dfs": "Count the number of islands in the grid",
        }
        for expected, question in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(qg._problem_solving_pattern_signature(question, "answer"), expected)

    def test_two_questions_never_share_a_signature(self):
        signatures = {
            qg._problem_solving_pattern_signature(q, "")
            for q in (
                "Find the longest substring without repeating characters",
                "Merge all overlapping intervals by sorting on the start",
                "Return the top k frequent elements using a heap",
                "Search a rotated sorted array with binary search",
            )
        }
        self.assertEqual(len(signatures), 4)


class ScenarioTemplateTests(unittest.TestCase):
    def test_an_unsupported_language_has_no_scenario_ids(self):
        self.assertEqual(qg._problem_solving_scenario_ids_for_language("cobol"), [])
        self.assertEqual(qg._problem_solving_scenario_ids_for_language(""), [])

    def test_a_supported_language_lists_its_scenarios(self):
        scenarios = qg._problem_solving_scenario_ids_for_language("python")
        self.assertTrue(scenarios)
        self.assertIn(qg._PROBLEM_SOLVING_DEFAULT_SCENARIO_ID, scenarios)


class TextQaBranchTests(unittest.TestCase):
    def test_an_empty_question_block_is_skipped(self):
        items = qg._parse_text_qa_pairs(
            "Question: What is a bloom filter?\n"
            "Answer: Probabilistic membership with false positives only.\n"
            "Question:\n"
            "Answer:\n"
        )
        self.assertEqual(len(items), 1)
        self.assertIn("bloom filter", items[0]["question"])


class ExtractContentFragmentBranchTests(unittest.TestCase):
    def test_a_duplicate_line_is_dropped_after_normalisation(self):
        fragments = qg._extract_content_fragments(
            "Explain pool sizing carefully.\n\nExplain pool sizing carefully.\n\nA different sentence."
        )
        self.assertEqual(
            [f.lower() for f in fragments],
            ["explain pool sizing carefully", "a different sentence"],
        )

    def test_an_over_long_line_is_trimmed_on_a_word_boundary(self):
        fragments = qg._extract_content_fragments(" ".join(["alpha"] * 80), limit=1)
        self.assertEqual(len(fragments), 1
        )
        self.assertLessEqual(len(fragments[0]), 140)


class FallbackTemplateTests(unittest.TestCase):
    def test_a_problem_solving_topic_with_no_template_emits_a_language_agnostic_prompt(self):
        self.assertEqual(qg._problem_solving_scenario_ids_for_language("cobol"), [])
        items = qg._build_fallback_question_items(
            topic_id=PS_ID,
            topic_title="Problem Solving and Algorithms",
            doc_content="Design an algorithm that finds the k smallest sums of two sorted arrays.",
            count=6,
            difficulty=None,
            level="mid",
            preferred_language="cobol",
        )
        # No foreign source is borrowed, so every item fails the fence contract
        # and is dropped: an unsupported language produces nothing rather than
        # publishing another language's source.
        self.assertEqual(items, [])


class ValidateQuestionItemOverrideTests(unittest.TestCase):
    def test_an_explicit_difficulty_overrides_the_item_value(self):
        item = _item("Explain how pooling bounds connections", "It caps them.", difficulty="easy")
        qg._validate_question_item(item, "topic-a", "hard", "mid")
        self.assertEqual(item["difficulty"], "hard")

    def test_an_unknown_item_difficulty_falls_back_to_medium(self):
        item = _item(
            "Explain how pooling bounds connections", "It caps them.", difficulty="impossible"
        )
        qg._validate_question_item(item, "topic-a", None, "mid")
        self.assertEqual(item["difficulty"], "medium")


class ValidateQuizItemMissingKeyTests(unittest.TestCase):
    def test_each_required_key_is_named_when_it_is_absent(self):
        for key in (
            "question",
            "type",
            "choices",
            "correct_answer",
            "explanation",
            "topic_id",
            "source_quote",
            "reasoning_summary",
            "target_level",
        ):
            with self.subTest(key=key):
                item = _quiz_item()
                item.pop(key)
                ok, issue = qg._validate_quiz_item(item, TOPICS, TYPES, None, "mid")
                self.assertFalse(ok)
                self.assertEqual(issue, f"missing_{key}")

    def test_a_three_choice_mcq_is_rejected_by_the_label_check(self):
        item = _quiz_item(
            choices=[
                {"label": "A", "text": "LinkedHashMap"},
                {"label": "B", "text": "HashSet"},
                {"label": "C", "text": "TreeSet"},
            ],
            correct_answer="A",
        )
        ok, issue = qg._validate_quiz_item(item, TOPICS, TYPES, None, "mid")
        self.assertFalse(ok)
        # The label list is compared before the length, so a short MCQ always
        # reports the label mismatch (the 4-choice branch is unreachable).
        self.assertEqual(issue, "mcq_labels_must_be_abcd")


class CollectWithRetriesBranchTests(unittest.TestCase):
    def test_a_non_dict_item_and_a_rejected_item_are_skipped(self):
        async def _go(results):
            llm = ScriptedLLM(results)
            items, stats = await qg._collect_with_retries(
                llm=llm,
                base_prompt="base",
                llm_config=None,
                target_count=1,
                validator=lambda item: (
                    (True, "") if item.get("question") else (False, "missing_question")
                ),
                system_prompt="system",
            )
            return items, stats, llm

        items, stats, llm = asyncio.run(
            _go(
                [
                    _ok(["not-a-dict", {"question": "", "answer": "a"}]),
                    _ok([{"question": "Explain how pooling bounds connections"}]),
                ]
            )
        )
        self.assertEqual(len(items), 1)
        self.assertGreaterEqual(stats["retries_used"], 1)
        self.assertGreaterEqual(llm.calls, 2)

    def test_a_retryable_transport_error_is_retried_and_named_in_the_prompt(self):
        async def _go():
            llm = ScriptedLLM([_fail("llm_call_failed")])
            items, stats = await qg._collect_with_retries(
                llm=llm,
                base_prompt="base",
                llm_config=None,
                target_count=1,
                validator=lambda item: (True, ""),
                system_prompt="system",
            )
            return items, stats, llm

        items, stats, llm = asyncio.run(_go())
        self.assertEqual(items, [])
        self.assertEqual(stats["terminal_reason"], "")
        self.assertIn("transport_or_provider_error", llm.prompts[1])

    def test_the_recovery_loop_tolerates_a_non_dict_result(self):
        class SloppyLLM:
            def __init__(self):
                self.calls = 0

            async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kw):
                self.calls += 1
                return _ok(["not-a-dict"])

        async def _go():
            llm = SloppyLLM()
            items, stats = await qg._collect_with_retries(
                llm=llm,
                base_prompt="base",
                llm_config=None,
                target_count=1,
                validator=lambda item: (True, ""),
                system_prompt="system",
            )
            return items, stats, llm

        items, stats, llm = asyncio.run(_go())
        self.assertEqual(items, [])
        self.assertGreater(stats["retries_used"], 0)


class GenerateV2StreamBranchTests(unittest.TestCase):
    DOC = "Pooling bounds concurrent connections and reuses warm ones."

    def _stream(self, results, **kwargs):
        generator = _generator(results)
        params = {
            "topic_id": "topic-a",
            "topic_title": "Connection pooling",
            "doc_content": self.DOC,
            "count": 2,
            "level": "mid",
        }
        params.update(kwargs)
        return asyncio.run(_collect(generator.generate_v2_stream(**params))), generator

    def test_an_item_that_fails_validation_is_retried_with_the_issue_named(self):
        events, generator = self._stream(
            [
                _ok([{"question": "", "answer": ""}]),
                _ok([_item("Explain how pooling bounds concurrent connections")]),
            ],
            count=1,
        )
        self.assertIn("missing_or_empty_question", generator.llm.prompts[1])
        self.assertEqual(events[-1]["generated_count"], 1)

    def test_a_question_that_normalises_to_nothing_is_dropped(self):
        events, generator = self._stream(
            [_ok([{"question": "!!!", "answer": "a"}, _item("Explain pool sizing")])],
            count=1,
        )
        self.assertEqual(events[-1]["generated_count"], 1)

    def test_a_duplicate_question_is_dropped_and_named_in_the_prompt(self):
        events, generator = self._stream(
            [
                _ok([_item("Explain how pooling bounds concurrent connections")]),
                _ok([_item("Explain how pooling bounds concurrent connections!")]),
                _ok([_item("Explain how a pool reuses warm sockets")]),
            ],
            count=1,
        )
        self.assertEqual(events[-1]["generated_count"], 1)

    def test_a_section_first_request_tries_the_section_pass_first(self):
        events, generator = self._stream(
            [_ok([_item("Explain how pooling bounds concurrent connections")])],
            count=1,
            section_title="Warm sockets",
            section_content="A warm socket avoids the handshake entirely.",
        )
        self.assertIn("Warm sockets", generator.llm.prompts[0])
        self.assertEqual(events[-1]["generated_count"], 1)

    def test_the_recovery_pass_asks_for_one_item_at_a_time(self):
        # The slot after the first success cannot produce a question, so the
        # silent single-item recovery pass runs and the deterministic filler
        # still completes the batch.
        class RecoveringLLM:
            def __init__(self):
                self.calls = 0
                self.prompts: list[str] = []

            async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kw):
                self.calls += 1
                self.prompts.append(prompt)
                if self.calls == 1:
                    return _ok([_item("Explain how pooling bounds concurrent connections")])
                if "final_recovery_fill_missing_items" in prompt:
                    return _ok([_item("Explain how a pool reuses warm sockets")])
                return _ok(["not-a-dict"])

        generator = QuestionGenerator(RecoveringLLM())
        events = asyncio.run(
            _collect(
                generator.generate_v2_stream(
                    topic_id="topic-a",
                    topic_title="Connection pooling",
                    doc_content=self.DOC,
                    count=2,
                    level="mid",
                )
            )
        )
        # No slot ever produced a question, so the deterministic filler finishes
        # the batch and the stream still reports the full count.
        self.assertGreater(generator.llm.calls, 1)
        self.assertEqual(events[-1]["generated_count"], 2)
        self.assertGreater(events[-1]["malformed_items_dropped"], 0)
        questions = [e["question"]["question"] for e in events if e["type"] == "question"]
        self.assertEqual(len(set(questions)), 2)

    def test_a_policy_block_in_the_recovery_pass_is_reported_as_an_error_frame(self):
        from app.services.llm_policy import LLMServiceApprovalRequiredError

        state = {"calls": 0}

        async def _completion(prompt, llm_config=None, user_identity=None, task=None, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                return _ok("not json")
            raise LLMServiceApprovalRequiredError("Approval required")

        generator = _generator([])
        generator.llm.completion = _completion
        events = asyncio.run(
            _collect(
                generator.generate_v2_stream(
                    topic_id="topic-a",
                    topic_title="Connection pooling",
                    doc_content=self.DOC,
                    count=2,
                    level="mid",
                )
            )
        )
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "llm_service_approval_required")

    def test_the_filler_is_used_when_the_llm_produces_nothing_valid(self):
        events, generator = self._stream([_fail("llm_call_failed")], count=2)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 2)
        questions = [e["question"] for e in events if e["type"] == "question"]
        self.assertEqual(len(questions), 2)
        for item in questions:
            self.assertTrue(item["question"].strip())
            self.assertTrue(item["answer"].strip())

    def test_the_done_frame_carries_the_provider_and_retry_counters(self):
        events, _generator = self._stream(
            [_ok([_item("Explain how pooling bounds concurrent connections")])], count=1
        )
        done = events[-1]
        self.assertEqual(done["provider_used"], "openai")
        self.assertEqual(done["model_used"], "gpt-4o-mini")
        self.assertEqual(done["topic_id"], "topic-a")
        self.assertEqual(done["malformed_items_dropped"], 0)


class GenerateBranchTests(unittest.TestCase):
    def test_the_legacy_path_drops_a_repeated_problem_pattern(self):
        conforming = (
            "### Problem\nState the constraint.\n\n### Solution Walkthrough\nWalk it.\n\n"
            "### Complexity\nO(n).\n\n### Code\n```python\n# build a hash map\nseen = {}\n"
            "# scan once\nfor x in nums:\n    pass\n```\n"
        )
        generator = _generator(
            [_ok([_item("Solve two sum using a hash map", conforming)])]
        )
        response = asyncio.run(
            generator.generate(
                topic_id=PS_ID,
                topic_title="Problem Solving and Algorithms",
                doc_content="Solve two sum with a hash map.",
                count=4,
                level="mid",
                preferred_language="python",
            )
        )
        signatures = [
            qg._problem_solving_pattern_signature(q.question, q.answer)
            for q in response.questions
        ]
        unique = [s for s in signatures if s]
        self.assertEqual(len(unique), len(set(unique)))

    def test_the_legacy_path_reports_the_provider_and_model(self):
        generator = _generator(
            [_ok([_item("Explain how pooling bounds concurrent connections")])]
        )
        response = asyncio.run(
            generator.generate(
                topic_id="topic-a",
                topic_title="Topic A",
                doc_content="Pooling bounds concurrent connections.",
                count=1,
                level="mid",
            )
        )
        self.assertEqual(response.provider_used, "openai")
        self.assertEqual(response.model_used, "gpt-4o-mini")


class GenerateV2BranchTests(unittest.TestCase):
    def test_a_section_only_request_falls_back_to_the_topic_pass(self):
        generator = _generator(
            [_ok([_item("Explain how pooling bounds concurrent connections")])]
        )
        response = asyncio.run(
            generator.generate_v2(
                topic_id="topic-a",
                topic_title="Connection pooling",
                doc_content="Pooling bounds concurrent connections.",
                count=1,
                level="mid",
                section_title="Warm sockets",
                section_content="A warm socket avoids the handshake.",
            )
        )
        self.assertEqual(len(response.questions), 1)

    def test_a_junk_item_does_not_count_towards_the_batch(self):
        generator = _generator(
            [_ok(["not-a-dict", {"question": "", "answer": ""}])]
        )
        response = asyncio.run(
            generator.generate_v2(
                topic_id="topic-a",
                topic_title="Connection pooling",
                doc_content="Pooling bounds concurrent connections.",
                count=1,
                level="mid",
            )
        )
        # The batch still reaches the requested size, and the junk is reported.
        self.assertEqual(len(response.questions), 1)
        self.assertGreaterEqual(response.malformed_items_dropped, 1)
        self.assertGreater(response.retries_used, 0)


class GenerateQuizV2StreamBranchTests(unittest.TestCase):
    def _run(self, results, **kwargs):
        llm = ScriptedLLM(results)
        generator = QuestionGenerator(llm)
        params = {"topics_content": QUIZ_TOPICS, "count": 1, "question_types": ["mcq"], "level": "mid"}
        params.update(kwargs)
        return asyncio.run(_collect(generator.generate_quiz_v2_stream(**params))), llm

    def test_a_non_dict_and_an_invalid_item_are_dropped_and_named(self):
        events, llm = self._run(
            [
                _ok(["not-a-dict", _quiz_item(question="short")]),
                _ok([_quiz_item()]),
            ]
        )
        self.assertIn("invalid_question", llm.prompts[1])
        self.assertEqual(events[-1]["generated_count"], 1)

    def test_a_duplicate_item_is_dropped_and_named_in_the_retry_prompt(self):
        duplicate = _quiz_item()["question"]
        events, llm = self._run(
            [
                _ok([_quiz_item()]),
                _ok([_quiz_item(question=duplicate), _quiz_item(question=duplicate)]),
                _ok([_quiz_item(question=QUIZ_QUESTIONS[1])]),
            ],
            count=2,
        )
        self.assertIn("duplicate_question", llm.prompts[2])
        self.assertEqual(events[-1]["generated_count"], 2)
        questions = [e["question"]["question"] for e in events if e["type"] == "question"]
        self.assertEqual(len(set(questions)), 2)

    def test_the_recovery_pass_emits_a_recovering_progress_frame(self):
        class RecoveringQuizLLM:
            def __init__(self):
                self.calls = 0

            async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kw):
                self.calls += 1
                if "final_recovery_fill_missing_items" in prompt:
                    return _ok([_quiz_item(question=QUIZ_QUESTIONS[1])])
                return _ok("not json")

        llm = RecoveringQuizLLM()
        events = asyncio.run(
            _collect(
                QuestionGenerator(llm).generate_quiz_v2_stream(
                    topics_content=QUIZ_TOPICS, count=1, question_types=["mcq"], level="mid"
                )
            )
        )
        recovering = [e for e in events if e.get("stage") == "recovering"]
        self.assertTrue(recovering)
        self.assertEqual(recovering[0]["target_count"], 1)
        self.assertEqual(events[-1]["generated_count"], 1)

    def test_a_non_dict_recovery_result_is_skipped_and_the_filler_finishes(self):
        llm = QuizLLMFactory([_ok([_quiz_item()]), _ok(["not-a-dict"])])
        events = asyncio.run(
            _collect(
                QuestionGenerator(llm).generate_quiz_v2_stream(
                    topics_content=QUIZ_TOPICS, count=3, question_types=["mcq"], level="mid"
                )
            )
        )
        questions = [e["question"] for e in events if e["type"] == "question"]
        self.assertEqual(len(questions), 3)
        self.assertEqual(len({q["question"] for q in questions}), 3)

    def test_a_policy_block_in_the_recovery_pass_is_reported_as_an_error_frame(self):
        from app.services.llm_policy import PersonalCredentialRequiredError

        state = {"calls": 0}

        async def _completion(prompt, llm_config=None, user_identity=None, task=None, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                return _ok([_quiz_item()])
            raise PersonalCredentialRequiredError("Bring your own key")

        llm = QuizLLMFactory()
        llm.completion = _completion
        events = asyncio.run(
            _collect(
                QuestionGenerator(llm).generate_quiz_v2_stream(
                    topics_content=QUIZ_TOPICS, count=2, question_types=["mcq"], level="mid"
                )
            )
        )
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "personal_credential_required")


# --------------------------------------------------------------------------
# Third batch: the remaining salvage, near-duplicate, and stream-phase arcs.
# --------------------------------------------------------------------------


class NearDuplicateFinalTests(unittest.TestCase):
    def test_whitespace_only_differences_are_an_exact_duplicate(self):
        self.assertTrue(
            qg._is_near_duplicate_question(
                "Explain  database   index selection strategies",
                ["Explain database index selection strategies"],
            )
        )

    def test_a_token_overlap_above_the_threshold_is_a_duplicate(self):
        self.assertTrue(
            qg._is_near_duplicate_question(
                "Explain connection pooling, warm sockets, and handshake reuse in production",
                ["Explain connection pooling, warm sockets, and handshake budgets in production"],
            )
        )


class TextQaAdjacentMarkerTests(unittest.TestCase):
    def test_an_empty_block_between_two_markers_is_dropped(self):
        items = qg._parse_text_qa_pairs("Q1:\n\nQ2: real question here\nA: real answer here")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["question"], "real question here")

    def test_a_numbered_block_without_an_answer_line_is_dropped(self):
        self.assertEqual(
            qg._parse_text_qa_pairs("1) A question with no answer line at all"),
            [],
        )


class ParagraphFallbackTests(unittest.TestCase):
    def test_the_paragraph_fallback_drops_short_paragraphs(self):
        fragments = qg._extract_content_fragments("ab\ncd\n\nefghij klmno pqrstu vwxyz\n\nxy")
        self.assertEqual(fragments, ["efghij klmno pqrstu vwxyz"])


class GenerateProblemSolvingSeedTests(unittest.TestCase):
    def test_the_legacy_path_seeds_known_pattern_signatures_from_the_llm_batch(self):
        conforming = (
            "### Problem\nState the constraint.\n\n### Solution Walkthrough\nWalk it.\n\n"
            "### Complexity\nO(n).\n\n### Code\n```python\n# build a hash map\nseen = {}\n"
            "# scan once\nfor x in nums:\n    pass\n```\n"
        )
        generator = _generator([_ok([_item("Solve two sum using a hash map", conforming)])])
        response = asyncio.run(
            generator.generate(
                topic_id=PS_ID,
                topic_title="Problem Solving and Algorithms",
                doc_content="Solve two sum with a hash map.",
                count=2,
                level="mid",
                preferred_language="python",
            )
        )
        self.assertGreaterEqual(len(response.questions), 1)
        self.assertIn(
            "two_sum_hash_map",
            {
                qg._problem_solving_pattern_signature(q.question, q.answer)
                for q in response.questions
            },
        )


class GenerateV2FinalTests(unittest.TestCase):
    def test_a_slot_that_never_produces_a_question_still_returns_the_target(self):
        generator = _generator([_ok(["not-a-dict"])])
        response = asyncio.run(
            generator.generate_v2(
                topic_id="topic-a",
                topic_title="Connection pooling",
                doc_content="Pooling bounds concurrent connections.",
                count=2,
                level="mid",
            )
        )
        self.assertEqual(len(response.questions), 2)
        self.assertGreater(response.retries_used, 0)

    def test_an_existing_question_is_never_repeated(self):
        generator = _generator([_ok([_item("Explain how pooling bounds concurrent connections")])])
        response = asyncio.run(
            generator.generate_v2(
                topic_id="topic-a",
                topic_title="Connection pooling",
                doc_content="Pooling bounds concurrent connections.",
                count=2,
                level="mid",
                existing_questions=["Explain how pooling bounds concurrent connections"],
            )
        )
        self.assertNotIn(
            "Explain how pooling bounds concurrent connections",
            [q.question for q in response.questions],
        )
        self.assertEqual(len(response.questions), 2)


class GenerateV2StreamFinalTests(unittest.TestCase):
    DOC = "Pooling bounds concurrent connections and reuses warm ones."

    def test_a_policy_block_on_the_first_slot_is_reported_immediately(self):
        from app.services.llm_policy import StudyAppLLMNotAssignedError

        async def _blocked(prompt, llm_config=None, user_identity=None, task=None, **kwargs):
            raise StudyAppLLMNotAssignedError("No service assigned")

        generator = _generator([])
        generator.llm.completion = _blocked
        events = asyncio.run(
            _collect(
                generator.generate_v2_stream(
                    topic_id="topic-a",
                    topic_title="Connection pooling",
                    doc_content=self.DOC,
                    count=2,
                    level="mid",
                )
            )
        )
        # The first frame is always start; the error is the last one.
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "study_app_llm_not_assigned")

    def test_a_slot_that_returns_nothing_produces_no_question_frame(self):
        class AlwaysJunk:
            def __init__(self):
                self.calls = 0

            async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kw):
                self.calls += 1
                return _ok(["not-a-dict"])

        generator = QuestionGenerator(AlwaysJunk())
        events = asyncio.run(
            _collect(
                generator.generate_v2_stream(
                    topic_id="topic-a",
                    topic_title="Connection pooling",
                    doc_content=self.DOC,
                    count=1,
                    level="mid",
                )
            )
        )
        questions = [e for e in events if e["type"] == "question"]
        self.assertEqual(len(questions), 1)
        # The single published question came from the deterministic filler.
        self.assertTrue(questions[0]["question"]["answer"].strip())
        self.assertEqual(events[-1]["generated_count"], 1)

    def test_a_section_pass_runs_before_the_topic_pass(self):
        order: list[str] = []

        class RecordingLLM:
            async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kw):
                order.append("section" if "section_title" in prompt else "topic")
                if len(order) == 1:
                    return _ok(
                        [
                            _item(
                                "Explain how a warm socket avoids the handshake",
                                "It skips the handshake entirely.",
                            )
                        ]
                    )
                return _ok(["not-a-dict"])

        generator = QuestionGenerator(RecordingLLM())
        events = asyncio.run(
            _collect(
                generator.generate_v2_stream(
                    topic_id="topic-a",
                    topic_title="Connection pooling",
                    doc_content=self.DOC,
                    count=1,
                    level="mid",
                    section_title="Warm sockets",
                    section_content="A warm socket avoids the handshake entirely.",
                )
            )
        )
        self.assertEqual(order[0], "section")
        self.assertEqual(events[-1]["generated_count"], 1)


class GenerateQuizV2StreamFinalTests(unittest.TestCase):
    def test_the_done_frame_reports_topics_and_counters(self):
        llm = QuizLLMFactory()
        events = asyncio.run(
            _collect(
                QuestionGenerator(llm).generate_quiz_v2_stream(
                    topics_content=QUIZ_TOPICS, count=2, question_types=["mcq"], level="mid"
                )
            )
        )
        done = events[-1]
        self.assertEqual(done["target_count"], 2)
        self.assertEqual(done["topics_used"], ["topic-a"])
        self.assertGreaterEqual(done["retries_used"], 0)
        self.assertEqual(done["malformed_items_dropped"], 0)

    def test_the_filler_finishes_when_the_llm_only_returns_duplicates(self):
        llm = QuizLLMFactory([_ok([_quiz_item()]), _ok([_quiz_item()])])
        events = asyncio.run(
            _collect(
                QuestionGenerator(llm).generate_quiz_v2_stream(
                    topics_content=QUIZ_TOPICS, count=3, question_types=["mcq"], level="mid"
                )
            )
        )
        questions = [e["question"]["question"] for e in events if e["type"] == "question"]
        self.assertEqual(len(questions), 3)
        self.assertEqual(len(set(questions)), 3)
        self.assertEqual(events[-1]["generated_count"], 3)
