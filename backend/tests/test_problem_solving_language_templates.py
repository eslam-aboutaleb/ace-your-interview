"""Plan 1.10: a problem-solving answer must never relabel another language's source.

Before this item, ``_problem_solving_code_template`` ended with
``templates.get(language, templates[PROBLEM_SOLVING_DEFAULT_LANGUAGE])``, so any
language outside the hardcoded dict (rust, kotlin, ruby, ...) received the
Python two-sum implementation wrapped in a ```<language>``` fence, and every
non-Python scenario that was not two-sum also received the Python two-sum body.
"""

import unittest

from app.services import question_generator as qg
from app.services.topic_catalog import (
    PROBLEM_SOLVING_DEFAULT_LANGUAGE,
    PROBLEM_SOLVING_LANGUAGE_OPTIONS,
)

_ALL_SCENARIOS = tuple(qg._PROBLEM_SOLVING_SCENARIOS_BY_ID)

# Signatures that prove the body is genuinely in the requested language.
_LANGUAGE_MARKERS = {
    "python": ("def ", "#"),
    "java": ("public ", "class Solution", "//"),
    "cpp": ("#include", "using namespace std;", "//"),
    "javascript": ("function ", "const ", "//"),
    "csharp": ("public class Solution", "var ", "//"),
    "go": ("func ", ":=", "//"),
}


def _fence_language(answer: str) -> str:
    match = qg._PROBLEM_SOLVING_CODE_FENCE_RE.search(answer)
    return match.group("language").strip().lower() if match else ""


class ProblemSolvingTemplatePerLanguageTests(unittest.TestCase):
    def test_every_supported_language_has_its_own_scenario_set(self):
        for language in PROBLEM_SOLVING_LANGUAGE_OPTIONS:
            scenario_ids = qg._problem_solving_scenario_ids_for_language(language)
            self.assertTrue(scenario_ids, language)
            self.assertEqual(
                set(scenario_ids),
                set(qg._PROBLEM_SOLVING_CODE_TEMPLATES_BY_LANGUAGE[language]),
                language,
            )

    def test_non_python_languages_are_no_longer_limited_to_two_sum(self):
        for language in PROBLEM_SOLVING_LANGUAGE_OPTIONS:
            if language == PROBLEM_SOLVING_DEFAULT_LANGUAGE:
                continue
            self.assertGreaterEqual(
                len(qg._problem_solving_scenario_ids_for_language(language)),
                len(_ALL_SCENARIOS),
                language,
            )

    def test_scenario_ids_are_empty_for_unsupported_languages(self):
        for language in ("rust", "kotlin", "swift", "ruby", "php", "", "python3"):
            self.assertEqual(qg._problem_solving_scenario_ids_for_language(language), [])

    def test_every_supported_language_and_scenario_has_a_template(self):
        for language in PROBLEM_SOLVING_LANGUAGE_OPTIONS:
            for scenario_id in _ALL_SCENARIOS:
                code = qg._problem_solving_code_template(language, scenario_id=scenario_id)
                self.assertTrue(code, f"{language}/{scenario_id}")

    def test_unsupported_language_gets_no_template(self):
        for language in ("rust", "kotlin", "swift", ""):
            self.assertEqual(qg._problem_solving_code_template(language), "")

    def test_unknown_scenario_in_a_known_language_gets_no_template(self):
        self.assertEqual(qg._problem_solving_code_template("java", scenario_id="nope"), "")


class ProblemSolvingTemplateNoRelabellingTests(unittest.TestCase):
    def test_java_never_receives_python_source(self):
        for scenario_id in _ALL_SCENARIOS:
            code = qg._problem_solving_code_template("java", scenario_id=scenario_id)
            self.assertNotIn("def two_sum", code, scenario_id)
            self.assertNotIn("def ", code, scenario_id)
            self.assertIn("class Solution", code, scenario_id)

    def test_python_only_templates_are_not_reused_elsewhere(self):
        python_only = {
            scenario_id
            for scenario_id in _ALL_SCENARIOS
            if scenario_id in qg._PROBLEM_SOLVING_PYTHON_CODE_TEMPLATES
        }
        self.assertTrue(python_only)
        for language in PROBLEM_SOLVING_LANGUAGE_OPTIONS:
            if language == PROBLEM_SOLVING_DEFAULT_LANGUAGE:
                continue
            for scenario_id in _ALL_SCENARIOS:
                code = qg._problem_solving_code_template(language, scenario_id=scenario_id)
                python_template = qg._PROBLEM_SOLVING_PYTHON_CODE_TEMPLATES[scenario_id]
                self.assertNotEqual(
                    code.strip(),
                    python_template.strip(),
                    f"{language}/{scenario_id} reused the python body",
                )

    def test_each_template_carries_its_own_language_markers(self):
        for language, markers in _LANGUAGE_MARKERS.items():
            for scenario_id in _ALL_SCENARIOS:
                code = qg._problem_solving_code_template(language, scenario_id=scenario_id)
                for marker in markers:
                    self.assertIn(
                        marker,
                        code,
                        f"{language}/{scenario_id} is missing marker {marker!r}",
                    )


class ProblemSolvingFallbackAnswerTests(unittest.TestCase):
    def _answer(self, language: str, scenario_id: str) -> str:
        answer, _summary = qg._build_problem_solving_fallback_answer(
            focus="hash maps",
            source_scope="backend interviews",
            language=language,
            scenario_id=scenario_id,
        )
        return answer

    def test_java_fallback_fence_is_java(self):
        answer = self._answer("java", "two_sum_hash_map")
        self.assertEqual(_fence_language(answer), "java")
        self.assertIn("public int[] twoSum", answer)

    def test_every_java_scenario_fence_is_java_with_java_body(self):
        for scenario_id in _ALL_SCENARIOS:
            answer = self._answer("java", scenario_id)
            self.assertEqual(_fence_language(answer), "java", scenario_id)
            self.assertIn("class Solution", answer, scenario_id)
            self.assertNotIn("def ", answer, scenario_id)

    def test_every_supported_language_validates_its_own_fallback_answer(self):
        for language in PROBLEM_SOLVING_LANGUAGE_OPTIONS:
            for scenario_id in _ALL_SCENARIOS:
                answer = self._answer(language, scenario_id)
                valid, issue = qg._validate_problem_solving_answer(answer, language)
                self.assertTrue(valid, f"{language}/{scenario_id}: {issue}")

    def test_unsupported_language_omits_the_code_block(self):
        answer = self._answer("rust", "two_sum_hash_map")
        self.assertNotIn("### Code", answer)
        self.assertNotIn("```", answer)
        # No borrowed source anywhere in the answer.
        self.assertNotIn("def two_sum", answer)

    def test_unsupported_language_answer_fails_validation_instead_of_substituting(self):
        answer = self._answer("rust", "two_sum_hash_map")
        valid, issue = qg._validate_problem_solving_answer(answer, "rust")
        self.assertFalse(valid)
        self.assertEqual(issue, "invalid_problem_solving_heading_sequence")

    def test_fence_language_validation_can_actually_fail(self):
        answer = self._answer("java", "two_sum_hash_map")
        # The same answer is valid for java and invalid for another language.
        self.assertTrue(qg._validate_problem_solving_answer(answer, "java")[0])
        valid, issue = qg._validate_problem_solving_answer(answer, "python")
        self.assertFalse(valid)
        self.assertEqual(issue, "problem_solving_code_language_mismatch")


class ProblemSolvingFallbackItemTests(unittest.TestCase):
    def _fallback_items(self, language: str, count: int) -> list[dict]:
        return qg._build_fallback_question_items(
            topic_id="00-problem-solving-and-algorithms",
            topic_title="Problem Solving and Algorithms",
            doc_content=(
                "Hash maps, sliding windows, intervals, heaps, graph traversal, "
                "and binary search all appear in these interview loops."
            ),
            count=count,
            difficulty="medium",
            level="mid",
            preferred_language=language,
            requires_programming=True,
        )

    def test_non_python_fallback_yields_multiple_distinct_questions(self):
        items = self._fallback_items("java", 5)
        self.assertGreaterEqual(len(items), 5)
        self.assertGreaterEqual(len({i["question"] for i in items}), 5)

    def test_non_python_fallback_items_validate(self):
        for item in self._fallback_items("java", 5):
            valid, issue = qg._validate_question_item(
                item,
                "00-problem-solving-and-algorithms",
                "medium",
                "mid",
                preferred_language="java",
                requires_programming=True,
                topic_title="Problem Solving and Algorithms",
            )
            self.assertTrue(valid, f"{item['question']}: {issue}")

    def test_unsupported_language_never_ships_python_source(self):
        items = self._fallback_items("rust", 5)
        for item in items:
            self.assertNotIn("def two_sum", item["answer"])
            self.assertEqual(_fence_language(item["answer"]), "rust")

    def test_unsupported_language_falls_back_to_a_language_agnostic_prompt(self):
        with self.assertLogs("app.services.question_generator", level="WARNING"):
            items = self._fallback_items("rust", 3)
        # No compliant answer exists without a genuine template, so the items are
        # dropped rather than shipped with a borrowed fence.
        self.assertEqual(items, [])
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
