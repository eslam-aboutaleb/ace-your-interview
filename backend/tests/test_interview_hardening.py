"""Deterministic interview branching, system-design, code grading and persona.

Everything here is offline: no provider is contacted. These tests lock the
hard parts of the interview loop that used to be soft.

* the weakest rubric dimension is selected deterministically and rendered into
  the next-question prompt as an explicit structured field,
* ``system_design`` is a real session type with its own contract, rubric
  dimensions, and deterministic mermaid-diagram scoring,
* a coding answer with a ``code_contract`` is graded deterministically —
  candidate code is never executed,
* the interviewer persona is threaded into the prompts.
"""

import unittest

from app.services.interview_generator import (
    InterviewGenerator,
    SYSTEM_DESIGN_DIMENSIONS,
    advance_progression_phase,
    analyse_mermaid_diagram,
    extract_code_block,
    grade_coding_answer,
    normalise_code_contract,
    select_follow_up_focus,
)


def _session(**overrides):
    base = {
        "session_id": "s-1",
        "track": "backend",
        "level": "mid",
        "interview_type": "mixed",
        "turn_count": 5,
        "turns_completed": 1,
        "asked_questions": [],
        "memory_summary": "",
        "target_role": "Backend Engineer",
        "focus_areas": [],
    }
    base.update(overrides)
    return base


def _turn(turn_index, **scores):
    return {
        "turn_index": turn_index,
        "question": "Q",
        "user_answer": "A",
        "rubric": {**scores},
    }


class FollowUpFocusSelectionTests(unittest.TestCase):
    def test_no_scored_turn_opens_the_interview(self):
        focus = select_follow_up_focus(_session(), [])
        self.assertEqual(focus["mode"], "opening")
        self.assertEqual(focus["dimensions"], [])
        prompt = InterviewGenerator._question_prompt(_session(), [])
        self.assertIn("focus_mode: opening", prompt)

    def test_the_weakest_dimension_is_chosen_deterministically(self):
        turns = [_turn(2, technical_accuracy=5, reasoning_depth=1, communication_clarity=4,
                       completeness=3, confidence_signal=4)]
        focus = select_follow_up_focus(_session(), turns)
        self.assertEqual(focus["mode"], "remediate")
        self.assertEqual(focus["dimensions"], ["reasoning_depth"])
        self.assertEqual(focus["labels"], ["reasoning depth"])
        self.assertEqual(focus["threshold"], 3)
        self.assertIn("reasoning_depth=1", focus["evidence"])

    def test_tied_weakest_dimensions_are_both_targeted(self):
        turns = [_turn(1, technical_accuracy=2, reasoning_depth=4, communication_clarity=2,
                       completeness=4, confidence_signal=4)]
        focus = select_follow_up_focus(_session(), turns)
        self.assertEqual(
            focus["dimensions"],
            ["technical_accuracy", "communication_clarity"],
        )

    def test_more_than_two_tied_dimensions_are_capped(self):
        turns = [_turn(1, technical_accuracy=1, reasoning_depth=1, communication_clarity=1,
                       completeness=4, confidence_signal=4)]
        focus = select_follow_up_focus(_session(), turns)
        self.assertEqual(len(focus["dimensions"]), 2)
        self.assertEqual(
            focus["dimensions"],
            ["technical_accuracy", "reasoning_depth"],
        )

    def test_a_cleared_rubric_advances_the_progression_phase(self):
        turns = [_turn(3, technical_accuracy=4, reasoning_depth=4, communication_clarity=4,
                       completeness=4, confidence_signal=4)]
        focus = select_follow_up_focus(_session(), turns)
        self.assertEqual(focus["mode"], "advance")
        # Every dimension is tied at the minimum, so the stable dimension order
        # decides; the point is that the next question pushes the phase.
        self.assertEqual(focus["dimensions"][0], "technical_accuracy")
        # Turn position alone would keep this an opening turn; the cleared
        # rubric advances the ladder one step.
        self.assertEqual(focus["base_phase"], "opening")
        self.assertEqual(focus["phase"], "middle")
        self.assertTrue(focus["phase_advanced"])
        prompt = InterviewGenerator._question_prompt(_session(), turns)
        self.assertIn("focus_mode: advance", prompt)
        # The session config now reports the advanced phase, not the positional
        # one, and the block names the phase it moved on from.
        self.assertIn("- progression_phase: middle\n", prompt)
        self.assertIn("phase_advanced_from: opening", prompt)
        self.assertIn("the progression phase advances to middle", prompt)
        # The phase-specific rules follow the advance too.
        self.assertIn("This is the middle phase of the interview.", prompt)

    def test_a_below_threshold_rubric_does_not_advance_the_phase(self):
        turns = [_turn(1, technical_accuracy=4, reasoning_depth=1, communication_clarity=4,
                       completeness=4, confidence_signal=4)]
        focus = select_follow_up_focus(_session(focus_threshold=3), turns)
        self.assertEqual(focus["mode"], "remediate")
        self.assertEqual(focus["base_phase"], "opening")
        self.assertEqual(focus["phase"], "opening")
        self.assertFalse(focus["phase_advanced"])
        prompt = InterviewGenerator._question_prompt(_session(), turns)
        self.assertIn("This is the opening phase of the interview.", prompt)

    def test_the_phase_advance_is_capped_at_the_last_phase(self):
        self.assertEqual(advance_progression_phase("opening"), "middle")
        self.assertEqual(advance_progression_phase("middle"), "late")
        # A run of cleared rubrics keeps the hardest phase rather than falling
        # off the end of the ladder.
        self.assertEqual(advance_progression_phase("late"), "late")
        # An unknown phase is returned untouched, never silently reopened.
        self.assertEqual(advance_progression_phase("warmup"), "warmup")

    def test_a_resumed_session_recomputes_the_same_phase(self):
        # Resume replays the stored turns, so the branch and the phase are a
        # pure function of what has already happened.
        turns = [_turn(2, technical_accuracy=5, reasoning_depth=5, communication_clarity=5,
                       completeness=5, confidence_signal=5)]
        first = select_follow_up_focus(_session(), turns)
        again = select_follow_up_focus(dict(_session()), list(turns))
        self.assertEqual(first, again)
        self.assertEqual(first["phase"], "middle")

    def test_the_focus_phase_travels_with_the_stored_branch(self):
        turns = [_turn(1, technical_accuracy=4, reasoning_depth=4, communication_clarity=4,
                       completeness=4, confidence_signal=4)]
        focus = select_follow_up_focus(_session(), turns)
        rule = InterviewGenerator._follow_up_focus_rules(focus)
        self.assertEqual(len(rule), 1)
        self.assertIn("progression phase has advanced to middle", rule[0])
        # A focus recorded before the phase fields existed still renders a
        # phase instead of an empty slot.
        rule = InterviewGenerator._follow_up_focus_rules(
            {k: v for k, v in focus.items() if not k.startswith("phase")}
        )
        self.assertIn("advanced to middle", rule[0])

    def test_a_degraded_turn_never_drives_the_branch(self):
        turns = [
            _turn(1, technical_accuracy=2, reasoning_depth=2, communication_clarity=2,
                  completeness=2, confidence_signal=2),
            {"turn_index": 2, "degraded": True, "rubric": {}},
        ]
        focus = select_follow_up_focus(_session(), turns)
        self.assertEqual(focus["mode"], "remediate")
        self.assertEqual(focus["turn_index"], 1)

    def test_the_focus_threshold_is_read_from_the_session(self):
        turns = [_turn(1, technical_accuracy=3, reasoning_depth=3, communication_clarity=3,
                       completeness=3, confidence_signal=3)]
        self.assertEqual(
            select_follow_up_focus(_session(), turns)["mode"], "advance"
        )
        focus = select_follow_up_focus(_session(focus_threshold=4), turns)
        self.assertEqual(focus["mode"], "remediate")
        self.assertEqual(focus["threshold"], 4)

    def test_the_prompt_carries_the_selection_as_a_structured_field(self):
        turns = [_turn(1, technical_accuracy=5, reasoning_depth=1, communication_clarity=4,
                       completeness=3, confidence_signal=4)]
        prompt = InterviewGenerator._question_prompt(_session(), turns)
        self.assertIn("Follow-up focus (deterministic", prompt)
        self.assertIn("focus_mode: remediate", prompt)
        self.assertIn("focus_dimensions: reasoning_depth", prompt)
        self.assertIn("focus_labels: reasoning depth", prompt)
        self.assertIn("focus_threshold: 3", prompt)
        self.assertIn("evidence: turn 1 scores:", prompt)
        self.assertIn("must target reasoning depth", prompt)
        # The old soft sentence is gone.
        self.assertNotIn("focus on the weakest dimension", prompt)

    def test_the_contract_binds_the_question_to_the_focus(self):
        turns = [_turn(1, technical_accuracy=5, reasoning_depth=1, communication_clarity=4,
                       completeness=3, confidence_signal=4)]
        rule = InterviewGenerator._follow_up_focus_rules(
            select_follow_up_focus(_session(), turns)
        )
        self.assertEqual(len(rule), 1)
        self.assertIn("reasoning depth", rule[0])
        self.assertIn("expected_signals", rule[0])


class SystemDesignFlowTests(unittest.TestCase):
    def test_only_the_explicit_session_type_is_a_system_design_session(self):
        self.assertTrue(
            InterviewGenerator._is_system_design_session(_session(interview_type="system_design"))
        )
        self.assertFalse(InterviewGenerator._is_system_design_session(_session(interview_type="mixed")))
        # A system-design *track* without the session type keeps its own flow.
        self.assertFalse(
            InterviewGenerator._is_system_design_session(
                _session(interview_type="mixed", track="system-design")
            )
        )

    def test_the_question_prompt_has_its_own_contract(self):
        prompt = InterviewGenerator._question_prompt(
            _session(interview_type="system_design"), []
        )
        self.assertIn("system-design interviewer", prompt)
        self.assertIn("mermaid diagram", prompt)
        self.assertIn("architecture, tradeoffs, scalability, failure modes", prompt)
        # System design is a real skill pack, not the general one.
        self.assertNotIn("senior interviewer running a personalized mock", prompt)

    def test_the_question_word_limit_is_wider_for_designs(self):
        long_question = " ".join(["design"] * 60)
        ok, issue = InterviewGenerator._validate_question_payload(
            {"question": long_question, "competency_focus": "quota",
             "expected_signals": ["a", "b"]},
            [],
            max_words=70,
        )
        self.assertTrue(ok, issue)

    def test_the_evaluator_prompt_scores_the_design_dimensions(self):
        prompt = InterviewGenerator._evaluate_prompt(
            _session(interview_type="system_design"),
            "Design a rate limiter.",
            "I would use a token bucket.",
            1,
        )
        for dimension in SYSTEM_DESIGN_DIMENSIONS:
            self.assertIn(f'"{dimension}"', prompt)
        self.assertIn("system_design", prompt)

    def test_the_design_rubric_is_normalised_and_clamped(self):
        generator = InterviewGenerator.__new__(InterviewGenerator)
        scores = generator._normalise_design_scores(
            {"architecture": 9, "tradeoffs": -2, "scalability": 3,
             "failure_modes": None, "communication": "4"}
        )
        self.assertEqual(
            scores,
            {"architecture": 5, "tradeoffs": 0, "scalability": 3,
             "failure_modes": None, "communication": 4},
        )

    def test_an_empty_design_rubric_is_dropped(self):
        generator = InterviewGenerator.__new__(InterviewGenerator)
        self.assertEqual(
            generator._normalise_design_scores({"architecture": None}), {}
        )
        self.assertEqual(generator._normalise_design_scores("nope"), {})

    def test_a_non_design_session_gets_no_design_rubric(self):
        generator = InterviewGenerator.__new__(InterviewGenerator)
        self.assertEqual(
            generator._normalise_design_scores({"architecture": 4}, session=_session()),
            {},
        )

    def test_a_structured_diagram_scores_its_structure(self):
        analysis = analyse_mermaid_diagram(
            "```mermaid\n"
            "graph TD\n"
            "  client[Client] --> gateway[Gateway]\n"
            "  gateway --> service[Order Service]\n"
            "  service --> db[(Postgres)]\n"
            "```\n"
            "I would pick the queue instead of a direct write."
        )
        self.assertTrue(analysis["present"])
        self.assertEqual(analysis["nodes"], 4)
        self.assertEqual(analysis["edges"], 3)
        self.assertEqual(analysis["labelled_components"], 4)
        self.assertGreaterEqual(analysis["tradeoff_mentions"], 1)
        self.assertEqual(analysis["structure_score"], 4)
        self.assertIn("components_labelled", analysis["signals"])

    def test_an_anonymous_diagram_has_no_structure(self):
        analysis = analyse_mermaid_diagram("```mermaid\ngraph TD\n  a --> b\n```")
        self.assertTrue(analysis["present"])
        self.assertIn("diagram_without_structure", analysis["signals"])
        self.assertEqual(analysis["structure_score"], 0)

    def test_no_diagram_is_reported_not_crashed(self):
        analysis = analyse_mermaid_diagram("No diagram here.")
        self.assertFalse(analysis["present"])
        self.assertIn("diagram_missing", analysis["signals"])

    def test_a_structured_diagram_floors_architecture(self):
        analysis = {"present": True, "nodes": 6, "edges": 6,
                    "labelled_components": 6, "tradeoff_mentions": 1}
        design, applied = _apply_diagram(analysis, {"architecture": 1, "communication": 4})
        self.assertEqual(design["architecture"], 3)
        self.assertEqual(design["communication"], 4)
        self.assertIn("architecture_floored_at_3:diagram_structure_present", applied)

    def test_a_diagram_without_structure_caps_architecture_and_communication(self):
        analysis = {"present": True, "nodes": 3, "edges": 0,
                    "labelled_components": 0, "tradeoff_mentions": 0}
        design, applied = _apply_diagram(analysis, {"architecture": 5, "communication": 5})
        self.assertEqual(design["architecture"], 2)
        self.assertEqual(design["communication"], 2)
        self.assertEqual(len(applied), 2)

    def test_reports_keep_the_design_dimension_averages(self):
        class _Topics:
            def __iter__(self):
                return iter([])

        class _Parser:
            def list_topics(self, track=None):
                return []

        generator = InterviewGenerator.__new__(InterviewGenerator)
        generator.parser = _Parser()
        session = _session(interview_type="system_design")
        turns = [
            {
                "rubric": {"technical_accuracy": 3, "reasoning_depth": 3,
                           "communication_clarity": 3, "completeness": 3,
                           "confidence_signal": 3, "overall": 60},
                "system_design_scores": {"architecture": 2, "tradeoffs": 4,
                                         "scalability": 3, "failure_modes": 1,
                                         "communication": 4},
                "strengths": [],
                "improvements": [],
            }
        ]
        report = generator.build_report(session=session, turns=turns)
        self.assertEqual(
            report["design_dimension_averages"],
            {"architecture": 2.0, "tradeoffs": 4.0, "scalability": 3.0,
             "failure_modes": 1.0, "communication": 4.0},
        )
        self.assertTrue(any("weakest system-design" in s for s in report["next_steps"]))


class CodeGradingTests(unittest.TestCase):
    contract = {
        "code_contract": {
            "function_name": "two_sum",
            "language": "python",
            "test_cases": [
                {"input": "[2,7]", "expected_output": "[0, 1]", "description": "first pair"},
                {"input": "[3,2,4]", "expected_output": "[1, 2]", "description": "offset pair"},
            ],
        }
    }

    def test_the_fenced_block_is_extracted(self):
        language, code = extract_code_block(
            "Some prose.\n\n```python\ndef f():\n    return 1\n```\n"
        )
        self.assertEqual(language, "python")
        self.assertIn("def f()", code)

    def test_a_malformed_contract_is_ignored(self):
        self.assertEqual(normalise_code_contract({"code_contract": "nope"}), {})
        self.assertEqual(
            normalise_code_contract(
                {"code_contract": {"test_cases": [{"expected_output": ""}]}}
            ),
            {},
        )

    def test_no_code_means_no_credit(self):
        grade = grade_coding_answer("I would just use a hash map.", self.contract)
        self.assertTrue(grade["available"])
        self.assertFalse(grade["code_present"])
        self.assertFalse(grade["passed"])
        self.assertEqual(grade["reason"], "no_code_block")
        self.assertFalse(grade["executed"])

    def test_a_named_entry_point_must_be_declared(self):
        grade = grade_coding_answer("```python\nlookup(pair):\n    return pair\n```", self.contract)
        self.assertIs(grade["entry_point_declared"], False)
        self.assertFalse(grade["passed"])
        self.assertEqual(grade["reason"], "entry_point_not_declared")

    def test_matching_every_expected_output_passes(self):
        grade = grade_coding_answer(
            "```python\ndef two_sum(nums, target):\n    # returns [0, 1] then [1, 2]\n    return [0, 1]\n```",
            self.contract,
        )
        self.assertTrue(grade["code_present"])
        self.assertTrue(grade["entry_point_declared"])
        self.assertEqual(grade["matched_outputs"], 2)
        self.assertTrue(grade["passed"])
        self.assertEqual(grade["reason"], "all_test_cases_matched")

    def test_a_missing_expected_output_fails_the_grade(self):
        grade = grade_coding_answer(
            "```python\ndef two_sum(nums, target):\n    return None\n```",
            self.contract,
        )
        self.assertFalse(grade["passed"])
        self.assertEqual(grade["reason"], "expected_outputs_missing")

    def test_no_contract_leaves_the_rubric_alone(self):
        rubric = _apply_code({"technical_accuracy": 1}, None)
        self.assertEqual(rubric["technical_accuracy"], 1)


def _apply_diagram(analysis, design):
    from app.services.interview_generator import _apply_diagram_evidence

    return _apply_diagram_evidence(design, analysis)


def _apply_code(rubric, grade):
    from app.services.interview_generator import _apply_code_grade_evidence

    out, applied = _apply_code_grade_evidence(rubric, grade)
    return {**out, "_applied": applied} if applied else out


class CodeGradeEvidenceTests(unittest.TestCase):
    def _grade(self, answer):
        return grade_coding_answer(answer, CodeGradingTests.contract)

    def test_missing_code_caps_technical_accuracy(self):
        out = _apply_code({"technical_accuracy": 5}, self._grade("just prose"))
        self.assertEqual(out["technical_accuracy"], 1)
        self.assertIn("technical_accuracy_capped_at_1", out["_applied"][0])

    def test_an_undeclared_entry_point_caps_technical_accuracy(self):
        out = _apply_code(
            {"technical_accuracy": 5},
            self._grade("```python\nlookup():\n    return 1\n```"),
        )
        self.assertEqual(out["technical_accuracy"], 2)
        self.assertIn("capped_at_2", out["_applied"][0])

    def test_every_expected_output_present_raises_the_floor(self):
        out = _apply_code(
            {"technical_accuracy": 1},
            self._grade(
                "```python\ndef two_sum(n, t):\n    return [0, 1]  # [1, 2]\n```"
            ),
        )
        self.assertEqual(out["technical_accuracy"], 3)
        self.assertIn("floored_at_3", out["_applied"][0])

    def test_a_high_score_is_not_lowered_by_a_pass(self):
        out = _apply_code(
            {"technical_accuracy": 5},
            self._grade(
                "```python\ndef two_sum(n, t):\n    return [0, 1]  # [1, 2]\n```"
            ),
        )
        self.assertEqual(out["technical_accuracy"], 5)


class DeterministicEvidenceEndToEndTests(unittest.IsolatedAsyncioTestCase):
    """The deterministic evidence reaches the rubric through the real call."""

    class _StubLLM:
        def __init__(self, eval_payload: str):
            self._eval_payload = eval_payload
            self.eval_prompts: list[str] = []

        async def completion(
            self, prompt, llm_config=None, user_identity=None, **kwargs
        ):
            if "interview evaluator" in prompt:
                self.eval_prompts.append(prompt)
            return {
                "success": True,
                "analysis": self._eval_payload,
                "metadata": {},
                "error_code": "",
                "finish_reason": "stop",
            }

    class _StubParser:
        def list_topics(self, track=None):
            return []

    async def test_a_returned_code_contract_is_persisted_in_the_contract(self):
        from app.services.interview_generator import InterviewGenerator

        class _QuestionLLM:
            async def completion(self, prompt, llm_config=None, **kwargs):
                return {
                    "success": True,
                    "analysis": (
                        '{"question":"Return the two indices that add up to the '
                        'target.","competency_focus":"hash map",'
                        '"expected_signals":["names the data structure",'
                        '"states complexity"],'
                        '"code_contract":{"function_name":"two_sum",'
                        '"language":"python","test_cases":'
                        '[{"input":"[2,7]","expected_output":"[0, 1]"},'
                        '{"input":"[3,2]","expected_output":""}]}}'
                    ),
                    "metadata": {},
                    "error_code": "",
                    "finish_reason": "stop",
                }

        generator = InterviewGenerator(_QuestionLLM(), self._StubParser())
        generated = await generator.generate_question(
            session=_session(interview_type="coding"), turns=[]
        )
        # The contract keeps the deterministic grading contract so the answer
        # can be graded against the test cases, and drops the empty one.
        code_contract = generated["code_contract"]
        self.assertEqual(code_contract["function_name"], "two_sum")
        self.assertEqual(code_contract["language"], "python")
        self.assertEqual(len(code_contract["test_cases"]), 1)
        self.assertEqual(
            code_contract["test_cases"][0]["expected_output"], "[0, 1]"
        )
        # The branch the question was generated for travels with it.
        self.assertEqual(generated["follow_up_focus"]["mode"], "opening")

    def _generator(self, eval_payload: str):
        from app.services.interview_generator import InterviewGenerator

        llm = self._StubLLM(eval_payload)
        return InterviewGenerator(llm, self._StubParser()), llm

    async def test_a_code_answer_without_code_cannot_score_high(self):
        generator, llm = self._generator(
            '{"rubric":{"technical_accuracy":5,"reasoning_depth":4,'
            '"communication_clarity":4,"completeness":3,"confidence_signal":3,'
            '"overall":90},"strengths":["Clear"],"improvements":["Add code"],'
            '"follow_up_note":"Write the code next time."}'
        )
        result = await generator.evaluate_answer(
            session=_session(interview_type="coding"),
            question="Return the two indices that add up to the target.",
            user_answer="I would use a hash map, obviously.",
            turn_index=0,
            question_contract={
                "competency_focus": "hash map lookups",
                "expected_signals": ["names the data structure"],
                "code_contract": {
                    "function_name": "two_sum",
                    "language": "python",
                    "test_cases": [
                        {"input": "[2,7]", "expected_output": "[0, 1]"}
                    ],
                },
            },
        )
        # The model claimed 5/5 on accuracy; the deterministic grade caps it
        # because no code block was submitted.
        self.assertEqual(result["rubric"]["technical_accuracy"], 1)
        self.assertFalse(result["code_grade"]["passes"] if False else result["code_grade"]["passed"])
        self.assertEqual(result["code_grade"]["reason"], "no_code_block")
        self.assertIn(
            "technical_accuracy_capped_at_1:no_code_block",
            result["deterministic_evidence_applied"],
        )
        # The evaluator was shown the measurement before it scored.
        self.assertIn("Deterministic code evidence", llm.eval_prompts[-1])
        self.assertIn("code_block_present: False", llm.eval_prompts[-1])

    async def test_a_design_answer_is_scored_against_the_measured_diagram(self):
        generator, llm = self._generator(
            '{"rubric":{"technical_accuracy":3,"reasoning_depth":3,'
            '"communication_clarity":3,"completeness":3,"confidence_signal":3,'
            '"overall":60},"system_design":{"architecture":5,"tradeoffs":5,'
            '"scalability":5,"failure_modes":5,"communication":5},'
            '"strengths":["Clear"],"improvements":["Add detail"],'
            '"follow_up_note":"Add operators."}'
        )
        result = await generator.evaluate_answer(
            session=_session(interview_type="system_design"),
            question="Design a fan-out service.",
            user_answer=(
                "```mermaid\n"
                "graph TD\n"
                "  a --> b\n"
                "  b --> c\n"
                "```\n"
                "That is the design."
            ).replace("\\n", "\n"),
            turn_index=0,
        )
        # Three anonymous nodes and no labels: no architectural evidence, so the
        # deterministic structure caps architecture and communication at 2.
        self.assertEqual(result["system_design"]["architecture"], 2)
        self.assertEqual(result["system_design"]["communication"], 2)
        self.assertTrue(result["diagram_analysis"]["present"])
        self.assertEqual(len(result["deterministic_evidence_applied"]), 2)
        self.assertIn(
            "Deterministic diagram evidence", llm.eval_prompts[-1]
        )
        self.assertIn("diagram_without_structure", llm.eval_prompts[-1])


class InterviewPersonaTests(unittest.TestCase):
    def test_every_persona_normalises(self):
        self.assertEqual(InterviewGenerator._persona(_session(persona="bar-raiser")), "bar-raiser")
        self.assertEqual(InterviewGenerator._persona(_session(persona="SUPPORTIVE")), "supportive")
        # Unknown or absent values fall back to the existing neutral behaviour.
        self.assertEqual(InterviewGenerator._persona(_session(persona="coach")), "neutral")
        self.assertEqual(InterviewGenerator._persona(_session()), "neutral")

    def test_the_persona_is_threaded_into_the_question_prompt(self):
        prompt = InterviewGenerator._question_prompt(
            _session(interview_type="system_design", persona="bar-raiser"), []
        )
        self.assertIn("- persona: bar-raiser", prompt)
        self.assertIn("bar-raiser: hold the top-of-band bar", prompt)

    def test_the_persona_is_threaded_into_the_evaluation_prompt(self):
        prompt = InterviewGenerator._evaluate_prompt(
            _session(persona="supportive"), "Q?", "A", 1
        )
        self.assertIn("- persona: supportive", prompt)
        self.assertIn("supportive: acknowledge", prompt)

    def test_neutral_prompts_carry_no_persona_rules(self):
        prompt = InterviewGenerator._question_prompt(_session(persona="neutral"), [])
        self.assertIn("- persona: neutral", prompt)
        self.assertNotIn("bar-raiser: hold", prompt)


if __name__ == "__main__":
    unittest.main()
