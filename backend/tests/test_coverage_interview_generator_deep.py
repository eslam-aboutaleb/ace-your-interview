"""Coverage for app/services/interview_generator.py.

Complements ``tests/test_interview_generator.py`` (retry bounds, degraded
turns, note repair) with the deterministic layer underneath it: phase banding,
progression rules per interview type, near-duplicate detection, every payload
validation rejection, the per-type default coaching-note builders, and
``build_report``'s readiness bandings.

Nothing here reaches a network: the injected LLM replays canned results.
"""

import asyncio
import json
import unittest

from app.services import interview_generator as ig
from app.services.interview_generator import InterviewGenerator


def _failure(error_code: str) -> dict:
    return {
        "success": False,
        "analysis": "",
        "metadata": {},
        "error": "boom",
        "error_code": error_code,
        "finish_reason": "",
        "usage": {},
    }


def _success(analysis: str) -> dict:
    return {
        "success": True,
        "analysis": analysis,
        "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
        "error": "",
        "error_code": "",
        "finish_reason": "stop",
        "usage": {},
    }


class ScriptedLLM:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if self._results:
            return dict(self._results.pop(0))
        return _failure("llm_call_failed")


class RepeatingLLM:
    """Replays one success for every call, forever."""

    def __init__(self, analysis):
        self.analysis = analysis
        self.calls = 0
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        return _success(self.analysis)


class FakeParser:
    def __init__(self, topics=None):
        self._topics = topics or []

    def list_topics(self, track=None, level=None, q=None):
        return self._topics


class FakeMCP:
    def __init__(self, context="MCP says the team runs Kafka."):
        self.context = context
        self.calls: list[dict] = []

    async def gather_context(self, **kwargs):
        self.calls.append(kwargs)
        return self.context


def _gen(results=None, parser=None, mcp=None) -> InterviewGenerator:
    return InterviewGenerator(ScriptedLLM(results or []), parser or FakeParser(), mcp)


_SESSION = {
    "session_id": "s1",
    "track": "backend",
    "level": "mid",
    "interview_type": "technical",
    "turn_count": 5,
    "turns_completed": 0,
}


class ClampIntTests(unittest.TestCase):
    def test_a_non_numeric_value_falls_back_to_the_default(self):
        self.assertEqual(ig._clamp_int("abc", 1, 10, 3), 3)
        self.assertEqual(ig._clamp_int(None, 1, 10, 7), 7)
        self.assertEqual(ig._clamp_int(["3"], 1, 10, 4), 4)

    def test_numeric_strings_are_parsed_and_clamped_to_the_band(self):
        self.assertEqual(ig._clamp_int("6", 1, 10, 3), 6)
        self.assertEqual(ig._clamp_int("99", 1, 10, 3), 10)
        self.assertEqual(ig._clamp_int(-5, 0, 5, 3), 0)


class DegradedTurnTests(unittest.TestCase):
    def test_a_non_dict_turn_is_never_treated_as_degraded(self):
        self.assertIs(ig._is_degraded_turn("not a turn"), False)
        self.assertIs(ig._is_degraded_turn(None), False)

    def test_an_explicit_degraded_flag_wins_even_with_numbers(self):
        self.assertIs(
            ig._is_degraded_turn({"degraded": True, "rubric": {"overall": 90}}), True
        )

    def test_an_all_none_rubric_is_degraded_without_the_flag(self):
        self.assertIs(
            ig._is_degraded_turn({"rubric": {"overall": None, "technical_accuracy": None}}),
            True,
        )
        self.assertIs(ig._is_degraded_turn({"rubric": {}}), True)
        self.assertIs(ig._is_degraded_turn({}), True)

    def test_a_turn_with_any_numeric_rubric_is_scored(self):
        self.assertIs(ig._is_degraded_turn({"rubric": {"overall": 0}}), False)


class NearDuplicateTests(unittest.TestCase):
    def test_an_empty_candidate_is_never_a_near_duplicate(self):
        self.assertIs(InterviewGenerator._is_near_duplicate_question("", ["anything"]), False)
        self.assertIs(InterviewGenerator._is_near_duplicate_question("???", ["anything"]), False)

    def test_blank_prior_entries_are_skipped(self):
        self.assertIs(
            InterviewGenerator._is_near_duplicate_question(
                "Explain database index selection strategies", ["", "   "]
            ),
            False,
        )

    def test_an_exact_normalised_match_is_a_duplicate(self):
        self.assertIs(
            InterviewGenerator._is_near_duplicate_question(
                "Explain database index selection strategies",
                ["Explain database  index selection strategies"],
            ),
            True,
        )

    def test_a_near_identical_rephrasing_is_a_duplicate(self):
        self.assertIs(
            InterviewGenerator._is_near_duplicate_question(
                "Explain how database index selection strategies work",
                ["Explain how database index selection strategies work well"],
            ),
            True,
        )

    def test_a_high_token_overlap_is_a_duplicate(self):
        # Too different for the exact check and just under the 0.9 ratio, but
        # the token sets still overlap by well over the 70% Jaccard threshold.
        self.assertIs(
            InterviewGenerator._is_near_duplicate_question(
                "Explain connection pool sizing for a throughput sensitive request path",
                ["Explain connection pool sizing for a latency sensitive request path"],
            ),
            True,
        )
        self.assertIs(
            InterviewGenerator._is_near_duplicate_question(
                "Describe how a consistent hash ring redistributes load after a node failure",
                ["Describe how a consistent hash ring handles a node failure"],
            ),
            False,
        )

    def test_a_genuinely_different_question_is_not_a_duplicate(self):
        self.assertIs(
            InterviewGenerator._is_near_duplicate_question(
                "Describe connection pooling behaviour under contention",
                ["Explain database index selection strategies"],
            ),
            False,
        )

    def test_a_punctuation_only_prior_yields_no_tokens_and_is_skipped(self):
        # "!?" normalises to "" for the prior, so the token-overlap check bails.
        self.assertIs(
            InterviewGenerator._is_near_duplicate_question(
                "Describe connection pooling behaviour under contention", ["!?"]
            ),
            False,
        )


class TurnPhaseTests(unittest.TestCase):
    def test_a_two_turn_session_opens_then_lates(self):
        self.assertEqual(
            InterviewGenerator._turn_phase({"turn_count": 2, "turns_completed": 0}), "opening"
        )
        self.assertEqual(
            InterviewGenerator._turn_phase({"turn_count": 2, "turns_completed": 1}), "late"
        )

    def test_the_middle_band_is_reached_between_the_opening_and_late_turns(self):
        self.assertEqual(
            InterviewGenerator._turn_phase({"turn_count": 5, "turns_completed": 2}), "middle"
        )

    def test_completed_turns_are_clamped_to_the_declared_total(self):
        self.assertEqual(
            InterviewGenerator._turn_phase({"turn_count": 3, "turns_completed": 99}), "late"
        )
        # A non-positive turn_count is clamped up to 1, and a negative
        # completed count is clamped to 0, so the only turn is the late one.
        self.assertEqual(
            InterviewGenerator._turn_phase({"turn_count": 0, "turns_completed": -4}), "late"
        )

    def test_a_missing_turn_count_defaults_to_three(self):
        self.assertEqual(InterviewGenerator._turn_phase({}), "opening")
        self.assertEqual(InterviewGenerator._turn_phase({"turns_completed": 2}), "late")


class ProgressionRuleTests(unittest.TestCase):
    def _rules(self, session, turns_completed):
        session = {**session, "turn_count": 5, "turns_completed": turns_completed}
        return InterviewGenerator._question_progression_rules(session)

    def test_coding_sessions_get_a_phase_specific_coding_rule(self):
        coding = {"interview_type": "coding"}
        self.assertIn("Middle coding turns", self._rules(coding, 2)[-1])
        self.assertIn("Late coding turns", self._rules(coding, 4)[-1])
        self.assertIn("Opening coding turns", self._rules(coding, 0)[-1])
        self.assertIn("middle phase", self._rules(coding, 2)[0])

    def test_behavioral_sessions_get_a_phase_specific_behavioral_rule(self):
        behavioral = {"interview_type": "behavioral"}
        self.assertIn("Middle behavioral turns", self._rules(behavioral, 2)[-1])
        self.assertIn("Late behavioral turns", self._rules(behavioral, 4)[-1])
        self.assertIn("Opening behavioral turns", self._rules(behavioral, 0)[-1])

    def test_a_general_session_gets_a_phase_specific_general_rule(self):
        general = {"interview_type": "technical"}
        self.assertIn("Middle turns", self._rules(general, 2)[-1])
        self.assertIn("Late turns", self._rules(general, 4)[-1])
        self.assertIn("Opening turns", self._rules(general, 0)[-1])

    def test_an_unknown_interviewer_style_is_neutralised(self):
        self.assertEqual(
            InterviewGenerator._interviewer_style({"interviewer_style": "sarcastic"}), "neutral"
        )
        self.assertEqual(
            InterviewGenerator._interviewer_style({"interviewer_style": "  CHALLENGING "}),
            "challenging",
        )
        self.assertEqual(InterviewGenerator._interviewer_style({}), "neutral")

    def test_only_the_deep_feedback_mode_is_deep(self):
        self.assertEqual(InterviewGenerator._feedback_mode({"feedback_mode": "DEEP"}), "deep")
        self.assertEqual(InterviewGenerator._feedback_mode({"feedback_mode": "novel"}), "concise")
        self.assertEqual(InterviewGenerator._feedback_mode({}), "concise")

    def test_rule_lines_strip_bullets_and_blank_lines(self):
        self.assertEqual(
            InterviewGenerator._rule_lines_from_block("\n- one\n\n  - two  \nthree\n"),
            ["one", "two", "three"],
        )
        self.assertEqual(InterviewGenerator._rule_lines_from_block(""), [])


class ValidateQuestionPayloadTests(unittest.TestCase):
    def _payload(self, **overrides):
        base = {
            "question": "Explain how connection pooling improves throughput",
            "competency_focus": "resource reuse",
            "expected_signals": ["pool sizing", "contention"],
        }
        base.update(overrides)
        return base

    def test_a_conforming_payload_is_accepted(self):
        ok, issue = InterviewGenerator._validate_question_payload(self._payload(), [])
        self.assertTrue(ok)
        self.assertEqual(issue, "")

    def test_a_too_short_question_is_rejected(self):
        ok, issue = InterviewGenerator._validate_question_payload(
            self._payload(question="Short?"), []
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "question_too_short")

    def test_a_question_over_the_word_limit_is_rejected(self):
        ok, issue = InterviewGenerator._validate_question_payload(
            self._payload(question=" ".join(["word"] * (max_words_limit() + 1))), [],
            max_words=max_words_limit(),
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "question_too_long")

    def test_a_thin_competency_focus_is_rejected(self):
        ok, issue = InterviewGenerator._validate_question_payload(
            self._payload(competency_focus="ab"), []
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "missing_competency_focus")

    def test_expected_signals_must_be_a_list_of_two_to_five(self):
        for signals in ("a", [], ["one"], ["a", "b", "c", "d", "e", "f"]):
            with self.subTest(signals=signals):
                ok, issue = InterviewGenerator._validate_question_payload(
                    self._payload(expected_signals=signals), []
                )
                self.assertFalse(ok)
                self.assertEqual(issue, "invalid_expected_signals")

    def test_an_exact_repeat_of_a_asked_question_is_rejected(self):
        ok, issue = InterviewGenerator._validate_question_payload(
            self._payload(), ["Explain how connection pooling improves throughput"]
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "duplicate_question")

    def test_a_semantic_repeat_of_a_asked_question_is_rejected(self):
        ok, issue = InterviewGenerator._validate_question_payload(
            {"question": "Explain caching strategies for write heavy traffic",
             "competency_focus": "cache design",
             "expected_signals": ["ttl", "invalidation"]},
            ["Explain caching strategies for read heavy traffic"],
        )
        self.assertFalse(ok)
        self.assertEqual(issue, "near_duplicate_question")


class ValidateEvalPayloadTests(unittest.TestCase):
    def _payload(self, **overrides):
        base = {
            "rubric": {
                "technical_accuracy": 4,
                "reasoning_depth": 3,
                "communication_clarity": 4,
                "completeness": 3,
                "confidence_signal": 3,
                "overall": 74,
            },
            "strengths": ["Good structure"],
            "improvements": ["Add metrics"],
            "follow_up_note": "A long enough coaching note about the answer.",
        }
        base.update(overrides)
        return base

    def test_a_conforming_payload_is_accepted(self):
        ok, issue = InterviewGenerator._validate_eval_payload(self._payload())
        self.assertTrue(ok)
        self.assertEqual(issue, "")

    def test_a_missing_or_non_dict_rubric_is_rejected(self):
        ok, issue = InterviewGenerator._validate_eval_payload(self._payload(rubric=None))
        self.assertFalse(ok)
        self.assertEqual(issue, "missing_rubric")

    def test_each_required_rubric_key_is_named_in_the_rejection(self):
        for key in (
            "technical_accuracy",
            "reasoning_depth",
            "communication_clarity",
            "completeness",
            "confidence_signal",
            "overall",
        ):
            with self.subTest(key=key):
                rubric = dict(self._payload()["rubric"])
                rubric.pop(key)
                ok, issue = InterviewGenerator._validate_eval_payload(
                    self._payload(rubric=rubric)
                )
                self.assertFalse(ok)
                self.assertEqual(issue, f"missing_{key}")

    def test_empty_bullet_lists_are_rejected(self):
        for field, issue in (("strengths", "missing_strengths"), ("improvements", "missing_improvements")):
            with self.subTest(field=field):
                ok, reason = InterviewGenerator._validate_eval_payload(self._payload(**{field: []}))
                self.assertFalse(ok)
                self.assertEqual(reason, issue)
        ok, issue = InterviewGenerator._validate_eval_payload(self._payload(strengths="a bullet"))
        self.assertEqual(issue, "missing_strengths")

    def test_a_stub_follow_up_note_is_rejected(self):
        ok, issue = InterviewGenerator._validate_eval_payload(self._payload(follow_up_note="  ok  "))
        self.assertFalse(ok)
        self.assertEqual(issue, "missing_follow_up_note")


class HeadingRepairTests(unittest.TestCase):
    def test_a_heading_that_is_not_required_is_left_untouched(self):
        note = "### Bonus notes\nsome body\n"
        out, repaired = InterviewGenerator._normalise_follow_up_headings(note)
        self.assertEqual(out, note.strip())
        self.assertFalse(repaired)

    def test_a_trailing_colon_on_a_required_heading_is_stripped(self):
        out, repaired = InterviewGenerator._normalise_follow_up_headings(
            "### What to improve next:\n- tighten it\n"
        )
        self.assertEqual(out, "### What to improve next\n- tighten it")
        self.assertTrue(repaired)

    def test_an_already_canonical_note_is_not_flagged_as_repaired(self):
        note = "### What to improve next\n- tighten it\n"
        out, repaired = InterviewGenerator._normalise_follow_up_headings(note)
        self.assertEqual(out, note.strip())
        self.assertFalse(repaired)

    def test_an_empty_note_reports_every_required_heading_missing(self):
        self.assertEqual(
            InterviewGenerator._follow_up_note_missing_headings(""),
            list(ig._FOLLOW_UP_NOTE_HEADINGS),
        )
        self.assertFalse(InterviewGenerator._follow_up_note_has_required_sections(""))

    def test_a_missing_section_is_filled_from_the_default_and_the_others_survive(self):
        gen = _gen()
        note, repaired = gen._follow_up_note_repair(
            "A short preamble.\n\n"
            "### What strong interviewers wanted to hear\n\n"
            "### What to improve next\n- be concrete\n",
            session={"track": "backend"},
            strengths=["Named the constraint"],
            improvements=["Quantify the tradeoff"],
        )
        self.assertTrue(repaired)
        # The written section and the preamble survive; the absent section and
        # the empty one are both filled from the deterministic default, whose
        # intro is the raw model note.
        self.assertTrue(note.startswith("A short preamble."))
        self.assertIn("- be concrete", note)
        self.assertIn(
            "What strong interviewers wanted to hear\nA short preamble.", note
        )
        self.assertTrue(InterviewGenerator._follow_up_note_has_required_sections(note))
        # Deterministic: the same input repairs to the same note.
        self.assertEqual(
            note,
            gen._follow_up_note_repair(
                "A short preamble.\n\n### What strong interviewers wanted to hear\n\n"
                "### What to improve next\n- be concrete\n",
                session={"track": "backend"},
                strengths=["Named the constraint"],
                improvements=["Quantify the tradeoff"],
            )[0],
        )

    def test_an_empty_section_never_overwrites_the_default_body(self):
        gen = _gen()
        # The heading is present but has no body, so it is not counted as
        # written and the default body (built from the strengths) wins.
        note, repaired = gen._follow_up_note_repair(
            "### What strong interviewers wanted to hear\n\n### What to improve next\n",
            session={"track": "backend"},
            strengths=["Named the constraint"],
            improvements=["Quantify the tradeoff"],
        )
        self.assertTrue(repaired)
        self.assertIn("Named the constraint", note)
        self.assertIn("- Quantify the tradeoff", note)
        self.assertTrue(InterviewGenerator._follow_up_note_has_required_sections(note))

    def test_a_note_whose_headings_all_exist_is_returned_untouched(self):
        gen = _gen()
        note = (
            "### What strong interviewers wanted to hear\n"
            "Wanted the capacity math.\n\n"
            "### What to improve next\n- Quantify it.\n\n"
            "### Stronger sample answer\nI would size the queue first."
        )
        out, repaired = gen._follow_up_note_repair(
            note,
            session={"track": "backend"},
            strengths=["Good structure"],
            improvements=["Add a metric"],
        )
        self.assertEqual(out, note)
        self.assertFalse(repaired)

    def test_the_default_note_is_built_per_session_type_and_feedback_mode(self):
        gen = _gen()
        coding = gen._build_default_follow_up_note(
            session={"interview_type": "coding", "feedback_mode": "deep"},
            strengths=["Chose a hash map"],
            improvements=["State complexity"],
            note_seed="",
        )
        behavioral = gen._build_default_follow_up_note(
            session={"interview_type": "behavioral", "feedback_mode": "concise"},
            strengths=[],
            improvements=[],
            note_seed="",
        )
        general = gen._build_default_follow_up_note(
            session={"interview_type": "technical", "feedback_mode": "deep"},
            strengths=["Framed the goal"],
            improvements=["Add a metric"],
            note_seed="a seed paragraph",
        )
        self.assertIn("name the invariant", coding.lower())
        self.assertIn("STAR form", behavioral)
        self.assertIn("a seed paragraph", general)
        # Empty bullet lists fall back to generic coaching bullets.
        self.assertIn("- Make the structure more explicit", behavioral)
        self.assertIn("- Chose a hash map", coding)
        # The deep mode's sample answer is strictly longer than the concise one.
        self.assertGreater(
            len(InterviewGenerator._default_stronger_sample_answer(
                {"interview_type": "coding"}, "deep"
            )),
            len(InterviewGenerator._default_stronger_sample_answer(
                {"interview_type": "coding"}, "concise"
            )),
        )
        self.assertGreater(
            len(InterviewGenerator._default_stronger_sample_answer(
                {"interview_type": "behavioral"}, "deep"
            )),
            len(InterviewGenerator._default_stronger_sample_answer(
                {"interview_type": "behavioral"}, "concise"
            )),
        )

    def test_default_focus_lines_cover_all_three_session_types(self):
        self.assertIn(
            "invariant",
            " ".join(InterviewGenerator._default_follow_up_focus_lines(
                {"interview_type": "coding"}
            )),
        )
        self.assertIn(
            "situation/task",
            " ".join(InterviewGenerator._default_follow_up_focus_lines(
                {"interview_type": "behavioral"}
            )),
        )
        self.assertIn(
            "constraints",
            " ".join(InterviewGenerator._default_follow_up_focus_lines(
                {"interview_type": "technical"}
            )),
        )


class NormaliseEvalPayloadTests(unittest.TestCase):
    def test_out_of_band_rubric_numbers_are_clamped_and_defaults_filled(self):
        gen = _gen()
        out = gen._normalise_eval_payload(
            {
                "rubric": {
                    "technical_accuracy": 99,
                    "reasoning_depth": -4,
                    "overall": "78",
                },
                "strengths": ["  ", "Named the constraint", "Chose a hash map"],
                "improvements": ["Quantify the tradeoff", "", "State complexity"],
                "follow_up_note": (
                    "### What strong interviewers wanted to hear\nWanted math.\n\n"
                    "### What to improve next\n- Quantify.\n\n"
                    "### Stronger sample answer\nI would size the queue first."
                ),
            },
            session={"track": "backend"},
        )
        self.assertEqual(out["rubric"]["technical_accuracy"], 5)
        self.assertEqual(out["rubric"]["reasoning_depth"], 0)
        self.assertEqual(out["rubric"]["overall"], 78)
        # Unspecified rubric dimensions default to the neutral middle.
        self.assertEqual(out["rubric"]["completeness"], 3)
        self.assertEqual(out["strengths"], ["Named the constraint", "Chose a hash map"])
        self.assertEqual(out["improvements"], ["Quantify the tradeoff", "State complexity"])
        self.assertIs(out["degraded"], False)

    def test_blank_bullet_lists_are_replaced_with_generic_coaching_text(self):
        gen = _gen()
        out = gen._normalise_eval_payload(
            {"rubric": {}, "strengths": ["  "], "improvements": "not a list"},
            session={"track": "backend"},
        )
        self.assertEqual(len(out["strengths"]), 1)
        self.assertEqual(len(out["improvements"]), 1)
        self.assertEqual(out["rubric"]["overall"], 60)


class GenerateQuestionMcpTests(unittest.TestCase):
    def test_external_context_is_appended_to_the_question_prompt(self):
        mcp = FakeMCP()
        llm = ScriptedLLM(
            [
                _success(
                    json.dumps(
                        {
                            "question": "How do you keep a shared cache coherent?",
                            "competency_focus": "cache coherence",
                            "expected_signals": ["invalidation", "ttl"],
                        }
                    )
                )
            ]
        )
        gen = InterviewGenerator(llm, FakeParser(), mcp)
        out = asyncio.run(gen.generate_question(session=_SESSION, turns=[]))

        self.assertEqual(mcp.calls[0]["flow"], "interview")
        self.assertIn("MCP says the team runs Kafka.", llm.prompts[0])
        self.assertIn("cache coherent", out["question"])

    def test_an_empty_mcp_context_adds_no_external_block(self):
        mcp = FakeMCP(context="")
        llm = ScriptedLLM(
            [
                _success(
                    json.dumps(
                        {
                            "question": "How do you keep a shared cache coherent?",
                            "competency_focus": "cache coherence",
                            "expected_signals": ["invalidation", "ttl"],
                        }
                    )
                )
            ]
        )
        asyncio.run(InterviewGenerator(llm, FakeParser(), mcp).generate_question(
            session=_SESSION, turns=[]
        ))
        self.assertNotIn("External context", llm.prompts[0])

    def test_a_terminal_provider_outage_aborts_immediately_with_its_code(self):
        for code in ("llm_budget_exceeded", "llm_truncated", "llm_call_failed"):
            with self.subTest(code=code):
                llm = ScriptedLLM([_failure(code)])
                gen = InterviewGenerator(llm, FakeParser())
                with self.assertRaises(RuntimeError) as ctx:
                    asyncio.run(gen.generate_question(session=_SESSION, turns=[]))
                self.assertIn(code, str(ctx.exception))
                self.assertEqual(llm.calls, 1)

    def test_the_retry_prompt_names_the_validation_issue_and_keeps_the_base_task(self):
        llm = ScriptedLLM(
            [
                _success(json.dumps({"question": "Short?"})),
                _success(
                    json.dumps(
                        {
                            "question": "How do you keep a shared cache coherent?",
                            "competency_focus": "cache coherence",
                            "expected_signals": ["invalidation", "ttl"],
                        }
                    )
                ),
            ]
        )
        asyncio.run(InterviewGenerator(llm, FakeParser()).generate_question(
            session=_SESSION, turns=[]
        ))
        self.assertEqual(llm.calls, 2)
        self.assertIn(
            "Your previous JSON was invalid (question_too_short)", llm.prompts[1]
        )
        # The original task is preserved so the retry stays on-topic.
        self.assertIn("Generate the NEXT interview question as strict JSON.", llm.prompts[1])

    def test_recent_turn_questions_join_the_no_repeat_history(self):
        repeat = json.dumps(
            {
                "question": "How do you keep a shared cache coherent?",
                "competency_focus": "cache coherence",
                "expected_signals": ["invalidation", "ttl"],
            }
        )
        llm = RepeatingLLM(repeat)
        with self.assertRaisesRegex(RuntimeError, "before timeout"):
            asyncio.run(
                InterviewGenerator(llm, FakeParser()).generate_question(
                    session=_SESSION,
                    turns=[
                        {
                            "turn_index": 1,
                            "question": "How do you keep a shared cache coherent?",
                            "user_answer": "TTL plus invalidation.",
                        }
                    ],
                )
            )
        # The turn that was just answered is in the history, so replaying the
        # same question is refused for every retry instead of repeating a turn.
        self.assertEqual(llm.calls, ig._MAX_ATTEMPTS)
        self.assertIn("duplicate_question", llm.prompts[1])
        self.assertIn("Recent turns", llm.prompts[0])

    def test_the_word_limit_is_widened_for_coding_and_behavioral_sessions(self):
        coding = json.dumps(
            {
                "question": " ".join(["word"] * 50),
                "competency_focus": "invariants",
                "expected_signals": ["a", "b"],
            }
        )
        out = asyncio.run(
            InterviewGenerator(
                ScriptedLLM([_success(coding)]), FakeParser()
            ).generate_question(
                session={**_SESSION, "interview_type": "coding", "interview_type_key": 1},
                turns=[],
            )
        )
        # 50 words is over the general 45-word cap but under the coding 65 cap.
        self.assertEqual(out["competency_focus"], "invariants")

        behavioral = json.dumps(
            {
                "question": " ".join(["word"] * 50),
                "competency_focus": "ownership",
                "expected_signals": ["a", "b"],
            }
        )
        out = asyncio.run(
            InterviewGenerator(ScriptedLLM([_success(behavioral)]), FakeParser()).generate_question(
                session={**_SESSION, "interview_type": "behavioral"}, turns=[]
            )
        )
        self.assertEqual(out["competency_focus"], "ownership")

    def test_a_coding_question_over_sixty_five_words_is_rejected(self):
        llm = ScriptedLLM(
            [
                _success(
                    json.dumps(
                        {
                            "question": " ".join(["word"] * 70),
                            "competency_focus": "invariants",
                            "expected_signals": ["a", "b"],
                        }
                    )
                )
            ]
        )
        with self.assertRaises(RuntimeError):
            asyncio.run(
                InterviewGenerator(llm, FakeParser()).generate_question(
                    session={**_SESSION, "interview_type": "coding"}, turns=[]
                )
            )
        self.assertIn("question_too_long", llm.prompts[1])


class EvaluateAnswerMcpTests(unittest.TestCase):
    def test_external_context_is_appended_to_the_evaluation_prompt(self):
        mcp = FakeMCP()
        llm = ScriptedLLM([_success(_valid_eval())])
        asyncio.run(
            InterviewGenerator(llm, FakeParser(), mcp).evaluate_answer(
                session=_SESSION, question="Q?", user_answer="A.", turn_index=1
            )
        )
        self.assertEqual(mcp.calls[0]["flow"], "interview")
        self.assertIn("MCP says the team runs Kafka.", llm.prompts[0])

    def test_an_empty_mcp_context_adds_no_external_block_to_the_eval_prompt(self):
        mcp = FakeMCP(context="")
        llm = ScriptedLLM([_success(_valid_eval())])
        asyncio.run(
            InterviewGenerator(llm, FakeParser(), mcp).evaluate_answer(
                session=_SESSION, question="Q?", user_answer="A.", turn_index=1
            )
        )
        self.assertNotIn("External context", llm.prompts[0])

    def test_the_retry_prompt_names_the_eval_validation_issue(self):
        llm = ScriptedLLM(
            [_success(json.dumps({"rubric": {"overall": 80}})), _success(_valid_eval())]
        )
        asyncio.run(
            InterviewGenerator(llm, FakeParser()).evaluate_answer(
                session=_SESSION, question="Q?", user_answer="A.", turn_index=1
            )
        )
        self.assertEqual(llm.calls, 2)
        self.assertIn("missing_technical_accuracy", llm.prompts[1])


class BuildReportReadinessTests(unittest.TestCase):
    def _turn(self, overall, **overrides):
        turn = {
            "turn_index": 1,
            "question": "Q1",
            "user_answer": "A1",
            "rubric": {
                "technical_accuracy": overall / 20,
                "reasoning_depth": overall / 20,
                "communication_clarity": overall / 20,
                "completeness": overall / 20,
                "confidence_signal": overall / 20,
                "overall": overall,
            },
            "strengths": ["Clear structure"],
            "improvements": ["Add a metric"],
            "created_at": "2026-01-01T00:00:00Z",
        }
        turn.update(overrides)
        return turn

    def test_each_readiness_band_is_selected_by_the_overall_average(self):
        gen = _gen()
        for overall, label in ((90, "Strong"), (75, "On Track"), (55, "Developing"), (10, "Needs Work")):
            with self.subTest(overall=overall):
                report = gen.build_report(session={"track": "backend"}, turns=[self._turn(overall)])
                self.assertEqual(report["readiness_label"], label)
                self.assertEqual(report["overall_score"], float(overall))

    def test_a_weak_dimension_adds_the_prioritise_next_step_and_a_focus_area(self):
        gen = _gen()
        report = gen.build_report(
            session={"track": "backend"},
            turns=[
                {
                    "turn_index": 1,
                    "question": "Q1",
                    "user_answer": "A1",
                    "rubric": {
                        "technical_accuracy": 1.0,
                        "reasoning_depth": 4.0,
                        "communication_clarity": 4.0,
                        "completeness": 4.0,
                        "confidence_signal": 4.0,
                        "overall": 70,
                    },
                    "strengths": [" Clear structure ", "", "Clear structure", "Named the tradeoff"],
                    "improvements": ["Add a metric"],
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
        )
        self.assertEqual(report["weak_competencies"], ["technical accuracy"])
        self.assertIn(
            "Prioritize weak competencies in your next mock interview round.",
            report["next_steps"],
        )
        self.assertIn("- technical accuracy", report["summary"])
        # Duplicate and blank strengths collapse; the list is capped at four.
        self.assertEqual(report["strengths"], ["Clear structure", "Named the tradeoff"])

    def test_the_strength_list_is_capped_at_four_entries(self):
        gen = _gen()
        report = gen.build_report(
            session={"track": "backend"},
            turns=[
                {
                    "turn_index": 1,
                    "question": "Q1",
                    "user_answer": "A1",
                    "rubric": {
                        "technical_accuracy": 5.0,
                        "reasoning_depth": 5.0,
                        "communication_clarity": 5.0,
                        "completeness": 5.0,
                        "confidence_signal": 5.0,
                        "overall": 96,
                    },
                    "strengths": ["one", "two", "three", "four", "five"],
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
        )
        self.assertEqual(report["strengths"], ["one", "two", "three", "four"])

    def test_a_strong_session_reports_no_weak_competency(self):
        gen = _gen()
        report = gen.build_report(session={"track": "backend"}, turns=[self._turn(95)])
        self.assertEqual(
            report["weak_competencies"], ["No major weak competency detected"]
        )
        self.assertEqual(len(report["next_steps"]), 2)

    def test_non_numeric_rubric_values_are_skipped_without_shifting_the_average(self):
        gen = _gen()
        report = gen.build_report(
            session={"track": "backend"},
            turns=[
                {
                    "turn_index": 1,
                    "question": "Q1",
                    "user_answer": "A1",
                    "rubric": {
                        "technical_accuracy": "four",
                        "reasoning_depth": None,
                        "communication_clarity": [],
                        "completeness": {},
                        "confidence_signal": "",
                        "overall": 80,
                    },
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
        )
        self.assertEqual(report["rubric_averages"]["technical_accuracy"], 0.0)
        self.assertEqual(report["overall_score"], 80.0)
        self.assertEqual(report["strengths"], [])

    def test_recommended_topics_come_from_the_parser_for_the_session_track(self):
        parser = FakeParser(
            topics=[
                type("T", (), {"id": f"t{i}"})() for i in range(6)
            ]
        )
        gen = _gen(parser=parser)
        report = gen.build_report(session={"track": "backend"}, turns=[self._turn(90)])
        self.assertEqual(report["recommended_topic_ids"], ["t0", "t1", "t2", "t3"])

    def test_an_empty_turn_list_reports_not_started(self):
        report = _gen().build_report(session={"track": "backend"}, turns=[])
        self.assertEqual(report["completed_turns"], 0)
        self.assertEqual(report["created_at"], "")
        self.assertEqual(report["readiness_label"], "Not Started")

    def test_a_mixed_session_counts_the_degraded_turns_in_completed_turns(self):
        gen = _gen()
        report = gen.build_report(
            session={"track": "backend"},
            turns=[self._turn(80), {"turn_index": 2, "degraded": True, "rubric": {}}],
        )
        self.assertEqual(report["completed_turns"], 2)
        self.assertEqual(report["rubric_averages"]["overall"], 80.0)
        self.assertIn("1 turn(s) could not be evaluated", report["summary"])


def _valid_eval() -> str:
    return json.dumps(
        {
            "rubric": {
                "technical_accuracy": 4,
                "reasoning_depth": 3,
                "communication_clarity": 4,
                "completeness": 3,
                "confidence_signal": 3,
                "overall": 74,
            },
            "strengths": ["Good structure"],
            "improvements": ["Add metrics"],
            "follow_up_note": (
                "### What strong interviewers wanted to hear\nWanted explicit math.\n\n"
                "### What to improve next\n- Quantify the trade-off.\n\n"
                "### Stronger sample answer\nI would size the queue first."
            ),
        }
    )


def max_words_limit() -> int:
    return 45