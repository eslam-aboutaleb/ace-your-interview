"""Stack/track-aware skill taxonomy: per-track packs and call-site wiring.

The four track packs (backend, frontend, system design, AI stack) must own
their own vocabulary, while an empty or unknown track has to keep the DSA
pack byte-for-byte — the problem-solving roadmap is generated against it.
"""

from __future__ import annotations

import asyncio
import json
import unittest

from app.services.llm_skills import (
    AI_STACK_SKILLS,
    BACKEND_SKILLS,
    FRONTEND_SKILLS,
    SYSTEM_DESIGN_SKILLS,
    TECHNICAL_SKILLS,
    normalise_track,
    skill_context_block,
    skills_for_session,
    technical_skill_block,
)
from app.services.question_generator import (
    QuestionGenerator,
    _build_card_prompt,
    _build_prompt,
    _build_quiz_prompt,
)

TRACK_PACKS = {
    "backend": BACKEND_SKILLS,
    "frontend": FRONTEND_SKILLS,
    "system_design": SYSTEM_DESIGN_SKILLS,
    "ai_stack": AI_STACK_SKILLS,
}

RECORD_KEYS = (
    "name",
    "category",
    "description",
    "patterns",
    "common_mistakes",
    "interview_tricks",
    "complexity",
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


class TrackPackShapeTests(unittest.TestCase):
    def test_packs_carry_the_technical_record_shape(self):
        for track, pack in TRACK_PACKS.items():
            self.assertTrue(pack, f"{track} pack is empty")
            for skill_id, record in pack.items():
                for key in RECORD_KEYS:
                    self.assertIn(key, record, f"{track}/{skill_id} misses {key}")
                self.assertTrue(record["name"])
                self.assertTrue(record["description"])
                self.assertTrue(record["complexity"])
                self.assertTrue(record["patterns"])
                self.assertTrue(record["common_mistakes"])
                self.assertTrue(record["interview_tricks"])

    def test_every_pack_has_three_to_six_patterns_per_skill(self):
        for track, pack in TRACK_PACKS.items():
            for skill_id, record in pack.items():
                self.assertGreaterEqual(len(record["patterns"]), 3, f"{track}/{skill_id}")
                self.assertLessEqual(len(record["patterns"]), 6, f"{track}/{skill_id}")

    def test_packs_do_not_share_ids_with_the_dsa_pack(self):
        for track, pack in TRACK_PACKS.items():
            self.assertEqual(set(pack) & set(TECHNICAL_SKILLS), set(), track)

    def test_expected_track_skill_ids_exist(self):
        expected = {
            "backend": "http-semantics",
            "frontend": "react-state-and-rendering",
            "system_design": "kubernetes-orchestration",
            "ai_stack": "prompt-design",
        }
        for track, skill_id in expected.items():
            self.assertIn(skill_id, TRACK_PACKS[track])


class TrackSelectionTests(unittest.TestCase):
    def test_normalise_track_accepts_aliases_and_rejects_unknown(self):
        self.assertEqual(normalise_track("System-Design"), "system_design")
        self.assertEqual(normalise_track(" ai_stack "), "ai_stack")
        self.assertEqual(normalise_track("AI Stack"), "ai_stack")
        self.assertEqual(normalise_track("react"), "")
        self.assertEqual(normalise_track(""), "")
        self.assertEqual(normalise_track(None), "")

    def test_track_pack_is_used_for_every_level(self):
        for track, pack in TRACK_PACKS.items():
            for level in ("junior", "mid", "senior"):
                selected = skills_for_session(
                    level=level, session_type="coding", track=track
                )["technical"]
                self.assertTrue(selected)
                for skill_id in selected:
                    self.assertIn(skill_id, pack, f"{track}/{level}/{skill_id}")

    def test_frontend_track_yields_frontend_skills_not_dsa(self):
        selected = skills_for_session(
            level="mid", session_type="general", track="frontend"
        )["technical"]
        self.assertIn("react-state-and-rendering", selected)
        self.assertNotIn("arrays-two-pointer", selected)
        block = technical_skill_block(selected, "frontend")
        self.assertIn("React State & Rendering", block)
        self.assertNotIn("Arrays & Two-Pointer Technique", block)

    def test_system_design_track_yields_orchestration_skills(self):
        selected = skills_for_session(
            level="senior", session_type="system-design", track="system_design"
        )["technical"]
        self.assertIn("kubernetes-orchestration", selected)
        selected_mid = skills_for_session(
            level="mid", session_type="general", track="system_design"
        )["technical"]
        self.assertIn("kubernetes-orchestration", selected_mid)
        self.assertNotIn("dynamic-programming", selected + selected_mid)

    def test_empty_and_unknown_track_keep_the_dsa_pack(self):
        for level in ("junior", "mid", "senior"):
            base = skills_for_session(level=level, session_type="general")
            for track in ("", "unknown-track", None, "algorithms"):
                self.assertEqual(
                    skills_for_session(level=level, session_type="general", track=track),
                    base,
                    f"level={level} track={track!r}",
                )

    def test_empty_track_block_is_byte_identical_to_the_old_selection(self):
        for level in ("junior", "mid", "senior"):
            for session_type in ("coding", "general", "system-design"):
                self.assertEqual(
                    skill_context_block(level=level, session_type=session_type),
                    skill_context_block(
                        level=level, session_type=session_type, track=""
                    ),
                )

    def test_track_changes_the_technical_block_only(self):
        dsa = skill_context_block(level="mid", session_type="coding")
        frontend = skill_context_block(
            level="mid", session_type="coding", track="frontend"
        )
        self.assertNotEqual(dsa, frontend)
        interview = "Interview strategies to model and probe"
        educational = "Educational techniques to apply"
        self.assertIn(interview, frontend)
        self.assertIn(educational, frontend)
        self.assertIn(interview, dsa)
        self.assertIn(educational, dsa)

    def test_technical_block_falls_back_to_dsa_for_an_unknown_track(self):
        selected = skills_for_session(
            level="mid", session_type="general", track="frontend"
        )["technical"]
        self.assertEqual(
            technical_skill_block(selected, "not-a-track"),
            technical_skill_block(selected, ""),
        )


class QuestionPromptTrackTests(unittest.TestCase):
    def test_question_prompt_uses_the_topic_track(self):
        prompt = _build_prompt(
            topic_id="06-frontend-core-architecture",
            topic_title="Frontend Architecture",
            doc_content="Rendering is driven by state.",
            count=1,
            level="mid",
            track="frontend",
        )
        self.assertIn("React State & Rendering", prompt)
        self.assertNotIn("Arrays & Two-Pointer Technique", prompt)

    def test_question_prompt_without_a_track_keeps_dsa(self):
        prompt = _build_prompt(
            topic_id="06-frontend-core-architecture",
            topic_title="Frontend Architecture",
            doc_content="Rendering is driven by state.",
            count=1,
            level="junior",
        )
        self.assertIn("Arrays & Two-Pointer Technique", prompt)
        self.assertNotIn("React State & Rendering", prompt)

    def test_problem_solving_ignores_its_backend_catalogue_track(self):
        kwargs = dict(
            topic_id="00-problem-solving-and-algorithms",
            topic_title="Problem Solving and Algorithms",
            doc_content="Hash maps and sliding windows.",
            count=1,
            level="mid",
            preferred_language="python",
            requires_programming=True,
        )
        baseline = _build_prompt(**kwargs)
        tracked = _build_prompt(**kwargs, track="backend")
        self.assertIn("Stacks & Monotonic Variants", baseline)
        self.assertEqual(baseline, tracked)

    def test_quiz_prompt_uses_the_dominant_topic_track(self):
        prompt = _build_quiz_prompt(
            [
                {"id": "t1", "title": "Caching", "content": "x", "track": "system_design"},
                {"id": "t2", "title": "Queues", "content": "x", "track": "backend"},
            ],
            count=2,
            level="senior",
        )
        self.assertIn("Kubernetes Orchestration", prompt)
        self.assertNotIn("Bit Manipulation", prompt)

    def test_quiz_prompt_without_a_track_keeps_dsa(self):
        prompt = _build_quiz_prompt(
            [{"id": "t1", "title": "Caching", "content": "x"}],
            count=2,
            level="junior",
        )
        self.assertIn("Arrays & Two-Pointer Technique", prompt)
        self.assertNotIn("Load Estimation", prompt)

    def test_card_prompt_uses_the_requested_level_and_track(self):
        prompt = _build_card_prompt(
            topic_id="13-ai-stack-llm-and-prompting",
            topic_title="AI Stack",
            doc_content="Prompt design closes the model's degrees of freedom.",
            count=1,
            level="junior",
            track="ai_stack",
        )
        self.assertIn("Prompt Design", prompt)
        self.assertNotIn("Recursion & Divide-and-Conquer", prompt)

    def test_card_prompt_defaults_are_unchanged(self):
        prompt = _build_card_prompt(
            topic_id="13-ai-stack-llm-and-prompting",
            topic_title="AI Stack",
            doc_content="Prompt design closes the model's degrees of freedom.",
            count=1,
        )
        # The previous hardcoded mid/general section.
        self.assertIn("Stacks & Monotonic Variants", prompt)
        self.assertNotIn("Prompt Design", prompt)

    def test_generate_cards_threads_the_topic_track(self):
        items = [
            {
                "front": "What is Caching Strategies?",
                "back": "A cache removes load from the origin with an eviction policy.",
                "source_quote": "A cache removes load from the origin.",
                "source_section": "Caching",
            }
        ]
        llm = _FakeLLMClient(items)
        generator = QuestionGenerator(llm)
        asyncio.run(
            generator.generate_cards(
                topic_id="10-system-design-foundations",
                topic_title="System Design Foundations",
                doc_content="A cache removes load from the origin.",
                count=1,
                level="mid",
                topic_track="system_design",
            )
        )
        self.assertTrue(llm.prompts)
        self.assertIn("Caching Strategies", llm.prompts[0])
        self.assertNotIn("Arrays & Two-Pointer Technique", llm.prompts[0])


class InterviewTrackTests(unittest.TestCase):
    def test_interview_selection_uses_the_session_track(self):
        session = {
            "session_id": "s",
            "track": "ai_stack",
            "level": "mid",
            "interview_type": "coding",
            "turns_completed": 1,
        }
        from app.services.interview_generator import InterviewGenerator

        prompt = InterviewGenerator._question_prompt(session, [])
        # The interview block still probes coding interview skills.
        self.assertIn("Clarify Before Coding", prompt)
        self.assertIn("track: ai_stack", prompt)

    def test_interview_prompt_does_not_leak_the_dsa_pack(self):
        session = {
            "session_id": "s",
            "track": "frontend",
            "level": "senior",
            "interview_type": "coding",
            "turns_completed": 1,
        }
        from app.services.interview_generator import InterviewGenerator

        selected = skills_for_session(
            level=session["level"],
            session_type=InterviewGenerator._session_type_for_skills(session),
            track=session["track"],
            include_educational=False,
        )
        self.assertIn("react-state-and-rendering", selected["technical"])
        self.assertNotIn("arrays-two-pointer", selected["technical"])
        prompt = InterviewGenerator._question_prompt(session, [])
        self.assertNotIn("Arrays & Two-Pointer Technique", prompt)
