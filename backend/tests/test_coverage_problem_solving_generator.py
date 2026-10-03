"""Coverage for app/services/problem_solving_generator.py.

Complements ``tests/test_problem_solving_generator.py`` (stream contract and
retry behaviour) with the pure helpers: language normalisation, level banding,
heading formatting, the deterministic fallback builders, and the
``_supported_options`` narrowing shim.
"""

import asyncio
import json
import unittest

from app.services import problem_solving_generator as psg
from app.services.problem_solving_generator import ProblemSolvingGenerator
from app.services.topic_catalog import (
    PROBLEM_SOLVING_DEFAULT_LANGUAGE,
    PROBLEM_SOLVING_LANGUAGE_OPTIONS,
    PROBLEM_SOLVING_TARGET_SECTIONS,
)


class SupportedOptionsTests(unittest.TestCase):
    def test_var_keyword_clients_receive_every_option(self):
        class _Wide:
            async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
                return {}

        options = {"task": "final", "system": "s", "structured": True, "max_tokens_cap": 1500}
        self.assertEqual(psg._supported_options(_Wide(), options), options)

    def test_narrow_clients_only_receive_options_they_declare(self):
        class _Narrow:
            async def completion(self, prompt, llm_config=None, user_identity=None, task=None):
                return {}

        self.assertEqual(
            psg._supported_options(_Narrow(), {"task": "final", "system": "s", "structured": True}),
            {"task": "final"},
        )

    def test_a_client_without_an_introspectable_signature_gets_everything(self):
        class _Opaque:
            completion = object()

        options = {"task": "final", "structured": True}
        self.assertEqual(psg._supported_options(_Opaque(), options), options)

    def test_a_narrow_client_still_produces_a_usable_generation(self):
        class _NarrowLLM:
            def __init__(self):
                self.calls = 0

            async def completion(self, prompt, llm_config=None, user_identity=None, task=None):
                self.calls += 1
                return {"success": False, "analysis": "", "metadata": {}, "error": "", "error_code": "llm_call_failed"}

        llm = _NarrowLLM()
        generator = ProblemSolvingGenerator(llm)
        topic = asyncio.run(generator.generate_topic(preferred_language="java"))
        self.assertEqual(len(topic.sections), PROBLEM_SOLVING_TARGET_SECTIONS)
        self.assertEqual(llm.calls, (PROBLEM_SOLVING_TARGET_SECTIONS // psg._STREAM_BATCH_SIZE) * psg._MAX_ATTEMPTS)


class LanguageNormalisationTests(unittest.TestCase):
    def test_supported_languages_pass_through_case_and_whitespace_normalised(self):
        for language in PROBLEM_SOLVING_LANGUAGE_OPTIONS:
            with self.subTest(language=language):
                self.assertEqual(
                    psg._normalise_language(f"  {language.upper()}  "), language.lower()
                )

    def test_unknown_and_empty_languages_fall_back_to_the_default(self):
        self.assertEqual(psg._normalise_language("cobol"), PROBLEM_SOLVING_DEFAULT_LANGUAGE)
        self.assertEqual(psg._normalise_language(""), PROBLEM_SOLVING_DEFAULT_LANGUAGE)
        self.assertEqual(psg._normalise_language(None), PROBLEM_SOLVING_DEFAULT_LANGUAGE)


class LevelAndHeadingTests(unittest.TestCase):
    def test_level_for_index_bands_into_thirds(self):
        total = 12
        self.assertEqual(psg._level_for_index(0, total), "junior")
        self.assertEqual(psg._level_for_index(3, total), "junior")
        self.assertEqual(psg._level_for_index(4, total), "mid")
        self.assertEqual(psg._level_for_index(7, total), "mid")
        self.assertEqual(psg._level_for_index(8, total), "senior")
        self.assertEqual(psg._level_for_index(11, total), "senior")

    def test_a_non_positive_total_pins_everything_to_junior(self):
        self.assertEqual(psg._level_for_index(0, 0), "junior")
        self.assertEqual(psg._level_for_index(99, -1), "junior")

    def test_tiny_totals_still_produce_all_three_bands(self):
        # one_third floors at 1 and two_third is forced above it, so a 3-item
        # target still yields junior, mid, senior in order.
        self.assertEqual(
            [psg._level_for_index(i, 3) for i in range(3)], ["junior", "mid", "senior"]
        )

    def test_format_heading_prefixes_the_level_once(self):
        self.assertEqual(psg._format_heading("mid", "Hash Maps"), "Mid: Hash Maps")
        self.assertEqual(psg._format_heading("senior", "senior: already prefixed"), "senior: already prefixed")
        self.assertEqual(psg._format_heading("junior", "MID: mixed case"), "MID: mixed case")
        self.assertEqual(psg._format_heading("mid", "   "), "Mid: ")

    def test_format_heading_truncates_long_headings(self):
        self.assertEqual(len(psg._format_heading("mid", "x" * 400)), len("Mid: ") + 150)
        self.assertEqual(len(psg._format_heading("mid", "Junior: " + "x" * 400)), 160)


class NormaliseSectionsTests(unittest.TestCase):
    def test_a_complete_payload_is_returned_verbatim(self):
        sections = [
            {"heading": f"H{i}", "content": "c" * 120} for i in range(3)
        ]
        out = psg._normalise_sections(sections, target_sections=3, language="python")
        self.assertEqual(len(out), 3)
        self.assertEqual([s["heading"] for s in out], ["Junior: H0", "Mid: H1", "Senior: H2"])
        self.assertEqual(psg._level_for_index(2, 3), "senior")

    def test_non_list_input_falls_straight_to_the_fallback(self):
        out = psg._normalise_sections("not-a-list", target_sections=3, language="python")
        self.assertEqual(len(out), 3)
        self.assertTrue(all("python" in s["heading"] for s in out))

    def test_short_content_duplicates_and_junk_items_are_dropped(self):
        out = psg._normalise_sections(
            [
                "not-a-dict",
                {"heading": "Too Short", "content": "tiny"},
                {"heading": "Kept", "content": "c" * 120},
                {"heading": "kept", "content": "c" * 120},
                {"heading": "  ", "content": "c" * 120},
            ],
            target_sections=2,
            language="python",
        )
        headings = [s["heading"] for s in out]
        # The level comes from the *input* position, so the two dropped items
        # still consume bands and "Kept" (input index 2) lands in the senior band.
        self.assertEqual(headings[0], "Senior: Kept")
        # The duplicate collapses, so the remaining slot is fallback filler.
        self.assertEqual(len(out), 2)
        self.assertNotIn("Senior: Kept", headings[1:])

    def test_the_target_is_a_hard_cap(self):
        out = psg._normalise_sections(
            [{"heading": f"H{i}", "content": "c" * 120} for i in range(20)],
            target_sections=5,
            language="python",
        )
        self.assertEqual(len(out), 5)

    def test_content_is_capped_at_3000_characters(self):
        out = psg._normalise_sections(
            [{"heading": "Long", "content": "c" * 5000}], target_sections=1, language="python"
        )
        self.assertEqual(len(out[0]["content"]), 3000)


class FallbackSectionTests(unittest.TestCase):
    def test_fallback_fills_to_the_target_with_level_prefixed_headings(self):
        sections = psg._fallback_sections("go", 30)
        self.assertEqual(len(sections), 30)
        self.assertEqual(len({s["heading"].lower() for s in sections}), 30)
        self.assertTrue(all("go" in s["heading"] for s in sections))
        self.assertEqual(len({s["content"] for s in sections}) > 0, True)
        for index, section in enumerate(sections):
            self.assertTrue(
                section["heading"].startswith(psg._level_for_index(index, 30).title())
            )

    def test_fallback_preserves_a_valid_seed(self):
        seed = [{"heading": "Junior: Kept", "content": "real content"}]
        sections = psg._fallback_sections("rust", 4, seed=seed)
        self.assertEqual(sections[0]["heading"], "Junior: Kept")
        self.assertEqual(sections[0]["content"], "real content")
        self.assertEqual(len(sections), 4)

    def test_a_seed_that_already_covers_the_target_is_returned_unchanged(self):
        seed = psg._fallback_sections("python", 3)
        self.assertEqual(psg._fallback_sections("python", 3, seed=seed), seed)

    def test_a_zero_target_returns_nothing(self):
        self.assertEqual(psg._fallback_sections("python", 0), [])


class NormaliseBatchTests(unittest.TestCase):
    def test_consumed_headings_are_reported_not_committed(self):
        seen = {"junior: existing"}
        out, consumed = psg._normalise_batch_sections(
            sections=[{"heading": "New", "content": "c" * 120}],
            target_sections=120,
            language="python",
            start_index=0,
            batch_size=4,
            seen_headings=seen,
        )
        self.assertEqual([s["heading"] for s in out], ["Junior: New"])
        self.assertEqual(consumed, {"junior: new"})
        self.assertNotIn("junior: new", seen)

    def test_non_list_input_returns_an_empty_pair(self):
        out, consumed = psg._normalise_batch_sections(
            sections={"heading": "x"},
            target_sections=120,
            language="python",
            start_index=0,
            batch_size=4,
            seen_headings=set(),
        )
        self.assertEqual(out, [])
        self.assertEqual(consumed, set())

    def test_junk_and_short_items_are_skipped_without_consuming_the_batch(self):
        out, consumed = psg._normalise_batch_sections(
            sections=[
                "not-a-dict",
                {"heading": "Too Short", "content": "tiny"},
                {"heading": "  ", "content": "tiny"},
                {"heading": "Kept", "content": "c" * 120},
            ],
            target_sections=120,
            language="python",
            start_index=0,
            batch_size=4,
            seen_headings=set(),
        )
        self.assertEqual([s["heading"] for s in out], ["Junior: Kept"])
        self.assertEqual(consumed, {"junior: kept"})

    def test_batch_size_caps_the_output(self):
        out, _ = psg._normalise_batch_sections(
            sections=[{"heading": f"H{i}", "content": "c" * 120} for i in range(10)],
            target_sections=120,
            language="python",
            start_index=0,
            batch_size=3,
            seen_headings=set(),
        )
        self.assertEqual(len(out), 3)

    def test_levels_follow_the_global_index_not_the_batch_index(self):
        out, _ = psg._normalise_batch_sections(
            sections=[{"heading": "A", "content": "c" * 120}, {"heading": "B", "content": "c" * 120}],
            target_sections=120,
            language="python",
            start_index=100,
            batch_size=4,
            seen_headings=set(),
        )
        self.assertEqual([s["heading"].split(":")[0] for s in out], ["Senior", "Senior"])

    def test_headings_already_seen_are_skipped_and_reported_as_not_consumed(self):
        seen = {"junior: dup"}
        out, consumed = psg._normalise_batch_sections(
            sections=[
                {"heading": "dup", "content": "c" * 120},
                {"heading": "fresh", "content": "c" * 120},
            ],
            target_sections=120,
            language="python",
            start_index=0,
            batch_size=4,
            seen_headings=seen,
        )
        self.assertEqual([s["heading"] for s in out], ["Junior: fresh"])
        self.assertEqual(consumed, {"junior: fresh"})


class FallbackBatchTests(unittest.TestCase):
    def test_fallback_batch_never_repeats_a_seen_heading(self):
        seen: set[str] = set()
        first = psg._fallback_batch_sections(
            language="cpp", target_sections=120, start_index=0, batch_size=5, seen_headings=seen
        )
        second = psg._fallback_batch_sections(
            language="cpp", target_sections=120, start_index=5, batch_size=5, seen_headings=seen
        )
        headings = [s["heading"].lower() for s in first + second]
        self.assertEqual(len(headings), 10)
        self.assertEqual(len(set(headings)), 10)
        self.assertEqual(len(seen), 10)

    def test_fallback_batch_uses_the_global_index_for_numbering(self):
        out = psg._fallback_batch_sections(
            language="java",
            target_sections=120,
            start_index=60,
            batch_size=2,
            seen_headings=set(),
        )
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0]["heading"].startswith("Mid: "))
        self.assertTrue(out[0]["heading"].endswith("#61"))
        self.assertTrue(out[1]["heading"].endswith("#62"))
        self.assertTrue(all(s["content"] for s in out))

    def test_a_collision_gets_a_numeric_suffix(self):
        seen: set[str] = set()
        first = psg._fallback_batch_sections(
            language="python",
            target_sections=120,
            start_index=0,
            batch_size=1,
            seen_headings=seen,
        )
        # Re-running the same position must not reuse the heading.
        second = psg._fallback_batch_sections(
            language="python",
            target_sections=120,
            start_index=0,
            batch_size=1,
            seen_headings=seen,
        )
        self.assertNotEqual(first[0]["heading"], second[0]["heading"])
        # Suffix 1 reproduces the original heading, so the loop steps to 2.
        self.assertTrue(second[0]["heading"].endswith("#1 (python) #1.2"))
        self.assertEqual(len(seen), 2)

    def test_fallback_batch_drops_the_level_prefix_from_the_pool(self):
        out = psg._fallback_batch_sections(
            language="python",
            target_sections=120,
            start_index=0,
            batch_size=1,
            seen_headings=set(),
        )
        # Exactly one level prefix, not "Junior: Junior: ...".
        self.assertEqual(out[0]["heading"].count("Junior:"), 1)


class BuildTopicDetailTests(unittest.TestCase):
    def test_raw_content_contains_every_section(self):
        sections = psg._fallback_sections("python", 2)
        rendered = psg._build_raw_content("T", " D ", sections)
        self.assertTrue(rendered.startswith("# T"))
        for section in sections:
            self.assertIn(f"## {section['heading']}", rendered)

    def test_topic_detail_carries_the_problem_solving_metadata(self):
        topic = psg._build_topic_detail(
            language="kotlin",
            title="Problem Solving and Algorithms (kotlin)",
            description="desc",
            sections=psg._fallback_sections("kotlin", 2),
        )
        self.assertEqual(topic.id, "00-problem-solving-and-algorithms")
        self.assertEqual(topic.selected_language, "kotlin")
        self.assertTrue(topic.requires_programming)
        self.assertTrue(topic.is_dynamic_topic)
        self.assertTrue(topic.content_ready)
        self.assertEqual(topic.response_detail, "concise")
        self.assertEqual(topic.language_options, list(PROBLEM_SOLVING_LANGUAGE_OPTIONS))


class StreamShapeTests(unittest.TestCase):
    def test_events_are_progress_then_sections_then_done(self):
        class _AlwaysValid:
            async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
                return {"success": False, "analysis": "", "metadata": {}, "error": "", "error_code": ""}

        generator = ProblemSolvingGenerator(_AlwaysValid())

        async def _collect():
            return [
                event
                async for event in generator.generate_topic_stream(
                    preferred_language="python", target_sections=24
                )
            ]

        events = asyncio.run(_collect())
        progress = [e for e in events if e["type"] == "progress"]
        sections = [e for e in events if e["type"] == "section"]
        done = [e for e in events if e["type"] == "done"]
        batch_count = PROBLEM_SOLVING_TARGET_SECTIONS // psg._STREAM_BATCH_SIZE
        self.assertEqual(len(progress), batch_count)
        self.assertEqual(progress[0]["batch_count"], batch_count)
        self.assertEqual(progress[0]["message"], "Generating sections 1-12 of 120.")
        self.assertEqual(progress[1]["message"], "Generating sections 13-24 of 120.")
        self.assertEqual(progress[-1]["message"], "Generating sections 109-120 of 120.")
        self.assertEqual(len(sections), PROBLEM_SOLVING_TARGET_SECTIONS)
        self.assertEqual([s["index"] for s in sections], list(range(1, 121)))
        self.assertTrue(all(s["total_sections"] == 120 for s in sections))
        self.assertEqual(len(done), 1)
        self.assertEqual(len(done[0]["topic"].sections), PROBLEM_SOLVING_TARGET_SECTIONS)

    def test_a_zero_or_negative_target_is_pinned_to_the_catalog_size(self):
        class _Fail:
            async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
                return {"success": False, "analysis": "", "metadata": {}, "error": "", "error_code": ""}

        generator = ProblemSolvingGenerator(_Fail())
        for target in (0, -5):
            with self.subTest(target=target):
                topic = asyncio.run(
                    generator.generate_topic(
                        preferred_language="python", target_sections=target
                    )
                )
                self.assertEqual(len(topic.sections), PROBLEM_SOLVING_TARGET_SECTIONS)

    def test_a_non_default_target_is_forced_back_to_the_catalog_size(self):
        class _Fail:
            async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
                return {"success": False, "analysis": "", "metadata": {}, "error": "", "error_code": ""}

        generator = ProblemSolvingGenerator(_Fail())
        topic = asyncio.run(
            generator.generate_topic(preferred_language="python", target_sections=12)
        )
        self.assertEqual(len(topic.sections), PROBLEM_SOLVING_TARGET_SECTIONS)

    def test_generate_topic_falls_back_when_the_stream_never_yields_done(self):
        class _NoDone:
            async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
                return {"success": False, "analysis": "", "metadata": {}, "error": "", "error_code": ""}

        generator = ProblemSolvingGenerator(_NoDone())
        original = generator.generate_topic_stream

        async def _empty_stream(**kwargs):
            if False:
                yield {}

        generator.generate_topic_stream = _empty_stream
        try:
            topic = asyncio.run(
                generator.generate_topic(preferred_language="klingon", target_sections=6)
            )
        finally:
            generator.generate_topic_stream = original
        # An unsupported language normalises to the default in the fallback too.
        self.assertEqual(topic.selected_language, PROBLEM_SOLVING_DEFAULT_LANGUAGE)
        self.assertEqual(len(topic.sections), PROBLEM_SOLVING_TARGET_SECTIONS)
        self.assertTrue(topic.title.endswith(f"({PROBLEM_SOLVING_DEFAULT_LANGUAGE})"))

    def test_a_done_event_without_a_topic_is_ignored(self):
        class _Fail:
            async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
                return {"success": False, "analysis": "", "metadata": {}, "error": "", "error_code": ""}

        generator = ProblemSolvingGenerator(_Fail())

        async def _bogus_stream(**kwargs):
            yield {"type": "done", "topic": {"not": "a TopicDetail"}}

        generator.generate_topic_stream = _bogus_stream
        topic = asyncio.run(generator.generate_topic(preferred_language="python"))
        self.assertEqual(len(topic.sections), PROBLEM_SOLVING_TARGET_SECTIONS)
        self.assertEqual(topic.title, "Problem Solving and Algorithms (python)")


class McpContextTests(unittest.TestCase):
    def test_gateway_context_is_requested_once_and_reused_for_every_batch(self):
        class _MCP:
            def __init__(self):
                self.calls = []

            async def gather_context(self, **kwargs):
                self.calls.append(kwargs)
                return "external notes about arrays"

        class _Valid:
            """Returns a fresh batch of headings on every call."""

            def __init__(self):
                self.prompts = []
                self._issued = 0

            async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
                self.prompts.append(prompt)
                headings = [f"H{self._issued + i}" for i in range(psg._STREAM_BATCH_SIZE)]
                self._issued += len(headings)
                payload = {
                    "title": "Accepted",
                    "description": "good",
                    "sections": [{"heading": h, "content": "y" * 120} for h in headings],
                }
                return {"success": True, "analysis": json.dumps(payload), "metadata": {}, "error": ""}

        mcp = _MCP()
        llm = _Valid()
        generator = ProblemSolvingGenerator(llm, mcp)

        topic = asyncio.run(generator.generate_topic(preferred_language="python"))

        batch_count = PROBLEM_SOLVING_TARGET_SECTIONS // psg._STREAM_BATCH_SIZE
        self.assertEqual(len(mcp.calls), 1)
        self.assertEqual(mcp.calls[0]["flow"], "custom_topic")
        self.assertEqual(mcp.calls[0]["topic_id"], "00-problem-solving-and-algorithms")
        # One accepted response per batch: no retries were needed.
        self.assertEqual(len(llm.prompts), batch_count)
        for prompt in llm.prompts:
            self.assertIn('<untrusted_input label="external_context">', prompt)
            self.assertIn("external notes about arrays", prompt)
        self.assertEqual(topic.title, "Accepted")
        self.assertEqual(topic.description, "good")
        self.assertEqual(len(topic.sections), PROBLEM_SOLVING_TARGET_SECTIONS)


class PartialBatchRetentionTests(unittest.TestCase):
    def test_the_longest_valid_partial_batch_is_kept_and_the_rest_is_filled(self):
        batch_count = PROBLEM_SOLVING_TARGET_SECTIONS // psg._STREAM_BATCH_SIZE
        partial_size = 5

        class _PartialLLM:
            def __init__(self):
                self.calls = 0

            async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
                self.calls += 1
                batch_index = self.calls // psg._MAX_ATTEMPTS
                headings = [
                    f"B{batch_index}-{i}" for i in range(partial_size)
                ]
                payload = {
                    "title": "Partial",
                    "description": "partial",
                    "sections": [{"heading": h, "content": "y" * 120} for h in headings],
                }
                return {"success": True, "analysis": json.dumps(payload), "metadata": {}, "error": ""}

        llm = _PartialLLM()
        generator = ProblemSolvingGenerator(llm)
        topic = asyncio.run(generator.generate_topic(preferred_language="python"))

        # Every batch retried to the attempt cap, then topped up deterministically.
        self.assertEqual(llm.calls, batch_count * psg._MAX_ATTEMPTS)
        self.assertEqual(len(topic.sections), PROBLEM_SOLVING_TARGET_SECTIONS)
        headings = [s["heading"] for s in topic.sections]
        self.assertEqual(len(set(headings)), PROBLEM_SOLVING_TARGET_SECTIONS)
        # The first batch keeps the LLM's five accepted sections ahead of filler.
        self.assertTrue(headings[0].startswith("Junior: B0-0"))
        self.assertTrue(headings[partial_size - 1].startswith("Junior: B0-4"))
        self.assertIn("python", headings[partial_size])


if __name__ == "__main__":
    unittest.main()
