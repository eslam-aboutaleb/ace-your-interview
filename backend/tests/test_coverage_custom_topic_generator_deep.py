"""Coverage for app/services/custom_topic_generator.py.

Complements ``tests/test_custom_topic_generator.py`` (valid/invalid payloads
through the public API) with the deterministic layer beneath it: breadth
estimation, track/level normalisation, generic-heading rejection, the coverage
dimension enforcement (including its out-of-range fallbacks), the stream batch
fallback filler, and the ``missing done`` degradation path.

Every test drives either a pure helper or an injected fake LLM, so nothing here
can open a socket.
"""

import asyncio
import json
import unittest

from app.services import custom_topic_generator as ctg
from app.services.custom_topic_generator import CustomTopicGenerator
from app.services.llm_client import BUDGET_EXCEEDED_CODE


class ScriptedLLM:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0
        self.options: list[dict] = []
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        self.options.append({"task": task, **kwargs})
        if self._results:
            return dict(self._results.pop(0))
        return {
            "success": False,
            "analysis": "",
            "metadata": {},
            "error": "no_response",
            "error_code": "llm_call_failed",
        }


class AlwaysFailLLM:
    """A terminal LLM outcome for every call, however many batches ask."""

    def __init__(self, error_code):
        self.error_code = error_code
        self.calls = 0
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        return {
            "success": False,
            "analysis": "",
            "metadata": {},
            "error": "boom",
            "error_code": self.error_code,
        }


def _ok(payload) -> dict:
    return {
        "success": True,
        "analysis": payload if isinstance(payload, str) else json.dumps(payload),
        "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
        "error": "",
        "error_code": "",
    }


async def _collect(generator):
    events = []
    async for event in generator:
        events.append(event)
    return events


class EstimateTargetSectionsTests(unittest.TestCase):
    def test_a_medium_length_topic_takes_the_four_and_five_word_credit(self):
        # 5 words clears the >=4 band but not the >=6 band.
        self.assertEqual(ctg._estimate_target_sections("Kafka consumer group basics"), 120)

    def test_a_three_word_topic_takes_no_word_count_credit(self):
        # 3 words lands between the >=4 and <=2 bands.
        self.assertEqual(ctg._estimate_target_sections("Kafka stream basics"), 105)

    def test_a_two_word_narrow_topic_is_penalised(self):
        self.assertEqual(ctg._estimate_target_sections("Kafka basics"), 90)

    def test_a_long_topic_takes_the_six_word_credit_and_the_ai_stack_bonus(self):
        wide = ctg._estimate_target_sections(
            "RAG prompt engineering retrieval fundamentals and evaluation"
        )
        # ai_stack track adds +10 on top of the >=6 word bonus.
        self.assertEqual(ctg._infer_track("RAG prompt engineering retrieval"), "ai_stack")
        self.assertGreater(wide, ctg._estimate_target_sections("RAG prompt engineering"))

    def test_a_system_design_topic_takes_the_broadness_bonus(self):
        self.assertEqual(ctg._infer_track("service reliability"), "system_design")
        # Same two words, but the system-design track adds +20 to the score.
        self.assertEqual(ctg._estimate_target_sections("service reliability"), 135)
        self.assertEqual(ctg._estimate_target_sections("kafka partitions"), 105)

    def test_every_estimate_is_clamped_to_the_declared_bounds(self):
        for topic in ("", "overview", "architecture patterns " * 40):
            with self.subTest(topic=topic[:24]):
                estimate = ctg._estimate_target_sections(topic)
                self.assertGreaterEqual(estimate, ctg._MIN_SECTIONS)
                self.assertLessEqual(estimate, ctg._MAX_SECTIONS)


class ClampAndTrackTests(unittest.TestCase):
    def test_the_target_section_count_is_clamped_to_the_bounds(self):
        self.assertEqual(ctg._clamp_target_sections(1), ctg._MIN_SECTIONS)
        self.assertEqual(ctg._clamp_target_sections(10_000), ctg._MAX_SECTIONS)
        self.assertEqual(ctg._clamp_target_sections(120), 120)

    def test_track_inference_covers_every_declared_track_and_the_default(self):
        cases = {
            "distributed consensus": "system_design",
            "scaling a queue": "system_design",
            "service reliability": "system_design",
            "react rendering": "frontend",
            "css layout": "frontend",
            "prompt engineering": "ai_stack",
            "rag pipelines": "ai_stack",
            "machine learning serving": "ai_stack",
            "kafka partitions": "backend",
            "": "backend",
        }
        for topic, expected in cases.items():
            with self.subTest(topic=topic):
                self.assertEqual(ctg._infer_track(topic), expected)

    def test_an_explicit_track_string_is_validated_against_the_known_set(self):
        self.assertEqual(ctg._normalise_track(" System-Design "), "system_design")
        self.assertEqual(ctg._normalise_track("AI Stack"), "ai_stack")
        self.assertEqual(ctg._normalise_track("platform"), "")
        self.assertEqual(ctg._normalise_track(""), "")

    def test_levels_accept_a_list_a_comma_string_or_fall_back_to_the_default(self):
        self.assertEqual(
            ctg._normalise_levels(["Junior", "mid", "senior"]),
            ["junior", "mid", "senior"],
        )
        self.assertEqual(
            ctg._normalise_levels("junior, mid , senior"), ["junior", "mid", "senior"]
        )
        self.assertEqual(ctg._normalise_levels(["junior", "junior"]), ["junior"])
        self.assertEqual(ctg._normalise_levels("expert"), list(ctg._DEFAULT_LEVELS))
        self.assertEqual(ctg._normalise_levels(None), list(ctg._DEFAULT_LEVELS))

    def test_the_level_for_position_bands_a_total_into_thirds(self):
        self.assertEqual(ctg._level_for_position(0, 12), "junior")
        self.assertEqual(ctg._level_for_position(3, 12), "junior")
        self.assertEqual(ctg._level_for_position(4, 12), "mid")
        self.assertEqual(ctg._level_for_position(11, 12), "senior")
        self.assertEqual(ctg._level_for_position(5, 0), "junior")

    def test_level_rank_orders_the_bands(self):
        self.assertEqual(
            [ctg._level_rank(level) for level in ("junior", "mid", "senior", "unknown")],
            [0, 1, 2, 2],
        )

    def test_a_heading_that_is_already_prefixed_is_not_prefixed_twice(self):
        self.assertEqual(ctg._format_heading("mid", "Hash Maps"), "Mid: Hash Maps")
        self.assertEqual(ctg._format_heading("mid", "Mid: Hash Maps"), "Mid: Hash Maps")
        self.assertEqual(ctg._format_heading("mid", "   "), "")


class HeadingIsGenericTests(unittest.TestCase):
    def test_an_empty_short_or_single_word_heading_is_generic(self):
        for heading in ("", "   ", "Intro", "Advanced"):
            with self.subTest(heading=heading):
                self.assertTrue(ctg._heading_is_generic(heading))

    def test_an_anchored_placeholder_pattern_is_generic(self):
        # The patterns are fully anchored, so only a bare numbered placeholder
        # (not "Module 3 architecture in practice") is rejected here.
        for heading in ("Chapter 12", "Module 3", "Topic 2", "Part i", "Section 4"):
            with self.subTest(heading=heading):
                self.assertTrue(ctg._heading_is_generic(heading))

    def test_a_specific_heading_is_not_generic(self):
        for heading in (
            "Consistent hashing and virtual node placement",
            "Introduction and overview of the design",
            "Module 3 architecture in practice",
        ):
            with self.subTest(heading=heading):
                self.assertFalse(ctg._heading_is_generic(heading))


class NormaliseSectionsTests(unittest.TestCase):
    def test_an_empty_input_is_autofilled_to_the_target(self):
        out = ctg._normalise_sections([], "Kafka consumer groups", 55)
        self.assertEqual(len(out), 55)
        self.assertTrue(out[0]["heading"].startswith("Junior:"))
        self.assertIn("Kafka consumer groups", " ".join(s["heading"] for s in out))
        # The synthetic marker never reaches a caller.
        self.assertEqual(ctg._SYNTHETIC_KEY, "_synthetic")
        for section in out:
            self.assertNotIn(ctg._SYNTHETIC_KEY, section)

    def test_junk_items_are_dropped_and_duplicates_collapse(self):
        out = ctg._normalise_sections(
            [
                "not-a-dict",
                {"heading": "Too Short", "content": "tiny"},
                {"heading": "Consumer group rebalancing", "content": "How the group rebalances."},
                {"heading": "consumer group rebalancing", "content": "Duplicate heading."},
                {"heading": "Overview", "content": "A generic heading should be dropped here."},
                {"heading": "ISR shrinking and recovery", "content": "How the ISR shrinks."},
            ],
            "Kafka",
            50,
        )
        headings = [s["heading"].lower() for s in out]
        self.assertEqual(headings.count("junior: consumer group rebalancing"), 1)
        self.assertNotIn("junior: overview", headings)
        self.assertEqual(len(out), 50)

    def test_an_explicit_level_is_honoured_and_the_order_is_ranked_by_level(self):
        out = ctg._normalise_sections(
            [
                {
                    "heading": "Senior scaling topic",
                    "content": "Scaling detail here.",
                    "level": "senior",
                },
                {
                    "heading": "Junior basics topic",
                    "content": "Basics detail here.",
                    "level": "junior",
                },
            ],
            "Kafka",
            50,
        )
        self.assertTrue(out[0]["heading"].startswith("Junior:"))
        self.assertEqual(len(out), 50)

    def test_a_short_batch_is_topped_up_with_deterministic_filler(self):
        provided = [
            {"heading": f"Kafka topic number {i}", "content": f"Body for topic {i}."}
            for i in range(40)
        ]
        out = ctg._normalise_sections(provided, "Kafka", 50)
        self.assertEqual(len(out), 50)
        self.assertEqual(out[0]["heading"], "Junior: Kafka topic number 0")
        # With no explicit level the band comes from the input position.
        self.assertEqual(out[39]["heading"], "Senior: Kafka topic number 39")
        self.assertNotEqual(out[40]["heading"], out[39]["heading"])

    def test_a_non_list_input_still_produces_a_full_roadmap(self):
        out = ctg._normalise_sections("not-a-list", "Kafka", 50)
        self.assertEqual(len(out), 50)
        self.assertTrue(out[0]["heading"].startswith("Junior:"))

    def test_a_full_batch_is_not_topped_up(self):
        provided = [
            {"heading": f"Kafka topic number {i}", "content": f"Body for topic {i}."}
            for i in range(60)
        ]
        out = ctg._normalise_sections(provided, "Kafka", 50)
        self.assertEqual(len(out), 50)

    def test_filler_that_would_repeat_an_existing_heading_is_skipped(self):
        # Seed the last provided heading with the exact filler heading the
        # autofill pass would produce next, so it collides and is skipped.
        collision = ctg._autofill_section("Kafka", 5, 50)["heading"]
        provided = [
            {
                "heading": collision if index == 4 else f"Kafka topic number {index}",
                "content": f"Body for topic {index}.",
            }
            for index in range(5)
        ]
        out = ctg._normalise_sections(provided, "Kafka", 50)
        self.assertEqual(
            [h for h in (s["heading"] for s in out) if h.lower() == collision.lower()],
            [collision],
            "the colliding filler heading must not be emitted twice",
        )


class EnforceCoverageDimensionsTests(unittest.TestCase):
    def test_an_empty_section_list_is_returned_unchanged(self):
        self.assertEqual(ctg._enforce_coverage_dimensions([], "Kafka", 50), [])

    def test_an_already_complete_roadmap_is_left_alone(self):
        sections = ctg._normalise_sections([], "Distributed system design scaling", 60)
        self.assertEqual(ctg._enforce_coverage_dimensions(sections, "Kafka", 60), sections)

    def test_a_narrow_roadmap_has_its_missing_dimensions_injected(self):
        narrow = [
            {"heading": f"Junior: Consistent hashing detail {i}", "content": "Detail body here."}
            for i in range(50)
        ]
        out = ctg._enforce_coverage_dimensions(narrow, "Kafka", 50)
        self.assertEqual(len(out), 50)
        headings = " ".join(s["heading"] for s in out)
        self.assertIn("Security", headings)
        self.assertIn("Operations", headings)

    def test_more_missing_dimensions_than_slots_reuses_slots_without_duplicating(self):
        narrow = [
            {"heading": f"Junior: Consumer group detail {i}", "content": "Detail body here."}
            for i in range(50)
        ]
        out = ctg._enforce_coverage_dimensions(narrow, "Kafka", 50)
        keys = [s["heading"] for s in out]
        self.assertEqual(len(keys), len(set(keys)), "replacements must not duplicate a heading")
        self.assertEqual(len(out), 50)

    def test_a_duplicate_replacement_heading_is_deduped_with_a_suffix(self):
        # Synthetic sections earn no coverage credit, so a synthetic placeholder
        # can hold the exact heading an injected dimension wants. The suffix
        # loop must then pick a fresh heading rather than duplicate one.
        collision = ctg._build_dimension_section(
            "Kafka", level="senior", dimension="security"
        )["heading"]
        narrow = [
            {
                "heading": collision,
                "content": "Placeholder that earns no coverage credit.",
                ctg._SYNTHETIC_KEY: True,
            },
            {
                # Covers every dimension except security.
                "heading": "Junior: Consumer group rebalancing",
                "content": (
                    "core concepts setup api debug test latency resilience "
                    "architecture operations"
                ),
            },
        ]
        out = ctg._enforce_coverage_dimensions(narrow, "Kafka", 2)
        headings = [s["heading"] for s in out]
        self.assertEqual(len(headings), len(set(headings)))
        self.assertIn(f"{collision} 2", headings)

    def test_exhausting_the_band_then_the_whole_list_stops_safely(self):
        # Two slots cannot hold ten missing dimensions: the junior band holds
        # one, the whole-list fallback holds the second, and the rest stop.
        narrow = [
            {"heading": "Junior: Consumer group rebalancing", "content": "Detail body here."},
            {"heading": "Junior: ISR shrink and recovery", "content": "Detail body here."},
        ]
        out = ctg._enforce_coverage_dimensions(narrow, "Kafka", 2)
        self.assertEqual(len(out), 2)
        headings = [s["heading"] for s in out]
        self.assertEqual(len(headings), len(set(headings)))
        # Both slots ended up holding real dimension content.
        self.assertIn("Junior: Kafka: Foundations and Core Terminology", headings)
        self.assertIn("Junior: Kafka: Workflow, Tooling, and Delivery", headings)


class FallbackTopicDetailTests(unittest.TestCase):
    def test_the_fallback_detail_is_a_complete_deterministic_roadmap(self):
        detail = ctg._fallback_topic_detail("Rust ownership", "custom-rust-ownership", 52)
        self.assertEqual(detail.id, "custom-rust-ownership")
        self.assertEqual(detail.title, "Rust Ownership Interview Roadmap")
        self.assertEqual(detail.track, "backend")
        self.assertEqual(detail.levels, list(ctg._DEFAULT_LEVELS))
        self.assertEqual(len(detail.sections), 52)
        self.assertIn("## Junior:", detail.raw_content)
        self.assertIn("## Senior:", detail.raw_content)
        self.assertTrue(detail.description)

    def test_a_system_design_topic_gets_the_system_design_track(self):
        detail = ctg._fallback_topic_detail("distributed reliability", "custom-dr", 50)
        self.assertEqual(detail.track, "system_design")


class NormaliseStreamBatchSectionsTests(unittest.TestCase):
    def test_a_non_list_payload_consumes_nothing(self):
        out, consumed = ctg._normalise_stream_batch_sections(
            sections={"heading": "x"},
            topic="Kafka",
            target_sections=50,
            start_index=0,
            batch_size=5,
            seen_headings=set(),
        )
        self.assertEqual(out, [])
        self.assertEqual(consumed, set())

    def test_junk_items_are_skipped_and_good_ones_consumed(self):
        out, consumed = ctg._normalise_stream_batch_sections(
            sections=[
                "not-a-dict",
                {"heading": "Tiny", "content": "no"},
                {"heading": "Consumer group rebalancing", "content": "How the group rebalances."},
            ],
            topic="Kafka",
            target_sections=50,
            start_index=0,
            batch_size=5,
            seen_headings=set(),
        )
        self.assertEqual([s["heading"] for s in out], ["Junior: Consumer group rebalancing"])
        self.assertEqual(consumed, {"junior: consumer group rebalancing"})

    def test_an_already_seen_heading_is_not_consumed_again(self):
        out, consumed = ctg._normalise_stream_batch_sections(
            sections=[{"heading": "Consumer group rebalancing", "content": "How it rebalances."}],
            topic="Kafka",
            target_sections=50,
            start_index=0,
            batch_size=5,
            seen_headings={"junior: consumer group rebalancing"},
        )
        self.assertEqual(out, [])
        self.assertEqual(consumed, set())

    def test_the_batch_size_caps_the_number_of_accepted_sections(self):
        out, _ = ctg._normalise_stream_batch_sections(
            sections=[
                {"heading": f"Kafka detail topic {i}", "content": f"Body number {i}."}
                for i in range(10)
            ],
            topic="Kafka",
            target_sections=50,
            start_index=0,
            batch_size=3,
            seen_headings=set(),
        )
        self.assertEqual(len(out), 3)


class FallbackStreamBatchSectionsTests(unittest.TestCase):
    def test_the_filler_is_marked_synthetic_and_numbered(self):
        out, consumed = ctg._fallback_stream_batch_sections(
            topic="Kafka",
            target_sections=50,
            start_index=0,
            batch_size=4,
            seen_headings=set(),
        )
        self.assertEqual(len(out), 4)
        self.assertEqual(len(consumed), 4)
        for section in out:
            # The marker is set here and stripped by the final normalise.
            self.assertTrue(section[ctg._SYNTHETIC_KEY])
            self.assertIn("#", section["heading"])

    def test_a_collision_with_an_already_seen_heading_gets_a_numeric_suffix(self):
        first, _ = ctg._fallback_stream_batch_sections(
            topic="Kafka", target_sections=50, start_index=0, batch_size=1, seen_headings=set()
        )
        seen = {first[0]["heading"].lower()}
        second, _ = ctg._fallback_stream_batch_sections(
            topic="Kafka", target_sections=50, start_index=0, batch_size=1, seen_headings=seen
        )
        self.assertNotEqual(second[0]["heading"].lower(), first[0]["heading"].lower())
        self.assertIn("#1.2", second[0]["heading"])


class GenerateTopicStreamTests(unittest.TestCase):
    def test_a_fully_successful_stream_commits_the_batch_metadata(self):
        batch_size = ctg._stream_batch_size(50)
        payload = {
            "title": "Kafka Interview Roadmap",
            "description": "Deep Kafka roadmap.",
            "track": "backend",
            "levels": ["junior", "mid", "senior"],
            "sections": [
                {"heading": f"Kafka topic number {i}", "content": f"Detailed body number {i}."}
                for i in range(batch_size)
            ],
        }
        llm = ScriptedLLM([_ok(payload)])
        events = asyncio.run(
            _collect(
                CustomTopicGenerator(llm).generate_topic_stream(topic="Kafka", target_sections=50)
            )
        )
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["topic"].title, "Kafka Interview Roadmap")
        self.assertEqual(done["topic"].description, "Deep Kafka roadmap.")
        self.assertEqual(len(done["topic"].sections), 50)
        self.assertEqual(llm.options[0]["task"], "final")
        self.assertTrue(llm.options[0]["structured"])

    def test_a_blank_topic_is_replaced_by_a_generic_label(self):
        events = asyncio.run(
            _collect(
                CustomTopicGenerator(ScriptedLLM([])).generate_topic_stream(
                    topic="   ", target_sections=50
                )
            )
        )
        self.assertEqual(events[-1]["topic"].id, "custom-custom-topic")

    def test_a_generic_heading_survives_the_stream_but_not_the_direct_builder(self):
        # The stream batch normaliser runs its generic-heading check on the
        # already level-prefixed heading ("Junior: Overview"), which the
        # anchored patterns cannot match, so the placeholder reaches the caller.
        batch_size = ctg._stream_batch_size(50)
        sections = [{"heading": "Overview", "content": "A broad placeholder section."}]
        sections += [
            {"heading": f"Kafka topic number {i}", "content": f"Detailed body number {i}."}
            for i in range(batch_size - 1)
        ]
        events = asyncio.run(
            _collect(
                CustomTopicGenerator(ScriptedLLM([_ok({"sections": sections})])).generate_topic_stream(
                    topic="Kafka", target_sections=50
                )
            )
        )
        headings = [s["heading"] for s in events[-1]["topic"].sections]
        self.assertIn("Junior: Overview", headings)

        # The same heading is dropped when it reaches the non-stream builder
        # unprefixed, which is what the router's direct-payload path relies on.
        direct = ctg._normalise_sections(
            [{"heading": "Overview", "content": "A broad placeholder section."}], "Kafka", 50
        )
        self.assertNotIn("Junior: Overview", [s["heading"] for s in direct])

    def test_an_explicit_out_of_range_target_is_clamped(self):
        events = asyncio.run(
            _collect(
                CustomTopicGenerator(ScriptedLLM([])).generate_topic_stream(
                    topic="Kafka", target_sections=5
                )
            )
        )
        self.assertEqual(len(events[-1]["topic"].sections), ctg._MIN_SECTIONS)

    def test_a_terminal_budget_rejection_is_not_retried_and_falls_back(self):
        batch_size = ctg._stream_batch_size(50)
        batches = -(-50 // batch_size)
        llm = AlwaysFailLLM(BUDGET_EXCEEDED_CODE)
        events = asyncio.run(
            _collect(
                CustomTopicGenerator(llm).generate_topic_stream(topic="Kafka", target_sections=50)
            )
        )
        # Exactly one call per batch: re-sending the identical prompt at the
        # identical cap cannot clear a budget rejection.
        self.assertEqual(llm.calls, batches)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(len(events[-1]["topic"].sections), 50)
        self.assertNotIn("_synthetic", events[-1]["topic"].sections[0])

    def test_a_retryable_failure_is_retried_up_to_the_attempt_cap(self):
        llm = AlwaysFailLLM("llm_call_failed")
        asyncio.run(
            _collect(
                CustomTopicGenerator(llm).generate_topic_stream(topic="Kafka", target_sections=50)
            )
        )
        batch_size = ctg._stream_batch_size(50)
        batches = -(-50 // batch_size)
        self.assertEqual(llm.calls, batches * ctg._MAX_ATTEMPTS)

    def test_a_rejected_attempt_does_not_poison_the_next_batch(self):
        batch_size = ctg._stream_batch_size(50)
        good = [
            {"heading": f"Kafka topic number {i}", "content": f"Detailed body number {i}."}
            for i in range(batch_size)
        ]
        # First attempt is short by one section; the second completes the batch.
        llm = ScriptedLLM(
            [
                _ok({"sections": good[:-1]}),
                _ok(
                    {
                        "title": "Kafka Interview Roadmap",
                        "description": "Deep Kafka roadmap.",
                        "track": "backend",
                        "levels": ["junior", "mid", "senior"],
                        "sections": good,
                    }
                ),
            ]
        )
        events = asyncio.run(
            _collect(
                CustomTopicGenerator(llm).generate_topic_stream(topic="Kafka", target_sections=50)
            )
        )
        sections = events[-1]["topic"].sections
        self.assertEqual(len(sections), 50)
        headings = [s["heading"] for s in sections]
        self.assertEqual(len(headings), len(set(headings)))

    def test_external_context_reaches_the_batch_prompt(self):
        class MCP:
            async def gather_context(self, **kwargs):
                self.kwargs = kwargs
                return "The team runs Kafka with Kafka Streams."

        mcp = MCP()
        llm = AlwaysFailLLM(BUDGET_EXCEEDED_CODE)
        asyncio.run(
            _collect(
                CustomTopicGenerator(llm, mcp).generate_topic_stream(
                    topic="Kafka", target_sections=50
                )
            )
        )
        self.assertEqual(mcp.kwargs["flow"], "custom_topic")
        self.assertIn("Kafka Streams", llm.prompts[0])

    def test_generate_topic_strips_and_defaults_its_topic_argument(self):
        generator = CustomTopicGenerator(AlwaysFailLLM(BUDGET_EXCEEDED_CODE))
        detail = asyncio.run(generator.generate_topic(topic="  Kafka  "))
        self.assertEqual(detail.id, "custom-kafka")
        # No explicit target means the breadth estimator decides the length.
        self.assertGreaterEqual(len(detail.sections), ctg._MIN_SECTIONS)
        blank = asyncio.run(generator.generate_topic(topic="   "))
        self.assertEqual(blank.id, "custom-custom-topic")

    def test_generate_topic_clamps_an_out_of_range_target(self):
        generator = CustomTopicGenerator(AlwaysFailLLM(BUDGET_EXCEEDED_CODE))
        detail = asyncio.run(
            generator.generate_topic(topic="Kafka", target_sections=10_000)
        )
        self.assertEqual(len(detail.sections), ctg._MAX_SECTIONS)


class SupportedOptionsTests(unittest.TestCase):
    def test_a_var_keyword_client_receives_every_option(self):
        class _Wide:
            async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
                return {}

        options = {"task": "final", "system": "s", "structured": True}
        self.assertEqual(ctg._supported_options(_Wide(), options), options)

    def test_a_narrow_client_only_receives_declared_options(self):
        class _Narrow:
            async def completion(self, prompt, llm_config=None, user_identity=None, task=None):
                return {}

        self.assertEqual(
            ctg._supported_options(_Narrow(), {"task": "final", "structured": True}),
            {"task": "final"},
        )

    def test_an_unintrospectable_client_gets_everything(self):
        class _Opaque:
            completion = object()

        self.assertEqual(ctg._supported_options(_Opaque(), {"task": "final"}), {"task": "final"})


class StreamBatchSizeTests(unittest.TestCase):
    def test_the_batch_size_is_bounded_at_both_ends(self):
        self.assertEqual(ctg._stream_batch_size(0), 8)
        self.assertEqual(ctg._stream_batch_size(50), 8)
        self.assertEqual(ctg._stream_batch_size(100), 10)
        self.assertEqual(ctg._stream_batch_size(1000), 20)