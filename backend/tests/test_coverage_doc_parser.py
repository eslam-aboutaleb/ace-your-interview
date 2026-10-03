"""Coverage for app/services/doc_parser.py.

Complements ``tests/test_doc_parser.py`` (markdown handbook happy path) with
the curriculum-JSON loader, the path resolvers, and every frontmatter /
section-extraction helper branch.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import doc_parser as dp
from app.services.doc_parser import DocParser


def _write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


class PathResolutionTests(unittest.TestCase):
    def test_absolute_markdown_path_is_returned_verbatim(self):
        target = str(Path(tempfile.gettempdir()) / "handbook")
        self.assertEqual(DocParser._resolve_markdown_path(target), target)

    def test_relative_markdown_path_resolves_against_the_cwd_when_present(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            os.chdir(td)
            try:
                os.makedirs("docs", exist_ok=True)
                resolved = DocParser._resolve_markdown_path("docs")
            finally:
                os.chdir(cwd)
            self.assertEqual(resolved, str((Path(td) / "docs").resolve()))
            self.assertTrue(os.path.isdir(resolved))

    def test_relative_markdown_path_falls_back_to_the_backend_root(self):
        resolved = DocParser._resolve_markdown_path("definitely/not/here/anywhere")
        backend_root = Path(dp.__file__).resolve().parents[2]
        self.assertEqual(resolved, str((backend_root / "definitely/not/here/anywhere").resolve()))
        self.assertFalse(os.path.isdir(resolved))

    def test_absolute_curriculum_path_is_returned_verbatim(self):
        target = str(Path(tempfile.gettempdir()) / "curriculum.json")
        self.assertEqual(DocParser._resolve_curriculum_path(target), target)

    def test_relative_curriculum_file_resolves_against_the_cwd_when_it_is_a_file(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            os.chdir(td)
            try:
                _write(os.path.join(td, "curriculum.json"), '{"topics": []}')
                resolved = DocParser._resolve_curriculum_path("curriculum.json")
            finally:
                os.chdir(cwd)
            self.assertEqual(resolved, str((Path(td) / "curriculum.json").resolve()))

    def test_relative_curriculum_path_falls_back_to_the_backend_root_when_not_a_file(self):
        resolved = DocParser._resolve_curriculum_path("a-directory-not-a-file")
        backend_root = Path(dp.__file__).resolve().parents[2]
        self.assertEqual(resolved, str((backend_root / "a-directory-not-a-file").resolve()))


class CurriculumJsonTests(unittest.TestCase):
    def _parser(self, payload) -> DocParser:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = _write(os.path.join(td.name, "curriculum.json"), json.dumps(payload))
        return DocParser(curriculum_path=path)

    def test_topics_are_normalised_and_summarised(self):
        parser = self._parser(
            {
                "topics": [
                    {
                        "id": "  01-backend  ",
                        "title": "  Backend  ",
                        "description": "  HTTP and APIs.  ",
                        "track": "Backend",
                        "levels": ["Junior", "junior", "wizard", "senior"],
                        "sections": [
                            {"heading": " Goal ", "content": " Learn it. "},
                            {"heading": "", "content": "dropped"},
                            {"heading": "No content"},
                            "not-a-dict",
                        ],
                    }
                ]
            }
        )
        self.assertEqual(parser.source_kind, "curriculum_json")

        topic = parser.get_topic("01-backend")
        self.assertIsNotNone(topic)
        self.assertEqual(topic.title, "Backend")
        self.assertEqual(topic.description, "HTTP and APIs.")
        self.assertEqual(topic.track, "backend")
        # Duplicates collapse and unknown levels drop; three valid remain.
        self.assertEqual(topic.levels, ["junior", "senior"])
        self.assertEqual([s["heading"] for s in topic.sections], ["Goal"])
        # raw_content is rendered from the title/description/sections.
        self.assertIn("# Backend", topic.raw_content)
        self.assertIn("## Goal", topic.raw_content)
        self.assertEqual(parser.get_topic_content("01-backend"), topic.raw_content)

        summary = parser.list_topics()[0]
        self.assertEqual(summary.id, "01-backend")
        self.assertEqual(summary.section_count, 1)
        self.assertEqual(summary.estimated_questions, 3)  # max(3, 1 * 2)

    def test_missing_file_yields_no_topics(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        parser = DocParser(curriculum_path=os.path.join(td.name, "absent.json"))
        self.assertEqual(parser.list_topics(), [])
        self.assertIsNone(parser.get_topic("anything"))
        self.assertEqual(parser.get_topic_content("anything"), "")

    def test_non_list_topics_key_is_ignored(self):
        parser = self._parser({"topics": {"id": "x"}})
        self.assertEqual(parser.list_topics(), [])

    def test_malformed_topics_list_items_are_skipped(self):
        parser = self._parser(
            {
                "topics": [
                    "not-a-dict",
                    {"id": "", "title": "No id"},
                    {"id": "no-title"},
                    {"title": "No id either"},
                    {"id": " ok ", "title": " Kept ", "raw_content": "  custom raw  "},
                ]
            }
        )
        topics = parser.list_topics()
        self.assertEqual([t.id for t in topics], ["ok"])
        # An explicit raw_content wins over the rendered form.
        self.assertEqual(parser.get_topic_content("ok"), "custom raw")

    def test_track_is_inferred_when_absent_or_invalid(self):
        parser = self._parser(
            {
                "topics": [
                    {"id": "10-system_design-foundations", "title": "Foundations"},
                    {"id": "13-ai_stack-llm-and-prompting", "title": "LLM and Prompting"},
                    {"id": "06-frontend-core", "title": "Core Architecture"},
                    {"id": "99-mystery", "title": "Mystery"},
                    {"id": "bad-track", "title": "Bad Track", "track": "kubernetes"},
                ]
            }
        )
        tracks = {t.id: t.track for t in parser.list_topics()}
        self.assertEqual(tracks["10-system_design-foundations"], "system_design")
        self.assertEqual(tracks["13-ai_stack-llm-and-prompting"], "ai_stack")
        self.assertEqual(tracks["06-frontend-core"], "frontend")
        self.assertEqual(tracks["99-mystery"], "backend")
        self.assertEqual(tracks["bad-track"], "backend")

    def test_levels_default_to_the_full_ladder(self):
        parser = self._parser(
            {
                "topics": [
                    {"id": "a", "title": "A", "levels": None},
                    {"id": "b", "title": "B", "levels": "senior"},
                    {"id": "c", "title": "C", "levels": ["nope"]},
                    {"id": "d", "title": "D", "levels": 5},
                ]
            }
        )
        self.assertEqual(parser.get_topic("a").levels, ["junior", "mid", "senior"])
        self.assertEqual(parser.get_topic("b").levels, ["senior"])
        self.assertEqual(parser.get_topic("c").levels, ["junior", "mid", "senior"])
        self.assertEqual(parser.get_topic("d").levels, ["junior", "mid", "senior"])

    def test_list_topics_is_sorted_by_id(self):
        parser = self._parser(
            {"topics": [{"id": "03-c", "title": "C"}, {"id": "01-a", "title": "A"}, {"id": "02-b", "title": "B"}]}
        )
        self.assertEqual([t.id for t in parser.list_topics()], ["01-a", "02-b", "03-c"])

    def test_track_level_and_query_filters_are_all_applied(self):
        parser = self._parser(
            {
                "topics": [
                    {
                        "id": "01-backend",
                        "title": "Backend Contracts",
                        "description": "APIs and validations.",
                        "track": "backend",
                        "levels": ["junior", "mid"],
                    },
                    {
                        "id": "06-frontend",
                        "title": "Frontend Performance",
                        "description": "Rendering.",
                        "track": "frontend",
                        "levels": ["senior"],
                    },
                ]
            }
        )
        self.assertEqual([t.id for t in parser.list_topics(track="backend")], ["01-backend"])
        self.assertEqual([t.id for t in parser.list_topics(level="senior")], ["06-frontend"])
        self.assertEqual([t.id for t in parser.list_topics(q="contracts")], ["01-backend"])
        # Filters compose.
        self.assertEqual(
            [t.id for t in parser.list_topics(track="frontend", level="senior", q="rendering")],
            ["06-frontend"],
        )
        self.assertEqual(parser.list_topics(track="system_design"), [])
        self.assertEqual(parser.list_topics(q="nothing-matches"), [])
        # An unknown filter value normalises to "", which disables that filter.
        self.assertEqual(len(parser.list_topics(level="wizard")), 2)
        # An unknown filter value normalises to "" and so matches everything.
        self.assertEqual(len(parser.list_topics(track="nonsense")), 2)
        self.assertEqual(len(parser.list_topics(level="nonsense")), 2)


class MarkdownDirectoryTests(unittest.TestCase):
    def test_directory_without_a_handbook_subdir_is_read_directly(self):
        with tempfile.TemporaryDirectory() as td:
            _write(os.path.join(td, "01-direct.md"), "# Direct\n\nBody text.\n")
            _write(os.path.join(td, "notes.txt"), "ignored")
            _write(os.path.join(td, "README.md"), "# Readme\n\nShould be skipped.\n")

            parser = DocParser(docs_path=td)
            ids = [t.id for t in parser.list_topics()]
            self.assertEqual(ids, ["01-direct"])
            self.assertEqual(parser.source_kind, "markdown_override")
            # An absolute docs_path is recorded verbatim, not resolved.
            self.assertEqual(parser.source_path, td)

    def test_missing_directory_yields_no_topics(self):
        with tempfile.TemporaryDirectory() as td:
            parser = DocParser(docs_path=os.path.join(td, "nope"))
            self.assertEqual(parser.list_topics(), [])

    def test_raw_content_is_the_body_after_frontmatter(self):
        with tempfile.TemporaryDirectory() as td:
            _write(
                os.path.join(td, "01-x.md"),
                "---\ntrack: backend\n---\n# Title\n\nBody.\n\n## Sec\n\nMore.\n",
            )
            parser = DocParser(docs_path=td)
            raw = parser.get_topic_content("01-x")
            self.assertNotIn("track: backend", raw)
            self.assertTrue(raw.startswith("# Title"))
            topic = parser.get_topic("01-x")
            self.assertEqual(topic.raw_content, raw)
            self.assertEqual(topic.track, "backend")


class FrontmatterTests(unittest.TestCase):
    def test_document_without_frontmatter_is_returned_unchanged(self):
        metadata, body = DocParser._extract_frontmatter("# Title\n\nBody.\n")
        self.assertEqual(metadata, {})
        self.assertEqual(body, "# Title\n\nBody.\n")

    def test_unterminated_frontmatter_is_not_parsed(self):
        text = "---\ntrack: backend\n# Title\n"
        metadata, body = DocParser._extract_frontmatter(text)
        self.assertEqual(metadata, {})
        self.assertEqual(body, text)

    def test_inline_list_and_comma_separated_levels(self):
        metadata, _ = DocParser._extract_frontmatter(
            "---\nlevels: [junior, mid, senior]\n---\nbody\n"
        )
        self.assertEqual(metadata["levels"], ["junior", "mid", "senior"])

        metadata, _ = DocParser._extract_frontmatter("---\nlevels: junior, mid\n---\nbody\n")
        self.assertEqual(metadata["levels"], ["junior", "mid"])

    def test_block_list_levels_terminate_on_the_next_key(self):
        text = "---\nlevels:\n  - junior\n  - mid\ntrack: ai_stack\n---\nbody\n"
        metadata, _ = DocParser._extract_frontmatter(text)
        self.assertEqual(metadata["levels"], ["junior", "mid"])
        self.assertEqual(metadata["track"], "ai_stack")

    def test_comments_blank_lines_and_colonless_lines_are_ignored(self):
        text = "---\n# a comment\n\nno-colon-here\ntrack: backend\n---\nbody\n"
        metadata, _ = DocParser._extract_frontmatter(text)
        self.assertEqual(metadata, {"track": "backend"})

    def test_keys_are_lowercased_and_values_trimmed(self):
        metadata, _ = DocParser._extract_frontmatter("---\nTRACK:   Backend  \n---\nbody\n")
        self.assertEqual(metadata["track"], "Backend")

    def test_absent_levels_leaves_the_key_out(self):
        metadata, _ = DocParser._extract_frontmatter("---\ntrack: backend\n---\nbody\n")
        self.assertNotIn("levels", metadata)


class ExtractionHelperTests(unittest.TestCase):
    def test_title_prefers_the_first_h1_then_falls_back_to_the_filename(self):
        self.assertEqual(DocParser._extract_title("intro\n# Real Title\n# Second\n", "f.md"), "Real Title")
        self.assertEqual(DocParser._extract_title("no heading here", "some-file-name.md"), "Some File Name")

    def test_description_stops_at_the_second_heading(self):
        content = "# T\n\nFirst line.\nSecond line.\n\n## Section\n\nNot the description.\n"
        self.assertEqual(DocParser._extract_description(content), "First line. Second line.")

    def test_description_stops_at_a_heading_with_no_blank_line_between(self):
        content = "# T\nFirst line.\n## Section\nBody.\n"
        self.assertEqual(DocParser._extract_description(content), "First line.")

    def test_consecutive_headings_capture_the_text_after_the_last_one(self):
        content = "# T\n\n## Section\n\nSection description.\n"
        self.assertEqual(DocParser._extract_description(content), "Section description.")

    def test_description_is_empty_without_a_heading(self):
        self.assertEqual(DocParser._extract_description("just prose\n"), "")

    def test_description_is_capped_at_300_characters(self):
        content = "# T\n\n" + ("word " * 200)
        self.assertEqual(len(DocParser._extract_description(content)), 300)

    def test_description_stops_at_the_first_blank_line_after_text(self):
        self.assertEqual(DocParser._extract_description("# T\n\npara one\n\npara two\n"), "para one")

    def test_sections_split_on_h2_and_h3(self):
        content = "# T\n\nintro\n\n## One\n\nbody one\n\n### Nested\n\nnested body\n\n## Two\n\nbody two\n"
        sections = DocParser._extract_sections(content)
        self.assertEqual([s["heading"] for s in sections], ["One", "Nested", "Two"])
        self.assertEqual(sections[0]["content"], "body one")
        self.assertEqual(sections[1]["content"], "nested body")

    def test_document_without_sections_gets_an_overview(self):
        sections = DocParser._extract_sections("# T\n\nonly a preamble\n")
        self.assertEqual(sections, [{"heading": "Overview", "content": "# T\n\nonly a preamble"}])

    def test_empty_document_yields_no_sections(self):
        self.assertEqual(DocParser._extract_sections("   \n"), [])

    def test_section_content_is_capped_at_12000_characters(self):
        sections = DocParser._extract_sections(f"# T\n\n## Big\n\n{'x' * 13000}\n")
        self.assertEqual(len(sections[0]["content"]), 12000)

    def test_render_raw_content_round_trips_the_parts(self):
        rendered = DocParser._render_raw_content(
            title=" Title ",
            description=" Desc ",
            sections=[{"heading": " H ", "content": " C "}],
        )
        self.assertEqual(rendered, "# Title\n\nDesc\n\n## H\n\nC")

    def test_normalise_sections_requires_heading_and_content(self):
        self.assertEqual(
            DocParser._normalise_sections(
                [
                    {"heading": " H ", "content": " C "},
                    {"heading": "  ", "content": "dropped"},
                    {"heading": "NoContent", "content": "   "},
                    42,
                ]
            ),
            [{"heading": "H", "content": "C"}],
        )
        self.assertEqual(DocParser._normalise_sections("not-a-list"), [])
        self.assertEqual(DocParser._normalise_sections(None), [])

    def test_track_and_level_normalisation(self):
        self.assertEqual(DocParser._normalise_track("System Design"), "system_design")
        self.assertEqual(DocParser._normalise_track(" ai-stack "), "ai_stack")
        self.assertEqual(DocParser._normalise_track("platform"), "")
        self.assertEqual(DocParser._normalise_track(None), "")
        self.assertEqual(DocParser._normalise_level(" Senior "), "senior")
        self.assertEqual(DocParser._normalise_level("staff"), "")
        self.assertEqual(DocParser._normalise_level(""), "")

    def test_infer_track_keyword_order(self):
        # os.devnull is not a regular file, so the loader is a no-op.
        infer = DocParser(curriculum_path=os.devnull)._infer_track
        self.assertEqual(infer("x", "System Design"), "system_design")
        self.assertEqual(infer("system_design-thing", "y"), "system_design")
        # A hyphenated id alone is not a keyword match; the fallback applies.
        self.assertEqual(infer("10-system-design-foundations", "Foundations"), "backend")
        self.assertEqual(infer("x", "Building RAG Pipelines"), "ai_stack")
        self.assertEqual(infer("x", "Agent Orchestration"), "ai_stack")
        self.assertEqual(infer("x", "LLM Serving"), "ai_stack")
        self.assertEqual(infer("x", "Frontend Rendering"), "frontend")
        self.assertEqual(infer("x", "UI Composition"), "frontend")
        self.assertEqual(infer("x", "Backend Persistence"), "backend")
        self.assertEqual(infer("x", "HTTP Caching"), "backend")
        # Fallback for anything unrecognised.
        self.assertEqual(infer("zzz", "Nothing Matches"), "backend")

    def test_normalise_levels_dedupes_and_falls_back(self):
        parser = DocParser(curriculum_path=os.devnull)
        self.assertEqual(parser._normalise_levels(["mid", "MID", "senior"]), ["mid", "senior"])
        self.assertEqual(parser._normalise_levels("junior"), ["junior"])
        self.assertEqual(parser._normalise_levels(["nope"]), ["junior", "mid", "senior"])
        self.assertEqual(parser._normalise_levels(None), ["junior", "mid", "senior"])


class SettingsWiringTests(unittest.TestCase):
    def test_settings_curriculum_path_is_used_when_no_argument_is_given(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write(
                os.path.join(td, "curriculum.json"),
                json.dumps({"topics": [{"id": "x", "title": "X"}]}),
            )
            with patch.object(dp, "get_settings") as settings:
                settings.return_value.curriculum_path = path
                parser = DocParser()
            self.assertEqual([t.id for t in parser.list_topics()], ["x"])
            self.assertEqual(parser.source_path, path)

    def test_explicit_argument_beats_the_configured_path(self):
        with tempfile.TemporaryDirectory() as td:
            configured = _write(
                os.path.join(td, "configured.json"),
                json.dumps({"topics": [{"id": "from-settings", "title": "S"}]}),
            )
            explicit = _write(
                os.path.join(td, "explicit.json"),
                json.dumps({"topics": [{"id": "from-argument", "title": "A"}]}),
            )
            with patch.object(dp, "get_settings") as settings:
                settings.return_value.curriculum_path = configured
                parser = DocParser(curriculum_path=explicit)
            self.assertEqual([t.id for t in parser.list_topics()], ["from-argument"])

    def test_invalid_json_raises_rather_than_silently_loading_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write(os.path.join(td, "curriculum.json"), "{not json")
            with self.assertRaises(json.JSONDecodeError):
                DocParser(curriculum_path=path)


if __name__ == "__main__":
    unittest.main()
