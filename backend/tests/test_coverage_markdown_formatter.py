"""Coverage for ``app.services.markdown_formatter``.

The formatter runs over LLM output before it reaches the UI, so the tests pin
the two behaviours that matter: code fences are preserved byte-for-byte, and
prose is collapsed/reflowed without destroying headings, tables or lists.
"""

import unittest

from app.services.markdown_formatter import format_markdown_readable


class EmptyInputTests(unittest.TestCase):
    def test_empty_content_returns_empty_string(self):
        self.assertEqual(format_markdown_readable(""), "")
        self.assertEqual(format_markdown_readable(None), "")
        self.assertEqual(format_markdown_readable("   \n\n  "), "")


class StructurePreservationTests(unittest.TestCase):
    def test_headings_tables_and_bullets_survive_verbatim(self):
        raw = (
            "# Title\n"
            "\n"
            "| a | b |\n"
            "| --- | --- |\n"
            "\n"
            "- first item\n"
            "* second item\n"
            "1. numbered item\n"
        )
        out = format_markdown_readable(raw)
        for line in ("# Title", "| a | b |", "| --- | --- |", "- first item", "* second item", "1. numbered item"):
            self.assertIn(line, out)

    def test_numbered_list_with_trailing_space_is_recognised(self):
        out = format_markdown_readable("intro\n\n1.  spaced\n")
        self.assertEqual(out.splitlines()[-1], "1.  spaced")
        self.assertEqual(out.splitlines()[0], "intro")

    def test_prose_lines_are_joined_into_one_paragraph(self):
        out = format_markdown_readable("first line\nsecond line\nthird line")
        self.assertEqual(out, "first line second line third line")

    def test_repeated_blank_lines_collapse_to_one(self):
        out = format_markdown_readable("alpha\n\n\n\n\nbeta")
        self.assertEqual(out, "alpha\n\nbeta")

    def test_whitespace_is_normalised_inside_a_paragraph(self):
        out = format_markdown_readable("  lots   of\t\tspace  \n   here  ")
        self.assertEqual(out, "lots of space here")

    def test_leading_and_trailing_blank_lines_are_trimmed(self):
        out = format_markdown_readable("\n\n\n# Heading\n\nbody\n\n\n")
        self.assertEqual(out, "# Heading\n\nbody")


class CodeFenceTests(unittest.TestCase):
    def test_fence_only_content_is_preserved(self):
        raw = "```python\nprint(1)\n```"
        self.assertEqual(format_markdown_readable(raw), raw)

    def test_prose_before_and_after_a_fence_is_formatted(self):
        raw = "intro   text\n\n```py\nx = 1\n```\n\ntrailing   text"
        out = format_markdown_readable(raw)
        self.assertIn("intro text", out)
        self.assertIn("```py\nx = 1\n```", out)
        self.assertIn("trailing text", out)
        self.assertNotIn("   text", out)

    def test_markdown_inside_a_fence_is_not_reflowed(self):
        raw = "```md\n-   a\n\n\n#  not a heading\n```"
        self.assertEqual(format_markdown_readable(raw), raw)

    def test_fence_at_start_of_content_has_no_preceding_segment(self):
        raw = "```\ncode\n```\n\ntail"
        out = format_markdown_readable(raw)
        self.assertTrue(out.startswith("```\ncode\n```"))
        self.assertTrue(out.endswith("tail"))

    def test_multiple_fences_are_all_preserved(self):
        raw = "a\n\n```\n1\n```\n\nb\n\n```\n2\n```\n\nc"
        out = format_markdown_readable(raw)
        self.assertEqual(out.count("```"), 4)
        self.assertIn("\n1\n", out)
        self.assertIn("\n2\n", out)

    def test_unterminated_fence_is_treated_as_prose(self):
        # With no closing fence there is nothing to protect, so the lines are
        # joined as an ordinary paragraph.
        out = format_markdown_readable("intro\n\n```py\nx = 1")
        self.assertEqual(out, "intro\n\n```py x = 1")


class LongParagraphTests(unittest.TestCase):
    def test_short_paragraph_is_returned_as_is(self):
        out = format_markdown_readable("Short and sweet.")
        self.assertEqual(out, "Short and sweet.")

    def test_long_paragraph_without_sentence_boundaries_is_kept_whole(self):
        text = "word " * 100  # >360 chars, no ". " boundary at all
        out = format_markdown_readable(text)
        self.assertNotIn("\n", out)
        self.assertEqual(len(out.split()), 100)

    def test_long_paragraph_is_split_on_sentence_boundaries(self):
        sentence = "This is a deliberately long sentence that carries real content. "
        text = sentence * 8
        out = format_markdown_readable(text)
        self.assertIn("\n\n", out)
        for chunk in out.split("\n\n"):
            self.assertLessEqual(len(chunk), 400)
        self.assertTrue(out.startswith("This is a deliberately long"))
        self.assertTrue(out.endswith("."))

    def test_exclamation_and_question_marks_are_also_boundaries(self):
        text = ("Boundary! " * 40) + ("Other boundary? " * 40)
        out = format_markdown_readable(text)
        self.assertIn("\n\n", out)
        self.assertTrue(out.endswith("?"))

    def test_short_final_sentence_is_kept_on_the_last_chunk(self):
        text = ("Filler sentence for length padding here. " * 12) + "Done."
        out = format_markdown_readable(text)
        self.assertTrue(out.endswith("Done."))
        self.assertIn("\n\n", out)

    def test_long_paragraph_next_to_a_heading(self):
        text = "word " * 100
        out = format_markdown_readable(f"# Title\n\n{text}")
        self.assertTrue(out.startswith("# Title\n\n"))
        self.assertNotIn("\n\n\n", out)


if __name__ == "__main__":
    unittest.main()