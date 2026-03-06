import os
import tempfile
import unittest

from app.services.doc_parser import DocParser


class DocParserTests(unittest.TestCase):
    def test_parses_frontmatter_track_and_levels(self):
        with tempfile.TemporaryDirectory() as td:
            handbook = os.path.join(td, "owner-handbook")
            os.makedirs(handbook, exist_ok=True)
            with open(os.path.join(handbook, "01-backend.md"), "w", encoding="utf-8") as f:
                f.write(
                    """---
track: backend
levels: [junior, mid, senior]
---
# Backend Basics

Understand request handling.

## Goal
Learn backend basics.
"""
                )

            parser = DocParser(docs_path=td)
            topics = parser.list_topics()
            self.assertEqual(len(topics), 1)
            self.assertEqual(topics[0].track, "backend")
            self.assertEqual(topics[0].levels, ["junior", "mid", "senior"])

    def test_track_level_and_query_filters(self):
        with tempfile.TemporaryDirectory() as td:
            handbook = os.path.join(td, "owner-handbook")
            os.makedirs(handbook, exist_ok=True)

            with open(os.path.join(handbook, "01-backend.md"), "w", encoding="utf-8") as f:
                f.write(
                    """---
track: backend
levels: [junior, mid]
---
# Backend Contracts

APIs and validations.
"""
                )
            with open(os.path.join(handbook, "02-frontend.md"), "w", encoding="utf-8") as f:
                f.write(
                    """---
track: frontend
levels: [senior]
---
# Frontend Performance

Rendering and profiling.
"""
                )

            parser = DocParser(docs_path=td)

            backend_only = parser.list_topics(track="backend")
            self.assertEqual(len(backend_only), 1)
            self.assertEqual(backend_only[0].id, "01-backend")

            senior_only = parser.list_topics(level="senior")
            self.assertEqual(len(senior_only), 1)
            self.assertEqual(senior_only[0].id, "02-frontend")

            query = parser.list_topics(q="contracts")
            self.assertEqual(len(query), 1)
            self.assertEqual(query[0].id, "01-backend")


if __name__ == "__main__":
    unittest.main()
