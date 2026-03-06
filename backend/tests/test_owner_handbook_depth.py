import unittest

from app.services.doc_parser import DocParser


class OwnerHandbookDepthTests(unittest.TestCase):
    def test_static_curriculum_depth_thresholds(self):
        parser = DocParser()
        topics = parser.list_topics()
        self.assertEqual(len(topics), 22)

        section_counts = [topic.section_count for topic in topics]
        self.assertGreaterEqual(sum(section_counts), 1000)
        self.assertTrue(all(count >= 40 for count in section_counts))


if __name__ == "__main__":
    unittest.main()
