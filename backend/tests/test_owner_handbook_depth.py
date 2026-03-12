import json
import unittest
from pathlib import Path

from app.services.doc_parser import DocParser


class StaticCurriculumDepthTests(unittest.TestCase):
    def test_static_curriculum_depth_thresholds(self):
        parser = DocParser()
        topics = parser.list_topics()
        self.assertEqual(parser.source_kind, "curriculum_json")

        section_counts = [topic.section_count for topic in topics]
        self.assertEqual(len(topics), 22)
        self.assertGreaterEqual(sum(section_counts), 1000)
        self.assertTrue(all(count >= 40 for count in section_counts))

    def test_static_curriculum_catalog_matches_parser(self):
        parser = DocParser()
        topics = parser.list_topics()
        topics_by_id = {topic.id: parser.get_topic(topic.id) for topic in topics}

        catalog_path = (
            Path(__file__).resolve().parents[1] / "app" / "data" / "static_curriculum.json"
        )
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog_topics = catalog.get("topics", [])
        catalog_ids = {str(item.get("id", "")).strip() for item in catalog_topics if isinstance(item, dict)}

        self.assertEqual(set(topics_by_id.keys()), catalog_ids)
        self.assertTrue(str(parser.source_path).endswith("app/data/static_curriculum.json"))

        for topic_id, topic_detail in topics_by_id.items():
            self.assertIsNotNone(topic_detail, f"Missing topic detail for {topic_id}")
            assert topic_detail is not None
            self.assertTrue(topic_detail.description.strip())
            self.assertGreaterEqual(len(topic_detail.sections), 40)
            self.assertTrue(topic_detail.raw_content.strip())
            self.assertTrue(
                any(section["content"].strip() for section in topic_detail.sections),
                f"Topic {topic_id} has empty sections",
            )


if __name__ == "__main__":
    unittest.main()
