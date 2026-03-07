import unittest
import json
from pathlib import Path

from app.services.doc_parser import DocParser


class OwnerHandbookDepthTests(unittest.TestCase):
    def test_static_curriculum_depth_thresholds(self):
        parser = DocParser()
        topics = parser.list_topics()
        self.assertEqual(len(topics), 22)

        section_counts = [topic.section_count for topic in topics]
        self.assertGreaterEqual(sum(section_counts), 1000)
        self.assertTrue(all(count >= 40 for count in section_counts))

    def test_static_curriculum_has_manifest_cluster_coverage_per_topic(self):
        parser = DocParser()
        topics = parser.list_topics()
        topics_by_id = {topic.id: parser.get_topic(topic.id) for topic in topics}

        manifest_path = Path(__file__).resolve().parents[1] / "docs" / "owner-handbook" / "coverage_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_topics = manifest.get("topics", {})

        self.assertEqual(set(topics_by_id.keys()), set(manifest_topics.keys()))

        missing_clusters: list[str] = []
        for topic_id, item in manifest_topics.items():
            topic_detail = topics_by_id.get(topic_id)
            self.assertIsNotNone(topic_detail, f"Missing topic detail for {topic_id}")
            corpus = (topic_detail.raw_content if topic_detail else "").lower()
            for cluster in item.get("coverage_clusters", []):
                if str(cluster).lower() not in corpus:
                    missing_clusters.append(f"{topic_id}:{cluster}")

        self.assertEqual(
            missing_clusters,
            [],
            f"Coverage gaps found in static curriculum: {', '.join(missing_clusters)}",
        )


if __name__ == "__main__":
    unittest.main()
