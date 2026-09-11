import json
import tempfile
import unittest
from pathlib import Path

from engine.exporter import crawl_response_dict, write_crawl_json
from engine.models import CrawlResponse, CrawlResult, CrawlStats


class ExporterTests(unittest.TestCase):
    def test_export_contains_llm_context_and_rank(self):
        response = CrawlResponse(
            results=[
                CrawlResult(
                    url="https://example.edu/research",
                    title="Robot research",
                    score=0.9,
                    depth=1,
                    reason="matches",
                    text="Robots improve picking accuracy.",
                    headings=["Findings"],
                    word_count=4,
                )
            ],
            stats=CrawlStats(pages_crawled=1),
            objective="warehouse robots",
            seeds=["https://example.edu"],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = write_crawl_json(response, Path(directory) / "crawl.json")
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["objective"], "warehouse robots")
        self.assertEqual(data["pages"][0]["rank"], 1)
        self.assertIn("Robots improve picking accuracy", data["pages"][0]["llm_context"])
        self.assertEqual(crawl_response_dict(response)["schema_version"], "1.0")
