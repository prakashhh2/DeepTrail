import unittest

from engine.crawler import crawl
from engine.models import CrawlRequest, FetchResult


class FakeFetcher:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.fetched: list[str] = []

    async def fetch(self, url: str) -> FetchResult:
        self.fetched.append(url)
        html = self.pages.get(url)
        if html is None:
            return FetchResult(url=url, status_code=404, content_type="text/html", text=None, error="HTTP 404")
        return FetchResult(url=url, status_code=200, content_type="text/html", text=html)


class CustomScorer:
    def __init__(self) -> None:
        self.prepared_objective = None
        self.page_calls = 0
        self.link_calls = 0

    def prepare(self, objective: str) -> None:
        self.prepared_objective = objective

    def score_page(self, objective, page) -> float:
        self.page_calls += 1
        return 0.4

    def score_link(self, objective, link) -> float:
        self.link_calls += 1
        return 0.9

    def explain(self, objective, page, score) -> str:
        return "custom scorer"


class CrawlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_crawler_stops_after_max_pages(self):
        pages = {
            "https://example.edu/": """
                <title>Home</title>
                <a href="/scholarships">Scholarships</a>
                <a href="/sports">Sports</a>
            """,
            "https://example.edu/scholarships": "<title>Scholarships</title><p>International undergraduate scholarships.</p>",
            "https://example.edu/sports": "<title>Sports</title><p>Tickets.</p>",
        }
        fetcher = FakeFetcher(pages)

        response = await crawl(
            CrawlRequest(
                objective="international undergraduate scholarships",
                seeds=["https://example.edu"],
                max_pages=2,
                max_depth=2,
                respect_robots_txt=False,
                crawl_delay_seconds=0,
                concurrency=1,
            ),
            fetcher=fetcher,
        )

        self.assertEqual(response.stats.pages_crawled, 2)
        self.assertEqual(len(fetcher.fetched), 2)
        self.assertIn("https://example.edu/scholarships", fetcher.fetched)
        self.assertNotIn("https://example.edu/sports", fetcher.fetched)

    async def test_same_domain_links_are_enforced(self):
        pages = {
            "https://example.edu/": """
                <title>Home</title>
                <a href="https://other.edu/scholarships">Other scholarships</a>
                <a href="/aid">Financial aid</a>
            """,
            "https://example.edu/aid": "<title>Aid</title><p>Scholarship aid.</p>",
        }
        fetcher = FakeFetcher(pages)

        response = await crawl(
            CrawlRequest(
                objective="scholarships",
                seeds=["https://example.edu"],
                max_pages=3,
                max_depth=2,
                same_domain_only=True,
                respect_robots_txt=False,
                crawl_delay_seconds=0,
                concurrency=1,
            ),
            fetcher=fetcher,
        )

        self.assertEqual(response.stats.pages_crawled, 2)
        self.assertNotIn("https://other.edu/scholarships", fetcher.fetched)

    async def test_custom_scorer_is_used_through_dependency_injection(self):
        fetcher = FakeFetcher({
            "https://example.edu/": '<title>Home</title><a href="/next">Next</a>',
            "https://example.edu/next": "<title>Next</title><p>Content</p>",
        })
        scorer = CustomScorer()

        response = await crawl(
            CrawlRequest(
                objective="custom objective",
                seeds=["https://example.edu"],
                max_pages=2,
                max_depth=2,
                respect_robots_txt=False,
                crawl_delay_seconds=0,
                concurrency=1,
            ),
            fetcher=fetcher,
            scorer=scorer,
        )

        self.assertEqual(scorer.prepared_objective, "custom objective")
        self.assertEqual(scorer.page_calls, 2)
        self.assertEqual(scorer.link_calls, 1)
        self.assertTrue(response.results)
        self.assertEqual(response.results[0].reason, "custom scorer")


if __name__ == "__main__":
    unittest.main()
