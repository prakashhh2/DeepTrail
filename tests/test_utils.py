import unittest

from engine.frontier import CrawlFrontier
from engine.policies import CrawlPolicy
from engine.models import CrawlRequest
from engine.utils import normalize_url


class URLUtilityTests(unittest.TestCase):
    def test_normalize_url_removes_fragments_default_ports_and_tracking_params(self):
        self.assertEqual(
            normalize_url("HTTPS://Example.com:443/a/../Path/?b=2&utm_source=x&a=1#section"),
            "https://example.com/Path/?a=1&b=2",
        )

    def test_frontier_rejects_queued_duplicates(self):
        frontier = CrawlFrontier()
        self.assertTrue(frontier.add("https://example.com/a", depth=0, score=0.5))
        self.assertFalse(frontier.add("https://example.com/a", depth=1, score=0.9))

    def test_domain_restriction(self):
        request = CrawlRequest(
            objective="scholarships",
            seeds=["https://example.edu"],
            same_domain_only=True,
        )
        policy = CrawlPolicy.from_request(request)

        self.assertTrue(policy.allows_url("https://example.edu/admissions", depth=1))
        self.assertFalse(policy.allows_url("https://other.edu/scholarships", depth=1))

    def test_maximum_depth(self):
        request = CrawlRequest(objective="scholarships", seeds=["https://example.edu"], max_depth=1)
        policy = CrawlPolicy.from_request(request)

        self.assertTrue(policy.allows_url("https://example.edu/a", depth=1))
        self.assertFalse(policy.allows_url("https://example.edu/b", depth=2))


if __name__ == "__main__":
    unittest.main()
