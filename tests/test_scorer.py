import unittest

from engine.models import DiscoveredLink, PageData
from engine.scorer import KeywordRelevanceScorer


class KeywordRelevanceScorerTests(unittest.TestCase):
    def test_relevant_page_scores_higher_than_unrelated_page(self):
        scorer = KeywordRelevanceScorer()
        objective = "international undergraduate computer science scholarships"
        relevant = PageData(
            url="https://example.edu/scholarships/computer-science",
            title="Scholarships for International Undergraduate Students",
            text="Computer science students can apply for scholarship funding.",
            links=[],
            depth=1,
            incoming_anchor_text="Scholarships",
        )
        unrelated = PageData(
            url="https://example.edu/athletics",
            title="Athletics",
            text="Team schedules and sports tickets.",
            links=[],
            depth=1,
        )

        self.assertGreater(scorer.score_page(objective, relevant), scorer.score_page(objective, unrelated))

    def test_relevant_link_scores_higher_than_unrelated_link(self):
        scorer = KeywordRelevanceScorer()
        objective = "international undergraduate computer science scholarships"
        relevant = DiscoveredLink(
            url="https://example.edu/international-scholarships",
            anchor_text="International scholarships",
            source_url="https://example.edu",
            depth=1,
        )
        unrelated = DiscoveredLink(
            url="https://example.edu/sports",
            anchor_text="Sports",
            source_url="https://example.edu",
            depth=1,
        )

        self.assertGreater(scorer.score_link(objective, relevant), scorer.score_link(objective, unrelated))


if __name__ == "__main__":
    unittest.main()
