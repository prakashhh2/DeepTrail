import unittest

from engine.models import DiscoveredLink, PageData
from engine.scorer import HybridRelevanceScorer, KeywordRelevanceScorer, SemanticRelevanceScorer


class FakeEmbeddingBackend:
    """A tiny deterministic backend for scorer tests."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            if any(term in lowered for term in ("cancer", "tumor", "neural", "medical imaging")):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def _page(*, title: str = "", text: str = "", url: str = "https://example.edu/", anchor: str = "") -> PageData:
    return PageData(url=url, title=title, text=text, links=[], depth=1, incoming_anchor_text=anchor)


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


class SemanticRelevanceScorerTests(unittest.TestCase):
    def test_semantically_similar_text_with_different_wording_scores_higher(self):
        backend = FakeEmbeddingBackend()
        scorer = SemanticRelevanceScorer(embedding_backend=backend)
        objective = "Find research about AI being used for cancer diagnosis"
        related = _page(title="Tumor detection", text="Deep neural networks detect tumors from medical imaging.")
        unrelated = _page(title="Football results", text="Weekend scores and league tables.")

        related_score = scorer.score_page(objective, related)
        unrelated_score = scorer.score_page(objective, unrelated)

        self.assertGreater(related_score, unrelated_score)
        self.assertEqual(related_score, 1.0)
        self.assertEqual(unrelated_score, 0.0)

    def test_link_scoring_uses_anchor_and_url_without_fetching(self):
        backend = FakeEmbeddingBackend()
        scorer = SemanticRelevanceScorer(embedding_backend=backend)
        objective = "Find research about AI being used for cancer diagnosis"
        related = DiscoveredLink(
            url="https://example.edu/medical-imaging/tumors",
            anchor_text="Neural tumor detection research",
            source_url="https://example.edu",
            depth=1,
        )
        unrelated = DiscoveredLink(
            url="https://example.edu/football",
            anchor_text="Football results",
            source_url="https://example.edu",
            depth=1,
        )

        self.assertGreater(scorer.score_link(objective, related), scorer.score_link(objective, unrelated))
        self.assertEqual(sum(len(call) for call in backend.calls), 3)

    def test_objective_embedding_is_reused_and_repeated_text_is_cached(self):
        backend = FakeEmbeddingBackend()
        scorer = SemanticRelevanceScorer(embedding_backend=backend)
        objective = "AI cancer diagnosis"
        page = _page(title="Tumor detection", text="Neural medical imaging")

        scorer.prepare(objective)
        scorer.score_page(objective, page)
        scorer.score_page(objective, page)

        self.assertEqual(backend.calls[0], [objective])
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[1][0].startswith("Title:"), True)

    def test_empty_context_and_backend_errors_are_safe(self):
        backend = FakeEmbeddingBackend()
        scorer = SemanticRelevanceScorer(embedding_backend=backend)
        self.assertEqual(scorer.score_page("objective", _page()), 0.0)

        class FailingBackend:
            def encode(self, texts):
                raise RuntimeError("inference failed")

        failing = SemanticRelevanceScorer(embedding_backend=FailingBackend())
        self.assertEqual(failing.score_page("objective", _page(text="some content")), 0.0)

    def test_score_is_always_normalized(self):
        scorer = SemanticRelevanceScorer(embedding_backend=FakeEmbeddingBackend())
        score = scorer.score_page("objective", _page(text="football"))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class HybridRelevanceScorerTests(unittest.TestCase):
    def test_hybrid_score_uses_configured_weights_and_explains_components(self):
        class FixedScorer:
            def prepare(self, objective):
                pass

            def score_page(self, objective, page):
                return 0.75

            def score_link(self, objective, link):
                return 0.75

            def explain(self, objective, page, score):
                return "fixed"

        class FixedKeywordScorer(FixedScorer):
            def score_page(self, objective, page):
                return 0.25

            def score_link(self, objective, link):
                return 0.25

        scorer = HybridRelevanceScorer(
            semantic_scorer=FixedScorer(),
            keyword_scorer=FixedKeywordScorer(),
            semantic_weight=0.8,
            keyword_weight=0.2,
        )
        page = _page(text="content")

        self.assertEqual(scorer.score_page("objective", page), 0.65)
        self.assertIn("Semantic similarity: 0.75", scorer.explain("objective", page, 0.65))
        self.assertIn("keyword relevance: 0.25", scorer.explain("objective", page, 0.65))

    def test_invalid_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            HybridRelevanceScorer(semantic_weight=-1.0)
        with self.assertRaises(ValueError):
            HybridRelevanceScorer(semantic_weight=0.0, keyword_weight=0.0)


if __name__ == "__main__":
    unittest.main()
