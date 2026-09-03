from __future__ import annotations

from math import isfinite, sqrt
from threading import RLock
from typing import Any, Protocol, Sequence

from engine.models import DiscoveredLink, PageData
from engine.utils import cosine_similarity, tokenize, url_path_text


class RelevanceScorer(Protocol):
    def score_page(self, objective: str, page: PageData) -> float:
        ...

    def score_link(self, objective: str, link: DiscoveredLink) -> float:
        ...

    def explain(self, objective: str, page: PageData, score: float) -> str:
        ...


class EmbeddingBackend(Protocol):
    """Small adapter interface for local or test embedding implementations."""

    def encode(self, texts: Sequence[str]) -> Any:
        ...


class KeywordRelevanceScorer:
    def prepare(self, objective: str) -> None:
        """Keep the scorer lifecycle compatible with embedding-based scorers."""

    def score_page(self, objective: str, page: PageData) -> float:
        title_similarity = cosine_similarity(objective, page.title)
        anchor_similarity = cosine_similarity(objective, page.incoming_anchor_text)
        url_similarity = cosine_similarity(objective, url_path_text(page.url))
        content_similarity = cosine_similarity(objective, page.text[:20_000])

        score = (
            title_similarity * 0.30
            + anchor_similarity * 0.25
            + url_similarity * 0.15
            + content_similarity * 0.30
        )
        return min(1.0, round(score, 4))

    def score_link(self, objective: str, link: DiscoveredLink) -> float:
        anchor_similarity = cosine_similarity(objective, link.anchor_text)
        url_similarity = cosine_similarity(objective, url_path_text(link.url))
        score = anchor_similarity * 0.65 + url_similarity * 0.35
        return min(1.0, round(score, 4))

    def explain(self, objective: str, page: PageData, score: float) -> str:
        objective_terms = set(tokenize(objective))
        page_terms = set(tokenize(f"{page.title} {page.incoming_anchor_text} {url_path_text(page.url)} {page.text[:5000]}"))
        shared_terms = sorted(objective_terms & page_terms)

        if shared_terms:
            preview = ", ".join(shared_terms[:6])
            return f"Page matches objective terms: {preview}."
        if score > 0:
            return "Page has weak lexical similarity to the objective."
        return "Page had little direct lexical overlap with the objective."


class SemanticRelevanceScorer:
    """Score pages and links with cosine similarity between text embeddings.

    ``embedding_backend`` is deliberately injected rather than imported at module
    load time.  This keeps the crawler usable without ML dependencies and makes
    deterministic test doubles straightforward.  The default backend is loaded
    lazily on the first scoring call.
    """

    def __init__(
        self,
        *,
        embedding_backend: EmbeddingBackend | Any | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_page_chars: int = 12_000,
    ) -> None:
        if max_page_chars < 1:
            raise ValueError("max_page_chars must be at least 1")

        self.embedding_backend = embedding_backend
        self.model_name = model_name
        self.max_page_chars = max_page_chars
        self._backend: Any | None = embedding_backend
        self._backend_error: Exception | None = None
        self._objective: str | None = None
        self._objective_embedding: tuple[float, ...] | None = None
        self._embedding_cache: dict[str, tuple[float, ...]] = {}
        self._lock = RLock()

    def prepare(self, objective: str) -> None:
        """Encode an objective once and reuse it for all pages and links."""

        objective = objective.strip()
        with self._lock:
            if objective == self._objective:
                return
            self._objective = objective
            self._objective_embedding = None
            self._backend_error = None

            if not objective:
                return
            embeddings = self._encode_many_locked([objective])
            if embeddings:
                self._objective_embedding = embeddings[0]

    def score_page(self, objective: str, page: PageData) -> float:
        self._ensure_objective(objective)
        context = self.page_text(page)
        return self._score_text(context)

    def score_link(self, objective: str, link: DiscoveredLink) -> float:
        self._ensure_objective(objective)
        return self._score_text(self.link_text(link))

    def score_links(self, objective: str, links: Sequence[DiscoveredLink]) -> list[float]:
        """Batch link embeddings while retaining the public single-link API."""

        self._ensure_objective(objective)
        texts = [self.link_text(link) for link in links]
        with self._lock:
            embeddings = self._embeddings_for_texts_locked(texts)
            objective_embedding = self._objective_embedding
        return [self._similarity(objective_embedding, embedding) for embedding in embeddings]

    def explain(self, objective: str, page: PageData, score: float) -> str:
        # ``score`` is supplied by the crawler, so explaining a result never
        # needs an additional inference call.
        return f"Semantic similarity: {score:.2f}; final relevance: {score:.2f}"

    def page_text(self, page: PageData) -> str:
        """Build a bounded, labeled representation instead of embedding HTML."""

        sections = [
            self._section("Title", page.title),
            self._section("Headings", " ".join(getattr(page, "headings", []))),
            self._section("Description", getattr(page, "description", "")),
            self._section("Incoming link", page.incoming_anchor_text),
            self._section("URL", self._url_text(page.url)),
        ]
        metadata = " ".join(section for section in sections if section)
        content_budget = max(0, self.max_page_chars - len(metadata) - 1)
        content = self._clip(page.text, content_budget)
        return " ".join(part for part in (metadata, self._section("Content", content)) if part).strip()

    @staticmethod
    def link_text(link: DiscoveredLink) -> str:
        parts = [
            SemanticRelevanceScorer._section("Anchor", link.anchor_text),
            SemanticRelevanceScorer._section("URL", SemanticRelevanceScorer._url_text(link.url)),
        ]
        return " ".join(part for part in parts if part).strip()

    def _ensure_objective(self, objective: str) -> None:
        if objective.strip() != self._objective or self._objective_embedding is None:
            self.prepare(objective)

    def _score_text(self, text: str) -> float:
        with self._lock:
            objective_embedding = self._objective_embedding
            embeddings = self._embeddings_for_texts_locked([text]) if text else [None]
        return self._similarity(objective_embedding, embeddings[0])

    def _embeddings_for_texts_locked(self, texts: Sequence[str]) -> list[tuple[float, ...] | None]:
        if not texts:
            return []

        result: list[tuple[float, ...] | None] = [self._embedding_cache.get(text) if text else None for text in texts]
        missing = list(dict.fromkeys(text for text, embedding in zip(texts, result) if text and embedding is None))
        if missing:
            encoded = self._encode_many_locked(missing)
            for text, embedding in zip(missing, encoded):
                self._embedding_cache[text] = embedding
            result = [self._embedding_cache.get(text) if text else None for text in texts]
        return result

    def _encode_many_locked(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts or self._backend_error is not None:
            return []

        try:
            backend = self._get_backend_locked()
            raw = backend.encode(list(texts))
            vectors = _as_vectors(raw, expected_count=len(texts))
            if len(vectors) != len(texts):
                raise ValueError("embedding backend returned an unexpected number of vectors")
            return vectors
        except Exception as exc:
            # Scoring is advisory; a model failure must not take down a crawl.
            self._backend_error = exc
            return []

    def _get_backend_locked(self) -> Any:
        if self._backend is not None:
            return self._backend
        if self._backend_error is not None:
            raise self._backend_error

        try:
            from sentence_transformers import SentenceTransformer

            self._backend = SentenceTransformer(self.model_name)
            return self._backend
        except Exception as exc:
            self._backend_error = exc
            raise

    @staticmethod
    def _section(label: str, value: str) -> str:
        value = " ".join(str(value or "").split())
        return f"{label}: {value}" if value else ""

    @staticmethod
    def _url_text(url: str) -> str:
        text = url_path_text(str(url or ""))
        return text if tokenize(text) else ""

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        text = " ".join(str(text or "").split())
        return text[:limit].rstrip() if limit else ""

    @staticmethod
    def _similarity(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> float:
        if (
            not left
            or not right
            or len(left) != len(right)
            or not all(isfinite(value) for value in (*left, *right))
        ):
            return 0.0
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
        return round(max(0.0, min(1.0, cosine)), 4)


class HybridRelevanceScorer:
    """Combine lexical and semantic relevance while preserving score range."""

    def __init__(
        self,
        *,
        semantic_scorer: SemanticRelevanceScorer | None = None,
        keyword_scorer: KeywordRelevanceScorer | None = None,
        semantic_weight: float = 0.8,
        keyword_weight: float = 0.2,
    ) -> None:
        if (
            not isfinite(semantic_weight)
            or not isfinite(keyword_weight)
            or semantic_weight < 0
            or keyword_weight < 0
            or semantic_weight + keyword_weight <= 0
        ):
            raise ValueError("scoring weights must be non-negative and have a positive total")

        total = semantic_weight + keyword_weight
        self.semantic_weight = semantic_weight / total
        self.keyword_weight = keyword_weight / total
        self.semantic_scorer = semantic_scorer or SemanticRelevanceScorer()
        self.keyword_scorer = keyword_scorer or KeywordRelevanceScorer()

    def prepare(self, objective: str) -> None:
        self.semantic_scorer.prepare(objective)
        self.keyword_scorer.prepare(objective)

    def score_page(self, objective: str, page: PageData) -> float:
        semantic = self.semantic_scorer.score_page(objective, page)
        keyword = self.keyword_scorer.score_page(objective, page)
        return self._combine(semantic, keyword)

    def score_link(self, objective: str, link: DiscoveredLink) -> float:
        semantic = self.semantic_scorer.score_link(objective, link)
        keyword = self.keyword_scorer.score_link(objective, link)
        return self._combine(semantic, keyword)

    def score_links(self, objective: str, links: Sequence[DiscoveredLink]) -> list[float]:
        semantic_scores = self.semantic_scorer.score_links(objective, links)
        return [
            self._combine(semantic, self.keyword_scorer.score_link(objective, link))
            for semantic, link in zip(semantic_scores, links)
        ]

    def explain(self, objective: str, page: PageData, score: float) -> str:
        semantic = self.semantic_scorer.score_page(objective, page)
        keyword = self.keyword_scorer.score_page(objective, page)
        return f"Semantic similarity: {semantic:.2f}; keyword relevance: {keyword:.2f}; final relevance: {score:.2f}"

    def _combine(self, semantic: float, keyword: float) -> float:
        return round(max(0.0, min(1.0, self.semantic_weight * semantic + self.keyword_weight * keyword)), 4)


def _as_vectors(raw: Any, *, expected_count: int) -> list[tuple[float, ...]]:
    """Convert common list/numpy embedding outputs into plain float vectors."""

    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if expected_count == 1 and _is_vector(raw):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise TypeError("embedding backend must return a vector sequence")

    vectors: list[tuple[float, ...]] = []
    for vector in raw:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not _is_vector(vector):
            raise TypeError("embedding backend returned an invalid vector")
        vectors.append(tuple(float(value) for value in vector))
    return vectors


def _is_vector(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(item, (int, float)) for item in value)
