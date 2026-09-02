from __future__ import annotations

from typing import Protocol

from engine.models import DiscoveredLink, PageData
from engine.utils import cosine_similarity, tokenize, url_path_text


class RelevanceScorer(Protocol):
    def score_page(self, objective: str, page: PageData) -> float:
        ...

    def score_link(self, objective: str, link: DiscoveredLink) -> float:
        ...

    def explain(self, objective: str, page: PageData, score: float) -> str:
        ...


class KeywordRelevanceScorer:
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
