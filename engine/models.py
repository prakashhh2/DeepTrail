from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CrawlRequest:
    objective: str
    seeds: list[str]
    max_pages: int = 50
    max_depth: int = 3
    same_domain_only: bool = True
    concurrency: int = 5
    request_timeout_seconds: float = 10.0
    retry_limit: int = 2
    user_agent: str = "DeepTrainCrawler/0.1 (+https://example.com/bot)"
    crawl_delay_seconds: float = 0.5
    respect_robots_txt: bool = True
    result_limit: int = 10


@dataclass(slots=True)
class DiscoveredLink:
    url: str
    anchor_text: str
    source_url: str
    depth: int
    score: float = 0.0


@dataclass(slots=True)
class PageData:
    url: str
    title: str
    text: str
    links: list[DiscoveredLink]
    depth: int
    incoming_anchor_text: str = ""


@dataclass(slots=True)
class FetchResult:
    url: str
    status_code: int | None
    content_type: str
    text: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.text is not None


@dataclass(slots=True)
class CrawlResult:
    url: str
    title: str
    score: float
    depth: int
    reason: str


@dataclass(slots=True)
class CrawlStats:
    pages_crawled: int = 0
    pages_failed: int = 0
    urls_discovered: int = 0
    duplicates_skipped: int = 0
    crawl_duration_ms: int = 0


@dataclass(slots=True)
class CrawlResponse:
    results: list[CrawlResult]
    stats: CrawlStats
    errors: list[str] = field(default_factory=list)
