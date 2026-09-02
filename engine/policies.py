from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from engine.models import CrawlRequest
from engine.utils import normalize_url, registrable_domain

logger = logging.getLogger(__name__)


NON_HTML_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".rss",
    ".svg",
    ".tar",
    ".webm",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}


def is_probably_html_url(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return not any(path.endswith(extension) for extension in NON_HTML_EXTENSIONS)


def is_html_content_type(content_type: str) -> bool:
    if not content_type:
        return False
    return "text/html" in content_type.lower() or "application/xhtml+xml" in content_type.lower()


@dataclass(slots=True)
class CrawlPolicy:
    request: CrawlRequest
    seed_domains: set[str]

    @classmethod
    def from_request(cls, request: CrawlRequest) -> "CrawlPolicy":
        seed_domains = {
            registrable_domain(normalized)
            for seed in request.seeds
            if (normalized := normalize_url(seed))
        }
        return cls(request=request, seed_domains=seed_domains)

    def allows_url(self, url: str, depth: int) -> bool:
        normalized = normalize_url(url)
        if normalized is None:
            return False
        if depth > self.request.max_depth:
            return False
        if not is_probably_html_url(normalized):
            return False
        if self.request.same_domain_only and registrable_domain(normalized) not in self.seed_domains:
            return False
        return True


class DomainRateLimiter:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_access: dict[str, float] = {}

    async def wait(self, url: str) -> None:
        if self.delay_seconds <= 0:
            return

        domain = registrable_domain(url)
        lock = self._locks.setdefault(domain, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            elapsed = now - self._last_access.get(domain, 0.0)
            wait_for = self.delay_seconds - elapsed
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_access[domain] = time.monotonic()


class RobotsTxtChecker:
    def __init__(self, user_agent: str, timeout_seconds: float = 5.0) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self._cache: dict[str, RobotFileParser | None] = {}

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        parser = await self._get_parser(origin)
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    async def _get_parser(self, origin: str) -> RobotFileParser | None:
        if origin in self._cache:
            return self._cache[origin]

        robots_url = urljoin(origin, "/robots.txt")
        parser = RobotFileParser()
        parser.set_url(robots_url)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = await client.get(robots_url, headers={"User-Agent": self.user_agent})
            if response.status_code >= 400:
                self._cache[origin] = None
                return None
            parser.parse(response.text.splitlines())
            self._cache[origin] = parser
            return parser
        except httpx.HTTPError as exc:
            logger.debug("Could not fetch robots.txt from %s: %s", origin, exc)
            self._cache[origin] = None
            return None
