from __future__ import annotations

import asyncio
import logging

import httpx

from engine.models import FetchResult
from engine.policies import is_html_content_type

logger = logging.getLogger(__name__)


class AsyncPageFetcher:
    def __init__(
        self,
        user_agent: str,
        timeout_seconds: float = 10.0,
        retry_limit: int = 2,
        max_response_bytes: int = 2_000_000,
    ) -> None:
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be at least 1")
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.retry_limit = max(0, retry_limit)
        self.max_response_bytes = max_response_bytes
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "AsyncPageFetcher":
        self._client = httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        )
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str) -> FetchResult:
        if self._client is None:
            raise RuntimeError("AsyncPageFetcher must be used as an async context manager")

        last_error = "unknown fetch error"
        for attempt in range(self.retry_limit + 1):
            try:
                response = await self._client.get(url)
                content_type = response.headers.get("content-type", "")
                if response.status_code >= 400:
                    if response.status_code in {408, 425, 429, 500, 502, 503, 504} and attempt < self.retry_limit:
                        await asyncio.sleep(min(2.0, 0.25 * (attempt + 1)))
                        continue
                    return FetchResult(
                        url=str(response.url),
                        status_code=response.status_code,
                        content_type=content_type,
                        text=None,
                        error=f"HTTP {response.status_code}",
                    )
                if len(response.content) > self.max_response_bytes:
                    return FetchResult(
                        url=str(response.url),
                        status_code=response.status_code,
                        content_type=content_type,
                        text=None,
                        error=f"Response exceeds {self.max_response_bytes} byte limit",
                    )
                if not is_html_content_type(content_type) and not (
                    not content_type and _looks_like_html(response.content)
                ):
                    return FetchResult(
                        url=str(response.url),
                        status_code=response.status_code,
                        content_type=content_type,
                        text=None,
                        error=f"Unsupported content type: {content_type or 'unknown'}",
                    )
                return FetchResult(
                    url=str(response.url),
                    status_code=response.status_code,
                    content_type=content_type,
                    text=response.text,
                )
            except httpx.HTTPError as exc:
                last_error = str(exc)
                logger.warning("Fetch failed for %s on attempt %s: %s", url, attempt + 1, exc)
                if attempt < self.retry_limit:
                    await asyncio.sleep(min(2.0, 0.25 * (attempt + 1)))

        return FetchResult(url=url, status_code=None, content_type="", text=None, error=last_error)


def _looks_like_html(content: bytes) -> bool:
    sample = content.lstrip()[:512].lower()
    return sample.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))


def fetch_page(url: str) -> str | None:
    async def _fetch() -> str | None:
        async with AsyncPageFetcher(user_agent="DeepTrainCrawler/0.1") as fetcher:
            result = await fetcher.fetch(url)
            return result.text if result.ok else None

    return asyncio.run(_fetch())
