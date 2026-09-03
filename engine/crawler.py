from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from engine.fetcher import AsyncPageFetcher
from engine.frontier import CrawlFrontier, FrontierItem
from engine.models import CrawlRequest, CrawlResponse, CrawlResult, CrawlStats, DiscoveredLink, FetchResult
from engine.parser import HTMLPageParser
from engine.policies import CrawlPolicy, DomainRateLimiter, RobotsTxtChecker
from engine.scorer import HybridRelevanceScorer, RelevanceScorer
from engine.utils import normalize_url

logger = logging.getLogger(__name__)


class PageFetcher(Protocol):
    async def fetch(self, url: str) -> FetchResult:
        ...


@dataclass(slots=True)
class _ProcessedPage:
    result: CrawlResult | None
    links: list[DiscoveredLink]
    error: str | None = None


async def crawl(
    request: CrawlRequest,
    *,
    fetcher: PageFetcher | None = None,
    parser: HTMLPageParser | None = None,
    scorer: RelevanceScorer | None = None,
    robots_checker: RobotsTxtChecker | None = None,
) -> CrawlResponse:
    _validate_request(request)

    started_at = time.monotonic()
    stats = CrawlStats()
    errors: list[str] = []
    results: list[CrawlResult] = []
    visited: set[str] = set()

    policy = CrawlPolicy.from_request(request)
    frontier = CrawlFrontier()
    page_parser = parser or HTMLPageParser()
    relevance_scorer = scorer or HybridRelevanceScorer()
    scoring_executor = ThreadPoolExecutor(max_workers=max(1, request.concurrency))
    rate_limiter = DomainRateLimiter(request.crawl_delay_seconds)
    robots = robots_checker or RobotsTxtChecker(request.user_agent, timeout_seconds=min(5.0, request.request_timeout_seconds))

    for seed in request.seeds:
        normalized = normalize_url(seed)
        if normalized is None:
            errors.append(f"Invalid seed URL: {seed}")
            continue
        if not policy.allows_url(normalized, depth=0):
            errors.append(f"Seed URL is outside crawl policy: {seed}")
            continue
        if not frontier.add(normalized, depth=0, score=1.0):
            stats.duplicates_skipped += 1

    owns_fetcher = fetcher is None
    active_fetcher = fetcher

    async def run_loop() -> None:
        nonlocal active_fetcher
        while frontier and stats.pages_crawled < request.max_pages:
            batch: list[FrontierItem] = []
            remaining = request.max_pages - stats.pages_crawled
            batch_size = min(max(1, request.concurrency), remaining)

            while frontier and len(batch) < batch_size:
                item = frontier.pop()
                if item.url in visited:
                    stats.duplicates_skipped += 1
                    continue
                if not policy.allows_url(item.url, item.depth):
                    continue
                visited.add(item.url)
                batch.append(item)

            if not batch:
                continue

            processed_pages = await asyncio.gather(
                *(
                    _process_item(
                        item,
                        request=request,
                        fetcher=active_fetcher,
                        parser=page_parser,
                        scorer=relevance_scorer,
                        policy=policy,
                        rate_limiter=rate_limiter,
                        robots_checker=robots,
                        scoring_executor=scoring_executor,
                    )
                    for item in batch
                )
            )

            for processed in processed_pages:
                if processed.error:
                    stats.pages_failed += 1
                    errors.append(processed.error)
                    continue

                if processed.result is None:
                    continue

                stats.pages_crawled += 1
                results.append(processed.result)

                for link in processed.links:
                    stats.urls_discovered += 1
                    if not policy.allows_url(link.url, link.depth):
                        continue
                    if link.url in visited or frontier.contains(link.url):
                        stats.duplicates_skipped += 1
                        continue
                    if not frontier.add(
                        link.url,
                        depth=link.depth,
                        score=link.score,
                        anchor_text=link.anchor_text,
                        source_url=link.source_url,
                    ):
                        stats.duplicates_skipped += 1

    try:
        await _prepare_scorer(relevance_scorer, request.objective, scoring_executor)
        if owns_fetcher:
            async with AsyncPageFetcher(
                user_agent=request.user_agent,
                timeout_seconds=request.request_timeout_seconds,
                retry_limit=request.retry_limit,
            ) as default_fetcher:
                active_fetcher = default_fetcher
                await run_loop()
        else:
            await run_loop()
    finally:
        scoring_executor.shutdown(wait=True, cancel_futures=True)

    results.sort(key=lambda result: result.score, reverse=True)
    stats.crawl_duration_ms = int((time.monotonic() - started_at) * 1000)
    return CrawlResponse(results=results[: request.result_limit], stats=stats, errors=errors)


async def _process_item(
    item: FrontierItem,
    *,
    request: CrawlRequest,
    fetcher: PageFetcher | None,
    parser: HTMLPageParser,
    scorer: RelevanceScorer,
    policy: CrawlPolicy,
    rate_limiter: DomainRateLimiter,
    robots_checker: RobotsTxtChecker,
    scoring_executor: ThreadPoolExecutor,
) -> _ProcessedPage:
    if fetcher is None:
        raise RuntimeError("Crawler fetcher was not initialized")

    try:
        if request.respect_robots_txt and not await robots_checker.allowed(item.url):
            logger.info("Skipping %s because robots.txt disallows it", item.url)
            return _ProcessedPage(result=None, links=[], error=f"Robots.txt disallowed URL: {item.url}")

        await rate_limiter.wait(item.url)
        fetch_result = await fetcher.fetch(item.url)
        if not fetch_result.ok:
            logger.info("Fetch failed for %s: %s", item.url, fetch_result.error)
            return _ProcessedPage(result=None, links=[], error=f"Failed to fetch {item.url}: {fetch_result.error}")

        if not policy.allows_url(fetch_result.url, item.depth):
            logger.info("Skipping redirected URL outside crawl policy: %s", fetch_result.url)
            return _ProcessedPage(
                result=None,
                links=[],
                error=f"Redirected URL outside crawl policy: {fetch_result.url}",
            )

        page = parser.parse(
            fetch_result.text or "",
            fetch_result.url,
            depth=item.depth,
            incoming_anchor_text=item.anchor_text,
        )
        page_score = await _score_page(scorer, request.objective, page, scoring_executor)
        link_scores = await _score_links(scorer, request.objective, page.links, scoring_executor)
        for link, link_score in zip(page.links, link_scores):
            link.score = link_score

        result = CrawlResult(
            url=page.url,
            title=page.title,
            score=page_score,
            depth=page.depth,
            reason=await _explain(scorer, request.objective, page, page_score, scoring_executor),
        )
        return _ProcessedPage(result=result, links=page.links)
    except Exception as exc:
        logger.exception("Unexpected crawler error for %s", item.url)
        return _ProcessedPage(result=None, links=[], error=f"Unexpected error for {item.url}: {exc}")


def crawl_sync(request: CrawlRequest) -> CrawlResponse:
    return asyncio.run(crawl(request))


def _validate_request(request: CrawlRequest) -> None:
    if not request.objective.strip():
        raise ValueError("objective is required")
    if not request.seeds:
        raise ValueError("at least one seed URL is required")
    if request.max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    if request.max_depth < 0:
        raise ValueError("max_depth cannot be negative")
    if request.concurrency < 1:
        raise ValueError("concurrency must be at least 1")


async def _run_in_executor(executor: ThreadPoolExecutor, function, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, functools.partial(function, *args))


async def _prepare_scorer(scorer: RelevanceScorer, objective: str, executor: ThreadPoolExecutor) -> None:
    prepare = getattr(scorer, "prepare", None)
    if prepare is None:
        return
    prepared = await _run_in_executor(executor, prepare, objective)
    if inspect.isawaitable(prepared):
        await prepared


async def _score_page(scorer: RelevanceScorer, objective: str, page, executor: ThreadPoolExecutor) -> float:
    return await _run_in_executor(executor, scorer.score_page, objective, page)


async def _score_links(
    scorer: RelevanceScorer,
    objective: str,
    links: list[DiscoveredLink],
    executor: ThreadPoolExecutor,
) -> list[float]:
    if not links:
        return []

    score_links = getattr(scorer, "score_links", None)
    if score_links is not None:
        scores = await _run_in_executor(executor, score_links, objective, links)
        return list(scores)

    # Keep compatibility with the original single-link scorer contract.  Each
    # call runs off the event loop; scorers that support batching use the hook
    # above and avoid this sequential fallback.
    scores: list[float] = []
    for link in links:
        scores.append(await _run_in_executor(executor, scorer.score_link, objective, link))
    return scores


async def _explain(
    scorer: RelevanceScorer,
    objective: str,
    page,
    score: float,
    executor: ThreadPoolExecutor,
) -> str:
    return await _run_in_executor(executor, scorer.explain, objective, page, score)
