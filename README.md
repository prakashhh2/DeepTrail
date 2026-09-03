# DeepTrail crawler engine

DeepTrain is a small asynchronous web crawler that prioritizes discovered URLs
with relevance scores. It supports domain/depth policies, robots.txt checks,
concurrent fetching, retries, and a bounded result set.

## Installation

```bash
python -m pip install -r engine/requirement/requirement.txt
```

The default crawler scorer is `HybridRelevanceScorer`. It combines keyword
overlap with semantic similarity using the local
`sentence-transformers/all-MiniLM-L6-v2` model. Sentence Transformers downloads
the model the first time it is used. If model loading or inference fails,
semantic scoring safely contributes zero and keyword scoring continues to work.

## Command line usage

```bash
python -m engine.main "companies developing autonomous warehouse robots" \
  https://example.com --max-pages 25 --max-depth 2
```

Useful options include `--concurrency`, `--crawl-delay-seconds`,
`--allow-external-domains`, and `--ignore-robots-txt`.

## Python usage

```python
import asyncio

from engine.crawler import crawl
from engine.models import CrawlRequest


async def main():
    response = await crawl(
        CrawlRequest(
            objective="companies developing autonomous warehouse robots",
            seeds=["https://example.com"],
            max_pages=25,
        )
    )
    for result in response.results:
        print(result.score, result.url, result.reason)


asyncio.run(main())
```

## Scoring

`KeywordRelevanceScorer` measures lexical overlap across page metadata,
content, anchors, and URL paths. `SemanticRelevanceScorer` embeds a bounded
page representation containing the title, headings, description, incoming
anchor, URL path, and page text. Links are scored from their anchor text and
URL before the target is fetched.

`HybridRelevanceScorer` uses semantic and keyword scores weighted 0.8 and 0.2
by default. Weights are configurable and normalized to keep scores in the
`0.0`–`1.0` range. The objective embedding is prepared once per scorer/objective,
and repeated page/link representations are cached.

Custom scorers still use the original dependency-injection interface:

```python
response = await crawl(request, scorer=my_scorer)
```

A scorer needs `score_page`, `score_link`, and `explain`. An optional `prepare`
method can initialize per-crawl state, and an optional `score_links` method can
batch link scoring.

## Development

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

Embedding tests use a deterministic mock backend and do not download a model.
