from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.models import CrawlResponse


def crawl_response_dict(response: CrawlResponse) -> dict[str, Any]:
    """Return a stable, LLM-friendly corpus document."""

    pages = []
    for rank, result in enumerate(response.results, start=1):
        page = asdict(result)
        page["rank"] = rank
        page["llm_context"] = _context_block(page)
        pages.append(page)

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": response.objective,
        "seeds": response.seeds,
        "stats": asdict(response.stats),
        "errors": response.errors,
        "pages": pages,
    }


def write_crawl_json(response: CrawlResponse, path: str | Path) -> Path:
    """Write the ranked crawl corpus as UTF-8 JSON and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(crawl_response_dict(response), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def _context_block(page: dict[str, Any]) -> str:
    parts = [
        f"Title: {page.get('title', '')}" if page.get("title") else "",
        f"URL: {page.get('url', '')}" if page.get("url") else "",
        f"Description: {page.get('description', '')}" if page.get("description") else "",
        f"Headings: {' | '.join(page.get('headings', []))}" if page.get("headings") else "",
        page.get("text", ""),
    ]
    return "\n\n".join(part for part in parts if part).strip()
