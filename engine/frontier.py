from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from itertools import count


@dataclass(order=True, slots=True)
class FrontierItem:
    priority: float
    sequence: int
    url: str = field(compare=False)
    depth: int = field(compare=False)
    score: float = field(compare=False)
    anchor_text: str = field(compare=False, default="")
    source_url: str = field(compare=False, default="")


class CrawlFrontier:
    def __init__(self) -> None:
        self._heap: list[FrontierItem] = []
        self._queued_urls: set[str] = set()
        self._counter = count()

    def add(self, url: str, depth: int, score: float, anchor_text: str = "", source_url: str = "") -> bool:
        if url in self._queued_urls:
            return False

        item = FrontierItem(
            priority=-score,
            sequence=next(self._counter),
            url=url,
            depth=depth,
            score=score,
            anchor_text=anchor_text,
            source_url=source_url,
        )
        heapq.heappush(self._heap, item)
        self._queued_urls.add(url)
        return True

    def pop(self) -> FrontierItem:
        item = heapq.heappop(self._heap)
        self._queued_urls.discard(item.url)
        return item

    def contains(self, url: str) -> bool:
        return url in self._queued_urls

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)
