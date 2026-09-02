from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

from engine.crawler import crawl_sync
from engine.models import CrawlRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DeepTrain crawl.")
    parser.add_argument("objective")
    parser.add_argument("seeds", nargs="+")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--allow-external-domains", action="store_true")
    parser.add_argument("--ignore-robots-txt", action="store_true")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--crawl-delay-seconds", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    response = crawl_sync(
        CrawlRequest(
            objective=args.objective,
            seeds=args.seeds,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            same_domain_only=not args.allow_external_domains,
            respect_robots_txt=not args.ignore_robots_txt,
            concurrency=args.concurrency,
            crawl_delay_seconds=args.crawl_delay_seconds,
        )
    )
    print(json.dumps(asdict(response), indent=2))


if __name__ == "__main__":
    main()
