from __future__ import annotations

from bs4 import BeautifulSoup

from engine.models import DiscoveredLink, PageData
from engine.policies import is_probably_html_url
from engine.utils import collapse_whitespace, normalize_url


class HTMLPageParser:
    def parse(self, html: str, url: str, depth: int, incoming_anchor_text: str = "") -> PageData:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "template", "svg"]):
            tag.decompose()

        title = collapse_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""
        description_tag = soup.find("meta", attrs={"name": lambda value: value and value.lower() == "description"})
        description = collapse_whitespace(str(description_tag.get("content", ""))) if description_tag else ""
        headings = [collapse_whitespace(tag.get_text(" ", strip=True)) for tag in soup.find_all(["h1", "h2", "h3"])]
        text = collapse_whitespace(soup.get_text(" ", strip=True))

        links: list[DiscoveredLink] = []
        seen: set[str] = set()
        for tag in soup.find_all("a", href=True):
            normalized = normalize_url(str(tag["href"]), base_url=url)
            if normalized is None or normalized in seen or not is_probably_html_url(normalized):
                continue
            seen.add(normalized)
            links.append(
                DiscoveredLink(
                    url=normalized,
                    anchor_text=collapse_whitespace(tag.get_text(" ", strip=True)),
                    source_url=url,
                    depth=depth + 1,
                )
            )

        return PageData(
            url=url,
            title=title,
            text=text,
            links=links,
            depth=depth,
            incoming_anchor_text=incoming_anchor_text,
            headings=[heading for heading in headings if heading],
            description=description,
        )


def extract_links(html: str, base_url: str) -> list[str]:
    return [link.url for link in HTMLPageParser().parse(html, base_url, depth=0).links]
