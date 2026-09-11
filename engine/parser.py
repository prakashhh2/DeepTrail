from __future__ import annotations

import re

from bs4 import BeautifulSoup, Comment, Tag

from engine.models import DiscoveredLink, PageData
from engine.policies import is_probably_html_url
from engine.utils import collapse_whitespace, normalize_url


_REMOVE_TAGS = {
    "script", "style", "noscript", "template", "svg", "canvas", "iframe",
    "form", "nav", "footer", "aside",
}
_BOILERPLATE_RE = re.compile(
    r"(?:^|[-_ ])(?:nav|menu|footer|header|sidebar|breadcrumb|cookie|consent|"
    r"modal|popup|advert|banner|social|share|related|comments?)(?:$|[-_ ])",
    re.IGNORECASE,
)
_CONTENT_HINT_RE = re.compile(r"(?:article|content|entry|main|post|story|body)", re.IGNORECASE)


class HTMLPageParser:
    """Extract readable page evidence and high-signal links from HTML."""

    def __init__(self, *, max_text_chars: int = 60_000) -> None:
        if max_text_chars < 1:
            raise ValueError("max_text_chars must be at least 1")
        self.max_text_chars = max_text_chars

    def parse(self, html: str, url: str, depth: int, incoming_anchor_text: str = "") -> PageData:
        soup = BeautifulSoup(html, "html.parser")
        self._remove_non_content(soup)

        title = self._meta_content(soup, "title")
        description = self._description(soup)
        canonical_url = self._canonical_url(soup, url)
        root = self._content_root(soup)
        headings = self._blocks(root, {"h1", "h2", "h3"})
        if not title and headings:
            title = headings[0]
        text = self._readable_text(root)
        if not text:
            text = collapse_whitespace(root.get_text(" ", strip=True))
        text = text[: self.max_text_chars].rstrip()

        links: list[DiscoveredLink] = []
        seen: dict[str, DiscoveredLink] = {}
        for tag in root.find_all("a", href=True):
            normalized = normalize_url(str(tag["href"]), base_url=url)
            if normalized is None or not is_probably_html_url(normalized):
                continue
            anchor_text = collapse_whitespace(
                tag.get_text(" ", strip=True) or str(tag.get("aria-label", "")) or str(tag.get("title", ""))
            )
            candidate = DiscoveredLink(
                url=normalized,
                anchor_text=anchor_text,
                source_url=url,
                depth=depth + 1,
                context=self._link_context(tag),
            )
            existing = seen.get(normalized)
            if existing is None:
                seen[normalized] = candidate
                links.append(candidate)
            elif not existing.anchor_text and anchor_text:
                existing.anchor_text = anchor_text
                existing.context = candidate.context

        return PageData(
            url=url,
            title=title,
            text=text,
            links=links,
            depth=depth,
            incoming_anchor_text=incoming_anchor_text,
            headings=[heading for heading in headings if heading],
            description=description,
            canonical_url=canonical_url,
            word_count=len(text.split()),
        )

    @staticmethod
    def _remove_non_content(soup: BeautifulSoup) -> None:
        for element in soup.find_all(string=lambda value: isinstance(value, Comment)):
            element.extract()
        for tag in soup.find_all(list(_REMOVE_TAGS)):
            tag.decompose()
        for tag in list(soup.find_all(True)):
            if not isinstance(tag, Tag):
                continue
            if tag.parent is None or tag.attrs is None:
                continue
            if str(tag.get("aria-hidden", "")).lower() == "true":
                tag.decompose()
                continue
            marker = " ".join(
                part for part in (str(tag.get("id", "")), " ".join(tag.get("class", []))) if part
            )
            if marker and _BOILERPLATE_RE.search(marker):
                tag.decompose()

    @staticmethod
    def _content_root(soup: BeautifulSoup) -> Tag:
        candidates = list(soup.find_all(["main", "article"]))
        candidates.extend(soup.find_all(attrs={"role": "main"}))
        candidates.extend(
            tag for tag in soup.find_all(["div", "section"])
            if _CONTENT_HINT_RE.search(" ".join([str(tag.get("id", "")), " ".join(tag.get("class", []))]))
        )
        if not candidates:
            return soup.body or soup

        def quality(tag: Tag) -> tuple[float, int]:
            text_length = len(collapse_whitespace(tag.get_text(" ", strip=True)))
            link_length = sum(len(collapse_whitespace(link.get_text(" ", strip=True))) for link in tag.find_all("a"))
            paragraphs = len(tag.find_all(["p", "h1", "h2", "h3", "li", "blockquote"]))
            density = text_length - min(text_length * 0.7, link_length * 0.6)
            return density + paragraphs * 40, text_length

        return max(candidates, key=quality)

    @staticmethod
    def _blocks(root: Tag, names: set[str]) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for tag in root.find_all(names):
            value = collapse_whitespace(tag.get_text(" ", strip=True))
            if value and value not in seen:
                seen.add(value)
                values.append(value)
        return values

    @classmethod
    def _readable_text(cls, root: Tag) -> str:
        blocks = cls._blocks(root, {"h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "td"})
        return "\n\n".join(blocks)

    @staticmethod
    def _description(soup: BeautifulSoup) -> str:
        for attrs in (
            {"name": lambda value: value and value.lower() == "description"},
            {"property": lambda value: value and value.lower() in {"og:description", "twitter:description"}},
        ):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                return collapse_whitespace(str(tag["content"]))
        return ""

    @staticmethod
    def _meta_content(soup: BeautifulSoup, name: str) -> str:
        tag = soup.find(name)
        return collapse_whitespace(tag.get_text(" ", strip=True)) if tag else ""

    @staticmethod
    def _canonical_url(soup: BeautifulSoup, base_url: str) -> str:
        tag = soup.find("link", rel=lambda value: value and "canonical" in value)
        if not tag or not tag.get("href"):
            return ""
        return normalize_url(str(tag["href"]), base_url=base_url) or ""

    @staticmethod
    def _link_context(tag: Tag) -> str:
        heading = tag.find_previous(["h1", "h2", "h3"])
        return collapse_whitespace(heading.get_text(" ", strip=True)) if heading else ""


def extract_links(html: str, base_url: str) -> list[str]:
    return [link.url for link in HTMLPageParser().parse(html, base_url, depth=0).links]
