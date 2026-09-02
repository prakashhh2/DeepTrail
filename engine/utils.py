from __future__ import annotations

import re
import posixpath
from collections import Counter
from math import sqrt
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit


TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "msclkid", "mc_cid", "mc_eid"}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_url(url: str, base_url: str | None = None) -> str | None:
    joined = urljoin(base_url, url) if base_url else url
    parts = urlsplit(joined.strip())

    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return None

    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    if not hostname:
        return None

    netloc = hostname
    if parts.port and not ((scheme == "http" and parts.port == 80) or (scheme == "https" and parts.port == 443)):
        netloc = f"{netloc}:{parts.port}"

    path = quote(_normalize_path(unquote(parts.path or "/")), safe="/:@")
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_KEYS and not key.lower().startswith(TRACKING_QUERY_PREFIXES)
    ]
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def _normalize_path(path: str) -> str:
    had_trailing_slash = path.endswith("/")
    normalized = posixpath.normpath(path)
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if had_trailing_slash and normalized != "/":
        normalized = f"{normalized}/"
    return normalized


def registrable_domain(url: str) -> str:
    host = urlsplit(url).hostname or ""
    return host.lower().removeprefix("www.")


def url_path_text(url: str) -> str:
    parts = urlsplit(url)
    text = f"{parts.path} {parts.query}"
    return re.sub(r"[/_\-?=&.%]+", " ", text)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def cosine_similarity(left: str, right: str) -> float:
    left_counts = Counter(tokenize(left))
    right_counts = Counter(tokenize(right))
    if not left_counts or not right_counts:
        return 0.0

    shared = set(left_counts) & set(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in shared)
    left_norm = sqrt(sum(count * count for count in left_counts.values()))
    right_norm = sqrt(sum(count * count for count in right_counts.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
