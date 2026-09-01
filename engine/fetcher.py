import httpx


def fetch_page(url: str) -> str | None:
    try:
        response = httpx.get(
            url,
            timeout=10,
            follow_redirects=True,
            headers={
                "User-Agent": "SimpleCrawler/0.1"
            }
        )

        response.raise_for_status()

        # Only process HTML pages
        content_type = response.headers.get("content-type", "")

        if "text/html" not in content_type:
            return None

        return response.text

    except httpx.HTTPError as e:
        print(f"Failed to fetch {url}: {e}")
        return None