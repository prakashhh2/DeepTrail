from bs4 import BeautifulSoup
from urllib.parse import urljoin


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    links = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"]

        # Convert relative URLs to absolute URLs
        full_url = urljoin(base_url, href)

        # Only HTTP/HTTPS links
        if full_url.startswith(("http://", "https://")):
            links.append(full_url)

    return links