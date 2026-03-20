"""arXiv URL resolver + RSS feed parser."""

from __future__ import annotations

import asyncio
import re
from typing import Optional

import aiohttp
import feedparser

from models.paper import Paper, Author

# Match arXiv IDs like 2301.12345 or 2301.12345v2
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
# Match full arXiv URLs
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?")


def extract_arxiv_id(url_or_id: str) -> Optional[str]:
    """Extract a canonical arXiv ID (without version) from a URL or raw ID."""
    m = _ARXIV_URL_RE.search(url_or_id)
    if m:
        return m.group(1)
    m = _ARXIV_ID_RE.search(url_or_id)
    if m:
        return m.group(1)
    return None


async def resolve_arxiv_url(url_or_id: str) -> Paper:
    """Resolve an arXiv URL or ID to a Paper using the arXiv Atom API."""
    arxiv_id = extract_arxiv_id(url_or_id)
    if not arxiv_id:
        raise ValueError(f"Could not extract arXiv ID from: {url_or_id}")

    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url) as resp:
            if resp.status != 200:
                raise ValueError(f"arXiv API returned HTTP {resp.status}")
            xml_text = await resp.text()

    feed = feedparser.parse(xml_text)
    if not feed.entries:
        raise ValueError(f"No arXiv paper found for ID: {arxiv_id}")

    entry = feed.entries[0]

    authors = []
    for a in getattr(entry, "authors", []):
        name = a.get("name", "") if isinstance(a, dict) else str(a)
        if name:
            authors.append(Author(name=name))

    year = None
    published = getattr(entry, "published", None)
    if published:
        try:
            year = int(published[:4])
        except (ValueError, IndexError):
            pass

    pdf_url = None
    for link in getattr(entry, "links", []):
        if isinstance(link, dict) and link.get("type") == "application/pdf":
            pdf_url = link.get("href")
            break
    if not pdf_url:
        pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"

    return Paper(
        arxiv_id=arxiv_id,
        title=getattr(entry, "title", "Untitled").replace("\n", " ").strip(),
        abstract=getattr(entry, "summary", None),
        authors=authors,
        year=year,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=pdf_url,
        source="arxiv",
    )


ARXIV_TOP_LEVEL = [
    "cs", "econ", "eess", "math", "astro-ph", "cond-mat",
    "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math-ph", "nlin", "nucl-ex", "nucl-th", "physics",
    "quant-ph", "q-bio", "q-fin", "stat",
]


async def _fetch_single_rss(session: aiohttp.ClientSession, category: str) -> list[Paper]:
    """Fetch a single RSS feed, returning [] on failure."""
    rss_url = f"https://rss.arxiv.org/rss/{category}"
    try:
        async with session.get(rss_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                print(f"[arxiv] RSS returned {resp.status} for {category}")
                return []
            text = await resp.text()
    except Exception as e:
        print(f"[arxiv] Failed to fetch {category}: {e}")
        return []

    feed = feedparser.parse(text)
    papers = []
    for entry in feed.entries:
        arxiv_id = extract_arxiv_id(entry.get("link", "") or entry.get("id", ""))

        authors = []
        author_str = entry.get("author", "") or entry.get("dc_creator", "")
        if author_str:
            for name in author_str.split(","):
                name = name.strip()
                if name:
                    authors.append(Author(name=name))

        papers.append(Paper(
            arxiv_id=arxiv_id,
            title=(entry.get("title", "Untitled")).replace("\n", " ").strip(),
            abstract=entry.get("summary") or entry.get("description"),
            authors=authors,
            url=entry.get("link"),
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
            source="arxiv",
        ))

    return papers


async def fetch_arxiv_rss(category: str | None = None) -> list[Paper]:
    """Fetch latest papers from arXiv RSS feed(s).

    Args:
        category: arXiv category like "cs.AI", "cs.CL", "stat.ML".
                  If None, fetches all top-level categories.

    Returns:
        List of Paper objects, deduplicated by arXiv ID.
    """
    async with aiohttp.ClientSession() as session:
        if category:
            return await _fetch_single_rss(session, category)

        # Fetch all top-level categories in parallel
        print(f"[arxiv] Fetching all {len(ARXIV_TOP_LEVEL)} top-level categories...")
        results = await asyncio.gather(
            *[_fetch_single_rss(session, cat) for cat in ARXIV_TOP_LEVEL]
        )

    # Deduplicate by arxiv_id
    seen = set()
    papers = []
    for batch in results:
        for p in batch:
            key = p.arxiv_id or p.title
            if key not in seen:
                seen.add(key)
                papers.append(p)

    print(f"[arxiv] Fetched {len(papers)} unique papers across all categories")
    return papers
