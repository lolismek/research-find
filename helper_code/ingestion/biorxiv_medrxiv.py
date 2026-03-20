"""
Client for interacting with the bioRxiv/medRxiv API.

bioRxiv and medRxiv are preprint servers for biology and medicine respectively.
They provide cutting-edge research before formal peer review and publication.

API Documentation: https://api.biorxiv.org/

Note: This API provides metadata for preprints, including:
- bioRxiv: biology preprints
- medRxiv: medicine/health sciences preprints
"""

import aiohttp
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

BIORXIV_API_BASE_URL = "https://api.biorxiv.org"

# The API returns at most 100 papers per request
_PAGE_SIZE = 100
# Safety limit to prevent infinite pagination
_MAX_PAGES = 50


async def _biorxiv_api_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Internal helper to make a GET request to the bioRxiv/medRxiv API.
    """
    url = f"{BIORXIV_API_BASE_URL}{endpoint}"

    async with session.get(url, params=params) as response:
        if response.status != 200:
            error_text = await response.text()
            raise ValueError(f"bioRxiv/medRxiv API request failed: HTTP {response.status}, " f"URL: {response.url}, Error: {error_text}")
        try:
            return await response.json()
        except aiohttp.ContentTypeError:
            error_text = await response.text()
            raise ValueError(f"bioRxiv/medRxiv API response was not valid JSON. " f"URL: {response.url}, Response: {error_text}")


def _format_date(date_obj: datetime) -> str:
    """Format datetime to YYYY-MM-DD for API."""
    return date_obj.strftime("%Y-%m-%d")


def _matches_query(
    paper: Dict[str, Any],
    query_terms: List[str],
    match_all_terms: bool,
    min_match_fraction: Optional[float],
) -> bool:
    """Check if a paper matches query terms."""
    if not query_terms:
        return True

    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    authors = (paper.get("authors") or "").lower()
    category = (paper.get("category") or "").lower()
    searchable_text = f"{title} {abstract} {authors} {category}"

    if min_match_fraction is not None:
        matched = sum(1 for term in query_terms if term in searchable_text)
        return matched >= max(1, len(query_terms) * min_match_fraction)
    elif match_all_terms:
        return all(term in searchable_text for term in query_terms)
    else:
        return any(term in searchable_text for term in query_terms)


def _normalize_paper(paper: Dict[str, Any], srv: str) -> Dict[str, Any]:
    """Normalize a bioRxiv/medRxiv paper dict to common format."""
    return {
        "paper_id": paper.get("doi"),
        "doi": paper.get("doi"),
        "url": f"https://www.{srv}.org/content/{paper.get('doi')}v{paper.get('version', '1')}",
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "year": int(paper.get("date", "").split("-")[0]) if paper.get("date") else None,
        "date": paper.get("date"),
        "venue": srv.capitalize(),
        "is_open_access": True,
        "authors": paper.get("authors"),
        "category": paper.get("category"),
        "version": paper.get("version"),
        "source": srv,
        "type": "preprint",
        "published": paper.get("published"),
        "jatsxml": paper.get("jatsxml"),
    }


async def _fetch_all_pages(
    session: aiohttp.ClientSession,
    srv: str,
    start_date_str: str,
    end_date_str: str,
) -> List[Dict[str, Any]]:
    """Fetch all pages from the bioRxiv/medRxiv details endpoint."""
    all_papers: List[Dict[str, Any]] = []
    cursor = 0

    for page in range(_MAX_PAGES):
        endpoint = f"/details/{srv}/{start_date_str}/{end_date_str}/{cursor}"

        if page == 0:
            print(f"[bioRxiv] Querying: {BIORXIV_API_BASE_URL}{endpoint}")

        response_data = await _biorxiv_api_request(session, endpoint)

        messages = response_data.get("messages", [])
        collection = response_data.get("collection", [])

        # Check for API errors
        if messages:
            for msg in messages:
                if msg.get("status") == "error":
                    print(f"Warning: {srv} API error: {msg.get('text', 'Unknown error')}")

            # Extract total count from messages if available
            # Messages format: [{"status": "ok", "count": 150, "total": 12345}]
            total = None
            for msg in messages:
                if isinstance(msg, dict) and "total" in msg:
                    total = msg["total"]
                    if page == 0:
                        print(f"[bioRxiv] Total available from {srv}: {total}")

        if not collection:
            break

        all_papers.extend(collection)
        cursor += len(collection)

        if page == 0:
            print(f"[bioRxiv] Page 1: {len(collection)} papers from {srv}")

        # The API returns fewer than _PAGE_SIZE results when we've hit the end
        if len(collection) < _PAGE_SIZE:
            break

    if len(all_papers) > _PAGE_SIZE:
        print(f"[bioRxiv] Fetched {len(all_papers)} total papers from {srv} across {page + 1} pages")

    return all_papers


async def search_biorxiv_medrxiv_papers(
    session: aiohttp.ClientSession,
    query: str,
    server: str = "biorxiv",
    interval_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    match_all_terms: bool = False,
    min_match_fraction: Optional[float] = None,
    max_results: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Search bioRxiv or medRxiv for papers matching a query.

    The API provides several endpoints. This function uses the content detail
    endpoint which allows searching for papers posted within a date range,
    then filters by query. Handles pagination automatically (API returns
    up to 100 papers per call).

    Args:
        session: An aiohttp.ClientSession object.
        query: Search terms to filter papers (case-insensitive).
               Will search in title, abstract, authors, and category.
               Use empty string "" to return all papers without filtering.
        server: Which server to search - 'biorxiv', 'medrxiv', or 'both'.
        interval_days: Number of days to look back from today.
                      If None, uses start_date/end_date parameters.
                      If None and no dates provided, defaults to last 90 days.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        match_all_terms: If True, ALL query terms must match (AND logic).
                        If False, ANY query term must match (OR logic).
                        Default False (more permissive).
        min_match_fraction: If set (0.0-1.0), requires at least this
                           fraction of query terms to match. Overrides
                           match_all_terms. E.g., 0.5 = at least half
                           the terms must be present.
        max_results: Maximum number of results to return. None = all results.

    Returns:
        A list of dictionaries containing paper metadata.
    """
    # Determine date range
    if interval_days:
        end = datetime.now()
        start = end - timedelta(days=interval_days)
        start_date_str = _format_date(start)
        end_date_str = _format_date(end)
    elif start_date and end_date:
        start_date_str = start_date
        end_date_str = end_date
    else:
        end = datetime.now()
        start = end - timedelta(days=90)
        start_date_str = _format_date(start)
        end_date_str = _format_date(end)

    all_results: List[Dict[str, Any]] = []
    servers_to_query = ["biorxiv", "medrxiv"] if server.lower() == "both" else [server.lower()]
    query_terms = query.lower().split()

    for srv in servers_to_query:
        try:
            # Fetch all pages from the API
            collection = await _fetch_all_pages(session, srv, start_date_str, end_date_str)

            # Filter by query terms
            matched_count = 0
            for paper in collection:
                if _matches_query(paper, query_terms, match_all_terms, min_match_fraction):
                    matched_count += 1
                    all_results.append(_normalize_paper(paper, srv))

                    if max_results and len(all_results) >= max_results:
                        break

            print(f"[bioRxiv] Matched {matched_count}/{len(collection)} papers from {srv} for query: {query}")

        except ValueError as e:
            print(f"Error querying {srv}: {e}")
            continue

        if max_results and len(all_results) >= max_results:
            break

    return all_results
