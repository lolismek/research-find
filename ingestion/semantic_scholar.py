"""
Semantic Scholar API client for paper search.

Used by evidence_service. Handles rate limiting (~1 req/sec) and
retries on 429/5xx with tuned backoff — S2 500s can be sustained,
so we back off aggressively (5s → 15s → 30s).

API docs: https://api.semanticscholar.org/api-docs/graph
"""

import os
import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp

BASE_URL = "https://api.semanticscholar.org"
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

# Fields to request. Excludes restricted fields (citations list, references
# list, embedding, tldr) that require partner-level API access and return 403.
FIELDS = ",".join([
    "paperId", "externalIds", "url", "title", "abstract",
    "venue", "publicationVenue", "year", "referenceCount",
    "citationCount", "influentialCitationCount", "isOpenAccess",
    "openAccessPdf", "fieldsOfStudy", "s2FieldsOfStudy",
    "publicationTypes", "publicationDate", "journal", "authors",
])

# Minimum seconds between requests. Unauthenticated limit is 1 req/sec;
# we add a small buffer to avoid tripping 429s.
_MIN_INTERVAL = 1.15
_last_call = 0.0


async def _wait_for_rate_limit():
    """Block until enough time has passed since the last API call."""
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < _MIN_INTERVAL:
        await asyncio.sleep(_MIN_INTERVAL - elapsed)
    _last_call = time.time()


async def _request(
    session: aiohttp.ClientSession,
    url: str,
    params: Dict[str, Any],
    max_retries: int = 3,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    GET with retry. Backoff is tuned per error type:
      - 429 (rate limit): 2s, 5s, 10s  — clears quickly
      - 5xx (server err): 5s, 15s, 30s — S2 outages are often sustained
      - timeout:          5s, 10s, 15s
    """
    headers = {"x-api-key": API_KEY} if API_KEY else {}

    for attempt in range(max_retries + 1):
        await _wait_for_rate_limit()
        try:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()

                if resp.status in (429, 500, 502, 503, 504) and attempt < max_retries:
                    wait = (
                        [2, 5, 10][attempt] if resp.status == 429
                        else [5, 15, 30][attempt]
                    )
                    print(f"  S2: HTTP {resp.status}, retry {attempt + 1}/{max_retries} in {wait}s")
                    await asyncio.sleep(wait)
                    continue

                # Non-retryable or retries exhausted
                body = await resp.text()
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history,
                    status=resp.status, message=body[:200],
                )
        except asyncio.TimeoutError:
            if attempt < max_retries:
                wait = 5 * (attempt + 1)
                print(f"  S2: timeout, retry {attempt + 1}/{max_retries} in {wait}s")
                await asyncio.sleep(wait)
                continue
            raise

    raise RuntimeError("S2: exhausted retries")


async def paper_search(
    session: aiohttp.ClientSession,
    query: str = None,
    limit: int = 100,
    offset: int = 0,
    fields: Optional[str] = None,
    year: Optional[str] = None,
    venue: Optional[str] = None,
    open_access: Optional[bool] = None,
    fields_of_study: Optional[List[str]] = None,
    min_citation_count: Optional[int] = None,
    publication_types: Optional[List[str]] = None,
    **_kwargs,
) -> Dict[str, Any]:
    """Search S2 for papers. Returns {"data": [...], "total": N, ...}."""
    params: Dict[str, Any] = {
        "offset": offset,
        "limit": limit,
        "fields": fields or FIELDS,
    }
    if query:
        params["query"] = query
    if year:
        params["year"] = year
    if venue:
        params["venue"] = venue
    if min_citation_count:
        params["minCitationCount"] = str(min_citation_count)
    if open_access is True:
        params["openAccessPdf"] = "openAccessPdf"
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)
    if publication_types:
        params["publicationTypes"] = ",".join(publication_types)

    return await _request(session, f"{BASE_URL}/graph/v1/paper/search", params)
