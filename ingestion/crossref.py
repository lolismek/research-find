"""
Crossref API client for citation enrichment.

Used by evidence_service to backfill citation counts, funder info,
and license data for papers that have DOIs.

Rate limit: ~50 req/sec with polite pool (we use mailto header).
API docs: https://api.crossref.org/swagger-ui/index.html
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp

CROSSREF_BASE_URL = "https://api.crossref.org/works"
CROSSREF_MAILTO = "pranav@sedona.health"


async def fetch_crossref_metadata(
    session: aiohttp.ClientSession,
    doi: str,
    timeout: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata for a single DOI from Crossref.

    Returns dict with citation_count, reference_count, funders, license,
    or None on failure.
    """
    url = f"{CROSSREF_BASE_URL}/{doi}"
    headers = {"User-Agent": f"SedonaHealth/1.0 (mailto:{CROSSREF_MAILTO})"}

    try:
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status == 404:
                return None
            if response.status == 429:
                return None  # caller can retry
            response.raise_for_status()
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None

    msg = data.get("message", {})
    funders = [f.get("name") for f in msg.get("funder", []) if f.get("name")]
    licenses = [lic.get("URL") for lic in msg.get("license", []) if lic.get("URL")]

    return {
        "citation_count": msg.get("is-referenced-by-count", 0),
        "reference_count": len(msg.get("reference", [])),
        "references": msg.get("reference", []),
        "funders": funders or None,
        "license": licenses[0] if licenses else None,
    }


async def enrich_papers_with_crossref(
    papers: List[Dict[str, Any]],
    concurrency: int = 8,
) -> int:
    """
    Enrich a list of paper dicts in-place with Crossref citation data.

    Only enriches papers that have a DOI and citation_count == 0
    (i.e., papers from PubMed/bioRxiv that lack citation info).

    Args:
        papers: List of normalized paper dicts (mutated in-place).
        concurrency: Max parallel requests to Crossref.

    Returns:
        Number of papers successfully enriched.
    """
    to_enrich = [p for p in papers if p.get("doi") and not p.get("citation_count")]
    if not to_enrich:
        return 0

    enriched = 0
    sem = asyncio.Semaphore(concurrency)

    async def _enrich_one(session: aiohttp.ClientSession, paper: Dict[str, Any]) -> bool:
        async with sem:
            meta = await fetch_crossref_metadata(session, paper["doi"])
            if meta and meta["citation_count"] > 0:
                paper["citation_count"] = meta["citation_count"]
                if meta.get("funders"):
                    paper["funders"] = meta["funders"]
                if meta.get("license"):
                    paper["license"] = meta["license"]
                if meta.get("references"):
                    paper["crossref_references"] = meta["references"]
                return True
            return False

    async with aiohttp.ClientSession() as session:
        tasks = [_enrich_one(session, p) for p in to_enrich]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        enriched = sum(1 for r in results if r is True)

    return enriched
