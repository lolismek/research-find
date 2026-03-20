"""
Unpaywall API client.

Looks up Open Access status for DOIs. Used by evidence_service
to find legal OA versions of paywalled papers.

Please limit use to 100,000 calls per day.
"""

from __future__ import annotations

import os
import asyncio
from http import HTTPStatus
from typing import Any

import aiohttp

UNPAYWALL_BASE_URL = "https://api.unpaywall.org/v2/"
UNPAYWALL_TIMEOUT = float(os.environ.get("UNPAYWALL_TIMEOUT", "10.0"))
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "pranav@sedona.health")


async def make_request(url: str, session: aiohttp.ClientSession) -> dict[str, Any]:
    """Make a request to the Unpaywall API with retry logic."""
    try:
        for attempt in range(5):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=UNPAYWALL_TIMEOUT)) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ServerDisconnectedError:
                if attempt == 4:
                    raise Exception("Server disconnected repeatedly")
                await asyncio.sleep(0.1 * (attempt + 1))
            except aiohttp.ClientResponseError as e:
                if e.status == HTTPStatus.NOT_FOUND:
                    raise Exception(f"DOI not found: {url}")
                elif e.status == HTTPStatus.BAD_REQUEST:
                    raise Exception(f"Bad request: {url}")
                elif e.status == HTTPStatus.UNAUTHORIZED:
                    raise Exception("Invalid or missing API key")
                elif e.status == HTTPStatus.TOO_MANY_REQUESTS:
                    raise Exception("Rate limit exceeded")
                elif e.status in {HTTPStatus.INTERNAL_SERVER_ERROR, HTTPStatus.GATEWAY_TIMEOUT}:
                    if attempt == 4:
                        raise Exception(f"Server error: {e.status}")
                    await asyncio.sleep(0.1 * (attempt + 1))
                else:
                    raise Exception(f"Unexpected error: {e.status}")
    except aiohttp.ClientError as e:
        raise Exception(f"Client error: {str(e)}")


async def check_doi_oa_status(
    doi: str,
    session: aiohttp.ClientSession | None = None,
) -> dict[str, Any]:
    """
    Check the Open Access status of a DOI using Unpaywall.

    Args:
        doi: The DOI to look up.
        session: Optional shared aiohttp session.  When ``None`` a
            temporary session is created (legacy behaviour).

    Returns:
        dict with is_oa, oa_status, best_oa_location, url, title, year, journal_name.
    """

    async def _do(s: aiohttp.ClientSession) -> dict[str, Any]:
        url = f"{UNPAYWALL_BASE_URL}{doi}?email={UNPAYWALL_EMAIL}"
        try:
            data = await make_request(url, s)
            return {
                "is_oa": data.get("is_oa", False),
                "oa_status": data.get("oa_status"),
                "best_oa_location": data.get("best_oa_location"),
                "url": data.get("best_oa_location", {}).get("url") if data.get("best_oa_location") else None,
                "title": data.get("title"),
                "year": data.get("year"),
                "journal_name": data.get("journal_name"),
            }
        except Exception as e:
            return {
                "is_oa": False,
                "oa_status": None,
                "best_oa_location": None,
                "url": None,
                "error": str(e),
            }

    if session is not None:
        return await _do(session)
    async with aiohttp.ClientSession() as s:
        return await _do(s)
