"""
Client for interacting with the Europe PMC RESTful API.

This module provides functions to search for publications and retrieve
details from Europe PMC, handling pagination automatically.
"""

import aiohttp
import asyncio  # Typically not needed in the library code, but can be useful for example usage or specific delays
from typing import Optional, List, Dict, Any

EUROPE_PMC_API_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"


async def _europe_pmc_api_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Internal helper to make a GET request to the Europe PMC API.
    
    Args:
        session: aiohttp client session
        endpoint: API endpoint path
        params: Query parameters
        timeout: Request timeout in seconds (default: 30)
    """
    url = f"{EUROPE_PMC_API_BASE_URL}{endpoint}"
    
    # Add timeout to prevent hanging
    timeout_obj = aiohttp.ClientTimeout(total=timeout)

    try:
        async with session.get(url, params=params, timeout=timeout_obj) as response:
            if response.status != 200:
                error_text = await response.text()
                # Attempt to parse error_text if it's JSON, otherwise use raw text
                try:
                    error_json = await response.json()  # Try to get JSON error details
                    error_detail = error_json.get("error", error_text)
                except aiohttp.ContentTypeError:  # Not JSON
                    error_detail = error_text
                except Exception:  # Fallback for other JSON parsing errors
                    error_detail = error_text

                raise ValueError(f"Europe PMC API request failed: HTTP {response.status}, URL: {response.url}, Error: {error_detail}")
            try:
                return await response.json()
            except aiohttp.ContentTypeError:
                # Handle cases where response is not JSON, though API docs say it should be.
                error_text = await response.text()
                raise ValueError(f"Europe PMC API response was not valid JSON. URL: {response.url}, Response: {error_text}")
    except asyncio.TimeoutError:
        raise ValueError(f"Europe PMC API request timed out after {timeout} seconds. URL: {url}")


async def search_europe_pmc_papers(
    session: aiohttp.ClientSession,
    query: str,
    page_size: int = 100,
    result_type: str = "lite",
    sort: Optional[str] = None,
    max_results: Optional[int] = 100,
    max_pages: int = 100,
) -> List[Dict[str, Any]]:
    """
    Queries Europe PMC for papers and handles pagination to return all results.

    Args:
        session: An aiohttp.ClientSession object.
        query: The search query string (e.g., "covid OR sars-cov-2").
        page_size: Number of results per page. Min: 1, Max: 1000. Default: 100.
                   The API default is 25.
        result_type: The type of results to return. Options include:
                     'idlist': returns a list of IDs and sources.
                     'lite': returns key metadata (API default).
                     'core': returns full metadata including abstract.
        sort: Optional sorting parameter. Examples:
              'P_PDATE_D desc' (publication date descending),
              'CITED desc' (citation count descending).
              The API documentation also mentions query-embedded sort like 'malaria sort_date:y',
              which should be part of the `query` string itself if used.
        max_results: Maximum number of results to return (stops pagination early).
        max_pages: Maximum number of pages to fetch (safety limit, default: 100).

    Returns:
        A list of dictionaries, where each dictionary represents a paper's metadata.
        Returns an empty list if no results are found or in case of an error before fetching any results.
    """
    all_results: List[Dict[str, Any]] = []
    current_cursor_mark = "*"  # Initial cursorMark for the first page
    has_more_pages = True

    if not (1 <= page_size <= 1000):
        print(f"⚠️  Warning: page_size {page_size} is outside the valid range (1-1000). Adjusting to a default of 100.")
        page_size = 100

    request_count = 0
    
    print(f"🔍 Europe PMC query: {query}")
    print(f"📄 Page size: {page_size}, Result type: {result_type}")

    while has_more_pages:
        # Safety check: prevent infinite loops
        if request_count >= max_pages:
            print(f"⚠️  Reached maximum page limit ({max_pages}). Stopping pagination.")
            break
            
        # Safety check: stop if we've reached max_results
        if max_results and len(all_results) >= max_results:
            print(f"✓ Reached desired number of results ({max_results}). Stopping pagination.")
            all_results = all_results[:max_results]
            break
        params: Dict[str, Any] = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "cursorMark": current_cursor_mark,
            "resultType": result_type,
        }
        if sort:
            params["sort"] = sort

        try:
            print(f"📡 Fetching page {request_count + 1} (cursor: {current_cursor_mark[:20] if len(current_cursor_mark) > 20 else current_cursor_mark}...)")
            response_data = await _europe_pmc_api_request(session, "/search", params)
            request_count += 1
        except ValueError as e:
            print(f"❌ Error during API request to Europe PMC: {e}")
            break  # Stop pagination on error

        result_list_data = response_data.get("resultList", {})
        results_this_page = result_list_data.get("result", [])
        hit_count = response_data.get("hitCount", 0)
        
        # Log total available results on first page
        if request_count == 1:
            print(f"📊 Europe PMC found {hit_count} total results")

        # Ensure results_this_page is always a list
        if not isinstance(results_this_page, list):
            results_this_page = [results_this_page] if results_this_page else []

        print(f"   ✓ Got {len(results_this_page)} results on this page (total so far: {len(all_results) + len(results_this_page)})")

        if results_this_page:
            all_results.extend(results_this_page)

        next_cursor_mark = response_data.get("nextCursorMark")

        # Pagination termination conditions based on Europe PMC documentation:
        # - No nextCursorMark is provided.
        # - nextCursorMark is the same as the current_cursor_mark.
        # - No results were returned on this page (an additional safeguard).
        if not next_cursor_mark or next_cursor_mark == current_cursor_mark or not results_this_page:
            print(f"✓ Pagination complete (no more pages)")
            has_more_pages = False
        else:
            current_cursor_mark = next_cursor_mark

        # Add a small delay to be polite to the API, especially in long pagination loops.
        # Europe PMC doesn't specify strict public rate limits for search, but it's good practice.
        if has_more_pages and request_count % 10 == 0:
            await asyncio.sleep(0.2)

    print(f"✓ Europe PMC search complete: {len(all_results)} results returned")
    return all_results
