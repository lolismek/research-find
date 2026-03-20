"""
Client for interacting with the PubMed API via NCBI E-utilities.

This module provides functions to search for publications and retrieve
details from PubMed, handling pagination automatically using the
E-utilities API system.

Based on: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import asyncio

import aiohttp
from typing import Optional, List, Dict, Any, Union
import xml.etree.ElementTree as ET

PUBMED_API_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


async def _pubmed_api_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> Union[Dict[str, Any], str]:
    """GET request to E-utilities with retry on 429 (rate limit)."""

    url = f"{PUBMED_API_BASE_URL}/{endpoint}"

    default_params = {
        "tool": "python_pubmed_client",
        "email": "pranav@sedona.health",
    }

    if params:
        default_params.update(params)

    for attempt in range(max_retries + 1):
        async with session.get(url, params=default_params) as response:
            if response.status == 429 and attempt < max_retries:
                wait = 2 * (attempt + 1)
                print(f"PubMed rate limit hit, waiting {wait}s...")
                await asyncio.sleep(wait)
                continue

            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"PubMed API request failed: HTTP {response.status}, " f"URL: {response.url}, Error: {error_text}")

            content_type = response.headers.get("content-type", "").lower()

            if "json" in content_type:
                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    error_text = await response.text()
                    raise ValueError(f"PubMed API response was not valid JSON. " f"URL: {response.url}, Response: {error_text}")
            else:
                return await response.text()


def _parse_esearch_xml(xml_response: str) -> Dict[str, Any]:
    """Parse XML response from ESearch to extract IDs and metadata."""
    try:
        root = ET.fromstring(xml_response)

        # Extract basic search information
        count = root.find(".//Count")
        ret_max = root.find(".//RetMax")
        ret_start = root.find(".//RetStart")

        # Extract PMIDs
        id_list = root.find(".//IdList")
        pmids = []
        if id_list is not None:
            pmids = [id_elem.text for id_elem in id_list.findall("Id")]

        # Extract translation stack and query translation if available
        translation_set = root.find(".//TranslationSet")
        query_translation = None
        if translation_set is not None:
            trans = translation_set.find(".//Translation/To")
            if trans is not None:
                query_translation = trans.text

        return {"count": int(count.text) if count is not None else 0, "retmax": int(ret_max.text) if ret_max is not None else 0, "retstart": int(ret_start.text) if ret_start is not None else 0, "pmids": pmids, "query_translation": query_translation}
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse ESearch XML response: {e}")


def _parse_esummary_xml(xml_response: str) -> List[Dict[str, Any]]:
    """Parse XML response from ESummary to extract article summaries."""
    try:
        root = ET.fromstring(xml_response)
        summaries = []

        for doc_sum in root.findall(".//DocSum"):
            summary = {}

            # Get PMID
            pmid_elem = doc_sum.find("Id")
            if pmid_elem is not None:
                summary["pmid"] = pmid_elem.text

            # Parse all items in the summary
            for item in doc_sum.findall("Item"):
                name = item.get("Name", "")
                item_type = item.get("Type", "")

                if item_type == "List":
                    # Handle list items (like authors, articleids)
                    list_items = []
                    for sub_item in item.findall("Item"):
                        list_items.append(sub_item.text or "")
                        # Promote named sub-items (e.g. ArticleIds/pmc,
                        # ArticleIds/doi) to top-level fields so that
                        # downstream code can access them directly.
                        sub_name = sub_item.get("Name", "")
                        if sub_name and sub_item.text:
                            key = sub_name.lower()
                            if key not in summary:
                                summary[key] = sub_item.text
                    summary[name.lower()] = list_items
                else:
                    # Handle regular items
                    summary[name.lower()] = item.text or ""

            summaries.append(summary)

        return summaries
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse ESummary XML response: {e}")


def _parse_efetch_abstracts(xml_response: str) -> Dict[str, str]:
    """Parse EFetch XML to extract abstracts keyed by PMID."""
    abstracts = {}
    try:
        root = ET.fromstring(xml_response)
        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None:
                continue
            pmid = pmid_elem.text

            # Abstract can have multiple AbstractText elements (structured abstract)
            abstract_parts = []
            for abs_text in article.findall(".//Abstract/AbstractText"):
                label = abs_text.get("Label")
                text = abs_text.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)

            if abstract_parts:
                abstracts[pmid] = " ".join(abstract_parts)
    except ET.ParseError:
        pass  # Best-effort — don't fail the whole pipeline
    return abstracts


async def _fetch_abstracts_for_pmids(
    session: aiohttp.ClientSession,
    pmids: List[str],
) -> Dict[str, str]:
    """Fetch abstracts for a list of PMIDs using EFetch. Returns {pmid: abstract}."""
    if not pmids:
        return {}

    # EFetch supports up to 200 IDs per request
    all_abstracts = {}
    batch_size = 200
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        try:
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "rettype": "abstract",
                "retmode": "xml",
            }
            response = await _pubmed_api_request(session, "efetch.fcgi", fetch_params)
            abstracts = _parse_efetch_abstracts(response)
            all_abstracts.update(abstracts)
        except Exception as e:
            print(f"Warning: EFetch abstract retrieval failed for batch: {e}")
    return all_abstracts


async def search_pubmed_papers(
    session: aiohttp.ClientSession,
    query: str,
    max_results: Optional[int] = None,
    return_type: str = "summary",
    sort: Optional[str] = None,
    date_range: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search PubMed for papers and return results with automatic pagination.

    Args:
        session: An aiohttp.ClientSession object.
        query: The search query string (e.g., "covid-19 AND vaccine").
        max_results: Maximum number of results to return.
                    If None, returns all results.
        return_type: Type of data to return:
                    'summary': article summaries (default)
                    'ids': just PMIDs
        sort: Sorting parameter. Options include:
              'relevance' (default), 'pub_date', 'author', 'journal'
        date_range: Optional date filtering with 'mindate' and 'maxdate'
                   in YYYY/MM/DD format

    Returns:
        A list of dictionaries containing paper metadata.
    """
    all_results: List[Dict[str, Any]] = []
    batch_size = 500  # E-utilities recommended batch size
    ret_start = 0

    # First, get the count of total results using ESearch
    search_params = {"db": "pubmed", "term": query, "rettype": "count", "retmode": "xml"}

    if sort:
        search_params["sort"] = sort

    if date_range:
        if "mindate" in date_range:
            search_params["mindate"] = date_range["mindate"]
        if "maxdate" in date_range:
            search_params["maxdate"] = date_range["maxdate"]

    # Get total count
    try:
        count_response = await _pubmed_api_request(session, "esearch.fcgi", search_params)
        count_data = _parse_esearch_xml(count_response)
        total_count = count_data["count"]

        if total_count == 0:
            return []

        # Determine how many results to actually fetch
        if max_results is not None:
            total_to_fetch = min(max_results, total_count)
        else:
            total_to_fetch = total_count

        print(f"Found {total_count} results, fetching {total_to_fetch}")

    except ValueError as e:
        print(f"Error getting result count from PubMed: {e}")
        return []

    # Now fetch results in batches
    while ret_start < total_to_fetch:
        current_batch_size = min(batch_size, total_to_fetch - ret_start)

        # Update search parameters for this batch
        search_params.update({"retmax": current_batch_size, "retstart": ret_start, "rettype": "uilist", "retmode": "xml"})

        try:
            # Get PMIDs for this batch
            search_response = await _pubmed_api_request(session, "esearch.fcgi", search_params)
            search_data = _parse_esearch_xml(search_response)
            pmids = search_data["pmids"]

            if not pmids:
                break

            if return_type == "ids":
                # Just return PMID data
                for pmid in pmids:
                    all_results.append({"pmid": pmid})
            else:
                # Get summaries for these PMIDs
                summary_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}

                summary_response = await _pubmed_api_request(session, "esummary.fcgi", summary_params)
                summaries = _parse_esummary_xml(summary_response)
                all_results.extend(summaries)

            ret_start += current_batch_size

            print(f"Fetched {len(all_results)} / {total_to_fetch} results")

        except ValueError as e:
            print(f"Error during PubMed API request: {e}")
            break

    # Enrich summaries with abstracts via EFetch (ESummary doesn't include them).
    # 1s delay avoids PubMed rate limit (3 req/sec unauthenticated).
    if return_type == "summary" and all_results:
        pmids = [r["pmid"] for r in all_results if r.get("pmid")]
        if pmids:
            await asyncio.sleep(1.0)
            print(f"Fetching abstracts for {len(pmids)} PubMed papers...")
            abstracts = await _fetch_abstracts_for_pmids(session, pmids)
            enriched = 0
            for result in all_results:
                pmid = result.get("pmid")
                if pmid and pmid in abstracts:
                    result["abstract"] = abstracts[pmid]
                    enriched += 1
            print(f"  Got abstracts for {enriched}/{len(pmids)} papers")

    return all_results


