"""Unified paper resolution: any URL/DOI/title -> Paper with embedding."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Optional

import aiohttp

from models.paper import Paper
from services.arxiv import resolve_arxiv_url, extract_arxiv_id
from services.grobid import process_pdf_from_url

# Semantic Scholar API config
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
S2_FIELDS = (
    "paperId,externalIds,url,title,abstract,venue,year,"
    "citationCount,influentialCitationCount,isOpenAccess,"
    "openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,authors,embedding"
)
# Search endpoint times out with embedding field — fetch it separately
S2_SEARCH_FIELDS = (
    "paperId,externalIds,url,title,abstract,venue,year,"
    "citationCount,influentialCitationCount,isOpenAccess,"
    "openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,authors"
)

_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s]+)")
_S2_URL_RE = re.compile(r"semanticscholar\.org/paper/[^/]*/([a-f0-9]{40})", re.I)


async def _s2_get(
    session: aiohttp.ClientSession,
    path: str,
    fields: str | None = None,
) -> Optional[dict]:
    """GET from S2 API, returning None on 404/timeout."""
    url = f"{S2_BASE}{path}"
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    params = {"fields": fields or S2_FIELDS}
    try:
        async with session.get(
            url, headers=headers, params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status in (404, 504):
                return None
            if resp.status != 200:
                text = await resp.text()
                print(f"[resolver] S2 API error {resp.status}: {text[:200]}")
                return None
            return await resp.json()
    except asyncio.TimeoutError:
        print(f"[resolver] S2 request timed out: {path}")
        return None


async def _s2_search(session: aiohttp.ClientSession, query: str) -> Optional[dict]:
    """Search S2 for a paper by title, return best match."""
    url = f"{S2_BASE}/paper/search"
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    params = {"query": query, "limit": 5, "fields": S2_SEARCH_FIELDS}
    async with session.get(url, headers=headers, params=params) as resp:
        if resp.status != 200:
            return None
        data = await resp.json()
    papers = data.get("data", [])
    if not papers:
        return None
    # Pick best title match
    query_lower = query.lower().strip()
    for p in papers:
        if p.get("title", "").lower().strip() == query_lower:
            return p
    return papers[0]


async def resolve_paper(input_str: str, enrich_grobid: bool = False) -> Paper:
    """Resolve any input (DOI, arXiv URL, S2 URL, title) to a Paper.

    Resolution order:
    1. DOI pattern -> S2 /paper/DOI:{doi} (includes embedding)
    2. arXiv URL/ID -> arXiv API, then S2 for embedding
    3. S2 URL -> extract paper ID, call S2
    4. Title string -> search S2, pick best match

    After resolution, optionally process PDF through GROBID.
    """
    input_str = input_str.strip().strip('"').strip("'")
    print(f"[resolver] Resolving: {input_str}")

    async with aiohttp.ClientSession() as session:
        paper = None

        # 1. DOI
        doi_match = _DOI_RE.search(input_str)
        if doi_match:
            doi = doi_match.group(1).rstrip(".")
            print(f"[resolver] Detected DOI: {doi}")
            data = await _s2_get(session, f"/paper/DOI:{doi}")
            if data:
                paper = Paper.from_s2_dict(data)
            else:
                paper = Paper(doi=doi, title=input_str, source="doi_only")

        # 2. arXiv
        if paper is None and ("arxiv" in input_str.lower() or extract_arxiv_id(input_str)):
            print(f"[resolver] Detected arXiv input")
            paper = await resolve_arxiv_url(input_str)
            # Try to get S2 embedding
            if paper.arxiv_id:
                data = await _s2_get(session, f"/paper/ARXIV:{paper.arxiv_id}")
                if data:
                    s2_paper = Paper.from_s2_dict(data)
                    paper.paper_id = s2_paper.paper_id
                    paper.doi = paper.doi or s2_paper.doi
                    paper.embedding = s2_paper.embedding
                    paper.citation_count = s2_paper.citation_count or paper.citation_count
                    paper.fields_of_study = s2_paper.fields_of_study or paper.fields_of_study
                    paper.is_open_access = s2_paper.is_open_access
                    if not paper.abstract and s2_paper.abstract:
                        paper.abstract = s2_paper.abstract

        # 3. S2 URL
        if paper is None:
            s2_match = _S2_URL_RE.search(input_str)
            if s2_match:
                paper_id = s2_match.group(1)
                print(f"[resolver] Detected S2 paper ID: {paper_id}")
                data = await _s2_get(session, f"/paper/{paper_id}")
                if data:
                    paper = Paper.from_s2_dict(data)

        # 4. Title search
        if paper is None:
            print(f"[resolver] Searching S2 by title: {input_str}")
            data = await _s2_search(session, input_str)
            if data:
                paper = Paper.from_s2_dict(data)
                print(f"[resolver] Found: {paper.title}")
                # Fetch embedding via single-paper endpoint
                if paper.paper_id and not paper.embedding:
                    full = await _s2_get(session, f"/paper/{paper.paper_id}")
                    if full and isinstance(full.get("embedding"), dict):
                        paper.embedding = full["embedding"].get("vector")
            else:
                raise ValueError(f"Could not resolve paper: {input_str}")

        # Optional GROBID enrichment
        if enrich_grobid and paper.pdf_url:
            try:
                tei_data = await process_pdf_from_url(paper.pdf_url)
                Paper.from_grobid_tei(tei_data, base_paper=paper)
            except Exception as e:
                print(f"GROBID enrichment failed: {e}")

    return paper
