"""Tool handler implementations for the Claude agent."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ingestion.evidence_service import search_papers as _search_papers_multi
from services.paper_resolver import resolve_paper
from services.neo4j_store import store_paper, get_paper, list_papers, search_similar
from services.arxiv import fetch_arxiv_rss
from background.rss_monitor import get_monitor


async def handle_search_papers(query: str, limit: int = 10) -> dict[str, Any]:
    """Search across S2/PubMed/EPMC and return formatted results."""
    result = await _search_papers_multi(query, limit=limit)
    papers = result.get("papers", [])[:limit]

    formatted = []
    for i, p in enumerate(papers, 1):
        formatted.append({
            "rank": i,
            "title": p.get("title"),
            "authors": _format_authors(p.get("authors")),
            "year": p.get("year"),
            "venue": p.get("venue"),
            "doi": p.get("doi"),
            "citations": p.get("citation_count") or p.get("citationCount", 0),
            "is_open_access": p.get("is_open_access"),
            "pdf_url": p.get("pdf_url"),
            "url": p.get("url"),
            "abstract": (p.get("abstract") or "")[:300],
            "source": p.get("source"),
        })

    return {
        "query": query,
        "total_found": result.get("total_found", 0),
        "showing": len(formatted),
        "papers": formatted,
    }


async def handle_add_paper(identifier: str, process_pdf: bool = False) -> dict[str, Any]:
    """Resolve a paper and add it to Neo4j."""
    paper = await resolve_paper(identifier, enrich_grobid=process_pdf)
    paper.added_at = datetime.utcnow()
    merge_key = await store_paper(paper)

    return {
        "status": "added",
        "merge_key": merge_key,
        "title": paper.title,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "authors": [a.name for a in paper.authors[:5]],
        "year": paper.year,
        "citation_count": paper.citation_count,
        "has_embedding": paper.embedding is not None,
        "has_grobid": paper.grobid_abstract is not None,
    }


async def handle_get_paper_details(
    doi: str | None = None,
    arxiv_id: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Get paper details from Neo4j."""
    paper = await get_paper(doi=doi, arxiv_id=arxiv_id, title=title)
    if not paper:
        return {"error": "Paper not found in database"}

    return {
        "title": paper.title,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "authors": [a.name for a in paper.authors],
        "year": paper.year,
        "venue": paper.venue,
        "abstract": paper.abstract,
        "citation_count": paper.citation_count,
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "is_open_access": paper.is_open_access,
        "fields_of_study": paper.fields_of_study,
        "keywords": paper.keywords,
        "has_embedding": paper.embedding is not None,
        "has_grobid": paper.grobid_abstract is not None,
        "added_at": str(paper.added_at) if paper.added_at else None,
    }


async def handle_list_stored_papers(limit: int = 20) -> dict[str, Any]:
    """List papers in the database."""
    papers = await list_papers(limit=limit)
    formatted = []
    for p in papers:
        formatted.append({
            "title": p.title,
            "doi": p.doi,
            "arxiv_id": p.arxiv_id,
            "year": p.year,
            "authors": [a.name for a in p.authors[:3]],
            "citation_count": p.citation_count,
            "added_at": str(p.added_at) if p.added_at else None,
        })

    return {"count": len(formatted), "papers": formatted}


async def handle_monitor_arxiv_topic(
    category: str,
    schedule: str = "immediate",
    top_n: int = 5,
) -> dict[str, Any]:
    """Start monitoring an arXiv category."""
    monitor = get_monitor()
    monitor.start_monitoring(category, schedule=schedule, top_n=top_n)
    return {
        "status": "monitoring_started",
        "category": category,
        "schedule": schedule,
        "top_n": top_n,
        "active_monitors": list(monitor.active_categories()),
    }


async def handle_find_similar_papers(
    doi: str | None = None,
    arxiv_id: str | None = None,
    title: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Find similar papers using vector similarity."""
    paper = await get_paper(doi=doi, arxiv_id=arxiv_id, title=title)
    if not paper:
        return {"error": "Reference paper not found in database"}
    if not paper.embedding:
        return {"error": f"Paper '{paper.title}' has no embedding. Re-add it to fetch the embedding."}

    similar = await search_similar(paper.embedding, limit=limit + 1)
    # Filter out the reference paper itself
    results = []
    for p in similar:
        if p.doi == paper.doi and paper.doi:
            continue
        if p.arxiv_id == paper.arxiv_id and paper.arxiv_id:
            continue
        results.append({
            "title": p.title,
            "doi": p.doi,
            "arxiv_id": p.arxiv_id,
            "year": p.year,
            "authors": [a.name for a in p.authors[:3]],
            "citation_count": p.citation_count,
        })
        if len(results) >= limit:
            break

    return {
        "reference_paper": paper.title,
        "similar_count": len(results),
        "similar_papers": results,
    }


# Tool dispatch map
TOOL_HANDLERS = {
    "search_papers": handle_search_papers,
    "add_paper": handle_add_paper,
    "get_paper_details": handle_get_paper_details,
    "list_stored_papers": handle_list_stored_papers,
    "monitor_arxiv_topic": handle_monitor_arxiv_topic,
    "find_similar_papers": handle_find_similar_papers,
}


async def dispatch_tool(name: str, args: dict[str, Any]) -> str:
    """Dispatch a tool call and return the JSON result string."""
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = await handler(**args)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _format_authors(authors: Any) -> list[str]:
    """Normalize authors from various formats to a list of name strings."""
    if not authors:
        return []
    result = []
    for a in authors[:5]:
        if isinstance(a, dict):
            result.append(a.get("name", "Unknown"))
        elif isinstance(a, str):
            result.append(a)
    return result
