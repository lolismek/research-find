"""Tool handler implementations for the Claude agent."""

from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime
from typing import Any

import aiohttp

from ingestion.evidence_service import search_papers as _search_papers_multi
from services.paper_resolver import resolve_paper, fetch_s2_references
from services.embeddings import schedule_embedding, schedule_concept_embeddings
from services.concept_extractor import collect_raw_concepts, normalize_concepts
from services.neo4j_store import (
    store_paper, get_paper, list_papers, search_similar,
    store_concepts, create_covers_edges, create_added_edge,
    store_s2_ref_ids, reconcile_cites_edges, update_related_to,
    list_concepts_without_embeddings, create_follows_edge,
)
from background.rss_monitor import get_monitor

_background_tasks: set[asyncio.Task] = set()


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


async def handle_add_paper(
    identifier: str,
    process_pdf: bool = False,
    source: str = "manual",
    _user_phone: str | None = None,
) -> dict[str, Any]:
    """Resolve a paper and add it to Neo4j, then enrich graph in background."""
    # Synchronous: resolve + store (user waits for confirmation)
    paper = await resolve_paper(identifier, enrich_grobid=process_pdf)
    paper.added_at = datetime.utcnow()
    merge_key = await store_paper(paper)
    key_type = "doi" if paper.doi else ("arxiv_id" if paper.arxiv_id else "title")

    # Fire background enrichment
    task = asyncio.create_task(
        _enrich_paper_graph(paper, merge_key, key_type, source, _user_phone)
    )
    _background_tasks.add(task)

    def _on_done(t):
        _background_tasks.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc:
            import traceback
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    task.add_done_callback(_on_done)

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


async def _enrich_paper_graph(
    paper,
    merge_key: str,
    key_type: str,
    source: str,
    user_phone: str | None,
) -> None:
    """Background task: extract concepts, create edges, fetch references, embed."""
    print(f"[enrich] Starting enrichment for: {paper.title[:60]} (pdf_url={paper.pdf_url}, keywords={paper.keywords})")

    # 0. Resolve PDF URL if missing, then run GROBID for keywords
    if not paper.keywords:
        # PDF URL fallback chain: S2 openAccessPdf → arXiv → Unpaywall
        pdf_url = paper.pdf_url
        if not pdf_url and paper.doi:
            try:
                from ingestion.unpaywall import check_doi_oa_status
                oa_info = await check_doi_oa_status(paper.doi)
                if oa_info.get("is_oa") and oa_info.get("url"):
                    pdf_url = oa_info["url"]
                    print(f"[enrich] Unpaywall found PDF: {pdf_url[:80]}")
            except Exception as e:
                print(f"[enrich] Unpaywall lookup failed (non-fatal): {e}")

        if pdf_url:
            try:
                from services.grobid import process_pdf_from_url
                from models.paper import Paper as PaperModel
                tei_data = await process_pdf_from_url(pdf_url)
                PaperModel.from_grobid_tei(tei_data, base_paper=paper)
                # Persist GROBID data + resolved pdf_url to Neo4j
                from services.neo4j_store import _get_driver, _session_kwargs
                driver = _get_driver()
                if key_type == "doi":
                    match = "MATCH (p:Paper {doi: $key})"
                elif key_type == "arxiv_id":
                    match = "MATCH (p:Paper {arxiv_id: $key})"
                else:
                    match = "MATCH (p:Paper {title: $key})"
                query = (
                    f"{match} SET p.keywords = $keywords, "
                    "p.grobid_abstract = $grobid_abstract, p.pdf_url = $pdf_url"
                )
                async with driver.session(**_session_kwargs()) as session:
                    await session.run(
                        query, key=merge_key,
                        keywords=paper.keywords,
                        grobid_abstract=paper.grobid_abstract,
                        pdf_url=pdf_url,
                    )
                print(f"[enrich] GROBID: {len(paper.keywords or [])} keywords extracted")
            except Exception as e:
                print(f"[enrich] GROBID enrichment failed (non-fatal): {e}")
        else:
            print(f"[enrich] No PDF URL available, skipping GROBID")

    # 1. Extract and normalize concepts
    try:
        raw_concepts = collect_raw_concepts(paper)
        if raw_concepts:
            concepts = await normalize_concepts(raw_concepts)
        else:
            concepts = []
    except Exception as e:
        print(f"[enrich] Concept extraction failed: {e}")
        concepts = []

    # 2. Store Concept nodes + COVERS edges
    try:
        if concepts:
            await store_concepts(concepts)
            await create_covers_edges(merge_key, key_type, concepts)
    except Exception as e:
        print(f"[enrich] COVERS edges failed: {e}")

    # 3. Update RELATED_TO weights
    try:
        if concepts:
            await update_related_to(concepts)
    except Exception as e:
        print(f"[enrich] RELATED_TO failed: {e}")

    # 4. Create ADDED edge (User -> Paper)
    try:
        if user_phone:
            await create_added_edge(user_phone, merge_key, key_type, source)
    except Exception as e:
        print(f"[enrich] ADDED edge failed: {e}")

    # 5. Fetch S2 references, store ref IDs, reconcile CITES edges both directions
    try:
        if paper.paper_id:
            async with aiohttp.ClientSession() as session:
                refs = await fetch_s2_references(session, paper.paper_id)
            ref_ids = [r["paperId"] for r in refs if r.get("paperId")]
            if ref_ids:
                await store_s2_ref_ids(merge_key, key_type, ref_ids)
            await reconcile_cites_edges(paper.paper_id)
    except Exception as e:
        print(f"[enrich] CITES edges failed: {e}")

    # 6. Schedule paper embedding
    try:
        schedule_embedding(paper)
    except Exception as e:
        print(f"[enrich] Paper embedding scheduling failed: {e}")

    # 7. Schedule concept embeddings for any missing
    try:
        missing = await list_concepts_without_embeddings()
        if missing:
            schedule_concept_embeddings(missing)
    except Exception as e:
        print(f"[enrich] Concept embedding scheduling failed: {e}")

    print(f"[enrich] Done enriching: {paper.title[:60]}")


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


async def handle_fetch_arxiv_papers(
    category: str | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Fetch latest arXiv papers on demand."""
    monitor = get_monitor()
    return await monitor.fetch_on_demand(category, top_n=top_n)


async def handle_set_notification_time(
    hour: int,
    minute: int = 0,
) -> dict[str, Any]:
    """Set the daily digest notification time."""
    if not (0 <= hour <= 23):
        return {"error": "hour must be 0-23"}
    if not (0 <= minute <= 59):
        return {"error": "minute must be 0-59"}
    monitor = get_monitor()
    monitor.set_notification_time(hour, minute)
    t = monitor.get_notification_time()
    return {
        "status": "notification_time_updated",
        "notification_time": f"{t.hour:02d}:{t.minute:02d}",
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


async def handle_follow_concept(
    concept_name: str,
    _user_phone: str | None = None,
) -> dict[str, Any]:
    """Follow a research concept."""
    from services.concept_extractor import normalize_text
    name = normalize_text(concept_name)
    if not name:
        return {"error": "Empty concept name"}
    if not _user_phone:
        return {"error": "No user identity — please log in with a phone number"}

    await create_follows_edge(_user_phone, name, explicit=True)
    return {"status": "following", "concept": name}


# Tool dispatch map
TOOL_HANDLERS = {
    "search_papers": handle_search_papers,
    "add_paper": handle_add_paper,
    "get_paper_details": handle_get_paper_details,
    "list_stored_papers": handle_list_stored_papers,
    "fetch_arxiv_papers": handle_fetch_arxiv_papers,
    "set_notification_time": handle_set_notification_time,
    "find_similar_papers": handle_find_similar_papers,
    "follow_concept": handle_follow_concept,
}


async def dispatch_tool(
    name: str, args: dict[str, Any], user_phone: str | None = None,
) -> str:
    """Dispatch a tool call and return the JSON result string.

    If the handler accepts `_user_phone`, inject it automatically.
    """
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        sig = inspect.signature(handler)
        if "_user_phone" in sig.parameters:
            args = {**args, "_user_phone": user_phone}
        result = await handler(**args)
        return json.dumps(result, default=str)
    except Exception as e:
        import traceback
        traceback.print_exc()
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
