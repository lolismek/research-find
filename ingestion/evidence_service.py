"""Multi-source paper search with deduplication, enrichment, and ranking.

Searches Semantic Scholar, PubMed, and Europe PMC, deduplicates by DOI,
enriches via Unpaywall (OA) and Crossref (citations), then ranks by
journal prestige and log-scaled citation impact.
"""

from __future__ import annotations

import asyncio
import math
import re
from typing import List, Dict, Any, Optional
import aiohttp
from datetime import datetime

from etl.literature.ingestion.semantic_scholar import (
    paper_search as s2_paper_search,
)
from etl.literature.ingestion.europe_pmc import (
    search_europe_pmc_papers,
)
from etl.literature.ingestion.unpaywall import (
    check_doi_oa_status,
)
from etl.literature.ingestion.crossref import (
    enrich_papers_with_crossref,
)
from etl.literature.ingestion.pubmed_api import (
    search_pubmed_papers,
)

# Simple ANSI colors for console logs
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


# ── Constants & utilities ──────────────────────────────────────────────────

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s]+")

VENUE_TIER_1 = {
    "The New England Journal of Medicine",
    "Lancet",
    "JAMA",
    "BMJ",
    "Nature Medicine",
    "Science",
    "Cell",
    "PNAS",
    "Circulation",
}

VENUE_TIER_2 = {
    "Nature",
    "Nature Communications",
    "JAMA Internal Medicine",
    "European Heart Journal",
    "Annals of Internal Medicine",
    "JACC",
}


def _get_current_year() -> int:
    return datetime.utcnow().year


# ── Query helpers ─────────────────────────────────────────────────────────


def _quote_query(query: str) -> str:
    """Wrap multi-word queries in quotes for phrase matching.

    Skips quoting if already quoted or contains Boolean operators.
    """
    q = query.strip()
    if q.startswith('"') and q.endswith('"'):
        return q
    if " AND " in q or " OR " in q:
        return q
    if " " in q:
        return f'"{q}"'
    return q


_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "its",
    "are", "was", "were", "been", "has", "have", "had", "not",
    "but", "also", "can", "may", "will", "should", "could",
    # Generic academic terms
    "supplementation", "treatment", "study", "effect", "effects",
    "role", "review", "analysis", "research", "results",
    "associated", "patients", "clinical", "human", "novel",
    "based", "using", "between", "during", "after", "before",
})


def _extract_key_terms(query: str) -> list[str]:
    """Extract significant terms from a query (>3 chars, not stopwords)."""
    raw = query.strip().strip('"')
    return [
        w for w in raw.lower().split()
        if len(w) > 3 and w not in _STOPWORDS
    ]


def _text_match_score(
    query: str, title: str | None, abstract: str | None,
) -> tuple[float, float]:
    """Score how well title/abstract match the query.

    Returns (title_score, abstract_score) each in [0, 1].
    Exact phrase match gets a bonus.
    """
    terms = _extract_key_terms(query)
    if not terms:
        return (0.0, 0.0)

    phrase = query.strip().strip('"').lower()

    def _score(text: str | None) -> float:
        if not text:
            return 0.0
        lower = text.lower()
        matched = sum(1 for t in terms if t in lower)
        coverage = matched / len(terms)
        # Exact phrase bonus
        if phrase in lower:
            coverage = min(1.0, coverage + 0.5)
        return round(coverage, 4)

    return (_score(title), _score(abstract))


def _filter_irrelevant(
    papers: List[Dict[str, Any]], query: str,
) -> List[Dict[str, Any]]:
    """Drop papers whose title+abstract contain NONE of the key query terms.

    Safety: if filtering would leave <3 papers, skip filtering entirely.
    """
    terms = _extract_key_terms(query)
    if not terms:
        return papers

    kept: List[Dict[str, Any]] = []
    for p in papers:
        combined = (
            (p.get("title") or "") + " " + (p.get("abstract") or "")
        ).lower()
        if any(t in combined for t in terms):
            kept.append(p)

    if len(kept) < 3:
        return papers
    return kept


def _make_dedup_key(paper_id: Optional[str], doi: Optional[str]) -> str:
    """Canonical dedup key: prefer DOI, fall back to paper_id."""
    if doi:
        return f"doi:{doi.lower()}"
    return f"id:{paper_id or 'na'}|doi:{(doi or '').lower()}"


def _add_to_seen(
    seen: Dict[str, Dict[str, Any]],
    key: str,
    norm: Dict[str, Any],
    query: str,
) -> None:
    """Insert or merge a paper into the dedup dict."""
    existing = seen.get(key)
    if not existing:
        seen[key] = norm
    else:
        if query not in existing["matched_terms"]:
            existing["matched_terms"].append(query)


# ── Source normalizers ─────────────────────────────────────────────────────


def _normalize_s2_paper(p: Dict[str, Any], query: str) -> tuple[str, Dict[str, Any]]:
    """Normalize a Semantic Scholar paper to common format."""
    ext = p.get("externalIds") or {}
    paper_id = p.get("paperId")
    doi = ext.get("DOI")
    key = _make_dedup_key(paper_id, doi)
    pdf_url = None
    try:
        oa = p.get("openAccessPdf") or {}
        pdf_url = oa.get("url")
    except Exception:
        pass
    norm = {
        "paper_id": paper_id,
        "doi": doi,
        "url": p.get("url"),
        "title": p.get("title"),
        "abstract": p.get("abstract"),
        "year": p.get("year"),
        "venue": p.get("venue") or (p.get("journal") or {}).get("name"),
        "is_open_access": p.get("isOpenAccess"),
        "open_access_pdf": p.get("openAccessPdf"),
        "pdf_url": pdf_url,
        "pmc_id": None,
        "pdf_status": None,
        "citation_count": p.get("citationCount") or 0,
        "influential_citation_count": p.get("influentialCitationCount") or 0,
        "authors": p.get("authors"),
        "fields_of_study": p.get("fieldsOfStudy") or p.get("s2FieldsOfStudy"),
        "publication_types": p.get("publicationTypes"),
        "matched_terms": [query],
        "source": "semantic_scholar",
    }
    return key, norm


def _extract_doi_from_pubmed(p: Dict[str, Any]) -> Optional[str]:
    """Extract DOI from PubMed ESummary dict."""
    eloc = p.get("elocationid", "")
    if isinstance(eloc, str):
        m = _DOI_RE.search(eloc)
        if m:
            return m.group(0).rstrip(".")
    for aid in (p.get("articleids") or []):
        if isinstance(aid, str):
            m = _DOI_RE.search(aid)
            if m:
                return m.group(0).rstrip(".")
    return None


def _normalize_pubmed_paper(p: Dict[str, Any], query: str) -> tuple[str, Dict[str, Any]]:
    """Normalize a PubMed paper to common format."""
    pmid = p.get("pmid")
    doi = _extract_doi_from_pubmed(p)
    key = _make_dedup_key(pmid, doi)
    pub_year = None
    if p.get("pubdate"):
        try:
            pub_year = int(str(p.get("pubdate")).split()[0])
        except (ValueError, IndexError):
            pass
    norm = {
        "paper_id": pmid,
        "doi": doi,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "title": p.get("title"),
        "abstract": p.get("abstract"),
        "year": pub_year,
        "venue": p.get("source") or p.get("fulljournalname"),
        "is_open_access": False,
        "pdf_url": None,
        "pmc_id": p.get("pmc"),
        "pdf_status": None,
        "citation_count": 0,
        "authors": p.get("authors", []),
        "publication_types": p.get("pubtype", []),
        "matched_terms": [query],
        "source": "pubmed",
    }
    return key, norm


def _extract_pdf_from_epmc(full_text_url_list: Optional[Any]) -> Optional[str]:
    if not full_text_url_list:
        return None
    items = []
    if isinstance(full_text_url_list, dict):
        items = full_text_url_list.get("fullTextUrl") or []
    elif isinstance(full_text_url_list, list):
        items = full_text_url_list
    else:
        return None
    for it in items or []:
        try:
            style = (it.get("documentStyle") or "").lower()
            url = it.get("url")
            if style == "pdf" and isinstance(url, str):
                return url
        except Exception:
            continue
    return None


def _normalize_epmc_paper(p: Dict[str, Any], query: str) -> tuple[str, Dict[str, Any]]:
    """Normalize a Europe PMC paper to common format."""
    paper_id = p.get("id")
    doi = p.get("doi")
    key = _make_dedup_key(paper_id, doi)
    pdf_url = _extract_pdf_from_epmc(p.get("fullTextUrlList"))
    norm = {
        "paper_id": paper_id,
        "doi": doi,
        "url": pdf_url or p.get("fullTextUrlList"),
        "title": p.get("title"),
        "abstract": p.get("abstractText"),
        "year": p.get("pubYear"),
        "venue": p.get("journalTitle") or (p.get("journalInfo") or {}).get("journal", {}).get("title"),
        "is_open_access": p.get("isOpenAccess"),
        "open_access_pdf": None,
        "pdf_url": pdf_url,
        "pmc_id": p.get("pmcid"),
        "pdf_status": None,
        "citation_count": p.get("citedByCount") or 0,
        "influential_citation_count": 0,
        "authors": p.get("authorList"),
        "fields_of_study": None,
        "publication_types": p.get("pubTypeList"),
        "matched_terms": [query],
        "source": "europe_pmc",
    }
    return key, norm


# ── Source fetchers ────────────────────────────────────────────────────────


async def _fetch_s2(
    session: aiohttp.ClientSession,
    query: str,
    limit: int,
    seen: Dict[str, Dict[str, Any]],
    label: str = "",
) -> int:
    """Fetch from Semantic Scholar and merge into seen. Returns count added."""
    resp = await s2_paper_search(session, query=_quote_query(query), limit=limit)
    data = resp.get("data", [])
    print(f"{label}S2: {len(data)} papers")
    added = 0
    for p in data:
        key, norm = _normalize_s2_paper(p, query)
        if key not in seen:
            added += 1
        _add_to_seen(seen, key, norm, query)
    return added


async def _fetch_pubmed(
    session: aiohttp.ClientSession,
    query: str,
    limit: int,
    seen: Dict[str, Dict[str, Any]],
    label: str = "",
) -> int:
    """Fetch from PubMed and merge into seen. Returns count added."""
    results = await search_pubmed_papers(
        session, query=_quote_query(query), max_results=limit,
        return_type="summary", sort="relevance",
    )
    print(f"{label}PubMed: {len(results)} papers")
    added = 0
    for p in results:
        key, norm = _normalize_pubmed_paper(p, query)
        if key not in seen:
            added += 1
        _add_to_seen(seen, key, norm, query)
    return added


async def _fetch_epmc(
    session: aiohttp.ClientSession,
    query: str,
    limit: int,
    seen: Dict[str, Dict[str, Any]],
    label: str = "",
    publication_types: Optional[List[str]] = None,
) -> int:
    """Fetch from Europe PMC and merge into seen. Returns count added."""
    type_filter = ""
    if publication_types:
        pub_type_bits = [f'PUB_TYPE:"{str(t).replace(chr(34), "")}"' for t in publication_types]
        if pub_type_bits:
            type_filter = " AND (" + " OR ".join(pub_type_bits) + ")"
    epmc_query = f"({_quote_query(query)}) AND OPEN_ACCESS:Y AND HAS_PDF:Y{type_filter}"
    results = await search_europe_pmc_papers(
        session, epmc_query, page_size=min(limit, 100),
        max_results=limit,
        result_type="core", sort=None,
    )
    print(f"{label}EPMC: {len(results)} papers")
    added = 0
    for p in results:
        key, norm = _normalize_epmc_paper(p, query)
        if key not in seen:
            added += 1
        _add_to_seen(seen, key, norm, query)
    return added


# ── Enrichment & scoring ──────────────────────────────────────────────────


async def _url_is_pdf(session: aiohttp.ClientSession, url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        async with session.head(url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            ctype = resp.headers.get("Content-Type", "").lower()
            if "pdf" in ctype:
                return True
    except Exception:
        pass
    return url.lower().endswith(".pdf")


async def _verify_pdf_status(session: aiohttp.ClientSession, papers: List[Dict[str, Any]]) -> None:
    """Set pdf_status for all papers in-place (concurrent with semaphore)."""
    sem = asyncio.Semaphore(20)

    async def _check_one(p: Dict[str, Any]) -> None:
        async with sem:
            if await _url_is_pdf(session, p["pdf_url"]):
                p["pdf_status"] = "direct_pdf"
            else:
                p["pdf_status"] = "none"

    # Set non-PDF statuses synchronously, gather PDF checks
    to_check = []
    for p in papers:
        if p.get("pmc_id"):
            p["pdf_status"] = "pmc_xml"
        elif p.get("pdf_url"):
            to_check.append(p)
        else:
            p["pdf_status"] = "none"

    if to_check:
        await asyncio.gather(*[_check_one(p) for p in to_check])


async def _enrich_unpaywall(papers: List[Dict[str, Any]]) -> int:
    """Enrich papers with OA info from Unpaywall (concurrent with shared session).

    Checks papers that either have no PDF URL at all, or whose PDF URL
    failed verification (pdf_status == "none").  This rescues papers
    where a search source returned a broken URL (redirect to HTML, 403, etc.).
    """
    to_check = [
        p for p in papers
        if p.get("doi") and (not p.get("pdf_url") or p.get("pdf_status") == "none")
    ]
    if not to_check:
        return 0

    sem = asyncio.Semaphore(20)
    oa_found = 0

    async def _enrich_one(session: aiohttp.ClientSession, paper: Dict[str, Any]) -> bool:
        async with sem:
            try:
                oa_info = await check_doi_oa_status(paper["doi"], session=session)
                if oa_info and oa_info.get("is_oa") and oa_info.get("url"):
                    paper["pdf_url"] = oa_info["url"]
                    paper["pdf_status"] = "unpaywall_oa"
                    paper["is_open_access"] = True
                    paper["oa_via"] = "unpaywall"
                    return True
            except Exception:
                pass
            return False

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[_enrich_one(session, p) for p in to_check])
        oa_found = sum(1 for r in results if r)

    return oa_found


def _study_design_weight(publication_types: Optional[Any]) -> float:
    if not publication_types:
        return 0.0
    types = []
    if isinstance(publication_types, list):
        types = publication_types
    elif isinstance(publication_types, dict):
        pt = publication_types.get("pubType") if "pubType" in publication_types else None
        if isinstance(pt, list):
            types = [t.get("name") for t in pt if isinstance(t, dict)]
    else:
        types = [str(publication_types)]

    types_lower = {str(t).lower() for t in types}
    weight = 0.0
    if "randomized controlled trial" in types_lower:
        weight = max(weight, 4.0)
    if "meta-analysis" in types_lower:
        weight = max(weight, 3.2)
    if "systematic review" in types_lower or "review" in types_lower:
        weight = max(weight, 2.4)
    if "guideline" in types_lower:
        weight = max(weight, 2.8)
    return weight


def _venue_tier_weight(venue: Optional[str]) -> float:
    if not venue:
        return 0.0
    if venue in VENUE_TIER_1:
        return 8.0
    if venue in VENUE_TIER_2:
        return 6.0
    return 0.0


def _citations_per_year(citations: int, year: Optional[int]) -> float:
    if not year:
        return 0.0
    years = max(1, _get_current_year() - int(year) + 1)
    return round(float(citations or 0) / years, 4)


def _recency_weight(year: Optional[int]) -> float:
    if not year:
        return 0.0
    return max(0.0, min(1.0, (int(year) - 2000) / 30.0))


def compute_relevance_score(
    paper: Dict[str, Any], matched_terms: List[str], query: str | None = None,
) -> float:
    """Quality score combining query relevance, journal prestige, and citations."""
    citations = int(paper.get("citationCount") or paper.get("citation_count") or 0)
    influential = int(paper.get("influentialCitationCount") or paper.get("influential_citation_count") or 0)

    # Primary: log-scaled citations (0-14 range)
    citation_score = math.log2(citations + 1) + 0.5 * math.log2(influential + 1)

    # Primary: venue prestige (tier 1 → 8.0, tier 2 → 6.0)
    venue_w = _venue_tier_weight(paper.get("venue"))

    # Primary: query relevance (0-12 range)
    relevance_w = 0.0
    if query:
        title_score, abstract_score = _text_match_score(
            query, paper.get("title"), paper.get("abstract"),
        )
        relevance_w = 8.0 * title_score + 4.0 * abstract_score

    # Secondary: study design (RCT → 4.0, meta-analysis → 3.2)
    design_w = _study_design_weight(paper.get("publication_types"))

    # Secondary: citations per year, capped at 2.0
    cpy = _citations_per_year(citations, paper.get("year"))
    cpy_score = min(2.0, 0.3 * cpy)

    # Minor: recency, accessibility, query match
    recency_w = _recency_weight(paper.get("year"))
    status = paper.get("pdf_status")
    parse_w = 0.3 if status == "pmc_xml" else (0.2 if status == "direct_pdf" else 0.0)

    score = (
        citation_score
        + venue_w
        + relevance_w
        + 2.0 * design_w
        + cpy_score
        + 0.3 * recency_w
        + parse_w
        + 0.2 * len(matched_terms)
    )
    return round(score, 4)


def _score_and_sort(papers: List[Dict[str, Any]], query: str | None = None) -> None:
    """Score and sort papers in-place by quality."""
    for p in papers:
        p["score"] = compute_relevance_score(p, p.get("matched_terms", []), query=query)
    papers.sort(key=lambda x: (
        -(x.get("score") or 0),
        0 if x.get("pdf_status") in ("pmc_xml", "direct_pdf") else 1,
        -(x.get("citation_count") or 0),
    ))


def _compute_statistics(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for search results."""
    sources_used = sorted(set(p.get("source", "unknown") for p in papers))
    source_counts: Dict[str, int] = {}
    for p in papers:
        src = p.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    pmc_xml = sum(1 for p in papers if p.get("pdf_status") == "pmc_xml")
    direct_pdf = sum(1 for p in papers if p.get("pdf_status") == "direct_pdf")
    abstract_only = sum(1 for p in papers if p.get("pdf_status") == "none")
    high_impact = sum(1 for p in papers if p.get("venue") in (VENUE_TIER_1 | VENUE_TIER_2))

    return {
        "by_source": source_counts,
        "pmc_xml": pmc_xml,
        "direct_pdf": direct_pdf,
        "full_text": pmc_xml + direct_pdf,
        "abstract_only": abstract_only,
        "high_impact_venues": high_impact,
        "sources_used": sources_used,
        "top_venues": list(set(p.get("venue") for p in papers[:20] if p.get("venue")))[:10],
    }


# ── Core search (used by both public APIs) ─────────────────────────────────


async def _run_multi_source_search(
    session: aiohttp.ClientSession,
    query: str,
    limit: int,
    seen: Dict[str, Dict[str, Any]],
    source_pref: str = "auto",
    publication_types: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """Run search across all sources, merging results into ``seen``."""
    pref = (source_pref or "auto").lower()
    label_prefix = f"  " if verbose else f"{GREEN}[Evidence] "
    label_suffix = RESET if not verbose else ""

    def _label(name: str) -> str:
        return f"{label_prefix}{name}: " if verbose else f"{GREEN}[Evidence] {name}: {RESET}"

    # Single-source shortcuts
    if pref == "semantic_scholar":
        try:
            await _fetch_s2(session, query, limit, seen, _label("S2"))
        except Exception as e:
            if verbose:
                print(f"  S2 error: {e}")
        return
    if pref == "europe_pmc":
        try:
            await _fetch_epmc(session, query, limit, seen, _label("EPMC"), publication_types)
        except Exception as e:
            if verbose:
                print(f"  EPMC error: {e}")
        return

    # All sources in parallel, best-effort
    sources = [
        ("S2", lambda: _fetch_s2(session, query, limit, seen, _label("S2"))),
        ("PubMed", lambda: _fetch_pubmed(session, query, limit, seen, _label("PubMed"))),
        ("EPMC", lambda: _fetch_epmc(session, query, limit, seen, _label("EPMC"), publication_types)),
    ]
    results = await asyncio.gather(
        *[fn() for _, fn in sources],
        return_exceptions=True,
    )
    for (name, _), result in zip(sources, results):
        if isinstance(result, Exception) and verbose:
            print(f"  {name} error: {result}")


async def _enrich_and_rank(
    session: aiohttp.ClientSession,
    papers: List[Dict[str, Any]],
    verbose: bool = True,
    query: str | None = None,
) -> None:
    """Verify PDF access, enrich via Unpaywall/Crossref, score and sort in-place."""
    if verbose:
        print(f"\n{GREEN}Verifying PDF access...{RESET}")
    await _verify_pdf_status(session, papers)

    if verbose:
        print(f"{GREEN}Enriching with Unpaywall...{RESET}")
    oa_found = await _enrich_unpaywall(papers)
    if oa_found and verbose:
        print(f"  Found {oa_found} OA versions via Unpaywall")

    needs_crossref = any(not (p.get("citation_count") or p.get("citationCount")) for p in papers)
    if needs_crossref:
        if verbose:
            print(f"{GREEN}Enriching with Crossref...{RESET}")
        try:
            cr_count = await enrich_papers_with_crossref(papers)
            if cr_count and verbose:
                print(f"  Crossref enriched citations for {cr_count} papers")
        except Exception as e:
            if verbose:
                print(f"  Crossref error: {e}")
    elif verbose:
        print(f"{GREEN}Skipping Crossref (all papers have citation data){RESET}")

    if query:
        filtered = _filter_irrelevant(papers, query)
        papers[:] = filtered

    _score_and_sort(papers, query=query)


# ── Public APIs ────────────────────────────────────────────────────────────


async def search_papers(
    query: str,
    *,
    limit: int = 50,
    year_range: Optional[str] = None,
    source_pref: str = "auto",
) -> Dict[str, Any]:
    """Search for papers across all sources with a simple query.

    Searches S2 + PubMed + Europe PMC, deduplicates by DOI, enriches
    with Unpaywall (OA) and Crossref (citations), then ranks by journal
    prestige and log-scaled citation impact.

    Returns dict with 'papers' (ranked list), 'sources_used', 'statistics'.
    """
    async with aiohttp.ClientSession() as session:
        seen: Dict[str, Dict[str, Any]] = {}

        print(f"\n{GREEN}{'='*60}{RESET}")
        print(f"{GREEN}Searching: '{query}'{RESET}")
        print(f"{GREEN}{'='*60}{RESET}\n")

        await _run_multi_source_search(session, query, limit, seen, source_pref)
        papers = list(seen.values())

        await _enrich_and_rank(session, papers, query=query)

        stats = _compute_statistics(papers)

        print(f"\n{GREEN}{'='*60}{RESET}")
        print(f"{GREEN}Search Complete!{RESET}")
        print(f"{GREEN}{'='*60}{RESET}")
        print(f"Total unique papers: {len(papers)}")
        print(f"  PMC XML (full text):  {stats['pmc_xml']}")
        print(f"  Direct PDF (GROBID):  {stats['direct_pdf']}")
        print(f"  Abstract only:        {stats['abstract_only']}")
        print(f"High-impact venues: {stats['high_impact_venues']}")
        for src in stats["sources_used"]:
            print(f"  - {src}: {stats['by_source'].get(src, 0)} papers")
        print(f"{GREEN}{'='*60}{RESET}\n")

        return {
            "query": query,
            "papers": papers[:limit],
            "total_found": len(papers),
            "sources_used": stats["sources_used"],
            "statistics": stats,
        }


async def search_evidence_for_pair(
    session: aiohttp.ClientSession,
    intervention: str,
    biomarker: str,
    *,
    limit: int = 25,
    year: Optional[str] = None,
    open_access: Optional[bool] = None,
    min_citation_count: Optional[int] = None,
    publication_types: Optional[List[str]] = None,
    source_pref: str = "auto",
) -> Dict[str, Any]:
    """Search for papers matching an intervention-biomarker pair."""
    query = f"{intervention} and {biomarker}"
    print(f"{GREEN}[Evidence] Searching for {intervention} + {biomarker}{RESET}")

    seen: Dict[str, Dict[str, Any]] = {}
    await _run_multi_source_search(
        session, query, limit, seen, source_pref,
        publication_types=publication_types, verbose=False,
    )
    papers = list(seen.values())

    await _enrich_and_rank(session, papers, verbose=False, query=query)

    return {
        "intervention": intervention,
        "biomarker": biomarker,
        "queries_run": [query],
        "papers": papers[:limit],
    }


async def search_evidence(
    interventions: List[str],
    biomarkers: List[str],
    *,
    limit_per_pair: int = 25,
    year_range: Optional[str] = None,
    open_access: Optional[bool] = None,
    min_citation_count: Optional[int] = None,
    include_publication_types: Optional[List[str]] = None,
    source_pref: str = "auto",
) -> List[Dict[str, Any]]:
    """Search for papers matching intervention-biomarker pairs."""
    async with aiohttp.ClientSession() as session:
        results: List[Dict[str, Any]] = []
        for intervention in interventions:
            for biomarker in biomarkers:
                res = await search_evidence_for_pair(
                    session, intervention, biomarker,
                    limit=limit_per_pair,
                    year=year_range,
                    open_access=open_access,
                    min_citation_count=min_citation_count,
                    publication_types=include_publication_types,
                    source_pref=source_pref,
                )
                results.append(res)
        return results
