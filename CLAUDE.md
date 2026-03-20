# CLAUDE.md

## Project Overview

research-find is a conversational research discovery assistant. It uses a Claude agent (Sonnet) with tool-use to search academic databases, ingest papers into a Neo4j knowledge graph, and deliver daily RSS digests. The web UI is a minimal aiohttp WebSocket chat (POC — intended to be replaced by iMessage via Twilio).

## Running

```bash
# Requires .env with ANTHROPIC_API_KEY, OPENAI_API_KEY, NEO4J_URI, NEO4J_PASSWORD
pip install -r requirements.txt
python main.py
# Server starts on http://localhost:8000
```

## Code Layout

- `main.py` — entry point, starts aiohttp server, inits Neo4j, starts RSS monitor
- `web.py` — WebSocket handler, Claude agent loop with tool dispatch
- `agent/tools.py` — JSON schemas for Claude tool definitions
- `agent/handlers.py` — tool implementations + `_enrich_paper_graph()` background task
- `models/paper.py` — Pydantic models: `Paper`, `Author`, `Concept`, `User`
- `services/neo4j_store.py` — all Neo4j operations (CRUD, vector search, edges)
- `services/paper_resolver.py` — resolves DOI/arXiv/S2 URL/title → `Paper` via Semantic Scholar API
- `services/embeddings.py` — OpenAI `text-embedding-3-small` embedding + backfill
- `services/concept_extractor.py` — Claude Haiku concept extraction + embedding-based dedup
- `services/grobid.py` — GROBID PDF→TEI XML parsing
- `services/rss_feeds.py` — RSS feed catalog (Nature, Science, PNAS, arXiv, bioRxiv, medRxiv, etc.), async fetching
- `background/rss_monitor.py` — daily digest scheduler, on-demand fetch, config persistence
- `ingestion/` — API clients for Semantic Scholar, PubMed, Europe PMC, Crossref, Unpaywall

## Key Patterns

- **Background enrichment**: when a paper is added, `_enrich_paper_graph()` runs as an `asyncio.Task` — it resolves PDFs, runs GROBID, extracts concepts via Haiku, creates graph edges, fetches S2 references, and schedules embeddings. All steps are non-fatal.
- **Paper resolution order**: DOI → arXiv → S2 URL → title search (in `paper_resolver.py`)
- **Concept dedup**: new concepts are embedded and compared against existing concept embeddings in Neo4j (cosine ≥ 0.92 threshold) before creating new nodes.
- **Neo4j merge keys**: papers are merged by DOI first, then arXiv ID, then title.
- **RSS feed resolution**: `resolve_feeds()` in `rss_feeds.py` builds the feed URL dict from source/category/user-prefs. Specialty sources (medrxiv, biorxiv, arxiv) support subcategory filtering.

## Environment Variables

- `ANTHROPIC_API_KEY` — required, for Claude Sonnet (agent) and Haiku (concepts)
- `OPENAI_API_KEY` — required, for text-embedding-3-small
- `NEO4J_URI`, `NEO4J_PASSWORD` — required
- `NEO4J_USERNAME` — optional (defaults to "neo4j")
- `NEO4J_DATABASE` — optional
- `SEMANTIC_SCHOLAR_API_KEY` — optional, increases S2 rate limits
- `GROBID_URL` — optional (defaults to http://localhost:8070)

## Dependencies

anthropic, openai, neo4j, feedparser, aiohttp, pydantic, python-dotenv, lxml
