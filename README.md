# research-find

A conversational research discovery assistant that helps you find, organize, and explore academic papers through a chat interface. Powered by Claude, it searches multiple academic databases, stores papers in a Neo4j knowledge graph, and delivers daily digests from RSS feeds across major journals and preprint servers.

## Features

- **Multi-source paper search** — queries Semantic Scholar, PubMed, and Europe PMC in parallel
- **Paper ingestion** — add papers by URL, DOI, arXiv ID, or title; metadata is resolved via Semantic Scholar
- **Knowledge graph** — papers, concepts, users, and citation relationships stored in Neo4j with vector indexes for similarity search
- **PDF processing** — optional full-text extraction via GROBID (abstracts, sections, references, keywords)
- **Concept extraction** — automatic topic extraction using Claude Haiku with embedding-based deduplication
- **Vector similarity** — find similar papers using OpenAI `text-embedding-3-small` (1536-dim) embeddings stored in Neo4j
- **RSS digest** — daily digest from Nature, Science, PNAS, ScienceDaily, arXiv, bioRxiv, medRxiv, and Peter Attia's podcast
- **Configurable feeds** — choose which sources, categories, and specialties to follow
- **Concept following** — track research topics you're interested in

## Architecture

```
main.py                  — Entry point (aiohttp server on port 8000)
web.py                   — WebSocket chat UI + Claude agent loop
agent/
  tools.py               — Tool schemas for the Claude agent
  handlers.py            — Tool handler implementations + background enrichment
models/
  paper.py               — Pydantic models (Paper, Author, Concept, User)
services/
  neo4j_store.py         — Neo4j CRUD, vector search, graph edges
  paper_resolver.py      — URL/DOI/arXiv/title → Paper resolution via S2 API
  embeddings.py          — OpenAI embedding service (papers + concepts)
  concept_extractor.py   — Haiku-based concept extraction + dedup
  grobid.py              — GROBID PDF → TEI XML → structured data
  arxiv.py               — arXiv URL/ID resolution
  rss_feeds.py           — RSS feed catalog, filtering, and async fetching
background/
  rss_monitor.py         — Daily digest scheduler + on-demand fetch
ingestion/
  evidence_service.py    — Multi-source search orchestration
  semantic_scholar.py    — Semantic Scholar API client
  pubmed_api.py          — PubMed API client
  europe_pmc.py          — Europe PMC API client
  crossref.py            — Crossref API client
  unpaywall.py           — Unpaywall OA lookup
  query_expander.py      — Query expansion
```

## Neo4j Graph Schema

- **Nodes**: `Paper`, `Concept`, `User`
- **Edges**: `COVERS` (Paper→Concept), `CITES` (Paper→Paper), `ADDED` (User→Paper), `FOLLOWS` (User→Concept), `RELATED_TO` (Concept→Concept)
- **Vector indexes**: `paper_embedding` (1536-dim cosine), `concept_embedding` (1536-dim cosine)

## Setup

### Prerequisites

- Python 3.11+
- Neo4j Aura (or local Neo4j with vector index support)
- GROBID server (optional, for PDF processing)

### Environment Variables

Create a `.env` file:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
NEO4J_DATABASE=...          # optional
SEMANTIC_SCHOLAR_API_KEY=... # optional, increases rate limits
GROBID_URL=...              # optional, defaults to http://localhost:8070
```

### Install & Run

```bash
pip install -r requirements.txt
python main.py
```

Open http://localhost:8000 in your browser, enter a phone number to identify yourself, and start chatting.

## Agent Tools

The Claude agent exposes these tools during conversation:

| Tool | Description |
|------|-------------|
| `search_papers` | Search across S2, PubMed, Europe PMC |
| `add_paper` | Add a paper by URL, DOI, arXiv ID, or title |
| `get_paper_details` | Retrieve stored paper details |
| `list_stored_papers` | List papers in the database |
| `fetch_rss_papers` | Fetch latest papers from RSS feeds on demand |
| `fetch_arxiv_papers` | Fetch latest arXiv papers (routes through RSS) |
| `configure_rss_feeds` | Set which feeds to include in the daily digest |
| `set_notification_time` | Change the daily digest time |
| `find_similar_papers` | Find similar papers via vector similarity |
| `follow_concept` | Follow a research concept/topic |

## Configuration

`config.json` stores persistent preferences:

```json
{
  "notification_hour": 10,
  "notification_minute": 0,
  "rss_categories": [],
  "medrxiv_specialties": [],
  "biorxiv_specialties": [],
  "arxiv_categories": []
}
```

These can be modified through the chat interface using `configure_rss_feeds` and `set_notification_time`.
