# research-find

A conversational research discovery assistant that helps you find, organize, and explore academic papers through a chat interface. Powered by Claude (Sonnet) with tool-use, it searches multiple academic databases in parallel, ingests papers into a Neo4j knowledge graph, extracts concepts via Claude Haiku, computes vector embeddings, and delivers personalized daily RSS digests ranked by your research interests.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Setup](#setup)
- [How It Works](#how-it-works)
- [Agent Tools](#agent-tools)
  - [search_papers](#search_papers)
  - [add_paper](#add_paper)
  - [get_paper_details](#get_paper_details)
  - [list_stored_papers](#list_stored_papers)
  - [find_similar_papers](#find_similar_papers)
  - [fetch_rss_papers](#fetch_rss_papers)
  - [fetch_arxiv_papers](#fetch_arxiv_papers)
  - [configure_rss_feeds](#configure_rss_feeds)
  - [set_notification_time](#set_notification_time)
  - [add_insight](#add_insight)
- [Neo4j Knowledge Graph](#neo4j-knowledge-graph)
  - [Node Types](#node-types)
  - [Edge Types](#edge-types)
  - [Vector Indexes](#vector-indexes)
  - [Constraints](#constraints)
- [Background Enrichment Pipeline](#background-enrichment-pipeline)
- [Personalization System](#personalization-system)
- [RSS Feed System](#rss-feed-system)
- [Ingestion Clients](#ingestion-clients)
- [Services](#services)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)

## Features

- **Multi-source paper search** -- queries Semantic Scholar, PubMed, and Europe PMC in parallel, deduplicates by DOI, enriches with Unpaywall (open access) and Crossref (citations), then ranks by journal prestige, citation impact, and query relevance
- **Paper ingestion** -- add papers by URL, DOI, arXiv ID, Semantic Scholar URL, or title; metadata is resolved through a cascading resolution chain (DOI -> arXiv -> S2 URL -> title search)
- **Knowledge graph** -- papers, concepts, users, insights, and citation relationships stored in Neo4j with full-text and vector indexes
- **Background enrichment** -- when a paper is added, a non-blocking async pipeline resolves PDFs, runs GROBID, extracts concepts, creates graph edges, fetches S2 references, and schedules embeddings
- **Concept extraction** -- automatic topic extraction using Claude Haiku with embedding-based deduplication (cosine >= 0.92 threshold) against existing concepts in the graph
- **Vector similarity** -- find similar papers using OpenAI `text-embedding-3-small` (1536-dim) embeddings stored in Neo4j vector indexes
- **User insights** -- record opinions, observations, and critiques about papers; insights adjust the paper's user-specific score and are linked to relevant concepts
- **Personalized ranking** -- search results and RSS digests are re-ranked by cosine similarity to an auto-generated user interest embedding derived from multi-hop graph signals
- **RSS digest** -- daily digest from Nature, Science, PNAS, ScienceDaily, arXiv, bioRxiv, medRxiv, and Peter Attia's podcast, with configurable sources and subcategories
- **Embedding cache** -- in-memory LRU cache (5000 entries, ~12 MB) for OpenAI embeddings to avoid redundant API calls

## Architecture

```
main.py                      Entry point (aiohttp server on port 8000)
web.py                       WebSocket chat UI + Claude Sonnet agent loop
agent/
  tools.py                   Tool schemas (JSON) for the Claude agent
  handlers.py                Tool handler implementations + background enrichment tasks
  chat.py                    Alternative CLI chat loop (no WebSocket)
models/
  paper.py                   Pydantic models: Paper, Author, Concept, Insight, User
services/
  neo4j_store.py             Neo4j CRUD, vector search, graph edges, interest signals
  paper_resolver.py          URL/DOI/arXiv/S2/title -> Paper resolution via S2 API
  embeddings.py              OpenAI embedding service (papers + concepts) with LRU cache
  concept_extractor.py       Haiku-based concept extraction + embedding dedup + insight matching
  interest_profile.py        Multi-hop graph signal aggregation + Haiku blurb synthesis
  grobid.py                  GROBID PDF -> TEI XML -> structured data
  arxiv.py                   arXiv URL/ID resolution + RSS feed parser
  rss_feeds.py               RSS feed catalog, filtering, and async fetching
background/
  rss_monitor.py             Daily digest scheduler, on-demand fetch, personalized ranking
ingestion/
  evidence_service.py        Multi-source search orchestration + ranking pipeline
  semantic_scholar.py        Semantic Scholar API client with retry/backoff
  pubmed_api.py              PubMed E-utilities client (ESearch + ESummary + EFetch)
  europe_pmc.py              Europe PMC REST API client with cursor pagination
  crossref.py                Crossref API client for citation enrichment
  unpaywall.py               Unpaywall API client for open access lookup
  query_expander.py          Claude-powered query expansion for diverse search coverage
```

## Setup

### Prerequisites

- Python 3.11+
- Neo4j Aura (or local Neo4j 5.x with vector index support)
- GROBID server (optional, for PDF full-text extraction)

### Install & Run

```bash
# 1. Clone the repository
git clone <repo-url> && cd research-find

# 2. Create a .env file (see Environment Variables below)

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Start GROBID for PDF processing
docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0

# 5. Start the server
python main.py
# Server starts on http://localhost:8000
```

Open http://localhost:8000 in your browser, enter a phone number to identify yourself, and start chatting.

## How It Works

1. **User connects** via WebSocket and identifies with a phone number. A `User` node is created/merged in Neo4j.
2. **User sends a message** which is forwarded to Claude Sonnet along with the full conversation history and available tool definitions.
3. **Claude decides** which tool(s) to call based on the user's request. Tool calls are executed in parallel when independent.
4. **Tool results** are sent back to Claude, which may call more tools or generate a final text response.
5. **Background tasks** run asynchronously after tool completion (e.g., paper enrichment, embedding computation) without blocking the conversation.

The daily RSS monitor runs alongside the chat server, pre-fetching and ranking papers before the scheduled notification time.

## Agent Tools

### search_papers

Search across Semantic Scholar, PubMed, and Europe PMC simultaneously.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query (topic, keywords, or research question) |
| `limit` | integer | No | Maximum results to return (default: 10) |
| `personalize` | boolean | No | Re-rank results by personal research interest similarity (default: false) |

**Workflow:**
1. Sends the query to all three sources in parallel (`evidence_service.py`)
2. Normalizes results from each source into a common format
3. Deduplicates by DOI across sources
4. Verifies PDF accessibility (HEAD requests) for all results
5. Enriches with Unpaywall (finds OA versions of paywalled papers)
6. Enriches with Crossref (backfills citation counts for papers missing them)
7. Filters out irrelevant results by keyword matching
8. Scores and ranks by: query relevance (title/abstract match), log-scaled citations, journal prestige (tier-1/tier-2 venues), study design (RCTs > meta-analyses > reviews), citations-per-year, recency, and PDF availability
9. If `personalize=true`: regenerates the user's interest blurb from graph signals, embeds all results in a single batch, re-ranks by cosine similarity to the user's interest embedding
10. If not personalized: attaches the user's existing interest blurb so the LLM can decide whether to suggest personalization

**Journal Prestige Tiers:**
- **Tier 1** (8.0 weight): NEJM, Lancet, JAMA, BMJ, Nature Medicine, Science, Cell, PNAS, Circulation
- **Tier 2** (6.0 weight): Nature, Nature Communications, JAMA Internal Medicine, European Heart Journal, Annals of Internal Medicine, JACC

---

### add_paper

Add a paper to the knowledge graph by any identifier.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `identifier` | string | Yes | Paper URL (arXiv, DOI, Semantic Scholar), DOI string, arXiv ID, or title |
| `process_pdf` | boolean | No | Run GROBID on the PDF for full-text extraction (default: false) |
| `source` | string | No | How the paper was discovered: `"manual"`, `"recommended"`, or `"rss"` (default: `"manual"`) |
| `force` | boolean | No | Force re-add even if the paper already exists (default: false) |

**Resolution Order** (`paper_resolver.py`):
1. **DOI pattern detected** -> Semantic Scholar `/paper/DOI:{doi}`
2. **arXiv URL or ID** -> arXiv Atom API, then S2 for enrichment
3. **Semantic Scholar URL** -> Extract paper ID from URL, call S2
4. **Title string** -> Search S2 by title, pick best match

**Synchronous Phase** (user waits):
- Resolve the identifier to a `Paper` object with metadata
- Store/merge the paper in Neo4j (merge key: DOI > arXiv ID > title)
- Return confirmation to the user

**Background Enrichment Phase** (runs async, see [Background Enrichment Pipeline](#background-enrichment-pipeline)):
- PDF resolution + GROBID processing
- Concept extraction via Haiku
- Graph edge creation (COVERS, ADDED, CITES, RELATED_TO)
- S2 reference fetching and citation reconciliation
- Paper and concept embedding scheduling

---

### get_paper_details

Retrieve full details of a stored paper.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `doi` | string | No | DOI of the paper |
| `arxiv_id` | string | No | arXiv ID of the paper |
| `title` | string | No | Title or title fragment (case-insensitive CONTAINS search) |

At least one parameter should be provided. Returns all stored metadata including embedding status, GROBID processing status, fields of study, keywords, and timestamps.

---

### list_stored_papers

List papers in the database, ordered by most recently added.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | No | Maximum papers to return (default: 20) |

---

### find_similar_papers

Find papers similar to a reference paper using vector cosine similarity.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `doi` | string | No | DOI of the reference paper |
| `arxiv_id` | string | No | arXiv ID of the reference paper |
| `title` | string | No | Title of the reference paper |
| `limit` | integer | No | Number of similar papers to return (default: 10) |

**Workflow:**
1. Look up the reference paper in Neo4j
2. Retrieve its embedding (requires the paper to have been embedded)
3. Query the `paper_embedding` vector index with cosine similarity
4. Filter out the reference paper itself from results
5. Return ranked similar papers

---

### fetch_rss_papers

Fetch latest papers from RSS feeds on demand.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source` | string | No | Feed source: `"nature"`, `"science_aaas"`, `"pnas"`, `"science_daily"`, `"medrxiv"`, `"biorxiv"`, `"arxiv"`, `"peter_attia"`. Omit for all sources. |
| `category` | string | No | Subcategory within source (e.g., `"cs.AI"` for arxiv, `"neuroscience"` for biorxiv, `"Oncology"` for medrxiv) |
| `top_n` | integer | No | Number of top papers to return (default: 10) |

**Workflow:**
1. Regenerate the user's interest blurb from graph signals (if user is logged in)
2. Resolve feed URLs based on source/category/user preferences
3. Fetch all feeds in parallel (up to 15 concurrent, 7-day window)
4. Flatten, sort by date, and deduplicate entries (by DOI > arXiv ID > title)
5. Embed all entries and rank by cosine similarity to user's interest embedding
6. Return top N papers

---

### fetch_arxiv_papers

Convenience wrapper that routes through `fetch_rss_papers` with `source="arxiv"`.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category` | string | No | arXiv category (e.g., `"cs.AI"`, `"cs.CL"`, `"stat.ML"`). Omit for all. |
| `top_n` | integer | No | Number of top papers to return (default: 5) |

---

### configure_rss_feeds

Configure which RSS sources and subcategories appear in the daily digest.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `rss_categories` | string[] | No | Top-level sources: `"nature"`, `"science_aaas"`, `"pnas"`, `"science_daily"`, `"peter_attia"` |
| `medrxiv_specialties` | string[] | No | medRxiv subjects (e.g., `["Oncology", "Neurology"]`) |
| `biorxiv_specialties` | string[] | No | bioRxiv subjects (e.g., `["neuroscience", "genetics"]`) |
| `arxiv_categories` | string[] | No | arXiv categories (e.g., `["cs.AI", "cs.LG"]`) |

Preferences are persisted to `config.json` and take effect immediately for both on-demand fetches and the daily digest.

---

### set_notification_time

Set when the daily RSS digest is delivered.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `hour` | integer | Yes | Hour of the day (0-23) |
| `minute` | integer | No | Minute of the hour (0-59, default: 0) |

Restarts the daily loop with the new schedule. Persisted to `config.json`.

---

### add_insight

Record a user observation, opinion, or critique about a paper.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `paper_identifier` | string | Yes | DOI, arXiv ID, or title of the paper |
| `insight_text` | string | Yes | The user's observation or comment |
| `sentiment` | string | Yes | `"positive"`, `"negative"`, or `"neutral"` |
| `score_impact` | number | Yes | How much to adjust the paper's ADDED score (e.g., +0.3 or -0.4) |

**Score Guidelines:**
- Positive insights: +0.1 to +0.5 (minor praise to significant appreciation)
- Neutral observations: 0.0
- Negative insights: -0.1 to -0.7 (mild criticism to fundamental issues)

**Workflow:**
1. Resolve the paper in Neo4j
2. Create an `Insight` node with text, sentiment, and score_impact
3. Create `ABOUT` edge from Insight to Paper
4. **Background enrichment:**
   - Get the paper's concepts (extract inline if none exist yet)
   - Match insight text to concepts using two-pass matching:
     - Pass 1: Substring/word-overlap match (>= 60% concept word overlap)
     - Pass 2: Embedding similarity (cosine >= 0.70)
   - Create `COVERS` edges from Insight to matched Concepts
   - Update `ADDED` edge score (clamped to [0.0, 2.0])

The agent is instructed to call this automatically whenever the user comments on a paper, without asking for permission.

## Neo4j Knowledge Graph

### Node Types

| Node | Properties | Description |
|------|-----------|-------------|
| **Paper** | `paper_id`, `doi`, `arxiv_id`, `title`, `abstract`, `authors_json`, `year`, `venue`, `url`, `pdf_url`, `is_open_access`, `citation_count`, `fields_of_study`, `source`, `grobid_abstract`, `keywords`, `embedding` (1536-dim), `s2_ref_ids`, `added_at` | Academic paper with metadata and optional embedding |
| **Concept** | `name` (unique), `embedding` (1536-dim) | Research topic/concept extracted from papers |
| **User** | `phone_number` (unique), `name`, `created_at`, `interest_blurb`, `blurb_updated_at`, `interest_embedding` (1536-dim) | User with research interest profile |
| **Insight** | `insight_id` (unique), `text`, `sentiment`, `score_impact`, `created_at` | User's observation or opinion about a paper |

### Edge Types

| Edge | From -> To | Properties | Description |
|------|-----------|------------|-------------|
| **COVERS** | Paper -> Concept | -- | Paper covers this research concept |
| **COVERS** | Insight -> Concept | -- | Insight relates to this concept |
| **CITES** | Paper -> Paper | -- | Citation relationship (from S2 reference data) |
| **ADDED** | User -> Paper | `added_at`, `source`, `score` (0.0-2.0, default 1.0) | User added this paper to their library |
| **ABOUT** | Insight -> Paper | `user_phone` | This insight is about a specific paper |
| **RELATED_TO** | Concept -> Concept | `weight` (incremented each time both concepts co-occur on a paper) | Co-occurrence relationship between concepts |

### Vector Indexes

| Index | Node | Property | Dimensions | Similarity |
|-------|------|----------|-----------|------------|
| `paper_embedding` | Paper | `embedding` | 1536 | cosine |
| `concept_embedding` | Concept | `embedding` | 1536 | cosine |

Both indexes use OpenAI `text-embedding-3-small` embeddings.

### Constraints

- `Paper.doi` -- unique
- `Paper.arxiv_id` -- unique
- `Concept.name` -- unique
- `User.phone_number` -- unique
- `Insight.insight_id` -- unique

### Merge Strategy

Papers are merged (upserted) using this priority:
1. If the paper has a DOI -> merge on `doi`
2. If it has an arXiv ID -> merge on `arxiv_id`
3. Otherwise -> merge on `title`

## Background Enrichment Pipeline

When a paper is added via `add_paper`, the following enrichment steps run asynchronously after the user receives confirmation. All steps are non-fatal (failures are logged but don't break the pipeline).

```
add_paper (sync)
    |
    v
_enrich_paper_graph (async background task)
    |
    +-- Step 0: Resolve PDF URL + GROBID
    |   - If no keywords yet, try PDF URL fallback chain:
    |     S2 openAccessPdf -> arXiv PDF -> Unpaywall
    |   - Download PDF and process through GROBID
    |   - Extract keywords, abstract, sections, references
    |   - Persist GROBID data to Neo4j
    |
    +-- Step 1: Extract concepts via Haiku
    |   - Build context from title, abstract, fields, keywords,
    |     section headings, methodology, and conclusion
    |   - Call Claude Haiku to extract 5-15 research concepts
    |   - Fallback: use metadata fields_of_study + keywords
    |
    +-- Step 2: Normalize & deduplicate concepts
    |   - Embed each concept name
    |   - Query concept_embedding vector index (cosine >= 0.92)
    |   - If match found: map to existing canonical concept
    |   - If no match: create new Concept node with embedding
    |
    +-- Step 3: Create COVERS edges (Paper -> Concept)
    |
    +-- Step 4: Update RELATED_TO weights
    |   - For all pairs of concepts on this paper,
    |     increment the co-occurrence weight
    |
    +-- Step 5: Create ADDED edge (User -> Paper)
    |   - Records source (manual/recommended/rss)
    |   - Initial score: 1.0
    |
    +-- Step 6: Fetch S2 references + reconcile CITES
    |   - GET /paper/{id}/references from Semantic Scholar
    |   - Store reference paper IDs on the Paper node
    |   - Create CITES edges (forward: this paper cites existing papers)
    |   - Create CITES edges (reverse: existing papers that cite this one)
    |   - Purely DB-local -- no additional API calls for reconciliation
    |
    +-- Step 7: Schedule paper embedding
    |   - Embed Title + Abstract via OpenAI
    |   - Store 1536-dim vector on Paper node
    |
    +-- Step 8: Schedule concept embeddings
        - Find all concepts without embeddings
        - Batch embed and store on Concept nodes
```

## Personalization System

The system builds a dynamic research interest profile for each user by analyzing their graph neighborhood.

### Interest Signal Gathering (`neo4j_store.get_user_interest_signals`)

Five Cypher queries extract multi-hop signals from the graph:

1. **Recent papers** -- User's added papers (score >= 1.0) with their concepts, ordered by recency (limit 20)
2. **User insights** -- Insight text, sentiment, and linked concepts for papers with score >= 1.0 (limit 30)
3. **Concept neighbors** -- RELATED_TO neighbors with weight >= 2 for all of the user's paper concepts
4. **Citation neighborhood** -- Papers cited by >= 2 of the user's papers that the user hasn't added (foundational works)
5. **Concept neighbor papers** -- Papers sharing >= 3 concepts with the user's papers that the user hasn't added

### Concept Ranking (`interest_profile._rank_concepts`)

Concepts are scored with recency weighting:
- Paper concepts: 2.0x weight if paper score >= 1.5, 1.0x otherwise; multiplied by recency (1.5x for 7 days, 1.0x for 30 days, 0.5x older)
- Insight concepts: +3.0 flat bonus (user's explicit observations)
- RELATED_TO neighbors: +0.5 * (weight / max_weight) for neighbors with weight >= 3

### Blurb Generation (`interest_profile._synthesize_blurb`)

A structured context document with 6 sections (top concepts, recent papers, insights, foundational cited works, concept neighbors, nearby papers) is sent to Claude Haiku. Haiku generates a 150-250 word research interest profile optimized for embedding comparison:
- Uses precise technical vocabulary that would appear in relevant abstracts
- Groups related interests into 3-5 sentences
- Avoids meta-commentary, paper titles, or filler language

The blurb and its embedding are stored on the User node and refreshed before each personalized search or RSS digest.

### Search Re-Ranking

When `personalize=true` on `search_papers`:
1. Generate/refresh the user's interest blurb
2. Embed all search result abstracts in a single batch API call
3. Score each result by cosine similarity to the user's interest embedding
4. Re-sort by similarity score

## RSS Feed System

### Feed Catalog (`rss_feeds.py`)

| Source | Feed Count | Description |
|--------|-----------|-------------|
| **Nature** | 11 feeds | Nature Medicine, Nature Biotechnology, Nature, Nature Genetics, Nature Immunology, Nature Neuroscience, Nature Materials, Nature Physics, Nature Chemistry, Nature Communications, Nature Computational Science |
| **Science (AAAS)** | 1 feed | Science journal table of contents |
| **PNAS** | 1 feed | PNAS table of contents |
| **ScienceDaily** | 14 feeds | Health, Science, Materials, Physics, Chemistry, Engineering, Nanotech, Quantum Physics, Biochemistry, Computer Science, AI, Earth Science, Astronomy, Astrophysics |
| **medRxiv** | ~50 feeds | All medical specialties (Oncology, Neurology, Cardiology, etc.) |
| **bioRxiv** | 25 feeds | Neuroscience, Genetics, Genomics, Immunology, Cancer Biology, Biochemistry, Biophysics, Cell Biology, Microbiology, Molecular Biology, and more |
| **arXiv** | 12 feeds | Condensed Matter (4), Physics (3), CS (AI, ML, CV), Quantitative Biology (2) |
| **Peter Attia** | 1 feed | Podcast RSS (health/longevity) |

### Daily Digest Workflow

```
1. Sleep until (notification_time - 1 hour)
2. Regenerate user interest blurb from graph signals
3. Resolve feed URLs from user preferences
4. Fetch all feeds in parallel (15 concurrent, 7-day window)
5. Flatten and deduplicate entries (DOI > arXiv ID > title)
6. Embed all entries in batches of 100
7. Rank by cosine similarity to user's interest embedding
8. Sleep until notification_time
9. Send top 10 papers to connected chat
10. Loop (repeat daily)
```

### Feed Resolution Priority

- `source + category` specified: single feed URL
- `source` only: all feeds for that source
- Neither specified: user preferences from config, or full catalog if no preferences set

## Ingestion Clients

### Semantic Scholar (`ingestion/semantic_scholar.py`)
- Search endpoint: `/graph/v1/paper/search`
- Rate limiting: 1.15s minimum between requests (unauthenticated), faster with API key
- Retry: 429 (2s/5s/10s), 5xx (5s/15s/30s), timeout (5s/10s/15s)
- Fields: paperId, externalIds, title, abstract, venue, year, citationCount, isOpenAccess, openAccessPdf, fieldsOfStudy, authors, and more

### PubMed (`ingestion/pubmed_api.py`)
- Uses NCBI E-utilities: ESearch (find IDs) + ESummary (metadata) + EFetch (abstracts)
- Automatic pagination in batches of 500
- Abstract enrichment via separate EFetch call (ESummary doesn't include abstracts)
- Retry on 429 with exponential backoff

### Europe PMC (`ingestion/europe_pmc.py`)
- REST API with cursor-based pagination
- Supports `core` result type for full metadata including abstracts
- Page size up to 1000, with safety limits on total pages

### Crossref (`ingestion/crossref.py`)
- Enriches papers with citation counts, funder info, and license data
- Only called for papers missing citation data
- 8 concurrent requests with semaphore

### Unpaywall (`ingestion/unpaywall.py`)
- Looks up open access status for DOIs
- Used to find legal OA versions of paywalled papers
- Rescues papers where the original PDF URL was broken

### Query Expander (`ingestion/query_expander.py`)
- Claude-powered expansion of broad topics into 8-12 diverse sub-queries
- Targets different study designs: RCTs, mechanisms, animal models, meta-analyses, safety, dose-response, etc.

## Services

### Paper Resolver (`services/paper_resolver.py`)
Cascading resolution chain that converts any identifier to a `Paper`:
1. DOI regex match -> S2 `/paper/DOI:{doi}`
2. arXiv URL/ID -> arXiv Atom API + S2 enrichment
3. S2 URL -> extract hex paper ID -> S2 API
4. Title search -> S2 search, pick best title match

Also fetches S2 references and citations for graph enrichment.

### Embeddings (`services/embeddings.py`)
- Model: OpenAI `text-embedding-3-small` (1536 dimensions)
- In-memory LRU cache: 5000 entries (~12 MB), keyed by SHA-256 of normalized text
- Batch embedding: sends only uncached texts to OpenAI, fills results from cache
- Paper text = `"Title: {title}\nAbstract: {abstract}"`
- Background scheduling: fire-and-forget tasks for paper and concept embeddings
- Backfill functions: `backfill_embeddings()` and `backfill_concept_embeddings()` for batch processing all unembedded entities

### Concept Extractor (`services/concept_extractor.py`)
- **Haiku extraction**: sends paper context (title, abstract, fields, keywords, methodology section, conclusion section) to Claude Haiku; returns 5-15 normalized, lowercase concept strings
- **Metadata fallback**: collects `fields_of_study` + `keywords` if Haiku is unavailable
- **Deduplication**: embeds each concept, queries Neo4j `concept_embedding` vector index at cosine >= 0.92; maps to existing concept if match found, otherwise creates new node
- **Insight matching**: two-pass matching of insight text to paper concepts:
  - Pass 1: substring/word overlap (>= 60% overlap)
  - Pass 2: embedding cosine similarity (>= 0.70)

### GROBID (`services/grobid.py`)
- Downloads PDF from URL, sends to GROBID `/api/processFulltextDocument`
- Parses TEI XML output to extract: title, abstract, authors, section headings + text, references (with DOIs), and keywords
- 120-second timeout for large PDFs

### Interest Profile (`services/interest_profile.py`)
- Gathers 5 types of graph signals via Cypher queries
- Ranks concepts with recency weighting and insight bonuses
- Synthesizes a 150-250 word blurb via Claude Haiku
- Stores blurb + embedding on the User node
- Fallback: concatenates ranked concepts and paper titles if Haiku fails

## Configuration

`config.json` stores persistent user preferences (auto-created on first use):

```json
{
  "notification_hour": 9,
  "notification_minute": 0,
  "rss_categories": ["nature", "science_aaas"],
  "medrxiv_specialties": ["Oncology", "Neurology"],
  "biorxiv_specialties": ["neuroscience", "genetics"],
  "arxiv_categories": ["cs.AI", "cs.LG"]
}
```

Modifiable through the chat interface via `configure_rss_feeds` and `set_notification_time`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | For Claude Sonnet (agent) and Haiku (concept extraction, interest profiles) |
| `OPENAI_API_KEY` | Yes | For `text-embedding-3-small` embeddings |
| `NEO4J_URI` | Yes | Neo4j connection URI (e.g., `neo4j+s://xxxxx.databases.neo4j.io`) |
| `NEO4J_PASSWORD` | Yes | Neo4j password |
| `NEO4J_USERNAME` | No | Neo4j username (default: `"neo4j"`) |
| `NEO4J_DATABASE` | No | Neo4j database name (default: default database) |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Increases S2 rate limits from ~1/sec to ~10/sec |
| `GROBID_URL` | No | GROBID server URL (default: `http://localhost:8070`) |

## Dependencies

```
anthropic>=0.40.0      # Claude Sonnet + Haiku
openai>=1.30.0         # text-embedding-3-small
neo4j>=5.20.0          # Async Neo4j driver
feedparser>=6.0.0      # RSS/Atom feed parsing
aiohttp>=3.9.0         # Async HTTP + WebSocket server
pydantic>=2.5.0        # Data models
python-dotenv>=1.0.0   # .env file loading
lxml>=5.0.0            # GROBID TEI XML parsing
```
