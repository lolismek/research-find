# Extracting Paper Embeddings from Semantic Scholar

## Summary

There are two methods. Both return 768-dim SPECTER v1 embeddings.

| Method | Type | Speed | Coverage |
|--------|------|-------|----------|
| `fields=embedding` (Graph API) | **Pre-computed retrieval** | <1s if cached, ~28s if computed on-the-fly | Not all papers have embeddings (returns `null`) |
| SPECTER v1 `/invoke` endpoint | **On-the-fly generation** | ~0.3-1s typical, spikes under load | Any paper (just needs title + abstract) |

## Method 1: Pre-computed — S2 Graph API `fields=embedding`

```
GET https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=embedding
```

Returns a pre-computed SPECTER embedding if one exists in S2's database.

**Response format:**
```json
{
  "paperId": "cb92a7f9d9dbcf9145e32fdfa0e70e2a6b828eb1",
  "embedding": {
    "model": "specter_v1",
    "vector": [-5.399, -4.762, ...]
  }
}
```

**Caveats:**
- Returns `"embedding": null` when no pre-computed embedding exists (e.g., "Attention is All You Need" has none)
- When the embedding isn't cached, the call can take ~28s (appears to attempt on-the-fly computation)
- Single API call, no extra dependencies — **fastest when it works**

## Method 2: On-the-fly — SPECTER v1 `/invoke` endpoint

**Endpoint:** `POST https://model-apis.semanticscholar.org/specter/v1/invoke`

Runs the SPECTER model on title + abstract you provide. Always returns an embedding.

### Fastest Approach (2 API calls)

1. **Fetch title + abstract** from S2 Graph API (~0.3-1s)
2. **POST to SPECTER v1** to generate the embedding (~0.3-1s typical)

### Code

```python
import requests

S2_API_KEY = "your-key-here"
SPECTER_URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16  # max papers per request


def get_embedding_by_query(query: str) -> list[float]:
    """Search for a paper and return its SPECTER embedding."""
    # Step 1: Search for paper (returns title + abstract in one call)
    r = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": 1, "fields": "title,abstract,paperId"},
        headers={"x-api-key": S2_API_KEY},
    )
    paper = r.json()["data"][0]

    # Step 2: Get SPECTER embedding
    r2 = requests.post(SPECTER_URL, json=[{
        "paper_id": paper["paperId"],
        "title": paper["title"],
        "abstract": paper.get("abstract", ""),
    }])
    return r2.json()["preds"][0]["embedding"]


def get_embedding_by_id(paper_id: str) -> list[float]:
    """Get SPECTER embedding for a known paper ID."""
    # Step 1: Fetch title + abstract
    r = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
        params={"fields": "title,abstract"},
        headers={"x-api-key": S2_API_KEY},
    )
    paper = r.json()

    # Step 2: Get SPECTER embedding
    r2 = requests.post(SPECTER_URL, json=[{
        "paper_id": paper["paperId"],
        "title": paper["title"],
        "abstract": paper.get("abstract", ""),
    }])
    return r2.json()["preds"][0]["embedding"]
```

### Optimal Strategy: Try cached first, fall back to generation

```python
def get_embedding_fast(paper_id: str) -> list[float]:
    """Try pre-computed first (fast), fall back to SPECTER generation."""
    # Try cached embedding (with short timeout)
    r = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
        params={"fields": "embedding,title,abstract"},
        headers={"x-api-key": S2_API_KEY},
        timeout=5,
    )
    data = r.json()
    emb = data.get("embedding")
    if emb and isinstance(emb, dict) and emb.get("vector"):
        return emb["vector"]  # Pre-computed, done in 1 call

    # Fall back to SPECTER generation
    r2 = requests.post(SPECTER_URL, json=[{
        "paper_id": paper_id,
        "title": data.get("title", ""),
        "abstract": data.get("abstract", ""),
    }])
    return r2.json()["preds"][0]["embedding"]
```

### Batch Embedding (up to 16 papers)

```python
def get_embeddings_batch(papers: list[dict]) -> dict[str, list[float]]:
    """papers: list of {"paper_id": ..., "title": ..., "abstract": ...}"""
    results = {}
    for i in range(0, len(papers), MAX_BATCH_SIZE):
        chunk = papers[i:i + MAX_BATCH_SIZE]
        r = requests.post(SPECTER_URL, json=chunk)
        for pred in r.json()["preds"]:
            results[pred["paper_id"]] = pred["embedding"]
    return results
```

## Test Results

### "Attention is All You Need"
- **Paper ID:** `204e3073870fae3d05bcbc2f6a8e263d9b72e776`
- **Pre-computed embedding:** `null` (not available)
- **SPECTER v1 generated:** 768 dims, first 5: `[-2.0916, -2.5462, -0.0823, 3.3069, 1.1264]`

### FAQ example paper (cb92a7f9...)
- **Pre-computed embedding:** available, model `specter_v1`, 768 dims, returned in <1s

### BERT (df2b0e26...)
- **Pre-computed embedding:** available, model `specter_v1`, 768 dims, but took ~28s (computed on-the-fly?)

## Key Details

| Detail | Value |
|--------|-------|
| Pre-computed endpoint | `GET /graph/v1/paper/{id}?fields=embedding` |
| Generation endpoint | `POST https://model-apis.semanticscholar.org/specter/v1/invoke` |
| Embedding dimensions | 768 |
| Model | `specter_v1` (both methods) |
| Max batch size (generation) | 16 papers per request |
| Required fields (generation) | `paper_id`, `title`, `abstract` |
| Auth for Graph API | API key via `x-api-key` header |
| Auth for SPECTER /invoke | None required |
| SPECTER v2 /invoke | Not publicly available ("Missing Authentication Token") |

## Sources

- [allenai/paper-embedding-public-apis](https://github.com/allenai/paper-embedding-public-apis) — SPECTER v1 /invoke docs
- [allenai/s2-folks](https://github.com/allenai/s2-folks) — S2 API release notes & FAQ (confirms `fields=embedding` is active)
- S2 API key is in `.env` as `SEMANTIC_SCHOLAR_API_KEY`
