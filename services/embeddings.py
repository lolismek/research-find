"""OpenAI embedding service with async background processing."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections import OrderedDict
from typing import Optional

from openai import AsyncOpenAI

from models.paper import Paper

MODEL = "text-embedding-3-small"  # 1536 dims
_client: Optional[AsyncOpenAI] = None
_background_tasks: set[asyncio.Task] = set()

# In-memory LRU embedding cache: hash(text) -> embedding vector
# ~12 MB at 5000 entries (1536 floats each). Keeps the most recent entries.
_CACHE_MAX = 5000
_embed_cache: OrderedDict[str, list[float]] = OrderedDict()


def _cache_key(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _cache_get(text: str) -> list[float] | None:
    key = _cache_key(text)
    if key in _embed_cache:
        _embed_cache.move_to_end(key)
        return _embed_cache[key]
    return None


def _cache_put(text: str, embedding: list[float]) -> None:
    key = _cache_key(text)
    _embed_cache[key] = embedding
    _embed_cache.move_to_end(key)
    if len(_embed_cache) > _CACHE_MAX:
        _embed_cache.popitem(last=False)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _build_text(paper: Paper) -> str:
    """Build the text to embed: Title + Abstract."""
    parts = [f"Title: {paper.title}"]
    abstract = paper.abstract or paper.grobid_abstract
    if abstract:
        parts.append(f"Abstract: {abstract}")
    return "\n".join(parts)


async def embed_text(text: str) -> list[float]:
    """Embed a single text string. Returns 1536-dim vector."""
    cached = _cache_get(text)
    if cached is not None:
        return cached
    client = _get_client()
    response = await client.embeddings.create(model=MODEL, input=text)
    embedding = response.data[0].embedding
    _cache_put(text, embedding)
    return embedding


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one API call (up to 2048 inputs).

    Checks cache first; only sends uncached texts to OpenAI.
    """
    results: list[list[float] | None] = []
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = _cache_get(text)
        if cached is not None:
            results.append(cached)
        else:
            results.append(None)
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        client = _get_client()
        response = await client.embeddings.create(model=MODEL, input=uncached_texts)
        new_embs = [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        for idx, emb in zip(uncached_indices, new_embs):
            results[idx] = emb
            _cache_put(texts[idx], emb)

    if uncached_texts:
        print(f"[embeddings] Cache: {len(texts) - len(uncached_texts)} hits, {len(uncached_texts)} misses")

    return results  # type: ignore[return-value]


async def embed_paper(paper: Paper) -> list[float]:
    """Embed a paper's title + abstract. Returns 1536-dim vector."""
    text = _build_text(paper)
    return await embed_text(text)


async def _embed_and_store(paper: Paper) -> None:
    """Background task: compute embedding and update Neo4j."""
    from services.neo4j_store import update_embedding

    try:
        text = _build_text(paper)
        embedding = await embed_text(text)

        # Determine the key to find this paper in Neo4j
        key = paper.doi or paper.arxiv_id or paper.title
        key_type = "doi" if paper.doi else ("arxiv_id" if paper.arxiv_id else "title")

        await update_embedding(key, key_type, embedding)
        print(f"[embeddings] Stored embedding for: {paper.title[:60]}")
    except Exception as e:
        print(f"[embeddings] Failed to embed '{paper.title[:40]}': {e}")


def schedule_embedding(paper: Paper) -> None:
    """Fire-and-forget: schedule embedding computation in the background."""
    task = asyncio.create_task(_embed_and_store(paper))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    print(f"[embeddings] Scheduled embedding for: {paper.title[:60]}")


def schedule_concept_embeddings(names: list[str]) -> None:
    """Fire-and-forget: schedule embedding computation for concepts missing embeddings."""
    if not names:
        return
    task = asyncio.create_task(_embed_and_store_concepts(names))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    print(f"[embeddings] Scheduled concept embeddings for {len(names)} concepts")


async def _embed_and_store_concepts(names: list[str]) -> None:
    """Background task: embed concept names and store on Concept nodes."""
    from services.neo4j_store import update_concept_embedding

    try:
        embeddings = await embed_texts(names)
        for name, embedding in zip(names, embeddings):
            await update_concept_embedding(name, embedding)
        print(f"[embeddings] Stored embeddings for {len(names)} concepts")
    except Exception as e:
        print(f"[embeddings] Failed to embed concepts: {e}")


async def backfill_concept_embeddings() -> int:
    """Find all concepts without embeddings and compute them. Returns count."""
    from services.neo4j_store import list_concepts_without_embeddings, update_concept_embedding

    names = await list_concepts_without_embeddings()
    if not names:
        print("[embeddings] All concepts have embeddings")
        return 0

    print(f"[embeddings] Backfilling {len(names)} concept embeddings...")
    total = 0
    for i in range(0, len(names), 100):
        batch = names[i:i + 100]
        embeddings = await embed_texts(batch)
        for name, embedding in zip(batch, embeddings):
            await update_concept_embedding(name, embedding)
            total += 1
        print(f"[embeddings] Backfilled {total}/{len(names)} concepts")
    return total


async def embed_papers_batch(papers: list[Paper]) -> list[list[float]]:
    """Embed multiple papers in one API call. Returns list of vectors."""
    texts = [_build_text(p) for p in papers]
    return await embed_texts(texts)


async def backfill_embeddings() -> int:
    """Find all papers without embeddings and compute them. Returns count."""
    from services.neo4j_store import list_papers_without_embeddings, update_embedding

    papers = await list_papers_without_embeddings()
    if not papers:
        print("[embeddings] All papers have embeddings")
        return 0

    print(f"[embeddings] Backfilling {len(papers)} papers...")

    # Batch in groups of 100
    total = 0
    for i in range(0, len(papers), 100):
        batch = papers[i:i + 100]
        texts = [_build_text(p) for p in batch]
        embeddings = await embed_texts(texts)

        for paper, embedding in zip(batch, embeddings):
            key = paper.doi or paper.arxiv_id or paper.title
            key_type = "doi" if paper.doi else ("arxiv_id" if paper.arxiv_id else "title")
            await update_embedding(key, key_type, embedding)
            total += 1

        print(f"[embeddings] Backfilled {total}/{len(papers)}")

    return total
