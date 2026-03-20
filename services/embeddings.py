"""OpenAI embedding service with async background processing."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from openai import AsyncOpenAI

from models.paper import Paper

MODEL = "text-embedding-3-small"  # 1536 dims
_client: Optional[AsyncOpenAI] = None
_background_tasks: set[asyncio.Task] = set()


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
    client = _get_client()
    response = await client.embeddings.create(model=MODEL, input=text)
    return response.data[0].embedding


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one API call (up to 2048 inputs)."""
    client = _get_client()
    response = await client.embeddings.create(model=MODEL, input=texts)
    return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]


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
