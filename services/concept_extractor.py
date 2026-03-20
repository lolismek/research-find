"""Concept extraction from paper metadata + embedding-based deduplication."""

from __future__ import annotations

import re

from models.paper import Paper


def normalize_text(name: str) -> str:
    """Normalize a concept name: lowercase, strip, collapse spaces."""
    name = name.strip().lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", name)


def collect_raw_concepts(paper: Paper) -> list[str]:
    """Collect and normalize concept names from paper metadata."""
    raw = []
    if paper.fields_of_study:
        raw.extend(paper.fields_of_study)
    if paper.keywords:
        raw.extend(paper.keywords)

    normalized = set()
    for name in raw:
        n = normalize_text(name)
        if n:
            normalized.add(n)

    return sorted(normalized)


async def normalize_concepts(raw_names: list[str]) -> list[str]:
    """Deduplicate concepts via embedding similarity against existing concepts.

    For each raw name:
    1. Embed the name
    2. Query Neo4j vector index for a similar existing concept (cosine >= 0.92)
    3. If match found, map to existing concept name; otherwise keep as-is
    4. Store the embedding on new concepts immediately

    Returns deduplicated list of canonical concept names.
    """
    from services.embeddings import embed_text
    from services.neo4j_store import find_similar_concept, update_concept_embedding

    canonical = {}  # raw_name -> canonical_name

    for name in raw_names:
        if name in canonical:
            continue

        embedding = await embed_text(name)
        match = await find_similar_concept(embedding, threshold=0.92)

        if match:
            canonical[name] = match
        else:
            canonical[name] = name
            # Store embedding on the new concept immediately
            await update_concept_embedding(name, embedding)

    return sorted(set(canonical.values()))
