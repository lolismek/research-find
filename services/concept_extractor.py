"""Concept extraction via Haiku + embedding-based deduplication."""

from __future__ import annotations

import json
import re

import anthropic

from models.paper import Paper

_EXTRACT_PROMPT = """\
Extract 5-15 research concepts/topics from this paper. Return specific, \
meaningful concepts that a researcher would use to categorize this work. \
Avoid generic terms like "research", "analysis", "methodology".

Return ONLY a JSON array of lowercase strings. Example:
["self-attention", "machine translation", "encoder-decoder architecture"]

Paper information:
"""


def normalize_text(name: str) -> str:
    """Normalize a concept name: lowercase, strip, collapse spaces."""
    name = name.strip().lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", name)


def _build_extraction_context(paper: Paper) -> str:
    """Build context string from paper metadata + GROBID sections."""
    parts = [f"Title: {paper.title}"]

    abstract = paper.abstract or paper.grobid_abstract
    if abstract:
        parts.append(f"Abstract: {abstract}")

    if paper.fields_of_study:
        parts.append(f"Fields of study: {', '.join(paper.fields_of_study)}")

    if paper.keywords:
        parts.append(f"Keywords: {', '.join(paper.keywords)}")

    if paper.sections:
        headings = [s.get("heading", "") for s in paper.sections if s.get("heading")]
        if headings:
            parts.append(f"Section names: {', '.join(headings)}")

        # Find methodology/methods section
        for s in paper.sections:
            h = (s.get("heading") or "").lower()
            if any(kw in h for kw in ("method", "approach", "model architecture")):
                text = (s.get("text") or "")[:1500]
                if text:
                    parts.append(f"Methodology section: {text}")
                break

        # Find conclusion section
        for s in paper.sections:
            h = (s.get("heading") or "").lower()
            if "conclusion" in h:
                text = (s.get("text") or "")[:1000]
                if text:
                    parts.append(f"Conclusion section: {text}")
                break

    return "\n\n".join(parts)


async def extract_concepts(paper: Paper) -> list[str]:
    """Extract concepts from a paper using Haiku.

    Uses title, abstract, section names, methodology, and conclusion
    as context for concept extraction.

    Returns normalized, deduplicated concept names.
    """
    context = _build_extraction_context(paper)
    prompt = _EXTRACT_PROMPT + context

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Parse JSON array from response
    # Handle cases where Haiku wraps in markdown code blocks
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        print(f"[concepts] Failed to parse Haiku response: {text[:200]}")
        return []

    if not isinstance(raw, list):
        return []

    concepts = set()
    for item in raw:
        if isinstance(item, str):
            n = normalize_text(item)
            if n and len(n) > 2:
                concepts.add(n)

    return sorted(concepts)


def collect_raw_concepts(paper: Paper) -> list[str]:
    """Collect and normalize concept names from paper metadata only.

    Used as a fallback when Haiku extraction is unavailable.
    """
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
    from services.neo4j_store import (
        find_similar_concept, update_concept_embedding, store_concepts,
    )

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
            # Create the node FIRST, then set its embedding
            await store_concepts([name])
            await update_concept_embedding(name, embedding)

    return sorted(set(canonical.values()))
