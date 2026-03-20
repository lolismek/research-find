"""Neo4j Aura CRUD + vector storage for papers."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from neo4j import AsyncGraphDatabase

from models.paper import Paper, Author

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        uri = os.environ["NEO4J_URI"]
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ["NEO4J_PASSWORD"]
        _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    return _driver


async def close():
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


async def init_db():
    """Create constraints and indexes."""
    driver = _get_driver()
    async with driver.session() as session:
        # Unique constraint on DOI
        await session.run(
            "CREATE CONSTRAINT paper_doi IF NOT EXISTS "
            "FOR (p:Paper) REQUIRE p.doi IS UNIQUE"
        )
        # Unique constraint on arxiv_id
        await session.run(
            "CREATE CONSTRAINT paper_arxiv IF NOT EXISTS "
            "FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE"
        )
        # Vector index for SPECTER embeddings (768-dim, cosine)
        await session.run(
            "CREATE VECTOR INDEX paper_embedding IF NOT EXISTS "
            "FOR (p:Paper) ON (p.embedding) "
            "OPTIONS {indexConfig: {"
            " `vector.dimensions`: 768,"
            " `vector.similarity_function`: 'cosine'"
            "}}"
        )


async def store_paper(paper: Paper) -> str:
    """Store or update a paper in Neo4j. Returns the merge key used."""
    driver = _get_driver()

    # Build merge clause — prefer DOI, fall back to arxiv_id, then title
    if paper.doi:
        merge_clause = "MERGE (p:Paper {doi: $doi})"
        merge_key = paper.doi
    elif paper.arxiv_id:
        merge_clause = "MERGE (p:Paper {arxiv_id: $arxiv_id})"
        merge_key = paper.arxiv_id
    else:
        merge_clause = "MERGE (p:Paper {title: $title})"
        merge_key = paper.title

    props = {
        "paper_id": paper.paper_id,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": [a.model_dump() for a in paper.authors],
        "year": paper.year,
        "venue": paper.venue,
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "is_open_access": paper.is_open_access,
        "citation_count": paper.citation_count,
        "fields_of_study": paper.fields_of_study,
        "source": paper.source,
        "grobid_abstract": paper.grobid_abstract,
        "keywords": paper.keywords,
        "added_at": paper.added_at or datetime.utcnow().isoformat(),
    }

    # Store embedding separately (Neo4j vector property)
    if paper.embedding:
        props["embedding"] = paper.embedding

    # Build SET clauses
    set_parts = []
    for key in props:
        set_parts.append(f"p.{key} = ${key}")
    set_clause = "SET " + ", ".join(set_parts)

    query = f"{merge_clause} {set_clause}"

    async with driver.session() as session:
        await session.run(query, **props)

    return merge_key


async def get_paper(
    doi: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[Paper]:
    """Retrieve a paper by identifier."""
    driver = _get_driver()

    if doi:
        query = "MATCH (p:Paper {doi: $doi}) RETURN p"
        params = {"doi": doi}
    elif arxiv_id:
        query = "MATCH (p:Paper {arxiv_id: $arxiv_id}) RETURN p"
        params = {"arxiv_id": arxiv_id}
    elif title:
        query = "MATCH (p:Paper) WHERE toLower(p.title) CONTAINS toLower($title) RETURN p LIMIT 1"
        params = {"title": title}
    else:
        return None

    async with driver.session() as session:
        result = await session.run(query, **params)
        record = await result.single()

    if not record:
        return None

    return _record_to_paper(record["p"])


async def list_papers(limit: int = 20) -> list[Paper]:
    """List stored papers, most recently added first."""
    driver = _get_driver()
    query = "MATCH (p:Paper) RETURN p ORDER BY p.added_at DESC LIMIT $limit"

    async with driver.session() as session:
        result = await session.run(query, limit=limit)
        records = await result.data()

    return [_record_to_paper(r["p"]) for r in records]


async def search_similar(embedding: list[float], limit: int = 10) -> list[Paper]:
    """Find similar papers using vector similarity search."""
    driver = _get_driver()
    query = (
        "CALL db.index.vector.queryNodes('paper_embedding', $limit, $embedding) "
        "YIELD node, score "
        "RETURN node AS p, score ORDER BY score DESC"
    )

    async with driver.session() as session:
        result = await session.run(query, limit=limit, embedding=embedding)
        records = await result.data()

    return [_record_to_paper(r["p"]) for r in records]


def _record_to_paper(node) -> Paper:
    """Convert a Neo4j node dict to a Paper model."""
    props = dict(node)

    # Reconstruct Author objects from stored dicts
    authors = []
    for a in (props.get("authors") or []):
        if isinstance(a, dict):
            authors.append(Author(**a))
        elif isinstance(a, str):
            authors.append(Author(name=a))

    return Paper(
        paper_id=props.get("paper_id"),
        doi=props.get("doi"),
        arxiv_id=props.get("arxiv_id"),
        title=props.get("title", "Untitled"),
        abstract=props.get("abstract"),
        authors=authors,
        year=props.get("year"),
        venue=props.get("venue"),
        url=props.get("url"),
        pdf_url=props.get("pdf_url"),
        is_open_access=props.get("is_open_access"),
        citation_count=props.get("citation_count") or 0,
        fields_of_study=props.get("fields_of_study"),
        source=props.get("source", "neo4j"),
        grobid_abstract=props.get("grobid_abstract"),
        keywords=props.get("keywords"),
        embedding=props.get("embedding"),
        added_at=props.get("added_at"),
    )
