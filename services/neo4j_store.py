"""Neo4j Aura CRUD + vector storage for papers."""

from __future__ import annotations

import itertools
import json
import os
from datetime import datetime
from typing import Optional

from neo4j import AsyncGraphDatabase

from models.paper import Paper, Author

_driver = None
_database = None


def _get_driver():
    global _driver, _database
    if _driver is None:
        uri = os.environ["NEO4J_URI"]
        user = os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ["NEO4J_PASSWORD"]
        _database = os.environ.get("NEO4J_DATABASE")
        _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    return _driver


def _session_kwargs():
    """Return kwargs for driver.session(**_session_kwargs()), including database if set."""
    if _database:
        return {"database": _database}
    return {}


async def close():
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


async def init_db():
    """Create constraints and indexes."""
    driver = _get_driver()
    async with driver.session(**_session_kwargs()) as session:
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
        # Vector index for OpenAI text-embedding-3-small (1536-dim, cosine)
        # Drop old 768-dim index if it exists, then create 1536-dim
        try:
            await session.run("DROP INDEX paper_embedding IF EXISTS")
        except Exception:
            pass
        await session.run(
            "CREATE VECTOR INDEX paper_embedding IF NOT EXISTS "
            "FOR (p:Paper) ON (p.embedding) "
            "OPTIONS {indexConfig: {"
            " `vector.dimensions`: 1536,"
            " `vector.similarity_function`: 'cosine'"
            "}}"
        )
        # Concept constraints & indexes
        await session.run(
            "CREATE CONSTRAINT concept_name IF NOT EXISTS "
            "FOR (c:Concept) REQUIRE c.name IS UNIQUE"
        )
        await session.run(
            "CREATE VECTOR INDEX concept_embedding IF NOT EXISTS "
            "FOR (c:Concept) ON (c.embedding) "
            "OPTIONS {indexConfig: {"
            " `vector.dimensions`: 1536,"
            " `vector.similarity_function`: 'cosine'"
            "}}"
        )
        # User constraint
        await session.run(
            "CREATE CONSTRAINT user_phone IF NOT EXISTS "
            "FOR (u:User) REQUIRE u.phone_number IS UNIQUE"
        )
        # Insight constraint
        await session.run(
            "CREATE CONSTRAINT insight_id IF NOT EXISTS "
            "FOR (i:Insight) REQUIRE i.insight_id IS UNIQUE"
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
        "authors_json": json.dumps([a.model_dump() for a in paper.authors]),
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

    # Build SET clauses
    set_parts = []
    for key in props:
        set_parts.append(f"p.{key} = ${key}")
    set_clause = "SET " + ", ".join(set_parts)

    query = f"{merge_clause} {set_clause}"

    async with driver.session(**_session_kwargs()) as session:
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

    async with driver.session(**_session_kwargs()) as session:
        result = await session.run(query, **params)
        record = await result.single()

    if not record:
        return None

    return _record_to_paper(record["p"])


async def list_papers(limit: int = 20) -> list[Paper]:
    """List stored papers, most recently added first."""
    driver = _get_driver()
    query = "MATCH (p:Paper) RETURN p ORDER BY p.added_at DESC LIMIT $limit"

    async with driver.session(**_session_kwargs()) as session:
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

    async with driver.session(**_session_kwargs()) as session:
        result = await session.run(query, limit=limit, embedding=embedding)
        records = await result.data()

    return [_record_to_paper(r["p"]) for r in records]


async def update_embedding(key: str, key_type: str, embedding: list[float]) -> None:
    """Update the embedding for a paper identified by key_type (doi/arxiv_id/title)."""
    driver = _get_driver()
    if key_type == "doi":
        match = "MATCH (p:Paper {doi: $key})"
    elif key_type == "arxiv_id":
        match = "MATCH (p:Paper {arxiv_id: $key})"
    else:
        match = "MATCH (p:Paper {title: $key})"

    query = f"{match} SET p.embedding = $embedding"
    async with driver.session(**_session_kwargs()) as session:
        await session.run(query, key=key, embedding=embedding)


async def list_papers_without_embeddings() -> list[Paper]:
    """List all papers that don't have embeddings yet."""
    driver = _get_driver()
    query = "MATCH (p:Paper) WHERE p.embedding IS NULL RETURN p"
    async with driver.session(**_session_kwargs()) as session:
        result = await session.run(query)
        records = await result.data()
    return [_record_to_paper(r["p"]) for r in records]


async def store_user(phone_number: str, name: str | None = None) -> None:
    """Create or update a User node."""
    driver = _get_driver()
    query = (
        "MERGE (u:User {phone_number: $phone}) "
        "ON CREATE SET u.created_at = datetime(), u.name = $name"
    )
    async with driver.session(**_session_kwargs()) as session:
        await session.run(query, phone=phone_number, name=name)


async def store_concepts(names: list[str]) -> None:
    """MERGE Concept nodes for each name."""
    if not names:
        return
    driver = _get_driver()
    query = "UNWIND $names AS name MERGE (c:Concept {name: name})"
    async with driver.session(**_session_kwargs()) as session:
        await session.run(query, names=names)


async def create_covers_edges(
    paper_key: str, key_type: str, concept_names: list[str],
) -> None:
    """Create Paper-[:COVERS]->Concept edges."""
    if not concept_names:
        return
    driver = _get_driver()
    if key_type == "doi":
        match_paper = "MATCH (p:Paper {doi: $val})"
    elif key_type == "arxiv_id":
        match_paper = "MATCH (p:Paper {arxiv_id: $val})"
    else:
        match_paper = "MATCH (p:Paper {title: $val})"

    query = (
        "UNWIND $names AS cn "
        f"{match_paper} "
        "MATCH (c:Concept {name: cn}) "
        "MERGE (p)-[:COVERS]->(c)"
    )
    async with driver.session(**_session_kwargs()) as session:
        await session.run(query, val=paper_key, names=concept_names)


async def create_added_edge(
    phone: str,
    paper_key: str,
    key_type: str,
    source: str,
    added_at: str | None = None,
) -> None:
    """Create User-[:ADDED]->Paper edge."""
    driver = _get_driver()
    if key_type == "doi":
        match_paper = "MATCH (p:Paper {doi: $val})"
    elif key_type == "arxiv_id":
        match_paper = "MATCH (p:Paper {arxiv_id: $val})"
    else:
        match_paper = "MATCH (p:Paper {title: $val})"

    at = added_at or datetime.utcnow().isoformat()
    query = (
        "MATCH (u:User {phone_number: $phone}) "
        f"{match_paper} "
        "MERGE (u)-[r:ADDED]->(p) "
        "ON CREATE SET r.added_at = datetime($at), r.source = $source, r.score = 1.0"
    )
    async with driver.session(**_session_kwargs()) as session:
        await session.run(query, phone=phone, val=paper_key, at=at, source=source)


async def store_insight(
    insight_id: str, text: str, sentiment: str, score_impact: float,
) -> None:
    """MERGE an Insight node."""
    driver = _get_driver()
    query = (
        "MERGE (i:Insight {insight_id: $insight_id}) "
        "ON CREATE SET i.text = $text, i.sentiment = $sentiment, "
        "i.score_impact = $score_impact, i.created_at = datetime()"
    )
    async with driver.session(**_session_kwargs()) as session:
        await session.run(
            query, insight_id=insight_id, text=text,
            sentiment=sentiment, score_impact=score_impact,
        )


async def create_about_edge(
    insight_id: str, paper_key: str, key_type: str, user_phone: str,
) -> None:
    """Create Insight -[:ABOUT {user_phone}]-> Paper edge."""
    driver = _get_driver()
    if key_type == "doi":
        match_paper = "MATCH (p:Paper {doi: $val})"
    elif key_type == "arxiv_id":
        match_paper = "MATCH (p:Paper {arxiv_id: $val})"
    else:
        match_paper = "MATCH (p:Paper {title: $val})"

    query = (
        "MATCH (i:Insight {insight_id: $insight_id}) "
        f"{match_paper} "
        "MERGE (i)-[r:ABOUT]->(p) "
        "ON CREATE SET r.user_phone = $phone"
    )
    async with driver.session(**_session_kwargs()) as session:
        await session.run(
            query, insight_id=insight_id, val=paper_key, phone=user_phone,
        )


async def create_insight_covers_edges(
    insight_id: str, concept_names: list[str],
) -> None:
    """Create Insight -[:COVERS]-> Concept edges."""
    if not concept_names:
        return
    driver = _get_driver()
    query = (
        "UNWIND $names AS cn "
        "MATCH (i:Insight {insight_id: $insight_id}) "
        "MATCH (c:Concept {name: cn}) "
        "MERGE (i)-[:COVERS]->(c)"
    )
    async with driver.session(**_session_kwargs()) as session:
        await session.run(query, insight_id=insight_id, names=concept_names)


async def get_paper_concepts(paper_key: str, key_type: str) -> list[str]:
    """Return concept names linked to a paper via COVERS."""
    driver = _get_driver()
    if key_type == "doi":
        match_paper = "MATCH (p:Paper {doi: $val})"
    elif key_type == "arxiv_id":
        match_paper = "MATCH (p:Paper {arxiv_id: $val})"
    else:
        match_paper = "MATCH (p:Paper {title: $val})"

    query = f"{match_paper}-[:COVERS]->(c:Concept) RETURN c.name AS name"
    async with driver.session(**_session_kwargs()) as session:
        result = await session.run(query, val=paper_key)
        records = await result.data()
    return [r["name"] for r in records]


async def update_added_score(
    phone: str, paper_key: str, key_type: str, delta: float,
) -> float:
    """Adjust score on ADDED edge, clamp to [0.0, 2.0]. Returns new score."""
    driver = _get_driver()
    if key_type == "doi":
        match_paper = "MATCH (p:Paper {doi: $val})"
    elif key_type == "arxiv_id":
        match_paper = "MATCH (p:Paper {arxiv_id: $val})"
    else:
        match_paper = "MATCH (p:Paper {title: $val})"

    query = (
        "MATCH (u:User {phone_number: $phone}) "
        f"{match_paper} "
        "MATCH (u)-[r:ADDED]->(p) "
        "SET r.score = CASE "
        "  WHEN coalesce(r.score, 1.0) + $delta < 0.0 THEN 0.0 "
        "  WHEN coalesce(r.score, 1.0) + $delta > 2.0 THEN 2.0 "
        "  ELSE coalesce(r.score, 1.0) + $delta END "
        "RETURN r.score AS score"
    )
    async with driver.session(**_session_kwargs()) as session:
        result = await session.run(query, phone=phone, val=paper_key, delta=delta)
        record = await result.single()
    if record:
        return record["score"]
    return 1.0


async def store_s2_ref_ids(
    paper_key: str, key_type: str, ref_ids: list[str],
) -> None:
    """Store the list of S2 paper IDs that this paper references."""
    if not ref_ids:
        return
    driver = _get_driver()

    if key_type == "doi":
        match = "MATCH (p:Paper {doi: $key})"
    elif key_type == "arxiv_id":
        match = "MATCH (p:Paper {arxiv_id: $key})"
    else:
        match = "MATCH (p:Paper {title: $key})"

    query = f"{match} SET p.s2_ref_ids = $ref_ids"
    async with driver.session(**_session_kwargs()) as session:
        await session.run(query, key=paper_key, ref_ids=ref_ids)


async def reconcile_cites_edges(paper_id: str) -> None:
    """Create CITES edges in both directions using stored s2_ref_ids.

    Forward:  this paper's s2_ref_ids -> MATCH existing papers by paper_id
    Reverse:  existing papers whose s2_ref_ids contain this paper's paper_id
    No S2 API calls — purely DB-local.
    """
    if not paper_id:
        return
    driver = _get_driver()

    async with driver.session(**_session_kwargs()) as session:
        # Forward: this paper cites existing papers
        await session.run(
            "MATCH (src:Paper {paper_id: $pid}) "
            "WHERE src.s2_ref_ids IS NOT NULL "
            "WITH src "
            "MATCH (tgt:Paper) "
            "WHERE tgt.paper_id IN src.s2_ref_ids "
            "MERGE (src)-[:CITES]->(tgt)",
            pid=paper_id,
        )
        # Reverse: existing papers that cite this paper
        await session.run(
            "MATCH (tgt:Paper {paper_id: $pid}) "
            "WITH tgt "
            "MATCH (src:Paper) "
            "WHERE src.s2_ref_ids IS NOT NULL AND $pid IN src.s2_ref_ids "
            "MERGE (src)-[:CITES]->(tgt)",
            pid=paper_id,
        )


async def create_follows_edge(
    phone: str, concept_name: str, explicit: bool = True,
) -> None:
    """Create User-[:FOLLOWS]->Concept edge."""
    driver = _get_driver()
    query = (
        "MATCH (u:User {phone_number: $phone}) "
        "MERGE (c:Concept {name: $name}) "
        "MERGE (u)-[r:FOLLOWS]->(c) "
        "ON CREATE SET r.since = datetime(), r.explicit = $explicit"
    )
    async with driver.session(**_session_kwargs()) as session:
        await session.run(query, phone=phone, name=concept_name, explicit=explicit)


async def update_related_to(concept_names: list[str]) -> None:
    """Update RELATED_TO weights for all pairs of concepts."""
    if len(concept_names) < 2:
        return
    driver = _get_driver()
    pairs = [
        sorted(pair)
        for pair in itertools.combinations(concept_names, 2)
    ]
    query = (
        "UNWIND $pairs AS pair "
        "MATCH (a:Concept {name: pair[0]}) "
        "MATCH (b:Concept {name: pair[1]}) "
        "MERGE (a)-[r:RELATED_TO]->(b) "
        "ON CREATE SET r.weight = 1 "
        "ON MATCH SET r.weight = r.weight + 1"
    )
    async with driver.session(**_session_kwargs()) as session:
        await session.run(query, pairs=pairs)


async def update_concept_embedding(name: str, embedding: list[float]) -> None:
    """Set the embedding on a Concept node."""
    driver = _get_driver()
    query = "MATCH (c:Concept {name: $name}) SET c.embedding = $embedding"
    async with driver.session(**_session_kwargs()) as session:
        await session.run(query, name=name, embedding=embedding)


async def list_concepts_without_embeddings() -> list[str]:
    """Return concept names that have no embedding."""
    driver = _get_driver()
    query = "MATCH (c:Concept) WHERE c.embedding IS NULL RETURN c.name AS name"
    async with driver.session(**_session_kwargs()) as session:
        result = await session.run(query)
        records = await result.data()
    return [r["name"] for r in records]


async def find_similar_concept(
    embedding: list[float], threshold: float = 0.92,
) -> str | None:
    """Find the most similar existing concept by vector similarity.

    Returns the concept name if cosine similarity >= threshold, else None.
    Returns None gracefully if the vector index is empty or not yet ready.
    """
    driver = _get_driver()
    query = (
        "CALL db.index.vector.queryNodes('concept_embedding', 1, $embedding) "
        "YIELD node, score "
        "RETURN node.name AS name, score"
    )
    try:
        async with driver.session(**_session_kwargs()) as session:
            result = await session.run(query, embedding=embedding)
            record = await result.single()
    except Exception:
        # Index empty or not yet populated — no match possible
        return None

    if record and record["score"] >= threshold:
        return record["name"]
    return None


async def store_interest_blurb(
    phone: str, blurb: str, embedding: list[float] | None = None,
) -> None:
    """Store the interest blurb and its embedding on the User node."""
    driver = _get_driver()
    query = (
        "MATCH (u:User {phone_number: $phone}) "
        "SET u.interest_blurb = $blurb, u.blurb_updated_at = datetime(), "
        "    u.interest_embedding = $embedding"
    )
    async with driver.session(**_session_kwargs()) as session:
        await session.run(query, phone=phone, blurb=blurb, embedding=embedding)


async def get_user_interest_signals(phone: str) -> dict:
    """Gather multi-hop graph signals about a user's research interests.

    Runs 5 Cypher queries and returns a dict with keys:
    recent_papers, insights, followed_concepts, citation_neighborhood, concept_neighbor_papers
    """
    driver = _get_driver()
    result = {
        "recent_papers": [],
        "insights": [],
        "followed_concepts": [],
        "citation_neighborhood": [],
        "concept_neighbor_papers": [],
    }

    async with driver.session(**_session_kwargs()) as session:
        # Query 1: Recent papers + their concepts (score >= 1.0 only)
        try:
            r1 = await session.run(
                "MATCH (u:User {phone_number: $phone})-[a:ADDED]->(p:Paper) "
                "WHERE coalesce(a.score, 1.0) >= 1.0 "
                "WITH p, a ORDER BY a.added_at DESC LIMIT 20 "
                "OPTIONAL MATCH (p)-[:COVERS]->(c:Concept) "
                "RETURN p.title AS title, "
                "       p.abstract AS abstract, "
                "       p.fields_of_study AS fields, "
                "       p.keywords AS keywords, "
                "       p.year AS year, "
                "       coalesce(a.score, 1.0) AS score, "
                "       a.added_at AS added_at, "
                "       collect(DISTINCT c.name) AS concepts",
                phone=phone,
            )
            result["recent_papers"] = await r1.data()
            print(f"[interest_signals] Q1 recent_papers: {len(result['recent_papers'])}")
        except Exception as e:
            print(f"[interest_signals] Q1 recent_papers failed: {e}")

        # Query 2: User's insights + insight concepts
        try:
            r2 = await session.run(
                "MATCH (u:User {phone_number: $phone})-[a:ADDED]->(p:Paper)<-[ab:ABOUT]-(i:Insight) "
                "WHERE ab.user_phone = $phone AND coalesce(a.score, 1.0) >= 1.0 "
                "WITH i, p.title AS paper_title ORDER BY i.created_at DESC LIMIT 30 "
                "OPTIONAL MATCH (i)-[:COVERS]->(c:Concept) "
                "RETURN i.text AS insight_text, "
                "       i.sentiment AS sentiment, "
                "       i.score_impact AS score_impact, "
                "       paper_title, "
                "       collect(DISTINCT c.name) AS insight_concepts",
                phone=phone,
            )
            result["insights"] = await r2.data()
            print(f"[interest_signals] Q2 insights: {len(result['insights'])}")
        except Exception as e:
            print(f"[interest_signals] Q2 insights failed: {e}")

        # Query 3: Concept neighbors (paper concepts + followed concepts)
        try:
            r3 = await session.run(
                "MATCH (u:User {phone_number: $phone}) "
                "OPTIONAL MATCH (u)-[a:ADDED]->(p:Paper)-[:COVERS]->(pc:Concept) "
                "WHERE coalesce(a.score, 1.0) >= 1.0 "
                "WITH u, collect(DISTINCT pc) AS paper_concepts "
                "OPTIONAL MATCH (u)-[:FOLLOWS]->(fc:Concept) "
                "WITH u, paper_concepts, collect(DISTINCT fc) AS followed_concepts "
                "WITH paper_concepts + followed_concepts AS all_concepts "
                "UNWIND all_concepts AS c "
                "WITH DISTINCT c "
                "OPTIONAL MATCH (c)-[r:RELATED_TO]-(neighbor:Concept) "
                "WHERE r.weight >= 2 "
                "RETURN c.name AS concept, "
                "       collect(DISTINCT {name: neighbor.name, weight: r.weight}) AS neighbors",
                phone=phone,
            )
            result["followed_concepts"] = await r3.data()
            print(f"[interest_signals] Q3 concepts: {len(result['followed_concepts'])}")
        except Exception as e:
            print(f"[interest_signals] Q3 concepts failed: {e}")

        # Query 4: Foundational cited papers (cited by >= 2 user papers)
        try:
            r4 = await session.run(
                "MATCH (u:User {phone_number: $phone})-[a:ADDED]->(p:Paper)-[:CITES]->(cited:Paper) "
                "WHERE coalesce(a.score, 1.0) >= 1.0 "
                "WITH u, cited, count(DISTINCT p) AS citing_count "
                "WHERE citing_count >= 2 AND NOT (u)-[:ADDED]->(cited) "
                "RETURN cited.title AS title, "
                "       cited.citation_count AS global_citations, "
                "       cited.year AS year, "
                "       citing_count "
                "ORDER BY citing_count DESC, cited.citation_count DESC "
                "LIMIT 10",
                phone=phone,
            )
            result["citation_neighborhood"] = await r4.data()
            print(f"[interest_signals] Q4 cited works: {len(result['citation_neighborhood'])}")
        except Exception as e:
            print(f"[interest_signals] Q4 cited works failed: {e}")

        # Query 5: Nearby papers sharing many concepts (not added by user)
        try:
            r5 = await session.run(
                "MATCH (u:User {phone_number: $phone})-[a:ADDED]->(p:Paper)-[:COVERS]->(c:Concept)<-[:COVERS]-(neighbor:Paper) "
                "WHERE coalesce(a.score, 1.0) >= 1.0 "
                "WITH u, neighbor, collect(DISTINCT c.name) AS shared "
                "WHERE size(shared) >= 3 AND NOT (u)-[:ADDED]->(neighbor) "
                "RETURN neighbor.title AS title, "
                "       neighbor.abstract AS abstract, "
                "       shared, "
                "       size(shared) AS shared_count "
                "ORDER BY shared_count DESC "
                "LIMIT 10",
                phone=phone,
            )
            result["concept_neighbor_papers"] = await r5.data()
            print(f"[interest_signals] Q5 nearby papers: {len(result['concept_neighbor_papers'])}")
        except Exception as e:
            print(f"[interest_signals] Q5 nearby papers failed: {e}")

    return result


def _record_to_paper(node) -> Paper:
    """Convert a Neo4j node dict to a Paper model."""
    props = dict(node)

    # Reconstruct Author objects from JSON string
    authors = []
    authors_raw = props.get("authors_json") or props.get("authors") or "[]"
    if isinstance(authors_raw, str):
        try:
            authors_raw = json.loads(authors_raw)
        except (json.JSONDecodeError, TypeError):
            authors_raw = []
    for a in authors_raw:
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
        added_at=_parse_datetime(props.get("added_at")),
    )


def _parse_datetime(val) -> Optional[datetime]:
    """Convert Neo4j DateTime or ISO string to Python datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            return None
    # Neo4j DateTime object — convert via iso_format()
    try:
        return datetime.fromisoformat(val.iso_format())
    except Exception:
        return None
