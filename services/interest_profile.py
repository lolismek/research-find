"""Generate a text blurb summarizing a user's research interests from graph signals."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

import anthropic


async def generate_user_interest_blurb(phone_number: str) -> str:
    """Generate a dense research interest blurb for embedding-based paper ranking.

    Also stores the blurb on the User node in Neo4j.
    """
    from services.neo4j_store import get_user_interest_signals, store_interest_blurb
    from services.embeddings import embed_text

    print(f"[interest_profile] Generating blurb for {phone_number}")
    signals = await get_user_interest_signals(phone_number)

    n_papers = len(signals["recent_papers"])
    n_insights = len(signals["insights"])
    n_concepts = len(signals["followed_concepts"])
    n_cited = len(signals["citation_neighborhood"])
    n_nearby = len(signals["concept_neighbor_papers"])
    print(f"[interest_profile] Signals: {n_papers} papers, {n_insights} insights, "
          f"{n_concepts} concepts, {n_cited} cited works, {n_nearby} nearby papers")

    if not signals["recent_papers"]:
        print("[interest_profile] No papers found, returning empty blurb")
        return ""

    ranked_concepts = _rank_concepts(signals)
    print(f"[interest_profile] Top concepts: {', '.join(c for c, _ in ranked_concepts[:5])}")

    context = _build_signal_context(signals, ranked_concepts)
    blurb = await _synthesize_blurb(context)

    if blurb:
        print("[interest_profile] Embedding blurb...")
        embedding = await embed_text(blurb)
        await store_interest_blurb(phone_number, blurb, embedding)
        print(f"[interest_profile] Blurb + embedding stored ({len(blurb.split())} words, {len(embedding)}-dim)")
        print(f"[interest_profile] --- BLURB ---\n{blurb}\n[interest_profile] --- END ---")
    else:
        print("[interest_profile] Blurb generation produced empty result")

    return blurb


def _rank_concepts(signals: dict) -> list[tuple[str, float]]:
    """Aggregate concept scores across all signal types with recency weighting."""
    scores: dict[str, float] = defaultdict(float)
    now = datetime.now(timezone.utc)

    def _recency_multiplier(added_at) -> float:
        if added_at is None:
            return 1.0
        if isinstance(added_at, str):
            try:
                added_at = datetime.fromisoformat(added_at)
            except ValueError:
                return 1.0
        # Handle Neo4j DateTime objects
        if not isinstance(added_at, datetime):
            try:
                added_at = datetime.fromisoformat(added_at.iso_format())
            except Exception:
                return 1.0
        # Ensure both are tz-aware for subtraction
        if added_at.tzinfo is None:
            added_at = added_at.replace(tzinfo=timezone.utc)
        days = (now - added_at).days
        if days <= 7:
            return 1.5
        if days <= 30:
            return 1.0
        return 0.5

    # Paper concepts
    for paper in signals["recent_papers"]:
        paper_score = paper.get("score", 1.0) or 1.0
        recency = _recency_multiplier(paper.get("added_at"))
        weight = 2.0 if paper_score >= 1.5 else 1.0
        for concept in (paper.get("concepts") or []):
            if concept:
                scores[concept] += weight * recency

    # Insight concepts (+3.0)
    for insight in signals["insights"]:
        for concept in (insight.get("insight_concepts") or []):
            if concept:
                scores[concept] += 3.0

    # Followed concepts (+2.5)
    followed_names = set()
    for entry in signals["followed_concepts"]:
        name = entry.get("concept")
        if name:
            followed_names.add(name)
            scores[name] += 2.5

    # RELATED_TO neighbors of top concepts
    max_weight = 1
    for entry in signals["followed_concepts"]:
        for n in (entry.get("neighbors") or []):
            w = n.get("weight") or 0
            if w > max_weight:
                max_weight = w

    for entry in signals["followed_concepts"]:
        for n in (entry.get("neighbors") or []):
            name = n.get("name")
            w = n.get("weight") or 0
            if name and w >= 3:
                scores[name] += 0.5 * (w / max_weight)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:12]


def _build_signal_context(
    signals: dict, ranked_concepts: list[tuple[str, float]],
) -> str:
    """Assemble structured text sections from graph signals."""
    parts = []

    # Section 1: Top concepts
    if ranked_concepts:
        lines = ["SECTION 1: TOP RESEARCH CONCEPTS (ranked)"]
        for i, (name, score) in enumerate(ranked_concepts, 1):
            lines.append(f"{i}. {name} (score: {score:.1f})")
        parts.append("\n".join(lines))

    # Section 2: Recent papers (up to 10, sorted by score * recency)
    now = datetime.now(timezone.utc)
    papers = signals["recent_papers"]

    def _paper_sort_key(p):
        score = p.get("score", 1.0) or 1.0
        added_at = p.get("added_at")
        days = 30
        if added_at:
            try:
                if isinstance(added_at, str):
                    dt = datetime.fromisoformat(added_at)
                elif isinstance(added_at, datetime):
                    dt = added_at
                else:
                    dt = datetime.fromisoformat(added_at.iso_format())
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                days = max((now - dt).days, 1)
            except Exception:
                pass
        return score / days

    papers_sorted = sorted(papers, key=_paper_sort_key, reverse=True)[:10]
    if papers_sorted:
        lines = ["SECTION 2: RECENT PAPERS (up to 10, sorted by score x recency)"]
        for p in papers_sorted:
            abstract = (p.get("abstract") or "")[:150]
            year = p.get("year") or "?"
            score = p.get("score", 1.0) or 1.0
            lines.append(f'- "{p.get("title", "Untitled")}" (score: {score}, {year}): {abstract}')
        parts.append("\n".join(lines))

    # Section 3: User insights (up to 15)
    insights = signals["insights"][:15]
    if insights:
        lines = ["SECTION 3: USER INSIGHTS (up to 15, most recent first)"]
        for ins in insights:
            sentiment = ins.get("sentiment") or "neutral"
            lines.append(
                f'- About "{ins.get("paper_title", "?")}": "{ins.get("insight_text", "")}" ({sentiment})'
            )
        parts.append("\n".join(lines))

    # Section 4: Explicitly followed topics
    followed = set()
    for entry in signals["followed_concepts"]:
        name = entry.get("concept")
        if name:
            followed.add(name)
    if followed:
        parts.append("SECTION 4: EXPLICITLY FOLLOWED TOPICS\n" + ", ".join(sorted(followed)))

    # Section 5: Foundational cited works
    citations = signals["citation_neighborhood"]
    if citations:
        lines = ["SECTION 5: FOUNDATIONAL CITED WORKS (cited by N user papers)"]
        for c in citations:
            gc = c.get("global_citations") or 0
            lines.append(
                f'- "{c.get("title", "?")}" (cited by {c.get("citing_count", 0)} papers, global citations: {gc})'
            )
        parts.append("\n".join(lines))

    # Section 6: Concept neighbors
    neighbor_entries = []
    for entry in signals["followed_concepts"]:
        concept = entry.get("concept")
        neighbors = entry.get("neighbors") or []
        neighbor_names = [n["name"] for n in neighbors if n.get("name")]
        if concept and neighbor_names:
            neighbor_entries.append(f"{concept} (related to: {', '.join(neighbor_names[:5])})")
    if neighbor_entries:
        parts.append("SECTION 6: CONCEPT NEIGHBORS (adjacent interests)\n" + "; ".join(neighbor_entries[:15]))

    # Section 7: Nearby papers
    nearby = signals["concept_neighbor_papers"]
    if nearby:
        lines = ["SECTION 7: NEARBY PAPERS (sharing concepts, not added by user)"]
        for n in nearby:
            shared = ", ".join((n.get("shared") or [])[:5])
            lines.append(
                f'- "{n.get("title", "?")}" (shares {n.get("shared_count", 0)} concepts: {shared})'
            )
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


_SYNTHESIS_PROMPT = """\
Based on the signals below, write a concise research interest profile (150-250 words). \
This text will be embedded as a vector and compared via cosine similarity against \
paper abstracts, so use precise technical terms that would appear in relevant abstracts.

Rules:
- Write cohesive sentences that connect related topics naturally
- Use concrete technical vocabulary — the kind found in abstracts, not commentary
- State what the researcher works on, not how they feel about it
- Group related interests into 3-5 short sentences, each covering a thread
- No paper titles, author names, or filler ("notably", "particularly", "demonstrates deep conviction")
- No meta-commentary about the researcher's engagement or trajectory

Good: "Their work focuses on replacing recurrence with self-attention for sequence \
transduction, using multi-head scaled dot-product attention and positional encoding \
in encoder-decoder transformers."

Bad: "This researcher demonstrates deep conviction in the centrality of self-attention, \
regarding it as crucial to enabling effective sequence-to-sequence learning."

SIGNALS:
"""


async def _synthesize_blurb(context: str) -> str:
    """Call Haiku to synthesize the interest blurb, with fallback."""
    try:
        print("[interest_profile] Calling Haiku for synthesis...")
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": _SYNTHESIS_PROMPT + context}],
        )
        print("[interest_profile] Haiku synthesis complete")
        return response.content[0].text.strip()
    except Exception as e:
        print(f"[interest_profile] Haiku synthesis failed: {e}, using fallback")
        return _fallback_blurb(context)


def _fallback_blurb(context: str) -> str:
    """Extract concepts and titles from context as a simple fallback."""
    concepts = []
    titles = []
    for line in context.split("\n"):
        line = line.strip()
        # Extract ranked concepts
        if line and line[0].isdigit() and ". " in line and "(score:" in line:
            name = line.split(". ", 1)[1].split(" (score:")[0]
            concepts.append(name)
        # Extract paper titles
        if line.startswith('- "') and "(score:" in line:
            title = line.split('"')[1]
            titles.append(title)

    parts = []
    if concepts:
        parts.append("Research interests: " + ", ".join(concepts[:15]))
    if titles:
        parts.append("Recent papers: " + "; ".join(titles[:5]))
    return ". ".join(parts) + "." if parts else ""
