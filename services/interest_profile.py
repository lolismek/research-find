"""Generate a text blurb summarizing a user's research interests from graph signals."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import anthropic


async def generate_user_interest_blurb(phone_number: str) -> str:
    """Generate a dense research interest blurb for embedding-based paper ranking."""
    from services.neo4j_store import get_user_interest_signals

    signals = await get_user_interest_signals(phone_number)

    if not signals["recent_papers"]:
        return ""

    ranked_concepts = _rank_concepts(signals)
    context = _build_signal_context(signals, ranked_concepts)
    return await _synthesize_blurb(context)


def _rank_concepts(signals: dict) -> list[tuple[str, float]]:
    """Aggregate concept scores across all signal types with recency weighting."""
    scores: dict[str, float] = defaultdict(float)
    now = datetime.utcnow()

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
            w = n.get("weight", 0)
            if w > max_weight:
                max_weight = w

    for entry in signals["followed_concepts"]:
        for n in (entry.get("neighbors") or []):
            name = n.get("name")
            w = n.get("weight", 0)
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
    now = datetime.utcnow()
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
You are writing a research interest profile for an academic researcher. Based on the \
signals below (their recent papers, insights, concepts, and citation patterns), write \
a dense, information-rich paragraph (200-400 words) describing their current research \
interests, methodological preferences, and intellectual focus areas.

Requirements:
- Write in third person ("This researcher...")
- Pack densely with specific technical terms, methodologies, and research areas
- Emphasize areas where the user has written insights (active intellectual engagement)
- Weight recent activity more heavily than older
- Mention intersections between different research threads if apparent
- Include methodological preferences if they emerge from the papers
- Note any emerging or shifting interests from the most recent papers
- Do NOT include paper titles or author names
- Do NOT include filler phrases or meta-commentary
- The output will be embedded as a vector and compared against paper abstracts, \
so optimize for semantic overlap with relevant academic abstracts

SIGNALS:
"""


async def _synthesize_blurb(context: str) -> str:
    """Call Haiku to synthesize the interest blurb, with fallback."""
    try:
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": _SYNTHESIS_PROMPT + context}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"[interest_profile] Haiku synthesis failed: {e}")
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
