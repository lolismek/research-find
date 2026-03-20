"""LLM-powered query expansion for literature search.

Takes a broad topic (e.g. "hyperbaric oxygen therapy") and generates 8-12
diverse sub-queries targeting different study designs, mechanisms, and outcomes
to reduce study design skew in search results.
"""

from __future__ import annotations

import os
import logging
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a biomedical literature search specialist. Given a research topic, "
    "generate diverse PubMed search queries that cover different aspects of the topic.\n\n"
    "Each query should target a DIFFERENT angle:\n"
    "- Randomized controlled trials (RCTs) and clinical trials\n"
    "- Molecular mechanisms (specific pathways, genes, signaling cascades)\n"
    "- Animal model studies (in vivo)\n"
    "- In vitro / cell culture studies\n"
    "- Biomarker and diagnostic studies\n"
    "- Safety, adverse effects, and contraindications\n"
    "- Meta-analyses and systematic reviews\n"
    "- Dose-response and pharmacokinetic studies\n"
    "- Comparative effectiveness studies\n"
    "- Specific disease applications and outcomes\n\n"
    "Guidelines:\n"
    "- Use PubMed-friendly terms (MeSH-style vocabulary)\n"
    "- Each query should be 3-8 words, specific enough to return relevant papers\n"
    "- Avoid duplicating the same angle across queries\n"
    "- Include both broad mechanism queries and specific pathway/gene queries\n"
    "- Do NOT include boolean operators (AND/OR) — keep queries as simple phrases"
)


class ExpandedQueries(BaseModel):
    """Pydantic model for structured LLM output."""

    queries: list[str] = Field(
        ...,
        min_length=4,
        max_length=16,
        description="Diverse PubMed search queries covering different study designs and angles",
    )


class QueryExpander:
    """Generates diverse sub-queries from a broad topic using Anthropic Claude."""

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic()

    async def expand(self, topic: str, *, max_queries: int = 12) -> list[str]:
        """Expand a topic into diverse sub-queries.

        Args:
            topic: Broad research topic (e.g. "hyperbaric oxygen therapy").
            max_queries: Maximum number of queries to generate (8-12 typical).

        Returns:
            List of PubMed-friendly search queries.
        """
        user_prompt = (
            f"Generate {max_queries} diverse search queries for the topic: "
            f'"{topic}"\n\n'
            f"Cover different study designs, mechanisms, and clinical outcomes.\n\n"
            f"Return your answer as JSON matching this schema:\n"
            f'{{"queries": ["query1", "query2", ...]}}'
        )

        response = await self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        import json
        text = response.content[0].text
        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            result = ExpandedQueries(**parsed)
        else:
            raise ValueError(f"Could not parse JSON from response: {text[:200]}")

        queries = result.queries[:max_queries]
        logger.info(f"Query expansion: '{topic}' -> {len(queries)} sub-queries")
        for i, q in enumerate(queries):
            logger.debug(f"  [{i+1}] {q}")

        return queries
