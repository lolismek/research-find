"""LLM-powered query expansion for literature search.

Takes a broad topic (e.g. "hyperbaric oxygen therapy") and generates 8-12
diverse sub-queries targeting different study designs, mechanisms, and outcomes
to reduce study design skew in search results.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from etl.literature.extraction.llm_client import LLMClient

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
    """Generates diverse sub-queries from a broad topic using LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

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
            f"Cover different study designs, mechanisms, and clinical outcomes."
        )

        result = await self._llm_client.parse(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=ExpandedQueries,
            max_tokens=1000,
        )

        queries = result.queries[:max_queries]
        logger.info(f"Query expansion: '{topic}' -> {len(queries)} sub-queries")
        for i, q in enumerate(queries):
            logger.debug(f"  [{i+1}] {q}")

        return queries
