"""Unified Paper model normalizing across all sources."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Author(BaseModel):
    name: str
    author_id: Optional[str] = None


class Concept(BaseModel):
    name: str
    embedding: Optional[list[float]] = None


class User(BaseModel):
    phone_number: str
    name: Optional[str] = None
    created_at: Optional[datetime] = None


class Paper(BaseModel):
    paper_id: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    title: str
    abstract: Optional[str] = None
    authors: list[Author] = []
    year: Optional[int] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: Optional[bool] = None
    citation_count: int = 0
    fields_of_study: Optional[list[str]] = None
    source: str  # "semantic_scholar", "arxiv", "pubmed", etc.

    # GROBID enrichment
    grobid_abstract: Optional[str] = None
    sections: Optional[list[dict]] = None
    references: Optional[list[dict]] = None
    keywords: Optional[list[str]] = None

    # S2 references (populated during background enrichment)
    s2_references: Optional[list[dict]] = None

    # Embedding (OpenAI text-embedding-3-small, 1536-dim)
    embedding: Optional[list[float]] = None

    added_at: Optional[datetime] = None

    @classmethod
    def from_s2_dict(cls, data: dict) -> Paper:
        """Create Paper from a Semantic Scholar API response dict."""
        ext = data.get("externalIds") or {}
        authors_raw = data.get("authors") or []
        authors = []
        for a in authors_raw:
            if isinstance(a, dict):
                authors.append(Author(
                    name=a.get("name", "Unknown"),
                    author_id=a.get("authorId"),
                ))
            elif isinstance(a, str):
                authors.append(Author(name=a))

        pdf_url = None
        oa_pdf = data.get("openAccessPdf")
        if isinstance(oa_pdf, dict):
            pdf_url = oa_pdf.get("url")

        fields = data.get("fieldsOfStudy") or []
        s2_fields = data.get("s2FieldsOfStudy") or []
        all_fields = fields or [f.get("category", "") for f in s2_fields if isinstance(f, dict)]

        return cls(
            paper_id=data.get("paperId"),
            doi=ext.get("DOI"),
            arxiv_id=ext.get("ArXiv"),
            title=data.get("title", "Untitled"),
            abstract=data.get("abstract"),
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue") or (data.get("journal") or {}).get("name"),
            url=data.get("url"),
            pdf_url=pdf_url,
            is_open_access=data.get("isOpenAccess"),
            citation_count=data.get("citationCount") or 0,
            fields_of_study=all_fields or None,
            source="semantic_scholar",
        )

    @classmethod
    def from_arxiv_entry(cls, entry: dict) -> Paper:
        """Create Paper from a parsed arXiv RSS/API entry."""
        authors = []
        for a in (entry.get("authors") or []):
            if isinstance(a, dict):
                authors.append(Author(name=a.get("name", "Unknown")))
            elif isinstance(a, str):
                authors.append(Author(name=a))

        year = None
        published = entry.get("published") or entry.get("updated")
        if published:
            try:
                year = int(published[:4])
            except (ValueError, IndexError):
                pass

        arxiv_id = entry.get("arxiv_id") or entry.get("id", "")
        # Clean up arxiv_id from URL form
        if "arxiv.org" in arxiv_id:
            arxiv_id = arxiv_id.rstrip("/").split("/")[-1]
        # Strip version suffix for canonical ID
        if arxiv_id and "v" in arxiv_id:
            base = arxiv_id.rsplit("v", 1)
            if base[1].isdigit():
                arxiv_id = base[0]

        return cls(
            arxiv_id=arxiv_id,
            title=entry.get("title", "Untitled"),
            abstract=entry.get("summary") or entry.get("abstract"),
            authors=authors,
            year=year,
            url=entry.get("link") or entry.get("url"),
            pdf_url=entry.get("pdf_url"),
            source="arxiv",
        )

    @classmethod
    def from_grobid_tei(cls, tei_data: dict, base_paper: Paper | None = None) -> Paper:
        """Create or enrich a Paper from GROBID TEI XML parsed data."""
        if base_paper:
            base_paper.grobid_abstract = tei_data.get("abstract") or base_paper.grobid_abstract
            base_paper.sections = tei_data.get("sections") or base_paper.sections
            base_paper.references = tei_data.get("references") or base_paper.references
            base_paper.keywords = tei_data.get("keywords") or base_paper.keywords
            if not base_paper.abstract and tei_data.get("abstract"):
                base_paper.abstract = tei_data["abstract"]
            return base_paper

        authors = []
        for a in (tei_data.get("authors") or []):
            if isinstance(a, dict):
                authors.append(Author(name=a.get("name", "Unknown")))
            elif isinstance(a, str):
                authors.append(Author(name=a))

        return cls(
            title=tei_data.get("title", "Untitled"),
            abstract=tei_data.get("abstract"),
            grobid_abstract=tei_data.get("abstract"),
            authors=authors,
            sections=tei_data.get("sections"),
            references=tei_data.get("references"),
            keywords=tei_data.get("keywords"),
            source="grobid",
        )

    def display_str(self) -> str:
        """Format paper for display in chat."""
        parts = [f"**{self.title}**"]
        if self.authors:
            names = ", ".join(a.name for a in self.authors[:3])
            if len(self.authors) > 3:
                names += f" et al."
            parts.append(f"  Authors: {names}")
        if self.year:
            parts.append(f"  Year: {self.year}")
        if self.venue:
            parts.append(f"  Venue: {self.venue}")
        if self.doi:
            parts.append(f"  DOI: {self.doi}")
        if self.arxiv_id:
            parts.append(f"  arXiv: {self.arxiv_id}")
        if self.citation_count:
            parts.append(f"  Citations: {self.citation_count}")
        if self.url:
            parts.append(f"  URL: {self.url}")
        return "\n".join(parts)
