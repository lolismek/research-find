"""Structural parsing of scientific papers.

Three paths tried in order:
1. PMC/JATS XML — best quality, deterministic, free
2. PDF via Grobid — fallback for paywalled/non-PMC papers
3. Abstract-only — last resort when no full text available
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import aiohttp
from lxml import etree

from src.schemas.paper_schema import (
    Paper,
    PaperFigure,
    PaperSection,
    PaperTable,
    ParseSource,
    SectionType,
)

logger = logging.getLogger(__name__)

# JATS section title -> SectionType mapping
_JATS_SECTION_MAP: dict[str, SectionType] = {
    "intro": SectionType.INTRODUCTION,
    "introduction": SectionType.INTRODUCTION,
    "background": SectionType.BACKGROUND,
    "methods": SectionType.METHODS,
    "materials and methods": SectionType.METHODS,
    "materials & methods": SectionType.METHODS,
    "experimental": SectionType.METHODS,
    "experimental procedures": SectionType.METHODS,
    "study design": SectionType.METHODS,
    "patients and methods": SectionType.METHODS,
    "results": SectionType.RESULTS,
    "results and discussion": SectionType.RESULTS,
    "discussion": SectionType.DISCUSSION,
    "conclusion": SectionType.CONCLUSION,
    "conclusions": SectionType.CONCLUSION,
    "supplementary": SectionType.SUPPLEMENTARY,
    "supplementary material": SectionType.SUPPLEMENTARY,
    "supplementary materials": SectionType.SUPPLEMENTARY,
    "references": SectionType.REFERENCES,
}

# Grobid TEI section title patterns
_GROBID_SECTION_MAP = _JATS_SECTION_MAP.copy()


def _classify_section(title: str) -> SectionType:
    """Map a section heading to a SectionType."""
    normalized = title.strip().lower()
    # Strip leading numbering (e.g. "1. Introduction", "2.1 Methods")
    normalized = re.sub(r"^\d+(\.\d+)*\.?\s*", "", normalized)

    # Exact match first
    if normalized in _JATS_SECTION_MAP:
        return _JATS_SECTION_MAP[normalized]

    # Prefix match (e.g. "conclusions and future" → CONCLUSION)
    for key, section_type in _JATS_SECTION_MAP.items():
        if normalized.startswith(key):
            return section_type

    # Keyword match for common patterns
    if "method" in normalized or "material" in normalized or "experimental" in normalized:
        return SectionType.METHODS
    if "result" in normalized:
        return SectionType.RESULTS
    if "discuss" in normalized:
        return SectionType.DISCUSSION
    if "conclu" in normalized:
        return SectionType.CONCLUSION
    if "intro" in normalized or "background" in normalized:
        return SectionType.INTRODUCTION
    if "supplement" in normalized:
        return SectionType.SUPPLEMENTARY

    return SectionType.OTHER


def _elem_text(elem: etree._Element) -> str:
    """Recursively extract all text from an lxml element."""
    return "".join(elem.itertext()).strip()


# ── PMC / JATS XML parsing ──────────────────────────────────────────────────


class JATSParser:
    """Parse PMC JATS XML into structured ``PaperSection`` objects."""

    def parse(self, xml_content: str | bytes) -> list[PaperSection]:
        if isinstance(xml_content, str):
            xml_content = xml_content.encode("utf-8")
        root = etree.fromstring(xml_content)
        sections: list[PaperSection] = []

        # Abstract
        for abstract_el in root.iter("abstract"):
            sections.append(self._parse_abstract(abstract_el))

        # Body sections
        for body_el in root.iter("body"):
            for sec_el in body_el.findall("sec"):
                sections.append(self._parse_sec(sec_el))

        # References
        for ref_list in root.iter("ref-list"):
            sections.append(PaperSection(
                section_type=SectionType.REFERENCES,
                heading="References",
                text=_elem_text(ref_list),
            ))

        return sections

    def _parse_abstract(self, el: etree._Element) -> PaperSection:
        """Parse a JATS <abstract> element (may have <sec> children for structured abstracts)."""
        subsections: list[PaperSection] = []
        structured_children = el.findall("sec")

        if structured_children:
            for sec in structured_children:
                title_el = sec.find("title")
                heading = _elem_text(title_el) if title_el is not None else ""
                subsections.append(PaperSection(
                    section_type=_classify_section(heading) if heading else SectionType.OTHER,
                    heading=heading,
                    text=_elem_text(sec),
                ))
            combined = "\n\n".join(s.text for s in subsections if s.text)
        else:
            combined = _elem_text(el)

        return PaperSection(
            section_type=SectionType.ABSTRACT,
            heading="Abstract",
            text=combined,
            subsections=subsections,
        )

    def _parse_sec(self, el: etree._Element) -> PaperSection:
        """Parse a JATS <sec> element recursively."""
        title_el = el.find("title")
        heading = _elem_text(title_el) if title_el is not None else ""

        # Try sec-type attribute first (authoritative), then heading text
        sec_type_attr = el.get("sec-type", "")
        if sec_type_attr and sec_type_attr in _JATS_SECTION_MAP:
            section_type = _JATS_SECTION_MAP[sec_type_attr]
        elif heading:
            section_type = _classify_section(heading)
        else:
            section_type = SectionType.OTHER

        # Collect paragraph text (skip nested <sec>)
        paragraphs: list[str] = []
        for p in el.findall("p"):
            txt = _elem_text(p)
            if txt:
                paragraphs.append(txt)

        # Tables
        tables: list[PaperTable] = []
        for table_wrap in el.findall(".//table-wrap"):
            tables.append(self._parse_table(table_wrap))

        # Figures
        figures: list[PaperFigure] = []
        for fig_el in el.findall(".//fig"):
            figures.append(self._parse_figure(fig_el))

        # Subsections
        subsections: list[PaperSection] = []
        for child_sec in el.findall("sec"):
            subsections.append(self._parse_sec(child_sec))

        return PaperSection(
            section_type=section_type,
            heading=heading,
            text="\n\n".join(paragraphs),
            subsections=subsections,
            tables=tables,
            figures=figures,
        )

    def _parse_table(self, el: etree._Element) -> PaperTable:
        label_el = el.find("label")
        caption_el = el.find("caption")
        headers: list[str] = []
        rows: list[list[str]] = []

        thead = el.find(".//thead")
        if thead is not None:
            for th_row in thead.findall("tr"):
                headers = [_elem_text(th) for th in th_row.findall("th")]

        tbody = el.find(".//tbody")
        if tbody is not None:
            for tr in tbody.findall("tr"):
                rows.append([_elem_text(td) for td in tr.findall("td")])

        return PaperTable(
            label=_elem_text(label_el) if label_el is not None else None,
            caption=_elem_text(caption_el) if caption_el is not None else None,
            headers=headers,
            rows=rows,
        )

    def _parse_figure(self, el: etree._Element) -> PaperFigure:
        label_el = el.find("label")
        caption_el = el.find("caption")
        return PaperFigure(
            label=_elem_text(label_el) if label_el is not None else None,
            caption=_elem_text(caption_el) if caption_el is not None else None,
        )


# ── Grobid TEI XML parsing ─────────────────────────────────────────────────

_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


class GrobidParser:
    """Parse Grobid TEI XML output into structured ``PaperSection`` objects."""

    def parse(self, tei_xml: str | bytes) -> list[PaperSection]:
        if isinstance(tei_xml, str):
            tei_xml = tei_xml.encode("utf-8")
        root = etree.fromstring(tei_xml)
        sections: list[PaperSection] = []

        # Abstract
        abstract_el = root.find(".//tei:profileDesc/tei:abstract", _TEI_NS)
        if abstract_el is not None:
            sections.append(PaperSection(
                section_type=SectionType.ABSTRACT,
                heading="Abstract",
                text=_elem_text(abstract_el),
            ))

        # Body divisions
        body_el = root.find(".//tei:text/tei:body", _TEI_NS)
        if body_el is not None:
            for div in body_el.findall("tei:div", _TEI_NS):
                sections.append(self._parse_div(div))

        return sections

    def _parse_div(self, div: etree._Element) -> PaperSection:
        head_el = div.find("tei:head", _TEI_NS)
        heading = _elem_text(head_el) if head_el is not None else ""
        section_type = _classify_section(heading) if heading else SectionType.OTHER

        paragraphs: list[str] = []
        for p in div.findall("tei:p", _TEI_NS):
            txt = _elem_text(p)
            if txt:
                paragraphs.append(txt)

        return PaperSection(
            section_type=section_type,
            heading=heading,
            text="\n\n".join(paragraphs),
        )


# ── Grobid client ──────────────────────────────────────────────────────────


class GrobidClient:
    """Send a PDF to a Grobid server and return TEI XML."""

    def __init__(self, base_url: str = "http://localhost:8070"):
        self.base_url = base_url.rstrip("/")

    async def parse_pdf(self, pdf_path: Path) -> str:
        """Send PDF to Grobid ``processFulltextDocument`` endpoint."""
        url = f"{self.base_url}/api/processFulltextDocument"
        with open(pdf_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("input", f, filename=pdf_path.name, content_type="application/pdf")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    resp.raise_for_status()
                    return await resp.text()


# ── PMC XML fetcher ─────────────────────────────────────────────────────────


async def fetch_pmc_xml(pmc_id: str) -> Optional[str]:
    """Download JATS XML from Europe PMC for a given PMC ID."""
    # Europe PMC API requires the PMC prefix (e.g. "PMC8844085")
    pmc_id_clean = pmc_id if pmc_id.startswith("PMC") else f"PMC{pmc_id}"
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc_id_clean}/fullTextXML"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return await resp.text()
                logger.warning(f"PMC XML fetch failed for {pmc_id}: HTTP {resp.status}")
                return None
    except Exception as e:
        logger.warning(f"PMC XML fetch error for {pmc_id}: {e}")
        return None


# ── Unified parser ──────────────────────────────────────────────────────────


class PaperParser:
    """Try PMC XML, Grobid PDF, or abstract-only to parse a paper's structure."""

    def __init__(self, grobid_url: str = "http://localhost:8070"):
        self.jats_parser = JATSParser()
        self.grobid_parser = GrobidParser()
        self.grobid_client = GrobidClient(grobid_url)

    async def parse(self, paper: Paper, pdf_path: Optional[Path] = None) -> Paper:
        """Parse paper full text and populate ``paper.sections``.

        Tries in order:
        1. PMC XML (if pmc_id available)
        2. Grobid (if pdf_path provided)
        3. Abstract-only fallback
        """
        # Path 1: PMC XML
        if paper.pmc_id:
            xml = await fetch_pmc_xml(paper.pmc_id)
            if xml:
                try:
                    paper.sections = self.jats_parser.parse(xml)
                    paper.parse_source = ParseSource.PMC_XML
                    logger.info(f"Parsed {paper.pmc_id} via PMC XML ({len(paper.sections)} sections)")
                    return paper
                except Exception as e:
                    logger.warning(f"PMC XML parse failed for {paper.pmc_id}: {e}")

        # Path 2: Grobid
        if pdf_path and pdf_path.exists():
            try:
                tei_xml = await self.grobid_client.parse_pdf(pdf_path)
                paper.sections = self.grobid_parser.parse(tei_xml)
                paper.parse_source = ParseSource.GROBID
                logger.info(f"Parsed {paper.doi or paper.title[:40]} via Grobid ({len(paper.sections)} sections)")
                return paper
            except Exception as e:
                logger.warning(f"Grobid parse failed for {pdf_path}: {e}")

        # Path 3: Abstract-only fallback
        sections: list[PaperSection] = []
        if paper.structured_abstract:
            subsections = []
            for heading, text in paper.structured_abstract.items():
                subsections.append(PaperSection(
                    section_type=_classify_section(heading),
                    heading=heading,
                    text=text,
                ))
            sections.append(PaperSection(
                section_type=SectionType.ABSTRACT,
                heading="Abstract",
                text=paper.abstract or "",
                subsections=subsections,
            ))
        elif paper.abstract:
            sections.append(PaperSection(
                section_type=SectionType.ABSTRACT,
                heading="Abstract",
                text=paper.abstract,
            ))

        paper.sections = sections
        paper.parse_source = ParseSource.ABSTRACT_ONLY
        logger.info(f"Abstract-only fallback for {paper.doi or paper.title[:40]}")
        return paper
